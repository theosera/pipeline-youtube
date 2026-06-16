"""CLI entry point for pipeline-youtube.

Orchestrates the full pipeline: fetch playlist metadata → process each
video through stages 01-04 → stage 05 synthesis if ≥3 succeed.

Concurrency model: `--concurrency N` runs up to N videos in parallel
via `asyncio.to_thread` + `asyncio.Semaphore`. Default is 1 (sequential).
Whisper (tier 3 fallback) has its own file-based global lock so it never
runs more than one instance even under concurrency > 1.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from .checkpoint import get_completed_video_ids
from .cli_config import (
    _SYNTHESIS_PROFILE_CHOICES,
    DEFAULT_CONFIG_PATH,
    _load_config,
)
from .config import VaultRootError, set_dry_run, set_vault_root
from .genres import CODE_BEARING_GENRES, classify_playlist_genre
from .glossary import Glossary, correction_glossary, known_pairs, load_sheet
from .parallel import orchestrate_sub_agents, parse_video_range, strip_cli_option
from .playlist import VideoMeta, fetch_metadata, validate_youtube_url
from .proper_noun_sheet import (
    _promote_corrections_to_glossary,
    _proper_noun_sheet_path,
    _update_proper_noun_sheet,
)
from .providers.claude_cli import ClaudeBinaryError, get_resolved_claude_binary
from .resume import (
    _collect_existing_learning_bodies,
    _filter_to_reviewed,
    _load_existing_04_body,
    _parse_run_timestamp,
)
from .run_result import VideoRunResult, _print_cost_breakdown
from .sanitize import configure_alert_sink
from .stages.capture import ASSETS_REL_PATH, sweep_stale_tmp
from .stages.capture_backend import DockerBackendNotReady, DockerCaptureBackend
from .stages.synthesis import MIN_PLAYLIST_SIZE, log_synthesis_preflight, run_stage_synthesis
from .synthesis.agents import compute_synthesis_timeouts
from .transcript.whisper_fallback import configure_whisper, describe_whisper
from .video_processing import _process_video, _run_videos_concurrent


@click.command()
@click.argument("url", required=False)
@click.option("--dry-run", is_flag=True, help="Do not write to vault; print to stdout only.")
@click.option(
    "--local-media",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Build hands-on from a LOCAL folder of the playlist's video files "
        "(fully offline — no YouTube access). Stage 01 transcribes each file "
        "with Whisper (run with --extra whisper); Stage 03 captures from it. "
        "URL becomes optional. Name files with the 11-char video_id, e.g. "
        "yt-dlp '%(id)s.%(ext)s' or 'Title [%(id)s].mp4'. See docs/local-media.md."
    ),
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 5),
    default=1,
    help="Videos in parallel (1-5, default 1).",
)
@click.option(
    "--sub-agents",
    type=click.IntRange(1, 8),
    default=1,
    show_default=True,
    help=(
        "Run stages 01-04 as N parallel sub-agent processes, each owning a "
        "contiguous slice of the playlist (remainder on the last shard); Stage "
        "05 runs once afterwards. Default 1 = original single-process flow. "
        "Opt in with e.g. --sub-agents 3. See docs/sub-agents.md."
    ),
)
@click.option(
    "--video-range",
    default=None,
    hidden=True,
    help="Internal: 0-based half-open 'start:end' slice processed by one sub-agent shard.",
)
@click.option(
    "--run-timestamp",
    default=None,
    hidden=True,
    help="Internal: ISO run_time shared across sub-agent shards so they write one playlist folder.",
)
@click.option(
    "--code-bearing/--no-code-bearing",
    "code_bearing_override",
    default=None,
    hidden=True,
    help="Internal: parent-pinned genre decision passed to sub-agent shards (skips the router).",
)
@click.option("--skip-synthesis", is_flag=True, help="Skip stage 05 after 01-04 finish.")
@click.option(
    "--synthesis-only",
    is_flag=True,
    help="Skip stages 01-04 and re-run only stage 05 against existing 04 md files for today's date.",
)
@click.option(
    "--force-video",
    multiple=True,
    help="Force reprocess specific video IDs even if checkpoint shows complete. Repeatable.",
)
@click.option(
    "--capture-format",
    type=click.Choice(["auto", "webp", "gif"]),
    default="auto",
    help="Animated capture output format. Default auto picks WebP when possible.",
)
@click.option(
    "--model",
    default="sonnet",
    help="Claude model alias for stages 02/04/05 (sonnet, haiku, opus, or full ID).",
)
@click.option(
    "--min-playlist-size",
    type=click.IntRange(1, 100),
    default=MIN_PLAYLIST_SIZE,
    show_default=True,
    help="Skip stage 05 when fewer than N videos succeed (default 3).",
)
@click.option(
    "--max-chapters",
    type=click.IntRange(1, 30),
    default=None,
    help="Cap β's chapter count via prompt constraint. Unset = let β decide.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config.json path.",
)
@click.option(
    "--stop-after-capture",
    is_flag=True,
    help=(
        "Phase 1: run stages 01-03 and stop. User reviews Stage 02 in Obsidian, "
        "flips frontmatter `reviewed: true`, then runs again with --resume-reviewed."
    ),
)
@click.option(
    "--resume-reviewed",
    is_flag=True,
    help=(
        "Phase 3: skip stages 01-03; only process videos whose 02_Summary.md "
        "frontmatter has `reviewed: true`. Runs stages 04 and 05 on the filtered set."
    ),
)
@click.option(
    "--capture-backend",
    type=click.Choice(["host", "docker"]),
    default=None,
    help=(
        "Stage 03 execution backend. 'host' (default) runs yt-dlp/ffmpeg directly. "
        "'docker' runs them inside the hardened pipeline-youtube-capture image — "
        "build it first via `docker build -f docker/Dockerfile.capture -t "
        "pipeline-youtube-capture:latest .`. Overrides config.json."
    ),
)
@click.option(
    "--synthesis-timeout",
    type=click.IntRange(60),
    default=None,
    help=(
        "Per-agent timeout for Stage 05 in seconds. "
        "Default: auto (300 + 60 × video_count). Overrides config.json."
    ),
)
@click.option(
    "--synthesis-profile",
    type=click.Choice(list(_SYNTHESIS_PROFILE_CHOICES)),
    default=None,
    help=(
        "Agent Teams composition for Stage 05. 'auto' (default) picks "
        "'standard' (≤15 videos), 'parallel' (16-30), or 'parallel+full' "
        "(>30). 'full' adds a Reviewer pass. Overrides config.json."
    ),
)
def cli(
    url: str | None,
    dry_run: bool,
    concurrency: int,
    sub_agents: int,
    video_range: str | None,
    run_timestamp: str | None,
    code_bearing_override: bool | None,
    skip_synthesis: bool,
    synthesis_only: bool,
    force_video: tuple[str, ...],
    capture_format: str,
    model: str,
    min_playlist_size: int,
    max_chapters: int | None,
    config_path: Path | None,
    stop_after_capture: bool,
    resume_reviewed: bool,
    capture_backend: str | None,
    synthesis_timeout: int | None,
    synthesis_profile: str | None,
    local_media: Path | None,
) -> None:
    """Process a YouTube playlist or single-video URL end-to-end."""
    if not url and not local_media:
        click.echo("Usage: pipeline-youtube <playlist-or-video-url> [options]")
        click.echo("   or: pipeline-youtube --local-media <dir>   (fully offline)")
        sys.exit(2)

    if url:
        try:
            validate_youtube_url(url)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    # Mutually-exclusive phase flags
    phase_flags = sum(bool(x) for x in (stop_after_capture, resume_reviewed, synthesis_only))
    if phase_flags > 1:
        raise click.UsageError(
            "--stop-after-capture, --resume-reviewed, and --synthesis-only are mutually exclusive."
        )

    # Sub-agent orchestration owns the full 01-04 → 05 flow; it is incompatible
    # with the alternate phase flags and with --dry-run (workers write the 04
    # files that the post-merge Stage 05 reads back).
    if sub_agents > 1 and (dry_run or synthesis_only or resume_reviewed or stop_after_capture):
        raise click.UsageError(
            "--sub-agents > 1 cannot be combined with --dry-run or the "
            "--synthesis-only / --resume-reviewed / --stop-after-capture phase flags."
        )

    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg = _load_config(cfg_path, fallback_model=model)
    try:
        set_vault_root(cfg.vault_root, strict=True)
    except VaultRootError as exc:
        raise click.UsageError(str(exc)) from exc
    set_dry_run(dry_run)
    configure_whisper(backend=cfg.whisper_backend, model=cfg.whisper_model)
    vault_root = cfg.vault_root
    models = cfg.models
    filler_words = cfg.filler_words

    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_alert_sink(logs_dir / "sanitize_alerts.jsonl")

    swept = sweep_stale_tmp(project_root / "tmp")
    if swept:
        click.echo(f"swept {swept} stale tmp video file(s)")

    try:
        claude_bin, claude_ver = get_resolved_claude_binary()
        click.echo(f"claude: {claude_bin} ({claude_ver})")
    except ClaudeBinaryError as exc:
        raise click.UsageError(str(exc)) from exc

    # Resolve the Stage 03 capture backend. CLI flag beats config.json; both
    # default to "host". The preflight for Docker mode is deferred until we
    # know Stage 03 will actually run — workflows that skip capture
    # (`--synthesis-only`, `--resume-reviewed`) must not fail just because
    # the docker daemon happens to be unavailable at that moment.
    active_capture_backend: Any = None
    backend_choice = capture_backend or cfg.capture_backend
    # Capture runs in every mode except --synthesis-only (which only re-runs
    # Stage 05 over existing 04 md). In particular --resume-reviewed still calls
    # _process_video()/Stage 03, so it must run the docker preflight and be
    # subject to the local-media guard below.
    will_run_capture = not synthesis_only
    # --local-media files live outside the container's bind mounts (tmp/ + the
    # Vault assets folder), so the docker backend's ffmpeg can't read them.
    # Reject the combination up front instead of failing per-video deep inside
    # Stage 03.
    if local_media and backend_choice == "docker" and will_run_capture:
        raise click.UsageError(
            "--local-media is incompatible with the docker capture backend: the "
            "hardened container only mounts tmp/ and the Vault assets folder, so "
            "your media directory is not visible to ffmpeg. Re-run with the host "
            "backend (--capture-backend host)."
        )
    if backend_choice == "docker":
        assets_dir = vault_root / ASSETS_REL_PATH
        assets_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        active_capture_backend = DockerCaptureBackend(
            tmp_dir=tmp_dir,
            assets_dir=assets_dir,
            image=cfg.capture_docker_image,
        )
        if will_run_capture:
            try:
                active_capture_backend.preflight()
            except DockerBackendNotReady as exc:
                raise click.UsageError(str(exc)) from exc
            click.echo(f"capture_backend: docker ({cfg.capture_docker_image})")
        else:
            click.echo(
                f"capture_backend: docker ({cfg.capture_docker_image}) "
                "[preflight deferred: capture not needed this run]"
            )
    else:
        click.echo("capture_backend: host")

    effective_synthesis_timeout = synthesis_timeout or cfg.synthesis_timeout
    effective_synthesis_profile = synthesis_profile or cfg.synthesis_profile or "auto"

    click.echo(f"vault_root: {vault_root}")
    click.echo(f"dry_run: {dry_run}")
    click.echo(f"model: {model}")
    click.echo(f"whisper: {describe_whisper()}")
    click.echo(f"capture_format: {capture_format}")
    click.echo(f"concurrency: {concurrency}")
    click.echo(f"min_playlist_size: {min_playlist_size}")
    click.echo(f"max_chapters: {max_chapters if max_chapters is not None else 'auto'}")
    click.echo(
        f"synthesis_timeout: {effective_synthesis_timeout}s"
        if effective_synthesis_timeout
        else "synthesis_timeout: auto"
    )
    click.echo(f"synthesis_profile: {effective_synthesis_profile}")

    # --local-media: build the video list from a local folder (no YouTube).
    # media_map (video_id → file path) is threaded into stages 01/03 so they
    # transcribe/capture the local file instead of downloading. Empty otherwise.
    media_map: dict[str, Path] = {}
    if local_media:
        from .local_media import build_local_videos

        videos, media_map = build_local_videos(local_media)
        if not videos:
            click.echo(f"No media files found in {local_media}")
            sys.exit(1)
        click.echo(f"local-media: {len(videos)} file(s) from {local_media}")
    else:
        if url is None:
            raise click.UsageError(
                "A playlist/video URL is required unless --local-media is given."
            )
        click.echo("fetching metadata...")
        videos = fetch_metadata(url)
        if not videos:
            click.echo("No videos found.")
            sys.exit(1)

    playlist_title = videos[0].playlist_title or videos[0].title or "single video"
    click.echo(f"playlist: {playlist_title!r}")
    click.echo(f"videos: {len(videos)}")

    # Stage 00.5: Router. One cheap haiku call decides whether downstream
    # code-bearing features (GitHub URL extraction, concept/practice split)
    # apply. Errors collapse to Genre.OTHER → default behavior. The parent
    # classifies once and pins the result for every sub-agent shard (internal
    # --code-bearing/--no-code-bearing), so a transient router error on one
    # worker can't leave shards disagreeing on code_bearing.
    if code_bearing_override is not None:
        code_bearing = code_bearing_override
        click.echo(f"genre: (inherited from parent) code_bearing={code_bearing}")
    else:
        genre, genre_rationale = classify_playlist_genre(
            playlist_title, videos, model=models["router"]
        )
        code_bearing = genre in CODE_BEARING_GENRES
        click.echo(f"genre: {genre.value} (code_bearing={code_bearing}) — {genre_rationale[:120]}")

    # Sub-agent orchestration (opt-in via --sub-agents N; default 1 keeps the
    # original single-process flow). Splits the playlist into N contiguous
    # shards, runs stages 01-04 as independent parallel worker processes, then
    # runs Stage 05 once over the merged output. See docs/sub-agents.md.
    if sub_agents > 1:
        orchestrator_run_time = _parse_run_timestamp(run_timestamp)
        click.echo(f"run_time: {orchestrator_run_time.isoformat(timespec='seconds')}")
        exit_code = orchestrate_sub_agents(
            total_videos=len(videos),
            shard_count=sub_agents,
            run_time=orchestrator_run_time,
            logs_dir=logs_dir,
            base_argv=strip_cli_option(sys.argv[1:], "--sub-agents"),
            run_synthesis=not skip_synthesis,
            code_bearing=code_bearing,
        )
        sys.exit(exit_code)

    # Sub-agent shard slicing (internal --video-range). Restricts the per-video
    # work below to this shard's contiguous slice; genre above is already pinned.
    if video_range is not None:
        try:
            shard_start, shard_end = parse_video_range(video_range)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        videos = videos[shard_start:shard_end]
        click.echo(
            f"sub-agent shard: videos [{shard_start}:{shard_end}] → {len(videos)} this shard"
        )
        if not videos:
            click.echo("No videos in this shard; nothing to do.")
            return

    est_timeouts = compute_synthesis_timeouts(len(videos), override=effective_synthesis_timeout)
    total_duration = sum(v.duration or 0 for v in videos)
    click.echo(
        f"synthesis_estimate: {len(videos)} videos"
        f" → timeout α={est_timeouts['alpha']}s β={est_timeouts['beta']}s"
        f" leader={est_timeouts['leader']}s"
        + (f", total_duration={total_duration // 60}min" if total_duration else "")
    )

    run_time = _parse_run_timestamp(run_timestamp)
    click.echo(f"run_time: {run_time.isoformat(timespec='seconds')}")

    # Per-playlist proper-noun sheet (a byproduct of Stage 01b correction).
    # Gated on transcript_correction. On load we (a) feed already-known terms to
    # Stage 01b so they skip the web search, and (b) promote the user's
    # corrections into glossary.json. The sheet's resolved spellings are applied
    # to the Stage 05 output further below.
    proper_noun_sheet_path: Path | None = None
    known_terms: list[tuple[str, str]] | None = None
    if cfg.transcript_correction and not dry_run:
        proper_noun_sheet_path = _proper_noun_sheet_path(videos[0], run_time)
        start_sheet = load_sheet(proper_noun_sheet_path)
        known_terms = known_pairs(start_sheet) or None
        if cfg.glossary_path is not None:
            promoted = _promote_corrections_to_glossary(start_sheet, cfg.glossary_path)
            if promoted:
                click.echo(
                    f"glossary: promoted {promoted} corrected term(s) into {cfg.glossary_path.name}"
                )

    folder_override: str | None = None
    if synthesis_only:
        click.echo("\n=== --synthesis-only: loading existing 04 md files ===")
        matched_videos, matched_bodies, folder_override = _collect_existing_learning_bodies(
            videos, playlist_title, run_time
        )
        click.echo(f"matched: {len(matched_videos)}/{len(videos)} videos")
        if len(matched_videos) < min_playlist_size:
            click.echo(
                f"[skip] only {len(matched_videos)} matched (< {min_playlist_size}), "
                "stage 05 skipped"
            )
            return
        synthesis_videos = matched_videos
        synthesis_bodies = matched_bodies
    else:
        # Checkpoint: detect already-completed videos in one pass
        force_set = set(force_video)
        completed_ids = get_completed_video_ids(playlist_title, run_time) if not dry_run else set()
        if completed_ids:
            skippable = completed_ids - force_set
            if skippable:
                click.echo(f"checkpoint: {len(skippable)} videos already complete, will skip")

        # Separate videos into skip (checkpoint) and process lists
        to_process: list[tuple[int, VideoMeta]] = []
        results: list[VideoRunResult] = []
        for i, video in enumerate(videos, 1):
            if video.video_id in completed_ids and video.video_id not in force_set:
                click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
                click.echo("  [skip] checkpoint: stage 04 already exists")
                body = _load_existing_04_body(video.video_id, playlist_title, run_time)
                results.append(VideoRunResult(video=video, learning_md_body=body))
            else:
                to_process.append((i, video))

        if resume_reviewed:
            # Phase 3: filter to videos whose 02_Summary.md has `reviewed: true`.
            to_process = _filter_to_reviewed(to_process, playlist_title, run_time)

        # Process remaining videos
        if to_process and concurrency > 1:
            process_videos = [v for _, v in to_process]
            concurrent_results = asyncio.run(
                _run_videos_concurrent(
                    process_videos,
                    run_time,
                    concurrency=concurrency,
                    dry_run=dry_run,
                    capture_format=capture_format,
                    models=models,
                    filler_words=filler_words,
                    stop_after_capture=stop_after_capture,
                    capture_backend=active_capture_backend,
                    code_bearing=code_bearing,
                    glossary=cfg.glossary,
                    media_map=media_map,
                    correct_transcript=cfg.transcript_correction,
                    known_terms=known_terms,
                )
            )
            results.extend(concurrent_results)
        else:
            for i, video in to_process:
                click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
                result = _process_video(
                    video,
                    run_time,
                    dry_run=dry_run,
                    capture_format=capture_format,
                    models=models,
                    filler_words=filler_words,
                    stop_after_capture=stop_after_capture,
                    capture_backend=active_capture_backend,
                    code_bearing=code_bearing,
                    glossary=cfg.glossary,
                    media_path=media_map.get(video.video_id),
                    correct_transcript=cfg.transcript_correction,
                    known_terms=known_terms,
                )
                results.append(result)

        # Write the proper nouns Stage 01b confirmed into the per-playlist sheet
        # (preserving any user corrections) so they can be reviewed before
        # Stage 05 and reused on the next run. Done before the stop-after-capture
        # return so the sheet is available during a manual review pause.
        if proper_noun_sheet_path is not None:
            _update_proper_noun_sheet(proper_noun_sheet_path, results)

        if stop_after_capture:
            click.echo(
                "\n[stop-after-capture] Phase 1 complete. Review 02_Summary.md, "
                "set `reviewed: true`, then re-run with --resume-reviewed."
            )
            return

        succeeded = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]

        click.echo("\n=== Video processing summary ===")
        click.echo(f"succeeded: {len(succeeded)}/{len(videos)}")
        for f in failed:
            click.echo(f"  FAIL {f.video.video_id}: {f.error}")

        if skip_synthesis:
            click.echo("[skip] --skip-synthesis: stage 05 bypassed")
            return

        if len(succeeded) < min_playlist_size:
            click.echo(
                f"[skip] only {len(succeeded)} videos succeeded (< {min_playlist_size}), "
                "stage 05 skipped"
            )
            return
        synthesis_videos = [r.video for r in succeeded]
        synthesis_bodies = [r.learning_md_body or "" for r in succeeded]

    # Apply the user's proper-noun corrections to the Stage 05 output: build a
    # glossary from the sheet's user-corrected rows (correction = canonical,
    # system spelling = alias) and rewrite the MOC + chapters with it.
    proper_noun_glossary: Glossary | None = None
    if proper_noun_sheet_path is not None:
        sheet_glossary = correction_glossary(load_sheet(proper_noun_sheet_path))
        if sheet_glossary.entries:
            proper_noun_glossary = sheet_glossary

    click.echo("\n=== Stage 05 Synthesis (Agent Teams) ===")
    synth_timeouts = compute_synthesis_timeouts(
        len(synthesis_videos), override=effective_synthesis_timeout
    )
    click.echo(log_synthesis_preflight(len(synthesis_videos), synthesis_bodies, synth_timeouts))
    synthesis_result = run_stage_synthesis(
        synthesis_videos,
        synthesis_bodies,
        run_time=run_time,
        playlist_title=playlist_title,
        model=model,
        agent_models={k: models[k] for k in ("alpha", "beta", "leader", "reviewer")},
        min_playlist_size=min_playlist_size,
        max_chapters=max_chapters,
        dry_run=dry_run,
        folder_name_override=folder_override,
        synthesis_timeout=effective_synthesis_timeout,
        profile=effective_synthesis_profile,
        proper_noun_glossary=proper_noun_glossary,
    )

    if synthesis_result.skipped:
        click.echo(f"[skip] {synthesis_result.skip_reason}")
    elif synthesis_result.error:
        click.echo(f"[error] synthesis: {synthesis_result.error}")
    else:
        click.echo(f"MOC:       {synthesis_result.moc_path}")
        click.echo(f"chapters:  {len(synthesis_result.chapter_paths)}")
        for p in synthesis_result.chapter_paths:
            click.echo(f"  - {p.name}")
        click.echo(f"meta:      {synthesis_result.meta_path}")
        click.echo(
            f"tokens:    in={synthesis_result.total_input_tokens}"
            f" out={synthesis_result.total_output_tokens}"
            f" cache_read={synthesis_result.total_cache_read_tokens}"
            f" cache_create={synthesis_result.total_cache_creation_tokens}"
        )
        click.echo(f"cost:      ${synthesis_result.total_cost_usd:.3f}")
        click.echo(f"duration:  {synthesis_result.total_duration_ms / 1000:.1f}s")

    if not synthesis_only:
        _print_cost_breakdown(results, synthesis_result)


if __name__ == "__main__":
    cli()
