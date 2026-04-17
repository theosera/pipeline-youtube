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
import contextlib
import json
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

from .checkpoint import get_completed_video_ids
from .config import set_dry_run, set_vault_root
from .obsidian import format_playlist_folder_name
from .path_safety import ensure_safe_path
from .pipeline import LEARNING_BASE, UNIT_DIRS, compute_note_paths, create_placeholder_notes
from .playlist import VideoMeta, fetch_metadata
from .stages.capture import run_stage_capture
from .stages.learning import run_stage_learning
from .stages.scripts import run_stage_scripts
from .stages.summary import run_stage_summary
from .stages.synthesis import MIN_PLAYLIST_SIZE, run_stage_synthesis
from .stats import record_transcript_stat

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def _load_vault_root(config_path: Path) -> Path:
    if not config_path.exists():
        raise click.UsageError(
            f"config.json not found at {config_path}. "
            "Copy config.example.json to config.json and set vault_root."
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    vault_root = data.get("vault_root")
    if not vault_root or vault_root == "/path/to/your/Obsidian Vault":
        raise click.UsageError("config.json vault_root is not configured.")
    path = Path(vault_root).expanduser()
    if not path.exists():
        raise click.UsageError(f"vault_root does not exist: {path}")
    return path


@dataclass
class VideoRunResult:
    video: VideoMeta
    learning_md_path: Path | None = None
    learning_md_body: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.learning_md_body is not None


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + 4 :].lstrip()


_VIDEO_ID_IN_FRONTMATTER = re.compile(r'^video_id:\s*"([^"]+)"', re.MULTILINE)


def _load_existing_04_body(video_id: str, playlist_title: str, run_date: datetime) -> str | None:
    """Read the stage 04 body for a checkpoint-skipped video.

    Returns the frontmatter-stripped body, or None if the file can't be found.
    """
    from .checkpoint import _find_learning_folder

    folder = _find_learning_folder(playlist_title, run_date)
    if folder is None:
        return None
    for md in folder.glob("*.md"):
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        m = _VIDEO_ID_IN_FRONTMATTER.search(text)
        if m and m.group(1) == video_id:
            return _strip_frontmatter(text)
    return None


def _collect_existing_learning_bodies(
    videos: list[VideoMeta],
    playlist_title: str,
    run_time: datetime,
) -> tuple[list[VideoMeta], list[str], str]:
    """Scan the existing 04_Lerning_Material folder for the given playlist date
    and return `(videos, bodies, folder_name)` aligned by input video_id order.

    Also returns the resolved folder name so stage 05 can reuse the exact
    legacy name instead of creating a new one next to it.
    """
    from .config import get_vault_root

    rel_base = f"{LEARNING_BASE}/{UNIT_DIRS['learning']}"
    safe_rel_base = ensure_safe_path(rel_base)
    base_dir = get_vault_root() / safe_rel_base

    preferred = format_playlist_folder_name(run_time, playlist_title)
    learning_dir = base_dir / preferred
    folder_name = preferred

    if not learning_dir.exists() and base_dir.exists():
        # Fallback: match any sibling folder that begins with today's YYYY-MM-DD
        # and contains the sanitized playlist title as a substring. Handles
        # both the new YYYY-MM-DD HHmm <title> format and the legacy
        # YYYY-MM-DD <title> format from runs before the HHmm fix.
        from .obsidian import sanitize_title_for_filename

        date_prefix = run_time.strftime("%Y-%m-%d")
        title_needle = sanitize_title_for_filename(playlist_title)
        candidates = [
            p
            for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(date_prefix) and title_needle in p.name
        ]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            learning_dir = candidates[0]
            folder_name = learning_dir.name
            click.echo(f"(fallback: using legacy folder {folder_name!r})")

    if not learning_dir.exists():
        raise click.UsageError(
            f"04 folder not found: {learning_dir}. "
            "--synthesis-only requires stage 04 files from a prior run on the same date."
        )

    by_video_id: dict[str, str] = {}
    for md in sorted(learning_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        m = _VIDEO_ID_IN_FRONTMATTER.search(text)
        if not m:
            continue
        by_video_id[m.group(1)] = _strip_frontmatter(text)

    matched_videos: list[VideoMeta] = []
    matched_bodies: list[str] = []
    for v in videos:
        body = by_video_id.get(v.video_id)
        if body:
            matched_videos.append(v)
            matched_bodies.append(body)
    return matched_videos, matched_bodies, folder_name


def _process_video(
    video: VideoMeta,
    run_time: datetime,
    *,
    dry_run: bool,
    capture_format: str,
    model: str,
) -> VideoRunResult:
    try:
        paths = compute_note_paths(video, run_time)
        create_placeholder_notes(video, run_time, dry_run=dry_run)

        click.echo("  [01] scripts...", nl=False)
        transcript = run_stage_scripts(video, paths["scripts"], dry_run=dry_run)
        with contextlib.suppress(Exception):
            record_transcript_stat(video, transcript)
        click.echo(
            f" source={transcript.source.value}"
            f" snippets={len(transcript.snippets)}"
            f" lang={transcript.language or '-'}"
        )

        if not transcript.snippets:
            return VideoRunResult(video=video, error="no_transcript_snippets")

        click.echo("  [02] summary...", nl=False)
        summary_resp = run_stage_summary(
            video, paths["summary"], transcript, model=model, dry_run=dry_run
        )
        click.echo(
            f" in={summary_resp.input_tokens or 0}"
            f" out={summary_resp.output_tokens or 0}"
            f" cost=${summary_resp.total_cost_usd or 0:.3f}"
        )

        click.echo("  [03] capture...", nl=False)
        capture_result = run_stage_capture(
            video,
            paths["summary"],
            paths["capture"],
            capture_format=capture_format,  # type: ignore[arg-type]
            dry_run=dry_run,
        )
        if capture_result.error and not capture_result.outcomes:
            click.echo(f" FAILED: {capture_result.error}")
        else:
            click.echo(
                f" {capture_result.success_count}/{len(capture_result.ranges)} ranges"
                f" fmt={capture_result.capture_format}"
            )

        click.echo("  [04] learning...", nl=False)
        learning_resp = run_stage_learning(
            video,
            paths["summary"],
            paths["capture"],
            paths["learning"],
            run_time=run_time,
            model=model,
            dry_run=dry_run,
        )
        click.echo(
            f" in={learning_resp.input_tokens or 0}"
            f" out={learning_resp.output_tokens or 0}"
            f" cost=${learning_resp.total_cost_usd or 0:.3f}"
        )

        if dry_run:
            body = learning_resp.text.strip()
        else:
            body = _strip_frontmatter(paths["learning"].read_text(encoding="utf-8"))

        return VideoRunResult(
            video=video,
            learning_md_path=paths["learning"],
            learning_md_body=body,
        )
    except Exception as e:
        traceback.print_exc()
        return VideoRunResult(video=video, error=f"{type(e).__name__}: {e}")


async def _run_videos_concurrent(
    videos: list[VideoMeta],
    run_time: datetime,
    *,
    concurrency: int,
    dry_run: bool,
    capture_format: str,
    model: str,
) -> list[VideoRunResult]:
    """Process multiple videos concurrently with bounded parallelism."""
    sem = asyncio.Semaphore(concurrency)

    async def _task(i: int, video: VideoMeta) -> VideoRunResult:
        async with sem:
            click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
            return await asyncio.to_thread(
                _process_video,
                video,
                run_time,
                dry_run=dry_run,
                capture_format=capture_format,
                model=model,
            )

    tasks = [_task(i, v) for i, v in enumerate(videos, 1)]
    return list(await asyncio.gather(*tasks))


@click.command()
@click.argument("url", required=False)
@click.option("--dry-run", is_flag=True, help="Do not write to vault; print to stdout only.")
@click.option(
    "--concurrency",
    type=click.IntRange(1, 5),
    default=1,
    help="Videos in parallel (1-5, default 1).",
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
def cli(
    url: str | None,
    dry_run: bool,
    concurrency: int,
    skip_synthesis: bool,
    synthesis_only: bool,
    force_video: tuple[str, ...],
    capture_format: str,
    model: str,
    min_playlist_size: int,
    max_chapters: int | None,
    config_path: Path | None,
) -> None:
    """Process a YouTube playlist or single-video URL end-to-end."""
    if not url:
        click.echo("Usage: pipeline-youtube <playlist-or-video-url> [options]")
        sys.exit(2)

    cfg_path = config_path or DEFAULT_CONFIG_PATH
    vault_root = _load_vault_root(cfg_path)
    set_vault_root(vault_root)
    set_dry_run(dry_run)

    click.echo(f"vault_root: {vault_root}")
    click.echo(f"dry_run: {dry_run}")
    click.echo(f"model: {model}")
    click.echo(f"capture_format: {capture_format}")
    click.echo(f"concurrency: {concurrency}")
    click.echo(f"min_playlist_size: {min_playlist_size}")
    click.echo(f"max_chapters: {max_chapters if max_chapters is not None else 'auto'}")

    click.echo("fetching metadata...")
    videos = fetch_metadata(url)
    if not videos:
        click.echo("No videos found.")
        sys.exit(1)

    playlist_title = videos[0].playlist_title or videos[0].title or "single video"
    click.echo(f"playlist: {playlist_title!r}")
    click.echo(f"videos: {len(videos)}")

    run_time = datetime.now()
    click.echo(f"run_time: {run_time.isoformat(timespec='seconds')}")

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
                    model=model,
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
                    model=model,
                )
                results.append(result)

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

    click.echo("\n=== Stage 05 Synthesis (Agent Teams) ===")
    synthesis_result = run_stage_synthesis(
        synthesis_videos,
        synthesis_bodies,
        run_time=run_time,
        playlist_title=playlist_title,
        model=model,
        min_playlist_size=min_playlist_size,
        max_chapters=max_chapters,
        dry_run=dry_run,
        folder_name_override=folder_override,
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


if __name__ == "__main__":
    cli()
