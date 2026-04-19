"""Stage 05: Playlist-level synthesis via Agent Teams.

Runs after all per-video stages (01-04) complete for a playlist with
≥3 successful videos. Reads every 04_Learning_Material md in the
playlist folder and orchestrates the α→β→Leader agent chain to
produce:

    {vault}/Permanent Note/08_YouTube学習/05_Synthesis/
        {YYYY-MM-DD <playlist_title>}/
            00_MOC.md
            01_<chapter>.md
            02_<chapter>.md
            ...
            _meta/
                duplicate_score.json

Execution is **sequential** (α→β→Leader) because the roles depend on
each other's output. Coverage (α topics vs β chapter topic_ids) is
computed deterministically in Python via `compute_coverage()` — no
LLM call. Claude's server-side cache shares context across consecutive
calls within ~5 minutes, so the cumulative cache-creation overhead is
paid only once in practice.

Skipping rules
--------------
- Playlists with fewer than `MIN_PLAYLIST_SIZE` (default 3) videos: skip.
- Playlists where stage 04 failed for all videos: skip.
- Single-video URLs: caller should not invoke this stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import get_vault_root
from ..obsidian import format_playlist_folder_name
from ..path_safety import ensure_safe_path
from ..playlist import VideoMeta
from ..synthesis.agents import (
    AgentCallResult,
    call_alpha,
    call_beta,
    call_leader,
    compute_coverage,
)
from ..synthesis.body_validator import extract_allowed_embeds
from ..synthesis.chapter import write_chapter
from ..synthesis.moc import write_moc
from ..synthesis.scoring import (
    ChapterPlan,
    CoverageReport,
    LeaderOutput,
    SynthesisParseError,
    Topic,
)

SYNTHESIS_BASE = "Permanent Note/08_YouTube学習/05_Synthesis"
META_SUBDIR = "_meta"
DUPLICATE_SCORE_FILENAME = "duplicate_score.json"
MIN_PLAYLIST_SIZE = 3


@dataclass(frozen=True)
class SynthesisStageResult:
    topics: list[Topic] = field(default_factory=list)
    chapters: list[ChapterPlan] = field(default_factory=list)
    coverage: CoverageReport | None = None
    leader_output: LeaderOutput | None = None
    moc_path: Path | None = None
    chapter_paths: list[Path] = field(default_factory=list)
    meta_path: Path | None = None
    agent_results: list[AgentCallResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens or 0 for r in self.agent_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens or 0 for r in self.agent_results)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(r.cache_creation_tokens or 0 for r in self.agent_results)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens or 0 for r in self.agent_results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd or 0.0 for r in self.agent_results)

    @property
    def total_duration_ms(self) -> int:
        return sum(r.duration_ms or 0 for r in self.agent_results)


def run_stage_synthesis(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    *,
    run_time: datetime,
    playlist_title: str,
    model: str = "sonnet",
    agent_models: dict[str, str] | None = None,
    min_playlist_size: int = MIN_PLAYLIST_SIZE,
    max_chapters: int | None = None,
    dry_run: bool = False,
    folder_name_override: str | None = None,
) -> SynthesisStageResult:
    """Orchestrate α→β→γ→leader and write MOC + chapter md files.

    Parameters
    ----------
    videos:
        Per-video metadata. Must align 1:1 with `learning_md_bodies`.
        Videos whose stage 04 failed should be filtered out BEFORE
        calling this function.
    learning_md_bodies:
        Frontmatter-stripped 04 md bodies for each video.
    run_time:
        Shared datetime for the synthesis folder and all frontmatter.
    playlist_title:
        Used for the output folder name and MOC title.
    model:
        Default model used for any agent not explicitly overridden.
    agent_models:
        Optional `{"alpha", "beta", "leader"}` override map.
        Missing keys fall back to `model`. (`gamma` accepted for
        config backward-compat but ignored — coverage is now a Python
        set diff, no LLM.)
    """
    am = agent_models or {}
    alpha_model = am.get("alpha", model)
    beta_model = am.get("beta", model)
    leader_model = am.get("leader", model)

    if len(videos) != len(learning_md_bodies):
        return SynthesisStageResult(
            error=f"length mismatch: {len(videos)} videos vs {len(learning_md_bodies)} bodies"
        )

    if len(videos) < min_playlist_size:
        return SynthesisStageResult(
            skipped=True,
            skip_reason=f"playlist has {len(videos)} videos (< {min_playlist_size})",
        )

    vault_root = get_vault_root()
    playlist_folder_name = folder_name_override or format_playlist_folder_name(
        run_time, playlist_title
    )
    rel_path = f"{SYNTHESIS_BASE}/{playlist_folder_name}"
    safe_rel = ensure_safe_path(rel_path)
    playlist_dir = vault_root / safe_rel

    agent_results: list[AgentCallResult] = []

    try:
        topics, alpha_res = call_alpha(
            videos, learning_md_bodies, model=alpha_model, playlist_title=playlist_title
        )
    except SynthesisParseError as e:
        return SynthesisStageResult(error=f"alpha_parse_failed: {e}")
    agent_results.append(alpha_res)

    try:
        chapters, beta_res = call_beta(topics, model=beta_model, max_chapters=max_chapters)
    except SynthesisParseError as e:
        return SynthesisStageResult(
            topics=topics, agent_results=agent_results, error=f"beta_parse_failed: {e}"
        )
    agent_results.append(beta_res)

    coverage = compute_coverage(topics, chapters)

    # Reflexion retry: if β missed any α-extracted topics, re-run β once
    # with the missing IDs fed back as an error instruction. One retry is
    # enough in practice — β either fixes it or is confused in a way
    # more retries won't help. The Leader still gets whatever β produces
    # on the final attempt, so this is a quality improvement, not a hard
    # gate (Gemini 2026-04-20 proposal approach A: 確定的自己修復).
    if coverage.missing_topic_ids:
        try:
            chapters, retry_res = call_beta(
                topics,
                model=beta_model,
                max_chapters=max_chapters,
                missing_topic_ids=coverage.missing_topic_ids,
            )
            agent_results.append(retry_res)
            coverage = compute_coverage(topics, chapters)
        except SynthesisParseError:
            # Retry parse fail: keep the first-attempt chapters and let
            # the Leader handle the residual miss. Don't abort the stage.
            pass

    try:
        leader_output, leader_res = call_leader(
            videos,
            learning_md_bodies,
            topics,
            chapters,
            coverage,
            model=leader_model,
            playlist_title=playlist_title,
        )
    except SynthesisParseError as e:
        return SynthesisStageResult(
            topics=topics,
            chapters=chapters,
            coverage=coverage,
            agent_results=agent_results,
            error=f"leader_parse_failed: {e}",
        )
    agent_results.append(leader_res)

    if dry_run:
        return SynthesisStageResult(
            topics=topics,
            chapters=chapters,
            coverage=coverage,
            leader_output=leader_output,
            agent_results=agent_results,
        )

    # Write files
    playlist_dir.mkdir(parents=True, exist_ok=True)

    allowed_assets = extract_allowed_embeds(learning_md_bodies)

    moc_path = playlist_dir / "00_MOC.md"
    write_moc(
        leader_output.moc,
        moc_path,
        run_time=run_time,
        playlist_title=playlist_title,
        allowed_assets=allowed_assets,
    )

    chapter_paths: list[Path] = []
    for chapter_body in leader_output.chapters:
        path = write_chapter(
            chapter_body,
            playlist_dir,
            run_time=run_time,
            playlist_title=playlist_title,
            allowed_assets=allowed_assets,
        )
        chapter_paths.append(path)

    meta_dir = playlist_dir / META_SUBDIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / DUPLICATE_SCORE_FILENAME
    meta_path.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": t.topic_id,
                        "label": t.label,
                        "aliases": t.aliases,
                        "source_videos": t.source_videos,
                        "duplication_count": t.duplication_count,
                        "category": t.category,
                        "summary": t.summary,
                    }
                    for t in topics
                ],
                "chapters": [
                    {
                        "index": c.index,
                        "label": c.label,
                        "category": c.category,
                        "topic_ids": c.topic_ids,
                    }
                    for c in chapters
                ],
                "coverage": {
                    "covered_topic_ids": coverage.covered_topic_ids,
                    "missing_topic_ids": coverage.missing_topic_ids,
                    "notes": coverage.notes,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return SynthesisStageResult(
        topics=topics,
        chapters=chapters,
        coverage=coverage,
        leader_output=leader_output,
        moc_path=moc_path,
        chapter_paths=chapter_paths,
        meta_path=meta_path,
        agent_results=agent_results,
    )
