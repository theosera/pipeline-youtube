"""Write per-chapter md files for a Stage 05 synthesis result."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..obsidian import build_frontmatter, sanitize_title_for_filename
from ..path_safety import ensure_safe_path
from .scoring import SynthesisChapterBody


def chapter_filename(index: int, label: str) -> str:
    """Build a safe chapter md filename: `{NN}_{sanitized-label}.md`.

    - Index is zero-padded to 2 digits (`01`, `02`, ..., `10`, `11`)
    - Label is sanitized with `sanitize_title_for_filename`
    - Max length 200 chars (filesystem safety)
    """
    safe_label = sanitize_title_for_filename(label) or f"chapter-{index}"
    name = f"{index:02d}_{safe_label}.md"
    return name[:200]


def write_chapter(
    chapter: SynthesisChapterBody,
    playlist_dir: Path,
    *,
    run_time: datetime,
    playlist_title: str,
) -> Path:
    """Write a single chapter md and return the absolute path.

    Filename is `{NN}_{label}.md` under `playlist_dir`. The body is
    whatever the Leader agent produced; we prepend frontmatter only.
    """
    filename = chapter_filename(chapter.chapter_index, chapter.label)
    target = playlist_dir / filename

    fm = build_frontmatter(
        dt=run_time,
        title=chapter.label,
        url="",
        tags=["memo", "youtube", "synthesis"],
        extra={
            "playlist": playlist_title,
            "chapter": str(chapter.chapter_index),
            "category": chapter.category,
            "sources": ", ".join(chapter.source_video_ids),
        },
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fm + "\n" + chapter.body_markdown.strip() + "\n", encoding="utf-8")
    return target


def validate_chapter_relative_path(relative_path: str) -> str:
    """Run a chapter output path through the 7-layer path-safety filter."""
    return ensure_safe_path(relative_path)
