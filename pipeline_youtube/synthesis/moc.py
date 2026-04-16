"""Write the `00_MOC.md` hub note for a Stage 05 synthesis result."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..obsidian import build_frontmatter
from .scoring import SynthesisMoc


def write_moc(
    moc: SynthesisMoc,
    target_path: Path,
    *,
    run_time: datetime,
    playlist_title: str,
) -> None:
    """Write `00_MOC.md` with frontmatter + leader-produced body.

    The file is written atomically in one `write_text` call. Parent
    directories are created if missing.
    """
    fm = build_frontmatter(
        dt=run_time,
        title=moc.title or f"{playlist_title} ハンズオン",
        url="",
        tags=["memo", "youtube", "synthesis", "moc"],
        extra={"playlist": playlist_title},
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(fm + "\n" + moc.body_markdown.strip() + "\n", encoding="utf-8")
