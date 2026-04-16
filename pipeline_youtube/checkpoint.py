"""Video-level checkpoint: skip already-processed videos.

Checks whether a stage 04 md file with a matching `video_id` frontmatter
field exists in the expected playlist folder. If so, the video can be
skipped entirely (stages 01-04 all write to the same playlist folder,
and 04 is the last to complete, so its presence implies 01-03 are also
done).

Design note (ミノ駆動本 ch8 単一責任):
    This module does ONE thing — answer "is this video already done?"
    It does not decide what to do about it; that's the caller's job.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import get_vault_root
from .obsidian import format_playlist_folder_name, sanitize_title_for_filename
from .path_safety import ensure_safe_path
from .pipeline import LEARNING_BASE, UNIT_DIRS

_VIDEO_ID_RE = re.compile(r'^video_id:\s*"([^"]+)"', re.MULTILINE)


def _find_learning_folder(playlist_title: str, run_date: datetime) -> Path | None:
    """Locate the 04_Lerning_Material playlist folder for a given date.

    Tries the canonical name first (`YYYY-MM-DD-HHmm <title>`), then
    falls back to any folder starting with today's date prefix and
    containing the sanitized playlist title. Returns None if nothing
    matches.
    """
    vault_root = get_vault_root()
    base = vault_root / ensure_safe_path(f"{LEARNING_BASE}/{UNIT_DIRS['learning']}")

    if not base.exists():
        return None

    # Canonical name
    canonical = base / format_playlist_folder_name(run_date, playlist_title)
    if canonical.exists():
        return canonical

    # Fallback: date prefix + title substring (handles legacy folder names)
    date_prefix = run_date.strftime("%Y-%m-%d")
    title_needle = sanitize_title_for_filename(playlist_title)
    if not title_needle:
        return None

    # Also handle `/`-separated playlist titles (take last segment)
    from .obsidian import _strip_playlist_category_prefix

    display_title = _strip_playlist_category_prefix(playlist_title)
    title_needle = sanitize_title_for_filename(display_title)

    for child in base.iterdir():
        if child.is_dir() and child.name.startswith(date_prefix) and title_needle in child.name:
            return child
    return None


def is_video_complete(
    video_id: str,
    playlist_title: str,
    run_date: datetime,
) -> bool:
    """Return True if a stage 04 md with matching video_id already exists.

    Scans the 04_Lerning_Material playlist folder for any .md file whose
    YAML frontmatter contains `video_id: "<video_id>"`.
    """
    folder = _find_learning_folder(playlist_title, run_date)
    if folder is None or not folder.exists():
        return False

    for md in folder.glob("*.md"):
        try:
            # Only read the first 500 bytes — frontmatter is at the top
            head = md.read_bytes()[:500].decode("utf-8", errors="replace")
        except OSError:
            continue
        m = _VIDEO_ID_RE.search(head)
        if m and m.group(1) == video_id:
            return True
    return False


def get_completed_video_ids(
    playlist_title: str,
    run_date: datetime,
) -> set[str]:
    """Return the set of video_ids that have completed stage 04.

    Useful for batch skip decisions without calling is_video_complete
    in a loop (one folder scan instead of N).
    """
    folder = _find_learning_folder(playlist_title, run_date)
    if folder is None or not folder.exists():
        return set()

    ids: set[str] = set()
    for md in folder.glob("*.md"):
        try:
            head = md.read_bytes()[:500].decode("utf-8", errors="replace")
        except OSError:
            continue
        m = _VIDEO_ID_RE.search(head)
        if m:
            ids.add(m.group(1))
    return ids
