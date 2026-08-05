"""Video-level checkpoint: skip already-processed videos.

Checks whether a stage 04 md file with a matching `video_id` frontmatter
field exists in the expected playlist folder. If so, the video can be
skipped entirely (stages 01-04 all write to the same playlist folder,
and 04 is the last to complete, so its presence implies 01-03 are also
done).

Design note (ミノ駆動本 ch8 単一責任):
    This module does ONE thing — answer "is this video already done?"
    It does not decide what to do about it; that's the caller's job.

Trust model (M3 hardening)
--------------------------
The vault directory is treated as semi-trusted: a user with write access
to the vault could in principle plant a crafted md file to make the
pipeline skip real videos (self-DoS). `extract_trusted_video_id` applies
three defensive layers before accepting a checkpoint marker:

  1. Requires a well-formed YAML frontmatter block (`---...---`) at the
     top of the file. `video_id:` anywhere else is ignored.
  2. Validates the video_id matches YouTube's canonical format
     (11 chars from `[A-Za-z0-9_-]`). Arbitrary strings, partial IDs,
     and path-like values are rejected.
  3. If a `URL:` field is also present in the frontmatter, enforces
     that it references the same video_id (integrity cross-check).

On failure the file is silently skipped — same safe fallback as
"no matching file found".
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..obsidian import (
    format_playlist_folder_name,
    playlist_folder_title,
    sanitize_title_for_filename,
)
from ..pipeline import LEARNING_BASE, LEGACY_LEARNING_DIR, UNIT_DIRS
from .path_safety import ensure_safe_path

# YouTube video IDs are always 11 chars from [A-Za-z0-9_-].
_YT_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Match a YAML frontmatter block that opens the file: `---\n...\n---`.
_FRONTMATTER_BLOCK_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

# Inside-frontmatter field lines.
_VIDEO_ID_LINE_RE = re.compile(r'^video_id:\s*"([^"\n]+)"\s*$', re.MULTILINE)
_URL_LINE_RE = re.compile(r'^URL:\s*"([^"\n]+)"\s*$', re.MULTILINE)

# Cap the frontmatter scan — long titles/playlists still fit comfortably.
_FRONTMATTER_SCAN_BYTES = 2048


def extract_trusted_video_id(md_bytes: bytes) -> str | None:
    """Extract a validated video_id from a pipeline-produced md file.

    Returns the 11-char YouTube video_id on success, else `None`. See
    module docstring for the three defense layers this enforces.
    Never raises — any I/O or parse failure returns `None`.
    """
    try:
        head = md_bytes[:_FRONTMATTER_SCAN_BYTES].decode("utf-8", errors="replace")
    except Exception:
        return None

    fm_match = _FRONTMATTER_BLOCK_RE.match(head)
    if fm_match is None:
        return None
    block = fm_match.group(1)

    vid_match = _VIDEO_ID_LINE_RE.search(block)
    if vid_match is None:
        return None
    video_id = vid_match.group(1)

    if _YT_VIDEO_ID_RE.match(video_id) is None:
        return None

    # Integrity cross-check: if URL is present, it must embed the same video_id.
    url_match = _URL_LINE_RE.search(block)
    if url_match is not None:
        url = url_match.group(1)
        if f"v={video_id}" not in url and f"/{video_id}" not in url:
            return None

    return video_id


def read_trusted_video_id(md_path: Path) -> str | None:
    """Read an md file and return its validated video_id, or None on any failure."""
    try:
        data = md_path.read_bytes()
    except OSError:
        return None
    return extract_trusted_video_id(data)


def _find_learning_folders(
    playlist_title: str, run_date: datetime, *, vault_root: Path
) -> list[Path]:
    """Every 04_Learning_Material folder this run may read, newest first.

    The canonical name (`YYYY-MM-DD-HHmm <title>`) comes first when it exists;
    then the other same-day folders whose *title* equals the sanitized playlist
    title (legacy date-only names included).

    Two rules build that same-day set, in this order — first narrow, then order:

    1. Exact title. A substring rule would let a shorter playlist
       (`ML Python`) claim a longer same-day sibling (`ML Python Advanced`)
       as its Stage 04 folder, skip 01-04 for any overlapping `video_id`, and
       feed the foreign body into Stage 05. The comparison lives in
       `obsidian.playlist_folder_title`, shared with the resume path so both
       same-day fallbacks stay in step.
    2. Newest first among what survives rule 1. `iterdir()` order is
       filesystem-dependent, so a first-match rule would let a morning
       run shadow an afternoon `--force-video` rewrite of the same playlist.
       Folder names open with `YYYY-MM-DD-HHmm`, so a descending name sort is
       a time sort — the same rule `resume._unit_folder_candidates` uses.

    Why a list and not one folder: a partial rerun (`--force-video` on a single
    video) writes a folder holding only that video. Reading completion from the
    newest folder alone loses the videos that only an earlier folder has, so
    their stages 01-04 run again even though they finished. Callers that decide
    completion union across the whole list; callers that need a body walk it
    newest first and stop at the folder that actually holds the `video_id`.

    Historical `04_Lerning_Material` (typo) folders are also searched
    so existing vaults continue to work without renaming. See
    `pipeline.LEGACY_LEARNING_DIR`.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    bases = [
        vault_root
        / ensure_safe_path(f"{LEARNING_BASE}/{UNIT_DIRS['learning']}", vault_root=vault_root),
        vault_root
        / ensure_safe_path(f"{LEARNING_BASE}/{LEGACY_LEARNING_DIR}", vault_root=vault_root),
    ]
    bases = [b for b in bases if b.exists()]
    if not bases:
        return []

    canonical = bases[0] / format_playlist_folder_name(run_date, playlist_title)

    # Same-day folders with an exact title (handles legacy date-only names and
    # pre-concealment invisible characters in the title segment).
    date_prefix = run_date.strftime("%Y-%m-%d")

    # Also handle `/`-separated playlist titles (take last segment). Guard on the
    # stripped title: an empty needle must not match every dated folder.
    from ..obsidian import _strip_playlist_category_prefix, limit_title_for_path_component

    display_title = _strip_playlist_category_prefix(playlist_title)
    # Capped like playlist_folder_title's side, or a long title never matches.
    title_needle = limit_title_for_path_component(sanitize_title_for_filename(display_title))

    matches: list[Path] = []
    if title_needle:
        for b in bases:
            try:
                children = list(b.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_dir() or not child.name.startswith(date_prefix):
                    continue
                # Folders created before the concealment defense may still contain
                # zero-width/bidi characters. Comparing sanitized titles keeps those
                # completed runs discoverable after upgrading.
                if playlist_folder_title(child.name) == title_needle:
                    matches.append(child)
    # Folder names start with YYYY-MM-DD-HHmm (legacy: YYYY-MM-DD), so a
    # descending name sort puts the newest same-day run first.
    matches.sort(key=lambda child: child.name, reverse=True)

    folders: list[Path] = []
    if canonical.exists():
        folders.append(canonical)
    folders.extend(child for child in matches if child != canonical)
    return folders


def _find_learning_folder(
    playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """The folder this run owns — canonical if present, else the newest sibling.

    Kept for callers that need exactly one folder. Completion checks must not
    stop here; see `_find_learning_folders` for why.
    """
    folders = _find_learning_folders(playlist_title, run_date, vault_root=vault_root)
    return folders[0] if folders else None


def is_video_complete(
    video_id: str,
    playlist_title: str,
    run_date: datetime,
    *,
    vault_root: Path,
) -> bool:
    """Return True if a stage 04 md with matching video_id already exists.

    Scans **every** same-day 04_Learning_Material folder this playlist may own
    for an .md file whose YAML frontmatter contains `video_id: "<video_id>"`.
    A partial rerun leaves the other videos only in the earlier folder, so
    stopping at the newest one would report them as incomplete.
    """
    return any(
        read_trusted_video_id(md) == video_id
        for folder in _find_learning_folders(playlist_title, run_date, vault_root=vault_root)
        for md in folder.glob("*.md")
    )


def get_completed_video_ids(
    playlist_title: str,
    run_date: datetime,
    *,
    vault_root: Path,
) -> set[str]:
    """Return the set of video_ids that have completed stage 04.

    Useful for batch skip decisions without calling is_video_complete
    in a loop (one sweep instead of N). The ids are the **union** over every
    same-day folder of this playlist: a morning run finishing A and B followed
    by an afternoon `--force-video A` leaves B only in the morning folder, and
    counting the afternoon folder alone would re-run stages 01-04 for B.
    """
    ids: set[str] = set()
    for folder in _find_learning_folders(playlist_title, run_date, vault_root=vault_root):
        for md in folder.glob("*.md"):
            vid = read_trusted_video_id(md)
            if vid is not None:
                ids.add(vid)
    return ids
