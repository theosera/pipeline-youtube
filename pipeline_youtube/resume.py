"""Lookup of prior-run output for the resume / synthesis-only flows.

Extracted from `main.py`. These helpers locate existing Stage 02/04 notes on
disk (by trusted frontmatter video_id) so `--synthesis-only`,
`--resume-reviewed`, and checkpoint-skip can rebuild their inputs without
reprocessing.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import click

from .checkpoint import read_trusted_video_id
from .obsidian import format_playlist_folder_name, playlist_folder_title
from .path_safety import ensure_safe_path
from .pipeline import LEARNING_BASE, UNIT_DIRS
from .playlist import VideoMeta
from .run_result import _strip_frontmatter

# ``resolve_unique_path`` appends ``-2``, ``-3``, … on same-folder collisions.
_COLLISION_SUFFIX_RE = re.compile(r"-(\d+)$")


def _collision_suffix_n(stem: str) -> int:
    """Return the ``resolve_unique_path`` collision ordinal for a note stem.

    Unsuffixed stems count as ``1``. A trailing ``-N`` with ``N >= 2`` is the
    collision ordinal; bare titles that happen to end in ``-1`` stay at ``1``.
    """
    match = _COLLISION_SUFFIX_RE.search(stem)
    if match is None:
        return 1
    n = int(match.group(1))
    return n if n >= 2 else 1


def _prefer_latest_unit_md(candidates: list[Path]) -> Path | None:
    """Pick the freshest note among several files for the same ``video_id``.

    Same-folder rewrites (``--force-video`` + matching ``--run-timestamp``)
    leave both ``Title.md`` and ``Title-2.md``. A ``sorted(glob)`` last-wins
    dict keeps the *stale* unsuffixed file because ``Title-2.md`` sorts before
    ``Title.md`` (``'-' < '.'``). Prefer newer mtime, then higher collision
    suffix, so later consumers (checkpoint skip / ``--synthesis-only``) feed
    Stage 05 the forced rewrite.
    """
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, int, str]:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = -1
        return (mtime_ns, _collision_suffix_n(path.stem), path.name)

    return max(candidates, key=sort_key)


def _parse_run_timestamp(run_timestamp: str | None) -> datetime:
    """Resolve the shared run_time, surfacing a bad --run-timestamp as a clean CLI error."""
    if not run_timestamp:
        return datetime.now()
    try:
        return datetime.fromisoformat(run_timestamp)
    except ValueError as exc:
        raise click.UsageError(f"invalid --run-timestamp: {run_timestamp!r}") from exc


def _unit_base_dir(unit_key: str, *, vault_root: Path) -> Path | None:
    """Resolve the vault dir holding a unit's playlist folders, or None if absent.

    ``vault_root`` is injected by the caller (``runtime.vault_root``); the
    relative path still goes through ``ensure_safe_path`` so a malformed
    ``UNIT_DIRS`` entry can never escape the vault.
    """
    if unit_key not in UNIT_DIRS:
        raise ValueError(f"unknown unit key: {unit_key!r}")
    rel = f"{LEARNING_BASE}/{UNIT_DIRS[unit_key]}"
    base = vault_root / ensure_safe_path(rel, vault_root=vault_root)
    return base if base.exists() else None


def _find_unit_md(
    video_id: str,
    playlist_title: str,
    run_date: datetime,
    unit_key: str,
    *,
    vault_root: Path,
    preferred_folder_name: str | None = None,
) -> Path | None:
    """Locate an existing unit md for `video_id` within a given run date.

    Used by Phase 3 (`--resume-reviewed`) to look up Stage 02/03 notes written
    in a prior Phase 1 run. Falls back across date-prefix matches — same day
    first, then earlier days — so users can review overnight and resume without
    re-passing ``--run-timestamp``.

    ``preferred_folder_name`` pins the search to the playlist folder already
    resolved for a sibling unit (e.g. reuse the Stage 02 folder when locating
    Stage 03), so 02/03/04 stay aligned even when several same-day folders exist.
    A pinned lookup that misses returns None rather than falling back — see below.
    """
    base = _unit_base_dir(unit_key, vault_root=vault_root)
    if base is None:
        return None

    if preferred_folder_name:
        # Fail closed. The caller pinned this folder so 02/03/04 all come from
        # one Phase 1 run; falling through to another same-day run would pair
        # the reviewed summary with unrelated captures, whose embeds are
        # path-qualified to *that* run's playlist folder. Returning None lets
        # the caller report reviewed_capture_not_found instead of mixing runs.
        preferred = base / preferred_folder_name
        if not preferred.exists():
            return None
        for md in preferred.glob("*.md"):
            if read_trusted_video_id(md) == video_id:
                return md
        return None

    for candidate_folder in _unit_folder_candidates(base, playlist_title, run_date):
        if not candidate_folder.exists():
            continue
        for md in candidate_folder.glob("*.md"):
            if read_trusted_video_id(md) == video_id:
                return md
    return None


def _find_summary_md(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """Locate the existing 02_Summary.md for `video_id` within a given run date.

    Used by Phase 3 (`--resume-reviewed`) to look up summaries written
    in a prior Phase 1 run. Falls back across date-prefix matches so
    users can resume on a different clock day.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    return _find_unit_md(video_id, playlist_title, run_date, "summary", vault_root=vault_root)


def _find_reviewed_summary_md(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """Locate a 02_Summary.md for ``video_id`` with frontmatter ``reviewed: true``.

    Phase 1 reruns create a newer folder whose summaries still have
    ``reviewed: false``. Looking up "newest note for video_id, then check
    reviewed" would skip the older folder the operator actually marked.
    Scan newest-first and return the first matching *reviewed* summary.

    Because candidates now extend to earlier days, this reviewed-first rule is
    what keeps the widening safe: an old folder is only ever reached when
    nothing newer holds an approved summary for this video.
    """
    from .obsidian import read_frontmatter_field

    base = _unit_base_dir("summary", vault_root=vault_root)
    if base is None:
        return None

    for candidate_folder in _unit_folder_candidates(base, playlist_title, run_date):
        if not candidate_folder.exists():
            continue
        for md in candidate_folder.glob("*.md"):
            if read_trusted_video_id(md) != video_id:
                continue
            value = read_frontmatter_field(md, "reviewed")
            if value and value.lower() == "true":
                return md
    return None


def _find_existing_04_md(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """Locate the stage 04 md for a checkpoint-skipped video.

    Walks the same-day folders newest first and stops at the first one that
    actually holds ``video_id``. An afternoon `--force-video` on one video
    leaves a folder without the others, so reading only the newest folder
    would hand Stage 05 nothing for them even though their notes exist in the
    morning folder — the same partial-rerun case `get_completed_video_ids`
    unions over. Within the chosen folder `_prefer_latest_unit_md` still picks
    the freshest `-N` collision copy.
    """
    from .checkpoint import _find_learning_folders

    for folder in _find_learning_folders(playlist_title, run_date, vault_root=vault_root):
        matches = [md for md in folder.glob("*.md") if read_trusted_video_id(md) == video_id]
        latest = _prefer_latest_unit_md(matches)
        if latest is not None:
            return latest
    return None


def _load_existing_04_body(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> str | None:
    """Read the stage 04 body for a checkpoint-skipped video.

    Returns the frontmatter-stripped body, or None if the file can't be found.
    Uses the same M3 hardened frontmatter validation as `is_video_complete`.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    md = _find_existing_04_md(video_id, playlist_title, run_date, vault_root=vault_root)
    if md is None:
        return None
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return None
    return _strip_frontmatter(text)


def _learning_path_for_reviewed_summary(
    summary_md: Path, video: VideoMeta, *, vault_root: Path
) -> Path:
    """Return the Stage 04 output path that matches a reviewed Stage 02 note.

    Phase 3 must write 04 into the same playlist-folder name as Phase 1's 02/03
    notes. Using ``compute_note_paths`` with Phase 3's wall-clock ``run_time``
    would create a sibling folder and orphan the reviewed inputs.
    """
    from .obsidian import resolve_unique_path

    rel_folder = f"{LEARNING_BASE}/{UNIT_DIRS['learning']}/{summary_md.parent.name}"
    safe_rel_folder = ensure_safe_path(rel_folder, vault_root=vault_root)
    folder = vault_root / safe_rel_folder
    candidate = folder / summary_md.name
    if not candidate.exists() or read_trusted_video_id(candidate) == video.video_id:
        return candidate
    return resolve_unique_path(folder, summary_md.stem, ".md")


def _filter_to_reviewed(
    to_process: list[tuple[int, VideoMeta]],
    playlist_title: str,
    run_time: datetime,
    *,
    vault_root: Path,
) -> list[tuple[int, VideoMeta]]:
    """Keep only videos that have a 02_Summary.md with `reviewed: true`.

    Searches every candidate playlist folder (newest first), not only the
    newest note for the video_id — a later unreviewed Phase 1 rerun must
    not hide an older summary the operator already approved.

    A match from a previous day is echoed rather than used silently: resuming
    across midnight is the normal case, but *which* run Stage 04 is about to
    consume is exactly the thing an operator needs to see confirmed.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    from .obsidian import read_frontmatter_field

    today = run_time.strftime("%Y-%m-%d")
    kept: list[tuple[int, VideoMeta]] = []
    for i, video in to_process:
        summary_md = _find_reviewed_summary_md(
            video.video_id, playlist_title, run_time, vault_root=vault_root
        )
        if summary_md is not None:
            folder_date = summary_md.parent.name[:10]
            if folder_date != today:
                click.echo(f"  [resume] {video.video_id}: reviewed on {folder_date}")
            kept.append((i, video))
            continue
        # Nothing reviewed matched. Re-scan without the reviewed filter so the
        # skip line names the real cause: "no summary written at all" and
        # "summary written but never approved" need different operator action.
        any_summary = _find_summary_md(
            video.video_id, playlist_title, run_time, vault_root=vault_root
        )
        if any_summary is None:
            click.echo(f"  [skip] {video.video_id}: no 02_Summary.md found")
        else:
            value = read_frontmatter_field(any_summary, "reviewed")
            click.echo(f"  [skip] {video.video_id}: reviewed={value!r}")
    return kept


def _unit_folder_candidates(base: Path, playlist_title: str, run_date: datetime):
    """Yield likely playlist folders holding unit files.

    Order: canonical, then same-day folders (newest first), then folders from
    earlier days (newest first). Every non-canonical candidate must match the
    sanitized playlist title *exactly* (see ``playlist_folder_title``).

    The earlier-day tier is what makes Phase 3 work across midnight. The
    workflow is "Phase 1 → a human reads 02_Summary.md → Phase 3", and that
    review routinely happens the next morning; a same-day-only search then
    reports "no 02_Summary.md found" for work that is sitting right there.
    Same-day is offered first, and only folders *older* than ``run_date`` are
    considered, so a stray future-dated folder (clock skew, a hand-typed
    ``--run-timestamp``) can never outrank today's run.

    Exact title matching is required on both tiers. A same-day substring rule
    previously admitted a longer playlist created today (``testlist Advanced``)
    when resuming ``testlist``. ``reviewed: true`` only gates approval — it does
    not prove the folder belongs to this playlist — so an overlapping
    ``video_id`` would let Phase 3 consume the wrong 02/03 notes and write
    Stage 04 into the Advanced folder.
    """
    from .obsidian import (
        _strip_playlist_category_prefix,
        limit_title_for_path_component,
        sanitize_title_for_filename,
    )

    canonical_name = format_playlist_folder_name(run_date, playlist_title)
    yield base / canonical_name

    date_prefix = run_date.strftime("%Y-%m-%d")
    display_title = _strip_playlist_category_prefix(playlist_title)
    # Capped like playlist_folder_title's side, or a long title never matches.
    title_needle = limit_title_for_path_component(sanitize_title_for_filename(display_title))
    if not title_needle:
        return
    try:
        matches = [
            child
            for child in base.iterdir()
            if (
                child.is_dir()
                # Exact title, not a substring: widening past today must not
                # start matching unrelated folders that merely share a word with
                # the playlist title. playlist_folder_title also normalizes
                # pre-defense names that still carry invisible characters, and
                # returns "" for anything without a YYYY-MM-DD prefix — so this
                # one test also rejects undated directories.
                and playlist_folder_title(child.name) == title_needle
                and child.name != canonical_name
            )
        ]
    except OSError:
        return
    # iterdir() order is filesystem-dependent, so with two Phase 1 runs on one
    # day the caller could silently get either. Folder names start with
    # YYYY-MM-DD-HHmm, so a descending name sort puts the newest run first —
    # the one the operator most likely just reviewed.
    matches.sort(key=lambda child: child.name, reverse=True)
    yield from (child for child in matches if child.name.startswith(date_prefix))
    # Earlier days last, so a same-day reviewed summary always wins. The date is
    # a fixed-width YYYY-MM-DD prefix, so a string compare orders it.
    yield from (child for child in matches if child.name[:10] < date_prefix)


def _collect_existing_learning_bodies(
    videos: list[VideoMeta],
    playlist_title: str,
    run_time: datetime,
    *,
    vault_root: Path,
) -> tuple[list[VideoMeta], list[str], str]:
    """Scan the existing 04_Learning_Material folder for this playlist and
    return `(videos, bodies, folder_name)` aligned by input video_id order.

    Folder resolution goes through the shared ``_unit_folder_candidates``, so
    ``--synthesis-only`` follows the same order Phase 3 uses: the canonical
    folder for ``run_time``, then same-day runs (newest first), then earlier
    days (newest first). Re-synthesizing material produced yesterday no longer
    needs ``--run-timestamp``; the run date still wins whenever it has a folder
    that actually contains this playlist's video_ids.

    Unlike Phase 3, this path has no ``reviewed: true`` gate, so a candidate is
    accepted only once it holds one of this playlist's ``video_id``s. An *empty*
    same-day ``testlist`` folder must not hide yesterday's completed ``testlist``
    material; ``_unit_folder_candidates`` already keeps a different playlist
    (``testlist Advanced``) out by exact title.

    Also returns the resolved folder name so stage 05 can reuse the exact
    legacy name instead of creating a new one next to it.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    rel_base = f"{LEARNING_BASE}/{UNIT_DIRS['learning']}"
    safe_rel_base = ensure_safe_path(rel_base, vault_root=vault_root)
    base_dir = vault_root / safe_rel_base

    preferred = format_playlist_folder_name(run_time, playlist_title)
    from .obsidian import (
        _strip_playlist_category_prefix,
        limit_title_for_path_component,
        sanitize_title_for_filename,
    )

    # --synthesis-only has no reviewed:true gate, so the exact title is
    # re-asserted here: the generator yields the canonical folder
    # unconditionally, and a caller-supplied playlist_title that sanitizes
    # differently must not slip through on that tier. The video_id check below
    # is what keeps an empty same-day folder from hiding yesterday's complete
    # run.
    # Capped like playlist_folder_title's side: without it a long title rejects
    # even its own canonical folder, whose name format_* already capped.
    title_needle = limit_title_for_path_component(
        sanitize_title_for_filename(_strip_playlist_category_prefix(playlist_title))
    )

    def _scan_learning_bodies(folder: Path) -> dict[str, Path]:
        """Map each trusted video_id in ``folder`` to the freshest note holding it.

        A same-folder rewrite (``--force-video`` with a matching
        ``--run-timestamp``) leaves both ``Title.md`` and ``Title-2.md`` for one
        video_id. Keeping the *path* rather than the body lets
        ``_prefer_latest_unit_md`` pick the rewrite instead of whichever name
        happens to sort last.
        """
        found: dict[str, Path] = {}
        for md in folder.glob("*.md"):
            vid = read_trusted_video_id(md)
            if vid is None:
                continue
            prev = found.get(vid)
            found[vid] = md if prev is None else (_prefer_latest_unit_md([prev, md]) or md)
        return found

    learning_dir: Path | None = None
    by_video_id: dict[str, Path] = {}
    for candidate in _unit_folder_candidates(base_dir, playlist_title, run_time):
        if not candidate.exists():
            continue
        if title_needle and playlist_folder_title(candidate.name) != title_needle:
            continue
        scanned = _scan_learning_bodies(candidate)
        matched_ids = [v.video_id for v in videos if v.video_id in scanned]
        if matched_ids:
            learning_dir = candidate
            by_video_id = scanned
            break
        # Exact folder but none of this playlist's videos — keep looking
        # (a same-day partial run must not hide an earlier complete one).
        if learning_dir is None:
            learning_dir = candidate
            by_video_id = scanned

    if learning_dir is None:
        raise click.UsageError(
            f"04 folder not found under {base_dir}. "
            "--synthesis-only requires stage 04 files from a prior run of this playlist."
        )

    folder_name = learning_dir.name
    if folder_name != preferred:
        # Which prior run is being re-synthesized is the one thing an operator
        # cannot infer from the command line, so name it either way.
        if folder_name[:10] != run_time.strftime("%Y-%m-%d"):
            click.echo(f"(resuming from {folder_name!r}, a run on {folder_name[:10]})")
        else:
            click.echo(f"(fallback: using folder {folder_name!r})")

    matched_videos: list[VideoMeta] = []
    matched_bodies: list[str] = []
    for v in videos:
        md = by_video_id.get(v.video_id)
        if md is None:
            continue
        try:
            data = md.read_bytes()
        except OSError:
            continue
        # Decode with replacement rather than strictly: a note that picked up a
        # bad byte should degrade, not abort the whole synthesis run.
        body = _strip_frontmatter(data.decode("utf-8", errors="replace"))
        if not body:
            # A truncated or hand-emptied 04 note still carries trusted
            # frontmatter, so the scan above finds it — but it holds no learning
            # material. Counting it would let it pass ``min_playlist_size`` and
            # run Stage 05 on an empty source. (``_strip_frontmatter`` also
            # returns "" for a whitespace-only body.)
            continue
        matched_videos.append(v)
        matched_bodies.append(body)
    return matched_videos, matched_bodies, folder_name
