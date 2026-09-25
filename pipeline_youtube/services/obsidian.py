"""Obsidian note naming, YAML frontmatter, and collision avoidance.

Ports the filename rules from Permanent Note/_Template/Template_Memo.md
(Templater script) so pipeline-generated notes are indistinguishable
from user-created ones.

Key rules from Template_Memo.md:
  - Unsafe chars `\\ / : * ? " < > |` are replaced with a space
  - Whitespace is collapsed to a single space and stripped
  - Base name: "YYYY-MM-DD-HHmm <title>" or "YYYY-MM-DD HHmm" when empty
  - On collision, suffix -2, -3, ... is appended
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

from .confusables import strip_invisibles

_FILENAME_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')
_WHITESPACE_RE = re.compile(r"\s+")

# ext4 / APFS / NTFS all cap a single path component at 255 bytes. Stage 05
# chapter notes already keep stems ≤ 200 bytes (``synthesis.chapter``); the
# same ceiling applies here so ``YYYY-MM-DD-HHmm <title>`` plus a collision
# suffix (``-2``) and ``.md`` still fit. Without this, a YouTube-max CJK
# title (100 chars ≈ 300 UTF-8 bytes) makes ``mkdir`` / ``write_text`` raise
# ``OSError: [Errno 36] File name too long`` and aborts the video.
_MAX_PATH_COMPONENT_BYTES = 200
# ``format_video_note_base`` / ``format_playlist_folder_name`` always use a
# fixed-width ``YYYY-MM-DD-HHmm `` prefix when a title is present (16 bytes).
_DATE_TIME_TITLE_PREFIX = "YYYY-MM-DD-HHmm "
_DATE_TIME_TITLE_PREFIX_BYTES = len(_DATE_TIME_TITLE_PREFIX.encode("utf-8"))


def sanitize_title_for_filename(raw: str | None) -> str:
    """Strip invisibles, replace OS-unsafe chars with space, collapse, strip.

    Invisible / bidi / zero-width control chars (RLO override, zero-width
    joiners, BOM) are removed first via ``strip_invisibles`` so a crafted
    YouTube title can never plant a spoofed or bidi-reversed on-disk filename.
    Whitespace controls survive that pass and are collapsed to a single space
    below (preserving the ``\\t`` -> space behavior). Homoglyph *detection*
    (mixed-script) is intentionally not done here — it happens once at the
    fetch boundary (``playlist.fetch_metadata``) so this pure, widely-reused
    chokepoint never emits duplicate alerts on read/dedup scans.

    Does **not** byte-truncate: callers that build on-disk names must run
    ``limit_title_for_path_component`` (or go through ``format_*``) so resume
    needles stay aligned with what was written.
    """
    if not raw:
        return ""
    cleaned, _ = strip_invisibles(raw)
    cleaned = _FILENAME_UNSAFE_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _utf8_byte_truncate(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to ``max_bytes`` UTF-8 bytes without splitting a codepoint."""
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def limit_title_for_path_component(safe_title: str) -> str:
    """Bound a sanitized title so ``YYYY-MM-DD-HHmm <title>`` stays ≤ 200 bytes.

    Plain truncation, used for note names: two notes cut to the same stem are
    kept apart by ``resolve_unique_path``'s ``-2`` suffix. Playlist folders
    carry the playlist's identity, so they use ``playlist_title_for_path``.
    """
    budget = _MAX_PATH_COMPONENT_BYTES - _DATE_TIME_TITLE_PREFIX_BYTES
    return _utf8_byte_truncate(safe_title, budget)


# A folder title over the budget keeps its start, then "~" and 8 hex of a
# SHA-256 of the whole sanitized title (9 bytes).
_TITLE_DIGEST_SEP = "~"
_TITLE_DIGEST_HEX = 8
_PLAYLIST_TITLE_BUDGET = _MAX_PATH_COMPONENT_BYTES - _DATE_TIME_TITLE_PREFIX_BYTES
_TITLE_HEAD_BYTES = _PLAYLIST_TITLE_BUDGET - len(_TITLE_DIGEST_SEP) - _TITLE_DIGEST_HEX
# That capped form is 180-184 bytes with a non-space before "~": the head is
# cut at 175 bytes, loses up to 3 bytes of a split codepoint, then rstrip
# drops at most one space (sanitize leaves no runs). A title that fits only
# needs escaping when it has that length and ending.
_CAPPED_TITLE_MIN_BYTES = _TITLE_HEAD_BYTES - 3 - 1 + len(_TITLE_DIGEST_SEP) + _TITLE_DIGEST_HEX
_CAPPED_TITLE_TAIL_RE = re.compile(
    rf"\S{re.escape(_TITLE_DIGEST_SEP)}[0-9a-f]{{{_TITLE_DIGEST_HEX}}}\Z"
)


def _reads_as_capped_title(safe_title: str) -> bool:
    """True if a title that fits could equal another title's capped form."""
    size = len(safe_title.encode("utf-8"))
    return (
        _CAPPED_TITLE_MIN_BYTES <= size <= _PLAYLIST_TITLE_BUDGET
        and _CAPPED_TITLE_TAIL_RE.search(safe_title) is not None
    )


def playlist_title_for_path(safe_title: str) -> str:
    """The title part of a playlist folder name, within the same byte budget.

    A title that fits is returned as is. A longer one keeps as much of its
    start as fits before ``~<8 hex>``, a digest of the whole ``safe_title``,
    so two playlists sharing that start still get different folders — and a
    checkpoint cannot count one playlist's notes as the other's. A title that
    fits but reads like such a form gets one too, or a playlist named after
    another's folder would share it.
    """
    raw = safe_title.encode("utf-8")
    if len(raw) <= _PLAYLIST_TITLE_BUDGET and not _reads_as_capped_title(safe_title):
        return safe_title
    digest = hashlib.sha256(raw).hexdigest()[:_TITLE_DIGEST_HEX]
    head = _utf8_byte_truncate(safe_title, _TITLE_HEAD_BYTES)
    return f"{head}{_TITLE_DIGEST_SEP}{digest}"


def format_video_note_base(dt: datetime, title: str | None) -> str:
    """Generate base filename for a video note.

    - With title:  'YYYY-MM-DD-HHmm <title>'
    - Without:     'YYYY-MM-DD HHmm'

    The title portion is UTF-8-byte-truncated so the stem stays within
    ``_MAX_PATH_COMPONENT_BYTES`` (see that constant).
    """
    safe_title = limit_title_for_path_component(sanitize_title_for_filename(title))
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H%M")
    if safe_title:
        return f"{date_str}-{time_str} {safe_title}"
    return f"{date_str} {time_str}"


def _strip_playlist_category_prefix(raw: str | None) -> str:
    """YouTube playlist titles sometimes encode a category as `<category>/<name>`.

    The `<category>` part is a user-defined grouping label, not the actual
    playlist name, so we drop it and keep only the last non-empty segment
    when splitting on ASCII `/`. Full-width `／` (U+FF0F) is left alone
    because it is commonly used inside Japanese titles as legitimate
    punctuation and must not be split on.

    Examples:
        "2026Agent Teams/AI駆動経営" -> "AI駆動経営"
        "A/B/C"                      -> "C"
        "plain title"                 -> "plain title"
        "Agent Teams／3 人編成"       -> "Agent Teams／3 人編成"  (full-width kept)
    """
    if not raw:
        return ""
    segments = [s.strip() for s in raw.split("/")]
    segments = [s for s in segments if s]
    if not segments:
        return ""
    return segments[-1]


def format_playlist_folder_name(dt: datetime, playlist_title: str | None) -> str:
    """Generate playlist folder name: 'YYYY-MM-DD-HHmm <playlist_title>'.

    Matches the per-video note naming convention in `format_video_note_base`
    (date-time joined by a hyphen, then a space before the title) so folders
    line up cleanly with the Obsidian memo template.

    When the raw playlist title contains ASCII `/`, only the last segment is
    used as the display title — see `_strip_playlist_category_prefix`.

    The title portion goes through ``playlist_title_for_path`` so the folder
    name stays within ``_MAX_PATH_COMPONENT_BYTES`` (ext4/APFS component limit)
    without two long titles sharing one folder.
    """
    display_title = _strip_playlist_category_prefix(playlist_title)
    safe_title = playlist_title_for_path(sanitize_title_for_filename(display_title))
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H%M")
    if safe_title:
        return f"{date_str}-{time_str} {safe_title}"
    return f"{date_str}-{time_str}"


# Playlist folders are named "YYYY-MM-DD-HHmm <title>" (legacy runs omit HHmm).
# The match covers the date (and time when present); the rest is the title.
_DATED_FOLDER_PREFIX_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:-\d{4})?")


def playlist_folder_title(folder_name: str) -> str:
    """Return a playlist folder's sanitized title, stripped of its date prefix.

    The comparison-side inverse of `format_playlist_folder_name`. Sanitizing
    keeps an on-disk name comparable with a needle built from a live playlist
    title even when the folder predates the concealment defenses.

    A name without a `YYYY-MM-DD` prefix yields `""`, so comparing against a
    non-empty needle also rejects undated directories.

    Both same-day fallback paths compare with this — `resume`'s candidate
    generator and `services.checkpoint._find_learning_folder` — so the rule
    that a fallback folder must match the playlist title *exactly* lives in one
    place. A substring rule would let "ML Python" claim a same-day sibling
    "ML Python Advanced" and consume the wrong playlist's notes.

    Compare the result with ``playlist_title_needles`` — it covers both the
    capped form new folders carry and the full title of folders written
    before the byte cap.
    """
    match = _DATED_FOLDER_PREFIX_RE.match(folder_name)
    if match is None:
        return ""
    return sanitize_title_for_filename(folder_name[match.end() :].strip())


def playlist_title_needles(playlist_title: str | None) -> frozenset[str]:
    """The ``playlist_folder_title`` values that belong to this playlist.

    Two forms, equal when the title fits the budget: what
    ``format_playlist_folder_name`` writes now (``playlist_title_for_path``),
    and the full sanitized title that folders written before the byte cap
    still carry (up to 255 - 16 = 239 bytes). Matching either one exactly
    finds both, and a playlist that merely shares a long title's start matches
    neither. A title that reads like a capped form keeps only the escaped
    form: a folder bearing the title itself may be a long title's capped
    folder, and the name cannot tell which, so its pre-cap folders are given
    up (a rerun) rather than risk consuming another playlist's notes. Empty
    when the title sanitizes to nothing: callers must then accept no fallback
    folder.
    """
    full = sanitize_title_for_filename(_strip_playlist_category_prefix(playlist_title))
    if not full:
        return frozenset()
    if _reads_as_capped_title(full):
        return frozenset({playlist_title_for_path(full)})
    return frozenset({full, playlist_title_for_path(full)})


def resolve_unique_path(folder: Path, base_name: str, ext: str = ".md") -> Path:
    """Find an unused file path under `folder`; append -2, -3, ... on collision."""
    candidate = folder / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = folder / f"{base_name}-{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _escape_yaml(s: str | None) -> str:
    """YAML-safe escape, matching pipeline/storage.ts escapeFrontmatter."""
    if not s:
        return ""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", "")
        .replace("---", "\\-\\-\\-")
    )


# Allowlist of `extra` keys accepted by `build_frontmatter`. Every
# caller in this codebase uses only these keys; locking them down
# prevents future refactors from accidentally forwarding attacker-
# controlled keys (e.g. a hypothetical `extra[user_input] = ...`).
_ALLOWED_EXTRA_KEYS = frozenset(
    {
        "playlist",
        "video_id",
        "reviewed",
        "one_liner",
        "chapter",
        "category",
        "sources",
    }
)


def build_frontmatter(
    dt: datetime,
    title: str | None,
    url: str = "",
    tags: list[str] | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    """Build YAML frontmatter matching Template_Memo.md output format.

    `extra` keys must be in `_ALLOWED_EXTRA_KEYS`; unknown keys raise
    ValueError so no caller can silently forward attacker-controlled
    keys into the YAML block.
    """
    tags = tags if tags is not None else ["memo", "youtube"]
    extra = extra or {}
    unknown = set(extra) - _ALLOWED_EXTRA_KEYS
    if unknown:
        raise ValueError(
            f"build_frontmatter: disallowed `extra` keys {sorted(unknown)!r}; "
            f"allowed: {sorted(_ALLOWED_EXTRA_KEYS)!r}"
        )
    date_str = dt.strftime("%Y-%m-%d %H:%M")

    # Strip invisibles from the visible title too: a zero-width / bidi char in
    # the YAML `title:` value is pure concealment (no display purpose) and would
    # otherwise ride along into the note's metadata. Legitimate visible
    # typography (dashes, curly quotes, CJK) is preserved.
    safe_title = strip_invisibles(title)[0] if title else title
    lines = [
        "---",
        f"date: {date_str}",
        f'title: "{_escape_yaml(safe_title)}"',
        f'URL: "{_escape_yaml(url)}"',
    ]
    for key, val in extra.items():
        # `extra` values (notably `playlist` = the raw playlist title) are also
        # attacker-controlled external text, so strip concealment chars before
        # serializing — otherwise they ride into the YAML via this loop even
        # though the `title` field above is cleaned.
        safe_val = strip_invisibles(str(val))[0]
        lines.append(f'{key}: "{_escape_yaml(safe_val)}"')
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")
    lines.append("")  # trailing newline after the closing ---
    return "\n".join(lines)


_FRONTMATTER_FIELD_TEMPLATE = '{key}: "{value}"'

# Bound on how much of an md we read when looking for frontmatter fields.
# 500 was too small: a Stage 02 summary with a long CJK title + playlist + the
# `one_liner` upsert routinely lands the closing `---` past byte 500, and the
# reader then returned None for *every* field — including `reviewed`. Phase 3
# (`--resume-reviewed`) gates on that field, so those notes were silently
# skipped after the operator flipped `reviewed: true`. 8 KiB covers the
# worst-case title (YouTube 100 chars) + playlist + one_liner (60) with ample
# headroom while still bounding pathological files for batch scans.
_FRONTMATTER_READ_LIMIT = 8192


def read_frontmatter_field(md_path: Path, field_name: str) -> str | None:
    """Return the string value of `field_name` from the YAML frontmatter.

    Reads up to ``_FRONTMATTER_READ_LIMIT`` bytes (enough for any frontmatter
    this pipeline writes; see the constant). Matches both quoted
    (`key: "value"`) and bare (`key: value`) forms. Returns None when the
    field is absent, the closing ``---`` is beyond the limit, or the file is
    unreadable.
    """
    try:
        with md_path.open("rb") as f:
            head = f.read(_FRONTMATTER_READ_LIMIT).decode("utf-8", errors="ignore")
    except OSError:
        return None
    if not head.startswith("---"):
        return None
    end = head.find("\n---", 3)
    if end == -1:
        return None
    block = head[:end]
    pattern = re.compile(rf'^{re.escape(field_name)}:\s*(?:"([^"]*)"|(\S.*))\s*$', re.MULTILINE)
    m = pattern.search(block)
    if not m:
        return None
    return (m.group(1) if m.group(1) is not None else m.group(2)).strip()


def upsert_frontmatter_field(md_text: str, key: str, value: str) -> str:
    """Insert or update `key: "value"` inside the leading `---` frontmatter.

    If the text lacks a frontmatter block, the input is returned unchanged.
    """
    if not md_text.startswith("---"):
        return md_text
    end = md_text.find("\n---", 3)
    if end == -1:
        return md_text
    head = md_text[: end + 1]
    tail = md_text[end + 1 :]
    line = _FRONTMATTER_FIELD_TEMPLATE.format(key=key, value=_escape_yaml(value))
    existing = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    if existing.search(head):
        head = existing.sub(line, head, count=1)
    else:
        head = head.rstrip("\n") + "\n" + line + "\n"
    return head + tail
