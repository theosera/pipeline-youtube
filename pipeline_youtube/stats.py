"""JSONL stats logger for transcript-source tracking.

Writes one line per video to `transcript_stats_YYYY-MM-DD.jsonl`. The default
output directory is `<project_root>/logs/` — outside any Obsidian vault and
git-ignored, so run metadata (channel / title / fallback reasons) never lands in
a synced or shared vault. Set the `PIPELINE_YOUTUBE_STATS_DIR` environment
variable to redirect the output elsewhere (e.g. an XDG state dir). The stats
feed into decision (1) from the plan — "which channels/genres frequently need
Whisper fallback" — so a later agent can use them to skip ahead to
Whisper for specific channels without retrying the doomed earlier tiers.

Schema per line:
    {
      "video_id": "dQw4w9WgXcQ",
      "channel": "Example Channel",
      "title": "...",
      "playlist_title": "...",
      "transcript_source": "official|auto-generated|whisper|error",
      "language": "ja",
      "retrieved_at": "2026-04-15T03:22:01+00:00",
      "fallback_reason": "official:no_manual_transcript_in_languages; ...",
      "snippet_count": 123
    }
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from .playlist import VideoMeta
from .sanitize import sanitize_untrusted_text
from .transcript.base import TranscriptResult

_MAX_FIELD_LEN = 500

# Override the stats output directory. Default keeps the path out of any vault
# (see module docstring); set this to relocate run metadata explicitly.
_STATS_DIR_ENV = "PIPELINE_YOUTUBE_STATS_DIR"


def _safe(s: str | None, context: str) -> str | None:
    """Preserve None; sanitize actual strings with context for alerts."""
    if s is None:
        return None
    return sanitize_untrusted_text(s, _MAX_FIELD_LEN, context=context)


def _default_stats_path() -> Path:
    override = os.environ.get(_STATS_DIR_ENV, "").strip()
    if override:
        logs_dir = Path(override).expanduser()
    else:
        # Project root is the parent of the `pipeline_youtube` package dir.
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"transcript_stats_{date.today().isoformat()}.jsonl"


def record_transcript_stat(
    video: VideoMeta,
    result: TranscriptResult,
    stats_path: Path | None = None,
) -> Path:
    """Append a single JSONL line describing a transcript fetch outcome.

    Untrusted string fields (channel, title, playlist_title,
    fallback_reason, error) pass through `sanitize_untrusted_text` to
    strip ANSI / control / zero-width sequences that could attack a
    terminal reading the log back (CVE-2024-22423 class).

    Returns the path actually written to (useful for tests that pass an
    explicit `stats_path`).
    """
    path = stats_path or _default_stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "video_id": video.video_id,
        "channel": _safe(video.channel, "stats.channel"),
        "title": _safe(video.title, "stats.title"),
        "playlist_title": _safe(video.playlist_title, "stats.playlist_title"),
        "transcript_source": result.source.value,
        "language": result.language,
        "retrieved_at": result.retrieved_at,
        "fallback_reason": _safe(result.fallback_reason, "stats.fallback_reason"),
        "snippet_count": len(result.snippets),
        "error": _safe(result.error, "stats.error"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path
