"""Tier 3 transcript fetcher: local Whisper transcription.

Downloads the audio track via yt-dlp, runs openai-whisper, and returns
a TranscriptResult with word-level timestamps. This is the last resort
when YouTube provides neither official nor auto-generated captions.

Global lock
-----------
Whisper is GPU/memory intensive. A file-based lock at
`{project_root}/tmp/.whisper.lock` ensures only one Whisper process
runs at a time across all pipeline instances. Other videos queue up
behind the lock. The lock file is NOT deleted on release so the path
stays stable.

Optional dependency
-------------------
`openai-whisper` is declared under `[project.optional-dependencies]`
(`uv sync --extra whisper`). If not installed, `fetch_whisper` raises
`TranscriptNotAvailable("whisper_not_installed")` immediately so the
fallback chain terminates gracefully.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .base import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TMP_DIR = _PROJECT_ROOT / "tmp"
_LOCK_PATH = _TMP_DIR / ".whisper.lock"

# Default whisper model — "small" balances speed and accuracy for most
# YouTube content. Override via config.json whisper_model field (future).
DEFAULT_WHISPER_MODEL = "small"


def _ensure_tmp() -> None:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)


def _download_audio(video_id: str) -> Path:
    """Download audio-only track via yt-dlp as m4a/mp3.

    Returns the path to the downloaded file inside tmp/.
    Raises TranscriptNotAvailable on download failure.
    """
    _ensure_tmp()
    out_template = str(_TMP_DIR / f"whisper_{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        import yt_dlp  # type: ignore[import-untyped]
    except ImportError as e:
        raise TranscriptNotAvailable("yt_dlp_not_installed") from e

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise TranscriptNotAvailable(f"audio_download_failed: {e}") from e

    # Find the downloaded file (extension may vary)
    candidates = sorted(_TMP_DIR.glob(f"whisper_{video_id}.*"))
    candidates = [c for c in candidates if c.suffix != ".lock"]
    if not candidates:
        raise TranscriptNotAvailable("audio_file_not_found_after_download")
    return candidates[0]


def _run_whisper(
    audio_path: Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Run openai-whisper on the audio file and return segments.

    Returns a list of segment dicts with keys: start, end, text.
    Raises TranscriptNotAvailable if whisper is not installed or fails.
    """
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError as e:
        raise TranscriptNotAvailable("whisper_not_installed") from e

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
        )
    except Exception as e:
        raise TranscriptNotAvailable(f"whisper_transcribe_failed: {e}") from e

    return result.get("segments", [])


def _segments_to_snippets(segments: list[dict[str, Any]]) -> list[TranscriptSnippet]:
    """Convert whisper segments to TranscriptSnippet list."""
    snippets: list[TranscriptSnippet] = []
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if text:
            snippets.append(TranscriptSnippet(text=text, start=start, duration=end - start))
    return snippets


def _detect_language(segments: list[dict[str, Any]]) -> str | None:
    """Best-effort language detection from whisper output."""
    # Whisper segments don't carry language, but the model result does.
    # We pass language through the caller chain instead.
    return None


def fetch_whisper(
    video_id: str,
    languages: list[str],
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
) -> TranscriptResult:
    """Tier 3 fetcher: download audio + Whisper transcribe.

    Acquires a file-based global lock before running. Only one Whisper
    instance runs at a time regardless of --concurrency.

    Parameters
    ----------
    video_id:
        YouTube video ID.
    languages:
        Preferred languages. The first entry is used as Whisper's
        `language` hint. If empty, Whisper auto-detects.
    model_name:
        Whisper model size (tiny/base/small/medium/large).
    """
    # Check whisper is importable before acquiring lock
    try:
        import whisper  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise TranscriptNotAvailable("whisper_not_installed") from e

    # File-based lock — only one whisper process at a time
    try:
        import filelock  # type: ignore[import-untyped]
    except ImportError:
        # filelock not installed — fall back to no lock (still works,
        # just no cross-process protection)
        filelock = None  # type: ignore[assignment]

    _ensure_tmp()
    lock_ctx = filelock.FileLock(_LOCK_PATH, timeout=-1) if filelock else _noop_lock()

    with lock_ctx:
        audio_path: Path | None = None
        try:
            audio_path = _download_audio(video_id)
            lang_hint = languages[0] if languages else None
            segments = _run_whisper(audio_path, model_name=model_name, language=lang_hint)
            snippets = _segments_to_snippets(segments)

            if not snippets:
                raise TranscriptNotAvailable("whisper_produced_no_segments")

            return build_result(
                video_id=video_id,
                source=TranscriptSource.WHISPER,
                language=lang_hint,
                snippets=snippets,
            )
        finally:
            # Clean up audio file
            if audio_path is not None:
                with contextlib.suppress(OSError):
                    audio_path.unlink(missing_ok=True)


class _noop_lock:
    """No-op context manager when filelock is not installed."""

    def __enter__(self) -> _noop_lock:
        return self

    def __exit__(self, *args: object) -> None:
        pass
