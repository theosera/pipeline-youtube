"""N-second window chunking for transcript snippets.

Implements decision (2) from the plan: window-based chunking instead of
punctuation-based. The rule is simple: start a new chunk whenever the
current chunk's span from its first snippet would exceed `window_seconds`
if we added the next snippet's start time.

This preserves snippet boundaries (we never split a snippet's text) and
produces natural-looking chunks similar to the dummy data in
`Permanent Note/08_YouTube学習/01_Scripts_Processing_Unit/`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import TranscriptSnippet


@dataclass(frozen=True)
class Chunk:
    start: float  # seconds from video start
    text: str  # concatenated, whitespace-collapsed

    @property
    def mmss(self) -> str:
        """Format start as MM:SS for the markdown link label."""
        total = int(self.start)
        mm, ss = divmod(total, 60)
        return f"{mm:02d}:{ss:02d}"

    @property
    def start_int(self) -> int:
        """Integer seconds for the YouTube &t= query param."""
        return int(self.start)


def chunk_by_window(
    snippets: list[TranscriptSnippet],
    window_seconds: float = 30.0,
) -> list[Chunk]:
    """Group snippets into windows of roughly `window_seconds` each.

    Algorithm:
      - Start a new chunk when the next snippet's start time is at or
        beyond `chunk_start + window_seconds` (and the current chunk
        has at least one snippet so we don't emit empties).
      - Text is concatenated with single spaces and leading/trailing
        whitespace is stripped.

    Returns an empty list for empty input.
    """
    if not snippets:
        return []
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")

    chunks: list[Chunk] = []
    chunk_start: float = snippets[0].start
    chunk_texts: list[str] = []

    for snippet in snippets:
        if snippet.start >= chunk_start + window_seconds and chunk_texts:
            chunks.append(
                Chunk(
                    start=chunk_start,
                    text=_join_texts(chunk_texts),
                )
            )
            chunk_start = snippet.start
            chunk_texts = []
        chunk_texts.append(snippet.text)

    if chunk_texts:
        chunks.append(Chunk(start=chunk_start, text=_join_texts(chunk_texts)))

    return chunks


def _join_texts(texts: list[str]) -> str:
    """Concatenate snippet texts, strip + collapse internal whitespace."""
    parts: list[str] = []
    for t in texts:
        stripped = " ".join(t.split())
        if stripped:
            parts.append(stripped)
    return " ".join(parts)
