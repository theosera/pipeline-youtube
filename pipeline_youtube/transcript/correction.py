"""Stage 01b: LLM + web-search correction of a fast, low-accuracy transcript.

Stage 01a produces a transcript quickly with no LLM (YouTube auto-captions, or
fast Whisper for ``--local-media``), accepting ASR/caption errors. This module
repairs those errors — especially misrecognized proper nouns and technical
terms — with an LLM (default Opus) that can **autonomously web-search** to
fact-check uncertain terms.

Timestamps are preserved by construction: the model is given numbered chunks
and must return a 1:1 JSON correction keyed by the chunk index. We re-attach the
original chunk ``start`` to each corrected text, so the model can never move a
timestamp. Anything that doesn't round-trip cleanly (bad JSON, missing index)
falls back to the original chunk — correction is best-effort and must never
break Stage 01 or shift the timeline that Stage 02/03 depend on.

Alongside the corrected text, the model returns a short ``note`` (why it
changed a line) and the ``sources`` (URLs it consulted via web search). These
are surfaced as a sibling ``… — corrections.md`` audit report so a human can see
*why* each correction was made (see ``render_correction_report``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ..playlist import VideoMeta
from ..providers.claude_cli import ClaudeCliError, ClaudeResponse, invoke_claude
from .base import TranscriptSnippet
from .chunking import Chunk

# How many chunks to correct per LLM call. Long videos produce hundreds of
# 30s chunks; batching keeps each request bounded and localizes failures (a
# bad batch falls back to raw text without poisoning the rest).
DEFAULT_BATCH_SIZE = 40

# Web-search-backed Opus calls are slower than plain generation; give them
# more headroom than the default 600s.
DEFAULT_TIMEOUT = 1200

CORRECTION_SYSTEM_PROMPT = (
    "あなたは音声認識・字幕の誤りを直す校正者です。各行は YouTube の粗い文字起こしの"
    "1チャンクで、`[idx] (MM:SS) text` 形式です。次の規則で **text のみ** を校正してください。\n"
    "- 前後の文脈から、誤変換・脱字・不自然な日本語を、話者の本来の意図を保ったまま正す。\n"
    "- 固有名詞・専門用語・製品名などに少しでも不確かさがあれば、**WebSearch ツールで"
    "事実確認**してから正しい表記に直す（推測で確定しない）。\n"
    "- 要約・言い換え・情報の追加や削除はしない。意味を保った最小限の校正に留める。\n"
    "- 文脈推論でも検索でも判別不能な深刻な欠落のみ、捏造せず `[聴取不能]` とする。\n"
    "- 行の統合・分割・並べ替え・idx や時刻の改変は禁止。入力の idx と1:1で対応させる。\n"
    "\n"
    "出力は **JSON 配列のみ**（前置き・コードフェンス・説明文を一切付けない）。"
    'スキーマ: [{"idx": <int>, "text": "<校正後テキスト>", "note": "<訂正理由 or 空文字>", '
    '"sources": ["<参照URL>", ...]}, ...]。\n'
    "- text を変更した行のみ、**なぜそう直したか**を `note` に簡潔に書き、web 検索で参照した"
    "URL を `sources` に列挙する（検索していなければ空配列）。\n"
    "- 変更しなかった行は `note` を空文字、`sources` を空配列にする。\n"
    "- 入力の各 idx をちょうど1回ずつ含めること。"
)

# An invoke callable matching `invoke_claude`'s keyword interface — injectable
# so tests can stub the LLM without touching the network.
InvokeFn = Callable[..., ClaudeResponse]


@dataclass(frozen=True)
class CorrectionEntry:
    """One audited correction: what changed at ``mmss`` and why."""

    mmss: str
    before: str
    after: str
    note: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CorrectionResult:
    """Corrected chunks plus the per-change audit entries."""

    chunks: list[Chunk]
    entries: list[CorrectionEntry]


def _build_prompt(batch: list[tuple[int, Chunk]]) -> str:
    """Render a batch of (index, chunk) as numbered `[idx] (MM:SS) text` lines."""
    return "\n".join(f"[{idx}] ({chunk.mmss}) {chunk.text}" for idx, chunk in batch)


def _strip_code_fence(text: str) -> str:
    """Drop a leading/trailing markdown code fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


@dataclass(frozen=True)
class _Correction:
    text: str
    note: str
    sources: tuple[str, ...]


def _parse_corrections(text: str) -> dict[int, _Correction]:
    """Parse the model's JSON array into ``{idx: _Correction}``.

    Raises ``ValueError`` if the payload is not a JSON array of objects with at
    least ``idx`` (int) and ``text`` (str), so the caller can fall back to the
    raw chunks for this batch. ``note``/``sources`` are optional.
    """
    payload = json.loads(_strip_code_fence(text))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array, got {type(payload).__name__}")
    mapping: dict[int, _Correction] = {}
    for item in payload:
        if not isinstance(item, dict) or "idx" not in item or "text" not in item:
            raise ValueError("each item must be an object with 'idx' and 'text'")
        idx = item["idx"]
        corrected = item["text"]
        if not isinstance(idx, int) or not isinstance(corrected, str):
            raise ValueError("'idx' must be int and 'text' must be str")
        note = item.get("note")
        note = note if isinstance(note, str) else ""
        raw_sources = item.get("sources")
        sources = (
            tuple(s for s in raw_sources if isinstance(s, str))
            if isinstance(raw_sources, list)
            else ()
        )
        mapping[idx] = _Correction(text=corrected, note=note, sources=sources)
    return mapping


def correct_chunks(
    chunks: list[Chunk],
    *,
    model: str,
    invoke: InvokeFn = invoke_claude,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
    allowed_tools: tuple[str, ...] = ("WebSearch",),
) -> CorrectionResult:
    """Return corrected chunks (timestamps unchanged) plus audit entries.

    Processes ``chunks`` in batches; each batch is corrected by one LLM call
    with web search enabled. A batch that fails to round-trip (LLM error, bad
    JSON) is left untouched. Per-chunk: if the model returned a non-empty,
    *different* text for that index, use it and record a ``CorrectionEntry``
    (before/after/note/sources); otherwise keep the original text. The
    ``start`` of every chunk is preserved verbatim.
    """
    if not chunks:
        return CorrectionResult(chunks=chunks, entries=[])

    corrected: list[Chunk] = list(chunks)
    entries: list[CorrectionEntry] = []
    for batch_start in range(0, len(chunks), batch_size):
        batch = [
            (i, chunks[i]) for i in range(batch_start, min(batch_start + batch_size, len(chunks)))
        ]
        try:
            response = invoke(
                prompt=_build_prompt(batch),
                append_system_prompt=CORRECTION_SYSTEM_PROMPT,
                model=model,
                allowed_tools=list(allowed_tools),
                timeout=timeout,
            )
            mapping = _parse_corrections(response.text)
        except (ClaudeCliError, ValueError, json.JSONDecodeError):
            # Best-effort: a failed batch keeps its raw chunks rather than
            # breaking Stage 01 or shifting the timeline.
            continue
        for idx, chunk in batch:
            correction = mapping.get(idx)
            if correction is None or not correction.text or correction.text == chunk.text:
                continue
            corrected[idx] = Chunk(start=chunk.start, text=correction.text)
            entries.append(
                CorrectionEntry(
                    mmss=chunk.mmss,
                    before=chunk.text,
                    after=correction.text,
                    note=correction.note,
                    sources=correction.sources,
                )
            )
    return CorrectionResult(chunks=corrected, entries=entries)


def render_correction_report(video: VideoMeta, entries: list[CorrectionEntry]) -> str:
    """Render the per-change audit report (the sibling ``… — corrections.md``).

    Lists each changed chunk with before→after, the model's reason, and the web
    sources it consulted — so a human can see *why* the transcript was changed.
    """
    lines: list[str] = [
        f"# 訂正レポート — {video.title}",
        "",
        f"Stage 01b の誤変換訂正で変更した {len(entries)} 箇所の根拠（before→after / 理由 / 参照）。",
        "",
    ]
    for e in entries:
        lines.append(f"## [{e.mmss}]")
        lines.append(f"- before: {e.before}")
        lines.append(f"- after: {e.after}")
        if e.note:
            lines.append(f"- 理由: {e.note}")
        if e.sources:
            lines.append("- 参照:")
            lines.extend(f"  - {s}" for s in e.sources)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def chunks_to_snippets(chunks: list[Chunk], *, last_end: float) -> list[TranscriptSnippet]:
    """Turn corrected chunks back into transcript snippets for downstream stages.

    The corrected text must flow into the ``TranscriptResult`` that Stage 02
    (and thus 03/04) consumes — not just the rendered 01 markdown. Each chunk
    becomes one snippet; ``start`` is preserved and ``duration`` spans to the
    next chunk (the last chunk runs to ``last_end``, the original transcript's
    end), so the timeline is unchanged.
    """
    snippets: list[TranscriptSnippet] = []
    for i, chunk in enumerate(chunks):
        next_start = chunks[i + 1].start if i + 1 < len(chunks) else last_end
        duration = max(next_start - chunk.start, 0.0)
        snippets.append(TranscriptSnippet(text=chunk.text, start=chunk.start, duration=duration))
    return snippets
