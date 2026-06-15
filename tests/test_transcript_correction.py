"""Tests for Stage 01b transcript correction (transcript/correction.py)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.claude_cli import ClaudeCliError, ClaudeResponse
from pipeline_youtube.transcript.chunking import Chunk
from pipeline_youtube.transcript.correction import (
    _parse_corrections,
    chunks_to_snippets,
    correct_chunks,
    render_correction_report,
)


def _response(text: str) -> ClaudeResponse:
    return ClaudeResponse(text=text, model="opus")


def _stub_invoke(text: str):
    calls: list[dict[str, Any]] = []

    def invoke(**kwargs: Any) -> ClaudeResponse:
        calls.append(kwargs)
        return _response(text)

    invoke.calls = calls  # type: ignore[attr-defined]
    return invoke


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="dQw4w9WgXcQ",
        title="Test",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        duration=60,
        channel="c",
        upload_date=None,
        playlist_title=None,
    )


class TestParseCorrections:
    def test_plain_json(self) -> None:
        out = _parse_corrections('[{"idx": 0, "text": "直した"}, {"idx": 1, "text": "B"}]')
        assert out[0].text == "直した"
        assert out[1].text == "B"

    def test_note_and_sources(self) -> None:
        out = _parse_corrections(
            '[{"idx": 0, "text": "Google", "note": "誤変換", "sources": ["https://x"]}]'
        )
        assert out[0].note == "誤変換"
        assert out[0].sources == ("https://x",)

    def test_strips_code_fence(self) -> None:
        assert _parse_corrections('```json\n[{"idx": 2, "text": "X"}]\n```')[2].text == "X"

    def test_non_array_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON array"):
            _parse_corrections('{"idx": 0, "text": "a"}')

    def test_missing_keys_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_corrections('[{"idx": 0}]')

    def test_bad_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_corrections("not json")


class TestCorrectChunks:
    def _chunks(self) -> list[Chunk]:
        return [Chunk(start=0.0, text="ぐぐる"), Chunk(start=30.0, text="てんさーふろー")]

    def test_applies_corrections_and_preserves_timestamps(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}, {"idx": 1, "text": "TensorFlow"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out.chunks] == ["Google", "TensorFlow"]
        assert [c.start for c in out.chunks] == [0.0, 30.0]

    def test_records_audit_entries(self) -> None:
        invoke = _stub_invoke(
            '[{"idx": 0, "text": "Google", "note": "ぐぐる→Google", "sources": ["https://g"]},'
            ' {"idx": 1, "text": "TensorFlow"}]'
        )
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert len(out.entries) == 2
        e0 = out.entries[0]
        assert (e0.mmss, e0.before, e0.after, e0.note) == (
            "00:00",
            "ぐぐる",
            "Google",
            "ぐぐる→Google",
        )
        assert e0.sources == ("https://g",)

    def test_unchanged_text_makes_no_entry(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "ぐぐる"}, {"idx": 1, "text": "TensorFlow"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [e.before for e in out.entries] == ["てんさーふろー"]  # only the changed one

    def test_enables_web_search_tool(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}, {"idx": 1, "text": "x"}]')
        correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert invoke.calls[0]["allowed_tools"] == ["WebSearch"]  # type: ignore[attr-defined]
        assert invoke.calls[0]["model"] == "opus"  # type: ignore[attr-defined]

    def test_missing_index_keeps_original(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert out.chunks[0].text == "Google"
        assert out.chunks[1].text == "てんさーふろー"

    def test_bad_json_falls_back_to_original(self) -> None:
        invoke = _stub_invoke("the model rambled instead of returning JSON")
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out.chunks] == ["ぐぐる", "てんさーふろー"]
        assert out.entries == []

    def test_llm_error_falls_back_to_original(self) -> None:
        def invoke(**kwargs: Any) -> ClaudeResponse:
            raise ClaudeCliError("boom")

        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out.chunks] == ["ぐぐる", "てんさーふろー"]

    def test_empty_input(self) -> None:
        out = correct_chunks([], model="opus", invoke=_stub_invoke("[]"))
        assert out.chunks == []
        assert out.entries == []

    def test_chunks_to_snippets_preserves_timeline(self) -> None:
        chunks = [
            Chunk(start=0.0, text="A"),
            Chunk(start=30.0, text="B"),
            Chunk(start=70.0, text="C"),
        ]
        snippets = chunks_to_snippets(chunks, last_end=95.0)
        assert [s.text for s in snippets] == ["A", "B", "C"]
        assert [s.start for s in snippets] == [0.0, 30.0, 70.0]
        assert [s.duration for s in snippets] == [30.0, 40.0, 25.0]

    def test_batching_splits_calls(self) -> None:
        chunks = [Chunk(start=float(i), text=str(i)) for i in range(5)]

        def invoke(**kwargs: Any) -> ClaudeResponse:
            prompt = kwargs["prompt"]
            idxs = [int(line[1 : line.index("]")]) for line in prompt.splitlines()]
            return _response(json.dumps([{"idx": i, "text": "ok"} for i in idxs]))

        out = correct_chunks(chunks, model="opus", invoke=invoke, batch_size=2)
        assert all(c.text == "ok" for c in out.chunks)
        assert [c.start for c in out.chunks] == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestRenderCorrectionReport:
    def test_report_lists_changes_with_reasons_and_sources(self) -> None:
        invoke = _stub_invoke(
            '[{"idx": 0, "text": "Google", "note": "ぐぐる→Google", "sources": ["https://g"]},'
            ' {"idx": 1, "text": "TensorFlow", "note": "誤変換", "sources": []}]'
        )
        chunks = [Chunk(start=0.0, text="ぐぐる"), Chunk(start=30.0, text="てんさーふろー")]
        result = correct_chunks(chunks, model="opus", invoke=invoke)
        md = render_correction_report(_video(), result.entries)
        assert "[00:00]" in md and "[00:30]" in md
        assert "before: ぐぐる" in md and "after: Google" in md
        assert "ぐぐる→Google" in md
        # Exact-line match (not a URL substring check, which trips CodeQL).
        assert any(line.strip() == "- https://g" for line in md.splitlines())
