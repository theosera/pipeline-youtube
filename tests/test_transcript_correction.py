"""Tests for Stage 01b transcript correction (transcript/correction.py)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline_youtube.providers.claude_cli import ClaudeCliError, ClaudeResponse
from pipeline_youtube.transcript.chunking import Chunk
from pipeline_youtube.transcript.correction import (
    _parse_corrections,
    correct_chunks,
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


class TestParseCorrections:
    def test_plain_json(self) -> None:
        out = _parse_corrections('[{"idx": 0, "text": "直した"}, {"idx": 1, "text": "B"}]')
        assert out == {0: "直した", 1: "B"}

    def test_strips_code_fence(self) -> None:
        fenced = '```json\n[{"idx": 2, "text": "X"}]\n```'
        assert _parse_corrections(fenced) == {2: "X"}

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
        assert [c.text for c in out] == ["Google", "TensorFlow"]
        assert [c.start for c in out] == [0.0, 30.0]  # unchanged

    def test_enables_web_search_tool(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}, {"idx": 1, "text": "x"}]')
        correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert invoke.calls[0]["allowed_tools"] == ["WebSearch"]  # type: ignore[attr-defined]
        assert invoke.calls[0]["model"] == "opus"  # type: ignore[attr-defined]

    def test_missing_index_keeps_original(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}]')  # idx 1 absent
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert out[0].text == "Google"
        assert out[1].text == "てんさーふろー"  # untouched

    def test_empty_correction_keeps_original(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": ""}, {"idx": 1, "text": "x"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert out[0].text == "ぐぐる"  # empty correction ignored

    def test_bad_json_falls_back_to_original(self) -> None:
        invoke = _stub_invoke("the model rambled instead of returning JSON")
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out] == ["ぐぐる", "てんさーふろー"]

    def test_llm_error_falls_back_to_original(self) -> None:
        def invoke(**kwargs: Any) -> ClaudeResponse:
            raise ClaudeCliError("boom")

        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out] == ["ぐぐる", "てんさーふろー"]

    def test_empty_input(self) -> None:
        assert correct_chunks([], model="opus", invoke=_stub_invoke("[]")) == []

    def test_batching_splits_calls(self) -> None:
        chunks = [Chunk(start=float(i), text=str(i)) for i in range(5)]

        # Each batch echoes idx→"ok"; with batch_size=2 → 3 calls.
        def invoke(**kwargs: Any) -> ClaudeResponse:
            prompt = kwargs["prompt"]
            idxs = [int(line[1 : line.index("]")]) for line in prompt.splitlines()]
            return _response(json.dumps([{"idx": i, "text": "ok"} for i in idxs]))

        out = correct_chunks(chunks, model="opus", invoke=invoke, batch_size=2)
        assert all(c.text == "ok" for c in out)
        assert [c.start for c in out] == [0.0, 1.0, 2.0, 3.0, 4.0]
