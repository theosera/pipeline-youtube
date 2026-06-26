"""Tests for the JSONL transcript stats logger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stats import _default_stats_path, record_transcript_stat
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="abc123",
        title="Test Video",
        url="https://www.youtube.com/watch?v=abc123",
        duration=300,
        channel="Test Channel",
        upload_date="20260414",
        playlist_title="My Playlist",
    )


def _result(source: TranscriptSource, fallback_reason: str | None = None):
    return build_result(
        video_id="abc123",
        source=source,
        language="ja",
        snippets=[TranscriptSnippet(text="hi", start=0.0, duration=1.0)],
        fallback_reason=fallback_reason,
    )


class TestRecordTranscriptStat:
    def test_writes_jsonl_line(self, tmp_path: Path):
        path = tmp_path / "stats.jsonl"
        result = _result(TranscriptSource.OFFICIAL)
        record_transcript_stat(_video(), result, stats_path=path)

        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["video_id"] == "abc123"
        assert record["channel"] == "Test Channel"
        assert record["playlist_title"] == "My Playlist"
        assert record["transcript_source"] == "official"
        assert record["language"] == "ja"
        assert record["snippet_count"] == 1
        assert record["fallback_reason"] is None

    def test_appends_multiple_entries(self, tmp_path: Path):
        path = tmp_path / "stats.jsonl"
        record_transcript_stat(_video(), _result(TranscriptSource.OFFICIAL), stats_path=path)
        record_transcript_stat(
            _video(),
            _result(TranscriptSource.AUTO, "official:no_manual_transcript_in_languages"),
            stats_path=path,
        )
        record_transcript_stat(
            _video(),
            _result(TranscriptSource.ERROR, "official:disabled; auto:disabled"),
            stats_path=path,
        )

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        r2 = json.loads(lines[1])
        assert r2["transcript_source"] == "auto-generated"
        assert r2["fallback_reason"] == "official:no_manual_transcript_in_languages"

    def test_unicode_preserved(self, tmp_path: Path):
        """Japanese text must not be \\u-escaped so logs are human-readable."""
        video = VideoMeta(
            video_id="vid",
            title="日本語タイトル",
            url="https://example.com",
            duration=None,
            channel="日本語チャンネル",
            upload_date=None,
            playlist_title="日本語プレイリスト",
        )
        path = tmp_path / "stats.jsonl"
        record_transcript_stat(video, _result(TranscriptSource.OFFICIAL), stats_path=path)
        content = path.read_text(encoding="utf-8")
        assert "日本語タイトル" in content
        assert "日本語チャンネル" in content

    def test_auto_creates_parent_dir(self, tmp_path: Path):
        path = tmp_path / "nested" / "deeper" / "stats.jsonl"
        record_transcript_stat(_video(), _result(TranscriptSource.OFFICIAL), stats_path=path)
        assert path.exists()


class TestDefaultStatsPath:
    def test_default_dir_is_project_logs(self, monkeypatch: pytest.MonkeyPatch):
        # No override → the path stays under <project_root>/logs (outside any
        # vault), the existing behaviour #13 preserves.
        monkeypatch.delenv("PIPELINE_YOUTUBE_STATS_DIR", raising=False)
        path = _default_stats_path()
        assert path.parent.name == "logs"
        assert path.name.startswith("transcript_stats_")

    def test_env_override_redirects_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        target = tmp_path / "xdg-state" / "pipeline-youtube"
        monkeypatch.setenv("PIPELINE_YOUTUBE_STATS_DIR", str(target))
        path = _default_stats_path()
        assert path.parent == target

        # The default write path (no explicit stats_path) honours the override.
        written = record_transcript_stat(_video(), _result(TranscriptSource.OFFICIAL))
        assert written.parent == target
        assert written.exists()
