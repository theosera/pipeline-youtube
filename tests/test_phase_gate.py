"""Tests for WS5: 3-phase separation via `--resume-reviewed`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.main import _filter_to_reviewed, _find_summary_md
from pipeline_youtube.playlist import VideoMeta


def _vid(video_id: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=f"title {video_id}",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="ch",
        upload_date=None,
        playlist_title="testlist",
    )


def _write_summary(path: Path, video_id: str, reviewed: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\nreviewed: "{reviewed}"\n---\n\nbody\n',
        encoding="utf-8",
    )


class TestFindSummaryMd:
    def test_canonical_folder(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("pipeline_youtube.main.get_vault_root", lambda: tmp_path, raising=False)
        from pipeline_youtube import config
        from pipeline_youtube import main as main_mod

        config.set_vault_root(tmp_path)

        dt = datetime(2026, 4, 18, 8, 0)
        canonical = (
            f"{main_mod.LEARNING_BASE}/{main_mod.UNIT_DIRS['summary']}/2026-04-18-0800 testlist"
        )
        summary = tmp_path / canonical / "note.md"
        _write_summary(summary, "vid1", "true")

        found = _find_summary_md("vid1", "testlist", dt)
        assert found == summary

    def test_missing_returns_none(self, tmp_path: Path):
        from pipeline_youtube import config

        config.set_vault_root(tmp_path)
        assert _find_summary_md("missing", "testlist", datetime(2026, 4, 18)) is None


class TestFilterToReviewed:
    @pytest.fixture
    def vault(self, tmp_path: Path):
        from pipeline_youtube import config
        from pipeline_youtube import main as main_mod

        config.set_vault_root(tmp_path)
        dt = datetime(2026, 4, 18, 8, 0)
        folder = (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["summary"]
            / "2026-04-18-0800 testlist"
        )
        _write_summary(folder / "a.md", "vid_a", "true")
        _write_summary(folder / "b.md", "vid_b", "false")
        _write_summary(folder / "c.md", "vid_c", "true")
        return dt

    def test_keeps_only_reviewed_true(self, vault):
        to_process = [(1, _vid("vid_a")), (2, _vid("vid_b")), (3, _vid("vid_c"))]
        kept = _filter_to_reviewed(to_process, "testlist", vault)
        assert [v.video_id for _, v in kept] == ["vid_a", "vid_c"]

    def test_videos_without_summary_are_skipped(self, vault):
        to_process = [(1, _vid("unknown"))]
        kept = _filter_to_reviewed(to_process, "testlist", vault)
        assert kept == []

    def test_case_insensitive_true(self, tmp_path: Path, monkeypatch):
        from pipeline_youtube import config
        from pipeline_youtube import main as main_mod

        config.set_vault_root(tmp_path)
        dt = datetime(2026, 4, 18, 8, 0)
        folder = (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["summary"]
            / "2026-04-18-0800 testlist"
        )
        _write_summary(folder / "a.md", "vid_a", "TRUE")
        kept = _filter_to_reviewed([(1, _vid("vid_a"))], "testlist", dt)
        assert len(kept) == 1
