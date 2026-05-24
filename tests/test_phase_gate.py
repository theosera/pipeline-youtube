"""Tests for WS5: 3-phase separation via `--resume-reviewed`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.main import (
    _filter_to_reviewed,
    _find_existing_04_md,
    _find_summary_md,
    _process_video,
)
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.claude_cli import ClaudeResponse


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


def _write_capture(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\n---\n\n[00:00 ~ 00:05]\n![[capture.webp]]\n',
        encoding="utf-8",
    )


def _write_learning(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\n---\n\nlearning body\n',
        encoding="utf-8",
    )


# 11-char YouTube-shaped IDs (matches the M3 hardened extractor format).
_VID_A = "aaaaaaaaaaA"
_VID_B = "bbbbbbbbbbB"
_VID_C = "ccccccccccC"
_VID_1 = "vid1xxxxxxA"


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
        _write_summary(summary, _VID_1, "true")

        found = _find_summary_md(_VID_1, "testlist", dt)
        assert found == summary

    def test_missing_returns_none(self, tmp_path: Path):
        from pipeline_youtube import config

        config.set_vault_root(tmp_path)
        assert _find_summary_md("missingxxxx", "testlist", datetime(2026, 4, 18)) is None


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
        _write_summary(folder / "a.md", _VID_A, "true")
        _write_summary(folder / "b.md", _VID_B, "false")
        _write_summary(folder / "c.md", _VID_C, "true")
        return dt

    def test_keeps_only_reviewed_true(self, vault):
        to_process = [(1, _vid(_VID_A)), (2, _vid(_VID_B)), (3, _vid(_VID_C))]
        kept = _filter_to_reviewed(to_process, "testlist", vault)
        assert [v.video_id for _, v in kept] == [_VID_A, _VID_C]

    def test_videos_without_summary_are_skipped(self, vault):
        to_process = [(1, _vid("unknownXXXX"))]
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
        _write_summary(folder / "a.md", _VID_A, "TRUE")
        kept = _filter_to_reviewed([(1, _vid(_VID_A))], "testlist", dt)
        assert len(kept) == 1


class TestResumeReviewedProcessing:
    def test_existing_04_lookup_keeps_checkpoint_skips_in_original_folder(self, tmp_path: Path):
        from pipeline_youtube import config
        from pipeline_youtube import main as main_mod

        config.set_vault_root(tmp_path)
        dt = datetime(2026, 4, 18, 12, 0)
        phase1_folder = "2026-04-18-0800 testlist"
        learning = (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["learning"]
            / phase1_folder
            / "a.md"
        )
        _write_learning(learning, _VID_A)

        assert _find_existing_04_md(_VID_A, "testlist", dt) == learning

    def test_runs_only_stage_04_against_existing_reviewed_notes(
        self, tmp_path: Path, monkeypatch
    ):
        from pipeline_youtube import config
        from pipeline_youtube import main as main_mod

        config.set_vault_root(tmp_path)
        resume_time = datetime(2026, 4, 18, 12, 0)
        phase1_folder = "2026-04-18-0800 testlist"
        summary = (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["summary"]
            / phase1_folder
            / "a.md"
        )
        capture = (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["capture"]
            / phase1_folder
            / "a.md"
        )
        _write_summary(summary, _VID_A, "true")
        _write_capture(capture, _VID_A)

        def forbidden_stage(*args, **kwargs):
            raise AssertionError("stages 01-03 must be skipped during --resume-reviewed")

        monkeypatch.setattr(main_mod, "run_stage_scripts", forbidden_stage)
        monkeypatch.setattr(main_mod, "run_stage_summary", forbidden_stage)
        monkeypatch.setattr(main_mod, "run_stage_capture", forbidden_stage)

        def fake_learning(
            video,
            summary_md_path,
            capture_md_path,
            learning_md_path,
            **kwargs,
        ):
            assert summary_md_path == summary
            assert capture_md_path == capture
            assert kwargs["run_time"] == resume_time
            learning_md_path.parent.mkdir(parents=True, exist_ok=True)
            learning_md_path.write_text(
                f'---\nvideo_id: "{video.video_id}"\n---\n\nlearning body\n',
                encoding="utf-8",
            )
            return ClaudeResponse(
                text="learning body",
                model="sonnet",
                input_tokens=1,
                output_tokens=2,
                total_cost_usd=0.01,
            )

        monkeypatch.setattr(main_mod, "run_stage_learning", fake_learning)

        result = _process_video(
            _vid(_VID_A),
            resume_time,
            dry_run=False,
            capture_format="auto",
            models={"stage_02": "sonnet", "stage_04": "sonnet"},
            resume_reviewed=True,
            playlist_title="testlist",
        )

        assert result.ok
        assert result.learning_md_body and result.learning_md_body.strip() == "learning body"
        assert result.learning_md_path == (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["learning"]
            / phase1_folder
            / "a.md"
        )
        assert not (
            tmp_path
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["summary"]
            / "2026-04-18-1200 testlist"
        ).exists()
