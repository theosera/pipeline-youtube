from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline_youtube import config
from pipeline_youtube.main import _collect_existing_learning_bodies
from pipeline_youtube.pipeline import LEARNING_BASE, UNIT_DIRS
from pipeline_youtube.playlist import VideoMeta


def _video(video_id: str = "abcDEF12345") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title="Video",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=900,
        channel="Test",
        upload_date="20260415",
        playlist_title="Course/AI Driven",
    )


def test_synthesis_only_fallback_strips_playlist_category_prefix(tmp_path: Path):
    config.set_vault_root(tmp_path)
    try:
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["learning"]
        learning_dir = base / "2026-04-15-0914 AI Driven"
        learning_dir.mkdir(parents=True)
        learning_dir.joinpath("video.md").write_text(
            '---\n'
            'video_id: "abcDEF12345"\n'
            'URL: "https://www.youtube.com/watch?v=abcDEF12345"\n'
            '---\n\n'
            '# Learning body\n',
            encoding="utf-8",
        )

        videos, bodies, folder_name = _collect_existing_learning_bodies(
            [_video()],
            "Course/AI Driven",
            datetime(2026, 4, 15, 10, 30),
        )

        assert [v.video_id for v in videos] == ["abcDEF12345"]
        assert bodies == ["# Learning body\n"]
        assert folder_name == "2026-04-15-0914 AI Driven"
    finally:
        config.reset_vault_root()
