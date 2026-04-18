"""Tests for checkpoint.py — video completion detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.checkpoint import (
    get_completed_video_ids,
    is_video_complete,
)
from pipeline_youtube.config import reset_vault_root, set_vault_root


@pytest.fixture()
def vault(tmp_path: Path):
    """Set up a vault with a 04_Learning_Material playlist folder."""
    set_vault_root(tmp_path)
    yield tmp_path
    reset_vault_root()


def _create_04_md(vault: Path, folder_name: str, video_id: str, title: str = "test") -> Path:
    """Create a minimal 04 md with video_id frontmatter."""
    folder = vault / "Permanent Note" / "08_YouTube学習" / "04_Learning_Material" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    md = folder / f"2026-04-16-0914 {title}.md"
    md.write_text(
        f'---\ndate: 2026-04-16 09:14\ntitle: "{title}"\n'
        f'video_id: "{video_id}"\ntags: [memo, youtube]\n---\n\nBody.\n',
        encoding="utf-8",
    )
    return md


class TestIsVideoComplete:
    def test_no_folder_returns_false(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        assert is_video_complete("abc123", "AI駆動経営", dt) is False

    def test_existing_video_returns_true(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16-0914 AI駆動経営", "abc123", "テスト動画")
        assert is_video_complete("abc123", "AI駆動経営", dt) is True

    def test_different_video_id_returns_false(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16-0914 AI駆動経営", "abc123")
        assert is_video_complete("xyz789", "AI駆動経営", dt) is False

    def test_legacy_folder_name_fallback(self, vault):
        """Should find videos in legacy folder (no HHmm, old title format)."""
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16 AI駆動経営", "abc123")
        assert is_video_complete("abc123", "AI駆動経営", dt) is True

    def test_slash_playlist_title(self, vault):
        """Playlist title with `/` — display title is last segment."""
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16-0914 AI駆動経営", "abc123")
        assert is_video_complete("abc123", "2026Agent Teams/AI駆動経営", dt) is True

    def test_multiple_videos(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        folder_name = "2026-04-16-0914 AI駆動経営"
        _create_04_md(vault, folder_name, "vid1", "動画1")
        _create_04_md(vault, folder_name, "vid2", "動画2")
        assert is_video_complete("vid1", "AI駆動経営", dt) is True
        assert is_video_complete("vid2", "AI駆動経営", dt) is True
        assert is_video_complete("vid3", "AI駆動経営", dt) is False


class TestGetCompletedVideoIds:
    def test_empty_folder(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        assert get_completed_video_ids("AI駆動経営", dt) == set()

    def test_collects_all_ids(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        folder_name = "2026-04-16-0914 AI駆動経営"
        _create_04_md(vault, folder_name, "vid1", "動画1")
        _create_04_md(vault, folder_name, "vid2", "動画2")
        _create_04_md(vault, folder_name, "vid3", "動画3")
        ids = get_completed_video_ids("AI駆動経営", dt)
        assert ids == {"vid1", "vid2", "vid3"}

    def test_no_folder_returns_empty_set(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        assert get_completed_video_ids("nonexistent", dt) == set()
