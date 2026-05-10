"""Tests for pipeline.py placeholder creation and path computation.

Locks the behavior that 04_Learning_Material is NOT pre-created as an
empty file, to prevent Templater folder-template hijacking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import (
    DEFAULT_PLACEHOLDER_UNITS,
    UNIT_DIRS,
    compute_note_paths,
    create_placeholder_notes,
    reserve_note_paths,
)
from pipeline_youtube.playlist import VideoMeta


@pytest.fixture
def vault(tmp_path: Path):
    config.set_vault_root(tmp_path)
    config.set_dry_run(False)
    yield config.get_vault_root()
    config.reset_vault_root()


def _video():
    return VideoMeta(
        video_id="_h3decBW12Q",
        title="Anthropicが公開したハーネス設計、全部解説します",
        url="https://www.youtube.com/watch?v=_h3decBW12Q",
        duration=945,
        channel="テストチャンネル",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


def _video_with(video_id: str, title: str):
    return VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=945,
        channel="テストチャンネル",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


class TestDefaultPlaceholders:
    def test_default_creates_only_three_units(self, vault):
        """04 learning placeholder must NOT be created by default.

        Rationale: Templater folder-templates hijack empty md files.
        See pipeline.py module docstring for full context.
        """
        assert DEFAULT_PLACEHOLDER_UNITS == ("scripts", "summary", "capture")
        assert "learning" not in DEFAULT_PLACEHOLDER_UNITS

    def test_default_does_not_create_04_file(self, vault):
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)

        paths = create_placeholder_notes(video, run_time)

        # Only 3 units in the returned dict
        assert set(paths.keys()) == {"scripts", "summary", "capture"}
        # And on disk: 04 file must not exist
        playlist_folder = "2026-04-15 Harness Engineering"
        note_name = "2026-04-15-2123 Anthropicが公開したハーネス設計、全部解説します.md"
        ghost_04 = (
            vault
            / "Permanent Note/08_YouTube学習/04_Learning_Material"
            / playlist_folder
            / note_name
        )
        assert not ghost_04.exists(), (
            "04 placeholder must not be pre-created — Templater would hijack it"
        )

    def test_explicit_units_tuple_can_include_learning(self, vault):
        """Legacy behavior: if a caller explicitly requests all 4, allow it."""
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)

        paths = create_placeholder_notes(
            video,
            run_time,
            units=("scripts", "summary", "capture", "learning"),
        )

        assert set(paths.keys()) == {"scripts", "summary", "capture", "learning"}
        assert paths["learning"].exists()

    def test_unknown_unit_raises(self, vault):
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)
        with pytest.raises(ValueError, match="unknown unit key"):
            create_placeholder_notes(video, run_time, units=("bogus",))  # type: ignore[arg-type]


class TestComputeNotePaths:
    def test_default_returns_all_four_paths_without_writing(self, vault):
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)

        paths = compute_note_paths(video, run_time)

        assert set(paths.keys()) == {"scripts", "summary", "capture", "learning"}
        # None of them should be written
        for p in paths.values():
            assert not p.exists()

    def test_subset_units(self, vault):
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)

        paths = compute_note_paths(video, run_time, units=("learning",))

        assert set(paths.keys()) == {"learning"}
        assert "04_Learning_Material" in str(paths["learning"])
        assert not paths["learning"].exists()

    def test_collision_resolution(self, vault):
        """If a 04 md already exists, compute_note_paths returns a -2 suffix."""
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)

        # Pre-create a colliding file
        first_paths = compute_note_paths(video, run_time, units=("learning",))
        first_paths["learning"].parent.mkdir(parents=True, exist_ok=True)
        first_paths["learning"].write_text("existing", encoding="utf-8")

        # Compute again — should yield -2 suffix
        second_paths = compute_note_paths(video, run_time, units=("learning",))
        assert second_paths["learning"] != first_paths["learning"]
        assert "-2.md" in str(second_paths["learning"])

    def test_unit_dirs_still_has_learning(self, vault):
        """UNIT_DIRS must retain 'learning' so stage 04 can look it up."""
        assert "learning" in UNIT_DIRS
        assert UNIT_DIRS["learning"] == "04_Learning_Material"


class TestReservedNotePaths:
    def test_duplicate_titles_reserve_distinct_learning_paths_before_04_exists(self, vault):
        """Concurrent videos with the same title must not share the future 04 path."""
        run_time = datetime(2026, 4, 15, 21, 23)
        first = _video_with("aaaaaaaaaaa", "Same Title")
        second = _video_with("bbbbbbbbbbb", "Same Title")

        first_paths = reserve_note_paths(first, run_time)
        create_placeholder_notes(first, run_time, precomputed_paths=first_paths)

        # Stage 04 deliberately has not written yet, but its name is reserved.
        assert not first_paths["learning"].exists()

        second_paths = reserve_note_paths(second, run_time)
        create_placeholder_notes(second, run_time, precomputed_paths=second_paths)

        assert second_paths["learning"] != first_paths["learning"]
        assert second_paths["learning"].name.endswith("-2.md")
        assert not second_paths["learning"].exists()

        for unit in ("scripts", "summary", "capture"):
            assert first_paths[unit].exists()
            assert second_paths[unit].exists()
            assert second_paths[unit] != first_paths[unit]

    def test_precomputed_placeholder_creation_refuses_existing_file(self, vault):
        """A stale or externally claimed reserved path must fail instead of overwrite."""
        video = _video()
        run_time = datetime(2026, 4, 15, 21, 23)
        paths = reserve_note_paths(video, run_time)
        paths["scripts"].parent.mkdir(parents=True, exist_ok=True)
        paths["scripts"].write_text("existing", encoding="utf-8")

        with pytest.raises(FileExistsError):
            create_placeholder_notes(video, run_time, units=("scripts",), precomputed_paths=paths)

        assert paths["scripts"].read_text(encoding="utf-8") == "existing"
