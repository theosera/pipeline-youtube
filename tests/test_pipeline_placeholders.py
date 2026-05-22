"""Tests for pipeline.py placeholder creation and path computation.

Locks the behavior that 04_Learning_Material is NOT pre-created as an
empty file, to prevent Templater folder-template hijacking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import (
    DEFAULT_PLACEHOLDER_UNITS,
    UNIT_DIRS,
    compute_note_paths,
    create_placeholder_notes,
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

    def test_concurrent_same_title_placeholders_get_distinct_paths(self, vault, monkeypatch):
        """Same-title videos must not overwrite each other's placeholders."""
        from pipeline_youtube import pipeline as pipeline_mod

        # Old check-then-write behavior would trust this stale answer in both
        # threads and write the same file twice. The production code must still
        # allocate distinct paths via exclusive file creation.
        monkeypatch.setattr(
            pipeline_mod,
            "resolve_unique_path",
            lambda folder, base_name, ext=".md": folder / f"{base_name}{ext}",
        )

        run_time = datetime(2026, 4, 15, 21, 23)
        videos = [
            VideoMeta(
                video_id=f"duplicate{i:02d}",
                title="Same Title",
                url=f"https://www.youtube.com/watch?v=duplicate{i:02d}",
                duration=60,
                channel="テストチャンネル",
                upload_date="20260414",
                playlist_title="Harness Engineering",
            )
            for i in range(2)
        ]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda v: create_placeholder_notes(v, run_time, units=("scripts",)),
                    videos,
                )
            )

        paths = [result["scripts"] for result in results]
        assert len(set(paths)) == 2
        assert sorted(path.name for path in paths) == [
            "2026-04-15-2123 Same Title-2.md",
            "2026-04-15-2123 Same Title.md",
        ]
        written = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        assert 'video_id: "duplicate00"' in written
        assert 'video_id: "duplicate01"' in written


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
