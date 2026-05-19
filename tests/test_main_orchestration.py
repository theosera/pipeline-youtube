from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import main as main_mod
from pipeline_youtube.config import reset_vault_root, set_vault_root
from pipeline_youtube.playlist import VideoMeta


def _video(video_id: str, title: str = "Video", playlist_title: str = "Playlist") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=120,
        channel="Test",
        upload_date="20260416",
        playlist_title=playlist_title,
    )


def _write_learning_note(folder: Path, video_id: str, title: str, body: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{title}.md").write_text(
        "\n".join(
            [
                "---",
                "date: 2026-04-16 09:14",
                f'title: "{title}"',
                f'URL: "https://www.youtube.com/watch?v={video_id}"',
                f'video_id: "{video_id}"',
                "tags: [memo, youtube]",
                "---",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def vault(tmp_path: Path):
    set_vault_root(tmp_path)
    yield tmp_path
    reset_vault_root()


def test_successful_results_are_restored_to_playlist_order():
    videos = [_video(f"vid{i:08d}") for i in range(1, 6)]
    results = [
        main_mod.VideoRunResult(video=videos[0], learning_md_body="body 1"),
        main_mod.VideoRunResult(video=videos[2], learning_md_body="body 3"),
        main_mod.VideoRunResult(video=videos[4], learning_md_body="body 5"),
        main_mod.VideoRunResult(video=videos[1], learning_md_body="body 2"),
        main_mod.VideoRunResult(video=videos[3], learning_md_body="body 4"),
    ]

    ordered = main_mod._successful_results_in_playlist_order(videos, results)

    assert [result.video.video_id for result in ordered] == [video.video_id for video in videos]
    assert [result.learning_md_body for result in ordered] == [
        "body 1",
        "body 2",
        "body 3",
        "body 4",
        "body 5",
    ]


def test_synthesis_only_fallback_strips_slash_playlist_category(vault: Path):
    run_time = datetime(2026, 4, 16, 10, 5)
    playlist_title = "2026Agent Teams/AI駆動経営"
    videos = [
        _video("abc123DEFGH", "Video 1", playlist_title),
        _video("xyz789IJKLM", "Video 2", playlist_title),
    ]
    learning_dir = (
        vault
        / "Permanent Note"
        / "08_YouTube学習"
        / "04_Learning_Material"
        / "2026-04-16-0914 AI駆動経営"
    )
    _write_learning_note(learning_dir, "abc123DEFGH", "Video 1", "Body 1.")
    _write_learning_note(learning_dir, "xyz789IJKLM", "Video 2", "Body 2.")

    matched_videos, matched_bodies, folder_name = main_mod._collect_existing_learning_bodies(
        videos, playlist_title, run_time
    )

    assert [video.video_id for video in matched_videos] == ["abc123DEFGH", "xyz789IJKLM"]
    assert matched_bodies == ["Body 1.\n", "Body 2.\n"]
    assert folder_name == "2026-04-16-0914 AI駆動経営"
