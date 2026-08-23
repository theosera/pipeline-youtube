"""Worker-scoped Stage 03 temp paths must not share dest across processes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stages.capture import _download_video, _tmp_video_path


def _video(video_id: str = "dQw4w9WgXcQ") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title="t",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="c",
        upload_date=None,
        playlist_title=None,
    )


class TestTmpVideoPathIdentity:
    def test_same_thread_is_stable(self):
        video = _video()
        assert _tmp_video_path(video) == _tmp_video_path(video)

    def test_same_thread_differs_by_video_id(self):
        assert _tmp_video_path(_video("aaaaaaaaaaa")) != _tmp_video_path(_video("bbbbbbbbbbb"))

    def test_pid_and_thread_slot_are_part_of_the_name(self):
        with (
            patch("pipeline_youtube.stages.capture.os.getpid", return_value=111),
            patch("pipeline_youtube.stages.capture.threading.get_ident", return_value=222),
        ):
            path = _tmp_video_path(_video("dQw4w9WgXcQ"))
        assert path.name == "dQw4w9WgXcQ-111-222.mp4"

    def test_overlapping_workers_do_not_share_a_dest(self):
        """Two CLI processes (or two threads) downloading the same video.

        Concrete trigger: playlist A and playlist B both contain video X and
        run at the same time (default ``--concurrency 1`` in each process).
        The old ``tmp/{video_id}.mp4`` dest was unlinked by whichever worker
        started second, corrupting the first worker's capture source.
        """
        video = _video()
        with (
            patch("pipeline_youtube.stages.capture.os.getpid", return_value=10),
            patch("pipeline_youtube.stages.capture.threading.get_ident", return_value=1),
        ):
            dest_a = _tmp_video_path(video)
        with (
            patch("pipeline_youtube.stages.capture.os.getpid", return_value=11),
            patch("pipeline_youtube.stages.capture.threading.get_ident", return_value=1),
        ):
            dest_b = _tmp_video_path(video)
        assert dest_a != dest_b
        assert dest_a.parent == dest_b.parent


class TestDownloadDoesNotClobberSiblingDest:
    def test_unlink_before_download_leaves_the_other_worker_file(self, tmp_path: Path):
        dest_a = tmp_path / "dQw4w9WgXcQ-10-1.mp4"
        dest_b = tmp_path / "dQw4w9WgXcQ-11-1.mp4"
        dest_a.write_bytes(b"worker-a")
        dest_b.write_bytes(b"worker-b")

        def fake_backend_download(url: str, dest: Path, *, resolution: str) -> None:
            dest.write_bytes(b"fresh-a")

        class _Backend:
            def download_video(self, url: str, dest: Path, *, resolution: str) -> None:
                fake_backend_download(url, dest, resolution=resolution)

        _download_video(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            dest_a,
            backend=_Backend(),
        )

        assert dest_a.read_bytes() == b"fresh-a"
        assert dest_b.read_bytes() == b"worker-b"
