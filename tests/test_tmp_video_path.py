"""Tmp ownership across overlapping processes, threads and capture modes (#200)."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stages import capture
from pipeline_youtube.stages.capture_backend import DockerCaptureBackend, HostCaptureBackend


@pytest.fixture(autouse=True)
def isolated_project(tmp_path: Path, monkeypatch):
    # Exercise the real path builder without writing to the checkout's tmp.
    monkeypatch.setattr(capture, "__file__", str(tmp_path / "pipeline_youtube/stages/capture.py"))


@pytest.fixture
def video() -> VideoMeta:
    return VideoMeta(
        video_id="abc123abc12",
        title="test",
        url="https://www.youtube.com/watch?v=abc123abc12",
        duration=60,
        channel="test",
        upload_date=None,
        playlist_title=None,
    )


def _path_for(video: VideoMeta, *, pid: int = 100, tid: int = 200) -> Path:
    # Keep identity patches out of executor/backend internals.
    with patch("os.getpid", return_value=pid), patch("threading.get_ident", return_value=tid):
        return capture._tmp_video_path(video)


def test_filename_contains_process_and_thread_ids(video: VideoMeta, tmp_path: Path):
    path = _path_for(video, pid=1234, tid=5678)
    assert path == tmp_path / "tmp/abc123abc12-1234-5678.mp4"


def test_different_processes_get_different_destinations(video: VideoMeta):
    assert _path_for(video, pid=100) != _path_for(video, pid=101)


def test_different_threads_get_different_destinations(video: VideoMeta):
    assert _path_for(video, tid=200) != _path_for(video, tid=201)


def test_same_thread_reuses_same_path(video: VideoMeta):
    assert capture._tmp_video_path(video) == capture._tmp_video_path(video)


def test_live_threads_get_different_destinations(video: VideoMeta):
    barrier = threading.Barrier(2, timeout=5)

    def work() -> Path:
        path = capture._tmp_video_path(video)
        barrier.wait()  # Both workers must still be alive (thread IDs can be reused).
        return path

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(work) for _ in range(2)]
        paths = [future.result(timeout=10) for future in futures]
    assert paths[0] != paths[1]


@pytest.mark.parametrize("other_identity", [{"pid": 101}, {"tid": 201}], ids=["process", "thread"])
def test_download_unlink_preserves_other_worker_files(video: VideoMeta, other_identity):
    dest = _path_for(video)
    other = _path_for(video, **other_identity)
    for suffix in (".mp4", ".mkv", ".webm"):
        dest.with_suffix(suffix).write_bytes(b"old own download")
        other.with_suffix(suffix).write_bytes(b"other worker")

    def download(*args, **kwargs):
        # Real _download_video unlinks the MP4; the real host backend clears
        # same-stem alternative containers before the fake network boundary.
        for suffix in (".mp4", ".mkv", ".webm"):
            assert not dest.with_suffix(suffix).exists()
            assert other.with_suffix(suffix).read_bytes() == b"other worker"
        dest.write_bytes(b"new own download")
        return Mock(wait=Mock(return_value=0))

    with patch("subprocess.Popen", side_effect=download), patch("os.killpg"):
        capture._download_video(video.watch_url, dest, backend=HostCaptureBackend())

    assert dest.read_bytes() == b"new own download"
    for suffix in (".mp4", ".mkv", ".webm"):
        assert other.with_suffix(suffix).read_bytes() == b"other worker"


@pytest.mark.parametrize("mode", ["stage03", "hands_on"])
@pytest.mark.parametrize("other_identity", [{"pid": 101}, {"tid": 201}], ids=["process", "thread"])
def test_capture_cleanup_preserves_other_worker_video(
    video: VideoMeta, tmp_path: Path, monkeypatch, mode: str, other_identity
):
    other = _path_for(video, **other_identity)
    other.write_bytes(b"other worker")
    own = _path_for(video)
    (tmp_path / ".obsidian").mkdir()
    summary = tmp_path / "02.md"
    summary.write_text("### [00:00 ~ 00:05] test\n", encoding="utf-8")
    note = tmp_path / "03.md"
    note.write_text("---\n---\n", encoding="utf-8")

    def download(_url: str, dest: Path, **_kwargs):
        assert dest == own
        dest.write_bytes(b"own download")

    def extract(source: Path, output: Path, **_kwargs):
        assert source == own
        assert source.read_bytes() == b"own download"
        output.write_bytes(b"RIFF" + (8).to_bytes(4, "little") + b"WEBPVP8L")

    backend = Mock()
    backend.download_video.side_effect = download
    monkeypatch.setattr(capture, "_dispatch_extractor", lambda _strategy: extract)
    monkeypatch.setattr(
        capture, "_resolve_capture_format", lambda *_args: capture._FormatChoice("webp", "direct")
    )
    with patch("os.getpid", return_value=100), patch("threading.get_ident", return_value=200):
        if mode == "stage03":
            result = capture.run_stage_capture(
                video, summary, note, backend=backend, vault_root=tmp_path
            )
        else:
            result = capture.capture_step_clips(
                video,
                [capture.SummaryRange(0, 5, "test")],
                assets_subfolder="hands-on",
                backend=backend,
                vault_root=tmp_path,
            )

    assert result.error is None
    assert result.success_count == 1
    assert not own.exists()
    assert other.read_bytes() == b"other worker"


def test_prefetch_uses_callers_path_before_and_after_worker_finishes(video: VideoMeta):
    expected = capture._tmp_video_path(video)
    caller_tid = threading.get_ident()
    downloads: list[tuple[int, Path]] = []

    def download(_url: str, dest: Path, *_args, **_kwargs):
        downloads.append((threading.get_ident(), dest))
        dest.write_bytes(b"prefetched")

    with patch.object(capture, "_download_video", side_effect=download):
        handle = capture.prefetch_video_download(video)
        assert handle.path == expected == capture._tmp_video_path(video)
        assert handle.wait(timeout=5) is None

    assert downloads == [(downloads[0][0], expected)]
    assert downloads[0][0] != caller_tid
    assert capture._tmp_video_path(video) == expected
    assert expected.read_bytes() == b"prefetched"


def test_generated_path_translates_to_docker_work_mount(video: VideoMeta, tmp_path: Path):
    path = capture._tmp_video_path(video)
    backend = DockerCaptureBackend(tmp_dir=tmp_path / "tmp", assets_dir=tmp_path / "assets")
    assert path.parent == backend.tmp_dir
    assert backend._host_to_container(path) == f"/work/{path.name}"


def test_sweep_removes_stale_generated_path(video: VideoMeta):
    path = capture._tmp_video_path(video)
    path.write_bytes(b"stale download")
    past = time.time() - 48 * 3600
    os.utime(path, (past, past))
    assert capture.sweep_stale_tmp(path.parent) == 1
    assert not path.exists()
