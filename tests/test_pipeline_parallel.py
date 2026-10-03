"""Tests for WS4: Stage 03 download runs concurrently with Stage 02 LLM call."""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stages.capture import VideoPrefetch, _tmp_video_path, prefetch_video_download


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="abc1234567",
        title="test",
        url="https://www.youtube.com/watch?v=abc1234567",
        duration=60,
        channel="ch",
        upload_date=None,
        playlist_title=None,
    )


@pytest.fixture(autouse=True)
def _isolated_video_tmp_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "pipeline_youtube.stages.capture.__file__",
        str(tmp_path / "pipeline_youtube/stages/capture.py"),
    )


class TestPrefetchHandle:
    def test_wait_returns_none_on_success(self, tmp_path: Path):
        def fake_download(url: str, dest: Path, resolution: str = "480", **kw: Any) -> None:
            dest.write_bytes(b"fake mp4")

        with patch("pipeline_youtube.stages.capture._download_video", fake_download):
            handle = prefetch_video_download(_video())
            assert isinstance(handle, VideoPrefetch)
            assert handle.wait(timeout=5.0) is None
            assert handle.path.exists()
            handle.path.unlink(missing_ok=True)

    def test_wait_returns_exception_on_failure(self):
        def fake_download(url: str, dest: Path, resolution: str = "480", **kw: Any) -> None:
            raise RuntimeError("boom")

        with patch("pipeline_youtube.stages.capture._download_video", fake_download):
            handle = prefetch_video_download(_video())
            err = handle.wait(timeout=5.0)
            assert isinstance(err, RuntimeError)
            assert "boom" in str(err)

    def test_wait_returns_download_timeout_without_retrying(self, tmp_path: Path):
        error = FutureTimeoutError("download itself timed out")
        future: Future[None] = Future()
        future.set_exception(error)
        handle = VideoPrefetch(path=tmp_path / "video.mp4", future=future)

        assert handle.wait(timeout=None) is error

    def test_finite_wait_timeout_leaves_download_pending(self, tmp_path: Path):
        future: Future[None] = Future()
        handle = VideoPrefetch(path=tmp_path / "video.mp4", future=future)

        assert isinstance(handle.wait(timeout=0), FutureTimeoutError)
        assert not future.done()
        assert not future.cancelled()

        future.set_result(None)
        assert handle.wait(timeout=0) is None

    def test_wait_preserves_default_timeout(self, tmp_path: Path):
        future = Mock()
        handle = VideoPrefetch(path=tmp_path / "video.mp4", future=future)

        assert handle.wait() is None
        assert 599 < future.result.call_args.kwargs["timeout"] <= 600

    def test_wait_uses_deadline_from_start_not_new_budget(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("pipeline_youtube.stages.capture.time.monotonic", lambda: 1000.0)
        future = Mock()
        handle = VideoPrefetch(path=tmp_path / "video.mp4", future=future, started_at=450.0)
        assert handle.wait(timeout=None) is None
        future.result.assert_called_once_with(timeout=50.0)


@pytest.fixture
def process_with_prefetch(tmp_path: Path, monkeypatch):
    """Exercise the real Stage 03 while keeping all I/O inside tmp_path."""
    from pipeline_youtube import video_processing as vp_mod
    from pipeline_youtube.pipeline import NoteReservations
    from pipeline_youtube.providers.claude_cli import ClaudeResponse
    from pipeline_youtube.stages import capture as cap_mod
    from pipeline_youtube.transcript.base import (
        TranscriptResult,
        TranscriptSnippet,
        TranscriptSource,
    )

    paths = {
        stage: tmp_path / f"{stage}.md" for stage in ("scripts", "summary", "capture", "learning")
    }
    paths["capture"].write_text("---\n---\n", encoding="utf-8")
    transcript = TranscriptResult(
        video_id=_video().video_id,
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[TranscriptSnippet(text="test", start=0, duration=5)],
    )
    response = ClaudeResponse(text="learning body", model="fake")

    def fake_summary(*args: Any, **kwargs: Any):
        paths["summary"].write_text(
            "---\n---\n\n## 要点タイムライン\n### [00:00 ~ 00:05] heading\n本文\n",
            encoding="utf-8",
        )
        return response

    def fake_learning(*args: Any, **kwargs: Any):
        paths["learning"].write_text(response.text, encoding="utf-8")
        return response

    def fake_download(url: str, dest: Path, **kwargs: Any) -> None:
        dest.write_bytes(b"fallback mp4")

    def fake_extractor(video_path: Path, output_path: Path, **kwargs: Any) -> None:
        assert video_path.exists()
        output_path.write_bytes(b"RIFF" + (8).to_bytes(4, "little") + b"WEBPVP8L")

    download_spy = Mock(side_effect=fake_download)
    capture_spy = Mock(wraps=cap_mod.run_stage_capture)
    monkeypatch.setattr(vp_mod, "reserve_note_paths", lambda *args, **kwargs: paths)
    monkeypatch.setattr(vp_mod, "run_stage_scripts", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(vp_mod, "record_transcript_stat", lambda *args, **kwargs: None)
    monkeypatch.setattr(vp_mod, "run_stage_summary", fake_summary)
    monkeypatch.setattr(vp_mod, "run_stage_learning", fake_learning)
    monkeypatch.setattr(vp_mod, "run_stage_capture", capture_spy)
    monkeypatch.setattr(cap_mod, "_download_video", download_spy)
    monkeypatch.setattr(cap_mod, "_dispatch_extractor", lambda _strategy: fake_extractor)
    monkeypatch.setattr(
        cap_mod,
        "_resolve_capture_format",
        lambda _req, _backend: cap_mod._FormatChoice(ext="webp", strategy="direct"),
    )
    monkeypatch.setattr(cap_mod, "ensure_safe_path", lambda p, **kwargs: p)

    def process(handle: VideoPrefetch | None = None):
        if handle is not None:
            monkeypatch.setattr(vp_mod, "prefetch_video_download", lambda *args, **kwargs: handle)
        result = vp_mod._process_video(
            _video(),
            datetime(2026, 10, 3, 12, 0),
            dry_run=False,
            capture_format="webp",
            models={"stage_02": "fake", "stage_04": "fake"},
            vault_root=tmp_path,
            reservations=NoteReservations(),
        )
        return result, capture_spy, download_spy

    return process


class TestPrefetchHandoff:
    def test_never_finishing_prefetch_returns_within_deadline(
        self, tmp_path: Path, process_with_prefetch, monkeypatch
    ):
        """Self-bounded regression: even the broken implementation can be released."""
        from pipeline_youtube.stages import capture as cap_mod

        monkeypatch.setattr(cap_mod, "PREFETCH_TIMEOUT_SECONDS", 0.05, raising=False)
        future: Future[None] = Future()
        future.set_running_or_notify_cancel()
        handle = VideoPrefetch(path=tmp_path / "pending.mp4", future=future)
        finished = threading.Event()
        results = []

        def process():
            try:
                results.append(process_with_prefetch(handle))
            finally:
                finished.set()

        worker = threading.Thread(target=process, daemon=True)
        worker.start()
        try:
            assert finished.wait(0.5), "_process_video exceeded the prefetch deadline"
            result, capture_spy, download_spy = results[0]
            assert not result.ok
            assert "prefetch_timeout" in result.error
            capture_spy.assert_not_called()
            download_spy.assert_not_called()
            assert not future.done(), "a wait timeout must not pretend the writer stopped"
        finally:
            future.set_exception(RuntimeError("test releases the pending writer"))
            worker.join(timeout=1)
        assert not worker.is_alive()

    def test_pending_prefetch_finishes_before_stage03_without_second_download(
        self, tmp_path: Path, process_with_prefetch
    ):
        path = _tmp_video_path(_video())
        future: Future[None] = Future()
        future.set_running_or_notify_cancel()

        def finish_download():
            path.write_bytes(b"prefetched mp4")
            future.set_result(None)

        handle = VideoPrefetch(path=path, future=future)
        timer = threading.Timer(0.05, finish_download)
        timer.start()
        try:
            result, capture_spy, download_spy = process_with_prefetch(handle)
        finally:
            timer.join(timeout=1)

        assert result.ok
        capture_spy.assert_called_once()
        download_spy.assert_not_called()
        assert capture_spy.call_args.kwargs["prefetched_video_path"] == handle.path
        assert future.done()

    def test_completed_prefetch_failure_allows_stage03_download(
        self, tmp_path: Path, process_with_prefetch
    ):
        path = _tmp_video_path(_video())
        future: Future[None] = Future()
        future.set_exception(RuntimeError("download failed"))
        handle = VideoPrefetch(path=path, future=future)

        result, capture_spy, download_spy = process_with_prefetch(handle)

        assert result.ok
        capture_spy.assert_called_once()
        assert capture_spy.call_args.kwargs["prefetched_video_path"] is None
        download_spy.assert_called_once()
        assert download_spy.call_args.args[1] == handle.path

    def test_stage02_exception_keeps_path_owned_until_writer_finishes(
        self, process_with_prefetch, monkeypatch
    ):
        from pipeline_youtube import video_processing as vp_mod
        from pipeline_youtube.stages import capture as cap_mod

        monkeypatch.setattr(cap_mod, "PREFETCH_TIMEOUT_SECONDS", 0.05)
        started = threading.Event()
        release = threading.Event()
        calls = []
        handles = []
        real_prefetch = cap_mod.prefetch_video_download
        summary = vp_mod.run_stage_summary

        def tracked_prefetch(*args, **kwargs):
            handle = real_prefetch(*args, **kwargs)
            handles.append(handle)
            return handle

        def download(url, path, *args, **kwargs):
            calls.append(path)
            started.set()
            assert release.wait(2), "test writer was not released"
            path.write_bytes(b"complete")

        def failing_summary(*args, **kwargs):
            assert started.wait(1)
            raise RuntimeError("Stage 02 failed")

        monkeypatch.setattr(vp_mod, "prefetch_video_download", tracked_prefetch)
        monkeypatch.setattr(cap_mod, "_download_video", download)
        monkeypatch.setattr(vp_mod, "run_stage_summary", failing_summary)
        try:
            first, capture_spy, _ = process_with_prefetch()
            assert "Stage 02 failed" in first.error
            assert not handles[0].future.done()
            monkeypatch.setattr(vp_mod, "run_stage_summary", summary)
            second, _, _ = process_with_prefetch()  # same thread, video, and path
            assert second.error == "prefetch_timeout"
            capture_spy.assert_not_called()
            assert len(calls) == 1, "second writer started on the still-owned path"
            assert handles[0] is handles[1]
        finally:
            release.set()
            for handle in handles:
                handle.future.result(timeout=1)
        # Callback completion is distinct from result() becoming observable.
        deadline = time.monotonic() + 1
        while handles[0].path in cap_mod._prefetch_by_path and time.monotonic() < deadline:
            time.sleep(0.001)
        third, _, _ = process_with_prefetch()
        assert third.ok
        assert len(calls) == 2
        assert calls[0] == calls[1]

    @pytest.mark.parametrize(
        "error", [TimeoutError("backend timeout"), subprocess.TimeoutExpired("fake", 600)]
    )
    def test_backend_timeout_quarantines_path_without_retry(
        self, process_with_prefetch, monkeypatch, error
    ):
        from pipeline_youtube.stages import capture as cap_mod

        download = Mock(side_effect=error)
        monkeypatch.setattr(cap_mod, "_download_video", download)
        first, capture_spy, _ = process_with_prefetch()
        second, _, _ = process_with_prefetch()
        assert first.error == second.error == "prefetch_timeout"
        capture_spy.assert_not_called()
        download.assert_called_once()


class TestParallelOverlap:
    @pytest.mark.asyncio
    async def test_download_overlaps_with_llm(self, tmp_path: Path, monkeypatch):
        """If download and LLM each take 0.5s, sequential is ~1s, parallel is ~0.5s."""

        def slow_download(url: str, dest: Path, resolution: str = "480", **kw: Any) -> None:
            time.sleep(0.5)
            dest.write_bytes(b"x")

        # Start prefetch; simulate Stage 02 by sleeping the same amount
        with patch("pipeline_youtube.stages.capture._download_video", slow_download):
            t0 = time.monotonic()
            handle = prefetch_video_download(_video())
            time.sleep(0.5)  # simulate Stage 02 LLM latency
            err = handle.wait(timeout=5.0)
            elapsed = time.monotonic() - t0

        assert err is None
        # Allow some overhead but must be well under sequential 1.0s
        assert elapsed < 0.9, f"expected overlap, got elapsed={elapsed:.2f}s"
        handle.path.unlink(missing_ok=True)


class TestPrefetchedPathConsumed:
    def test_capture_skips_download_when_prefetch_present(self, tmp_path: Path, monkeypatch):
        from pipeline_youtube.stages import capture as cap_mod

        # Prepare a fake summary md with one range and a fake prefetched video
        summary_md = tmp_path / "02.md"
        summary_md.write_text(
            "---\n---\n\n## 要点タイムライン\n### [00:00 ~ 00:05] heading\n本文\n",
            encoding="utf-8",
        )
        capture_md = tmp_path / "03.md"
        capture_md.write_text("---\n---\n", encoding="utf-8")

        fake_video = tmp_path / "fake.mp4"
        fake_video.write_bytes(b"x")

        called: dict[str, int] = {"download": 0, "extract": 0}

        def never_download(*args: Any, **kwargs: Any) -> None:
            called["download"] += 1

        def fake_extractor(video_path: Path, output_path: Path, **kwargs: Any) -> None:
            called["extract"] += 1
            # Stage 03 publishes only a whole image (#190): the smallest WebP.
            output_path.write_bytes(b"RIFF" + (8).to_bytes(4, "little") + b"WEBPVP8L")

        monkeypatch.setattr(cap_mod, "_download_video", never_download)
        monkeypatch.setattr(cap_mod, "_dispatch_extractor", lambda _strategy: fake_extractor)
        monkeypatch.setattr(
            cap_mod,
            "_resolve_capture_format",
            lambda _req, _backend: cap_mod._FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(cap_mod, "ensure_safe_path", lambda p, **kw: p)

        result = cap_mod.run_stage_capture(
            _video(),
            summary_md,
            capture_md,
            prefetched_video_path=fake_video,
            vault_root=tmp_path,
        )

        assert called["download"] == 0
        assert called["extract"] == 1
        assert result.outcomes and result.outcomes[0].success
        # No network download happened, so the flag must report False.
        assert result.video_downloaded is False

    def test_capture_fails_closed_when_local_media_source_missing(
        self, tmp_path: Path, monkeypatch
    ):
        """--local-media (allow_download=False) must never fall back to YouTube."""
        from pipeline_youtube.stages import capture as cap_mod

        summary_md = tmp_path / "02.md"
        summary_md.write_text(
            "---\n---\n\n## 要点タイムライン\n### [00:00 ~ 00:05] heading\n本文\n",
            encoding="utf-8",
        )
        capture_md = tmp_path / "03.md"
        capture_md.write_text("---\n---\n", encoding="utf-8")
        missing_video = tmp_path / "missing.mp4"

        called: dict[str, int] = {"download": 0}

        def never_download(*args: Any, **kwargs: Any) -> None:
            called["download"] += 1

        monkeypatch.setattr(cap_mod, "_download_video", never_download)
        monkeypatch.setattr(
            cap_mod,
            "_resolve_capture_format",
            lambda _req, _backend: cap_mod._FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(cap_mod, "ensure_safe_path", lambda p, **kw: p)

        result = cap_mod.run_stage_capture(
            _video(),
            summary_md,
            capture_md,
            prefetched_video_path=missing_video,
            allow_download=False,
            vault_root=tmp_path,
        )

        assert called["download"] == 0
        assert result.error is not None
        assert result.error.startswith("local_media_file_missing")
        assert result.outcomes == []
