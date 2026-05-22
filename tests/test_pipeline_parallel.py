"""Tests for WS4: Stage 03 download runs concurrently with Stage 02 LLM call."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pipeline_youtube import config
from pipeline_youtube import main as main_mod
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.claude_cli import ClaudeResponse
from pipeline_youtube.stages.capture import CaptureResult, VideoPrefetch, prefetch_video_download
from pipeline_youtube.transcript.base import TranscriptSnippet, TranscriptSource, build_result


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
            output_path.write_bytes(b"img")

        monkeypatch.setattr(cap_mod, "_download_video", never_download)
        monkeypatch.setattr(cap_mod, "_dispatch_extractor", lambda _strategy: fake_extractor)
        monkeypatch.setattr(
            cap_mod,
            "_resolve_capture_format",
            lambda _req, _backend: cap_mod._FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(cap_mod, "get_vault_root", lambda: tmp_path)
        monkeypatch.setattr(cap_mod, "ensure_safe_path", lambda p: p)

        result = cap_mod.run_stage_capture(
            _video(),
            summary_md,
            capture_md,
            prefetched_video_path=fake_video,
        )

        assert called["download"] == 0
        assert called["extract"] == 1
        assert result.outcomes and result.outcomes[0].success


class TestDuplicateTitlePathSafety:
    @pytest.mark.asyncio
    async def test_duplicate_titles_are_serialized_under_video_concurrency(
        self, tmp_path: Path, monkeypatch
    ):
        """Same-title videos share filename state, so their 01-04 writes must not overlap."""
        config.set_vault_root(tmp_path)
        run_time = datetime(2026, 4, 15, 21, 23)
        videos = [
            VideoMeta(
                video_id=f"dupetitle{i}",
                title="Same Title",
                url=f"https://www.youtube.com/watch?v=dupetitle{i}",
                duration=60,
                channel="ch",
                upload_date=None,
                playlist_title="Test Playlist",
            )
            for i in range(2)
        ]

        active_scripts = 0
        max_active_scripts = 0
        seen_script_paths: list[Path] = []

        def fake_scripts(video, path, *, dry_run, include_code_blocks=False):
            nonlocal active_scripts, max_active_scripts
            assert path.exists()
            seen_script_paths.append(path)
            active_scripts += 1
            max_active_scripts = max(max_active_scripts, active_scripts)
            time.sleep(0.05)
            active_scripts -= 1
            return build_result(
                video_id=video.video_id,
                source=TranscriptSource.OFFICIAL,
                language="ja",
                snippets=[TranscriptSnippet("字幕", 0.0, 5.0)],
            )

        def fake_summary(*args, **kwargs):
            return ClaudeResponse(text="summary", model="haiku")

        def fake_capture(*args, **kwargs):
            return CaptureResult(
                ranges=[],
                outcomes=[],
                video_downloaded=False,
                capture_format="webp",
            )

        def fake_learning(video, summary_path, capture_path, learning_path, **kwargs):
            learning_path.parent.mkdir(parents=True, exist_ok=True)
            learning_path.write_text(f"---\n---\n\nbody {video.video_id}\n", encoding="utf-8")
            return ClaudeResponse(text=f"body {video.video_id}", model="sonnet")

        monkeypatch.setattr(main_mod, "run_stage_scripts", fake_scripts)
        monkeypatch.setattr(main_mod, "run_stage_summary", fake_summary)
        monkeypatch.setattr(main_mod, "run_stage_capture", fake_capture)
        monkeypatch.setattr(main_mod, "run_stage_learning", fake_learning)
        monkeypatch.setattr(main_mod, "prefetch_video_download", lambda *a, **kw: None)

        try:
            results = await main_mod._run_videos_concurrent(
                videos,
                run_time,
                concurrency=2,
                dry_run=False,
                capture_format="auto",
                models={"stage_02": "haiku", "stage_04": "sonnet"},
            )
        finally:
            config.reset_vault_root()

        assert max_active_scripts == 1
        assert len({path.name for path in seen_script_paths}) == 2
        assert [result.error for result in results] == [None, None]
        learning_names = sorted(result.learning_md_path.name for result in results)
        assert learning_names == [
            "2026-04-15-2123 Same Title-2.md",
            "2026-04-15-2123 Same Title.md",
        ]
