"""End-to-end CLI test with yt-dlp / ffmpeg / claude -p mocked.

Runs a 3-video playlist through stages 01-05 to catch regressions in
the orchestration layer (stage sequencing, per-stage model routing,
checkpoint / phase-gate logic, cost aggregation).

All external calls are stubbed:
  - `yt-dlp`'s `fetch_metadata` returns 3 synthetic `VideoMeta`.
  - `run_stage_scripts` returns a canned Japanese transcript result.
  - `prefetch_video_download` / `run_stage_capture` return a success
    stub; no real video download occurs.
  - `invoke_claude` returns stage-appropriate canned bodies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline_youtube import main as main_mod
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.claude_cli import ClaudeResponse
from pipeline_youtube.stages.capture import CaptureResult, SummaryRange
from pipeline_youtube.synthesis import agents as agents_mod
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

SUMMARY_OUTPUT = (
    "ONE_LINER: 本日の核心論点\n\n"
    "## 全体サマリ\n\n動画全体の主要な論点を記載。\n\n"
    "## 要点タイムライン\n\n"
    "### [00:00 ~ 00:30] intro\n本文。\n\n"
    "### [00:30 ~ 01:00] key point\n本文。\n"
)

LEARNING_OUTPUT = (
    "## 学習のポイント\n\n### [00:00 ~ 00:30] intro\n時系列メモ。\n\n## 要点\n本文。\n"
)

ALPHA_OUT = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t1",
                "label": "コンテキスト管理",
                "source_videos": ["vid001", "vid002", "vid003"],
                "duplication_count": 3,
                "category": "core",
                "summary": "s",
            }
        ]
    },
    ensure_ascii=False,
)
BETA_OUT = json.dumps(
    {
        "chapters": [
            {
                "index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "topic_ids": ["t1"],
                "source_videos": ["vid001", "vid002", "vid003"],
                "rationale": "r",
            }
        ]
    },
    ensure_ascii=False,
)
LEADER_OUT = json.dumps(
    {
        "moc": {
            "title": "Test Playlist ハンズオン",
            "body_markdown": "# MOC\n- [[01_コンテキスト管理の基礎]]",
        },
        "chapters": [
            {
                "chapter_index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "source_video_ids": ["vid001", "vid002", "vid003"],
                "body_markdown": "## 概念定義\n\n本文。\n",
            }
        ],
    },
    ensure_ascii=False,
)


def _fake_response(text: str, model: str = "sonnet", cost: float = 0.01) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model=model,
        input_tokens=100,
        output_tokens=100,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=cost,
        duration_ms=1000,
    )


def _videos() -> list[VideoMeta]:
    return [
        VideoMeta(
            video_id=f"vid{i:03d}",
            title=f"Video {i}",
            url=f"https://www.youtube.com/watch?v=vid{i:03d}",
            duration=120,
            channel="Test",
            upload_date="20260418",
            playlist_title="Test Playlist",
        )
        for i in range(1, 4)
    ]


def _transcript_result(video_id: str):
    return build_result(
        video_id=video_id,
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[
            TranscriptSnippet("字幕A", 0.0, 30.0),
            TranscriptSnippet("字幕B", 30.0, 30.0),
        ],
    )


def _capture_success() -> CaptureResult:
    return CaptureResult(
        ranges=[SummaryRange(0, 30, "intro"), SummaryRange(30, 60, "key")],
        outcomes=[],
        video_downloaded=True,
        capture_format="webp",
    )


def _stub_invoke_claude_factory():
    """Each call routes to the right canned body based on prompt content."""

    def _route(prompt: str, **kw):
        # Stage 05 agents are dispatched in sequence α→β→Leader; detect by
        # what the prompt/append system prompt contains.
        sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
        if "トピック" in sp or "alpha" in sp.lower() or "topic_id" in prompt:
            return ALPHA_OUT
        if "chapters" in prompt and "index" not in sp:
            return BETA_OUT
        return None

    # Simpler: deterministic queue
    queue = [
        _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01),  # vid1 summary
        _fake_response(LEARNING_OUTPUT, model="sonnet", cost=0.05),  # vid1 learning
        _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01),  # vid2 summary
        _fake_response(LEARNING_OUTPUT, model="sonnet", cost=0.05),  # vid2 learning
        _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01),  # vid3 summary
        _fake_response(LEARNING_OUTPUT, model="sonnet", cost=0.05),  # vid3 learning
        _fake_response(ALPHA_OUT, model="haiku", cost=0.02),
        _fake_response(BETA_OUT, model="sonnet", cost=0.03),
        # γ removed — coverage is now a Python set diff, no LLM call.
        _fake_response(LEADER_OUT, model="opus", cost=0.15),
    ]

    def fake_invoke(**kw):
        if not queue:
            pytest.fail("invoke_claude called more times than canned responses")
        return queue.pop(0)

    return fake_invoke


@pytest.fixture
def vault(tmp_path: Path):
    from pipeline_youtube import config

    (tmp_path / ".obsidian").mkdir()  # satisfy strict mode
    yield tmp_path
    config.reset_vault_root()


class TestE2EPlaylist:
    def test_full_cli_3_videos(self, vault: Path, monkeypatch):
        # Mock Stage 01 transcripts (bypass real youtube-transcript-api)
        def fake_scripts(video, path, *, dry_run, include_code_blocks=False):
            return _transcript_result(video.video_id)

        monkeypatch.setattr(main_mod, "run_stage_scripts", fake_scripts)

        # Mock fetch_metadata (no network)
        monkeypatch.setattr(main_mod, "fetch_metadata", lambda url: _videos())

        # Mock Stage 03 capture (no ffmpeg / yt-dlp)
        monkeypatch.setattr(main_mod, "run_stage_capture", lambda *a, **kw: _capture_success())
        monkeypatch.setattr(main_mod, "prefetch_video_download", lambda video: None)

        # Bypass claude binary validation
        monkeypatch.setattr(
            main_mod, "get_resolved_claude_binary", lambda: ("/fake/claude", "claude 2.1.109")
        )

        # Stub Router (genre classification) — avoid real LLM call
        from pipeline_youtube.genres import Genre

        monkeypatch.setattr(
            main_mod, "classify_playlist_genre", lambda *a, **kw: (Genre.OTHER, "stubbed")
        )

        # Stub every invoke_claude in both stages + synthesis
        from pipeline_youtube.stages import learning as learning_mod
        from pipeline_youtube.stages import summary as summary_mod

        fake_invoke = _stub_invoke_claude_factory()
        monkeypatch.setattr(summary_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(learning_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        # Write a minimal config.json pointing at the vault
        cfg = vault / "config.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/playlist?list=PL_fake",
                "--config",
                str(cfg),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        # Stages executed
        assert "[01] scripts" in result.output
        assert "[02] summary" in result.output
        assert "[03] capture" in result.output
        assert "[04] learning" in result.output
        # Stage 05 ran
        assert "Stage 05 Synthesis" in result.output
        # MOC + 1 chapter written
        moc = vault / "Permanent Note/08_YouTube学習/05_Synthesis"
        assert moc.exists()
        chapter_files = list(moc.rglob("01_*.md"))
        assert len(chapter_files) == 1
        # Cost breakdown appeared
        assert "Cost breakdown" in result.output
        assert "stage_02" in result.output
        assert "leader" in result.output

    def test_stop_after_capture_skips_04_and_05(self, vault: Path, monkeypatch):
        def fake_scripts(video, path, *, dry_run, include_code_blocks=False):
            return _transcript_result(video.video_id)

        monkeypatch.setattr(main_mod, "run_stage_scripts", fake_scripts)
        monkeypatch.setattr(main_mod, "fetch_metadata", lambda url: _videos())
        monkeypatch.setattr(main_mod, "run_stage_capture", lambda *a, **kw: _capture_success())
        monkeypatch.setattr(main_mod, "prefetch_video_download", lambda video: None)
        monkeypatch.setattr(
            main_mod, "get_resolved_claude_binary", lambda: ("/fake/claude", "claude 2.1.109")
        )
        from pipeline_youtube.genres import Genre
        from pipeline_youtube.stages import learning as learning_mod
        from pipeline_youtube.stages import summary as summary_mod

        monkeypatch.setattr(
            main_mod, "classify_playlist_genre", lambda *a, **kw: (Genre.OTHER, "stubbed")
        )

        invoke_count = {"n": 0}

        def fake_invoke(**kw):
            invoke_count["n"] += 1
            return _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01)

        monkeypatch.setattr(summary_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(learning_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        cfg = vault / "config.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/playlist?list=PL_fake",
                "--config",
                str(cfg),
                "--stop-after-capture",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        # Only 3 Stage 02 calls — no Stage 04 or synthesis agents
        assert invoke_count["n"] == 3
        assert "stop-after-capture" in result.output
        assert "[04] learning" not in result.output
        assert "Stage 05 Synthesis" not in result.output

    def test_resume_reviewed_runs_only_stage_04_on_existing_phase1_files(
        self, vault: Path, monkeypatch
    ):
        video = VideoMeta(
            video_id="abc1234567A",
            title="Reviewed Video",
            url="https://www.youtube.com/watch?v=abc1234567A",
            duration=120,
            channel="Test",
            upload_date="20260418",
            playlist_title="Test Playlist",
        )
        note_name = "2026-04-18-0800 Reviewed Video.md"
        phase1_folder = "2026-04-18-0800 Test Playlist"
        summary_md = (
            vault
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["summary"]
            / phase1_folder
            / note_name
        )
        capture_md = (
            vault
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["capture"]
            / phase1_folder
            / note_name
        )
        fm = (
            '---\n'
            'date: 2026-04-18 08:00\n'
            'title: "Reviewed Video"\n'
            'URL: "https://www.youtube.com/watch?v=abc1234567A"\n'
            'playlist: "Test Playlist"\n'
            'video_id: "abc1234567A"\n'
            'reviewed: "true"\n'
            'tags: [memo, youtube]\n'
            '---\n\n'
        )
        summary_md.parent.mkdir(parents=True, exist_ok=True)
        summary_md.write_text(fm + "MANUAL REVIEWED SUMMARY\n", encoding="utf-8")
        capture_md.parent.mkdir(parents=True, exist_ok=True)
        capture_md.write_text(
            fm + "[00:00 ~ 00:30]\n![[reviewed-capture.webp]]\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(main_mod, "fetch_metadata", lambda url: [video])
        monkeypatch.setattr(
            main_mod, "run_stage_scripts", lambda *a, **kw: pytest.fail("stage 01 reran")
        )
        monkeypatch.setattr(
            main_mod, "run_stage_summary", lambda *a, **kw: pytest.fail("stage 02 reran")
        )
        monkeypatch.setattr(
            main_mod, "run_stage_capture", lambda *a, **kw: pytest.fail("stage 03 reran")
        )
        monkeypatch.setattr(
            main_mod, "get_resolved_claude_binary", lambda: ("/fake/claude", "claude 2.1.109")
        )
        from pipeline_youtube.genres import Genre

        monkeypatch.setattr(
            main_mod, "classify_playlist_genre", lambda *a, **kw: (Genre.OTHER, "stubbed")
        )

        learning_calls = []

        def fake_learning(video, summary_path, capture_path, learning_path, **kw):
            learning_calls.append((summary_path, capture_path, learning_path))
            assert "MANUAL REVIEWED SUMMARY" in summary_path.read_text(encoding="utf-8")
            assert capture_path == capture_md
            learning_path.parent.mkdir(parents=True, exist_ok=True)
            learning_path.write_text(
                fm.replace('reviewed: "true"\n', "") + "LEARNING FROM REVIEWED\n",
                encoding="utf-8",
            )
            return _fake_response("LEARNING FROM REVIEWED", model="sonnet", cost=0.05)

        monkeypatch.setattr(main_mod, "run_stage_learning", fake_learning)

        cfg = vault / "config.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/playlist?list=PL_fake",
                "--config",
                str(cfg),
                "--resume-reviewed",
                "--skip-synthesis",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "[01] scripts" not in result.output
        assert "[02] summary" not in result.output
        assert "[03] capture" not in result.output
        assert "[04] learning" in result.output
        assert len(learning_calls) == 1
        _summary_path, _capture_path, learning_path = learning_calls[0]
        assert learning_path == (
            vault
            / main_mod.LEARNING_BASE
            / main_mod.UNIT_DIRS["learning"]
            / phase1_folder
            / note_name
        )
        assert 'reviewed: "true"' in summary_md.read_text(encoding="utf-8")
        assert "MANUAL REVIEWED SUMMARY" in summary_md.read_text(encoding="utf-8")
