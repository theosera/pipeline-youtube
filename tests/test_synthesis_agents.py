"""Tests for synthesis.agents with claude_cli mocked."""

from __future__ import annotations

import json

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.claude_cli import ClaudeResponse
from pipeline_youtube.synthesis import agents as agents_mod
from pipeline_youtube.synthesis.agents import (
    call_alpha,
    call_beta,
    call_leader,
    compute_coverage,
    format_learning_materials,
)
from pipeline_youtube.synthesis.scoring import (
    ChapterPlan,
    CoverageReport,
    SynthesisParseError,
    Topic,
)


def _video(video_id: str, title: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=900,
        channel="Test Channel",
        upload_date="20260415",
        playlist_title="Test Playlist",
    )


def _fake_response(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=3,
        output_tokens=500,
        cache_creation_tokens=24000,
        cache_read_tokens=15000,
        total_cost_usd=0.10,
        duration_ms=20000,
        session_id="fake",
        stop_reason="end_turn",
    )


# =====================================================
# format_learning_materials
# =====================================================


class TestFormatLearningMaterials:
    def test_delimits_by_video_header(self):
        videos = [_video("vid1", "First Video"), _video("vid2", "Second Video")]
        bodies = ["## 概念: A\n\n要点", "## 概念: B\n\n別の要点"]
        formatted = format_learning_materials(videos, bodies)
        assert "## VIDEO: vid1: First Video" in formatted
        assert "## VIDEO: vid2: Second Video" in formatted
        assert "## 概念: A" in formatted
        assert "## 概念: B" in formatted
        # Videos separated by --- delimiter
        assert "\n---\n" in formatted

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            format_learning_materials([_video("v1", "t1")], ["body 1", "body 2"])

    def test_sanitizes_control_chars(self):
        videos = [_video("v1", "title\x01with\x08control")]
        bodies = ["body\u200bwith\x0czero-width"]
        formatted = format_learning_materials(videos, bodies)
        assert "\x01" not in formatted
        assert "\x08" not in formatted
        assert "\u200b" not in formatted


# =====================================================
# call_alpha
# =====================================================


SAMPLE_ALPHA_OUTPUT = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "コンテキスト管理",
                "source_videos": ["vid1", "vid2", "vid3"],
                "duplication_count": 3,
                "category": "core",
                "summary": "コンテキストウィンドウの管理。",
                "excerpts": [],
            },
            {
                "topic_id": "t002",
                "label": "Agent Teams 構成",
                "source_videos": ["vid1", "vid2"],
                "duplication_count": 2,
                "category": "supporting",
                "summary": "複数エージェントの分業。",
            },
        ]
    },
    ensure_ascii=False,
)


class TestCallAlpha:
    def test_happy_path(self, monkeypatch):
        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_response(SAMPLE_ALPHA_OUTPUT)

        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        videos = [
            _video("vid1", "First"),
            _video("vid2", "Second"),
            _video("vid3", "Third"),
        ]
        bodies = ["body1", "body2", "body3"]

        topics, result = call_alpha(videos, bodies, playlist_title="Test Playlist")

        assert len(topics) == 2
        assert topics[0].topic_id == "t001"
        assert topics[0].category == "core"
        assert topics[1].category == "supporting"

        # System prompt is append mode
        assert "append_system_prompt" in captured
        assert (
            "TopicExtractor" in captured["append_system_prompt"]
            or "topic" in captured["append_system_prompt"].lower()
        )

        # Prompt wraps materials in untrusted_content
        prompt = captured["prompt"]
        assert "<untrusted_content>" in prompt
        assert "Test Playlist" in prompt
        assert "## VIDEO: vid1: First" in prompt

        # Usage metadata propagated
        assert result.output_tokens == 500
        assert result.cache_creation_tokens == 24000

    def test_parse_error_propagates(self, monkeypatch):
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: _fake_response("not valid json at all"),
        )

        with pytest.raises(SynthesisParseError):
            call_alpha([_video("v1", "t1")], ["body"], playlist_title="x")


# =====================================================
# call_beta
# =====================================================


SAMPLE_BETA_OUTPUT = json.dumps(
    {
        "chapters": [
            {
                "index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "topic_ids": ["t001"],
                "source_videos": ["vid1", "vid2", "vid3"],
                "rationale": "全動画で言及される最重要概念",
            },
            {
                "index": 2,
                "label": "Agent Teams 実装",
                "category": "supporting",
                "topic_ids": ["t002"],
                "source_videos": ["vid1", "vid2"],
                "rationale": "2 本で取り上げられる実装手法",
            },
        ]
    },
    ensure_ascii=False,
)


class TestCallBeta:
    def test_happy_path(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [
            Topic(
                topic_id="t001",
                label="コンテキスト管理",
                source_videos=["vid1", "vid2", "vid3"],
                duplication_count=3,
                category="core",
            ),
            Topic(
                topic_id="t002",
                label="Agent Teams 構成",
                source_videos=["vid1", "vid2"],
                duplication_count=2,
                category="supporting",
            ),
        ]
        chapters, result = call_beta(topics)

        assert len(chapters) == 2
        assert chapters[0].index == 1
        assert chapters[0].category == "core"
        assert chapters[1].category == "supporting"

        # Prompt includes serialized topics
        prompt = captured["prompt"]
        assert "t001" in prompt
        assert "t002" in prompt
        assert (
            "ChapterArchitect" in captured["append_system_prompt"]
            or "章" in captured["append_system_prompt"]
        )

    def test_max_chapters_injects_prompt_constraint(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [
            Topic(topic_id="t001", label="A", duplication_count=3, category="core"),
        ]
        call_beta(topics, max_chapters=5)

        prompt = captured["prompt"]
        assert "最大 5 章" in prompt

    def test_unset_max_chapters_omits_constraint(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [
            Topic(topic_id="t001", label="A", duplication_count=3, category="core"),
        ]
        call_beta(topics)

        prompt = captured["prompt"]
        assert "最大" not in prompt
        assert "追加制約" not in prompt


# =====================================================
# compute_coverage (replaces call_gamma)
# =====================================================


class TestComputeCoverage:
    def test_all_covered(self):
        topics = [
            Topic(topic_id="t001", label="x", duplication_count=3, category="core"),
            Topic(topic_id="t002", label="y", duplication_count=2, category="supporting"),
        ]
        chapters = [
            ChapterPlan(
                index=1, label="ch1", category="core", topic_ids=["t001", "t002"], source_videos=[]
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == []
        assert report.notes == ""

    def test_missing_topic(self):
        topics = [
            Topic(topic_id="t001", label="a", duplication_count=1, category="unique"),
            Topic(topic_id="t002", label="b", duplication_count=1, category="unique"),
            Topic(topic_id="t003", label="c", duplication_count=1, category="unique"),
        ]
        chapters = [
            ChapterPlan(
                index=1, label="ch1", category="unique", topic_ids=["t001"], source_videos=[]
            ),
            ChapterPlan(
                index=2, label="ch2", category="unique", topic_ids=["t002"], source_videos=[]
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == ["t003"]

    def test_chapter_references_unknown_topic_is_not_covered(self):
        """Chapter topic_ids not in α topics must not appear in covered_topic_ids."""
        topics = [
            Topic(topic_id="t001", label="x", duplication_count=1, category="unique"),
        ]
        chapters = [
            ChapterPlan(
                index=1,
                label="ch1",
                category="unique",
                topic_ids=["t001", "t999"],  # t999 is a hallucinated id
                source_videos=[],
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001"]
        assert report.missing_topic_ids == []

    def test_empty_inputs(self):
        report = compute_coverage([], [])
        assert report.covered_topic_ids == []
        assert report.missing_topic_ids == []

    def test_sorted_output(self):
        """Output lists are sorted for deterministic downstream diffs."""
        topics = [
            Topic(topic_id="t003", label="c", duplication_count=1, category="unique"),
            Topic(topic_id="t001", label="a", duplication_count=1, category="unique"),
            Topic(topic_id="t002", label="b", duplication_count=1, category="unique"),
        ]
        chapters = [
            ChapterPlan(
                index=1,
                label="ch1",
                category="unique",
                topic_ids=["t002", "t001"],
                source_videos=[],
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == ["t003"]


# =====================================================
# call_leader
# =====================================================


SAMPLE_LEADER_OUTPUT = json.dumps(
    {
        "moc": {
            "title": "Test Playlist ハンズオン",
            "body_markdown": "# Test Playlist ハンズオン\n\n## 章構成\n- [[01_基礎]]",
        },
        "chapters": [
            {
                "chapter_index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "source_video_ids": ["vid1", "vid2", "vid3"],
                "body_markdown": "> [!important]\n> コア概念\n\n## 概念定義\n\n本文。",
            }
        ],
    },
    ensure_ascii=False,
)


class TestCallLeader:
    def test_happy_path(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_LEADER_OUTPUT))[1],
        )

        videos = [_video("vid1", "t1"), _video("vid2", "t2"), _video("vid3", "t3")]
        bodies = ["b1", "b2", "b3"]
        topics = [Topic(topic_id="t001", label="x", duplication_count=3, category="core")]
        chapters = [
            ChapterPlan(index=1, label="ch1", category="core", topic_ids=["t001"], source_videos=[])
        ]
        coverage = CoverageReport(covered_topic_ids=["t001"], missing_topic_ids=[], notes="ok")

        leader_out, result = call_leader(
            videos,
            bodies,
            topics,
            chapters,
            coverage,
            playlist_title="Test Playlist",
        )

        assert leader_out.moc.title == "Test Playlist ハンズオン"
        assert len(leader_out.chapters) == 1
        assert leader_out.chapters[0].body_markdown.startswith("> [!important]")

        prompt = captured["prompt"]
        # Leader receives all 4 inputs: topics, chapters, coverage, materials
        assert "t001" in prompt
        assert "ch1" in prompt
        assert "## VIDEO: vid1: t1" in prompt
