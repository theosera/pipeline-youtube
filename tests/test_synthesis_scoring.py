"""Tests for synthesis.scoring (pure functions, no claude calls)."""

from __future__ import annotations

import json

import pytest

from pipeline_youtube.synthesis.scoring import (
    Category,
    ChapterPlan,
    CoverageReport,
    LeaderOutput,
    SynthesisParseError,
    Topic,
    derive_category,
    extract_json,
    parse_alpha_topics,
    parse_beta_chapters,
    parse_gamma_coverage,
    parse_leader_output,
)


class TestDeriveCategory:
    def test_core_three_or_more(self):
        assert derive_category(3) == "core"
        assert derive_category(5) == "core"

    def test_supporting_exactly_two(self):
        assert derive_category(2) == "supporting"

    def test_unique_one_or_zero(self):
        assert derive_category(1) == "unique"
        assert derive_category(0) == "unique"


class TestExtractJson:
    def test_strict_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_json_with_prose_prefix(self):
        raw = 'ここに JSON を返します:\n{"a": 1, "b": [2, 3]}'
        assert extract_json(raw) == {"a": 1, "b": [2, 3]}

    def test_json_with_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert extract_json(raw) == {"a": 1}

    def test_json_with_trailing_prose(self):
        raw = '{"a": 1}\n\n以上です。'
        assert extract_json(raw) == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(SynthesisParseError, match="empty"):
            extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(SynthesisParseError, match="no JSON"):
            extract_json("just prose, no object at all")


SAMPLE_ALPHA_JSON = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "コンテキスト不安",
                "aliases": ["context anxiety"],
                "source_videos": ["vid1", "vid2", "vid3"],
                "duplication_count": 3,
                "category": "core",
                "summary": "AI が焦ってタスクを強引にまとめる現象。",
                "excerpts": [
                    {"video_id": "vid1", "range": "[01:56 ~ 03:32]", "quote": "..."},
                ],
            },
            {
                "topic_id": "t002",
                "label": "GAN 方式",
                "source_videos": ["vid1", "vid4"],
                "duplication_count": 2,
                "category": "supporting",
                "summary": "生成と評価を分離する。",
            },
            {
                "topic_id": "t003",
                "label": "個別実験",
                "source_videos": ["vid4"],
                "duplication_count": 1,
                "category": "unique",
                "summary": "特定動画のみの話題。",
            },
        ]
    },
    ensure_ascii=False,
)


class TestParseAlphaTopics:
    def test_parses_three_topics(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        assert len(topics) == 3

    def test_core_topic_fields(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        t = topics[0]
        assert t.topic_id == "t001"
        assert t.label == "コンテキスト不安"
        assert t.aliases == ["context anxiety"]
        assert t.source_videos == ["vid1", "vid2", "vid3"]
        assert t.duplication_count == 3
        assert t.category == "core"
        assert len(t.excerpts) == 1
        assert t.excerpts[0].range_str == "[01:56 ~ 03:32]"

    def test_supporting_topic(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        t = topics[1]
        assert t.category == "supporting"
        assert t.duplication_count == 2

    def test_unique_topic(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        t = topics[2]
        assert t.category == "unique"

    def test_missing_category_derived_from_count(self):
        raw = json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "t001",
                        "label": "no category",
                        "source_videos": ["v1", "v2"],
                        # category omitted
                    }
                ]
            }
        )
        topics = parse_alpha_topics(raw)
        assert topics[0].category == "supporting"
        assert topics[0].duplication_count == 2

    def test_invalid_category_falls_back(self):
        raw = json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "t001",
                        "label": "bad cat",
                        "category": "nonsense",
                        "source_videos": ["v1"],
                        "duplication_count": 1,
                    }
                ]
            }
        )
        topics = parse_alpha_topics(raw)
        assert topics[0].category == "unique"

    def test_malformed_topics_field(self):
        raw = json.dumps({"topics": "not a list"})
        with pytest.raises(SynthesisParseError, match="topics must be a list"):
            parse_alpha_topics(raw)

    def test_empty_topics_list(self):
        assert parse_alpha_topics('{"topics": []}') == []


class TestParseBetaChapters:
    def test_parses_chapters(self):
        raw = json.dumps(
            {
                "chapters": [
                    {
                        "index": 1,
                        "label": "Chapter One",
                        "category": "core",
                        "topic_ids": ["t001", "t002"],
                        "source_videos": ["vid1"],
                        "rationale": "because",
                    }
                ]
            }
        )
        chapters = parse_beta_chapters(raw)
        assert len(chapters) == 1
        c = chapters[0]
        assert c.index == 1
        assert c.label == "Chapter One"
        assert c.category == "core"
        assert c.topic_ids == ["t001", "t002"]
        assert c.rationale == "because"

    def test_index_defaults_to_position(self):
        raw = json.dumps({"chapters": [{"label": "A"}, {"label": "B"}]})
        chapters = parse_beta_chapters(raw)
        assert chapters[0].index == 1
        assert chapters[1].index == 2

    def test_invalid_category_falls_back_to_unique(self):
        raw = json.dumps({"chapters": [{"label": "x", "category": "bogus"}]})
        chapters = parse_beta_chapters(raw)
        assert chapters[0].category == "unique"


class TestParseGammaCoverage:
    def test_parses_coverage(self):
        raw = json.dumps(
            {
                "covered_topic_ids": ["t001", "t002"],
                "missing_topic_ids": ["t003"],
                "notes": "コア概念の順序を改善すべき",
            }
        )
        report = parse_gamma_coverage(raw)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == ["t003"]
        assert "コア概念" in report.notes


class TestParseLeaderOutput:
    def test_parses_moc_and_chapters(self):
        raw = json.dumps(
            {
                "moc": {
                    "title": "Test Playlist ハンズオン",
                    "body_markdown": "# Test Playlist ハンズオン\n\n## 章構成",
                },
                "chapters": [
                    {
                        "chapter_index": 1,
                        "label": "基礎概念",
                        "category": "core",
                        "source_video_ids": ["vid1", "vid2"],
                        "body_markdown": "> [!important]\n## 概念定義\n\n...",
                    },
                    {
                        "chapter_index": 2,
                        "label": "応用編",
                        "category": "supporting",
                        "source_video_ids": ["vid3"],
                        "body_markdown": "## 応用\n\n...",
                    },
                ],
            },
            ensure_ascii=False,
        )
        out = parse_leader_output(raw)
        assert isinstance(out, LeaderOutput)
        assert out.moc.title == "Test Playlist ハンズオン"
        assert len(out.chapters) == 2
        assert out.chapters[0].category == "core"
        assert out.chapters[0].body_markdown.startswith("> [!important]")

    def test_malformed_moc_raises(self):
        raw = json.dumps({"moc": "not a dict", "chapters": []})
        with pytest.raises(SynthesisParseError, match="moc"):
            parse_leader_output(raw)

    def test_malformed_chapters_raises(self):
        raw = json.dumps({"moc": {"title": "x", "body_markdown": "y"}, "chapters": "bogus"})
        with pytest.raises(SynthesisParseError, match="chapters"):
            parse_leader_output(raw)
