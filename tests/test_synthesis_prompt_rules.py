"""Prompt-content regression tests for the P1-P5 synthesis improvements.

These lock in the text of specific instructions in the α / β / Leader
system prompts so future edits don't silently drop them. Real LLM
behavior is out of scope — we only verify the instructions are *present*
and unambiguous.

P1: 核心要素 に出典必須化 (Leader)
P2: 矢印圧縮禁止 (Leader)
P3: 章あたり最低 5 トピック (β)
P4: MOC に概念別索引テーブル (Leader)
P5: 学習順序は時間別コース (Leader)
"""

from __future__ import annotations

from pipeline_youtube.synthesis.agents import (
    BETA_SYSTEM_PROMPT,
    LEADER_SYSTEM_PROMPT,
)


class TestP1InlineCitations:
    def test_core_elements_require_inline_citations(self):
        """Leader prompt must force `(出典: [[...]])` on every 核心要素 item."""
        assert "核心要素" in LEADER_SYSTEM_PROMPT
        # The exact phrasing must include both `各項目末尾` (every item's end)
        # and the citation template so claude can't weasel out to "only 1-2
        # items get citations".
        assert "各項目末尾" in LEADER_SYSTEM_PROMPT
        assert "出典: [[<動画 note 名>#^MM-SS]]" in LEADER_SYSTEM_PROMPT


class TestP2ArrowExpansion:
    def test_arrow_chains_must_be_expanded(self):
        """Leader prompt must forbid `A→B→C→D` style step compression."""
        # Looking for the 3-step arrow ban instruction verbatim.
        assert "工程列挙の展開" in LEADER_SYSTEM_PROMPT
        assert "矢印" in LEADER_SYSTEM_PROMPT
        assert "3 ステップ以上" in LEADER_SYSTEM_PROMPT
        assert "独立した箇条書き" in LEADER_SYSTEM_PROMPT


class TestP3MinTopicsPerChapter:
    def test_beta_requires_five_topics_per_chapter(self):
        """β must refuse to emit chapters with < 5 topics, even for `unique`."""
        assert "5 トピック" in BETA_SYSTEM_PROMPT
        # Explicitly mentions unique so the model doesn't excuse thin
        # single-video chapters.
        assert "unique 章でも 5 以上" in BETA_SYSTEM_PROMPT


class TestP4ConceptIndexInMoc:
    def test_moc_must_include_concept_index_table(self):
        """MOC gets a `## 概念別索引` table so readers can cross-look topics."""
        assert "## 概念別索引" in LEADER_SYSTEM_PROMPT
        # Table columns must be explicit; a prose list doesn't count.
        assert "| 概念 | 章 |" in LEADER_SYSTEM_PROMPT


class TestP5LearningPaths:
    def test_learning_section_lists_three_courses(self):
        """`## 学習順序の推奨` must split into 3 reader-intent courses."""
        assert "全章通読コース" in LEADER_SYSTEM_PROMPT
        assert "30 分で要点把握コース" in LEADER_SYSTEM_PROMPT
        assert "深掘りコース" in LEADER_SYSTEM_PROMPT


class TestExistingConstraintsIntact:
    """Guard against regression of earlier prompt guarantees."""

    def test_leader_still_forbids_hallucinated_images(self):
        assert "新規ファイル名創作禁止" in LEADER_SYSTEM_PROMPT

    def test_leader_still_requires_json_only(self):
        assert "JSON 単体" in LEADER_SYSTEM_PROMPT

    def test_beta_still_limits_title_filename_chars(self):
        # The Obsidian-unsafe set must stay filtered.
        for ch in ("\\", "/", ":", "*", "?", '"', "<", ">", "|"):
            assert ch in BETA_SYSTEM_PROMPT
