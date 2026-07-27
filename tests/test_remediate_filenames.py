"""Tests for scripts/remediate_filenames.py (concealment remediation planner).

Concealment code points are built with ``chr()`` so this source stays free of
literal invisible glyphs.
"""

from __future__ import annotations

from pathlib import Path

from scripts.remediate_filenames import (
    RenamePlan,
    find_wikilink_refs,
    main,
    rewrite_wikilink_targets,
    scan,
)

ZW = chr(0x200B)  # ZERO WIDTH SPACE
CYR = chr(0x430)  # CYRILLIC SMALL LETTER A


def _note(vault: Path, stem: str, title: str) -> Path:
    folder = vault / "Permanent Note" / "08_YouTube学習" / "01_Scripts_Processing_Unit"
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.md"
    p.write_text(f'---\ntitle: "{title}"\n---\nbody\n', encoding="utf-8")
    return p


class TestScan:
    def test_invisible_note_becomes_rename_plan(self, tmp_path: Path):
        _note(tmp_path, f"2026 AI {ZW}{ZW}had", f"AI {ZW}{ZW}had")
        plans, homoglyphs = scan(tmp_path)
        assert len(plans) == 1
        plan = plans[0]
        assert isinstance(plan, RenamePlan)
        assert plan.new_stem == "2026 AI had"  # ZWSP stripped, space kept
        assert plan.invisible_code_points == ("U+200B", "U+200B")

    def test_homoglyph_note_flagged_not_renamed(self, tmp_path: Path):
        _note(tmp_path, f"2026 {CYR}pple", f"{CYR}pple")
        plans, homoglyphs = scan(tmp_path)
        assert plans == []  # never auto-rename a mixed-script title
        fields = {h.field for h in homoglyphs}
        assert fields == {"filename", "title"}
        assert all(h.tokens == (f"{CYR}pple",) for h in homoglyphs)

    def test_clean_note_not_flagged(self, tmp_path: Path):
        _note(tmp_path, "2026 The AI System", "The AI System")
        plans, homoglyphs = scan(tmp_path)
        assert plans == [] and homoglyphs == []

    def test_scope_limited_to_youtube_pipeline(self, tmp_path: Path):
        # a concealed note OUTSIDE 08_YouTube学習 must be ignored.
        other = tmp_path / "00_Diary" / f"2026 x{ZW}y.md"
        other.parent.mkdir(parents=True)
        other.write_text("---\ntitle: x\n---\n", encoding="utf-8")
        plans, homoglyphs = scan(tmp_path)
        assert plans == [] and homoglyphs == []


class TestWikilinkRefs:
    def test_finds_plain_and_embed_links(self, tmp_path: Path):
        stem = f"2026 AI {ZW}had"
        moc = tmp_path / "Permanent Note" / "MOC.md"
        moc.parent.mkdir(parents=True, exist_ok=True)
        moc.write_text(f"[[{stem}|alias]] and ![[{stem}#^12-30]]\n", encoding="utf-8")
        refs = find_wikilink_refs(tmp_path, {stem})
        assert moc in refs
        assert len(refs[moc]) == 2

    def test_finds_targets_containing_a_single_closing_bracket(self, tmp_path: Path):
        # YouTube titles often keep `[tag]` after sanitize_title_for_filename.
        # The old capture stopped at the first `]`, so these refs were invisible
        # to dry-run reporting and --apply rewrites.
        stem = f"2026-01-01-1200 [LLM] Agent{ZW}Teams"
        moc = tmp_path / "Permanent Note" / "MOC.md"
        moc.parent.mkdir(parents=True, exist_ok=True)
        moc.write_text(
            f"[[{stem}|alias]] and ![[{stem}#^00-30]]\n",
            encoding="utf-8",
        )
        refs = find_wikilink_refs(tmp_path, {stem})
        assert moc in refs
        assert refs[moc] == [f"[[{stem}|alias]]", f"![[{stem}#^00-30]]"]

    def test_no_refs_when_stem_absent(self, tmp_path: Path):
        (tmp_path / "note.md").write_text("[[unrelated]]\n", encoding="utf-8")
        assert find_wikilink_refs(tmp_path, {"missing stem"}) == {}


class TestRewriteWikilinkTargets:
    def test_preserves_prefix_alias_and_anchor(self):
        renamed = {"old stem": "new stem"}
        text = "[[old stem|alias]] and ![[old stem#^12-30]]"
        out = rewrite_wikilink_targets(text, renamed)
        assert out == "[[new stem|alias]] and ![[new stem#^12-30]]"

    def test_rewrites_targets_containing_a_single_closing_bracket(self):
        old = "2026-01-01-1200 [LLM] Agent\u200bTeams"
        new = "2026-01-01-1200 [LLM] AgentTeams"
        text = f"[[{old}|alias]] and ![[{old}#^00-30]]"
        assert rewrite_wikilink_targets(text, {old: new}) == (
            f"[[{new}|alias]] and ![[{new}#^00-30]]"
        )

    def test_target_equal_to_prefix_char_does_not_corrupt(self):
        # regression: a target that also occurs in the "![[" prefix must rewrite
        # only the target, never the link syntax (naive group(0).replace would
        # yield "SAFE[[SAFE]]").
        assert rewrite_wikilink_targets("![[!]]", {"!": "SAFE"}) == "![[SAFE]]"

    def test_unrenamed_target_untouched(self):
        assert rewrite_wikilink_targets("[[keep]]", {"x": "y"}) == "[[keep]]"


class TestApply:
    def test_rewrites_links_inside_notes_that_are_also_renamed(self, tmp_path: Path):
        old_a = _note(tmp_path, f"A{ZW}", "A")
        old_b = _note(tmp_path, f"B{ZW}", "B")
        old_a.write_text(f"[[B{ZW}]]\n", encoding="utf-8")
        old_b.write_text(f"[[A{ZW}]]\n", encoding="utf-8")

        assert main(["--vault", str(tmp_path), "--apply"]) == 0

        new_a = old_a.with_name("A.md")
        new_b = old_b.with_name("B.md")
        assert not old_a.exists()
        assert not old_b.exists()
        assert new_a.read_text(encoding="utf-8") == "[[B]]\n"
        assert new_b.read_text(encoding="utf-8") == "[[A]]\n"

    def test_rewrites_bracketed_title_refs_on_apply(self, tmp_path: Path):
        old_stem = f"2026-01-01-1200 [LLM] Agent{ZW}Teams"
        new_stem = "2026-01-01-1200 [LLM] AgentTeams"
        target = _note(tmp_path, old_stem, old_stem)
        ref = _note(tmp_path, "index", "index")
        ref.write_text(
            f"see [[{old_stem}|alias]] and ![[{old_stem}#^00-30]]\n",
            encoding="utf-8",
        )

        assert main(["--vault", str(tmp_path), "--apply"]) == 0

        assert not target.exists()
        assert (target.parent / f"{new_stem}.md").exists()
        assert ref.read_text(encoding="utf-8") == (
            f"see [[{new_stem}|alias]] and ![[{new_stem}#^00-30]]\n"
        )
