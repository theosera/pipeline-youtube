"""Tests for scripts/remediate_filenames.py (concealment remediation planner).

Concealment code points are built with ``chr()`` so this source stays free of
literal invisible glyphs.
"""

from __future__ import annotations

from pathlib import Path

from scripts.remediate_filenames import RenamePlan, find_wikilink_refs, scan

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

    def test_no_refs_when_stem_absent(self, tmp_path: Path):
        (tmp_path / "note.md").write_text("[[unrelated]]\n", encoding="utf-8")
        assert find_wikilink_refs(tmp_path, {"missing stem"}) == {}
