#!/usr/bin/env python3
"""Dry-run remediation planner for concealment in youtube-pipeline note names.

Walks a LOCAL Obsidian vault's youtube-pipeline output
(``Permanent Note/08_YouTube学習``), reuses ``pipeline_youtube.services``
(the same detector + naming rules as the write-time defense) to find
invisible / homoglyph concealment in each note's filename and its frontmatter
``title:`` field, and prints a remediation plan:

  * **invisible chars** (zero-width, bidi, RLO, ...) -> a concrete rename
    proposal ``<old> -> <cleaned>`` (safe auto-fix: just strip the invisibles);
  * **mixed-script homoglyph tokens** -> reported for HUMAN review only, never
    auto-renamed (a legitimately Cyrillic/Greek title must not be corrupted);
  * every ``[[wikilink]]`` / ``![[embed]]`` elsewhere in the vault that points
    at a to-be-renamed note, so the operator knows exactly what would break.

DRY-RUN by default — it renames nothing and writes nothing. Pass ``--apply`` to
perform the renames (collision-safe ``-2``/``-3`` suffixing) and rewrite the
reported wikilink references in place.

Why local: the vault MCP connector exposes create/update but not rename/move,
so filename remediation cannot run through it — this script operates directly
on the vault on disk. Point ``--vault`` at your vault root.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the package importable when run from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_youtube.services.confusables import (  # noqa: E402
    analyze_filename_text,
    strip_invisibles,
)
from pipeline_youtube.services.obsidian import (  # noqa: E402
    read_frontmatter_field,
    resolve_unique_path,
)

# youtube-pipeline output root (mirrors pipeline.LEARNING_BASE). Scope is limited
# to this subtree — the remediation never touches the rest of the vault.
_LEARNING_BASE = "Permanent Note/08_YouTube学習"
# Match [[target]], [[target|alias]], [[target#anchor]] and the ![[...]] embed
# form. Group 1 is the raw target (up to | # or ]]).
_WIKILINK_RE = re.compile(r"!?\[\[([^\]|#\n]+)")


@dataclass(frozen=True)
class RenamePlan:
    path: Path
    old_stem: str
    new_stem: str
    invisible_code_points: tuple[str, ...]


@dataclass(frozen=True)
class HomoglyphFlag:
    path: Path
    field: str  # "filename" or "title"
    tokens: tuple[str, ...]


def _removed_code_points(raw: str, cleaned: str) -> tuple[str, ...]:
    """The U+XXXX code points strip_invisibles removed from ``raw`` (in order)."""
    cursor = 0
    removed: list[str] = []
    for ch in raw:
        if cursor < len(cleaned) and cleaned[cursor] == ch:
            cursor += 1
        else:
            removed.append(f"U+{ord(ch):04X}")
    return tuple(removed)


def scan(vault_root: Path) -> tuple[list[RenamePlan], list[HomoglyphFlag]]:
    """Scan youtube-pipeline notes; return (rename plans, homoglyph flags)."""
    base = vault_root / _LEARNING_BASE
    plans: list[RenamePlan] = []
    homoglyphs: list[HomoglyphFlag] = []
    if not base.exists():
        return plans, homoglyphs

    for md in sorted(base.rglob("*.md")):
        stem = md.stem
        cleaned_stem, removed = strip_invisibles(stem)
        if removed and cleaned_stem != stem:
            plans.append(
                RenamePlan(
                    path=md,
                    old_stem=stem,
                    new_stem=cleaned_stem,
                    invisible_code_points=_removed_code_points(stem, cleaned_stem),
                )
            )
        # Mixed-script detection on the filename and the frontmatter title.
        for field, value in (
            ("filename", stem),
            ("title", read_frontmatter_field(md, "title") or ""),
        ):
            report = analyze_filename_text(value)
            if report.mixed_script_tokens:
                homoglyphs.append(
                    HomoglyphFlag(path=md, field=field, tokens=report.mixed_script_tokens)
                )
    return plans, homoglyphs


def find_wikilink_refs(vault_root: Path, stems: set[str]) -> dict[Path, list[str]]:
    """Return {md_path: [referencing lines]} for links whose target is in stems."""
    refs: dict[Path, list[str]] = {}
    if not stems:
        return refs
    for md in sorted(vault_root.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits: list[str] = []
        for m in _WIKILINK_RE.finditer(text):
            if m.group(1).strip() in stems:
                hits.append(m.group(0))
        if hits:
            refs[md] = hits
    return refs


def _rel(vault_root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(vault_root))
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path, help="Obsidian vault root")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files and rewrite wikilinks (default: dry-run).",
    )
    args = parser.parse_args(argv)

    vault_root = args.vault.expanduser().resolve()
    if not vault_root.is_dir():
        print(f"error: vault root not found: {vault_root}", file=sys.stderr)
        return 2

    plans, homoglyphs = scan(vault_root)
    old_stems = {p.old_stem for p in plans}
    refs = find_wikilink_refs(vault_root, old_stems)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"# youtube-pipeline concealment remediation ({mode})")
    print(f"# vault: {vault_root}")
    print(f"# scope: {_LEARNING_BASE}\n")

    print(f"## invisible-char renames ({len(plans)})")
    for p in plans:
        print(f"- {_rel(vault_root, p.path)}")
        print(f"    strip {list(p.invisible_code_points)}  ->  {p.new_stem}.md")

    print(f"\n## mixed-script homoglyphs — MANUAL review, not auto-renamed ({len(homoglyphs)})")
    for h in homoglyphs:
        print(f"- [{h.field}] {_rel(vault_root, h.path)}  tokens={list(h.tokens)}")

    ref_count = sum(len(v) for v in refs.values())
    print(f"\n## wikilink references to renamed notes ({ref_count} in {len(refs)} files)")
    for md, hits in refs.items():
        print(f"- {_rel(vault_root, md)}: {hits}")

    if not args.apply:
        print("\n[dry-run] nothing changed. Re-run with --apply to rename + rewrite.")
        return 0

    # --apply: rename each file (collision-safe) and rewrite wikilink targets.
    renamed: dict[str, str] = {}
    for p in plans:
        dest = resolve_unique_path(p.path.parent, p.new_stem, p.path.suffix)
        p.path.rename(dest)
        renamed[p.old_stem] = dest.stem
        print(f"[renamed] {_rel(vault_root, p.path)} -> {dest.name}")

    for md, _hits in refs.items():
        text = md.read_text(encoding="utf-8", errors="ignore")

        def _sub(m: re.Match[str]) -> str:
            target = m.group(1).strip()
            new = renamed.get(target)
            return m.group(0).replace(target, new) if new else m.group(0)

        md.write_text(_WIKILINK_RE.sub(_sub, text), encoding="utf-8")
        print(f"[rewrote links] {_rel(vault_root, md)}")

    print(f"\n[apply] {len(plans)} renamed, {len(refs)} files re-linked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
