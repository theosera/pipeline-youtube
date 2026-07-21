#!/usr/bin/env python3
"""Deterministic pre-filter for the homoglyph-audit skill.

Reuses the single source of truth — ``pipeline_youtube.services.confusables`` —
so the audit's ground-truth signal can never drift from the write-time defense
shipped in the pipeline. This is what makes the audit an *external* check with
the *same* detector, not a second, divergent implementation.

I/O: reads candidate records as JSONL on stdin (one object per line, each with
a ``"path"`` and any of ``"filename"`` / ``"title"``). For every record that
carries a concealment signal it prints one JSONL line with the deterministic
findings; records with no signal are dropped, so the LLM triage step only ever
sees the flagged subset (cheaper, and a smaller prompt-injection surface).

Read-only: computes signals and prints JSON. It never touches the vault, the
network, or any file other than stdin. It performs no vault mutation — renaming
or fixing anything is a separate, human-authorized step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make pipeline_youtube importable regardless of cwd. This file lives at
# <repo>/.claude/skills/homoglyph-audit/prefilter.py, so the repo root is three
# parents up. Importing the real module (not a copy) keeps the detector DRY.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline_youtube.services.confusables import (  # noqa: E402
    analyze_filename_text,
    strip_invisibles,
)


def _removed_code_points(raw: str, cleaned: str) -> list[str]:
    """Return the U+XXXX code points strip_invisibles removed from ``raw``.

    strip_invisibles only deletes chars in place (never reorders or adds), so a
    single forward walk recovers exactly which invisible/control chars were
    stripped — no benign visible chars (Japanese, typography) are listed.
    """
    cursor = 0
    removed: list[str] = []
    for ch in raw:
        if cursor < len(cleaned) and cleaned[cursor] == ch:
            cursor += 1
        else:
            removed.append(f"U+{ord(ch):04X}")
    return removed


def _analyze_field(value: str | None) -> dict[str, object] | None:
    report = analyze_filename_text(value)
    if not report.has_signal:
        return None
    cleaned, _ = strip_invisibles(value or "")
    return {
        "invisible_removed": report.invisible_removed,
        "invisible_code_points": _removed_code_points(value or "", cleaned),
        "mixed_script_tokens": list(report.mixed_script_tokens),
    }


def main() -> int:
    flagged = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        findings: dict[str, object] = {}
        for field in ("filename", "title"):
            got = _analyze_field(record.get(field))
            if got is not None:
                findings[field] = got
        if not findings:
            continue
        flagged += 1
        print(json.dumps({"path": record.get("path"), "findings": findings}, ensure_ascii=False))
    print(f"# homoglyph-audit prefilter: {flagged} flagged record(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
