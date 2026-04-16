"""Prompt injection mitigation, ported from pipeline/classifier.ts:209-216.

Block-list regex patterns are easily bypassed, so the defense relies on:
  1. Structural sanitization (strip control chars, zero-width unicode)
  2. Strict length cap (reduce payload surface)
  3. XML delimiter wrapping (<untrusted_content>) + explicit prompt policy
  4. Downstream AI-output structural validation

This module handles (1), (2), (3). Downstream validation lives in the
AI provider modules.
"""

from __future__ import annotations

import re

# Keep \t (\x09) and \n (\x0a) — these are legitimate content.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zero-width and invisible Unicode: ZWSP, ZWNJ, ZWJ, LRM, RLM, LS, PS,
# various directional marks, word joiner, BOM, interlinear annotations.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff\ufff9-\ufffb]")

UNTRUSTED_OPEN = "<untrusted_content>"
UNTRUSTED_CLOSE = "</untrusted_content>"


def sanitize_untrusted_text(raw: str | None, max_length: int) -> str:
    """Strip control / zero-width / null bytes and cap length.

    Preserves tabs and newlines (legitimate in transcripts/summaries).
    """
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", raw)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = cleaned.replace("\x00", "")
    return cleaned[:max_length]


def wrap_untrusted(content: str) -> str:
    """Wrap sanitized content in <untrusted_content> delimiter for AI prompts.

    The prompt-side system policy must explicitly instruct the model to
    treat anything inside these tags as data, not instructions.
    """
    return f"{UNTRUSTED_OPEN}\n{content}\n{UNTRUSTED_CLOSE}"
