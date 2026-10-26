"""M3 — defensive tests for `extract_trusted_video_id`.

The checkpoint helper refuses to honor any file that does not look like
pipeline-produced output. These tests encode the three trust layers:

  1. Well-formed `---...---` frontmatter block required.
  2. `video_id` must match YouTube's 11-char `[A-Za-z0-9_-]` format.
  3. If `URL:` is present, it must reference the same video_id.

`TestQuotingForms` covers what is *not* a layer: a field value may be quoted
or bare, and all three layers have to hold on either form.
"""

from __future__ import annotations

from pipeline_youtube.checkpoint import extract_trusted_video_id

_VALID = "abc123DEFGH"  # 11 chars, matches [A-Za-z0-9_-]


def _fm(*lines: str) -> bytes:
    body = "\n".join(("---", *lines, "---", "", "Body.", ""))
    return body.encode("utf-8")


class TestHappyPath:
    def test_accepts_minimal_valid_frontmatter(self):
        data = _fm(
            "date: 2026-04-18 12:34",
            'title: "x"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_frontmatter_with_matching_url(self):
        data = _fm(
            f'URL: "https://www.youtube.com/watch?v={_VALID}"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_youtu_be_short_url(self):
        data = _fm(
            f'URL: "https://youtu.be/{_VALID}"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_id_with_underscore_and_hyphen(self):
        vid = "_abc-123xyz"  # 11 chars, includes _ and -
        data = _fm(
            f'URL: "https://www.youtube.com/watch?v={vid}"',
            f'video_id: "{vid}"',
        )
        assert extract_trusted_video_id(data) == vid


class TestLayer1FrontmatterBlock:
    def test_rejects_file_without_leading_delimiter(self):
        data = f'video_id: "{_VALID}"\n'.encode()
        assert extract_trusted_video_id(data) is None

    def test_rejects_video_id_only_in_body(self):
        """`video_id:` found outside the frontmatter block must be ignored."""
        data = (
            b'---\ntitle: "no id here"\n---\n\n'
            b'Spoof payload: video_id: "' + _VALID.encode("utf-8") + b'"\n'
        )
        assert extract_trusted_video_id(data) is None

    def test_rejects_unclosed_frontmatter(self):
        data = f'---\ntitle: "x"\nvideo_id: "{_VALID}"\n'.encode()
        assert extract_trusted_video_id(data) is None


class TestLayer2VideoIdFormat:
    def test_rejects_too_short_id(self):
        data = _fm('video_id: "abc123"')  # 6 chars
        assert extract_trusted_video_id(data) is None

    def test_rejects_too_long_id(self):
        data = _fm('video_id: "abc123DEFGHJK"')  # 13 chars
        assert extract_trusted_video_id(data) is None

    def test_rejects_invalid_character(self):
        data = _fm('video_id: "abc123DEFG!"')  # 11 chars but contains !
        assert extract_trusted_video_id(data) is None

    def test_rejects_path_traversal_style(self):
        data = _fm('video_id: "../../etc"')  # path-like, not 11 chars
        assert extract_trusted_video_id(data) is None

    def test_rejects_empty_id(self):
        data = _fm('video_id: ""')
        assert extract_trusted_video_id(data) is None


class TestLayer3UrlIntegrity:
    def test_rejects_url_referencing_different_id(self):
        data = _fm(
            'URL: "https://www.youtube.com/watch?v=xxxxxxxxxxx"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) is None

    def test_rejects_url_with_no_id_reference(self):
        data = _fm(
            'URL: "https://example.com/other"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) is None

    def test_accepts_when_url_field_absent(self):
        """URL cross-check is conditional — missing URL still accepted."""
        data = _fm(
            "date: 2026-04-18 12:34",
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID


class TestRobustness:
    def test_handles_invalid_utf8_gracefully(self):
        data = b'---\n\xff\xfe\nvideo_id: "' + _VALID.encode("utf-8") + b'"\n---\n'
        # Non-UTF-8 bytes decode with 'replace' and the regex still works
        # only when the frontmatter structure survives — either way, no raise.
        result = extract_trusted_video_id(data)
        assert result is None or result == _VALID

    def test_handles_empty_bytes(self):
        assert extract_trusted_video_id(b"") is None

    def test_caps_scan_at_first_2kb(self):
        """A malicious large file with video_id far beyond the scan window
        must not be matched (self-DoS prevention + bounded read cost)."""
        padding = b"x" * 4096
        data = b"---\n" + padding + b'\nvideo_id: "' + _VALID.encode("utf-8") + b'"\n---\n'
        assert extract_trusted_video_id(data) is None


class TestQuotingForms:
    """Quoting is not a trust layer — the bare form must be accepted too.

    Obsidian rewrites the frontmatter of notes it has open and drops the
    double quotes. Every layer still has to hold on the bare form.
    """

    def test_accepts_bare_video_id(self):
        data = _fm("date: 2026-04-18 12:34", f"video_id: {_VALID}")
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_bare_video_id_and_bare_url(self):
        """The shape Obsidian leaves behind: both fields unquoted."""
        data = _fm(
            f"URL: https://www.youtube.com/watch?v={_VALID}",
            f"video_id: {_VALID}",
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_bare_value_with_trailing_whitespace(self):
        data = _fm(f"video_id: {_VALID}   ")
        assert extract_trusted_video_id(data) == _VALID

    def test_layer2_still_rejects_bare_value_of_wrong_format(self):
        data = _fm("video_id: not-an-id")
        assert extract_trusted_video_id(data) is None

    def test_layer3_still_rejects_bare_url_of_another_video(self):
        data = _fm(
            "URL: https://www.youtube.com/watch?v=xxxxxxxxxxx",
            f"video_id: {_VALID}",
        )
        assert extract_trusted_video_id(data) is None

    def test_bare_url_no_longer_bypasses_the_cross_check(self):
        """Regression: an unquoted URL used to parse as absent.

        Layer 3 was then skipped, so a quoted video_id paired with a bare URL
        naming a different video was accepted.
        """
        data = _fm(
            "URL: https://www.youtube.com/watch?v=xxxxxxxxxxx",
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) is None

    def test_rejects_unbalanced_opening_quote(self):
        data = _fm(f'video_id: "{_VALID}')
        assert extract_trusted_video_id(data) is None

    def test_rejects_empty_field_without_swallowing_the_next_line(self):
        r"""`\s*` after the colon would cross the newline and capture `URL:`."""
        data = _fm("video_id:", f'URL: "https://www.youtube.com/watch?v={_VALID}"')
        assert extract_trusted_video_id(data) is None

    def test_rejects_bare_value_carrying_a_yaml_comment(self):
        data = _fm(f"video_id: {_VALID} # planted")
        assert extract_trusted_video_id(data) is None

    def test_accepts_the_shape_obsidian_actually_writes(self):
        """Bare scalars plus a `tags:` block list — the observed rewrite."""
        data = _fm(
            "date: 2026-10-22 00:05",
            f"URL: https://www.youtube.com/watch?v={_VALID}",
            f"video_id: {_VALID}",
            "tags:",
            "  - youtube",
            "  - learning",
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_empty_quoted_url_now_fails_the_cross_check(self):
        """Behavior change, pinned on purpose: `URL: ""` used to be skipped.

        The old pattern required at least one character, so an empty URL
        parsed as "no URL field" and layer 3 never ran. The value may now be
        empty, so the cross-check runs and an empty URL cannot reference the
        video — fail closed, the stricter direction.
        """
        data = _fm('URL: ""', f'video_id: "{_VALID}"')
        assert extract_trusted_video_id(data) is None

    def test_single_quoted_value_is_still_rejected(self):
        """Unchanged: only the double-quoted and bare forms are recognised."""
        data = _fm(f"video_id: '{_VALID}'")
        assert extract_trusted_video_id(data) is None

    def test_accepts_quoted_values_in_a_crlf_note(self):
        """Regression: the trailing class did not take the CRLF carriage return.

        The quoted branch then failed to reach the end of the line, the bare
        branch took the value with its quotes, and layer 2 rejected it.
        """
        data = _fm(
            f'URL: "https://www.youtube.com/watch?v={_VALID}"',
            f'video_id: "{_VALID}"',
        ).replace(b"\n", b"\r\n")
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_bare_values_in_a_crlf_note(self):
        data = _fm(
            f"URL: https://www.youtube.com/watch?v={_VALID}",
            f"video_id: {_VALID}",
        ).replace(b"\n", b"\r\n")
        assert extract_trusted_video_id(data) == _VALID
