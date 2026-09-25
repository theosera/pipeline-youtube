"""Tests for obsidian.py note naming, frontmatter, and collision avoidance."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.checkpoint import get_completed_video_ids, is_video_complete
from pipeline_youtube.config import reset_vault_root, set_vault_root
from pipeline_youtube.obsidian import (
    _escape_yaml,
    build_frontmatter,
    format_playlist_folder_name,
    format_video_note_base,
    limit_title_for_path_component,
    playlist_folder_title,
    playlist_title_for_path,
    playlist_title_needles,
    resolve_unique_path,
    sanitize_title_for_filename,
)
from pipeline_youtube.pipeline import LEARNING_BASE, UNIT_DIRS
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.resume import _collect_existing_learning_bodies, _unit_folder_candidates
from pipeline_youtube.services.obsidian import _MAX_PATH_COMPONENT_BYTES


class TestSanitizeTitle:
    def test_simple(self):
        assert sanitize_title_for_filename("hello world") == "hello world"

    def test_unsafe_chars_replaced(self):
        assert sanitize_title_for_filename("foo/bar:baz") == "foo bar baz"

    def test_quotes_replaced(self):
        assert sanitize_title_for_filename('test "quoted" name') == "test quoted name"

    def test_all_unsafe_chars(self):
        raw = "a\\b/c:d*e?f<g>h|i"
        assert sanitize_title_for_filename(raw) == "a b c d e f g h i"

    def test_collapse_multiple_spaces(self):
        assert sanitize_title_for_filename("a   b\tc") == "a b c"

    def test_strip_edges(self):
        assert sanitize_title_for_filename("  hello  ") == "hello"

    def test_empty(self):
        assert sanitize_title_for_filename("") == ""

    def test_none(self):
        assert sanitize_title_for_filename(None) == ""

    def test_japanese_preserved(self):
        assert sanitize_title_for_filename("ハーネス設計") == "ハーネス設計"

    def test_mixed_jp_en(self):
        assert (
            sanitize_title_for_filename("Anthropicが公開したハーネス設計、全部解説します")
            == "Anthropicが公開したハーネス設計、全部解説します"
        )


class TestFormatVideoNoteBase:
    def test_with_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, "Test Video") == "2026-04-14-2141 Test Video"

    def test_empty_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, "") == "2026-04-14 2141"

    def test_none_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, None) == "2026-04-14 2141"

    def test_matches_dummy_data(self):
        """Must match the existing dummy-data filename in 08_YouTube学習."""
        dt = datetime(2026, 4, 14, 21, 41)
        result = format_video_note_base(dt, "Anthropicが公開したハーネス設計、全部解説します")
        assert result == "2026-04-14-2141 Anthropicが公開したハーネス設計、全部解説します"

    def test_unsafe_chars_in_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, "Slash/and:colon") == "2026-04-14-2141 Slash and colon"

    def test_zero_padded_time(self):
        dt = datetime(2026, 1, 2, 3, 5)
        assert format_video_note_base(dt, "Test") == "2026-01-02-0305 Test"


class TestFormatPlaylistFolder:
    def test_with_title(self):
        dt = datetime(2026, 4, 14, 13, 45)
        assert (
            format_playlist_folder_name(dt, "Harness Engineering")
            == "2026-04-14-1345 Harness Engineering"
        )

    def test_midnight_pads_zeros(self):
        dt = datetime(2026, 4, 14, 0, 0)
        assert (
            format_playlist_folder_name(dt, "Harness Engineering")
            == "2026-04-14-0000 Harness Engineering"
        )

    def test_empty_title(self):
        dt = datetime(2026, 4, 14, 9, 5)
        assert format_playlist_folder_name(dt, "") == "2026-04-14-0905"

    def test_none_title(self):
        dt = datetime(2026, 4, 14, 9, 5)
        assert format_playlist_folder_name(dt, None) == "2026-04-14-0905"

    def test_strips_ascii_slash_category_prefix(self):
        """`2026Agent Teams/AI駆動経営` -> drop category, keep `AI駆動経営` only."""
        dt = datetime(2026, 4, 16, 9, 14)
        assert (
            format_playlist_folder_name(dt, "2026Agent Teams/AI駆動経営")
            == "2026-04-16-0914 AI駆動経営"
        )

    def test_strips_multiple_slashes(self):
        dt = datetime(2026, 4, 16, 9, 14)
        assert format_playlist_folder_name(dt, "A/B/C Title") == "2026-04-16-0914 C Title"

    def test_fullwidth_slash_is_kept(self):
        """Full-width `／` is legitimate Japanese punctuation, not a separator."""
        dt = datetime(2026, 4, 16, 9, 14)
        assert (
            format_playlist_folder_name(dt, "Agent Teams／3 人編成")
            == "2026-04-16-0914 Agent Teams／3 人編成"
        )

    def test_long_cjk_title_stays_within_path_component_bytes(self):
        """YouTube allows 100-char titles; 80+ CJK chars exceed ext4's 255-byte cap."""
        dt = datetime(2026, 8, 5, 11, 0)
        long_title = "漢" * 100
        folder = format_playlist_folder_name(dt, long_title)
        note = format_video_note_base(dt, long_title)
        assert len(folder.encode("utf-8")) <= _MAX_PATH_COMPONENT_BYTES
        assert len(note.encode("utf-8")) <= _MAX_PATH_COMPONENT_BYTES
        # Collision suffix + .md must still fit under the OS 255-byte limit.
        assert len(f"{note}-99.md".encode()) <= 255

    def test_long_cjk_folder_is_mkdirable(self, tmp_path: Path):
        dt = datetime(2026, 8, 5, 11, 0)
        folder = tmp_path / format_playlist_folder_name(dt, "漢" * 100)
        folder.mkdir()
        assert folder.is_dir()
        note_path = resolve_unique_path(folder, format_video_note_base(dt, "あ" * 100), ".md")
        note_path.write_text("ok", encoding="utf-8")
        assert note_path.exists()

    def test_path_title_needle_matches_truncated_folder(self):
        """Resume/checkpoint needles must use the same title budget as format_*."""
        full = "漢" * 100
        limited = playlist_title_for_path(sanitize_title_for_filename(full))
        dt = datetime(2026, 8, 5, 11, 0)
        folder = format_playlist_folder_name(dt, full)
        assert limited
        assert limited == folder.split(" ", 1)[1]
        assert limited in playlist_title_needles(full)


# 70 CJK chars = 210 bytes: over the 184-byte title budget (200 - 16-byte
# "YYYY-MM-DD-HHmm " prefix) yet still a legal on-disk name before the cap
# existed (16 + 210 = 226 <= 255). A new folder keeps 58 chars (174 bytes)
# plus "~" and 8 hex of a digest: 174 + 9 = 183 bytes.
_LONG = "漢" * 70
# Shares _LONG's first 61 chars (all a plain 184-byte cut would keep), then
# differs — a different playlist.
_SAME_START = "漢" * 61 + "字" * 9
# A different playlist literally titled like _LONG's capped folder
# ("漢" * 58 + "~ee2b8cc6", 183 bytes): it fits, so it is not cut itself.
_LOOK_ALIKE = playlist_title_for_path(_LONG)
_VID_A = "abc123DEFGH"


@pytest.fixture()
def vault(tmp_path: Path):
    set_vault_root(tmp_path)
    yield tmp_path
    reset_vault_root()


def _learning_dir(vault: Path) -> Path:
    return vault / LEARNING_BASE / UNIT_DIRS["learning"]


def _write_04(folder: Path, video_id: str, body: str = "learning body") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "note.md").write_text(
        f'---\ntitle: "x"\nURL: "https://www.youtube.com/watch?v={video_id}"\n'
        f'video_id: "{video_id}"\n---\n\n{body}\n',
        encoding="utf-8",
    )


def _video(video_id: str, playlist_title: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title="t",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="ch",
        upload_date=None,
        playlist_title=playlist_title,
    )


class TestCappedTitleStaysMatchable:
    """A capped folder name must keep matching the exact-title lookups.

    `format_playlist_folder_name` caps a long title and adds a digest of the
    whole title; resume and checkpoint find a playlist's folders by checking
    `playlist_folder_title(name)` against `playlist_title_needles`. A long
    title finds the folders it wrote and the full-title ones written before the
    cap existed, and nothing that merely shares its start or is spelled like
    its capped form.
    """

    def test_folder_title_round_trips_capped_and_legacy_names(self):
        capped = format_playlist_folder_name(datetime(2026, 8, 5, 9, 0), _LONG)
        legacy = f"2026-08-05-0900 {_LONG}"
        needles = playlist_title_needles(_LONG)
        assert len(playlist_folder_title(capped).encode()) == 183
        assert playlist_folder_title(capped) in needles
        assert playlist_folder_title(legacy) == _LONG
        assert playlist_folder_title(legacy) in needles

    def test_checkpoint_finds_a_same_day_capped_folder(self, vault):
        morning = format_playlist_folder_name(datetime(2026, 8, 5, 9, 0), _LONG)
        _write_04(_learning_dir(vault) / morning, _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)  # canonical 1800 does not exist
        assert is_video_complete(_VID_A, _LONG, evening, vault_root=vault) is True
        assert get_completed_video_ids(_LONG, evening, vault_root=vault) == {_VID_A}

    def test_checkpoint_finds_a_legacy_uncapped_folder(self, vault):
        """Written before the cap: the full 70-char title is on disk."""
        _write_04(_learning_dir(vault) / f"2026-08-05-0900 {_LONG}", _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, _LONG, evening, vault_root=vault) is True

    def test_synthesis_only_accepts_its_own_capped_canonical_folder(self, vault):
        dt = datetime(2026, 8, 5, 9, 0)
        canonical = format_playlist_folder_name(dt, _LONG)
        _write_04(_learning_dir(vault) / canonical, _VID_A)
        videos, bodies, folder_name = _collect_existing_learning_bodies(
            [_video(_VID_A, _LONG)], _LONG, dt, vault_root=vault
        )
        assert [v.video_id for v in videos] == [_VID_A]
        assert bodies == ["learning body\n"]
        assert folder_name == canonical

    def test_phase3_finds_an_earlier_day_capped_folder(self, vault):
        base = _learning_dir(vault)
        yesterday = base / format_playlist_folder_name(datetime(2026, 8, 4, 21, 0), _LONG)
        yesterday.mkdir(parents=True)
        candidates = list(_unit_folder_candidates(base, _LONG, datetime(2026, 8, 5, 9, 0)))
        assert yesterday in candidates

    def test_titles_differing_within_the_cap_stay_apart(self, vault):
        """Capping must not loosen the exact-title rule inside the budget."""
        other = "漢" * 60 + "字" * 10  # differs at char 61, inside the cap
        _write_04(_learning_dir(vault) / f"2026-08-05-0900 {other}", _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, _LONG, evening, vault_root=vault) is False

    def test_note_names_are_cut_without_a_digest(self):
        """Notes cut to the same stem are kept apart by `-2`, not a digest."""
        note = format_video_note_base(datetime(2026, 8, 5, 9, 0), _LONG)
        assert note.split(" ", 1)[1] == limit_title_for_path_component(_LONG)
        assert "~" not in note

    def test_titles_sharing_the_kept_start_get_distinct_folders(self):
        """The digest keeps two long titles apart past the cut."""
        dt = datetime(2026, 8, 5, 9, 0)
        assert format_playlist_folder_name(dt, _SAME_START) != format_playlist_folder_name(
            dt, _LONG
        )
        assert playlist_title_needles(_SAME_START).isdisjoint(playlist_title_needles(_LONG))

    def test_checkpoint_ignores_a_new_folder_of_a_same_start_playlist(self, vault):
        """Another playlist's capped folder must not count as this one's run."""
        other = format_playlist_folder_name(datetime(2026, 8, 5, 9, 0), _SAME_START)
        _write_04(_learning_dir(vault) / other, _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, _LONG, evening, vault_root=vault) is False
        assert get_completed_video_ids(_LONG, evening, vault_root=vault) == set()

    def test_checkpoint_ignores_a_legacy_folder_of_a_same_start_playlist(self, vault):
        """A pre-cap folder is matched on its full title, not its first bytes."""
        _write_04(_learning_dir(vault) / f"2026-08-05-0900 {_SAME_START}", _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, _LONG, evening, vault_root=vault) is False

    def test_a_title_spelled_like_a_capped_form_gets_its_own_folder(self):
        dt = datetime(2026, 8, 5, 9, 0)
        folder = format_playlist_folder_name(dt, _LOOK_ALIKE)
        assert folder != format_playlist_folder_name(dt, _LONG)
        assert len(folder.encode()) <= _MAX_PATH_COMPONENT_BYTES
        assert playlist_title_needles(_LOOK_ALIKE).isdisjoint(playlist_title_needles(_LONG))

    def test_checkpoint_keeps_a_look_alike_out_of_the_long_titles_folder(self, vault):
        """That name is _LONG's folder; a pre-cap one of the look-alike's is given up."""
        _write_04(_learning_dir(vault) / f"2026-08-05-0900 {_LOOK_ALIKE}", _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, _LOOK_ALIKE, evening, vault_root=vault) is False
        assert get_completed_video_ids(_LOOK_ALIKE, evening, vault_root=vault) == set()

    def test_phase3_keeps_a_look_alike_out_of_the_long_titles_folder(self, vault):
        base = _learning_dir(vault)
        yesterday = base / format_playlist_folder_name(datetime(2026, 8, 4, 21, 0), _LONG)
        yesterday.mkdir(parents=True)
        candidates = list(_unit_folder_candidates(base, _LOOK_ALIKE, datetime(2026, 8, 5, 9, 0)))
        assert yesterday not in candidates

    def test_every_capped_form_is_escaped_when_used_as_a_title(self):
        """Capped forms span exactly 180-184 bytes, and each one is escaped.

        The kept head ends wherever the 175-byte cut lands: inside a 1-4 byte
        codepoint at any offset, with or without a space before it.
        """
        sizes = set()
        for kept in range(165, 176):
            for gap in ("", " "):
                for char in ("b", "é", "漢", "𠮷"):
                    title = "a" * kept + gap + char * 30
                    capped = playlist_title_for_path(title)
                    sizes.add(len(capped.encode()))
                    assert playlist_title_for_path(capped) != capped
                    assert playlist_title_needles(capped).isdisjoint(playlist_title_needles(title))
        assert (min(sizes), max(sizes)) == (180, 184)

    @pytest.mark.parametrize(
        "title",
        [
            "Release notes~deadbeef",  # far below 180 bytes
            "a" * 170 + "~deadbeef",  # 179 bytes: one under the shortest capped form
            "a" * 172 + " ~deadbeef",  # rstrip never leaves a space before "~"
            "a" * 173 + "~DEADBEEF",  # hexdigest is lower-case
        ],
        ids=["short", "179-bytes", "space-before-sep", "upper-hex"],
    )
    def test_a_title_no_capped_form_can_equal_keeps_its_name_and_folders(self, title, vault):
        folder = format_playlist_folder_name(datetime(2026, 8, 5, 9, 0), title)
        assert folder == f"2026-08-05-0900 {title}"
        _write_04(_learning_dir(vault) / folder, _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, title, evening, vault_root=vault) is True

    def test_a_long_title_ending_like_a_digest_keeps_its_pre_cap_folder(self, vault):
        """Over the budget it is cut anyway, so its full title stays a needle."""
        long_title = "漢" * 60 + "~deadbeef"  # 189 bytes
        _write_04(_learning_dir(vault) / f"2026-08-05-0900 {long_title}", _VID_A)
        evening = datetime(2026, 8, 5, 18, 0)
        assert is_video_complete(_VID_A, long_title, evening, vault_root=vault) is True


class TestResolveUniquePath:
    def test_fresh_folder(self, tmp_path: Path):
        assert resolve_unique_path(tmp_path, "note", ".md") == tmp_path / "note.md"

    def test_first_collision(self, tmp_path: Path):
        (tmp_path / "note.md").write_text("x")
        assert resolve_unique_path(tmp_path, "note", ".md") == tmp_path / "note-2.md"

    def test_multiple_collisions(self, tmp_path: Path):
        (tmp_path / "note.md").write_text("x")
        (tmp_path / "note-2.md").write_text("x")
        (tmp_path / "note-3.md").write_text("x")
        assert resolve_unique_path(tmp_path, "note", ".md") == tmp_path / "note-4.md"

    def test_nonexistent_folder(self, tmp_path: Path):
        folder = tmp_path / "does_not_exist"
        # Should still return the first candidate; caller is responsible for mkdir
        assert resolve_unique_path(folder, "note", ".md") == folder / "note.md"

    def test_custom_extension(self, tmp_path: Path):
        (tmp_path / "image.png").write_text("x")
        assert resolve_unique_path(tmp_path, "image", ".png") == tmp_path / "image-2.png"


class TestEscapeYaml:
    def test_plain(self):
        assert _escape_yaml("hello") == "hello"

    def test_quotes(self):
        assert _escape_yaml('a"b') == 'a\\"b'

    def test_backslash(self):
        assert _escape_yaml("a\\b") == "a\\\\b"

    def test_newline_to_space(self):
        assert _escape_yaml("a\nb") == "a b"

    def test_cr_removed(self):
        assert _escape_yaml("a\rb") == "ab"

    def test_yaml_separator_neutralized(self):
        assert _escape_yaml("foo---bar") == "foo\\-\\-\\-bar"

    def test_empty(self):
        assert _escape_yaml("") == ""

    def test_none(self):
        assert _escape_yaml(None) == ""


class TestBuildFrontmatter:
    def test_basic(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(dt, "Test", url="https://example.com")
        assert fm.startswith("---\n")
        assert "date: 2026-04-14 21:41\n" in fm
        assert 'title: "Test"\n' in fm
        assert 'URL: "https://example.com"\n' in fm
        assert "tags: [memo, youtube]\n" in fm
        assert fm.endswith("---\n")

    def test_yaml_escaping_in_title(self):
        dt = datetime(2026, 1, 1, 0, 0)
        fm = build_frontmatter(dt, 'Title with "quotes"', url="")
        assert 'title: "Title with \\"quotes\\""' in fm

    def test_extra_fields(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(
            dt,
            "Test",
            url="",
            extra={"playlist": "Harness Engineering", "video_id": "abc123"},
        )
        assert 'playlist: "Harness Engineering"' in fm
        assert 'video_id: "abc123"' in fm

    def test_custom_tags(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(dt, "Test", tags=["custom", "another"])
        assert "tags: [custom, another]" in fm

    def test_empty_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(dt, None)
        assert 'title: ""' in fm

    def test_invisible_chars_stripped_from_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        rlo, zwsp = chr(0x202E), chr(0x200B)
        fm = build_frontmatter(dt, f"clean{zwsp}{rlo}title")
        assert 'title: "cleantitle"' in fm
        assert rlo not in fm and zwsp not in fm

    def test_mixed_script_title_preserved_in_frontmatter(self):
        # Homoglyph tokens are reported at the fetch boundary, never rewritten
        # here — the title stays byte-for-byte in the note metadata.
        dt = datetime(2026, 4, 14, 21, 41)
        cyr_a = chr(0x430)
        fm = build_frontmatter(dt, f"{cyr_a}pple")
        assert f'title: "{cyr_a}pple"' in fm

    def test_invisible_chars_stripped_from_extra_values(self):
        # `extra` values (e.g. the raw playlist title) are attacker-controlled
        # too and must be cleaned, not just the `title` field.
        dt = datetime(2026, 4, 14, 21, 41)
        zwsp, rlo = chr(0x200B), chr(0x202E)
        fm = build_frontmatter(dt, "T", extra={"playlist": f"My{zwsp}{rlo}List", "video_id": "v1"})
        assert 'playlist: "MyList"' in fm
        assert zwsp not in fm and rlo not in fm


class TestFilenameConcealment:
    """Invisible/bidi/zero-width chars must never reach an on-disk filename;
    mixed-script homoglyph letters are preserved (detection is upstream)."""

    def test_rlo_override_stripped(self):
        rlo = chr(0x202E)
        assert rlo not in sanitize_title_for_filename(f"report{rlo}gpj.exe")
        assert sanitize_title_for_filename(f"report{rlo}gpj.exe") == "reportgpj.exe"

    def test_zero_width_family_stripped(self):
        zwsp, zwnj, bom = chr(0x200B), chr(0x200C), chr(0xFEFF)
        assert sanitize_title_for_filename(f"a{zwsp}b{zwnj}c{bom}d") == "abcd"

    def test_tab_still_collapses_to_space(self):
        # regression: stripping invisibles must not disturb \t -> space.
        assert sanitize_title_for_filename("a   b\tc") == "a b c"

    def test_japanese_and_fullwidth_solidus_preserved(self):
        raw = "Agent Teams／ハーネス設計"
        assert sanitize_title_for_filename(raw) == raw

    def test_mixed_script_letters_preserved(self):
        cyr_a = chr(0x430)
        assert sanitize_title_for_filename(f"{cyr_a}pple") == f"{cyr_a}pple"

    def test_format_video_note_base_strips_invisibles(self):
        rlo = chr(0x202E)
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, f"Ti{rlo}tle") == "2026-04-14-2141 Title"
