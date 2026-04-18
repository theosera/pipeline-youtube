"""Tests for #1: claude binary hijack mitigation (absolute-path pinning + --version probe)."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_youtube.providers import claude_cli as mod
from pipeline_youtube.providers.claude_cli import ClaudeBinaryError, _resolve_claude_binary


@pytest.fixture(autouse=True)
def _reset_cache():
    mod._reset_claude_binary_cache_for_tests()
    yield
    mod._reset_claude_binary_cache_for_tests()


def _make_fake_claude(tmp_path: Path, stdout: str = "claude 2.1.109") -> Path:
    """Write a bash stub that prints `stdout` and exits 0."""
    fake = tmp_path / "claude"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "{stdout}"\n', encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return fake


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-style stubs only")
class TestEnvOverride:
    def test_override_absolute_path_accepted(self, tmp_path: Path, monkeypatch):
        fake = _make_fake_claude(tmp_path)
        monkeypatch.setenv("PIPELINE_YOUTUBE_CLAUDE_BIN", str(fake))
        resolved = _resolve_claude_binary()
        assert resolved == str(fake.resolve())

    def test_override_relative_path_rejected(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_fake_claude(tmp_path)
        monkeypatch.setenv("PIPELINE_YOUTUBE_CLAUDE_BIN", "./claude")
        with pytest.raises(ClaudeBinaryError, match="absolute"):
            _resolve_claude_binary()

    def test_override_nonexistent_rejected(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("PIPELINE_YOUTUBE_CLAUDE_BIN", str(tmp_path / "nope"))
        with pytest.raises(ClaudeBinaryError, match="executable"):
            _resolve_claude_binary()

    def test_override_non_executable_rejected(self, tmp_path: Path, monkeypatch):
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\n", encoding="utf-8")
        # NOT chmod +x
        monkeypatch.setenv("PIPELINE_YOUTUBE_CLAUDE_BIN", str(fake))
        with pytest.raises(ClaudeBinaryError, match="executable"):
            _resolve_claude_binary()

    def test_impostor_version_output_rejected(self, tmp_path: Path, monkeypatch):
        """A binary whose --version doesn't mention 'claude' is flagged."""
        fake = _make_fake_claude(tmp_path, stdout="totally-not-the-right-tool v1.0")
        monkeypatch.setenv("PIPELINE_YOUTUBE_CLAUDE_BIN", str(fake))
        with pytest.raises(ClaudeBinaryError, match="does not mention"):
            _resolve_claude_binary()

    def test_cache_returns_same_path(self, tmp_path: Path, monkeypatch):
        fake = _make_fake_claude(tmp_path)
        monkeypatch.setenv("PIPELINE_YOUTUBE_CLAUDE_BIN", str(fake))
        first = _resolve_claude_binary()
        # Delete stub — second call should still succeed via cache
        fake.unlink()
        second = _resolve_claude_binary()
        assert first == second


class TestPathLookup:
    def test_missing_from_path(self, monkeypatch):
        monkeypatch.delenv("PIPELINE_YOUTUBE_CLAUDE_BIN", raising=False)
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(ClaudeBinaryError, match="not found in PATH"),
        ):
            _resolve_claude_binary()
