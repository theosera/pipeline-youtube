"""Tests for #7: tmp directory + downloaded file have owner-only permissions."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from pipeline_youtube.stages.capture import _restrict_tmp_video, _tmp_video_path


def _video(video_id: str = "abc1234567"):
    from pipeline_youtube.playlist import VideoMeta

    return VideoMeta(
        video_id=video_id,
        title="t",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="c",
        upload_date=None,
        playlist_title=None,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX perms")
class TestTmpDirPermissions:
    def test_tmp_dir_is_700(self):
        path = _tmp_video_path(_video())
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"

    def test_restrict_tmp_video_sets_600(self, tmp_path: Path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        os.chmod(f, 0o644)  # start wide-open
        _restrict_tmp_video(f)
        mode = stat.S_IMODE(f.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_restrict_missing_file_is_noop(self, tmp_path: Path):
        _restrict_tmp_video(tmp_path / "missing.mp4")  # must not raise
