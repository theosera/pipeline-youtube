"""Tests for YouTube URL whitelist validation (H1)."""

from __future__ import annotations

import pytest

from pipeline_youtube.playlist import validate_youtube_url


class TestValidateYouTubeUrlAccepted:
    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_short_youtu_be(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_mobile(self):
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_playlist(self):
        url = "https://www.youtube.com/playlist?list=PLabc"
        assert validate_youtube_url(url) == url

    def test_http_scheme_allowed(self):
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url


class TestValidateYouTubeUrlRejected:
    def test_file_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_youtube_url("file:///etc/passwd")

    def test_internal_host(self):
        with pytest.raises(ValueError, match="host"):
            validate_youtube_url("http://localhost/watch?v=abc")

    def test_third_party_host(self):
        with pytest.raises(ValueError, match="host"):
            validate_youtube_url("https://evil.example.com/watch?v=abc")

    def test_ftp_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_youtube_url("ftp://www.youtube.com/watch?v=abc")

    def test_empty(self):
        with pytest.raises(ValueError):
            validate_youtube_url("")

    def test_spoofed_subdomain(self):
        with pytest.raises(ValueError, match="host"):
            validate_youtube_url("https://www.youtube.com.evil.com/watch?v=abc")
