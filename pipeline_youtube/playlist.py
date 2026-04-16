"""Playlist and video metadata fetching via yt-dlp's Python API.

Uses `extract_flat='in_playlist'` so only metadata is fetched (no
actual video downloads). No YouTube Data API key is required —
yt-dlp scrapes the public playlist page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yt_dlp  # type: ignore[import-untyped]


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    url: str
    duration: int | None  # seconds, may be None on flat-playlist mode
    channel: str | None
    upload_date: str | None  # YYYYMMDD format from yt-dlp
    playlist_title: str | None

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def timestamp_url(self) -> str:
        """Base URL suitable for appending &t=<seconds>."""
        return self.watch_url


_BASE_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
}


def fetch_metadata(url: str) -> list[VideoMeta]:
    """Fetch metadata for a playlist URL or a single-video URL.

    Returns a list of VideoMeta (single entry for a video URL, multiple
    for a playlist). On a playlist URL, the playlist title is propagated
    to every VideoMeta.playlist_title for downstream folder naming.
    """
    opts = {**_BASE_OPTS, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info: dict[str, Any] = ydl.extract_info(url, download=False)  # type: ignore[assignment]

    if info is None:
        return []

    playlist_title: str | None = None
    if info.get("_type") == "playlist":
        playlist_title = info.get("title")

    entries = info.get("entries")
    if entries is None:
        entries = [info]

    videos: list[VideoMeta] = []
    for entry in entries:
        if entry is None:
            continue
        video_id = entry.get("id") or ""
        if not video_id:
            continue
        videos.append(
            VideoMeta(
                video_id=video_id,
                title=entry.get("title") or "",
                url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                duration=entry.get("duration"),
                channel=entry.get("channel") or entry.get("uploader"),
                upload_date=entry.get("upload_date"),
                playlist_title=playlist_title,
            )
        )
    return videos
