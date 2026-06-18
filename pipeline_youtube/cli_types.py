"""CLI 層で受け渡す不変データ型 (DTO) の集約。

``CliRequest`` / ``Runtime`` / ``ResolvedInput`` / ``ExecutionPlan`` /
``RunMode`` をここ 1 か所に置く。これらは葉 (leaf) の型定義であり、
``cli_config`` / ``playlist`` などのドメイン型だけに依存し、``command`` や各段
モジュール (runtime/input_resolver/…) には依存しない。

各段モジュールはこのモジュールからのみ型を取り込むことで、``command`` への
逆 import (module-level cyclic import) を発生させない。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .cli_config import CliConfig
from .playlist import VideoMeta


@dataclass(frozen=True, slots=True)
class CliRequest:
    """Parsed CLI options (the "受付票"). Immutable snapshot of one invocation."""

    url: str | None
    dry_run: bool
    concurrency: int
    sub_agents: int
    video_range: str | None
    run_timestamp: str | None
    code_bearing_override: bool | None
    skip_synthesis: bool
    synthesis_only: bool
    force_video: tuple[str, ...]
    capture_format: str
    model: str
    min_playlist_size: int
    max_chapters: int | None
    config_path: Path | None
    stop_after_capture: bool
    resume_reviewed: bool
    capture_backend: str | None
    synthesis_timeout: int | None
    synthesis_profile: str | None
    local_media: Path | None


@dataclass(frozen=True, slots=True)
class Runtime:
    """Assembled runtime dependencies for one invocation (the "道具一式")."""

    cfg: CliConfig
    vault_root: Path
    models: dict[str, str]
    filler_words: tuple[str, ...]
    project_root: Path
    logs_dir: Path
    capture_backend: Any
    synthesis_timeout: int | None
    synthesis_profile: str


@dataclass(frozen=True, slots=True)
class ResolvedInput:
    """The video list (+ local-media map) and its genre classification."""

    videos: list[VideoMeta]
    media_map: dict[str, Path]
    playlist_title: str
    code_bearing: bool


class RunMode(StrEnum):
    """How this invocation drives stages 01-04 → 05."""

    NORMAL = "normal"
    LOCAL_MEDIA = "local-media"
    SYNTHESIS_ONLY = "synthesis-only"
    RESUME_REVIEWED = "resume-reviewed"
    SUB_AGENT_PARENT = "sub-agent-parent"
    SUB_AGENT_WORKER = "sub-agent-worker"


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """The decided run mode, shared run timestamp, and worker shard slice."""

    mode: RunMode
    run_time: datetime
    video_range: tuple[int, int] | None
