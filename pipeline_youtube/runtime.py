"""実行時依存の組み立て (composition of runtime dependencies)。

config.json を読み、vault / whisper / alert sink / claude バイナリ /
capture backend を初期化し、その結果を不変の ``Runtime`` にまとめて返す。
``main`` 起動時の「道具を揃える係」。各 configure_* の HOW は専用モジュールが
持ち、ここは配線のみ。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .cli_config import DEFAULT_CONFIG_PATH, CliConfig, _load_config
from .config import VaultRootError, set_dry_run, set_vault_root
from .providers.claude_cli import ClaudeBinaryError, get_resolved_claude_binary
from .sanitize import configure_alert_sink
from .stages.capture import ASSETS_REL_PATH, sweep_stale_tmp
from .stages.capture_backend import DockerBackendNotReady, DockerCaptureBackend
from .transcript.whisper_fallback import configure_whisper, describe_whisper

if TYPE_CHECKING:
    from .command import CliRequest


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


def build_runtime(request: CliRequest) -> Runtime:
    """Load config and initialize vault / whisper / claude / capture / logger."""
    cfg_path = request.config_path or DEFAULT_CONFIG_PATH
    cfg = _load_config(cfg_path, fallback_model=request.model)
    try:
        set_vault_root(cfg.vault_root, strict=True)
    except VaultRootError as exc:
        raise click.UsageError(str(exc)) from exc
    set_dry_run(request.dry_run)
    configure_whisper(backend=cfg.whisper_backend, model=cfg.whisper_model)
    vault_root = cfg.vault_root
    models = cfg.models
    filler_words = cfg.filler_words

    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_alert_sink(logs_dir / "sanitize_alerts.jsonl")

    swept = sweep_stale_tmp(project_root / "tmp")
    if swept:
        click.echo(f"swept {swept} stale tmp video file(s)")

    try:
        claude_bin, claude_ver = get_resolved_claude_binary()
        click.echo(f"claude: {claude_bin} ({claude_ver})")
    except ClaudeBinaryError as exc:
        raise click.UsageError(str(exc)) from exc

    # Resolve the Stage 03 capture backend. CLI flag beats config.json; both
    # default to "host". The preflight for Docker mode is deferred until we
    # know Stage 03 will actually run — workflows that skip capture
    # (`--synthesis-only`, `--resume-reviewed`) must not fail just because
    # the docker daemon happens to be unavailable at that moment.
    active_capture_backend: Any = None
    backend_choice = request.capture_backend or cfg.capture_backend
    # Capture runs in every mode except --synthesis-only (which only re-runs
    # Stage 05 over existing 04 md). In particular --resume-reviewed still calls
    # _process_video()/Stage 03, so it must run the docker preflight and be
    # subject to the local-media guard below.
    will_run_capture = not request.synthesis_only
    # --local-media files live outside the container's bind mounts (tmp/ + the
    # Vault assets folder), so the docker backend's ffmpeg can't read them.
    # Reject the combination up front instead of failing per-video deep inside
    # Stage 03.
    if request.local_media and backend_choice == "docker" and will_run_capture:
        raise click.UsageError(
            "--local-media is incompatible with the docker capture backend: the "
            "hardened container only mounts tmp/ and the Vault assets folder, so "
            "your media directory is not visible to ffmpeg. Re-run with the host "
            "backend (--capture-backend host)."
        )
    if backend_choice == "docker":
        assets_dir = vault_root / ASSETS_REL_PATH
        assets_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        active_capture_backend = DockerCaptureBackend(
            tmp_dir=tmp_dir,
            assets_dir=assets_dir,
            image=cfg.capture_docker_image,
        )
        if will_run_capture:
            try:
                active_capture_backend.preflight()
            except DockerBackendNotReady as exc:
                raise click.UsageError(str(exc)) from exc
            click.echo(f"capture_backend: docker ({cfg.capture_docker_image})")
        else:
            click.echo(
                f"capture_backend: docker ({cfg.capture_docker_image}) "
                "[preflight deferred: capture not needed this run]"
            )
    else:
        click.echo("capture_backend: host")

    effective_synthesis_timeout = request.synthesis_timeout or cfg.synthesis_timeout
    effective_synthesis_profile = request.synthesis_profile or cfg.synthesis_profile or "auto"

    click.echo(f"vault_root: {vault_root}")
    click.echo(f"dry_run: {request.dry_run}")
    click.echo(f"model: {request.model}")
    click.echo(f"whisper: {describe_whisper()}")
    click.echo(f"capture_format: {request.capture_format}")
    click.echo(f"concurrency: {request.concurrency}")
    click.echo(f"min_playlist_size: {request.min_playlist_size}")
    click.echo(
        f"max_chapters: {request.max_chapters if request.max_chapters is not None else 'auto'}"
    )
    click.echo(
        f"synthesis_timeout: {effective_synthesis_timeout}s"
        if effective_synthesis_timeout
        else "synthesis_timeout: auto"
    )
    click.echo(f"synthesis_profile: {effective_synthesis_profile}")

    return Runtime(
        cfg=cfg,
        vault_root=vault_root,
        models=models,
        filler_words=filler_words,
        project_root=project_root,
        logs_dir=logs_dir,
        capture_backend=active_capture_backend,
        synthesis_timeout=effective_synthesis_timeout,
        synthesis_profile=effective_synthesis_profile,
    )
