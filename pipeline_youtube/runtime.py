"""実行時依存の組み立て (composition of runtime dependencies)。

config.json を読み、vault / whisper / alert sink / claude バイナリ /
capture backend を初期化し、その結果を不変の ``Runtime`` にまとめて返す。
``main`` 起動時の「道具を揃える係」。各 configure_* の HOW は専用モジュールが
持ち、ここは配線のみ。
"""

from __future__ import annotations

from pathlib import Path

import click

from .capture_runtime import resolve_capture_backend
from .cli_config import DEFAULT_CONFIG_PATH, _load_config
from .cli_types import CliRequest, Runtime
from .config import VaultRootError, set_dry_run, set_vault_root
from .providers.claude_cli import ClaudeBinaryError, get_resolved_claude_binary
from .sanitize import configure_alert_sink
from .stages.capture import sweep_stale_tmp
from .transcript.whisper_fallback import configure_whisper, describe_whisper


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

    # Resolve the Stage 03 capture backend (host / docker preflight / local-media
    # guard). HOW lives in capture_runtime; here we just wire it.
    active_capture_backend = resolve_capture_backend(request, cfg, vault_root, project_root)

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
