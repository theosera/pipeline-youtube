"""Tests for WS2: per-stage / per-agent model cascade loaded from config.json."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from pipeline_youtube.main import _load_config


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadConfig:
    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(click.UsageError, match="config.json not found"):
            _load_config(tmp_path / "does-not-exist.json", fallback_model="sonnet")

    def test_placeholder_vault_rejected(self, tmp_path: Path):
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": "/path/to/your/Obsidian Vault"},
        )
        with pytest.raises(click.UsageError, match="vault_root"):
            _load_config(cfg, fallback_model="sonnet")

    def test_models_omitted_uses_fallback(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(tmp_path / "config.json", {"vault_root": str(vault)})
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.vault_root == vault
        assert result.models == {
            "stage_02": "sonnet",
            "stage_04": "sonnet",
            "alpha": "sonnet",
            "beta": "sonnet",
            "gamma": "sonnet",
            "leader": "sonnet",
        }

    def test_partial_models_filled_with_fallback(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {"alpha": "haiku", "leader": "opus"},
            },
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.models["alpha"] == "haiku"
        assert result.models["leader"] == "opus"
        assert result.models["beta"] == "sonnet"
        assert result.models["gamma"] == "sonnet"
        assert result.models["stage_02"] == "sonnet"
        assert result.models["stage_04"] == "sonnet"

    def test_all_models_explicit(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {
                    "stage_02": "haiku",
                    "stage_04": "sonnet",
                    "alpha": "haiku",
                    "beta": "sonnet",
                    "gamma": "haiku",
                    "leader": "opus",
                },
            },
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.models["stage_02"] == "haiku"
        assert result.models["leader"] == "opus"
        assert result.models["gamma"] == "haiku"

    def test_unknown_model_key_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {"delta": "haiku"},
            },
        )
        with pytest.raises(click.UsageError, match="unknown model keys"):
            _load_config(cfg, fallback_model="sonnet")

    def test_models_must_be_object(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "models": "sonnet"},
        )
        with pytest.raises(click.UsageError, match="'models' must be an object"):
            _load_config(cfg, fallback_model="sonnet")
