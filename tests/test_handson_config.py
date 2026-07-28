"""Verifies the hands-on config surface: resolve_config_path precedence and
the handson model keys in _load_config.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from pipeline_youtube.cli_config import (
    _MODEL_KEYS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HANDSON_CONFIG_PATH,
    _load_config,
    resolve_config_path,
)


class TestResolveConfigPath:
    def test_explicit_config_wins_in_both_modes(self, tmp_path: Path):
        explicit = tmp_path / "my.json"
        assert resolve_config_path(explicit, handson=False) == explicit
        assert resolve_config_path(explicit, handson=True) == explicit

    def test_handson_defaults_to_handson_config(self):
        assert resolve_config_path(None, handson=True) == DEFAULT_HANDSON_CONFIG_PATH
        assert DEFAULT_HANDSON_CONFIG_PATH.name == "config.handson.json"

    def test_normal_mode_defaults_to_config_json(self):
        assert resolve_config_path(None, handson=False) == DEFAULT_CONFIG_PATH


class TestHandsonModelKeys:
    def test_handson_keys_are_registered(self):
        assert {"handson_segment", "handson_plan", "handson_step", "handson_moc"} <= _MODEL_KEYS

    def test_config_with_handson_keys_loads(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = tmp_path / "config.handson.json"
        cfg.write_text(
            json.dumps(
                {
                    "vault_root": str(vault),
                    "models": {"handson_step": "opus", "handson_moc": "opus"},
                }
            ),
            encoding="utf-8",
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.models["handson_step"] == "opus"
        assert result.models["handson_segment"] == "sonnet"  # fallback

    def test_unknown_key_still_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"vault_root": str(vault), "models": {"handson_typo": "opus"}}),
            encoding="utf-8",
        )
        with pytest.raises(click.UsageError, match="unknown model keys"):
            _load_config(cfg, fallback_model="sonnet")

    def test_missing_handson_config_hint_names_matching_example(self, tmp_path: Path):
        with pytest.raises(
            click.UsageError, match=r"Copy config\.handson\.example\.json to config\.handson\.json"
        ):
            _load_config(tmp_path / "config.handson.json", fallback_model="sonnet")
