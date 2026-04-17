"""Module-level configuration state.

Mirrors the `pipeline/config.ts` pattern: simple setters/getters so
tests can swap vault_root without a full config file. JSON config
loading is layered on top in later steps.
"""

from __future__ import annotations

from pathlib import Path

_vault_root: Path | None = None
_dry_run: bool = False


def set_vault_root(path: str | Path) -> None:
    global _vault_root
    _vault_root = Path(path).expanduser().resolve()


def get_vault_root() -> Path:
    if _vault_root is None:
        raise RuntimeError("vault_root is not set. Call set_vault_root() before using path_safety.")
    return _vault_root


def reset_vault_root() -> None:
    global _vault_root
    _vault_root = None


def set_dry_run(flag: bool) -> None:
    global _dry_run
    _dry_run = flag


def is_dry_run() -> bool:
    return _dry_run
