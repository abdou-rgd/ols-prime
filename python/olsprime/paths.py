"""Location of the local data store."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

VAULT_DIR_ENV = "OLSPRIME_VAULT_DIR"


def data_dir() -> Path:
    """Data root: ``$OLSPRIME_DATA_DIR`` when set, else ``<repo>/data``."""
    override = os.environ.get("OLSPRIME_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data"


def runs_dir() -> Path:
    """Run outputs and the vault key: ``$OLSPRIME_RUNS_DIR`` when set, else ``<repo>/runs``."""
    override = os.environ.get("OLSPRIME_RUNS_DIR")
    return Path(override) if override else REPO_ROOT / "runs"


def default_vault_dir() -> Path:
    """The vault tree's home: ``<parent of the repository>/ols-prime-vault``, outside the repository."""
    return REPO_ROOT.parent / "ols-prime-vault"


def vault_dir() -> Path:
    """Vault tree root: ``$OLSPRIME_VAULT_DIR`` when set, else :func:`default_vault_dir`.

    The vault tree holds the archives and Parquet files of the vault months once
    ``python -m olsprime.vault move`` has run (``docs/DATA.md``, "The physical seal"). Only
    ``olsprime/vault.py`` and ``olsprime/protocol.py`` name it.
    """
    override = os.environ.get(VAULT_DIR_ENV)
    return Path(override) if override else default_vault_dir()
