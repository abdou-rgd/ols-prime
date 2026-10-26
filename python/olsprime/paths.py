"""Location of the local data store."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Data root: ``$OLSPRIME_DATA_DIR`` when set, else ``<repo>/data``."""
    override = os.environ.get("OLSPRIME_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data"


def runs_dir() -> Path:
    """Run outputs and the vault key: ``$OLSPRIME_RUNS_DIR`` when set, else ``<repo>/runs``."""
    override = os.environ.get("OLSPRIME_RUNS_DIR")
    return Path(override) if override else REPO_ROOT / "runs"
