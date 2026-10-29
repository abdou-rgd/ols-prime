"""The vault manifest: every stored table, its checksum and what the seal lets through.

For each Parquet file under ``<data>/parquet/`` (the klines and the universe file) the
manifest records its path, size and SHA-256, its stored row count (from the Parquet
metadata), its sealed row count and the first and last sealed time. A report that
cites ``digest`` cites all of it: change one byte of one table, or the vault cut-off, and
the digest changes.

The digest is the SHA-256 of the canonical body: the manifest without ``digest`` and
``generated_at``, as JSON with sorted keys, no spaces and ASCII only. ``generated_at`` is
left out so that the same data gives the same digest tomorrow. :func:`verify_manifest`
recomputes it. The vault state (``vault_open``) is part of the body, because it changes the
sealed counts.

The manifest also reports the vault tree (``olsprime.vault``): its stage (0 in ``data/``, 1 moved and
plaintext, 2 encrypted), file count, bytes, and the digest of the tree's own ``MANIFEST.json``. That
summary reads names, footers and sizes, never a vault row. It is part of the body, so moving or
encrypting the vault changes this manifest's digest.

This module opens no Parquet file: the row counts and the spans come from helpers of
:mod:`olsprime.protocol`, the one module that reads stored tables. A manifest built with
the vault open logs each read to ``ACCESS.log``, like any read of the vault.

Usage::

    uv run python -m olsprime.manifest [--data-dir DIR] [--out PATH]

The default output is ``<runs>/manifest/<UTC timestamp>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olsprime.paths import data_dir as default_data_dir
from olsprime.paths import runs_dir
from olsprime.protocol import (
    VAULT,
    sealed_klines_span,
    sealed_universe_span,
    stored_row_count,
    vault_open,
)
from olsprime.vault import tree_summary

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
NOT_IN_BODY = ("digest", "generated_at")


def _iso(when: datetime | None) -> str | None:
    return None if when is None else when.astimezone(UTC).isoformat(timespec="milliseconds")


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_json(body: dict[str, Any]) -> str:
    """Sorted keys, no spaces, ASCII: one text per content."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def body_digest(manifest: dict[str, Any]) -> str:
    """SHA-256 of the canonical body of ``manifest`` (without ``digest`` and ``generated_at``)."""
    body = {key: value for key, value in manifest.items() if key not in NOT_IN_BODY}
    return hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()


def verify_manifest(manifest: dict[str, Any]) -> bool:
    """Whether the manifest's ``digest`` matches its body."""
    return manifest.get("digest") == body_digest(manifest)


def _describe(data_dir: Path, path: Path, *, caller: str) -> dict[str, Any]:
    """One manifest entry; the kind (klines or universe) comes from the path below ``parquet/``."""
    relative = path.relative_to(data_dir).as_posix()
    parts = path.relative_to(data_dir / "parquet").parts
    entry: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "stored_rows": stored_row_count(path),
    }
    if len(parts) == 4 and parts[:2] == ("spot", "klines"):
        interval = parts[2]
        rows, first, last = sealed_klines_span(path, interval, caller=caller)
        entry |= {"kind": "klines", "symbol": path.stem, "interval": interval, "time_column": "open_time"}
    elif parts[:1] == ("universe",):
        rows, first, last = sealed_universe_span(path, caller=caller)
        entry |= {"kind": "universe", "time_column": "rebalance_time"}
    else:
        raise ValueError(f"{relative}: neither a klines file nor a universe file; the manifest does not know its seal")
    entry |= {"sealed_rows": rows, "first_sealed": _iso(first), "last_sealed": _iso(last)}
    return entry


def build_manifest(data_dir: Path, *, generated_at: datetime | None = None) -> dict[str, Any]:
    """The manifest of every Parquet file under ``data_dir/parquet``, sorted by path."""
    caller = "manifest.build_manifest"
    files = sorted((data_dir / "parquet").rglob("*.parquet"), key=lambda p: p.relative_to(data_dir).as_posix())
    entries = [_describe(data_dir, path, caller=caller) for path in files]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "vault_start": _iso(VAULT.start),
        "vault_open": vault_open(),
        "vault_tree": tree_summary(data_dir),
        "files": entries,
        "totals": {
            "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "stored_rows": sum(e["stored_rows"] for e in entries),
            "sealed_rows": sum(e["sealed_rows"] for e in entries),
        },
    }
    manifest["generated_at"] = _iso(generated_at or datetime.now(UTC))
    manifest["digest"] = body_digest(manifest)
    return manifest


def default_output(when: datetime | None = None) -> Path:
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return runs_dir() / "manifest" / f"{stamp}.json"


def write_manifest(manifest: dict[str, Any], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(out.name + ".part")
    staged.write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    staged.replace(out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olsprime.manifest",
        description="Write the manifest of the stored Parquet tables: checksums, stored and sealed rows, one digest.",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(), help="data root (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=None, help="output file (default: <runs>/manifest/<UTC timestamp>.json)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = build_manifest(args.data_dir)
    out = write_manifest(manifest, args.out or default_output())
    totals = manifest["totals"]
    print(
        f"{totals['files']} files, {totals['bytes']:,} bytes, {totals['stored_rows']:,} stored rows, "
        f"{totals['sealed_rows']:,} sealed rows; vault {'OPEN' if manifest['vault_open'] else 'sealed'}"
    )
    tree = manifest["vault_tree"]
    print(f"vault tree: stage {tree['stage']}, {tree['files']} files, {tree['bytes']:,} bytes, manifest {tree['manifest_digest']}")
    print(f"digest {manifest['digest']}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
