"""The physical seal: the vault months move out of ``data/`` into a vault tree, then are encrypted.

The guards of ``olsprime.protocol`` and of the tests live inside the process, and four rounds of audit
showed that such guards can be taken down from inside it. This module makes the vault unreadable
instead of watched, in two stages::

    uv run python -m olsprime.vault {plan,move,encrypt,keygen,verify-decrypt,status}

**Stage 0, in ``data/``.** The monthly archives of the vault months (read from the file names: month at or
after ``VAULT.start``) sit in ``data/raw/``, and the Parquet files of their pairs hold the vault rows.

**Stage 1, out of ``data/`` (``move``).** For each affected (symbol, interval):

1. the Parquet file is rebuilt from the non-vault archives into a staging folder beside ``parquet/``;
2. the sealed read of the old file is compared with the new file's, frame for frame, schema included, and
   the new file's stored row count must equal the old file's sealed row count (no vault row left);
3. the old file and its quality report join the vault tree, mirroring their paths, and the new files take
   their place;
4. every vault archive and its ``.CHECKSUM`` file, once ``is_verified`` passes, is renamed into the vault tree.

Nothing is deleted. A rename keeps the bytes; where a file must be replaced by another of the same name (step 3)
the old one is first hard-linked into the vault tree, then the new one is renamed over it, so the path is never
empty. A journal (``JOURNAL.json`` in the vault tree) records each step, and a second run finishes the job or does
nothing. ``MANIFEST.json`` lists every file of the tree with its size, SHA-256 and original path.

**Stage 2, encrypted (``encrypt``).** Each file of the tree is encrypted to ``<name>.age`` for an age X25519
recipient whose private key never touches this machine (``keygen`` makes one, for Abdou to run with the output on
a medium taken away). The ciphertext SHA-256 goes into the manifest. A plaintext file is removed **only** with
``--remove-plaintext``, after its ciphertext is written, flushed and re-hashed from disk, and only once
``verify-decrypt`` has stamped the manifest: it decrypts every ``.age`` file in memory with the identity named by
``OLSPRIME_VAULT_IDENTITY`` and compares the plaintext's SHA-256 with the manifest's.

The open path (final evaluation only) is in :mod:`olsprime.protocol`: with both keys in place ``load_klines``
adds the vault tree's rows, decrypting in memory at stage 2 with the identity named by
``OLSPRIME_VAULT_IDENTITY``.

What this does not do: the vault months are public on data.binance.vision. The seal stops our own pipeline and
our agents, not a person who downloads them. Until stage 2 the tree is plaintext, outside ``data/`` and guarded
in the process only. Until the identity leaves this machine, stage 2 is only as safe as that file.

The tool never sets the open-vault variable, never writes the key file and never parses a vault archive or row:
it hashes bytes and renames files. Its row-level checks read the *sealed* view of a table through
``olsprime.protocol`` (and refuse to run with the vault open, where that view would show the vault).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olsprime import paths, protocol
from olsprime.data.binance_vision import (
    KLINES_PREFIX,
    Month,
    archive_name_parts,
    build_parquet,
    cached_archives,
    is_verified,
    month_spans,
    parquet_path,
)

log = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.json"
JOURNAL_NAME = "JOURNAL.json"
STAGING_DIR = "parquet.vault-stage"  # beside data/parquet, same volume; emptied by the move
SCHEMA_VERSION = 1
BOOKKEEPING = frozenset({MANIFEST_NAME, JOURNAL_NAME})
CHECKSUM_SUFFIX = ".CHECKSUM"
STAMP_KEY = "decrypt_check"  # in the manifest: written by verify-decrypt, required by encrypt --remove-plaintext

Checkpoint = Callable[[str], None]


class VaultError(RuntimeError):
    """The tool refuses: a root in the wrong place, an anomaly in the store, a file that is not what the plan said."""


# --- Hashing and small helpers --------------------------------------------------------


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha_or_none(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def _iso(when: datetime | None = None) -> str:
    return (when or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".part")
    with staged.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(body, indent=1, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)


def _inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def check_roots(data_dir: Path, vault_root: Path) -> None:
    """Refuse a vault root inside the repository or the data root (or holding the data root), or on another volume."""
    if _inside(vault_root, paths.REPO_ROOT):
        raise VaultError(f"the vault root {vault_root} is inside the repository: a git add or a glob of it would reach the vault")
    if _inside(vault_root, data_dir) or _inside(data_dir, vault_root):
        raise VaultError(f"the vault root {vault_root} and the data root {data_dir} are nested: they must be apart")
    anchor = vault_root
    while not anchor.exists():
        anchor = anchor.parent
    if data_dir.exists() and os.stat(data_dir).st_dev != os.stat(anchor).st_dev:
        raise VaultError(f"the vault root {vault_root} is on another volume than {data_dir}: the move is a rename, not a copy")


def _refuse_if_open() -> None:
    if protocol.vault_open():
        raise VaultError("the vault is open (both keys in place): this tool works on a sealed vault, close it first")


# --- Finding the vault months, by name ------------------------------------------------


@dataclass(frozen=True)
class VaultArchive:
    """One archive of a vault month under ``raw/``, with its CHECKSUM file (``None`` if missing)."""

    symbol: str
    interval: str
    month: Month
    zip_rel: str  # below raw/
    zip_bytes: int
    checksum_bytes: int | None
    verified: bool
    sha256: str  # of the archive, hashed the way is_verified does
    checksum_sha256: str | None = None  # of the CHECKSUM file; the journal pins it, as it pins the archive's

    @property
    def checksum_rel(self) -> str:
        return self.zip_rel + CHECKSUM_SUFFIX


def scan_vault_archives(raw_dir: Path) -> tuple[list[VaultArchive], list[str]]:
    """Every archive of a vault month under ``raw_dir``, found from the file names alone, and the anomalies.

    Anomalies: an archive without its CHECKSUM file, a CHECKSUM file without its archive, an archive that fails
    ``is_verified``, a name that is not an archive's in an archive folder, a leftover ``.part`` of a vault month.
    The files are hashed, never parsed.
    """
    base = raw_dir / KLINES_PREFIX
    found: list[VaultArchive] = []
    anomalies: list[str] = []
    if not base.is_dir():
        return found, anomalies
    first_vault = protocol.VAULT.start
    for symbol_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for interval_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
            zips: dict[str, Path] = {}
            checksums: set[str] = set()
            for entry in sorted(interval_dir.iterdir()):
                name = entry.name
                core = name.removesuffix(".part").removesuffix(CHECKSUM_SUFFIX).removesuffix(".zip")
                is_zip_name = name.endswith((".zip", ".zip" + CHECKSUM_SUFFIX, ".zip.part", ".zip" + CHECKSUM_SUFFIX + ".part"))
                if not is_zip_name:
                    continue
                parts = archive_name_parts(core + ".zip")
                if parts is None or parts[:2] != (symbol_dir.name, interval_dir.name):
                    anomalies.append(f"{entry.relative_to(raw_dir).as_posix()}: not an archive name of {symbol_dir.name} {interval_dir.name}")
                    continue
                if parts[2].start() < first_vault:
                    continue
                relative = entry.relative_to(raw_dir).as_posix()
                if name.endswith(".part"):
                    anomalies.append(f"{relative}: leftover partial file of a vault month")
                elif name.endswith(CHECKSUM_SUFFIX):
                    checksums.add(core + ".zip")
                else:
                    zips[name] = entry
            for zip_name, zip_path in zips.items():
                symbol, interval, month = archive_name_parts(zip_name)  # type: ignore[misc]
                relative = zip_path.relative_to(raw_dir).as_posix()
                has_checksum = zip_name in checksums
                verified = has_checksum and is_verified(zip_path)
                if not has_checksum:
                    anomalies.append(f"{relative}: no CHECKSUM file beside it")
                elif not verified:
                    anomalies.append(f"{relative}: does not match its CHECKSUM file")
                checksum_path = zip_path.with_name(zip_name + CHECKSUM_SUFFIX)
                found.append(
                    VaultArchive(
                        symbol,
                        interval,
                        month,
                        relative,
                        zip_path.stat().st_size,
                        checksum_path.stat().st_size if has_checksum else None,
                        verified,
                        file_sha256(zip_path),
                        _sha_or_none(checksum_path),
                    )
                )
            for orphan in sorted(checksums - set(zips)):
                anomalies.append(f"{(interval_dir / (orphan + CHECKSUM_SUFFIX)).relative_to(raw_dir).as_posix()}: CHECKSUM file without its archive")
    return sorted(found, key=lambda a: (a.symbol, a.interval, a.month)), anomalies


def _klines_parquet_files(parquet_dir: Path) -> list[Path]:
    base = parquet_dir / "spot" / "klines"
    return sorted(base.glob("*/*.parquet")) if base.is_dir() else []


def vault_bars_by_statistics(parquet_dir: Path) -> tuple[list[tuple[str, str]], int]:
    """``(pairs whose Parquet footer shows an open_time at or after VAULT.start, files without statistics)``.

    A footer read: no row is touched.
    """
    holding: list[tuple[str, str]] = []
    unknown = 0
    for path in _klines_parquet_files(parquet_dir):
        bounds = protocol.stored_open_time_bounds(path)
        if bounds is None:
            unknown += 1
        elif bounds[1] >= protocol.VAULT.start:
            holding.append((path.stem, path.parent.name))
    return holding, unknown


# --- The plan -------------------------------------------------------------------------


def _parquet_rel(symbol: str, interval: str) -> str:
    return parquet_path(Path("parquet"), symbol, interval).as_posix()


def _quality_rel(symbol: str, interval: str) -> str:
    return parquet_path(Path("parquet"), symbol, interval).with_name(f"{symbol}.quality.json").as_posix()


@dataclass(frozen=True)
class ParquetItem:
    """A Parquet file that holds vault rows, and the rebuild it would get."""

    symbol: str
    interval: str
    rel: str  # below data/
    quality_rel: str | None  # None: no quality report beside it
    bytes: int
    stored_rows: int
    sealed_rows: int
    rebuild_archives: int  # non-vault archives the new file is built from
    first_month: Month | None
    last_month: Month | None

    @property
    def vault_rows(self) -> int:
        return self.stored_rows - self.sealed_rows


@dataclass
class Plan:
    data_dir: Path
    vault_root: Path
    archives: list[VaultArchive] = field(default_factory=list)
    parquet: list[ParquetItem] = field(default_factory=list)
    left_alone: list[str] = field(default_factory=list)  # pairs with vault archives whose file holds no vault row
    anomalies: list[str] = field(default_factory=list)
    footer_files_without_statistics: int = 0

    @property
    def has_work(self) -> bool:
        return bool(self.archives or self.parquet)


def build_plan(data_dir: Path, vault_root: Path) -> Plan:
    """What ``move`` would do, changing nothing. Reads no row of the vault: counts, footers, names, hashes."""
    _refuse_if_open()
    check_roots(data_dir, vault_root)
    raw_dir, parquet_dir = data_dir / "raw", data_dir / "parquet"
    archives, anomalies = scan_vault_archives(raw_dir)
    plan = Plan(data_dir, vault_root, archives, anomalies=anomalies)
    pairs = sorted({(a.symbol, a.interval) for a in archives})
    for symbol, interval in pairs:
        path = parquet_path(parquet_dir, symbol, interval)
        if not path.is_file():
            plan.left_alone.append(f"{symbol} {interval}: vault archives but no Parquet file")
            continue
        stored = protocol.stored_row_count(path)
        sealed = protocol.sealed_klines_span(path, interval, caller="vault.plan")[0]
        rebuild = [
            parts[2]
            for archive in cached_archives(raw_dir, symbol, interval)
            if (parts := archive_name_parts(archive.name)) is not None and parts[2].start() < protocol.VAULT.start
        ]
        if stored == sealed:
            plan.left_alone.append(f"{symbol} {interval}: the Parquet file holds no vault row ({stored:,} rows)")
            continue
        if not rebuild:
            plan.anomalies.append(f"{symbol} {interval}: no archive before the vault to rebuild the Parquet file from")
        quality = path.with_name(f"{symbol}.quality.json")
        plan.parquet.append(
            ParquetItem(
                symbol,
                interval,
                _parquet_rel(symbol, interval),
                _quality_rel(symbol, interval) if quality.is_file() else None,
                path.stat().st_size,
                stored,
                sealed,
                len(rebuild),
                min(rebuild) if rebuild else None,
                max(rebuild) if rebuild else None,
            )
        )
    holding, unknown = vault_bars_by_statistics(parquet_dir)
    plan.footer_files_without_statistics = unknown
    named = {(i.symbol, i.interval) for i in plan.parquet} | set(pairs)
    for symbol, interval in holding:
        if (symbol, interval) not in named:
            plan.anomalies.append(
                f"{symbol} {interval}: the Parquet footer shows bars at or after the vault start, but no vault archive is under raw/"
            )
    return plan


@dataclass(frozen=True)
class RebuildCheck:
    symbol: str
    interval: str
    ok: bool
    new_rows: int
    message: str


def stage_and_verify(data_dir: Path, symbol: str, interval: str, stage_dir: Path) -> tuple[Path, Path | None, int]:
    """Rebuild a pair's Parquet file from its non-vault archives into ``stage_dir`` and check it against the old one.

    Returns ``(new Parquet file, new quality report, rows)``. Raises :class:`VaultError` unless:

    - the sealed read of the old file and the sealed read of the new one are the same frames, schema included;
    - the new file's stored row count equals the old file's sealed row count (no vault row left in it).
    """
    builder = logging.getLogger("olsprime.data.binance_vision")
    level = builder.level
    builder.setLevel(logging.ERROR)  # the build names each vault archive it leaves out: expected here, 18 lines a table
    try:
        report = build_parquet(symbol, interval, raw_dir=data_dir / "raw", parquet_dir=stage_dir)
    finally:
        builder.setLevel(level)
    if report is None:
        raise VaultError(f"{symbol} {interval}: no verified archive before the vault to rebuild from")
    old = parquet_path(data_dir / "parquet", symbol, interval)
    new = parquet_path(stage_dir, symbol, interval)
    before = protocol.load_klines_file(old, interval, caller="vault.verify old")
    after = protocol.load_klines_file(new, interval, caller="vault.verify new")
    if before.schema != after.schema:
        raise VaultError(f"{symbol} {interval}: the rebuilt file has another schema than the old one ({after.schema} vs {before.schema})")
    if not before.equals(after):
        raise VaultError(f"{symbol} {interval}: the sealed read of the rebuilt file differs from the old file's")
    stored = protocol.stored_row_count(new)
    if stored != before.height:
        raise VaultError(f"{symbol} {interval}: the rebuilt file stores {stored:,} rows, the old file's sealed read has {before.height:,}")
    quality = new.with_name(f"{symbol}.quality.json")
    return new, quality if quality.is_file() else None, stored


def verify_rebuilds(plan: Plan) -> list[RebuildCheck]:
    """The first two steps of ``move`` for every file of the plan, in a temporary folder: nothing under data/ changes."""
    checks = []
    with tempfile.TemporaryDirectory(prefix="olsprime-vault-verify-") as scratch:
        for item in plan.parquet:
            try:
                _, _, rows = stage_and_verify(plan.data_dir, item.symbol, item.interval, Path(scratch))
                checks.append(RebuildCheck(item.symbol, item.interval, True, rows, "sealed reads identical, no vault row left"))
            except VaultError as error:
                checks.append(RebuildCheck(item.symbol, item.interval, False, 0, str(error)))
    return checks


def _megabytes(size: int) -> str:
    return f"{size / 1e6:,.1f} MB" if size >= 1_000_000 else f"{size / 1e3:,.1f} kB"


def format_plan(plan: Plan, checks: Sequence[RebuildCheck] | None = None) -> str:
    lines = [
        "Vault plan (dry run: nothing was changed)",
        f"  data root   : {plan.data_dir}",
        f"  vault root  : {plan.vault_root}" + ("" if plan.vault_root.exists() else "  (does not exist yet)"),
        f"  vault start : {protocol.VAULT.start.isoformat()}",
        "",
    ]
    total = sum(a.zip_bytes for a in plan.archives)
    checksums = sum(a.checksum_bytes is not None for a in plan.archives)
    lines.append(f"Archives of vault months under data/raw/: {len(plan.archives)} (+ {checksums} CHECKSUM files), {_megabytes(total)}")
    by_pair: dict[tuple[str, str], list[VaultArchive]] = {}
    for archive in plan.archives:
        by_pair.setdefault((archive.symbol, archive.interval), []).append(archive)
    for (symbol, interval), group in by_pair.items():
        months = month_spans(a.month for a in group)
        ok = sum(a.verified for a in group)
        lines.append(
            f"  {symbol} {interval}: {len(group)} archives {months}, {_megabytes(sum(a.zip_bytes for a in group))}, {ok}/{len(group)} verified"
        )
        lines.extend(f"      {a.zip_rel}  {a.zip_bytes:,} B  sha256 {a.sha256[:16]}..." for a in group)
    lines.append("")
    lines.append(f"Parquet files that hold vault rows: {len(plan.parquet)}")
    verdict = {(c.symbol, c.interval): c for c in checks or ()}
    for item in plan.parquet:
        lines.append(
            f"  {item.rel}  {item.bytes:,} B  stored {item.stored_rows:,} rows, sealed {item.sealed_rows:,}, vault rows {item.vault_rows:,}"
        )
        lines.append(f"      quality report: {item.quality_rel or 'none'}")
        span = f"{item.first_month}..{item.last_month}" if item.first_month else "none"
        lines.append(f"      rebuild: from {item.rebuild_archives} archives {span}, expects {item.sealed_rows:,} rows")
        check = verdict.get((item.symbol, item.interval))
        if check is not None:
            lines.append(f"      verified in a temporary folder: {'OK' if check.ok else 'FAILED'}: {check.message} ({check.new_rows:,} rows)")
    if plan.left_alone:
        lines.append("")
        lines.append("Pairs with vault archives whose Parquet file needs no rebuild:")
        lines.extend(f"  {text}" for text in plan.left_alone)
    lines.append("")
    if plan.footer_files_without_statistics:
        lines.append(f"Parquet files without footer statistics (not scanned for vault bars): {plan.footer_files_without_statistics}")
    lines.append(f"Anomalies: {len(plan.anomalies)}" + (" (move refuses until they are settled)" if plan.anomalies else ""))
    lines.extend(f"  {text}" for text in plan.anomalies)
    return "\n".join(lines)


# --- The vault manifest ---------------------------------------------------------------


def _canonical(body: dict[str, Any]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def manifest_digest(manifest: dict[str, Any]) -> str:
    """SHA-256 of the canonical body (without ``digest`` and ``generated_at``), as :mod:`olsprime.manifest` does."""
    body = {key: value for key, value in manifest.items() if key not in ("digest", "generated_at")}
    return hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()


def _load_manifest(vault_root: Path) -> dict[str, Any]:
    path = vault_root / MANIFEST_NAME
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "vault_start": protocol.VAULT.start.isoformat(), "files": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _store_manifest(vault_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = sorted(manifest["files"], key=lambda e: e["path"])
    stamp = manifest.get(STAMP_KEY)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "vault_start": protocol.VAULT.start.isoformat(),
        "files": files,
        "totals": {
            "files": len(files),
            "bytes": sum(e["bytes"] for e in files),
            "encrypted": sum("ciphertext" in e for e in files),
        },
    }
    if stamp is not None:  # kept across rewrites; a manifest without one has the body it always had
        manifest[STAMP_KEY] = stamp
    manifest["generated_at"] = _iso()
    manifest["digest"] = manifest_digest(manifest)
    _write_json(vault_root / MANIFEST_NAME, manifest)
    return manifest


def _entry(vault_root: Path, rel: str, kind: str) -> dict[str, Any]:
    path = vault_root / rel
    return {"path": rel, "kind": kind, "original": f"data/{rel}", "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _set_entries(vault_root: Path, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    manifest = _load_manifest(vault_root)
    by_path = {e["path"]: e for e in manifest["files"]}
    for entry in entries:
        previous = by_path.get(entry["path"], {})
        by_path[entry["path"]] = {**{k: v for k, v in previous.items() if k in ("ciphertext",)}, **entry}
    manifest["files"] = list(by_path.values())
    return _store_manifest(vault_root, manifest)


def tree_files(vault_root: Path) -> list[str]:
    """Data files of the vault tree (plain and ``.age``), relative, without the manifest, the journal and ``.part`` files."""
    if not vault_root.is_dir():
        return []
    found = []
    for folder, _, names in os.walk(vault_root):
        for name in names:
            if (name in BOOKKEEPING and Path(folder) == vault_root) or name.endswith(".part"):
                continue
            found.append((Path(folder) / name).relative_to(vault_root).as_posix())
    return sorted(found)


# --- The journal and the move ---------------------------------------------------------


def _journal_path(vault_root: Path) -> Path:
    return vault_root / JOURNAL_NAME


def _load_journal(vault_root: Path) -> dict[str, Any] | None:
    path = _journal_path(vault_root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@dataclass
class MoveResult:
    resumed: bool
    archives_moved: int
    parquet_swapped: int
    manifest_digest: str | None
    nothing_to_do: bool = False


def _no_checkpoint(label: str) -> None:
    return None


def _new_journal(plan: Plan) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "created": _iso(),
        "finished": None,
        "data_dir": str(plan.data_dir),
        "vault_dir": str(plan.vault_root),
        "parquet": [
            {
                "id": f"{item.symbol}/{item.interval}",
                "symbol": item.symbol,
                "interval": item.interval,
                "old": item.rel,
                "old_sha256": file_sha256(plan.data_dir / item.rel),
                "old_quality": item.quality_rel,
                "old_quality_sha256": file_sha256(plan.data_dir / item.quality_rel) if item.quality_rel else None,
                "stage": "planned",
                "new_sha256": None,
                "new_quality_sha256": None,
            }
            for item in plan.parquet
        ],
        "archives": [{"rel": a.zip_rel, "sha256": a.sha256, "checksum_sha256": a.checksum_sha256} for a in plan.archives],
    }


def _link_into_vault(source: Path, dest: Path, expected: str) -> None:
    """Hard-link ``source`` to ``dest`` (the old bytes now also live in the vault tree). Idempotent."""
    present = _sha_or_none(dest)
    if present is not None:
        if present != expected:
            raise VaultError(f"{dest} exists in the vault tree with other bytes than the planned file")
        return
    if file_sha256(source) != expected:
        raise VaultError(f"{source} changed since the plan was made; run plan again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError as error:
        raise VaultError(f"cannot hard-link {source} into the vault tree ({error}): the move needs one volume with links") from None


def _swap_in(staged: Path, target: Path, old_sha: str | None, new_sha: str) -> None:
    """Rename ``staged`` over ``target`` unless it is already there; refuse a ``target`` that is neither old nor new."""
    current = _sha_or_none(target)
    if current == new_sha:
        return
    if current != old_sha:
        raise VaultError(f"{target} is neither the planned old file nor the new one")
    if _sha_or_none(staged) != new_sha:
        raise VaultError(f"the staged file {staged} is not the verified one")
    os.replace(staged, target)


def _stage_parquet_job(job: dict[str, Any], data_dir: Path, save: Callable[[], None], checkpoint: Checkpoint) -> None:
    """Phase 1: rebuild the file into the staging folder and verify it. Touches nothing under ``parquet/``."""
    stage_dir = data_dir / STAGING_DIR
    symbol, interval = job["symbol"], job["interval"]
    old = data_dir / job["old"]
    staged = parquet_path(stage_dir, symbol, interval)
    staged_quality = staged.with_name(f"{symbol}.quality.json")
    if job["stage"] in ("staged", "linked") and _sha_or_none(old) == job["old_sha256"]:
        # Not swapped yet: the verified staging files must still be there, or the job starts again.
        if _sha_or_none(staged) != job["new_sha256"] or _sha_or_none(staged_quality) != job["new_quality_sha256"]:
            job["stage"] = "planned"
            save()
    if job["stage"] == "planned":
        _, quality, _ = stage_and_verify(data_dir, symbol, interval, stage_dir)
        job["new_sha256"] = file_sha256(staged)
        job["new_quality_sha256"] = file_sha256(quality) if quality else None
        job["stage"] = "staged"
        save()
        checkpoint(f"parquet:{job['id']}:staged")


def _swap_parquet_job(
    job: dict[str, Any], data_dir: Path, vault_root: Path, save: Callable[[], None], checkpoint: Checkpoint
) -> None:
    """Phase 2: the old bytes join the vault tree (hard link), then the verified file is renamed over the old path."""
    symbol, interval = job["symbol"], job["interval"]
    label = f"parquet:{job['id']}"
    old, old_quality = data_dir / job["old"], (data_dir / job["old_quality"]) if job["old_quality"] else None
    staged = parquet_path(data_dir / STAGING_DIR, symbol, interval)
    staged_quality = staged.with_name(f"{symbol}.quality.json")
    if job["stage"] == "staged":
        _link_into_vault(old, vault_root / job["old"], job["old_sha256"])
        if old_quality is not None:
            _link_into_vault(old_quality, vault_root / job["old_quality"], job["old_quality_sha256"])
        job["stage"] = "linked"
        save()
        checkpoint(f"{label}:linked")
    if job["stage"] == "linked":
        _swap_in(staged, old, job["old_sha256"], job["new_sha256"])
        checkpoint(f"{label}:parquet-swapped")
        if job["new_quality_sha256"] is not None:
            _swap_in(staged_quality, data_dir / _quality_rel(symbol, interval), job["old_quality_sha256"], job["new_quality_sha256"])
        elif old_quality is not None:
            raise VaultError(f"{symbol} {interval}: the rebuild wrote no quality report to replace the old one")
        job["stage"] = "swapped"
        save()
        checkpoint(f"{label}:swapped")


def _rename_once(source: Path, dest: Path, expected: str) -> bool:
    """Rename ``source`` to ``dest``; a no-op if it is already there. Returns whether a rename happened."""
    if dest.exists():
        if source.exists():
            raise VaultError(f"both {source} and {dest} exist: settle it by hand")
        if file_sha256(dest) != expected:
            raise VaultError(f"{dest} does not have the recorded SHA-256")
        return False
    if not source.is_file():
        raise VaultError(f"{source} is missing, and {dest} does not exist either")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, dest)
    return True


def _archive_paths(job: dict[str, Any], data_dir: Path, vault_root: Path) -> tuple[Path, Path, Path, Path]:
    zip_src, zip_dest = data_dir / "raw" / job["rel"], vault_root / "raw" / job["rel"]
    return zip_src, zip_dest, zip_src.with_name(zip_src.name + CHECKSUM_SUFFIX), zip_dest.with_name(zip_dest.name + CHECKSUM_SUFFIX)


def _check_archive_pins(job: dict[str, Any], data_dir: Path, vault_root: Path) -> None:
    """Refuse an archive or CHECKSUM file that is not the one the journal pinned at plan time.

    A journal written before the CHECKSUM file was pinned has ``None`` there: that pin is taken later, in the job.
    """
    zip_src, _, checksum_src, checksum_dest = _archive_paths(job, data_dir, vault_root)
    if zip_src.exists() and file_sha256(zip_src) != job["sha256"]:
        raise VaultError(f"{zip_src} changed since the plan was made: it is not moved")
    checksum_here = checksum_src if checksum_src.exists() else checksum_dest
    if job["checksum_sha256"] is not None and checksum_here.is_file() and file_sha256(checksum_here) != job["checksum_sha256"]:
        raise VaultError(f"{checksum_here} changed since the plan was made: its archive is not moved")


def _run_archive_job(job: dict[str, Any], data_dir: Path, vault_root: Path, save: Callable[[], None], checkpoint: Checkpoint) -> int:
    rel = job["rel"]
    zip_src, zip_dest, checksum_src, checksum_dest = _archive_paths(job, data_dir, vault_root)
    moved = 0
    if zip_src.exists() and not is_verified(zip_src):
        raise VaultError(f"{zip_src} does not pass is_verified: it is not moved")
    _check_archive_pins(job, data_dir, vault_root)
    checksum_here = checksum_src if checksum_src.exists() else checksum_dest
    if not checksum_here.is_file():
        raise VaultError(f"{checksum_src} is missing")
    if job["checksum_sha256"] is None:
        job["checksum_sha256"] = file_sha256(checksum_here)
        save()
    moved += _rename_once(zip_src, zip_dest, job["sha256"])
    checkpoint(f"archive:{Path(rel).name}:moved")
    moved += _rename_once(checksum_src, checksum_dest, job["checksum_sha256"])
    checkpoint(f"archive:{Path(rel).name}:checksum-moved")
    return moved


@contextmanager
def _os_refusal(step: str) -> Iterator[None]:
    """Turn an ``OSError`` of a move step (a file held open by another process, a full disk) into one refusal.

    Every step of the move is journaled and re-entrant, so nothing is lost: ``move`` run again resumes.
    """
    try:
        yield
    except OSError as error:
        culprit = error.filename2 or error.filename or "a file"
        reason = error.strerror or type(error).__name__
        raise VaultError(
            f"while {step}, {culprit} could not be changed ({type(error).__name__}: {reason}); another process may hold it open. "
            "Nothing was lost; run move again to resume"
        ) from None


def _remove_empty_dirs(folder: Path) -> None:
    if not folder.is_dir():
        return
    for current, _, _ in os.walk(folder, topdown=False):
        try:
            os.rmdir(current)  # fails, harmlessly, on a folder that holds a file
        except OSError:
            pass


def move(data_dir: Path, vault_root: Path, *, checkpoint: Checkpoint = _no_checkpoint) -> MoveResult:
    """Stage 1: move the vault months out of ``data_dir`` into ``vault_root``; see the module docstring.

    Resumable: an unfinished journal is finished first, and a second call after a crash does the rest or nothing.
    ``checkpoint(label)`` is called after every durable step, so a test can raise there to simulate a crash.
    """
    _refuse_if_open()
    check_roots(data_dir, vault_root)
    journal = _load_journal(vault_root)
    resumed = journal is not None and journal["finished"] is None
    if resumed:
        assert journal is not None
        if Path(journal["data_dir"]).resolve() != data_dir.resolve():
            raise VaultError(f"an unfinished move of {journal['data_dir']} is in {vault_root}: finish it with that data root")
    else:
        plan = build_plan(data_dir, vault_root)
        if plan.anomalies:
            raise VaultError("anomalies in the store, nothing moved:\n  " + "\n  ".join(plan.anomalies))
        if not plan.has_work:
            return MoveResult(False, 0, 0, _load_manifest(vault_root).get("digest"), nothing_to_do=True)
        journal = _new_journal(plan)
        _write_json(_journal_path(vault_root), journal)
    checkpoint("journal-written")

    def save() -> None:
        _write_json(_journal_path(vault_root), journal)

    for archive_job in journal["archives"]:  # a changed archive or CHECKSUM file refuses before any table is touched
        with _os_refusal(f"checking {archive_job['rel']}"):
            _check_archive_pins(archive_job, data_dir, vault_root)
    swapped = 0
    for job in journal["parquet"]:  # every rebuild is verified before the first table is touched
        with _os_refusal(f"rebuilding {job['id']}"):
            _stage_parquet_job(job, data_dir, save, checkpoint)
    for job in journal["parquet"]:
        before = job["stage"]
        with _os_refusal(f"swapping {job['id']}"):
            _swap_parquet_job(job, data_dir, vault_root, save, checkpoint)
        swapped += before != "swapped"
    moved = 0
    for archive_job in journal["archives"]:
        with _os_refusal(f"moving {archive_job['rel']}"):
            moved += _run_archive_job(archive_job, data_dir, vault_root, save, checkpoint)
    with _os_refusal("writing the manifest"):
        moved_files = [
            *(_entry(vault_root, job["old"], "parquet") for job in journal["parquet"]),
            *(_entry(vault_root, job["old_quality"], "quality") for job in journal["parquet"] if job["old_quality"]),
            *(_entry(vault_root, "raw/" + job["rel"], "archive") for job in journal["archives"]),
            *(_entry(vault_root, "raw/" + job["rel"] + CHECKSUM_SUFFIX, "checksum") for job in journal["archives"]),
        ]
        manifest = _set_entries(vault_root, moved_files)
        checkpoint("manifest-written")
        journal["finished"] = _iso()
        save()
    _remove_empty_dirs(data_dir / STAGING_DIR)
    return MoveResult(resumed, moved, swapped, manifest["digest"])


# --- Encryption -----------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def parse_recipient(text: str):
    import pyrage

    try:
        return pyrage.x25519.Recipient.from_str(text.strip())
    except pyrage.RecipientError as error:
        raise VaultError(f"not an age X25519 recipient (age1...): {error}") from None


@dataclass
class EncryptResult:
    encrypted: int
    reused: int
    plaintext_removed: int
    manifest_digest: str


def encrypt(vault_root: Path, recipient_text: str, *, remove_plaintext: bool = False) -> EncryptResult:
    """Stage 2: encrypt each plaintext file of the vault tree to ``<name>.age`` for ``recipient_text``.

    The ciphertext is written to a ``.part`` file, flushed, renamed, and hashed from disk; its SHA-256 goes to the
    manifest. A plaintext file is removed only if ``remove_plaintext`` is true, after that hash is recorded. This
    is the one place in the module that deletes a file. Needs no identity: the private key is not on this machine.
    """
    import pyrage

    _refuse_if_open()
    journal = _load_journal(vault_root)
    if journal is not None and journal["finished"] is None:
        raise VaultError("an unfinished move is in the vault tree: run move again first")
    if not (vault_root / MANIFEST_NAME).is_file():
        raise VaultError(f"{vault_root} has no {MANIFEST_NAME}: nothing was moved there, or it is not a vault tree")
    recipient = parse_recipient(recipient_text)
    current = _load_manifest(vault_root)
    by_path = {e["path"]: e for e in current["files"]}
    plain = [p for p in tree_files(vault_root) if not p.endswith(".age")]
    unknown = [p for p in plain if p not in by_path]
    if unknown:
        raise VaultError("files in the vault tree that the manifest does not list: " + ", ".join(unknown))
    changed = [rel for rel in plain if file_sha256(vault_root / rel) != by_path[rel]["sha256"]]
    if changed:
        raise VaultError("files that do not match the SHA-256 in the manifest, nothing encrypted: " + ", ".join(changed))
    if remove_plaintext:
        _require_decrypt_stamp(vault_root, current, by_path, plain, recipient_text.strip())

    def store() -> dict[str, Any]:
        current["files"] = list(by_path.values())
        return _store_manifest(vault_root, current)

    encrypted = reused = removed = 0
    for rel in plain:
        entry, source = by_path[rel], vault_root / rel
        cipher = source.with_name(source.name + ".age")
        known = entry.get("ciphertext")
        if known and known["recipient"] == recipient_text.strip() and _sha_or_none(cipher) == known["sha256"]:
            reused += 1
        else:
            part = cipher.with_name(cipher.name + ".part")
            pyrage.encrypt_file(str(source), str(part), [recipient])
            _fsync_file(part)
            os.replace(part, cipher)
            entry["ciphertext"] = {
                "path": rel + ".age",
                "bytes": cipher.stat().st_size,
                "sha256": file_sha256(cipher),
                "recipient": recipient_text.strip(),
            }
            store()
            encrypted += 1
        if remove_plaintext:
            if file_sha256(cipher) != entry["ciphertext"]["sha256"]:
                raise VaultError(f"{cipher.name} no longer matches its recorded SHA-256: the plaintext stays")
            source.unlink()
            removed += 1
    manifest = store()
    return EncryptResult(encrypted, reused, removed, manifest["digest"])


def _recipient_digest(recipient_text: str) -> str:
    return hashlib.sha256(recipient_text.strip().encode("ascii", errors="replace")).hexdigest()


def _require_decrypt_stamp(
    vault_root: Path, manifest: dict[str, Any], by_path: dict[str, dict[str, Any]], plain: Sequence[str], recipient_text: str
) -> None:
    """Refuse ``--remove-plaintext`` unless ``verify-decrypt`` has shown, for this recipient, that every ciphertext opens.

    The stamp names, for each file, the ciphertext SHA-256 it checked; a file encrypted again since, or for another
    recipient, is not covered. Checked before anything is encrypted or removed.
    """
    stamp = manifest.get(STAMP_KEY)
    if not stamp:
        raise VaultError(
            "no decrypt check in the manifest: run verify-decrypt first, with the identity mounted "
            "(OLSPRIME_VAULT_IDENTITY), so that no plaintext is removed on an unproven ciphertext"
        )
    if stamp.get("recipient_sha256") != _recipient_digest(recipient_text):
        raise VaultError("the decrypt check was made for another recipient than --recipient: run verify-decrypt again")
    covered = stamp.get("covered", {})
    for rel in plain:
        entry = by_path[rel]
        known = entry.get("ciphertext")
        cipher = vault_root / (rel + ".age")
        if not known or known["recipient"] != recipient_text or _sha_or_none(cipher) != known["sha256"] or covered.get(rel) != known["sha256"]:
            raise VaultError(
                f"the decrypt check does not cover {rel}.age (not encrypted yet, encrypted again since, or for another recipient): "
                "run encrypt without --remove-plaintext, then verify-decrypt, then this"
            )


@dataclass
class DecryptCheckResult:
    files: int
    recipient_sha256: str
    manifest_digest: str


def verify_decrypt(vault_root: Path) -> DecryptCheckResult:
    """Decrypt every ``.age`` file in memory with the identity of ``OLSPRIME_VAULT_IDENTITY`` and compare SHA-256s.

    Each plaintext must hash to ``files[].sha256`` in the manifest. On success a stamp (time, count, SHA-256 of the
    identity's public recipient, the ciphertext hash checked per file) goes into the manifest; the identity never
    does, and no decrypted byte is written or printed. Run once by Abdou, with the identity on a mounted medium.
    """
    import pyrage

    _refuse_if_open()
    journal = _load_journal(vault_root)
    if journal is not None and journal["finished"] is None:
        raise VaultError("an unfinished move is in the vault tree: run move again first")
    if not (vault_root / MANIFEST_NAME).is_file():
        raise VaultError(f"{vault_root} has no {MANIFEST_NAME}: nothing was moved there, or it is not a vault tree")
    named = os.environ.get(protocol.IDENTITY_ENV)
    if not named:
        raise VaultError(f"{protocol.IDENTITY_ENV} is not set: mount the medium that holds the identity and name its file")
    try:
        identity = protocol.read_identity(Path(named))
    except protocol.VaultSealed as error:
        raise VaultError(str(error)) from None
    recipient = str(identity.to_public())
    current = _load_manifest(vault_root)
    entries = current["files"]
    if not entries:
        raise VaultError("the manifest lists no file: nothing to check")
    missing = [e["path"] for e in entries if "ciphertext" not in e]
    if missing:
        raise VaultError("files that are not encrypted yet (run encrypt first): " + ", ".join(missing))
    listed = {e["ciphertext"]["path"] for e in entries}
    stray = [p for p in tree_files(vault_root) if p.endswith(".age") and p not in listed]
    if stray:
        raise VaultError("ciphertext files that the manifest does not list: " + ", ".join(stray))
    problems: list[str] = []
    covered: dict[str, str] = {}
    for entry in entries:
        known, cipher = entry["ciphertext"], vault_root / entry["ciphertext"]["path"]
        if not cipher.is_file():
            problems.append(f"{known['path']}: missing")
            continue
        blob = cipher.read_bytes()
        if hashlib.sha256(blob).hexdigest() != known["sha256"]:
            problems.append(f"{known['path']}: does not match the ciphertext SHA-256 in the manifest")
            continue
        if known["recipient"] != recipient:
            problems.append(f"{known['path']}: encrypted for {known['recipient']}, not for the recipient of this identity")
            continue
        try:
            plain = pyrage.decrypt(blob, [identity])
        except pyrage.DecryptError as error:
            problems.append(f"{known['path']}: does not decrypt with this identity ({error})")
            continue
        if hashlib.sha256(plain).hexdigest() != entry["sha256"]:
            problems.append(f"{known['path']}: decrypts to bytes other than the recorded SHA-256")
            continue
        covered[entry["path"]] = known["sha256"]
    if problems:
        raise VaultError(f"{len(problems)} of {len(entries)} files fail the decrypt check, no stamp written:\n  " + "\n  ".join(problems))
    current[STAMP_KEY] = {
        "at": _iso(),
        "files": len(covered),
        "recipient_sha256": _recipient_digest(recipient),
        "covered": covered,
    }
    manifest = _store_manifest(vault_root, current)
    return DecryptCheckResult(len(covered), current[STAMP_KEY]["recipient_sha256"], manifest["digest"])


def keygen(out: Path, data_dir: Path, vault_root: Path) -> str:
    """Write a new age identity to ``out`` and return its public recipient; refuse a path in the repository, the data or vault tree."""
    import pyrage

    for label, folder in (("the repository", paths.REPO_ROOT), ("the data root", data_dir), ("the vault tree", vault_root)):
        if _inside(out, folder):
            raise VaultError(f"{out} is inside {label}: the private key must leave this machine, write it to a medium you take away")
    if out.exists():
        raise VaultError(f"{out} exists: keygen never overwrites a file")
    if not out.parent.is_dir():
        raise VaultError(f"{out.parent} does not exist: mount the medium first")
    identity = pyrage.x25519.Identity.generate()
    recipient = str(identity.to_public())
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# created: {_iso()}\n# public key: {recipient}\n{identity}\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        out.chmod(0o600)
    except OSError:
        pass
    return recipient


# --- Status ---------------------------------------------------------------------------


def status(data_dir: Path, vault_root: Path) -> dict[str, Any]:
    """Where the vault stands: stage 0 (in ``data/``), 1 (moved, plaintext) or 2 (encrypted, plaintext removed).

    Reads names, footers and sizes only. ``vault_months_in_data`` lists what is still under ``data/``.
    """
    archives, anomalies = scan_vault_archives(data_dir / "raw")
    holding, _ = vault_bars_by_statistics(data_dir / "parquet")
    in_data = ["raw/" + a.zip_rel for a in archives] + [f"{s} {i} (Parquet footer)" for s, i in holding]
    files = tree_files(vault_root)
    encrypted = [p for p in files if p.endswith(".age")]
    plaintext = [p for p in files if not p.endswith(".age")]
    size = sum((vault_root / p).stat().st_size for p in files)
    journal = _load_journal(vault_root)
    if in_data:
        stage = 0
    elif plaintext:
        stage = 1
    elif encrypted:
        stage = 2
    else:
        stage = 0
    manifest_path = vault_root / MANIFEST_NAME
    digest = None
    manifest_ok = None
    stamp = None
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = stored.get("digest")
        manifest_ok = digest == manifest_digest(stored)
        stamp = stored.get(STAMP_KEY)
    return {
        "stage": stage,
        "vault_months_in_data": in_data,
        "archives_still_in_data": len(archives),
        "tables_still_in_data": len(holding),
        "decrypt_check": {k: stamp[k] for k in ("at", "files", "recipient_sha256")} if stamp else None,
        "anomalies": anomalies,
        "tree_files": len(files),
        "tree_bytes": size,
        "plaintext_files": len(plaintext),
        "encrypted_files": len(encrypted),
        "unfinished_move": journal is not None and journal["finished"] is None,
        "manifest_digest": digest,
        "manifest_ok": manifest_ok,
    }


def tree_summary(data_dir: Path | None = None, vault_root: Path | None = None) -> dict[str, Any]:
    """What :mod:`olsprime.manifest` reports about the vault tree: stage, file count, bytes, its manifest's digest."""
    state = status(data_dir if data_dir is not None else paths.data_dir(), vault_root if vault_root is not None else paths.vault_dir())
    return {
        "stage": state["stage"],
        "files": state["tree_files"],
        "bytes": state["tree_bytes"],
        "archives_still_in_data": state["archives_still_in_data"],
        "tables_still_in_data": state["tables_still_in_data"],
        "manifest_digest": state["manifest_digest"],
        "manifest_ok": state["manifest_ok"],
    }


def format_status(state: dict[str, Any]) -> str:
    meaning = {0: "in data/ (nothing moved yet)", 1: "moved, plaintext", 2: "encrypted, plaintext removed"}[state["stage"]]
    lines = [
        f"stage {state['stage']}: {meaning}",
        f"vault tree: {state['tree_files']} files, {state['tree_bytes']:,} bytes "
        f"({state['plaintext_files']} plaintext, {state['encrypted_files']} encrypted)",
        f"manifest digest: {state['manifest_digest']}" + ("" if state["manifest_ok"] in (None, True) else "  (DOES NOT MATCH ITS BODY)"),
    ]
    if state["unfinished_move"]:
        lines.append("an unfinished move is journaled: run move again")
    check = state["decrypt_check"]
    if check:
        lines.append(f"decrypt check: {check['files']} files at {check['at']}, recipient digest {check['recipient_sha256'][:16]}...")
    lines.append(
        f"still under data/: {state['archives_still_in_data']} vault archives and {state['tables_still_in_data']} tables holding vault bars"
    )
    if state["vault_months_in_data"]:
        lines.extend(f"  {text}" for text in state["vault_months_in_data"])
    if state["anomalies"]:
        lines.append(f"anomalies: {len(state['anomalies'])}")
        lines.extend(f"  {text}" for text in state["anomalies"])
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olsprime.vault",
        description="The physical seal: move the vault months out of data/, then encrypt them to an age recipient.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="data root (default: $OLSPRIME_DATA_DIR or <repo>/data)")
    parser.add_argument("--vault-dir", type=Path, default=None, help="vault root (default: $OLSPRIME_VAULT_DIR or <repo parent>/ols-prime-vault)")
    commands = parser.add_subparsers(dest="command", required=True)
    plan_cmd = commands.add_parser("plan", help="dry run: list what move would do, change nothing")
    plan_cmd.add_argument("--verify", action="store_true", help="also rebuild each file in a temporary folder and compare the sealed reads")
    commands.add_parser("move", help="stage 1: rename the vault months into the vault tree (resumable)")
    encrypt_cmd = commands.add_parser("encrypt", help="stage 2: encrypt the vault tree to an age recipient")
    encrypt_cmd.add_argument("--recipient", required=True, help="age1... public key whose private key is not on this machine")
    encrypt_cmd.add_argument(
        "--remove-plaintext",
        action="store_true",
        help="delete each plaintext file once its ciphertext is written and re-hashed (Abdou only)",
    )
    keygen_cmd = commands.add_parser("keygen", help="write a new age identity to --out and print its public recipient (Abdou only)")
    keygen_cmd.add_argument("--out", type=Path, required=True, help="a path on a medium you take away")
    commands.add_parser(
        "verify-decrypt",
        help="decrypt every .age file in memory with $OLSPRIME_VAULT_IDENTITY, compare SHA-256s, stamp the manifest (Abdou only)",
    )
    commands.add_parser("status", help="stage, file counts and bytes, vault months still under data/")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    data_dir = args.data_dir if args.data_dir is not None else paths.data_dir()
    vault_root = args.vault_dir if args.vault_dir is not None else paths.vault_dir()
    try:
        if args.command == "plan":
            plan = build_plan(data_dir, vault_root)
            print(format_plan(plan, verify_rebuilds(plan) if args.verify else None))
            return 1 if plan.anomalies else 0
        if args.command == "move":
            result = move(data_dir, vault_root)
            if result.nothing_to_do:
                print("nothing to do: no vault month under data/")
            else:
                print(
                    f"{'resumed and ' if result.resumed else ''}finished: {result.archives_moved} files renamed, "
                    f"{result.parquet_swapped} Parquet files rebuilt; manifest digest {result.manifest_digest}"
                )
            return 0
        if args.command == "encrypt":
            outcome = encrypt(vault_root, args.recipient, remove_plaintext=args.remove_plaintext)
            print(
                f"{outcome.encrypted} encrypted, {outcome.reused} already encrypted, {outcome.plaintext_removed} plaintext removed; "
                f"manifest digest {outcome.manifest_digest}"
            )
            return 0
        if args.command == "keygen":
            print(keygen(args.out, data_dir, vault_root))
            return 0
        if args.command == "verify-decrypt":
            checked = verify_decrypt(vault_root)
            print(
                f"{checked.files} files decrypt to their recorded SHA-256; stamp written (recipient digest {checked.recipient_sha256}); "
                f"manifest digest {checked.manifest_digest}"
            )
            return 0
        print(format_status(status(data_dir, vault_root)))
        return 0
    except VaultError as error:
        print(f"refused: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
