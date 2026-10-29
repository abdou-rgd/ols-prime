"""The splits of ``docs/PROTOCOL.md`` in code, and the lock on the vault.

The vault (2026-01 -> 2026-10) is opened once, at the end, for all final models
together. Every read of stored klines for modelling goes through :func:`load_klines`
(or :func:`load_klines_file`), which calls :func:`seal`; no other module reads a
Parquet file. While the vault is sealed, a bar whose values become final at or after
``VAULT.start`` (its ``known_at``, see :func:`load_klines`) never leaves this module.

The vault opens only with two keys at once:

- the environment variable ``OLSPRIME_OPEN_VAULT=1``;
- the file ``<runs>/vault/OPEN``, which a person writes by hand at the final
  evaluation: the date (ISO, ``2026-11-01``) alone on the first line, then the ids of
  the final models. An empty file, or a first line that is not a date, raises
  :class:`VaultSealed`: the file is a decision, not a ``touch``.

The universe file is a stored table too: :func:`load_universe` seals it on its
``rebalance_time``. The read helpers for the vault manifest (:func:`stored_row_count`,
:func:`sealed_klines_span`, :func:`sealed_universe_span`) live here for the same reason:
no other module opens a Parquet file.

Each read with the vault open appends a line to ``<runs>/vault/ACCESS.log``.
Downloading and the structural checks of ``build_parquet`` are not reads for
modelling and stay outside this lock.

The physical seal (iteration 3, ``python -m olsprime.vault``) moves the vault months out of ``data/``
into a vault tree (:func:`olsprime.paths.vault_dir`). With both keys in place, :func:`load_klines` adds
the rows of the vault tree to the pair's table:

- stage 1: the plaintext Parquet file of the vault tree, read through :func:`load_klines_file`;
- stage 2: the ``.age`` file of the vault tree, decrypted in memory with the identity named by
  ``OLSPRIME_VAULT_IDENTITY``. Without that identity (or with another) nothing opens: :class:`VaultSealed`.

Without both keys nothing reads the vault tree. It is the one thing the file system keeps from a
process that has taken the guards down: at stage 2 the rows are ciphertext without the identity.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from olsprime.data.binance_vision import parquet_path
from olsprime.data.klines import TIMESTAMP, interval_ms
from olsprime import paths
from olsprime.paths import runs_dir

OPEN_ENV = "OLSPRIME_OPEN_VAULT"
IDENTITY_ENV = "OLSPRIME_VAULT_IDENTITY"
ENCRYPTED_SUFFIX = ".age"


def _is_utc(value: datetime) -> bool:
    zone = value.tzinfo
    return zone is timezone.utc or (isinstance(zone, ZoneInfo) and zone.key == "UTC")


@dataclass(frozen=True)
class Period:
    """A half-open interval ``[start, end)`` of UTC instants."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            value = getattr(self, name)
            if not _is_utc(value):
                raise ValueError(
                    f"{name} must be a UTC datetime (tzinfo datetime.UTC or ZoneInfo('UTC'); "
                    f"a zone that merely has offset 0 today, such as Europe/London, is not UTC), got {value!r}"
                )
        if not self.start < self.end:
            raise ValueError(f"empty period: {self.start} >= {self.end}")

    def __contains__(self, when: datetime) -> bool:
        return self.start <= when < self.end


def _utc(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


# Keep in step with the split table of docs/PROTOCOL.md (a test compares the two).
DEVELOPMENT = Period(_utc(2017, 8), _utc(2025, 1))
VALIDATION = Period(_utc(2025, 1), _utc(2026, 1))
VAULT = Period(_utc(2026, 1), _utc(2026, 11))


class VaultSealed(RuntimeError):
    """Something asked for vault data while the vault is sealed."""


def vault_key_file() -> Path:
    return runs_dir() / "vault" / "OPEN"


def access_log_file() -> Path:
    return runs_dir() / "vault" / "ACCESS.log"


def vault_open() -> bool:
    """Whether both keys are present: ``OLSPRIME_OPEN_VAULT=1`` and a dated ``<runs>/vault/OPEN``.

    With the variable unset (or not ``1``) the vault is sealed, whatever the file holds.
    With the variable set and the file present, the first line of the file must be an ISO
    date; otherwise :class:`VaultSealed` is raised, so a bare ``touch`` never opens it.
    """
    if os.environ.get(OPEN_ENV) != "1":
        return False
    key = vault_key_file()
    if not key.is_file():
        return False
    try:
        lines = key.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = []
    first = lines[0].strip() if lines else ""
    try:
        date.fromisoformat(first)
    except ValueError:
        raise VaultSealed(f"OPEN file has no date: the first line of {key} must be an ISO date, got {first!r}") from None
    return True


def seal(frame: pl.DataFrame, *, time_col: str = "open_time", caller: str) -> pl.DataFrame:
    """Return ``frame`` without the rows of the vault, unless the vault is open.

    ``time_col`` must be a ``Datetime("ms", "UTC")`` column, so that an integer or a
    string is never compared with the cut-off by mistake (``TypeError`` otherwise).
    Sealed: only rows with ``time_col < VAULT.start`` are returned (a null time is
    dropped). If the frame also has an ``open_time`` column (``Datetime("ms", "UTC")``
    as well), the rows whose ``open_time`` is in the vault are dropped too: a bar that
    opens in the vault is final in the vault, so this never drops a row the ``time_col``
    seal keeps, and it defeats a forged ``time_col`` (``known_at`` set to 2020 by hand).
    Open: every row is returned and one line is appended to ``ACCESS.log``:
    UTC time, caller, rows returned, vault rows among them.
    """
    if time_col not in frame.columns:
        raise TypeError(f"cannot seal: no column {time_col!r}")
    dtype = frame.schema[time_col]
    if dtype != TIMESTAMP:
        raise TypeError(f"cannot seal: {time_col!r} is {dtype}, expected {TIMESTAMP}")
    before_vault = pl.col(time_col) < VAULT.start
    if not vault_open():
        if time_col != "open_time" and "open_time" in frame.columns:
            opened = frame.schema["open_time"]
            if opened != TIMESTAMP:
                raise TypeError(f"cannot seal: 'open_time' is {opened}, expected {TIMESTAMP}")
            before_vault = before_vault & (pl.col("open_time") < VAULT.start)
        return frame.filter(before_vault)
    vault_rows = frame.select((~before_vault.fill_null(False)).sum()).item()
    log = access_log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    caller_text = " ".join(caller.split())  # one line per read
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{caller_text}\trows={frame.height}\tvault_rows={vault_rows}\n")
    return frame


def known_at_expr(interval: str) -> pl.Expr:
    """``max(close_time, open_time + interval - 1 ms)``: when a bar's values are final."""
    nominal = pl.col("open_time") + pl.duration(milliseconds=interval_ms(interval) - 1, time_unit="ms")
    return pl.max_horizontal(pl.col("close_time"), nominal).alias("known_at")


def _stored_columns(columns: list[str] | None) -> list[str] | None:
    """The columns to read from the file for the requested ``columns``: ``known_at`` is computed, not stored."""
    if columns is None:
        return None
    stored = [c for c in columns if c != "known_at"]
    needed = [c for c in ("open_time", "close_time") if c not in stored]
    return stored + needed


def _sealed_frame(frame: pl.DataFrame, interval: str, columns: list[str] | None, caller: str) -> pl.DataFrame:
    """Add ``known_at``, seal on it, and select ``columns`` (every stored column and ``known_at`` when ``None``)."""
    out_columns = [*frame.columns, "known_at"] if columns is None else list(columns)
    frame = frame.with_columns(known_at_expr(interval))
    frame = seal(frame, time_col="known_at", caller=caller)
    return frame.select(out_columns)


def load_klines_file(
    path: Path,
    interval: str,
    *,
    columns: list[str] | None = None,
    caller: str,
) -> pl.DataFrame:
    """Read a klines Parquet file and seal it; see :func:`load_klines`."""
    interval_ms(interval)  # fail early on an unknown interval
    stored = _stored_columns(columns)
    frame = pl.read_parquet(path) if stored is None else pl.read_parquet(path, columns=stored)
    return _sealed_frame(frame, interval, columns, caller)


def load_klines(
    symbol: str,
    interval: str,
    *,
    data_dir: Path | None = None,
    columns: list[str] | None = None,
    caller: str | None = None,
) -> pl.DataFrame:
    """Stored klines of a pair, sealed: no bar final at or after ``VAULT.start`` unless the vault is open.

    ``data_dir`` is the data root (default: :func:`olsprime.paths.data_dir`).

    ``known_at`` is the time a bar's values are final, ``max(close_time, open_time +
    interval - 1 ms)``. The stored ``close_time`` alone is wrong for 15 bars of the 1 h
    series, once earlier than the open (``docs/DATA.md``, "Bad close times"), and
    ``open_time + interval - 1 ms`` alone is 1 ms early for the bar of 2017-09-06
    15:00; the later of the two is never early. The column comes when ``columns`` is
    ``None`` or lists ``"known_at"``; ``columns=["close"]`` returns ``close`` alone.

    The seal cuts on ``known_at``, for every interval: a bar is kept only if its values
    were final before ``VAULT.start``. A 1 w bar opened on 2025-12-29 is dropped.
    ``open_time`` and ``close_time`` are always read for it, and dropped afterwards
    when ``columns`` leaves them out.

    With both keys in place the pair's file in the vault tree is read too (plain or decrypted in
    memory, see the module docstring), the two parts are joined on ``open_time`` (the data file's
    row wins) and sorted; each part is logged on its own. Sealed, the vault tree is not touched.
    """
    root = data_dir if data_dir is not None else paths.data_dir()
    path = parquet_path(root / "parquet", symbol, interval)
    who = caller or f"load_klines({symbol},{interval})"
    if not vault_open():
        return load_klines_file(path, interval, columns=columns, caller=who)
    wanted = None if columns is None else list(dict.fromkeys([*columns, "open_time"]))
    frame = load_klines_file(path, interval, columns=wanted, caller=who)
    part = _vault_tree_part(symbol, interval, wanted, who)
    if part is not None:
        frame = pl.concat([frame, part]).unique(subset="open_time", keep="first", maintain_order=True).sort("open_time")
    return frame.select(frame.columns if columns is None else columns)


# --- The vault tree (physical seal) ---------------------------------------------------


def vault_tree_klines_path(symbol: str, interval: str, root: Path | None = None) -> Path:
    """Where the vault tree keeps a pair's Parquet file (plain); the encrypted one adds ``.age``."""
    return parquet_path((root if root is not None else paths.vault_dir()) / "parquet", symbol, interval)


def read_identity(path: Path):
    """The age X25519 identity in ``path`` (an age key file: ``#`` comment lines, then ``AGE-SECRET-KEY-1...``)."""
    import pyrage

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise VaultSealed(f"cannot read the identity file {path}: {error}") from None
    secret = next((line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")), "")
    try:
        return pyrage.x25519.Identity.from_str(secret)
    except pyrage.IdentityError:
        raise VaultSealed(f"{path} holds no age identity (an AGE-SECRET-KEY-1... line)") from None


def decrypt_vault_bytes(ciphertext: bytes, *, what: str) -> bytes:
    """Decrypt ``ciphertext`` with the identity named by ``OLSPRIME_VAULT_IDENTITY``; :class:`VaultSealed` if it cannot."""
    import pyrage

    named = os.environ.get(IDENTITY_ENV)
    if not named:
        raise VaultSealed(f"{what} is encrypted and {IDENTITY_ENV} is not set: stage 2 opens only with the identity")
    identity = read_identity(Path(named))
    try:
        return pyrage.decrypt(ciphertext, [identity])
    except pyrage.DecryptError as error:
        raise VaultSealed(f"{what}: the identity in {named} does not open it ({error})") from None


def _vault_tree_part(symbol: str, interval: str, columns: list[str] | None, caller: str) -> pl.DataFrame | None:
    """The pair's rows in the vault tree, sealed like any read (so logged, with the vault open); ``None`` if it has none.

    Only called with both keys in place. A plain file (stage 1) is read through :func:`load_klines_file`; an
    encrypted one (stage 2) is decrypted in memory.
    """
    interval_ms(interval)
    plain = vault_tree_klines_path(symbol, interval)
    who = f"{caller} [vault tree]"
    if plain.is_file():
        return load_klines_file(plain, interval, columns=columns, caller=who)
    cipher = plain.with_name(plain.name + ENCRYPTED_SUFFIX)
    if cipher.is_file():
        data = decrypt_vault_bytes(cipher.read_bytes(), what=cipher.name)
        frame = pl.read_parquet(io.BytesIO(data), columns=_stored_columns(columns))
        return _sealed_frame(frame, interval, columns, who)
    return None


# --- Universe and manifest helpers ---------------------------------------------------


def load_universe(path: Path, *, columns: list[str] | None = None, caller: str) -> pl.DataFrame:
    """Read a universe Parquet file and seal it on ``rebalance_time``.

    Sealed: only rebalances with ``rebalance_time < VAULT.start`` come back (a rebalance
    at 2026-01-01 00:00 UTC belongs to the vault). Open: every row comes back and the
    read is logged, as in :func:`seal`. ``rebalance_time`` is always read for the cut
    and dropped afterwards when ``columns`` leaves it out.
    """
    if columns is None:
        frame = pl.read_parquet(path)
        out_columns = list(frame.columns)
    else:
        out_columns = list(columns)
        frame = pl.read_parquet(path, columns=out_columns if "rebalance_time" in out_columns else [*out_columns, "rebalance_time"])
    return seal(frame, time_col="rebalance_time", caller=caller).select(out_columns)


def stored_row_count(path: Path) -> int:
    """Rows in a Parquet file as the file's own metadata states them, vault rows included.

    No row is read and nothing is sealed: a count discloses no value. Use
    :func:`sealed_klines_span` or :func:`sealed_universe_span` for what a model may see.
    """
    return pq.read_metadata(path).num_rows


def stored_open_time_bounds(path: Path) -> tuple[datetime, datetime] | None:
    """``(first, last)`` ``open_time`` as the Parquet statistics state them, vault bars included; ``None`` without statistics.

    No row is read: the statistics are the file's own footer. Like :func:`stored_row_count`, it tells
    whether a file holds bars of the vault, not what they hold.
    """
    meta = pq.read_metadata(path)
    if "open_time" not in meta.schema.names:
        return None
    index = meta.schema.names.index("open_time")
    lows, highs = [], []
    for group in range(meta.num_row_groups):
        stats = meta.row_group(group).column(index).statistics
        if stats is None or not stats.has_min_max:
            return None
        lows.append(stats.min)
        highs.append(stats.max)
    if not lows:
        return None
    return _as_utc(min(lows)), _as_utc(max(highs))


def _as_utc(when: datetime) -> datetime:
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


def _span(times: pl.Series) -> tuple[int, datetime | None, datetime | None]:
    return times.len(), times.min(), times.max()  # min and max of an empty series are None


def sealed_klines_span(path: Path, interval: str, *, caller: str) -> tuple[int, datetime | None, datetime | None]:
    """``(rows, first open_time, last open_time)`` of a klines file once sealed (see :func:`load_klines_file`)."""
    return _span(load_klines_file(path, interval, columns=["open_time"], caller=caller)["open_time"])


def sealed_universe_span(path: Path, *, caller: str) -> tuple[int, datetime | None, datetime | None]:
    """``(rows, first rebalance_time, last rebalance_time)`` of a universe file once sealed."""
    return _span(load_universe(path, columns=["rebalance_time"], caller=caller)["rebalance_time"])
