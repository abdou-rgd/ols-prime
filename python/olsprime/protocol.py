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
"""

from __future__ import annotations

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


def load_klines_file(
    path: Path,
    interval: str,
    *,
    columns: list[str] | None = None,
    caller: str,
) -> pl.DataFrame:
    """Read a klines Parquet file and seal it; see :func:`load_klines`."""
    interval_ms(interval)  # fail early on an unknown interval
    if columns is None:
        frame = pl.read_parquet(path)
        out_columns = [*frame.columns, "known_at"]
    else:
        out_columns = list(columns)
        stored = [c for c in out_columns if c != "known_at"]
        needed = [c for c in ("open_time", "close_time") if c not in stored]
        frame = pl.read_parquet(path, columns=stored + needed)
    frame = frame.with_columns(known_at_expr(interval))
    frame = seal(frame, time_col="known_at", caller=caller)
    return frame.select(out_columns)


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
    """
    root = data_dir if data_dir is not None else paths.data_dir()
    path = parquet_path(root / "parquet", symbol, interval)
    return load_klines_file(path, interval, columns=columns, caller=caller or f"load_klines({symbol},{interval})")


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


def _span(times: pl.Series) -> tuple[int, datetime | None, datetime | None]:
    return times.len(), times.min(), times.max()  # min and max of an empty series are None


def sealed_klines_span(path: Path, interval: str, *, caller: str) -> tuple[int, datetime | None, datetime | None]:
    """``(rows, first open_time, last open_time)`` of a klines file once sealed (see :func:`load_klines_file`)."""
    return _span(load_klines_file(path, interval, columns=["open_time"], caller=caller)["open_time"])


def sealed_universe_span(path: Path, *, caller: str) -> tuple[int, datetime | None, datetime | None]:
    """``(rows, first rebalance_time, last rebalance_time)`` of a universe file once sealed."""
    return _span(load_universe(path, columns=["rebalance_time"], caller=caller)["rebalance_time"])
