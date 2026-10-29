"""Download Binance spot klines from data.binance.vision and store them as Parquet.

Monthly archives are published at::

    https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.zip

with a ``.zip.CHECKSUM`` file beside each one holding its SHA-256. An archive enters
the cache under ``data/raw/`` (which mirrors the bucket layout) only once its digest
matches, and a verified archive is never downloaded again. A month without an
archive (HTTP 404: before the listing, after the delisting, or not published yet)
is reported as missing, not raised.

Each archive holds one headerless CSV with 12 columns. Binance switched spot
timestamps from milliseconds to microseconds on 2025-01-01; the unit is read from
each value's order of magnitude, never from the date, and ``open_time`` and
``close_time`` are stored as milliseconds since the epoch, UTC.

The Parquet file of a (symbol, interval) pair is rebuilt from every verified archive
of a month before the vault in the cache, so a narrower run never truncates it.

Months at or after ``VAULT.start`` (``docs/PROTOCOL.md``, "Splits") are not downloaded
here and never enter a Parquet file built here: they live in the vault tree outside
``data/`` (``python -m olsprime.vault``). The final evaluation will fetch them through the
vault module, a later step. Usage::

    uv run python -m olsprime.data.binance_vision --symbols BTCUSDT ETHUSDT \\
        --intervals 1h 1d --start 2017-08 --end 2025-12
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import httpx
import polars as pl

from olsprime import __version__
from olsprime.data.klines import INTERVAL_MS, KLINE_SCHEMA, TIMESTAMP
from olsprime.data.quality import QualityReport, check_quality
from olsprime.paths import data_dir

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision"
KLINES_PREFIX = "spot/monthly/klines"

# Column order of the archived CSV files (they have no header).
_CSV_SCHEMA: dict[str, pl.DataType] = {
    "open_time": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "close_time": pl.Int64(),
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
    "taker_buy_base_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
    "ignore": pl.String(),
}

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, order=True)
class Month:
    """A calendar month, the granularity of the monthly archives."""

    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"month must be in 1..12, got {self.month}")

    @classmethod
    def parse(cls, text: str) -> Month:
        """Parse ``YYYY-MM``."""
        match = re.fullmatch(r"(\d{4})-(\d{2})", text.strip())
        if match is None:
            raise ValueError(f"expected YYYY-MM, got {text!r}")
        return cls(int(match[1]), int(match[2]))

    @classmethod
    def of(cls, when: datetime) -> Month:
        """The month containing ``when`` (taken as is: convert to UTC first)."""
        return cls(when.year, when.month)

    def shift(self, months: int) -> Month:
        index = self.year * 12 + self.month - 1 + months
        return Month(index // 12, index % 12 + 1)

    def start(self) -> datetime:
        """First instant of the month, UTC."""
        return datetime(self.year, self.month, 1, tzinfo=UTC)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


class VaultMonthRefused(ValueError):
    """A download or a rebuild was asked for a month at or after ``VAULT.start``."""


def vault_first_month() -> Month:
    """The first vault month (2026-01), read from ``olsprime.protocol.VAULT``.

    Imported at the call: ``olsprime.protocol`` imports this module.
    """
    from olsprime.protocol import VAULT

    return Month.of(VAULT.start)


def is_vault_month(month: Month) -> bool:
    return month >= vault_first_month()


def vault_refusal(what: str, months: Iterable[Month]) -> str:
    """The message of a refusal: which months, and where they are fetched instead."""
    first = vault_first_month()
    named = month_spans([m for m in months if m >= first])
    return (
        f"{what}: {named} belong to the vault (months from {first}, docs/PROTOCOL.md). They are not downloaded here; "
        "the final evaluation will fetch them through olsprime.vault, a later step that is not built yet."
    )


def month_range(start: Month, end: Month) -> list[Month]:
    """Months from ``start`` to ``end``, both included."""
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current.shift(1)
    return months


def last_complete_month() -> Month:
    return Month.of(datetime.now(UTC)).shift(-1)


def month_arg(text: str) -> Month:
    """argparse type for ``YYYY-MM`` arguments."""
    try:
        return Month.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


# --- Archive layout -------------------------------------------------------------


def archive_relpath(symbol: str, interval: str, month: Month) -> str:
    """Path of a monthly archive below the bucket's ``data/`` root, and below ``data/raw/``."""
    return f"{KLINES_PREFIX}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def cached_archives(raw_dir: Path, symbol: str, interval: str) -> list[Path]:
    """Archives of a pair present in the cache, oldest first, verified or not."""
    return sorted((raw_dir / KLINES_PREFIX / symbol / interval).glob(f"{symbol}-{interval}-*.zip"))


_ARCHIVE_NAME = re.compile(r"^(?P<symbol>[A-Z0-9]+)-(?P<interval>[0-9]+[smhdwM])-(?P<month>\d{4}-\d{2})\.zip$")


def archive_name_parts(name: str) -> tuple[str, str, Month] | None:
    """``(symbol, interval, month)`` read from an archive's file name, or ``None`` if it is not one."""
    match = _ARCHIVE_NAME.match(name)
    if match is None:
        return None
    try:
        return match["symbol"], match["interval"], Month.parse(match["month"])
    except ValueError:  # a month like 2026-13
        return None


def parquet_path(parquet_dir: Path, symbol: str, interval: str) -> Path:
    return parquet_dir / "spot" / "klines" / interval / f"{symbol}.parquet"


def _checksum_path(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.name + ".CHECKSUM")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, path)


# --- Checksums ------------------------------------------------------------------


class ChecksumError(ValueError):
    """An archive does not match its published SHA-256, or a CHECKSUM file is malformed."""


def parse_checksum(text: str) -> str:
    """Hex digest of a ``sha256sum``-style CHECKSUM file: ``<digest>  <file name>``."""
    fields = text.split()
    digest = fields[0].lower() if fields else ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ChecksumError(f"malformed CHECKSUM file: {text[:120]!r}")
    return digest


def verify_checksum(data: bytes, expected: str, name: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ChecksumError(f"{name}: SHA-256 is {actual}, published {expected}")


def is_verified(zip_path: Path) -> bool:
    """Whether a cached archive exists and matches the CHECKSUM file stored beside it."""
    checksum_path = _checksum_path(zip_path)
    if not (zip_path.is_file() and checksum_path.is_file()):
        return False
    try:
        expected = parse_checksum(checksum_path.read_text(encoding="ascii", errors="replace"))
    except ChecksumError:
        return False
    with zip_path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == expected


# --- HTTP -----------------------------------------------------------------------


def make_client(workers: int = 8, timeout: float = 30.0) -> httpx.Client:
    """An HTTP client sized for ``workers`` concurrent requests."""
    return httpx.Client(
        timeout=timeout,
        limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
        headers={"User-Agent": f"olsprime/{__version__}"},
        follow_redirects=True,
    )


def fetch(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str] | None = None,
    attempts: int = 5,
    backoff: float = 1.0,
) -> bytes | None:
    """GET ``url`` and return the body, or ``None`` on 404.

    Transport errors, 429 and 5xx answers are retried with exponential backoff; any
    other error status raises :class:`httpx.HTTPStatusError` at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code == 404:
                return None
            return response.raise_for_status().content
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = (
                not isinstance(exc, httpx.HTTPStatusError)
                or exc.response.status_code in _RETRYABLE_STATUS
            )
            if not retryable or attempt == attempts:
                raise
            delay = backoff * 2 ** (attempt - 1)
            log.warning("GET %s failed (%s), retry %d in %.0f s", url, exc, attempt, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


# --- Download -------------------------------------------------------------------


class Status(StrEnum):
    DOWNLOADED = "downloaded"
    CACHED = "cached"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True)
class MonthResult:
    symbol: str
    interval: str
    month: Month
    status: Status
    error: str | None = None


def _fetch_verified(client: httpx.Client, url: str) -> tuple[bytes, bytes] | None:
    """Archive and CHECKSUM file once they agree; ``None`` when the month has no archive."""
    for attempt in (1, 2):  # retry once: the CHECKSUM may be replaced between our two requests
        checksum = fetch(client, url + ".CHECKSUM")
        if checksum is None:
            return None
        archive = fetch(client, url)
        if archive is None:
            log.warning("%s: CHECKSUM published without its archive", url)
            return None
        try:
            verify_checksum(archive, parse_checksum(checksum.decode("ascii", "replace")), url)
        except ChecksumError:
            if attempt == 2:
                raise
        else:
            return archive, checksum
    raise AssertionError("unreachable")


def download_month(
    client: httpx.Client,
    symbol: str,
    interval: str,
    month: Month,
    *,
    raw_dir: Path,
    force: bool = False,
) -> MonthResult:
    """Fetch one monthly archive into the cache, unless a verified copy is already there.

    A month at or after ``VAULT.start`` raises :class:`VaultMonthRefused`, before any request.
    """
    if is_vault_month(month):
        raise VaultMonthRefused(vault_refusal(f"{symbol} {interval}", [month]))
    relpath = archive_relpath(symbol, interval, month)
    zip_path = raw_dir / relpath
    if not force and is_verified(zip_path):
        return MonthResult(symbol, interval, month, Status.CACHED)
    try:
        fetched = _fetch_verified(client, f"{BASE_URL}/data/{relpath}")
    except (httpx.HTTPError, ChecksumError) as exc:
        log.error("%s: %s", relpath, exc)
        return MonthResult(symbol, interval, month, Status.FAILED, str(exc))
    if fetched is None:
        return MonthResult(symbol, interval, month, Status.MISSING)
    archive, checksum = fetched
    _atomic_write(zip_path, archive)
    _atomic_write(_checksum_path(zip_path), checksum)
    return MonthResult(symbol, interval, month, Status.DOWNLOADED)


def download_archives(
    tasks: Iterable[tuple[str, str, Month]],
    *,
    raw_dir: Path,
    client: httpx.Client | None = None,
    workers: int = 8,
    force: bool = False,
) -> list[MonthResult]:
    """Fetch (symbol, interval, month) archives concurrently; results keep the task order.

    Raises :class:`VaultMonthRefused` before any request if one task names a vault month.
    """
    tasks = list(tasks)
    vault_months = [month for _, _, month in tasks if is_vault_month(month)]
    if vault_months:
        raise VaultMonthRefused(vault_refusal("download", vault_months))
    owned = client is None
    client = client or make_client(workers)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(download_month, client, symbol, interval, month, raw_dir=raw_dir, force=force)
                for symbol, interval, month in tasks
            ]
            results = []
            step = max(1, len(futures) // 10)
            for done, future in enumerate(futures, 1):
                results.append(future.result())
                if done % step == 0 or done == len(futures):
                    log.info("%d/%d archives done", done, len(futures))
    finally:
        if owned:
            client.close()
    return results


def download_klines(
    symbols: Iterable[str],
    intervals: Iterable[str],
    months: Iterable[Month],
    **kwargs,
) -> list[MonthResult]:
    """Fetch every month of every (symbol, interval) pair; see :func:`download_archives`."""
    months = list(months)
    intervals = list(intervals)
    tasks = [(s, i, m) for s in symbols for i in intervals for m in months]
    return download_archives(tasks, **kwargs)


# --- Parsing --------------------------------------------------------------------


def to_epoch_ms(column: str) -> pl.Expr:
    """Normalise integer epoch timestamps in milliseconds or microseconds to milliseconds.

    The unit is read from the order of magnitude, not from the date: between 1973 and
    5138, epoch milliseconds have 12 to 14 digits and epoch microseconds 15 to 17. Any
    other value (seconds, nanoseconds, garbage) maps to null.
    """
    ts = pl.col(column)
    return (
        pl.when(ts.is_between(10**11, 10**14, closed="left"))
        .then(ts)
        .when(ts.is_between(10**14, 10**17, closed="left"))
        .then(ts // 1_000)
        .otherwise(None)
    )


def parse_klines_csv(data: bytes, symbol: str) -> pl.DataFrame:
    """Parse one archive's CSV into a :data:`KLINE_SCHEMA` frame, rows in file order."""
    head, _, rest = data.partition(b"\n")
    if not head[:1].isdigit():  # a header row (or a blank first line): skip it
        data = rest
    first_row = data.partition(b"\n")[0].strip()
    if not first_row:
        return pl.DataFrame(schema=KLINE_SCHEMA)
    columns = first_row.count(b",") + 1
    if columns != len(_CSV_SCHEMA):
        raise ValueError(f"{symbol}: expected {len(_CSV_SCHEMA)} CSV columns, found {columns}")

    raw = pl.read_csv(io.BytesIO(data), has_header=False, schema=_CSV_SCHEMA)
    raw = raw.with_columns(open_ms=to_epoch_ms("open_time"), close_ms=to_epoch_ms("close_time"))
    invalid = raw.filter(pl.col("open_ms").is_null() | pl.col("close_ms").is_null())
    if invalid.height:
        row = invalid.row(0, named=True)
        raise ValueError(
            f"{symbol}: timestamps neither in ms nor in us: "
            f"open_time={row['open_time']}, close_time={row['close_time']}"
        )
    return raw.select(
        pl.lit(symbol, pl.String).alias("symbol"),
        pl.col("open_ms").cast(TIMESTAMP).alias("open_time"),
        "open",
        "high",
        "low",
        "close",
        "volume",
        pl.col("close_ms").cast(TIMESTAMP).alias("close_time"),
        "quote_volume",
        "trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )


def read_klines_zip(path: Path, symbol: str) -> pl.DataFrame:
    """Parse the single CSV inside a monthly archive."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected one CSV member, found {members}")
        return parse_klines_csv(archive.read(members[0]), symbol)


# --- Parquet --------------------------------------------------------------------


def build_parquet(symbol: str, interval: str, *, raw_dir: Path, parquet_dir: Path) -> QualityReport | None:
    """Rebuild a pair's Parquet file from every verified archive of a month before the vault in the cache.

    An archive of a vault month (read from its file name) is left out, with a warning that names it,
    whether it is verified or not: the vault rows do not enter a table built here. The quality report is computed on the raw concatenation (archive order, then file
    order), so duplicates and ordering problems stay visible; the stored series is
    then sorted by ``open_time``, keeping the first row of each open time. The report
    is saved beside the Parquet file as ``{SYMBOL}.quality.json``. Returns ``None``
    when the cache holds no verified archive for the pair.
    """
    archives = []
    for path in cached_archives(raw_dir, symbol, interval):
        parts = archive_name_parts(path.name)
        if parts is None or is_vault_month(parts[2]):
            log.warning(
                "%s: %s, left out of the Parquet file",
                path.name,
                "not an archive name" if parts is None else f"vault month {parts[2]}",
            )
        elif is_verified(path):
            archives.append(path)
        else:
            log.warning("%s: not verified, left out of the Parquet file", path.name)
    if not archives:
        return None

    raw = pl.concat([read_klines_zip(path, symbol) for path in archives])
    report = check_quality(raw, interval, symbol=symbol)
    bars = raw.unique(subset="open_time", keep="first", maintain_order=True).sort("open_time")

    out = parquet_path(parquet_dir, symbol, interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    bars.write_parquet(partial, compression="zstd", statistics=True)
    os.replace(partial, out)
    _atomic_write(out.with_name(f"{symbol}.quality.json"), json.dumps(report.to_dict(), indent=2).encode())
    return report


# --- CLI ------------------------------------------------------------------------


def run(
    symbols: Sequence[str],
    intervals: Sequence[str],
    start: Month,
    end: Month,
    *,
    data_dir: Path,
    client: httpx.Client | None = None,
    workers: int = 8,
    force: bool = False,
) -> tuple[list[MonthResult], dict[tuple[str, str], QualityReport | None]]:
    """Download the archives of ``start..end``, then rebuild each pair's Parquet file.

    Raises :class:`VaultMonthRefused` before any request if ``end`` is in the vault.
    """
    if is_vault_month(end):
        raise VaultMonthRefused(vault_refusal("download", month_range(max(start, vault_first_month()), end)))
    raw_dir, parquet_dir = data_dir / "raw", data_dir / "parquet"
    results = download_klines(
        symbols,
        intervals,
        month_range(start, end),
        raw_dir=raw_dir,
        client=client,
        workers=workers,
        force=force,
    )
    reports = {
        (symbol, interval): build_parquet(symbol, interval, raw_dir=raw_dir, parquet_dir=parquet_dir)
        for symbol in symbols
        for interval in intervals
    }
    return results, reports


def month_spans(months: Iterable[Month]) -> str:
    """Compact text for a set of months, e.g. ``2017-01..2017-07, 2026-10``."""
    spans: list[list[Month]] = []
    for month in sorted(set(months)):
        if spans and spans[-1][1].shift(1) == month:
            spans[-1][1] = month
        else:
            spans.append([month, month])
    return ", ".join(str(a) if a == b else f"{a}..{b}" for a, b in spans)


def configure_logging() -> None:
    """INFO logs for the CLIs, without one line per HTTP request from httpx."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olsprime.data.binance_vision",
        description="Download Binance spot klines from data.binance.vision into a verified cache and Parquet files.",
    )
    parser.add_argument("--symbols", nargs="+", required=True, type=str.upper, help="e.g. BTCUSDT ETHUSDT")
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1h", "1d"],
        choices=list(INTERVAL_MS),
        metavar="INTERVAL",
        help="kline intervals (default: 1h 1d)",
    )
    parser.add_argument("--start", type=month_arg, required=True, help="first month, YYYY-MM")
    parser.add_argument(
        "--end",
        type=month_arg,
        default=min(last_complete_month(), vault_first_month().shift(-1)),
        help="last month, YYYY-MM (default: the last complete month, at most the month before the vault)",
    )
    parser.add_argument("--data-dir", type=Path, default=data_dir(), help="data root (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent downloads (default: 8)")
    parser.add_argument("--force", action="store_true", help="download again even when the cache is verified")
    args = parser.parse_args(argv)
    configure_logging()
    if is_vault_month(args.end):
        parser.error(vault_refusal("--end", month_range(max(args.start, vault_first_month()), args.end)))

    results, reports = run(
        args.symbols,
        args.intervals,
        args.start,
        args.end,
        data_dir=args.data_dir,
        workers=args.workers,
        force=args.force,
    )
    counts = Counter(result.status for result in results)
    print("archives: " + ", ".join(f"{counts[status]} {status}" for status in Status))
    for (symbol, interval), report in reports.items():
        print(report.summary() if report else f"{symbol} {interval}: no data")
        missing = [
            r.month
            for r in results
            if (r.symbol, r.interval) == (symbol, interval) and r.status is Status.MISSING
        ]
        if missing:
            print(f"  months without an archive: {month_spans(missing)}")
    return 1 if counts[Status.FAILED] else 0


if __name__ == "__main__":
    raise SystemExit(main())
