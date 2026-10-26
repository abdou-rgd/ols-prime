"""Point-in-time universe of liquid Binance spot USDT pairs.

At 00:00 UTC on the first day of each month, the universe takes the top N USDT
pairs by quote volume over the previous 30 days. The ranking reads only daily bars
that closed before that instant, and the candidates come from the
data.binance.vision bucket, which keeps the archives of delisted pairs: there is
neither look-ahead nor survivorship bias. A pair delisted during the month stays a
member until the next rebalance, so the backtest has to exit on its last bar.

Stablecoins (with fiat- and gold-pegged tokens) and leveraged tokens (BTCUP,
ETHDOWN, BNBBULL, BEAR...) are excluded. Usage::

    uv run python -m olsprime.universe --top 30 --start 2017-10 --end 2025-12

The vault (2026-01 onwards, see ``olsprime.protocol``) is sealed: ranking a month from
2026-01 on raises ``VaultSealed`` unless both keys of the vault are in place.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Collection, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from xml.etree import ElementTree

import httpx
import polars as pl

from olsprime.data.binance_vision import (
    KLINES_PREFIX,
    Month,
    Status,
    build_parquet,
    configure_logging,
    download_archives,
    fetch,
    make_client,
    month_arg,
    month_range,
    parquet_path,
)
from olsprime.data.klines import TIMESTAMP
from olsprime.paths import data_dir
from olsprime.protocol import VAULT, VaultSealed, load_klines, vault_open

log = logging.getLogger(__name__)

S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_S3 = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

QUOTE = "USDT"

# Base assets pegged to a fiat currency or to gold. UST is listed but not USTC, the
# ticker it took after losing its peg in 2022 and trading as a free-floating coin.
STABLECOINS = frozenset(
    {
        # US dollar
        "BFUSD", "BUSD", "DAI", "FDUSD", "FRAX", "GUSD", "LUSD", "PAX", "PYUSD", "RLUSD", "SUSD",
        "TUSD", "UST", "USD1", "USDC", "USDD", "USDE", "USDP", "USDS", "USDSB", "USDSOLD", "XUSD",
        # other fiat currencies
        "AEUR", "AUD", "BKRW", "EUR", "EURC", "EURI", "GBP",
        # gold
        "PAXG", "XAUT",
    }
)

# Leveraged tokens are named <BASE><suffix>: Binance's BTCUP / BTCDOWN (2020-2024)
# and the FTX tokens ETHBULL / ETHBEAR, plus a bare BULL / BEAR for bitcoin (early 2020).
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

UNIVERSE_SCHEMA: dict[str, pl.DataType] = {
    "rebalance_time": TIMESTAMP,
    "rank": pl.UInt32(),
    "symbol": pl.String(),
    "trailing_quote_volume": pl.Float64(),
    "days": pl.UInt32(),
}


# --- Bucket listing -------------------------------------------------------------


def list_bucket(client: httpx.Client, prefix: str) -> tuple[list[str], list[str]]:
    """Common prefixes and object keys right below ``prefix``, across all pages."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        params = {"prefix": prefix, "delimiter": "/", "marker": marker}
        body = fetch(client, S3_LIST_URL, params=params)
        if body is None:
            raise RuntimeError(f"bucket listing not found: {S3_LIST_URL}")
        root = ElementTree.fromstring(body)
        page_prefixes = [e.text or "" for e in root.iterfind("s3:CommonPrefixes/s3:Prefix", _S3)]
        page_keys = [e.text or "" for e in root.iterfind("s3:Contents/s3:Key", _S3)]
        prefixes += page_prefixes
        keys += page_keys
        if root.findtext("s3:IsTruncated", "false", _S3) != "true":
            return prefixes, keys
        next_marker = root.findtext("s3:NextMarker", "", _S3) or max(page_prefixes + page_keys, default="")
        if next_marker <= marker:
            raise RuntimeError(f"bucket listing of {prefix!r} does not advance past {marker!r}")
        marker = next_marker


def list_spot_symbols(client: httpx.Client, quote: str = QUOTE) -> list[str]:
    """Every spot pair quoted in ``quote`` with monthly klines in the bucket, delisted ones included."""
    prefixes, _ = list_bucket(client, f"data/{KLINES_PREFIX}/")
    symbols = (prefix.rstrip("/").rsplit("/", 1)[-1] for prefix in prefixes)
    return sorted(s for s in symbols if s.endswith(quote) and len(s) > len(quote))


def list_archive_months(client: httpx.Client, symbol: str, interval: str) -> list[Month]:
    """Months for which the bucket holds a monthly archive of ``symbol``."""
    _, keys = list_bucket(client, f"data/{KLINES_PREFIX}/{symbol}/{interval}/")
    pattern = re.compile(rf"/{re.escape(symbol)}-{re.escape(interval)}-(\d{{4}})-(\d{{2}})\.zip$")
    return sorted({Month(int(m[1]), int(m[2])) for key in keys if (m := pattern.search(key))})


# --- Eligibility ----------------------------------------------------------------


def base_asset(symbol: str, quote: str = QUOTE) -> str:
    if not symbol.endswith(quote) or len(symbol) == len(quote):
        raise ValueError(f"{symbol!r} is not a {quote} pair")
    return symbol[: -len(quote)]


def is_leveraged_token(base: str, bases: Collection[str]) -> bool:
    """Whether ``base`` is a leveraged token, given every base asset listed.

    The suffix alone would also catch real coins such as JUP or SYRUP, so the stem
    in front of it must be a listed base asset itself.
    """
    if base in ("BULL", "BEAR"):
        return True
    return any(
        base.endswith(suffix) and base[: -len(suffix)] in bases for suffix in LEVERAGED_SUFFIXES
    )


def eligible_symbols(symbols: Iterable[str], quote: str = QUOTE) -> list[str]:
    """``symbols`` (the full bucket listing) minus stablecoins and leveraged tokens."""
    by_base = {base_asset(symbol, quote): symbol for symbol in symbols}
    return sorted(
        symbol
        for base, symbol in by_base.items()
        if base not in STABLECOINS and not is_leveraged_token(base, by_base.keys())
    )


# --- Vault ---------------------------------------------------------------------


def _check_vault(end: Month) -> None:
    """Refuse to rank rebalances of the vault period while the vault is sealed."""
    if end.start() >= VAULT.start and not vault_open():
        raise VaultSealed(
            f"cannot rank up to {end}: rebalances from {Month.of(VAULT.start)} on belong to the sealed vault"
        )


# --- Ranking --------------------------------------------------------------------


def build_universe(
    daily: pl.DataFrame,
    *,
    start: Month,
    end: Month,
    top_n: int = 30,
    lookback_days: int = 30,
    min_days: int = 25,
) -> pl.DataFrame:
    """Rank pairs by trailing quote volume at each month start from ``start`` to ``end``.

    ``daily`` holds daily bars (``symbol``, ``open_time``, ``quote_volume``) of eligible
    pairs only (see :func:`eligible_symbols`). The rebalance at instant ``d`` reads
    the bars that open in ``[d - lookback_days, d)``, all closed before ``d``. A pair
    qualifies if it has at least ``min_days`` of them (new listings season first, and
    a few days of exchange outage are tolerated) and one on the last day before ``d``
    (pairs no longer trading drop out). Qualifying pairs are ranked by summed quote
    volume, ties broken by symbol, and the first ``top_n`` are members until the next
    rebalance; early months may have fewer.

    Raises :class:`~olsprime.protocol.VaultSealed` when ``end`` reaches the vault
    (2026-01) and the vault is sealed.

    Returns one row per member, with the columns of :data:`UNIVERSE_SCHEMA`.
    """
    if not 1 <= min_days <= lookback_days:
        raise ValueError(f"min_days must be in 1..lookback_days ({lookback_days}), got {min_days}")
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    _check_vault(end)
    bars = daily.select("symbol", "open_time", "quote_volume").unique(["symbol", "open_time"])
    lookback, day = timedelta(days=lookback_days), timedelta(days=1)
    members = []
    for month in month_range(start, end):
        at = month.start()
        ranked = (
            bars.filter(pl.col("open_time").is_between(at - lookback, at, closed="left"))
            .group_by("symbol")
            .agg(
                trailing_quote_volume=pl.col("quote_volume").sum(),
                days=pl.len(),
                last_bar=pl.col("open_time").max(),
            )
            .filter((pl.col("days") >= min_days) & (pl.col("last_bar") == at - day))
            .sort(["trailing_quote_volume", "symbol"], descending=[True, False])
            .head(top_n)
            .with_row_index("rank", offset=1)
        )
        if ranked.is_empty():
            log.warning("%s: no pair qualifies; is the daily data of the previous month there?", month)
        members.append(
            ranked.select(
                pl.lit(at).cast(TIMESTAMP).alias("rebalance_time"),
                pl.col("rank").cast(pl.UInt32),  # the index type is 64-bit in some polars builds
                "symbol",
                "trailing_quote_volume",
                pl.col("days").cast(pl.UInt32),
            )
        )
    return pl.concat(members)


def members_at(universe: pl.DataFrame, when: datetime) -> list[str]:
    """Members in force at ``when`` (a timezone-aware datetime), best rank first."""
    if when.tzinfo is None:
        raise ValueError("when must be timezone-aware")
    at = Month.of(when.astimezone(UTC)).start()
    return universe.filter(pl.col("rebalance_time") == at).sort("rank")["symbol"].to_list()


# --- CLI ------------------------------------------------------------------------


def universe_path(parquet_dir: Path, top_n: int) -> Path:
    return parquet_dir / "universe" / f"spot_{QUOTE.lower()}_top{top_n}.parquet"


def download_daily(
    client: httpx.Client,
    first: Month,
    last: Month,
    *,
    raw_dir: Path,
    parquet_dir: Path,
    workers: int = 16,
) -> list[str]:
    """Fetch the daily archives of every eligible pair for ``first..last``; return the pairs."""
    symbols = list_spot_symbols(client)
    candidates = eligible_symbols(symbols)
    log.info(
        "%d %s pairs in the bucket, %d left without stablecoins and leveraged tokens",
        len(symbols),
        QUOTE,
        len(candidates),
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        listings = list(pool.map(partial(list_archive_months, client, interval="1d"), candidates))
    tasks = [
        (symbol, "1d", month)
        for symbol, months in zip(candidates, listings, strict=True)
        for month in months
        if first <= month <= last
    ]
    results = download_archives(tasks, raw_dir=raw_dir, client=client, workers=workers)
    failed = sum(r.status is Status.FAILED for r in results)
    if failed:
        log.error("%d archives failed and are left out of the ranking", failed)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(partial(build_parquet, interval="1d", raw_dir=raw_dir, parquet_dir=parquet_dir), candidates))
    return candidates


def run(
    *,
    data_dir: Path,
    start: Month,
    end: Month,
    top_n: int = 30,
    lookback_days: int = 30,
    min_days: int = 25,
    client: httpx.Client | None = None,
    workers: int = 16,
    download: bool = True,
) -> pl.DataFrame:
    """Fetch the daily archives the ranking windows need, rank, and save the universe.

    With ``download=False`` nothing goes over the network: the candidates are the
    pairs whose daily Parquet file is already on disk.
    """
    _check_vault(end)  # before any download
    raw_dir, parquet_dir = data_dir / "raw", data_dir / "parquet"
    if download:
        first = Month.of(start.start() - timedelta(days=lookback_days))
        owned = client is None
        client = client or make_client(workers)
        try:
            candidates = download_daily(
                client, first, end.shift(-1), raw_dir=raw_dir, parquet_dir=parquet_dir, workers=workers
            )
        finally:
            if owned:
                client.close()
    else:
        on_disk = [p.stem for p in (parquet_dir / "spot" / "klines" / "1d").glob("*.parquet")]
        candidates = eligible_symbols(s for s in on_disk if s.endswith(QUOTE) and s != QUOTE)

    paths = [p for s in candidates if (p := parquet_path(parquet_dir, s, "1d")).is_file()]
    if not paths:
        raise RuntimeError(f"no daily Parquet file under {parquet_dir}")
    daily = pl.concat(
        [
            load_klines(
                p.stem,
                "1d",
                data_dir=data_dir,
                columns=["symbol", "open_time", "quote_volume"],
                caller="universe.run",
            )
            for p in paths
        ]
    )
    universe = build_universe(
        daily, start=start, end=end, top_n=top_n, lookback_days=lookback_days, min_days=min_days
    )
    out = universe_path(parquet_dir, top_n)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(out.name + ".part")
    universe.write_parquet(staged, compression="zstd", statistics=True)
    os.replace(staged, out)
    log.info("universe written to %s", out)
    return universe


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olsprime.universe",
        description="Build the point-in-time universe of Binance spot USDT pairs.",
    )
    parser.add_argument("--top", type=int, default=30, help="members per month (default: 30)")
    parser.add_argument("--start", type=month_arg, required=True, help="first rebalance month, YYYY-MM")
    parser.add_argument(
        "--end",
        type=month_arg,
        default=Month(2025, 12),
        help="last rebalance month, YYYY-MM (default: 2025-12, the last month before the vault)",
    )
    parser.add_argument("--lookback-days", type=int, default=30, help="ranking window (default: 30)")
    parser.add_argument("--min-days", type=int, default=25, help="daily bars needed in the window (default: 25)")
    parser.add_argument("--data-dir", type=Path, default=data_dir(), help="data root (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=16, help="concurrent requests (default: 16)")
    parser.add_argument("--no-download", action="store_true", help="rank the daily Parquet files already on disk")
    args = parser.parse_args(argv)
    configure_logging()

    universe = run(
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        top_n=args.top,
        lookback_days=args.lookback_days,
        min_days=args.min_days,
        workers=args.workers,
        download=not args.no_download,
    )
    rebalances = universe["rebalance_time"].n_unique()
    print(f"{rebalances} rebalances, {universe['symbol'].n_unique()} distinct pairs")
    if rebalances:
        last = universe["rebalance_time"].max()
        print(f"members from {last:%Y-%m-%d}: " + " ".join(members_at(universe, last)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
