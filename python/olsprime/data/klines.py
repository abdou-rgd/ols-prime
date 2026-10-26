"""Kline (candlestick) format shared by the downloader, the quality report and the universe."""

from __future__ import annotations

import polars as pl

DAY_MS = 86_400_000

# Fixed-length Binance intervals. The calendar month ("1mo") is left out on purpose:
# its bars have no constant length.
INTERVAL_MS: dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 3_600_000,
    "2h": 2 * 3_600_000,
    "4h": 4 * 3_600_000,
    "6h": 6 * 3_600_000,
    "8h": 8 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d": DAY_MS,
    "3d": 3 * DAY_MS,
    "1w": 7 * DAY_MS,
}

# Epoch milliseconds, UTC: the time type of every stored timestamp.
TIMESTAMP = pl.Datetime("ms", "UTC")

# Schema of parsed bars and of the Parquet files.
KLINE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "open_time": TIMESTAMP,
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "close_time": TIMESTAMP,
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
    "taker_buy_base_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
}


def interval_ms(interval: str) -> int:
    """Length of one bar of ``interval`` in milliseconds."""
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        supported = ", ".join(INTERVAL_MS)
        raise ValueError(f"unsupported interval {interval!r}; expected one of {supported}") from None
