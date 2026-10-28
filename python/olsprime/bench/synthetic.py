"""Synthetic bars with the schema of ``load_klines``, for offline tests of the bench.

Nothing here reads data: the bars are generated from a seed, so the tests of the pipeline,
the invariance test and Report n°1 run on any machine in seconds.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from olsprime.data.klines import TIMESTAMP, interval_ms


def synthetic_bars(
    symbol: str = "BTCUSDT",
    *,
    start: datetime = datetime(2017, 1, 1, tzinfo=UTC),
    days: int = 560,
    interval: str = "1d",
    seed: int = 0,
    price: float = 1_000.0,
    drift: float = 0.0005,
    vol: float = 0.03,
) -> pl.DataFrame:
    """``days`` consecutive bars of a log-normal walk, valid OHLC, with ``known_at`` as ``load_klines`` sets it."""
    step = interval_ms(interval)
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, size=days)
    close = price * np.exp(np.cumsum(r))
    open_ = np.concatenate([[price], close[:-1]]) * np.exp(rng.normal(0.0, vol / 10, size=days))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, vol / 2, size=days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, vol / 2, size=days)))
    volume = np.exp(rng.normal(5.0, 0.5, size=days))
    quote = volume * close
    share = rng.uniform(0.3, 0.7, size=days)
    start_ms = int(start.timestamp() * 1000)
    open_ms = start_ms + step * np.arange(days, dtype=np.int64)
    return pl.DataFrame(
        {
            "symbol": [symbol] * days,
            "open_time": pl.Series(open_ms, dtype=pl.Int64).cast(TIMESTAMP),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": pl.Series(open_ms + step - 1, dtype=pl.Int64).cast(TIMESTAMP),
            "quote_volume": quote,
            "trades": pl.Series(rng.integers(100, 1000, size=days), dtype=pl.Int64),
            "taker_buy_base_volume": volume * share,
            "taker_buy_quote_volume": quote * share,
            "known_at": pl.Series(open_ms + step - 1, dtype=pl.Int64).cast(TIMESTAMP),
        }
    )
