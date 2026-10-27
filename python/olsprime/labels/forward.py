"""Forward labels, each stamped with the instant it becomes known.

A decision at the close of bar t executes at the open of bar t + 1, and a position
held for h bars exits at the open of bar t + 1 + h. The label of row t therefore
depends on prices that print at ``entry_time`` and ``exit_time``, and it is
**known only at** ``label_known_at = exit_time``. A model must never see the label
columns of a row at prediction time, and the walk-forward splitter trains on a row
only once ``label_known_at`` is behind the fit (purge).

Class thresholds (``docs/DESIGN.md`` section 5: thresholds set by fees and volatility)::

    threshold_t = max(k * sigma_t * sqrt(h), floor)

- ``sigma_t`` is ``f_vol_{vol_window}`` at t: the sample std of the last ``vol_window``
  one-bar log returns, known at t.
- ``floor`` is the cost of a round trip: 0.0024 = Binance's taker fee, 0.10 % per side
  (0.20 %), plus 2 bps of slippage per side (0.04 %). Under that move a long cannot pay
  for itself, so the class is "neutral" (1). The floor is Binance's, since the history
  is Binance's; the costs of the venue we would trade on are applied in the backtest,
  not here.
- Class 0 (down) if ``fwd_return < -threshold``; 2 (up) if ``fwd_return > threshold``;
  1 (neutral) otherwise.

``sigma_t`` is unknown during the warm-up and across gaps; there the threshold, and so
the label, is null. It is never replaced by the floor.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from olsprime.data.klines import TIMESTAMP, interval_ms
from olsprime.features.basic import grid_flags, split_symbols, trailing_volatility, with_known_at

# What a model must never see at prediction time. ``threshold`` is not here: it is known at t.
LABEL_COLUMNS: tuple[str, ...] = ("entry_time", "exit_time", "label_known_at", "fwd_return", "label")

_REQUIRED = ("symbol", "open_time", "open", "close")


def _bar_at(open_ms: np.ndarray, target_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row of the bar opening exactly at each target time, and whether it exists (index 0 where not)."""
    n = len(open_ms)
    idx = np.searchsorted(open_ms, target_ms)
    found = idx < n
    idx = np.where(found, idx, 0)
    found &= open_ms[idx] == target_ms
    return np.where(found, idx, 0), found


def _labels_one(part: pl.DataFrame, interval: str, h: int, vol_window: int, k: float, floor: float) -> pl.DataFrame:
    step = interval_ms(interval)
    open_ms = part["open_time"].dt.epoch("ms").to_numpy()
    close = part["close"].to_numpy().astype(np.float64)
    sigma = trailing_volatility(close, grid_flags(open_ms, step), vol_window)
    with np.errstate(all="ignore"):
        threshold = np.maximum(k * sigma * math.sqrt(h), floor)  # NaN sigma stays NaN

    opens = part["open"].to_numpy().astype(np.float64)
    entry_idx, entry_ok = _bar_at(open_ms, open_ms + step)
    exit_idx, exit_ok = _bar_at(open_ms, open_ms + (1 + h) * step)
    # Both prices must print at exactly 1 and 1 + h intervals after t (found by time, not by row
    # offset: a hole between entry and exit does not matter to the return), and be positive.
    with np.errstate(all="ignore"):
        open_entry, open_exit = opens[entry_idx], opens[exit_idx]
        both = entry_ok & exit_ok & (open_entry > 0) & (open_exit > 0)
        fwd = np.where(both, np.log(np.where(both, open_exit, 1.0) / np.where(both, open_entry, 1.0)), np.nan)

    def stamp(name: str, ms: np.ndarray) -> pl.Series:
        times = pl.Series(name, ms, dtype=pl.Int64).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
        return pl.select(pl.when(pl.Series(both)).then(times).alias(name)).to_series()

    frame = part.select("symbol", "open_time", pl.col("known_at").alias("decision_time")).with_columns(
        stamp("entry_time", open_ms[entry_idx]),
        stamp("exit_time", open_ms[exit_idx]),
        stamp("label_known_at", open_ms[exit_idx]),
        pl.Series("fwd_return", fwd, dtype=pl.Float64, nan_to_null=True),
        pl.Series("threshold", threshold, dtype=pl.Float64, nan_to_null=True),
    ).with_columns(
        pl.when(pl.col("fwd_return").is_null() | pl.col("threshold").is_null())
        .then(None)
        .when(pl.col("fwd_return") < -pl.col("threshold"))
        .then(0)
        .when(pl.col("fwd_return") > pl.col("threshold"))
        .then(2)
        .otherwise(1)
        .cast(pl.UInt8)
        .alias("label")
    )
    return frame.select(
        "symbol",
        "open_time",
        "decision_time",
        "entry_time",
        "exit_time",
        "label_known_at",
        "fwd_return",
        "threshold",
        "label",
    )


def forward_labels(
    bars: pl.DataFrame,
    interval: str,
    h: int,
    *,
    vol_window: int = 30,
    k: float = 0.5,
    floor: float = 0.0024,
) -> pl.DataFrame:
    """Three-class forward label over h bars, one row per bar; see the module docstring.

    Columns: ``symbol``, ``open_time``, ``decision_time``, ``entry_time``, ``exit_time``,
    ``label_known_at``, ``fwd_return``, ``threshold`` (Float64), ``label`` (UInt8).
    Everything after ``decision_time`` except ``threshold`` is null when there is no bar
    opening exactly 1 interval, or exactly 1 + h intervals, after bar t. That covers the
    end of the data and the vault (the seal removed those bars) and gaps. The bars are found
    by time, not by row offset. Rows keep the order of ``bars``.
    """
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError(f"horizon h must be an integer >= 1, got {h!r}")
    if vol_window < 2:
        raise ValueError(f"vol_window must be >= 2, got {vol_window}")
    if k < 0 or floor < 0:
        raise ValueError(f"k and floor must be >= 0, got k={k}, floor={floor}")
    interval_ms(interval)
    missing = [c for c in _REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars lack columns {missing}")
    bars = with_known_at(bars, interval)
    parts = split_symbols(bars, interval)
    if not parts:
        schema: dict[str, pl.DataType] = {
            "symbol": pl.String(),
            "open_time": TIMESTAMP,
            "decision_time": TIMESTAMP,
            "entry_time": TIMESTAMP,
            "exit_time": TIMESTAMP,
            "label_known_at": TIMESTAMP,
            "fwd_return": pl.Float64(),
            "threshold": pl.Float64(),
            "label": pl.UInt8(),
        }
        return pl.DataFrame(schema=schema)
    return pl.concat([_labels_one(part, interval, h, vol_window, k, floor) for part in parts])
