"""Point-in-time features of one pair's bars.

The input is what :func:`olsprime.protocol.load_klines` returns: sealed bars with a
``known_at`` column, the instant a bar's values are final. A row of the output is
the feature vector a decision taken at the close of bar t may use, stamped
``decision_time = known_at[t]``. Every feature at t is a function of bars <= t only;
a test perturbs every later bar and checks that nothing at or before t moves.

Rules shared by every feature:

- **Grid.** Bars are on the grid when each opens exactly one interval after the
  previous one. A window is valid only when every gap inside it is exactly one
  interval; otherwise the value is null. Nothing is ever filled or interpolated.
- **Warm-up.** A value is null until its window is full.
- **Zero or negative inputs.** A price or volume that would send a logarithm or a
  division out of range (``ln 0``, ``x / 0``) gives a null, never ``inf`` or ``NaN``.
- **No scaling.** Standardizing is a model's job, inside the fold.

All features are Float64 columns named ``f_*``. Definitions (``c`` close, ``h`` high,
``l`` low, ``q`` quote volume, ``b`` taker buy quote volume):

==================  ==========================================================
``f_ret_{k}``       ``ln(c_t / c_{t-k})``, k in 1, 3, 7, 14, 30, 90
``f_vol_{k}``       sample std (ddof 1) of the last k one-bar log returns, k in 7, 30
``f_ma_gap_{k}``    ``ln(c_t / mean(c_{t-k+1..t}))``, k in 10, 30, 90
``f_range_1``       ``ln(h_t / l_t)``
``f_range_7``       mean of ``f_range_1`` over the last 7 bars
``f_qvol_z_30``     ``(ln q_t - m) / s``, m and s the mean and sample std (ddof 1)
                    of ``ln q`` over bars t-30 .. t-1
``f_taker_7``       mean over the last 7 bars of ``b / q``
==================  ==========================================================
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from olsprime.data.klines import TIMESTAMP, interval_ms
from olsprime.protocol import known_at_expr

RET_WINDOWS = (1, 3, 7, 14, 30, 90)
VOL_WINDOWS = (7, 30)
MA_WINDOWS = (10, 30, 90)
QVOL_WINDOW = 30
TAKER_WINDOW = 7
RANGE_WINDOW = 7

FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"f_ret_{k}" for k in RET_WINDOWS),
    *(f"f_vol_{k}" for k in VOL_WINDOWS),
    *(f"f_ma_gap_{k}" for k in MA_WINDOWS),
    "f_range_1",
    f"f_range_{RANGE_WINDOW}",
    f"f_qvol_z_{QVOL_WINDOW}",
    f"f_taker_{TAKER_WINDOW}",
)

_REQUIRED = ("symbol", "open_time", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")


def _window_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """Row t holds ``x[t-k+1 .. t]``; NaN before the series starts."""
    padded = np.concatenate([np.full(k - 1, np.nan), x])
    return sliding_window_view(padded, k)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """``x[t-k]`` at position t, NaN before the series starts."""
    out = np.full(x.shape, np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def grid_flags(open_ms: np.ndarray, step_ms: int) -> np.ndarray:
    """``flags[t]``: bar t opens exactly one interval after bar t - 1 (False for t = 0)."""
    flags = np.zeros(len(open_ms), dtype=bool)
    if len(open_ms) > 1:
        flags[1:] = np.diff(open_ms) == step_ms
    return flags


def _all_gaps_ok(flags: np.ndarray, m: int) -> np.ndarray:
    """``ok[t]``: the last m gaps (bars t-m .. t) are all exactly one interval. ``m = 0`` is always true."""
    n = len(flags)
    if m == 0:
        return np.ones(n, dtype=bool)
    counts = np.cumsum(flags.astype(np.int64))
    before = np.zeros(n, dtype=np.int64)
    if m < n:
        before[m:] = counts[: n - m]
    ok = counts - before == m
    ok[: min(m, n)] = False
    return ok


def _log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``ln(num / den)``, NaN where either is not strictly positive or finite."""
    with np.errstate(all="ignore"):
        valid = (numerator > 0) & (denominator > 0)
        out = np.where(valid, np.log(np.where(valid, numerator, 1.0) / np.where(valid, denominator, 1.0)), np.nan)
    return out


def trailing_volatility(close: np.ndarray, flags: np.ndarray, window: int) -> np.ndarray:
    """Sample std (ddof 1) of the last ``window`` one-bar log returns, NaN where the window is not valid.

    Returns ``ln(c_s / c_{s-1})`` for s in (t - window, t]; valid when those ``window``
    gaps are all on the grid. This is ``f_vol_{window}`` and the sigma-hat of the label
    threshold: both call it, so they cannot drift apart.
    """
    if window < 2:
        raise ValueError(f"volatility window must be >= 2, got {window}")
    one_bar = _log_ratio(close, _shift(close, 1))
    one_bar[~flags] = np.nan
    with np.errstate(all="ignore"):
        out = np.std(_window_matrix(one_bar, window), axis=1, ddof=1)
    out[~_all_gaps_ok(flags, window)] = np.nan
    return out


def _clean(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.where(ok & np.isfinite(values), values, np.nan)
    return out


def _check(bars: pl.DataFrame, interval: str) -> None:
    interval_ms(interval)
    missing = [c for c in _REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(f"bars lack columns {missing}")
    if bars.schema["open_time"] != TIMESTAMP:
        raise TypeError(f"open_time is {bars.schema['open_time']}, expected {TIMESTAMP}")


def _check_increasing(symbol: str, open_ms: np.ndarray) -> None:
    if len(open_ms) > 1 and not np.all(np.diff(open_ms) > 0):
        raise ValueError(
            f"{symbol}: open_time is not strictly increasing (duplicate or unsorted bars); "
            "clean the bars first, features never reorder or drop rows silently"
        )


def with_known_at(bars: pl.DataFrame, interval: str) -> pl.DataFrame:
    """``bars`` with its ``known_at`` column, computed the way ``load_klines`` does when missing."""
    if "known_at" in bars.columns:
        return bars
    if "close_time" not in bars.columns:
        raise ValueError("bars need 'known_at' (from load_klines) or 'close_time' to compute it")
    return bars.with_columns(known_at_expr(interval))


def split_symbols(bars: pl.DataFrame, interval: str) -> list[pl.DataFrame]:
    """One validated, strictly increasing frame per symbol, in order of first appearance."""
    _check(bars, interval)
    parts = bars.partition_by("symbol", maintain_order=True) if bars.height else []
    for part in parts:
        _check_increasing(part["symbol"][0], part["open_time"].dt.epoch("ms").to_numpy())
    return parts


def _features_one(part: pl.DataFrame, interval: str) -> pl.DataFrame:
    step = interval_ms(interval)
    open_ms = part["open_time"].dt.epoch("ms").to_numpy()
    close, high, low = (part[c].to_numpy().astype(np.float64) for c in ("close", "high", "low"))
    qvol = part["quote_volume"].to_numpy().astype(np.float64)
    taker = part["taker_buy_quote_volume"].to_numpy().astype(np.float64)
    flags = grid_flags(open_ms, step)
    cols: dict[str, np.ndarray] = {}

    for k in RET_WINDOWS:
        cols[f"f_ret_{k}"] = _clean(_log_ratio(close, _shift(close, k)), _all_gaps_ok(flags, k))

    for k in VOL_WINDOWS:
        cols[f"f_vol_{k}"] = _clean(trailing_volatility(close, flags, k), np.ones(len(close), dtype=bool))

    for k in MA_WINDOWS:
        with np.errstate(all="ignore"):
            mean_close = np.mean(_window_matrix(close, k), axis=1)
        cols[f"f_ma_gap_{k}"] = _clean(_log_ratio(close, mean_close), _all_gaps_ok(flags, k - 1))

    range_1 = _log_ratio(high, low)
    cols["f_range_1"] = _clean(range_1, np.ones(len(close), dtype=bool))
    with np.errstate(all="ignore"):
        range_mean = np.mean(_window_matrix(range_1, RANGE_WINDOW), axis=1)
    cols[f"f_range_{RANGE_WINDOW}"] = _clean(range_mean, _all_gaps_ok(flags, RANGE_WINDOW - 1))

    with np.errstate(all="ignore"):
        log_q = np.where(qvol > 0, np.log(np.where(qvol > 0, qvol, 1.0)), np.nan)
        windows = _window_matrix(log_q, QVOL_WINDOW)
        prev_mean = _shift(np.mean(windows, axis=1), 1)
        prev_std = _shift(np.std(windows, axis=1, ddof=1), 1)
        z = np.where(prev_std > 0, (log_q - prev_mean) / np.where(prev_std > 0, prev_std, 1.0), np.nan)
    cols[f"f_qvol_z_{QVOL_WINDOW}"] = _clean(z, _all_gaps_ok(flags, QVOL_WINDOW))

    with np.errstate(all="ignore"):
        share = np.where(qvol > 0, taker / np.where(qvol > 0, qvol, 1.0), np.nan)
        share_mean = np.mean(_window_matrix(share, TAKER_WINDOW), axis=1)
    cols[f"f_taker_{TAKER_WINDOW}"] = _clean(share_mean, _all_gaps_ok(flags, TAKER_WINDOW - 1))

    out = part.select("symbol", "open_time", pl.col("known_at").alias("decision_time"))
    return out.with_columns(
        [pl.Series(name, cols[name], dtype=pl.Float64, nan_to_null=True) for name in FEATURE_COLUMNS]
    )


def basic_features(bars: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Point-in-time features, one row per bar; see the module docstring for every definition.

    Columns: ``symbol``, ``open_time``, ``decision_time`` (bar t's ``known_at``), then
    :data:`FEATURE_COLUMNS`, all Float64. Rows keep the order of ``bars``. Bars must be
    strictly increasing in ``open_time`` per symbol, else ``ValueError``.
    """
    bars = with_known_at(bars, interval)
    parts = split_symbols(bars, interval)
    if not parts:
        schema = {"symbol": pl.String, "open_time": TIMESTAMP, "decision_time": TIMESTAMP}
        schema.update({name: pl.Float64 for name in FEATURE_COLUMNS})
        return pl.DataFrame(schema=schema)
    return pl.concat([_features_one(part, interval) for part in parts])
