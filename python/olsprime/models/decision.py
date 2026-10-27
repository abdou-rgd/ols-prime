"""The long/flat decision rule of ``docs/DESIGN.md`` section 5, and the venue costs it clears.

Go long only when the expected edge clears the venue's round-trip cost, with hysteresis:

- flat -> long when ``exp_return > cost_round_trip``;
- long -> flat when ``exp_return < 0``;
- otherwise the state is kept;
- a missing forecast (NaN or null) means flat, and the state becomes flat.

The cost of a round trip is ``2 * (taker + slippage)``: every order is a taker order at
the open, and slippage is 2 bps per side (``docs/PROTOCOL.md``). The fee schedules mirror
the constants of ``crates/backtest/src/types.rs``; ``tests/test_decision.py`` checks them
against that file.

The rule is sequential: apply it to one symbol's forecasts in time order, across fold
boundaries (the state at t uses only forecasts made at or before t, each by a model fitted
before t's month, so carrying it from one fold into the next leaks nothing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

SLIPPAGE = 0.0002  # per side, as a fraction of the open: 2 bps


@dataclass(frozen=True)
class FeeSchedule:
    """A venue's spot fees as fractions of notional (0.001 = 0.10 %)."""

    name: str
    maker: float
    taker: float


# October 2026, re-check before use. Binance is history only (closed to French residents).
FEE_SCHEDULES: dict[str, FeeSchedule] = {
    s.name: s
    for s in (
        FeeSchedule("zero", 0.0, 0.0),
        FeeSchedule("binance-spot", 0.001, 0.001),
        FeeSchedule("coinbase-advanced-entry", 0.0025, 0.005),
        FeeSchedule("kraken-pro-entry", 0.004, 0.008),
        FeeSchedule("bitvavo-entry", 0.0015, 0.0025),
    )
}


def round_trip_cost(fee_schedule_name: str, *, slippage: float = SLIPPAGE) -> float:
    """``2 * (taker + slippage)`` for the named schedule: what a long must earn to break even."""
    try:
        schedule = FEE_SCHEDULES[fee_schedule_name]
    except KeyError:
        known = ", ".join(FEE_SCHEDULES)
        raise ValueError(f"unknown fee schedule {fee_schedule_name!r}; expected one of {known}") from None
    if not 0.0 <= slippage < 1.0:
        raise ValueError(f"slippage must be in [0, 1), got {slippage}")
    return 2.0 * (schedule.taker + slippage)


def long_flat(exp_return: np.ndarray, cost_round_trip: float) -> np.ndarray:
    """Targets in {0.0, 1.0} for one symbol's forecasts in time order; see the module docstring."""
    if not (np.isfinite(cost_round_trip) and cost_round_trip >= 0.0):
        raise ValueError(f"cost_round_trip must be finite and >= 0, got {cost_round_trip}")
    e = np.asarray(exp_return, dtype=np.float64)
    if e.ndim != 1:
        raise ValueError(f"exp_return must be one-dimensional, got shape {e.shape}")
    target = np.zeros(e.shape, dtype=np.float64)
    long = False
    for i, x in enumerate(e):
        if not np.isfinite(x):  # NaN (null) and, defensively, +-inf: flat
            long = False
        elif long:
            long = not x < 0.0
        else:
            long = x > cost_round_trip
        target[i] = 1.0 if long else 0.0
    return target


def apply_long_flat(predictions: pl.DataFrame, cost_round_trip: float) -> pl.DataFrame:
    """Fill the null ``target`` of each symbol from its ``exp_return`` by :func:`long_flat`.

    ``predictions`` holds the output of one baseline over every fold (``symbol``,
    ``decision_time``, ``exp_return``, ``target``). Within a symbol the rule runs in
    ``decision_time`` order; a ``target`` already set is kept, and must then be set on every
    row of that symbol. Rows come back in their input order.
    """
    for col in ("symbol", "decision_time", "exp_return", "target"):
        if col not in predictions.columns:
            raise ValueError(f"predictions lack column {col!r}")
    frame = predictions.with_row_index("_row")
    parts = []
    for part in frame.sort("decision_time").partition_by("symbol", maintain_order=True):
        set_count = part["target"].is_not_null().sum()
        if set_count == part.height:
            parts.append(part)
            continue
        if set_count:
            raise ValueError(f"{part['symbol'][0]}: target is set on some rows but not others")
        e = part["exp_return"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        parts.append(part.with_columns(pl.Series("target", long_flat(e, cost_round_trip), dtype=pl.Float64)))
    if not parts:
        return predictions
    return pl.concat(parts).sort("_row").drop("_row")
