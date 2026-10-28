"""The long/flat rule with hysteresis, and the venue costs it must clear."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from olsprime.data.klines import TIMESTAMP
from olsprime.models.decision import FEE_SCHEDULES, SLIPPAGE, apply_long_flat, long_flat, round_trip_cost
from olsprime.paths import REPO_ROOT

NAN = float("nan")
COST = 0.01


@pytest.mark.parametrize(
    ("exp_return", "expected"),
    [
        ([0.0, 0.02], [0, 1]),  # enter above the cost
        ([COST], [0]),  # at the cost exactly: no entry (strict)
        ([0.005, 0.009, 0.0], [0, 0, 0]),  # inside the band while flat: stay flat
        ([0.02, 0.005, 0.0, 0.009], [1, 1, 1, 1]),  # inside the band while long: hold (0 is not < 0)
        ([0.02, -1e-9], [1, 0]),  # exit below 0
        ([0.02, -0.001, 0.005], [1, 0, 0]),  # after an exit, the band does not re-enter
        ([0.02, NAN, 0.005, 0.02], [1, 0, 0, 1]),  # null -> flat, and the state is flat after it
        ([NAN, NAN], [0, 0]),
        ([0.02, np.inf], [1, 0]),  # a non-finite forecast is treated as missing
        ([], []),
    ],
)
def test_hysteresis(exp_return, expected):
    out = long_flat(np.array(exp_return, dtype=float), COST)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.array(expected, dtype=float))


def test_zero_cost_enters_on_any_positive_edge():
    np.testing.assert_array_equal(long_flat(np.array([0.0, 1e-12, 0.0, -1e-12]), 0.0), [0, 1, 1, 0])


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        long_flat(np.array([0.1]), -0.01)
    with pytest.raises(ValueError):
        long_flat(np.array([0.1]), NAN)
    with pytest.raises(ValueError):
        long_flat(np.zeros((2, 2)), 0.01)


def test_the_rule_is_causal():
    """Changing forecasts after t never changes a target at or before t."""
    rng = np.random.default_rng(0)
    e = rng.normal(0.0, 0.01, 500)
    base = long_flat(e, 0.004)
    for t in (0, 100, 250, 498):
        changed = e.copy()
        changed[t + 1 :] = rng.normal(0.0, 0.05, changed.size - t - 1)
        np.testing.assert_array_equal(long_flat(changed, 0.004)[: t + 1], base[: t + 1])


# --- Costs --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cost"),
    [
        ("zero", 0.0004),
        ("binance-spot", 0.0024),  # the label floor
        ("coinbase-advanced-entry", 0.0104),  # 1.04 %, H9
        ("kraken-pro-entry", 0.0164),  # 1.64 %, H9
        ("bitvavo-entry", 0.0054),
    ],
)
def test_round_trip_cost_is_two_takers_plus_two_slippages(name, cost):
    assert SLIPPAGE == 0.0002
    assert round_trip_cost(name) == pytest.approx(cost, abs=1e-15)


def test_unknown_schedule_raises_and_the_table_is_complete():
    assert set(FEE_SCHEDULES) == {
        "zero",
        "binance-spot",
        "coinbase-advanced-entry",
        "kraken-pro-entry",
        "bitvavo-entry",
    }
    with pytest.raises(ValueError, match="unknown fee schedule"):
        round_trip_cost("binance")
    assert round_trip_cost("kraken-pro-entry", slippage=0.0) == pytest.approx(0.016)


def rust_schedules() -> dict[str, tuple[float, float]]:
    source = (REPO_ROOT / "crates" / "backtest" / "src" / "types.rs").read_text(encoding="utf-8")
    pattern = re.compile(r'name:\s*"([^"]+)",\s*maker:\s*([0-9_.]+),\s*taker:\s*([0-9_.]+)')
    return {m[1]: (float(m[2].replace("_", "")), float(m[3].replace("_", ""))) for m in pattern.finditer(source)}


def test_fee_table_matches_the_rust_backtest():
    rust = rust_schedules()
    if not rust:
        pytest.skip("crates/backtest/src/types.rs not found or not parseable")
    for name, schedule in FEE_SCHEDULES.items():
        assert name in rust, f"{name} has no constant in crates/backtest"
        assert (schedule.maker, schedule.taker) == rust[name], name


# --- On a frame of predictions ------------------------------------------------------


def predictions(symbols: list[str], exp_return: list[float], target: list[float | None] | None = None) -> pl.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    day = timedelta(days=1)
    counters: dict[str, int] = {}
    times = []
    for s in symbols:
        counters[s] = counters.get(s, 0) + 1
        times.append(start + counters[s] * day - timedelta(milliseconds=1))
    return pl.DataFrame(
        {
            "symbol": symbols,
            "decision_time": pl.Series(times, dtype=TIMESTAMP),
            "exp_return": pl.Series(exp_return, dtype=pl.Float64, nan_to_null=True),
            "target": pl.Series(target or [None] * len(symbols), dtype=pl.Float64),
        }
    )


def test_apply_runs_the_rule_per_symbol_in_time_order_and_keeps_the_input_order():
    frame = predictions(["A", "B", "A", "B", "A", "B"], [0.02, 0.0, 0.005, 0.02, -0.01, NAN])
    shuffled = frame[[4, 1, 0, 5, 2, 3]]
    out = apply_long_flat(shuffled, COST)
    assert out.drop("target").equals(shuffled.drop("target"))
    by_row = dict(zip(zip(out["symbol"], out["decision_time"], strict=True), out["target"], strict=True))
    a = [by_row[k] for k in zip(frame["symbol"], frame["decision_time"], strict=True)]
    # A: 0.02 enter, 0.005 hold, -0.01 exit; B: 0.0 flat, 0.02 enter, null flat
    assert a == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_apply_keeps_targets_a_baseline_set_and_refuses_a_mix():
    set_ = predictions(["A", "A"], [NAN, NAN], [1.0, 0.0])
    assert apply_long_flat(set_, COST).equals(set_)
    mixed = predictions(["A", "A"], [0.02, NAN], [1.0, None])
    with pytest.raises(ValueError, match="some rows"):
        apply_long_flat(mixed, COST)
