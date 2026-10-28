"""The bench pipeline (olsprime.bench.pipeline): what a model sees, when a target executes, which bars are scored."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from olsprime.bench import pipeline as pp
from olsprime.bench.synthetic import synthetic_bars
from olsprime.data.binance_vision import Month
from olsprime.labels import LABEL_COLUMNS
from olsprime.models.baselines import Climatology, MaMomentum, Ridge, buy_and_hold, flat
from olsprime.models.decision import round_trip_cost

INTERVAL, H = "1d", 1
START, END = Month(2018, 3), Month(2018, 6)


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return synthetic_bars(days=560, seed=3)


@pytest.fixture(scope="module")
def frame(bars: pl.DataFrame) -> pl.DataFrame:
    return pp.build_frame(bars, INTERVAL, H)


@pytest.fixture(scope="module")
def folds(frame: pl.DataFrame):
    return pp.make_folds(frame, INTERVAL, H, START, END)


class Spy:
    """A baseline that records what fit and predict receive."""

    name = "spy"
    config = None

    def __init__(self, log: list) -> None:
        self.log = log

    def fit(self, train: pl.DataFrame) -> None:
        self.log.append(("fit", tuple(train.columns), train["label_known_at"].max()))

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        self.log.append(("predict", tuple(test.columns), test.height))
        return _flat_prediction(test)


def _flat_prediction(test: pl.DataFrame) -> pl.DataFrame:
    n = test.height
    return pl.DataFrame(
        {
            "symbol": test["symbol"],
            "decision_time": test["decision_time"],
            "p_down": [None] * n,
            "p_neutral": [None] * n,
            "p_up": [None] * n,
            "exp_return": [None] * n,
            "target": [0.0] * n,
        },
        schema_overrides={k: pl.Float64 for k in ("p_down", "p_neutral", "p_up", "exp_return", "target")},
    )


def test_frame_joins_features_and_labels_row_for_row(bars, frame):
    assert frame.height == bars.height
    assert frame["open_time"].equals(bars["open_time"])
    assert frame["decision_time"].equals(bars["known_at"])
    for c in (*LABEL_COLUMNS, "threshold", "f_ret_1", "f_ma_gap_30"):
        assert c in frame.columns


def test_predict_never_receives_a_label_column(frame, folds):
    log: list = []
    pp.walk_forward_predict(lambda: Spy(log), frame, folds)
    predicts = [entry for entry in log if entry[0] == "predict"]
    fits = [entry for entry in log if entry[0] == "fit"]
    assert len(predicts) == len(folds) == len(fits)
    for _, columns, _ in predicts:
        assert not set(columns) & set(LABEL_COLUMNS)
        assert "threshold" in columns  # known at t, allowed
    for _, columns, _ in fits:
        assert set(LABEL_COLUMNS) <= set(columns)  # fit gets the labels it may use


def test_training_labels_are_known_before_each_fit(frame, folds):
    log: list = []
    pp.walk_forward_predict(lambda: Spy(log), frame, folds)
    fits = [entry for entry in log if entry[0] == "fit"]
    for (_, _, last_known), fold in zip(fits, folds, strict=True):
        assert last_known <= fold.train_cutoff < fold.fit_time


def test_one_prediction_per_test_decision_in_order(frame, folds):
    preds = pp.walk_forward_predict(lambda: MaMomentum(30), frame, folds)
    idx = np.concatenate([f.test_idx for f in folds])
    assert preds.height == idx.size
    assert preds["decision_time"].equals(frame[idx]["decision_time"])
    assert preds["decision_time"].is_sorted()


def test_the_scored_window_is_the_same_for_every_strategy(bars, frame, folds):
    makers = [flat, buy_and_hold, lambda: MaMomentum(10), Climatology, Ridge]
    windows = [pp.walk_forward_predict(m, frame, folds)["decision_time"] for m in makers]
    for w in windows[1:]:
        assert w.equals(windows[0])
    scored = pp.align(bars, pp.walk_forward_predict(flat, frame, folds), INTERVAL)
    assert scored["open_time"][0] == START.start()
    assert scored["open_time"][-1] < END.shift(1).start()
    assert scored.height == windows[0].len()


def test_align_refuses_a_gap(bars, frame, folds):
    preds = pp.walk_forward_predict(flat, frame, folds)
    holed = preds.filter(pl.col("open_time") != preds["open_time"][10])
    with pytest.raises(ValueError, match="regular grid"):
        pp.align(bars, holed, INTERVAL)


def test_forecaster_targets_follow_the_rule_at_the_venue_cost(frame, folds):
    preds = pp.walk_forward_predict(Ridge, frame, folds)
    assert preds["target"].null_count() == preds.height
    for venue in pp.VENUES:
        t = pp.targets(preds, venue)
        assert set(np.unique(t)) <= {0.0, 1.0}
        e = preds["exp_return"].to_numpy()
        entered = (t[1:] == 1.0) & (t[:-1] == 0.0)
        assert np.all(e[1:][entered] > round_trip_cost(venue))  # flat -> long only above the round trip


def test_target_at_bar_t_executes_at_bar_t_plus_1_open(bars, frame, folds):
    preds = pp.walk_forward_predict(flat, frame, folds)
    scored = pp.align(bars, preds, INTERVAL)
    target = np.zeros(scored.height)
    target[5:9] = 1.0  # decided at the closes of bars 5..8
    res = pp.simulate_strategy(scored, target, "binance-spot")
    fills = res.fills
    # Long decided at the closes of bars 5..8, flat at bar 9's close: buy at bar 6's open, sell at bar 10's.
    assert list(fills.bar) == [6, 10]
    opens = scored["open"].to_numpy()
    slip = pp.SLIPPAGE_BPS / 1e4
    assert fills.price[0] == pytest.approx(opens[6] * (1 + slip), rel=1e-15)
    assert fills.price[1] == pytest.approx(opens[10] * (1 - slip), rel=1e-15)
    assert np.asarray(res.equity)[0] == pp.INITIAL_CASH  # bar 0 never trades


def test_last_target_never_executes(bars, frame, folds):
    scored = pp.align(bars, pp.walk_forward_predict(flat, frame, folds), INTERVAL)
    target = np.zeros(scored.height)
    target[-1] = 1.0
    res = pp.simulate_strategy(scored, target, "kraken-pro-entry")
    assert len(res.fills) == 0
    assert res.fees_paid == 0.0


def test_lot_step_and_min_notional_are_the_pairs(bars, frame, folds):
    scored = pp.align(bars, pp.walk_forward_predict(flat, frame, folds), INTERVAL)
    res = pp.simulate_strategy(scored, np.ones(scored.height), "zero")
    qty = np.asarray(res.fills.qty)[0]
    step = pp.SYMBOL_FILTERS["BTCUSDT"].step_size
    assert abs(qty / step - round(qty / step)) < 1e-6


def test_ridge_equals_the_rust_core_on_every_fold(frame, folds):
    checks = []
    pp.walk_forward_predict(Ridge, frame, folds, on_fit=lambda f, m, t: checks.append(pp.ridge_crosscheck(m, t)))
    assert len(checks) == len(folds)
    assert max(c["coef_max_abs_diff"] for c in checks) < 1e-9
    assert max(c["fitted_max_abs_diff"] for c in checks) < 1e-9


def test_unpurged_folds_train_on_labels_known_after_the_fit(frame, folds):
    leaky = pp.unpurged(frame, folds)
    known = frame["label_known_at"]
    for honest, bad in zip(folds, leaky, strict=True):
        assert known[honest.train_idx].max() <= honest.train_cutoff
        assert known[bad.train_idx].max() > honest.fit_time  # the planted leak
        assert np.array_equal(honest.test_idx, bad.test_idx)


def test_panel_spans_are_contiguous_membership_runs():
    from datetime import UTC, datetime

    from olsprime.bench.panel import Span, spans

    def t(y, m):
        return datetime(y, m, 1, tzinfo=UTC)

    universe = pl.DataFrame(
        {
            "rebalance_time": [t(2018, 12), t(2019, 1), t(2019, 2), t(2019, 4), t(2019, 2), t(2025, 1)],
            "symbol": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB"],
        }
    ).with_columns(pl.col("rebalance_time").cast(pl.Datetime("ms", "UTC")))
    got = spans(universe, Month(2019, 1), Month(2024, 12))
    assert got == [
        Span("AAA", t(2019, 1), t(2019, 3)),  # 2018-12 is before the window, 2019-03 breaks the run
        Span("AAA", t(2019, 4), t(2019, 5)),
        Span("BBB", t(2019, 2), t(2019, 3)),  # 2025-01 is after the window
    ]
