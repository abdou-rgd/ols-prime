"""The walk-forward pipeline of the bench: bars -> features and labels -> folds -> predictions -> targets -> PnL.

One function per step, so that the future-invariance test (:mod:`olsprime.bench.invariance`)
and Report n°1 (:mod:`olsprime.bench.report1`) run exactly the same code:

1. :func:`build_frame`: point-in-time features joined to forward labels, one row per bar.
2. :func:`make_folds`: monthly walk-forward folds with purge and embargo (``eval.splits``).
3. :func:`walk_forward_predict`: a fresh baseline per fold, fitted on the fold's training
   rows and asked to predict its test rows **with the label columns dropped by the bench**.
4. :func:`targets`: the position decided at each scored bar's close, from the baseline's own
   ``target`` or, for a forecaster, from the long/flat rule at the venue's round-trip cost.
5. :func:`simulate_strategy`: the Rust simulator (``olsprime_rs.simulate``). The target decided
   at bar t's close executes at bar t + 1's open; the last target never executes.

Execution model of the bench, fixed here for every strategy and venue: taker liquidity,
2 bps of slippage per side, Binance's order filters for the pair (a snapshot of
``exchangeInfo``, see :data:`SYMBOL_FILTERS`), and the fee schedule of the venue.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from olsprime.data.binance_vision import Month
from olsprime.data.klines import TIMESTAMP, interval_ms
from olsprime.eval.splits import Fold, walk_forward
from olsprime.features import FEATURE_COLUMNS, basic_features
from olsprime.labels import LABEL_COLUMNS, forward_labels
from olsprime.models.baselines import PREDICTION_SCHEMA, Baseline, Ridge, ridge_closed_form
from olsprime.models.decision import apply_long_flat, round_trip_cost

PAIRS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
# The five schedules of docs/PROTOCOL.md ("Costs"): zero is the gross reference.
VENUES: tuple[str, ...] = ("zero", "binance-spot", "bitvavo-entry", "coinbase-advanced-entry", "kraken-pro-entry")
LIQUIDITY = "taker"
SLIPPAGE_BPS = 2.0
INITIAL_CASH = 10_000.0


@dataclass(frozen=True)
class SymbolFilters:
    """Exchange order filters: minimum notional (quote) and lot step (base)."""

    min_notional: float
    step_size: float


# Snapshot of Binance spot exchangeInfo (NOTIONAL.minNotional, LOT_SIZE.stepSize), October 2026;
# the same values as backtest::SymbolFilters::BINANCE_BTCUSDT / BINANCE_ETHUSDT.
SYMBOL_FILTERS: dict[str, SymbolFilters] = {
    "BTCUSDT": SymbolFilters(min_notional=5.0, step_size=0.00001),
    "ETHUSDT": SymbolFilters(min_notional=5.0, step_size=0.0001),
}

OUTPUT_COLUMNS: tuple[str, ...] = ("p_down", "p_neutral", "p_up", "exp_return", "target")

BaselineFactory = Callable[[], Baseline]
FitHook = Callable[[Fold, Baseline, pl.DataFrame], None]


# --- 1. Features and labels ----------------------------------------------------------


def build_frame(bars: pl.DataFrame, interval: str, h: int) -> pl.DataFrame:
    """Features joined to labels on ``(symbol, open_time)``, one row per bar, in the bars' order.

    Columns: ``symbol``, ``open_time``, ``decision_time``, the ``f_*`` features, then the label
    columns (``entry_time`` ... ``label``) and ``threshold``. Both sides stamp the row with the
    same ``decision_time`` (bar t's ``known_at``); the label side's copy is checked and dropped.
    """
    features = basic_features(bars, interval)
    labels = forward_labels(bars, interval, h)
    joined = features.join(
        labels.rename({"decision_time": "_label_decision_time"}),
        on=["symbol", "open_time"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if joined.height != features.height or not joined["_label_decision_time"].equals(joined["decision_time"]):
        raise ValueError("features and labels disagree on the rows or on decision_time")
    return joined.drop("_label_decision_time")


# --- 2. Folds ----------------------------------------------------------------------------


def make_folds(frame: pl.DataFrame, interval: str, h: int, start: Month, end: Month) -> list[Fold]:
    """Monthly walk-forward folds, purge and embargo = h (the protocol's defaults)."""
    return walk_forward(frame, interval=interval, h=h, start=start, end=end)


def unpurged(frame: pl.DataFrame, folds: Sequence[Fold]) -> list[Fold]:
    """!!! A PLANTED LEAK, for the invariance test only. !!!

    The same folds, but each trains on every labelled row decided before the fit, as if the
    splitter had no purge and no embargo: a label whose exit price prints after ``fit_time``
    trains the model. :mod:`olsprime.bench.invariance` must flag a model fitted on these folds.
    """
    decision = frame["decision_time"].dt.epoch("ms").to_numpy()
    labelled = (frame["label"].is_not_null() & frame["label_known_at"].is_not_null()).to_numpy()
    out = []
    for fold in folds:
        fit_ms = int(fold.fit_time.timestamp() * 1000)
        train_idx = np.flatnonzero(labelled & (decision < fit_ms))
        out.append(
            Fold(
                fit_time=fold.fit_time,
                test_start=fold.test_start,
                test_end=fold.test_end,
                train_cutoff=fold.fit_time,
                train_idx=train_idx,
                test_idx=fold.test_idx,
                last_label_known_at=None,
            )
        )
    return out


# --- 3. Predictions ------------------------------------------------------------------------


def test_rows(frame: pl.DataFrame, idx: np.ndarray) -> pl.DataFrame:
    """The rows a baseline may see at prediction time: ``LABEL_COLUMNS`` are dropped here, by the bench."""
    return frame[idx].drop([c for c in LABEL_COLUMNS if c in frame.columns])


def walk_forward_predict(
    baseline_factory: BaselineFactory,
    frame: pl.DataFrame,
    folds: Sequence[Fold],
    *,
    on_fit: FitHook | None = None,
) -> pl.DataFrame:
    """Fit a fresh baseline per fold and predict its test rows; one row per test decision.

    Columns: ``symbol``, ``open_time``, ``decision_time``, ``fit_time`` (the fold), then
    :data:`OUTPUT_COLUMNS`. Rows come in fold order, each fold in the frame's row order.
    ``on_fit(fold, model, train)`` is called after each fit (the ridge cross-check uses it).
    """
    parts: list[pl.DataFrame] = []
    seen_rows: set[int] = set()
    for fold in folds:
        overlap = seen_rows.intersection(fold.test_idx.tolist())
        if overlap:
            raise ValueError(f"fold {fold.test_start:%Y-%m}: test rows already predicted by an earlier fold")
        seen_rows.update(fold.test_idx.tolist())
        model = baseline_factory()
        train = frame[fold.train_idx]
        model.fit(train)
        if on_fit is not None:
            on_fit(fold, model, train)
        test = test_rows(frame, fold.test_idx)
        pred = model.predict(test)
        if pred.height != test.height:
            raise ValueError(f"{model.name}: {pred.height} predictions for {test.height} test rows")
        if not (pred["decision_time"].equals(test["decision_time"]) and pred["symbol"].equals(test["symbol"])):
            raise ValueError(f"{model.name}: predictions are not in the order of the test rows")
        parts.append(
            pl.DataFrame(
                [
                    test["symbol"],
                    test["open_time"],
                    test["decision_time"],
                    pl.Series("fit_time", [fold.fit_time] * test.height, dtype=TIMESTAMP),
                    *(pred[c].cast(PREDICTION_SCHEMA[c]) for c in OUTPUT_COLUMNS),
                ]
            )
        )
    if not parts:
        raise ValueError("no fold to predict")
    return pl.concat(parts)


# --- 4. Targets ----------------------------------------------------------------------------


def targets(predictions: pl.DataFrame, venue: str) -> np.ndarray:
    """The position decided at each prediction row, in the rows' order.

    A baseline that sets ``target`` keeps it. A forecaster (``target`` null, ``exp_return``
    set) goes through the long/flat rule with hysteresis at the venue's round-trip cost,
    ``2 * (taker + 2 bps)``, run per symbol in time order across fold boundaries.
    """
    cost = round_trip_cost(venue)
    filled = apply_long_flat(predictions.select("symbol", "decision_time", "exp_return", "target"), cost)
    out = filled["target"].cast(pl.Float64).to_numpy()
    if np.isnan(out).any():
        raise ValueError("a target is still missing after the decision rule")
    if ((out < 0.0) | (out > 1.0)).any():
        raise ValueError("targets must lie in [0, 1]")
    return out


def align(bars: pl.DataFrame, predictions: pl.DataFrame, interval: str) -> pl.DataFrame:
    """The scored bars: the bars whose close is a prediction row's decision, in time order.

    One symbol, one bar per prediction, on a regular grid (the simulator's annualization
    assumes it): anything else raises.
    """
    symbols = predictions["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise ValueError(f"align one symbol at a time, got {symbols}")
    if not predictions["decision_time"].is_sorted():
        raise ValueError("predictions must be in time order")
    keyed = predictions.select("open_time", "decision_time").with_row_index("_row")
    scored = keyed.join(
        bars.filter(pl.col("symbol") == symbols[0]), on="open_time", how="left", validate="1:1", maintain_order="left"
    )
    if scored["known_at"].null_count() or not scored["known_at"].equals(scored["decision_time"]):
        raise ValueError("a prediction has no bar, or its decision_time is not the bar's known_at")
    step = np.diff(scored["open_time"].dt.epoch("ms").to_numpy())
    if (step != interval_ms(interval)).any():
        raise ValueError("the scored bars are not on a regular grid; cut the series at gaps before scoring")
    return scored.drop("_row", "decision_time")


# --- 5. Simulation -----------------------------------------------------------------------


def simulate_strategy(
    bars: pl.DataFrame,
    targets: np.ndarray,
    venue: str,
    *,
    initial_cash: float = INITIAL_CASH,
):
    """``olsprime_rs.simulate`` on the scored bars: taker, 2 bps slippage, the pair's filters.

    ``targets[t]`` is decided at bar t's close and executes at bar t + 1's open; bar 0 never
    trades and ``targets[-1]`` never executes. The equity is marked at each close, so an open
    position at the end pays no exit fee. Returns the binding's ``SimResult``.
    """
    import olsprime_rs

    symbols = bars["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise ValueError(f"simulate one symbol at a time, got {symbols}")
    filters = SYMBOL_FILTERS[symbols[0]]
    t = np.ascontiguousarray(targets, dtype=np.float64)
    if t.shape != (bars.height,):
        raise ValueError(f"{t.size} targets for {bars.height} bars")
    col = {c: np.ascontiguousarray(bars[c].cast(pl.Float64).to_numpy()) for c in ("open", "high", "low", "close", "volume")}
    return olsprime_rs.simulate(
        np.ascontiguousarray(bars["open_time"].dt.epoch("ms").to_numpy().astype(np.int64)),
        col["open"],
        col["high"],
        col["low"],
        col["close"],
        col["volume"],
        t,
        initial_cash=float(initial_cash),
        fees=venue,
        liquidity=LIQUIDITY,
        slippage_bps=SLIPPAGE_BPS,
        min_notional=filters.min_notional,
        step_size=filters.step_size,
    )


def period_returns(equity: np.ndarray) -> np.ndarray:
    """Per-bar simple returns ``e_t / e_{t-1} - 1``, as ``backtest::metrics`` computes them."""
    e = np.asarray(equity, dtype=np.float64)
    return e[1:] / e[:-1] - 1.0


# --- The ridge cross-check against the Rust core -------------------------------------------


def ridge_design(model: Ridge, train: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The standardized design ``z`` and target ``y`` the ridge was refitted on, rebuilt from ``train``.

    The same rows as ``Ridge.fit``: labelled, every feature present, in time order.
    """
    assert model.features_ is not None and model.mean_ is not None and model.std_ is not None
    rows = train.filter(pl.col("fwd_return").is_not_null()).sort(["decision_time", "symbol"], maintain_order=True)
    x = np.column_stack([rows[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in model.features_])
    y = rows["fwd_return"].to_numpy().astype(np.float64)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return (x[keep] - model.mean_) / model.std_, y[keep]


def ridge_crosscheck(model: Ridge, train: pl.DataFrame) -> dict[str, float]:
    """Max |numpy - Rust| on the intercept and coefficients of the fold's ridge, and on its fitted values.

    numpy: :func:`olsprime.models.baselines.ridge_closed_form` (what the baseline uses), checked
    first to reproduce the fitted model exactly. Rust: ``olsprime_rs.ridge_fit`` at the same λ on
    the same standardized design, intercept unpenalized.
    """
    from olsprime.bench import rust

    assert model.lambda_ is not None and model.beta_ is not None and model.intercept_ is not None
    z, y = ridge_design(model, train)
    intercept, beta = ridge_closed_form(z, y, model.lambda_)
    if not (intercept == model.intercept_ and np.array_equal(beta, model.beta_)):
        raise AssertionError("the rebuilt ridge design does not reproduce the fitted baseline")
    r_intercept, r_beta = rust.ridge_fit(z, y, model.lambda_)
    fitted = intercept + z @ beta
    r_fitted = r_intercept + z @ r_beta
    return {
        "lambda": float(model.lambda_),
        "rows": float(z.shape[0]),
        "coef_max_abs_diff": float(max(abs(r_intercept - intercept), float(np.max(np.abs(r_beta - beta))))),
        "fitted_max_abs_diff": float(np.max(np.abs(r_fitted - fitted))),
    }


def fold_window(folds: Sequence[Fold]) -> tuple[datetime, datetime]:
    """``[first test_start, last test_end)`` of the folds: the scored window."""
    return folds[0].test_start, folds[-1].test_end


__all__ = [
    "FEATURE_COLUMNS",
    "INITIAL_CASH",
    "OUTPUT_COLUMNS",
    "PAIRS",
    "SLIPPAGE_BPS",
    "SYMBOL_FILTERS",
    "VENUES",
    "align",
    "build_frame",
    "fold_window",
    "make_folds",
    "period_returns",
    "ridge_crosscheck",
    "ridge_design",
    "simulate_strategy",
    "targets",
    "test_rows",
    "unpurged",
    "walk_forward_predict",
]
