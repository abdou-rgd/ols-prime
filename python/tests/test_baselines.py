"""The baselines: interface, purity, closed forms, the cheater, and a learning check.

Offline and synthetic. ``synthetic_frame`` builds features joined to labels with the
columns of the data contract; ``random_walk_bars`` builds bars in the stored schema for
the end-to-end runs through the real feature, label and split modules.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from olsprime.data.binance_vision import Month
from olsprime.data.klines import KLINE_SCHEMA, TIMESTAMP
from olsprime.eval.splits import walk_forward
from olsprime.features import basic_features
from olsprime.features.basic import FEATURE_COLUMNS
from olsprime.labels import LABEL_COLUMNS, forward_labels
from olsprime.models import (
    BASELINE_NAMES,
    PREDICTION_SCHEMA,
    Cheater,
    Climatology,
    LabelLeak,
    MaMomentum,
    RandomTurnover,
    Ridge,
    XGBoost,
    apply_long_flat,
    describe,
    honest_baselines,
    round_trip_cost,
)
from olsprime.models.baselines import RidgeConfig, XGBoostConfig, counter_uniform

DAY = timedelta(days=1)
START = datetime(2019, 1, 1, tzinfo=UTC)
TEST_ONLY = ("symbol", "open_time", "decision_time", "threshold", *FEATURE_COLUMNS)


# --- Synthetic frames ---------------------------------------------------------------


def synthetic_frame(
    n: int = 500,
    symbols: tuple[str, ...] = ("AAAUSDT", "BBBUSDT"),
    *,
    seed: int = 0,
    beta: dict[str, float] | None = None,
    noise: float = 0.03,
    h: int = 1,
    start: datetime = START,
    warmup_nulls: int = 5,
) -> pl.DataFrame:
    """Features joined to labels (the contract's columns and types), daily, one block per symbol.

    ``fwd_return = Σ beta_f · f + noise``; the first ``warmup_nulls`` rows of each symbol have
    null ``f_ret_90`` and ``f_ma_gap_90``, as during a real warm-up.
    """
    rng = np.random.default_rng(seed)
    beta = beta or {}
    parts = []
    for s, symbol in enumerate(symbols):
        open_time = [start + i * DAY for i in range(n)]
        feats = {c: rng.normal(0.0, 1.0, n) * 0.02 for c in FEATURE_COLUMNS}
        signal = sum(b * feats[c] for c, b in beta.items()) if beta else np.zeros(n)
        fwd = signal + rng.normal(0.0, noise, n) + 0.0005 * s
        threshold = np.full(n, max(0.5 * noise * np.sqrt(h), 0.0024))
        label = np.where(fwd < -threshold, 0, np.where(fwd > threshold, 2, 1)).astype(np.uint8)
        frame = pl.DataFrame(
            {
                "symbol": [symbol] * n,
                "open_time": pl.Series(open_time, dtype=TIMESTAMP),
            }
        ).with_columns(
            (pl.col("open_time") + DAY - timedelta(milliseconds=1)).alias("decision_time"),
            *(pl.Series(c, v, dtype=pl.Float64) for c, v in feats.items()),
            (pl.col("open_time") + DAY).alias("entry_time"),
            (pl.col("open_time") + (1 + h) * DAY).alias("exit_time"),
            (pl.col("open_time") + (1 + h) * DAY).alias("label_known_at"),
            pl.Series("fwd_return", fwd, dtype=pl.Float64),
            pl.Series("threshold", threshold, dtype=pl.Float64),
            pl.Series("label", label, dtype=pl.UInt8),
        )
        if warmup_nulls:
            row = pl.int_range(pl.len()) < warmup_nulls
            frame = frame.with_columns(pl.when(~row).then(pl.col(c)).alias(c) for c in ("f_ret_90", "f_ma_gap_90"))
        parts.append(frame)
    return pl.concat(parts)


def split(frame: pl.DataFrame, at: datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training rows whose labels are known by ``at`` minus one day; test rows from ``at``, unlabelled."""
    train = frame.filter(pl.col("label_known_at") <= at - DAY)
    test = frame.filter(pl.col("decision_time") >= at).select(TEST_ONLY)
    return train, test


def all_baselines(bars: pl.DataFrame | None = None) -> list:
    bars = random_walk_bars(400) if bars is None else bars
    return [*honest_baselines("1d"), Cheater(bars)]


def random_walk_bars(
    n: int, symbols: tuple[str, ...] = ("AAAUSDT", "BBBUSDT"), *, seed: int = 0, start: datetime = START, vol=0.03
) -> pl.DataFrame:
    """Bars in the stored schema: a continuous random walk, so ``open[t+1] = close[t]``."""
    rng = np.random.default_rng(seed)
    parts = []
    for symbol in symbols:
        log_close = np.log(100.0) + np.cumsum(rng.normal(0.0, vol, n))
        close = np.exp(log_close)
        open_ = np.concatenate([[100.0], close[:-1]])
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
        quote = rng.uniform(1e6, 2e6, n)
        open_time = [start + i * DAY for i in range(n)]
        parts.append(
            pl.DataFrame(
                {
                    "symbol": [symbol] * n,
                    "open_time": open_time,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": quote / close,
                    "close_time": [t + DAY - timedelta(milliseconds=1) for t in open_time],
                    "quote_volume": quote,
                    "trades": [100] * n,
                    "taker_buy_base_volume": quote / close / 2,
                    "taker_buy_quote_volume": quote * rng.uniform(0.4, 0.6, n),
                },
                schema=KLINE_SCHEMA,
            )
        )
    return pl.concat(parts)


def contract_frame(bars: pl.DataFrame, h: int = 1) -> pl.DataFrame:
    """Features joined to labels through the real modules."""
    labels = forward_labels(bars, "1d", h).drop("decision_time")
    return basic_features(bars, "1d").join(labels, on=["symbol", "open_time"], how="left", maintain_order="left")


def log_loss(proba: np.ndarray, label: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(label.size), label], 1e-15, 1.0))))


def as_arrays(pred: pl.DataFrame) -> dict[str, np.ndarray]:
    return {c: pred[c].fill_null(np.nan).to_numpy() for c in PREDICTION_SCHEMA if c not in ("symbol", "decision_time")}


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return synthetic_frame(500, beta={"f_ret_1": 0.6, "f_vol_7": -0.4})


@pytest.fixture(scope="module")
def fold(frame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return split(frame, START + 420 * DAY)


# --- The fixed list -----------------------------------------------------------------


def test_the_variant_list_is_fixed_and_names_are_unique():
    assert BASELINE_NAMES == (
        "flat", "buy-and-hold", "ma-momentum-10", "ma-momentum-30", "ma-momentum-90",
        "random-turnover", "climatology", "ridge", "xgboost", "cheater",
    )  # fmt: skip
    assert tuple(b.name for b in all_baselines()) == BASELINE_NAMES
    with pytest.raises(ValueError, match="fixed"):
        MaMomentum(20)


def test_configs_are_frozen_dataclasses_and_describe_serializes_them():
    for b in all_baselines():
        record = describe(b)
        assert record["name"] == b.name and isinstance(record["config"], dict)
        with pytest.raises(AttributeError):
            b.config.__setattr__("x", 1)  # frozen
    assert describe(XGBoost())["config"]["nthread"] == 1


# --- Interface ----------------------------------------------------------------------


@pytest.mark.parametrize("index", range(len(BASELINE_NAMES)))
@pytest.mark.parametrize("column", LABEL_COLUMNS)
def test_predict_refuses_every_label_column(fold, index, column):
    train, test = fold
    baseline = all_baselines()[index]
    baseline.fit(train)
    leaked = test.with_columns(pl.lit(None).alias(column))
    with pytest.raises(LabelLeak, match=column):
        baseline.predict(leaked)


@pytest.mark.parametrize("index", range(len(BASELINE_NAMES)))
def test_predict_refuses_columns_outside_the_contract(fold, index):
    train, test = fold
    baseline = all_baselines()[index]
    baseline.fit(train)
    with pytest.raises(ValueError, match="unexpected columns"):
        baseline.predict(test.with_columns(pl.lit(1.0).alias("close")))


@pytest.mark.parametrize("index", range(len(BASELINE_NAMES)))
def test_output_has_the_contract_columns_types_and_input_order(fold, index):
    train, test = fold
    baseline = all_baselines()[index]
    baseline.fit(train)
    shuffled = test.sample(fraction=1.0, shuffle=True, seed=3)
    pred = baseline.predict(shuffled)
    assert pred.schema == pl.Schema(PREDICTION_SCHEMA)
    assert pred["symbol"].equals(shuffled["symbol"]) and pred["decision_time"].equals(shuffled["decision_time"])
    # Predictions do not depend on the order of the test rows.
    back = pred.with_columns(shuffled["open_time"]).sort(["symbol", "open_time"]).drop("open_time")
    direct = baseline.predict(test.sort(["symbol", "open_time"]))
    assert back.equals(direct)


@pytest.mark.parametrize("index", range(len(BASELINE_NAMES)))
def test_positions_or_probabilities_never_both_and_always_valid(fold, index):
    train, test = fold
    baseline = all_baselines()[index]
    baseline.fit(train)
    pred = baseline.predict(test)
    probabilistic = baseline.name in ("climatology", "ridge", "xgboost")
    if probabilistic:
        assert pred["target"].null_count() == pred.height  # the decision rule sets it
        p = pred.select("p_down", "p_neutral", "p_up").drop_nulls().to_numpy()
        assert p.shape[0] > 0 and (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
        assert pred["exp_return"].null_count() == pred["p_up"].null_count()
    else:
        for c in ("p_down", "p_neutral", "p_up", "exp_return"):
            assert pred[c].null_count() == pred.height, c
        assert pred["target"].null_count() == 0
        assert set(pred["target"].unique().to_list()) <= {0.0, 1.0}


def test_null_features_give_null_forecasts_and_the_rule_keeps_them_flat(fold):
    train, test = fold
    holed = test.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("f_ret_3")).alias("f_ret_3")
    )
    missing = (holed["f_ret_3"].is_null()).to_numpy()
    for baseline in (Ridge(), XGBoost()):
        baseline.fit(train)
        pred = baseline.predict(holed)
        for c in ("p_down", "p_neutral", "p_up", "exp_return"):
            assert (pred[c].is_null().to_numpy() == missing).all(), (baseline.name, c)
        ruled = apply_long_flat(pred, 0.0)
        assert (ruled["target"].to_numpy()[missing] == 0.0).all()
    momentum = MaMomentum(90).predict(holed.with_columns(pl.lit(None, dtype=pl.Float64).alias("f_ma_gap_90")))
    assert (momentum["target"] == 0.0).all()  # null feature -> flat


@pytest.mark.parametrize("make", [lambda: RandomTurnover(0, "1d"), Climatology, Ridge, XGBoost])
def test_predict_before_fit_raises(fold, make):
    with pytest.raises(RuntimeError, match="before fit"):
        make().predict(fold[1])


# --- Determinism --------------------------------------------------------------------


@pytest.mark.parametrize("index", range(len(BASELINE_NAMES)))
def test_same_seed_and_data_give_identical_outputs_bit_for_bit(fold, index):
    train, test = fold
    outputs = []
    for shuffle_seed in (None, 11):
        baseline = all_baselines()[index]
        rows = train if shuffle_seed is None else train.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)
        baseline.fit(rows)  # the fit does not depend on the order of the training rows either
        outputs.append(as_arrays(baseline.predict(test)))
    for c in outputs[0]:
        a, b = outputs[0][c], outputs[1][c]
        assert a.tobytes() == b.tobytes(), c


def test_xgboost_is_bit_for_bit_deterministic_with_fixed_nthread(fold):
    """Repeated fits agree bit for bit at a fixed nthread; other thread counts give other trees."""
    train, test = fold
    runs = []
    for _ in range(3):
        model = XGBoost()
        model.fit(train)
        runs.append(model.predict_proba(test))
        assert model.config.nthread == 1
    assert runs[0].tobytes() == runs[1].tobytes() == runs[2].tobytes()


# --- Random turnover ----------------------------------------------------------------


def test_random_turnover_matches_momentum_30_flip_rate_and_differs_by_seed():
    frame = synthetic_frame(3000, ("AAAUSDT",), seed=4, warmup_nulls=0)
    # Persistent momentum signal: an AR(1) gap, so ma-momentum-30 flips rarely.
    rng = np.random.default_rng(1)
    gap = np.zeros(3000)
    for i in range(1, 3000):
        gap[i] = 0.9 * gap[i - 1] + rng.normal()
    frame = frame.with_columns(pl.Series("f_ma_gap_30", gap))
    train, test = frame.head(1000), frame.tail(2000).select(TEST_ONLY)
    momentum = (gap[:1000] > 0).astype(float)
    expected = float(np.mean(np.diff(momentum) != 0))
    model = RandomTurnover(7, "1d")
    model.fit(train)
    assert model.switch_probability_ == pytest.approx(expected, abs=1e-15)
    target = model.predict(test)["target"].to_numpy()
    rate = np.mean(np.diff(target) != 0)
    assert abs(rate - expected) < 4 * np.sqrt(expected * (1 - expected) / 2000)
    assert 0.35 < target.mean() < 0.65  # long half the time in the long run
    other = RandomTurnover(8, "1d")
    other.fit(train)
    assert not np.array_equal(other.predict(test)["target"].to_numpy(), target)


def test_random_turnover_flip_rate_counts_only_consecutive_bars():
    frame = synthetic_frame(10, ("AAAUSDT",), warmup_nulls=0).with_columns(
        pl.Series("f_ma_gap_30", [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    model = RandomTurnover(0, "1d")
    model.fit(frame.filter(pl.int_range(pl.len()) != 2))  # bar 2 missing: 1->3 is no transition
    # transitions 0-1 (flip), 3-4 (flip), 4-5 ... 8-9: 7 transitions, 2 flips
    assert model.switch_probability_ == pytest.approx(2 / 7)


def test_random_turnover_chain_continues_across_folds():
    """A pure function of the bar: two adjacent months predicted apart equal both predicted at once."""
    frame = synthetic_frame(400, warmup_nulls=0)
    model = RandomTurnover(3, "1d")
    model.fit(frame.head(300))
    test = frame.select(TEST_ONLY)
    whole = model.predict(test)["target"]
    first, second = test.head(250), test.tail(test.height - 250)
    apart = pl.concat([model.predict(first), model.predict(second)])["target"]
    assert whole.equals(apart)


def test_counter_uniform_is_stable_and_uniform():
    index = np.arange(200_000)
    u = counter_uniform(0, "BTCUSDT", 0, index)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 0.01
    assert counter_uniform(0, "BTCUSDT", 0, index[5:9]).tobytes() == u[5:9].tobytes()
    assert not np.array_equal(counter_uniform(0, "ETHUSDT", 0, index[:100]), u[:100])
    assert not np.array_equal(counter_uniform(0, "BTCUSDT", 1, index[:100]), u[:100])
    # Pinned values: a change of the generator changes every random-turnover result.
    np.testing.assert_array_equal(u[:3], [0.8583847078231648, 0.28976911188833554, 0.06827389525867411])


# --- Climatology --------------------------------------------------------------------


def test_climatology_is_the_training_class_frequency(fold):
    train, test = fold
    model = Climatology()
    model.fit(train)
    label = train["label"].to_numpy().astype(int)
    freq = np.bincount(label, minlength=3) / label.size
    pred = model.predict(test)
    np.testing.assert_array_equal(
        pred.select("p_down", "p_neutral", "p_up").to_numpy(), np.tile(freq, (test.height, 1))
    )
    fwd = train["fwd_return"].to_numpy()
    expected = sum(freq[k] * fwd[label == k].mean() for k in range(3))
    np.testing.assert_allclose(pred["exp_return"].to_numpy(), expected, rtol=1e-12)
    np.testing.assert_allclose(expected, fwd.mean(), rtol=1e-10)  # Σ p_k m_k is the mean


# --- Ridge --------------------------------------------------------------------------


def independent_ridge(x: np.ndarray, y: np.ndarray, lam: float):
    """Ridge with an unpenalized intercept by augmented least squares (SVD), not normal equations."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    p = z.shape[1]
    design = np.column_stack([np.ones(len(y)), z])
    augmented = np.vstack([design, np.column_stack([np.zeros(p), np.sqrt(lam) * np.eye(p)])])
    coef, *_ = np.linalg.lstsq(augmented, np.concatenate([y, np.zeros(p)]), rcond=None)
    hat = design @ np.linalg.pinv(augmented.T @ augmented) @ design.T
    residual = y - design @ coef
    sigma = np.sqrt(residual @ residual / (len(y) - np.trace(hat)))
    return mean, std, coef[0], coef[1:], sigma


def complete_rows(train: pl.DataFrame) -> pl.DataFrame:
    return train.drop_nulls(subset=list(FEATURE_COLUMNS)).sort(["decision_time", "symbol"])


@pytest.mark.parametrize("lam", [1.0, 100.0, 1e4])
def test_ridge_equals_an_independent_closed_form(fold, lam):
    train, test = fold
    model = Ridge(RidgeConfig(lambdas=(lam,)))
    model.fit(train)
    rows = complete_rows(train)
    x, y = rows.select(FEATURE_COLUMNS).to_numpy(), rows["fwd_return"].to_numpy()
    mean, std, intercept, beta, sigma = independent_ridge(x, y, lam)
    assert model.lambda_ == lam
    np.testing.assert_allclose(model.beta_, beta, rtol=1e-9, atol=1e-13)
    assert model.intercept_ == pytest.approx(intercept, rel=1e-12)
    assert model.sigma_ == pytest.approx(sigma, rel=1e-9)
    t = test.drop_nulls(subset=list(FEATURE_COLUMNS))
    mu = intercept + ((t.select(FEATURE_COLUMNS).to_numpy() - mean) / std) @ beta
    pred = model.predict(t)
    np.testing.assert_allclose(pred["exp_return"].to_numpy(), mu, rtol=1e-9, atol=1e-14)
    theta = t["threshold"].to_numpy()
    np.testing.assert_allclose(pred["p_up"].to_numpy(), 1 - norm.cdf((theta - mu) / sigma), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(pred["p_down"].to_numpy(), norm.cdf((-theta - mu) / sigma), rtol=1e-9, atol=1e-14)


def test_ridge_chooses_lambda_inside_the_fold_on_the_purged_last_fifth(fold):
    train, _ = fold
    model = Ridge()
    model.fit(train)
    rows = complete_rows(train)
    n = rows.height
    val_start = rows["decision_time"][n - round(0.2 * n)]
    h_interval = DAY  # h = 1
    inner = rows.filter((pl.col("label_known_at") <= val_start - h_interval) & (pl.col("decision_time") < val_start))
    val = rows.filter(pl.col("decision_time") >= val_start)
    # The purge is real: the last inner label is known at least h bars before validation starts.
    assert inner["label_known_at"].max() <= val_start - h_interval
    assert inner.height < rows.height - val.height  # rows dropped by the purge and embargo
    mse = {}
    for lam in (1.0, 10.0, 100.0, 1e3, 1e4):
        mean, std, b0, b, _ = independent_ridge(
            inner.select(FEATURE_COLUMNS).to_numpy(), inner["fwd_return"].to_numpy(), lam
        )
        resid = val["fwd_return"].to_numpy() - b0 - ((val.select(FEATURE_COLUMNS).to_numpy() - mean) / std) @ b
        mse[lam] = float(np.mean(resid**2))
    assert model.selection_.keys() == mse.keys()
    np.testing.assert_allclose(list(model.selection_.values()), list(mse.values()), rtol=1e-9)
    assert model.lambda_ == min(mse, key=lambda k: (mse[k], -k))


def test_ridge_gaussian_probabilities_sum_to_one_even_far_in_the_tails(fold):
    train, test = fold
    model = Ridge()
    model.fit(train)
    pred = model.predict(test)
    p = pred.select("p_down", "p_neutral", "p_up").drop_nulls().to_numpy()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-15)
    extreme = test.drop_nulls().head(3).with_columns(pl.Series("threshold", [1e-6, 0.5, 5.0]))
    q = model.predict(extreme).select("p_down", "p_neutral", "p_up").to_numpy()
    assert (q >= 0).all()
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-15)


def test_ridge_recovers_a_planted_signal_and_beats_climatology():
    beta = {"f_ret_1": 0.8, "f_vol_7": -0.5, "f_taker_7": 0.3}
    frame = synthetic_frame(1200, seed=2, beta=beta, noise=0.02)
    train, test_x = split(frame, START + 1000 * DAY)
    labels = frame.filter(pl.col("decision_time") >= START + 1000 * DAY)
    model = Ridge()
    model.fit(train)
    slope = dict(zip(model.features_, model.beta_ / model.std_, strict=True))
    for c in FEATURE_COLUMNS:
        assert slope[c] == pytest.approx(beta.get(c, 0.0), abs=0.08), c
    clim = Climatology()
    clim.fit(train)
    keep = test_x.select(pl.all_horizontal(pl.col(FEATURE_COLUMNS).is_not_null())).to_series().to_numpy()
    label = labels["label"].to_numpy().astype(int)[keep]
    p_ridge = model.predict(test_x).select("p_down", "p_neutral", "p_up").to_numpy()[keep]
    p_clim = clim.predict(test_x).select("p_down", "p_neutral", "p_up").to_numpy()[keep]
    ll_ridge, ll_clim = log_loss(p_ridge, label), log_loss(p_clim, label)
    assert ll_ridge < ll_clim - 0.05, (ll_ridge, ll_clim)


# --- XGBoost ------------------------------------------------------------------------


def test_xgboost_can_learn_a_tiny_training_set():
    """Overfit sanity with the fixed hyperparameters: near-zero training log-loss on separable data.

    ``min_child_weight = 5`` bounds the hessian of a leaf, so the reachable training log-loss
    falls like 1/n (about 0.18 at 60 rows, 0.06 at 150, 0.03 at 300): 300 rows is the tiny set.
    """
    frame = synthetic_frame(300, ("AAAUSDT",), seed=5, warmup_nulls=0)
    rng = np.random.default_rng(0)
    label = rng.integers(0, 3, frame.height)
    frame = frame.with_columns(
        pl.Series("label", label, dtype=pl.UInt8),
        pl.Series("f_ret_1", label + rng.uniform(-0.3, 0.3, frame.height)),  # separable by one feature
    )
    model = XGBoost()
    model.fit(frame)
    proba = model.predict_proba(frame.select(TEST_ONLY))
    ll = log_loss(proba, label)
    assert ll < 0.05, ll  # climatology would score about ln 3 = 1.10
    assert (proba.argmax(axis=1) == label).all()


def test_xgboost_on_noise_stays_near_climatology(fold):
    """No signal in the labels' classes beyond chance: out of sample it must not be confident."""
    frame = synthetic_frame(700, seed=9)  # no beta: pure noise
    train, test = split(frame, START + 600 * DAY)
    labels = frame.filter(pl.col("decision_time") >= START + 600 * DAY)["label"].to_numpy().astype(int)
    model, clim = XGBoost(), Climatology()
    model.fit(train)
    clim.fit(train)
    p = model.predict_proba(test)
    keep = np.isfinite(p).all(axis=1)
    p_clim = clim.predict(test).select("p_down", "p_neutral", "p_up").to_numpy()
    assert log_loss(p[keep], labels[keep]) < log_loss(p_clim[keep], labels[keep]) + 0.15


def test_xgboost_nthread_is_part_of_the_config():
    assert XGBoostConfig().params()["nthread"] == 1
    assert XGBoostConfig().params()["tree_method"] == "hist"
    assert XGBoostConfig().params()["seed"] == 0


# --- The cheater --------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_the_cheater_beats_buy_and_hold_gross_on_any_series_with_up_bars(seed):
    bars = random_walk_bars(300, seed=seed, vol=0.02 + 0.01 * seed)
    frame = contract_frame(bars).filter(pl.col("fwd_return").is_not_null())
    cheat, hold = Cheater(bars), all_baselines(bars)[1]
    test = frame.select(TEST_ONLY)
    fwd = frame["fwd_return"].to_numpy()
    gross_cheat = float(fwd @ cheat.predict(test)["target"].to_numpy())
    gross_hold = float(fwd @ hold.predict(test)["target"].to_numpy())
    assert (fwd > 0).any() and (fwd < 0).any()
    assert gross_cheat > gross_hold
    assert gross_cheat == pytest.approx(float(np.sum(np.maximum(fwd, 0.0))), rel=1e-12)  # long on every up bar


def test_the_cheater_breaks_future_invariance_and_honest_baselines_do_not():
    """What H1 must catch: perturb bars after t, and the cheater's decision at t moves."""
    bars = random_walk_bars(500, seed=1)
    t = START + 449 * DAY
    after = pl.col("open_time") > t
    flipped = bars.with_columns(
        pl.when(after).then(pl.col("close")).otherwise(pl.col("open")).alias("open"),
        pl.when(after).then(pl.col("open")).otherwise(pl.col("close")).alias("close"),
    )  # every later candle changes colour
    rows = []
    for b in (bars, flipped):
        frame = contract_frame(b)
        train = frame.filter(pl.col("label_known_at") <= t - 40 * DAY)  # fitted well before t
        test = frame.filter(pl.col("open_time") <= t).select(TEST_ONLY)
        preds = {}
        for baseline in all_baselines(b):
            baseline.fit(train)
            preds[baseline.name] = as_arrays(baseline.predict(test))
        rows.append(preds)
    for name in BASELINE_NAMES:
        same = all(rows[0][name][c].tobytes() == rows[1][name][c].tobytes() for c in rows[0][name])
        assert same == (name != "cheater"), name


# --- End to end, on the real modules ------------------------------------------------


def test_every_baseline_runs_through_the_real_features_labels_and_folds():
    bars = random_walk_bars(700, seed=3)
    frame = contract_frame(bars)
    folds = walk_forward(frame, interval="1d", h=1, start=Month(2020, 6), end=Month(2020, 8))
    assert len(folds) == 3
    cost = round_trip_cost("coinbase-advanced-entry")
    for baseline in all_baselines(bars):
        preds = []
        for f in folds:
            baseline.fit(frame[f.train_idx])
            test = frame[f.test_idx].drop(LABEL_COLUMNS)  # the bench strips them; predict would refuse
            preds.append(baseline.predict(test))
        ruled = apply_long_flat(pl.concat(preds), cost)
        assert ruled["target"].null_count() == 0, baseline.name
        assert set(ruled["target"].unique().to_list()) <= {0.0, 1.0}
