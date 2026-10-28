"""The baselines of the bench: the bar every later model must clear (``docs/PROTOCOL.md``).

Each baseline is one variant of the deflated Sharpe ratio, and the list is fixed
(:data:`BASELINE_NAMES`): adding one adds a variant. Every baseline has

- ``name``: unique, it names the variant;
- ``config``: a frozen dataclass, saved with every run;
- ``fit(train)``: the training rows of one fold, features joined to labels, every label column;
- ``predict(test)``: the test rows **without** the label columns. It sees ``symbol``,
  ``open_time``, ``decision_time``, ``threshold`` and the ``f_*`` features, and refuses any
  other column (a label column above all). It returns one row per test row, in input order,
  with the columns of :data:`PREDICTION_SCHEMA`:

  - ``p_down``, ``p_neutral``, ``p_up``: null for a non-probabilistic baseline;
  - ``exp_return``: the expected ``fwd_return``, null if not modelled;
  - ``target``: the position in [0, 1], null when the decision rule sets it
    (:func:`olsprime.models.decision.apply_long_flat`).

Baselines are pure: none reads data or files, and randomness comes only from a seed in the
config. The one exception is :class:`Cheater`, which leaks **on purpose** so that the
bench's leakage test (H1) can catch it.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from olsprime.data.klines import TIMESTAMP, interval_ms
from olsprime.labels import LABEL_COLUMNS

# The fixed variant list. Changing it changes the variant count of every reported Sharpe.
MA_WINDOWS = (10, 30, 90)
BASELINE_NAMES: tuple[str, ...] = (
    "flat",
    "buy-and-hold",
    *(f"ma-momentum-{w}" for w in MA_WINDOWS),
    "random-turnover",
    "climatology",
    "ridge",
    "xgboost",
    "cheater",
)

KEY_COLUMNS = ("symbol", "open_time", "decision_time")
FEATURE_PREFIX = "f_"
# What predict may see besides the f_* features. ``threshold`` is known at t.
TEST_COLUMNS = (*KEY_COLUMNS, "threshold")

PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "decision_time": TIMESTAMP,
    "p_down": pl.Float64(),
    "p_neutral": pl.Float64(),
    "p_up": pl.Float64(),
    "exp_return": pl.Float64(),
    "target": pl.Float64(),
}
PROBABILITY_COLUMNS = ("p_down", "p_neutral", "p_up")
N_CLASSES = 3  # 0 down, 1 neutral, 2 up


class LabelLeak(ValueError):
    """``predict`` was handed a label column."""


@runtime_checkable
class Baseline(Protocol):
    name: str
    config: Any

    def fit(self, train: pl.DataFrame) -> None: ...

    def predict(self, test: pl.DataFrame) -> pl.DataFrame: ...


def describe(baseline: Baseline) -> dict[str, Any]:
    """``name`` and ``config`` as plain values, for the run record."""
    return {"name": baseline.name, "config": asdict(baseline.config)}


# --- Shared checks and helpers ------------------------------------------------------


def check_test_frame(test: pl.DataFrame) -> list[str]:
    """Refuse label columns and anything else predict must not see; return the feature names."""
    leaked = [c for c in LABEL_COLUMNS if c in test.columns]
    if leaked:
        raise LabelLeak(f"predict refuses label columns {leaked}: strip LABEL_COLUMNS before predicting")
    unknown = [c for c in test.columns if c not in TEST_COLUMNS and not c.startswith(FEATURE_PREFIX)]
    if unknown:
        raise ValueError(f"predict sees only {TEST_COLUMNS} and f_* features; unexpected columns {unknown}")
    missing = [c for c in KEY_COLUMNS if c not in test.columns]
    if missing:
        raise ValueError(f"test rows lack columns {missing}")
    return [c for c in test.columns if c.startswith(FEATURE_PREFIX)]


def _check_train_frame(train: pl.DataFrame, needed: tuple[str, ...] = ()) -> None:
    missing = [c for c in (*KEY_COLUMNS, *needed) if c not in train.columns]
    if missing:
        raise ValueError(f"training rows lack columns {missing}")


def _chronological(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows in time order, symbols breaking ties: fit does not depend on the input order."""
    return frame.sort(["decision_time", "symbol"], maintain_order=True)


def _matrix(frame: pl.DataFrame, names: list[str]) -> np.ndarray:
    """The feature columns as float64, nulls as NaN."""
    if not names:
        return np.empty((frame.height, 0))
    return np.column_stack([frame[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in names])


def _complete(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=1)


def _output(
    test: pl.DataFrame,
    *,
    proba: np.ndarray | None = None,
    exp_return: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> pl.DataFrame:
    """The prediction frame; NaN becomes null."""
    n = test.height
    nan = np.full(n, np.nan)
    proba = np.column_stack([nan, nan, nan]) if proba is None else proba
    columns = {
        "p_down": proba[:, 0],
        "p_neutral": proba[:, 1],
        "p_up": proba[:, 2],
        "exp_return": nan if exp_return is None else exp_return,
        "target": nan if target is None else target,
    }
    return pl.DataFrame(
        [
            test["symbol"].cast(pl.String),
            test["decision_time"].cast(TIMESTAMP),
            *(
                pl.Series(k, np.asarray(v, dtype=np.float64), dtype=pl.Float64, nan_to_null=True)
                for k, v in columns.items()
            ),
        ]
    )


def _require_fitted(baseline: Any, attribute: str) -> None:
    if getattr(baseline, attribute, None) is None:
        raise RuntimeError(f"{baseline.name}: predict before fit")


def _class_means(fwd_return: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Mean ``fwd_return`` of each class in training; 0 for an empty class (its weight is 0 too)."""
    means = np.zeros(N_CLASSES)
    for k in range(N_CLASSES):
        mask = label == k
        if mask.any():
            means[k] = float(np.mean(fwd_return[mask]))
    return means


# --- Positions without a forecast ---------------------------------------------------


@dataclass(frozen=True)
class ConstantConfig:
    target: float


class Constant:
    """A fixed position: ``flat`` (0) or ``buy-and-hold`` (1)."""

    def __init__(self, name: str, target: float) -> None:
        if (name, target) not in (("flat", 0.0), ("buy-and-hold", 1.0)):
            raise ValueError(f"constant baselines are flat (0) and buy-and-hold (1), got {name!r} = {target}")
        self.name = name
        self.config = ConstantConfig(target=float(target))

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        return _output(test, target=np.full(test.height, self.config.target))


def flat() -> Constant:
    return Constant("flat", 0.0)


def buy_and_hold() -> Constant:
    return Constant("buy-and-hold", 1.0)


@dataclass(frozen=True)
class MomentumConfig:
    window: int


def _momentum_target(frame: pl.DataFrame, window: int) -> np.ndarray:
    """1 iff ``f_ma_gap_{window} > 0`` (close above its moving average); null -> 0."""
    column = f"f_ma_gap_{window}"
    if column not in frame.columns:
        raise ValueError(f"momentum needs feature {column!r}")
    gap = frame[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
    with np.errstate(invalid="ignore"):
        return (gap > 0.0).astype(np.float64)  # NaN > 0 is False


class MaMomentum:
    """Long while the close is above its ``window``-bar moving average, else flat."""

    def __init__(self, window: int) -> None:
        if window not in MA_WINDOWS:
            raise ValueError(f"momentum windows are fixed to {MA_WINDOWS}, got {window}")
        self.name = f"ma-momentum-{window}"
        self.config = MomentumConfig(window=window)

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        return _output(test, target=_momentum_target(test, self.config.window))


# --- Random positions with matched turnover -----------------------------------------

_GAMMA = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1


def _splitmix64_int(x: int) -> int:
    z = (x + _GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def counter_uniform(seed: int, symbol: str, stream: int, index: np.ndarray) -> np.ndarray:
    """Uniforms in [0, 1) keyed by (seed, symbol, stream, bar index): a stateless, stable RNG.

    The value at a bar does not depend on which other bars are asked for, so a fold's
    chain continues the previous fold's exactly. ``zlib.crc32`` keys the symbol because
    Python's own ``hash`` of a string changes from one process to the next.
    """
    base = _splitmix64_int(_splitmix64_int(_splitmix64_int(seed) ^ zlib.crc32(symbol.encode())) ^ stream)
    with np.errstate(over="ignore"):
        counter = np.uint64(base) + (np.asarray(index, dtype=np.int64).astype(np.uint64) + np.uint64(1)) * np.uint64(
            _GAMMA
        )
    bits = _splitmix64(_splitmix64(counter)) >> np.uint64(11)
    return bits.astype(np.float64) * 2.0**-53


@dataclass(frozen=True)
class RandomTurnoverConfig:
    seed: int
    interval: str
    reference_window: int = 30  # turnover matched to ma-momentum-30


class RandomTurnover:
    """A long/flat Markov chain whose switch probability is ma-momentum-30's training flip rate.

    The chain lives on the absolute bar grid (bar index ``open_time // interval``), built by
    regeneration: at each bar, with probability ``2p``, the state is redrawn by a fair coin;
    otherwise it is kept. It therefore switches with probability ``p`` per bar and is long
    half the time in the long run. The state at a bar is the coin of the last regeneration
    at or before it, so it is a pure function of (seed, symbol, bar, p): predicting two
    adjacent months gives what predicting both at once gives, and a refit that moves ``p``
    a little rarely moves the state at the fold boundary. ``p`` must be at most 1/2.

    The bench runs 100 seeds and reports their distribution as one variant.
    """

    name = "random-turnover"

    def __init__(self, seed: int, interval: str) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
        interval_ms(interval)
        self.config = RandomTurnoverConfig(seed=seed, interval=interval)
        self.switch_probability_: float | None = None

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train, (f"f_ma_gap_{self.config.reference_window}",))
        step = interval_ms(self.config.interval)
        flips = transitions = 0
        for part in train.sort(["symbol", "open_time"]).partition_by("symbol", maintain_order=True):
            target = _momentum_target(part, self.config.reference_window)
            consecutive = np.diff(part["open_time"].dt.epoch("ms").to_numpy()) == step
            flips += int(np.sum((np.diff(target) != 0) & consecutive))
            transitions += int(np.sum(consecutive))
        if transitions == 0:
            raise ValueError("random-turnover: no two consecutive training bars to measure a flip rate on")
        p = flips / transitions
        if p > 0.5:
            raise ValueError(f"random-turnover: flip rate {p:.3f} > 1/2 is beyond a regeneration chain")
        self.switch_probability_ = p

    def _states(self, symbol: str, index: np.ndarray) -> np.ndarray:
        p = self.switch_probability_
        assert p is not None
        n0, n1 = int(index.min()), int(index.max())
        window = 256
        while True:
            start = max(0, n0 - window)
            grid = np.arange(start, n1 + 1, dtype=np.int64)
            regen = counter_uniform(self.config.seed, symbol, 0, grid) < 2.0 * p
            if start == 0:
                regen[0] = True  # bar 0 of the epoch is the chain's origin
            if regen[: n0 - start + 1].any():
                break
            window *= 4
        last = np.maximum.accumulate(np.where(regen, np.arange(grid.size), -1))
        coin = counter_uniform(self.config.seed, symbol, 1, grid[last]) < 0.5
        return coin[index - start].astype(np.float64)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        _require_fitted(self, "switch_probability_")
        step = interval_ms(self.config.interval)
        index = test["open_time"].dt.epoch("ms").to_numpy() // step
        if index.size and index.min() < 0:
            raise ValueError("random-turnover: bars before 1970 are off its grid")
        symbols = test["symbol"].to_numpy()
        target = np.zeros(test.height)
        for symbol in np.unique(symbols):
            rows = symbols == symbol
            target[rows] = self._states(str(symbol), index[rows])
        return _output(test, target=target)


# --- Forecasters --------------------------------------------------------------------


@dataclass(frozen=True)
class ClimatologyConfig:
    pass


class Climatology:
    """Training class frequencies as constant probabilities; ``exp_return = sum_k p_k m_k``."""

    name = "climatology"

    def __init__(self) -> None:
        self.config = ClimatologyConfig()
        self.proba_: np.ndarray | None = None
        self.class_means_: np.ndarray | None = None

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train, ("label", "fwd_return"))
        rows = _chronological(train.filter(pl.col("label").is_not_null() & pl.col("fwd_return").is_not_null()))
        if rows.height == 0:
            raise ValueError("climatology: no labelled training rows")
        label = rows["label"].cast(pl.Int64).to_numpy()
        if label.min() < 0 or label.max() >= N_CLASSES:
            raise ValueError(f"labels must be in 0..{N_CLASSES - 1}")
        self.proba_ = np.bincount(label, minlength=N_CLASSES) / label.size
        self.class_means_ = _class_means(rows["fwd_return"].to_numpy(), label)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        _require_fitted(self, "proba_")
        assert self.proba_ is not None and self.class_means_ is not None
        proba = np.tile(self.proba_, (test.height, 1))
        exp_return = np.full(test.height, float(self.proba_ @ self.class_means_))
        return _output(test, proba=proba, exp_return=exp_return)


@dataclass(frozen=True)
class RidgeConfig:
    lambdas: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4)
    validation_fraction: float = 0.2


def ridge_closed_form(z: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Ridge on centred columns with an unpenalized intercept, by Cholesky: ``(intercept, beta)``.

    ``z`` must be centred on these rows; then the intercept is ``mean(y)`` exactly and
    ``beta = (zᵀz + λI)⁻¹ zᵀ(y − ȳ)``. No explicit inverse.
    """
    y_bar = float(np.mean(y))
    gram = z.T @ z + lam * np.eye(z.shape[1])
    beta = cho_solve(cho_factor(gram, lower=True), z.T @ (y - y_bar))
    return y_bar, beta


class Ridge:
    """Ridge on the standardized features, Gaussian class probabilities around its forecast.

    - Standardization uses the training rows' means and stds (ddof 0; a constant feature keeps std 1).
    - λ is chosen inside the fold: the last ``validation_fraction`` of training rows in time
      validate, and an inner training row is used only if its label is known at least
      ``h`` bars before the first validation decision (purge plus an embargo of h, as in
      the outer splitter; ``h·interval`` is read off ``exit_time − entry_time``). The
      criterion is the validation mean squared error of ``fwd_return``; a tie goes to the
      larger λ. The model is then refitted on all training rows.
    - ``sigma_`` is the residual std on all training rows, with the ridge's effective
      degrees of freedom ``tr(H) = Σ e/(e+λ)`` (e the eigenvalues of zᵀz) plus one for the
      intercept taken off the denominator.
    - ``p_up = 1 − Φ((θ − μ̂)/σ̂)``, ``p_down = Φ((−θ − μ̂)/σ̂)``, ``p_neutral`` the rest.
    """

    name = "ridge"

    def __init__(self, config: RidgeConfig | None = None) -> None:
        self.config = config or RidgeConfig()
        if not self.config.lambdas or min(self.config.lambdas) <= 0:
            raise ValueError("ridge: every lambda must be > 0")
        if not 0.0 < self.config.validation_fraction < 1.0:
            raise ValueError("ridge: validation_fraction must be in (0, 1)")
        self.features_: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.beta_: np.ndarray | None = None
        self.sigma_: float | None = None
        self.lambda_: float | None = None
        self.selection_: dict[float, float] = {}

    @staticmethod
    def _standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std > 0.0, std, 1.0)
        return mean, std

    def _select(self, rows: pl.DataFrame, x: np.ndarray, y: np.ndarray) -> float:
        n = rows.height
        n_val = max(1, int(round(self.config.validation_fraction * n)))
        decision = rows["decision_time"]
        val_start = decision[n - n_val]
        embargo = (rows["exit_time"] - rows["entry_time"]).max()
        cutoff = val_start - embargo
        val = (decision >= val_start).to_numpy()
        inner = ((rows["label_known_at"] <= cutoff) & (decision < val_start)).to_numpy()
        if inner.sum() <= x.shape[1] + 1 or not val.any():
            raise ValueError(
                f"ridge: too few rows to choose lambda ({int(inner.sum())} inner, {int(val.sum())} validation)"
            )
        mean, std = self._standardizer(x[inner])
        z_inner, z_val = (x[inner] - mean) / std, (x[val] - mean) / std
        best, best_mse = None, math.inf
        self.selection_ = {}
        for lam in sorted(self.config.lambdas):
            intercept, beta = ridge_closed_form(z_inner, y[inner], lam)
            mse = float(np.mean((y[val] - intercept - z_val @ beta) ** 2))
            self.selection_[lam] = mse
            if mse <= best_mse:  # ascending grid: a tie goes to the larger lambda
                best, best_mse = lam, mse
        assert best is not None
        return best

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train, ("fwd_return", "entry_time", "exit_time", "label_known_at"))
        names = [c for c in train.columns if c.startswith(FEATURE_PREFIX)]
        if not names:
            raise ValueError("ridge: no f_* features in the training rows")
        rows = _chronological(train.filter(pl.col("fwd_return").is_not_null()))
        x = _matrix(rows, names)
        keep = _complete(x) & np.isfinite(rows["fwd_return"].to_numpy())
        rows, x = rows.filter(pl.Series(keep)), x[keep]
        y = rows["fwd_return"].to_numpy().astype(np.float64)
        lam = self._select(rows, x, y)

        mean, std = self._standardizer(x)
        z = (x - mean) / std
        intercept, beta = ridge_closed_form(z, y, lam)
        residual = y - intercept - z @ beta
        eig = np.clip(np.linalg.eigvalsh(z.T @ z), 0.0, None)
        dof = y.size - 1.0 - float(np.sum(eig / (eig + lam)))
        if dof <= 0:
            raise ValueError("ridge: no residual degrees of freedom")
        sigma = float(np.sqrt(residual @ residual / dof))
        if not sigma > 0:
            raise ValueError("ridge: zero residual std")
        self.features_, self.mean_, self.std_ = names, mean, std
        self.intercept_, self.beta_, self.sigma_, self.lambda_ = intercept, beta, sigma, lam

    def forecast(self, test: pl.DataFrame) -> np.ndarray:
        """μ̂ per row, NaN where a feature is missing."""
        _require_fitted(self, "beta_")
        assert self.features_ is not None and self.mean_ is not None and self.std_ is not None
        assert self.beta_ is not None and self.intercept_ is not None
        missing = [c for c in self.features_ if c not in test.columns]
        if missing:
            raise ValueError(f"ridge: test rows lack features {missing}")
        x = _matrix(test, self.features_)
        mu = np.full(test.height, np.nan)
        ok = _complete(x)
        mu[ok] = self.intercept_ + ((x[ok] - self.mean_) / self.std_) @ self.beta_
        return mu

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        if "threshold" not in test.columns:
            raise ValueError("ridge: test rows lack 'threshold'")
        mu = self.forecast(test)
        assert self.sigma_ is not None
        theta = test["threshold"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        ok = np.isfinite(mu) & np.isfinite(theta) & (theta > 0)
        proba = np.full((test.height, N_CLASSES), np.nan)
        p_up = norm.sf((theta[ok] - mu[ok]) / self.sigma_)
        p_down = norm.cdf((-theta[ok] - mu[ok]) / self.sigma_)
        proba[ok] = np.column_stack([p_down, np.maximum(1.0 - p_up - p_down, 0.0), p_up])
        return _output(test, proba=proba, exp_return=mu)


@dataclass(frozen=True)
class XGBoostConfig:
    max_depth: int = 3
    n_estimators: int = 300
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    seed: int = 0
    # One thread: the hist builder's row sampling depends on the thread count, so outputs are
    # bit-identical only at a fixed nthread (measured: up to 0.11 apart in probability between
    # 1, 4 and 12 threads). One thread ties them to the data and seed alone, at ~20 % more time.
    nthread: int = 1
    tree_method: str = "hist"

    def params(self) -> dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": N_CLASSES,
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "seed": self.seed,
            "nthread": self.nthread,
            "tree_method": self.tree_method,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }


class XGBoost:
    """Gradient-boosted trees, ``multi:softprob`` on the 3 classes; ``exp_return = Σ_k p_k m_k``.

    Trains on the rows whose features are all present (sorted by time, so the fit does not
    depend on the input order); predicts null where a feature is missing.
    """

    name = "xgboost"

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        self.config = config or XGBoostConfig()
        self.features_: list[str] | None = None
        self.booster_: xgb.Booster | None = None
        self.class_means_: np.ndarray | None = None

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train, ("label", "fwd_return"))
        names = [c for c in train.columns if c.startswith(FEATURE_PREFIX)]
        if not names:
            raise ValueError("xgboost: no f_* features in the training rows")
        rows = _chronological(train.filter(pl.col("label").is_not_null() & pl.col("fwd_return").is_not_null()))
        x = _matrix(rows, names)
        keep = _complete(x)
        if not keep.any():
            raise ValueError("xgboost: no complete training rows")
        x = x[keep]
        label = rows["label"].cast(pl.Int64).to_numpy()[keep]
        fwd = rows["fwd_return"].to_numpy()[keep]
        dtrain = xgb.DMatrix(x, label=label, feature_names=names, nthread=self.config.nthread)
        self.booster_ = xgb.train(self.config.params(), dtrain, num_boost_round=self.config.n_estimators)
        self.features_ = names
        self.class_means_ = _class_means(fwd, label)

    def predict_proba(self, test: pl.DataFrame) -> np.ndarray:
        _require_fitted(self, "booster_")
        assert self.booster_ is not None and self.features_ is not None
        missing = [c for c in self.features_ if c not in test.columns]
        if missing:
            raise ValueError(f"xgboost: test rows lack features {missing}")
        x = _matrix(test, self.features_)
        ok = _complete(x)
        proba = np.full((test.height, N_CLASSES), np.nan)
        if ok.any():
            raw = self.booster_.predict(xgb.DMatrix(x[ok], feature_names=self.features_, nthread=self.config.nthread))
            raw = raw.astype(np.float64)
            proba[ok] = raw / raw.sum(axis=1, keepdims=True)
        return proba

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        proba = self.predict_proba(test)
        assert self.class_means_ is not None
        return _output(test, proba=proba, exp_return=proba @ self.class_means_)


# --- The cheater --------------------------------------------------------------------


@dataclass(frozen=True)
class CheaterConfig:
    leaks: str = field(default="reads the next bar: close[t+1] > open[t+1]")


class Cheater:
    """!!! LEAKS ON PURPOSE. NOT A MODEL. !!!

    Long at t iff ``close[t+1] > open[t+1]``: it is built with the bars and reads the bar
    *after* the decision. It exists only so that the bench's leakage test (H1, the
    future-invariance test of ``docs/PROTOCOL.md``) has something to catch. A bench run in
    which the cheater passes is broken. Never report it as a result, never copy it.
    """

    name = "cheater"

    def __init__(self, bars: pl.DataFrame) -> None:
        missing = [c for c in ("symbol", "open_time", "open", "close") if c not in bars.columns]
        if missing:
            raise ValueError(f"cheater: bars lack columns {missing}")
        self.config = CheaterConfig()
        # THE LEAK: each bar is paired with the NEXT bar's candle colour.
        self._peek = (
            bars.select("symbol", pl.col("open_time").cast(TIMESTAMP), "open", "close")
            .sort(["symbol", "open_time"])
            .select(
                "symbol",
                "open_time",
                (pl.col("close").shift(-1).over("symbol") > pl.col("open").shift(-1).over("symbol")).alias("_next_up"),
            )
        )

    def fit(self, train: pl.DataFrame) -> None:
        _check_train_frame(train)

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        peeked = (
            test.select("symbol", pl.col("open_time").cast(TIMESTAMP))
            .with_row_index("_row")
            .join(self._peek, on=["symbol", "open_time"], how="left")
            .sort("_row")
        )
        if peeked.height != test.height:
            raise ValueError("cheater: duplicate bars")
        target = peeked["_next_up"].fill_null(False).cast(pl.Float64).to_numpy()
        return _output(test, target=target)


# --- The fixed list -----------------------------------------------------------------


def honest_baselines(interval: str, *, seed: int = 0) -> list[Baseline]:
    """Every baseline but the cheater, in :data:`BASELINE_NAMES` order (one random-turnover seed)."""
    return [
        flat(),
        buy_and_hold(),
        *(MaMomentum(w) for w in MA_WINDOWS),
        RandomTurnover(seed, interval),
        Climatology(),
        Ridge(),
        XGBoost(),
    ]
