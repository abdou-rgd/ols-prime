"""Statistics of the bench: intervals and tests for every number reported in ``docs/RESULTS.md``.

Pure numpy and scipy, no I/O. Every random function takes a ``seed`` (an ``int`` or a
``numpy.random.Generator``), keyword-only and required, so that every interval is reproducible.

Conventions
-----------
- ``r`` is a 1-D array of per-period simple returns, in time order.
- ``SR`` is the per-period Sharpe ratio, ``mean / std`` with ddof 1. Nothing here subtracts a
  risk-free rate; pass excess returns if one is wanted.
- ``g3`` is the skewness and ``g4`` the **non-excess** kurtosis (normal = 3), both population
  moments (``scipy.stats.skew(bias=True)``, ``scipy.stats.kurtosis(fisher=False, bias=True)``).
- ``T`` is the number of returns.
- Forecasts are probability arrays of shape ``(n, K)`` (``K = 3`` on the bench: down, flat, up);
  labels are integers in ``{0, ..., K - 1}``. Rows must be in time order for the bootstraps.

References
----------
- Bailey, D. H. and López de Prado, M. (2012), "The Sharpe ratio efficient frontier",
  Journal of Risk 15(2). Probabilistic Sharpe ratio.
- Bailey, D. H. and López de Prado, M. (2014), "The deflated Sharpe ratio", Journal of Portfolio
  Management 40(5), Eq. (1) and (2).
- Mertens, E. (2002), "Comments on variance of the IID estimator in Lo (2002)". Opdyke, J. D.
  (2007), "Comparing Sharpe ratios: so where are the p-values?", gives the same non-normal iid
  formula; neither corrects for serial correlation, which the stationary bootstrap handles.
- Lo, A. W. (2002), "The statistics of Sharpe ratios", Financial Analysts Journal 58(4).
- Politis, D. N. and Romano, J. P. (1994), "The stationary bootstrap", JASA 89(428).
- Politis, D. N. and White, H. (2004), "Automatic block-length selection for the dependent
  bootstrap", Econometric Reviews 23(1); corrected by Patton, A., Politis, D. N. and White, H.
  (2009), Econometric Reviews 28(4).
- Diebold, F. X. and Mariano, R. S. (1995), "Comparing predictive accuracy", JBES 13(3).
- Harvey, D., Leybourne, S. and Newbold, P. (1997), "Testing the equality of prediction mean
  squared errors", IJF 13(2).
- Newey, W. K. and West, K. D. (1987), Econometrica 55(3).
- Kull, M. et al. (2019), "Beyond temperature scaling: obtaining well-calibrated multiclass
  probabilities with Dirichlet calibration", NeurIPS. Class-wise ECE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats as _st

__all__ = [
    "EULER_GAMMA",
    "Interval",
    "DieboldMariano",
    "RowLosses",
    "ReliabilityTable",
    "sharpe",
    "annualized",
    "moments",
    "sharpe_se",
    "sharpe_ci",
    "sharpe_ci_bootstrap",
    "sharpe_diff_ci_bootstrap",
    "psr",
    "expected_max_sharpe",
    "trials_var_sr",
    "dsr",
    "optimal_block_length",
    "stationary_bootstrap_indices",
    "log_loss",
    "brier",
    "brier_skill",
    "per_row_losses",
    "reliability_table",
    "ece",
    "classwise_ece",
    "ece_ci_bootstrap",
    "bss_ci_bootstrap",
    "diebold_mariano",
]

EULER_GAMMA = 0.5772156649015329
"""The Euler-Mascheroni constant, used by the expected maximum of N Sharpe ratios."""

Seed = int | np.random.Generator
Strategy = Literal["width", "mass"]

# Bootstrap draws are generated in chunks of at most this many indices, to bound memory.
_CHUNK_ELEMENTS = 2_000_000


# ---------------------------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval.

    ``p_value`` is set only by the comparison bootstraps (Sharpe difference, Brier skill): it is
    the two-sided percentile-bootstrap p-value of H0 "the quantity is 0", that is
    ``2 * min(P*(stat <= 0), P*(stat >= 0))``, the smallest level whose percentile interval
    excludes 0. ``mean_block`` is the stationary bootstrap's mean block length, ``n_valid`` the
    number of bootstrap draws on which the statistic was defined.
    """

    estimate: float
    low: float
    high: float
    level: float
    method: str
    mean_block: float | None = None
    n_boot: int | None = None
    n_valid: int | None = None
    p_value: float | None = None

    def excludes(self, value: float = 0.0) -> bool:
        """Whether ``value`` lies outside the closed interval ``[low, high]``."""
        return bool(value < self.low or value > self.high)


@dataclass(frozen=True)
class DieboldMariano:
    """Result of :func:`diebold_mariano`.

    ``mean_diff`` is the mean of ``d = loss_a - loss_b``: negative means forecast ``a`` has the
    lower loss. ``statistic`` has the sign of ``mean_diff``. ``kernel_used`` and ``h_used`` are the
    HAC kernel and horizon actually used; they differ from the requested ones only after a
    fallback, which ``note`` then describes.
    """

    statistic: float
    p_value: float
    mean_diff: float
    h: int
    h_used: int
    n: int
    note: str = ""
    kernel_used: str = "rectangular"


@dataclass(frozen=True)
class RowLosses:
    """Per-row losses of a probabilistic forecast, in row order."""

    log_loss: np.ndarray
    brier: np.ndarray


@dataclass(frozen=True)
class ReliabilityTable:
    """Binned calibration of a score against a binary outcome; empty bins are dropped.

    ``low`` and ``high`` are the bin edges for ``strategy="width"`` and the smallest and largest
    score in the bin for ``strategy="mass"``. ``confidence`` is the mean score and ``frequency``
    the observed frequency of the outcome in each bin.
    """

    low: np.ndarray
    high: np.ndarray
    count: np.ndarray
    confidence: np.ndarray
    frequency: np.ndarray

    @property
    def ece(self) -> float:
        """``sum_b n_b / n * |frequency_b - confidence_b|``."""
        n = self.count.sum()
        if n == 0:
            return float("nan")
        return float(np.sum(self.count / n * np.abs(self.frequency - self.confidence)))


# ---------------------------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------------------------


def _returns(r: np.ndarray, *, min_len: int = 3) -> np.ndarray:
    x = np.asarray(r, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {x.shape}")
    if x.size < min_len:
        raise ValueError(f"need at least {min_len} returns, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain NaN or infinite values")
    return x


def _level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return float(level)


def _rng(seed: Seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        return np.random.default_rng(int(seed))
    raise TypeError("seed must be an int or a numpy.random.Generator")


def _probs(p: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    q = np.asarray(p, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 2:
        raise ValueError(f"probabilities must have shape (n, K) with K >= 2, got {q.shape}")
    if q.shape[0] == 0:
        raise ValueError("no forecasts")
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1")
    if y is None:
        return q, None
    labels = np.asarray(y)
    if labels.shape != (q.shape[0],):
        raise ValueError(f"labels must have shape ({q.shape[0]},), got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(labels == np.round(labels)):
            raise ValueError("labels must be integers")
    labels = labels.astype(np.int64)
    if labels.min() < 0 or labels.max() >= q.shape[1]:
        raise ValueError(f"labels must lie in [0, {q.shape[1] - 1}]")
    return q, labels


def _reference(p_ref: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A reference forecast: either per row ``(n, K)`` or one distribution ``(K,)``, broadcast."""
    q = np.asarray(p_ref, dtype=np.float64)
    if q.ndim == 1:
        q = np.broadcast_to(q, shape)
    q, _ = _probs(q)
    if q.shape != shape:
        raise ValueError(f"reference forecast has shape {q.shape}, expected {shape}")
    return q


# ---------------------------------------------------------------------------------------------
# Sharpe ratios
# ---------------------------------------------------------------------------------------------


def sharpe(r: np.ndarray) -> float:
    """Per-period Sharpe ratio ``mean(r) / std(r, ddof=1)``."""
    x = _returns(r, min_len=2)
    s = x.std(ddof=1)
    if s == 0.0:
        raise ValueError("returns have zero variance; the Sharpe ratio is undefined")
    return float(x.mean() / s)


def annualized(sr: float, periods_per_year: float) -> float:
    """``sr * sqrt(periods_per_year)``: the iid scaling (365 for daily crypto, 8760 for hourly)."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return float(sr) * math.sqrt(periods_per_year)


def moments(r: np.ndarray) -> tuple[float, float]:
    """Population skewness ``g3`` and non-excess kurtosis ``g4`` (normal = 3) of ``r``."""
    x = _returns(r)
    g3 = float(_st.skew(x, bias=True))
    g4 = float(_st.kurtosis(x, fisher=False, bias=True))
    return g3, g4


def _se_from(sr: float, g3: float, g4: float, t: int) -> float:
    # By Pearson's inequality g4 >= g3^2 + 1, the radicand is >= (1 - g3*SR/2)^2 >= 0; the clip
    # only absorbs rounding.
    radicand = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    return math.sqrt(max(radicand, 0.0) / (t - 1))


def sharpe_se(r: np.ndarray) -> float:
    """Standard error of the per-period Sharpe ratio for iid, possibly non-normal, returns.

    ``sqrt((1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1))`` (Mertens 2002; Opdyke 2007). This is the
    radicand inside the PSR of Bailey and López de Prado (2012). With normal returns
    (``g3 = 0``, ``g4 = 3``) it reduces to Lo's (2002) ``sqrt((1 + SR^2/2) / T)``, with ``T - 1``
    in place of ``T``. Negative skewness and fat tails widen it when ``SR > 0``.

    It assumes iid returns: with serial correlation, use :func:`sharpe_ci_bootstrap`.
    """
    x = _returns(r)
    sr = sharpe(x)
    g3, g4 = moments(x)
    return _se_from(sr, g3, g4, x.size)


def sharpe_ci(r: np.ndarray, periods_per_year: float = 1.0, level: float = 0.95) -> Interval:
    """Analytic interval ``SR +- z * sharpe_se(r)``, annualized by ``sqrt(periods_per_year)``.

    Valid for iid returns only; it under-covers under positive autocorrelation.
    """
    level = _level(level)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    x = _returns(r)
    sr = sharpe(x)
    se = sharpe_se(x)
    z = float(_st.norm.ppf(0.5 + level / 2.0))
    k = math.sqrt(periods_per_year)
    return Interval(sr * k, (sr - z * se) * k, (sr + z * se) * k, level, "analytic (Mertens 2002)")


# ---------------------------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------------------------


def optimal_block_length(x: np.ndarray) -> int:
    """Mean block length of the stationary bootstrap (Politis and White 2004, corrected by Patton,
    Politis and White 2009), rounded to an integer in ``[1, ceil(min(3 sqrt(T), T / 3))]``.

    The steps follow Patton's reference code (``opt_block_length_REV_dec07.m``):

    1. ``K_N = max(5, ceil(sqrt(log10 T)))``, ``m_max = ceil(sqrt T) + K_N``, threshold
       ``c * sqrt(log10(T) / T)`` with ``c = 2``.
    2. ``m_hat`` is the smallest ``m >= 1`` such that the ``K_N`` autocorrelations at lags
       ``m, ..., m + K_N - 1`` are all below the threshold in absolute value; if there is none,
       the last significant lag. ``M = min(2 m_hat, m_max)``.
    3. With the flat-top window ``lambda(s) = min(1, 2 (1 - |s|))`` on ``|s| <= 1`` and the
       autocovariances ``R(k)`` (divisor ``T``):
       ``G = sum_{|k|<=M} lambda(k/M) |k| R(k)``, ``g = sum_{|k|<=M} lambda(k/M) R(k)``,
       ``D_SB = 2 g^2`` (the 2009 correction), ``B = (2 G^2 / D_SB)^(1/3) T^(1/3)``.

    White noise gives about 1; positive autocorrelation gives longer blocks. The ``arch`` package
    differs slightly in its lag-window choices, so the two need not agree exactly.
    """
    v = _returns(x, min_len=8)
    t = v.size
    c = v - v.mean()
    r0 = float(c @ c) / t
    b_max = int(math.ceil(min(3.0 * math.sqrt(t), t / 3.0)))
    if r0 == 0.0:
        return 1
    k_n = max(5, int(math.ceil(math.sqrt(math.log10(t)))))
    m_max = int(math.ceil(math.sqrt(t))) + k_n
    m_max = min(m_max, t - 1)
    acov = np.array([float(c[: t - k] @ c[k:]) / t for k in range(m_max + 1)])
    rho = acov[1:] / r0  # lags 1..m_max
    threshold = 2.0 * math.sqrt(math.log10(t) / t)
    insignificant = np.abs(rho) < threshold
    m_hat = None
    for m in range(1, m_max - k_n + 2):
        if np.all(insignificant[m - 1 : m - 1 + k_n]):
            m_hat = m
            break
    if m_hat is None:
        significant = np.flatnonzero(~insignificant)
        m_hat = int(significant[-1]) + 1 if significant.size else 1
    big_m = min(2 * m_hat, m_max)
    lags = np.arange(1, big_m + 1)
    s = lags / big_m
    lam = np.minimum(1.0, 2.0 * (1.0 - s))
    g_cap = 2.0 * float(np.sum(lam * lags * acov[1 : big_m + 1]))
    g_small = r0 + 2.0 * float(np.sum(lam * acov[1 : big_m + 1]))
    d_sb = 2.0 * g_small * g_small
    if d_sb <= 0.0 or g_cap == 0.0:
        return 1
    b = (2.0 * g_cap * g_cap / d_sb) ** (1.0 / 3.0) * t ** (1.0 / 3.0)
    return int(min(max(1, round(b)), b_max))


def _sb_chunk(t: int, p: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` stationary-bootstrap index paths of length ``t``, circular, new block w.p. ``p``."""
    starts = rng.integers(0, t, size=(n, t))
    new = rng.random((n, t)) < p
    new[:, 0] = True
    steps = np.arange(t)
    pos = np.where(new, steps, 0)
    np.maximum.accumulate(pos, axis=1, out=pos)
    return (np.take_along_axis(starts, pos, axis=1) + (steps - pos)) % t


def _sb_chunks(t: int, mean_block: float, n_boot: int, rng: np.random.Generator):
    if not mean_block >= 1.0:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    p = 1.0 / min(float(mean_block), float(t))
    rows = max(1, _CHUNK_ELEMENTS // t)
    done = 0
    while done < n_boot:
        n = min(rows, n_boot - done)
        yield _sb_chunk(t, p, n, rng)
        done += n


def stationary_bootstrap_indices(t: int, mean_block: float, n_boot: int, *, seed: Seed) -> np.ndarray:
    """Index array of shape ``(n_boot, t)`` for the stationary bootstrap (Politis and Romano 1994).

    Each path starts at a uniform index; at every later step it starts a new block at a uniform
    index with probability ``1 / mean_block``, else moves to the next index, wrapping around.
    Block lengths are geometric with mean ``mean_block``; ``mean_block = 1`` is the iid bootstrap.
    """
    if t < 1:
        raise ValueError("t must be positive")
    rng = _rng(seed)
    return np.concatenate(list(_sb_chunks(int(t), mean_block, int(n_boot), rng)), axis=0)


def _row_sharpe(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0.0, m / s, np.nan)


def _percentile(
    draws: np.ndarray, estimate: float, level: float, method: str, mean_block: float, with_p: bool
) -> Interval:
    finite = draws[np.isfinite(draws)]
    if finite.size < max(10, draws.size // 2):
        raise ValueError(f"only {finite.size} of {draws.size} bootstrap draws were defined")
    a = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [a, 1.0 - a])
    p = None
    if with_p:
        p = float(min(1.0, 2.0 * min(np.mean(finite <= 0.0), np.mean(finite >= 0.0))))
    return Interval(
        float(estimate), float(low), float(high), level, method, float(mean_block), int(draws.size), int(finite.size), p
    )


def sharpe_ci_bootstrap(
    r: np.ndarray,
    periods_per_year: float = 1.0,
    level: float = 0.95,
    n_boot: int = 2000,
    *,
    seed: Seed,
    mean_block: float | None = None,
) -> Interval:
    """Stationary-bootstrap percentile interval of the Sharpe ratio, annualized.

    With ``mean_block=None`` the mean block length is :func:`optimal_block_length` of ``r``.
    Robust to serial correlation and non-normality, but percentile intervals of a ratio are
    slightly short at small T.
    """
    level = _level(level)
    x = _returns(r)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    b = float(optimal_block_length(x) if mean_block is None else mean_block)
    rng = _rng(seed)
    draws = np.concatenate([_row_sharpe(x[idx]) for idx in _sb_chunks(x.size, b, n_boot, rng)])
    k = math.sqrt(periods_per_year)
    return _percentile(draws * k, sharpe(x) * k, level, "stationary bootstrap, percentile", b, False)


def sharpe_diff_ci_bootstrap(
    r_a: np.ndarray,
    r_b: np.ndarray,
    periods_per_year: float = 1.0,
    level: float = 0.95,
    n_boot: int = 2000,
    *,
    seed: Seed,
    mean_block: float | None = None,
) -> Interval:
    """Paired stationary-bootstrap interval of ``SR(r_a) - SR(r_b)``, annualized.

    Both series are resampled with the same indices, so their correlation is kept. With
    ``mean_block=None`` the block length is the larger of the two series' optimal lengths.
    ``p_value`` is the two-sided percentile-bootstrap p-value of H0 "equal Sharpe ratios".
    """
    level = _level(level)
    a = _returns(r_a)
    b_ = _returns(r_b)
    if a.shape != b_.shape:
        raise ValueError(f"paired series must have the same length, got {a.size} and {b_.size}")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    blk = float(max(optimal_block_length(a), optimal_block_length(b_)) if mean_block is None else mean_block)
    rng = _rng(seed)
    draws = np.concatenate(
        [_row_sharpe(a[idx]) - _row_sharpe(b_[idx]) for idx in _sb_chunks(a.size, blk, n_boot, rng)]
    )
    k = math.sqrt(periods_per_year)
    est = (sharpe(a) - sharpe(b_)) * k
    return _percentile(draws * k, est, level, "paired stationary bootstrap, percentile", blk, True)


# ---------------------------------------------------------------------------------------------
# Probabilistic and deflated Sharpe ratios
# ---------------------------------------------------------------------------------------------


def psr(r: np.ndarray, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey and López de Prado 2012):
    ``Phi[(SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)]``.

    ``SR`` and ``sr_star`` are per-period, not annualized. Equal to
    ``Phi((SR - SR*) / sharpe_se(r))``. iid returns are assumed.
    """
    x = _returns(r)
    sr = sharpe(x)
    se = sharpe_se(x)
    if se == 0.0:
        return float(1.0 if sr > sr_star else 0.0 if sr < sr_star else 0.5)
    return float(_st.norm.cdf((sr - float(sr_star)) / se))


def expected_max_sharpe(n_trials: int, var_sr: float | None) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe ratios under the null of no skill
    (Bailey and López de Prado 2014, Eq. 2):
    ``sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))``.

    ``N = 1`` is undefined in the paper (``Phi^-1(0)``); it is special-cased to 0, so that the DSR
    of a single trial is ``PSR(0)``. For ``N >= 2``, ``var_sr`` (the cross-sectional variance of the
    trials' estimated Sharpe ratios) is required.
    """
    n = int(n_trials)
    if n != n_trials or n < 1:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    if n == 1:
        return 0.0
    if var_sr is None or not np.isfinite(var_sr) or var_sr < 0.0:
        raise ValueError("var_sr must be a finite, non-negative variance when n_trials >= 2")
    q1 = _st.norm.ppf(1.0 - 1.0 / n)
    q2 = _st.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2))


def trials_var_sr(trial_sharpes: np.ndarray) -> float:
    """Cross-sectional variance (ddof 1) of the per-period Sharpe ratios of all trials.

    Every variant tried counts, including the abandoned ones. With few trials this estimate is
    itself very noisy, and so is the DSR that uses it; at least two trials are needed.
    """
    s = np.asarray(trial_sharpes, dtype=np.float64)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("need the Sharpe ratios of at least two trials")
    if not np.all(np.isfinite(s)):
        raise ValueError("trial Sharpe ratios must be finite")
    return float(s.var(ddof=1))


def dsr(r: np.ndarray, n_trials: int, var_sr: float | None = None) -> float:
    """Deflated Sharpe ratio (Bailey and López de Prado 2014, Eq. 1):
    ``psr(r, expected_max_sharpe(n_trials, var_sr))``.

    ``r`` is the selected trial's returns; ``n_trials`` counts every variant tried; ``var_sr`` is
    :func:`trials_var_sr` of all trials (per-period, ddof 1), required when ``n_trials >= 2``. At
    ``N = 1`` the DSR is ``PSR(0)`` and ``var_sr`` is ignored. With few trials, ``var_sr`` is noisy:
    report it beside the DSR.
    """
    return psr(r, expected_max_sharpe(n_trials, var_sr))


# ---------------------------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------------------------


def per_row_losses(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> RowLosses:
    """Per-row log-loss ``-log p_y`` (clipped at ``eps``) and multi-class Brier
    ``sum_k (p_k - 1{y = k})^2``."""
    q, labels = _probs(p, y)
    assert labels is not None
    rows = np.arange(q.shape[0])
    ll = -np.log(np.clip(q[rows, labels], eps, 1.0))
    onehot = np.zeros_like(q)
    onehot[rows, labels] = 1.0
    br = np.sum((q - onehot) ** 2, axis=1)
    return RowLosses(ll, br)


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    """Mean log-loss, natural log."""
    return float(per_row_losses(p, y, eps).log_loss.mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean multi-class Brier score ``sum_k (p_k - 1{y = k})^2``, in ``[0, 2]``."""
    return float(per_row_losses(p, y).brier.mean())


def brier_skill(p: np.ndarray, p_ref: np.ndarray, y: np.ndarray) -> float:
    """Brier skill score ``1 - B / B_ref``. ``p_ref`` is ``(n, K)`` or one ``(K,)`` distribution,
    such as climatology fitted on the training fold."""
    q, labels = _probs(p, y)
    ref = _reference(p_ref, q.shape)
    b_ref = brier(ref, labels)
    if b_ref == 0.0:
        raise ValueError("the reference forecast is perfect; the skill score is undefined")
    return 1.0 - brier(q, labels) / b_ref


def _calibration(score: np.ndarray, outcome: np.ndarray, n_bins: int, strategy: Strategy) -> ReliabilityTable:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    n = score.size
    if strategy == "width":
        idx = np.minimum((score * n_bins).astype(np.int64), n_bins - 1)
        count = np.bincount(idx, minlength=n_bins)
        s_sum = np.bincount(idx, weights=score, minlength=n_bins)
        o_sum = np.bincount(idx, weights=outcome, minlength=n_bins)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        keep = count > 0
        low, high = edges[:-1][keep], edges[1:][keep]
    elif strategy == "mass":
        order = np.argsort(score, kind="stable")
        groups = [g for g in np.array_split(order, min(n_bins, n)) if g.size]
        count = np.array([g.size for g in groups])
        s_sum = np.array([score[g].sum() for g in groups])
        o_sum = np.array([outcome[g].sum() for g in groups])
        low = np.array([score[g].min() for g in groups])
        high = np.array([score[g].max() for g in groups])
        keep = count > 0
    else:
        raise ValueError(f"strategy must be 'width' or 'mass', got {strategy!r}")
    count, s_sum, o_sum = count[keep], s_sum[keep], o_sum[keep]
    return ReliabilityTable(low, high, count, s_sum / count, o_sum / count)


def reliability_table(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: Strategy = "width",
    cls: int | None = None,
) -> ReliabilityTable:
    """Reliability diagram data.

    With ``cls=None``, the top-label view: the score is ``max_k p_k`` and the outcome is whether
    the argmax class happened. With ``cls=k``, the one-vs-rest view of class ``k``: the score is
    ``p_k`` and the outcome ``1{y = k}``.
    """
    q, labels = _probs(p, y)
    assert labels is not None
    if cls is None:
        score = q.max(axis=1)
        outcome = (q.argmax(axis=1) == labels).astype(np.float64)
    else:
        if not 0 <= cls < q.shape[1]:
            raise ValueError(f"cls must lie in [0, {q.shape[1] - 1}]")
        score = q[:, cls]
        outcome = (labels == cls).astype(np.float64)
    return _calibration(score, outcome, n_bins, strategy)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10, strategy: Strategy = "width") -> float:
    """Top-label expected calibration error ``sum_b n_b/n |acc_b - conf_b|``.

    ECE is biased upward: with finite n, the within-bin sampling noise of ``acc_b`` adds about
    ``sqrt(2 / (pi * n_b)) * sqrt(acc (1 - acc))`` per bin even for perfect calibration. Compare
    ECEs only at equal n and binning, and read them with :func:`ece_ci_bootstrap`.
    """
    return reliability_table(p, y, n_bins, strategy).ece


def classwise_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10, strategy: Strategy = "width") -> float:
    """Class-wise ECE (Kull et al. 2019): the mean over classes of the one-vs-rest ECE of ``p_k``.

    Unlike top-label ECE it sees miscalibration of the non-argmax classes, such as a model that
    is right about "flat" but wrong about the split between "down" and "up". Same upward bias.
    """
    q, labels = _probs(p, y)
    return float(np.mean([reliability_table(q, labels, n_bins, strategy, cls=k).ece for k in range(q.shape[1])]))


def ece_ci_bootstrap(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: Strategy = "width",
    kind: Literal["top", "classwise"] = "top",
    level: float = 0.95,
    n_boot: int = 1000,
    *,
    seed: Seed,
    mean_block: float | None = None,
) -> Interval:
    """Stationary-bootstrap percentile interval of the ECE.

    With ``mean_block=None`` the block length is :func:`optimal_block_length` of the per-row
    Brier loss. The interval inherits the upward bias of ECE; it measures sampling noise, not bias.
    """
    level = _level(level)
    q, labels = _probs(p, y)
    assert labels is not None
    fn = ece if kind == "top" else classwise_ece if kind == "classwise" else None
    if fn is None:
        raise ValueError(f"kind must be 'top' or 'classwise', got {kind!r}")
    b = float(_default_block(per_row_losses(q, labels).brier) if mean_block is None else mean_block)
    rng = _rng(seed)
    draws = [fn(q[i], labels[i], n_bins, strategy) for idx in _sb_chunks(q.shape[0], b, n_boot, rng) for i in idx]
    return _percentile(
        np.asarray(draws), fn(q, labels, n_bins, strategy), level, f"{kind} ECE, stationary bootstrap", b, False
    )


def _default_block(x: np.ndarray) -> int:
    v = np.asarray(x, dtype=np.float64)
    if v.size < 8 or np.all(v == v[0]):
        return 1
    return optimal_block_length(v)


def bss_ci_bootstrap(
    p: np.ndarray,
    p_ref: np.ndarray,
    y: np.ndarray,
    level: float = 0.95,
    n_boot: int = 2000,
    *,
    seed: Seed,
    mean_block: float | None = None,
) -> Interval:
    """Paired stationary-bootstrap interval of the Brier skill score ``1 - B / B_ref``.

    The per-row Brier losses of both forecasts are resampled with the same indices. With
    ``mean_block=None``, the block length is :func:`optimal_block_length` of the per-row loss
    differential. Rows must be in time order; with several assets per timestamp, aggregate the
    losses per timestamp first, or the blocks will cut across assets instead of time.
    ``p_value`` is the two-sided percentile-bootstrap p-value of H0 "BSS = 0".
    """
    level = _level(level)
    q, labels = _probs(p, y)
    assert labels is not None
    ref = _reference(p_ref, q.shape)
    b_m = per_row_losses(q, labels).brier
    b_r = per_row_losses(ref, labels).brier
    if b_r.sum() == 0.0:
        raise ValueError("the reference forecast is perfect; the skill score is undefined")
    blk = float(_default_block(b_m - b_r) if mean_block is None else mean_block)
    rng = _rng(seed)
    parts = []
    for idx in _sb_chunks(b_m.size, blk, n_boot, rng):
        den = b_r[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            parts.append(np.where(den > 0.0, 1.0 - b_m[idx].mean(axis=1) / den, np.nan))
    est = 1.0 - b_m.mean() / b_r.mean()
    return _percentile(np.concatenate(parts), est, level, "paired stationary bootstrap, percentile", blk, True)


# ---------------------------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------------------------


def _hac(d: np.ndarray, lags: int, kernel: str) -> float:
    """Long-run variance of ``d``, divisor T: ``gamma_0 + 2 sum_{k=1}^{L} w_k gamma_k`` with
    ``w_k = 1 - k/(L+1)`` (Bartlett, Newey and West 1987) or ``w_k = 1`` (rectangular)."""
    t = d.size
    c = d - d.mean()
    out = float(c @ c) / t
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0) if kernel == "bartlett" else 1.0
        out += 2.0 * w * float(c[k:] @ c[:-k]) / t
    return out


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    kernel: Literal["rectangular", "bartlett"] = "rectangular",
) -> DieboldMariano:
    """Diebold-Mariano test of equal expected loss, with the Harvey-Leybourne-Newbold correction.

    - ``d = loss_a - loss_b``; a negative statistic means ``a`` has the lower loss.
    - The variance of the mean of ``d`` is a HAC estimate with ``h - 1`` lags, divided by T. The
      default is the original DM (1995) rectangular kernel, which weights the ``h - 1``
      autocovariances by 1; ``kernel="bartlett"`` gives Newey-West weights ``1 - k/h``, which are
      positive semi-definite.
    - Fallbacks: if the rectangular variance is ``<= 0``, the test uses Bartlett with the same
      lags; if the variance is still ``<= 0``, it uses ``h = 1`` (no lags). ``note``,
      ``kernel_used`` and ``h_used`` record which one was used.
    - The statistic is multiplied by ``sqrt((T + 1 - 2h + h(h - 1)/T) / T)`` (HLN 1997) and
      compared with a Student t with ``T - 1`` degrees of freedom; the p-value is two-sided.
    - For ``h``-step forecasts with overlapping horizons, pass the horizon as ``h``.

    Size. HLN derived their correction for the rectangular kernel. For an MA(h - 1) loss
    differential, Bartlett with ``h - 1`` lags captures only part of the long-run variance (33/49
    for an equal-weight MA(6)) and over-rejects: about 9-10 % at a nominal 5 % for ``h = 7`` and
    T = 500-1000; the rectangular kernel gives about 5 % (both measured in
    ``tests/test_stats.py``). Hence the rectangular default (protocol change log, iteration 2).
    With ``h = 1`` the two are identical.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("loss series must be 1-D and of the same length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("losses contain NaN or infinite values")
    if kernel not in ("bartlett", "rectangular"):
        raise ValueError(f"kernel must be 'bartlett' or 'rectangular', got {kernel!r}")
    h = int(h)
    t = a.size
    if h < 1:
        raise ValueError("h must be >= 1")
    if t < max(3, 2 * h):
        raise ValueError(f"need at least {max(3, 2 * h)} losses for h = {h}, got {t}")
    d = a - b
    mean_d = float(d.mean())
    h_used, kernel_used, notes = h, str(kernel), []
    lrv = _hac(d, h - 1, kernel_used)
    if lrv <= 0.0 and h > 1 and kernel_used == "rectangular":
        notes.append(f"rectangular HAC variance with h = {h} was not positive; fell back to Bartlett")
        kernel_used = "bartlett"
        lrv = _hac(d, h - 1, kernel_used)
    if lrv <= 0.0 and h > 1:
        notes.append(f"Bartlett HAC variance with h = {h} was not positive; fell back to h = 1")
        h_used = 1
        lrv = _hac(d, 0, kernel_used)
    note = "; ".join(notes)
    if lrv <= 0.0:
        raise ValueError("the loss differential is constant; the Diebold-Mariano test is undefined")
    stat = mean_d / math.sqrt(lrv / t)
    stat *= math.sqrt((t + 1 - 2 * h_used + h_used * (h_used - 1) / t) / t)
    p = float(2.0 * _st.t.sf(abs(stat), df=t - 1))
    return DieboldMariano(float(stat), p, mean_d, h, h_used, t, note, kernel_used)
