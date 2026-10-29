"""The eight Matrix arms of iteration 3 (``docs/PROTOCOL.md``, "Matrix on hourly bars (iteration 3)").

One pooled memory, :class:`olsprime_rs.Matrix`, over the hourly panel of the universe members
(:mod:`olsprime.features.panel`). An entry is a complete row: x holds the 15 features, y the
one-hot class followed by ``fwd_return``. The arms read it:

==================== ==========================================================================
``climatology-panel`` class frequencies of the labelled rows known at the fit
``matrix-select``     the (window, rho) of lowest mean prequential log-loss over the fit's
                      selection rows
``matrix-1y``         the 365-day window, rho selected the same way
``softmax-1y``        Nadaraya-Watson over the 365-day window, u = x / sigma, tau selected
``knn-1y``            mean one-hot label of the k nearest entries of the 365-day window, k selected
``matrix-adaptive``   matrix-select's window; at each decision d, the rho of lowest mean
                      prequential log-loss over the rows known by d and decided after d - 30 days
``matrix-stack``      sum_w alpha_w p_w over the three windows, each at its own selected rho,
                      alpha on the simplex grid of step 0.05
``matrix-cusum``      matrix-select, refitted at the first fit and then only after a CUSUM alarm
==================== ==========================================================================

Every probability is ``p_k = max(y_k, 1e-3) / sum_j max(y_j, 1e-3)``. A readout the core refuses
(empty, singular, non-finite, too few entries) falls back to the arm's climatology; the fallbacks
are counted per arm.

Causality, by construction
--------------------------
Each output at decision time d depends only on bars final by d. The one-line argument for each
piece:

- **Entries.** Every complete row is pushed once, stamped with its ``label_known_at`` (the open of
  its exit bar). A read at ``as_of = d`` sees only entries with ``time <= d``: their features and
  their label are built from bars that opened at or before d, all final by d. So an outcome joins
  the memory only once it is known, inside the test month too.
- **Reads.** ``ridge_batch`` and ``kernel_batch`` read the window ``d - w < time <= d``; a
  centred read of ``[l, r)`` depends on ``(l, r)`` alone (``docs/DESIGN.md`` §3.1), so later pushes
  never move it, not by one bit.
- **The query** is the row's own features, built from bars up to its own bar, final at d.
- **sigma.** The sigma_j^2 of month M is the variance of feature j over the entries known by
  ``M.start - 24 h``, read from the tree as ``centred_stats(0, end)``: entries known before any
  decision of M. Every readout of a row of month M uses it, in the test months and before them.
- **The prequential record.** Every candidate (each window with each rho, each tau, each k)
  forecasts every complete row from the first month that has a sigma (2017-11), at the row's own
  decision time with the sigma of the row's month: each forecast is causal by the two points
  above. Rows before the first test month only build the record.
- **Selection rows** of a fit at F: rows decided in ``(F - 365 d, F)`` whose label is known by
  ``F - 24 h`` (purge and embargo of h). Their losses use labels known by ``F - 24 h`` and their
  forecasts were made at their own decision times, before F: every fit-time choice (window, rho,
  tau, k, alpha) is a function of bars final by ``F - 24 h``.
- **Defaults.** A selection that finds fewer than 1,000 rows takes a fixed default (365-day
  window, rho = 1e-2, tau = sqrt 15, k = 200, all of alpha on the 365-day window): a constant,
  causal trivially. They are counted per arm.
- **Climatology** at a fit F: class frequencies of the labelled rows with ``label_known_at <= F - 24 h``.
- **matrix-adaptive** at decision d: the rows with ``label_known_at <= d`` and decided after
  ``d - 30 d``, scored with the record's forecasts, made at their own (earlier) decision times.
- **matrix-stack**: the per-window rho and alpha are fit-time choices on the selection rows.
- **matrix-cusum.** Day D's value is built from the decisions of the bars opening at 00:00 UTC on
  D, with their labels; it enters the CUSUM at their exit open, ``D + 25 h`` (members whose label
  is known later are left out of that day, so the value is known then). An alarm acts at the first
  00:00 UTC after that instant, ``D + 2 days``, with a refit on rows known by the fit time minus
  24 h: every input of a refit is known before it acts. Between refits the arm keeps its sigma (of
  ``F - 24 h``), window and rho, so its readouts are its own (``ridge_batch`` with that scale).

Determinism
-----------
Every reduction behind a compared output is sequential (``np.add.accumulate``), never numpy's
pairwise ``sum``, whose SIMD peeling can depend on the memory alignment: the same inputs give the
same bits, which the panel H1 (:mod:`olsprime.bench.panel_invariance`) compares. The core's batch
reads have the same bits whatever the thread count and the other queries.

Hooks
-----
:class:`MatrixSystem` exposes the five places a leak could enter as methods
(:meth:`~MatrixSystem.entry_times`, :meth:`~MatrixSystem.ridge_as_of`,
:meth:`~MatrixSystem.kernel_as_of`, :meth:`~MatrixSystem.variance_at`,
:meth:`~MatrixSystem.selection_mask`). The planted online leaks of the panel H1 override them in
``olsprime/bench/panel_invariance.py``; nothing here imports them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

import olsprime_rs as rs
from olsprime.features.basic import FEATURE_COLUMNS

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
P = len(FEATURE_COLUMNS)  # 15
Q = 4  # one-hot class (3) and fwd_return
N_CLASSES = 3

ARMS: tuple[str, ...] = (
    "climatology-panel",
    "matrix-select",
    "matrix-1y",
    "softmax-1y",
    "knn-1y",
    "matrix-adaptive",
    "matrix-stack",
    "matrix-cusum",
)
KERNEL_ARMS = frozenset({"softmax-1y", "knn-1y"})

# The CUSUM threshold of matrix-cusum: an average run length of 365 days under independent N(0, 1)
# values, two-sided Page CUSUM with k = 0.5. Set by :func:`cusum_threshold` with CUSUM_RUNS runs
# from CUSUM_SEED, before any run on real data (iteration 3). tests/test_matrix_arms.py checks it.
CUSUM_K = 0.5
CUSUM_ARL_DAYS = 365.0
CUSUM_SEED = 20261029
CUSUM_RUNS = 1_000_000
CUSUM_THRESHOLD = 4.761  # simulated ARL 365.031 days (standard error 0.359), 10^6 runs


@dataclass(frozen=True)
class MatrixConfig:
    """Everything the arms fix in advance. Saved with every run (``dataclasses.asdict``)."""

    windows_days: tuple[int | None, ...] = (30, 365, None)  # None: all history
    rhos: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    tau_factors: tuple[float, ...] = (0.5, 1.0, 2.0)  # times sqrt(15)
    ks: tuple[int, ...] = (50, 200, 1000)
    kernel_window_days: int = 365
    prob_floor: float = 1e-3
    horizon_ms: int = 24 * HOUR_MS  # the label horizon: purge and embargo
    selection_days: int = 365
    min_selection_rows: int = 1000
    default_window: int = 1  # index into windows_days: 365 days
    default_rho: float = 1e-2
    default_tau_factor: float = 1.0
    default_k: int = 200
    default_alpha: tuple[float, float, float] = (0.0, 1.0, 0.0)  # all on the 365-day window
    adaptive_days: int = 30
    stack_grid: int = 20  # alpha in multiples of 1/20 = 0.05
    first_test: tuple[int, int] = (2019, 1)
    last_test: tuple[int, int] = (2024, 12)
    cusum_k: float = CUSUM_K
    cusum_threshold: float = CUSUM_THRESHOLD
    cusum_max_fits: int = 36
    block_log2: int = 6
    threads: int = 0  # 0: every core; the outputs do not depend on it

    def __post_init__(self) -> None:
        if self.rhos.count(self.default_rho) != 1 or self.ks.count(self.default_k) != 1:
            raise ValueError("the defaults must be on their grids")
        if self.tau_factors.count(self.default_tau_factor) != 1:
            raise ValueError("the default tau must be on its grid")
        if self.windows_days[self.default_window] != self.kernel_window_days:
            raise ValueError("the default window is the 365-day window")


@dataclass(frozen=True)
class Scope:
    """What a run computes. The default computes everything; the panel H1 limits a perturbed run.

    - ``until_ms``: only rows decided at or before it get a forecast, a fit at or before it is made.
    - ``kernel_from_ms``: the kernel candidates forecast only rows decided after it.
    - ``kernel_months``: the test months (epoch ms of their first instant) of the kernel arms.
    - ``arms``: the arms computed (their dependencies come with them).
    """

    until_ms: int | None = None
    kernel_from_ms: int | None = None
    kernel_months: tuple[int, ...] | None = None
    arms: tuple[str, ...] = ARMS


# --- Small pure helpers ---------------------------------------------------------------------


def seqsum(a: np.ndarray, axis: int = 0) -> np.ndarray | float:
    """Sum by sequential accumulation: the same inputs in the same order give the same bits."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape[axis] == 0:
        out = np.zeros(np.delete(a.shape, axis))
        return float(out) if out.ndim == 0 else out
    out = np.take(np.add.accumulate(a, axis=axis), -1, axis=axis)
    return float(out) if np.ndim(out) == 0 else out


def argmin_first(scores: np.ndarray) -> int:
    """Index of the smallest score, the first one on ties; NaN never wins."""
    s = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(s).any():
        raise ValueError("no finite score")
    return int(np.argmin(np.where(np.isnan(s), np.inf, s)))


def prob_map(y: np.ndarray, floor: float) -> np.ndarray:
    """``p_k = max(y_k, floor) / sum_j max(y_j, floor)`` on the last axis (3 classes); NaN stays NaN."""
    a, b, c = (np.maximum(y[..., k], floor) for k in range(N_CLASSES))
    s = (a + b) + c
    return np.stack([a / s, b / s, c / s], axis=-1)


def label_loss(probs: np.ndarray, label: np.ndarray) -> np.ndarray:
    """``-log p_label`` per row, for probs of shape (n, ..., 3)."""
    idx = label.reshape((-1,) + (1,) * (probs.ndim - 1))
    return -np.log(np.take_along_axis(probs, idx, axis=-1)[..., 0])


def stack_alphas(grid: int = 20) -> np.ndarray:
    """The simplex grid of step 1/grid over three windows, lexicographic in (a_30, a_365)."""
    rows = [(i, j, grid - i - j) for i in range(grid + 1) for j in range(grid + 1 - i)]
    return np.array(rows, dtype=np.float64) / float(grid)


def mix(alpha: np.ndarray, parts: Sequence[np.ndarray]) -> np.ndarray:
    """``a_0 p_0 + a_1 p_1 + a_2 p_2``, in that order, elementwise."""
    return (alpha[0] * parts[0] + alpha[1] * parts[1]) + alpha[2] * parts[2]


def standardized_residuals(fwd: np.ndarray, forecast: np.ndarray, leverage: np.ndarray, rss: np.ndarray, n_w: np.ndarray) -> np.ndarray:
    """Recursive residuals of the ``fwd_return`` readout over their predictive standard deviation.

    ``(y - yhat) / (sigma_hat sqrt(1 + 1/n_w + leverage))`` with ``sigma_hat^2 = rss / (n_w - p - 1)``,
    p = 15: the residual mean square of the window at the decision time. NaN where undefined
    (a refused cell, ``n_w <= p + 1`` or a zero rss).
    """
    dof = n_w.astype(np.float64) - (P + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(dof > 0, rss / dof, np.nan)
        sd = np.sqrt(s2) * np.sqrt(1.0 + 1.0 / n_w.astype(np.float64) + leverage)
        z = np.where(sd > 0, (fwd - forecast) / sd, np.nan)
    return z


def month_start_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)


def next_month_ms(ms: int) -> int:
    d = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return month_start_ms(d.year + (d.month == 12), d.month % 12 + 1)


def fold_months_ms(cfg: MatrixConfig) -> list[int]:
    out, m = [], month_start_ms(*cfg.first_test)
    last = month_start_ms(*cfg.last_test)
    while m <= last:
        out.append(m)
        m = next_month_ms(m)
    return out


def iso(ms: int | None) -> str | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="milliseconds")


# Every time here is a millisecond epoch of the development period. The core reads a time in another
# unit without complaint: an ``as_of`` in seconds empties every window and every row silently falls
# back to climatology (audit pass 1b). So every time is checked before it reaches the core.
TIMES_FROM = month_start_ms(2017, 1)
TIMES_UNTIL = month_start_ms(2025, 1)
NOT_COMPUTED = 255  # a record cell that was not read (outside a scope)
STATUS_NAMES = {0: "ok", 1: "empty", 2: "singular", 3: "non-finite", 4: "too-few", NOT_COMPUTED: "not-computed"}


def check_ms(name: str, times: np.ndarray, *, slack: int = 0) -> np.ndarray:
    """Fail loudly unless every time is a millisecond epoch in [2017-01-01, 2025-01-01 + slack)."""
    t = np.asarray(times)
    if t.dtype != np.int64:
        raise TypeError(f"{name}: times must be int64 milliseconds, got {t.dtype}")
    if t.size and (t.min() < TIMES_FROM or t.max() >= TIMES_UNTIL + slack):
        raise ValueError(
            f"{name}: times from {int(t.min())} to {int(t.max())} are not millisecond epochs in "
            f"[2017-01-01, 2025-01-01): a time in seconds or microseconds would empty or fill every window silently"
        )
    return times


def check_variance(var: np.ndarray, fit_ms: int) -> None:
    """Fail loudly on a fold variance of 0: the kernels refuse it, and the ridge would leave the feature unpenalized."""
    bad = np.flatnonzero(~(var > 0))
    if bad.size:
        names = ", ".join(FEATURE_COLUMNS[j] for j in bad)
        raise ValueError(f"the fold variance is 0 (or not finite) at the fit of {iso(int(fit_ms))} for {names}")


def feature_ratios(x: np.ndarray) -> dict[str, float]:
    """r_j = |mean_j| / std_j of each feature over the rows given: the size of the raw mean the refusal bound pays for."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(mean) / std
    return {name: float(v) for name, v in zip(FEATURE_COLUMNS, r, strict=True)}


# --- The CUSUM threshold --------------------------------------------------------------------


@dataclass(frozen=True)
class CusumRecords:
    """Every new running maximum of ``max(S+, S-)`` of each simulated path, until it passed ``h_max``.

    The run length at threshold h of a path is the time of its first record above h, so one
    simulation gives the run length at every ``h <= h_max``, monotone in h.
    """

    n_runs: int
    seed: int
    k: float
    h_max: float
    path: np.ndarray
    time: np.ndarray
    value: np.ndarray


def cusum_simulate(n_runs: int, seed: int, *, k: float = CUSUM_K, h_max: float = 5.5, block: int = 256) -> CusumRecords:
    """Two-sided Page CUSUMs ``S+ = max(0, S+ + z - k)``, ``S- = max(0, S- - z - k)`` on iid N(0, 1), from 0."""
    rng = np.random.default_rng(seed)
    alive = np.arange(n_runs)
    sp, sm, mx = np.zeros(n_runs), np.zeros(n_runs), np.zeros(n_runs)
    t0 = 0
    paths, times, values = [], [], []
    while alive.size:
        z = rng.standard_normal((alive.size, block))
        m = np.empty_like(z)
        p, q = sp[alive], sm[alive]
        for j in range(block):
            p = np.maximum(0.0, p + z[:, j] - k)
            q = np.maximum(0.0, q - z[:, j] - k)
            m[:, j] = np.maximum(p, q)
        cm = np.maximum.accumulate(np.concatenate([mx[alive][:, None], m], axis=1), axis=1)
        r, c = np.nonzero(cm[:, 1:] > cm[:, :-1])
        paths.append(alive[r])
        times.append(t0 + c + 1)
        values.append(cm[:, 1:][r, c])
        sp[alive], sm[alive], mx[alive] = p, q, cm[:, -1]
        alive = alive[cm[:, -1] <= h_max]
        t0 += block
    path, tm, val = np.concatenate(paths), np.concatenate(times), np.concatenate(values)
    order = np.lexsort((tm, path))
    return CusumRecords(n_runs, seed, k, h_max, path[order], tm[order], val[order])


def cusum_arl(rec: CusumRecords, h: float) -> tuple[float, float]:
    """Mean run length (alarm when ``max(S+, S-) > h``) and its standard error, for ``h <= h_max``."""
    if not h <= rec.h_max:
        raise ValueError(f"h = {h} is above the simulated h_max = {rec.h_max}")
    sel = rec.value > h
    _, first = np.unique(rec.path[sel], return_index=True)
    rl = rec.time[sel][first].astype(np.float64)
    if rl.size != rec.n_runs:
        raise AssertionError("a path never passed h")
    return float(rl.mean()), float(rl.std(ddof=1) / math.sqrt(rec.n_runs))


def cusum_threshold(target: float = CUSUM_ARL_DAYS, *, n_runs: int = CUSUM_RUNS, seed: int = CUSUM_SEED, k: float = CUSUM_K) -> dict[str, Any]:
    """The threshold, to 3 decimals, whose simulated ARL is closest to ``target``.

    Bisection on the continuous h (the simulated ARL is monotone in h), then the closer of the two
    3-decimal neighbours. Returns the threshold, its simulated ARL and standard error.
    """
    rec = cusum_simulate(n_runs, seed, k=k)
    lo, hi = 0.5, rec.h_max
    if cusum_arl(rec, hi)[0] < target:
        raise ValueError("h_max is too small for the target")
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cusum_arl(rec, mid)[0] < target else (lo, mid)
    grid = sorted({math.floor(hi * 1000) / 1000, math.ceil(hi * 1000) / 1000})
    arls = {g: cusum_arl(rec, g) for g in grid}
    h = min(grid, key=lambda g: (abs(arls[g][0] - target), g))
    return {"threshold": round(h, 3), "arl": arls[h][0], "arl_se": arls[h][1], "target": target, "k": k,
            "n_runs": n_runs, "seed": seed, "neighbours": {f"{g:.3f}": arls[g][0] for g in grid}}  # fmt: skip


def page_cusum(z: Sequence[float], k: float, h: float) -> tuple[list[float], list[float], list[bool]]:
    """The two-sided Page CUSUM on a sequence, restarting at 0 after every alarm (for tests and the report)."""
    sp = sm = 0.0
    ups, downs, alarms = [], [], []
    for v in z:
        sp, sm = max(0.0, sp + v - k), max(0.0, sm - v - k)
        alarm = sp > h or sm > h
        ups.append(sp)
        downs.append(sm)
        alarms.append(alarm)
        if alarm:
            sp = sm = 0.0
    return ups, downs, alarms


# --- The panel as arrays --------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PanelArrays:
    """The panel's columns as numpy arrays, in the panel's order ``(decision_time, symbol)``."""

    symbols: tuple[str, ...]
    symbol: np.ndarray  # int64 code into symbols (sorted names)
    open_ms: np.ndarray
    decision_ms: np.ndarray
    month_ms: np.ndarray
    lka_ms: np.ndarray  # label_known_at, -1 where null
    x: np.ndarray  # (n, 15), NaN where null
    label: np.ndarray  # int64, -1 where null
    fwd: np.ndarray
    complete: np.ndarray
    features_ok: np.ndarray
    labelled: np.ndarray

    @property
    def n(self) -> int:
        return int(self.open_ms.size)


def _ms(series: pl.Series) -> np.ndarray:
    return series.dt.epoch("ms").fill_null(-1).to_numpy().astype(np.int64)


def panel_arrays(panel: pl.DataFrame) -> PanelArrays:
    """Read the panel once into arrays; the panel must be sorted by ``(decision_time, symbol)``."""
    symbols = tuple(sorted(panel["symbol"].unique().to_list()))
    code = {s: i for i, s in enumerate(symbols)}
    sym = np.array([code[s] for s in panel["symbol"].to_list()], dtype=np.int64)
    dec = _ms(panel["decision_time"])
    if np.any(np.diff(dec) < 0):
        raise ValueError("the panel must be sorted by decision_time")
    x = np.ascontiguousarray(panel.select(FEATURE_COLUMNS).to_numpy().astype(np.float64))
    label = panel["label"].cast(pl.Int64).fill_null(-1).to_numpy().astype(np.int64)
    fwd = panel["fwd_return"].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float64)
    features_ok = np.isfinite(x).all(axis=1)
    labelled = (label >= 0) & np.isfinite(fwd) & panel["label_known_at"].is_not_null().to_numpy()
    complete = panel["complete"].to_numpy().astype(bool) & features_ok & labelled
    lka = _ms(panel["label_known_at"])
    if np.any(labelled & (lka < dec)):
        raise ValueError("a label is known before its decision")
    return PanelArrays(symbols, sym, _ms(panel["open_time"]), dec, _ms(panel["month"]), lka, x, label, fwd, complete, features_ok, labelled)


# --- Results --------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonthStats:
    """The fold statistics of one month (or of a CUSUM refit): sigma^2, climatology, entry count."""

    fit_ms: int
    var: np.ndarray | None
    clim: np.ndarray
    clim_rows: int
    entries: int


@dataclass(eq=False)
class Record:
    """The prequential record: every candidate's forecast at every complete row of a month with a sigma."""

    candidates: tuple[str, ...]
    rows: np.ndarray  # indices into the panel arrays, panel order
    month: np.ndarray
    probs: np.ndarray  # (N, C, 3)
    ok: np.ndarray  # (N, C)
    status: np.ndarray  # (N, C) uint8: the core's status of each cell, NOT_COMPUTED where it was not read
    loss: np.ndarray  # (N, C)
    kernel: np.ndarray  # (N,) the kernel candidates were computed at this row
    mid: np.ndarray  # positions (into rows) of the rows whose bar opens at 00:00 UTC
    mid_fwd: np.ndarray  # (Nm, W, R) the fwd_return forecast
    mid_lev: np.ndarray  # (Nm, W, R)
    mid_rss: np.ndarray  # (Nm, W, R) rss of fwd_return
    mid_n: np.ndarray  # (Nm, W) window counts


@dataclass(eq=False)
class ArmsResult:
    """Every arm's forecast at the test rows, with every fit-time choice behind it."""

    config: MatrixConfig
    scope: Scope
    candidates: tuple[str, ...]
    rows: np.ndarray  # indices into the panel arrays of the output rows, panel order
    probs: dict[str, np.ndarray]  # arm -> (n, 3); NaN rows were not computed (scoped runs)
    fallback: dict[str, np.ndarray]  # arm -> (n,) bool, the readout fell back to climatology
    status: dict[str, np.ndarray]  # arm -> (n,) uint8, the core's status of the cell read (NOT_COMPUTED outside the scope)
    fits: list[dict[str, Any]]  # the monthly fits, in order
    adaptive: dict[str, np.ndarray]  # per decision time of matrix-adaptive
    cusum: dict[str, Any]
    defaults: dict[str, int]
    record_fallbacks: dict[str, int]
    status_counts: dict[str, dict[str, int]] = field(default_factory=dict)  # arm -> status name -> output rows
    record_status: dict[str, dict[str, int]] = field(default_factory=dict)  # candidate -> status name -> record rows
    months: dict[int, MonthStats] = field(default_factory=dict)

    def fallback_counts(self) -> dict[str, int]:
        return {arm: int(v.sum()) for arm, v in self.fallback.items()}


# --- The system -----------------------------------------------------------------------------


class MatrixSystem:
    """The Matrix arms over one panel: the memory, the sigma schedule, the record and every fit."""

    name = "honest"

    def __init__(self, panel: pl.DataFrame | PanelArrays, cfg: MatrixConfig = MatrixConfig()) -> None:
        self.cfg = cfg
        self.data = panel if isinstance(panel, PanelArrays) else panel_arrays(panel)
        d = self.data
        self.windows = [None if w is None else int(w) * DAY_MS for w in cfg.windows_days]
        self.kernel_window = cfg.kernel_window_days * DAY_MS
        self.rhos = np.array(cfg.rhos, dtype=np.float64)
        self.taus = np.array([f * math.sqrt(P) for f in cfg.tau_factors], dtype=np.float64)
        self.ks = np.array(cfg.ks, dtype=np.int64)
        W, R = len(self.windows), len(self.rhos)
        self.n_ridge = W * R
        self.candidates = (
            *(f"ridge:{_wname(w)}:rho={r:g}" for w in cfg.windows_days for r in cfg.rhos),
            *(f"softmax:tau={f:g}*sqrt15" for f in cfg.tau_factors),
            *(f"knn:k={k}" for k in cfg.ks),
        )
        self.softmax0 = self.n_ridge
        self.knn0 = self.n_ridge + len(self.taus)
        # Entries: every complete row, pushed in the order of its stamp (ties: symbol, then open).
        check_ms("decision_time", d.decision_ms)
        rows = np.flatnonzero(d.complete)
        stamp = self.entry_times(rows)
        check_ms("entry stamp", stamp)
        order = np.lexsort((d.open_ms[rows], d.symbol[rows], stamp))
        self.entry_rows = rows[order]
        self.entry_stamp = np.ascontiguousarray(stamp[order], dtype=np.int64)
        self.matrix = rs.Matrix(P, Q, block_log2=cfg.block_log2)
        if self.entry_rows.size:
            ys = np.zeros((self.entry_rows.size, Q))
            ys[np.arange(self.entry_rows.size), d.label[self.entry_rows]] = 1.0
            ys[:, 3] = d.fwd[self.entry_rows]
            self.matrix.extend(self.entry_stamp, np.ascontiguousarray(d.x[self.entry_rows]), ys)
        # Climatology: cumulative class counts of the labelled rows in the order they become known.
        lab = np.flatnonzero(d.labelled)
        lab = lab[np.argsort(d.lka_ms[lab], kind="stable")]
        self._clim_known = d.lka_ms[lab]
        onehot = np.zeros((lab.size + 1, N_CLASSES), dtype=np.int64)
        onehot[np.arange(1, lab.size + 1), d.label[lab]] = 1
        self._clim_counts = np.cumsum(onehot, axis=0)
        self._month_cache: dict[int, MonthStats] = {}

    # --- hooks: the honest definitions (the panel H1 plants its leaks by overriding them) ---

    def entry_times(self, rows: np.ndarray) -> np.ndarray:
        """The stamp of each entry: its ``label_known_at``."""
        return self.data.lka_ms[rows]

    def ridge_as_of(self, rows: np.ndarray) -> np.ndarray:
        """The instant a ridge readout of a row reads up to: its decision time."""
        return self.data.decision_ms[rows]

    def kernel_as_of(self, rows: np.ndarray) -> np.ndarray:
        """The instant a kernel readout of a row reads up to: its decision time."""
        return self.data.decision_ms[rows]

    def variance_at(self, fit_ms: int, month_ms: int | None) -> tuple[np.ndarray | None, int]:
        """sigma^2 of a fit at ``fit_ms``: the variance (divisor n) of each feature over the entries known by ``fit_ms - h``."""
        _, end = self.matrix.window_range(int(fit_ms - self.cfg.horizon_ms), None)
        if end < 2:
            return None, int(end)
        st = self.matrix.centred_stats(0, end)
        var = np.ascontiguousarray(np.diag(st.m_xx) / st.n)
        check_variance(var, fit_ms)
        return var, int(end)

    def selection_mask(self, fit_ms: int, rec: Record) -> np.ndarray:
        """The selection rows of a fit: decided in the 365 days before it, label known by ``fit_ms - h``."""
        d = self.data
        dec, lka = d.decision_ms[rec.rows], d.lka_ms[rec.rows]
        return (dec > fit_ms - self.cfg.selection_days * DAY_MS) & (dec < fit_ms) & (lka <= fit_ms - self.cfg.horizon_ms)

    # --- fold statistics ---

    def climatology_at(self, fit_ms: int) -> tuple[np.ndarray, int]:
        """Class frequencies of the labelled rows with ``label_known_at <= fit_ms - h``, through the probability map."""
        i = int(np.searchsorted(self._clim_known, fit_ms - self.cfg.horizon_ms, side="right"))
        counts = self._clim_counts[i].astype(np.float64)
        if i == 0:
            return np.full(N_CLASSES, 1.0 / N_CLASSES), 0
        return prob_map(counts / float(i), self.cfg.prob_floor), i

    def month_stats(self, month_ms: int) -> MonthStats:
        if month_ms not in self._month_cache:
            var, n = self.variance_at(month_ms, month_ms)
            clim, n_clim = self.climatology_at(month_ms)
            self._month_cache[month_ms] = MonthStats(month_ms, var, clim, n_clim, n)
        return self._month_cache[month_ms]

    def fit_stats(self, fit_ms: int) -> MonthStats:
        """The statistics of a refit at any instant (matrix-cusum): sigma^2 and climatology at ``fit_ms``."""
        var, n = self.variance_at(fit_ms, None)
        clim, n_clim = self.climatology_at(fit_ms)
        return MonthStats(fit_ms, var, clim, n_clim, n)

    # --- readouts ---

    def _ridge(self, rows: np.ndarray, as_of: np.ndarray, var: np.ndarray, windows: list[int | None], rhos: np.ndarray, *, extras: bool) -> dict[str, np.ndarray]:
        return self.matrix.ridge_batch(
            np.ascontiguousarray(self.data.x[rows]), check_ms("ridge as_of", np.ascontiguousarray(as_of, dtype=np.int64), slack=DAY_MS), windows,
            np.ascontiguousarray(rhos, dtype=np.float64), var, leverage=extras, rss=extras, threads=self.cfg.threads,
        )  # fmt: skip

    def _kernel(self, rows: np.ndarray, var: np.ndarray) -> dict[str, np.ndarray]:
        return self.matrix.kernel_batch(
            np.ascontiguousarray(self.data.x[rows]), check_ms("kernel as_of", np.ascontiguousarray(self.kernel_as_of(rows), dtype=np.int64), slack=DAY_MS),
            self.kernel_window, np.sqrt(var), self.taus, self.ks, threads=self.cfg.threads,
        )  # fmt: skip

    def record(self, scope: Scope = Scope(), *, kernels: bool = True) -> Record:
        """Every candidate at every complete row of every month with a sigma (decided by ``scope.until_ms``)."""
        d, cfg = self.data, self.cfg
        W, R, C = len(self.windows), len(self.rhos), len(self.candidates)
        q_mask = d.complete.copy()
        if scope.until_ms is not None:
            q_mask &= d.decision_ms <= scope.until_ms
        months = sorted({int(m) for m in np.unique(d.month_ms[q_mask])})
        months = [m for m in months if self.month_stats(m).var is not None]
        rows = np.flatnonzero(q_mask & np.isin(d.month_ms, months))
        n = rows.size
        probs = np.full((n, C, N_CLASSES), np.nan)
        ok = np.zeros((n, C), dtype=bool)
        status = np.full((n, C), NOT_COMPUTED, dtype=np.uint8)
        kern = np.zeros(n, dtype=bool)
        mid = np.flatnonzero(d.open_ms[rows] % DAY_MS == 0)
        nm = mid.size
        mid_fwd, mid_lev, mid_rss = np.full((nm, W, R), np.nan), np.full((nm, W, R), np.nan), np.full((nm, W, R), np.nan)
        mid_n = np.zeros((nm, W), dtype=np.int64)
        pos_of_mid = np.full(n, -1, dtype=np.int64)
        pos_of_mid[mid] = np.arange(nm)
        month_of = d.month_ms[rows]
        for m in months:
            st = self.month_stats(m)
            at = np.flatnonzero(month_of == m)
            if not at.size:
                continue
            r = rows[at]
            out = self._ridge(r, self.ridge_as_of(r), st.var, self.windows, self.rhos, extras=False)
            fc = out["forecast"].reshape(at.size, W * R, Q)
            code = out["status"].reshape(at.size, W * R)
            good = code == rs.STATUS_OK
            p = prob_map(fc[..., :N_CLASSES], cfg.prob_floor)
            p[~good] = st.clim
            probs[at, : W * R], ok[at, : W * R], status[at, : W * R] = p, good, code
            m_at = at[pos_of_mid[at] >= 0]
            if m_at.size:
                ex = self._ridge(rows[m_at], self.ridge_as_of(rows[m_at]), st.var, self.windows, self.rhos, extras=True)
                j = pos_of_mid[m_at]
                mid_fwd[j], mid_lev[j], mid_rss[j] = ex["forecast"][..., 3], ex["leverage"], ex["rss"][..., 3]
                mid_n[j] = ex["count"]
            if not kernels:
                continue
            k_at = at if scope.kernel_from_ms is None else at[d.decision_ms[r] > scope.kernel_from_ms]
            if not k_at.size:
                continue
            kern[k_at] = True
            ko = self._kernel(rows[k_at], st.var)
            for first, key, status_key in ((self.softmax0, "softmax", "softmax_status"), (self.knn0, "knn", "knn_status")):
                vals, good = ko[key], ko[status_key] == rs.STATUS_OK
                p = prob_map(vals[..., :N_CLASSES], cfg.prob_floor)
                p[~good] = st.clim
                width = vals.shape[1]
                probs[k_at, first : first + width], ok[k_at, first : first + width] = p, good
                status[k_at, first : first + width] = ko[status_key]
        loss = np.full((n, C), np.nan)
        lab = d.label[rows]
        computed = np.isfinite(probs[:, :, 0])
        loss[computed] = label_loss(probs, lab)[computed]
        return Record(self.candidates, rows, month_of, probs, ok, status, loss, kern, mid, mid_fwd, mid_lev, mid_rss, mid_n)

    # --- fits ---

    def _scores(self, rec: Record, sel: np.ndarray) -> np.ndarray:
        n = int(sel.sum())
        if n == 0:
            return np.full(len(self.candidates), np.nan)
        return seqsum(rec.loss[sel], axis=0) / float(n)

    def monthly_fit(self, rec: Record, month_ms: int, *, kernels: bool) -> dict[str, Any]:
        """Every fit-time choice of the monthly arms at the first instant of a test month."""
        cfg = self.cfg
        W, R = len(self.windows), len(self.rhos)
        st = self.month_stats(month_ms)
        sel = self.selection_mask(month_ms, rec)
        n_sel = int(sel.sum())
        default = n_sel < cfg.min_selection_rows
        if kernels and not rec.kernel[sel].all():
            raise ValueError(f"the kernel record does not cover the selection rows of {iso(month_ms)}")
        scores = self._scores(rec, sel)
        if not kernels:
            scores[self.n_ridge :] = np.nan
        rho_d = cfg.rhos.index(cfg.default_rho)
        if default:
            w_sel, r_sel = cfg.default_window, rho_d
            r_1y, r_w = rho_d, [rho_d] * W
            tau, k = cfg.tau_factors.index(cfg.default_tau_factor), cfg.ks.index(cfg.default_k)
            alpha = np.array(cfg.default_alpha)
            alpha_index, alpha_scores = None, None
        else:
            best = argmin_first(scores[: self.n_ridge])
            w_sel, r_sel = divmod(best, R)
            r_w = [argmin_first(scores[w * R : (w + 1) * R]) for w in range(W)]
            r_1y = r_w[cfg.default_window]
            tau = argmin_first(scores[self.softmax0 : self.knn0]) if kernels else -1
            k = argmin_first(scores[self.knn0 :]) if kernels else -1
            alphas = stack_alphas(cfg.stack_grid)
            parts = [rec.probs[sel, w * R + r_w[w], :] for w in range(W)]
            lab = self.data.label[rec.rows[sel]]
            labp = [np.take_along_axis(p, lab[:, None], axis=1)[:, 0] for p in parts]
            alpha_scores = np.array([seqsum(-np.log(mix(a, labp))) / float(n_sel) for a in alphas])
            alpha_index = argmin_first(alpha_scores)
            alpha = alphas[alpha_index]
        return {
            "month_ms": month_ms,
            "n_selection": n_sel,
            "default": bool(default),
            "scores": scores,
            "select": (int(w_sel), int(r_sel)),
            "rho_by_window": tuple(int(r) for r in r_w),
            "rho_1y": int(r_1y),
            "tau": int(tau),
            "k": int(k),
            "alpha": alpha,
            "alpha_index": alpha_index,
            "alpha_scores": alpha_scores,
            "var": st.var,
            "clim": st.clim,
            "clim_rows": st.clim_rows,
            "entries": st.entries,
        }

    def adaptive_choices(self, rec: Record, month_ms: int, window: int, decisions: np.ndarray) -> dict[str, np.ndarray]:
        """matrix-adaptive's rho at each decision time d of a month, from the record of ``window``.

        The rows scored at d are those with ``label_known_at <= d`` and decided after ``d - 30 d``:
        ``{i : lka_i <= d < dec_i + 30 d}``. Their loss sums come from two prefix sums over the rows
        that can be in such a set for some d of the month (``lka <= next month + 1 day`` and
        ``max(lka, dec + 30 d) > month start``): sorted by ``lka`` (rows already known) and by
        ``max(lka, dec + 30 d)`` (rows already out). A prefix up to d holds only rows known by d.
        """
        cfg = self.cfg
        R = len(self.rhos)
        d = self.data
        span = cfg.adaptive_days * DAY_MS
        lka, dec = d.lka_ms[rec.rows], d.decision_ms[rec.rows]
        exit_ = np.maximum(lka, dec + span)
        pool = np.flatnonzero((lka <= next_month_ms(month_ms) + DAY_MS) & (exit_ > month_ms))
        loss = rec.loss[:, window * R : (window + 1) * R]
        oa = pool[np.argsort(lka[pool], kind="stable")]
        ob = pool[np.argsort(exit_[pool], kind="stable")]
        zero = np.zeros((1, R))
        ca = np.concatenate([zero, np.add.accumulate(loss[oa], axis=0)]) if oa.size else zero
        cb = np.concatenate([zero, np.add.accumulate(loss[ob], axis=0)]) if ob.size else zero
        ia = np.searchsorted(lka[oa], decisions, side="right")
        ib = np.searchsorted(exit_[ob], decisions, side="right")
        count = (ia - ib).astype(np.int64)
        sums = ca[ia] - cb[ib]
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = sums / count[:, None].astype(np.float64)
        default = count < cfg.min_selection_rows
        choice = np.full(decisions.size, cfg.rhos.index(cfg.default_rho), dtype=np.int64)
        for i in np.flatnonzero(~default):
            choice[i] = argmin_first(scores[i])
        return {"decision_ms": decisions, "count": count, "scores": scores, "default": default, "choice": choice}

    def _cusum_scale(self, rec: Record, sel: np.ndarray, w: int, r: int) -> tuple[float, int, bool]:
        """s: the standard deviation (ddof 1) of the raw daily values over the days of the selection rows, at (w, rho)."""
        d = self.data
        mid_sel = sel[rec.mid]
        j = np.flatnonzero(mid_sel)
        rows = rec.rows[rec.mid[j]]
        known = d.lka_ms[rows] <= d.open_ms[rows] + DAY_MS + HOUR_MS
        j, rows = j[known], rows[known]
        z = standardized_residuals(d.fwd[rows], rec.mid_fwd[j, w, r], rec.mid_lev[j, w, r], rec.mid_rss[j, w, r], rec.mid_n[j, w])
        days = d.open_ms[rows]
        raw = []
        for day in np.unique(days):
            v = z[(days == day) & np.isfinite(z)]
            if v.size:
                raw.append(seqsum(v) / float(v.size))
        if len(raw) < 2:
            return 1.0, len(raw), True
        a = np.array(raw)
        mean = seqsum(a) / float(a.size)
        s = math.sqrt(seqsum((a - mean) ** 2) / float(a.size - 1))
        if not s > 0:
            return 1.0, len(raw), True
        return s, len(raw), False

    def cusum_fit(self, rec: Record, fit_ms: int) -> dict[str, Any]:
        cfg = self.cfg
        R = len(self.rhos)
        st = self.fit_stats(fit_ms)
        sel = self.selection_mask(fit_ms, rec)
        n_sel = int(sel.sum())
        scores = self._scores(rec, sel)[: self.n_ridge]
        default = n_sel < cfg.min_selection_rows
        if default:
            w, r = cfg.default_window, cfg.rhos.index(cfg.default_rho)
        else:
            w, r = divmod(argmin_first(scores), R)
        s, s_days, s_default = self._cusum_scale(rec, sel, w, r)
        if st.var is None:
            raise ValueError(f"no sigma at the CUSUM fit {iso(fit_ms)}")
        return {"fit_ms": int(fit_ms), "window": int(w), "rho": int(r), "n_selection": n_sel, "default": bool(default),
                "scores": scores, "var": st.var, "clim": st.clim, "s": s, "s_days": s_days, "s_default": s_default}  # fmt: skip

    def _cusum_readout(self, fit: dict[str, Any], rows: np.ndarray, *, extras: bool) -> dict[str, np.ndarray]:
        out = self._ridge(rows, self.ridge_as_of(rows), fit["var"], [self.windows[fit["window"]]], self.rhos[[fit["rho"]]], extras=extras)
        code = out["status"][:, 0, 0]
        res = {"forecast": out["forecast"][:, 0, 0, :], "ok": code == rs.STATUS_OK, "status": code, "count": out["count"][:, 0]}
        if extras:
            res["leverage"], res["rss"] = out["leverage"][:, 0, 0], out["rss"][:, 0, 0, 3]
        return res

    def run_cusum(self, rec: Record, out_pos: np.ndarray, scope: Scope) -> dict[str, Any]:
        """matrix-cusum over the test months: its fits, its daily CUSUM and its forecasts at ``out_pos`` (record positions)."""
        cfg, d = self.cfg, self.data
        months = fold_months_ms(cfg)
        first, end = months[0], next_month_ms(months[-1])
        until = scope.until_ms if scope.until_ms is not None else np.iinfo(np.int64).max
        rows = rec.rows[out_pos]
        mid_rows = out_pos[d.open_ms[rows] % DAY_MS == 0]
        days = np.unique(d.open_ms[rec.rows[mid_rows]])
        fits = [self.cusum_fit(rec, first)]
        sp = sm = 0.0
        path: dict[str, list] = {k: [] for k in ("day_ms", "enters_ms", "members", "raw", "s", "z", "up", "down", "alarm", "fit")}
        skipped: list[int] = []  # days decided before a refit that acted before they entered: left out of the restarted CUSUM
        mid_open = d.open_ms[rec.rows[mid_rows]]
        for day in days:
            enters = int(day + DAY_MS + HOUR_MS)
            if enters > until:
                break
            # The fit in force at the day's decisions makes all of its value: readouts, sigma-hat, leverage and s.
            fit_i = max(i for i, f in enumerate(fits) if f["fit_ms"] <= day)
            if fit_i < len(fits) - 1:  # a refit acted after this day's decisions: the restarted CUSUM starts after it
                skipped.append(int(day))
                continue
            pos = mid_rows[mid_open == day]
            r = rec.rows[pos]
            r = r[d.lka_ms[r] <= enters]
            if not r.size:
                continue
            ro = self._cusum_readout(fits[fit_i], r, extras=True)
            z = standardized_residuals(d.fwd[r], ro["forecast"][:, 3], ro["leverage"], ro["rss"], ro["count"])
            z = z[np.isfinite(z)]
            if not z.size:
                continue
            raw = seqsum(z) / float(z.size)
            s = fits[fit_i]["s"]
            v = raw / s
            sp, sm = max(0.0, sp + v - cfg.cusum_k), max(0.0, sm - v - cfg.cusum_k)
            alarm = sp > cfg.cusum_threshold or sm > cfg.cusum_threshold
            for key, val in zip(path, (int(day), enters, int(z.size), raw, s, v, sp, sm, bool(alarm), fit_i), strict=True):
                path[key].append(val)
            if alarm:
                acts = (enters // DAY_MS + 1) * DAY_MS  # the first 00:00 UTC after the value is known
                if acts < end and acts <= until:
                    fits.append(self.cusum_fit(rec, acts))
                sp = sm = 0.0
        # Forecasts: each output row under the fit in force at its decision time.
        probs = np.full((out_pos.size, N_CLASSES), np.nan)
        status = np.full(out_pos.size, NOT_COMPUTED, dtype=np.uint8)
        dec = d.decision_ms[rows]
        times = np.array([f["fit_ms"] for f in fits], dtype=np.int64)
        which = np.searchsorted(times, dec, side="right") - 1
        for i, fit in enumerate(fits):
            at = np.flatnonzero((which == i) & (dec <= until))
            if not at.size:
                continue
            ro = self._cusum_readout(fit, rows[at], extras=False)
            p = prob_map(ro["forecast"][:, :N_CLASSES], cfg.prob_floor)
            p[~ro["ok"]] = fit["clim"]
            probs[at], status[at] = p, ro["status"]
        days_arr = {k: np.array(v) for k, v in path.items()}
        return {"fits": fits, "path": days_arr, "skipped_days": np.array(skipped, dtype=np.int64), "probs": probs, "status": status}

    # --- everything ---

    def run(self, scope: Scope = Scope()) -> ArmsResult:
        cfg, d = self.cfg, self.data
        arms = tuple(a for a in ARMS if a in scope.arms)
        unknown = set(scope.arms) - set(ARMS)
        if unknown:
            raise ValueError(f"unknown arms {sorted(unknown)}")
        kernels = bool(KERNEL_ARMS & set(arms))
        rec = self.record(scope, kernels=kernels)
        tests = fold_months_ms(cfg)
        test_set = set(tests)
        out_pos = np.flatnonzero(np.isin(rec.month, tests))
        rows = rec.rows[out_pos]
        n = out_pos.size
        probs = {a: np.full((n, N_CLASSES), np.nan) for a in arms}
        status = {a: np.full(n, NOT_COMPUTED, dtype=np.uint8) for a in arms}  # the core's status of the cell each output read
        R = len(self.rhos)
        fits: list[dict[str, Any]] = []
        adaptive: dict[str, list] = {k: [] for k in ("decision_ms", "count", "scores", "default", "choice")}
        kernel_months = set(scope.kernel_months) if scope.kernel_months is not None else test_set
        month_of = rec.month[out_pos]
        monthly = [a for a in arms if a != "matrix-cusum"]
        for m in tests:
            if scope.until_ms is not None and m > scope.until_ms:
                break
            at = np.flatnonzero(month_of == m)
            if not at.size or not monthly:
                continue
            use_k = kernels and m in kernel_months
            fit = self.monthly_fit(rec, m, kernels=use_k)
            fits.append(fit)
            pos = out_pos[at]

            def take(arm: str, c: int | np.ndarray, at: np.ndarray = at, pos: np.ndarray = pos) -> None:
                probs[arm][at], status[arm][at] = rec.probs[pos, c], rec.status[pos, c]

            if "climatology-panel" in arms:
                probs["climatology-panel"][at], status["climatology-panel"][at] = fit["clim"], rs.STATUS_OK
            if "matrix-select" in arms:
                w, r = fit["select"]
                take("matrix-select", w * R + r)
            if "matrix-1y" in arms:
                take("matrix-1y", cfg.default_window * R + fit["rho_1y"])
            if use_k and "softmax-1y" in arms:
                take("softmax-1y", self.softmax0 + fit["tau"])
            if use_k and "knn-1y" in arms:
                take("knn-1y", self.knn0 + fit["k"])
            if "matrix-stack" in arms:
                alpha = fit["alpha"]
                cells = [w * R + fit["rho_by_window"][w] for w in range(len(self.windows))]
                probs["matrix-stack"][at] = mix(alpha, [rec.probs[pos, c] for c in cells])
                # a row counts under the worst status among the windows it weighs (0 when every weighted cell is ok)
                codes = [np.where(alpha[w] > 0, rec.status[pos, c], 0) for w, c in enumerate(cells)]
                status["matrix-stack"][at] = np.maximum.reduce(codes)
            if "matrix-adaptive" in arms:
                w = fit["select"][0]
                dec = d.decision_ms[rows[at]]
                uniq, inv = np.unique(dec, return_inverse=True)
                ch = self.adaptive_choices(rec, m, w, uniq)
                for key in adaptive:
                    adaptive[key].append(ch[key])
                take("matrix-adaptive", w * R + ch["choice"][inv])
        cusum: dict[str, Any] = {}
        if "matrix-cusum" in arms:
            cusum = self.run_cusum(rec, out_pos, scope)
            probs["matrix-cusum"], status["matrix-cusum"] = cusum.pop("probs"), cusum.pop("status")
        computed_out = {a: np.isfinite(p[:, 0]) for a, p in probs.items()}
        fallback = {a: computed_out[a] & (status[a] != rs.STATUS_OK) for a in arms}
        status_counts = {
            a: {STATUS_NAMES[int(code)]: int((status[a][computed_out[a]] == code).sum()) for code in STATUS_NAMES if code != NOT_COMPUTED}
            for a in arms
        }
        adaptive_arr = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in adaptive.items()}
        n_default_fits = int(sum(f["default"] for f in fits))
        defaults = {a: n_default_fits for a in arms if a not in ("climatology-panel", "matrix-adaptive", "matrix-cusum")}
        if "matrix-adaptive" in arms:
            defaults["matrix-adaptive"] = int(adaptive_arr["default"].sum()) if adaptive_arr["default"].size else 0
            defaults["matrix-adaptive (window, at the fits)"] = n_default_fits
        if "matrix-cusum" in arms:
            defaults["matrix-cusum"] = int(sum(f["default"] for f in cusum["fits"]))
        record_fallbacks = {c: int(((rec.status[:, i] != NOT_COMPUTED) & ~rec.ok[:, i]).sum()) for i, c in enumerate(self.candidates)}
        record_status = {
            c: {STATUS_NAMES[code]: int((rec.status[:, i] == code).sum()) for code in STATUS_NAMES if code not in (0, NOT_COMPUTED)}
            for i, c in enumerate(self.candidates)
        }
        self.last_record = rec
        return ArmsResult(cfg, scope, self.candidates, rows, probs, fallback, status, fits, adaptive_arr, cusum, defaults,
                          record_fallbacks, status_counts, record_status, dict(self._month_cache))  # fmt: skip


def _wname(w: int | None) -> str:
    return "all" if w is None else f"{w}d"


def run_arms(panel: pl.DataFrame, cfg: MatrixConfig = MatrixConfig(), scope: Scope = Scope()) -> ArmsResult:
    """The honest arms on a panel."""
    return MatrixSystem(panel, cfg).run(scope)
