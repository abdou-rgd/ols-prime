"""``why``: why is a Matrix arm's forecast at one row what it is (``docs/PROTOCOL.md``, "``olsprime.explain``").

The panel and the memory are rebuilt from the store exactly as ``olsprime.bench.report2`` builds them
(:func:`olsprime.features.panel.hourly_panel`, then :class:`olsprime.models.matrix.MatrixSystem`), and the
panel's fingerprint must be the run's. Every fit-time choice is read from the run's ``report2.json``;
nothing it fixes is refitted. The one choice it does not hold, ``matrix-adaptive``'s rho at each decision,
is rebuilt from the candidates' forecasts of the rows it scores, with the arm's own code
(:meth:`MatrixSystem.adaptive_choices`), and the forecast check then holds it to the bit.

What a ridge cell shows (``matrix-select``, ``matrix-1y``, ``matrix-adaptive``, ``matrix-cusum``, and each
window of ``matrix-stack``):

- **the cell**: the raw readout (3 classes and ``fwd_return``), its probabilities, its status, and the
  choices in force (window, rho, sigma's month, the fit they come from);
- **the tree view**: the exact contributions (``centred_contribution``) of the entries grouped by their
  stamp, ``label_known_at``, by year, month or day. Sorted by |c|, top k, the rest summed, and a
  closing residual: the parts printed, then the residual, summed left to right give the raw readout.
  The residual is what the group contributions, computed apart, miss the readout by; it must lie within
  a rounding bound derived below (:func:`ridge_closing_bound`), or the call fails loudly. The grouping is
  by stamp, not by decision time. Its partition is checked apart from its sum
  (:meth:`Explainer.check_partition`): the groups are recounted from the panel's ``label_known_at``
  column, and each group's contribution is recomputed as sum w_i y_i within :func:`partition_bound`;
- **the Matrix-free twin** (:meth:`Explainer.twin`): the forecast rebuilt from panel rows alone with
  ``numpy.linalg``, held to the core's within :func:`twin_bound`;
- **the per-feature view** from ``beta`` and ``mean_y``: ``mean_y`` then ``beta_j * xt_j``, and whether their
  sum in the core's order (products rounded one by one, j = 0, 1, ... in order, ``mean_y`` last) gives the
  raw readout to the bit, as a fact checked at this call;
- **the top-k entries by |w|** with pair, decision time, stamp, label, ``fwd_return``, weight and the
  feature deltas ``x - x_q``;
- **the diagnostics**: sum w, sum |w|, the negative mass, the effective sample size 1 / sum w^2, the
  leverage and n in the window.

The kernel arms (``softmax-1y``, ``knn-1y``) take their forecast from the core, for the bit check; their
weights are recomputed in numpy (distances in the core's order, so kNN's are the core's bits), and they
have no per-feature view. The gap between the numpy readout and the core's is reported apart, with its
own bound (:func:`kernel_gap_bound`); the tree's closing residual then holds that gap plus rounding.

**The two loud checks.** A failure raises :class:`LoudCheckFailed`: the call stops, nothing is printed as
a result.

1. **Stamps and fold statistics** (:data:`CHECK1_COVERS`, :data:`CHECK1_LEAVES`). Every entry of every
   weighted window has its ``label_known_at`` recomputed from its panel row (its bar's open + 25 h, which
   must equal the panel's column), and for the top k also from the raw bars' ``open_time`` through the
   sealed loader; each must be at or before the row's decision time. Matrix's own stamps are never used.
   The sigma^2 and the climatology in force are recomputed from panel rows and held to the protocol's
   rule, bit for bit: the entries with a recomputed stamp at or before ``F - 24 h`` (F the fit time) must
   be a prefix of the memory and give sigma^2; the labelled rows with a recomputed stamp at or before
   ``F - 24 h`` give the climatology. ``matrix-adaptive`` also checks the rows behind its rho, and the
   sigma^2 and climatology of their months.
2. The probability must equal the arm's probability for that row in ``series.npz``, to the bit. A row that
   is not scored has no such check, and the output says so.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import olsprime_rs as rs
import polars as pl

from olsprime.data.binance_vision import Month
from olsprime.explain.common import LoudCheckFailed, Refused, RunFolder, month_label
from olsprime.features.basic import FEATURE_COLUMNS
from olsprime.features.panel import HORIZON, INTERVAL, fingerprint, hourly_panel
from olsprime.models.matrix import (
    DAY_MS,
    HOUR_MS,
    N_CLASSES,
    STATUS_NAMES,
    MatrixSystem,
    MonthStats,
    Record,
    iso,
    label_loss,
    mix,
    month_start_ms,
    next_month_ms,
    prob_map,
)
from olsprime.protocol import DEVELOPMENT, load_klines

OUTPUTS = ("down", "neutral", "up", "fwd_return")
RIDGE_ARMS = ("matrix-select", "matrix-1y", "matrix-adaptive", "matrix-cusum")
KERNEL_ARMS = ("softmax-1y", "knn-1y")
WHY_ARMS = (*RIDGE_ARMS, "matrix-stack", *KERNEL_ARMS)
GROUPINGS = ("year", "month", "day")
LABEL_LAG_MS = (1 + HORIZON) * HOUR_MS  # label_known_at = the exit bar's open = the row's bar open + 25 h
GROUPING_NOTE = "entries are grouped by their stamp, label_known_at (when the label became known), not by their decision time"
DEFAULTS_NOTE = ("no fit in force at as_of (it is before the run's first fit): the arm makes no forecast here, and the "
                 "candidate shown is the one the registered defaults name (365-day window, rho 1e-2, tau sqrt 15, k 200, "
                 "alpha all on the 365-day window), read with the sigma of the row's month, as the prequential record reads it")  # fmt: skip
ADAPTIVE_DEFAULT_NOTE = ("no fit in force at as_of: the registered default rho on the default window, as Report n°2's run makes no "
                         "matrix-adaptive forecast before the first fit; no pool is read")  # fmt: skip
CHECKS = ("both", "stamps", "none")  # "stamps" and "none" are for tests below the command line (H1 on the explainer)
U = 2.0**-53  # the unit roundoff of float64
CHECK1_COVERS = (
    ("every entry of every weighted window (for matrix-stack, the windows with alpha > 0): label_known_at recomputed from its panel row "
     "(bar open + 25 h), equal to the panel's column, at or before as_of"),
    "the top-k entries: their exit bar found in the raw bars through the sealed loader, opening at or before as_of",
    ("sigma^2 in force: the entries with a recomputed stamp at or before F - 24 h form a prefix of the memory, their count is the "
    "one the arm used, and their variance equals sigma^2 to the bit"),
    "the climatology in force: the labelled rows with a recomputed stamp at or before F - 24 h, counted and mapped, equal it to the bit",
    "matrix-adaptive: the rows behind its rho (recomputed stamps at or before as_of), and the sigma^2 and climatology of their months",
)
CHECK1_LEAVES = (
    "the features of the row and of each entry (built by the panel from bars up to each decision: the panel H1 tests them)",
    "the fit-time choices read from report2.json (window, rho, tau, k, alpha, the cusum fits): made by the run, not refitted here",
    "the windows of the forecasts behind matrix-adaptive's rho: recomputed with the arm's code, their stamps not re-checked here",
    ("that no entry known by as_of is missing from a window: not check 1's. For the ridge arms the Matrix-free twin checks it (row count "
     "and forecast, from panel rows alone); for the kernel arms nothing does"),
    ("the values of the entries behind sigma^2: check 1 holds their range to the panel's stamps, not their values, and the twin reads "
     "sigma^2 as Matrix gives it, so an entry corrupted behind sigma^2 but outside the window would move both sides alike"),
)


def check_extension() -> None:
    """Refuse a stale ``olsprime_rs``: the per-feature view needs ``CentredReadout.beta`` and ``.mean_y``."""
    cls = getattr(rs, "CentredReadout", None)
    missing = [name for name in ("beta", "mean_y", "mean_x", "a") if cls is None or not hasattr(cls, name)]
    if missing:
        raise Refused(f"the installed olsprime_rs has no CentredReadout.{', .'.join(missing)}: it predates explain; "
                      "rebuild the extension (uv sync) before calling why or sample")  # fmt: skip


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(_bits(a), _bits(b)))


def _wname(days: int | None) -> str:
    return "all" if days is None else f"{days}d"


def seq(values: list[float]) -> float:
    """Left-to-right sum of Python floats, from the first."""
    if not values:
        return 0.0
    acc = values[0]
    for v in values[1:]:
        acc += v
    return acc


def close_parts(parts: list[float], target: float, bound: float, what: str) -> list[float]:
    """The closing residual: terms that, added after ``parts`` left to right, give ``target``; loud beyond ``bound``.

    Closing is true by construction; the check is that the residual's size, ``sum |terms|``, is within the
    rounding bound derived for the parts (:func:`ridge_closing_bound`, :func:`kernel_closing_bound`).
    """
    acc, extra = seq(parts), []
    for _ in range(4):
        if acc == target:
            break
        res = target - acc
        extra.append(res)
        acc += res
    size = math.fsum(abs(x) for x in extra)
    if acc != target or not size <= bound:
        raise LoudCheckFailed(f"tree view, {what}: the groups miss the raw readout {target!r} by {size:.3g}, beyond the rounding "
                              f"bound {bound:.3g}: the parts do not partition the readout")  # fmt: skip
    return extra


def ridge_closing_bound(n: int, groups: int, p: int, scale: np.ndarray, readout_terms: np.ndarray, solve: np.ndarray,
                        sum_abs_c: np.ndarray) -> np.ndarray:  # fmt: skip
    """Per output c, a first-order bound on |raw_c - sum_g c_g| for a ridge cell (u = 2^-53):

        b_c = 4u [ (n + G + p + 8) (T_c + sum_g |c_g|) + (p + 2) R_c + 2 (3p + 1) S_c ].

    Derivation. In exact arithmetic sum_g c_g = ybar + M_yx a = ybar + beta xt = raw (the centred identity,
    ``centred_contribution``). The computed values differ from it in four ways:

    - *the statistics.* The window's (ybar, xbar, M_yx) and each group's (n_g, ybar_g, xbar_g, M_yx,g) are
      sums over at most n entries (scans, then tree merges), each with error at most about n u times the sum
      of the magnitudes of its terms. Every term that enters raw or a c_g, once multiplied by a, is bounded
      by T_c = m_c (2 + 8 A1), with m_c = max_i |y_ic| over the window and A1 = sum_i |x_i - xbar|^T |a|:
      |ybar| <= m_c, |M_yx,c a| <= 2 m_c A1, and over the groups sum_g |M_yx,g,c| |a| <= 4 m_c A1 and
      sum_g n_g |ybar_g,c| |xbar_g - xbar|^T |a| <= m_c A1;
    - *the arithmetic of each c_g* (p + 4 operations) and of the sums over the groups (the groups, the
      rest, at most G terms): (p + G + 8) u sum_g |c_g|;
    - *the raw readout* ybar + sum_j beta_cj xt_j, products rounded one by one: (p + 2) u R_c with
      R_c = |ybar_c| + sum_j |beta_cj xt_j|;
    - *the two solves.* beta and a come from the same Cholesky factor L, each solve exact for A + E with
      |E| <= gamma_{3p+1} |L||L^T| (Higham, Thm 10.4); beta xt - M_yx a = beta (E_2 - E_1) a, so its size is at
      most 2 (3p + 1) u S_c with S_c = |beta_c| |L||L^T| |a|, read here with numpy's factor of A.

    The factor 4 covers the second-order terms (gamma_k = k u / (1 - k u), and k u < 1e-6 here) and the
    constants folded above. It is a bound to first order, not a proof: a residual beyond it means the
    groups do not partition the readout, and the call fails loudly.
    """
    return 4.0 * U * ((n + groups + p + 8) * (scale + sum_abs_c) + (p + 2) * readout_terms + 2 * (3 * p + 1) * solve)


def kernel_gap_bound(kind: str, w: np.ndarray, ys: np.ndarray, readout: np.ndarray, dist: np.ndarray, dist_scale: np.ndarray,
                     *, p: int, inv: float = 0.0, k: int = 0) -> np.ndarray:  # fmt: skip
    """Per output c, a first-order bound on |numpy readout_c - core readout_c| for a kernel cell.

    Both sides compute the distances in the same order, d_i = sum_j (u_q,j - u_i,j)^2 with u = x / sigma;
    each is within e_i = 2 (p + 6) u D_i of the exact one, with D_i = sum_j (|u_q,j| + |u_i,j|)^2.

    - **softmax**, w_i = exp(-(d_i - d_min) inv) normalized, inv = 1 / (2 tau^2). The core keeps a running
      minimum over tiles of 128 entries and rescales its sums when it drops (at most n / 128 + 2 times),
      with its own exp (within 4 eps of libm). Each weight then has a relative error of at most
      delta_i = u [8 (R + 2) + (2 R + 6) x_i] + (e_i + e_min) inv, with x_i = (d_i - d_min) inv and R the
      number of tiles; normalized, the readout moves by at most sum_i w_i delta_i (|y_ic| + |F_c|), plus the
      sums, (n + 2) u sum_i w_i |y_ic|, and the division, 2 u |F_c|. Both sides: twice that.
    - **kNN**, the mean of the k nearest by (d_i, i). An entry whose distance is within 2 (e_i + e_k) of the
      k-th's may be in one side's k nearest and not in the other's: each such swap moves the readout by at
      most (the range of y_c over the ambiguous entries) / k. Plus the sums of k outcomes on both sides,
      2 (k + 2) u sum |y_ic| / k, and the divisions, 4 u |F_c|. With the core's order replicated here the
      gap is usually exactly 0.

    The bound is doubled for the second-order terms. A gap beyond it fails loudly.
    """
    e = 2.0 * (p + 6) * U * dist_scale
    F = np.abs(readout)
    if kind == "softmax":
        n = dist.size
        imin = int(np.argmin(dist))
        tiles = n / 128.0 + 2.0
        x = (dist - dist[imin]) * inv
        delta = U * (8.0 * (tiles + 2.0) + (2.0 * tiles + 6.0) * x) + (e + e[imin]) * inv
        one = (w * delta) @ (np.abs(ys) + F) + (n + 2) * U * (w @ np.abs(ys)) + 2.0 * U * F
        return 2.0 * 2.0 * one
    order = np.lexsort((np.arange(dist.size), dist))
    kth = order[k - 1]
    amb = np.abs(dist - dist[kth]) <= 2.0 * (e + e[kth])
    top = np.zeros(dist.size, dtype=bool)
    top[order[:k]] = True
    swaps = min(int((amb & top).sum()), int((amb & ~top).sum()))
    spread = (ys[amb].max(axis=0) - ys[amb].min(axis=0)) if amb.any() else np.zeros(ys.shape[1])
    one = swaps * spread / k + 2.0 * (k + 2) * U * (np.abs(ys[order[:k]]).sum(axis=0) / k) + 4.0 * U * F
    return 2.0 * one


def kernel_closing_bound(gap_bound: np.ndarray, n: int, groups: int, top: int, w: np.ndarray, ys: np.ndarray, sum_abs_c: np.ndarray,
                         raw: np.ndarray) -> np.ndarray:  # fmt: skip
    """The kernel tree's residual bound: the numpy-core gap's bound plus the rounding of the group sums.

    The groups are sums of w_i y_i over runs of entries, then summed (the top, the rest): at most
    (n + G + top + 4) u (sum_i w_i |y_ic| + sum_g |c_g| + |raw_c|), times 4 for the second-order terms.
    """
    return gap_bound + 4.0 * U * (n + groups + top + 4) * (w @ np.abs(ys) + sum_abs_c + np.abs(raw))


def _group_key(stamp_ms: np.ndarray, by: str) -> np.ndarray:
    unit = {"year": "datetime64[Y]", "month": "datetime64[M]", "day": "datetime64[D]"}[by]
    return np.asarray(stamp_ms, dtype=np.int64).astype("datetime64[ms]").astype(unit)


def partition_bound(n_g: np.ndarray, runs: np.ndarray, s1: np.ndarray, abs_c: np.ndarray, *, p: int = 0, s2: np.ndarray | None = None,
                    m_g: np.ndarray | None = None, s3: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Per group g and output c, a first-order bound on |c_g (tree) - sum_{i in g} w_i y_ic (recomputed)|.

    The recomputed sum adds the rounded products w_i y_ic in sequence (``numpy.bincount``): within
    (n_g + 1) u S1 of the exact sum of the weights it is given, S1 = sum_{i in g} |w_i y_ic|.

    - **kernel cells**: the tree's c_g is a dot product of the same weights and outcomes, summed over
      ``runs`` runs of entries: within (n_g + runs) u S1 + runs u |c_g|. Bound: 2 [(n_g + runs + 2) u S1
      + runs u |c_g|].
    - **ridge cells**: c_g comes from the group's own statistics (``centred_contribution``) and w_i from
      ``centred_weights``; both use the readout's xbar and a, and with those exact,
      (n_g / n) ybar_g + (M_yx,g + n_g ybar_g (xbar_g - xbar)) a = sum_{i in g} (1/n + (x_i - xbar)^T a) y_i
      for any xbar. What differs is rounding: each w_i is within (p + 3) u mu_i of its exact value,
      mu_i = 1/n + |x_i - xbar|^T |a|, which moves the sum by at most (p + 3) u S2, S2 = sum_i |y_ic| mu_i;
      the group's statistics (sums of at most n_g terms, then the p + 4 operations of c_g) are within
      (n_g + p + 8) u m_g,c S3, m_g,c = max_{i in g} |y_ic| and S3 = sum_i (1/n + 5 (mu_i - 1/n) + 2 nu_i),
      nu_i = |x_i|^T |a| (the group's mean, |xbar_g - xbar| and |M_yx,g| |a| are all bounded by these). Plus
      the runs, runs u |c_g|. Bound: 4 [(n_g + 2) u S1 + (p + 3) u S2 + (n_g + p + 8) u m_g,c S3 + runs u |c_g|].

    The factors 2 and 4 cover the second-order terms. Honest groups agree to about 1e-16, far inside.
    """
    if s2 is None or m_g is None or s3 is None:
        return 2.0 * U * ((n_g + runs + 2.0) * s1 + runs * abs_c)
    return 4.0 * U * ((n_g + 2.0) * s1 + (p + 3) * s2 + (n_g + p + 8) * m_g * s3 + runs * abs_c)


def twin_bound(*, n: int, my_abs: np.ndarray, mx_abs: np.ndarray, beta: np.ndarray, a: np.ndarray, Gxx: np.ndarray, Gyx: np.ndarray,
               lam: np.ndarray, A: np.ndarray, resid: np.ndarray, z: np.ndarray, rhs: np.ndarray, mean_y: np.ndarray,
               xt: np.ndarray) -> np.ndarray:  # fmt: skip
    """Per output c, a first-order bound on |twin forecast_c - core forecast_c| (u = 2^-53).

    Both compute f = ybar + M_yx A^-1 xt from the same n rows, in different orders. To first order the
    difference is the sum of each side's errors in ybar, xbar, the centred sums and the solve:

    - ybar: each side within n u mean|y_c|, so 2 n u mean|y_c|;
    - xbar, through xt = x_q - xbar: |beta_c| 2 n u mean|x|;
    - M_yx and M_xx (sums of n centred products; the core's are scans and tree merges, the twin's BLAS
      products): each within 2 n u of G_yx = sum |y_i - ybar||x_i - xbar|^T and G_xx = sum |x_i - xbar||x_i - xbar|^T,
      and an error E in M_yx or M_xx moves f by E a or beta E a (a = A^-1 xt): 4 n u (G_yx,c |a| + |beta_c| G_xx |a|);
    - the penalty rho n sigma^2, rounded on each side: 2 u |beta_c| (Lambda |a|);
    - the core's Cholesky solve, exact for A + E with |E| <= gamma_{3p+1} |L||L^T| (Higham, Thm 10.4):
      (3p + 1) u |beta_c| |L||L^T| |a|, with L numpy's factor of the twin's A;
    - the twin's solve, a posteriori: its beta_c is exact for the residual r_c = A z_c - M_xy,c, which moves
      f by r_c^T a; the residual is itself computed within (p + 1) u (|A||z_c| + |M_xy,c|);
    - the two readouts ybar + beta xt: 2 (p + 2) u (|ybar_c| + |beta_c| |xt|).

    Times 4 for the second-order terms. It reads the twin's own beta and a: nothing from Matrix.
    """
    p = A.shape[0]
    aa, bb = np.abs(a), np.abs(beta)
    try:
        L = np.abs(np.linalg.cholesky(A))
    except np.linalg.LinAlgError:
        raise LoudCheckFailed("the Matrix-free twin: numpy cannot factor the penalised matrix, so no bound") from None
    post = aa @ (np.abs(resid) + (p + 1) * U * (np.abs(A) @ np.abs(z) + np.abs(rhs)))
    one = (2.0 * n * U * my_abs + 2.0 * n * U * (bb @ mx_abs) + 4.0 * n * U * (Gyx @ aa + bb @ (Gxx @ aa)) + 2.0 * U * (bb @ (lam * aa))
           + (3 * p + 1) * U * (bb @ (L @ (L.T @ aa))) + post + 2.0 * (p + 2) * U * (np.abs(mean_y) + bb @ np.abs(xt)))  # fmt: skip
    return 4.0 * one


@dataclass(frozen=True)
class Choice:
    """The choices in force at a row for one ridge or kernel cell."""

    window: int  # index into cfg.windows_days
    rho: int | None
    var: np.ndarray
    clim: np.ndarray
    sigma: str
    fit: str | None
    note: str | None = None
    tau: int | None = None
    k: int | None = None
    stats: MonthStats | None = None
    fit_ms: int | None = None


class Explainer:
    """The panel, the memory and the run's choices, built once for any number of calls.

    ``panel`` is for tests below the command line (H1 on the explainer): a panel built elsewhere, whose
    fingerprint is then recorded in ``panel_fingerprint`` and not required to be the run's.
    """

    def __init__(self, run: RunFolder, *, data_dir: Any = None, universe: pl.DataFrame | None = None,
                 system_factory: Callable[..., MatrixSystem] = MatrixSystem, panel: pl.DataFrame | None = None) -> None:  # fmt: skip
        self.run, self.cfg, self.data_dir = run, run.cfg, data_dir
        cfg = run.cfg
        given = panel is not None
        if panel is None:
            panel = hourly_panel(universe, first=Month(*cfg.first_month), last=Month(*cfg.last_month), symbols=cfg.symbols, data_dir=data_dir)
        fp = fingerprint(panel)
        self.panel_fingerprint = fp
        self.panel_verified = fp == run.report["panel"]["fingerprint_sha256"]
        if not given and not self.panel_verified:
            raise Refused(f"the panel rebuilt from the store has fingerprint {fp}, not the run's "
                          f"{run.report['panel']['fingerprint_sha256']}: the store or the code changed since the run")  # fmt: skip
        self.system = system_factory(panel, cfg.matrix)
        d = self.system.data
        if list(d.symbols) != list(run.report["panel"]["symbols"]) or [str(s) for s in run.arrays["symbols"]] != list(d.symbols):
            raise Refused("the run's symbols are not the rebuilt panel's")
        mc = cfg.matrix
        self.mc = mc
        self.fits = {month_start_ms(int(f["month"][:4]), int(f["month"][5:7])): f for f in run.report["fits"]}
        self.cusum_fits = sorted(
            (_parse_iso_ms(f["fit"]), mc.windows_days.index(f["window"]), mc.rhos.index(f["rho"])) for f in run.report["cusum"]["fits"]
        )
        key = row_key(run.arrays["symbol"], run.arrays["open_time"])
        self._series_order = np.argsort(key, kind="stable")
        self._series_keys = key[self._series_order]
        if np.any(np.diff(self._series_keys) == 0):
            raise Refused("series.npz holds a (symbol, open time) twice")
        self.arm_index = {a: i for i, a in enumerate(run.arms)}
        self._bars: dict[str, np.ndarray] = {}
        self._pools: dict[tuple[int, int], Record] = {}
        self.ys = np.zeros((self.system.entry_rows.size, 4))
        er = self.system.entry_rows
        self.ys[np.arange(er.size), d.label[er]] = 1.0
        self.ys[:, 3] = d.fwd[er]
        # check 1's own stamps, recomputed from panel rows: entries in memory order, labelled rows
        self._entry_stamp = d.open_ms[er] + LABEL_LAG_MS
        lab = np.flatnonzero(d.labelled)
        self._lab_stamp = d.open_ms[lab] + LABEL_LAG_MS
        self._lab_label = d.label[lab].astype(np.int64)
        self._rules: dict[int, tuple[np.ndarray | None, int, np.ndarray, int]] = {}

    # --- rows and choices ---

    def find_row(self, pair: str, t: int) -> int:
        """The pair's row whose decision time, or whose bar's open, is ``t``."""
        d = self.system.data
        if pair not in d.symbols:
            raise Refused(f"{pair} is not a member of the run's panel")
        code = d.symbols.index(pair)
        hit = np.flatnonzero((d.symbol == code) & ((d.decision_ms == t) | (d.open_ms == t)))
        if hit.size != 1:
            raise Refused(f"{pair} has no row whose decision time or bar open is {iso(t)}")
        row = int(hit[0])
        if not d.features_ok[row]:
            raise Refused(f"{pair} at {iso(t)}: a feature is missing, so no arm reads this row")
        return row

    def month_choice(self, row: int, arm: str) -> dict[str, Any]:
        """The monthly arms' choices at a row: from the fit of the row's month, or the registered defaults before the first fit."""
        d, mc = self.system.data, self.mc
        m = int(d.month_ms[row])
        st = self.system.month_stats(m)
        if st.var is None:
            raise Refused(f"the month {month_label(m)} has no sigma (fewer than two entries known by its start minus 24 h)")
        fit = self.fits.get(m)
        rho_d = mc.rhos.index(mc.default_rho)
        if fit is None:
            out = {"select": (mc.default_window, rho_d), "rho_1y": rho_d, "tau": mc.tau_factors.index(mc.default_tau_factor),
                   "k": mc.ks.index(mc.default_k), "rho_by_window": [rho_d] * len(mc.windows_days),
                   "alpha": np.array(mc.default_alpha), "fit": None, "note": DEFAULTS_NOTE}  # fmt: skip
        else:
            out = {"select": (mc.windows_days.index(fit["select"]["window"]), mc.rhos.index(fit["select"]["rho"])),
                   "rho_1y": mc.rhos.index(fit["rho_1y"]), "tau": mc.tau_factors.index(fit["tau_factor"]), "k": mc.ks.index(fit["k"]),
                   "rho_by_window": [mc.rhos.index(r) for r in fit["stack_rho"]], "alpha": np.array(fit["alpha"], dtype=np.float64),
                   "fit": fit["month"] + (" (a default fit: fewer than 1,000 selection rows)" if fit["default"] else ""), "note": None}  # fmt: skip
        out.update(var=st.var, clim=st.clim, sigma=f"month {month_label(m)}", stats=st, fit_ms=m)
        return out

    def cusum_choice(self, row: int) -> Choice:
        d, mc = self.system.data, self.mc
        dec = int(d.decision_ms[row])
        live = [f for f in self.cusum_fits if f[0] <= dec]
        if not live:
            mo = self.month_choice(row, "matrix-cusum")
            return Choice(mc.default_window, mc.rhos.index(mc.default_rho), mo["var"], mo["clim"], mo["sigma"], None, DEFAULTS_NOTE,
                          stats=mo["stats"], fit_ms=mo["fit_ms"])  # fmt: skip
        fit_ms, w, r = live[-1]
        st = self.system.fit_stats(fit_ms)
        if st.var is None:
            raise Refused(f"matrix-cusum's fit at {iso(fit_ms)} has no sigma")
        return Choice(w, r, st.var, st.clim, f"fit {iso(fit_ms)} (matrix-cusum's sigma is that of its fit time)", f"cusum fit {iso(fit_ms)}",
                      stats=st, fit_ms=fit_ms)  # fmt: skip

    def adaptive_rho(self, row: int, window: int) -> tuple[int, dict[str, Any], np.ndarray]:
        """matrix-adaptive's rho at the row's decision, rebuilt with the arm's code from the candidates' forecasts.

        The pool of the row's month is the one :meth:`MatrixSystem.adaptive_choices` builds from the
        prequential record: every complete row of a month with a sigma, with ``label_known_at`` up to the
        next month's start plus a day and still in some 30-day span of the month. Its forecasts are
        re-read cell by cell (``ridge_batch`` gives each cell the same bits whatever the other queries),
        so the prefix sums, and the choice, are the record's. Returns the rho's index, the scores, and the
        rows whose losses the choice read.
        """
        sysm, mc = self.system, self.mc
        d = sysm.data
        m, dec = int(d.month_ms[row]), int(d.decision_ms[row])
        R = len(sysm.rhos)
        key = (m, window)
        if key not in self._pools:
            cand = np.flatnonzero(d.complete)
            months = [int(x) for x in np.unique(d.month_ms[cand]) if sysm.month_stats(int(x)).var is not None]
            cand = cand[np.isin(d.month_ms[cand], months)]
            lka, decs = d.lka_ms[cand], d.decision_ms[cand]
            exit_ = np.maximum(lka, decs + mc.adaptive_days * DAY_MS)
            pool = cand[(lka <= next_month_ms(m) + DAY_MS) & (exit_ > m)]
            C = len(sysm.candidates)
            loss = np.full((pool.size, C), np.nan)
            for mm in np.unique(d.month_ms[pool]):
                at = np.flatnonzero(d.month_ms[pool] == mm)
                st = sysm.month_stats(int(mm))
                rr = pool[at]
                out = sysm._ridge(rr, sysm.ridge_as_of(rr), st.var, [sysm.windows[window]], sysm.rhos, extras=False)
                fc = out["forecast"].reshape(at.size, R, 4)
                good = out["status"].reshape(at.size, R) == rs.STATUS_OK
                p = prob_map(fc[..., :N_CLASSES], mc.prob_floor)
                p[~good] = st.clim
                loss[at, window * R : (window + 1) * R] = label_loss(p, d.label[rr])
            n = pool.size
            self._pools[key] = Record(sysm.candidates, pool, d.month_ms[pool], np.zeros((n, 0, 3)), np.zeros((n, 0), bool),
                                      np.zeros((n, 0), np.uint8), loss, np.zeros(n, bool), np.zeros(0, np.int64))  # fmt: skip
        rec = self._pools[key]
        ch = sysm.adaptive_choices(rec, m, window, np.array([dec], dtype=np.int64))
        lka, decs = d.lka_ms[rec.rows], d.decision_ms[rec.rows]
        read = rec.rows[(lka <= dec) & (np.maximum(lka, decs + mc.adaptive_days * DAY_MS) > dec)]
        info = {"rows_scored": int(ch["count"][0]), "default": bool(ch["default"][0]),
                "scores": {f"{mc.rhos[i]:g}": float(ch["scores"][0, i]) for i in range(R)},
                "rule": "the rho of lowest mean prequential log-loss over the rows known by the decision and decided after it minus 30 days"}  # fmt: skip
        return int(ch["choice"][0]), info, read

    def series_row(self, row: int) -> int | None:
        """The row's index in series.npz, or None when it is not scored."""
        d = self.system.data
        k = row_key(d.symbol[row : row + 1], d.open_ms[row : row + 1])[0]
        i = int(np.searchsorted(self._series_keys, k))
        return int(self._series_order[i]) if i < self._series_keys.size and self._series_keys[i] == k else None

    # --- raw bars ---

    def bar_opens(self, symbol: str) -> np.ndarray:
        """The open times of a pair's raw hourly bars, through the sealed loader, cut as the panel cuts them."""
        if symbol not in self._bars:
            bars = load_klines(symbol, INTERVAL, data_dir=self.data_dir, columns=["open_time", "known_at"], caller="explain.why")
            bars = bars.filter(pl.col("known_at") < DEVELOPMENT.end)
            self._bars[symbol] = np.sort(bars["open_time"].dt.epoch("ms").to_numpy().astype(np.int64))
        return self._bars[symbol]

    # --- the loud checks ---

    def _rows_known(self, rr: np.ndarray, as_of: int, name: str) -> int | None:
        """Panel rows ``rr``: the recomputed stamp equals the panel's and is at or before ``as_of``; the latest such stamp."""
        d = self.system.data
        recomputed = d.open_ms[rr] + LABEL_LAG_MS
        bad = rr[d.lka_ms[rr] != recomputed]
        if bad.size:
            r0 = int(bad[0])
            raise LoudCheckFailed(f"stamp check: {bad.size} {name} have a panel label_known_at that is not their bar's open + 25 h, "
                                  f"e.g. {d.symbols[d.symbol[r0]]} at {iso(int(d.open_ms[r0]))}")  # fmt: skip
        is_late = recomputed > as_of
        late = rr[is_late]
        if late.size:
            r0 = int(late[np.argmax(recomputed[is_late])])
            raise LoudCheckFailed(f"stamp check: {late.size} of {rr.size} {name} read at {iso(as_of)} have a label known later, "
                                  f"e.g. {d.symbols[d.symbol[r0]]} decided {iso(int(d.decision_ms[r0]))}, label known "
                                  f"{iso(int(d.open_ms[r0] + LABEL_LAG_MS))}")  # fmt: skip
        return int(recomputed.max()) if rr.size else None

    def rule_check(self, what: str, fit_ms: int, st: MonthStats) -> dict[str, Any]:
        """sigma^2 and the climatology of a fit at ``fit_ms``, recomputed from panel rows and held to the protocol's rule to the bit."""
        sysm, mc = self.system, self.mc
        cutoff = int(fit_ms - mc.horizon_ms)
        if fit_ms not in self._rules:
            known = self._entry_stamp <= cutoff
            k = int(known.sum())
            if not bool(known[:k].all()):
                raise LoudCheckFailed(f"stamp check, {what}: the entries with a recomputed stamp at or before {iso(cutoff)} are not a "
                                      "prefix of the memory: Matrix orders its entries by other stamps")  # fmt: skip
            var = None
            if k >= 2:
                cs = sysm.matrix.centred_stats(0, k)
                var = np.ascontiguousarray(np.diag(np.asarray(cs.m_xx)) / cs.n)
            sel = self._lab_stamp <= cutoff
            i = int(sel.sum())
            counts = np.bincount(self._lab_label[sel], minlength=N_CLASSES).astype(np.float64)
            clim = np.full(N_CLASSES, 1.0 / N_CLASSES) if i == 0 else prob_map(counts / float(i), mc.prob_floor)
            self._rules[fit_ms] = (var, k, clim, i)
        var, k, clim, i = self._rules[fit_ms]
        if (var is None) != (st.var is None) or (var is not None and not _same_bits(var, st.var)) or st.entries != k:
            raise LoudCheckFailed(f"stamp check, {what}: sigma^2 in force is not the variance of the {k} entries with a recomputed stamp "
                                  f"at or before {iso(cutoff)} (the arm read {st.entries} entries): sigma reads entries known later")  # fmt: skip
        if not _same_bits(clim, st.clim) or st.clim_rows != i:
            raise LoudCheckFailed(f"stamp check, {what}: the climatology in force is not that of the {i} labelled rows with a recomputed "
                                  f"stamp at or before {iso(cutoff)} (the arm read {st.clim_rows})")  # fmt: skip
        return {"what": what, "fit_time": iso(fit_ms), "known_by": iso(cutoff), "entries_behind_sigma": k, "climatology_rows": i,
                "sigma_equal_to_the_rule": True, "climatology_equal_to_the_rule": True}  # fmt: skip

    def stamp_check(self, row: int, spans: list[tuple[int, int]], top_entries: list[int], extra_rows: np.ndarray | None = None,
                    extra_what: str = "", rules: list[tuple[str, int, MonthStats]] | None = None) -> dict[str, Any]:  # fmt: skip
        """Loud check 1: the entries of every weighted span, the fold statistics in force, and the top entries on raw bars.

        ``spans`` are the entry ranges of the weighted cells, each checked on its own (a stack's windows
        overlap, and each is counted), so ``entries.checked`` is the sum of the cells' ``n_window``.
        """
        d, sysm = self.system.data, self.system
        as_of = int(d.decision_ms[row])
        out: dict[str, Any] = {"as_of": iso(as_of), "matrix_stamps_used": False}
        checked, latest = [], None
        for s, e in spans:
            if e <= s:
                raise LoudCheckFailed(f"stamp check: a weighted cell at {iso(as_of)} has no entry to check ([{s}, {e}))")
            t = self._rows_known(sysm.entry_rows[s:e], as_of, "entries")
            checked.append(e - s)
            latest = t if latest is None else max(latest, t)
        out["entries"] = {"checked": int(sum(checked)), "per_weighted_window": checked, "latest_label_known_at": None if latest is None else iso(latest)}
        if extra_rows is not None:
            t = self._rows_known(extra_rows, as_of, extra_what)
            out[extra_what] = {"checked": int(extra_rows.size), "latest_label_known_at": None if t is None else iso(t)}
        out["fold_statistics"] = [self.rule_check(what, f, st) for what, f, st in (rules or [])]
        raw = 0
        for pos in top_entries:
            r0 = int(sysm.entry_rows[pos])
            sym = d.symbols[d.symbol[r0]]
            exit_open = int(d.open_ms[r0] + LABEL_LAG_MS)
            opens = self.bar_opens(sym)
            i = int(np.searchsorted(opens, exit_open))
            if i >= opens.size or opens[i] != exit_open:
                raise LoudCheckFailed(f"stamp check: {sym}'s raw bars have no bar opening at {iso(exit_open)}, the exit of the entry decided "
                                      f"{iso(int(d.decision_ms[r0]))}")  # fmt: skip
            if opens[i] > as_of:
                raise LoudCheckFailed(f"stamp check: the raw exit bar of {sym}'s entry decided {iso(int(d.decision_ms[r0]))} opens at "
                                      f"{iso(int(opens[i]))}, after {iso(as_of)}")  # fmt: skip
            raw += 1
        out["top_entries_checked_on_raw_bars"] = raw
        out["covers"] = list(CHECK1_COVERS)
        out["does_not_cover"] = list(CHECK1_LEAVES)
        out["passed"] = True
        return out

    def forecast_check(self, row: int, arm: str, probs: np.ndarray) -> dict[str, Any]:
        """Loud check 2: the probabilities equal the arm's in ``series.npz`` to the bit, where the row is scored."""
        d = self.system.data
        i = self.series_row(row)
        if i is None:
            return {"applies": False, "note": "this row is not scored, so series.npz holds no forecast for it: only the stamp check applies"}
        a = self.run.arrays
        if int(a["label_known_at"][i]) != int(d.open_ms[row] + LABEL_LAG_MS) or int(a["label"][i]) != int(d.label[row]):
            raise LoudCheckFailed(f"forecast check: series.npz row {i} does not carry this row's label and stamp")
        want = a["probs"][self.arm_index[arm], i]
        if not _same_bits(want, probs):
            raise LoudCheckFailed(f"forecast check: {arm} at {d.symbols[d.symbol[row]]} {iso(int(d.decision_ms[row]))}: rebuilt "
                                  f"{[float(x).hex() for x in probs]}, series.npz {[float(x).hex() for x in want]}")  # fmt: skip
        return {"applies": True, "series_row": i, "equal_to_the_bit": True, "probs_hex": [float(x).hex() for x in want]}

    # --- cells ---

    def ridge_cell(self, row: int, w: int, r: int, var: np.ndarray, clim: np.ndarray, *, by: str, top: int, full: bool,
                   twin: bool = False) -> dict[str, Any]:  # fmt: skip
        """One ridge cell at a row: readout, probabilities, (``full``) the tree, per-feature and top-entry views, (``twin``) the Matrix-free twin."""
        sysm, mc = self.system, self.mc
        d = sysm.data
        rows = np.array([row], dtype=np.int64)
        out = sysm._ridge(rows, sysm.ridge_as_of(rows), var, [sysm.windows[w]], np.array([mc.rhos[r]]), extras=True)
        status = int(out["status"][0, 0, 0])
        raw = out["forecast"][0, 0, 0, :].copy()
        s, e = int(out["start"][0, 0]), int(out["end"][0, 0])
        ok = status == rs.STATUS_OK
        probs = prob_map(raw[None, :N_CLASSES], mc.prob_floor)[0] if ok else np.array(clim, dtype=np.float64)
        cell: dict[str, Any] = {"window": _wname(mc.windows_days[w]), "rho": mc.rhos[r], "status": STATUS_NAMES[status], "n_window": e - s,
                                "raw": dict(zip(OUTPUTS, raw.tolist(), strict=True)), "probs": probs, "fallback_to_climatology": not ok}  # fmt: skip
        res = {"cell": cell, "probs": probs, "span": (s, e), "top_positions": [], "weights": None}
        if not ok:
            cell["note"] = "the core refused the window: the arm's climatology is the forecast, and no entry is weighted"
            return res
        x_q = np.ascontiguousarray(d.x[row])
        ro = sysm.matrix.centred_query(x_q, s, e, float(mc.rhos[r]), var)
        if not _same_bits(ro.forecast, raw):
            raise LoudCheckFailed(f"centred_query and ridge_batch disagree on the cell ({cell['window']}, rho {cell['rho']:g}) at "
                                  f"{iso(int(d.decision_ms[row]))}: the core's bit identity is broken")  # fmt: skip
        wts = np.asarray(sysm.matrix.centred_weights(ro))
        res["weights"] = wts
        res["diagnostics"] = diagnostics(wts, float(ro.leverage))
        order = np.lexsort((np.arange(wts.size), -np.abs(wts)))[:top]
        res["top_positions"] = [s + int(i) for i in order]
        solve = None
        if full:
            groups, meta = self.group_contributions(s, e, lambda a, b: np.asarray(sysm.matrix.centred_contribution(ro, a, b)), by)
            sum_abs = np.sum([np.abs(c) for _, c in groups], axis=0)
            mags = self.ridge_mags(ro, s, e)
            solve = self.solve_magnitude(ro, s, e, float(mc.rhos[r]), var)
            bound = self.ridge_bound(ro, s, e, x_q, len(groups), sum_abs, mags, solve)
            res["tree"] = tree_view(groups, raw, top, bound, "ridge_closing_bound: statistics, group sums, readout and the two solves")
            res["tree"]["partition"] = self.check_partition(s, e, by, groups, meta, wts, mags)
            res["top_entries"] = self.entries(row, s, order, wts)
        res["features"] = feature_view(ro, x_q, raw)
        if twin:
            res["twin_job"] = (row, w, float(mc.rhos[r]), var, raw, e - s)  # run by explain, after check 1
        return res

    def ridge_mags(self, ro: Any, s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
        """Per entry of ``[s, e)``: mu_i = 1/n + |x_i - xbar|^T |a| and nu_i = |x_i|^T |a| (in chunks)."""
        d, sysm = self.system.data, self.system
        a = np.abs(np.asarray(ro.a, dtype=np.float64))
        mean_x = np.asarray(ro.mean_x, dtype=np.float64)
        rows = sysm.entry_rows[s:e]
        mu, nu = np.empty(rows.size), np.empty(rows.size)
        for lo in range(0, rows.size, 1 << 18):
            x = d.x[rows[lo : lo + (1 << 18)]]
            mu[lo : lo + x.shape[0]] = np.abs(x - mean_x) @ a
            nu[lo : lo + x.shape[0]] = np.abs(x) @ a
        return mu + 1.0 / max(rows.size, 1), nu

    def solve_magnitude(self, ro: Any, s: int, e: int, rho: float, var: np.ndarray) -> np.ndarray:
        """S_c = |beta_c| |L||L^T| |a|, with numpy's Cholesky factor of A = M_xx + rho n diag(sigma^2) of the window."""
        cs = self.system.matrix.centred_stats(s, e)
        n = float(cs.n)
        A = np.array(cs.m_xx, dtype=np.float64) + np.diag(rho * n * np.asarray(var, dtype=np.float64))
        try:
            L = np.abs(np.linalg.cholesky(A))
        except np.linalg.LinAlgError:
            raise LoudCheckFailed("numpy cannot factor the penalised matrix the core solved, so no rounding bound") from None
        a = np.abs(np.asarray(ro.a, dtype=np.float64))
        return np.abs(np.asarray(ro.beta, dtype=np.float64)) @ (L @ (L.T @ a))

    def ridge_bound(self, ro: Any, s: int, e: int, x_q: np.ndarray, groups: int, sum_abs_c: np.ndarray,
                    mags: tuple[np.ndarray, np.ndarray], solve: np.ndarray) -> np.ndarray:  # fmt: skip
        """The magnitudes behind :func:`ridge_closing_bound` for one ridge cell, read from the window's entries."""
        p = x_q.size
        beta = np.abs(np.asarray(ro.beta, dtype=np.float64))
        mean_x = np.asarray(ro.mean_x, dtype=np.float64)
        readout_terms = np.abs(np.asarray(ro.mean_y, dtype=np.float64)) + beta @ np.abs(x_q - mean_x)
        a1 = float((mags[0] - 1.0 / max(e - s, 1)).sum())
        m = np.abs(self.ys[s:e]).max(axis=0)
        return ridge_closing_bound(e - s, groups, p, m * (2.0 + 8.0 * a1), readout_terms, solve, sum_abs_c)

    # --- the Matrix-free twin ---

    def twin(self, row: int, w: int, rho: float, var: np.ndarray, raw: np.ndarray, n_window: int) -> dict[str, Any]:
        """The ridge forecast rebuilt from panel rows alone, in numpy, held to the core's within :func:`twin_bound`.

        The rows: every complete row of the panel whose ``label_known_at`` column lies in
        ``(as_of - window, as_of]`` (``as_of`` the row's decision time; all of them up to ``as_of`` for the
        whole-history window). They are centred on their own means, and the ridge
        ``A beta^T = M_xy``, ``A = M_xx + rho n diag(sigma^2)`` (sigma^2 the one in force, held to the rule
        by check 1), is solved by ``numpy.linalg.solve``. Nothing is read from Matrix but the core's
        forecast and the window's entry count, the two things compared. A different count, or a forecast
        beyond the bound, fails loudly.
        """
        d = self.system.data
        if not hasattr(self, "_twin_rows"):
            rows = np.flatnonzero(d.complete)
            order = np.argsort(d.lka_ms[rows], kind="stable")
            self._twin_rows, self._twin_lka = rows[order], d.lka_ms[rows[order]]
        as_of = int(d.decision_ms[row])
        days = self.mc.windows_days[w]
        hi = int(np.searchsorted(self._twin_lka, as_of, side="right"))
        lo = 0 if days is None else int(np.searchsorted(self._twin_lka, as_of - int(days) * DAY_MS, side="right"))
        rows = self._twin_rows[lo:hi]
        n = rows.size
        where = f"the Matrix-free twin at {d.symbols[d.symbol[row]]} {iso(as_of)}, window {_wname(days)}"
        if n != n_window:
            raise LoudCheckFailed(f"{where}: the panel has {n} complete rows with label_known_at in the window, Matrix's window "
                                  f"holds {n_window} entries")  # fmt: skip
        p, q = d.x.shape[1], 4
        chunk = 1 << 18

        def y_of(rr: np.ndarray) -> np.ndarray:
            y = np.zeros((rr.size, q))
            y[np.arange(rr.size), d.label[rr]] = 1.0
            y[:, 3] = d.fwd[rr]
            return y

        sx, sy, ax, ay = np.zeros(p), np.zeros(q), np.zeros(p), np.zeros(q)
        for c0 in range(0, n, chunk):
            rr = rows[c0 : c0 + chunk]
            x, y = d.x[rr], y_of(rr)
            sx += x.sum(axis=0)
            sy += y.sum(axis=0)
            ax += np.abs(x).sum(axis=0)
            ay += np.abs(y).sum(axis=0)
        mx, my = sx / n, sy / n
        Mxx, Myx, Gxx, Gyx = np.zeros((p, p)), np.zeros((q, p)), np.zeros((p, p)), np.zeros((q, p))
        for c0 in range(0, n, chunk):
            rr = rows[c0 : c0 + chunk]
            xc, yc = d.x[rr] - mx, y_of(rr) - my
            Mxx += xc.T @ xc
            Myx += yc.T @ xc
            axc = np.abs(xc)
            Gxx += axc.T @ axc
            Gyx += np.abs(yc).T @ axc
        lam = rho * float(n) * np.asarray(var, dtype=np.float64)
        A = Mxx + np.diag(lam)
        xt = d.x[row] - mx
        rhs = np.column_stack([Myx.T, xt])
        try:
            z = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            raise LoudCheckFailed(f"{where}: numpy cannot solve the twin's penalised system, so the forecast cannot be rebuilt") from None
        beta_t, a_t = z[:, :q].T, z[:, q]
        forecast = my + beta_t @ xt
        resid = A @ z - rhs
        bound = twin_bound(n=n, my_abs=ay / n, mx_abs=ax / n, beta=beta_t, a=a_t, Gxx=Gxx, Gyx=Gyx, lam=lam, A=A,
                           resid=resid[:, :q], z=z[:, :q], rhs=rhs[:, :q], mean_y=my, xt=xt)  # fmt: skip
        gap = np.abs(forecast - raw)
        if not bool(np.all(gap <= bound)):
            o = int(np.argmax(gap - bound))
            raise LoudCheckFailed(f"{where}: the twin's {OUTPUTS[o]} is {forecast[o]!r}, the core's {raw[o]!r}: a gap of {gap[o]:.3g}, "
                                  f"beyond its bound {bound[o]:.3g}")  # fmt: skip
        return {"rows": int(n), "gap": dict(zip(OUTPUTS, gap.tolist(), strict=True)), "bound": dict(zip(OUTPUTS, bound.tolist(), strict=True)),
                "within_bound": True, "gap_over_bound": float(np.max(gap / bound))}  # fmt: skip

    def kernel_cell(self, row: int, kind: str, idx: int, var: np.ndarray, clim: np.ndarray, *, by: str, top: int, full: bool) -> dict[str, Any]:
        """A kernel cell: the forecast from the core (for the bit check), the weights recomputed in numpy."""
        sysm, mc = self.system, self.mc
        d = sysm.data
        ko = sysm._kernel(np.array([row], dtype=np.int64), var)
        s, e = int(ko["start"][0]), int(ko["end"][0])
        raw = ko[kind][0, idx].copy()
        status = int(ko[f"{kind}_status"][0, idx])
        ok = status == rs.STATUS_OK
        probs = prob_map(raw[None, :N_CLASSES], mc.prob_floor)[0] if ok else np.array(clim, dtype=np.float64)
        param = {"tau": float(sysm.taus[idx]), "tau_factor": mc.tau_factors[idx]} if kind == "softmax" else {"k": int(sysm.ks[idx])}
        cell = {"window": _wname(mc.kernel_window_days), **param, "status": STATUS_NAMES[status], "n_window": e - s,
                "raw": dict(zip(OUTPUTS, raw.tolist(), strict=True)), "probs": probs, "fallback_to_climatology": not ok,
                "weights_note": "the forecast is the core's; the weights below are recomputed in numpy (distances in the core's order), "
                                "and their readout's gap to the core's is reported with its bound"}  # fmt: skip
        res = {"cell": cell, "probs": probs, "span": (s, e), "top_positions": [], "weights": None}
        if not ok:
            cell["note"] = "the core refused the read: the arm's climatology is the forecast, and no entry is weighted"
            return res
        sig = np.sqrt(var)
        u = d.x[sysm.entry_rows[s:e]] / sig
        uq = d.x[row] / sig
        dist = np.zeros(u.shape[0])
        for j in range(u.shape[1]):  # the core's order: d_i = sum_j (u_q,j - u_i,j)^2, j = 0, 1, ...
            t = uq[j] - u[:, j]
            dist += t * t
        dist_scale = ((np.abs(u) + np.abs(uq)) ** 2).sum(axis=1)
        ys = self.ys[s:e]
        p = u.shape[1]
        if kind == "softmax":
            inv = 1.0 / (2.0 * float(sysm.taus[idx]) * float(sysm.taus[idx]))
            wts = np.exp(-(dist - dist.min()) * inv)
            wts = wts / wts.sum()
            readout = wts @ ys
            gap_bound = kernel_gap_bound("softmax", wts, ys, readout, dist, dist_scale, p=p, inv=inv)
        else:
            k = int(sysm.ks[idx])
            near = np.lexsort((np.arange(dist.size), dist))[:k]
            wts = np.zeros(dist.size)
            wts[near] = 1.0 / k
            readout = np.cumsum(ys[near], axis=0)[-1] / k  # the core's order: the k outcomes by (distance, index), then / k
            gap_bound = kernel_gap_bound("knn", wts, ys, readout, dist, dist_scale, p=p, k=k)
        gap = np.abs(readout - raw)
        if not bool(np.all(gap <= gap_bound)):
            o = int(np.argmax(gap - gap_bound))
            raise LoudCheckFailed(f"{kind} at {iso(int(d.decision_ms[row]))}: the numpy readout misses the core's on {OUTPUTS[o]} by "
                                  f"{gap[o]:.3g}, beyond its bound {gap_bound[o]:.3g}")  # fmt: skip
        cell["numpy_readout_gap"] = dict(zip(OUTPUTS, gap.tolist(), strict=True))
        cell["numpy_readout_gap_bound"] = dict(zip(OUTPUTS, gap_bound.tolist(), strict=True))
        res["weights"] = wts
        res["diagnostics"] = diagnostics(wts, None)
        order = np.lexsort((np.arange(wts.size), -np.abs(wts)))[:top]
        res["top_positions"] = [s + int(i) for i in order]
        if full:
            groups, meta = self.group_contributions(s, e, lambda a, b: wts[a - s : b - s] @ self.ys[a:b], by)
            sum_abs = np.sum([np.abs(c) for _, c in groups], axis=0)
            bound = kernel_closing_bound(gap_bound, e - s, len(groups), top, wts, ys, sum_abs, raw)
            res["tree"] = tree_view(groups, raw, top, bound, "kernel_closing_bound: the numpy-core gap's bound plus the group sums")
            res["tree"]["note"] = "kernel weights recomputed in numpy: the closing residual also holds their readout's gap to the core's"
            res["tree"]["partition"] = self.check_partition(s, e, by, groups, meta, wts, None)
            res["top_entries"] = self.entries(row, s, order, wts)
        return res

    def group_stamps(self, s: int, e: int) -> np.ndarray:
        """The grouping stamp of entries ``[s, e)``: ``label_known_at`` recomputed as the bar's open + 25 h."""
        d, sysm = self.system.data, self.system
        return d.open_ms[sysm.entry_rows[s:e]] + LABEL_LAG_MS

    def group_contributions(self, s: int, e: int, contribution: Callable[[int, int], np.ndarray],
                            by: str) -> tuple[list[tuple[str, np.ndarray]], dict[str, tuple[int, int]]]:  # fmt: skip
        """The contribution of each stamp group of entries ``[s, e)``, in entry order; and per group, its entry count and runs.

        A group is summed over runs of consecutive entries; ``runs`` is how many (1 when the memory's order
        follows the stamps, as it does).
        """
        key = _group_key(self.group_stamps(s, e), by)
        cut = np.flatnonzero(key[1:] != key[:-1]) + 1
        bounds = np.concatenate([[0], cut, [key.size]])
        sums: dict[str, np.ndarray] = {}
        meta: dict[str, tuple[int, int]] = {}
        for a, b in itertools.pairwise(bounds):
            label = str(key[a])
            c = np.asarray(contribution(s + int(a), s + int(b)), dtype=np.float64)
            sums[label] = sums[label] + c if label in sums else c
            n0, r0 = meta.get(label, (0, 0))
            meta[label] = (n0 + int(b - a), r0 + 1)
        return list(sums.items()), meta

    def check_partition(self, s: int, e: int, by: str, groups: list[tuple[str, np.ndarray]], meta: dict[str, tuple[int, int]],
                        wts: np.ndarray, mags: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:  # fmt: skip
        """The tree's partition, checked apart from its sum (a loud failure otherwise).

        1. The groups are recounted from the panel's own ``label_known_at`` column (which check 1 holds
           equal to open + 25 h): the same labels, with the same entry counts.
        2. Each group's contribution is recomputed as sum_{i in g} w_i y_i, from the cell's weights and the
           entries' outcomes, grouped by that column, and must equal the tree's within
           :func:`partition_bound` (``mags`` given: a ridge cell; None: a kernel cell).
        """
        d, sysm = self.system.data, self.system
        key = _group_key(d.lka_ms[sysm.entry_rows[s:e]], by)
        labels, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        names = [str(x) for x in labels]
        recount = dict(zip(names, counts.tolist(), strict=True))
        tree = {lab: meta[lab][0] for lab, _ in groups}
        if recount != tree:
            differ = sorted(k for k in set(recount) | set(tree) if recount.get(k) != tree.get(k))
            eg = [(k, tree.get(k), recount.get(k)) for k in differ[:2]]
            raise LoudCheckFailed(f"tree view: the groups are not those of the panel's label_known_at: {len(differ)} of {len(recount)} "
                                  f"groups differ in label or entry count, e.g. (group, tree's count, panel's count) {eg}")  # fmt: skip
        ys = self.ys[s:e]
        terms = wts[:, None] * ys
        G = labels.size

        def per_group(v: np.ndarray) -> np.ndarray:
            return np.stack([np.bincount(inv, weights=v[:, o], minlength=G) for o in range(v.shape[1])], axis=1)

        twin = per_group(terms)
        s1 = per_group(np.abs(terms))
        n_g = counts.astype(np.float64)[:, None]
        tree_c = np.array([dict(groups)[x] for x in names])
        runs = np.array([meta[x][1] for x in names], dtype=np.float64)[:, None]
        if mags is None:
            bound = partition_bound(n_g, runs, s1, np.abs(tree_c))
        else:
            mu, nu = mags
            n = float(e - s)
            m_g = np.zeros((G, ys.shape[1]))
            np.maximum.at(m_g, inv, np.abs(ys))
            s2 = per_group(np.abs(ys) * mu[:, None])
            s3 = np.bincount(inv, weights=1.0 / n + 5.0 * (mu - 1.0 / n) + 2.0 * nu, minlength=G)[:, None]
            bound = partition_bound(n_g, runs, s1, np.abs(tree_c), p=d.x.shape[1], s2=s2, m_g=m_g, s3=s3)
        gap = np.abs(tree_c - twin)
        if not bool(np.all(gap <= bound)):
            g, o = np.unravel_index(int(np.argmax(gap - bound)), gap.shape)
            raise LoudCheckFailed(f"tree view: group {names[g]}'s contribution to {OUTPUTS[o]} is {tree_c[g, o]!r}, but its entries' "
                                  f"sum w_i y_i is {twin[g, o]!r}: a gap of {gap[g, o]:.3g}, beyond its bound {bound[g, o]:.3g}")  # fmt: skip
        ratio = np.where(bound > 0, gap / np.where(bound > 0, bound, 1.0), 0.0)
        return {"recounted_from": "the panel's label_known_at column", "groups": int(G), "same_labels_and_counts": True,
                "sum_w_y_within_bound": True, "max_gap": float(gap.max()) if G else 0.0,
                "max_gap_over_bound": float(ratio.max()) if G else 0.0}  # fmt: skip

    def entries(self, row: int, s: int, order: np.ndarray, wts: np.ndarray) -> list[dict[str, Any]]:
        d, sysm = self.system.data, self.system
        out = []
        for i in order:
            r0 = int(sysm.entry_rows[s + int(i)])
            out.append({"pair": d.symbols[d.symbol[r0]], "decision_time": iso(int(d.decision_ms[r0])),
                        "label_known_at": iso(int(d.open_ms[r0] + LABEL_LAG_MS)), "label": int(d.label[r0]), "fwd_return": float(d.fwd[r0]),
                        "weight": float(wts[i]), "delta": dict(zip(FEATURE_COLUMNS, (d.x[r0] - d.x[row]).tolist(), strict=True))})  # fmt: skip
        return out

    # --- an arm at a row ---

    def explain(self, row: int, arm: str, *, by: str, top: int, full: bool = True, checks: str = "both",
                twin: bool = False) -> dict[str, Any]:
        """Everything ``why`` says about one arm at one row, both loud checks passed.

        ``checks`` is "both" on the command line; "stamps" (check 1 only) and "none" are for tests below it,
        such as H1 on the explainer, which compares results on perturbed panels with no series to match.
        ``twin`` adds the Matrix-free twin to every weighted ridge cell (:meth:`twin`); ``why`` and ``sample`` turn it on.
        """
        if arm not in WHY_ARMS:
            raise Refused(f"why explains {', '.join(WHY_ARMS)}; {arm} has no readout to explain" if arm == "climatology-panel"
                          else f"unknown arm {arm}")  # fmt: skip
        if checks not in CHECKS:
            raise ValueError(f"checks must be one of {CHECKS}")
        mc = self.mc
        result: dict[str, Any] = {"arm": arm}
        extra_rows, extra_what = None, ""
        rules: list[tuple[str, int, MonthStats]] = []
        jobs: list[dict[str, Any]] = []
        if arm == "matrix-stack":
            mo = self.month_choice(row, arm)
            rules.append((f"sigma and climatology of month {month_label(mo['fit_ms'])}", mo["fit_ms"], mo["stats"]))
            alpha = mo["alpha"]
            windows, parts, spans, tops = [], [], [], []
            for w in range(len(mc.windows_days)):
                weighted = bool(alpha[w] > 0)
                c = self.ridge_cell(row, w, mo["rho_by_window"][w], mo["var"], mo["clim"], by=by, top=top, full=full and weighted,
                                   twin=twin and weighted)
                parts.append(c["probs"][None, :])
                jobs.append(c)
                if weighted and c["weights"] is not None:
                    spans.append(c["span"])
                    tops += c["top_positions"]
                windows.append({"alpha": float(alpha[w]), **_cell_view(c)})
            probs = mix(alpha, parts)[0]
            result["choices"] = {"rho_by_window": {_wname(mc.windows_days[w]): mc.rhos[mo["rho_by_window"][w]] for w in range(len(windows))},
                                 "alpha": alpha.tolist(), "sigma": mo["sigma"], "fit": mo["fit"], "note": mo["note"]}  # fmt: skip
            result["windows"] = windows
            result["probs"] = probs
        else:
            if arm in KERNEL_ARMS:
                mo = self.month_choice(row, arm)
                rules.append((f"sigma and climatology of month {month_label(mo['fit_ms'])}", mo["fit_ms"], mo["stats"]))
                kind, idx = ("softmax", mo["tau"]) if arm == "softmax-1y" else ("knn", mo["k"])
                c = self.kernel_cell(row, kind, idx, mo["var"], mo["clim"], by=by, top=top, full=full)
                choices = {"window": _wname(mc.kernel_window_days), **({"tau_factor": mc.tau_factors[idx]} if kind == "softmax" else {"k": mc.ks[idx]}),
                           "sigma": mo["sigma"], "fit": mo["fit"], "note": mo["note"]}  # fmt: skip
            elif arm == "matrix-cusum":
                ch = self.cusum_choice(row)
                assert ch.stats is not None and ch.fit_ms is not None
                rules.append((f"sigma and climatology of {ch.sigma}", ch.fit_ms, ch.stats))
                c = self.ridge_cell(row, ch.window, ch.rho, ch.var, ch.clim, by=by, top=top, full=full, twin=twin)
                choices = {"window": _wname(mc.windows_days[ch.window]), "rho": mc.rhos[ch.rho], "sigma": ch.sigma, "fit": ch.fit, "note": ch.note}
            else:
                mo = self.month_choice(row, arm)
                rules.append((f"sigma and climatology of month {month_label(mo['fit_ms'])}", mo["fit_ms"], mo["stats"]))
                w, r = mo["select"]
                if arm == "matrix-1y":
                    w, r = mc.default_window, mo["rho_1y"]
                choices = {"sigma": mo["sigma"], "fit": mo["fit"], "note": mo["note"]}
                if arm == "matrix-adaptive" and mo["fit"] is None:
                    choices["adaptive"] = {"rows_scored": 0, "default": True, "scores": {}, "rule": ADAPTIVE_DEFAULT_NOTE}
                elif arm == "matrix-adaptive":
                    r, info, extra_rows = self.adaptive_rho(row, w)
                    extra_what = "rows behind the adaptive rho"
                    choices["adaptive"] = info
                    for mm in np.unique(self.system.data.month_ms[extra_rows]).tolist():
                        if int(mm) != mo["fit_ms"]:
                            rules.append((f"sigma and climatology of month {month_label(int(mm))} (rows behind the adaptive rho)", int(mm),
                                          self.system.month_stats(int(mm))))  # fmt: skip
                choices.update(window=_wname(mc.windows_days[w]), rho=mc.rhos[r])
                c = self.ridge_cell(row, w, r, mo["var"], mo["clim"], by=by, top=top, full=full, twin=twin)
            spans = [c["span"]] if c["weights"] is not None else []
            jobs.append(c)
            tops = c["top_positions"]
            result["choices"] = choices
            result.update(_cell_view(c))
            result["probs"] = c["probs"]
        if checks == "none":
            result["checks"] = {"skipped": "below the command line, for tests: no loud check ran"}
        else:
            stamps = self.stamp_check(row, spans, tops, extra_rows, extra_what, rules)
            result["checks"] = {"stamps": stamps}
        for c in jobs:  # the Matrix-free twin: after check 1 (a stamp leak is named as one), before check 2
            if "twin_job" in c:
                c["cell"]["twin"] = self.twin(*c.pop("twin_job"))
        if checks == "both":
            result["checks"]["forecast"] = self.forecast_check(row, arm, np.asarray(result["probs"], dtype=np.float64))
        result["row"] = self.row_view(row)
        return result

    def row_view(self, row: int) -> dict[str, Any]:
        d = self.system.data
        i = self.series_row(row)
        return {"pair": d.symbols[d.symbol[row]], "open_time": iso(int(d.open_ms[row])), "decision_time": iso(int(d.decision_ms[row])),
                "month": month_label(int(d.month_ms[row])), "scored": i is not None, "series_row": i,
                "label": int(d.label[row]) if d.label[row] >= 0 else None, "fwd_return": float(d.fwd[row]) if np.isfinite(d.fwd[row]) else None}  # fmt: skip


def row_key(symbol: np.ndarray, open_ms: np.ndarray) -> np.ndarray:
    """One int64 per (symbol code, bar open): symbol x 10^13 + open (opens are below 10^13 ms, codes below 10^5)."""
    return np.asarray(symbol, dtype=np.int64) * np.int64(10**13) + np.asarray(open_ms, dtype=np.int64)


def _parse_iso_ms(text: str) -> int:
    from datetime import datetime

    return int(datetime.fromisoformat(text).timestamp() * 1000)


def _cell_view(c: dict[str, Any]) -> dict[str, Any]:
    out = {"cell": c["cell"]}
    for key in ("diagnostics", "tree", "features", "top_entries"):
        if key in c:
            out[key] = c[key]
    return out


def diagnostics(w: np.ndarray, leverage: float | None) -> dict[str, Any]:
    """Sum w, sum |w|, the negative mass, the share of negative weights, the effective sample size and the leverage (fsum: exact sums, correctly rounded)."""
    w = np.ascontiguousarray(w, dtype=np.float64)
    neg = w[w < 0]
    sq = math.fsum((w * w).tolist())
    return {"n": int(w.size), "sum_w": math.fsum(w.tolist()), "sum_abs_w": math.fsum(np.abs(w).tolist()),
            "negative_mass": math.fsum((-neg).tolist()), "share_negative": float(neg.size / w.size) if w.size else None,
            "effective_sample_size": 1.0 / sq if sq > 0 else None, "leverage": leverage,
            "weights_sha256": hashlib.sha256(w.astype("<f8").tobytes()).hexdigest()}  # fmt: skip


def tree_view(groups: list[tuple[str, np.ndarray]], raw: np.ndarray, top: int, bound: np.ndarray, bound_name: str) -> dict[str, Any]:
    """Per output: the top-k groups by |c|, the rest summed in group order, and the closing residual to the raw readout.

    The parts, summed left to right in the order given (top groups, rest, closing residual), give the raw
    readout: ``closed`` is true by construction. What is checked is the residual's size against
    ``bound`` (per output); beyond it the call fails loudly (:func:`close_parts`).
    """
    out: dict[str, Any] = {"grouping": GROUPING_NOTE, "groups": len(groups), "bound": bound_name}
    for o, name in enumerate(OUTPUTS):
        vals = [float(c[o]) for _, c in groups]
        order = sorted(range(len(vals)), key=lambda i: (-abs(vals[i]), i))
        head = order[:top]
        tail = sorted(order[top:])
        rest = seq([vals[i] for i in tail])
        parts = [vals[i] for i in head] + ([rest] if tail else [])
        residual = close_parts(parts, float(raw[o]), float(bound[o]), f"output {name}")
        out[name] = {"top": [{"group": groups[i][0], "c": vals[i]} for i in head], "rest": rest if tail else None, "rest_groups": len(tail),
                     "closing_residual": residual, "closing_residual_size": math.fsum(abs(x) for x in residual),
                     "rounding_bound": float(bound[o]), "closed": True, "raw": float(raw[o])}  # fmt: skip
    return out


def feature_view(ro: Any, x_q: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    """mean_y, then beta_j * xt_j sorted by |part|; their sum in the core's order, and whether it gives the raw readout to the bit."""
    beta, mean_y, mean_x = np.asarray(ro.beta), np.asarray(ro.mean_y), np.asarray(ro.mean_x)
    xt = [float(a) - float(b) for a, b in zip(x_q, mean_x, strict=True)]
    out: dict[str, Any] = {"order": "products rounded one by one, summed j = 0, 1, ... in feature order, mean_y added last: the core's order"}
    for c, name in enumerate(OUTPUTS):
        parts = [float(beta[c, j]) * xt[j] for j in range(len(xt))]
        total = float(mean_y[c]) + seq(parts)
        order = sorted(range(len(parts)), key=lambda j: (-abs(parts[j]), j))
        out[name] = {"mean_y": float(mean_y[c]), "parts": [{"feature": FEATURE_COLUMNS[j], "part": parts[j], "beta": float(beta[c, j]), "xt": xt[j]}
                                                           for j in order],
                     "sum_in_core_order": total, "bit_exact": total == float(raw[c]) and math.copysign(1, total) == math.copysign(1, float(raw[c])),
                     "numpy_gap": float(abs(float(mean_y[c]) + float(beta[c] @ np.array(xt)) - float(raw[c])))}  # fmt: skip
    out["bit_exact_all"] = all(out[n]["bit_exact"] for n in OUTPUTS)
    out["statement"] = ("checked at this call: summed in the core's order, the per-feature parts give the raw readout to the bit" if out["bit_exact_all"]
                        else "checked at this call: summed in the core's order, the per-feature parts miss the raw readout by rounding (see each output)")  # fmt: skip
    return out
