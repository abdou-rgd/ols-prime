"""Report n°2: Matrix on the hourly bars of the universe members (``docs/HYPOTHESES.md``, H2, H3, H10-H12).

Usage::

    uv run --no-sync python -m olsprime.bench.report2 [--out DIR]
    uv run --no-sync python -m olsprime.bench.report2 --check DIR

One deterministic command, as pre-registered in ``docs/PROTOCOL.md``, "Matrix on hourly bars
(iteration 3)":

- **Pipeline.** The hourly panel (:func:`olsprime.features.panel.hourly_panel`, sealed, cut to
  the development period), the eight arms (:mod:`olsprime.models.matrix`), scored on the test
  months 2019-01 -> 2024-12.
- **Scored rows.** The test rows with a label and complete features, the same for every arm.
  Each score is averaged over the scored rows of each decision hour; every statistic is computed
  on these hourly series, and every reported mean is a mean over hours. The report states its
  coverage: the scored hours out of the test months' hours.
- **Statistics.** Log-loss, Brier, Brier skill score against ``climatology-panel`` and top-label
  ECE (10 equal-width bins, on the scored rows) per arm, each with a stationary-bootstrap 95 %
  interval (Politis-White block length, 2,000 resamples, percentile; the ECE resamples whole
  hours). For each pre-registered comparison: Diebold-Mariano on the hourly log-loss differences
  (two-sided, rectangular kernel, 23 lags, Harvey-Leybourne-Newbold), and the stationary bootstrap
  of the mean hourly difference with the Politis-White length of that difference, 2,000
  resamples, 95 % percentile interval. Each difference is also given as a share of climatology's
  log-loss, and each comparison gives the share of hours at which the two arms' losses are
  identical (above one half: "few informative hours"; descriptive).
- **Decision rules.** A comparison passes when DM gives p < 0.05 and the bootstrap interval lies
  on the same side of 0. H2: BSS of ``matrix-select`` above 0 with its interval above 0. H3:
  ``matrix-1y`` beats ``softmax-1y`` and ``knn-1y``, both passing. H10, H11: ``matrix-adaptive``,
  ``matrix-stack`` beat ``matrix-select``. H12: the one-sided 95 % upper bound of the mean
  difference ``matrix-cusum - matrix-select`` (mean + t_0.95(T - 1) x the HLN-corrected DM standard
  error) and the bootstrap's 95th percentile both below 0.005 x climatology's log-loss, and at most
  36 fits.
- **The previewed months, apart.** Descriptive, not a verdict: H2's BSS and every comparison on
  the test months that smoke runs scored before the registered run (through 2020-06), and on the
  others, each part with its own climatology and bootstrap seeds (:func:`preview_split`). A
  part's comparison says "rule met", never "passes". Under its verdicts, where they are read,
  the report says which test months the smoke runs scored.

Outputs: ``report-2.md``, ``report2.json``, ``config.json``, ``series.npz`` and ``run.json`` (the only
file with wall-clock times) to ``runs/<UTC date>-report2/`` (suffixed if taken) or ``--out``.

- ``series.npz`` holds the hourly series of every arm and every scored row's probabilities, label and
  times (:data:`SERIES_KEYS`), so that no later study refits an arm. Its bytes are the same on every
  re-run (:func:`write_npz`), and ``report2.json`` gives its SHA-256 under ``series``.
- ``--check DIR`` re-runs into a temporary folder and fails, with a per-file summary and a diff, if any
  file but ``run.json`` differs, ``series.npz`` included.
- ``--compare-with OLDER [--out DIR]`` re-runs once into a kept folder and compares it with an older
  run that predates ``series.npz`` (iteration 3's ``runs/2026-10-29-report2/``). Only the differences
  named in :data:`ITERATION3_ALLOWANCES`, each with its reason, are allowed; every other byte must match.
  ``--existing`` compares the folder at ``--out`` without re-running.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import struct
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as st

from olsprime.bench import provenance as prov_mod
from olsprime.data.binance_vision import Month
from olsprime.eval import stats
from olsprime.features.panel import FIRST_MONTH, LAST_MONTH, fingerprint, hourly_panel
from olsprime.models.matrix import (
    ARMS,
    DAY_MS,
    HOUR_MS,
    NOT_COMPUTED,
    STATUS_NAMES,
    ArmsResult,
    MatrixConfig,
    MatrixSystem,
    feature_ratios,
    fold_months_ms,
    iso,
    month_start_ms,
    next_month_ms,
)
from olsprime.paths import runs_dir
from olsprime.protocol import DEVELOPMENT

REPORT_FILE = "report-2.md"
SERIES_FILE = "series.npz"
COMPARED = ("config.json", "report2.json", REPORT_FILE, SERIES_FILE)
CLIMATOLOGY = "climatology-panel"
# The pre-registered comparisons: (hypothesis, arm a, arm b); the difference is a - b, a must be lower.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("H3", "matrix-1y", "softmax-1y"),
    ("H3", "matrix-1y", "knn-1y"),
    ("H10", "matrix-adaptive", "matrix-select"),
    ("H11", "matrix-stack", "matrix-select"),
    ("H12", "matrix-cusum", "matrix-select"),
)
VARIANTS_BEFORE = 10  # the baselines of Report n°1


@dataclass(frozen=True)
class Report2Config:
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    first_month: tuple[int, int] = (FIRST_MONTH.year, FIRST_MONTH.month)
    last_month: tuple[int, int] = (LAST_MONTH.year, LAST_MONTH.month)
    symbols: tuple[str, ...] | None = None  # None: every member (the registered run)
    n_boot: int = 2000
    level: float = 0.95
    bootstrap_seed: int = 20261029
    dm_h: int = 24  # h - 1 = 23 lags
    ece_bins: int = 10
    h12_margin: float = 0.005  # times climatology's log-loss
    readout_sample: int = 1000
    readout_seed: int = 20261029
    variants_before: int = VARIANTS_BEFORE
    data_end: str = DEVELOPMENT.end.isoformat()
    previewed_through: tuple[int, int] = (2020, 6)  # smoke runs scored the test months up to here first (PROTOCOL, after audit pass 1c)

    @property
    def registered(self) -> bool:
        return self == Report2Config()


def _seed(cfg: Report2Config, *parts: str) -> int:
    return zlib.crc32("|".join((str(cfg.bootstrap_seed), *parts)).encode())


# --- Hourly series --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Scored:
    """The scored rows and their hours."""

    rows: np.ndarray  # positions into ArmsResult.rows
    label: np.ndarray
    hour: np.ndarray  # index of each row's hour
    hours: np.ndarray  # epoch ms of each hour
    count: np.ndarray  # rows per hour


def scored_rows(res: ArmsResult, system: MatrixSystem) -> Scored:
    d = system.data
    rows = res.rows
    keep = d.complete[rows] & (d.label[rows] >= 0)
    for arm, p in res.probs.items():
        if np.isnan(p[keep]).any():
            raise ValueError(f"{arm} has no forecast at a scored row")
    pos = np.flatnonzero(keep)
    open_ms = d.open_ms[rows[pos]]
    hours, inv = np.unique(open_ms, return_inverse=True)
    return Scored(pos, d.label[rows[pos]], inv, hours, np.bincount(inv, minlength=hours.size))


def hourly(values: np.ndarray, sc: Scored) -> np.ndarray:
    return np.bincount(sc.hour, weights=values, minlength=sc.hours.size) / sc.count


def calendar_hours(mc: MatrixConfig) -> int:
    """The hours of the test months, scored or not (audit pass 1c: the report states its coverage)."""
    months = fold_months_ms(mc)
    return int((next_month_ms(months[-1]) - months[0]) // HOUR_MS)


def _chunks(t: int, block: float, n_boot: int, seed: int, size: int = 100):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        n = min(size, n_boot - done)
        yield stats.stationary_bootstrap_indices(t, block, n, seed=rng)
        done += n


def _block(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8 or np.all(x == x[0]):
        return 1
    return stats.optimal_block_length(x)


def _interval(draws: np.ndarray, estimate: float, level: float, block: int) -> dict[str, Any]:
    draws = draws[np.isfinite(draws)]
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [a, 1.0 - a])
    return {"estimate": float(estimate), "low": float(lo), "high": float(hi), "block": int(block), "n_boot": int(draws.size)}


def mean_interval(x: np.ndarray, cfg: Report2Config, *tag: str) -> dict[str, Any]:
    """Stationary bootstrap of the mean of an hourly series, Politis-White block of that series."""
    b = _block(x)
    draws = np.concatenate([x[idx].mean(axis=1) for idx in _chunks(x.size, b, cfg.n_boot, _seed(cfg, *tag))])
    out = _interval(draws, float(x.mean()), cfg.level, b)
    out["q95"] = float(np.quantile(draws, 0.95))
    return out


def bss_interval(b_arm: np.ndarray, b_clim: np.ndarray, cfg: Report2Config, *tag: str) -> dict[str, Any]:
    """Paired bootstrap of 1 - mean(b_arm*) / mean(b_clim*), same blocks, Politis-White length of b_arm - b_clim."""
    b = _block(b_arm - b_clim)
    parts = []
    for idx in _chunks(b_arm.size, b, cfg.n_boot, _seed(cfg, *tag)):
        parts.append(1.0 - b_arm[idx].mean(axis=1) / b_clim[idx].mean(axis=1))
    return _interval(np.concatenate(parts), 1.0 - b_arm.mean() / b_clim.mean(), cfg.level, b)


def ece_bins(p: np.ndarray, y: np.ndarray, sc: Scored, n_bins: int) -> np.ndarray:
    """Per hour and bin, sum(correct) - sum(confidence) of the top label: (T, n_bins)."""
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    b = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    key = sc.hour * n_bins + b
    out = np.bincount(key, weights=correct - conf, minlength=sc.hours.size * n_bins)
    return out.reshape(sc.hours.size, n_bins)


def ece_tolerance(n: int) -> float:
    """How far the ECE by hours and ``stats.ece`` may differ through rounding alone, over n rows.

    What the cross-check guards (``docs/PROTOCOL.md``, change log, 2026-10-29, after audit phase 2):

    - A row lost or counted twice between the hourly sums and ``stats.ece``. Such a row moves the
      ECE by about |correct - confidence| / n: for a row of climatology, whose confidence is about
      0.44, 220 to 285 times this tolerance on the panel. A confident and right forecast, whose
      confidence nearly equals its correctness, moves it by almost nothing.
    - Not a row put in another bin. Both paths bin the same way, and a row moved between two bins
      whose sums share a sign leaves the ECE unchanged: this check cannot see it.

    The bound. ``stats.ece`` counts each bin's correct rows exactly and adds its confidences in one
    sequential sum; divided by n, that sum's rounding moves the ECE by at most about
    u n mean(confidence), with u = eps / 2, so below n eps / 2. When one forecast repeats over many
    rows, as climatology's does within a fit, those errors add up instead of cancelling: 1.3e-9
    relative on the panel, which stopped the official run of 2026-10-29. The hourly path adds about
    30 rows per cell, then the cells over H hours: at most about H u. So 4 n eps is more than eight
    times the worst case while the mean confidence stays below 0.96.
    """
    return 4.0 * n * float(np.finfo(np.float64).eps)


def ece_interval(p: np.ndarray, y: np.ndarray, sc: Scored, brier_hourly: np.ndarray, cfg: Report2Config, *tag: str) -> dict[str, Any]:
    """Top-label ECE on the scored rows; the bootstrap resamples whole hours (block: PW of the hourly Brier)."""
    d = ece_bins(p, y, sc, cfg.ece_bins)
    est = float(np.abs(d.sum(axis=0)).sum() / sc.count.sum())
    check = stats.ece(p, y, cfg.ece_bins)
    if abs(est - check) > ece_tolerance(int(sc.count.sum())):
        raise AssertionError(f"ECE {est} differs from stats.ece {check} by more than rounding")
    b = _block(brier_hourly)
    parts = []
    t = sc.hours.size
    for idx in _chunks(t, b, cfg.n_boot, _seed(cfg, *tag)):
        for row in idx:  # no BLAS: plain numpy sums, the same bits on every re-run
            w = np.bincount(row, minlength=t).astype(np.float64)
            parts.append(np.abs((w[:, None] * d).sum(axis=0)).sum() / (w * sc.count).sum())
    return _interval(np.array(parts), est, cfg.level, b)


# --- Comparisons and rules ------------------------------------------------------------------


def dm_standard_error(dm: stats.DieboldMariano, d: np.ndarray, h: int) -> float:
    """The HLN-corrected standard error of the mean difference behind ``dm``: ``mean / statistic``."""
    if dm.statistic != 0.0:
        return abs(dm.mean_diff / dm.statistic)
    shifted = stats.diebold_mariano(d + 1.0, np.zeros_like(d), h=h)  # the variance does not move with the mean
    return abs(shifted.mean_diff / shifted.statistic)


def passes(dm_p: float, low: float, high: float, mean: float) -> bool:
    """A comparison where a must have the lower loss: DM p < 0.05, the mean below 0, the interval below 0."""
    return bool(dm_p < 0.05 and mean < 0.0 and high < 0.0)


def h12_rule_met(upper: float, q95: float, margin: float) -> bool:
    """H12's non-inferiority bounds: the one-sided 95 % upper bound and the bootstrap's 95th percentile both below the margin.

    The registered H12 also needs at most ``cusum_max_fits`` fits; that condition belongs to the whole period only.
    """
    return bool(upper < margin and q95 < margin)


def compare_arms(ll: dict[str, np.ndarray], a: str, b: str, ll_clim: float, cfg: Report2Config, *tag: str) -> dict[str, Any]:
    d = ll[a] - ll[b]
    dm = stats.diebold_mariano(ll[a], ll[b], h=cfg.dm_h)
    boot = mean_interval(d, cfg, "diff", a, b, *tag)
    se = dm_standard_error(dm, d, cfg.dm_h)
    t95 = float(st.t.ppf(0.95, df=d.size - 1))
    return {
        "a": a,
        "b": b,
        "mean_diff": float(d.mean()),
        "share_of_climatology": float(d.mean() / ll_clim),
        "dm_statistic": dm.statistic,
        "dm_p": dm.p_value,
        "dm_kernel": dm.kernel_used,
        "dm_lags": dm.h_used - 1,
        "dm_note": dm.note,
        "dm_se": se,
        "upper_95_one_sided": float(d.mean() + t95 * se),
        "bootstrap": boot,
        "passes": passes(dm.p_value, boot["low"], boot["high"], float(d.mean())),
        "hours": int(d.size),
        **identical_hours(d),
    }


FEW_INFORMATIVE = 0.5  # above this share of identical hours, a comparison rests on few informative hours (audit pass 1e)


def identical_hours(d: np.ndarray) -> dict[str, Any]:
    """The hours at which two arms' hourly log-losses are identical: both arms read the same cells at every scored row.

    Diebold-Mariano and the bootstrap then rest on the other hours alone, while their degrees of
    freedom count every hour. Descriptive: the verdicts do not use it.
    """
    same = d == 0
    return {"identical_hours": int(same.sum()), "share_of_hours_identical": float(same.mean())}


def _informative(c: dict[str, Any]) -> str:
    share = c["share_of_hours_identical"]
    return f"{share:.1%} ({c['hours'] - c['identical_hours']} informative)" + (", few informative hours" if share > FEW_INFORMATIVE else "")


MIN_SPLIT_HOURS = 100


def h12_part_rule(max_fits: int) -> str:
    """The rule an H12 part's ``rule_met`` reads, said in the JSON (audit phase 2, logged item 3)."""
    return (f"H12's bounds on this part: upper_95_one_sided and the bootstrap's q95 both below this part's h12_margin; "
            f"the condition on fits (at most {max_fits}) belongs to the whole period only")  # fmt: skip


def _compare_part(
    part: dict[str, np.ndarray], a: str, b: str, clim: float, cfg: Report2Config, name: str, *, h12_margin: float | None = None
) -> dict[str, Any]:
    """compare_arms on one part, or a record of why it has no test: a part can hold two arms that agree at every hour.

    A part describes, so its comparisons say ``rule_met``: "passes" is the registered rule's word,
    and the verdicts are those of the whole period (audit pass 1e). With ``h12_margin``, ``rule_met``
    is H12's own rule on that part's margin, not the superiority rule (audit phase 2, logged item 3).
    """
    d = part[a] - part[b]
    if np.all(d == d[0]):  # e.g. matrix-adaptive taking matrix-select's rho at every decision: Diebold-Mariano is undefined
        return {"a": a, "b": b, "mean_diff": float(d.mean()), "hours": int(d.size), **identical_hours(d),
                "undefined": "the difference is the same at every hour"}  # fmt: skip
    c = compare_arms(part, a, b, clim, cfg, name)
    generic = c.pop("passes")
    if h12_margin is None:
        c["rule_met"] = generic
    else:
        c["rule_met"] = h12_rule_met(c["upper_95_one_sided"], c["bootstrap"]["q95"], h12_margin)
        c["rule"] = h12_part_rule(cfg.matrix.cusum_max_fits)
    return c


def preview_split(ll: dict[str, np.ndarray], br: dict[str, np.ndarray], sc: Scored, cfg: Report2Config) -> dict[str, Any]:
    """Descriptive, not a verdict: H2's BSS and every comparison on the previewed test months alone, and on the others alone.

    Smoke runs scored the test months through ``cfg.previewed_through`` before the registered run,
    with the same arms and rules (``docs/PROTOCOL.md``, change log, after audit pass 1c). The two
    parts show whether a result rests on those months. Each part has its own climatology
    log-loss for the shares, and its own bootstrap seeds.
    """
    end = next_month_ms(month_start_ms(*cfg.previewed_through))
    months = fold_months_ms(cfg.matrix)
    out: dict[str, Any] = {"previewed_through": "%d-%02d" % cfg.previewed_through, "previewed_months": sum(1 for m in months if m < end),
                           "test_months": len(months)}  # fmt: skip
    for name, keep in (("previewed", sc.hours < end), ("unseen", sc.hours >= end)):
        if int(keep.sum()) < MIN_SPLIT_HOURS:
            out[name] = None
            continue
        part = {a: v[keep] for a, v in ll.items()}
        clim = float(part[CLIMATOLOGY].mean())
        out[name] = {
            "hours": int(keep.sum()),
            "first_hour": iso(int(sc.hours[keep][0])),
            "last_hour": iso(int(sc.hours[keep][-1])),
            "climatology_log_loss": clim,
            "h12_margin": cfg.h12_margin * clim,
            "bss_matrix_select": bss_interval(br["matrix-select"][keep], br[CLIMATOLOGY][keep], cfg, "bss", "matrix-select", name),
            "comparisons": [dict(_compare_part(part, a, b, clim, cfg, name, h12_margin=cfg.h12_margin * clim if h == "H12" else None),
                                 hypothesis=h) for h, a, b in COMPARISONS],  # fmt: skip
        }
    return out


def cusum_summary(cusum: dict[str, Any], mc: MatrixConfig) -> dict[str, Any]:
    """matrix-cusum's fits and counts, with the day each refit left out of the restarted CUSUM (audit pass 1c).

    A refit at F leaves out the day decided at ``F - 1 day``, whose value would enter after F. That
    day is in the CUSUM only if it has a complete row at 00:00 UTC; when it has none, there is
    nothing to leave out, and the fit's ``day_left_out`` is None.
    """
    fits, path = cusum["fits"], cusum["path"]
    skipped = sorted(int(x) for x in cusum["skipped_days"])
    rows = []
    for i, f in enumerate(fits):
        before = int(f["fit_ms"]) - DAY_MS
        rows.append({"fit": iso(f["fit_ms"]), "window": mc.windows_days[f["window"]], "rho": mc.rhos[f["rho"]],
                     "s": f["s"], "s_days": f["s_days"], "s_default": f["s_default"], "default": f["default"],
                     "n_selection": f["n_selection"], "day_left_out": iso(before)[:10] if i and before in skipped else None})  # fmt: skip
    return {
        "threshold": mc.cusum_threshold,
        "k": mc.cusum_k,
        "fits": rows,
        "days": int(path["day_ms"].size),
        "days_left_out_at_refits": len(skipped),
        "days_left_out": [iso(x)[:10] for x in skipped],
        "refits_without_a_day_left_out": sum(1 for r in rows[1:] if r["day_left_out"] is None),
        "fits_with_s_set_to_1": sum(1 for f in fits if f["s_default"]),
        "alarms": int(path["alarm"].sum()) if path["alarm"].size else 0,
    }


def fallbacks_by_status(res: ArmsResult, arm: str, sc: Scored) -> dict[str, int]:
    """The scored rows where ``arm`` fell back to climatology, counted by the status of the cell it read."""
    code = res.status[arm][sc.rows][res.fallback[arm][sc.rows]]
    return {STATUS_NAMES[c]: int((code == c).sum()) for c in STATUS_NAMES if c not in (0, NOT_COMPUTED)}


def verdicts(metrics: dict[str, Any], comparisons: list[dict[str, Any]], n_cusum_fits: int, cfg: Report2Config) -> dict[str, Any]:
    """Each hypothesis by its pre-registered rule."""
    by = {(c["a"], c["b"]): c for c in comparisons}
    bss = metrics["matrix-select"]["bss"]
    margin = cfg.h12_margin * metrics[CLIMATOLOGY]["log_loss"]["estimate"]
    h12 = by[("matrix-cusum", "matrix-select")]
    h3 = [by[("matrix-1y", "softmax-1y")], by[("matrix-1y", "knn-1y")]]
    return {
        "H2": {"holds": bool(bss["estimate"] > 0 and bss["low"] > 0), "rule": "BSS of matrix-select > 0, its 95 % interval above 0"},
        "H3": {"holds": all(c["passes"] for c in h3), "each": {c["b"]: c["passes"] for c in h3},
               "rule": "matrix-1y below softmax-1y and knn-1y, each comparison passing"},  # fmt: skip
        "H10": {"holds": by[("matrix-adaptive", "matrix-select")]["passes"], "rule": "matrix-adaptive below matrix-select, passing"},
        "H11": {"holds": by[("matrix-stack", "matrix-select")]["passes"], "rule": "matrix-stack below matrix-select, passing"},
        "H12": {
            "holds": h12_rule_met(h12["upper_95_one_sided"], h12["bootstrap"]["q95"], margin) and n_cusum_fits <= cfg.matrix.cusum_max_fits,
            "margin": margin,
            "upper_95_one_sided": h12["upper_95_one_sided"],
            "bootstrap_q95": h12["bootstrap"]["q95"],
            "fits": n_cusum_fits,
            "max_fits": cfg.matrix.cusum_max_fits,
            "rule": "upper bound and bootstrap q95 of mean(cusum - select) below 0.005 x climatology's log-loss, and at most 36 fits",
        },
    }


# --- The descriptive readout ----------------------------------------------------------------


def describe_readout(system: MatrixSystem, res: ArmsResult, sc: Scored, cfg: Report2Config) -> dict[str, Any]:
    """For a seeded sample of matrix-1y's scored decisions: negative weights and the contributions of the last 30 days and older."""
    d = system.data
    mc = cfg.matrix
    n = min(cfg.readout_sample, sc.rows.size)
    rng = np.random.default_rng(cfg.readout_seed)
    pick = np.sort(rng.choice(sc.rows.size, size=n, replace=False))
    fit_by_month = {f["month_ms"]: f for f in res.fits}
    neg, total, shares = 0, 0, []
    recent, older, gap = [], [], 0.0
    refused = 0
    for i in pick:
        row = int(res.rows[sc.rows[i]])
        m = int(d.month_ms[row])
        fit = fit_by_month[m]
        var = system.month_stats(m).var
        rho = mc.rhos[fit["rho_1y"]]
        at = int(d.decision_ms[row])
        s, e = system.matrix.window_range(at, mc.kernel_window_days * DAY_MS)
        s30, _ = system.matrix.window_range(at, 30 * DAY_MS)
        try:
            r = system.matrix.centred_query(np.ascontiguousarray(d.x[row]), s, e, rho, var)
        except ValueError:
            refused += 1
            continue
        w = system.matrix.centred_weights(r)
        neg += int((w < 0).sum())
        total += int(w.size)
        shares.append(float((w < 0).mean()))
        c_new = system.matrix.centred_contribution(r, s30, e)
        c_old = system.matrix.centred_contribution(r, s, s30)
        recent.append(c_new)
        older.append(c_old)
        gap = max(gap, float(np.abs(c_new + c_old - r.forecast).max()))
    names = ["down", "neutral", "up", "fwd_return"]
    rec, old = np.array(recent), np.array(older)
    return {
        "decisions": int(n),
        "refused": refused,
        "share_negative_pooled": neg / total if total else None,
        "share_negative_mean": float(np.mean(shares)) if shares else None,
        "share_negative_quantiles": [float(q) for q in np.quantile(shares, [0.05, 0.5, 0.95])] if shares else None,
        "contribution_last_30_days_mean": dict(zip(names, rec.mean(axis=0).tolist(), strict=True)) if rec.size else None,
        "contribution_older_mean": dict(zip(names, old.mean(axis=0).tolist(), strict=True)) if old.size else None,
        "abs_contribution_last_30_days_mean": dict(zip(names, np.abs(rec).mean(axis=0).tolist(), strict=True)) if rec.size else None,
        "abs_contribution_older_mean": dict(zip(names, np.abs(old).mean(axis=0).tolist(), strict=True)) if old.size else None,
        "max_partition_error": gap,
    }


RATIO_ALERT = 1e3  # audit pass 1b: an r above this would degrade windows to climatology through the refusal bound


def feature_ratio_summary(system: MatrixSystem, res: ArmsResult) -> dict[str, Any]:
    """Each feature's r = |mean| / std over the panel's complete rows, and its largest value over the folds' sigma sets."""
    d = system.data
    panel_r = feature_ratios(d.x[d.complete])
    fold_r = np.zeros(len(panel_r))
    for st_ in res.months.values():
        if st_.var is None or st_.entries < 2:
            continue
        c = system.matrix.centred_stats(0, st_.entries)
        fold_r = np.maximum(fold_r, np.abs(c.mean_x) / np.sqrt(st_.var))
    names = list(panel_r)
    worst = max(max(panel_r.values()), float(fold_r.max(initial=0.0)))
    return {"panel": panel_r, "max_over_folds": dict(zip(names, fold_r.tolist(), strict=True)), "largest": worst,
            "alert_above": RATIO_ALERT, "alert": bool(worst > RATIO_ALERT)}  # fmt: skip


# --- The report -----------------------------------------------------------------------------


def compute(panel: pl.DataFrame, cfg: Report2Config) -> tuple[dict[str, Any], dict[str, float], dict[str, np.ndarray]]:
    """Every number of Report n°2 from a panel, the time of each stage (seconds) and the arrays of ``series.npz``."""
    timings: dict[str, float] = {}
    t = time.perf_counter()
    system = MatrixSystem(panel, cfg.matrix)
    timings["memory"] = time.perf_counter() - t
    t = time.perf_counter()
    res = system.run()
    timings["arms"] = time.perf_counter() - t
    t = time.perf_counter()
    sc = scored_rows(res, system)
    ll, br, metrics = {}, {}, {}
    for arm in ARMS:
        p = res.probs[arm][sc.rows]
        losses = stats.per_row_losses(p, sc.label)
        ll[arm], br[arm] = hourly(losses.log_loss, sc), hourly(losses.brier, sc)
    for arm in ARMS:
        p = res.probs[arm][sc.rows]
        metrics[arm] = {
            "log_loss": mean_interval(ll[arm], cfg, "ll", arm),
            "brier": mean_interval(br[arm], cfg, "brier", arm),
            "bss": bss_interval(br[arm], br[CLIMATOLOGY], cfg, "bss", arm) if arm != CLIMATOLOGY else None,
            "ece": ece_interval(p, sc.label, sc, br[arm], cfg, "ece", arm),
            "fallbacks": int(res.fallback[arm][sc.rows].sum()),
            "fallbacks_by_status": fallbacks_by_status(res, arm, sc),
        }
    ll_clim = metrics[CLIMATOLOGY]["log_loss"]["estimate"]
    comparisons = [dict(compare_arms(ll, a, b, ll_clim, cfg), hypothesis=h) for h, a, b in COMPARISONS]
    split = preview_split(ll, br, sc, cfg)
    cusum_fits = res.cusum["fits"]
    timings["statistics"] = time.perf_counter() - t
    t = time.perf_counter()
    readout = describe_readout(system, res, sc, cfg)
    timings["readout"] = time.perf_counter() - t
    d = system.data
    out = {
        "registered": cfg.registered,
        "panel": {"rows": int(d.n), "complete": int(d.complete.sum()), "members": len(d.symbols), "fingerprint_sha256": fingerprint(panel),
                  "entries": int(system.entry_rows.size), "symbols": list(d.symbols)},  # fmt: skip
        "scored": {"rows": int(sc.rows.size), "hours": int(sc.hours.size), "first_hour": iso(int(sc.hours[0])), "last_hour": iso(int(sc.hours[-1])),
                   "calendar_hours": calendar_hours(cfg.matrix), "share_of_calendar_hours": sc.hours.size / calendar_hours(cfg.matrix)},  # fmt: skip
        "arms": metrics,
        "comparisons": comparisons,
        "verdicts": verdicts(metrics, comparisons, len(cusum_fits), cfg),
        "preview_split": split,
        "fallbacks": {arm: metrics[arm]["fallbacks"] for arm in ARMS},
        "fallbacks_by_status": {arm: metrics[arm]["fallbacks_by_status"] for arm in ARMS},
        "record_fallbacks": res.record_fallbacks,
        "status_counts": res.status_counts,
        "record_status": res.record_status,
        "feature_ratio": feature_ratio_summary(system, res),
        "defaults": res.defaults,
        "cusum": cusum_summary(res.cusum, cfg.matrix),
        "fits": [
            {"month": iso(f["month_ms"])[:7], "n_selection": f["n_selection"], "default": f["default"],
             "select": {"window": cfg.matrix.windows_days[f["select"][0]], "rho": cfg.matrix.rhos[f["select"][1]]},
             "rho_1y": cfg.matrix.rhos[f["rho_1y"]], "tau_factor": cfg.matrix.tau_factors[f["tau"]], "k": cfg.matrix.ks[f["k"]],
             "stack_rho": [cfg.matrix.rhos[r] for r in f["rho_by_window"]], "alpha": [float(a) for a in f["alpha"]]}
            for f in res.fits
        ],  # fmt: skip
        "adaptive": {
            "decisions": int(res.adaptive["choice"].size),
            "rho_counts": {f"{cfg.matrix.rhos[i]:g}": int((res.adaptive["choice"] == i).sum()) for i in range(len(cfg.matrix.rhos))},
        },
        "readout": readout,
        "variants": {"before": cfg.variants_before, "this_report": len(ARMS), "total": cfg.variants_before + len(ARMS)},
    }
    return out, timings, series_arrays(system, res, sc, ll, br)


# --- series.npz -----------------------------------------------------------------------------

# The keys of series.npz, in the archive's order, with their dtypes. A = arms (ARMS order), T = scored hours,
# N = scored rows (in the order of Scored.rows, the panel's order), S = symbols.
SERIES_KEYS: tuple[tuple[str, str, str], ...] = (
    ("arms", "<U", "(A,) the arms' names, ARMS order"),
    ("hours", "<i8", "(T,) epoch ms of each scored hour: the open of its bars"),
    ("count", "<i8", "(T,) scored rows per hour"),
    ("log_loss", "<f8", "(A, T) hourly mean log-loss of each arm"),
    ("brier", "<f8", "(A, T) hourly mean Brier score of each arm"),
    ("probs", "<f8", "(A, N, 3) every arm's probabilities at every scored row, the exact values scored"),
    ("label", "<i8", "(N,) the class: 0 down, 1 neutral, 2 up"),
    ("fwd_return", "<f8", "(N,) the realized forward return behind the label"),
    ("symbol", "<i8", "(N,) code into symbols"),
    ("symbols", "<U", "(S,) the symbols' names, sorted"),
    ("open_time", "<i8", "(N,) epoch ms of the row's bar open"),
    ("decision_time", "<i8", "(N,) epoch ms of the row's decision"),
    ("label_known_at", "<i8", "(N,) epoch ms at which the row's label is known"),
    ("hour", "<i8", "(N,) index of the row's hour into hours"),
)


def series_arrays(system: MatrixSystem, res: ArmsResult, sc: Scored, ll: dict[str, np.ndarray], br: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The arrays of ``series.npz``: the hourly series the statistics read, and the scored rows behind them."""
    d = system.data
    rows = res.rows[sc.rows]  # panel indices of the scored rows
    out = {
        "arms": np.array(ARMS),
        "hours": sc.hours,
        "count": sc.count,
        "log_loss": np.stack([ll[a] for a in ARMS]),
        "brier": np.stack([br[a] for a in ARMS]),
        "probs": np.stack([res.probs[a][sc.rows] for a in ARMS]),
        "label": sc.label,
        "fwd_return": d.fwd[rows],
        "symbol": d.symbol[rows],
        "symbols": np.array(d.symbols),
        "open_time": d.open_ms[rows],
        "decision_time": d.decision_ms[rows],
        "label_known_at": d.lka_ms[rows],
        "hour": sc.hour,
    }
    return {k: np.ascontiguousarray(out[k], dtype=out[k].dtype if kind == "<U" else np.dtype(kind)) for k, kind, _ in SERIES_KEYS}


_ZIP_DATE = (0 << 9) | (1 << 5) | 1  # 1980-01-01, the first date a zip entry can hold
_ZIP_TIME = 0
_ZIP_LIMIT = 0xFFFFFFFF  # no zip64: a larger archive is refused, not written differently


class _CrcWriter:
    """A file-like sink for ``numpy.lib.format.write_array`` that keeps the CRC-32 and size of what it passes on."""

    def __init__(self, f: Any) -> None:
        self.f, self.crc, self.size = f, 0, 0

    def write(self, data: bytes) -> int:
        view = memoryview(data).cast("B")
        self.crc = zlib.crc32(view, self.crc)
        self.size += view.nbytes
        return self.f.write(view)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write ``arrays`` as an ``.npz`` that ``numpy.load`` reads, with the same bytes on every run.

    ``np.savez`` stamps each entry with the current local time, and ``zipfile`` records the platform
    that wrote it. This writes the zip by hand instead: entries in the order of ``arrays``, stored
    without compression, every date 1980-01-01 00:00, version 2.0 (MS-DOS), no extra field, no
    comment, attributes 0, and each member a version 1.0 ``.npy`` without pickles. It writes no
    zip64 record and refuses an archive that would need one (4 GiB).
    """
    entries: list[tuple[bytes, int, int, int]] = []
    with open(path, "wb") as f:
        for name, arr in arrays.items():
            fname = f"{name}.npy".encode("ascii")
            offset = f.tell()
            f.write(struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 0, _ZIP_TIME, _ZIP_DATE, 0, 0, 0, len(fname), 0) + fname)
            sink = _CrcWriter(f)
            np.lib.format.write_array(sink, np.ascontiguousarray(arr), version=(1, 0), allow_pickle=False)
            end = f.tell()
            if sink.size >= _ZIP_LIMIT or end >= _ZIP_LIMIT:
                raise ValueError(f"{path.name}: {name} takes the archive past 4 GiB, which needs zip64")
            f.seek(offset + 14)
            f.write(struct.pack("<III", sink.crc, sink.size, sink.size))
            f.seek(end)
            entries.append((fname, sink.crc, sink.size, offset))
        start = f.tell()
        for fname, crc, size, offset in entries:
            f.write(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 0, _ZIP_TIME, _ZIP_DATE, crc, size, size,
                                len(fname), 0, 0, 0, 0, 0, offset) + fname)  # fmt: skip
        size = f.tell() - start
        if f.tell() >= _ZIP_LIMIT or len(entries) > 0xFFFF:
            raise ValueError(f"{path.name}: the archive needs zip64")
        f.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), size, start, 0))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def config_record(cfg: Report2Config, provenance: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    import olsprime_rs
    import scipy

    return {
        "report": "report-2",
        "config": asdict(cfg),
        "registered": cfg.registered,
        "arms": list(ARMS),
        "comparisons": [list(c) for c in COMPARISONS],
        "versions": {"numpy": np.__version__, "polars": pl.__version__, "scipy": scipy.__version__,
                     "olsprime_rs": olsprime_rs.__version__, "python": sys.version.split()[0]},  # fmt: skip
        "provenance": provenance,
        "manifest": manifest,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(_clean(obj), sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.write_bytes(_json_bytes(obj))


def _f(x: float | None, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(ci: dict[str, Any] | None, digits: int = 4) -> str:
    if ci is None:
        return "-"
    return f"{ci['estimate']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


def _preview_lines(m: dict[str, Any]) -> list[str]:
    """The previewed test months and the others, apart (descriptive)."""
    ps = m["preview_split"]
    parts = [(name, ps[name]) for name in ("previewed", "unseen")]
    head = [f"{name}, {p['first_hour'][:7]} -> {p['last_hour'][:7]}, {p['hours']} hours" if p else f"{name}: too few hours" for name, p in parts]
    lines = [
        "## The previewed months, apart",
        "",
        f"Before this run, smoke runs on the real panel scored {ps['previewed_months']} of the {ps['test_months']} test months, through "
        f"{ps['previewed_through']}, with the same arms and rules. No rule changed after them (`docs/PROTOCOL.md`, change log, "
        "2026-10-29, after audit pass 1c). Each number below is computed on those months alone and on the others alone, with the "
        "same statistics. It is descriptive: the verdicts are those of the whole period, above.",
        "",
        f"| quantity | {head[0]} | {head[1]} |",
        "|---|---|---|",
        "| BSS of matrix-select | " + " | ".join(_ci(p["bss_matrix_select"]) if p else "-" for _, p in parts) + " |",
    ]
    for i, (h, a, b) in enumerate(COMPARISONS):
        cells = []
        for _, p in parts:
            if p is None:
                cells.append("-")
                continue
            c = p["comparisons"][i]
            if "undefined" in c:
                cells.append(f"{c['mean_diff']:+.5f}; no test: {c['undefined']}")
                continue
            boot = c["bootstrap"]
            word = "rule met" if c["rule_met"] else "rule not met"
            if h == "H12":
                cells.append(f"{c['mean_diff']:+.5f}; upper bound {c['upper_95_one_sided']:+.5f}, bootstrap q95 {boot['q95']:+.5f}, "
                             f"margin {p['h12_margin']:.5f}{H12_PART_WORD.format(word=word)}")  # fmt: skip
            else:
                cells.append(f"{c['mean_diff']:+.5f} [{boot['low']:+.5f}, {boot['high']:+.5f}], DM p {c['dm_p']:.4f}, {word}")
            cells[-1] += f"; identical hours {_informative(c)}"
        lines.append(f"| {h}: {a} - {b} | " + " | ".join(cells) + " |")
    return [*lines, "", H12_PART_NOTE.format(max_fits=m["verdicts"]["H12"]["max_fits"]), ""]


# Audit phase 2, logged item 3: H12's parts read H12's own rule, and the report says so under the table.
H12_PART_WORD = ", {word}"
H12_PART_NOTE = ("H12's parts read H12's own rule: the one-sided upper bound and the bootstrap's 95th percentile both below that "
                 "part's margin. Its condition on fits, at most {max_fits}, belongs to the whole period only.")  # fmt: skip

# Audit phase 2, logged item 1: a label reads only its entry and exit bars (RESULTS.md, Report n°2's row, the correction).
COVERAGE_BEFORE_ITERATION3 = "and for the 25 bars before it, whose labels need a bar inside it."
COVERAGE_BEFORE = ("and for 1 + g rows of each member before a gap of g bars (25 at most): a label reads only its entry and exit "
                   "bars, so the row just before the gap, whose entry bar falls in it, and the g rows 25 bars before the gap's "
                   "bars, whose exit bars fall in it, have none.")  # fmt: skip


def coverage_line(m: dict[str, Any], before: str = COVERAGE_BEFORE) -> str:
    s = m["scored"]
    return (
        f"Coverage: {s['hours']} of the test months' {s['calendar_hours']} hours are scored ({s['share_of_calendar_hours']:.1%}). "
        "An hour is scored when a member has a scored row in it. A row is not scored when a feature or its label is missing: for "
        f"about 90 bars after a gap in a pair's bars, while the features' longest window refills, {before} An exchange outage "
        "makes that gap for every member at once, so every verdict below is conditional on the hours away from outages."
    )


def series_line(series: dict[str, Any]) -> str:
    return f"- Hourly series and scored rows (`{series['file']}`, every arm's probabilities included), SHA-256: `{series['sha256']}`."


def render(m: dict[str, Any], config: dict[str, Any]) -> str:
    v = m["verdicts"]
    word = {True: "holds", False: "does not hold"}
    lines = ["# Report n°2: Matrix on hourly bars", ""]
    if not m["registered"]:
        lines += ["**NOT THE REGISTERED RUN.** This run's configuration differs from the pre-registered one "
                  "(a smoke run or a test): its numbers are not results.", ""]  # fmt: skip
    lines += [
        "Pre-registered in `docs/PROTOCOL.md`, \"Matrix on hourly bars (iteration 3)\". Development period only; the "
        "validation year and the vault stay closed.",
        "",
        "## Verdicts",
        "",
        "| hypothesis | verdict | rule |",
        "|---|---|---|",
        *(f"| {h} | {word[v[h]['holds']]} | {v[h]['rule']} |" for h in ("H2", "H3", "H10", "H11", "H12")),
        "",
        f"Before this run, smoke runs on the real panel scored {m['preview_split']['previewed_months']} of the "
        f"{m['preview_split']['test_months']} test months, through {m['preview_split']['previewed_through']}, with the same arms and "
        "rules, and no rule changed after them (`docs/PROTOCOL.md`, change log, 2026-10-29, after audit pass 1c). Those months and "
        "the others are shown apart below, as a description.",
        "",
        f"Panel: {m['panel']['rows']} rows, {m['panel']['complete']} complete, {m['panel']['members']} members, "
        f"{m['panel']['entries']} entries; fingerprint `{m['panel']['fingerprint_sha256']}`. Scored: {m['scored']['rows']} rows over "
        f"{m['scored']['hours']} decision hours ({m['scored']['first_hour'][:13]} -> {m['scored']['last_hour'][:13]}). Every mean is "
        "a mean over hours of the hourly means; intervals are stationary-bootstrap 95 % percentile intervals.",
        "",
        coverage_line(m),
        "",
        "## Arms",
        "",
        "| arm | log-loss | Brier | BSS vs climatology | ECE | fallbacks | defaults |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        a = m["arms"][arm]
        lines.append(f"| {arm} | {_ci(a['log_loss'])} | {_ci(a['brier'])} | {_ci(a['bss'])} | {_ci(a['ece'])} | {a['fallbacks']} | "
                     f"{m['defaults'].get(arm, '-')} |")  # fmt: skip
    lines += [
        "",
        "## Comparisons",
        "",
        "Difference = a - b in mean hourly log-loss (negative: a is better).",
        "",
        "| hypothesis | a | b | difference | share of climatology | DM statistic | DM p | DM kernel | bootstrap 95 % | passes | identical hours |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in m["comparisons"]:
        b = c["bootstrap"]
        lines.append(
            f"| {c['hypothesis']} | {c['a']} | {c['b']} | {c['mean_diff']:+.5f} | {c['share_of_climatology']:+.4%} | "
            f"{c['dm_statistic']:+.3f} | {c['dm_p']:.4f} | {c['dm_kernel']}, {c['dm_lags']} lags | [{b['low']:+.5f}, {b['high']:+.5f}] | "
            f"{'yes' if c['passes'] else 'no'} | {_informative(c)} |"
        )
    lines += ["", f"Identical hours: the share of hours at which both arms' hourly log-losses are equal, because they read the same "
              f"cells at every scored row. Above {FEW_INFORMATIVE:.0%}, the tests rest on few informative hours; this is descriptive "
              "and changes no verdict."]  # fmt: skip
    notes = [f"{c['a']} - {c['b']}: {c['dm_note']}" for c in m["comparisons"] if c["dm_note"]]
    lines += ["", "Diebold-Mariano fallbacks: " + ("; ".join(notes) if notes else "none") + "."]
    h12 = v["H12"]
    lines += [
        "",
        "## H12",
        "",
        f"Margin 0.005 x climatology's log-loss = {h12['margin']:.5f}. One-sided 95 % upper bound of mean(cusum - select): "
        f"{h12['upper_95_one_sided']:+.5f}; bootstrap 95th percentile: {h12['bootstrap_q95']:+.5f}. Fits: {h12['fits']} "
        f"(at most {h12['max_fits']}).",
        "",
        f"CUSUM: two-sided Page, k = {m['cusum']['k']}, threshold {m['cusum']['threshold']:.3f} (ARL 365 days under iid N(0, 1)); "
        f"{m['cusum']['days']} daily values, {m['cusum']['alarms']} alarms; {m['cusum']['days_left_out_at_refits']} days decided "
        f"just before a refit left out of the restarted CUSUM ({', '.join(m['cusum']['days_left_out']) or 'none'}); "
        f"{m['cusum']['refits_without_a_day_left_out']} refits had no such day to leave out, because it had no complete row at "
        f"00:00 UTC; {m['cusum']['fits_with_s_set_to_1']} of {len(m['cusum']['fits'])} fits with s set to 1 (fewer than two daily "
        "values, or s = 0). A day's value is made entirely by the fit in force at its decisions: readouts, residual mean square "
        "rss / (n_w - 16), leverage and s, which is read the same way over the days of the fit's selection rows.",
        "",
        "| fit | window (days) | rho | s | days behind s | s set to 1 | default | day left out |",
        "|---|---|---|---|---|---|---|---|",
        *(f"| {f['fit'][:10]} | {f['window'] or 'all'} | {f['rho']:g} | {f['s']:.4f} | {f['s_days']} | {f['s_default']} | {f['default']} | "
          f"{f['day_left_out'] or ('-' if i == 0 else 'none')} |" for i, f in enumerate(m["cusum"]["fits"])),  # fmt: skip
        "",
        *_preview_lines(m),
        "## Counts",
        "",
        "Fallbacks to climatology at the scored rows, by the status of the cell read:",
        "",
        "| arm | fallbacks | empty | singular | non-finite | too few |",
        "|---|---|---|---|---|---|",
        *(f"| {arm} | {m['fallbacks'][arm]} | {c.get('empty', 0)} | {c.get('singular', 0)} | {c.get('non-finite', 0)} | {c.get('too-few', 0)} |"
          for arm, c in m["fallbacks_by_status"].items()),  # fmt: skip
        "",
        f"Defaults (selections with fewer than 1,000 rows): {', '.join(f'{k} {n}' for k, n in m['defaults'].items())}.",
        "",
        "Refused cells in the prequential record (all rows from 2017-11), per candidate: "
        + ", ".join(f"{k} {n}" for k, n in m["record_fallbacks"].items() if n)
        + ("" if any(m["record_fallbacks"].values()) else "none")
        + ".",
        "",
        "Status of the cell each scored output read (empty and singular are refusals; singular is the refusal bound of "
        "`docs/DESIGN.md` §3.2):",
        "",
        "| arm | ok | empty | singular | non-finite | too few |",
        "|---|---|---|---|---|---|",
        *(f"| {arm} | {c.get('ok', 0)} | {c.get('empty', 0)} | {c.get('singular', 0)} | {c.get('non-finite', 0)} | {c.get('too-few', 0)} |"
          for arm, c in m["status_counts"].items()),  # fmt: skip
        "",
        f"r = |mean| / std of each feature (the refusal bound grows with it): largest {m['feature_ratio']['largest']:.3g}"
        + (f", ABOVE {m['feature_ratio']['alert_above']:g}" if m["feature_ratio"]["alert"] else "") + ".",
        "",
        "| feature | r over the panel | largest r over the folds' sigma sets |",
        "|---|---|---|",
        *(f"| {k} | {m['feature_ratio']['panel'][k]:.3g} | {m['feature_ratio']['max_over_folds'][k]:.3g} |" for k in m["feature_ratio"]["panel"]),
        "",
        f"matrix-adaptive's rho over {m['adaptive']['decisions']} decision times: "
        + ", ".join(f"{k}: {n}" for k, n in m["adaptive"]["rho_counts"].items())
        + ".",
        "",
        "## The readout, described",
        "",
    ]
    r = m["readout"]
    if r["share_negative_pooled"] is not None:
        lines += [
            f"{r['decisions']} seeded scored decisions of matrix-1y ({r['refused']} refused). Share of negative weights: "
            f"{r['share_negative_pooled']:.4f} pooled, {r['share_negative_mean']:.4f} on average per decision "
            f"(5 %, 50 %, 95 %: {', '.join(f'{q:.3f}' for q in r['share_negative_quantiles'])}).",
            "",
            "| output | last 30 days, mean | older, mean | last 30 days, mean abs | older, mean abs |",
            "|---|---|---|---|---|",
            *(f"| {k} | {r['contribution_last_30_days_mean'][k]:+.5f} | {r['contribution_older_mean'][k]:+.5f} | "
              f"{r['abs_contribution_last_30_days_mean'][k]:.5f} | {r['abs_contribution_older_mean'][k]:.5f} |"
              for k in r["contribution_last_30_days_mean"]),  # fmt: skip
            "",
            f"The two contributions add up to the forecast within {r['max_partition_error']:.1e}.",
        ]
    lines += [
        "",
        "## Variants",
        "",
        f"Variants tried so far: {m['variants']['before']} before this report, plus the {m['variants']['this_report']} arms: "
        f"{m['variants']['total']}. Trading metrics wait for a portfolio simulator; each arm still counts.",
        "",
        *prov_mod.render_md(config["provenance"]),
        f"- Panel fingerprint: `{m['panel']['fingerprint_sha256']}`.",
        *([series_line(m["series"])] if "series" in m else []),
        "- Parquet files read (SHA-256, stored and sealed rows; every file is in `config.json`): " + ", ".join(
            f"`{f['path']}` `{f['sha256']}` {f.get('stored_rows')}/{f.get('sealed_rows')}"
            for f in config["manifest"].get("files", [])
            if f.get("kind") != "klines" or (f.get("interval") == "1h" and f.get("symbol") in set(m["panel"]["symbols"]))) + ".",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def run_report(
    cfg: Report2Config,
    out: Path,
    *,
    data_dir: Path | None = None,
    universe: pl.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the panel, run every arm, score, and write the run folder. Returns the metrics."""
    started = time.perf_counter()
    manifest = prov_mod.current_manifest() if manifest is None else prov_mod.strip_wall_clock(manifest)
    provenance = prov_mod.record(manifest) if provenance is None else provenance
    t = time.perf_counter()
    panel = hourly_panel(universe, first=Month(*cfg.first_month), last=Month(*cfg.last_month), symbols=cfg.symbols, data_dir=data_dir)
    t_panel = time.perf_counter() - t
    metrics, timings, series = compute(panel, cfg)
    timings = {"panel": t_panel, **timings}
    config = config_record(cfg, provenance, manifest)
    out.mkdir(parents=True, exist_ok=True)
    write_npz(out / SERIES_FILE, series)
    metrics["series"] = {"file": SERIES_FILE, "sha256": sha256_file(out / SERIES_FILE)}
    _write_json(out / "config.json", config)
    _write_json(out / "report2.json", metrics)
    metrics, config = (json.loads((out / n).read_text(encoding="utf-8")) for n in ("report2.json", "config.json"))
    (out / REPORT_FILE).write_text(render(metrics, config), encoding="utf-8", newline="\n")
    run = {"finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - started, 1),
           "stages_seconds": {k: round(v, 1) for k, v in timings.items()}}  # fmt: skip
    _write_json(out / "run.json", run)
    return metrics


def compared_files(folder: Path) -> list[str]:
    return [n for n in COMPARED if (folder / n).exists()]


def _lines(data: bytes) -> list[str]:
    return data.decode("utf-8", "replace").splitlines()


def _file_summary(name: str, x: bytes | None, y: bytes | None, *, first: str, second: str, same: str = "identical") -> str:
    """One line for one compared file: identical, only on one side, or its count of differing lines (bytes, for series.npz)."""
    if x is None or y is None:
        return f"{name}: only in {first if y is None else second}"
    if x == y:
        return f"{name}: {same}"
    if name == SERIES_FILE:
        return (f"{name}: differs (binary; {len(x)} and {len(y)} bytes, SHA-256 {hashlib.sha256(x).hexdigest()[:16]}... and "
                f"{hashlib.sha256(y).hexdigest()[:16]}...)")  # fmt: skip
    gone = added = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, _lines(x), _lines(y), autojunk=False).get_opcodes():
        if tag != "equal":
            gone, added = gone + i2 - i1, added + j2 - j1
    if gone + added == 0:
        return f"{name}: differs in its bytes, not in its lines (line endings or encoding)"
    return f"{name}: {gone + added} differing lines ({gone} only in {first}, {added} only in {second})"


def _read(folder: Path, name: str) -> bytes | None:
    path = folder / name
    return path.read_bytes() if path.is_file() else None


def summarize_runs(expected: Path, actual: Path) -> list[str]:
    """Each compared file, in COMPARED order: identical, or how far it differs (audit phase 2, logged item 4)."""
    out = []
    for name in COMPARED:
        x, y = _read(expected, name), _read(actual, name)
        if x is not None or y is not None:
            out.append(_file_summary(name, x, y, first="the expected run", second="the re-run"))
    return out


def _unified(x: bytes, y: bytes, a: str, b: str, max_lines: int) -> str:
    lines = list(difflib.unified_diff(_lines(x), _lines(y), a, b, lineterm="", n=1))
    return "\n".join(lines[:max_lines]) + ("\n..." if len(lines) > max_lines else "")


def diff_runs(expected: Path, actual: Path, *, max_lines: int = 40) -> list[str]:
    """One problem per compared file that differs, with a diff of at most ``max_lines`` lines (none for series.npz)."""
    problems: list[str] = []
    a, b = compared_files(expected), compared_files(actual)
    for name in sorted(set(a) ^ set(b)):
        problems.append(f"{name}: only in {'the expected run' if name in a else 'the re-run'}")
    for name in sorted(set(a) & set(b)):
        x, y = (expected / name).read_bytes(), (actual / name).read_bytes()
        if x == y:
            continue
        if name == SERIES_FILE:
            problems.append(_file_summary(name, x, y, first="the expected run", second="the re-run"))
            continue
        problems.append(f"{name}: differs\n" + _unified(x, y, f"expected/{name}", f"re-run/{name}", max_lines))
    return problems


def _print(text: str) -> None:
    print(text.encode("ascii", "replace").decode("ascii"))


def check(expected: Path, cfg: Report2Config, **kwargs: Any) -> int:
    if not (expected / "report2.json").is_file():
        print(f"{expected}: not a report2 run folder (no report2.json)")
        return 2
    scratch = Path(tempfile.mkdtemp(prefix="report2-check-"))
    try:
        run_report(cfg, scratch, **kwargs)
        summary = summarize_runs(expected, scratch)
        problems = diff_runs(expected, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if problems:
        print(f"CHECK FAILED: {len(problems)} file(s) differ from {expected}")
        for line in summary:
            _print(f"  {line}")
        for p in problems:
            _print(p)
        return 1
    print(f"check passed: {len(compared_files(expected))} files identical to {expected} (run.json excluded)")
    for line in summary:
        _print(f"  {line}")
    return 0


# --- The comparison with a run that predates series.npz (iteration 3) -------------------------


@dataclass
class _Older:
    """An older run's compared files, parsed, edited by the allowances toward what the new code writes."""

    config: dict[str, Any]
    report: dict[str, Any]
    md: list[str]
    new_config: dict[str, Any]
    new_report: dict[str, Any]
    new_series_sha256: str | None
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Allowance:
    """A named difference allowed between an older run and the re-run, with its reason. ``apply`` edits the older
    run's parsed files into what the re-run must hold and returns whether it found that difference to allow."""

    name: str
    files: tuple[str, ...]
    reason: str
    apply: Callable[[_Older], bool]


def _replace_line(md: list[str], old: str, new: str) -> bool:
    hits = [i for i, line in enumerate(md) if line == old]
    if len(hits) != 1:
        return False
    md[hits[0]] = new
    return True


def _allow_commit(s: _Older) -> bool:
    old, new = s.config["provenance"], s.new_config.get("provenance", {})
    if old.get("git_commit") == new.get("git_commit"):
        return False
    edited = {**old, "git_commit": new.get("git_commit")}
    ok = _replace_line(s.md, prov_mod.render_md(old)[2], prov_mod.render_md(edited)[2])
    s.config["provenance"] = edited
    return ok


def _allow_code(s: _Older) -> bool:
    old, new = s.config["provenance"], s.new_config.get("provenance", {})
    old_code, new_code = old.get("code", {}), new.get("code", {})
    if (old_code.get("sha256"), old_code.get("files")) == (new_code.get("sha256"), new_code.get("files")):
        return False
    edited = {**old, "code": {**old_code, "sha256": new_code.get("sha256"), "files": new_code.get("files")}}  # globs must match
    ok = _replace_line(s.md, prov_mod.render_md(old)[3], prov_mod.render_md(edited)[3])
    s.config["provenance"] = edited
    return ok


def _allow_series(s: _Older) -> bool:
    if "series" in s.report:
        return False
    rec = s.new_report.get("series")
    digest = rec.get("sha256") if isinstance(rec, dict) else None
    if not (isinstance(rec, dict) and set(rec) == {"file", "sha256"} and rec["file"] == SERIES_FILE and isinstance(digest, str)):
        s.problems.append(f"report2.json: the re-run's series entry is not {{'file': '{SERIES_FILE}', 'sha256': ...}}: {rec!r}")
        return False
    if digest != s.new_series_sha256:
        s.problems.append(f"report2.json: the series digest {digest} is not the SHA-256 of the re-run's {SERIES_FILE}, {s.new_series_sha256}")
        return False
    s.report["series"] = {"file": SERIES_FILE, "sha256": digest}
    at = [i for i, line in enumerate(s.md) if line.startswith("- Panel fingerprint: ")]
    if len(at) != 1:
        return False
    s.md.insert(at[0] + 1, series_line(s.report["series"]))
    return True


def _allow_coverage(s: _Older) -> bool:
    return _replace_line(s.md, coverage_line(s.report, COVERAGE_BEFORE_ITERATION3), coverage_line(s.report))


_H12_CELL = re.compile(r"(, margin \d+\.\d{5})(; identical hours )")


def _allow_h12_parts(s: _Older) -> bool:
    split = s.report.get("preview_split", {})
    max_fits = s.config["config"]["matrix"]["cusum_max_fits"]
    words, edited = [], False
    for name in ("previewed", "unseen"):
        part = split.get(name)
        if part is None:
            continue
        c = next(c for c in part["comparisons"] if c["hypothesis"] == "H12")
        if "undefined" in c:
            continue
        if "rule" in c:
            return False
        boot = c["bootstrap"]
        if c["rule_met"] != passes(c["dm_p"], boot["low"], boot["high"], c["mean_diff"]):
            s.problems.append(f"report2.json: the older {name} part's H12 rule_met is not the superiority rule it carried")
            return False
        c["rule_met"] = h12_rule_met(c["upper_95_one_sided"], boot["q95"], part["h12_margin"])
        c["rule"] = h12_part_rule(max_fits)
        words.append("rule met" if c["rule_met"] else "rule not met")
        edited = True
    rows = [i for i, line in enumerate(s.md) if line.startswith("| H12: ")]
    if not edited or len(rows) != 1:
        return False
    i = rows[0]
    cells = iter(words)
    new_row, n = _H12_CELL.subn(lambda mt: mt.group(1) + H12_PART_WORD.format(word=next(cells)) + mt.group(2), s.md[i])
    if n != len(words) or s.md[i + 1 : i + 3] != ["", "## Counts"]:
        return False
    s.md[i] = new_row
    s.md[i + 2 : i + 2] = [H12_PART_NOTE.format(max_fits=s.report["verdicts"]["H12"]["max_fits"]), ""]
    return True


# The differences allowed between iteration 3's run (runs/2026-10-29-report2/) and a re-run at a later commit. Nothing else.
ITERATION3_ALLOWANCES: tuple[Allowance, ...] = (
    Allowance("commit", ("config.json", REPORT_FILE),
              "the re-run is at a later commit: provenance.git_commit and the Markdown's Git commit line", _allow_commit),
    Allowance("code digest", ("config.json", REPORT_FILE),
              "the code changed (this file, bench/power.py and the rest of the digest's files): provenance.code's sha256 and file "
              "count, and the Markdown's Code digest line; the globs must match", _allow_code),
    Allowance("series", ("report2.json", REPORT_FILE),
              "iteration 4 saves series.npz: report2.json gains series = {file, sha256}, the SHA-256 of the re-run's own file, and "
              "the Provenance list gains its line after the panel fingerprint", _allow_series),
    Allowance("coverage sentence", (REPORT_FILE,),
              "audit phase 2, logged item 1: a gap of g bars leaves 1 + g rows of each member unlabelled before it, not 25 bars; "
              "only that clause of the Coverage paragraph changes, its numbers stay", _allow_coverage),
    Allowance("H12's rule in the parts", ("report2.json", REPORT_FILE),
              "audit phase 2, logged item 3: each part's H12 rule_met becomes H12's own rule on that part's numbers, with a rule "
              "key that keeps the condition on fits for the whole period; the H12 row of the previewed months' table gains its "
              "rule word, and a note follows the table", _allow_h12_parts),
)  # fmt: skip


def compare_with_older(older: Path, new: Path, allowances: Sequence[Allowance] = ITERATION3_ALLOWANCES,
                       *, max_lines: int = 40) -> tuple[list[str], list[Allowance], list[str]]:  # fmt: skip
    """Compare a re-run in ``new`` with an older run whose files differ only by the named ``allowances``.

    The older run's ``config.json``, ``report2.json`` and ``report-2.md`` are edited by each allowance
    into what the re-run must hold, then compared byte for byte with the re-run's. ``series.npz`` must
    be in the re-run, with the digest its ``report2.json`` gives; if the older run has one too, the two
    must be identical. Returns the problems (empty: the comparison passes), the allowances met and a
    per-file summary.
    """
    names = ("config.json", "report2.json", REPORT_FILE)
    old_bytes = {n: _read(older, n) for n in names}
    new_bytes = {n: _read(new, n) for n in names}
    missing = [f"{n}: missing in {'the older run' if old_bytes[n] is None else 'the re-run'}" for n in names
               if old_bytes[n] is None or new_bytes[n] is None]  # fmt: skip
    series_new = new / SERIES_FILE
    if not series_new.is_file():
        missing.append(f"{SERIES_FILE}: missing in the re-run")
    if missing:
        return missing, [], []
    parsed = {n: json.loads(old_bytes[n]) for n in ("config.json", "report2.json")}
    problems = [f"{n}: the older file is not in the form this code writes, so it cannot be compared byte for byte"
                for n, obj in parsed.items() if _json_bytes(obj) != old_bytes[n]]  # fmt: skip
    if problems:
        return problems, [], []
    state = _Older(parsed["config.json"], parsed["report2.json"], old_bytes[REPORT_FILE].decode("utf-8").split("\n"),
                   json.loads(new_bytes["config.json"]), json.loads(new_bytes["report2.json"]), sha256_file(series_new))  # fmt: skip
    met = [a for a in allowances if a.apply(state)]
    problems = list(state.problems)
    expected = {"config.json": _json_bytes(state.config), "report2.json": _json_bytes(state.report), REPORT_FILE: "\n".join(state.md).encode("utf-8")}
    summary = []
    for n in names:
        summary.append(_file_summary(n, expected[n], new_bytes[n], first="the older run", second="the re-run", same="identical after the allowances"))
        if expected[n] != new_bytes[n]:
            problems.append(f"{n}: differs beyond the allowances\n" + _unified(expected[n], new_bytes[n], f"older+allowances/{n}", f"re-run/{n}", max_lines))
    series_old = _read(older, SERIES_FILE)
    if series_old is not None and series_old != series_new.read_bytes():
        problems.append(f"{SERIES_FILE}: the older run has one, and it differs from the re-run's")
    summary.append(f"{SERIES_FILE}: " + ("only in the re-run" if series_old is None else "identical" if series_old == series_new.read_bytes() else "differs"))
    return problems, met, summary


def compare_command(older: Path, new: Path, cfg: Report2Config, *, rerun: bool, **kwargs: Any) -> int:
    if not (older / "report2.json").is_file():
        print(f"{older}: not a report2 run folder (no report2.json)")
        return 2
    if rerun:
        if new.exists() and any(new.iterdir()):
            print(f"{new}: not empty; the re-run goes to a new folder (or pass --existing to compare it as it is)")
            return 2
        run_report(cfg, new, **kwargs)
    problems, met, summary = compare_with_older(older, new)
    _print(f"compared {new} with {older}; allowed differences, each with its reason:")
    for a in ITERATION3_ALLOWANCES:
        _print(f"  [{'met' if a in met else 'not met'}] {a.name} ({', '.join(a.files)}): {a.reason}")
    for line in summary:
        _print(f"  {line}")
    if problems:
        print(f"COMPARISON FAILED: {len(problems)} difference(s) beyond the allowances")
        for p in problems:
            _print(p)
        return 1
    print("comparison passed: every other byte of config.json, report2.json and report-2.md matches")
    return 0


def _default_out() -> Path:
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-report2"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.report2", description="Report no. 2: Matrix on hourly bars.")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default: <runs>/<UTC date>-report2, suffixed if taken)")
    parser.add_argument("--check", type=Path, default=None, metavar="DIR", help="re-run and compare with the run in DIR, byte for byte")
    parser.add_argument("--compare-with", type=Path, default=None, metavar="OLDER",
                        help="re-run once into --out (kept) and compare with OLDER under the named allowances")  # fmt: skip
    parser.add_argument("--existing", action="store_true", help="with --compare-with: compare the folder at --out without re-running")
    args = parser.parse_args(argv)
    cfg = Report2Config()
    if args.check is not None:
        if args.compare_with is not None:
            parser.error("--check and --compare-with are separate commands")
        return check(args.check, cfg)
    if args.compare_with is not None:
        if args.existing and args.out is None:
            parser.error("--existing needs --out")
        return compare_command(args.compare_with, args.out or _default_out(), cfg, rerun=not args.existing)
    if args.existing:
        parser.error("--existing goes with --compare-with")
    out = args.out or _default_out()
    started = time.perf_counter()
    metrics = run_report(cfg, out)
    verdict = ", ".join(f"{h}: {'holds' if v['holds'] else 'does not hold'}" for h, v in metrics["verdicts"].items())
    print(f"report2 written to {out} in {time.perf_counter() - started:.0f} s; {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
