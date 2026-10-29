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

Outputs: ``report-2.md``, ``report2.json``, ``config.json`` and ``run.json`` (the only file with
wall-clock times) to ``runs/<UTC date>-report2/`` (suffixed if taken) or ``--out``. ``--check
DIR`` re-runs into a temporary folder and fails, with a diff, if any file but ``run.json`` differs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import shutil
import sys
import tempfile
import time
import zlib
from collections.abc import Sequence
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
COMPARED = ("config.json", "report2.json", REPORT_FILE)
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


def ece_interval(p: np.ndarray, y: np.ndarray, sc: Scored, brier_hourly: np.ndarray, cfg: Report2Config, *tag: str) -> dict[str, Any]:
    """Top-label ECE on the scored rows; the bootstrap resamples whole hours (block: PW of the hourly Brier)."""
    d = ece_bins(p, y, sc, cfg.ece_bins)
    est = float(np.abs(d.sum(axis=0)).sum() / sc.count.sum())
    check = stats.ece(p, y, cfg.ece_bins)
    if not math.isclose(est, check, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(f"ECE {est} differs from stats.ece {check}")
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


def _compare_part(part: dict[str, np.ndarray], a: str, b: str, clim: float, cfg: Report2Config, name: str) -> dict[str, Any]:
    """compare_arms on one part, or a record of why it has no test: a part can hold two arms that agree at every hour.

    A part describes, so its comparisons say ``rule_met``: "passes" is the registered rule's word,
    and the verdicts are those of the whole period (audit pass 1e).
    """
    d = part[a] - part[b]
    if np.all(d == d[0]):  # e.g. matrix-adaptive taking matrix-select's rho at every decision: Diebold-Mariano is undefined
        return {"a": a, "b": b, "mean_diff": float(d.mean()), "hours": int(d.size), **identical_hours(d),
                "undefined": "the difference is the same at every hour"}  # fmt: skip
    c = compare_arms(part, a, b, clim, cfg, name)
    c["rule_met"] = c.pop("passes")
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
            "comparisons": [dict(_compare_part(part, a, b, clim, cfg, name), hypothesis=h) for h, a, b in COMPARISONS],
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
            "holds": bool(h12["upper_95_one_sided"] < margin and h12["bootstrap"]["q95"] < margin and n_cusum_fits <= cfg.matrix.cusum_max_fits),
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


def compute(panel: pl.DataFrame, cfg: Report2Config) -> tuple[dict[str, Any], dict[str, float]]:
    """Every number of Report n°2 from a panel, and the time of each stage (seconds)."""
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
    return out, timings


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


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_clean(obj), sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


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
            if h == "H12":
                cells.append(f"{c['mean_diff']:+.5f}; upper bound {c['upper_95_one_sided']:+.5f}, bootstrap q95 {boot['q95']:+.5f}, "
                             f"margin {p['h12_margin']:.5f}")  # fmt: skip
            else:
                cells.append(f"{c['mean_diff']:+.5f} [{boot['low']:+.5f}, {boot['high']:+.5f}], DM p {c['dm_p']:.4f}, "
                             f"{'rule met' if c['rule_met'] else 'rule not met'}")  # fmt: skip
            cells[-1] += f"; identical hours {_informative(c)}"
        lines.append(f"| {h}: {a} - {b} | " + " | ".join(cells) + " |")
    return [*lines, ""]


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
        f"Coverage: {m['scored']['hours']} of the test months' {m['scored']['calendar_hours']} hours are scored "
        f"({m['scored']['share_of_calendar_hours']:.1%}). An hour is scored when a member has a scored row in it. A row is not scored "
        "when a feature or its label is missing: for about 90 bars after a gap in a pair's bars, while the features' longest window "
        "refills, and for the 25 bars before it, whose labels need a bar inside it. An exchange outage makes that gap for every "
        "member at once, so every verdict below is conditional on the hours away from outages.",
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
    metrics, timings = compute(panel, cfg)
    timings = {"panel": t_panel, **timings}
    config = config_record(cfg, provenance, manifest)
    out.mkdir(parents=True, exist_ok=True)
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


def diff_runs(expected: Path, actual: Path, *, max_lines: int = 40) -> list[str]:
    problems: list[str] = []
    a, b = compared_files(expected), compared_files(actual)
    for name in sorted(set(a) ^ set(b)):
        problems.append(f"{name}: only in {'the expected run' if name in a else 'the re-run'}")
    for name in sorted(set(a) & set(b)):
        x, y = (expected / name).read_bytes(), (actual / name).read_bytes()
        if x == y:
            continue
        lines = list(difflib.unified_diff(x.decode("utf-8", "replace").splitlines(), y.decode("utf-8", "replace").splitlines(),
                                          f"expected/{name}", f"re-run/{name}", lineterm="", n=1))  # fmt: skip
        problems.append(f"{name}: differs\n" + "\n".join(lines[:max_lines]) + ("\n..." if len(lines) > max_lines else ""))
    return problems


def check(expected: Path, cfg: Report2Config, **kwargs: Any) -> int:
    if not (expected / "report2.json").is_file():
        print(f"{expected}: not a report2 run folder (no report2.json)")
        return 2
    scratch = Path(tempfile.mkdtemp(prefix="report2-check-"))
    try:
        run_report(cfg, scratch, **kwargs)
        problems = diff_runs(expected, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if problems:
        print(f"CHECK FAILED: {len(problems)} file(s) differ from {expected}")
        for p in problems:
            print(p.encode("ascii", "replace").decode("ascii"))
        return 1
    print(f"check passed: {len(compared_files(expected))} files identical to {expected} (run.json excluded)")
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
    parser.add_argument("--check", type=Path, default=None, metavar="DIR", help="re-run and compare with the run in DIR")
    args = parser.parse_args(argv)
    cfg = Report2Config()
    if args.check is not None:
        return check(args.check, cfg)
    out = args.out or _default_out()
    started = time.perf_counter()
    metrics = run_report(cfg, out)
    verdict = ", ".join(f"{h}: {'holds' if v['holds'] else 'does not hold'}" for h, v in metrics["verdicts"].items())
    print(f"report2 written to {out} in {time.perf_counter() - started:.0f} s; {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
