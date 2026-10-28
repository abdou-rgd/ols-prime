"""Report n°1: what fees do to simple strategies (``docs/HYPOTHESES.md``, H9).

Usage::

    uv run python -m olsprime.bench.report1 [--out DIR] [--workers N]
    uv run python -m olsprime.bench.report1 --check DIR

Method, fixed in :class:`Report1Config` and saved with every run:

- **Data.** BTCUSDT and ETHUSDT, 1d, through ``load_klines`` (sealed), cut at the end of the
  development period: no bar of the validation year or the vault enters the computation.
- **Models are per pair.** Each pair's frame, folds and fits use that pair's rows alone.
- **Pipeline.** Point-in-time features, labels with h = 1, monthly walk-forward folds with
  purge and embargo h over the test months 2019-01 -> 2024-12 (:mod:`olsprime.bench.pipeline`).
- **Scored window.** The bars decided in the test months, 2019-01-01 -> 2024-12-31, the same
  for every strategy and venue. The target decided at a bar's close executes at the next open;
  the first bar never trades and the last target never executes. The equity is marked at each
  close, so a position open at the end pays no exit fee.
- **Strategies.** The baselines of ``olsprime.models.baselines`` and nothing more;
  ``random-turnover`` runs 100 seeds. The cheater is the H1 control, shown on its own line.
- **Venues.** The five fee schedules of ``docs/PROTOCOL.md``, taker rate, 2 bps slippage per
  side, Binance's order filters for the pair.

Outputs: ``runs/<UTC date>-report1/`` (``config.json``, ``manifest.json``, ``metrics.json``,
``equity/*.csv``, ``report-1.md`` and ``run.json``, the only file with wall-clock times) and
``docs/reports/report-1.md``. Same inputs give byte-identical files; ``--check DIR`` re-runs
into a temporary folder and fails, with a diff, if any file but ``run.json`` differs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import shutil
import sys
import tempfile
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from olsprime.bench import pipeline as pp
from olsprime.bench.provenance import CODE_GLOBS, code_digest, git_commit, record, strip_wall_clock  # noqa: F401 (re-exported)
from olsprime.data.binance_vision import Month
from olsprime.eval import stats
from olsprime.eval.splits import describe as describe_folds
from olsprime.models.baselines import (
    BASELINE_NAMES,
    MA_WINDOWS,
    Baseline,
    Cheater,
    Climatology,
    MaMomentum,
    RandomTurnover,
    Ridge,
    XGBoost,
    buy_and_hold,
    describe,
    flat,
)
from olsprime.paths import REPO_ROOT, runs_dir
from olsprime.protocol import DEVELOPMENT

log = logging.getLogger(__name__)

RANDOM = "random-turnover"
CHEATER = "cheater"
BUY_AND_HOLD = "buy-and-hold"
FORECASTERS = ("climatology", "ridge", "xgboost")
H9_VENUES = ("coinbase-advanced-entry", "kraken-pro-entry")
# The active baselines of H9, from the fixed list: everything but flat, buy and hold and the cheater.
ACTIVE = tuple(n for n in BASELINE_NAMES if n not in ("flat", BUY_AND_HOLD, CHEATER))
# DSR variants: every candidate with a defined Sharpe; flat (never exposed) and the cheater are excluded.
DSR_CANDIDATES = tuple(n for n in BASELINE_NAMES if n not in ("flat", CHEATER))
REPORT_FILE = "report-1.md"
COMPARED = ("config.json", "manifest.json", "metrics.json", REPORT_FILE)  # plus equity/*.csv


@dataclass(frozen=True)
class Report1Config:
    pairs: tuple[str, ...] = pp.PAIRS
    interval: str = "1d"
    h: int = 1
    test_start: tuple[int, int] = (2019, 1)
    test_end: tuple[int, int] = (2024, 12)
    venues: tuple[str, ...] = pp.VENUES
    h9_venues: tuple[str, ...] = H9_VENUES
    liquidity: str = pp.LIQUIDITY
    slippage_bps: float = pp.SLIPPAGE_BPS
    initial_cash: float = pp.INITIAL_CASH
    filters: dict[str, dict[str, float]] = field(default_factory=lambda: {k: asdict(v) for k, v in pp.SYMBOL_FILTERS.items()})
    random_seeds: int = 100
    n_boot: int = 2000
    level: float = 0.95
    bootstrap_seed: int = 20261027
    periods_per_year: float = 365.0
    models_per_pair: bool = True  # each pair's frame, folds and fits use that pair's rows alone
    data_end: str = DEVELOPMENT.end.isoformat()  # bars final at or after this are dropped before anything
    ridge_tolerance: float = 1e-9  # numpy vs olsprime_rs.ridge_fit, max abs difference


# --- Strategies --------------------------------------------------------------------------


def factories(interval: str, bars: pl.DataFrame, seeds: int) -> dict[str, list[Any]]:
    """Name -> list of zero-argument factories (one per seed for random-turnover), from the fixed list."""
    table: dict[str, list[Any]] = {
        "flat": [flat],
        BUY_AND_HOLD: [buy_and_hold],
        **{f"ma-momentum-{w}": [lambda w=w: MaMomentum(w)] for w in MA_WINDOWS},
        RANDOM: [lambda s=s: RandomTurnover(s, interval) for s in range(seeds)],
        "climatology": [Climatology],
        "ridge": [Ridge],
        "xgboost": [XGBoost],
        CHEATER: [lambda: Cheater(bars)],  # !!! the H1 control, it leaks on purpose !!!
    }
    if tuple(table) != BASELINE_NAMES:
        raise AssertionError(f"the bench's strategies {tuple(table)} differ from BASELINE_NAMES {BASELINE_NAMES}")
    return table


def _seed(cfg: Report1Config, *parts: str) -> int:
    """A bootstrap seed per statistic, stable across processes (Python's str hash is not)."""
    return zlib.crc32("|".join((str(cfg.bootstrap_seed), *parts)).encode())


# --- Stage 1: predictions and simulations of one pair --------------------------------------


def data_end(cfg: Report1Config) -> datetime:
    """``cfg.data_end`` as a UTC instant: the config is the source, and it may never reach past the development period."""
    end = datetime.fromisoformat(cfg.data_end)
    if end.tzinfo is None or end.utcoffset() != timedelta(0):
        raise ValueError(f"data_end must be a UTC instant, got {cfg.data_end!r}")
    if end > DEVELOPMENT.end:
        raise ValueError(f"data_end {cfg.data_end} reaches past the development period ({DEVELOPMENT.end:%Y-%m-%d})")
    return end


def load_pair(pair: str, cfg: Report1Config) -> pl.DataFrame:
    from olsprime.protocol import load_klines

    bars = load_klines(pair, cfg.interval, caller="bench.report1")
    return bars.filter(pl.col("known_at") < data_end(cfg))


def run_pair(pair: str, bars: pl.DataFrame, cfg: Report1Config) -> dict[str, Any]:
    """Predict with every strategy, simulate every venue and compute the per-pair metrics."""
    frame = pp.build_frame(bars, cfg.interval, cfg.h)
    folds = pp.make_folds(frame, cfg.interval, cfg.h, Month(*cfg.test_start), Month(*cfg.test_end))
    table = factories(cfg.interval, bars, cfg.random_seeds)

    checks: list[dict[str, float]] = []

    def crosscheck(fold, model, train):  # noqa: ANN001
        checks.append({"fit_time_ms": float(int(fold.fit_time.timestamp() * 1000)), **pp.ridge_crosscheck(model, train)})

    preds: dict[str, list[pl.DataFrame]] = {}
    for name, makers in table.items():
        hook = crosscheck if name == "ridge" else None
        preds[name] = [pp.walk_forward_predict(make, frame, folds, on_fit=hook) for make in makers]
    reference = preds["flat"][0]["decision_time"]
    for name, runs in preds.items():
        for run in runs:
            if not run["decision_time"].equals(reference):
                raise AssertionError(f"{name}: scored rows differ from the other strategies'")
    scored = pp.align(bars, preds["flat"][0], cfg.interval)

    worst = max(max(c["coef_max_abs_diff"], c["fitted_max_abs_diff"]) for c in checks)
    if not worst < cfg.ridge_tolerance:
        raise AssertionError(f"{pair}: numpy and olsprime_rs ridge differ by {worst:.3e} >= {cfg.ridge_tolerance}")

    venues: dict[str, Any] = {}
    curves: dict[str, pl.DataFrame] = {}
    for venue in cfg.venues:
        venues[venue], curves[venue] = run_venue(pair, venue, scored, preds, cfg)

    fd = describe_folds(folds)
    return {
        "pair": pair,
        "bars_used": {"rows": bars.height, "first": bars["open_time"][0].isoformat(), "last": bars["open_time"][-1].isoformat()},
        "scored": {
            "bars": scored.height,
            "first": scored["open_time"][0].isoformat(),
            "last": scored["open_time"][-1].isoformat(),
            "returns": scored.height - 1,
        },
        "folds": {
            "count": len(folds),
            "first_fit": folds[0].fit_time.isoformat(),
            "last_fit": folds[-1].fit_time.isoformat(),
            "n_train_min": int(fd["n_train"].min()),
            "n_train_max": int(fd["n_train"].max()),
            "n_test_total": int(fd["n_test"].sum()),
            "last_label_known_at": fd["last_label_known_at"].max().isoformat(),
        },
        "ridge_crosscheck": {
            "folds": len(checks),
            "worst_coef_max_abs_diff": max(c["coef_max_abs_diff"] for c in checks),
            "worst_fitted_max_abs_diff": max(c["fitted_max_abs_diff"] for c in checks),
            "lambdas": _counts([c["lambda"] for c in checks]),
        },
        "forecasts": forecast_metrics(pair, frame, folds, preds, cfg),
        "venues": venues,
    }, curves


def _counts(values: Sequence[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[repr(v)] = out.get(repr(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def _sim(scored: pl.DataFrame, target: np.ndarray, venue: str, cfg: Report1Config):
    res = pp.simulate_strategy(scored, target, venue, initial_cash=cfg.initial_cash)
    return res, pp.period_returns(np.asarray(res.equity))


def _per_period_sharpe(r: np.ndarray) -> float | None:
    if r.size < 3 or not np.isfinite(r).all() or np.all(r == r[0]):
        return None
    return stats.sharpe(r)


def run_venue(
    pair: str, venue: str, scored: pl.DataFrame, preds: dict[str, list[pl.DataFrame]], cfg: Report1Config
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Simulate every strategy at one venue and score it; the random-turnover seeds as a distribution."""
    import olsprime_rs

    ppy = cfg.periods_per_year
    returns: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    sims: dict[str, Any] = {}
    for name, runs in preds.items():
        if name == RANDOM:
            continue
        res, r = _sim(scored, pp.targets(runs[0], venue), venue, cfg)
        sims[name], returns[name] = res, r

    # random-turnover: every seed simulated; the median-Sharpe seed represents the variant.
    seed_rows = []
    seed_sims = []
    for seed, run in enumerate(preds[RANDOM]):
        res, r = _sim(scored, pp.targets(run, venue), venue, cfg)
        m = olsprime_rs.metrics(res, ppy).as_dict()
        sr = _per_period_sharpe(r)
        seed_rows.append({"seed": seed, "sharpe_pp": sr, "fills": len(res.fills), **m})
        seed_sims.append((res, r))
    defined = sorted((row["sharpe_pp"], row["seed"]) for row in seed_rows if row["sharpe_pp"] is not None)
    if not defined:
        raise AssertionError(f"{pair} {venue}: no random-turnover seed has a defined Sharpe")
    median_seed = defined[(len(defined) - 1) // 2][1]
    sims[RANDOM], returns[RANDOM] = seed_sims[median_seed]

    # DSR: N = candidates with a defined Sharpe, V = variance of their per-period Sharpes.
    sr_pp = {name: _per_period_sharpe(r) for name, r in returns.items()}
    variants = [n for n in DSR_CANDIDATES if sr_pp.get(n) is not None]
    n_trials = len(variants)
    var_sr = stats.trials_var_sr(np.array([sr_pp[n] for n in variants])) if n_trials >= 2 else None

    for name in BASELINE_NAMES:
        results[name] = trading_metrics(
            pair, venue, name, sims[name], returns[name], cfg, n_trials=n_trials if name in variants else None, var_sr=var_sr
        )
    for name in BASELINE_NAMES:
        results[name]["lot_cap_first_binds"] = lot_cap_first_binds(sims[name], scored, pair)
    results[RANDOM]["seed"] = median_seed
    results[RANDOM]["distribution"] = seed_distribution(seed_rows, cfg)
    bh_sr = sr_pp[BUY_AND_HOLD]
    results[RANDOM]["distribution"]["share_of_seeds_with_sharpe_above_buy_and_hold"] = (
        None if bh_sr is None else sum(1 for s, _ in defined if s > bh_sr) / len(seed_rows)
    )

    h9 = {}
    for name in ACTIVE:
        h9[name] = sharpe_difference(pair, venue, name, returns[name], returns[BUY_AND_HOLD], sr_pp[name], bh_sr, cfg)

    curves = pl.DataFrame(
        [
            pl.Series("open_time", scored["open_time"].dt.to_string("%Y-%m-%d")),
            *(pl.Series(f"{n}-seed{median_seed}" if n == RANDOM else n, np.asarray(sims[n].equity)) for n in BASELINE_NAMES),
        ]
    )
    return {
        "dsr": {"variants": variants, "n_trials": n_trials, "var_sr_per_period": var_sr},
        "strategies": results,
        "h9": h9,
    }, curves


# The simulator's book holds at most 2^50 lots (crates/backtest/src/sim.rs, MAX_LOTS); a buy beyond is clamped.
MAX_LOTS = 2**50


def lot_cap_first_binds(res: Any, scored: pl.DataFrame, pair: str) -> str | None:
    """The first scored date whose position sits at the simulator's lot cap, or None if it never does.

    Only the cheater gets there: compounding at its fake edge, its book would outgrow any market.
    """
    step = pp.SYMBOL_FILTERS[pair].step_size
    if not step > 0:
        return None
    position = np.asarray(res.position, dtype=np.float64)
    at_cap = np.flatnonzero(position >= MAX_LOTS * step * (1.0 - 1e-9))
    return None if at_cap.size == 0 else scored["open_time"][int(at_cap[0])].strftime("%Y-%m-%d")


def trading_metrics(
    pair: str, venue: str, name: str, res: Any, r: np.ndarray, cfg: Report1Config, *, n_trials: int | None, var_sr: float | None
) -> dict[str, Any]:
    import olsprime_rs

    ppy = cfg.periods_per_year
    m = olsprime_rs.metrics(res, ppy).as_dict()
    out: dict[str, Any] = {
        "total_return": m["total_return"],
        "annualized_return": m["annualized_return"],
        "annualized_volatility": m["annualized_volatility"],
        "sharpe": None,
        "sharpe_ci_analytic": None,
        "sharpe_ci_bootstrap": None,
        "psr0": None,
        "dsr": None,
        "sortino": m["sortino"],
        "max_drawdown": m["max_drawdown"],
        "turnover": m["turnover"],
        "fees_paid": m["fees_paid"],
        "exposure": m["exposure"],
        "fills": len(res.fills),
    }
    sr = _per_period_sharpe(r)
    if sr is None:
        return out
    if m["sharpe"] is None or abs(stats.annualized(sr, ppy) - m["sharpe"]) > 1e-9:
        raise AssertionError(f"{pair} {venue} {name}: the Sharpe of eval.stats and of backtest::metrics disagree")
    analytic = stats.sharpe_ci(r, ppy, cfg.level)
    boot = stats.sharpe_ci_bootstrap(r, ppy, cfg.level, cfg.n_boot, seed=_seed(cfg, pair, venue, name, "sharpe"))
    out |= {
        "sharpe": stats.annualized(sr, ppy),
        "sharpe_ci_analytic": [analytic.low, analytic.high],
        "sharpe_ci_bootstrap": [boot.low, boot.high],
        "bootstrap_mean_block": boot.mean_block,
        "psr0": stats.psr(r, 0.0),
    }
    if n_trials is not None:
        out["dsr"] = stats.dsr(r, n_trials, var_sr)
    return out


def seed_distribution(rows: list[dict[str, Any]], cfg: Report1Config) -> dict[str, Any]:
    """Median and 5th / 95th percentiles of each metric over the random-turnover seeds."""
    ppy = cfg.periods_per_year
    out: dict[str, Any] = {"seeds": len(rows)}
    keys = ("total_return", "annualized_return", "sharpe", "sortino", "max_drawdown", "turnover", "fees_paid", "exposure", "fills")
    for key in keys:
        if key == "sharpe":
            values = [stats.annualized(row["sharpe_pp"], ppy) for row in rows if row["sharpe_pp"] is not None]
        else:
            values = [row[key] for row in rows if row[key] is not None]
        if values:
            p5, p50, p95 = (float(v) for v in np.percentile(np.asarray(values, dtype=np.float64), [5, 50, 95]))
            out[key] = {"median": p50, "p5": p5, "p95": p95, "n": len(values)}
        else:
            out[key] = None
    return out


def sharpe_difference(
    pair: str,
    venue: str,
    name: str,
    r: np.ndarray,
    r_bh: np.ndarray,
    sr: float | None,
    sr_bh: float | None,
    cfg: Report1Config,
) -> dict[str, Any]:
    """Net annualized Sharpe of ``name`` minus buy and hold's, paired stationary-bootstrap interval."""
    if sr is None or sr_bh is None:
        return {"defined": False, "reason": "a Sharpe ratio is undefined (no variance: never exposed)", "above_zero": False}
    ci = stats.sharpe_diff_ci_bootstrap(
        r, r_bh, cfg.periods_per_year, cfg.level, cfg.n_boot, seed=_seed(cfg, pair, venue, name, "h9")
    )
    block = sensitivity_block(r, r_bh)
    sens = stats.sharpe_diff_ci_bootstrap(
        r, r_bh, cfg.periods_per_year, cfg.level, cfg.n_boot, seed=_seed(cfg, pair, venue, name, "h9-sensitivity"), mean_block=block
    )
    return {
        "defined": True,
        "difference": ci.estimate,
        "ci": [ci.low, ci.high],
        "p_value": ci.p_value,
        "mean_block": ci.mean_block,
        "above_zero": bool(ci.low > 0.0),
        "sensitivity": {
            "rule": "mean block max(PW(d), PW((d - mean d)^2)), d the paired return difference",
            "mean_block": block,
            "ci": [sens.low, sens.high],
            "p_value": sens.p_value,
            "above_zero": bool(sens.low > 0.0),
        },
    }


def _pw(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8 or np.all(x == x[0]):
        return 1
    return stats.optimal_block_length(x)


def sensitivity_block(r: np.ndarray, r_bh: np.ndarray) -> int:
    """``max(PW(d), PW((d - mean d)^2))`` with ``d = r - r_bh``: a block that also sees volatility clustering.

    The registered block (Politis-White on each return series) reads the autocorrelation of the
    returns only; the squares of the paired difference carry its volatility clustering.
    """
    d = np.asarray(r, dtype=np.float64) - np.asarray(r_bh, dtype=np.float64)
    return max(_pw(d), _pw((d - d.mean()) ** 2))


def forecast_metrics(
    pair: str, frame: pl.DataFrame, folds: Sequence[Any], preds: dict[str, list[pl.DataFrame]], cfg: Report1Config
) -> dict[str, Any]:
    """Log-loss, Brier, BSS against climatology, ECE and Diebold-Mariano, on the scored rows with a label."""
    idx = np.concatenate([f.test_idx for f in folds])
    truth = frame[idx].select("decision_time", "label", "label_known_at")
    probs = {}
    mask = truth["label"].is_not_null().to_numpy() & (truth["label_known_at"] < data_end(cfg)).fill_null(False).to_numpy()
    for name in FORECASTERS:
        p = preds[name][0].select("p_down", "p_neutral", "p_up").to_numpy().astype(np.float64)
        probs[name] = p
        mask &= np.isfinite(p).all(axis=1)
    y = truth["label"].to_numpy()[mask].astype(np.int64)
    clim = probs["climatology"][mask]
    clim_losses = stats.per_row_losses(clim, y)
    out: dict[str, Any] = {
        "rows": int(mask.sum()),
        "dropped_rows": int((~mask).sum()),
        "class_frequencies": [float(np.mean(y == k)) for k in range(3)],
        "models": {},
    }
    for name in FORECASTERS:
        p = probs[name][mask]
        losses = stats.per_row_losses(p, y)
        entry: dict[str, Any] = {
            "log_loss": float(losses.log_loss.mean()),
            "brier": float(losses.brier.mean()),
            "ece_top": stats.ece(p, y),
            "ece_classwise": stats.classwise_ece(p, y),
        }
        if name != "climatology":
            bss = stats.bss_ci_bootstrap(p, clim, y, cfg.level, cfg.n_boot, seed=_seed(cfg, pair, name, "bss"))
            dm = stats.diebold_mariano(losses.log_loss, clim_losses.log_loss, h=cfg.h)
            entry["dm_log_loss_newey_west"] = dm_newey_west(losses.log_loss, clim_losses.log_loss)
            entry |= {
                "bss": bss.estimate,
                "bss_ci": [bss.low, bss.high],
                "bss_p_value": bss.p_value,
                "dm_log_loss": {"statistic": dm.statistic, "p_value": dm.p_value, "mean_diff": dm.mean_diff, "note": dm.note},
            }
        else:
            entry |= {"bss": 0.0}
        out["models"][name] = entry
    return out


def newey_west_lags(t: int) -> int:
    """``floor(4 (T / 100)^(2/9))``, the usual automatic Bartlett bandwidth (7 at T = 2,190)."""
    return int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def dm_newey_west(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, Any]:
    """Sensitivity of Diebold-Mariano: HAC variance with the Bartlett kernel and :func:`newey_west_lags` lags.

    The registered test (``eval.stats.diebold_mariano`` at h = 1) uses no lag. Here the long-run
    variance of ``d = loss_a - loss_b`` is ``g_0 + 2 sum_k (1 - k/(L+1)) g_k`` (divisor T); the
    statistic ``mean(d) / sqrt(lrv / T)`` is compared with a Student t with T - 1 degrees of
    freedom, two-sided, without the HLN small-sample factor (which is for h-step forecasts).
    """
    from scipy import stats as st

    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    t = d.size
    lags = newey_west_lags(t)
    c = d - d.mean()
    lrv = float(c @ c) / t
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(c[k:] @ c[:-k]) / t
    if not lrv > 0:
        return {"lags": lags, "statistic": None, "p_value": None, "note": "long-run variance not positive"}
    stat = float(d.mean() / math.sqrt(lrv / t))
    return {"lags": lags, "statistic": stat, "p_value": float(2.0 * st.t.sf(abs(stat), df=t - 1)), "note": ""}


# --- Stage 2: the whole report -------------------------------------------------------------


def _pair_task(pair: str, cfg: Report1Config) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    return run_pair(pair, load_pair(pair, cfg), cfg)


def h9_verdicts(pairs: dict[str, Any], cfg: Report1Config) -> dict[str, Any]:
    """Per venue: H9 holds if no active baseline's interval lies entirely above 0, on any pair."""
    out = {}
    for venue in cfg.venues:
        above = [
            f"{pair}:{name}"
            for pair, data in pairs.items()
            for name, d in data["venues"][venue]["h9"].items()
            if d["above_zero"]
        ]
        entries = [d for data in pairs.values() for d in data["venues"][venue]["h9"].values()]
        above_sens = [
            f"{pair}:{name}"
            for pair, data in pairs.items()
            for name, d in data["venues"][venue]["h9"].items()
            if d["defined"] and d["sensitivity"]["above_zero"]
        ]
        out[venue] = {
            "registered": venue in cfg.h9_venues,
            "holds": not above,
            "intervals_above_zero": above,
            "comparisons": len(entries),
            "with_interval": sum(1 for d in entries if d["defined"]),
            "undefined": sum(1 for d in entries if not d["defined"]),
            "sensitivity_holds": not above_sens,
            "sensitivity_intervals_above_zero": above_sens,
        }
    return out


def build_metrics(pair_results: dict[str, dict[str, Any]], cfg: Report1Config) -> dict[str, Any]:
    return {"pairs": pair_results, "h9": h9_verdicts(pair_results, cfg)}


def config_record(cfg: Report1Config, provenance: dict[str, Any]) -> dict[str, Any]:
    """The run's configuration, the baselines' configs and the provenance of code and libraries."""
    import olsprime_rs
    import scipy
    import xgboost

    from olsprime.bench.synthetic import synthetic_bars

    sample = [makers[0]() for makers in factories(cfg.interval, synthetic_bars(days=3), 1).values()]
    return {
        "report": "report-1",
        "config": asdict(cfg),
        "baselines": [describe(b) for b in sample],
        "baseline_names": list(BASELINE_NAMES),
        "random_turnover": f"{cfg.random_seeds} seeds (0 .. {cfg.random_seeds - 1}), one variant",
        "versions": {
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scipy": scipy.__version__,
            "xgboost": xgboost.__version__,
            "olsprime_rs": olsprime_rs.__version__,
            "python": sys.version.split()[0],
        },
        "provenance": provenance,
    }


def run_report(cfg: Report1Config, out: Path, *, workers: int = 1, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute everything and write the run folder; returns the metrics."""
    from olsprime.bench.provenance import current_manifest

    started = time.perf_counter()
    manifest = current_manifest() if manifest is None else strip_wall_clock(manifest)  # wall clock: run.json
    provenance = record(manifest)
    if workers > 1 and len(cfg.pairs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(cfg.pairs))) as pool:
            done = list(pool.map(_pair_task, cfg.pairs, [cfg] * len(cfg.pairs)))
    else:
        done = [_pair_task(p, cfg) for p in cfg.pairs]
    return write_run(cfg, out, done, manifest, provenance, started=started, workers=workers)


def write_run(
    cfg: Report1Config,
    out: Path,
    done: Sequence[tuple[dict[str, Any], dict[str, pl.DataFrame]]],
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    *,
    started: float,
    workers: int,
) -> dict[str, Any]:
    pair_results = {res["pair"]: res for res, _ in done}
    metrics = build_metrics(pair_results, cfg)
    config = config_record(cfg, provenance)
    out.mkdir(parents=True, exist_ok=True)
    (out / "equity").mkdir(exist_ok=True)
    _write_json(out / "config.json", config)
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "metrics.json", metrics)
    for res, curves in done:
        for venue, curve in curves.items():
            curve.write_csv(out / "equity" / f"{res['pair']}-{venue}.csv")
    metrics, config = (json.loads((out / n).read_text(encoding="utf-8")) for n in ("metrics.json", "config.json"))
    (out / REPORT_FILE).write_text(render(metrics, config, manifest), encoding="utf-8", newline="\n")
    run = {
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 1),
        "workers": workers,
    }
    _write_json(out / "run.json", run)
    return metrics


def _clean(value: Any) -> Any:
    """NaN and infinities become null, numpy scalars plain Python: strict, stable JSON."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, float) and value == 0.0:
        return 0.0  # -0.0 (Rust's sum of no fills) prints as "-0.0"
    return value


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_clean(obj), sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


# --- Rendering -----------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f} %"


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _ci(ci: Sequence[float] | None, digits: int = 2) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _dist(d: dict[str, Any] | None, fmt) -> str:  # noqa: ANN001
    return "n/a" if d is None else f"{fmt(d['median'])} [{fmt(d['p5'])}, {fmt(d['p95'])}]"


def _strategy_row(name: str, s: dict[str, Any]) -> str:
    return (
        f"| {name} | {_pct(s['total_return'], 0)} | {_pct(s['annualized_return'])} | {_num(s['sharpe'])} | "
        f"{_ci(s['sharpe_ci_analytic'])} | {_ci(s['sharpe_ci_bootstrap'])} | {_num(s['psr0'], 3)} | {_num(s['dsr'], 3)} | "
        f"{_num(s['sortino'])} | {_pct(s['max_drawdown'], 0)} | {_num(s['turnover'], 1)} | {_num(s['fees_paid'], 0)} | "
        f"{_pct(s['exposure'], 0)} | {s['fills']} |"
    )


def render(metrics: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any]) -> str:
    """``report-1.md`` from the run's records alone: no clock, no machine, no path."""
    cfg = config["config"]
    prov = config["provenance"]
    pairs = metrics["pairs"]
    files = [
        f
        for f in manifest.get("files", [])
        if f.get("kind") == "klines" and f.get("interval") == cfg["interval"] and f.get("symbol") in cfg["pairs"]
    ]
    if len(files) != len(cfg["pairs"]):
        raise ValueError(f"the manifest lists {len(files)} of the {len(cfg['pairs'])} files used")
    L: list[str] = []
    a = L.append
    a("# Report n°1: what fees do to simple strategies")
    a("")
    a("Generated by `uv run python -m olsprime.bench.report1`. Every number below comes from `metrics.json` of the same run;")
    a("re-run with `--check <run folder>` to reproduce it byte for byte, on the same machine and library versions (XGBoost's `hist`")
    a("with `nthread=1` is deterministic for one binary and CPU; nothing more was shown).")
    a("")
    a("## Provenance")
    a("")
    a(f"- Git commit: `{prov['git_commit']}`.")
    a(f"- Code digest (SHA-256 of {prov['code']['files']} source and lock files, independent of git): `{prov['code']['sha256']}`.")
    a(f"- Data manifest digest: `{prov['manifest_digest']}` ({manifest.get('totals', {}).get('files', '?')} files; vault open: {manifest.get('vault_open')}).")
    a("- Files used:")
    a("")
    a("| file | SHA-256 | stored rows | sealed rows | first sealed | last sealed |")
    a("|---|---|---|---|---|---|")
    for f in files:
        a(f"| `{f['path']}` | `{f['sha256']}` | {f['stored_rows']} | {f['sealed_rows']} | {str(f['first_sealed'])[:10]} | {str(f['last_sealed'])[:10]} |")
    a("")
    versions = ", ".join(f"{k} {v}" for k, v in config["versions"].items())
    a(f"- Libraries: {versions}.")
    n_variants = {(p, v): pairs[p]["venues"][v]["dsr"]["n_trials"] for p in pairs for v in cfg["venues"]}
    a(
        f"- Variants: {len(config['baseline_names'])} baselines in the fixed list (`BASELINE_NAMES`), of which "
        f"{len(DSR_CANDIDATES)} are DSR candidates (flat and the cheater excluded); random-turnover is one variant of "
        f"{cfg['random_seeds']} seeds. The DSR's N per pair and venue (candidates with a defined Sharpe) ranges from "
        f"{min(n_variants.values())} to {max(n_variants.values())}; see the DSR table."
    )
    a("")
    a("## Method")
    a("")
    t0, t1 = cfg["test_start"], cfg["test_end"]
    a(f"- **Data.** {', '.join(cfg['pairs'])}, {cfg['interval']} bars through `load_klines` (sealed), cut at {cfg['data_end'][:10]}: no bar of the validation year or the vault enters.")
    a("- **Models are per pair.** Each pair's frame, folds and fits use that pair's rows alone; BTCUSDT and ETHUSDT are never pooled.")
    a(
        f"- **Walk-forward.** Point-in-time features, three-class labels with h = {cfg['h']}, one refit per month over the test months "
        f"{t0[0]}-{t0[1]:02d} -> {t1[0]}-{t1[1]:02d}, expanding window, purge and embargo h. Rows before the window only train."
    )
    for p in cfg["pairs"]:
        fo, sc = pairs[p]["folds"], pairs[p]["scored"]
        a(
            f"  - {p}: {fo['count']} folds, {fo['n_train_min']}-{fo['n_train_max']} training rows; scored bars {sc['first'][:10]} -> "
            f"{sc['last'][:10]} ({sc['bars']} bars, {sc['returns']} daily returns), the same for every strategy and venue."
        )
    a(
        f"- **Execution.** A target decided at a bar's close executes at the next open, at the {cfg['liquidity']} rate, with "
        f"{cfg['slippage_bps']:g} bps of slippage per side and Binance's order filters ("
        + "; ".join(f"{k}: min notional {v['min_notional']:g}, step {v['step_size']:g}" for k, v in cfg["filters"].items())
        + f"). Initial cash {cfg['initial_cash']:,.0f} USDT. The first scored bar never trades, the last target never executes, and an open position "
        "is marked at the last close without an exit fee."
    )
    a(
        "- **Forecasters to targets.** climatology, ridge and xgboost go long when the expected return clears the venue's round trip, "
        "`2 * (taker + 2 bps)`, and back to flat when it turns negative (hysteresis). Their targets therefore depend on the venue."
    )
    a(
        "- **Statistics.** Sharpe ratios are annualized with sqrt(365) from daily simple returns, no risk-free rate. Analytic interval: "
        "Mertens (2002), iid. Bootstrap interval: stationary bootstrap (Politis and Romano 1994), Politis-White block length, "
        f"{cfg['n_boot']} draws, percentile; the block length reads the serial correlation of the returns, not of their squares. "
        "Sensitivity, for every paired interval of H9 and of the headline: the mean block max(PW(d), PW((d - mean d)^2)), with d "
        "the paired return difference, so that volatility clustering also sets the block. The registered analysis stays the primary. "
        "Diebold-Mariano: registered at h = 1 with no lag; sensitivity with a Newey-West (Bartlett) variance and floor(4 (T/100)^(2/9)) "
        "lags. PSR(0) and DSR after Bailey and Lopez de Prado (2012, 2014); for the DSR, N is the number of "
        "candidates with a defined Sharpe at that pair and venue and V the variance of their per-period Sharpes."
    )
    a(
        f"- **random-turnover.** {cfg['random_seeds']} seeds. Its line gives the median and the 5th and 95th percentiles over the seeds; "
        "its intervals, PSR, DSR and H9 comparison use the seed with the median Sharpe at that pair and venue (named in the tables)."
    )
    a(
        "- **Assumptions.** Today's fee schedules are applied to 2019-2024, a stated assumption, not a leak. Prices are Binance's USDT "
        "prices; Bitvavo, Coinbase and Kraken quote BTC and ETH in EUR or USD."
    )
    a("")
    for p in cfg["pairs"]:
        rc = pairs[p]["ridge_crosscheck"]
        a(
            f"- Ridge cross-check, {p}: on all {rc['folds']} folds the numpy ridge equals `olsprime_rs.ridge_fit` on the standardized "
            f"features; worst max abs difference {rc['worst_coef_max_abs_diff']:.2e} on the coefficients and "
            f"{rc['worst_fitted_max_abs_diff']:.2e} on the fitted values (tolerance {cfg['ridge_tolerance']:g})."
        )
    a("")

    # Headline
    a("## Headline: buy and hold against the best active baseline")
    a("")
    a("Net annualized Sharpe ratio with its stationary-bootstrap 95 % interval. The best active baseline is the one with the highest "
      "point Sharpe at that pair and venue: picking it is itself a selection, which the DSR column of the detailed tables accounts for.")
    a("")
    a("| pair | venue | buy and hold | best active baseline | its Sharpe | difference to B&H [95 % paired bootstrap] | sensitivity block | sensitivity 95 % |")
    a("|---|---|---|---|---|---|---|---|")
    for p in cfg["pairs"]:
        for v in cfg["venues"]:
            st = pairs[p]["venues"][v]["strategies"]
            bh = st[BUY_AND_HOLD]
            cands = [(st[n]["sharpe"], n) for n in ACTIVE if st[n]["sharpe"] is not None]
            if cands:
                best_sr, best = max(cands)
                d = pairs[p]["venues"][v]["h9"][best]
                label = f"{best} (seed {st[RANDOM]['seed']})" if best == RANDOM else best
                sens = d.get("sensitivity") or {}
                a(f"| {p} | {v} | {_num(bh['sharpe'])} {_ci(bh['sharpe_ci_bootstrap'])} | {label} | {_num(best_sr)} | "
                  f"{_num(d.get('difference'))} {_ci(d.get('ci'))} (block {_num(d.get('mean_block'), 0)}) | {_num(sens.get('mean_block'), 0)} | {_ci(sens.get('ci'))} |")
            else:
                a(f"| {p} | {v} | {_num(bh['sharpe'])} {_ci(bh['sharpe_ci_bootstrap'])} | none defined | n/a | n/a | n/a | n/a |")
    a("")

    # H9
    a("## H9")
    a("")
    a("H9 (registered 2026-10-27): at the entry-tier taker fees of Coinbase Advanced and Kraken Pro, no active baseline beats buy and hold "
      "on BTCUSDT or ETHUSDT. It holds at a venue if no paired 95 % interval of (net Sharpe - B&H's net Sharpe) lies entirely above 0, "
      "and is refuted if one does. The other venues are **descriptive**: H9 names only those two.")
    a("")
    a("| venue | status | comparisons | verdict | intervals entirely above 0 | sensitivity verdict | sensitivity intervals above 0 |")
    a("|---|---|---|---|---|---|---|")
    for v in cfg["venues"]:
        h = metrics["h9"][v]
        status = "registered" if h["registered"] else "descriptive"

        def word(holds: bool, registered: bool = h["registered"]) -> str:
            return ("holds" if holds else "refuted") if registered else ("no interval above 0" if holds else "an interval above 0")

        counts = f"{h['comparisons']}: {h['with_interval']} with an interval, {h['undefined']} undefined (never exposed)"
        a(f"| {v} | {status} | {counts} | {word(h['holds'])} | {', '.join(h['intervals_above_zero']) or 'none'} | "
          f"{word(h['sensitivity_holds'])} | {', '.join(h['sensitivity_intervals_above_zero']) or 'none'} |")
    a("")
    changed = [v for v in cfg["venues"] if metrics["h9"][v]["holds"] != metrics["h9"][v]["sensitivity_holds"]]
    a("The block-length sensitivity changes the verdict at: " + (", ".join(changed) if changed else "no venue") + ".")
    a("")
    for v in cfg["venues"]:
        tag = "registered" if v in cfg["h9_venues"] else "descriptive"
        a(f"### {v} ({tag})")
        a("")
        a("| pair | baseline | net Sharpe | B&H net Sharpe | difference | 95 % paired bootstrap | block | p | above 0 | sensitivity block | sensitivity 95 % | above 0 (sensitivity) |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in cfg["pairs"]:
            st = pairs[p]["venues"][v]["strategies"]
            for n in ACTIVE:
                d = pairs[p]["venues"][v]["h9"][n]
                label = f"{n} (seed {st[RANDOM]['seed']})" if n == RANDOM else n
                if d["defined"]:
                    sens = d["sensitivity"]
                    a(f"| {p} | {label} | {_num(st[n]['sharpe'])} | {_num(st[BUY_AND_HOLD]['sharpe'])} | {_num(d['difference'])} | "
                      f"{_ci(d['ci'])} | {_num(d['mean_block'], 0)} | {_num(d['p_value'], 3)} | {'yes' if d['above_zero'] else 'no'} | "
                      f"{sens['mean_block']} | {_ci(sens['ci'])} | {'yes' if sens['above_zero'] else 'no'} |")
                else:
                    a(f"| {p} | {label} | n/a | {_num(st[BUY_AND_HOLD]['sharpe'])} | n/a | {d['reason']} | n/a | n/a | no | n/a | n/a | no |")
        a("")

    # Trading tables
    a("## Trading metrics")
    a("")
    a("Net of fees and slippage. Turnover is traded notional over mean equity per year (a full round trip adds 2). Fees in USDT on "
      f"{cfg['initial_cash']:,.0f} of initial cash. PSR(0) and DSR are probabilities. n/a: undefined (no variance, or not a DSR candidate). "
      "Fills: the number of orders executed. A strategy with a handful of fills still gets a Sharpe and intervals; read them with "
      "that count in mind. No row is filtered.")
    a("")
    head = "| strategy | net total | net ann. | Sharpe | 95 % analytic | 95 % bootstrap | PSR(0) | DSR | Sortino | max DD | turnover/yr | fees | exposure | fills |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    for p in cfg["pairs"]:
        for v in cfg["venues"]:
            st = pairs[p]["venues"][v]["strategies"]
            a(f"### {p}, {v}")
            a("")
            a(head)
            a(sep)
            for n in BASELINE_NAMES:
                if n in (CHEATER, RANDOM):
                    continue
                a(_strategy_row(n, st[n]))
            rt = st[RANDOM]
            a(_strategy_row(f"{RANDOM}, median-Sharpe seed {rt['seed']}", rt))
            dist = rt["distribution"]
            a(
                f"| {RANDOM}, {dist['seeds']} seeds: median [5 %, 95 %] | {_dist(dist['total_return'], lambda x: _pct(x, 0))} | "
                f"{_dist(dist['annualized_return'], _pct)} | {_dist(dist['sharpe'], _num)} | | | | | {_dist(dist['sortino'], _num)} | "
                f"{_dist(dist['max_drawdown'], lambda x: _pct(x, 0))} | {_dist(dist['turnover'], lambda x: _num(x, 1))} | "
                f"{_dist(dist['fees_paid'], lambda x: _num(x, 0))} | {_dist(dist['exposure'], lambda x: _pct(x, 0))} | "
                f"{_dist(dist.get('fills'), lambda x: f'{x:.0f}')} |"
            )
            a("")
            a("H1 control, not a candidate (it reads the next bar on purpose; see `h1.md`):")
            a("")
            a(head)
            a(sep)
            a(_strategy_row("cheater (LEAKS ON PURPOSE)", st[CHEATER]))
            a("")
            cap = st[CHEATER].get("lot_cap_first_binds")
            if cap:
                a(f"The cheater's book reaches the simulator's cap of 2^50 lots on {cap}, the first scored bar where its held "
                  "position sits at the cap: from then on its buys are clamped, so its trading figures are not a measurement.")
            else:
                a("The cheater's book never reaches the simulator's cap of 2^50 lots here (its held position is below the cap "
                  "at every scored bar); its figures are still those of a leak, "
                  "not a measurement.")
            a("")

    # DSR
    a("## Deflated Sharpe: the variants counted")
    a("")
    a("| pair | venue | N | V (per-period Sharpe variance) | variants with a defined Sharpe |")
    a("|---|---|---|---|---|")
    for p in cfg["pairs"]:
        for v in cfg["venues"]:
            d = pairs[p]["venues"][v]["dsr"]
            vv = "n/a" if d["var_sr_per_period"] is None else f"{d['var_sr_per_period']:.3e}"
            a(f"| {p} | {v} | {d['n_trials']} | {vv} | {', '.join(d['variants'])} |")
    a("")
    a("With N this small, V is itself a noisy estimate, and so is the DSR that uses it.")
    a("")

    # Forecasts
    a("## Forecasts (venue-independent)")
    a("")
    a("On the scored decisions whose label is known before the end of the development period and for which all three forecasters "
      "give probabilities. BSS: Brier skill score against climatology (fitted in each fold), paired stationary-bootstrap 95 % interval. "
      "ECE: 10 equal-width bins, top-label and class-wise. Diebold-Mariano on the per-row log-loss against climatology (registered: "
      "h = 1, no lag, HLN correction; a negative statistic means a lower loss than climatology). Sensitivity: the same test with a "
      "Newey-West (Bartlett) variance and floor(4 (T/100)^(2/9)) lags.")
    a("")
    a("| pair | model | rows | log-loss | Brier | BSS [95 %] | ECE top | ECE class-wise | DM stat | DM p | DM NW lags | DM NW stat | DM NW p |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in cfg["pairs"]:
        fc = pairs[p]["forecasts"]
        for n in FORECASTERS:
            m = fc["models"][n]
            bss = _num(m["bss"], 4) + (" " + _ci(m["bss_ci"], 4) if "bss_ci" in m else " (reference)")
            dm = m.get("dm_log_loss")
            nw = m.get("dm_log_loss_newey_west")
            a(
                f"| {p} | {n} | {fc['rows']} | {m['log_loss']:.4f} | {m['brier']:.4f} | {bss} | {m['ece_top']:.4f} | "
                f"{m['ece_classwise']:.4f} | {_num(dm['statistic'] if dm else None)} | {_num(dm['p_value'] if dm else None, 3)} | "
                f"{nw['lags'] if nw else 'n/a'} | {_num(nw['statistic'] if nw else None)} | {_num(nw['p_value'] if nw else None, 3)} |"
            )
    a("")
    for p in cfg["pairs"]:
        fc = pairs[p]["forecasts"]
        freq = ", ".join(f"{x:.3f}" for x in fc["class_frequencies"])
        a(f"- {p}: class frequencies (down, neutral, up) on the scored rows: {freq}; rows dropped: {fc['dropped_rows']}.")
    a("")
    return "\n".join(L) + "\n"


# --- Check --------------------------------------------------------------------------------


def compared_files(folder: Path) -> list[str]:
    names = [n for n in COMPARED if (folder / n).exists()]
    names += sorted(f"equity/{p.name}" for p in (folder / "equity").glob("*.csv")) if (folder / "equity").is_dir() else []
    return names


def diff_runs(expected: Path, actual: Path, *, max_lines: int = 40) -> list[str]:
    """Human-readable differences between two run folders (``run.json`` excluded); empty when identical."""
    problems: list[str] = []
    a, b = compared_files(expected), compared_files(actual)
    for name in sorted(set(a) ^ set(b)):
        problems.append(f"{name}: only in {'the expected run' if name in a else 'the re-run'}")
    for name in sorted(set(a) & set(b)):
        x, y = (expected / name).read_bytes(), (actual / name).read_bytes()
        if x == y:
            continue
        lines = list(
            difflib.unified_diff(
                x.decode("utf-8", "replace").splitlines(),
                y.decode("utf-8", "replace").splitlines(),
                f"expected/{name}",
                f"re-run/{name}",
                lineterm="",
                n=1,
            )
        )
        problems.append(f"{name}: differs\n" + "\n".join(lines[:max_lines]) + ("\n..." if len(lines) > max_lines else ""))
    return problems


def check(expected: Path, cfg: Report1Config, *, workers: int) -> int:
    if not (expected / "metrics.json").is_file():
        print(f"{expected}: not a report1 run folder (no metrics.json)")
        return 2
    scratch = Path(tempfile.mkdtemp(prefix="report1-check-"))
    try:
        run_report(cfg, scratch, workers=workers)
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
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-report1"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.report1", description="Report no. 1: what fees do to simple strategies.")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default: <runs>/<UTC date>-report1, suffixed if taken)")
    parser.add_argument("--check", type=Path, default=None, metavar="DIR", help="re-run and compare with the run in DIR")
    parser.add_argument("--workers", type=int, default=2, help="processes, one per pair (results do not depend on it)")
    parser.add_argument("--no-docs", action="store_true", help="do not copy the report to docs/reports/")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Report1Config()
    if args.check is not None:
        return check(args.check, cfg, workers=args.workers)
    out = args.out or _default_out()
    started = time.perf_counter()
    metrics = run_report(cfg, out, workers=args.workers)
    if not args.no_docs:
        docs = REPO_ROOT / "docs" / "reports"
        docs.mkdir(parents=True, exist_ok=True)
        # Written from the run's records in memory, the same bytes as the run folder's copy.
        (docs / REPORT_FILE).write_bytes((out / REPORT_FILE).read_bytes())
    verdicts = ", ".join(f"{v}: {'holds' if h['holds'] else 'refuted'}" for v, h in metrics["h9"].items() if h["registered"])
    print(f"report1 written to {out} in {time.perf_counter() - started:.0f} s; H9 {verdicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
