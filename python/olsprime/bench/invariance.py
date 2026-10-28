"""H1, future invariance: the bench catches leakage (``docs/HYPOTHESES.md``, H1).

For a pair, a model and a cut bar t:

1. run the whole pipeline (features, labels, folds, fit, predict, decision rule) on the bars;
2. perturb every value of every bar after t (:func:`olsprime.bench.perturb.perturb_after`),
   in **both directions**: once with the future tripled and once divided by three, each with its
   own random walk and factors, and the same body swap on the first later bar;
3. rebuild everything from the perturbed bars and run the pipeline again;
4. every output decided at or before ``known_at[t]`` must be identical, bit for bit:
   ``p_down``, ``p_neutral``, ``p_up``, ``exp_return`` and the ``target`` at every venue.

A row that differs under either direction is a **violation**; the flags are kept per direction.
An honest model has none at any cut.

**Controls.** H1 holds only if the bench also catches what it must, exactly where the design
predicts (:func:`predicted`), and can see a signal when there is one. A control is **caught** at a
cut only where an output moves by more than :data:`CAUGHT_ABOVE`, or turns missing, or stops being
missing: a change at the level of rounding carries no leak, so it never counts as a catch. (Before
this rule, a control whose leak its inner model undid exactly was caught by rounding alone, at every
cut on Windows and at all but one on Linux.)

- the cheater (reads the next bar's candle), the next-bar peekers on ``trades``, ``quote_volume``
  and the taker share, and the price off-by-one (:mod:`olsprime.bench.leaks`): every cut;
- the transductive scaler (features scaled with the whole test month, ahead of a ridge that
  centres but does not rescale them): every cut whose month goes on after it, never at the last
  bar of a month;
- the models fitted without purge or embargo (:func:`olsprime.bench.pipeline.unpurged`): exactly
  the cuts at the first bar of a test month. A model fitted at ``fit_time`` without the purge
  trains on the label of the bar just before the fit, whose exit price is the open of the second
  bar of the month: only a cut at the month's first bar perturbs that open while the month's
  first decision is still compared;
- the **positive control** (:func:`positive_control`): ridge fitted on labels misaligned one bar
  early must reach a Brier skill score above ``positive_bss_min`` against climatology on every
  pair, or the pipeline is dead and H1 says so.

**The limit.** Perturbation detects a leak only through a column it perturbs. A model that
memorizes the future by its time stamps, such as a calendar learned from the future or a lookup
keyed on ``open_time``, passes this test; only code review and the seal stop it.

The pipeline sees the bars final before the end of the development period only; the
validation year stays out. The one exception is :func:`blind_cheater`.

Only the folds with ``test_start <= known_at[t]`` are re-run on the perturbed bars: the later
folds have no compared output, and the decision rule carries its state forward only.

Usage::

    uv run python -m olsprime.bench.invariance [--out DIR] [--workers N]

It runs BTCUSDT and ETHUSDT 1d over the test months 2019-01 -> 2024-12 and writes ``h1.json``,
``h1.md`` and ``run.json`` to ``runs/<UTC date>-h1`` (suffixed if taken) or to ``--out``. ``h1.json`` carries the
same provenance as Report n°1 under ``provenance``: the git commit, the code digest and the data manifest digest.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from olsprime.bench import pipeline as pp
from olsprime.bench import provenance
from olsprime.bench.leaks import PROBES, OffByOneReturn, Probe, ProbePeeker, TransductiveScaler
from olsprime.bench.perturb import (
    CLV_FLOOR,
    FILL_FLOOR,
    SMALL_BODY,
    DOWN_CLV,
    DOWN_FILL,
    DOWN_FLOOR,
    PUSH,
    UP_CAP,
    UP_CLV,
    UP_FILL,
    WINDOW,
    Perturbation,
    perturb_after,
)
from olsprime.data.binance_vision import Month
from olsprime.eval import stats
from olsprime.eval.splits import Fold
from olsprime.models.baselines import (
    MA_WINDOWS,
    Baseline,
    Cheater,
    Climatology,
    MaMomentum,
    RandomTurnover,
    Ridge,
    XGBoost,
    buy_and_hold,
    flat,
)
from olsprime.paths import runs_dir

log = logging.getLogger(__name__)

# What a model must show, cut by cut (see predicted()).
PASS = "pass"  # never flagged
FLAG_EVERY_CUT = "flag-every-cut"
FLAG_FIRST_BAR = "flag-first-bar-of-test-month"  # the planted purge leak
FLAG_UNLESS_MONTH_END = "flag-unless-last-bar-of-month"  # the transductive scaler
FLAG_BY_CELL = "flag-where-a-run-changes-its-cell"  # the colour x size probes, cut by cut

# A control is caught at a cut only where an output moves by more than this (see compare()). Rounding moves
# them by about 1e-16; the weakest real catch on the synthetic bars moves a probability by 2.7e-3.
CAUGHT_ABOVE = 1e-9

LAST_BEFORE_FOLD = "last-bar-before-fold-start"
FIRST_OF_MONTH = "first-bar-of-test-month"
RANDOM = "random-bar"


@dataclass(frozen=True)
class ModelSpec:
    """A model under test: a factory built from the bars (the leaks need them), and what must happen."""

    name: str
    make: Callable[[pl.DataFrame], Callable[[], Baseline]]
    expect: str = PASS
    unpurged: bool = False  # fitted on folds without purge or embargo: a planted leak
    probe: Probe | None = None  # a peeker: its cuts are ``?`` where the perturbation could not cross it


@dataclass(frozen=True)
class Cut:
    index: int  # row position in the pair's bars
    open_time: datetime
    known_at: datetime
    kind: str
    opens_fold: bool = False  # the cut bar opens at a fold's fit_time: the first bar of a test month
    month_continues: bool = True  # the next bar is in the same calendar month


@dataclass(frozen=True)
class H1Config:
    pairs: tuple[str, ...] = pp.PAIRS
    interval: str = "1d"
    h: int = 1
    start: tuple[int, int] = (2019, 1)
    end: tuple[int, int] = (2024, 12)
    venues: tuple[str, ...] = pp.VENUES
    random_turnover_seeds: tuple[int, ...] = (0, 1, 2)
    cut_seed: int = 20261027
    perturbation_seed: int = 7
    perturbation: Perturbation = Perturbation()
    # Both directions at every cut: the future tripled and divided by three.
    level_shifts: tuple[float, ...] = (3.0, 1.0 / 3.0)
    positive_bss_min: float = 0.5
    per_pair_models: bool = True  # each pair's frame, folds and fits use that pair's rows alone


DIRECTION_NAMES = {3.0: "x3", 1.0 / 3.0: "/3"}


def _direction(shift: float) -> str:
    return DIRECTION_NAMES.get(shift, f"x{shift:g}")


def _const(factory: Callable[[], Baseline]) -> Callable[[pl.DataFrame], Callable[[], Baseline]]:
    return lambda bars: factory


def model_specs(interval: str, seeds: Sequence[int] = (0, 1, 2)) -> list[ModelSpec]:
    """Every honest baseline (three random-turnover seeds), then every control H1 must catch."""
    specs = [
        ModelSpec("flat", _const(flat)),
        ModelSpec("buy-and-hold", _const(buy_and_hold)),
        *(ModelSpec(f"ma-momentum-{w}", _const(lambda w=w: MaMomentum(w))) for w in MA_WINDOWS),
        *(ModelSpec(f"random-turnover-seed{s}", _const(lambda s=s: RandomTurnover(s, interval))) for s in seeds),
        ModelSpec("climatology", _const(Climatology)),
        ModelSpec("ridge", _const(Ridge)),
        ModelSpec("xgboost", _const(XGBoost)),
        # !!! Deliberate leaks below: the controls H1 must catch, each where the design predicts. !!!
        ModelSpec("cheater", lambda bars: (lambda: Cheater(bars)), expect=FLAG_EVERY_CUT),
        *(
            ModelSpec(p.name, lambda bars, p=p: (lambda: ProbePeeker(bars, p)), expect=FLAG_BY_CELL if p.cell else FLAG_EVERY_CUT, probe=p)
            for p in PROBES
        ),
        ModelSpec("off-by-one-return", lambda bars: (lambda: OffByOneReturn(bars)), expect=FLAG_EVERY_CUT),
        ModelSpec("transductive-scaler", _const(TransductiveScaler), expect=FLAG_UNLESS_MONTH_END),
        ModelSpec("climatology-unpurged", _const(Climatology), expect=FLAG_FIRST_BAR, unpurged=True),
        ModelSpec("ridge-unpurged", _const(Ridge), expect=FLAG_FIRST_BAR, unpurged=True),
        ModelSpec("xgboost-unpurged", _const(XGBoost), expect=FLAG_FIRST_BAR, unpurged=True),
    ]
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    return specs


def predicted(expect: str, cut: Cut | dict[str, Any]) -> bool:
    """Whether a model with this expectation must be flagged at this cut."""
    get = cut.get if isinstance(cut, dict) else lambda k: getattr(cut, k)
    if expect == PASS:
        return False
    if expect == FLAG_EVERY_CUT:
        return True
    if expect == FLAG_FIRST_BAR:
        return bool(get("opens_fold"))
    if expect == FLAG_UNLESS_MONTH_END:
        return bool(get("month_continues"))
    if expect == FLAG_BY_CELL:
        return bool(get("predicted_by_cell"))
    raise ValueError(f"unknown expectation {expect!r}")


# --- Cuts ----------------------------------------------------------------------------


def choose_cuts(bars: pl.DataFrame, folds: Sequence[Fold], *, seed: int, n_random: int = 2) -> list[Cut]:
    """At least eight cuts: the last bar before three fold starts, the first bar of three test months, random bars.

    Positions are spread over the folds; the random bars are drawn among the scored bars
    (decisions of the test months) that have a later bar. Each cut records whether it opens a
    fold and whether its month goes on after it, which the controls' predictions use.
    """
    if len(folds) < 4:
        raise ValueError("need at least four folds to place the cuts")
    open_ms = bars["open_time"].dt.epoch("ms").to_numpy()
    known = bars["known_at"]
    fit_ms = {int(f.fit_time.timestamp() * 1000) for f in folds}

    def position(when: datetime) -> int:
        return int(np.searchsorted(open_ms, int(when.timestamp() * 1000)))

    n = len(folds)
    picks: list[tuple[int, str]] = []
    for f in sorted({max(1, n // 4), n // 2, min(n - 1, (3 * n) // 4)}):  # fold 0's previous bar is not scored
        picks.append((position(folds[f].test_start) - 1, LAST_BEFORE_FOLD))
    for f in sorted({0, n // 3, min(n - 1, (2 * n) // 3 + 1)}):
        picks.append((position(folds[f].test_start), FIRST_OF_MONTH))
    first, last = position(folds[0].test_start), min(position(folds[-1].test_end), bars.height - 1)
    rng = np.random.default_rng(seed)
    taken = {i for i, _ in picks}
    while sum(kind == RANDOM for _, kind in picks) < n_random:
        i = int(rng.integers(first, last))
        if i not in taken:
            picks.append((i, RANDOM))
            taken.add(i)
    cuts = []
    for i, kind in picks:
        if not 0 <= i < bars.height - 1:
            raise ValueError(f"cut at row {i} has no later bar")
        here, nxt = bars["open_time"][i], bars["open_time"][i + 1]
        cuts.append(
            Cut(
                i,
                here,
                known[i],
                kind,
                opens_fold=int(open_ms[i]) in fit_ms,
                month_continues=(here.year, here.month) == (nxt.year, nxt.month),
            )
        )
    return cuts


# --- Running one model -------------------------------------------------------------------


def outputs(spec: ModelSpec, bars: pl.DataFrame, cfg: H1Config, *, before: datetime | None = None) -> pl.DataFrame:
    """The compared outputs of one model on one pair's bars.

    ``decision_time``, ``p_*``, ``exp_return`` and ``target_<venue>`` for every venue. With
    ``before``, only the folds whose test month starts at or before it are run.
    """
    frame = pp.build_frame(bars, cfg.interval, cfg.h)
    folds = pp.make_folds(frame, cfg.interval, cfg.h, Month(*cfg.start), Month(*cfg.end))
    if spec.unpurged:
        folds = pp.unpurged(frame, folds)
    if before is not None:
        folds = [f for f in folds if f.test_start <= before]
    preds = pp.walk_forward_predict(spec.make(bars), frame, folds)
    cols = [pl.col("decision_time"), *(pl.col(c) for c in pp.OUTPUT_COLUMNS[:-1])]
    out = preds.select(cols)
    return out.with_columns([pl.Series(f"target_{v}", pp.targets(preds, v), dtype=pl.Float64) for v in cfg.venues])


def _values(series: pl.Series) -> np.ndarray:
    return series.cast(pl.Float64).fill_null(np.nan).to_numpy()


def _bits(series: pl.Series) -> np.ndarray:
    return _values(series).view(np.uint64)


def compare(original: pl.DataFrame, perturbed: pl.DataFrame, until: datetime) -> dict[str, Any]:
    """Bit-for-bit comparison of the rows decided at or before ``until``.

    ``mask`` marks the rows that differ by a single bit (``violations``). ``caught_mask`` marks the
    rows where an output moves by more than :data:`CAUGHT_ABOVE`, turns missing or stops being missing
    (``caught``). ``largest`` is the largest change between two present values (``inf`` when the rows
    themselves differ).
    """
    a = original.filter(pl.col("decision_time") <= until)
    b = perturbed.filter(pl.col("decision_time") <= until)
    if a.height == 0:
        raise ValueError(f"no output decided at or before {until}: the cut is outside the scored window")
    if not a["decision_time"].equals(b["decision_time"]):
        every = np.ones(a.height, dtype=bool)
        return {"rows": a.height, "violations": a.height, "caught": a.height, "largest": float("inf"),
                "columns": {"decision_time": a.height}, "first": None, "mask": every, "caught_mask": every}  # fmt: skip
    differs = np.zeros(a.height, dtype=bool)
    above = np.zeros(a.height, dtype=bool)
    largest = 0.0
    per_column = {}
    for c in a.columns:
        if c == "decision_time":
            continue
        d = _bits(a[c]) != _bits(b[c])
        per_column[c] = int(d.sum())
        differs |= d
        x, y = _values(a[c]), _values(b[c])
        both = ~np.isnan(x) & ~np.isnan(y)
        change = np.abs(x[both] - y[both])
        above[both] |= change > CAUGHT_ABOVE
        above |= np.isnan(x) != np.isnan(y)
        largest = max(largest, float(change.max(initial=0.0)))
    first = a["decision_time"][int(np.argmax(differs))] if differs.any() else None
    return {
        "rows": a.height,
        "violations": int(differs.sum()),
        "caught": int(above.sum()),
        "largest": largest,
        "columns": {k: v for k, v in per_column.items() if v},
        "first": None if first is None else first.isoformat(),
        "mask": differs,
        "caught_mask": above,
    }


def run_model(spec: ModelSpec, bars: pl.DataFrame, cuts: Sequence[Cut], cfg: H1Config) -> list[dict[str, Any]]:
    """One model, every cut, both directions: the comparison record of each cut.

    ``violations`` counts the rows that differ under either direction; ``by_direction`` gives
    the count under each (``x3``, ``/3``). ``caught`` and ``caught_by_direction`` count the rows
    that move by more than rounding (:func:`compare`), and ``largest_change`` is the largest move.
    """
    original = outputs(spec, bars, cfg)
    records = []
    for k, cut in enumerate(cuts):
        # ``?`` is predicted from the original bars, before perturbing: a declared reason or nothing.
        mirror = is_mirrored(k)
        unknown = spec.probe.unknown_reason(bars, cut.index, mirror) if spec.probe is not None else None
        by_cell = spec.probe.predicted_flag(bars, cut.index, mirror) if spec.probe is not None else None
        results = {}
        for j, shift in enumerate(cfg.level_shifts):
            config = replace(cfg.perturbation, level_shift=shift, mirror=mirror)
            seed = cfg.perturbation_seed + 1000 * j + k
            perturbed = outputs(spec, perturb_after(bars, cut.index, seed=seed, config=config), cfg, before=cut.known_at)
            results[_direction(shift)] = compare(original, perturbed, cut.known_at)
        union = np.logical_or.reduce([r["mask"] for r in results.values()])
        caught = np.logical_or.reduce([r["caught_mask"] for r in results.values()])
        columns: dict[str, int] = {}
        for r in results.values():
            for c, v in r["columns"].items():
                columns[c] = max(columns.get(c, 0), v)
        firsts = [r["first"] for r in results.values() if r["first"] is not None]
        records.append(
            {
                "cut": cut.index,
                "kind": cut.kind,
                "opens_fold": cut.opens_fold,
                "month_continues": cut.month_continues,
                "rows": next(iter(results.values()))["rows"],
                "violations": int(union.sum()),
                "by_direction": {d: r["violations"] for d, r in results.items()},
                "caught": int(caught.sum()),
                "caught_by_direction": {d: r["caught"] for d, r in results.items()},
                "largest_change": max(r["largest"] for r in results.values()),
                "columns": columns,
                "first": min(firsts) if firsts else None,
                "unknown_reason": unknown,
                "mirror": mirror,
                **({"predicted_by_cell": by_cell} if by_cell is not None else {}),
            }
        )
    return records


def is_mirrored(k: int) -> bool:
    """The k-th cut of the bench's list (0-based) takes the mirror shape when k is odd: fixed and deterministic."""
    return k % 2 == 1


def _cell(b: pl.DataFrame, row: int) -> list[str]:
    o, c = float(b["open"][row]), float(b["close"][row])
    colour = "green" if c > o else "red" if c < o else "doji"
    return [colour, "small" if abs(np.log(c / o)) < SMALL_BODY else "big"]


def realised_cells(bars: pl.DataFrame, cuts: Sequence[Cut], cfg: H1Config) -> list[dict[str, Any]]:
    """Per cut: the (colour, size) of bar t + 1 in the original and in each run, read on the perturbed bars."""
    out = []
    for k, cut in enumerate(cuts):
        row = {"cut": cut.index, "mirror": is_mirrored(k), "original": _cell(bars, cut.index + 1)}
        for j, shift in enumerate(cfg.level_shifts):
            config = replace(cfg.perturbation, level_shift=shift, mirror=is_mirrored(k))
            seed = cfg.perturbation_seed + 1000 * j + k
            row[_direction(shift)] = _cell(perturb_after(bars, cut.index, seed=seed, config=config), cut.index + 1)
        out.append(row)
    return out


def verdict(spec: ModelSpec, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Whether the model showed exactly its predicted pattern: flagged at the predicted cuts and nowhere else.

    An honest model is flagged by a single differing bit. A control is flagged only where it is caught,
    an output moving by more than :data:`CAUGHT_ABOVE`: a control flagged by rounding alone is off its pattern.
    """
    honest = spec.expect == PASS
    count, per_direction = ("violations", "by_direction") if honest else ("caught", "caught_by_direction")
    flagged = [r[count] > 0 for r in records]
    # ``?``: a declared reason, predicted from the original bars, that the probe cannot cross at that cut.
    expected = [None if r.get("unknown_reason") else predicted(spec.expect, r) for r in records]
    directions = list(records[0][per_direction]) if records else []
    known = [(f, e) for f, e in zip(flagged, expected, strict=True) if e is not None]
    return {
        "model": spec.name,
        "expect": spec.expect,
        "flagged_cuts": int(sum(flagged)),
        "predicted_cuts": int(sum(bool(e) for e in expected)),
        "unknown_cuts": [r["cut"] for r, e in zip(records, expected, strict=True) if e is None],
        "unknown_reasons": {str(r["cut"]): r["unknown_reason"] for r in records if r.get("unknown_reason")},
        "flagged_by_direction": {d: int(sum(r[per_direction][d] > 0 for r in records)) for d in directions},
        "cuts": len(records),
        "violations": int(sum(r["violations"] for r in records)),
        "caught_rows": int(sum(r["caught"] for r in records)),
        # A control's weakest catch: over its caught cuts, the smallest of each cut's largest move.
        "smallest_catch": None if honest else min((r["largest_change"] for r, f in zip(records, flagged, strict=True) if f), default=None),
        "rounding_only_cuts": [r["cut"] for r in records if r["violations"] > 0 and r["caught"] == 0],
        "pattern": "".join("X" if f else "." for f in flagged),
        "predicted_pattern": "".join("?" if e is None else "X" if e else "." for e in expected),
        "ok": all(f == e for f, e in known),
    }


def run_pair(bars: pl.DataFrame, cfg: H1Config, specs: Sequence[ModelSpec], *, cuts: Sequence[Cut] | None = None) -> dict[str, Any]:
    """Every model on one pair's bars (serially): the cuts, the records and the verdicts."""
    frame = pp.build_frame(bars, cfg.interval, cfg.h)
    folds = pp.make_folds(frame, cfg.interval, cfg.h, Month(*cfg.start), Month(*cfg.end))
    cuts = list(cuts) if cuts is not None else choose_cuts(bars, folds, seed=cfg.cut_seed)
    models = {}
    for spec in specs:
        records = run_model(spec, bars, cuts, cfg)
        models[spec.name] = {"verdict": verdict(spec, records), "records": records}
    return {"cuts": [_cut_json(c) for c in cuts], "folds": len(folds), "models": models, "cells": realised_cells(bars, cuts, cfg)}


def _cut_json(cut: Cut) -> dict[str, Any]:
    return {
        "index": cut.index,
        "open_time": cut.open_time.isoformat(),
        "known_at": cut.known_at.isoformat(),
        "kind": cut.kind,
        "opens_fold": cut.opens_fold,
        "month_continues": cut.month_continues,
    }


# --- The positive control -----------------------------------------------------------------


def shift_labels_early(frame: pl.DataFrame) -> pl.DataFrame:
    """``label[t] := label[t - 1]`` and the same for ``fwd_return``: labels misaligned one bar early.

    The label of row t then measures ``open[t] -> open[t + 1]``, which overlaps bar t itself and
    is nearly computable from the features at t. A pipeline that can see a signal must find it.
    The times of the labels are not moved, so the purge keeps its meaning.
    """
    return frame.with_columns([pl.col(c).shift(1).over("symbol") for c in ("fwd_return", "label")])


def positive_control(bars: pl.DataFrame, cfg: H1Config) -> dict[str, Any]:
    """Ridge's Brier skill against climatology, both refitted per fold, with real and with early-shifted labels."""
    frame = pp.build_frame(bars, cfg.interval, cfg.h)
    out: dict[str, Any] = {"threshold": cfg.positive_bss_min}
    for tag, fr in (("real_labels", frame), ("labels_one_bar_early", shift_labels_early(frame))):
        folds = pp.make_folds(fr, cfg.interval, cfg.h, Month(*cfg.start), Month(*cfg.end))
        ridge = pp.walk_forward_predict(Ridge, fr, folds)
        clim = pp.walk_forward_predict(Climatology, fr, folds)
        idx = np.concatenate([f.test_idx for f in folds])
        label = fr[idx]["label"]
        p = ridge.select("p_down", "p_neutral", "p_up").to_numpy().astype(np.float64)
        q = clim.select("p_down", "p_neutral", "p_up").to_numpy().astype(np.float64)
        ok = label.is_not_null().to_numpy() & np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
        y = label.to_numpy()[ok].astype(np.int64)
        out[tag] = {"rows": int(ok.sum()), "bss": stats.brier_skill(p[ok], q[ok], y)}
    out["passes"] = bool(out["labels_one_bar_early"]["bss"] > cfg.positive_bss_min)
    return out


def h1_holds(pairs: dict[str, dict[str, Any]], specs: Sequence[ModelSpec], positive: dict[str, Any] | None = None) -> dict[str, Any]:
    """H1's verdict: honest models never flagged, every control flagged exactly as predicted, the pipeline alive."""
    honest = [s.name for s in specs if s.expect == PASS]
    controls = [s.name for s in specs if s.expect != PASS]
    honest_violations = sum(p["models"][m]["verdict"]["violations"] for p in pairs.values() for m in honest)
    wrong = [f"{pair}:{m}" for pair, p in pairs.items() for m in controls if not p["models"][m]["verdict"]["ok"]]
    cheater = [p["models"]["cheater"]["verdict"] for p in pairs.values() if "cheater" in p["models"]]
    alive = None if positive is None else all(v["passes"] for v in positive.values())
    unknown = {f"{pair}:{m}": p["models"][m]["verdict"].get("unknown_cuts", []) for pair, p in pairs.items() for m in controls}
    # A control that is ``?`` at every cut of a pair tested nothing there: H1 cannot hold on it.
    vacuous = [
        f"{pair}:{m}" for pair, p in pairs.items() for m in controls
        if p["models"][m]["verdict"]["cuts"] and len(p["models"][m]["verdict"].get("unknown_cuts", [])) == p["models"][m]["verdict"]["cuts"]
    ]  # fmt: skip
    return {
        "unknown_cuts": {k: v for k, v in unknown.items() if v},
        "controls_unknown_at_every_cut": vacuous,
        "honest_models": honest,
        "controls": controls,
        "honest_violations": int(honest_violations),
        "controls_off_pattern": wrong,
        "cheater_flagged_cuts": int(sum(c["flagged_cuts"] for c in cheater)),
        "cheater_cuts": int(sum(c["cuts"] for c in cheater)),
        "pipeline_alive": alive,
        "holds": bool(honest_violations == 0 and not wrong and not vacuous and alive is not False),
    }


# --- The seal leaves the cheater blind -------------------------------------------------


def blind_cheater(bars: pl.DataFrame, cfg: H1Config) -> dict[str, Any]:
    """The cheater at the last sealed bar: the seal leaves it blind there.

    **Fenced.** This is the one place of the bench that builds features and labels over the whole
    sealed series, the validation year included, and it does so only to show the seal. No model
    is fitted here, and the only value it reports from 2025 is the cheater's target at the last
    bar, which is 0 by construction. Do not extend it: no metric, no fit, no other row.

    ``bars`` come from ``load_klines`` with the vault sealed, so the last bar (2025-12-31 on the
    real data) has no successor. The cheater's decision there needs the next bar's candle; its
    leak reads nothing, a missing next bar counts as 0, and it is flat by default.
    """
    from olsprime.protocol import VAULT

    frame = pp.build_frame(bars, cfg.interval, cfg.h)
    last = frame.tail(1)
    cheater = Cheater(bars)
    target = float(cheater.predict(pp.test_rows(last, np.arange(1)))["target"][0])
    return {
        "last_sealed_open_time": bars["open_time"][-1].isoformat(),
        "last_sealed_known_at": bars["known_at"][-1].isoformat(),
        "bars_at_or_after_vault_start": int(bars.filter(pl.col("open_time") >= VAULT.start).height),
        "vault_start": VAULT.start.isoformat(),
        "label_of_last_bar_is_null": bool(last["label"].is_null().item()),
        "cheater_target_at_last_bar": target,
        "explanation": (
            "the seal leaves the cheater blind at the last sealed bar: its decision needs the next bar's open and "
            "close, that bar opens at the vault start and load_klines does not return it, so a missing next bar counts "
            "as 0 and the cheater is flat by default"
        ),
    }


# --- CLI ------------------------------------------------------------------------------


def development_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """The bars final before the end of the development period: the validation year stays out of H1."""
    from olsprime.protocol import DEVELOPMENT

    return bars.filter(pl.col("known_at") < DEVELOPMENT.end)


def _load(pair: str, cfg: H1Config) -> pl.DataFrame:
    from olsprime.protocol import load_klines

    return load_klines(pair, cfg.interval, caller="bench.invariance")


def _task(pair: str, model: str, cfg: H1Config, cuts: list[Cut]) -> tuple[str, str, list[dict[str, Any]]]:
    spec = {s.name: s for s in model_specs(cfg.interval, cfg.random_turnover_seeds)}[model]
    return pair, model, run_model(spec, development_bars(_load(pair, cfg)), cuts, cfg)


def _positive_task(pair: str, cfg: H1Config) -> tuple[str, dict[str, Any]]:
    return pair, positive_control(development_bars(_load(pair, cfg)), cfg)


def run_real(cfg: H1Config, *, workers: int = 1, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """H1 on the sealed real data of every pair; models run in parallel processes when ``workers > 1``.

    The provenance (commit, code digest, manifest digest) is taken first, before any data is read, as in Report n°1.
    """
    manifest = provenance.current_manifest() if manifest is None else provenance.strip_wall_clock(manifest)
    prov = provenance.record(manifest)
    specs = model_specs(cfg.interval, cfg.random_turnover_seeds)
    pairs: dict[str, dict[str, Any]] = {}
    cut_sets: dict[str, list[Cut]] = {}
    blind = {}
    for pair in cfg.pairs:
        sealed = _load(pair, cfg)
        bars = development_bars(sealed)
        frame = pp.build_frame(bars, cfg.interval, cfg.h)
        folds = pp.make_folds(frame, cfg.interval, cfg.h, Month(*cfg.start), Month(*cfg.end))
        cut_sets[pair] = choose_cuts(bars, folds, seed=cfg.cut_seed)
        pairs[pair] = {"cuts": [_cut_json(c) for c in cut_sets[pair]], "folds": len(folds), "models": {},
                       "cells": realised_cells(bars, cut_sets[pair], cfg)}  # fmt: skip
        blind[pair] = blind_cheater(sealed, cfg)  # fenced: the whole sealed series, to show its last bar
    jobs = [(pair, s.name) for pair in cfg.pairs for s in specs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            positive_futures = [pool.submit(_positive_task, p, cfg) for p in cfg.pairs]
            futures = [pool.submit(_task, p, m, cfg, cut_sets[p]) for p, m in jobs]
            results = [f.result() for f in futures]
            positives = [f.result() for f in positive_futures]
    else:
        positives = [_positive_task(p, cfg) for p in cfg.pairs]
        results = [_task(p, m, cfg, cut_sets[p]) for p, m in jobs]
    by_name = {s.name: s for s in specs}
    for pair, model, records in results:
        pairs[pair]["models"][model] = {"verdict": verdict(by_name[model], records), "records": records}
    for pair in cfg.pairs:  # specs order, whatever order the workers finished in
        pairs[pair]["models"] = {s.name: pairs[pair]["models"][s.name] for s in specs}
    positive = dict(positives)
    return {
        "config": _config_json(cfg),
        "pairs": pairs,
        "positive_control": positive,
        "seal": blind,
        "h1": h1_holds(pairs, specs, positive),
        "provenance": prov,
    }


def _config_json(cfg: H1Config) -> dict[str, Any]:
    out = asdict(cfg)
    out["models"] = [
        {"name": s.name, "expect": s.expect, "unpurged": s.unpurged} for s in model_specs(cfg.interval, cfg.random_turnover_seeds)
    ]
    return out


LIMIT = (
    "**The limit of this test.** Perturbation detects a leak only through what it perturbs. Every value of every "
    "later bar moves, and so does every ratio of two values of a bar (asserted at run time), but the time stamps and "
    "the grid cannot: a model that memorizes the future by `open_time` (a calendar learned from the future, a lookup "
    "keyed on the date) passes. Code review and the seal, not this test, stop it.\n\n"
    f"**By construction**, against bars t-{WINDOW}..t (the last {WINDOW + 1} bars, beaten by a factor {PUSH:g}): bar "
    "t + 1's prices, VWAP, taker VWAP, volume, quote volume, trades, trade size and taker shares (against the "
    "window's values strictly inside (0, 1)) are above them in the x3 run and below them in the /3 run; its body, "
    "lower wick and range are below them in the x3 run (a small bar) and above them in the /3 run (a big bar), and "
    "so is its upper wick, except at odd cuts (below). Its fill (body / range) is above bar t's in the x3 run and "
    "below it in the /3 run. **Colour and close location alternate with the cut's parity** (its position in the cut "
    "list): at even cuts the x3 bar is green and closes near its high, the /3 bar red, upper-wick-dominant, closing "
    "near its low; at odd cuts the mirror image, the x3 bar red closing near its low, the /3 bar green, "
    "lower-wick-dominant, closing near its high. Each comparison of one of these quantities with bar t (fill, close "
    "location) or with any of bars t-90..t (the rest) flips in exactly one run, the candle's colour too, and each of "
    "the four colour x size cells is realised at half the cuts: the `red-and-small` and `green-and-big` controls are "
    "predicted cut by cut from the cell each run realises (table above each pair's results).\n\n"
    "**Bands.** A threshold on bar t + 1 flips by construction only inside these bands (log units for body, wicks "
    f"and range): body [{UP_CAP['body']:.1%}, {DOWN_FLOOR['body']:.0%}), each wick [{UP_CAP['wick']:.2%}, "
    f"{DOWN_FLOOR['wick']:.0%}), range [{UP_CAP['body'] + 2 * UP_CAP['wick']:.1%}, {DOWN_FLOOR['body'] / DOWN_FILL:.0%}), "
    f"fill [{DOWN_FILL:g}, {UP_FILL:g}] (the x3 fill is aimed just above {UP_FILL:g}), close location "
    f"[{DOWN_CLV:g}, {UP_CLV:g}]. The /3 fill never goes below {FILL_FLOOR:g} (an exact floor), and no run's close "
    f"location target goes past {CLV_FLOOR:g} or {1 - CLV_FLOOR:g}, which bounds the wick the /3 bar asks for; a "
    "reference at or past either is a declared `?`, at both parities. For a "
    f"taker share s the band depends on the window: `s > c` flips for c from min s / {PUSH:g} up to "
    f"1 - (1 - max s) / {PUSH:g} only, min and max over the window's values strictly inside (0, 1), each share on "
    "its own window. Thresholds outside the bands are not guaranteed to flip. The big bar is far from any bar a "
    "venue would print: its range reaches about 20 log units on the bench's cuts (Audit A: 19.8 with the bench's "
    "seed, up to 20.7 over seeds 0 to 2), set by the fill floor (range <= body / 0.02 with a body above the "
    "window's largest).\n\n"
    "**Not by construction, with high probability only:** comparisons with bars older than t-90 (Audit A's script, "
    "on BTCUSDT under this perturbation: 8 of 8 cuts at t-120 and 7 of 8 at t-364: an older bar flips only "
    "where the pushed level happens to clear it); at odd cuts, the upper wick against bars t-90..t-1 (it is capped "
    "so that the green /3 bar can close near its high; against bar t it is guaranteed, or a declared `?`); shape "
    "functions other than fill, close location and colour x size, such as the wick ratio and hammer-like patterns, "
    "and conjunctions of colour or size with them (wick ratio x colour), whose remaining freedom is drawn per run; "
    "the close-to-close magnitude |ln(close[t+1]/close[t])|, at least ln 1.5 in both runs, so a comparison with a "
    "smaller move flips only where it was originally false; and any function of several later bars, such as "
    "`close[t+2] > close[t+1]` (Audit A's script: 7 of 8 cuts under round 2's perturbation, 8 of 8 under round 3's, "
    "7 of 8 under this one)."
    "\n\n"
    "**`?` cuts** are predicted from the original bars before perturbing, for declared reasons only (a bar without "
    "volume, a reference undefined or at a bound of its quantity, a reference fill or close location past the "
    "floors, an upper wick above the mirrored cap, a count of 1 that cannot go lower, a reference before the first "
    "bar). They are listed with their reason and left out of the verdict; any other failure to cross puts the "
    "control off its pattern, and a control that is `?` at every cut of a pair fails H1. A leak that shows only "
    "through a 0/1 target is caught only where the perturbation flips that target."
)


def render_md(result: dict[str, Any]) -> str:
    h1 = result["h1"]
    cfg = result["config"]
    directions = [_direction(s) for s in cfg["level_shifts"]]
    lines = [
        "# H1: future invariance",
        "",
        f"**Verdict: H1 {'holds' if h1['holds'] else 'FAILS'}.** Honest models: {len(h1['honest_models'])}, violations: "
        f"{h1['honest_violations']}. Controls: {len(h1['controls'])}, off their predicted pattern: "
        f"{', '.join(h1['controls_off_pattern']) or 'none'}; `?` at every cut: "
        f"{', '.join(h1.get('controls_unknown_at_every_cut', [])) or 'none'}. Cheater flagged at {h1['cheater_flagged_cuts']} of "
        f"{h1['cheater_cuts']} cuts. Positive control: the pipeline "
        + {True: "sees a signal when there is one", False: "IS DEAD (no signal seen)", None: "was not run"}[h1["pipeline_alive"]]
        + ".",
        "",
        f"Pipeline per pair (models are per pair): features, labels with h = {cfg['h']}, monthly walk-forward folds over "
        f"the test months {cfg['start'][0]}-{cfg['start'][1]:02d} -> {cfg['end'][0]}-{cfg['end'][1]:02d}, fit, predict, "
        "the long/flat rule at every venue. A violation is a row decided at or before the cut whose p_*, exp_return or "
        "target at any venue differs by a single bit after every value of every later bar is perturbed, in either "
        f"direction ({' and '.join(directions)}: the future multiplied by 3 or divided by 3, each with its own walk). "
        "An honest model is flagged by a single violation. A control is flagged (caught) at a cut only where one of "
        f"these outputs moves by more than {CAUGHT_ABOVE:g}, turns missing or stops being missing: a change at the level "
        "of rounding carries no leak and never counts as a catch.",
        "",
        "Patterns read cut by cut, in the order listed: X flagged, . not flagged. A model is ok when its pattern equals "
        "the predicted one exactly. Rows flagged: the violating rows of an honest model, the caught rows of a control, "
        "summed over the cuts. A control's smallest catch is, over its caught cuts, the smallest of each cut's largest "
        "change; rounding-only cuts are those where an output differs by some bits but by no more than "
        f"{CAUGHT_ABOVE:g}.",
        "",
        LIMIT,
        "",
    ]
    for pair, data in result["pairs"].items():
        cuts = data["cuts"]
        lines += [f"## {pair}", "", f"{data['folds']} folds. Cuts:", ""]
        lines += [
            f"{k + 1}. row {c['index']}, {c['open_time'][:10]} ({c['kind']}"
            f"{', opens a fold' if c['opens_fold'] else ''}{', last bar of its month' if not c['month_continues'] else ''})"
            for k, c in enumerate(cuts)
        ]
        if data.get("cells"):
            lines += ["", "Bar t + 1's (colour, size) per cut; size: body below or above "
                      f"{SMALL_BODY:.0%}; odd cuts take the mirror shape.", "",
                      "| cut | row | shape | original | " + " | ".join(directions) + " |",
                      "|---|---|---|---|" + "---|" * len(directions)]  # fmt: skip
            for k, cell in enumerate(data["cells"]):
                lines.append(
                    f"| {k + 1} | {cell['cut']} | {'mirror' if cell['mirror'] else 'plain'} | {' '.join(cell['original'])} | "
                    + " | ".join(" ".join(cell[d]) for d in directions) + " |"
                )
        head = " | ".join(f"flagged {d}" for d in directions)
        lines += [
            "",
            f"| model | must | {head} | flagged (either) | pattern | predicted | rows flagged | smallest catch | "
            "rounding-only cuts | ok |",
            "|---|---|" + "---|" * len(directions) + "---|---|---|---|---|---|---|",
        ]
        for name, m in data["models"].items():
            v = m["verdict"]
            per = " | ".join(f"{v['flagged_by_direction'][d]}/{v['cuts']}" for d in directions)
            honest = v["expect"] == PASS
            catch = "" if honest or v["smallest_catch"] is None else f"{v['smallest_catch']:.2g}"
            lines.append(
                f"| {name} | {v['expect']} | {per} | {v['flagged_cuts']}/{v['cuts']} | `{v['pattern']}` | "
                f"`{v['predicted_pattern']}` | {v['violations'] if honest else v['caught_rows']} | {catch} | "
                f"{len(v['rounding_only_cuts'])} | {'yes' if v['ok'] else 'NO'} |"
            )
        unknown = {name: m["verdict"].get("unknown_reasons", {}) for name, m in data["models"].items()}
        unknown = {k: v for k, v in unknown.items() if v}
        lines.append(
            "`?` cuts (a declared reason, read on the original bars before perturbing; out of the verdict): "
            + ("; ".join(f"{k} at row {row} ({why})" for k, v in unknown.items() for row, why in v.items()) if unknown else "none")
            + "."
        )
        lines.append("")
    pc = result.get("positive_control", {})
    if pc:
        lines += [
            "## Positive control: the pipeline can see a signal",
            "",
            "Ridge and climatology refitted per fold, on the real labels and on labels misaligned one bar early "
            "(`label[t] := label[t-1]`, an outcome that overlaps bar t). Brier skill score of ridge against climatology; "
            f"the control passes above {next(iter(pc.values()))['threshold']:g}.",
            "",
            "| pair | rows | BSS, real labels | BSS, labels one bar early | passes |",
            "|---|---|---|---|---|",
        ]
        for pair, v in pc.items():
            lines.append(
                f"| {pair} | {v['labels_one_bar_early']['rows']} | {v['real_labels']['bss']:+.4f} | "
                f"{v['labels_one_bar_early']['bss']:+.4f} | {'yes' if v['passes'] else 'NO'} |"
            )
        lines.append("")
    lines += ["## The seal leaves the cheater blind", ""]
    for pair, s in result["seal"].items():
        lines.append(
            f"- {pair}: last sealed bar opens {s['last_sealed_open_time'][:10]}; bars at or after the vault start: "
            f"{s['bars_at_or_after_vault_start']}; its label is null: {s['label_of_last_bar_is_null']}; "
            f"the cheater's target there: {s['cheater_target_at_last_bar']:g} ({s['explanation']})."
        )
    if "provenance" in result:
        lines += ["", *provenance.render_md(result["provenance"])]
    return "\n".join(lines) + "\n"


def _strip_masks(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_masks(v) for k, v in obj.items() if k != "mask"}
    if isinstance(obj, list):
        return [_strip_masks(v) for v in obj]
    return obj


def _default_out() -> Path:
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-h1"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.invariance", description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <runs>/<UTC date>-h1, suffixed if taken)")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes (results do not depend on it)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    cfg = H1Config()
    started = time.perf_counter()
    result = _strip_masks(run_real(cfg, workers=args.workers))
    out = args.out or _default_out()
    out.mkdir(parents=True, exist_ok=True)
    (out / "h1.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "h1.md").write_text(render_md(result), encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps({"finished_at": datetime.now(UTC).isoformat(), "seconds": round(time.perf_counter() - started, 1), "workers": args.workers}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    h1 = result["h1"]
    print(
        f"H1 {'holds' if h1['holds'] else 'FAILS'}: honest violations {h1['honest_violations']}, controls off pattern "
        f"{h1['controls_off_pattern'] or 'none'}, cheater flagged {h1['cheater_flagged_cuts']}/{h1['cheater_cuts']}, "
        f"pipeline alive {h1['pipeline_alive']}; written to {out}"
    )
    return 0 if h1["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
