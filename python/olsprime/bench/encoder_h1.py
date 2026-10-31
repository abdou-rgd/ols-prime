"""The panel H1 of the encoder arms (``docs/PROTOCOL.md``, iteration 4, "H1 for the encoder arms").

Usage::

    uv run --no-sync python -m olsprime.bench.encoder_h1 [--out DIR] [--device cuda|cpu]
        [--stage precheck|h1|floor|all] [--driver VERSION] [--synthetic]

``--stage all`` (the default) runs the pre-check, then the H1 stage only if the pre-check passed.
``--stage h1`` runs the H1 stage whatever the pre-check says (it is reported as it stands).
``--stage floor`` is the fallback's description, built but not planned: the H1 stage plus the
floor over three honest runs. ``--synthetic`` runs every stage on a synthetic sub-panel (the GPU
smoke) and never reads stored data. ``--driver`` records the GPU driver's version in ``run.json``
(no subprocess may ask ``nvidia-smi``; pass ``$(nvidia-smi --query-gpu=driver_version
--format=csv,noheader)``).

Set-up: iteration 3's panel H1, unchanged
-----------------------------------------
- The 10-member sub-panel, loaded by :func:`olsprime.bench.panel_invariance.load_subpanel`, so
  every read goes through ``load_klines`` and the seal. The panel is rebuilt by
  :func:`~olsprime.bench.panel_invariance.build_panel` and its windows by
  :func:`olsprime.train.fit.encoder_data`, from the original or the perturbed bars alike.
- The 8 cuts of :func:`~olsprime.bench.panel_invariance.choose_cuts` with iteration 3's
  configuration, asserted equal to :data:`RECORDED_CUTS` (``runs/2026-10-29-h1-panel``).
- The perturbation of :func:`~olsprime.bench.panel_invariance.perturbed_bars`: both directions
  (x3 and /3), with H1's seeds and mirror scheme (:func:`olsprime.bench.invariance.perturbation_at`).
  A member that ``perturb_after`` refuses is left out of that cut, unperturbed, and counted.
- **The two purge cuts** of the 2021 fit (F = 2021-01-01T00:00 UTC): F - 24 h
  (2020-12-31T00:00) and F - 30 days (2020-12-02T00:00). They are the 9th and 10th cuts of the
  same list: the k-th cut (0-based) of ``perturbation_at`` with k = 8 for F - 24 h and k = 9 for
  F - 30 days. So their seeds are H1's ``7 + 1000 j + k`` (15 and 1015, 16 and 1016, for x3 and /3),
  distinct from the 8 cuts' (k <= 7), and their mirror follows the parity of k (F - 24 h plain,
  F - 30 days mirrored), so the two purge cuts take the two colourings, as consecutive cuts do.

The runs (seed 1)
-----------------
- **Honest run** (unperturbed): each system trains the fit of each of its years on the sub-panel
  (the cut years 2020, 2021, 2022 and 2024; 2021 also for the purge cuts) and forecasts every
  complete row of that year. It is saved whole: the fit's inputs, every epoch's parameters, the
  slice losses, the epoch kept, every output, and the memory's stamps and raw window moments at
  every decision up to the year's last cut.
- **Perturbed run** at a cut: the sub-panel and its windows are rebuilt from the perturbed bars,
  the fit of the cut's year is retrained on them, and at the 8 cuts the rows of that year decided
  at or before the cut bar's ``known_at`` are forecast.

What is compared (:func:`compared`), at or before the cut, by H1's rule
--------------------------------------------------------------------------
- At the 8 cuts: the fit's inputs (training and slice rows, standardization, climatology, the
  weeks and the order of the episodes of every epoch, the step count), every parameter after every
  epoch, every epoch's slice loss and the epoch kept; per compared decision, the memory's window
  (its range in the stamp-sorted entries, and those entries' stamps, symbols and open times) and its
  raw moments ``(n, sum z, sum z z^T, sum y, sum y z^T)`` with the readout (count, means, beta and
  m); every output (the probabilities before and after the floor, and the climatology fallback).
- At F - 24 h: the 2021 fit's inputs, every parameter of every epoch, the slice losses and the
  epoch kept. No output.
- At F - 30 days: the training rows, the standardization, the weeks and order of the episodes,
  and every parameter after every epoch. Not the slice, the epoch kept or the climatology.

An honest system is flagged by a single differing bit. A planted leak is caught at a cut and in a
direction only where a compared value moves by more than
:data:`olsprime.bench.invariance.CAUGHT_ABOVE`, turns missing or stops being missing
(:func:`olsprime.bench.panel_invariance.compare`). Each direction must match the prediction.

The systems
-----------
- ``enc-ols``, ``enc-softmax``, ``enc-linear``: the honest arms, at the 8 cuts and the 2 purge cuts.
- ``cheat-stamp`` (``enc-ols``): every memory entry stamped with its row's decision time
  (:class:`StampedAtDecision`), in training, in the slice and at test time.
- ``cheat-value`` (``enc-ols``): ``EncoderConfig(value_leak=True)``, the row's own one-hot label in
  the value of the last position of every head.
- ``cheat-stamp-day`` (``enc-ols``): ``cheat-stamp`` whose memory holds only the last 24 hours by
  decision time, (d - 24 h, d] (``MemoryConfig(window_ms=24 h)``, :class:`StampedAtDecisionDay`), in
  training, in the slice and at test time: every entry it reads has a label not yet known at d.
- ``cheat-purge`` (``enc-ols``): the 2021 fit trains on the labels known by F (``train_gap_ms = 0``).
  At the purge cuts only.
- ``cheat-slice`` (``enc-ols``): the 2021 fit trains on the labels known by F - 24 h, its own slice
  included (``train_gap_ms = 24 h``). At the purge cuts only.
- ``pos-window`` (``enc-ols``): the positive control, trained, stored and scored on
  :func:`olsprime.train.fit.past_window_labels`; its entries keep ``label_known_at``. Unperturbed:
  it runs the honest fits of the cut years and is scored only.

Predicted patterns (:func:`labels_read`, :func:`predicted`)
-------------------------------------------------------------
One rule for every system, computed per cut from the original panel and the system's own hooks:
a system is predicted caught at a cut when some compared item reads a label whose exit bar opens
after the cut hour (``label_known_at > cut``; a label is known at its exit bar's open). The labels
each compared item reads:

- the parameters (and every forecast, through the kept model): the training rows (training mask);
- the slice losses and the epoch kept: the slice rows, and the memory entries their readouts reach
  (stamps in ``(first decision - 30 days, last decision]``, stamped by the system's hook);
- the climatology: the labelled rows known by F - 24 h;
- the forecasts and the memory: the entries their readouts reach, and with the value leak the
  forecast rows' own labels.

What the rule gives, and why:

- honest arms: never (asserted: an honest arm predicted caught is a rule error). Training reads
  labels known by F - 30 days, the slice and climatology by F - 24 h, a forecast entries known by
  its decision, all at or before the cut hour.
- ``cheat-stamp``: caught at each of the 8 cuts with a compared decision: an entry decided in the
  24 hours before the cut joins at its decision, its label known after the cut. Invariant at both
  purge cuts: training reads labels known by F - 30 days, and a slice row's label is known by
  F - 24 h, so an entry decided before it has its label known by F - 24 h too.
- ``cheat-stamp-day``: the same as ``cheat-stamp``, by the same argument; the rule reads its window
  (``cfg.memory.window_ms``, 24 h) from the system itself.
- ``cheat-value``: caught at each of the 8 cuts: a forecast row decided in the last 24 hours reads
  its own label, known after the cut. Invariant at both purge cuts: training and slice labels are
  known by F - 30 days and F - 24 h.
- ``cheat-purge``: caught at both purge cuts: it trains on labels known by F, after each.
- ``cheat-slice``: caught at F - 30 days, invariant at F - 24 h: it trains on labels known by
  F - 24 h, after the first and at the second.

The computed prediction is the one tested, direction by direction.

Skill (:func:`skill`)
---------------------
On the complete rows of the sub-panel in the cut years, from the honest runs: per system the
log-loss, the Brier score, the ECE and the Brier skill score against ``climatology-panel`` (the
monthly climatology of iteration 3, :func:`monthly_climatology`), with Report n°2's paired
stationary bootstrap (:func:`olsprime.bench.report2.bss_interval`) over the hourly series. The
registered check, control by control (:data:`SKILL_CONTROLS`: ``cheat-stamp``, ``cheat-value`` and
``cheat-stamp-day``): at least 0.1, with an interval above every honest arm's. ``pos-window`` is
scored on the same rows against the climatology of its own label and must reach 0.5.

Three verdicts, reported apart
------------------------------
- **H1**, as ``docs/HYPOTHESES.md`` defines it: every honest arm bit for bit at every cut, and every
  control caught on its predicted pattern, in each direction where predicted and in none where not.
- **The skill check**, control by control.
- **``pos-window``'s check.**

Each prints pass or fail; H1's line is never printed without the skill check's beside it, and a
failed skill check does not turn H1's verdict into a failure: it is its own finding. The report's
order: the pre-check, ``cheat-stamp``'s catch with its skill, ``pos-window``, then the rest. Exit
codes: 2 the pre-check fails, 1 H1 fails, 3 H1 holds but the skill check or ``pos-window``'s fails.

Pre-check
---------
Two honest runs of each arm with identical inputs, compared on everything above (every fit year,
every output and memory value up to each year's last cut); and the scope test: the first cut's
year forecast only up to the first cut, with the honest fit, must give the full run's outputs and
memory values on those rows. Reported per arm and per item, bit for bit.

**Scope at every cut (a description, audit 4e).** In the pre-check stage, before H1 starts, the same
scope test runs at each of the 8 regular cuts for the three honest arms and ``cheat-stamp``
(:meth:`Runner.scope_every_cut`): forecasts only, from the honest run of the cut's year
(``cheat-stamp``'s run 1 is the one H1 uses). It is reported apart, per system and cut, and enters
neither the pre-check's registered verdict nor the exit code.

The fallback, built but not planned
-----------------------------------
``--device cpu`` runs the same H1 on CPU torch, one thread. ``--stage floor`` adds runs 2 and 3 of
every perturbed system and gives, per item, the largest pairwise difference over the three honest
runs (the floor), and each control's smallest catch as a multiple of it (the largest ratio over the
caught items of a caught cut and direction, minimized over them); under 100 times the floor it is
weakly separated. The budget's options (:attr:`EncoderH1Config.controls_at`,
:attr:`EncoderH1Config.honest_retrain_at`) are built; by default every system runs at every cut.

Resumable
---------
Each finished part, (system, arm, fit year, cut, direction), is saved under ``<out>/parts/`` with
its SHA-256 beside it. A restart verifies each part and skips it (a part whose digest fails is
recomputed); it refuses to start if the configuration, the code digest, the data manifest or the
build changed (``parts/IDENTITY.json``; the build is :func:`olsprime.train.determinism.build`, with
torch, CUDA, cuDNN, the GPU and the settings in force, and the driver passed by ``--driver``), and
names each field that changed.

Outputs: ``encoder-h1.json`` (sorted keys, no wall clock), ``encoder-h1.md``, ``config.json`` (the
configuration and provenance) and ``run.json`` (wall clock, GPU, driver, CUDA and torch only), to
``runs/<UTC date>-encoder-h1/`` or ``--out``. A restart the same day resumes the same folder.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from olsprime.bench import panel_invariance as pi
from olsprime.bench import provenance
from olsprime.bench.invariance import CAUGHT_ABOVE
from olsprime.bench.report2 import Report2Config, Scored, bss_interval, hourly
from olsprime.eval import stats
from olsprime.models.encoder import ARMS, EncoderConfig, Standardizer
from olsprime.models.matrix import N_CLASSES, PanelArrays, prob_map
from olsprime.models.memory import MemoryConfig, window_stats
from olsprime.paths import runs_dir
from olsprime.train import checkpoint
from olsprime.train.determinism import build as build_info
from olsprime.train.determinism import deterministic
from olsprime.train.fit import (
    DAY_MS,
    HOUR_MS,
    EncoderData,
    EncoderSystem,
    FitConfig,
    FitResult,
    encoder_data,
    past_window_labels,
    rows_of_year,
    year_start_ms,
)

CONTROLS: tuple[str, ...] = ("cheat-stamp", "cheat-value", "cheat-stamp-day", "cheat-purge", "cheat-slice")
# The controls whose skill is checked: those that forecast (cheat-purge and cheat-slice compare no output).
SKILL_CONTROLS: tuple[str, ...] = ("cheat-stamp", "cheat-value", "cheat-stamp-day")
POSITIVE = "pos-window"
# The 8 cuts of iteration 3's panel H1 (runs/2026-10-29-h1-panel/h1-panel.json), UTC hours.
RECORDED_CUTS: tuple[str, ...] = (
    "2020-03-01T00:00", "2020-05-29T14:00", "2021-07-31T23:00", "2022-07-05T17:00",
    "2024-03-04T13:00", "2024-10-06T15:00", "2024-11-13T15:00", "2024-12-03T13:00",
)  # fmt: skip
PURGE_24H, PURGE_30D = "F-24h", "F-30d"
PURGE_OFFSETS: tuple[tuple[str, int], ...] = ((PURGE_24H, 24 * HOUR_MS), (PURGE_30D, 30 * DAY_MS))
STAGES: tuple[str, ...] = ("precheck", "h1", "floor", "all")
FILE_DIRECTION = {"x3": "x3", "/3": "div3"}
CATEGORIES: tuple[str, ...] = ("inputs", "params", "slice", "forecast", "memory", "output")
# The F - 30 days cut compares these fit inputs (and every parameter), not the slice nor the climatology.
PURGE_30D_INPUTS = frozenset({"inputs:train_rows", "inputs:standardization", "inputs:weeks", "inputs:orders"})


# --- Systems ------------------------------------------------------------------------------------


class BenchSystem(EncoderSystem):
    """An arm exactly as :class:`olsprime.train.fit.EncoderSystem` builds it, which also keeps the raw
    window moments ``(n, sum z, sum z z^T, sum y, sum y z^T)`` of the forecasts it is asked to record.

    It reads them through the public read-only hook :meth:`~olsprime.train.fit.EncoderSystem.on_readout`,
    from the same prefix sums and window bounds the readout uses
    (:func:`olsprime.models.memory.window_stats`); nothing it computes reaches an output.
    """

    moments: list[tuple[np.ndarray, ...]] | None = None

    def on_readout(self, prefix: Any, lo: torch.Tensor, hi: torch.Tensor) -> None:
        if self.moments is not None:
            self.moments.append(tuple(t.numpy().copy() for t in window_stats(prefix, lo, hi)))


# !!! A planted leak, on purpose. NOT A MODEL. Never report it as a result. !!!
class StampedAtDecision(BenchSystem):
    """!!! LEAKS (``cheat-stamp``): every memory entry joins at its row's decision time, before its label is known. !!!"""

    name = "cheat-stamp"

    def entry_stamps(self, rows: np.ndarray) -> np.ndarray:
        return self.data.arrays.decision_ms[rows]


# !!! A planted leak, on purpose. NOT A MODEL. Never report it as a result. !!!
class StampedAtDecisionDay(StampedAtDecision):
    """!!! LEAKS (``cheat-stamp-day``): ``cheat-stamp`` whose memory holds the last 24 hours, (d - 24 h, d]:
    every entry it reads has a label not yet known at d. The window is :attr:`Spec.memory_window_ms`. !!!"""

    name = "cheat-stamp-day"


class PosWindow(BenchSystem):
    """The positive control ``pos-window``: the label of the 24 hours that end at bar t; entries keep ``label_known_at``."""

    name = POSITIVE

    def label_array(self) -> np.ndarray:
        return past_window_labels(self.data)


@dataclass(frozen=True)
class Spec:
    name: str
    arm: str
    cls: type[BenchSystem]
    honest: bool
    value_leak: bool = False
    train_gap_ms: int = 30 * DAY_MS
    at_regular_cuts: bool = True  # cheat-purge and cheat-slice run at the purge cuts only
    perturbed: bool = True  # pos-window is scored, never perturbed
    forecasts: bool = True  # cheat-purge and cheat-slice compare no output
    memory_window_ms: int | None = None  # cheat-stamp-day: 24 h; None: the registered window (MemoryConfig's default)

    @property
    def leak(self) -> str:
        return {
            "cheat-stamp": "memory entries stamped at their decision time, in training, the slice and the forecasts",
            "cheat-stamp-day": "cheat-stamp with a memory of the last 24 hours, (d - 24 h, d]: every entry read has a label not yet known at d",
            "cheat-value": "the row's own one-hot label added to the value of the last position in every head",
            "cheat-purge": "the purge cuts' fit (2021) trains on the labels known by F (no purge)",
            "cheat-slice": "the purge cuts' fit (2021) trains on the labels known by F - 24 h (its own slice included)",
            POSITIVE: "label: the class of the 24 hours that end at bar t, inside the window (positive control)",
        }.get(self.name, "honest")


SPECS: dict[str, Spec] = {
    **{arm: Spec(arm, arm, BenchSystem, True) for arm in ARMS},
    "cheat-stamp": Spec("cheat-stamp", "enc-ols", StampedAtDecision, False),
    "cheat-stamp-day": Spec("cheat-stamp-day", "enc-ols", StampedAtDecisionDay, False, memory_window_ms=24 * HOUR_MS),
    "cheat-value": Spec("cheat-value", "enc-ols", BenchSystem, False, value_leak=True),
    "cheat-purge": Spec("cheat-purge", "enc-ols", BenchSystem, False, train_gap_ms=0, at_regular_cuts=False, forecasts=False),
    "cheat-slice": Spec("cheat-slice", "enc-ols", BenchSystem, False, train_gap_ms=24 * HOUR_MS, at_regular_cuts=False, forecasts=False),
    POSITIVE: Spec(POSITIVE, "enc-ols", PosWindow, False, perturbed=False),
}


# --- Configuration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderH1Config:
    """Everything the encoder H1 fixes. Saved with every run; a resume needs the same."""

    panel: pi.PanelH1Config = field(default_factory=pi.PanelH1Config)
    arms: tuple[str, ...] = ARMS
    controls: tuple[str, ...] = CONTROLS
    positive: bool = True
    seed: int = 1
    device: str = "cuda"
    cpu_threads: int = 1  # under --device cpu
    cuts: tuple[str, ...] = RECORDED_CUTS
    choose_cuts: bool = True  # draw iteration 3's cuts and assert they equal ``cuts``; False: take ``cuts`` as given
    purge_fit_year: int = 2021
    controls_at: tuple[int, ...] | None = None  # the budget: the regular cuts (1-based) the planted controls run at; None: all
    honest_retrain_at: tuple[int, ...] | None = None  # the CPU budget: regular cuts where the honest arms retrain; None: all
    encoder: tuple[tuple[str, Any], ...] = ()  # EncoderConfig overrides: tests and smoke runs only
    fit: tuple[tuple[str, Any], ...] = ()  # FitConfig overrides: tests and smoke runs only
    n_boot: int = 2000
    min_control_bss: float = 0.1
    min_positive_bss: float = 0.5
    floor_runs: int = 3
    weak_multiple: float = 100.0
    synthetic: str | None = None  # a description of the synthetic sub-panel; None: the real one

    def __post_init__(self) -> None:
        unknown = (set(self.arms) - set(ARMS)) | (set(self.controls) - set(CONTROLS))
        if unknown:
            raise ValueError(f"unknown systems {sorted(unknown)}")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("the device is cuda or cpu")
        if self.controls and "enc-ols" not in self.arms:
            raise ValueError("the planted controls are enc-ols: they need the enc-ols arm beside them")

    @property
    def registered(self) -> bool:
        """The pre-registered H1, on the GPU or under the CPU fallback."""
        return dataclasses.replace(self, device="cuda") == EncoderH1Config()

    def specs(self) -> list[Spec]:
        return [SPECS[n] for n in (*self.arms, *self.controls)] + ([SPECS[POSITIVE]] if self.positive else [])


def config_json(cfg: EncoderH1Config) -> dict[str, Any]:
    out = dataclasses.asdict(cfg)
    out["registered"] = cfg.registered
    out["systems"] = [{"name": s.name, "arm": s.arm, "honest": s.honest, "leak": s.leak, "train_gap_ms": s.train_gap_ms,
                       "at_regular_cuts": s.at_regular_cuts, "perturbed": s.perturbed,
                       "memory_window_ms": s.memory_window_ms} for s in cfg.specs()]  # fmt: skip
    out["caught_above"] = CAUGHT_ABOVE
    out["purge_offsets_ms"] = dict(PURGE_OFFSETS)
    return out


def fit_config(cfg: EncoderH1Config, spec: Spec, year: int) -> FitConfig:
    enc = EncoderConfig(arm=spec.arm, value_leak=spec.value_leak, **dict(cfg.encoder))
    extra = dict(cfg.fit)
    if spec.memory_window_ms is not None:  # only cheat-stamp-day; every other system keeps FitConfig's memory
        extra["memory"] = dataclasses.replace(extra.get("memory", MemoryConfig()), window_ms=spec.memory_window_ms)
    return FitConfig(fit_ms=year_start_ms(year), seed=cfg.seed, encoder=enc, train_gap_ms=spec.train_gap_ms, **extra)


# --- Cuts ---------------------------------------------------------------------------------------


def hour_text(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M")


def hour_ms(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M").replace(tzinfo=UTC).timestamp() * 1000)


@dataclass(frozen=True)
class H1Cut:
    k: int  # its index in the perturbation schedule (perturbation_at): 0..7 the cuts, 8 F - 24 h, 9 F - 30 days
    label: str  # "1".."8", "F-24h", "F-30d"
    hour_ms: int  # the open of the cut hour
    kind: str
    year: int  # the fit compared

    @property
    def until_ms(self) -> int:
        return self.hour_ms + HOUR_MS - 1

    @property
    def regular(self) -> bool:
        return self.label not in (PURGE_24H, PURGE_30D)

    def panel_cut(self) -> pi.PanelCut:
        return pi.PanelCut(self.hour_ms, self.kind)


def h1_cuts(cfg: EncoderH1Config, arrays: PanelArrays) -> list[H1Cut]:
    """The 8 cuts (iteration 3's, checked) then the two purge cuts, k = 8 and 9 of the perturbation schedule."""
    if cfg.choose_cuts:
        drawn = pi.choose_cuts(arrays, cfg.panel)
        hours = tuple(hour_text(c.hour_ms) for c in drawn)
        if hours != cfg.cuts:
            raise ValueError(f"the cuts drawn {hours} are not iteration 3's {cfg.cuts}")
        regular = [(c.hour_ms, c.kind) for c in drawn]
    else:
        regular = [(hour_ms(h), "given-hour") for h in cfg.cuts]
    out = [H1Cut(k, str(k + 1), h, kind, datetime.fromtimestamp(h / 1000, tz=UTC).year) for k, (h, kind) in enumerate(regular)]
    f = year_start_ms(cfg.purge_fit_year)
    for i, (label, offset) in enumerate(PURGE_OFFSETS):
        out.append(H1Cut(len(regular) + i, label, f - offset, "purge-" + label[2:], cfg.purge_fit_year))
    return out


def runs_at(cfg: EncoderH1Config, spec: Spec, cut: H1Cut) -> bool:
    if not spec.perturbed:
        return False
    if not cut.regular:
        return True
    if not spec.at_regular_cuts:
        return False
    return spec.honest or cfg.controls_at is None or int(cut.label) in cfg.controls_at


def retrains_at(cfg: EncoderH1Config, spec: Spec, cut: H1Cut) -> bool:
    return not (spec.honest and cut.regular and cfg.honest_retrain_at is not None and int(cut.label) not in cfg.honest_retrain_at)


def scope_of(cut: H1Cut, retrain: bool) -> str:
    if cut.label == PURGE_24H:
        return "purge-24h"
    if cut.label == PURGE_30D:
        return "purge-30d"
    return "cut" if retrain else "cut-forecast"


# --- Data ---------------------------------------------------------------------------------------


@dataclass(eq=False)
class Subpanel:
    bars: dict[str, pl.DataFrame]
    months: dict[str, list[datetime]]
    data: EncoderData


def build_data(bars: dict[str, pl.DataFrame], months: dict[str, list[datetime]]) -> EncoderData:
    """The panel (iteration 3's ``build_panel``) and its windows, from these bars."""
    return encoder_data(pi.build_panel(bars, months), bars)


def subpanel(bars: dict[str, pl.DataFrame], months: dict[str, list[datetime]]) -> Subpanel:
    return Subpanel(bars, months, build_data(bars, months))


def synthetic_subpanel(members: int = 2, *, start: datetime = datetime(2018, 11, 1, tzinfo=UTC), hours: int | Sequence[int] = 24 * 100,
                       seed: int = 0) -> tuple[dict[str, pl.DataFrame], dict[str, list[datetime]]]:  # fmt: skip
    """Seeded synthetic hourly bars of ``members`` pairs (every month a member month). Nothing here reads data."""
    from olsprime.bench.synthetic import synthetic_bars

    lengths = [hours] * members if isinstance(hours, int) else list(hours)
    bars, months = {}, {}
    for i, n in enumerate(lengths):
        symbol = f"S{i:02d}USDT"
        frame = synthetic_bars(symbol, start=start, days=n, interval="1h", seed=seed * 100 + i, drift=0.0, vol=0.01)
        bars[symbol] = frame
        months[symbol] = sorted(set(frame["open_time"].dt.truncate("1mo").to_list()))
    return bars, months


def make_system(spec: Spec, data: EncoderData, cfg: EncoderH1Config, year: int) -> BenchSystem:
    return spec.cls(data, fit_config(cfg, spec, year), device=cfg.device)


# --- What a run saves, and what is compared -----------------------------------------------------


def _ident(time_ms: np.ndarray, symbol: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.stack([time_ms[rows], symbol[rows]], axis=1).astype(np.int64)


def fit_arrays(system: EncoderSystem, res: FitResult) -> dict[str, np.ndarray]:
    """The fit's inputs, its slice and every epoch's parameters (native dtypes)."""
    a = system.data.arrays
    std = res.standardizer
    out = {
        "inputs:train_rows": _ident(a.open_ms, a.symbol, res.train_rows),
        "inputs:slice_rows": _ident(a.open_ms, a.symbol, res.slice_rows),
        "inputs:standardization": np.concatenate([std.mean, std.std, [float(std.rows)]]),
        "inputs:climatology": np.concatenate([res.climatology, [float(res.climatology_rows)]]),
        "inputs:weeks": np.asarray(res.weeks, dtype=np.int64),
        "inputs:orders": np.stack(res.orders).astype(np.int64),
        "inputs:counts": np.array([res.steps, res.n_parameters, res.complete_without_window], dtype=np.int64),
        "slice:losses": np.asarray(res.slice_losses, dtype=np.float64),
        "slice:kept_epoch": np.array([res.kept_epoch], dtype=np.int64),
    }
    for e, state in enumerate(res.epoch_states, start=1):
        for name, value in state.items():
            out[f"params:epoch{e}:{name}"] = np.asarray(value)
    return out


def forecast_arrays(system: BenchSystem, res: FitResult, rows: np.ndarray, mem_until: int | None) -> dict[str, np.ndarray]:
    """The forecasts of ``rows`` (panel order), and the memory at every decision up to ``mem_until`` (None: none)."""
    a = system.data.arrays
    system.moments = [] if mem_until is not None else None
    try:
        out = system.forecast(res, rows)
        moments = system.moments
    finally:
        system.moments = None
    arr = {
        "aux:positions": np.asarray(rows, dtype=np.int64),
        "forecast:rows": _ident(a.decision_ms, a.symbol, rows),
        "output:raw": out["raw"],
        "output:probs": out["probs"],
        "output:fallback": np.asarray(out["fallback"], dtype=bool),
    }
    if mem_until is None:
        return arr
    if moments is None or len(moments) != 1:
        raise RuntimeError("the forecast made no single readout to record")
    pred = rows[~out["fallback"]]
    m = int((a.decision_ms[pred] <= mem_until).sum())  # a prefix: panel order is decision order
    n, s_z, s_zz, s_y, s_yz = moments[0]
    lo, hi = out["window_lo"][:m].astype(np.int64), out["window_hi"][:m].astype(np.int64)
    start, end = (int(lo.min()), int(hi.max())) if m else (0, 0)
    pool_rows, pool_stamps = system.sorted_pool(np.flatnonzero(system.data.usable))
    arr |= {
        "memory:rows": _ident(a.decision_ms, a.symbol, pred[:m]),
        "memory:window": np.stack([lo, hi], axis=1),
        "memory:entry_range": np.array([start, end], dtype=np.int64),
        "memory:entries": np.stack([pool_stamps[start:end], a.symbol[pool_rows[start:end]], a.open_ms[pool_rows[start:end]]], axis=1),
        "memory:n": n[:m], "memory:s_z": s_z[:m], "memory:s_zz": s_zz[:m], "memory:s_y": s_y[:m], "memory:s_yz": s_yz[:m],
        "memory:count": out["count"][:m], "memory:mean_z": out["mean_z"][:m], "memory:mean_y": out["mean_y"][:m],
        "memory:beta": out["beta"][:m], "memory:m": out["m"][:m],
    }  # fmt: skip
    return arr


def _forecast_items(arr: dict[str, np.ndarray], until: int | None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    rows = arr["forecast:rows"]
    k = rows.shape[0] if until is None else int((rows[:, 0] <= until).sum())
    for key in ("forecast:rows", "output:raw", "output:probs", "output:fallback"):
        out[key] = arr[key][:k]
    if "memory:rows" not in arr:
        return out
    mrows = arr["memory:rows"]
    m = mrows.shape[0] if until is None else int((mrows[:, 0] <= until).sum())
    start = int(arr["memory:entry_range"][0])
    end = int(arr["memory:window"][:m, 1].max()) if m else start
    for key, value in arr.items():
        if key.startswith("memory:") and key not in ("memory:entries", "memory:entry_range"):
            out[key] = value[:m]
    out["memory:entry_range"] = np.array([start, end])
    out["memory:entries"] = arr["memory:entries"][: end - start]
    return out


def compared(arr: dict[str, np.ndarray], scope: str, until: int | None = None) -> dict[str, np.ndarray]:
    """The compared items of a run, as float64 arrays, for a scope.

    ``full`` (the pre-check): every item. ``cut``: the fit and the forecasts and memory up to
    ``until``. ``cut-forecast``: the forecasts and memory only (no retrain, the budget's option).
    ``purge-24h``: the fit only. ``purge-30d``: the training rows, standardization, weeks, orders
    and every parameter.
    """
    if scope not in ("full", "cut", "cut-forecast", "purge-24h", "purge-30d"):
        raise ValueError(f"unknown scope {scope!r}")
    out: dict[str, np.ndarray] = {}
    for key, value in arr.items():
        cat = key.split(":", 1)[0]
        if cat not in ("inputs", "slice", "params"):
            continue
        if scope in ("full", "cut", "purge-24h") or (scope == "purge-30d" and (cat == "params" or key in PURGE_30D_INPUTS)):
            out[key] = value
    if scope in ("full", "cut", "cut-forecast"):
        if "forecast:rows" not in arr:
            raise ValueError("this run made no forecast to compare")
        out |= _forecast_items(arr, None if scope == "full" else until)
    return {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in out.items()}


def result_from(arr: dict[str, np.ndarray], cfg: FitConfig) -> FitResult:
    """The fit a run saved, enough to forecast with: its kept parameters, standardization and climatology."""
    kept = int(arr["slice:kept_epoch"][0])
    pre = f"params:epoch{kept}:"
    state = {k[len(pre) :]: v for k, v in arr.items() if k.startswith(pre)}
    s = arr["inputs:standardization"]
    q = (s.size - 1) // 2
    c = arr["inputs:climatology"]
    std = Standardizer(s[:q].copy(), s[q : 2 * q].copy(), int(s[-1]))
    empty = np.zeros(0, dtype=np.int64)
    return FitResult(cfg, state, std, c[:N_CLASSES].copy(), int(c[N_CLASSES]), empty, empty, 0, arr["inputs:weeks"], [],
                     arr["slice:losses"].tolist(), kept, 0, 0)  # fmt: skip


# --- Comparison ---------------------------------------------------------------------------------


def item_moves(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Per differing item: values differing in a bit, values caught (H1's rule) and the largest move (inf: shape or presence)."""
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if x is None or y is None or x.shape != y.shape:
            size = max(0 if x is None else x.size, 0 if y is None else y.size, 1)
            out[key] = {"differing": size, "caught": size, "largest": math.inf}
            continue
        bits = np.ascontiguousarray(x).view(np.uint64) != np.ascontiguousarray(y).view(np.uint64)
        if not bits.any():
            continue
        both = ~np.isnan(x) & ~np.isnan(y)
        move = np.abs(x[both] - y[both])
        caught = int((move > CAUGHT_ABOVE).sum() + (np.isnan(x) != np.isnan(y)).sum())
        out[key] = {"differing": int(bits.sum()), "caught": caught, "largest": float(move.max(initial=0.0))}
    return out


def category(key: str) -> str:
    return key.split(":", 1)[0]


def compare_record(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    """H1's comparison of two item sets (iteration 3's ``compare``), with every differing item and the counts per category."""
    summary = pi.compare(a, b)
    items = item_moves(a, b)
    if sum(v["differing"] for v in items.values()) != summary["violations"] or sum(v["caught"] for v in items.values()) != summary["caught"]:
        raise AssertionError("the per-item comparison disagrees with H1's compare")
    by_cat: dict[str, dict[str, int]] = {}
    for key in sorted(set(a) | set(b)):
        c = by_cat.setdefault(category(key), {"items": 0, "values": 0, "differing": 0, "caught": 0})
        c["items"] += 1
        c["values"] += int((a.get(key) if key in a else b[key]).size)
        if key in items:
            c["differing"] += items[key]["differing"]
            c["caught"] += items[key]["caught"]
    return {"violations": summary["violations"], "caught": summary["caught"], "largest_change": summary["largest_change"],
            "compared_values": int(sum(v.size for v in a.values())), "items": items, "by_category": by_cat}  # fmt: skip


# --- Predictions --------------------------------------------------------------------------------


def labels_read(system: EncoderSystem, cut: H1Cut, scope: str) -> np.ndarray:
    """The rows whose label some compared item of ``system`` reads at ``cut`` (see the module docstring)."""
    d, a, cfg = system.data, system.data.arrays, system.cfg
    w = cfg.memory.window_ms
    pool = np.flatnonzero(d.usable)
    stamps = system.entry_stamps(pool).astype(np.int64)

    def memory(q_rows: np.ndarray) -> np.ndarray:
        if not q_rows.size:
            return q_rows
        as_of = system.readout_as_of(q_rows).astype(np.int64)
        return pool[(stamps > as_of.min() - w) & (stamps <= as_of.max())]

    parts = []
    if scope != "cut-forecast":
        parts.append(np.flatnonzero(system.training_mask()))
    if scope in ("cut", "purge-24h"):
        s = np.flatnonzero(system.slice_mask())
        parts += [s, memory(s), np.flatnonzero(a.labelled & (system.y >= 0) & (a.lka_ms <= cfg.fit_ms - cfg.purge_ms))]
    if scope in ("cut", "cut-forecast"):
        r = rows_of_year(a, cut.year)
        r = r[(a.decision_ms[r] <= cut.until_ms) & d.usable[r]]
        parts.append(memory(r))
        if cfg.encoder.value_leak:
            parts.append(r)
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def predicted(system: EncoderSystem, cut: H1Cut, scope: str) -> bool:
    """Caught when some compared item reads a label whose exit bar opens after the cut hour."""
    rows = labels_read(system, cut, scope)
    return bool(np.any(system.data.arrays.lka_ms[rows] > cut.hour_ms))


# --- Skill --------------------------------------------------------------------------------------


def monthly_climatology(arrays: PanelArrays, y: np.ndarray, months_ms: np.ndarray, floor: float = 1e-3, horizon_ms: int = 24 * HOUR_MS) -> np.ndarray:
    """``climatology-panel`` of iteration 3 for a label ``y``: the class frequencies of the labelled rows with
    ``label_known_at <= month - h``, through the probability floor, per month given (``MatrixSystem.climatology_at``)."""
    lab = np.flatnonzero(arrays.labelled & (y >= 0))
    lab = lab[np.argsort(arrays.lka_ms[lab], kind="stable")]
    known = arrays.lka_ms[lab]
    onehot = np.zeros((lab.size + 1, N_CLASSES), dtype=np.int64)
    onehot[np.arange(1, lab.size + 1), y[lab]] = 1
    counts = np.cumsum(onehot, axis=0)
    out = np.empty((months_ms.size, N_CLASSES))
    for i, m in enumerate(months_ms):
        j = int(np.searchsorted(known, int(m) - horizon_ms, side="right"))
        out[i] = np.full(N_CLASSES, 1.0 / N_CLASSES) if j == 0 else prob_map(counts[j].astype(np.float64) / float(j), floor)
    return out


def _score(probs: np.ndarray, y: np.ndarray, clim: np.ndarray, open_ms: np.ndarray, rcfg: Report2Config, name: str) -> dict[str, Any]:
    hours, inv = np.unique(open_ms, return_inverse=True)
    sc = Scored(np.arange(y.size), y, inv, hours, np.bincount(inv, minlength=hours.size))
    arm, ref = stats.per_row_losses(probs, y), stats.per_row_losses(clim, y)
    b_arm, b_clim = hourly(arm.brier, sc), hourly(ref.brier, sc)
    return {"rows": int(y.size), "hours": int(hours.size), "log_loss": float(hourly(arm.log_loss, sc).mean()),
            "brier": float(b_arm.mean()), "climatology_log_loss": float(hourly(ref.log_loss, sc).mean()),
            "climatology_brier": float(b_clim.mean()), "ece": float(stats.ece(probs, y)),
            "bss": bss_interval(b_arm, b_clim, rcfg, "encoder-h1", "bss", name)}  # fmt: skip


def skill(data: EncoderData, outputs: dict[str, dict[int, dict[str, np.ndarray]]], years: Sequence[int], cfg: EncoderH1Config) -> dict[str, Any]:
    """The Brier skill of each system on the complete rows of the cut years, from its honest runs (module docstring)."""
    a = data.arrays
    rows = np.concatenate([rows_of_year(a, y) for y in years])
    rcfg = Report2Config(n_boot=cfg.n_boot)
    clim = monthly_climatology(a, a.label, a.month_ms[rows])
    out: dict[str, Any] = {"years": list(years), "reference": "climatology-panel"}
    systems: dict[str, Any] = {}
    for name, per_year in outputs.items():
        if name == POSITIVE:
            continue
        pos = np.concatenate([per_year[y]["aux:positions"] for y in years])
        if not np.array_equal(pos, rows):
            raise ValueError(f"{name}: the saved forecasts do not cover the scored rows")
        probs = np.concatenate([per_year[y]["output:probs"] for y in years])
        s = _score(probs, a.label[rows], clim, a.open_ms[rows], rcfg, name)
        s["fallbacks"] = int(sum(per_year[y]["output:fallback"].sum() for y in years))
        systems[name] = s
    out["systems"] = systems
    honest = [s for n, s in systems.items() if SPECS[n].honest]
    check = {}
    expected = [n for n in SKILL_CONTROLS if n in cfg.controls]  # every configured control that forecasts
    for name in expected:
        if name not in systems:
            raise ValueError(f"{name}: no honest run to score for the skill check")
        b = systems[name]["bss"]
        above = all(b["low"] > h["bss"]["high"] for h in honest)
        check[name] = {"bss": b["estimate"], "low": b["low"], "high": b["high"], "at_least": cfg.min_control_bss,
                       "interval_above_every_honest": above, "met": bool(b["estimate"] >= cfg.min_control_bss and above)}  # fmt: skip
    out["control_check"] = check
    out["control_check_met"] = bool(check) and all(c["met"] for c in check.values())
    if POSITIVE in outputs:
        per_year = outputs[POSITIVE]
        y_pos = past_window_labels(data)
        pos = np.concatenate([per_year[y]["aux:positions"] for y in years])
        if not np.array_equal(pos, rows):
            raise ValueError("pos-window: the saved forecasts do not cover the scored rows")
        keep = y_pos[rows] >= 0
        r = rows[keep]
        probs = np.concatenate([per_year[y]["output:probs"] for y in years])[keep]
        clim_pos = monthly_climatology(a, y_pos, a.month_ms[r])
        s = _score(probs, y_pos[r], clim_pos, a.open_ms[r], rcfg, POSITIVE)
        s["rows_without_label"] = int((~keep).sum())
        s["reference"] = "climatology of its own label"
        s["at_least"] = cfg.min_positive_bss
        s["met"] = bool(s["bss"]["estimate"] >= cfg.min_positive_bss)
        out["positive"] = s
    return out


# --- Parts --------------------------------------------------------------------------------------


class ResumeRefused(RuntimeError):
    """The parts folder belongs to another configuration, code or data."""


class Interrupted(RuntimeError):
    """Raised after ``stop_after`` new parts (tests of the resume)."""


class Parts:
    """Finished parts under ``<out>/parts/``, each beside its SHA-256; written atomically, verified when read."""

    def __init__(self, out: Path, identity: dict[str, Any], *, log: Callable[[str], None] = lambda s: None, stop_after: int | None = None) -> None:
        self.dir = out / "parts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.stop_after = stop_after
        self.saved = 0
        text = json.dumps(identity, sort_keys=True, indent=1) + "\n"
        path = self.dir / "IDENTITY.json"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                old, new = json.loads(path.read_text(encoding="utf-8")), json.loads(text)
                changed = []
                for k in sorted(set(old) | set(new)):
                    a, b = old.get(k), new.get(k)
                    if a == b:
                        continue
                    if isinstance(a, dict) and isinstance(b, dict):  # name the field: build.driver, build.torch, ...
                        changed += [f"{k}.{f}" for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
                    else:
                        changed.append(k)
                raise ResumeRefused(f"{self.dir} holds parts of another run: {', '.join(changed)} changed. Use another --out.")
        else:
            _write_atomic(path, text.encode("utf-8"))

    def get(self, key: str) -> bytes | None:
        path, digest = self.dir / key, self.dir / (key + ".sha256")
        if not (path.exists() and digest.exists()):
            return None
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest.read_text(encoding="ascii").strip():
            self.log(f"part {key} fails its digest: recomputed")
            return None
        return data

    def put(self, key: str, data: bytes) -> str:
        if self.stop_after is not None and self.saved >= self.stop_after:
            raise Interrupted(f"stopped after {self.saved} new parts")
        sha = hashlib.sha256(data).hexdigest()
        _write_atomic(self.dir / key, data)
        _write_atomic(self.dir / (key + ".sha256"), (sha + "\n").encode("ascii"))
        self.saved += 1
        return sha

    def get_json(self, key: str) -> Any:
        data = self.get(key)
        return None if data is None else json.loads(data)

    def put_json(self, key: str, obj: Any) -> None:
        self.put(key, json.dumps(obj, sort_keys=True).encode("utf-8"))

    def get_arrays(self, key: str) -> dict[str, np.ndarray] | None:
        data = self.get(key)
        return None if data is None else checkpoint.from_bytes(data)[0]

    def put_arrays(self, key: str, arrays: dict[str, np.ndarray], meta: dict[str, str]) -> None:
        self.put(key, checkpoint.to_bytes(arrays, meta))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _digest(items: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(items):
        h.update(key.encode() + b"|" + str(items[key].shape).encode() + b"|")
        h.update(np.ascontiguousarray(items[key]).tobytes())
    return h.hexdigest()


# --- The runner ---------------------------------------------------------------------------------


class Runner:
    """Every run of the encoder H1 over one sub-panel, through the parts store."""

    def __init__(self, sub: Subpanel, cfg: EncoderH1Config, parts: Parts, *, log: Callable[[str], None] = lambda s: None) -> None:
        self.sub, self.cfg, self.parts, self.log = sub, cfg, parts, log
        self.cuts = h1_cuts(cfg, sub.data.arrays)
        self.regular = [c for c in self.cuts if c.regular]
        self.cut_years = sorted({c.year for c in self.regular})
        self.timings: list[dict[str, Any]] = []
        self._cache: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}

    # --- plan ---

    def years(self, spec: Spec) -> list[int]:
        if spec.name == POSITIVE:
            return self.cut_years
        return sorted({c.year for c in self.cuts if runs_at(self.cfg, spec, c)})

    def mem_until(self, year: int) -> int | None:
        ends = [c.until_ms for c in self.regular if c.year == year]
        return max(ends) if ends else None

    def perturbed_specs(self) -> list[Spec]:
        return [s for s in self.cfg.specs() if s.perturbed]

    def plan(self, stage: str) -> dict[str, Any]:
        """The fits a stage makes and their training steps, from the panel's training masks (no label value is read)."""
        precheck = stage in ("precheck", "all")
        h1 = stage in ("h1", "floor", "all")
        fits: dict[tuple[str, int], int] = {}
        for spec in self.cfg.specs():
            for year in self.years(spec):
                n = 0
                if precheck and spec.honest:
                    n += 2
                elif precheck and spec.name == "cheat-stamp" and year in {c.year for c in self.regular}:
                    n += 1  # its run 1, for the scope test at every cut; the same run H1 uses
                if h1:
                    n = max(n, 1) + 2 * sum(1 for c in self.cuts if c.year == year and runs_at(self.cfg, spec, c) and retrains_at(self.cfg, spec, c))
                if stage == "floor" and spec.perturbed:
                    n += self.cfg.floor_runs - (2 if spec.honest and precheck else 1)
                if n:
                    fits[(spec.name, year)] = n
        steps = 0
        for (name, year), n in fits.items():
            system = make_system(SPECS[name], self.sub.data, self.cfg, year)
            a = self.sub.data.arrays
            weeks = np.unique((a.decision_ms[system.training_mask()] - 4 * DAY_MS) // (7 * DAY_MS)).size
            steps += n * weeks * system.cfg.epochs
        return {"fits": int(sum(fits.values())), "steps": int(steps)}

    # --- unperturbed runs ---

    def run_key(self, name: str, year: int, rep: int) -> str:
        return f"run.{name}.{year}.r{rep}.safetensors"

    def ensure_run(self, spec: Spec, year: int, rep: int) -> dict[str, np.ndarray]:
        """The unperturbed run ``rep`` of a system's fit of ``year`` (1: H1's honest run; 2: the pre-check's; 3: the floor's)."""
        cached = self._cache.get((spec.name, year, rep))
        if cached is not None:
            return cached
        key = self.run_key(spec.name, year, rep)
        arr = self.parts.get_arrays(key)
        if arr is None:
            t = time.perf_counter()
            system = make_system(spec, self.sub.data, self.cfg, year)
            res = system.fit()
            arr = fit_arrays(system, res)
            if spec.forecasts:
                rows = rows_of_year(self.sub.data.arrays, year)
                arr |= forecast_arrays(system, res, rows, None if spec.name == POSITIVE else self.mem_until(year))
            self.parts.put_arrays(key, arr, {"system": spec.name, "year": str(year), "rep": str(rep), "config": system.cfg.to_json()})
            self._timed(spec.name, year, f"run {rep}", res, time.perf_counter() - t)
        if rep == 1 and spec.forecasts and self.parts.get(f"out.{spec.name}.{year}.safetensors") is None:
            out = {k: arr[k] for k in ("aux:positions", "output:raw", "output:probs", "output:fallback")}
            self.parts.put_arrays(f"out.{spec.name}.{year}.safetensors", out, {"system": spec.name, "year": str(year)})
        self._cache[(spec.name, year, rep)] = arr
        return arr

    def _timed(self, name: str, year: int, what: str, res: FitResult | None, seconds: float) -> None:
        steps = res.steps if res is not None else 0
        train = res.timing.get("train_s", 0.0) if res is not None else 0.0
        self.timings.append({"system": name, "year": year, "what": what, "steps": steps, "train_s": train, "seconds": seconds})
        tail = f", {steps} steps, {train / steps:.3f} s/step" if steps else ""
        self.log(f"{name} {year} {what}: {seconds:.1f} s{tail}")

    def drop_cache(self, keep_year: int | None = None) -> None:
        self._cache = {k: v for k, v in self._cache.items() if k[1] == keep_year}

    # --- pre-check ---

    def precheck(self) -> None:
        arms = [SPECS[a] for a in self.cfg.arms]
        for year in sorted({y for s in arms for y in self.years(s)}):
            for spec in arms:
                if year not in self.years(spec):
                    continue
                key = f"pre.{spec.name}.{year}.json"
                if self.parts.get_json(key) is None:
                    a1, a2 = self.ensure_run(spec, year, 1), self.ensure_run(spec, year, 2)
                    rec = compare_record(compared(a1, "full"), compared(a2, "full"))
                    self.parts.put_json(key, rec)
                    self.log(f"pre-check {spec.name} {year}: {rec['violations']} differing values of {rec['compared_values']}")
            self.drop_cache()
        first = self.regular[0]
        for spec in arms:
            key = f"scope.{spec.name}.{first.year}.json"
            if self.parts.get_json(key) is not None:
                continue
            rec = self.scope_record(spec, first)
            self.parts.put_json(key, rec)
            self.log(f"scope test {spec.name}: {rec['violations']} differing values of {rec['compared_values']}")
        self.drop_cache()

    def scope_record(self, spec: Spec, cut: H1Cut) -> dict[str, Any]:
        """The scope test at ``cut``: the honest fit of the cut's year loaded (no training), its forecasts of the rows
        decided at or before the cut compared bit for bit with the full honest run's on those rows (outputs and memory)."""
        a1 = self.ensure_run(spec, cut.year, 1)
        system = make_system(spec, self.sub.data, self.cfg, cut.year)
        rows = rows_of_year(self.sub.data.arrays, cut.year)
        rows = rows[self.sub.data.arrays.decision_ms[rows] <= cut.until_ms]
        part = forecast_arrays(system, result_from(a1, system.cfg), rows, cut.until_ms)
        rec = compare_record(compared(a1, "cut-forecast", cut.until_ms), compared(part, "cut-forecast", cut.until_ms))
        rec["cut"], rec["rows"] = cut.label, int(rows.size)
        return rec

    def scope_systems(self) -> list[Spec]:
        """The systems of the scope test at every cut: the honest arms and ``cheat-stamp``."""
        return [SPECS[n] for n in (*self.cfg.arms, *(c for c in self.cfg.controls if c == "cheat-stamp"))]

    def scope_every_cut(self) -> None:
        """The scope test at every regular cut (a description, apart from the pre-check's verdict). Forecasts only:
        it loads the honest runs (``cheat-stamp``'s run 1 is the one H1 uses)."""
        for year in sorted({c.year for c in self.regular}):
            for spec in self.scope_systems():
                for cut in (c for c in self.regular if c.year == year):
                    key = f"scopecut.{spec.name}.{cut.label}.json"
                    if self.parts.get_json(key) is not None:
                        continue
                    t = time.perf_counter()
                    rec = self.scope_record(spec, cut)
                    self.parts.put_json(key, rec)
                    self.log(f"scope at cut {cut.label} {spec.name}: {rec['violations']} differing values of {rec['compared_values']}, "
                             f"{rec['rows']} rows forecast in {time.perf_counter() - t:.1f} s")  # fmt: skip
            self.drop_cache()

    # --- H1 ---

    def cut_key(self, name: str, cut: H1Cut, direction: str) -> str:
        return f"cut.{name}.{cut.year}.{cut.label}.{FILE_DIRECTION[direction]}.json"

    def h1(self) -> None:
        specs = self.cfg.specs()
        directions = [pi.DIRECTIONS.get(s, f"x{s:g}") for s in self.cfg.panel.level_shifts]
        for year in sorted({y for s in specs for y in self.years(s)}):
            for spec in specs:
                if year in self.years(spec):
                    self.ensure_run(spec, year, 1)
            for cut in sorted((c for c in self.cuts if c.year == year), key=lambda c: c.hour_ms):
                here = [s for s in self.perturbed_specs() if runs_at(self.cfg, s, cut)]
                for j, direction in enumerate(directions):
                    todo = [s for s in here if self.parts.get_json(self.cut_key(s.name, cut, direction)) is None]
                    if not todo:
                        continue
                    t = time.perf_counter()
                    pb, left = pi.perturbed_bars(self.sub.bars, cut.panel_cut(), cut.k, j, self.cfg.panel)
                    data_p = build_data(pb, self.sub.months)
                    self.log(f"cut {cut.label} {direction}: perturbed sub-panel built in {time.perf_counter() - t:.1f} s")
                    for spec in todo:
                        self.perturbed(spec, cut, direction, data_p, left)
            self.drop_cache()

    def perturbed(self, spec: Spec, cut: H1Cut, direction: str, data_p: EncoderData, left: dict[str, str]) -> None:
        t = time.perf_counter()
        base = self.ensure_run(spec, cut.year, 1)
        retrain = retrains_at(self.cfg, spec, cut)
        scope = scope_of(cut, retrain)
        system = make_system(spec, data_p, self.cfg, cut.year)
        res = None
        if retrain:
            res = system.fit()
            arr = fit_arrays(system, res)
        else:
            arr = {}
        if cut.regular:
            rows = rows_of_year(data_p.arrays, cut.year)
            rows = rows[data_p.arrays.decision_ms[rows] <= cut.until_ms]
            if not rows.size:
                raise ValueError(f"cut {cut.label} has no row of {cut.year} decided by it")
            arr |= forecast_arrays(system, res if res is not None else result_from(base, system.cfg), rows, cut.until_ms)
        items = compared(arr, scope, cut.until_ms)
        rec = compare_record(compared(base, scope, cut.until_ms), items)
        rec |= {"left_out": left, "retrained": retrain, "scope": scope, "perturbed_digest": _digest(items)}
        self.parts.put_json(self.cut_key(spec.name, cut, direction), rec)
        self._timed(spec.name, cut.year, f"cut {cut.label} {direction}", res, time.perf_counter() - t)

    # --- floor ---

    def floor(self) -> None:
        for spec in self.perturbed_specs():
            for year in self.years(spec):
                cuts = [c for c in self.cuts if c.year == year and runs_at(self.cfg, spec, c)]
                keys = {c.label: f"floor.{spec.name}.{year}.{c.label}.json" for c in cuts}
                if all(self.parts.get_json(k) is not None for k in keys.values()):
                    continue
                reps = [self.ensure_run(spec, year, r) for r in range(1, self.cfg.floor_runs + 1)]
                for cut in cuts:
                    if self.parts.get_json(keys[cut.label]) is not None:
                        continue
                    scope = scope_of(cut, retrains_at(self.cfg, spec, cut))
                    items = [compared(r, scope, cut.until_ms) for r in reps]
                    floor: dict[str, float] = {}
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            for key, v in item_moves(items[i], items[j]).items():
                                floor[key] = max(floor.get(key, 0.0), v["largest"] if v["largest"] > 0 else _ulp_move(items[i][key], items[j][key]))
                    cats = sorted({category(k) for k in items[0]})
                    self.parts.put_json(keys[cut.label], {"floor": floor, "items": len(items[0]), "categories": cats, "runs": len(items)})
                self.drop_cache()

    # --- the report's inputs ---

    def outputs(self) -> dict[str, dict[int, dict[str, np.ndarray]]]:
        out: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for spec in self.cfg.specs():
            if not spec.forecasts or spec.name in ("cheat-purge", "cheat-slice"):
                continue
            per = {}
            for y in self.cut_years:
                arr = self.parts.get_arrays(f"out.{spec.name}.{y}.safetensors")
                if arr is None:
                    return {}
                per[y] = arr
            out[spec.name] = per
        return out


def _ulp_move(x: np.ndarray, y: np.ndarray) -> float:
    """A move below every float's resolution (a NaN payload or a signed zero): counted as the smallest positive float."""
    return float(np.finfo(np.float64).tiny) if not np.array_equal(x, y, equal_nan=True) else 0.0


# --- Assembling the result ----------------------------------------------------------------------


def _top_items(items: dict[str, dict[str, Any]], n: int = 20) -> dict[str, dict[str, Any]]:
    keys = sorted(items, key=lambda k: (-items[k]["caught"], -items[k]["differing"], k))[:n]
    return {k: items[k] for k in sorted(keys)}


def _record_view(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in ("violations", "caught", "largest_change", "compared_values", "by_category")} | {"items": _top_items(rec["items"])}


def precheck_summary(runner: Runner) -> dict[str, Any] | None:
    arms: dict[str, Any] = {}
    for name in runner.cfg.arms:
        spec = SPECS[name]
        years = {}
        for y in runner.years(spec):
            rec = runner.parts.get_json(f"pre.{name}.{y}.json")
            if rec is None:
                return None
            years[str(y)] = _record_view(rec)
        scope = runner.parts.get_json(f"scope.{name}.{runner.regular[0].year}.json")
        if scope is None:
            return None
        cats: dict[str, dict[str, int]] = {}
        for rec in years.values():
            for c, v in rec["by_category"].items():
                t = cats.setdefault(c, {"items": 0, "values": 0, "differing": 0})
                t["items"] += v["items"]
                t["values"] += v["values"]
                t["differing"] += v["differing"]
        arms[name] = {"years": years, "by_category": cats, "identical": all(r["violations"] == 0 for r in years.values()),
                      "scope": {k: scope[k] for k in ("cut", "rows", "violations", "compared_values", "by_category")} | {"items": _top_items(scope["items"])},
                      "scope_identical": scope["violations"] == 0}  # fmt: skip
    return {"arms": arms, "passes": all(a["identical"] and a["scope_identical"] for a in arms.values())}


def scope_every_cut_summary(runner: Runner) -> dict[str, Any] | None:
    """The scope test at every regular cut, per system and cut: a description, apart from the pre-check's verdict."""
    systems: dict[str, Any] = {}
    for spec in runner.scope_systems():
        cuts = {}
        for cut in runner.regular:
            rec = runner.parts.get_json(f"scopecut.{spec.name}.{cut.label}.json")
            if rec is None:
                return None
            cuts[cut.label] = {k: rec[k] for k in ("rows", "violations", "compared_values", "by_category")} | {"items": _top_items(rec["items"])}
        systems[spec.name] = {"cuts": cuts, "identical": all(c["violations"] == 0 for c in cuts.values())}
    return {"systems": systems, "identical": all(s["identical"] for s in systems.values()), "description": True}


def _pattern(cuts: Sequence[H1Cut], marks: dict[str, str]) -> str:
    regular = "".join(marks.get(c.label, "-") for c in cuts if c.regular)
    purge = "".join(marks.get(c.label, "-") for c in cuts if not c.regular)
    return f"{regular}|{purge}"


def system_summary(runner: Runner, spec: Spec, directions: Sequence[str]) -> dict[str, Any] | None:
    records = []
    for cut in runner.cuts:
        if not runs_at(runner.cfg, spec, cut):
            continue
        by_dir = {}
        for d in directions:
            rec = runner.parts.get_json(runner.cut_key(spec.name, cut, d))
            if rec is None:
                return None
            by_dir[d] = rec
        retrain = retrains_at(runner.cfg, spec, cut)
        system = make_system(spec, runner.sub.data, runner.cfg, cut.year)
        first = by_dir[directions[0]]
        records.append({
            "cut": cut.label, "hour": hour_text(cut.hour_ms), "kind": cut.kind, "year": cut.year, "retrained": retrain,
            "scope": scope_of(cut, retrain), "predicted": predicted(system, cut, scope_of(cut, retrain)),
            "left_out": first["left_out"], "compared_values": first["compared_values"],
            "violations": int(sum(r["violations"] for r in by_dir.values())), "caught": int(sum(r["caught"] for r in by_dir.values())),
            "by_direction": {d: _record_view(r) | {"perturbed_digest": r["perturbed_digest"]} for d, r in by_dir.items()},
            "_items": {d: r["items"] for d, r in by_dir.items()},
        })  # fmt: skip
    return {"name": spec.name, "arm": spec.arm, "honest": spec.honest, "leak": spec.leak, "records": records, "verdict": verdict(spec, records, runner.cuts, directions)}


def verdict(spec: Spec, records: Sequence[dict[str, Any]], cuts: Sequence[H1Cut], directions: Sequence[str]) -> dict[str, Any]:
    predicted_marks = {r["cut"]: "X" if r["predicted"] else "." for r in records}
    if spec.honest:
        flagged = {r["cut"]: "X" if r["violations"] else "." for r in records}
        rule_errors = [r["cut"] for r in records if r["predicted"]]
        return {"ok": not any(r["violations"] for r in records) and not rule_errors, "pattern": _pattern(cuts, flagged),
                "predicted_pattern": _pattern(cuts, predicted_marks), "violations": int(sum(r["violations"] for r in records)),
                "rule_errors": rule_errors}  # fmt: skip
    patterns, ok, catches = {}, True, []
    for d in directions:
        marks = {}
        for r in records:
            hit = r["by_direction"][d]["caught"] > 0
            marks[r["cut"]] = "X" if hit else "."
            ok &= hit == r["predicted"]
            if hit:
                catches.append(r["by_direction"][d]["largest_change"])
        patterns[d] = _pattern(cuts, marks)
    finite = [c for c in catches if c is not None and math.isfinite(c)]
    return {"ok": bool(ok), "pattern_by_direction": patterns, "predicted_pattern": _pattern(cuts, predicted_marks),
            "smallest_catch": min(finite) if finite else None}  # fmt: skip


def floor_summary(runner: Runner, systems: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Per system: the floor over the honest runs per category, and each control's smallest catch as a multiple of it."""
    out: dict[str, Any] = {}
    for s in systems:
        spec = SPECS[s["name"]]
        floors_by_cut: dict[str, dict[str, float]] = {}
        per_cat: dict[str, float] = {}
        for r in s["records"]:
            part = runner.parts.get_json(f"floor.{spec.name}.{r['year']}.{r['cut']}.json")
            if part is None:
                return None
            floors_by_cut[r["cut"]] = part["floor"]
            for c in part["categories"]:
                per_cat.setdefault(c, 0.0)
            for key, v in part["floor"].items():
                per_cat[category(key)] = max(per_cat[category(key)], v)
        entry: dict[str, Any] = {"floor_by_category": {c: per_cat[c] for c in CATEGORIES if c in per_cat},
                                 "floor_zero": all(v == 0.0 for v in per_cat.values())}  # fmt: skip
        if not spec.honest:
            smallest = None
            for r in s["records"]:
                floor = floors_by_cut[r["cut"]]
                for d, items in r["_items"].items():
                    if r["by_direction"][d]["caught"] == 0:
                        continue
                    ratios = [math.inf if floor.get(k, 0.0) == 0 else v["largest"] / floor[k] for k, v in items.items() if v["caught"]]
                    sep = max(ratios) if ratios else math.inf
                    smallest = sep if smallest is None else min(smallest, sep)
            entry["smallest_catch_multiple"] = smallest  # inf: every caught item's floor is 0
            entry["caught_anywhere"] = smallest is not None
            entry["weakly_separated"] = smallest is not None and smallest < runner.cfg.weak_multiple
        out[spec.name] = entry
    return out


def assemble(runner: Runner, stages: Sequence[str], prov: dict[str, Any], members: Sequence[str]) -> dict[str, Any]:
    cfg = runner.cfg
    directions = [pi.DIRECTIONS.get(s, f"x{s:g}") for s in cfg.panel.level_shifts]
    result: dict[str, Any] = {
        "registered": cfg.registered,
        "device": cfg.device,
        "config": config_json(cfg),
        "members": list(members),
        "stages": list(stages),
        "cuts": [{"cut": c.label, "k": c.k, "hour": hour_text(c.hour_ms), "until": pi.iso(c.until_ms), "kind": c.kind, "year": c.year}
                 for c in runner.cuts],  # fmt: skip
        "precheck": precheck_summary(runner),
        "scope_every_cut": scope_every_cut_summary(runner) if "precheck" in stages else None,
        "provenance": prov,
    }
    systems = []
    if "h1" in stages:
        for spec in runner.perturbed_specs():
            s = system_summary(runner, spec, directions)
            if s is None:
                raise RuntimeError(f"{spec.name}: the H1 stage left a part missing")
            systems.append(s)
        outputs = runner.outputs()
        result["skill"] = skill(runner.sub.data, outputs, runner.cut_years, cfg) if outputs else None
    result["floor"] = floor_summary(runner, systems) if "floor" in stages else None
    for s in systems:
        for rec in s["records"]:
            rec.pop("_items")
    result["systems"] = systems
    honest = [s for s in systems if s["honest"]]
    controls = [s for s in systems if not s["honest"]]
    sk = result.get("skill") or {}
    # H1 as docs/HYPOTHESES.md defines it: every honest arm bit for bit at every cut, every control caught on
    # its predicted pattern in each direction. The skill check and pos-window's check are their own verdicts.
    result["h1"] = None if not systems else {
        "honest_violations": int(sum(s["verdict"]["violations"] for s in honest)),
        "honest_flagged": [s["name"] for s in honest if not s["verdict"]["ok"]],
        "controls_off_pattern": [s["name"] for s in controls if not s["verdict"]["ok"]],
        "holds": bool(all(s["verdict"]["ok"] for s in systems)),
    }  # fmt: skip
    check, positive = sk.get("control_check"), sk.get("positive")
    result["verdicts"] = None if not systems else {
        "h1": result["h1"]["holds"],
        "skill_check": None if not check else bool(sk["control_check_met"]),
        "skill_check_by_control": None if not check else {n: c["met"] for n, c in check.items()},
        "pos_window": None if positive is None else bool(positive["met"]),
    }  # fmt: skip
    stamp = next((s for s in systems if s["name"] == "cheat-stamp"), None)
    result["cheat_stamp_caught_bit_for_bit"] = None if stamp is None else bool(stamp["verdict"]["ok"] and all(s["verdict"]["ok"] for s in honest))
    return result


# --- Output -------------------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _f(x: float | None, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _g(x: float | None) -> str:
    return "inf (shape or missing)" if x is None else f"{x:.3g}"


def _pass(ok: bool | None) -> str:
    return "not run" if ok is None else ("pass" if ok else "FAIL")


def _skill_line(r: dict[str, Any], name: str) -> str:
    """A control's skill check, on one line: its BSS, interval and pass or fail."""
    c = ((r.get("skill") or {}).get("control_check") or {}).get(name)
    if c is None:
        return "not run."
    above = "above" if c["interval_above_every_honest"] else "NOT above"
    return (f"Brier skill score {_f(c['bss'])} [{_f(c['low'])}, {_f(c['high'])}], interval {above} every honest arm's; "
            f"at least {c['at_least']}: **{_pass(c['met'])}**")


def render_md(r: dict[str, Any]) -> str:
    lines = ["# The panel H1 of the encoder arms", ""]
    if not r["registered"]:
        lines += ["**NOT THE REGISTERED RUN.** This run's configuration differs from the pre-registered one "
                  f"({'a synthetic sub-panel' if r['config']['synthetic'] else 'a smoke run or a test'}): its verdicts are not results.", ""]  # fmt: skip
    lines += [f"Device: `{r['device']}`. Stages run: {', '.join(r['stages']) or 'none'}.", ""]
    pre = r["precheck"]
    lines += ["## 1. The pre-check", ""]
    if pre is None:
        lines += ["Not run: H1 is not known to be bit for bit on this device.", ""]
    else:
        lines += [f"**The pre-check {'passes' if pre['passes'] else 'FAILS'}.** Two honest runs of each arm with identical inputs, "
                  "compared bit for bit on every fit year (inputs, every epoch's parameters, slice, memory, outputs), and the scope test "
                  "at the first cut.", "",
                  "| arm | item | values | differing |", "|---|---|---|---|"]  # fmt: skip
        for arm, a in pre["arms"].items():
            for c in CATEGORIES:
                if c in a["by_category"]:
                    v = a["by_category"][c]
                    lines.append(f"| {arm} | {c} | {v['values']} | {v['differing']} |")
            s = a["scope"]
            lines.append(f"| {arm} | scope test, cut {s['cut']} ({s['rows']} rows) | {s['compared_values']} | {s['violations']} |")
        if not pre["passes"]:
            lines += ["", "**First finding: the pre-check fails on this device.** Under the registered fallback, H1 runs on CPU torch; "
                      "this device's run is a description, with its floor."]  # fmt: skip
        lines.append("")
    lines += ["## 2. `cheat-stamp`", ""]
    caught = r.get("cheat_stamp_caught_bit_for_bit")
    stamp = next((s for s in r["systems"] if s["name"] == "cheat-stamp"), None)
    if stamp is None:
        lines += ["Not run.", ""]
    else:
        v = stamp["verdict"]
        pats = "; ".join(f"{d} `{p}`" for d, p in v["pattern_by_direction"].items())
        if caught:
            lines += [f"Caught bit for bit where predicted: {pats}, predicted `{v['predicted_pattern']}`; smallest catch {_g(v['smallest_catch'])}.", ""]
        else:
            lines += [f"**First finding: H1 cannot catch `cheat-stamp` bit for bit.** Caught {pats}, predicted `{v['predicted_pattern']}`; "
                      f"honest arms flagged: {', '.join((r['h1'] or {}).get('honest_flagged', [])) or 'none'}.", ""]  # fmt: skip
        lines += [f"Its skill: {_skill_line(r, 'cheat-stamp')}", ""]
    lines += ["## 3. `pos-window`", ""]
    pos = (r.get("skill") or {}).get("positive")
    if pos is None:
        lines += ["Not run.", ""]
    else:
        b = pos["bss"]
        lines += [f"Brier skill score {_f(b['estimate'])} [{_f(b['low'])}, {_f(b['high'])}] against the climatology of its own label, over "
                  f"{pos['rows']} rows and {pos['hours']} hours: **{'reaches' if pos['met'] else 'DOES NOT REACH'} {pos['at_least']}**."
                  + (f" {pos['rows_without_label']} rows had no label of its own." if pos["rows_without_label"] else ""), ""]  # fmt: skip
    h1, verdicts = r.get("h1"), r.get("verdicts")
    if h1 is not None and verdicts is not None:
        sk = r.get("skill") or {}
        lines += ["## 4. Three verdicts, reported apart", "",
                  "Each stands alone: a failed skill check does not make H1 fail, and H1 holding does not make the skill check pass.", "",
                  f"- **H1: {_pass(verdicts['h1'])}.** Every honest arm bit for bit at every cut (violations: {h1['honest_violations']}), "
                  "and every control caught on its predicted pattern, in each direction where predicted and in none where not "
                  f"(off pattern: {', '.join(h1['controls_off_pattern']) or 'none'}).",
                  f"- **The skill check: {_pass(verdicts['skill_check'])}.** Control by control, at least {r['config']['min_control_bss']} "
                  "with an interval above every honest arm's: "
                  + ("; ".join(f"`{n}` {_skill_line(r, n)}" for n in (sk.get("control_check") or {})) or "not run") + ".",
                  f"- **`pos-window`'s check: {_pass(verdicts['pos_window'])}.**"
                  + ("" if verdicts["pos_window"] is None else f" At least {sk['positive']['at_least']} against the climatology of its own label (section 3)."),
                  "", "A planted control is caught at a cut and in a direction where a compared value moves by more than "
                  f"{CAUGHT_ABOVE:g}, turns missing or stops being missing. An honest arm must not differ by a single bit.", ""]  # fmt: skip
    sc = r.get("scope_every_cut")
    if sc is not None:
        lines += ["## 5. Scope at every cut (description)", "",
                  "The registered scope test runs at the first cut. Here it runs at each regular cut, for the honest arms and "
                  "`cheat-stamp`: the honest fit of the cut's year loaded (no training), its forecasts of the rows decided at or "
                  "before the cut compared bit for bit with the full honest run's on those rows (outputs and memory statistics). "
                  "A description: it does not enter the pre-check's verdict nor the exit code.", "",
                  f"**{'Identical everywhere' if sc['identical'] else 'NOT IDENTICAL somewhere'}.**", "",
                  "| system | cut | rows | compared values | differing | items |", "|---|---|---|---|---|---|"]  # fmt: skip
        for name, s in sc["systems"].items():
            for label, c in s["cuts"].items():
                moved = [k for k, v in c["items"].items() if v["differing"]]
                items = ", ".join(moved[:4]) + (" ..." if len(moved) > 4 else "")
                lines.append(f"| {name} | {label} | {c['rows']} | {c['compared_values']} | {c['violations']} | {items or '-'} |")
        lines.append("")
    lines += ["## Cuts", "", "| cut | hour (UTC) | kind | fit | members left out |", "|---|---|---|---|---|"]
    lefts = {}
    for s in r["systems"]:
        for rec in s["records"]:
            lefts.setdefault(rec["cut"], rec["left_out"])
    for c in r["cuts"]:
        left = lefts.get(c["cut"], {})
        text = "; ".join(f"{k} ({why})" for k, why in sorted(left.items())) or "none"
        lines.append(f"| {c['cut']} | {c['hour']} | {c['kind']} | {c['year']} | {len(left)}: {text} |")
    lines.append("")
    if r["systems"]:
        lines += ["## Systems", "", "Patterns: the 8 cuts, then `|`, then F - 24 h and F - 30 days. X flagged, . not, - not run.", "",
                  "| system | arm | flagged | predicted | ok | violations or smallest catch |", "|---|---|---|---|---|---|"]  # fmt: skip
        for s in r["systems"]:
            v = s["verdict"]
            if s["honest"]:
                lines.append(f"| {s['name']} | {s['arm']} | `{v['pattern']}` | `{v['predicted_pattern']}` | {'yes' if v['ok'] else 'NO'} | {v['violations']} |")
            else:
                pats = "<br>".join(f"{d} `{p}`" for d, p in v["pattern_by_direction"].items())
                lines.append(f"| {s['name']} | {s['arm']} | {pats} | `{v['predicted_pattern']}` | {'yes' if v['ok'] else 'NO'} | {_g(v['smallest_catch'])} |")
        lines.append("")
        for s in r["systems"]:
            lines += [f"### {s['name']}", "", f"{s['leak']}.", "",
                      "| cut | scope | predicted | compared values | x3 differing / caught | /3 differing / caught | largest change | items |",
                      "|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for rec in s["records"]:
                d = list(rec["by_direction"].values())
                items = sorted({k for x in d for k in x["items"]})
                moves = [math.inf if x["largest_change"] is None else x["largest_change"] for x in d]
                lines.append(f"| {rec['cut']} | {rec['scope']} | {'X' if rec['predicted'] else '.'} | {rec['compared_values']} | "
                             f"{d[0]['violations']} / {d[0]['caught']} | {d[1]['violations']} / {d[1]['caught']} | {max(moves):.3g} | "
                             f"{', '.join(items[:4])}{' ...' if len(items) > 4 else ''} |")  # fmt: skip
            lines.append("")
    sk = r.get("skill")
    if sk:
        lines += ["## Skill on the cut years", "", f"Complete rows of the sub-panel in {', '.join(map(str, sk['years']))}, honest runs, "
                  "against `climatology-panel`; Report n°2's paired stationary bootstrap of the hourly Brier series.", "",
                  "| system | log-loss | Brier | ECE | BSS [95 %] | fallbacks |", "|---|---|---|---|---|---|"]  # fmt: skip
        for name, s in sk["systems"].items():
            b = s["bss"]
            lines.append(f"| {name} | {_f(s['log_loss'])} | {_f(s['brier'])} | {_f(s['ece'])} | {_f(b['estimate'])} [{_f(b['low'])}, {_f(b['high'])}] | {s['fallbacks']} |")
        lines += ["", "The registered skill check, control by control (" + ", ".join(f"`{n}`" for n in sk["control_check"]) + "): at least "
                  f"{r['config']['min_control_bss']}, with an interval above every honest arm's: "
                  f"**{_pass(sk['control_check_met'] if sk['control_check'] else None)}**. It is its own verdict, apart from H1's.", ""]  # fmt: skip
    fl = r.get("floor")
    if fl:
        lines += ["## The floor (the fallback's description)", "", f"Largest pairwise difference over {r['config']['floor_runs']} honest runs, per item "
                  f"category; a control's smallest catch as a multiple of its item's floor (weakly separated under {r['config']['weak_multiple']:g} x).", "",
                  "| system | " + " | ".join(CATEGORIES) + " | smallest catch / floor |", "|---" * (len(CATEGORIES) + 2) + "|"]  # fmt: skip
        for name, e in fl.items():
            cats = " | ".join("-" if c not in e["floor_by_category"] else _g(e["floor_by_category"][c]) for c in CATEGORIES)
            if "smallest_catch_multiple" not in e:
                text = "-"
            elif not e["caught_anywhere"]:
                text = "no catch"
            elif e["smallest_catch_multiple"] is None:
                text = "inf (floor 0)"
            else:
                text = f"{e['smallest_catch_multiple']:.3g}" + (" (weak)" if e["weakly_separated"] else "")
            lines.append(f"| {name} | {cats} | {text} |")
        lines.append("")
    lines += ["## Limits", "",
              "- H1 tests seed 1's pipeline; the model is the ensemble of three seeds.",
              "- Only the sub-panel's bars are perturbed and the systems run on the sub-panel alone.",
              "- The honest runs' forecasts are a first look at the encoder's skill on the sub-panel's cut years.", ""]  # fmt: skip
    if r.get("provenance"):
        lines += provenance.render_md(r["provenance"])
    return "\n".join(lines) + "\n"


def write(result: dict[str, Any], out: Path, run: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    clean = _clean(result)
    (out / "encoder-h1.json").write_text(json.dumps(clean, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (out / "encoder-h1.md").write_text(render_md(clean), encoding="utf-8", newline="\n")
    config = {"config": clean["config"], "provenance": clean["provenance"]}
    (out / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out / "run.json").write_text(json.dumps(run, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def identity_build(device: str, driver: str | None) -> dict[str, Any]:
    """The build a resume must match: :func:`olsprime.train.determinism.build` (it holds no wall clock) and the
    GPU driver as passed by ``--driver``."""
    return {**build_info(device), "driver": driver}


def run_record(device: str, started_at: datetime, seconds: float, driver: str | None) -> dict[str, Any]:
    """The wall clock and the machine: GPU name, driver, CUDA and torch versions. Nothing else."""
    info = build_info(device)
    return {"started_at": started_at.isoformat(timespec="seconds"), "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds": round(seconds, 1), "gpu": info.get("gpu"), "driver": driver, "cuda": info.get("cuda_build"), "torch": info.get("torch")}  # fmt: skip


# --- Entry points -------------------------------------------------------------------------------


def run(sub: Subpanel, cfg: EncoderH1Config, out: Path, stage: str, *, prov: dict[str, Any], log: Callable[[str], None] = lambda s: None,
        stop_after: int | None = None, driver: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Run ``stage`` (resuming from ``out/parts``), then write the outputs. Returns the result."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    started, started_at = time.perf_counter(), datetime.now(UTC)
    threads = torch.get_num_threads()
    if cfg.device == "cpu":
        torch.set_num_threads(cfg.cpu_threads)
    try:
        with deterministic():
            # The build under the settings in force (torch, CUDA, cuDNN, GPU, threads) and the driver: a resume
            # across a rebuilt environment or an updated driver is refused, naming what changed.
            identity = {"config": config_json(cfg), "code_sha256": prov.get("code", {}).get("sha256"),
                        "manifest_digest": prov.get("manifest_digest"), "build": identity_build(cfg.device, driver)}  # fmt: skip
            parts = Parts(out, json.loads(json.dumps(_clean(identity))), log=log, stop_after=stop_after)
            runner = Runner(sub, cfg, parts, log=log)
            plan = runner.plan(stage)
            log(f"plan of stage {stage}: {plan['fits']} fits, {plan['steps']} training steps (parts already saved are skipped)")
            stages: list[str] = []
            if stage in ("precheck", "all"):
                runner.precheck()
                runner.scope_every_cut()  # a description: outside the pre-check's verdict and the exit code
                stages.append("precheck")
            pre = precheck_summary(runner)
            if stage in ("h1", "floor") or (stage == "all" and pre is not None and pre["passes"]):
                runner.h1()
                stages.append("h1")
            elif stage == "all":
                log("the pre-check failed: the H1 stage does not start (run --stage h1 to run it anyway)")
            if stage == "floor":
                runner.floor()
                stages.append("floor")
            if pre is not None and "precheck" not in stages:
                stages.insert(0, "precheck")
            result = assemble(runner, stages, prov, sorted(sub.bars))
    finally:
        torch.set_num_threads(threads)
    write(result, out, run_record(cfg.device, started_at, time.perf_counter() - started, driver))
    log(f"steps timed: {sum(t['steps'] for t in runner.timings)}; fits: {sum(1 for t in runner.timings if t['steps'])}")
    return result


SMOKE_START = datetime(2018, 9, 1, tzinfo=UTC)


def smoke_config(device: str) -> EncoderH1Config:
    """The GPU smoke: the registered encoder and fit on 10 synthetic members, 2018-09 to 2019-02, the 2019 fit."""
    return EncoderH1Config(device=device, cuts=("2019-01-20T13:00", "2019-02-10T00:00"), choose_cuts=False, purge_fit_year=2019, n_boot=200,
                           synthetic="10 synthetic members, hourly from 2018-09-01 for 165 days (seed 0); the 2019 fit")  # fmt: skip


def _default_out() -> Path:
    return runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-encoder-h1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.encoder_h1", description="The panel H1 of the encoder arms.")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <runs>/<UTC date>-encoder-h1; a restart resumes it)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--driver", default=None, help="the GPU driver's version, recorded in run.json")
    parser.add_argument("--synthetic", action="store_true", help="the smoke: a synthetic sub-panel, never stored data")
    args = parser.parse_args(argv)
    from olsprime.train.determinism import apply

    apply()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available: run in WSL2 with the CUDA environment, or pass --device cpu")
    log = lambda s: print(f"{datetime.now(UTC):%H:%M:%S} {s}", flush=True)  # noqa: E731
    if args.synthetic:
        cfg = smoke_config(args.device)
        prov = provenance.record({})
        bars, months = synthetic_subpanel(10, start=SMOKE_START, hours=24 * 165)
        out = args.out or runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-encoder-h1-smoke"
    else:
        cfg = EncoderH1Config(device=args.device)
        prov = provenance.record(provenance.current_manifest())  # before any data is read
        bars, months = pi.load_subpanel(cfg.panel)
        out = args.out or _default_out()
    t = time.perf_counter()
    sub = subpanel(bars, months)
    log(f"sub-panel: {len(bars)} members, {sub.data.arrays.n} rows, built in {time.perf_counter() - t:.1f} s")
    result = run(sub, cfg, out, args.stage, prov=prov, log=log, driver=args.driver)
    pre, v = result["precheck"], result["verdicts"]
    log(f"pre-check: {'not run' if pre is None else 'passes' if pre['passes'] else 'FAILS'}; "
        + ("H1: not run" if v is None else f"H1: {_pass(v['h1'])}; skill check: {_pass(v['skill_check'])}; pos-window: {_pass(v['pos_window'])}")
        + f"; written to {out}")  # fmt: skip
    # Exit codes: 2 the pre-check fails; 1 H1 fails; 3 H1 holds but the skill check or pos-window's check fails.
    if pre is not None and not pre["passes"]:
        return 2
    if v is None:
        return 0
    if not v["h1"]:
        return 1
    return 3 if v["skill_check"] is False or v["pos_window"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
