"""!!! LEAKY FUNCTIONS, ON PURPOSE. NOT MODELS. !!!

H1's adversary (``docs/PROTOCOL.md``, "Leakage tests", "The adversary (iteration 3)"). This module
builds thousands of functions that read the future **on purpose**, to measure how many of them H1's
perturbation catches. Nothing here may be imported by model code (``olsprime.models``,
``olsprime.train``); a test enforces it. Never report a leak's output as a forecast.

**What the reported share describes.** The population is exactly the output of
:func:`generate` for ``(n, seed)``: leak ``i`` is drawn from its own generator,
``numpy.random.default_rng([seed, i])``, so the first 200 leaks of a run of 10,000 are the leaks of
a run of 200. The share caught is this generator's detection rate, not a rate over all leaks.

**A leak** is a deterministic function of one pair's bars that gives, for every row ``s``, one
output: ``target`` in {0, 1}, ``exp_return`` (continuous), or three probabilities. It reads at
least one bar after ``s`` (at most ``s + FUTURE_REACH``, 10) and no bar before
``s - 189``. It has no fit. Values are per-bar quantities (:data:`QUANTITY_NAMES`) read at
``s + o``; a row outside the series reads NaN, and every comparison with NaN is false.

- **Quantities.** Every column :mod:`olsprime.bench.perturb` moves (:data:`LEVELS`: the four
  prices, ``volume``, ``quote_volume``, ``trades``, both taker volumes), the ratios inside a bar
  (:data:`RATIOS`: VWAP, taker VWAP, trade size, taker base and quote shares, body, upper and lower
  wick, range, in log units for the last four) and the shapes (fill = body / range, close location,
  ``wick_ratio = ln(upper / lower)``, the colour). An undefined value (a zero denominator) is NaN.
- **Draws used below.** ``j``, a future offset: 1, 2, 3 or 5 with weights 0.4, 0.2, 0.2, 0.2.
  ``k``, a past offset: uniform on 0..10 (probability 0.5), 11..90 (0.25) or 91..180 (0.25), so a
  quarter of the references lie beyond H1's 90-bar window. ``op``: ``>`` or ``<``, even odds.
  ``margin``: 1 (probability 0.7) or log-uniform on [1/4, 4]. A comparison is
  ``lhs op margin * rhs``.
- **Thresholds and their position.** On body, each wick, range, fill and close location, a
  threshold is drawn inside H1's band (:data:`BANDS`, the bands of ``perturb.py``) with probability
  0.5, log-uniformly for the log quantities and uniformly for fill and close location, and
  otherwise below or above it (:data:`OUTSIDE`, even odds). Its ``position`` tag is ``inside`` or
  ``outside``. On a taker share the threshold is uniform on [0.05, 0.95] and its band depends on
  the window (``window-dependent``); on the wick ratio it is uniform on [-2, 2] (``none``). For a
  comparison with a margin the ``position`` is ``inside`` when ``1/PUSH <= margin <= PUSH``.

**Families,** drawn with the fixed weights :data:`FAMILY_WEIGHTS`:

1. ``levels`` (0.20): a level ``q`` of bar ``s + j`` against the same level at ``s - k``
   (probability 0.75) or at another future bar ``s + j'``, ``j' != j``.
2. ``ratios`` (0.20): a ratio ``q`` of bar ``s + j`` against the same ratio at ``s - k``
   (probability 0.55), else against a threshold (scale-free ratios), the close at ``s - k`` (VWAP
   and taker VWAP) or the same ratio at ``s - k`` (trade size).
3. ``shapes`` (0.20), one of eleven kinds with even odds: a fill or a close-location threshold, a
   fill or a close location against ``s - k`` (``k = 0`` with probability 0.5), the colour (green
   or red), the same colour as bar ``s - k``, a wick-ratio threshold, a wick ratio against
   ``s - k`` (``k = 0`` with probability 0.5, margin 1: the ratio can be negative), a hammer (lower wick ``>= a`` x body and upper wick ``<=`` body, ``a`` uniform on
   [1, 3]), a shooting star (the mirror), and a colour x size cell (red or green; small or big
   against a body boundary that is ``SMALL_BODY`` with probability 0.5, else a body threshold).
4. ``aggregates`` (0.15): ``op`` in max, min, mean, sum of ``q`` (a level or a ratio) over bars
   ``s + 1 .. s + m``, ``m`` in {2, 3, 5, 10}, against the same aggregate over ``s - k - m + 1 ..
   s - k`` or against ``q[s - k]`` (even odds).
5. ``continuous`` (0.10), as ``exp_return`` (probability 0.8) or probabilities, one of four kinds:
   ``ln(q[s + j] / q[s - k])`` for a positive level or VWAP, taker VWAP or trade size;
   ``q[s + j] / mean(q[s - w + 1 .. s]) - 1`` for a level or a ratio, ``w`` in {5, 20, 90, 180};
   the raw value of a scale-free ratio or shape at ``s + j``; ``ln(agg(q[s + 1 .. s + m]) /
   close[s])`` for the high, low, close or VWAP and ``agg`` in max, min, mean. These should always
   be caught.
6. ``combinations`` (0.15): AND, OR, XOR of two leaks of families 1-4, or a weighted sum
   ``w1 x a + w2 x b`` (``|w|`` uniform on [0.25, 1], random signs) of two leaks of families 1-5,
   a binary one counting as 0 or 1. A weighted sum is an ``exp_return``.

Families 1-4 and the logical combinations output ``target`` (probability 0.8) or probabilities.
Probabilities are ``p_up = 0.8 sigmoid(z)``, ``p_down = 0.8 sigmoid(-z)``, ``p_neutral = 0.2``,
with ``z = 2 x target - 1`` for a binary leak and ``z`` the value for a continuous one.

**Tags.** Each leak carries ``family``, ``quantity``, ``operation``, ``offset`` (its future
reach), ``reference`` (what the future is compared with), ``position``, ``output`` and
``designed``. ``designed`` is ``yes`` when the protocol's by-construction list covers the leak, so
that it flips or moves in exactly one run at every H1 cut barring a declared ``?``: a same-quantity
comparison of bar ``s + 1`` with bar ``s - k``, margin 1, ``k <= 90`` (``k = 0`` for the fill, the
close location and the upper wick); a threshold inside its band on bar ``s + 1``; the colour of bar
``s + 1``, or its sameness with bar ``s - k``; a continuous function, as ``exp_return``, of one
later level, two-column ratio or taker share. ``cell`` marks a colour x size cell of bar ``s + 1``
with a boundary inside the body band, flagged where the realised cells predict. Everything else is
``no``: the measurement proper.

**Honest twins.** Every leak has a twin: the same expression with every future offset moved into
the past, ``s + j`` becoming ``s - j + 1`` (a future window ``s + 1 .. s + m`` becomes
``s - m + 1 .. s``). A twin reads bars ``<= s`` only, so its outputs at rows ``<= t`` cannot move
when the bars after ``t`` do: a twin caught, or differing by a single bit, is a bench bug.

**Evaluation** (:func:`run`). H1's pairs, interval, folds and perturbation, exactly:

- **H1's cuts**: :func:`olsprime.bench.invariance.choose_cuts` with H1's seed, 8 per pair on
  BTCUSDT and ETHUSDT 1d, both directions, with H1's seeds and parities
  (:func:`olsprime.bench.invariance.perturbation_at`);
- **the long schedule**: :data:`LONG_CUTS` = 64 cuts per pair drawn without replacement among the
  scored rows that have a later bar (seed ``long_cut_seed``), perturbed by the same function with
  ``perturbation_seed = long_perturbation_seed``, parity by position in the time-sorted list.

The perturbations are computed once. Each leak is evaluated, vectorized over every cut and
direction at once, on the rows ``t - 10 .. t`` that are scored (decisions of the test months, as in
H1): a row before ``t - 10`` reads only bars ``<= t``, which ``perturb_after`` returns bit for bit.
A leak is **caught** at a cut under H1's rule (:data:`olsprime.bench.invariance.CAUGHT_ABOVE`): an
output moves by more than 1e-9, turns missing or stops being missing, in either direction. A
catch's **size** is the largest move of the leak's own outputs at that cut (0 when it is caught
through a missing value only). The sizes describe the scale of the generator's outputs, O(1) by
construction, not H1's sensitivity.

**Two headline rates.** The share caught at one of H1's cuts at least (Wilson), and the mean
per-cut rate (normal interval over leaks), which is closer to H1's own rule.

**Attenuation curve** (:func:`attenuation_curve`). The share caught when every leak's contribution
is multiplied by ``eps`` in :data:`EPSILONS`: a target becomes ``eps x target``, an ``exp_return``
``eps x exp_return``, and the probabilities move from the leak-free ``(0.4, 0.2, 0.4)`` by ``eps``
times their departure from it. ``eps = 1`` returns the outputs untouched and must reproduce the
unattenuated verdicts exactly. It shows where the 1e-9 rule stops catching weak leaks.

**Checks of the design.** Each missed cut of a ``designed: yes`` leak is matched against H1's
declared ``?`` reasons, read on the original bars by the probe of the same comparison
(:func:`declared_reason`); an unexplained one is listed as a candidate hole. A miss whose output is
the same at every compared row of the original bars is counted apart: at the compared rows it
carries no information. So is a twin constant at the compared rows (``close[s] > close[s]``, the
twin of a next-bar comparison with bar s), which cannot be caught: the report gives their share.

**Never produced** (:data:`NEVER_PRODUCED`, listed in the report): attenuated leaks (measured by
the curve instead), fitted or transductive leaks, reads beyond ``s + 10``, continuous
future-to-future functions, time-gated or sparse leaks, leaks that read the time stamps or whether
a bar exists, cross-pair leaks, and leaks that act only through the long/flat rule's hysteresis.

**Equivalence.** 50 seeded leaks of the population and their twins are wrapped as baselines
(:class:`LeakModel`) and run through H1's own path (:func:`olsprime.bench.invariance.run_model`:
``outputs``, ``compare``, the long/flat rule at every venue). Their caught cuts, per direction, must
equal the fast path's, and the twins must show no violation. The first 5 of them, attenuated by
1e-8 and 1e-10, go through the same check. If any disagree, the run stops: no rate is reported.

Usage::

    uv run --no-sync python -m olsprime.bench.adversary [--out DIR] [--n 10000] [--workers N]

It writes ``adversary.json``, ``adversary.md`` and ``run.json`` to ``runs/<UTC date>-h1-adversary``
(suffixed if taken) or to ``--out``, with H1's provenance block. The result is descriptive: H1's
registered rule does not change.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import expit

from olsprime.bench import invariance as inv
from olsprime.bench import pipeline as pp
from olsprime.bench import provenance
from olsprime.bench.leaks import TAKER_SHARE, THRESHOLD, LeakConfig, Probe
from olsprime.bench.perturb import (
    DOWN_CLV,
    DOWN_FILL,
    DOWN_FLOOR,
    PERTURBED_COLUMNS,
    PUSH,
    SMALL_BODY,
    UP_CAP,
    UP_CLV,
    UP_FILL,
    WINDOW,
    perturb_after,
)
from olsprime.data.binance_vision import Month
from olsprime.eval.splits import Fold
from olsprime.models.baselines import PREDICTION_SCHEMA, check_test_frame
from olsprime.paths import runs_dir

log = logging.getLogger(__name__)

# --- The population ---------------------------------------------------------------------

FAMILIES: tuple[str, ...] = ("levels", "ratios", "shapes", "aggregates", "continuous", "combinations")
FAMILY_WEIGHTS: tuple[float, ...] = (0.20, 0.20, 0.20, 0.15, 0.10, 0.15)
LEVELS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume",
)  # fmt: skip
RATIOS: tuple[str, ...] = (
    "vwap", "taker_vwap", "trade_size", "taker_base_share", "taker_quote_share", "body", "upper_wick", "lower_wick", "range",
)  # fmt: skip
SHARES: tuple[str, ...] = ("taker_base_share", "taker_quote_share")
LOG_RATIOS: tuple[str, ...] = ("body", "upper_wick", "lower_wick", "range")
SCALE_FREE: tuple[str, ...] = (*SHARES, *LOG_RATIOS)
SCALED_RATIOS: tuple[str, ...] = ("vwap", "taker_vwap", "trade_size")
QUANTITY_NAMES: tuple[str, ...] = (*LEVELS, *RATIOS, "fill", "close_location", "wick_ratio", "green", "red")
# Pushed against bars t-90..t by construction (perturb.py): every level, the two-column ratios and shares,
# body, lower wick and range. The upper wick, the fill and the close location only against bar t.
PUSHED_VS_WINDOW = frozenset({*LEVELS, *SCALED_RATIOS, *SHARES, "body", "lower_wick", "range"})
PUSHED_VS_BAR_T = frozenset({"upper_wick", "fill", "close_location"})
# Moves on every later bar by construction: every value and every ratio of two values (check_coverage).
MOVES_EVERY_BAR = frozenset({*LEVELS, *SCALED_RATIOS, *SHARES})

FUTURE_OFFSETS: tuple[int, ...] = (1, 2, 3, 5)
FUTURE_WEIGHTS: tuple[float, ...] = (0.4, 0.2, 0.2, 0.2)
AGG_LENGTHS: tuple[int, ...] = (2, 3, 5, 10)
AGG_OPS: tuple[str, ...] = ("max", "min", "mean", "sum")
TRAIL_WINDOWS: tuple[int, ...] = (5, 20, 90, 180)
MAX_PAST = 180
FUTURE_REACH = 10  # the largest future offset a leak may read
PAST_REACH = 200  # the window kept before each cut: >= FUTURE_REACH + 189, the deepest past read
COMPARED = FUTURE_REACH + 1  # rows t-10 .. t: the only rows ``<= t`` that can read a bar after t
WIDTH = PAST_REACH + FUTURE_REACH + 1

# H1's bands for a threshold on bar t + 1 (perturb.py), [lo, hi); and the draws outside them.
BANDS: dict[str, tuple[float, float]] = {
    "body": (UP_CAP["body"], DOWN_FLOOR["body"]),
    "upper_wick": (UP_CAP["wick"], DOWN_FLOOR["wick"]),
    "lower_wick": (UP_CAP["wick"], DOWN_FLOOR["wick"]),
    "range": (UP_CAP["body"] + 2 * UP_CAP["wick"], DOWN_FLOOR["body"] / DOWN_FILL),
    "fill": (DOWN_FILL, UP_FILL),
    "close_location": (DOWN_CLV, UP_CLV),
}
OUTSIDE: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "body": ((1e-4, BANDS["body"][0]), (BANDS["body"][1], 0.5)),
    "upper_wick": ((2e-5, BANDS["upper_wick"][0]), (BANDS["upper_wick"][1], 0.2)),
    "lower_wick": ((2e-5, BANDS["lower_wick"][0]), (BANDS["lower_wick"][1], 0.2)),
    "range": ((2e-4, BANDS["range"][0]), (BANDS["range"][1], 2.0)),
    "fill": ((0.01, BANDS["fill"][0]), (BANDS["fill"][1], 0.99)),
    "close_location": ((0.01, BANDS["close_location"][0]), (BANDS["close_location"][1], 0.99)),
}
LINEAR_THRESHOLDS = frozenset({"fill", "close_location"})  # drawn uniformly; the log quantities log-uniformly
NEUTRAL = 0.2  # p_neutral of a leak that outputs probabilities

LONG_CUTS = 64
LONG_KIND = "long-random"
OUTPUTS = ("target", "exp_return", "probs")
TAG_KEYS = ("family", "quantity", "operation", "offset", "reference", "position", "output", "designed")


def _s(o: int) -> str:
    return "s" if o == 0 else f"s+{o}" if o > 0 else f"s-{-o}"


def _future_twin(o: int) -> int:
    return 1 - o if o > 0 else o


# --- Quantities and views ------------------------------------------------------------------


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def quantities(bars: pl.DataFrame) -> dict[str, np.ndarray]:
    """Every quantity of :data:`QUANTITY_NAMES` on every row of one pair's bars (time order); NaN where undefined."""
    col = {c: bars[c].cast(pl.Float64).to_numpy() for c in PERTURBED_COLUMNS}
    o, h, lo, c = col["open"], col["high"], col["low"], col["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        body = np.abs(np.log(c / o))
        upper = np.log(h / np.maximum(o, c))
        lower = np.log(np.minimum(o, c) / lo)
        rng = np.log(h / lo)
        q = {
            **col,
            "vwap": _ratio(col["quote_volume"], col["volume"]),
            "taker_vwap": _ratio(col["taker_buy_quote_volume"], col["taker_buy_base_volume"]),
            "trade_size": _ratio(col["volume"], col["trades"]),
            "taker_base_share": _ratio(col["taker_buy_base_volume"], col["volume"]),
            "taker_quote_share": _ratio(col["taker_buy_quote_volume"], col["quote_volume"]),
            "body": body,
            "upper_wick": upper,
            "lower_wick": lower,
            "range": rng,
            "fill": np.where(rng > 0, body / rng, np.nan),
            "close_location": np.where(h > lo, (c - lo) / (h - lo), np.nan),
            "wick_ratio": np.where((upper > 0) & (lower > 0), np.log(upper / lower), np.nan),
            "green": (c > o).astype(np.float64),
            "red": (c < o).astype(np.float64),
        }
    return {k: _clean(q[k]) for k in QUANTITY_NAMES}


def _clean(x: np.ndarray) -> np.ndarray:
    """Float64, every non-finite value a canonical NaN (as a null read back by the bench)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


class SeriesView:
    """``at(q, o)``: quantity ``q`` at row ``s + o`` for every row ``s`` of the series, NaN outside it."""

    def __init__(self, q: dict[str, np.ndarray]) -> None:
        self.q = q
        self.n = len(q["close"])

    def at(self, name: str, o: int) -> np.ndarray:
        a, n = self.q[name], self.n
        out = np.full(n, np.nan)
        if abs(o) < n:
            if o >= 0:
                out[: n - o] = a[o:]
            else:
                out[-o:] = a[: n + o]
        return out


class WindowView:
    """Stacked windows of bars ``t - PAST_REACH .. t + FUTURE_REACH``, one row per (cut, run).

    ``at(q, o)`` gives, for each window, ``q`` at rows ``s + o`` for the compared rows
    ``s = t - FUTURE_REACH .. t``: a view, never a copy.
    """

    def __init__(self, windows: dict[str, np.ndarray]) -> None:
        self.w = windows

    def at(self, name: str, o: int) -> np.ndarray:
        if not -(PAST_REACH - FUTURE_REACH) <= o <= FUTURE_REACH:
            raise ValueError(f"offset {o} is outside the kept window")
        start = PAST_REACH - FUTURE_REACH + o
        return self.w[name][:, start : start + COMPARED]


class RecordingView:
    """A view that records every ``(quantity, offset)`` read: what a leak actually reads."""

    def __init__(self, inner: SeriesView | WindowView) -> None:
        self.inner = inner
        self.reads: set[tuple[str, int]] = set()

    def at(self, name: str, o: int) -> np.ndarray:
        self.reads.add((name, o))
        return self.inner.at(name, o)


# --- Expressions ---------------------------------------------------------------------------
# Value nodes give floats; predicates (``binary = True``) give booleans; scores give floats.


@dataclass(frozen=True)
class At:
    q: str
    o: int
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        return view.at(self.q, self.o)

    def offsets(self) -> frozenset[int]:
        return frozenset({self.o})

    def twin(self) -> At:
        return At(self.q, _future_twin(self.o))

    def render(self) -> str:
        return f"{self.q}[{_s(self.o)}]"


@dataclass(frozen=True)
class Agg:
    """``op`` of ``q`` over rows ``s + lo .. s + hi`` (all future or all past)."""

    q: str
    op: str
    lo: int
    hi: int
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        stack = np.stack([view.at(self.q, o) for o in range(self.lo, self.hi + 1)])
        return {"max": np.max, "min": np.min, "mean": np.mean, "sum": np.sum}[self.op](stack, axis=0)

    def offsets(self) -> frozenset[int]:
        return frozenset(range(self.lo, self.hi + 1))

    def twin(self) -> Agg:
        if self.lo >= 1:
            return Agg(self.q, self.op, 1 - self.hi, 1 - self.lo)
        if self.hi <= 0:
            return self
        raise ValueError("an aggregate window must be all future or all past")

    def render(self) -> str:
        return f"{self.op}({self.q}[{_s(self.lo)}..{_s(self.hi)}])"


@dataclass(frozen=True)
class TrailMean:
    """Mean of ``q`` over rows ``s - w + 1 .. s``: the past only."""

    q: str
    w: int
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        return np.mean(np.stack([view.at(self.q, -o) for o in range(self.w)]), axis=0)

    def offsets(self) -> frozenset[int]:
        return frozenset(range(-self.w + 1, 1))

    def twin(self) -> TrailMean:
        return self

    def render(self) -> str:
        return f"mean({self.q}[{_s(-self.w + 1)}..s])"


@dataclass(frozen=True)
class Const:
    c: float
    binary = False

    def evaluate(self, view: Any) -> float:
        return self.c

    def offsets(self) -> frozenset[int]:
        return frozenset()

    def twin(self) -> Const:
        return self

    def render(self) -> str:
        return f"{self.c:.4g}"


@dataclass(frozen=True)
class Cmp:
    """``lhs op margin * rhs``; false where either side is missing."""

    lhs: Any
    op: str
    margin: float
    rhs: Any
    binary = True

    def evaluate(self, view: Any) -> np.ndarray:
        a, b = self.lhs.evaluate(view), self.rhs.evaluate(view)
        with np.errstate(invalid="ignore"):
            return a > self.margin * b if self.op == ">" else a < self.margin * b

    def offsets(self) -> frozenset[int]:
        return self.lhs.offsets() | self.rhs.offsets()

    def twin(self) -> Cmp:
        return Cmp(self.lhs.twin(), self.op, self.margin, self.rhs.twin())

    def render(self) -> str:
        scale = "" if self.margin == 1.0 else f"{self.margin:.3g} x "
        return f"{self.lhs.render()} {self.op} {scale}{self.rhs.render()}"


@dataclass(frozen=True)
class Same:
    """Bar ``s + o``'s colour equals bar ``s + r``'s."""

    o: int
    r: int
    binary = True

    def evaluate(self, view: Any) -> np.ndarray:
        return view.at("green", self.o) == view.at("green", self.r)

    def offsets(self) -> frozenset[int]:
        return frozenset({self.o, self.r})

    def twin(self) -> Same:
        return Same(_future_twin(self.o), _future_twin(self.r))

    def render(self) -> str:
        return f"colour[{_s(self.o)}] == colour[{_s(self.r)}]"


@dataclass(frozen=True)
class Colour:
    colour: str  # "green" or "red"
    o: int
    binary = True

    def evaluate(self, view: Any) -> np.ndarray:
        return view.at(self.colour, self.o) > 0.5

    def offsets(self) -> frozenset[int]:
        return frozenset({self.o})

    def twin(self) -> Colour:
        return Colour(self.colour, _future_twin(self.o))

    def render(self) -> str:
        return f"{self.colour}[{_s(self.o)}]"


@dataclass(frozen=True)
class Pattern:
    """``hammer``: lower wick >= a x body and upper wick <= body; ``star``: the mirror."""

    kind: str
    o: int
    a: float
    binary = True

    def evaluate(self, view: Any) -> np.ndarray:
        body, up, lo = (view.at(q, self.o) for q in ("body", "upper_wick", "lower_wick"))
        long_, short = (lo, up) if self.kind == "hammer" else (up, lo)
        with np.errstate(invalid="ignore"):
            return (long_ >= self.a * body) & (short <= body)

    def offsets(self) -> frozenset[int]:
        return frozenset({self.o})

    def twin(self) -> Pattern:
        return Pattern(self.kind, _future_twin(self.o), self.a)

    def render(self) -> str:
        long_, short = ("lower", "upper") if self.kind == "hammer" else ("upper", "lower")
        return f"{self.kind}[{_s(self.o)}] ({long_} wick >= {self.a:.2f} x body, {short} wick <= body)"


@dataclass(frozen=True)
class Cell:
    """Bar ``s + o`` is ``colour`` and ``size`` (small: body < c; big: body > c)."""

    colour: str
    size: str
    o: int
    c: float
    binary = True

    def evaluate(self, view: Any) -> np.ndarray:
        body = view.at("body", self.o)
        with np.errstate(invalid="ignore"):
            sized = body < self.c if self.size == "small" else body > self.c
        return (view.at(self.colour, self.o) > 0.5) & sized

    def offsets(self) -> frozenset[int]:
        return frozenset({self.o})

    def twin(self) -> Cell:
        return Cell(self.colour, self.size, _future_twin(self.o), self.c)

    def render(self) -> str:
        return f"{self.colour}-and-{self.size}[{_s(self.o)}] (body {'<' if self.size == 'small' else '>'} {self.c:.4g})"


@dataclass(frozen=True)
class Logic:
    op: str  # AND, OR, XOR
    a: Any
    b: Any
    binary = True

    def evaluate(self, view: Any) -> np.ndarray:
        x, y = self.a.evaluate(view), self.b.evaluate(view)
        return {"AND": np.logical_and, "OR": np.logical_or, "XOR": np.logical_xor}[self.op](x, y)

    def offsets(self) -> frozenset[int]:
        return self.a.offsets() | self.b.offsets()

    def twin(self) -> Logic:
        return Logic(self.op, self.a.twin(), self.b.twin())

    def render(self) -> str:
        return f"({self.a.render()}) {self.op} ({self.b.render()})"


@dataclass(frozen=True)
class LogRatio:
    a: Any
    b: Any
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        x, y = self.a.evaluate(view), self.b.evaluate(view)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((x > 0) & (y > 0), np.log(x / y), np.nan)

    def offsets(self) -> frozenset[int]:
        return self.a.offsets() | self.b.offsets()

    def twin(self) -> LogRatio:
        return LogRatio(self.a.twin(), self.b.twin())

    def render(self) -> str:
        return f"ln({self.a.render()} / {self.b.render()})"


@dataclass(frozen=True)
class Rel:
    a: Any
    b: Any
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        x, y = self.a.evaluate(view), self.b.evaluate(view)
        with np.errstate(divide="ignore", invalid="ignore"):
            return x / y - 1.0

    def offsets(self) -> frozenset[int]:
        return self.a.offsets() | self.b.offsets()

    def twin(self) -> Rel:
        return Rel(self.a.twin(), self.b.twin())

    def render(self) -> str:
        return f"{self.a.render()} / {self.b.render()} - 1"


@dataclass(frozen=True)
class Raw:
    a: Any
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        return self.a.evaluate(view)

    def offsets(self) -> frozenset[int]:
        return self.a.offsets()

    def twin(self) -> Raw:
        return Raw(self.a.twin())

    def render(self) -> str:
        return self.a.render()


@dataclass(frozen=True)
class WSum:
    """``w1 x a + w2 x b``; a binary part counts as 0 or 1."""

    w1: float
    a: Any
    w2: float
    b: Any
    binary = False

    def evaluate(self, view: Any) -> np.ndarray:
        x, y = (np.asarray(p.evaluate(view), dtype=np.float64) for p in (self.a, self.b))
        return self.w1 * x + self.w2 * y

    def offsets(self) -> frozenset[int]:
        return self.a.offsets() | self.b.offsets()

    def twin(self) -> WSum:
        return WSum(self.w1, self.a.twin(), self.w2, self.b.twin())

    def render(self) -> str:
        return f"{self.w1:+.3f} x [{self.a.render()}] {self.w2:+.3f} x [{self.b.render()}]"


@dataclass(frozen=True)
class Leak:
    """!!! A LEAK, ON PURPOSE (or, with ``twin`` set, its honest twin). !!!"""

    id: int
    family: str
    expr: Any
    output: str  # target, exp_return or probs
    tags: tuple[tuple[str, str], ...]
    twin: bool = False
    scale: float = 1.0  # the attenuation epsilon: the leak's contribution to its outputs is multiplied by it

    @property
    def tag(self) -> dict[str, str]:
        return dict(self.tags)

    def honest_twin(self) -> Leak:
        if self.twin:
            raise ValueError("a twin has no twin")
        return replace(self, expr=self.expr.twin(), twin=True)

    def attenuated(self, eps: float) -> Leak:
        """The same leak with its contribution multiplied by ``eps`` (see :meth:`outputs`)."""
        if not 0.0 < eps <= 1.0:
            raise ValueError(f"eps must be in (0, 1], got {eps}")
        return replace(self, scale=float(eps))

    def reach(self) -> tuple[int, int]:
        """The smallest and largest offsets the expression reads."""
        o = self.expr.offsets()
        return min(o), max(o)

    def render(self) -> str:
        scale = "" if self.scale == 1.0 else f"{self.scale:g} x "
        return f"{self.output} = {scale}{self.expr.render()}"

    def outputs(self, view: Any) -> dict[str, np.ndarray]:
        """The leak's outputs on every row of the view: ``target``, ``exp_return`` or ``p_down``/``p_neutral``/``p_up``.

        With ``scale = eps < 1`` the leak's **contribution** is attenuated: its departure from the leak-free output
        (target 0, ``exp_return`` 0, the probabilities at ``z = 0``: ``(0.4, 0.2, 0.4)``) is multiplied by ``eps``.
        At ``eps = 1`` the outputs are returned untouched, bit for bit.
        """
        v = self.expr.evaluate(view)
        x = np.asarray(v, dtype=np.float64) if self.expr.binary else _clean(v)
        eps = self.scale
        if self.output == "target":
            if not self.expr.binary:
                raise ValueError("a target leak must be binary")
            return {"target": x if eps == 1.0 else eps * x}
        if self.output == "exp_return":
            return {"exp_return": x if eps == 1.0 else eps * x}
        z = 2.0 * x - 1.0 if self.expr.binary else x
        missing = np.isnan(z)
        half = (1.0 - NEUTRAL) / 2.0  # p_down = p_up at z = 0
        p_down, p_up = (1.0 - NEUTRAL) * expit(-z), (1.0 - NEUTRAL) * expit(z)
        if eps != 1.0:
            p_down, p_up = half + eps * (p_down - half), half + eps * (p_up - half)
        return {
            "p_down": np.where(missing, np.nan, p_down),
            "p_neutral": np.where(missing, np.nan, NEUTRAL),
            "p_up": np.where(missing, np.nan, p_up),
        }


# --- The generator ---------------------------------------------------------------------------


def _pick(rng: np.random.Generator, seq: Sequence[Any], p: Sequence[float] | None = None) -> Any:
    if p is None:
        return seq[int(rng.integers(len(seq)))]
    return seq[int(rng.choice(len(seq), p=np.asarray(p) / np.sum(p)))]


def _future(rng: np.random.Generator) -> int:
    return int(_pick(rng, FUTURE_OFFSETS, FUTURE_WEIGHTS))


def _past(rng: np.random.Generator) -> int:
    u = rng.random()
    if u < 0.5:
        return int(rng.integers(0, 11))
    if u < 0.75:
        return int(rng.integers(11, WINDOW + 1))
    return int(rng.integers(WINDOW + 1, MAX_PAST + 1))


def _past_label(k: int) -> str:
    if k == 0:
        return "bar s"
    return f"bar s-k, 1<=k<={WINDOW}" if k <= WINDOW else f"bar s-k, {WINDOW}<k<={MAX_PAST}"


def _op(rng: np.random.Generator) -> str:
    return ">" if rng.random() < 0.5 else "<"


def _margin(rng: np.random.Generator) -> float:
    return 1.0 if rng.random() < 0.7 else float(np.exp(rng.uniform(math.log(0.25), math.log(4.0))))


def _margin_position(m: float) -> str:
    return "inside" if 1.0 / PUSH <= m <= PUSH else "outside"


def _threshold(rng: np.random.Generator, q: str) -> tuple[float, str]:
    """A threshold for ``q`` and its position: inside H1's band (probability 0.5), else below or above it."""
    if q in SHARES:
        return float(rng.uniform(0.05, 0.95)), "window-dependent"
    if q == "wick_ratio":
        return float(rng.uniform(-2.0, 2.0)), "none"
    inside = rng.random() < 0.5
    lo, hi = BANDS[q] if inside else OUTSIDE[q][int(rng.random() < 0.5)]
    c = float(rng.uniform(lo, hi)) if q in LINEAR_THRESHOLDS else float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
    return c, _band_position(q, c)


def _band_position(q: str, c: float) -> str:
    lo, hi = BANDS[q]
    return "inside" if lo <= c < hi else "outside"


def _binary_output(rng: np.random.Generator) -> str:
    return "target" if rng.random() < 0.8 else "probs"


def _tags(**kw: Any) -> dict[str, str]:
    return {k: str(v) for k, v in kw.items()}


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _draw_levels(rng: np.random.Generator) -> tuple[Any, dict[str, str]]:
    q, j, op, m = _pick(rng, LEVELS), _future(rng), _op(rng), _margin(rng)
    if rng.random() < 0.75:
        k = _past(rng)
        expr = Cmp(At(q, j), op, m, At(q, -k))
        return expr, _tags(quantity=q, operation="vs-past", offset=_s(j), reference=_past_label(k), position=_margin_position(m),
                           designed=_yes(j == 1 and m == 1.0 and k <= WINDOW))  # fmt: skip
    j2 = int(_pick(rng, tuple(o for o in FUTURE_OFFSETS if o != j)))
    return Cmp(At(q, j), op, m, At(q, j2)), _tags(
        quantity=q, operation="vs-future", offset=_s(max(j, j2)), reference="another future bar", position=_margin_position(m), designed="no"
    )


def _designed_vs_past(q: str, j: int, k: int, m: float) -> bool:
    if j != 1 or m != 1.0:
        return False
    if q in PUSHED_VS_BAR_T:
        return k == 0
    return q in PUSHED_VS_WINDOW and k <= WINDOW


def _draw_ratios(rng: np.random.Generator) -> tuple[Any, dict[str, str]]:
    q, j, op = _pick(rng, RATIOS), _future(rng), _op(rng)
    if rng.random() < 0.55 or q == "trade_size":
        k, m = _past(rng), _margin(rng)
        return Cmp(At(q, j), op, m, At(q, -k)), _tags(
            quantity=q, operation="vs-past", offset=_s(j), reference=_past_label(k), position=_margin_position(m),
            designed=_yes(_designed_vs_past(q, j, k, m)),
        )  # fmt: skip
    if q in SCALED_RATIOS:  # a price-like ratio against a past close
        k, m = _past(rng), _margin(rng)
        return Cmp(At(q, j), op, m, At("close", -k)), _tags(
            quantity=q, operation="vs-past-close", offset=_s(j), reference=_past_label(k), position=_margin_position(m), designed="no"
        )
    c, pos = _threshold(rng, q)
    return Cmp(At(q, j), op, 1.0, Const(c)), _tags(
        quantity=q, operation="threshold", offset=_s(j), reference="threshold", position=pos, designed=_yes(j == 1 and pos == "inside")
    )


SHAPE_KINDS = ("fill-threshold", "clv-threshold", "fill-vs-past", "clv-vs-past", "colour", "same-colour",
               "wick-ratio-threshold", "wick-ratio-vs-past", "hammer", "star", "cell")  # fmt: skip


def _draw_shapes(rng: np.random.Generator) -> tuple[Any, dict[str, str]]:
    kind, j = _pick(rng, SHAPE_KINDS), _future(rng)
    if kind in ("fill-threshold", "clv-threshold", "wick-ratio-threshold"):
        q = {"fill-threshold": "fill", "clv-threshold": "close_location", "wick-ratio-threshold": "wick_ratio"}[kind]
        c, pos = _threshold(rng, q)
        return Cmp(At(q, j), _op(rng), 1.0, Const(c)), _tags(
            quantity=q, operation="threshold", offset=_s(j), reference="threshold", position=pos,
            designed=_yes(j == 1 and pos == "inside" and q != "wick_ratio"),
        )  # fmt: skip
    if kind in ("fill-vs-past", "clv-vs-past", "wick-ratio-vs-past"):
        q = {"fill-vs-past": "fill", "clv-vs-past": "close_location", "wick-ratio-vs-past": "wick_ratio"}[kind]
        k = 0 if rng.random() < 0.5 else _past(rng)
        m = 1.0 if q == "wick_ratio" else _margin(rng)
        return Cmp(At(q, j), _op(rng), m, At(q, -k)), _tags(
            quantity=q, operation="vs-past", offset=_s(j), reference=_past_label(k), position=_margin_position(m),
            designed=_yes(q != "wick_ratio" and _designed_vs_past(q, j, k, m)),
        )  # fmt: skip
    if kind == "colour":
        colour = _pick(rng, ("green", "red"))
        return Colour(colour, j), _tags(quantity="colour", operation=colour, offset=_s(j), reference="none", position="none",
                                        designed=_yes(j == 1))  # fmt: skip
    if kind == "same-colour":
        k = _past(rng)
        return Same(j, -k), _tags(quantity="colour", operation="same-colour", offset=_s(j), reference=_past_label(k), position="none",
                                  designed=_yes(j == 1))  # fmt: skip
    if kind in ("hammer", "star"):
        a = float(rng.uniform(1.0, 3.0))
        return Pattern(kind, j, a), _tags(quantity="wicks and body", operation=kind, offset=_s(j), reference="none", position="none",
                                          designed="no")  # fmt: skip
    colour, size = _pick(rng, ("green", "red")), _pick(rng, ("small", "big"))
    if rng.random() < 0.5:
        c, pos = SMALL_BODY, _band_position("body", SMALL_BODY)
    else:
        c, pos = _threshold(rng, "body")
    return Cell(colour, size, j, c), _tags(
        quantity="colour and body", operation=f"{colour}-and-{size}", offset=_s(j), reference="threshold", position=pos,
        designed="cell" if j == 1 and pos == "inside" else "no",
    )  # fmt: skip


def _draw_aggregates(rng: np.random.Generator) -> tuple[Any, dict[str, str]]:
    q, op, m, k, cmp, margin = _pick(rng, (*LEVELS, *RATIOS)), _pick(rng, AGG_OPS), int(_pick(rng, AGG_LENGTHS)), _past(rng), _op(rng), _margin(rng)
    lhs = Agg(q, op, 1, m)
    if rng.random() < 0.5:
        rhs, ref = Agg(q, op, -k - m + 1, -k), f"past window ending at {_past_label(k)}"
    else:
        rhs, ref = At(q, -k), _past_label(k)
    return Cmp(lhs, cmp, margin, rhs), _tags(
        quantity=q, operation=op, offset=f"s+1..s+{m}", reference=ref, position=_margin_position(margin), designed="no"
    )


CONTINUOUS_KINDS = ("log-ratio", "over-trailing-mean", "value", "aggregate-log-ratio")


def _draw_continuous(rng: np.random.Generator) -> tuple[Any, dict[str, str]]:
    kind, j = _pick(rng, CONTINUOUS_KINDS), _future(rng)
    if kind == "log-ratio":
        q, k = _pick(rng, (*LEVELS, *SCALED_RATIOS)), _past(rng)
        return LogRatio(At(q, j), At(q, -k)), _tags(quantity=q, operation=kind, offset=_s(j), reference=_past_label(k),
                                                    position="none", designed=_yes(q in MOVES_EVERY_BAR))  # fmt: skip
    if kind == "over-trailing-mean":
        q, w = _pick(rng, (*LEVELS, *RATIOS)), int(_pick(rng, TRAIL_WINDOWS))
        return Rel(At(q, j), TrailMean(q, w)), _tags(quantity=q, operation=kind, offset=_s(j), reference=f"trailing mean of {w}",
                                                     position="none", designed=_yes(q in MOVES_EVERY_BAR))  # fmt: skip
    if kind == "value":
        q = _pick(rng, (*SCALE_FREE, "fill", "close_location", "wick_ratio"))
        return Raw(At(q, j)), _tags(quantity=q, operation=kind, offset=_s(j), reference="none", position="none",
                                    designed=_yes(q in MOVES_EVERY_BAR))  # fmt: skip
    q, op, m = _pick(rng, ("high", "low", "close", "vwap")), _pick(rng, ("max", "min", "mean")), int(_pick(rng, AGG_LENGTHS))
    return LogRatio(Agg(q, op, 1, m), At("close", 0)), _tags(quantity=q, operation=f"{kind} ({op})", offset=f"s+1..s+{m}",
                                                            reference="bar s", position="none", designed="no")  # fmt: skip


DRAW: dict[str, Callable[[np.random.Generator], tuple[Any, dict[str, str]]]] = {
    "levels": _draw_levels,
    "ratios": _draw_ratios,
    "shapes": _draw_shapes,
    "aggregates": _draw_aggregates,
    "continuous": _draw_continuous,
}
COMBINATION_OPS = ("AND", "OR", "XOR", "WSUM")


def _draw_combination(rng: np.random.Generator) -> tuple[Any, dict[str, str], str]:
    op = _pick(rng, COMBINATION_OPS)
    pool = FAMILIES[:4] if op != "WSUM" else FAMILIES[:5]
    weights = FAMILY_WEIGHTS[: len(pool)]
    parts = []
    for _ in range(2):
        fam = _pick(rng, pool, weights)
        parts.append((fam, *DRAW[fam](rng)))
    (fa, a, _), (fb, b, _) = parts
    offsets = a.offsets() | b.offsets()
    tags = _tags(quantity="two leaks", operation=op, offset=_s(max(offsets)), reference=f"{fa} + {fb}", position="combination",
                 designed="no")  # fmt: skip
    if op == "WSUM":
        w = [float(rng.uniform(0.25, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0) for _ in range(2)]
        return WSum(w[0], a, w[1], b), tags, "exp_return"
    return Logic(op, a, b), tags, _binary_output(rng)


def draw_leak(seed: int, i: int) -> Leak:
    """Leak ``i`` of the population of ``seed``: drawn from ``default_rng([seed, i])`` alone."""
    rng = np.random.default_rng([seed, i])
    family = _pick(rng, FAMILIES, FAMILY_WEIGHTS)
    if family == "combinations":
        expr, tags, output = _draw_combination(rng)
    else:
        expr, tags = DRAW[family](rng)
        if family == "continuous":
            output = "exp_return" if rng.random() < 0.8 else "probs"
            if output == "probs":
                tags["designed"] = "no"
        else:
            output = _binary_output(rng)
    tags = {"family": family, **tags, "output": output}
    leak = Leak(i, family, expr, output, tuple((k, tags[k]) for k in TAG_KEYS))
    lo, hi = leak.reach()
    if not (1 <= hi <= FUTURE_REACH and lo >= -(PAST_REACH - FUTURE_REACH)):
        raise AssertionError(f"leak {i} reads offsets {lo}..{hi}, outside the generator's reach")
    return leak


def generate(n: int, seed: int) -> list[Leak]:
    """The population: leaks ``0 .. n - 1`` of ``seed`` (see the module docstring)."""
    return [draw_leak(seed, i) for i in range(n)]


# --- The full H1 path: a leak as a baseline --------------------------------------------------


class LeakModel:
    """!!! LEAKS: decides at t from the bars it is built with, through a :class:`Leak`. !!!"""

    def __init__(self, bars: pl.DataFrame, leak: Leak) -> None:
        self.name = f"adversary-{leak.id}{'-twin' if leak.twin else ''}"
        self.config = LeakConfig(leaks=leak.render())
        b = bars.sort("open_time")
        self._row = {t: i for i, t in enumerate(b["open_time"].to_list())}
        self._out = leak.outputs(SeriesView(quantities(b)))

    def fit(self, train: pl.DataFrame) -> None:
        pass

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        idx = np.array([self._row[t] for t in test["open_time"].to_list()], dtype=np.int64)
        nulls = pl.Series([None] * test.height, dtype=pl.Float64)

        def column(name: str) -> pl.Series:
            if name not in self._out:
                return nulls
            return pl.Series(self._out[name][idx], dtype=PREDICTION_SCHEMA[name], nan_to_null=True)

        return pl.DataFrame(
            {
                "symbol": test["symbol"],
                "decision_time": test["decision_time"],
                **{c: column(c) for c in ("p_down", "p_neutral", "p_up", "exp_return", "target")},
            }
        )


def leak_spec(leak: Leak) -> inv.ModelSpec:
    """A leak (or its twin) as an H1 model under test."""
    name = f"adversary-{leak.id}{'-twin' if leak.twin else ''}"
    return inv.ModelSpec(name, lambda bars, leak=leak: (lambda: LeakModel(bars, leak)), expect=inv.PASS if leak.twin else inv.FLAG_EVERY_CUT)


def full_path(leak: Leak, bars: pl.DataFrame, cuts: Sequence[inv.Cut], cfg: inv.H1Config) -> dict[str, Any]:
    """H1's own path for one leak on one pair: caught and violated, per cut and direction."""
    records = inv.run_model(leak_spec(leak), bars, cuts, cfg)
    dirs = [inv._direction(s) for s in cfg.level_shifts]
    return {
        "caught": [[r["caught_by_direction"][d] > 0 for d in dirs] for r in records],
        "violated": [[r["by_direction"][d] > 0 for d in dirs] for r in records],
    }


def _full_path_task(
    pair: str, bars: pl.DataFrame, cuts: list[inv.Cut], cfg: inv.H1Config, leaks: list[Leak]
) -> list[tuple[str, int, float, bool, dict[str, Any]]]:
    return [(pair, leak.id, leak.scale, leak.twin, full_path(leak, bars, cuts, cfg)) for leak in leaks]


# --- The fast path: every perturbation once, every leak vectorized ---------------------------


@dataclass(frozen=True)
class AdversaryConfig:
    n: int = 10_000
    seed: int = 20261029
    h1: inv.H1Config = field(default_factory=inv.H1Config)
    long_cuts: int = LONG_CUTS
    long_cut_seed: int = 20261030
    long_perturbation_seed: int = 11
    equivalence_leaks: int = 50
    equivalence_seed: int = 3
    # The first of the equivalence leaks, attenuated, through H1's own path too: the 1e-9 rule in both paths.
    attenuated_equivalence_leaks: int = 5
    attenuated_equivalence_eps: tuple[float, ...] = (1e-8, 1e-10)
    miss_sample: int = 40
    miss_sample_seed: int = 5


def scored_rows(folds: Sequence[Fold]) -> np.ndarray:
    """The rows H1 compares: the decisions of the test months."""
    return np.unique(np.concatenate([f.test_idx for f in folds]))


def long_schedule(bars: pl.DataFrame, folds: Sequence[Fold], *, seed: int, n: int) -> list[inv.Cut]:
    """``n`` cuts drawn without replacement among the scored rows that have a later bar, in time order."""
    rows = scored_rows(folds)
    rows = rows[rows < bars.height - 1]
    if rows.size < n:
        raise ValueError(f"{rows.size} scored rows for {n} cuts")
    pick = np.sort(np.random.default_rng(seed).choice(rows, size=n, replace=False))
    fit_ms = {int(f.fit_time.timestamp() * 1000) for f in folds}
    open_ms = bars["open_time"].dt.epoch("ms").to_numpy()
    out = []
    for i in pick.tolist():
        here, nxt = bars["open_time"][i], bars["open_time"][i + 1]
        out.append(
            inv.Cut(i, here, bars["known_at"][i], LONG_KIND, opens_fold=int(open_ms[i]) in fit_ms,
                    month_continues=(here.year, here.month) == (nxt.year, nxt.month))  # fmt: skip
        )
    return out


def _window(q: dict[str, np.ndarray], cut: int) -> dict[str, np.ndarray]:
    """Rows ``cut - PAST_REACH .. cut + FUTURE_REACH`` of every quantity, NaN outside the series."""
    n = len(q["close"])
    lo, hi = cut - PAST_REACH, cut + FUTURE_REACH + 1
    out = {}
    for name, a in q.items():
        w = np.full(WIDTH, np.nan)
        src_lo, src_hi = max(lo, 0), min(hi, n)
        w[src_lo - lo : src_hi - lo] = a[src_lo:src_hi]
        out[name] = w
    return out


@dataclass
class Bench:
    """Every (pair, schedule, cut) once: original windows, then one block of windows per direction."""

    view: WindowView
    cuts: list[dict[str, Any]]  # pair, schedule, k, index, open_time, kind, mirror
    scored: np.ndarray  # (cuts, COMPARED): the compared rows t-10..t that H1 scores
    directions: tuple[str, ...]
    h1_cuts: dict[str, list[inv.Cut]]  # per pair, for the full path

    @property
    def n_cuts(self) -> int:
        return len(self.cuts)

    def rows(self, schedule: str) -> np.ndarray:
        return np.array([i for i, c in enumerate(self.cuts) if c["schedule"] == schedule], dtype=np.int64)


def schedules(bars: pl.DataFrame, folds: Sequence[Fold], cfg: AdversaryConfig) -> list[tuple[str, list[inv.Cut], inv.H1Config]]:
    """H1's cuts with H1's configuration, and the long schedule with its own perturbation seed."""
    h1_cuts = inv.choose_cuts(bars, folds, seed=cfg.h1.cut_seed)
    long_ = long_schedule(bars, folds, seed=cfg.long_cut_seed, n=cfg.long_cuts)
    return [("h1", h1_cuts, cfg.h1), ("long", long_, replace(cfg.h1, perturbation_seed=cfg.long_perturbation_seed))]


def build_bench(bars_by_pair: dict[str, pl.DataFrame], cfg: AdversaryConfig) -> Bench:
    """Compute every perturbation once and keep, per cut and run, the window of quantities a leak can read."""
    h1 = cfg.h1
    blocks: list[list[dict[str, np.ndarray]]] = [[] for _ in range(1 + len(h1.level_shifts))]
    cuts, masks, h1_cuts = [], [], {}
    for pair, bars in bars_by_pair.items():
        frame = pp.build_frame(bars, h1.interval, h1.h)
        folds = pp.make_folds(frame, h1.interval, h1.h, Month(*h1.start), Month(*h1.end))
        scored = np.zeros(bars.height, dtype=bool)
        scored[scored_rows(folds)] = True
        q0 = quantities(bars)
        for schedule, cut_list, hcfg in schedules(bars, folds, cfg):
            if schedule == "h1":
                h1_cuts[pair] = cut_list
            for k, cut in enumerate(cut_list):
                blocks[0].append(_window(q0, cut.index))
                for j in range(len(h1.level_shifts)):
                    seed, config = inv.perturbation_at(hcfg, k, j)
                    blocks[1 + j].append(_window(quantities(perturb_after(bars, cut.index, seed=seed, config=config)), cut.index))
                rows = cut.index - FUTURE_REACH + np.arange(COMPARED)
                masks.append(np.where(rows >= 0, scored[np.clip(rows, 0, None)], False))
                cuts.append({"pair": pair, "schedule": schedule, "k": k, "index": cut.index, "open_time": cut.open_time.isoformat(),
                             "kind": cut.kind, "mirror": inv.is_mirrored(k)})  # fmt: skip
    stacked = {name: np.stack([w[name] for block in blocks for w in block]) for name in QUANTITY_NAMES}
    return Bench(WindowView(stacked), cuts, np.stack(masks), tuple(inv._direction(s) for s in h1.level_shifts), h1_cuts)


@dataclass
class FastResult:
    caught: np.ndarray  # (cuts, directions) bool: H1's catch rule
    violated: np.ndarray  # (cuts, directions) bool: a single differing bit
    size: np.ndarray  # (cuts,) the largest move of the leak's outputs at the cut, either direction
    constant: bool = False  # the same output, bit for bit, at every compared row of the original bars


def evaluate_fast(leak: Leak, bench: Bench) -> FastResult:
    """The leak on every cut and direction at once: H1's catch rule on the compared rows that H1 scores."""
    c, d = bench.n_cuts, len(bench.directions)
    caught = np.zeros((c, d), dtype=bool)
    violated = np.zeros((c, d), dtype=bool)
    size = np.zeros(c)
    constant = True
    for arr in leak.outputs(bench.view).values():
        arr = np.broadcast_to(np.asarray(arr, dtype=np.float64), (c * (1 + d), COMPARED))
        a = arr[:c]
        bits = np.ascontiguousarray(a[bench.scored]).view(np.uint64)
        constant = constant and bool(bits.size == 0 or np.all(bits == bits[0]))
        for j in range(d):
            b = arr[c * (1 + j) : c * (2 + j)]
            both = ~np.isnan(a) & ~np.isnan(b)
            change = np.where(both, np.abs(a - b), 0.0)
            moved = ((change > inv.CAUGHT_ABOVE) | (np.isnan(a) != np.isnan(b))) & bench.scored
            caught[:, j] |= moved.any(axis=1)
            violated[:, j] |= ((a.view(np.uint64) != b.view(np.uint64)) & bench.scored).any(axis=1)
            size = np.maximum(size, np.where(bench.scored, change, 0.0).max(axis=1))
    return FastResult(caught, violated, size, constant)


# --- Statistics -------------------------------------------------------------------------------

Z95 = 1.959963984540054  # the 97.5 % quantile of the standard normal


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``k`` successes in ``n`` trials (95 % by default)."""
    if n <= 0:
        return (math.nan, math.nan)
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _share(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"caught": int(k), "n": int(n), "share": (k / n) if n else math.nan, "wilson95": [lo, hi]}


def _mean_rate(rates: np.ndarray) -> dict[str, Any]:
    """Mean of per-leak rates with a normal 95 % interval over leaks (the leaks are independent draws)."""
    n = rates.size
    if n == 0:
        return {"n_leaks": 0, "mean": math.nan, "ci95": [math.nan, math.nan]}
    m = float(rates.mean())
    se = float(rates.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {"n_leaks": int(n), "mean": m, "ci95": [max(0.0, m - Z95 * se), min(1.0, m + Z95 * se)]}


SIZE_BINS = ((inv.CAUGHT_ABOVE, 1e-8), (1e-8, 1e-6), (1e-6, 1e-4), (1e-4, 1e-2), (1e-2, 1.0), (1.0, math.inf))


def _size_summary(sizes: np.ndarray) -> dict[str, Any]:
    """Distribution of catch sizes; a size of 0 is a catch through a missing value only."""
    sizes = np.asarray(sizes, dtype=np.float64)
    present = sizes[sizes > 0]
    bins = {f"({lo:g}, {hi:g}]": int(((present > lo) & (present <= hi)).sum()) for lo, hi in SIZE_BINS}
    q = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 1.0]
    return {
        "catches": int(sizes.size),
        "missing_value_only": int((sizes == 0).sum()),
        "within_10x_of_caught_above": int(((sizes > 0) & (sizes <= 10 * inv.CAUGHT_ABOVE)).sum()),
        "share_within_10x": float(((sizes > 0) & (sizes <= 10 * inv.CAUGHT_ABOVE)).mean()) if sizes.size else math.nan,
        "quantiles": {f"{p:g}": float(np.quantile(present, p)) for p in q} if present.size else {},
        "bins": bins,
    }


# --- The run ------------------------------------------------------------------------------------


@dataclass
class Evaluated:
    caught: np.ndarray  # (leaks, cuts, directions)
    size: np.ndarray  # (leaks, cuts)
    twin_caught: np.ndarray  # (leaks, cuts, directions)
    twin_violated: np.ndarray  # (leaks, cuts, directions)
    constant: np.ndarray  # (leaks,): the same output at every compared row of the original bars
    twin_constant: np.ndarray  # (leaks,): the twin gives the same output at every compared row (it cannot be caught)


def evaluate_all(leaks: Sequence[Leak], bench: Bench) -> Evaluated:
    n, c, d = len(leaks), bench.n_cuts, len(bench.directions)
    out = Evaluated(np.zeros((n, c, d), bool), np.zeros((n, c)), np.zeros((n, c, d), bool), np.zeros((n, c, d), bool),
                    np.zeros(n, bool), np.zeros(n, bool))  # fmt: skip
    for i, leak in enumerate(leaks):
        r = evaluate_fast(leak, bench)
        out.caught[i], out.size[i], out.constant[i] = r.caught, r.size, r.constant
        t = evaluate_fast(leak.honest_twin(), bench)
        out.twin_caught[i], out.twin_violated[i], out.twin_constant[i] = t.caught, t.violated, t.constant
    return out


EPSILONS: tuple[float, ...] = (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)


def attenuation_curve(leaks: Sequence[Leak], bench: Bench, ev: Evaluated, epsilons: Sequence[float] = EPSILONS) -> list[dict[str, Any]]:
    """The share caught when every leak's contribution is multiplied by eps (:meth:`Leak.outputs`).

    At any of H1's cuts (Wilson) and per cut (mean over leaks, at H1's cuts and over the long schedule),
    overall and per output. ``eps = 1`` must reproduce the unattenuated verdicts exactly, or this raises.
    """
    h1_rows, long_rows = bench.rows("h1"), bench.rows("long")
    output = np.array([leak.output for leak in leaks])
    rows = []
    for eps in epsilons:
        caught = np.stack([evaluate_fast(leak.attenuated(eps), bench).caught for leak in leaks]) if leaks else ev.caught
        if eps == 1.0 and not np.array_equal(caught, ev.caught):
            raise AssertionError("eps = 1 does not reproduce the unattenuated run")
        cut = caught.any(axis=2)
        any_h1 = cut[:, h1_rows].any(axis=1)
        rows.append({
            "eps": float(eps),
            "any_h1_cut": _share(int(any_h1.sum()), len(leaks)),
            "per_cut_rate_h1": _mean_rate(cut[:, h1_rows].mean(axis=1)),
            "per_cut_rate_long": _mean_rate(cut[:, long_rows].mean(axis=1)) if long_rows.size else _mean_rate(np.array([])),
            "any_h1_cut_by_output": {o: _share(int(any_h1[output == o].sum()), int((output == o).sum())) for o in OUTPUTS if (output == o).any()},
            "reproduces_unattenuated": bool(eps != 1.0 or np.array_equal(caught, ev.caught)),
        })  # fmt: skip
    return rows


# The adversary's quantities that H1's probes know (olsprime.bench.leaks.QUANTITIES): their declared ``?`` reasons apply.
PROBE_QUANTITY: dict[str, str] = {
    "close": "close", "trades": "trades", "quote_volume": "quote_volume", "taker_quote_share": TAKER_SHARE,
    "taker_base_share": "taker_base_share", "range": "range", "body": "body", "vwap": "vwap", "taker_vwap": "taker_vwap",
    "trade_size": "trade_size", "fill": "fill", "close_location": "close_location", "upper_wick": "upper_wick",
    "lower_wick": "lower_wick",
}  # fmt: skip


def declared_reason(leak: Leak, bars: pl.DataFrame, cut: int, mirror: bool) -> str | None:
    """H1's declared ``?`` for a designed comparison of bar s + 1 at this cut, through the probe of the same comparison.

    Read on the original bars, as H1 does (:meth:`olsprime.bench.leaks.Probe.unknown_reason`). None when no
    declared reason applies, or when H1 has no probe for the quantity.
    """
    e = leak.expr
    if not (isinstance(e, Cmp) and isinstance(e.lhs, At) and e.lhs.o == 1 and e.lhs.q in PROBE_QUANTITY):
        return None
    q = PROBE_QUANTITY[e.lhs.q]
    if isinstance(e.rhs, At) and e.rhs.q == e.lhs.q and e.rhs.o <= 0:
        return Probe("adversary", q, back=-e.rhs.o).unknown_reason(bars, cut, mirror)
    if isinstance(e.rhs, Const):
        return Probe("adversary", q, THRESHOLD, threshold=e.rhs.c).unknown_reason(bars, cut, mirror)
    return None


def designed_misses(leaks: Sequence[Leak], bench: Bench, ev: Evaluated, bars_by_pair: dict[str, pl.DataFrame], *, limit: int = 50) -> dict[str, Any]:
    """Every cut where a ``designed: yes`` leak is not caught, and whether one of H1's declared ``?`` reasons explains it.

    An unexplained one is a candidate hole in H1's design: the report lists it.
    """
    rows = []
    for i, leak in enumerate(leaks):
        if leak.tag["designed"] != "yes":
            continue
        for r in np.flatnonzero(~ev.caught[i].any(axis=1)).tolist():
            c = bench.cuts[r]
            reason = declared_reason(leak, bars_by_pair[c["pair"]], c["index"], c["mirror"])
            rows.append({"leak": leak.id, "expression": leak.render(), "pair": c["pair"], "schedule": c["schedule"], "k": c["k"],
                         "cut": c["index"], "open_time": c["open_time"], "reason": reason})  # fmt: skip
    explained = [r for r in rows if r["reason"] is not None]
    unexplained = [r for r in rows if r["reason"] is None]
    return {
        "designed_leaks": int(sum(leak.tag["designed"] == "yes" for leak in leaks)),
        "cuts_missed": len(rows),
        "explained_by_declared_reason": len(explained),
        "reasons": dict(Counter(r["reason"].split(" (")[0] for r in explained)),
        "unexplained_count": len(unexplained),
        "unexplained": unexplained[:limit],
    }


def equivalence_ids(n: int, cfg: AdversaryConfig) -> list[int]:
    k = min(cfg.equivalence_leaks, n)
    return sorted(np.random.default_rng(cfg.equivalence_seed).choice(n, size=k, replace=False).tolist())


def check_equivalence(
    leaks: Sequence[Leak],
    bench: Bench,
    bars_by_pair: dict[str, pl.DataFrame],
    cfg: AdversaryConfig,
    *,
    workers: int = 1,
    twins: bool = True,
) -> dict[str, Any]:
    """The fast path against H1's own path, cut for cut and direction for direction, on the leaks (and their twins).

    Leaks are told apart by ``(id, scale)``, so the same leak may be checked at several attenuations.
    """
    units = [unit for leak in leaks for unit in ((leak, leak.honest_twin()) if twins else (leak,))]
    jobs = []
    chunk = max(1, math.ceil(len(units) / max(1, 3 * workers)))
    for pair, bars in bars_by_pair.items():
        for s in range(0, len(units), chunk):
            jobs.append((pair, bars, bench.h1_cuts[pair], cfg.h1, units[s : s + chunk]))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_full_path_task, *job) for job in jobs]
            results = [r for f in futures for r in f.result()]
    else:
        results = [r for job in jobs for r in _full_path_task(*job)]
    by_leak = {(leak.id, leak.scale): leak for leak in leaks}
    mismatches, twin_violations, compared = [], [], 0
    for pair, leak_id, scale, twin, full in sorted(results, key=lambda r: (r[0], r[1], r[2], r[3])):
        rows = np.array([i for i, c in enumerate(bench.cuts) if c["pair"] == pair and c["schedule"] == "h1"])
        leak = by_leak[(leak_id, scale)].honest_twin() if twin else by_leak[(leak_id, scale)]
        fast = evaluate_fast(leak, bench)
        compared += 1
        f_caught, f_violated = fast.caught[rows].tolist(), fast.violated[rows].tolist()
        if full["caught"] != f_caught:
            mismatches.append({"pair": pair, "leak": leak_id, "twin": twin, "expression": leak.render(),
                               "full_caught": full["caught"], "fast_caught": f_caught})  # fmt: skip
        if twin and (any(any(v) for v in full["violated"]) or any(any(v) for v in f_violated)):
            twin_violations.append({"pair": pair, "leak": leak_id, "expression": leak.render(),
                                    "full_violated": full["violated"], "fast_violated": f_violated})  # fmt: skip
    return {
        "leaks": [leak.id for leak in leaks],
        "scales": sorted({leak.scale for leak in leaks}, reverse=True),
        "units_compared": compared,
        "cuts_per_pair": {p: len(c) for p, c in bench.h1_cuts.items()},
        "mismatches": mismatches,
        "twin_violations": twin_violations,
        "agree": not mismatches and not twin_violations,
    }


def _tag_table(leaks: Sequence[Leak], missed: np.ndarray, key: str) -> list[dict[str, Any]]:
    total, miss = Counter(), Counter()
    for leak, m in zip(leaks, missed, strict=True):
        v = leak.tag[key]
        total[v] += 1
        miss[v] += int(m)
    rows = [{"value": v, "missed": miss[v], "n": total[v], "miss_share": miss[v] / total[v]} for v in total]
    return sorted(rows, key=lambda r: (-r["missed"], r["value"]))


def summarize(
    leaks: Sequence[Leak], bench: Bench, ev: Evaluated, cfg: AdversaryConfig, bars_by_pair: dict[str, pl.DataFrame] | None = None
) -> dict[str, Any]:
    """Every number of the report, from the fast path's verdicts (and the designed misses, given the bars)."""
    h1_rows, long_rows = bench.rows("h1"), bench.rows("long")
    caught_cut = ev.caught.any(axis=2)  # (leaks, cuts): either direction, as H1 flags a cut
    h1_any = caught_cut[:, h1_rows].any(axis=1)
    h1_all = caught_cut[:, h1_rows].all(axis=1)
    long_rate = caught_cut[:, long_rows].mean(axis=1) if long_rows.size else np.zeros(len(leaks))
    long_any = caught_cut[:, long_rows].any(axis=1) if long_rows.size else np.zeros(len(leaks), bool)
    fam = np.array([leak.family for leak in leaks])

    def by(values: np.ndarray, keys: Sequence[str], fn: Callable[[np.ndarray], Any]) -> dict[str, Any]:
        return {k: fn(values == k) for k in keys if (values == k).any()}

    designed = np.array([leak.tag["designed"] for leak in leaks])
    output = np.array([leak.output for leak in leaks])
    h1 = {
        "overall": _share(int(h1_any.sum()), len(leaks)),
        "by_family": by(fam, FAMILIES, lambda m: _share(int(h1_any[m].sum()), int(m.sum()))),
        "by_designed": by(designed, ("yes", "cell", "no"), lambda m: _share(int(h1_any[m].sum()), int(m.sum()))),
        "by_output": by(output, OUTPUTS, lambda m: _share(int(h1_any[m].sum()), int(m.sum()))),
        "caught_at_every_cut": _share(int(h1_all.sum()), len(leaks)),
        "per_cut_rate": _mean_rate(caught_cut[:, h1_rows].mean(axis=1)),
        "per_cut": [
            {**bench.cuts[r], "share": float(caught_cut[:, r].mean()), "wilson95": list(wilson(int(caught_cut[:, r].sum()), len(leaks))),
             **{f"share_{d}": float(ev.caught[:, r, j].mean()) for j, d in enumerate(bench.directions)}}
            for r in h1_rows
        ],  # fmt: skip
    }
    long_ = {
        "cuts": int(long_rows.size),
        "per_cut_rate": _mean_rate(long_rate),
        "per_cut_rate_by_family": by(fam, FAMILIES, lambda m: _mean_rate(long_rate[m])),
        "per_cut_rate_by_designed": by(designed, ("yes", "cell", "no"), lambda m: _mean_rate(long_rate[m])),
        "caught_at_one_cut_at_least": _share(int(long_any.sum()), len(leaks)),
        "caught_at_one_cut_at_least_by_family": by(fam, FAMILIES, lambda m: _share(int(long_any[m].sum()), int(m.sum()))),
        "across_cuts": (lambda s: {"min": float(s.min()), "median": float(np.median(s)), "max": float(s.max())})(caught_cut[:, long_rows].mean(axis=0))
        if long_rows.size
        else {},
    }
    sizes_h1 = ev.size[:, h1_rows][caught_cut[:, h1_rows]]
    sizes_long = ev.size[:, long_rows][caught_cut[:, long_rows]]
    smallest = np.array([ev.size[i, h1_rows][caught_cut[i, h1_rows]].min() for i in np.flatnonzero(h1_any)])
    sizes = {
        "h1_cuts": _size_summary(sizes_h1),
        "long_schedule": _size_summary(sizes_long),
        "smallest_catch_per_caught_leak_h1": _size_summary(smallest),
    }
    twin_constant = int(ev.twin_constant.sum())
    twins = {
        "evaluated": int(len(leaks)),
        "cuts": int(bench.n_cuts),
        "false_catches": int(ev.twin_caught.any(axis=(1, 2)).sum()),
        "bit_violations": int(ev.twin_violated.any(axis=(1, 2)).sum()),
        # A twin constant at the compared rows cannot be caught: "0 caught" says nothing about it.
        "constant_at_compared_rows": twin_constant,
        "constant_share": _share(twin_constant, len(leaks)),
        "non_constant": int(len(leaks) - twin_constant),
        "non_constant_caught": int((ev.twin_caught.any(axis=(1, 2)) & ~ev.twin_constant).sum()),
    }
    missed = ~h1_any
    rng = np.random.default_rng(cfg.miss_sample_seed)
    miss_ids = np.flatnonzero(missed)
    sample = sorted(rng.choice(miss_ids, size=min(cfg.miss_sample, miss_ids.size), replace=False).tolist()) if miss_ids.size else []
    misses = {
        "count": int(missed.sum()),
        "by_tag": {key: _tag_table(leaks, missed, key) for key in TAG_KEYS},
        "long_schedule_also_missed": int((missed & ~long_any).sum()),
        "constant_on_original_rows": int((missed & ev.constant).sum()),
        "sample": [
            {"id": leaks[i].id, "family": leaks[i].family, "expression": leaks[i].render(), "twin": leaks[i].honest_twin().render(),
             "tags": leaks[i].tag, "long_cuts_caught": int(caught_cut[i, long_rows].sum()), "constant_on_original_rows": bool(ev.constant[i])}
            for i in sample
        ],  # fmt: skip
    }
    per_leak = [
        {
            "id": leak.id,
            "family": leak.family,
            "expression": leak.render(),
            "tags": leak.tag,
            "h1_pattern": "".join("X" if x else "." for x in caught_cut[i, h1_rows]),
            "long_cuts_caught": int(caught_cut[i, long_rows].sum()),
            "smallest_catch_h1": float(ev.size[i, h1_rows][caught_cut[i, h1_rows]].min()) if h1_any[i] else None,
            "constant_on_original_rows": bool(ev.constant[i]),
        }
        for i, leak in enumerate(leaks)
    ]
    population = {
        "n": len(leaks),
        "by_family": dict(Counter(leak.family for leak in leaks)),
        "by_output": dict(Counter(leak.output for leak in leaks)),
        "by_designed": dict(Counter(leak.tag["designed"] for leak in leaks)),
    }
    out = {"population": population, "h1_cuts": h1, "long_schedule": long_, "catch_sizes": sizes, "twins": twins,
           "misses": misses, "never_produced": list(NEVER_PRODUCED), "leaks": per_leak}  # fmt: skip
    if bars_by_pair is not None:
        out["designed_misses"] = designed_misses(leaks, bench, ev, bars_by_pair)
    return out


# What this generator never produces (docs/PROTOCOL.md, "The adversary"): the share caught says nothing about these.
NEVER_PRODUCED: tuple[str, ...] = (
    "attenuated leaks, whose contribution to the outputs is small: measured by the attenuation curve, not by the population",
    "fitted or transductive leaks (scalers, hyperparameters, unpurged fits): only H1's designed controls cover them",
    f"reads beyond s+{FUTURE_REACH} (FUTURE_REACH): the hourly label's exit at s+25 is out of reach",
    "continuous future-to-future functions, such as the h = 1 label ln(open[s+2] / open[s+1]) or a future volatility",
    "time-gated or sparse leaks, active at one row per month, or under conditions beyond the 1/4..4 margins",
    "leaks that read the time stamps or whether a bar exists (H1's stated limit)",
    "cross-pair leaks: every leak reads its own pair's bars only",
    "leaks that act only through the long/flat rule's hysteresis",
)


def results_row(result: dict[str, Any]) -> str:
    """The text of the adversary's row in ``docs/RESULTS.md``: both headline rates, the twins and the attenuation."""
    o, pc = result["h1_cuts"]["overall"], result["h1_cuts"]["per_cut_rate"]
    tw = result["twins"]
    curve = "; ".join(f"eps {r['eps']:g}: {_pct(r['any_h1_cut']['share'])} any cut, {_pct(r['per_cut_rate_h1']['mean'])} per cut"
                      for r in result.get("attenuation", []))  # fmt: skip
    return (
        f"{o['n']} random leaks (seed {result['config']['seed']}) through H1's perturbation at H1's 16 cuts. Caught at one cut at "
        f"least: {_pct(o['share'])}, 95 % Wilson {_ci(o)}. Mean per-cut rate (closer to H1's per-cut rule): {_pct(pc['mean'])}, 95 % {_ci(pc)} "
        f"(normal, over leaks). Twins caught: {tw['false_catches']} of {tw['evaluated']}, of which {tw['constant_at_compared_rows']} "
        f"({_pct(tw['constant_share']['share'])}) are constant at the compared rows and cannot be caught. Attenuation: {curve}. "
        "Descriptive: this generator's detection rate, not all leaks'."
    )


def _config_json(cfg: AdversaryConfig) -> dict[str, Any]:
    out = asdict(cfg)
    out["families"] = dict(zip(FAMILIES, FAMILY_WEIGHTS, strict=True))
    out["future_offsets"] = dict(zip(FUTURE_OFFSETS, FUTURE_WEIGHTS, strict=True))
    out["bands"] = BANDS
    out["outside_bands"] = OUTSIDE
    out["caught_above"] = inv.CAUGHT_ABOVE
    out["future_reach"], out["past_reach"] = FUTURE_REACH, PAST_REACH
    return out


def run(bars_by_pair: dict[str, pl.DataFrame], cfg: AdversaryConfig, *, workers: int = 1, prov: dict[str, Any] | None = None) -> dict[str, Any]:
    """The adversary on the given bars: equivalence first, then every leak and twin. No rate if the paths disagree."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    leaks = generate(cfg.n, cfg.seed)
    bench = build_bench(bars_by_pair, cfg)
    timings["perturbations"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    ids = equivalence_ids(cfg.n, cfg)
    eq = check_equivalence([leaks[i] for i in ids], bench, bars_by_pair, cfg, workers=workers)
    weak = [leaks[i].attenuated(e) for i in ids[: cfg.attenuated_equivalence_leaks] for e in cfg.attenuated_equivalence_eps]
    if weak:  # the 1e-9 rule, near its edge, in both paths
        eq["attenuated"] = check_equivalence(weak, bench, bars_by_pair, cfg, workers=workers, twins=False)
        eq["agree"] = bool(eq["agree"] and eq["attenuated"]["agree"])
        eq["mismatches"] = [*eq["mismatches"], *eq["attenuated"]["mismatches"]]
    timings["equivalence"] = time.perf_counter() - t1
    result: dict[str, Any] = {"config": _config_json(cfg), "cuts": bench.cuts, "equivalence": eq}
    if prov is not None:
        result["provenance"] = prov
    if not eq["agree"]:
        result["stopped"] = "the fast path disagrees with H1's own path: no rate is reported"
        result["_timings"] = timings
        return result
    t2 = time.perf_counter()
    ev = evaluate_all(leaks, bench)
    timings["fast_path"] = time.perf_counter() - t2
    result.update(summarize(leaks, bench, ev, cfg, bars_by_pair))
    t3 = time.perf_counter()
    result["attenuation"] = attenuation_curve(leaks, bench, ev)
    timings["attenuation"] = time.perf_counter() - t3
    result["results_row"] = results_row(result)
    result["_timings"] = timings
    return result


def ok(result: dict[str, Any]) -> bool:
    """The run is sound: the paths agree and no twin moved by a single bit."""
    twins = result.get("twins", {})
    return bool(result["equivalence"]["agree"] and "stopped" not in result and twins.get("false_catches") == 0 and twins.get("bit_violations") == 0)


# --- Report -----------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f} %"


def _ci(s: dict[str, Any]) -> str:
    lo, hi = s.get("wilson95", s.get("ci95"))
    return f"[{_pct(lo)}, {_pct(hi)}]"


def render_md(result: dict[str, Any]) -> str:
    cfg = result["config"]
    eq = result["equivalence"]
    directions = [inv._direction(s) for s in cfg["h1"]["level_shifts"]]
    lines = [
        "# H1's adversary: random leaks through H1's perturbation",
        "",
        "Descriptive: this measures the detection rate of one seeded generator of leaks (module docstring of "
        "`olsprime.bench.adversary`), not of all leaks. H1's registered rule does not change.",
        "",
        f"Population: {cfg['n']} leaks, seed {cfg['seed']}. Pairs {', '.join(cfg['h1']['pairs'])} {cfg['h1']['interval']}, "
        f"test months {cfg['h1']['start'][0]}-{cfg['h1']['start'][1]:02d} -> {cfg['h1']['end'][0]}-{cfg['h1']['end'][1]:02d}. "
        f"A leak is caught at a cut where one of its outputs moves by more than {cfg['caught_above']:g}, turns missing or "
        "stops being missing, at a scored row decided at or before the cut, in either direction (x3, /3).",
        "",
        "## Equivalence with H1's own path",
        "",
        f"{len(eq['leaks'])} seeded leaks and their twins, run through `invariance.run_model` at H1's cuts "
        f"({', '.join(f'{p}: {n}' for p, n in eq['cuts_per_pair'].items())}): {eq['units_compared']} (leak or twin, pair) "
        f"units compared cut for cut and direction for direction. Mismatches: {len(eq['mismatches'])}. Twin violations: "
        f"{len(eq['twin_violations'])}. **{'The paths agree.' if eq['agree'] else 'THE PATHS DISAGREE: the run stopped.'}**",
        "",
    ]
    if "attenuated" in eq:
        at = eq["attenuated"]
        lines += [
            f"Near the rule's edge: {len(set(at['leaks']))} of these leaks attenuated by eps in "
            f"{{{', '.join(f'{e:g}' for e in at['scales'])}}}, {at['units_compared']} (leak, pair) units through the same path: "
            f"{len(at['mismatches'])} mismatches.",
            "",
        ]
    if "stopped" in result:
        lines += [f"**Stopped:** {result['stopped']}.", ""]
        for m in eq["mismatches"][:20]:
            lines.append(f"- {m['pair']} leak {m['leak']}{' (twin)' if m['twin'] else ''}: `{m['expression']}`")
        if "provenance" in result:
            lines += ["", *provenance.render_md(result["provenance"])]
        return "\n".join(lines) + "\n"
    h1, lg, sizes, twins, misses = result["h1_cuts"], result["long_schedule"], result["catch_sizes"], result["twins"], result["misses"]
    o = h1["overall"]
    lines += [
        "## Honest twins",
        "",
        f"{twins['evaluated']} twins at {twins['cuts']} cuts each (both schedules), both directions: **{twins['false_catches']} "
        f"caught, {twins['bit_violations']} differing by a single bit** (each must be 0; one is a bench bug). "
        f"{twins['constant_at_compared_rows']} twins ({_pct(twins['constant_share']['share'])}, Wilson {_ci(twins['constant_share'])}) "
        "give the same output at every compared row of the original bars, such as `close[s] > close[s]`, the twin of a "
        "next-bar comparison with bar s: a constant twin cannot be caught, so the zero rests on the "
        f"{twins['non_constant']} others, of which {twins['non_constant_caught']} are caught.",
        "",
    ]
    dm = result.get("designed_misses")
    if dm is not None:
        lines += [
            "## Designed leaks",
            "",
            f"{dm['designed_leaks']} leaks are covered by the protocol's by-construction list (`designed: yes`). They are missed "
            f"at {dm['cuts_missed']} (leak, cut) pairs over both schedules; {dm['explained_by_declared_reason']} of them are cuts "
            "where H1 itself declares a `?`, read on the original bars by the probe of the same comparison ("
            + (", ".join(f"{k}: {v}" for k, v in dm["reasons"].items()) or "none")
            + f"). **Unexplained: {dm['unexplained_count']}** (each is a candidate hole in H1's design).",
            "",
        ]
        lines += [f"- leak {u['leak']} at {u['pair']} {u['schedule']} cut {u['k'] + 1} (row {u['cut']}, {u['open_time'][:10]}): "
                  f"`{u['expression']}`" for u in dm["unexplained"]]  # fmt: skip
        if dm["unexplained"]:
            lines.append("")
    lines += [
        "## Share caught at H1's cuts",
        "",
        "Two headline rates. **Any cut:** a leak caught at one of H1's "
        f"{len(h1['per_cut'])} cuts at least, {o['caught']} of {o['n']}, **{_pct(o['share'])}**, 95 % Wilson {_ci(o)}. This is "
        "weaker than H1's own rule, which asks a control to be caught at each of its predicted cuts. **Per cut:** the share of "
        f"H1's cuts at which a leak is caught, averaged over leaks, **{_pct(h1['per_cut_rate']['mean'])}**, 95 % "
        f"{_ci(h1['per_cut_rate'])} (normal, over leaks; a Wilson interval does not apply to a mean of rates). Caught at every "
        f"cut: {_pct(h1['caught_at_every_cut']['share'])} {_ci(h1['caught_at_every_cut'])}. The table below gives the any-cut share.",
        "",
        "| group | caught | n | share | 95 % Wilson |",
        "|---|---|---|---|---|",
    ]
    for title, table in (("family", h1["by_family"]), ("designed", h1["by_designed"]), ("output", h1["by_output"])):
        for k, s in table.items():
            lines.append(f"| {title}: {k} | {s['caught']} | {s['n']} | {_pct(s['share'])} | {_ci(s)} |")
    lines += [
        "",
        "`designed`: `yes` where the protocol's by-construction list covers the leak (it must flip or move at every cut, "
        "barring a declared `?`), `cell` for a colour x size cell of bar s+1 (flagged where the realised cells predict), "
        "`no` for the rest, which is the measurement proper.",
        "",
        "Per cut (share of leaks caught at that cut):",
        "",
        "| pair | # | row | date | kind | parity | either | 95 % Wilson | " + " | ".join(directions) + " |",
        "|---|---|---|---|---|---|---|---|" + "---|" * len(directions),
    ]
    for c in h1["per_cut"]:
        lines.append(
            f"| {c['pair']} | {c['k'] + 1} | {c['index']} | {c['open_time'][:10]} | {c['kind']} | {'mirror' if c['mirror'] else 'plain'} | "
            f"{_pct(c['share'])} | {_ci(c)} | " + " | ".join(_pct(c[f"share_{d}"]) for d in directions) + " |"
        )
    r = lg["per_cut_rate"]
    lines += [
        "",
        f"## Per-cut rate over the long schedule ({lg['cuts']} cuts, both pairs)",
        "",
        "The share of cuts at which a leak is caught, averaged over leaks, with a normal 95 % interval over leaks (the leaks are "
        "independent draws; the cuts of one leak are not, so a Wilson interval over (leak, cut) pairs would be too narrow). "
        f"**Overall "
        f"{_pct(r['mean'])} {_ci(r)}.** Across cuts, the share of leaks caught ranges from {_pct(lg['across_cuts'].get('min'))} "
        f"to {_pct(lg['across_cuts'].get('max'))} (median {_pct(lg['across_cuts'].get('median'))}). Caught at one long cut at "
        f"least: {_pct(lg['caught_at_one_cut_at_least']['share'])} {_ci(lg['caught_at_one_cut_at_least'])}.",
        "",
        "| group | per-cut rate | 95 % interval | caught at one cut at least |",
        "|---|---|---|---|",
    ]
    for k, s in lg["per_cut_rate_by_family"].items():
        lines.append(f"| family: {k} | {_pct(s['mean'])} | {_ci(s)} | {_pct(lg['caught_at_one_cut_at_least_by_family'][k]['share'])} |")
    for k, s in lg["per_cut_rate_by_designed"].items():
        lines.append(f"| designed: {k} | {_pct(s['mean'])} | {_ci(s)} | |")
    lines += [
        "",
        "## Size of the catches",
        "",
        "The size of a catch is the largest move of the leak's own outputs at a caught cut (a target moves by 1). A size of 0 is a "
        f"catch through a missing value only. Within 10x of the rule: at most {10 * cfg['caught_above']:g}. **These sizes describe "
        "the scale of the generator's outputs, which are O(1) by construction (targets in {0, 1}, log-ratios, probabilities), "
        "not H1's sensitivity**: a share within 10x of 0 is not a margin of the bench. The attenuation curve below measures "
        "where the rule stops catching weak leaks.",
        "",
        "| set | catches | missing only | within 10x | min | 1 % | 5 % | median | max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in sizes.items():
        q = s["quantiles"]
        lines.append(
            f"| {name} | {s['catches']} | {s['missing_value_only']} | {s['within_10x_of_caught_above']} ({_pct(s['share_within_10x'])}) | "
            + " | ".join(f"{q[p]:.3g}" if p in q else "" for p in ("0", "0.01", "0.05", "0.5", "1"))
            + " |"
        )
    lines += ["", "Bins of the sizes at H1's cuts: " + ", ".join(f"{k}: {v}" for k, v in sizes["h1_cuts"]["bins"].items()) + ".", ""]
    curve = result.get("attenuation", [])
    if curve:
        outputs = [o_ for o_ in OUTPUTS if o_ in curve[0]["any_h1_cut_by_output"]]
        lines += [
            "## Attenuation curve",
            "",
            "Every leak's contribution to its outputs multiplied by eps: a target becomes eps x target, an `exp_return` eps x "
            "exp_return, and the probabilities move from the leak-free (0.4, 0.2, 0.4) by eps times their departure from it. "
            "Same perturbations, same rule. eps = 1 reproduces the unattenuated run exactly"
            f"{'' if all(r['reproduces_unattenuated'] for r in curve) else ' (IT DOES NOT: a bench bug)'}.",
            "",
            "| eps | any H1 cut | 95 % Wilson | per cut, H1's cuts | 95 % | per cut, long schedule | " + " | ".join(f"any cut, {o_}" for o_ in outputs) + " |",
            "|---|---|---|---|---|---|" + "---|" * len(outputs),
        ]
        for r in curve:
            lines.append(
                f"| {r['eps']:g} | {_pct(r['any_h1_cut']['share'])} | {_ci(r['any_h1_cut'])} | {_pct(r['per_cut_rate_h1']['mean'])} | "
                f"{_ci(r['per_cut_rate_h1'])} | {_pct(r['per_cut_rate_long']['mean'])} | "
                + " | ".join(_pct(r["any_h1_cut_by_output"][o_]["share"]) for o_ in outputs) + " |"
            )
        lines.append("")
    lines += [
        "## The misses",
        "",
        f"{misses['count']} leaks are caught at none of H1's cuts; {misses['long_schedule_also_missed']} of them at none of the "
        f"long schedule's cuts either. {misses['constant_on_original_rows']} of the misses give the same output at every "
        "compared row of the original bars: at the compared rows they carry no information, whatever they read. "
        "Counts by tag (missed of n):",
        "",
    ]
    for key, rows in misses["by_tag"].items():
        cells = "; ".join(f"{r['value']}: {r['missed']}/{r['n']}" for r in rows if r["missed"]) or "none"
        lines.append(f"- **{key}**: {cells}.")
    lines += ["", "A seeded sample of the missed leaks, with their twins:", ""]
    for m in misses["sample"]:
        constant = ", constant at the compared rows" if m["constant_on_original_rows"] else ""
        lines.append(f"- #{m['id']} ({m['family']}, long cuts caught {m['long_cuts_caught']}{constant}): `{m['expression']}`; twin `{m['twin']}`")
    lines += [
        "",
        "## What the generator never produces",
        "",
        "The shares above say nothing about these families: the population was fixed before the smoke run and is not extended.",
        "",
        *(f"- {item}." for item in result.get("never_produced", NEVER_PRODUCED)),
    ]
    if "results_row" in result:
        lines += ["", "## Row for `docs/RESULTS.md`", "", result["results_row"]]
    if "provenance" in result:
        lines += ["", *provenance.render_md(result["provenance"])]
    return "\n".join(lines) + "\n"


def _default_out() -> Path:
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-h1-adversary"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def _load_development(pair: str, cfg: inv.H1Config) -> pl.DataFrame:
    return inv.development_bars(inv._load(pair, cfg))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.adversary", description="H1's adversary: random leaks through H1's perturbation.")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <runs>/<UTC date>-h1-adversary, suffixed if taken)")
    parser.add_argument("--n", type=int, default=AdversaryConfig.n, help="number of leaks (default 10000)")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes for the equivalence check (results do not depend on it)")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    cfg = AdversaryConfig(n=args.n)
    prov = provenance.record(provenance.current_manifest())  # before any data is read, as in H1
    bars = {pair: _load_development(pair, cfg.h1) for pair in cfg.h1.pairs}
    result = run(bars, cfg, workers=args.workers, prov=prov)
    timings = result.pop("_timings")
    out = args.out or _default_out()
    out.mkdir(parents=True, exist_ok=True)
    (out / "adversary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "adversary.md").write_text(render_md(result), encoding="utf-8")
    seconds = {k: round(v, 1) for k, v in timings.items()}
    (out / "run.json").write_text(
        json.dumps({"finished_at": datetime.now(UTC).isoformat(), "seconds": round(time.perf_counter() - started, 1),
                    "stages": seconds, "workers": args.workers, "n": args.n}, indent=2) + "\n",  # fmt: skip
        encoding="utf-8",
    )
    sound = ok(result)
    if "stopped" in result:
        print(f"adversary STOPPED: {result['stopped']}; written to {out}")
    else:
        o = result["h1_cuts"]["overall"]
        print(
            f"adversary: {o['caught']}/{o['n']} caught at H1's cuts ({100 * o['share']:.1f} %, Wilson "
            f"[{100 * o['wilson95'][0]:.1f}, {100 * o['wilson95'][1]:.1f}] %); twins caught {result['twins']['false_catches']}, "
            f"bit violations {result['twins']['bit_violations']}; equivalence {'agrees' if result['equivalence']['agree'] else 'DISAGREES'}; "
            f"written to {out}"
        )
    return 0 if sound else 1


if __name__ == "__main__":
    raise SystemExit(main())
