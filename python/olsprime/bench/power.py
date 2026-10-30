"""The size and power of the registered pass rule (``docs/PROTOCOL.md``, "Size and power of the pass rule").

Usage::

    uv run --no-sync python -m olsprime.bench.power --series <run>/series.npz --sha256 <digest> [--out DIR] [--workers N]

It describes; the pass rule does not change, and nothing is refitted. It reads the hourly log-loss
series of a Report n°2 run from its ``series.npz``, refusing a file whose SHA-256 is not the one given
or not the one the run's ``report2.json`` records.

- **The rule, as registered.** For each comparison and each replicate series x: Diebold-Mariano
  (two-sided, rectangular kernel, h - 1 = 23 lags, Harvey-Leybourne-Newbold) and the stationary
  bootstrap of the mean with the Politis-White length recomputed on x, 2,000 resamples, 95 %
  percentile interval. The bootstrap's seed is the comparison's registered seed
  (``report2._seed(cfg, "diff", a, b)``) on every replicate, so each replicate draws the indices the
  registered rule would draw at its block length. Superiority (a must be lower) passes as
  ``report2.passes``; H12's non-inferiority holds as ``report2.h12_rule_met`` (its condition on fits
  is not a property of a series and is left out).
- **Replicates.** d = ll_a - ll_b over the T scored hours, centred: c = d - mean(d). For a block
  length L, a replicate rotates the hours by a random offset in [0, L), cuts them into consecutive
  circular blocks of L hours (the last one shorter when L does not divide T) and multiplies each
  block by an independent random sign. The flips are drawn from the registered seed (``20261030``),
  the comparison and L, so a comparison's replicates do not depend on the others.
- **Size.** 2,000 replicates per L of flip(c), or flip(c) + m for H12, whose margin m is
  0.005 x climatology's log-loss: the rejection rates of DM alone, the bootstrap alone, both on the
  same side (the pass rule, two-sided) and the pass rule in its registered direction, each with a
  95 % Wilson interval. Every (comparison, L) whose two-sided size exceeds 7.5 % is named.
- **Power, model-free.** flip(c) + delta, delta in the registered direction (a better), on a grid of
  shares of climatology's log-loss, 1,000 replicates per point; the smallest grid delta with power at
  least 80 %, and the value interpolated between grid points. A constant shift is the easiest gain
  to see, so that delta is a lower bound of the detectable mean gain. For H12, the rate at which it
  holds when the true difference is 0. Common random numbers: every delta reads the first 1,000
  flips of the size replicates, so a curve's points differ by delta alone. Block lengths 24, 48, 168
  and 720 hours (audit pass 4a added 48).
- **Power, concentrated** (audit pass 4a). At L = 168 only, on the same flips: the whole shift sits on
  a random 10 % of the 720-hour circular blocks (rotated by their own random offset), and nothing
  elsewhere, scaled so that the mean shift is exactly delta (about 10 delta on a chosen hour). Its
  curve and detectable delta stand beside the constant one.
- **Conditional on the seed.** Every replicate's bootstrap draws from the registered seed, as the rule
  would: the Wilson intervals describe the rule with that seed (common random numbers across replicates).
- **A diagnostic.** At the size point, the variance ratio: the replicates' mean squared error over
  their mean squared DM standard error, and the DM size it implies. A replicate keeps only the T / L
  block sums of the series, so at large L the size is conditional on those few sums.
- **Before the study**, the rule is recomputed on the observed differences by ``report2.compare_arms``
  and compared with ``report2.json``, number by number; a difference is recorded, not refused, since
  the digest already ties the series to its run.
- **Speed.** The resampled mean is linear in the series: for a fixed T, block length and seed the
  index matrix is fixed, and the 2,000 resampled means of many replicates are one product of the
  count matrix (2,000 x T) with the replicates (T x n). Replicates are grouped by their recomputed
  block length, across L and design points, and run through that product in worker processes, each
  with one BLAS thread (``--workers 1`` runs in-process, as the tests must). The intervals equal those
  of the direct path ``x[idx].mean(axis=1)`` up to the last bits of the means.

Outputs, to ``runs/<UTC date>-power/`` (suffixed if taken) or ``--out``: ``power.json`` and
``power.md`` (byte-identical on re-runs), ``config.json`` (the configuration, the series' digest,
the Report n°2 run it came from and the provenance) and ``run.json`` (wall clock only). JSON is
written with sorted keys.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile
import zlib
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as st

from olsprime.bench import provenance as prov_mod
from olsprime.bench import report2 as r2
from olsprime.bench.adversary import wilson
from olsprime.eval import stats
from olsprime.paths import runs_dir

POWER_FILE = "power.md"
DM_ALPHA = 0.05  # the registered DM threshold, as in report2.passes


@dataclass(frozen=True)
class PowerConfig:
    """The study's registered design (``docs/PROTOCOL.md``, "Size and power of the pass rule")."""

    flip_seed: int = 20261030
    block_lengths: tuple[int, ...] = (24, 48, 168, 720)  # hours: one day, two (audit pass 4a), one week, thirty days
    size_replicates: int = 2000
    power_replicates: int = 1000  # the first of the size flips, for every delta (common random numbers)
    deltas: tuple[float, ...] = (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01)  # shares of climatology's log-loss
    size_alert: float = 0.075
    power_target: float = 0.8
    # the concentrated grid (audit pass 4a): at this L only, the shift sits on a share of the span-hour blocks
    concentrated_block: int | None = 168  # None: no concentrated grid (small tests)
    concentrated_span: int = 720
    concentrated_share: float = 0.1

    @property
    def registered(self) -> bool:
        return self == PowerConfig()

    def check(self) -> None:
        if not 0 < self.power_replicates <= self.size_replicates:
            raise ValueError("power_replicates must lie in (0, size_replicates]: power reads the size flips")
        if any(L < 1 for L in self.block_lengths) or any(d <= 0 for d in self.deltas) or list(self.deltas) != sorted(self.deltas):
            raise ValueError("block lengths must be positive, and deltas positive and increasing")
        if self.concentrated_block is not None and (self.concentrated_block not in self.block_lengths or self.concentrated_span < 1
                                                    or not 0.0 < self.concentrated_share <= 1.0):  # fmt: skip
            raise ValueError("the concentrated grid runs at one of the block lengths, on a share in (0, 1] of the span-hour blocks")


@dataclass(frozen=True)
class Rule:
    """The registered pass rule's parameters, read from the Report n°2 run's ``config.json``."""

    n_boot: int = 2000
    level: float = 0.95
    dm_h: int = 24
    bootstrap_seed: int = 20261029

    @classmethod
    def from_report2(cls, config: dict[str, Any]) -> Rule:
        c = config["config"]
        return cls(int(c["n_boot"]), float(c["level"]), int(c["dm_h"]), int(c["bootstrap_seed"]))

    def report2_config(self) -> r2.Report2Config:
        return r2.Report2Config(n_boot=self.n_boot, level=self.level, dm_h=self.dm_h, bootstrap_seed=self.bootstrap_seed)

    def seed(self, a: str, b: str) -> int:
        """The registered bootstrap seed of the comparison a - b over the whole period."""
        return r2._seed(self.report2_config(), "diff", a, b)


@dataclass(frozen=True)
class Comparison:
    """A registered comparison of mean hourly log-loss, difference a - b.

    ``margin`` None: superiority, a must be lower (``report2.passes``). A float: non-inferiority with
    that margin as a share of climatology's log-loss (``report2.h12_rule_met``).
    """

    hypothesis: str
    a: str
    b: str
    margin: float | None = None

    @property
    def kind(self) -> str:
        return "superiority" if self.margin is None else "non-inferiority"


def report2_comparisons(h12_margin: float = 0.005) -> tuple[Comparison, ...]:
    """Report n°2's five comparisons (``report2.COMPARISONS``), H12 as non-inferiority."""
    return tuple(Comparison(h, a, b, h12_margin if h == "H12" else None) for h, a, b in r2.COMPARISONS)


# --- Reading the series by its digest ---------------------------------------------------------


class SeriesRefused(ValueError):
    """The series file is not the one the caller and its run name."""


SERIES_KEYS_READ = ("arms", "hours", "count", "log_loss")


def load_series(path: Path, sha256: str, keys: Sequence[str] = SERIES_KEYS_READ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], str]:
    """The hourly series of a Report n°2 run, read only if the file's SHA-256 is ``sha256`` and the run's ``report2.json`` says so.

    Returns the arrays of ``keys`` (by default those the study reads), the run's ``report2.json`` and
    ``config.json``, and the SHA-256 of that ``report2.json``. The arrays are read from the very bytes
    that were hashed, without pickles. This is the one reader of ``series.npz`` in the package.
    """
    keys = tuple(dict.fromkeys((*SERIES_KEYS_READ, *keys)))
    path = Path(path)
    want = sha256.strip().lower()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != want:
        raise SeriesRefused(f"{path}: SHA-256 {digest}, not the {want} given")
    report_path, config_path = path.parent / "report2.json", path.parent / "config.json"
    if not report_path.is_file() or not config_path.is_file():
        raise SeriesRefused(f"{path.parent}: no report2.json and config.json beside the series, so its run cannot vouch for it")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    rec = report.get("series")
    if not isinstance(rec, dict) or rec.get("file") != path.name or rec.get("sha256") != digest:
        raise SeriesRefused(f"{path}: its run's report2.json records {rec!r}, not this file's SHA-256 {digest}")
    config = json.loads(config_path.read_bytes())
    # the archive is opened here, from the hashed bytes, and each .npy member read on its own: no pickles, no inner archive
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = {name.removesuffix(".npy"): name for name in zf.namelist() if name.endswith(".npy")}
        missing = [k for k in keys if k not in members]
        if missing:
            raise SeriesRefused(f"{path}: no {', '.join(missing)}")
        arrays = {}
        for k in keys:
            with zf.open(members[k]) as member:
                arrays[k] = np.load(member, allow_pickle=False)
    a, t = arrays["arms"].shape[0], arrays["hours"].shape[0]
    if arrays["log_loss"].shape != (a, t) or arrays["count"].shape != (t,):
        raise SeriesRefused(f"{path}: log_loss {arrays['log_loss'].shape} and count {arrays['count'].shape} do not match {a} arms x {t} hours")
    return arrays, report, config, hashlib.sha256(report_bytes).hexdigest()


# --- Replicates -------------------------------------------------------------------------------


def flip_plan(cfg: PowerConfig, comp: Comparison, length: int, t: int) -> tuple[np.ndarray, np.ndarray]:
    """The offsets (n,) and block signs (n, ceil(t / L)) of a comparison's replicates at block length L."""
    ss = np.random.SeedSequence([cfg.flip_seed, zlib.crc32(f"{comp.a}|{comp.b}".encode()), int(length)])
    rng = np.random.default_rng(ss)
    n = cfg.size_replicates
    offsets = rng.integers(0, length, size=n)
    signs = rng.integers(0, 2, size=(n, -(-t // length))).astype(np.float64) * 2.0 - 1.0
    return offsets, signs


def block_index(t: int, length: int, offset: int) -> np.ndarray:
    """Each hour's block: the hours rotated by ``offset`` and cut into consecutive circular blocks of ``length``."""
    return ((np.arange(t) - int(offset)) % t) // int(length)


def flipped(c: np.ndarray, length: int, offset: int, signs: np.ndarray) -> np.ndarray:
    return signs[block_index(c.size, length, offset)] * c


def concentration_plan(cfg: PowerConfig, comp: Comparison, t: int) -> tuple[np.ndarray, np.ndarray]:
    """The offsets (n,) and chosen span-hour blocks (n, k) of a comparison's concentrated replicates, n = power_replicates.

    The hours are rotated by a random offset in [0, span) and cut into ceil(t / span) circular blocks, as for the flips;
    exactly k = round(share x blocks) of them (``chosen_blocks``) are drawn without replacement. Its own stream: the
    flips' seed sequence with a fourth word, the span.
    """
    assert cfg.concentrated_block is not None
    span = int(cfg.concentrated_span)
    ss = np.random.SeedSequence([cfg.flip_seed, zlib.crc32(f"{comp.a}|{comp.b}".encode()), int(cfg.concentrated_block), span])
    rng = np.random.default_rng(ss)
    n, (blocks, k) = cfg.power_replicates, chosen_blocks(cfg, t)
    offsets = rng.integers(0, span, size=n)
    chosen = np.sort(np.argsort(rng.random((n, blocks)), axis=1, kind="stable")[:, :k], axis=1)
    return offsets, chosen


def chosen_blocks(cfg: PowerConfig, t: int) -> tuple[int, int]:
    """The number of span-hour circular blocks of t hours, ceil(t / span), and exactly round(share x blocks) of them to shift.

    Python's round (half to even); for Report n°2's 50,626 hours, 71 blocks and round(7.1) = 7. A series too short for one
    chosen block is refused, not given a different share.
    """
    blocks = -(-t // int(cfg.concentrated_span))
    k = round(cfg.concentrated_share * blocks)
    if k < 1:
        raise ValueError(f"{t} hours make {blocks} blocks of {cfg.concentrated_span}: {cfg.concentrated_share:.0%} of them rounds to none")
    return blocks, k


def concentration(t: int, span: int, offset: int, chosen: np.ndarray) -> np.ndarray:
    """The weights (t,) of a concentrated shift: equal on the hours of the chosen blocks, 0 elsewhere, with mean exactly 1.

    With every chosen block whole, the weight is t / (k x span), about 1 / share (10 for 10 %); a short last block, if
    chosen, raises it so that the mean shift stays delta.
    """
    on = np.isin(block_index(t, span, offset), chosen)
    return on * (t / int(on.sum()))


@dataclass(frozen=True)
class Point:
    """A design point: the mean shift added to the replicates, how it is spread, and how many of the flips it reads."""

    name: str  # "size", "power" or "concentrated"
    delta_share: float  # signed share of climatology's log-loss (negative: a better); 0 for the null
    shift: float  # absolute shift of the mean hourly difference
    replicates: int

    @property
    def concentrated(self) -> bool:
        return self.name == "concentrated"


def design_points(comp: Comparison, cfg: PowerConfig, clim: float, length: int) -> list[Point]:
    """The points at block length L. The concentrated ones exist for superiority at ``cfg.concentrated_block`` only."""
    if comp.margin is None:  # superiority: the null is 0; power plants delta in the registered direction, a lower
        out = [Point("size", 0.0, 0.0, cfg.size_replicates)] + [Point("power", -d, -d * clim, cfg.power_replicates) for d in cfg.deltas]
        if length == cfg.concentrated_block:
            out += [Point("concentrated", -d, -d * clim, cfg.power_replicates) for d in cfg.deltas]
        return out
    m = comp.margin * clim  # non-inferiority: size sits on the margin, power at a true difference of 0
    return [Point("size", comp.margin, m, cfg.size_replicates), Point("power", 0.0, 0.0, cfg.power_replicates)]


# --- The rule on one replicate ----------------------------------------------------------------


def dm_part(x: np.ndarray, rule: Rule, t95: float) -> tuple[float, float, float, int]:
    """Diebold-Mariano's p, the mean, the one-sided 95 % upper bound and the Politis-White block of x, as report2 computes them."""
    try:
        dm = stats.diebold_mariano(x, np.zeros_like(x), h=rule.dm_h)
    except ValueError:  # a constant replicate: the test is undefined, and it rejects nothing
        return math.nan, float(x.mean()), math.nan, 1
    mean = float(x.mean())
    se = r2.dm_standard_error(dm, x, rule.dm_h)
    return dm.p_value, mean, mean + t95 * se, r2._block(x)


def boot_counts(idx: np.ndarray, t: int) -> np.ndarray:
    """The count matrix of a chunk of bootstrap index paths: row j counts how often each hour appears in path j."""
    n = idx.shape[0]
    flat = (idx + (np.arange(n, dtype=np.int64) * t)[:, None]).ravel()
    return np.bincount(flat, minlength=n * t).reshape(n, t)


def count_matrix(t: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, t) counts of the registered bootstrap's index paths, in uint16; exactly the paths of ``report2.mean_interval``."""
    parts = []
    for idx in r2._chunks(t, block, n_boot, seed):
        counts = boot_counts(idx, t)
        if counts.max() > np.iinfo(np.uint16).max:
            raise OverflowError("an hour appears more than 65,535 times in one bootstrap path")
        parts.append(counts.astype(np.uint16))
    return np.concatenate(parts, axis=0)


def grouped_means(counts: np.ndarray, x: np.ndarray, *, product_columns: int = 256, chunk_rows: int = 100) -> np.ndarray:
    """The resampled means (n_boot, n) of the replicates x (t, n): ``counts @ x / t``, chunk by chunk."""
    t, n = x.shape
    out = np.empty((counts.shape[0], n))
    for j in range(0, n, product_columns):
        xs = np.ascontiguousarray(x[:, j : j + product_columns])
        for i in range(0, counts.shape[0], chunk_rows):
            out[i : i + chunk_rows, j : j + product_columns] = (counts[i : i + chunk_rows].astype(np.float64) @ xs) / t
    return out


def interval_from_means(means: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """low, high (the percentile interval) and q95 of each column of resampled means, as ``report2.mean_interval``."""
    a = (1.0 - level) / 2.0
    low, high, q95 = np.empty(means.shape[1]), np.empty(means.shape[1]), np.empty(means.shape[1])
    for j in range(means.shape[1]):
        draws = means[:, j]
        finite = draws[np.isfinite(draws)]
        low[j], high[j] = np.quantile(finite, [a, 1.0 - a])
        q95[j] = np.quantile(draws, 0.95)
    return low, high, q95


# --- Worker jobs ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Batches:
    """How the work is cut into jobs. It changes no result (a test runs two cuttings), so it is not part of the design."""

    replicates: int = 50  # replicates per Diebold-Mariano and Politis-White job
    columns: int = 1024  # replicates per count-matrix job
    product_columns: int = 256  # replicates per matrix product inside a job


@dataclass(frozen=True)
class _Job:
    """One comparison: its centred difference and its design points at every L."""

    c: np.ndarray
    comp: Comparison
    cfg: PowerConfig
    rule: Rule
    points: dict[int, tuple[Point, ...]]  # per L
    product_columns: int = 256


class _Replicates:
    """Builds replicate r of point p at block length L, exactly as both stages must see it."""

    def __init__(self, job: _Job) -> None:
        self.job, self.t = job, job.c.size
        self.flips: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.conc: tuple[np.ndarray, np.ndarray] | None = None
        self.weights: tuple[int, np.ndarray] | None = None  # the last replicate's concentration weights

    def base(self, length: int, r: int) -> np.ndarray:
        if length not in self.flips:
            self.flips[length] = flip_plan(self.job.cfg, self.job.comp, length, self.t)
        offsets, signs = self.flips[length]
        return flipped(self.job.c, length, offsets[r], signs[r])

    def shifted(self, base: np.ndarray, point: Point, r: int) -> np.ndarray:
        if not point.concentrated:
            return base + point.shift
        if self.conc is None:
            self.conc = concentration_plan(self.job.cfg, self.job.comp, self.t)
        if self.weights is None or self.weights[0] != r:
            offsets, chosen = self.conc
            self.weights = (r, concentration(self.t, self.job.cfg.concentrated_span, offsets[r], chosen[r]))
        return base + point.shift * self.weights[1]


def _job_dm(job: _Job, length: int, start: int, stop: int) -> np.ndarray:
    """Rows (point, rep, p, mean, upper, block) for the flips [start, stop) at L, at every point that reads them."""
    reps = _Replicates(job)
    t95 = float(st.t.ppf(0.95, df=job.c.size - 1))
    rows = []
    for r in range(start, stop):
        base = reps.base(length, r)
        for p, point in enumerate(job.points[length]):
            if r < point.replicates:
                rows.append((p, r, *dm_part(reps.shifted(base, point, r), job.rule, t95)))
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def _job_boot(job: _Job, block: int, columns: np.ndarray) -> np.ndarray:
    """(low, high, q95) of the replicates ``columns`` (rows of (L, point, rep)), all with the Politis-White block ``block``.

    The count matrix depends on T, the block and the comparison's seed only, so one job serves every L and every point.
    """
    reps = _Replicates(job)
    x = np.empty((job.c.size, columns.shape[0]))
    for k, (length, p, r) in enumerate(columns):
        x[:, k] = reps.shifted(reps.base(int(length), int(r)), job.points[int(length)][int(p)], int(r))
    counts = count_matrix(job.c.size, block, job.rule.n_boot, job.rule.seed(job.comp.a, job.comp.b))
    means = grouped_means(counts, x, product_columns=job.product_columns)
    return np.column_stack(interval_from_means(means, job.rule.level))


class _Inline:
    """``workers = 1``: run each job at once in this process. Pools pickle their results, which pytest's tripwire refuses."""

    def __enter__(self) -> _Inline:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def submit(self, fn: Any, *args: Any) -> Future:
        fut: Future = Future()
        fut.set_result(fn(*args))
        return fut


@contextlib.contextmanager
def _executor(workers: int) -> Iterator[Any]:
    if workers <= 1:
        with _Inline() as ex:
            yield ex
        return
    with _one_blas_thread(), ProcessPoolExecutor(max_workers=workers) as pool:
        yield pool


@contextlib.contextmanager
def _one_blas_thread() -> Iterator[None]:
    """Worker processes inherit one BLAS thread each: the products do not oversubscribe, and their bits do not depend
    on how many workers run."""
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in names}
    os.environ.update({k: "1" for k in names})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- The study --------------------------------------------------------------------------------


def _rate(flags: Sequence[bool]) -> dict[str, Any]:
    k, n = int(sum(bool(f) for f in flags)), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": [lo, hi]}


def detectable(deltas: Sequence[float], power: Sequence[float], target: float) -> dict[str, Any]:
    """The smallest grid delta with power >= target, and the delta interpolated linearly between it and the grid point before."""
    for i, (d, p) in enumerate(zip(deltas, power, strict=True)):
        if p >= target:
            if i == 0:
                return {"grid": d, "interpolated": None, "note": "reached at the grid's first point: the interpolation would leave the grid"}
            d0, p0 = deltas[i - 1], power[i - 1]
            return {"grid": d, "interpolated": d0 + (target - p0) * (d - d0) / (p - p0), "note": "interpolated linearly between grid points"}
    return {"grid": None, "interpolated": None, "note": f"power stays below {target:.0%} on the whole grid"}


def power_at(share: float, deltas: Sequence[float], power: Sequence[float]) -> float | None:
    """Power interpolated linearly at an effect ``share`` (positive, in the registered direction) inside the grid; None outside."""
    if not deltas[0] <= share <= deltas[-1]:
        return None
    return float(np.interp(share, deltas, power))


def _classify(comp: Comparison, point: Point, res: np.ndarray, margin_abs: float, t95: float) -> dict[str, Any]:
    """Rates of a design point. ``res`` columns: p, mean, upper, block, low, high, q95.

    At the size point it also gives ``variance_ratio``: the mean squared error of the replicates' means
    around the true difference over the mean squared HLN-corrected DM standard error. Near 1, DM's
    variance is right for these replicates; above 1, it is too small and DM over-rejects, by about
    ``dm_size_from_ratio`` (normal approximation). With few blocks (T / L small) the ratio, and the size,
    are conditional on the realized block sums of the series.
    """
    p, mean, upper, low, high, q95 = res[:, 0], res[:, 1], res[:, 2], res[:, 4], res[:, 5], res[:, 6]
    out: dict[str, Any] = {"replicates": int(res.shape[0]), "undefined": int(np.isnan(p).sum())}
    if comp.margin is None:
        dm = [bool(v < DM_ALPHA) for v in p]
        if point.name == "size":
            out["dm"] = _rate(dm)
            out["bootstrap"] = _rate([bool(lo > 0.0 or hi < 0.0) for lo, hi in zip(low, high, strict=True)])
            out["pass_two_sided"] = _rate([d and ((m < 0.0 and hi < 0.0) or (m > 0.0 and lo > 0.0))
                                           for d, m, lo, hi in zip(dm, mean, low, high, strict=True)])  # fmt: skip
        else:
            out["dm_registered"] = _rate([d and m < 0.0 for d, m in zip(dm, mean, strict=True)])
            out["bootstrap_registered"] = _rate([bool(hi < 0.0) for hi in high])
        out["pass_registered"] = _rate([r2.passes(*v) for v in zip(p, low, high, mean, strict=True)])
    else:
        out["upper_below_margin"] = _rate([bool(u < margin_abs) for u in upper])
        out["q95_below_margin"] = _rate([bool(q < margin_abs) for q in q95])
        out["holds"] = _rate([r2.h12_rule_met(u, q, margin_abs) for u, q in zip(upper, q95, strict=True)])
    blocks = res[:, 3].astype(np.int64)
    out["blocks"] = {"min": int(blocks.min()), "median": float(np.median(blocks)), "max": int(blocks.max()), "distinct": int(np.unique(blocks).size)}
    if point.name == "size":
        ok = np.isfinite(upper)
        se2 = float(np.mean(((upper[ok] - mean[ok]) / t95) ** 2))
        ratio = float(np.mean((mean[ok] - point.shift) ** 2)) / se2
        out["variance_ratio"] = ratio
        out["dm_size_from_ratio"] = float(2.0 * st.norm.sf(st.norm.ppf(1.0 - DM_ALPHA / 2.0) / math.sqrt(ratio)))
    return out


def observed_rule(ll: dict[str, np.ndarray], comp: Comparison, rule: Rule, clim: float) -> dict[str, Any]:
    """The registered rule on the observed difference, by report2's own path (``compare_arms``)."""
    c = r2.compare_arms(ll, comp.a, comp.b, clim, rule.report2_config())
    keep = ("mean_diff", "share_of_climatology", "dm_statistic", "dm_p", "dm_se", "upper_95_one_sided", "bootstrap", "passes")
    return {k: c[k] for k in keep}


def reproduction(mine: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Whether the rule recomputed from the series gives report2.json's numbers to the bit, and else how far they are.

    The digest already ties the series to its run, so a difference here says the environment moved (a numpy
    or scipy version, a BLAS), not the data: it is recorded, and the study still runs.
    """
    flat_mine = {**{k: v for k, v in mine.items() if k != "bootstrap"}, **{f"bootstrap.{k}": v for k, v in mine["bootstrap"].items()}}
    flat_rec = {**{k: v for k, v in recorded.items() if k in mine and k != "bootstrap"},
                **{f"bootstrap.{k}": v for k, v in recorded.get("bootstrap", {}).items()}}  # fmt: skip
    worst, differ = 0.0, []
    for k, v in flat_mine.items():
        r = flat_rec.get(k)
        if v != r:
            differ.append(k)
            if isinstance(v, float) and isinstance(r, float) and r != 0.0:
                worst = max(worst, abs(v - r) / abs(r))
    return {"identical": not differ, "differing": sorted(differ), "max_relative_difference": worst}


def study(ll: dict[str, np.ndarray], comparisons: Sequence[Comparison], rule: Rule, cfg: PowerConfig, *,
          climatology: str = r2.CLIMATOLOGY, workers: int = 1, batches: Batches = Batches(),
          timings: dict[str, float] | None = None) -> dict[str, Any]:  # fmt: skip
    """The size and power of the pass rule on each comparison's hourly difference.

    Deterministic: every replicate, its group and its result follow from the data and the design alone;
    ``workers`` and ``batches`` only cut the same computations into jobs. With ``workers`` > 1 the jobs
    run in processes with one BLAS thread each; with 1 they run here, where a multi-threaded BLAS may
    round the long dot products of Politis-White and Diebold-Mariano differently in the last bits.
    """
    cfg.check()
    clim = float(ll[climatology].mean())
    t = int(ll[climatology].size)
    t95 = float(st.t.ppf(0.95, df=t - 1))
    if cfg.concentrated_block is not None:
        chosen_blocks(cfg, t)  # refuse a series too short for the concentrated grid before any work
    jobs: dict[int, _Job] = {}
    undefined: dict[int, str] = {}
    for i, comp in enumerate(comparisons):
        d = ll[comp.a] - ll[comp.b]
        c = d - d.mean()
        if np.all(c == 0.0):
            undefined[i] = "the difference is the same at every hour"
            continue
        jobs[i] = _Job(c, comp, cfg, rule, {L: tuple(design_points(comp, cfg, clim, L)) for L in cfg.block_lengths}, batches.product_columns)
    timings = {} if timings is None else timings
    started = time.perf_counter()
    step = batches.replicates
    with _executor(int(workers)) as pool:
        futures = {}
        for i, job in jobs.items():
            for length in cfg.block_lengths:
                for s in range(0, cfg.size_replicates, step):
                    futures[(i, length, s)] = pool.submit(_job_dm, job, length, s, min(s + step, cfg.size_replicates))
        first = {(i, L): np.concatenate([futures[(i, L, s)].result() for s in range(0, cfg.size_replicates, step)])
                 for i in jobs for L in cfg.block_lengths}  # fmt: skip
        timings["dm_and_politis_white"] = time.perf_counter() - started
        started = time.perf_counter()
        boot_futures = []
        groups = 0
        for i in jobs:
            keys = [(i, length) for length in cfg.block_lengths]
            # every replicate of the comparison, L by L in the config's order: L, point, rep, p, mean, upper, block
            res = np.concatenate([np.column_stack([np.full(first[k].shape[0], k[1], dtype=np.float64), first[k]]) for k in keys])
            starts = np.cumsum([0] + [first[k].shape[0] for k in keys])
            for block in np.unique(res[:, 6].astype(np.int64)):
                rows = np.flatnonzero(res[:, 6] == block)  # deterministic: the order of the replicates, not of the workers
                groups += 1
                for s in range(0, rows.size, batches.columns):
                    part = rows[s : s + batches.columns]
                    cols = res[part][:, :3].astype(np.int64)
                    boot_futures.append((keys, starts, part, pool.submit(_job_boot, jobs[i], int(block), cols)))
        boot = {key: np.full((first[key].shape[0], 3), np.nan) for key in first}
        for keys, starts, part, fut in boot_futures:
            values = fut.result()
            which = np.searchsorted(starts, part, side="right") - 1
            for w, key in enumerate(keys):
                sel = which == w
                boot[key][part[sel] - starts[w]] = values[sel]
        timings["grouped_bootstrap"] = time.perf_counter() - started
    out_comps = []
    for i, comp in enumerate(comparisons):
        margin_abs = None if comp.margin is None else comp.margin * clim
        entry: dict[str, Any] = {"hypothesis": comp.hypothesis, "a": comp.a, "b": comp.b, "kind": comp.kind, "margin_share": comp.margin,
                                 "margin": margin_abs}  # fmt: skip
        d = ll[comp.a] - ll[comp.b]
        entry["observed"] = {"mean_diff": float(d.mean()), "share_of_climatology": float(d.mean() / clim)}
        if i in undefined:
            entry["undefined"] = undefined[i]
            out_comps.append(entry)
            continue
        entry["size"], entry["power"] = {}, {}
        if comp.margin is None and cfg.concentrated_block is not None:
            entry["power_concentrated"] = {}
        for length in cfg.block_lengths:
            pts = jobs[i].points[length]
            res = np.column_stack([first[(i, length)], boot[(i, length)]])[:, 2:]  # p, mean, upper, block, low, high, q95
            pidx = first[(i, length)][:, 0].astype(np.int64)
            by_point = [res[pidx == k] for k in range(len(pts))]
            size = _classify(comp, pts[0], by_point[0], margin_abs if margin_abs is not None else 0.0, t95)
            entry["size"][str(length)] = size
            if comp.margin is None:
                for name, where in (("power", entry["power"]), ("concentrated", entry.get("power_concentrated"))):
                    curve = [dict(_classify(comp, pt, by_point[k], 0.0, t95), delta_share=-pt.delta_share, delta=pt.shift)
                             for k, pt in enumerate(pts) if pt.name == name]  # fmt: skip
                    if curve:
                        rates = [p["pass_registered"]["rate"] for p in curve]
                        where[str(length)] = {"points": curve, "detectable": detectable(list(cfg.deltas), rates, cfg.power_target)}
            else:
                entry["power"][str(length)] = {"at_zero": _classify(comp, pts[1], by_point[1], margin_abs, t95)}
        out_comps.append(entry)
    alerts = []
    for e in out_comps:
        for length, s in _by_length(e.get("size", {})):
            rate = s["pass_two_sided"]["rate"] if e["kind"] == "superiority" else s["holds"]["rate"]
            if rate > cfg.size_alert:
                alerts.append({"hypothesis": e["hypothesis"], "a": e["a"], "b": e["b"], "block_length": int(length), "size": rate})
    return {
        "hours": t,
        "climatology_log_loss": clim,
        "comparisons": out_comps,
        "size_alerts": alerts,
        "bootstrap_groups": groups,
        "replicates": int(sum(p.replicates for job in jobs.values() for pts in job.points.values() for p in pts)),
        "concentrated": None if cfg.concentrated_block is None else concentrated_design(cfg, t),
    }


def concentrated_design(cfg: PowerConfig, t: int) -> dict[str, Any]:
    """How the concentrated grid spreads its shift over t hours: the blocks, how many are chosen, the weight on a whole block."""
    blocks, k = chosen_blocks(cfg, t)
    return {"block_length": cfg.concentrated_block, "span": cfg.concentrated_span, "share": cfg.concentrated_share, "span_blocks": blocks,
            "chosen": k, "weight_whole_blocks": t / (k * cfg.concentrated_span), "last_block_hours": t - (blocks - 1) * cfg.concentrated_span}  # fmt: skip


def readings(result: dict[str, Any], cfg: PowerConfig) -> list[dict[str, Any]]:
    """Each superiority comparison's observed effect against the effect the rule detects with 80 % power, per L.

    ``per_block_length`` reads the constant shift, whose detectable delta is a lower bound of the detectable mean gain;
    ``concentrated`` reads the concentrated grid at its L, where a gain spread as the grid spreads it is read.
    """

    def row(curve: dict[str, Any], share: float) -> dict[str, Any]:
        det = curve["detectable"]
        rates = [p["pass_registered"]["rate"] for p in curve["points"]]
        in_direction = share < 0
        return {
            "detectable": det["grid"],
            "detectable_interpolated": det["interpolated"],
            "observed_below_detectable": bool(-share < (det["interpolated"] or det["grid"])) if in_direction and det["grid"] is not None else None,
            "power_at_observed": power_at(-share, list(cfg.deltas), rates) if in_direction else None,
        }

    out = []
    for e in result["comparisons"]:
        if e["kind"] != "superiority" or "undefined" in e:
            continue
        share = e["observed"]["share_of_climatology"]
        out.append({"hypothesis": e["hypothesis"], "a": e["a"], "b": e["b"], "observed_share": share,
                    "direction": "registered (a lower)" if share < 0 else "opposite to the registered direction",
                    "per_block_length": {length: row(curve, share) for length, curve in e["power"].items()},
                    "concentrated": {length: row(curve, share) for length, curve in e.get("power_concentrated", {}).items()}})  # fmt: skip
    return out


# --- Outputs ----------------------------------------------------------------------------------


def _by_length(d: dict[str, Any]) -> list[tuple[str, Any]]:
    """A per-L dict's items in increasing L. JSON with sorted keys orders them as strings ("168" before "24")."""
    return sorted(d.items(), key=lambda kv: int(kv[0]))


def _r(rate: dict[str, Any]) -> str:
    lo, hi = rate["wilson95"]
    return f"{100 * rate['rate']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def render(power: dict[str, Any], config: dict[str, Any]) -> str:
    res, cfg = power["result"], config["config"]
    lines = ["# The size and power of the pass rule", ""]
    if not power["registered"]:
        lines += ["**NOT THE REGISTERED STUDY.** Its design or its series differs from the registered one (a test, or a synthetic "
                  "series): its numbers are not results.", ""]  # fmt: skip
    lines += [
        "Registered in `docs/PROTOCOL.md`, \"Size and power of the pass rule\". It describes what a verdict under the pass rule is "
        "worth; the rule does not change, and no arm is refitted.",
        "",
        f"Series: `{config['series']['file']}` of `{config['series']['run']}`, SHA-256 `{config['series']['sha256']}`, "
        f"{res['hours']} hours; climatology's log-loss {res['climatology_log_loss']:.5f}. Replicates: {res['replicates']}, "
        f"in {res['bootstrap_groups']} groups of equal Politis-White block.",
        "",
        f"Each replicate is the centred hourly difference, rotated by a random offset in [0, L), cut into consecutive circular blocks of "
        f"L hours, each block multiplied by an independent random sign (flip seed {cfg['flip_seed']}). The rule runs on it as registered: "
        f"Diebold-Mariano (two-sided, rectangular kernel, {config['rule']['dm_h'] - 1} lags, HLN) and the stationary bootstrap of the "
        f"mean ({config['rule']['n_boot']} resamples, {config['rule']['level']:.0%} percentile interval) with the Politis-White block "
        "recomputed on the replicate and the comparison's registered seed. Rates in %, with 95 % Wilson intervals.",
        "",
        "## Size",
        "",
        "Superiority: the true difference is 0. DM: p < 0.05; bootstrap: its interval excludes 0; pass rule: both on the same side; "
        "registered: the pass rule with a lower. H12: the true difference sits on its margin, and the rate is that at which it holds.",
        "",
        "| comparison | L (hours) | DM | bootstrap | pass rule, two-sided | pass rule, registered | PW block: median [min, max] "
        "| variance ratio (DM size it implies) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in res["comparisons"]:
        name = f"{e['hypothesis']}: {e['a']} - {e['b']}"
        if "undefined" in e:
            lines.append(f"| {name} | - | no test: {e['undefined']} | | | | | |")
            continue
        for length, s in _by_length(e["size"]):
            blk = f"{s['blocks']['median']:g} [{s['blocks']['min']}, {s['blocks']['max']}]"
            vr = f"{s['variance_ratio']:.3f} ({100 * s['dm_size_from_ratio']:.1f})"
            if e["kind"] == "superiority":
                lines.append(f"| {name} | {length} | {_r(s['dm'])} | {_r(s['bootstrap'])} | {_r(s['pass_two_sided'])} | "
                             f"{_r(s['pass_registered'])} | {blk} | {vr} |")  # fmt: skip
            else:
                lines.append(f"| {name} | {length} | upper bound below m: {_r(s['upper_below_margin'])} | q95 below m: "
                             f"{_r(s['q95_below_margin'])} | holds: {_r(s['holds'])} | - | {blk} | {vr} |")  # fmt: skip
    alerts = res["size_alerts"]
    lines += ["", f"Above {cfg['size_alert']:.1%}: " + ("; ".join(f"{a['hypothesis']} {a['a']} - {a['b']} at L = {a['block_length']} "
                                                                f"({100 * a['size']:.1f} %)" for a in alerts) if alerts else "none") + "."]  # fmt: skip
    deltas = cfg["deltas"]

    def table(key: str) -> list[str]:
        rows = ["| comparison | L | " + " | ".join(f"{100 * d:g} %" for d in deltas) + " | detectable at 80 % |", "|---|---|" + "---|" * len(deltas) + "---|"]
        for e in res["comparisons"]:
            if e["kind"] != "superiority" or "undefined" in e:
                continue
            for length, pw in _by_length(e.get(key, {})):
                det = pw["detectable"]
                text = "not on the grid" if det["grid"] is None else f"{100 * det['grid']:g} %" + (
                    f" (interpolated {100 * det['interpolated']:.3f} %)" if det["interpolated"] is not None else " (at the grid's first point)")
                rows.append(f"| {e['hypothesis']}: {e['a']} - {e['b']} | {length} | "
                            + " | ".join(f"{100 * p['pass_registered']['rate']:.1f}" for p in pw["points"]) + f" | {text} |")  # fmt: skip
        return rows

    lines += [
        "",
        "## Power",
        "",
        "flip(d - mean(d)) + delta, delta in the registered direction (a better), as a share of climatology's log-loss; the rate at "
        f"which the comparison passes as registered. The first {cfg['power_replicates']} flips of the size replicates serve every delta.",
        "",
        "A constant shift is the easiest gain to see, so the detectable delta of this grid is a **lower bound of the detectable mean "
        "gain**: a gain concentrated in regimes needs more (the concentrated grid below).",
        "",
        *table("power"),
    ]
    conc = res.get("concentrated")
    if conc is not None:
        lines += [
            "",
            "### Concentrated gain",
            "",
            f"At L = {conc['block_length']} only, on the same first {cfg['power_replicates']} flips: the hours are rotated by a random "
            f"offset in [0, {conc['span']}) and cut into {conc['span_blocks']} circular blocks of {conc['span']} hours (the last one "
            f"{conc['last_block_hours']} hours); {conc['chosen']} of them ({conc['share']:.0%}), drawn at random for each replicate, "
            f"carry the whole shift and the others none. The shift on a chosen hour is delta x T / (hours chosen), "
            f"{conc['weight_whole_blocks']:.3f} delta when every chosen block is whole, so that the mean shift is exactly delta, as in the "
            "constant grid.",
            "",
            *table("power_concentrated"),
        ]
    for e in res["comparisons"]:
        if e["kind"] == "non-inferiority" and "undefined" not in e:
            lines += ["", f"{e['hypothesis']} ({e['a']} - {e['b']}), the rate at which it holds when the true difference is 0: "
                      + "; ".join(f"L = {length}: {_r(pw['at_zero']['holds'])}" for length, pw in _by_length(e["power"])) + "."]  # fmt: skip
    lines += ["", "## The observed effects, read against the detectable ones", "",
              "The constant-shift detectable sizes are lower bounds of the detectable mean gain; the concentrated ones read a gain "
              "spread over a tenth of the period.", ""]  # fmt: skip
    for rd in power["readings"]:
        per = rd["per_block_length"]
        parts = []
        for length, v in _by_length(per):
            det = "not reached on the grid" if v["detectable"] is None else (
                f"{100 * (v['detectable_interpolated'] or v['detectable']):.3f} %")  # fmt: skip
            at = "" if v["power_at_observed"] is None else f", power at the observed effect {100 * v['power_at_observed']:.1f} % (interpolated)"
            parts.append(f"L = {length}: detectable {det}{at}")
        for length, v in _by_length(rd.get("concentrated", {})):
            det = "not reached on the grid" if v["detectable"] is None else f"{100 * (v['detectable_interpolated'] or v['detectable']):.3f} %"
            at = "" if v["power_at_observed"] is None else f", power at the observed effect {100 * v['power_at_observed']:.1f} % (interpolated)"
            parts.append(f"concentrated, L = {length}: detectable {det}{at}")
        lines.append(f"- {rd['hypothesis']} {rd['a']} - {rd['b']}: observed {100 * rd['observed_share']:+.4f} % of climatology's "
                     f"log-loss ({rd['direction']}); " + "; ".join(parts) + ".")  # fmt: skip
    lines += [
        "",
        "## Notes",
        "",
        "- The resampled means come from a product of the bootstrap's count matrix with the replicates, grouped by block length. They "
        "equal the direct path `x[idx].mean(axis=1)` up to the last bits, so an interval can differ from the direct path's in its last "
        "bits only.",
        "- Common random numbers: every delta of a comparison and L reads the same flips, so its power curve moves with delta alone; "
        "the concentrated grid reads those same flips too.",
        "- The Wilson intervals of size and power are conditional on the registered bootstrap seed: every replicate draws its bootstrap "
        "indices from that one seed (common random numbers across replicates), as the registered rule would, so they describe the rule "
        "with that seed, not a rule whose bootstrap is drawn afresh.",
        "- The variance ratio is the replicates' mean squared error around the true difference over their mean squared DM standard "
        "error; the DM size it implies is the normal approximation 2(1 - Phi(1.96 / sqrt(ratio))). A replicate keeps only the T / L "
        "block sums of the series, so at large L the size is conditional on how those few sums happened to fall.",
        "- The observed rule, recomputed from the series by `report2.compare_arms`, reproduces every comparison of the run's "
        "`report2.json`: " + ("yes" if power["observed_reproduces_report2"] else "NO: " + "; ".join(
            f"{name}, {', '.join(v['differing'])} (largest relative difference {v['max_relative_difference']:.1e})"
            for name, v in power["observed"].items() if not v["identical"])) + ".",  # fmt: skip
        "",
        *prov_mod.render_md(config["provenance"]),
        f"- The data manifest digest is that of the Report n°2 run, `{config['series']['run']}`, which made the series.",
        "",
    ]
    return "\n".join(lines)


def config_record(cfg: PowerConfig, rule: Rule, comparisons: Sequence[Comparison], series: dict[str, Any], report_config: dict[str, Any],
                  provenance: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    import scipy

    return {
        "study": "power",
        "config": asdict(cfg),
        "registered": cfg.registered,
        "rule": asdict(rule),
        "comparisons": [asdict(c) for c in comparisons],
        "series": series,
        "report_run": {"provenance": report_config.get("provenance"), "registered": report_config.get("registered")},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]},
        "provenance": provenance,
    }


def run_study(series_path: Path, sha256: str, out: Path, *, cfg: PowerConfig = PowerConfig(), workers: int = 1, batches: Batches = Batches(),
              comparisons: Sequence[Comparison] | None = None, provenance: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Read the series by its digest, run the study and write the run folder. Returns ``power.json``'s content."""
    started = time.perf_counter()
    arrays, report, report_config, report_sha = load_series(series_path, sha256)
    rule = Rule.from_report2(report_config)
    margin = float(report_config["config"].get("h12_margin", 0.005))
    comparisons = report2_comparisons(margin) if comparisons is None else tuple(comparisons)
    arms = [str(a) for a in arrays["arms"]]
    ll = {a: np.ascontiguousarray(arrays["log_loss"][k]) for k, a in enumerate(arms)}
    clim = float(ll[r2.CLIMATOLOGY].mean())
    if clim != report["arms"][r2.CLIMATOLOGY]["log_loss"]["estimate"]:
        raise SeriesRefused("climatology's mean hourly log-loss in the series is not the one of report2.json")
    observed = {}
    for comp in comparisons:
        recorded = next((c for c in report.get("comparisons", []) if (c["a"], c["b"]) == (comp.a, comp.b)), None)
        if recorded is not None:
            observed[f"{comp.a} - {comp.b}"] = reproduction(observed_rule(ll, comp, rule, clim), recorded)
    timings: dict[str, float] = {}
    result = study(ll, comparisons, rule, cfg, workers=workers, batches=batches, timings=timings)
    power = {
        "registered": bool(cfg.registered and report.get("registered") is True),
        "result": result,
        "readings": [],
        "observed_reproduces_report2": bool(observed) and all(v["identical"] for v in observed.values()),
        "observed": observed,
    }
    power["readings"] = readings(result, cfg)
    series = {"file": series_path.name, "sha256": sha256.strip().lower(), "run": series_path.parent.name, "report2_json_sha256": report_sha}
    if provenance is None:
        provenance = {**prov_mod.record({"digest": report_config.get("provenance", {}).get("manifest_digest")})}
    config = config_record(cfg, rule, comparisons, series, report_config, provenance)
    out.mkdir(parents=True, exist_ok=True)
    r2._write_json(out / "config.json", config)
    r2._write_json(out / "power.json", power)
    power, config = (json.loads((out / n).read_text(encoding="utf-8")) for n in ("power.json", "config.json"))
    (out / POWER_FILE).write_text(render(power, config), encoding="utf-8", newline="\n")
    run = {"finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - started, 1),
           "stages_seconds": {k: round(v, 1) for k, v in timings.items()}, "workers": int(workers), "batches": asdict(batches)}  # fmt: skip
    r2._write_json(out / "run.json", run)
    return power


def _default_out() -> Path:
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-power"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.power", description="The size and power of the registered pass rule.")
    parser.add_argument("--series", type=Path, required=True, help="series.npz of a Report n°2 run")
    parser.add_argument("--sha256", required=True, help="the SHA-256 the series must have (the run's report2.json must agree)")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default: <runs>/<UTC date>-power, suffixed if taken)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)), help="worker processes")
    args = parser.parse_args(argv)
    out = args.out or _default_out()
    try:
        power = run_study(args.series, args.sha256, out, workers=args.workers)
    except SeriesRefused as e:
        print(f"REFUSED: {e}")
        return 2
    alerts = power["result"]["size_alerts"]
    print(f"power written to {out}; size alerts: {len(alerts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
