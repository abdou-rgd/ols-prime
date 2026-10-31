"""H1's blind spots: the adversary's misses, classified by future-shuffle twins.

``docs/PROTOCOL.md``, "The encoder, and what a verdict is worth (iteration 4)", section "H1's blind spots".
The adversary (:mod:`olsprime.bench.adversary`) counts the random leaks H1's perturbation misses. This module
says what each miss is. The result is descriptive: H1's rule does not change.

**The leaks** are regenerated from the adversary's own configuration and seed
(:func:`olsprime.bench.adversary.generate`), and H1's verdict on each is recomputed with the adversary's own
bench restricted to H1's cuts (:func:`olsprime.bench.adversary.build_bench` with no long schedule) and its own
fast path (:func:`olsprime.bench.adversary.evaluate_fast`). With ``--adversary``, every leak's H1 pattern and
constant flag must equal the run's, leak by leak, or nothing is classified.

**A future-shuffle twin** of a pair's bars at a cut t (:func:`shuffle_after`): every bar after t keeps its
time stamps (``open_time``, ``close_time``, ``known_at``) and its ``symbol``, and takes every other value
(open, high, low, close, volume, quote volume, trades, taker base and quote volumes) of one bar drawn
uniformly, with replacement, from the same pair's bars at or before the cut, bar t included. Bars at or
before the cut are returned bit for bit. The quantities are recomputed from the twin bars
(:func:`olsprime.bench.adversary.quantities`), exactly as H1's perturbation recomputes them, and windowed as
``build_bench`` windows them.

**Seeds.** Twin ``j`` of the ``k``-th of a pair's H1 cuts (0-based, in H1's order) draws from its own
generator, ``numpy.random.default_rng([20261030, crc32(pair), k, j])``: the draws do not depend on the order
of the pairs, of the leaks or on the worker count. They are made once per (pair, cut, twin) and shared by
every leak. The source rows of bars t+1..t+10, the only later bars a leak can read, are written to the report.

**Moved.** A twin moves a leak at a cut when, at a compared row (rows t-10..t that H1 scores), one of its
outputs differs from its output on the original bars by more than ``CAUGHT_ABOVE`` (1e-9), turns missing or
stops being missing: H1's own catch rule, through ``evaluate_fast`` itself, with the K twins in place of
H1's two directions.

**The classes.** A leak caught at none of H1's cuts (a miss) falls in exactly one class, in this order:

1. ``constant``: the same output, bit for bit, at every compared row of H1's cuts on the original bars;
2. ``unchanged``: no twin moves it at any cut; on these bars it does not depend on the future;
3. ``blind-spot``: some twin moves it at some cut, yet H1 did not catch it.

The adversary's own constant flag is computed over the compared rows of both its schedules (H1's cuts and
the long schedule); this module reports it beside the H1-only flag, which is the one that classifies.

**Checks.** Every leak's honest twin (the adversary's: the same expression moved into the past) runs through
the same future-shuffle twins and must not differ by a single bit: one that does is a bench bug, and the run
fails. Every caught leak is also scored: the share that some twin moves is a sanity check, near 1.

Usage::

    uv run --no-sync python -m olsprime.bench.twins [--adversary <run folder> --sha256 <digest of its adversary.json>]
        [--out DIR] [--workers N]

It writes ``twins.json`` and ``twins.md`` (sorted keys, no wall clock, byte-identical across runs and worker
counts), ``config.json`` (the configuration and the provenance) and ``run.json`` (the wall clock) to
``runs/<UTC date>-twins`` (suffixed if taken) or to ``--out``. ``--synthetic`` runs on synthetic bars, for
tests and smoke runs only; it reads no data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from olsprime.bench import adversary as adv
from olsprime.bench import invariance as inv
from olsprime.bench import pipeline as pp
from olsprime.bench import provenance
from olsprime.bench.perturb import PERTURBED_COLUMNS, UNCHANGED_COLUMNS, Perturbation
from olsprime.data.binance_vision import Month
from olsprime.paths import runs_dir

log = logging.getLogger(__name__)

TWIN_SEED = 20261030  # registered: docs/PROTOCOL.md, "H1's blind spots"
K_TWINS = 8
CLASSES: tuple[str, ...] = ("constant", "unchanged", "blind-spot")
TABLE_KEYS: tuple[str, ...] = ("family", "quantity", "operation", "offset")
TIME_COLUMNS: tuple[str, ...] = ("open_time", "close_time", "known_at")
KEPT_COLUMNS: tuple[str, ...] = UNCHANGED_COLUMNS  # the time stamps and the symbol (the same pair's bars)
SHUFFLED_COLUMNS: tuple[str, ...] = PERTURBED_COLUMNS  # every other value of a bar
SAMPLE_LIMIT = 40  # caught leaks no twin moves, listed in the report

# Synthetic smoke runs only (``--synthetic``): two pairs of synthetic daily bars, a short test period.
SYNTHETIC_PAIRS: dict[str, int] = {"BTCUSDT": 11, "ETHUSDT": 12}
SYNTHETIC_DAYS = 560
SYNTHETIC_H1 = inv.H1Config(start=(2018, 3), end=(2018, 6), venues=("zero", "binance-spot", "kraken-pro-entry"))


@dataclass(frozen=True)
class TwinsConfig:
    k: int = K_TWINS
    seed: int = TWIN_SEED
    adversary: adv.AdversaryConfig = field(default_factory=adv.AdversaryConfig)


# --- The twins ---------------------------------------------------------------------------------


def twin_rng(seed: int, pair: str, k: int, j: int) -> np.random.Generator:
    """The generator of twin ``j`` at the ``k``-th of ``pair``'s H1 cuts: independent of any order."""
    return np.random.default_rng([seed, zlib.crc32(pair.encode("ascii")), k, j])


def twin_sources(n: int, cut: int, rng: np.random.Generator) -> np.ndarray:
    """For every row of a series of ``n`` bars, the row whose values it takes: itself up to ``cut``, then draws.

    The draws are uniform with replacement on rows ``0 .. cut``, the cut's own bar included.
    """
    if not 0 <= cut < n - 1:
        raise ValueError(f"cut must leave at least one later bar: 0 <= cut < {n - 1}, got {cut}")
    return np.concatenate([np.arange(cut + 1, dtype=np.int64), rng.integers(0, cut + 1, size=n - cut - 1, dtype=np.int64)])


def shuffle_after(bars: pl.DataFrame, cut: int, rng: np.random.Generator) -> pl.DataFrame:
    """A future-shuffle twin: each bar after ``cut`` keeps its time stamps and takes every other value of a past bar.

    ``bars`` is one pair's bars in time order; ``cut`` a row position with at least one later bar. Each row
    after the cut takes open, high, low, close, volume, quote volume, trades and both taker volumes of one row
    drawn uniformly, with replacement, among rows ``0 .. cut``. Rows ``0 .. cut`` are returned bit for bit, and
    every time column everywhere. A column that is neither a time stamp, the symbol nor a bar value raises,
    as in :func:`olsprime.bench.perturb.perturb_after`.
    """
    if bars["symbol"].n_unique() != 1:
        raise ValueError("shuffle one symbol at a time")
    if not bars["open_time"].is_sorted() or bars["open_time"].n_unique() != bars.height:
        raise ValueError("bars must be strictly increasing in open_time")
    unknown = [c for c in bars.columns if c not in KEPT_COLUMNS and c not in SHUFFLED_COLUMNS]
    if unknown:
        raise ValueError(f"the twins do not know columns {unknown}: decide how they move first")
    missing = [c for c in SHUFFLED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"the twins need every bar column; missing {missing}")
    src = pl.Series("src", twin_sources(bars.height, cut, rng))
    return bars.with_columns([bars[c].gather(src) for c in SHUFFLED_COLUMNS])


# --- The benches --------------------------------------------------------------------------------


@dataclass
class TwinBench:
    """H1's cuts three ways, each an :class:`olsprime.bench.adversary.Bench` that ``evaluate_fast`` reads.

    ``h1``: the adversary's own bench at H1's cuts (the original windows, then x3 and /3). ``twins``: the same
    cuts and compared rows, the original windows, then one block per twin. ``every``: the original windows at
    both of the adversary's schedules, for its own constant flag.
    """

    h1: adv.Bench
    twins: adv.Bench
    every: adv.Bench
    reads: list[dict[str, Any]]  # per (pair, cut): the source rows of bars t+1..t+10 in each twin


def _scored_mask(bars: pl.DataFrame, cfg: adv.AdversaryConfig) -> tuple[np.ndarray, list[Any]]:
    h1 = cfg.h1
    frame = pp.build_frame(bars, h1.interval, h1.h)
    folds = pp.make_folds(frame, h1.interval, h1.h, Month(*h1.start), Month(*h1.end))
    scored = np.zeros(bars.height, dtype=bool)
    scored[adv.scored_rows(folds)] = True
    return scored, folds


def _compared(scored: np.ndarray, cut: int) -> np.ndarray:
    rows = cut - adv.FUTURE_REACH + np.arange(adv.COMPARED)
    return np.where(rows >= 0, scored[np.clip(rows, 0, None)], False)


def _stack(blocks: list[list[dict[str, np.ndarray]]]) -> adv.WindowView:
    return adv.WindowView({name: np.stack([w[name] for block in blocks for w in block]) for name in adv.QUANTITY_NAMES})


def build(bars_by_pair: dict[str, pl.DataFrame], cfg: TwinsConfig) -> TwinBench:
    """Every perturbation and every twin once, for every pair, in the pairs' order."""
    h1 = adv.build_bench(bars_by_pair, replace(cfg.adversary, long_cuts=0))
    if any(c["schedule"] != "h1" for c in h1.cuts):
        raise AssertionError("the H1 bench holds a cut outside H1's schedule")
    blocks: list[list[dict[str, np.ndarray]]] = [[] for _ in range(1 + cfg.k)]
    every_blocks: list[dict[str, np.ndarray]] = []
    every_cuts, every_masks, reads = [], [], []
    position = 0
    for pair, bars in bars_by_pair.items():
        q0 = adv.quantities(bars)
        scored, folds = _scored_mask(bars, cfg.adversary)
        for k, cut in enumerate(h1.h1_cuts[pair]):
            c = h1.cuts[position]
            if (c["pair"], c["k"], c["index"]) != (pair, k, cut.index):
                raise AssertionError("the twins' cuts are not the H1 bench's, in its order")
            position += 1
            blocks[0].append(adv._window(q0, cut.index))
            sources = []
            for j in range(cfg.k):
                # Two fresh generators of the same seed: the twin's draws, and the same draws recorded for the report.
                twin = shuffle_after(bars, cut.index, twin_rng(cfg.seed, pair, k, j))
                src = twin_sources(bars.height, cut.index, twin_rng(cfg.seed, pair, k, j))
                blocks[1 + j].append(adv._window(adv.quantities(twin), cut.index))
                sources.append(src[cut.index + 1 : cut.index + 1 + adv.FUTURE_REACH].tolist())
            reads.append({"pair": pair, "k": k, "index": cut.index, "sources": sources})
        for schedule, cut_list, _ in adv.schedules(bars, folds, cfg.adversary):
            for k, cut in enumerate(cut_list):
                every_blocks.append(adv._window(q0, cut.index))
                every_masks.append(_compared(scored, cut.index))
                every_cuts.append({"pair": pair, "schedule": schedule, "k": k, "index": cut.index})
    if position != h1.n_cuts:
        raise AssertionError("the twins do not cover every H1 cut")
    names = tuple(f"twin-{j}" for j in range(cfg.k))
    twins = adv.Bench(_stack(blocks), h1.cuts, h1.scored, names, h1.h1_cuts)
    every = adv.Bench(_stack([every_blocks]), every_cuts, np.stack(every_masks), (), {})
    return TwinBench(h1, twins, every, reads)


# --- Evaluation ----------------------------------------------------------------------------------


@dataclass
class Evaluated:
    caught: np.ndarray  # (leaks, cuts): H1 catches the leak at the cut, either direction
    moved: np.ndarray  # (leaks, cuts, K): the twin moves the leak at the cut
    constant: np.ndarray  # (leaks,): the same output at every compared row of H1's cuts
    constant_every: np.ndarray  # (leaks,): the adversary's own flag, over both schedules
    honest_moved: np.ndarray  # (leaks,): the leak's honest twin moved by a future-shuffle twin
    honest_violated: np.ndarray  # (leaks,): ... or differing by a single bit


def evaluate(leaks: Sequence[adv.Leak], tb: TwinBench) -> Evaluated:
    n, c, k = len(leaks), tb.h1.n_cuts, len(tb.twins.directions)
    out = Evaluated(np.zeros((n, c), bool), np.zeros((n, c, k), bool), np.zeros(n, bool), np.zeros(n, bool),
                    np.zeros(n, bool), np.zeros(n, bool))  # fmt: skip
    for i, leak in enumerate(leaks):
        h = adv.evaluate_fast(leak, tb.h1)
        t = adv.evaluate_fast(leak, tb.twins)
        if h.constant != t.constant:
            raise AssertionError(f"leak {leak.id}: the two benches disagree on its original outputs")
        honest = adv.evaluate_fast(leak.honest_twin(), tb.twins)
        out.caught[i] = h.caught.any(axis=1)
        out.moved[i] = t.caught
        out.constant[i] = h.constant
        out.constant_every[i] = adv.evaluate_fast(leak, tb.every).constant
        out.honest_moved[i] = honest.caught.any()
        out.honest_violated[i] = honest.violated.any()
    return out


def evaluate_all(leaks: Sequence[adv.Leak], tb: TwinBench, *, workers: int = 1) -> Evaluated:
    """:func:`evaluate` in contiguous chunks, on ``workers`` threads: the same arrays for any worker count.

    Threads, not processes: a process pool on Windows lost workers to ``WinError 6`` on its queue's semaphore
    while the 13 MB bench was being sent to each, and the whole evaluation is about 2 ms per leak anyway. Each
    chunk reads the shared bench and writes only its own arrays; they are joined in the leaks' order.
    """
    if workers <= 1 or len(leaks) < 2:
        return evaluate(leaks, tb)
    chunk = max(1, math.ceil(len(leaks) / (4 * workers)))
    parts = [list(leaks[s : s + chunk]) for s in range(0, len(leaks), chunk)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda part: evaluate(part, tb), parts))
    return Evaluated(*(np.concatenate([getattr(r, f.name) for r in results]) for f in fields(Evaluated)))


# --- Classification and statistics ----------------------------------------------------------------


def classify(ev: Evaluated) -> np.ndarray:
    """``caught``, or for a miss exactly one of :data:`CLASSES`, in that order of precedence."""
    caught = ev.caught.any(axis=1)
    moved = ev.moved.any(axis=(1, 2))
    return np.where(caught, "caught", np.where(ev.constant, "constant", np.where(moved, "blind-spot", "unchanged")))


def _part(k: int, n: int) -> dict[str, Any]:
    lo, hi = adv.wilson(k, n)
    return {"count": int(k), "of": int(n), "share": (k / n) if n else math.nan, "wilson95": [lo, hi]}


def _tables(leaks: Sequence[adv.Leak], mask: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """The adversary's tag tables (``missed`` of ``n`` leaks with the tag), the tags present in ``mask`` only."""
    return {key: [r for r in adv._tag_table(leaks, mask, key) if r["missed"]] for key in TABLE_KEYS}


def _pattern(row: np.ndarray) -> str:
    return "".join("X" if x else "." for x in row)


def _twin_pattern(moved: np.ndarray) -> str:
    """Per cut, how many of the K twins move the leak (a digit, or + above 9)."""
    return "".join(str(int(m)) if m <= 9 else "+" for m in moved.sum(axis=1))


def summarize(leaks: Sequence[adv.Leak], tb: TwinBench, ev: Evaluated, cfg: TwinsConfig) -> dict[str, Any]:
    """Every number of the report, from the arrays."""
    cls = classify(ev)
    caught = cls == "caught"
    missed = ~caught
    moved_cut = ev.moved.any(axis=2)  # (leaks, cuts)
    moved_any = moved_cut.any(axis=1)
    n_miss = int(missed.sum())
    classes = {name: _part(int((cls == name).sum()), n_miss) for name in CLASSES}
    caught_unmoved = caught & ~moved_any
    per_leak = [
        {
            "id": leak.id,
            "class": str(cls[i]),
            "h1_cuts_caught": int(ev.caught[i].sum()),
            "h1_pattern": _pattern(ev.caught[i]),
            "constant": bool(ev.constant[i]),
            "constant_both_schedules": bool(ev.constant_every[i]),
            "twin_cuts_moved": int(moved_cut[i].sum()),
            "twin_moves": int(ev.moved[i].sum()),
            "twin_pattern": _twin_pattern(ev.moved[i]),
        }
        for i, leak in enumerate(leaks)
    ]

    def listing(mask: np.ndarray, limit: int | None = None) -> list[dict[str, Any]]:
        ids = np.flatnonzero(mask).tolist()
        return [{"id": leaks[i].id, "family": leaks[i].family, "expression": leaks[i].render(), "tags": leaks[i].tag,
                 "twin_pattern": per_leak[i]["twin_pattern"], "h1_pattern": per_leak[i]["h1_pattern"],
                 "constant": bool(ev.constant[i])} for i in (ids if limit is None else ids[:limit])]  # fmt: skip

    return {
        "population": {"n": len(leaks), "caught": int(caught.sum()), "missed": n_miss},
        "classes": classes,
        "misses": {
            "count": n_miss,
            "by_class": {name: _tables(leaks, cls == name) for name in CLASSES},
            "by_tag": _tables(leaks, missed),
            # The adversary's flag: constant over both of its schedules' compared rows (its "constant" misses).
            "constant_both_schedules": int((missed & ev.constant_every).sum()),
            "constant_h1_only": int((missed & ev.constant & ~ev.constant_every).sum()),
            # A constant miss is classed constant first; some twin may still move it (the past holds a value that flips it).
            "constant_but_moved_by_a_twin": int((missed & ev.constant & moved_any).sum()),
        },
        "blind_spots": listing(cls == "blind-spot"),
        "unchanged": listing(cls == "unchanged"),
        "caught_moved_by_a_twin": {
            **_part(int((caught & moved_any).sum()), int(caught.sum())),
            "not_moved": int(caught_unmoved.sum()),
            "not_moved_constant": int((caught_unmoved & ev.constant).sum()),
            "not_moved_by_tag": _tables(leaks, caught_unmoved),
            "not_moved_sample": listing(caught_unmoved, SAMPLE_LIMIT),
        },
        "honest_twins": {
            "evaluated": len(leaks),
            "moved": int(ev.honest_moved.sum()),
            "bit_violations": int(ev.honest_violated.sum()),
            "ids": [leaks[i].id for i in np.flatnonzero(ev.honest_moved | ev.honest_violated).tolist()],
        },
        "per_cut": [
            {**{key: tb.h1.cuts[r][key] for key in ("pair", "k", "index", "open_time", "kind", "mirror")},
             "caught_share": float(ev.caught[:, r].mean()) if len(leaks) else math.nan,
             "moved_share": float(moved_cut[:, r].mean()) if len(leaks) else math.nan,
             "misses_moved": int((moved_cut[:, r] & missed).sum())}
            for r in range(tb.h1.n_cuts)
        ],  # fmt: skip
        "twin_reads": tb.reads,
        "leaks": per_leak,
    }


# --- The adversary's run: digest, configuration and cross-check ----------------------------------------


class Refused(ValueError):
    """An adversary run that cannot be used: a wrong digest, a stopped run or an unreadable configuration."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _tuples(x: Any) -> Any:
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def config_from_json(d: dict[str, Any]) -> adv.AdversaryConfig:
    """The :class:`AdversaryConfig` an ``adversary.json`` records; it must write back to the same JSON exactly."""
    try:
        h1d = dict(d["h1"])
        pert = Perturbation(**h1d.pop("perturbation"))
        h1 = inv.H1Config(**{k: _tuples(v) for k, v in h1d.items()}, perturbation=pert)
        names = [f.name for f in fields(adv.AdversaryConfig) if f.name != "h1"]
        cfg = adv.AdversaryConfig(**{k: _tuples(d[k]) for k in names}, h1=h1)
    except (KeyError, TypeError) as e:
        raise Refused(f"the adversary's configuration cannot be read: {e}") from e
    if _normal(adv._config_json(cfg)) != _normal(d):
        raise Refused("the adversary's configuration does not write back to itself: a field this code does not know")
    return cfg


def load_adversary(folder: Path, sha256: str) -> tuple[dict[str, Any], adv.AdversaryConfig]:
    """``folder/adversary.json``, refused unless its SHA-256 is ``sha256`` and the run went to the end."""
    path = folder / "adversary.json" if folder.is_dir() else folder
    digest = sha256_file(path)
    if digest != sha256.strip().lower():
        raise Refused(f"{path.name}'s SHA-256 is {digest}, not the {sha256} given: refused")
    result = json.loads(path.read_text(encoding="utf-8"))
    if "stopped" in result or "leaks" not in result:
        raise Refused("the adversary run stopped before its rates: it has no misses to classify")
    return result, config_from_json(result["config"])


def cross_check(run: dict[str, Any], digest: str, leaks: Sequence[adv.Leak], tb: TwinBench, ev: Evaluated) -> dict[str, Any]:
    """This module's H1 verdicts against the adversary run's, leak by leak: cuts, expressions, patterns, constants."""
    mine_cuts = _normal([{k: c[k] for k in ("pair", "k", "index", "open_time", "kind", "mirror")} for c in tb.h1.cuts])
    theirs_cuts = _normal([{k: c[k] for k in ("pair", "k", "index", "open_time", "kind", "mirror")} for c in run["cuts"] if c["schedule"] == "h1"])
    records = {r["id"]: r for r in run["leaks"]}
    mismatches = []
    for i, leak in enumerate(leaks):
        r = records.get(leak.id)
        mine = {"expression": leak.render(), "h1_pattern": _pattern(ev.caught[i]), "constant_on_original_rows": bool(ev.constant_every[i])}
        theirs = None if r is None else {k: r[k] for k in mine}
        if theirs != mine:
            mismatches.append({"id": leak.id, "this_run": mine, "adversary": theirs})
    missed_mine = sorted(leak.id for i, leak in enumerate(leaks) if not ev.caught[i].any())
    missed_theirs = sorted(r["id"] for r in run["leaks"] if "X" not in r["h1_pattern"])
    agree = (mine_cuts == theirs_cuts and not mismatches and missed_mine == missed_theirs and len(records) == len(leaks)
             and run["misses"]["count"] == len(missed_mine))  # fmt: skip
    return {
        "sha256": digest,
        "leaks_compared": len(leaks),
        "cuts_equal": mine_cuts == theirs_cuts,
        "misses_equal": missed_mine == missed_theirs,
        "misses": len(missed_mine),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:20],
        "agree": bool(agree),
    }


# --- The run ------------------------------------------------------------------------------------------


def config_json(cfg: TwinsConfig) -> dict[str, Any]:
    return {
        "k": cfg.k,
        "seed": cfg.seed,
        "generator": "numpy.random.default_rng([seed, zlib.crc32(pair), k, j]): k the cut's position in the pair's H1 cuts, j the twin",
        "kept_columns": list(KEPT_COLUMNS),
        "shuffled_columns": list(SHUFFLED_COLUMNS),
        "drawn_from": "rows 0 .. t, the cut's own bar included, uniform with replacement",
        "moved_above": inv.CAUGHT_ABOVE,
        "classes": list(CLASSES),
        "adversary": _normal(adv._config_json(cfg.adversary)),
    }


def run(
    bars_by_pair: dict[str, pl.DataFrame],
    cfg: TwinsConfig,
    *,
    workers: int = 1,
    adversary_run: dict[str, Any] | None = None,
    adversary_digest: str | None = None,
) -> dict[str, Any]:
    """The twins on the given bars. With an adversary run, its misses must be this run's, or nothing is classified."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    leaks = adv.generate(cfg.adversary.n, cfg.adversary.seed)
    tb = build(bars_by_pair, cfg)
    timings["benches"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    ev = evaluate_all(leaks, tb, workers=workers)
    timings["leaks"] = time.perf_counter() - t1
    result: dict[str, Any] = {"config": config_json(cfg), "pairs": list(bars_by_pair)}
    if adversary_run is not None:
        result["cross_check"] = cross_check(adversary_run, adversary_digest or "", leaks, tb, ev)
        if not result["cross_check"]["agree"]:
            result["stopped"] = "this run's H1 verdicts differ from the adversary run's: nothing is classified"
            result["_timings"] = timings
            return result
    result.update(summarize(leaks, tb, ev, cfg))
    result["_timings"] = timings
    return result


def ok(result: dict[str, Any]) -> bool:
    """Sound: no stop, the cross-check (if any) agrees, and no honest leak moved or differed by a bit."""
    honest = result.get("honest_twins", {})
    return bool("stopped" not in result and result.get("cross_check", {"agree": True})["agree"]
                and honest.get("moved") == 0 and honest.get("bit_violations") == 0)  # fmt: skip


# --- Report -----------------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f} %"


def _ci(s: dict[str, Any]) -> str:
    lo, hi = s["wilson95"]
    return f"[{_pct(lo)}, {_pct(hi)}]"


def _tag_lines(tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    return [f"- **{key}**: " + ("; ".join(f"{r['value']}: {r['missed']}/{r['n']}" for r in rows) or "none") + "." for key, rows in tables.items()]


def render_md(result: dict[str, Any]) -> str:
    cfg = result["config"]
    a = cfg["adversary"]
    lines = [
        "# H1's blind spots: the adversary's misses and their future-shuffle twins",
        "",
        "Descriptive: this classifies the misses of one seeded generator of leaks (`olsprime.bench.adversary`). "
        "H1's registered rule does not change.",
        "",
        f"Population: {a['n']} leaks, seed {a['seed']}, regenerated from the adversary's configuration. Pairs "
        f"{', '.join(result['pairs'])} {a['h1']['interval']}, H1's cuts (seed {a['h1']['cut_seed']}), test months "
        f"{a['h1']['start'][0]}-{a['h1']['start'][1]:02d} -> {a['h1']['end'][0]}-{a['h1']['end'][1]:02d}.",
        "",
        f"**The twins.** K = {cfg['k']} per cut, seed {cfg['seed']}. In a twin, each bar after the cut t keeps "
        f"{', '.join(f'`{c}`' for c in cfg['kept_columns'])} and takes {', '.join(f'`{c}`' for c in cfg['shuffled_columns'])} "
        f"of one bar drawn from {cfg['drawn_from']}. One generator per (pair, cut, twin): `{cfg['generator']}`. "
        f"A twin **moves** a leak at a cut when, at a compared row (t-10..t, scored by H1), one of its outputs differs "
        f"from the original by more than {cfg['moved_above']:g}, turns missing or stops being missing: H1's catch rule.",
        "",
    ]
    if "cross_check" in result:
        x = result["cross_check"]
        lines += [
            "## Cross-check with the adversary run",
            "",
            f"`adversary.json` SHA-256 `{x['sha256']}`. {x['leaks_compared']} leaks compared: H1 cuts equal: {x['cuts_equal']}; "
            f"expression, H1 pattern and constant flag mismatches: {x['mismatches']}; misses equal: {x['misses_equal']} "
            f"({x['misses']}). **{'The runs agree.' if x['agree'] else 'THE RUNS DISAGREE: nothing is classified.'}**",
            "",
        ]
        for m in x["first_mismatches"]:
            lines.append(f"- leak {m['id']}: this run {m['this_run']}, adversary {m['adversary']}")
        if x["first_mismatches"]:
            lines.append("")
    if "stopped" in result:
        lines += [f"**Stopped:** {result['stopped']}."]
        return "\n".join(lines) + "\n"
    p, cl, ms, ht = result["population"], result["classes"], result["misses"], result["honest_twins"]
    lines += [
        "## Honest leaks under the twins",
        "",
        f"Every leak's honest twin (the same expression moved into the past) through the same {cfg['k']} future-shuffle "
        f"twins at every cut: **{ht['moved']} moved, {ht['bit_violations']} differing by a single bit** (each must be 0; "
        "one is a bench bug).",
        "",
        "## The misses, in three classes",
        "",
        f"{p['missed']} of {p['n']} leaks are caught at none of H1's cuts. Each falls in exactly one class, in this order: "
        "**constant** at the compared rows of H1's cuts on the original bars; **unchanged** under every twin at every cut, "
        "so on these bars it does not depend on the future; **blind spot**, moved by some twin yet not caught by H1.",
        "",
        "| class | misses | share of the misses | 95 % Wilson |",
        "|---|---|---|---|",
        *(f"| {name} | {cl[name]['count']} | {_pct(cl[name]['share'])} | {_ci(cl[name])} |" for name in CLASSES),
        "",
        f"The adversary's own constant flag reads the compared rows of both its schedules: {ms['constant_both_schedules']} "
        f"misses are constant there, and {ms['constant_h1_only']} more are constant at H1's cuts only. "
        f"Constant misses still moved by some twin: {ms['constant_but_moved_by_a_twin']}. A past bar holds a value that "
        "changes them, which H1's perturbation did not reach; they stay in the constant class.",
        "",
        "Counts by tag over the misses (missed of n leaks with the tag):",
        "",
        *_tag_lines(ms["by_tag"]),
        "",
    ]
    for name in CLASSES:
        lines += [f"### {name}", "", *_tag_lines(ms["by_class"][name]), ""]
    lines += ["## True blind spots", ""]
    if result["blind_spots"]:
        lines += ["Twin pattern: per H1 cut, in H1's order, how many of the K twins move the leak.", ""]
        lines += [f"- #{b['id']} ({b['family']}; {b['tags']['quantity']}, {b['tags']['operation']}, {b['tags']['offset']}, "
                  f"position {b['tags']['position']}; twins `{b['twin_pattern']}`): `{b['expression']}`" for b in result["blind_spots"]]  # fmt: skip
    else:
        lines.append("None.")
    lines += ["", "## Unchanged under every twin", ""]
    lines += [f"- #{u['id']} ({u['family']}): `{u['expression']}`" for u in result["unchanged"]] or ["None."]
    s = result["caught_moved_by_a_twin"]
    lines += [
        "",
        "## Sanity check: the caught leaks",
        "",
        f"Of the {s['of']} caught leaks, some twin moves **{s['count']} ({_pct(s['share'])}, Wilson {_ci(s)})**. "
        f"{s['not_moved']} are never moved, {s['not_moved_constant']} of them constant at the compared rows: H1's "
        "perturbation pushes bar t+1 past every bar of the window, to values no past bar holds, so a twin, which "
        "only replays the past, cannot reach them. By tag:",
        "",
        *_tag_lines(s["not_moved_by_tag"]),
        "",
        f"The first {len(s['not_moved_sample'])} of them:",
        "",
        *(f"- #{u['id']} (H1 `{u['h1_pattern']}`{', constant' if u['constant'] else ''}): `{u['expression']}`" for u in s["not_moved_sample"]),
        "",
        "## Per cut",
        "",
        "| pair | # | row | date | kind | parity | caught by H1 | moved by a twin | misses moved |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in result["per_cut"]:
        lines.append(f"| {c['pair']} | {c['k'] + 1} | {c['index']} | {c['open_time'][:10]} | {c['kind']} | "
                     f"{'mirror' if c['mirror'] else 'plain'} | {_pct(c['caught_share'])} | {_pct(c['moved_share'])} | {c['misses_moved']} |")  # fmt: skip
    lines += ["", "## What the twins read", "", "Source rows of bars t+1..t+10 in each twin (the only later bars a leak reads):", ""]
    for r in result["twin_reads"]:
        lines.append(f"- {r['pair']} cut {r['k'] + 1} (row {r['index']}): " + "; ".join(" ".join(str(x) for x in s_) for s_ in r["sources"]))
    return "\n".join(lines) + "\n"


# --- CLI --------------------------------------------------------------------------------------------------


def _default_out() -> Path:
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-twins"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def synthetic_pairs() -> dict[str, pl.DataFrame]:
    """The synthetic bars of ``--synthetic``: no data is read."""
    from olsprime.bench.synthetic import synthetic_bars

    return {pair: synthetic_bars(pair, days=SYNTHETIC_DAYS, seed=seed) for pair, seed in SYNTHETIC_PAIRS.items()}


def write(out: Path, result: dict[str, Any], config: dict[str, Any], run_info: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "twins.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "twins.md").write_text(render_md(result), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "run.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.twins", description="H1's blind spots: the adversary's misses and their future-shuffle twins.")
    parser.add_argument("--adversary", type=Path, default=None, help="an adversary run folder (or its adversary.json): its config, and the cross-check")
    parser.add_argument("--sha256", default=None, help="the SHA-256 of that adversary.json (required with --adversary)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <runs>/<UTC date>-twins, suffixed if taken)")
    parser.add_argument("--workers", type=int, default=4, help="threads for the leaks (results do not depend on it)")
    parser.add_argument("--synthetic", action="store_true", help="synthetic bars, for tests and smoke runs: reads no data")
    parser.add_argument("--n", type=int, default=None, help="number of leaks, with --synthetic and without --adversary only")
    args = parser.parse_args(argv)
    if (args.adversary is None) != (args.sha256 is None):
        parser.error("--adversary and --sha256 go together")
    if args.n is not None and (not args.synthetic or args.adversary is not None):
        parser.error("--n is for --synthetic runs without --adversary: an official run takes the adversary's n")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    adversary_run, digest = None, None
    if args.adversary is not None:
        try:
            adversary_run, adversary_cfg = load_adversary(args.adversary, args.sha256)
        except Refused as e:
            print(f"twins REFUSED: {e}")
            return 2
        digest = args.sha256.strip().lower()
    elif args.synthetic:
        adversary_cfg = replace(adv.AdversaryConfig(), h1=SYNTHETIC_H1, **({"n": args.n} if args.n is not None else {}))
    else:
        adversary_cfg = adv.AdversaryConfig()
    cfg = TwinsConfig(adversary=adversary_cfg)
    if args.synthetic:
        prov = {**provenance.record({}), "bars": "synthetic", "synthetic_pairs": SYNTHETIC_PAIRS, "synthetic_days": SYNTHETIC_DAYS}
        bars = synthetic_pairs()
    else:
        prov = provenance.record(provenance.current_manifest())  # before any data is read, as in H1
        bars = {pair: adv._load_development(pair, cfg.adversary.h1) for pair in cfg.adversary.h1.pairs}
    result = run(bars, cfg, workers=args.workers, adversary_run=adversary_run, adversary_digest=digest)
    timings = result.pop("_timings")
    out = args.out or _default_out()
    config = {**config_json(cfg), "provenance": prov, "adversary_run": None if args.adversary is None else str(args.adversary),
              "adversary_sha256": digest}  # fmt: skip
    write(out, result, config, {"finished_at": datetime.now(UTC).isoformat(), "seconds": round(time.perf_counter() - started, 1),
                                "stages": {k: round(v, 1) for k, v in timings.items()}, "workers": args.workers, "n": cfg.adversary.n})  # fmt: skip
    if "stopped" in result:
        print(f"twins STOPPED: {result['stopped']}; written to {out}")
        return 1
    cl = result["classes"]
    print(
        f"twins: {result['population']['missed']} misses of {result['population']['n']}: "
        + ", ".join(f"{name} {cl[name]['count']}" for name in CLASSES)
        + f"; caught leaks moved by a twin {100 * result['caught_moved_by_a_twin']['share']:.1f} %; honest leaks moved "
        f"{result['honest_twins']['moved']}, bit violations {result['honest_twins']['bit_violations']}; written to {out}"
    )
    return 0 if ok(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
