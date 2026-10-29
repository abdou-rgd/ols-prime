"""The panel H1: future invariance of every Matrix arm on a sub-panel (``docs/PROTOCOL.md``, iteration 3).

Usage::

    uv run --no-sync python -m olsprime.bench.panel_invariance [--out DIR]

**Set-up.**

- The sub-panel: the 10 members with the most member months in the development period
  (:func:`olsprime.features.panel.h1_subpanel_members`). The arms run on the panel of these
  members alone, rebuilt with :func:`~olsprime.features.panel.pair_rows` and
  :func:`~olsprime.features.panel.assemble`, exactly as :func:`~olsprime.features.panel.hourly_panel`
  builds it.
- 8 cut hours, drawn with a fixed seed among the hours of the test months that have a row: one
  at a month's first hour, one at a month's last hour, six at random.
- At each cut, every member's bars after the cut are perturbed by H1's own perturbation
  (:func:`olsprime.bench.perturb.perturb_after`), at that pair's own row for the cut (its last bar
  opening at or before the cut hour), in both directions (the future x3 and /3), with H1's seeds
  and mirror scheme (:func:`olsprime.bench.invariance.perturbation_at`). A member on which
  ``perturb_after`` raises is left out of that cut, unperturbed, and counted: one with no bar after
  the cut (delisted, such as MATICUSDT after 2024-09), or one with no bar at or before it (not yet
  listed; all its bars lie after the cut and no output at or before the cut can read them).
- The panel is rebuilt from the perturbed bars, and the arms are run on it.

**Comparison.** For a cut at hour c of month M, ``until`` is the cut bar's ``known_at``
(``c + 1 h - 1 ms``). Compared bit for bit, between the run on the original bars and the run on
the perturbed bars:

- every output (the three probabilities and the fallback flag) of every arm at every test row
  decided at or before ``until``; for ``softmax-1y`` and ``knn-1y``, the rows of month M;
- every fit-time choice made at or before ``until``: per monthly fit, sigma^2, the climatology,
  the selection-row count, every candidate's score (the kernel candidates' at M only), the
  selected window, rho, tau, k and alpha, and alpha's scores; per decision of
  ``matrix-adaptive``, its row count, scores and rho; every ``matrix-cusum`` fit (time, sigma^2,
  window, rho, scores, s) and every day of its CUSUM that entered at or before ``until``.

An honest arm must not differ by a single bit. A planted leak is caught at a cut only where a
compared value moves by more than :data:`olsprime.bench.invariance.CAUGHT_ABOVE`, turns missing or
stops being missing.

**What is recomputed, and why it is enough.** The run on the original bars computes everything
once. Each perturbed run (:class:`olsprime.models.matrix.Scope`) recomputes:

- the whole sub-panel (features and labels of every member, every month) and the whole memory;
- sigma^2 and the climatology of every month;
- the ridge record of every complete row decided at or before ``until`` (every month from 2017-11);
- the kernel record of the rows decided in ``(M - 365 d, until]``;
- every monthly fit of the ridge arms from 2019-01 to M, their outputs and ``matrix-adaptive``'s
  choices up to ``until``; the kernel arms' fit at M and their outputs in M;
- ``matrix-cusum`` from its first fit to ``until``: every fit, every day of its CUSUM and every output.

Left out: the kernel record before ``M - 365 d`` and the kernel arms' fits and outputs before M,
and everything decided after ``until``. An honest arm's compared outputs cannot depend on them:
the kernel arms' outputs in M read the memory at their own decision times and their fit at M,
which reads the record only at its selection rows, decided in ``(M - 365 d, M)``; and nothing
decided after ``until`` is compared. The rows decided after ``until`` are not forecast at all in
the perturbed run, so an arm that read another row's forecast from the future would fail there.

**Planted online leaks** (:data:`LEAKS`), each with its predicted cuts computed from the original
panel (:func:`predicted`): entries stamped at their decision time; a window that ends one hour
after the decision; sigma fitted on the whole test month (never at a month's last hour, where every
feature of the month is known); a fit-time selection without the purge; a kernel whose candidates
reach one label horizon past the decision.

**A limit.** Only the sub-panel's bars are perturbed and the arms run on the sub-panel alone. An
input built from other members' bars would not move. No arm has one: every input of an arm is a
row of the panel it is given. Keep it so.

Outputs: ``h1-panel.json``, ``h1-panel.md`` (both with provenance) and ``run.json`` to
``runs/<UTC date>-h1-panel/`` (suffixed if taken) or ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from olsprime.bench import provenance
from olsprime.bench.invariance import CAUGHT_ABOVE, H1Config, perturbation_at
from olsprime.bench.perturb import Perturbation, perturb_after
from olsprime.features.panel import FIRST_MONTH, LAST_MONTH, assemble, h1_subpanel_members, member_months, pair_rows
from olsprime.models.matrix import (
    ARMS,
    DAY_MS,
    HOUR_MS,
    KERNEL_ARMS,
    ArmsResult,
    MatrixConfig,
    MatrixSystem,
    PanelArrays,
    Scope,
    fold_months_ms,
    iso,
    next_month_ms,
    seqsum,
)
from olsprime.paths import runs_dir

FIRST_HOUR = "first-hour-of-month"
LAST_HOUR = "last-hour-of-month"
RANDOM = "random-hour"
DIRECTIONS = {3.0: "x3", 1.0 / 3.0: "/3"}


@dataclass(frozen=True)
class PanelH1Config:
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    subpanel_size: int = 10
    n_cuts: int = 8
    cut_seed: int = 20261029
    perturbation_seed: int = 7  # H1's
    perturbation: Perturbation = Perturbation()
    level_shifts: tuple[float, ...] = (3.0, 1.0 / 3.0)
    first_month: tuple[int, int] = (FIRST_MONTH.year, FIRST_MONTH.month)
    last_month: tuple[int, int] = (LAST_MONTH.year, LAST_MONTH.month)

    def h1(self) -> H1Config:
        return H1Config(perturbation_seed=self.perturbation_seed, perturbation=self.perturbation, level_shifts=self.level_shifts)


@dataclass(frozen=True)
class PanelCut:
    hour_ms: int  # the open of the cut hour
    kind: str

    @property
    def until_ms(self) -> int:
        """The cut bar's known_at: outputs decided at or before it are compared."""
        return self.hour_ms + HOUR_MS - 1

    @property
    def month_ms(self) -> int:
        d = datetime.fromtimestamp(self.hour_ms / 1000, tz=UTC)
        return int(datetime(d.year, d.month, 1, tzinfo=UTC).timestamp() * 1000)


# --- !!! Planted online leaks, on purpose. NOT MODELS. Never report one as a result. !!! -----


class StampedAtDecision(MatrixSystem):
    """!!! LEAKS: every entry is stamped with its decision time instead of its label_known_at. !!!"""

    name = "entries-stamped-at-decision"

    def entry_times(self, rows: np.ndarray) -> np.ndarray:
        return self.data.decision_ms[rows]


class WindowPastDecision(MatrixSystem):
    """!!! LEAKS: the ridge window ends one hour after the decision. !!!"""

    name = "window-ends-one-hour-late"

    def ridge_as_of(self, rows: np.ndarray) -> np.ndarray:
        return self.data.decision_ms[rows] + HOUR_MS


class MonthSigma(MatrixSystem):
    """!!! LEAKS: sigma^2 of month M also takes the features of every row of month M (transductive). !!!"""

    name = "sigma-on-the-whole-test-month"

    def variance_at(self, fit_ms: int, month_ms: int | None) -> tuple[np.ndarray | None, int]:
        var, n = super().variance_at(fit_ms, month_ms)
        if month_ms is None or var is None:
            return var, n
        _, end = self.matrix.window_range(int(fit_ms - self.cfg.horizon_ms), None)
        st = self.matrix.centred_stats(0, end)
        x = self.data.x[(self.data.month_ms == month_ms) & self.data.features_ok]
        if not x.shape[0]:
            return var, n
        n1, n2 = float(st.n), float(x.shape[0])
        m2 = seqsum(x, axis=0) / n2
        ss2 = seqsum((x - m2) ** 2, axis=0)
        delta = m2 - st.mean_x
        total = np.diag(st.m_xx) + ss2 + (n1 * n2 / (n1 + n2)) * delta * delta
        return np.ascontiguousarray(total / (n1 + n2)), n


class Unpurged(MatrixSystem):
    """!!! LEAKS: the fit-time selection reads rows whose labels are not yet known (no purge, no embargo). !!!"""

    name = "selection-without-purge"

    def selection_mask(self, fit_ms: int, rec: Any) -> np.ndarray:
        dec = self.data.decision_ms[rec.rows]
        return (dec > fit_ms - self.cfg.selection_days * DAY_MS) & (dec < fit_ms)


class KernelPastDecision(MatrixSystem):
    """!!! LEAKS: the kernel candidates reach one label horizon past the decision. !!!"""

    name = "kernel-reaches-one-horizon-late"

    def kernel_as_of(self, rows: np.ndarray) -> np.ndarray:
        return self.data.decision_ms[rows] + self.cfg.horizon_ms


# --- Predictions of the planted leaks, from the original panel -----------------------------


def _test_rows_until(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut, *, month_only: bool) -> np.ndarray:
    months = fold_months_ms(cfg)
    m = data.complete & np.isin(data.month_ms, months) & (data.decision_ms <= cut.until_ms)
    if month_only:
        m &= data.month_ms == cut.month_ms
    return np.flatnonzero(m)


def predict_stamped(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut) -> bool:
    """An entry decided by the cut whose label is known after it (read at once when stamped at its decision)."""
    c = data.complete
    return bool(np.any(c & (data.decision_ms <= cut.until_ms) & (data.lka_ms > cut.until_ms)))


def _late_entry(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut, reach: int, *, month_only: bool) -> bool:
    rows = _test_rows_until(data, cfg, cut, month_only=month_only)
    if not rows.size:
        return False
    d_max = int(data.decision_ms[rows].max())
    lka = data.lka_ms[data.complete]
    return bool(np.any((lka > cut.until_ms) & (lka <= d_max + reach)))


def predict_window(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut) -> bool:
    """An entry known after the cut but within an hour of a compared decision."""
    return _late_entry(data, cfg, cut, HOUR_MS, month_only=False)


def predict_sigma(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut) -> bool:
    """A row of the cut's month that opens after the cut: its features move sigma^2 of the month."""
    return bool(np.any((data.month_ms == cut.month_ms) & data.features_ok & (data.open_ms > cut.hour_ms)))


def predict_unpurged(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut) -> bool:
    """A fit at or before the cut whose unpurged selection rows include a label known after the cut."""
    span = cfg.selection_days * DAY_MS
    for f in fold_months_ms(cfg):
        if f > cut.until_ms:
            break
        rows = data.complete & (data.decision_ms > f - span) & (data.decision_ms < f)
        if np.any(rows & (data.lka_ms > cut.until_ms)):
            return True
    return False


def predict_kernel(data: PanelArrays, cfg: MatrixConfig, cut: PanelCut) -> bool:
    """An entry known after the cut but within one horizon of a compared kernel decision."""
    return _late_entry(data, cfg, cut, cfg.horizon_ms, month_only=True)


@dataclass(frozen=True)
class SystemSpec:
    name: str
    cls: type[MatrixSystem]
    arms: tuple[str, ...]
    predict: Callable[[PanelArrays, MatrixConfig, PanelCut], bool] | None = None  # None: honest, never flagged

    @property
    def honest(self) -> bool:
        return self.predict is None


HONEST = SystemSpec("honest", MatrixSystem, ARMS)
LEAKS: tuple[SystemSpec, ...] = (
    SystemSpec(StampedAtDecision.name, StampedAtDecision, ("matrix-select",), predict_stamped),
    SystemSpec(WindowPastDecision.name, WindowPastDecision, ("matrix-select",), predict_window),
    SystemSpec(MonthSigma.name, MonthSigma, ("matrix-select",), predict_sigma),
    SystemSpec(Unpurged.name, Unpurged, ("matrix-select",), predict_unpurged),
    SystemSpec(KernelPastDecision.name, KernelPastDecision, ("softmax-1y", "knn-1y"), predict_kernel),
)


# --- Cuts and the perturbed panel -----------------------------------------------------------


def choose_cuts(data: PanelArrays, cfg: PanelH1Config) -> list[PanelCut]:
    """One cut at a test month's first hour, one at a test month's last hour, the rest random hours; sorted by time."""
    months = [m for m in fold_months_ms(cfg.matrix) if np.any(data.month_ms == m)]
    if len(months) < 2:
        raise ValueError("the sub-panel needs rows in at least two test months")
    hours = np.unique(data.open_ms[np.isin(data.month_ms, months)])
    rng = np.random.default_rng(cfg.cut_seed)
    first = months[int(rng.integers(len(months)))]
    last = next_month_ms(months[int(rng.integers(len(months)))]) - HOUR_MS
    cuts = {first: FIRST_HOUR, last: LAST_HOUR}
    if len(cuts) != 2:
        raise ValueError("the first-hour and last-hour cuts coincide")
    pool = hours[~np.isin(hours, list(cuts))]
    for h in rng.choice(pool, size=cfg.n_cuts - len(cuts), replace=False):
        cuts[int(h)] = RANDOM
    return [PanelCut(h, kind) for h, kind in sorted(cuts.items())]


def cut_row(bars: pl.DataFrame, hour_ms: int) -> int:
    """The pair's own row for the cut: its last bar opening at or before the cut hour (-1 if none)."""
    open_ms = bars["open_time"].dt.epoch("ms").to_numpy()
    return int(np.searchsorted(open_ms, hour_ms, side="right")) - 1


def perturbed_bars(bars: dict[str, pl.DataFrame], cut: PanelCut, k: int, j: int, cfg: PanelH1Config) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Every member's bars after the cut perturbed in direction j at the k-th cut; the members left out, with why."""
    seed, config = perturbation_at(cfg.h1(), k, j)
    out, left = {}, {}
    for symbol, b in bars.items():
        i = cut_row(b, cut.hour_ms)
        if i < 0:
            left[symbol] = "no bar at or before the cut"
            out[symbol] = b
        elif i >= b.height - 1:
            left[symbol] = "no bar after the cut"
            out[symbol] = b
        else:
            out[symbol] = perturb_after(b, i, seed=seed, config=config)
    return out, left


def build_panel(bars: dict[str, pl.DataFrame], months: dict[str, list[datetime]]) -> pl.DataFrame:
    return assemble(pair_rows(bars[s], months[s]) for s in sorted(bars))


# --- Comparison -----------------------------------------------------------------------------


def compared_items(res: ArmsResult, data: PanelArrays, cut: PanelCut, n_ridge: int) -> dict[str, np.ndarray]:
    """Every compared value of a run at a cut, as float arrays keyed by name (see the module docstring)."""
    until, month = cut.until_ms, cut.month_ms
    items: dict[str, np.ndarray] = {}
    dec, sym, mon = data.decision_ms[res.rows], data.symbol[res.rows], data.month_ms[res.rows]
    mask = dec <= until
    items["rows"] = np.concatenate([dec[mask], sym[mask]]).astype(np.float64)
    for arm, p in res.probs.items():
        m = mask & (mon == month) if arm in KERNEL_ARMS else mask
        items[f"output:{arm}"] = p[m].ravel()
        items[f"fallback:{arm}"] = res.fallback[arm][m].astype(np.float64)
        items[f"status:{arm}"] = res.status[arm][m].astype(np.float64)
    kernel = bool(KERNEL_ARMS & set(res.probs))
    for fit in res.fits:
        if fit["month_ms"] > until:
            continue
        pre = f"fit:{iso(fit['month_ms'])[:7]}:"
        items[pre + "n_selection"] = np.array([fit["n_selection"], fit["default"]], dtype=np.float64)
        items[pre + "var"] = np.asarray(fit["var"], dtype=np.float64)
        items[pre + "clim"] = np.asarray(fit["clim"], dtype=np.float64)
        items[pre + "scores:ridge"] = np.asarray(fit["scores"][:n_ridge], dtype=np.float64)
        items[pre + "choices:ridge"] = np.array([*fit["select"], *fit["rho_by_window"], fit["rho_1y"]], dtype=np.float64)
        items[pre + "alpha"] = np.asarray(fit["alpha"], dtype=np.float64)
        items[pre + "alpha_scores"] = np.zeros(0) if fit["alpha_scores"] is None else np.asarray(fit["alpha_scores"])
        if kernel and fit["month_ms"] == month:
            items[pre + "scores:kernel"] = np.asarray(fit["scores"][n_ridge:], dtype=np.float64)
            items[pre + "choices:kernel"] = np.array([fit["tau"], fit["k"]], dtype=np.float64)
    ad = res.adaptive
    if "matrix-adaptive" in res.probs:
        m = ad["decision_ms"] <= until
        for key in ("decision_ms", "count", "choice", "default"):
            items[f"adaptive:{key}"] = ad[key][m].astype(np.float64)
        items["adaptive:scores"] = ad["scores"][m].ravel()
    if res.cusum:
        for i, fit in enumerate(f for f in res.cusum["fits"] if f["fit_ms"] <= until):
            pre = f"cusum-fit:{i}:"
            items[pre + "choice"] = np.array([fit["fit_ms"], fit["window"], fit["rho"], fit["n_selection"], fit["default"],
                                              fit["s"], fit["s_days"], fit["s_default"]], dtype=np.float64)  # fmt: skip
            items[pre + "var"] = np.asarray(fit["var"], dtype=np.float64)
            items[pre + "scores"] = np.asarray(fit["scores"], dtype=np.float64)
        path = res.cusum["path"]
        m = np.asarray(path["enters_ms"]) <= until
        for key, v in path.items():
            items[f"cusum-day:{key}"] = np.asarray(v, dtype=np.float64)[m] if m.size else np.zeros(0)
    return items


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    """Bit-for-bit comparison of two item sets: the differing values, the caught ones (> CAUGHT_ABOVE), the largest move."""
    differing: dict[str, int] = {}
    caught: dict[str, int] = {}
    largest = 0.0
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if x is None or y is None or x.shape != y.shape:
            size = max(0 if x is None else x.size, 0 if y is None else y.size, 1)
            differing[key], caught[key], largest = size, size, math.inf
            continue
        bits = np.ascontiguousarray(x).view(np.uint64) != np.ascontiguousarray(y).view(np.uint64)
        if not bits.any():
            continue
        differing[key] = int(bits.sum())
        both = ~np.isnan(x) & ~np.isnan(y)
        move = np.abs(x[both] - y[both])
        above = int((move > CAUGHT_ABOVE).sum() + (np.isnan(x) != np.isnan(y)).sum())
        if above:
            caught[key] = above
        largest = max(largest, float(move.max(initial=0.0)))
    return {"violations": int(sum(differing.values())), "caught": int(sum(caught.values())), "largest_change": largest,
            "differing_items": dict(list(differing.items())[:20]), "caught_items": dict(list(caught.items())[:20])}  # fmt: skip


def scope_at(cut: PanelCut, cfg: MatrixConfig, arms: tuple[str, ...]) -> Scope:
    return Scope(until_ms=cut.until_ms, kernel_from_ms=cut.month_ms - cfg.selection_days * DAY_MS, kernel_months=(cut.month_ms,), arms=arms)


def run_system(
    spec: SystemSpec,
    bars: dict[str, pl.DataFrame],
    months: dict[str, list[datetime]],
    cuts: Sequence[PanelCut],
    cfg: PanelH1Config,
    *,
    original_panel: pl.DataFrame | None = None,
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """One system (honest or a leak) at every cut and in both directions: records and the verdict."""
    panel = original_panel if original_panel is not None else build_panel(bars, months)
    started = time.perf_counter()
    system = spec.cls(panel, cfg.matrix)
    original = system.run(Scope(arms=spec.arms))
    data0 = system.data
    log(f"{spec.name}: original run {time.perf_counter() - started:.1f} s")
    n_ridge = system.n_ridge
    records = []
    for k, cut in enumerate(cuts):
        base = compared_items(original, data0, cut, n_ridge)
        by_direction, left_out = {}, {}
        for j, shift in enumerate(cfg.level_shifts):
            t = time.perf_counter()
            pb, left = perturbed_bars(bars, cut, k, j, cfg)
            left_out = left
            sys_p = spec.cls(build_panel(pb, months), cfg.matrix)
            res = sys_p.run(scope_at(cut, cfg.matrix, spec.arms))
            by_direction[DIRECTIONS.get(shift, f"x{shift:g}")] = compare(base, compared_items(res, sys_p.data, cut, n_ridge))
            log(f"{spec.name}: cut {k + 1} {DIRECTIONS.get(shift)} {time.perf_counter() - t:.1f} s")
        records.append(
            {
                "cut": k + 1,
                "hour": iso(cut.hour_ms),
                "kind": cut.kind,
                "predicted": None if spec.honest else bool(spec.predict(data0, cfg.matrix, cut)),
                "left_out": left_out,
                "compared_values": int(sum(v.size for v in base.values())),
                "violations": int(sum(r["violations"] for r in by_direction.values())),
                "caught": int(sum(r["caught"] for r in by_direction.values())),
                "by_direction": by_direction,
            }
        )
    return {"name": spec.name, "arms": list(spec.arms), "honest": spec.honest, "records": records, "verdict": verdict(spec, records)}


def verdict(spec: SystemSpec, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if spec.honest:
        flagged = [r["violations"] > 0 for r in records]
        return {"ok": not any(flagged), "pattern": "".join("X" if f else "." for f in flagged), "predicted_pattern": "." * len(records),
                "violations": int(sum(r["violations"] for r in records))}  # fmt: skip
    flagged = [r["caught"] > 0 for r in records]
    predicted = [bool(r["predicted"]) for r in records]
    smallest = min((max(d["largest_change"] for d in r["by_direction"].values()) for r, f in zip(records, flagged, strict=True) if f), default=None)
    smallest = None if smallest is None or not math.isfinite(smallest) else smallest
    return {"ok": flagged == predicted, "pattern": "".join("X" if f else "." for f in flagged),
            "predicted_pattern": "".join("X" if p else "." for p in predicted), "smallest_catch": smallest}  # fmt: skip


def run_panel_h1(
    bars: dict[str, pl.DataFrame],
    months: dict[str, list[datetime]],
    cfg: PanelH1Config,
    *,
    specs: Sequence[SystemSpec] = (HONEST, *LEAKS),
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    panel = build_panel(bars, months)
    data = MatrixSystem(panel, cfg.matrix).data
    cuts = choose_cuts(data, cfg)
    systems = [run_system(s, bars, months, cuts, cfg, original_panel=panel, log=log) for s in specs]
    honest = [s for s in systems if s["honest"]]
    leaks = [s for s in systems if not s["honest"]]
    left = systems[0]["records"] if systems else []
    return {
        "config": _config_json(cfg),
        "members": sorted(bars),
        "cuts": [{"cut": k + 1, "hour": iso(c.hour_ms), "until": iso(c.until_ms), "kind": c.kind, "month": iso(c.month_ms)[:7]} for k, c in enumerate(cuts)],
        "left_out": {str(r["cut"]): r["left_out"] for r in left},
        "systems": systems,
        "h1": {
            "honest_violations": int(sum(s["verdict"]["violations"] for s in honest)),
            "leaks_off_pattern": [s["name"] for s in leaks if not s["verdict"]["ok"]],
            "holds": bool(all(s["verdict"]["ok"] for s in systems)),
        },
    }


def _config_json(cfg: PanelH1Config) -> dict[str, Any]:
    out = asdict(cfg)
    out["leaks"] = [{"name": s.name, "arms": list(s.arms), "predict": s.predict.__doc__} for s in LEAKS]
    return out


# --- Output ---------------------------------------------------------------------------------


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


def render_md(result: dict[str, Any]) -> str:
    h1 = result["h1"]
    lines = [
        "# The panel H1: future invariance of the Matrix arms",
        "",
        f"**Verdict: the panel H1 {'holds' if h1['holds'] else 'FAILS'}.** Honest violations: {h1['honest_violations']}. "
        f"Planted leaks off their predicted pattern: {', '.join(h1['leaks_off_pattern']) or 'none'}.",
        "",
        f"Sub-panel ({len(result['members'])} members): {', '.join(result['members'])}. Each cut perturbs every member's bars "
        "after the cut hour in both directions (x3 and /3) with H1's perturbation, rebuilds the panel and re-runs the arms. "
        "Every output and fit-time choice decided at or before the cut is compared bit for bit (see the module docstring "
        f"for the list and for what is recomputed). A leak is caught at a cut only where a value moves by more than {CAUGHT_ABOVE:g}.",
        "",
        "**A limit.** Only the sub-panel's bars are perturbed, and the arms run on the sub-panel alone. An input built from "
        "other members' bars would not move. No arm has one: every input of an arm is a row of the panel it is given.",
        "",
        "## Cuts",
        "",
        "| cut | hour (UTC) | kind | members left out |",
        "|---|---|---|---|",
    ]
    for c in result["cuts"]:
        left = result["left_out"].get(str(c["cut"]), {})
        text = "; ".join(f"{s} ({why})" for s, why in sorted(left.items())) or "none"
        lines.append(f"| {c['cut']} | {c['hour'][:16]} | {c['kind']} | {len(left)}: {text} |")
    lines += ["", "## Systems", "", "| system | arms | pattern | predicted | ok | violations or smallest catch |", "|---|---|---|---|---|---|"]
    for s in result["systems"]:
        v = s["verdict"]
        tail = str(v["violations"]) if s["honest"] else ("-" if v.get("smallest_catch") is None else f"{v['smallest_catch']:.2g}")
        lines.append(f"| {s['name']} | {', '.join(s['arms'])} | `{v['pattern']}` | `{v['predicted_pattern']}` | {'yes' if v['ok'] else 'NO'} | {tail} |")
    lines += ["", "Pattern: X flagged, . not flagged, cut by cut. An honest system is flagged by one differing bit; a leak by a catch.", ""]
    for s in result["systems"]:
        lines += [f"### {s['name']}", "", "| cut | compared values | violations | caught | largest change | items |", "|---|---|---|---|---|---|"]
        for r in s["records"]:
            items = sorted({k for d in r["by_direction"].values() for k in d["differing_items"]})
            moves = [math.inf if d["largest_change"] is None else d["largest_change"] for d in r["by_direction"].values()]
            lines.append(f"| {r['cut']} | {r['compared_values']} | {r['violations']} | {r['caught']} | {max(moves):.3g} | {', '.join(items[:6])} |")
        lines.append("")
    if "provenance" in result:
        lines += provenance.render_md(result["provenance"])
    return "\n".join(lines) + "\n"


def write(result: dict[str, Any], out: Path, seconds: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    clean = _clean(result)
    (out / "h1-panel.json").write_text(json.dumps(clean, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (out / "h1-panel.md").write_text(render_md(clean), encoding="utf-8", newline="\n")
    run = {"finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "seconds": round(seconds, 1)}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")


def load_subpanel(cfg: PanelH1Config) -> tuple[dict[str, pl.DataFrame], dict[str, list[datetime]]]:
    """The sub-panel's bars (sealed, cut to the development period) and member months."""
    from olsprime.data.binance_vision import Month
    from olsprime.features.panel import INTERVAL, TOP_N
    from olsprime.paths import data_dir
    from olsprime.protocol import DEVELOPMENT, load_klines, load_universe
    from olsprime.universe import universe_path

    universe = load_universe(universe_path(data_dir() / "parquet", TOP_N), caller="bench.panel_invariance")
    members = h1_subpanel_members(universe, size=cfg.subpanel_size)
    mm = member_months(universe, first=Month(*cfg.first_month), last=Month(*cfg.last_month))
    months = {s: mm.filter(pl.col("symbol") == s)["month"].to_list() for s in members}
    bars = {}
    for s in members:
        b = load_klines(s, INTERVAL, caller="bench.panel_invariance")
        bars[s] = b.filter(pl.col("known_at") < DEVELOPMENT.end)
    return bars, months


def _default_out() -> Path:
    base = runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-h1-panel"
    out, k = base, 2
    while out.exists():
        out, k = base.with_name(f"{base.name}-{k}"), k + 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.panel_invariance", description="The panel H1 of the Matrix arms.")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <runs>/<UTC date>-h1-panel, suffixed if taken)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cfg = PanelH1Config()
    prov = provenance.record(provenance.current_manifest())  # before any data is read
    bars, months = load_subpanel(cfg)
    result = run_panel_h1(bars, months, cfg, log=lambda s: print(s, flush=True))
    result["provenance"] = prov
    out = args.out or _default_out()
    write(result, out, time.perf_counter() - started)
    h1 = result["h1"]
    print(f"panel H1 {'holds' if h1['holds'] else 'FAILS'}: honest violations {h1['honest_violations']}, "
          f"leaks off pattern {h1['leaks_off_pattern'] or 'none'}; written to {out}")  # fmt: skip
    return 0 if h1["holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
