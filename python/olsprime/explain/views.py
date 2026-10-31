"""``series`` and ``sample`` (``docs/PROTOCOL.md``, "``olsprime.explain``").

- **series** reads only the run's ``series.npz``, by its digest, and never builds the panel. With one arm
  it spreads the period's Brier gain over climatology, ``d_t = b_clim,t - b_arm,t`` (positive: the arm
  beat climatology in hour t), over its hours; with two arms, the log-loss difference
  ``d_t = ll_a,t - ll_b,t`` (negative: a beat b). Each hour's part is ``d_t / sum_t d_t``, so the parts of
  the period add up to 1. It gives the share carried by the top 1 %, 5 % and 10 % of hours, by |part| and
  by signed part, a Lorenz-style summary, the parts by year or month, and the top k hours with their rows.
  Sums are ``math.fsum``: exact sums, correctly rounded, whatever their order.
- **sample** draws n scored rows of the period with ``numpy.random.default_rng(seed)`` alone, and runs
  ``why``'s diagnostics on each, both loud checks included, without the per-entry lists. It describes the
  sample: deciles of the negative mass with the mean log-loss and the calibration gap in each, and the
  mean |part| of each feature. For the ridge arms each row also gets the Matrix-free twin (``why``'s
  :meth:`Explainer.twin`): its forecast rebuilt from panel rows alone, held to the core's within a bound,
  and the count of rows within it (a row beyond it fails loudly).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from olsprime.eval import stats
from olsprime.explain.common import Period, Refused, RunFolder
from olsprime.explain.why import KERNEL_ARMS, OUTPUTS, Explainer, row_key
from olsprime.features.basic import FEATURE_COLUMNS
from olsprime.models.matrix import iso

CLIMATOLOGY = "climatology-panel"
TOP_FRACTIONS = (0.01, 0.05, 0.10)
LORENZ_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
SAMPLE_RAW_TOP = 10  # sample's stamp check reads the raw bars of each row's top 10 entries by |w|


def _fsum(x: np.ndarray) -> float:
    return math.fsum(np.asarray(x, dtype=np.float64).tolist())


def _ratio(a: float, b: float) -> float | None:
    return a / b if b != 0 else None


def series_view(run: RunFolder, arms: list[str], period: Period, by: str, top: int) -> dict[str, Any]:
    """How one arm's Brier gain over climatology, or two arms' log-loss difference, is spread over the period's hours."""
    a, names = run.arrays, run.arms
    for arm in arms:
        if arm not in names:
            raise Refused(f"unknown arm {arm}; the run has {', '.join(names)}")
    hours = a["hours"]
    keep = (hours >= period.start_ms) & (hours < period.end_ms)
    h, count = hours[keep], a["count"][keep]
    if len(arms) == 1:
        if arms[0] == CLIMATOLOGY:
            raise Refused("series with one arm shows its Brier skill against climatology-panel: give another arm")
        b_arm, b_clim = a["brier"][names.index(arms[0])][keep], a["brier"][names.index(CLIMATOLOGY)][keep]
        d = b_clim - b_arm
        what = {"quantity": "Brier gain over climatology-panel per hour, d_t = b_clim - b_arm (positive: the arm beat climatology)",
                "arm": arms[0], "brier_arm": _fsum(b_arm) / h.size if h.size else None,
                "brier_climatology": _fsum(b_clim) / h.size if h.size else None,
                "bss": 1.0 - _fsum(b_arm) / _fsum(b_clim) if h.size else None}  # fmt: skip
        side = {"b_arm": b_arm, "b_clim": b_clim}
    else:
        if arms[0] == arms[1]:
            raise Refused("series with two arms needs two different arms")
        la, lb = a["log_loss"][names.index(arms[0])][keep], a["log_loss"][names.index(arms[1])][keep]
        d = la - lb
        what = {"quantity": f"log-loss difference per hour, d_t = ll({arms[0]}) - ll({arms[1]}) (negative: {arms[0]} was better)",
                "arm": arms[0], "arm_b": arms[1], "mean_difference": _fsum(d) / h.size if h.size else None,
                "identical_hours": int((d == 0).sum())}  # fmt: skip
        side = {"ll_a": la, "ll_b": lb}
    out: dict[str, Any] = {"kind": "series", "period": str(period), "hours": int(h.size), "rows": int(count.sum()), **what}
    if not h.size:
        out["note"] = "the run's series.npz has no scored hour in this period"
        return out
    total = _fsum(d)
    gross = _fsum(np.abs(d))
    out.update(total=total, gross=gross, gross_over_net=_ratio(gross, abs(total)))
    if total == 0:
        out["note"] = "the period's total is 0: shares are undefined"
        return out
    part = d / total
    n = h.size
    idx = np.arange(n)
    shares: dict[str, Any] = {}
    by_abs = np.lexsort((idx, -np.abs(d)))
    by_signed = np.lexsort((idx, -part))
    for f in TOP_FRACTIONS:
        m = max(1, math.ceil(f * n))
        shares[f"top_{f:.0%}"] = {"hours": m, "by_abs_part": _fsum(d[by_abs[:m]]) / total, "by_signed_part": _fsum(d[by_signed[:m]]) / total,
                                  "abs_share_of_gross": _fsum(np.abs(d[by_abs[:m]])) / gross}  # fmt: skip
    out["top_shares"] = shares
    cum = np.cumsum(part[by_signed])
    lorenz = {f"{f:.0%}": float(cum[max(1, math.ceil(f * n)) - 1]) for f in LORENZ_FRACTIONS}
    half = int(np.argmax(cum >= 0.5)) + 1 if (cum >= 0.5).any() else None
    out["lorenz"] = {"order": "hours sorted by signed part, largest first; cumulative share of the period's total",
                     "cumulative_share_at_hour_fraction": lorenz, "hours_to_reach_half": half,
                     "hour_fraction_to_reach_half": half / n if half else None,
                     "peak_cumulative_share": float(cum.max()), "peak_at_hours": int(np.argmax(cum)) + 1}  # fmt: skip
    unit = {"year": "datetime64[Y]", "month": "datetime64[M]"}[by]
    key = h.astype("datetime64[ms]").astype(unit)
    groups = []
    for g in np.unique(key):
        sel = key == g
        entry = {"group": str(g), "hours": int(sel.sum()), "rows": int(count[sel].sum()), "sum": _fsum(d[sel]), "share": _fsum(d[sel]) / total}
        if len(arms) == 1:
            entry["bss"] = _fsum(d[sel]) / _fsum(side["b_clim"][sel])
        groups.append(entry)
    out["by"] = by
    out["groups"] = groups
    hours_out = []
    for i in by_abs[:top]:
        e = {"hour": iso(int(h[i])), "d": float(d[i]), "part": float(part[i]), "rows": int(count[i])}
        e.update({k: float(v[i]) for k, v in side.items()})
        hours_out.append(e)
    out["top_hours"] = hours_out
    return out


def series_rows(ex: Explainer, idx: np.ndarray) -> np.ndarray:
    """The panel rows of series.npz rows ``idx`` (matched on symbol and bar open)."""
    d, a = ex.system.data, ex.run.arrays
    key = row_key(d.symbol, d.open_ms)
    order = np.argsort(key, kind="stable")
    want = row_key(a["symbol"][idx], a["open_time"][idx])
    pos = np.searchsorted(key[order], want)
    pos = np.minimum(pos, order.size - 1)
    rows = order[pos]
    if not np.array_equal(key[rows], want):
        raise Refused("a scored row of series.npz is not in the rebuilt panel")
    return rows


def sample_view(ex: Explainer, arm: str, period: Period, n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """n scored rows of the period, drawn by the seed alone: why's diagnostics and both loud checks on each."""
    a = ex.run.arrays
    open_ = a["open_time"]
    pool = np.flatnonzero((open_ >= period.start_ms) & (open_ < period.end_ms))
    m = min(n, pool.size)
    rng = np.random.default_rng(seed)
    pick = pool[np.sort(rng.choice(pool.size, size=m, replace=False))] if m else pool[:0]
    rows = series_rows(ex, pick) if m else pick
    lines: list[dict[str, Any]] = []
    arm_i = ex.arm_index[arm]
    for i, row in zip(pick.tolist(), rows.tolist(), strict=True):
        res = ex.explain(int(row), arm, by="year", top=SAMPLE_RAW_TOP, full=False, twin=True)
        p = a["probs"][arm_i, i]
        label = int(a["label"][i])
        ll = float(stats.per_row_losses(p[None, :], np.array([label])).log_loss[0])
        line: dict[str, Any] = {"kind": "sample-row", "arm": arm, "pair": res["row"]["pair"], "decision_time": res["row"]["decision_time"],
                                "series_row": i, "label": label, "probs": p, "log_loss": ll,
                                "checks": {"stamps": res["checks"]["stamps"]["passed"], "forecast": res["checks"]["forecast"]["equal_to_the_bit"]}}  # fmt: skip
        if arm == "matrix-stack":
            line["alpha"] = res["choices"]["alpha"]
            line["windows"] = [{"window": w["cell"]["window"], "alpha": w["alpha"], "status": w["cell"]["status"],
                                "diagnostics": w.get("diagnostics"), "features": _parts(w.get("features"))} for w in res["windows"]]  # fmt: skip
            weighted = [(w["alpha"], w["diagnostics"]) for w in res["windows"] if w["alpha"] > 0 and w.get("diagnostics")]
            mass = sum(al for al, _ in weighted)
            line["diagnostics"] = ({"negative_mass": sum(al * dg["negative_mass"] for al, dg in weighted) / mass,
                                    "note": "matrix-stack: the alpha-weighted mean of the weighted windows' negative masses"} if weighted else None)  # fmt: skip
        else:
            line["status"] = res["cell"]["status"]
            dg = res.get("diagnostics")
            line["diagnostics"] = None if dg is None else {k: dg[k] for k in ("n", "negative_mass", "effective_sample_size", "leverage", "sum_abs_w")}
            line["features"] = _parts(res.get("features"))
        cells = [w["cell"] for w in res["windows"]] if arm == "matrix-stack" else [res["cell"]]
        twins = [c["twin"] for c in cells if "twin" in c]
        line["twin"] = None if arm in KERNEL_ARMS or not twins else {
            "cells": len(twins), "within_bound": sum(1 for t in twins if t["within_bound"]), "rows": [t["rows"] for t in twins],
            "gap_over_bound": max(t["gap_over_bound"] for t in twins)}  # fmt: skip
        lines.append(line)
    summary = describe(lines, arm, seed, period, n, pool.size)
    if arm in KERNEL_ARMS:
        summary["twin"] = {"note": "the kernel arms have no Matrix-free twin"}
    else:
        with_twin = [ln["twin"] for ln in lines if ln["twin"]]
        summary["twin"] = {"rows_checked": len(with_twin), "rows_within_bound": sum(1 for t in with_twin if t["within_bound"] == t["cells"]),
                           "cells_checked": sum(t["cells"] for t in with_twin),
                           "max_gap_over_bound": max((t["gap_over_bound"] for t in with_twin), default=None),
                           "what": "each weighted ridge cell's forecast rebuilt from panel rows alone (numpy.linalg), held to the core's "
                                   "within twin_bound; a row beyond it fails loudly, so every row checked is within"}  # fmt: skip
    return lines, summary


def _parts(features: dict[str, Any] | None) -> dict[str, dict[str, float]] | None:
    if not features:
        return None
    return {o: {p["feature"]: p["part"] for p in features[o]["parts"]} | {"mean_y": features[o]["mean_y"]} for o in OUTPUTS} | {
        "bit_exact_all": features["bit_exact_all"]
    }


def describe(lines: list[dict[str, Any]], arm: str, seed: int, period: Period, n: int, pool: int) -> dict[str, Any]:
    """Deciles of the negative mass with the mean log-loss and calibration gap in each; the mean |part| of each feature."""
    out: dict[str, Any] = {"kind": "sample", "arm": arm, "period": str(period), "seed": seed, "n_asked": n, "scored_rows_in_period": pool,
                           "rows": len(lines)}  # fmt: skip
    with_diag = [ln for ln in lines if ln.get("diagnostics")]
    out["rows_without_a_readout"] = len(lines) - len(with_diag)
    if with_diag:
        neg = np.array([ln["diagnostics"]["negative_mass"] for ln in with_diag])
        ll = np.array([ln["log_loss"] for ln in with_diag])
        p = np.array([ln["probs"] for ln in with_diag], dtype=np.float64)
        y = np.array([ln["label"] for ln in with_diag])
        conf, correct = p.max(axis=1), (p.argmax(axis=1) == y).astype(np.float64)
        order = np.argsort(neg, kind="stable")
        deciles = []
        for k, part in enumerate(np.array_split(order, 10)):
            if not part.size:
                continue
            deciles.append({"decile": k + 1, "rows": int(part.size), "negative_mass_min": float(neg[part].min()), "negative_mass_max": float(neg[part].max()),
                            "mean_log_loss": _fsum(ll[part]) / part.size,
                            "calibration_gap": (_fsum(conf[part]) - _fsum(correct[part])) / part.size})  # fmt: skip
        out["deciles_of_negative_mass"] = deciles
        out["calibration_gap_definition"] = "mean top-label confidence minus the share of rows whose top label is the realized class"
        out["mean_log_loss"] = _fsum(ll) / ll.size
    if arm in KERNEL_ARMS:
        out["mean_abs_part"] = None
        out["mean_abs_part_note"] = "the kernel arms have no per-feature view"
    elif arm == "matrix-stack":
        out["mean_abs_part"] = None
        out["mean_abs_part_note"] = "matrix-stack mixes probabilities, not readouts: its per-feature parts are per window, in each row's line"
    else:
        feats = [ln["features"] for ln in lines if ln.get("features")]
        out["mean_abs_part"] = {o: {f: _fsum(np.abs([fe[o][f] for fe in feats])) / len(feats) for f in FEATURE_COLUMNS} for o in OUTPUTS} if feats else None
        out["per_feature_bit_exact_rows"] = sum(1 for fe in feats if fe["bit_exact_all"])
        out["per_feature_rows"] = len(feats)
    out["checks"] = {"stamps_passed": sum(1 for ln in lines if ln["checks"]["stamps"]),
                     "forecast_equal_to_the_bit": sum(1 for ln in lines if ln["checks"]["forecast"])}  # fmt: skip
    return out
