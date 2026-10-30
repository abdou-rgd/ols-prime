"""The command line of ``explain``: ``why``, ``series`` and ``sample``.

Usage (from ``python/``)::

    uv run --no-sync python -m olsprime.explain why PAIR AS_OF --run DIR --sha256 HEX --arm ARM [--by month] [--top 10]
        [--at PAIR AS_OF ...] [--test-months] [--jsonl FILE]
    uv run --no-sync python -m olsprime.explain series ARM [ARM_B] --run DIR --sha256 HEX --test-months [--period 2019-01:2020-06]
        [--by month] [--top 10] [--jsonl FILE]
    uv run --no-sync python -m olsprime.explain sample ARM --run DIR --sha256 HEX --test-months [--period ...] [--n 1000] [--seed 0]
        [--jsonl FILE]

``AS_OF`` is the row's decision time or its bar's open (ISO 8601, UTC when no zone is given). Each answer is
one JSON line (sorted keys, no wall clock) plus a short text summary. Without ``--jsonl`` the JSON lines
follow the summaries on stdout, after a line ``=== json lines ===``. Exit codes: 0 answered, 2 refused,
3 a loud check failed (nothing is printed as a result).

Order of a call: the arguments are checked (periods, every ``--at``, ``--jsonl``), the extension is probed
(``why`` and ``sample``), ``series.npz`` is hashed against the digest given; then the contact lines are
written, and only then is the run folder opened. ``why`` writes one contact line per row asked (``AS_OF``
and each ``--at``), all up front: a call that then stops on one row (a refusal, a loud check) has still
logged every row it named, so the log over-counts contacts rather than missing one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

import polars as pl

from olsprime.explain.common import (
    DEFAULT_PERIOD,
    EXPLORATORY,
    MAX_SAMPLE,
    MAX_TOP,
    TEST_FLAG,
    LoudCheckFailed,
    Refused,
    check_as_of,
    check_jsonl,
    check_period,
    contacts_file,
    json_line,
    log_contact,
    month_label,
    open_run,
    parse_period,
    parse_time,
    provenance,
    verify_digest,
)
from olsprime.explain.views import sample_view, series_view
from olsprime.explain.why import (
    GROUPINGS,
    OUTPUTS,
    WHY_ARMS,
    Explainer,
    check_extension,
)
from olsprime.models.matrix import ARMS, MatrixSystem, iso

SERIES_KEYS_READ = ("arms", "hours", "count", "log_loss", "brier")
ROW_KEYS_READ = ("arms", "hours", "count", "log_loss", "probs", "label", "symbol", "symbols", "open_time", "label_known_at")
JSON_MARK = "=== json lines ==="


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m olsprime.explain", description="Why is a Matrix forecast what it is (Report n°2's arms).")
    sub = p.add_subparsers(dest="command", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--run", type=Path, required=True, help="a Report n°2 run folder")
        q.add_argument("--sha256", required=True, help="the SHA-256 of its series.npz (its report2.json must agree)")
        q.add_argument(TEST_FLAG, dest="test_months", action="store_true",
                       help=f"name a period or as_of other than the default {DEFAULT_PERIOD}: test months, logged as a contact")  # fmt: skip
        q.add_argument("--jsonl", type=Path, default=None, help="write the JSON lines here (default: stdout, after the summaries)")

    w = sub.add_parser("why", help="one arm's forecast at one row, broken down, with two loud checks")
    w.add_argument("pair")
    w.add_argument("as_of", help="the row's decision time or its bar's open, ISO 8601")
    w.add_argument("--arm", required=True, help="one of " + ", ".join(WHY_ARMS))
    w.add_argument("--by", choices=GROUPINGS, default="month")
    w.add_argument("--top", type=int, default=10)
    w.add_argument("--at", nargs=2, action="append", default=[], metavar=("PAIR", "AS_OF"), help="another row, in the same process")
    common(w)
    s = sub.add_parser("series", help="how an arm's Brier skill, or two arms' log-loss difference, is spread over hours")
    s.add_argument("arms", nargs="+", metavar="ARM")
    s.add_argument("--period", default=str(DEFAULT_PERIOD).replace("..", ":"))
    s.add_argument("--by", choices=("year", "month"), default="month")
    s.add_argument("--top", type=int, default=10)
    common(s)
    m = sub.add_parser("sample", help="why's diagnostics on a seeded sample of scored rows")
    m.add_argument("arm", help="one of " + ", ".join(WHY_ARMS))
    m.add_argument("--period", default=str(DEFAULT_PERIOD).replace("..", ":"))
    m.add_argument("--n", type=int, default=1000)
    m.add_argument("--seed", type=int, required=True)
    common(m)
    return p


def _bounded(name: str, value: int, high: int) -> None:
    if not 1 <= value <= high:
        raise Refused(f"{name} must be in 1..{high}, got {value}")


def _why_arm(arm: str) -> None:
    if arm == "climatology-panel":
        raise Refused("climatology-panel has no readout to explain (its forecast is the class frequencies at the fit); series reads it")
    if arm not in WHY_ARMS:
        raise Refused(f"unknown arm {arm}; why and sample explain {', '.join(WHY_ARMS)}")


def main(
    argv: Sequence[str] | None = None,
    *,
    data_dir: Path | None = None,
    universe: pl.DataFrame | None = None,
    contacts: Path | None = None,
    system_factory: Callable[..., MatrixSystem] = MatrixSystem,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run one call. ``data_dir``, ``universe``, ``contacts`` and ``system_factory`` are for tests."""
    out, err = out or sys.stdout, err or sys.stderr
    args = _parser().parse_args(argv)
    try:
        jsonl = check_jsonl(args.jsonl)
        summaries, lines = _dispatch(args, data_dir=data_dir, universe=universe, contacts=contacts_file(contacts), system_factory=system_factory)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=err)
        return 2
    except LoudCheckFailed as exc:
        print(f"LOUD CHECK FAILED, the call stops and prints no result: {exc}", file=err)
        return 3
    text = "\n".join(summaries) + "\n"
    payload = "".join(json_line(x) + "\n" for x in lines)
    if jsonl is not None:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(jsonl, "x", encoding="utf-8", newline="\n") as f:
                f.write(payload)
        except FileExistsError:
            print(f"REFUSED: --jsonl {jsonl} appeared during the call: explain writes a new file only", file=err)
            return 2
        out.write(text)
    else:
        out.write(text + JSON_MARK + "\n" + payload)
    return 0


def _dispatch(args: argparse.Namespace, **kw: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if args.command == "why":
        return _why(args, **kw)
    if args.command == "series":
        return _series(args, **kw)
    return _sample(args, **kw)


def _why(args: argparse.Namespace, *, data_dir, universe, contacts, system_factory) -> tuple[list[str], list[dict[str, Any]]]:
    _bounded("--top", args.top, MAX_TOP)
    _why_arm(args.arm)
    calls = [(args.pair, parse_time(args.as_of))] + [(p, parse_time(t)) for p, t in args.at]
    for _, t in calls:
        check_as_of(t, args.test_months)
    check_extension()
    verify_digest(args.run, args.sha256)
    for _, t in calls:  # one contact line per row asked, all up front (this over-logs when a later row stops the call)
        log_contact(contacts, subcommand="why", period=f"as_of {iso(t)} (month {month_label(t)}; entries known by it)", arm=args.arm,
                    series_sha256=args.sha256)  # fmt: skip
    run = open_run(args.run, args.sha256, ROW_KEYS_READ)
    ex = Explainer(run, data_dir=data_dir, universe=universe, system_factory=system_factory)
    prov = provenance(run, panel_verified=ex.panel_verified, contacts=contacts)
    results = []
    for pair, t in calls:
        row = ex.find_row(pair, t)
        check_as_of(int(ex.system.data.decision_ms[row]), args.test_months)
        res = ex.explain(row, args.arm, by=args.by, top=args.top, twin=True)
        res.update(kind="why", by=args.by, top=args.top, provenance=prov,
                   exploratory=EXPLORATORY if int(ex.system.data.decision_ms[row]) >= DEFAULT_PERIOD.end_ms else None)
        results.append(res)
    return [why_text(r) for r in results], results


def _series(args: argparse.Namespace, *, contacts, **_: Any) -> tuple[list[str], list[dict[str, Any]]]:
    _bounded("--top", args.top, MAX_TOP)
    if len(args.arms) > 2:
        raise Refused("series takes one arm or two")
    for a in args.arms:
        if a not in ARMS:
            raise Refused(f"unknown arm {a}")
    period = parse_period(args.period)
    check_period(period, args.test_months)
    if not args.test_months:
        raise Refused(f"series reads series.npz, which holds test rows only: name that contact with {TEST_FLAG}")
    verify_digest(args.run, args.sha256)
    log_contact(contacts, subcommand="series", period=str(period), arm=",".join(args.arms), series_sha256=args.sha256)
    run = open_run(args.run, args.sha256, SERIES_KEYS_READ)
    res = series_view(run, list(args.arms), period, args.by, args.top)
    res["provenance"] = provenance(run, panel_verified=False, contacts=contacts)
    res["exploratory"] = EXPLORATORY
    return [series_text(res)], [res]


def _sample(args: argparse.Namespace, *, data_dir, universe, contacts, system_factory) -> tuple[list[str], list[dict[str, Any]]]:
    _bounded("--n", args.n, MAX_SAMPLE)
    _why_arm(args.arm)
    period = parse_period(args.period)
    check_period(period, args.test_months)
    if not args.test_months:
        raise Refused(f"sample reads series.npz, which holds test rows only: name that contact with {TEST_FLAG}")
    check_extension()
    verify_digest(args.run, args.sha256)
    log_contact(contacts, subcommand="sample", period=str(period), arm=args.arm, series_sha256=args.sha256)
    run = open_run(args.run, args.sha256, ROW_KEYS_READ)
    ex = Explainer(run, data_dir=data_dir, universe=universe, system_factory=system_factory)
    lines, summary = sample_view(ex, args.arm, period, args.n, args.seed)
    summary["provenance"] = provenance(run, panel_verified=ex.panel_verified, contacts=contacts)
    summary["exploratory"] = EXPLORATORY
    return [sample_text(summary)], [*lines, summary]


# --- Text summaries (at most about 40 lines each) ---------------------------------------------


def _g(x: Any, digits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def _probs(p: Any) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in p) + "]"


def _check_lines(r: dict[str, Any]) -> list[str]:
    st, fc = r["checks"]["stamps"], r["checks"]["forecast"]
    extra = "; ".join(f"{k} {v['checked']}" for k, v in st.items() if isinstance(v, dict) and "checked" in v)
    rules = "; ".join(f"{x['what']}: {x['entries_behind_sigma']} entries and {x['climatology_rows']} labelled rows known by "
                      f"{x['known_by'][:16]}" for x in st["fold_statistics"])  # fmt: skip
    lines = [(f"check 1, stamps and fold statistics: passed. Stamps ({extra}) recomputed from their panel rows, never from Matrix's, "
             f"latest {st['entries']['latest_label_known_at'] or '-'} <= as_of; {st['top_entries_checked_on_raw_bars']} top entries on raw bars"),
             f"  sigma^2 and climatology equal to the rule's, to the bit ({rules})",
             "  not covered by check 1: the features (panel H1), the fit-time choices from report2.json, the windows behind adaptive's rho"]  # fmt: skip
    if fc["applies"]:
        lines.append(f"check 2, forecast: equal to series.npz row {fc['series_row']} to the bit")
    else:
        lines.append("check 2, forecast: NOT APPLICABLE, row not scored (no series.npz value); only check 1 applies")
    return lines


def _cell_lines(c: dict[str, Any], top: dict[str, Any] | None) -> list[str]:
    cell = c["cell"]
    raw = cell["raw"]
    lines = [f"cell: {cell['status']}, n {cell['n_window']}; raw " + ", ".join(f"{k} {raw[k]:+.5f}" for k in OUTPUTS)
             + f"; probs {_probs(cell['probs'])}" + (" (FALLBACK to climatology)" if cell["fallback_to_climatology"] else "")]
    dg = c.get("diagnostics")
    if dg:
        lines.append(f"diagnostics: sum w {dg['sum_w']:.12g}, sum |w| {dg['sum_abs_w']:.5g}, negative mass {dg['negative_mass']:.5g} "
                     f"({_g(dg['share_negative'], 3)} of weights), ESS {_g(dg['effective_sample_size'], 5)}, leverage {_g(dg['leverage'], 5)}")
    if "numpy_readout_gap" in cell:
        gap, gb = cell["numpy_readout_gap"], cell["numpy_readout_gap_bound"]
        worst = max(OUTPUTS, key=lambda k: gap[k])
        lines.append(f"weights recomputed in numpy: their readout differs from the core's by at most {gap[worst]:.2g} ({worst}), "
                     f"within its bound {gb[worst]:.2g}")  # fmt: skip
    tree = c.get("tree")
    if tree:
        o = max(range(3), key=lambda i: float(cell["probs"][i]))
        name = OUTPUTS[o]
        t = tree[name]
        parts = ", ".join(f"{p['group']} {p['c']:+.4g}" for p in t["top"][:6])
        lines.append(f"tree, output '{name}' ({tree['groups']} groups; {tree['grouping']}):")
        lines.append(f"  {parts}; rest ({t['rest_groups']} groups) {_g(t['rest'])}; closed by a residual of {t['closing_residual_size']:.2g}, "
                     f"within the rounding bound {t['rounding_bound']:.2g}")  # fmt: skip
        ft = tree["fwd_return"]
        lines.append("  fwd_return: " + ", ".join(f"{p['group']} {p['c']:+.3g}" for p in ft["top"][:6])
                     + f"; closed by a residual of {ft['closing_residual_size']:.2g}, within the rounding bound {ft['rounding_bound']:.2g}")
        pt = tree["partition"]
        lines.append(f"  partition: {pt['groups']} groups recounted from the panel's label_known_at, same labels and counts; each "
                     f"sum w_i y_i within its bound (largest gap / bound {pt['max_gap_over_bound']:.2g})")  # fmt: skip
    if "twin" in cell:
        tw = cell["twin"]
        worst = max(OUTPUTS, key=lambda k: tw["gap"][k] / tw["bound"][k])
        lines.append(f"Matrix-free twin ({tw['rows']} panel rows, numpy.linalg): within its bound, gap {tw['gap'][worst]:.2g} <= "
                     f"{tw['bound'][worst]:.2g} ({worst})")  # fmt: skip
    fv = c.get("features")
    if fv:
        lines.append(f"per feature ({fv['statement']}):")
        for name in OUTPUTS:
            f = fv[name]
            lines.append(f"  {name}: mean_y {f['mean_y']:+.4g}; " + ", ".join(f"{p['feature']} {p['part']:+.3g}" for p in f["parts"][:5])
                         + ("" if f["bit_exact"] else f"; NOT bit-exact, numpy gap {f['numpy_gap']:.2g}"))
    ents = c.get("top_entries")
    if ents:
        lines.append(f"top {min(len(ents), 5)} of {len(ents)} entries by |w| (pair, decided, label known, label, fwd_return, w):")
        for e in ents[:5]:
            lines.append(f"  {e['pair']} {e['decision_time'][:16]} {e['label_known_at'][:16]} {e['label']} {e['fwd_return']:+.4f} {e['weight']:+.3g}")
    return lines


def why_text(r: dict[str, Any]) -> str:
    row = r["row"]
    ch = r["choices"]
    head = [f"why {row['pair']} decided {row['decision_time']} (bar {row['open_time'][:16]}, month {row['month']}), arm {r['arm']}"
            + ("" if row["scored"] else "  [NOT SCORED: only the stamp check applies]")]
    desc = ", ".join(f"{k} {v}" for k, v in ch.items() if k not in ("note", "adaptive") and v is not None)
    head.append(f"choices: {desc}")
    if ch.get("adaptive"):
        a = ch["adaptive"]
        head.append(f"  adaptive rho rebuilt from {a['rows_scored']} rows" + (" (default: fewer than 1,000)" if a["default"] else "")
                    + "; scores " + ", ".join(f"{k} {_g(v, 5)}" for k, v in a["scores"].items()))
    if ch.get("note"):
        head.append(f"  note: {ch['note'][:160]}")
    head.append(f"probs {_probs(r['probs'])}")
    body: list[str] = []
    if r["arm"] == "matrix-stack":
        for w in r["windows"]:
            body.append(f"window {w['cell']['window']}, alpha {w['alpha']:.2f}, rho {w['cell']['rho']:g}:")
            body += ["  " + x for x in _cell_lines(w, None)[:2 if w["alpha"] == 0 else 5]]
            fv = w.get("features")
            if w["alpha"] > 0 and fv:
                body.append("  per feature, up: " + ", ".join(f"{p['feature']} {p['part']:+.3g}" for p in fv["up"]["parts"][:4])
                            + f" ({'bit-exact' if fv['bit_exact_all'] else 'NOT bit-exact'} in the core's order, checked)")
    else:
        body = _cell_lines(r, None)
    p = r["provenance"]
    tail = [*_check_lines(r), (f"provenance: commit {str(p['commit'])[:12]} dirty {p['dirty']}, manifest {str(p['manifest_digest'])[:12]}, "
            f"panel {p['panel_fingerprint'][:12]} ({'verified' if p['panel_fingerprint_verified'] else 'NOT verified'}), "
            f"series {p['series_sha256'][:12]}"), f"contact logged in {p['contacts_file']}"]  # fmt: skip
    if r.get("exploratory"):
        tail.append("EXPLORATORY, test months: never tested as a hypothesis on them")
    return "\n".join(head + body + tail)


def series_text(r: dict[str, Any]) -> str:
    lines = [f"series {r['arm']}" + (f" - {r['arm_b']}" if "arm_b" in r else "") + f", period {r['period']}: {r['hours']} hours, {r['rows']} rows",
             r["quantity"]]
    if "bss" in r:
        lines.append(f"BSS {_g(r['bss'], 5)} (Brier {_g(r['brier_arm'], 6)} vs climatology {_g(r['brier_climatology'], 6)})")
    if "mean_difference" in r:
        lines.append(f"mean difference {_g(r['mean_difference'], 5)}; identical hours {r['identical_hours']}")
    if "note" in r:
        return "\n".join([*lines, r["note"]])
    lines.append(f"total {r['total']:.6g}, gross {r['gross']:.6g}, gross / |net| {_g(r['gross_over_net'], 4)}")
    if r["total"] < 0:
        lines.append("the sum of d is negative here: the largest signed parts (d / total) are then the hours that hurt most")
    for k, v in r["top_shares"].items():
        lines.append(f"{k} of hours ({v['hours']}): share by |part| {v['by_abs_part']:+.3f}, by signed part {v['by_signed_part']:+.3f}, "
                     f"of gross {v['abs_share_of_gross']:.3f}")
    lz = r["lorenz"]
    lines.append("Lorenz (signed parts, largest first): " + ", ".join(f"{k} {v:.3f}" for k, v in lz["cumulative_share_at_hour_fraction"].items()))
    lines.append(f"  half the total by {lz['hours_to_reach_half']} hours; peak {lz['peak_cumulative_share']:.3f} at {lz['peak_at_hours']} hours")
    groups = r["groups"]
    shown = groups if len(groups) <= 16 else sorted(groups, key=lambda g: -abs(g["share"]))[:16]
    lines.append(f"by {r['by']}" + ("" if shown is groups else f" (16 largest |share| of {len(groups)})") + ":")
    for g in shown:
        lines.append(f"  {g['group']}: {g['hours']} h, share {g['share']:+.3f}" + (f", BSS {g['bss']:+.4f}" if "bss" in g else ""))
    lines.append("top hours by |part|:")
    for h in r["top_hours"][:10]:
        lines.append(f"  {h['hour'][:16]} part {h['part']:+.4f} d {h['d']:+.4g} rows {h['rows']}")
    lines.append(f"contact logged in {r['provenance']['contacts_file']}")
    lines.append("EXPLORATORY, test months: never tested as a hypothesis on them")
    return "\n".join(lines)


def sample_text(r: dict[str, Any]) -> str:
    lines = [(f"sample {r['arm']}, period {r['period']}, seed {r['seed']}: {r['rows']} of {r['scored_rows_in_period']} scored rows "
             f"({r['rows_without_a_readout']} without a readout)"),
             f"checks: stamps passed {r['checks']['stamps_passed']}, forecast equal to the bit {r['checks']['forecast_equal_to_the_bit']}"]  # fmt: skip
    tw = r["twin"]
    lines.append(tw["note"] if "note" in tw else f"Matrix-free twin: {tw['rows_within_bound']} of {tw['rows_checked']} rows within the bound "
                 f"({tw['cells_checked']} cells; largest gap / bound {_g(tw['max_gap_over_bound'], 3)})")
    if "mean_log_loss" in r:
        lines.append(f"mean log-loss {r['mean_log_loss']:.5f}; deciles of negative mass (rows, range, mean log-loss, calibration gap):")
        for d in r["deciles_of_negative_mass"]:
            lines.append(f"  {d['decile']}: {d['rows']}, {d['negative_mass_min']:.4g}..{d['negative_mass_max']:.4g}, "
                         f"{d['mean_log_loss']:.5f}, {d['calibration_gap']:+.4f}")
    if r.get("mean_abs_part"):
        lines.append(f"per-feature sums bit-exact in the core's order: {r['per_feature_bit_exact_rows']} of {r['per_feature_rows']} rows")
        for o in OUTPUTS:
            m = r["mean_abs_part"][o]
            top = sorted(m, key=lambda f: -m[f])[:5]
            lines.append(f"  mean |part|, {o}: " + ", ".join(f"{f} {m[f]:.3g}" for f in top))
    elif r.get("mean_abs_part_note"):
        lines.append(r["mean_abs_part_note"])
    lines.append(f"contact logged in {r['provenance']['contacts_file']}")
    lines.append("EXPLORATORY, test months: never tested as a hypothesis on them")
    return "\n".join(lines)
