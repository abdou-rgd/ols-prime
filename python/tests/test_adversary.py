"""H1's adversary on synthetic bars: the generator, the twins, the fast path against H1's own path, the statistics."""

from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from olsprime.bench import adversary as adv
from olsprime.bench import invariance as inv
from olsprime.bench.perturb import PERTURBED_COLUMNS
from olsprime.bench.synthetic import synthetic_bars
from olsprime.paths import REPO_ROOT

H1 = inv.H1Config(start=(2018, 3), end=(2018, 6), venues=("zero", "binance-spot", "kraken-pro-entry"))
CFG = adv.AdversaryConfig(n=400, seed=17, h1=H1, long_cuts=8, equivalence_leaks=4)


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return synthetic_bars(days=560, seed=11)


@pytest.fixture(scope="module")
def bench(bars) -> adv.Bench:
    return adv.build_bench({"BTCUSDT": bars}, CFG)


@pytest.fixture(scope="module")
def population() -> list[adv.Leak]:
    return adv.generate(CFG.n, CFG.seed)


def _hand(i: int, expr, output: str = "target", **tags: str) -> adv.Leak:  # noqa: ANN001
    base = {k: "hand" for k in adv.TAG_KEYS} | {"family": "hand", "output": output} | tags
    return adv.Leak(i, "hand", expr, output, tuple((k, base[k]) for k in adv.TAG_KEYS))


NEXT_CLOSE = _hand(9001, adv.Cmp(adv.At("close", 1), ">", 1.0, adv.At("close", 0)), designed="yes")
BODY_OUTSIDE = _hand(9002, adv.Cmp(adv.At("body", 1), ">", 1.0, adv.Const(0.5)), position="outside", designed="no")
RANGE_INSIDE = _hand(9003, adv.Cmp(adv.At("range", 1), ">", 1.0, adv.Const(0.03)), position="inside", designed="yes")
VOLUME_RETURN = _hand(9004, adv.LogRatio(adv.At("volume", 1), adv.At("volume", 0)), output="exp_return", designed="yes")


# --- The generator -------------------------------------------------------------------------


def test_the_generator_is_deterministic_for_a_seed(population):
    again = adv.generate(CFG.n, CFG.seed)
    assert [leak.render() for leak in again] == [leak.render() for leak in population]
    assert [leak.tags for leak in again] == [leak.tags for leak in population]
    assert again == population  # frozen dataclasses: the expressions themselves are equal
    prefix = adv.generate(50, CFG.seed)
    assert prefix == population[:50]  # leak i depends on (seed, i) alone: a smaller run is a prefix
    other = adv.generate(50, CFG.seed + 1)
    assert [leak.render() for leak in other] != [leak.render() for leak in prefix]


def test_the_population_covers_every_family_quantity_and_band_position():
    leaks = adv.generate(3000, 1)
    families = Counter(leak.family for leak in leaks)
    for fam, w in zip(adv.FAMILIES, adv.FAMILY_WEIGHTS, strict=True):
        assert abs(families[fam] / len(leaks) - w) < 0.03, (fam, families[fam])
    rendered = " ".join(leak.render() for leak in leaks)
    for q in PERTURBED_COLUMNS:  # every column perturb.py moves, trades and both taker volumes included
        assert f"{q}[" in rendered, q
    for q in adv.RATIOS:
        assert f"{q}[" in rendered, q
    for q in ("fill[", "close_location[", "wick_ratio[", "green[", "red[", "hammer[", "star[", "colour[", "-and-small[", "-and-big["):
        assert q in rendered, q
    positions = Counter(leak.tag["position"] for leak in leaks)
    assert positions["inside"] and positions["outside"] and positions["window-dependent"]
    assert {leak.tag["designed"] for leak in leaks} == {"yes", "cell", "no"}
    assert {leak.output for leak in leaks} == set(adv.OUTPUTS)
    assert {"AND", "OR", "XOR", "WSUM"} <= {leak.tag["operation"] for leak in leaks}
    beyond = [leak for leak in leaks if "90<k" in leak.tag["reference"]]
    assert beyond  # references beyond H1's 90-bar window


def test_threshold_positions_follow_the_bands():
    assert adv._band_position("body", 0.02) == "inside" and adv._band_position("body", 0.5) == "outside"
    assert adv._band_position("body", 0.004) == "inside" and adv._band_position("body", 0.10) == "outside"  # [lo, hi)
    assert adv._band_position("fill", 0.5) == "inside" and adv._band_position("fill", 0.9) == "outside"
    assert adv._band_position("close_location", 0.1) == "outside"
    assert adv._margin_position(1.4) == "inside" and adv._margin_position(2.0) == "outside"


def test_every_leak_reads_a_later_bar_and_every_twin_reads_none(bars, population):
    view = adv.SeriesView(adv.quantities(bars))
    for leak in population:
        for unit, later in ((leak, True), (leak.honest_twin(), False)):
            rec = adv.RecordingView(view)
            unit.outputs(rec)
            read = {o for _, o in rec.reads}
            assert read == set(unit.expr.offsets()), unit.render()  # what it declares is what it reads
            if later:
                assert 1 <= max(read) <= adv.FUTURE_REACH, unit.render()
            else:
                assert max(read) <= 0, unit.render()
            assert min(read) >= -(adv.PAST_REACH - adv.FUTURE_REACH), unit.render()


def test_a_twin_is_the_same_expression_moved_into_the_past():
    leak = _hand(1, adv.Cmp(adv.Agg("volume", "max", 1, 3), "<", 1.2, adv.At("volume", -7)))
    assert leak.honest_twin().render() == "target = max(volume[s-2..s]) < 1.2 x volume[s-7]"
    assert NEXT_CLOSE.honest_twin().render() == "target = close[s] > close[s]"
    with pytest.raises(ValueError, match="no twin"):
        leak.honest_twin().honest_twin()


# --- The views -----------------------------------------------------------------------------


def test_the_window_view_equals_the_series_view_bit_for_bit(bars, population):
    """The kept windows hold every bar a leak reads: rows t-10..t equal the whole series' outputs, at every cut."""
    q = adv.quantities(bars)
    series = adv.SeriesView(q)
    cuts = [5, 120, 300, bars.height - 2]  # near the start (padding), inside, near the end (padding)
    windows = [adv._window(q, c) for c in cuts]
    view = adv.WindowView({name: np.stack([w[name] for w in windows]) for name in adv.QUANTITY_NAMES})
    for leak in population[:200]:
        for unit in (leak, leak.honest_twin()):
            full, fast = unit.outputs(series), unit.outputs(view)
            for name in full:
                for r, c in enumerate(cuts):
                    rows = c - adv.FUTURE_REACH + np.arange(adv.COMPARED)
                    ok = rows >= 0
                    a = np.asarray(full[name])[rows[ok]]
                    b = np.broadcast_to(fast[name], (len(cuts), adv.COMPARED))[r][ok]
                    assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), (unit.render(), name, c)


# --- Twins and hand-built leaks ----------------------------------------------------------------


def test_twins_are_never_caught(bench, population):
    ev = adv.evaluate_all(population, bench)
    assert not ev.twin_caught.any() and not ev.twin_violated.any()
    assert ev.caught.any(axis=(1, 2)).mean() > 0.8  # and the leaks themselves are mostly caught


def test_a_twin_ignores_any_change_after_the_cut(bars, population):
    """Not only H1's perturbation: garbage after the cut leaves a twin's outputs at rows <= t bit for bit."""
    cut = 300
    rng = np.random.default_rng(0)
    q = adv.quantities(bars)
    garbage = {k: np.concatenate([v[: cut + 1], rng.uniform(-5, 5, size=v.size - cut - 1)]) for k, v in q.items()}
    for leak in population:
        twin = leak.honest_twin()
        a, b = twin.outputs(adv.SeriesView(q)), twin.outputs(adv.SeriesView(garbage))
        for name in a:
            assert np.array_equal(np.asarray(a[name])[: cut + 1].view(np.uint64), np.asarray(b[name])[: cut + 1].view(np.uint64))


def _h1_rows(bench: adv.Bench) -> np.ndarray:
    return bench.rows("h1")


def test_a_next_bar_close_comparison_is_caught_at_every_cut_in_exactly_one_direction(bench):
    r = adv.evaluate_fast(NEXT_CLOSE, bench)
    rows = _h1_rows(bench)
    assert r.caught[rows].sum(axis=1).tolist() == [1] * rows.size  # flips in exactly one run, at every cut
    assert np.all(r.size[rows] == 1.0)  # a target moves by 1


def test_a_threshold_inside_its_band_is_caught_and_one_outside_can_be_missed(bench):
    rows = _h1_rows(bench)
    inside = adv.evaluate_fast(RANGE_INSIDE, bench)
    assert inside.caught[rows].sum(axis=1).tolist() == [1] * rows.size
    outside = adv.evaluate_fast(BODY_OUTSIDE, bench)
    # A body above 50 % is never printed by these bars, and the perturbed bar t+1 reaches 10-15 %: no flip.
    assert not outside.caught[rows].any()
    s = adv.summarize([NEXT_CLOSE, BODY_OUTSIDE, RANGE_INSIDE, VOLUME_RETURN], bench,
                      adv.evaluate_all([NEXT_CLOSE, BODY_OUTSIDE, RANGE_INSIDE, VOLUME_RETURN], bench), CFG)  # fmt: skip
    assert s["misses"]["count"] == 1 and s["misses"]["sample"][0]["id"] == BODY_OUTSIDE.id
    position = {r["value"]: r for r in s["misses"]["by_tag"]["position"]}
    assert position["outside"]["missed"] == 1


def test_a_designed_miss_is_explained_by_h1s_declared_reason_or_listed(bars, bench):
    """A bar t with a fill of 0.01, below the /3 run's floor: H1 declares ``?`` there, and the adversary says why.
    A leak tagged ``designed`` by mistake misses without a declared reason: it is listed as a candidate hole."""
    cut = bench.h1_cuts["BTCUSDT"][0].index
    c = float(bars["close"][cut])
    at = pl.int_range(pl.len()) == cut
    doji = bars.with_columns([pl.when(at).then(pl.lit(v)).otherwise(pl.col(k)).alias(k) for k, v in
                              (("open", c), ("high", c * np.exp(0.5)), ("low", c * np.exp(-0.49)), ("close", c * np.exp(0.0099)))])  # fmt: skip
    fill = _hand(9005, adv.Cmp(adv.At("fill", 1), "<", 1.0, adv.At("fill", 0)), designed="yes")
    mislabelled = replace(BODY_OUTSIDE, tags=tuple((k, "yes" if k == "designed" else v) for k, v in BODY_OUTSIDE.tags))
    b = adv.build_bench({"BTCUSDT": doji}, CFG)
    ev = adv.evaluate_all([fill, mislabelled], b)
    dm = adv.designed_misses([fill, mislabelled], b, ev, {"BTCUSDT": doji})
    first = [m for m in dm["unexplained"] if m["leak"] == fill.id]
    assert not first, first  # every miss of the fill leak is a declared ``?``
    assert dm["explained_by_declared_reason"] >= 1 and any("floor" in r for r in dm["reasons"])
    assert {m["leak"] for m in dm["unexplained"]} == {mislabelled.id}
    assert dm["unexplained_count"] == b.n_cuts  # missed everywhere, never explained
    assert "Unexplained: " in adv.render_md({**adv.run({"BTCUSDT": bars}, replace(CFG, n=5, equivalence_leaks=1))})


def test_a_constant_miss_is_counted_apart(bench):
    always = _hand(9006, adv.Cmp(adv.At("fill", 1), "<", 1.0, adv.Const(2.0)))  # a fill is never above 1
    ev = adv.evaluate_all([always, NEXT_CLOSE], bench)
    assert ev.constant.tolist() == [True, False]
    s = adv.summarize([always, NEXT_CLOSE], bench, ev, CFG)
    assert s["misses"]["count"] == 1 and s["misses"]["constant_on_original_rows"] == 1


def test_constant_twins_are_counted(bench, population):
    """``close[s] > close[s]``, the twin of a next-bar comparison with bar s, is constant: it cannot be caught."""
    assert adv.evaluate_fast(NEXT_CLOSE.honest_twin(), bench).constant
    assert adv.evaluate_fast(VOLUME_RETURN.honest_twin(), bench).constant  # ln(volume[s] / volume[s]) = 0
    ev = adv.evaluate_all(population, bench)
    s = adv.summarize(population, bench, ev, CFG)
    tw = s["twins"]
    assert tw["constant_at_compared_rows"] == int(ev.twin_constant.sum()) > 0
    assert tw["non_constant"] == len(population) - tw["constant_at_compared_rows"] > 0
    assert tw["non_constant_caught"] == 0 and tw["constant_share"]["n"] == len(population)


# --- Attenuation ------------------------------------------------------------------------------------


def test_eps_one_returns_the_outputs_bit_for_bit_and_eps_scales_the_contribution(bars, population):
    view = adv.SeriesView(adv.quantities(bars))
    for leak in population[:100]:
        a, b = leak.outputs(view), leak.attenuated(1.0).outputs(view)
        assert all(np.array_equal(a[k].view(np.uint64), b[k].view(np.uint64)) for k in a)
        w = leak.attenuated(1e-3).outputs(view)
        if leak.output == "probs":
            for k in ("p_down", "p_up"):
                ok = ~np.isnan(a[k])
                assert np.allclose(w[k][ok] - 0.4, 1e-3 * (a[k][ok] - 0.4), atol=1e-15)
            total = w["p_down"] + w["p_neutral"] + w["p_up"]
            assert np.allclose(total[~np.isnan(total)], 1.0)
        else:
            (k,) = a
            assert np.array_equal(np.isnan(w[k]), np.isnan(a[k]))
            ok = ~np.isnan(a[k])
            assert np.array_equal(w[k][ok], 1e-3 * a[k][ok])
    with pytest.raises(ValueError):
        NEXT_CLOSE.attenuated(0.0)
    assert NEXT_CLOSE.attenuated(1e-10).render() == "target = 1e-10 x close[s+1] > close[s]"


def test_the_auditors_probe_at_the_rules_edge_in_both_paths(bars, bench):
    """1e-10 x (close[s+1] > close[s]) moves its target by 1e-10, below the rule: caught at 0 of 8 cuts by the fast path
    and by H1's own path. At 2e-9 it is above the rule: 8 of 8 in both."""
    rows = _h1_rows(bench)
    cuts = bench.h1_cuts["BTCUSDT"]
    for eps, expected in ((1e-10, 0), (2e-9, len(cuts))):
        leak = NEXT_CLOSE.attenuated(eps)
        fast = adv.evaluate_fast(leak, bench).caught[rows].any(axis=1)
        full = adv.full_path(leak, bars, cuts, H1)
        assert int(fast.sum()) == expected == sum(any(c) for c in full["caught"]), eps
    eq = adv.check_equivalence([NEXT_CLOSE.attenuated(1e-10), NEXT_CLOSE.attenuated(2e-9), VOLUME_RETURN.attenuated(1e-10)],
                               bench, {"BTCUSDT": bars}, CFG, twins=False)  # fmt: skip
    assert eq["agree"] and eq["units_compared"] == 3, eq


def test_the_attenuation_curve_reproduces_the_run_at_eps_one_and_falls(bench, population):
    ev = adv.evaluate_all(population, bench)
    curve = adv.attenuation_curve(population, bench, ev)
    assert [r["eps"] for r in curve] == list(adv.EPSILONS)
    s = adv.summarize(population, bench, ev, CFG)
    assert curve[0]["any_h1_cut"] == s["h1_cuts"]["overall"] and curve[0]["per_cut_rate_h1"] == s["h1_cuts"]["per_cut_rate"]
    assert all(r["reproduces_unattenuated"] for r in curve)
    shares = [r["any_h1_cut"]["share"] for r in curve]
    assert shares == sorted(shares, reverse=True) and shares[-1] < shares[0]
    # Below the rule, a target leak moves by eps < 1e-9: never caught.
    assert curve[-1]["any_h1_cut_by_output"]["target"]["caught"] == 0
    broken = adv.Evaluated(~ev.caught, ev.size, ev.twin_caught, ev.twin_violated, ev.constant, ev.twin_constant)
    with pytest.raises(AssertionError, match="eps = 1"):
        adv.attenuation_curve(population[:20], bench, replace(broken, caught=broken.caught[:20]), epsilons=(1.0,))


def test_a_continuous_leak_is_caught_at_every_cut_in_both_directions(bench):
    r = adv.evaluate_fast(VOLUME_RETURN, bench)
    assert r.caught.all()
    assert np.all(r.size > 0.1)  # the volume moves by a factor 1.5 at least: ln 1.5


# --- Equivalence with H1's own path ----------------------------------------------------------------


def _equivalence_leaks(population: list[adv.Leak]) -> list[adv.Leak]:
    by_output = {}
    for leak in population:
        by_output.setdefault(leak.output, leak)
    return [NEXT_CLOSE, BODY_OUTSIDE, VOLUME_RETURN, *by_output.values()]


def test_the_fast_path_equals_h1s_own_path(bars, bench, population):
    leaks = _equivalence_leaks(population)
    assert {leak.output for leak in leaks} == set(adv.OUTPUTS)
    eq = adv.check_equivalence(leaks, bench, {"BTCUSDT": bars}, CFG)
    assert eq["agree"], eq
    assert eq["units_compared"] == 2 * len(leaks)  # each leak and its twin
    full = adv.full_path(NEXT_CLOSE, bars, bench.h1_cuts["BTCUSDT"], H1)
    assert all(sum(c) == 1 for c in full["caught"])  # through the pipeline too: one direction per cut


def test_the_equivalence_check_sees_a_broken_fast_path(bars, bench):
    """Drop the cut's own row from the fast path's compared rows: a next-bar leak escapes it, and the check says so."""
    broken = replace(bench, scored=bench.scored.copy())
    broken.scored[:, -1] = False
    eq = adv.check_equivalence([NEXT_CLOSE], broken, {"BTCUSDT": bars}, CFG)
    assert not eq["agree"] and eq["mismatches"][0]["leak"] == NEXT_CLOSE.id


def test_a_disagreement_stops_the_run_before_any_rate(bars, monkeypatch):
    monkeypatch.setattr(adv, "full_path", lambda leak, b, cuts, cfg: {"caught": [[False, False]] * len(cuts), "violated": [[False, False]] * len(cuts)})
    cfg = replace(CFG, n=20, equivalence_leaks=3)
    result = adv.run({"BTCUSDT": bars}, cfg)
    assert "stopped" in result and "h1_cuts" not in result and not adv.ok(result)
    assert "THE PATHS DISAGREE" in adv.render_md(result)


def test_the_run_reports_every_part(bars):
    cfg = replace(CFG, n=60, equivalence_leaks=2)
    result = adv.run({"BTCUSDT": bars}, cfg)
    assert adv.ok(result)
    assert result["twins"]["false_catches"] == 0 and result["twins"]["bit_violations"] == 0
    o = result["h1_cuts"]["overall"]
    assert o["n"] == 60 and o["wilson95"][0] <= o["share"] <= o["wilson95"][1]
    assert len(result["h1_cuts"]["per_cut"]) == 8 and result["long_schedule"]["cuts"] == cfg.long_cuts
    assert len(result["leaks"]) == 60 and all(len(r["h1_pattern"]) == 8 for r in result["leaks"])
    assert result["equivalence"]["attenuated"]["agree"] and result["equivalence"]["attenuated"]["scales"] == [1e-8, 1e-10]
    assert [r["eps"] for r in result["attenuation"]] == list(adv.EPSILONS)
    assert result["never_produced"] == list(adv.NEVER_PRODUCED)
    text = adv.render_md(result)
    for part in ("Equivalence", "Near the rule's edge", "Honest twins", "a constant twin cannot be caught",
                 "Share caught at H1's cuts", "Two headline rates", "**Any cut:**", "**Per cut:**", "Per-cut rate",
                 "Size of the catches", "not H1's sensitivity", "Attenuation curve", "| 1e-10 |", "The misses",
                 "at the compared rows they carry no information", "What the generator never produces", "hysteresis",
                 "s+25", "Row for `docs/RESULTS.md`", "Mean per-cut rate (closer to H1's per-cut rule)"):  # fmt: skip
        assert part in text, part
    assert "on these series" not in text
    per_cut_rows = [line for line in text.splitlines() if line.startswith("| BTCUSDT |")]
    assert len(per_cut_rows) == 8 and all(line.count("[") >= 1 for line in per_cut_rows)  # each with its Wilson interval
    again = adv.run({"BTCUSDT": bars}, cfg)
    again.pop("_timings"), result.pop("_timings")
    assert again == result  # deterministic


def test_the_long_schedule_is_seeded_and_scored(bars):
    from olsprime.bench import pipeline as pp
    from olsprime.data.binance_vision import Month

    frame = pp.build_frame(bars, H1.interval, H1.h)
    folds = pp.make_folds(frame, H1.interval, H1.h, Month(*H1.start), Month(*H1.end))
    a = adv.long_schedule(bars, folds, seed=1, n=16)
    assert [c.index for c in a] == [c.index for c in adv.long_schedule(bars, folds, seed=1, n=16)]
    scored = set(adv.scored_rows(folds).tolist())
    assert len({c.index for c in a}) == 16 and all(c.index in scored and c.index < bars.height - 1 for c in a)
    assert [c.index for c in a] == sorted(c.index for c in a)


# --- Statistics ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.27753279),  # z^2 / (n + z^2)
        (10, 10, 0.72246721, 1.0),  # n / (n + z^2)
        (5, 10, 0.23659309, 0.76340691),
    ],
)
def test_the_wilson_interval_on_known_values(k, n, lo, hi):
    got = adv.wilson(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-7)


def test_the_wilson_interval_equals_the_roots_of_its_quadratic():
    """(n + z^2) p^2 - (2k + z^2) p + k^2 / n = 0: the two roots are the bounds."""
    z2 = adv.Z95**2
    for k, n in ((1, 100), (37, 263), (9_990, 10_000)):
        roots = sorted(np.roots([n + z2, -(2 * k + z2), k * k / n]).real)
        assert adv.wilson(k, n) == pytest.approx(tuple(roots), abs=1e-12)


def test_the_wilson_bounds_solve_the_score_equation():
    z = adv.Z95
    for k, n in ((3, 17), (40, 41), (500, 10_000)):
        p = k / n
        for b in adv.wilson(k, n):
            if 0.0 < b < 1.0:
                assert (p - b) ** 2 == pytest.approx(z * z * b * (1 - b) / n, rel=1e-9)
    assert all(math.isnan(x) for x in adv.wilson(0, 0))
    with pytest.raises(ValueError):
        adv.wilson(3, 2)


# --- The fence -------------------------------------------------------------------------------------


def test_no_model_code_imports_the_adversary_or_the_leaks():
    package = REPO_ROOT / "python" / "olsprime"
    for sub in ("models", "train"):
        for path in (package / sub).rglob("*.py") if (package / sub).is_dir() else []:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
                for name in names:
                    assert "adversary" not in name and "bench.leaks" not in name, (path, name)
    text = (package / "bench" / "adversary.py").read_text(encoding="utf-8")
    assert text.startswith('"""!!! LEAKY FUNCTIONS, ON PURPOSE. NOT MODELS. !!!')
