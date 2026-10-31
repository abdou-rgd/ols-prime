"""H1's blind spots on synthetic bars: the future-shuffle twins, the three classes, the cross-check, determinism."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from olsprime.bench import adversary as adv
from olsprime.bench import invariance as inv
from olsprime.bench import twins as tw
from olsprime.bench.perturb import PERTURBED_COLUMNS
from olsprime.bench.synthetic import synthetic_bars

H1 = inv.H1Config(start=(2018, 3), end=(2018, 6), venues=("zero", "binance-spot", "kraken-pro-entry"))
ADV = adv.AdversaryConfig(n=200, seed=17, h1=H1, long_cuts=8, equivalence_leaks=1, attenuated_equivalence_leaks=0)
CFG = tw.TwinsConfig(adversary=ADV)


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return synthetic_bars(days=560, seed=11)


@pytest.fixture(scope="module")
def bench(bars) -> tw.TwinBench:
    return tw.build({"BTCUSDT": bars}, CFG)


def _hand(i: int, expr, output: str = "target") -> adv.Leak:  # noqa: ANN001
    base = {k: "hand" for k in adv.TAG_KEYS} | {"family": "hand", "output": output}
    return adv.Leak(i, "hand", expr, output, tuple((k, base[k]) for k in adv.TAG_KEYS))


NEXT_UP = adv.Cmp(adv.At("close", 1), ">", 1.0, adv.At("close", 0))  # reads bar s+1
PAST_UP = adv.Cmp(adv.At("close", 0), ">", 1.0, adv.At("close", -1))  # reads bars s and s-1 only
CAUGHT = _hand(9101, NEXT_UP)
CONSTANT = _hand(9102, adv.Cmp(adv.At("fill", 1), "<", 1.0, adv.Const(2.0)))  # a fill is never above 1
CANCELS = _hand(9103, adv.Logic("XOR", NEXT_UP, NEXT_UP))  # always false: constant
CANCELS_THEN_PAST = _hand(9104, adv.Logic("XOR", adv.Logic("XOR", NEXT_UP, NEXT_UP), PAST_UP))  # = PAST_UP: reads the future, needs none
HONEST = _hand(9105, PAST_UP)
FULL_FILL = _hand(9106, adv.Cmp(adv.At("fill", 1), ">", 1.0, adv.Const(0.97)))  # outside H1's fill band [0.2, 0.8]


# --- The twins ---------------------------------------------------------------------------------


def test_a_twin_keeps_every_time_stamp_and_replays_bars_at_or_before_the_cut(bars):
    cut = 300
    twin = tw.shuffle_after(bars, cut, tw.twin_rng(tw.TWIN_SEED, "BTCUSDT", 0, 0))
    assert twin.columns == bars.columns and twin.schema == bars.schema and twin.height == bars.height
    for c in ("symbol", "open_time", "close_time", "known_at"):
        assert twin[c].equals(bars[c]), c
    assert twin[: cut + 1].equals(bars[: cut + 1])  # bit for bit at and before the cut
    past = {tuple(row) for row in bars.select(PERTURBED_COLUMNS)[: cut + 1].iter_rows()}
    later = twin.select(PERTURBED_COLUMNS)[cut + 1 :]
    assert all(tuple(row) in past for row in later.iter_rows())  # every later bar is a whole past bar
    assert not later.equals(bars.select(PERTURBED_COLUMNS)[cut + 1 :])
    again = tw.shuffle_after(bars, cut, tw.twin_rng(tw.TWIN_SEED, "BTCUSDT", 0, 0))
    assert again.equals(twin)  # seeded
    assert not tw.shuffle_after(bars, cut, tw.twin_rng(tw.TWIN_SEED, "BTCUSDT", 0, 1)).equals(twin)


def test_the_draws_include_the_cuts_own_bar_and_nothing_after_it():
    src = tw.twin_sources(5_000, 3, np.random.default_rng(0))
    assert src[:4].tolist() == [0, 1, 2, 3]
    assert set(src[4:].tolist()) == {0, 1, 2, 3}  # rows 0 .. cut, the cut included, with replacement
    with pytest.raises(ValueError, match="later bar"):
        tw.twin_sources(10, 9, np.random.default_rng(0))


def test_a_twin_refuses_a_column_it_does_not_know(bars):
    with pytest.raises(ValueError, match="do not know"):
        tw.shuffle_after(bars.with_columns(pl.lit(1.0).alias("funding")), 300, np.random.default_rng(0))


def test_the_twins_quantities_are_the_drawn_bars_quantities(bars):
    """Recomputed from the twin bars, every quantity of a later row is its source row's, bit for bit."""
    cut = 300
    src = tw.twin_sources(bars.height, cut, tw.twin_rng(tw.TWIN_SEED, "BTCUSDT", 2, 5))
    q0 = adv.quantities(bars)
    q = adv.quantities(tw.shuffle_after(bars, cut, tw.twin_rng(tw.TWIN_SEED, "BTCUSDT", 2, 5)))
    for name in adv.QUANTITY_NAMES:
        assert np.array_equal(q[name].view(np.uint64), q0[name][src].view(np.uint64)), name


def test_the_draws_do_not_depend_on_the_order_of_the_pairs(bars):
    other = synthetic_bars("ETHUSDT", days=560, seed=12)
    a = tw.build({"BTCUSDT": bars, "ETHUSDT": other}, CFG)
    b = tw.build({"ETHUSDT": other, "BTCUSDT": bars}, CFG)
    key = lambda r: (r["pair"], r["k"])  # noqa: E731
    assert sorted(a.reads, key=key) == sorted(b.reads, key=key)
    assert {r["pair"] for r in a.reads} == {"BTCUSDT", "ETHUSDT"}
    assert all(len(r["sources"]) == tw.K_TWINS and all(len(s) == adv.FUTURE_REACH for s in r["sources"]) for r in a.reads)
    assert all(max(max(s) for s in r["sources"]) <= r["index"] for r in a.reads)


def test_the_bench_is_h1s_cuts_and_compared_rows(bench):
    assert bench.twins.cuts == bench.h1.cuts and np.array_equal(bench.twins.scored, bench.h1.scored)
    assert bench.h1.n_cuts == 8 and all(c["schedule"] == "h1" for c in bench.h1.cuts)
    assert bench.twins.directions == tuple(f"twin-{j}" for j in range(tw.K_TWINS))
    assert bench.every.n_cuts == 8 + ADV.long_cuts  # both schedules, for the adversary's own constant flag


# --- The three classes ------------------------------------------------------------------------------


def _full_fill_bars(bars: pl.DataFrame, cuts: list[int]) -> pl.DataFrame:
    """Full-bodied bars (fill 0.99) at every third row up to 300 and at row t-5 of every cut: never at t or t+1."""
    rows = set(range(0, 301, 3)) | {c - 5 for c in cuts}
    assert not rows & ({*cuts} | {c + 1 for c in cuts})
    at = pl.int_range(pl.len()).is_in(sorted(rows))
    o = pl.col("open")
    return bars.with_columns(
        pl.when(at).then(o * np.exp(0.0198)).otherwise(pl.col("close")).alias("close"),
        pl.when(at).then(o * np.exp(0.0199)).otherwise(pl.col("high")).alias("high"),
        pl.when(at).then(o * np.exp(-0.0001)).otherwise(pl.col("low")).alias("low"),
    )


def test_a_planted_leak_of_each_class_lands_in_its_class(bars, bench):
    cuts = [c.index for c in bench.h1.h1_cuts["BTCUSDT"]]
    full = _full_fill_bars(bars, cuts)
    fb = tw.build({"BTCUSDT": full}, CFG)
    assert [c.index for c in fb.h1.h1_cuts["BTCUSDT"]] == cuts  # the cuts follow the calendar, not the prices
    leaks = [CAUGHT, CONSTANT, CANCELS, CANCELS_THEN_PAST, FULL_FILL, HONEST]
    ev = tw.evaluate(leaks, fb)
    got = dict(zip((leak.id for leak in leaks), tw.classify(ev).tolist(), strict=True))
    assert got == {CAUGHT.id: "caught", CONSTANT.id: "constant", CANCELS.id: "constant", CANCELS_THEN_PAST.id: "unchanged",
                   FULL_FILL.id: "blind-spot", HONEST.id: "unchanged"}  # fmt: skip
    i = leaks.index(FULL_FILL)
    assert not ev.caught[i].any() and not ev.constant[i]  # H1 never pushes a fill past 0.97 here, and the leak varies
    assert ev.moved[i].any(axis=1).sum() >= 4  # past full bodies come back at bar t+1 under many twins
    assert ev.moved[leaks.index(CAUGHT)].any()  # the next close's direction is replayed from the past too
    s = tw.summarize(leaks, fb, ev, CFG)
    assert {k: v["count"] for k, v in s["classes"].items()} == {"constant": 2, "unchanged": 2, "blind-spot": 1}
    assert [b["id"] for b in s["blind_spots"]] == [FULL_FILL.id]
    assert sum(v["count"] for v in s["classes"].values()) == s["misses"]["count"] == 5
    assert all(v["wilson95"][0] <= v["share"] <= v["wilson95"][1] for v in s["classes"].values())


def test_an_honest_leak_is_never_moved_by_a_twin(bench):
    """Every honest twin of the population, and a hand-built past-only leak: not a single bit moves."""
    population = adv.generate(ADV.n, ADV.seed)
    ev = tw.evaluate([*population, HONEST], bench)
    assert not ev.honest_moved.any() and not ev.honest_violated.any()
    assert not ev.moved[-1].any()
    r = adv.evaluate_fast(HONEST, bench.twins)
    assert not r.violated.any() and not r.constant  # it varies, yet no twin touches it
    for leak in population[:50]:
        assert not adv.evaluate_fast(leak.honest_twin(), bench.twins).violated.any(), leak.render()


def test_moved_is_h1s_catch_rule(bench):
    """A move of 1e-10 is not a move; one of 2e-9 is, wherever the twin changes the leak."""
    full = tw.evaluate([CAUGHT], bench).moved
    assert full.any()
    assert not tw.evaluate([CAUGHT.attenuated(1e-10)], bench).moved.any()
    assert np.array_equal(tw.evaluate([CAUGHT.attenuated(2e-9)], bench).moved, full)


def test_h1s_verdicts_are_the_adversarys(bars, bench):
    """The twins' H1 bench is the adversary's at H1's cuts: the same catches and the same constant flag."""
    population = adv.generate(ADV.n, ADV.seed)
    full = adv.build_bench({"BTCUSDT": bars}, ADV)
    theirs = adv.evaluate_all(population, full)
    mine = tw.evaluate(population, bench)
    rows = full.rows("h1")
    assert np.array_equal(mine.caught, theirs.caught[:, rows].any(axis=2))
    assert np.array_equal(mine.constant_every, theirs.constant)
    assert (mine.constant | ~mine.constant_every).all()  # constant at both schedules implies constant at H1's cuts


# --- The adversary run: digest, configuration, cross-check -------------------------------------------


def test_the_configuration_round_trips_through_json():
    for cfg in (adv.AdversaryConfig(), ADV):
        assert tw.config_from_json(tw._normal(adv._config_json(cfg))) == cfg
    broken = tw._normal(adv._config_json(ADV))
    broken["h1"]["new_field"] = 1
    with pytest.raises(tw.Refused):
        tw.config_from_json(broken)
    broken = tw._normal(adv._config_json(ADV))
    broken["bands"]["body"] = [0.0, 1.0]  # a constant of the code, not of the config: it must write back
    with pytest.raises(tw.Refused, match="write back"):
        tw.config_from_json(broken)


def _adversary_run(bars_by_pair: dict[str, pl.DataFrame], cfg: adv.AdversaryConfig, folder: Path) -> str:
    """An adversary run written as ``python -m olsprime.bench.adversary`` writes it; returns its digest."""
    result = adv.run(bars_by_pair, cfg)
    result.pop("_timings")
    assert adv.ok(result)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adversary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return hashlib.sha256((folder / "adversary.json").read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def adversary_folder(tmp_path_factory) -> tuple[Path, str]:
    folder = tmp_path_factory.mktemp("adv") / "run"
    cfg = replace(adv.AdversaryConfig(), n=120, h1=tw.SYNTHETIC_H1, long_cuts=8, equivalence_leaks=1, attenuated_equivalence_leaks=0)
    return folder, _adversary_run(tw.synthetic_pairs(), cfg, folder)


def test_the_cross_check_with_an_adversary_run_passes(adversary_folder):
    folder, digest = adversary_folder
    run, cfg = tw.load_adversary(folder, digest)
    result = tw.run(tw.synthetic_pairs(), tw.TwinsConfig(adversary=cfg), adversary_run=run, adversary_digest=digest)
    x = result["cross_check"]
    assert x["agree"] and x["cuts_equal"] and x["misses_equal"] and x["mismatches"] == 0, x
    assert x["misses"] == run["misses"]["count"] == result["misses"]["count"]
    assert result["misses"]["constant_both_schedules"] == run["misses"]["constant_on_original_rows"]
    assert tw.ok(result)
    # A run whose verdicts differ is refused leak by leak: nothing is classified.
    tampered = json.loads(json.dumps(run))
    first = tampered["leaks"][0]
    first["h1_pattern"] = "." * len(first["h1_pattern"]) if "X" in first["h1_pattern"] else "X" + first["h1_pattern"][1:]
    stopped = tw.run(tw.synthetic_pairs(), tw.TwinsConfig(adversary=cfg), adversary_run=tampered, adversary_digest=digest)
    assert "stopped" in stopped and "classes" not in stopped and not tw.ok(stopped)
    assert stopped["cross_check"]["mismatches"] == 1 and not stopped["cross_check"]["misses_equal"]
    assert "THE RUNS DISAGREE" in tw.render_md(stopped)


def test_the_digest_check_refuses_a_changed_file(adversary_folder, tmp_path):
    folder, digest = adversary_folder
    copy = tmp_path / "copy"
    copy.mkdir()
    text = (folder / "adversary.json").read_text(encoding="utf-8")
    (copy / "adversary.json").write_text(text.replace('"seed": ', '"seed":  ', 1), encoding="utf-8")  # same JSON, other bytes
    tw.load_adversary(folder, digest)  # the original passes
    with pytest.raises(tw.Refused, match="SHA-256"):
        tw.load_adversary(copy, digest)
    out = tmp_path / "out"
    assert tw.main(["--synthetic", "--adversary", str(copy), "--sha256", digest, "--out", str(out), "--workers", "1"]) == 2
    assert not out.exists()  # refused before anything is computed or written
    with pytest.raises(SystemExit):
        tw.main(["--adversary", str(folder)])  # a run without its digest is not read at all


def test_the_cli_cross_checks_on_synthetic_bars(adversary_folder, tmp_path):
    folder, digest = adversary_folder
    out = tmp_path / "x"
    assert tw.main(["--synthetic", "--adversary", str(folder), "--sha256", digest, "--out", str(out), "--workers", "1"]) == 0
    result = json.loads((out / "twins.json").read_text(encoding="utf-8"))
    assert result["cross_check"]["agree"] and result["cross_check"]["sha256"] == digest
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["adversary_sha256"] == digest and config["provenance"]["bars"] == "synthetic"
    assert config["adversary"]["n"] == 120  # the run's own configuration, not the default


# --- The run and its determinism ------------------------------------------------------------------------


def test_the_cli_is_deterministic_across_runs_and_worker_counts(tmp_path):
    outs = [tmp_path / name for name in ("a", "b", "c")]
    for out, workers in zip(outs, ("1", "1", "3"), strict=True):
        assert tw.main(["--synthetic", "--n", "150", "--out", str(out), "--workers", workers]) == 0
    for name in ("twins.json", "twins.md"):
        first = (outs[0] / name).read_bytes()
        assert all((o / name).read_bytes() == first for o in outs[1:]), name
    result = json.loads((outs[0] / "twins.json").read_text(encoding="utf-8"))
    assert "finished_at" not in (outs[0] / "twins.json").read_text(encoding="utf-8")
    assert "finished_at" in json.loads((outs[0] / "run.json").read_text(encoding="utf-8"))
    assert result["population"]["n"] == 150 and len(result["leaks"]) == 150 and len(result["per_cut"]) == 16
    assert result["honest_twins"]["moved"] == 0 and result["honest_twins"]["bit_violations"] == 0
    assert result["config"]["seed"] == 20261030 and result["config"]["k"] == 8
    misses = [r for r in result["leaks"] if r["class"] != "caught"]
    assert len(misses) == result["misses"]["count"] == sum(v["count"] for v in result["classes"].values())
    assert {r["class"] for r in result["leaks"]} <= {"caught", *tw.CLASSES}
    text = (outs[0] / "twins.md").read_text(encoding="utf-8")
    for part in ("Honest leaks under the twins", "The misses, in three classes", "95 % Wilson", "True blind spots",
                 "Sanity check: the caught leaks", "Per cut", "What the twins read", "H1's registered rule does not change"):  # fmt: skip
        assert part in text, part


def test_the_cli_refuses_n_with_an_official_run():
    with pytest.raises(SystemExit):
        tw.main(["--n", "10"])
