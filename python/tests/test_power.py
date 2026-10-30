"""The size and power study of the pass rule (``olsprime.bench.power``), on synthetic series only.

- the grouped bootstrap (a count-matrix product) equals the direct one, ``x[idx].mean(axis=1)``, within 1e-12 relative;
- the flips are circular blocks, and the bootstrap seed is the registered one;
- iid normal differences give a two-sided size near 5 % (tolerance: three binomial standard errors at 2,000 replicates);
- an AR(1) difference with rho = 0.9 at L = 24 gives what the theory of the flips predicts;
- power rises with delta, and H12's non-inferiority holds more often at 0 than on its margin;
- the predicted variance ratio is exact over offsets and signs, matches an AR(1)'s theory, and holds the measured one;
- the digest check refuses a changed file, the path is refused outside the runs root, and a series whose rule does not
  reproduce report2.json is refused; the command gives the same bytes with one worker or two (a process pool each);
- on Report n°2's synthetic run, the study reproduces every registered comparison before it starts.

Nothing reads the repository's ``data/``: the series are built here or by Report n°2 on its fake store.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

from olsprime.bench import power as pw
from olsprime.bench import report2 as r2
from olsprime.paths import runs_dir
from tests.test_report2 import fake_root, two_runs  # noqa: F401 (fixtures: Report n°2 on its synthetic store)

PROVENANCE = {"git_commit": "c" * 40, "code": {"sha256": "d" * 64, "files": 1}, "manifest_digest": "0" * 64}
RULE_CFG = r2.Report2Config(n_boot=400)  # the registered rule with fewer resamples, for speed


def ar1(rho: float, t: int, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(t)
    x = np.empty(t)
    x[0] = e[0] / math.sqrt(1.0 - rho * rho)
    for i in range(1, t):
        x[i] = rho * x[i - 1] + e[i]
    return x


def arms(d: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Hourly log-losses whose difference x - y is d, with climatology's near 1."""
    base = 1.0 + 0.01 * rng.standard_normal(d.size)
    return {r2.CLIMATOLOGY: base + 0.002, "x": base + d, "y": base}


def fake_run(folder: Path, ll: dict[str, np.ndarray], comparisons: list[pw.Comparison], cfg: r2.Report2Config = RULE_CFG) -> str:
    """A Report n°2 run folder holding only what the study reads: series.npz, report2.json and config.json."""
    folder.mkdir(parents=True, exist_ok=True)
    names = list(ll)
    t = ll[names[0]].size
    arrays = {"arms": np.array(names), "hours": np.arange(t, dtype=np.int64) * 3_600_000, "count": np.ones(t, dtype=np.int64),
              "log_loss": np.stack([ll[a] for a in names])}  # fmt: skip
    r2.write_npz(folder / r2.SERIES_FILE, arrays)
    digest = r2.sha256_file(folder / r2.SERIES_FILE)
    clim = float(ll[r2.CLIMATOLOGY].mean())
    report = {
        "registered": False,
        "series": {"file": r2.SERIES_FILE, "sha256": digest},
        "arms": {r2.CLIMATOLOGY: {"log_loss": {"estimate": clim}}},
        "comparisons": [dict(r2.compare_arms(ll, c.a, c.b, clim, cfg), hypothesis=c.hypothesis) for c in comparisons],
    }
    r2._write_json(folder / "report2.json", report)
    r2._write_json(folder / "config.json", {"config": asdict(cfg), "provenance": PROVENANCE, "registered": False})
    return digest


# --- the parts ------------------------------------------------------------------------------


def test_the_grouped_bootstrap_equals_the_direct_one():
    """The resampled means of a count-matrix product equal x[idx].mean(axis=1) within 1e-12 relative, and so do the intervals."""
    rng = np.random.default_rng(7)
    t, n = 2500, 5
    x = 0.5 + rng.standard_normal((t, n))
    rule = pw.Rule.from_report2({"config": asdict(RULE_CFG)})
    seed = rule.seed("x", "y")
    for block in (1, 7, 40):
        direct = np.concatenate([x[idx].mean(axis=1) for idx in r2._chunks(t, block, rule.n_boot, seed)])
        grouped = pw.grouped_means(pw.count_matrix(t, block, rule.n_boot, seed), x, product_columns=2)
        np.testing.assert_allclose(grouped, direct, rtol=1e-12, atol=0.0)
    for j in range(n):  # the intervals of report2.mean_interval, with its own Politis-White block
        xj = x[:, j]
        ref = r2.mean_interval(xj, rule.report2_config(), "diff", "x", "y")
        block = r2._block(xj)
        low, high, q95 = pw.interval_from_means(pw.grouped_means(pw.count_matrix(t, block, rule.n_boot, seed), xj[:, None]), rule.level)
        assert block == ref["block"]
        for mine, theirs in ((low[0], ref["low"]), (high[0], ref["high"]), (q95[0], ref["q95"])):
            assert mine == pytest.approx(theirs, rel=1e-12, abs=0.0)


def test_the_seed_and_the_comparisons_are_the_registered_ones():
    rule = pw.Rule.from_report2({"config": asdict(r2.Report2Config())})
    assert rule == pw.Rule()
    for _, a, b in r2.COMPARISONS:
        assert rule.seed(a, b) == r2._seed(r2.Report2Config(), "diff", a, b)
    comps = pw.report2_comparisons()
    assert [(c.hypothesis, c.a, c.b) for c in comps] == list(r2.COMPARISONS)
    assert [c.kind for c in comps] == ["superiority"] * 4 + ["non-inferiority"] and comps[-1].margin == 0.005
    cfg = pw.PowerConfig()
    assert cfg.registered and cfg.flip_seed == 20261030 and cfg.block_lengths == (24, 48, 168, 720)
    assert (cfg.size_replicates, cfg.power_replicates) == (2000, 1000)
    assert cfg.deltas == (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01)
    assert (cfg.concentrated_block, cfg.concentrated_span, cfg.concentrated_share) == (168, 720, 0.1)
    with pytest.raises(ValueError, match="concentrated"):
        pw.PowerConfig(block_lengths=(24,)).check()  # the concentrated grid needs its L in the design


def test_the_concentrated_shift_has_mean_delta_on_a_tenth_of_the_720_hour_blocks():
    """On Report n°2's 50,626 hours: 71 circular blocks of 720 hours (the last of 226), 7 of them chosen per replicate. The
    weights are equal on the chosen hours, 0 elsewhere, and their mean is exactly 1, so the replicate's mean moves by delta."""
    cfg, comp, t = pw.PowerConfig(power_replicates=200), pw.Comparison("H3", "matrix-1y", "knn-1y"), 50_626
    design = pw.concentrated_design(cfg, t)
    assert (design["span_blocks"], design["chosen"], design["last_block_hours"]) == (71, 7, 226)
    assert pw.chosen_blocks(cfg, 7200) == (10, 1) and pw.chosen_blocks(cfg, 7201) == (11, 1)
    with pytest.raises(ValueError, match="rounds to none"):
        pw.chosen_blocks(cfg, 3600)  # 5 blocks: 10 % rounds to none, and the share is not quietly changed
    with pytest.raises(ValueError, match="rounds to none"):
        pw.study(arms(np.ones(3600) + np.arange(3600) % 3, np.random.default_rng(0)), [pw.Comparison("S", "x", "y")], pw.Rule(), cfg)
    assert design["weight_whole_blocks"] == pytest.approx(50_626 / 5_040)
    offsets, chosen = pw.concentration_plan(cfg, comp, t)
    again = pw.concentration_plan(cfg, comp, t)
    assert np.array_equal(offsets, again[0]) and np.array_equal(chosen, again[1])
    assert offsets.shape == (200,) and chosen.shape == (200, 7) and np.all((offsets >= 0) & (offsets < 720))
    assert all(np.unique(row).size == 7 for row in chosen) and chosen.max() <= 70
    assert not np.array_equal(offsets, pw.flip_plan(cfg, comp, 720, t)[0][:200])  # its own stream, not the flips'
    whole = 0
    for r in range(200):
        w = pw.concentration(t, 720, offsets[r], chosen[r])
        assert w.mean() == pytest.approx(1.0, rel=1e-12, abs=0.0)
        on = w > 0
        blocks = pw.block_index(t, 720, offsets[r])
        assert set(np.unique(blocks[on]).tolist()) == set(chosen[r].tolist())  # whole blocks, the chosen ones
        assert np.unique(w[on]).size == 1 and w[on][0] == pytest.approx(t / on.sum())
        whole += int(on.sum() == 7 * 720)
    assert 150 <= whole < 200  # the short block is chosen about 7/71 of the time
    c = np.random.default_rng(0).standard_normal(t)
    job = pw._Job(c - c.mean(), comp, cfg, pw.Rule(), {168: tuple(pw.design_points(comp, cfg, 1.0, 168))})
    reps = pw._Replicates(job)
    points = job.points[168]
    assert [p.name for p in points].count("concentrated") == len(cfg.deltas) and [p.name for p in points].count("power") == len(cfg.deltas)
    base = reps.base(168, 3)
    for p in points:
        x = reps.shifted(base, p, 3)
        assert x.mean() - base.mean() == pytest.approx(p.shift, rel=1e-9, abs=1e-15)
    assert all(p.name != "concentrated" for L in (24, 48, 720) for p in pw.design_points(comp, cfg, 1.0, L))
    assert all(p.name != "concentrated" for p in pw.design_points(pw.Comparison("H12", "a", "b", 0.005), cfg, 1.0, 168))


def test_a_concentrated_gain_is_harder_to_see_than_a_constant_one():
    """The same flips and the same mean shift: the replicate's mean is the same, but the concentrated shift adds a slow
    component that widens DM's standard error and the bootstrap's interval, so its power is at most the constant one's."""
    rng = np.random.default_rng(8)
    ll = arms(0.05 * ar1(0.3, 7200, rng), rng)
    deltas = (0.0005, 0.001, 0.002, 0.004)
    cfg = pw.PowerConfig(block_lengths=(168,), size_replicates=200, power_replicates=200, deltas=deltas)
    res = pw.study(ll, [pw.Comparison("S", "x", "y")], pw.Rule.from_report2({"config": asdict(RULE_CFG)}), cfg, workers=1)
    e = res["comparisons"][0]
    const = [p["pass_registered"]["rate"] for p in e["power"]["168"]["points"]]
    conc = [p["pass_registered"]["rate"] for p in e["power_concentrated"]["168"]["points"]]
    assert [p["delta_share"] for p in e["power_concentrated"]["168"]["points"]] == list(deltas)
    assert all(b <= a for a, b in zip(const, conc, strict=True)) and sum(conc) < sum(const), (const, conc)
    assert res["concentrated"]["chosen"] == 1 and res["concentrated"]["span_blocks"] == 10
    assert res["replicates"] == 200 + 2 * 4 * 200
    det_c, det_k = e["power"]["168"]["detectable"], e["power_concentrated"]["168"]["detectable"]
    assert det_k["grid"] is None or det_c["grid"] <= det_k["grid"], (det_c, det_k)


def test_the_flips_are_signs_on_circular_blocks():
    t, length = 50, 7
    c = np.arange(1.0, t + 1.0)
    for offset in range(length):
        blocks = pw.block_index(t, length, offset)
        sizes = np.bincount(blocks)
        assert sorted(sizes.tolist()) == [t % length] + [length] * (t // length)
        for k in range(sizes.size):  # each block is one run of consecutive hours on the circle
            hours = np.flatnonzero(blocks == k)
            assert sum((h + 1) % t not in set(hours) for h in hours) == 1
        assert blocks[offset] == 0
    cfg = pw.PowerConfig(size_replicates=20)
    comp = pw.Comparison("H", "x", "y")
    offsets, signs = pw.flip_plan(cfg, comp, length, t)
    again = pw.flip_plan(cfg, comp, length, t)
    assert np.array_equal(offsets, again[0]) and np.array_equal(signs, again[1])
    assert not np.array_equal(signs, pw.flip_plan(cfg, pw.Comparison("H", "x", "z"), length, t)[1])
    assert set(np.unique(signs).tolist()) == {-1.0, 1.0} and np.all((offsets >= 0) & (offsets < length))
    x = pw.flipped(c, length, offsets[0], signs[0])
    assert np.array_equal(np.abs(x), c)
    blocks = pw.block_index(t, length, offsets[0])
    assert all(len(set(np.sign(x[blocks == k]).tolist())) == 1 for k in range(blocks.max() + 1))


# --- size and power -------------------------------------------------------------------------


def _size(result: dict, length: int) -> dict:
    return result["comparisons"][0]["size"][str(length)]


def test_iid_differences_give_a_size_near_five_percent():
    """Tolerance: DM at 5 % +- 1.5 points, about three binomial standard errors at 2,000 replicates (0.49 point each); the
    pass rule, which needs both tests, in [2.5 %, 6.5 %] and never above either test alone."""
    rng = np.random.default_rng(11)
    ll = arms(0.05 * rng.standard_normal(4000), rng)
    cfg = pw.PowerConfig(block_lengths=(24,), concentrated_block=None, size_replicates=2000, power_replicates=10, deltas=(0.01,))
    res = pw.study(ll, [pw.Comparison("T", "x", "y")], pw.Rule.from_report2({"config": asdict(RULE_CFG)}), cfg, workers=1)
    s = _size(res, 24)
    assert 0.035 <= s["dm"]["rate"] <= 0.065, s["dm"]
    assert 0.025 <= s["pass_two_sided"]["rate"] <= 0.065, s["pass_two_sided"]
    assert s["pass_two_sided"]["rate"] <= min(s["dm"]["rate"], s["bootstrap"]["rate"])  # both tests on the same side
    assert s["pass_registered"]["rate"] <= s["pass_two_sided"]["rate"]
    assert 0.85 <= s["variance_ratio"] <= 1.15 and s["undefined"] == 0
    lo, hi = s["dm"]["wilson95"]
    assert lo < s["dm"]["rate"] < hi and s["dm"]["n"] == 2000
    assert res["size_alerts"] == [] and res["replicates"] == 2000 + 10


def flip_variance_ratio(rho: float, length: float, lags: int = 23) -> float:
    """For sign flips in blocks of L of a unit AR(1), lag k keeps (1 - k/L)+ of its autocovariance rho^k: the replicate's
    long-run variance over what the rectangular kernel with ``lags`` lags estimates. L = inf: the AR(1) itself."""
    k = np.arange(1, 20_000)
    w = np.ones(k.size) if math.isinf(length) else np.clip(1.0 - k / length, 0.0, None)
    g = w * rho**k
    return float((1.0 + 2.0 * g.sum()) / (1.0 + 2.0 * g[:lags].sum()))


def test_an_ar1_difference_at_one_day_blocks_gives_what_the_theory_of_the_flips_predicts():
    """AR(1), rho = 0.9, L = 24 hours.

    Flips in blocks of 24 keep lags 1-23 of the dependence (scaled by 1 - k/24) and kill the rest. The
    rectangular kernel's 23 lags then cover the replicate's long-run variance exactly: the theory
    predicts a variance ratio of 1 and DM's nominal 5 %, with no inflation. The AR(1) itself has a
    ratio of 1.092 (DM about 6.1 %): one-day blocks cannot show an excess size that comes from
    dependence longer than a day. Measured here (T = 6,000, 2,000 replicates): DM 5.9 %, the
    bootstrap 7.8 %, the pass rule two-sided 5.7 %, variance ratio 0.99.
    """
    assert flip_variance_ratio(0.9, 24) == pytest.approx(1.0, abs=1e-12)
    assert flip_variance_ratio(0.9, math.inf) == pytest.approx(1.0917, abs=1e-4)
    predicted = 2.0 * st.norm.sf(1.959963984540054 / math.sqrt(flip_variance_ratio(0.9, math.inf)))
    assert predicted == pytest.approx(0.0607, abs=1e-4)
    rng = np.random.default_rng(2)
    ll = arms(0.05 * ar1(0.9, 6000, rng), rng)
    cfg = pw.PowerConfig(block_lengths=(24,), concentrated_block=None, size_replicates=2000, power_replicates=10, deltas=(0.01,))
    s = _size(pw.study(ll, [pw.Comparison("T", "x", "y")], pw.Rule.from_report2({"config": asdict(RULE_CFG)}), cfg, workers=1), 24)
    assert 0.9 <= s["variance_ratio"] <= 1.1, s["variance_ratio"]
    assert 0.035 <= s["dm"]["rate"] <= 0.08, s["dm"]
    assert s["pass_two_sided"]["rate"] <= 0.08 and s["bootstrap"]["rate"] <= 0.11, s
    assert s["blocks"]["median"] > 10  # Politis-White sees the dependence on every replicate


def test_the_flip_moments_are_exact_over_every_offset_and_sign():
    """On 30 hours in blocks of 7 (five blocks, the last of 2), every offset and every one of the 2^5 sign vectors:
    the mean squared replicate mean and the mean of the replicate's raw 3-lag rectangular sum equal flip_moments."""
    c = np.random.default_rng(4).standard_normal(30)
    c -= c.mean()
    t, length, lags = c.size, 7, 3
    n_sum = d_sum = 0.0
    count = 0
    for offset in range(length):
        blocks = pw.block_index(t, length, offset)
        for bits in range(2 ** (blocks.max() + 1)):
            signs = np.array([1.0 if bits >> b & 1 else -1.0 for b in range(blocks.max() + 1)])
            x = signs[blocks] * c
            n_sum += t * x.mean() ** 2
            d_sum += float(x @ x) / t + 2.0 * sum(float(x[:-k] @ x[k:]) / t for k in range(1, lags + 1))
            count += 1
    n, d = pw.flip_moments(c, length, lags)
    assert n == pytest.approx(n_sum / count, rel=1e-12) and d == pytest.approx(d_sum / count, rel=1e-12)


def test_the_predicted_variance_ratio_matches_the_theory_of_an_ar1():
    """AR(1), rho = 0.97, T = 200,000. A flip in blocks of L keeps lag k with weight (L - k)+ / L: the population ratio is
    sum_k w_k rho^k over sum_{|k| <= 23} w_k rho^k, 1 at L = 24, 1.680 at 168 and 1.894 at 720; with no flips, the
    Bartlett sum up to 720 over the rectangular one up to 23, 1.867. The prediction, from the series alone, lands within
    its sampling error (about 3 %, measured on three seeds; tolerance 7 %)."""
    rng = np.random.default_rng(0)
    c = ar1(0.97, 200_000, rng)
    c -= c.mean()
    assert pw.predicted_ratio(c, 24, 24) == pytest.approx(1.0, abs=1e-3)
    for length in (168, 720):
        assert pw.predicted_ratio(c, length, 24) == pytest.approx(flip_variance_ratio(0.97, length), rel=0.07), length
    k = np.arange(1, 20_000)
    no_flip = (1 + 2 * (np.clip(1 - k / 720, 0, None) * 0.97**k).sum()) / (1 + 2 * (0.97 ** k[:23]).sum())
    pred = pw.variance_prediction(c, (24, 168, 720), 24)
    assert pred["no_flips"]["ratio"] == pytest.approx(no_flip, rel=0.07) and pred["no_flips"]["bartlett_bandwidth"] == 720
    assert pred["no_flips"]["dm_size_implied"] == pytest.approx(pw.dm_size_from_ratio(pred["no_flips"]["ratio"]))
    assert pred["no_flips"]["dm_size_implied"] > 0.1  # a 1.9 ratio: DM near 15 % on this dependence


def test_the_measured_variance_ratio_falls_in_its_predicted_band():
    """AR(1), rho = 0.9, T = 6,000, 1,000 size replicates at L = 24, 168 and 720: every measured ratio lies within three
    Monte-Carlo standard errors of the ratio predicted from the series before the replicates were read."""
    rng = np.random.default_rng(2)
    ll = arms(0.05 * ar1(0.9, 6000, rng), rng)
    cfg = pw.PowerConfig(block_lengths=(24, 168, 720), concentrated_block=None, size_replicates=1000, power_replicates=10, deltas=(0.01,))
    res = pw.study(ll, [pw.Comparison("T", "x", "y")], pw.Rule.from_report2({"config": asdict(RULE_CFG)}), cfg, workers=1)
    assert res["variance_ratio_outside_band"] == []
    for length in (24, 168, 720):
        s = _size(res, length)
        lo, hi = s["variance_ratio_band"]
        assert lo <= s["variance_ratio"] <= hi and 0.0 < s["variance_ratio_se"] < 0.1, (length, s["variance_ratio"], lo, hi)
        assert hi - lo == pytest.approx(6.0 * s["variance_ratio_se"]) and s["variance_ratio_outside_band"] is False
    nf = res["comparisons"][0]["variance_no_flips"]
    assert nf["bartlett_bandwidth"] == 720 and nf["ratio_se"] == pytest.approx(nf["ratio"] * math.sqrt(4 * 720 / (3 * 6000)))


def test_power_rises_with_delta_and_h12_holds_more_often_at_zero_than_on_its_margin():
    rng = np.random.default_rng(5)
    ll = arms(0.1 * rng.standard_normal(3000), rng)
    deltas = (0.001, 0.003, 0.006, 0.012)
    cfg = pw.PowerConfig(block_lengths=(24,), concentrated_block=None, size_replicates=300, power_replicates=300, deltas=deltas)
    comps = [pw.Comparison("S", "x", "y"), pw.Comparison("N", "x", "y", margin=0.005)]
    res = pw.study(ll, comps, pw.Rule.from_report2({"config": asdict(RULE_CFG)}), cfg, workers=1)
    sup, ni = res["comparisons"]
    curve = sup["power"]["24"]["points"]
    power = [p["pass_registered"]["rate"] for p in curve]
    assert [p["delta_share"] for p in curve] == list(deltas)
    assert all(b >= a for a, b in zip(power, power[1:], strict=False)), power  # common random numbers: monotone
    assert power[0] <= 0.3 and power[-1] >= 0.95, power
    det = sup["power"]["24"]["detectable"]
    assert det["grid"] in deltas[1:] and det["grid"] == next(d for d, p in zip(deltas, power, strict=True) if p >= 0.8)
    i = deltas.index(det["grid"])
    assert deltas[i - 1] <= det["interpolated"] <= det["grid"]
    assert ni["kind"] == "non-inferiority" and ni["margin"] == pytest.approx(0.005 * res["climatology_log_loss"])
    on_margin, at_zero = ni["size"]["24"]["holds"]["rate"], ni["power"]["24"]["at_zero"]["holds"]["rate"]
    assert on_margin <= 0.1 and at_zero > on_margin + 0.3, (on_margin, at_zero)
    assert pw.detectable([0.1, 0.2], [0.85, 0.9], 0.8)["interpolated"] is None
    assert pw.detectable([0.1, 0.2], [0.5, 0.7], 0.8)["grid"] is None
    assert pw.power_at(0.15, [0.1, 0.2], [0.4, 0.6]) == pytest.approx(0.5) and pw.power_at(0.05, [0.1, 0.2], [0.4, 0.6]) is None


# --- the command ----------------------------------------------------------------------------


def test_the_digest_check_refuses_a_changed_file(tmp_path, capsys):
    rng = np.random.default_rng(3)
    comps = [pw.Comparison("S", "x", "y")]
    run = runs_dir() / "run"
    digest = fake_run(run, arms(0.05 * rng.standard_normal(500), rng), comps)
    path = run / r2.SERIES_FILE
    arrays, report, _, _ = pw.load_series(path, digest.upper())
    assert report["series"]["sha256"] == digest and arrays["log_loss"].shape == (3, 500)
    with pytest.raises(pw.SeriesRefused, match="not the"):
        pw.load_series(path, "0" * 64)
    data = bytearray(path.read_bytes())
    data[-200] ^= 1  # one bit of the archive
    path.write_bytes(bytes(data))
    with pytest.raises(pw.SeriesRefused, match="not the"):
        pw.load_series(path, digest)  # the digest given no longer matches
    changed = r2.sha256_file(path)
    with pytest.raises(pw.SeriesRefused, match="report2.json records"):
        pw.load_series(path, changed)  # the file's own digest, but its run vouches for another
    assert pw.main(["--series", str(path), "--sha256", changed, "--out", str(tmp_path / "out")]) == 2
    assert "REFUSED" in capsys.readouterr().out and not (tmp_path / "out").exists()
    (run / "report2.json").unlink()
    with pytest.raises(pw.SeriesRefused, match="no report2.json"):
        pw.load_series(path, changed)


def test_the_path_is_refused_before_a_byte_is_read(tmp_path, monkeypatch):
    """Only a file under the runs root, symlinks resolved, and under no data root or the vault key's folder: refused with
    the right digest too, so the refusal comes before the bytes."""
    rng = np.random.default_rng(6)
    comps = [pw.Comparison("S", "x", "y")]
    ll = arms(0.05 * rng.standard_normal(300), rng)
    outside = tmp_path / "elsewhere"
    digest = fake_run(outside, ll, comps)
    with pytest.raises(pw.SeriesRefused, match="not a file under the runs root"):
        pw.load_series(outside / r2.SERIES_FILE, digest)
    with pytest.raises(pw.SeriesRefused, match="cannot be resolved"):
        pw.load_series(runs_dir() / "missing" / r2.SERIES_FILE, digest)
    key_folder = runs_dir() / "vault" / "copy"
    assert fake_run(key_folder, ll, comps) == digest
    with pytest.raises(pw.SeriesRefused, match="never opens"):
        pw.load_series(key_folder / r2.SERIES_FILE, digest)
    inside_data = runs_dir() / "store"
    assert fake_run(inside_data, ll, comps) == digest
    monkeypatch.setenv("OLSPRIME_DATA_DIR", str(inside_data))  # a data root that happens to sit under the runs root
    with pytest.raises(pw.SeriesRefused, match="never opens"):
        pw.load_series(inside_data / r2.SERIES_FILE, digest)
    link = runs_dir() / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:  # Windows without the symlink privilege: the resolution itself is covered above
        pytest.skip("symlinks cannot be created here")
    with pytest.raises(pw.SeriesRefused, match="not a file under the runs root"):
        pw.load_series(link / r2.SERIES_FILE, digest)


def test_a_series_whose_rule_does_not_reproduce_report2_is_refused():
    """The auditor's plant: x's hourly log-loss rolled by one hour, the archive rewritten and its digest updated in
    report2.json, the comparisons left as they were. The digest checks pass; the recomputed rule does not, and the study
    refuses before any replicate. Below the bound a difference is recorded; a verdict that differs is refused."""
    rng = np.random.default_rng(12)
    comps = [pw.Comparison("S", "x", "y")]
    run = runs_dir() / "planted"
    ll = arms(0.05 * ar1(0.5, 800, rng), rng)
    fake_run(run, ll, comps)
    names = list(ll)
    planted = {**ll, "x": np.roll(ll["x"], 1)}
    arrays = {"arms": np.array(names), "hours": np.arange(800, dtype=np.int64) * 3_600_000, "count": np.ones(800, dtype=np.int64),
              "log_loss": np.stack([planted[a] for a in names])}  # fmt: skip
    r2.write_npz(run / r2.SERIES_FILE, arrays)
    digest = r2.sha256_file(run / r2.SERIES_FILE)
    report = json.loads((run / "report2.json").read_text(encoding="utf-8"))
    report["series"]["sha256"] = digest
    r2._write_json(run / "report2.json", report)
    assert pw.load_series(run / r2.SERIES_FILE, digest)[0]["log_loss"].shape == (3, 800)  # the digests agree
    cfg = pw.PowerConfig(block_lengths=(24,), concentrated_block=None, size_replicates=20, power_replicates=10, deltas=(0.01,))
    with pytest.raises(pw.SeriesRefused, match="beyond rounding"):
        pw.run_study(run / r2.SERIES_FILE, digest, runs_dir() / "out", cfg=cfg, comparisons=comps, provenance=PROVENANCE)
    assert not (runs_dir() / "out").exists()
    mine = pw.observed_rule(ll, comps[0], pw.Rule.from_report2({"config": asdict(RULE_CFG)}), float(ll[r2.CLIMATOLOGY].mean()))
    recorded = report["comparisons"][0]
    assert pw.reproduction(mine, recorded)["identical"]
    rounding = pw.reproduction({**mine, "dm_se": mine["dm_se"] * (1 + 5e-10)}, recorded)
    assert rounding["differing"] == ["dm_se"] and rounding["max_relative_difference"] <= pw.REPRODUCTION_BOUND
    with pytest.raises(pw.SeriesRefused, match="dm_se"):
        pw.reproduction({**mine, "dm_se": mine["dm_se"] * (1 + 2e-9)}, recorded)
    with pytest.raises(pw.SeriesRefused, match="passes"):
        pw.reproduction({**mine, "passes": not mine["passes"]}, recorded)
    # every recorded value of the comparison is compared, the count of exactly identical hours and the kernel included
    assert {"hours", "identical_hours", "share_of_hours_identical", "dm_kernel", "dm_lags", "dm_note"} <= set(mine)
    with pytest.raises(pw.SeriesRefused, match="identical_hours"):
        pw.reproduction({**mine, "identical_hours": mine["identical_hours"] + 1}, recorded)
    with pytest.raises(pw.SeriesRefused, match="dm_lags"):
        pw.reproduction({**mine, "dm_lags": mine["dm_lags"] - 1}, recorded)
    # climatology's mean: the same bound and the same words
    fake_run(run, ll, comps)  # the run as it was, digests and all
    digest = r2.sha256_file(run / r2.SERIES_FILE)
    report = json.loads((run / "report2.json").read_text(encoding="utf-8"))
    clim = report["arms"][r2.CLIMATOLOGY]["log_loss"]["estimate"]
    for moved, refused in ((clim * (1 + 1e-12), False), (clim * (1 + 1e-6), True)):
        report["arms"][r2.CLIMATOLOGY]["log_loss"]["estimate"] = moved
        r2._write_json(run / "report2.json", report)
        if refused:
            with pytest.raises(pw.SeriesRefused, match="climatology's mean hourly log-loss in the series differs from report2.json beyond rounding"):
                pw.run_study(run / r2.SERIES_FILE, digest, runs_dir() / "out2", cfg=cfg, comparisons=comps, provenance=PROVENANCE)
        else:
            assert pw.run_study(run / r2.SERIES_FILE, digest, runs_dir() / "out1", cfg=cfg, comparisons=comps,
                                provenance=PROVENANCE)["observed_reproduces_report2"] is True  # fmt: skip


def _dies_once(marker: str, value: int) -> int:
    """A worker call that kills its process the first time any worker runs it, then returns ``value``."""
    path = Path(marker)
    if not path.exists():
        path.write_text("died", encoding="utf-8")
        os._exit(3)
    return value


def test_a_broken_pool_is_rebuilt_and_the_missing_calls_run_again(tmp_path):
    """A worker that dies breaks the whole pool: it is rebuilt, the calls without a result run again, and the results come
    back in order. The restart is noted."""
    restarts: list[str] = []
    marker = str(tmp_path / "marker")
    out = pw._run_calls([(_dies_once, (marker, k)) for k in range(6)], 2, True, restarts)
    assert out == list(range(6)) and Path(marker).exists()
    assert 1 <= len(restarts) <= pw.POOL_RESTARTS and all("pool rebuilt" in r for r in restarts), restarts


def _stalls_once(marker: str, value: int) -> int:
    """A worker call that hangs the first time any worker runs it (a pool that never finishes), then returns ``value``."""
    path = Path(marker)
    if not path.exists():
        path.write_text("stalled", encoding="utf-8")
        time.sleep(3600)
    return value


def _raises_on_zero(k: int) -> int:
    """A worker call that fails on its own (not a pool failure) for k = 0, and is slow otherwise."""
    if k == 0:
        raise ValueError("planted failure of call 0")
    time.sleep(5)
    return k


def test_a_calls_own_exception_propagates_unchanged_and_leaves_no_worker():
    """A call that raises is not a pool failure: no restart, the exception comes back as it is, and every worker of the
    pool is stopped before it does."""
    before = {p.pid for p in multiprocessing.active_children()}
    restarts: list[str] = []
    with pytest.raises(ValueError, match="planted failure of call 0"):
        pw._run_calls([(_raises_on_zero, (k,)) for k in range(6)], 2, True, restarts)
    assert restarts == []
    assert not [p for p in multiprocessing.active_children() if p.pid not in before and p.is_alive()]


def test_a_stalled_pool_is_stopped_and_rebuilt(tmp_path):
    """No call finishes within the stall bound: the pool's workers are stopped, the pool rebuilt, the calls run again."""
    restarts: list[str] = []
    marker = str(tmp_path / "marker")
    out = pw._run_calls([(_stalls_once, (marker, k)) for k in range(2)], 1, True, restarts, stall_seconds=20.0)
    assert out == [0, 1] and Path(marker).exists()
    assert len(restarts) == 1 and "no call finished for 20 s" in restarts[0], restarts


def test_the_command_gives_the_same_bytes_with_one_worker_or_two():
    """The command's path (run_study) always runs a process pool with one BLAS thread per job, ``--workers 1`` included:
    one worker and the default cutting, then two workers and another cutting, give the same power.json and power.md.
    config.json differs only in the worker count it records."""
    rng = np.random.default_rng(9)
    comps = [pw.Comparison("S", "x", "y"), pw.Comparison("N", "x", "y", margin=0.005)]
    run = runs_dir() / "run"
    digest = fake_run(run, arms(0.05 * ar1(0.5, 1500, rng), rng), comps)
    cfg = pw.PowerConfig(block_lengths=(24, 168), concentrated_span=150, size_replicates=120, power_replicates=60, deltas=(0.002, 0.01))
    outs = [runs_dir() / "one", runs_dir() / "two"]
    cuts = ((1, pw.Batches()), (2, pw.Batches(replicates=7, columns=13, product_columns=5)))
    for out, (workers, batches) in zip(outs, cuts, strict=True):
        pw.run_study(run / r2.SERIES_FILE, digest, out, cfg=cfg, workers=workers, batches=batches, comparisons=comps, provenance=PROVENANCE)
    for name in ("power.json", "power.md"):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    configs = [json.loads((o / "config.json").read_text(encoding="utf-8")) for o in outs]
    assert [c.pop("execution") for c in configs] == [{"workers": w, "executor": "ProcessPoolExecutor (spawn)", "blas_threads_per_job": 1} for w in (1, 2)]
    assert configs[0] == configs[1]
    run_json = json.loads((outs[0] / "run.json").read_text(encoding="utf-8"))
    assert "finished_at" in run_json and set(run_json["stages_seconds"]) == {"dm_and_politis_white", "grouped_bootstrap", "variance_prediction"}
    assert run_json["memory_per_bootstrap_job_bytes"] == pw.job_memory(1500, pw.Batches(), RULE_CFG.n_boot)
    assert isinstance(run_json["pool_restarts"], list)  # restarts, if any, are noted beside the wall clock, never in power.json
    assert 0.6e9 < pw.job_memory(50_626, pw.Batches(), 2000) < 0.7e9  # the official run: about 0.67 GB per job
    config = configs[0]
    assert config["series"]["sha256"] == digest and config["series"]["run"] == "run" and config["registered"] is False
    assert config["config"]["flip_seed"] == 20261030
    text = (outs[0] / "power.json").read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=1) + "\n"  # sorted keys
    md = (outs[0] / pw.POWER_FILE).read_text(encoding="utf-8")
    assert "NOT THE REGISTERED STUDY" in md and "## Size" in md and "## Power" in md and "### Concentrated gain" in md
    assert md.index("| S: x - y | 24 |") < md.index("| S: x - y | 168 |")  # L in increasing order, not JSON's string order
    assert "lower bound of the detectable mean gain" in md and "a gain concentrated in regimes needs more" in md
    assert "conditional on the registered bootstrap seed" in md and "common random numbers across replicates" in md
    assert "predicted [band of 3 MC s.e.]" in md and "### The variance ratio with no flips" in md and "a process pool" in md
    power = json.loads(text)
    assert set(power["result"]["comparisons"][0]["power_concentrated"]) == {"168"}
    assert "power_concentrated" not in power["result"]["comparisons"][1]  # H12-like: non-inferiority has no concentrated grid


def test_on_report2s_synthetic_run_the_study_first_reproduces_every_comparison(two_runs):  # noqa: F811
    a, _, metrics, _ = two_runs
    run = runs_dir() / a.name  # the reader opens only a run under the runs root: the fixture's run, copied there
    run.mkdir()
    for name in (r2.SERIES_FILE, "report2.json", "config.json"):
        (run / name).write_bytes((a / name).read_bytes())
    cfg = pw.PowerConfig(block_lengths=(24, 168), concentrated_span=72, size_replicates=60, power_replicates=30, deltas=(0.001, 0.01))
    power = pw.run_study(run / r2.SERIES_FILE, metrics["series"]["sha256"], runs_dir() / "out", cfg=cfg, provenance=PROVENANCE)
    assert power["observed_reproduces_report2"] is True and len(power["observed"]) == len(r2.COMPARISONS)
    assert all(v == {"identical": True, "differing": [], "max_relative_difference": 0.0} for v in power["observed"].values())
    arrays, _, config, _ = pw.load_series(run / r2.SERIES_FILE, metrics["series"]["sha256"])
    ll = {str(name): arrays["log_loss"][k] for k, name in enumerate(arrays["arms"])}
    recorded = next(c for c in metrics["comparisons"] if c["hypothesis"] == "H11")
    clim = metrics["arms"][r2.CLIMATOLOGY]["log_loss"]["estimate"]
    mine = pw.observed_rule(ll, pw.Comparison("H11", recorded["a"], recorded["b"]), pw.Rule.from_report2(config), clim)
    assert pw.reproduction(mine, recorded)["identical"]
    moved = {**mine, "dm_p": mine["dm_p"] * (1 + 1e-12), "bootstrap": {**mine["bootstrap"], "low": mine["bootstrap"]["low"] * (1 + 2e-12)}}
    rep = pw.reproduction(moved, recorded)
    assert rep["differing"] == ["bootstrap.low", "dm_p"] and 1e-12 <= rep["max_relative_difference"] <= 3e-12
    res = power["result"]
    assert res["hours"] == metrics["scored"]["hours"]
    assert res["climatology_log_loss"] == metrics["arms"][r2.CLIMATOLOGY]["log_loss"]["estimate"]
    assert [c["hypothesis"] for c in res["comparisons"]] == [h for h, _, _ in r2.COMPARISONS]
    h12 = res["comparisons"][-1]
    assert h12["kind"] == "non-inferiority" and set(h12["power"]["24"]) == {"at_zero"}
    assert h12["margin"] == pytest.approx(metrics["verdicts"]["H12"]["margin"], rel=1e-15)
    assert all(set(c["size"]) == {"24", "168"} for c in res["comparisons"])
    assert [r["hypothesis"] for r in power["readings"]] == ["H3", "H3", "H10", "H11"]
    md = (runs_dir() / "out" / pw.POWER_FILE).read_text(encoding="utf-8")
    assert "reproduces every comparison of the run's `report2.json`: yes" in md
