"""Report n°3 (``olsprime.bench.report3``) on a synthetic store, CPU torch.

- the fits stage saves every part, ``--arms`` restricts it, a run cut in the middle and resumed gives the
  same bytes as a run straight through, and a changed configuration or code refuses to resume;
- the ensemble is the mean of the seeds' probabilities before the floor;
- the score stage, on synthetic hourly series: H4's two bounds against a direct computation, the
  testability gate both ways, H5's rule and H13's, including "H4 does not hold";
- the pairing refuses a missing hour and a shifted row;
- ``--check`` passes on an unchanged run and fails on one changed digit, in an output or in a part;
- the timing protocol: its warm-ups, the ABAB order and the scaling to 10^4, on a fake clock and on CPU;
- the FLOPs formulas against torch's counter of matrix products, and the cost table's columns.

Every test runs on Report n°2's fake store (``tests/test_report2.py``): nothing reads the repository's ``data/``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats as st

from olsprime.bench import report2 as r2
from olsprime.bench import report3 as r3
from olsprime.eval import stats
from olsprime.models.encoder import EncoderConfig, EncoderModel
from olsprime.models.matrix import prob_map
from olsprime.train import checkpoint
from tests.test_report2 import CFG as R2CFG
from tests.test_report2 import FAKE_MANIFEST, fake_root  # noqa: F401 (fixture: Report n°2's synthetic store)

PROV = {"git_commit": "c" * 40, "code": {"sha256": "d" * 64, "files": 1}, "manifest_digest": "0" * 64}
TINY = (("d_model", 16), ("n_heads", 2), ("d_mlp", 32), ("n_layers", 1))
SHORT = (("epochs", 2), ("warmup_steps", 3), ("eval_batch", 256))
CFG = r3.Report3Config(years=(2019,), seeds=(1, 2), device="cpu", encoder=TINY, fit=SHORT, timing_batch=16, timing_batches=3, timing_warmups=3,
                       timing_repeats=4, synthetic="test: Report n°2's fake store")  # fmt: skip


@pytest.fixture(scope="module")
def report2_run(tmp_path_factory, fake_root) -> r3.Report2Run:  # noqa: F811
    """Report n°2 on the fake store, read by its digest from a runs root of this module (the reader opens nothing else)."""
    root, universe = fake_root
    runs = tmp_path_factory.mktemp("runs-report3")
    out = runs / "report2"
    r2.run_report(R2CFG, out, data_dir=root, universe=universe, manifest=FAKE_MANIFEST, provenance=PROV)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OLSPRIME_RUNS_DIR", str(runs))
        return r3.read_report2(out / r2.SERIES_FILE, r2.sha256_file(out / r2.SERIES_FILE))


@pytest.fixture(scope="module")
def inputs(report2_run, fake_root) -> r3.Inputs:  # noqa: F811
    root, universe = fake_root
    panel, bars = r3.load_panel(report2_run.config, data_dir=root, universe=universe)
    return r3.build_inputs(report2_run, panel, bars, CFG)


@pytest.fixture(scope="module")
def full(inputs, tmp_path_factory) -> tuple[Path, dict]:
    """Every stage straight through: fits, timing, the Matrix timing and the score."""
    out = tmp_path_factory.mktemp("report3")
    r3.run(inputs, CFG, out, "fits", prov=PROV)
    r3.run(inputs, CFG, out, "timing", prov=PROV)
    r3.run(inputs, CFG, out, "matrix-timing", prov=PROV, cpu_name="test CPU")
    metrics = r3.run(inputs, CFG, out, "score", prov=PROV)
    return out, metrics


def _part_files(out: Path) -> dict[str, bytes]:
    """Every saved file but the wall clocks (timing.json) and the timing stages' outputs."""
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted((out / "parts" / "fits").rglob("*"))
            if p.is_file() and not p.name.startswith("timing.json")}  # fmt: skip


# --- The panel and the fits ----------------------------------------------------------------------


def test_the_panel_is_report2s_own_and_its_scored_rows_pair_one_for_one(inputs, report2_run):
    a = inputs.data.arrays
    assert r3.fingerprint(inputs.panel) == report2_run.fingerprint
    s = report2_run.series
    np.testing.assert_array_equal(a.decision_ms[inputs.scored], s["decision_time"])
    np.testing.assert_array_equal(a.label[inputs.scored], s["label"])
    assert set(inputs.years) == {2019} and inputs.years[2019].size == inputs.scored.size
    shifted = dataclasses.replace(report2_run, series={**s, "decision_time": s["decision_time"] + np.where(np.arange(s["label"].size) == 5, 3_600_000, 0)})
    with pytest.raises(r3.PairingRefused):
        r3.scored_panel_rows(a, shifted)


def test_every_part_is_saved_with_its_digests(full):
    out, _ = full
    parts = r3.Parts(out, None)
    for arm in CFG.arms:
        for seed in CFG.seeds:
            rec = r3.verified_part(parts, r3.part_key(arm, seed, 2019), CFG)
            assert rec is not None and rec["arm"] == arm and rec["seed"] == seed
            assert len(rec["epoch_digests"]) == 2 and len(rec["slice_losses"]) == 2 and rec["kept_epoch"] in (1, 2)
            fc = r3.read_forecast(parts, r3.part_key(arm, seed, 2019))
            assert fc["raw"].shape == (rec["forecast_rows"], 3) and np.isfinite(fc["raw"]).all()
            arrays, meta = checkpoint.from_bytes(parts.get(r3.part_key(arm, seed, 2019) + "/fit.safetensors"))
            assert json.loads(meta["epoch_digests"]) == rec["epoch_digests"]
            assert parts.get_json(r3.part_key(arm, seed, 2019) + "/timing.json")["seconds"] > 0
    assert parts.identity()["series_sha256"] == json.loads((out / "report3.json").read_text())["report2"]["series_sha256"]


def test_arms_restrict_the_stage_and_a_resumed_run_gives_the_same_bytes(inputs, full, tmp_path):
    straight, _ = full
    out = tmp_path / "resumed"
    r3.run(inputs, CFG, out, "fits", prov=PROV, arms=("enc-ols",))
    done = sorted(p.parent.relative_to(out / "parts" / "fits").as_posix() for p in (out / "parts" / "fits").rglob("part.json"))
    assert done == ["enc-ols/s1/2019", "enc-ols/s2/2019"]
    with pytest.raises(r3.Interrupted):
        r3.run(inputs, CFG, out, "fits", prov=PROV, stop_after=1)
    assert len(list((out / "parts" / "fits").rglob("part.json"))) == 3
    r3.run(inputs, CFG, out, "fits", prov=PROV)
    assert _part_files(out) == _part_files(straight)
    assert (out / "parts" / "IDENTITY.json").read_bytes() == (straight / "parts" / "IDENTITY.json").read_bytes()


def test_a_changed_config_or_code_refuses_to_resume(inputs, full):
    out, _ = full
    before = _part_files(out)
    other = dataclasses.replace(CFG, fit=(("epochs", 3), ("warmup_steps", 3), ("eval_batch", 256)))
    with pytest.raises(r3.ResumeRefused, match="config"):
        r3.run(inputs, other, out, "fits", prov=PROV)
    with pytest.raises(r3.ResumeRefused, match="code_sha256"):
        r3.run(inputs, CFG, out, "fits", prov={**PROV, "code": {"sha256": "e" * 64, "files": 1}})
    with pytest.raises(r3.ResumeRefused, match="config"):
        r3.run(inputs, other, out, "score", prov=PROV)
    # every stage binds the fits' code and data manifest, not only the fits stage (audit 4f, minor 3)
    for stage in ("timing", "matrix-timing", "score"):
        with pytest.raises(r3.ResumeRefused, match="code_sha256"):
            r3.run(inputs, CFG, out, stage, prov={**PROV, "code": {"sha256": "e" * 64, "files": 1}})
        with pytest.raises(r3.ResumeRefused, match="manifest_digest"):
            r3.run(inputs, CFG, out, stage, prov={**PROV, "manifest_digest": "1" * 64})
    assert _part_files(out) == before


def test_the_device_is_no_part_of_the_identity_but_the_fits_share_one(inputs, full, report2_run):
    """audit 4f, minor 6: --device on a stage that does not train changes nothing; a resume on another device is refused."""
    out, _ = full
    on_gpu = dataclasses.replace(CFG, device="cuda")
    assert r3.config_json(on_gpu) == r3.config_json(CFG) and "device" not in r3.config_json(CFG)
    assert r3.check(out, report2_run, on_gpu, PROV) == 0  # the re-score with another --device gives the same bytes
    with pytest.raises(r3.ResumeRefused, match="one device"):
        r3.run(inputs, on_gpu, out, "fits", prov=PROV)
    assert dataclasses.replace(r3.Report3Config(), device="cpu").registered


def _copy_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file():
            (dst / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            (dst / p.relative_to(src)).write_bytes(p.read_bytes())


def _plant(parts: r3.Parts, src: str, dst: str, *, forge: str) -> None:
    """Copy part ``src`` over part ``dst``, digests and all; ``forge`` how far: 'copy' as is, 'record' rewrites part.json to
    name ``dst`` (resealed), 'forecast' keeps ``dst``'s own checkpoint and part.json but swaps in ``src``'s forecast."""
    names = ("fit.safetensors", "forecast.safetensors", "part.json", "timing.json")
    if forge == "forecast":
        data = parts.get(f"{src}/forecast.safetensors")
        sha = parts.put(f"{dst}/forecast.safetensors", data)
        rec = parts.get_json(f"{dst}/part.json")
        parts.put_json(f"{dst}/part.json", {**rec, "forecast_sha256": sha})
        return
    for n in names:
        parts.put(f"{dst}/{n}", parts.get(f"{src}/{n}"))
    if forge == "record":
        arm, seed, year = r3.parse_key(dst)
        rec = parts.get_json(f"{dst}/part.json")
        fc = r3.fit_config(CFG, arm, seed, year)
        parts.put_json(f"{dst}/part.json", {**rec, "arm": arm, "seed": seed, "year": year, "fit_ms": fc.fit_ms, "fit_config": json.loads(fc.to_json())})


@pytest.mark.parametrize(("src", "forge", "what"), [
    (("enc-softmax", 2), "copy", "part.json: arm"),  # the auditor's copy: another arm's part, digests and all
    (("enc-ols", 1), "copy", "part.json: seed"),  # another seed's
    (("enc-softmax", 2), "record", "fit.safetensors: encoder.arm"),  # part.json forged too: the checkpoint's config still says softmax
    (("enc-ols", 1), "record", "fit.safetensors: seed"),
    (("enc-softmax", 2), "forecast", "forecast.safetensors: arm"),  # only the forecast swapped, part.json resealed around it
])  # fmt: skip
def test_a_part_copied_into_another_folder_is_refused(inputs, full, report2_run, tmp_path, src, forge, what):
    """audit 4f, major 1: a part verifies its digests and its place: arm, seed, year and fit instant, in every file."""
    straight, _ = full
    out = tmp_path / "planted"
    _copy_tree(straight, out)
    parts = r3.Parts(out, None)
    _plant(parts, r3.part_key(*src, 2019), r3.part_key("enc-ols", 2, 2019), forge=forge)
    with pytest.raises(r3.PartRefused, match=what):
        r3.verified_part(parts, r3.part_key("enc-ols", 2, 2019), CFG)
    with pytest.raises(r3.PartRefused, match=what):
        r3.run(inputs, CFG, out, "score", prov=PROV)
    with pytest.raises(r3.PartRefused, match=what):
        r3.run(inputs, CFG, out, "fits", prov=PROV)  # refused, never silently refitted over
    assert any(what in p for p in r3.verify_parts(out, CFG))
    assert r3.check(out, report2_run, CFG, PROV) == 1


def test_a_timing_file_that_fails_its_digest_is_refused_not_read_as_absent(full, inputs, capsys):
    """audit 4f, minor 1: a tampered timing.json must not turn H13 into "not run"."""
    out, _ = full
    for name in (r3.TIMING_FILE, r3.MATRIX_TIMING_FILE):
        path = out / "parts" / name
        data = path.read_bytes()
        try:
            path.write_bytes(data.replace(b'"repeats": 4', b'"repeats": 5'))
            with pytest.raises(r3.PartRefused, match="fails its digest"):
                r3.Parts(out, None).get_json(name)
            with pytest.raises(r3.PartRefused, match="fails its digest"):
                r3.run(inputs, CFG, out, "score", prov=PROV)
        finally:
            path.write_bytes(data)
    digest_file = out / "parts" / (r3.TIMING_FILE + ".sha256")
    digest_bytes = digest_file.read_bytes()
    try:
        digest_file.unlink()
        with pytest.raises(r3.PartRefused, match="without its digest"):
            r3.Parts(out, None).get_json(r3.TIMING_FILE)
    finally:
        digest_file.write_bytes(digest_bytes)
    assert r3.Parts(out, None).get_json("no-such-file.json") is None
    capsys.readouterr()


def test_score_refuses_two_timings_of_different_samples(full, inputs):
    """audit 4f, minor 4: the GPU and the CPU columns must time the same rows."""
    out, _ = full
    parts = r3.Parts(out, None)
    path = out / "parts" / r3.MATRIX_TIMING_FILE
    data, digest_bytes = path.read_bytes(), (out / "parts" / (r3.MATRIX_TIMING_FILE + ".sha256")).read_bytes()
    try:
        parts.put_json(r3.MATRIX_TIMING_FILE, {**json.loads(data), "sample_sha256": "f" * 64})
        with pytest.raises(r3.PairingRefused, match="different samples"):
            r3.run(inputs, CFG, out, "score", prov=PROV)
    finally:
        path.write_bytes(data)
        (out / "parts" / (r3.MATRIX_TIMING_FILE + ".sha256")).write_bytes(digest_bytes)


# --- The ensemble ---------------------------------------------------------------------------------


def test_the_ensemble_is_the_mean_before_the_floor(full, report2_run):
    raw = {1: np.array([[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]]), 2: np.array([[0.4, 0.3, 0.3], [0.3, 0.7, 0.0]])}  # seed 1 under the floor, the mean not
    fc = r3.Forecasts(raw, np.zeros(2, dtype=bool))
    want = prob_map((raw[1] + raw[2]) / 2.0, 1e-3)
    after = (prob_map(raw[1], 1e-3) + prob_map(raw[2], 1e-3)) / 2.0
    np.testing.assert_allclose(r3.arm_probabilities(fc, 1e-3), want, rtol=0, atol=1e-15)
    assert np.abs(r3.arm_probabilities(fc, 1e-3) - after).max() > 1e-5
    out, _ = full
    parts = r3.Parts(out, None)
    fcs = r3.load_forecasts(parts, CFG, report2_run)
    saved = r2.sha256_file(out / r3.SERIES_FILE)
    assert saved == json.loads((out / "report3.json").read_text())["series"]["sha256"]
    for arm in CFG.arms:
        mean = (fcs[arm].raw[1] + fcs[arm].raw[2]) / 2.0
        np.testing.assert_allclose(r3.arm_probabilities(fcs[arm], 1e-3), prob_map(mean, 1e-3), rtol=0, atol=1e-15)


# --- The score stage on synthetic hourly series ----------------------------------------------------------


T = 3000
RULE = r2.Report2Config(n_boot=300)


def series_set(rng: np.random.Generator, *, ols_gain: float, ols_vs_soft: float, skill: bool, select_gap: float) -> tuple[dict, dict]:
    """Hourly log-loss and Brier of climatology, matrix-select and the three arms, with chosen differences."""
    base_ll = 1.06 + 0.05 * rng.standard_normal(T)
    base_br = 0.62 + 0.03 * rng.standard_normal(T)
    e = lambda s: s * rng.standard_normal(T)  # noqa: E731
    ll = {r3.CLIMATOLOGY: base_ll, r3.MATRIX_SELECT: base_ll - 0.004 + e(0.004)}
    ll["enc-ols"] = ll[r3.MATRIX_SELECT] + select_gap + e(0.003)
    ll["enc-softmax"] = ll["enc-ols"] - ols_vs_soft + e(0.003)
    ll["enc-linear"] = ll["enc-softmax"] + 0.001 + e(0.003)
    gain = 0.02 if skill else -0.001
    br = {r3.CLIMATOLOGY: base_br, r3.MATRIX_SELECT: base_br - 0.005 + e(0.003)}
    for arm in ("enc-ols", "enc-softmax", "enc-linear"):
        br[arm] = base_br - gain - ols_gain + e(0.004)
    return ll, br


def _timing(ols: float, soft: float) -> dict:
    """Medians ``ols`` and ``soft``; each repeat's seconds around them, so that the paired ratios differ from the ratio of medians."""
    return {"arms": {"enc-ols": {"whole": {"median": ols}, "seconds": [0.5 * ols, ols, 1.5 * ols]},
                     "enc-softmax": {"whole": {"median": soft}, "seconds": [soft, 1.5 * soft, 0.5 * soft]},
                     "enc-linear": {"whole": {"median": 0.05}, "seconds": [0.05] * 3}},
            "device": "cuda", "gpu": "NVIDIA GeForce RTX 3060"}  # fmt: skip


def test_h4s_two_bounds_equal_a_direct_computation():
    rng = np.random.default_rng(1)
    ll, br = series_set(rng, ols_gain=0.0, ols_vs_soft=-0.001, skill=True, select_gap=0.0)
    out = r3.score_hourly(ll, br, RULE, CFG, _timing(0.08, 0.1))
    d = ll["enc-ols"] - ll["enc-softmax"]
    c = d - d.mean()
    h = RULE.dm_h
    lrv = float(c @ c) / T + 2.0 * sum(float(c[k:] @ c[:-k]) / T for k in range(1, h))
    se = math.sqrt(lrv / T) / math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    upper = d.mean() + st.t.ppf(0.95, T - 1) * se
    block = stats.optimal_block_length(d)
    rng_b = np.random.default_rng(r2._seed(RULE, "diff", "enc-ols", "enc-softmax"))
    draws, done = [], 0
    while done < RULE.n_boot:
        n = min(100, RULE.n_boot - done)
        draws.append(d[stats.stationary_bootstrap_indices(T, block, n, seed=rng_b)].mean(axis=1))
        done += n
    q95 = float(np.quantile(np.concatenate(draws), 0.95))
    h4 = out["verdicts"]["H4"]
    assert h4["upper_95_one_sided"] == pytest.approx(upper, rel=1e-10)
    assert h4["bootstrap_q95"] == pytest.approx(q95, rel=1e-12)
    assert h4["margin"] == pytest.approx(0.005 * ll[r3.CLIMATOLOGY].mean(), rel=1e-15)
    assert h4["testable"] and h4["status"] == "holds" and upper < h4["margin"] and q95 < h4["margin"]
    assert out["verdicts"]["H13"]["status"] == "holds" and out["verdicts"]["H13"]["median_ratio"] == pytest.approx(0.8)
    # the paired ratios 0.04/0.1, 0.08/0.15, 0.12/0.05: their median 0.533, not the ratio of medians 0.8
    assert out["verdicts"]["H13"]["paired_median_ratio"] == pytest.approx(0.08 / 0.15)


def test_the_testability_gate_both_ways():
    rng = np.random.default_rng(2)
    ll, br = series_set(rng, ols_gain=0.0, ols_vs_soft=-0.001, skill=False, select_gap=0.0)
    out = r3.score_hourly(ll, br, RULE, CFG, _timing(0.08, 0.1))
    v = out["verdicts"]
    assert not v["H4"]["testable"] and v["H4"]["status"] == "not testable" and not v["H4"]["holds"]
    assert v["H4"]["bounds_below_margin"]  # the bounds alone would pass: the gate is what stops a vacuous H4
    assert v["H13"]["status"] == "not testable" and not v["H13"]["holds"]
    assert all(out["arms"][a]["bss"]["low"] <= 0 for a in ("enc-ols", "enc-softmax"))
    rng = np.random.default_rng(2)
    ll, br = series_set(rng, ols_gain=0.0, ols_vs_soft=-0.001, skill=False, select_gap=0.0)
    br["enc-softmax"] = br[r3.CLIMATOLOGY] - 0.02 + 0.004 * np.random.default_rng(3).standard_normal(T)  # only softmax learned
    v = r3.score_hourly(ll, br, RULE, CFG, _timing(0.08, 0.1))["verdicts"]
    assert v["H4"]["testable"] and v["H4"]["skill_above_0"] == {"enc-ols": False, "enc-softmax": True} and v["H4"]["status"] == "holds"


def test_h5s_rule():
    rng = np.random.default_rng(4)
    ll, br = series_set(rng, ols_gain=0.0, ols_vs_soft=0.0, skill=True, select_gap=-0.01)
    v = r3.score_hourly(ll, br, RULE, CFG, None)
    c = v["comparisons"]["H5"]
    assert c["a"] == "enc-ols" and c["b"] == r3.MATRIX_SELECT and c["dm_p"] < 0.05 and c["bootstrap"]["high"] < 0
    assert v["verdicts"]["H5"]["status"] == "holds" and "cadence" in v["verdicts"]["H5"]["limit"]
    ll2, br2 = series_set(np.random.default_rng(4), ols_gain=0.0, ols_vs_soft=0.0, skill=True, select_gap=0.0)
    assert r3.score_hourly(ll2, br2, RULE, CFG, None)["verdicts"]["H5"]["status"] == "does not hold"
    ll3, br3 = series_set(np.random.default_rng(4), ols_gain=0.0, ols_vs_soft=0.0, skill=True, select_gap=+0.01)  # worse: never passes
    assert r3.score_hourly(ll3, br3, RULE, CFG, None)["verdicts"]["H5"]["status"] == "does not hold"


def test_h13s_rule_including_h4_does_not_hold():
    rng = np.random.default_rng(5)
    ll, br = series_set(rng, ols_gain=0.0, ols_vs_soft=-0.001, skill=True, select_gap=0.0)
    for ols, soft, status in ((0.09, 0.1, "holds"), (0.0905, 0.1, "does not hold"), (0.12, 0.1, "does not hold")):
        v = r3.score_hourly(ll, br, RULE, CFG, _timing(ols, soft))["verdicts"]
        assert v["H4"]["status"] == "holds" and v["H13"]["status"] == status, (ols, soft)
    assert r3.score_hourly(ll, br, RULE, CFG, None)["verdicts"]["H13"]["status"] == "not run"
    # audit 4f, minor A: a timing on the CPU, or on CUDA without its GPU named, is not H13's registered timing, whatever its ratio
    for where in ({"device": "cpu", "gpu": None}, {"device": "cuda", "gpu": None}):
        h13 = r3.score_hourly(ll, br, RULE, CFG, {**_timing(0.05, 0.1), **where})["verdicts"]["H13"]
        assert h13["status"] == "not run under the registered conditions" and not h13["holds"] and h13["registered_conditions"] is False
    assert r3.score_hourly(ll, br, RULE, CFG, _timing(0.05, 0.1))["verdicts"]["H13"]["registered_conditions"] is True
    ll_bad, br_bad = series_set(np.random.default_rng(5), ols_gain=0.0, ols_vs_soft=0.02, skill=True, select_gap=0.0)  # ols much worse
    v = r3.score_hourly(ll_bad, br_bad, RULE, CFG, _timing(0.01, 0.1))["verdicts"]
    assert v["H4"]["testable"] and v["H4"]["status"] == "does not hold"
    assert v["H13"]["status"] == "does not hold" and v["H13"]["median_ratio"] == pytest.approx(0.1)


# --- Pairing --------------------------------------------------------------------------------------------


def test_the_pairing_refuses_a_missing_hour_and_a_shifted_row(full, report2_run):
    out, _ = full
    parts = r3.Parts(out, None)
    fc = r3.read_forecast(parts, r3.part_key("enc-ols", 1, 2019))
    ok = r3.pair_forecasts([fc], report2_run, "ok")
    assert ok["raw"].shape[0] == report2_run.series["label"].size
    hour = fc["open_time"][100]
    lost = {k: (v[fc["open_time"] != hour] if isinstance(v, np.ndarray) else v) for k, v in fc.items()}
    with pytest.raises(r3.PairingRefused, match="have no forecast"):
        r3.pair_forecasts([lost], report2_run, "a missing hour")
    shifted = {**fc, "decision_time": fc["decision_time"].copy()}
    shifted["decision_time"][7] += 3_600_000
    with pytest.raises(r3.PairingRefused):
        r3.pair_forecasts([shifted], report2_run, "a shifted row")
    relabelled = {**fc, "label": fc["label"].copy()}
    relabelled["label"][7] = (relabelled["label"][7] + 1) % 3
    with pytest.raises(r3.PairingRefused, match="label"):
        r3.pair_forecasts([relabelled], report2_run, "another label")
    doubled = {k: (np.concatenate([v, v[:1]]) if isinstance(v, np.ndarray) else v) for k, v in fc.items()}
    with pytest.raises(r3.PairingRefused):
        r3.pair_forecasts([doubled], report2_run, "a row twice")


# --- The report and --check -----------------------------------------------------------------------------


def test_the_report_holds_every_registered_number(full):
    out, m = full
    assert set(m["verdicts"]) == {"H4", "H5", "H13"} and m["registered"] is False
    assert m["variants"] == {"before": 18, "this_report": 3, "total": 21}
    for arm in CFG.arms:
        a = m["arms"][arm]
        for key in ("log_loss", "brier", "bss", "ece"):
            assert a[key]["low"] <= a[key]["estimate"] <= a[key]["high"], (arm, key)
        assert set(a["seeds"]["log_loss"]) == {"1", "2"} and a["seeds"]["range"] >= 0 and a["fallbacks"] >= 0
    assert set(m["comparisons"]) == {"H4", "H5", "enc-linear vs enc-ols", "enc-linear vs enc-softmax"}
    assert m["h4_margin"] == pytest.approx(0.005 * m["climatology_log_loss"])
    md = (out / r3.REPORT_FILE).read_text(encoding="utf-8")
    assert "NOT THE REGISTERED RUN" in md and "## The cost table" in md and "cadence" in md
    assert "the whole process, panel and windows included" in md and "early-stopping statistic, not results" in md
    assert "memory readout alone" in md and "Descriptions beside the verdict, not verdicts" in md and "Fitted on: cpu" in md
    for f in r3.COMPARED + ("run.json",):
        assert (out / f).is_file(), f
    run = json.loads((out / "run.json").read_text())
    assert run["timing"]["device"] == "cpu" and run["fits_builds"] and "finished_at" in run
    assert "finished_at" not in (out / "report3.json").read_text()
    assert m["fits_devices"] == ["cpu"] and len(m["fits_builds"]) == 1 and m["fits_builds"][0]["device"] == "cpu"
    assert m["fits_builds"] == run["fits_builds"] and "in 1 build(s)" in md


def test_the_registered_run_needs_gpu_fits_in_one_build_and_a_gpu_timing(full, report2_run):
    """audit 4f, minors A and B: the registered flag reads the fits' devices, their builds and the timing's device."""
    reg = dataclasses.replace(report2_run, report={**report2_run.report, "registered": True})
    gpu = {"gpu": "NVIDIA GeForce RTX 3060", "driver": "617.42", "cuda": "13.0", "torch": "2.11.0", "device": "cuda"}
    timing = {"device": "cuda", "gpu": "NVIDIA GeForce RTX 3060"}
    cfg = r3.Report3Config()
    assert r3.registered_run(cfg, reg, ["cuda"], [gpu], timing) and r3.registered_run(cfg, reg, ["cuda"], [gpu], None)
    assert not r3.registered_run(cfg, reg, ["cuda"], [gpu], {"device": "cpu", "gpu": None})  # a CPU timing against GPU fits
    assert not r3.registered_run(cfg, reg, ["cuda"], [gpu, {**gpu, "driver": "618.00"}], timing)  # two builds
    assert not r3.registered_run(cfg, reg, ["cpu"], [gpu], timing) and not r3.registered_run(cfg, report2_run, ["cuda"], [gpu], timing)
    assert not r3.registered_run(CFG, reg, ["cuda"], [gpu], timing)


def test_more_than_one_fit_build_is_surfaced_in_the_compared_outputs(inputs, full, tmp_path):
    """audit 4f, minor B: a fit on another driver shows in report3.json and report-3.md, which --check compares."""
    straight, _ = full
    out = tmp_path / "two-builds"
    _copy_tree(straight, out)
    parts = r3.Parts(out, None)
    key = r3.part_key("enc-linear", 2, 2019) + "/timing.json"
    parts.put_json(key, {**parts.get_json(key), "driver": "another driver"})
    m = r3.run(inputs, CFG, out, "score", prov=PROV)
    assert len(m["fits_builds"]) == 2 and {b["driver"] for b in m["fits_builds"]} == {None, "another driver"} and not m["registered"]
    assert "in 2 build(s)" in (out / r3.REPORT_FILE).read_text(encoding="utf-8")
    assert json.loads((out / "report3.json").read_text())["fits_builds"] == m["fits_builds"]


def test_the_check_passes_and_sees_one_changed_digit(full, report2_run, capsys):
    out, _ = full
    assert r3.check(out, report2_run, CFG, PROV) == 0
    path = out / "report3.json"
    text = path.read_text(encoding="utf-8")
    at = text.index('"climatology_log_loss": ') + len('"climatology_log_loss": ') + 4
    digit = text[at]
    assert digit.isdigit()
    try:
        path.write_text(text[:at] + str((int(digit) + 1) % 10) + text[at + 1 :], encoding="utf-8", newline="\n")
        assert r3.check(out, report2_run, CFG, PROV) == 1
    finally:
        path.write_text(text, encoding="utf-8", newline="\n")
    part = out / "parts" / r3.part_key("enc-softmax", 2, 2019) / "forecast.safetensors"
    data = part.read_bytes()
    try:
        part.write_bytes(data[:-3] + bytes([data[-3] ^ 1]) + data[-2:])
        assert r3.check(out, report2_run, CFG, PROV) == 1
        assert "parts/fits/enc-softmax/s2/2019/forecast.safetensors: fails its digest" in capsys.readouterr().out
    finally:
        part.write_bytes(data)
    assert r3.check(out, report2_run, CFG, PROV) == 0
    assert r3.check(out / "nowhere", report2_run, CFG, PROV) == 2
    capsys.readouterr()


# --- Timing ------------------------------------------------------------------------------------------------


def test_the_timing_protocol_warms_up_alternates_and_scales():
    ticks = iter(range(10_000))
    clock = lambda: float(next(ticks))  # noqa: E731
    calls: list[tuple[str, int]] = []
    runners = {a: (lambda b, a=a: calls.append((a, b))) for a in ("A", "B")}
    synced: list[int] = []
    events: list[tuple[int, str, str, int]] = []
    secs = r3.run_protocol(runners, n_batches=3, warmups=3, repeats=4, sync=lambda: synced.append(1), clock=clock, events=events)
    per_arm = [("warmup", 0), ("warmup", 1), ("warmup", 2), ("timed", 0), ("timed", 1), ("timed", 2)]
    assert events == [(r, a, kind, b) for r in range(4) for a in ("A", "B") for kind, b in per_arm]
    assert [a for a, _ in calls][::6] == ["A", "B"] * 4  # ABAB
    assert secs == {"A": [3.0] * 4, "B": [3.0] * 4}  # each timed batch reads the clock twice, one tick apart
    assert len(synced) == 4 * 2 * (1 + 2 * 3)  # before the timed batches, and before and after each
    assert r3.per_decisions([3.0, 6.0], 12_288, 10_000) == [3.0 / 12_288 * 10_000, 6.0 / 12_288 * 10_000]
    assert r3.summary([3.0, 1.0, 2.0]) == {"min": 1.0, "median": 2.0, "max": 3.0}


def test_the_timing_stage_on_cpu(full, report2_run):
    out, m = full
    t = r3.Parts(out, None).get_json(r3.TIMING_FILE)
    assert t["device"] == "cpu" and t["rows"] == 48 and t["repeats"] == 4 and t["order"] == list(CFG.arms)
    assert t["first_repeat_events"] == [[0, a, kind, b] for a in CFG.arms for kind, b in
                                        [("warmup", 0), ("warmup", 1), ("warmup", 2), ("timed", 0), ("timed", 1), ("timed", 2)]]  # fmt: skip
    for arm in CFG.arms:
        a = t["arms"][arm]
        assert len(a["seconds"]) == 4 and all(s > 0 for s in a["seconds"])
        assert a["per_decisions_s"] == [s / 48 * 10_000 for s in a["seconds"]]
        assert a["whole"]["median"] == float(np.median(a["per_decisions_s"])) and a["operator"]["median"] > 0
        assert len(a["readout_seconds"]) == 4 and a["readout"]["median"] == float(np.median(a["readout_per_decisions_s"])) > 0
    assert t["sample_sha256"] == r3.Parts(out, None).get_json(r3.MATRIX_TIMING_FILE)["sample_sha256"]  # one sample for both machines
    usable = np.ones(report2_run.series["label"].size, dtype=bool)
    sample = r3.timing_sample(usable, CFG)
    assert sample.size == 48 and np.all(np.diff(sample) > 0)
    np.testing.assert_array_equal(sample, r3.timing_sample(usable, CFG))
    assert m["verdicts"]["H13"]["median_ratio"] == pytest.approx(t["arms"]["enc-ols"]["whole"]["median"] / t["arms"]["enc-softmax"]["whole"]["median"])


def test_the_matrix_stage_may_run_first_and_then_binds_the_folder(inputs, report2_run, tmp_path):
    """The Matrix CPU stage runs on another machine: on a fresh folder it writes the identity the fits must then match."""
    out = tmp_path / "matrix-first"
    r3.run(inputs, CFG, out, "matrix-timing", prov=PROV)
    parts = r3.Parts(out, None)
    assert parts.identity() == json.loads(r3._json_text(r3.identity_of(CFG, report2_run, PROV)))
    mt = parts.get_json(r3.MATRIX_TIMING_FILE)
    assert set(mt["arms"]) == set(r3.MATRIX_ARMS) and mt["repeats"] == 4 and mt["rows"] == 48
    assert mt["sample_sha256"] == r3.Parts(out, None).get_json(r3.MATRIX_TIMING_FILE)["sample_sha256"]
    other = dataclasses.replace(CFG, timing_repeats=5)
    with pytest.raises(r3.ResumeRefused):
        r3.run(inputs, other, out, "matrix-timing", prov=PROV)
    with pytest.raises(r3.ResumeRefused):
        r3.run(inputs, other, out, "fits", prov=PROV)


def test_the_cost_table(full):
    _, m = full
    ct = m["cost_table"]
    assert set(ct["arms"]) == set(CFG.arms) | set(r3.MATRIX_ARMS) and ct["cpu"] == "test CPU"
    for arm in CFG.arms:
        r = ct["arms"][arm]
        assert r["parameters"] == sum(p.numel() for p in EncoderModel(EncoderConfig(arm=arm, **dict(TINY))).parameters())
        assert r["gpu_per_decisions_s"] > 0 and r["training_hours"] > 0 and r["fits_timed"] == 2 and "cpu_per_decisions_s" not in r
    for arm in r3.MATRIX_ARMS:
        r = ct["arms"][arm]
        assert r["parameters"] == 0 and r["cpu_per_decisions_s"] > 0 and "gpu_per_decisions_s" not in r
        assert (r["flops"]["total"] == 0) == (arm == r3.CLIMATOLOGY)
        assert (r["memory_bytes"] is None) == (arm == r3.CLIMATOLOGY)
    assert ct["arms"]["enc-ols"]["parameters"] == ct["arms"]["enc-softmax"]["parameters"] + 2  # one lambda per head, 1 layer x 2 heads
    mt = m["matrix_timing"]
    assert ct["arms"][r3.MATRIX_SELECT]["memory_bytes"] == r3.matrix_memory_bytes(mt["memory_entries"], block_log2=mt["block_log2"])["total"]


def test_the_matrix_memory_footprint_matches_the_crates_own_sizing():
    """crates/ols-core/src/matrix.rs: at 2·10^6 entries, p = 15, q = 4, blocks of 2^6: about 6.2·10^4 nodes per tree, 160 MB each."""
    m = r3.matrix_memory_bytes(2_000_000, 15, 4, 6)
    assert 62_000 <= m["nodes_per_tree"] <= 62_600 and m["trees"] / 2 == pytest.approx(160e6, rel=0.01)
    assert m["entries"] == 2_000_000 * 8 * 20 and m["total"] == m["entries"] + m["trees"]
    assert r3.matrix_memory_bytes(64, 15, 4, 6)["nodes_per_tree"] == 1 and r3.matrix_memory_bytes(65, 15, 4, 6)["nodes_per_tree"] == 3


def test_h4s_descriptions_beside_the_verdict(full, report2_run):
    """The auditor's check: H4's bounds on each seed-matched pair, and reversed on the ensembles; descriptions, the same statistics."""
    out, m = full
    d = m["h4_descriptions"]
    assert set(d["seed_pairs"]) == {"1", "2"} and d["reversed"]["a"] == "enc-softmax" and d["reversed"]["b"] == "enc-ols"
    assert d["reversed"]["mean_diff"] == pytest.approx(-m["comparisons"]["H4"]["mean_diff"], abs=1e-15)
    s = report2_run.series
    sc = r2.Scored(np.arange(s["label"].size), s["label"], s["hour"], s["hours"], s["count"])
    fcs = r3.load_forecasts(r3.Parts(out, None), CFG, report2_run)
    rcfg = r3.rule_config(report2_run.config)
    ll = {f"{a} seed 1": r2.hourly(stats.per_row_losses(prob_map(fcs[a].raw[1], 1e-3), sc.label).log_loss, sc) for a in ("enc-ols", "enc-softmax")}
    want = r2.compare_arms(ll, "enc-ols seed 1", "enc-softmax seed 1", m["climatology_log_loss"], rcfg)
    got = d["seed_pairs"]["1"]
    assert got["upper_95_one_sided"] == pytest.approx(want["upper_95_one_sided"], rel=1e-12)
    assert got["bootstrap_q95"] == pytest.approx(want["bootstrap"]["q95"], rel=1e-12)
    assert got["bounds_below_margin"] == (want["upper_95_one_sided"] < m["h4_margin"] and want["bootstrap"]["q95"] < m["h4_margin"])
    assert m["verdicts"]["H4"]["upper_95_one_sided"] == m["comparisons"]["H4"]["upper_95_one_sided"]  # the verdict is still the ensembles'


# --- FLOPs ------------------------------------------------------------------------------------------------


def _products(cfg: EncoderConfig) -> int:
    """The matrix products of one window's encoding, counted independently: 2 m n k each."""
    t, d, h, e, m = cfg.window, cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp
    attn = {"softmax": 2 * 2 * t * t * e, "linear": 2 * 2 * t * e * e + 2 * t * e}[cfg.attention]  # linear: K^T V, Q (K^T V), Q sum(K)
    layer = 2 * t * d * 3 * d + 2 * t * d * d + 2 * 2 * t * d * m + h * attn
    return 2 * t * cfg.n_quantities * d + cfg.n_layers * layer + 2 * d * cfg.key_dim


@pytest.mark.parametrize("arm", ["enc-softmax", "enc-linear"])
def test_the_products_of_the_flops_formula_match_torchs_counter(arm):
    """torch's FlopCounterMode counts matrix products only; the formula's products must equal them."""
    from torch.utils.flop_counter import FlopCounterMode

    cfg = EncoderConfig(arm=arm)
    model = EncoderModel(cfg).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model.encode(torch.zeros(1, cfg.window, cfg.n_quantities))
    assert counter.get_total_flops() == _products(cfg)
    f = r3.encoder_flops(cfg)
    assert f["total"] == sum(v for k, v in f.items() if k != "total") and f["total"] > _products(cfg)


def test_the_ols_head_has_about_a_quarter_of_the_softmax_heads_flops():
    """docs/HYPOTHESES.md: analytic FLOPs favour OLS by about four at T = 64, d_head = 8."""
    ratio = r3.attention_flops("softmax", 64, 8) / r3.attention_flops("ols", 64, 8)
    assert 3.5 < ratio < 5.0
    e = 8
    assert r3.attention_flops("ols", 64, e) == 2 * 64 * 81 + 4 * 64 * 9 * 8 + 243 + 2 * 81 * 8 + 6 * 64 * 8 + 2 * 64 + 8
    assert r3.encoder_flops(EncoderConfig(arm="enc-ols"))["total"] < r3.encoder_flops(EncoderConfig(arm="enc-softmax"))["total"]
