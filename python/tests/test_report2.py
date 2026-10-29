"""Report n°2 (``olsprime.bench.report2``) on a synthetic store.

- two runs write byte-identical compared files, and the check sees a changed byte;
- every arm is scored, the refusals and the feature ratios are reported, the run is marked as not
  the registered one;
- the verdict rules follow the pre-registration on hand-built numbers;
- the HLN standard error, the ECE by hours and the bootstrap intervals agree with ``eval.stats``.

Every test runs on a fake store under a temporary data root: nothing reads the repository's ``data/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from olsprime.bench import report2 as r2
from olsprime.eval import stats
from olsprime.models.matrix import ARMS, MatrixConfig
from tests.test_panel import fake_bars, universe_frame
from tests.test_protocol import store

START = datetime(2018, 9, 1, tzinfo=UTC)
MONTHS = [(2018, 10), (2018, 11), (2018, 12), (2019, 1), (2019, 2)]
HOURS = {"AAAUSDT": 24 * 200, "BBBUSDT": 24 * 200, "CCCUSDT": 24 * 160}
CFG = r2.Report2Config(
    matrix=MatrixConfig(first_test=(2019, 1), last_test=(2019, 2)), first_month=(2018, 10), last_month=(2019, 2), n_boot=200, readout_sample=50
)
FAKE_MANIFEST = {
    "digest": "0" * 64,
    "vault_open": False,
    "totals": {"files": len(HOURS)},
    "files": [
        {"path": f"parquet/spot/klines/1h/{s}.parquet", "kind": "klines", "symbol": s, "interval": "1h",
         "sha256": "f" * 64, "stored_rows": n, "sealed_rows": n,
         "first_sealed": "2018-09-01T00:00:00.000+00:00", "last_sealed": "2019-03-19T23:00:00.000+00:00"}
        for s, n in HOURS.items()
    ],
}  # fmt: skip
PROVENANCE = {"git_commit": "c" * 40, "code": {"sha256": "d" * 64, "files": 1}, "manifest_digest": "0" * 64}


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> tuple[Path, object]:
    root = tmp_path_factory.mktemp("store")
    for seed, (s, n) in enumerate(HOURS.items()):
        store(root, fake_bars(s, n, seed=seed, start=START), s, "1h")
    return root, universe_frame({s: MONTHS for s in HOURS})


def _run(out: Path, root: Path, universe) -> dict:
    return r2.run_report(CFG, out, data_dir=root, universe=universe, manifest=FAKE_MANIFEST, provenance=PROVENANCE)


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, fake_root):
    root, universe = fake_root
    a, b = tmp_path_factory.mktemp("run-a"), tmp_path_factory.mktemp("run-b")
    return a, b, _run(a, root, universe), _run(b, root, universe)


def test_two_runs_are_byte_identical(two_runs):
    a, b, _, _ = two_runs
    assert r2.compared_files(a) == list(r2.COMPARED)
    for name in r2.compared_files(a):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert r2.diff_runs(a, b) == []
    assert "run.json" not in r2.compared_files(a)
    assert "finished_at" in json.loads((a / "run.json").read_text(encoding="utf-8"))


def test_the_check_reruns_and_sees_a_changed_byte(two_runs, fake_root, tmp_path, capsys):
    a, _, _, _ = two_runs
    root, universe = fake_root
    kwargs = {"data_dir": root, "universe": universe, "manifest": FAKE_MANIFEST, "provenance": PROVENANCE}
    assert r2.check(a, CFG, **kwargs) == 0
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in r2.compared_files(a):
        (copy / name).write_bytes((a / name).read_bytes())
    text = (copy / "report2.json").read_text(encoding="utf-8")
    (copy / "report2.json").write_text(text.replace('"scored"', '"scorex"', 1), encoding="utf-8")
    (copy / r2.REPORT_FILE).unlink()
    problems = r2.diff_runs(a, copy)
    assert any(p.startswith("report2.json: differs") and "scorex" in p for p in problems)
    assert any("only in the expected run" in p for p in problems)
    assert r2.check(copy, CFG, **kwargs) == 1
    assert r2.check(tmp_path / "nowhere", CFG, **kwargs) == 2
    capsys.readouterr()


def test_every_arm_is_scored_and_the_report_says_it_is_not_the_registered_run(two_runs):
    a, _, metrics, _ = two_runs
    assert not CFG.registered and r2.Report2Config().registered
    assert metrics["registered"] is False
    assert set(metrics["arms"]) == set(ARMS)
    for arm in ARMS:
        m = metrics["arms"][arm]
        for key in ("log_loss", "brier"):
            assert m[key]["low"] < m[key]["estimate"] < m[key]["high"], (arm, key)
        assert 0 <= m["ece"]["estimate"] < 1 and m["ece"]["n_boot"] == CFG.n_boot
        assert (m["bss"] is None) == (arm == r2.CLIMATOLOGY)
        assert set(metrics["status_counts"][arm]) >= {"ok", "empty", "singular"}
    assert metrics["scored"]["first_hour"].startswith("2019-01-01")
    assert metrics["scored"]["last_hour"] < "2019-03"
    assert {c["hypothesis"] for c in metrics["comparisons"]} == {"H3", "H10", "H11", "H12"}
    assert set(metrics["verdicts"]) == {"H2", "H3", "H10", "H11", "H12"}
    assert metrics["variants"] == {"before": 10, "this_report": 8, "total": 18}
    report = (a / r2.REPORT_FILE).read_text(encoding="utf-8")
    assert "NOT THE REGISTERED RUN" in report
    for arm in ARMS:
        assert arm in report


def test_the_feature_ratios_are_reported_with_their_alert(two_runs):
    _, _, metrics, _ = two_runs
    fr = metrics["feature_ratio"]
    assert fr["alert_above"] == r2.RATIO_ALERT and fr["alert"] is False
    assert len(fr["panel"]) == 15 and set(fr["max_over_folds"]) == set(fr["panel"])
    assert all(v >= 0 for v in fr["panel"].values())
    assert fr["largest"] == max(*fr["panel"].values(), *fr["max_over_folds"].values())


def test_refused_cells_are_counted_apart_from_empty_ones(two_runs):
    _, _, metrics, _ = two_runs
    for arm in ARMS:
        counts = metrics["status_counts"][arm]
        assert counts["empty"] >= 0 and counts["singular"] >= 0
        assert metrics["fallbacks"][arm] <= sum(v for k, v in counts.items() if k != "ok")
        by = metrics["fallbacks_by_status"][arm]
        assert set(by) == {"empty", "singular", "non-finite", "too-few"}
        assert sum(by.values()) == metrics["fallbacks"][arm]


def test_the_cusum_counts_its_fits_with_s_set_to_1(two_runs):
    a, _, metrics, _ = two_runs
    c = metrics["cusum"]
    assert c["fits_with_s_set_to_1"] == sum(1 for f in c["fits"] if f["s_default"])
    assert all(f["s"] == 1.0 for f in c["fits"] if f["s_default"])
    assert "fits with s set to 1" in (a / r2.REPORT_FILE).read_text(encoding="utf-8")


# --- the rules on hand-built numbers --------------------------------------------------------


def _comparison(a: str, b: str, *, passes: bool = False, upper: float = 0.0, q95: float = 0.0) -> dict:
    return {"a": a, "b": b, "passes": passes, "upper_95_one_sided": upper, "bootstrap": {"q95": q95}}


def _stats(bss: tuple[float, float], h3=(True, True), h10=False, h11=False, h12=(0.0, 0.0)) -> tuple[dict, list]:
    metrics = {"matrix-select": {"bss": {"estimate": bss[0], "low": bss[1]}}, r2.CLIMATOLOGY: {"log_loss": {"estimate": 1.0}}}
    comps = [
        _comparison("matrix-1y", "softmax-1y", passes=h3[0]),
        _comparison("matrix-1y", "knn-1y", passes=h3[1]),
        _comparison("matrix-adaptive", "matrix-select", passes=h10),
        _comparison("matrix-stack", "matrix-select", passes=h11),
        _comparison("matrix-cusum", "matrix-select", upper=h12[0], q95=h12[1]),
    ]
    return metrics, comps


def test_the_verdicts_follow_the_registered_rules():
    v = r2.verdicts(*_stats((0.01, 0.001)), 3, CFG)
    assert v["H2"]["holds"] and v["H3"]["holds"] and not v["H10"]["holds"] and not v["H11"]["holds"]
    assert not r2.verdicts(*_stats((0.01, -0.001)), 3, CFG)["H2"]["holds"]
    assert not r2.verdicts(*_stats((0.01, 0.001), h3=(True, False)), 3, CFG)["H3"]["holds"]
    assert r2.verdicts(*_stats((0.0, 0.0), h10=True, h11=True), 3, CFG)["H10"]["holds"]
    # H12: both bounds strictly below 0.005 x climatology's log-loss (here 1.0), and at most 36 fits
    assert r2.verdicts(*_stats((0, 0), h12=(0.0049, 0.0049)), 36, CFG)["H12"]["holds"]
    assert not r2.verdicts(*_stats((0, 0), h12=(0.0049, 0.0049)), 37, CFG)["H12"]["holds"]
    assert not r2.verdicts(*_stats((0, 0), h12=(0.005, 0.0049)), 3, CFG)["H12"]["holds"]
    assert not r2.verdicts(*_stats((0, 0), h12=(0.0049, 0.0051)), 3, CFG)["H12"]["holds"]


def test_a_comparison_passes_only_with_all_three_conditions():
    assert r2.passes(0.01, -0.3, -0.1, -0.2)
    assert not r2.passes(0.05, -0.3, -0.1, -0.2)
    assert not r2.passes(0.01, -0.3, 0.0, -0.2)
    assert not r2.passes(0.01, -0.3, -0.1, 0.0)


def test_the_hln_standard_error_does_not_move_with_the_mean():
    rng = np.random.default_rng(3)
    d = rng.standard_normal(500) * 0.1 + 0.02
    dm = stats.diebold_mariano(d, np.zeros_like(d), h=24)
    se = r2.dm_standard_error(dm, d, 24)
    assert se == pytest.approx(abs(dm.mean_diff / dm.statistic))
    shifted = d + 0.5
    assert r2.dm_standard_error(stats.diebold_mariano(shifted, np.zeros_like(d), h=24), shifted, 24) == pytest.approx(se, rel=1e-9)
    flat = np.zeros_like(d)
    flat[::2], flat[1::2] = 0.1, -0.1
    zero = stats.diebold_mariano(flat, np.zeros_like(flat), h=24)
    assert zero.statistic == 0.0 and r2.dm_standard_error(zero, flat, 24) > 0


def _scored(n_hours: int, per_hour: int) -> r2.Scored:
    hour = np.repeat(np.arange(n_hours), per_hour)
    return r2.Scored(np.arange(hour.size), np.zeros(hour.size, np.int64), hour, np.arange(n_hours) * 3_600_000, np.full(n_hours, per_hour))


def test_the_ece_by_hours_is_the_ece_of_the_rows():
    rng = np.random.default_rng(5)
    sc = _scored(40, 7)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=sc.hour.size)
    y = rng.integers(0, 3, size=sc.hour.size)
    d = r2.ece_bins(p, y, sc, 10)
    assert np.abs(d.sum(axis=0)).sum() / sc.count.sum() == pytest.approx(stats.ece(p, y, 10), rel=1e-12)
    out = r2.ece_interval(p, y, sc, rng.random(40), CFG, "ece", "test")
    assert out["n_boot"] == CFG.n_boot and out["low"] <= out["high"]
    assert out["estimate"] == pytest.approx(stats.ece(p, y, 10), rel=1e-12)


def test_the_bootstrap_is_seeded_by_tag_and_brackets_the_mean():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(300)
    a, b = r2.mean_interval(x, CFG, "ll", "x"), r2.mean_interval(x, CFG, "ll", "x")
    assert a == b
    assert a["low"] < a["estimate"] < a["high"] and a["q95"] <= a["high"]  # q95 is one-sided, high is the 97.5 % point
    assert r2.mean_interval(x, CFG, "ll", "y") != a
    clim = 0.6 + 0.01 * rng.random(300)
    arm = clim - 0.01
    bss = r2.bss_interval(arm, clim, CFG, "bss", "x")
    assert bss["estimate"] == pytest.approx(1 - arm.mean() / clim.mean())
    assert 0 < bss["low"] <= bss["estimate"] <= bss["high"]
