"""Report n°2 (``olsprime.bench.report2``) on a synthetic store.

- two runs write byte-identical compared files, ``series.npz`` included, and the check sees a changed byte;
- ``series.npz`` reproduces the report's means, carries its rows, and its digest is in ``report2.json``;
- the comparison with an older run allows the named differences and nothing else;
- H12's parts read H12's own rule;
- every arm is scored, the refusals and the feature ratios are reported, the run is marked as not
  the registered one;
- the verdict rules follow the pre-registration on hand-built numbers;
- the HLN standard error, the ECE by hours and the bootstrap intervals agree with ``eval.stats``.

Every test runs on a fake store under a temporary data root: nothing reads the repository's ``data/``.
"""

from __future__ import annotations

import hashlib
import json
import struct
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
    matrix=MatrixConfig(first_test=(2019, 1), last_test=(2019, 2)), first_month=(2018, 10), last_month=(2019, 2), n_boot=200, readout_sample=50,
    previewed_through=(2019, 1),
)  # fmt: skip
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


def test_each_refit_names_the_day_it_left_out():
    day, f0 = 86_400_000, int(datetime(2019, 1, 1, tzinfo=UTC).timestamp() * 1000)
    fits = [{"fit_ms": f, "window": 1, "rho": 2, "s": 1.5, "s_days": 300, "s_default": False, "default": False, "n_selection": 5000}
            for f in (f0, f0 + 40 * day, f0 + 90 * day)]  # fmt: skip
    path = {"day_ms": f0 + np.arange(5) * day, "alarm": np.array([False, True, False, False, True])}
    # the day before the second fit had a complete 00:00 row and is left out; the day before the third had none
    out = r2.cusum_summary({"fits": fits, "path": path, "skipped_days": np.array([f0 + 39 * day])}, MatrixConfig())
    assert [f["day_left_out"] for f in out["fits"]] == [None, "2019-02-09", None]
    assert out["days_left_out"] == ["2019-02-09"] and out["days_left_out_at_refits"] == 1
    assert out["refits_without_a_day_left_out"] == 1 and out["alarms"] == 2


def test_the_previewed_months_are_reported_apart(two_runs):
    a, _, metrics, _ = two_runs
    ps = metrics["preview_split"]
    assert (ps["previewed_through"], ps["previewed_months"], ps["test_months"]) == ("2019-01", 1, 2)
    prev, rest = ps["previewed"], ps["unseen"]
    total = metrics["scored"]["hours"]
    assert prev["hours"] + rest["hours"] == total
    assert prev["last_hour"] < "2019-02" <= rest["first_hour"]
    for part in (prev, rest):
        assert [c["hypothesis"] for c in part["comparisons"]] == [h for h, _, _ in r2.COMPARISONS]
        assert part["bss_matrix_select"]["n_boot"] == CFG.n_boot
        assert part["h12_margin"] == pytest.approx(CFG.h12_margin * part["climatology_log_loss"])
    # the two parts' hourly means add up to the whole period's
    for i, c in enumerate(metrics["comparisons"]):
        parts = prev["comparisons"][i]["mean_diff"] * prev["hours"] + rest["comparisons"][i]["mean_diff"] * rest["hours"]
        assert parts / total == pytest.approx(c["mean_diff"], rel=1e-9, abs=1e-15)
    clim = prev["climatology_log_loss"] * prev["hours"] + rest["climatology_log_loss"] * rest["hours"]
    assert clim / total == pytest.approx(metrics["arms"][r2.CLIMATOLOGY]["log_loss"]["estimate"], rel=1e-12)
    # on this noise matrix-adaptive takes matrix-select's rho, 10, at every January decision: the two arms
    # agree at every previewed hour, so that part has no H10 test, recorded instead of raised. matrix-cusum
    # refitted on 2019-01-20 after an alarm, so H12 has a test in both parts.
    undefined = [(name, c["hypothesis"]) for name, p in (("previewed", prev), ("unseen", rest)) for c in p["comparisons"] if "undefined" in c]
    assert undefined == [("previewed", "H10")]
    h10 = prev["comparisons"][2]
    assert (h10["a"], h10["b"], h10["mean_diff"], h10["hours"]) == ("matrix-adaptive", "matrix-select", 0.0, prev["hours"])
    assert metrics["adaptive"]["rho_counts"]["10"] < metrics["adaptive"]["decisions"]
    assert [f["fit"][:10] for f in metrics["cusum"]["fits"]] == ["2019-01-01", "2019-01-20"]
    # identical hours: matrix-adaptive reads matrix-select's cell at each decision hour where it takes rho = 10,
    # and matrix-cusum is matrix-select over the 19 x 24 hours before its refit
    whole_h10, prev_h12, rest_h12 = metrics["comparisons"][2], prev["comparisons"][-1], rest["comparisons"][-1]
    assert whole_h10["identical_hours"] == metrics["adaptive"]["rho_counts"]["10"] == 1404
    assert (prev_h12["identical_hours"], rest_h12["identical_hours"]) == (19 * 24, 0)
    assert all("passes" not in c and ("rule_met" in c) != ("undefined" in c) for p in (prev, rest) for c in p["comparisons"])
    report = (a / r2.REPORT_FILE).read_text(encoding="utf-8")
    assert "## The previewed months, apart" in report and "no test: the difference is the same at every hour" in report
    # where the verdicts are read, and again where the parts are shown
    verdicts, arms = report.index("## Verdicts"), report.index("## Arms")
    assert verdicts < report.index("smoke runs on the real panel scored 1 of the 2 test months, through 2019-01") < arms
    assert report.count("scored 1 of the 2 test months") == 2
    h10_row = next(line for line in report.splitlines() if line.startswith("| H10 | matrix-adaptive | matrix-select |"))
    assert h10_row.endswith("| 99.2% (12 informative), few informative hours |")
    section = report.split("## The previewed months, apart")[1].split("## Counts")[0]
    assert "rule not met" in section and "passes" not in section and "does not pass" not in section
    assert "identical hours 61.3% (288 informative), few informative hours" in section


def test_the_report_states_its_coverage_and_its_dm_kernels(two_runs):
    a, _, metrics, _ = two_runs
    sc = metrics["scored"]
    assert sc["calendar_hours"] == (31 + 28) * 24
    assert 0 < sc["hours"] <= sc["calendar_hours"] and sc["share_of_calendar_hours"] == sc["hours"] / sc["calendar_hours"]
    report = (a / r2.REPORT_FILE).read_text(encoding="utf-8")
    assert f"{sc['hours']} of the test months' {sc['calendar_hours']} hours are scored" in report
    for c in metrics["comparisons"]:
        assert (c["dm_kernel"], c["dm_lags"], c["dm_note"]) == ("rectangular", 23, "")
    assert "rectangular, 23 lags" in report and "Diebold-Mariano fallbacks: none." in report
    assert "| day left out |" in report


# --- the rules on hand-built numbers --------------------------------------------------------


def test_a_part_where_two_arms_agree_at_every_hour_has_no_test():
    rng = np.random.default_rng(0)
    y, z = rng.integers(1, 1024, (2, 300)) / 1024  # exact in binary: y + 0.25 - y is 0.25 at every hour
    w = y.copy()
    w[::30] = z[::30]  # w agrees with y at 290 of the 300 hours
    part = {"x": y + 0.25, "y": y, "z": z, "w": w}
    assert r2._compare_part(part, "x", "y", 1.0, CFG, "t") == {
        "a": "x", "b": "y", "mean_diff": 0.25, "hours": 300, "identical_hours": 0, "share_of_hours_identical": 0.0,
        "undefined": "the difference is the same at every hour"}  # fmt: skip
    # a part describes: the registered word "passes" becomes rule_met, and nothing else changes
    whole = r2.compare_arms(part, "z", "y", 1.0, CFG, "t")
    rest = {k: v for k, v in whole.items() if k != "passes"}
    assert r2._compare_part(part, "z", "y", 1.0, CFG, "t") == {**rest, "rule_met": whole["passes"]}
    near = r2.compare_arms(part, "w", "y", 1.0, CFG, "t")
    assert (near["identical_hours"], near["share_of_hours_identical"]) == (int((z[::30] == y[::30]).sum()) + 290, pytest.approx(near["identical_hours"] / 300))
    assert r2._informative(near).endswith(f" ({300 - near['identical_hours']} informative), few informative hours")
    assert r2._informative(whole) == f"{whole['share_of_hours_identical']:.1%} ({300 - whole['identical_hours']} informative)"


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


def test_the_ece_check_lets_rounding_through_but_not_a_lost_row():
    """Climatology repeats one calibrated forecast over many rows. At the panel's size, the ECE by hours and
    stats.ece then differ through rounding alone by more than 1e-9 of the small ECE: the official run of
    2026-10-29 stopped on that check."""
    rng = np.random.default_rng(20261029)
    sc = _scored(52_000, 27)
    n = sc.hour.size
    shares = np.array([0.294, 0.436, 0.270])
    y = rng.choice(3, size=n, p=shares)
    p = np.tile(shares, (n, 1))
    ece = stats.ece(p, y, 10)
    gap = abs(np.abs(r2.ece_bins(p, y, sc, 10).sum(axis=0)).sum() / n - ece)
    assert gap > 1e-9 * ece  # the old check, rel_tol 1e-9, stopped here
    assert gap <= r2.ece_tolerance(n)
    out = r2.ece_interval(p, y, sc, rng.random(sc.hours.size), CFG, "ece", "test")
    assert out["n_boot"] == CFG.n_boot
    assert abs(stats.ece(p[1:], y[1:], 10) - ece) > 100 * r2.ece_tolerance(n)  # a lost row still trips it


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


# --- series.npz (iteration 4) ---------------------------------------------------------------


def _series(folder: Path) -> dict:
    """Every array of a run's series.npz, through the package's one reader of it (by digest)."""
    from olsprime.bench import power

    digest = json.loads((folder / "report2.json").read_text(encoding="utf-8"))["series"]["sha256"]
    arrays, _, _, _ = power.load_series(folder / r2.SERIES_FILE, digest, keys=[k for k, _, _ in r2.SERIES_KEYS])
    return arrays


def test_series_npz_reproduces_the_reports_means_and_holds_its_rows(two_runs):
    a, _, metrics, _ = two_runs
    s = _series(a)
    assert [str(x) for x in s["arms"]] == list(ARMS)
    for key, kind, _ in r2.SERIES_KEYS:
        assert s[key].dtype.str.startswith(kind), key
    t, n = metrics["scored"]["hours"], metrics["scored"]["rows"]
    assert s["log_loss"].shape == s["brier"].shape == (len(ARMS), t) and s["probs"].shape == (len(ARMS), n, 3)
    assert s["count"].sum() == n and np.all(np.diff(s["hours"]) > 0)
    sc = r2.Scored(np.arange(n), s["label"], s["hour"], s["hours"], s["count"])
    for i, arm in enumerate(ARMS):  # the same arrays, the same means, to the bit
        assert float(s["log_loss"][i].mean()) == metrics["arms"][arm]["log_loss"]["estimate"]
        assert float(s["brier"][i].mean()) == metrics["arms"][arm]["brier"]["estimate"]
        losses = stats.per_row_losses(s["probs"][i], s["label"])  # the scored rows give back the hourly series
        assert np.array_equal(r2.hourly(losses.log_loss, sc), s["log_loss"][i])
        assert np.array_equal(r2.hourly(losses.brier, sc), s["brier"][i])
    for c in metrics["comparisons"]:
        ia, ib = ARMS.index(c["a"]), ARMS.index(c["b"])
        assert float((s["log_loss"][ia] - s["log_loss"][ib]).mean()) == c["mean_diff"]
    assert [str(x) for x in s["symbols"]] == metrics["panel"]["symbols"]
    assert np.array_equal(s["open_time"], s["hours"][s["hour"]])  # each row sits in its bar's open hour
    assert np.all(s["decision_time"] > s["open_time"]) and np.all(s["label_known_at"] > s["decision_time"])
    assert np.all(np.isfinite(s["fwd_return"])) and set(np.unique(s["label"]).tolist()) <= {0, 1, 2}
    # the label is the class of fwd_return against its threshold: down rows have lower returns than up rows
    assert s["fwd_return"][s["label"] == 0].max() < s["fwd_return"][s["label"] == 2].min()


def test_series_npz_is_deterministic_and_its_digest_is_in_report2_json(two_runs, tmp_path):
    a, b, metrics, _ = two_runs
    data = (a / r2.SERIES_FILE).read_bytes()
    assert data == (b / r2.SERIES_FILE).read_bytes()
    assert metrics["series"] == {"file": r2.SERIES_FILE, "sha256": hashlib.sha256(data).hexdigest()}
    assert r2.series_line(metrics["series"]) in (a / r2.REPORT_FILE).read_text(encoding="utf-8").splitlines()
    # entries in SERIES_KEYS order, stored, every one dated 1980-01-01 00:00 in its local and central headers
    names = [f"{k}.npy".encode() for k, _, _ in r2.SERIES_KEYS]
    positions = [data.index(name) for name in names]
    assert positions == sorted(positions)
    local = struct.pack("<IHHHHH", 0x04034B50, 20, 0, 0, 0, 0x21)
    central = struct.pack("<IHHHHHH", 0x02014B50, 20, 20, 0, 0, 0, 0x21)
    assert data.startswith(local) and data.count(local) == len(names) and data.count(central) == len(names)
    # a later write of the same arrays gives the same bytes, whatever the clock says
    arrays = {k: np.arange(5, dtype=np.int64) * i for i, k in enumerate(("x", "y"))}
    r2.write_npz(tmp_path / "one.npz", arrays)
    r2.write_npz(tmp_path / "two.npz", arrays)
    assert (tmp_path / "one.npz").read_bytes() == (tmp_path / "two.npz").read_bytes()


def _older_from(new: Path, older: Path) -> None:
    """Undo, by hand, every allowed difference: what iteration 3's code would have written for the same numbers."""
    older.mkdir()
    config = json.loads((new / "config.json").read_text(encoding="utf-8"))
    report = json.loads((new / "report2.json").read_text(encoding="utf-8"))
    md = (new / r2.REPORT_FILE).read_text(encoding="utf-8").split("\n")
    prov = config["provenance"]
    old_prov = {**prov, "git_commit": "b" * 40, "code": {**prov["code"], "sha256": "e" * 64, "files": prov["code"]["files"] + 1}}
    for new_line, old_line in zip(r2.prov_mod.render_md(prov)[2:4], r2.prov_mod.render_md(old_prov)[2:4], strict=True):
        md[md.index(new_line)] = old_line
    config["provenance"] = old_prov
    md.remove(r2.series_line(report.pop("series")))
    words = []
    for name in ("previewed", "unseen"):
        c = next(c for c in report["preview_split"][name]["comparisons"] if c["hypothesis"] == "H12")
        words.append("rule met" if c["rule_met"] else "rule not met")
        boot = c["bootstrap"]
        c["rule_met"] = r2.passes(c["dm_p"], boot["low"], boot["high"], c["mean_diff"])
        del c["rule"]
    i = next(i for i, line in enumerate(md) if line.startswith("| H12: "))
    for w in words:
        md[i] = md[i].replace(f", {w}; identical hours", "; identical hours", 1)
    assert md[i + 2].startswith("H12's parts read H12's own rule") and md[i + 3] == ""
    del md[i + 2 : i + 4]
    j = next(j for j, line in enumerate(md) if line.startswith("Coverage: "))
    md[j] = md[j].replace(r2.COVERAGE_BEFORE, r2.COVERAGE_BEFORE_ITERATION3)
    (older / "config.json").write_bytes(r2._json_bytes(config))
    (older / "report2.json").write_bytes(r2._json_bytes(report))
    (older / r2.REPORT_FILE).write_bytes("\n".join(md).encode("utf-8"))


def test_the_comparison_with_an_older_run_allows_the_named_differences_and_nothing_else(two_runs, tmp_path, capsys):
    a, _, _, _ = two_runs
    older = tmp_path / "older"
    _older_from(a, older)
    problems, met, summary = r2.compare_with_older(older, a)
    assert problems == [], problems
    assert [x.name for x in met] == [x.name for x in r2.ITERATION3_ALLOWANCES]
    assert summary[-1] == f"{r2.SERIES_FILE}: only in the re-run"
    assert all(line.endswith("identical after the allowances") for line in summary[:3])
    assert r2.compare_command(older, a, CFG, rerun=False) == 0
    out = capsys.readouterr().out
    assert out.count("[met]") == len(r2.ITERATION3_ALLOWANCES) and "comparison passed" in out
    assert any("only in the re-run" in p for p in r2.diff_runs(older, a))  # a plain diff still fails

    def changed(name: str, old: bytes, new: bytes) -> list[str]:
        bad = tmp_path / f"bad-{len(list(tmp_path.iterdir()))}"
        bad.mkdir()
        for n in ("config.json", "report2.json", r2.REPORT_FILE):
            (bad / n).write_bytes((older / n).read_bytes())
        text = (bad / name).read_bytes()
        assert old in text, (name, old)
        (bad / name).write_bytes(text.replace(old, new, 1))
        return r2.compare_with_older(bad, a)[0]

    report = json.loads((older / "report2.json").read_text(encoding="utf-8"))
    ll = repr(report["arms"]["matrix-1y"]["log_loss"]["estimate"]).encode()
    k = next(i for i in range(4, len(ll)) if ll[i : i + 1].isdigit())  # one changed digit in one number
    one_digit = ll[:k] + (b"1" if ll[k : k + 1] != b"1" else b"2") + ll[k + 1 :]
    # refused either as a difference beyond the allowances or, if the edited float is no longer written as Python
    # writes it, as a file this code could not have written: the comparison fails both ways
    assert any(p.startswith("report2.json: ") for p in changed("report2.json", ll, one_digit))
    small = repr(report["comparisons"][0]["dm_lags"]).encode()  # an integer: the edit stays canonical
    assert any(p.startswith("report2.json: differs beyond the allowances")
               for p in changed("report2.json", b'"dm_lags": ' + small, b'"dm_lags": ' + small + b"1"))  # fmt: skip
    hours = str(report["scored"]["hours"]).encode()  # a digit inside the Coverage line, whose clause may change
    assert any(p.startswith(f"{r2.REPORT_FILE}: differs") for p in changed(r2.REPORT_FILE, b"Coverage: " + hours, b"Coverage: 9" + hours))
    assert any(p.startswith("config.json: differs") for p in changed("config.json", b'"n_boot": 200', b'"n_boot": 201'))
    assert any(p.startswith(f"{r2.REPORT_FILE}: differs") for p in changed(r2.REPORT_FILE, b"## Arms", b"## Arm"))
    # an older H12 part whose rule_met was not the superiority rule it carried is refused, not rewritten
    h12 = next(c for c in report["preview_split"]["unseen"]["comparisons"] if c["hypothesis"] == "H12")
    word, other = (b"true", b"false") if h12["rule_met"] else (b"false", b"true")
    text = (older / "report2.json").read_bytes()
    at = text.index(b'"rule_met": ', text.index(b'"hypothesis": "H12"', text.index(b'"unseen"')))  # keys are sorted: it follows
    assert text[at + 12 : at + 12 + len(word)] == word
    bad = tmp_path / "bad-rule"
    bad.mkdir()
    for n in ("config.json", r2.REPORT_FILE):
        (bad / n).write_bytes((older / n).read_bytes())
    (bad / "report2.json").write_bytes(text[: at + 12] + other + text[at + 12 + len(word) :])
    problems, met, _ = r2.compare_with_older(bad, a)
    assert any("not the superiority rule" in p for p in problems) and "H12's rule in the parts" not in [x.name for x in met]
    capsys.readouterr()


def test_h12s_parts_read_h12s_own_rule(two_runs):
    a, _, metrics, _ = two_runs
    for name in ("previewed", "unseen"):
        part = metrics["preview_split"][name]
        c = next(c for c in part["comparisons"] if c["hypothesis"] == "H12")
        assert c["rule_met"] == r2.h12_rule_met(c["upper_95_one_sided"], c["bootstrap"]["q95"], part["h12_margin"])
        assert "whole period only" in c["rule"] and "at most 36" in c["rule"]
        assert all("rule" not in o for o in part["comparisons"] if o["hypothesis"] != "H12")
    report = (a / r2.REPORT_FILE).read_text(encoding="utf-8")
    assert r2.H12_PART_NOTE.format(max_fits=36) in report
    row = next(line for line in report.splitlines() if line.startswith("| H12: "))
    assert row.count("rule met") + row.count("rule not met") == 2
    # on hand-built numbers: a difference far inside the margin meets H12's rule, though it fails the superiority rule
    rng = np.random.default_rng(4)
    y = 1.0 + 0.1 * rng.random(400)
    part = {"x": y + 0.001 * rng.standard_normal(400), "y": y}
    c = r2._compare_part(part, "x", "y", 1.0, CFG, "t", h12_margin=0.05)
    assert c["rule_met"] is True and r2.compare_arms(part, "x", "y", 1.0, CFG, "t")["passes"] is False


def test_the_per_file_summary_comes_first(two_runs, tmp_path, capsys):
    a, _, _, _ = two_runs
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in r2.compared_files(a):
        (copy / name).write_bytes((a / name).read_bytes())
    assert r2.summarize_runs(a, copy) == [f"{n}: identical" for n in r2.COMPARED]
    md = (copy / r2.REPORT_FILE).read_bytes()
    (copy / r2.REPORT_FILE).write_bytes(md.replace(b"## Arms", b"## Armz", 1))
    (copy / r2.SERIES_FILE).write_bytes((a / r2.SERIES_FILE).read_bytes() + b"\0")
    summary = r2.summarize_runs(a, copy)
    assert summary[2] == f"{r2.REPORT_FILE}: 2 differing lines (1 only in the expected run, 1 only in the re-run)"
    assert summary[3].startswith(f"{r2.SERIES_FILE}: differs (binary; ")
    assert any(p.startswith(f"{r2.SERIES_FILE}: differs") for p in r2.diff_runs(a, copy))


def test_the_ece_docstring_says_what_the_check_cannot_see():
    doc = r2.ece_tolerance.__doc__
    assert "Not a row put in another bin" in doc and "cannot see it" in doc and "counted twice" in doc
