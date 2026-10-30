"""``olsprime.explain`` (``why``, ``series``, ``sample``) on the synthetic store of ``test_report2.py``.

A Report n°2 run is made once on that store; every call reads it by the SHA-256 of its ``series.npz``.

- ``why``'s tree is closed by a residual within its derived rounding bound (a group wrong by 0.3 fails
  loudly), and its per-feature view gives the readout to the bit when summed in the core's order;
- the forecast check passes at a scored row for every arm ``why`` explains, and fails when one bit of
  ``series.npz`` changes (the digest re-recorded); check 1 counts every entry of every weighted window,
  fails when a stamp is planted after ``as_of`` (down to as_of + 1 ms, and for every arm), and when
  sigma is read 40 days late (at an unscored row, where it is the only check); an unscored row runs
  only check 1, and says so;
- a period other than the default needs ``--test-months``, and 2025 is refused with it; a wrong digest,
  a stale extension and a ``--jsonl`` under the runs' explain or vault folder are refused before any
  contact line; one call writes one contact line per row asked; two identical calls give identical bytes;
- ``series``'s shares match a direct computation; ``sample`` is a function of its seed;
- H1 on the explainer: with every member's bars after a cut perturbed, ``why`` at rows decided by the
  cut does not move by a bit, for every arm; a planted late sigma moves, and check 1 stops it.

Every contact goes to a temporary log (and ``conftest`` points the runs folder at a temporary folder).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from olsprime.bench import power
from olsprime.bench import report2 as r2
from olsprime.explain import cli
from olsprime.explain import why as why_mod
from olsprime.explain.common import (
    DEFAULT_PERIOD,
    LoudCheckFailed,
    Refused,
    check_as_of,
    check_period,
    json_line,
    open_run,
    parse_period,
    parse_time,
)
from olsprime.explain.why import (
    OUTPUTS,
    WHY_ARMS,
    Explainer,
    close_parts,
    seq,
    tree_view,
)
from olsprime.features.basic import FEATURE_COLUMNS
from olsprime.features.panel import hourly_panel
from olsprime.models.matrix import (
    DAY_MS,
    MatrixSystem,
    iso,
    month_start_ms,
    panel_arrays,
)
from olsprime.paths import runs_dir
from tests.test_panel import fake_bars, universe_frame
from tests.test_protocol import store
from tests.test_report2 import CFG, FAKE_MANIFEST, HOURS, MONTHS, PROVENANCE, START

ALL_KEYS = [k for k, _, _ in r2.SERIES_KEYS]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The synthetic store, its universe, and one Report n°2 run on it."""
    root = tmp_path_factory.mktemp("store")
    for seed, (s, n) in enumerate(HOURS.items()):
        store(root, fake_bars(s, n, seed=seed, start=START), s, "1h")
    universe = universe_frame({s: MONTHS for s in HOURS})
    run = tmp_path_factory.mktemp("run")
    r2.run_report(CFG, run, data_dir=root, universe=universe, manifest=FAKE_MANIFEST, provenance=PROVENANCE)
    sha = json.loads((run / "report2.json").read_text(encoding="utf-8"))["series"]["sha256"]
    arrays, _, _, _ = power.load_series(run / r2.SERIES_FILE, sha, ALL_KEYS)
    return {"root": root, "universe": universe, "run": run, "sha": sha, "arrays": arrays}


def call(world, argv: list[str], tmp_path: Path, *, run: Path | None = None, sha: str | None = None, factory=MatrixSystem,
         contacts: Path | None = None, jsonl: Path | None = None) -> tuple[int, str, str, list[dict]]:  # fmt: skip
    """Run one explain call; returns its exit code, stdout, stderr and JSON lines."""
    out, err = io.StringIO(), io.StringIO()
    jsonl = jsonl or tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}.jsonl"
    full = [*argv, "--run", str(run or world["run"]), "--sha256", sha or world["sha"], "--jsonl", str(jsonl)]
    code = cli.main(full, data_dir=world["root"], universe=world["universe"], contacts=contacts or tmp_path / "CONTACTS.log",
                    system_factory=factory, out=out, err=err)  # fmt: skip
    lines = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines()] if code == 0 and jsonl.exists() else []
    return code, out.getvalue(), err.getvalue(), lines


def scored_row(world, symbol: str = "AAAUSDT", hour: str = "2019-01-15T12:00:00+00:00") -> tuple[int, str]:
    """A scored row of series.npz near ``hour``: its index and its bar's open."""
    a = world["arrays"]
    code = [str(s) for s in a["symbols"]].index(symbol)
    i = int(np.flatnonzero((a["symbol"] == code) & (a["open_time"] >= parse_time(hour)))[0])
    return i, iso(int(a["open_time"][i]))


def bits(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)


# --- why: the views add up ---------------------------------------------------------------------


@pytest.mark.parametrize("by", ["year", "month", "day"])
def test_why_parts_add_up_to_the_raw_readout(world, tmp_path, by):
    _, at = scored_row(world)
    code, text, err, lines = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-select", "--by", by, "--top", "3", "--test-months"], tmp_path)
    assert code == 0, err
    (r,) = lines
    tree, raw = r["tree"], r["cell"]["raw"]
    assert "not by their decision time" in tree["grouping"] and "not by their decision time" in text
    for o in OUTPUTS:
        t = tree[o]
        parts = [p["c"] for p in t["top"]] + ([t["rest"]] if t["rest"] is not None else []) + t["closing_residual"]
        assert seq(parts) == raw[o] and t["closed"] is True and t["raw"] == raw[o]
        assert len(t["top"]) == min(3, tree["groups"]) and t["rest_groups"] == tree["groups"] - len(t["top"])
        # the residual is rounding: within the derived bound, and the bound itself is small
        assert t["closing_residual_size"] == math.fsum(abs(x) for x in t["closing_residual"]) <= t["rounding_bound"] < 1e-9
        assert [abs(p["c"]) for p in t["top"]] == sorted((abs(p["c"]) for p in t["top"]), reverse=True)
    assert "closed by a residual of" in text and "within the rounding bound" in text
    assert len(text.splitlines()) <= 40


def test_a_tree_that_does_not_partition_the_readout_fails_loudly():
    groups = [("2018-10", np.array([0.2, 0.3, 0.5, 0.01])), ("2018-11", np.array([0.1, 0.0, -0.1, 0.02]))]
    raw = groups[0][1] + groups[1][1]
    bound = np.full(4, 1e-12)
    t = tree_view(groups, raw, 1, bound, "test")
    assert all(t[o]["closing_residual_size"] <= 1e-12 for o in OUTPUTS)
    wrong = [(groups[0][0], groups[0][1] + np.array([0.3, 0.0, 0.0, 0.0])), groups[1]]  # one group wrong by 0.3
    with pytest.raises(LoudCheckFailed, match="beyond the rounding bound"):
        tree_view(wrong, raw, 1, bound, "test")
    with pytest.raises(LoudCheckFailed):
        close_parts([1.0, 2.0], 3.0 + 1e-9, 1e-12, "test")
    assert close_parts([1.0, 2.0], 3.0, 0.0, "test") == []


def test_the_per_feature_view_sums_to_the_readout_in_the_cores_order(world, tmp_path):
    _, at = scored_row(world)
    for arm in ("matrix-select", "matrix-1y", "matrix-cusum"):
        code, text, err, (r,) = call(world, ["why", "AAAUSDT", at, "--arm", arm, "--test-months"], tmp_path)
        assert code == 0, err
        fv, raw = r["features"], r["cell"]["raw"]
        for o in OUTPUTS:
            by_feature = {p["feature"]: p["part"] for p in fv[o]["parts"]}
            in_order = [by_feature[f] for f in FEATURE_COLUMNS]  # the core's order: feature 0, 1, ..., then mean_y last
            total = fv[o]["mean_y"] + seq(in_order)
            assert bits(total) == bits(raw[o]), (arm, o)
            assert fv[o]["bit_exact"] is True and fv[o]["sum_in_core_order"] == raw[o]
            assert fv[o]["numpy_gap"] < 1e-12
        assert fv["bit_exact_all"] is True and fv["statement"].startswith("checked at this call")
        assert "checked at this call" in text


def test_the_diagnostics_and_top_entries(world, tmp_path):
    _, at = scored_row(world)
    code, _, err, (r,) = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--top", "7", "--test-months"], tmp_path)
    assert code == 0, err
    dg = r["diagnostics"]
    assert abs(dg["sum_w"] - 1.0) < 1e-9 and dg["sum_abs_w"] >= 1.0 and dg["negative_mass"] == pytest.approx((dg["sum_abs_w"] - 1.0) / 2, abs=1e-9)
    assert dg["n"] == r["cell"]["n_window"] and dg["effective_sample_size"] > 1 and dg["leverage"] > 0
    ents = r["top_entries"]
    assert len(ents) == 7 and [abs(e["weight"]) for e in ents] == sorted((abs(e["weight"]) for e in ents), reverse=True)
    for e in ents:
        assert e["label_known_at"] <= r["row"]["decision_time"] and len(e["delta"]) == 15 and e["label"] in (0, 1, 2)
    assert r["checks"]["stamps"]["top_entries_checked_on_raw_bars"] == 7 and r["checks"]["stamps"]["matrix_stamps_used"] is False
    assert r["exploratory"].startswith("exploratory, test months")


# --- the loud checks ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", WHY_ARMS)
def test_the_forecast_check_passes_at_a_scored_row_for_every_arm(world, tmp_path, arm):
    i, at = scored_row(world, "BBBUSDT", "2019-02-10T05:00:00+00:00")
    code, text, err, (r,) = call(world, ["why", "BBBUSDT", at, "--arm", arm, "--top", "4", "--test-months"], tmp_path)
    assert code == 0, err
    fc = r["checks"]["forecast"]
    assert fc["applies"] is True and fc["series_row"] == i and fc["equal_to_the_bit"] is True
    arm_i = [str(x) for x in world["arrays"]["arms"]].index(arm)
    assert [float.fromhex(h) for h in fc["probs_hex"]] == world["arrays"]["probs"][arm_i, i].tolist()
    st = r["checks"]["stamps"]
    assert "to the bit" in text and st["passed"] is True
    # every entry of every weighted window is checked: the count is the windows' n, summed over a stack's weighted windows
    if arm == "matrix-stack":
        want = sum(w["cell"]["n_window"] for w in r["windows"] if w["alpha"] > 0 and not w["cell"]["fallback_to_climatology"])
    else:
        want = r["cell"]["n_window"]
    assert st["entries"]["checked"] == want > 0 and sum(st["entries"]["per_weighted_window"]) == want
    # sigma and the climatology in force equal the rule's, recomputed from panel rows
    month = r["row"]["month"]
    assert st["fold_statistics"] and all(x["sigma_equal_to_the_rule"] and x["climatology_equal_to_the_rule"] for x in st["fold_statistics"])
    if arm != "matrix-cusum":
        assert st["fold_statistics"][0]["fit_time"].startswith(month) and st["fold_statistics"][0]["entries_behind_sigma"] > 1
    assert any("features" in x for x in st["does_not_cover"]) and any("sigma" in x for x in st["covers"])
    if arm in ("softmax-1y", "knn-1y"):
        gap, bound = r["cell"]["numpy_readout_gap"], r["cell"]["numpy_readout_gap_bound"]
        assert "features" not in r and all(gap[o] <= bound[o] < 1e-9 for o in OUTPUTS)
        if arm == "knn-1y":  # the core's order replicated: the same bits
            assert all(gap[o] == 0.0 for o in OUTPUTS)
        assert "within its bound" in text
    if arm == "matrix-stack":
        assert len(r["windows"]) == 3 and abs(sum(w["alpha"] for w in r["windows"]) - 1) < 1e-12
    if arm == "matrix-adaptive":
        assert st["rows behind the adaptive rho"]["checked"] == r["choices"]["adaptive"]["rows_scored"]


def _bit_flipped_run(world, tmp_path: Path, arm: str, i: int) -> tuple[Path, str]:
    """A copy of the run whose series.npz differs by one bit, at ``arm``'s probability of row i, with its digest re-recorded."""
    new = tmp_path / "flipped"
    new.mkdir()
    arrays = {k: np.array(v) for k, v in world["arrays"].items()}
    j = [str(x) for x in arrays["arms"]].index(arm)
    flat = arrays["probs"].view(np.uint64)
    flat[j, i, 2] ^= np.uint64(1)
    r2.write_npz(new / r2.SERIES_FILE, arrays)
    sha = r2.sha256_file(new / r2.SERIES_FILE)
    report = json.loads((world["run"] / "report2.json").read_text(encoding="utf-8"))
    report["series"]["sha256"] = sha
    (new / "report2.json").write_bytes(r2._json_bytes(report))
    (new / "config.json").write_bytes((world["run"] / "config.json").read_bytes())
    old = (world["run"] / r2.SERIES_FILE).read_bytes()
    flipped = (new / r2.SERIES_FILE).read_bytes()
    # one bit of the data, plus the CRC-32 of that member in its local and central headers
    diff = [k for k, (x, y) in enumerate(zip(old, flipped, strict=True)) if x != y]
    crc = {k for sig, at in ((b"PK\x03\x04", 14), (b"PK\x01\x02", 16)) for p in _find_all(old, sig) for k in range(p + at, p + at + 4)}
    data_bytes = [k for k in diff if k not in crc]
    assert len(old) == len(flipped) and len(data_bytes) == 1 and (old[data_bytes[0]] ^ flipped[data_bytes[0]]).bit_count() == 1
    return new, sha


def _find_all(data: bytes, sig: bytes) -> list[int]:
    out, at = [], data.find(sig)
    while at >= 0:
        out.append(at)
        at = data.find(sig, at + 1)
    return out


def test_the_forecast_check_fails_when_the_series_changes_by_one_bit(world, tmp_path):
    i, at = scored_row(world)
    new, sha = _bit_flipped_run(world, tmp_path, "matrix-select", i)
    code, text, err, lines = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-select", "--test-months"], tmp_path, run=new, sha=sha)
    assert code == 3 and "LOUD CHECK FAILED" in err and "forecast check" in err
    assert text == "" and lines == []  # nothing is printed as a result
    # another arm at the same row, and the same arm at another row, still pass
    assert call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--test-months"], tmp_path, run=new, sha=sha)[0] == 0


def _planted(target: int, stamp: int) -> type[MatrixSystem]:
    """!!! A planted leak, for the tests: one entry's Matrix stamp set to ``stamp`` instead of its label_known_at. !!!"""

    class Planted(MatrixSystem):
        def entry_times(self, rows: np.ndarray) -> np.ndarray:
            t = self.data.lka_ms[rows].copy()
            t[rows == target] = stamp
            return t

    return Planted


class AllPlanted(MatrixSystem):
    """!!! A planted leak: every entry stamped with its decision time. !!!"""

    def entry_times(self, rows):
        return self.data.decision_ms[rows]


class LateSigma(MatrixSystem):
    """!!! A planted leak: sigma^2 of a fit read 40 days late (entries known by F + 40 d - 24 h). !!!"""

    def variance_at(self, fit_ms, month_ms):
        return super().variance_at(fit_ms + 40 * DAY_MS, month_ms)


def _panel(world):
    return panel_arrays(hourly_panel(world["universe"], first=r2.Month(*CFG.first_month), last=r2.Month(*CFG.last_month), data_dir=world["root"]))


def test_the_stamp_check_fails_when_a_stamp_is_planted_after_as_of(world, tmp_path):
    _, at = scored_row(world)
    ok, _, _, (r,) = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--test-months"], tmp_path)
    assert ok == 0
    d = _panel(world)
    as_of = parse_time(r["row"]["decision_time"])
    ccc = d.symbols.index("CCCUSDT")
    # plant: the row of CCCUSDT decided 3 hours before as_of, whose label is known only 22 hours after it
    target = int(np.flatnonzero((d.symbol == ccc) & (d.decision_ms == as_of - 3 * 3_600_000) & d.complete)[0])
    assert d.lka_ms[target] > as_of
    code, text, err, lines = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--test-months"], tmp_path,
                                  factory=_planted(target, int(d.decision_ms[target])))  # fmt: skip
    assert code == 3 and "stamp check" in err and "1 of" in err and "CCCUSDT" in err
    assert text == "" and lines == []
    # the tightest plant: an entry whose label is known at as_of + 1 ms, stamped at as_of so that the window reads it
    edge = int(np.flatnonzero((d.symbol == ccc) & (d.open_ms == as_of + 1 - 25 * 3_600_000) & d.complete)[0])
    assert d.lka_ms[edge] == as_of + 1
    code, text, err, lines = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--test-months"], tmp_path,
                                  factory=_planted(edge, as_of))  # fmt: skip
    assert code == 3 and "1 of" in err and iso(as_of + 1) in err and text == "" and lines == []
    # every entry stamped with its decision time: caught for every arm why explains
    for arm in WHY_ARMS:
        code, text, err, lines = call(world, ["why", "AAAUSDT", at, "--arm", arm, "--test-months"], tmp_path, factory=AllPlanted)
        assert code == 3 and "stamp check" in err and text == "" and lines == [], arm


def test_check_1_catches_a_sigma_read_40_days_late_at_an_unscored_row(world, tmp_path):
    argv = ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm"]
    for arm in WHY_ARMS:
        code, _, err, (r,) = call(world, [*argv, arm], tmp_path)
        assert code == 0 and r["row"]["scored"] is False and r["checks"]["forecast"]["applies"] is False, err
        code, text, err, lines = call(world, [*argv, arm], tmp_path, factory=LateSigma)
        assert code == 3 and "sigma reads entries known later" in err and text == "" and lines == [], arm


def test_an_unscored_row_runs_only_the_stamp_check_and_says_so(world, tmp_path):
    code, text, err, (r,) = call(world, ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-select"], tmp_path)
    assert code == 0, err
    assert r["row"]["scored"] is False and r["checks"]["forecast"]["applies"] is False
    assert "only the stamp check applies" in r["checks"]["forecast"]["note"] and "NOT SCORED" in text and "NOT APPLICABLE" in text
    assert r["checks"]["stamps"]["passed"] is True and r["choices"]["fit"] is None and "defaults" in r["choices"]["note"]
    assert r["exploratory"] is None and "EXPLORATORY" not in text
    assert r["checks"]["stamps"]["entries"]["latest_label_known_at"] <= r["row"]["decision_time"]


# --- periods, digests, contacts, determinism --------------------------------------------------


def test_a_period_other_than_the_default_needs_the_flag_and_2025_is_refused(world, tmp_path):
    _, at = scored_row(world)
    code, _, err, _ = call(world, ["why", "AAAUSDT", at, "--arm", "matrix-select"], tmp_path)
    assert code == 2 and "--test-months" in err
    for argv in (["why", "AAAUSDT", "2025-01-02T00:00:00+00:00", "--arm", "matrix-select", "--test-months"],
                 ["series", "matrix-select", "--period", "2024-12:2025-01", "--test-months"],
                 ["sample", "matrix-select", "--period", "2025-03", "--n", "5", "--seed", "1", "--test-months"]):  # fmt: skip
        code, _, err, _ = call(world, argv, tmp_path)
        assert code == 2 and "2025" in err, argv
    assert call(world, ["series", "matrix-select", "--period", "2019-01:2019-02"], tmp_path)[0] == 2
    assert call(world, ["series", "matrix-select"], tmp_path)[0] == 2  # series reads test rows: always the flag
    assert call(world, ["sample", "matrix-1y", "--n", "3", "--seed", "0"], tmp_path)[0] == 2
    # the rules themselves
    check_period(DEFAULT_PERIOD, False)
    with pytest.raises(Refused, match="--test-months"):
        check_period(parse_period("2018-11:2019-01"), False)
    check_period(parse_period("2018-11:2019-01"), True)
    for p in ("2024-06:2025-01", "2025-01", "2026-02"):
        with pytest.raises(Refused, match="2025 or later"):
            check_period(parse_period(p), True)
    with pytest.raises(Refused, match="2025 or later"):
        check_as_of(month_start_ms(2025, 1), True)
    check_as_of(month_start_ms(2025, 1) - 1, True)
    with pytest.raises(Refused, match="outside the default"):
        check_as_of(month_start_ms(2019, 1), False)
    assert call(world, ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "climatology-panel"], tmp_path)[0] == 2  # refused by _why_arm
    assert call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--top", "51", "--test-months"], tmp_path)[0] == 2
    assert call(world, ["sample", "matrix-1y", "--n", "20001", "--seed", "0", "--test-months", "--period", "2019-01"], tmp_path)[0] == 2


def test_a_wrong_digest_is_refused(world, tmp_path):
    code, text, err, lines = call(world, ["series", "matrix-select", "--period", "2019-01:2019-02", "--test-months"], tmp_path, sha="0" * 64)
    assert code == 2 and "SHA-256" in err and text == "" and lines == []
    # the right digest of a file that its run's report2.json does not record
    other = tmp_path / "other"
    other.mkdir()
    for n in ("config.json", "report2.json"):
        (other / n).write_bytes((world["run"] / n).read_bytes())
    data = bytearray((world["run"] / r2.SERIES_FILE).read_bytes())
    data[-30] ^= 1
    (other / r2.SERIES_FILE).write_bytes(bytes(data))
    sha = r2.sha256_file(other / r2.SERIES_FILE)
    code, _, err, _ = call(world, ["series", "matrix-select", "--period", "2019-01", "--test-months"], tmp_path, run=other, sha=sha)
    assert code == 2 and "report2.json" in err


def test_one_call_writes_one_contact_line(world, tmp_path):
    log = tmp_path / "explain" / "CONTACTS.log"
    _, at = scored_row(world)
    assert call(world, ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-1y"], tmp_path, contacts=log)[0] == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    c = json.loads(lines[0])
    assert set(c) == {"utc", "subcommand", "period", "arm", "commit", "dirty", "series_sha256"}
    assert (c["subcommand"], c["arm"], c["series_sha256"]) == ("why", "matrix-1y", world["sha"]) and "2018-12" in c["period"]
    # several --at values share one process; each still gets its own contact line
    assert call(world, ["why", "AAAUSDT", at, "--arm", "matrix-1y", "--test-months", "--at", "BBBUSDT", at], tmp_path, contacts=log)[0] == 0
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3
    assert call(world, ["series", "knn-1y", "matrix-1y", "--period", "2019-01:2019-02", "--test-months"], tmp_path, contacts=log)[0] == 0
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert (last["subcommand"], last["arm"], last["period"]) == ("series", "knn-1y,matrix-1y", "2019-01..2019-02")
    # a call refused before it opens the run folder touches nothing, and logs nothing
    assert call(world, ["series", "matrix-1y", "--period", "2025-01", "--test-months"], tmp_path, contacts=log)[0] == 2
    assert len(log.read_text(encoding="utf-8").splitlines()) == 4
    # exactly one line per sample call
    assert call(world, ["sample", "matrix-1y", "--period", "2019-02", "--n", "2", "--seed", "5", "--test-months"], tmp_path, contacts=log)[0] == 0
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert len(log.read_text(encoding="utf-8").splitlines()) == 5 and (last["subcommand"], last["period"]) == ("sample", "2019-02..2019-02")
    # an --at outside the default period, after an allowed first row, is refused without the flag: nothing logged
    code, text, err, lines = call(world, ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-1y", "--at", "BBBUSDT", at], tmp_path,
                                  contacts=log)  # fmt: skip
    assert code == 2 and "--test-months" in err and text == "" and lines == []
    # a wrong digest is refused after hashing alone, before any contact line
    for argv in (["series", "matrix-1y", "--period", "2019-01", "--test-months"], ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-1y"],
                 ["sample", "matrix-1y", "--period", "2019-01", "--n", "2", "--seed", "0", "--test-months"]):  # fmt: skip
        code, _, err, _ = call(world, argv, tmp_path, contacts=log, sha="0" * 64)
        assert code == 2 and "SHA-256" in err, argv
    assert len(log.read_text(encoding="utf-8").splitlines()) == 5


def test_the_contacts_file_is_named_in_the_provenance_and_the_text(world, tmp_path):
    log = tmp_path / "x" / ".." / "explain" / "CONTACTS.log"
    code, text, err, (r,) = call(world, ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-1y"], tmp_path, contacts=log)
    resolved = str(log.resolve())
    assert code == 0 and r["provenance"]["contacts_file"] == resolved and f"contact logged in {resolved}" in text, err
    assert r["provenance"]["panel_fingerprint_verified"] is True
    code, text, _, (s,) = call(world, ["series", "matrix-1y", "--period", "2019-01", "--test-months"], tmp_path, contacts=log)
    assert code == 0 and s["provenance"]["contacts_file"] == resolved and resolved in text


def test_a_stale_extension_is_refused_before_any_contact(world, tmp_path, monkeypatch):
    import types

    stale = types.SimpleNamespace(CentredReadout=type("CentredReadout", (), {"mean_x": None, "a": None, "forecast": None}))
    monkeypatch.setattr(why_mod, "rs", stale)
    log = tmp_path / "CONTACTS.log"
    for argv in (["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-1y"],
                 ["sample", "matrix-1y", "--period", "2019-01", "--n", "2", "--seed", "0", "--test-months"]):  # fmt: skip
        code, text, err, lines = call(world, argv, tmp_path, contacts=log)
        assert code == 2 and "CentredReadout.beta, .mean_y" in err and text == "" and lines == [], argv
    assert not log.exists()


def test_jsonl_is_a_new_file_outside_the_runs_explain_and_vault_folders(world, tmp_path):
    log = tmp_path / "CONTACTS.log"
    argv = ["series", "matrix-1y", "--period", "2019-01", "--test-months"]
    for target in (runs_dir() / "explain" / "out.jsonl", runs_dir() / "vault" / "x.jsonl", runs_dir() / "explain" / "CONTACTS.log"):
        code, text, err, _ = call(world, argv, tmp_path, contacts=log, jsonl=target)
        assert code == 2 and "explain never writes there" in err and text == "" and not target.exists(), target
    link = tmp_path / "link"
    try:
        link.symlink_to(runs_dir() / "vault", target_is_directory=True)
    except OSError:
        link = None  # no symlink privilege on this machine
    if link is not None:
        code, _, err, _ = call(world, argv, tmp_path, contacts=log, jsonl=link / "x.jsonl")
        assert code == 2 and "explain never writes there" in err
    existing = tmp_path / "existing.jsonl"
    existing.write_text("keep\n", encoding="utf-8")
    code, _, err, _ = call(world, argv, tmp_path, contacts=log, jsonl=existing)
    assert code == 2 and "exists" in err and existing.read_text(encoding="utf-8") == "keep\n"
    assert not log.exists()  # every refusal came before the contact line


def test_two_identical_calls_give_identical_bytes(world, tmp_path):
    _, at = scored_row(world)
    for argv in (["why", "AAAUSDT", at, "--arm", "matrix-stack", "--by", "day", "--test-months", "--at", "CCCUSDT", "2018-11-20T03:00:00+00:00"],
                 ["series", "matrix-select", "--period", "2019-01:2019-02", "--by", "month", "--test-months"],
                 ["sample", "matrix-adaptive", "--period", "2019-01:2019-02", "--n", "6", "--seed", "3", "--test-months"]):  # fmt: skip
        outs = []
        for _ in range(2):
            code, text, err, _ = call(world, argv, tmp_path)
            assert code == 0, err
            jsonl = max(tmp_path.glob("out-*.jsonl"), key=lambda p: int(p.stem[4:]))
            outs.append((text, jsonl.read_bytes()))
        assert outs[0] == outs[1], argv
    # without --jsonl, the JSON lines follow the summaries on stdout
    out = io.StringIO()
    argv = ["series", "matrix-1y", "--period", "2019-01", "--test-months", "--run", str(world["run"]), "--sha256", world["sha"]]
    assert cli.main(argv, contacts=tmp_path / "c.log", out=out, err=io.StringIO()) == 0
    text, _, payload = out.getvalue().partition(cli.JSON_MARK + "\n")
    assert text and json.loads(payload)["kind"] == "series"


# --- series and sample ------------------------------------------------------------------------


def test_series_shares_match_a_direct_computation(world, tmp_path):
    a = world["arrays"]
    arms = [str(x) for x in a["arms"]]
    code, text, err, (r,) = call(world, ["series", "matrix-select", "--period", "2019-01:2019-02", "--by", "month", "--top", "5", "--test-months"], tmp_path)
    assert code == 0, err
    d = a["brier"][arms.index("climatology-panel")] - a["brier"][arms.index("matrix-select")]
    total = d.sum()
    assert r["hours"] == d.size and r["total"] == pytest.approx(total, rel=1e-12)
    bss = 1 - a["brier"][arms.index("matrix-select")].mean() / a["brier"][arms.index("climatology-panel")].mean()
    assert r["bss"] == pytest.approx(bss, rel=1e-12)
    report = json.loads((world["run"] / "report2.json").read_text(encoding="utf-8"))
    assert r["bss"] == pytest.approx(report["arms"]["matrix-select"]["bss"]["estimate"], rel=1e-12)
    for f in (0.01, 0.05, 0.10):
        m = max(1, math.ceil(f * d.size))
        top_abs = np.argsort(-np.abs(d), kind="stable")[:m]
        top_signed = np.argsort(-(d / total), kind="stable")[:m]
        s = r["top_shares"][f"top_{f:.0%}"]
        assert s["hours"] == m
        assert s["by_abs_part"] == pytest.approx(d[top_abs].sum() / total, rel=1e-9)
        assert s["by_signed_part"] == pytest.approx(d[top_signed].sum() / total, rel=1e-9)
    months = a["hours"].astype("datetime64[ms]").astype("datetime64[M]")
    assert [g["group"] for g in r["groups"]] == ["2019-01", "2019-02"]
    for g in r["groups"]:
        sel = months == np.datetime64(g["group"])
        assert g["share"] == pytest.approx(d[sel].sum() / total, rel=1e-9) and g["hours"] == int(sel.sum())
    assert sum(g["share"] for g in r["groups"]) == pytest.approx(1.0, rel=1e-12)
    top = r["top_hours"]
    assert len(top) == 5 and [abs(h["part"]) for h in top] == sorted((abs(h["part"]) for h in top), reverse=True)
    assert r["lorenz"]["cumulative_share_at_hour_fraction"]["100%"] == pytest.approx(1.0, rel=1e-9)
    # two arms: the hourly log-loss difference
    code, _, _, (r2b,) = call(world, ["series", "matrix-1y", "knn-1y", "--period", "2019-01:2019-02", "--test-months"], tmp_path)
    dd = a["log_loss"][arms.index("matrix-1y")] - a["log_loss"][arms.index("knn-1y")]
    assert code == 0 and r2b["mean_difference"] == pytest.approx(dd.mean(), rel=1e-12) and r2b["identical_hours"] == int((dd == 0).sum())
    assert r2b["mean_difference"] == pytest.approx(report["comparisons"][1]["mean_diff"], rel=1e-12)
    assert len(text.splitlines()) <= 40


def test_sample_is_a_function_of_its_seed(world, tmp_path):
    argv = ["sample", "matrix-1y", "--period", "2019-01:2019-02", "--n", "8", "--test-months"]
    one = call(world, [*argv, "--seed", "11"], tmp_path)
    two = call(world, [*argv, "--seed", "11"], tmp_path)
    other = call(world, [*argv, "--seed", "12"], tmp_path)
    assert one[0] == two[0] == other[0] == 0
    assert one[3] == two[3] and one[1] == two[1]
    rows = [ln["series_row"] for ln in one[3][:-1]]
    a = world["arrays"]
    pool = np.flatnonzero((a["open_time"] >= month_start_ms(2019, 1)) & (a["open_time"] < month_start_ms(2019, 3)))
    assert rows == pool[np.sort(np.random.default_rng(11).choice(pool.size, size=8, replace=False))].tolist()
    assert rows != [ln["series_row"] for ln in other[3][:-1]]
    summary = one[3][-1]
    assert summary["kind"] == "sample" and summary["rows"] == 8 and summary["checks"] == {"stamps_passed": 8, "forecast_equal_to_the_bit": 8}
    assert sum(dc["rows"] for dc in summary["deciles_of_negative_mass"]) == 8 - summary["rows_without_a_readout"]
    assert summary["per_feature_bit_exact_rows"] == summary["per_feature_rows"]
    tw = summary["twin"]
    assert tw["rows_checked"] == tw["rows_within_bound"] == 8 - summary["rows_without_a_readout"] and tw["max_gap_over_bound"] < 1
    for ln in one[3][:-1]:
        assert ln["twin"] is None or (ln["twin"]["within_bound"] == ln["twin"]["cells"] == 1 and ln["twin"]["rows"] == [ln["diagnostics"]["n"]])
    assert "Matrix-free twin: " in one[1]
    for ln in one[3][:-1]:
        p = ln["probs"]
        assert ln["log_loss"] == pytest.approx(-math.log(p[ln["label"]]), rel=1e-12)
        assert ln["checks"] == {"stamps": True, "forecast": True} and "top_entries" not in ln
    assert len(one[1].splitlines()) <= 40
    # the kernel and stack arms sample too, with their own notes
    for arm in ("softmax-1y", "matrix-stack"):
        code, _, err, lines = call(world, ["sample", arm, "--period", "2019-02", "--n", "3", "--seed", "0", "--test-months"], tmp_path)
        assert code == 0, err and lines[-1]["mean_abs_part"] is None


# --- the tree's partition and the Matrix-free twin ----------------------------------------------


def test_the_trees_partition_is_checked_apart_from_its_sum(world, tmp_path, monkeypatch):
    _, at = scored_row(world)
    argv = ["why", "AAAUSDT", at, "--by", "day", "--test-months", "--arm"]
    code, _, err, (r,) = call(world, [*argv, "matrix-select"], tmp_path)
    pt = r["tree"]["partition"]
    assert code == 0 and pt["same_labels_and_counts"] and pt["sum_w_y_within_bound"] and pt["groups"] == r["tree"]["groups"], err
    assert pt["max_gap"] < 1e-14 and pt["max_gap_over_bound"] < 1
    # !!! plant: the grouping stamps shifted by -1 h. Every group sum still closes on the readout; the recount catches it !!!
    honest_stamps = why_mod.Explainer.group_stamps
    monkeypatch.setattr(why_mod.Explainer, "group_stamps", lambda self, s, e: honest_stamps(self, s, e) - 3_600_000)
    for arm in WHY_ARMS:
        code, text, err, lines = call(world, [*argv, arm], tmp_path)
        assert code == 3 and "not those of the panel's label_known_at" in err and text == "" and lines == [], arm
    monkeypatch.setattr(why_mod.Explainer, "group_stamps", honest_stamps)
    # !!! plant: the groups' labels and counts kept, 1e-9 of one group's contribution moved to the next: the sum still closes !!!
    honest_groups = why_mod.Explainer.group_contributions

    def moved(self, s, e, contribution, by):
        groups, meta = honest_groups(self, s, e, contribution, by)
        (l0, c0), (l1, c1) = groups[0], groups[1]
        return [(l0, c0 + 1e-9), (l1, c1 - 1e-9), *groups[2:]], meta

    monkeypatch.setattr(why_mod.Explainer, "group_contributions", moved)
    for arm in ("matrix-1y", "knn-1y", "softmax-1y"):
        code, text, err, lines = call(world, [*argv, arm], tmp_path)
        assert code == 3 and "sum w_i y_i" in err and "beyond its bound" in err and text == "", arm


class ShortWindow(MatrixSystem):
    """!!! A planted Matrix-side defect: every ridge window ends just before its latest stamp (a partition point off). !!!"""

    def ridge_as_of(self, rows):
        dec = self.data.decision_ms[rows]
        i = np.searchsorted(self.entry_stamp, dec, side="right")
        return np.where(i > 0, self.entry_stamp[np.maximum(i - 1, 0)] - 1, dec)


def _one_entry_late(target: int) -> type[MatrixSystem]:
    """!!! A planted Matrix-side defect: one entry stamped a day after its label is known (a stamp out of order). Check 1 sees nothing. !!!"""

    class OneLate(MatrixSystem):
        def entry_times(self, rows):
            t = self.data.lka_ms[rows].copy()
            t[rows == target] += DAY_MS
            return t

    return OneLate


def test_the_matrix_free_twin_catches_a_matrix_side_defect(world, tmp_path):
    # honest: every weighted ridge cell of why carries the twin, within its bound
    for arm in ("matrix-select", "matrix-stack", "matrix-cusum", "matrix-adaptive"):
        code, text, err, (r,) = call(world, ["why", "BBBUSDT", "2019-02-10T05:00:00+00:00", "--arm", arm, "--test-months"], tmp_path)
        cells = [w["cell"] for w in r["windows"] if w["alpha"] > 0] if arm == "matrix-stack" else [r["cell"]]
        assert code == 0 and all(c["twin"]["within_bound"] and c["twin"]["rows"] == c["n_window"] for c in cells), err
        assert all(c["twin"]["gap"][o] <= c["twin"]["bound"][o] < 1e-6 for c in cells for o in OUTPUTS)
    # a window a few entries short (all entries of its latest stamp): sample stops on the twin, before check 2
    code, text, err, lines = call(world, ["sample", "matrix-1y", "--period", "2019-02", "--n", "3", "--seed", "0", "--test-months"], tmp_path,
                                  factory=ShortWindow)  # fmt: skip
    assert code == 3 and "Matrix-free twin" in err and "complete rows with label_known_at in the window" in err and text == ""
    # one entry short, at an unscored row: check 2 does not apply and check 1 passes, so the twin alone catches it
    d = _panel(world)
    as_of = parse_time("2018-12-10T12:59:59.999+00:00")
    known = np.flatnonzero(d.complete & (d.lka_ms <= as_of))
    target = int(known[np.argmax(d.lka_ms[known])])
    argv = ["why", "AAAUSDT", "2018-12-10T12:00:00+00:00", "--arm", "matrix-1y"]
    assert call(world, argv, tmp_path)[0] == 0
    code, text, err, lines = call(world, argv, tmp_path, factory=_one_entry_late(target))
    assert code == 3 and "Matrix-free twin" in err and text == "" and lines == [], err
    # the same count but one entry's state corrupted in the memory: check 1, the sigma rule and the tree all read Matrix and agree; the twin does not
    code, text, err, lines = call(world, [*argv[:-1], "matrix-select"], tmp_path, factory=_corrupt_entry(target))
    assert code == 3 and "Matrix-free twin" in err and "beyond its bound" in err and text == "" and lines == [], err


def _corrupt_entry(target: int) -> type[MatrixSystem]:
    """!!! A planted Matrix-side defect: one entry's state pushed to the memory with 1 added to every feature. !!!"""
    from olsprime.models import matrix as mx

    class Corrupt(MatrixSystem):
        def __init__(self, panel, cfg):
            super().__init__(panel, cfg)
            d, er = self.data, self.entry_rows
            xs = np.ascontiguousarray(d.x[er]).copy()
            xs[er == target] += 1.0
            ys = np.zeros((er.size, mx.Q))
            ys[np.arange(er.size), d.label[er]] = 1.0
            ys[:, 3] = d.fwd[er]
            self.matrix = mx.rs.Matrix(mx.P, mx.Q, block_log2=cfg.block_log2)
            self.matrix.extend(self.entry_stamp, xs, ys)

    return Corrupt


# --- H1 on the explainer -----------------------------------------------------------------------

H1_CUTS = ("2018-12-10T12:00:00+00:00", "2019-01-01T00:00:00+00:00", "2019-02-10T05:00:00+00:00")  # default period; a test month's first hour; a test hour
H1_ROWS_BEFORE = (0, 5)  # rows at the cut hour (decided at the cut bar's known_at) and 5 hours before
H1_EXCLUDED = {
    "checks.forecast": "check 2 matches the row's own label against series.npz's, and that label is known 25 h after the decision, after the cut",
    "row.label, row.fwd_return": "the row's own outcome, shown beside the forecast and never read by it: known after the cut by construction",
    "provenance": "not in the result object (it is added by the command line); its panel fingerprint hashes every row, the perturbed ones too",
}


def _comparable(res: dict) -> str:
    """The whole why result but the fields H1_EXCLUDED names: tree, per-feature parts, weights' digests, diagnostics, choices, check 1."""
    r = dict(res)
    r["row"] = {k: v for k, v in res["row"].items() if k not in ("label", "fwd_return")}
    r["checks"] = {k: v for k, v in res["checks"].items() if k != "forecast"}
    return json_line(r)


def test_h1_on_the_explainer(world):
    """Every member's bars after the cut perturbed (H1's perturbation and seeds, both directions): why at rows decided by the cut
    must not move by a bit, for every arm; a sigma read 40 days late moves, and check 1 stops it."""
    import polars as pl

    from olsprime.bench.panel_invariance import (
        RANDOM,
        PanelCut,
        PanelH1Config,
        build_panel,
        perturbed_bars,
    )
    from olsprime.features.panel import INTERVAL, fingerprint, member_months
    from olsprime.protocol import DEVELOPMENT, load_klines

    cfg = PanelH1Config()
    run = open_run(world["run"], world["sha"], cli.ROW_KEYS_READ)
    members = member_months(world["universe"], first=r2.Month(*CFG.first_month), last=r2.Month(*CFG.last_month))
    months = {s: members.filter(pl.col("symbol") == s)["month"].to_list() for s in HOURS}
    bars = {s: load_klines(s, INTERVAL, data_dir=world["root"], caller="tests.test_explain").filter(pl.col("known_at") < DEVELOPMENT.end)
            for s in HOURS}  # fmt: skip
    honest_panel = build_panel(bars, months)
    assert fingerprint(honest_panel) == run.report["panel"]["fingerprint_sha256"]  # the reference is the run's own panel

    def explainer(panel, factory=MatrixSystem):
        return Explainer(run, data_dir=world["root"], panel=panel, system_factory=factory)

    def rows(ex, hour_ms):
        return [(s, ex.find_row(s, hour_ms - h * 3_600_000)) for s in ("AAAUSDT", "BBBUSDT") for h in H1_ROWS_BEFORE]

    honest, late = explainer(honest_panel), explainer(honest_panel, LateSigma)
    assert honest.panel_verified
    compared, plant_moved = 0, set()
    for k, text in enumerate(H1_CUTS):
        cut = PanelCut(parse_time(text), RANDOM)
        ref = {}
        for s, row in rows(honest, cut.hour_ms):
            assert honest.system.data.decision_ms[row] <= cut.until_ms
            for arm in WHY_ARMS:
                res = honest.explain(row, arm, by="day", top=5, twin=True)  # both checks, on the run's panel
                assert res["checks"]["stamps"]["passed"] and res["checks"]["forecast"]["applies"] == res["row"]["scored"] == (res["row"]["month"] >= "2019-01")
                ref[s, row, arm] = _comparable(res)
        late_ref = {(s, row, arm): _comparable(late.explain(row, arm, by="day", top=5, checks="none"))
                    for s, row in rows(late, cut.hour_ms) for arm in WHY_ARMS} if k == 0 else {}  # fmt: skip
        for j in range(len(cfg.level_shifts)):
            new_bars, left = perturbed_bars(bars, cut, k, j, cfg)
            # a member with no bar after the cut (CCCUSDT ends earlier) is left as it is, as the panel H1 leaves it
            assert {"AAAUSDT", "BBBUSDT"}.isdisjoint(left) and all(v == "no bar after the cut" for v in left.values())
            panel = build_panel(new_bars, months)
            assert fingerprint(panel) != fingerprint(honest_panel)  # the future moved
            ex = explainer(panel)
            assert not ex.panel_verified
            for s, row in rows(ex, cut.hour_ms):
                for arm in WHY_ARMS:
                    assert _comparable(ex.explain(row, arm, by="day", top=5, checks="stamps", twin=True)) == ref[s, row, arm], (text, j, s, arm)
                    compared += 1
            if k == 0:  # the planted sigma: its why moves with the future, and check 1 stops it
                ex_late = explainer(panel, LateSigma)
                for s, row in rows(ex_late, cut.hour_ms):
                    for arm in WHY_ARMS:
                        if _comparable(ex_late.explain(row, arm, by="day", top=5, checks="none")) != late_ref[s, row, arm]:
                            plant_moved.add(arm)
                        with pytest.raises(LoudCheckFailed, match="sigma reads entries known later"):
                            ex_late.explain(row, arm, by="day", top=5, checks="stamps")
    assert compared == len(H1_CUTS) * len(cfg.level_shifts) * 4 * len(WHY_ARMS)
    assert plant_moved == set(WHY_ARMS)  # H1 catches the late sigma in every arm
