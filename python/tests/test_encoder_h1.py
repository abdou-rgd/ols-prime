"""The panel H1 of the encoder arms (``olsprime.bench.encoder_h1``) on a synthetic sub-panel, CPU torch.

- the pre-check passes bit for bit, with its scope test;
- every honest arm is invariant at every cut and at both purge cuts, bit for bit;
- ``cheat-stamp``, ``cheat-value`` and ``cheat-stamp-day`` are caught where predicted, ``cheat-purge`` at both purge cuts,
  ``cheat-slice`` at F - 30 days only, in both directions;
- a planted defect in the compare path is caught: one parameter bit flipped, one memory row moved;
- the scope test at every cut (a description) is identical on honest runs and flags a planted
  defect, without touching the pre-check's verdict;
- a run cut in the middle and resumed gives the same bytes as a run straight through; a resume
  with another configuration, build or driver is refused; a part whose digest fails is recomputed;
- the fallback's floor mode on CPU;
- the purge cuts' seeds, the climatology and the registered configuration.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from olsprime.bench import encoder_h1 as eh
from olsprime.bench import panel_invariance as pi
from olsprime.bench import provenance
from olsprime.bench.invariance import H1Config, perturbation_at
from olsprime.models.matrix import MatrixSystem, panel_arrays
from olsprime.train import checkpoint
from olsprime.train.fit import DAY_MS, HOUR_MS, year_start_ms

START = datetime(2018, 11, 12, tzinfo=UTC)  # a Monday
HOURS = [24 * 70, 24 * 70, 24 * 38]  # S02USDT's last bar opens 2018-12-19 23:00: no bar after the cut
TINY = (("d_model", 16), ("n_heads", 2), ("d_mlp", 32), ("n_layers", 1))
SHORT = (("epochs", 2), ("warmup_steps", 3), ("eval_batch", 256))
CFG = eh.EncoderH1Config(device="cpu", cuts=("2019-01-10T13:00",), choose_cuts=False, purge_fit_year=2019,
                         encoder=TINY, fit=SHORT, n_boot=50, synthetic="test: 3 synthetic members from 2018-11-12")  # fmt: skip
SMALL = dataclasses.replace(CFG, arms=("enc-ols",), controls=("cheat-stamp",), positive=False)
PROV = provenance.record({})


@pytest.fixture(scope="module")
def sub() -> eh.Subpanel:
    bars, months = eh.synthetic_subpanel(3, start=START, hours=HOURS)
    return eh.subpanel(bars, months)


@pytest.fixture(scope="module")
def full(sub, tmp_path_factory) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("encoder-h1")
    return eh.run(sub, CFG, out, "all", prov=PROV), out


@pytest.fixture(scope="module")
def small(sub, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("encoder-h1-small")
    eh.run(sub, SMALL, out, "all", prov=PROV)
    return out


def system(result: dict, name: str) -> dict:
    return next(s for s in result["systems"] if s["name"] == name)


# --- The stages on the synthetic sub-panel ---------------------------------------------------------


def test_the_precheck_passes_bit_for_bit_with_its_scope_test(full):
    result, _ = full
    pre = result["precheck"]
    assert pre["passes"] and set(pre["arms"]) == set(eh.ARMS)
    for a in pre["arms"].values():
        assert a["identical"] and a["scope_identical"]
        assert {"inputs", "params", "slice", "forecast", "memory", "output"} <= set(a["by_category"])
        assert all(v["differing"] == 0 and v["values"] > 0 for v in a["by_category"].values())
        assert a["scope"]["rows"] > 0 and a["scope"]["compared_values"] > 0
    assert result["stages"] == ["precheck", "h1"]
    sc = result["scope_every_cut"]
    assert list(sc["systems"]) == [*eh.ARMS, "cheat-stamp"] and sc["identical"]


def test_every_honest_arm_is_invariant_at_every_cut_and_both_purge_cuts(full):
    result, _ = full
    for arm in eh.ARMS:
        s = system(result, arm)
        v = s["verdict"]
        assert v["ok"] and v["violations"] == 0 and v["pattern"] == v["predicted_pattern"] == ".|.." and not v["rule_errors"]
        assert [r["cut"] for r in s["records"]] == ["1", "F-24h", "F-30d"]
        for r in s["records"]:
            assert set(r["by_direction"]) == {"x3", "/3"} and r["compared_values"] > 0 and r["retrained"]
    assert result["h1"]["honest_violations"] == 0 and result["h1"]["honest_flagged"] == []


@pytest.mark.parametrize(("name", "pattern"), [("cheat-stamp", "X|.."), ("cheat-value", "X|.."), ("cheat-stamp-day", "X|.."), ("cheat-purge", "-|XX"), ("cheat-slice", "-|.X")])
def test_each_planted_control_is_caught_exactly_where_predicted_in_both_directions(full, name, pattern):
    result, _ = full
    v = system(result, name)["verdict"]
    assert v["predicted_pattern"] == pattern
    assert v["pattern_by_direction"] == {"x3": pattern, "/3": pattern}
    assert v["ok"] and v["smallest_catch"] > 1e-9
    assert name not in result["h1"]["controls_off_pattern"]
    assert result["cheat_stamp_caught_bit_for_bit"] is True


def test_cheat_stamp_day_reads_only_the_last_24_hours_whose_labels_are_not_yet_known(full, sub):
    _, out = full
    arr, _ = checkpoint.load(out / "parts" / "run.cheat-stamp-day.2019.r1.safetensors")
    a = sub.data.arrays
    row_of = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(a.symbol, a.open_ms, strict=True))}
    start = int(arr["memory:entry_range"][0])
    entries = arr["memory:entries"]
    read = 0
    for (d, _), (lo, hi) in zip(arr["memory:rows"], arr["memory:window"], strict=True):
        e = entries[lo - start : hi - start]
        assert np.all((e[:, 0] > d - 24 * HOUR_MS) & (e[:, 0] <= d))  # (d - 24 h, d], stamped at decision
        known = a.lka_ms[[row_of[(int(s), int(t))] for _, s, t in e]]
        assert np.all(known > d)  # every entry it reads has a label not yet known at d
        read += len(e)
    assert read > 0 and np.array_equal(arr["memory:n"], (arr["memory:window"][:, 1] - arr["memory:window"][:, 0]).astype(arr["memory:n"].dtype))


def test_only_cheat_stamp_day_changes_the_memory_and_the_hook_is_public():
    from olsprime.models.memory import MemoryConfig
    from olsprime.train.fit import EncoderSystem, FitConfig

    day = eh.fit_config(CFG, eh.SPECS["cheat-stamp-day"], 2019)
    assert day.memory == dataclasses.replace(MemoryConfig(), window_ms=24 * HOUR_MS) and day.train_gap_ms == 30 * DAY_MS
    for name in (*eh.ARMS, "cheat-stamp", "cheat-value", "cheat-purge", "cheat-slice", eh.POSITIVE):
        assert eh.fit_config(CFG, eh.SPECS[name], 2019).memory == MemoryConfig() == FitConfig(fit_ms=0, seed=1).memory
    reg = eh.EncoderH1Config()
    honest = eh.fit_config(reg, eh.SPECS["enc-ols"], 2021)
    assert honest == FitConfig(fit_ms=year_start_ms(2021), seed=1, encoder=honest.encoder)  # every honest default untouched
    assert issubclass(eh.SPECS["cheat-stamp-day"].cls, eh.StampedAtDecision)
    # The bench reads the memory through the public read-only hook, not a private method.
    assert "on_readout" in vars(eh.BenchSystem) and "_readout_fixed" not in vars(eh.BenchSystem)
    assert "on_readout" in vars(EncoderSystem)


def test_the_purge_cuts_compare_what_the_protocol_lists(full):
    result, out = full
    stamp = system(result, "cheat-purge")
    scopes = {r["cut"]: r["scope"] for r in stamp["records"]}
    assert scopes == {"F-24h": "purge-24h", "F-30d": "purge-30d"}
    arr, _ = checkpoint.load(out / "parts" / "run.enc-ols.2019.r1.safetensors")
    at24 = eh.compared(arr, "purge-24h")
    at30 = eh.compared(arr, "purge-30d")
    cats24 = {eh.category(k) for k in at24}
    assert cats24 == {"inputs", "params", "slice"}  # no output, no memory
    assert {"inputs:climatology", "inputs:slice_rows", "slice:losses", "slice:kept_epoch"} <= set(at24)
    assert {k for k in at30 if not k.startswith("params:")} == {"inputs:train_rows", "inputs:standardization", "inputs:weeks", "inputs:orders"}
    assert sum(k.startswith("params:epoch2:") for k in at30) == sum(k.startswith("params:epoch1:") for k in at30) > 0  # every epoch
    at_cut = eh.compared(arr, "cut", eh.hour_ms("2019-01-10T13:00") + HOUR_MS - 1)
    assert {"forecast", "memory", "output"} <= {eh.category(k) for k in at_cut}
    assert at_cut["forecast:rows"][:, 0].max() <= eh.hour_ms("2019-01-10T13:00") + HOUR_MS - 1


def test_a_member_without_a_later_bar_is_left_out_and_counted(full):
    result, _ = full
    rec = {r["cut"]: r for r in system(result, "enc-ols")["records"]}
    assert rec["1"]["left_out"] == {"S02USDT": "no bar after the cut"}
    assert rec["F-24h"]["left_out"] == {"S02USDT": "no bar after the cut"} and rec["F-30d"]["left_out"] == {}


def test_the_skill_and_the_positive_control_are_reported_with_report2s_intervals(full):
    result, _ = full
    sk = result["skill"]
    assert set(sk["systems"]) == {*eh.ARMS, *eh.SKILL_CONTROLS}
    for s in sk["systems"].values():
        b = s["bss"]
        assert b["low"] <= b["estimate"] <= b["high"] and b["n_boot"] > 0 and s["rows"] > 0
    assert list(sk["control_check"]) == ["cheat-stamp", "cheat-value", "cheat-stamp-day"] and isinstance(sk["control_check_met"], bool)
    assert sk["control_check_met"] == all(c["met"] for c in sk["control_check"].values())
    pos = sk["positive"]
    assert pos["rows"] > 0 and pos["at_least"] == 0.5 and isinstance(pos["met"], bool)
    # Three verdicts, apart: H1 does not read the skill check, nor pos-window's.
    assert result["h1"]["holds"] == all(s["verdict"]["ok"] for s in result["systems"])
    assert result["verdicts"] == {"h1": result["h1"]["holds"], "skill_check": sk["control_check_met"],
                                  "skill_check_by_control": {n: c["met"] for n, c in sk["control_check"].items()},
                                  "pos_window": pos["met"]}  # fmt: skip


def test_the_report_opens_with_the_precheck_then_cheat_stamp_then_pos_window(full):
    _, out = full
    md = (out / "encoder-h1.md").read_text(encoding="utf-8")
    assert md.index("## 1. The pre-check") < md.index("## 2. `cheat-stamp`") < md.index("## 3. `pos-window`") < md.index("## 4. Three verdicts")
    assert "Its skill: Brier skill score" in md[md.index("## 2. `cheat-stamp`") : md.index("## 3. `pos-window`")]
    lines = md.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith("- **H1: "))
    # H1's line never stands without the skill check's beside it; each prints pass or fail.
    assert lines[at + 1].startswith("- **The skill check: ") and lines[at + 2].startswith("- **`pos-window`'s check: ")
    assert all(any(w in lines[at + i] for w in (": pass.**", ": FAIL.**")) for i in range(3))
    assert all(f"`{n}` Brier skill score" in lines[at + 1] for n in eh.SKILL_CONTROLS)
    assert "NOT THE REGISTERED RUN" in md
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert set(run) == {"started_at", "finished_at", "seconds", "gpu", "driver", "cuda", "torch"}
    text = (out / "encoder-h1.json").read_text(encoding="utf-8")
    assert "finished_at" not in text and "seconds" not in text
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["provenance"]["git_commit"] == PROV["git_commit"]


# --- The compare path ------------------------------------------------------------------------------


def test_a_planted_defect_in_the_compare_path_is_caught(full):
    _, out = full
    arr, _ = checkpoint.load(out / "parts" / "run.enc-ols.2019.r1.safetensors")
    base = eh.compared(arr, "cut", eh.hour_ms("2019-01-10T13:00") + HOUR_MS - 1)
    # one bit of one parameter flipped: an honest system is flagged, though no catch (a rounding-sized move)
    flipped = {k: v.copy() for k, v in base.items()}
    name = next(k for k in flipped if k.startswith("params:epoch2:"))
    flipped[name].view(np.uint64)[3] ^= np.uint64(1)
    rec = eh.compare_record(base, flipped)
    assert rec["violations"] == 1 and rec["caught"] == 0 and rec["items"][name]["differing"] == 1
    assert rec["by_category"]["params"]["differing"] == 1
    honest = eh.verdict(eh.SPECS["enc-ols"], [_record("1", rec, predicted=False)], _cuts(), ["x3", "/3"])
    assert not honest["ok"] and honest["pattern"] == "X-|--"
    # one row of the memory's statistics moved: caught
    moved = {k: v.copy() for k, v in base.items()}
    moved["memory:s_zz"][7] += 1e-6
    rec = eh.compare_record(base, moved)
    assert rec["caught"] == 64 and rec["items"]["memory:s_zz"]["largest"] == pytest.approx(1e-6)
    # a memory entry that disappears changes a shape: caught
    short = dict(base, **{"memory:entries": base["memory:entries"][:-1]})
    assert eh.compare_record(base, short)["caught"] > 0


def _cuts() -> list[eh.H1Cut]:
    return [eh.H1Cut(0, "1", 0, "given-hour", 2019), eh.H1Cut(1, "2", 0, "given-hour", 2019),
            eh.H1Cut(2, "F-24h", 0, "purge-24h", 2019), eh.H1Cut(3, "F-30d", 0, "purge-30d", 2019)]  # fmt: skip


def _record(cut: str, rec: dict, *, predicted: bool) -> dict:
    return {"cut": cut, "predicted": predicted, "violations": rec["violations"], "by_direction": {"x3": rec, "/3": rec}}


def test_a_control_caught_in_one_direction_only_is_off_its_pattern():
    hit = {"violations": 3, "caught": 3, "largest_change": 0.1}
    miss = {"violations": 0, "caught": 0, "largest_change": 0.0}
    rec = {"cut": "1", "predicted": True, "violations": 3, "by_direction": {"x3": hit, "/3": miss}}
    v = eh.verdict(eh.SPECS["cheat-stamp"], [rec], _cuts(), ["x3", "/3"])
    assert not v["ok"] and v["pattern_by_direction"] == {"x3": "X-|--", "/3": ".-|--"}


# --- Resume, determinism and the floor -------------------------------------------------------------


def test_a_run_cut_in_the_middle_and_resumed_gives_the_same_bytes(sub, small, tmp_path):
    out = tmp_path / "resumed"
    with pytest.raises(eh.Interrupted):
        eh.run(sub, SMALL, out, "all", prov=PROV, stop_after=5)
    assert len(list((out / "parts").glob("*.sha256"))) == 5
    eh.run(sub, SMALL, out, "all", prov=PROV)
    for name in ("encoder-h1.json", "encoder-h1.md", "config.json"):
        assert (out / name).read_bytes() == (small / name).read_bytes(), name


SCOPE = dataclasses.replace(SMALL, cuts=("2019-01-05T13:00", "2019-01-10T13:00"))


def _flip_low_bit(x: np.ndarray) -> np.ndarray:
    y = np.ascontiguousarray(x).copy()
    y.reshape(-1).view(np.uint64)[0] ^= np.uint64(1)
    return y


def test_the_scope_test_at_every_cut_is_a_description_and_flags_a_planted_defect(sub, tmp_path, monkeypatch):
    out = tmp_path / "scope"
    r = eh.run(sub, SCOPE, out, "precheck", prov=PROV)
    sc = r["scope_every_cut"]
    assert r["stages"] == ["precheck"] and r["precheck"]["passes"] and r["verdicts"] is None
    assert list(sc["systems"]) == ["enc-ols", "cheat-stamp"] and sc["identical"] and sc["description"]
    for s in sc["systems"].values():
        assert list(s["cuts"]) == ["1", "2"] and s["identical"]
        assert 0 < s["cuts"]["1"]["rows"] < s["cuts"]["2"]["rows"]
        for c in s["cuts"].values():
            assert c["violations"] == 0 and c["compared_values"] > 0 and {"output", "memory"} <= set(c["by_category"])
    md = (out / "encoder-h1.md").read_text(encoding="utf-8")
    assert "## 5. Scope at every cut (description)" in md and "**Identical everywhere.**" in md
    # A planted defect on the restricted forecasts: a memory bit for enc-ols, an output bit for cheat-stamp.
    for p in (out / "parts").glob("scopecut.*"):
        p.unlink()
    real = eh.forecast_arrays

    def tampered(system, res, rows, mem_until):
        arr = real(system, res, rows, mem_until)
        key = "output:probs" if isinstance(system, eh.StampedAtDecision) else "memory:s_yz"
        return arr | {key: _flip_low_bit(arr[key])}

    monkeypatch.setattr(eh, "forecast_arrays", tampered)
    r2 = eh.run(sub, SCOPE, out, "precheck", prov=PROV)
    assert r2["precheck"] == r["precheck"] and r2["precheck"]["passes"]  # the registered verdict is not touched
    sc2 = r2["scope_every_cut"]
    assert not sc2["identical"]
    for name, key in (("enc-ols", "memory:s_yz"), ("cheat-stamp", "output:probs")):
        for c in sc2["systems"][name]["cuts"].values():
            assert c["violations"] == 1 and c["items"][key]["differing"] == 1
    assert "**NOT IDENTICAL somewhere.**" in (out / "encoder-h1.md").read_text(encoding="utf-8")


def test_a_resume_refuses_another_build_or_driver_and_names_it(sub, small, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    (out / "parts").mkdir(parents=True)
    for p in (small / "parts").iterdir():
        (out / "parts" / p.name).write_bytes(p.read_bytes())
    ident = json.loads((out / "parts" / "IDENTITY.json").read_text(encoding="utf-8"))
    assert {"torch", "cuda_build", "device", "settings", "driver"} <= set(ident["build"]) and ident["build"]["driver"] is None
    assert not {"started_at", "finished_at", "seconds"} & set(ident["build"])  # no wall clock
    with pytest.raises(eh.ResumeRefused, match=r"build\.driver"):
        eh.run(sub, SMALL, out, "all", prov=PROV, driver="999.99")
    real = eh.build_info
    monkeypatch.setattr(eh, "build_info", lambda device: {**real(device), "torch": "0.0.0+planted"})
    with pytest.raises(eh.ResumeRefused, match=r"build\.torch") as e:
        eh.run(sub, SMALL, out, "all", prov=PROV)
    assert "build.driver" not in str(e.value) and "config" not in str(e.value)
    monkeypatch.setattr(eh, "build_info", real)
    eh.run(sub, SMALL, out, "all", prov=PROV)  # the same build resumes
    assert (out / "encoder-h1.json").read_bytes() == (small / "encoder-h1.json").read_bytes()


def test_a_resume_verifies_every_part_and_refuses_another_configuration(sub, small, tmp_path):
    out = tmp_path / "copy"
    (out / "parts").mkdir(parents=True)
    for p in (small / "parts").iterdir():
        (out / "parts" / p.name).write_bytes(p.read_bytes())
    part = next((out / "parts").glob("cut.cheat-stamp.*.x3.json"))
    part.write_bytes(part.read_bytes().replace(b'"caught": ', b'"caught": 1', 1))  # a corrupted part: its digest fails
    logs: list[str] = []
    eh.run(sub, SMALL, out, "all", prov=PROV, log=logs.append)
    assert any("fails its digest: recomputed" in s for s in logs)
    assert (out / "encoder-h1.json").read_bytes() == (small / "encoder-h1.json").read_bytes()
    with pytest.raises(eh.ResumeRefused, match="config"):
        eh.run(sub, dataclasses.replace(SMALL, seed=2), out, "all", prov=PROV)
    with pytest.raises(eh.ResumeRefused, match="code_sha256"):
        eh.run(sub, SMALL, out, "all", prov=PROV | {"code": {"sha256": "0" * 64}})


def test_the_floor_mode_on_cpu(sub, small, tmp_path):
    out = tmp_path / "floor"
    (out / "parts").mkdir(parents=True)
    for p in (small / "parts").iterdir():
        (out / "parts" / p.name).write_bytes(p.read_bytes())
    result = eh.run(sub, SMALL, out, "floor", prov=PROV)
    assert result["stages"] == ["precheck", "h1", "floor"]
    fl = result["floor"]
    assert set(fl) == {"enc-ols", "cheat-stamp"}
    assert fl["enc-ols"]["floor_zero"] and set(fl["enc-ols"]["floor_by_category"]) >= {"inputs", "params", "memory", "output"}
    # on CPU the three honest runs agree to the bit: every catch is an infinite multiple of a zero floor
    assert fl["cheat-stamp"]["caught_anywhere"] and math.isinf(fl["cheat-stamp"]["smallest_catch_multiple"])
    assert fl["cheat-stamp"]["weakly_separated"] is False
    assert (out / "parts" / "run.cheat-stamp.2019.r3.safetensors").exists()
    assert "## The floor" in (out / "encoder-h1.md").read_text(encoding="utf-8")


def test_the_floor_is_the_largest_pairwise_difference_and_a_catch_is_a_multiple_of_it():
    a = {"x": np.array([1.0, 2.0]), "y": np.array([0.0])}
    b = {"x": np.array([1.0 + 1e-7, 2.0]), "y": np.array([0.0])}
    c = {"x": np.array([1.0 - 1e-7, 2.0]), "y": np.array([0.0])}
    moves = [eh.item_moves(p, q) for p, q in ((a, b), (a, c), (b, c))]
    assert max(m["x"]["largest"] for m in moves) == pytest.approx(2e-7)
    assert all("y" not in m for m in moves)


# --- Pieces ----------------------------------------------------------------------------------------


def test_the_purge_cuts_are_the_ninth_and_tenth_cuts_of_h1s_schedule(sub):
    cuts = eh.h1_cuts(CFG, sub.data.arrays)
    f = year_start_ms(2019)
    purge = {c.label: c for c in cuts if not c.regular}
    assert purge["F-24h"].hour_ms == f - 24 * HOUR_MS and purge["F-30d"].hour_ms == f - 30 * DAY_MS
    assert [c.k for c in cuts] == [0, 1, 2] and purge["F-24h"].year == purge["F-30d"].year == 2019
    real = [eh.H1Cut(8, "F-24h", 0, "purge-24h", 2021), eh.H1Cut(9, "F-30d", 0, "purge-30d", 2021)]
    seeds = [[perturbation_at(H1Config(), c.k, j) for j in (0, 1)] for c in real]
    assert [[s for s, _ in row] for row in seeds] == [[15, 1015], [16, 1016]]
    assert [row[0][1].mirror for row in seeds] == [False, True]
    assert eh.hour_text(year_start_ms(2021) - 24 * HOUR_MS) == "2020-12-31T00:00"
    assert eh.hour_text(year_start_ms(2021) - 30 * DAY_MS) == "2020-12-02T00:00"


def test_the_registered_cuts_are_iteration_3s_and_are_checked():
    recorded = _recorded()
    assert len(eh.RECORDED_CUTS) == 8 and (not recorded or eh.RECORDED_CUTS == tuple(h[:16] for h in recorded))
    bars, months = eh.synthetic_subpanel(1, start=datetime(2018, 12, 20, tzinfo=UTC), hours=24 * 60)  # rows in 2019-01 and 2019-02
    arrays = panel_arrays(pi.build_panel(bars, months))
    drawn = tuple(eh.hour_text(c.hour_ms) for c in pi.choose_cuts(arrays, pi.PanelH1Config()))
    cuts = eh.h1_cuts(dataclasses.replace(CFG, choose_cuts=True, cuts=drawn), arrays)
    assert len(cuts) == 10 and [c.k for c in cuts if not c.regular] == [8, 9]
    with pytest.raises(ValueError, match="iteration 3"):
        eh.h1_cuts(dataclasses.replace(CFG, choose_cuts=True, cuts=eh.RECORDED_CUTS), arrays)


def _recorded() -> list[str]:
    from olsprime.paths import REPO_ROOT

    path = REPO_ROOT / "runs" / "2026-10-29-h1-panel" / "h1-panel.json"
    if not path.exists():  # runs/ is not in git: CI checks the constant only
        return []
    return [c["hour"] for c in json.loads(path.read_text(encoding="utf-8"))["cuts"]]


def test_the_climatology_is_climatology_panels(sub):
    a = sub.data.arrays
    months = np.unique(a.month_ms)
    mine = eh.monthly_climatology(a, a.label, months)
    matrix = MatrixSystem(a)
    for m, p in zip(months, mine, strict=True):
        assert np.array_equal(p, matrix.climatology_at(int(m))[0])


def test_the_registered_configuration():
    assert eh.EncoderH1Config().registered and eh.EncoderH1Config(device="cpu").registered
    assert not CFG.registered and not SMALL.registered
    assert eh.EncoderH1Config().cuts == eh.RECORDED_CUTS and eh.EncoderH1Config().seed == 1
    with pytest.raises(ValueError):
        eh.EncoderH1Config(arms=("enc-softmax",))
    assert eh.SPECS["cheat-purge"].train_gap_ms == 0 and eh.SPECS["cheat-slice"].train_gap_ms == 24 * HOUR_MS
    assert eh.SPECS["enc-ols"].train_gap_ms == 30 * DAY_MS and eh.SPECS["cheat-value"].value_leak
