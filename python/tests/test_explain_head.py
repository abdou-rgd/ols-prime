"""``explain head`` (``olsprime.explain.head``) on a tiny synthetic Report n°3 run, CPU torch.

The run: Report n°2 on a synthetic panel of five members (``report3.synthetic_report2``, test months 2019-01 and 2019-02),
then Report n°3's fits (the 2019 fit, two seeds, a small encoder of 2 blocks x 2 heads of d_head 8 over T = 64)
and its score stage, in this module's runs root. Nothing reads the repository's ``data/``.

- **Outputs.** One answer per seed; the forecast recomputed to the bit; the row of W, its row sums within their
  bounds; the diagonal mass of a planted identity-like head and its closed form (the hat matrix's trace with a free
  intercept); the effective rank against a hand computation.
- **Refusals** (exit 2): no ``--test-months``, a row before 2019, a part copied from another arm or seed, a part
  re-sealed or truncated after Report n°3 scored it (audit pass 4g), a wrong digest,
  another build (the real CPU thread count changed), another panel.
- **Loud checks** (exit 3, nothing printed): a planted wrong forecast (one ulp, report3.json forged to match), a
  planted row-sum failure, and one below the fp32 bound that only the fp64 check sees.
- **Contacts.** One line per row asked, written before the run folder is opened; none for a call refused before.
- **Determinism.** Two calls give the same bytes.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from olsprime.bench import report2 as r2
from olsprime.bench import report3 as r3
from olsprime.explain import cli
from olsprime.explain import head as hd
from olsprime.models.encoder import ols_weights
from olsprime.models.matrix import MatrixConfig, iso
from olsprime.train import checkpoint

PROV = {"git_commit": "c" * 40, "code": {"sha256": "d" * 64, "files": 1}, "manifest_digest": "0" * 64}
TINY = (("d_model", 16), ("n_heads", 2), ("d_mlp", 32))  # 2 blocks x 2 heads, d_head 8, T = 64
SHORT = (("epochs", 1), ("warmup_steps", 3), ("eval_batch", 256))
R2CFG = r2.Report2Config(matrix=MatrixConfig(first_test=(2019, 1), last_test=(2019, 2)), first_month=(2018, 10), last_month=(2019, 2),
                         readout_sample=20, previewed_through=(2019, 1), n_boot=50)  # fmt: skip
MEMBERS, HOURS = 5, 24 * (30 + 31 + 30 + 31 + 31 + 28 + 4)  # 2018-09-01 -> 2019-03-05: the labels of February's last rows
CFG = r3.Report3Config(years=(2019,), seeds=(1, 2), device="cpu", encoder=TINY, fit=SHORT, synthetic="test: explain head")
ARM, SEED_KEY = "enc-ols", r3.part_key("enc-ols", 1, 2019)


@dataclass(eq=False)
class World:
    runs: Path
    out: Path
    sha: str
    panel: pl.DataFrame
    bars: dict[str, pl.DataFrame]
    pair: str
    as_of: str
    row: int


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    runs = tmp_path_factory.mktemp("runs-head")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OLSPRIME_RUNS_DIR", str(runs))
        panel, bars, series, sha2 = r3.synthetic_report2(runs / "report2", members=MEMBERS, hours=HOURS, r2cfg=R2CFG)
        inputs = r3.build_inputs(r3.read_report2(series, sha2), panel, bars, CFG)
        out = runs / "report3"
        r3.run(inputs, CFG, out, "fits", prov=PROV)
        r3.run(inputs, CFG, out, "score", prov=PROV)
    data = inputs.data
    rows = inputs.years[2019]
    usable = rows[data.usable[rows]]
    row = int(usable[(usable.size * 2) // 3])  # inside a middle batch of 256
    a = data.arrays
    return World(runs, out, r2.sha256_file(out / r3.SERIES_FILE), panel, bars, a.symbols[a.symbol[row]], iso(int(a.open_ms[row])), row)


def call(w: World, *extra: str, contacts: Path, run: Path | None = None, sha: str | None = None, as_of: str | None = None,
         flag: bool = True, panel=None) -> tuple[int, str, list[dict], str]:  # fmt: skip
    argv = ["head", w.pair, as_of or w.as_of, "--run", str(run or w.out), "--sha256", sha or w.sha, *(["--test-months"] if flag else []), *extra]
    out, err = io.StringIO(), io.StringIO()
    source = panel or (lambda: (w.panel, w.bars))
    code = cli.main(argv, contacts=contacts, out=out, err=err, panel_source=source)
    text = out.getvalue()
    lines = [json.loads(x) for x in text.split(cli.JSON_MARK + "\n")[-1].splitlines() if x.strip()] if cli.JSON_MARK in text else []
    return code, text, lines, err.getvalue()


def _ms(text: str) -> int:
    return int(np.datetime64(text[:23], "ms").astype(np.int64))


def _contacts(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _copy_run(w: World, dst: Path) -> Path:
    for p in sorted(w.out.rglob("*")):
        if p.is_file():
            (dst / p.relative_to(w.out)).parent.mkdir(parents=True, exist_ok=True)
            (dst / p.relative_to(w.out)).write_bytes(p.read_bytes())
    return dst


@pytest.fixture(scope="module")
def answer(world, tmp_path_factory) -> tuple[list[dict], str, Path]:
    contacts = tmp_path_factory.mktemp("contacts-answer") / "CONTACTS.log"
    code, text, lines, err = call(world, contacts=contacts)
    assert code == 0, err
    return lines, text, contacts


# --- Outputs ------------------------------------------------------------------------------------


def test_each_seed_is_given_apart_and_its_forecast_is_recomputed_to_the_bit(world, answer):
    lines, text, contacts = answer
    assert len(_contacts(contacts)) == 1  # one row asked, two seeds: one contact line
    assert [r["seed"] for r in lines] == [1, 2] and all(r["kind"] == "head" and r["arm"] == ARM for r in lines)
    parts = r3.Parts(world.out, None)
    for r in lines:
        key = r3.part_key(ARM, r["seed"], 2019)
        assert r["fit"]["part"] == key and r["fit"]["year"] == 2019 and r["fit"]["device"] == "cpu"
        fc = r3.read_forecast(parts, key)
        (j,) = np.flatnonzero((fc["open_time"] == _ms(r["row"]["open_time"])) & (np.asarray(fc["symbols"])[fc["symbol"]] == world.pair))
        saved = fc["raw"][j]
        assert np.array_equal(np.asarray(r["forecast"]["raw"]).view(np.uint64), saved.view(np.uint64))
        assert r["forecast"]["equal_to_the_bit"] is True and r["forecast"]["how"]["eval_batch"] == 256
        assert r["forecast"]["how"]["batches"] > 1 and r["forecast"]["how"]["batch_rows"] == 256
        assert len(r["heads"]) == 4 and [(h["block"], h["head"]) for h in r["heads"]] == [(1, 1), (1, 2), (2, 1), (2, 2)]
        bars = r["row"]["window"]["bars"]
        assert len(bars) == 64 and bars[-1] == world.as_of
        step = np.diff([_ms(b) for b in bars])
        assert np.all(step == 3_600_000)
        assert r["exploratory"] and r["provenance"]["panel_fingerprint_verified"] is True
    assert "equal to the saved forecast to the bit" in text and "EXPLORATORY" in text
    assert lines[0]["heads"][0]["row"]["weights"] != lines[1]["heads"][0]["row"]["weights"]  # two seeds, two models


def test_every_row_sums_to_one_within_its_bound(answer):
    lines, _, _ = answer
    for r in lines:
        assert r["checks"]["row_sums"]["passed"] and r["checks"]["row_sums"]["rows_checked"] == 4 * 64
        assert 0 <= r["checks"]["row_sums"]["max_dev_over_bound"] < 1
        for h in r["heads"]:
            weights = h["row"]["weights"]
            assert len(weights) == 64
            assert abs(math.fsum(weights) - h["row_sum"]["decision_token"]) < 1e-12
            assert abs(h["row_sum"]["decision_token"] - 1) <= h["row_sum"]["decision_token_bound"]
            assert h["row_sum"]["max_dev"] <= h["row_sum"]["bound_at_max_dev"]
            x = h["row_sum"]["fp64"]  # the tight check: fp64 W, the same bound taken in fp64
            assert x["max_dev"] <= x["bound_at_max_dev"] and x["max_dev_over_bound"] < 1
            assert x["decision_token_bound"] < 1e-6 * h["row_sum"]["decision_token_bound"]
            assert h["lambda"] > 1e-3 and h["condition"] >= 1
            assert h["self_weight"] == weights[-1]
            assert 0 <= h["w_fp64_gap"] < 1e-4 and h["wv_gap"]["max_abs"] < 1e-4 * max(1.0, h["wv_gap"]["output_scale"])
            assert h["row"]["negative_mass"] <= 0 <= h["row"]["sum_abs"]


def test_the_effective_rank_against_a_hand_computation(answer):
    rng = np.random.default_rng(7)
    u, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    v, _ = np.linalg.qr(rng.normal(size=(64, 64)))
    s = np.zeros(64)
    s[:3] = (3.0, 2.0, 1.0)
    w = u @ np.diag(s) @ v.T
    by_hand = math.exp(0.5 * math.log(2) + math.log(3) / 3 + math.log(6) / 6)  # p = (1/2, 1/3, 1/6)
    st = hd.head_stats(w, 8)
    assert abs(st["effective_rank"] - by_hand) < 1e-12
    np.testing.assert_allclose(st["singular_values"][:3], [3, 2, 1], rtol=0, atol=1e-12)
    assert st["singular_tail_share"] < 1e-15 and st["rank_cap"] == 9
    assert hd.effective_rank(np.array([5.0, 0.0, 0.0])) == 1.0
    assert abs(hd.effective_rank(np.ones(9)) - 9) < 1e-12
    for r in answer[0]:
        for h in r["heads"]:
            assert 1 <= h["effective_rank"] <= 9 * (1 + 1e-4)  # rank <= d_head + 1 = 9; the tail is rounding
            assert h["singular_tail_share"] < 1e-5
            assert abs(h["effective_rank"] - hd.effective_rank(np.asarray(h["singular_values"]))) <= 1e-4 * h["effective_rank"]


def test_the_diagonal_mass_of_a_planted_identity_like_head_and_its_closed_form():
    g = torch.Generator().manual_seed(3)
    # identity-like: T = d_head + 1 tokens, queries equal to keys, lambda tiny: K~ is square and W = K~ K~^-1 = I
    k = torch.randn(1, 1, 9, 8, generator=g, dtype=torch.float64)
    w = ols_weights(k, k, torch.tensor([1e-9], dtype=torch.float64))[0, 0].numpy()
    st = hd.head_stats(w, 8)
    assert st["diagonal_mass"] > 1 - 1e-6 and st["self_weight"] > 1 - 1e-6 and abs(st["effective_rank"] - 9) < 1e-4
    # T = 64, queries equal to keys: W is the ridge hat matrix with a free intercept, trace = 1 + sum s^2 / (s^2 + T lambda)
    k = torch.randn(1, 1, 64, 8, generator=g, dtype=torch.float64)
    lam = 0.1
    w = ols_weights(k, k, torch.tensor([lam], dtype=torch.float64))[0, 0].numpy()
    kn = k[0, 0].numpy()
    kn = kn / np.sqrt((kn**2).mean(axis=1, keepdims=True) + 1e-12)
    s = np.linalg.svd(kn - kn.mean(axis=0), compute_uv=False)
    closed = 1 + np.sum(s**2 / (s**2 + 64 * lam))
    assert abs(hd.head_stats(w, 8)["diagonal_mass"] * 64 - closed) < 1e-10
    assert hd.head_stats(w, 8)["diagonal_mass"] < 9 / 64 + 1e-12  # a hat matrix of rank 9 cannot copy its input


# --- Refusals -----------------------------------------------------------------------------------


def test_refused_without_the_flag_before_2019_or_with_a_wrong_digest_and_nothing_is_logged(world, tmp_path):
    contacts = tmp_path / "CONTACTS.log"
    code, text, _, err = call(world, contacts=contacts, flag=False)
    assert code == 2 and "--test-months" in err and text == ""
    code, _, _, err = call(world, contacts=contacts, as_of="2018-12-20T10:00")
    assert code == 2 and "before 2019-01-01" in err
    code, _, _, err = call(world, contacts=contacts, sha="0" * 64)
    assert code == 2 and "SHA-256" in err
    code, _, _, err = call(world, contacts=contacts, as_of="2025-01-02T00:00")
    assert code == 2 and "out of explain's reach" in err
    assert not contacts.exists()


@pytest.mark.parametrize(("src", "what"), [(("enc-softmax", 1), "part.json: arm"), (("enc-ols", 2), "part.json: seed")])
def test_a_part_copied_from_another_arm_or_seed_is_refused(world, tmp_path, src, what):
    run = _copy_run(world, tmp_path / "copied")
    parts = r3.Parts(run, None)
    for name in ("fit.safetensors", "forecast.safetensors", "part.json", "timing.json"):
        parts.put(f"{SEED_KEY}/{name}", parts.get(f"{r3.part_key(*src, 2019)}/{name}"))
    contacts = tmp_path / "CONTACTS.log"
    code, text, _, err = call(world, "--seed", "1", contacts=contacts, run=run)
    assert code == 2 and what in err and text == ""
    assert len(_contacts(contacts)) == 1  # refused after the run folder was opened: the contact is logged
    assert call(world, "--seed", "2", contacts=contacts, run=run)[0] == 0  # the other seed's own part still answers


def _reseal_fit(parts: r3.Parts, key: str, name: str) -> None:
    """One ulp in one weight of the checkpoint, every digest that binds it re-sealed (audit pass 4g's probe A)."""
    arrays, meta = checkpoint.from_bytes(parts.get(f"{key}/fit.safetensors"))
    flat = arrays["param." + name].reshape(-1)
    flat[0] = np.nextafter(flat[0], flat.dtype.type(np.inf))
    fit_sha = parts.put(f"{key}/fit.safetensors", checkpoint.to_bytes(arrays, meta))
    fa, fm = checkpoint.from_bytes(parts.get(f"{key}/forecast.safetensors"))
    fc_sha = parts.put(f"{key}/forecast.safetensors", checkpoint.to_bytes(fa, {**fm, "fit_sha256": fit_sha}))
    parts.put_json(f"{key}/part.json", {**parts.get_json(f"{key}/part.json"), "fit_sha256": fit_sha, "forecast_sha256": fc_sha})


@pytest.mark.parametrize("name", ["key.weight", "encoder.position", "encoder.ln_f.weight", "encoder.blocks.1.mlp.2.weight"])
def test_a_resealed_checkpoint_is_refused_by_report3s_record(world, tmp_path, name):
    """audit pass 4g, major: a part re-sealed after Report n°3 scored it verifies against itself; report3.json refuses it."""
    run = _copy_run(world, tmp_path / "resealed")
    parts = r3.Parts(run, None)
    _reseal_fit(parts, SEED_KEY, name)
    assert r3.verified_part(parts, SEED_KEY, CFG) is not None  # the folder alone accepts it
    code, text, _, err = call(world, "--seed", "1", contacts=tmp_path / "CONTACTS.log", run=run)
    assert code == 2 and "fit_sha256, forecast_sha256" in err and "re-sealed" in err and text == ""


def test_a_truncated_and_resealed_forecast_is_refused(world, tmp_path):
    run = _copy_run(world, tmp_path / "truncated")
    parts = r3.Parts(run, None)
    fa, fm = checkpoint.from_bytes(parts.get(f"{SEED_KEY}/forecast.safetensors"))
    sha = parts.put(f"{SEED_KEY}/forecast.safetensors", checkpoint.to_bytes({k: v[:-10] for k, v in fa.items()}, fm))
    parts.put_json(f"{SEED_KEY}/part.json", {**parts.get_json(f"{SEED_KEY}/part.json"), "forecast_sha256": sha})
    code, _, _, err = call(world, "--seed", "1", contacts=tmp_path / "CONTACTS.log", run=run)
    assert code == 2 and "forecast_sha256" in err
    # report3.json forged to match the digest: the row count it records still refuses the shorter forecast
    _rebind(run, SEED_KEY, forecast_sha256=sha)
    code, _, _, err = call(world, "--seed", "1", contacts=tmp_path / "CONTACTS.log", run=run)
    assert code == 2 and "rows" in err and "records" in err


def _rebind(run: Path, key: str, **fields) -> None:
    """Forge report3.json's entry for ``key`` (its series record unchanged, so the call still opens the run)."""
    arm, seed, year = r3.parse_key(key)
    report = json.loads((run / "report3.json").read_bytes())
    report["fits"] = [{**f, **fields} if (f["arm"], f["seed"], f["year"]) == (arm, seed, year) else f for f in report["fits"]]
    (run / "report3.json").write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def test_another_build_or_another_panel_is_refused(world, tmp_path):
    contacts = tmp_path / "CONTACTS.log"
    code, _, _, err = call(world, "--seed", "1", contacts=contacts, panel=lambda: (world.panel.slice(1), world.bars))
    assert code == 2 and "fingerprint" in err
    threads = torch.get_num_threads()
    try:  # the real path: the fit recorded its CPU threads, this process now has another count
        torch.set_num_threads(threads + 1 if threads == 1 else threads - 1)
        code, _, _, err = call(world, "--seed", "1", contacts=contacts)
    finally:
        torch.set_num_threads(threads)
    assert code == 2 and "build is not the fit's" in err and f"cpu_threads {threads} at the fit" in err
    assert call(world, "--seed", "1", contacts=contacts)[0] == 0


def test_a_row_that_is_not_scored_is_refused(world, tmp_path):
    contacts = tmp_path / "CONTACTS.log"
    code, _, _, err = call(world, contacts=contacts, as_of=world.as_of[:14] + "30")  # no bar opens at half past
    assert code == 2 and "no row" in err


# --- Loud checks --------------------------------------------------------------------------------


def test_a_planted_wrong_forecast_fails_loudly(world, tmp_path):
    run = _copy_run(world, tmp_path / "planted")
    parts = r3.Parts(run, None)
    arrays, meta = checkpoint.from_bytes(parts.get(f"{SEED_KEY}/forecast.safetensors"))
    (j,) = np.flatnonzero((arrays["open_time"] == _ms(world.as_of)) & (np.asarray(json.loads(meta["symbols"]))[arrays["symbol"]] == world.pair))
    arrays["raw"][j, 0] = np.nextafter(arrays["raw"][j, 0], 1.0)  # one ulp
    sha = parts.put(f"{SEED_KEY}/forecast.safetensors", checkpoint.to_bytes(arrays, meta))
    parts.put_json(f"{SEED_KEY}/part.json", {**parts.get_json(f"{SEED_KEY}/part.json"), "forecast_sha256": sha})
    assert r3.verified_part(parts, SEED_KEY, CFG) is not None  # the part itself verifies
    assert call(world, "--seed", "1", contacts=tmp_path / "CONTACTS.log", run=run)[0] == 2  # report3.json's record refuses it
    _rebind(run, SEED_KEY, forecast_sha256=sha)  # report3.json forged too: only the model check can see it now
    code, text, lines, err = call(world, "--seed", "1", contacts=tmp_path / "CONTACTS.log", run=run)
    assert code == 3 and "LOUD CHECK FAILED" in err and "not the saved forecast" in err and "1 ulps" in err
    assert text == "" and lines == []


def test_a_planted_row_sum_failure_fails_loudly(world, tmp_path, monkeypatch):
    real = hd.ols_weights

    def planted(q, k, lam):
        w = real(q, k, lam).clone()
        w[..., 1, 5, 7] += 0.5  # head 2, row 6: no longer sums to 1
        return w

    monkeypatch.setattr(hd, "ols_weights", planted)
    code, text, _, err = call(world, "--seed", "2", contacts=tmp_path / "CONTACTS.log")
    assert code == 3 and "do not sum to 1" in err and "head 2 row 6" in err and text == ""


def test_a_row_sum_error_below_the_fp32_bound_is_caught_by_the_fp64_check(world, tmp_path, monkeypatch):
    """audit pass 4g, minor: the registered fp32 bound is loose; W formed in fp64 is held to the same bound taken in fp64."""
    real = hd.ols_weights

    def planted(q, k, lam):
        w = real(q, k, lam)
        if w.dtype == torch.float64:  # the fp32 W, its check and the forecast are untouched
            w = w.clone()
            w[..., 1, 5, 7] += 1e-9
        return w

    monkeypatch.setattr(hd, "ols_weights", planted)
    code, text, _, err = call(world, "--seed", "2", contacts=tmp_path / "CONTACTS.log")
    assert code == 3 and "head 2 row 6 (fp64 W)" in err and text == ""
    assert "row 6:" not in err  # the fp32 check alone passed: 1e-9 is far below its bound


# --- Contacts and determinism -------------------------------------------------------------------


def test_one_contact_line_per_row_written_before_the_run_folder_is_opened(world, tmp_path, monkeypatch):
    contacts = tmp_path / "CONTACTS.log"
    seen = []
    real = hd.open_head_run

    def spy(run_dir, sha256):
        seen.append(len(_contacts(contacts)))
        return real(run_dir, sha256)

    monkeypatch.setattr(hd, "open_head_run", spy)
    code, _, lines, err = call(world, contacts=contacts)
    assert code == 0, err
    assert seen == [1] and len(lines) == 2  # two seeds, one row: one line, written before the folder was opened
    (line,) = _contacts(contacts)
    assert line["subcommand"] == "head" and line["arm"] == "enc-ols every seed" and line["series_sha256"] == world.sha
    assert line["period"] == f"as_of {iso(_ms(world.as_of))}; the forecasts and labels of fit year 2019; memory through as_of"
    call(world, "--seed", "2", contacts=contacts)
    assert len(_contacts(contacts)) == 2 and _contacts(contacts)[1]["arm"] == "enc-ols seed 2"


def test_two_calls_give_the_same_bytes(world, answer, tmp_path):
    _, text, contacts = answer
    code, again, _, _ = call(world, contacts=contacts)
    assert code == 0 and again == text
    files = []
    for name in ("a.jsonl", "b.jsonl"):
        code, _, _, err = call(world, "--jsonl", str(tmp_path / name), contacts=contacts)
        assert code == 0, err
        files.append((tmp_path / name).read_bytes())
    assert files[0] == files[1] and files[0] == text.split(cli.JSON_MARK + "\n")[-1].encode("utf-8")


def test_the_fit_years_logged_for_an_as_of():
    """A bar's close at 2020-01-01T00:00 ends a 2019 bar: both years are named; any other instant names one."""
    assert hd.logged_fit_years(_ms("2020-01-01T00:00:00")) == (2019, 2020)
    assert hd.logged_fit_years(_ms("2020-01-01T01:00:00")) == (2020,)
    assert hd.logged_fit_years(_ms("2019-12-31T23:00:00")) == (2019,)
