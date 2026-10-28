"""Report n°1 on synthetic bars: byte-identical re-runs, a check that sees any change, the fixed variant list."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from olsprime.bench import report1 as r1
from olsprime.bench.synthetic import synthetic_bars
from olsprime.models.baselines import BASELINE_NAMES

CFG = r1.Report1Config(test_start=(2018, 3), test_end=(2018, 6), random_seeds=4, n_boot=200)
FAKE_MANIFEST = {
    "digest": "0" * 64,
    "vault_open": False,
    "totals": {"files": 2},
    "files": [
        {"path": f"parquet/spot/klines/1d/{p}.parquet", "kind": "klines", "symbol": p, "interval": "1d",
         "sha256": "f" * 64, "stored_rows": 560, "sealed_rows": 560,
         "first_sealed": "2017-01-01T00:00:00.000+00:00", "last_sealed": "2018-07-13T00:00:00.000+00:00"}
        for p in CFG.pairs
    ],
}  # fmt: skip
PROVENANCE = {"git_commit": "c" * 40, "code": {"sha256": "d" * 64, "files": 1}, "manifest_digest": "0" * 64}


def _run(out: Path):
    done = [r1.run_pair(p, synthetic_bars(p, days=560, seed=i), CFG) for i, p in enumerate(CFG.pairs)]
    return r1.write_run(CFG, out, done, FAKE_MANIFEST, PROVENANCE, started=0.0, workers=1)


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("run-a"), tmp_path_factory.mktemp("run-b")
    return a, b, _run(a), _run(b)


def test_two_runs_are_byte_identical(two_runs):
    a, b, _, _ = two_runs
    files = r1.compared_files(a)
    assert {"config.json", "manifest.json", "metrics.json", r1.REPORT_FILE} <= set(files)
    assert len([f for f in files if f.startswith("equity/")]) == len(CFG.pairs) * len(CFG.venues)
    for name in files:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert r1.diff_runs(a, b) == []


def test_wall_clock_stays_out_of_the_compared_files(two_runs):
    a, _, _, _ = two_runs
    assert "run.json" not in r1.compared_files(a)
    assert "finished_at" in json.loads((a / "run.json").read_text())
    for name in r1.compared_files(a):
        assert "finished_at" not in (a / name).read_text(encoding="utf-8")
    assert "generated_at" not in json.loads((a / "manifest.json").read_text())


def test_the_check_sees_a_changed_byte(two_runs, tmp_path):
    a, _, _, _ = two_runs
    copy = tmp_path / "copy"
    for name in r1.compared_files(a):
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        (copy / name).write_bytes((a / name).read_bytes())
    assert r1.diff_runs(a, copy) == []
    metrics = (copy / "metrics.json").read_text(encoding="utf-8")
    (copy / "metrics.json").write_text(metrics.replace('"rows"', '"rowz"', 1), encoding="utf-8")
    (copy / "equity" / f"BTCUSDT-{CFG.venues[0]}.csv").unlink()
    problems = r1.diff_runs(a, copy)
    assert any(p.startswith("metrics.json: differs") and "rowz" in p for p in problems)
    assert any("only in the expected run" in p for p in problems)


def test_every_strategy_venue_and_pair_is_scored(two_runs):
    _, _, metrics, _ = two_runs
    for pair in CFG.pairs:
        venues = metrics["pairs"][pair]["venues"]
        assert set(venues) == set(CFG.venues)  # metrics.json sorts its keys
        for venue in CFG.venues:
            assert set(venues[venue]["strategies"]) == set(BASELINE_NAMES)
            assert set(venues[venue]["h9"]) == set(r1.ACTIVE)
            rt = venues[venue]["strategies"][r1.RANDOM]
            assert rt["distribution"]["seeds"] == CFG.random_seeds
            dsr = venues[venue]["dsr"]
            assert "flat" not in dsr["variants"] and r1.CHEATER not in dsr["variants"]
            assert dsr["n_trials"] == len(dsr["variants"])
        assert set(metrics["pairs"][pair]["forecasts"]["models"]) == set(r1.FORECASTERS)
        assert metrics["pairs"][pair]["ridge_crosscheck"]["worst_coef_max_abs_diff"] < 1e-9


def test_the_variant_list_comes_from_code():
    table = r1.factories("1d", synthetic_bars(days=3), 2)
    assert tuple(table) == BASELINE_NAMES
    assert len(table[r1.RANDOM]) == 2
    assert r1.ACTIVE == ("ma-momentum-10", "ma-momentum-30", "ma-momentum-90", "random-turnover", "climatology", "ridge", "xgboost")


def test_h9_verdicts_follow_the_registered_rule(two_runs):
    _, _, metrics, _ = two_runs
    for venue, verdict in metrics["h9"].items():
        assert verdict["registered"] == (venue in r1.H9_VENUES)
        above = [
            n
            for p in CFG.pairs
            for n, d in metrics["pairs"][p]["venues"][venue]["h9"].items()
            if d["defined"] and d["ci"][0] > 0
        ]
        assert verdict["holds"] == (not above)


def test_the_report_keeps_the_cheater_apart_and_cites_its_inputs(two_runs):
    a, _, _, _ = two_runs
    text = (a / r1.REPORT_FILE).read_text(encoding="utf-8")
    assert "Models are per pair" in text
    assert "H1 control, not a candidate" in text
    assert text.count("cheater (LEAKS ON PURPOSE)") == len(CFG.pairs) * len(CFG.venues)
    assert FAKE_MANIFEST["digest"] in text and PROVENANCE["git_commit"] in text
    for f in FAKE_MANIFEST["files"]:
        assert f"`{f['path']}` | `{f['sha256']}` | {f['stored_rows']} | {f['sealed_rows']}" in text
    assert "descriptive" in text and "registered" in text


def test_git_commit_reads_loose_and_packed_refs(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert r1.git_commit(tmp_path) == "a" * 40
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack-refs\n" + "b" * 40 + " refs/heads/main\n")
    assert r1.git_commit(tmp_path) == "b" * 40
    (git / "HEAD").write_text("c" * 40 + "\n")  # detached
    assert r1.git_commit(tmp_path) == "c" * 40


# --- Audit A fixes -------------------------------------------------------------------------


def test_the_report_counts_comparisons_and_shows_fills_and_sensitivities(two_runs):
    a, _, metrics, _ = two_runs
    text = (a / r1.REPORT_FILE).read_text(encoding="utf-8")
    for venue, h in metrics["h9"].items():
        assert h["comparisons"] == h["with_interval"] + h["undefined"]
        assert f"{h['comparisons']}: {h['with_interval']} with an interval, {h['undefined']} undefined (never exposed)" in text
    assert "| exposure | fills |" in text
    assert "sensitivity block" in text and "DM NW lags" in text
    assert "on the same machine and library versions" in text
    assert "not of their squares" in text
    assert "The block-length sensitivity changes the verdict at:" in text
    assert "2^50 lots" in text


def test_newey_west_lags_and_statistic():
    import numpy as np

    assert r1.newey_west_lags(2190) == 7
    assert r1.newey_west_lags(100) == 4
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    out = r1.dm_newey_west(a, b)
    d = a - b
    c = d - d.mean()
    lags = out["lags"]
    lrv = c @ c / d.size + 2 * sum((1 - k / (lags + 1)) * (c[k:] @ c[:-k]) / d.size for k in range(1, lags + 1))
    assert out["statistic"] == pytest.approx(d.mean() / np.sqrt(lrv / d.size), rel=1e-12)
    assert 0.0 < out["p_value"] <= 1.0


def test_the_sensitivity_block_sees_volatility_clustering():
    import numpy as np

    from olsprime.eval import stats

    rng = np.random.default_rng(1)
    n = 2000
    sigma = np.exp(np.convolve(rng.normal(0, 0.5, n), np.ones(50) / np.sqrt(50), mode="same"))  # slow volatility regimes
    r = rng.normal(0, 0.01, n) * sigma
    bh = rng.normal(0, 0.01, n)
    block = r1.sensitivity_block(r, bh)
    d = r - bh
    assert block >= stats.optimal_block_length(d)
    assert block > max(stats.optimal_block_length(r), stats.optimal_block_length(bh))  # the squares need longer blocks


def test_data_end_comes_from_the_config_and_never_passes_development():
    from dataclasses import replace
    from datetime import UTC, datetime

    assert r1.data_end(CFG) == datetime(2025, 1, 1, tzinfo=UTC)
    assert r1.data_end(replace(CFG, data_end="2024-06-01T00:00:00+00:00")) == datetime(2024, 6, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="development"):
        r1.data_end(replace(CFG, data_end="2025-06-01T00:00:00+00:00"))
    with pytest.raises(ValueError, match="UTC"):
        r1.data_end(replace(CFG, data_end="2024-06-01T00:00:00"))


def test_the_lot_cap_date_is_the_first_bar_at_the_cap():
    import numpy as np
    import polars as pl

    step = r1.pp.SYMBOL_FILTERS["BTCUSDT"].step_size
    cap = r1.MAX_LOTS * step

    class Res:
        position = np.array([0.0, 1.0, cap / 2, cap, cap])

    scored = pl.DataFrame({"open_time": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 5), "1d", eager=True, time_zone="UTC")})
    assert r1.lot_cap_first_binds(Res(), scored, "BTCUSDT") == "2024-01-04"
    Res.position = np.zeros(5)
    assert r1.lot_cap_first_binds(Res(), scored, "BTCUSDT") is None
