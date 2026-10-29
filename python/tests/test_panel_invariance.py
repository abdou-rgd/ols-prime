"""The panel H1 (``olsprime.bench.panel_invariance``) on synthetic bars.

- perturbing the future leaves every output and fit-time choice at or before the cut unchanged,
  bit for bit, for every arm, at every cut and in both directions;
- each planted online leak is caught at its predicted cuts and nowhere else;
- the cuts are seeded, with one at a month's first hour and one at its last;
- a member with no bar after a cut is left out of that cut and counted;
- a cut in a month with no row decided by it flags nothing, and a report that is not the
  registered run says so (audit pass 1c).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from olsprime.bench import panel_invariance as pi
from olsprime.bench.synthetic import synthetic_bars
from olsprime.models.matrix import HOUR_MS, MatrixConfig, MatrixSystem

START = datetime(2018, 9, 1, tzinfo=UTC)
MONTHS = [datetime(y, m, 1, tzinfo=UTC) for y, m in [(2018, 10), (2018, 11), (2018, 12), (2019, 1), (2019, 2)]]
CFG = pi.PanelH1Config(matrix=MatrixConfig(first_test=(2019, 1), last_test=(2019, 2), cusum_threshold=1.0))
HOURS = {"AAAUSDT": 24 * 200, "BBBUSDT": 24 * 200, "CCCUSDT": 24 * 160}  # CCC's last bar opens 2019-02-07 23:00


def subpanel() -> tuple[dict, dict]:
    bars = {s: synthetic_bars(s, start=START, days=n, interval="1h", seed=i, vol=0.004, drift=0.0) for i, (s, n) in enumerate(HOURS.items())}
    return bars, {s: MONTHS for s in bars}


@pytest.fixture(scope="module")
def h1() -> dict:
    bars, months = subpanel()
    return pi.run_panel_h1(bars, months, CFG)


def test_every_arm_is_invariant_bit_for_bit_at_every_cut(h1: dict) -> None:
    honest = next(s for s in h1["systems"] if s["honest"])
    assert set(honest["arms"]) == set(pi.ARMS)
    assert honest["verdict"]["ok"] and honest["verdict"]["violations"] == 0
    for r in honest["records"]:
        assert set(r["by_direction"]) == {"x3", "/3"}
        assert r["compared_values"] > 0
    assert h1["h1"]["holds"] and h1["h1"]["honest_violations"] == 0


@pytest.mark.parametrize("name", [s.name for s in pi.LEAKS])
def test_each_planted_leak_is_caught_exactly_where_predicted(h1: dict, name: str) -> None:
    s = next(x for x in h1["systems"] if x["name"] == name)
    v = s["verdict"]
    assert v["ok"], (v["pattern"], v["predicted_pattern"])
    assert "X" in v["predicted_pattern"]
    assert v["smallest_catch"] is None or v["smallest_catch"] > 1e-9


def test_the_predictions_have_their_registered_shape(h1: dict) -> None:
    kinds = [c["kind"] for c in h1["cuts"]]
    by = {s["name"]: s["verdict"]["predicted_pattern"] for s in h1["systems"] if not s["honest"]}
    last = kinds.index(pi.LAST_HOUR)
    first = kinds.index(pi.FIRST_HOUR)
    sigma = by[pi.MonthSigma.name]
    assert sigma[last] == "." and sigma.count("X") == len(kinds) - 1  # 7 of 8: never at a month's last hour
    unpurged = by[pi.Unpurged.name]
    assert unpurged[first] == "X" and unpurged[last] == "."
    for leak in (pi.StampedAtDecision, pi.WindowPastDecision, pi.KernelPastDecision):
        assert by[leak.name] == "X" * len(kinds)


def test_the_cuts_are_seeded_with_a_first_and_a_last_hour() -> None:
    bars, months = subpanel()
    data = MatrixSystem(pi.build_panel(bars, months), CFG.matrix).data
    a, b = pi.choose_cuts(data, CFG), pi.choose_cuts(data, CFG)
    assert a == b and len(a) == 8
    kinds = [c.kind for c in a]
    assert kinds.count(pi.FIRST_HOUR) == 1 and kinds.count(pi.LAST_HOUR) == 1
    first = next(c for c in a if c.kind == pi.FIRST_HOUR)
    last = next(c for c in a if c.kind == pi.LAST_HOUR)
    assert first.hour_ms == first.month_ms
    assert datetime.fromtimestamp((last.hour_ms + HOUR_MS) / 1000, tz=UTC).day == 1
    assert [c.hour_ms for c in a] == sorted(c.hour_ms for c in a)
    other = pi.choose_cuts(data, pi.PanelH1Config(matrix=CFG.matrix, cut_seed=1))
    assert [c.hour_ms for c in other] != [c.hour_ms for c in a]


def test_a_member_without_a_later_bar_is_left_out_and_counted(h1: dict) -> None:
    last_bar = START.timestamp() * 1000 + (HOURS["CCCUSDT"] - 1) * HOUR_MS
    for c in h1["cuts"]:
        hour = datetime.fromisoformat(c["hour"]).timestamp() * 1000
        left = h1["left_out"][str(c["cut"])]
        assert left == ({"CCCUSDT": "no bar after the cut"} if hour >= last_bar else {})


def test_a_cut_in_a_month_with_no_row_decided_by_it_flags_nothing() -> None:
    """Probe D of audit pass 1c: every member's bar at a month's first hour dropped, a first-hour cut there."""
    bars, months = subpanel()
    hole = datetime(2019, 2, 1, tzinfo=UTC)
    bars = {s: b.filter(pl.col("open_time") != hole) for s, b in bars.items()}
    cut = pi.PanelCut(int(hole.timestamp() * 1000), pi.FIRST_HOUR)
    data = MatrixSystem(pi.build_panel(bars, months), CFG.matrix).data
    assert np.any(data.complete & (data.month_ms == cut.month_ms))  # the original run fits February
    assert not pi.fit_compared(data, cut, cut.month_ms)  # the run scoped to the cut does not
    for spec in (pi.HONEST, *(s for s in pi.LEAKS if s.cls in (pi.MonthSigma, pi.Unpurged))):
        out = pi.run_system(spec, bars, months, [cut], CFG)
        assert out["verdict"]["ok"], (spec.name, out["verdict"])
        assert out["verdict"]["pattern"] == out["verdict"]["predicted_pattern"] == "."


def test_a_report_that_is_not_the_registered_run_says_so(h1: dict) -> None:
    assert pi.PanelH1Config().registered and not CFG.registered
    assert h1["registered"] is False
    clean = pi._clean(h1)
    assert "NOT THE REGISTERED RUN" in pi.render_md(clean)
    assert "NOT THE REGISTERED RUN" not in pi.render_md({**clean, "registered": True})


def test_perturbed_bars_keep_the_past_and_move_the_future() -> None:
    bars, _ = subpanel()
    cut = pi.PanelCut(int(datetime(2019, 1, 10, 5, tzinfo=UTC).timestamp() * 1000), pi.RANDOM)
    out, left = pi.perturbed_bars(bars, cut, 0, 0, CFG)
    assert left == {}
    for s, b in bars.items():
        i = pi.cut_row(b, cut.hour_ms)
        assert b["open_time"][i].timestamp() * 1000 == cut.hour_ms
        assert out[s][: i + 1].equals(b[: i + 1])
        assert not np.any(out[s]["close"].to_numpy()[i + 1 :] == b["close"].to_numpy()[i + 1 :])


def test_compare_counts_bits_and_catches() -> None:
    a = {"x": np.array([1.0, 2.0, np.nan]), "y": np.array([0.0])}
    same = pi.compare(a, {k: v.copy() for k, v in a.items()})
    assert same["violations"] == 0 and same["caught"] == 0
    tiny = pi.compare(a, {"x": np.array([1.0 + 1e-15, 2.0, np.nan]), "y": np.array([0.0])})
    assert tiny["violations"] == 1 and tiny["caught"] == 0
    big = pi.compare(a, {"x": np.array([1.0, 2.0, 0.5]), "y": np.array([0.0])})
    assert big["caught"] == 1
    missing = pi.compare(a, {"x": a["x"]})
    assert missing["caught"] == 1 and missing["largest_change"] == float("inf")
