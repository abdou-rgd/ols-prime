"""H1 on synthetic bars: honest models pass bit for bit, the cheater and planted leaks are flagged."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from olsprime.bench import invariance as inv
from olsprime.bench import leaks
from olsprime.bench.perturb import PERTURBED_COLUMNS, PRICE_COLUMNS, UNCHANGED_COLUMNS, Perturbation, check_coverage, perturb_after
from olsprime.bench.synthetic import synthetic_bars
from olsprime.models.baselines import TEST_COLUMNS, Ridge, check_test_frame

CFG = inv.H1Config(start=(2018, 3), end=(2018, 6), venues=("zero", "binance-spot", "kraken-pro-entry"))


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return synthetic_bars(days=560, seed=11)


@pytest.fixture(scope="module")
def cuts(bars):
    from olsprime.bench import pipeline as pp
    from olsprime.data.binance_vision import Month

    frame = pp.build_frame(bars, CFG.interval, CFG.h)
    folds = pp.make_folds(frame, CFG.interval, CFG.h, Month(*CFG.start), Month(*CFG.end))
    return inv.choose_cuts(bars, folds, seed=CFG.cut_seed)


def _specs(names):
    table = {s.name: s for s in inv.model_specs(CFG.interval, CFG.random_turnover_seeds)}
    return [table[n] for n in names]


# --- The perturbation --------------------------------------------------------------------


@pytest.mark.parametrize("level", [3.0, 1.0 / 3.0])
def test_perturbation_keeps_every_bar_up_to_the_cut_bit_for_bit(bars, level):
    cut = 300
    out = perturb_after(bars, cut, seed=1, config=Perturbation(level_shift=level))
    for c in bars.columns:
        assert bars[c][: cut + 1].equals(out[c][: cut + 1]), c
    for c in UNCHANGED_COLUMNS:  # the grid and the key, everywhere
        assert out[c].equals(bars[c]), c


@pytest.mark.parametrize("level", [3.0, 1.0 / 3.0])
def test_every_value_of_every_later_bar_moves(bars, level):
    """Coverage: every column but the grid and the key differs on every bar after the cut, trades included."""
    cut = 300
    out = perturb_after(bars, cut, seed=2, config=Perturbation(level_shift=level))
    later = slice(cut + 1, bars.height)
    moved = [c for c in bars.columns if c not in UNCHANGED_COLUMNS]
    assert set(moved) == set(PERTURBED_COLUMNS)  # nothing escapes, trades is in
    for c in moved:
        assert np.all(bars[c].to_numpy()[later] != out[c].to_numpy()[later]), c
    assert out["trades"].dtype == bars["trades"].dtype
    o, h, lo, c = (out[k].to_numpy() for k in PRICE_COLUMNS)
    assert np.all(lo <= np.minimum(o, c)) and np.all(h >= np.maximum(o, c)) and np.all(lo > 0)
    assert np.all(out["taker_buy_quote_volume"].to_numpy() <= out["quote_volume"].to_numpy())
    share = (out["taker_buy_quote_volume"] / out["quote_volume"]).to_numpy()[cut + 2 :]
    assert share.min() < 0.2 and share.max() > 0.8  # redrawn in (0, 1): it moves both ways


def test_the_coverage_check_refuses_an_unmoved_value(bars):
    cut = 300
    out = perturb_after(bars, cut, seed=2)
    check_coverage(bars, out, cut)
    stale = out.with_columns(pl.Series("trades", bars["trades"]))
    with pytest.raises(AssertionError, match="trades"):
        check_coverage(bars, stale, cut)
    moved_early = out.with_columns(pl.Series("volume", out["volume"].to_numpy() * np.r_[2.0, np.ones(out.height - 1)]))
    with pytest.raises(AssertionError, match="at or before the cut"):
        check_coverage(bars, moved_early, cut)
    with pytest.raises(ValueError, match="does not know"):
        perturb_after(bars.with_columns(pl.lit(1.0).alias("funding_rate")), cut, seed=0)


def _collapse(bars: pl.DataFrame, row: int) -> pl.DataFrame:
    """A tenfold volume collapse at ``row``, beyond the reach of a factor 3 alone."""
    return bars.with_columns(
        [pl.when(pl.int_range(pl.len()) == row).then(pl.col(c) / 10).otherwise(pl.col(c)).alias(c)
         for c in ("volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume")]
        + [pl.when(pl.int_range(pl.len()) == row).then(pl.col("trades") // 10).otherwise(pl.col("trades")).alias("trades")]
    )  # fmt: skip


HIGH_IN_DOWN = {"body", "range", "upper_wick", "lower_wick"}  # big in the /3 run, small in the x3 run
CELL_INDICATORS = {"red_small", "green_big"}  # 0/1 cells, predicted cut by cut, not pushed
AGAINST_BAR_T = {"fill", "close_location"}  # pushed past bar t alone


def _reference_values(name: str, q0: np.ndarray, cut: int) -> np.ndarray:
    """The values bar t + 1 must cross: the window (or bar t), inside the quantity's open interval."""
    from olsprime.bench.leaks import BOUNDS
    from olsprime.bench.perturb import WINDOW

    ref = q0[cut : cut + 1] if name in AGAINST_BAR_T else q0[cut - WINDOW : cut + 1]
    lo, hi = BOUNDS.get(name, (0.0, np.inf))
    ref = ref[np.isfinite(ref)]
    return ref[(ref > lo) & (ref < hi)] if name != "close" else ref


@pytest.mark.parametrize("cut", [100, 300, 450])
def test_every_one_bar_comparison_with_the_window_flips_in_exactly_one_direction(cut):
    """Bar t + 1 crosses bars t-90..t (bar t for fill and close location) in opposite ways in the two runs."""
    from olsprime.bench.leaks import QUANTITIES

    bars = _collapse(synthetic_bars(days=560, seed=cut), cut + 1)
    for mirror in (False, True):
        up, down = (perturb_after(bars, cut, seed=5, config=Perturbation(level_shift=lv, mirror=mirror)) for lv in (3.0, 1 / 3))
        for name, expr in QUANTITIES.items():
            if name in CELL_INDICATORS:
                continue
            q0, qu, qd = (b.select(expr.cast(pl.Float64).alias("q"))["q"].to_numpy() for b in (bars, up, down))
            high, low = (qd, qu) if (name in HIGH_IN_DOWN) != (mirror and name == "close_location") else (qu, qd)
            if mirror and name == "close_location":
                high, low = (qd, qu)  # the mirror: the big green /3 bar closes near its high
            ref = q0[cut : cut + 1] if mirror and name == "upper_wick" else _reference_values(name, q0, cut)
            assert ref.size and np.all(high[cut + 1] > ref) and np.all(low[cut + 1] <= ref), (name, mirror)
            assert qu[cut + 1] != q0[cut + 1] and qd[cut + 1] != q0[cut + 1], (name, mirror)
        green_up = up["close"][cut + 1] > up["open"][cut + 1]
        red_down = down["close"][cut + 1] < down["open"][cut + 1]
        assert (green_up and red_down) if not mirror else (not green_up and not red_down)


def _shape_of(b: pl.DataFrame, row: int) -> dict[str, float]:
    o, h, lo, c = (float(b[k][row]) for k in ("open", "high", "low", "close"))
    body, rng = abs(np.log(c / o)), np.log(h / lo)
    return {"body": body, "range": rng, "fill": body / rng, "clv": (c - lo) / (h - lo),
            "upper": np.log(h / max(o, c)), "lower": np.log(min(o, c) / lo)}  # fmt: skip


@pytest.mark.parametrize("cut", [100, 300, 450])
def test_the_two_runs_give_bar_t_plus_one_opposite_shapes_and_the_bands(cut):
    """Up: small, green, body-dominant (fill >= 0.8, close location >= 0.75). Down: big, red, upper wick >= 3 x lower."""
    from olsprime.bench.perturb import DOWN_CLV, DOWN_FILL, DOWN_FLOOR, UP_CAP, UP_CLV, UP_FILL, UPPER_OVER_LOWER

    bars = synthetic_bars(days=560, seed=cut + 1)
    for seed in range(5):
        up, down = (perturb_after(bars, cut, seed=seed, config=Perturbation(level_shift=lv)) for lv in (3.0, 1 / 3))
        u, d = _shape_of(up, cut + 1), _shape_of(down, cut + 1)
        assert u["fill"] >= UP_FILL and u["clv"] >= UP_CLV and up["close"][cut + 1] > up["open"][cut + 1]
        assert u["body"] <= UP_CAP["body"] and u["upper"] <= UP_CAP["wick"] and u["lower"] <= UP_CAP["wick"]
        assert u["range"] <= UP_CAP["body"] + 2 * UP_CAP["wick"] <= 0.005
        assert d["fill"] <= DOWN_FILL and d["clv"] <= DOWN_CLV and down["close"][cut + 1] < down["open"][cut + 1]
        assert d["upper"] >= UPPER_OVER_LOWER * d["lower"] and d["lower"] >= DOWN_FLOOR["wick"]
        assert d["body"] >= DOWN_FLOOR["body"] and d["range"] >= DOWN_FLOOR["body"] / DOWN_FILL >= 0.5
        # Thresholds inside the bands flip: 2 % on the body, 3 % on the range, 0.5 on the close location.
        assert u["body"] <= 0.02 < d["body"] and u["range"] <= 0.03 < d["range"] and d["clv"] <= 0.5 < u["clv"]


def test_a_doji_turns_green_in_the_up_run_and_red_in_the_down_run(bars):
    cut = 300
    doji = bars.with_columns(pl.when(pl.int_range(pl.len()) == cut + 1).then(pl.col("open")).otherwise(pl.col("close")).alias("close"))
    up, down = (perturb_after(doji, cut, seed=5, config=Perturbation(level_shift=lv)) for lv in (3.0, 1 / 3))
    assert up["close"][cut + 1] > up["open"][cut + 1] and down["close"][cut + 1] < down["open"][cut + 1]


def test_the_coverage_check_refuses_an_unmoved_ratio(bars):
    """A VWAP left as it was (volume and quote volume scaled together) is refused, as one unmoved value is."""
    cut = 300
    out = perturb_after(bars, cut, seed=2)
    row = cut + 7
    at = pl.int_range(pl.len()) == row  # both doubled: exact in binary, so the VWAP is bit for bit the original
    same_vwap = out.with_columns(
        pl.when(at).then(pl.lit(bars["volume"][row] * 2.0)).otherwise(pl.col("volume")).alias("volume"),
        pl.when(at).then(pl.lit(bars["quote_volume"][row] * 2.0)).otherwise(pl.col("quote_volume")).alias("quote_volume"),
    )
    with pytest.raises(AssertionError, match="ratio quote_volume/volume|ratio volume/quote_volume"):
        check_coverage(bars, same_vwap, cut)


def _probe(name: str):  # noqa: ANN202
    from olsprime.bench.leaks import PROBES

    return {p.name: p for p in PROBES}[name]


def test_a_declared_reason_read_on_the_original_bars_gives_a_question_mark(bars, cuts):
    """A taker share of 1 at bar t cannot be exceeded: ``?`` with its reason, out of the verdict, never a pass."""
    cut = cuts[0].index  # a scored bar, so that the bench can run the probe there
    rows = pl.int_range(pl.len()).is_in([cut, cut + 1])
    full = bars.with_columns(
        pl.when(rows).then(pl.col("quote_volume")).otherwise(pl.col("taker_buy_quote_volume")).alias("taker_buy_quote_volume"),
        pl.when(rows).then(pl.col("volume")).otherwise(pl.col("taker_buy_base_volume")).alias("taker_buy_base_volume"),
    )
    assert "bound" in _probe("taker-share-peeker").unknown_reason(full, cut)
    assert _probe("range-peeker").unknown_reason(full, cut) is None and _probe("vwap-peeker").unknown_reason(full, cut) is None
    (spec,) = _specs(["taker-share-peeker"])
    (record,) = inv.run_model(spec, full, cuts[:1], CFG)
    assert "bound" in record["unknown_reason"]
    v = inv.verdict(spec, [record])
    assert v["predicted_pattern"] == "?" and v["unknown_cuts"] == [cut] and str(cut) in v["unknown_reasons"]
    fake = [dict(record), {**record, "cut": 400, "unknown_reason": None, "violations": 0, "caught": 0}]
    assert not inv.verdict(spec, fake)["ok"]  # the known cut still decides


def test_a_bar_without_volume_declares_its_volume_probes_unknown(bars):
    cut = 300
    dead = pl.int_range(pl.len()) == cut + 1
    z = bars.with_columns(
        [pl.when(dead).then(0.0).otherwise(pl.col(c)).alias(c) for c in ("volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume")]
        + [pl.when(dead).then(0).otherwise(pl.col("trades")).alias("trades")]
    )  # fmt: skip
    for name in ("vwap-peeker", "taker-vwap-peeker", "trade-size-peeker", "taker-share-peeker", "quote-volume-peeker", "vwap-next-continuous"):
        assert _probe(name).unknown_reason(z, cut) == "zero volume at t+1", name
    for name in ("range-peeker", "body-peeker", "fill-peeker", "close-location-peeker", "close-vs-6-bars-back"):
        assert _probe(name).unknown_reason(z, cut) is None, name


def test_a_reference_before_the_first_bar_is_declared_not_wrapped(bars):
    """``cut < back`` would read ``values[-k]``, a future bar: it is a declared ``?`` instead."""
    probe = _probe("close-vs-6-bars-back")
    assert probe.unknown_reason(bars, 3) == "reference before the first bar"
    assert probe.unknown_reason(bars, 5) is None


def test_a_sabotaged_push_fails_h1_instead_of_reading_question_marks(bars, cuts, monkeypatch):
    """Turn the VWAP push off: the VWAP peeker falls off its pattern and H1 fails; nothing hides behind ``?``."""
    from olsprime.bench import perturb

    honest_first_bar = perturb.first_bar

    def sabotaged(b, cut, level, rng=None, *, mirror=False):  # noqa: ANN001, ANN202
        out = honest_first_bar(b, cut, level, rng, mirror=mirror)
        v0, q0 = float(b["volume"][cut + 1]), float(b["quote_volume"][cut + 1])
        if v0 > 0 and out["volume"] > 0:
            share = out["taker_buy_quote_volume"] / out["quote_volume"]
            out["quote_volume"] = out["volume"] * (q0 / v0) * (1.0 + 1e-6)  # the VWAP barely moves
            out["taker_buy_quote_volume"] = share * out["quote_volume"]
        return out

    monkeypatch.setattr(perturb, "first_bar", sabotaged)
    specs = _specs(["flat", "vwap-peeker"])
    pair = inv.run_pair(bars, CFG, specs, cuts=cuts)
    v = pair["models"]["vwap-peeker"]["verdict"]
    assert v["unknown_cuts"] == [] and not v["ok"], v
    assert not inv.h1_holds({"BTCUSDT": pair}, specs)["holds"]


def test_a_control_unknown_at_every_cut_fails_h1():
    specs = _specs(["flat", "vwap-peeker"])
    blank = {"verdict": {"violations": 0}}
    unknown = {"ok": True, "cuts": 2, "unknown_cuts": [1, 2], "flagged_cuts": 0}
    h1 = inv.h1_holds({"P": {"models": {"flat": blank, "vwap-peeker": {"verdict": unknown}}}}, specs)
    assert h1["controls_unknown_at_every_cut"] == ["P:vwap-peeker"] and not h1["holds"]


def test_a_bar_without_volume_keeps_its_zeros_and_its_nulls():
    from olsprime.features import basic_features

    bars = synthetic_bars(days=200, seed=4)
    dead = pl.int_range(pl.len()) == 150
    bars = bars.with_columns(
        [pl.when(dead).then(0.0).otherwise(pl.col(c)).alias(c) for c in ("volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume")]
        + [pl.when(dead).then(0).otherwise(pl.col("trades")).alias("trades")]
    )  # fmt: skip
    out = perturb_after(bars, 120, seed=1)
    assert out["volume"][150] == 0 and out["trades"][150] == 0 and out["taker_buy_quote_volume"][150] == 0
    f0, f1 = basic_features(bars, "1d"), basic_features(out, "1d")
    for c in f0.columns:
        assert f0[c].is_null().equals(f1[c].is_null()), c


def test_every_ratio_of_every_later_bar_moves(bars):
    """Bodies, ranges, VWAPs, taker VWAPs and trade sizes move on every later bar: one factor per bar would keep them."""
    from olsprime.bench.leaks import QUANTITIES

    cut = 300
    for lv in (3.0, 1 / 3):
        out = perturb_after(bars, cut, seed=3, config=Perturbation(level_shift=lv))
        for name, expr in QUANTITIES.items():
            if name in CELL_INDICATORS:
                continue
            q0, q1 = (b.select(expr.cast(pl.Float64).alias("q"))["q"].to_numpy()[cut + 1 :] for b in (bars, out))
            assert np.all(q0 != q1), name
        vwap = (out["quote_volume"] / out["volume"]).to_numpy()[cut + 2 :]
        assert np.all(vwap >= out["low"].to_numpy()[cut + 2 :] * (1 - 1e-12))
        assert np.all(vwap <= out["high"].to_numpy()[cut + 2 :] * (1 + 1e-12))


def test_perturbation_refuses_a_cut_without_a_later_bar(bars):
    with pytest.raises(ValueError, match="later bar"):
        perturb_after(bars, bars.height - 1, seed=0)


# --- Cuts and comparison -----------------------------------------------------------------


def test_at_least_eight_cuts_of_the_three_kinds(cuts):
    kinds = [c.kind for c in cuts]
    assert len(cuts) >= 8
    assert kinds.count(inv.LAST_BEFORE_FOLD) == 3
    assert kinds.count(inv.FIRST_OF_MONTH) == 3
    assert kinds.count(inv.RANDOM) == 2
    assert len({c.index for c in cuts}) == len(cuts)
    for c in cuts:
        if c.kind == inv.FIRST_OF_MONTH:
            assert c.open_time.day == 1
        if c.kind == inv.LAST_BEFORE_FOLD:
            assert (c.open_time + timedelta(days=1)).day == 1  # the next bar opens a test month


def test_compare_sees_a_single_bit():
    t = pl.Series("decision_time", [1, 2, 3]).cast(pl.Datetime("ms", "UTC"))
    a = pl.DataFrame({"decision_time": t, "exp_return": [0.1, 0.2, None]})
    b = a.with_columns(pl.Series("exp_return", [0.1, np.nextafter(0.2, 1.0), None]))
    assert inv.compare(a, a, t[2])["violations"] == 0  # nulls compare equal
    res = inv.compare(a, b, t[2])
    assert res["violations"] == 1 and res["columns"] == {"exp_return": 1}
    assert inv.compare(a, b, t[0])["violations"] == 0  # after the cut: not compared
    # One bit is a violation, not a catch: rounding carries no leak.
    assert res["caught"] == 0 and 0 < res["largest"] < 1e-15


def test_compare_catches_a_move_above_rounding_or_a_missing_value():
    t = pl.Series("decision_time", [1, 2, 3]).cast(pl.Datetime("ms", "UTC"))
    a = pl.DataFrame({"decision_time": t, "exp_return": [0.1, 0.2, None], "p_up": [0.3, 0.3, 0.3]})
    moved = a.with_columns(pl.Series("p_up", [0.3, 0.3 + 2 * inv.CAUGHT_ABOVE, 0.3]))
    res = inv.compare(a, moved, t[2])
    assert res["violations"] == 1 and res["caught"] == 1 and res["largest"] > inv.CAUGHT_ABOVE
    almost = a.with_columns(pl.Series("p_up", [0.3, 0.3 + inv.CAUGHT_ABOVE / 2, 0.3]))
    assert inv.compare(a, almost, t[2])["caught"] == 0  # at or below the threshold: not a catch
    for values in ([None, 0.2, None], [0.1, 0.2, 0.0]):  # a value turns missing, or a missing one appears
        res = inv.compare(a, a.with_columns(pl.Series("exp_return", values)), t[2])
        assert res["violations"] == 1 and res["caught"] == 1, values
        assert res["largest"] == 0.0  # no two present values differ


# --- Honest models pass, leaks are flagged -------------------------------------------------

FAST_HONEST = (
    "flat",
    "buy-and-hold",
    "ma-momentum-10",
    "ma-momentum-30",
    "ma-momentum-90",
    "random-turnover-seed0",
    "random-turnover-seed1",
    "random-turnover-seed2",
    "climatology",
    "ridge",
)


@pytest.mark.parametrize("name", FAST_HONEST)
def test_every_honest_baseline_passes_bit_for_bit(name, bars, cuts):
    (spec,) = _specs([name])
    records = inv.run_model(spec, bars, cuts, CFG)
    assert [r["violations"] for r in records] == [0] * len(cuts)
    assert all(r["rows"] > 0 for r in records)
    assert inv.verdict(spec, records)["ok"]


def test_the_cheater_is_flagged_at_every_cut(bars, cuts):
    (spec,) = _specs(["cheater"])
    records = inv.run_model(spec, bars, cuts, CFG)
    assert all(r["violations"] > 0 for r in records)
    assert all(r["first"] == c.known_at.isoformat() for r, c in zip(records, cuts, strict=True))  # it reads the next bar
    assert inv.verdict(spec, records)["ok"]
    # Bar t + 1 is green in the x3 run and red (or a doji) in the /3 run: the cheater's target is 1 and 0,
    # so exactly one run differs from the original at every cut, whatever the original colour.
    assert all(sum(r["by_direction"][d] > 0 for d in ("x3", "/3")) == 1 for r in records)


@pytest.mark.parametrize("name", ["ridge-unpurged", "climatology-unpurged"])
def test_a_model_trained_without_the_purge_is_flagged_exactly_at_the_fold_openings(name, bars, cuts):
    (spec,) = _specs([name])
    records = inv.run_model(spec, bars, cuts, CFG)
    assert [r["violations"] > 0 for r in records] == [c.opens_fold for c in cuts]
    assert any(c.opens_fold for c in cuts) and not all(c.opens_fold for c in cuts)
    assert inv.verdict(spec, records)["ok"]


@pytest.mark.parametrize("name", [*(p.name for p in leaks.PROBES if p.cell is None), "off-by-one-return"])
def test_each_planted_peeker_is_flagged_at_every_cut(name, bars, cuts):
    """Audit A's leaks (values, ratios inside a bar, thresholds, an older bar, a shifted return): permanent controls."""
    (spec,) = _specs([name])
    records = inv.run_model(spec, bars, cuts, CFG)
    v = inv.verdict(spec, records)
    assert v["pattern"] == "X" * len(cuts), v
    assert v["ok"] and v["unknown_cuts"] == []
    assert set(v["flagged_by_direction"]) == {"x3", "/3"}
    if spec.probe is not None and spec.probe.mode in ("next-vs-ref", "threshold"):
        # A comparison flips in exactly one run per cut: the two directions add up to the cuts.
        assert sum(v["flagged_by_direction"].values()) == len(cuts), v


def test_the_transductive_scaler_is_flagged_unless_its_month_is_over(bars, cuts):
    (spec,) = _specs(["transductive-scaler"])
    records = inv.run_model(spec, bars, cuts, CFG)
    assert [r["caught"] > 0 for r in records] == [c.month_continues for c in cuts]
    assert not all(c.month_continues for c in cuts)  # the last-bar-before-fold cuts end their month
    assert all(r["violations"] == 0 for r in records if not r["month_continues"])  # not a bit at a month's end
    # A real leak, far above rounding: on these bars its weakest catch moves an output by 8e-3.
    assert min(r["largest_change"] for r in records if r["month_continues"]) > 1e-4
    assert inv.verdict(spec, records)["ok"]


class _AbsorbedScaler(leaks.TransductiveScaler):
    """The transductive scaler before the fix: its ridge standardizes again, which undoes the leak exactly."""

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        assert self._train is not None
        feats = [c for c in test.columns if c.startswith("f_")]
        both = pl.concat([self._train.select(feats), test.select(feats)])
        mean, std = both.select(pl.col(feats).mean()).row(0), both.select(pl.col(feats).std(ddof=0)).row(0)
        scale = [(pl.col(c) - m) / (s if s else 1.0) for c, m, s in zip(feats, mean, std, strict=True)]
        inner = Ridge()
        inner.fit(self._train.with_columns(scale))
        return inner.predict(test.with_columns(scale))


def test_a_control_caught_by_rounding_alone_is_off_its_pattern(bars, cuts):
    """The absorbed scaler moves its outputs by about 1e-16 at most: a catch on one platform, not on
    another. Counted as no catch at all, it is off its pattern everywhere, and H1 says so."""
    spec = inv.ModelSpec("absorbed-scaler", lambda b: _AbsorbedScaler, expect=inv.FLAG_UNLESS_MONTH_END)
    records = inv.run_model(spec, bars, cuts, CFG)
    assert all(r["caught"] == 0 and r["largest_change"] < 1e-12 for r in records)
    v = inv.verdict(spec, records)
    assert v["pattern"] == "." * len(cuts) and not v["ok"]
    assert v["rounding_only_cuts"] == [r["cut"] for r in records if r["violations"] > 0]


def test_an_honest_model_is_flagged_by_a_single_bit():
    """Rounding never excuses an honest model: one differing bit is a violation."""
    (spec,) = _specs(["ridge"])
    record = {"cut": 1, "violations": 1, "by_direction": {"x3": 1, "/3": 0}, "caught": 0,
              "caught_by_direction": {"x3": 0, "/3": 0}, "largest_change": 1e-17, "opens_fold": False,
              "month_continues": True}  # fmt: skip
    v = inv.verdict(spec, [record])
    assert v["pattern"] == "X" and not v["ok"] and v["rounding_only_cuts"] == [1]


def test_a_control_flagged_off_its_pattern_is_not_ok(cuts):
    (spec,) = _specs(["ridge-unpurged"])
    records = [{"cut": c.index, "violations": 1, "by_direction": {"x3": 1, "/3": 0}, "caught": 1,
                "caught_by_direction": {"x3": 1, "/3": 0}, "largest_change": 0.5, "opens_fold": c.opens_fold,
                "month_continues": c.month_continues} for c in cuts]  # fmt: skip
    assert not inv.verdict(spec, records)["ok"]  # flagged everywhere, but predicted only at the fold openings


def test_the_positive_control_sees_a_signal_offline(bars):
    """Labels misaligned one bar early are nearly computable from the features: ridge must find them."""
    out = inv.positive_control(bars, CFG)
    assert out["labels_one_bar_early"]["bss"] > CFG.positive_bss_min
    assert out["passes"]
    assert out["real_labels"]["bss"] < 0.1  # the synthetic walk carries no signal
    assert out["labels_one_bar_early"]["rows"] > 100


def test_xgboost_passes_and_its_unpurged_twin_is_flagged(bars, cuts):
    picked = [c for c in cuts if c.kind == inv.FIRST_OF_MONTH][1:2] + [c for c in cuts if c.kind == inv.RANDOM][:1]
    honest, leaky = _specs(["xgboost", "xgboost-unpurged"])
    assert [r["violations"] for r in inv.run_model(honest, bars, picked, CFG)] == [0, 0]
    assert inv.run_model(leaky, bars, picked, CFG)[0]["violations"] > 0


class FullSampleMean:
    """A planted leak of another kind: a forecast scaled by the mean close of ALL the bars, future included.

    ``continuous=True`` reports ``exp_return = ln(close_t / mean)``; ``False`` reports only the
    target ``close_t > mean``. Full-sample scaling is the classic leak of a careless pipeline.
    """

    name = "full-sample-mean"
    config = None

    def __init__(self, bars: pl.DataFrame, *, continuous: bool) -> None:
        self.mean = float(bars["close"].mean())
        self.close = dict(zip(bars["open_time"].to_list(), bars["close"].to_list(), strict=True))
        self.continuous = continuous

    def fit(self, train: pl.DataFrame) -> None:
        pass

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        assert set(test.columns) <= set(TEST_COLUMNS) | {c for c in test.columns if c.startswith("f_")}
        close = np.array([self.close[t] for t in test["open_time"].to_list()])
        n = test.height
        nulls = pl.Series([None] * n, dtype=pl.Float64)
        if self.continuous:
            exp_return, target = pl.Series(np.log(close / self.mean)), nulls
        else:
            exp_return, target = nulls, pl.Series((close > self.mean).astype(np.float64))
        return pl.DataFrame(
            {"symbol": test["symbol"], "decision_time": test["decision_time"], "p_down": nulls, "p_neutral": nulls,
             "p_up": nulls, "exp_return": exp_return, "target": target}
        )  # fmt: skip


def test_a_full_sample_statistic_is_flagged_at_every_cut(bars, cuts):
    spec = inv.ModelSpec("full-sample-mean", lambda b: (lambda: FullSampleMean(b, continuous=True)), expect=inv.FLAG_EVERY_CUT)
    records = inv.run_model(spec, bars, cuts, CFG)
    assert all(r["violations"] == r["rows"] for r in records)  # every compared row moves, at every cut
    assert inv.verdict(spec, records)["ok"]


def test_a_leak_seen_only_through_a_binary_target_is_caught_less_often(bars, cuts):
    """A limit of the test, kept visible: a leak that never moves a continuous output is caught
    only where it flips a decision. The same full-sample leak, reported as a 0/1 target alone,
    is flagged at some cuts, not at all of them, even with the future tripled and divided by
    three (without the level shifts it escaped every cut of this series)."""
    spec = inv.ModelSpec("full-sample-sign", lambda b: (lambda: FullSampleMean(b, continuous=False)), expect=inv.FLAG_EVERY_CUT)
    flagged = [r["violations"] > 0 for r in inv.run_model(spec, bars, cuts, CFG)]
    assert 0 < sum(flagged) < len(cuts)


def test_the_seal_leaves_the_cheater_blind_at_the_last_bar(bars):
    out = inv.blind_cheater(bars, CFG)
    assert out["cheater_target_at_last_bar"] == 0.0
    assert out["label_of_last_bar_is_null"]
    assert out["last_sealed_open_time"] == bars["open_time"][-1].isoformat()


def test_h1_verdict_requires_every_part():
    specs = _specs(["flat", "cheater"])

    def verdict(ok, violations):
        return {"verdict": {"violations": violations, "ok": ok, "flagged_cuts": violations, "cuts": 8}}

    good = {"P": {"models": {"flat": verdict(True, 0), "cheater": verdict(True, 8)}}}
    alive = {"P": {"passes": True}}
    assert inv.h1_holds(good, specs, alive)["holds"]
    assert inv.h1_holds(good, specs)["holds"]  # positive control not run: no verdict on it
    bad = {"P": {"models": {"flat": verdict(False, 1), "cheater": verdict(True, 8)}}}
    assert not inv.h1_holds(bad, specs, alive)["holds"]
    blind = {"P": {"models": {"flat": verdict(True, 0), "cheater": verdict(False, 7)}}}
    assert inv.h1_holds(blind, specs, alive)["controls_off_pattern"] == ["P:cheater"]
    assert not inv.h1_holds(blind, specs, alive)["holds"]
    dead = {"P": {"passes": False}}
    assert not inv.h1_holds(good, specs, dead)["holds"]


def test_the_report_states_the_limit_and_the_positive_control(bars, cuts):
    specs = _specs(["flat", "cheater"])
    pair = inv.run_pair(bars, CFG, specs, cuts=cuts)
    positive = {"BTCUSDT": inv.positive_control(bars, CFG)}
    result = {"config": inv._config_json(CFG), "pairs": {"BTCUSDT": pair}, "positive_control": positive,
              "seal": {"BTCUSDT": inv.blind_cheater(bars, CFG)}, "h1": inv.h1_holds({"BTCUSDT": pair}, specs, positive)}  # fmt: skip
    text = inv.render_md(inv._strip_masks(result))
    assert "Verdict: H1 holds" in text
    assert "The limit of this test" in text and "open_time" in text
    assert "Positive control" in text and "flagged x3" in text and "flagged /3" in text


# --- Provenance ----------------------------------------------------------------------------


def _tree(root):
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (root / ".git" / "refs" / "heads" / "main").write_text("e" * 40 + "\n", encoding="ascii")
    pkg = root / "python" / "olsprime"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "__pycache__" / "a.py").write_text("ignored\n", encoding="utf-8")
    (root / "Cargo.lock").write_text("lock\n", encoding="utf-8")


def test_the_code_digest_is_stable_for_an_unchanged_tree_and_sees_one_byte(tmp_path):
    import hashlib

    from olsprime.bench import provenance, report1
    from olsprime.paths import REPO_ROOT

    _tree(tmp_path)
    first = provenance.code_digest(tmp_path)
    assert provenance.code_digest(tmp_path) == first and first["files"] == 2  # __pycache__ is not source
    expected = hashlib.sha256()  # the documented bytes: "<posix path>\0<sha256 hex of the file>\n", sorted by path
    for rel in ("Cargo.lock", "python/olsprime/a.py"):
        expected.update(rel.encode() + b"\0" + hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest().encode() + b"\n")
    assert first["sha256"] == expected.hexdigest() and first["globs"] == list(provenance.CODE_GLOBS)
    (tmp_path / "python" / "olsprime" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert provenance.code_digest(tmp_path)["sha256"] != first["sha256"]
    assert provenance.code_digest(REPO_ROOT) == provenance.code_digest(REPO_ROOT)
    assert report1.code_digest is provenance.code_digest and report1.git_commit is provenance.git_commit  # one source


def test_h1_records_its_provenance_before_reading_any_data(bars, monkeypatch):
    import json

    from olsprime.bench import provenance

    calls: list[str] = []
    record = provenance.record

    def recording(manifest, root=provenance.REPO_ROOT):  # noqa: ANN001
        calls.append("provenance")
        return record(manifest, root)

    def load(pair, cfg):  # noqa: ANN001
        calls.append("data")
        return bars

    monkeypatch.setattr(provenance, "record", recording)
    monkeypatch.setattr(inv, "_load", load)
    specs = _specs(["flat", "cheater"])
    monkeypatch.setattr(inv, "model_specs", lambda interval, seeds: specs)
    manifest = {"digest": "f" * 64, "generated_at": "2026-10-28T00:00:00+00:00", "vault_open": False}
    result = inv.run_real(replace(CFG, pairs=("BTCUSDT",)), manifest=manifest)

    assert calls[0] == "provenance" and calls.count("provenance") == 1
    prov = json.loads(json.dumps(result, default=str))["provenance"]
    assert set(prov) == set(provenance.PROVENANCE_KEYS) == {"git_commit", "code", "manifest_digest"}
    assert prov["manifest_digest"] == "f" * 64
    assert prov["git_commit"] is None or (len(prov["git_commit"]) == 40 and int(prov["git_commit"], 16) >= 0)
    assert len(prov["code"]["sha256"]) == 64 and int(prov["code"]["sha256"], 16) >= 0 and prov["code"]["files"] > 0
    text = inv.render_md(inv._strip_masks(result))
    tail = text[text.index("## Provenance") :]
    assert prov["code"]["sha256"] in tail and "f" * 64 in tail and f"{prov['code']['files']} source" in tail
    assert str(prov["git_commit"]) in tail


@pytest.mark.parametrize("cut", [100, 300, 450])
def test_the_mirror_shape_at_odd_cuts(cut):
    """Odd cuts: x3 small red closing near its low; /3 big green, lower wick >= 3 x upper, closing near its high."""
    from olsprime.bench.perturb import DOWN_CLV, DOWN_FILL, DOWN_FLOOR, UP_CLV, UP_FILL, UPPER_OVER_LOWER

    bars = synthetic_bars(days=560, seed=cut + 2)
    for seed in range(5):
        up, down = (perturb_after(bars, cut, seed=seed, config=Perturbation(level_shift=lv, mirror=True)) for lv in (3.0, 1 / 3))
        u, d = _shape_of(up, cut + 1), _shape_of(down, cut + 1)
        assert up["close"][cut + 1] < up["open"][cut + 1] and u["clv"] <= DOWN_CLV and u["fill"] >= UP_FILL
        assert down["close"][cut + 1] > down["open"][cut + 1] and d["clv"] >= UP_CLV and d["fill"] <= DOWN_FILL
        assert d["lower"] >= UPPER_OVER_LOWER * d["upper"] and d["body"] >= DOWN_FLOOR["body"]
        assert u["body"] < 0.01 < d["body"]  # small and big: the (red, small) and (green, big) cells


def test_the_cells_alternate_and_the_cell_controls_are_predicted_from_them(bars, cuts):
    """Even cuts realise (green, small) and (red, big), odd cuts (red, small) and (green, big)."""
    from olsprime.bench.perturb import CELLS

    cells = inv.realised_cells(bars, cuts, CFG)
    for k, row in enumerate(cells):
        assert row["mirror"] == (k % 2 == 1)
        assert tuple(row["x3"]) == CELLS[row["mirror"]]["x3"] and tuple(row["/3"]) == CELLS[row["mirror"]]["/3"]
    for name in ("red-and-small", "green-and-big"):
        (spec,) = _specs([name])
        assert spec.expect == inv.FLAG_BY_CELL
        records = inv.run_model(spec, bars, cuts, CFG)
        v = inv.verdict(spec, records)
        assert v["ok"] and v["unknown_cuts"] == [], v
        # At odd cuts one run realises the probe's cell: always flagged. At even cuts neither does:
        # flagged exactly where the original bar t + 1 was in that cell.
        for k, (r, c) in enumerate(zip(records, cuts, strict=True)):
            orig = bool(spec.probe.values(bars)[c.index + 1] > 0.5)
            assert r["predicted_by_cell"] == (True if k % 2 else orig)


def _with_bar_t(bars: pl.DataFrame, t: int, o: float, h: float, lo: float, c: float) -> pl.DataFrame:
    at = pl.int_range(pl.len()) == t
    return bars.with_columns([pl.when(at).then(pl.lit(v)).otherwise(pl.col(k)).alias(k)
                              for k, v in (("open", o), ("high", h), ("low", lo), ("close", c))])  # fmt: skip


def test_the_fill_floor_is_exact(bars):
    """A bar t with a fill of 0.021: the /3 fill goes to the floor 0.02, never below, at both parities."""
    from olsprime.bench.perturb import FILL_FLOOR

    cut = 300
    c = float(bars["close"][cut])
    b = _with_bar_t(bars, cut, c, c * np.exp(0.5), c * np.exp(-0.4762), c * np.exp(0.021))  # body 0.021, range ~0.9762+0.021
    f_t = _shape_of(b, cut)["fill"]
    assert 0.02 < f_t < 0.03
    for mirror in (False, True):
        for seed in range(5):
            down = perturb_after(b, cut, seed=seed, config=Perturbation(level_shift=1 / 3, mirror=mirror))
            assert FILL_FLOOR <= _shape_of(down, cut + 1)["fill"] <= f_t


def test_the_close_location_floor_and_its_declared_question_mark(bars):
    """A bar t closing at 1 % of its range from its low, or from its high: declared ``?`` at both parities.

    At every cut one run aims below bar t's close location and the other above it, and neither goes past the
    floor 0.02 or the ceiling 0.98: at an even cut the /3 bar is floored, at an odd cut the small x3 bar is.
    """
    cut = 300
    c = float(bars["close"][cut])
    low, high = c * 0.99, c * 1.09
    b = _with_bar_t(bars, cut, c * 1.05, high, low, low + 0.01 * (high - low))
    probe = _probe("close-location-peeker")
    assert all("floor" in probe.unknown_reason(b, cut, mirror=m) for m in (False, True))
    top = _with_bar_t(bars, cut, c * 1.0, high, low, high - 0.01 * (high - low))
    assert all("ceiling" in probe.unknown_reason(top, cut, mirror=m) for m in (False, True))
    inside = _with_bar_t(bars, cut, c * 1.0, high, low, low + 0.5 * (high - low))
    assert all(probe.unknown_reason(inside, cut, mirror=m) is None for m in (False, True))
    for mirror in (False, True):  # the floor bounds the target, so the /3 bar stays within a few log units
        down = perturb_after(b, cut, seed=1, config=Perturbation(level_shift=1 / 3, mirror=mirror))
        d = _shape_of(down, cut + 1)
        assert (d["clv"] <= 0.25) if not mirror else (d["clv"] >= 0.75)
        assert d["range"] < 20.0
