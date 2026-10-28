"""Forward labels: entry and exit at the right opens, known-at stamps, nulls, thresholds, future invariance."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from olsprime.data.binance_vision import parquet_path
from olsprime.features import basic_features
from olsprime.labels import LABEL_COLUMNS, forward_labels
from olsprime.protocol import VAULT, load_klines
from tests.test_features import DAY_MS, T0, make_bars, perturb_after


def at(day: int) -> datetime:
    return datetime.fromtimestamp((T0 + day * DAY_MS) / 1000, tz=UTC)


def distinct_bars(n: int = 100, **kwargs) -> pl.DataFrame:
    """Opens 1000, 1001, ... and closes 5000, 5001, ...: confusing one with the other or shifting a bar shows."""
    return make_bars(n, **kwargs).with_columns(
        open=pl.int_range(pl.len()).cast(pl.Float64) * 3 + 1000.0,
        close=pl.int_range(pl.len()).cast(pl.Float64) + 5000.0,
    )


def test_columns_types_and_constants():
    labels = forward_labels(make_bars(80), "1d", 1)
    assert labels.columns == [
        "symbol", "open_time", "decision_time", "entry_time", "exit_time",
        "label_known_at", "fwd_return", "threshold", "label",
    ]  # fmt: skip
    assert labels.schema["label"] == pl.UInt8 and labels.schema["fwd_return"] == pl.Float64
    assert LABEL_COLUMNS == ("entry_time", "exit_time", "label_known_at", "fwd_return", "label")
    assert "threshold" not in LABEL_COLUMNS
    assert labels.height == 80


@pytest.mark.parametrize("h", [1, 3, 7])
def test_fwd_return_uses_open_of_t_plus_1_and_open_of_t_plus_1_plus_h(h):
    bars = distinct_bars()
    labels = forward_labels(bars, "1d", h)
    opens = bars["open"].to_numpy()
    for t in (0, 10, 50, 90 - h):
        # NumPy's log may take a SIMD path (AVX-512 on some Linux runners) that differs from math.log by
        # an ulp, so allow 1e-14 relative. The usual mistakes below are more than 1e-6 away.
        assert labels["fwd_return"][t] == pytest.approx(math.log(opens[t + 1 + h] / opens[t + 1]), rel=1e-14, abs=0)
        assert labels["entry_time"][t] == at(t + 1)
        assert labels["exit_time"][t] == at(t + 1 + h)
        assert labels["label_known_at"][t] == labels["exit_time"][t]
        assert labels["decision_time"][t] == bars["known_at"][t]
    # the usual mistakes all give a different number: closes, no entry lag, off-by-one exit
    closes = bars["close"].to_numpy()
    t = 10
    wrong = [
        math.log(closes[t + h] / closes[t]),
        math.log(opens[t + h] / opens[t]),  # exits one bar early, enters at the decision bar
        math.log(opens[t + 2 + h] / opens[t + 2]),
        math.log(opens[t + 1 + h] / opens[t]),
    ]
    assert all(abs(labels["fwd_return"][t] - w) > 1e-6 for w in wrong)


def test_exit_is_strictly_after_the_decision_and_after_entry():
    labels = forward_labels(make_bars(100), "1d", 7).drop_nulls("label_known_at")
    assert (labels["entry_time"] > labels["decision_time"]).all()
    assert ((labels["exit_time"] - labels["entry_time"]) == timedelta(days=7)).all()
    assert (labels["label_known_at"] > labels["decision_time"]).all()


def test_nulls_at_the_end_of_the_data():
    bars = distinct_bars(60)
    for h in (1, 7):
        labels = forward_labels(bars, "1d", h)
        last_valid = 60 - 1 - 1 - h  # t + 1 + h must be <= 59
        for name in LABEL_COLUMNS:
            nulls = labels[name].is_null().to_numpy()
            # the class also needs the threshold, which is null during the 30-bar volatility warm-up
            first = 30 if name == "label" else 0
            assert nulls[last_valid + 1 :].all() and not nulls[first : last_valid + 1].any(), (name, h)
        assert labels["threshold"].null_count() == 30
        assert labels["label"][:30].null_count() == 30 and labels["fwd_return"][:30].null_count() == 0


def test_nulls_across_gaps_and_missing_next_bar():
    bars = distinct_bars(100, drop=(50, 70, 71))
    labels = forward_labels(bars, "1d", 3)
    row = {ms: i for i, ms in enumerate(bars["open_time"].dt.epoch("ms").to_list())}

    def label_row(day: int) -> dict:
        return labels.row(row[T0 + day * DAY_MS], named=True)

    # day 49: next bar (50) is missing -> null
    assert label_row(49)["fwd_return"] is None and label_row(49)["entry_time"] is None
    # day 46: the exit bar (t+4 = day 50) is missing; day 47 exits on day 51, entry on 48: both present
    assert label_row(46)["fwd_return"] is None  # exit would be day 50: missing
    assert label_row(47)["fwd_return"] is not None  # exit day 51: present, entry day 48 present
    # day 51 entering on 52, exiting on 55: clean
    assert label_row(51)["fwd_return"] is not None
    # days 66, 67: exit 70/71 missing. day 68: entry 69, exit 72 -> clean
    assert label_row(66)["fwd_return"] is None and label_row(67)["fwd_return"] is None
    assert label_row(68)["fwd_return"] is not None
    # day 69: the next bar (70) is missing
    assert label_row(69)["entry_time"] is None
    # the null label stays null and is never replaced by a class
    assert label_row(49)["label"] is None and label_row(49)["label_known_at"] is None


def test_bars_are_found_by_time_not_by_row_offset():
    """A hole between entry and exit leaves the label valid and uses the true exit open (day t+1+h),
    never the row t+1+h, which would be a later bar."""
    bars = distinct_bars(40, drop=(12,))
    labels = forward_labels(bars, "1d", 5)
    opens = {ms: o for ms, o in zip(bars["open_time"].dt.epoch("ms").to_list(), bars["open"].to_list(), strict=True)}
    row = labels.filter(pl.col("open_time") == at(8)).row(0, named=True)
    assert row["exit_time"] == at(14)
    assert row["fwd_return"] == pytest.approx(math.log(opens[T0 + 14 * DAY_MS] / opens[T0 + 9 * DAY_MS]), rel=1e-14, abs=0)
    # an exit bar that is itself the hole is null, and so is an entry bar that is the hole
    assert labels.filter(pl.col("open_time") == at(6))["exit_time"][0] is None  # t+1+5 = 12
    assert labels.filter(pl.col("open_time") == at(11))["entry_time"][0] is None  # t+1 = 12


def test_label_classes_follow_the_threshold():
    n = 100
    bars = distinct_bars(n, seed=1)
    labels = forward_labels(bars, "1d", 1, vol_window=30, k=0.5, floor=0.0024)
    f = labels.drop_nulls("label")
    fwd, thr = f["fwd_return"].to_numpy(), f["threshold"].to_numpy()
    expected = np.where(fwd < -thr, 0, np.where(fwd > thr, 2, 1))
    assert f["label"].to_list() == expected.tolist()
    assert set(f["label"].unique().to_list()) <= {0, 1, 2}


def test_threshold_formula_and_floor():
    bars = make_bars(120, seed=4)
    vol = basic_features(bars, "1d")["f_vol_30"].to_numpy()
    for h, k, floor in [(1, 0.5, 0.0024), (7, 0.5, 0.0024), (3, 2.0, 0.0), (1, 0.0, 0.0024), (1, 0.5, 0.5)]:
        thr = forward_labels(bars, "1d", h, k=k, floor=floor)["threshold"].to_numpy()
        want = np.maximum(k * vol * math.sqrt(h), floor)
        assert np.array_equal(np.isnan(thr), np.isnan(want))
        assert np.allclose(thr[~np.isnan(thr)], want[~np.isnan(want)], rtol=1e-12, atol=0)
    # a flat market: the floor binds and every move below 24 bp is neutral
    flat = make_bars(80).with_columns(close=pl.lit(100.0), open=pl.lit(100.0))
    out = forward_labels(flat, "1d", 1)
    assert out["threshold"].drop_nulls().unique().to_list() == [0.0024]
    assert out["label"].drop_nulls().unique().to_list() == [1]


def test_label_classes_on_hand_built_moves():
    n = 60
    opens = np.full(n, 100.0)
    opens[41:] = 100.0 * math.exp(0.05)  # entry at day 41 is 100, ... jump up later
    bars = make_bars(n).with_columns(open=pl.Series(opens), close=pl.lit(100.0))
    labels = forward_labels(bars, "1d", 1, floor=0.0024)
    # closes are flat -> vol 0 -> threshold is the floor. Jump of ln(e^0.05) between open[40] and open[41]
    assert labels["fwd_return"][39] == pytest.approx(0.05) and labels["label"][39] == 2
    assert labels["label"][45] == 1 and labels["fwd_return"][45] == 0.0
    down = make_bars(n).with_columns(
        open=pl.Series(np.where(np.arange(n) >= 41, 90.0, 100.0)), close=pl.lit(100.0)
    )
    assert forward_labels(down, "1d", 1)["label"][39] == 0


def test_threshold_and_decision_time_do_not_depend_on_the_future():
    bars = make_bars(150, seed=6)
    base = forward_labels(bars, "1d", 3)
    for t in (30, 80, 120):
        changed = forward_labels(perturb_after(bars, t), "1d", 3)
        cols = ["symbol", "open_time", "decision_time", "threshold"]
        assert base.select(cols)[: t + 1].equals(changed.select(cols)[: t + 1])
        # the label itself is different once a perturbed price enters it: that is why it is stamped
        assert not base[: t + 1].equals(changed[: t + 1]) or t + 4 > bars.height


def test_a_label_only_depends_on_bars_up_to_label_known_at():
    """Perturb every bar from the exit bar onward except... nothing before label_known_at may matter."""
    bars = make_bars(120, seed=7)
    base = forward_labels(bars, "1d", 5)
    t = 40
    exit_day = t + 1 + 5
    # change the exit bar's open and everything after: row t must change; changing only the bars
    # strictly after the exit bar must not touch row t
    after_exit = perturb_after(bars, exit_day)
    changed = forward_labels(after_exit, "1d", 5)
    assert base[t].equals(changed[t])
    assert base["label_known_at"][t] == at(exit_day)
    before_exit = perturb_after(bars, exit_day - 1)
    assert not base[t].equals(forward_labels(before_exit, "1d", 5)[t])


def test_labels_per_symbol_never_cross():
    a, b = make_bars(60, seed=1, symbol="AAAUSDT"), make_bars(60, seed=2, symbol="BBBUSDT")
    both = forward_labels(pl.concat([a, b]), "1d", 2)
    solo = forward_labels(a, "1d", 2)
    assert both.filter(pl.col("symbol") == "AAAUSDT").equals(solo)
    assert both.filter(pl.col("symbol") == "BBBUSDT")["fwd_return"][-3:].null_count() == 3  # the end of B is not A's start


def test_input_errors_and_refusals():
    bars = make_bars(40)
    for h in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="horizon"):
            forward_labels(bars, "1d", h)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vol_window"):
        forward_labels(bars, "1d", 1, vol_window=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        forward_labels(pl.concat([bars, bars[3:4]]), "1d", 1)
    assert forward_labels(bars.clear(), "1d", 1).height == 0


def test_the_sealed_vault_ends_the_labels(tmp_path):
    """Through load_klines: bars final in the vault are gone, so labels reaching them are null."""
    start = int(datetime(2025, 11, 20, tzinfo=UTC).timestamp()) * 1000
    raw = make_bars(60, start_ms=start).drop("known_at")  # 2025-11-20 .. 2026-01-18
    path = parquet_path(tmp_path / "parquet", "TESTUSDT", "1d")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.write_parquet(path)
    sealed = load_klines("TESTUSDT", "1d", data_dir=tmp_path)
    assert sealed.height == 42 and sealed["open_time"].max() == datetime(2025, 12, 31, tzinfo=UTC)
    assert (sealed["known_at"] < VAULT.start).all()
    labels = forward_labels(sealed, "1d", 3)
    last = sealed.height - 1
    assert labels["fwd_return"][last - 4] is not None  # exits on the last sealed bar's open
    assert labels["fwd_return"][last - 3 :].null_count() == 4  # t + 1 + 3 > last: the exit bar is sealed away
    assert labels["label_known_at"].drop_nulls().max() < VAULT.start
