"""Point-in-time features: direct numpy references, gaps, zero volume, future invariance."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from olsprime.data.klines import TIMESTAMP
from olsprime.features import FEATURE_COLUMNS, basic_features
from olsprime.protocol import known_at_expr

DAY_MS = 86_400_000
T0 = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp()) * 1000


def make_bars(
    n: int,
    *,
    interval_ms: int = DAY_MS,
    start_ms: int = T0,
    seed: int = 0,
    drop: tuple[int, ...] = (),
    zero_volume: tuple[int, ...] = (),
    symbol: str = "TESTUSDT",
) -> pl.DataFrame:
    """``n`` random-walk bars in the ``load_klines`` schema (with ``known_at``), then the rows ``drop`` removed.

    Opens and closes differ, so a shifted index changes every value. Positions in
    ``drop`` and ``zero_volume`` count on the full grid of ``n`` bars.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.01, n))
    high = np.maximum(open_, close) * np.exp(rng.uniform(0.001, 0.02, n))
    low = np.minimum(open_, close) * np.exp(-rng.uniform(0.001, 0.02, n))
    quote_volume = np.exp(rng.normal(10.0, 0.5, n))
    taker = quote_volume * rng.uniform(0.3, 0.7, n)
    for i in zero_volume:
        quote_volume[i] = 0.0
        taker[i] = 0.0
    open_ms = start_ms + np.arange(n, dtype=np.int64) * interval_ms
    frame = pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "open_time": open_ms,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": quote_volume / close,
            "close_time": open_ms + interval_ms - 1,
            "quote_volume": quote_volume,
            "trades": np.full(n, 42, dtype=np.int64),
            "taker_buy_base_volume": taker / close,
            "taker_buy_quote_volume": taker,
        }
    ).with_columns(
        pl.col("open_time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        pl.col("close_time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
    )
    assert frame.schema["open_time"] == TIMESTAMP
    frame = frame.with_columns(known_at_expr({DAY_MS: "1d", 3_600_000: "1h"}[interval_ms]))
    keep = [i for i in range(n) if i not in set(drop)]
    return frame[keep]


def reference_features(bars: pl.DataFrame, step_ms: int) -> list[dict[str, float | None]]:
    """Features by the plain definition, looking bars up by time: a window needs every one of its bars."""
    by_time = {row["open_time"]: row for row in bars.iter_rows(named=True)}
    step = timedelta(milliseconds=step_ms)

    def window(t: datetime, size: int, first_back: int = 0) -> list[dict] | None:
        """Bars t-(size-1)*step .. t as rows, or None if one is missing."""
        rows = [by_time.get(t - j * step) for j in range(size - 1 + first_back, -1, -1)]
        return None if any(r is None for r in rows) else rows

    out = []
    for t in bars["open_time"].to_list():
        row: dict[str, float | None] = {}
        for k in (1, 3, 7, 14, 30, 90):
            w = window(t, k + 1)
            row[f"f_ret_{k}"] = None if w is None else math.log(w[-1]["close"] / w[0]["close"])
        for k in (7, 30):
            w = window(t, k + 1)
            if w is None:
                row[f"f_vol_{k}"] = None
            else:
                closes = np.array([r["close"] for r in w])
                row[f"f_vol_{k}"] = float(np.std(np.log(closes[1:] / closes[:-1]), ddof=1))
        for k in (10, 30, 90):
            w = window(t, k)
            row[f"f_ma_gap_{k}"] = (
                None if w is None else math.log(w[-1]["close"] / float(np.mean([r["close"] for r in w])))
            )
        cur = by_time[t]
        row["f_range_1"] = math.log(cur["high"] / cur["low"])
        w = window(t, 7)
        row["f_range_7"] = None if w is None else float(np.mean([math.log(r["high"] / r["low"]) for r in w]))
        w = window(t, 31)
        if w is None or any(r["quote_volume"] <= 0 for r in w):
            row["f_qvol_z_30"] = None
        else:
            logs = np.log([r["quote_volume"] for r in w])
            prev = logs[:-1]
            row["f_qvol_z_30"] = float((logs[-1] - prev.mean()) / prev.std(ddof=1))
        w = window(t, 7)
        if w is None or any(r["quote_volume"] <= 0 for r in w):
            row["f_taker_7"] = None
        else:
            row["f_taker_7"] = float(np.mean([r["taker_buy_quote_volume"] / r["quote_volume"] for r in w]))
        out.append(row)
    return out


def assert_matches(features: pl.DataFrame, expected: list[dict[str, float | None]]) -> None:
    assert features.height == len(expected)
    for name in FEATURE_COLUMNS:
        got = features[name].to_list()
        for i, (g, e) in enumerate(zip(got, [r[name] for r in expected], strict=True)):
            if e is None:
                assert g is None, f"{name}[{i}] should be null, got {g}"
            else:
                assert g is not None, f"{name}[{i}] should be {e}, got null"
                assert g == pytest.approx(e, rel=1e-9, abs=1e-12), f"{name}[{i}]"


def test_schema_and_decision_time():
    bars = make_bars(120)
    features = basic_features(bars, "1d")
    assert features.columns == ["symbol", "open_time", "decision_time", *FEATURE_COLUMNS]
    assert all(features.schema[name] == pl.Float64 for name in FEATURE_COLUMNS)
    assert features.schema["decision_time"] == TIMESTAMP
    assert features["open_time"].to_list() == bars["open_time"].to_list()
    # decision_time is bar t's known_at: the close, 1 ms before the next open
    assert features["decision_time"].to_list() == bars["known_at"].to_list()
    assert (features["decision_time"] - features["open_time"]).unique().to_list() == [timedelta(milliseconds=DAY_MS - 1)]
    assert len(FEATURE_COLUMNS) == 6 + 2 + 3 + 2 + 1 + 1


def test_each_feature_equals_a_direct_numpy_computation():
    bars = make_bars(200, seed=3)
    assert_matches(basic_features(bars, "1d"), reference_features(bars, DAY_MS))


def test_hand_computed_values_on_a_tiny_series():
    """Closes 1, 2, 4, 8, 16, 32, 64, 128: ln-returns are all ln 2, so a few values are known by hand."""
    n = 8
    bars = make_bars(n).with_columns(
        close=pl.Series([2.0**i for i in range(n)]),
        high=pl.Series([2.0**i * 2 for i in range(n)]),
        low=pl.Series([2.0**i for i in range(n)]),
    )
    features = basic_features(bars, "1d")
    ln2 = math.log(2)
    assert features["f_ret_1"].to_list()[1:] == pytest.approx([ln2] * (n - 1))
    assert features["f_ret_1"][0] is None
    assert features["f_ret_3"].to_list()[3:] == pytest.approx([3 * ln2] * (n - 3))
    assert features["f_ret_3"].to_list()[:3] == [None] * 3
    assert features["f_vol_7"][7] == pytest.approx(0.0, abs=1e-12)  # constant returns
    assert features["f_vol_7"].null_count() == 7
    assert features["f_range_1"].to_list() == pytest.approx([ln2] * n)
    # ma_gap_10 needs 10 bars: all null on 8 bars
    assert features["f_ma_gap_10"].null_count() == n


def test_warm_up_nulls_are_exactly_the_incomplete_windows():
    features = basic_features(make_bars(150), "1d")
    first_valid = {
        **{f"f_ret_{k}": k for k in (1, 3, 7, 14, 30, 90)},
        "f_vol_7": 7,
        "f_vol_30": 30,
        "f_ma_gap_10": 9,
        "f_ma_gap_30": 29,
        "f_ma_gap_90": 89,
        "f_range_1": 0,
        "f_range_7": 6,
        "f_qvol_z_30": 30,
        "f_taker_7": 6,
    }
    assert set(first_valid) == set(FEATURE_COLUMNS)
    for name, first in first_valid.items():
        nulls = features[name].is_null().to_numpy()
        assert nulls[:first].all() and not nulls[first:].any(), name


def test_gaps_make_windows_null_and_nothing_is_filled():
    bars = make_bars(160, seed=5, drop=(40, 41, 100))
    features = basic_features(bars, "1d")
    assert features.height == 157  # no row invented
    assert_matches(features, reference_features(bars, DAY_MS))
    # concretely: the first bar after the 40-41 hole cannot compute a 1-bar return
    times = bars["open_time"].to_list()
    after = times.index(datetime.fromtimestamp((T0 + 42 * DAY_MS) / 1000, tz=UTC))
    assert features["f_ret_1"][after] is None and features["f_ret_1"][after + 1] is not None
    assert features["f_ret_3"][after + 1] is None and features["f_ret_3"][after + 3] is not None
    assert features["f_range_1"][after] is not None  # a one-bar feature needs no neighbour
    # the null never turns into NaN/inf
    for name in FEATURE_COLUMNS:
        assert features[name].is_nan().sum() == 0 and features[name].is_infinite().sum() == 0


def test_a_missing_bar_far_back_nulls_only_the_windows_that_contain_it():
    bars = make_bars(130, drop=(20,))
    features = basic_features(bars, "1d")
    open_ms = bars["open_time"].dt.epoch("ms").to_list()
    index = {(ms - T0) // DAY_MS: i for i, ms in enumerate(open_ms)}
    # f_ret_90 at day 109 spans 19..109 -> contains the hole; day 111 spans 21..111 -> clean
    assert features["f_ret_90"][index[109]] is None
    assert features["f_ret_90"][index[111]] is not None
    # f_ma_gap_30 at day 49 spans 20..49 (hole), at day 50 spans 21..50 (clean)
    assert features["f_ma_gap_30"][index[49]] is None
    assert features["f_ma_gap_30"][index[50]] is not None


def test_zero_volume_nulls_the_volume_features_only_where_they_would_divide_by_zero():
    bars = make_bars(120, seed=2, zero_volume=(60,))
    features = basic_features(bars, "1d")
    assert_matches(features, reference_features(bars, DAY_MS))
    # f_taker_7 is null for the 7 windows containing bar 60: days 60..66
    nulls = features["f_taker_7"].is_null().to_numpy()
    assert nulls[60:67].all() and not nulls[67:].any() and not nulls[6:60].any()
    # f_qvol_z_30: bar 60 itself, and the 30 bars whose reference window holds it: days 60..90
    z_nulls = features["f_qvol_z_30"].is_null().to_numpy()
    assert z_nulls[60:91].all() and not z_nulls[91:].any() and not z_nulls[30:60].any()
    # price features are untouched
    assert features["f_ret_7"].null_count() == 7 and features["f_ma_gap_10"].null_count() == 9


def test_constant_volume_gives_null_z_not_inf():
    bars = make_bars(80).with_columns(quote_volume=pl.lit(500.0), taker_buy_quote_volume=pl.lit(250.0))
    features = basic_features(bars, "1d")
    assert features["f_qvol_z_30"].null_count() == 80  # std 0 -> null
    assert features["f_taker_7"][10] == pytest.approx(0.5)


def test_hourly_bars_use_the_hourly_grid():
    bars = make_bars(200, interval_ms=3_600_000, seed=8, drop=(70,))
    features = basic_features(bars, "1h")
    assert_matches(features, reference_features(bars, 3_600_000))
    # the same bars read as daily are off the grid: every window longer than one bar is null
    wrong = basic_features(bars, "1d")
    assert wrong["f_ret_1"].null_count() == wrong.height


# --- Future invariance (H1 at the feature level) ------------------------------------


def perturb_after(bars: pl.DataFrame, t: int, seed: int = 99) -> pl.DataFrame:
    """Bars with every value of every row after position t replaced by different numbers."""
    rng = np.random.default_rng(seed)
    m = bars.height - t - 1
    tail = {
        "open": rng.uniform(1, 1000, m),
        "high": rng.uniform(1000, 2000, m),
        "low": rng.uniform(0.1, 1, m),
        "close": rng.uniform(1, 1000, m),
        "volume": rng.uniform(0, 1e6, m),
        "quote_volume": rng.uniform(0, 1e9, m),
        "taker_buy_quote_volume": rng.uniform(0, 1e9, m),
    }
    head = bars[: t + 1]
    new_tail = bars[t + 1 :].with_columns([pl.Series(name, values) for name, values in tail.items()])
    return pl.concat([head, new_tail])


@pytest.mark.parametrize("t", [0, 5, 35, 99, 150])
def test_future_invariance_bit_for_bit(t):
    bars = make_bars(200, seed=11, drop=(50,), zero_volume=(120,))
    base = basic_features(bars, "1d")
    changed = basic_features(perturb_after(bars, t), "1d")
    assert base[: t + 1].equals(changed[: t + 1])  # exact: same bits, same nulls
    assert not base[t + 1 :].equals(changed[t + 1 :])  # the perturbation did reach the later rows


@pytest.mark.parametrize("t", [40, 120])
def test_features_do_not_depend_on_how_much_future_exists(t):
    """The sealed vault removes the future: truncating the bars at t changes nothing at or before t."""
    bars = make_bars(200, seed=12, drop=(30,))
    full = basic_features(bars, "1d")
    cut = basic_features(bars[: t + 1], "1d")
    assert full[: t + 1].equals(cut)


def test_features_per_symbol_and_row_order():
    a, b = make_bars(100, seed=1, symbol="AAAUSDT"), make_bars(100, seed=2, symbol="BBBUSDT")
    both = basic_features(pl.concat([a, b]), "1d")
    assert both.filter(pl.col("symbol") == "AAAUSDT").drop("symbol").equals(basic_features(a, "1d").drop("symbol"))
    assert both.filter(pl.col("symbol") == "BBBUSDT").drop("symbol").equals(basic_features(b, "1d").drop("symbol"))
    assert both["symbol"].to_list() == ["AAAUSDT"] * 100 + ["BBBUSDT"] * 100  # a window never crosses symbols


def test_known_at_is_computed_when_absent():
    bars = make_bars(60)
    assert basic_features(bars.drop("known_at"), "1d").equals(basic_features(bars, "1d"))


def test_unsorted_or_duplicate_bars_are_refused_not_repaired():
    bars = make_bars(60)
    with pytest.raises(ValueError, match="strictly increasing"):
        basic_features(pl.concat([bars, bars[10:11]]), "1d")
    with pytest.raises(ValueError, match="strictly increasing"):
        basic_features(bars.reverse(), "1d")


def test_input_errors_and_empty_input():
    bars = make_bars(60)
    with pytest.raises(ValueError, match="unsupported interval"):
        basic_features(bars, "7d")
    with pytest.raises(ValueError, match="lack columns"):
        basic_features(bars.drop("close"), "1d")
    empty = basic_features(bars.clear(), "1d")
    assert empty.height == 0 and empty.columns == ["symbol", "open_time", "decision_time", *FEATURE_COLUMNS]


def test_non_positive_prices_give_null_not_nan():
    bars = make_bars(60).with_columns(
        close=pl.when(pl.int_range(pl.len()) == 20).then(0.0).otherwise(pl.col("close"))
    )
    features = basic_features(bars, "1d")
    assert features["f_ret_1"][20] is None and features["f_ret_1"][21] is None
    assert features["f_ret_1"][22] is not None
    for name in FEATURE_COLUMNS:
        assert features[name].is_nan().sum() == 0 and features[name].is_infinite().sum() == 0
