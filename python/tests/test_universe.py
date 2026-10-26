from datetime import UTC, date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from olsprime.data.binance_vision import Month
from olsprime.universe import (
    UNIVERSE_SCHEMA,
    build_universe,
    eligible_symbols,
    list_archive_months,
    list_spot_symbols,
    members_at,
    run,
    universe_path,
)
from tests.fakes import DAY_MS, FakeBucket, month_rows


def daily(symbol: str, first: date, last: date, quote_volume: float) -> pl.DataFrame:
    """One bar per day from ``first`` to ``last``, both included."""
    days = pl.datetime_range(
        datetime(first.year, first.month, first.day, tzinfo=UTC),
        datetime(last.year, last.month, last.day, tzinfo=UTC),
        "1d",
        time_unit="ms",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame({"symbol": symbol, "open_time": days, "quote_volume": quote_volume})


def at(year: int, month: int, day: int = 15) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# --- Listing and eligibility ------------------------------------------------------


def test_listing_follows_pagination_and_keeps_delisted_pairs():
    bucket = FakeBucket(page_size=2)
    # BULLUSDT traded from 2020-01 to 2020-03 only; the bucket still holds its archives.
    for month in (Month(2020, 1), Month(2020, 2), Month(2020, 3)):
        bucket.add_month("BULLUSDT", "1d", month, month_rows(month, DAY_MS))
    for symbol in ("BTCUSDT", "ETHBTC", "ETHUSDT", "LUNAUSDT", "USDT"):
        bucket.add_month(symbol, "1d", Month(2026, 9), month_rows(Month(2026, 9), DAY_MS))
    with bucket.client() as client:
        assert list_spot_symbols(client) == ["BTCUSDT", "BULLUSDT", "ETHUSDT", "LUNAUSDT"]
        assert list_archive_months(client, "BULLUSDT", "1d") == [Month(2020, 1), Month(2020, 2), Month(2020, 3)]
    listing_requests = [url for url in bucket.requests if url.host.startswith("s3")]
    assert len(listing_requests) > 2  # more than one page each time


def test_stablecoins_and_leveraged_tokens_are_excluded():
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "JUPUSDT", "SYRUPUSDT",
        "BTCUPUSDT", "BTCDOWNUSDT", "ETHBULLUSDT", "BNBBEARUSDT", "BULLUSDT", "BEARUSDT",
        "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT", "PAXGUSDT",
    ]
    assert eligible_symbols(symbols) == ["BNBUSDT", "BTCUSDT", "ETHUSDT", "JUPUSDT", "SYRUPUSDT"]


# --- Ranking ----------------------------------------------------------------------


def test_delisted_pair_is_a_member_until_it_stops_trading():
    data = pl.concat(
        [
            daily("OLDUSDT", date(2020, 1, 1), date(2020, 6, 14), 1e9),  # last bar 2020-06-14
            daily("AAAUSDT", date(2020, 1, 1), date(2020, 12, 31), 1e6),
            daily("BBBUSDT", date(2020, 1, 1), date(2020, 12, 31), 1e5),
        ]
    )
    universe = build_universe(data, start=Month(2020, 3), end=Month(2020, 8), top_n=2)
    for month in (3, 4, 5, 6):
        assert members_at(universe, at(2020, month)) == ["OLDUSDT", "AAAUSDT"]
    # Still held after its last bar: nobody knew on 2020-06-01 that it would stop.
    assert "OLDUSDT" in members_at(universe, at(2020, 6, 20))
    for month in (7, 8):
        assert members_at(universe, at(2020, month)) == ["AAAUSDT", "BBBUSDT"]


def test_universe_uses_no_future_information():
    data = pl.concat(
        [
            daily("AAAUSDT", date(2020, 1, 1), date(2020, 12, 31), 3e6),
            daily("BBBUSDT", date(2020, 1, 1), date(2020, 12, 31), 2e6),
            daily("CCCUSDT", date(2020, 1, 1), date(2020, 12, 31), 1e6),
        ]
    )
    cutoff = datetime(2020, 6, 1, tzinfo=UTC)
    future = pl.col("open_time") >= cutoff
    # Rewrite everything from the cutoff on: CCC explodes, AAA stops trading, a new pair lists.
    altered = pl.concat(
        [
            data.filter(~future),
            data.filter(future & (pl.col("symbol") == "BBBUSDT")),
            data.filter(future & (pl.col("symbol") == "CCCUSDT")).with_columns(quote_volume=pl.lit(1e12)),
            daily("NEWUSDT", date(2020, 6, 1), date(2020, 12, 31), 1e13),
        ]
    )
    kwargs = {"start": Month(2020, 2), "end": Month(2020, 10), "top_n": 2}
    original, rewritten = build_universe(data, **kwargs), build_universe(altered, **kwargs)
    known = pl.col("rebalance_time") <= cutoff
    assert_frame_equal(original.filter(known), rewritten.filter(known))
    assert not original.filter(~known).equals(rewritten.filter(~known))  # the rewrite does matter later


def test_new_listing_needs_min_days_of_history():
    data = pl.concat(
        [
            daily("AAAUSDT", date(2020, 1, 1), date(2020, 12, 31), 1e6),
            daily("NEWUSDT", date(2020, 3, 10), date(2020, 12, 31), 1e9),
        ]
    )
    universe = build_universe(data, start=Month(2020, 4), end=Month(2020, 5), top_n=1, min_days=25)
    assert members_at(universe, at(2020, 4)) == ["AAAUSDT"]  # 22 days of NEW before 2020-04-01
    assert members_at(universe, at(2020, 5)) == ["NEWUSDT"]


def test_ranking_columns_and_ties():
    data = pl.concat(
        [
            daily("BBBUSDT", date(2020, 1, 1), date(2020, 3, 31), 1e6),
            daily("AAAUSDT", date(2020, 1, 1), date(2020, 3, 31), 1e6),
            daily("CCCUSDT", date(2020, 1, 1), date(2020, 3, 31), 2e6),
        ]
    )
    universe = build_universe(data, start=Month(2020, 3), end=Month(2020, 3), top_n=3)
    assert list(universe.schema.items()) == list(UNIVERSE_SCHEMA.items())
    assert universe["symbol"].to_list() == ["CCCUSDT", "AAAUSDT", "BBBUSDT"]  # tie broken by name
    assert universe["rank"].to_list() == [1, 2, 3]
    assert universe["days"].to_list() == [30, 30, 30]
    assert universe["trailing_quote_volume"].to_list() == [60e6, 30e6, 30e6]


def test_members_at_needs_an_aware_datetime():
    data = daily("AAAUSDT", date(2020, 1, 1), date(2020, 3, 31), 1e6)
    universe = build_universe(data, start=Month(2020, 3), end=Month(2020, 3))
    with pytest.raises(ValueError, match="timezone-aware"):
        members_at(universe, datetime(2020, 3, 15))


# --- End to end ---------------------------------------------------------------------


def test_run_ranks_the_bucket_end_to_end(tmp_path):
    bucket = FakeBucket(page_size=3)
    volumes = {"AAAUSDT": 1e6, "BBBUSDT": 1e5, "USDCUSDT": 1e12, "AAAUPUSDT": 1e11}
    for symbol, volume in volumes.items():
        for month in (Month(2020, 1), Month(2020, 2), Month(2020, 3)):
            bucket.add_month(symbol, "1d", month, month_rows(month, DAY_MS, quote_volume=volume))
    with bucket.client() as client:
        universe = run(data_dir=tmp_path, start=Month(2020, 3), end=Month(2020, 4), top_n=2, client=client, workers=2)
    assert members_at(universe, at(2020, 3)) == ["AAAUSDT", "BBBUSDT"]
    assert members_at(universe, at(2020, 4)) == ["AAAUSDT", "BBBUSDT"]
    saved = pl.read_parquet(universe_path(tmp_path / "parquet", 2))
    assert_frame_equal(saved, universe)

    offline = run(data_dir=tmp_path, start=Month(2020, 3), end=Month(2020, 4), top_n=2, download=False)
    assert_frame_equal(offline, universe)
