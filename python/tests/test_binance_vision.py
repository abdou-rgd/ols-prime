import hashlib
import json
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
import pytest

from olsprime.data.binance_vision import (
    ChecksumError,
    Month,
    Status,
    archive_relpath,
    download_klines,
    fetch,
    is_verified,
    month_range,
    month_spans,
    parquet_path,
    parse_checksum,
    parse_klines_csv,
    run,
    to_epoch_ms,
    verify_checksum,
)
from olsprime.data.klines import KLINE_SCHEMA
from tests.fakes import DAY_MS, HOUR_MS, FakeBucket, epoch_ms, kline_rows, klines_csv, month_rows

# First rows of the real BTCUSDT-1d-2024-12 (milliseconds) and BTCUSDT-1d-2025-01 (microseconds).
REAL_MS_ROW = (
    b"1733011200000,96407.99000000,97836.00000000,95693.88000000,97185.18000000,16938.60452000,"
    b"1733097599999,1641327626.60622060,3342200,8114.89569000,786489040.19363780,0\n"
)
REAL_US_ROW = (
    b"1735689600000000,93576.00000000,95151.15000000,92888.00000000,94591.79000000,10373.32613000,"
    b"1735775999999999,975444194.13799830,1516556,5347.73648000,502914035.64059070,0\n"
)


# --- Months -----------------------------------------------------------------------


def test_month_parsing_and_ranges():
    assert Month.parse("2017-08") == Month(2017, 8)
    assert str(Month(2017, 8)) == "2017-08"
    assert Month(2024, 12).shift(1) == Month(2025, 1)
    assert Month(2025, 1).shift(-2) == Month(2024, 11)
    assert month_range(Month(2024, 11), Month(2025, 2)) == [
        Month(2024, 11),
        Month(2024, 12),
        Month(2025, 1),
        Month(2025, 2),
    ]
    assert month_spans([Month(2017, 1), Month(2017, 2), Month(2017, 3), Month(2026, 10)]) == (
        "2017-01..2017-03, 2026-10"
    )
    for bad in ("2017-8", "2017-13", "201708"):
        with pytest.raises(ValueError):
            Month.parse(bad)


# --- Checksums --------------------------------------------------------------------


def test_checksum_good_and_bad():
    data = b"archive bytes"
    digest = hashlib.sha256(data).hexdigest()
    assert parse_checksum(f"{digest}  BTCUSDT-1h-2024-12.zip\n") == digest
    verify_checksum(data, digest, "good.zip")
    with pytest.raises(ChecksumError, match="SHA-256"):
        verify_checksum(b"tampered bytes", digest, "bad.zip")
    with pytest.raises(ChecksumError, match="malformed"):
        parse_checksum("not-a-digest  BTCUSDT-1h-2024-12.zip")


# --- Timestamps and parsing -------------------------------------------------------


def test_timestamp_unit_comes_from_magnitude_not_date():
    # Microseconds in 2024 and milliseconds in 2025: a rule keyed on the 2025-01-01
    # switch would get both wrong.
    t2024, t2025 = epoch_ms(2024, 6, 1), epoch_ms(2025, 6, 1)
    frame = pl.DataFrame({"ts": [t2024 * 1_000, t2025, t2025 * 1_000, t2024]})
    assert frame.select(to_epoch_ms("ts").alias("ts"))["ts"].to_list() == [t2024, t2025, t2025, t2024]


def test_real_rows_in_ms_and_us_parse_to_the_same_layout():
    before = parse_klines_csv(REAL_MS_ROW, "BTCUSDT").row(0, named=True)
    after = parse_klines_csv(REAL_US_ROW, "BTCUSDT").row(0, named=True)
    assert before["open_time"] == datetime(2024, 12, 1, tzinfo=UTC)
    assert before["close_time"] == datetime(2024, 12, 1, 23, 59, 59, 999_000, tzinfo=UTC)
    assert after["open_time"] == datetime(2025, 1, 1, tzinfo=UTC)
    assert after["close_time"] == datetime(2025, 1, 1, 23, 59, 59, 999_000, tzinfo=UTC)
    assert (before["open"], before["close"], before["trades"]) == (96407.99, 97185.18, 3342200)


def test_parse_reads_twelve_headerless_columns():
    rows = kline_rows(epoch_ms(2024, 12, 1), 3, HOUR_MS)
    bars = parse_klines_csv(klines_csv(rows), "BTCUSDT")
    assert list(bars.schema.items()) == list(KLINE_SCHEMA.items())
    assert bars.height == 3
    assert bars["symbol"].unique().to_list() == ["BTCUSDT"]
    assert bars["open_time"].to_list() == [datetime(2024, 12, 1, h, tzinfo=UTC) for h in range(3)]
    first = bars.row(0, named=True)
    assert (first["open"], first["high"], first["low"], first["close"]) == (100.0, 102.0, 99.0, 101.0)
    assert (first["volume"], first["quote_volume"], first["trades"]) == (10.0, 1000.0, 42)


def test_parse_skips_a_header_row():
    rows = kline_rows(epoch_ms(2024, 12, 1), 2, HOUR_MS)
    assert parse_klines_csv(klines_csv(rows, header=True), "BTCUSDT").height == 2


def test_parse_rejects_a_wrong_column_count():
    rows = [row[:11] for row in kline_rows(epoch_ms(2024, 12, 1), 2, HOUR_MS)]
    with pytest.raises(ValueError, match="12 CSV columns"):
        parse_klines_csv(klines_csv(rows), "BTCUSDT")


@pytest.mark.parametrize("value", [1_733_011_200, 1_733_011_200_000_000_000], ids=["seconds", "nanoseconds"])
def test_parse_rejects_other_timestamp_units(value):
    row = kline_rows(epoch_ms(2024, 12, 1), 1, HOUR_MS)[0]
    row[0] = str(value)
    with pytest.raises(ValueError, match="neither in ms nor in us"):
        parse_klines_csv(klines_csv([row]), "BTCUSDT")


def test_parse_of_an_empty_csv_keeps_the_schema():
    assert list(parse_klines_csv(b"", "BTCUSDT").schema.items()) == list(KLINE_SCHEMA.items())


# --- Download and cache -----------------------------------------------------------

NOV, DEC = Month(2024, 11), Month(2024, 12)


def daily_bucket(*months: Month, symbol: str = "BTCUSDT") -> FakeBucket:
    bucket = FakeBucket()
    for month in months:
        bucket.add_month(symbol, "1d", month, month_rows(month, DAY_MS))
    return bucket


def test_download_verifies_then_serves_from_the_cache(tmp_path):
    bucket = daily_bucket(NOV, DEC)
    with bucket.client() as client:
        first = download_klines(["BTCUSDT"], ["1d"], [NOV, DEC], raw_dir=tmp_path, client=client)
        requests_after_first_run = len(bucket.requests)
        second = download_klines(["BTCUSDT"], ["1d"], [NOV, DEC], raw_dir=tmp_path, client=client)
    assert [r.status for r in first] == [Status.DOWNLOADED, Status.DOWNLOADED]
    assert [r.status for r in second] == [Status.CACHED, Status.CACHED]
    assert len(bucket.requests) == requests_after_first_run  # nothing fetched again
    assert all(is_verified(tmp_path / archive_relpath("BTCUSDT", "1d", m)) for m in (NOV, DEC))


def test_bad_checksum_fails_and_leaves_nothing_in_the_cache(tmp_path):
    bucket = FakeBucket()
    bucket.add_month("BTCUSDT", "1d", DEC, month_rows(DEC, DAY_MS), bad_checksum=True)
    with bucket.client() as client:
        [result] = download_klines(["BTCUSDT"], ["1d"], [DEC], raw_dir=tmp_path, client=client)
    assert result.status is Status.FAILED
    assert "SHA-256" in result.error
    assert not list(tmp_path.rglob("*.zip*"))  # no archive, no CHECKSUM, no partial file


def test_missing_month_is_reported_not_raised(tmp_path):
    bucket = daily_bucket(DEC)
    with bucket.client() as client:
        results = download_klines(["BTCUSDT"], ["1d"], [NOV, DEC], raw_dir=tmp_path, client=client)
    assert [r.status for r in results] == [Status.MISSING, Status.DOWNLOADED]


def test_tampered_cache_is_downloaded_again(tmp_path):
    bucket = daily_bucket(DEC)
    cached = tmp_path / archive_relpath("BTCUSDT", "1d", DEC)
    with bucket.client() as client:
        download_klines(["BTCUSDT"], ["1d"], [DEC], raw_dir=tmp_path, client=client)
        cached.write_bytes(b"corrupted on disk")
        assert not is_verified(cached)
        [result] = download_klines(["BTCUSDT"], ["1d"], [DEC], raw_dir=tmp_path, client=client)
    assert result.status is Status.DOWNLOADED
    assert is_verified(cached)


def test_fetch_retries_server_errors_but_not_client_errors():
    answers = iter([httpx.Response(503), httpx.Response(200, content=b"ok")])
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(answers))) as client:
        assert fetch(client, "https://data.binance.vision/x", backoff=0) == b"ok"

    calls = []

    def forbidden(request):
        calls.append(request)
        return httpx.Response(403)

    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            fetch(client, "https://data.binance.vision/x", backoff=0)
    assert len(calls) == 1


# --- Parquet --------------------------------------------------------------------


def test_parquet_is_continuous_across_the_microsecond_switch(tmp_path):
    jan = Month(2025, 1)
    bucket = FakeBucket()
    bucket.add_month("BTCUSDT", "1h", DEC, month_rows(DEC, HOUR_MS, unit="ms"))
    bucket.add_month("BTCUSDT", "1h", jan, month_rows(jan, HOUR_MS, unit="us"))
    with bucket.client() as client:
        _, reports = run(["BTCUSDT"], ["1h"], DEC, jan, data_dir=tmp_path, client=client)

    path = parquet_path(tmp_path / "parquet", "BTCUSDT", "1h")
    bars = pl.read_parquet(path)
    assert list(bars.schema.items()) == list(KLINE_SCHEMA.items())
    assert bars.height == 62 * 24
    assert bars["open_time"].diff().drop_nulls().unique().to_list() == [timedelta(hours=1)]
    assert bars["open_time"][744] == datetime(2025, 1, 1, tzinfo=UTC)
    assert reports[("BTCUSDT", "1h")].is_clean
    saved = json.loads(path.with_name("BTCUSDT.quality.json").read_text())
    assert (saved["rows"], saved["missing_bars"]) == (62 * 24, 0)


def test_a_narrower_run_does_not_truncate_the_parquet(tmp_path):
    oct_ = Month(2024, 10)
    bucket = daily_bucket(oct_, NOV, DEC)
    with bucket.client() as client:
        run(["BTCUSDT"], ["1d"], oct_, DEC, data_dir=tmp_path, client=client)
        results, _ = run(["BTCUSDT"], ["1d"], DEC, DEC, data_dir=tmp_path, client=client)
    assert [r.status for r in results] == [Status.CACHED]
    assert pl.read_parquet(parquet_path(tmp_path / "parquet", "BTCUSDT", "1d")).height == 31 + 30 + 31


def test_duplicates_are_reported_then_dropped(tmp_path):
    # The same bar in two archives: the report sees it, the Parquet file keeps one copy.
    bucket = FakeBucket()
    nov_rows = month_rows(NOV, DAY_MS)
    dec_rows = month_rows(DEC, DAY_MS)
    bucket.add_month("BTCUSDT", "1d", NOV, nov_rows)
    bucket.add_month("BTCUSDT", "1d", DEC, [nov_rows[-1], *dec_rows])
    with bucket.client() as client:
        _, reports = run(["BTCUSDT"], ["1d"], NOV, DEC, data_dir=tmp_path, client=client)
    report = reports[("BTCUSDT", "1d")]
    assert (report.duplicate_rows, report.conflicting_duplicates) == (1, 0)
    bars = pl.read_parquet(parquet_path(tmp_path / "parquet", "BTCUSDT", "1d"))
    assert bars.height == 30 + 31
    assert bars["open_time"].is_unique().all() and bars["open_time"].is_sorted()
