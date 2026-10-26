import json
from datetime import UTC, datetime

import polars as pl

from olsprime.data.binance_vision import parse_klines_csv
from olsprime.data.quality import Gap, check_quality
from tests.fakes import HOUR_MS, epoch_ms, kline_rows, klines_csv

START = epoch_ms(2024, 1, 1)


def hourly(rows: list[list[str]]) -> pl.DataFrame:
    return parse_klines_csv(klines_csv(rows), "TESTUSDT")


def rows(count: int = 48) -> list[list[str]]:
    return kline_rows(START, count, HOUR_MS)


def test_clean_series():
    report = check_quality(hourly(rows()), "1h")
    assert report.is_clean
    assert (report.symbol, report.rows) == ("TESTUSDT", 48)
    assert report.first_open_time == datetime(2024, 1, 1, tzinfo=UTC)
    assert report.last_open_time == datetime(2024, 1, 2, 23, tzinfo=UTC)


def test_gaps_are_located_and_counted():
    data = rows()
    del data[10:13]  # 10:00, 11:00 and 12:00 are missing
    del data[30]
    report = check_quality(hourly(data), "1h")
    assert report.gaps == (
        Gap(datetime(2024, 1, 1, 10, tzinfo=UTC), datetime(2024, 1, 1, 13, tzinfo=UTC), 3),
        Gap(datetime(2024, 1, 2, 9, tzinfo=UTC), datetime(2024, 1, 2, 10, tzinfo=UTC), 1),
    )
    assert report.missing_bars == 4
    assert not report.is_clean
    assert report.duplicate_rows == report.non_monotonic == report.ohlc_inconsistent == 0


def test_duplicates_exact_and_conflicting():
    data = rows()
    data.insert(6, list(data[5]))  # exact copy of the 05:00 bar
    conflicting = list(data[20])
    conflicting[4] = "120.50000000"  # same open time, another (still consistent) close
    data.insert(21, conflicting)
    report = check_quality(hourly(data), "1h")
    assert (report.duplicate_rows, report.conflicting_duplicates) == (2, 1)
    assert report.non_monotonic == 0
    assert report.gaps == ()


def test_non_monotonic_times():
    data = rows()
    data[3], data[4] = data[4], data[3]
    report = check_quality(hourly(data), "1h")
    assert report.non_monotonic == 1
    assert (report.duplicate_rows, report.gaps) == (0, ())


def test_inconsistent_ohlc():
    data = rows()
    data[2][2] = "50.00000000"  # high below open and close
    data[3][3] = "200.00000000"  # low above open and close
    data[4][1] = "-1.00000000"  # negative open
    data[5][4] = "NaN"  # close is not a number
    report = check_quality(hourly(data), "1h")
    assert report.ohlc_inconsistent == 4
    assert report.invalid_volume == 0


def test_off_grid_open_and_bad_close_time():
    data = rows()
    data[6][0] = str(int(data[6][0]) + 60_000)  # opens at 06:01
    data[7][6] = str(int(data[7][6]) + 1)  # closes 1 ms late
    report = check_quality(hourly(data), "1h")
    assert (report.off_grid, report.bad_close_time) == (1, 2)
    assert report.gaps == ()


def test_invalid_volume():
    data = rows()
    data[1][5] = "-3.00000000"
    data[2][7] = "NaN"
    report = check_quality(hourly(data), "1h")
    assert report.invalid_volume == 2


def test_report_round_trips_through_json():
    data = rows()
    del data[10]
    report = check_quality(hourly(data), "1h")
    saved = json.loads(json.dumps(report.to_dict()))
    assert saved["gaps"] == [
        {"start": "2024-01-01T10:00:00+00:00", "end": "2024-01-01T11:00:00+00:00", "missing_bars": 1}
    ]
    assert "gaps: 1 (1 missing bars)" in report.summary()
    assert report.summary().isascii()


def test_empty_series():
    report = check_quality(hourly([]), "1h")
    assert report.rows == 0 and report.first_open_time is None
    assert report.summary() == "1h: no rows"
