"""The vault lock: splits, seal, load_klines, and the guards that keep readers behind it."""

import ast
import json
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from olsprime import protocol, universe
from olsprime.data import quality
from olsprime.data.binance_vision import (
    Month,
    archive_relpath,
    build_parquet,
    cached_archives,
    download_klines,
    parquet_path,
    parse_klines_csv,
)
from olsprime.data.klines import TIMESTAMP
from olsprime.paths import REPO_ROOT, runs_dir
from olsprime.protocol import (
    DEVELOPMENT,
    VALIDATION,
    VAULT,
    Period,
    VaultSealed,
    load_klines,
    seal,
    vault_open,
)
from tests.allowlist import ALLOWED
from tests.fakes import DAY_MS, HOUR_MS, FakeBucket, kline_rows, klines_csv, month_rows

PACKAGE = Path(protocol.__file__).parent
ARROW = "→"


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def bars(first: datetime, count: int, step_ms: int, symbol: str = "TESTUSDT") -> pl.DataFrame:
    """Consistent bars in the stored schema, from ``first``."""
    start_ms = int(first.timestamp()) * 1000
    return parse_klines_csv(klines_csv(kline_rows(start_ms, count, step_ms)), symbol)


def store(data_dir: Path, frame: pl.DataFrame, symbol: str, interval: str) -> None:
    path = parquet_path(data_dir / "parquet", symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def open_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    key = runs_dir() / "vault" / "OPEN"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text("2026-11-01\nfinal models: test\n")
    monkeypatch.setenv("OLSPRIME_OPEN_VAULT", "1")


def access_lines() -> list[str]:
    log = runs_dir() / "vault" / "ACCESS.log"
    return log.read_text().splitlines() if log.exists() else []


# --- Splits -------------------------------------------------------------------------


def test_periods_are_half_open_utc_and_contiguous():
    assert DEVELOPMENT.end == VALIDATION.start and VALIDATION.end == VAULT.start
    assert utc(2025, 12, 31, 23) in VALIDATION and VAULT.start not in VALIDATION
    assert VAULT.start in VAULT and VAULT.end not in VAULT
    with pytest.raises(ValueError, match="UTC"):
        Period(datetime(2020, 1, 1), utc(2020, 2, 1))
    with pytest.raises(ValueError, match="empty"):
        Period(utc(2020, 2, 1), utc(2020, 1, 1))


def test_period_accepts_only_the_two_true_utc_zones():
    stdlib = Period(datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2020, 2, 1, tzinfo=timezone.utc))
    zone_info = Period(datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")), datetime(2020, 2, 1, tzinfo=ZoneInfo("UTC")))
    assert stdlib.start == zone_info.start and stdlib.end == zone_info.end
    assert datetime(2020, 1, 15, tzinfo=ZoneInfo("UTC")) in stdlib  # the two forms compare as one instant
    # Europe/London has offset 0 in January and +1 h in summer: it is not UTC.
    london = ZoneInfo("Europe/London")
    assert datetime(2020, 1, 1, tzinfo=london).utcoffset() == timedelta(0)
    with pytest.raises(ValueError, match="UTC"):
        Period(datetime(2020, 1, 1, tzinfo=london), utc(2020, 2, 1))
    with pytest.raises(ValueError, match="UTC"):
        Period(utc(2020, 1, 1), datetime(2020, 2, 1, tzinfo=london))
    with pytest.raises(ValueError, match="UTC"):
        Period(datetime(2020, 1, 1), utc(2020, 2, 1))  # naive
    with pytest.raises(ValueError, match="UTC"):
        Period(datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=1))), utc(2020, 2, 1))


def test_protocol_table_names_the_months_of_the_code():
    text = (REPO_ROOT / "docs" / "PROTOCOL.md").read_text(encoding="utf-8")
    pattern = rf"^\|\s*(\d{{4}}-\d{{2}}) {ARROW} (\d{{4}}-\d{{2}})\s*\|\s*(\w+)"
    table = {use: (Month.parse(a), Month.parse(b)) for a, b, use in re.findall(pattern, text, re.M)}
    assert set(table) == {"development", "validation", "vault"}
    for name, period in (("development", DEVELOPMENT), ("validation", VALIDATION), ("vault", VAULT)):
        # The table's last month is inclusive, the period's end is exclusive.
        assert table[name] == (Month.of(period.start), Month.of(period.end).shift(-1)), name


# --- seal ---------------------------------------------------------------------------


@pytest.mark.parametrize(("step_ms", "last_kept"), [(HOUR_MS, utc(2025, 12, 31, 23)), (DAY_MS, utc(2025, 12, 31))])
def test_seal_keeps_the_last_bar_before_the_vault_and_drops_the_first_inside(step_ms, last_kept):
    first = last_kept - timedelta(milliseconds=2 * step_ms)
    frame = bars(first, 5, step_ms)  # two bars before, the last kept, and two in the vault
    assert VAULT.start in frame["open_time"].to_list()
    sealed = seal(frame, caller="test")
    assert sealed["open_time"].max() == last_kept
    assert sealed.height == 3
    assert access_lines() == []  # a sealed read leaves no trace and creates nothing
    assert not (runs_dir() / "vault").exists()


def test_seal_refuses_a_time_column_that_is_not_a_utc_millisecond_datetime():
    frame = bars(utc(2025, 12, 31), 3, HOUR_MS)
    as_int = frame.with_columns(pl.col("open_time").dt.epoch("ms"))
    with pytest.raises(TypeError, match="open_time"):
        seal(as_int, caller="test")
    with pytest.raises(TypeError):
        seal(frame.with_columns(pl.col("open_time").dt.to_string("%Y-%m-%d")), caller="test")
    with pytest.raises(TypeError):
        seal(frame.with_columns(pl.col("open_time").cast(pl.Datetime("us", "UTC"))), caller="test")
    with pytest.raises(TypeError):
        seal(frame.with_columns(pl.col("open_time").dt.replace_time_zone(None)), caller="test")
    with pytest.raises(TypeError):
        seal(frame, time_col="missing", caller="test")


def test_neither_key_alone_opens_the_vault(monkeypatch):
    frame = bars(utc(2025, 12, 31, 22), 6, HOUR_MS)  # 2 before the vault, 4 inside
    monkeypatch.setenv("OLSPRIME_OPEN_VAULT", "1")  # variable alone
    assert not vault_open() and seal(frame, caller="test").height == 2
    monkeypatch.delenv("OLSPRIME_OPEN_VAULT")
    key = runs_dir() / "vault" / "OPEN"
    key.parent.mkdir(parents=True)
    key.write_text("file alone\n")  # file alone
    assert not vault_open() and seal(frame, caller="test").height == 2
    monkeypatch.setenv("OLSPRIME_OPEN_VAULT", "true")  # wrong value
    assert not vault_open()
    assert access_lines() == []


def test_with_both_keys_every_row_comes_back_and_each_read_is_logged(monkeypatch):
    frame = bars(utc(2025, 12, 31, 22), 6, HOUR_MS)
    open_vault(monkeypatch)
    assert vault_open()
    assert seal(frame, caller="first reader").height == 6
    assert len(access_lines()) == 1
    assert seal(frame.head(2), caller="second\nreader").height == 2
    lines = access_lines()
    assert len(lines) == 2
    when, caller, rows, vault_rows = lines[0].split("\t")
    assert datetime.fromisoformat(when).tzinfo is not None
    assert (caller, rows, vault_rows) == ("first reader", "rows=6", "vault_rows=4")
    assert lines[1].split("\t")[1:] == ["second reader", "rows=2", "vault_rows=0"]


def test_the_vault_key_file_must_start_with_a_date(monkeypatch):
    frame = bars(utc(2025, 12, 31, 22), 6, HOUR_MS)  # 2 before the vault, 4 inside
    key = runs_dir() / "vault" / "OPEN"
    key.parent.mkdir(parents=True)
    # Environment key unset: sealed, whatever the file holds, and nothing raises.
    for content in ("", "not a date\n", "2026-11-01\n"):
        key.write_text(content)
        assert not vault_open() and seal(frame, caller="test").height == 2
    # Environment key set: the first line must be an ISO date.
    monkeypatch.setenv("OLSPRIME_OPEN_VAULT", "1")
    for content in ("", "\n", "\n2026-11-01\n", "final models: test\n", "2026-11-01, final models\n", "2026-13-45\n"):
        key.write_text(content)
        with pytest.raises(VaultSealed, match="OPEN file has no date"):
            vault_open()
        with pytest.raises(VaultSealed, match="OPEN file has no date"):
            seal(frame, caller="test")  # a bad file never opens the vault, and never falls back silently
    key.write_bytes(b"\xff\xfe\x00 not utf-8")
    with pytest.raises(VaultSealed, match="no date"):
        vault_open()
    for content in ("2026-11-01\n", "2026-11-01", "2026-11-01\nfinal models: a, b\n", " 2026-11-01 \n"):
        key.write_text(content)
        assert vault_open() and seal(frame, caller="test").height == 6
    assert len(access_lines()) == 4
    key.unlink()
    assert not vault_open()  # the variable alone is still not enough


# --- load_klines --------------------------------------------------------------------


def test_load_klines_seals_and_adds_known_at(tmp_path):
    frame = bars(utc(2025, 12, 31, 20), 8, HOUR_MS)  # 4 before the vault, 4 inside
    store(tmp_path, frame, "TESTUSDT", "1h")
    loaded = load_klines("TESTUSDT", "1h", data_dir=tmp_path)
    assert loaded.height == 4 and loaded["open_time"].max() == utc(2025, 12, 31, 23)
    assert loaded.columns == [*frame.columns, "known_at"]
    assert loaded.schema["known_at"] == TIMESTAMP
    assert (loaded["known_at"] == loaded["open_time"] + timedelta(hours=1, milliseconds=-1)).all()


def test_load_klines_with_a_column_subset_still_seals(tmp_path):
    store(tmp_path, bars(utc(2025, 12, 31, 20), 8, HOUR_MS), "TESTUSDT", "1h")
    only_close = load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["close"])
    assert only_close.columns == ["close"] and only_close.height == 4  # no known_at unless asked for
    both = load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["known_at", "close"])
    assert both.columns == ["known_at", "close"] and both.height == 4
    with_time = load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["open_time", "symbol"])
    assert with_time.columns == ["open_time", "symbol"] and with_time["open_time"].max() == utc(2025, 12, 31, 23)


def test_seal_cuts_on_when_the_bar_is_final_not_when_it_opens(tmp_path):
    week = 7 * DAY_MS
    # Weekly bars open on Mondays: 2025-12-15 and 2025-12-22 end before the vault; 2025-12-29 carries 4 vault days.
    store(tmp_path, bars(utc(2025, 12, 15), 4, week), "TESTUSDT", "1w")
    kept = load_klines("TESTUSDT", "1w", data_dir=tmp_path, columns=["open_time", "known_at"])
    assert kept["open_time"].to_list() == [utc(2025, 12, 15), utc(2025, 12, 22)]
    assert kept["known_at"].max() < VAULT.start
    # The 1 h bar of 2025-12-31 23:00 and the 1 d bar of 2025-12-31 are final just before the vault.
    store(tmp_path, bars(utc(2025, 12, 31, 22), 4, HOUR_MS), "TESTUSDT", "1h")
    store(tmp_path, bars(utc(2025, 12, 30), 4, DAY_MS), "TESTUSDT", "1d")
    assert load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["open_time"])["open_time"].to_list() == [
        utc(2025, 12, 31, 22),
        utc(2025, 12, 31, 23),
    ]
    assert load_klines("TESTUSDT", "1d", data_dir=tmp_path, columns=["open_time"])["open_time"].max() == utc(2025, 12, 31)


def test_a_bad_close_time_cannot_smuggle_a_vault_bar_out(tmp_path):
    frame = bars(utc(2025, 12, 31, 22), 2, HOUR_MS).with_columns(
        close_time=pl.Series([utc(2025, 12, 31, 22, 59, 59, 999_000), utc(2026, 1, 1, 0, 30)]).dt.cast_time_unit("ms").dt.replace_time_zone("UTC")
    )
    store(tmp_path, frame, "TESTUSDT", "1h")
    # The last bar opens before the vault but closes inside it: sealed.
    assert load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["open_time"]).height == 1


def test_load_klines_opened_returns_the_vault_and_logs_the_caller(tmp_path, monkeypatch):
    store(tmp_path, bars(utc(2025, 12, 31, 20), 8, HOUR_MS), "TESTUSDT", "1h")
    open_vault(monkeypatch)
    assert load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["close"], caller="final").height == 8
    (line,) = access_lines()
    assert line.split("\t")[1:] == ["final", "rows=8", "vault_rows=4"]


def test_known_at_is_never_before_the_bar_is_final(tmp_path):
    t = utc(2020, 6, 1, 10)
    off_grid = utc(2018, 2, 11, 3, 28)
    opens = [t, t + timedelta(hours=1), t + timedelta(hours=2), t + timedelta(hours=3), off_grid]
    closes = [
        opens[0] + timedelta(hours=1, milliseconds=-1),  # on the grid
        opens[1] + timedelta(minutes=30),  # closes early
        opens[2] - timedelta(minutes=12),  # closes before it opens
        opens[3] + timedelta(hours=1),  # closes 1 ms late
        utc(2018, 2, 11, 4, 0, 0, 999_000),  # off-grid bar, close earlier than open + 1 h - 1 ms
    ]
    frame = pl.DataFrame({"symbol": "T", "open_time": opens, "close_time": closes, "close": 1.0}).with_columns(
        pl.col("open_time", "close_time").cast(TIMESTAMP)
    )
    store(tmp_path, frame, "TESTUSDT", "1h")
    loaded = load_klines("TESTUSDT", "1h", data_dir=tmp_path, columns=["open_time", "known_at"])
    assert loaded["known_at"].to_list() == [
        opens[0] + timedelta(hours=1, milliseconds=-1),
        opens[1] + timedelta(hours=1, milliseconds=-1),
        opens[2] + timedelta(hours=1, milliseconds=-1),
        opens[3] + timedelta(hours=1),
        off_grid + timedelta(hours=1, milliseconds=-1),
    ]
    assert (loaded["known_at"] >= loaded["open_time"] + timedelta(hours=1, milliseconds=-1)).all()


def test_build_parquet_leaves_the_vault_months_out_and_says_so(tmp_path, caplog):
    # The downloader refuses a vault month, so the January archive is put in the cache by hand (as a
    # store from before the physical seal has it).
    bucket = FakeBucket()
    dec, jan = Month(2025, 12), Month(2026, 1)
    for month in (dec, jan):
        bucket.add_month("BTCUSDT", "1d", month, month_rows(month, DAY_MS))
    with bucket.client() as client:
        download_klines(["BTCUSDT"], ["1d"], [dec], raw_dir=tmp_path / "raw", client=client)
    jan_zip = tmp_path / "raw" / archive_relpath("BTCUSDT", "1d", jan)
    jan_zip.parent.mkdir(parents=True, exist_ok=True)
    key = "data/" + archive_relpath("BTCUSDT", "1d", jan)
    jan_zip.write_bytes(bucket.objects[key])
    jan_zip.with_name(jan_zip.name + ".CHECKSUM").write_bytes(bucket.objects[key + ".CHECKSUM"])
    assert len(cached_archives(tmp_path / "raw", "BTCUSDT", "1d")) == 2
    with caplog.at_level("WARNING"):
        report = build_parquet("BTCUSDT", "1d", raw_dir=tmp_path / "raw", parquet_dir=tmp_path / "parquet")
    assert any("BTCUSDT-1d-2026-01.zip" in r.getMessage() and "vault" in r.getMessage() for r in caplog.records)
    assert report.rows == 31 and report.is_clean
    path = parquet_path(tmp_path / "parquet", "BTCUSDT", "1d")
    assert json.loads(path.with_name("BTCUSDT.quality.json").read_text())["rows"] == 31
    assert protocol.stored_row_count(path) == 31  # no vault row in the table
    assert pl.read_parquet(path)["open_time"].max() < VAULT.start  # the file itself, unsealed, holds none either
    assert load_klines("BTCUSDT", "1d", data_dir=tmp_path).height == 31


# --- Readers ------------------------------------------------------------------------


def test_quality_cli_reports_on_the_sealed_frame(tmp_path, capsys):
    store(tmp_path, bars(utc(2025, 12, 30), 5, DAY_MS), "TESTUSDT", "1d")  # 2 before the vault, 3 inside
    path = parquet_path(tmp_path / "parquet", "TESTUSDT", "1d")
    assert quality.main([str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == 2 and report["last_open_time"].startswith("2025-12-31")
    assert report["bad_close_time"] == 0


def daily_rows(first: datetime, last: datetime, symbol: str, quote_volume: float = 1e6) -> pl.DataFrame:
    days = pl.datetime_range(first, last, "1d", time_unit="ms", time_zone="UTC", eager=True)
    return pl.DataFrame({"symbol": symbol, "open_time": days, "quote_volume": quote_volume})


def test_build_universe_refuses_the_vault_months_while_sealed(monkeypatch):
    data = daily_rows(utc(2025, 10, 1), utc(2026, 2, 1), "AAAUSDT")
    ok = universe.build_universe(data, start=Month(2025, 11), end=Month(2025, 12), top_n=1)
    assert ok.height == 2
    with pytest.raises(VaultSealed):
        universe.build_universe(data, start=Month(2025, 12), end=Month(2026, 1), top_n=1)
    with pytest.raises(VaultSealed):
        universe.build_universe(data, start=Month(2026, 2), end=Month(2026, 2), top_n=1)
    open_vault(monkeypatch)
    opened = universe.build_universe(data, start=Month(2025, 12), end=Month(2026, 1), top_n=1)
    assert opened["rebalance_time"].to_list() == [utc(2025, 12, 1), utc(2026, 1, 1)]


def test_universe_run_reads_sealed_and_refuses_before_downloading(tmp_path, monkeypatch):
    for symbol in ("AAAUSDT", "BBBUSDT"):
        store(tmp_path, bars(utc(2025, 10, 1), 140, DAY_MS, symbol), symbol, "1d")  # to 2026-02-16
    seen = []
    real = universe.build_universe
    monkeypatch.setattr(universe, "build_universe", lambda daily, **kw: seen.append(daily) or real(daily, **kw))
    result = universe.run(data_dir=tmp_path, start=Month(2025, 11), end=Month(2025, 12), top_n=2, download=False)
    assert result.height == 4
    assert seen[0]["open_time"].max() == utc(2025, 12, 31) and seen[0]["symbol"].n_unique() == 2
    # download=True with no client: the guard must fire before any network attempt.
    with pytest.raises(VaultSealed):
        universe.run(data_dir=tmp_path, start=Month(2025, 12), end=Month(2026, 1), download=True)


def test_universe_cli_ends_before_the_vault_by_default(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return pl.DataFrame(schema=universe.UNIVERSE_SCHEMA)

    monkeypatch.setattr(universe, "run", fake_run)
    assert universe.main(["--start", "2017-10", "--data-dir", str(tmp_path), "--no-download"]) == 0
    assert captured["end"] == Month(2025, 12)


def test_seal_defeats_a_forged_time_column_when_the_frame_has_open_time():
    frame = bars(utc(2025, 12, 31, 22), 6, HOUR_MS)  # 2 before the vault, 4 inside
    forged = frame.with_columns(known_at=pl.lit(utc(2020, 1, 1)).cast(TIMESTAMP))
    assert seal(forged, time_col="known_at", caller="test").height == 2  # the vault bars do not walk out
    assert seal(forged, time_col="known_at", caller="test")["open_time"].max() == utc(2025, 12, 31, 23)
    # Without open_time there is nothing to cross-check: that is the limit of this defence,
    # which is why only load_klines (it computes known_at itself) may call seal.
    assert seal(forged.drop("open_time"), time_col="known_at", caller="test").height == 6
    # open_time must itself be a UTC millisecond datetime, or the cross-check could be skipped.
    with pytest.raises(TypeError, match="open_time"):
        seal(forged.with_columns(pl.col("open_time").dt.epoch("ms")), time_col="known_at", caller="test")
    with pytest.raises(TypeError, match="open_time"):
        seal(forged.with_columns(pl.col("open_time").dt.to_string("%Y")), time_col="known_at", caller="test")


@pytest.mark.parametrize(("interval", "step_ms"), [("1h", HOUR_MS), ("1d", DAY_MS), ("1w", 7 * DAY_MS)])
def test_the_open_time_cross_check_never_drops_what_known_at_keeps(interval, step_ms):
    frame = bars(VAULT.start - timedelta(milliseconds=15 * step_ms), 30, step_ms)  # half before the vault, half inside
    frame = frame.with_columns(protocol.known_at_expr(interval))
    by_known_at_only = frame.filter(pl.col("known_at") < VAULT.start)
    assert 0 < by_known_at_only.height < frame.height
    assert seal(frame, time_col="known_at", caller="test").equals(by_known_at_only)
    assert (by_known_at_only["open_time"] < VAULT.start).all()


# --- Static guards ------------------------------------------------------------------
#
# A name filter on the syntax tree. It stops carelessness, not intent: a determined
# person can always read a file by a route that is not on the list. What it guarantees is
# that every route found so far, and every module that exists, goes through load_klines.

PYTHON_ROOT = PACKAGE.parent  # python/

# Names (calls, attributes, imports) that read stored tables or build frames from outside.
READERS = frozenset(
    {
        "read_parquet", "scan_parquet", "read_table", "ParquetFile", "read_ipc", "scan_ipc", "read_ipc_stream",
        "read_delta", "scan_delta", "read_database", "scan_pyarrow_dataset", "from_pandas", "from_arrow",
        "deserialize",
        # added in iteration 3: wrapped by the tripwire, missing here until the vault-tree work compared the two lists
        "read_database_uri", "read_ipc_schema", "read_parquet_metadata", "read_parquet_schema", "scan_arrow_c_stream",
        "scan_external_reader", "scan_iceberg", "scan_lance",
    }
)  # fmt: skip
# Names that read raw text or archives (the downloaded cache).
RAW_READERS = frozenset(
    {
        "read_csv", "scan_csv", "read_csv_batched", "read_klines_zip", "ZipFile", "genfromtxt", "loadtxt",
        "read_json", "read_ndjson", "scan_ndjson", "read_avro", "read_excel", "read_lines", "scan_lines", "read_ods",
    }
)
# Names that load code or attributes by string, which no name filter can follow.
DYNAMIC_NAMES = frozenset(
    {"import_module", "__import__", "__builtins__", "attrgetter", "methodcaller", "__getattribute__"}
)  # a name or an attribute reached by a string, past every name filter
# ``seal`` is protocol's own lock: other code reaches data through load_klines, which computes ``known_at`` itself.
SEAL = "seal"
BANNED_MODULES = (
    "pyarrow", "fastparquet", "duckdb", "pandas", "csv", "zipfile", "importlib",
    "subprocess",  # shelling out to parquet-tools, duckdb, ...
    "pickle", "joblib", "shelve", "marshal",  # a cached object holding vault rows
    "runpy",  # runs a file by path, outside the static walk and the allow-list
    "zipimport",  # zipimporter(archive).get_data(member) reads a zip member without zipfile
    "gc", "ctypes",  # gc.get_referents finds the originals behind a wrapper, ctypes rewrites memory
)  # fmt: skip
DATA_LIBRARIES = ("polars", "pyarrow", "pandas", "numpy", "duckdb")  # getattr/vars on these is dynamic access
FILE_SUFFIXES = (".parquet", ".zip")
# The guards' own handles. Reaching them from a test is how a test would switch them off.
DISARM_ATTRIBUTES = frozenset(
    {
        "__globals__", "__wrapped__", "__closure__", "__bases__", "__base__", "__mro__", "__subclasses__",
        "__code__", "__defaults__", "__kwdefaults__", "__func__",  # the original behind a wrapper
        "undo", "reload",  # MonkeyPatch.undo(), importlib.reload()
    }
)  # fmt: skip
# Test files that hold the seal allowances. A helper imported from one of them puts that file on the stack of
# the call, and the loaders' exemption for files that may open a fake vault would cover the caller too.
EXEMPT_TESTS = frozenset({"test_tripwire", "test_protocol", "test_manifest", "test_vault"})
# What other tests may take from them: data builders and readers of a log, nothing that runs a loader.
SAFE_TEST_HELPERS = frozenset({"bars", "store", "utc", "access_lines", "open_vault"})  # open_vault: a seal-patch of its own
GUARD_MODULES = frozenset({"conftest", "tripwire", "allowlist"})  # tests/conftest.py, tests/tripwire.py, tests/allowlist.py
# Keys of the vault. Forging both opens it: ``open_vault`` (test_protocol), ``vault_key_file`` and ``OPEN_ENV``
# (protocol), and the variable's name as a string, spelled out or built by concatenation.
VAULT_KEY_NAMES = frozenset({"open_vault", "vault_key_file", "OPEN_ENV"})
VAULT_KEY_STRING = "OLSPRIME_OPEN_VAULT"
# The vault tree of the physical seal (iteration 3), outside data/. Only olsprime/vault.py and olsprime/protocol.py
# reach it (plus the files that define, isolate or judge it, by the allow-list): the names that find its root, its files
# and its identity, and the strings that spell the two variables and the default folder.
VAULT_TREE_NAMES = frozenset(
    {"vault_dir", "default_vault_dir", "VAULT_DIR_ENV", "IDENTITY_ENV", "vault_tree_klines_path", "read_identity", "decrypt_vault_bytes"}
)
# "OLSPRIME_VAULT_" is the prefix of both variables: it also catches an f-string or a concatenation that spells the
# rest from a variable (f"OLSPRIME_VAULT_{name}"), where only the first fragment is a literal.
VAULT_TREE_STRINGS = ("OLSPRIME_VAULT_", "ols-prime-vault")
# Code run from a string: the bare builtins only, since ``re.compile`` and ``expr.list.eval`` are ordinary
# (olsprime/universe.py compiles a regular expression). ``builtins.eval`` is caught through MODULE_ATTRIBUTES.
EXEC_BUILTINS = frozenset({"eval", "exec", "compile"})
# Attributes of a standard or numeric module that read a binary file, run a shell or copy data away,
# by the rule that flags them. Matched on the attribute name AND the module alias it hangs on, so that
# ``copy.copy(x)`` and ``df.copy()`` are not shutil.
MODULE_ATTRIBUTES: dict[str, tuple[str, frozenset[str]]] = {
    "os": ("shell", frozenset({"system", "popen"})),
    "numpy": ("binary-load", frozenset({"load", "fromfile", "memmap"})),
    "shutil": ("copy", frozenset({"copy", "copyfile", "copy2", "copytree"})),
    "builtins": ("exec", EXEC_BUILTINS),
}
# Notebooks: a line that starts with ``%`` or ``!`` is IPython, not Python. It is blanked for the parser
# but still scanned for these data tokens, so that ``!parquet-tools cat x.parquet`` does not slip through.
MAGIC_DATA_TOKENS = re.compile(r"parquet|\.zip\b|duckdb|\.ipc\b|\.arrow\b|vault", re.I)

# Where a rule is allowed, by file: tests/allowlist.py (shared with the run-time tripwire, tests/tripwire.py).


def _root_name(node: ast.AST) -> str | None:
    """The leftmost name of ``a.b(c)[d]``: ``a``."""
    while isinstance(node, ast.Attribute | ast.Call | ast.Subscript):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _library_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to polars, pyarrow, pandas, numpy or duckdb, or to something inside them."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name.split(".")[0] for a in node.names if a.name.split(".")[0] in DATA_LIBRARIES}
        elif isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] in DATA_LIBRARIES:
            aliases |= {a.asname or a.name for a in node.names}
    return aliases


def _literal_data_file(call: ast.Call) -> bool:
    """``open('x.parquet')``, ``io.open('x.zip')``, ``Path('x.parquet').open()``, ``.read_bytes()``."""
    func = call.func
    target = None
    if (isinstance(func, ast.Name) and func.id == "open") or (isinstance(func, ast.Attribute) and func.attr == "open"):
        target = call
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):  # Path('x.zip').open()
            target = func.value
    if isinstance(func, ast.Attribute) and func.attr in ("read_bytes", "read_text") and isinstance(func.value, ast.Call):
        target = func.value
    if target is None or not target.args:
        return False
    first = target.args[0]
    return isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.lower().endswith(FILE_SUFFIXES)


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """``{local name: module}`` for the modules of MODULE_ATTRIBUTES, from ``import os as o``."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in MODULE_ATTRIBUTES:
                    bound[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
    return bound


def _data_package_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to ``olsprime``, ``olsprime.data`` or a module of it (``binance_vision``, ``survey``, ...).

    ``getattr(binance_vision, "read_" + "klines_zip")`` reaches an unsealed reader by a string, like
    ``getattr(pl, ...)`` on a data library.
    """
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "olsprime":
                    bound.add(a.asname or "olsprime")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # from . import binance_vision, from .data import survey
                bound |= {a.asname or a.name for a in node.names}
            elif module == "olsprime":
                bound |= {a.asname or a.name for a in node.names if a.name == "data"}
            elif module == "olsprime.data" or module.startswith("olsprime.data."):
                bound |= {a.asname or a.name for a in node.names}
    return bound


def _call_functions(tree: ast.AST) -> set[int]:
    """``id`` of every node that is called (``f`` in ``f(x)``): a name that is not in it is passed as a value."""
    return {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}


# What olsprime.protocol holds that nobody may rebind: patching any of them switches the seal off.
PROTOCOL_NAMES = frozenset({"seal", "vault_open", "VAULT", "load_klines", "load_klines_file", "load_universe"})
PATCHERS = frozenset({"setattr", "delattr", "patch"})  # setattr(...), monkeypatch.setattr, mock.patch, patch(...)
SYS_PATH_MUTATORS = frozenset({"insert", "append", "extend"})


def _bound_names(tree: ast.AST, module: str) -> set[str]:
    """Local names bound to ``module`` (``sys``) or to a submodule that was imported from a package.

    ``import sys``, ``import sys as s``, ``from olsprime import protocol as p``, ``import olsprime.protocol as p``
    and the relative ``from . import protocol`` bind a name.
    """
    package, _, leaf = module.rpartition(".")
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module and (a.asname or not package):
                    bound.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and package:
            from_package = (node.module or "") == package or (node.level and (node.module or "") in ("", package))
            if from_package:
                bound |= {a.asname or a.name for a in node.names if a.name == leaf}
    return bound


def _is_protocol(node: ast.AST, protocol_names: set[str], olsprime_names: set[str]) -> bool:
    """Whether ``node`` is the module olsprime.protocol: a name bound to it, or ``olsprime.protocol`` spelled out."""
    if isinstance(node, ast.Name):
        return node.id in protocol_names
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "protocol"
        and isinstance(node.value, ast.Name)
        and node.value.id in olsprime_names
    )


def _text(node: ast.AST) -> str | None:
    """The string a literal, a concatenation of literals or an f-string of literals spells; ``None`` if unknown."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _text(node.left), _text(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = [_text(v) for v in node.values]
        return None if None in parts else "".join(parts)  # type: ignore[arg-type]
    return None


def _patcher(call: ast.Call) -> str | None:
    """``setattr``, ``delattr`` or ``patch`` for ``setattr(...)``, ``mp.setattr(...)``, ``mock.patch(...)``, ``patch.object(...)``."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "object":  # mock.patch.object, patch.object
        inner = func.value
        inner_name = inner.id if isinstance(inner, ast.Name) else inner.attr if isinstance(inner, ast.Attribute) else None
        return "patch.object" if inner_name == "patch" else None
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    return name if name in PATCHERS else None


def _extra_findings(tree: ast.AST, add) -> None:
    """Rules added after audit A: ``sys.modules``, ``sys.path`` and patching of the seal."""
    sys_names = _bound_names(tree, "sys")
    protocol_names = _bound_names(tree, "olsprime.protocol")
    olsprime_names = {"olsprime"}  # ``import olsprime`` / ``import olsprime.protocol`` bind this name
    olsprime_names |= {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.name == "olsprime" and a.asname}

    def is_sys(node: ast.AST, attribute: str) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == attribute and _root_name(node.value) in sys_names

    for node in ast.walk(tree):
        # sys.modules[...] and sys.modules.get(...) reach a module by a string, past the name filter.
        if is_sys(node, "modules"):
            add("dynamic", node, "sys.modules reaches a module by its name")
        if isinstance(node, ast.ImportFrom) and not node.level and node.module == "sys":
            for a in node.names:
                if a.name in ("modules", "path"):
                    add("dynamic", node, f"from sys import {a.name}")
        # sys.path edits put a folder outside python/ (a helper written to tmp_path) on the import path.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in SYS_PATH_MUTATORS and is_sys(node.func.value, "path"):
                add("dynamic", node, f"sys.path.{node.func.attr} imports code from outside python/")
            if node.func.attr in ("syspath_prepend", "addsitedir"):
                add("dynamic", node, f"{node.func.attr} imports code from outside python/")
        stores: list[ast.AST] = []
        if isinstance(node, ast.AugAssign | ast.AnnAssign):
            stores = [node.target]
        elif isinstance(node, ast.Assign):
            stores = list(node.targets)
        elif isinstance(node, ast.Delete):
            stores = list(node.targets)
        for target in stores:
            base = target.value if isinstance(target, ast.Subscript) else target
            if is_sys(base, "path"):
                add("dynamic", node, "sys.path is rebound or edited")
            # protocol.vault_open = ..., protocol.VAULT = ..., del protocol.load_klines
            if isinstance(target, ast.Attribute) and (
                _is_protocol(target.value, protocol_names, olsprime_names) or target.attr in PROTOCOL_NAMES - {"seal"}
            ):
                add("seal-patch", node, f"rebinds {target.attr}")
        if isinstance(node, ast.Call):
            patcher = _patcher(node)
            if patcher and node.args:
                target = node.args[0]
                attribute = node.args[1] if len(node.args) > 1 else None
                target_text = _text(target)
                attribute_text = _text(attribute) if attribute is not None else None
                if _is_protocol(target, protocol_names, olsprime_names):
                    add("seal-patch", node, f"{patcher} on olsprime.protocol")
                elif target_text is not None and (
                    "olsprime.protocol" in target_text or target_text.rsplit(".", 1)[-1] in PROTOCOL_NAMES
                ):
                    add("seal-patch", node, f"{patcher} of {target_text!r}")
                elif attribute_text in PROTOCOL_NAMES:
                    add("seal-patch", node, f"{patcher} of the attribute {attribute_text!r}")
            # vars(protocol), getattr(protocol, ...), protocol.__dict__ are dynamic access, like on a data library.
            if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "vars", "setattr", "delattr"):
                if node.args and _is_protocol(node.args[0], protocol_names, olsprime_names):
                    add("dynamic", node, f"{node.func.id} on olsprime.protocol")
        if isinstance(node, ast.Attribute) and node.attr == "__dict__" and _is_protocol(node.value, protocol_names, olsprime_names):
            add("dynamic", node, "__dict__ of olsprime.protocol")
        # getattr(sys, "mod" + "ules") and vars(sys) are sys.modules by another road.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "vars"):
            if node.args and _root_name(node.args[0]) in sys_names:
                add("dynamic", node, f"{node.func.id} on sys")
        if isinstance(node, ast.Attribute) and node.attr == "__dict__" and _root_name(node.value) in sys_names:
            add("dynamic", node, "__dict__ of sys")

    data_names = _data_package_aliases(tree)
    called = _call_functions(tree)
    for node in ast.walk(tree):
        # getattr(binance_vision, "read_" + "klines_zip") and vars(binance_vision): the data package by a string.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "vars"):
            if node.args and _root_name(node.args[0]) in data_names:
                add("dynamic", node, f"{node.func.id} on a module of olsprime.data")
        if isinstance(node, ast.Attribute) and node.attr == "__dict__" and _root_name(node.value) in data_names:
            add("dynamic", node, "__dict__ of a module of olsprime.data")
        # functools.reduce(getattr, path, obj): getattr handed over as a value is a getattr nobody can follow.
        if isinstance(node, ast.Name) and node.id in ("getattr", "vars") and isinstance(node.ctx, ast.Load) and id(node) not in called:
            add("dynamic", node, f"{node.id} passed as a value")

        # The guards' handles: the original behind a wrapper, the patches' undo, a reload of the seal.
        if isinstance(node, ast.Attribute) and node.attr in DISARM_ATTRIBUTES:
            add("disarm", node, f".{node.attr} reaches behind a guard")
        if isinstance(node, ast.Name) and node.id == "reload":
            add("disarm", node, "reload")
        if isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] == "reload":
            add("disarm", node, "reload")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "tests" and GUARD_MODULES & set(a.name.split(".")[1:]):
                    add("disarm", node, f"import {a.name}")
                elif a.name in GUARD_MODULES:
                    add("disarm", node, f"import {a.name}")
        # Importing from a test file that holds the seal allowances: its helpers carry the exemption to the caller.
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "tests" and EXEMPT_TESTS & set(a.name.split(".")[1:]):
                    add("disarm", node, f"import {a.name}")
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".") if node.module else []
            in_tests = parts[:1] == ["tests"] or node.level > 0
            leaf = parts[-1] if parts else ""
            if in_tests and leaf in EXEMPT_TESTS and any(a.name not in SAFE_TEST_HELPERS for a in node.names):
                add("disarm", node, f"from {node.module} import a helper that carries the seal exemption")
            elif in_tests and (not parts or parts == ["tests"]) and any(a.name in EXEMPT_TESTS for a in node.names):
                add("disarm", node, "from tests import a test file that holds the seal allowances")
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".") if node.module else []
            brings = any(a.name in GUARD_MODULES for a in node.names)
            if parts[:1] == ["tests"] and (GUARD_MODULES & set(parts[1:]) or (len(parts) == 1 and brings)):
                add("disarm", node, f"from {node.module} import ...")
            elif node.level and (GUARD_MODULES & set(parts) or brings):
                add("disarm", node, f"from {'.' * node.level}{node.module or ''} import ...")
            elif not node.level and len(parts) == 1 and parts[0] in GUARD_MODULES:
                add("disarm", node, f"from {node.module} import ...")
        if isinstance(node, ast.Attribute) and node.attr in GUARD_MODULES and _root_name(node.value) == "tests":
            add("disarm", node, f"tests.{node.attr}")

        # The vault's keys: forging both opens the real vault. Rebinding the seal is a ``seal-patch`` too.
        if isinstance(node, ast.Name | ast.Attribute):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name in VAULT_KEY_NAMES:
                add("seal-patch", node, f"{name} forges a key of the vault")
        elif isinstance(node, ast.alias) and node.name.rsplit(".", 1)[-1] in VAULT_KEY_NAMES:
            add("seal-patch", node, f"{node.name} forges a key of the vault")
        if isinstance(node, ast.Constant | ast.BinOp | ast.JoinedStr):
            text = _text(node)
            if text is not None and VAULT_KEY_STRING in text:
                add("seal-patch", node, f"the name {VAULT_KEY_STRING} forges a key of the vault")
            if text is not None and any(spelled in text for spelled in VAULT_TREE_STRINGS):
                add("vault-tree", node, "the text names the vault tree, its folder or its identity variable")


def notebook_code(text: str) -> tuple[str, list[str]]:
    """The Python of a notebook's code cells, joined, and the magic lines that name data.

    Cells are joined in order, each cell's lines keep their place so that line numbers
    are those of the joined text. Lines starting with ``%`` or ``!`` are blanked; a cell
    starting with ``%%`` (a cell magic: its body is not Python) is blanked whole. The
    blanked text is returned too, to be scanned by MAGIC_DATA_TOKENS.
    """
    cells = json.loads(text).get("cells", [])
    lines: list[str] = []
    magic: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        body = "".join(source) if isinstance(source, list) else source
        whole = body.lstrip().startswith("%%")
        for line in body.splitlines():
            if whole or line.lstrip().startswith(("%", "!")):
                magic.append(line)
                line = ""
            lines.append(line)
        lines.append("")
    return "\n".join(lines), magic


def findings(source: str, filename: str) -> list[tuple[str, str]]:
    """``(rule, text)`` of every bypass of ``load_klines`` in ``source``, before the allow-list.

    ``filename`` ending in ``.ipynb`` means ``source`` is notebook JSON. Code that does not
    parse is itself a finding (``unparseable``), never a silent skip.
    """
    found: list[tuple[str, str]] = []

    def add(rule: str, node: ast.AST | None, detail: str) -> None:
        found.append((rule, f"{filename}:{getattr(node, 'lineno', '?')}: [{rule}] {detail}"))

    if filename.endswith(".ipynb"):
        try:
            source, magic = notebook_code(source)
        except (ValueError, AttributeError) as error:
            return [("unparseable", f"{filename}: [unparseable] not a notebook: {error}")]
        for line in magic:
            if MAGIC_DATA_TOKENS.search(line):
                add("magic", None, f"IPython line names data: {line.strip()[:60]}")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [*found, ("unparseable", f"{filename}:{error.lineno}: [unparseable] {error.msg}")]
    aliases = _library_aliases(tree)
    module_aliases = _module_aliases(tree)
    _extra_findings(tree, add)

    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.alias):
            names.append(node.name.rsplit(".", 1)[-1])  # the real name, whatever the alias
        for name in names:
            if name in READERS:
                add("reader", node, name)
            elif name in RAW_READERS:
                add("raw-reader", node, name)
            elif name in DYNAMIC_NAMES:
                add("dynamic", node, name)
            elif name == SEAL:
                add("seal", node, name)
            elif name in VAULT_TREE_NAMES:
                add("vault-tree", node, f"{name} reaches the vault tree")
        # Relative imports (``from .binance_vision import ...``) are skipped on purpose: they can only name
        # a module inside the package, and every module that reads data is already named in ALLOWED or
        # flagged by its own file. Today ``olsprime/data/binance_vision.py`` is the only data-reading
        # submodule; a second one (a cache, a feature store) must be added to ALLOWED or to the rules here.
        if isinstance(node, ast.Name) and node.id in EXEC_BUILTINS and isinstance(node.ctx, ast.Load):
            add("exec", node, f"{node.id} runs code built from a string")
        if isinstance(node, ast.Attribute) and _root_name(node.value) in module_aliases:
            rule, attributes = MODULE_ATTRIBUTES[module_aliases[_root_name(node.value)]]
            if node.attr in attributes:
                add(rule, node, f"{module_aliases[_root_name(node.value)]}.{node.attr}")
        if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "") in MODULE_ATTRIBUTES:
            rule, attributes = MODULE_ATTRIBUTES[node.module]
            for a in node.names:
                if a.name in attributes:
                    add(rule, node, f"from {node.module} import {a.name}")
        if isinstance(node, ast.Import | ast.ImportFrom) and not getattr(node, "level", 0):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            else:  # from pyarrow import parquet: the module and each name it brings
                modules = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
            for module in modules:
                for banned in BANNED_MODULES:
                    if module == banned or module.startswith(banned + "."):
                        add(f"import:{banned}", node, f"import {module}")
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in ("getattr", "vars") and node.args and _root_name(node.args[0]) in aliases:
                add("dynamic", node, f"{func_name} on a data library")
            if _literal_data_file(node):
                add("literal-open", node, "opens a .parquet or .zip file by name")
        if isinstance(node, ast.Attribute) and node.attr == "__dict__" and _root_name(node.value) in aliases:
            add("dynamic", node, "__dict__ of a data library")
    return found


def violations(source: str, filename: str) -> list[str]:
    """What ``source``, living at ``filename`` (relative to python/), may not do; the allow-list applied."""
    allowed = ALLOWED.get(filename, {})
    # A file that holds the seal allowance (it defines, calls or tests the seal) may patch it.
    exempt = {"seal-patch"} if "seal" in allowed else set()
    return [text for rule, text in findings(source, filename) if rule not in allowed and rule not in exempt]


def python_files(root: Path = PYTHON_ROOT) -> list[Path]:
    """Every ``.py`` and ``.ipynb`` file under ``root`` except ``.venv``, notebook checkpoints and caches."""
    skipped = {".venv", "__pycache__", ".ipynb_checkpoints"}
    found = [p for pattern in ("*.py", "*.ipynb") for p in root.rglob(pattern)]
    return sorted(p for p in found if not skipped & set(p.relative_to(root).parts))


def test_no_module_or_test_bypasses_load_klines():
    files = python_files()
    names = [p.relative_to(PYTHON_ROOT).as_posix() for p in files]
    assert len(files) >= 12 and "olsprime/protocol.py" in names and "tests/test_protocol.py" in names
    assert not any(".venv" in n for n in names)
    offenders = [v for p, n in zip(files, names, strict=True) for v in violations(p.read_text(encoding="utf-8"), n)]
    assert offenders == []


def test_every_allowance_is_exact_and_still_used():
    for name, rules in ALLOWED.items():
        path = PYTHON_ROOT / name
        assert path.is_file(), f"stale allow-list entry {name}"
        source = path.read_text(encoding="utf-8")
        used = {rule for rule, _ in findings(source, name)}
        for rule, reason in rules.items():
            assert reason and rule in used, f"{name} is allowed {rule} but never uses it"
        # The same code anywhere else is flagged: the allowance belongs to the file, not to the code.
        assert violations(source, "olsprime/features/elsewhere.py")


# Cheater snippets: (id, source, rules that must flag it). Each is a way to read data
# without load_klines, found by the auditor or by the guard's author.
ELSEWHERE = "olsprime/features/elsewhere.py"
CHEATERS = [
    ("getattr-string-concat", "import polars as pl\nf = getattr(pl, 'read_' + 'parquet')", {"dynamic"}),
    ("importlib-import_module", "import importlib\nm = importlib.import_module('duckdb')", {"import:importlib", "dynamic"}),
    ("from-importlib", "from importlib import import_module as im\nim('duck' + 'db')", {"import:importlib", "dynamic"}),
    ("dunder-import", "m = __import__('duckdb')", {"dynamic"}),
    ("read_csv_batched", "import polars as pl\npl.read_csv_batched('x.csv')", {"raw-reader"}),
    ("read_database", "import polars as pl\npl.read_database('select 1', conn)", {"reader"}),
    ("scan_pyarrow_dataset", "import polars as pl\npl.scan_pyarrow_dataset(ds)", {"reader"}),
    ("from_pandas", "import polars as pl\npl.from_pandas(df)", {"reader"}),
    ("from_arrow", "import polars as pl\npl.from_arrow(table)", {"reader"}),
    (
        "zipfile-and-csv-on-raw-cache",
        "import zipfile, csv\nz = zipfile.ZipFile('raw.zip')\nrows = list(csv.reader(z.open('a.csv')))",
        {"import:zipfile", "import:csv", "raw-reader"},
    ),
    ("zipfile-from-import", "from zipfile import ZipFile\nZipFile(p)", {"import:zipfile", "raw-reader"}),
    ("csv-alone", "import csv\ncsv.reader(handle)", {"import:csv"}),
    ("pyarrow-ipc-open_file", "import pyarrow as pa\npa.ipc.open_file(src)", {"import:pyarrow"}),
    ("pyarrow-submodule-import", "from pyarrow import ipc\nipc.open_stream(src)", {"import:pyarrow"}),
    ("pyarrow-parquet-alias", "from pyarrow import parquet as pq\npq.read_table('x')", {"import:pyarrow", "reader"}),
    (
        "open-parquet-then-deserialize",
        "import polars as pl\nwith open('x.parquet', 'rb') as f:\n    df = pl.DataFrame.deserialize(f)",
        {"literal-open", "reader"},
    ),
    ("open-zip-literal", "data = open('cache/BTCUSDT-1h-2024-01.zip', 'rb').read()", {"literal-open"}),
    ("path-open-parquet", "from pathlib import Path\nPath('x.parquet').open('rb')", {"literal-open"}),
    ("path-read_bytes-zip", "from pathlib import Path\nPath('x.ZIP').read_bytes()", {"literal-open"}),
    ("numpy-genfromtxt", "import numpy as np\nnp.genfromtxt('x.csv', delimiter=',')", {"raw-reader"}),
    ("numpy-loadtxt", "import numpy as np\nnp.loadtxt('x.csv')", {"raw-reader"}),
    ("vars-of-polars", "import polars as pl\nvars(pl)['read_parquet']('x')", {"dynamic"}),
    ("dict-of-polars", "import polars as pl\npl.__dict__['scan_' + 'parquet']('x')", {"dynamic"}),
    ("getattr-on-from-import", "from polars import DataFrame as D\ngetattr(D, 'deserialize')", {"dynamic"}),
    (
        "forged-time-column",
        "from olsprime.protocol import seal\nimport polars as pl\n"
        "seal(frame.with_columns(known_at=pl.lit(2020)), time_col='known_at', caller='me')",
        {"seal"},
    ),
    ("aliased-read_parquet", "from polars import read_parquet as rp\nrp('x')", {"reader"}),
    ("pandas", "import pandas as pd\npd.read_parquet('x')", {"import:pandas", "reader"}),
    ("duckdb", "import duckdb\nduckdb.sql('select * from \"x.parquet\"')", {"import:duckdb"}),
    ("pl-read_parquet", "import polars as pl\npl.read_parquet('x')", {"reader"}),
    ("pl-scan_parquet", "import polars as pl\npl.scan_parquet('x').collect()", {"reader"}),
    ("pl-read_csv", "import polars as pl\npl.read_csv('x')", {"raw-reader"}),
    ("pl-scan_ndjson", "import polars as pl\npl.scan_ndjson('x')", {"raw-reader"}),
    ("eval", "df = eval('pl.' + 'read_' + 'parquet')('x')", {"exec"}),
    ("exec", "exec(\"import polars as pl; df = pl.read_parquet('x')\")", {"exec"}),
    ("compile", "code = compile(src, '<s>', 'exec')", {"exec"}),
    ("builtins-eval", "import builtins\nbuiltins.eval(src)", {"exec"}),
    ("subprocess-parquet-tools", "import subprocess\nsubprocess.run(['parquet-tools', 'cat', 'x.parquet'])", {"import:subprocess"}),
    ("from-subprocess", "from subprocess import check_output as co\nco(['duckdb', '-c', 'select 1'])", {"import:subprocess"}),
    ("os-system", "import os\nos.system('parquet-tools cat x')", {"shell"}),
    ("os-popen", "import os as o\no.popen('duckdb')", {"shell"}),
    ("from-os-system", "from os import system\nsystem('ls')", {"shell"}),
    ("np-load", "import numpy as np\nnp.load('rows.npy')", {"binary-load"}),
    ("np-load-allow-pickle", "import numpy\nnumpy.load('rows.npz', allow_pickle=True)", {"binary-load"}),
    ("np-fromfile", "import numpy as np\nnp.fromfile('rows.bin')", {"binary-load"}),
    ("np-memmap", "import numpy as np\nnp.memmap('rows.bin')", {"binary-load"}),
    ("from-numpy-load", "from numpy import load\nload('rows.npy')", {"binary-load"}),
    ("pickle-load", "import pickle\npickle.load(open('rows.pkl', 'rb'))", {"import:pickle"}),
    ("joblib-load", "import joblib\njoblib.load('rows.joblib')", {"import:joblib"}),
    ("shelve", "import shelve\nshelve.open('rows')", {"import:shelve"}),
    ("shutil-copy", "import shutil\nshutil.copy(src, dst)", {"copy"}),
    ("shutil-copyfile", "import shutil\nshutil.copyfile(src, dst)", {"copy"}),
    ("shutil-copy2", "import shutil as sh\nsh.copy2(src, dst)", {"copy"}),
    ("shutil-copytree", "import shutil\nshutil.copytree(src, dst)", {"copy"}),
    ("from-shutil-copy", "from shutil import copy as c\nc(src, dst)", {"copy"}),
    # --- Audit A, D1: the two paths that passed both guards ---
    (
        "audit-a-helper-on-sys-path",  # bypass2/test_evasion.py: a helper written to tmp_path, then imported
        "import sys\nfrom pathlib import Path\n(tmp_path / 'helper_audit_a.py').write_text(HELPER)\n"
        "sys.path.insert(0, str(tmp_path))\nimport helper_audit_a\ndf = helper_audit_a.raw(RAW_VAULT)",
        {"dynamic"},
    ),
    (
        "audit-a-sys-modules-fragment",  # bypass2/test_evasion.py: pyarrow's fragment API through sys.modules
        "import sys\nds = sys.modules['pyarrow.dataset']\nfs = sys.modules['pyarrow.fs']\n"
        "t = ds.ParquetFileFormat().make_fragment(PQ_1D, filesystem=fs.LocalFileSystem()).to_table()",
        {"dynamic"},
    ),
    ("sys-modules-get", "import sys\nds = sys.modules.get('pyarrow.dataset')", {"dynamic"}),
    ("sys-modules-aliased", "import sys as s\nm = s.modules['polars']", {"dynamic"}),
    ("from-sys-import-modules", "from sys import modules\nm = modules['polars']", {"dynamic"}),
    ("getattr-on-sys", "import sys\nm = getattr(sys, 'mod' + 'ules')['polars']", {"dynamic"}),
    ("vars-of-sys", "import sys\nm = vars(sys)['mod' + 'ules']", {"dynamic"}),
    ("sys-path-append", "import sys\nsys.path.append(str(tmp_path))", {"dynamic"}),
    ("sys-path-extend", "import sys\nsys.path.extend([str(tmp_path)])", {"dynamic"}),
    ("sys-path-augmented", "import sys\nsys.path += [str(tmp_path)]", {"dynamic"}),
    ("sys-path-slice", "import sys\nsys.path[0:0] = [str(tmp_path)]", {"dynamic"}),
    ("sys-path-rebound", "import sys\nsys.path = [str(tmp_path)]", {"dynamic"}),
    ("pytest-syspath-prepend", "monkeypatch.syspath_prepend(str(tmp_path))", {"dynamic"}),
    (
        "raw-zip-and-read_csv-of-an-archive",
        "import io, zipfile\nimport polars as pl\n"
        "z = zipfile.ZipFile('data/raw/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-03.zip')\n"
        "df = pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), has_header=False)",
        {"import:zipfile", "raw-reader"},
    ),
    (
        "make_fragment-and-to_table",
        "import pyarrow.dataset as ds\nds.ParquetFileFormat().make_fragment(p).to_table()",
        {"import:pyarrow"},
    ),
    ("from_paths", "from pyarrow import dataset as ds\nds.FileSystemDataset.from_paths(paths)", {"import:pyarrow"}),
    # --- Audit A, D2: switching the seal off ---
    (
        "monkeypatch-vault_open",
        "from olsprime import protocol\nmonkeypatch.setattr(protocol, 'vault_open', lambda: True)",
        {"seal-patch"},
    ),
    (
        "monkeypatch-computed-name",
        "from olsprime import protocol\nmonkeypatch.setattr(protocol, 'vault_' + 'open', lambda: True)",
        {"seal-patch"},
    ),
    (
        "monkeypatch-fstring-name",
        "from olsprime import protocol\nname = 'open'\nmonkeypatch.setattr(protocol, f'vault_{name}', lambda: True)",
        {"seal-patch"},
    ),
    ("monkeypatch-VAULT", "from olsprime import protocol as p\nmonkeypatch.setattr(p, 'VAULT', other_period)", {"seal-patch"}),
    ("monkeypatch-the-seal", "from olsprime import protocol\nmonkeypatch.setattr(protocol, 'seal', lambda f, **k: f)", {"seal-patch"}),
    ("monkeypatch-string-target", "monkeypatch.setattr('olsprime.protocol.vault_open', lambda: True)", {"seal-patch"}),
    ("monkeypatch-load_klines-elsewhere", "monkeypatch.setattr(universe, 'load_klines', fake)", {"seal-patch"}),
    ("builtin-setattr", "import olsprime.protocol\nsetattr(olsprime.protocol, 'VAULT', other_period)", {"seal-patch", "dynamic"}),
    ("mock-patch-string", "from unittest import mock\nmock.patch('olsprime.protocol.load_klines_file')", {"seal-patch"}),
    ("mock-patch-concatenated-string", "from unittest import mock\nmock.patch('olsprime.' + 'protocol.vault_open')", {"seal-patch"}),
    (
        "mock-patch-object",
        "from unittest import mock\nfrom olsprime import protocol\nmock.patch.object(protocol, 'vault_open', return_value=True)",
        {"seal-patch"},
    ),
    ("assign-vault_open", "import olsprime.protocol as protocol\nprotocol.vault_open = lambda: True", {"seal-patch"}),
    ("assign-through-the-package", "import olsprime.protocol\nolsprime.protocol.VAULT = other_period", {"seal-patch"}),
    ("delete-load_universe", "from olsprime import protocol\ndel protocol.load_universe", {"seal-patch"}),
    ("vars-of-protocol", "from olsprime import protocol\nvars(protocol)['vault_open'] = lambda: True", {"dynamic"}),
    ("dict-of-protocol", "from olsprime import protocol\nprotocol.__dict__['vault_open'] = lambda: True", {"dynamic"}),
    ("getattr-on-protocol", "from olsprime import protocol\ngetattr(protocol, 'vault_' + 'open')", {"dynamic"}),
    ("relative-import-of-protocol", "from . import protocol\nmonkeypatch.setattr(protocol, 'seal', None)", {"seal-patch"}),
]


def notebook(*cells: str | list[str], markdown: str = "# notes") -> str:
    """A minimal notebook (nbformat 4) with ``cells`` as code cells and one markdown cell."""
    body = [{"cell_type": "markdown", "metadata": {}, "source": markdown}]
    body += [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": c} for c in cells]
    return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": body})


NB = "notebooks/colab.ipynb"
NOTEBOOK_CHEATERS = [
    ("nb-scan-parquet", notebook("import polars as pl", "df = pl.scan_parquet('data/x.parquet').collect()"), {"reader"}),
    ("nb-alias-across-cells", notebook(["import polars as pl\n"], ["f = getattr(pl, 'read_' + 'parquet')"]), {"dynamic"}),
    (
        "nb-magics-and-shell-escape",
        notebook(
            "%pip install -q polars\n!parquet-tools cat data/x.parquet\nimport polars as pl\npl.read_parquet('x')",
            "%%bash\nduckdb -c \"select * from 'x.parquet'\"",
        ),
        {"reader", "magic"},
    ),
    ("nb-shell-escape-alone", notebook("!cp data/parquet/spot/klines/1h/BTCUSDT.parquet /content/"), {"magic"}),
    ("nb-not-json", "{ this is not a notebook", {"unparseable"}),
    ("nb-code-that-does-not-parse", notebook("def broken(:\n    pass"), {"unparseable"}),
]
CONTROLS = [
    ("load_klines-import", "from olsprime.protocol import load_klines"),
    ("load_klines-call", "from olsprime.protocol import load_klines\nload_klines('BTCUSDT', '1h', caller='features')"),
    (
        "polars-expression",
        "import polars as pl\nout = df.with_columns((pl.col('close') / pl.col('open')).alias('r')).filter(pl.col('r') > 0)",
    ),
    ("polars-write", "import polars as pl\ndf.write_parquet('out.parquet')\nout = pl.DataFrame({'a': [1]})"),
    ("getattr-on-own-object", "import polars as pl\nvalue = getattr(config, 'name')\nvars(config)"),
    ("open-text-file", "with open('notes.txt') as f:\n    f.read()\nopen('data.json', 'rb')"),
    ("ordinary-imports", "import json, io, os\nfrom pathlib import Path\nimport numpy as np\nnp.zeros(3)"),
    ("relative-import", "from . import csv_helpers\nfrom .zipfile import x"),
    ("re-compile", "import re\npattern = re.compile(r'(\\d+)\\.zip$')"),
    ("polars-list-eval", "import polars as pl\nexpr = pl.col('a').list.eval(pl.element() * 2)"),
    ("os-path-and-environ", "import os\nos.path.join('a', 'b')\nos.environ.get('X')\nos.getcwd()"),
    ("shutil-rmtree", "import shutil\nshutil.rmtree(tmp)\nshutil.which('git')"),
    ("copy-module-and-frame-copy", "import copy\ncopy.copy(x)\ncopy.deepcopy(x)\nframe = df.clone()\nrows = lst.copy()"),
    ("numpy-save-and-compute", "import numpy as np\nnp.save('w.npy', np.zeros(3))\nnp.linalg.lstsq(a, b)"),
    ("json-loads", "import json\njson.loads(text)\nvalue = json.load(handle)"),
    ("sys-reads", "import sys\nprint(sys.argv, sys.platform, sys.version_info, sys.path)\nsys.exit(0)\nsys.stdout.write('x')"),
    ("sys-getattr-frozen", "import json\nvalue = getattr(config, 'frozen', False)"),
    ("monkeypatch-env-and-other-modules", "monkeypatch.setenv('X', '1')\nmonkeypatch.setattr(universe, 'build_universe', f)\nmonkeypatch.setattr(socket.socket, 'connect', f)"),
    ("calling-the-protocol", "from olsprime import protocol\nprotocol.load_klines('BTCUSDT', '1h', caller='x')\nprotocol.vault_open()\nprotocol.VAULT.start"),
    ("setattr-on-own-object", "import olsprime.protocol\nsetattr(self, 'value', 1)\nmock.patch.object(thing, 'size', 3)"),
    ("assign-own-attributes", "self.total = 1\nconfig.seal_length = 2\nobj.vault = 3"),
]
NOTEBOOK_CONTROLS = [
    ("nb-plain-load_klines", notebook("from olsprime.protocol import load_klines", "df = load_klines('BTCUSDT', '1h')")),
    ("nb-magics-only", notebook("%matplotlib inline\n!pip install -q torch\nimport numpy as np\nnp.zeros(3)")),
    ("nb-markdown-mentions-parquet", notebook("x = 1", markdown="we read x.parquet with pl.read_parquet here")),
    ("nb-empty", notebook()),
]


# Round 2: the second audit's evasions that a static rule can see (the runtime ones are in test_tripwire.py).
CHEATERS += [
    ("e10-read_klines_zip", "from olsprime.data import binance_vision as bv\nbv.read_klines_zip(z, 'BTCUSDT')", {"raw-reader"}),
    ("e10b-getattr-on-a-data-module", "from olsprime.data import binance_vision as bv\ngetattr(bv, 'read_' + 'klines_zip')(z, 'BTCUSDT')", {"dynamic"}),
    ("e10c-vars-on-a-data-module", "from olsprime.data import binance_vision\nvars(binance_vision)['read_' + 'klines_zip']", {"dynamic"}),
    ("e10d-dict-of-a-data-module", "import olsprime.data.survey as sv\nsv.__dict__['x']", {"dynamic"}),
    ("e10e-relative-data-module", "from . import binance_vision as bv\ngetattr(bv, name)", {"dynamic"}),
    ("e16-forge-keys-by-hand", "monkeypatch.setenv('OLSPRIME_OPEN_VAULT', '1')", {"seal-patch"}),
    ("e16a-forge-keys-concatenated", "import os\nos.environ['OLSPRIME_OPEN_' + 'VAULT'] = '1'", {"seal-patch"}),
    ("e16b-open_vault-helper", "from tests.test_protocol import open_vault\nopen_vault(monkeypatch)", {"seal-patch"}),
    ("e16c-vault_key_file", "from olsprime.protocol import vault_key_file\nvault_key_file().write_text('2026-11-01')", {"seal-patch"}),
    ("e16d-OPEN_ENV", "from olsprime import protocol\nmonkeypatch.setenv(protocol.OPEN_ENV, '1')", {"seal-patch"}),
    ("e1-wrapped-seal", "protocol.seal = protocol.seal.__wrapped__", {"disarm", "seal"}),
    ("e2-globals-of-load_klines", "g = load_klines.__globals__\ng['vault_open'] = lambda: True", {"disarm"}),
    ("e4-patch-is_allowed", "from tests import tripwire\nmonkeypatch.setattr(tripwire, 'is_allowed', lambda f, r: True)", {"disarm"}),
    ("e4a-import-tests.tripwire", "import tests.tripwire as tw", {"disarm"}),
    ("e4b-from-tests.tripwire", "from tests.tripwire import is_allowed", {"disarm"}),
    ("e4c-from-tests.allowlist", "from tests.allowlist import ALLOWED\nALLOWED['tests/x.py'] = {'reader': 'mine'}", {"disarm"}),
    ("e4d-import-allowlist-from-tests", "from tests import allowlist", {"disarm"}),
    ("e4e-relative-tripwire", "from .tripwire import is_allowed", {"disarm"}),
    ("e4f-relative-conftest", "from . import conftest", {"disarm"}),
    ("f2-helper-of-an-exempt-test", "from tests.test_tripwire import run_source", {"disarm"}),
    ("f2a-helper-of-test_manifest", "from tests.test_manifest import anything", {"disarm"}),
    ("f2b-unsafe-helper-of-test_protocol", "from tests.test_protocol import findings", {"disarm"}),
    ("f2c-import-the-test-module", "import tests.test_tripwire as tt", {"disarm"}),
    ("f2d-from-tests-import-test-file", "from tests import test_tripwire", {"disarm"}),
    ("f2e-relative-test-module", "from .test_tripwire import run_source", {"disarm"}),
    ("e5-reload", "import importlib\nimportlib.reload(protocol)", {"import:importlib", "disarm"}),
    ("e5a-reload-by-name", "from importlib import reload\nreload(protocol)", {"import:importlib", "disarm"}),
    ("e6-wrapped-reader", "import polars as pl\npl.read_parquet.__wrapped__(p)", {"disarm", "reader"}),
    ("e6b-closure", "import polars as pl\ncells = pl.read_parquet.__closure__", {"disarm"}),
    ("e7-bases", "import pyarrow.parquet as pq\nbase = pq.ParquetFile.__bases__[0]", {"disarm"}),
    ("e7a-base", "base = SomeClass.__base__", {"disarm"}),
    ("e7b-mro", "cls = SomeClass.__mro__[1]", {"disarm"}),
    ("e7c-subclasses", "classes = SomeClass.__subclasses__()", {"disarm"}),
    ("e7d-code-object", "wrapper.__code__ = original.__code__", {"disarm"}),
    ("e15-runpy", "import runpy\nrunpy.run_path(helper)", {"import:runpy"}),
    ("e15a-from-runpy", "from runpy import run_path", {"import:runpy"}),
    ("n1-zipimporter", "import zipimport\nzipimport.zipimporter(archive).get_data('m.csv')", {"import:zipimport"}),
    ("n1a-from-zipimport", "from zipimport import zipimporter", {"import:zipimport"}),
    ("undo-then-runpy", "from tests.conftest import PATCHES\nPATCHES.undo()", {"disarm"}),
    ("undo-by-any-name", "patches = pytest.MonkeyPatch()\npatches.undo()", {"disarm"}),
    ("gc-referents", "import gc\ngc.get_referents(wrapper)", {"import:gc"}),
    ("ctypes", "import ctypes", {"import:ctypes"}),
    ("s1-attrgetter", "import operator\noperator.attrgetter('read_' + 'parquet')(pl)", {"dynamic"}),
    ("s1a-from-operator", "from operator import attrgetter as ag", {"dynamic"}),
    ("s1b-methodcaller", "import operator\noperator.methodcaller('read_' + 'parquet', p)(pl)", {"dynamic"}),
    ("s2-reduce-getattr", "import functools\nfunctools.reduce(getattr, ['read_' + 'parquet'], pl)", {"dynamic"}),
    ("s2a-map-getattr", "list(map(getattr, [pl, pl], ['read_parquet', 'scan_parquet']))", {"dynamic"}),
    ("s3-getattribute", "import polars as pl\npl.__getattribute__('read_' + 'parquet')", {"dynamic"}),
    ("s4-builtins-import", "zf = __builtins__['__import__']('zip' + 'file')", {"dynamic"}),
]

CHEATERS += [
    ("v1-vault_dir", "from olsprime.paths import vault_dir\nroot = vault_dir()", {"vault-tree"}),
    ("v1a-vault_dir-through-paths", "from olsprime import paths\nroot = paths.vault_dir() / 'parquet'", {"vault-tree"}),
    ("v1b-the-variable-by-name", "import os\nroot = os.environ['OLSPRIME_VAULT_DIR']", {"vault-tree"}),
    ("v1c-the-variable-concatenated", "import os\nroot = os.environ['OLSPRIME_VAULT_' + 'DIR']", {"vault-tree"}),
    ("v1d-the-identity-variable", "import os\nkey = os.environ.get('OLSPRIME_VAULT_IDENTITY')", {"vault-tree"}),
    ("v1e-the-default-folder", "from pathlib import Path\nroot = Path.home() / 'ols-prime-vault'", {"vault-tree"}),
    ("v1f-the-tree-file-of-a-pair", "from olsprime.protocol import vault_tree_klines_path\nvault_tree_klines_path('BTCUSDT', '1d')", {"vault-tree"}),
    ("v1g-decrypt-by-hand", "from olsprime.protocol import decrypt_vault_bytes\ndecrypt_vault_bytes(blob, what='x')", {"vault-tree"}),
    ("v1h-the-identity-reader", "from olsprime import protocol\nprotocol.read_identity(path)", {"vault-tree"}),
    ("v1i-the-variable-in-an-fstring", "import os\nos.environ[f'OLSPRIME_VAULT_{name}']", {"vault-tree"}),
    ("v2-read_parquet_schema", "import polars as pl\npl.read_parquet_schema('x')", {"reader"}),
    ("v2a-read_parquet_metadata", "import polars as pl\npl.read_parquet_metadata('x')", {"reader"}),
    ("v2b-scan_lines", "import polars as pl\npl.scan_lines('x')", {"raw-reader"}),
]

CONTROLS += [
    ("the-vault-module-by-name", "from olsprime import vault\nvault.status(data_root, vault_root)\nvault.plan_text(1)"),
    ("getattr-called-on-own-object", "value = getattr(config, 'name', None)\nvars(config)"),
    ("data-module-used-by-name", "from olsprime.data import survey\nsurvey.run(path)"),
    ("tests-helpers", "from tests.fakes import klines_csv\nfrom tests.test_protocol import bars"),
    ("dunder-on-own-class", "class A:\n    def __init__(self):\n        self.__dict__['x'] = 1"),
]


@pytest.mark.parametrize(("name", "source", "expected"), CHEATERS, ids=[c[0] for c in CHEATERS])
def test_the_guard_flags_each_cheater_snippet(name, source, expected):
    flagged = {rule for rule, _ in findings(source, ELSEWHERE)}
    assert expected <= flagged, f"{name}: expected {sorted(expected)}, flagged {sorted(flagged)}"
    assert violations(source, ELSEWHERE)  # and the allow-list does not excuse it outside its files


@pytest.mark.parametrize(("name", "source"), CONTROLS, ids=[c[0] for c in CONTROLS])
def test_the_guard_lets_the_negative_controls_through(name, source):
    assert violations(source, ELSEWHERE) == []


@pytest.mark.parametrize(("name", "source", "expected"), NOTEBOOK_CHEATERS, ids=[c[0] for c in NOTEBOOK_CHEATERS])
def test_the_guard_flags_each_notebook_cheater(name, source, expected):
    flagged = {rule for rule, _ in findings(source, NB)}
    assert expected <= flagged, f"{name}: expected {sorted(expected)}, flagged {sorted(flagged)}"
    assert violations(source, NB)


@pytest.mark.parametrize(("name", "source"), NOTEBOOK_CONTROLS, ids=[c[0] for c in NOTEBOOK_CONTROLS])
def test_the_guard_lets_clean_notebooks_through(name, source):
    assert violations(source, NB) == []


def test_notebook_magics_are_blanked_not_fatal_and_keep_their_line_numbers():
    cells = ["%matplotlib inline\n!pip install x\nimport polars as pl", ["%%time\n", "slow thing(\n"], "pl.scan_parquet('x')"]
    code, magic = notebook_code(notebook(*cells))
    ast.parse(code)  # the whole notebook parses once the magics are blanked
    assert magic == ["%matplotlib inline", "!pip install x", "%%time", "slow thing("]
    (hit,) = [text for rule, text in findings(notebook(*cells), NB) if rule == "reader"]
    lineno = next(i for i, line in enumerate(code.splitlines(), 1) if "scan_parquet" in line)
    assert hit.startswith(f"{NB}:{lineno}:")


def test_the_walk_covers_notebooks_and_skips_the_virtual_environment(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "scan.ipynb").write_text(notebook("import polars as pl\npl.read_parquet('x')"))
    (tmp_path / "pkg" / ".ipynb_checkpoints").mkdir()
    (tmp_path / "pkg" / ".ipynb_checkpoints" / "scan-checkpoint.ipynb").write_text(notebook("x = 1"))
    (tmp_path / ".venv" / "lib" / "site.py").write_text("import pickle\n")
    (tmp_path / ".venv" / "lib" / "other.ipynb").write_text(notebook("import pickle"))
    names = [p.relative_to(tmp_path).as_posix() for p in python_files(tmp_path)]
    assert names == ["pkg/a.py", "pkg/scan.ipynb"]
    offenders = [v for p in python_files(tmp_path) for v in violations(p.read_text(), p.relative_to(tmp_path).as_posix())]
    assert len(offenders) == 1 and "pkg/scan.ipynb" in offenders[0] and "[reader]" in offenders[0]


def test_the_allow_list_is_per_file_and_per_rule():
    assert violations("import polars as pl\npl.read_parquet('x')", "olsprime/protocol.py") == []
    assert violations("import polars as pl\npl.read_csv('x')", "olsprime/protocol.py")  # only the reader rule
    assert violations("import polars as pl\npl.read_csv('x')", "olsprime/data/binance_vision.py") == []
    assert violations("import polars as pl\npl.read_parquet('x')", "olsprime/data/binance_vision.py")
    assert violations("import zipfile", "tests/fakes.py") == []
    assert violations("import zipfile", "olsprime/data/quality.py")
    assert violations("import polars as pl\npl.read_parquet('x')", "tests/test_models.py")  # a new test is not excused


def test_guard_coverage_table(capsys):
    """Which rule flags which snippet; run with ``-s`` to see it."""
    rows = []
    for name, source, expected in CHEATERS:
        flagged = sorted({rule for rule, _ in findings(source, ELSEWHERE)})
        rows.append((name, ", ".join(flagged) or "-", "caught" if expected <= set(flagged) else "MISSED"))
    for name, source, expected in NOTEBOOK_CHEATERS:
        flagged = sorted({rule for rule, _ in findings(source, NB)})
        rows.append((name, ", ".join(flagged) or "-", "caught" if expected <= set(flagged) else "MISSED"))
    for name, source in CONTROLS:
        flagged = sorted({rule for rule, _ in findings(source, ELSEWHERE)})
        rows.append((f"(control) {name}", ", ".join(flagged) or "-", "passes" if not flagged else "FALSE ALARM"))
    for name, source in NOTEBOOK_CONTROLS:
        flagged = sorted({rule for rule, _ in findings(source, NB)})
        rows.append((f"(control) {name}", ", ".join(flagged) or "-", "passes" if not flagged else "FALSE ALARM"))
    width = max(len(r[0]) for r in rows)
    with capsys.disabled():
        print("\nGuard coverage")
        for name, flagged, verdict in rows:
            print(f"  {name:<{width}}  {verdict:<11} {flagged}")
    assert all(verdict in ("caught", "passes") for _, _, verdict in rows)


# --- Rust ---------------------------------------------------------------------------


def banned_crates(lock_text: str) -> list[str]:
    names = re.findall(r'^name = "([^"]+)"', lock_text, re.M)
    return sorted(n for n in names if re.match(r"(parquet|arrow|csv)", n))


def test_cargo_lock_has_no_crate_that_reads_data():
    # No Rust code may read data until a Rust seal exists.
    synthetic = 'version = 3\n\n[[package]]\nname = "arrow-array"\n\n[[package]]\nname = "csv-core"\n\n[[package]]\nname = "serde"\n'
    assert banned_crates(synthetic) == ["arrow-array", "csv-core"]
    text = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
    assert len(re.findall(r'^name = "', text, re.M)) > 0  # the real lock, not an empty stub
    assert banned_crates(text) == []


def test_the_test_suite_never_touches_the_repository_vault():
    assert not str(runs_dir()).startswith(str(REPO_ROOT))


def test_a_file_with_the_seal_allowance_may_patch_it_and_no_other_may():
    snippet = "from olsprime import protocol\nmonkeypatch.setattr(protocol, 'vault_' + 'open', lambda: True)"
    assert [rule for rule, _ in findings(snippet, ELSEWHERE)] == ["seal-patch"]
    assert violations(snippet, ELSEWHERE)
    assert violations(snippet, "tests/test_models.py")
    assert violations(snippet, "olsprime/protocol.py") == []  # holds the seal allowance
    assert violations(snippet, "tests/test_protocol.py") == []
    assert "seal" in ALLOWED["tests/test_protocol.py"] and "seal-patch" not in ALLOWED["tests/test_protocol.py"]


def test_the_protocol_names_the_guard_refuses_are_the_ones_the_tripwire_refuses():
    from tests.tripwire import SEAL_NAMES

    assert PROTOCOL_NAMES == set(SEAL_NAMES)
    assert all(hasattr(protocol, name) for name in PROTOCOL_NAMES)
