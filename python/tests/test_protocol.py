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


def test_build_parquet_still_stores_and_reports_every_row(tmp_path):
    bucket = FakeBucket()
    months = [Month(2025, 12), Month(2026, 1)]
    for month in months:
        bucket.add_month("BTCUSDT", "1d", month, month_rows(month, DAY_MS))
    with bucket.client() as client:
        download_klines(["BTCUSDT"], ["1d"], months, raw_dir=tmp_path / "raw", client=client)
    assert len(cached_archives(tmp_path / "raw", "BTCUSDT", "1d")) == 2
    report = build_parquet("BTCUSDT", "1d", raw_dir=tmp_path / "raw", parquet_dir=tmp_path / "parquet")
    assert report.rows == 31 + 31 and report.is_clean
    path = parquet_path(tmp_path / "parquet", "BTCUSDT", "1d")
    assert json.loads(path.with_name("BTCUSDT.quality.json").read_text())["rows"] == 62
    assert pl.read_parquet(path).height == 62  # the file keeps the vault rows
    assert load_klines("BTCUSDT", "1d", data_dir=tmp_path).height == 31  # the sealed read does not


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
    }
)  # fmt: skip
# Names that read raw text or archives (the downloaded cache).
RAW_READERS = frozenset(
    {
        "read_csv", "scan_csv", "read_csv_batched", "read_klines_zip", "ZipFile", "genfromtxt", "loadtxt",
        "read_json", "read_ndjson", "scan_ndjson", "read_avro", "read_excel",
    }
)
# Names that load code or attributes by string, which no name filter can follow.
DYNAMIC_NAMES = frozenset({"import_module", "__import__"})
# ``seal`` is protocol's own lock: other code reaches data through load_klines, which computes ``known_at`` itself.
SEAL = "seal"
BANNED_MODULES = (
    "pyarrow", "fastparquet", "duckdb", "pandas", "csv", "zipfile", "importlib",
    "subprocess",  # shelling out to parquet-tools, duckdb, ...
    "pickle", "joblib", "shelve", "marshal",  # a cached object holding vault rows
)  # fmt: skip
DATA_LIBRARIES = ("polars", "pyarrow", "pandas", "numpy", "duckdb")  # getattr/vars on these is dynamic access
FILE_SUFFIXES = (".parquet", ".zip")
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

# Where a rule is allowed, by file relative to python/, each with its reason. Exact, not by pattern.
ALLOWED: dict[str, dict[str, str]] = {
    "olsprime/protocol.py": {
        "reader": "the one reader of stored klines, always followed by seal",
        "seal": "defines and calls the seal",
    },
    "olsprime/data/binance_vision.py": {
        "raw-reader": "parses the downloaded archives into the stored schema",
        "import:zipfile": "opens the downloaded archives",
    },
    "tests/fakes.py": {
        "raw-reader": "writes fake archives",
        "import:zipfile": "writes fake archives",
    },
    "tests/test_binance_vision.py": {"reader": "reads back the Parquet file the build just wrote"},
    "tests/test_universe.py": {"reader": "reads back the universe file the run just wrote"},
    "tests/test_protocol.py": {
        "reader": "shows that the stored file keeps the vault rows",
        "seal": "tests the seal itself",
    },
}


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
    return [text for rule, text in findings(source, filename) if rule not in allowed]


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
]
NOTEBOOK_CONTROLS = [
    ("nb-plain-load_klines", notebook("from olsprime.protocol import load_klines", "df = load_klines('BTCUSDT', '1h')")),
    ("nb-magics-only", notebook("%matplotlib inline\n!pip install -q torch\nimport numpy as np\nnp.zeros(3)")),
    ("nb-markdown-mentions-parquet", notebook("x = 1", markdown="we read x.parquet with pl.read_parquet here")),
    ("nb-empty", notebook()),
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
