"""The survey that feeds docs/DATA.md: aggregate quality, likely ticker reuse, universe summary."""

import json
from datetime import timedelta

import polars as pl

from olsprime.data import survey
from olsprime.data.binance_vision import parquet_path
from olsprime.data.quality import check_quality
from tests.fakes import DAY_MS
from tests.test_protocol import bars, store, utc


def report(symbol: str, **overrides) -> dict:
    base = {"symbol": symbol, "interval": "1d", "rows": 100, "gaps": []} | {k: 0 for k in survey.PROBLEMS}
    return base | overrides


def store_daily(data_dir, symbol: str, frame: pl.DataFrame) -> None:
    """The Parquet file and the quality report that ``build_parquet`` would write."""
    store(data_dir, frame, symbol, "1d")
    out = parquet_path(data_dir / "parquet", symbol, "1d").with_name(f"{symbol}.quality.json")
    out.write_text(json.dumps(check_quality(frame, "1d", symbol=symbol).to_dict()))


def with_gap(first, before: int, gap_days: int, after: int, symbol: str, jump: float) -> pl.DataFrame:
    """``before`` bars, ``gap_days`` missing days, then ``after`` bars whose prices are ``jump`` times higher."""
    head = bars(first, before, DAY_MS, symbol)
    tail = bars(first + timedelta(days=before + gap_days), after, DAY_MS, symbol)
    scaled = tail.with_columns(pl.col("open", "high", "low", "close") * jump)
    return pl.concat([head, scaled])


def test_aggregate_sums_every_problem_and_ranks_the_worst_pairs():
    reports = [
        report("AAAUSDT", missing_bars=3, gaps=[{}, {}]),
        report("BBBUSDT"),
        report("CCCUSDT", duplicate_rows=2, conflicting_duplicates=1, off_grid=5),
    ]
    total = survey.aggregate(reports)
    assert (total["pairs"], total["rows"], total["gaps"], total["pairs_with_gaps"], total["clean_pairs"]) == (3, 300, 2, 1, 1)
    assert (total["missing_bars"], total["duplicate_rows"], total["conflicting_duplicates"], total["off_grid"]) == (3, 2, 1, 5)
    assert [r["symbol"] for r in survey.worst(reports, 2)] == ["CCCUSDT", "AAAUSDT"]  # 7 problems, then 3
    assert survey.total_problems(reports[2]) == 7  # a conflicting duplicate is a duplicate: counted once


def test_ticker_reuse_needs_a_long_gap_and_a_big_jump(tmp_path):
    start = utc(2022, 1, 1)
    store_daily(tmp_path, "LUNAUSDT", with_gap(start, 20, 18, 20, "LUNAUSDT", jump=0.001))  # a different coin
    store_daily(tmp_path, "QUIETUSDT", with_gap(start, 20, 18, 20, "QUIETUSDT", jump=1.02))  # long gap, same level
    store_daily(tmp_path, "SHORTUSDT", with_gap(start, 20, 3, 20, "SHORTUSDT", jump=5.0))  # big jump, short gap
    store_daily(tmp_path, "CLEANUSDT", bars(start, 40, DAY_MS, "CLEANUSDT"))
    found = survey.ticker_reuse(tmp_path, ["LUNAUSDT", "QUIETUSDT", "SHORTUSDT", "CLEANUSDT", "ABSENTUSDT"])
    assert [r.symbol for r in found] == ["LUNAUSDT"]
    (luna,) = found
    assert luna.gap_start == utc(2022, 1, 21) and luna.gap_days == 18  # 18 missing bars
    assert luna.ratio < 0.01


def test_a_jump_upwards_is_flagged_too_and_a_gap_into_the_vault_is_not_judged(tmp_path):
    store_daily(tmp_path, "UPUSDT", with_gap(utc(2022, 1, 1), 10, 10, 10, "UPUSDT", jump=4.0))
    store_daily(tmp_path, "EDGEUSDT", with_gap(utc(2025, 12, 1), 20, 20, 5, "EDGEUSDT", jump=9.0))  # resumes in 2026
    assert [r.symbol for r in survey.ticker_reuse(tmp_path, ["UPUSDT", "EDGEUSDT"])] == ["UPUSDT"]


def test_universe_summary_counts_members_per_month():
    rows = [(utc(2020, m, 1), s) for m, symbols in ((1, "ABC"), (2, "AB"), (3, "ABCD")) for s in symbols]
    frame = pl.DataFrame(rows, schema=["rebalance_time", "symbol"], orient="row")
    summary = survey.universe_summary(frame)
    assert (summary["rebalances"], summary["min"], summary["median"], summary["max"], summary["distinct"]) == (3, 2, 3, 4, 4)
    assert summary["first"] == utc(2020, 1, 1) and summary["last"] == utc(2020, 3, 1)


def test_the_report_renders_a_table_per_interval(tmp_path):
    store_daily(tmp_path, "LUNAUSDT", with_gap(utc(2022, 1, 1), 20, 18, 20, "LUNAUSDT", jump=0.001))
    text = survey.render(tmp_path)
    assert "### Quality, 1d" in text and "LUNAUSDT" in text and "| pairs |" in text


def test_the_report_with_a_universe_splits_members_from_every_stored_pair(tmp_path):
    from olsprime.protocol import VAULT
    from olsprime.universe import UNIVERSE_SCHEMA, universe_path

    store_daily(tmp_path, "LUNAUSDT", with_gap(utc(2022, 1, 1), 20, 18, 20, "LUNAUSDT", jump=0.001))
    store_daily(tmp_path, "OTHERUSDT", bars(utc(2022, 1, 1), 40, DAY_MS, "OTHERUSDT"))
    rows = [(utc(2022, 1, 1), 1, "LUNAUSDT", 1e6, 30), (utc(2022, 2, 1), 1, "LUNAUSDT", 1e6, 30), (VAULT.start, 1, "LUNAUSDT", 1e6, 30)]
    universe = pl.DataFrame(rows, schema=list(UNIVERSE_SCHEMA), orient="row").cast(UNIVERSE_SCHEMA)
    path = universe_path(tmp_path / "parquet", 30)
    path.parent.mkdir(parents=True)
    universe.write_parquet(path)
    text = survey.render(tmp_path)
    assert "Quality, 1d (universe members)" in text and "Quality, 1d, every stored pair" in text
    assert "2 rebalances, 2022-01 to 2022-02" in text  # the vault rebalance is sealed away
    assert "| LUNAUSDT | 2022-01-21 | 18 |" in text
