"""The hourly panel: member months only, no survivorship, the cut at the end of development, h = 24 by time.

Every test runs on a fake store under a temporary data root: nothing reads the repository's ``data/``.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from olsprime.bench.synthetic import synthetic_bars
from olsprime.data.binance_vision import Month
from olsprime.data.klines import TIMESTAMP
from olsprime.features import FEATURE_COLUMNS, basic_features
from olsprime.features import panel as panel_module
from olsprime.features.panel import (
    PANEL_COLUMNS,
    fingerprint,
    h1_subpanel_members,
    hourly_panel,
    member_months,
    summarize,
)
from olsprime.labels import LABEL_COLUMNS, forward_labels
from olsprime.protocol import DEVELOPMENT, VaultSealed, known_at_expr, load_klines
from olsprime.universe import UNIVERSE_SCHEMA, universe_path
from tests.test_protocol import store, utc

HOUR = timedelta(hours=1)
START = utc(2024, 10, 1)
H = 24


def fake_bars(
    symbol: str,
    hours: int,
    *,
    seed: int,
    start: datetime = START,
    drop: tuple[datetime, ...] = (),
    rescale_after_development: float | None = None,
) -> pl.DataFrame:
    """Hourly bars in the stored schema; ``drop`` removes bars (gaps), the rescale rewrites 2025 prices."""
    bars = synthetic_bars(symbol, start=start, days=hours, interval="1h", seed=seed, vol=0.004, drift=0.0)
    if rescale_after_development is not None:
        late = pl.col("open_time") >= DEVELOPMENT.end
        k = rescale_after_development
        bars = bars.with_columns(
            *(pl.when(late).then(pl.col(c) * k).otherwise(pl.col(c)).alias(c) for c in ("open", "high", "low", "close")),
            pl.when(late).then(pl.col("quote_volume") * k * k).otherwise(pl.col("quote_volume")).alias("quote_volume"),
        )
    if drop:
        bars = bars.filter(~pl.col("open_time").is_in(list(drop)))
    return bars.drop("known_at")


def universe_frame(members: dict[str, list[tuple[int, int]]]) -> pl.DataFrame:
    """A universe table: ``{symbol: [(year, month), ...]}`` with ranks in alphabetical order per month."""
    rows = sorted((utc(y, m, 1), symbol) for symbol, months in members.items() for y, m in months)
    rank: dict[datetime, int] = {}
    data: dict[str, list] = {"rebalance_time": [], "rank": [], "symbol": [], "trailing_quote_volume": [], "days": []}
    for at, symbol in rows:
        rank[at] = rank.get(at, 0) + 1
        data["rebalance_time"].append(at)
        data["rank"].append(rank[at])
        data["symbol"].append(symbol)
        data["trailing_quote_volume"].append(1.0e6)
        data["days"].append(30)
    return pl.DataFrame(data, schema=UNIVERSE_SCHEMA)


NOV, DEC = (2024, 11), (2024, 12)
MEMBERS = {
    "AAAUSDT": [NOV, DEC],  # member throughout
    "BBBUSDT": [DEC],  # has bars in October and November, joins in December
    "CCCUSDT": [NOV, DEC],  # delisted: its last bar opens 2024-12-09 23:00
    "DDDUSDT": [NOV],  # leaves the universe in December, keeps trading
}
BARS_HOURS = {"AAAUSDT": 104 * 24, "BBBUSDT": 104 * 24, "CCCUSDT": 70 * 24, "DDDUSDT": 104 * 24}


def build_store(root: Path, *, rescale: float | None = None) -> pl.DataFrame:
    for seed, (symbol, hours) in enumerate(BARS_HOURS.items()):
        store(root, fake_bars(symbol, hours, seed=seed, rescale_after_development=rescale), symbol, "1h")
    return universe_frame(MEMBERS)


@pytest.fixture
def fake_store(tmp_path: Path) -> tuple[Path, pl.DataFrame]:
    return tmp_path, build_store(tmp_path)


@pytest.fixture
def panel(fake_store: tuple[Path, pl.DataFrame]) -> pl.DataFrame:
    root, universe = fake_store
    return hourly_panel(universe, data_dir=root)


def rows_of(panel: pl.DataFrame, symbol: str, month: tuple[int, int] | None = None) -> pl.DataFrame:
    frame = panel.filter(pl.col("symbol") == symbol)
    return frame if month is None else frame.filter(pl.col("month") == utc(*month, 1))


# --- Shape ---------------------------------------------------------------------------


def test_columns_types_and_the_label_columns_are_the_labels(panel):
    assert panel.columns == list(PANEL_COLUMNS)
    assert PANEL_COLUMNS[4 : 4 + len(FEATURE_COLUMNS)] == FEATURE_COLUMNS and len(FEATURE_COLUMNS) == 15
    assert PANEL_COLUMNS[-6:-1] == LABEL_COLUMNS and PANEL_COLUMNS[-1] == "complete"
    assert panel.schema["month"] == TIMESTAMP and panel.schema["decision_time"] == TIMESTAMP
    assert panel.schema["complete"] == pl.Boolean and panel.schema["label"] == pl.UInt8
    assert all(panel.schema[c] == pl.Float64 for c in (*FEATURE_COLUMNS, "threshold", "fwd_return"))


# --- Membership ----------------------------------------------------------------------


def test_rows_exist_only_in_member_months(panel, fake_store):
    _, universe = fake_store
    allowed = set(member_months(universe).iter_rows())
    assert set(panel.select("symbol", "month").unique().iter_rows()) <= allowed
    # the hours are all there: 30 days of November, 31 of December, the delisted pair to its last bar
    expected = {
        ("AAAUSDT", NOV): 720,
        ("AAAUSDT", DEC): 744,
        ("BBBUSDT", DEC): 744,
        ("CCCUSDT", NOV): 720,
        ("CCCUSDT", DEC): 9 * 24,
        ("DDDUSDT", NOV): 720,
    }
    counts = {(s, (m.year, m.month)): n for s, m, n in panel.group_by("symbol", "month").len().iter_rows()}
    assert counts == expected
    # the month is the month of the decision time
    assert (panel["month"] == panel["decision_time"].dt.truncate("1mo")).all()


def test_a_pair_contributes_nothing_before_its_first_member_month(panel):
    bbb = rows_of(panel, "BBBUSDT")
    assert bbb["decision_time"].min() >= utc(2024, 12, 1)
    assert rows_of(panel, "BBBUSDT", NOV).is_empty()
    assert rows_of(panel, "DDDUSDT", DEC).is_empty()  # left the universe: its bars still exist, its rows do not
    assert panel.filter(pl.col("symbol") == "DDDUSDT")["decision_time"].max() < utc(2024, 12, 1)


def test_features_use_the_whole_history_not_just_the_member_months(fake_store):
    root, universe = fake_store
    panel = hourly_panel(universe, data_dir=root)
    first = rows_of(panel, "BBBUSDT").sort("decision_time").head(1)
    assert first["decision_time"][0] == utc(2024, 12, 1) + HOUR - timedelta(milliseconds=1)
    assert all(first[c][0] is not None for c in FEATURE_COLUMNS)  # the 90-bar windows reach back into October
    # the same pair cut to its member months would still be in warm-up at that hour
    bars = load_klines("BBBUSDT", "1h", data_dir=root)
    december = bars.filter(pl.col("open_time") >= utc(2024, 12, 1))
    assert basic_features(december, "1h").head(1)["f_ret_90"][0] is None
    whole = basic_features(bars, "1h").filter(pl.col("open_time") == utc(2024, 12, 1))
    assert whole.select(FEATURE_COLUMNS).row(0) == first.select(FEATURE_COLUMNS).row(0)


def test_a_delisted_pair_keeps_its_rows_until_its_last_bar(panel):
    ccc = rows_of(panel, "CCCUSDT").sort("decision_time")
    assert ccc["open_time"].max() == utc(2024, 12, 9, 23)
    # its last 25 hours have no exit bar: the label is null, the row stays
    tail = ccc.tail(25)
    assert tail["label"].null_count() == 25 and ccc.head(ccc.height - 25)["label"].null_count() == 0


def test_member_months_guards_and_shape(fake_store):
    _, universe = fake_store
    months = member_months(universe)
    assert months.columns == ["symbol", "month"] and months.height == 6
    assert months.rows() == sorted(months.rows(), key=lambda r: (r[1], r[0]))
    assert member_months(universe, first=Month(2024, 12), last=Month(2024, 12))["symbol"].to_list() == [
        "AAAUSDT",
        "BBBUSDT",
        "CCCUSDT",
    ]
    with pytest.raises(ValueError, match="validation year"):
        member_months(universe, last=Month(2025, 1))
    with pytest.raises(VaultSealed):
        member_months(universe, last=Month(2026, 1))
    with pytest.raises(ValueError, match="before first"):
        member_months(universe, first=Month(2024, 12), last=Month(2024, 11))
    with pytest.raises(ValueError, match="twice"):
        member_months(pl.concat([universe, universe.head(1)]))
    off_month = universe.with_columns(rebalance_time=pl.col("rebalance_time") + timedelta(hours=1))
    with pytest.raises(ValueError, match="first instant"):
        member_months(off_month)


def test_months_of_the_validation_year_are_refused_by_the_panel(fake_store):
    root, universe = fake_store
    with pytest.raises(ValueError, match="validation year"):
        hourly_panel(universe, last=Month(2025, 1), data_dir=root)
    with pytest.raises(VaultSealed):
        hourly_panel(universe, last=Month(2026, 3), data_dir=root)
    january = universe_frame({"AAAUSDT": [NOV, (2025, 1)]})
    only_development = hourly_panel(january, data_dir=root)  # the 2025 rebalance is outside the default range
    assert set(only_development["month"]) == {utc(2024, 11, 1)}


def test_a_member_without_a_file_raises(fake_store):
    root, universe = fake_store
    ghost = pl.concat([universe, universe_frame({"ZZZUSDT": [DEC]})])
    with pytest.raises(OSError):
        hourly_panel(ghost, data_dir=root)


def test_the_symbols_argument_keeps_those_members_only(fake_store, panel):
    root, universe = fake_store
    sub = hourly_panel(universe, symbols=["CCCUSDT", "AAAUSDT"], data_dir=root)
    assert sub.equals(panel.filter(pl.col("symbol").is_in(["AAAUSDT", "CCCUSDT"])))
    with pytest.raises(ValueError, match="not members"):
        hourly_panel(universe, symbols=["AAAUSDT", "NOPEUSDT"], data_dir=root)


# --- The cut at the end of development -----------------------------------------------


def test_the_cut_nulls_the_labels_that_need_a_2025_bar(fake_store, panel):
    root, _ = fake_store
    aaa = rows_of(panel, "AAAUSDT", DEC).sort("open_time")
    last_labelled = utc(2024, 12, 30, 22)  # exit at t + 25 h = 2024-12-31 23:00, the last bar before the cut
    labelled = aaa.filter(pl.col("label").is_not_null())
    assert labelled["open_time"].max() == last_labelled
    assert labelled.filter(pl.col("open_time") == last_labelled)["exit_time"][0] == utc(2024, 12, 31, 23)
    late = aaa.filter(pl.col("open_time") > last_labelled)
    assert late.height == 25
    for name in ("entry_time", "exit_time", "label_known_at", "fwd_return", "label"):
        assert late[name].null_count() == 25, name
    assert not late["complete"].any()
    assert panel["label_known_at"].drop_nulls().max() < DEVELOPMENT.end
    # without the cut the same rows are labelled: the null is the cut's doing, not the fixture's
    full = forward_labels(load_klines("AAAUSDT", "1h", data_dir=root), "1h", H)
    uncut = full.filter(pl.col("open_time").is_in(late["open_time"].to_list()))
    assert uncut["label"].null_count() < 25
    assert uncut.filter(pl.col("open_time") == utc(2024, 12, 31, 22))["label"][0] is not None


def test_the_cut_keeps_every_earlier_label_exactly_as_forward_labels_computes_it(fake_store, panel):
    root, _ = fake_store
    for symbol in ("AAAUSDT", "BBBUSDT"):  # the two members of December: the cut touches their last 25 hours
        bars = load_klines(symbol, "1h", data_dir=root)  # the full bars, 2025 included
        assert bars["open_time"].max() > DEVELOPMENT.end
        full = forward_labels(bars, "1h", H)
        mine = rows_of(panel, symbol).sort("open_time")
        reference = full.filter(pl.col("open_time").is_in(mine["open_time"].to_list())).sort("open_time")
        known = (reference["label_known_at"] < DEVELOPMENT.end).fill_null(False)
        cols = ["decision_time", "entry_time", "exit_time", "label_known_at", "fwd_return", "threshold", "label"]
        assert known.sum() > 0 and (~known).sum() >= 25
        assert mine.filter(known).select(cols).equals(reference.filter(known).select(cols)), symbol
        # the labels that needed a later bar are null in the panel, and only those
        assert mine.filter(~known)["label"].null_count() == (~known).sum(), symbol
        # the features are those of the full bars as well
        feats = basic_features(bars, "1h").filter(pl.col("open_time").is_in(mine["open_time"].to_list())).sort("open_time")
        assert feats.select(FEATURE_COLUMNS).equals(mine.select(FEATURE_COLUMNS)), symbol


def test_nothing_from_2025_reaches_the_panel(fake_store, tmp_path_factory, panel):
    _, universe = fake_store
    other = tmp_path_factory.mktemp("rewritten_future")
    build_store(other, rescale=7.0)  # every 2025 price and volume rewritten
    twin = hourly_panel(universe, data_dir=other)
    assert twin.equals(panel) and fingerprint(twin) == fingerprint(panel)


# --- h = 24 by time -----------------------------------------------------------------


def test_the_horizon_is_found_by_time_across_gaps(tmp_path):
    t1, t2, t3 = utc(2024, 11, 10), utc(2024, 11, 12), utc(2024, 11, 14)
    holes = (t1 + 10 * HOUR, t2 + 25 * HOUR, t3 + 1 * HOUR)
    whole = fake_bars("AAAUSDT", 104 * 24, seed=7)
    store(tmp_path, fake_bars("AAAUSDT", 104 * 24, seed=7, drop=holes), "AAAUSDT", "1h")
    panel = hourly_panel(universe_frame({"AAAUSDT": [NOV, DEC]}), data_dir=tmp_path)
    opens = dict(zip(whole["open_time"].to_list(), whole["open"].to_list(), strict=True))

    def row(at: datetime) -> dict:
        return panel.filter(pl.col("open_time") == at).to_dicts()[0]

    # a hole between the entry and the exit does not matter: both prices print exactly 1 and 25 hours later
    inside = row(t1)
    assert inside["entry_time"] == t1 + HOUR and inside["exit_time"] == t1 + 25 * HOUR
    assert inside["fwd_return"] == pytest.approx(math.log(opens[t1 + 25 * HOUR] / opens[t1 + HOUR]), rel=1e-12)
    assert inside["label_known_at"] == inside["exit_time"] and inside["label"] is not None
    # no bar opens at t + 25 h: the row stays, its label is null (not the bar 25 rows later)
    assert row(t2)["label"] is None and row(t2)["exit_time"] is None
    # no bar opens at t + 1 h
    assert row(t3)["label"] is None and row(t3)["entry_time"] is None
    # the features see the hole too: the bar after it has no one-bar return
    assert row(t1 + 11 * HOUR)["f_ret_1"] is None and row(t1 + 9 * HOUR)["f_ret_1"] is not None
    # and the label is the same one-day label for the others
    assert row(t1 + 2 * HOUR)["exit_time"] == t1 + 27 * HOUR


def test_a_different_horizon_moves_the_exit(fake_store):
    root, universe = fake_store
    short = hourly_panel(universe, h=6, symbols=["AAAUSDT"], data_dir=root).sort("open_time").head(1)
    assert short["exit_time"][0] - short["entry_time"][0] == 6 * HOUR


# --- Determinism, order, completeness -------------------------------------------------


def test_determinism_and_ordering(fake_store, panel):
    root, universe = fake_store
    again = hourly_panel(universe, data_dir=root)
    shuffled = hourly_panel(universe.sample(fraction=1.0, shuffle=True, seed=3), data_dir=root)
    assert again.equals(panel) and shuffled.equals(panel)
    assert fingerprint(again) == fingerprint(panel)
    keys = panel.select("decision_time", "symbol")
    assert keys.equals(keys.sort(["decision_time", "symbol"]))
    assert keys.is_duplicated().sum() == 0
    assert fingerprint(panel.head(100)) != fingerprint(panel)


def test_complete_means_every_feature_and_the_label(tmp_path):
    hole = utc(2024, 11, 15, 10)
    store(tmp_path, fake_bars("AAAUSDT", 104 * 24, seed=4, drop=(hole,)), "AAAUSDT", "1h")
    panel = hourly_panel(universe_frame({"AAAUSDT": [NOV, DEC]}), data_dir=tmp_path)
    names = (*FEATURE_COLUMNS, "label")
    expected = panel.select(pl.all_horizontal([pl.col(c).is_not_null() for c in names]).alias("complete"))
    assert panel["complete"].equals(expected["complete"])
    flags = dict(zip(panel["open_time"].to_list(), panel["complete"].to_list(), strict=True))
    assert flags[utc(2024, 11, 3)] is True
    assert flags[hole + HOUR] is False  # no one-bar return across the hole
    assert flags[hole - HOUR] is False  # its entry bar is the hole
    assert flags[hole - 25 * HOUR] is False  # its exit bar is the hole
    assert flags[hole + 95 * HOUR] is True  # the 90-bar windows have left the hole behind
    assert flags[utc(2024, 12, 31, 5)] is False  # the label needs a 2025 bar
    incomplete = panel.filter(~pl.col("complete"))
    assert incomplete.select(pl.any_horizontal([pl.col(c).is_null() for c in names]).all()).item()
    assert panel["complete"].sum() > 0.9 * panel.height


# --- The sub-panel -------------------------------------------------------------------


def test_subpanel_takes_the_most_member_months_and_breaks_ties_by_symbol():
    # nine symbols with distinct counts 12..4; K, L, M tie at 3; N has 1. M also has five months in 2025,
    # which would put it ahead of K and L if the validation year were counted.
    months = [(2024, m) for m in range(1, 13)]
    members: dict[str, list[tuple[int, int]]] = {
        f"{name}USDT": months[:count] for name, count in zip("ABCDEFGHI", range(12, 3, -1), strict=True)
    }
    members.update({"MUSDT": months[:3] + [(2025, m) for m in range(1, 6)], "LUSDT": months[3:6], "KUSDT": months[6:9]})
    members["NUSDT"] = months[:1]
    universe = universe_frame(members)
    chosen = h1_subpanel_members(universe)
    assert chosen == [f"{n}USDT" for n in "ABCDEFGHI"] + ["KUSDT"]
    assert h1_subpanel_members(universe, size=3) == ["AUSDT", "BUSDT", "CUSDT"]
    # a plain tie at the cut goes to the alphabet, whatever the row order
    flipped = universe.sample(fraction=1.0, shuffle=True, seed=1)
    assert h1_subpanel_members(flipped) == chosen


# --- Rows from bars (the panel H1) ---------------------------------------------------


def test_a_row_belongs_to_the_month_of_its_bars_open_even_when_it_closes_late():
    last_hour = utc(2024, 11, 30, 23)
    late = pl.col("open_time") == last_hour
    bars = fake_bars("AAAUSDT", 104 * 24, seed=0)
    bars = bars.with_columns(  # closes 1 ms late, at 2024-12-01 00:00:00.000, as the bar of 2017-09-06 15:00 does
        pl.when(late).then(pl.col("close_time") + timedelta(milliseconds=1)).otherwise(pl.col("close_time")).alias("close_time")
    ).with_columns(known_at_expr("1h"))
    assert bars.filter(late)["known_at"].item() == utc(2024, 12, 1)
    november = panel_module.pair_rows(bars, [utc(2024, 11, 1)])
    december = panel_module.pair_rows(bars, [utc(2024, 12, 1)])
    row = november.filter(late)
    assert row["month"].item() == utc(2024, 11, 1) and row["decision_time"].item() == utc(2024, 12, 1)
    assert december.filter(late).is_empty()
    assert (november.height, december.height) == (30 * 24, 31 * 24)


def test_pair_rows_and_assemble_rebuild_the_panel_from_bars(fake_store, panel):
    root, universe = fake_store
    parts = []
    for (symbol,), rows in member_months(universe).group_by("symbol", maintain_order=True):
        bars = load_klines(symbol, "1h", data_dir=root)  # every stored column, 2025 included: pair_rows cuts and selects
        assert bars["open_time"].max() > DEVELOPMENT.end or symbol in ("CCCUSDT", "DDDUSDT")
        parts.append(panel_module.pair_rows(bars, rows["month"].to_list()))
    assert panel_module.assemble(reversed(parts)).equals(panel)
    two = pl.concat([load_klines(s, "1h", data_dir=root) for s in ("AAAUSDT", "BBBUSDT")])
    with pytest.raises(ValueError, match="one pair"):
        panel_module.pair_rows(two, [utc(2024, 12, 1)])
    assert panel_module.pair_rows(two.clear(), [utc(2024, 12, 1)]).is_empty()
    with pytest.raises(ValueError, match="no member"):
        panel_module.assemble([pl.DataFrame()])


# --- The summary and the command -----------------------------------------------------


def test_summary_counts_match_the_panel(fake_store, panel):
    _, universe = fake_store
    summary = summarize(panel, universe, first=Month(2024, 11), last=Month(2024, 12))
    assert [m["month"] for m in summary["months"]] == ["2024-11", "2024-12"]
    nov, dec = summary["months"]
    assert (nov["members"], nov["members_with_rows"], nov["rows"]) == (3, 3, 3 * 720)
    assert (dec["members"], dec["members_with_rows"]) == (3, 3)
    assert summary["totals"]["rows"] == panel.height == nov["rows"] + dec["rows"]
    assert summary["totals"]["complete"] == int(panel["complete"].sum())
    assert summary["totals"]["labelled"] == panel["label"].drop_nulls().len()
    assert sum(summary["totals"]["class_share_complete"].values()) == pytest.approx(1.0)
    assert summary["distinct_symbols"] == 4 and summary["member_months"] == 6
    assert summary["fingerprint_sha256"] == fingerprint(panel)
    assert len(summary["subpanel_members"]) == 4


def test_the_command_prints_and_writes_the_summary(fake_store, tmp_path_factory, capsys):
    root, universe = fake_store
    path = universe_path(root / "parquet", 30)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.write_parquet(path)
    out = tmp_path_factory.mktemp("runs_out")
    argv = ["--summary", "--data-dir", str(root), "--out-dir", str(out), "--first", "2024-11", "--last", "2024-12"]
    assert panel_module.main(argv) == 0
    printed = capsys.readouterr().out
    assert "2024-11" in printed and "sub-panel members" in printed and "fingerprint" in printed
    assert panel_module.main(argv) == 0  # a second run does not overwrite the first
    written = sorted(out.glob("*-panel*/summary.json"))
    assert len(written) == 2 and written[0].parent.name != written[1].parent.name
    body = json.loads(written[0].read_text(encoding="utf-8"))
    expected = hourly_panel(universe, first=Month(2024, 11), last=Month(2024, 12), data_dir=root)
    assert body["totals"]["rows"] == expected.height and body["fingerprint_sha256"] == fingerprint(expected)
    assert body["run"]["wall_seconds"] >= 0 and body["cut"].endswith("2025-01-01")
    assert panel_module.main([]) == 0  # no flag: help only
    assert "usage" in capsys.readouterr().out.lower()


def test_the_default_universe_is_the_sealed_universe_file(fake_store, panel):
    root, universe = fake_store
    path = universe_path(root / "parquet", 30)
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.write_parquet(path)
    assert hourly_panel(data_dir=root).equals(panel)
