"""Walk-forward folds: purge, embargo, guards, several symbols, and a leak proof through the labels."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from olsprime.data.binance_vision import Month
from olsprime.eval.splits import Fold, describe, walk_forward
from olsprime.features import basic_features
from olsprime.labels import LABEL_COLUMNS, forward_labels
from olsprime.protocol import DEVELOPMENT, VALIDATION, VAULT, Period, VaultSealed
from tests.test_features import make_bars, perturb_after

START = datetime(2018, 1, 1, tzinfo=UTC)
START_MS = int(START.timestamp()) * 1000
FAR = datetime(2027, 1, 1, tzinfo=UTC)


def build_frame(bars: pl.DataFrame, h: int, interval: str = "1d") -> pl.DataFrame:
    labels = forward_labels(bars, interval, h)
    return labels.join(basic_features(bars, interval).drop("decision_time"), on=["symbol", "open_time"], how="left")


def day_of(when: datetime) -> int:
    return (when - START).days


def frame_days(frame: pl.DataFrame, idx: np.ndarray) -> list[int]:
    return [day_of(t) for t in frame["open_time"].gather(idx).to_list()]


def labeled_before_fit(frame: pl.DataFrame, fit_time: datetime) -> np.ndarray:
    """Rows a naive splitter would train on: decided before the fit, label present, no purge."""
    return np.flatnonzero((frame["decision_time"] < fit_time).to_numpy() & frame["label"].is_not_null().to_numpy())


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return make_bars(1100, start_ms=START_MS, seed=21)  # 2018-01-01 .. 2021-01-04


def folds_for(frame, h=7, **kw):
    kw.setdefault("start", Month(2019, 1))
    kw.setdefault("end", Month(2020, 6))
    kw.setdefault("min_train_rows", 100)
    return walk_forward(frame, interval="1d", h=h, **kw)


def test_one_fold_per_month_with_the_months_decisions_as_test(bars):
    frame = build_frame(bars, 7)
    folds = folds_for(frame)
    assert [f.test_start for f in folds] == [Month(2019, 1).shift(i).start() for i in range(18)]
    for f in folds:
        assert f.fit_time == f.test_start and f.test_end == Month.of(f.test_start).shift(1).start()
        decisions = frame["decision_time"].gather(f.test_idx)
        assert decisions.min() >= f.test_start and decisions.max() < f.test_end
        assert f.test_idx.size == (f.test_end - f.test_start).days  # every decision of the month
        assert np.all(np.diff(f.test_idx) > 0) and np.all(np.diff(f.train_idx) > 0)


@pytest.mark.parametrize("h, embargo", [(1, None), (7, None), (7, 0), (3, 10)])
def test_no_training_label_is_unknown_at_the_cutoff(bars, h, embargo):
    frame = build_frame(bars, h)
    folds = folds_for(frame, h=h, embargo_bars=embargo)
    assert folds
    for f in folds:
        eb = h if embargo is None else embargo
        assert f.train_cutoff == f.fit_time - timedelta(days=eb)
        known = frame["label_known_at"].gather(f.train_idx)
        assert known.null_count() == 0 and frame["label"].gather(f.train_idx).null_count() == 0
        assert known.max() <= f.train_cutoff
        assert f.last_label_known_at == known.max()
        assert not set(f.train_idx.tolist()) & set(f.test_idx.tolist())
        assert frame["decision_time"].gather(f.train_idx).max() < f.test_start


def test_h7_purge_drops_exactly_the_expected_rows(bars):
    """label_known_at = open of day d + 8, cutoff = fit - 7 days -> d <= fit - 15. Labels exist from day 30."""
    frame = build_frame(bars, 7)
    for f in folds_for(frame, h=7):
        fit_day = day_of(f.fit_time)
        assert frame_days(frame, f.train_idx) == list(range(30, fit_day - 15 + 1))
        # A splitter without purge or embargo trains on every labeled row decided before the fit:
        # the 14 days fit-14 .. fit-1 are what the purge and the embargo hold back.
        naive = labeled_before_fit(frame, f.fit_time)
        held_back = sorted(set(naive.tolist()) - set(f.train_idx.tolist()))
        assert frame_days(frame, np.array(held_back)) == list(range(fit_day - 14, fit_day))


def test_purge_alone_embargo_zero_and_h1(bars):
    frame = build_frame(bars, 7)
    fit_day = day_of(folds_for(frame, h=7)[0].fit_time)
    f0 = folds_for(frame, h=7, embargo_bars=0)[0]  # purge only: d + 8 <= fit_day
    assert frame_days(frame, f0.train_idx)[-1] == fit_day - 8
    frame1 = build_frame(bars, 1)
    f1 = folds_for(frame1, h=1)[0]  # label_known_at = d + 2, embargo 1: d + 2 <= fit_day - 1
    assert frame_days(frame1, f1.train_idx)[-1] == fit_day - 3


def test_training_labels_depend_only_on_data_printed_by_the_cutoff(bars):
    """End-to-end leak proof: change every bar after the cutoff; the training rows' labels do not move."""
    h = 7
    frame = build_frame(bars, h)
    cols = ["open_time", *LABEL_COLUMNS, "threshold"]
    for month in (Month(2019, 3), Month(2020, 1)):
        f = folds_for(frame, h=h, start=month, end=month)[0]
        cutoff_day = day_of(f.train_cutoff)  # the bar opening at the cutoff prints at the cutoff: usable
        changed = build_frame(perturb_after(bars, cutoff_day), h)
        assert frame[cols].gather(f.train_idx).equals(changed[cols].gather(f.train_idx))
        # control: the rows a purge-less splitter would add do depend on later bars
        naive = labeled_before_fit(frame, f.fit_time)
        assert not frame[cols].gather(naive).equals(changed[cols].gather(naive))


def test_null_labels_never_train_and_unlabeled_test_rows_are_kept():
    bars = make_bars(900, start_ms=START_MS, seed=5, drop=tuple(range(300, 306)))
    frame = build_frame(bars, 3)
    folds = folds_for(frame, h=3, start=Month(2018, 12), end=Month(2019, 3))
    assert len(folds) == 4
    for f in folds:
        assert frame["label"].gather(f.train_idx).null_count() == 0
        assert frame["label_known_at"].gather(f.train_idx).max() <= f.train_cutoff
    # 2018-10-28 is day 300: the six missing days (and the rows whose entry/exit fall in the hole) are
    # unlabeled; they were in the training window of later folds only as absent rows.
    assert frame.height == 894 and frame["label"].null_count() > 30


def test_unlabeled_rows_in_a_test_month_are_test_rows():
    frame = build_frame(make_bars(900, start_ms=START_MS, seed=5), 3)
    f = folds_for(frame, h=3, start=Month(2020, 6), end=Month(2020, 6))[0]  # data ends 2020-06-18
    labels = frame["label"].gather(f.test_idx)
    assert labels.null_count() == 3 + 1 and f.test_idx.size == 18  # the last 4 days cannot be labeled yet


def test_small_folds_are_skipped_and_logged(bars, caplog):
    frame = build_frame(bars, 1)
    with caplog.at_level(logging.WARNING, logger="olsprime.eval.splits"):
        folds = walk_forward(frame, interval="1d", h=1, start=Month(2018, 1), end=Month(2018, 12), min_train_rows=200)
    skipped = [r for r in caplog.records if "skipped" in r.getMessage()]
    assert skipped and "min_train_rows" in skipped[0].getMessage()
    assert all(f.train_idx.size >= 200 for f in folds)
    assert len(folds) + len(skipped) == 12
    assert folds[0].test_start >= datetime(2018, 8, 1, tzinfo=UTC)


def test_default_min_train_rows_is_365(bars):
    frame = build_frame(bars, 1)
    folds = walk_forward(frame, interval="1d", h=1, start=Month(2018, 1), end=Month(2019, 12))
    assert folds[0].train_idx.size >= 365 and folds[0].test_start >= datetime(2019, 2, 1, tzinfo=UTC)


def test_period_guard_and_vault_guard():
    frame = build_frame(make_bars(3000, start_ms=START_MS, seed=9), 1)  # runs far into 2026
    kw = {"interval": "1d", "h": 1, "min_train_rows": 10}
    # the development period's last month is allowed: its end is the period's end
    assert walk_forward(frame, start=Month(2024, 12), end=Month(2024, 12), **kw)
    with pytest.raises(ValueError, match="outside the period"):
        walk_forward(frame, start=Month(2025, 1), end=Month(2025, 1), **kw)
    with pytest.raises(ValueError, match="outside the period"):
        walk_forward(frame, start=Month(2017, 7), end=Month(2018, 2), **kw)
    # validation works when asked, up to the vault's first instant
    assert walk_forward(frame, start=Month(2025, 12), end=Month(2025, 12), period=VALIDATION, **kw)
    # the vault is refused whatever the period says
    everything = Period(datetime(2017, 8, 1, tzinfo=UTC), FAR)
    for period in (DEVELOPMENT, VALIDATION, everything, VAULT):
        with pytest.raises(VaultSealed):
            walk_forward(frame, start=Month(2026, 1), end=Month(2026, 1), period=period, **kw)
        with pytest.raises(VaultSealed):
            walk_forward(frame, start=Month(2025, 11), end=Month(2026, 2), period=period, **kw)
    with pytest.raises(VaultSealed):
        walk_forward(frame, start=Month(2026, 10), end=Month(2026, 10), period=everything, **kw)


def test_the_guards_run_before_any_fold_is_built():
    frame = build_frame(make_bars(3000, start_ms=START_MS, seed=9), 1)
    with pytest.raises(VaultSealed):  # a forbidden month at the end of the range stops the whole call
        walk_forward(
            frame, interval="1d", h=1, start=Month(2024, 1), end=Month(2026, 1),
            min_train_rows=10, period=Period(START, FAR),
        )  # fmt: skip


def test_several_symbols_are_cut_by_time_across_all():
    a = make_bars(1100, start_ms=START_MS, seed=1, symbol="AAAUSDT")
    b = make_bars(1100, start_ms=START_MS, seed=2, symbol="BBBUSDT", drop=(400, 401))
    frame_a, frame_b = build_frame(a, 3), build_frame(b, 3)
    frame = pl.concat([frame_a, frame_b])
    kw = {"interval": "1d", "h": 3, "start": Month(2019, 1), "end": Month(2019, 12), "min_train_rows": 100}
    both, one_a, one_b = walk_forward(frame, **kw), walk_forward(frame_a, **kw), walk_forward(frame_b, **kw)
    assert len(both) == len(one_a) == len(one_b) == 12
    for fb, fa, fo in zip(both, one_a, one_b, strict=True):
        assert fb.train_idx.size == fa.train_idx.size + fo.train_idx.size
        assert fb.test_idx.size == fa.test_idx.size + fo.test_idx.size
        assert set(frame["symbol"].gather(fb.test_idx).to_list()) == {"AAAUSDT", "BBBUSDT"}
        assert frame["label_known_at"].gather(fb.train_idx).max() <= fb.train_cutoff
        assert not set(fb.train_idx.tolist()) & set(fb.test_idx.tolist())
        assert fb.train_idx[fb.train_idx < frame_a.height].tolist() == fa.train_idx.tolist()  # A comes first
        b_rows = fb.train_idx[fb.train_idx >= frame_a.height] - frame_a.height
        assert b_rows.tolist() == fo.train_idx.tolist()


def test_describe_has_one_row_per_fold(bars):
    folds = folds_for(build_frame(bars, 7), h=7)
    table = describe(folds)
    assert table.height == len(folds)
    assert table.columns == [
        "fit_time", "test_start", "test_end", "train_cutoff", "n_train", "n_test", "last_label_known_at",
    ]  # fmt: skip
    assert table["n_train"].to_list() == [f.train_idx.size for f in folds]
    assert table["n_test"].to_list() == [f.test_idx.size for f in folds]
    assert (table["last_label_known_at"] <= table["train_cutoff"]).all()
    assert describe([]).height == 0


def test_frame_checks_and_fold_is_immutable(bars):
    frame = build_frame(bars, 1)
    span = {"start": Month(2019, 1), "end": Month(2019, 2)}
    with pytest.raises(ValueError, match="lacks columns"):
        walk_forward(frame.drop("label"), interval="1d", h=1, **span)
    with pytest.raises(ValueError, match="horizon"):
        walk_forward(frame, interval="1d", h=0, **span)
    with pytest.raises(ValueError, match="embargo"):
        walk_forward(frame, interval="1d", h=1, embargo_bars=-1, **span)
    with pytest.raises(TypeError):
        walk_forward(frame.with_columns(pl.col("label_known_at").dt.epoch("ms")), interval="1d", h=1, **span)
    broken = frame.with_columns(label_known_at=pl.col("decision_time") - pl.duration(days=1))
    with pytest.raises(ValueError, match="known before its own decision"):
        walk_forward(broken, interval="1d", h=1, **span)
    fold = folds_for(frame, h=1)[0]
    assert isinstance(fold, Fold)
    with pytest.raises(AttributeError):
        fold.train_idx = np.array([0])  # type: ignore[misc]
