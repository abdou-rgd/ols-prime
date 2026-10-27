"""Walk-forward folds with purge and embargo (``docs/PROTOCOL.md``, "Splits").

One fold per calendar month. The model of a fold is fitted at ``fit_time``, 00:00 UTC
on the month's first day, and predicts every decision taken during that month.

A row trains a fold only if its label is not null and was already known at the
training cutoff::

    label_known_at <= train_cutoff = fit_time - embargo_bars * interval

* The **purge** is the condition itself: a label that is not yet known at the fit (its
  exit price has not printed) never trains the model.
* The **embargo** widens the gap by ``embargo_bars`` more bars; the default is h, the
  protocol's "purge and embargo equal the horizon".

The window expands from the first row. Folds cut by time across all symbols of the
frame. Test rows are every row whose ``decision_time`` falls in the month, including
rows whose label is null (their features are real; the evaluation drops the unlabeled
ones, and the fold records how many there are in :func:`describe`).

The vault is guarded here as well as in ``load_klines``: a test month that touches
``VAULT`` raises :class:`VaultSealed`, whatever ``period`` says.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from olsprime.data.binance_vision import Month, month_range
from olsprime.data.klines import TIMESTAMP, interval_ms
from olsprime.protocol import DEVELOPMENT, VAULT, Period, VaultSealed

log = logging.getLogger(__name__)

_REQUIRED = ("decision_time", "label_known_at", "label")


@dataclass(frozen=True, eq=False)
class Fold:
    """One monthly refit. ``train_idx`` and ``test_idx`` are row indices into the frame, ascending."""

    fit_time: datetime  # = test_start: the model is fitted at 00:00 UTC on the month's first day
    test_start: datetime
    test_end: datetime  # exclusive, the next month's start
    train_cutoff: datetime  # fit_time - embargo
    train_idx: np.ndarray
    test_idx: np.ndarray
    last_label_known_at: datetime | None = None  # latest label_known_at among the training rows


def _epoch_ms(when: datetime) -> int:
    return int(when.timestamp() * 1000)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def walk_forward(
    frame: pl.DataFrame,
    *,
    interval: str,
    h: int,
    start: Month,
    end: Month,
    embargo_bars: int | None = None,
    min_train_rows: int = 365,
    period: Period = DEVELOPMENT,
) -> list[Fold]:
    """Monthly walk-forward folds from ``start`` to ``end``, both months included.

    ``frame`` holds ``decision_time``, ``label_known_at`` and ``label`` (features joined
    to labels on ``(symbol, open_time)``; several symbols are fine). Raises
    :class:`VaultSealed` if any test month reaches ``VAULT.start`` (a training label cannot
    then reach it: ``label_known_at <= train_cutoff <= fit_time < VAULT.start``), and
    ``ValueError`` if a test month falls outside ``period``. A month with fewer than
    ``min_train_rows`` training rows, or with no test row, is skipped and logged.
    """
    step = interval_ms(interval)
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError(f"horizon h must be an integer >= 1, got {h!r}")
    embargo_bars = h if embargo_bars is None else embargo_bars
    if embargo_bars < 0:
        raise ValueError(f"embargo_bars must be >= 0, got {embargo_bars}")
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"frame lacks columns {missing}")
    for name in ("decision_time", "label_known_at"):
        if frame.schema[name] != TIMESTAMP:
            raise TypeError(f"{name} is {frame.schema[name]}, expected {TIMESTAMP}")

    months = month_range(start, end)
    # Guards first, so that nothing is computed for a forbidden range. The vault comes before the period:
    # no ``period`` argument can open it.
    for month in months:
        if month.shift(1).start() > VAULT.start:
            raise VaultSealed(f"test month {month} reaches the vault, which opens at {VAULT.start:%Y-%m-%d}")
    for month in months:
        if month.start() not in period or month.shift(1).start() > period.end:
            raise ValueError(f"test month {month} is outside the period {period.start:%Y-%m-%d} .. {period.end:%Y-%m-%d}")

    decision = frame["decision_time"].dt.epoch("ms").to_numpy()
    known_valid = frame["label_known_at"].is_not_null() & frame["label"].is_not_null()
    trainable = known_valid.to_numpy()
    known = frame["label_known_at"].dt.epoch("ms").fill_null(0).to_numpy()
    if np.any(trainable & (known < decision)):
        raise ValueError("a label is known before its own decision time: label_known_at < decision_time")

    folds: list[Fold] = []
    for month in months:
        test_start, test_end = month.start(), month.shift(1).start()
        cutoff = test_start - timedelta(milliseconds=embargo_bars * step)
        cutoff_ms = _epoch_ms(cutoff)
        train_idx = np.flatnonzero(trainable & (known <= cutoff_ms))
        test_idx = np.flatnonzero((decision >= _epoch_ms(test_start)) & (decision < _epoch_ms(test_end)))
        if train_idx.size < min_train_rows:
            log.warning("fold %s skipped: %d training rows < min_train_rows=%d", month, train_idx.size, min_train_rows)
            continue
        if test_idx.size == 0:
            log.warning("fold %s skipped: no test row", month)
            continue
        folds.append(
            Fold(
                fit_time=test_start,
                test_start=test_start,
                test_end=test_end,
                train_cutoff=cutoff,
                train_idx=train_idx,
                test_idx=test_idx,
                last_label_known_at=_from_ms(int(known[train_idx].max())),
            )
        )
    return folds


def describe(folds: list[Fold]) -> pl.DataFrame:
    """One row per fold: dates, train and test counts, and the last ``label_known_at`` used."""
    return pl.DataFrame(
        {
            "fit_time": [f.fit_time for f in folds],
            "test_start": [f.test_start for f in folds],
            "test_end": [f.test_end for f in folds],
            "train_cutoff": [f.train_cutoff for f in folds],
            "n_train": [int(f.train_idx.size) for f in folds],
            "n_test": [int(f.test_idx.size) for f in folds],
            "last_label_known_at": [f.last_label_known_at for f in folds],
        },
        schema={
            "fit_time": TIMESTAMP,
            "test_start": TIMESTAMP,
            "test_end": TIMESTAMP,
            "train_cutoff": TIMESTAMP,
            "n_train": pl.Int64,
            "n_test": pl.Int64,
            "last_label_known_at": TIMESTAMP,
        },
    )
