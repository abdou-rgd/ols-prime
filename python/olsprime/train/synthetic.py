"""Synthetic panels for the encoder's tests and its sizing. Nothing here reads data.

- :func:`small_panel` builds a real panel (features, labels, ``complete``) and its windows from
  seeded synthetic hourly bars, through :func:`olsprime.features.panel.pair_rows`,
  :func:`~olsprime.features.panel.assemble` and :func:`olsprime.models.encoder.build_windows`, the
  calls the real runs make.
- :func:`sizing_data` builds arrays of the real panel's shape directly (30 members every hour,
  hourly rows, 64-bar windows, a share of complete rows), fast enough for 1.6 million rows. Its
  values are noise: it measures time and memory, not skill.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import polars as pl

from olsprime.bench.synthetic import synthetic_bars
from olsprime.features.panel import assemble, pair_rows
from olsprime.models.encoder import N_QUANTITIES, WINDOW, Windows
from olsprime.models.matrix import PanelArrays
from olsprime.train.fit import EncoderData, encoder_data

HOUR_MS = 3_600_000

# The real development panel's share of complete rows (docs/DATA.md: 1,591,462 of 1,655,881).
COMPLETE_SHARE = 1_591_462 / 1_655_881


def small_panel(symbols: Sequence[str] = ("AAAUSDT", "BBBUSDT"), *, start: datetime = datetime(2018, 1, 1, tzinfo=UTC), hours: int = 24 * 120,
                seed: int = 0, gap_at: int | None = None) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:  # fmt: skip
    """A panel of hourly rows over ``hours`` bars per pair, every month a member month, and its bars.

    ``gap_at`` drops that bar of the first pair (a hole in the grid), to exercise the flags.
    """
    bars: dict[str, pl.DataFrame] = {}
    parts = []
    for i, symbol in enumerate(symbols):
        frame = synthetic_bars(symbol, start=start, days=hours, interval="1h", seed=seed * 100 + i, drift=0.0, vol=0.01)
        if gap_at is not None and i == 0:
            frame = frame.filter(pl.int_range(pl.len()) != gap_at)
        bars[symbol] = frame
        months = sorted(set(frame["open_time"].dt.truncate("1mo").to_list()))
        parts.append(pair_rows(frame, months))
    return assemble(parts), bars


def small_data(**kwargs) -> EncoderData:
    panel, bars = small_panel(**kwargs)
    return encoder_data(panel, bars)


def sizing_data(*, members: int = 30, start: datetime = datetime(2017, 10, 1, tzinfo=UTC), end: datetime = datetime(2025, 1, 1, tzinfo=UTC),
                complete_share: float = COMPLETE_SHARE, seed: int = 0) -> EncoderData:  # fmt: skip
    """``members`` pairs, one row per pair and hour in [start, end), with the real panel's timing.

    Bar t opens at ``open``; its decision is ``open + 1 h - 1 ms``, its label is known at
    ``open + 25 h``. Labels are drawn with the real class shares; a row is complete with
    probability ``complete_share``. Every pair has 64 bars before ``start``, so every window is valid.
    """
    rng = np.random.default_rng(seed)
    t0 = int(start.timestamp() * 1000)
    hours = int((int(end.timestamp() * 1000) - t0) // HOUR_MS)
    open_h = t0 + HOUR_MS * np.arange(hours, dtype=np.int64)
    n = hours * members
    open_ms = np.repeat(open_h, members)
    symbol = np.tile(np.arange(members, dtype=np.int64), hours)
    decision = open_ms + HOUR_MS - 1
    lka = open_ms + 25 * HOUR_MS
    label = rng.choice(3, size=n, p=[0.294, 0.436, 0.270]).astype(np.int64)
    complete = rng.random(n) < complete_share
    month = np.array([int(datetime(d.year, d.month, 1, tzinfo=UTC).timestamp() * 1000)
                      for d in (datetime.fromtimestamp(ms / 1000, tz=UTC) for ms in open_h)], dtype=np.int64)  # fmt: skip
    arrays = PanelArrays(
        symbols=tuple(f"S{i:02d}USDT" for i in range(members)),
        symbol=symbol,
        open_ms=open_ms,
        decision_ms=decision,
        month_ms=np.repeat(month, members),
        lka_ms=lka,
        x=np.zeros((n, 15)),
        label=label,
        fwd=np.zeros(n),
        complete=complete,
        features_ok=np.ones(n, dtype=bool),
        labelled=np.ones(n, dtype=bool),
    )
    per = hours + WINDOW
    bars = rng.standard_normal((members * per, N_QUANTITIES))
    bar_open = np.tile(t0 + HOUR_MS * np.arange(-WINDOW, hours, dtype=np.int64), members)
    end_idx = symbol * per + WINDOW + np.repeat(np.arange(hours, dtype=np.int64), members)
    windows = Windows(arrays.symbols, bars, bar_open, end_idx, np.ones(n, dtype=bool))
    return EncoderData(arrays, windows)
