"""Quality report for a series of klines.

The report counts:

- duplicates: rows whose open time was already seen, and how many of those open
  times carry rows that disagree;
- non-monotonic times: rows that open earlier than the row before them;
- gaps: runs of missing bars between consecutive distinct open times;
- inconsistent OHLC: missing, non-finite or non-positive prices, a high below the
  open, close or low, or a low above the open or close;
- off-grid open times, close times other than open time + interval - 1 ms, and
  missing, NaN or negative volumes and trade counts.

Rows are checked in the order given: run it on the raw concatenation of archives
to see duplicates and ordering problems before they are cleaned away (``build_parquet``
does, and covers every row). The command line reads a stored file through
``olsprime.protocol.load_klines_file``, so its report stops before the vault. Usage::

    uv run python -m olsprime.data.quality ../data/parquet/spot/klines/1h/BTCUSDT.parquet
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from olsprime.data.klines import DAY_MS, interval_ms

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_VOLUME_COLUMNS = ("volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume")


@dataclass(frozen=True)
class Gap:
    """Bars missing between two consecutive open times."""

    start: datetime  # open time of the first missing bar
    end: datetime  # open time of the bar that resumes the series
    missing_bars: int


@dataclass(frozen=True)
class QualityReport:
    symbol: str | None
    interval: str
    rows: int
    first_open_time: datetime | None
    last_open_time: datetime | None
    duplicate_rows: int
    conflicting_duplicates: int
    non_monotonic: int
    gaps: tuple[Gap, ...]
    ohlc_inconsistent: int
    off_grid: int
    bad_close_time: int
    invalid_volume: int

    @property
    def missing_bars(self) -> int:
        return sum(gap.missing_bars for gap in self.gaps)

    @property
    def is_clean(self) -> bool:
        """No problem of any kind, gaps included."""
        return not (
            self.duplicate_rows
            or self.non_monotonic
            or self.gaps
            or self.ohlc_inconsistent
            or self.off_grid
            or self.bad_close_time
            or self.invalid_volume
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-ready view: datetimes as ISO 8601 strings."""
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "rows": self.rows,
            "first_open_time": _iso(self.first_open_time),
            "last_open_time": _iso(self.last_open_time),
            "duplicate_rows": self.duplicate_rows,
            "conflicting_duplicates": self.conflicting_duplicates,
            "non_monotonic": self.non_monotonic,
            "missing_bars": self.missing_bars,
            "gaps": [
                {"start": _iso(gap.start), "end": _iso(gap.end), "missing_bars": gap.missing_bars}
                for gap in self.gaps
            ],
            "ohlc_inconsistent": self.ohlc_inconsistent,
            "off_grid": self.off_grid,
            "bad_close_time": self.bad_close_time,
            "invalid_volume": self.invalid_volume,
        }

    def summary(self) -> str:
        """A few lines of plain ASCII text."""
        name = f"{self.symbol} {self.interval}" if self.symbol else self.interval
        if not self.rows:
            return f"{name}: no rows"
        lines = [
            f"{name}: {self.rows:,} rows, "
            f"{_short(self.first_open_time)} -> {_short(self.last_open_time)} UTC (open times)"
        ]
        gaps = f"  gaps: {len(self.gaps)} ({self.missing_bars:,} missing bars)"
        if self.gaps:
            worst = max(self.gaps, key=lambda gap: gap.missing_bars)
            gaps += f", largest {_short(worst.start)} -> {_short(worst.end)} ({worst.missing_bars} bars)"
        lines.append(gaps)
        lines.append(
            f"  duplicates: {self.duplicate_rows} ({self.conflicting_duplicates} conflicting), "
            f"non-monotonic: {self.non_monotonic}, OHLC inconsistent: {self.ohlc_inconsistent}, "
            f"off-grid: {self.off_grid}, bad close_time: {self.bad_close_time}, "
            f"invalid volume: {self.invalid_volume}"
        )
        return "\n".join(lines)


def check_quality(bars: pl.DataFrame, interval: str, *, symbol: str | None = None) -> QualityReport:
    """Check ``bars`` (``open_time`` and OHLC columns at least) against ``interval``.

    The close-time and volume checks run when their columns are present.
    """
    step = interval_ms(interval)
    if symbol is None and "symbol" in bars.columns and bars.height:
        symbol = bars["symbol"][0]
    opens = _epoch_ms(bars["open_time"])
    if opens.null_count():
        raise ValueError("open_time has null values")

    distinct = opens.unique().sort()
    duplicate_rows = bars.height - distinct.len()
    conflicting = 0
    if duplicate_rows:
        repeated = bars.filter(opens.is_duplicated())
        conflicting = repeated.unique().group_by("open_time").len().filter(pl.col("len") > 1).height
    non_monotonic = int((opens.diff() < 0).sum())

    deltas = distinct.diff()
    gaps = tuple(
        Gap(
            start=_datetime(distinct[i - 1] + step),
            end=_datetime(distinct[i]),
            missing_bars=deltas[i] // step - 1,
        )
        for i in (deltas >= 2 * step).arg_true()
    )

    o, h, l, c = (pl.col(name) for name in ("open", "high", "low", "close"))
    bad_price = pl.any_horizontal([~(p.is_finite() & (p > 0)).fill_null(True) for p in (o, h, l, c)])
    ohlc = bad_price | (h < pl.max_horizontal(o, l, c)) | (l > pl.min_horizontal(o, c))
    ohlc_inconsistent = bars.select(ohlc.fill_null(True).sum()).item()

    # Bars shorter than a day sit on a grid that starts at the epoch; 3d and 1w bars do not.
    off_grid = int((opens % step != 0).sum()) if DAY_MS % step == 0 else 0

    bad_close_time = 0
    if "close_time" in bars.columns:
        closes = _epoch_ms(bars["close_time"])
        bad_close_time = int((closes - opens != step - 1).fill_null(True).sum())

    # Spelled out because polars orders NaN above every number: NaN < 0 is false.
    checks = [
        pl.col(name).is_null() | pl.col(name).is_nan() | (pl.col(name) < 0)
        for name in _VOLUME_COLUMNS
        if name in bars.columns
    ]
    if "trades" in bars.columns:
        checks.append(pl.col("trades").is_null() | (pl.col("trades") < 0))
    invalid_volume = bars.select(pl.any_horizontal(checks).sum()).item() if checks else 0

    return QualityReport(
        symbol=symbol,
        interval=interval,
        rows=bars.height,
        first_open_time=_datetime(distinct[0]) if distinct.len() else None,
        last_open_time=_datetime(distinct[-1]) if distinct.len() else None,
        duplicate_rows=duplicate_rows,
        conflicting_duplicates=conflicting,
        non_monotonic=non_monotonic,
        gaps=gaps,
        ohlc_inconsistent=ohlc_inconsistent,
        off_grid=off_grid,
        bad_close_time=bad_close_time,
        invalid_volume=invalid_volume,
    )


def _epoch_ms(times: pl.Series) -> pl.Series:
    return times.dt.epoch("ms") if isinstance(times.dtype, pl.Datetime) else times.cast(pl.Int64)


def _datetime(epoch_ms: int) -> datetime:
    return _EPOCH + timedelta(milliseconds=epoch_ms)


def _iso(when: datetime | None) -> str | None:
    return when.isoformat() if when else None


def _short(when: datetime | None) -> str:
    return when.strftime("%Y-%m-%d %H:%M") if when else "-"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olsprime.data.quality",
        description="Quality report for kline Parquet files.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Parquet files of klines")
    parser.add_argument(
        "--interval",
        help="bar interval (default: the parent directory name, as in .../klines/1h/BTCUSDT.parquet)",
    )
    parser.add_argument("--json", action="store_true", help="print the full reports as JSON")
    args = parser.parse_args(argv)
    from olsprime.protocol import load_klines_file  # here: protocol imports binance_vision, which imports this module

    for path in args.paths:
        interval = args.interval or path.parent.name
        report = check_quality(load_klines_file(path, interval, caller=f"quality.main({path.name})"), interval)
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
