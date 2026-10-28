"""Survey of the stored data: the numbers that ``docs/DATA.md`` quotes, from the stored files.

Three questions, each answered from files and not from memory:

- **Quality**: the per-pair ``{SYMBOL}.quality.json`` reports, summed per interval, with the
  ten worst pairs. Those reports are written by ``build_parquet`` on the raw concatenation of
  the archives, vault months included: they are structural checks, not reads for modelling.
- **Ticker reuse**: a daily series with a gap longer than ``min_gap_days`` after which the price
  level jumps by more than ``max_jump`` (``LUNAUSDT`` in 2022). Flagged, never repaired. The
  prices come through :func:`olsprime.protocol.load_klines`, so the check stops before the vault.
- **Universe**: members per month and distinct members, through :func:`olsprime.protocol.load_universe`.

Usage::

    uv run python -m olsprime.data.survey [--data-dir DIR] [--top 30]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from olsprime.data.binance_vision import parquet_path
from olsprime.data.klines import DAY_MS, interval_ms
from olsprime.paths import data_dir as default_data_dir
from olsprime.protocol import load_klines, load_universe

PROBLEMS = (
    "duplicate_rows",
    "conflicting_duplicates",
    "non_monotonic",
    "missing_bars",
    "ohlc_inconsistent",
    "off_grid",
    "bad_close_time",
    "invalid_volume",
)


def quality_reports(parquet_dir: Path, interval: str, symbols: Collection[str] | None = None) -> list[dict]:
    """The saved quality reports of an interval, by symbol; only ``symbols`` when given."""
    folder = parquet_dir / "spot" / "klines" / interval
    reports = (json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.quality.json")))
    return [r for r in reports if symbols is None or r["symbol"] in symbols]


def total_problems(report: dict) -> int:
    """Rows or bars affected, any kind: the key for the worst pairs. Conflicting duplicates are already duplicates."""
    return sum(report[k] for k in PROBLEMS if k != "conflicting_duplicates")


def aggregate(reports: Iterable[dict]) -> dict[str, int]:
    reports = list(reports)
    out = {"pairs": len(reports), "rows": sum(r["rows"] for r in reports)}
    out["gaps"] = sum(len(r["gaps"]) for r in reports)
    out |= {k: sum(r[k] for r in reports) for k in PROBLEMS}
    out["pairs_with_gaps"] = sum(bool(r["gaps"]) for r in reports)
    out["clean_pairs"] = sum(total_problems(r) == 0 for r in reports)
    return out


def worst(reports: Iterable[dict], count: int = 10) -> list[dict]:
    """The ``count`` pairs with the most problems, ties by symbol."""
    return sorted(reports, key=lambda r: (-total_problems(r), r["symbol"]))[:count]


# --- Ticker reuse ---------------------------------------------------------------------


@dataclass(frozen=True)
class Reuse:
    symbol: str
    gap_start: datetime  # open time of the first missing bar
    gap_days: float
    close_before: float
    open_after: float

    @property
    def ratio(self) -> float:
        return self.open_after / self.close_before


def ticker_reuse(
    data_dir: Path,
    symbols: Iterable[str],
    *,
    interval: str = "1d",
    min_gap_days: float = 7,
    max_jump: float = 0.5,
) -> list[Reuse]:
    """Pairs whose daily series has a gap of more than ``min_gap_days`` and a price jump over ``max_jump`` across it.

    The jump is ``open after / close before - 1`` across the gap, in either direction.
    A gap that ends in the vault is out of sight (the read is sealed) and is not judged.
    """
    flagged = []
    for symbol in symbols:
        report_path = parquet_path(data_dir / "parquet", symbol, interval).with_name(f"{symbol}.quality.json")
        if not report_path.is_file():
            continue
        gaps = [
            g
            for g in json.loads(report_path.read_text(encoding="utf-8"))["gaps"]
            if g["missing_bars"] * interval_ms(interval) > min_gap_days * DAY_MS
        ]
        if not gaps:
            continue
        frame = load_klines(symbol, interval, data_dir=data_dir, columns=["open_time", "open", "close"]).sort("open_time")
        for gap in gaps:
            start = datetime.fromisoformat(gap["start"])
            resume = datetime.fromisoformat(gap["end"])
            before = frame.filter(pl.col("open_time") < start).tail(1)
            after = frame.filter(pl.col("open_time") >= resume).head(1)
            if before.is_empty() or after.is_empty():
                continue
            close_before, open_after = before["close"].item(), after["open"].item()
            if close_before > 0 and abs(open_after / close_before - 1) > max_jump:
                flagged.append(Reuse(symbol, start, (resume - start) / timedelta(days=1), close_before, open_after))
    return flagged


# --- Universe -------------------------------------------------------------------------


def universe_summary(universe: pl.DataFrame) -> dict[str, float | int]:
    """Members per rebalance (min, median, max), rebalances and distinct members."""
    per_month = universe.group_by("rebalance_time").len()["len"].to_list()
    return {
        "rebalances": len(per_month),
        "min": min(per_month),
        "median": statistics.median(per_month),
        "max": max(per_month),
        "distinct": universe["symbol"].n_unique(),
        "first": universe["rebalance_time"].min(),
        "last": universe["rebalance_time"].max(),
    }


# --- Report ---------------------------------------------------------------------------


def _markdown_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(f"{c:,}" if isinstance(c, int) else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _quality_section(title: str, reports: list[dict]) -> list[str]:
    total = aggregate(reports)
    out = [f"### {title}\n"]
    out.append(
        _markdown_table(
            ["pairs", "rows", "gaps (pairs with)", "missing bars", "duplicates", "non-monotonic", "OHLC", "off-grid", "bad close", "volume", "clean pairs"],
            [[total["pairs"], total["rows"], f"{total['gaps']:,} ({total['pairs_with_gaps']})", total["missing_bars"],
              total["duplicate_rows"], total["non_monotonic"], total["ohlc_inconsistent"], total["off_grid"],
              total["bad_close_time"], total["invalid_volume"], total["clean_pairs"]]],
        )  # fmt: skip
    )
    out.append("\nTen worst pairs (most gaps, duplicates and other problems):\n")
    out.append(
        _markdown_table(
            ["pair", "rows", "gaps", "missing bars", "duplicates", "non-monotonic", "OHLC", "off-grid", "bad close", "volume"],
            [[r["symbol"], r["rows"], len(r["gaps"]), r["missing_bars"], r["duplicate_rows"], r["non_monotonic"],
              r["ohlc_inconsistent"], r["off_grid"], r["bad_close_time"], r["invalid_volume"]] for r in worst(reports)],
        )  # fmt: skip
    )
    return [*out, ""]


def render(data_dir: Path, top_n: int = 30, intervals: Sequence[str] = ("1h", "1d")) -> str:
    parquet_dir = data_dir / "parquet"
    universe_file = parquet_dir / "universe" / f"spot_usdt_top{top_n}.parquet"
    universe = load_universe(universe_file, caller="survey") if universe_file.is_file() else None
    members = None if universe is None else set(universe["symbol"].unique())
    out = []
    for interval in intervals:
        reports = quality_reports(parquet_dir, interval, members)
        if reports:
            out += _quality_section(f"Quality, {interval}" + (" (universe members)" if members else ""), reports)
        everyone = quality_reports(parquet_dir, interval)
        if members and len(everyone) > len(reports):
            out += _quality_section(f"Quality, {interval}, every stored pair", everyone)
    if universe is not None:
        summary = universe_summary(universe)
        out.append(
            f"### Universe\n\n{summary['rebalances']} rebalances, {summary['first']:%Y-%m} to {summary['last']:%Y-%m}; "
            f"members per month: min {summary['min']}, median {summary['median']}, max {summary['max']}; "
            f"{summary['distinct']} distinct members.\n"
        )
        reuse = ticker_reuse(data_dir, sorted(members))
        out.append("### Likely ticker reuse (members, 1d: gap over 7 days, price level changed by over 50 %)\n")
        out.append(
            _markdown_table(
                ["pair", "gap starts", "gap days", "close before", "open after", "ratio"],
                [[r.symbol, f"{r.gap_start:%Y-%m-%d}", f"{r.gap_days:.0f}", f"{r.close_before:.6g}", f"{r.open_after:.6g}", f"{r.ratio:.3g}"] for r in reuse],
            )  # fmt: skip
            if reuse
            else "none"
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.data.survey", description="Summarise the stored data.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(), help="data root (default: %(default)s)")
    parser.add_argument("--top", type=int, default=30, help="universe size of the file to read (default: 30)")
    args = parser.parse_args(argv)
    print(render(args.data_dir, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
