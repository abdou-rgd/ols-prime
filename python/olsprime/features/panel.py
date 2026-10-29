"""The point-in-time hourly panel of the universe members (iteration 3, ``docs/PROTOCOL.md``).

One row per (member, hour) in the months when the pair is a member of the universe, with the
15 features of :mod:`olsprime.features.basic`, the one-day forward label of
:mod:`olsprime.labels.forward` and the instant that label becomes known. Matrix reads it.

Rules, each pre-registered:

- **Membership.** A pair is a member of month M if it is in the rebalance at M's first day,
  ranked on data before M. Rows exist only in member months. A pair's months before it first
  joins never appear: they would carry its later selection (survivorship). A delisted pair
  keeps its rows until its last bar; its later member months have no bars, so no rows.
- **Features** are computed on each pair's whole hourly history up to t, so their windows
  count hours, and the first member hour already has its warm-up behind it.
- **Labels** are ``forward_labels(bars, "1h", 24)``: entry at the open of t + 1, exit at the
  open of t + 25, known at that exit open (``label_known_at``).
- **The cut.** Bars are cut to ``known_at < DEVELOPMENT.end`` *before* features and labels. A
  feature at t needs no bar after t, so the cut changes none; a label that needs a bar of 2025
  is null. The validation year never reaches a label.
- **Months** at or after ``DEVELOPMENT.end`` are refused, as the splitter does.

Bars come through :func:`olsprime.protocol.load_klines` and the universe through
:func:`olsprime.protocol.load_universe`; this module opens no file. The panel is built in
memory, one pair at a time: :func:`pair_rows` builds one pair's rows from its bars, and
:func:`assemble` joins the pairs. The panel H1 calls both with perturbed bars. Usage::

    uv run --no-sync python -m olsprime.features.panel --summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from olsprime import paths
from olsprime.data.binance_vision import Month, month_arg, month_range
from olsprime.data.klines import TIMESTAMP
from olsprime.features.basic import FEATURE_COLUMNS, basic_features
from olsprime.labels.forward import LABEL_COLUMNS, forward_labels
from olsprime.protocol import DEVELOPMENT, VAULT, VaultSealed, load_klines, load_universe

INTERVAL = "1h"
HORIZON = 24  # bars: one day
FIRST_MONTH = Month(2017, 10)  # the first rebalance of the universe
LAST_MONTH = Month(2024, 12)  # the last month before the validation year
TOP_N = 30  # the universe file the panel reads
SUBPANEL_SIZE = 10

PANEL_COLUMNS: tuple[str, ...] = (
    "symbol",
    "open_time",
    "decision_time",
    "month",
    *FEATURE_COLUMNS,
    "threshold",
    *LABEL_COLUMNS,
    "complete",
)

# What the features and the labels need of a bar. ``known_at`` is added by ``load_klines``.
_BAR_COLUMNS = [
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "taker_buy_quote_volume",
    "known_at",
]


# --- Membership ---------------------------------------------------------------------------


def _check_months(first: Month, last: Month) -> None:
    """The splitter's guards: nothing in the vault, nothing outside the development period."""
    if last < first:
        raise ValueError(f"last month {last} is before first month {first}")
    for month in (first, last):
        if month.start() >= VAULT.start:
            raise VaultSealed(f"month {month} reaches the vault, which opens at {VAULT.start:%Y-%m-%d}")
    if last.start() >= DEVELOPMENT.end:
        raise ValueError(
            f"month {last} is at or after the end of the development period ({DEVELOPMENT.end:%Y-%m-%d}); "
            "the panel stays out of the validation year"
        )
    if first.start() < DEVELOPMENT.start:
        raise ValueError(f"month {first} is before the development period ({DEVELOPMENT.start:%Y-%m-%d})")


def member_months(universe: pl.DataFrame, *, first: Month = FIRST_MONTH, last: Month = LAST_MONTH) -> pl.DataFrame:
    """One row per (symbol, month) of membership, sorted by ``(month, symbol)``.

    ``universe`` is the sealed universe table (``rebalance_time``, ``symbol``, ...). A member of
    month M is a member of the rebalance at M's first day. ``month`` is that instant (Datetime
    ms, UTC). Rebalances outside ``first .. last`` are left out; a ``last`` at or after
    ``DEVELOPMENT.end`` is refused.
    """
    _check_months(first, last)
    for name in ("rebalance_time", "symbol"):
        if name not in universe.columns:
            raise ValueError(f"universe lacks column {name!r}")
    if universe.schema["rebalance_time"] != TIMESTAMP:
        raise TypeError(f"rebalance_time is {universe.schema['rebalance_time']}, expected {TIMESTAMP}")
    table = universe.select("symbol", pl.col("rebalance_time").alias("month"))
    if table.filter(pl.col("month") != pl.col("month").dt.truncate("1mo")).height:
        raise ValueError("a rebalance is not at the first instant of a month")
    if table.is_duplicated().any():
        raise ValueError("the universe lists a symbol twice in one rebalance")
    return (
        table.filter(pl.col("month").is_between(first.start(), last.start()))  # both ends included
        .sort(["month", "symbol"])
        .select("symbol", "month")
    )


def h1_subpanel_members(universe: pl.DataFrame, *, size: int = SUBPANEL_SIZE) -> list[str]:
    """The ``size`` members with the most member months in the development period, ties broken by symbol."""
    counts = member_months(universe).group_by("symbol").agg(months=pl.len()).sort(["months", "symbol"], descending=[True, False])
    return counts.head(size)["symbol"].to_list()


# --- The panel ----------------------------------------------------------------------------


def _month_expr() -> pl.Expr:
    return pl.col("decision_time").dt.truncate("1mo").cast(TIMESTAMP).alias("month")


def pair_rows(bars: pl.DataFrame, months: Sequence[datetime], *, h: int = HORIZON) -> pl.DataFrame:
    """Every row of one pair in its member ``months``: features and labels from its whole (cut) history.

    ``bars`` are one pair's hourly bars as :func:`olsprime.protocol.load_klines` returns them, with
    at least the columns the features and labels need. The cut to ``known_at < DEVELOPMENT.end``
    happens here, before features and labels, whatever the caller passes.
    """
    if bars.is_empty():
        return pl.DataFrame()
    if bars["symbol"].n_unique() != 1:
        raise ValueError("pair_rows takes one pair at a time")
    symbol = bars["symbol"][0]
    bars = bars.select(_BAR_COLUMNS).filter(pl.col("known_at") < DEVELOPMENT.end)  # before features and labels: no 2025 bar reaches a label
    if bars.is_empty():
        return pl.DataFrame()
    features = basic_features(bars, INTERVAL)
    labels = forward_labels(bars, INTERVAL, h)
    joined = features.join(
        labels.rename({"decision_time": "_label_decision_time"}),
        on=["symbol", "open_time"],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if joined.height != features.height or not joined["_label_decision_time"].equals(joined["decision_time"]):
        raise ValueError(f"{symbol}: features and labels disagree on the rows or on decision_time")
    return (
        joined.drop("_label_decision_time")
        .with_columns(_month_expr())
        .filter(pl.col("month").is_in(list(months)))
    )


def assemble(parts: Iterable[pl.DataFrame]) -> pl.DataFrame:
    """The panel from its pairs' rows: ``complete`` added, columns in order, sorted by ``(decision_time, symbol)``."""
    parts = [part for part in parts if part.height]
    if not parts:
        raise ValueError("no member has a bar in its member months")
    panel = pl.concat(parts).with_columns(
        complete=pl.all_horizontal([pl.col(c).is_not_null() for c in (*FEATURE_COLUMNS, "label")])
    )
    return panel.select(PANEL_COLUMNS).sort(["decision_time", "symbol"], maintain_order=True)


def hourly_panel(
    universe: pl.DataFrame | None = None,
    *,
    h: int = HORIZON,
    first: Month = FIRST_MONTH,
    last: Month = LAST_MONTH,
    symbols: Sequence[str] | None = None,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """The panel: one row per (member, hour) in member months, sorted by ``(decision_time, symbol)``.

    Columns, in order: ``symbol``, ``open_time``, ``decision_time`` (bar t's ``known_at``), ``month``
    (the month of ``decision_time``, which is a member month of the symbol), the 15
    :data:`FEATURE_COLUMNS`, ``threshold``, the label columns (``entry_time``, ``exit_time``,
    ``label_known_at``, ``fwd_return``, ``label``) and ``complete`` (every feature and the label
    non-null).

    ``universe`` defaults to the sealed universe file of the data root; ``data_dir`` to
    :func:`olsprime.paths.data_dir`. ``symbols`` keeps only those members (the sub-panel). A
    member with no stored 1 h file raises: nothing is skipped silently.
    """
    root = data_dir if data_dir is not None else paths.data_dir()
    if universe is None:
        from olsprime.universe import universe_path  # the file's name lives with the universe

        universe = load_universe(universe_path(root / "parquet", TOP_N), caller="features.panel.hourly_panel")
    members = member_months(universe, first=first, last=last)
    if symbols is not None:
        unknown = sorted(set(symbols) - set(members["symbol"]))
        if unknown:
            raise ValueError(f"not members in {first}..{last}: {unknown}")
        members = members.filter(pl.col("symbol").is_in(list(symbols)))
    parts = []
    for symbol, rows in members.group_by("symbol", maintain_order=True):
        bars = load_klines(symbol[0], INTERVAL, data_dir=root, columns=_BAR_COLUMNS, caller="features.panel.hourly_panel")
        parts.append(pair_rows(bars, rows["month"].to_list(), h=h))
    return assemble(parts)


# --- The summary --------------------------------------------------------------------------


def _counts(frame: pl.DataFrame) -> dict[str, object]:
    complete = frame.filter(pl.col("complete"))
    shares: dict[str, float | None] = {}
    for label in (0, 1, 2):
        shares[str(label)] = (complete["label"] == label).sum() / complete.height if complete.height else None
    return {
        "rows": frame.height,
        "labelled": int(frame["label"].is_not_null().sum()),
        "complete": complete.height,
        "class_share_complete": shares,
    }


def fingerprint(panel: pl.DataFrame) -> str:
    """SHA-256 over every column of the panel (values, nulls and names), for the audit trail."""
    digest = hashlib.sha256()
    for name in panel.columns:
        series = panel[name]
        digest.update(name.encode() + b"\0")
        if series.dtype == pl.String:
            digest.update("\n".join(series.fill_null("\0")).encode())
        elif isinstance(series.dtype, pl.Datetime):
            digest.update(series.dt.epoch("ms").fill_null(-(2**63)).to_numpy().astype("<i8").tobytes())
        elif series.dtype == pl.Boolean:
            digest.update(series.fill_null(False).to_numpy().astype("u1").tobytes())
        elif series.dtype.is_float():
            digest.update(series.fill_null(np.nan).to_numpy().astype("<f8").tobytes())
        else:
            digest.update(series.cast(pl.Int64).fill_null(-1).to_numpy().astype("<i8").tobytes())
    return digest.hexdigest()


def summarize(panel: pl.DataFrame, universe: pl.DataFrame, *, first: Month = FIRST_MONTH, last: Month = LAST_MONTH) -> dict:
    """Per month: members, members with rows, rows, labelled, complete, class shares of the complete rows."""
    members = member_months(universe, first=first, last=last)
    months = []
    for month in month_range(first, last):
        at = month.start()
        rows = panel.filter(pl.col("month") == at)
        entry = {
            "month": str(month),
            "members": int((members["month"] == at).sum()),
            "members_with_rows": rows["symbol"].n_unique(),
            **_counts(rows),
        }
        months.append(entry)
    return {
        "interval": INTERVAL,
        "h": HORIZON,
        "first_month": str(first),
        "last_month": str(last),
        "cut": f"known_at < {DEVELOPMENT.end:%Y-%m-%d}",
        "distinct_symbols": panel["symbol"].n_unique(),
        "member_months": members.height,
        "totals": _counts(panel),
        "fingerprint_sha256": fingerprint(panel),
        "subpanel_members": h1_subpanel_members(universe),
        "months": months,
    }


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1024, 1)  # kilobytes on Linux


def _fresh_dir(base: Path, stamp: str) -> Path:
    """``<base>/<stamp>-panel``, suffixed ``-2``, ``-3`` ... when taken."""
    candidate, n = base / f"{stamp}-panel", 1
    while candidate.exists():
        n += 1
        candidate = base / f"{stamp}-panel-{n}"
    candidate.mkdir(parents=True)
    return candidate


def _print_summary(summary: dict) -> None:
    def pct(value: float | None) -> str:
        return "      -" if value is None else f"{100 * value:6.2f}%"

    print(f"{'month':<8}{'members':>8}{'with rows':>10}{'rows':>10}{'labelled':>10}{'complete':>10}{'down':>9}{'neutral':>9}{'up':>9}")
    for entry in [*summary["months"], {"month": "total", "members": summary["member_months"], "members_with_rows": summary["distinct_symbols"], **summary["totals"]}]:
        s = entry["class_share_complete"]
        print(
            f"{entry['month']:<8}{entry['members']:>8}{entry['members_with_rows']:>10}{entry['rows']:>10}"
            f"{entry['labelled']:>10}{entry['complete']:>10}{pct(s['0']):>9}{pct(s['1']):>9}{pct(s['2']):>9}"
        )
    print("(total: members = member months, with rows = distinct symbols)")
    print("sub-panel members:", " ".join(summary["subpanel_members"]))
    print("fingerprint sha256:", summary["fingerprint_sha256"])
    run = summary["run"]
    print(f"wall time {run['wall_seconds']:.1f} s, peak RSS {run['peak_rss_mb']} MB")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olsprime.features.panel",
        description="Build the hourly panel of the universe members and summarize it.",
    )
    parser.add_argument("--summary", action="store_true", help="print the per-month summary and write summary.json")
    parser.add_argument("--first", type=month_arg, default=FIRST_MONTH, help="first month (default: %(default)s)")
    parser.add_argument("--last", type=month_arg, default=LAST_MONTH, help="last month (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=None, help="data root (default: olsprime.paths.data_dir())")
    parser.add_argument("--out-dir", type=Path, default=None, help="runs folder (default: olsprime.paths.runs_dir())")
    parser.add_argument("--no-write", action="store_true", help="do not write summary.json")
    args = parser.parse_args(argv)
    if not args.summary:
        parser.print_help()
        return 0

    started = time.perf_counter()
    root = args.data_dir if args.data_dir is not None else paths.data_dir()
    from olsprime.universe import universe_path

    universe = load_universe(universe_path(root / "parquet", TOP_N), caller="features.panel.main")
    panel = hourly_panel(universe, first=args.first, last=args.last, data_dir=root)
    summary = summarize(panel, universe, first=args.first, last=args.last)
    summary["run"] = {
        "wall_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_mb": _peak_rss_mb(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _print_summary(summary)
    if not args.no_write:
        base = args.out_dir if args.out_dir is not None else paths.runs_dir()
        target = _fresh_dir(base, f"{datetime.now(UTC):%Y-%m-%d}")
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("written:", target / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
