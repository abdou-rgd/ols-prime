"""The universe panel: the non-learned strategies on every top-30 member, span by span (descriptive).

Usage::

    uv run python -m olsprime.bench.panel [--out DIR]

Method, fixed in :class:`PanelConfig`:

- **Members.** The monthly top-30 USDT universe (``load_universe``, sealed), rebalances in the
  test months 2019-01 -> 2024-12. A member's **span** is a run of consecutive months in the
  universe; it is held from the first month's start to the last month's end.
- **Bars.** Each member's 1d bars through ``load_klines``, cut at the end of the development
  period. A span whose bars are not a complete daily grid (a gap, such as LUNAUSDT's ticker
  reuse) or that has fewer than ``min_span_bars`` bars is skipped and counted.
- **Strategies.** ``buy-and-hold``, ``ma-momentum-{10,30,90}`` and ``random-turnover``
  (``seeds`` seeds; the median-Sharpe seed per span). random-turnover is fitted on the member's
  rows whose labels are known before the span starts (purge and embargo of h = 1 bar); a span
  without ``min_train_rows`` such rows has no random-turnover line.
- **Exit.** The target of a span's last bar is 0: the position is sold at the next open, the
  first bar after the span, which the simulation includes when it exists.
- **Execution.** As in Report n°1 (taker, 2 bps, five venues), with a minimum notional of 5 USDT
  and no lot step: the members' real lot steps are not in the snapshot.

It is descriptive: no hypothesis names it, and the spans overlap in time, so the cross-section
of Sharpe ratios is not a sample of independent trials.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from olsprime.bench import pipeline as pp
from olsprime.data.binance_vision import Month
from olsprime.data.klines import interval_ms
from olsprime.eval import stats
from olsprime.models.baselines import MA_WINDOWS, MaMomentum, RandomTurnover, buy_and_hold
from olsprime.models.decision import round_trip_cost
from olsprime.paths import data_dir, runs_dir
from olsprime.protocol import DEVELOPMENT

log = logging.getLogger(__name__)

STRATEGIES = ("buy-and-hold", *(f"ma-momentum-{w}" for w in MA_WINDOWS), "random-turnover")


@dataclass(frozen=True)
class PanelConfig:
    universe: str = "parquet/universe/spot_usdt_top30.parquet"
    interval: str = "1d"
    h: int = 1
    start: tuple[int, int] = (2019, 1)
    end: tuple[int, int] = (2024, 12)
    venues: tuple[str, ...] = pp.VENUES
    seeds: int = 10
    min_span_bars: int = 60
    min_train_rows: int = 120
    min_notional: float = 5.0
    initial_cash: float = pp.INITIAL_CASH
    periods_per_year: float = 365.0


@dataclass(frozen=True)
class Span:
    symbol: str
    start: datetime  # first month's start
    end: datetime  # exclusive: the month after the last one


def spans(universe: pl.DataFrame, start: Month, end: Month) -> list[Span]:
    """Contiguous membership runs per symbol, rebalances in ``[start, end]``, sorted by symbol then time."""
    lo, hi = start.start(), end.shift(1).start()
    rows = universe.filter((pl.col("rebalance_time") >= lo) & (pl.col("rebalance_time") < hi))
    out = []
    for (symbol,), part in sorted(rows.group_by("symbol"), key=lambda kv: kv[0][0]):
        months = sorted({Month(t.year, t.month) for t in part["rebalance_time"].to_list()}, key=lambda m: (m.year, m.month))
        run_start = prev = months[0]
        for m in months[1:]:
            if m != prev.shift(1):
                out.append(Span(symbol, run_start.start(), prev.shift(1).start()))
                run_start = m
            prev = m
        out.append(Span(symbol, run_start.start(), prev.shift(1).start()))
    return out


def _filters(cfg: PanelConfig) -> pp.SymbolFilters:
    return pp.SymbolFilters(min_notional=cfg.min_notional, step_size=0.0)


def _simulate(bars: pl.DataFrame, target: np.ndarray, venue: str, cfg: PanelConfig) -> np.ndarray:
    import olsprime_rs

    res = olsprime_rs.simulate(
        np.ascontiguousarray(bars["open_time"].dt.epoch("ms").to_numpy().astype(np.int64)),
        *(np.ascontiguousarray(bars[c].cast(pl.Float64).to_numpy()) for c in ("open", "high", "low", "close", "volume")),
        np.ascontiguousarray(target, dtype=np.float64),
        initial_cash=cfg.initial_cash,
        fees=venue,
        liquidity=pp.LIQUIDITY,
        slippage_bps=pp.SLIPPAGE_BPS,
        min_notional=_filters(cfg).min_notional,
        step_size=_filters(cfg).step_size,
    )
    return pp.period_returns(np.asarray(res.equity))


def _sharpe(r: np.ndarray, ppy: float) -> float | None:
    if r.size < 3 or np.all(r == r[0]):
        return None
    return stats.annualized(stats.sharpe(r), ppy)


def run_span(span: Span, bars: pl.DataFrame, frame: pl.DataFrame, cfg: PanelConfig) -> dict[str, Any] | str:
    """Net Sharpe of every strategy on one span at every venue, or the reason it is skipped."""
    step = interval_ms(cfg.interval)
    in_span = (frame["decision_time"] >= span.start) & (frame["decision_time"] < span.end)
    idx = np.flatnonzero(in_span.to_numpy())
    if idx.size < cfg.min_span_bars:
        return "short"
    sim_idx = np.arange(idx[0], min(idx[-1] + 2, bars.height))  # the span, plus the bar the exit executes at
    sim_bars = bars[sim_idx]
    if np.any(np.diff(sim_bars["open_time"].dt.epoch("ms").to_numpy()) != step) or np.any(np.diff(idx) != 1):
        return "gap"
    test = pp.test_rows(frame, idx)
    cut_ms = int(span.start.timestamp() * 1000) - cfg.h * step  # purge and embargo h before the span
    known = frame["label_known_at"].dt.epoch("ms").to_numpy()
    trainable = frame["label"].is_not_null().to_numpy() & frame["label_known_at"].is_not_null().to_numpy()
    train = frame[np.flatnonzero(trainable & (np.nan_to_num(known.astype(np.float64), nan=np.inf) <= cut_ms))]

    targets: dict[str, list[np.ndarray]] = {}
    for name, make in (("buy-and-hold", buy_and_hold), *((f"ma-momentum-{w}", lambda w=w: MaMomentum(w)) for w in MA_WINDOWS)):
        targets[name] = [make().predict(test)["target"].to_numpy()]
    if train.height >= cfg.min_train_rows:
        runs = []
        for seed in range(cfg.seeds):
            model = RandomTurnover(seed, cfg.interval)
            model.fit(train)
            runs.append(model.predict(test)["target"].to_numpy())
        targets["random-turnover"] = runs

    out: dict[str, Any] = {"symbol": span.symbol, "start": span.start.isoformat(), "end": span.end.isoformat(), "bars": int(idx.size), "sharpe": {}}
    for venue in cfg.venues:
        round_trip_cost(venue)  # fail early on an unknown venue
        out["sharpe"][venue] = {}
        for name, runs in targets.items():
            values = []
            for t in runs:
                full = np.zeros(sim_bars.height)
                full[: idx.size] = t
                full[idx.size - 1] = 0.0  # exit at the end of the span
                values.append(_sharpe(_simulate(sim_bars, full, venue, cfg), cfg.periods_per_year))
            defined = sorted(v for v in values if v is not None)
            out["sharpe"][venue][name] = defined[(len(defined) - 1) // 2] if defined else None
    return out


def cross_section(records: Sequence[dict[str, Any]], cfg: PanelConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for venue in cfg.venues:
        out[venue] = {}
        for name in STRATEGIES:
            pairs = [(r["sharpe"][venue].get(name), r["sharpe"][venue].get("buy-and-hold")) for r in records]
            s = np.array([a for a, _ in pairs if a is not None])
            beat = [a > b for a, b in pairs if a is not None and b is not None]
            out[venue][name] = None if s.size == 0 else {
                "spans": int(s.size),
                "median": float(np.median(s)),
                "p25": float(np.percentile(s, 25)),
                "p75": float(np.percentile(s, 75)),
                "share_above_zero": float(np.mean(s > 0)),
                "share_above_buy_and_hold": None if name == "buy-and-hold" or not beat else float(np.mean(beat)),
            }  # fmt: skip
    return out


def run_panel(cfg: PanelConfig) -> dict[str, Any]:
    from olsprime.protocol import load_klines, load_universe

    universe = load_universe(data_dir() / cfg.universe, caller="bench.panel")
    all_spans = spans(universe, Month(*cfg.start), Month(*cfg.end))
    records, skipped = [], {"short": 0, "gap": 0, "no-bars": 0}
    for symbol in sorted({s.symbol for s in all_spans}):
        try:
            bars = load_klines(symbol, cfg.interval, caller="bench.panel")
        except FileNotFoundError:
            skipped["no-bars"] += sum(s.symbol == symbol for s in all_spans)
            continue
        bars = bars.filter(pl.col("known_at") < DEVELOPMENT.end)
        if bars.height == 0:
            skipped["no-bars"] += sum(s.symbol == symbol for s in all_spans)
            continue
        frame = pp.build_frame(bars, cfg.interval, cfg.h)
        for span in (s for s in all_spans if s.symbol == symbol):
            res = run_span(span, bars, frame, cfg)
            if isinstance(res, str):
                skipped[res] += 1
            else:
                records.append(res)
    return {
        "config": asdict(cfg),
        "spans_total": len(all_spans),
        "spans_scored": len(records),
        "skipped": skipped,
        "symbols_scored": len({r["symbol"] for r in records}),
        "cross_section": cross_section(records, cfg),
        "spans": records,
    }


def render_md(result: dict[str, Any]) -> str:
    cfg = result["config"]
    lines = [
        "# Universe panel: non-learned strategies on the top-30 members (descriptive)",
        "",
        f"{result['spans_scored']} membership spans of {result['symbols_scored']} pairs scored, of {result['spans_total']} "
        f"(skipped: {', '.join(f'{k} {v}' for k, v in result['skipped'].items())}). Test months {cfg['start'][0]}-{cfg['start'][1]:02d} -> "
        f"{cfg['end'][0]}-{cfg['end'][1]:02d}, 1d bars, exit at each span's end, taker fees, 2 bps slippage, min notional "
        f"{cfg['min_notional']:g} USDT, no lot step. random-turnover: median of {cfg['seeds']} seeds per span.",
        "",
        "Spans overlap in time and share the market factor: the cross-section is not a sample of independent trials, and no "
        "interval is given. No hypothesis names this panel.",
        "",
        "| venue | strategy | spans | median net Sharpe | 25 % | 75 % | share > 0 | share > B&H on the same span |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for venue, table in result["cross_section"].items():
        for name, d in table.items():
            if d is None:
                lines.append(f"| {venue} | {name} | 0 | n/a | n/a | n/a | n/a | n/a |")
                continue
            beat = "n/a" if d["share_above_buy_and_hold"] is None else f"{100 * d['share_above_buy_and_hold']:.0f} %"
            lines.append(
                f"| {venue} | {name} | {d['spans']} | {d['median']:.2f} | {d['p25']:.2f} | {d['p75']:.2f} | "
                f"{100 * d['share_above_zero']:.0f} % | {beat} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.panel", description="The universe panel (descriptive).")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <runs>/<UTC date>-panel)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    result = run_panel(PanelConfig())
    out = args.out or runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-panel"
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "panel.md").write_text(render_md(result), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({"seconds": round(time.perf_counter() - started, 1)}) + "\n", encoding="utf-8")
    print(f"panel: {result['spans_scored']} spans of {result['symbols_scored']} pairs; written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
