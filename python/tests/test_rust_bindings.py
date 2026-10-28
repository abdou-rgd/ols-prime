"""The Python bindings of the Rust core (``olsprime_rs``, built from ``crates/olsprime-py``).

The bindings add no arithmetic, so most tests here are exact: bit-for-bit comparisons against
hand-computed values, against a line-by-line Python port of the simulator, or against the
binding itself on perturbed inputs. The OLS identities are compared with numpy's closed forms
to a relative 1e-10 or so. Everything is offline and runs in well under a second.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import olsprime_rs as rs
import pytest

# --- helpers ------------------------------------------------------------------------------


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal dtype, shape and bit pattern: -0.0 differs from 0.0, NaN equals itself."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return a.tobytes() == b.tobytes()


def fills_equal(x: rs.Fills, y: rs.Fills, upto: int | None = None) -> bool:
    """The fills of x and y, restricted to bars <= upto, are identical bit for bit."""
    ax, ay = x.as_dict(), y.as_dict()
    mx = np.ones(len(x), bool) if upto is None else ax["bar"] <= upto
    my = np.ones(len(y), bool) if upto is None else ay["bar"] <= upto
    return all(same_bits(ax[k][mx], ay[k][my]) for k in ax)


def random_bars(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """A valid OHLCV series: a geometric random walk with wicks, on a 1-minute grid."""
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.003, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.003, n))
    volume = rng.uniform(0.0, 10.0, n)
    open_time = 1_700_000_000_000 + 60_000 * np.arange(n, dtype=np.int64)
    return {"open_time": open_time, "open": open_, "high": high, "low": low, "close": close, "volume": volume}


def random_targets(rng: np.random.Generator, n: int) -> np.ndarray:
    """A mix of flat, full, held and fractional targets, so that every branch trades."""
    kind = rng.integers(0, 4, n)
    return np.where(kind == 0, 0.0, np.where(kind == 1, 1.0, rng.uniform(0.0, 1.0, n)))


def run(bars: dict[str, np.ndarray], targets: np.ndarray, **cfg) -> rs.SimResult:
    cfg.setdefault("initial_cash", 10_000.0)
    cfg.setdefault("fees", "binance-spot")
    return rs.simulate(
        bars["open_time"], bars["open"], bars["high"], bars["low"], bars["close"],
        bars["volume"], targets, **cfg,
    )


# --- a line-by-line port of crates/backtest/src/sim.rs -----------------------------------
#
# Same IEEE operations in the same order, so it must agree with the binding bit for bit. If it
# does on random configurations, the binding reads every input right and returns every output
# untouched. It also documents the execution model in twenty lines.

DUST_FRACTION = 1e-12
MAX_LOTS = 1 << 50
EPS = np.finfo(np.float64).eps


def _floor_lots(q: float, step: float) -> int:
    if math.isnan(q) or q <= 0.0:
        return 0
    n = min(math.floor(q / step), MAX_LOTS)
    if n < MAX_LOTS and float(n + 1) * step <= q:
        n += 1
    while n > 0 and float(n) * step > q:
        n -= 1
    return n


def reference_simulate(bars, targets, *, initial_cash, maker, taker, liquidity="taker",
                       slippage_bps=2.0, min_notional=0.0, step_size=0.0,
                       rebalance_threshold=0.0):
    rate = taker if liquidity == "taker" else maker
    slip = slippage_bps / 1e4
    step = step_size
    cash, lots, units = float(initial_cash), 0, 0.0
    position = lambda: float(lots) * step if step > 0.0 else units
    equity, pos_out, cash_out, fills = [], [], [], []
    fees_paid = 0.0
    for t in range(len(targets)):
        o, c = float(bars["open"][t]), float(bars["close"][t])
        if t > 0:
            target = float(targets[t - 1])
            pos = position()
            held = pos * o
            eq = cash + held
            fill = None
            if eq > 0.0:
                if target == 0.0:
                    side = "sell_full" if pos > 0.0 else None
                else:
                    w = held / eq
                    if abs(target - w) <= rebalance_threshold:
                        side = None
                    else:
                        delta = target * eq - held
                        side = "buy" if delta > 0.0 else ("sell" if delta < 0.0 else None)
                if side == "buy":
                    price = o * (1.0 + slip)
                    cap = cash / (price * (1.0 + rate))
                    raw = min(delta / price, cap)

                    def cost(q, price=price):
                        notional = q * price
                        fee = notional * rate
                        return notional, fee, notional + fee

                    if step > 0.0:
                        n = min(_floor_lots(raw, step), MAX_LOTS - lots)
                        while n > 0 and cost(float(n) * step)[2] > cash:
                            n -= 1
                        qty, nl = float(n) * step, n
                    else:
                        qty, tries, nl = raw, 0, 0
                        while qty > 0.0 and cost(qty)[2] > cash:
                            tries += 1
                            assert tries <= 64
                            qty *= 1.0 - 2.0 * EPS * float(tries)
                    notional, fee, total = cost(qty)
                    if not (qty <= 0.0 or notional < min_notional
                            or notional <= DUST_FRACTION * eq or total > cash):
                        if step > 0.0:
                            lots += nl
                        else:
                            units += qty
                        cash -= total
                        fill = (t, rs.BUY, qty, price, notional, fee)
                elif side in ("sell", "sell_full"):
                    price = o * (1.0 - slip)
                    raw = pos if side == "sell_full" else min(-delta / o, pos)
                    full = raw >= pos
                    if step > 0.0:
                        n = lots if full else min(_floor_lots(raw, step), lots)
                        qty = float(n) * step
                    else:
                        n, qty = 0, (pos if full else raw)
                    notional = qty * price
                    partial = side == "sell" and not full
                    if not (qty <= 0.0 or notional < min_notional
                            or (partial and notional <= DUST_FRACTION * eq)):
                        fee = notional * rate
                        if step > 0.0:
                            lots -= n
                        elif full:
                            units = 0.0
                        else:
                            units -= qty
                        cash += notional - fee
                        fill = (t, rs.SELL, qty, price, notional, fee)
            if fill is not None:
                fees_paid += fill[5]
                fills.append(fill)
        p = position()
        cash_out.append(cash)
        pos_out.append(p)
        equity.append(cash + p * c)
    return np.array(equity), np.array(pos_out), np.array(cash_out), fills, fees_paid


# --- simulate: hand-computed cases -------------------------------------------------------

THREE_BARS = {
    "open_time": np.array([0, 60_000, 120_000], dtype=np.int64),
    "open": np.array([100.0, 125.0, 140.0]),
    "high": np.array([101.0, 131.0, 151.0]),
    "low": np.array([99.0, 124.0, 139.0]),
    "close": np.array([100.0, 130.0, 150.0]),
    "volume": np.ones(3),
}


def test_a_round_trip_matches_the_hand_computation() -> None:
    # Decided at close 0, bought at open 1 (125); exited at open 2 (140). targets[-1] = 1 is
    # never executed. Taker 0.10 %, no slippage: every number below is exact or one rounding.
    r = run(THREE_BARS, np.array([0.5, 0.0, 1.0]), initial_cash=1000.0, slippage_bps=0.0)
    buy_fee = 500.0 * 0.001
    cash1 = 1000.0 - (500.0 + buy_fee)
    sell_fee = 560.0 * 0.001
    cash2 = cash1 + (560.0 - sell_fee)
    assert cash1 == 499.5
    assert same_bits(r.cash, np.array([1000.0, cash1, cash2]))
    assert same_bits(r.position, np.array([0.0, 4.0, 0.0]))
    assert same_bits(r.equity, np.array([1000.0, cash1 + 4.0 * 130.0, cash2]))
    f = r.fills
    assert len(f) == 2
    assert same_bits(f.bar, np.array([1, 2], dtype=np.int64))
    assert same_bits(f.time, np.array([60_000, 120_000], dtype=np.int64))
    assert same_bits(f.side, np.array([rs.BUY, rs.SELL], dtype=np.int8))
    assert same_bits(f.qty, np.array([4.0, 4.0]))
    assert same_bits(f.price, np.array([125.0, 140.0]))
    assert same_bits(f.notional, np.array([500.0, 560.0]))
    assert same_bits(f.fee, np.array([buy_fee, sell_fee]))
    assert r.fees_paid == 0.0 + buy_fee + sell_fee
    assert len(r) == 3


def test_slippage_and_maker_rates_follow_the_formulas() -> None:
    r = run(THREE_BARS, np.array([0.5, 0.0, 0.0]), initial_cash=1000.0,
            fees="coinbase-advanced-entry", liquidity="maker", slippage_bps=5.0)
    rate, s = 0.0025, 5.0 / 1e4
    buy = 125.0 * (1.0 + s)
    qty = min(500.0 / buy, 1000.0 / (buy * (1.0 + rate)))
    assert qty == 500.0 / buy, "half the equity: not capped by cash"
    cash1 = 1000.0 - (qty * buy + qty * buy * rate)
    sell = 140.0 * (1.0 - s)
    assert same_bits(r.fills.price, np.array([buy, sell]))
    assert same_bits(r.fills.qty, np.array([qty, qty]))
    assert same_bits(r.fills.fee, np.array([qty * buy * rate, qty * sell * rate]))
    assert same_bits(r.cash, np.array([1000.0, cash1, cash1 + (qty * sell - qty * sell * rate)]))


def test_filters_round_to_lots_and_skip_small_orders() -> None:
    lots = run(THREE_BARS, np.array([0.5, 0.0, 0.0]), initial_cash=1000.0,
               slippage_bps=0.0, step_size=0.3)
    # floor(4 / 0.3) = 13 lots, and 13 * 0.3 = 3.9000000000000004 <= 4.
    assert same_bits(lots.fills.qty, np.array([13 * 0.3, 13 * 0.3]))
    assert lots.position[2] == 0.0, "a full exit leaves no dust"
    skipped = run(THREE_BARS, np.array([0.5, 0.0, 0.0]), initial_cash=1000.0,
                  slippage_bps=0.0, min_notional=600.0)
    assert len(skipped.fills) == 0
    assert same_bits(skipped.equity, np.full(3, 1000.0))


def test_the_defaults_are_the_protocol_values() -> None:
    bars, targets = random_bars(np.random.default_rng(3), 50), random_targets(
        np.random.default_rng(4), 50)
    default = run(bars, targets)
    explicit = run(bars, targets, liquidity="taker", slippage_bps=2.0, min_notional=0.0,
                   step_size=0.0, rebalance_threshold=0.0)
    assert same_bits(default.equity, explicit.equity)
    assert fills_equal(default.fills, explicit.fills)
    assert not same_bits(default.equity, run(bars, targets, slippage_bps=0.0).equity)


# --- simulate: the binding returns exactly what Rust computes ----------------------------

CONFIGS = [
    {"fees": "binance-spot"},
    {"fees": "kraken-pro-entry", "liquidity": "maker", "slippage_bps": 7.5},
    {"fees": "zero", "slippage_bps": 0.0},
    {"fees": "bitvavo-entry", "step_size": 0.001, "min_notional": 5.0},
    {"fees": "coinbase-advanced-entry", "rebalance_threshold": 0.1, "step_size": 0.0001},
    {"fees": rs.FeeSchedule("custom-venue", 0.0003, 0.0007), "min_notional": 50.0},
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: str(c["fees"]))
@pytest.mark.parametrize("seed", range(8))
def test_simulate_equals_the_reference_port_bit_for_bit(cfg: dict, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 120))
    bars, targets = random_bars(rng, n), random_targets(rng, n)
    r = run(bars, targets, **cfg)
    fees = cfg["fees"]
    fees = fees if isinstance(fees, rs.FeeSchedule) else rs.FeeSchedule.named(fees)
    ref_cfg = {k: v for k, v in cfg.items() if k != "fees"}
    eq, pos, cash, fills, fees_paid = reference_simulate(
        bars, targets, initial_cash=10_000.0, maker=fees.maker, taker=fees.taker, **ref_cfg)
    assert same_bits(r.equity, eq)
    assert same_bits(r.position, pos)
    assert same_bits(r.cash, cash)
    assert r.fees_paid.hex() == fees_paid.hex()
    f = r.fills
    assert len(f) == len(fills)
    if fills:
        bar, side, qty, price, notional, fee = (np.array(c) for c in zip(*fills))
        assert same_bits(f.bar, bar.astype(np.int64))
        assert same_bits(f.time, bars["open_time"][bar])
        assert same_bits(f.side, side.astype(np.int8))
        for got, want in [(f.qty, qty), (f.price, price), (f.notional, notional), (f.fee, fee)]:
            assert same_bits(got, want.astype(np.float64))


# --- simulate against the compiled crate: the golden file ----------------------------------
#
# crates/backtest/examples/golden.rs runs backtest::simulate itself and writes its inputs and
# outputs as exact bits; crates/backtest/tests/golden.rs keeps the file equal to what the crate
# computes today. A port can share a bug with the simulator, the crate cannot.

GOLDEN = Path(__file__).parent / "golden" / "simulate.txt"
GOLDEN_FORMAT = "format olsprime-golden-simulate 1"
SCALARS = ("initial_cash", "slippage_bps", "min_notional", "step_size", "rebalance_threshold")


def _bits(fields) -> np.ndarray:
    """16-hex-digit bit patterns as uint64; `.view(np.float64)` gives the floats."""
    assert all(len(h) == 16 for h in fields)
    return np.array([int(h, 16) for h in fields], dtype=np.uint64)


def _float(h: str) -> float:
    return float(_bits([h]).view(np.float64)[0])


def _bits_of(x: float) -> int:
    return int(np.float64(x).view(np.uint64))


def load_golden(path: Path = GOLDEN) -> list[dict]:
    """The cases of the golden file. Strict: any line out of place is an error."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln and not ln.startswith("#")]
    pos = 0

    def take(key: str, n: int | None = None) -> list[str]:
        nonlocal pos
        head, *rest = lines[pos].split(" ")
        assert head == key, f"line {pos}: expected {key!r}, got {lines[pos]!r}"
        assert n is None or len(rest) == n, lines[pos]
        pos += 1
        return rest

    assert lines[0] == GOLDEN_FORMAT
    pos = 1
    cases = []
    for _ in range(int(take("cases", 1)[0])):
        case: dict = {"name": take("case", 1)[0]}
        name, maker, taker = take("fees", 3)
        case["fees"] = (name, int(maker, 16), int(taker, 16))
        case["liquidity"] = take("liquidity", 1)[0]
        case["scalars"] = {k: take(k, 1)[0] for k in SCALARS}
        bars = [take("b", 6) for _ in range(int(take("bars", 1)[0]))]
        case["open_time"] = np.array([int(b[0]) for b in bars], dtype=np.int64)
        case["ohlcv"] = [_bits([b[k] for b in bars]) for k in range(1, 6)]
        case["targets"] = _bits([take("t", 1)[0] for _ in range(int(take("targets", 1)[0]))])
        head = lines[pos].split(" ", 1)
        if head[0] == "error":
            case["error"] = head[1]
            pos += 1
        else:
            take("ok", 0)
            r = [take("r", 3) for _ in bars]
            case["error"] = None
            case["equity"], case["position"], case["cash"] = (
                _bits([x[k] for x in r]) for k in range(3))
            fills = [take("f", 7) for _ in range(int(take("fills", 1)[0]))]
            case["fills"] = {
                "bar": np.array([int(f[0]) for f in fills], dtype=np.int64),
                "time": np.array([int(f[1]) for f in fills], dtype=np.int64),
                "side": np.array([{"buy": rs.BUY, "sell": rs.SELL}[f[2]] for f in fills],
                                 dtype=np.int8),
                **{k: _bits([f[3 + i] for f in fills])
                   for i, k in enumerate(("qty", "price", "notional", "fee"))},
            }
            case["fees_paid"] = int(take("fees_paid", 1)[0], 16)
        take("end", 0)
        cases.append(case)
    assert pos == len(lines), f"trailing lines from {pos}"
    return cases


GOLDEN_CASES = load_golden()


def _golden_fees(case: dict):
    """The case's schedule as the binding takes it: a built-in name, or a custom schedule."""
    name, maker, taker = case["fees"]
    if name in rs.FeeSchedule.names():
        builtin = rs.FeeSchedule.named(name)
        assert (_bits_of(builtin.maker), _bits_of(builtin.taker)) == (maker, taker), name
        return name
    return rs.FeeSchedule(name, _float(f"{maker:016x}"), _float(f"{taker:016x}"))


def _golden_run(case: dict) -> rs.SimResult:
    o, h, l, c, v = (col.view(np.float64) for col in case["ohlcv"])
    scalars = {k: _float(x) for k, x in case["scalars"].items()}
    return rs.simulate(case["open_time"], o, h, l, c, v, case["targets"].view(np.float64),
                       fees=_golden_fees(case), liquidity=case["liquidity"], **scalars)


def _u64(x: np.ndarray) -> np.ndarray:
    assert x.dtype == np.float64
    return x.view(np.uint64)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES])
def test_simulate_equals_the_compiled_crate_bit_for_bit(case: dict) -> None:
    if case["error"] is not None:
        with pytest.raises(rs.SimError) as info:
            _golden_run(case)
        assert str(info.value) == case["error"]
        return
    r = _golden_run(case)
    for name in ("equity", "position", "cash"):
        assert np.array_equal(_u64(getattr(r, name)), case[name]), name
    assert _bits_of(r.fees_paid) == case["fees_paid"]
    got = r.fills.as_dict()
    assert got.keys() == case["fills"].keys()
    for name, want in case["fills"].items():
        have = got[name] if want.dtype != np.uint64 else _u64(got[name])
        assert have.dtype == want.dtype and np.array_equal(have, want), name


def test_the_golden_file_covers_errors_trades_and_edges() -> None:
    names = {c["name"] for c in GOLDEN_CASES}
    assert len(names) == len(GOLDEN_CASES) >= 40
    errors = [c for c in GOLDEN_CASES if c["error"] is not None]
    assert len(errors) >= 20
    assert sum(len(c["fills"]["bar"]) for c in GOLDEN_CASES if c["error"] is None) >= 200
    # NaN, -0.0 and a subnormal reach the binding as bits.
    every_target = np.concatenate([c["targets"] for c in GOLDEN_CASES])
    assert {0x8000_0000_0000_0000, 1} <= set(every_target.tolist())
    assert any(np.isnan(c["targets"].view(np.float64)).any() for c in errors)
    assert any(c["fees"][0] not in rs.FeeSchedule.names() for c in GOLDEN_CASES)


@pytest.mark.parametrize("case", [c for c in GOLDEN_CASES if c["error"] is None],
                         ids=[c["name"] for c in GOLDEN_CASES if c["error"] is None])
def test_the_reference_port_equals_the_compiled_crate(case: dict) -> None:
    """The port above is only as good as its agreement with the crate; check it here too."""
    o, h, l, c, v = (col.view(np.float64) for col in case["ohlcv"])
    _, maker, taker = case["fees"]
    scalars = {k: _float(x) for k, x in case["scalars"].items()}
    eq, pos, cash, fills, fees_paid = reference_simulate(
        {"open": o, "close": c}, case["targets"].view(np.float64),
        maker=_float(f"{maker:016x}"), taker=_float(f"{taker:016x}"),
        liquidity=case["liquidity"], **scalars)
    assert np.array_equal(_u64(eq), case["equity"])
    assert np.array_equal(_u64(pos), case["position"])
    assert np.array_equal(_u64(cash), case["cash"])
    assert _bits_of(fees_paid) == case["fees_paid"]
    assert [f[0] for f in fills] == case["fills"]["bar"].tolist()


@pytest.mark.parametrize("seed", range(20))
def test_documented_invariants_hold(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(1, 200))
    bars, targets = random_bars(rng, n), random_targets(rng, n)
    cfg = CONFIGS[seed % len(CONFIGS)]
    r = run(bars, targets, **cfg)
    assert len(r.equity) == len(r.position) == len(r.cash) == n
    assert np.all(r.cash >= 0.0) and np.all(r.position >= 0.0)
    assert r.equity[0] == 10_000.0
    f = r.fills
    assert np.all((f.bar >= 1) & (f.bar <= n - 1)), "bar 0 never trades"
    assert same_bits(f.time, bars["open_time"][f.bar])
    assert np.all(f.qty > 0.0)
    assert np.all(np.diff(f.bar) > 0), "at most one fill per bar"
    # targets[last] is validated but never executed: any other valid value changes nothing.
    other = targets.copy()
    other[-1] = 1.0 - other[-1]
    r2 = run(bars, other, **cfg)
    assert same_bits(r.equity, r2.equity) and same_bits(r.cash, r2.cash)
    assert fills_equal(r.fills, r2.fills)


@pytest.mark.parametrize("seed", range(10))
def test_outputs_up_to_k_ignore_the_future(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(5, 150))
    bars, targets = random_bars(rng, n), random_targets(rng, n)
    cfg = CONFIGS[seed % len(CONFIGS)]
    base = run(bars, targets, **cfg)
    for k in sorted({0, 1, n // 3, n // 2, n - 2}):
        # Replace every bar after k by another valid path (same times), and every target
        # decided at or after the close of k.
        future = random_bars(np.random.default_rng(seed + 7 * k + 1), n)
        scale = bars["close"][k] / future["open"][k + 1]
        alt = {c: v.copy() for c, v in bars.items()}
        for c in ("open", "high", "low", "close"):
            alt[c][k + 1:] = future[c][k + 1:] * scale
        alt["volume"][k + 1:] = future["volume"][k + 1:]
        alt_targets = targets.copy()
        alt_targets[k:] = random_targets(np.random.default_rng(seed + k), n - k)
        r = run(alt, alt_targets, **cfg)
        for name in ("equity", "position", "cash"):
            assert same_bits(getattr(base, name)[: k + 1], getattr(r, name)[: k + 1]), (name, k)
        assert fills_equal(base.fills, r.fills, upto=k)


def test_strided_and_reversed_views_are_read_in_logical_order() -> None:
    rng = np.random.default_rng(11)
    bars, targets = random_bars(rng, 60), random_targets(rng, 60)
    base = run(bars, targets)
    # Every column as a stride-2 view into a wider buffer, and a reversed view of a reversed
    # copy: neither is C-contiguous.
    strided = {}
    for c, v in bars.items():
        buf = np.empty(2 * len(v), dtype=v.dtype)
        buf[::2] = v
        buf[1::2] = v[::-1]
        strided[c] = buf[::2]
        assert not strided[c].flags.c_contiguous
    assert same_bits(run(strided, targets).equity, base.equity)
    reversed_ = {c: v[::-1].copy()[::-1] for c, v in bars.items()}
    r = run(reversed_, targets[::-1].copy()[::-1])
    assert same_bits(r.equity, base.equity)
    assert fills_equal(r.fills, base.fills)


def test_result_arrays_are_copies() -> None:
    r = run(THREE_BARS, np.array([0.5, 0.0, 1.0]))
    e = r.equity
    e[:] = -1.0
    r.fills.qty[:] = -1.0
    assert r.equity[0] == 10_000.0
    assert np.all(r.fills.qty > 0.0)


# --- metrics -----------------------------------------------------------------------------


def test_metrics_mirror_the_rust_definitions() -> None:
    rng = np.random.default_rng(21)
    bars, targets = random_bars(rng, 300), random_targets(rng, 300)
    r = run(bars, targets)
    ppy = rs.periods_per_year(60_000)
    assert ppy == 365.0 * 86_400_000.0 / 60_000
    m = rs.metrics(r, ppy)
    eq = r.equity
    ret = eq[1:] / eq[:-1] - 1.0
    assert m.periods == len(eq) - 1
    assert m.total_return == eq[-1] / eq[0] - 1.0
    assert m.annualized_return == pytest.approx((1 + m.total_return) ** (ppy / m.periods) - 1,
                                                rel=1e-12)
    std = ret.std(ddof=1)
    assert m.annualized_volatility == pytest.approx(std * math.sqrt(ppy), rel=1e-10)
    assert m.sharpe == pytest.approx(ret.mean() / std * math.sqrt(ppy), rel=1e-9)
    down = math.sqrt(np.mean(np.minimum(ret, 0.0) ** 2))
    assert m.sortino == pytest.approx(ret.mean() / down * math.sqrt(ppy), rel=1e-9)
    assert m.max_drawdown == rs.max_drawdown(eq)
    peak = np.maximum.accumulate(eq)
    assert m.max_drawdown == pytest.approx(np.max(1.0 - eq / peak), rel=1e-12)
    assert m.turnover == pytest.approx(
        r.fills.notional.sum() / eq.mean() * ppy / m.periods, rel=1e-10)
    assert m.exposure == np.mean(r.position > 0.0)
    assert m.fees_paid == r.fees_paid
    assert set(m.as_dict()) == {
        "periods", "total_return", "annualized_return", "annualized_volatility", "sharpe",
        "sortino", "max_drawdown", "turnover", "exposure", "fees_paid"}
    assert m == rs.metrics(r, ppy)


def test_flat_equity_has_no_ratio() -> None:
    r = run(THREE_BARS, np.zeros(3))
    m = rs.metrics(r, rs.periods_per_year(60_000))
    assert (m.sharpe, m.sortino, m.annualized_volatility, m.max_drawdown) == (None, None, 0.0, 0.0)
    assert rs.max_drawdown(np.array([], dtype=np.float64)) == 0.0
    assert rs.max_drawdown(np.array([2.0, 1.0, 3.0])) == 0.5


# --- ols-core ----------------------------------------------------------------------------


def design(rng: np.random.Generator, n: int, p: int, q: int):
    """States with a leading 1 and outcomes linear in them, plus noise."""
    x = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    y = x @ rng.normal(size=(p, q)) + 0.1 * rng.normal(size=(n, q))
    return x, y


def penalty(p: int, lam: float, intercept: bool) -> np.ndarray:
    d = np.full(p, lam)
    if intercept:
        d[0] = 0.0
    return np.diag(d)


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("lam", [0.0, 0.5, 10.0])
def test_ridge_fit_is_the_closed_form(intercept: bool, lam: float) -> None:
    x, y = design(np.random.default_rng(31), 80, 4, 2)
    stats = rs.SuffStats.from_rows(x, y)
    assert (stats.count, stats.n, stats.p, stats.q) == (80, 80.0, 4, 2)
    np.testing.assert_allclose(stats.gram, x.T @ x, rtol=1e-12)
    np.testing.assert_allclose(stats.cross, y.T @ x, rtol=1e-12)
    cfg = rs.RidgeConfig(lam, intercept=intercept)
    model = rs.ridge_fit(stats, cfg)
    beta = np.linalg.solve(x.T @ x + penalty(4, lam, intercept), x.T @ y).T
    assert model.beta.shape == (2, 4)
    np.testing.assert_allclose(model.beta, beta, rtol=1e-10)
    xq = np.array([1.0, 0.3, -0.2, 2.0])
    np.testing.assert_allclose(model.predict(xq), beta @ xq, rtol=1e-10)
    a = rs.solve_query(stats, cfg, xq)
    np.testing.assert_allclose(a, np.linalg.solve(x.T @ x + penalty(4, lam, intercept), xq),
                               rtol=1e-10)
    rss = ((y - x @ beta.T) ** 2).sum(axis=0)
    np.testing.assert_allclose(rs.residual_sum_squares(stats, model), rss, rtol=1e-8)


def test_ols_attention_weights_sum_to_one_with_an_intercept() -> None:
    rng = np.random.default_rng(41)
    keys, values = design(rng, 60, 3, 2)
    queries = np.column_stack([np.ones(7), rng.normal(size=(7, 2))])
    att = rs.ols_attention(keys, values, queries, rs.RidgeConfig.with_intercept(0.7))
    assert att.weights.shape == (7, 60) and att.forecast.shape == (7, 2)
    np.testing.assert_allclose(att.weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert (att.weights < 0).any(), "OLS weights are signed"
    np.testing.assert_allclose(att.forecast, att.weights @ values, rtol=1e-12)
    model = rs.ridge_fit(rs.SuffStats.from_rows(keys, values), rs.RidgeConfig.with_intercept(0.7))
    np.testing.assert_allclose(att.forecast, queries @ model.beta.T, rtol=1e-10)
    # Without a free intercept the rows do not sum to one.
    plain = rs.ols_attention(keys, values, queries, rs.RidgeConfig(0.7))
    assert not np.allclose(plain.weights.sum(axis=1), 1.0)


def filled_matrix(n: int = 50, block_log2: int = 2):
    x, y = design(np.random.default_rng(51), n, 3, 2)
    times = 1_000 * np.arange(n, dtype=np.int64) // 2  # pairs share a time
    m = rs.Matrix(3, 2, block_log2=block_log2)
    m.extend(times, x, y)
    return m, times, x, y


@pytest.mark.parametrize("start,end", [(0, 50), (3, 47), (5, 40), (17, 33), (12, 16)])
def test_matrix_query_is_batch_ridge_on_the_range(start: int, end: int) -> None:
    m, _, x, y = filled_matrix()
    cfg = rs.RidgeConfig.with_intercept(0.3)
    xq = np.array([1.0, -0.4, 1.1])
    readout = m.query(xq, start, end, cfg)
    assert (readout.start, readout.end) == (start, end)
    xr, yr = x[start:end], y[start:end]
    a = np.linalg.solve(xr.T @ xr + penalty(3, 0.3, True), xq)
    np.testing.assert_allclose(readout.a, a, rtol=1e-10)
    np.testing.assert_allclose(readout.forecast, yr.T @ xr @ a, rtol=1e-10)
    w = m.weights(readout)
    np.testing.assert_allclose(w, xr @ a, rtol=1e-10)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(w @ yr, readout.forecast, rtol=1e-10)
    # The contributions of a partition of the range add up to the forecast.
    cut = (start + end) // 2
    parts = m.contribution(readout, start, cut) + m.contribution(readout, cut, end)
    np.testing.assert_allclose(parts, readout.forecast, rtol=1e-10)
    # The tree's statistics are the batch statistics of the range.
    s = m.range_stats(start, end)
    batch = rs.SuffStats.from_rows(xr, yr)
    assert s.count == batch.count == end - start
    np.testing.assert_allclose(s.gram, batch.gram, rtol=1e-12)
    np.testing.assert_allclose(s.cross, batch.cross, rtol=1e-12)
    summary = m.summarize(start, end, cfg)
    np.testing.assert_allclose(summary.mean_y, yr.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(summary.model.beta, rs.ridge_fit(batch, cfg).beta, rtol=1e-10)


def test_matrix_extend_is_push_and_time_range_is_causal() -> None:
    m, times, x, y = filled_matrix(block_log2=1)
    pushed = rs.Matrix(3, 2, block_log2=1)
    for t, xi, yi in zip(times, x, y):
        pushed.push(int(t), xi, yi)
    assert len(m) == len(pushed) == 50
    assert same_bits(m.times, pushed.times) and same_bits(m.times, times)
    for start, end in [(0, 50), (7, 29), (31, 32)]:
        assert same_bits(m.range_stats(start, end).gram, pushed.range_stats(start, end).gram)
    for as_of in [-1, 0, 999, 1_000, 12_345, 10**9]:
        for from_time in [-(10**9), 0, 500, 3_000]:
            start = int(np.searchsorted(times, from_time, side="left"))
            end = max(int(np.searchsorted(times, as_of, side="right")), start)
            assert m.time_range(from_time, as_of) == (start, end)
    # A rejected batch changes nothing: the third entry goes back in time.
    before = m.range_stats(0, len(m))
    with pytest.raises(rs.OlsError, match="time went backwards"):
        m.extend(np.array([10**6, 10**6 + 1, 0], dtype=np.int64), x[:3], y[:3])
    assert len(m) == 50 and m.range_stats(0, 50) == before


def test_rls_is_weighted_ridge() -> None:
    gamma, lam = 0.97, 0.3
    x, y = design(np.random.default_rng(61), 60, 3, 1)
    rls = rs.Rls(3, 1, lam, gamma)
    for xi, yi in zip(x, y):
        rls.update(xi, yi)
    n = len(x)
    assert (rls.updates, rls.gamma) == (n, gamma)
    w = gamma ** (n - 1 - np.arange(n))
    g = (x * w[:, None]).T @ x + gamma**n * lam * np.eye(3)
    beta = np.linalg.solve(g, (x * w[:, None]).T @ y).T
    np.testing.assert_allclose(rls.beta, beta, rtol=1e-9)
    # The same weighted ridge through the statistics path.
    stats = rs.SuffStats(3, 1)
    for wi, xi, yi in zip(w, x, y):
        stats.push(xi, yi, float(wi))
    model = rs.ridge_fit(stats, rs.RidgeConfig(gamma**n * lam))
    np.testing.assert_allclose(model.beta, rls.beta, rtol=1e-9)
    np.testing.assert_allclose(rls.predict(x[0]), beta @ x[0], rtol=1e-9)


def test_suffstats_merge_and_copy() -> None:
    x, y = design(np.random.default_rng(71), 20, 2, 1)
    left, right = rs.SuffStats.from_rows(x[:8], y[:8]), rs.SuffStats.from_rows(x[8:], y[8:])
    union = left.merged(right)
    np.testing.assert_allclose(union.gram, rs.SuffStats.from_rows(x, y).gram, rtol=1e-12)
    assert left.count == 8, "merged leaves its operands unchanged"
    twin = left.copy()
    twin.merge(twin)  # aliasing is safe: the operand is copied first
    assert twin.count == 16 and left.count == 8
    assert same_bits(twin.gram, 2.0 * left.gram)
    assert rs.SuffStats(2, 1).mean_y() is None
    np.testing.assert_allclose(union.mean_y(), y.mean(axis=0), rtol=1e-12)


# --- errors ------------------------------------------------------------------------------


def _sim_case(**changes):
    bars = {c: v.copy() for c, v in THREE_BARS.items()}
    targets = changes.pop("targets", np.zeros(3))
    for (c, i), v in changes.pop("set", {}).items():
        bars[c][i] = v
    return lambda: run(bars, targets, **changes)


EMPTY = {c: v[:0] for c, v in THREE_BARS.items()}

SIM_ERRORS = [
    (lambda: run(EMPTY, np.zeros(0)), r"^no bars$"),
    (_sim_case(targets=np.zeros(2)), r"^3 bars but 2 targets$"),
    (_sim_case(set={("open", 1): -1.0}), r"^invalid bar 1: OHLC must be finite and > 0$"),
    (_sim_case(set={("low", 0): 100.5}), r"^invalid bar 0: low above min\(open, close\)$"),
    (_sim_case(set={("volume", 2): np.nan}), r"^invalid bar 2: volume must be finite"),
    (_sim_case(set={("open_time", 2): 60_000}),
     r"^open_time of bar 2 is not strictly after the previous bar's$"),
    (_sim_case(targets=np.array([1.5, 0.0, 0.0])),
     r"^invalid target 1.5 at 0: must be finite and in \[0, 1\]$"),
    (_sim_case(initial_cash=0.0), r"^invalid config: initial_cash = 0$"),
    (_sim_case(slippage_bps=1e4), r"^invalid config: slippage_bps = 10000$"),
    (_sim_case(rebalance_threshold=1.0), r"^invalid config: rebalance_threshold = 1$"),
    (_sim_case(step_size=-0.1), r"^invalid config: filters.step_size = -0.1$"),
]


@pytest.mark.parametrize("call,message", SIM_ERRORS)
def test_sim_errors_are_sim_error_with_the_rust_message(call, message) -> None:
    with pytest.raises(rs.SimError, match=message) as info:
        call()
    assert isinstance(info.value, ValueError)


def _foreign_readout():
    m1, m2 = filled_matrix()[0], filled_matrix()[0]
    r = m1.query(np.array([1.0, 0.0, 0.0]), 0, 50, rs.RidgeConfig(1.0))
    return m2.weights(r)


def _pushed_matrix(*entries):
    m = rs.Matrix(1, 1)
    for t, v in entries:
        m.push(t, np.array([v]), np.array([0.0]))
    return m


OLS_ERRORS = [
    (lambda: rs.ridge_fit(rs.SuffStats(2, 1), rs.RidgeConfig.with_intercept(1.0)),
     r"^the regularized Gram matrix is not positive definite$"),
    (lambda: rs.RidgeConfig(-1.0), r"^invalid ridge penalty -1$"),
    (lambda: rs.RidgeConfig(float("nan"), intercept=True), r"^invalid ridge penalty NaN$"),
    (lambda: rs.Rls(2, 1, 0.0), r"^invalid ridge penalty 0$"),
    (lambda: rs.Rls(2, 1, 1.0, gamma=1.5), r"^invalid forgetting factor 1.5"),
    (lambda: rs.SuffStats(2, 1).push(np.ones(3), np.ones(1)),
     r"^dimension mismatch: expected 2, got 3$"),
    (lambda: rs.SuffStats.from_rows(np.ones((3, 2)), np.ones((2, 1))),
     r"^dimension mismatch: expected 3, got 2$"),
    (lambda: rs.Matrix(1, 1).range_stats(0, 1), r"^range \[0, 1\) is not within \[0, 0\)$"),
    (lambda: rs.Matrix(1, 1).summarize(0, 0, rs.RidgeConfig(1.0)), r"^the range holds no entry$"),
    (lambda: _pushed_matrix((5, 1.0), (4, 1.0)),
     r"^time went backwards: last entry at 5, new entry at 4$"),
    (lambda: rs.SuffStats(1, 1).push(np.ones(1), np.ones(1), -1.0), r"^negative weight -1$"),
    (lambda: _pushed_matrix((0, np.inf)), r"^non-finite value in input$"),
    (_foreign_readout, r"^the readout was produced by another Matrix instance$"),
    (lambda: rs.Matrix(0, 1), r"^invalid shape: p and q must be at least 1$"),
    (lambda: rs.Matrix(1, 1, block_log2=21), r"^invalid shape: block_log2 must be at most 20$"),
    (lambda: rs.ols_attention(np.ones((3, 0)), np.ones((3, 1)), np.ones((1, 0)),
                              rs.RidgeConfig(1.0)), r"^invalid shape"),
    (lambda: rs.SuffStats(0, 1), r"^invalid shape"),
]


@pytest.mark.parametrize("call,message", OLS_ERRORS)
def test_ols_errors_are_ols_error_with_the_rust_message(call, message) -> None:
    with pytest.raises(rs.OlsError, match=message) as info:
        call()
    assert isinstance(info.value, ValueError)


def test_contribution_outside_the_readout_is_out_of_range() -> None:
    m = filled_matrix()[0]
    r = m.query(np.array([1.0, 0.0, 0.0]), 10, 20, rs.RidgeConfig(1.0))
    with pytest.raises(rs.OlsError, match=r"^range \[5, 15\) is not within \[10, 20\)$"):
        m.contribution(r, 5, 15)


# --- risk --------------------------------------------------------------------------------

USDT = rs.QUOTE_SCALE


def doc_example(**account_changes):
    """The example of the risk crate's docs: 100 USDT, buy 0.001 BTC at 67,000 USDT."""
    account = {"equity": 100 * USDT, "peak_equity": 100 * USDT, "day_start_equity": 100 * USDT,
                   "position": 0, "bars_since_order": 10}
    account.update(account_changes)
    limits = rs.Limits(max_position=100 * USDT, max_order=50 * USDT, min_order=5 * USDT,
                       lot_step=1_000, max_daily_loss=10 * USDT, max_drawdown_bps=2_000,
                       cooldown_bars=3)
    request = rs.OrderRequest(side=rs.Side.BUY, qty=100_000, price=67_000 * USDT)
    return rs.Account(**account), limits, request


def test_check_order_approves_the_doc_example() -> None:
    assert (rs.QTY_SCALE, rs.QUOTE_SCALE, rs.BPS_SCALE) == (100_000_000, 1_000_000, 10_000)
    decision = rs.check_order(*doc_example())
    assert isinstance(decision, rs.Approved)
    assert decision.side == rs.Side.BUY
    assert decision.qty == 74_000
    assert decision.notional == 49_580_000  # 49.58 USDT


def test_check_order_rejects_with_the_kernel_reason() -> None:
    cooldown = rs.check_order(*doc_example(bars_since_order=1))
    assert isinstance(cooldown, rs.Rejected)
    assert cooldown.reason == rs.Rejection.COOLDOWN
    assert cooldown.message == "cooldown active: too few bars since the last order"
    # A 30 % drawdown trips the 20 % kill switch.
    killed = rs.check_order(*doc_example(equity=70 * USDT))
    assert killed.reason == rs.Rejection.KILL_SWITCH
    account, limits, _ = doc_example()
    sell = rs.OrderRequest(side=rs.Side.SELL, qty=1_000, price=67_000 * USDT)
    assert rs.check_order(account, limits, sell).reason == rs.Rejection.NO_POSITION


def test_risk_types_take_exact_integers_by_keyword_only() -> None:
    account, _limits, _request = doc_example()
    assert account == doc_example()[0] and hash(account) == hash(doc_example()[0])
    assert rs.Account(equity=np.int64(1), peak_equity=1, day_start_equity=1, position=0,
                      bars_since_order=np.uint32(0)).equity == 1
    good = {"equity": 1, "peak_equity": 1, "day_start_equity": 1, "position": 0, "bars_since_order": 0}
    for field, bad, error in [
        ("equity", 1.0, TypeError),
        ("equity", np.float64(1.0), TypeError),
        ("equity", "1", TypeError),
        ("position", True, TypeError),
        ("equity", 2**63, ValueError),
        ("equity", 2**200, ValueError),
        ("bars_since_order", -1, ValueError),
        ("bars_since_order", 2**32, ValueError),
    ]:
        with pytest.raises(error, match=field):
            rs.Account(**{**good, field: bad})
    with pytest.raises(TypeError):
        rs.Account(1, 1, 1, 0, 0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        rs.OrderRequest(side="buy", qty=1, price=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rs.Approved()  # type: ignore[call-arg]


# --- misuse ------------------------------------------------------------------------------


@pytest.mark.parametrize("column,bad", [
    ("open", THREE_BARS["open"].astype(np.float32)),
    ("open", THREE_BARS["open"].astype(np.int64)),
    ("open", THREE_BARS["open"].astype(">f8")),
    ("open", list(THREE_BARS["open"])),
    ("open", THREE_BARS["open"].reshape(3, 1)),
    ("open_time", THREE_BARS["open_time"].astype(np.float64)),
    ("open_time", THREE_BARS["open_time"].astype(np.int32)),
])
def test_wrong_dtypes_and_shapes_are_type_errors(column, bad) -> None:
    bars = {**THREE_BARS, column: bad}
    with pytest.raises(TypeError, match=column):
        run(bars, np.zeros(3))


def test_wrong_names_and_lengths_are_value_errors() -> None:
    with pytest.raises(ValueError, match="column high has 2 values but open_time has 3"):
        run({**THREE_BARS, "high": THREE_BARS["high"][:2]}, np.zeros(3))
    with pytest.raises(ValueError,
                       match=r'unknown fee schedule "binance"; expected one of \["zero", "binance'):
        run(THREE_BARS, np.zeros(3), fees="binance")
    with pytest.raises(ValueError, match="unknown fee schedule"):
        rs.FeeSchedule.named("Kraken-Pro-Entry")
    with pytest.raises(TypeError, match="fees must be a FeeSchedule"):
        run(THREE_BARS, np.zeros(3), fees=0.001)
    with pytest.raises(ValueError, match="unknown liquidity"):
        run(THREE_BARS, np.zeros(3), liquidity="Taker")
    with pytest.raises(TypeError):
        rs.simulate(*THREE_BARS.values(), np.zeros(3), 1000.0, "zero")  # keyword-only config
    with pytest.raises(ValueError, match="periods_per_year"):
        rs.metrics(run(THREE_BARS, np.zeros(3)), 0.0)
    with pytest.raises(ValueError, match="bar length"):
        rs.periods_per_year(0)
    with pytest.raises(TypeError):
        rs.Matrix(1, 1).push(0, np.ones(1, dtype=np.float32), np.ones(1))
    with pytest.raises(rs.OlsError, match="expected 3, got 2"):
        rs.Matrix(3, 1).extend(np.zeros(2, dtype=np.int64), np.ones((2, 2)), np.ones((2, 1)))


def test_the_audit_examples_of_overflow_are_value_errors() -> None:
    """Audit B: these raised PyO3's OverflowError, which is not a ValueError."""
    for call, name in [
        (lambda: rs.Matrix(1, 1).range_stats(-1, 2), "start"),
        (lambda: rs.Matrix(1, 1).time_range(2**63, 0), "from_time"),
        (lambda: rs.Matrix(2, 1, block_log2=-1), "block_log2"),
        (lambda: rs.periods_per_year(2**70), "bar_ms"),
    ]:
        with pytest.raises(ValueError, match=rf"^{name} ") as info:
            call()
        assert type(info.value) is ValueError, type(info.value)


def _two_entries() -> rs.Matrix:
    m = rs.Matrix(1, 1)
    m.extend(np.array([0, 1], dtype=np.int64), np.ones((2, 1)), np.ones((2, 1)))
    return m


_RIDGE = rs.RidgeConfig(1.0)
_ACCOUNT = {"equity": 1, "peak_equity": 1, "day_start_equity": 1, "position": 0,
            "bars_since_order": 0}
_LIMITS = {"max_position": 1, "max_order": 1, "min_order": 1, "lot_step": 1,
           "max_daily_loss": 1, "max_drawdown_bps": 1, "cooldown_bars": 1}

# Every int and float parameter of the stubs, as a call that passes `v` there and valid values
# everywhere else. With v = 1 (an int) or 0.5 (a float) each call succeeds.
SCALAR_PARAMS = {
    ("FeeSchedule.__init__", "maker"): lambda v: rs.FeeSchedule("v", v, 0.0),
    ("FeeSchedule.__init__", "taker"): lambda v: rs.FeeSchedule("v", 0.0, v),
    **{("simulate", k): (lambda v, k=k: run(THREE_BARS, np.zeros(3), **{k: v})) for k in SCALARS},
    ("metrics", "periods_per_year"): lambda v: rs.metrics(run(THREE_BARS, np.zeros(3)), v),
    ("periods_per_year", "bar_ms"): lambda v: rs.periods_per_year(v),
    ("RidgeConfig.__init__", "lam"): lambda v: rs.RidgeConfig(v),
    ("RidgeConfig.with_intercept", "lam"): lambda v: rs.RidgeConfig.with_intercept(v),
    ("SuffStats.__init__", "p"): lambda v: rs.SuffStats(v, 1),
    ("SuffStats.__init__", "q"): lambda v: rs.SuffStats(1, v),
    ("SuffStats.push", "w"): lambda v: rs.SuffStats(1, 1).push(np.ones(1), np.ones(1), v),
    **{("Rls.__init__", k): (lambda v, k=k: rs.Rls(**{"p": 1, "q": 1, "lam": 1.0, "gamma": 1.0,
                                                      k: v}))
       for k in ("p", "q", "lam", "gamma")},
    **{("Matrix.__init__", k): (lambda v, k=k: rs.Matrix(**{"p": 1, "q": 1, "block_log2": 2,
                                                            k: v}))
       for k in ("p", "q", "block_log2")},
    ("Matrix.push", "time"): lambda v: _two_entries().push(v, np.ones(1), np.ones(1)),
    **{("Matrix.time_range", k): (lambda v, k=k: _two_entries().time_range(
        **{"from_time": 0, "as_of": 1, k: v})) for k in ("from_time", "as_of")},
    **{("Matrix.range_stats", k): (lambda v, k=k: _two_entries().range_stats(
        **{"start": 0, "end": 2, k: v})) for k in ("start", "end")},
    **{("Matrix.query", k): (lambda v, k=k: _two_entries().query(
        np.ones(1), **{"start": 0, "end": 2, k: v}, cfg=_RIDGE)) for k in ("start", "end")},
    **{("Matrix.contribution", k): (lambda v, k=k: (lambda m: m.contribution(
        m.query(np.ones(1), 0, 2, _RIDGE), **{"start": 0, "end": 2, k: v}))(_two_entries()))
       for k in ("start", "end")},
    **{("Matrix.summarize", k): (lambda v, k=k: _two_entries().summarize(
        **{"start": 0, "end": 2, k: v}, cfg=_RIDGE)) for k in ("start", "end")},
    **{("Account.__init__", k): (lambda v, k=k: rs.Account(**{**_ACCOUNT, k: v}))
       for k in _ACCOUNT},
    **{("Limits.__init__", k): (lambda v, k=k: rs.Limits(**{**_LIMITS, k: v})) for k in _LIMITS},
    **{("OrderRequest.__init__", k): (lambda v, k=k: rs.OrderRequest(
        side=rs.Side.BUY, **{"qty": 1, "price": 1, k: v})) for k in ("qty", "price")},
}


def _stub_scalar_params() -> dict[tuple[str, str], str]:
    """(qualified name, parameter) -> "int" or "float", for every scalar parameter of the stubs."""
    tree = ast.parse((Path(rs.__file__).parent / "__init__.pyi").read_text(encoding="utf-8"))
    out = {}

    def walk(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, node.name + ".")
            elif isinstance(node, ast.FunctionDef):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs:
                    if arg.annotation is not None and ast.unparse(arg.annotation) in ("int", "float"):
                        out[(prefix + node.name, arg.arg)] = ast.unparse(arg.annotation)

    walk(tree.body)
    return out


STUB_SCALARS = _stub_scalar_params()
INT_OUT_OF_RANGE = [2**64, -(2**64), 2**200, -(2**200)]
FLOAT_OUT_OF_RANGE = [2**1024, -(2**1024), 10**400]


def test_every_scalar_parameter_of_the_stubs_is_checked_below() -> None:
    assert set(SCALAR_PARAMS) == set(STUB_SCALARS)


@pytest.mark.parametrize("key", sorted(STUB_SCALARS), ids="{0[0]}:{0[1]}".format)
def test_out_of_range_scalars_are_value_errors_naming_the_argument(key) -> None:
    call, kind = SCALAR_PARAMS[key], STUB_SCALARS[key]
    call(1 if kind == "int" else 0.5)  # the other arguments are valid
    for bad in INT_OUT_OF_RANGE if kind == "int" else FLOAT_OUT_OF_RANGE:
        with pytest.raises(ValueError, match=rf"^{key[1]} ") as info:
            call(bad)
        assert type(info.value) is ValueError, (bad, type(info.value))
    for bad in (True, "1", None, [1]) + ((1.0, np.float64(1.0)) if kind == "int" else ()):
        with pytest.raises(TypeError, match=rf"^{key[1]} must be an? {kind}, got ") as info:
            call(bad)
        assert type(info.value) is TypeError, (bad, type(info.value))


def test_numpy_scalars_are_accepted_where_ints_and_floats_are() -> None:
    m = rs.Matrix(np.int64(1), np.uint8(1), block_log2=np.int32(2))
    m.push(np.int64(5), np.ones(1), np.ones(1))
    assert m.time_range(np.int64(0), np.uint64(5)) == (0, 1)
    assert m.range_stats(np.intp(0), np.int64(1)).count == 1
    assert rs.RidgeConfig(np.float32(0.5)).lam == 0.5
    assert rs.periods_per_year(np.int64(60_000)) == rs.periods_per_year(60_000)
    # Ints where floats go are converted, as Python itself would.
    assert rs.RidgeConfig(2).lam == 2.0


def test_microsecond_bar_times_are_refused() -> None:
    """Spot files switched to microseconds on 2025-01-01: a µs series must not pass as ms."""
    ms = 1_735_689_600_000 + 60_000 * np.arange(3, dtype=np.int64)  # 2025-01-01, in ms
    run({**THREE_BARS, "open_time": ms}, np.zeros(3))
    with pytest.raises(ValueError, match=r"^open_time\[0\] = 1735689600000000 looks like "
                                         r"microseconds") as info:
        run({**THREE_BARS, "open_time": ms * 1000}, np.zeros(3))
    assert type(info.value) is ValueError
    # The boundary is 10**14 (the year 5138 in ms), and the error names the first time beyond.
    last = np.array([0, 60_000, 10**14 - 1], dtype=np.int64)
    assert len(run({**THREE_BARS, "open_time": last}, np.zeros(3))) == 3
    with pytest.raises(ValueError, match=r"^open_time\[1\] = 100000000000000 looks like"):
        run({**THREE_BARS, "open_time": np.array([0, 10**14, 2**62], dtype=np.int64)},
            np.zeros(3))
    # Unit before order: the simulator would also refuse this decreasing series.
    with pytest.raises(ValueError, match=r"^open_time\[0\] = 9223372036854775807 looks like"):
        run({**THREE_BARS, "open_time": np.array([2**63 - 1, 0, 1], dtype=np.int64)},
            np.zeros(3))


def test_matrix_refuses_microsecond_times() -> None:
    """Matrix times are milliseconds too: push and extend refuse a time >= 10**14."""
    assert rs.MICROSECOND_TIMES_FROM == 10**14
    m = rs.Matrix(1, 1)
    m.push(rs.MICROSECOND_TIMES_FROM - 1, np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match=r"^time = 100000000000000 looks like microseconds: "
                                         r"Matrix times are milliseconds since the epoch") as info:
        m.push(rs.MICROSECOND_TIMES_FROM, np.ones(1), np.ones(1))
    assert type(info.value) is ValueError
    assert "  " not in str(info.value)
    m = rs.Matrix(1, 1)
    times = np.array([0, 1_735_689_600_000_000, 1_735_689_600_000_001], dtype=np.int64)
    with pytest.raises(ValueError, match=r"^times\[1\] = 1735689600000000 looks like "
                                         r"microseconds") as info:
        m.extend(times, np.ones((3, 1)), np.ones((3, 1)))
    assert type(info.value) is ValueError
    assert len(m) == 0  # all or nothing
    m.extend(times // 1000, np.ones((3, 1)), np.ones((3, 1)))
    assert len(m) == 3


def test_matrix_time_range_refuses_microsecond_bounds() -> None:
    """An as_of in microseconds would expose every entry: a silent look-ahead."""
    m = rs.Matrix(1, 1)
    times = 1_735_689_600_000 + 60_000 * np.arange(3, dtype=np.int64)  # 2025-01-01, in ms
    m.extend(times, np.ones((3, 1)), np.ones((3, 1)))
    assert m.time_range(0, int(times[0])) == (0, 1)
    assert m.time_range(-(2**63), rs.MICROSECOND_TIMES_FROM - 1) == (0, 3)
    for kwargs, name in [({"from_time": 0, "as_of": 1_735_689_600_000_000}, "as_of"),
                         ({"from_time": rs.MICROSECOND_TIMES_FROM, "as_of": 0}, "from_time"),
                         ({"from_time": 0, "as_of": 2**63 - 1}, "as_of")]:
        with pytest.raises(ValueError, match=rf"^{name} = {kwargs[name]} looks like "
                                             r"microseconds: Matrix times are milliseconds") as info:
            m.time_range(**kwargs)
        assert type(info.value) is ValueError


def test_the_microsecond_message_of_simulate_is_one_line() -> None:
    ms = 1_735_689_600_000 + 60_000 * np.arange(3, dtype=np.int64)
    with pytest.raises(ValueError, match=r"since the epoch, and every time") as info:
        run({**THREE_BARS, "open_time": ms * 1000}, np.zeros(3))
    assert "  " not in str(info.value)


# --- sizes --------------------------------------------------------------------------------
# A failed allocation aborts the interpreter instead of raising, so every allocation sized by a
# caller's shape is refused first. These run in-process on purpose: the refusal happens before
# anything is allocated, and an abort would still fail the run, loudly.

_P = 2**20  # a 2**20 x 2**20 Gram matrix is 8 TiB


def _wide(rows: int, cols: int) -> np.ndarray:
    """A float64 array of any logical shape that costs 8 bytes: a zero-stride view."""
    return np.broadcast_to(np.zeros(1), (rows, cols))


_LIMIT = r"above the limit MAX_ALLOC_BYTES = 1073741824 \(1 GiB\)$"

OVERSIZE = [
    ("SuffStats", lambda: rs.SuffStats(_P, 1),
     r"^SuffStats\(p=1048576, q=1\) would take 8796109799448 bytes, "),
    # The largest p and q an int can pass: the byte count leaves u128, and is still refused.
    ("SuffStats:usize", lambda: rs.SuffStats(2**64 - 1, 2**64 - 1),
     r"^SuffStats\(p=18446744073709551615, q=18446744073709551615\) would take more than "
     r"2\^128 bytes, "),
    ("SuffStats.from_rows", lambda: rs.SuffStats.from_rows(np.zeros((0, _P)), np.zeros((0, 1))),
     r"^SuffStats\(p=1048576, q=1\) from xs of shape \(0, 1048576\) and ys of shape \(0, 1\) "
     r"would take 8796109799448 bytes, "),
    ("SuffStats.from_rows:xs", lambda: rs.SuffStats.from_rows(_wide(2**28, 1), _wide(2**28, 1)),
     r"^xs of shape \(268435456, 1\) would take 2147483648 bytes, "),
    ("ols_attention:gram", lambda: rs.ols_attention(np.zeros((1, _P)), np.zeros((1, 1)),
                                                   np.zeros((1, _P)), _RIDGE),
     r"^ols_attention: the Gram matrix of keys of shape \(1048576, 1048576\), for keys "
     r"\(1, 1048576\), values \(1, 1\) and queries \(1, 1048576\), would take "),
    ("ols_attention:weights", lambda: rs.ols_attention(np.zeros((2**17, 1)), np.zeros((2**17, 1)),
                                                      np.zeros((2**17, 1)), _RIDGE),
     r"^ols_attention: the weights of shape \(131072, 131072\), .* would take 137438953472 "),
    ("ols_attention:keys", lambda: rs.ols_attention(_wide(2**28, 1), _wide(2**28, 1),
                                                   np.zeros((0, 1)), _RIDGE),
     r"^keys of shape \(268435456, 1\) would take 2147483648 bytes, "),
    ("Rls", lambda: rs.Rls(_P, 1, 1.0),
     r"^Rls\(p=1048576, q=1\) would take 8796101410816 bytes, "),
    ("Matrix", lambda: rs.Matrix(_P, 1),
     r"^Matrix\(p=1048576, q=1\): one tree node would take 8796109799448 bytes, "),
    ("Matrix.push:x", lambda: rs.Matrix(1, 1).push(0, _wide(1, 2**28)[0], np.ones(1)),
     r"^x \(268435456 values\) would take 2147483648 bytes, "),
    ("Matrix.extend:xs", lambda: rs.Matrix(1, 1).extend(np.broadcast_to(np.zeros(1, np.int64),
                                                                         (2**28,)),
                                                         _wide(2**28, 1), _wide(2**28, 1)),
     r"^times \(268435456 values\) would take 2147483648 bytes, "),
    ("max_drawdown", lambda: rs.max_drawdown(_wide(1, 2**40)[0]),
     r"^equity \(1099511627776 values\) would take 8796093022208 bytes, "),
    ("simulate", lambda: run({k: np.broadcast_to(v[:1], (2**25,)) for k, v in THREE_BARS.items()},
                             np.broadcast_to(np.zeros(1), (2**25,))),
     r"^open_time \(33554432 bars\) would take 1610612736 bytes, "),
]


@pytest.mark.parametrize("name,call,message", OVERSIZE, ids=[c[0] for c in OVERSIZE])
def test_oversize_shapes_are_value_errors_before_any_allocation(name, call, message) -> None:
    with pytest.raises(ValueError, match=message + ".*" + _LIMIT) as info:
        call()
    assert type(info.value) is ValueError, type(info.value)


def test_the_size_limit_is_exported_and_real_shapes_pass() -> None:
    assert rs.MAX_ALLOC_BYTES == 2**30
    # A protocol-sized state passes everywhere the limit applies.
    assert rs.SuffStats(64, 4).p == 64
    assert rs.Rls(64, 4, 1.0).updates == 0
    assert rs.Matrix(64, 4).p == 64
    keys = np.random.default_rng(0).normal(size=(50, 3))
    att = rs.ols_attention(keys, np.ones((50, 1)), keys[:4], _RIDGE)
    assert att.weights.shape == (4, 50)
    # A zero-stride view under the limit is read, in its logical order, like any array.
    assert rs.max_drawdown(np.broadcast_to(np.ones(1), (1000,))) == 0.0


def test_fee_schedules() -> None:
    assert rs.FeeSchedule.names() == ["zero", "binance-spot", "binance-spot-bnb",
                                      "coinbase-advanced-entry", "kraken-pro-entry",
                                      "bitvavo-entry"]
    bitvavo = rs.FeeSchedule.named("bitvavo-entry")
    assert bitvavo == rs.FeeSchedule.BITVAVO_ENTRY
    assert (bitvavo.maker, bitvavo.taker) == (0.0015, 0.0025)
    assert bitvavo.rate("taker") == 0.0025 and bitvavo.rate("maker") == 0.0015
    k = rs.FeeSchedule.KRAKEN_PRO_ENTRY
    assert (k.name, k.maker, k.taker) == ("kraken-pro-entry", 0.004, 0.008)
    with pytest.raises(ValueError, match="built-in"):
        rs.FeeSchedule("binance-spot", 0.0, 0.0)
    with pytest.raises(ValueError, match="maker"):
        rs.FeeSchedule("v", 1.0, 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        rs.FeeSchedule("", 0.0, 0.0)
    # A schedule and its name give the same simulation.
    by_name = run(THREE_BARS, np.array([1.0, 0.0, 0.0]), fees="kraken-pro-entry")
    by_obj = run(THREE_BARS, np.array([1.0, 0.0, 0.0]), fees=k)
    custom = run(THREE_BARS, np.array([1.0, 0.0, 0.0]),
                 fees=rs.FeeSchedule("same-rates", 0.004, 0.008))
    assert same_bits(by_name.equity, by_obj.equity) and same_bits(by_name.equity, custom.equity)


# --- packaging ---------------------------------------------------------------------------


def _stub_names(body: list[ast.stmt]) -> set[str]:
    names = set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {n for n in names if not n.startswith("_") or n == "__version__"}


def test_the_stubs_describe_the_module_exactly() -> None:
    here = Path(rs.__file__).parent
    assert (here / "py.typed").exists()
    tree = ast.parse((here / "__init__.pyi").read_text(encoding="utf-8"))
    # maturin's __init__.py re-exports the native submodule, itself named olsprime_rs.
    public = {n for n in dir(rs) if not n.startswith("_")
              and not isinstance(getattr(rs, n), type(rs))} | {"__version__"}
    assert _stub_names(tree.body) == public
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        members = {n for n in dir(getattr(rs, cls.name)) if not n.startswith("_")}
        if issubclass(getattr(rs, cls.name), BaseException):
            continue
        assert _stub_names(cls.body) == members, cls.name
    assert re.fullmatch(r"\d+\.\d+\.\d+", rs.__version__)
