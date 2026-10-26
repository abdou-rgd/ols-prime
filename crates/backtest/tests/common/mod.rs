//! Shared builders and proptest strategies for the integration tests.
#![allow(dead_code)]

use backtest::{Bar, FeeSchedule, Fill, Liquidity, SimConfig, SymbolFilters};
use proptest::prelude::*;

pub const T0: i64 = 1_600_000_000_000;
pub const HOUR: i64 = 3_600_000;

/// A bar at index `i` of an hourly grid, with wicks 1 % beyond the body.
pub fn bar(i: usize, open: f64, close: f64) -> Bar {
    Bar {
        open_time: T0 + i as i64 * HOUR,
        open,
        high: open.max(close) * 1.01,
        low: open.min(close) * 0.99,
        close,
        volume: 1.0,
    }
}

/// Bars from (open, close) pairs.
pub fn bars(oc: &[(f64, f64)]) -> Vec<Bar> {
    oc.iter()
        .enumerate()
        .map(|(i, &(o, c))| bar(i, o, c))
        .collect()
}

/// `n` bars at a constant price: open = high = low = close.
pub fn flat_bars(n: usize, price: f64) -> Vec<Bar> {
    (0..n)
        .map(|i| Bar {
            open_time: T0 + i as i64 * HOUR,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: 1.0,
        })
        .collect()
}

/// Zero fees, zero slippage, no filters, no threshold.
pub fn frictionless(cash: f64) -> SimConfig {
    SimConfig {
        slippage_bps: 0.0,
        ..SimConfig::new(cash, FeeSchedule::ZERO)
    }
}

/// Largest `n` with `n·step ≤ q` in `f64`, written independently of the
/// simulator from its specification.
pub fn floor_lots(q: f64, step: f64) -> u64 {
    if q.is_nan() || q <= 0.0 {
        return 0;
    }
    let mut n = ((q / step).floor() as u64).min(1 << 50);
    while n < (1 << 50) && (n + 1) as f64 * step <= q {
        n += 1;
    }
    while n > 0 && n as f64 * step > q {
        n -= 1;
    }
    n
}

pub fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

pub fn bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// A fill as raw bits, so comparisons are bitwise rather than `==` on f64.
pub fn fill_bits(f: &Fill) -> (usize, i64, backtest::Side, [u64; 4]) {
    (
        f.bar,
        f.time,
        f.side,
        [
            f.qty.to_bits(),
            f.price.to_bits(),
            f.notional.to_bits(),
            f.fee.to_bits(),
        ],
    )
}

/// Random walk of `n` bars: price in roughly [1, 1e5], ±10 % bodies, ±1 %
/// gaps between a close and the next open, up to 5 % wicks, volumes that are
/// sometimes exactly 0. Most bars are one hour apart; one in four is a gap of
/// 1 ms to 10 h, as on the real series (gaps, off-grid bars).
pub fn arb_bars(n: usize) -> impl Strategy<Value = Vec<Bar>> {
    arb_bars_from(n, 10.0f64..10_000.0)
}

/// Like [`arb_bars`], with a first price drawn log-uniformly in [1e-6, 1e8].
pub fn arb_bars_wide(n: usize) -> impl Strategy<Value = Vec<Bar>> {
    arb_bars_from(n, (-6.0f64..8.0).prop_map(|e| 10f64.powf(e)))
}

fn arb_bars_from(n: usize, p0: impl Strategy<Value = f64>) -> impl Strategy<Value = Vec<Bar>> {
    (
        p0,
        prop::collection::vec(
            (
                -0.01f64..0.01,
                -0.1f64..0.1,
                0.0f64..0.05,
                0.0f64..0.05,
                prop_oneof![1 => Just(0.0), 4 => 0.0f64..1e6],
                prop_oneof![3 => Just(HOUR), 1 => 1i64..=10 * HOUR],
            ),
            n,
        ),
    )
        .prop_map(|(p0, steps)| {
            let mut prev = p0;
            let mut time = T0;
            steps
                .iter()
                .map(|&(gap, body, up, down, volume, dt)| {
                    let open = prev * gap.exp();
                    let close = open * body.exp();
                    prev = close;
                    time += dt;
                    Bar {
                        open_time: time,
                        open,
                        high: open.max(close) * (1.0 + up),
                        low: open.min(close) * (1.0 - down),
                        close,
                        volume,
                    }
                })
                .collect()
        })
}

/// `a[..=k]` followed by `b[k + 1..]`, with `b`'s times shifted so that they
/// keep their own spacing and start after `a[k]`. Everything after bar `k`
/// (prices, volumes and times) comes from `b`.
pub fn splice(a: &[Bar], b: &[Bar], k: usize) -> Vec<Bar> {
    let mut out = a[..=k].to_vec();
    let shift = a[k].open_time - b[k].open_time;
    out.extend(b[k + 1..].iter().map(|x| Bar {
        open_time: x.open_time + shift,
        ..*x
    }));
    out
}

/// Targets mixing hard 0, hard 1 and arbitrary fractions.
pub fn arb_targets(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(prop_oneof![Just(0.0), Just(1.0), 0.0f64..=1.0], n)
}

pub fn arb_filters() -> impl Strategy<Value = SymbolFilters> {
    prop_oneof![
        Just(SymbolFilters::NONE),
        Just(SymbolFilters::BINANCE_BTCUSDT),
        Just(SymbolFilters::BINANCE_ETHUSDT),
        (
            0.0f64..50.0,
            prop_oneof![
                Just(0.0),
                Just(0.000_01),
                Just(0.001),
                Just(0.01),
                Just(0.1)
            ]
        )
            .prop_map(|(min_notional, step_size)| SymbolFilters {
                min_notional,
                step_size
            }),
    ]
}

/// Random fees (up to 1 %), slippage (up to 50 bps), liquidity, filters and
/// threshold.
pub fn arb_cfg() -> impl Strategy<Value = SimConfig> {
    (
        100.0f64..1e6,
        0.0f64..0.01,
        0.0f64..0.01,
        any::<bool>(),
        0.0f64..50.0,
        arb_filters(),
        prop_oneof![Just(0.0), 0.0f64..0.2],
    )
        .prop_map(
            |(initial_cash, maker, taker, is_maker, slippage_bps, filters, rebalance_threshold)| {
                SimConfig {
                    initial_cash,
                    fees: FeeSchedule {
                        name: "random",
                        maker,
                        taker,
                    },
                    liquidity: if is_maker {
                        Liquidity::Maker
                    } else {
                        Liquidity::Taker
                    },
                    slippage_bps,
                    filters,
                    rebalance_threshold,
                }
            },
        )
}

/// The whole valid config domain, edges included: fee rates up to 0.999,
/// slippage up to 9 999 bps, odd lot steps from 1e-12 to 1e3, minimum
/// notionals up to 1e4, cash from 1e-6 to 1e12, thresholds up to 0.999.
pub fn arb_cfg_wide() -> impl Strategy<Value = SimConfig> {
    (
        (-6.0f64..12.0).prop_map(|e| 10f64.powf(e)),
        prop_oneof![Just(0.0), 0.0f64..0.999],
        prop_oneof![Just(0.0), 0.0f64..0.999],
        any::<bool>(),
        prop_oneof![Just(0.0), 0.0f64..9_999.0],
        (
            prop_oneof![Just(0.0), 0.0f64..1e4],
            prop_oneof![
                Just(0.0),
                Just(1e-12),
                Just(1e-8),
                Just(0.3),
                Just(7.0),
                Just(1e3),
                (-12.0f64..3.0).prop_map(|e| 10f64.powf(e)),
            ],
        )
            .prop_map(|(min_notional, step_size)| SymbolFilters {
                min_notional,
                step_size,
            }),
        prop_oneof![Just(0.0), 0.0f64..0.999],
    )
        .prop_map(
            |(initial_cash, maker, taker, is_maker, slippage_bps, filters, rebalance_threshold)| {
                SimConfig {
                    initial_cash,
                    fees: FeeSchedule {
                        name: "random-wide",
                        maker,
                        taker,
                    },
                    liquidity: if is_maker {
                        Liquidity::Maker
                    } else {
                        Liquidity::Taker
                    },
                    slippage_bps,
                    filters,
                    rebalance_threshold,
                }
            },
        )
}
