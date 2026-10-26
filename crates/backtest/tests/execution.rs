//! Hand-checked tests of the execution model.

mod common;

use backtest::{FeeSchedule, Liquidity, Side, SimConfig, SymbolFilters, simulate};
use common::*;

fn prices() -> Vec<backtest::Bar> {
    bars(&[
        (100.0, 101.0),
        (102.0, 99.0),
        (98.5, 104.0),
        (105.0, 110.0),
        (109.0, 95.0),
        (96.0, 97.5),
        (97.0, 120.0),
        (121.0, 118.0),
    ])
}

#[test]
fn flat_targets_keep_equity_constant() {
    let b = prices();
    let mut cfg = SimConfig::new(10_000.0, FeeSchedule::KRAKEN_PRO_ENTRY);
    cfg.filters = SymbolFilters::BINANCE_BTCUSDT;
    let r = simulate(&b, &vec![0.0; b.len()], &cfg).unwrap();
    assert!(r.fills.is_empty());
    assert_eq!(r.fees_paid, 0.0);
    assert!(r.equity.iter().all(|&e| e == 10_000.0));
    assert!(r.cash.iter().all(|&c| c == 10_000.0));
    assert!(r.position.iter().all(|&p| p == 0.0));
}

#[test]
fn buy_and_hold_matches_closed_form() {
    let b = prices();
    let cash = 10_000.0;
    for fees in [
        FeeSchedule::ZERO,
        FeeSchedule::BINANCE_SPOT,
        FeeSchedule::BINANCE_SPOT_BNB,
        FeeSchedule::COINBASE_ADVANCED_ENTRY,
        FeeSchedule::KRAKEN_PRO_ENTRY,
    ] {
        let cfg = SimConfig {
            slippage_bps: 0.0,
            ..SimConfig::new(cash, fees)
        };
        let r = simulate(&b, &vec![1.0; b.len()], &cfg).unwrap();
        let last = b.len() - 1;
        let expect = cash * b[last].close / (b[1].open * (1.0 + fees.taker));
        assert!(rel_err(r.equity[last], expect) < 1e-12, "{}", fees.name);
        assert_eq!(r.equity[0], cash);
        assert_eq!(r.fills.len(), 1, "{}", fees.name);
        assert_eq!(r.fills[0].bar, 1);
        assert_eq!(r.fills[0].side, Side::Buy);
        assert!(r.cash.iter().all(|&c| c >= 0.0));
    }
}

#[test]
fn round_trip_at_constant_price_costs_both_sides() {
    let b = flat_bars(5, 250.0);
    let cash = 1_000.0;
    let (s, f) = (2.0e-4, 0.005);
    let cfg = SimConfig::new(cash, FeeSchedule::COINBASE_ADVANCED_ENTRY);
    let r = simulate(&b, &[1.0, 0.0, 0.0, 0.0, 0.0], &cfg).unwrap();
    let expect = cash * (1.0 - s) * (1.0 - f) / ((1.0 + s) * (1.0 + f));
    assert!(rel_err(r.equity[4], expect) < 1e-12);
    assert_eq!(r.fills.len(), 2);
    assert_eq!((r.fills[0].bar, r.fills[0].side), (1, Side::Buy));
    assert_eq!((r.fills[1].bar, r.fills[1].side), (2, Side::Sell));
    assert_eq!(r.position[2], 0.0);
    assert_eq!(r.fills[0].qty, r.fills[1].qty);
}

#[test]
fn decisions_fill_at_the_next_open_with_slippage() {
    let b = prices();
    let cfg = SimConfig::new(5_000.0, FeeSchedule::BINANCE_SPOT);
    let s = cfg.slippage_bps / 1e4;
    let mut targets = vec![0.0; b.len()];
    targets[2] = 1.0;
    targets[3] = 1.0;
    targets[4] = 1.0;
    // targets[5] = 0: the exit is decided at close 5 and fills at open 6.
    let r = simulate(&b, &targets, &cfg).unwrap();
    assert_eq!(r.fills.len(), 2);
    let (buy, sell) = (r.fills[0], r.fills[1]);
    assert_eq!(
        (buy.bar, buy.side, buy.time),
        (3, Side::Buy, b[3].open_time)
    );
    assert_eq!(buy.price, b[3].open * (1.0 + s));
    assert_eq!(
        (sell.bar, sell.side, sell.time),
        (6, Side::Sell, b[6].open_time)
    );
    assert_eq!(sell.price, b[6].open * (1.0 - s));
    // Nothing is held before the fill bar.
    assert!(r.position[..3].iter().all(|&p| p == 0.0));
    assert!(r.position[3] > 0.0);

    // The fill bar's high, low and close play no part in its fill.
    let mut edited = b.clone();
    edited[3].close = 1.0;
    edited[3].low = 0.5;
    edited[3].high = 10_000.0;
    let r2 = simulate(&edited, &targets, &cfg).unwrap();
    assert_eq!(fill_bits(&r2.fills[0]), fill_bits(&buy));
    assert_eq!(r2.cash[3], r.cash[3]);
    assert_eq!(r2.position[3], r.position[3]);
    assert_ne!(r2.equity[3], r.equity[3]);
}

#[test]
fn last_target_is_never_executed() {
    let b = prices();
    let cfg = SimConfig::new(5_000.0, FeeSchedule::BINANCE_SPOT);
    let mut targets = vec![0.0; b.len()];
    *targets.last_mut().unwrap() = 1.0;
    let r = simulate(&b, &targets, &cfg).unwrap();
    assert!(r.fills.is_empty());

    let mut a = vec![0.3; b.len()];
    let ra = simulate(&b, &a, &cfg).unwrap();
    *a.last_mut().unwrap() = 0.9;
    let rb = simulate(&b, &a, &cfg).unwrap();
    assert_eq!(ra, rb);
}

#[test]
fn fees_are_counted_once_and_the_ledger_reconciles() {
    let b = prices();
    let mut cfg = SimConfig::new(7_777.0, FeeSchedule::KRAKEN_PRO_ENTRY);
    cfg.filters = SymbolFilters::BINANCE_ETHUSDT;
    let targets = [0.5, 1.0, 0.2, 0.0, 0.7, 0.7, 0.1, 1.0];
    let r = simulate(&b, &targets, &cfg).unwrap();
    assert!(r.fills.len() >= 5);
    let mut sum = 0.0;
    let mut cash = cfg.initial_cash;
    for f in &r.fills {
        assert_eq!(f.notional, f.qty * f.price);
        assert_eq!(f.fee, f.notional * 0.008);
        sum += f.fee;
        match f.side {
            Side::Buy => cash -= f.notional + f.fee,
            Side::Sell => cash += f.notional - f.fee,
        }
        assert_eq!(cash, r.cash[f.bar]);
    }
    assert_eq!(r.fees_paid, sum);
    assert_eq!(*r.cash.last().unwrap(), cash);
}

#[test]
fn maker_liquidity_uses_the_maker_rate() {
    let b = prices();
    let mut cfg = SimConfig::new(1_000.0, FeeSchedule::COINBASE_ADVANCED_ENTRY);
    cfg.liquidity = Liquidity::Maker;
    let r = simulate(&b, &vec![1.0; b.len()], &cfg).unwrap();
    assert_eq!(r.fills[0].fee, r.fills[0].notional * 0.0025);
    assert_eq!(
        FeeSchedule::COINBASE_ADVANCED_ENTRY.rate(Liquidity::Taker),
        0.005
    );
}

#[test]
fn min_notional_skips_small_orders() {
    let b = prices();
    let mut cfg = SimConfig::new(4.0, FeeSchedule::BINANCE_SPOT);
    cfg.filters = SymbolFilters::BINANCE_BTCUSDT;
    let r = simulate(&b, &vec![1.0; b.len()], &cfg).unwrap();
    assert!(r.fills.is_empty(), "4 USDT is below the 5 USDT minimum");

    // A rebalance from 1.0 to 0.999 sells about 1 USDT: skipped. The full
    // exit that follows is large enough and goes through.
    let mut cfg = SimConfig::new(1_000.0, FeeSchedule::BINANCE_SPOT);
    cfg.filters = SymbolFilters {
        min_notional: 5.0,
        step_size: 0.0,
    };
    let b = flat_bars(5, 100.0);
    let r = simulate(&b, &[1.0, 0.999, 0.0, 0.0, 0.0], &cfg).unwrap();
    assert_eq!(r.fills.len(), 2);
    assert_eq!((r.fills[1].bar, r.fills[1].side), (3, Side::Sell));
    assert!(r.fills.iter().all(|f| f.notional >= 5.0));
}

#[test]
fn fill_quantities_are_step_multiples_and_cash_stays_nonnegative() {
    let b = prices();
    for filters in [
        SymbolFilters::BINANCE_BTCUSDT,
        SymbolFilters::BINANCE_ETHUSDT,
    ] {
        let mut cfg = SimConfig::new(3_333.33, FeeSchedule::BINANCE_SPOT);
        cfg.filters = filters;
        let targets = [1.0, 0.37, 0.81, 0.0, 1.0, 0.5, 0.05, 1.0];
        let r = simulate(&b, &targets, &cfg).unwrap();
        assert!(!r.fills.is_empty());
        for f in &r.fills {
            let n = (f.qty / filters.step_size).round();
            assert_eq!(f.qty, n * filters.step_size, "qty {}", f.qty);
            assert!(n >= 1.0);
        }
        assert!(r.cash.iter().all(|&c| c >= 0.0));
        // The cash left after a cash-capped buy is less than one more lot.
        let first = r.fills[0];
        let lot_cost = filters.step_size * first.price * 1.001;
        assert!(r.cash[1] < lot_cost);
    }
}

#[test]
fn full_exit_with_a_lot_step_leaves_no_dust() {
    let b = prices();
    let mut cfg = SimConfig::new(9_876.54, FeeSchedule::BINANCE_SPOT);
    cfg.filters = SymbolFilters::BINANCE_BTCUSDT;
    let targets = [1.0, 0.63, 0.29, 0.0, 0.0, 0.0, 0.0, 0.0];
    let r = simulate(&b, &targets, &cfg).unwrap();
    assert_eq!(r.position[4], 0.0);
}

#[test]
fn threshold_suppresses_small_rebalances_but_not_exits() {
    let b = flat_bars(6, 100.0);
    let mut cfg = frictionless(1_000.0);
    cfg.rebalance_threshold = 0.1;
    let r = simulate(&b, &[0.5, 0.55, 0.45, 0.7, 0.0, 0.0], &cfg).unwrap();
    let sides: Vec<_> = r.fills.iter().map(|f| (f.bar, f.side)).collect();
    // 0.5 → 0.55 and → 0.45 are within 0.1; 0.7 is not; 0 always exits.
    assert_eq!(sides, [(1, Side::Buy), (4, Side::Buy), (5, Side::Sell)]);
    assert_eq!(r.position[5], 0.0);
}

/// The threshold is strict: a gap exactly equal to it does not trade. All
/// values here are exact in binary, so the boundary is hit exactly.
#[test]
fn threshold_boundary_is_strict() {
    let b = flat_bars(5, 100.0);
    let mut cfg = frictionless(1_000.0);
    cfg.rebalance_threshold = 0.25;
    let r = simulate(&b, &[0.5, 0.75, 0.25, 0.8125, 0.0], &cfg).unwrap();
    assert_eq!(r.position[1], 5.0);
    let sides: Vec<_> = r.fills.iter().map(|f| (f.bar, f.side)).collect();
    // |0.75 − 0.5| = |0.25 − 0.5| = 0.25 exactly: no trade; 0.8125 trades.
    assert_eq!(sides, [(1, Side::Buy), (4, Side::Buy)]);
    assert_eq!(r.position[4], 8.125);
}

/// A partial sell is sized against the open, `|Δ| / open`, not against the
/// slipped execution price; a buy is sized against the execution price.
#[test]
fn partial_sell_sizes_against_the_open_with_slippage() {
    let b = flat_bars(3, 200.0);
    let mut cfg = frictionless(1_000.0);
    cfg.slippage_bps = 50.0;
    let r = simulate(&b, &[1.0, 0.25, 0.25], &cfg).unwrap();
    let (buy, sell) = (r.fills[0], r.fills[1]);
    assert_eq!(buy.qty, 1_000.0 / (200.0 * (1.0 + 0.005)));
    let held = r.position[1] * 200.0;
    let delta = 0.25 * (r.cash[1] + held) - held;
    assert_eq!(sell.qty, -delta / 200.0);
    assert_eq!(sell.price, 200.0 * (1.0 - 0.005));
}

#[test]
fn partial_sell_sizes_against_the_open() {
    let b = flat_bars(4, 200.0);
    let cfg = frictionless(1_000.0);
    let r = simulate(&b, &[1.0, 0.25, 0.25, 0.25], &cfg).unwrap();
    assert_eq!(r.fills.len(), 2);
    let sell = r.fills[1];
    assert_eq!(sell.side, Side::Sell);
    assert!(rel_err(sell.qty, 3.75) < 1e-12);
    assert!(rel_err(r.position[3], 1.25) < 1e-12);
    assert!(rel_err(r.equity[3], 1_000.0) < 1e-12);
}

#[test]
fn defaults_and_constants() {
    let cfg = SimConfig::new(1.0, FeeSchedule::BINANCE_SPOT);
    assert_eq!(cfg.liquidity, Liquidity::Taker);
    assert_eq!(cfg.slippage_bps, 2.0);
    assert_eq!(cfg.filters, SymbolFilters::NONE);
    assert_eq!(cfg.rebalance_threshold, 0.0);
    assert_eq!(
        FeeSchedule::BINANCE_SPOT_BNB.rate(Liquidity::Maker),
        0.00075
    );
    assert_eq!(FeeSchedule::KRAKEN_PRO_ENTRY.rate(Liquidity::Taker), 0.008);
    assert_eq!(SymbolFilters::BINANCE_ETHUSDT.step_size, 0.0001);
}

#[test]
fn single_bar_runs_without_trading() {
    let b = flat_bars(1, 10.0);
    let r = simulate(&b, &[1.0], &SimConfig::new(50.0, FeeSchedule::ZERO)).unwrap();
    assert_eq!(r.equity, [50.0]);
    assert!(r.fills.is_empty());
}

/// Deterministic xorshift64*, so the sweeps below need no extra dependency
/// and are reproducible.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + self.unit() * (hi.ln() - lo.ln())).exp()
    }
}

/// `(notional, fee, total)` of `qty` at `price`, in the simulator's order.
fn cost(qty: f64, price: f64, rate: f64) -> (f64, f64, f64) {
    let notional = qty * price;
    let fee = notional * rate;
    (notional, fee, notional + fee)
}

/// Solvency by construction, not by luck, without a lot step. A cash-capped
/// buy of `qty = cash / (price·(1 + fee))` costs more than the cash, as
/// computed in `f64`, in a sizeable share of cases. The sweep first proves
/// that share is not zero, then that the simulator never overspends there and
/// loses at most a few ulps of quantity doing so.
#[test]
fn cash_capped_buys_never_overspend_where_naive_rounding_would() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut naive_overspends = 0;
    let cases = 20_000;
    for _ in 0..cases {
        let cash = rng.log_uniform(1e-3, 1e9);
        let open = rng.log_uniform(1e-4, 1e6);
        let rate = rng.unit() * 0.01;
        let slippage_bps = rng.unit() * 50.0;
        let price = open * (1.0 + slippage_bps / 1e4);
        let naive = cash / (price * (1.0 + rate));
        let over = cost(naive, price, rate).2 > cash;
        naive_overspends += usize::from(over);

        let cfg = SimConfig {
            slippage_bps,
            ..SimConfig::new(
                cash,
                FeeSchedule {
                    name: "sweep",
                    maker: rate,
                    taker: rate,
                },
            )
        };
        let b = flat_bars(2, open);
        let r = simulate(&b, &[1.0, 1.0], &cfg).unwrap();
        assert_eq!(r.fills.len(), 1);
        let f = r.fills[0];
        assert!(
            f.notional + f.fee <= cash,
            "overspent: {cash} {open} {rate}"
        );
        assert!(r.cash[1] >= 0.0);
        assert!(f.qty <= naive && f.qty >= naive * (1.0 - 1e-14));
        if !over {
            assert_eq!(f.qty, naive, "no shrink is needed when the naive buy fits");
        }
    }
    eprintln!("{naive_overspends} of {cases} naive buys overspend");
    assert!(
        naive_overspends > cases / 100,
        "only {naive_overspends} of {cases} naive buys overspend: the sweep is vacuous"
    );
}

/// The same with a lot step, on the hardest inputs: the cash is one ulp below
/// the exact cost of `n` lots, so the cash cap lands on `n` lots in floating
/// point about as often as on `n − 1`. Where flooring alone would buy `n`
/// lots and overspend, the simulator buys exactly `n − 1`.
#[test]
fn lot_buys_shrink_by_one_lot_where_flooring_alone_would_overspend() {
    let mut rng = Rng(0xd1b5_4a32_d192_ed03);
    let mut shrunk = 0;
    let cases = 20_000;
    for _ in 0..cases {
        let step = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0][(rng.unit() * 6.0) as usize];
        let n = 2 + (rng.log_uniform(1.0, 1e9) as u64);
        let open = rng.log_uniform(1e-2, 1e5);
        let rate = [0.0, 0.00075, 0.001, 0.005, 0.008][(rng.unit() * 5.0) as usize];
        let exact = cost(n as f64 * step, open, rate).2;
        let cash = exact.next_down();

        // What flooring the cash cap would buy, before the shrink.
        let cap = cash / (open * (1.0 + rate));
        let mut n0 = (cap / step).floor() as u64;
        if ((n0 + 1) as f64 * step) <= cap {
            n0 += 1;
        }
        while n0 as f64 * step > cap {
            n0 -= 1;
        }
        let flooring_overspends = cost(n0 as f64 * step, open, rate).2 > cash;

        let mut cfg = SimConfig::new(
            cash,
            FeeSchedule {
                name: "sweep",
                maker: rate,
                taker: rate,
            },
        );
        cfg.slippage_bps = 0.0;
        cfg.filters = SymbolFilters {
            min_notional: 0.0,
            step_size: step,
        };
        let r = simulate(&flat_bars(2, open), &[1.0, 1.0], &cfg).unwrap();
        let f = r.fills[0];
        assert!(f.notional + f.fee <= cash);
        assert!(r.cash[1] >= 0.0);
        let lots = (f.qty / step).round() as u64;
        if flooring_overspends {
            shrunk += 1;
            assert_eq!(lots, n0 - 1, "shrinks by exactly one lot");
        } else {
            assert_eq!(lots, n0, "no shrink when the floored buy fits");
        }
    }
    eprintln!("{shrunk} of {cases} floored buys overspend");
    assert!(
        shrunk > cases / 100,
        "only {shrunk} of {cases} cases need a shrink: the sweep is vacuous"
    );
}

/// Metrics read on simulator output. Buy and hold over n bars is exposed on
/// n − 1 of them; switching the whole equity in or out at every open of a
/// frictionless constant-price market trades the equity once per bar, so its
/// annualized turnover is exactly `ppy`, up to rounding.
#[test]
fn exposure_and_turnover_on_simulated_runs() {
    use backtest::metrics::{compute, periods_per_year};
    let ppy = periods_per_year(HOUR);
    let b = prices();
    let r = simulate(&b, &vec![1.0; b.len()], &frictionless(1_000.0)).unwrap();
    let m = compute(&r, ppy);
    assert_eq!(m.exposure, (b.len() - 1) as f64 / b.len() as f64);
    let mean = r.equity.iter().sum::<f64>() / b.len() as f64;
    let expect = r.fills[0].notional / mean * ppy / (b.len() - 1) as f64;
    assert!(rel_err(m.turnover, expect) < 1e-12);

    let n = 101;
    let flips: Vec<f64> = (0..n).map(|t| if t % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let r = simulate(&flat_bars(n, 37.0), &flips, &frictionless(1_000.0)).unwrap();
    assert_eq!(r.fills.len(), n - 1);
    let m = compute(&r, ppy);
    assert!(rel_err(m.turnover, ppy) < 1e-12, "{}", m.turnover);
    assert_eq!(m.exposure, 50.0 / 101.0);
    assert_eq!(m.fees_paid, 0.0);
}

/// Only the open of the fill bar sets the fill: changing it moves the price,
/// while the previous bar's close (the decision bar) does not enter it.
#[test]
fn the_fill_follows_the_open_and_nothing_else() {
    let b = prices();
    let cfg = SimConfig::new(5_000.0, FeeSchedule::BINANCE_SPOT);
    let s = cfg.slippage_bps / 1e4;
    let mut targets = vec![0.0; b.len()];
    targets[2] = 1.0;
    let base = simulate(&b, &targets, &cfg).unwrap();

    let mut moved_open = b.clone();
    moved_open[3].open = 100.0;
    moved_open[3].low = 99.0;
    let r = simulate(&moved_open, &targets, &cfg).unwrap();
    assert_eq!(r.fills[0].price, 100.0 * (1.0 + s));
    assert_ne!(r.fills[0].price, base.fills[0].price);

    let mut moved_close = b.clone();
    moved_close[2].close = 103.0;
    moved_close[2].high = 104.0;
    let r = simulate(&moved_close, &targets, &cfg).unwrap();
    assert_eq!(fill_bits(&r.fills[0]), fill_bits(&base.fills[0]));
}

/// A rebalance of `fraction` of equity at bar 2, up (a buy) or down (a sell),
/// from half the equity bought at bar 1. Frictionless, filter-free and at a
/// constant price, so equity is exactly 1 000 at every open and the dust floor
/// is the only rule that can skip the order. Returns the fills at bar 2.
fn rebalance_by(fraction: f64, side: Side) -> Vec<backtest::Fill> {
    let target = match side {
        Side::Buy => 0.5 + fraction,
        Side::Sell => 0.5 - fraction,
    };
    let r = simulate(
        &flat_bars(3, 100.0),
        &[0.5, target, target],
        &frictionless(1_000.0),
    )
    .unwrap();
    assert_eq!(r.fills[0].bar, 1);
    assert_eq!(r.position[1], 5.0);
    r.fills.into_iter().filter(|f| f.bar == 2).collect()
}

/// The dust floor, pinned from above: rebalances worth 10⁻⁶ down to 2·10⁻¹²
/// of equity are real orders and fill, at the size asked. With the next test,
/// this pins the floor inside [5·10⁻¹³, 2·10⁻¹²) of equity: any floor outside
/// that interval fails one of the two.
#[test]
fn rebalances_above_the_dust_floor_fill() {
    for fraction in [1e-6, 1e-9, 1e-11, 2e-12] {
        for side in [Side::Buy, Side::Sell] {
            let fills = rebalance_by(fraction, side);
            assert_eq!(fills.len(), 1, "{fraction:e} {side:?} must fill");
            assert_eq!(fills[0].side, side);
            // `target·E − held` carries at most 2⁻⁵⁴·10³ + ulp(500)/2 ≈ 8·10⁻¹⁴
            // of absolute rounding: 4·10⁻⁵ of the smallest order tried.
            let expect = fraction * 1_000.0;
            assert!(
                rel_err(fills[0].notional, expect) < 1e-4,
                "{fraction:e} {side:?}: notional {} vs {expect}",
                fills[0].notional
            );
        }
    }
}

/// The dust floor, pinned from below: rebalances worth 5·10⁻¹³ of equity or
/// less are rounding noise and are skipped, both ways.
#[test]
fn rebalances_below_the_dust_floor_are_skipped() {
    for fraction in [5e-13, 1e-13, 1e-14] {
        for side in [Side::Buy, Side::Sell] {
            assert!(
                rebalance_by(fraction, side).is_empty(),
                "{fraction:e} {side:?} is dust"
            );
        }
    }
}

/// A full exit is never dust, whatever its size. Half the equity is bought at
/// 100, then the price collapses so that the position is worth 10⁻ᵏ of equity,
/// down to 10⁻²⁹⁶, far below the dust floor; a target of 0 still sells all of
/// it, with or without a lot step.
#[test]
fn a_full_exit_is_never_skipped_as_dust() {
    for step in [0.0, 0.001] {
        for k in [6, 12, 13, 20, 50, 100, 200, 296] {
            let crash = 100.0 * 10f64.powi(-k);
            let b = bars(&[(100.0, 100.0), (100.0, crash), (crash, crash)]);
            let mut cfg = frictionless(1_000.0);
            cfg.filters.step_size = step;
            let r = simulate(&b, &[0.5, 0.0, 0.0], &cfg).unwrap();
            assert_eq!(r.fills.len(), 2, "step {step}, 1e-{k}");
            let exit = r.fills[1];
            assert_eq!((exit.bar, exit.side, exit.qty), (2, Side::Sell, 5.0));
            if k >= 13 {
                assert!(exit.notional <= 1e-12 * r.cash[1], "below the floor");
            }
            assert_eq!(r.position[2], 0.0, "step {step}, 1e-{k}");
        }
    }
}
