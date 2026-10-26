//! Property tests: causality, solvency, conservation laws, identities and
//! exact invariances.
//!
//! Comparisons are bitwise wherever the simulator's own arithmetic can be
//! replayed in the same order. A tolerance appears only where a law is
//! evaluated in a different order than the simulator does, and each one is
//! justified next to it.

mod common;

use backtest::{Bar, FeeSchedule, Side, SimConfig, SimResult, simulate};
use common::*;
use proptest::prelude::*;

/// Unit roundoff of f64, u = ε / 2 = 2⁻⁵³.
const U: f64 = f64::EPSILON / 2.0;

/// Bars from the moderate or the wide price domain.
fn arb_any_bars(n: usize) -> impl Strategy<Value = Vec<Bar>> {
    prop_oneof![arb_bars(n), arb_bars_wide(n)]
}

/// A config from the bench's realistic domain or from the whole valid domain.
fn arb_any_cfg() -> impl Strategy<Value = SimConfig> {
    prop_oneof![arb_cfg(), arb_cfg_wide()]
}

/// A run of length 2..=64 with its config, in the realistic domain.
fn arb_run() -> impl Strategy<Value = (Vec<Bar>, Vec<f64>, SimConfig)> {
    (2usize..=64).prop_flat_map(|n| (arb_bars(n), arb_targets(n), arb_cfg()))
}

/// A run of length 1..=64 anywhere in the valid input domain.
fn arb_run_any() -> impl Strategy<Value = (Vec<Bar>, Vec<f64>, SimConfig)> {
    (1usize..=64).prop_flat_map(|n| (arb_any_bars(n), arb_targets(n), arb_any_cfg()))
}

fn scale_bars(bars: &[Bar], k: f64) -> Vec<Bar> {
    bars.iter()
        .map(|b| Bar {
            open: b.open * k,
            high: b.high * k,
            low: b.low * k,
            close: b.close * k,
            ..*b
        })
        .collect()
}

fn past_fills(r: &SimResult, k: usize) -> Vec<(usize, i64, Side, [u64; 4])> {
    r.fills
        .iter()
        .filter(|f| f.bar <= k)
        .map(fill_bits)
        .collect()
}

/// Lot count of a fill: exact because the simulator caps the book at 2⁵⁰
/// lots (see `MAX_LOTS`); the round trip back to `qty` is asserted bitwise.
fn lots_of(qty: f64, step: f64) -> Result<u64, TestCaseError> {
    let n = (qty / step).round();
    prop_assert!(
        (1.0..=2f64.powi(50)).contains(&n),
        "lots {} of qty {}",
        n,
        qty
    );
    prop_assert_eq!(
        (n * step).to_bits(),
        qty.to_bits(),
        "qty {} step {}",
        qty,
        step
    );
    Ok(n as u64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// Replacing every bar after k (prices, volumes and times) and every
    /// target from k on leaves every output up to bar k unchanged, bit for
    /// bit. `targets[k]` executes at the open of bar k + 1, so it may not
    /// influence bar k either.
    #[test]
    fn future_invariance(
        (bars_a, bars_b, targets_a, targets_b, k, cfg) in (2usize..=64).prop_flat_map(|n| (
            arb_any_bars(n), arb_any_bars(n), arb_targets(n), arb_targets(n), 0..n, arb_any_cfg(),
        ))
    ) {
        let bars_p = splice(&bars_a, &bars_b, k);
        let mut targets_p = targets_a[..k].to_vec();
        targets_p.extend_from_slice(&targets_b[k..]);
        // The perturbed inputs really share the past and only the past.
        prop_assert_eq!(&bars_p[..=k], &bars_a[..=k]);
        prop_assert_eq!(bits(&targets_p[..k]), bits(&targets_a[..k]));

        let a = simulate(&bars_a, &targets_a, &cfg).unwrap();
        let p = simulate(&bars_p, &targets_p, &cfg).unwrap();
        prop_assert_eq!(bits(&a.equity[..=k]), bits(&p.equity[..=k]));
        prop_assert_eq!(bits(&a.position[..=k]), bits(&p.position[..=k]));
        prop_assert_eq!(bits(&a.cash[..=k]), bits(&p.cash[..=k]));
        prop_assert_eq!(past_fills(&a, k), past_fills(&p, k));
    }

    /// A run cut after bar k is the prefix of the full run, bit for bit: the
    /// simulator cannot even tell that the future exists.
    #[test]
    fn prefix_truncation((bars, targets, cfg, k) in (1usize..=64).prop_flat_map(|n| (
        arb_any_bars(n), arb_targets(n), arb_any_cfg(), 0..n,
    ))) {
        let full = simulate(&bars, &targets, &cfg).unwrap();
        let cut = simulate(&bars[..=k], &targets[..=k], &cfg).unwrap();
        prop_assert_eq!(bits(&cut.equity), bits(&full.equity[..=k]));
        prop_assert_eq!(bits(&cut.position), bits(&full.position[..=k]));
        prop_assert_eq!(bits(&cut.cash), bits(&full.cash[..=k]));
        prop_assert_eq!(past_fills(&cut, usize::MAX), past_fills(&full, k));
        let fees: f64 = full.fills.iter().filter(|f| f.bar <= k).fold(0.0, |s, f| s + f.fee);
        prop_assert_eq!(cut.fees_paid.to_bits(), fees.to_bits());
    }

    /// No leverage, no shorting, no negative equity, over the whole valid
    /// input domain: fee rates up to 0.999, slippage up to 9 999 bps, cash
    /// from 1e-6 to 1e12, prices from 1e-6 to 1e8, odd lot steps.
    #[test]
    fn solvency((bars, targets, cfg) in arb_run_any()) {
        let r = simulate(&bars, &targets, &cfg).unwrap();
        prop_assert_eq!(r.equity.len(), bars.len());
        prop_assert_eq!(r.position.len(), bars.len());
        prop_assert_eq!(r.cash.len(), bars.len());
        for t in 0..bars.len() {
            prop_assert!(r.cash[t] >= 0.0 && r.cash[t].is_finite(), "cash[{}] = {}", t, r.cash[t]);
            prop_assert!(r.position[t] >= 0.0 && r.position[t].is_finite(), "position[{}] = {}", t, r.position[t]);
            prop_assert!(r.equity[t] >= 0.0 && r.equity[t].is_finite(), "equity[{}] = {}", t, r.equity[t]);
        }
        prop_assert!(r.fees_paid >= 0.0);
    }

    /// Conservation laws, replayed from the fills alone at every bar:
    /// - cash = initial − Σ buy cost + Σ sell proceeds, in fill order: bitwise;
    /// - position = Σ buy qty − Σ sell qty: bitwise, in exact integer lots
    ///   with a step and in fill order without;
    /// - equity[t] = cash[t] + position[t]·close[t]: bitwise;
    /// - fees_paid = Σ fee in fill order: bitwise;
    /// - the same laws with the buy and sell sums accumulated separately,
    ///   within a summation-order bound.
    ///
    /// Every buy also costs at most the cash it is paid from, and every sell
    /// sells at most the position.
    #[test]
    fn conservation_laws((bars, targets, cfg) in arb_run_any()) {
        let r = simulate(&bars, &targets, &cfg).unwrap();
        let step = cfg.filters.step_size;
        let mut fills = r.fills.iter().peekable();

        // In-order replay of the simulator's own updates.
        let (mut cash, mut units, mut lots, mut fees) = (cfg.initial_cash, 0.0f64, 0u64, 0.0f64);
        // Separate sums, as an accountant would write them.
        let (mut bought, mut sold, mut qty_in, mut qty_out) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut m = 0usize;

        for (t, bar) in bars.iter().enumerate() {
            if let Some(f) = fills.next_if(|f| f.bar == t) {
                m += 1;
                let n = if step > 0.0 { lots_of(f.qty, step)? } else { 0 };
                match f.side {
                    Side::Buy => {
                        let cost = f.notional + f.fee;
                        prop_assert!(cost <= cash, "buy of {} with {} cash", cost, cash);
                        cash -= cost;
                        bought += cost;
                        units += f.qty;
                        lots += n;
                        qty_in += f.qty;
                    }
                    Side::Sell => {
                        let proceeds = f.notional - f.fee;
                        prop_assert!(proceeds >= 0.0);
                        if step > 0.0 {
                            prop_assert!(n <= lots, "sells {} of {} lots", n, lots);
                        } else {
                            prop_assert!(f.qty <= units, "sells {} of {}", f.qty, units);
                        }
                        cash += proceeds;
                        sold += proceeds;
                        units -= f.qty;
                        lots -= n;
                        qty_out += f.qty;
                    }
                }
                fees += f.fee;
            }
            let pos = if step > 0.0 { lots as f64 * step } else { units };
            prop_assert_eq!(cash.to_bits(), r.cash[t].to_bits(), "cash at bar {}", t);
            prop_assert_eq!(pos.to_bits(), r.position[t].to_bits(), "position at bar {}", t);
            prop_assert_eq!(
                (r.cash[t] + r.position[t] * bar.close).to_bits(),
                r.equity[t].to_bits(),
                "equity at bar {}", t
            );

            // Separate sums reorder the additions. Recursive summation of
            // m + 1 terms errs by at most m·u·Σ|terms| (Higham, Accuracy and
            // Stability, §4.2); the in-order replay and the separate sums
            // each obey this bound, and the final combination adds two more
            // roundings. Hence 2·(m + 2)·u·Σ|terms|.
            let closed_cash = cfg.initial_cash - bought + sold;
            let tol = 2.0 * (m + 2) as f64 * U * (cfg.initial_cash + bought + sold);
            prop_assert!((closed_cash - cash).abs() <= tol, "cash {} vs {} at {}", closed_cash, cash, t);
            // Same bound for quantities; with a step, the position is also
            // `lots·step` rounded once, which is one more term of the bound.
            let closed_pos = qty_in - qty_out;
            let tol = 2.0 * (m + 3) as f64 * U * (qty_in + qty_out);
            prop_assert!((closed_pos - pos).abs() <= tol, "position {} vs {} at {}", closed_pos, pos, t);
        }
        prop_assert!(fills.next().is_none(), "a fill at bar 0, out of order, or two in one bar");
        prop_assert_eq!(fees.to_bits(), r.fees_paid.to_bits());
    }

    /// PnL attribution, bar by bar. In exact arithmetic,
    ///
    /// `equity[t] − equity[t−1] = pos[t−1]·(open[t] − close[t−1])`
    /// `                        + pos[t]·(close[t] − open[t])`
    /// `                        − fee[t] − qty[t]·|price[t] − open[t]|`:
    ///
    /// the overnight move on the old position, the intrabar move on the new
    /// one, and the two costs of the fill at the open, each charged exactly
    /// once. A fee counted twice, slippage on the wrong side, a fill at any
    /// price other than this open, or any cash leak would break it.
    #[test]
    fn pnl_attribution((bars, targets, cfg) in arb_run_any()) {
        let r = simulate(&bars, &targets, &cfg).unwrap();
        for t in 1..bars.len() {
            let (o, c_prev, c) = (bars[t].open, bars[t - 1].close, bars[t].close);
            let (p_prev, p) = (r.position[t - 1], r.position[t]);
            let (cost, notional) = match r.fills.iter().find(|f| f.bar == t) {
                Some(f) => {
                    let slip = f.qty * (f.price - o).abs();
                    (f.fee + slip, f.notional + f.qty * o)
                }
                None => (0.0, 0.0),
            };
            let lhs = r.equity[t] - r.equity[t - 1];
            let rhs = p_prev * (o - c_prev) + p * (c - o) - cost;
            // Both sides are evaluated from rounded values. Every rounding on
            // either side, in the simulator (cash update, position update,
            // marking) or here, is at most u times one of the magnitudes in
            // `scale`; fewer than 20 roundings occur, so 32·u·scale bounds
            // their sum. It is absolute, so cancellation cannot hide behind it.
            let scale = r.equity[t - 1] + r.equity[t] + r.cash[t - 1] + r.cash[t]
                + p_prev * (o + c_prev) + p * (o + c) + notional + cost;
            prop_assert!(
                (lhs - rhs).abs() <= 32.0 * U * scale,
                "bar {}: Δequity {} vs attribution {} (scale {})", t, lhs, rhs, scale
            );
        }
    }

    /// Every fill obeys the execution model: it happens at the open of the
    /// bar after the decision, at `open·(1 + s)` for a buy and `open·(1 − s)`
    /// for a sell, with `notional = qty·price` and `fee = notional·rate`, all
    /// bitwise; it clears the minimum notional; its quantity is a whole number
    /// of lots. A target of 0 is flat by the next close unless the exit is
    /// below the minimum notional.
    #[test]
    fn fills_follow_the_model((bars, targets, cfg) in arb_run_any()) {
        let r = simulate(&bars, &targets, &cfg).unwrap();
        let rate = cfg.fees.rate(cfg.liquidity);
        let s = cfg.slippage_bps / 1e4;
        for f in &r.fills {
            prop_assert!(f.bar >= 1);
            prop_assert_eq!(f.time, bars[f.bar].open_time);
            let open = bars[f.bar].open;
            let expect_price = match f.side {
                Side::Buy => open * (1.0 + s),
                Side::Sell => open * (1.0 - s),
            };
            prop_assert_eq!(f.price.to_bits(), expect_price.to_bits());
            prop_assert!(f.qty > 0.0);
            prop_assert_eq!(f.notional.to_bits(), (f.qty * f.price).to_bits());
            prop_assert_eq!(f.fee.to_bits(), (f.notional * rate).to_bits());
            prop_assert!(f.notional >= cfg.filters.min_notional);
            if cfg.filters.step_size > 0.0 {
                lots_of(f.qty, cfg.filters.step_size)?;
            }
            // The side and the size are the ones the specification asks for.
            let (cash, pos) = (r.cash[f.bar - 1], r.position[f.bar - 1]);
            let held = pos * open;
            let equity = cash + held;
            let target = targets[f.bar - 1];
            let delta = target * equity - held;
            let step = cfg.filters.step_size;
            match f.side {
                Side::Buy => {
                    prop_assert!(target > 0.0 && delta > 0.0);
                    let raw = (delta / f.price).min(cash / (f.price * (1.0 + rate)));
                    if step > 0.0 {
                        // Floored to the lot and clamped to the 2⁵⁰-lot book,
                        // minus one lot if that overspends.
                        let held_lots = (pos / step).round() as u64;
                        let n0 = floor_lots(raw, step).min((1 << 50) - held_lots);
                        let n = lots_of(f.qty, step)?;
                        prop_assert!(n == n0 || n + 1 == n0, "{} lots, floor {}", n, n0);
                    } else {
                        // Exactly the raw size, or a few ulps less to fit.
                        prop_assert!(f.qty <= raw && f.qty >= raw * (1.0 - 1e-13));
                    }
                }
                Side::Sell => {
                    prop_assert!(delta < 0.0 || target == 0.0);
                    let raw = if target == 0.0 { pos } else { (-delta / open).min(pos) };
                    if raw >= pos {
                        prop_assert_eq!(f.qty.to_bits(), pos.to_bits(), "a full sale sells all");
                    } else if step > 0.0 {
                        prop_assert_eq!(lots_of(f.qty, step)?, floor_lots(raw, step));
                    } else {
                        prop_assert_eq!(f.qty.to_bits(), raw.to_bits());
                    }
                }
            }
        }
        for t in 0..bars.len() - 1 {
            if targets[t] == 0.0 && r.position[t + 1] > 0.0 {
                let exit = r.position[t] * (bars[t + 1].open * (1.0 - s));
                prop_assert!(exit < cfg.filters.min_notional);
            }
        }
    }

    /// The rebalance threshold: away from a full exit, no fill happens when
    /// the target is within the threshold of the weight at the open.
    #[test]
    fn threshold_is_respected((bars, targets, cfg) in arb_run_any()) {
        let r = simulate(&bars, &targets, &cfg).unwrap();
        for f in &r.fills {
            let target = targets[f.bar - 1];
            if target == 0.0 {
                continue;
            }
            let held = r.position[f.bar - 1] * bars[f.bar].open;
            let weight = held / (r.cash[f.bar - 1] + held);
            prop_assert!((target - weight).abs() > cfg.rebalance_threshold);
        }
    }

    /// Without costs or filters, holding everything tracks the price exactly:
    /// `equity[t] = cash · close[t] / open[1]` for t ≥ 1, from a single fill
    /// at bar 1.
    #[test]
    fn frictionless_buy_and_hold_tracks_price(
        (bars, cash) in (2usize..=64).prop_flat_map(|n| (arb_bars(n), 100.0f64..1e6))
    ) {
        let r = simulate(&bars, &vec![1.0; bars.len()], &frictionless(cash)).unwrap();
        prop_assert_eq!(r.fills.len(), 1);
        prop_assert_eq!(r.fills[0].bar, 1);
        prop_assert_eq!(r.equity[0], cash);
        for t in 1..bars.len() {
            let expect = cash * bars[t].close / bars[1].open;
            // The simulator computes qty = cash / open (one rounding), may
            // shrink it by a few ulps so that qty·open ≤ cash, then marks
            // residue + qty·close (two roundings); `expect` has two of its own.
            // That is under 16 ulps, far below 1e-12.
            prop_assert!(rel_err(r.equity[t], expect) < 1e-12, "t = {}", t);
        }
    }

    /// At constant prices, trading can only cost: equity never rises beyond
    /// rounding, and without costs it never moves beyond rounding.
    #[test]
    fn constant_prices_offer_no_free_lunch(
        (targets, cfg, price) in (2usize..=64).prop_flat_map(|n| (arb_targets(n), arb_cfg(), 1.0f64..1e5))
    ) {
        let bars = flat_bars(targets.len(), price);
        let r = simulate(&bars, &targets, &cfg).unwrap();
        for t in 1..bars.len() {
            // A trade changes equity by −(fee + slippage) ≤ 0 in exact
            // arithmetic; the cash update and the marking each add one
            // rounding of at most u·equity. 1e-12 ≫ 2u.
            prop_assert!(r.equity[t] <= r.equity[t - 1] * (1.0 + 1e-12));
        }
        let free = SimConfig { fees: FeeSchedule::ZERO, slippage_bps: 0.0, ..cfg };
        let r = simulate(&bars, &targets, &free).unwrap();
        for e in &r.equity {
            // At most 64 trades, each moving equity by a few roundings.
            prop_assert!(rel_err(*e, cfg.initial_cash) < 1e-12);
        }
    }

    /// Quoting prices in another unit (×2ʲ, with the lot step ×2⁻ʲ) changes no
    /// decision: equity and cash are bitwise identical and quantities scale
    /// exactly. Powers of two make every product exact, so any deviation would
    /// be a real dependence on the price level.
    #[test]
    fn price_unit_invariance((bars, targets, cfg) in arb_run(), j in -8i32..=8) {
        let k = 2f64.powi(j);
        let mut cfg_k = cfg;
        cfg_k.filters.step_size = cfg.filters.step_size / k;
        let a = simulate(&bars, &targets, &cfg).unwrap();
        let b = simulate(&scale_bars(&bars, k), &targets, &cfg_k).unwrap();
        prop_assert_eq!(bits(&a.equity), bits(&b.equity));
        prop_assert_eq!(bits(&a.cash), bits(&b.cash));
        let pos_a: Vec<f64> = a.position.iter().map(|p| p / k).collect();
        prop_assert_eq!(bits(&pos_a), bits(&b.position));
        prop_assert_eq!(a.fills.len(), b.fills.len());
        for (fa, fb) in a.fills.iter().zip(&b.fills) {
            prop_assert_eq!((fa.qty / k).to_bits(), fb.qty.to_bits());
            prop_assert_eq!(fa.notional.to_bits(), fb.notional.to_bits());
        }
    }

    /// Scaling the account (cash, minimum notional and lot step by 2ʲ) scales
    /// every output by 2ʲ exactly: the simulator has no hidden size dependence.
    #[test]
    fn account_size_homogeneity((bars, targets, cfg) in arb_run(), j in -8i32..=8) {
        let k = 2f64.powi(j);
        let mut cfg_k = cfg;
        cfg_k.initial_cash = cfg.initial_cash * k;
        cfg_k.filters.min_notional = cfg.filters.min_notional * k;
        cfg_k.filters.step_size = cfg.filters.step_size * k;
        let a = simulate(&bars, &targets, &cfg).unwrap();
        let b = simulate(&bars, &targets, &cfg_k).unwrap();
        let scaled = |xs: &[f64]| bits(&xs.iter().map(|x| x * k).collect::<Vec<_>>());
        prop_assert_eq!(scaled(&a.equity), bits(&b.equity));
        prop_assert_eq!(scaled(&a.cash), bits(&b.cash));
        prop_assert_eq!(scaled(&a.position), bits(&b.position));
        prop_assert_eq!((a.fees_paid * k).to_bits(), b.fees_paid.to_bits());
    }

    /// Time stamps only order the bars: shifting every `open_time` by a
    /// constant changes nothing but the fills' `time`.
    #[test]
    fn time_shift_invariance((bars, targets, cfg) in arb_run_any(), shift in -1_000_000_000_000i64..1_000_000_000_000) {
        let shifted: Vec<Bar> = bars.iter().map(|b| Bar { open_time: b.open_time + shift, ..*b }).collect();
        let a = simulate(&bars, &targets, &cfg).unwrap();
        let b = simulate(&shifted, &targets, &cfg).unwrap();
        prop_assert_eq!(bits(&a.equity), bits(&b.equity));
        prop_assert_eq!(a.fills.len(), b.fills.len());
        for (fa, fb) in a.fills.iter().zip(&b.fills) {
            prop_assert_eq!(fa.time + shift, fb.time);
        }
    }

    /// The last target never matters.
    #[test]
    fn last_target_is_ignored((bars, targets, cfg) in arb_run_any(), last in 0.0f64..=1.0) {
        let mut other = targets.clone();
        *other.last_mut().unwrap() = last;
        prop_assert_eq!(
            simulate(&bars, &targets, &cfg).unwrap(),
            simulate(&bars, &other, &cfg).unwrap()
        );
    }
}
