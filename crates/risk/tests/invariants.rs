//! Property tests for the risk kernel invariants P1–P7, a differential test
//! against an independent reference implementation, and a coverage check
//! that the generators reach every outcome and every overflow site.
//!
//! Tests run in the debug profile, so any arithmetic overflow panics: every
//! property below is also a sampled check of P6 (no panic).
//!
//! The reference re-derives every notional in `i128` from its definition and
//! finds quantities by binary search, so it shares no arithmetic with the
//! kernel: a wrong closed form, rounding direction or narrowing in `lib.rs`
//! shows up as a disagreement.

use proptest::prelude::*;
use proptest::strategy::{BoxedStrategy, ValueTree};
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use risk::*;

const CASES: u32 = 4096;

type Input = (Account, Limits, OrderRequest);

// ---- Generators ----------------------------------------------------------

const NONNEG_EDGES: &[i64] = &[
    0,
    1,
    2,
    999,
    1_000,
    QUOTE_SCALE - 1,
    QUOTE_SCALE,
    QTY_SCALE - 1,
    QTY_SCALE,
    QTY_SCALE + 1,
    1 << 31,
    1 << 32,
    1 << 53,
    i64::MAX / QTY_SCALE,
    i64::MAX / QTY_SCALE + 1,
    (1 << 62) - 1,
    i64::MAX / 2,
    i64::MAX - 1,
    i64::MAX,
];

const NEG_EDGES: &[i64] = &[i64::MIN, i64::MIN + 1, -QTY_SCALE, -2, -1];

const U32_EDGES: &[u32] = &[0, 1, 2, 9_999, 10_000, 10_001, u32::MAX - 1, u32::MAX];

/// Log-uniform over [2^min_bits, 2^(max_bits + 1)) ∩ i64, so every scale is
/// equally likely. `max_bits ≤ 62`.
fn log_uniform(min_bits: u32, max_bits: u32) -> impl Strategy<Value = i64> {
    (min_bits..=max_bits).prop_flat_map(|b| {
        let lo = 1i64 << b;
        let hi = if b >= 62 { i64::MAX } else { (lo << 1) - 1 };
        lo..=hi
    })
}

/// Non-negative over the full i64 range, edges included.
fn wide_nonneg() -> impl Strategy<Value = i64> {
    prop_oneof![
        5 => log_uniform(0, 62),
        2 => prop::sample::select(NONNEG_EDGES),
    ]
}

/// Positive over the full i64 range, edges included.
fn wide_pos() -> impl Strategy<Value = i64> {
    wide_nonneg().prop_map(|x| x.max(1))
}

/// Full i64 range, biased towards non-negative values so that deep paths are
/// reached with extreme magnitudes.
fn wide_i64() -> impl Strategy<Value = i64> {
    prop_oneof![
        12 => wide_nonneg(),
        1 => log_uniform(0, 62).prop_map(|x| -x),
        1 => prop::sample::select(NEG_EDGES),
        1 => any::<i64>(),
    ]
}

fn wide_u32() -> impl Strategy<Value = u32> {
    prop_oneof![
        any::<u32>(),
        prop::sample::select(U32_EDGES),
        0u32..=16,
        0u32..=BPS_SCALE,
    ]
}

fn side() -> impl Strategy<Value = Side> {
    prop_oneof![Just(Side::Buy), Just(Side::Sell)]
}

/// `x / 2^shift`, and 0 once the shift reaches the width.
fn shr(x: i64, shift: u32) -> i64 {
    x.checked_shr(shift).unwrap_or(0)
}

/// Every field independent over its full range. Mostly `InvalidInput`: this
/// generator tests validation, not the deep paths.
fn wide_inputs() -> BoxedStrategy<Input> {
    let account = (wide_i64(), wide_i64(), wide_i64(), wide_i64(), wide_u32()).prop_map(
        |(equity, peak, day, position, bars)| Account {
            equity: Quote(equity),
            peak_equity: Quote(peak),
            day_start_equity: Quote(day),
            position: Qty(position),
            bars_since_order: bars,
        },
    );
    let limits = (
        wide_i64(),
        wide_i64(),
        wide_i64(),
        wide_i64(),
        wide_i64(),
        wide_u32(),
        wide_u32(),
    )
        .prop_map(
            |(max_pos, max_order, min_order, lot, daily, dd, cool)| Limits {
                max_position: Quote(max_pos),
                max_order: Quote(max_order),
                min_order: Quote(min_order),
                lot_step: Qty(lot),
                max_daily_loss: Quote(daily),
                max_drawdown_bps: dd,
                cooldown_bars: cool,
            },
        );
    let request = (side(), wide_i64(), wide_i64()).prop_map(|(side, qty, price)| OrderRequest {
        side,
        qty: Qty(qty),
        price: Price(price),
    });
    (account, limits, request).boxed()
}

/// Market-scale inputs: prices from ~0.13 USDT to ~17.6M USDT per unit,
/// positions and requests up to ~176k units, consistent equity curves.
fn realistic_inputs() -> BoxedStrategy<Input> {
    let account = (
        log_uniform(20, 42),
        0i64..=4_000,
        -1_500i64..=1_500,
        prop_oneof![Just(0), log_uniform(0, 40)],
        0u32..=8,
    )
        .prop_map(|(peak, dd_bps, day_bps, position, bars)| {
            let equity = peak - peak * dd_bps / 10_000;
            Account {
                equity: Quote(equity),
                peak_equity: Quote(peak),
                day_start_equity: Quote(equity + equity * day_bps / 10_000),
                position: Qty(position),
                bars_since_order: bars,
            }
        });
    let limits = (
        log_uniform(16, 42),
        log_uniform(16, 40),
        prop_oneof![Just(0), log_uniform(10, 32)],
        log_uniform(0, 27),
        log_uniform(10, 40),
        prop_oneof![4 => 500u32..=3_000, 1 => prop::sample::select(vec![0u32, BPS_SCALE])],
        0u32..=4,
    )
        .prop_map(
            |(max_pos, max_order, min_order, lot, daily, dd, cool)| Limits {
                max_position: Quote(max_pos),
                max_order: Quote(max_order),
                min_order: Quote(min_order),
                lot_step: Qty(lot),
                max_daily_loss: Quote(daily),
                max_drawdown_bps: dd,
                cooldown_bars: cool,
            },
        );
    let request =
        (side(), log_uniform(0, 44), log_uniform(17, 44)).prop_map(|(side, qty, price)| {
            OrderRequest {
                side,
                qty: Qty(qty),
                price: Price(price),
            }
        });
    (account, limits, request).boxed()
}

/// Valid by construction, buy gates usually open, and every magnitude spread
/// over the full i64 range. The overflow paths need a position or a price
/// near the top of i64 *and* an otherwise valid, ungated buy; neither of the
/// other generators produces both (see `generators_reach_every_outcome`).
fn extreme_inputs() -> BoxedStrategy<Input> {
    let account = (
        wide_pos(),
        0u32..=64,
        prop_oneof![Just(0), wide_i64()],
        prop_oneof![
            1 => Just(0),
            5 => wide_nonneg(),
            // Within i64::MAX / 2^k of the top, for `position + qty`.
            3 => (0u32..=64).prop_map(|k| i64::MAX - shr(i64::MAX, k)),
        ],
        wide_u32(),
    )
        .prop_map(|(peak, shift, day_delta, position, bars)| {
            // A drawdown of peak / 2^shift: total at 0, none from 63 on.
            let equity = peak - shr(peak, shift);
            Account {
                equity: Quote(equity),
                peak_equity: Quote(peak),
                day_start_equity: Quote(equity.saturating_add(day_delta)),
                position: Qty(position),
                bars_since_order: bars,
            }
        });
    let limits = (
        prop_oneof![3 => wide_nonneg(), 1 => Just(i64::MAX)],
        wide_nonneg(),
        0u32..=64,
        wide_pos(),
        wide_nonneg(),
        prop_oneof![Just(BPS_SCALE), 0..=BPS_SCALE],
        prop_oneof![Just(0u32), wide_u32()],
    )
        .prop_map(|(max_pos, max_order, shift, lot, daily, dd, cool)| Limits {
            max_position: Quote(max_pos),
            max_order: Quote(max_order),
            min_order: Quote(shr(max_order, shift)),
            lot_step: Qty(lot),
            max_daily_loss: Quote(daily),
            max_drawdown_bps: dd,
            cooldown_bars: cool,
        });
    // Prices below QTY_SCALE (one quote unit per satoshi-sized qty unit) are
    // over-weighted: only there can a qty near i64::MAX have a notional that
    // fits, which is what `position + qty` overflow needs.
    let price = prop_oneof![3 => wide_pos(), 1 => log_uniform(0, 26)];
    let request = (side(), wide_pos(), price).prop_map(|(side, qty, price)| OrderRequest {
        side,
        qty: Qty(qty),
        price: Price(price),
    });
    (account, limits, request).boxed()
}

fn inputs() -> BoxedStrategy<Input> {
    prop_oneof![wide_inputs(), realistic_inputs(), extreme_inputs()].boxed()
}

/// Forces one buy gate on: 0 = kill switch, 1 = daily loss, 2 = cooldown.
fn force_gate(mut input: Input, gate: u8) -> Input {
    let (a, l, _) = &mut input;
    match gate % 3 {
        // (peak − equity)·1e4 ≥ 0 = 0·peak whenever equity ≤ peak.
        0 => {
            l.max_drawdown_bps = 0;
            a.equity = a.equity.min(a.peak_equity);
        }
        1 => {
            l.max_daily_loss = Quote(0);
            a.day_start_equity = a.equity;
        }
        _ => {
            a.bars_since_order = a.bars_since_order.min(u32::MAX - 1);
            l.cooldown_bars = a.bars_since_order + 1;
        }
    }
    input
}

/// Valid values for the fields a sell must not depend on.
fn sell_irrelevant_fields() -> BoxedStrategy<(Account, Limits)> {
    (
        (wide_i64(), wide_pos(), wide_i64(), any::<u32>()),
        (wide_nonneg(), wide_nonneg(), 0..=BPS_SCALE, any::<u32>()),
    )
        .prop_map(|((equity, peak, day, bars), (max_pos, daily, dd, cool))| {
            let account = Account {
                equity: Quote(equity),
                peak_equity: Quote(peak),
                day_start_equity: Quote(day),
                position: Qty(0),
                bars_since_order: bars,
            };
            let limits = Limits {
                max_position: Quote(max_pos),
                max_order: Quote(0),
                min_order: Quote(0),
                lot_step: Qty(1),
                max_daily_loss: Quote(daily),
                max_drawdown_bps: dd,
                cooldown_bars: cool,
            };
            (account, limits)
        })
        .boxed()
}

/// (cap, price) pairs: independent, or with the cap set to the exact
/// ceil-notional of some qty, so that the boundary q·p = cap·S is hit.
fn cap_and_price() -> impl Strategy<Value = (i64, i64)> {
    prop_oneof![
        (wide_i64(), wide_i64()),
        (wide_nonneg(), wide_pos()).prop_map(|(q, p)| {
            let cap = i64::try_from(ceil_ref(q, p)).unwrap_or(i64::MAX);
            (cap, p)
        }),
    ]
}

// ---- Independent reference ----------------------------------------------

fn scale() -> i128 {
    i128::from(QTY_SCALE)
}

/// floor(q·p / S), exact, for q, p ≥ 0.
fn floor_ref(q: i64, p: i64) -> i128 {
    i128::from(q) * i128::from(p) / scale()
}

/// ceil(q·p / S), exact, for q, p ≥ 0: floor plus one on a remainder (a
/// different formula from the kernel's `(x + S − 1) / S`).
fn ceil_ref(q: i64, p: i64) -> i128 {
    let x = i128::from(q) * i128::from(p);
    x / scale() + i128::from(x % scale() != 0)
}

fn narrow(x: i128) -> Result<Quote, Rejection> {
    i64::try_from(x).map(Quote).map_err(|_| Rejection::Overflow)
}

/// Does `q` fit the cap at `price`, by the definition ceil(q·p/S) ≤ cap?
fn fits(q: i64, cap: Quote, price: Price) -> bool {
    ceil_ref(q, price.0) <= i128::from(cap.0)
}

/// Largest `k·lot ≤ limit` with `fits(k·lot)`, by binary search on k. Sound
/// because `fits` is monotone in q (`notional_ceil_is_monotone`) and
/// `fits(0)` for every cap ≥ 0. Requires `limit ≥ 0`, `lot > 0`, `cap ≥ 0`.
fn largest_fitting_lot(limit: i64, lot: i64, cap: Quote, price: Price) -> i64 {
    let (mut lo, mut hi) = (0i64, limit / lot);
    while lo < hi {
        let mid = lo + (hi - lo - 1) / 2 + 1;
        if fits(mid * lot, cap, price) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo * lot
}

/// The binary-search oracle for `max_qty_within`: the largest qty in
/// [0, i64::MAX] that fits.
fn max_qty_oracle(cap: i64, price: i64) -> Option<Qty> {
    (cap >= 0 && price > 0).then(|| Qty(largest_fitting_lot(i64::MAX, 1, Quote(cap), Price(price))))
}

fn tripped(a: &Account, l: &Limits) -> (bool, bool, bool) {
    let (eq, peak, day) = (
        i128::from(a.equity.0),
        i128::from(a.peak_equity.0),
        i128::from(a.day_start_equity.0),
    );
    (
        (peak - eq) * 10_000 >= i128::from(l.max_drawdown_bps) * peak,
        day - eq >= i128::from(l.max_daily_loss.0),
        a.bars_since_order < l.cooldown_bars,
    )
}

fn valid_ref(a: &Account, l: &Limits, r: &OrderRequest) -> bool {
    r.qty.0 > 0
        && r.price.0 > 0
        && l.lot_step.0 > 0
        && [l.max_position, l.max_order, l.min_order, l.max_daily_loss]
            .iter()
            .all(|q| q.0 >= 0)
        && l.min_order <= l.max_order
        && l.max_drawdown_bps <= 10_000
        && a.position.0 >= 0
        && a.peak_equity.0 > 0
}

/// The quantity bounds of a valid, ungated request: (limit, cap) such that
/// the approved qty is the largest lot multiple ≤ limit that fits cap.
fn bounds_ref(a: &Account, l: &Limits, r: &OrderRequest) -> Result<(i64, Quote), Rejection> {
    match r.side {
        Side::Sell => {
            if a.position.0 == 0 {
                return Err(Rejection::NoPosition);
            }
            Ok((r.qty.0.min(a.position.0), l.max_order))
        }
        Side::Buy => {
            match tripped(a, l) {
                (true, _, _) => return Err(Rejection::KillSwitch),
                (_, true, _) => return Err(Rejection::DailyLoss),
                (_, _, true) => return Err(Rejection::Cooldown),
                _ => {}
            }
            let held = narrow(ceil_ref(a.position.0, r.price.0))?;
            let room = l.max_position.0 - held.0;
            if room <= 0 {
                return Err(Rejection::PositionCap);
            }
            Ok((r.qty.0, Quote(room.min(l.max_order.0))))
        }
    }
}

/// The brief's specification, restated without sharing arithmetic with the
/// kernel.
fn reference(a: &Account, l: &Limits, r: &OrderRequest) -> Result<(Side, Qty, Quote), Rejection> {
    if !valid_ref(a, l, r) {
        return Err(Rejection::InvalidInput);
    }
    let (limit, cap) = bounds_ref(a, l, r)?;
    let q = largest_fitting_lot(limit, l.lot_step.0, cap, r.price);
    if r.side == Side::Buy && a.position.0.checked_add(q).is_none() {
        return Err(Rejection::Overflow);
    }
    let floor = narrow(floor_ref(q, r.price.0))?;
    if q == 0 || floor < l.min_order {
        return Err(Rejection::BelowMinimum);
    }
    let ceil = narrow(ceil_ref(q, r.price.0))?;
    Ok((r.side, Qty(q), ceil))
}

fn run(input: &Input) -> Result<Approved, Rejection> {
    check_order(&input.0, &input.1, &input.2)
}

// ---- Properties ----------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: CASES,
        // Integration tests have no lib.rs for proptest to anchor regression
        // files to; failing seeds are printed instead.
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// Differential: the kernel equals the reference on every input.
    #[test]
    fn matches_reference(input in inputs()) {
        let got = run(&input).map(|a| (a.side(), a.qty(), a.notional()));
        prop_assert_eq!(got, reference(&input.0, &input.1, &input.2));
    }

    /// Differential on the extreme generator alone, where both overflow sites
    /// and approvals at full-range magnitudes live.
    #[test]
    fn matches_reference_extreme(input in extreme_inputs()) {
        let got = run(&input).map(|a| (a.side(), a.qty(), a.notional()));
        prop_assert_eq!(got, reference(&input.0, &input.1, &input.2));
    }

    /// Differential on inputs with a forced buy gate, where `matches_reference`
    /// rarely lands.
    #[test]
    fn matches_reference_gated(input in inputs(), gate in 0u8..3) {
        let input = force_gate(input, gate);
        let got = run(&input).map(|a| (a.side(), a.qty(), a.notional()));
        prop_assert_eq!(got, reference(&input.0, &input.1, &input.2));
    }

    /// P1: every approval is positive, a lot multiple, at most the request,
    /// carries its ceil-notional, fits max_order and meets min_order.
    #[test]
    fn p1_approved_orders_are_well_formed(input in inputs()) {
        let (_, l, r) = &input;
        if let Ok(ok) = run(&input) {
            prop_assert_eq!(ok.side(), r.side);
            prop_assert!(ok.qty().0 > 0);
            prop_assert_eq!(ok.qty().0 % l.lot_step.0, 0);
            prop_assert!(ok.qty() <= r.qty);
            prop_assert_eq!(Some(ok.notional()), notional_ceil(ok.qty(), r.price));
            prop_assert!(ok.notional() <= l.max_order);
            prop_assert!(notional_floor(ok.qty(), r.price).is_some_and(|f| f >= l.min_order));
        }
    }

    /// P2: an approved buy keeps the position's ceil-notional within the cap.
    #[test]
    fn p2_buy_respects_position_cap(input in inputs()) {
        let (a, l, r) = &input;
        if let Ok(ok) = run(&input) && ok.side() == Side::Buy {
            let next = a.position.0.checked_add(ok.qty().0);
            prop_assert!(next.is_some());
            let n = next.and_then(|q| notional_ceil(Qty(q), r.price));
            prop_assert!(n.is_some_and(|n| n <= l.max_position), "{:?}", n);
        }
    }

    /// P3: an approved sell never exceeds the position.
    #[test]
    fn p3_sell_never_shorts(input in inputs()) {
        if let Ok(ok) = run(&input) && ok.side() == Side::Sell {
            prop_assert!(ok.qty() <= input.0.position);
        }
    }

    /// P4: kill switch, daily loss or cooldown ⇒ no approved buy, both on
    /// natural inputs and with one gate forced on. On valid inputs the
    /// rejection is exactly the first tripped gate.
    #[test]
    fn p4_gated_account_never_buys(input in inputs(), gate in 0u8..3, force in any::<bool>()) {
        let mut input = if force { force_gate(input, gate) } else { input };
        input.2.side = Side::Buy;
        let (k, d, c) = tripped(&input.0, &input.1);
        prop_assert_eq!(k, kill_switch_tripped(&input.0, &input.1));
        prop_assert_eq!(d, daily_loss_tripped(&input.0, &input.1));
        if k || d || c {
            let expected = if !valid_ref(&input.0, &input.1, &input.2) {
                Rejection::InvalidInput
            } else if k {
                Rejection::KillSwitch
            } else if d {
                Rejection::DailyLoss
            } else {
                Rejection::Cooldown
            };
            prop_assert_eq!(run(&input), Err(expected));
        }
    }

    /// P5: a sell is never rejected by a buy gate, even with one forced on.
    #[test]
    fn p5_sell_is_never_gated(input in inputs(), gate in 0u8..3, force in any::<bool>()) {
        let mut input = if force { force_gate(input, gate) } else { input };
        input.2.side = Side::Sell;
        prop_assert!(!matches!(
            run(&input),
            Err(Rejection::KillSwitch | Rejection::DailyLoss | Rejection::Cooldown)
        ));
    }

    /// P5, stronger: a sell's outcome depends only on the position, the
    /// request, `lot_step`, `max_order` and `min_order`. Equity, peak, day
    /// start, bars, `max_position` and every trip wire can be swapped for any
    /// other valid values without changing it, so no account state and no
    /// trip-wire setting can trap a position.
    #[test]
    fn p5_sell_outcome_depends_only_on_sell_fields(
        input in inputs(),
        one in sell_irrelevant_fields(),
        two in sell_irrelevant_fields(),
    ) {
        let (a, l, mut r) = input;
        r.side = Side::Sell;
        let with = |(oa, ol): (Account, Limits)| {
            let account = Account { position: a.position, ..oa };
            let limits = Limits {
                max_order: l.max_order,
                min_order: l.min_order,
                lot_step: l.lot_step,
                ..ol
            };
            check_order(&account, &limits, &r)
        };
        prop_assert_eq!(with(one), with(two));
    }

    /// `Overflow` only comes from its two reachable sites: a sell never
    /// overflows, and a buy only when the held notional or
    /// `position + request.qty` leaves i64.
    #[test]
    fn overflow_only_at_reachable_sites(input in extreme_inputs()) {
        let (a, _, r) = &input;
        if run(&input) == Err(Rejection::Overflow) {
            prop_assert_eq!(r.side, Side::Buy);
            prop_assert!(
                notional_ceil(a.position, r.price).is_none()
                    || a.position.0.checked_add(r.qty.0).is_none()
            );
        }
    }

    /// The kernel shrinks a request only as far as it must: one more lot
    /// would break a bound (request, position, cap, or i64).
    #[test]
    fn approved_qty_is_maximal(input in inputs()) {
        let (a, l, r) = &input;
        if let Ok(ok) = run(&input) {
            let (limit, cap) = bounds_ref(a, l, r).expect("approved implies bounds");
            let more = ok.qty().0.checked_add(l.lot_step.0);
            let blocked = more.is_none_or(|m| {
                m > limit
                    || !fits(m, cap, r.price)
                    || (r.side == Side::Buy && a.position.0.checked_add(m).is_none())
            });
            prop_assert!(blocked, "{:?} + one lot would still be acceptable", ok);
        }
    }

    /// Re-submitting an approval returns the same approval: the kernel is
    /// idempotent on its own output.
    #[test]
    fn approval_is_a_fixed_point(input in inputs()) {
        let (a, l, r) = &input;
        if let Ok(ok) = run(&input) {
            let again = OrderRequest { qty: ok.qty(), ..*r };
            prop_assert_eq!(check_order(a, l, &again), Ok(ok));
        }
    }

    /// P6: the helpers never panic on raw inputs, and the notionals obey
    /// their defining inequalities floor·S ≤ q·p < (floor + 1)·S and
    /// floor ≤ ceil ≤ floor + 1.
    #[test]
    fn p6_notional_helpers_are_total_and_exact(q in wide_i64(), p in wide_i64()) {
        let (floor, ceil) = (notional_floor(Qty(q), Price(p)), notional_ceil(Qty(q), Price(p)));
        if q < 0 || p < 0 {
            prop_assert!(floor.is_none() && ceil.is_none());
        } else {
            let prod = i128::from(q) * i128::from(p);
            let s = scale();
            prop_assert_eq!(floor, narrow(floor_ref(q, p)).ok());
            prop_assert_eq!(ceil, narrow(ceil_ref(q, p)).ok());
            if let Some(f) = floor {
                let f = i128::from(f.0);
                prop_assert!(f * s <= prod && prod < (f + 1) * s);
            }
            if let (Some(f), Some(c)) = (floor, ceil) {
                prop_assert!(f <= c && c.0 - f.0 <= 1);
                prop_assert_eq!(c == f, prod % s == 0);
            }
        }
    }

    /// P6: the trip-wires never panic on raw inputs and match their
    /// definitions.
    #[test]
    fn p6_trip_wires_are_total(input in wide_inputs()) {
        let (k, d, _) = tripped(&input.0, &input.1);
        prop_assert_eq!(k, kill_switch_tripped(&input.0, &input.1));
        prop_assert_eq!(d, daily_loss_tripped(&input.0, &input.1));
    }

    /// Losing more never un-trips a wire: both are monotone in equity.
    #[test]
    fn trip_wires_are_monotone_in_equity(input in inputs(), lower in wide_i64()) {
        let (a, l, _) = &input;
        let worse = Account { equity: Quote(lower.min(a.equity.0)), ..*a };
        prop_assert!(!kill_switch_tripped(a, l) || kill_switch_tripped(&worse, l));
        prop_assert!(!daily_loss_tripped(a, l) || daily_loss_tripped(&worse, l));
    }

    /// P7: `max_qty_within` is exact: q fits the cap, q + 1 does not.
    #[test]
    fn p7_max_qty_within_is_exact((cap, price) in cap_and_price()) {
        match max_qty_within(Quote(cap), Price(price)) {
            None => prop_assert!(cap < 0 || price <= 0),
            Some(q) => {
                prop_assert!(q.0 >= 0);
                prop_assert!(fits(q.0, Quote(cap), Price(price)));
                prop_assert!(notional_ceil(q, Price(price)).is_some_and(|n| n.0 <= cap));
                if let Some(next) = q.0.checked_add(1) {
                    prop_assert!(!fits(next, Quote(cap), Price(price)));
                }
            }
        }
    }

    /// P7 against the oracle: the closed form floor(cap·S/p), saturated at
    /// i64::MAX, equals a binary search over every qty in [0, i64::MAX].
    #[test]
    fn p7_max_qty_within_matches_binary_search((cap, price) in cap_and_price()) {
        prop_assert_eq!(max_qty_within(Quote(cap), Price(price)), max_qty_oracle(cap, price));
    }

    /// P7 as a Galois connection: for every q ≥ 0,
    /// ceil(q·p/S) ≤ cap ⇔ q ≤ max_qty_within(cap, p). Half the cases put q
    /// within two of the boundary.
    #[test]
    fn p7_max_qty_within_is_a_galois_connection(
        (cap, price) in cap_and_price(),
        q in wide_nonneg(),
        delta in -2i64..=2,
        near in any::<bool>(),
    ) {
        if let Some(m) = max_qty_within(Quote(cap), Price(price)) {
            let q = if near { m.0.saturating_add(delta).max(0) } else { q };
            prop_assert_eq!(fits(q, Quote(cap), Price(price)), q <= m.0);
        }
    }

    /// Notionals are monotone in qty, so "fits the cap" is a prefix of qty
    /// values and the binary search in the reference is sound.
    #[test]
    fn notional_ceil_is_monotone(a in wide_nonneg(), b in wide_nonneg(), p in wide_nonneg()) {
        let (lo, hi) = (a.min(b), a.max(b));
        if let Some(n_hi) = notional_ceil(Qty(hi), Price(p)) {
            prop_assert!(notional_ceil(Qty(lo), Price(p)).is_some_and(|n_lo| n_lo <= n_hi));
        }
    }

    /// The lemma behind P2: ceil is subadditive,
    /// ceil((a + b)·p/S) ≤ ceil(a·p/S) + ceil(b·p/S).
    #[test]
    fn notional_ceil_is_subadditive(a in wide_nonneg(), b in wide_nonneg(), p in wide_nonneg()) {
        if let Some(sum) = a.checked_add(b) {
            prop_assert!(ceil_ref(sum, p) <= ceil_ref(a, p) + ceil_ref(b, p));
        }
    }
}

// ---- Coverage: the properties above are not vacuous ----------------------

const OUTCOMES: [&str; 10] = [
    "approved buy",
    "approved sell",
    "InvalidInput",
    "KillSwitch",
    "DailyLoss",
    "Cooldown",
    "PositionCap",
    "NoPosition",
    "BelowMinimum",
    "Overflow",
];

/// Finer cases inside the outcomes: which bound binds an approval, and which
/// of the two reachable overflow sites fired.
const DETAILS: [&str; 8] = [
    "buy bound by request",
    "buy bound by order cap",
    "buy bound by position room",
    "sell bound by request",
    "sell bound by position",
    "sell bound by order cap",
    "Overflow at held notional",
    "Overflow at position + qty",
];

fn outcome_slot(result: Result<Approved, Rejection>) -> usize {
    match result {
        Ok(a) if a.side() == Side::Buy => 0,
        Ok(_) => 1,
        Err(e) => 2 + e as usize,
    }
}

fn detail_slot(input: &Input, result: Result<Approved, Rejection>) -> Option<usize> {
    let (a, l, r) = input;
    match result {
        Ok(_) => {
            let (limit, cap) = bounds_ref(a, l, r).ok()?;
            let cap_qty = max_qty_oracle(cap.0, r.price.0)?.0;
            Some(match r.side {
                Side::Buy if r.qty.0 <= cap_qty => 0,
                Side::Buy if cap < l.max_order => 2,
                Side::Buy => 1,
                Side::Sell if limit.min(cap_qty) == r.qty.0 => 3,
                Side::Sell if limit.min(cap_qty) == a.position.0 => 4,
                Side::Sell => 5,
            })
        }
        Err(Rejection::Overflow) if notional_ceil(a.position, r.price).is_none() => Some(6),
        Err(Rejection::Overflow) => Some(7),
        Err(_) => None,
    }
}

fn tally<S: Strategy<Value = Input>>(strategy: S, samples: usize) -> ([usize; 10], [usize; 8]) {
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    );
    let strategy = (strategy, 0u8..3, any::<bool>());
    let (mut outcomes, mut details) = ([0usize; 10], [0usize; 8]);
    for _ in 0..samples {
        let (input, gate, force) = strategy.new_tree(&mut runner).unwrap().current();
        let input = if force {
            force_gate(input, gate)
        } else {
            input
        };
        let result = run(&input);
        outcomes[outcome_slot(result)] += 1;
        if let Some(d) = detail_slot(&input, result) {
            details[d] += 1;
        }
    }
    (outcomes, details)
}

/// Samples each generator deterministically and checks that every outcome,
/// every binding bound and both overflow sites occur often enough for the
/// implications in P1–P5 to bite. The per-generator table is printed with
/// `--nocapture`.
#[test]
fn generators_reach_every_outcome() {
    const SAMPLES: usize = 8_000;
    let tables = [
        ("wide", tally(wide_inputs(), SAMPLES)),
        ("realistic", tally(realistic_inputs(), SAMPLES)),
        ("extreme", tally(extreme_inputs(), SAMPLES)),
    ];
    eprintln!(
        "{:>28} {:>9} {:>9} {:>9}",
        "", "wide", "realistic", "extreme"
    );
    let rows = OUTCOMES.iter().enumerate().map(|(i, n)| (*n, i, true));
    let rows = rows.chain(DETAILS.iter().enumerate().map(|(i, n)| (*n, i, false)));
    for (name, i, is_outcome) in rows {
        let counts: Vec<usize> = tables
            .iter()
            .map(|(_, (o, d))| if is_outcome { o[i] } else { d[i] })
            .collect();
        eprintln!(
            "{name:>28} {:>9} {:>9} {:>9}",
            counts[0], counts[1], counts[2]
        );
        let total: usize = counts.iter().sum();
        let floor = if is_outcome { 50 } else { 20 };
        assert!(
            total >= floor,
            "{name} reached only {total} times in {}",
            3 * SAMPLES
        );
    }
}
