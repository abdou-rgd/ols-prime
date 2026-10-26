//! Worked example and hand-built cases for every `Rejection` variant.

use risk::*;

const fn usdt(x: i64) -> Quote {
    Quote(x * QUOTE_SCALE)
}

/// 67,000 USDT per BTC.
const BTC_67K: Price = Price(67_000_000_000);
/// 50,000 USDT per BTC: one satoshi is exactly 500 quote units.
const BTC_50K: Price = Price(50_000_000_000);

fn flat_account() -> Account {
    Account {
        equity: usdt(100),
        peak_equity: usdt(100),
        day_start_equity: usdt(100),
        position: Qty(0),
        bars_since_order: 10,
    }
}

fn limits() -> Limits {
    Limits {
        max_position: usdt(100),
        max_order: usdt(50),
        min_order: usdt(5),
        lot_step: Qty(1_000),
        max_daily_loss: usdt(10),
        max_drawdown_bps: 2_000,
        cooldown_bars: 3,
    }
}

const fn order(side: Side, qty: i64, price: Price) -> OrderRequest {
    OrderRequest {
        side,
        qty: Qty(qty),
        price,
    }
}

const fn buy(qty: i64, price: Price) -> OrderRequest {
    order(Side::Buy, qty, price)
}

const fn sell(qty: i64, price: Price) -> OrderRequest {
    order(Side::Sell, qty, price)
}

fn holding(sats: i64) -> Account {
    Account {
        position: Qty(sats),
        ..flat_account()
    }
}

/// An account 21 % below its peak and 21 USDT down on the day: the kill
/// switch and the daily loss limit have both tripped, and an order was just
/// sent, so the cooldown is active too.
fn wrecked(sats: i64) -> Account {
    Account {
        equity: usdt(79),
        peak_equity: usdt(100),
        day_start_equity: usdt(100),
        position: Qty(sats),
        bars_since_order: 0,
    }
}

fn parts(a: Approved) -> (Side, Qty, Quote) {
    (a.side(), a.qty(), a.notional())
}

#[test]
fn worked_example() {
    assert_eq!(max_qty_within(usdt(50), BTC_67K), Some(Qty(74_626)));
    let ok = check_order(&flat_account(), &limits(), &buy(100_000, BTC_67K)).unwrap();
    assert_eq!(parts(ok), (Side::Buy, Qty(74_000), Quote(49_580_000)));
}

#[test]
fn small_buy_is_approved_unchanged() {
    // 0.0001 BTC at 67k = 6.70 USDT: inside every cap, already a lot multiple.
    let ok = check_order(&flat_account(), &limits(), &buy(10_000, BTC_67K)).unwrap();
    assert_eq!(parts(ok), (Side::Buy, Qty(10_000), Quote(6_700_000)));
}

#[test]
fn buy_is_cut_to_the_remaining_position_room() {
    // 60 USDT held at 50k (120,000 sats), cap 100: 40 USDT of room left,
    // i.e. 80,000 sats, below the 50 USDT order cap.
    let ok = check_order(&holding(120_000), &limits(), &buy(1_000_000, BTC_50K)).unwrap();
    assert_eq!(parts(ok), (Side::Buy, Qty(80_000), usdt(40)));
}

#[test]
fn notional_rounding_is_conservative() {
    // 1 sat at 67k is 670 quote units exactly; 1 sat at 1 quote unit/BTC is
    // 1e-8 quote units: floor 0, ceil 1.
    assert_eq!(notional_floor(Qty(1), BTC_67K), Some(Quote(670)));
    assert_eq!(notional_ceil(Qty(1), BTC_67K), Some(Quote(670)));
    assert_eq!(notional_floor(Qty(1), Price(1)), Some(Quote(0)));
    assert_eq!(notional_ceil(Qty(1), Price(1)), Some(Quote(1)));
    assert_eq!(notional_floor(Qty(-1), Price(1)), None);
    assert_eq!(notional_ceil(Qty(1), Price(-1)), None);
    assert_eq!(notional_ceil(Qty(i64::MAX), Price(i64::MAX)), None);
    assert_eq!(
        max_qty_within(Quote(i64::MAX), Price(1)),
        Some(Qty(i64::MAX))
    );
    assert_eq!(max_qty_within(Quote(-1), Price(1)), None);
    assert_eq!(max_qty_within(Quote(1), Price(0)), None);
}

#[test]
fn ceil_can_overflow_where_floor_does_not() {
    // q·p = i64::MAX·S + 21_056_361: the floor is exactly i64::MAX, so the
    // ceil is i64::MAX + 1 and must be reported as an overflow.
    let (q, p) = (Qty(9_223_371_944_621_056_361), Price(100_000_001));
    assert_eq!(notional_floor(q, p), Some(Quote(i64::MAX)));
    assert_eq!(notional_ceil(q, p), None);
}

#[test]
fn trip_wires_are_inclusive() {
    let mut account = flat_account();
    let limits = limits();
    account.equity = usdt(80); // exactly 20 % = 2,000 bps down, 20 USDT on the day
    assert!(kill_switch_tripped(&account, &limits));
    assert!(daily_loss_tripped(&account, &limits));
    account.equity = Quote(80_000_001);
    assert!(!kill_switch_tripped(&account, &limits));
    account.day_start_equity = Quote(90_000_000);
    assert!(!daily_loss_tripped(&account, &limits));
    account.day_start_equity = Quote(90_000_001);
    assert!(daily_loss_tripped(&account, &limits));
}

// ---- One test per rejection variant -------------------------------------

#[test]
fn rejects_invalid_input() {
    let a = flat_account();
    let l = limits();
    let r = buy(10_000, BTC_67K);
    let cases: [(Account, Limits, OrderRequest); 12] = [
        (a, l, buy(0, BTC_67K)),
        (a, l, buy(-1, BTC_67K)),
        (a, l, buy(10_000, Price(0))),
        (
            a,
            Limits {
                lot_step: Qty(0),
                ..l
            },
            r,
        ),
        (
            a,
            Limits {
                max_position: Quote(-1),
                ..l
            },
            r,
        ),
        (
            a,
            Limits {
                max_order: Quote(-1),
                ..l
            },
            r,
        ),
        (
            a,
            Limits {
                min_order: Quote(-1),
                ..l
            },
            r,
        ),
        (
            a,
            Limits {
                max_daily_loss: Quote(-1),
                ..l
            },
            r,
        ),
        (
            a,
            Limits {
                min_order: usdt(51),
                ..l
            },
            r,
        ),
        (
            a,
            Limits {
                max_drawdown_bps: 10_001,
                ..l
            },
            r,
        ),
        (
            Account {
                position: Qty(-1),
                ..a
            },
            l,
            sell(1_000, BTC_67K),
        ),
        (
            Account {
                peak_equity: Quote(0),
                equity: Quote(0),
                ..a
            },
            l,
            r,
        ),
    ];
    for (i, (a, l, r)) in cases.iter().enumerate() {
        assert_eq!(
            check_order(a, l, r),
            Err(Rejection::InvalidInput),
            "case {i}"
        );
    }
}

#[test]
fn rejects_buy_on_kill_switch() {
    assert_eq!(
        check_order(&wrecked(0), &limits(), &buy(10_000, BTC_67K)),
        Err(Rejection::KillSwitch)
    );
}

#[test]
fn rejects_buy_on_daily_loss() {
    // 11 USDT down on the day, but only 11 % below peak.
    let account = Account {
        equity: usdt(89),
        ..flat_account()
    };
    assert_eq!(
        check_order(&account, &limits(), &buy(10_000, BTC_67K)),
        Err(Rejection::DailyLoss)
    );
}

#[test]
fn rejects_buy_during_cooldown() {
    let account = Account {
        bars_since_order: 2,
        ..flat_account()
    };
    assert_eq!(
        check_order(&account, &limits(), &buy(10_000, BTC_67K)),
        Err(Rejection::Cooldown)
    );
    // The bar that ends the cooldown is allowed.
    let account = Account {
        bars_since_order: 3,
        ..flat_account()
    };
    assert!(check_order(&account, &limits(), &buy(10_000, BTC_67K)).is_ok());
}

#[test]
fn rejects_buy_at_position_cap() {
    // 200,000 sats at 50k is exactly 100 USDT: zero room.
    assert_eq!(
        check_order(&holding(200_000), &limits(), &buy(10_000, BTC_50K)),
        Err(Rejection::PositionCap)
    );
    // Over the cap after a price rise.
    assert_eq!(
        check_order(&holding(200_000), &limits(), &buy(10_000, BTC_67K)),
        Err(Rejection::PositionCap)
    );
}

#[test]
fn rejects_sell_without_position() {
    assert_eq!(
        check_order(&flat_account(), &limits(), &sell(10_000, BTC_67K)),
        Err(Rejection::NoPosition)
    );
}

#[test]
fn rejects_below_minimum() {
    let (a, l) = (flat_account(), limits());
    // 5,000 sats at 67k = 3.35 USDT < 5 USDT.
    assert_eq!(
        check_order(&a, &l, &buy(5_000, BTC_67K)),
        Err(Rejection::BelowMinimum)
    );
    // Rounds down to zero lots.
    assert_eq!(
        check_order(&a, &l, &buy(999, BTC_67K)),
        Err(Rejection::BelowMinimum)
    );
    // Room of 4 USDT left under the position cap: below the minimum.
    assert_eq!(
        check_order(&holding(192_000), &l, &buy(100_000, BTC_50K)),
        Err(Rejection::BelowMinimum)
    );
    // Dust below one lot cannot be sold.
    assert_eq!(
        check_order(&holding(999), &l, &sell(999, BTC_67K)),
        Err(Rejection::BelowMinimum)
    );
}

#[test]
fn rejects_overflow() {
    let wide = Limits {
        max_position: Quote(i64::MAX),
        max_order: Quote(i64::MAX),
        min_order: Quote(0),
        lot_step: Qty(1),
        ..limits()
    };
    // The held notional does not fit in i64.
    assert_eq!(
        check_order(&holding(i64::MAX), &wide, &buy(1, Price(i64::MAX))),
        Err(Rejection::Overflow)
    );
    // The new position does not fit in i64.
    assert_eq!(
        check_order(&holding(i64::MAX - 10), &wide, &buy(100, Price(1))),
        Err(Rejection::Overflow)
    );
}

// ---- Sells stay open -----------------------------------------------------

#[test]
fn equity_above_a_stale_peak_is_no_drawdown() {
    // The caller has not yet raised the peak to the new equity: that is a
    // zero drawdown, not an invalid account, and it must not block a sell.
    let account = Account {
        equity: usdt(120),
        day_start_equity: usdt(120),
        ..holding(30_000)
    };
    let l = limits();
    assert!(!kill_switch_tripped(&account, &l));
    let ok = check_order(&account, &l, &sell(30_000, BTC_50K)).unwrap();
    assert_eq!(parts(ok), (Side::Sell, Qty(30_000), usdt(15)));
    let ok = check_order(&account, &l, &buy(10_000, BTC_50K)).unwrap();
    assert_eq!(parts(ok), (Side::Buy, Qty(10_000), usdt(5)));
}

#[test]
fn sell_is_allowed_after_every_trip_wire() {
    let account = wrecked(100_000);
    let l = limits();
    assert!(kill_switch_tripped(&account, &l));
    assert!(daily_loss_tripped(&account, &l));
    let ok = check_order(&account, &l, &sell(100_000, BTC_50K)).unwrap();
    assert_eq!(parts(ok), (Side::Sell, Qty(100_000), usdt(50)));
}

#[test]
fn sell_is_clamped_to_position_and_order_cap() {
    let l = limits();
    // Asks for more than held: clamped to the 30,000 sats held.
    let ok = check_order(&holding(30_000), &l, &sell(1_000_000, BTC_50K)).unwrap();
    assert_eq!(parts(ok), (Side::Sell, Qty(30_000), usdt(15)));
    // Holds 1 BTC: clamped to the 50 USDT order cap = 100,000 sats at 50k.
    let ok = check_order(&holding(QTY_SCALE), &l, &sell(QTY_SCALE, BTC_50K)).unwrap();
    assert_eq!(parts(ok), (Side::Sell, Qty(100_000), usdt(50)));
}

#[test]
fn rejection_is_a_std_error_with_a_message() {
    let all = [
        Rejection::InvalidInput,
        Rejection::KillSwitch,
        Rejection::DailyLoss,
        Rejection::Cooldown,
        Rejection::PositionCap,
        Rejection::NoPosition,
        Rejection::BelowMinimum,
        Rejection::Overflow,
    ];
    let messages: std::collections::BTreeSet<String> = all
        .iter()
        .map(|r| {
            let e: &dyn std::error::Error = r;
            e.to_string()
        })
        .collect();
    assert_eq!(messages.len(), all.len());
    assert!(messages.iter().all(|m| !m.is_empty()));
}
