//! Fixed-point risk kernel: the last check between any model and an exchange.
//!
//! [`check_order`] is a pure function of the account, the limits and the
//! request. It uses integers only, contains no loop and cannot panic, so the
//! same code is property-tested here and model-checked by the Kani harnesses
//! in `proofs` (`cargo kani -p risk`, Linux only).
//!
//! # Units
//!
//! - [`Qty`]: 1 base unit = [`QTY_SCALE`] = 1e8 qty units (one satoshi for BTC).
//! - [`Quote`]: 1 USDT = [`QUOTE_SCALE`] = 1e6 quote units.
//! - [`Price`]: quote units for one *whole* base unit, so
//!   `notional = qty · price / QTY_SCALE`, computed in `i128`.
//!
//! `i64 · i64` fits in `i128` (|product| < 2^126), so every intermediate
//! product below is exact; only the final narrowing to `i64` can fail.
//!
//! # Example
//!
//! A flat account with 100 USDT buys 0.001 BTC at 67,000 USDT, under a
//! 50 USDT per-order cap and a 0.00001 BTC lot step. The order cap allows
//! floor(50e6 · 1e8 / 67e9) = 74,626 sats, which the lot rounds down to 74,000.
//!
//! ```
//! use risk::*;
//!
//! let usdt = |x: i64| Quote(x * QUOTE_SCALE);
//! let account = Account {
//!     equity: usdt(100),
//!     peak_equity: usdt(100),
//!     day_start_equity: usdt(100),
//!     position: Qty(0),
//!     bars_since_order: 10,
//! };
//! let limits = Limits {
//!     max_position: usdt(100),
//!     max_order: usdt(50),
//!     min_order: usdt(5),
//!     lot_step: Qty(1_000),
//!     max_daily_loss: usdt(10),
//!     max_drawdown_bps: 2_000,
//!     cooldown_bars: 3,
//! };
//! let request = OrderRequest { side: Side::Buy, qty: Qty(100_000), price: Price(67_000_000_000) };
//!
//! let approved = check_order(&account, &limits, &request).unwrap();
//! assert_eq!(approved.side(), Side::Buy);
//! assert_eq!(approved.qty(), Qty(74_000));
//! assert_eq!(approved.notional(), Quote(49_580_000)); // 49.58 USDT
//! ```

// `no_std` except under Kani: the kernel needs nothing from `std`, and the
// harness build keeps the default prelude so that nothing Kani injects can
// depend on it being present.
#![cfg_attr(not(kani), no_std)]
#![deny(missing_docs)]

use core::fmt;

#[cfg(kani)]
mod proofs;
// The contracts are compiled for Kani and for the unit tests that check them
// against the helpers on random inputs.
#[cfg(any(kani, test))]
mod spec;

/// Qty units per whole base unit (1e-8 resolution).
pub const QTY_SCALE: i64 = 100_000_000;

/// Quote units per whole quote unit, e.g. per USDT (1e-6 resolution).
pub const QUOTE_SCALE: i64 = 1_000_000;

/// Basis points in 100 %.
pub const BPS_SCALE: u32 = 10_000;

/// A base-asset quantity in units of 1 / [`QTY_SCALE`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Qty(pub i64);

/// Quote units (1 / [`QUOTE_SCALE`]) for one whole base unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Price(pub i64);

/// A quote-asset amount in units of 1 / [`QUOTE_SCALE`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Quote(pub i64);

/// Order direction. The kernel is long-only: a sell can only reduce a long.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum Side {
    /// Increase the long position.
    Buy,
    /// Reduce the long position.
    Sell,
}

/// Static risk limits. Validated by [`check_order`] on every call.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Limits {
    /// Cap on the position's ceil-notional after a buy.
    pub max_position: Quote,
    /// Cap on one order's ceil-notional.
    pub max_order: Quote,
    /// Exchange minimum on one order's floor-notional.
    pub min_order: Quote,
    /// Exchange lot step: every approved qty is a multiple of it.
    pub lot_step: Qty,
    /// Buys stop once `day_start_equity − equity ≥ max_daily_loss`.
    pub max_daily_loss: Quote,
    /// Buys stop once the drawdown from peak reaches this many basis points.
    pub max_drawdown_bps: u32,
    /// Buys need at least this many bars since the previous order.
    pub cooldown_bars: u32,
}

/// Account state at the decision bar's close.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Account {
    /// Marked-to-market equity.
    pub equity: Quote,
    /// Running maximum of equity; must be > 0. An `equity` above it reads as
    /// a zero drawdown.
    pub peak_equity: Quote,
    /// Equity at the start of the trading day.
    pub day_start_equity: Quote,
    /// Current long position (never negative: no shorting).
    pub position: Qty,
    /// Bars elapsed since the last order was sent.
    pub bars_since_order: u32,
}

/// What a model asks for. The kernel may shrink `qty`, never grow it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct OrderRequest {
    /// Direction.
    pub side: Side,
    /// Requested quantity; the approved quantity is at most this.
    pub qty: Qty,
    /// Reference price used for every notional check.
    pub price: Price,
}

/// An order that passed every check.
///
/// The fields are private and only [`check_order`] builds this type, so an
/// `Approved` cannot be forged downstream: whoever holds one holds an order
/// that satisfies the kernel's invariants.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Approved {
    side: Side,
    qty: Qty,
    notional: Quote,
}

impl Approved {
    /// Direction, equal to the request's.
    #[must_use]
    pub const fn side(&self) -> Side {
        self.side
    }

    /// Approved quantity: positive, a lot multiple, at most the request.
    #[must_use]
    pub const fn qty(&self) -> Qty {
        self.qty
    }

    /// `notional_ceil(qty, price)`: the conservative (rounded-up) notional.
    #[must_use]
    pub const fn notional(&self) -> Quote {
        self.notional
    }
}

/// Why an order was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rejection {
    /// The account, the limits or the request violate a precondition of
    /// [`check_order`].
    InvalidInput,
    /// Buy refused: the drawdown kill switch has tripped.
    KillSwitch,
    /// Buy refused: the daily loss limit has been reached.
    DailyLoss,
    /// Buy refused: too few bars since the previous order.
    Cooldown,
    /// Buy refused: the position already fills the position cap.
    PositionCap,
    /// Sell refused: there is no position to reduce.
    NoPosition,
    /// The largest allowed lot multiple is zero or below the exchange minimum.
    BelowMinimum,
    /// A notional or the new position does not fit in `i64`.
    Overflow,
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidInput => "invalid account, limits or order request",
            Self::KillSwitch => "drawdown kill switch tripped: buys are blocked",
            Self::DailyLoss => "daily loss limit reached: buys are blocked",
            Self::Cooldown => "cooldown active: too few bars since the last order",
            Self::PositionCap => "position cap reached",
            Self::NoPosition => "no position to sell",
            Self::BelowMinimum => "order is zero after lot rounding or below the exchange minimum",
            Self::Overflow => "arithmetic overflow in a notional or the position",
        })
    }
}

impl core::error::Error for Rejection {}

/// `qty · price` in `i128`, or `None` if either factor is negative.
/// Exact: |i64 · i64| ≤ 2^126 < i128::MAX.
#[cfg_attr(kani, kani::ensures(|r: &Option<i128>| spec::product_spec(qty, price, *r)))]
fn product(qty: Qty, price: Price) -> Option<i128> {
    if qty.0 < 0 || price.0 < 0 {
        return None;
    }
    Some(i128::from(qty.0) * i128::from(price.0))
}

/// floor(qty · price / QTY_SCALE).
///
/// `None` if `qty` or `price` is negative, or if the result exceeds `i64`.
#[must_use]
#[cfg_attr(kani, kani::ensures(|r: &Option<Quote>| spec::floor_spec(qty, price, *r)))]
pub fn notional_floor(qty: Qty, price: Price) -> Option<Quote> {
    let p = product(qty, price)?;
    i64::try_from(p / i128::from(QTY_SCALE)).ok().map(Quote)
}

/// ceil(qty · price / QTY_SCALE): the notional rounded against us, used for
/// every cap.
///
/// `None` if `qty` or `price` is negative, or if the result exceeds `i64`.
#[must_use]
#[cfg_attr(kani, kani::ensures(|r: &Option<Quote>| spec::ceil_spec(qty, price, *r)))]
pub fn notional_ceil(qty: Qty, price: Price) -> Option<Quote> {
    let p = product(qty, price)?;
    let s = i128::from(QTY_SCALE);
    // p ≥ 0 and p + s − 1 < 2^126 + 2^27: no overflow, and integer division
    // of a non-negative numerator is floor, so this is ceil(p / s).
    i64::try_from((p + s - 1) / s).ok().map(Quote)
}

/// The largest qty whose ceil-notional at `price` is at most `cap`:
/// floor(cap · QTY_SCALE / price).
///
/// Exactness: for an integer cap C, ceil(q·p/S) ≤ C ⇔ q·p ≤ C·S ⇔
/// q ≤ floor(C·S/p), so the result fits and the next qty does not.
///
/// The true quotient can exceed `i64::MAX` (tiny price, large cap); it then
/// saturates to `i64::MAX`, which still fits the cap, and every qty fits.
///
/// `None` if `cap < 0` or `price ≤ 0`.
#[must_use]
#[cfg_attr(kani, kani::ensures(|r: &Option<Qty>| spec::max_qty_spec(cap, price, *r)))]
pub fn max_qty_within(cap: Quote, price: Price) -> Option<Qty> {
    if cap.0 < 0 || price.0 <= 0 {
        return None;
    }
    // cap · S < 2^63 · 2^27: exact in i128.
    let q = i128::from(cap.0) * i128::from(QTY_SCALE) / i128::from(price.0);
    Some(Qty(i64::try_from(q).unwrap_or(i64::MAX)))
}

/// Drawdown kill switch: `(peak − equity) · 10_000 ≥ max_drawdown_bps · peak`,
/// exact in `i128`.
///
/// Meaningful for `peak_equity > 0`, which [`check_order`] enforces; defined
/// and panic-free for every input. An equity above the peak gives a negative
/// drop, which never trips. |peak − equity| < 2^64 and the bps factors are
/// < 2^32, so both products fit in `i128` with room to spare.
#[must_use]
pub fn kill_switch_tripped(account: &Account, limits: &Limits) -> bool {
    let peak = i128::from(account.peak_equity.0);
    let drop = peak - i128::from(account.equity.0);
    drop * i128::from(BPS_SCALE) >= i128::from(limits.max_drawdown_bps) * peak
}

/// Daily loss limit: `day_start_equity − equity ≥ max_daily_loss`, exact in
/// `i128`.
#[must_use]
pub fn daily_loss_tripped(account: &Account, limits: &Limits) -> bool {
    i128::from(account.day_start_equity.0) - i128::from(account.equity.0)
        >= i128::from(limits.max_daily_loss.0)
}

/// Preconditions of [`check_order`]; see its documentation.
fn inputs_valid(account: &Account, limits: &Limits, request: &OrderRequest) -> bool {
    request.qty.0 > 0
        && request.price.0 > 0
        && limits.lot_step.0 > 0
        && limits.max_position.0 >= 0
        && limits.max_order.0 >= 0
        && limits.min_order.0 >= 0
        && limits.max_daily_loss.0 >= 0
        && limits.min_order <= limits.max_order
        && limits.max_drawdown_bps <= BPS_SCALE
        && account.position.0 >= 0
        && account.peak_equity.0 > 0
}

/// Largest multiple of `lot` that is ≤ `qty`. Requires `qty ≥ 0` and
/// `lot > 0`, so neither `%` nor `-` can overflow.
#[cfg_attr(kani, kani::requires(qty >= 0 && lot > 0))]
#[cfg_attr(kani, kani::ensures(|r: &i64| spec::lot_spec(qty, lot, *r)))]
fn round_down_to_lot(qty: i64, lot: i64) -> i64 {
    qty - qty % lot
}

/// Shared tail of both sides: reject an empty or sub-minimum order, else
/// approve it with its ceil-notional.
fn approve(side: Side, qty: i64, price: Price, min_order: Quote) -> Result<Approved, Rejection> {
    if qty == 0 {
        return Err(Rejection::BelowMinimum);
    }
    let qty = Qty(qty);
    if notional_floor(qty, price).ok_or(Rejection::Overflow)? < min_order {
        return Err(Rejection::BelowMinimum);
    }
    let notional = notional_ceil(qty, price).ok_or(Rejection::Overflow)?;
    Ok(Approved {
        side,
        qty,
        notional,
    })
}

/// Checks one order against the account and the limits, and returns the
/// largest acceptable order no bigger than the request, or why there is none.
///
/// The function is pure, integer-only, loop-free and panic-free. Checks run
/// in this order; the first failure wins.
///
/// 1. [`Rejection::InvalidInput`] unless all of: `qty > 0`, `price > 0`,
///    `lot_step > 0`; `max_position`, `max_order`, `min_order`,
///    `max_daily_loss` all `≥ 0`; `min_order ≤ max_order`;
///    `max_drawdown_bps ≤ 10_000`; `position ≥ 0`; `peak_equity > 0`.
///    `min_order ≤ max_order` and `max_drawdown_bps ≤ 10_000` are static
///    configuration, so a bad value fails the first order of a session
///    rather than trapping a position later. Of the runtime account state,
///    only `position` and `peak_equity` are checked (a non-positive peak
///    fails closed, sells included). `equity`, `day_start_equity` and
///    `bars_since_order` are deliberately *not* validated: an equity above a
///    stale peak only means no drawdown, and must not block a sell.
/// 2. **Sell** (risk-reducing, so never gated by the kill switch, the daily
///    loss limit or the cooldown): [`Rejection::NoPosition`] if the position
///    is 0; otherwise `qty = min(request, position, max_qty_within(max_order))`
///    rounded down to the lot.
/// 3. **Buy**: [`Rejection::KillSwitch`], then [`Rejection::DailyLoss`], then
///    [`Rejection::Cooldown`] (`bars_since_order < cooldown_bars`). Then
///    `room = max_position − notional_ceil(position)`, and
///    [`Rejection::PositionCap`] if `room ≤ 0`; otherwise
///    `qty = min(request, max_qty_within(min(room, max_order)))` rounded down
///    to the lot, and [`Rejection::Overflow`] if `position + qty` overflows.
/// 4. Both sides: [`Rejection::BelowMinimum`] if `qty == 0` or
///    `notional_floor(qty) < min_order`. Any helper returning `None` gives
///    [`Rejection::Overflow`]. Only two overflow sites are reachable: the
///    held notional and `position + qty`. The approved qty fits a cap
///    ≤ `i64::MAX`, so its own notionals always fit; those checks stay as
///    defence in depth.
///
/// Because ceil is subadditive, `notional_ceil(position + qty) ≤
/// notional_ceil(position) + notional_ceil(qty) ≤ max_position` after a buy.
///
/// # Errors
///
/// Returns the first [`Rejection`] met in the order above.
pub fn check_order(
    account: &Account,
    limits: &Limits,
    request: &OrderRequest,
) -> Result<Approved, Rejection> {
    if !inputs_valid(account, limits, request) {
        return Err(Rejection::InvalidInput);
    }
    match request.side {
        Side::Sell => check_sell(account, limits, request),
        Side::Buy => check_buy(account, limits, request),
    }
}

fn check_sell(
    account: &Account,
    limits: &Limits,
    request: &OrderRequest,
) -> Result<Approved, Rejection> {
    let position = account.position.0;
    if position == 0 {
        return Err(Rejection::NoPosition);
    }
    let order_cap = max_qty_within(limits.max_order, request.price).ok_or(Rejection::Overflow)?;
    let qty = request.qty.0.min(position).min(order_cap.0);
    let qty = round_down_to_lot(qty, limits.lot_step.0);
    approve(Side::Sell, qty, request.price, limits.min_order)
}

fn check_buy(
    account: &Account,
    limits: &Limits,
    request: &OrderRequest,
) -> Result<Approved, Rejection> {
    if kill_switch_tripped(account, limits) {
        return Err(Rejection::KillSwitch);
    }
    if daily_loss_tripped(account, limits) {
        return Err(Rejection::DailyLoss);
    }
    if account.bars_since_order < limits.cooldown_bars {
        return Err(Rejection::Cooldown);
    }
    let held = notional_ceil(account.position, request.price).ok_or(Rejection::Overflow)?;
    // Both operands are ≥ 0, so the difference cannot overflow.
    let room = limits.max_position.0 - held.0;
    if room <= 0 {
        return Err(Rejection::PositionCap);
    }
    let cap = Quote(room.min(limits.max_order.0));
    let cap_qty = max_qty_within(cap, request.price).ok_or(Rejection::Overflow)?;
    let qty = round_down_to_lot(request.qty.0.min(cap_qty.0), limits.lot_step.0);
    if account.position.0.checked_add(qty).is_none() {
        return Err(Rejection::Overflow);
    }
    approve(Side::Buy, qty, request.price, limits.min_order)
}
