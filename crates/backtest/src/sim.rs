//! The simulator loop. See the crate docs for the execution model.

use crate::error::SimError;
use crate::types::{Bar, Fill, Side, SimConfig, SimResult};

/// An order whose notional is at most this fraction of equity at the open is
/// skipped as floating-point dust.
///
/// Why: a cash-capped buy leaves a residue of a few ulps of cash (the buy is
/// shrunk until its cost fits, see [`buy`]). Without this floor, a target of 1
/// would re-buy that residue at every open, producing one spurious fill per bar
/// with no economic content. 1e-12 is about 4·10³ ulps of equity, far below any
/// real order and far above rounding noise. It never applies to a full exit.
pub(crate) const DUST_FRACTION: f64 = 1e-12;

/// Largest number of lots the book may hold: 2⁵⁰.
///
/// Why: below 2⁵³ every integer is exact in `f64`, so `lots as f64` is exact,
/// each fill quantity is `n·step` rounded once, and the position is
/// `lots·step` rounded once. Staying three binades lower also makes the lot
/// count of a fill recoverable from its quantity: `round(qty / step)` is off
/// by at most 2⁻² and distinct counts give distinct quantities, so an
/// auditor can replay the book in exact integer lots. A buy that would exceed
/// the cap is clamped to it. With a step of 10⁻⁸ the cap is about 1.1·10⁷
/// base units, beyond any real account.
pub(crate) const MAX_LOTS: u64 = 1 << 50;

/// Runs a long/flat spot strategy over `bars`.
///
/// `targets[t]` is the fraction of equity to hold, decided at the close of bar
/// `t`; it executes at the open of bar `t + 1`. See the crate docs for the full
/// execution model.
///
/// # Errors
///
/// Returns a [`SimError`] when the config, the bars or the targets are
/// invalid. Validation happens before any bar is simulated.
pub fn simulate(bars: &[Bar], targets: &[f64], cfg: &SimConfig) -> Result<SimResult, SimError> {
    validate_config(cfg)?;
    validate_bars(bars)?;
    validate_targets(bars.len(), targets)?;

    let ctx = Ctx {
        rate: cfg.fees.rate(cfg.liquidity),
        slip: cfg.slippage_bps / 1e4,
        min_notional: cfg.filters.min_notional,
        step: cfg.filters.step_size,
        threshold: cfg.rebalance_threshold,
    };
    let mut book = Book {
        cash: cfg.initial_cash,
        lots: 0,
        units: 0.0,
    };

    let n = bars.len();
    let mut equity = Vec::with_capacity(n);
    let mut position = Vec::with_capacity(n);
    let mut cash = Vec::with_capacity(n);
    let mut fills = Vec::new();
    let mut fees_paid = 0.0;

    for (t, bar) in bars.iter().enumerate() {
        // Bar 0 has no trade: no decision exists before its open.
        if t > 0
            && let Some(mut fill) = execute(&mut book, &ctx, targets[t - 1], bar.open)
        {
            fill.bar = t;
            fill.time = bar.open_time;
            fees_paid += fill.fee;
            fills.push(fill);
        }
        let pos = book.position(ctx.step);
        cash.push(book.cash);
        position.push(pos);
        equity.push(book.cash + pos * bar.close);
    }

    Ok(SimResult {
        equity,
        position,
        cash,
        fills,
        fees_paid,
    })
}

/// Per-run constants.
struct Ctx {
    rate: f64,
    slip: f64,
    min_notional: f64,
    step: f64,
    threshold: f64,
}

/// Holdings. With a lot step, the position is an integer number of lots, so
/// every fill is an exact step multiple and a full exit leaves no dust. Without
/// a step, it is a plain base quantity.
struct Book {
    cash: f64,
    lots: u64,
    units: f64,
}

impl Book {
    fn position(&self, step: f64) -> f64 {
        if step > 0.0 {
            self.lots as f64 * step
        } else {
            self.units
        }
    }
}

/// Executes the target decided at the previous close, at this bar's `open`.
/// Returns the fill with `bar` and `time` left at 0 for the caller to set.
fn execute(book: &mut Book, ctx: &Ctx, target: f64, open: f64) -> Option<Fill> {
    let pos = book.position(ctx.step);
    let held = pos * open;
    let equity = book.cash + held;
    if equity.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    if target == 0.0 {
        return if pos > 0.0 {
            sell(book, ctx, open, equity, None)
        } else {
            None
        };
    }
    let weight = held / equity;
    if (target - weight).abs() <= ctx.threshold {
        return None;
    }
    let delta = target * equity - held;
    if delta > 0.0 {
        buy(book, ctx, open, equity, delta)
    } else if delta < 0.0 {
        sell(book, ctx, open, equity, Some(-delta))
    } else {
        None
    }
}

/// Buys `value` of quote worth at `open·(1+s)`, capped by cash net of fees.
fn buy(book: &mut Book, ctx: &Ctx, open: f64, equity: f64, value: f64) -> Option<Fill> {
    let price = open * (1.0 + ctx.slip);
    let cap = book.cash / (price * (1.0 + ctx.rate));
    let raw = (value / price).min(cap);
    let cost = |qty: f64| {
        let notional = qty * price;
        let fee = notional * ctx.rate;
        (notional, fee, notional + fee)
    };

    // Shrink the quantity until its cost, as computed in floating point, fits
    // in cash. `raw ≤ cap` holds in exact arithmetic, so this only removes
    // rounding: one lot at most, or a few ulps without a step.
    let (qty, lots) = if ctx.step > 0.0 {
        let mut n = floor_lots(raw, ctx.step).min(MAX_LOTS - book.lots);
        while n > 0 && cost(n as f64 * ctx.step).2 > book.cash {
            n -= 1;
        }
        (n as f64 * ctx.step, n)
    } else {
        let mut qty = raw;
        let mut tries = 0;
        while qty > 0.0 && cost(qty).2 > book.cash {
            tries += 1;
            if tries > 64 {
                return None;
            }
            qty *= 1.0 - 2.0 * f64::EPSILON * f64::from(tries);
        }
        (qty, 0)
    };
    let (notional, fee, total) = cost(qty);
    // `qty <= 0.0` is false for NaN, hence the explicit check. The last test
    // is unreachable after the shrink loops; it stays so that solvency does
    // not rest on their analysis alone.
    if qty.is_nan()
        || qty <= 0.0
        || notional < ctx.min_notional
        || notional <= DUST_FRACTION * equity
        || total > book.cash
    {
        return None;
    }
    if ctx.step > 0.0 {
        // `lots ≤ MAX_LOTS − book.lots` by the clamp above: no overflow.
        book.lots += lots;
    } else {
        book.units += qty;
    }
    // Solvency by construction: `total ≤ cash` was checked on the very values
    // subtracted here, and IEEE subtraction is correctly rounded, so the exact
    // difference `cash − total ≥ 0` cannot round below 0.
    book.cash -= total;
    Some(Fill {
        bar: 0,
        time: 0,
        side: Side::Buy,
        qty,
        price,
        notional,
        fee,
    })
}

/// Sells `value` of quote worth (valued at the open, capped by the position)
/// at `open·(1−s)`; `None` means a full exit.
fn sell(book: &mut Book, ctx: &Ctx, open: f64, equity: f64, value: Option<f64>) -> Option<Fill> {
    let price = open * (1.0 - ctx.slip);
    let pos = book.position(ctx.step);
    let raw = value.map_or(pos, |v| (v / open).min(pos));
    let full = raw >= pos;
    let (qty, lots) = if ctx.step > 0.0 {
        let n = if full {
            book.lots
        } else {
            floor_lots(raw, ctx.step).min(book.lots)
        };
        (n as f64 * ctx.step, n)
    } else {
        (if full { pos } else { raw }, 0)
    };
    let notional = qty * price;
    if qty.is_nan()
        || qty <= 0.0
        || notional < ctx.min_notional
        || (value.is_some() && !full && notional <= DUST_FRACTION * equity)
    {
        return None;
    }
    let fee = notional * ctx.rate;
    if ctx.step > 0.0 {
        book.lots -= lots;
    } else if full {
        book.units = 0.0;
    } else {
        // qty ≤ units, and IEEE subtraction is monotone, so this stays ≥ 0.
        book.units -= qty;
    }
    // `rate < 1`, so the correctly rounded `notional·rate ≤ notional`: the
    // proceeds are ≥ 0 and cash never falls on a sell.
    book.cash += notional - fee;
    Some(Fill {
        bar: 0,
        time: 0,
        side: Side::Sell,
        qty,
        price,
        notional,
        fee,
    })
}

/// Largest `n` with `n · step ≤ q`, where the product is evaluated in floating
/// point exactly as the fill quantity will be.
///
/// `q / step` can round either way across an integer (0.0003 / 0.0001 is
/// 2.9999999999999996), so the naive floor is corrected by one in each
/// direction.
///
/// The result is capped at [`MAX_LOTS`], so `n as f64` is always exact.
fn floor_lots(q: f64, step: f64) -> u64 {
    if q.is_nan() || q <= 0.0 {
        return 0;
    }
    // `as` saturates for out-of-range values; NaN is excluded above.
    let mut n = ((q / step).floor() as u64).min(MAX_LOTS);
    if n < MAX_LOTS && (n + 1) as f64 * step <= q {
        n += 1;
    }
    while n > 0 && n as f64 * step > q {
        n -= 1;
    }
    n
}

fn validate_config(cfg: &SimConfig) -> Result<(), SimError> {
    let bad = |field, value| Err(SimError::InvalidConfig { field, value });
    let unit = 0.0..1.0;
    let nonneg = |x: f64| x.is_finite() && x >= 0.0;
    if !(cfg.initial_cash.is_finite() && cfg.initial_cash > 0.0) {
        return bad("initial_cash", cfg.initial_cash);
    }
    if !unit.contains(&cfg.fees.maker) {
        return bad("fees.maker", cfg.fees.maker);
    }
    if !unit.contains(&cfg.fees.taker) {
        return bad("fees.taker", cfg.fees.taker);
    }
    if !(0.0..10_000.0).contains(&cfg.slippage_bps) {
        return bad("slippage_bps", cfg.slippage_bps);
    }
    if !unit.contains(&cfg.rebalance_threshold) {
        return bad("rebalance_threshold", cfg.rebalance_threshold);
    }
    if !nonneg(cfg.filters.min_notional) {
        return bad("filters.min_notional", cfg.filters.min_notional);
    }
    if !nonneg(cfg.filters.step_size) {
        return bad("filters.step_size", cfg.filters.step_size);
    }
    Ok(())
}

fn validate_bars(bars: &[Bar]) -> Result<(), SimError> {
    if bars.is_empty() {
        return Err(SimError::Empty);
    }
    let price = |p: f64| p.is_finite() && p > 0.0;
    for (index, b) in bars.iter().enumerate() {
        let reason = if !(price(b.open) && price(b.high) && price(b.low) && price(b.close)) {
            Some("OHLC must be finite and > 0")
        } else if b.low > b.open.min(b.close) {
            Some("low above min(open, close)")
        } else if b.high < b.open.max(b.close) {
            Some("high below max(open, close)")
        } else if !(b.volume.is_finite() && b.volume >= 0.0) {
            Some("volume must be finite and >= 0")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(SimError::InvalidBar { index, reason });
        }
        if index > 0 && b.open_time <= bars[index - 1].open_time {
            return Err(SimError::NonIncreasingTime { index });
        }
    }
    Ok(())
}

fn validate_targets(n_bars: usize, targets: &[f64]) -> Result<(), SimError> {
    if targets.len() != n_bars {
        return Err(SimError::LengthMismatch {
            bars: n_bars,
            targets: targets.len(),
        });
    }
    match targets
        .iter()
        .enumerate()
        .find(|&(_, t)| !(0.0..=1.0).contains(t))
    {
        Some((index, &value)) => Err(SimError::InvalidTarget { index, value }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_LOTS, floor_lots};

    #[test]
    fn floor_lots_corrects_division_rounding() {
        // In f64, 3.0 * 0.0001 is 0.00030000000000000003 > 0.0003: three
        // lots genuinely do not fit, while three lots as computed do.
        assert_eq!(floor_lots(0.0003, 0.0001), 2);
        assert_eq!(floor_lots(3.0 * 0.0001, 0.0001), 3);
        assert_eq!(floor_lots(0.000_299_999, 0.0001), 2);
        assert_eq!(floor_lots(0.0, 0.0001), 0);
        assert_eq!(floor_lots(f64::NAN, 0.0001), 0);
        assert_eq!(floor_lots(0.000_05, 0.0001), 0);
        // Absurdly fine steps saturate at MAX_LOTS instead of overflowing.
        assert_eq!(floor_lots(1e12, 1e-8), MAX_LOTS);
        assert_eq!(floor_lots(1.0, f64::MIN_POSITIVE), MAX_LOTS);
        for k in 0..10_000u64 {
            let q = k as f64 * 0.000_01;
            let n = floor_lots(q, 0.000_01);
            assert_eq!(n, k, "q = {q}");
            assert!(n as f64 * 0.000_01 <= q);
        }
    }
}
