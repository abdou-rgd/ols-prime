//! Function contracts of the arithmetic helpers, for Kani only.
//!
//! Each predicate below is the full functional specification of one helper:
//! for every input it admits exactly one result, the one the helper returns.
//! That uniqueness is what makes `#[kani::stub_verified]` both sound and
//! precise: a harness that replaces a helper by its contract sees the same
//! value the helper would compute, not merely a value with some property.
//!
//! The predicates avoid division. A quotient is pinned by its bracket,
//! `f·S ≤ p < (f + 1)·S`, which costs CBMC one multiplication by a constant
//! instead of a 128-bit divider. The one exception is the lot multiple in
//! [`round_down_to_lot`](crate::round_down_to_lot), for which no division-free
//! statement exists without a witness.
//!
//! # Exact products
//!
//! [`exact`] multiplies with `wrapping_mul`. For `a, b` in `i64`,
//! `|a·b| ≤ 2^126 < 2^127`, so the 128-bit product never wraps and equals the
//! mathematical one; the harness `proofs::exact_never_wraps` checks this. The
//! wrapping form keeps CBMC from encoding an overflow check, which doubles
//! the width of the multiplier, in every stub that uses the contract.

use crate::{Price, QTY_SCALE, Qty, Quote};

/// `QTY_SCALE` in `i128`.
const S: i128 = QTY_SCALE as i128;

/// The mathematical product `a·b`; see the module documentation.
pub(crate) fn exact(a: i64, b: i64) -> i128 {
    i128::from(a).wrapping_mul(i128::from(b))
}

/// Contract of `product`: `None` iff a factor is negative, else the exact
/// product.
pub(crate) fn product_spec(qty: Qty, price: Price, result: Option<i128>) -> bool {
    match result {
        None => qty.0 < 0 || price.0 < 0,
        Some(p) => qty.0 >= 0 && price.0 >= 0 && p == exact(qty.0, price.0),
    }
}

/// Contract of [`notional_floor`](crate::notional_floor):
/// `f = floor(q·p / S)`, stated as `f·S ≤ q·p < f·S + S`, and `None` iff a
/// factor is negative or `f > i64::MAX`, i.e. `q·p ≥ (i64::MAX + 1)·S`.
pub(crate) fn floor_spec(qty: Qty, price: Price, result: Option<Quote>) -> bool {
    if qty.0 < 0 || price.0 < 0 {
        return result.is_none();
    }
    let p = exact(qty.0, price.0);
    match result {
        None => p >= (i128::from(i64::MAX) + 1) * S,
        Some(f) => {
            let fs = i128::from(f.0) * S;
            fs <= p && p - fs < S
        }
    }
}

/// Contract of [`notional_ceil`](crate::notional_ceil):
/// `c = ceil(q·p / S)`, stated as `c·S − S < q·p ≤ c·S`, and `None` iff a
/// factor is negative or `c > i64::MAX`, i.e. `q·p > i64::MAX·S`.
pub(crate) fn ceil_spec(qty: Qty, price: Price, result: Option<Quote>) -> bool {
    if qty.0 < 0 || price.0 < 0 {
        return result.is_none();
    }
    let p = exact(qty.0, price.0);
    match result {
        None => p > i128::from(i64::MAX) * S,
        Some(c) => {
            let cs = i128::from(c.0) * S;
            p <= cs && cs - p < S
        }
    }
}

/// Contract of [`max_qty_within`](crate::max_qty_within):
/// `q = min(floor(cap·S / p), i64::MAX)`, stated as `q·p ≤ cap·S` and, unless
/// `q` saturated, `cap·S < q·p + p`. `None` iff `cap < 0` or `p ≤ 0`.
pub(crate) fn max_qty_spec(cap: Quote, price: Price, result: Option<Qty>) -> bool {
    if cap.0 < 0 || price.0 <= 0 {
        return result.is_none();
    }
    let Some(q) = result else {
        return false;
    };
    let c = i128::from(cap.0) * S;
    let m = exact(q.0, price.0);
    q.0 >= 0 && m <= c && (q.0 == i64::MAX || c - m < i128::from(price.0))
}

/// Contract of `round_down_to_lot`, under `qty ≥ 0` and `lot > 0`: the
/// largest multiple of `lot` that is at most `qty`.
pub(crate) fn lot_spec(qty: i64, lot: i64, result: i64) -> bool {
    0 <= result && result <= qty && qty - result < lot && result % lot == 0
}

/// The contracts are only as good as their proofs, and the proofs of
/// `notional_floor`, `notional_ceil`, `max_qty_within` and
/// `round_down_to_lot` do not finish in CBMC yet. These tests are the
/// sampled evidence meanwhile: on random inputs, each predicate accepts the
/// helper's result (soundness of the contract) and rejects its neighbours
/// (the contract pins a single value, so a stub cannot pick another).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{max_qty_within, notional_ceil, notional_floor, product, round_down_to_lot};
    use proptest::prelude::*;

    /// Full range, small values, real market magnitudes and the edges.
    fn int() -> impl Strategy<Value = i64> {
        prop_oneof![
            any::<i64>(),
            -3i64..1_000,
            -(1i64 << 40)..(1i64 << 40),
            0i64..(1i64 << 62),
            Just(0i64),
            Just(i64::MAX),
            Just(i64::MIN),
            (i64::MAX - 1_000)..=i64::MAX,
        ]
    }

    /// The candidates a wrong stub could pick instead of `x`.
    fn neighbours(x: i64) -> impl Iterator<Item = i64> {
        wide_neighbours(x.into()).filter_map(|y| i64::try_from(y).ok())
    }

    /// [`neighbours`] for the `i128` result of `product`: also `i128::MAX`.
    fn wide_neighbours(x: i128) -> impl Iterator<Item = i128> {
        [
            x.checked_sub(1),
            x.checked_add(1),
            Some(0),
            Some(i128::from(i64::MAX)),
            Some(i128::MAX),
        ]
        .into_iter()
        .flatten()
        .filter(move |&y| y != x)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4_096))]

        #[test]
        fn product_spec_pins_product(q in int(), p in int()) {
            let r = product(Qty(q), Price(p));
            prop_assert!(product_spec(Qty(q), Price(p), r));
            match r {
                Some(_) => prop_assert!(!product_spec(Qty(q), Price(p), None)),
                // A negative factor admits no product at all, 0 included.
                None => prop_assert!(!product_spec(Qty(q), Price(p), Some(0))),
            }
            for y in wide_neighbours(r.unwrap_or(0)) {
                prop_assert!(!product_spec(Qty(q), Price(p), Some(y)));
            }
        }

        #[test]
        fn floor_spec_pins_notional_floor(q in int(), p in int()) {
            let r = notional_floor(Qty(q), Price(p));
            prop_assert!(floor_spec(Qty(q), Price(p), r));
            if r.is_some() {
                prop_assert!(!floor_spec(Qty(q), Price(p), None));
            }
            let x = r.map_or(0, |f| f.0);
            for y in neighbours(x) {
                prop_assert!(!floor_spec(Qty(q), Price(p), Some(Quote(y))));
            }
        }

        #[test]
        fn ceil_spec_pins_notional_ceil(q in int(), p in int()) {
            let r = notional_ceil(Qty(q), Price(p));
            prop_assert!(ceil_spec(Qty(q), Price(p), r));
            if r.is_some() {
                prop_assert!(!ceil_spec(Qty(q), Price(p), None));
            }
            let x = r.map_or(0, |c| c.0);
            for y in neighbours(x) {
                prop_assert!(!ceil_spec(Qty(q), Price(p), Some(Quote(y))));
            }
        }

        #[test]
        fn max_qty_spec_pins_max_qty_within(c in int(), p in int()) {
            let r = max_qty_within(Quote(c), Price(p));
            prop_assert!(max_qty_spec(Quote(c), Price(p), r));
            if r.is_some() {
                prop_assert!(!max_qty_spec(Quote(c), Price(p), None));
            }
            let x = r.map_or(0, |q| q.0);
            for y in neighbours(x) {
                prop_assert!(!max_qty_spec(Quote(c), Price(p), Some(Qty(y))));
            }
        }

        #[test]
        fn lot_spec_pins_round_down_to_lot(q in int(), l in int()) {
            // Clearing the sign bit maps every draw into the precondition.
            let (q, l) = (q & i64::MAX, (l & i64::MAX).max(1));
            let r = round_down_to_lot(q, l);
            prop_assert!(lot_spec(q, l, r));
            let others = neighbours(r).chain([r.checked_add(l), r.checked_sub(l)].into_iter().flatten());
            for y in others {
                prop_assert!(!lot_spec(q, l, y));
            }
        }
    }
}
