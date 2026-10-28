//! Kani proof harnesses for the risk kernel (`cargo kani -p risk`, Linux).
//!
//! Kani checks every harness for panics, arithmetic overflow and division by
//! zero on top of the explicit assertions, so each harness also proves P6
//! (no panic) for the code it reaches. Nothing here loops, so no unwind
//! bound is needed.
//!
//! Invariants (`docs/DESIGN.md` §6, numbered as in the iteration-1 brief):
//! - P1 every approval is positive, a lot multiple, at most the request,
//!   carries `notional_ceil(qty)`, fits `max_order` and meets `min_order`;
//! - P2 an approved buy keeps `notional_ceil(position + qty) ≤ max_position`;
//! - P3 an approved sell has `qty ≤ position`;
//! - P4 kill switch, daily loss or cooldown ⇒ no approved buy;
//! - P5 a sell is never rejected with `KillSwitch`, `DailyLoss` or `Cooldown`;
//! - P6 no panic anywhere;
//! - P7 `max_qty_within` is exact.
//!
//! `assert_contract` also checks that `Overflow` only comes from the two
//! reachable sites (the held notional and `position + qty`).
//!
//! # Cost
//!
//! CBMC bit-blasts every `i128` product and quotient. The expensive one is
//! `max_qty_within`'s 128-bit division by a symbolic price, which every
//! harness through `check_order` contains. The `check_order` contract is
//! stated on three domains: `_full` (all inputs), [`MARKET`] and [`TINY`].
//! The bounded domains only bound magnitudes: signs stay free, so the
//! `InvalidInput` paths are explored, and every `u32` stays unconstrained.
//!
//! Measured in iteration 1 (Kani 0.68.0, CaDiCaL, one harness per process, a
//! 60-minute cap, up to 12 processes sharing a 12-thread, 15 GB machine):
//! only `buy_gates_block_every_buy` and `sells_are_never_gated` finished, in
//! 795 s and 29 s in that shared queue, and in 611 s and 14 s alone on the
//! final code. Those two are the ones CI proves (the `proven` job of
//! `.github/workflows/kani.yml`).
//!
//! Bounding magnitudes did not help: neither `kani::assume` nor narrow
//! sign-extended draws changed the circuit's size. `round_down_to_lot_full`
//! has 20,035 variables and 98,832 clauses, and its bounded twins about 20,210
//! and 99,000 at every bound from 2^8 to 2^32. The `check_order` circuits are
//! close too: 1,425,364 variables for `_full`, 1,317,974 for [`MARKET`] and
//! 1,207,018 for [`TINY`], whose harnesses also check fewer covers. So the
//! bounded harnesses time out like the `_full` ones. The table is in
//! `docs/ITERATIONS.md`.
//!
//! # Contracts (iteration 2)
//!
//! `product`, `notional_floor`, `notional_ceil`, `max_qty_within` and
//! `round_down_to_lot` carry Kani function contracts, written in
//! `src/spec.rs` without division except for the lot multiple. Each
//! `*_contract` harness proves one with `proof_for_contract`; the
//! `check_order_*`, `approval_*` and `sell_outcome_*` harnesses replace the
//! helpers by their contracts with `stub_verified`. Run every harness with
//! `-Z function-contracts -Z stubbing --no-assert-contracts`.
//!
//! A `stub_verified` verdict is conditional: it holds only as far as the
//! contracts it uses are proven. In iteration 2 (local runs, shared machine,
//! 40-minute cap) `product_contract` and `exact_never_wraps` finished; the
//! contracts of `notional_floor`, `notional_ceil`, `max_qty_within` and
//! `round_down_to_lot` did not, so they are assumed, with the unit tests of
//! `spec.rs` as sampled evidence. Stubbing halves the `check_order`
//! circuits (about 525,000 variables for [`TINY`] against 1.21 M), but what
//! remains is nonlinear: the lot multiple `(n·l) mod l = 0`, the order cap
//! `a ≤ b ⇒ a·p ≤ b·p`, and P2's `(a + b)·p = a·p + b·p`, each across
//! distinct multiplier circuits. [`assert_core`] leaves those three out.
//!
//! # Covers
//!
//! Every `kani::cover!` below is expected to be SATISFIED. One that is not
//! means the harness's domain or assumptions exclude a case it claims to
//! check, and should fail CI.

use super::*;

/// Magnitude bounds for a bounded harness: |field| ≤ bound.
struct Domain {
    price: i64,
    qty: i64,
    quote: i64,
    lot: i64,
}

/// Real markets with headroom: prices up to 2^40 ≈ 1.1e12 (1.1M USDT per
/// whole unit), quantities and positions up to 2^60 ≈ 1.15e18 (1.15e10
/// whole units: a meme-coin bag), quote amounts up to 2^50 ≈ 1.1e15 (1.1e9
/// USDT), lot step up to one whole unit. The held notional can overflow
/// here (2^60 · 2^40 / 1e8 > 2^63); `position + qty` cannot (≤ 2^61).
const MARKET: Domain = Domain {
    price: 1 << 40,
    qty: 1 << 60,
    quote: 1 << 50,
    lot: QTY_SCALE,
};

/// The smallest domain: prices up to 2^24 (16.7 USDT), quantities up to 2^32
/// (42.9 whole units), quote amounts up to 2^32 (4,295 USDT), lots up to
/// 2^16. Every outcome except `Overflow` is reachable (notionals stay below
/// 2^30). It was meant as a fallback small enough for CBMC to finish, but its
/// harnesses ran out of time too (see the module's `# Cost` section).
const TINY: Domain = Domain {
    price: 1 << 24,
    qty: 1 << 32,
    quote: 1 << 32,
    lot: 1 << 16,
};

fn assume_within(d: &Domain, account: &Account, limits: &Limits, request: &OrderRequest) {
    let qty = |q: Qty| -d.qty <= q.0 && q.0 <= d.qty;
    let quote = |q: Quote| -d.quote <= q.0 && q.0 <= d.quote;
    kani::assume(-d.price <= request.price.0 && request.price.0 <= d.price);
    kani::assume(qty(request.qty) && qty(account.position));
    kani::assume(-d.lot <= limits.lot_step.0 && limits.lot_step.0 <= d.lot);
    kani::assume(quote(limits.max_position) && quote(limits.max_order));
    kani::assume(quote(limits.min_order) && quote(limits.max_daily_loss));
    kani::assume(quote(account.equity) && quote(account.peak_equity));
    kani::assume(quote(account.day_start_equity));
}

fn buy_gated(account: &Account, limits: &Limits) -> bool {
    kill_switch_tripped(account, limits)
        || daily_loss_tripped(account, limits)
        || account.bars_since_order < limits.cooldown_bars
}

/// P1–P5 for one call of `check_order`: [`assert_core`] and
/// [`assert_arithmetic`].
fn assert_contract(
    account: &Account,
    limits: &Limits,
    request: &OrderRequest,
    result: Result<Approved, Rejection>,
) {
    assert_core(account, limits, request, result);
    assert_arithmetic(account, limits, request, result);
}

/// The part of P1–P5 that needs no nonlinear arithmetic once the helpers are
/// replaced by their contracts: the side, `0 < qty ≤ request`, the notional
/// is `notional_ceil(qty)`, the floor meets `min_order`, P3, P4, P5, and the
/// two reachable `Overflow` sites.
fn assert_core(
    account: &Account,
    limits: &Limits,
    request: &OrderRequest,
    result: Result<Approved, Rejection>,
) {
    match result {
        Ok(ok) => {
            // P1, without the lot multiple and the order cap.
            assert!(ok.side == request.side);
            assert!(ok.qty.0 > 0);
            assert!(limits.lot_step.0 > 0);
            assert!(ok.qty <= request.qty);
            assert!(notional_ceil(ok.qty, request.price) == Some(ok.notional));
            match notional_floor(ok.qty, request.price) {
                Some(floor) => {
                    assert!(floor >= limits.min_order);
                }
                None => panic!("approved order without a floor notional"),
            }
            match ok.side {
                Side::Buy => {
                    // P4
                    assert!(!buy_gated(account, limits));
                }
                // P3
                Side::Sell => {
                    assert!(ok.qty <= account.position);
                }
            }
        }
        Err(e) => {
            // Overflow is only reported at its two reachable sites: a sell
            // never overflows, and a buy only when the held notional or
            // `position + request.qty` leaves i64.
            if e == Rejection::Overflow {
                assert!(request.side == Side::Buy);
                assert!(
                    notional_ceil(account.position, request.price).is_none()
                        || account.position.0.checked_add(request.qty.0).is_none()
                );
            }
            // P5
            if request.side == Side::Sell {
                assert!(!matches!(
                    e,
                    Rejection::KillSwitch | Rejection::DailyLoss | Rejection::Cooldown
                ));
            }
        }
    }
}

/// The part of P1–P2 that rests on nonlinear integer facts, which SAT
/// solvers find hard at 64 bits: the lot multiple (`(n·l) mod l = 0`), the
/// order cap (`a ≤ b ⇒ a·p ≤ b·p`) and P2 (`(a + b)·p = a·p + b·p`).
fn assert_arithmetic(
    account: &Account,
    limits: &Limits,
    request: &OrderRequest,
    result: Result<Approved, Rejection>,
) {
    if let Ok(ok) = result {
        // P1
        assert!(limits.lot_step.0 > 0);
        assert!(ok.qty.0 % limits.lot_step.0 == 0);
        assert!(ok.notional <= limits.max_order);
        // P2
        if ok.side == Side::Buy {
            match account.position.0.checked_add(ok.qty.0) {
                Some(next) => match notional_ceil(Qty(next), request.price) {
                    Some(n) => {
                        assert!(n <= limits.max_position);
                    }
                    None => panic!("post-buy notional overflows"),
                },
                None => panic!("post-buy position overflows"),
            }
        }
    }
}

/// Both approvals and every rejection except `Overflow` are reachable.
fn cover_outcomes(result: &Result<Approved, Rejection>) {
    kani::cover!(
        matches!(result, Ok(a) if a.side == Side::Buy),
        "approved buy"
    );
    kani::cover!(
        matches!(result, Ok(a) if a.side == Side::Sell),
        "approved sell"
    );
    kani::cover!(*result == Err(Rejection::InvalidInput), "InvalidInput");
    kani::cover!(*result == Err(Rejection::KillSwitch), "KillSwitch");
    kani::cover!(*result == Err(Rejection::DailyLoss), "DailyLoss");
    kani::cover!(*result == Err(Rejection::Cooldown), "Cooldown");
    kani::cover!(*result == Err(Rejection::PositionCap), "PositionCap");
    kani::cover!(*result == Err(Rejection::NoPosition), "NoPosition");
    kani::cover!(*result == Err(Rejection::BelowMinimum), "BelowMinimum");
}

/// The held notional `notional_ceil(position)` does not fit in i64.
fn cover_held_overflow(
    account: &Account,
    request: &OrderRequest,
    result: &Result<Approved, Rejection>,
) {
    kani::cover!(
        *result == Err(Rejection::Overflow)
            && notional_ceil(account.position, request.price).is_none(),
        "Overflow at held notional"
    );
}

/// The held notional fits, but `position + qty` does not.
fn cover_position_overflow(
    account: &Account,
    request: &OrderRequest,
    result: &Result<Approved, Rejection>,
) {
    kani::cover!(
        *result == Err(Rejection::Overflow)
            && notional_ceil(account.position, request.price).is_some(),
        "Overflow at position + qty"
    );
}

// ---- check_order: P1–P6 ---------------------------------------------------

/// P1–P6 on fully arbitrary inputs: the brief's headline harness. Expected
/// to be the heaviest (128-bit symbolic division over full i64 ranges).
#[kani::proof]
#[kani::solver(cadical)]
#[kani::stub_verified(notional_floor)]
#[kani::stub_verified(notional_ceil)]
#[kani::stub_verified(max_qty_within)]
#[kani::stub_verified(round_down_to_lot)]
fn check_order_full() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let request: OrderRequest = kani::any();
    let result = check_order(&account, &limits, &request);
    cover_outcomes(&result);
    cover_held_overflow(&account, &request, &result);
    cover_position_overflow(&account, &request, &result);
    assert_contract(&account, &limits, &request, result);
}

/// P1–P5, and P6 for these inputs, on the [`MARKET`] domain.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::stub_verified(notional_floor)]
#[kani::stub_verified(notional_ceil)]
#[kani::stub_verified(max_qty_within)]
#[kani::stub_verified(round_down_to_lot)]
fn check_order_market() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let request: OrderRequest = kani::any();
    assume_within(&MARKET, &account, &limits, &request);
    let result = check_order(&account, &limits, &request);
    cover_outcomes(&result);
    cover_held_overflow(&account, &request, &result);
    assert_contract(&account, &limits, &request, result);
}

/// P1–P5, and P6 for these inputs, on the [`TINY`] domain.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::stub_verified(notional_floor)]
#[kani::stub_verified(notional_ceil)]
#[kani::stub_verified(max_qty_within)]
#[kani::stub_verified(round_down_to_lot)]
fn check_order_tiny() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let request: OrderRequest = kani::any();
    assume_within(&TINY, &account, &limits, &request);
    let result = check_order(&account, &limits, &request);
    cover_outcomes(&result);
    assert_contract(&account, &limits, &request, result);
}

/// [`assert_core`] on fully arbitrary inputs, with every helper replaced by
/// its contract. Sound only as far as those contracts are proven: see the
/// `*_contract` harnesses.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::stub_verified(notional_floor)]
#[kani::stub_verified(notional_ceil)]
#[kani::stub_verified(max_qty_within)]
#[kani::stub_verified(round_down_to_lot)]
fn check_order_core_full() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let request: OrderRequest = kani::any();
    let result = check_order(&account, &limits, &request);
    cover_outcomes(&result);
    cover_held_overflow(&account, &request, &result);
    cover_position_overflow(&account, &request, &result);
    assert_core(&account, &limits, &request, result);
}

/// Beyond P1: the approval is the *largest* acceptable lot multiple (one
/// more lot breaks the request, the position, the cap or i64), and
/// re-submitting it returns it unchanged. [`TINY`] domain, since it runs the
/// kernel twice.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::stub_verified(notional_floor)]
#[kani::stub_verified(notional_ceil)]
#[kani::stub_verified(max_qty_within)]
#[kani::stub_verified(round_down_to_lot)]
fn approval_is_maximal_and_stable_tiny() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let request: OrderRequest = kani::any();
    assume_within(&TINY, &account, &limits, &request);
    let result = check_order(&account, &limits, &request);
    kani::cover!(
        matches!(result, Ok(a) if a.side == Side::Buy),
        "approved buy"
    );
    kani::cover!(
        matches!(result, Ok(a) if a.side == Side::Sell),
        "approved sell"
    );
    kani::cover!(
        matches!(result, Ok(a) if a.qty < request.qty),
        "approval smaller than the request"
    );
    if let Ok(ok) = result {
        let (limit, cap) = match request.side {
            Side::Sell => (request.qty.0.min(account.position.0), limits.max_order),
            Side::Buy => {
                let held = notional_ceil(account.position, request.price);
                let room = limits.max_position.0 - held.map_or(i64::MAX, |h| h.0);
                (request.qty.0, Quote(room.min(limits.max_order.0)))
            }
        };
        // No overflow is reachable on TINY, so the bound is one of these.
        let more = ok.qty.0 + limits.lot_step.0;
        assert!(more > limit || notional_ceil(Qty(more), request.price).is_none_or(|n| n > cap));
        let again = OrderRequest {
            qty: ok.qty,
            ..request
        };
        assert!(check_order(&account, &limits, &again) == Ok(ok));
    }
}

// ---- Gates: P4 and P5 -----------------------------------------------------

/// P4 on fully arbitrary inputs, sharpened: a gated buy is rejected with
/// `InvalidInput` or with exactly the first tripped gate, in the order
/// kill switch, daily loss, cooldown. The arithmetic after the gates is
/// unreachable under the assumption, so this should stay cheap.
#[kani::proof]
fn buy_gates_block_every_buy() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let mut request: OrderRequest = kani::any();
    request.side = Side::Buy;
    let kill = kill_switch_tripped(&account, &limits);
    let daily = daily_loss_tripped(&account, &limits);
    let cool = account.bars_since_order < limits.cooldown_bars;
    kani::assume(kill || daily || cool);
    let result = check_order(&account, &limits, &request);
    let first = if kill {
        Rejection::KillSwitch
    } else if daily {
        Rejection::DailyLoss
    } else {
        Rejection::Cooldown
    };
    assert!(result == Err(Rejection::InvalidInput) || result == Err(first));
    kani::cover!(result == Err(Rejection::InvalidInput), "gated and invalid");
    kani::cover!(result == Err(Rejection::KillSwitch), "KillSwitch");
    kani::cover!(result == Err(Rejection::DailyLoss), "DailyLoss");
    kani::cover!(result == Err(Rejection::Cooldown), "Cooldown");
    kani::cover!(
        kill && daily && cool && result == Err(Rejection::KillSwitch),
        "all three gates tripped, kill switch reported"
    );
}

/// P5 on fully arbitrary inputs. The covers show that a sell is approved
/// with all three buy gates tripped at once: the kill switch cannot trap a
/// position.
#[kani::proof]
fn sells_are_never_gated() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let mut request: OrderRequest = kani::any();
    request.side = Side::Sell;
    let result = check_order(&account, &limits, &request);
    assert!(!matches!(
        result,
        Err(Rejection::KillSwitch | Rejection::DailyLoss | Rejection::Cooldown)
    ));
    let all_gates = kill_switch_tripped(&account, &limits)
        && daily_loss_tripped(&account, &limits)
        && account.bars_since_order < limits.cooldown_bars;
    kani::cover!(
        all_gates && result.is_ok(),
        "sell approved with every gate tripped"
    );
    kani::cover!(result == Err(Rejection::NoPosition), "NoPosition");
    kani::cover!(result == Err(Rejection::BelowMinimum), "BelowMinimum");
}

/// P5, stronger, on fully arbitrary inputs: a sell's outcome depends only on
/// the position, the request, `lot_step`, `max_order` and `min_order`.
/// Swapping equity, peak, day start, bars, `max_position` and every trip wire
/// for any other valid values leaves it unchanged.
#[kani::proof]
#[kani::solver(cadical)]
#[kani::stub_verified(notional_floor)]
#[kani::stub_verified(notional_ceil)]
#[kani::stub_verified(max_qty_within)]
#[kani::stub_verified(round_down_to_lot)]
fn sell_outcome_ignores_trip_wires() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let mut request: OrderRequest = kani::any();
    request.side = Side::Sell;
    let other_account = Account {
        position: account.position,
        ..kani::any()
    };
    let other_limits = Limits {
        max_order: limits.max_order,
        min_order: limits.min_order,
        lot_step: limits.lot_step,
        ..kani::any()
    };
    // The swapped fields must be valid on both sides, else one call is
    // `InvalidInput` for a reason that has nothing to do with selling.
    let valid = |a: &Account, l: &Limits| {
        a.peak_equity.0 > 0
            && l.max_position.0 >= 0
            && l.max_daily_loss.0 >= 0
            && l.max_drawdown_bps <= BPS_SCALE
    };
    kani::assume(valid(&account, &limits) && valid(&other_account, &other_limits));
    let result = check_order(&account, &limits, &request);
    let other = check_order(&other_account, &other_limits, &request);
    assert!(result == other);
    kani::cover!(
        result.is_ok() && buy_gated(&account, &limits) && !buy_gated(&other_account, &other_limits),
        "same approved sell with and without the gates"
    );
}

// ---- Contracts of the helpers ----------------------------------------------

/// `spec::exact` never wraps: the checked 128-bit product of two `i64`
/// exists and equals it. This is what lets the contracts use `wrapping_mul`.
#[kani::proof]
#[kani::solver(cadical)]
fn exact_never_wraps() {
    let a: i64 = kani::any();
    let b: i64 = kani::any();
    assert!(i128::from(a).checked_mul(i128::from(b)) == Some(spec::exact(a, b)));
}

/// `product` meets [`spec::product_spec`].
#[kani::proof_for_contract(product)]
#[kani::solver(cadical)]
fn product_contract() {
    let _ = product(kani::any(), kani::any());
}

/// `notional_floor` meets [`spec::floor_spec`], given `product`'s contract.
#[kani::proof_for_contract(notional_floor)]
#[kani::stub_verified(product)]
#[kani::solver(cadical)]
fn notional_floor_contract() {
    let _ = notional_floor(kani::any(), kani::any());
}

/// `notional_ceil` meets [`spec::ceil_spec`], given `product`'s contract.
#[kani::proof_for_contract(notional_ceil)]
#[kani::stub_verified(product)]
#[kani::solver(cadical)]
fn notional_ceil_contract() {
    let _ = notional_ceil(kani::any(), kani::any());
}

/// `max_qty_within` meets [`spec::max_qty_spec`]: P7 stated without
/// division. Its 128-bit division by a symbolic price is bit-blasted here
/// once, and replaced by the contract everywhere else.
#[kani::proof_for_contract(max_qty_within)]
#[kani::solver(cadical)]
fn max_qty_within_contract() {
    let _ = max_qty_within(kani::any(), kani::any());
}

/// `round_down_to_lot` meets [`spec::lot_spec`] under its precondition.
#[kani::proof_for_contract(round_down_to_lot)]
#[kani::solver(cadical)]
fn round_down_to_lot_contract() {
    let _ = round_down_to_lot(kani::any(), kani::any());
}

// ---- Helpers --------------------------------------------------------------

/// P6 for the notionals, plus their definition: `None` exactly on a negative
/// input or an i64 overflow, `floor·S ≤ q·p < (floor + 1)·S`,
/// `floor ≤ ceil ≤ floor + 1`, and `ceil == floor` iff S divides q·p.
#[kani::proof]
#[kani::solver(cadical)]
fn notional_helpers_full() {
    let qty: Qty = kani::any();
    let price: Price = kani::any();
    let floor = notional_floor(qty, price);
    let ceil = notional_ceil(qty, price);
    if qty.0 < 0 || price.0 < 0 {
        assert!(floor.is_none() && ceil.is_none());
        return;
    }
    let p = i128::from(qty.0) * i128::from(price.0);
    let s = i128::from(QTY_SCALE);
    if let Some(f) = floor {
        let f = i128::from(f.0);
        assert!(f * s <= p && p < (f + 1) * s);
    } else {
        assert!(p / s > i128::from(i64::MAX));
    }
    if let Some(c) = ceil {
        let c = i128::from(c.0);
        assert!((c - 1) * s < p && p <= c * s);
        assert!(floor.is_some());
    } else {
        assert!((p + s - 1) / s > i128::from(i64::MAX));
    }
    if let (Some(f), Some(c)) = (floor, ceil) {
        assert!(f <= c && c.0 - f.0 <= 1);
        assert!((c == f) == (p % s == 0));
    }
    kani::cover!(floor.is_none(), "floor overflows");
    kani::cover!(floor.is_some() && ceil.is_none(), "only ceil overflows");
    kani::cover!(
        matches!((floor, ceil), (Some(f), Some(c)) if f == c),
        "exact"
    );
    kani::cover!(
        matches!((floor, ceil), (Some(f), Some(c)) if f != c),
        "rounded"
    );
}

/// The lemma behind P2: ceil is subadditive, so
/// `ceil(position + qty) ≤ ceil(position) + ceil(qty) ≤ max_position`.
#[kani::proof]
#[kani::solver(cadical)]
fn ceil_is_subadditive_full() {
    let a: Qty = kani::any();
    let b: Qty = kani::any();
    let price: Price = kani::any();
    let sum = a.0.checked_add(b.0);
    kani::assume(sum.is_some());
    let (ca, cb) = (notional_ceil(a, price), notional_ceil(b, price));
    let cs = sum.and_then(|s| notional_ceil(Qty(s), price));
    if let (Some(ca), Some(cb), Some(cs)) = (ca, cb, cs) {
        assert!(i128::from(cs.0) <= i128::from(ca.0) + i128::from(cb.0));
    }
    kani::cover!(
        matches!((ca, cb, cs), (Some(x), Some(y), Some(z)) if z.0 < x.0 + y.0),
        "strictly subadditive"
    );
}

/// P7 on fully arbitrary inputs: the result fits the cap and one more qty
/// unit does not. Heavy: 128-bit division by a symbolic price.
#[kani::proof]
#[kani::solver(cadical)]
fn max_qty_within_exact_full() {
    let cap: Quote = kani::any();
    let price: Price = kani::any();
    assert_max_qty_exact(cap, price);
}

/// P7 on |cap| ≤ 2^50 and |price| ≤ 2^40 (the [`MARKET`] domain).
#[kani::proof]
#[kani::solver(cadical)]
fn max_qty_within_exact_market() {
    let cap: Quote = kani::any();
    let price: Price = kani::any();
    kani::assume(-MARKET.quote <= cap.0 && cap.0 <= MARKET.quote);
    kani::assume(-MARKET.price <= price.0 && price.0 <= MARKET.price);
    assert_max_qty_exact(cap, price);
}

fn assert_max_qty_exact(cap: Quote, price: Price) {
    let result = max_qty_within(cap, price);
    match result {
        None => {
            assert!(cap.0 < 0 || price.0 <= 0);
        }
        Some(q) => {
            assert!(cap.0 >= 0 && price.0 > 0);
            assert!(q.0 >= 0);
            match notional_ceil(q, price) {
                Some(n) => {
                    assert!(n <= cap);
                }
                None => panic!("max_qty_within returned a qty with no notional"),
            }
            if let Some(next) = q.0.checked_add(1) {
                // An overflowing notional exceeds every i64 cap.
                assert!(notional_ceil(Qty(next), price).is_none_or(|n| n > cap));
            }
        }
    }
    kani::cover!(result.is_none(), "rejected cap or price");
    kani::cover!(matches!(result, Some(q) if q.0 < i64::MAX), "q + 1 checked");
    kani::cover!(
        matches!(result, Some(q) if notional_ceil(q, price) == Some(cap) && q.0 > 0),
        "the cap is met exactly"
    );
}

/// P7 as a Galois connection on the [`MARKET`] domain: for every q ≥ 0,
/// `ceil(q·p/S) ≤ cap ⇔ q ≤ max_qty_within(cap, p)`. This is exactness
/// without assuming monotonicity separately.
#[kani::proof]
#[kani::solver(cadical)]
fn max_qty_within_galois_market() {
    let cap: Quote = kani::any();
    let price: Price = kani::any();
    let q: Qty = kani::any();
    kani::assume(0 <= cap.0 && cap.0 <= MARKET.quote);
    kani::assume(0 < price.0 && price.0 <= MARKET.price);
    kani::assume(0 <= q.0);
    let m = max_qty_within(cap, price);
    let fits = notional_ceil(q, price).is_some_and(|n| n <= cap);
    assert!(matches!(m, Some(m) if fits == (q <= m)));
    kani::cover!(fits, "q fits");
    kani::cover!(!fits && notional_ceil(q, price).is_some(), "q does not fit");
    kani::cover!(matches!(m, Some(m) if q == m), "q is the boundary");
}

/// The private lot rounding: the largest multiple of `lot` that is ≤ qty.
#[kani::proof]
fn round_down_to_lot_full() {
    let qty: i64 = kani::any();
    let lot: i64 = kani::any();
    kani::assume(qty >= 0 && lot > 0);
    let r = round_down_to_lot(qty, lot);
    assert!(0 <= r && r <= qty);
    assert!(r % lot == 0);
    assert!(qty - r < lot);
    kani::cover!(r < qty, "rounded down");
    kani::cover!(r == qty && r > 0, "already a multiple");
}

/// P6 for the trip wires, plus monotonicity: losing more equity never
/// un-trips either of them.
#[kani::proof]
fn tripwires_full() {
    let account: Account = kani::any();
    let limits: Limits = kani::any();
    let lower: Quote = kani::any();
    kani::assume(lower <= account.equity);
    let worse = Account {
        equity: lower,
        ..account
    };
    let kill = kill_switch_tripped(&account, &limits);
    let daily = daily_loss_tripped(&account, &limits);
    assert!(!kill || kill_switch_tripped(&worse, &limits));
    assert!(!daily || daily_loss_tripped(&worse, &limits));
    kani::cover!(kill, "kill switch tripped");
    kani::cover!(
        !kill && kill_switch_tripped(&worse, &limits),
        "a loss trips the kill switch"
    );
    kani::cover!(daily, "daily loss tripped");
    kani::cover!(!daily, "daily loss not tripped");
}
