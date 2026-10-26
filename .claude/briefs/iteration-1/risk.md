# Brief — finish `crates/risk` (iteration 1)

- **Agent:** `rust-engineer` (opus, high). It runs in parallel with `backtest.md`.
- **Motto:** we should be more ambitious, in the proofs and tests, never in the claims.

## Status at hand-off (2026-10-25)

A first run was stopped, before its own review and report, when orchestration moved to a fresh session. It left a draft:
- `src/lib.rs` and `src/proofs.rs` (Kani harnesses);
- `tests/examples.rs` and `tests/invariants.rs`.

`cargo test -p risk` at hand-off:
- `examples`: 16 passed.
- `invariants`: 12 passed and 1 failed. `generators_reach_every_outcome` panics at `tests/invariants.rs:485`; it is a coverage check of the proptest generators.
- The run was adding an independent reference oracle: a binary search, set against the closed form of `max_qty_within`.

Kani has never been run, and neither have clippy and fmt.

**Start from the draft.**
1. Find out why the coverage check fails, and fix the generators or the kernel, whichever is wrong.
2. Review everything against this brief, and finish the oracle.
3. Make sure the harnesses would compile under `cargo kani`.

## Goal

`crates/risk` is the last line of defense between any model and an exchange. It is a pure order checker: integer-only and panic-free. It ships with property tests now and Kani proof harnesses for WSL2/CI. `docs/DESIGN.md` §6 is the contract.

## Scope and constraints

- **Files.** Only touch `crates/risk/src/**` and `crates/risk/tests/**`.
- **Dependencies.** No dependency and no edit to any `Cargo.toml` or `Cargo.lock`. The only dev-dependency is `proptest = "1.12"`.
- **Errors.** Write `Display` and `std::error::Error` by hand. `cfg(kani)` is declared in the workspace `check-cfg`.
- **Shared workspace.** Another `rust-engineer` works in `crates/backtest` at the same time. `crates/ols-core` is finished: do not touch it.
  - Never run `cargo fmt --all`.
  - Use `cargo fmt -p risk`, `cargo clippy -p risk --all-targets -- -D warnings` and `cargo test -p risk`.
- **Git and docs.** Run no git command that changes state, do not commit, and do not edit `docs/`.
- **Kani.** It runs only on Linux. If WSL2 has Kani installed, run `cargo kani -p risk` there. Otherwise, write the harnesses so they compile under it, and say that you could not run them.

## Units

- `QTY_SCALE: i64 = 100_000_000`. One base unit is 1e8 qty units, so one qty unit is a satoshi for BTC.
- `QUOTE_SCALE: i64 = 1_000_000`. One USDT is 1e6 quote units.
- A price is the quote units paid for one whole base unit, scaled by QUOTE_SCALE. Notional = qty·price / QTY_SCALE, computed in i128.

## API (names are binding)

- **Newtypes** `Qty(pub i64)`, `Price(pub i64)` and `Quote(pub i64)`.
  - Derive Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord and Hash.
  - Add `#[cfg_attr(kani, derive(kani::Arbitrary))]`, as on every input type.
- `Side { Buy, Sell }`.
- `Limits { max_position: Quote, max_order: Quote, min_order: Quote, lot_step: Qty, max_daily_loss: Quote, max_drawdown_bps: u32, cooldown_bars: u32 }`.
- `Account { equity: Quote, peak_equity: Quote, day_start_equity: Quote, position: Qty, bars_since_order: u32 }`.
- `OrderRequest { side, qty: Qty, price: Price }`.
- `Approved { side, qty: Qty, notional: Quote /* ceil */ }`.
- `Rejection { InvalidInput, KillSwitch, DailyLoss, Cooldown, PositionCap, NoPosition, BelowMinimum, Overflow }`.
- **Helpers** (pub):
  - `notional_floor(Qty, Price) -> Option<Quote>` and `notional_ceil(..)`. Both return `None` on a negative input or an i64 overflow.
  - `max_qty_within(cap: Quote, price: Price) -> Option<Qty>` = floor(cap·QTY_SCALE / price). This is the largest qty whose ceil-notional is ≤ cap.
  - `kill_switch_tripped(&Account, &Limits) -> bool`: (peak − equity)·10_000 ≥ max_drawdown_bps·peak, in i128.
  - `daily_loss_tripped(..)`: day_start − equity ≥ max_daily_loss.
- **`pub fn check_order(&Account, &Limits, &OrderRequest) -> Result<Approved, Rejection>`.** It is loop-free, so Kani needs no unwind bounds. Checks run in this order:
  1. **InvalidInput** unless qty > 0, price > 0, lot_step > 0, every Quote limit ≥ 0, position ≥ 0 and peak_equity > 0. Refine this list if needed, and document it.
  2. **Sell.** Sells reduce risk, so they stay allowed after the kill switch, the daily loss or during the cooldown.
     - A position of 0 → NoPosition.
     - qty = min(request, position, max_qty_within(max_order)), rounded down to the lot.
     - qty == 0 or notional_floor < min_order → BelowMinimum.
  3. **Buy.**
     - KillSwitch, then DailyLoss, then Cooldown (bars_since_order < cooldown_bars).
     - room = max_position − notional_ceil(position). If room ≤ 0 → PositionCap.
     - qty = min(request, max_qty_within(min(room, max_order))), rounded down to the lot.
     - `checked_add` on the position fails → Overflow.
     - qty == 0 or notional_floor < min_order → BelowMinimum.
  4. A helper that returns `None` → Overflow.

## Invariants

Each one needs a Kani harness in `#[cfg(kani)] mod proofs`. Each one also needs a proptest with at least 4096 cases over the full i64 and u32 ranges, extremes included.

- **P1.** Every Approved satisfies all of:
  - qty > 0 and qty % lot_step == 0;
  - qty ≤ request.qty;
  - notional == notional_ceil(qty), and notional ≤ max_order;
  - notional_floor(qty) ≥ min_order.
- **P2.** After an approved Buy, notional_ceil(position + qty) ≤ max_position.
- **P3.** An approved Sell has qty ≤ position: no shorting.
- **P4.** A tripped kill switch, a tripped daily loss or an active cooldown means no approved Buy.
- **P5.** A Sell is never rejected with KillSwitch, DailyLoss or Cooldown.
- **P6.** `check_order` and every helper never panic, on any input.
- **P7.** `max_qty_within` is exact: q fits the cap, and q + 1 does not whenever q + 1 is representable.

Kani harnesses:
- One harness with fully arbitrary inputs, for P6, also asserting P1–P5.
- If i128 division looks too heavy for CBMC, add bounded harnesses with documented `kani::assume` domains, for example price ≤ 1e12 and qty ≤ 1e16.
- Say which harness proves what. A harness whose assumptions exclude every interesting input proves nothing.

## Worked example (unit test, and a doc example in lib.rs)

- **Setup.** A flat account with 100 USDT of equity.
- **Limits.** max_order 50 USDT, max_position 100 USDT, min_order 5 USDT, lot_step 0.00001 BTC = Qty(1_000).
- **Order.** A Buy of 0.001 BTC = Qty(100_000) at 67,000 USDT = Price(67_000_000_000).
- **Expected.** `Approved { Buy, Qty(74_000), Quote(49_580_000) }`. max_qty_within(50 USDT) is 74_626 sats, which rounds down to the lot as 74_000.

Every `Rejection` variant also gets a unit test.

## Done when

- `cargo test -p risk` passes.
- `cargo clippy -p risk --all-targets -- -D warnings` passes.
- `cargo fmt -p risk --check` passes.
- The harnesses are written, and run if Kani is available.

## Report

Under 300 words, in English, following your role file:
- files and public API;
- the invariants, with test counts;
- what is unverified (Kani);
- any deviation from this brief, and why;
- one bolder next step.
