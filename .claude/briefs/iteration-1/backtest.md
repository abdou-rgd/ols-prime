# Brief — finish `crates/backtest` (iteration 1)

- **Agent:** `rust-engineer` (opus, high). It runs in parallel with `risk.md`.
- **Motto:** we should be more ambitious, in the tests, never in the claims.

## Status at hand-off (2026-10-25)

A first run was stopped, before its own review and report, when orchestration moved to a fresh session. It left a draft:
- `src/{lib,error,types,sim,metrics}.rs`;
- `tests/{common/mod.rs,errors.rs,execution.rs,metrics.rs,properties.rs}`.

`cargo test -p backtest` gives 37 passed, 0 failed:

| Test target | Passed |
|---|---|
| unit | 1 |
| errors | 7 |
| execution | 14 |
| metrics | 7 |
| properties | 8 |

Clippy and fmt were not run, and nobody has checked the draft against this brief.

**Start from the draft.** Review it line by line against the sections below, fix every deviation and add what is missing. Rewrite only what is wrong.

## Goal

`crates/backtest` is the bar-based simulator for long/flat spot strategies. Every baseline and model of the bench will be scored by it. Correctness beats speed, and look-ahead must be impossible by construction.

## Scope and constraints

- **Files.** Only touch `crates/backtest/src/**` and `crates/backtest/tests/**`.
- **Dependencies.** No new dependency and no edit to any `Cargo.toml` or `Cargo.lock`.
  - Available: `thiserror = "2.0.21"`, and `proptest = "1.12"` as a dev-dependency.
  - Workspace settings: edition 2024, rust-version 1.89, `unsafe_code = "forbid"`.
- **Shared workspace.** Another `rust-engineer` works in `crates/risk` at the same time. `crates/ols-core` is finished: do not touch it.
  - Never run `cargo fmt --all`.
  - Use `cargo fmt -p backtest`, `cargo clippy -p backtest --all-targets -- -D warnings` and `cargo test -p backtest`.
  - "Blocking waiting for file lock" only means another build is running.
- **Git.** The repo has no commit yet. Run no git command that changes state, and do not commit.
- **Docs.** Do not edit `docs/`. Document the semantics in the crate's doc comments.

## API (names are binding; internals are yours)

- `Bar { open_time: i64 /* ms UTC */, open, high, low, close, volume: f64 }`.
- `SymbolFilters { min_notional: f64, step_size: f64 }`.
  - Consts: `NONE` (0, 0; a step of 0 means no rounding), `BINANCE_BTCUSDT` (5.0, 0.00001), `BINANCE_ETHUSDT` (5.0, 0.0001).
  - Document them as a snapshot to re-check against exchangeInfo.
- `Liquidity { Maker, Taker }`.
- `FeeSchedule { name: &'static str, maker: f64, taker: f64 }`.
  - Consts: `ZERO`, `BINANCE_SPOT` (0.001/0.001), `BINANCE_SPOT_BNB` (0.00075/0.00075), `COINBASE_ADVANCED_ENTRY` (0.0025/0.005), `KRAKEN_PRO_ENTRY` (0.004/0.008).
  - `fn rate(&self, Liquidity) -> f64`.
- `SimConfig { initial_cash, fees: FeeSchedule, liquidity, slippage_bps, filters, rebalance_threshold }`.
  - All fields are pub.
  - `SimConfig::new(initial_cash, fees)` defaults to Taker, 2.0 bps (the protocol's slippage), `NONE` filters and a threshold of 0.0.
- `pub fn simulate(bars: &[Bar], targets: &[f64], cfg: &SimConfig) -> Result<SimResult, SimError>`.
- `Side { Buy, Sell }`.
- `Fill { bar: usize, time: i64, side, qty, price /* executed, incl. slippage */, notional, fee }`.
- `SimResult { equity, position, cash: Vec<f64> /* one per bar */, fills: Vec<Fill>, fees_paid: f64 }`.
- `SimError` (thiserror). It covers: empty input, length mismatch, invalid bar (index), non-increasing time (index), invalid target (index, value) and invalid config.
- `pub mod metrics`:
  - `periods_per_year(bar_ms: i64) -> f64`. Crypto trades every day, so this is 365 × 86_400_000 / bar_ms.
  - `max_drawdown(equity: &[f64]) -> f64`, in [0, 1].
  - `compute(result: &SimResult, periods_per_year: f64) -> Metrics`. Metrics holds:
    - periods;
    - total and annualized return;
    - annualized volatility (sample std of per-period simple returns);
    - `sharpe: Option<f64>` (rf = 0) and `sortino: Option<f64>`, each `None` when undefined;
    - max drawdown;
    - annualized turnover (traded notional / mean equity);
    - exposure (fraction of bars with a position > 0);
    - fees paid.

## Semantics (binding)

1. **Targets.** `targets[t] ∈ [0, 1]` is the fraction of equity to hold, decided at the close of bar t.
   - It executes at the **open of bar t+1**. With s = slippage_bps / 1e4, buys fill at `open·(1+s)` and sells at `open·(1−s)`.
   - `targets[last]` is validated but never executed.
2. **Execution at an open.** Equity at the open is `cash + position·open`.
   - **Full exit.** A target of 0 with a position sells the whole position, with no threshold.
   - **Rebalance.** Otherwise, trade only if |target − current weight| > rebalance_threshold. The desired value is target × equity at the open.
   - **Buy qty** = min(Δvalue / exec_price, cash / (exec_price·(1+fee))).
   - **Sell qty** = min(|Δvalue| / open, position).
   - Floor the qty to the step. Skip the order if qty == 0 or qty·exec_price < min_notional.
3. **Fees.** Paid in quote, once per fill: fee = qty·exec_price·rate. A buy costs qty·exec·(1+rate); a sell yields qty·exec·(1−rate).
4. **Marking.** `equity[t] = cash[t] + position[t]·close[t]`, after any trade at open t. Bar 0 has no trade, so equity[0] = initial_cash.
5. **Invariants.** Cash ≥ 0 and position ≥ 0 at every bar: no leverage, no shorting. Handle float rounding explicitly: after flooring, a buy must never cost more than the cash.
6. **Validation.**
   - Bars: non-empty; open_time strictly increasing; OHLC finite and > 0; low ≤ min(open, close); high ≥ max(open, close); volume finite and ≥ 0.
   - Targets: finite, in [0, 1], same length as the bars.
   - Config: cash finite and > 0; rates in [0, 1); slippage in [0, 10_000) bps; threshold in [0, 1); filters finite and ≥ 0.

## Tests that must exist and pass

- **Flat.** All targets at 0 give constant equity, no fills and zero fees.
- **Buy and hold, closed form.** With targets all 1, `NONE` filters and zero slippage: `equity[last] = cash·close[last] / (open[1]·(1+fee))` to 1e-12 relative, from exactly one fill at bar 1.
- **Round trip at constant prices.** Buy, then sell. Final equity = cash·(1−s)(1−f) / ((1+s)(1+f)).
- **Next-open execution.** A target change at t fills at bar t+1, at open[t+1]·(1±s). Editing high, low or close of bar t+1 does not change that fill.
- **Bookkeeping.**
  - The last target is ignored.
  - Fees are counted once: fees_paid = Σ fill.fee = Σ notional·rate.
  - Min notional skips orders.
  - Every fill qty is a multiple of the step.
- **Errors.** Every `SimError` variant is reachable.
- **Metrics on hand-computed series.**
  - max_drawdown([100, 120, 90, 130]) = 0.25.
  - periods_per_year(3_600_000) = 8760.
  - A constant equity gives `sharpe == None`.
- **Proptest, future invariance.** Take random bars, targets and k. Perturb bars[k+1..] and targets[k..]. Then equity[..=k], position[..=k], cash[..=k] and the fills with bar ≤ k are **bitwise** identical.
- **Proptest, solvency.** Cash, position and equity are ≥ 0 at every bar, under random fees, slippage, filters and targets.
- **Proptest, price tracking.** With zero costs and no filters, all-1 targets track the price exactly: equity[t] = cash·close[t] / open[1] for t ≥ 1.

## Done when

- `cargo test -p backtest` passes.
- `cargo clippy -p backtest --all-targets -- -D warnings` passes.
- `cargo fmt -p backtest --check` passes.
- The crate docs state the execution model.

## Report

Under 300 words, in English, following your role file:
- files and public API;
- the invariants tested, with pass and fail counts;
- what you could not verify;
- any deviation from this brief, and why;
- one bolder next step.
