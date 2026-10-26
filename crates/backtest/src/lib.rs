//! Bar-based backtest for long/flat spot strategies.
//!
//! Every baseline and model of the bench is scored by [`simulate`]. Its rules
//! are chosen so that look-ahead is impossible by construction, and they are
//! pinned by the tests in `tests/`.
//!
//! # Execution model
//!
//! **Targets.** `targets[t] ∈ [0, 1]` is the fraction of equity to hold,
//! decided at the close of bar `t`. It executes at the **open of bar `t + 1`**,
//! never earlier. `targets[last]` is validated but never executed: its open
//! lies outside the data. With slippage `s = slippage_bps / 10⁴`, buys fill at
//! `open·(1 + s)` and sells at `open·(1 − s)`. Bar `t + 1`'s high, low and
//! close play no part in its fill.
//!
//! **At each open** `t ≥ 1` (bar 0 never trades), with equity
//! `E = cash + position·open` and weight `w = position·open / E`:
//! - target 0 with a position: sell the whole position (a full exit, whatever
//!   the threshold);
//! - otherwise trade only if `|target − w| > rebalance_threshold`, towards the
//!   desired value `target·E`, with `Δ = target·E − position·open`:
//!   - buy `qty = min(Δ / exec, cash / (exec·(1 + fee)))`;
//!   - sell `qty = min(|Δ| / open, position)`;
//! - floor `qty` to the lot step, then skip the order if `qty = 0` or
//!   `qty·exec < min_notional`.
//!
//! At most one fill happens per bar. A full exit below the minimum notional
//! is skipped like any other order: the position stays, as it would on the
//! exchange.
//!
//! **Fees** are paid in quote, once per fill: `fee = qty·exec·rate`, with the
//! rate of [`SimConfig::liquidity`]. A buy costs `qty·exec·(1 + rate)`; a sell
//! yields `qty·exec·(1 − rate)`.
//!
//! **Marking.** `equity[t] = cash[t] + position[t]·close[t]`, after any trade
//! at open `t`. `equity[0] = initial_cash`.
//!
//! # Invariants and floating point
//!
//! - Cash and position are ≥ 0 at every bar: no leverage, no shorting.
//! - A buy never costs more than the cash, as computed in `f64`: after
//!   flooring, the quantity is shrunk by one lot (or a few ulps without a
//!   step) until `notional + fee ≤ cash`.
//! - A sell never yields negative proceeds (`rate < 1`) and never sells more
//!   than the position. Both invariants follow from the code's comparisons on
//!   the exact `f64` values it then subtracts, not from tolerances.
//! - With a lot step, the position is held as an integer number of lots (at
//!   most 2⁵⁰, so every lot count is exact in `f64`): every fill is `n·step`
//!   rounded once, the position is `lots·step` rounded once, and a full exit
//!   leaves no dust.
//! - An order whose notional is at most 10⁻¹² of equity is skipped as rounding
//!   dust (never a full exit). Without this, a target of 1 would re-buy the
//!   few-ulp cash residue of a cash-capped buy at every bar.
//! - The loop is strictly forward and keeps no global state: the outputs up to
//!   bar `k` depend only on `bars[..=k]` and `targets[..k]`, bit for bit.
//!
//! # Time grid
//!
//! Bars only need strictly increasing `open_time`; no grid is checked, and a
//! volume of 0 is valid. Across a gap, a decision taken at the close of the
//! last bar before it executes at the first open after it. Annualization in
//! [`metrics::periods_per_year`] assumes a regular grid, so callers must
//! re-align series to the grid and cut them at gaps before scoring.
//!
//! # Not modelled
//!
//! Partial fills, market impact beyond the flat slippage, latency, fees paid
//! in a third asset (BNB is modelled as a lower quote rate), funding, and
//! filters other than minimum notional and lot step (tick size, price bands).

mod error;
pub mod metrics;
mod sim;
mod types;

pub use error::SimError;
pub use sim::simulate;
pub use types::{Bar, FeeSchedule, Fill, Liquidity, Side, SimConfig, SimResult, SymbolFilters};
