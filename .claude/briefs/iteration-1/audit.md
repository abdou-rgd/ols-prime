# Brief — audit iteration 1

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **When:** after `backtest.md` and `risk.md` are finished and the data pipeline is in.
- **Motto:** we should be more ambitious: trace every timestamp from the raw file to the decision.

## Scope

### 1. `crates/ols-core` (written by the orchestrator, never reviewed)
- **Identities.** Primal equals dual. Weights sum to one with a free intercept. The tree equals the batch. RLS equals weighted ridge.
  - Do the tests in `tests/identities.rs` prove these, or could their tolerances or generators hide an error?
- **Bitwise purity.** Is `Matrix::range_stats` really a pure function of `(start, end)`, as `reads_ignore_later_entries` claims?
- **Causality.** `time` is the moment an outcome becomes known. `time_range` is inclusive at `as_of`, and timestamps may be equal.
  - What must a caller do so that no outcome is read before t + h?
- **Numerics.**
  - λ = 0 with a free intercept;
  - near-singular Gram matrices;
  - cancellation in `residual_sum_squares`;
  - silent NaN.

### 2. `crates/backtest`
- **Execution.** A decision executes at the next open.
- **Future-invariance proptest.** Does it perturb everything after k, and compare bitwise?
- **Costs.** Fees counted once, slippage sides, minimum notional and lot step.
- **Solvency.** Is cash ≥ 0 guaranteed by construction?
- **Metrics.** Annualization over 365 days; the Sharpe and Sortino definitions; turnover.

### 3. `crates/risk`
- `docs/DESIGN.md` §6 against the tests and the Kani harnesses.
- **Harness assumptions.** Are any `kani::assume` domains so narrow that a harness proves nothing? Does any harness assert vacuously?
- **Overflow paths.** Are they covered?
- **Sells.** They must always stay allowed.

### 4. `python/olsprime` (the data pipeline)
- **Checksums** are verified.
- **Timestamps.** Spot files switched from ms to µs on 2025-01-01.
- **Bars.** Open time versus close time; gaps and duplicates.
- **Universe.** It must be point-in-time:
  - built only from data before each month;
  - delisted pairs kept;
  - stablecoins and leveraged tokens excluded.
- **Tests** run offline.
- **The vault is on disk.** The data from 2026-01 to 2026-09 has already been downloaded. Check that no code reads it, and say whether we need a lock in the code that refuses vault dates until the final evaluation.

### 5. Claims
`README.md`, the crate docs and `docs/` must state nothing that no test shows.

## Constraints

- **Never edit the repository.** Throwaway checks go in your temp directory.
- **Run the tests yourself:**
  - `cargo test --workspace`;
  - `cd python; uv run pytest`;
  - `cargo kani -p risk` in WSL2 if Kani is installed there. Otherwise, say so.
- **Data.** It lives in `data/` (gitignored). Never write there.

## Report

Follow your role file:
- ranked findings (Blocker, Major, Minor), each with `file:line`, the failure scenario and the fix;
- what you checked and found clean;
- a verdict;
- one bolder check to add to the bench.
