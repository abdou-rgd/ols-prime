# Brief — audit fixes in `backtest` and `ols-core` (iteration 1)

**Role.** `rust-engineer`. **Scratch folder:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\rust-engineer-fixes\` (yours alone; put any temp copy of the workspace there).

The `leakage-auditor` passed iteration 1 with fixes. Its evidence is in `...\scratchpad\audit\EVIDENCE.md` (mutant table, probes P1–P10). This brief covers its findings on `crates/backtest` and `crates/ols-core`. **Do not touch `crates/risk`**: its Kani proofs are running from the current sources, and the orchestrator will handle the risk findings when they finish.

## 1. The dust floor is untested (Major)

`DUST_FRACTION = 1e-12` (`crates/backtest/src/sim.rs:14`, used at `:185` and `:232`) skips orders of at most 1e-12·equity. The auditor's mutant B9 sets it to 1e-3: every rebalance under 0.1 % of equity is then silently skipped, and the whole suite still passes.
- Add tests that pin the floor from both sides, without fees, slippage or filters:
  - a rebalance worth 1e-6·equity fills;
  - one worth 1e-13·equity is skipped;
  - a full exit is never skipped as dust, whatever its size.
- Show that the new tests kill B9. Apply the mutant to a copy of the workspace in your scratch folder, never in the repository, and give the failing test names.

## 2. A singular solve can succeed silently (Minor, but it returns wrong numbers)

`RidgeConfig::factor` (`crates/ols-core/src/ridge.rs:45-51`) trusts `nalgebra`'s Cholesky, which only fails on a pivot ≤ 0. Rounding can leave a tiny positive pivot on an exactly singular matrix.
- Probe P4: two identical entries, λ = 0, free intercept. The Gram is singular, yet `query` returns `Ok(1.5)` when every outcome is 1.0.
- Probe P2: a near-collinear design at λ = 0 gives Σw = 0.999999999345. That fits the identity tests' 1e-9 tolerance only by luck.
- The generators draw λ ∈ [0.05, 10], so λ = 0 is never exercised.

What to do:
- First write the failing test for P4, then fix it.
- Reject a numerically singular factor with `OlsError::NotPositiveDefinite`. A pivot of the Cholesky factor is the Schur complement of the leading block, so compare the smallest squared pivot with the matrix's scale.
  - The auditor suggests the threshold `p·ε·max diag(G + Λ)`.
  - Pick the criterion yourself, and justify it in the doc comment with the maths.
  - Exactly singular designs (duplicates, collinear columns, n < p at λ = 0) must be rejected whatever the rounding does.
- Check that `a` and every forecast are finite before returning them.
- Add λ = 0 and λ = 1e-12 to the generators of the identity tests. Each identity must then either hold within a documented tolerance, or the case is rejected. A silently wrong answer is never acceptable. If a tolerance has to depend on conditioning, say so in the test.

## 3. A Readout is accepted by a memory that did not produce it (Minor)

`Matrix::check_readout` (`crates/ols-core/src/matrix.rs:316`) only checks `p` and the range. Probe P10: any memory with the same `p` and enough entries accepts another's `Readout`, and `weights` and `contribution` then return numbers computed from foreign entries.

Two scenarios must be rejected:
- **(a)** two independent memories with the same `p`;
- **(b)** a memory cloned before the readout was taken, which then received different entries.

Choose one of these designs, and say which in the doc comment:
- **(i)** a per-instance id from a global atomic counter, with a hand-written `Clone` that gives the copy a fresh id. A readout is then valid only on the instance that produced it.
- **(ii)** content validation: `check_readout` recomputes `C_R a` from `range_stats(start, end)` and compares it bitwise with `readout.forecast`. `range_stats` is pure (probe P9), so this accepts a faithful clone and rejects a diverged one.

Add an error variant if needed, and tests for (a) and (b).

## 4. Write down which timestamp Matrix's `time` is (Minor)

`matrix.rs:18-20` says that `time` is the moment an outcome becomes known, but not which timestamp that is for bar data. The answer, and the data pipeline will provide it as a `known_at` column:
- for an outcome realized by a bar, `time` is that bar's `known_at = max(close_time, open_time + interval − 1 ms)`, in milliseconds UTC. The stored `close_time` alone is wrong for 15 bars and once precedes the open.
- `as_of` is the decision bar's `known_at`.
- `time ≤ as_of` is inclusive, and this is correct, because a decision taken at a bar's close executes at the next bar's open.

Say this in the doc comment of `Matrix` and of `time_range`, and add one doc test or unit test that encodes it on two bars.

## Constraints

- Own `crates/backtest` and `crates/ols-core`, and only them.
- Public API changes inside `ols-core` are allowed. List them in your report.
- No new dependency.
- Run the workspace checks; all must be clean or green:
  - `cargo fmt --all --check`;
  - `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  - `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --locked`;
  - `cargo test --workspace --locked`, with 121 tests today plus yours.
- Do not commit, do not switch branches.

## Report (short)

What changed (files, API), the threshold and its justification, the tests with counts, the B9 mutant result, the design chosen in §3, and one bolder next step.
