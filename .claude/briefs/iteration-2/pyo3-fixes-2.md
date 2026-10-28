# Brief — PyO3 fixes, round 2 (iteration 2)

- **Agent:** a `rust-engineer`: the one who wrote the Kani contracts, resumed.
- **When:** 2026-10-28, after Audit B phase 2.
  - **Verdict:** pass with fixes.
  - **Kani:** passes. The four proven harnesses are genuine, the flags weaken nothing, and a mutation of `product` makes `product_contract` fail.
  - **PyO3:** one Major and three Minor findings, plus one Minor finding on Kani wording.
- **Motto:** we should be more ambitious. "Nothing else is raised" must be true for every input a caller can choose.

## Major — a shape the caller chooses kills the interpreter

**Where.** The claim is at `crates/olsprime-py/src/py/mod.rs:10-14` and `olsprime_rs.pyi:17-19`. The failing entry points are in `crates/olsprime-py/src/py/ols.rs`:
- `SuffStats.__new__` (`:86`);
- `SuffStats.from_rows` (`:99`);
- `ols_attention` (`:302`);
- `Rls.__new__` (`:331`);
- `Matrix.__new__` and `Matrix.push` (`:473`, `:480`).

**What fails.** For example `rs.SuffStats(2**20, 1)`, `rs.Rls(2**20, 1, 1.0)`, or `rs.SuffStats.from_rows(np.zeros((0, 2**20)), np.zeros((0, 1)))`.
- Each one ends with "memory allocation of 8796093022208 bytes failed" and Windows exit 0xC0000409.
- The p×p Gram allocation fails, and Rust aborts the whole process. No exception reaches Python, and whatever the bench was doing is lost.

**Fix.**
- **Refuse first.** Refuse a shape whose allocations would exceed a stated limit, before allocating anything, with a `ValueError` that names the argument and the limit.
  - Compute the bytes with checked arithmetic: the p×p and p×q matrices, and the batch size where one is allocated.
  - A limit of 1 GiB per object is a reasonable default.
- **Export the limit.** Export it to Python as a constant, document it in `olsprime_rs.pyi`, and use one helper for all five entry points.
- **Tests.** Add tests to `python/tests/test_rust_bindings.py` for the five entry points.
  - Run them **in-process**. The refusal happens before any allocation, and an abort would still fail the run loudly.
  - Do not use `subprocess`: the static guard forbids it, and `tests/allowlist.py` is being edited by another agent right now.

## Minor

1. **`kani.yml`, lines 13-22 and 107.**
   - Name the four contracts that are **not** proven on 2026-10-28: `notional_floor`, `notional_ceil`, `max_qty_within` and `round_down_to_lot`. Today only `proofs.rs:56-61` says so.
   - Say that only the `spec.rs` proptests sample them.
   - Comments only.
2. **`convert.rs:75-77`.** The `Microseconds` error text carries the indentation of its multi-line literal. Use `concat!` or a `\` line continuation.
3. **Microsecond times.**
   - `MICROSECOND_TIMES_FROM` is Rust-only: export it to Python.
   - `Matrix.push` and `Matrix.extend` (`ols.rs:480`, `:493`) accept a time in microseconds silently: apply the same refusal there.
   - Add tests for both.
4. **`crates/risk/src/spec.rs:135-142`.** `product_spec_pins_product` checks only `Some(v + 1)` and `None`. Use `neighbours`, as the other four tests do.
   - Change only code under `#[cfg(test)]`, so that no Kani harness changes.
   - Say in your report whether the change compiles under `cfg(kani)`.

**No action.** The Kani 0.68 warnings "call to saturating_add/saturating_sub: not enough arguments" inside `enter/exit_contract_clause` come from the contract-depth counter. The auditor's mutation shows the `ensures` check is live. The orchestrator records this in the outcome.

## Constraints

- **Files.**
  - `crates/olsprime-py/**`, `olsprime_rs.pyi` included;
  - `python/tests/test_rust_bindings.py`;
  - `crates/risk/src/spec.rs`, under `#[cfg(test)]` only;
  - `.github/workflows/kani.yml`, comments only.
- **Out of bounds.**
  - `python/tests/tripwire.py`, `conftest.py`, `allowlist.py`, `test_tripwire.py`, `test_protocol.py`: a data-engineer is editing them.
  - `python/olsprime/bench/**` and its tests: an ml-engineer is finishing there.
  - `docs/`.
- **Python.** Never rebuild the shared venv (`python/.venv`): other agents are running tests in it. Test the bindings in a scratch venv:
  - set `UV_PROJECT_ENVIRONMENT` to a folder under your scratch;
  - run `uv sync --locked`, then `uv run --no-sync pytest tests/test_rust_bindings.py` with the same variable set;
  - use your own `CARGO_TARGET_DIR` under your scratch.
- **Checks.**
  - `cargo fmt --all --check`;
  - `cargo clippy --workspace --all-targets -- -D warnings`;
  - `cargo clippy -p olsprime-py --all-targets --features python -- -D warnings`;
  - `cargo test --workspace`.
- **Time.** Report by 03:50 UTC.

## Done when

- The checks above are green.
- The binding tests pass in the scratch venv.
- Every one of the five entry points refuses an oversize shape with a `ValueError`.

## Report

- The files changed.
- The limit chosen, and why.
- The test counts.
- Whether `spec.rs`'s change touches anything under `cfg(kani)`.
- One bolder next step.
