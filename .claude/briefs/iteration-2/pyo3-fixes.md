# Brief — Binding fixes and a Rust oracle after Audit B, phase 1 (iteration 2)

- **Agent:** `rust-engineer` (opus, high).
- **When:** 2026-10-28, after Audit B's phase 1 ("pass with fixes": no Blocker, no Major, 6 Minor; three of them are this brief's).
- **Motto:** we should be more ambitious. Today the binding is checked against a Python port of the simulator. Check it against the compiled crate itself.

Read first:
- `.claude/briefs/iteration-2/pyo3.md`, the brief the bindings were built from;
- `crates/olsprime-py/`: `src/lib.rs`, `src/convert.rs`, `src/py/*.rs` and `olsprime_rs.pyi`;
- `python/tests/test_rust_bindings.py`.

## Fixes

1. **Overflow errors leak out as `OverflowError`.**
   - Where: `olsprime_rs.pyi:4-6` and `src/py/mod.rs:10-12` promise only `TypeError`, `ValueError`, `SimError` and `OlsError`.
   - What fails: integer scalars outside their Rust range raise PyO3's `OverflowError`, which is not a `ValueError`. Examples: `Matrix.range_stats(-1, 2)`, `Matrix.time_range(2**63, 0)`, `Matrix(2, 1, block_log2=-1)`, `periods_per_year(2**70)`.
   - The fix: extract those integers through `&Bound<PyAny>` and a helper that maps overflow to `ValueError` with the argument's name. `src/py/risk.rs:14-41` already has one, `int_field`: reuse it or generalize it.
   - Then the documented promise is true. Add tests for each of the four examples.
2. **Microseconds passed as milliseconds.**
   - Where: `src/py/sim.rs:283` and `src/convert.rs:103` accept any strictly increasing `open_time`.
   - What fails: spot files switched to microseconds on 2025-01-01 (`docs/PROTOCOL.md`), and µs passed as ms would be annualized wrongly.
   - The fix: refuse any bar time ≥ 10¹⁴ with a `ValueError` saying it looks like microseconds. In milliseconds that is the year 5138; in microseconds it is 1973.
   - Put the check in `BarColumns::to_bars`, or wherever `simulate` and `periods_per_year` take times. Add a test with a µs series.
3. **A Rust oracle for the bit-for-bit claim.**
   - Today `test_simulate_equals_the_reference_port_bit_for_bit` compares the binding with a line-by-line Python port. A port can share a bug with the simulator; the compiled crate cannot.
   - The fix:
     1. **The example.** Add `crates/backtest/examples/golden.rs`. With a fixed seed and a simple deterministic generator such as SplitMix64, it builds a few cases: random bars, targets, a fee schedule and filters, including fees, filters and dust edge cases.
     2. **The golden file.** The example runs `backtest::simulate` and writes the inputs and every output to a small, committed text file. Every float goes in as its exact bits (hex `u64`), with `None` and the errors explicit. A suggested path is `python/tests/golden/simulate.txt`; stay under 200 kB.
     3. **The Rust test.** A test in `crates/backtest` regenerates the cases and compares them with the committed file, so any change to the simulator forces a conscious regeneration. Document the regeneration command in the example's doc comment.
     4. **The Python test.** In `python/tests/test_rust_bindings.py`, read the file, call `olsprime_rs.simulate` on the inputs, and compare every output bit for bit.

Not yours: Audit B's Minors 1, 2, 5 and 6 are wording in `docs/DESIGN.md` and `README.md`. The orchestrator fixes them.

## Constraints

- **Files.**
  - `crates/olsprime-py/**`;
  - `crates/backtest/examples/**` and new test files under `crates/backtest/tests/`;
  - `python/tests/test_rust_bindings.py`;
  - the new golden file.
- **Out of bounds.**
  - Do not touch `crates/risk/**`: another `rust-engineer` is adding Kani contracts there.
  - Do not touch `crates/backtest/src/**`, unless a fix truly needs it. If it does, say so in your report, and say why.
  - Do not touch any other Python file.
- **Builds.**
  - Use your own `CARGO_TARGET_DIR` under your scratch folder: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\pyo3-fixes\`.
  - `crates/risk` is mid-edit. If it fails to compile, wait, then retry.
- **Do not run `uv sync`, or a plain `uv run`, in `python/`.** Other agents are running tests and the bench from that environment right now. On Windows, replacing a loaded `.pyd` fails or breaks their runs.
  - Test the Python side in a scratch environment instead:
    1. `uv venv <scratch>/venv --python 3.12`;
    2. `maturin build --release --features python -o <scratch>/wheels` from `crates/olsprime-py`, or the equivalent through `uv tool run maturin`;
    3. `uv pip install --python <scratch>/venv <wheel> numpy pytest polars`;
    4. run `python -m pytest C:\Users\abdou\ols-prime\python\tests\test_rust_bindings.py` with that venv's Python, plus whatever that test imports.
  - If that test needs the `olsprime` package, also install `C:\Users\abdou\ols-prime\python` into that venv with `--no-deps` after its dependencies.
  - The orchestrator rebuilds the shared environment at integration.
- **Rust checks.** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo clippy -p olsprime-py --all-targets --features python -- -D warnings`, `cargo test -p backtest -p olsprime-py`. Also the MSRV check `cargo +1.89 check -p olsprime-py --all-targets --features python --locked`, if 1.89 is installed.
- **Git.** Run no git command that changes state.

## Done when

- The three fixes are in and tested.
- Every check above is green.
- The golden file round-trips:
  - Rust regenerates it identically;
  - the Python binding matches it bit for bit in the scratch environment.

## Report

- What changed, file by file.
- The golden file's path, size and cases.
- The commands you ran, with their results.
- Anything you saw in passing that is wrong.
- One bolder next step.
