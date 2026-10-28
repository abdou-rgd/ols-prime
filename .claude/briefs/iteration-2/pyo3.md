# Brief — PyO3 bindings (iteration 2)

- **Agent:** `rust-engineer` (opus, high).
- **Runs in parallel with:**
  - a second `rust-engineer` in `crates/risk` (`kani-contracts.md`);
  - two `data-engineer`s and two `ml-engineer`s in `python/olsprime/`.
- **Motto:** we should be more ambitious, in the tests, never in the claims.

## Goal

A Python extension module, `olsprime_rs`, built from a new crate `crates/olsprime-py`. It exposes the simulator, its metrics, the ridge / Matrix / RLS core and the risk kernel to the Python bench.

Report n°1 will score every baseline through `olsprime_rs.simulate`. So the bindings must:
- return exactly what the Rust functions return, bit for bit;
- be hard to misuse: wrong dtypes, lengths or names fail loudly.

## API, Python side

Choose the details, but keep this shape and write type stubs (`olsprime_rs.pyi`, `py.typed`).

- **Simulator.**

  ```python
  simulate(open_time, open, high, low, close, volume, targets, *,
           initial_cash, fees, liquidity="taker", slippage_bps=2.0,
           min_notional=0.0, step_size=0.0, rebalance_threshold=0.0) -> SimResult
  ```

  - 1-D numpy arrays of one length: `open_time` int64 (ms since the epoch, UTC), the rest float64. Accept non-contiguous input or refuse it with a clear error; never read it wrong.
  - `fees` is a `FeeSchedule` or the name of a constant: `"zero"`, `"binance-spot"`, `"binance-spot-bnb"`, `"coinbase-advanced-entry"`, `"kraken-pro-entry"`. A custom schedule takes a name, a maker rate and a taker rate.
  - `SimResult` gives numpy arrays `equity`, `position`, `cash`, the fills as numpy columns (`bar`, `time`, `side`, `qty`, `price`, `notional`, `fee`), and `fees_paid`.
  - `backtest::SimError` becomes `olsprime_rs.SimError`, a subclass of `ValueError`, with the Rust message.
- **Metrics.** `metrics(result, periods_per_year)` mirrors `backtest::metrics::Metrics`. Also expose `periods_per_year(bar_ms)` and `max_drawdown(equity)`.
- **ols-core.** `SuffStats`, `RidgeConfig` with `ridge_fit` and `RidgeModel.predict`, `ols_attention`, `Rls`, and `Matrix` with `push`, `len`, `query`, `weights`, `contribution`, `summarize`, `range_stats` and `time_range`. `OlsError` becomes `olsprime_rs.OlsError(ValueError)`. Iteration 3 builds Matrix on hand-made features on top of this, so make it pleasant: numpy in, numpy out.
- **risk.** `check_order(account, limits, request)` with small frozen classes, integers only, returning an approval or a rejection reason.
- `__version__`.

## Packaging

- **Crate.** `crates/olsprime-py`: package `olsprime-py`, library `olsprime_rs`, with its own `pyproject.toml` (maturin build backend). The Python distribution is `olsprime-rs`.
- **Wiring.** `python/pyproject.toml` depends on `olsprime-rs` through `[tool.uv.sources]` with a path to the crate. `uv sync` must build it natively on Windows (MSVC, Python 3.12 from `python/.python-version`). abi3 for 3.12 is fine.
- **CI.** The pytest jobs of `.github/workflows/ci.yml` (ubuntu and windows) then need a Rust toolchain. Pin actions as the file does, with full SHAs.
- **Workspace health.** These must stay green, natively on Windows and in CI:
  - `cargo fmt --all --check`;
  - `cargo clippy --workspace --all-targets -- -D warnings`;
  - `cargo test --workspace` (the `extension-module` feature must not break linking; enable it only from maturin);
  - the MSRV job, `cargo check --workspace --all-targets` on Rust 1.89;
  - `cargo kani -p risk`, which must not see the new crate.
- **Versions.** Check crates.io (`cargo search`, `cargo info`) for the current PyO3 and `numpy` (rust-numpy) releases that work together, and PyPI for maturin. Pin exact versions and say which in your report. You have no web tool; `cargo info` and `uv` reach the registries.

## Tests

- **Rust.** `cargo test -p olsprime-py` covers the conversion helpers, if any.
- **Python.** Write `python/tests/test_rust_bindings.py`, offline and fast (under 10 s):
  - `simulate` against hand-computed fills on small cases, and the documented invariants:
    - cash and position ≥ 0;
    - `equity[0] == initial_cash`;
    - `targets[last]` never executes.
  - Future invariance through the binding: perturb the bars after k, and every output up to k is identical, bit for bit.
  - `ridge_fit` equals the numpy closed form (rtol 1e-10). `ols_attention` weights sum to 1 with an intercept.
  - `Matrix.query` over a range equals batch ridge on that range. `Rls` equals weighted ridge.
  - Every error maps to its exception class, with a message.
  - `check_order`: one approved and one rejected example, taken from the risk crate's docs.
  - Misuse (wrong dtype, wrong length, unknown fee name) raises `TypeError` or `ValueError`.

## Scope and constraints

- **Files.** Only these:
  - `crates/olsprime-py/**`;
  - `python/tests/test_rust_bindings.py`;
  - `python/pyproject.toml`: the dependency and `[tool.uv.sources]` only;
  - `python/uv.lock`, `Cargo.lock`;
  - `.github/workflows/ci.yml`;
  - the root `Cargo.toml`, only if needed; say why.
- **Other crates.** Do not change the public API of `ols-core`, `backtest` or `risk`. If a binding needs something they lack (a derive, an accessor), add only non-breaking items and list them in your report.
- **Shared workspace.**
  - Another `rust-engineer` edits `crates/risk` and runs Kani in WSL2. Never run `cargo fmt --all`; use `cargo fmt -p olsprime-py`.
  - "Blocking waiting for file lock" only means another build is running.
  - Python agents edit `python/olsprime/{data,features,labels,eval,models,bench}` and `python/tests/` at the same time. Touch none of their files.
  - The orchestrator already added `scipy` and `xgboost-cpu`. Run no `uv add` or `uv remove` except for your own dependency, then `uv lock`.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\pyo3\`.
- **Git.** Run no git command that changes state, and do not commit. The branch is `iteration-2`.

## Done when

- In `python/`: `uv sync`, then `uv run pytest tests/test_rust_bindings.py` is green on Windows, and the whole `uv run pytest` is green. If another agent's test fails, report it; do not fix it.
- The four cargo checks above are green on Windows.
- MSRV: run `cargo +1.89 check --workspace --all-targets` if that toolchain is installed (`rustup toolchain list`). Otherwise say so.
- Optional but valuable: build the wheel in WSL2 too (`maturin build`, with `CARGO_TARGET_DIR` on the Linux disk), to show that Linux builds.

## Report (short)

- Files and the public Python API.
- Pinned versions.
- Tests, with counts.
- Every non-breaking addition to another crate.
- What you could not verify.
- One bolder next step.
