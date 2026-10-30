# Brief — `beta` and `mean_y` on `CentredReadout` (iteration 4)

- **Agent:** `rust-engineer` (opus, high).
- **Runs in parallel with:** two `ml-engineer`s. One works in `python/olsprime/bench/report2.py`, `python/olsprime/bench/power.py` and their tests. The other works in `python/pyproject.toml`, `python/uv.lock`, `.github/workflows/ci.yml`, `python/olsprime/models/`, `python/olsprime/train/` and new test files. Touch none of those files.
- **Motto:** we should be more ambitious, in the test, never in the claim.

## Goal

`olsprime.explain why` (iteration 4) breaks a Matrix forecast into per-feature parts, `β_j · z̃_j`, with `z̃ = x_q − x̄`. The readout must therefore expose the coefficients it used, and its intercept, with the bits it used.

Read first: `docs/DESIGN.md` §3.1–3.3, `crates/ols-core/src/matrix.rs` (`CentredReadout`, `centred_query`, `centred_weights`, `centred_contribution`), `crates/ols-core/src/centred.rs` (`CentredFit::beta`) and `crates/olsprime-py/src/py/ols.rs` with `crates/olsprime-py/olsprime_rs.pyi`.

## Deliverables

1. **Rust, `ols-core`.** On `CentredReadout`:
   - `beta() -> DMatrix<f64>`, shape (q, p): the β of the fit behind the readout, such that `forecast = mean_y + β (x_q − mean_x)`.
   - `mean_y() -> &DVector<f64>` (or the equivalent), shape (q,): ȳ of the range.
   - The values come from the same solve as `forecast`, not from a second solve. Keep every existing read's bits: no change to `forecast`, `a`, `leverage`, `rss`, `ridge_batch` or `kernel_batch`.
   - Document, on each getter, what it returns and how it relates to `forecast`.
2. **Bindings, `olsprime-py`.** The two getters as properties of `olsprime_rs.CentredReadout`, numpy float64, shapes (q, p) and (q,), with docstrings in `olsprime_rs.pyi`.
3. **Tests.**
   - Rust: on random windows (property test), `mean_y + beta · (x_q − mean_x)` equals `forecast` within a tolerance you derive and state; `beta` equals a closed form `M_yx (M_xx + Λ)⁻¹` computed independently from `centred_stats`; at ρ = 0 on a well-posed design, `beta` equals the OLS coefficients.
   - Python, in a new file `python/tests/test_readout_beta.py`: the same identity through the bindings, and the shapes and dtypes. Keep the file free of data reads.
4. **Exactness note.** Say whether `mean_y + beta · z̃`, computed in numpy, reproduces `forecast` to the bit or only within rounding, and why. `why` will state it.

## Constraints

- **Build.** Do not run `uv sync` on the shared environment `python/.venv`: another agent changes the project's dependencies in parallel. Build and test the extension in your own environment: set `UV_PROJECT_ENVIRONMENT` to a folder under your scratch and `CARGO_TARGET_DIR` to another one, then `uv sync` and `uv run pytest tests/test_readout_beta.py tests/test_rust_bindings.py tests/test_matrix_batch.py` from `python/`.
- **Checks.** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo clippy -p olsprime-py --all-targets --features python -- -D warnings` and `cargo test --workspace` must pass.
- **Files.** `crates/ols-core/src/matrix.rs` (and `centred.rs` if needed), `crates/olsprime-py/src/py/ols.rs`, `crates/olsprime-py/olsprime_rs.pyi`, Rust tests, and the new `python/tests/test_readout_beta.py`. Do not touch `Cargo.toml`, `Cargo.lock` or `crates/risk/`: a change there re-runs Kani's 74-to-91-minute job on GitHub.
- **Scripts.** Write scripts with a file tool. Never pipe or heredoc code into `python`.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\beta-getter\`.
- **Git.** Run no git command that changes state, and do not commit.

## Done when

- The two getters exist in Rust and Python, documented, with the tests above, and every check passes.

## Report (short)

- The API, with the exactness note.
- The tests and the tolerance.
- One bolder next step.
