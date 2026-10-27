# Brief — Kani function contracts (iteration 2)

- **Agent:** `rust-engineer` (opus, high).
- **Runs in parallel with:** a second `rust-engineer` building `crates/olsprime-py` (`pyo3.md`), and Python agents.
- **Motto:** we should be more ambitious, in the proofs, never in the claims.

## Goal

Give every one of the 14 harnesses in `crates/risk/src/proofs.rs` a verdict, and move as many as possible into the `proven` job.

Iteration 1 found that CBMC drowns in the 64- and 128-bit products and quotients of the arithmetic helpers: see "Kani's cost is in the arithmetic" in the Lessons of `docs/ITERATIONS.md`. The way out is function contracts:
1. Give `product`, `notional_floor`, `notional_ceil`, `max_qty_within` and `round_down_to_lot` contracts, through `#[cfg_attr(kani, kani::requires(...))]` and `#[cfg_attr(kani, kani::ensures(...))]`. State them without division where you can.
2. Prove each contract once, with `#[kani::proof_for_contract(f)]`.
3. Let the `check_order_*`, `approval_is_maximal_and_stable_tiny` and `sell_outcome_ignores_trip_wires` harnesses use the helpers through `#[kani::stub_verified(f)]`. Then no harness bit-blasts the same 128-bit division twice.
4. The `*_full` and `*_market` helper harnesses become the `proof_for_contract` harnesses, or stay as extra checks.

## What is known

- **Setup.** Kani 0.68.0 with CBMC 6.11.0 is installed in WSL2 Ubuntu (`~/.cargo/bin/cargo-kani`). The machine has 12 threads and 15 GB. The repository is at `/mnt/c/Users/abdou/ols-prime`.
  - `CARGO_TARGET_DIR` is not set. Put yours on the Linux disk, for example `~/olsprime-target/kani-contracts`.
  - The scripts and logs of iteration 1 are in `~/kani-runs`, `~/kani-narrow` and `~/kani-*.sh`. Reuse them.
- **Syntax.** Function contracts are unstable in Kani 0.68 (`-Z function-contracts`). You have no web tool. Confirm the syntax from `cargo kani --help`, `cargo kani -Z help` if it exists, and the sources of the `kani` crate and its macros under `~/.kani/kani-0.68.0`.
- **Timings from iteration 1.**
  - Proven: `buy_gates_block_every_buy` (611 s alone) and `sells_are_never_gated` (14 s alone).
  - The other 12 ran out of time at 60 minutes, on a shared machine.
  - Lot rounding on b-bit inputs: 0.85 s at 8 bits, 3.1 s at 12, 73 s at 16, more than 45 minutes at 20.
  - Kissat did no better than CaDiCaL.
- **Running now.** The 12 open harnesses run on GitHub runners, 340 minutes each, on the iteration 1 code: Actions run 40596921451 of `abdou-rgd/ols-prime`. `gh run view 40596921451` works from the Windows side. The orchestrator will read the verdicts; you may too.
- **Ideas, most promising first.**
  - Multiplicative postconditions: `r·price ≤ cap·QTY_SCALE < (r + 1)·price` rather than `r == cap·QTY_SCALE / price`. CBMC still encodes the implementation's division; measure whether the circuit and the time shrink.
  - Another back-end per harness: `#[kani::solver(minisat | kissat | cadical)]`, or an SMT back-end if Kani 0.68 can pass one through to CBMC (`--cbmc-args`, `--smt2`, `--z3`, `--bitwuzla`). `z3` may be installable without sudo, through `pip install z3-solver` in a venv. Check before you build anything.
  - Restructure a helper so that it proves faster, with identical behaviour. Pin the equivalence with a proptest against the old version over the full `i64` range, and keep the old version as the reference in the tests.
  - Do not use `kani::assume` on magnitudes: iteration 1 showed it does not shrink the circuit.
- **How to run.** One harness per process, in the background, under `timeout`, with logs in your scratch folder. At most 6 processes at once: the other `rust-engineer` builds at the same time. Record each harness's wall time and whether it ran alone or shared the machine.

## Scope and constraints

- **Files.** Only `crates/risk/src/**`, `crates/risk/tests/**` and `.github/workflows/kani.yml`.
- **Docs.** Do not edit `CLAUDE.md` or `README.md`. Report the new `proven` list, and the orchestrator updates them.
- **Behaviour.** `lib.rs` must behave exactly as before. The 40 tests stay green, and the property tests keep checking P1–P7.
- **Commands.** `cargo fmt -p risk`, `cargo clippy -p risk --all-targets -- -D warnings`, `cargo test -p risk`. Windows-native cargo is fine for these.
- **Scratch.**
  - Windows: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\kani-contracts\`.
  - WSL2: `~/kani-it2/`.
- **Git.** Run no git command that changes state, and do not commit.
- **Time.** Spend at most about 2 h 30 of wall time. Then stop the runs still going, record them as timeouts and report.

## Done when

- The contracts, their `proof_for_contract` harnesses and the `stub_verified` harnesses are written, compiled by Kani and run. Each harness has a verdict or a recorded timeout.
- `kani.yml`:
  - `proven` lists every harness that finished, with the flags it needs;
  - `open` lists the rest;
  - the file stays valid YAML (check it, for example with `uv run python -c "import yaml"` if available, or by eye).
- A table: harness, verdict, wall time, solver, alone or shared.

## Report (short)

- The contracts, and how each is proven.
- The table above.
- Be explicit about soundness: a `stub_verified` proof is only as good as the contract's own proof. Say which contracts are proven and which are only assumed.
- One bolder next step.
