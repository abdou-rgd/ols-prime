# Brief: narrow nondet inputs for the bounded Kani harnesses (rust-engineer)

## Goal

Test one hypothesis, then fix the Kani future-incompat warnings.

The hypothesis: the bounded harnesses time out because `kani::assume` bounds the magnitudes without shrinking CBMC's circuit. Every `i128` product, and the overflow check behind it, is still bit-blasted at full width.

The proposed fix: draw each bounded value from a narrow nondet integer, sign-extended to its type. The upper bits then become copies of one sign literal. CBMC's gate constructors fold such literals: `x XOR x` is false, and `false AND y` is false. So the absolute values inside the signed multiplier and divider get constant upper bits, and their partial products disappear.

## State (2026-10-26, 00:45)

Kani 0.68.0 runs locally, one harness per process, with a 60-minute cap and CaDiCaL:
- Proven: `buy_gates_block_every_buy` (795 s) and `sells_are_never_gated` (29 s).
- Timed out at 60 minutes: `round_down_to_lot_full`, `tripwires_full`, `check_order_tiny` and `approval_is_maximal_and_stable_tiny`.
- Still running: the market and full harnesses.
- Tried without success: Kissat (`--solver kissat`). `round_down_to_lot_full` and `tripwires_full` still timed out at 40 minutes.

The runs use `~/.cache/ols-prime/target-kani` on the repository at `/mnt/c/Users/abdou/ols-prime`. **Do not touch either until the orchestrator says so.** The last queued harness compiles around 00:55, and a source change before then would be picked up mid-run.

## Task A: the experiment, in a copy

1. Copy the workspace to the Linux disk, for example `~/kani-narrow/` (the root `Cargo.toml`, `Cargo.lock` and `crates/`).
2. Use your own `CARGO_TARGET_DIR`, for example `~/.cache/ols-prime/target-kani-narrow`.
3. In the copy's `crates/risk/src/proofs.rs`, change how the bounded harnesses draw their inputs. The harnesses are `check_order_tiny`, `approval_is_maximal_and_stable_tiny`, `check_order_market`, `max_qty_within_exact_market` and `max_qty_within_galois_market`.
   - Draw every bounded field from a helper that sign-extends a k-bit value, for example `(x << (64 - k)) >> (64 - k)` on an `i64` from `kani::any()`.
   - Choose k so that the k-bit range covers the whole documented domain, `[-bound, bound]` included. That needs `k ≥ log2(bound) + 2`: a 25-bit signed integer misses `+2^24`.
   - Keep `assume_within` and the other assumptions exactly as they are. The claims and the documented domains must not change: only the way values are drawn changes. Fields the domain leaves free stay `kani::any()`.
4. Run each changed harness with `timeout -k 30s 45m cargo kani -p risk --exact --harness proofs::NAME`, at most 5 at a time. Log the start and end time of each.
5. If the copy runs out of time, try the same idea on `round_down_to_lot_full`. A bounded twin, for example a `round_down_to_lot_tiny` on u32-range inputs, would show whether the division is what blocks it. Report this as data, not as a harness to add.

## Task B: the braces, in the repository, after the orchestrator's go

Wrap the `assert!` arms of `crates/risk/src/proofs.rs` in braces, at lines 106, 116, 123, 472 and 477. Kani's `assert!` expands with a trailing semicolon, and these arms use it in expression position. rustc's `semicolon_in_expressions_from_non_local_macros` lint (rust-lang/rust#79813) warns about this now and will reject it in a future release.

If Task A proves harnesses that the current code does not, port the narrow drawing to the repository in the same edit:
- add the helper, with a doc comment that states the folding argument in two sentences;
- update the module's `# Cost` section.

## Checks

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test -p risk`, natively.
- In WSL, with your own target dir: `cargo kani -p risk --exact --harness proofs::sells_are_never_gated`. It must report SUCCESSFUL, with no future-incompat warning.
- If narrow drawing was ported, re-run every newly proven harness on the repository's code and report its time.

## Done when

You report a table: harness, verdict, time and covers, for the current code and for the narrow version. You also report the diff of `proofs.rs`. Every number in the table must come from a run you made.
