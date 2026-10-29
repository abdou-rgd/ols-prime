# Brief — Matrix core at scale (iteration 3)

- **Agent:** `rust-engineer` (opus, high).
- **Runs in parallel with:** a `data-engineer` building the hourly panel (`panel.md`) and an `ml-engineer` building the H1 adversary (`adversary.md`). Neither touches `crates/`.
- **Motto:** we should be more ambitious, in what the core can do, never in what we claim about it.

## Goal

Make Matrix (`crates/ols-core`) fast and well-conditioned enough to serve every readout of iteration 3 from Python, through `olsprime_rs`. That means every hourly decision of the 274 universe members, 2018 to 2024, under several windows and penalties, plus the exact kernel readouts that H3 compares Matrix with.

Read first: `docs/DESIGN.md` §1, §3 and §7, and `docs/PROTOCOL.md`, section "Matrix on hourly bars (iteration 3)". That section is the pre-registration: the windows, the penalty grid and the arms come from there. `crates/ols-core/src/{matrix,stats,ridge}.rs` and `crates/olsprime-py/src/py/ols.rs` are the current code.

## The workload

- Entries: about 2.0 M, with p = 15 features (no constant column; the intercept is free, see below) and q = 4 outcomes (a one-hot class of 3, then `fwd_return`). They are pushed once, in time order. Times are `label_known_at` in ms UTC, non-decreasing, ties allowed.
- Queries: about 2.0 M decisions, each at `as_of` = its decision time, in non-decreasing order.
  - Each query is read under W = 3 windows (30 days, 365 days, all history) and R = 6 penalties.
  - At most 30 queries share an `as_of` (one per member pair).
- Window w at `as_of` d: the entries with `d − w < time ≤ d`. "All history" is every entry with `time ≤ d`.
- Kernel readouts (H3) over the 365-day window: about 263 k candidate entries per query.

## Deliverables

1. **Centred nodes** (DESIGN §3, planned for this iteration).
   - Nodes hold `(n, x̄, M_xx, ȳ, M_yx, M_yy)` and merge with the pairwise update of Chan, Golub and LeVeque (1979).
   - A range read is still a pure function of its range: the decomposition depends on `(start, end)` alone, and later pushes never change a read, not even in the last bit. H1 compares outputs bit for bit, so this is not optional.
   - Choose whether centred nodes replace the raw tree, with `range_stats` derived from them, or sit beside it. Criteria: exactness, purity, memory at 2 M entries, and every existing test green, including the bindings' tests. Say which you chose and why.
2. **The ridge in centred form**, with a free intercept and a penalty per coordinate.
   - `β = M_yx (M_xx + Λ)⁻¹` and `ŷ = ȳ + β (x_q − x̄)`, with `Λ = ρ · n_w · diag(s)`. Here `n_w` is the window's entry count, `s` a per-coordinate scale the caller passes (the fold's feature variances), and ρ a scalar.
   - This is ridge on features standardized inside the fold, solved exactly from the statistics.
   - A refusal criterion for a numerically singular `M_xx + Λ`, in the spirit of `RidgeConfig`'s τ, with its derivation or a stated conservative bound in the docs.
   - On request: the leverage `x̃_qᵀ (M_xx + Λ)⁻¹ x̃_q` with `x̃_q = x_q − x̄`, and the window's residual sum of squares per outcome at that penalty. H12 standardizes its recursive residuals with them.
3. **Batch readouts.** One call takes M queries, W windows and R penalties, and returns:
   - the forecasts, `M × W × R × q`;
   - the entry counts, `M × W`;
   - a status per cell, `M × W × R`: ok, empty window, or refused as singular. A refused cell is a status, never an exception that loses the whole batch.
   - It is parallel across queries (`std::thread::scope` is enough; ask before adding a dependency). The results are bit-identical whatever the thread count.
4. **Exact kernel readouts** over a window, for H3, on the same entry log.
   - Inputs: the query's features `x_q`, the window, a per-coordinate scale `σ` (distances use `u = x / σ`), a list of widths `τ` and a list of neighbour counts `k`.
   - Softmax: the Nadaraya-Watson estimate `Σ wᵢ yᵢ / Σ wᵢ` with `wᵢ = exp(−‖u_q − uᵢ‖² / (2τ²))`, computed stably (shift by the smallest distance).
   - kNN: the mean `y` of the k nearest entries in Euclidean distance on u. Break ties by entry index, lowest first, so the result is deterministic.
   - Outputs: `M × T × q` and `M × K × q`, plus the window counts. A window with fewer than k entries gives a status, not a guess.
   - Exact, f64, parallel and deterministic like the batch readouts.
5. **Python bindings** in `crates/olsprime-py`, following that crate's rules:
   - dtype checks before any value is read;
   - `MAX_ALLOC_BYTES` checked before allocating;
   - the times-in-ms check;
   - no panic crossing into Python;
   - the GIL released during the long calls if PyO3 0.30 allows it cleanly.

   Suggested surface: methods on `olsprime_rs.Matrix`, or a new class. Name them, document them, and give their exact signatures in your report: the `ml-engineer` codes against them next.
6. **Tests.**
   - Property tests:
     - centred merges against a two-pass reference;
     - the centred solve against raw ridge with an unpenalized intercept, on well-conditioned data;
     - shift invariance: add 10⁶ to a feature; the centred forecasts move by rounding only, where the raw path loses digits or refuses;
     - batch reads equal one-by-one reads;
     - purity after later pushes, bit for bit;
     - results independent of the thread count, bit for bit;
     - the window boundaries, `d − w < time ≤ d`, including equal times;
     - kernel readouts against a brute-force reference: exact equality for kNN under the tie rule, about 1e-12 relative for softmax;
     - the status codes.
   - Binding tests in a new file `python/tests/test_matrix_batch.py`. Other agents own the other test files.
7. **A benchmark example**, `cargo run --release -p ols-core --example matrix_bench`, at the workload above.
   - Report the wall times of the push, the batch readouts (per 10⁵ queries × W × R) and the kernel readouts (per 10⁴ queries at 263 k candidates), on 12 threads.
   - Target: the whole iteration's readouts in under about 20 minutes. If the kernels cannot get there, say so early. PROTOCOL fixes a fallback to the 30-day window for H3, and the orchestrator must choose it before any result.

## Constraints

- **Files.** `crates/ols-core/**`, `crates/olsprime-py/**`, `python/tests/test_matrix_batch.py`. Leave `crates/risk` alone: Kani runs on it in the background.
- **Behaviour.** Existing public APIs keep working, or are migrated together with every caller and test. `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo clippy -p olsprime-py --all-targets --features python -- -D warnings` and `cargo fmt --all --check` stay green. So does `cargo run -p backtest --example golden -- --check`.
- **Python.** Rebuild the extension with `cd python && uv sync` only when you need it. The other agents run `uv run --no-sync`. On Windows a loaded `.pyd` cannot be replaced: if `uv sync` fails on a locked file, wait and retry rather than killing anything.
- **Scripts.** Write scripts with a file tool, never through shell heredocs.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\rust-core\`.
- **Git.** Run no git command that changes state, and do not commit.
- **Docs.** Do not edit `docs/` or `CLAUDE.md`. Put what the docs need in your report.

## Done when

- Every deliverable above is in place, and every check listed under Behaviour is green.
- The benchmark has run, and its numbers are in your report.

## Report (short)

- The design choice for centred nodes, and the refusal criterion with its derivation.
- The exact Python signatures and the meaning of each output and status.
- The benchmark table.
- What is not done or not verified.
- One bolder next step.
