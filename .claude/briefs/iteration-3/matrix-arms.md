# Brief — Matrix arms, panel H1 and Report n°2 (iteration 3)

- **Agent:** `ml-engineer` (opus, high).
- **Runs after:** the `rust-engineer`'s Matrix core (`matrix-core.md`) and the `data-engineer`'s hourly panel (`panel.md`). Their APIs are summarized at the end of this brief.
- **Runs in parallel with:** an `ml-engineer` fixing the H1 adversary, in `python/olsprime/bench/adversary.py` and `python/tests/test_adversary.py`, and the `rust-engineer` on docs and stub follow-ups in `crates/olsprime-py/`, `python/tests/test_rust_bindings.py` and `docs/DESIGN.md`. Touch none of those files.
- **Motto:** we should be more ambitious, in the test, never in the claim. A clean negative result is a result.

## Goal

Run Matrix and its OLS-attention readout on hand-made features, over the hourly bars of the 274 universe members, and test H2, H3, H10, H11 and H12 on the development period, exactly as pre-registered.

Read first, in this order:
1. `docs/PROTOCOL.md`, "Matrix on hourly bars (iteration 3)". **This is the pre-registration. It fixes the panel, the entries, the windows, the penalty grid, the probability map, the folds, every arm, the scoring, the decision rules and the panel H1.** Do not change any of it. If something cannot be built as written, or is ambiguous, stop and ask the orchestrator before writing results. A change must be dated in the change log before any result exists.
2. `docs/HYPOTHESES.md` (H2, H3, H10–H12 and the change log) and `docs/DESIGN.md` (§1, §3, §5 and §7).
3. The code you build on:
   - `python/olsprime/features/panel.py` (the panel);
   - the new `olsprime_rs` Matrix methods (the core);
   - `python/olsprime/eval/stats.py` (DM with HLN, the stationary bootstrap, Politis-White, ECE, BSS);
   - `python/olsprime/bench/{invariance,perturb,pipeline,provenance,report1}.py` (how H1 and Report n°1 are built, provenance and `--check` included).

## Deliverables

1. **`python/olsprime/models/matrix.py`**: the eight arms of the pre-registration.
   - Arms: `climatology-panel`, `matrix-select`, `matrix-1y`, `softmax-1y`, `knn-1y`, `matrix-adaptive`, `matrix-stack` and `matrix-cusum`.
   - **Causality by construction.** Push every entry, stamped with its `label_known_at`, into one Matrix. Read every query with `as_of` = its `decision_time`. A read sees only entries with `time ≤ as_of`, so an outcome joins the memory only once it is known, inside the test month too.
     - Fit-time choices use only rows whose label is known by `fit_time − 24 h`, and the prequential forecasts made at those rows' own decision times.
     - Write this down in the module docstring, with the one-line argument for why each output at d depends only on bars final by d.
   - **Prequential forecasts.** A candidate (window, ρ, or τ, or k) needs its forecast at every past row to be selected. PROTOCOL's Folds now register how, after audit pass 1a:
     - the σ schedule: σ of month M from the entries known by M's first instant minus 24 h, used by every readout of a row of month M;
     - the record: every candidate of every arm forecasts every row from 2017-11, at the row's decision time, with the σ of the row's month; rows before 2019-01 only build the record;
     - the selection rows of a fit: decided in the 365 days before it, with labels known by the fit time minus 24 h;
     - the defaults, for a selection (at a fit, or at a decision of `matrix-adaptive`) that finds fewer than 1,000 rows: window 365 d, ρ = 10⁻², τ = √15, k = 200, α all on 365 d. Count them per arm.
     - State all of it in the docstring, with the one-line causal argument for each.
   - **H12.** `matrix-cusum` keeps its σ, window and ρ between refits. It needs its own readouts under its own schedule, including the `fwd_return` readout's leverage and residual mean square. PROTOCOL's CUSUM paragraph was rewritten after audit pass 1a: implement its Input, Timing and Threshold exactly.
     - The daily raw mean over members is divided by s, the standard deviation of the raw daily values over the days of the last fit's selection rows, recomputed with that fit's window and ρ.
     - Day d's value enters at 01:00 UTC on d + 1. An alarm acts at the first 00:00 UTC after that, with a refit on rows known by the fit time minus 24 h, and the CUSUM restarts at 0.
     - The threshold gives an ARL of **365 days** under independent N(0, 1) values. Set it by simulation: seeded, at least 10⁵ runs, the result given to 3 decimals. Report it and its simulated ARL before running anything on real data.
     - H12 now needs at most 36 fits, the first included.
     - "Residual mean square": take rss / (n_w − p − 1), and say so.
   - **Fallback.** A cell the core refuses (empty or singular) falls back to that arm's climatology. Count the fallbacks per arm.
2. **`python/olsprime/bench/panel_invariance.py`**, the pre-registered panel H1: `uv run --no-sync python -m olsprime.bench.panel_invariance [--out DIR]`.
   - **Set-up.**
     - The sub-panel comes from `h1_subpanel_members`.
     - 8 cut hours, seeded, in the test months: one at a month's first hour, one at a month's last hour.
     - Every member's bars after the cut are perturbed with `perturb_after`, at that pair's own row for the cut, in both directions, with H1's seeds scheme.
     - The panel is rebuilt from the perturbed bars.
   - **Comparison.** Every arm's outputs at decision times at or before the cut are compared bit for bit, along with each fold-time choice.
   - **Members with no bar after a cut** (`perturb_after` raises on them): leave them out of that cut explicitly, and count them in the report.
   - **A limit to state in the report:** only the sub-panel's bars are perturbed, so an input built from other members' bars would not move. No arm has one; keep it so.
   - **Planted online leaks**, each with an exact predicted pattern of cuts. Write the prediction in code, as `invariance.predicted` does, and check it:
     - entries stamped with their decision time;
     - a window that ends one hour after the decision;
     - σ fitted on the whole test month, predicted at 7 of the 8 cuts: not at the month's last hour, where every feature of the month is already known;
     - a fit-time selection without the purge;
     - a kernel whose candidates reach one label horizon past the decision.
   - **Cost.** You may limit the recomputation to what the compared outputs depend on: the cut's month and its fit-time inputs. Say exactly what is recomputed, and why an honest arm's compared outputs cannot depend on anything left out.
   - **Outputs.** `h1-panel.json` and `h1-panel.md`, with provenance, to `runs/<UTC date>-h1-panel/`.
3. **`python/olsprime/bench/report2.py`**, Report n°2 as one deterministic command: `uv run --no-sync python -m olsprime.bench.report2 [--out DIR] [--check DIR]`.
   - **Pipeline.** It builds the panel, runs every arm, and scores the test months 2019-01 → 2024-12 on the pre-registered scored rows. Every statistic is computed on the hourly series.
   - **Statistics.**
     - Log-loss, Brier, BSS and ECE per arm, with stationary-bootstrap intervals.
     - DM for each pre-registered comparison.
     - For each comparison, the second test now registered: the stationary bootstrap of the mean hourly log-loss difference, with the Politis-White length of that difference, 2,000 resamples, a 95 % percentile interval. Build it in `report2.py` from `stats.optimal_block_length` and `stats.stationary_bootstrap_indices`, and change nothing in `stats.py`.
     - Each difference also as a share of climatology's log-loss.
     - For H12, also the bootstrap's 95th percentile of the mean difference against the margin.
     - The verdict of each hypothesis, by its rule. A comparison passes only when DM gives p < 0.05 and the bootstrap interval lies on the same side of 0.
     - The fallback counts, the default counts, the number of CUSUM fits, and the CUSUM's s at each fit.
     - The descriptive readout: for the seeded 1,000 decisions of `matrix-1y`, the share of negative weights, and the contributions of the last 30 days and of older entries. These come from `weights` and `contribution`, or their batch equivalent.
   - **Variants.** Report "variants tried so far": 10 before this iteration, plus the 8 arms.
   - **Outputs.** `report-2.md`, `report2.json`, `run.json` and `config.json` to `runs/<UTC date>-report2/`, with the same provenance block as Report n°1: commit, code digest, data manifest digest, each Parquet file's SHA-256 and row counts.
   - **`--check`** re-runs it and compares every file byte for byte, as Report n°1 does.
   - **Runtime.** Report it. Aim for under about an hour on the 12-thread machine.
   - **If the one-year kernels cannot finish in time,** say so before any result. The pre-registration's fallback (the 30-day window for all three H3 arms) is then chosen, and dated in `PROTOCOL.md`'s change log by the orchestrator.
4. **Tests**, in new files `python/tests/test_matrix_arms.py`, `test_panel_invariance.py` and `test_report2.py`, on synthetic panels:
   - a numpy reference of the Matrix reads (centred ridge with the free intercept and the per-coordinate penalty, the windows, the kernels) agrees with the Rust path on small data;
   - each arm's selection logic is correct on hand-built losses;
   - perturbing the future leaves every output at or before the cut unchanged, bit for bit, for every arm;
   - each planted leak is caught at its predicted cuts and nowhere else;
   - the CUSUM simulation is reproducible;
   - the decision rules give the right verdict on hand-built statistics;
   - Report n°2's command is deterministic on a tiny synthetic store.
5. **Smoke runs only.** Run small smoke runs on the real data, for example a few months and a few members, to measure runtime. **Do not run the official panel H1 or Report n°2.** The `experiment-runner` does that after CI is green. Report your smoke numbers as smoke, not as results.

## Constraints

- **Files.**
  - New: `python/olsprime/models/matrix.py`, `python/olsprime/bench/panel_invariance.py`, `python/olsprime/bench/report2.py`, and the three test files.
  - Exports only: `python/olsprime/models/__init__.py`.
  - Ask before changing any other file. Ask the `data-engineer`'s owner (the orchestrator) before changing the panel, and the orchestrator before changing `eval/stats.py`.
- **Leaky controls** live in `panel_invariance.py`, labelled as in `leaks.py`, and nothing in `models/` imports them.
- **Python.** Use `uv run --no-sync`, unless the orchestrator tells you the extension must be rebuilt.
- **Scripts.** Write scripts with a file tool, never through shell heredocs.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\matrix-arms\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- The three modules and their tests are in place, and the whole Python suite is green.
- The smoke runs have run, and their runtimes are in your report.
- The CUSUM threshold is computed and reported.

## Report (short)

- How each arm is computed.
- The CUSUM threshold.
- The smoke runtimes, and the projected official runtimes.
- What the panel H1 recomputes.
- Every point where the pre-registration needed an interpretation, and the one you took.
- What is not done.
- One bolder next step.

## APIs of wave 1

Filled in by the orchestrator from the wave-1 reports, and re-checked against the code.

### The hourly panel (`olsprime.features.panel`, done)

- `hourly_panel(universe=None, *, h=24, first=Month(2017, 10), last=Month(2024, 12), symbols=None, data_dir=None) -> pl.DataFrame`.
  - One row per (member, hour) in the member months only, sorted by `(decision_time, symbol)`.
  - `universe` defaults to the sealed top-30 universe file; `symbols` keeps only those members (the sub-panel); `data_dir` defaults to `paths.data_dir()`. A member with no stored 1 h file raises.
  - Bars are cut to `known_at < DEVELOPMENT.end` before features and labels, so a label that needs a 2025 bar is null. A `last` at or after 2025-01 is refused, and so is any month in the vault.
- `PANEL_COLUMNS`, in order:
  - `symbol`, `open_time`, `decision_time` (bar t's `known_at`, its close), `month` (the month of `open_time`, Datetime ms UTC, after audit pass 1a; the real panel's fingerprint did not change);
  - the 15 `FEATURE_COLUMNS` of `features.basic`;
  - `threshold` (known at t, from past volatility);
  - the label columns `entry_time` (open of t + 1), `exit_time` (open of t + 25), `label_known_at` (that exit open), `fwd_return`, `label` (UInt8: 0 down, 1 neutral, 2 up);
  - `complete` (every feature and the label non-null).
- `member_months(universe, *, first, last)`: one row per (symbol, month) of membership.
- `h1_subpanel_members(universe, *, size=10)`: the members with the most member months, ties by symbol. On the real store: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, LTCUSDT, TRXUSDT, LINKUSDT, MATICUSDT, DOGEUSDT.
- `fingerprint(panel) -> str`: SHA-256 over every column. On the real store: `e64b39d8784af6039dc4810a1247fd957af94a8fe76240563855c45ba0442508`.
- Real store, re-checked by the orchestrator:
  - 1,655,881 rows over 2,273 member months, 2017-10 → 2024-12;
  - 233 distinct pairs: the other 41 of the 274 members first join in 2025;
  - 1,631,784 rows labelled, 1,591,462 complete;
  - between 2 and 30 members a month;
  - built in 12 to 41 s, with peak memory near 2 GB.
- **For the panel H1**, added by the orchestrator. The real panel's fingerprint is unchanged, and `test_panel.py` covers both functions.
  - `pair_rows(bars, months, *, h=24) -> pl.DataFrame` builds one pair's rows in its member `months` from its bars, as `load_klines` returns them; extra columns are dropped. The cut to `known_at < DEVELOPMENT.end` happens inside, before features and labels.
  - `assemble(parts) -> pl.DataFrame` joins the pairs' rows into the panel, with `complete`, the column order and the sort.
  - `hourly_panel` is exactly `assemble(pair_rows(load_klines(s), months_of_s) for each member s)`. So the panel H1 can perturb each sub-panel member's bars in memory with `perturb_after`, at that pair's own row for the cut, and rebuild with these two functions. It needs no perturbed store on disk.

### The Matrix core (`olsprime_rs`, done)

Filled in from the `rust-engineer`'s report and re-checked against `crates/olsprime-py/olsprime_rs.pyi`, which is the reference.

- **Design.** Matrix keeps a second tree of centred nodes (n, x̄, M_xx, ȳ, M_yx, M_yy) beside the raw tree, merged by Chan-Golub-LeVeque. Every raw read keeps its bits. A centred read depends only on (start, end), and later pushes never change it, bit for bit.
- **Entries.** `Matrix(p=15, q=4)`; `extend(times, xs, ys)` pushes n entries at once, in order: times are `label_known_at` in ms, ys are the one-hot class followed by `fwd_return`.
- **Windows.** `window_range(as_of, window=None) -> (start, end)`: the entries with as_of − window < time ≤ as_of, window in ms (> 0), `None` for all history.
- **One read.**
  - `centred_query(x_q, start, end, rho, scale) -> CentredReadout` solves ridge with a free intercept, Λ = ρ·n·diag(scale). **`scale` holds the fold's variances σⱼ²**, not the standard deviations.
  - The readout has `forecast` (ȳ + β(x_q − x̄), shape (q,)), `leverage` (x̃ᵀ(M_xx + Λ)⁻¹x̃), `rss` (q,) at this penalty, `mean_x` and `a`.
  - `centred_weights(readout)` gives one weight per entry; they sum to 1, and weights @ ys is the forecast. `centred_contribution(readout, start, end)` gives a sub-range's exact contribution. These serve the descriptive readout.
- **Batches**, in parallel without the GIL, with the same bits whatever the thread count or the order of the queries:
  - `ridge_batch(x_q (M,p), as_of (M,) int64, windows [span or None], rhos (R,), scale (p,), *, leverage=False, rss=False, threads=0)` returns a dict: `forecast` (M,W,R,q), `status` (M,W,R) uint8, `start`/`end`/`count` (M,W) int64 with count = n_w, and `leverage` (M,W,R) and `rss` (M,W,R,q) on request. Each cell has the bits of `centred_query`. One output array may take at most 1 GiB, so call it on chunks of at most about 10⁶ queries.
  - `kernel_batch(x_q (M,p), as_of (M,), window, sigma (p,), taus (T,), ks (K,) int64, *, threads=0)` returns `softmax` (M,T,q), `softmax_status` (M,T), `knn` (M,K,q), `knn_status` (M,K), and `start`/`end`/`count` (M,). **`sigma` holds the standard deviations σⱼ**: u = x / σ. Softmax weights are exp(−‖u_q − uᵢ‖²/(2τ²)); kNN takes the mean y of the k nearest, ties to the lowest entry index.
- **Status codes:** `STATUS_OK` 0, `STATUS_EMPTY` 1, `STATUS_SINGULAR` 2 (refused by τ), `STATUS_NON_FINITE` 3, `STATUS_TOO_FEW` 4 (k larger than the window). A cell that is not OK is NaN in every output, and falls back to the arm's climatology. Bad inputs raise errors; they never become statuses.
- **Refusal.** A window is refused when λ_min(D^-1/2 A D^-1/2) ≤ τ, with A = M_xx + Λ. τ is a first-order bound on the rounding, with conservative constants (see `CentredFit` in `crates/ols-core/src/centred.rs`). An exactly singular design is always refused.
- **Speed**, measured on this machine, 12 threads, 2M entries, under load from other agents:
  - pushing 2M entries takes 0.64 s;
  - ridge: 0.19 s per 10⁵ queries × 3 windows × 6 penalties, with leverage and rss, so a few seconds for the whole panel;
  - kernels over the 365-day window: 7.55 s per 10⁴ queries at 262,800 candidates (T = 3, K = 3), about 25 min for 2M queries.
  - **The orchestrator's decision:** the extension keeps its portable build, so H3 runs at the 365-day window, as registered. The fallback to 30 days is not taken. Batch the kernel queries so that the record of every candidate costs one pass.
