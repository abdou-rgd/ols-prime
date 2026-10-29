# Brief — Matrix arms, panel H1 and Report n°2 (iteration 3)

- **Agent:** `ml-engineer` (opus, high).
- **Runs after:** the `rust-engineer`'s Matrix core (`matrix-core.md`) and the `data-engineer`'s hourly panel (`panel.md`). Their APIs are summarized at the end of this brief.
- **Runs in parallel with:** a `data-engineer` on the physical seal (`seal.md`), and possibly an `ml-engineer` finishing the H1 adversary (`adversary.md`).
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
   - **Prequential forecasts.** A candidate (window, ρ, or τ, or k) needs its forecast at every past row to be selected.
     - Use one σ schedule for every month, including the warm-up months: σ of month M is computed from the entries known by M's start minus 24 h. A candidate's forecast at row r then uses the σ of r's month.
     - This makes every candidate's prequential record a well-defined causal series, and lets the selection read them.
     - State it in the docstring. It is an implementation of the pre-registration, not a new choice. If you think it is a choice, ask.
   - **H12.** `matrix-cusum` keeps its σ, window and ρ between refits. It needs its own readouts under its own schedule, including the `fwd_return` readout's leverage and residual mean square.
     - Set the CUSUM threshold by the simulation the pre-registration describes: seeded, at least 10⁵ runs, with the result given to 3 decimals.
     - Report the threshold and its simulated ARL before running anything on real data.
   - **Fallback.** A cell the core refuses (empty or singular) falls back to that arm's climatology. Count the fallbacks per arm.
2. **`python/olsprime/bench/panel_invariance.py`**, the pre-registered panel H1: `uv run --no-sync python -m olsprime.bench.panel_invariance [--out DIR]`.
   - **Set-up.**
     - The sub-panel comes from `h1_subpanel_members`.
     - 8 cut hours, seeded, in the test months: one at a month's first hour, one at a month's last hour.
     - Every member's bars after the cut are perturbed with `perturb_after`, at that pair's own row for the cut, in both directions, with H1's seeds scheme.
     - The panel is rebuilt from the perturbed bars.
   - **Comparison.** Every arm's outputs at decision times at or before the cut are compared bit for bit, along with each fold-time choice.
   - **Planted online leaks**, each with an exact predicted pattern of cuts. Write the prediction in code, as `invariance.predicted` does, and check it:
     - entries stamped with their decision time;
     - a window that ends one hour after the decision;
     - σ fitted on the whole test month;
     - a fit-time selection without the purge;
     - a kernel whose candidates reach one label horizon past the decision.
   - **Cost.** You may limit the recomputation to what the compared outputs depend on: the cut's month and its fit-time inputs. Say exactly what is recomputed, and why an honest arm's compared outputs cannot depend on anything left out.
   - **Outputs.** `h1-panel.json` and `h1-panel.md`, with provenance, to `runs/<UTC date>-h1-panel/`.
3. **`python/olsprime/bench/report2.py`**, Report n°2 as one deterministic command: `uv run --no-sync python -m olsprime.bench.report2 [--out DIR] [--check DIR]`.
   - **Pipeline.** It builds the panel, runs every arm, and scores the test months 2019-01 → 2024-12 on the pre-registered scored rows. Every statistic is computed on the hourly series.
   - **Statistics.**
     - Log-loss, Brier, BSS and ECE per arm, with stationary-bootstrap intervals.
     - DM for each pre-registered comparison.
     - The verdict of each hypothesis, by its rule.
     - The fallback counts, and the number of CUSUM refits.
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
  - `symbol`, `open_time`, `decision_time` (bar t's `known_at`, its close), `month` (the month of `decision_time`, Datetime ms UTC);
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
  - built in 18 to 41 s, with peak memory near 2 GB.
- **For the panel H1**, added by the orchestrator. The real panel's fingerprint is unchanged, and `test_panel.py` covers both functions.
  - `pair_rows(bars, months, *, h=24) -> pl.DataFrame` builds one pair's rows in its member `months` from its bars, as `load_klines` returns them; extra columns are dropped. The cut to `known_at < DEVELOPMENT.end` happens inside, before features and labels.
  - `assemble(parts) -> pl.DataFrame` joins the pairs' rows into the panel, with `complete`, the column order and the sort.
  - `hourly_panel` is exactly `assemble(pair_rows(load_klines(s), months_of_s) for each member s)`. So the panel H1 can perturb each sub-panel member's bars in memory with `perturb_after`, at that pair's own row for the cut, and rebuild with these two functions. It needs no perturbed store on disk.

### The Matrix core (`olsprime_rs`)

Filled in from the `rust-engineer`'s report.
