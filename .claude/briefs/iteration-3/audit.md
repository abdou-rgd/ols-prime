# Brief — Audit of iteration 3

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Phases:**
  - Phase 1 reviews the code, before the real vault move and before any official run.
  - Phase 2 reviews the results and the claims, after the runs.
  - One auditor at a time: several in parallel hit the session's usage limit in iteration 2.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

No result of iteration 3 is reported until you have tried to break it. Hunt for:
- look-ahead and leakage;
- survivorship;
- statistical mistakes;
- holes in the vault's seal;
- overclaiming.

Every finding gets a severity (Blocker, Major, Minor), a concrete failure case and, where you can, a probe that shows it.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md` ("Splits", "Leakage tests", "Matrix on hourly bars (iteration 3)" and the change log);
- `docs/HYPOTHESES.md`;
- `docs/DESIGN.md` (§1, §3, §7);
- `docs/DATA.md`;
- `docs/ITERATIONS.md`, the iteration 3 brief and iteration 2's lessons.

Then the diff of the branch `iteration-3` against `main` (`git diff main...iteration-3` and the working tree).

## Phase 1: the code

1. **The panel** (`python/olsprime/features/panel.py`).
   - Point-in-time membership: no row outside member months, nothing before a pair's first membership.
   - The cut at the end of the development period comes before features and labels.
   - Labels found by time, with h = 24.
   - The 10-member sub-panel.
2. **The Matrix core** (`crates/ols-core`, `crates/olsprime-py`).
   - Each read is a pure function of its range, bit for bit, and independent of the thread count.
   - The window bounds `d − w < time ≤ d`.
   - The centred merges and the centred solve, with its refusal criterion.
   - The kernel readouts' windows and tie rule.
   - Status codes instead of exceptions.
3. **The arms** (`python/olsprime/models/matrix.py`).
   - Every output at d depends only on bars final by d.
   - Fit-time choices use only labels known by `fit_time − 24 h`, and the prequential forecasts made at their own decision times.
   - The σ schedule.
   - The CUSUM: its input, its timing (an alarm acts only after the labels that raise it are known) and its threshold.
   - The fallbacks.
   - Does anything read the validation year or the vault?
4. **The panel H1** (`python/olsprime/bench/panel_invariance.py`).
   - Are the predicted patterns of the planted leaks right?
   - Does the restricted recomputation leave out anything a leak could hide in?
   - Plant at least two leaks of your own, on synthetic data, that the bench should catch, and say whether it does.
5. **The adversary** (`python/olsprime/bench/adversary.py`).
   - Is the fast path really H1's rule?
   - Are the honest twins really honest?
   - Is the Wilson interval right, and is "share of this generator's leaks" the only claim made?
   - Which families of leak does the generator never produce?
6. **Report n°2's statistics** (`python/olsprime/bench/report2.py`, `eval/stats.py`).
   - Hourly aggregation before inference.
   - DM with 23 lags and HLN.
   - The bootstrap of the BSS as a ratio.
   - The decision rules, word for word against the pre-registration.
   - The variant count.
7. **The physical seal** (`python/olsprime/vault.py`, `protocol.py`, `binance_vision.py`, `manifest.py`, the guards).
   - Can the move lose or expose a file?
   - Does any path read a vault row?
   - Can the open path be reached without both keys, or at stage 2 without the identity?
   - Do the downloader and `build_parquet` refuse every route back into `data/`?
   - Proofs of a hole run on fake trees only. **Never read a vault row, and never set either key.**

Report phase 1 with every finding. The orchestrator fixes or logs each one, then asks for a re-check of the fixes.

## Phase 2: the results

- Each row added to `docs/RESULTS.md`: does the number match the run's files, and does the verdict follow the registered rule?
- Each claim in `docs/ITERATIONS.md`'s outcome and in the reports: is it stated no stronger than the evidence?
- The vault move: do the logs show that only renames happened, and that sealed reads were checked equal? Is the "stage" claimed the stage reached?

## Constraints

- **Read-only.** You may run the tests, and probes in your scratch folder (`C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\audit\`), on synthetic or fake data only.
- Write probes with a file tool, never through shell heredocs.
- Use `uv run --no-sync`.
- No git command that changes state.

## Report

Phase 1 or phase 2:
- the findings, by severity, each with its failure case and probe;
- what you checked and found sound;
- what you could not check;
- one bolder next step.
