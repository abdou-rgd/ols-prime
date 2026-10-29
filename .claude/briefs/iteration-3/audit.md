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

Phase 1 runs in two passes, one auditor at a time.
- **Pass 1a**, while the Matrix core, the arms and the seal are still being built, covers items 0, 1 and 5. A flaw in the design is cheapest to fix before the arms implement it: a change to the pre-registration is still allowed, dated in the change log, as long as no result exists.
- **Pass 1b**, once the core and the seal are complete and while the arms are being built, covers items 2 and 7. It re-checks the fixes of pass 1a: `docs/PROTOCOL.md` and `docs/HYPOTHESES.md` (change log entries "after audit pass 1a"), the panel's month (`_month_expr` on `open_time`, with its test), and the adversary's report.
- **Pass 1c**, once the arms, the panel H1 and Report n°2 are complete, covers items 3, 4 and 6, and re-checks the fixes of pass 1b.

0. **The pre-registration itself** (`docs/PROTOCOL.md`, "The adversary (iteration 3)" and "Matrix on hourly bars (iteration 3)", with the change log). Attack the design before it is built.
   - Can each decision rule be met vacuously, or fail to be computable? Does each verdict follow from numbers the run produces?
   - Are the folds causal? Check the σ schedule, the purge at `fit_time − 24 h`, and the prequential records the selections read.
   - Is Diebold-Mariano with 23 lags and HLN right for one-day outcomes that overlap, once scores are averaged per decision hour?
   - Is the bootstrap of the BSS, a ratio, done right? Is the variant count right?
   - Does the CUSUM's timing guarantee that an alarm acts only once the labels that raise it are known?
   - Is the panel H1's restricted recomputation sound by design? Do its planted leaks have exact predicted patterns?
1. **The panel** (`python/olsprime/features/panel.py`). It is committed on `iteration-3` (89a114b). The orchestrator later added `pair_rows` and `assemble` for the panel H1.
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
5. **The adversary** (`python/olsprime/bench/adversary.py`, `python/tests/test_adversary.py`, and the small change to `bench/invariance.py`: `perturbation_at`).
   - Its smoke run on the real data, a smoke run and not a result, is in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\adversary\smoke-n200-final\`. Use it to check what the report claims and how it says it.
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
- Added on 2026-10-29, after the runs:
  - **The ECE tolerance.** Commit `31c8005` changed one evaluation check between the two official commits: Report n°2's ECE cross-check, from a fixed `rel_tol=1e-9` to 4·n·ε. Review it as a phase 1 change. Does the check still catch a lost row or a row in another bin at the panel's size? Does the PROTOCOL entry disclose the one number the stopped run showed?
  - **The hand-offs of passes 1c to 1f**, checked on the real results:
    - the previewed and unseen hours add up to the scored hours, over 18 previewed months of 72;
    - every undefined cell of the previewed months is explained;
    - every share of identical hours is read with its count of informative hours;
    - the CUSUM's days left out add up;
    - the panel H1's patterns are the predicted ones;
    - every `--check` passed;
    - the previewed part agrees with the 18-month smoke run as far as the two `report2.json` files allow.
  - **The 42 adversary misses that are not constant**, which Abdou asked about. They are the leaks of `adversary.json` whose `h1_pattern` has no `X` and whose `constant_on_original_rows` is false.
    - Classify them.
    - Say for each whether H1 cannot see it by design, or whether it is a hole H1 should close.
    - Propose a planted family for the next adversary run if a pattern emerges.
  - **The data manifest digest**, from `7c5f2397…` after the move to `0ee7ac0f…` in the runs.
  - **The 94 unscored hours around 2020-07-01**, and the Coverage paragraph of report-2.md.

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
