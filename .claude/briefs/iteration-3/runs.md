# Brief — Official runs of iteration 3

- **Agent:** `experiment-runner` (sonnet, medium).
- **Runs after:** the code is complete, the orchestrator's full verification has passed, and CI is green on GitHub on every platform for the commit named below. Nothing else runs on the machine meanwhile, so the timings mean something.
- **Motto:** we should be more ambitious, in what we run, never in what we claim.

## Goal

Run the official experiments of iteration 3, check that each one reproduces, and append their rows to `docs/RESULTS.md`.

Read first:
- `docs/PROTOCOL.md`, "Leakage tests" (including "The adversary") and "Matrix on hourly bars (iteration 3)". Every decision rule you apply is written there.
- `docs/HYPOTHESES.md`.
- `docs/RESULTS.md`, whose rows you match in style.
- `docs/reports/report-1.md`, the model for `docs/reports/report-2.md`.

## Runs, in this order

Commit under test: **filled in by the orchestrator**. Check that `git rev-parse HEAD` equals it and `git status --porcelain` is empty before each run. If either fails, stop.

1. **Report n°1 after the vault move.**
   - Run `cd python && uv run --no-sync python -m olsprime.bench.report1 --check ../runs/2026-10-28-report1`.
   - Expected: every number reproduces, and only the commit, code digest and data manifest lines differ.
   - Record exactly which lines differ.
2. **The panel H1.**
   - Run `uv run --no-sync python -m olsprime.bench.panel_invariance`.
   - It must hold: every honest arm has 0 violations, and every planted leak is on its predicted pattern.
   - **If it does not hold, stop.** Run nothing else, and report.
3. **The adversary.**
   - Run `uv run --no-sync python -m olsprime.bench.adversary` (N = 10,000).
   - Run it a second time to a second folder, and compare the two runs' `adversary.json` byte for byte. Timing fields in `run.json` may differ.
4. **Report n°2.**
   - Run `uv run --no-sync python -m olsprime.bench.report2`, then `--check` on its folder.
   - Copy `report-2.md` to `docs/reports/report-2.md`, as Report n°1 was.

Record each run's wall time.

## Rows

Append to `docs/RESULTS.md`, never editing an existing row. One row each for:
- the panel H1;
- the adversary (descriptive, under H1);
- H2, H3, H10, H11 and H12;
- Report n°2, descriptive.

Each row gives:
- the run folder and the commit;
- the variants tried so far: 10 before this iteration, plus the 8 arms of Report n°2;
- the numbers with their intervals;
- the verdict by the registered rule, word for word as the rule decides it.

A hypothesis that was not run gets no row, and you say so.

## Constraints

- **Do not edit code.** If a run fails, stop and report the error. The orchestrator decides.
- **Docs.** Edit only `docs/RESULTS.md` and `docs/reports/report-2.md`.
- **Scripts.** Write scripts with a file tool, never through shell heredocs.
- **Git.** Run no git command that changes state.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Report (short)

- The table of runs: command, folder, wall time, reproduced or not.
- The rows you appended.
- Anything odd.
- One bolder next step.
