# Brief — Official runs of H1 and Report n°1 (iteration 2)

- **Agent:** `experiment-runner` (sonnet, medium).
- **When:** after the bench (`bench.md`) is merged into the working tree by the orchestrator.
- **Updated 2026-10-28.** The runs are made on a clean, committed tree, so that the provenance block (git commit, code digest, data manifest) names exactly the commit that produced the numbers.
  - The tree contains the bench fixes of `bench-fixes.md`, `bench-fixes-2.md` and `bench-fixes-3.md`.
    - H1 now perturbs every column and every ratio, in both directions.
    - The two runs of a cut give bar t + 1 opposite candle shapes: body-dominant when the future is multiplied by 3, wick-dominant when it is divided by 3.
    - H1 carries planted leaks as controls, each with an exact predicted pattern, and a positive control.
    - A `?` cut is predicted from the original bars, with a declared reason. A cut that fails to cross without a declared reason fails H1.
    - H1 writes a `provenance` block in `h1.json` and at the end of `h1.md`.
  - H1 counts only if every honest model has 0 violations, every control sits exactly on its predicted pattern, and the positive control passes. The h1.md headline says so. Report every `?` cut it lists, with its reason.
  - The data is final: `python -m olsprime.manifest` must give digest `230e28e830f728ec7b3974c359f944e7d7bc28f8268694ef00448e672f52f3cd` (843 files). Stop and report if it does not.
  - Run `uv sync` once at the start, as step 1 says. The orchestrator makes sure no other agent is using the environment then.
- **Motto:** we should be more ambitious: use the compute we have, and record everything.

## Goal

Make the official runs of the two commands the bench provides, prove they reproduce, and prepare the rows of `docs/RESULTS.md`. **Append the rows only when the orchestrator says the audit has passed.** `CLAUDE.md` requires the `leakage-auditor` to clear every change before any result is reported.

## Runs (all in `python/`, vault sealed)

1. `uv sync`, then `uv run pytest`. Stop and report if anything fails.
2. `uv run python -m olsprime.manifest`. Record the digest.
3. `uv run python -m olsprime.bench.invariance`: H1 on BTCUSDT and ETHUSDT, 1d.
4. `uv run python -m olsprime.bench.report1`: the first official run of Report n°1.
5. `uv run python -m olsprime.bench.report1 --check <dir of run 4>`: the reproducibility check, which must pass byte for byte.
6. For every run, record the wall time, the git commit and the `git status --porcelain` state.
   - The tree must be clean before the first run.
   - Afterwards, only `runs/`, which git ignores, and the regenerated `docs/reports/report-1.md` may have changed.
   - Check that the commit in `h1.json` and in Report n°1's provenance block is the one `git rev-parse HEAD` gives.

Run exactly these commands, with their defaults. Any other configuration is an extra variant: do not run it.

## What to prepare (do not append yet)

Draft the rows in `runs/<UTC date>-report1/results-rows.md`, in the format of `docs/RESULTS.md`:

| Date | Hypothesis | Experiment | Variants tried so far | Result | Verdict |
|---|---|---|---|---|---|

- **H1:** violations per honest model (all zero, or not); the cheater flagged at how many cuts out of how many; whether the seal refuses the cheater's last decision.
- **H9:** one row per venue named by H9 (Coinbase, Kraken). Each row gives, per pair, buy and hold's net Sharpe with its interval, the best active baseline's, and the verdict.
- **Report n°1, descriptive:** one row with the headline at each venue. For example: "net Sharpe of buy and hold [interval], of the best active baseline [interval], DSR of the best".
- **Variants tried so far:** the count the report computes, cumulative over the project. No experiment ran before this iteration.
- **Pointers:** every row links to `docs/reports/report-1.md` and names its run folder.

## Report (short)

- A table of runs: command, wall time, exit status, output folder.
- The manifest digest and the result of `--check`.
- The draft rows, verbatim.
- Anomalies: NaN, a Sharpe above 3, a cheater that is not flagged, any honest violation. Flag them for `leakage-auditor`.
- One bolder next step.

## Constraints

- **Files.** Only `runs/` (gitignored), the generated `docs/reports/report-1.md`, and later `docs/RESULTS.md`, append-only. Change no code; report bugs with their traceback.
- **Never.** Never open the vault.
- **Git.** Run no git command that changes state.
