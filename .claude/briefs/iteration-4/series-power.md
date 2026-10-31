# Brief — Report n°2's series, and the size and power of the pass rule (iteration 4)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:** a `rust-engineer` (getters on `CentredReadout`, in `crates/` and `python/tests/test_readout_beta.py`) and another `ml-engineer` (the encoder, in `python/pyproject.toml`, `python/uv.lock`, `.github/workflows/ci.yml`, `python/olsprime/models/`, `python/olsprime/train/` and new test files). Touch none of those files.
- **Motto:** we should be more ambitious, in the test, never in the claim. A "does not hold" says nothing without the test's power: this task measures what a verdict is worth.

## Goal

Two deliverables, in this order.
1. Report n°2 saves its hourly series and its scored rows in `series.npz`, with the file's SHA-256 in `report2.json`, so that no later study refits an arm. Audit phase 2's four logged items are folded in.
2. A size and power study of the registered pass rule, which reads `series.npz` by its digest and refits nothing.

Read first: `docs/PROTOCOL.md` ("Matrix on hourly bars (iteration 3)", Scoring and Decision rules; the change-log entries of 2026-10-29, especially "after audit phase 2"), `docs/ITERATIONS.md` (iteration 3's outcome: the Audit row's phase 2 and "Bolder next step"), `docs/reports/report-2.md`, the correction in Report n°2's row of `docs/RESULTS.md` (its Coverage paragraph), `python/olsprime/bench/report2.py` and `python/olsprime/eval/stats.py`.

## 1. `series.npz` in `python/olsprime/bench/report2.py`

- **Contents.**
  - Per arm, in `ARMS` order: the hourly log-loss and Brier series (A × T, float64) and the hour stamps (T, int64 ms, the bar's open hour), with the rows per hour (T).
  - Per scored row (N rows, in the order of `Scored.rows`): every arm's probabilities (A × N × 3, float64, the exact values scored), `fwd_return`, `label`, the symbol (an int code with the names array), `open_time`, `decision_time` and `label_known_at` (int64 ms), and the row's hour index.
  - The arms' names and the symbols' names as arrays.
- **Determinism.** The file's bytes must be the same on every re-run. `np.savez` stamps zip entries with the current time; write the archive so that it does not (a fixed entry date, a fixed entry order, no compression or a fixed level), and test it.
- **Digest.** `report2.json` gains `series: {"file": "series.npz", "sha256": ...}`; `report-2.md` gives the digest in its Provenance list. `COMPARED` includes `series.npz`, so `--check` compares it byte for byte.
- **Audit phase 2's logged items**, in the same change:
  1. **Coverage.** The paragraph says a row goes unscored "for the 25 bars before" a gap. The label reads only its entry and exit bars, so a gap of g bars leaves 1 + g rows of each member unlabelled before it. Correct the sentence, in the words of the correction in `RESULTS.md`. Name the 2020-06-28 example only if the numbers come from the run itself.
  2. **`ece_tolerance`'s docstring** claims the check catches "a row lost or put in another bin". It guards a row lost or counted twice; it cannot see a row put in another bin, because both paths bin the same way. Rewrite it from the PROTOCOL change-log entry "after audit phase 2".
  3. **H12's rule in the parts.** `preview_split` gives H12's parts the generic superiority rule as `rule_met`. Give them H12's own rule instead: the one-sided upper bound and the bootstrap's 95th percentile both below that part's margin. Say in the JSON and in the Markdown that the fits condition (at most 36) belongs to the whole period only.
  4. **`diff_runs`** truncates its diff at 40 lines. Add a per-file summary first: each compared file, identical or with its count of differing lines.
- **`--check` against iteration 3's run.** Report n°2 is re-run once at the new commit; the official folder of iteration 3 is `runs/2026-10-29-report2/`, which has no `series.npz`. Add a way to re-run once into a kept folder and compare it with that older folder, where a fixed, named list of differences is allowed and nothing else:
  - the provenance lines (commit, code digest) in `config.json` and `report-2.md`;
  - the new `series` entry and its line;
  - the lines of the logged items 1 and 3.

  Every other byte of `config.json`, `report2.json` and `report-2.md` must match. The allowed differences live in code, each with its reason, and the command prints which of them it met. A plain `--check DIR` keeps its meaning: byte-identical, `series.npz` included.
- **Tests** in `python/tests/test_report2.py`, on the tiny synthetic store the file already uses: the series reproduce the report's means; two runs give the same `series.npz` bytes; the digest in `report2.json` is the file's; the comparison with an older run passes on the named differences and fails on any other (one changed digit in a number must fail); H12's parts use H12's rule.

## 2. The size and power study, `python/olsprime/bench/power.py`

`uv run --no-sync python -m olsprime.bench.power --series <run>/series.npz --sha256 <digest> [--out DIR]`. It refuses a file whose SHA-256 is not the one given, or not the one in the run's `report2.json`. It refits nothing, and it describes: the pass rule does not change.

- **Comparisons.** The five of Report n°2 (`COMPARISONS` in `report2.py`). Build the list so that later reports can pass their own comparisons and series, superiority and non-inferiority alike.
- **Size.** For each comparison, take the hourly difference d = ll_a − ll_b over the T scored hours and centre it, d − d̄. For each block length L ∈ {24, 168, 720} hours, draw 2,000 replicates. A replicate rotates the hours by a random offset in [0, L), cuts them into circular blocks of L hours, and multiplies each block by an independent random sign.
  - On each replicate, apply the registered rule exactly as `report2.py` does: Diebold-Mariano (two-sided, rectangular kernel, 23 lags, HLN), and the stationary bootstrap of the mean with the Politis-White length **recomputed on that replicate**, 2,000 resamples, 95 % percentile interval. Only the bootstrap's RNG seed is fixed across replicates: use the registered seed of that comparison (`_seed(cfg, "diff", a, b)`), so each replicate draws the indices the registered rule would draw at its block length.
  - Report, per comparison and per L, with 95 % Wilson intervals: the rejection rate of DM alone (p < 0.05), of the bootstrap alone (interval excludes 0), of both on the same side (the pass rule, two-sided), and of the pass rule in the registered direction (a lower).
  - For H12, the rule is non-inferiority: its size is the rate at which it holds when the true difference sits on the margin, d − d̄ + m with m = 0.005 × climatology's log-loss; it holds when the one-sided upper bound and the bootstrap's 95th percentile are both below m.
  - Name every (comparison, L) where the pass rule's two-sided size exceeds 7.5 %.
- **Power, model-free.** For each superiority comparison and each L: replicates of d* = flip(d − d̄) + δ, with δ in the registered direction (a better), on the grid {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0} % of climatology's log-loss, 1,000 replicates per point.
  - Report power(δ) with Wilson intervals, and the smallest δ on the grid with power at least 80 % (and the interpolated value between grid points, labelled as interpolated).
  - For H12, the rate at which it holds when the true difference is 0.
  - Then read H3's `matrix-1y − knn-1y`, −0.094 % of climatology's log-loss, against that comparison's detectable δ, per L.
- **Cost.** 2,000 bootstrap resamples over about 50,000 hours, for tens of thousands of replicates, is too slow as a loop. The resampled mean is linear in the series: for a fixed block length and seed, the index matrix is fixed, so the 2,000 resampled means of many replicates are one matrix product of a count matrix (2,000 × T) with the replicates (T × n). Group the replicates by their recomputed block length and use that. Test that it equals `x[idx].mean(axis=1)` within 1e-12 relative on random data, and say that the study's intervals can differ from the direct path in the last bits only. Use processes for Politis-White if it is the bottleneck. Measure and report the runtime.
- **Determinism.** One registered seed for the flips (`20261030`), fixed in the config and saved. Two runs give byte-identical `power.json` and `power.md`.
- **Outputs.** `power.json`, `power.md`, `config.json` (with the series digest and the run of `report2.json` it came from, and provenance as in `report2.py`) and `run.json` (wall clock only) to `runs/<UTC date>-power/` or `--out`. JSON is written with sorted keys.
- **Tests** in a new `python/tests/test_power.py`, on synthetic series: iid normal differences give a two-sided size near 5 % (state the tolerance); an AR(1) difference with ρ = 0.9 and L = 24 gives the inflated size the theory predicts for that dependence, or say what it gives; power rises with δ; the digest check refuses a changed file; the grouped bootstrap equals the direct one.

## Constraints

- **No real data.** Run nothing on the real store: no Report n°2, no study on its series. The orchestrator re-runs Report n°2 after the audit, and then runs the study. Everything you run uses the synthetic stores of the tests.
- **Files.** `python/olsprime/bench/report2.py`, `python/tests/test_report2.py`, and the new `python/olsprime/bench/power.py` and `python/tests/test_power.py`. Ask the orchestrator before touching any other file, `eval/stats.py` included.
- **Python.** `uv run --no-sync` from `python/`. Do not run `uv sync` or `uv add`: another agent changes the dependencies in parallel.
- **Scripts.** Write scripts with a file tool. Never pipe or heredoc code into `python`.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\series-power\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- `series.npz` is written, deterministic, digested and checked; the four logged items are folded in; the comparison with an older run works as specified.
- `bench/power.py` runs on synthetic series, deterministic, with the tests above.
- `uv run --no-sync pytest` passes, except failures you show to be caused by the other agents' files in progress.

## Report (short)

- The format of `series.npz`, key by key.
- The allowed differences of the comparison with iteration 3's run, as coded.
- The study's design as built, its runtime on a synthetic series of 50,000 hours, and the projected runtime on Report n°2's series.
- Every point where this brief needed an interpretation, and the one you took.
- One bolder next step.
