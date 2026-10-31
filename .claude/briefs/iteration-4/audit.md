# Brief — Audit of iteration 4's results (phase 2)

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **When:** after the official runs, before any row of `docs/RESULTS.md` or any claim of `docs/ITERATIONS.md` is final. The rows and the outcome are drafted first, so that you attack the words as written.
- **Phase 1 is done.** The code of iteration 4 went through passes 4a to 4g (`audit-prereg.md`, `audit-code-4c.md` to `audit-code-4g.md`), each with its second look. This pass reviews the results and the claims, plus the few pieces of code that ran without a pass of their own (listed below).
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

No result of iteration 4 is reported until you have tried to break it. For each number: does it match the run's files, does its verdict follow the registered rule word for word, and is the claim no stronger than the evidence? Every finding gets a severity (Blocker, Major, Minor), a concrete failure case and, where you can, a probe that shows it.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: "The encoder, and what a verdict is worth (iteration 4)" and every change-log entry dated 2026-10-29 or 2026-10-30;
- `docs/HYPOTHESES.md`;
- `docs/ITERATIONS.md`, iteration 4's brief and its draft outcome;
- `.claude/briefs/iteration-4/runs.md`, which fixed every command and reading before the runs;
- the draft rows at the end of `docs/RESULTS.md`.

## The runs

All under `C:\Users\abdou\ols-prime\runs\`. Each `run.json` names its commit and code digest. Official runs ran from clean clones (Windows: `C:\Users\abdou\ols-prime-official`; WSL2: `~/ols-prime-official` and `~/ols-prime-r3`).

| Run | What |
|---|---|
| `2026-10-30-report2`, `-check.log` | Report n°2 re-run on Windows, compared with iteration 3's run, and its `--check` |
| `2026-10-30-report2-wsl`, `-wsl.log`, `-wsl-check.log` | Report n°2 re-run in WSL2, compared with the Windows re-run, and its `--check` |
| `2026-10-30-platforms.json` | the orchestrator's comparison of the two re-runs, field by field |
| `2026-10-30-power`, `2026-10-30-power-2` | the size and power study, twice |
| `2026-10-30-twins`, `2026-10-30-twins-2` | the future-shuffle twins, twice |
| `2026-10-30-encoder-h1` (and `.log`) | the panel H1 for the encoder arms on the GPU |
| `2026-10-30-report3` (and its stage logs) | Report n°3: fits, timing, Matrix timing, score, `--check` |
| `2026-10-30-explain`, `explain/CONTACTS.log` | the five whys and every `explain` call |

## What to check

1. **Every draft row of `RESULTS.md`.** Recompute each number you can from the run's JSON. Check the variant count: 21 after this iteration, the planted leaks and `pos-window` being controls.
2. **The panel H1 for the encoder** (PROTOCOL, "H1 for the encoder arms", and the change-log entry "The panel H1, as built").
   - Did the pre-check pass on everything it compares, and did it gate the H1 stage?
   - Are the predicted patterns the computed ones, per cut, and does every control sit on its own?
   - The purge cuts: what is compared at F − 24 h and at F − 30 days, and is `cheat-slice` invariant at F − 24 h as predicted?
   - The three verdicts reported apart: H1, the skill check control by control, `pos-window`'s Brier skill score against 0.5. None may stand in for another.
   - The honest runs' forecasts on the cut years are a first look at the encoder's skill. Is that contact listed in the change log, and does Report n°3 say so?
3. **Report n°3** (PROTOCOL, "H4, H5 and H13", and the entry "Report n°3, as built").
   - H4's margin: 0.005 × the mean hourly log-loss of `climatology-panel`, taken from the WSL2 run of Report n°2 by digest. Both bounds, and the testability gate: a Brier skill interval above 0 for `enc-ols` or `enc-softmax`. If neither has it, H4 and H13 must read "not testable", never "holds".
   - H5 against `matrix-select`'s series from the same WSL2 run: Diebold-Mariano p < 0.05 and the bootstrap interval below 0, both.
   - H13: the timing conditions (quiet machine, ABAB repeats, the math backend for softmax, 12,288 rows with seed 20261030). Was anything else running on the GPU or the CPU while it timed? The logs and the orchestrator's notes say.
   - The cost table: one machine and one build per column. The fits may have trained while the panel H1 shared the GPU: if so, their wall times are upper bounds, and the table must say so.
   - The parameter counts (N + 8 for `enc-ols`), the windows that failed and took climatology, the per-seed log-losses and their spread.
   - `--check`: which files are byte-identical, and what is excluded.
4. **Report n°2 on two platforms** (the change-log entry "Report n°3 cannot be built as written on the Windows runs of Report n°2").
   - Is the comparison right that labels, times and the two named features match, and that only log-based columns differ in their last bits?
   - Do the verdicts of H2, H3, H10, H11 and H12 agree between the two runs? Is every difference described, none hidden in a rounding?
   - Check `compare_platforms.py` (in the orchestrator's scratchpad, `orchestrator\compare_platforms.py`): does it compare every array of both series and every leaf of both `report2.json`?
5. **The size and power study** (PROTOCOL, "Size and power of the pass rule").
   - Are `power.json` and `power.md` byte-identical between the two runs? Did the pool restart, and if so does the log say why?
   - Is every L with a two-sided size above 7.5 % named?
   - The smallest detectable δ, constant and concentrated, and H3's −0.094 % read against them: is "does not hold" worded as "too weak to see" only where the power says so?
6. **The twins.** 205 misses of 10,000: constant 193, unchanged 5, blind spot 7; caught leaks moved by a twin 98.3 % [98.0, 98.5]; honest twins 0. Check the counts against `twins.json`, and that the claim is only about this generator's misses.
7. **The five whys** (ITERATIONS, "The five whys"; runs.md §4).
   - Each call in `CONTACTS.log`, in the order run, with the line counts the change log cites.
   - The rows of question 1 were picked by the rule fixed before any output was read (BTCUSDT's row at the two largest positive and two largest negative d). Check `1-picked-hours.json` against the `series` output.
   - Questions 1, 2 and 5 get numbers. Questions 3 and 4 are labelled "exploratory, test months", and no hypothesis is tested on these months.
   - The first call on 2019–2024 hours came after the frozen design was committed. Check the commit's time against the first contact line.
8. **The contacts.** Is every contact with the real data of this iteration in the change log, with its months, and are previewed months shown apart?
9. **`explain head`**, if it ran: is each call refused or answered as audit pass 4g's fixes say, and is its contact logged?
10. **The draft outcome in `ITERATIONS.md`.** Is each claim stated no stronger than the evidence? Is every item not run reported as not run?

## Constraints

- **Read-only.** You may run the tests, and probes in your scratch folder (`C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit\`), on the run files and on synthetic data.
- **No new contact.** Do not run `explain`, any bench or any fit on the real data. Every number you need is in the run folders; if one is missing, say so.
- Write probes with a file tool, never through shell heredocs or pipes into `python`. Run them with `uv run --no-sync --project python python -I <file>`.
- No git command that changes state. Never set `OLSPRIME_OPEN_VAULT` and never read the vault.

## Addendum, 2026-10-31: scope and budget of this pass

**One pass, and be frugal.** Abdou's weekly limit is nearly spent: 16 % left, and 29 % for Fable. Do not run the test suites: the orchestrator runs them. Prefer reading the run's JSON to probes, and write at most a handful of short probes. Spend your effort where a wrong claim would do the most harm, in the order below. Keep the report under about 1,000 words, and say what you did not check.

**What changed since this brief was written.**
- `explain head` never ran, and LFM3's ablations and H6 were not run: item 9 falls, and the code is off this branch (`iteration-5-ablations`, `5be70a3`). The outcome must call them not run.
- The power study's second run stopped at start-up, so item 5's byte identity does not exist. The row must say so.
- The rows are at the end of `docs/RESULTS.md`, the outcome at the end of `docs/ITERATIONS.md`, and the last change-log entry at the end of `docs/PROTOCOL.md`, all uncommitted. `docs/reports/report-3.md` is a copy of the run's file. Placeholders ending in `_TODO` are the orchestrator's, for your verdict and the verification.
- Probes go in the scratch folder named below, in a new subfolder `audit-phase2`.

**The first target: H13.** The orchestrator found that the registered ratio, 0.4031, is an artefact, and chose to keep the number the rule computes while not claiming it. Attack that decision in both directions.
- The evidence: `runs/2026-10-31-h13-repeat/` (repeat1-timing.json, repeat2-timing.json, the two `smi.csv`), the logs `runs/2026-10-31-h13-repeat.log` and `-2.log`, the scripts `h13-gap\repeat-real.sh`, `repeat-empty.sh` and `wrap_empty.py` in the scratch folder, and `.claude/briefs/iteration-4/h13-gap-and-rows.md`. The report of the synthetic replays is summarized in `docs/RESULTS.md`; its logs are in `h13-gap\`.
- Does repeat 2 show what the records say? Nobody logged the reserved memory during the official timing: is it fair to call the official number the same artefact, given repeat 1 and the code? Is one repeat, by day, with Windows applications on the GPU, worded no stronger than it deserves?
- Is "the rule computes holds, not claimed" the right reading of `docs/PROTOCOL.md`, or should the verdict read otherwise? A registered verdict set aside after seeing a result is the kind of move this project forbids, even in the cautious direction. Say whether this one is justified and how it should be worded.
- Does the artefact reach anything else: the cost table's other columns, H4, H5, the Matrix timing, the 2026-10-30 claim that the readout pulls the ratio toward 1?

**Then, in order:** the three verdicts of the panel H1; H4 and H5 against their rules; the claims of the outcome, especially the reading of `cheat-stamp`'s failed skill check and H5's interval; the power, twins and platform rows; the five whys and the contacts. Sample the numbers instead of recomputing all of them where the runner's checks already cover them.

## Report

- The findings, by severity, each with its failure case and probe.
- What you checked and found sound.
- What you could not check.
- One bolder next step.
