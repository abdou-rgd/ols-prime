# Resume iteration 4 after the weekly reset (2026-11-04, 02:00 UTC)

- **Written:** 2026-10-30 at about 11:30 UTC, by the orchestrator.
- **Why the stop:** Abdou's weekly limit stood at 80 %, and they need margin for other projects. From this point the iteration runs unattended, and the paperwork waits for a fresh session after the reset.
- **Start that session fresh**, not by continuing the old one. Its context had grown to 234k tokens.

## What runs unattended

The script `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\orchestrator\chain-official.sh` logs to `runs/2026-10-30-chain.log`. It runs in this order:
1. **The panel H1** (`runs/2026-10-30-encoder-h1/`, log `runs/2026-10-30-encoder-h1.log`). It runs from `~/ols-prime-official` at `5f17a76`.
   - The pre-check passed: every arm bit for bit at 2020, 2021, 2022 and 2024. The scope tests gave 0 differing values.
   - It will hit its 16 h timeout at 23:38 UTC, and the chain resumes it with the same `--out`.
2. **Report n°3's 54 fits** (`runs/2026-10-30-report3/`, log `runs/2026-10-30-report3-fits.log`), from `~/ols-prime-r3` at `5f17a76`.
3. **When both have ended:** Report n°3's `timing`, `matrix-timing`, `score` and `check`, through `r3-official.sh 5f17a76 691e7ea7… <stage>`. Their logs are `runs/2026-10-30-report3-<stage>.log`.
4. **The shutdown:** `shutdown /s /t 300` if every step passed (Abdou's choice). Otherwise the PC stays on.

**Read the chain log with care.** Its line "the H1 did not pass (exit=3)", at 2026-10-31T00:31:22Z, is the chain's wording and is misleading. Exit 3 means that H1 holds and a skill check fails (`docs/PROTOCOL.md`). The H1 report says: pre-check passes; H1 passes; skill check FAIL on `cheat-stamp` alone (Brier skill score 0.0064 [0.0029, 0.0100]); `pos-window` passes. The chain treats that exit as not passed, so it does not shut the PC down by itself. These are the raw report's lines, not audited.

**Report n°3's check.** The chain's `check` stopped at exit 2 at 2026-10-31T00:43:06Z because `r3-official.sh` did not pass it `--series` and `--sha256`. That was a script error, not a result. Re-run from the same clone with those arguments, it passed at 00:46:52Z: every part verifies, and 4 files are identical (`runs/2026-10-30-report3-check.log`). The raw verdicts, not audited: H4 holds, H5 does not hold, H13 holds.

## Deferred to iteration 5 (logged in `docs/PROTOCOL.md`'s change log, entry "after audit pass 4h")

- **LFM3's ablations and H6:** not fitted, so H6 is reported as not run.
  - Their code is uncommitted in the working tree:
    - `python/olsprime/bench/ablations.py`;
    - `models/encoder.py`, `train/fit.py` and `train/__init__.py`;
    - `bench/report3.py` (m6);
    - `tests/test_ablations.py`, `test_encoder_ablations.py`, `test_train.py` and `test_report3.py`.
  - Audit pass 4h gave GO with fixes. Its brief is `audit-code-4h.md`, and the fixes' brief is `ablations-fixes-4h.md`.
  - The engineer was stopped for budget just after "every plant fails its named test". Their synthetic end-to-end run, their report and the audit's second look are not done.
  - The seed-split control beside H6 is registered in that same entry.
- **`explain head`:** built and audited (commit `da0433e`), but never called. The scripts `wsl-abl-setup.sh`, `head-official.sh` and `abl-official.sh` are ready in the scratchpad folder above.
- **The power study's second run**, for byte identity: not re-run, as logged.

## What is left for iteration 4, in the fresh session

1. **Read the results.**
   - Read `runs/2026-10-30-chain.log` and each stage's exit.
   - Read `encoder-h1.json` and `.md`, then `report3.json` and `.md`, and check that Report n°3's `--check` passed.
2. **Commit the documents.**
   - Commit `docs/PROTOCOL.md`, `docs/ITERATIONS.md` and the briefs `audit-code-4h.md`, `ablations-fixes-4h.md` and this file. Never `git add -A`: `.claude/worktrees/` is untracked.
   - Do not merge the ablations' code into `main` with iteration 4, since its fixes are unverified. Move it to a branch for iteration 5, or leave it out of the commit.
3. **The results audit** (`leakage-auditor`, brief `audit.md`). Keep it to what this iteration reports:
   - the panel H1;
   - Report n°3 (H4, H5, H13 and the cost table);
   - the size and power study;
   - the twins;
   - the two platforms;
   - the five whys.
4. **The records.**
   - Add the rows to `docs/RESULTS.md`. The drafts are `results-draft.md` and `outcome-draft.md` in this folder; `notes.md` is the running log.
   - Write the outcome in `docs/ITERATIONS.md`. Name every hypothesis that was not run as not run (H6), and end with one bolder step.
5. **The learning page.** The `teacher` writes it from `learning-page.md`. Publish it as an Artifact and link it from the outcome.
6. **Merge.** `git merge --no-ff` into `main`, then push.
