# Brief — audit pass 4a: iteration 4's pre-registration, before any run

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with** two `ml-engineer`s, who build code on synthetic data only: Report n°2's series and the power study (`series-power.md`), and the encoder's core with its GPU sizing (`encoder-core.md`). A `rust-engineer` has finished `beta` and `mean_y` on `CentredReadout` (`beta-getter.md`). Nothing has run on the real data in this iteration.
- **Motto:** we should be more ambitious, in how hard we attack the design, never in what we claim we found.

## Goal

The design of iteration 4 is cheapest to fix now: no result exists, and a change is still allowed, dated in the change log. Attack it before it is built and before anything runs. Find:
- what would let a result leak, overfit or be misread;
- what is ambiguous enough that two honest implementations would give different verdicts;
- what cannot be built or computed as written.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: "The encoder, and what a verdict is worth (iteration 4)", its change-log entry of 2026-10-29 "before any run of iteration 4", and what it builds on: "Splits", "Leakage tests" and "Matrix on hourly bars (iteration 3)";
- `docs/HYPOTHESES.md`: H1, H4–H6, H13, H14, the priors and the change log;
- `docs/ITERATIONS.md`: iteration 4's brief, the five whys and the plan, and iteration 3's outcome and lessons;
- `docs/DESIGN.md` §1, §3.2 and §4;
- `docs/reports/report-2.md`, and Report n°2's rows in `docs/RESULTS.md`.

Then, to check that the design can be built as written:
- the code it builds on: `python/olsprime/features/panel.py`, `python/olsprime/models/matrix.py`, `python/olsprime/bench/report2.py`, `panel_invariance.py` and `adversary.py`, `python/olsprime/eval/stats.py`, and `crates/olsprime-py/olsprime_rs.pyi`;
- the wave 1 briefs in `.claude/briefs/iteration-4/`. The code they produce is still in progress: review the briefs, not that code.

## What to attack, at least

1. **The encoder's causality.**
   - The window's 64 bars against the row's decision time.
   - The standardization and the climatology.
   - The memory: its stamps, its window `d − 30 days < label_known_at ≤ d`, the prefix sums inside an episode, and the cache keyed without gradient.
   - The fit's training rows, its slice, its purge and its early stopping.
   - The memory at test time, keyed by the frozen encoder, and the seed ensemble.

   Can any output at d read a label not known at d, or a bar not final at d?
2. **The GPU H1.**
   - The pre-check, and the fallback rule: the floor, 10 × the floor, and 10⁻⁹. Can a real leak hide under the floor?
   - The predicted patterns of `cheat-stamp` and `cheat-value`. Is `XXXXXXXX` right at every cut, given where the cuts fall (`choose_cuts`) and where each fit's training rows end?
   - The cuts fall in test months, after the start of the fit they test, while a fit's training ends 30 days before it. Can this H1 see a training-time leak at all, such as a fit that trains on rows of its own test year? If not, should a third planted control test that?
   - "A Brier skill score of at least 0.1, with a 95 % interval above every honest arm's": is it well defined, on which rows, and is it a sound check of each cheater?
3. **H4, H5 and H13.**
   - H4: the non-inferiority bounds; equal parameters, N against N + 8; the seed ensemble against the per-seed losses; whether three seeds and yearly fits make a fair comparison.
   - H5: the pass rule, the pairing of hours with `matrix-select`'s series, and the stated limit.
   - H13: are the timing conditions well defined and fair to both arms? Look at warm-up, synchronization, the batches, which rows are timed, the memory's share of the time, and the math backend.
4. **H6 and LFM3's ablations.** Are their rules well defined?
5. **The size and power study.**
   - Does it measure what it claims? Look at circular block sign flips under dependence and asymmetry, the recomputed Politis-White length, and the fixed bootstrap seed.
   - Is power under a constant shift δ a fair reading of "too weak to see"?
   - Is H12's non-inferiority size well defined?
6. **`explain`.**
   - Its contacts with the test months, and the rule that what it shows there is exploratory.
   - Can any of the five whys feed back into a registered test of this iteration?
   - Are `why`'s two loud checks sufficient?
   - Is the freeze, committed before any call on 2019–2024, enough?
7. **The future-shuffle twins.** Are they well defined, and do they measure what the three classes claim?
8. **H14**, registered now and run in iteration 5:
   - look-ahead in the rule: the readout at the decision, execution at the next open, the exits;
   - the cost model and the stated assumptions;
   - whether the claim can be met vacuously.
9. **Bookkeeping.**
   - The variant count; the priors; the dates.
   - Consistency between `PROTOCOL.md`, `HYPOTHESES.md` and `ITERATIONS.md`.
   - The rule for smoke runs on the real panel.
10. **Feasibility.** Can each rule be built as written with the existing code?
    - `Matrix.window_range`'s bounds.
    - The memory's penalty mapped to `centred_query`'s `scale`.
    - `PanelArrays`, the stamps, Report n°2's scored rows.

## Constraints

- **Read-only.** Never edit the repository. You may run the existing tests (`uv run --no-sync pytest`, `cargo test`), but other agents are editing files in parallel, so a failure may be theirs: say so rather than guess.
- **Probes**, if you need them, run on synthetic data only, never on the real store, `data/`, or the vault tree.
  - You have no file-writing tool. Write each probe into a file in your scratch folder with a quoted heredoc (`cat > file <<'EOF'`): `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4a\`.
  - Run it with `uv run --no-sync --project C:\Users\abdou\ols-prime\python python -I <file>`.
  - This is the one allowed use of a heredoc. Never pipe code into `python`.
- **Never** set `OLSPRIME_OPEN_VAULT`, never write `runs/vault/OPEN`, run no git command that changes state, and do not run `uv sync`.
- Put a timeout on any command that could hang.

## Report

- **Findings**, ranked Blocker, Major, Minor or Nit. For each:
  - the file and line;
  - what is wrong;
  - a concrete failure scenario, with a probe where one helps;
  - the fix, worded so that it can go into the protocol.
- **Clean:** what you checked and found sound.
- **Not checked:** what you could not check.
- **Verdict:** GO, GO with fixes, or NO-GO for the official runs.
- One bolder check to add to the bench.
