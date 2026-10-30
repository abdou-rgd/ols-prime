# Brief — audit pass 4c: wave 1's evaluation code, before Report n°2's re-run and the power study

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with** engineers building wave 2 (the encoder, the twins, `explain`) on synthetic data. Nothing has run on the real data in this iteration.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

Two official runs wait on this pass:
1. Report n°2 is re-run once, to write `series.npz` with its digest. Its numbers must not change: it is compared with iteration 3's run `runs/2026-10-29-report2/`, and only a fixed, named list of lines may differ.
2. The size and power study of the pass rule then reads that `series.npz`.

Break the code those runs use before they run.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: "Matrix on hourly bars (iteration 3)" (Scoring, Decision rules), the iteration 4 section "Size and power of the pass rule", and the change log since 2026-10-29;
- the briefs `.claude/briefs/iteration-4/series-power.md` and `beta-getter.md`.

Then the code, which sits uncommitted in the working tree of branch `iteration-4`:
- **Rust:** `git diff -- crates/`, with `crates/ols-core/tests/readout_beta.rs` and `python/tests/test_readout_beta.py`.
- **Report n°2:** `git diff -- python/olsprime/bench/report2.py python/tests/test_report2.py`.
- **The power study:** `python/olsprime/bench/power.py` and `python/tests/test_power.py`, both new.
- **The twins:** `python/olsprime/bench/twins.py` and `python/tests/test_twins.py`, both new.

## What to attack, at least

1. **Report n°2's re-run must not move a number.**
   - The comparison is `python -m olsprime.bench.report2 --compare-with runs/2026-10-29-report2 --out <dir>`. It allows exactly the five differences of `ITERATION3_ALLOWANCES`: the commit line, the code digest line, the new `series` entry, the Coverage sentence and H12's rule in the parts. After it, `--check <dir>` must reproduce all five files byte for byte, `series.npz` included.
   - The shared environment was synced to the new lock before this pass. numpy 2.5.4, scipy 1.18.1, polars 2.1.0 and Python 3.12.10 are unchanged, and CPU torch was added. Is anything the report records sensitive to the added packages?
   - Does any change touch a computation behind a reported number, or the order of a reduction?
   - Is every allowed difference in the comparison with the older run named, justified and narrow? Could a changed digit slip through it?
   - Does a plain `--check DIR` keep its meaning, byte for byte, `series.npz` included?
2. **`series.npz`.**
   - Are the series exactly the values scored, in `ARMS` order, with the rows in `Scored.rows` order, and the stamps right (ms, the bar's open)?
   - Are the file's bytes deterministic: entry dates, order, compression?
   - Is the digest in `report2.json` the file's?
   - Can a later study pair hours or rows wrongly from it?
3. **Audit phase 2's four logged items**, as folded in:
   - the Coverage sentence;
   - `ece_tolerance`'s docstring;
   - H12's own rule in the parts, with the fits condition for the whole period only;
   - `diff_runs`' per-file summary.
4. **The power study.**
   - Does it apply the registered pass rule exactly as `report2.py` does: DM rectangular with 23 lags and HLN, Politis-White recomputed per replicate, the comparison's registered bootstrap seed?
   - Is the grouped count-matrix bootstrap the direct one within the stated tolerance, for every group?
   - Are the circular block flips, the four block lengths, H12's non-inferiority size, the δ grid, the concentrated grid at L = 168 and the Wilson intervals as registered?
   - Does it refuse a wrong digest? Is it deterministic?
   - Does anything in it fit, select or tune on the series?
   - **One known wrinkle, to judge.** The engineer reports that a run with `--workers 1` gives the same `power.md` and `config.json` as a pool, but `power.json` differs in 7 of 5,452 values. They are the variance-ratio diagnostic and the DM size it implies, at most 1.3 × 10⁻¹⁵ relative, attributed to the main process's multi-threaded BLAS. No rate differs.
     - Is the attribution right?
     - Should the reduction be made independent of BLAS threads so that every worker count gives the same bytes, or is "official runs and re-checks both use the pool" an acceptable rule?
   - **Its reader and the guards.** `load_series` opens the archive explicitly with `zipfile` and reads each `.npy` member with `numpy.load(allow_pickle=False)`. That gives `power.py` three allow-list entries in `tests/allowlist.py`: `import:zipfile`, `raw-reader` and `binary-load`.
     - Is each allowance exact and justified?
     - Could the allowance let `power.py` read anything but a digest-checked `series.npz` under `runs/`?
     - Does the tripwire's path rule still refuse a path under `data/` or the vault from it?
5. **The getters `beta` and `mean_y`.**
   - Do they come from the same solve as the forecast?
   - Do they change no existing bit?
   - Is the exactness claim (bit-exact in the core's order, within a stated bound otherwise) true?
6. **The future-shuffle twins**: `python/olsprime/bench/twins.py` and `python/tests/test_twins.py`, both new. The specification is the protocol's "H1's blind spots". Judge the engineer's interpretations:
   - "constant" is judged at the compared rows of H1's 16 cuts, while the adversary's own flag covers both of its schedules; both are reported;
   - the classes take precedence in the order constant, then unchanged, then blind spot;
   - "caught at no cut" means none of H1's 16 cuts;
   - `symbol` is kept with the time stamps;
   - the workers are threads.

   Also check:
   - Do the twins draw only from rows at or before the cut, and keep every time stamp?
   - Is "moved" exactly H1's catch rule?
   - Does the cross-check with the official adversary run refuse every mismatch?
   - Is the official command's input, `runs/2026-10-29-h1-adversary/adversary.json` by its SHA-256, the right one?
7. **Tests.** Do they test what they claim? Plant at least one defect of your own in a scratch copy, for example a changed digit in a report line or a series row shifted by one, and say whether the tests and the comparison catch it.

## Constraints

- **Read-only.** You may run `uv run --no-sync pytest` on these test files, `cargo test -p ols-core` and the study on synthetic series. Other agents edit other files in parallel: a failure there is theirs, so say so rather than guess.
- **Probes** run on synthetic data only, never on the real store, `data/` or the vault tree.
  - Write each one into a file in your scratch folder with a quoted heredoc: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4c\`.
  - Run it with `uv run --no-sync --project C:\Users\abdou\ols-prime\python python -I <file>`.
  - Never pipe code into `python`.
- **Never** set `OLSPRIME_OPEN_VAULT`, never write `runs/vault/OPEN`, run no git command that changes state, and do not run `uv sync`.
- Put a timeout on any command that could hang.

## Report

- **Findings**, ranked Blocker, Major, Minor or Nit, each with file:line, a failure scenario (with a probe where one helps) and the fix.
- **Clean:** what you checked and found sound.
- **Not checked.**
- **Verdict:** GO, GO with fixes, or NO-GO for Report n°2's re-run, the power study and the twins' run.
- One bolder check for the bench.
