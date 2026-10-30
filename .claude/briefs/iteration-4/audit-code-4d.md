# Brief — audit pass 4d: `olsprime.explain`, before its first call on the real data

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with:**
  - the encoder agent, building on synthetic data;
  - the series-power agent, which is fixing how `bench/power.py::load_series` reads `series.npz`.
  Nothing has run on the real data in this iteration.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

`explain` is the first instrument of this project that is pointed at the test months on purpose. Its first calls are the five whys of `docs/ITERATIONS.md`, iteration 4. Before any of them runs, find every way it could:
- reach data it must not;
- touch data without logging the contact;
- report a decomposition that is not what the forecast did;
- claim more exactness than it has.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)": the section "`olsprime.explain`", the frozen clause and the change log since 2026-10-29, including the entry on the channel into the encoder's design;
- `docs/PROTOCOL.md`, "Matrix on hourly bars (iteration 3)": the prequential record, Scoring, the arms;
- `docs/ITERATIONS.md`, iteration 4: the five whys;
- the brief `.claude/briefs/iteration-4/explain.md`.

Then the code, uncommitted in the working tree of branch `iteration-4`:
- `python/olsprime/explain/`: `cli.py`, `common.py`, `gitstate.py`, `views.py`, `why.py`;
- `python/tests/test_explain.py`;
- the `.gitignore` diff (`git diff -- .gitignore`);
- what it reuses:
  - `olsprime/bench/report2.py`, which builds the panel and Matrix;
  - `olsprime/models/matrix.py`;
  - `olsprime/bench/power.py::load_series`, being changed now; judge its contract, not its current lines;
  - the Rust getters `beta` and `mean_y` in `crates/olsprime-py/src/py/ols.rs`.

## The engineer's interpretations, to judge

1. **Before the first fit.** Rows have no fit in force. `why` shows the candidate named by the registered defaults (365 days, ρ 1e-2, τ √15, k 200, α on 365 days), with the σ of the row's month, and says so.
2. **Stamp check 1** covers every entry of every window read.
   - The stamp is recomputed as the bar's open + 25 h and must equal the panel's `label_known_at`.
   - The top k entries are checked against the raw bars through `load_klines`.
   - For `matrix-stack`, only the windows with α > 0 are checked.
3. **"The parts add up exactly."** The parts are the top k groups by |c|, the rest summed in group order, and a separate rounding term.
4. **`series` with one arm** uses d_t = b_clim − b_arm and parts d_t / Σd.
5. **The flag.** `--test-months` allows any period other than the default. 2025 and later are refused even with it; so is anything before 2017-10. 2017-10 itself needs the flag.
6. **The contact line** is written for every call that passes its argument checks, before the run folder is opened, and one per `--at`. A call refused for its period logs nothing.
7. **Dirty flag.** It is read from `.git/index` (versions 2 and 3), so a change that is staged but not committed is not seen.
8. **`sample` on `matrix-stack`.** It uses the α-weighted negative mass and gives its parts per window.
9. **Output.** The JSON lines go after `=== json lines ===` on stdout. A failed loud check prints no result.
10. **`.gitignore`.** Three lines, so that only `CONTACTS.log` is tracked.

## What to attack, at least

1. **Reach.**
   - Can any subcommand, `--at` list, period syntax, time zone, millisecond edge or month boundary read a row decided at or after 2025-01-01, or a vault path?
   - Can the default period's gate be passed without the flag, for example with an `as_of` at the edge or with a `--at` after a first allowed one?
   - Does the panel rebuild cut at 2025-01-01 exactly as report2 does, through `load_klines` and the seal?
2. **The log of contacts.**
   - Can a call touch the data, the run folder or the panel without its line?
     - An exception after the checks.
     - The `--at` loop.
     - A call refused after it opened something.
   - Is the line complete (utc, subcommand, period, arm, commit, dirty, digest), and can it be forged or skipped by an argument?
   - Is the file append-only and tracked, with nothing else under `runs/explain/` tracked?
3. **The two loud checks.**
   - Are they independent of what they verify? Check 1 must recompute stamps from panel rows, never trust Matrix's stamps. Check 2 must compare with `series.npz`, never with a value computed by the same path.
   - Can either pass vacuously: zero entries, an empty window, a NaN compared with a NaN, a row not in `series.npz`?
   - Does a failed check always end in exit 3 with no result printed?
4. **Truth of the decomposition.**
   - Is the rounding term of interpretation 3 a residual by definition, which would make "adds up exactly" trivially true? If so, how should the output say it?
   - Is the per-feature view bit-exact in the core's order, and is the claim stated as checked at that call?
   - Are the grouping (by `label_known_at`, never by decision time), the top-k choice and the softmax/kNN "within rounding" claims all honest?
5. **The arms.**
   - `matrix-adaptive`: is ρ rebuilt only from losses known at the decision time, with the same pool rule as `adaptive_choices`?
   - `matrix-cusum`: is the fit in force the one used at the decision, never a later one?
   - `matrix-stack`: are the α and windows the ones used at the decision?
   - Does any arm read a fit, a σ or a climatology from a later month than the row's?
6. **`series` and `sample`.**
   - Are the shares, the Lorenz curve and the per-month split correct for both signs of Σd?
   - Does `sample` draw rows only from its period, as a pure function of its seed?
   - Are the deciles of negative mass computed as stated?
7. **The frozen design.** Does anything in `explain` write to or read from the encoder's code, config or runs, which would open the channel the protocol closes?
8. **Tests.** Do they test what they claim? Plant at least one defect of your own in a scratch copy, for example a stamp shifted by one hour, a forecast compared with itself, or a contact line skipped on one path, and say whether the tests catch it.

## Constraints

- **Read-only.** `tests/test_explain.py` currently errors at setup in the repository, because of `load_series` (see above). You may run it in a scratch copy of `python/` whose `tests/allowlist.py` also holds this entry:
  `"olsprime/bench/power.py": {"binary-load": ..., "raw-reader": ...}`.
  That is how the engineer ran it. Say so in your report.
- **Probes** run on synthetic data only, never on the real store, `data/` or the vault tree.
  - Write each one into a file in your scratch folder with a quoted heredoc: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4d\`.
  - Run it with `uv run --no-sync --project C:\Users\abdou\ols-prime\python python -I <file>`.
  - Never pipe code into `python`.
- **Never:**
  - call `python -m olsprime.explain` on the real data or write the real `runs/explain/CONTACTS.log`;
  - set `OLSPRIME_OPEN_VAULT` or write `runs/vault/OPEN`;
  - run a git command that changes state;
  - run `uv sync`.
- Put a timeout on any command that could hang.

## Report

- **Findings**, ranked Blocker, Major, Minor or Nit, each with:
  - file:line;
  - a failure scenario, with a probe where one helps;
  - the fix.
- **The ten interpretations:** for each one, accept or reject, with a reason.
- **Clean:** what you checked and found sound.
- **Not checked.**
- **Verdict:** GO, GO with fixes, or NO-GO for the five whys' calls on the real data.
- One bolder check for `explain`.
