# Brief — a control caught by rounding alone (iteration 2, after the first official H1 run)

**Found by.** CI on commit 9c414ca: `pytest (ubuntu-latest)` failed two tests that pass on Windows.

1. `test_the_transductive_scaler_is_flagged_unless_its_month_is_over`: on Linux the transductive scaler was not flagged at the fourth cut of the synthetic bars.
2. `test_spelled_paths_map_onto_the_file_system_paths`: the expected UNC spelling was written with Windows separators.

**Diagnosis of 1 (orchestrator).** `TransductiveScaler` standardized the features on the training rows plus the whole test month, then fitted the baseline `Ridge`, which standardizes again on its training rows. That second standardization undoes the first exactly, so in exact arithmetic the control's outputs do not depend on the test month at all. Measured on the synthetic bars: its outputs moved by at most 1.1e-16, and at one cut by 8.7e-19 on Windows and by 0 on Linux. The bit-for-bit rule counted rounding as a catch. Every other control's weakest catch moves an output by at least 2.7e-3 or flips a target (script: the orchestrator's scratch `diag/magnitudes.py`).

**Fix (orchestrator, small).**
- `olsprime/bench/invariance.py`:
  - `CAUGHT_ABOVE = 1e-9`;
  - `compare()` also returns `caught` (rows where an output moves by more than `CAUGHT_ABOVE`, turns missing or stops being missing) and `largest`;
  - `run_model()` records `caught`, `caught_by_direction` and `largest_change`;
  - `verdict()` flags an honest model by a single bit, as before, and a control only where it is caught; it reports `smallest_catch` and `rounding_only_cuts`;
  - the report states the rule and gains two columns.
- `olsprime/bench/leaks.py`: the transductive scaler's ridge centres the features but keeps their scale (`_RidgeAtGivenScale`), so the test month's spread reaches the penalty. Its catches now move an output by 8e-3 to 2.5e-2.
- Tests:
  - `compare()` on one bit, on a move above and below the threshold, and on missing values;
  - the transductive scaler's catches are far above rounding;
  - the old absorbed scaler is off its pattern;
  - an honest model is still flagged by one bit.
- `tests/test_tripwire.py`: the UNC assertion uses `os.sep`. The behaviour does not change: off Windows the prefix is stripped too, which errs towards a trip.
- `docs/PROTOCOL.md`: the rule under Leakage tests, and a dated change-log entry.

**Then.**
1. `leakage-auditor` reviews the fix.
2. H1 and Report n°1 are run again on the corrected code.
3. The rows of `docs/RESULTS.md` come from the second run.

**Done when.**
- The full suite is green on Windows and on Linux CI.
- The auditor passes the fix.
- The second H1 run holds with every control caught above rounding.

**Audit (Audit A, round 5): pass with fixes, two Minor points.**
- The table showed a control's differing bits as "violating rows". It now shows the caught rows of a control, under "rows flagged".
- `docs/RESULTS.md` should not let the 9c414ca H1 row stand alone. An erratum row is appended with the second run.

Measured by the auditor on the real data, through the sealed loader:
- every honest model moves by 0 at every cut;
- no model has a rounding-only cut;
- the weakest real catches are climatology-unpurged at 7.1e-4 and 7.3e-4, and the transductive scaler at 1.8e-3 and 8.0e-4;
- every other control flips a target or moves `exp_return` by 0.8 or more.

Its proposal for later is to make the margin a requirement: H1 fails if a control's smallest catch falls below 1e-6.
