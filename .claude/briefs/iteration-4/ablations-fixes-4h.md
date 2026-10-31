# Brief — audit pass 4h's fixes to LFM3's ablations, and a seed-split control for H6 (iteration 4, wave 3)

- **Agent:** `ml-engineer` (opus, high).
- **Why now:** audit pass 4h gave GO with fixes: no Blocker, one Major point, six Minor points and six Nits. The Major point must be fixed before the real score stage, the Minor points before the commit. The GPU stays busy with the panel H1 and Report n°3's fits until about 00:00–02:00 UTC, so nothing here needs it.
- **Motto:** we should be more ambitious, in what we build and test, never in what we claim.

## Goal

Fix every finding of audit pass 4h in the uncommitted ablations code of the main tree. Then add one control beside H6, registered by the orchestrator in `docs/PROTOCOL.md`'s change log before any real run.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: "LFM3's ablations and H6, if time remains", "Variants", and the change-log entries of 2026-10-30;
- `.claude/briefs/iteration-4/ablations.md`, the build's brief;
- `.claude/briefs/iteration-4/audit-code-4h.md`, the audit's brief;
- the audit's report, quoted in full below under "The findings".

## The findings to fix

### Major

**M1. A fitted arm can be reported as "not run" by a run that calls itself registered.** `ablations.py:669` builds `not_run` from `cfg.arms` alone. The fits stage accepts any arms, and the parts' identity strips `arms` (`:156-161`). So all 54 fits can run and a score with `--arms enc-ols-gate enc-ols-conv` then reports H6 "not run", 23 variants, `registered: true`.

Fix it this way:
- **The score takes an inventory of every arm of `ABLATIONS` in the folder.**
  - An arm in `cfg.arms`: as now, every part must verify, or the score refuses.
  - An arm not in `cfg.arms` whose 18 parts are all sealed: the score refuses, with a message telling the operator to score it.
  - An arm not in `cfg.arms` with some parts: reported as fitted in part (k of 18 parts) and not scored. It goes in `ablations.json` (for example `"fitted_in_part": {arm: {"parts": k, "of": 18}}`) and in the report's line "Not run: …", never in `not_run`. It is not a variant.
  - A part that does not verify refuses the score, whatever its arm.
  - An arm with no part is "not run".
- **The fits stage refuses an arm list that is not a prefix of `ABLATIONS`**, unless every part of each earlier arm is already sealed in the folder. "If time remains" means the arms are fitted in the protocol's order.
- `registered` keeps its rule. A run that stops in the middle of `enc-ols-conv` is still registered for `enc-ols-gate`, with `enc-ols-conv` fitted in part and H6 not run.

### Minor (five test gaps; the code is right but the suite cannot tell)

- **m1. The Brier term is taken before the floor.** The test `test_the_brier_arm_adds_the_protocols_brier_score_to_the_cross_entropy` (`test_train.py` around 372-385) draws `randn` logits, whose probabilities never fall under 10⁻³. A floored Brier term passes it. Add rows with a logit gap above 7, so that a floored and an unfloored Brier differ, and assert the unfloored one.
- **m2. The Brier arm's early-stopping statistic.** Nothing checks that `enc-ols-brier` keeps the epoch with the lowest slice log-loss. A plant that negates the slice loss under `BrierFitConfig` passes. In `test_two_fits_of_each_ablation_give_the_same_bits`, assert `kept_epoch == argmin(slice_losses) + 1` for the Brier arm. Recompute one slice loss from `predict` as a log-loss.
- **m3. The inclusive log-loss condition.** `test_ablations.py:296-308` has no case at `diff == tolerance`, and a strict `<` passes. Add `(0.008, 0.95, 0.002, 0.002, 0.001, True, "diff equal to the tolerance holds")`, or its equivalent in the test's own form.
- **m4. The ratio bootstrap's block length.** In `test_each_replicate_of_the_ece_ratio…` (`test_ablations.py:311-332`) both arms' Politis-White blocks coincide, so `block = _block(br_a)` passes. Give `p_b` an hourly Brier with longer dependence (an AR(1) pattern over hours), so that the two blocks differ, and keep the `max` assertion.
- **m5. The two guards that bind the numbers to Report n°3's series.**
  - The series-rebuild hash (`ablations.py:352-360`) is the only check that refuses a forecast forged with its whole chain re-sealed, and disabling it passes the suite. Add one test: change one row of one `enc-ols` forecast, re-seal it, rewrite `report3.json`'s records consistently (`part.json` digests, the `fits` entry, `arms.enc-ols`), and expect `BindingRefused` matching "rebuilt from its bound parts".
  - Add one test that pairs with a second Report n°2 run and expects "paired with Report n°2's series" (`check_rows`, `:305-309`).

### Minor (a reader bug off the ablations' path)

- **m6. `report3.load_fit` cannot rebuild an `enc-ols-brier` checkpoint.** `report3.py:747` calls `FitConfig(**c)`, and the saved config carries `brier_weight`, so it raises `TypeError`. Fix: `cls = BrierFitConfig if "brier_weight" in c else FitConfig`, or equivalent.
  - Every existing config must load exactly as before.
  - Test it: a Brier part's config must come back as a `BrierFitConfig` equal to the fit's. If it is cheap, also check that the loaded model predicts the part's saved forecasts to the bit.

### Nits

- **n1.** The score's refusal "the parts are not the ones its series was made from" (`:358-360`) also fires on a platform mismatch, since `np.log`'s last bits differ between Windows and WSL2. So the score stage is WSL2-bound, like Report n°3's. Say so in that message and in the module's docstring.
- **n2.** `:594` tests `ea <= 0.8 * eb` while the report prints `ratio = ea / eb`. Use the reported ratio in the condition, so that the two cannot disagree by an ulp.
- **n3.** `one_change` (`:187-193`) is tautological inside `check_against_report3`. The load-bearing check is `base_fit_config(cfg) == r3.fit_config(cfg3)` at `:204`. Say so in the docstring, and keep the call.
- **n4** (optional). `bind_base` reads every Report n°3 forecast twice. Fix it only if the change is small and keeps every output byte-identical.
- n5 and n6 go to the change log; that is the orchestrator's job.

## The seed-split control (registered by the orchestrator; build it as written)

This control comes from the audit's bolder check: it measures how far seed noise alone moves H6's ECE condition, with no new fit. It is a description, with no verdict, and not a variant.
- **The pairs.** For each ordered pair (i, j) of Report n°3's `enc-ols` seeds, with i ≠ j, six in all, take each seed's forecasts floored on their own, as `report3.per_seed` floors them, over the scored rows. Compute:
  - ECE_i and ECE_j, with `stats.ece`;
  - the ratio ECE_i / ECE_j and its paired stationary bootstrap, through H6's own machinery (`ece_ratio_interval`). Hours are resampled jointly, the block is the larger of the two seeds' `_block` of their hourly Brier scores, `n_boot` is Report n°2's, and the seed is `r2._seed(rcfg, "ece_ratio", "enc-ols seed i", "enc-ols seed j")`, with i and j the seed numbers;
  - the two-sided 95 % interval and the one-sided 95th percentile;
  - whether H6's ECE condition holds for i against j: the ratio is at most 0.8 and the 95th percentile is below 1.
- **The log-loss condition is not applied.** Two seeds of one model differ by at most the seed range, by construction.
- **Where it goes.** `ablations.json`, under a key of its own (for example `"h6_seed_split"`), and one short table in the report.
  - The report says it is a control with no verdict.
  - It says that H6 compares 3-seed ensembles, whose seed noise is smaller than a single seed's.
  - It changes nothing in H6's verdict, which is computed exactly as before.
- **When it runs.** It runs whenever H6 is scored, from the per-seed forecasts that `bind_base` already binds to `report3.json`. It needs no new input.
- **Tests:**
  - two identical seeds give a ratio of exactly 1, and the condition holds in neither direction;
  - a seed whose forecasts are sharpened with a temperature of 0.7 (a planted miscalibration) moves its ratio away from 1 in the right direction;
  - the seeds are the registered ones;
  - `--check` reproduces it byte for byte;
  - H6's verdict is identical with the control and without it.

## Constraints

- **Your files.**
  - `python/olsprime/bench/ablations.py`;
  - `python/olsprime/bench/report3.py`, for m6 only;
  - `python/tests/test_ablations.py`, `python/tests/test_train.py`, `python/tests/test_encoder_ablations.py` and `python/tests/test_report3.py`, the last for m6's test only.
  - Do not touch `python/olsprime/models/encoder.py`, `python/olsprime/train/fit.py`, `python/olsprime/explain/` or `docs/`. If a fix needs one of them, stop and tell me.
- **`enc-ols` and Report n°3 must not move by a bit.**
  - Report n°3 is being fitted from a clean clone at `5f17a76`.
  - The m6 fix changes the reader only. Every config without `brier_weight` loads exactly as before.
  - Report n°3's runner and its outputs stay byte-identical: `test_report3.py` passes unchanged, except for m6's new test.
- **Environment.** Work in the main tree, `C:\Users\abdou\ols-prime`, on Windows, with the shared `python\.venv` through `uv run --no-sync`. Its editable install points at the main tree. Never sync it, and never touch the WSL2 clones `~/ols-prime-official` or `~/ols-prime-r3`.
- **CPU only.** The GPU runs the official fits.
- **Memory.** Windows' commit charge is short. Before any run that trains, check `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory`. Run only with at least 4 GB free, and with `OMP_NUM_THREADS=2`.
- **Data.** Synthetic data and the test fixtures only. Never read the real panel, `data/`, `runs/` or the vault. Never set `OLSPRIME_OPEN_VAULT`.
- **Process.**
  - Write scripts with a file tool, never through a heredoc or a pipe into `python`, and run them with `uv run --no-sync python -I <file>`.
  - Give every long command a timeout.
  - No git command that changes state.
  - Do not kill processes you did not start.
  - Keep the files' existing style, including the hand-formatted lines marked `# fmt: skip`.

## Done when

- M1, m1–m6, n1–n3 are fixed, and n4 is fixed or left with a reason.
- The seed-split control is built and tested as written above.
- **Every plant of audit pass 4h that the suite missed now fails a named test.** Re-plant each one in a scratch copy, run the matching test, and list it:
  - Brier on floored probabilities;
  - the Brier arm's slice loss negated;
  - `ll_met` with a strict `<`;
  - the block from arm a only;
  - the series-rebuild guard disabled;
  - `check_rows` removed.

  The M1 scenarios, each with a test:
  - a complete arm left unscored is refused;
  - an arm fitted in part is reported as fitted in part, not as "not run", and is not a variant;
  - the fits stage refuses an arm list that is not a prefix.
- These pass: `test_ablations.py`, `test_encoder_ablations.py`, `test_train.py`, `test_report3.py`, `test_encoder_h1.py`, `test_encoder.py`, `test_explain_head.py`.
- If memory allows, a synthetic end-to-end run on CPU: `report3 --synthetic`, then the ablations, then `--check`, which passes. If memory does not allow it, say so.

## Report (short)

- The diff, file by file.
- The table of re-planted defects and the tests that catch them.
- The seed tags and the keys of the control, as built, for the change log.
- The test counts and the end-to-end evidence.
- One bolder next step.

## The findings

The audit's full report, for reference, is in the orchestrator's transcript. Its findings are restated above. Its "clean" list is not repeated here: the arms, H6's rule, the binding, the runner and the look-ahead checks were all accepted.
