# Brief — audit pass 4g: `explain head`, before it reads any real checkpoint

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with** the second look of audit pass 4e (another `leakage-auditor`, on the encoder's panel H1) and with the official runs: the panel H1 on the GPU in WSL2 for about nine hours, the power study and the five whys on Windows. None of them reads the code below.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

`explain head` explains, per OLS head of a fitted `enc-ols`, what the head attends to at one scored row: the row of W, its diagonal mass, its effective rank, its row sums, λ and the condition number, with two loud checks (the row sums, and the row's forecast recomputed from the checkpoint to the bit). It will run on real data only after Report n°3's fits, in WSL2, and every call is a logged contact on the test months. Break it before then: a contact it fails to log, a check that cannot fail, an explanation drawn from a fit that saw the row's future, or a W that is not the one behind the forecast.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: "`olsprime.explain`" and the encoder sections of iteration 4; in the change log, the entry that starts "- 2026-10-30, after audit passes 4c to 4f", whose "`explain head`, its readings" fix what `head` must compute;
- `.claude/briefs/iteration-4/explain-head.md`, the engineer's brief;
- the files under review, uncommitted in the main tree: `python/olsprime/explain/head.py` (new), `python/olsprime/explain/cli.py` (`git diff`), `python/tests/test_explain_head.py` (new), `python/tests/test_torch_free.py` (`git diff`);
- what they call: `python/olsprime/bench/report3.py` (`verified_part`, `load_fit`, `read_forecast`, `load_panel`, `check_panel`, `Parts`), `python/olsprime/train/fit.py` (`EncoderSystem.predict`, `_encode`), `python/olsprime/models/encoder.py` (`ols_weights`, `row_sum_tolerance`, `ols_condition`), `python/olsprime/explain/common.py`.

## A finding that shapes where `head` runs

On identical synthetic bars, `np.log` and `np.exp` give different last bits on Windows and in WSL2, so the panel's log-based columns and its fingerprint differ between the platforms (found by the engineer, reproduced by the orchestrator). Report n°3 therefore runs on a Report n°2 run made in WSL2, and `head` runs in WSL2 on Report n°3's folder. Check that `head` refuses a panel from the other platform loudly, before it reads any value.

## The engineer's interpretations, to accept or reject with a reason

1. Without `--seed`, every seed of the run's own `cfg.seeds`, each apart; one row per call, no `--at`; one contact line per call, whatever the number of seeds.
2. The fit in force is the fit of the row's year by the protocol's rule (the month of its bar's open): the fit Report n°3 forecast the row with. An `AS_OF` that matches two rows is refused.
3. The pass computes O = Q̃ (A⁻¹ (K̃ᵀV)) and never forms W. W is formed by `ols_weights` from the captured Q, K and λ of the pass that gave the forecast, and the attention recomputed from the captured input must reproduce the pass's output to the bit. W is formed on the row's slice, not on the whole batch.
4. The effective rank is taken over all 64 singular values. The tail is rounding (a share of about 1e-7), so the value can exceed 9 by about 1e-5. The 9 leading singular values are given.
5. Row sums are summed in fp64 over the fp32 weights, against `row_sum_tolerance` unchanged. The measured gaps were at most 0.005 of the bound, so the check catches gross errors only.
6. The panel is rebuilt from the store with the registered `Report2Config` through `report3.load_panel` when `report3.json` records a registered Report n°2; any other run hands its panel in through `panel_source`. The fingerprint is required either way.
7. Report n°3's `series.npz` is hashed, never parsed, so the floored ensemble is not checked.

## What to attack

1. **Contacts.** Is every read of a test-month value preceded by its contact line? Can any path (a refusal, an exception, `--jsonl` failing, a seed refused mid-call) read a real value without a line? Is a refused argument logged or not, and is that what the protocol asks?
2. **Look-ahead in the explanation.** Can `head` explain a row with a fit whose training saw the row's label or later bars? Check the year rule at the year's edges, the purge and `label_known_at`, and rows of 2019, the first fit year.
3. **The model check.** Does it recompute through the fits' own computation (`predict`, the fixed batches of `eval_batch` rows in panel order, the memory's prefix sums on the CPU in fp64, the deterministic settings, the device)? Could it compare the saved forecast with itself, or with a copy that cannot differ? Plant a one-ulp change in a checkpoint's weight and in the saved forecast, and see each stop the call.
4. **W.** Is the W reported the one behind the forecast: the right block and head, the right row, the window's open times in the right order, the decision token last? Is the bit-exact reproduction of the attention's output a real proof that the captured Q, K and λ are the pass's?
5. **The refusals.** A part copied from another arm, seed or year; another build or device; another panel; a row before 2019, after 2024 or unscored; a climatology fallback. Is any refusal reachable only after a value was read?
6. **The build check.** Can an honest official call be refused spuriously, for example because the CPU thread count of the fits' process differs from the call's?
7. **Determinism and guards.** JSON with sorted keys and no wall clock; no pickle; `numpy.load` banned; only the sealed loader reads bars; `python -m olsprime.explain` and `why`, `series` and `sample` still run without torch.
8. **The tests.** Does each one fail when the defect it names is planted?

## Constraints

- **Read-only.** Run the tests and your own probes. Write probes only as files in your scratch folder, `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4g\`, with a quoted heredoc if you must. Run them with `uv run --no-sync python -I <file>`.
- **Environments.** On Windows, a venv under your scratch: set `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` there, then `uv sync --locked`. Never sync the shared `python/.venv`. CPU only: the GPU runs the official panel H1. Do not touch `~/ols-prime-official` or `~/ols-official-gpu` in WSL2.
- **No real data.** Nothing on `data/`, the real store or the vault.
- **Processes.** Give every long command a timeout. You cannot kill a hung process here.
- **Git.** No command that changes state.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Report

- GO or NO-GO for `head` on Report n°3's real checkpoints.
- Findings by severity (Blocker, Major, Minor, Nit), each with file:line, a concrete failure scenario and a fix.
- Each of the seven interpretations: accepted or rejected, with the reason.
- One bolder check.
