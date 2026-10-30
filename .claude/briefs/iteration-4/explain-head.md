# Brief — `explain head`: what each OLS head of a fitted `enc-ols` attends to (iteration 4, wave 3)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with** the official GPU runs in WSL2 (the panel H1, then Report n°3's fits), which use the clean clone `~/ols-prime-official` and its venv `~/ols-official-gpu/venv`: never touch either. The GPU is busy with them; keep your GPU probes short and few.
- **Motto:** we should be more ambitious. Build the view that would convince a sceptic of what a head does, and claim nothing it has not checked.

## Goal

`docs/PROTOCOL.md`, "`olsprime.explain`", registers `head`: once the encoder exists, it gives per OLS head the row of W, its diagonal mass, its effective rank and its row sum, 1 by construction. Build it as a fourth subcommand of `olsprime.explain`, on Report n°3's fitted checkpoints. It runs on real data only after its audit and after Report n°3's fits.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: the iteration 4 section, "`olsprime.explain`" and the encoder; the change-log entries of 2026-10-30;
- `python/olsprime/explain/` (`cli.py`, `common.py`, `why.py`, `views.py`), whose rules `head` follows: contacts logged before the run folder is opened, digest checks, `--test-months`, `--jsonl`, exit codes 0, 2 and 3, sorted-key JSON with no wall clock, provenance on every answer;
- `python/olsprime/models/encoder.py`: `Encoder.ols_weights`, `ols_weights`, `row_sum_tolerance`, `ols_condition`;
- `python/olsprime/bench/report3.py`: its run folder, its parts (`fits/<arm>/s<seed>/<year>/`), their digests, `verified_part` and the hand-written safetensors checkpoints, and `train/checkpoint.py`.

## The readings, fixed by the orchestrator before any encoder run on real data

These are written in the protocol's change log. Build them exactly; if one cannot be built as written, stop and ask.
1. **The call.** `head PAIR AS_OF --run DIR --sha256 HEX [--seed 1|2|3] --test-months [--jsonl FILE]`. `DIR` is a Report n°3 run folder and `HEX` the SHA-256 of its `series.npz`, which its `report3.json` must agree with. Without `--seed`, all three seeds are given, each apart.
2. **The rows it explains.** Only scored rows of 2019–2024, which Report n°3 forecasts, so every call needs `--test-months` and is a logged contact. A row before 2019-01-01 is refused: no fit of Report n°3 is in force there.
3. **The model.** The `enc-ols` checkpoint of the seed's fit in force at the row's decision time: the fit of that calendar year. Its part is verified by digest and checked against its folder (arm, seed, year, `fit_ms`) before any value is read.
4. **W.** For each block b ∈ {1, 2} and head h ∈ {1, …, 4}: W = Q̃ (K̃ᵀK̃ + Λ)⁻¹ K̃ᵀ over the row's 64-bar window, a 64 × 64 matrix, computed in the forward pass that gives the row's forecast. W has rank at most d_head + 1 = 9.
5. **The row of W** is the row of the window's last token, the decision bar: 64 signed weights, oldest bar first, with their bars' open times.
6. **The diagonal mass** is trace(W) / 64, the mean weight a token puts on itself. A head that copies its input has a diagonal mass of 1. The decision token's own weight, W₆₄,₆₄, is given beside it.
7. **The effective rank** is exp of the Shannon entropy of W's normalized singular values (Roy and Vetterli, 2007), between 1 and 9 here.
8. **The row sum.** Every row of every head must sum to 1 within `row_sum_tolerance`. That is a loud check: a failure stops the call. The decision token's row sum and the largest |row sum − 1| are given, with the bound.
9. **The head's own state:** its λ and the condition number of K̃ᵀK̃ + Λ.
10. **The model check, loud.** The row's forecast is recomputed from the checkpoint and must equal the seed's saved forecast of that row. It must match to the bit when computed as the fits computed it: same device, same fixed batches of 4,096 rows in panel order, the memory's statistics as `predict` builds them. If that cannot be reproduced on some device, derive and state a bound, and refuse beyond it.

## Deliverables

1. `python/olsprime/explain/head.py`, new, and the subcommand in `python/olsprime/explain/cli.py`.
2. `python/tests/test_explain_head.py`, new, on synthetic data with CPU torch, so it runs on CI in a few minutes. It builds a tiny Report n°3 run with `report3.py --synthetic`, or its test helpers.
   - The outputs: row sums, the diagonal mass of a planted identity-like head, an effective rank against a hand computation.
   - Each refusal: no `--test-months`, a row before 2019, a part copied from another arm or seed, a wrong digest.
   - Each loud check: a planted wrong forecast, a planted row-sum failure.
   - Contacts: one line per row asked, written before the run folder is opened.
   - Determinism of the JSON.
3. A GPU smoke on a synthetic Report n°3 run in WSL2, in your own environment: the forecast check to the bit.

## Constraints

- **torch stays lazy.** `why`, `series` and `sample` must keep running in an environment without torch, and `python -m olsprime.explain` must not import torch unless `head` is called. Import `head.py`'s torch code inside the subcommand's dispatch, and add `olsprime.explain` and `olsprime.explain.cli` to `TORCH_FREE` in `tests/test_torch_free.py` (that edit is allowed).
- **Files.** New: `explain/head.py` and `tests/test_explain_head.py`. Edited: `explain/cli.py`, and `explain/common.py` only if a shared helper must change. Ask the orchestrator before touching anything else, `models/`, `train/` or `bench/` included.
- **No real data.** Nothing on `data/`, the real store or the vault.
- **Environments.** On Windows: `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` under your scratch, then `uv sync --locked`. Never sync the shared `python/.venv`. In WSL2: `~/olsprime-gpu/venv` (`UV_PROJECT_ENVIRONMENT=~/olsprime-gpu/venv CARGO_TARGET_DIR=~/olsprime-gpu/target`); do not rebuild it.
- **Scripts.** Write them with a file tool. Never pipe or heredoc code into `python`. Run Python with `uv run --no-sync python -I <file>`.
- **Processes.** Give every long command a timeout. You cannot kill a hung process here, and nobody is awake to do it.
- **Scratch:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\explain-head\`.
- **Git.** No command that changes state.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- `head` runs on a synthetic Report n°3 run, on CPU and on the GPU, and its forecast check passes to the bit where the fits' computation is reproduced.
- Its tests pass on Windows, with `test_explain.py`, `test_protocol.py` and `test_torch_free.py`.

## Report (short)

- The design as built, and how the forecast check reproduces the fits' computation.
- Every point where a reading above needed an interpretation, and the one you took.
- One bolder next step.
