# Brief — the encoder's core: torch, the three attention arms, the learned memory, training, and the sizing (iteration 4, wave 1)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:** another `ml-engineer` (in `python/olsprime/bench/report2.py`, `python/olsprime/bench/power.py` and their tests) and a read-only `leakage-auditor` reviewing the pre-registration. A `rust-engineer` has just added `beta` and `mean_y` to `CentredReadout` (`crates/`, `python/tests/test_readout_beta.py`): that work is done and sits uncommitted in the working tree. Touch none of those files, and nothing under `crates/`.
- **Motto:** we should be more ambitious. Build the model we would be proud to publish, with the tests that would convince a sceptic. Report only what the protocol supports.

## Goal

Build the encoder of `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)", section "The encoder arms", exactly as registered, test it against closed forms, and **size the workload** on the RTX 3060 before any run on real data. That section is the specification: architecture, inputs, the three attention operators, the memory over learned keys, the decision head, episodes, fits, seeds and determinism. Do not change any of it. If something cannot be built as written, or is ambiguous, stop and ask the orchestrator: a change must be dated in the change log before any run.

Read first, in this order:
1. `CLAUDE.md`, then `docs/PROTOCOL.md`'s iteration 4 section (all of it: H1 for the encoder arms and the planted training-time leaks shape your hooks), `docs/DESIGN.md` §1, §3.2 and §4, and `docs/HYPOTHESES.md` (H4–H6, H13).
2. The code you build on: `python/olsprime/features/panel.py` (`pair_rows`, `assemble`, `hourly_panel`, the bar columns), `python/olsprime/features/basic.py` (the gap rule), `python/olsprime/models/matrix.py` (`PanelArrays`, the stamps, `seqsum`, `prob_map`, the determinism notes), `olsprime_rs.Matrix` (`window_range`, `centred_query`, `centred_stats` in `crates/olsprime-py/olsprime_rs.pyi`), and `python/tests/{conftest,tripwire,allowlist}.py` with `docs/DATA.md`'s "Runtime tripwire".

## Deliverables (wave 1: no real data)

1. **torch as a dependency.**
   - CPU torch on Windows and on CI (GitHub's ubuntu and windows runners), CUDA torch in WSL2 for the 3060. The lockfile covers both. A suggestion: two conflicting dependency groups, `cpu` and `cu12x` (or `cu13x`), each sourcing torch from its PyTorch index, with `default-groups = ["dev", "cpu"]`, so that `uv sync --locked` and `uv run` keep giving CPU torch everywhere, and WSL2 runs `uv sync --locked --no-default-groups --group dev --group <cuda group>` in its own environment. Check it with the installed uv, and choose the newest stable torch whose CUDA build supports driver 617.42.
   - **Hard constraint: the lock changes by additions only.** Every package already locked keeps its exact version, numpy, scipy and polars above all: Report n°2 must re-run byte for byte on this environment. Show the summary of the `uv.lock` diff in your report.
   - Change `.github/workflows/ci.yml` only if CI needs it.
   - The WSL2 environment: `UV_PROJECT_ENVIRONMENT=~/olsprime-gpu/venv` and `CARGO_TARGET_DIR=~/olsprime-gpu/target`, both on the Linux disk, run from the repository at `/mnt/c/Users/abdou/ols-prime/python`. Use `uv run --no-sync` there once it is built. Write the exact commands in your report.
2. **torch and the guards, tested first** (`docs/DATA.md`: the torch wrapper of the tripwire was never tested against the real library).
   - In a new `python/tests/test_torch_guards.py`: `torch.load` of a file under the repository's `data/` trips; `torch.load` from a file without the `import:pickle` allowance trips; the static guard flags `torch.load` in a source file. Fix the tripwire or the static guard if the real library breaks them, and say what you changed.
   - Checkpoints never go through pickle. Save parameters as plain arrays (a deterministic `.npz` written with fixed zip entry dates, or safetensors if you add it under the same lock rule), and load them without `allow_pickle`.
3. **The modules**, in new files `python/olsprime/models/encoder.py` and `python/olsprime/models/memory.py`, and a new package `python/olsprime/train/`:
   - **The windows.** The eight bar quantities of row (s, t) over its 64 bars, built from the same bars `pair_rows` receives, so that the panel H1 can rebuild them from perturbed bars. A window that is not 65 consecutive hours is flagged, never silently filled. The standardization statistics are computed from the fit's training rows only, and are a function of them alone.
   - **The three arms**, sharing every weight but the operator. `enc-ols` exposes its W (64 × 64 per head) for `head`. Each arm reports its exact parameter count, with λ counted and the constant key coordinate not.
   - **The memory over learned keys**, with the stamp as a hook (`label_known_at` honest, decision time for `cheat-stamp`). Prefix sums over entries ordered by stamp, differenced in fp64, solved in fp64. Every reduction behind a forecast at d must depend only on the entries it covers, in a fixed order, so that a run whose future differs gives the same bits at d: sequential sums, or a reduction whose order does not depend on the array's length or later values.
   - **The decision head**, and the value-path hook for `cheat-value`: a learned projection of the row's own one-hot label, added to the last position's value in every head. It is off in every honest arm, and nothing in `models/` turns it on by default.
   - **Training** (`train/`): episodes of one calendar week; the cache without gradient; the fit at F with its training rows, purged slice and early stopping; AdamW with the registered schedule; seeds; the determinism settings; checkpoints as above. A fit's config is a frozen dataclass saved with it.
4. **Closed-form tests**, in new test files (`test_encoder.py`, `test_memory.py`, `test_train.py`), CPU torch, so that they run on CI:
   - an OLS head equals a numpy ridge forecast with the unpenalized constant coordinate, and every row of its W sums to 1 within a tolerance you derive;
   - the memory equals the Rust core on a synthetic episode: at every decision time d, the same entries as `Matrix.window_range(d, 30 days)`, and the readout of `centred_query` within rounding. Map the penalty exactly: Λ = ρ·diag(M_zz) + 10⁻⁶·n_w·I is `scale` = var_w + 10⁻⁶/ρ in `centred_query`;
   - `cheat-stamp`'s memory differs from the honest one exactly where a row decided before d has a label known after d;
   - parameter counts: equal for `enc-softmax` and `enc-linear`, + 8 for `enc-ols`;
   - a tiny overfit test (one batch), a gradient check through the OLS solve and the memory, and determinism on CPU (two runs give the same bits);
   - a fit uses no row whose label is known after F − 30 days, and its slice none known after F − 24 h, tested by planting an extreme label just past each limit.
5. **The sizing, on the RTX 3060 in WSL2, on synthetic data of the real panel's shape.** This fixes the schedule, so it comes before any run on real data.
   - Shape: 30 members per hour, hourly rows from 2017-10 to 2023-12 (the 2024 fit's training set, about 1.3 M rows), 64-bar windows, the real panel's share of complete rows.
   - Measure, for each arm: the time of one step (one weekly episode, with its cache), of one epoch, and of one whole 2024 fit of `enc-ols` (6 epochs, with the slice evaluations); the peak GPU memory; and the inference time per 10⁴ decisions under H13's conditions.
   - Project: one seed of one arm over the 6 fits; the 54 fits of H4 and H5 (3 arms × 3 seeds × 6 fits); the panel H1 on the 10-member sub-panel (per arm, the fits of the cut years plus 16 perturbed fits); the two planted leaks.
   - **Determinism pre-check on synthetic data:** two runs of the same fit on the GPU, with the registered settings. Are the parameters and outputs bit-identical? If not, find the op responsible and whether a deterministic path exists.
   - Record the driver, CUDA and torch versions and the GPU name.

## Constraints

- **No real data.** Run nothing on the real store in this wave: no panel, no bars, no smoke run. Synthetic data only. The real runs come after the audit of the pre-registration.
- **Files.** New: `python/olsprime/models/encoder.py`, `python/olsprime/models/memory.py`, `python/olsprime/train/`, `python/tests/test_encoder.py`, `test_memory.py`, `test_train.py` and `test_torch_guards.py`. Changed: `python/pyproject.toml`, `python/uv.lock`, `python/olsprime/models/__init__.py` (exports only), `.github/workflows/ci.yml` if needed, and `python/tests/tripwire.py`, `allowlist.py` or `conftest.py` only if the real torch requires it. Ask before touching anything else. Do not edit `docs/`: report the facts and the orchestrator writes them.
- **Environments.** Do not sync the shared `python/.venv`: the other agent's tests load its compiled extension, and Windows cannot replace a loaded `.pyd`. On Windows, build and test in your own environment: `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` set to two folders under your scratch, then `uv sync --locked` and `uv run --no-sync pytest` from `python/`. The orchestrator syncs the shared `.venv` after this wave. Never upgrade an existing package.
- **Scripts.** Write scripts with a file tool. Never pipe or heredoc code into `python`. In WSL2, run Python with `uv run --no-sync python -I <file>`.
- **Processes.** Give every long command a timeout (`timeout` in WSL2). You cannot kill a hung process here, and nobody is awake to do it.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\encoder-core\` (in WSL2: `/mnt/c/Users/abdou/AppData/Local/Temp/claude/C--Users-abdou--claude-spares-spare-j944HL/360b1384-377b-4a3a-9ef7-332b5c4fe2de/scratchpad/encoder-core/`), and `~/olsprime-gpu/` on the Linux disk.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- torch is locked as specified, and CPU tests run on Windows.
- The guards are tested against the real torch.
- The modules and their closed-form tests are in place, and `uv run pytest` passes on Windows.
- The sizing numbers and the determinism pre-check on synthetic data are in your report.

## Report (short)

- The dependency set-up: groups, versions, commands for Windows, CI and WSL2, and the `uv.lock` diff summary.
- The guards: what the tests showed, and any fix.
- The modules: exact parameter counts per arm; how the memory's causality is enforced; what the tests cover.
- **The sizing table** and the projections, with the GPU, driver, CUDA and torch versions.
- The determinism pre-check: bit-identical or not, and why.
- Every point where the pre-registration needed an interpretation, and the one you took.
- One bolder next step.
