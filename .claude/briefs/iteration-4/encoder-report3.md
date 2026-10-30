# Brief — Report n°3: the encoder arms on the panel, H4, H5, H13 and the cost table (iteration 4, wave 2)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:**
  - the `ml-engineer` building the encoder's GPU H1 (`olsprime/bench/encoder_h1.py`);
  - `leakage-auditor` passes.

  Touch only the files listed under Constraints.
- **Motto:** we should be more ambitious. Three attention operators, one memory, six years of test months, and a verdict whose worth the power study states. Claim only what the registered rules support.

## Goal

Build the runner and the report of the encoder arms, exactly as `docs/PROTOCOL.md` registers them in "The encoder, and what a verdict is worth (iteration 4)":
- "The encoder arms", for the fits, seeds and ensemble;
- "H4, H5 and H13", for the rules, the timing protocol and the cost table;
- "Variants".

The build readings are fixed in the change-log entry "2026-10-30, wave 1". If something cannot be built as written, or is ambiguous, stop and ask the orchestrator.

This wave runs on synthetic data only. The orchestrator runs the official fits on the real panel on the GPU, after the panel H1 and after an audit.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: the iteration 4 section, and iteration 3's "Matrix on hourly bars" (Scoring, Decision rules);
- `docs/ITERATIONS.md`, iteration 4: the sizing and the schedule;
- `docs/HYPOTHESES.md`: H4, H5, H13.

Then the code you build on:
- `python/olsprime/bench/report2.py`. You reuse, and never re-implement:
  - its panel, scored rows, hourly series, statistics (Diebold-Mariano with HLN, Politis-White, the stationary bootstrap, the registered seeds), Brier skill intervals and ECE;
  - its `series.npz` writer and its `--check`.
- `python/olsprime/bench/power.py::load_series`, the one reader of `series.npz`, by its digest;
- `python/olsprime/train/fit.py`, `python/olsprime/models/encoder.py`, `memory.py`, `python/olsprime/train/checkpoint.py` and `determinism.py`.

## Deliverables

### 1. A new module `python/olsprime/bench/report3.py`

It runs as `uv run --no-sync python -m olsprime.bench.report3 --series <Report n°2 run>/series.npz --sha256 <digest> [--out DIR] [--stage fits|timing|score|all] [--arms ...] [--check DIR]`, in WSL2 with the CUDA environment.

**The fits stage.**
- The panel is Report n°2's own, built as `report2.py` builds it, through `load_klines` and the seal. The windows come from the same bars.
- For each arm (`enc-ols`, `enc-softmax`, `enc-linear`), seed (1, 2, 3) and fit (1 January 2019 to 2024), it trains the fit and forecasts every scored row of its year. A row's year is that of its month.
- It saves:
  - the checkpoint and its digest;
  - every epoch's digest, the slice losses and the epoch kept;
  - the probabilities of each row before the floor, keyed by row: symbol and decision time.
- It is resumable as the H1 runner is: one part per (arm, seed, fit), verified on restart. It refuses to resume if the config or the code digest changed.
- `--arms` restricts the stage, so that the orchestrator can run the arms in turn.

**The ensemble.**
- For each arm, the mean of the three seeds' probabilities before the floor, then the protocol's floor.
- A scored row whose window fails takes the arm's climatology, and the report counts these rows.

**The timing stage**, H13's protocol exactly, on the 3060:
- the sample: 12,288 scored rows drawn with the seed 20261030;
- per repeat: 3 untimed warm-up batches, then 3 timed batches of 4,096, with `torch.cuda.synchronize()` before and after each;
- the whole pass: encoder, memory readout and head, with the memory's statistics already built;
- plain fp32 eager, the softmax through the math backend, the deterministic settings on;
- 10 repeats alternating the arms, ABAB.

It reports:
- the median ratio of `enc-ols` to `enc-softmax`;
- each arm's minimum, median and maximum;
- as a description, the attention operator's own time.

**The cost table**, every arm of H4 and H5 and the eight Matrix arms:
- parameters;
- analytic FLOPs per decision, with the formula written in the docstring;
- wall time per 10⁴ decisions;
- peak memory: `torch.cuda.max_memory_allocated` on GPU, the process's peak on CPU;
- GPU-hours of training, with the GPU named.

Matrix's timings come from a CPU stage on Windows, on the Ryzen 5 3600 with the portable build of `olsprime_rs`. Columns never mix machines.

**The score stage**, on CPU, deterministic, reading the saved forecasts and `series.npz` by its digest.
- The encoder arms' hourly series pair with Report n°2's hour by hour, on the same scored hours. Any mismatch of hours or rows is refused.
- **H4.**
  - The margin is 0.005 × `climatology-panel`'s mean hourly log-loss.
  - Two bounds on mean(ll_ols − ll_softmax):
    - the one-sided 95 % upper bound, the mean plus t₀.₉₅(T − 1) times the HLN-corrected Diebold-Mariano standard error, as for H12;
    - the stationary bootstrap's 95th percentile.
  - The testability condition first: `enc-ols` or `enc-softmax` must have a Brier skill interval above 0 against `climatology-panel`. Without it, H4 and H13 are reported as not testable.
- **H5.** `enc-ols` against `matrix-select`, whose series comes from `series.npz`. It passes with a Diebold-Mariano p < 0.05 and a bootstrap interval below 0. The cadence limit is stated.
- **H13.** H4 must hold, and the median ratio must be ≤ 0.9.
- `enc-linear` is described against both.
- Each seed's log-loss and their spread, as a description.
- The Brier skill score and ECE of every arm, as Report n°2 gives them.
- The variant count: 21.

**Outputs** to `runs/<UTC date>-report3/` or `--out`:
- `report3.json` (sorted keys, no wall clock), holding the digest of `series.npz`;
- `report-3.md`;
- the encoder arms' `series.npz`, written as `report2.py` writes its own, with its digest;
- `config.json`, with provenance;
- `run.json`, holding the wall clock and the GPU, driver, CUDA and torch versions.

`--check DIR` re-runs the score stage from the saved forecasts and compares byte for byte. The fits are compared by their digests, never re-run.

### 2. Tests, in a new `python/tests/test_report3.py`, on synthetic data, CPU torch, under a few minutes

- The fits stage on a tiny synthetic panel: every part saved, resume gives the same bytes, and a changed config refuses to resume.
- The ensemble is the mean before the floor.
- The score stage on synthetic series gives:
  - H4's two bounds against a direct computation;
  - the testability gate, both ways;
  - H5's rule;
  - H13's rule, including "H4 does not hold".
- Pairing: a missing hour or a shifted row is refused.
- `--check` passes on an unchanged run and fails on one changed digit.
- The timing stage's protocol, run on CPU with tiny sizes: the warm-ups, the ABAB order and the scaling to 10⁴.

### 3. A GPU smoke on synthetic data, in WSL2

Run every stage on a small synthetic panel. Measure the time, and project the official fits from it and from the sizing in `docs/ITERATIONS.md`. Never run on the real data.

## Constraints

- **No real data.** Nothing on the real store, `data/` or the vault.
- **Files.**
  - New: `python/olsprime/bench/report3.py` and `python/tests/test_report3.py`.
  - Ask the orchestrator before touching anything else.
  - Never edit these files, owned by other agents or the audit:
    - `report2.py` and `power.py`;
    - `olsprime/explain/`;
    - `encoder_h1.py`;
    - `train/` and `models/`. If you need a hook there, ask with the exact change.
  - `olsprime.models` must stay free of torch.
- **Environments.**
  - On Windows: `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` set to folders under your scratch, then `uv sync --locked`. Never sync the shared `python/.venv`.
  - In WSL2: `~/olsprime-gpu/venv`. Do not rebuild it unless it is broken.
- **Scripts.** Write them with a file tool. Never pipe or heredoc code into `python`.
- **Processes.** Give every long command a timeout. Nobody is awake to kill a hung process.
- **Scratch:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\encoder-report3\`.
- **Git.** No git command that changes state, and no commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- Every stage runs on synthetic data, on the GPU and on CPU, deterministic and resumable.
- Its tests pass on Windows.
- `uv run --no-sync pytest` passes, except for failures you show are caused by other agents' files in progress.

## Report (short)

- The design as built, with the smoke time and the projection.
- The FLOPs formulas.
- Every point where the protocol needed an interpretation, and the one you took.
- One bolder next step.
