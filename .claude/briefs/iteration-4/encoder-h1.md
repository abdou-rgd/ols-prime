# Brief — the panel H1 for the encoder arms on GPU, with its pre-check and planted controls (iteration 4, wave 2)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:**
  - a `leakage-auditor` auditing `olsprime.explain` (read-only);
  - the series-power `ml-engineer`, finishing `olsprime/bench/power.py`;
  - later, a second auditor on wave 1's code.

  Touch only the files listed under Constraints.
- **Motto:** we should be more ambitious. The bolder check of this iteration is here: a leak planted at training time must be caught bit for bit on a GPU. Build the instrument that would convince a sceptic, and claim nothing it has not shown.

## Goal

Build the registered panel H1 for the encoder arms. The specification is `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)", sections "H1 for the encoder arms", "The purge cuts", "Planted training-time leaks" and "The positive control, `pos-window`". Build it exactly as written. If something cannot be built as written, or is ambiguous, stop and ask the orchestrator: a change must be dated in the change log before any run. Also read the change-log entry "2026-10-30, wave 1", which fixes the encoder's build readings.

This wave runs on synthetic data only. The orchestrator runs the official H1 on the real sub-panel after an audit.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: the iteration 4 section, "Leakage tests" and the panel H1 of iteration 3;
- `docs/ITERATIONS.md`, iteration 4: the sizing and the schedule.

Then the code you build on:
- `python/olsprime/bench/panel_invariance.py`, the panel H1 of iteration 3, which you reuse as far as it goes:
  - `load_subpanel`, `choose_cuts`, `cut_row`, `perturbed_bars`, `build_panel`;
  - `compare` (H1's catch rule, through `olsprime.bench.invariance.CAUGHT_ABOVE`);
  - the predicted-pattern helpers;
  - `render_md` and `write`.
- `python/olsprime/train/fit.py`. Its hooks are, by name:
  - `EncoderSystem.entry_stamps`, for `cheat-stamp`;
  - `FitConfig(encoder=EncoderConfig(value_leak=True))`, for `cheat-value`;
  - `FitConfig.train_gap_ms`: 0 for `cheat-purge`, 24 h for `cheat-slice`;
  - `EncoderSystem.label_array` returning `past_window_labels`, for `pos-window`;
  - `readout_as_of`, `slice_mask`, `FitResult.epoch_states` and the epoch digests.
- `python/olsprime/models/encoder.py`, `python/olsprime/models/memory.py`, `python/olsprime/train/determinism.py` and `checkpoint.py`;
- the tests `python/tests/test_train.py`, whose mini-H1 is a seed for yours.

## Deliverables

### 1. A new module `python/olsprime/bench/encoder_h1.py`

It runs as `uv run --no-sync python -m olsprime.bench.encoder_h1 [--out DIR] [--device cuda|cpu] [--stage precheck|h1|all]`, in WSL2 with the CUDA environment of `docs/ITERATIONS.md`.

**Set-up: iteration 3's panel H1, unchanged.**
- The same 10-member sub-panel, loaded through `load_subpanel`, so every read goes through `load_klines` and the seal.
- The same 8 cuts, from `choose_cuts` with iteration 3's configuration. Assert that they equal the cuts recorded in `runs/2026-10-29-h1-panel/h1-panel.json`: 2020-03-01T00:00, 2020-05-29T14:00, 2021-07-31T23:00, 2022-07-05T17:00, 2024-03-04T13:00, 2024-10-06T15:00, 2024-11-13T15:00 and 2024-12-03T13:00.
- The same perturbation in both directions (x3 and /3), with the same seeds and mirror scheme. A member that `perturb_after` refuses is left out and counted, as before.
- **The two purge cuts** for the 2021 fit (F = 2021-01-01T00:00 UTC):
  - at F − 24 h = 2020-12-31T00:00 UTC;
  - at F − 30 days = 2020-12-02T00:00 UTC.

  They use the same perturbation, both directions, and seeds derived like the others'. Say how you derived them.

**The runs.** Seed 1, for each system below.
- The honest run trains, on the sub-panel, the fit of each cut's year: 2020, 2021, 2022 and 2024, with 2021 for the purge cuts. It forecasts every scored row of that year.
- Each perturbed run:
  - rebuilds the sub-panel and its windows from the perturbed bars;
  - retrains the fit of the cut's year on them;
  - forecasts the rows decided at or before the cut bar's `known_at`.

**What is compared**, at or before the cut, bit for bit, with H1's catch rule (a move above 10⁻⁹, or a NaN on one side only).
- At the 8 cuts:
  - the fit's inputs: training and slice rows, standardization, climatology and the order of the episodes;
  - every trained parameter of every epoch, each epoch's slice loss and the epoch kept;
  - the memory's stamps and statistics at each compared decision;
  - every output.
- At the purge cut at F − 24 h: the 2021 fit's inputs, every trained parameter, each epoch's slice loss and the epoch kept. No output is compared.
- At the purge cut at F − 30 days: the training rows, the standardization, the order of the episodes, and the trained parameters after every epoch. Not compared: the slice's labels and losses, the epoch kept and the climatology.

An honest system is flagged by a single differing bit. A planted leak is caught only by H1's catch rule.

**The pre-check**, the first stage, which must pass before the H1 stage starts unless `--stage h1` is given explicitly:
- two honest runs of each arm with identical inputs, compared on everything above;
- the scope test: the honest run restricted to the rows decided at or before the first cut must give the full honest run's outputs and memory statistics on those rows.

Report it per arm and per item, bit for bit.

**The systems:**
- the honest arms `enc-ols`, `enc-softmax` and `enc-linear`, at the 8 cuts and the 2 purge cuts;
- `cheat-stamp` and `cheat-value` (`enc-ols`), at the 8 cuts and the 2 purge cuts. They are predicted invariant at both purge cuts;
- `cheat-purge` and `cheat-slice` (`enc-ols`), at the 2 purge cuts only:
  - `cheat-purge` is predicted caught at both, in both directions;
  - `cheat-slice` is predicted caught at F − 30 days in both directions, and invariant at F − 24 h.

**Predicted patterns**, computed per cut from the data as iteration 3 computes them: a cheater is predicted caught at a cut when some compared item reads a label whose exit bar opens after the cut. The computed prediction is the one tested. Write the rule for each control in the docstring, with the reason.

**Skill**, on the scored rows of the sub-panel in the cut years, from the honest runs:
- the Brier skill score against `climatology-panel`, with the same 95 % interval machinery Report n°2 uses, for every honest arm, `cheat-stamp` and `cheat-value`;
- the registered check: both cheaters at least 0.1, each with an interval above every honest arm's interval.

**`pos-window`** (`enc-ols`, seed 1):
- its label is `past_window_labels`, and its entries keep `label_known_at`;
- it runs the honest fits of the cut years on the sub-panel;
- it is scored on the same rows against the climatology of its own label;
- it must reach at least 0.5.

**The fallback, built but not planned.** `--device cpu` runs the same H1 on CPU torch with the same settings and one thread. A "floor" mode computes the largest pairwise difference over three honest GPU runs, quantity by quantity, and gives each control's smallest catch as a multiple of it. Both are tested on synthetic data on CPU. They run on real data only if the registered pre-check fails.

**Resumable.** A run of several hours must survive an interruption.
- Each finished part, (system, arm, fit year, cut, direction), is saved under `<out>/parts/` with its digest.
- A restart verifies each part and skips it.
- It refuses to resume if the config or the code digest changed.

**Outputs** to `runs/<UTC date>-encoder-h1/` or `--out`:
- `encoder-h1.json` (sorted keys, no wall clock);
- `encoder-h1.md`;
- `config.json`, with provenance as `panel_invariance` writes it;
- `run.json`, holding only the wall clock, the GPU name and the driver, CUDA and torch versions.

The report opens with:
1. the pre-check;
2. `cheat-stamp`'s verdict, the first finding if it is not caught bit for bit;
3. `pos-window`'s.

Two runs on the same machine give byte-identical `encoder-h1.json` and `.md`.

### 2. Tests, in a new `python/tests/test_encoder_h1.py`, on synthetic data, CPU torch, so that they run on CI

Keep them under a few minutes in all, with tiny models and few weeks.
- A synthetic sub-panel with cuts and two purge cuts:
  - the pre-check passes;
  - every honest arm is invariant at every cut;
  - `cheat-stamp` and `cheat-value` are caught where predicted;
  - `cheat-purge` is caught at both purge cuts;
  - `cheat-slice` is caught at F − 30 days only.
- A planted defect in the compare path is caught: one parameter bit flipped, or one row of the memory statistics moved.
- Resume: a run cut in the middle and resumed gives the same bytes as one run straight through.
- Determinism of the outputs.
- The fallback's floor mode on CPU.

### 3. A GPU smoke on synthetic data, in WSL2

Run the whole stage list on a small synthetic sub-panel on the 3060, and measure its time. Then project the official run from your measured step times and the sizing in `docs/ITERATIONS.md`. Never run it on the real data.

## Constraints

- **No real data.** Nothing on the real store, `data/` or the vault.
- **Files.**
  - New: `python/olsprime/bench/encoder_h1.py` and `python/tests/test_encoder_h1.py`.
  - Ask the orchestrator before touching anything else. If `train/fit.py`, `panel_invariance.py` or another module needs a hook, ask with the exact change. Iteration 3's modules must keep their outputs byte-identical.
- **Environments.**
  - On Windows: `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` set to folders under your scratch, then `uv sync --locked` and `uv run --no-sync pytest` from `python/`. Never sync the shared `python/.venv`.
  - In WSL2: `~/olsprime-gpu/venv`, already built for CUDA by the encoder agent. Do not rebuild it unless it is broken, and say so if it is.
- **Scripts.** Write them with a file tool. Never pipe or heredoc code into `python`. In WSL2, run Python with `uv run --no-sync python -I <file>`.
- **Processes.** Give every long command a timeout. You cannot kill a hung process here, and nobody is awake to do it.
- **Scratch:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\encoder-h1\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- The module runs every stage on synthetic data, on CPU and on the GPU, deterministic and resumable.
- Its tests pass on Windows.
- `uv run --no-sync pytest` passes, except for failures you show are caused by other agents' files in progress.

## Report (short)

- The design as built, and the measured smoke time, with the projection of the official run.
- How the purge cuts' seeds were derived.
- The predicted-pattern rule of each control.
- Every point where the protocol needed an interpretation, and the one you took.
- One bolder next step.
