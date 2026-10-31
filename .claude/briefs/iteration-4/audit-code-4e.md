# Brief — audit pass 4e: the encoder's core and its panel H1, before any encoder run on real data

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with** the orchestrator's official CPU runs of Report n°2, the twins and the power study, which read none of the code below, and with audit pass 4f, another `leakage-auditor` on Report n°3's runner, `python/olsprime/bench/report3.py`. Leave that runner to pass 4f.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

The encoder has run on synthetic data only. Next, the orchestrator runs, on the real sub-panel and on the GPU in WSL2:
1. the panel H1 for the encoder arms, with its pre-check, five planted training-time leaks and the positive control `pos-window`;
2. then Report n°3's 54 fits, which use the same core.

Every number of H4, H5 and H13 rests on this code having no look-ahead and on H1 being able to see one if it had. Break both before they touch the real data.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)":
  - the encoder;
  - H1 for the encoder arms;
  - the purge cuts;
  - the planted training-time leaks, `cheat-stamp-day` included;
  - the three verdicts;
  - the positive control `pos-window`;
- the change-log entries "2026-10-30, wave 1" (the build's readings and the sizing) and "2026-10-30, before the first `explain` call" (`cheat-stamp-day` and the three verdicts);
- "Leakage tests", and the panel H1 of iteration 3, whose machinery is reused;
- the briefs `.claude/briefs/iteration-4/encoder-core.md` and `encoder-h1.md`.

Then the code:
- **The core, committed in 221591c:**
  - `python/olsprime/models/encoder.py`, `python/olsprime/models/memory.py`;
  - `python/olsprime/train/`: `fit.py`, `determinism.py`, `checkpoint.py`, `synthetic.py`;
  - its tests `python/tests/test_encoder.py`, `test_memory.py`, `test_train.py`.
- **The panel H1, in the working tree, uncommitted:**
  - `python/olsprime/bench/encoder_h1.py` and `python/tests/test_encoder_h1.py`, both new;
  - `git diff -- python/olsprime/train/fit.py`, a read-only hook added for it.
- **The guards:**
  - the torch rows of `python/tests/test_protocol.py`;
  - the allow-list entries of `python/tests/allowlist.py` for `tests/test_torch_guards.py` and `tests/test_torch_free.py`;
  - `python/tests/test_torch_guards.py` and `test_torch_free.py`.

## The engineer's interpretations, to accept or reject

The `ml-engineer` who built `encoder_h1.py` took these readings:
1. **The memory's "stamps and statistics"**, per compared decision: the window's range in the stamp-sorted entries; those entries' stamp, symbol and open time; the raw window moments (n, Σz, Σzzᵀ, Σy, Σyzᵀ); and the readout. They are read through a public no-op hook, `EncoderSystem.on_readout(prefix, lo, hi)`, called in `predict` before `_readout_fixed`. Honest outputs were re-checked byte-identical on the GPU after the hook was added.
2. **Inputs at the 8 cuts and at F − 24 h:** rows compared as (open time, symbol). "The order of the episodes" is the Monday stamps plus each epoch's permutation. The step, parameter and no-window counts are compared too.
3. **Directions:** a control must be caught in each direction where predicted, and in none where not. This is stricter than iteration 3's union of directions.
4. **The pre-check's scope:** memory is compared only up to each year's last cut. The scope test runs at the first cut only, with the fit reloaded from its saved file.
5. **Three verdicts, apart:** H1 needs the honest arms bit for bit and every control on its pattern. The skill check, per control (`cheat-stamp`, `cheat-value`, `cheat-stamp-day`), and `pos-window`'s check are separate verdicts. Exit codes: 2 if the pre-check fails, 1 if H1 fails, 3 if H1 holds but a skill check or `pos-window` fails.
6. **The floor** is computed per system from its own three unperturbed runs.
7. **Additions beyond the brief:** `--stage floor`, `--synthetic`, `--driver` (no subprocess may call `nvidia-smi`), and budget options, built but off by default.
8. **The device** is part of the config. Both `cuda` and `cpu` count as registered.
9. **The default output folder** has no suffix, so a restart the same day resumes it. An edit under `olsprime/` blocks a resume. The orchestrator runs it from a clean clone.
10. **`pos-window`** is scored on the cut years only. Rows without a label of their own are dropped and counted.
11. **The purge cuts' seeds** are those of a 9th and 10th cut of the same schedule: k = 8 for F − 24 h and k = 9 for F − 30 days in `perturbation_at`, so seeds 15 and 1015, then 16 and 1016. The mirror follows the parity of k.
12. **The predicted-pattern rule**, one for every system, computed per cut from the original panel through the system's own hooks: a system is predicted caught at a cut when some compared item reads a label whose `label_known_at` is after the cut hour. It predicts the honest arms never caught; `cheat-stamp`, `cheat-value` and `cheat-stamp-day` caught at each of the 8 cuts and at neither purge cut; `cheat-purge` caught at both purge cuts; `cheat-slice` at F − 30 days only.
13. **`cheat-stamp-day`:** `StampedAtDecisionDay(StampedAtDecision)` with `Spec.memory_window_ms` = 24 h, passed as `memory=replace(MemoryConfig(), window_ms=24 h)` for that control alone.

**Synthetic facts it reported, for context.** On the GPU, the pre-check matched to the bit and every control was caught exactly where predicted. Brier skill scores on a synthetic year: honest `enc-ols` −0.0011 [−0.0022, 0.0003]; `cheat-stamp` −0.0009; `cheat-value` 1.000; `cheat-stamp-day` 0.074 [0.067, 0.081], under the 0.1 bar but above every honest arm's interval; `pos-window` 0.515 [0.495, 0.536]. Projection of the official run: 170 fits, about 220,100 steps, about 9.2 GPU-hours, an upper bound.

## What to attack, at least

1. **Look-ahead in the memory.**
   - At a decision at time d, does every entry the memory reads, in training and at test time, have `label_known_at` ≤ d? Look also within a weekly episode, across a fixed 4,096-row batch and its padding, and at the episode's edges.
   - Is the 30-day window taken by `label_known_at`, with the boundaries the protocol states?
   - Is the centred ridge (ρ = 10⁻², fp64) centred on the window's entries only?
   - Can padding or a masked row reach a key, a value, a centring mean or a loss?
2. **Look-ahead in the fit.**
   - Each yearly fit, 2019 to 2024: are its training rows the ones whose labels are known by F − 24 h, F being the first hour of the test year?
   - Is the slice, which picks the epoch, inside the training period, and does nothing of the test year reach it?
   - Are the standardization (population std) and the climatology fitted on the training rows only?
   - Are the seeds, the episode order, the cosine schedule and the ensembling (mean of three seeds before the floor) as registered?
3. **The arms.**
   - Is `enc-ols` the registered OLS attention: RMS-scaled Q and K, ε = 10⁻¹², O = Q̃((K̃ᵀK̃ + Λ)⁻¹(K̃ᵀV)) with one Cholesky factor?
   - Are `enc-softmax` (math backend) and `enc-linear` the registered arms? Do the parameter counts match the change log's: 20,995 for `enc-softmax` and `enc-linear`, 21,003 for `enc-ols`?
   - Does each 64-bar window end at its decision bar, so that no bar after the decision enters it?
4. **H1's power to see a leak.**
   - Does the comparison cover every item the protocol lists at the 8 cuts and at the two purge cuts, bit for bit for the honest arms and with the catch rule for the controls?
   - Could a leak move a quantity that is not compared, so that H1 passes while the forecasts use the future?
   - Is each planted control built as written: `cheat-stamp`, `cheat-value`, `cheat-purge`, `cheat-slice` and `cheat-stamp-day`, whose memory holds only (d − 24 h, d] with every entry stamped at its decision time?
   - Is each predicted pattern computed from the data, per cut, by the stated rule, and is the computed one the one tested?
   - Are the purge cuts' perturbation seeds derived as the engineer states, and is the perturbation strong enough to move every item after the cut?
5. **The pre-check and determinism.**
   - Two honest runs with identical inputs, and the scope test: does the pre-check compare everything H1 compares?
   - Does the GPU path set every switch determinism needs, and fail loudly when one is missing?
   - Resume: can a restart reuse a part computed with other code or another config?
6. **The three verdicts and the skill check.**
   - Is each verdict reported apart, passing or failing, with none standing in for another?
   - Is the Brier skill score against `climatology-panel`, and its 95 % interval, the machinery Report n°2 uses, on the stated rows?
   - Is `pos-window`'s label the registered one, an fp32 sequential sum of the window's last 24 one-bar returns, classed with the row's own threshold, and is it scored against its own label's climatology?
7. **The guards.**
   - Can the encoder code read anything but the sealed store through `load_klines`?
   - Do checkpoints avoid pickle on both write and read?
   - Is each new allow-list entry exact, justified and still used?
   - Is `olsprime.models` still importable without torch?
8. **The fit.py hook.** Is it read-only? Can it change a bit of training or of a forecast?

## Constraints

- **Read-only.** Run the tests and your own probes. Write probes only as files in your scratch folder, `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4e\`, with a quoted heredoc if you must. Run them with `uv run --no-sync python -I <file>`.
- **Environments.** On Windows, a venv under your scratch: set `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` there, then `uv sync --locked`. Never sync the shared `python/.venv`. The GPU in WSL2 may be shared with the Report n°3 engineer's synthetic smoke, so keep your GPU probes short. You may run synthetic GPU probes in `~/olsprime-gpu/venv`, with `UV_PROJECT_ENVIRONMENT=~/olsprime-gpu/venv CARGO_TARGET_DIR=~/olsprime-gpu/target`. Do not rebuild that venv.
- **No real data.** Nothing on `data/`, the real store or the vault.
- **Processes.** Give every long command a timeout. You cannot kill a hung process here, and nobody is awake to do it.
- **Git.** No command that changes state.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Report

- GO or NO-GO for the panel H1 on the real sub-panel, and separately for Report n°3's fits using this core.
- Findings by severity (Blocker, Major, Minor, Nit), each with file:line, a concrete failure scenario and a fix.
- Each of the thirteen interpretations above, and any other you find in the code: accepted or rejected, with the reason.
- One bolder check: the cheapest probe that would most change our confidence in H1's verdict.

## Second look (2026-10-30, after the fixes, before the GPU H1)

The engineer made three changes after this pass: the build in the resume identity (Minor 2), the scope test at every regular cut as a description (Minor 3 and the bolder check), and `cumsum`'s note in `memory.py` (a Nit). The orchestrator also wrote the change-log entry that fixes the readings before any contact: the last entry of `docs/PROTOCOL.md`'s change log.

Review the commit named in the call (`git diff c2e6381 <commit> -- <path>`), not the working tree, where another engineer's `explain head` is in progress.
1. **The build in the identity** (`identity_build`, `Parts`). Does it hold anything that changes between two processes of one build (a wall clock, a PID, free memory), which would refuse an honest resume? Does it hold everything that can change a bit of a fit on CUDA? Is each refusal named?
2. **The scope test at every cut** (`scope_record`, `scope_every_cut`, `scope_every_cut_summary`, the report's section 5, `plan`).
   - Can it change a byte of the registered pre-check, of H1's parts or of the three verdicts? Look for shared state: caches, `drop_cache`, random generators, the deterministic context, the order in which runs are trained or loaded.
   - `cheat-stamp`'s run 1 is now trained in the pre-check stage instead of the H1 stage. Is it the same bytes either way?
   - Does it compare what its description says, at the right rows and cut?
   - Do the two new tests fail when they should?
3. **The change-log entry.** Does each statement say what the code and your earlier reports say, no more? Flag every overclaim, every wrong number and every reading the code does not implement. The `explain head` readings are fixed before it is built, so check only that they are buildable and unambiguous.

Use your scratch venv from the first look. CPU tests only; the GPU is about to run the official H1.

Report: GO or NO-GO for the GPU H1 at that commit; findings by severity with file:line and a fix; one bolder check.
