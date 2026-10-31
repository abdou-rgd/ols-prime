# Brief — audit pass 4h: LFM3's ablations and H6, before any fit on real data

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with** the official runs on the GPU in WSL2: Report n°3's 54 fits and the panel H1. Neither reads the code below.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

The ablations engineer built the three arms of `docs/PROTOCOL.md`, "LFM3's ablations and H6, if time remains", and a runner that fits and scores them:
- `enc-ols-gate`, a sigmoid gate on each OLS head's output;
- `enc-ols-conv`, a gated short convolution in place of the first block's attention;
- `enc-ols-brier`, the loss cross-entropy plus the Brier score, for H6.

The ablations are scored against Report n°3's `enc-ols`, read from Report n°3's run folder by digest, with no new fit of `enc-ols`. They will be fitted on the real panel once Report n°3's fits end. Break them before that. Look for:
- an honest arm, a default code path or a saved config that moved by a bit;
- an ablation that saw the future;
- an H6 verdict not computed as registered;
- a comparison against a part of Report n°3 that is not the one it scored.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`:
  - "LFM3's ablations and H6, if time remains", and the encoder sections above it;
  - the change-log entries of 2026-10-30, above all "Report n°3, as built" and the entry after the first `explain` call;
- `.claude/briefs/iteration-4/ablations.md`, the engineer's brief;
- the engineer's readings, below;
- the files under review, uncommitted in the main tree on top of the last commit (`git diff HEAD`, `git status`):
  - `python/olsprime/models/encoder.py`;
  - `python/olsprime/train/__init__.py` and `python/olsprime/train/fit.py`;
  - `python/olsprime/bench/ablations.py` (new);
  - `python/tests/test_train.py` and the new tests;
- what they call: `python/olsprime/bench/report3.py` (`verified_part`, `load_fit`, `read_forecast`, the score stage), `python/olsprime/eval/stats.py` (ECE, Diebold-Mariano with HLN, the stationary bootstrap), and `python/olsprime/explain/head.py` (`scored_record`, the way a part is bound to `report3.json`).

## Two facts that shape the attack

- **Report n°3 is being fitted at `5f17a76`, and other code reads its parts at later commits.** `explain head` recomputes `enc-ols`'s forecasts to the bit from the checkpoints, with the code of later commits. It rebuilds the config from `report3.json`'s record and loads `EncoderModel`. Any change to `EncoderConfig`'s fields, to its JSON, to `FitConfig`, to the config digest or to `EncoderModel`'s parameter names would break that identity. Such a change could also make an old checkpoint load into a different model. Check this first.
- **No new fit of `enc-ols`.** The ablations reuse Report n°3's per-seed forecasts, its ensemble and its per-seed log-losses. A part that is not the one Report n°3 scored must be refused before any number is computed.

## The engineer's readings, to accept or reject with a reason

From the engineer's report:
1. **The gate.**
   - σ(x W_g) has no bias, as the formula writes it.
   - W_g maps d_model to d_model, and its slice h gates head h's 8 outputs.
   - x is LN1(x), the block's normalized input.
   - The gate acts before the output projection, since it is per head.
   - W_g takes weight decay, like any linear map's weight.
   - Its initialization is the default one, drawn after every `enc-ols` module.
2. **The convolution:** `out(σ(x W_g) ⊙ conv3(x W_u + b_u))`.
   - The value map has a bias. The kernel has none, since b_u already offsets each channel.
   - The gate has no bias, like the gate arm's. The output projection has a bias, like attention's.
   - The window is zero-padded on the left.
   - The kernel is drawn U(±1/√3), as `nn.Conv1d` draws a depthwise kernel, and takes no weight decay.
   - It is computed as shifted products, not with cuDNN, for determinism.
   - It needs n_layers ≥ 2.
3. **The Brier term.**
   - Σ_k (p_k − 1{y=k})² over the 3 classes, in [0, 2]: the form the scores use.
   - It is taken on the softmax before the floor, averaged over the episode's rows, with weight 1.
   - The epoch kept is still chosen by the slice's log-loss.
4. **The ECE-ratio bootstrap.**
   - Hours are resampled jointly, using per-hour bin sums from `r2.ece_bins`, cross-checked against `stats.ece`.
   - The block is the larger of the two arms' `_block`.
   - The seed is `_seed(rcfg, "ece_ratio", "enc-ols-brier", "enc-ols")`, and n_boot is Report n°2's.
5. **"The 95 % upper bound"** is the bootstrap's one-sided 95th percentile, as for H4's bound. The two-sided interval and both ECEs are reported beside it.
6. **The log-loss condition** is inclusive (≤). If the Diebold-Mariano test is undefined (the same difference at every hour), the tolerance is the seed range alone.
7. **Variants:** 21 before; one more for each ablation scored.
8. **The registered run.**
   - The arms must be a prefix of the protocol's order.
   - Every fit must run on CUDA in one build, equal to Report n°3's, driver included.
   - The fits cannot share Report n°3's code digest, since that code predates `ablations.py`. The binding is by the data digests: `series.npz`, `report3.json`, and every Report n°3 part's `fit_sha256`, `forecast_sha256`, `forecast_rows` and `kept_epoch`.
9. **Each ablation is built as `enc-ols` first, then changed.** So under one seed it starts from `enc-ols`'s initial weights for every module it shares with it, and draws the same episode order.
10. **No new field in `EncoderConfig` or `FitConfig`.** `enc-ols-brier` uses a subclass, `BrierFitConfig`, whose saved config alone carries `brier_weight`.

Parameter counts: `enc-ols` 21,003; `enc-ols-gate` 23,051 (+2,048, two 32 × 32 maps); `enc-ols-conv` 20,007 (−996); `enc-ols-brier` 21,003.

The engineer's evidence, on CPU only:
- Old code (`ece1950`) against new: the checkpoints and forecasts of `enc-ols`, `enc-softmax`, `enc-linear`, `cheat-stamp` and `cheat-value` are byte-identical, 11 files of 11.
- Two fits of each ablation are identical.
- A synthetic Report n°3, then the ablations, then `--check`: the check passed.
- The full Windows suite: 1,519 passed, 4 skipped.

The GPU smoke runs in parallel with your audit; its log is `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\gpu_smoke.log`. Its script is `scratchpad\gpu_smoke.sh`, and its probe `scratchpad\identity_probe.py`.

The engineer's prior on H6, an analysis and not a result: near uniform forecasts, the Brier term's gradient in the logits is 2/3 of the cross-entropy's, in the same direction. AdamW is nearly invariant to gradient scale, so CE + Brier ≈ CE at first order, and H6 is expected to be close to a null. Check this reasoning.

## What to attack

1. **Byte identity.**
   - Every default path, the honest arms (`enc-ols`, `enc-softmax`, `enc-linear`), the controls (`cheat-stamp` and the others), `pos-window`, `config_json` and the digests must stay byte-identical to `ece1950`'s.
   - Read the engineer's GPU smoke, old code against new. Re-run its CPU equivalent if memory allows.
   - Check that the H1's and Report n°3's tests still pass unchanged.
2. **The arms, as registered.**
   - The gate: O ← σ(x W_g) ⊙ O per OLS head, with W_g mapping the block's normalized input to the head's 8 outputs.
   - The convolution: depthwise, causal, width 3, gated by a sigmoid, in the first block only. The second block keeps its OLS attention.
   - The Brier term: weight 1, added to the cross-entropy.
   - Give the parameter count of each arm, exactly.
   - Can any of them read a bar after the decision, or a label not yet known? Check the episodes, the cache keyed without gradient, the purge and the early-stopping slice.
3. **H6's rule, as registered.** Both conditions must hold:
   - the ECE condition: ECE ≤ 0.8 × `enc-ols`'s, and the paired stationary bootstrap's 95 % upper bound of the ratio below 1;
   - the log-loss condition: the difference no larger than the larger of two numbers, the range of `enc-ols`'s three per-seed log-losses and the HLN-corrected Diebold-Mariano standard error.

   Check the following:
   - The bootstrap resamples hours jointly for both arms, with the registered seed and the block-length rule. Each replicate's ECE is rebuilt exactly from per-hour bin sums. Plant a one-row change and see the ratio move.
   - The ECE is `stats.ece`, on the ensemble's floored forecasts, over the scored rows.
   - The per-seed log-losses come from Report n°3's record.
   - Can a degenerate case pass vacuously: both arms at climatology, or an ECE of 0?
4. **Binding to Report n°3.** A part copied from another arm, seed or year; a forecast cut or re-sealed; a `report3.json` or `series.npz` whose digest differs. Each must stop the runner before it reads a number.
5. **The runner.** Its own folder, resume, `--check` byte for byte, sorted JSON with no wall clock, no pickle, and `numpy.load` banned.
6. **The tests.** Does each one fail when the defect it names is planted?

## Constraints

- **Read-only.** Run the tests and your own probes. Write probes only as files in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4h\`, with a file tool. Run them with `uv run --no-sync python -I <file>`.
- **Environments.** On Windows, use a venv under your scratch folder: set `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` there, then run `uv sync --locked`. Never sync the shared `python/.venv`.
- **CPU only.** The GPU runs the official fits. Do not touch the WSL2 clones `~/ols-prime-official` and `~/ols-prime-r3`, or their environments.
- **Memory.** Windows' commit charge is short today. Before any run that trains, check `(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory`. Run only with at least 4 GB free, with `OMP_NUM_THREADS=2`. Otherwise read the code, and say so.
- **No real data.** Nothing on `data/`, `runs/`, the real store or the vault. Never set `OLSPRIME_OPEN_VAULT`, and never write `runs/vault/OPEN`.
- **Processes.** Give every long command a timeout. You cannot kill a hung process here.
- **Git.** No command that changes state.

## Report

- GO or NO-GO for the ablations' fits on the real panel.
- Findings by severity (Blocker, Major, Minor, Nit). Give each its file:line, a concrete failure scenario and a fix.
- Each of the engineer's readings: accepted or rejected, with the reason.
- One bolder check.
