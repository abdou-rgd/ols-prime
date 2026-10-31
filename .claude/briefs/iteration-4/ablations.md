# Brief — LFM3's ablations and H6 (wave 3, "if time remains")

- **Agent:** `ml-engineer` (opus, high).
- **Why now:** the panel H1 and Report n°3's 54 fits hold the GPU for about twelve hours. The ablations are the last encoder item of the priority order, and Abdou set no time limit. Build them now so they can run when Report n°3's fits end.
- **Motto:** we should be more ambitious, in what we build and test, never in what we claim.

## Goal

Build the three arms of `docs/PROTOCOL.md`, "LFM3's ablations and H6, if time remains", and the runner that fits and scores them, exactly as registered. Read that section, the encoder sections above it, and the change-log entries of 2026-10-30 first.
- **`enc-ols-gate`:** `enc-ols` with a sigmoid gate per OLS head, O ← σ(x W_g) ⊙ O, where W_g maps the block's normalized input to the head's 8 outputs.
- **`enc-ols-conv`:** `enc-ols` whose first block's attention is a gated short convolution: a depthwise causal convolution of width 3 over the window, gated by a sigmoid. The second block keeps its OLS attention.
- **`enc-ols-brier`, H6:** `enc-ols` with the loss cross-entropy plus the Brier score, weight 1. H6 holds if both:
  - its ECE is at most 0.8 × `enc-ols`'s, and the paired stationary bootstrap's 95 % upper bound of the ratio of the two ECEs is below 1;
  - its log-loss exceeds `enc-ols`'s by no more than the larger of the range (max − min) of `enc-ols`'s three per-seed log-losses and the HLN-corrected Diebold-Mariano standard error of the difference.
- The LFM3 arms are described against `enc-ols`, with no registered rule. Each ablation that runs counts as one variant.
- Everything else is `enc-ols`'s: the windows, the memory over learned keys, the episodes, the six yearly fits 2019–2024, three seeds (1, 2, 3) averaged before the floor, the batches of 4,096, the deterministic settings, the RTX 3060 in WSL2.

## Constraints

- **`enc-ols` must not move by a bit.** Report n°3 is being fitted right now from a clean clone at `5f17a76`, and `explain head` recomputes its forecasts to the bit with the code of the next commits. Every default code path, every saved config and every honest arm's output stays byte-identical. Prove it on the GPU as the panel H1's builder did for its hook: fit `enc-ols`, `enc-softmax` and `cheat-stamp` on a synthetic panel with the old and the new code, and compare every array and the metadata.
- **No new fit of `enc-ols`.** The ablations are scored against Report n°3's `enc-ols`: its per-seed forecasts and its ensemble, read from Report n°3's run folder and bound by digest to its `report3.json` and `series.npz`, the way `explain head` binds a part (`python/olsprime/explain/head.py`, `scored_record`). Its per-seed log-losses come from the same record.
- **Report n°3's runner, as committed, is the registered one.** Prefer a separate module, for example `python/olsprime/bench/ablations.py`, that reuses `report3`'s fitting and scoring functions, writes its own run folder (`runs/<UTC date>-ablations` by default), resumes like `report3`, and has its own `--check`. If a change to `report3.py` is unavoidable, it must leave Report n°3's outputs byte-identical, and you say why.
- **Statistics.** Reuse `eval/stats.py` and Report n°2's conventions: hourly aggregation before inference, Diebold-Mariano with 23 lags and HLN, the stationary bootstrap with its registered seed and block length rule. The ECE is `stats.ece` on the ensemble's floored forecasts over the scored rows. For the paired bootstrap of the ECE ratio, resample hours jointly for both arms; the ECE of a resample is a function of per-bin sums (count, sum of confidences, sum of hits), so per-hour bin sums make each replicate cheap and exact. Report the ratio, its interval and both ECEs.
- **The parameter counts** of the three arms, exactly, against `enc-ols`'s 21,003.
- **Readings.** The section is frozen: the protocol changes only for what cannot be built as written. Where the text leaves a choice open (the gate's bias, the convolution's padding and its gate's input, the Brier term's scale over three classes, anything else), pick the reading closest to the text and to LFM3's description, and list it in your report with its reason. The orchestrator writes them into the change log before any run on real data.
- **Your files.** `python/olsprime/models/encoder.py`, `python/olsprime/train/fit.py`, a new `python/olsprime/bench/ablations.py`, and their tests (`test_encoder*.py`, `test_train.py`, a new `test_ablations.py`). Do not touch `python/olsprime/explain/`, `python/tests/test_explain_head.py` or `python/tests/test_torch_free.py`: another engineer is finishing them. If you need a line in one of them, tell me.
- **Data.** Synthetic data only (`--synthetic` and the test fixtures). Never read the real panel, `data/`, `runs/` or the vault. Never set `OLSPRIME_OPEN_VAULT`.
- **Your own worktree.** You work in a git worktree of commit `ece1950`, apart from the main tree, where `explain head` is being finished and audited. Build your own environments from it, never the shared ones, whose editable install points at the main tree:
  - Windows: `uv sync --locked` in the worktree's `python/` (it builds the Rust extension);
  - WSL2: `UV_PROJECT_ENVIRONMENT=~/olsprime-abl-gpu/venv CARGO_TARGET_DIR=~/olsprime-abl-gpu/target uv sync --locked --no-default-groups --group dev --group cu134` in the worktree's `python/`. Check that `import olsprime` resolves inside your worktree before you trust a result.
  - Never run anything from `~/ols-prime-official` or `~/ols-prime-r3`, the clean clones of the official runs.
- **GPU.** The official runs share the 3060. Keep GPU smokes few and short.
- **Process.** Write scripts with a file tool, never through a heredoc or a pipe into `python`. Run them with `uv run --no-sync python -I <file>`. No git command that changes state. Do not kill processes you did not start.

## Done when

- The three arms build and train on a synthetic panel, on CPU and on the GPU, deterministically: two fits of each are bit-identical.
- `enc-ols`, `enc-softmax` and `cheat-stamp` are byte-identical to the old code on the GPU smoke.
- The runner fits the ablations, scores H6 by its registered rule and describes the two LFM3 arms against `enc-ols`, all from Report n°3's run folder by digest, on a synthetic Report n°3 run made with `report3 --synthetic`. `--check` reproduces its files byte for byte.
- Tests pass: yours, `test_report3.py`, `test_encoder_h1.py`, `test_train.py`, and the full suite on Windows.

## Report

- What you built, file by file.
- The readings you chose, each with its reason.
- The parameter counts, and the synthetic GPU step times of the three arms, with a projection of their 54 fits on the real panel from the sizing in `docs/ITERATIONS.md`.
- The exact official command.
- One bolder next step.
