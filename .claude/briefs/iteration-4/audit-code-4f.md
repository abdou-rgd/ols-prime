# Brief — audit pass 4f: Report n°3's runner and scoring, before the 54 fits on real data

- **Agent:** `leakage-auditor` (fable, high). Read-only.
- **Runs in parallel with:**
  - the `ml-engineer` finishing the panel H1 for the encoder (`bench/encoder_h1.py`);
  - later, a second auditor on the encoder's core and that H1 (pass 4e);
  - the orchestrator's official CPU runs of Report n°2, the twins and the power study, which read none of this code.
- **Motto:** we should be more ambitious, in how hard we attack, never in what we claim we found.

## Goal

Report n°3 gives H4, H5 and H13's verdicts and the cost table. Its runner trains 54 fits on the real panel (3 arms × 3 seeds × 6 yearly fits, 2019 to 2024, on the RTX 3060 in WSL2), forecasts every scored hour of 2019–2024, pairs them with Report n°2's rows and scores them. None of it has touched real data. Break it before it does.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)": the encoder, the fit, the seeds, the batches, "H4, H5 and H13" and the cost table; then "Matrix on hourly bars (iteration 3)", whose Scoring and Decision rules Report n°3 reuses;
- the change-log entries "2026-10-30, wave 1" and "2026-10-30, before the first `explain` call";
- the brief `.claude/briefs/iteration-4/encoder-report3.md`.

Then the code, in the working tree, uncommitted:
- `python/olsprime/bench/report3.py` and `python/tests/test_report3.py`, both new;
- what it calls in the encoder's core, committed in 221591c: `python/olsprime/train/fit.py` (with `git diff` for its new read-only `on_readout` hook), `python/olsprime/models/encoder.py`, `python/olsprime/models/memory.py`. The core itself is audited in pass 4e; read it here only as far as Report n°3 depends on it.

## The engineer's interpretations, to accept or reject

1. Timing uses each arm's seed-1 fit of the last year.
2. The 12,288 timed rows are drawn with seed 20261030 from the scored rows that have a valid window. The Matrix stage uses the same sample, recorded by its digest.
3. The memory readout is timed on the CPU in fp64, as `predict` does it.
4. The arms are timed in the order ols, softmax, linear in each repeat, so ols and softmax alternate as ABAB.
5. H13's ratio is median(ols) / median(softmax). The attention operator's own time is measured separately, on cached Q, K and V.
6. Matrix's time per decision is each arm's own readout at the last fit's choices; `matrix-adaptive` is timed at `matrix-select`'s ρ.
7. GPU and CPU columns are kept apart.
8. On Windows the CPU peak memory is recorded as None: psutil is absent and `ctypes` is banned by the guard. On Linux it is `ru_maxrss`, one peak for the whole process. The orchestrator plans to run the Matrix timing stage in WSL2, on the same Ryzen 5 3600 with the portable build, so that the column is filled.
9. Matrix's parameters are recorded as 0.
10. The statistics' parameters (bootstrap, level, DM lags, ECE bins) come from Report n°2's own `config.json`.
11. The panel is checked by its fingerprint, and every scored row is matched one for one, before any training.
12. `score` refuses only on a config or series mismatch. A code change surfaces through `--check`.

## What to attack, at least

1. **Look-ahead in the fits and the forecasts.**
   - Does each yearly fit train only on rows whose labels are known by F − 24 h, F being the first hour of its test year? Does its slice stay inside the training period?
   - Does each fit forecast only its own test year's rows? Can a row be forecast by a fit that saw its label?
   - At test time, does the memory read only entries whose `label_known_at` is at or before the row's decision time?
   - Do the standardization, the climatology and a failed window's fallback come from the training rows only?
2. **Pairing with Report n°2.**
   - Are the forecasts paired with Report n°2's scored rows exactly, by symbol code and decision time, and refused on a missing, extra, shifted, relabelled or doubled row?
   - Is `matrix-select`'s series read from `series.npz` by its digest, on the same scored hours?
   - Could a row scored by Report n°2 but without a valid encoder window be silently dropped, or scored in a way that favours an arm?
3. **The ensemble and the floor.** Is the mean of the three seeds' probabilities taken before the floor, with Report n°2's floor? Are failed windows counted and reported per arm?
4. **H4.**
   - Is the margin 0.005 × the mean hourly log-loss of Report n°2's `climatology-panel` over the scored hours?
   - Are both bounds as registered: the one-sided 95 % upper bound with t₀.₉₅(T − 1) and the HLN-corrected DM standard error, as for H12, and the stationary bootstrap's 95th percentile?
   - Is "not testable" applied exactly as written: neither `enc-ols` nor `enc-softmax` with a Brier skill score whose 95 % interval lies above 0? Does it carry over to H13?
5. **H5.** Diebold-Mariano p < 0.05 and the bootstrap interval below 0, against `matrix-select`, on the same hours. Is the limit about cadence stated?
6. **H13 and its timing protocol.** 3 untimed warm-up batches, 3 timed batches of 4,096, `torch.cuda.synchronize()` before and after each, 10 repeats alternating the arms, the median per 10⁴ decisions, fp32 eager, the math backend for softmax, the deterministic settings on. Does H13 require H4? Is the operator's own time reported only as a description? Does interpretation 3 bias the ratio, and in which direction?
7. **The cost table.** The five columns as registered. Check the analytic FLOPs formulas against the code of each arm, the parameter counts against the change log (20,995 for `enc-softmax` and `enc-linear`, 21,003 for `enc-ols`), and that no column mixes machines or builds.
8. **Reproducibility.** Resume by digest; refusal on a changed config or code digest; `--check` re-scoring from the saved forecasts byte for byte; determinism of `series.npz` and the reports.
9. **The guards.** Does report3 read the real data only through `load_klines` and the seal, and Report n°2's outputs only by digest? Do checkpoints avoid pickle on both write and read?

## Constraints

- **Read-only.** Write probes only as files in your scratch folder, `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\audit-4f\`, with a quoted heredoc if you must. Run them with `uv run --no-sync python -I <file>`.
- **Environments.** On Windows, a venv under your scratch: set `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` there, then `uv sync --locked`. Never sync the shared `python/.venv`. In WSL2 you may run short synthetic GPU probes in `~/olsprime-gpu/venv` (`UV_PROJECT_ENVIRONMENT=~/olsprime-gpu/venv CARGO_TARGET_DIR=~/olsprime-gpu/target`). Do not rebuild it, and do not touch `~/ols-prime-official` or `~/ols-official-gpu`: they are the orchestrator's clean clone for the official runs.
- **No real data.** Nothing on `data/`, the real store or the vault. You may read Report n°2's run outputs under `runs/` (`config.json`, `report2.json`, `report-2.md`), never `data/`.
- **Processes.** Give every long command a timeout. You cannot kill a hung process here, and nobody is awake to do it.
- **Git.** No command that changes state.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Report

- GO or NO-GO for Report n°3's 54 fits and its score on real data.
- Findings by severity (Blocker, Major, Minor, Nit), each with file:line, a concrete failure scenario and a fix.
- Each of the twelve interpretations: accepted or rejected, with the reason.
- One bolder check: the cheapest probe that would most change our confidence in H4's verdict.
