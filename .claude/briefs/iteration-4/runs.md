# Brief — the official runs of iteration 4 (wave 3)

- **Who:** the orchestrator. `experiment-runner` may take the long GPU stages.
- **When:** each run starts only after:
  - its code passed its audit;
  - its code is committed;
  - CI is green on every platform for that commit.
- **Rules:**
  - Every run on the real data is a contact. It is listed in `docs/PROTOCOL.md`'s change log as it happens, with its months.
  - Every `explain` call also writes its line to `runs/explain/CONTACTS.log`, which is committed with the next commit.
  - Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.
  - Run each command with a timeout, in the background, logging to the run folder.

Commands run from `python/` with `uv run --no-sync`, on Windows unless marked WSL2. The CUDA environment in WSL2 is:
```
UV_PROJECT_ENVIRONMENT=~/olsprime-gpu/venv CARGO_TARGET_DIR=~/olsprime-gpu/target
```
with the repository at `/mnt/c/Users/abdou/ols-prime/python`.

## Order

### 1. Report n°2's re-run (audit pass 4c: GO)
1. `python -m olsprime.bench.report2 --compare-with ../runs/2026-10-29-report2 --out ../runs/2026-10-30-report2`
   - Expect exit 0, and all five allowances of `ITERATION3_ALLOWANCES` reported `[met]`: commit, code digest, series, coverage sentence, H12's rule in the parts.
   - Anything else differing is a finding. Stop and report it.
2. `python -m olsprime.bench.report2 --check ../runs/2026-10-30-report2`
   - Expect all five files byte-identical, `series.npz` included.
3. Read the SHA-256 of `series.npz` from `runs/2026-10-30-report2/report2.json`, under `series.sha256`.

### 2. The power study (audit pass 4c: GO with fixes; run only once the fixes are committed and CI is green)
Use the official command the engineer gives after the fixes, with a pool, by the digest of step 1.3. Then re-check: a second run must give byte-identical `power.json` and `power.md`.

### 3. The twins (audit pass 4c: GO)
```
python -m olsprime.bench.twins --adversary ../runs/2026-10-29-h1-adversary --sha256 f572127b1eb1dbfcde30fa285976e6431192a25e82e0179b6074805c27325f54 --workers 8
```
Re-run it into a second folder: `twins.json` and `twins.md` must be byte-identical.

### 4. The five whys, with `olsprime.explain` (audit pass 4d: GO)
Each call is logged automatically. Record the line count of `CONTACTS.log` in the change log.
1. **Why do `matrix-1y` and `knn-1y` tie?**
   - `series matrix-1y knn-1y --period 2019-01:2024-12 --test-months`;
   - `why` at the hours where |d| is largest, of each sign, for both arms.
2. **Do the 18 previewed months' skill come from a few hours or from the regime?**
   - `series matrix-select --period 2019-01:2020-06 --test-months`;
   - beside it, `series matrix-select --period 2020-07:2024-12 --test-months`.
3. **Are hours with a high negative mass worse calibrated?** `sample matrix-select --period 2019-01:2024-12 --n 1000 --seed 0 --test-months`. The answer is exploratory, test months.
4. **Which feature carries the gain over climatology?** The per-feature parts of `sample`, or of `why` at sampled rows. The answer is exploratory, test months.
5. **What did the model do around the 2020-06-28 outage?** `why` for `matrix-select` on BTCUSDT and ETHUSDT rows from 2020-06-27 to 2020-07-01, with `--at` and `--test-months`.

### 5. The encoder (after its audit)
1. The panel H1 on the GPU in WSL2: its pre-check first, then H1 with every control, `pos-window` included.
2. Report n°3's fits, arm by arm: `enc-ols`, `enc-softmax`, `enc-linear`, 3 seeds × 6 fits each.
3. H13's timing, the cost table, the score stage, and `--check`.
4. If time remains: LFM3's ablations, then H6.

### 6. Kani
The five proven harnesses, in WSL2 with `-j 2`, during the long GPU fits. Use the flags of `KANI_FLAGS` in `.github/workflows/kani.yml`, never `--bitwuzla`.
