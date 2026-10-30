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

Commands run from `python/` with `uv run --no-sync`, on Windows unless marked WSL2.

**Clean clones (added on 2026-10-30, before the first run).** Official runs never run from the working tree, where engineers' files in progress would enter the code digest and an edit would block a resume. They run from clean clones of a pushed commit, with their own environments, writing to the main repository's `runs/`:
- **Windows:** `C:\Users\abdou\ols-prime-official`, its own `python/.venv` (`uv sync --locked`), with `OLSPRIME_DATA_DIR=C:/Users/abdou/ols-prime/data` and `OLSPRIME_RUNS_DIR=C:/Users/abdou/ols-prime/runs`.
- **WSL2:** `~/ols-prime-official`, with `UV_PROJECT_ENVIRONMENT=~/ols-official-gpu/venv CARGO_TARGET_DIR=~/ols-official-gpu/target` (synced with `--no-default-groups --group dev --group cu134`), `OLSPRIME_DATA_DIR=/mnt/c/Users/abdou/ols-prime/data`, `OLSPRIME_RUNS_DIR=/mnt/c/Users/abdou/ols-prime/runs` and `OLSPRIME_VAULT_DIR=/mnt/c/Users/abdou/ols-prime-vault`.
- The code digest hashes raw bytes. The Windows clone checks files out with CRLF and the WSL2 clone with LF, so their digests differ at the same commit. Every stage of one report therefore runs on one side: Report n°3 entirely in WSL2.
- A clone is moved to a new commit only between runs, never while one of its runs may still resume.

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
**When (fixed on 2026-10-30).** The protocol freezes the iteration 4 section at the first `explain` call on 2019–2024 hours: after it, the section changes only for what cannot be built as written. So the five whys run only once the change-log entry that fixes the readings of the panel H1, Report n°3 and `head` is committed, in practice during the GPU H1.

Each call is logged automatically. Record the line count of `CONTACTS.log` in the change log. Write each call's text to `runs/2026-10-30-explain/<n>-<name>.txt` and its JSON lines with `--jsonl runs/2026-10-30-explain/<n>-<name>.jsonl`.
1. **Why do `matrix-1y` and `knn-1y` tie?**
   - `series matrix-1y knn-1y --period 2019-01:2024-12 --test-months`;
   - `why` at the two hours of largest positive d and the two of largest negative d, for both arms. The row at each hour is BTCUSDT's, a rule fixed before any output is read, since `series` gives hours, not rows.
2. **Do the 18 previewed months' skill come from a few hours or from the regime?**
   - `series matrix-select --period 2019-01:2020-06 --test-months`;
   - beside it, `series matrix-select --period 2020-07:2024-12 --test-months`.
3. **Are hours with a high negative mass worse calibrated?** `sample matrix-select --period 2019-01:2024-12 --n 1000 --seed 0 --test-months`. The answer is exploratory, test months.
4. **Which feature carries the gain over climatology?** The per-feature parts of `sample`, or of `why` at sampled rows. The answer is exploratory, test months.
5. **What did the model do around the 2020-06-28 outage?** `why` for `matrix-select` on BTCUSDT and ETHUSDT rows from 2020-06-27 to 2020-07-01, with `--at` and `--test-months`. Hours 2020-06-28T01 → 2020-07-01T22 are unscored (the outage and the 90-bar refill), and so are 2020-06-27's first three hours, whose exit bars fall in the outage. So the rows are fixed now, by their bars' opens: 2020-06-27 at 03, 09, 15 and 21 h, and 2020-07-01T23, for both pairs.

### 5. The encoder (audit passes 4e and 4f: GO; from the WSL2 clone)
1. The panel H1 on the GPU: `python -m olsprime.bench.encoder_h1 --device cuda --stage all --driver 617.42 --out <runs>/2026-10-30-encoder-h1`. The pre-check runs first and gates the H1 stage. Pass the same `--out` on any restart: the default folder carries the UTC date.
2. Report n°3's fits, arm by arm: `enc-ols`, `enc-softmax`, `enc-linear`, 3 seeds × 6 fits each.
3. On a quiet machine (no Kani, nothing heavy on Windows): H13's timing and the Matrix timing, both in WSL2; then the score stage and `--check`.
4. If time remains: LFM3's ablations, then H6.

### 6. Kani
The five proven harnesses, in WSL2 with `-j 2`, from the WSL2 clone, before the H13 timing. Use the flags of `KANI_FLAGS` in `.github/workflows/kani.yml`, never `--bitwuzla`.
