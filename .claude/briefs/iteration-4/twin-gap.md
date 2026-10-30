# Brief — why did `explain`'s Matrix-free twin fail at one row?

- **Agent:** `ml-engineer` (opus, high). Diagnosis first: no change to repository code.
- **Motto:** we should be more ambitious, in what we build and test, never in what we claim.

## What happened

The third of the five whys ran, from the clean Windows clone at `5f17a76`:

```
python -m olsprime.explain sample matrix-select --period 2019-01:2024-12 --n 1000 --seed 0 --run C:\Users\abdou\ols-prime\runs\2026-10-30-report2 --sha256 03916fda8ffa9f97709216b2365e7a224dd8c8d71448d75035a7824fdc9cc0b9 --test-months --jsonl ...
```

It stopped on a loud check after about 20 minutes (the rows run in time order, so nearly all of the 1,000 rows had passed):

```
LOUD CHECK FAILED, the call stops and prints no result: the Matrix-free twin at SUIUSDT 2024-12-26T10:59:59.999+00:00, window all: the twin's fwd_return is np.float64(0.0012707678705418557), the core's np.float64(0.0012707731610964704): a gap of 5.29e-09, beyond its bound 2.67e-09
```

For comparison, the fifth why ran `why` for `matrix-select` at ten rows of 2020-06-27 and 2020-07-01 (window all, ρ 0.01, about 438,000 entries). There the twin's gap was about 2 × 10⁻¹⁵ against bounds of about 2 × 10⁻⁹ (`runs/2026-10-30-explain/5-why-outage.txt`). A gap of 5.3 × 10⁻⁹ is six orders of magnitude above that, so it looks like more than rounding.

Some facts from the code:
- The twin is `Explainer.twin` and its bound `twin_bound`, in `python/olsprime/explain/why.py`. It rebuilds the ridge from panel rows in numpy; the count of rows must equal Matrix's window count, and it did.
- Matrix's entries are pushed in `python/olsprime/models/matrix.py` (around line 520) with `ys[:, 3] = d.fwd[entry_rows]`, the same array the twin reads.
- The error message shows only the output with the largest gap minus bound. The class outputs' gaps at that row are unknown.
- A gap of 5.3 × 10⁻⁹ on `fwd_return` over about 1.6 million entries is what one entry's `fwd_return` differing by about 0.008 would give, or a relative error of about 10⁻⁷ in β times a large x̃. Both are guesses.

## Goal

Find out which side is wrong, by how much, and why. In order:
1. **Reproduce** with `explain why SUIUSDT 2024-12-26T10:00:00+00:00 --arm matrix-select --run ... --sha256 ... --test-months` (the bar opens at 10:00; the decision is at 10:59:59.999). Every `explain` call writes its line to `runs/explain/CONTACTS.log` by itself.
2. **A reference.** In a probe, rebuild the cell at that row: the core's raw readout, the twin's, and an exact or high-precision reference. For example, error-free products and `math.fsum` for the centred moments, then an exact solve with `fractions`. Give every output's gap and bound, for both sides against the reference.
3. **Localize.** Bisect over `as_of` in late December 2024 to find when the gap appears. Look at the entries that join in that interval, the query's x̃ at that row, ρ and σ², the condition of the penalised matrix, and the tree's nodes that cover the window.
4. **Judge the scope.**
   - Can the same cause move any forecast that Report n°2 scored, the class probabilities above all, beyond rounding?
   - Does the bound miss a term, or does the core compute something the twin does not?
   - Is `fwd_return`'s readout, which H14 will use in iteration 5, accurate?

## Constraints

- **No change to repository code**, in the main tree or any clone. Probes go in your scratch folder: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\twin-gap\`.
- **Environment.** Use the clean Windows clone at `5f17a76` read-only, with its own environment: `cd C:\Users\abdou\ols-prime-official\python` and `uv run --no-sync ...`. Set `OLSPRIME_DATA_DIR=C:/Users/abdou/ols-prime/data` and `OLSPRIME_RUNS_DIR=C:/Users/abdou/ols-prime/runs`, as the official runs do. Write nothing inside the clone. The main tree has another engineer's uncommitted work, so do not run from it.
- **Data.** The development panel (2017-10 → 2024-12) is yours to read, but every read is a contact with the real data. List each probe that reads the panel in your report, with its UTC time and the months it touched, so that the orchestrator can record it. Never read the vault, never set `OLSPRIME_OPEN_VAULT`, never write `runs/vault/OPEN`.
- **The machine is shared** with the official runs: the panel H1 and Report n°2 in WSL2, and the power study on Windows. Use one thread for BLAS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), stay under 4 GB of memory, and stop any probe that runs past 30 minutes.
- **Process.** Write scripts with a file tool, never through a heredoc or a pipe into `python`. Run them with `uv run --no-sync python -I <file>`. No git command that changes state. Do not kill processes you did not start.

## Report

- The cause, with the probe that shows it, and which side is wrong.
- Every output's gap at the row against the reference, for the core and the twin.
- The scope: what else it can move, and whether any registered result could change.
- The fix you propose, and where: the bound, the twin, the core or the arm. Do not apply it.
- Your contacts with the real data, each with its time and months.
- One bolder next step.
