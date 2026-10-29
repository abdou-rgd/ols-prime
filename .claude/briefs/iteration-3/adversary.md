# Brief — An adversary for H1 (iteration 3)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:** a `rust-engineer` on `crates/` (`matrix-core.md`) and a `data-engineer` on the hourly panel (`panel.md`).
- **Motto:** we should be more ambitious, in how hard we attack our own bench, never in what we claim it catches.

## Goal

H1 catches the leaks someone thought of: 27 hand-made controls. Iteration 2's audits showed that each new hand-made leak found a new blind spot. Turn the hand-picked list into a measured detection rate:
- a seeded generator plants thousands of random leaks;
- each one runs through H1's own perturbation;
- the report gives the share caught, with an interval, the size of each catch, and the misses;
- the misses show where the next designed run should go.

Read first:
- `docs/PROTOCOL.md`, "Leakage tests", including the new paragraph "The adversary (iteration 3)". That paragraph is the pre-registration. Do not deviate from it without asking.
- `docs/ITERATIONS.md`, iteration 2's outcome and lessons.
- `python/olsprime/bench/invariance.py`, `perturb.py`, `leaks.py`, `pipeline.py` and `synthetic.py`.

## Deliverables

1. **`python/olsprime/bench/adversary.py`.** A module docstring that defines the generator completely: a reader must be able to say what population the reported share describes.
   - **Leaks.** A leak is a deterministic function of a pair's bars that gives, for every row s, one output: `exp_return` (continuous), or `target` in {0, 1}, or three probabilities. It reads at least one bar after s.
   - **Families, sampled with fixed weights:**
     1. levels: a quantity of bar s + j (j ∈ {1, 2, 3, 5}) compared with the same quantity at s − k, for k from 0 to 180, beyond H1's 90-bar window too. Or compared with another future bar;
     2. ratios inside a future bar (VWAP, taker VWAP, trade size, taker shares, body, wicks, range), compared with the past or with a threshold;
     3. shapes: fill, close location, colour, wick ratio, hammer, doji, the colour × size cells, against thresholds inside **and** outside H1's bands;
     4. aggregates over the next m bars (max, min, mean, sum) compared with the past;
     5. continuous functions, such as future log-returns or a future quantity over a trailing mean, used as `exp_return` (these should always be caught);
     6. combinations: AND, OR or XOR of two leaks, or a weighted sum.

     Use every quantity that `perturb.py` moves, including `trades`, the volumes and both taker shares.
   - **Honest twins.** Every leak has a twin: the same expression with every future offset moved into the past (s + j becomes s − j + 1). It reads bars ≤ s only.
   - **Evaluation.** For each pair, cut and direction, compare each leak's outputs on the original bars and on `perturb_after`'s bars, at every row whose decision is at or before the cut. Use H1's catch rule: an output moves by more than `CAUGHT_ABOVE`, or turns missing, or stops being missing.
     - Use H1's cuts and seeds exactly, 8 cuts per pair on BTCUSDT and ETHUSDT 1d, both directions. Then a longer seeded schedule (64 cuts per pair) for per-cut rates.
     - Compute the perturbations once and evaluate every leak on them, vectorized. A leak is a function of the bars, with no fit, so it needs no pipeline.
   - **Equivalence check.** Wrap 50 seeded leaks as baselines and run them through the full H1 path: `outputs`, `compare`, `run_model`. Their caught cuts must equal the fast path's, cut for cut. If they differ, stop and report.
2. **The command,** `uv run --no-sync python -m olsprime.bench.adversary [--out DIR] [--n 10000] [--workers N]`.
   - It writes `adversary.json`, `adversary.md` and `run.json` to `runs/<UTC date>-h1-adversary/` (suffixed if taken), with the same provenance block as H1: commit, code digest, data manifest digest.
   - Its report gives:
     - the share caught at H1's 16 cuts (caught at one cut at least), with a 95 % Wilson interval, overall and per family;
     - the per-cut catch rate over the long schedule, overall and per family;
     - the size of each catch (the largest output move at a caught cut): its distribution, and the share within 10× of `CAUGHT_ABOVE`;
     - the honest twins' false-catch count, which must be 0;
     - the misses: counts by family, quantity, operation, offset and threshold position (inside or outside the bands), plus a sample of missed expressions in readable form.
   - It finishes in under about 30 minutes on the 12-thread machine, with N = 10,000. Measure it.
3. **Tests** in a new file `python/tests/test_adversary.py`, on synthetic bars:
   - the generator is deterministic for a seed;
   - every leak reads a later bar, and every twin reads none;
   - twins are never caught;
   - a few hand-built leaks with known verdicts behave as predicted (a next-bar close comparison is caught; a comparison outside a band is reported as possible to miss);
   - the fast path equals the full H1 path on a handful of leaks;
   - the Wilson interval is right on known values.
4. **Do not run the official command on the real data.** The `experiment-runner` does that after CI is green. A smoke run with `--n 200` on the real data is fine, and wanted: report its numbers as a smoke run, not as a result.

## Constraints

- **Files.** `python/olsprime/bench/adversary.py` and `python/tests/test_adversary.py`.
  - You may make small changes to `invariance.py` or `perturb.py` that do not change behaviour, to expose a function. `test_invariance.py` must stay green, and you must say exactly what changed: H1 is re-run this iteration, and its code digest will change.
- **Leaks are labelled.** Like `leaks.py`, the module states at the top that it builds leaky functions on purpose. Nothing in it may be imported by model code.
- **Python.** Always `uv run --no-sync ...`: a `rust-engineer` is editing the extension. Run the whole suite at the end with `cd python && uv run --no-sync pytest`.
- **Scripts.** Write scripts with a file tool, never through shell heredocs.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\adversary\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`. Bars come through `load_klines`, development period only, as H1 does.

## Done when

- The module, the command and the tests are in place.
- The equivalence check passes.
- The smoke run has run.
- The whole Python suite is green.

## Report (short)

- The generator in ten lines.
- The smoke run's numbers.
- Its runtime.
- The equivalence check's result.
- The families you expect to be missed most, and why.
- What is not done.
- One bolder next step.
