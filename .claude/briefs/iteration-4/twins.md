# Brief — H1's blind spots: the adversary's future-shuffle twins (iteration 4, wave 2)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with** other agents, named by the orchestrator when it launches you. Touch only the files listed under Constraints.
- **Motto:** we should be more ambitious, in the test, never in the claim. The adversary measured how many random leaks H1 catches. This task says what the misses are.

## Goal

The adversary of iteration 3 (`python -m olsprime.bench.adversary`) caught 9,795 of 10,000 random leaks at H1's cuts. Of the 205 misses, 163 are constant at the compared rows. The other 42 were classified by hand in audit phase 2.

Build the registered instrument that classifies every miss mechanically. The specification is `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)", section "H1's blind spots":
- each leak gets K = 8 seeded future-shuffle twins at each of H1's cuts;
- in a twin, each of the pair's bars after the cut keeps its time stamps and takes every other value of a bar drawn, with replacement, from the same pair's bars at or before the cut: open, high, low, close, volume, quote volume, trades, and the taker base and quote volumes (after audits 4a and 4b);
- a twin moves a leak when, at a compared row, one of its outputs differs by more than 10⁻⁹, turns missing or stops being missing: H1's own catch rule;
- the draws use the registered seed 20261030;
- a leak caught at no cut is one of three kinds: constant at the compared rows; unchanged under all K twins at every cut, so on these bars it does not depend on the future; or a true blind spot, which moves under some twin yet H1 did not catch it.

The result is descriptive. H1's rule does not change.

Read first:
- `CLAUDE.md`;
- `docs/PROTOCOL.md`: "Leakage tests", the adversary's section of iteration 3, and the iteration 4 section above;
- `docs/ITERATIONS.md`: iteration 3's outcome, the adversary row and audit phase 2's classification of the 42 misses;
- `python/olsprime/bench/adversary.py`: `Leak`, `generate`, `Bench`, `build_bench`, `_window`, `quantities`, `evaluate_fast`, `FastResult.constant`, `summarize` and `run`;
- `python/olsprime/bench/invariance.py`: `perturb_after`, `perturbation_at`, `CAUGHT_ABOVE`, the cuts;
- `python/tests/test_adversary.py`.

## Deliverables

1. **A new module `python/olsprime/bench/twins.py`**, run as `uv run --no-sync python -m olsprime.bench.twins [--adversary <run folder> --sha256 <digest of its adversary.json>] [--out DIR] [--workers N]`.
   - **The leaks.** It regenerates the adversary's leaks from the adversary's own config and seed, with its bench on H1's cuts.
   - **The cross-check.** With `--adversary`, it checks that its misses are the run's misses, leak by leak, and refuses a file whose digest is not the one given.
   - **The twins.** `shuffle_after(bars, cut, rng)` keeps every time column and replaces every other bar column of each row after the cut by the same columns of a row drawn with replacement before the cut. The draws come from the rows at or before the cut, the cut's own bar included, as the protocol now says. The quantities are then recomputed from the twin bars, exactly as H1's perturbation recomputes them, and windowed as `build_bench` does.
   - **The comparison.** Each leak's outputs on each twin window are compared with its outputs on the original window, at the compared rows H1 scores, with H1's catch rule: a move above `CAUGHT_ABOVE`, or a NaN on one side only.
   - **Seeds.** One generator per (pair, cut, twin), derived from 20261030 so that the result does not depend on the order or the worker count. The draws are made once per cut and shared by every leak.
2. **The classification**, for every leak:
   - whether H1 caught it, at how many of its cuts;
   - whether it is constant at the compared rows;
   - at how many cuts some twin moves it.

   A miss falls in exactly one class: constant, unchanged under every twin, or true blind spot. Report:
   - each class's count and share of the misses, with Wilson intervals;
   - the tag tables (family, quantity, operation, offset), as `adversary.py` gives them for misses;
   - the list of true blind spots, each with its description and tags.

   Also report, as a sanity check, the share of caught leaks that some twin moves. It should be near 1; say what keeps it from 1.
3. **Outputs.** `twins.json` (sorted keys, no wall clock), `twins.md`, `config.json` (with provenance, as `adversary.py` writes it) and `run.json` (wall clock only), to `runs/<UTC date>-twins/` or `--out`.
   - Two runs give byte-identical `twins.json` and `twins.md`.
   - So do runs with different worker counts.
4. **Tests** in a new `python/tests/test_twins.py`, on synthetic bars, as `test_adversary.py` makes them:
   - a planted leak of each class lands in its class:
     - a constant;
     - an XOR of the same future bar with itself, which cancels;
     - a threshold outside the band H1's perturbation flips, such as a fill above 0.97, which H1 misses and a twin moves;
   - an honest leak, reading only the past, is never moved by a twin;
   - the twins keep every time stamp, and every twin bar after the cut equals some bar before it;
   - determinism, across worker counts too;
   - the digest check refuses a changed file;
   - the cross-check with an adversary run on the synthetic bars passes.

## Constraints

- **No real data.** Run nothing on the real store. The orchestrator runs the official twins after the audit, and logs the contact.
- **Files.**
  - New: `python/olsprime/bench/twins.py` and `python/tests/test_twins.py`.
  - Ask the orchestrator before touching anything else. If `adversary.py` or `invariance.py` needs a small hook, ask with the exact change: their outputs must stay byte-identical.
- **Python.** `uv run --no-sync` from `python/`. Never run `uv sync`.
- **Scripts.** Write them with a file tool. Never pipe or heredoc code into `python`.
- **Scratch:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\twins\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- `python -m olsprime.bench.twins` runs on synthetic bars, deterministic, with the tests above.
- `uv run --no-sync pytest` passes, except failures you show to be caused by other agents' files in progress.

## Report (short)

- The design as built, and its runtime on synthetic bars, projected to the official N = 10,000.
- What "before the cut" means in your twins, and why.
- Every point where the protocol needed an interpretation, and the one you took.
- One bolder next step.
