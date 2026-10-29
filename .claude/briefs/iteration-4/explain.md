# Brief — `olsprime.explain`: `why`, `series` and `sample` (iteration 4, wave 2)

- **Agent:** `ml-engineer` (opus, high).
- **Starts after** Report n°2's series (`series-power.md`) and `beta` on `CentredReadout` (`beta-getter.md`), both done.
- **Runs in parallel with** other agents, named by the orchestrator when it launches you. Touch only the files listed under Constraints.
- **Motto:** we should be more ambitious. Build an instrument a sceptic would trust: every number it prints is checked against the run it explains, and it says loudly when it cannot be.

## Goal

`olsprime.explain` is an instrument for the orchestrator, read by Claude. It answers "why is this forecast what it is?" for the Matrix arms of Report n°2.

The specification is `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)", section "`olsprime.explain`". Build it exactly as written: the output, the log of contacts, the default period, its reach, the exploratory rule, `why` with its two loud checks, `series` and `sample`. `head`, for the encoder, belongs to another task.

Its first use, by the orchestrator after the design is frozen, is the five whys of `docs/ITERATIONS.md`, iteration 4's brief. Read them: the tool must be able to answer (1), (2) and (5) with numbers, and (3) and (4) as exploratory descriptions.

Read first:
1. `CLAUDE.md`.
2. `docs/PROTOCOL.md`: the `explain` section, and "Matrix on hourly bars (iteration 3)" for the arms.
3. `docs/DESIGN.md` §3.1–3.3: the tree of centred nodes, the centred ridge and its weights.
4. The code:
   - `python/olsprime/models/matrix.py`: the arms, `MatrixSystem`, σ, `prob_map` and the stamps;
   - `python/olsprime/bench/report2.py`: `compute`, `series_arrays`, the `series.npz` keys, `report2.json`'s `fits`, provenance and `_json_bytes`;
   - `python/olsprime/bench/provenance.py`;
   - `python/olsprime/features/panel.py`;
   - `crates/olsprime-py/olsprime_rs.pyi`: `window_range`, `centred_query`, `centred_weights`, `centred_contribution`, and the new `CentredReadout.beta` and `mean_y` with their bit note;
   - `python/tests/test_report2.py`, for its synthetic store.

## Deliverables

1. **A package `python/olsprime/explain/`**, run as `uv run --no-sync python -m olsprime.explain <subcommand> ...`.
   - It takes a Report n°2 run folder and the SHA-256 of its `series.npz`. It refuses the folder if the digest does not match the file, or does not match the one in the run's `report2.json`.
   - It rebuilds the panel and the Matrix from the store exactly as `report2.py` does, and reads the fits' choices from `report2.json`. It refits nothing that `report2.json` already fixes.
2. **`why <pair> <as_of> --arm <arm> --by {year,month,day} --top k`.** Several `--at` values may share one process, so that the panel is built once; each still gets its own contact line. For each call:
   - **The cell.** The arm's raw readout (the 3 classes and `fwd_return`), its probabilities, and the cell's status. Also the arm's choices in force at `as_of`: window, ρ, σ's month, and the fit it comes from.
   - **The tree view.** The contributions of the entries grouped by stamp (`label_known_at`) at the chosen granularity, sorted by |c|, top k, with the rest summed so that the parts add up to the raw readout exactly. Say in the output that the grouping is by stamp, not by decision time.
   - **The per-feature view**, from `beta` and `mean_y`: ȳ, then βⱼ·z̃ⱼ per feature, sorted by |part|.
     - Sum it in the core's order, as the binding's docstring states, and report whether that sum equals the raw readout to the bit.
     - The protocol promises "within rounding". If the bit-exact sum holds, say so as a checked fact, never as a promise.
   - **The top-k entries by |w|**, with their pair, decision time, stamp, label, `fwd_return`, weight, and feature deltas x − x_q.
   - **The diagnostics:** Σw (it must be 1 within rounding), Σ|w|, the negative mass Σ_{w<0}|w|, the effective sample size 1/Σw², the leverage, and n in the window.
   - **The kernel arms**, `softmax-1y` and `knn-1y`. The forecast comes from the core, for the bit check. The weights may be recomputed in numpy, within rounding, which the output says. They have no per-feature view.
   - **The other arms.** `matrix-stack` shows each window's readout and its α. `matrix-adaptive` needs the ρ chosen at that decision: rebuild it from the candidates' forecasts of the rows it scores, if that can be done to the bit at a modest cost. Otherwise refuse the arm with a clear message, and say so in your report.
   - **The two loud checks.** A failure stops the call with a non-zero exit, and nothing is printed as a result.
     1. Each weighted entry's `label_known_at` is recomputed from its panel row, and for the top-k also from the raw bars' `open_time`, through the store's sealed loaders. Each must be at or before `as_of`. Matrix's own stamps are never trusted for this.
     2. The probability must equal the arm's probability for that row in `series.npz`, to the bit. A row that is not scored has no such check: say so in the output, and mark it.
3. **`series <arm> [<arm_b>] --period --by {year,month} --top k`** reads only `series.npz`, by its digest. It never builds the panel.
   - With one arm: the hourly Brier skill against `climatology-panel`, as the share of the period's Brier difference that each hour carries.
   - With two arms: the hourly log-loss difference.
   - It gives the share carried by the top 1 %, 5 % and 10 % of hours, by |part| and by signed part, a Lorenz-style summary, the parts by year or month, and the top k hours with their rows per hour.
4. **`sample <arm> --period --n --seed`** draws n scored rows of the period with a generator seeded by `--seed` alone, and runs `why`'s diagnostics on each.
   - Each row gets its negative mass, effective sample size, leverage, Σ|w|, its per-feature parts and its log-loss, with no per-entry lists.
   - It adds a short description: deciles of the negative mass with the mean log-loss and the calibration gap in each, and the mean |part| of each feature.
   - Both checks of `why` run on every sampled row.
5. **Output.**
   - JSON lines with sorted keys and no wall clock, plus a short text summary of at most about 40 lines per call.
   - Bounded: k ≤ 50, n ≤ 20,000.
   - Every answer carries its provenance: commit, with a dirty flag; the data manifest digest; the panel fingerprint; the `series.npz` digest.
   - Two identical calls give identical bytes.
6. **The log of contacts.** Every call appends one line to `runs/explain/CONTACTS.log`: UTC time, subcommand, period touched, arm, commit and the series digest.
   - Un-ignore that one file in `.gitignore`, beside the vault's audit trail.
   - Tests write to a temporary log, never to the real one.
7. **Periods.**
   - The default period is 2017-11 → 2018-12.
   - Any other period, and any `as_of` outside it, must be named with an explicit flag (for example `--test-months`). Without it the call is refused.
   - The tool reads the development panel only. Any period or `as_of` in 2025 or later is refused, and a test shows it.
   - `series` and `sample` read `series.npz`, which holds test rows only, so they touch test months by construction. They need the explicit flag too, and are logged like any call (after audit pass 4a).
   - `why` on a row that is not scored, such as every row of the default period, runs only the first loud check, and says so.
8. **Tests** in a new `python/tests/test_explain.py`, on the synthetic store of `test_report2.py`, with a Report n°2 run on it:
   - `why`'s parts add up to the raw readout;
   - the per-feature view sums to the readout in the core's order;
   - the forecast check passes on a scored row, and fails when the series file is changed by one bit;
   - the stamp check fails when a stamp is planted after `as_of`;
   - a period outside the default needs the flag, and 2025 is refused;
   - a wrong digest is refused;
   - one call writes one contact line;
   - two calls give the same bytes;
   - `series`'s shares match a direct computation;
   - `sample` is a function of its seed.

## Constraints

- **No real data.** Run nothing on the real store. The orchestrator makes the first real calls after the frozen design is committed, and each is a contact.
- **Files.**
  - New: `python/olsprime/explain/` and `python/tests/test_explain.py`.
  - Changed: `.gitignore`, for the one line.
  - Ask the orchestrator before touching anything else, `report2.py` and `models/matrix.py` included. If you need a small hook in one of them, ask with the exact change.
- **Python.** `uv run --no-sync` from `python/`. Never run `uv sync`.
- **Scripts.** Write them with a file tool. Never pipe or heredoc code into `python`.
- **Scratch:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-j944HL\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\explain\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- `why`, `series` and `sample` run on the synthetic store, deterministic, with both loud checks and the contact log.
- The tests above pass, and `uv run --no-sync pytest` passes, except failures you show to be caused by other agents' files in progress.

## Report (short)

- The commands, with one example output of each on the synthetic store.
- Which arms `why` supports, and how each gets its bits.
- Whether the per-feature sum is bit-exact in the core's order on the synthetic store.
- Every point where the protocol needed an interpretation, and the one you took.
- One bolder next step.
