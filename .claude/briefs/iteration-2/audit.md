# Brief — Audit of iteration 2

- **Agents:** two `leakage-auditor`s (fable, high), in parallel and read-only.
  - **Audit A** covers the Python bench and the results.
  - **Audit B** covers the Rust work and its claims.
- **Motto:** we should be more ambitious: assume it leaks until you have shown otherwise, and trace every timestamp from the raw file to the decision.

Read `docs/PROTOCOL.md`, `docs/HYPOTHESES.md` and `docs/DESIGN.md` first, then `docs/ITERATIONS.md` (the iteration 2 brief) and the briefs in `.claude/briefs/iteration-2/`. The working tree is uncommitted: `git diff main` and `git status` show everything this iteration changed. Write throwaway checks only in your scratch folder; never edit the repository.

## Audit A — Python bench, data and results

- **Data.**
  - `olsprime/manifest.py` and the new read helpers in `protocol.py`, including `load_universe`.
  - The runtime tripwire in `tests/conftest.py` and the shared allow-list. Try to read a Parquet file in a test without tripping it.
  - The universe download, as reported in `docs/DATA.md`: survivorship, delisted pairs, ticker reuse.
- **Features and labels** (`features/`, `labels/`).
  - Every feature uses only bars final at t's close (`known_at`).
  - The label uses `open[t+1]` and `open[t+1+h]`, and its `label_known_at` is right.
  - Gaps, zero volume, the vault boundary.
  - Shift the labels by one bar and check that the cheater's edge appears and the honest models' does not.
- **Splits** (`eval/splits.py`). Purge and embargo, and no training label known after the fit. The guards for the period and the vault. Months at the edges.
- **Baselines** (`models/`).
  - `predict` never sees a label.
  - Scalers and λ are fitted inside the fold, and the inner validation is purged.
  - The random baseline's turnover is matched on training data only.
  - XGBoost is deterministic.
  - The decision rule's costs are those of the venue.
- **Statistics** (`eval/stats.py`). Check every formula against the cited papers, as far as you can: Sharpe SE, PSR, DSR (N and V), the stationary bootstrap and its block length, Diebold-Mariano with HAC and HLN, ECE. Is the bootstrap blind to autocorrelation anywhere?
- **The bench** (`bench/`).
  - The future-invariance test: does it perturb enough, compare bit for bit, and cut where a missing purge would show?
  - Could an honest-looking model pass it and still leak? Build one in your scratch folder and try.
  - Does the cheater really fail at every cut?
- **Report n°1** (`docs/reports/report-1.md` and the run folders under `runs/`).
  - Same scored window for every strategy.
  - Fees of the right venue at the taker rate, slippage, filters.
  - Annualization on a regular grid.
  - The variant count and V for the DSR.
  - H9's verdict follows its pre-registered rule, word for word.
  - Every claim in the draft rows is supported, and nothing more is claimed.
- **Costs.** The five schedules match `docs/PROTOCOL.md` and the constants in `crates/backtest`.

## Audit B — Rust: bindings and Kani

- **`crates/olsprime-py`.**
  - Does `simulate` pass the arrays without copying them wrong (strides, dtype, length)?
  - Is the output bit for bit the Rust result?
  - Are the errors mapped, and can any panic cross into Python?
  - Can a stale binary run: does uv rebuild when the Rust code changes?
  - The workspace still passes fmt, clippy, test and the MSRV check.
- **`crates/risk` and the Kani contracts.**
  - Is each contract the right specification: strong enough for the harnesses that use it, and true?
  - Is every `stub_verified` backed by a `proof_for_contract` that actually finished?
  - Do the timing claims and the `proven` list in `kani.yml` match the logs (`~/kani-it2/` in WSL2 and the agent's scratch folder)?
  - Did `lib.rs` behaviour change? The 40 tests and the property tests must still pin it.
  - The GitHub `open` run 40596921451 (`gh run view 40596921451`): what finished?
- **Docs.** Check the claims in `docs/DESIGN.md` §2–§5 and §7, added today after an external review, against the code.

## Report (each audit)

- **Findings, ranked:** Blocker, Major, Minor. Each one gives `file:line`, the concrete failure scenario and the fix.
- **Clean:** what you checked and found clean.
- **Verdict:** pass, pass with fixes, or fail.
- One bolder check to add to the bench.
