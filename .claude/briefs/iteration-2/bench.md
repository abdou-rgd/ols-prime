# Brief — The H1 bench and the Report n°1 command (iteration 2)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with** the agents whose modules you integrate. They are being written right now:
  - features, labels and splits (`features-labels-splits.md`, a data-engineer);
  - baselines and the decision rule (`baselines.md`, an ml-engineer);
  - statistics (`stats.md`, an ml-engineer);
  - PyO3 bindings `olsprime_rs` (`pyo3.md`, a rust-engineer);
  - the vault manifest (`universe-manifest-tripwire.md`, a data-engineer).
- **Motto:** we should be more ambitious: one command that a sceptic can run, re-run, and fail to fool.

## How to work while the others finish

Read every brief named above: their contract sections are the interfaces you code against. Start with what depends on nothing (the pipeline skeleton, the perturbation, the report layout, tests on fakes). Then integrate each module as it appears on disk: check with `ls`, and read its code before you call it. Never edit another agent's files. If an interface differs from its brief, adapt your code and list the difference in your report. If a module is still missing when everything else is done, report that; do not write a stand-in.

## 1. The walk-forward pipeline: `olsprime/bench/pipeline.py`

- `walk_forward_predict(baseline_factory, frame, folds) -> pl.DataFrame`. For each fold:
  - build a fresh baseline;
  - `fit` it on the fold's training rows;
  - `predict` the test rows, with `LABEL_COLUMNS` dropped by the bench itself.

  Concatenate the predictions, one row per test decision.
- `targets(predictions, venue) -> np.ndarray`, aligned to the scored bars:
  - with `exp_return`, the decision rule `long_flat` at the venue's round-trip cost;
  - otherwise the baseline's own `target`.
- `simulate_strategy(bars, targets, venue, *, initial_cash=10_000.0)`, through `olsprime_rs.simulate`:
  - taker liquidity, 2 bps slippage, Binance's filters for the pair (`SymbolFilters` snapshot: min notional 5, step 0.00001 BTC or 0.0001 ETH);
  - it returns the equity curve and the fills.
- **Cross-check.** On every fold, the ridge fitted with numpy equals `olsprime_rs.ridge_fit` (max abs difference below 1e-9 on standardized features). Log the worst difference. This is the first use of the Rust core from Python.

## 2. H1, future invariance: `olsprime/bench/invariance.py`

`docs/HYPOTHESES.md`, H1: the bench catches leakage. A model that reads its label is flagged, and every honest model passes the future-invariance test.

- **The test.** For a pair, a baseline and a cut bar t:
  1. Run the full pipeline (features, labels, folds, fit, predict, rule) on the original bars.
  2. Perturb every bar after t: scale open, high, low and close of each later bar by one positive random-walk factor (it keeps the bars valid), and scale the volumes.
  3. Rebuild everything from the perturbed bars and run the pipeline again.
  4. Every output with `decision_time ≤ known_at[t]` must be identical, bit for bit: `p_*`, `exp_return` and `target` at every venue.
- **Cut points.** For each pair, at least 8:
  - the last bar before three different fold starts, to catch a missing purge;
  - the first bar of three test months;
  - two random bars.
- **Models.** Run it on every baseline, with 3 seeds of `random-turnover`, and on the cheater.
- **Verdict.** H1 holds if every honest model has zero violations, and the cheater is flagged at every cut.
- **The seal.** On the real sealed data (`load_klines`, 1d), the cheater's decision at the last bar before the vault (2025-12-31) cannot be computed: the next bar is sealed. Show it, so the report can say the seal refuses the cheater.
- `python -m olsprime.bench.invariance` runs it on BTCUSDT and ETHUSDT 1d, and writes `runs/<UTC date>-h1/` with `h1.json` and a short `h1.md`.

## 3. Report n°1: `python -m olsprime.bench.report1`

"What fees do to simple strategies." Read H9 in `docs/HYPOTHESES.md` and the Costs section of `docs/PROTOCOL.md` first.

- **Data.** BTCUSDT and ETHUSDT, 1d, through `load_klines`. Features, labels with h = 1, and walk-forward folds with monthly refits over the test months 2019-01 → 2024-12 (DEVELOPMENT; the validation year and the vault are untouched).
- **Scored window.** The bars of 2019-01-01 → 2024-12-31. Every strategy and venue is scored on the same bars. Rows outside the window only train.
- **Strategies.** The baselines of `baselines.md`, all of them, and nothing more. `random-turnover` runs 100 seeds and reports the median and the 5th and 95th percentiles. The cheater appears only in a separate, labelled line: it is the H1 control, not a candidate.
- **Venues.** `zero`, `binance-spot`, `bitvavo-entry`, `coinbase-advanced-entry`, `kraken-pro-entry`, all at the taker rate.
- **Metrics per strategy, pair and venue.**
  - Trading: total and annualized net return, annualized Sharpe with its analytic 95 % interval and its stationary-bootstrap 95 % interval, PSR(0), DSR, Sortino, maximum drawdown, annual turnover, fees paid, exposure.
  - DSR: N = the number of strategy variants with a defined Sharpe on that pair and venue (`flat` and the cheater excluded). V = the variance of their per-period Sharpes. The variant list comes from code, never typed by hand.
  - Forecasts, for `climatology`, `ridge` and `xgboost` (venue-independent): log-loss, Brier, Brier skill against climatology with a paired bootstrap interval, top-label and class-wise ECE, and Diebold-Mariano on log-loss against climatology.
- **H9.** For `coinbase-advanced-entry` and `kraken-pro-entry`, each pair and each active baseline: the net Sharpe minus buy and hold's, with a paired stationary-bootstrap 95 % interval, and the verdict per venue. Report the other venues too, labelled descriptive: H9 names only those two.
- **Outputs.**
  - `runs/<UTC date>-report1/`: `config.json`, `manifest.json` (from `olsprime.manifest.build_manifest`, with its digest), `metrics.json`, and the equity curves.
  - `docs/reports/report-1.md`: the tables, generated. It prints the git commit, the manifest digest, the SHA-256 and sealed row count of each file used, and the variant count.
- **Determinism.** Same inputs give byte-identical `metrics.json` and `report-1.md`. Keep wall-clock times out of them, in a separate `run.json`. `--check DIR` re-runs and fails, with a diff, if anything differs from `DIR`.
- **Speed.** Aim for a few minutes. Parallelize across processes only if the results stay identical.

## 4. Stretch, only if time allows: the universe panel

On the monthly top-30 universe (`data/parquet/universe/spot_usdt_top30.parquet`, read through the protocol's `load_universe`, if the data-engineer has added it), run the non-learned strategies (`buy-and-hold`, the three `ma-momentum`, `random-turnover`) on each member over its contiguous membership spans. Exit at the end of each span. Report the cross-section of net Sharpes per venue. Skip it if the data or the loader is not ready.

## Tests (offline, synthetic, fast)

- **Pipeline.**
  - `predict` never receives a label column.
  - Targets align with bars: the target at bar t executes at bar t + 1's open.
  - The scored window is the same for every strategy.
- **Invariance.**
  - On a synthetic series, every honest baseline passes.
  - The cheater fails, and so does a planted leak: a model trained without the purge.
- **Report.** Determinism: two runs on a small synthetic dataset give byte-identical outputs.
- Run the whole `uv run pytest` at the end. If another agent's test fails, report it; do not fix it.

## Scope and constraints

- **Files.** Only these:
  - `python/olsprime/bench/**`;
  - `python/tests/test_bench_pipeline.py`, `test_invariance.py`, `test_report1.py`;
  - `docs/reports/report-1.md`, generated by your command.
- **Running.** Run each command once on the real data to prove it works. `experiment-runner` makes the official runs and writes `docs/RESULTS.md`, so do not edit it.
- **Never.** Never open the vault. Read data only through `olsprime.protocol`. Add no dependency.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\bench\`.
- **Git.** Run no git command that changes state, and do not commit. Reading `git rev-parse HEAD` and `git status --porcelain` for the report is fine.

## Report (short)

- Modules, CLIs and tests, with counts.
- The H1 result on real data: violations per model, and the cheater's flags.
- The headline of Report n°1: B&H against the best active baseline at each venue, and the H9 verdicts. These are claims for the orchestrator to verify, not results yet.
- The run time, and whether `--check` passes.
- Interface differences from the briefs.
- One bolder next step.
