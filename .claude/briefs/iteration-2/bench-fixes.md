# Brief — Bench fixes after Audit A, phase 1 (iteration 2)

- **Agent:** the `ml-engineer` who built the bench (`bench.md`), resumed.
- **When:** 2026-10-28, after Audit A's phase 1 ("pass with fixes": no Blocker, 1 Major, 9 Minor).
- **Motto:** we should be more ambitious. H1 claims "the bench catches leakage". Make the test earn that claim, or state its limits plainly.

## Findings to fix

### M1 — The perturbation is blind to `trades` and weak on volumes

**Where.** `python/olsprime/bench/perturb.py:25-27` and `:93-99`, pinned by `tests/test_invariance.py:49`.

**What fails.** On real BTCUSDT 1d, with the bench's own cuts and seeds, the auditor's planted leaks are flagged at:

| Leak | Flagged cuts |
|---|---|
| long iff `trades[t+1] > trades[t]` | 0 / 8 |
| a `quote_volume[t+1] > quote_volume[t]` peeker | 1 / 8 |
| a taker-share peeker | 3 / 8 |
| the price off-by-one, `ln(close[t+1]/close[t])` as `exp_return` | 8 / 8 |
| a fold-transductive scaler | 5 / 8 (0 at the three last-bar-before-fold cuts, as the design predicts) |

Its scripts are in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\audit-a\`: `leaks.py` and `leaks.log`.

**The fix.**
1. **Every column moves.**
   - Scale `trades` by the bar's volume factor and round to an integer.
   - Give the volumes and `trades` the same common level shift as the prices.
   - Make the taker share two-sided: draw the new share uniformly in (0, 1) and set `taker = share × total`.
2. **Both directions at every cut.** Run each cut twice:
   - once with the level shift ×3;
   - once with ÷3, a different seed for the walk and the factors, and the same body swap on the first later bar.
   - A one-bar comparison between a future value and a present one then flips at every cut, whichever way it pointed.
   - A row counts as a violation if it differs under either direction. Report the flags per direction.
3. **Coverage, asserted.** Every non-time column of every bar after the cut must differ from the original: every price, volume and taker column, and `trades`.
   - Assert it at runtime in `perturb_after`.
   - Pin it in `tests/test_invariance.py`, replacing the test that pins `trades` as untouched.
   - Only `open_time`, `close_time`, `known_at` and `symbol` stay. Say why in the docstring: the grid, and with it every null pattern, must stay the same.
4. **The planted leaks become permanent controls of the H1 run, beside the cheater.** Port the auditor's leaky models into `bench/invariance.py`, each with the flag pattern the design predicts, as `must`:
   - the trades peeker, the quote-volume peeker, the taker-share peeker and the price off-by-one: every cut;
   - the transductive scaler: every cut except the last-bar-before-fold cuts.
   - H1 holds only if every honest model passes and every planted leak shows exactly its predicted pattern.
5. **A positive control: the pipeline can see a signal when there is one.**
   - In the H1 run, fit ridge on labels misaligned one bar early.
   - Report its BSS against climatology. The auditor measured +0.97 on BTCUSDT, against −0.12 with the real labels.
   - The control must reach a BSS above +0.5 on both pairs, or H1's verdict says the pipeline is dead.
   - Add an offline version to the tests, on synthetic bars from `tests/fakes.py` or `bench/synthetic.py`.
6. **State the remaining limit in `h1.md`.** Perturbation cannot detect a leak that does not pass through a perturbed column. Examples: a calendar memorized from the future, or a lookup keyed on `open_time`.

### Minor fixes, all in the generated report and the bench

- **m1. Comparison counts.** `report1.py:739` prints "(14 comparisons)". Print the defined and undefined counts instead, for example "14 comparisons: 11 with an interval, 3 undefined (never exposed)". The verdict rule does not change.
- **m2. The cheater saturates the simulator.** Its book hits the simulator's `MAX_LOTS = 2^50` cap (`crates/backtest/src/sim.rs:26`), on a date that depends on the pair and venue.
  - Add a note to the cheater's line, giving the first date the cap binds: "its trading figures are not a measurement".
  - The auditor found ETH zero on 2022-06-29, BTC zero on 2024-08-11, and BTC at the fee-paying venues never.
- **m3. The block length sees only the returns.** The stationary bootstrap's block length comes from the autocorrelation of the returns alone, so volatility clustering is invisible to it.
  - **Keep the registered analysis as the primary.** That is Politis-White on the series the code uses now.
  - **Add a sensitivity to the H9 table and to the headline.** For each paired interval, use the mean block max(PW(d), PW((d − d̄)²)), where d is the paired difference series. Print the block it chose.
  - Say whether any verdict changes.
  - Fix the wording that the bootstrap is robust to serial correlation: "of the returns, not of their squares", unless the sensitivity covers it.
- **m4. Diebold-Mariano uses no lags at h = 1.**
  - Add a sensitivity column to the forecast table: Newey-West, with the Bartlett kernel and L = floor(4·(T/100)^(2/9)) lags. That is 7 lags at T = 2,190.
  - The primary stays as registered.
- **m6. Make the config the source of `data_end`.** `report1.py:108` records it in `config.json`, but `load_pair` (`:144`) reads `DEVELOPMENT.end` directly. Load from the config.
- **m7. Say what "byte for byte" was shown for.** The report says it reproduces byte for byte. Add "on the same machine and library versions": XGBoost's `hist` with `nthread=1` is deterministic for one binary and CPU, and nothing more was shown.
- **m8. Fence `blind_cheater` in its docstring** (`invariance.py:375`). It builds features and labels over the whole sealed series, the validation year included, only to show the seal.
  - No model is fitted there, and no number from 2025 is reported.
  - Its docstring must say so, and must forbid extending it.
- **m9. Thin trading.** Some strategies trade a handful of times and still get a Sharpe and intervals: ridge at Bitvavo on BTCUSDT runs 1.0 turnover per year with 0 % exposure. Add a `fills` column to the trading tables so the reader sees it. Do not filter any row.

Not changed: **m5**, `--check` rebuilding the whole manifest. A strict check that fails when any data file changes is the point. The official runs wait until the data is final.

## Constraints

- **Files.** `python/olsprime/bench/**` and the bench's tests (`test_invariance.py`, `test_report1.py`, `test_bench_pipeline.py`).
  - `python/olsprime/eval/stats.py` and `tests/test_stats.py` are allowed only for backward-compatible additions, such as an optional block-length argument. No existing default changes.
  - Touch nothing under `python/olsprime/data/`, `manifest.py`, `protocol.py`, `tests/conftest.py`, `tests/allowlist.py`, `tests/tripwire.py` or `tests/test_survey.py`: a `data-engineer` owns them right now.
  - Touch nothing under `crates/`.
- **Python.** Run it with `uv run --no-sync` from `python/`. A `rust-engineer` is editing `crates/risk`, and a plain `uv run` would rebuild the extension mid-edit.
- **Runs.**
  - Verification runs go to your own `--out` folders, under your scratch folder or `runs/<date>-h1-fix*`.
  - Never overwrite `runs/2026-10-27-*`.
  - Do not write `docs/reports/report-1.md`: use `--no-docs`. The official runs belong to the `experiment-runner`.
- **Never** open the vault.
- **Done when.**
  - `uv run --no-sync pytest` is green.
  - A fresh H1 run shows the new pattern table, the positive control and the limits.
  - A fresh Report n°1 run shows the fixes, and its `--check` passes against itself.

## Report

- What changed, file by file.
- The new H1 table for both pairs: flags per direction, the planted leaks, the positive control.
- The run times of H1 and of Report n°1.
- The H9 sensitivity: blocks and verdicts.
- The DM sensitivity.
- The test count.
- One bolder next step.
