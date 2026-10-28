# Evaluation protocol

Fixed before any model is trained. Any change gets a dated entry in the change log at the bottom.

## Data

- **Klines.** Spot klines from data.binance.vision, 1 h and 1 d, from August 2017.
  - Checksums are verified.
  - Timestamps are normalized to milliseconds UTC; spot files switched to microseconds on 2025-01-01.
- **Universe.** 20–30 USDT pairs, re-selected each month by trailing 30-day quote volume.
  - Only data available before that month is used.
  - Delisted pairs stay in the history.
  - Stablecoins and leveraged tokens are excluded.
- **Venue.** Binance serves for history only. Live trading targets a MiCA-authorized venue, and costs come from that venue's fee schedule.

## Splits

| Period | Use |
|---|---|
| 2017-08 → 2024-12 | development, walk-forward with monthly refits |
| 2025-01 → 2025-12 | validation and model selection |
| 2026-01 → 2026-10 | vault: opened once, at the end, for all final models together |

- Purge and embargo equal the label horizon h (1 day or 7 days).
- Everything estimated is fitted inside the fold: scalers, PCA, Gram matrices, thresholds, calibration.
- **The vault is sealed in code** (`python/olsprime/protocol.py`).
  - Every read of stored klines goes through `load_klines`, which drops each bar whose values are final at or after 2026-01-01 00:00 UTC. A test fails if any other module reads a Parquet file.
  - The vault opens only with two keys at once: the environment variable `OLSPRIME_OPEN_VAULT=1`, and a file `runs/vault/OPEN` written by hand at the final evaluation. Its first line is the date alone (ISO 8601) and the next lines are the ids of the final models. An empty file, or a first line that is not a date, keeps the vault shut with an error.
  - Each read with the vault open adds a line to `runs/vault/ACCESS.log`.
  - `OPEN` and `ACCESS.log` are the only files under `runs/` that git tracks, so the opening is recorded in the history.

## Leakage tests (mandatory for every model)

- **Future invariance:** perturbing any data after t leaves every decision at or before t unchanged.
  - Every value of every later bar moves: prices, volumes, trades and taker shares. So does every ratio of two values of a bar. The run asserts this coverage. Only the time stamps stay.
  - Both directions run at every cut: the future multiplied by 3, and divided by 3.
  - Bar t + 1 is built, not drawn. In one run, each of the quantities below sits above bars t − 90 to t; in the other, below them:
    - prices, body, wicks and range;
    - VWAP and taker VWAP;
    - volumes, trades and trade size;
    - taker shares.

    So a comparison between bar t + 1 and any of those bars flips in exactly one run.
  - Bar t + 1's candle takes opposite shapes in the two runs: small and filled by its body in one, large and dominated by one wick in the other. One run's candle is green and the other's red, and which is which alternates from cut to cut, so all four pairings of colour and size occur. Its fill (body over range) and its close location sit above bar t's in one run and below them in the other.
- **Cheater detection:** the bench must flag a deliberately leaking model, one that reads its own label.
- **Planted leaks:** the bench must also flag each of the following at exactly its predicted cuts:
  - models that compare one quantity of the next bar with the present: trades, quote volume, the two taker shares, range, body, each wick, fill, close location, VWAP, taker VWAP or trade size;
  - models that threshold the next bar's range, body or close location;
  - models that pick out a small red or a large green next bar;
  - models that use the next bar's range or VWAP as their forecast;
  - a model that compares the next close with an older one;
  - a return shifted the wrong way;
  - a scaler fitted on the whole test month, ahead of a ridge that does not rescale;
  - models fitted without the purge.

  A planted leak is caught at a cut only where an output moves by more than 10⁻⁹, turns missing or stops being missing. A change at the level of rounding carries no leak, so it never counts as a catch. An honest model, by contrast, must not differ by a single bit.
- **Positive control:** ridge fitted on labels shifted one bar early must beat climatology with a Brier skill score above 0.5. The pipeline must see a signal when there is one.
- **The limits.**
  - A model that reads the time stamps, such as a calendar learned from the future, passes the perturbation. Code review and the seal stop it, not this test.
  - Comparisons two or more bars ahead flip with high probability only, not by construction. So do comparisons with bars older than t − 90, the shape functions other than the fill, the close location and colour with size, such as the ratio of the two wicks, and their pairings with the colour.
  - A threshold on bar t + 1 flips by construction only inside the design's bands: a body between 0.4 % and 10 %, a range between 0.5 % and 50 %, a fill between 0.2 and 0.8, a close location between 0.25 and 0.75.
  - A cut where a quantity cannot cross is marked `?`, for example a taker share already at 1, or a bar without volume. The `?` is predicted from the original bars, before perturbing, and only for a declared reason. It is reported and left out of the verdict; it never counts as a pass. A failure to cross without a declared reason fails the test, and so does a control that is `?` at every cut.

## Metrics

- **Primary, forecast quality:**
  - log-loss;
  - Brier skill score against climatology (historical class frequencies);
  - expected calibration error and reliability diagrams;
  - Diebold-Mariano tests between models.
- **Secondary, trading:** net return, Sharpe and Sortino ratios, maximum drawdown, turnover, fees paid, exposure.
  - PnL has little statistical power here. Over the ten-month vault, the standard error of an annualized Sharpe ratio is about 1.1, so a Sharpe of 1 cannot be told apart from 0.
  - Every reported Sharpe ratio comes with the number of variants tried (see [RESULTS.md](RESULTS.md)) and a deflated Sharpe ratio.
- **Robustness, after Microsoft-Decision-1:** how often the decision flips under small perturbations of the inputs, such as feature noise or asset order.

## Costs

Results are reported under four fee schedules, plus zero fees as the gross reference. Rates as of October 2026; re-check before use.

| Venue | Maker | Taker | Source, read on 2026-10-27 |
|---|---|---|---|
| Binance spot (history only) | 0.10 % | 0.10 % | checked in iteration 1 |
| Bitvavo, entry tier (MiCA, Dutch AFM) | 0.15 % | 0.25 % | Bitvavo's API documentation and a search snippet of its fee page. The page itself was not read. |
| Coinbase Advanced, entry tier, EU | 0.25 % | 0.50 % | two news reports of Coinbase's announcement of 2026-09-16. Coinbase's own pages answer HTTP 403. |
| Kraken Pro, entry tier | 0.40 % | 0.80 % | Kraken's fee-schedule page, level 1 |

- Slippage is 2 bps per side. The exchange minimum notional and lot step are enforced.
- Every order is charged the taker rate: a decision at a close executes at the next open.
- Fee schedules change (Kraken on 2026-07-09, Coinbase on 2026-09-16). Applying today's rates to past years is a stated assumption, not a leak.
- The backtest uses Binance's USDT prices. Bitvavo, Coinbase and Kraken quote BTC and ETH in EUR or USD, so their price paths differ by the exchange rate.

## Baselines

Climatology, always flat, buy and hold, moving-average momentum, random positions with the same turnover, ridge on hand-made features, XGBoost.

## Change log

- 2026-10-25: initial version.
- 2026-10-25: the vault is sealed in code, after the iteration 1 audit. The splits do not change.
- 2026-10-26: git tracks the vault's key file and access log, after the re-audit.
- 2026-10-27, before Report n°1 runs:
  - Fees re-read at the source by a `scout`. Kraken is confirmed. Coinbase rests on secondary sources.
  - Bitvavo joins as a fourth schedule. It is the cheapest MiCA-authorized venue found that serves French residents.
  - Binance's exit from France is confirmed at the source: the AMF withdrew BINANCE France SAS's registration on 2026-07-02, at the end of the MiCA transition period.
  - A runtime tripwire in `python/tests/conftest.py` now backs the static guard during tests.
  - Diebold-Mariano tests use the original rectangular kernel with h − 1 lags and the Harvey-Leybourne-Newbold correction. They fall back to Bartlett only when the rectangular variance is not positive. Measured size at a nominal 5 %, with h = 7: 5.25 % rectangular against 8.75 % for Bartlett. With h = 1 the two are identical.
- 2026-10-28, after Audit A of iteration 2 and before the official runs:
  - The future-invariance test is strengthened. Before, `trades` did not move, the volumes had no common level shift, the taker share moved one way only, and each cut ran in one direction. The auditor's peeker on `trades` was flagged at 0 of 8 cuts. Now every column moves, and every cut runs in both directions.
  - A second pass of the audit found that the ratios inside a bar still did not move: range, body, VWAP, taker VWAP and trade size. Six planted ratio leaks were flagged at 0 of 8 cuts. Now every ratio moves, bar t + 1 is built against the last 91 bars, and those leaks are flagged at 8 of 8. The auditor's planted leaks become permanent controls, and a positive control is added (see Leakage tests). The registered hypothesis H1 does not change.
  - A third pass found that the candle's shape kept its proportions in both runs, so a leak on the next bar's close location or fill was flagged at only 3 and 2 of 8 cuts. Now the two runs give bar t + 1 opposite shapes, and those leaks are flagged at 8 of 8. Five shape leaks join the permanent controls.
  - The same pass found that a `?` was read from the perturbed bars: a push that stopped working would have read as `?`, not as a failure. Now a `?` is predicted from the original bars before perturbing, for a declared reason only.
  - A fourth pass found that colour and size were tied. The two runs gave a small green bar and a large red one, so a leak on a small red next bar was flagged at 0 of 8 cuts. Now the colours alternate from cut to cut, and two controls on colour with size join the bench.
  - Two sensitivities join Report n°1. The registered analyses do not change.
    - H9's paired bootstrap is re-run with a block of max(PW(d), PW((d − d̄)²)), where PW is the Politis-White length.
    - Diebold-Mariano is re-run with a Newey-West Bartlett variance and L = floor(4(T/100)^(2/9)) lags, without the Harvey-Leybourne-Newbold correction.
  - The vault guards are strengthened after two evasions the auditor found. `docs/DATA.md` gives the details and the remaining limits.
    - The runtime tripwire now goes up before any test module is imported.
    - It now also wraps the raw-archive readers (`zipfile`, polars' CSV and other text readers) and pyarrow's dataset constructors.
    - It trips on any wrapped read of a path under `data/`.
    - It watches the seal. A test may not rebind the seal's names, and a sealed read that returns a vault row fails the test.
    - The static guard now flags `sys.modules`, edits of `sys.path`, and any patch of the seal.
    - The tripwire runs only under pytest. The command-line runs rely on the seal in `olsprime.protocol` and on the static guard.
- 2026-10-28, after the first official H1 run (commit 9c414ca), which held:
  - CI on Linux found a control caught by rounding alone. The transductive scaler fed its scaled features to the baseline ridge, which standardizes again on its training rows and so undoes the scaling exactly. Its outputs moved by 10⁻¹⁶ at most. The bit-for-bit rule counted that as a catch at every cut on Windows, and at all but one cut on Linux.
  - Now a control is caught only where an output moves by more than 10⁻⁹ (see Leakage tests). Measured on the synthetic bars, every other control's weakest catch moves an output by at least 2.7 · 10⁻³ or flips a target.
  - The transductive scaler's ridge now centres the features but keeps their scale, so the test month's spread reaches the penalty. Its catches move an output by 8 · 10⁻³ to 2.5 · 10⁻² on the synthetic bars.
  - H1 is run again on the corrected code (commit 881a63c). The first run's row stays in `docs/RESULTS.md`, followed by an erratum row for the second run. Report n°1 does not use the changed code. Re-run at 881a63c with `--check`, it reproduces every number: only the commit and code digest lines differ, in `config.json` and `report-1.md`. The registered hypothesis H1 does not change.
  - A second audit of the guards found three more ways around them and eight smaller holes. Each is closed, or stated as a limit in `docs/DATA.md`.
    - Every test gets an empty data root of its own. With both keys forged, the two file loaders still refuse the repository's `data/`.
    - A raw archive under `data/raw/` is read only when every file on the call stack holds the allowance, the test file included.
    - The allow-list is read-only. After every test, a check confirms that every guard is still in place, and a breach ends the session.
    - The static guard flags the attributes that take a guard apart, such as `__globals__`, `__closure__` and `__bases__`, and bans `runpy`, `gc` and `ctypes`.
    - A third audit closed two more ordinary routes. A test allowed to read back its own Parquet files could name the repository's `data/` directly, and `zipimport` could read the raw archives. Now, under pytest, only the seal's loaders may read the repository's `data/`, and `zipimport` is both gated and banned.
    - In plain words: a guard inside the process can be taken down from inside the process. These guards stop accidents, not sabotage. The structural fix is to keep the vault rows where the process cannot read them without both keys.
  - **A vault exposure, recorded here.**
    - **What was read.** To prove the holes D1 and D2, the auditor's evasion tests read BTCUSDT 1d vault rows outside the seal: the 31 bars of the 2026-03 raw archive, and the vault months of the Parquet file.
    - **What the tests show.** They assert row counts. One of them prints the first bar of the 2026-03 archive when it succeeds.
    - **Why it does not reach a model.** The auditor writes no model code, and no feature, model or decision of the project uses these rows.
    - **What follows.** The vault stays the final test set. This exposure is stated in every report that uses it. Proofs of a hole now run against fake data only, and the new guards stop both paths.
