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
- **The physical seal (iteration 3).** `python -m olsprime.vault` moves the vault months out of `data/` into a vault tree outside the repository (stage 1), then encrypts that tree to an age key held off this machine (stage 2). `docs/DATA.md` gives the stages, the stage reached and their limits. The two keys now also decide whether the vault tree is read at all.
  - **Sealed.** `load_klines` reads `data/` only, and never touches the vault tree.
  - **Both keys, stage 1.** `load_klines` also reads the pair's plaintext file in the vault tree, joins the two on `open_time` (the file in `data/` wins) and sorts.
  - **Both keys, stage 2.** The `.age` file is decrypted in memory with the identity file named by `OLSPRIME_VAULT_IDENTITY`. With no identity, or a wrong one, `VaultSealed` is raised and nothing opens, even with both keys forged.
  - Each part read is logged in `ACCESS.log`, the tree's part tagged `[vault tree]`. `olsprime.vault` refuses to run while the vault is open.

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

**The adversary (iteration 3).** Beside the designed controls, H1 reports how many random leaks it catches.
- A seeded generator plants random leaks: functions of the next bars' levels, ratios and shapes, alone or combined, and compared with the past, with other future bars or with thresholds.
- Each leak runs through H1's own perturbation, at H1's cuts and at a longer schedule of cuts.
- The report gives the share caught with a 95 % Wilson interval, overall, per family and per cut. It also gives the size of each catch and the list of misses.
- Every leak has an honest twin, the same expression moved into the past. A twin must never be caught: one catch is a bench bug.
- The result is descriptive. It measures this generator's detection rate, not all leaks. H1's registered rule does not change.

## Matrix on hourly bars (iteration 3)

Fixed on 2026-10-29, before any Matrix run. It defines H2, H3 and H10–H12 for the development period.

**Panel.**
- Hourly bars of the universe members (`docs/DATA.md`), read through the seal, which keeps the vault out. 233 of the 274 members have a member month in the development period; the other 41 first join in 2025.
- The bars are cut to `known_at` < 2025-01-01 before any feature or label is computed. The validation year's bars are loaded and dropped there, so none reaches a feature, a label or a model, and a label that needs a 2025 bar is null.
- A row (pair, hour t) exists only in the months when the pair is a member of the universe. Membership is the rebalance at the first day of t's month, ranked on data before that month.
  - Rows from other months are neither entries nor queries.
  - A pair's months before it first joins would carry its later selection, a survivorship leak.
- Features: the 15 `FEATURE_COLUMNS`, computed on each pair's whole hourly history up to t, so their windows count hours.
- Labels: three classes over h = 24 bars (one day). Entry is at the open of t + 1 and exit at the open of t + 25. The threshold is max(0.5·σ̂·√24, 0.0024), with σ̂ = `f_vol_30`. A label is known at its exit bar's open.
- Test months: 2019-01 → 2024-12, as H9. Decisions are hourly.

**Matrix.**
- One pooled memory over all members.
  - An entry is (x, y): x holds the 15 features, y the one-hot class followed by `fwd_return`.
  - It is stamped with `label_known_at`.
  - A row with a null feature or label gives no entry.
- A readout at decision time d over window w uses the entries with d − w < time ≤ d. The windows are 30 days, 365 days and all history.
- Ridge in centred form, with a free intercept. Feature j gets the penalty ρ·n_w·σⱼ², where n_w is the window's entry count and σⱼ² the variance of feature j over the entries the fold's fit may use (see Folds). ρ ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1, 10}. This is ridge on features standardized inside the fold, solved exactly from the statistics.
- Probabilities, for every arm: pₖ = max(ŷₖ, 10⁻³) / Σⱼ max(ŷⱼ, 10⁻³).
- A readout the solver refuses (an empty or singular window) falls back to the arm's climatology for that row. The count is reported.

**Folds.**
- Monthly refits, at 00:00 UTC on the first day of each test month.
- A choice fitted at the fit time (σ, a window, ρ, stacking weights, a kernel width or k) uses only:
  - rows whose label is known by the fit time minus 24 h (purge and embargo of h);
  - and those rows' prequential forecasts, which each arm made at the row's own decision time.
- Matrix itself takes every entry once it is known, inside the test month too.

**Arms.** Each arm is one variant of the deflated Sharpe ratio.

| Arm | Hypothesis | Definition |
|---|---|---|
| `climatology-panel` | reference | Class frequencies of the rows known at the fit. |
| `matrix-select` | H2, and the reference of H10–H12 | At each fit, the window and ρ with the lowest mean prequential log-loss over the known rows of the last 365 days. |
| `matrix-1y` | H3 | Window 365 days; ρ selected at each fit the same way. |
| `softmax-1y` | H3 | Nadaraya-Watson on the entries of the 365-day window, weights ∝ exp(−‖u_q − uᵢ‖²/(2τ²)), where u holds the 15 features divided by their fold σ, and τ ∈ {0.5, 1, 2}·√15 is selected at each fit. |
| `knn-1y` | H3 | The mean one-hot label of the k nearest entries of the 365-day window (Euclidean on u), ties broken by entry order, k ∈ {50, 200, 1000} selected at each fit. |
| `matrix-adaptive` | H10 | `matrix-select`'s window. At each decision time d, the ρ with the lowest mean prequential log-loss over the rows whose labels are known by d and whose decisions fall after d − 30 days. |
| `matrix-stack` | H11 | p = Σ_w α_w p_w over the three windows, each at its own selected ρ. α lies on the simplex, fitted at each fit on the prequential log-loss of the known rows of the last 365 days, on a grid of step 0.05. |
| `matrix-cusum` | H12 | `matrix-select`, refitted at the first fit and then only after a CUSUM alarm, instead of every month (detailed below). |

- The CUSUM's input is one value per day d, built from the 00:00 UTC decisions, so the daily labels do not overlap. Each decision's recursive residual of the `fwd_return` readout, its realized `fwd_return` minus the forecast made at the decision time, is divided by its predictive standard deviation, σ̂·√(1 + 1/n_w + x̃ᵀ(M_xx + Λ)⁻¹x̃). Here x̃ is the query's features minus the window's means, M_xx the window's centred cross-products, and σ̂² the window's residual mean square at the decision time. The day's value is the mean over members.
- A two-sided Page CUSUM runs on these values with k = 0.5. Its threshold is set by simulation, before any result, to give an average run length of 30 days under independent N(0, 1) values.
- After an alarm, the arm refits at the next 00:00 UTC and the CUSUM restarts.

If the exact kernel readouts cannot run over the 365-day window within the iteration, H3 runs at the 30-day window for all three of its arms. This fallback is fixed here, before any result.

**Scoring.**
- Scored rows are the test rows with a label and complete features. They are the same for every arm.
- Each score is averaged over the scored rows of each decision hour. Every statistic is computed on these hourly series, so the dependence across pairs stays inside the hour. A reported mean is a mean over hours.
- Metrics: log-loss, Brier, Brier skill score against `climatology-panel`, expected calibration error.
- Diebold-Mariano runs on the hourly loss differences, two-sided, with the rectangular kernel, h − 1 = 23 lags and the Harvey-Leybourne-Newbold correction.
- Intervals: stationary block bootstrap of the hourly series, block length by Politis-White, 2,000 resamples, 95 % percentile.

**Decision rules.**
- **H2** holds if the Brier skill score of `matrix-select` is above 0 and its 95 % interval lies above 0.
- **H3** holds if `matrix-1y` has a lower log-loss than `softmax-1y` and than `knn-1y`, each with Diebold-Mariano p < 0.05. Each comparison is reported.
- **H10** holds if `matrix-adaptive` has a lower log-loss than `matrix-select`, with p < 0.05.
- **H11** holds if `matrix-stack` has a lower log-loss than `matrix-select`, with p < 0.05.
- **H12** holds if two conditions both hold:
  - the 95 % one-sided upper bound of the mean log-loss of `matrix-cusum` minus that of `matrix-select` is below 0.005 times climatology's log-loss. The bound is the mean difference plus the 95 % quantile of Student's t with T − 1 degrees of freedom, times the Diebold-Mariano standard error above (HLN-corrected), where T is the number of hours;
  - `matrix-cusum` refits fewer than 72 times.

**Future invariance on the panel.** Before any result, every arm passes H1 on a sub-panel.
- The sub-panel: the 10 members with the most member months in the development period.
- The cuts: 8 cut hours, drawn with a fixed seed in the test months, one of them at a month's first hour and one at its last.
- Every bar of every member after the cut is perturbed in both directions, as in H1. Every output at or before the cut is compared bit for bit.
- Planted online leaks must each be caught where predicted:
  - entries stamped with their decision time instead of `label_known_at`;
  - a window that ends one hour after the decision;
  - σ fitted on the whole test month;
  - a fit-time selection on rows whose labels are not yet known (no purge);
  - a kernel readout whose candidates reach one label horizon past the decision.

**The readout, described.** For a seeded sample of 1,000 test decisions of `matrix-1y`, the report gives the share of negative weights. It also gives the signed contribution of the last 30 days, and that of the older entries.

**Out of scope here.** Trading metrics of the Matrix arms wait for a portfolio simulator. Each arm still counts as a variant.

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
- 2026-10-29, before any run of iteration 3:
  - The design of "Matrix on hourly bars (iteration 3)" is fixed: panel, arms, folds, scoring and decision rules for H2, H3 and H10–H12. It includes a fallback for H3 to the 30-day window, should the one-year kernels not run in time.
  - H1 gains the adversary, a descriptive measure. Its registered rule does not change.
  - Two clarifications of the panel, the same day and before any Matrix run. They come from building it, and change nothing in the design.
    - 233 of the 274 members have rows in the development period; the other 41 first join in 2025.
    - The validation year's bars are loaded through the seal, then cut before features and labels. The earlier wording said they were not read at all.
  - The physical seal of the vault months, in "Splits": a vault tree outside `data/` (stage 1) and its age encryption (stage 2). The downloader and `build_parquet` now refuse vault months. No split or period changes.
