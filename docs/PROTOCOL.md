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
    - prices, body, wicks and range, except the upper wick at odd cuts, which crosses bar t's only;
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
  - Comparisons two or more bars ahead flip with high probability only, not by construction. So do comparisons with bars older than t − 90, the shape functions other than the fill, the close location and colour with size, such as the ratio of the two wicks, and their pairings with the colour. So does, at odd cuts, a comparison of bar t + 1's upper wick with a bar older than t.
  - A threshold on bar t + 1 flips by construction only inside the design's bands: a body between 0.4 % and 10 %, a range between 0.5 % and 50 %, a fill between 0.2 and 0.8, a close location between 0.25 and 0.75.
  - A cut where a quantity cannot cross is marked `?`, for example a taker share already at 1, or a bar without volume. The `?` is predicted from the original bars, before perturbing, and only for a declared reason. It is reported and left out of the verdict; it never counts as a pass. A failure to cross without a declared reason fails the test, and so does a control that is `?` at every cut.

**The adversary (iteration 3).** Beside the designed controls, H1 reports how many random leaks it catches.
- A seeded generator plants random leaks: functions of the next bars' levels, ratios and shapes, alone or combined, and compared with the past, with other future bars or with thresholds.
- Each leak runs through H1's own perturbation, at H1's cuts and at a longer schedule of cuts.
- The report gives the share caught with a 95 % Wilson interval, overall, per family and per cut, and the mean per-cut rate. It also gives the size of each catch and the list of misses.
  - The sizes describe the scale of the generator's outputs, not H1's sensitivity. So the report also gives an attenuation curve: the share caught when every leak's output is multiplied by ε ∈ {1, 10⁻², 10⁻⁴, 10⁻⁶, 10⁻⁸, 10⁻¹⁰}. It shows where the 10⁻⁹ rule stops catching weak leaks.
  - It names the families the generator never produces, such as fitted leaks, reads more than 10 bars ahead and leaks that read the time stamps.
- Every leak has an honest twin, the same expression moved into the past. A twin must never be caught: one catch is a bench bug. The report gives the share of twins that are constant at the compared rows, since a constant twin cannot be caught.
- The result is descriptive. It measures this generator's detection rate, not all leaks. H1's registered rule does not change.

## Matrix on hourly bars (iteration 3)

Fixed on 2026-10-29, before any Matrix run. It defines H2, H3 and H10–H12 for the development period.

**Panel.**
- Hourly bars of the universe members (`docs/DATA.md`), read through the seal, which keeps the vault out. 233 of the 274 members have a member month in the development period; the other 41 first join in 2025.
- The bars are cut to `known_at` < 2025-01-01 before any feature or label is computed. The validation year's bars are loaded and dropped there, so none reaches a feature, a label or a model, and a label that needs a 2025 bar is null.
- A row (pair, hour t) exists only in the months when the pair is a member of the universe. Membership is the rebalance at the first day of the month in which bar t opens, ranked on data before that month. A row's month is the month of its bar's open, even when a late close puts its decision time in the next month.
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
- **σ.** The σⱼ² of a month M is the variance of feature j over the entries known by M's first instant minus 24 h. Every readout of a row of month M uses the σ of M, in the test months and in the months before them alike.
- **The prequential record.** Every candidate of every arm (each window with each ρ, each τ, each k) forecasts every row, at the row's own decision time and with the σ of the row's month. The record starts at the first month that has a σ, 2017-11. Rows decided before 2019-01 are never scored: they only build the record.
- **The selection rows** of a fit are the rows decided in the 365 days before it whose labels are known by the fit time minus 24 h (purge and embargo of h).
- A choice fitted at the fit time (σ, a window, ρ, stacking weights, a kernel width or k) uses only rows whose label is known by the fit time minus 24 h, and those rows' forecasts in the record. Every selection minimizes the mean log-loss of the record over the selection rows.
- **Defaults.** A selection, at a fit or at a decision of `matrix-adaptive`, that finds fewer than 1,000 rows to read takes a fixed default instead: the 365-day window, ρ = 10⁻², τ = √15, k = 200, and all of α on the 365-day window. The count of defaults is reported per arm.
- Matrix itself takes every entry once it is known, inside the test month too.

**Arms.** Each of the eight arms is one variant of the deflated Sharpe ratio, `climatology-panel` included: 8 variants, 18 in all with the 10 baselines of Report n°1. The grids inside an arm are fitted inside the fold, so they are not variants.

| Arm | Hypothesis | Definition |
|---|---|---|
| `climatology-panel` | reference | Class frequencies of the rows known at the fit. |
| `matrix-select` | H2, and the reference of H10–H12 | At each fit, the window and ρ with the lowest mean prequential log-loss over the selection rows. |
| `matrix-1y` | H3 | Window 365 days; ρ selected at each fit the same way. |
| `softmax-1y` | H3 | Nadaraya-Watson on the entries of the 365-day window, weights ∝ exp(−‖u_q − uᵢ‖²/(2τ²)), where u holds the 15 features divided by their fold σ, and τ ∈ {0.5, 1, 2}·√15 is selected at each fit. |
| `knn-1y` | H3 | The mean one-hot label of the k nearest entries of the 365-day window (Euclidean on u), ties broken by entry order, k ∈ {50, 200, 1000} selected at each fit. |
| `matrix-adaptive` | H10 | `matrix-select`'s window. At each decision time d, the ρ with the lowest mean prequential log-loss over the rows whose labels are known by d and whose decisions fall after d − 30 days. |
| `matrix-stack` | H11 | p = Σ_w α_w p_w over the three windows, each at its own selected ρ. α lies on the simplex, fitted at each fit on the prequential log-loss of the selection rows, on a grid of step 0.05. |
| `matrix-cusum` | H12 | `matrix-select`, refitted at the first fit and then only after a CUSUM alarm, instead of every month (detailed below). |

The CUSUM of `matrix-cusum`:
- **Input.** One value per day d, from the decisions of the bars that open at 00:00 UTC on d (decision time 00:59:59.999), so the daily labels do not overlap.
  - Each decision's recursive residual of the `fwd_return` readout, its realized `fwd_return` minus the forecast made at the decision time, is divided by its predictive standard deviation, σ̂·√(1 + 1/n_w + x̃ᵀ(M_xx + Λ)⁻¹x̃). Here x̃ is the query's features minus the window's means, M_xx the window's centred cross-products, and σ̂² the window's residual mean square at the decision time.
  - The day's raw value is the mean of these over members. It is divided by s, the standard deviation of the raw daily values over the days of the last fit's selection rows, recomputed with that fit's window, ρ and σ, as the fit reads its live days. The members' residuals move together, so the raw mean does not have unit variance: s gives back the unit variance that the threshold assumes.
- **Timing.** Day d's value is known with its labels, at the exit open, 01:00 UTC on d + 1, and it enters the CUSUM then. An alarm acts at the first 00:00 UTC after that instant. The arm refits there, on rows whose labels are known by the fit time minus 24 h, and the CUSUM restarts at 0.
  - A day's value is computed entirely with the fit in force at its decision time: its readouts, σ̂, leverage and s. The restarted CUSUM takes only the days decided after the refit, so the day decided just before it is left out.
- **Threshold.** A two-sided Page CUSUM with k = 0.5. Its threshold is set by simulation, before any result, to give an average run length of 365 days under independent N(0, 1) values. That is about 6 false alarms over the 2,191 test days, against 72 monthly fits.

If the exact kernel readouts cannot run over the 365-day window within the iteration, H3 runs at the 30-day window for all three of its arms. This fallback is fixed here, before any result.

**Scoring.**
- Scored rows are the test rows with a label and complete features. They are the same for every arm. The report gives the share of the test months' hours that are scored, and why the others are not.
- Each score is averaged over the scored rows of each decision hour. Every statistic is computed on these hourly series, so the dependence across pairs stays inside the hour. A reported mean is a mean over hours.
- Metrics: log-loss, Brier, Brier skill score against `climatology-panel`, expected calibration error.
- Diebold-Mariano runs on the hourly loss differences, two-sided, with the rectangular kernel, h − 1 = 23 lags and the Harvey-Leybourne-Newbold correction.
- Intervals: stationary block bootstrap of the hourly series, block length by Politis-White, 2,000 resamples, 95 % percentile.
  - For the Brier skill score, the arm's and climatology's hourly Brier series are resampled with the same blocks. Each resample gives 1 − mean(b_arm*)/mean(b_clim*), and the Politis-White length is computed on the hourly difference b_arm − b_clim.
- A comparison of two arms has a second test: the same bootstrap of the mean hourly log-loss difference, with the Politis-White length of that difference. Diebold-Mariano with 23 lags covers the overlap of the labels exactly, but not a dependence that lasts longer, such as a skill that changes with the regime.
- Each difference is also reported as a share of climatology's log-loss, beside its p-value.
- Each comparison also gives the share of hours at which the two arms' hourly log-losses are identical, because both read the same cells at every scored row, and the count of the other hours, the informative ones. Above one half, the report says that the tests rest on few informative hours. This is a description and changes no verdict.
- The test months that smoke runs scored before the official run, 2019-01 → 2020-06, are also reported apart from the other 54. The Brier skill score of `matrix-select` and every comparison are computed on each part alone, with the same statistics and each part's own climatology. It is a description: the verdicts rest on the whole period, and a part's comparison says whether the rule is met, never that it passes. A part in which two arms agree at every hour has no test, and the report says so.

**Decision rules.** A comparison passes when the Diebold-Mariano test gives p < 0.05 and the bootstrap's 95 % interval of the difference lies on the same side of 0.
- **H2** holds if the Brier skill score of `matrix-select` is above 0 and its 95 % interval lies above 0.
- **H3** holds if `matrix-1y` has a lower log-loss than `softmax-1y` and than `knn-1y`, each comparison passing. Each comparison is reported.
- **H10** holds if `matrix-adaptive` has a lower log-loss than `matrix-select`, the comparison passing.
- **H11** holds if `matrix-stack` has a lower log-loss than `matrix-select`, the comparison passing.
- **H12** holds if two conditions both hold:
  - the 95 % one-sided upper bound of the mean log-loss of `matrix-cusum` minus that of `matrix-select` is below 0.005 times climatology's log-loss. The bound is the mean difference plus the 95 % quantile of Student's t with T − 1 degrees of freedom, times the Diebold-Mariano standard error above (HLN-corrected), where T is the number of hours. The bootstrap's 95th percentile of the mean difference must be below the margin too;
  - `matrix-cusum` makes at most 36 fits, the first included: half the 72 of the monthly schedule.

**Future invariance on the panel.** Before any result, every arm passes H1 on a sub-panel.
- The sub-panel: the 10 members with the most member months in the development period.
- The cuts: 8 cut hours, drawn with a fixed seed in the test months, one of them at a month's first hour and one at its last.
- Every bar of every member after the cut is perturbed in both directions, as in H1. Every output at or before the cut is compared bit for bit, with every fit-time choice behind it. The fit of the cut's month is compared when a row of that month is decided by the cut; without one, nothing of that month is compared.
- A member with no bar after a cut, or none at or before it (not yet listed), is left out of that cut, and the count is reported.
- Planted online leaks must each be caught where predicted:
  - entries stamped with their decision time instead of `label_known_at`;
  - a window that ends one hour after the decision;
  - σ fitted on the whole test month, predicted at 7 of the 8 cuts: at a month's last hour, every feature of the month is already known;
  - a fit-time selection on rows whose labels are not yet known (no purge);
  - a kernel readout whose candidates reach one label horizon past the decision.
- A limit: only the sub-panel's bars are perturbed. An input built from other members' bars would not move. No arm has such an input.
- A second limit: a member not yet listed at a cut stays unperturbed, since all its bars lie after the cut. An arm that read such a member's future bars would not be caught at the cuts before its listing.

**The readout, described.** For a seeded sample of 1,000 test decisions of `matrix-1y`, the report gives the share of negative weights. It also gives the signed contribution of the last 30 days, and that of the older entries.

**Out of scope here.** Trading metrics of the Matrix arms wait for a portfolio simulator. Each arm still counts as a variant.

## The encoder, and what a verdict is worth (iteration 4)

Registered on 2026-10-29, before any run of iteration 4, and audited before any run (see the change log). It defines the encoder arms of H4, H5 and H13, the size and power study of the pass rule, `olsprime.explain`, H1's blind spots, and H14, which is registered now and runs in iteration 5. Every choice below is the same for every arm, and nothing is tuned per arm.

**Priority order**, fixed before any result. A hypothesis that was not run is reported as not run.
1. These pre-registrations, and an audit pass on them before any run.
2. Report n°2's series, then the size and power study.
3. `olsprime.explain`: `why` and `head`.
4. The encoder: H4's three arms and H5, with the cost table and H13. LFM3's two ablations, then H6, run only if time remains, in that order.
5. H1's blind spots: the future-shuffle twins only.
6. Deferred to iteration 5, registered now: the portfolio simulator (H14) and H1's extreme-shape cut.

### The encoder arms

**Panel and scoring.** The panel, test months, scored rows, hourly series, statistics and decision rule of "Matrix on hourly bars (iteration 3)". Every encoder arm forecasts every scored row of Report n°2, so its hourly series pairs with Report n°2's hour by hour.

**Inputs: a window of bars.** Row (s, t) reads the 64 hourly bars of pair s that end with bar t. They are all final at its decision time, so attention inside the window may be bidirectional.
- Each bar j gives eight quantities:
  - `ret` = ln(c_j / c_{j−1}) and `gap` = ln(o_j / c_{j−1});
  - `range` = ln(h_j / l_j) and `body` = ln(c_j / o_j);
  - `upper` = ln(h_j / max(o_j, c_j)) and `lower` = ln(min(o_j, c_j) / l_j);
  - ln(1 + quote volume);
  - the taker share of the quote volume minus 1/2, which is 0 for a bar without volume.
- Each quantity is standardized with its mean and standard deviation over the last bar of every training row of the fit, then clipped to ±8. These statistics are fitted at the fit (see Fits).
- The window's 65 bars, from t − 64 for the first return, must be consecutive hours. A complete row always has them, since its 90-bar features require 90 gaps of one hour. A scored row whose window fails takes the arm's climatology, and the report counts it.

**Encoder**, shared by the three arms, which differ only in the attention operator:
- a linear map of the eight quantities to d_model = 32, plus a learned embedding of the 64 positions;
- 2 pre-LayerNorm blocks:
  - attention, with 4 heads of d_head = 8;
  - an MLP 32 → 64 → 32 with GELU;
  - a residual connection around each, and no dropout;
- a final LayerNorm. The state h is the output at bar t, the last position.

**The attention operators.** In every arm, Q, K and V are a head's 64 × 8 projections, made by the same weights.
- `enc-ols`, the OLS head of `docs/DESIGN.md` §4:
  - Queries and keys are scaled to unit root mean square at each position, with no parameter.
  - K̃ = [1 | K] and q̃ = [1 | q]: an unpenalized constant key coordinate, and a 1 in every query.
  - Λ = diag(0, 64λ, …, 64λ), with λ = softplus(θ) + 10⁻³ learned per head and θ initialised so that λ = 0.1.
  - W = Q̃ (K̃ᵀK̃ + Λ)⁻¹ K̃ᵀ and O = W V. It is solved by a Cholesky factorization in fp32, never by an explicit inverse.
  - Every row of W then sums to 1 (`docs/DESIGN.md` §1), so a row sum that is not 1 within rounding is a bug, and `head` checks it.
  - T / d_head = 8.
- `enc-softmax`: PyTorch's native scaled dot-product attention, softmax(Q Kᵀ / √8) V.
- `enc-linear`: φ(Q) (φ(K)ᵀ V) / (φ(Q) φ(K)ᵀ 1), with φ(x) = elu(x) + 1 (Katharopoulos et al., 2020).

**Parameters.** `enc-softmax` and `enc-linear` have the same count, N. `enc-ols` has N + 8: one λ per head, 2 layers × 4 heads. The constant key coordinate is a fixed input, not a parameter. The difference is under 0.1 %, inside the 1 % allowed, and the report gives the exact counts.

**Memory: Matrix over learned keys.** Every arm reads a Matrix whose keys it learns.
- Each row's key is z = W_z h ∈ R⁸ and its value is its one-hot label. The entry is stamped with the row's `label_known_at`.
- A readout at decision time d uses the entries with d − 30 days < `label_known_at` ≤ d.
- It is the centred ridge with a free intercept of `docs/DESIGN.md` §3.2, m = ȳ + β (z − z̄), with Λ = ρ·diag(M_zz) + 10⁻⁶·n_w·I and ρ = 10⁻². This is ridge on keys standardized inside the window. An empty window gives (1/3, 1/3, 1/3).
- The statistics accumulate in fp64 and the 8 × 8 solve is in fp64. Prefix sums of raw moments are differenced in fp64 before centring, never in fp32.

**Decision head.** [h, m] → 32 → GELU → 3 → softmax. The arm's probabilities then pass through the protocol's floor, pₖ = max(pₖ, 10⁻³) / Σⱼ max(pⱼ, 10⁻³).
- The loss is cross-entropy on the softmax output, before the floor. The floor applies to the scored probabilities only.
- The Brier term is H6's ablation, not part of the core arms.

**Training in episodes.** A step is one episode: the training rows decided in one calendar week, Monday 00:00 UTC to the next, over every member.
- **The memory's causality.** Inside an episode, the memory used for a query at decision time d holds exactly the entries with d − 30 days < `label_known_at` ≤ d.
  - The statistics at d are prefix sums over the entries ordered by `label_known_at`, so an outcome joins the memory only once it is known, inside the episode too.
  - The rows decided in the episode are keyed with gradient.
  - The older entries the window needs are keyed with the current weights, without gradient: a cache.
- **Its test.** On a synthetic episode, at every decision time d, the memory must hold the same entries as `Matrix.window_range(d, 30 days)` of the Rust core, and give its readout within rounding.

**Fits.** One fit per test year, at 00:00 UTC on 1 January, 2019 to 2024: 6 fits. A fit at F:
- **trains** on the complete rows whose labels are known by F − 30 days;
- **stops early** on a slice: the complete rows decided in [F − 30 days, F) whose labels are known by F − 24 h. The slice is never trained on, and every training label is known before its first decision, so it is purged;
- **fits** the inputs' standardization on the training rows, and takes its climatology from the labelled rows known by F − 24 h, as `climatology-panel` does;
- **runs** at most 6 epochs. An epoch takes every week of training rows once, in an order drawn from the seed. After each epoch, the slice's mean log-loss is computed in evaluation mode, with the memory as at test time. The fit keeps the epoch with the lowest slice loss, the first on ties;
- **optimizes** with AdamW:
  - learning rate 10⁻³ and betas (0.9, 0.999);
  - weight decay 0.01 on the weights of the linear maps only: the input map, the Q, K, V and output projections, the MLP, W_z and the decision head. Biases, LayerNorm parameters, the position embedding and θ have none;
  - linear warm-up over the first 100 steps, then cosine decay to 10⁻⁴ at the end of epoch 6;
  - the gradient norm clipped at 1;
- **forecasts** every scored row of its year. A row's year is that of its month, the month of its bar's open, as in iteration 3. Its memory takes every entry once its label is known, test rows included, keyed by the frozen encoder.

**Seeds.** Each arm has 3 seeds (1, 2, 3); a seed sets the initialisation and the order of the episodes.
- The arm's forecast is the mean of its seeds' probabilities before the floor, then floored. That mean is the model.
- Each seed's log-loss and their spread are reported as a description.

**Determinism.**
- The settings: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, cuDNN benchmark off and fixed seeds.
- The machine: one named GPU, the RTX 3060 in WSL2, and one build, whose driver, CUDA and torch versions are recorded.
- Colab runs only if the sizing below rules the 3060 out for the whole schedule. Its runs are named apart, and never enter a 3060 column.

**Batches.** Every batched computation outside a training step runs in fixed-size batches of 4,096 rows taken in panel order, by decision time and then symbol. That covers the encoder's forward pass for queries, entry keys and the cache, the slice evaluation and the forecasts.
- The last batch is padded to 4,096 rows with copies of its last row.
- So a row's outputs do not depend on which other rows are in the run. This matters because BLAS and cuBLAS choose their kernels by problem size, and the same row of a product can differ in its last bits between two sizes (audit pass 4a).
- Training steps keep their weekly episodes, which are the same in every run of a fit.

**Sizing, before the schedule.** One fit of an encoder arm on the largest training set (the 2024 fit), measured on a synthetic panel of the real panel's shape before any run on real data. The numbers and the schedule they allow are in `docs/ITERATIONS.md`, iteration 4's brief.

**Smoke runs on the real panel**, if any, train on rows decided before 2019 and forecast no test month. Any run that scores a test month before the official runs is listed in the change log, and its months are shown apart, as in iteration 3.

**Frozen.** Every choice of this section is fixed before the first encoder run on real data. It is committed before the first `explain` call on any hour of 2019–2024.
- After that first call, this section changes only for what cannot be built as written.
- Each such change-log entry cites the line count of `runs/explain/CONTACTS.log` at that moment.

### H4, H5 and H13

- **H4** holds if `enc-ols` is non-inferior to `enc-softmax` over the test months 2019-01 → 2024-12.
  - **The margin** is 0.005 × the mean hourly log-loss of Report n°2's `climatology-panel` over the scored hours.
  - **Two bounds** of mean(ll_ols − ll_softmax) must both lie below the margin:
    - its one-sided 95 % upper bound: the mean plus t₀.₉₅(T − 1) times the HLN-corrected Diebold-Mariano standard error, as for H12;
    - the stationary bootstrap's 95th percentile of the same mean.
  - **Testable only if an arm learned something.** `enc-ols` or `enc-softmax` must have a Brier skill score against `climatology-panel` whose 95 % interval lies above 0 over the test months. Two arms that both sit at climatology would meet the bounds vacuously, so without that skill H4, and with it H13, are reported as not testable, not as holding (audit pass 4a).

  `enc-linear` is a reference arm. Its comparisons with `enc-ols` and `enc-softmax` are described.
- **H5** holds if `enc-ols` has a lower log-loss than `matrix-select`, the comparison passing: Diebold-Mariano p < 0.05, and the bootstrap interval below 0.
  - `matrix-select`'s series is read from Report n°2's `series.npz`, by its digest, on the same scored hours.
  - **A limit.** The encoder refits yearly and `matrix-select` monthly, so H5 compares two systems, not two attentions at equal cadence.
- **H13** holds if H4 holds **and** the median wall time per 10⁴ decisions of `enc-ols`, over 10 repeats, is at most 0.9 × `enc-softmax`'s.
  - A decision is the whole forward pass: encoder, memory readout and head, with the memory's statistics already built.
  - The conditions: T = 64, d_head = 8, 4 heads, 2 layers, inference batches of 4,096 rows, the RTX 3060 in WSL2, and the driver, CUDA and torch versions recorded.
  - Both arms run in plain fp32 eager mode with no fused attention kernel. The softmax goes through the math backend of scaled dot-product attention. The deterministic settings are on, as in the runs.
  - **The timed rows.** A sample of 12,288 scored rows (3 batches of 4,096), drawn with the seed 20261030. The time is scaled to 10⁴ decisions.
  - **The protocol of a repeat.** 3 untimed warm-up batches, then the 3 timed batches, with `torch.cuda.synchronize()` before and after each.
  - **The repeats** alternate the arms, ABAB, against drift of the clock and the temperature.
  - **Reported:** the median ratio, the 10 repeats' minimum, median and maximum for each arm, and, as a description, the attention operator's own time beside the whole pass.
  - Analytic FLOPs favour OLS by construction when T ≫ d_head, so they stay descriptive.
- **The cost table** is descriptive. It covers every arm of H4 and H5 and the eight Matrix arms, with five columns:
  - parameters;
  - analytic FLOPs per decision;
  - wall time per 10⁴ decisions;
  - peak memory: `torch.cuda.max_memory_allocated` on GPU, the process's peak on CPU;
  - GPU-hours of training, with the GPU named.

  Each column holds one machine and one build: GPU timings on the 3060 in WSL2, Matrix's CPU timings on the Ryzen 5 3600 with the portable build. A Colab T4 or A100 is never mixed into a 3060 column.

### H1 for the encoder arms

**On GPU.** Each encoder arm runs the panel H1 of iteration 3 with its first seed: the same sub-panel, the same 8 cuts, both directions, plus the purge cut below.
- **The runs.**
  - The honest run trains, on the sub-panel, the fit of each cut's fit year, 2021 for the purge cuts. It forecasts every scored row of that year.
  - Each perturbed run rebuilds the sub-panel and its windows from the perturbed bars, retrains the fit of the cut's year on them, and forecasts up to the cut.
- **What is compared, at or before the cut:**
  - the fit's inputs: training and slice rows, standardization, climatology and the order of the episodes;
  - every trained parameter, each epoch's slice loss and the epoch kept;
  - the memory's stamps and statistics at each compared decision;
  - every output.
- **Pre-check, before H1.**
  - Two honest runs with identical inputs.
  - A scope test: the honest run restricted to the rows decided at or before the first cut must give the full honest run's outputs on those rows.
  - If both agree to the bit on everything above, H1 is bit for bit, as for the Matrix arms.
- **Fallback, registered now in case they do not** (rewritten after audit pass 4a).
  - The registered H1 for the encoder arms then runs on CPU torch with the same settings, where it must be bit for bit.
  - The GPU run is reported beside it as a description:
    - everything computed outside the training loop must still match to the bit: the inputs, the standardization, the climatology, the training, slice and selection rows, the order of the episodes, and the memory's stamps and membership;
    - the floor is the largest pairwise difference over three honest runs, quantity by quantity;
    - each control's smallest catch is given as a multiple of the floor, and a catch under 100 × the floor is called weakly separated.
  - A failed pre-check on the GPU is the first finding to report, before any H4 number.
  - This rule is fixed now, never after a violation.
- **The budget.** If the sizing projects this H1 beyond 12 hours, the planted controls run only at cuts 1, 3 and 8, in time order, and at the purge cuts. The honest arms run at every cut.
  - Under the CPU fallback, the honest arms' retrains alone may exceed the budget. Then the honest arms retrain only at cuts 1, 3 and 8 and at the purge cuts.
  - At the other cuts, they forecast up to the cut with the honest fit's parameters loaded, without retraining. The comparison there covers the memory and the outputs only, and the report says so.
- **Limits.**
  - H1 tests seed 1's pipeline, while the model is the ensemble of three seeds.
  - The honest runs' forecasts are a first look at the encoder's skill on the sub-panel's cut years. That contact is listed in the change log when it happens, and Report n°3 says so.

**The purge cuts.** Two more cuts, whose perturbed bars come before the 2021 fit, at F = 2021-01-01T00:00 UTC. They are the only cuts that can see a fit read labels too early.
- At each, the perturbed run rebuilds the sub-panel from the perturbed bars and retrains the 2021 fit.
- No output is compared, since every one is decided after F.
- **At F − 24 h, 2020-12-31T00:00 UTC.**
  - Compared: the 2021 fit's inputs, every trained parameter, each epoch's slice loss and the epoch kept.
  - Every input of an honest fit is known by F − 24 h, so every honest arm is predicted invariant.
  - So are `cheat-stamp` and `cheat-value`: in that fit, their leaks read only labels known by F − 30 days in training, and by F − 24 h in the slice.
- **At F − 30 days, 2020-12-02T00:00 UTC** (audit pass 4b's bolder check).
  - Compared: the training rows, the standardization, the order of the episodes, and the trained parameters after every epoch.
  - Not compared: the slice's labels and losses, the epoch kept and the climatology. They legitimately read labels known after F − 30 days.
  - Every training input of an honest fit is known by F − 30 days, so every honest arm, `cheat-stamp` and `cheat-value` are predicted invariant there.
- Together the two cuts bracket the purge window on real data. The planted-label test of `test_train.py` brackets it on synthetic data.

**Planted training-time leaks.** These are controls, not variants. Each runs with its first seed.
- **`cheat-stamp`, the panel H1's sixth online leak.** `enc-ols` with its memory prefix-summed by decision time: an entry joins at its row's decision time, before its label is known, in training and at test time.
- **`cheat-value`.** `enc-ols` whose last position, in every attention head, adds a learned projection of the row's own one-hot label to its value.
- **`cheat-purge`** (after audit pass 4a). `enc-ols` whose 2021 fit trains on the rows whose labels are known by F, with no purge.
  - It runs at the two purge cuts only.
  - It is predicted caught at both, in both directions: some of its training rows have exit bars that open after each cut.
- **`cheat-slice`** (after audit pass 4b). `enc-ols` whose 2021 fit trains on the rows whose labels are known by F − 24 h, its own slice included.
  - It runs at the two purge cuts only.
  - It is predicted caught at F − 30 days, in both directions, and invariant at F − 24 h.
- **`cheat-stamp-day`** (added on 2026-10-30, before any run on real data; see the change log). `enc-ols` whose memory holds only the last 24 hours by decision time, (d − 24 h, d], with every entry stamped at its decision time, in training and at test time.
  - Every entry it reads has a label not yet known at d.
  - It runs where `cheat-stamp` runs, with the same predicted-pattern rule and the same skill check.
  - It shows whether H1 catches a stamp leak that gives skill.
- **Predicted patterns.** They are computed per cut, as iteration 3's are. `cheat-stamp` and `cheat-value` are predicted caught at a cut when some compared row has a label whose exit bar opens after the cut. At the eight cuts this is expected to give `XXXXXXXX` for both; the computed prediction is the one tested.
- **Their skill.** `cheat-stamp` and `cheat-value` must also show a Brier skill score far above every honest arm's: at least 0.1, with a 95 % interval above every honest arm's interval. So must `cheat-stamp-day`. It is measured on the scored rows of the sub-panel in the cuts' years, which H1's honest run forecasts for every arm.
- **Three verdicts, reported apart** (made precise on 2026-10-30, before any run on real data).
  - H1, as `docs/HYPOTHESES.md` defines it: every honest arm bit for bit at every cut, and every control caught on its predicted pattern.
  - The skill check, control by control.
  - `pos-window`'s check, below.

  Each is reported as passing or failing, and none stands in for another.
- **The first finding.** If H1 cannot catch `cheat-stamp` bit for bit, on the GPU or under the fallback, that is the first finding to report, before any H4 number.

**The positive control, `pos-window`** (audit pass 4a's bolder check). `enc-ols`, seed 1, trained and scored on a label that lies inside its own window: the class of ln(c_t / c_{t−24}), under the panel's label rule with the row's own threshold.
- It runs the honest fits of the cut years on the sub-panel, and is scored on the same rows as the cheaters' skill, against the climatology of its own label.
- Its entries keep the panel's stamp, `label_known_at`, the open of the exit bar, although its own label is known earlier. Stamped at its decision time, a row would sit in its own memory, as in `cheat-stamp`, and inflate the skill.
- It must reach a Brier skill score of at least 0.5. That shows that the training loop learns a signal that is there.
- If it fails, that is reported before any H4 or H5 number.
- It is a control, not a variant.

### LFM3's ablations and H6, if time remains

Each is `enc-ols` with one change, three seeds and the same schedule. Each counts as a variant.
- **`enc-ols-gate`.** A sigmoid gate per OLS head, after LFM3: O ← σ(x W_g) ⊙ O, where W_g maps the block's normalized input to the head's 8 outputs.
- **`enc-ols-conv`.** A conv-first hybrid: the first block's attention is replaced by a gated short convolution, a depthwise causal convolution of width 3 over the window gated by a sigmoid. The second block keeps its OLS attention.
- **`enc-ols-brier`, H6.** The loss is cross-entropy plus the Brier score, with weight 1. H6 holds if both conditions hold (made precise after audit pass 4a):
  - its ECE is at most 0.8 × `enc-ols`'s, and the paired stationary bootstrap's 95 % upper bound of the ratio of the two ECEs is below 1;
  - its log-loss exceeds `enc-ols`'s by no more than a tolerance: the larger of two numbers, the range (max − min) of `enc-ols`'s three per-seed log-losses and the HLN-corrected Diebold-Mariano standard error of the difference.

  The LFM3 arms are described against `enc-ols`, with no registered rule.

### Size and power of the pass rule

The pass rule does not change. This study describes what a verdict under it is worth.
- **Its input.** It reads `series.npz` of a named run of Report n°2, by its SHA-256, and refits nothing.
- **Size.** For each registered comparison, take the hourly difference d.
  - Centre it, d − d̄. Then sign-flip it in circular blocks of L hours, for L ∈ {24, 48, 168, 720}, with 2,000 replicates per L. A replicate rotates the hours by a random offset in [0, L), cuts them into consecutive blocks of L, and multiplies each block by an independent random sign.
  - A block flip keeps the lag-k autocovariance with weight (L − k)/L. So L = 24 keeps only about two thirds of the long-run variance of the 24-hour overlap, and cannot show an over-rejection caused by the overlap itself. L = 48 was added for that, after audit pass 4a.
  - The pass rule runs on each replicate as registered. The Politis-White block length is recomputed on that replicate. Only the bootstrap's RNG seed is fixed across replicates: it is the registered seed of the comparison. The study's intervals are therefore conditional on that seed.
  - The study reports, per comparison and per L, with 95 % Wilson intervals, the rejection rate of each of these:
    - Diebold-Mariano alone;
    - the bootstrap alone;
    - both on the same side, the pass rule;
    - the pass rule in its registered direction.
  - For H12's non-inferiority rule, the size is the rate at which it holds when the true difference sits on the margin: (d − d̄) + m.
  - Any L under which the pass rule's two-sided size exceeds 7.5 % is named.
- **Power, model-free.**
  - The replicates are flip(d − d̄) + δ, with δ in the registered direction. The grid of δ is {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0} % of climatology's log-loss, with 1,000 replicates per point.
  - The study reports power(δ) and the smallest δ on the grid with power at least 80 %.
  - A constant shift is the easiest gain to see, so that detectable δ is a lower bound of the detectable mean gain. A gain concentrated in regimes needs more.
  - So a second grid, at L = 168 only, concentrates the shift. The series is cut into 720-hour blocks, rotated by a random offset. Exactly round(0.1 × the number of blocks) of them, drawn without replacement, get a shift scaled so that the mean shift over the series is δ, about 10 δ; the others get nothing. It reports the same power curve beside the constant one.
  - H3's `matrix-1y` − `knn-1y`, −0.094 %, is then read against those detectable sizes.
- **Seed.** The flips use one registered seed, 20261030.

### `olsprime.explain`

An instrument for the orchestrator, read by Claude.
- **Output.** Compact, bounded and reproducible: JSON lines and a short text summary.
  - JSON is written with sorted keys and no wall clock.
  - Every answer carries its provenance: commit, manifest digest and panel fingerprint.
- **The log of contacts.** Every call appends one line to the git-tracked `runs/explain/CONTACTS.log`: UTC time, subcommand, period touched, arm and commit.
- **The default period** is 2017-11 → 2018-12, the only development hours that are not test hours. Any other period must be named in the call, with an explicit flag, or the call is refused.
  - `series` and `sample` read `series.npz`, which holds test rows only, so they touch test months by construction. They are logged like any call.
- **Its reach.** It reads the development panel only, 2017-10 → 2024-12. The validation year and the vault are out of its reach.
- **Exploratory, never tested.** A pattern seen through `explain` on 2019–2024 hours is exploratory. It is never tested as a hypothesis on those months: it waits for the validation year's scheduled opening, or for the vault.
- **`why <pair> <as_of> --arm --by {year,month,day} --top k`** gives:
  - the forecast, the intercept and the cell's status;
  - the contributions of the memory's tree nodes at that granularity, sorted by |c|, with the rest summed so that the parts add up exactly. The grouping is by entry stamp (`label_known_at`), not by decision time, and the output says so;
  - the per-feature view βⱼ·z̃ⱼ. It sums to the forecast within rounding. Summed in the core's order, it can give the forecast to the bit: the output says which, as a fact checked at that call;
  - the top-k entries by |w|, with their labels and feature deltas;
  - diagnostics: Σ|w|, the negative mass, the effective sample size 1/Σw² and the leverage.
- **`why`'s two loud checks.** A failure stops the call.
  - Each weighted entry's `label_known_at` is recomputed from its panel row, and from the raw bar's `open_time` for the top-k, and must be at or before `as_of`. Matrix's own stamps are never trusted for this.
  - The forecast must equal the arm's probability in `series.npz` to the bit. A row that is not scored, such as every row of the default period, has no such check. The output then says so, and only the first check applies.
- **`series <arm> [<arm_b>] --period --by {year,month} --top k`** reads only a run's `series.npz`, by its digest. It shows how one of two things is spread over hours:
  - an arm's Brier skill against climatology;
  - or the log-loss difference of two arms.

  It gives the share carried by the top 1 %, 5 % and 10 % of hours, the parts by year or month, and the k hours that carry the most.
- **`sample <arm> --period --n --seed`** runs `why`'s diagnostics on a random sample of n scored rows of the period, drawn by the seed alone, without the per-entry lists. For each row it gives:
  - the negative mass, the effective sample size, the leverage and Σ|w|;
  - the per-feature parts;
  - the row's log-loss.
- **`head`**, once the encoder exists, gives per OLS head:
  - the row of W;
  - its diagonal mass, which shows a head that copies its input;
  - its effective rank;
  - its row sum, 1 by construction.

### H1's blind spots

- **The twins.** Each random leak of the adversary gets K = 8 seeded future-shuffle twins at each cut. In a twin, each of the pair's bars after the cut keeps its time stamps. It takes every other value of a bar drawn, with replacement, from the same pair's bars at or before the cut: open, high, low, close, volume, quote volume, trades, and the taker base and quote volumes. The draws use one registered seed, 20261030.
- **Moved by a twin.** A twin moves a leak when, at a compared row, one of its outputs differs by more than 10⁻⁹, or turns missing, or stops being missing. That is H1's own catch rule.
- **The misses.** A leak caught at no cut is classed as one of three kinds:
  - constant at the compared rows;
  - unchanged under all K twins at every cut: on these bars it does not depend on the future;
  - a true blind spot: some twin moves it, yet H1 did not catch it.
- **Descriptive.** The result measures this generator's misses; H1's rule does not change.
- **Deferred to iteration 5:** the extreme-shape cut. It will be a separate cut type, with its own predicted pattern for every existing control.

### H14, registered now, run in iteration 5

- **The claim.** At Coinbase's and Kraken's entry-tier taker fees plus 2 bps of slippage, the long/flat portfolio of `matrix-select` beats always-flat, and not only through the market's drift. Both conditions must hold, at each venue (the second added after audit pass 4a):
  - **Above flat.** Its net annualized Sharpe ratio has a 95 % stationary-bootstrap interval above 0. Always-flat earns 0 every hour, and cash earns nothing, so beating it means a Sharpe ratio above 0.
  - **Above random positions.** A portfolio long most of the time would inherit the market's drift: buy and hold had a Sharpe ratio interval above 0 in Report n°1. So a second interval must also lie above 0, against 100 random long/flat portfolios (made precise after audit pass 4b):
    - For each member, a random portfolio is a two-state chain on the hourly grid. Its entry and exit probabilities are set so that its expected share of hours long and its expected number of entries per hour equal those of `matrix-select`'s portfolio for that member over the test months.
    - Its positions are sized and costed by the same rule.
    - The 100 portfolios use the seeds 0 … 99, derived from 20261030.
    - The paired 95 % stationary-bootstrap interval resamples hours jointly across the portfolio's and the 100 random portfolios' net hourly returns. Each resample gives Sharpe(portfolio) − the mean over the 100 of Sharpe(random).
    - Report n°1's `random-turnover` matched turnover only and was summarized by its median seed. This control differs.
  - Buy and hold of the equal-weight member universe is reported beside it.
- **The rule:**
  - the expected return is the `fwd_return` readout of `matrix-select`'s cell;
  - a member is entered when its expected return exceeds the round trip, and exited when it falls below 0;
  - decisions are hourly and re-evaluate every holding, and each executes at the next open;
  - a position opens at min(1/n, 1/10) of the equity at that hour, n being the members with a signal, as far as the cash allows. It is held without rebalancing until it exits;
  - a holding whose pair leaves the universe is closed at the open that follows its last member month, as if the month's last decision were an exit;
  - a position whose bars end is closed at its last close;
  - every entry and every exit pays the taker fee plus slippage.
- **The series.** The initial equity is 10,000 USDT. The portfolio's net hourly returns are annualized with √8760. A portfolio that is never exposed has no Sharpe ratio, and H14 then fails.
- **Assumptions, stated.** Binance's 5 USDT minimum and the BTC and ETH lot steps are applied to every pair.
- **Prior:** fails. Each sizing or threshold choice counts as a variant.

### Variants

18 before this iteration, plus one per encoder arm: `enc-ols`, `enc-softmax` and `enc-linear`, 21 in all.
- A seed ensemble counts once.
- Each ablation that runs counts as one more.
- The planted leaks and the positive control are controls, not variants.

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
- 2026-10-29, after audit pass 1a and before any Matrix run. The audit found three flaws in the design of iteration 3, and nine smaller points.
  - **H12's CUSUM measured nothing.** An average run length of 30 days gives about 72 false alarms over the 2,191 test days, so "fewer than 72 refits" was a coin flip under the null. And the mean of the members' residuals, which move together, does not have the unit variance the threshold assumed. Now the daily value is scaled by its spread over the last fit's selection days, the run length is 365 days, and H12 needs at most 36 fits (`docs/HYPOTHESES.md`).
  - **The first fit's selections could not be computed.** They need prequential forecasts that no arm had made yet. Now every candidate forecasts from 2017-11, with the σ of each row's month. A selection with fewer than 1,000 rows takes fixed defaults, which are counted.
  - **The CUSUM's timing was ambiguous.** One reading refitted an hour before the alarm's labels were known. Now the input, the instant it is known and the instant an alarm acts are fixed.
  - A comparison of two arms now needs the stationary bootstrap's interval as well as Diebold-Mariano, and each difference is reported as a share of climatology's log-loss.
  - Defined more tightly: the selection rows, the paired bootstrap of the Brier skill score, the variant count (8, 18 in all), a row's month (that of its bar's open), and the panel H1's members with no bar after a cut.
  - Stated as limits: the upper wick at odd cuts in H1, and the panel H1's inputs from other members.
  - The adversary's report gains an attenuation curve, the families it never produces, the share of constant twins and the mean per-cut rate.
- 2026-10-29, after audit pass 1b and before any Matrix run: the CUSUM's day that straddles a refit is defined. Its value uses the fit in force at its decision time, and the restarted CUSUM leaves it out.
- 2026-10-29, after audit pass 1c and before any official Matrix run. The audit found no Blocker and no Major point, and six Minor points, all fixed before the runs. No rule, threshold, grid, seed or period changes.
  - **The CUSUM's s** is now read with the fit's own σ, as its live days are. It was read with the σ of each selection row's month: a choice the text left open, and not the faithful one. On two synthetic panels the two differ by 2.4 × 10⁻⁵ and 1.7 × 10⁻⁴ of s.
  - **The panel H1** compares the fit of a cut's month only when a row of that month is decided by the cut. Without one, the run scoped to the cut makes no fit for that month, and the honest system was flagged. The predictions that rest on that fit follow the same rule.
  - **A member not yet listed at a cut** is left out of it too, and the limit this opens is stated.
  - **The reports say more.**
    - The panel H1 marks a run that is not the registered one, as Report n°2 does.
    - Report n°2 lists the days each refit left out of the CUSUM.
    - It gives the Diebold-Mariano kernel and lags behind each comparison.
    - It gives its coverage in hours.
  - **Smoke runs on the real panel came before this entry.** They measured run times and checked the reports' shape:
    - the panel H1 on its sub-panel, with test months 2019-01 → 2019-06;
    - Report n°2 on 5 members, with test months 2019-01 → 2019-03;
    - Report n°2 on every member, with test months 2019-01 → 2020-06, 18 of the 72.

    The full development panel, every member from 2017-10 to 2024-12, was also built with no arm run on it:
    - once for `docs/DATA.md`: its rows, its members and the shares of the three label classes;
    - once to size and time the runs, using no label value. That pass measured each feature's |mean| / std, over the panel and over each month's σ set, from feature moments only. It also counted the rows and kernel pairs each run reads, at the eight H1 cuts too. The official H1 sub-panel was rebuilt to time it.

    Their numbers are not results, and none of the changes above was chosen from them. The official runs score those months again, with the rest.
- 2026-10-29, after audit pass 1d and before any official Matrix run. No rule, threshold, grid, seed or period changes.
  - **The previewed months, apart.** Report n°2 also gives the Brier skill score of `matrix-select` and every comparison on the 18 test months the smoke runs scored, and on the other 54 alone (Scoring). This is fixed before the run, and it describes: the verdicts rest on the whole period.
  - **Where the results are read.** Under its verdicts, the report states that those months were scored before the run, with the same arms and rules. The RESULTS rows of H2, H3 and H10–H12 say it too.
  - The entry above now lists the passes over the full development panel among the contacts with the real data.
- 2026-10-29, after audit pass 1e and before any official Matrix run. No rule, threshold, grid, seed or period changes.
  - **Identical hours.** Each comparison gives the share of hours at which both arms' hourly log-losses are identical and the count of informative hours, with the words "few informative hours" above one half (Scoring; the count after audit pass 1f). Two arms that read the same cells give exactly equal losses. A part where they differ at only a handful of hours would otherwise show an ordinary p-value, with degrees of freedom counting every hour.
  - **A part's words.** A comparison on the previewed months or on the others says whether the rule is met. "Passes" stays the word of the whole period's verdicts.
- 2026-10-29, during the official runs, after the panel H1 and the adversary and before Report n°2's verdicts. No rule, statistic, threshold, grid, seed or period changes.
  - **The first official Report n°2 stopped on an internal check.**
    - The run was at commit `3a22daa`. The arms ran over the whole test period. Then the first ECE, `climatology-panel`'s, differed from `stats.ece` by 1.3 × 10⁻⁹ relative, above that check's fixed tolerance of 10⁻⁹.
    - The cause is rounding. `stats.ece` adds a bin's rows in one sequential sum, and climatology repeats one calibrated forecast over many rows, so the rounding errors add up instead of cancelling.
    - A probe on synthetic data reproduced the gap. A lost row, the defect the check is there for, moves the ECE about 300 times more than the new tolerance.
    - The tolerance is now 4·n·ε, about eight times the worst-case rounding of the two sums over n rows: 1.2 × 10⁻⁹ in absolute terms for 1.4 million rows.
    - The run wrote no file. Its traceback showed one number, `climatology-panel`'s ECE over the test period, 0.0029. No other number of Report n°2 was seen before the re-run, which scores the same arms with the same rules.
- 2026-10-29, after audit phase 2, once Report n°2 had run. Wording only: no rule, statistic, threshold, grid, seed or period changes.
  - **What the ECE cross-check guards.**
    - It guards a row lost or counted twice between the hourly sums and `stats.ece`. Such a row moves the ECE by about |correct − confidence|/n. For a row of climatology, whose confidence is about 0.44, that is 220 to 285 times the tolerance on the panel.
    - A row whose confidence nearly equals its correctness, a confident and right forecast, moves it by almost nothing.
    - It cannot see a row put in another bin. Both paths bin the same way, and a row moved between two bins whose sums share a sign leaves the ECE unchanged.
    - The docstring of `ece_tolerance` says otherwise. It is corrected with the next change to `bench/report2.py`.
  - **The bound.** The two paths do not round alike.
    - `stats.ece` counts each bin's correct rows exactly, and adds its confidences in one sequential sum. Divided by n, that sum's rounding moves the ECE by at most about u·n·mean(confidence), where u = ε/2 is the unit roundoff. That is below n·ε/2.
    - The hourly path adds about 30 rows per cell, then the cells over about 50,000 hours. Its rounding is smaller still: at most about H·u, for H hours.
    - So 4·n·ε is more than eight times the worst case, as long as the mean confidence stays below 0.96. Climatology's is about 0.44.
    - For the run's 1,493,353 scored rows, the tolerance is 1.33 × 10⁻⁹. The entry above rounded the panel to 1.4 million rows and the tolerance to 1.2 × 10⁻⁹. The stopped run's gap was 3.7 × 10⁻¹² in absolute terms: 1.3 × 10⁻⁹ of an ECE of 0.0029.
- 2026-10-29, before any run of iteration 4. "The encoder, and what a verdict is worth (iteration 4)" is added. No rule, statistic, threshold, grid, seed or period of iteration 3 changes.
  - **What it registers:**
    - the encoder arms of H4, H5 and H13, with their fits, seeds, determinism settings and the cost table;
    - the panel H1 for the encoder arms on GPU, with a pre-check, a fallback rule fixed in advance, and two planted training-time leaks, `cheat-stamp` and `cheat-value`;
    - LFM3's two ablations and H6, run only if time remains;
    - the size and power study of the pass rule;
    - `olsprime.explain` and its log of contacts;
    - H1's future-shuffle twins;
    - H14, which runs in iteration 5.
  - **Variants:** 21, with the three new arms.
  - **Report n°2 is re-run once**, to save its hourly series and scored rows in `series.npz`, with their digest.
    - Audit phase 2's four logged items are folded in: the Coverage sentence, `ece_tolerance`'s docstring, H12's own rule in the parts, and a per-file summary in `diff_runs`.
    - Its numbers must not change. The re-run is compared with `runs/2026-10-29-report2/`, and only a fixed, named list of lines may differ.
  - **Contacts with the real data in this iteration** are listed in the entries below, as they happen. None has happened yet. The encoder is sized on a synthetic panel shaped like the real one, whose row counts come from `docs/DATA.md`.
- 2026-10-30, after audit pass 4a and before any run of iteration 4. The audit found five Major points, nine Minor points and seven Nits, and no Blocker. Every one is fixed in the section above. No rule of iteration 3 changes.
  - **H4 could hold vacuously.** Two arms that both sit at climatology meet its bounds. H4, and H13 with it, is now testable only if `enc-ols` or `enc-softmax` shows skill, an interval of its Brier skill score above 0.
  - **The GPU H1 would have flagged honest arms.** BLAS and cuBLAS choose their kernels by problem size. A probe on the design's shapes showed rows of the same product differing by up to 5.7 × 10⁻⁶ in fp32 between batch sizes. The perturbed run forecasts fewer rows than the honest one, so its batches differ.
    - Every batched computation outside a training step now runs in fixed batches of 4,096 rows, in panel order, the last one padded.
    - The pre-check gains a scope test.
  - **The fallback let leaks hide under one draw of noise.** If the GPU is not bit for bit, the registered H1 now runs on CPU torch, where it must be. The GPU run becomes a description, with a floor over three honest runs and catches given as multiples of it. A time budget for the controls is fixed in advance.
  - **H14 was confounded with the market's drift.** It now also needs a Sharpe ratio above that of random positions with the same exposure and turnover. Its sizing, exits at the universe's edge, initial equity, annualization and the never-exposed case are fixed.
  - **`explain` could open a channel into the encoder's design.** Wave 2 starts only after this pre-registration is committed. After the first `explain` call on 2019–2024, the encoder section changes only for what cannot be built as written, and each such entry cites the line count of `CONTACTS.log`. `series` and `sample` touch test months by construction. `why` outside the scored rows runs only its first check.
  - **The purge was invisible to the cuts**, which all fall at least 60 days after the fit they test.
    - A ninth cut, at F − 24 h for the 2021 fit, compares that fit itself.
    - A third control, `cheat-purge`, trains with no purge and is predicted caught there.
  - **The audit's bolder check is added: a positive control, `pos-window`.** It must learn a label that lies inside its own window to a Brier skill score of at least 0.5, or that is reported before any H4 or H5 number.
  - **Made precise:**
    - the controls' predicted patterns, computed per cut;
    - H13's timing protocol;
    - H6's rule, with an interval and a tolerance that three seeds cannot shrink to 0;
    - the twins' bar values and what "moved" means;
    - which parameters take weight decay;
    - the cross-entropy before the floor;
    - a row's year;
    - H4's margin;
    - the H1 runs' seed and their preview of skill, stated as limits.
  - **The power study**
    - adds L = 48, since L = 24 keeps only two thirds of the overlap's own dependence;
    - adds a second power grid whose shift is concentrated in a tenth of the blocks;
    - calls its detectable size a lower bound;
    - says that its intervals are conditional on the bootstrap's seed.
- 2026-10-30, after audit pass 4b, which re-checked the fixes of pass 4a, and before any run of iteration 4. GO to commit this pre-registration and start wave 2, with four Minor points and five Nits, all fixed above.
  - **H14's random positions** were not buildable as written. They are now two-state chains per member, matched in expected exposure and entries to `matrix-select`'s portfolio, 100 of them. The paired bootstrap of the Sharpe difference is defined.
  - **The purge cut saw only a leak past F − 24 h.** A second purge cut, at F − 30 days for the 2021 fit, compares the training and every epoch's parameters. A fourth control, `cheat-slice`, which trains on its own slice, is predicted caught there and not at F − 24 h. This was the audit's bolder check, and it is cheap once the first purge cut exists. The two cuts bracket the purge window.
  - **The budget had no answer** when the honest arms alone exceed it on CPU. They then retrain at four cuts, and only forecast with the honest parameters at the others.
  - **`pos-window` keeps the panel's stamp**, so that its row never sits in its own memory.
  - **Made precise:** the honest run's fit year at the purge cuts; the concentrated grid's exact count of shifted blocks; the twins draw from bars at or before the cut; `explain`'s explicit flag; and H14's change-log line in `docs/HYPOTHESES.md`, since a narrowed claim is still a changed one.
- 2026-10-30, wave 1, before any encoder run on real data and before the first `explain` call (`runs/explain/CONTACTS.log` does not exist yet). The encoder is built and sized on synthetic data only. Nothing registered above changes. Where the section leaves a choice, the build's reading is fixed here, so that no result can choose it.
  - **The attention block.** In every arm, Q, K and V come from one linear map with a bias. One output projection, with a bias, maps the four heads back to d_model.
  - **`enc-ols`.**
    - It computes O = Q̃ ((K̃ᵀK̃ + Λ)⁻¹ (K̃ᵀV)), the product W V taken in another order, with the same Cholesky factor. `head` computes W itself.
    - Queries and keys are scaled to unit root mean square with ε = 10⁻¹².
  - **`enc-softmax`** uses the math backend of scaled dot-product attention in training too, not only in H13's timing.
  - **The memory.**
    - Its key map W_z has no bias: the ridge is centred, so a bias would cancel.
    - Its entries are the complete rows with a valid window, the same rows that train and are forecast.
  - **The standardization** uses the population standard deviation.
  - **The fit.**
    - The slice loss that picks the epoch is the log-loss of the floored probabilities, as the scores read them. Training minimizes the cross-entropy before the floor.
    - All six epochs run. There is no patience rule.
    - The cosine reaches 10⁻⁴ at the last step of epoch 6.
  - **The controls.**
    - `cheat-value`'s label projection is a linear map, so it takes weight decay.
    - `pos-window`'s label sums the window's last 24 one-bar returns in order, in fp32, and classes the sum with the row's own threshold.
  - **Parameters:** 20,995 for `enc-softmax` and `enc-linear`, 21,003 for `enc-ols`. `cheat-value` adds 192.
  - **Checkpoints** are written by hand in the safetensors layout and never go through pickle.
  - **Sizing**, on a synthetic panel of 30 members, hourly from 2017-10 (`docs/ITERATIONS.md`, iteration 4).
    - Two whole 2024 fits of `enc-ols` on the RTX 3060 were bit-identical: every epoch's parameters, the slice losses, the epoch kept and the 253,193 forecasts. The scope test passed at every prefix.
    - So the registered H1 is planned on the GPU. Its own pre-check still runs first, on the real sub-panel.
    - The H1 with every control is projected at about 8.7 hours, under the 12-hour budget, so every cut runs.
  - **Contacts with the real data:** none yet.
- 2026-10-30, before the first `explain` call and before any encoder run on real data (`runs/explain/CONTACTS.log` does not exist yet). The panel H1 for the encoder arms is built and run on synthetic data. A fifth control is added and the verdicts are made precise. No rule is loosened.
  - **What the synthetic bench showed.** On the GPU, the pre-check matched to the bit, every honest arm was invariant at every cut, and the floor over three runs is 0. Every control was caught exactly where predicted, in both directions.
  - **A risk, named before the run.** In a synthetic probe, `cheat-stamp` was caught bit for bit, but its leak gave no measurable skill: a Brier skill score of −0.0009, against −0.0011 for honest `enc-ols`.
    - Its leaky entries are the last day's, about one in thirty of a 30-day window.
    - The synthetic members move independently. The real members move together, so the leak may weigh more on the real panel.
    - The registered check, at least 0.1 with an interval above every honest arm's, stands. Its result is reported as it falls.
  - **`cheat-stamp-day`, a fifth control**, described in "Planted training-time leaks". Its memory is made only of labels not yet known. It shows whether H1 catches a stamp leak that gives skill, whatever `cheat-stamp`'s skill turns out to be. With it, the panel H1 is projected at about 9 hours, under the 12-hour budget.
  - **Three verdicts, reported apart:** H1 as `docs/HYPOTHESES.md` defines it, the skill check control by control, and `pos-window`'s check. A failed skill check is a finding about the control, reported beside H1's verdict, never folded into it or dropped.
- 2026-10-30, after audit passes 4c to 4f and their second looks, before any encoder run on real data and before the first `explain` call (`runs/explain/CONTACTS.log` does not exist yet). No rule, statistic, threshold, grid, seed or period changes. Where the section leaves a choice, the build's reading is fixed here. Descriptions are added beside registered verdicts, and one line of the wave-1 entry is corrected.
  - **Audit pass 4c**, on Report n°2's series, the power study, the twins and the getters: no Blocker and no Major point, three Minor points and three Nits, all fixed before the study runs. GO for Report n°2's re-run and the twins. Its second look gave GO for the power study, with one Minor point and three Nits, all fixed.
    - Every command-line run of the study goes through a pool whose workers use one BLAS thread, so every worker count gives the same bytes. In one process, the main process's multithreaded BLAS moved the last bits of 7 of 5,452 values of `power.json`, never a rate.
    - On Windows a worker's pipe sometimes breaks (WinError 6), in about one test run in four. The pool is then rebuilt and only the calls without a result run again, at most three times. A pool in which no call finishes for 1,800 s counts as broken. Each restart is recorded in `run.json`. A call is a pure function of its arguments, and a pool broken on purpose by the audit gave the same bytes.
    - The study refuses a series whose recomputed comparisons differ from `report2.json`'s by more than 10⁻⁹ relative, instead of only recording it. Every value the comparison records is checked, the count of identical hours included, and so is climatology's mean log-loss.
    - The reader refuses a path outside the runs folder before it reads a byte.
    - The audit's bolder check is added as a description: the variance ratio each block length should give, predicted from the series' own autocovariances, beside the measured one. Each no-flips ratio is given with its standard error, about 14 %.
  - **Audit pass 4d**, on `olsprime.explain`, in three looks: two Major points, seven Minor points and six Nits, all fixed. GO for the five whys.
    - The residual that closed a forecast's parts was zero by construction. It is now bounded, and a residual above its bound stops the call (exit 3).
    - The shared environment held an extension built before `beta` and `mean_y`. It was synced, and every call probes the extension before it reads anything.
    - Each tree's partition is recounted group by group, not only checked by its sum.
    - A Matrix-free twin of the ridge forecast, in numpy, rebuilds each explained ridge forecast within a stated bound (the audit's bolder check).
    - One limit is stated, not fixed: check 1 holds the range of the entries behind σ² to the panel's stamps, not their values.
  - **Audit pass 4e**, on the encoder's core and its panel H1: no Blocker and no Major point, three Minor points and four Nits. GO for the panel H1 and for Report n°3's fits on this core. Its second look, at commit `abbc7d1`, gave GO for the panel H1 on the GPU, with one Minor point, on this entry's wording, corrected here, and five Nits. It trained `cheat-stamp`'s run 1 in four orders, and every part came out byte-identical.
    - A resume is refused if the build changed: torch and its CUDA build, cuDNN, Python, the platform, the GPU and the driver passed by `--driver`, and on CPU the thread count, as well as the config, the code and the manifest. The refusal names each field that changed. The platform holds WSL2's kernel version, so a kernel update between a stop and a resume also refuses the resume.
    - **A limit.** On CUDA the identity does not hold the CPU's thread count, although the memory's readout runs on the CPU there. The audit found the prefix sums and the readout bit-identical with 1, 2, 4 and 6 threads.
    - The registered scope test runs at the first cut. The same test also runs at every regular cut, as a description (below).
    - Torch 2.15's documentation lists `cumsum` on a CUDA floating tensor among the operations that act deterministically under `use_deterministic_algorithms(True)`; older releases listed it among those that raise. On the RTX 3060 it neither raised nor drifted across calls and prefix lengths, and it matched the CPU's sequential sum to the bit on a 5,000 × 100 fp64 probe. The pre-check remains the gate.
    - **A correction of the wave-1 entry.** `pos-window`'s label sums the 24 one-bar returns in fp64, as built, not in fp32. On 38,144 synthetic rows the two readings class every row alike, the closest margin being 308 fp32 ulps.
  - **Audit pass 4f**, on Report n°3: no Blocker, one Major point, seven Minor points and five Nits, all fixed before any fit. Its second look gave GO for the fits and the score, with three Minor points and three Nits, all fixed. It stopped a run at seven points where a file is written, and every resume gave the same bytes.
    - A saved part is checked against its folder: its arm, seed and year, and its checkpoint's and forecasts' own records. A copied folder would otherwise have been scored as another arm (the Major point).
    - A timing file that fails its digest stops the score, instead of reading as absent.
    - Every stage binds the code and manifest digests, and the score refuses two timing files drawn on different samples.
    - `report3.json` lists the fits' builds. Report n°3 is registered only if every fit ran on CUDA with one build, and H13's timing only if it ran on CUDA with the GPU named.
    - The tree is frozen at the fits' commit: the timings, the score and `--check` run at that commit. A fix found in between waits until `--check` has passed.
  - **The panel H1, as built.** Audit pass 4e accepted every reading.
    - At a compared decision, the memory's stamps and statistics are the window's range among the stamp-sorted entries, those entries' stamp, symbol and open time, the window's raw moments and the readout. They are read through a no-op hook in `predict`.
    - Inputs are compared as rows keyed by open time and symbol. The order of the episodes is the weeks' stamps and each epoch's permutation. The step, parameter and missing-window counts are compared too.
    - A control must be caught in each direction where it is predicted caught, and in neither where it is not. That is stricter than iteration 3, which took the union of the directions.
    - The pre-check compares the memory up to each fit year's last cut. The registered scope test runs at the first cut, with the fit reloaded from its file.
    - Every system's predicted pattern follows one rule, computed per cut through the system's own hooks: a system is predicted caught at a cut when a compared item reads a label whose `label_known_at` is after the cut hour.
    - `cheat-stamp` and `cheat-stamp-day` put a row's own label in its memory at its decision time. That is their leak, by design.
    - The purge cuts take the seeds of a 9th and a 10th cut of the same schedule (k = 8 and 9 in `perturbation_at`), and their mirror follows the parity of k.
    - `pos-window` is scored on the cut years. A row without a label of its own is dropped and counted.
    - The CPU fallback's floor is computed per system, from its own three unperturbed runs.
    - Exit codes: 2 if the pre-check fails, 1 if H1 fails, 3 if H1 holds and a skill check or `pos-window`'s check fails.
  - **Report n°3, as built.** Audit pass 4f accepted every reading.
    - H13's timing uses each arm's seed-1 fit of 2024. The 12,288 timed rows are drawn with the seed 20261030 from the scored rows that have a valid window. The Matrix timing uses the same sample, by its digest.
    - The memory readout is timed where `predict` computes it, on the CPU in fp64. It costs the same for every arm, about a fifth of the whole pass on synthetic data, so it pulls the ratio toward 1: it can only make a holding H13 fail.
    - Within each repeat the arms run in the order `enc-ols`, `enc-softmax`, `enc-linear`, so `enc-ols` and `enc-softmax` alternate.
    - H13's ratio is median(`enc-ols`) / median(`enc-softmax`), as registered.
    - Each Matrix arm is timed by its own readout at the last fit's choices, and `matrix-adaptive` at `matrix-select`'s ρ.
    - Every stage of Report n°3 runs in WSL2, on the Ryzen 5 3600 and the RTX 3060. Matrix's CPU column uses the portable build there, where the process's peak memory is `ru_maxrss`. That peak is the whole process's, panel and windows included, and the memory's own size is given beside it. Matrix's parameters are 0: its choices are selections.
    - The statistics' parameters come from Report n°2's `config.json`. The panel is checked by its fingerprint, and every scored row is paired one for one before any training.
  - **Descriptions, beside the registered verdicts.** None enters a verdict.
    - The scope test at every regular cut, for the honest arms and `cheat-stamp`, in the pre-check stage (audit pass 4e's bolder check). It loads each system's unperturbed run 1 of the cut's year; `cheat-stamp`'s is the control's own fit, the one H1 uses. At the registered setting, every system at every cut, it trains nothing that H1 does not train.
    - H13: the readout's own time, beside the attention operator's and the whole pass; and the median of the per-repeat ratios, beside the ratio of medians.
    - H4's two bounds on the three seed-matched pairs, and in the reversed direction, `enc-softmax` non-inferior to `enc-ols` (audit pass 4f's bolder check).
  - **`explain head`, its readings**, fixed before it is built.
    - It explains only scored rows of 2019–2024, which Report n°3 forecasts, so every call needs `--test-months` and is logged. A row before 2019 is refused: no fit of Report n°3 is in force there.
    - The model is the `enc-ols` checkpoint of the fit in force at the row's decision time, seed by seed, verified by its digest and against its folder.
    - Per block and head, W = Q̃ (K̃ᵀK̃ + Λ)⁻¹ K̃ᵀ over the row's 64-bar window. Its rank is at most 9.
    - The row of W is the decision bar's: 64 signed weights, oldest first.
    - The diagonal mass is trace(W) / 64. The decision token's own weight is given beside it.
    - The effective rank is the exponential of the Shannon entropy of W's normalized singular values (Roy and Vetterli, 2007).
    - Every row must sum to 1 within `row_sum_tolerance`. The row's forecast, recomputed from the checkpoint, must equal the saved one, to the bit where the fits' computation is reproduced. Either failure stops the call.
  - **Contacts with the real data in this iteration so far**, all from a clean clone of 221591c on Windows, before any `explain` call and any encoder run:
    1. Report n°2's re-run, `--compare-with runs/2026-10-29-report2`, 2026-10-30 02:42–03:05 UTC. It reads 2017-10 → 2024-12 and scores 2019-01 → 2024-12, as in iteration 3. The five allowed differences were met and every other byte matched. `series.npz`'s SHA-256 is `03916fda8ffa9f97709216b2365e7a224dd8c8d71448d75035a7824fdc9cc0b9`.
    2. Its `--check`, 03:05–03:28 UTC: the four files were byte-identical, `series.npz` included.
    3. The twins, twice, at 03:28 and 03:29 UTC, byte-identical. They read BTCUSDT's and ETHUSDT's daily bars through 2024-12, as iteration 3's H1 and adversary did.
- 2026-10-30, before the first `explain` call and before any encoder run on real data (`runs/explain/CONTACTS.log` does not exist yet). Report n°3 cannot be built as written on the Windows runs of Report n°2, so it is given a Report n°2 run made in WSL2. No rule, statistic, threshold, grid, seed or period changes.
  - **What was found.** On the same input bytes, `np.log` and `np.exp` give different last bits on Windows and in WSL2's Linux.
    - So the panel's log-based columns differ in their last bits between the two platforms: among the features the returns, volatilities, moving-average gaps and ranges, and also the threshold and the forward return. The panel's fingerprint differs with them.
    - On identical synthetic bars, exactly representable on both sides, the times, the labels, `f_qvol_z_30` and `f_taker_7` matched, and the other columns did not.
    - The `explain head` engineer found it, and the orchestrator reproduced it.
  - **Why it matters.**
    - Every stage of Report n°3 runs in WSL2, and Report n°3 checks its panel against Report n°2's fingerprint before any training. Every Report n°2 run so far was made on Windows, so Report n°3 would be refused.
    - Scoring an encoder trained on Linux features against Matrix forecasts made from Windows features would not compare two systems on one panel either.
  - **What is done.**
    - Report n°2 is re-run in WSL2, from the clean WSL2 clone, with the registered configuration.
    - Report n°3 reads that run, by its series' digest: its panel and fingerprint, `matrix-select`'s series for H5, and `climatology-panel`'s log-loss for H4's margin.
    - `explain head` runs in WSL2 too, on Report n°3's folder.
    - The Windows runs remain Report n°2's record of iteration 3. `why`, `series` and `sample` keep reading the Windows re-run.
  - **A description, beside it.** The two runs of Report n°2 are compared: the scored rows and their labels, each arm's scores, and the verdicts of H2, H3, H10, H11 and H12. A verdict that the platform's last bits can move is the first finding to report.
  - The panel H1 for the encoder arms is not affected: it builds its sub-panel and compares its own runs, all in WSL2.
- 2026-10-30, after audit pass 4g on `explain head`, before the first `explain` call (`runs/explain/CONTACTS.log` does not exist yet). No rule, statistic, threshold, grid, seed or period changes.
  - **Audit pass 4g**, on `explain head`: no Blocker, one Major point, two Minor points and two Nits. GO once the Major point is fixed, before any call on a real checkpoint. Each of the build's seven interpretations was accepted.
    - **The Major point.** `head` checked a part only against its own folder. A checkpoint changed by one bit, with its digests rewritten to match, was explained as if it were Report n°3's. `head` now refuses a part unless its checkpoint's and forecasts' digests, its count of forecasts and its epoch kept are the ones `report3.json` records for that arm, seed and year.
    - The forecast check is a fingerprint of the forecast, not a proof of the checkpoint's bytes. The digests bound to `report3.json` are the checkpoint's identity.
    - The registered row-sum check, on the fp32 W, is loose by three orders of magnitude. The same check on W formed in fp64 from the same Q, K and λ, with its own bound, is added beside it, and it stops the call too.
    - The contact line names all that a call reads: the fit year's forecasts and labels, and the memory through `as_of`.
    - The audit's bolder check, if it can be built without loosening a guard: with every seed, the recomputed forecasts, ensembled as the fits ensemble them, must equal the `enc-ols` probability of the row in Report n°3's `series.npz`, to the bit.
  - **Contacts with the real data since the last entry**, from clean clones at `5f17a76`, whose CI was green on Linux and Windows:
    4. The panel H1 for the encoder arms, on the GPU in WSL2, from 07:38 UTC. It reads the 10-member sub-panel's hourly bars through 2024-12, and forecasts and scores its rows in the cut years 2020, 2021, 2022 and 2024.
    5. The power study, on Windows, from 07:38 UTC. It reads Report n°2's series `03916fda…` by its digest.
    6. Report n°2's re-run in WSL2, from 07:39 UTC, compared with `runs/2026-10-30-report2`. It reads 2017-10 → 2024-12 and scores 2019-01 → 2024-12, as before.
