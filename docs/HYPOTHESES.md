# Pre-registered hypotheses

Each hypothesis is written before its experiment runs. An entry records:
- the claim and what would refute it;
- the metric and the decision rule.

Results go to [RESULTS.md](RESULTS.md), appended and never edited. Unless stated otherwise, comparisons follow [PROTOCOL.md](PROTOCOL.md): walk-forward on the development period, final check on the vault.

| ID | Claim | Metric and decision rule | Registered |
|---|---|---|---|
| H1 | The bench catches leakage: a model that reads its label is flagged, and every honest model passes the future-invariance test. | Cheater flagged in every run; zero invariance violations. | 2026-10-25 |
| H2 | Matrix on hand-made features forecasts better than climatology. | Brier skill score > 0, with a block-bootstrap 95 % interval excluding 0. | 2026-10-25 |
| H3 | Signed weights help: Matrix (OLS weights) beats the same memory with positive weights (kNN, softmax kernel). | Log-loss difference, Diebold-Mariano p < 0.05. | 2026-10-25 |
| H4 | OLS attention is at least as good as softmax attention in the encoder, at equal parameter count. | Log-loss non-inferiority, margin 0.5 % of climatology's log-loss. | 2026-10-25 |
| H5 | The encoder adds value over Matrix alone. | Log-loss improvement, Diebold-Mariano p < 0.05. | 2026-10-25 |
| H6 | The Brier term improves calibration without hurting log-loss. | Expected calibration error down by at least 20 % relative; log-loss not worse beyond seed noise. | 2026-10-25 |
| H7 | News tokens improve forecasts. | Brier skill score up, Diebold-Mariano p < 0.05. | 2026-10-25 |
| H8 | Return-to-go conditioning (Decision Transformer) does not help when our orders do not move prices. | No improvement over the same model without return-to-go. | 2026-10-25 |
| H9 | At the entry-tier taker fees of the venues open to French residents (Coinbase Advanced, Kraken Pro), no active baseline beats buy and hold on BTCUSDT or ETHUSDT on daily bars with h = 1 day, over the walk-forward test months 2019-01 → 2024-12 of the development period. Active baselines: moving-average momentum (10, 30, 90 days), random with matched turnover, climatology, ridge, XGBoost. | For each venue, pair and active baseline: its net annualized Sharpe minus buy and hold's, with a 95 % paired stationary-bootstrap interval. H9 holds at a venue if no interval lies entirely above 0. Refuted if one does. | 2026-10-27 |
| H10 | For Matrix on hand-made features, a λ chosen at each step by its prequential error, computed from Matrix's own statistics, beats a fixed λ chosen in validation (`docs/DESIGN.md` §7). | Log-loss difference, Diebold-Mariano p < 0.05. | 2026-10-27 |
| H11 | Stacking Matrix's 30-day, one-year and all-history readouts, with three weights fitted on prequential errors, beats the best single window chosen in validation. | Log-loss difference, Diebold-Mariano p < 0.05. | 2026-10-27 |
| H12 | Refitting when a CUSUM on Matrix's standardized recursive residuals alarms, at a false-alarm rate fixed in advance, is no worse than monthly refits, with at most half as many refits. | Log-loss non-inferiority, margin 0.5 % of climatology's log-loss, and at most 36 fits where the monthly schedule makes 72. | 2026-10-27 |
| H13 | OLS attention is as good as softmax attention in the encoder and cheaper to run. | H4 holds, and the median wall time per 10⁴ decisions of `enc-ols`, over 10 repeats, is at most 0.9 × `enc-softmax`'s, under the conditions of `docs/PROTOCOL.md`, iteration 4. | 2026-10-29 |
| H14 | At Coinbase's and Kraken's entry-tier taker fees plus 2 bps of slippage, the long/flat portfolio of `matrix-select` beats always-flat. | Net annualized Sharpe ratio, with a 95 % stationary-bootstrap interval above 0 (always-flat earns 0), **and** a paired interval of its Sharpe ratio minus that of random positions with the same exposure and turnover above 0, at each venue, under the rule of `docs/PROTOCOL.md`, iteration 4. Runs in iteration 5. | 2026-10-29 |

Prior expectations, written down so they can be proven wrong:
- H1 holds by construction once the tests exist.
- H2 and H5 may well fail at daily horizons on crypto, where little signal survives costs.
- H6 probably holds.
- H8 should hold, following Paster et al. (2022).
- H9 should hold: a round trip costs 1.04 % at Coinbase and 1.64 % at Kraken (taker plus 2 bps of slippage per side), against a daily standard deviation of about 3–5 % for these pairs. With six years of daily test data, an interval that excludes 0 would also need a large edge.
- H10–H12 come from an external review on 2026-10-27 (`docs/DESIGN.md` §7). Each adaptive loop is a fitted choice, so it may just as well chase noise. H11, with three weights, is the most likely to hold.
- Written on 2026-10-29, before any encoder run:
  - H4 should hold, narrowly. Both heads sit in a small encoder that learns its own queries and keys, so the arms' differences may be no larger than the spread between seeds. In Report n°2 a fixed softmax kernel lost 0.83 % of climatology's log-loss to the OLS readout on the same memory, but its keys were not learned.
  - H5 will probably fail. It needs a passing gain over `matrix-select`, whose Brier skill is 1.3 %, from 64 hours of bars and yearly refits. The power study says how large a gain the rule can see.
  - H13 will probably fail. At T = 64 the softmax is two small matrix products, while the OLS head adds a batched Cholesky factorization and two triangular solves, each its own kernel. Analytic FLOPs favour OLS by about four, but wall time at this size rarely follows FLOPs.
  - H14 will fail. A Brier skill of 0.013 is a small edge against a round trip of 1.04 % at Coinbase and 1.64 % at Kraken.

## Change log

- 2026-10-29, before any run: H2, H3 and H10–H12 are tested on the hourly bars of the universe members, over the development period. `docs/PROTOCOL.md`, "Matrix on hourly bars (iteration 3)", gives their arms and decision rules. No claim changes. One wording is made precise.
  - In H10 and H11, "chosen in validation" means chosen inside each fold, on the prequential errors of the rows whose labels are known at the fit.
  - The validation year stays closed until model selection. Choosing on it and then testing on 2019–2024 would be a look-ahead.
- 2026-10-29, after audit pass 1a and before any run: H12's "fewer refits" becomes "at most half as many refits". As first written, the CUSUM's false alarms alone gave about as many refits as the monthly schedule, so "fewer" was decided by noise. `docs/PROTOCOL.md` gives the CUSUM's new calibration, and the second test every comparison of H3, H10 and H11 now needs.
- 2026-10-29, before any run of iteration 4. `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)", gives the arms, fits, seeds and rules. No claim of H4–H6 changes; their rules are made precise.
  - **H4.** Non-inferiority now needs two bounds below the margin: the one-sided 95 % upper bound from the Diebold-Mariano standard error, and the stationary bootstrap's 95th percentile, as for H12. The arms are `enc-ols` and `enc-softmax`, at equal parameter count within 1 %.
  - **H5.** "Diebold-Mariano p < 0.05" becomes the pass rule of every comparison since iteration 3: p < 0.05, and the bootstrap interval on the same side. The arms are `enc-ols` and `matrix-select`.
  - **H6.** "Not worse beyond seed noise" means a log-loss above `enc-ols`'s by no more than the range of `enc-ols`'s three per-seed log-losses.
  - **H13 and H14 are new.** H13 comes from Abdou: quality and cost together. H14 is registered now and runs in iteration 5, with the portfolio simulator.
- 2026-10-30, after audit pass 4a and before any run of iteration 4. No claim is loosened. Two rules are made precise, and H14's is narrowed.
  - **H4 can no longer hold vacuously.** Two arms that both sit at climatology would meet its bounds. H4, and with it H13, is now testable only if `enc-ols` or `enc-softmax` has a Brier skill score whose 95 % interval lies above 0. Otherwise it is reported as not testable. The margin is Report n°2's `climatology-panel` mean hourly log-loss.
  - **H6** needs, beside the point ratio of 0.8, the bootstrap's 95 % upper bound of the ECE ratio below 1. Its log-loss tolerance is the larger of the per-seed range and the standard error of the difference: three seeds can give a range near 0 by chance.
  - **H14 is no longer confounded with the market's drift.** A long/flat portfolio that is long most of the time inherits it. H14 now also needs its Sharpe ratio to beat that of random positions with the same exposure and turnover. Its sizing, exits, initial equity and annualization are fixed in `docs/PROTOCOL.md`. After audit pass 4b, the same day, the random positions are defined as two-state chains matched per member, 100 of them.
