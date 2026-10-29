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

Prior expectations, written down so they can be proven wrong:
- H1 holds by construction once the tests exist.
- H2 and H5 may well fail at daily horizons on crypto, where little signal survives costs.
- H6 probably holds.
- H8 should hold, following Paster et al. (2022).
- H9 should hold: a round trip costs 1.04 % at Coinbase and 1.64 % at Kraken (taker plus 2 bps of slippage per side), against a daily standard deviation of about 3–5 % for these pairs. With six years of daily test data, an interval that excludes 0 would also need a large edge.
- H10–H12 come from an external review on 2026-10-27 (`docs/DESIGN.md` §7). Each adaptive loop is a fitted choice, so it may just as well chase noise. H11, with three weights, is the most likely to hold.

## Change log

- 2026-10-29, before any run: H2, H3 and H10–H12 are tested on the hourly bars of the universe members, over the development period. `docs/PROTOCOL.md`, "Matrix on hourly bars (iteration 3)", gives their arms and decision rules. No claim changes. One wording is made precise.
  - In H10 and H11, "chosen in validation" means chosen inside each fold, on the prequential errors of the rows whose labels are known at the fit.
  - The validation year stays closed until model selection. Choosing on it and then testing on 2019–2024 would be a look-ahead.
- 2026-10-29, after audit pass 1a and before any run: H12's "fewer refits" becomes "at most half as many refits". As first written, the CUSUM's false alarms alone gave about as many refits as the monthly schedule, so "fewer" was decided by noise. `docs/PROTOCOL.md` gives the CUSUM's new calibration, and the second test every comparison of H3, H10 and H11 now needs.
