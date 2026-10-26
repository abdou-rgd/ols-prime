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

Prior expectations, written down so they can be proven wrong:
- H1 holds by construction once the tests exist.
- H2 and H5 may well fail at daily horizons on crypto, where little signal survives costs.
- H6 probably holds.
- H8 should hold, following Paster et al. (2022).
