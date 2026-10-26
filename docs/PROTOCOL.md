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
- **Cheater detection:** the bench must flag a deliberately leaking model, one that reads its own label.

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

Results are reported under three fee schedules, as of October 2026 (re-check before use):

| Venue | Maker | Taker |
|---|---|---|
| Binance spot | 0.10 % | 0.10 % |
| Coinbase Advanced, entry tier | 0.25 % | 0.50 % |
| Kraken Pro, entry tier | 0.40 % | 0.80 % |

Slippage is 2 bps per side. The exchange minimum notional and lot step are enforced.

## Baselines

Climatology, always flat, buy and hold, moving-average momentum, random positions with the same turnover, ridge on hand-made features, XGBoost.

## Change log

- 2026-10-25: initial version.
- 2026-10-25: the vault is sealed in code, after the iteration 1 audit. The splits do not change.
- 2026-10-26: git tracks the vault's key file and access log, after the re-audit.
