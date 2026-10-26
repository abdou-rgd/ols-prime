# Data

Every source OLS Prime reads, with its license and what that license lets us do. A source is listed here before its first download.

Data is never committed: `data/` is gitignored. The code is MIT. That license covers neither the data nor what is derived from it, which keep their source's terms.

## 1. Binance Vision: spot klines

**What we read**
- Monthly spot kline archives from `https://data.binance.vision`:
  - `data/spot/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.zip`;
  - each with a `.zip.CHECKSUM` file beside it, holding its SHA-256.
- The bucket listing at `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision` (S3 ListObjects v1), to enumerate every USDT pair, delisted ones included.
- Code: `python/olsprime/data/binance_vision.py` and `python/olsprime/universe.py`.

**Terms.** The *Binance Vision Dataset Terms*, version 1.0, last updated 2026-08-26. They are published as [`TERMS_AND_CONDITIONS.md`](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) in `binance/binance-public-data`; we read them on 2026-10-25, at commit `d3357a5c33ba`. Our summary, which is not legal advice:

| Clause | What it says |
|---|---|
| 1.2 | The terms supplement the Binance Terms of Use. |
| 1.4 | Amendments apply only to data made available after them. Data already downloaded stays under the terms in force at download. |
| 2.4 | Anything derived from the data falls under these terms: charts, models, indicators. |
| 3 | The datasets are licensed under CC BY-NC-SA 4.0. Where these terms and the license both bind, the stricter obligation applies. Commercial use needs a separate written license. |
| 4.1 | The allowed non-commercial uses: academic research; non-monetized open educational projects; historical backtesting for purely personal, non-production research; non-commercial open-source evaluations. |
| 4.2–4.4 | Forbidden: live proprietary trading; automated commercial order generation; selling signals; reselling or hosting the data; building it into commercial products. |
| 4.5 | A redistributed derivative work must credit Binance Vision and be licensed under CC BY-NC-SA 4.0. |
| 7 | Respect the posted rate limits, and never script around them. On 2026-10-25, neither the site nor the repository posted any limit. |
| 8 | Never imply that Binance sponsors or endorses the project. |

The repository's own `MIT` license covers its scripts, not the datasets.

**What this means for OLS Prime**
- **Allowed use.** OLS Prime is personal, non-monetized, open-source research on historical data, which clause 4.1 allows.
- **Data.** We never commit, host or redistribute it.
- **Derived outputs.** Results, figures, reports and trained weights computed from this data may be published only:
  - with the credit "Source: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0";
  - and under CC BY-NC-SA 4.0 themselves.

  The code stays MIT.
- **Live trading (roadmap phase 6).** Clause 2.4 extends the terms to models derived from the data, and clause 4.2 forbids live proprietary trading. A real-money test of a model trained on this data, even 50 €, may therefore fall outside what the terms allow. Settle one of these before phase 6:
  - train the live model on market data whose terms allow live use, such as the chosen venue's own data;
  - or get Binance's written answer.
- **Terms of Use.** Binance has been closed to French residents since 2026-07-01. Clause 2.3 of the dataset terms grants access to any individual, but the Terms of Use they supplement (clause 1.2) have not been read yet.
- **Access.** The downloader:
  - caps concurrency: 8 workers for klines, 16 for the universe;
  - retries 429, 500, 502, 503 and 504 answers with exponential backoff;
  - never fetches a verified archive twice.

**On disk (2026-10-25)**

| Pair | Interval | Archives | Bars | Open times (UTC) | Gaps |
|---|---|---|---|---|---|
| BTCUSDT | 1h | 110 | 79,837 | 2017-08-17 04:00 → 2026-09-30 23:00 | 28 gaps, 127 missing bars |
| ETHUSDT | 1h | 110 | 79,837 | same | 28 gaps, 127 missing bars: the same exchange outages, though the 2018-02-08 outage ends 11 ms later |
| BTCUSDT | 1d | 110 | 3,332 | 2017-08-17 → 2026-09-30 | none |
| ETHUSDT | 1d | 110 | 3,332 | same | none |

- All 440 archives match their published SHA-256.
- 36 of them are vault months (2026-01 → 2026-09, see [PROTOCOL.md](PROTOCOL.md)). They stay in the raw cache and in the Parquet files, and the integrity checks cover them. Every read for modelling goes through `olsprime.protocol.load_klines`, which seals them: BTCUSDT 1 h then returns 73,285 bars, the last opening on 2025-12-31 23:00.
- Binance may replace an archive later. Its repository lists the replacements in `updates/`. The cache keeps the first verified copy, so a rebuild is reproducible, but it misses later corrections until the archive is downloaded again with `--force`.

**Quirks to handle downstream.** The quality report (`{SYMBOL}.quality.json` beside each Parquet file) counts them.
- **Timestamps.** From 2025-01-01, spot files use microseconds. The parser reads the unit from each value's magnitude and stores milliseconds UTC.
- **Off-grid bars.** The 1 h series have 43 of them, from 2018-02-09 09:28 to 2018-02-11 03:28. They are shifted by about 28 minutes after the largest outage: 32 bars missing from 2018-02-08 01:00.
- **Bad close times.** The 1 h series have 15 bars whose close time is not open + 1 h − 1 ms:
  - 12 end early, from 1 second to 60 minutes before the hour, around outages and maintenance windows;
  - the bar of 2020-12-21 14:00 closes 12.7 minutes before it opens;
  - the last off-grid bar (2018-02-11 03:28) closes at 04:00:00.999, and the bar of 2017-09-06 15:00 closes 1 ms late;
  - four of the 15 have volume 0.

  So a bar's values count as final at `known_at = max(close_time, open_time + interval − 1 ms)`, which `load_klines` adds, and never at the stored `close_time` alone. The vault seal cuts on `known_at`, and any Matrix entry built from a bar must use it as its time.
- **The last gap** in the 1 h series starts on 2023-03-24 at 13:00.
- **Ticker reuse.** `LUNAUSDT` is the old LUNA until 2022-05-13 and the new LUNA from 2022-05-31. Cut every series at its gaps before computing returns.

## 2. Reference parameters

These are not datasets. They are the venue figures that `crates/backtest` hard-codes. A `scout` agent checked them on 2026-10-25 through a summarizing fetch tool, so re-read the official page before relying on any one of them.

| Parameter | Value in the code | Source | Check on 2026-10-25 |
|---|---|---|---|
| Binance spot, VIP 0, maker and taker | 0.10 % / 0.10 %; 0.075 % with BNB | [binance.com fee schedule](https://www.binance.com/en/fee/schedule) | matches |
| Coinbase Advanced, entry tier | 0.25 % / 0.50 % | coinbase.com (every fee page answered HTTP 403, on two passes) | **unverified at the source**. Two news reports give 0.25 % / 0.50 % for the EU from 2026-09-16, with the first tier cut from 25,000 to 10,000 USD of 30-day volume. The 0.40 % / 0.60 % of an older help page predates that change. |
| Kraken Pro, tier 1 | 0.40 % / 0.80 % | [kraken.com fee schedule](https://www.kraken.com/features/fee-schedule) | matches; the page shows no separate EU schedule |
| BTCUSDT filters | `NOTIONAL` minNotional 5 USDT; `LOT_SIZE` stepSize 0.00001 | [`exchangeInfo`](https://data-api.binance.vision/api/v3/exchangeInfo?symbols=%5B%22BTCUSDT%22,%22ETHUSDT%22%5D) | matches |
| ETHUSDT filters | `NOTIONAL` minNotional 5 USDT; `LOT_SIZE` stepSize 0.0001 | same | matches |

**MiCA.** ESMA's interim register of crypto-asset service providers ([CSV](https://www.esma.europa.eu/sites/default/files/2024-12/CASPS.csv)) lists these venues as authorized and passported to France, as read on 2026-10-25:

| Venue | Entity | Authority | Date |
|---|---|---|---|
| Coinbase | Coinbase Luxembourg S.A. | CSSF | 2025-06-20 |
| Kraken | Payward Europe Solutions Limited | Central Bank of Ireland | 2025-06-25 |
| Bitstamp | Bitstamp Europe S.A. | CSSF | 2025-05-15 |
| Bitvavo | Bitvavo B.V. | AFM | 2025-06-26 |

Caveats:
- The authorization dates are our reading of the file's first date column. The Coinbase, Bitstamp and Bitvavo rows also carry a later date in the `ac_lastupdate` column (2026-01-22, 2025-05-16 and 2025-06-30), which is the row's last update, not its authorization.
- For Kraken the entity matters: Payward Global Solutions Limited is not passported to France.
- No Binance entity is in the register. That Binance stopped serving French residents on 2026-07-01 rests on press reports of its e-mail to users; we found no primary page.

## 3. Not used yet

Each of these gets an entry above before its first download:
- the news source for H7 (phase 5): its license must allow point-in-time research use;
- the market data of the venue chosen for the live loop (phase 6).
