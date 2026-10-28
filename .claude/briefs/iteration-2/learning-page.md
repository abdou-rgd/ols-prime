# Brief — The iteration 2 learning page (teacher)

- **Agent:** `teacher` (opus, high).
- **When:** after the outcome of iteration 2 is recorded in `docs/ITERATIONS.md` and the rows in `docs/RESULTS.md`.
- **Motto:** we should be more ambitious: teach deeper than a summary, and let the reader break the idea.

## The page

- **File:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\explainer\banc-honnete.html`. It is a new page, outside the repository.
- **Title:** two to four words, a name, for example `Le banc honnête`.
- **Previous page:** iteration 1's page, which Abdou loved, is at `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\explainer\comprendre-ols-prime.html`. Its published link is https://claude.ai/artifact/386T8BiAjdUfbmjMZ1niGw.
  - Read it first. Keep its quality bar, its voice and the kind of demos it has.
  - Give this page its own visual identity, drawn from this iteration's world: walk-forward timelines, fee schedules, bootstrap resamples, proof logs.
  - Open with what changed since that page, and link back to it rather than repeating it.

## What to teach

Ground every claim in the repository, and every number in the verified figures below.
1. **Why a bench before a model.** What "honest" means: look-ahead, leakage, survivorship, the vault, multiple testing.
2. **Point-in-time features and labels.** `known_at`, the decision at a close executing at the next open, `fwd_return = ln(open[t+1+h] / open[t+1])`, and the label threshold.
3. **Walk-forward with purge and embargo.**
   - Draw it, and derive the purge from `label_known_at ≤ fit_time − embargo`.
   - Demo: a timeline where the reader moves h and the embargo and sees which training rows drop.
4. **The baselines and the decision rule.** Each baseline, the hysteresis band between the round-trip cost and 0, and why the cost depends on the venue.
5. **What fees do.** The headline of Report n°1, venue by venue.
   - Demo: a fee slider applied to a strategy's turnover, showing the drag `turnover × cost` eating the gross return. Use the report's real turnover figures.
6. **The statistics.**
   - The Sharpe standard error for non-normal returns, derived from the delta method, with a worked example on the report's numbers.
   - PSR and DSR, with the expected maximum of N Sharpes. Demo: N trials of pure noise, and the best one's Sharpe.
   - The stationary bootstrap: why blocks, and what the block length does.
   - Diebold-Mariano with HAC.
7. **H1, the cheater and the future-invariance test.** How a model that reads the next bar is caught bit for bit, and why the vault seal refuses it at the boundary.
8. **PyO3.** How Python calls the Rust simulator, and why bit-for-bit equality matters.
9. **Kani function contracts.** `requires`, `ensures`, `proof_for_contract`, `stub_verified`; why a stub is only as sound as its contract's proof; what finished.
10. **The external review.** What Gemini got right, what the code already handled, and DESIGN §7's adaptation loops (GCV and its time-series trap, the exact gradient with respect to a block's weight, CUSUM). These are hypotheses H10–H12, not results.
11. **La suite:** the bolder next step, and what Abdou should master before iteration 3.

Close with "Teste-toi" (six to eight questions, answers in `<details>`), a glossary, and the references cited in the repository.

## Page contract (current platform rules, which win over your agent file)

- **Structure.** No `<!doctype>`, `<html>`, `<head>` or `<body>` tags: the platform wraps the page. Start with `<title>`, then `<meta name="claude:viewport-fit-sides" content="cover">`, then `<style>`. The wrapper pads `:root` top and bottom by the safe-area insets; keep that padding.
- **Gutters.** `padding-left: max(16px, env(safe-area-inset-left, 0px))` and the right-hand twin, once on `body` or one outer wrapper. Vertical padding uses `padding-block`, never a `padding` shorthand that zeroes the sides.
  - A sticky header uses `top: env(safe-area-inset-top, 0px)`, with `body::before` painting the band above it.
  - The body never scrolls sideways. Tables, formulas, diagrams and code scroll inside their own `overflow-x: auto` boxes.
  - Flex and grid children that hold text get `min-width: 0`.
- **Themes.** Every color is a token on bare `:root`, with the light values.
  - Redefine the tokens under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ...; color-scheme: dark } }`.
  - Redefine them again under `:root[data-theme="dark"] { ...; color-scheme: dark }`.
  - `body` gets an explicit token background, and no component uses a literal color.
- **External files.**
  - Scripts only from cdnjs.cloudflare.com (preferred), cdn.jsdelivr.net/npm or unpkg.com, each pinned to an exact version at least two weeks old (MathJax 3 SVG from cdnjs, as last time).
  - Stylesheets only from Google Fonts, with real fallback stacks.
  - Everything else inline. No fetch to other hosts, no iframes, no `alert`, `confirm` or `prompt`, no download links, no print button.
- **Mermaid.** The viewer renders `<pre class="mermaid">` natively, so load no Mermaid library. Draw what Mermaid cannot as inline SVG, with explicit fills and theme-token colors.
- **Readability.** Body text at least 17 px on phones, about 65 characters per line, `text-wrap: balance` on headings, `tabular-nums` in tables, a visible focus state, and `prefers-reduced-motion` respected.
- **Rest state.** Everything is visible at rest, with no `opacity: 0` waiting for an observer. Every demo opens filled and working.
- **Storage.** `localStorage` only for conveniences, inside try/catch.
- **Look.** No templated look. Avoid cream with a terracotta serif, near-black with an acid accent, purple gradients, Inter or Space Grotesk, emoji as section markers, and `rounded-lg` on everything.
- **Size.** Keep it well under 16 MB.

## Rules

- French throughout, with the narrow no-break space (U+202F) before `:` `;` `?` `!` and inside « », as the previous page does.
- MathJax `\(...\)` with every backslash intact. Write with your editor tools, never through a shell heredoc: one ate the backslashes once.
- Mark planned work as planned, and hypotheses as hypotheses. Results come only from `docs/RESULTS.md` and the figures below.
- **Checks:**
  - `node --check` on each inline script, extracted to a temp file;
  - every tag closed;
  - every external URL answers 200 (`curl -sI`);
  - a read at 400 px wide.
- Do not publish, and do not commit.

## Verified figures

The orchestrator appends them below once they are verified.

**Status, 2026-10-27 23:50 UTC.**
- The orchestrator re-ran every figure below:
  - the full test suite: 628 passed, 2 skipped;
  - `report1 --check runs/2026-10-27-report1-2`: 14 files identical, byte for byte.
- The audit is still running, so these figures are provisional.
- The Kani results and the universe numbers come later, by message.
- Write every figure into the prose directly. In your report, list the sections that carry figures, so they can be corrected quickly if the audit changes one.

**Setting of Report n°1.**
- BTCUSDT and ETHUSDT, daily bars, through `load_klines`, cut at 2025-01-01.
- 72 monthly folds, test months 2019-01 → 2024-12, expanding window of 470 to 2,631 training rows.
- 2,192 scored bars, which give 2,191 daily returns, the same for every strategy and venue.
- Labels have three classes with h = 1. Class frequencies (down, neutral, up) on the scored rows: BTCUSDT 0.229, 0.506, 0.265; ETHUSDT 0.240, 0.479, 0.281.
- 10,000 USDT of initial cash. Taker rate plus 2 bps of slippage per side; Binance's filters (min notional 5).
- The first scored bar never trades. An open position is marked at the last close, without an exit fee.

**Fees.** Round trip = 2 × (taker + 0.02 %).

| Venue | Taker | Round trip |
|---|---|---|
| zero (slippage only) | 0 | 0.04 % |
| binance-spot | 0.10 % | 0.24 % |
| bitvavo-entry | 0.25 % | 0.54 % |
| coinbase-advanced-entry | 0.50 % | 1.04 % |
| kraken-pro-entry | 0.80 % | 1.64 % |

**Headline.** Net annualized Sharpe, with its stationary-bootstrap 95 % interval. The difference to B&H carries a paired bootstrap 95 % interval.

| pair | venue | B&H | best active | its Sharpe | difference to B&H |
|---|---|---|---|---|---|
| BTCUSDT | zero | 1.15 [0.35, 1.96] | ma-momentum-30 | 1.35 | +0.20 [−0.37, 0.81] |
| BTCUSDT | binance-spot | 1.15 [0.37, 1.95] | ma-momentum-30 | 1.28 | +0.13 [−0.42, 0.73] |
| BTCUSDT | bitvavo-entry | 1.15 [0.34, 1.93] | ma-momentum-90 | 1.19 | +0.04 [−0.50, 0.58] |
| BTCUSDT | coinbase-advanced-entry | 1.15 [0.29, 1.96] | ma-momentum-90 | 1.11 | −0.04 [−0.58, 0.52] |
| BTCUSDT | kraken-pro-entry | 1.15 [0.36, 1.94] | ma-momentum-90 | 1.00 | −0.14 [−0.70, 0.40] |
| ETHUSDT | zero | 1.06 [0.23, 1.85] | ma-momentum-30 | 1.15 | +0.09 [−0.49, 0.67] |
| ETHUSDT | binance-spot | 1.06 [0.25, 1.87] | ma-momentum-30 | 1.10 | +0.05 [−0.55, 0.63] |
| ETHUSDT | bitvavo-entry | 1.06 [0.27, 1.86] | ma-momentum-30 | 1.03 | −0.02 [−0.61, 0.55] |
| ETHUSDT | coinbase-advanced-entry | 1.06 [0.28, 1.85] | ma-momentum-30 | 0.92 | −0.14 [−0.77, 0.46] |
| ETHUSDT | kraken-pro-entry | 1.06 [0.24, 1.87] | ma-momentum-30 | 0.78 | −0.28 [−0.89, 0.38] |

**H9.**
- **Holds at both registered venues, Coinbase and Kraken.** At each venue, none of the 14 comparisons (7 baselines × 2 pairs) has a paired interval entirely above 0.
- Some of those baselines are never exposed (climatology, and ridge on BTCUSDT): their Sharpe is undefined, and they count as "not above 0".
- The other venues are descriptive. No interval lies above 0 there either.

**The fee demo: real turnover and annualized net return.** Turnover is traded notional over mean equity, per year; a full round trip adds 2. Moving averages give the same targets at every venue. The forecasters' targets depend on the venue, through the hysteresis band.

| BTCUSDT | turnover/yr | zero | binance | bitvavo | coinbase | kraken |
|---|---|---|---|---|---|---|
| buy-and-hold | 0.0 | 70.5 % | 70.5 % | 70.5 % | 70.4 % | 70.3 % |
| ma-momentum-10 | ≈ 62 | 43.3 % | 34.6 % | 22.5 % | 4.6 % | −13.4 % |
| ma-momentum-30 | ≈ 30.4 | 68.0 % | 62.8 % | 55.3 % | 43.6 % | 30.7 % |
| ma-momentum-90 | ≈ 16 | 64.5 % | 61.9 % | 58.0 % | 51.7 % | 44.6 % |
| random-turnover (median of 100 seeds) | ≈ 29 | 27.9 % | 24.1 % | 18.6 % | 10.2 % | 0.6 % |

| ETHUSDT | turnover/yr | zero | binance | bitvavo | coinbase | kraken |
|---|---|---|---|---|---|---|
| buy-and-hold | 0.0 | 69.8 % | 69.8 % | 69.7 % | 69.6 % | 69.6 % |
| ma-momentum-10 | ≈ 63.8 | 49.9 % | 40.5 % | 27.5 % | 8.5 % | −10.6 % |
| ma-momentum-30 | ≈ 27 | 66.5 % | 61.9 % | 55.2 % | 44.7 % | 33.1 % |
| ma-momentum-90 | ≈ 16.6 | 42.6 % | 40.3 % | 36.9 % | 31.4 % | 25.0 % |

- **A rule of thumb that fits these numbers.** The orchestrator derived it, and it is not in the report: annual log-growth at a venue ≈ log-growth at zero − turnover × taker.
  - BTCUSDT ma-momentum-10: ln 1.433 − ln 0.866 = 0.50, against 61.5 × 0.008 = 0.49.
  - ma-momentum-30: 0.251 against 0.244.
  - ma-momentum-90: 0.129 against 0.127.
  - Use it in the slider demo, and say it is an approximation.
- **Net Sharpe of BTCUSDT ma-momentum-10:** 1.02 at zero, 0.88 at binance, 0.67 at bitvavo, 0.32 at coinbase and −0.09 at kraken.

**Worked example, the Sharpe standard error.** BTCUSDT buy and hold, zero fees, T = 2,191 daily simple returns.
- Daily mean 0.2064 %, daily standard deviation 3.436 %. Per-period Sharpe 0.0601, annualized 1.148 (× √365).
- Skewness γ₃ = −0.411. Kurtosis γ₄ = 14.03, not excess.
- Mertens: SE = √((1 − γ₃·SR + (γ₄ − 1)/4·SR²)/(T − 1)) = 0.02175 per period, 0.416 annualized. The iid-normal formula gives 0.409.
- Analytic 95 % interval [0.33, 1.96]. Bootstrap [0.35, 1.96], with a Politis-White mean block of 4 days.
- PSR(0) = 0.997. DSR = 0.966, with N = 8 candidates and V = 1.97e-4, the per-period variance of their Sharpes.
- ETHUSDT buy and hold: annualized Sharpe 1.06, γ₃ = −0.29, γ₄ = 11.86, SE 0.413.

**DSR, the variants counted.**
- 10 baselines are in the fixed list. 8 of them are DSR candidates: flat and the cheater are excluded.
- N per pair and venue counts the candidates with a defined Sharpe, from 6 to 8. V runs from 6.6e-5 to 7.6e-4.
- random-turnover counts as one variant of 100 seeds.

**Forecast quality.** Venue-independent; 2,190 rows per pair. Brier skill score against climatology, with a paired bootstrap 95 % interval. Diebold-Mariano on the per-row log-loss.

| pair | model | log-loss | BSS [95 %] | ECE top / class-wise | DM stat, p |
|---|---|---|---|---|---|
| BTCUSDT | climatology | 1.0359 | 0 (reference) | 0.018 / 0.012 | — |
| BTCUSDT | ridge | 1.1603 | −0.125 [−0.159, −0.093] | 0.078 / 0.144 | 10.8, < 0.001 |
| BTCUSDT | xgboost | 1.0886 | −0.051 [−0.068, −0.034] | 0.083 / 0.077 | 6.2, < 0.001 |
| ETHUSDT | climatology | 1.0533 | 0 (reference) | 0.008 / 0.008 | — |
| ETHUSDT | ridge | 1.1447 | −0.087 [−0.111, −0.065] | 0.063 / 0.127 | 8.5, < 0.001 |
| ETHUSDT | xgboost | 1.1039 | −0.046 [−0.063, −0.029] | 0.070 / 0.073 | 5.9, < 0.001 |

- Ridge's λ is chosen inside each fold from the grid (1, 10, 10², 10³, 10⁴). On BTCUSDT:
  - 10⁴ in 38 of the 72 folds, the edge of the grid;
  - 10³ in 27, 10² in 6 and 10 in 1.
  - The heaviest shrinkage on offer wins most often. The orchestrator reads this as a sign of little linear signal; it is an interpretation, not a test.
- Neither learned model beats climatology. This is the bar Matrix must clear in iteration 3 (H2).

**H1, the future-invariance test.**
- 2 pairs × 8 cuts per pair:
  - 3 cuts at the last bar before a fold starts;
  - 3 at the first bar of a test month;
  - 2 at random bars.
- At each cut, every bar after it is perturbed. A violation is any decision at or before the cut whose probabilities, expected return or target, at any venue, changes by a single bit.
- **The 11 honest models** have 0 violations at all 16 cuts: flat, B&H, ma-10/30/90, three random seeds, climatology, ridge and xgboost.
- **The cheater** reads the next bar. It is flagged at 16 of 16 cuts, with one violating row per cut: the decision at the cut itself.
- **The planted purge leaks** are the unpurged climatology, ridge and xgboost. They are flagged at exactly the 3 first-bar-of-test-month cuts per pair, and nowhere else. Without the purge, the fold's last training label needs the bar after the fit.
- **The seal leaves the cheater blind.** The last sealed bar opens on 2025-12-31. The next bar opens at the vault start, and `load_klines` does not return it. So that bar's label is null, and the cheater's target there is 0.
- One H1 run takes 454 s with 6 workers.

**Reproducibility and bindings.**
- Report n°1 runs in about 90 s, and `--check` reproduces it byte for byte: 14 files.
- Ridge cross-check, numpy against Rust `olsprime_rs.ridge_fit` on all 72 folds: worst gap 4.2e-16 on BTCUSDT and 1.7e-15 on ETHUSDT, against a tolerance of 1e-9.
- The provenance block prints the git commit, a code digest over 55 files, the data manifest digest (571 files at the time), and each file's SHA-256 with its stored and sealed row counts. BTCUSDT 1d stores 3,332 rows, of which 3,059 are sealed: 2017-08-17 → 2025-12-31.

**Diebold-Mariano kernel.** Measured size at a nominal 5 % with h = 7: 5.25 % for the rectangular kernel, 8.75 % for Bartlett. With h = 1 the two are identical.

**The universe panel (descriptive, no interval).**
- 224 membership spans of 131 pairs were scored, out of 479 spans; 254 were skipped as too short.
- Median net Sharpe of buy and hold: 0.26 at zero, 0.22 at kraken.
- ma-momentum-10's median falls from 0.34 at zero to −0.26 at kraken. It beats B&H on the same span in 47 % of spans at zero, and in 26 % at kraken.

**The external review (Gemini, 2026-10-27).** `docs/DESIGN.md` §2–§5, §7 and its change log record what was taken.
- **A real error was fixed.** The row-sum condition of §4: Λ = diag(0, λ, …), with the query's coordinate 0 equal to 1.
- **Two points were already handled in code.** The RLS update is symmetrized, and the wind-up is documented.
- **The rest became planned work.**
  - Matrix with centred nodes, merged with `M(A∪B) = M(A) + M(B) + (n_A n_B / n) δδᵀ`;
  - fp64 prefix sums;
  - label smoothing as an ablation;
  - about N/h effective observations when outcomes overlap.
- **The self-improvement loops became hypotheses H10–H12,** not results: a prequential λ, a stack of three windows, and CUSUM refits.

**Bolder next step, provisional.** Iteration 3 puts Matrix on this bench:
- on the 15 hand-made features of `FEATURE_COLUMNS`, across the whole monthly top-30 universe, at 1h and 1d;
- H2, H3 and H10–H12 are tested in one pre-registered run;
- the bar to clear is climatology's log-loss, which ridge and XGBoost failed to beat here.
The orchestrator will confirm it after the audit.

## Updated 2026-10-28 — H1 strengthened, and the new sensitivities

Source: `runs/2026-10-28-h1-fix/h1.md`, and a Report n°1 run in the orchestrator's scratch, both on the fixed bench. The official runs on the committed tree use the same code and seeds, so the figures should not move. The orchestrator confirms them before you write the final version.

**The perturbation now moves every column.** This is the M1 correction; the bullets of §7 that describe the perturbation must describe this version.
- **Prices.** A random walk is sampled at every open and close, times a common level factor.
- **Volumes and trades.** `volume`, `quote_volume` and `trades` get the same level factor, times one log-normal draw per bar. `trades` is rounded to an integer, and moved by one more trade if rounding gave it back unchanged.
- **Taker share.** The taker share of every later bar except the first is redrawn uniformly in (0, 1), so it moves both ways. The taker volumes are then share × total.
- **The first bar after the cut**, the one a one-bar leak reads:
  - its body is swapped;
  - it is then scaled by a single factor large enough that every price ends above every price of bar t with ×3, or below with ÷3;
  - the same holds for its volumes and trades, with their own factor, and its taker share is pushed past bar t's.
  - Why: ×3 alone was not enough. On 2023-07-01, BTCUSDT's quote volume fell to 0.196 of the previous day's, and tripling did not flip the comparison.
- **Both directions run at every cut.** A one-bar comparison flips in exactly one of the two directions, which is why each peeker is caught by ×3 at some cuts and by ÷3 at the others.
- **Coverage is asserted on every run.** Every value of every later bar must differ. Only `symbol`, `open_time`, `close_time` and `known_at` stay, plus the exact zeros of a bar with no volume, which keep the null pattern.

**Results (2 pairs × 8 cuts × 2 directions). Verdict: H1 holds.**

| model | BTC ×3 | BTC ÷3 | BTC either | ETH ×3 | ETH ÷3 | ETH either | predicted |
|---|---|---|---|---|---|---|---|
| 11 honest models | 0 | 0 | 0 | 0 | 0 | 0 | never |
| cheater | 8 | 8 | 8 | 8 | 8 | 8 | every cut |
| trades-peeker | 4 | 4 | 8 | 4 | 4 | 8 | every cut |
| quote-volume-peeker | 4 | 4 | 8 | 4 | 4 | 8 | every cut |
| taker-share-peeker | 5 | 3 | 8 | 2 | 6 | 8 | every cut |
| off-by-one-return | 8 | 8 | 8 | 8 | 8 | 8 | every cut |
| transductive-scaler | 5 | 5 | 5 | 5 | 5 | 5 | every cut except the 3 last bars of a month |
| climatology, ridge and xgboost, unpurged | 3 | 3 | 3 | 3 | 3 | 3 | the 3 first bars of a test month |

- The 11 honest models are flat, B&H, ma-momentum-10/30/90, three random-turnover seeds, climatology, ridge and xgboost. The 9 controls are the cheater, the 5 planted leaks and the 3 unpurged models. Every control matches its predicted pattern exactly. H1 holds only if every honest model has 0 and every control matches.
- Before the fix, the auditor's `trades` peeker was caught at 0 cuts out of 8, the quote-volume peeker at 1 and the taker-share peeker at 3.
- **Positive control.** Ridge is fitted on labels shifted one bar early. Its BSS against climatology is +0.9749 on BTC and +0.9798 on ETH, against −0.1246 and −0.0874 on the real labels. The threshold is 0.5.
- **Duration.** 498 s with 6 processes. The first run took 454 s.
- **The limit, now written in h1.md.**
  - The time stamps cannot move. A model that memorizes the future by date, such as a calendar learned from the future, passes the test. Code review and the seal stop it, not this test.
  - A leak that shows only through a 0/1 target is caught only where the perturbation flips that target.

**A factual error to correct in §7, "Trois sortes de coupes".**
- The page says that the last-bar-before-fold cuts catch "un coup d'œil au mois de test". This is wrong.
  - At the last bar of a month, that month is entirely in the past, so a scaler fitted on the whole test month is invisible there.
  - The transductive scaler is caught at the other 5 cuts: the first bars of a month and the random bars.
- What the last-bar-before-fold cuts do catch:
  - statistics of the whole sample, which reach later months;
  - and the cheater.
- Also remove "Dans la version qui a tourné, la colonne `trades` ne changeait pas non plus", or rewrite it as the history of M1. The bullet "La correction, en cours" is no longer in progress.

**The cheater's lot cap (m2).**
- The simulator caps a position at 2^50 lots.
- The report now dates the first scored bar where the cheater's held position sits at that cap:

  | pair | zero | Binance | Bitvavo | Coinbase | Kraken |
  |---|---|---|---|---|---|
  | BTC | 2024-08-12 | never | never | never | never |
  | ETH | 2022-07-02 | 2022-08-21 | 2022-11-10 | 2023-08-19 | 2024-12-11 |

- From those dates on, the cheater cannot grow its position any more. Its Sharpe of 10.60 and its 7 513 % a year (BTC, zero fees) are therefore minima.
- This changes nothing for the honest models, which never come near the cap.

**The two new sensitivities in Report n°1 (m3, m4). No verdict changes.**
- **H9 with longer blocks.** The block is max(PW(d), PW((d − d̄)²)), where PW is the Politis-White length. Volatility clustering lives in the squares, not in the returns.
  - BTC: block 4 → 25 at zero and Binance, 7 at the other three venues.
  - ETH: block 1–2 → 25.
  - Headline intervals:
    - BTC Coinbase −0.04 [−0.58, 0.52] → [−0.61, 0.54];
    - ETH Kraken −0.28 [−0.89, 0.38] → [−0.86, 0.26].
  - At the two registered venues, 11 of the 14 comparisons have an interval. The other 3 are undefined: the random-turnover variants are never exposed there. Under either block, no interval lies entirely above 0. H9 holds at Coinbase and at Kraken.
- **Diebold-Mariano with Newey-West.** This is the Bartlett kernel with L = floor(4(T/100)^(2/9)) = 7 lags at T = 2190, without the HLN correction.

  | pair | model | registered statistic | Newey-West | p |
  |---|---|---|---|---|
  | BTC | ridge | 10.83 | 8.80 | < 0.001 |
  | BTC | xgboost | 6.18 | 5.84 | < 0.001 |
  | ETH | ridge | 8.50 | 7.24 | < 0.001 |
  | ETH | xgboost | 5.87 | 5.76 | < 0.001 |

  Climatology stays significantly better on log-loss.
- **A fills column** now counts the orders each baseline actually executed. The orchestrator gives you the official figures if you need them.

**The ml-engineer's bolder idea, for §11 and not for tonight.** Turn H1 into an adversarial search. For each honest model and each cut, an optimizer such as CMA-ES or random restarts searches the later bars for any change that moves the decision at t. The report would then give the largest change any search could cause, which must be exactly 0, instead of "0 flags at 8 cuts × 2 directions".

## Updated 2026-10-28 (2) — the edits for the page, pass 1

The page is built from parts in the orchestrator's scratch: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\explainer\`.
- The parts are `part1.html`, `part2.html`, `part3.html` and `part4.js`.
- `python -I build.py` assembles them into `banc-honnete.html`.
- Edit the parts, never the built file, and rebuild after each pass.

Pass 1 covers what is final now. Pass 2 comes later, after the H1 round 2 and the official runs: §7's results, the provenance counts, the status pill and §11.

**1. §9, the `kani-resultats` paragraph.** Fill it from these facts (Kani 0.68.0, CaDiCaL).
- **The contracts.** They live in `crates/risk/src/spec.rs`, compiled only for Kani and for the tests. Each is a predicate that admits exactly one result, and none divides.
  - The floor of a notional is pinned by `f·S ≤ q·p < f·S + S`.
  - `max_qty_within` is pinned by `q·p ≤ cap·S < q·p + p`, unless q saturated.
  - A lot multiple is `r % lot == 0`, the one remaining division.
  - `#[cfg_attr(kani, kani::ensures(...))]` attaches them to `product`, `notional_floor`, `notional_ceil` and `max_qty_within`. `round_down_to_lot` also gets `requires(qty >= 0 && lot > 0)`. No code changed.
- **Proven (VERIFICATION SUCCESSFUL), the new `proven` job, 4 harnesses:**
  - `buy_gates_block_every_buy`, 1,300 s;
  - `sells_are_never_gated`, 28 s;
  - `product_contract`, the contract of the exact 128-bit product, 112 s;
  - `exact_never_wraps`, 185 s: the specs' i64 × i64 products never wrap in i128.
- **Assumed, not proven.** The contracts of `notional_floor`, `notional_ceil`, `max_qty_within` and `round_down_to_lot` each ran out of time at 40 minutes.
  - Small experiments show why. Two copies of the same 64×64→128 multiplier take about 30 s to prove equal. `f = p / 1e8 ⇒ f·S ≤ p < f·S + S` alone ran past 7 minutes.
  - What blocks is nonlinear integer arithmetic. A lot multiple is `(n·l) mod l = 0`, the order cap needs `a ≤ b ⇒ a·p ≤ b·p`, and P2 needs `(a+b)·p = a·p + b·p`. SAT solvers handle such facts badly.
- **Stubs.** The `check_order`, `approval` and `sell_outcome` harnesses now replace the helpers by their contracts (`stub_verified`).
  - The formulas shrink by a factor of 2 to 3, for example from 1.43 M to 595k variables for `check_order_full`.
  - It is not enough: they still run out of time at 30 to 40 minutes.
  - And even if they finished, their verdict would hold only as far as the four unproven contracts hold. Say this plainly: no verdict rests on an unproven contract today, because none of those harnesses finished.
- **Flags.** Every harness runs with `-Z function-contracts -Z stubbing --no-assert-contracts`. The last flag keeps a harness that calls a helper without stubbing it from re-proving the helper's contract inline at every call. Without it, `sells_are_never_gated` grew from 312k to 489k variables and stopped finishing in 10 minutes.
- **Tests.** Five property tests, 4,096 cases each, check that each predicate accepts the helper's result and rejects its neighbours. These tests are, for now, the only evidence for the four unproven contracts.
- **Next.** The 17 open harnesses go to GitHub's `open` job, at 340 minutes each.
  - Other solvers (Bitwuzla, z3) were downloaded, but not run: the permission system refused to run downloaded binaries, and Abdou will decide.
  - The bolder idea: prove the three nonlinear lemmas once, in a tool made for them (Verus or Lean), and hand them to Kani as verified stubs. SAT would then check only the bit-level structure, which is what it does well.
- **The orchestrator re-runs the four proven harnesses itself.** It will confirm or correct the times before publishing.

**2. Part 1, the tripwire paragraph (about line 426), and wherever the page counts entry points.** After Audit A phase 2:
- **The holes.** The auditor found two read paths that passed both guards (D1), and a one-line switch for the seal (D2).
  - D1, path A: a helper module written at run time, put on `sys.path` and imported under an innocent name, read the raw zip archives with `zipfile` and `pl.read_csv`.
  - D1, path B: a Parquet read through `sys.modules["pyarrow.dataset"]` and `make_fragment`.
  - D2: `monkeypatch.setattr(protocol, "vault_open", lambda: True)` opened the vault silently.
- **The fixes.** The tripwire now goes up before any test module is imported, in `pytest_configure`. So a reader captured at import time, in a dict or a `functools.partial`, is already the wrapper.
  - It wraps 50 entry points, up from 34. The new ones include `zipfile.ZipFile` for reading, the CSV, JSON and other text readers of polars, and the pyarrow dataset constructors.
  - A path rule makes any wrapped reader trip when it is handed a path under `data/`.
  - It watches the seal. A test cannot rebind `seal`, `vault_open`, `VAULT` or the loaders, and a `seal` call that returns a vault row fails the test, however the seal was switched off.
  - The static guard now flags `sys.modules`, edits of `sys.path`, and any patch of the seal.
  - `tests/test_tripwire.py` has 71 tests.
  - The orchestrator re-ran both of the auditor's evasion files: all 13 of their tests trip before reading anything, and the static guard flags them (20 and 3 findings).
- **The limits, plainly.**
  - It runs only under pytest.
  - `pyarrow._parquet` called directly, `ctypes`, a subprocess, or `open()` with a hand-written parser still pass.
  - The structural fix is the physical seal of iteration 3: vault rows in a separate tree, `data/vault/`, unreachable without both keys.

**3. An honest sentence on a vault exposure,** near the tripwire paragraph or the seal. To prove D1 and D2, the auditor's evasion tests read BTCUSDT 1d vault rows outside the seal: the 31 bars of the 2026-03 archive, and the vault months of the Parquet file.
- They assert row counts. One of them prints the first bar of March 2026 when it succeeds.
- No model, feature or decision uses these rows, and the auditor writes no model code.
- The vault stays the final test set, with this exposure recorded in `docs/PROTOCOL.md`. Proofs of a hole now run against fake data only.

**4. Part 1, about line 428.** "qui se reproduit à l'octet près avec `--check`" must add the nuance already used in part 2: on the same machine, with the same library versions.

**5. §7, the cheater's lot cap.** Use the table and the sentence of "The cheater's lot cap (m2)" in the section above. Do not touch the rest of §7 in this pass: the H1 round 2 is changing it.

**6. Survivorship (D3), if the page speaks of the universe.**
- 140 of the 568 daily files end before 2025-11-30: 61 members and 79 other pairs. 213 of the 274 members trade to the end of 2025.
- 19 of the 99 rebalances have fewer than 30 members, all of them up to 2019-04: 2019-01 has 19, 2019-02 has 19, 2019-03 has 22 and 2019-04 has 28.
- Residual risk (D4): a pair whose archives Binance removed from the bucket altogether is invisible, and nothing can detect it.

Keep the page's voice, its demos and its structure. Run `node --check part4.js` if you touch it, and check that the tags balance. Report the paragraphs you changed and anything you could not place.

## Updated 2026-10-28 (3) — H1, round 2 (for pass 2 of the page)

This section supersedes the perturbation design and the H1 table of the "H1 strengthened" section above. The figures come from `runs/2026-10-28-h1-fix2`, and the official run will confirm them.

**The second hole the audit found (M2).** After round 1, every *value* of every later bar moved, but no *ratio inside a bar* moved:
- on bar t+1, the range high/low and the body |ln(close/open)|;
- on every later bar, the VWAP quote_volume/volume, the taker VWAP, and the trade size volume/trades.

The reason is that one factor scaled the open and the close of bar t+1, and one factor scaled `volume` and `quote_volume` together. The auditor planted six ratio leaks, including the f_range_1 feature read one bar early: all six were flagged at 0 of 8 cuts, in both directions. The lesson for the page: "every value moves" is not "every function of two values moves", and a shifted feature reads a function.

**The fix.**
- **Bar t+1 is built, not drawn, against a window of 91 bars (t−90 to t).**
  - In the ×3 run, every quantity is set to at least 1.5 × the window's maximum and at least 3 × its own original value.
  - In the ÷3 run, at most the window's minimum / 1.5 and at most its original value / 3.
  - The quantities: every price, the body, both wicks, the range, the VWAP and the taker VWAP, volume, quote volume, trades, the trade size, and both taker shares. The shares are pushed through 1 − share, since they are bounded by 1.
  - So any comparison between bar t+1 and one of the 91 bars before it flips in exactly one of the two runs, by construction.
- **Colour.** Bar t+1 is green in the ×3 run and red in the ÷3 run, a doji included.
  - The cheater (long iff close[t+1] > open[t+1]) therefore changes in exactly one run per cut.
  - That is why its flags now read ×3 1/8 and ÷3 7/8 on BTC (2/8 and 6/8 on ETH): 7 of BTC's 8 bars t+1 were green, so ÷3 catches those 7 and ×3 the red one.
  - The old swap of the body flipped the colour in both runs, which is why it read 8/8 in each direction. It could not flip a doji.
- **Bands for thresholds (log units).**

  | | ×3 run | ÷3 run | thresholds that flip by construction |
  |---|---|---|---|
  | body | at least 10 % | at most 0.25 % | from 0.25 % up to, not including, 10 % |
  | each wick | at least 5 % | at most 0.125 % | |
  | range | at least 20 % | at most 0.5 % | from 0.5 % up to, not including, 20 % |

- **Later bars.** New random wicks; independent factors for volume and trades; the VWAP and the taker VWAP drawn inside the bar's new low–high range. Every ratio of two perturbed columns must change on every later bar. Coverage asserts it at run time, and only exact zeros are exempt.
- **`?` cuts.** Where a quantity cannot cross, the cut is marked `?`, reported, and left out of the verdict. Two examples: a taker share already at 1 at t and at t+1, and a bar without volume. A `?` never counts as a pass. No cut was `?` in this run.

**The auditor's six leaks against the new perturbation (BTCUSDT 1d):**

| leak | ×3 | ÷3 | either | before |
|---|---|---|---|---|
| range, t+1 vs t | 6 | 2 | 8/8 | 0/8 |
| range > 3 % | 4 | 4 | 8/8 | 0/8 |
| body > 2 % | 6 | 2 | 8/8 | 0/8 |
| VWAP, t+1 vs t | 3 | 5 | 8/8 | 0/8 |
| taker VWAP, t+1 vs t | 3 | 5 | 8/8 | 0/8 |
| trade size, t+1 vs t | 3 | 5 | 8/8 | 0/8 |

`close[t+2] > close[t+1]` is still caught at 7 of 8 cuts: likely, not certain, and the run says so.

**H1, round 2. Verdict: H1 holds** (runs/2026-10-28-h1-fix2, 391 s with 6 processes).
- The 11 honest models have 0 violations on both pairs.
- The 20 controls are the cheater, 14 probes, off-by-one-return, the transductive scaler and 3 unpurged models. All 20 sit exactly on their predicted pattern, and no cut is `?`.
- Flags ×3 / ÷3; every "either" is 8/8 except the last two lines:

  | control | BTC | ETH |
  |---|---|---|
  | cheater | 1/7 | 2/6 |
  | trades | 4/4 | 4/4 |
  | quote volume | 4/4 | 4/4 |
  | taker share (quote) | 5/3 | 2/6 |
  | taker share (base) | 5/3 | 2/6 |
  | range | 6/2 | 5/3 |
  | body | 4/4 | 4/4 |
  | VWAP | 3/5 | 2/6 |
  | taker VWAP | 3/5 | 2/6 |
  | trade size | 3/5 | 2/6 |
  | range > 3 % | 4/4 | 3/5 |
  | body > 2 % | 6/2 | 4/4 |
  | range, continuous | 8/8 | 8/8 |
  | VWAP vs close, continuous | 8/8 | 8/8 |
  | close vs 6 bars back | 3/5 | 3/5 |
  | off-by-one-return | 8/8 | 8/8 |
  | transductive scaler | `...XXXXX` | `...XXXXX` |
  | 3 unpurged models | `...XXX..` | `...XXX..` |

- Positive control: unchanged (+0.9749 BTC, +0.9798 ETH).
- `h1.md` now ends with a provenance block: commit, code digest and manifest digest.

**What §7 of the page should become in pass 2.**
- The perturbation bullets describe the final design above.
- The results list gives the official run.
- The M1 section tells the two holes, columns and then ratios. Both were found by the auditor planting leaks the bench had never seen.
- "Ce que H1 ne prouve pas" gives the honest limits: the time stamps, two bars ahead and more (likely, not certain), thresholds outside the band, `?` cuts.
- The demo 7 note may stay as it is. Its body swap is still a fair simplified picture, but say in one clause that the code now sets the colour per direction.
- Also correct the factual error noted earlier in "Trois sortes de coupes": the transductive scaler is caught at the 5 cuts that are not a month's last bar.

## Updated 2026-10-28 (4) — H1, round 3, and the rest of pass 2

This section supersedes the perturbation design and the H1 table of section (3). The figures come from `runs/2026-10-28-h1-fix3`. H1 is deterministic: the official run, on the committed tree, must give the same table, and the orchestrator will confirm it.

**The third hole (Audit A, second re-check).** Two Major findings.
- **The candle's shape did not flip.** When the floors bound, body : upper wick : lower wick came out 2 : 1 : 1 in both runs. So functions of three or four prices of bar t+1 moved the same way in both runs, and leaks on them were flagged at only some cuts:
  - close location (close − low)/(high − low) against bar t's: 3/8;
  - fill body/range against bar t's: 2/8;
  - hammer (lower wick > 2 × body): 3/8;
  - close location > 0.5: 6/8.
- **A `?` was read from the outcome.** `?` was decided from the perturbed bars. A regression that stopped pushing the VWAP would have given the VWAP peeker `????????`, and H1 would have "held". The lesson for the page: a verdict must never be computable vacuously.

**The fix.**
- **Opposite shapes, by construction.**
  - ×3 run: a small, green, body-dominant bar. Its fill is at least 0.8 and its close location at least 0.75. Its body is at most 0.4 %, each wick at most 0.05 %, so its range is at most 0.5 %. Its prices, VWAPs, volumes, trades and shares are still above the window: it is a small candle at a high level.
  - ÷3 run: a big, red bar dominated by its upper wick. Its fill is at most 0.2, its close location at most 0.25, and its upper wick at least 3 × its lower wick. Its body is at least 10 % and each wick at least 1 %, so its range is at least 50 %.
  - So body, wicks and range are now **below** the window in the ×3 run and **above** it in the ÷3 run, the opposite of round 2. Every other quantity keeps its round 2 direction.
  - Fill and close location are pushed against bar t only. A near-doji in the window would otherwise need an unbounded range: the first version blew a wick up until the prices underflowed.
  - Whatever freedom remains is drawn per run: how the ×3 run's wick budget splits, the body factor, and the ÷3 run's wick factors.
- **Bands** (log units for body, wicks and range). A threshold flips by construction only inside them; each band includes its lower end and excludes its upper end.

  | quantity | band |
  |---|---|
  | body | 0.4 % to 10 % |
  | each wick | 0.05 % to 1 % |
  | range | 0.5 % to 50 % |
  | fill | 0.2 to 0.8 |
  | close location | 0.25 to 0.75 |
  | each taker share | from min_W s / 1.5 to 1 − (1 − max_W s) / 1.5, on its own window |

- **A sound `?`.** A `?` is predicted from the original bars, before perturbing, for declared reasons only:
  - a bar without volume;
  - a reference that is undefined, or at a bound of its quantity;
  - a count of 1 that cannot go lower;
  - a reference before the first bar.

  Any other failure to cross puts the control off its pattern, and H1 fails. A control that is `?` at every cut fails H1 too. A test turns the VWAP push off, and H1 fails, as it must.
- **What stays probabilistic, stated in the report** (the auditor's script, BTCUSDT, final perturbation):
  - comparisons with bars older than t−90: 8/8 at t−120 and 7/8 at t−364, only where the pushed level happens to clear the old bar. A 365-day-high breakout at 2022-11-21 no longer flips;
  - the other shape functions: the wick ratio at 4/8 and the hammer at 3/8;
  - the close-to-close magnitude |ln(close[t+1]/close[t])|: at least ln 1.5 in both runs, so it cannot flip by construction (4/8);
  - functions of several later bars: `close[t+2] > close[t+1]` is now caught at 8/8, but not by construction.

**H1, round 3. Verdict: H1 holds** (6 min 31 s, 6 processes).
- The 11 honest models have 0 violations on both pairs.
- 25 controls: the cheater, 19 probes, off-by-one-return, the transductive scaler and 3 unpurged models. All 25 sit exactly on their predicted pattern.
- The cheater is flagged at 16 of 16 cuts.
- One `?`, on BTCUSDT: the fill peeker at 2023-06-30. Bar t's fill was 0.01382, below the 0.02 the design aims for in the ÷3 run. The page can use it as an example of a declared `?`. Do not claim that the ÷3 fill never goes below 0.02: the audit is checking whether that floor is a target or a guarantee.
- Flags ×3 / ÷3; every "either" is 8/8:

  | control | BTC | ETH |
  |---|---|---|
  | cheater | 1/7 | 2/6 |
  | trades | 4/4 | 4/4 |
  | quote volume | 4/4 | 4/4 |
  | taker share (quote) | 5/3 | 2/6 |
  | taker share (base) | 5/3 | 2/6 |
  | range | 2/6 | 3/5 |
  | body | 4/4 | 4/4 |
  | VWAP | 3/5 | 2/6 |
  | taker VWAP | 3/5 | 2/6 |
  | trade size | 3/5 | 2/6 |
  | range > 3 % | 4/4 | 5/3 |
  | body > 2 % | 2/6 | 4/4 |
  | close location | 2/6 | 4/4 |
  | fill | 4/4 | 3/5 |
  | close location > 0.5 | 2/6 | 3/5 |
  | upper wick | 5/3 | 4/4 |
  | lower wick | 2/6 | 1/7 |
  | close vs 6 bars back | 3/5 | 3/5 |
  | range, continuous | 8/8 | 8/8 |
  | VWAP vs close, continuous | 8/8 | 8/8 |
  | off-by-one-return | 8/8 | 8/8 |
  | transductive scaler | `...XXXXX` | `...XXXXX` |
  | 3 unpurged models | `...XXX..` | `...XXX..` |

- Positive control: unchanged (BSS −0.1246 on the real labels and +0.9749 one bar early on BTC; −0.0874 and +0.9798 on ETH).

**The arc of §7, for the page.** Three rounds, each one found by the auditor planting a leak the bench had never seen:
1. columns, `trades` among them;
2. ratios inside a bar;
3. the candle's shape, together with a verdict that could be computed vacuously.

"Every value moves" is not "every function of values moves". Two runs per cut cannot flip every function of a bar. So: control what can be controlled, randomize the rest, and say plainly what stays probabilistic. The ml-engineer's bolder idea, which the page may mention as an idea only: a small factorial design of runs over colour, fill, close location, wick ratio and move size, plus an adversary that searches for the leak H1 misses most.

**The rest of pass 2.**
- **Kani, §9.** Use the orchestrator's re-run on the final crate, in WSL2, four at a time: `sells_are_never_gated` 23 s, `product_contract` 87 s, `exact_never_wraps` 143 s, `buy_gates_block_every_buy` 945 s; 16 minutes in all. These replace the agent's times in section (2).
- **The guards, part 1.** Three more rounds after the one in section (2), each re-audited. The final figures:
  - the tripwire wraps 53 entry points;
  - the static corpus has 137 cheater snippets, each of which must be flagged, and 25 controls, none of which may be;
  - `test_tripwire.py` has 180 tests;
  - from round 2 on, every evasion case the auditor ran used fake data only: 56 cases in its round 2 re-check, then the path spellings of round 3.

  Rounds 2 to 4 together:
  - every test gets an empty data root of its own;
  - forged keys no longer open the real `data/` through the loaders;
  - a raw archive is read only when every file on the stack holds the allowance, the test included;
  - the allow-list is read-only, and a check after every test ends the session if a guard was taken down;
  - the static guard flags the attributes that take a guard apart (`__globals__`, `__closure__`, `__bases__`, ...) and bans `runpy`, `gc` and `ctypes`;
  - round 3 closes a direct read of the real `data/` by a file allowed to read Parquet, and `zipimport`;
  - round 4 closes rarer spellings of a path: a `file://` URI, the `\\?\` prefix, and a path relative to a pyarrow sub-tree filesystem.

  The page should state the lesson plainly: a guard inside the process can be taken down from inside the process. These guards stop accidents, not sabotage. The structural fix is to keep the vault rows where the process cannot read them without both keys, which is planned for iteration 3.
- **Provenance.** 843 Parquet files, manifest digest `230e28e8…`; the code digest covers 57 source and lock files. The official run's commit will be given.
- **Status pill, §11 and footer.** The orchestrator will give the final status and the bolder next step once `docs/ITERATIONS.md` holds them.

## Updated 2026-10-28 (5) — H1, round 4 (final)

This section supersedes section (4)'s colour design, its H1 table and its run. Everything else in section (4) stands. The figures come from `runs/2026-10-28-h1-fix4`, and the official run must reproduce them.

**The fourth hole (Audit A, third re-check).** Colour and size were tied: the two runs gave a small green bar and a large red one, and never the other two pairings.
- So a leak on a small red next bar was false in both runs. Think of "yesterday was a small red candle", read one bar early.
- It flipped only where the original bar t+1 was already small and red: 0 of 8 cuts on BTC.
- Such a feature would be caught with a probability of about 0.9 per model, not 1.

**The fix: the colour alternates with the cut's position in the cut list.**

| | ×3 run | ÷3 run |
|---|---|---|
| even cuts | small, green, closes near its high | big, red, upper wick ≥ 3 × lower, closes near its low |
| odd cuts | small, red, closes near its low | big, green, lower wick ≥ 3 × upper, closes near its high |

- Only colour and close location change direction. Every other push is unchanged, and each comparison still flips in exactly one run.
- Each of the four colour × size cells occurs at half the cuts.
- Two new permanent controls, `red-and-small` and `green-and-big`. Each one's predicted pattern is computed cut by cut from the cell each run realises by design and from the original bar t+1. It is exact, not a probability.
  - BTC: `.X.X.X.X` and `XX.XXX.X`.
  - ETH: `.XXX.X.X` and `XX.XXX.X`.
  - The auditor's leak "red and body < 1 %" goes from 0/8 to the four odd cuts, which are all the cuts where it can change.
- **Exact floors.**
  - The ÷3 fill never goes below 0.02.
  - The close location's target never goes past 0.02, or 0.98 at mirrored cuts.
  - A reference past either floor is a declared `?`.
- **An honest oddity worth one sentence.** The ÷3 bar can be absurd: its range reached 16.6 log units, a factor of about 16 million between high and low. The fill floor sets the range to body / 0.02 when bar t's fill is small. The prices stay valid, no venue would ever print such a bar, and the honest models do not care. The report says so.

**H1, round 4. Verdict: H1 holds** (7 min 3 s, 6 processes).
- The 11 honest models have 0 violations on both pairs.
- 27 controls, all on their predicted pattern.
- The cheater is flagged at 16 of 16 cuts.
- One `?`, the same as before: BTC fill at 2023-06-30.
- **Rows that change from section (4)'s table**, ×3 / ÷3:

  | control | BTC | ETH |
  |---|---|---|
  | cheater | 3/5 | 4/4 |
  | fill | 4/3 (7/8 either: the miss is the declared `?`) | 3/5 |
  | close location > 0.5 | 4/4 | 5/3 |
  | red-and-small (new) | 4/0, so 4/8 either | 4/2, so 5/8 either |
  | green-and-big (new) | 4/4, so 6/8 either | 4/4, so 6/8 either |

  Every other row is unchanged.
- **The auditor's scripts on the final perturbation:**
  - `close[t+2] > close[t+1]` at 7/8;
  - hammer 5/8, wick ratio 4/8, |return| 4/8: probabilistic, as stated;
  - the 2022-11-21 breakout still does not flip.

**The arc of §7 now has four rounds:** columns; ratios inside a bar; the candle's shape, with the vacuous `?`; colour × size. Each one was found by planting a leak the bench had never seen. The ml-engineer's bolder idea for iteration 3, an idea only: make H1 adaptive. Learn where the perturbation is least sure to catch a leak, and add a designed run there.

**Update, 2026-10-28, after the official runs (orchestrator).**
- The first official H1 run (commit 9c414ca) held. CI on Linux then showed that the transductive-scaler control was caught by rounding alone: its ridge standardized the features again, which undid the leak exactly, so its outputs moved by 1e-16 at most. See `rounding-fix.md`.
- **The fix.** A control is caught at a cut only where an output moves by more than 1e-9, turns missing or stops being missing. An honest model must still not move by a single bit. The scaler now feeds a ridge that centres its features but keeps the scale it is given.
- **The second official run** (commit 881a63c, `runs/2026-10-28-h1-2/`, 7 min 39 s with 4 processes) holds.
  - Honest models: 0 violations, and no output moved at all.
  - 27 controls on their predicted pattern, none with a rounding-only cut.
  - Smallest catch: 7.1e-4 on BTC and 7.3e-4 on ETH (climatology-unpurged). The transductive scaler's is 1.8e-3 and 8.0e-4.
  - The cheater is caught at 16 of 16 cuts.
- **Report n°1** does not use the changed code. Re-run at 881a63c with `--check`, only its commit and code digest lines differ. It takes about 75 s with 2 processes.
- **The page** gained a section on the control caught by rounding and a ninth question in « Teste-toi ». It cites H1 at 881a63c and Report n°1 at 9c414ca.
- **Published** after the orchestrator's checks: https://claude.ai/artifact/VDdujVgLyyi2PcMBnaM1Sd.
