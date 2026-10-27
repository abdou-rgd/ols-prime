# Brief — Facts at the source (iteration 2)

- **Agents:** two `scout`s (sonnet, medium), one per part. Read-only.
- **Motto:** we should be more ambitious: look one step further than the question.
- **Report.** Up to 900 words per part (this brief overrides the usual 600). One section per question, ending with a verdict. Quote numbers exactly, with the URL and the date you read them.
- **Context.** Read `docs/PROTOCOL.md` (cost table, metrics) and the Lessons of iteration 1 in `docs/ITERATIONS.md` first.
  - A summarizing fetch tool paraphrases numbers. Quote the sentence that carries each number.
  - Coinbase's pages answered HTTP 403 in iteration 1.

## Part A — Venues and fees (first scout)

The protocol's cost table, as of October 2026: Binance spot 0.10 % / 0.10 %, Coinbase Advanced entry tier 0.25 % / 0.50 %, Kraken Pro entry tier 0.40 % / 0.80 % (maker / taker). Report n°1 uses it. Check each number at the source today.

1. **Coinbase Advanced, for a retail client in France (Coinbase's EU entity).** The entry-tier maker and taker rates, from Coinbase's own pages. Try the help-center fee articles, the Advanced fee page, and the API documentation of fee tiers (`transaction_summary`). If the pages still refuse you, say which URLs you tried and what answered.
2. **Kraken Pro.** The entry tier, from Kraken's fee-schedule page and its support articles. Our table says 0.40 % / 0.80 %; earlier schedules said 0.25 % / 0.40 %. Settle which is current.
3. **Cheaper venues open to French residents.** Venues authorized under MiCA (check the ESMA register, or the national regulator's register) that serve French residents spot BTC and ETH, against EUR or USDC: for example Bitvavo, Bitstamp, Bitpanda, OKX, Bybit EU, Crypto.com, Gemini. For the three cheapest: the entry-tier maker and taker rates, the pairs, the authorizing authority and the date of authorization.
4. **Binance and France.** A primary source (AMF, Binance's own announcement) for Binance closing to French residents on 2026-07-01. Iteration 1 found only press reports.

## Part B — Formulas and tools (second scout)

1. **Sharpe ratio statistics.** Verify against the papers, and give equation numbers.
   - Lo (2002), "The Statistics of Sharpe Ratios": the iid standard error `√((1 + SR²/2)/T)`, and the annualization factor η(q) under serial correlation.
   - Mertens (2002), for non-normal iid returns: `Var(SR̂) = (1 + SR²/2 − γ₃·SR + (γ₄ − 3)/4·SR²)/T`, with γ₄ the non-excess kurtosis.
   - Opdyke (2007), for non-iid returns.
2. **PSR and DSR.**
   - PSR (Bailey and López de Prado 2012): `PSR(SR*) = Φ[(SR̂ − SR*)·√(T − 1) / √(1 − γ₃·SR̂ + (γ₄ − 1)/4·SR̂²)]`. Check `T − 1`, non-excess γ₄, and that SR is per period, not annualized.
   - DSR (Bailey and López de Prado 2014): `SR₀ = √V[SR_n]·((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))`, with γ Euler's constant, and DSR = PSR(SR₀). How do they estimate V[SR_n]? What happens at N = 1?
3. **Diebold-Mariano** (1995) with the Harvey, Leybourne and Newbold (1997) correction: the factor `√((T + 1 − 2h + h(h − 1)/T)/T)`, the Student t with T − 1 degrees of freedom, and the HAC variance with h − 1 lags.
4. **Stationary bootstrap** (Politis and Romano 1994) and its automatic mean block length (Politis and White 2004, corrected by Patton, Politis and White 2009): the formula, and the reference implementation (for example the `arch` package's `optimal_block_length`).
5. **Calibration error.** Top-label ECE (Guo et al. 2017) versus class-wise ECE: definitions, binning, and the known biases of equal-width bins.
6. **PyO3, rust-numpy, maturin.** The latest releases as of October 2026 and their MSRVs; the recommended way to keep `cargo test` working with the `extension-module` feature; how a uv project depends on a local maturin crate (`[tool.uv.sources]`, path, editable or not), on Windows too.
7. **Kani 0.68 function contracts.** The syntax of `kani::requires`, `ensures`, `proof_for_contract` and `stub_verified`; the unstable flag; known limitations.
   - Does Kani 0.68 offer an SMT back-end (bitwuzla, z3, cvc5) for a harness, or only SAT solvers?
   - Does any 2025–2026 Kani feature help with 128-bit multiplication and division?
