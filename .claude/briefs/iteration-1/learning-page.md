# Brief — iteration 1 learning page

Sent on 2026-10-25 to a general-purpose agent (opus, high), because the `teacher` agent did not exist yet. From iteration 2 on, `teacher` takes this task; its standing rules are in `.claude/agents/teacher.md`. Scratch paths are those of the session that ran iteration 1.

Write a detailed, intelligible explainer page about the OLS Prime project, in **French**, as one self-contained HTML file. The reader is Abdou, who builds the project with Claude Code agents and wants to *learn and understand* it, not just watch. They will read it tomorrow on a phone during their commute, so it must read well at about 400 px wide. Their motto, which applies to this page: "we should be more ambitious" — ambitious in depth and clarity, never in claims.

## Sources (read them; every claim must come from them)

Repo root `C:\Users\abdou\ols-prime` (read-only for you; never edit it):
- `CLAUDE.md`, `README.md`, `docs/DESIGN.md`, `docs/PROTOCOL.md`, `docs/HYPOTHESES.md`, `docs/ITERATIONS.md`, `docs/RESULTS.md`, `docs/DATA.md`, `.claude/agents/*.md`, `.github/workflows/ci.yml`.
- Code: `crates/ols-core/src/*.rs` (ridge primal/dual, RLS, Matrix), `crates/backtest/src/{lib,sim,metrics,types}.rs`, `crates/risk/src/{lib,proofs}.rs`, `python/olsprime/**/*.py`.

State of iteration 1, as verified by the orchestrator on 2026-10-25 (nothing is committed yet):
- Rust: 121 tests pass (backtest 56, ols-core 25, risk 40); fmt, clippy `-D warnings` and rustdoc clean. Python: 35 offline tests pass.
- Data on disk: BTCUSDT and ETHUSDT, 1 h and 1 d, 2017-08 → 2026-09, 440 archives with verified SHA-256 (details and quirks in `docs/DATA.md`).
- Kani 0.68.0 was installed in WSL2 after enabling SVM in the BIOS (the first Ubuntu install failed: virtualization was off in the firmware). The 14 proof harnesses are running now; their results are **not known yet**.
- The `leakage-auditor` audit is running now; its results are **not known yet**.
- Findings of the iteration:
  - Binance Vision data is CC BY-NC-SA 4.0 since 2026-08: non-commercial, derived outputs share-alike; live trading with models derived from it is a phase-6 question.
  - Binance and Kraken fees and the BTCUSDT/ETHUSDT filters were re-verified; Coinbase fees could not be.
  - Coinbase, Kraken, Bitstamp and Bitvavo hold MiCA authorizations passported to France (per ESMA's register as read on 2026-10-25).
  - The README's 14 citations were checked and three wordings fixed.
- Engineers' notable choices:
  - in backtest, a dust rule (orders ≤ 1e-12·equity skipped), a 2^50-lot cap, full exits below min notional skipped;
  - in risk, `Approved` cannot be forged (private fields), Kani `cover!` checks against vacuous proofs, extra `InvalidInput` checks.
- Never present a hypothesis (H1–H8) as a result: no experiment has run (`docs/RESULTS.md` is empty).

## What the page must teach (adapt the order if it reads better)

1. The project in one screen: what OLS Prime is, why crypto is only the test bench, why from scratch (an LLM's look-ahead contamination), the motto.
2. Prerequisites, from first principles: linear regression and least squares; the Gram matrix XᵀX; ridge and λ (bias–variance); attention in Transformers (Q, K, V, softmax), so the comparison lands.
3. OLS as attention (Goulet Coulombe): derive ŷ = Σ wᵢ yᵢ with wᵢ = xᵢᵀ(XᵀX + Λ)⁻¹x_q step by step; why weights are signed; the proof that a free intercept makes the weights sum to one; primal (p × p) versus dual (n × n) forms; why (XᵀX)⁻¹ must be fitted on the past only.
4. RLS with forgetting: the update equations, the Sherman–Morrison intuition, exponential memory, the equivalence with weighted ridge, covariance wind-up.
5. Matrix, the memory: sufficient statistics and why they add up; blocks and the binary-counter roll-up; reading any range from O(log n) nodes; exact signed contributions of any period; the OptMem analogy; causality (an entry joins only when its outcome is known).
6. The future encoder and decision head (phase 3–5, planned): OLS-attention heads, T ≫ d_head, typed options down/neutral/up, cross-entropy + Brier, calibration, the decision rule against round-trip costs.
7. The honest bench: look-ahead and its forms; walk-forward, purge and embargo; the development / validation / vault split; future-invariance and cheater tests; the metrics (log-loss, Brier skill score, calibration, Diebold–Mariano, Sharpe and why its standard error is about 1.1 over ten months, deflated Sharpe); costs (fees, slippage, min notional, lot step).
8. The backtest in detail: decide at the close of t, execute at the open of t+1; fills, fees, slippage; solvency by construction; the dust rule; the metric definitions (365-day annualization, turnover, exposure).
9. The risk kernel: why integers (satoshis, micro-USDT, i128); the invariants P1–P7; property-based testing (proptest) versus model checking (Kani, CBMC, SAT); what `kani::any`, `assume` and `cover!` do, and what a vacuous proof is.
10. The data: Binance Vision archives, checksums, the ms → µs switch of 2025-01-01, gaps and off-grid bars, the point-in-time universe (survivorship bias, delisted pairs, stablecoins and leveraged tokens, the LUNA ticker reuse), the license.
11. The architecture: repository layout, the data flow from raw archive to decision, the team of agents (orchestrator plus six roles, which model and why), CI.
12. What iteration 1 delivered (a table), then a section with `id="kani"` and another with `id="audit"`, each holding a short visible note "Résultats en cours — ajoutés par l'orchestrateur" that the orchestrator will replace.
13. What comes next: the roadmap phases, iteration 2 in detail, the hypotheses H1–H8 with their decision rules and prior expectations, and the open questions (Coinbase fees, the phase-6 license question).
14. A glossary, and the references with links (only those cited in the repo).

## How

- Explain like a strong teacher talking to a smart builder: intuition first, then the maths, then a small worked example with real numbers (the repo has some, e.g. the risk kernel's 74,000-satoshi example, the five-point Matrix doc example in `crates/ols-core/src/lib.rs`). Short sentences, plain French; keep standard English terms (ridge, attention, walk-forward) and define them once. End each big section with a "À retenir" box of 2–4 lines.
- Add small interactive demos in vanilla JS (no framework):
  - (a) an OLS-attention playground on a fixed 1-D sample: sliders for the query x_q and λ, an intercept toggle, bars of the signed weights wᵢ, the forecast as Σ wᵢ yᵢ, and Σ wᵢ shown live;
  - (b) a Matrix explorer: push entries one by one, watch blocks roll up into the tree, pick a range and see which nodes and partial blocks cover it;
  - (c) a fee calculator: round-trip cost under the three venue schedules plus 2 bps slippage per side, and the yearly drag for a chosen number of round trips.

  Each demo must open in a filled, working state and fit a phone screen.
- Maths: use MathJax 3 with SVG output from cdnjs (for example `https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js`; check with `curl -sI` that the exact URL answers 200 before using it). SVG output needs no font files, which the page's security policy would block. Make formulas that are wider than the screen scroll horizontally inside their own container.
- Diagrams: the viewer renders Mermaid natively from `<pre class="mermaid">` blocks (load no Mermaid library); wrap each in an `overflow-x: auto` container. Inline SVG is fine for the Matrix tree.
- Navigation for a commute: a compact table of contents with anchors, an estimated reading time per section, and the last section read remembered in `localStorage` (every access in try/catch; the page must work without it).

## Page contract (the hosting platform enforces it)

- Write page content directly: no `<!doctype>`, `<html>`, `<head>` or `<body>` tags. Start the file with `<title>` (a 2–4 word name, e.g. "Comprendre OLS Prime", no explanation after a dash or colon), then `<meta name="claude:viewport-fit-sides" content="cover">`, then your `<style>`.
- Side gutter on the outer wrapper: `padding-left: max(16px, env(safe-area-inset-left, 0px))` and `padding-right: max(16px, env(safe-area-inset-right, 0px))`, vertical padding with `padding-block`. The body never scrolls horizontally; only formulas, tables, diagrams and code scroll inside their own `overflow-x: auto` boxes. Give flex/grid children holding text `min-width: 0`.
- Scripts only from cdnjs.cloudflare.com, cdn.jsdelivr.net/npm, unpkg.com, pinned to an exact version at least two weeks old; stylesheets only from Google Fonts (always with a fallback stack); inline everything else. No fetch to other hosts, no iframes, no `alert`/`confirm`, no download links, no print button.
- Both themes: define every color as a token on bare `:root` (light values), redefine them under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { … color-scheme: dark } }` and again under `:root[data-theme="dark"]`; `body` gets an explicit token background. Never a literal color that works in one theme only.
- Design: a deliberate identity drawn from the subject's own world (least squares, matrices, candlesticks, order books, proof logs), readability first: body text ≥ 17 px on phones, about 65 characters per line, a clear type scale, `text-wrap: balance` on headings. Avoid the stock AI looks (cream + serif + terracotta, near-black + acid green, purple-blue gradient hero, Inter or Space Grotesk, emoji section markers, everything centered, the same rounded card everywhere). Respect `prefers-reduced-motion`; give focus a visible state.
- Write the file to `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\explainer\comprendre-ols-prime.html` (that folder is yours alone). Do not publish it: the orchestrator publishes it. Before reporting, run one syntax check of your inline scripts (for example by extracting them and running `node --check` if Node is available) and check that every tag is closed.

Report in under 200 words, in English: the file path, the sections, the demos, the external URLs used (and their check), and anything you could not ground in the repo.
