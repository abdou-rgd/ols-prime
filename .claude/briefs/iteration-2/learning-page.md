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
