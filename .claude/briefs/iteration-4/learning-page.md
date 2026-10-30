# Brief — The iteration 4 learning page (teacher)

- **Agent:** `teacher` (opus, high).
- **When:** after the outcome of iteration 4 is recorded in `docs/ITERATIONS.md` and its rows in `docs/RESULTS.md`.
- **Motto:** we should be more ambitious: teach deeper than a summary, and let the reader break the idea.

## The page

- **Folder:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\explainer\`. The page is new, and outside the repository.
  - Build it from parts with a small `build.py`, as iterations 2 and 3 did. Edit the parts, never the built file.
- **Title:** two to four words, a name. Give the one-sentence description in your report: it goes on the gallery card.
- **Previous pages.** Read the last one in full and skim the two before. Keep their quality bar, their voice and the kind of demos they have.
  - Iteration 1: [Comprendre OLS Prime](https://claude.ai/artifact/386T8BiAjdUfbmjMZ1niGw).
  - Iteration 2: [Le banc honnête](https://claude.ai/artifact/VDdujVgLyyi2PcMBnaM1Sd), local copy `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\explainer\banc-honnete.html`.
  - Iteration 3: [L'arbre et le coffre](https://claude.ai/artifact/8AkKGG3q97u8jU2Z9BjpbK), local copy and its parts in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\explainer\` (`arbre-et-coffre.html`, `build.py`, `check.py`).
- **Identity.** Give this page its own visual identity, drawn from this iteration's world: a window of 64 bars read by attention heads, a matrix of weights whose rows sum to 1, a GPU whose bits must not move, a power curve, a ledger of contacts with the data.
- **Opening.** Open with what changed since « L'arbre et le coffre », and link back to it rather than repeating it.

## What to teach

Ground every claim in the repository (`docs/PROTOCOL.md`, iteration 4's section and change log; `docs/DESIGN.md` §1 and §4; the code under `python/olsprime/models`, `train`, `bench` and `explain`), and every number in the verified figures below.

1. **From hand-made features to learned ones.** The encoder reads a window of 64 hourly bars, eight quantities per bar (returns, gap, range, body, wicks, volume, taker share), standardized inside the fit. A linear map to d_model = 32, a position embedding, 2 pre-LayerNorm blocks of 4 heads with d_head = 8, an MLP, residuals. Why a window may be read in both directions: every bar in it is final at the decision.
2. **OLS attention as a layer** (`docs/DESIGN.md` §4).
   - W = Q̃ (K̃ᵀK̃ + Λ)⁻¹ K̃ᵀ and O = W V: each query is a regression of the window's values on its keys.
   - The constant key coordinate, unpenalized: derive why every row of W then sums to 1, and why `head` checks it.
   - λ = softplus(θ) + 10⁻³ learned per head; the Cholesky solve in fp32, never an inverse.
   - Demo: 64 points in a window, a query, the OLS weights (signed) beside the softmax weights (positive) and the linear-attention weights, with a slider on λ. Show the row sum stay at 1 while the weights turn negative.
   - The two other arms: softmax attention, and linear attention with φ(x) = elu(x) + 1 (Katharopoulos et al., 2020). Same parameters to within 8, one λ per head.
3. **Matrix over learned keys.** The key z = W_z h ∈ R⁸, the value a one-hot label, the stamp `label_known_at`. The centred ridge of iteration 3, now on learned keys, inside a 30-day window. Prefix sums in fp64, differenced before centring.
   - **Causality while training:** inside a weekly episode, an outcome joins the memory only once it is known. The rows of the episode are keyed with gradient, older entries come from a cache keyed without gradient. Draw it on a timeline.
4. **Fitting without looking ahead.** Yearly fits at 1 January; training labels known by F − 30 days; the early-stopping slice [F − 30 days, F) with labels known by F − 24 h; six epochs, the best slice loss kept; AdamW, warm-up and cosine; three seeds whose probabilities are averaged before the floor. Draw the purge on a timeline.
5. **Bits on a GPU.** Why a GPU is harder than a CPU: atomics, kernel choice by problem size, TF32. What the iteration did: deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG`, fixed batches of 4,096 rows padded with copies, a pre-check of two runs and a scope test.
   - Two stories to tell plainly. Torch's documentation moved `cumsum` on CUDA from "raises" to "deterministic" between releases, so the claim was checked on the machine. And `np.log` and `np.exp` give different last bits on Windows and on Linux, so the same commit gives two panels: Report n°3 reads a Report n°2 re-run made in WSL2. A bit is a property of a build, not only of an algorithm.
6. **H1 for the encoder.** The pre-check. The planted training-time leaks: `cheat-stamp`, `cheat-stamp-day`, `cheat-value`, `cheat-purge`, `cheat-slice`, each with its predicted pattern of cuts; the two purge cuts at F − 24 h and F − 30 days that bracket the purge window; the positive control `pos-window`, which must reach a Brier skill score of 0.5. Three verdicts, reported apart, and why none may stand in for another.
7. **H4, H5, H13 and the cost table.**
   - Non-inferiority: a margin of 0.5 % of climatology's log-loss, a one-sided bound from Diebold-Mariano with HLN, and the bootstrap's 95th percentile, both below it.
   - The testability gate: two arms that both sit at climatology meet any non-inferiority bound. Explain why that would be a vacuous "holds".
   - H5 against `matrix-select`: two systems, not two attentions at equal cadence.
   - H13's timing: warm-ups, synchronize, ABAB repeats; why analytic FLOPs favour OLS by construction when T ≫ d_head and stay descriptive.
8. **What a verdict is worth: size and power.**
   - The block sign-flip: centre the hourly difference, flip blocks of L hours, run the pass rule on each replicate. Size is how often it passes when nothing is there.
   - Why blocks of 24 hours keep only about two thirds of the long-run variance of a 24-hour overlap, weight (L − k)/L, and why L = 48 was added.
   - Power: add a shift δ, constant or concentrated in 10 % of the months, and count the passes. The smallest δ with 80 % power.
   - Demo: an autocorrelated series with sliders on L and δ, showing the rejection rate move.
   - Read H3's −0.094 % against the detectable δ: does "does not hold" mean "no difference" or "too weak to see"?
9. **`olsprime.explain`, an instrument with a ledger.** `why` (the memory's tree nodes and the per-feature parts, with two loud checks), `series` (how much of a skill a few hours carry), `sample`, `head`. Every call written to `CONTACTS.log`. Why a pattern seen on test months is exploratory and is never tested on them. The freeze: after the first call, the protocol changes only for what cannot be built as written.
10. **The five whys and their answers.** Numbers for questions 1, 2 and 5; questions 3 and 4 labelled « exploratoire, mois de test ».
11. **H1's blind spots: the future-shuffle twins.** A twin keeps the future bars' time stamps and draws their values from the past. The three kinds of miss: constant, unchanged by every twin, true blind spot. What the 7 blind spots are, and what the next adversary should plant.
12. **Registered for iteration 5: H14.** The long/flat portfolio of `matrix-select` at Coinbase's and Kraken's taker fees, and why beating always-flat is not enough when the market drifts: the 100 random long/flat chains matched on exposure and entries.
13. **La suite:** the bolder next step, and what Abdou should master before iteration 5.

Close with:
- « Teste-toi », eight to ten questions with the answers in `<details>`;
- a glossary;
- the references cited in the repository.

## Page contract (the platform's current rules, read on 2026-10-30; they win over your agent file)

- **Structure.**
  - No `<!doctype>`, `<html>`, `<head>` or `<body>` tags: the platform wraps the page in a skeleton with a charset, a viewport meta with `viewport-fit=cover`, and a small reset.
  - Start with `<title>`, then `<meta name="claude:viewport-fit-sides" content="cover">`, then `<style>`. The title must sit in the first 8 KB.
  - The skeleton pads `:root` top and bottom by the safe-area insets; keep that padding. `[hidden]` is `display: none`: toggle with `el.hidden`.
- **Gutters.**
  - `padding-left: max(16px, env(safe-area-inset-left, 0px))` and the right-hand twin, once on `body` or on one outer wrapper. The meta tag above goes in only with both paddings.
  - Vertical padding uses `padding-block`, never a `padding` shorthand that zeroes the sides.
  - A sticky header uses `top: env(safe-area-inset-top, 0px)`, with a fixed `body::before` (`content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10`) painting the band above it.
  - The body never scrolls sideways at about 400 px. Tables, formulas, diagrams and code scroll inside their own `overflow-x: auto` boxes. Flex and grid children that hold text get `min-width: 0`.
- **Themes.**
  - Every color is a token, first defined on bare `:root` with the light values.
  - Redefine the tokens under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ...; color-scheme: dark } }`, and again under `:root[data-theme="dark"] { ...; color-scheme: dark }`.
  - `body` gets an explicit token background. No component uses a literal color, and no color is defined only inside a media or `[data-theme]` block. Chart and SVG text takes theme tokens; every drawn shape has an explicit fill.
- **External files.**
  - Scripts only from cdnjs.cloudflare.com (preferred), cdn.jsdelivr.net/npm or unpkg.com, each pinned to an exact version at least two weeks old: MathJax 3 SVG from cdnjs, as before. A library's stylesheet is inlined.
  - Stylesheets only from Google Fonts, with real fallback stacks.
  - Everything else is inline. No fetch to other hosts, no iframes, no `alert`, `confirm` or `prompt`, no download links, no print button. Links to other sites are plain `<a href>`.
- **Mermaid.** The viewer renders `<pre class="mermaid">` natively, so load no Mermaid library. Draw what Mermaid cannot as inline SVG.
- **Readability.**
  - Body text at least 17 px on phones, and about 65 characters per line.
  - `text-wrap: balance` on headings, and `tabular-nums` where digits line up.
  - A visible focus state, `prefers-reduced-motion` respected, and a stable `id` on every form control.
- **Rest state.** Everything is visible at rest, with no `opacity: 0` waiting for an observer. Every demo opens filled and working.
- **Storage.** `localStorage` only for conveniences, inside try/catch.
- **Look.**
  - Write a compact token plan as the `:root` block: 4 to 6 color tokens, a display face used with restraint, a body face and a utility face for data, and a one-line layout comment.
  - No templated look. Avoid warm cream with a terracotta serif, near-black with a lone acid accent, broadsheet hairlines with dense columns, purple-to-blue gradients, Inter or Space Grotesk, emoji as section markers, everything centered, and `rounded-lg` or an accent rail on every card.
  - Numbered markers only where the order is information. Card styling by role, not on every block.
  - Copy: active voice, plain words, no em-dash asides, no "not X, but Y" framing, no colon-then-reveal sentences.
- **Size.** Well under 16 MB.

## Rules

- **French throughout,** with the narrow no-break space (U+202F) before `:` `;` `?` `!` and inside « », as the previous pages do.
- **Maths.** MathJax `\(...\)` with every backslash intact. Write with your editor tools, never through a shell heredoc: one ate the backslashes once.
- **Status of claims.** Mark planned work as planned, and hypotheses as hypotheses. Results come only from `docs/RESULTS.md` and the figures below. A hypothesis that was not run is said not run.
- **Checks:**
  - `node --check` on each inline script, extracted to a temp file;
  - every tag closed;
  - every external URL answers 200 (`curl -sI`);
  - a read at 400 px wide.
- Do not publish, and do not commit.

## Verified figures

The orchestrator appends them below once they are verified.
