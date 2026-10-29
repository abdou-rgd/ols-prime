# Brief — The iteration 3 learning page (teacher)

- **Agent:** `teacher` (opus, high).
- **When:** after the outcome of iteration 3 is recorded in `docs/ITERATIONS.md` and its rows in `docs/RESULTS.md`.
- **Motto:** we should be more ambitious: teach deeper than a summary, and let the reader break the idea.

## The page

- **Folder:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\explainer\`. The page is new, and outside the repository.
  - You may build it from parts with a small `build.py`, as iteration 2 did. Edit the parts, never the built file.
- **Title:** two to four words, a name, for example `La mémoire scellée`. Give the one-sentence description in your report: it goes on the gallery card.
- **Previous pages.** Read both first. Keep their quality bar, their voice and the kind of demos they have.
  - Iteration 1: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\explainer\comprendre-ols-prime.backup-pre-kani.html` (the latest local copy), published at https://claude.ai/artifact/386T8BiAjdUfbmjMZ1niGw.
  - Iteration 2: `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\explainer\banc-honnete.html`, published at https://claude.ai/artifact/VDdujVgLyyi2PcMBnaM1Sd.
- **Identity.** Give this page its own visual identity, drawn from this iteration's world: a memory tree over time, sealed archives and a vault, a CUSUM chart, an adversary's catch ledger.
- **Opening.** Open with what changed since "Le banc honnête", and link back to it rather than repeating it.

## What to teach

Ground every claim in the repository, and every number in the verified figures below.
1. **Matrix, the memory.** The append-only log, the binary tree of sufficient statistics over time, and `time_range(from, as_of)`.
   - Why a read is a pure function of its range, bit for bit, whatever the thread count, and why that makes H1 possible.
   - Why an outcome joins the memory only at `label_known_at`.
2. **Centred nodes.**
   - Derive the merge of Chan, Golub and LeVeque: `M(A∪B) = M(A) + M(B) + (n_A n_B / n) δδᵀ`.
   - Show why raw sums lose digits when the mean is large next to the spread.
   - Demo: two blocks of points with a slider on the mean's offset. Show the error of the raw sums against the centred merge, in float32 or float64.
3. **Ridge in centred form.**
   - The free intercept, and the penalty ρ·n_w·σⱼ² per feature.
   - Why this is ridge on features standardized inside the fold.
   - The refusal criterion: when a cell falls back to climatology.
4. **OLS attention against softmax and kNN (H3).**
   - The signed weights `wᵢ = xᵢᵀ(XᵀX+Λ)⁻¹x_q`, and what a negative weight means.
   - Demo: a 2-D cloud and a query. Show the ridge weights, softmax weights and kNN weights side by side, with a slider on ρ and on τ.
   - Use the descriptive readout of Report n°2: the share of negative weights, and the weight of the last 30 days against older entries.
5. **The hourly panel.** Point-in-time membership and survivorship; h = 24 found by time; the label known at the exit bar's open; why rows exist only in member months.
6. **The eight arms, and selection without look-ahead.**
   - Prequential log-loss.
   - The purge at `fit_time − 24 h`, drawn on a timeline.
   - The σ schedule.
   - What each arm is: climatology, matrix-select, matrix-1y, softmax-1y, knn-1y, matrix-adaptive (H10), matrix-stack (H11), matrix-cusum (H12).
7. **CUSUM (H12).**
   - Page's two-sided test, k = 0.5, and the threshold h = 4.761 set by simulation for an ARL₀ of 365 days: about 6 false alarms over the 2,191 test days.
   - Audit pass 1a first found the registered ARL of 30 days: about 72 false alarms, a coin flip on "fewer than 72 refits". Tell that story: a threshold is a claim about false alarms.
   - The input: each member's residual over its predictive standard deviation, σ̂·√(1 + 1/n_w + x̃ᵀ(M_xx + Λ)⁻¹x̃), averaged over the day's members and divided by s, fitted in the fold so the daily mean has unit variance.
   - The timing: day d's value enters at 01:00 on d + 1, once its labels are known, and an alarm acts at the next 00:00 UTC with a refit purged at − 24 h. The day that straddles a refit is left out.
   - H12's bar: no worse than monthly refits within 0.5 % of climatology's log-loss, with at most 36 fits where the monthly schedule makes 72.
   - Demo: a series with a break, the S⁺ and S⁻ paths and the threshold, with sliders on the break's size and on k.
8. **Scoring.**
   - Why the hourly series is aggregated before any inference.
   - Diebold-Mariano with 23 lags and the HLN correction: overlapping one-day outcomes.
   - Why a comparison needs two tests to pass: DM's p < 0.05 and a stationary bootstrap interval on the same side of 0.
   - The paired bootstrap of a BSS, which is a ratio: both arms resampled on the same hours.
   - Effect sizes as a share of climatology's log-loss.
   - DSR and the variants counted: 8 arms, 18 variants in all.
   - The months already seen: smoke runs scored 18 of the 72 test months before the official run. Why the report shows them apart from the other 54, why the split is only a description fixed in advance, and why a part where two arms agree at every hour has no test.
9. **H1 on the panel.** The planted online leaks and their predicted patterns of cuts, and what the restricted recomputation leaves out, and why.
10. **The adversary.**
    - The random leaks and their honest twins, and the twins that are constant on the data.
    - The two headline rates: caught at any cut, and the mean per-cut rate.
    - The attenuation curve over ε, from 1 down to 10⁻¹⁰: how small a leak H1 still sees.
    - The families the generator never produces, listed in every report.
    - The Wilson interval. Demo: sliders on n and k, comparing Wilson with the Wald interval near 0 and 1.
    - What population the share describes, and what it does not.
    - The misses, and where they send the next designed run.
11. **The physical seal.**
    - Stage 1: renames only, a journal that pins each archive's and each CHECKSUM file's hash, sealed reads checked equal.
    - Stage 2, in three steps: `encrypt` to an age recipient (X25519) whose key never touches the machine; `verify-decrypt`, which decrypts in memory with the key on its removable medium and stamps the manifest; then `encrypt --remove-plaintext`, which refuses without that stamp.
    - The limits, plainly. The vault months are public on data.binance.vision: the seal stops our own pipeline and agents, not a person. Until stage 2 the vault tree is plaintext, guarded only in-process.
12. **Kani.**
    - `max_qty_within_contract` proven on GitHub, and the local re-run of the five proven harnesses.
    - Why a solver can lie: CBMC 6.11's `--bitwuzla` path writes an overflow flag in one order and reads it in the other, so it "proves" false overflow checks (cbmc#8723). A proof is only as sound as its toolchain.
13. **LFM3, read at Abdou's request.** Liquid AI's model: gated short convolutions with a few attention layers, a sigmoid gate on each attention head, and a Pareto search over quality and latency.
    - What it is not: there is no test-time regression layer in it.
    - What we may take: candidate ablation arms for H4, a head gate on OLS heads and a conv-first hybrid. These are ideas, read through a summarizing tool, not verified at the source.
14. **La suite:** the bolder next step, and what Abdou should master before iteration 4.

Close with:
- « Teste-toi », eight to ten questions with the answers in `<details>`;
- a glossary;
- the references cited in the repository.

## Page contract (the platform's current rules, which win over your agent file)

- **Structure.**
  - No `<!doctype>`, `<html>`, `<head>` or `<body>` tags: the platform wraps the page.
  - Start with `<title>`, then `<meta name="claude:viewport-fit-sides" content="cover">`, then `<style>`.
  - The wrapper pads `:root` top and bottom by the safe-area insets; keep that padding.
- **Gutters.**
  - `padding-left: max(16px, env(safe-area-inset-left, 0px))` and the right-hand twin, once on `body` or on one outer wrapper.
  - Vertical padding uses `padding-block`, never a `padding` shorthand that zeroes the sides.
  - A sticky header uses `top: env(safe-area-inset-top, 0px)`, with a fixed `body::before` painting the band above it.
  - The body never scrolls sideways. Tables, formulas, diagrams and code scroll inside their own `overflow-x: auto` boxes.
  - Flex and grid children that hold text get `min-width: 0`.
- **Themes.**
  - Every color is a token on bare `:root`, with the light values.
  - Redefine the tokens under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ...; color-scheme: dark } }`, and again under `:root[data-theme="dark"] { ...; color-scheme: dark }`.
  - `body` gets an explicit token background. No component uses a literal color.
- **External files.**
  - Scripts only from cdnjs.cloudflare.com (preferred), cdn.jsdelivr.net/npm or unpkg.com. Pin each to an exact version at least two weeks old: MathJax 3 SVG from cdnjs, as before.
  - Stylesheets only from Google Fonts, with real fallback stacks.
  - Everything else is inline. No fetch to other hosts, no iframes, no `alert`, `confirm` or `prompt`, no download links, no print button.
- **Mermaid.** The viewer renders `<pre class="mermaid">` natively, so load no Mermaid library. Draw what Mermaid cannot as inline SVG, with explicit fills and theme-token colors.
- **Readability.**
  - Body text at least 17 px on phones, and about 65 characters per line.
  - `text-wrap: balance` on headings, and `tabular-nums` in tables.
  - A visible focus state, and `prefers-reduced-motion` respected.
- **Rest state.** Everything is visible at rest, with no `opacity: 0` waiting for an observer. Every demo opens filled and working.
- **Storage.** `localStorage` only for conveniences, inside try/catch.
- **Look.**
  - No templated look. Avoid cream with a terracotta serif, near-black with a lone acid accent, broadsheet hairlines with dense columns, purple gradients, Inter or Space Grotesk, emoji as section markers, and `rounded-lg` on everything.
  - Numbered markers only where the order is information.
- **Size.** Well under 16 MB.

## Rules

- **French throughout,** with the narrow no-break space (U+202F) before `:` `;` `?` `!` and inside « », as the previous pages do.
- **Maths.** MathJax `\(...\)` with every backslash intact. Write with your editor tools, never through a shell heredoc: one ate the backslashes once.
- **Status of claims.** Mark planned work as planned, and hypotheses as hypotheses. Results come only from `docs/RESULTS.md` and the figures below.
- **Checks:**
  - `node --check` on each inline script, extracted to a temp file;
  - every tag closed;
  - every external URL answers 200 (`curl -sI`);
  - a read at 400 px wide.
- Do not publish, and do not commit.

## Verified figures

The orchestrator appends them below once they are verified.
