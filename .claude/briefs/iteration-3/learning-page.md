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
    - Why a solver can lie: CBMC 6.11's `--bitwuzla` path writes an overflow flag in one order and reads it in the other, so it "proves" false overflow checks. Upstream knew the cause: diffblue/cbmc#9144, fixed on `develop` by #9145 on 2026-10-12 but not released. With Abdou's OK, the iteration 2 session commented on diffblue/cbmc#8723 and opened model-checking/kani#5096 on 2026-10-29. A proof is only as sound as its toolchain.
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

### Verified by the orchestrator, 2026-10-29

Each figure below was re-read from the run's files or the commit named beside it.

**Commits and CI**
- The official runs ran at `3a22daa` (Report n°1's check, the panel H1, the adversary) and at `31c8005` (Report n°2). GitHub CI was green on Linux and Windows for each commit before its runs. Pull request abdou-rgd/ols-prime#3.
- `31c8005` changes one thing. Report n°2's internal check that its ECE equals `stats.ece` now allows the rounding of the two sums: 4·n·ε instead of a fixed 10⁻⁹ relative.
  - The first official Report n°2 stopped on that check after 17 minutes. Its gap was 1.3 × 10⁻⁹ relative, on climatology's ECE, because climatology repeats one forecast over many rows and the rounding errors add up.
  - A lost row moves the ECE about 300 times more than the new tolerance.
  - The PROTOCOL change log records it, with the one number the stopped run showed: climatology's ECE over the test period, 0.0029.
- Kani: the `proven` job, five harnesses, passed on GitHub in 91 minutes for `3a22daa` (run 40842249385) and in 74 minutes for `31c8005` (run 40854754950).

**The panel** (`docs/DATA.md`)
- 1,655,881 rows in 2,273 member months of 233 pairs. The other 41 of the 274 members first join in 2025, after the development period.
- 1,631,784 rows are labelled, and 1,591,462 have all 15 features. Classes: down 29.4 %, neutral 43.6 %, up 27.1 %.

**Matrix core** (measured under load, portable SSE2 build)
- 2 M pushes in 0.64 s.
- Ridge: 0.19 s per 10⁵ queries × 3 windows × 6 penalties. The whole panel takes a few seconds.
- One-year kernels: 7.55 s per 10⁴ queries over 262,800 candidates, about 25 minutes for the panel. An x86-64-v3 build gives the same bits in about 14 minutes.

**CUSUM**
- h = 4.761 for an ARL₀ of 365 days: simulated ARL 365.03, SE 0.36, 10⁶ runs.

**H1 on the panel**, `runs/2026-10-29-h1-panel/`, 34 min 48 s
- The verdict is that H1 holds: 0 honest violations at the 8 cuts, with 2.7 to 14.7 million values compared per cut, and every planted leak is on its predicted pattern.

| planted leak | arms | pattern | smallest catch |
|---|---|---|---|
| entries stamped at decision | matrix-select | `XXXXXXXX` | 0.069 |
| window ends one hour late | matrix-select | `XXXXXXXX` | 1.7 × 10⁻⁵ |
| σ on the whole test month | matrix-select | `XX.XXXXX`, as predicted: cut 3 falls at a month's last hour | 0.0039 |
| selection without purge | matrix-select | `X.......` | 0.05 |
| kernel reaches one horizon late | softmax-1y, knn-1y | `XXXXXXXX` | 0.053 |

**The adversary**, `runs/2026-10-29-h1-adversary/`, 3 min 15 s with 8 workers
- 9,795 of 10,000 random leaks are caught at one of H1's 16 cuts at least: 98.0 %, Wilson [97.7, 98.2] %.
- The mean per-cut rate is 86.9 % [86.4, 87.3] %. Over the long schedule of 128 cuts it is 89.0 % [88.6, 89.4] %.
- The 1,247 leaks that the protocol's by-construction list covers are all caught.
- Honest twins: 0 caught and 0 bit violations out of 10,000. Of them, 1,166 (11.7 %) are constant at the compared rows and cannot be caught.
- Of the 205 leaks caught at no cut, 163 are constant at the compared rows. The audit's phase 2 classified the other 42, and the orchestrator re-ran its probe on `adversary.json`:
  - 11 are thresholds outside the bands where H1's perturbation flips by construction, such as `fill[s+1] > 0.9731` or `body[s+1] < 0.4348`. The long schedule catches each at 1 to 12 of its 128 cuts, when a real extreme bar comes along.
  - 18 are aggregates of future bars compared with a past bar or window, so lopsided that H1's perturbation does not flip them, such as `sum(range[s+1..s+10]) < 0.359 × sum(range[s-25..s-16])`. The long schedule catches 16 of them, at 1 to 19 cuts.
  - 13 are combinations. Two of them depend on the past alone: `(colour[s+1] == colour[s-52]) XOR green[s+1]` is just "bar s−52 is red". So they are no leak, and the headline counts them as misses, which is conservative.
  - One is an XOR band on the taker shares, which fires only when the share lands between 0.58 and 0.76, where H1 pushes it to extremes. The other 10 are AND or OR with one operand that is nearly constant.
  - None shows a hole beyond the limits the protocol already states.
  - The audit's bolder step:
    - a future-shuffle twin for each leak, to tell "no real dependence on the future" from a true blind spot;
    - an extreme-shape cut for H1, with fill and close location at 0.99 and 0.01.
- Attenuation, the share caught at any cut: 98.0 % from ε = 1 down to 10⁻⁶, 97.8 % at 10⁻⁸ and 2.4 % at 10⁻¹⁰.
- A second run with 4 workers gives byte-identical `adversary.json` and `adversary.md`.
- This is the detection rate of this generator, not of all leaks.

**Report n°1 after the vault move** (`runs/2026-10-29-report1-check.log`)
- Every number reproduces.
- Only provenance lines differ: the commit, the code digest, the data manifest digest, and the hash and stored rows of the tables rebuilt by the move. Their sealed rows are unchanged.
- The data manifest digest moved twice, with the 843 file entries unchanged after the move. It was `230e28e8` in iteration 2 and `7c5f2397` just after the move. It is `0ee7ac0f` in the runs, because commit `deae747` renamed a field of the manifest's summary of the vault tree.

**Report n°2**, `runs/2026-10-29-report2/`, commit `31c8005`, 20 min 15 s; its `--check` reproduces its 3 files byte for byte
- Scored: 1,493,353 rows over 50,626 decision hours, 96.2 % of the 52,608 test hours, 2019-01 → 2024-12. The unscored hours sit around exchange outages: after a gap in the bars, a row waits about 90 bars for its features. For example, an outage on 2020-06-28 leaves 94 hours unscored, from 2020-06-28T01 to 2020-07-01T22.
  - Before a gap of g bars, 1 + g rows of each member lose their label, because the label reads only its entry bar (t+1) and its exit bar (t+25). report-2.md's Coverage paragraph says "the 25 bars before it", which is wrong: `docs/RESULTS.md` records the correction. Do not repeat that sentence.
  - The audit also found that the ECE cross-check cannot see a row put in another bin, since both paths bin the same way. It guards a row lost or counted twice (PROTOCOL change log, after audit phase 2).
- 0 fallbacks to climatology and 0 defaults in every arm: every scored output read a cell with status OK.
- Mean hourly log-loss [95 % stationary bootstrap], then the Brier skill score against climatology:

| arm | log-loss | BSS | ECE |
|---|---|---|---|
| climatology-panel | 1.0756 [1.0726, 1.0786] | — | 0.0029 |
| matrix-select | 1.0650 [1.0611, 1.0690] | 0.0130 [0.0108, 0.0156] | 0.0108 |
| matrix-1y | 1.0651 [1.0611, 1.0697] | 0.0134 [0.0112, 0.0156] | 0.0093 |
| softmax-1y | 1.0740 [1.0710, 1.0771] | 0.0032 [0.0018, 0.0046] | 0.0138 |
| knn-1y | 1.0661 [1.0620, 1.0701] | 0.0109 [0.0086, 0.0132] | 0.0083 |
| matrix-adaptive | 1.0663 [1.0626, 1.0702] | 0.0115 [0.0094, 0.0136] | 0.0096 |
| matrix-stack | 1.0644 [1.0603, 1.0686] | 0.0131 [0.0108, 0.0153] | 0.0122 |
| matrix-cusum | 1.0650 [1.0609, 1.0690] | 0.0130 [0.0107, 0.0153] | 0.0108 |

- The verdicts, by the registered rules. A comparison passes when Diebold-Mariano gives p < 0.05, the mean difference is below 0 and its bootstrap interval lies below 0.
  - **H2 holds.** The BSS of matrix-select is 0.0130 [0.0108, 0.0156]: small, and its interval is above 0.
  - **H3 does not hold.** matrix-1y beats softmax-1y: −0.00891, which is −0.83 % of climatology's log-loss, DM p < 0.0001, bootstrap [−0.01101, −0.00691]. It does not beat knn-1y: −0.00101 (−0.09 %), p 0.29, bootstrap [−0.00278, +0.00081]. H3 needs both.
  - **H10 does not hold.** matrix-adaptive is worse than matrix-select: +0.00129 (+0.12 %), DM p 0.006, bootstrap [+0.00049, +0.00228]. That the other direction is significant is a description, not a registered test. Its ρ is 0.0001, the smallest of the grid, at 18,965 of its 50,631 decisions.
  - **H11 does not hold, narrowly.** matrix-stack minus matrix-select: −0.00060 (−0.06 %), p 0.0725, bootstrap [−0.00122, +0.00001].
  - **H12 holds.** matrix-cusum against matrix-select: −0.00002. Its one-sided 95 % upper bound is +0.00053 and its bootstrap 95th percentile +0.00052, both below the margin of 0.00538, which is 0.5 % of climatology's log-loss. It made 13 fits, against a limit of 36, where the monthly schedule makes 72. The CUSUM raised 12 alarms over 2,108 daily values.
  - Identical hours, where both arms give the same hourly log-loss, are at most 31.9 % (H10), so every comparison rests on many informative hours.
- **The previewed months, apart.** This is a description, not a verdict. Smoke runs scored 18 of the 72 test months, 2019-01 → 2020-06, before the official run; no rule changed after them.
  - BSS of matrix-select: 0.0201 [0.0143, 0.0265] on those 18 months (12,130 hours), and 0.0108 [0.0083, 0.0132] on the other 54 months (38,496 hours). The skill is smaller on the months nobody had seen, and still above 0.
  - In each part, every comparison has the same outcome as on the whole period. Against softmax, the rule is met in both parts. Against kNN, and for H10 and H11, it is met in neither. H12's own rule is met in both.
- **The readout, described** on 1,000 seeded decisions of matrix-1y: 30.6 % of the weights are negative. The entries of the last 30 days give about a tenth of each forecast. The two contributions add up to the forecast within 5.5 × 10⁻¹⁵.
- Variants: 10 before this iteration, plus the 8 arms: 18.
