# Design

This document fixes the mathematics the code implements.

Notation:
- a state `z ∈ Rᵖ`: a feature vector, later the encoder's output;
- an outcome `y ∈ Rᵠ`: a return, or a one-hot class;
- a ridge penalty `λ ≥ 0`.

## 1. OLS as attention

Take training rows `X` (n × p) with outcomes `Y` (n × q). Ridge regression gives `β = Yᵀ X (XᵀX + Λ)⁻¹`, with `Λ = λI`. A query `x_q` then gets the forecast

```
ŷ = β x_q = Σᵢ wᵢ yᵢ,     wᵢ = xᵢᵀ a,     a = (XᵀX + Λ)⁻¹ x_q.
```

The forecast is a weighted sum of past outcomes. In attention terms:
- the query is `x_q`, the keys are the `xᵢ` and the values are the `yᵢ`;
- the weights are computed in the space whitened by `(XᵀX + Λ)⁻¹` rather than through a softmax;
- unlike softmax weights, they can be negative: a past day can count *against* its own outcome.

Everything about a query follows from the single vector `a`:
- **forecast:** `ŷ = C a`, where `C = Σᵢ yᵢ xᵢᵀ`;
- **weight of entry i:** `wᵢ = xᵢᵀ a`;
- **contribution of a set S of entries:** `C_S a = Σ_{i∈S} wᵢ yᵢ`. The contributions of a partition add up exactly to `ŷ`.

**Intercept.** Suppose coordinate 0 of every `x` is the constant 1 and is left unpenalized. Then every row of weights sums to one: `Σᵢ wᵢ = 1`.
- Proof: `Λe₀ = 0`, so `Xᵀ1 = (XᵀX + Λ) e₀`. Hence `Σᵢ wᵢ = 1ᵀX a = e₀ᵀ x_q = 1`.

**Numerical singularity.** `ols-core` refuses a solve with `NotPositiveDefinite` when `XᵀX + Λ`, scaled to a unit diagonal, has its smallest eigenvalue at or below `τ = 4p(N + p + 2)ε`. Here N is the number of entries and ε = 2⁻⁵². The derivation is in the `RidgeConfig` docs.
- A singular design never returns numbers.
- Features must be centred and scaled inside the fold. A feature beside the intercept with `r = |mean|/std` gives a smallest eigenvalue of about `1/(2r²)`.
- At p = 33 and λ = 0, such a feature is refused from r ≈ 1.3·10⁴ when N = 10⁵, and from r ≈ 4·10³ when N = 10⁶. Standardized features pass.

**Leakage.** The paper warns that masking future entries is not enough: `(XᵀX + Λ)⁻¹` itself must be estimated from the past only.
- Every statistic, scaler and PCA in OLS Prime is fitted inside the walk-forward fold.
- Matrix only ever holds entries whose outcome is already known.

## 2. Recursive least squares with forgetting

RLS keeps two quantities:
- the coefficients `β`;
- `P = (G + γⁿλI)⁻¹`, where `G = Σᵢ γ^(n−1−i) xᵢxᵢᵀ`.

Each observation updates them in O(p²):

```
k = P x / (γ + xᵀ P x)
β ← β + (y − β x) kᵀ
P ← (P − k (P x)ᵀ) / γ
```

After n updates, RLS equals weighted ridge with weights `γ^(n−1−i)` and penalty `γⁿλ`; the intercept is penalized too. It is the online, primal twin of Matrix: no stored history, exponential memory.

**Limits.** Forgetting RLS has two classic failure modes. `crates/ols-core/src/rls.rs` documents both, and no model uses RLS yet.
- **Wind-up.** The penalty `γⁿλ` vanishes as n grows. Along a direction the inputs stop exploring, the eigenvalue of `G` decays like `γᵏ`, so `P` grows like `γ⁻ᵏ`. Before a model relies on RLS, it needs a penalty floor.
  - Regularized forgetting, `P⁻¹ ← γP⁻¹ + xxᵀ + (1 − γ)λI`, keeps the penalty at exactly λ. That costs O(p³) per step, or O(p²) if the floor is added one coordinate per step.
  - Matrix avoids the problem: a window of recent entries, read from the tree, is an exact ridge with a constant λ.
- **Symmetry and definiteness.** Rounding in the rank-one downdate breaks the symmetry of `P`. The code restores it after every update, which does not guarantee that `P` stays positive definite. A square-root update (Cholesky or QR) would, and is the fallback if long runs drift.

## 3. Matrix

Matrix stores every entry `(zᵢ, yᵢ, tᵢ)` in time order.
- **Blocks.** Entries are grouped into blocks of `B = 2^b`. A finished block becomes a level-0 node holding the sufficient statistics

  ```
  S = (n, Σz, Σzzᵀ, Σyzᵀ, Σy, Σy²).
  ```
- **Roll-up.** When two sibling nodes are finished, they merge into their parent, like the carries of a binary counter. The statistics are sums, so merging is addition. The raw entries stay in the log: the tree is built over them, and the per-entry weights need them.
- **Ranges.** A range of entries `[l, r)` is covered by at most two partial blocks, scanned directly, plus O(log n) nodes.

For any period R, this gives:
- its ridge forecast for the current state, `ŷ = C_R (G_R + Λ)⁻¹ z_t`;
- the exact signed contribution of any sub-period, down to a single day;
- a one-line summary of the period: its own regression `β_R`, its mean outcome and its residual variance.

This is OptMem's idea — an append-only log with a binary tree of summaries built over it — with summaries that are exact instead of written by a model. Unlike OptMem, nothing is compacted away.

**Causality.** A decision taken at t over horizon h has an outcome known only at t + h. The entry joins Matrix then, never before.

**Numerics.** The nodes hold raw sums, which has two consequences.
- **Cancellation.** The residual sum of squares, `Σy² − 2βᵀΣyz + βᵀGβ`, cancels catastrophically when the fit is close (RSS ≪ Σy²). `residual_sum_squares` clamps a negative result to 0, which hides the error instead of flagging it. With daily returns, R² is near 0 and the cancellation is mild; in general it is not.
- **Conditioning.** An uncentred Gram matrix beside an intercept is ill-conditioned when a feature's mean is large against its spread (§1).

Planned for iteration 3: nodes in centred form, `(n, z̄, M_zz, ȳ, M_yz, M_yy)`.
- Two nodes merge with the pairwise update of Chan, Golub and LeVeque (1979): `M(A ∪ B) = M(A) + M(B) + (n_A·n_B / n)·δ δᵀ`, where δ is the difference of the two means.
- It stays exact and composable, at O(p²) per merge, and the intercept is then solved in centred form.

**Overlapping outcomes.** With a horizon h > 1, consecutive entries share h − 1 periods of their outcome, so a range of N entries carries only about N/h independent outcomes.
- Standard errors and tests must account for it: HAC with h − 1 lags, or the block bootstrap.
- A readout's ridge penalty should scale with h, which is the same as weighting each entry by 1/h.

## 4. Encoder (phase 3)

Tokens embed (asset × time patch) pairs from a window that lies entirely in the past. Attention inside the window may therefore be bidirectional. Each head computes signed T × T weights:

```
W = Q (KᵀK + λI)⁻¹ Kᵀ,     O = W V.
```

The rows of W sum to one under the two conditions of §1:
- the constant key coordinate is unpenalized, `Λ = diag(0, λ, …, λ)` rather than `λI`;
- every query has 1 in that coordinate.

In general they do not otherwise. Take a constant key column under `λI`, and a query q with 1 in that coordinate. Since `Kᵀ1 = KᵀK e₀`, its row sums to `1 − λ·e₀ᵀ(KᵀK + λI)⁻¹ q`, where e₀ picks the constant coordinate. That is 1 only at λ = 0, or in special cases such as a dual direction `(KᵀK + λI)⁻¹ q` with no constant component.

Design constraints:
- `T ≫ d_head`; otherwise `K (KᵀK)⁻¹ Kᵀ ≈ I` and the head copies its input;
- λ > 0, learned per head, on every coordinate except the constant one;
- Cholesky solves in fp32, with normalized keys.

During training, Matrix is causal inside episodes of consecutive days, through prefix sums of the statistics as in the mesa-layer. Older history acts as a cache without gradient.
- The prefix sums accumulate in fp64 and are cast to fp32 only for the factorization. A range read as the difference of two large fp32 prefix sums can lose positive definiteness.
- The tree's merges avoid that subtraction altogether.

## 5. Decision head and decision rule

- **Options.** The head outputs probabilities over typed options: down, neutral or up over horizon h. The thresholds are set by fees and volatility.
- **Inputs.** It reads the encoder state and Matrix readouts at several resolutions: 30 days, one year, all history.
- **Loss.** Cross-entropy plus a Brier term.
  - Label smoothing pulls every probability towards uniform. That biases the expected edge the rule compares with costs, so label smoothing is an ablation, off by default.
  - The probabilities the rule reads are recalibrated inside the fold, by temperature scaling on the fold's last segment. Their calibration error is reported.
- **Rule.** Go long only when the expected edge clears the venue's round-trip cost, with hysteresis. The risk kernel has the last word.
  - Iteration 2's baselines enter when the expected return exceeds the round-trip cost (taker fee plus 2 bps of slippage, each way) and exit when it falls below 0.

## 6. Risk kernel

All amounts are integers:
- quantities in 1e-8 base units;
- prices and quote amounts in 1e-6 quote units;
- intermediate products in i128.

Invariants, checked by property tests and Kani harnesses:
- an approved order is a positive multiple of the lot step, within the per-order cap, and at or above the exchange minimum;
- after an approved buy, the position's notional stays within the position cap;
- a sell never exceeds the position: no shorting;
- no buy once the drawdown kill switch or the daily loss limit has tripped, nor during the cooldown after an order;
- sells stay allowed, so the kill switch never traps a position.

## 7. Adaptation loops (planned)

Matrix already adapts without gradients: every realized outcome joins the memory, and the next readout uses it. Three loops would let the system tune itself from its own statistics.
- Each one is a fitted choice. It runs causally, inside the fold, so the future-invariance test (H1) applies to it as to any model.
- Each one counts as a variant for the deflated Sharpe.
- H10–H12 register them.

**λ from the statistics alone.** With the penalty `Λ = λI`, ridge has a closed form for every λ once `G = VDVᵀ` is diagonalized, at O(p³) once and O(p) per λ (for q = 1, with `c̃ = Vᵀc`). With the free intercept of §1, `Λ = diag(0, λ, …)`, `G + Λ` is no longer a shift of G's spectrum, so the formulas apply only to statistics centred first. These are the centred nodes planned in §3, with the intercept solved apart.
- `tr H(λ) = Σⱼ dⱼ/(dⱼ + λ)`;
- `RSS(λ) = Σy² − Σⱼ c̃ⱼ²·(dⱼ + 2λ)/(dⱼ + λ)²`.

So generalized cross-validation, `GCV(λ) = N·RSS(λ)/(N − tr H(λ))²`, needs only a node's statistics.
- GCV and leave-one-out assume exchangeable rows. With serial correlation, and with overlapping outcomes when h > 1, neighbours leak into each other's held-out predictions, and the λ chosen is too small.
- The honest criterion is the prequential error: predict each entry from the entries known before it. RLS with γ = 1 produces exactly these one-step ridge errors at a fixed λ (its recursive residuals), so a small grid of λ costs one RLS per value.

**Weights over memory.** Re-weighting nodes is exact: `G = Σ_b α_b G_b` and `C = Σ_b α_b C_b` give the ridge on the re-weighted entries. Exponential forgetting is the special case `α_b = γ^age`. The forecast's sensitivity to a block's weight has a closed form, from the block's statistics alone:

```
∂ŷ/∂α_b = (C_b − β G_b) a = Σ_{i∈b} (yᵢ − β zᵢ) zᵢᵀ a.
```

- This is the block's residual, seen along the query direction `a`. It is the exact gradient behind "mute the blocks whose contributions hurt".
- Fitting one weight per block invites overfitting. The low-dimensional version stacks the 30-day, one-year and all-history readouts with three weights, fitted on prequential errors (stacked regression, Breiman 1996).

**When to refit the encoder.** Comparing the residual variance of the last 30 days with the last year mixes up two things: crypto volatility clusters, so residual variance rises in every turbulent month even when `z` loses nothing. Two better tests:
- a test on skill, not variance: the prequential log-loss against climatology;
- a CUSUM on the standardized recursive residuals (Brown, Durbin and Evans 1975), with its false-alarm rate fixed in advance.

Two timescales follow: Matrix and the readout weights at every bar, gradient-free; the encoder and the head at monthly refits, as the protocol already does, plus early refits on alarm.

## Change log

- 2026-10-25: initial version.
- 2026-10-27: after an external review of this document (Gemini 4 Argon, passed on by Abdou), checked against the code:
  - §4 was wrong: a constant key column under `λI` does not make the rows of W sum to one. The conditions of §1 are now stated.
  - §2 records RLS wind-up and the loss of symmetry. The code already documented wind-up and restored symmetry after each update.
  - §3 records the cancellation in the residual sum of squares, which the code clamped silently, and the overlap of outcomes when h > 1. Centred nodes are planned for iteration 3.
  - §4 accumulates prefix sums in fp64.
  - §5 makes label smoothing an ablation and recalibrates the rule's probabilities inside the fold.
  - §7 adds the adaptation loops, after a second review, on self-improvement. They are registered as H10–H12.
- 2026-10-28: after Audit B of iteration 2.
  - §3 says that the raw entries stay in the log. Unlike OptMem's, Matrix's tree compacts nothing away.
  - §4 states the row sum under `λI` exactly: `1 − λ·e₀ᵀ(KᵀK + λI)⁻¹q`.
  - §7 limits the closed form for GCV to `Λ = λI`, or to centred statistics.
