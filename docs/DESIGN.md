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

## 3. Matrix

Matrix stores every entry `(zᵢ, yᵢ, tᵢ)` in time order.
- **Blocks.** Entries are grouped into blocks of `B = 2^b`. A finished block becomes a level-0 node holding the sufficient statistics

  ```
  S = (n, Σz, Σzzᵀ, Σyzᵀ, Σy, Σy²).
  ```
- **Roll-up.** When two sibling nodes are finished, they merge into their parent, like the carries of a binary counter. The statistics are sums, so merging is addition.
- **Ranges.** A range of entries `[l, r)` is covered by at most two partial blocks, scanned directly, plus O(log n) nodes.

For any period R, this gives:
- its ridge forecast for the current state, `ŷ = C_R (G_R + Λ)⁻¹ z_t`;
- the exact signed contribution of any sub-period, down to a single day;
- a one-line summary of the period: its own regression `β_R`, its mean outcome and its residual variance.

This is OptMem's idea — an append-only log whose older entries are rolled up into a binary tree of summaries — with summaries that are exact instead of written by a model.

**Causality.** A decision taken at t over horizon h has an outcome known only at t + h. The entry joins Matrix then, never before.

## 4. Encoder (phase 3)

Tokens embed (asset × time patch) pairs from a window that lies entirely in the past. Attention inside the window may therefore be bidirectional. Each head computes signed T × T weights:

```
W = Q (KᵀK + λI)⁻¹ Kᵀ,     O = W V.
```

A constant key column makes every row of W sum to one.

Design constraints:
- `T ≫ d_head`; otherwise `K (KᵀK)⁻¹ Kᵀ ≈ I` and the head copies its input;
- λ > 0, learned per head;
- Cholesky solves in fp32, with normalized keys.

During training, Matrix is causal inside episodes of consecutive days, through prefix sums of the statistics as in the mesa-layer. Older history acts as a cache without gradient.

## 5. Decision head and decision rule

- **Options.** The head outputs probabilities over typed options: down, neutral or up over horizon h. The thresholds are set by fees and volatility.
- **Inputs.** It reads the encoder state and Matrix readouts at several resolutions: 30 days, one year, all history.
- **Loss.** Label-smoothed cross-entropy plus a Brier term.
- **Rule.** Go long only when the expected edge clears the venue's round-trip cost, with hysteresis. The risk kernel has the last word.

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
