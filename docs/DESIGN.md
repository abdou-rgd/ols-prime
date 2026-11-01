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

**Numerics.** The nodes of the raw tree hold raw sums, which has two consequences. The centred tree of §3.1 removes the second. It only reduces the first: its residual sum of squares subtracts from the centred `M_yy`, not from `Σy²`, so it still cancels when the fit is close.
- **Cancellation.** The residual sum of squares, `Σy² − 2βᵀΣyz + βᵀGβ`, cancels catastrophically when the fit is close (RSS ≪ Σy²). `residual_sum_squares` clamps a negative result to 0, which hides the error instead of flagging it. With daily returns, R² is near 0 and the cancellation is mild; in general it is not.
- **Conditioning.** An uncentred Gram matrix beside an intercept is ill-conditioned when a feature's mean is large against its spread (§1).

**Overlapping outcomes.** With a horizon h > 1, consecutive entries share h − 1 periods of their outcome, so a range of N entries carries only about N/h independent outcomes.
- Standard errors and tests must account for it: HAC with h − 1 lags, or the block bootstrap.
- A readout's ridge penalty should scale with h, which is the same as weighting each entry by 1/h.

### 3.1 Centred nodes (iteration 3)

A second tree holds every node in centred form, `(n, z̄, M_zz, ȳ, M_yz, M_yy)`, where `M_zz = Σ(z − z̄)(z − z̄)ᵀ`.
- **Merge.** Two nodes merge with the pairwise update of Chan, Golub and LeVeque (1979): `M(A ∪ B) = M(A) + M(B) + (n_A·n_B / n)·δ δᵀ`, where δ is the difference of the two means. The cost is O(p²) per merge.
- **Leaves and partial blocks.** They are computed by the corrected two-pass scan: first the mean, then the centred sums minus `(Σd)(Σd)ᵀ/n`. That correction removes the first-order error of the rounded mean. A diagonal that rounds below zero, as a constant column can, is set to zero.
- **Symmetry.** Only one triangle of `M_zz` and `M_yy` is computed. It is then mirrored, so both blocks are exactly symmetric.

**Beside the raw tree, not in place of it.** The raw sums cannot be recovered exactly from centred nodes. Replacing the raw tree would change the bits of every raw read, and of every test and golden file built on them. So the centred tree sits beside it. Every raw read keeps its bits.
- **Memory.** Each tree costs about 160 MB at 2 million entries with p = 15, q = 4 and blocks of 2⁶ entries. Blocks of 2⁸ cost a quarter of that.

**Purity.** A centred read of `[l, r)` depends on `(l, r)` alone. It always takes the same path: the left partial block, the left nodes in order, the right nodes in reverse, then the right partial block. An entry pushed later never changes it, not even in the last bit. Property tests check this.

### 3.2 Centred ridge with a free intercept

For a window of `n_w` entries, a penalty `ρ ≥ 0` and a per-coordinate scale `s` (the fold's feature variances):

```
A = M_zz + Λ,   Λ = ρ·n_w·diag(s),   β = M_yz A⁻¹,   ŷ = ȳ + β (z_q − z̄).
```

- This is ridge on features standardized inside the fold, with an unpenalized intercept `b = ȳ − β z̄`.
- On request, the read also gives the leverage `z̃ᵀ A⁻¹ z̃`, with `z̃ = z_q − z̄`, and the window's residual sum of squares per outcome: `M_yy − β M_yzᵀ − β Λ βᵀ`, clamped at 0.

**Refusal.** The solve refuses a window whose `A` lies within rounding of a singular matrix. Let `D = diag(A)` and `S = D^(−1/2) A D^(−1/2)`. The solve accepts only if

```
λ_min(S) > τ,   τ = 4p(h + p + 2)ε · (1 + 2(L + 1)·r),
```

where:
- `ε = 2⁻⁵²`;
- `L` is the depth of the merge tree behind the statistics;
- `h = longest scan + 3 + 3L`;
- `r = maxⱼ √((M_jj + n z̄ⱼ²) / D_jj)` compares each coordinate's raw size with its scaled diagonal.

The derivation is first order in ε:
- **Scans.** A scan of b entries errs by at most about `(b + 3)ε √(M_jj M_kk)`.
- **Merges.** Each merge adds two roundings. The nodes at one depth are disjoint, so their errors add up to at most `3ε √(M_jj M_kk)` per depth, by Cauchy–Schwarz.
- **Rounded means.** The means of the merged nodes err relative to their values, not their spread. Through `δ δᵀ` this costs `2Lhε(rⱼ + rₖ)` after scaling. This is the only term where a large mean enters, and it enters linearly in r. The raw Gram matrix pays r².
- **Spectrum.** The error of `S` is bounded in the Frobenius norm. A factor of 4 and one extra depth cover the scaling and the eigenvalue solver. By Weyl's inequality, a design that is exactly singular then has a computed `λ_min(S) ≤ τ`, and it is refused.

This is a bound with conservative constants, not a proof. Property tests check it against an independent singular value decomposition, on designs that are exactly singular (duplicated rows, affine and constant columns, n ≤ p), including designs shifted by 10⁶.

### 3.3 Batch readouts

Two methods of `olsprime_rs.Matrix` serve the readouts of iteration 3. Each takes M queries `z_q` with their decision times `as_of` in milliseconds.
- **`ridge_batch`** reads every query under W windows and R penalties. A window holds the entries with `as_of − w < time ≤ as_of`; "all history" holds every entry with `time ≤ as_of`. It returns the forecasts (M × W × R × q), the window counts (M × W), a status per cell (M × W × R), and on request the leverage and the residual sums of squares.
- **`kernel_batch`** gives the exact kernel readouts of H3 over one window, with distances on `u = z / σ`:
  - softmax (Nadaraya-Watson): `Σ wᵢ yᵢ / Σ wᵢ`, `wᵢ = exp(−(dᵢ − d_min) / (2τ²))`, one per width τ;
  - k nearest neighbours: the mean y of the k nearest entries, ties going to the lowest entry index, one per k.

  Every entry of the window gets its exact distance. Nothing is sampled or approximated.

**Statuses.** A cell that is not ok holds NaN in every output. Bad inputs raise an error instead.

| Code | Name | Meaning |
|---|---|---|
| 0 | `STATUS_OK` | a number |
| 1 | `STATUS_EMPTY` | the window holds no entry |
| 2 | `STATUS_SINGULAR` | the solve refused the window (§3.2) |
| 3 | `STATUS_NON_FINITE` | an intermediate or the result overflowed |
| 4 | `STATUS_TOO_FEW` | k is larger than the window |

**Bit identity.** Each cell is computed by one thread, with fixed sequential code. Threads only decide which cells they take, and nothing is summed across threads. Tested guarantees:
- every `ridge_batch` cell has the bits of the one-by-one read `centred_query`;
- both batches give the same bits whatever the thread count, the order of the queries, and the entries pushed later;
- the kernel sums run in an order fixed by the entry indices, and the exponential is computed in the crate rather than by the platform's math library. So the portable build and an AVX2 build give the same bits. A hash of the outputs matched at 10⁴ queries.

The bindings follow the crate's rules:
- dtypes are checked before any value is read;
- every output and every internal buffer is sized against `MAX_ALLOC_BYTES` before anything is allocated;
- an `as_of` in microseconds is refused;
- Python's global interpreter lock is released during the computation.

**Speed.** The benchmark is `cargo run --release -p ols-core --example matrix_bench`. It ran on a Ryzen 5 3600 (6 cores, 12 threads), with 12 worker threads, 2 million entries, p = 15 and q = 4, in the portable build, which is the build we keep.

| Step | Time |
|---|---|
| push 2 million entries | 0.64 s |
| `ridge_batch`, per 10⁵ queries × 3 windows × 6 penalties, leverage and residuals included | 0.19 s |
| `kernel_batch`, per 10⁴ queries at 262,800 candidates, 3 widths and 3 values of k | 7.55 s |
| all readouts for 2 million queries, extrapolated | about 25 minutes |

Two stray processes kept two of the 12 hardware threads busy during these runs. The times are therefore pessimistic, and they vary from run to run.

**Not verified.**
- **Shift tolerance.** The shift-invariance tests use a loose tolerance: 10⁴·10⁶·ε in Rust, 10⁻⁷ in Python. The actual worst case was not measured.
- **The lock test.** The test that the global interpreter lock is released was not checked against a build that holds the lock.
- **Footprints.** The memory bounds that the bindings check are derived from the code. Nothing measures the actual allocations.

## 4. Encoder (iteration 4)

Roadmap phase 3, as built. A row is one pair at one hourly bar.

**Tokens.** One token per hourly bar of that pair. The window is the 64 bars ending at the decision bar, `t − 63` through `t`. All 64 are already final, so attention inside the window is bidirectional: there is no mask. A token that embeds an asset together with a span of time is not built.

Each bar carries eight quantities: `ln(c / c_prev)`, `ln(o / c_prev)`, `ln(h / l)`, `ln(c / o)`, `ln(h / max(o, c))`, `ln(min(o, c) / l)`, `ln(1 + quote volume)`, and the taker share of quote volume minus 1/2 (0 when the bar has no volume). The first two use the previous close, so the series starts at `t − 64`. The bars from `t − 64` through `t` must be consecutive hours, and every quantity of the 64 window bars must be finite. Otherwise the row takes the arm's climatology.

Each quantity is standardized with the mean and standard deviation (divisor n) of the last bar of every training row of the fit. Those sums are sequential, in fp64. The values are then clipped to ±8 and kept in fp32.

**Trunk.** A linear map sends the eight quantities to `d_model = 32`, plus a learned embedding of the 64 positions. Two pre-LayerNorm blocks follow. Each is attention with 4 heads of `d_head = 8`, then an MLP `32 → 64 → 32` with GELU, a residual around the attention and around the MLP, and no dropout. A final LayerNorm. The state `h ∈ R³²` is the output at the last position, bar t.

The three arms use this same trunk. Only the attention operator differs.
- `enc-ols` uses the OLS head below.
- `enc-softmax` uses `softmax(Q Kᵀ / √8) V`, scaled dot-product attention on the math backend.
- `enc-linear` uses `φ(Q) (φ(K)ᵀ V) / (φ(Q) φ(K)ᵀ 1)`, with `φ(x) = elu(x) + 1`.

**The OLS head.** Queries and keys are scaled to unit root mean square at each position, with no parameter. A constant coordinate is concatenated and left unpenalized:

```
K̃ = [1 | K],    q̃ = [1 | q],
Λ = diag(0, Tλ, …, Tλ),    λ = softplus(θ) + 10⁻³,
W = Q̃ (K̃ᵀK̃ + Λ)⁻¹ K̃ᵀ,    O = W V.
```

`λ` is learned per head. `θ` is initialised so that `λ = 0.1`. With `T = 64`, every key coordinate other than the constant carries the penalty `64λ`. The head runs in fp32, and the Cholesky factorization uses that precision. The explicit inverse is never formed. The forward pass evaluates `O = Q̃ (A⁻¹ (K̃ᵀ V))`, which is `W V` with the product associated the other way.

The rows of `W` sum to one, because this `Λ` meets the two conditions of §1:
- the constant key coordinate is unpenalized;
- every query has 1 in that coordinate.

They do not under a penalty on the constant. Take a constant key column under `λI`, and a query `q` with 1 in that coordinate. Since `Kᵀ1 = KᵀK e₀`, its row sums to `1 − λ·e₀ᵀ(KᵀK + λI)⁻¹ q`, where `e₀` picks the constant coordinate. That is 1 only at `λ = 0`, or in special cases such as a dual direction `(KᵀK + λI)⁻¹ q` with no constant component.

`T ≫ d_head`, or else `K (KᵀK)⁻¹ Kᵀ ≈ I` and the head copies its input. The built head has `T / d_head = 8`, and the code refuses a smaller ratio. `λ` stays positive: the floor is `10⁻³`.

**Memory.** Every arm reads one memory. The key is `z = W_z h ∈ R⁸` with no bias, the value is the one-hot label, and the stamp is the row's `label_known_at`. A readout at decision time `d` uses the entries with `d − 30 days < stamp ≤ d`. That is the only window. It is the centred ridge of §3.2, on keys centred inside the window:

```
m = ȳ + β (z_q − z̄),    β = M_yz (M_zz + Λ)⁻¹,
Λ = ρ · diag(M_zz) + 10⁻⁶ · n_w · I,    ρ = 10⁻².
```

An empty window gives `(1/3, 1/3, 1/3)`. The normal matrix is 8 × 8, and its Cholesky factorization is in fp64. The fp32 factorization above belongs to the OLS head. The memory's solve is a separate one, and it stays in fp64.

The window is the difference of two prefix sums of the raw moments `(n, Σz, Σzzᵀ, Σy, Σyzᵀ)`, in the manner of the mesa-layer. The prefixes accumulate in fp64, and the difference is taken in fp64 before centring. A difference of two large fp32 prefixes can lose positive definiteness, so the memory never casts those moments to fp32 for the factorization. The solved `m` is cast to the head's dtype, fp32 in the runs, only once the solve is done. The tree of §3 is a different object: its merges add node statistics. The encoder's memory does not read that tree.

Training walks the rows in weekly episodes, Monday 00:00 UTC to the next Monday, over every member. Inside an episode, the memory at `d` holds the training rows in the 30-day window. Rows of the episode are keyed with gradient. Older rows the window needs are keyed with the current weights and without gradient. A fit is once a year, at 00:00 UTC on 1 January, for each test year from 2019 through 2024.

A readout at one year, and a readout over all history, are not built beside this window. §7 still plans to stack the three, with three weights.

## 5. Decision head and decision rule

- **Options.** The head outputs probabilities over three classes, down, neutral or up, across horizon h. The threshold is set by fees and volatility:

  ```
  threshold_t = max(k · σ_t · √h, floor),
  ```

  with `k = 1/2` and `floor = 0.0024`. `σ_t` is the sample standard deviation of the last 30 one-bar log returns. On the hourly panel that window is 30 hours, and `h = 24` bars. The floor is one round trip on the Binance history: taker fee 0.10 % a side, plus 2 bps of slippage a side. The class is down when the forward log return is below `−threshold`, up when it is above `+threshold`, and neutral otherwise. The floor prices that history. The rule below prices the venue of a trade.
- **Inputs.** The head reads `[h, m]`: the encoder state `h ∈ R³²` concatenated with the one memory readout `m ∈ R³`, 35 inputs. The map is `35 → 32 → GELU → 3` logits, then a softmax. It does not read a one-year readout or an all-history readout. Those resolutions remain the stack planned in §7, and that stack is not built.
- **Loss.** The trained arms minimise the mean cross-entropy of the logits. That is the cross-entropy of the softmax, taken before the floor. The probabilities that are scored pass through `p_k = max(p_k, 10⁻³) / Σⱼ max(pⱼ, 10⁻³)`.
  - A Brier term added to the cross-entropy is H6, the registered ablation `enc-ols-brier`. It is not part of the arms that were trained. The registered run, on the real panel, has not been made.
  - Label smoothing is not built. It would pull every probability towards uniform, which biases the expected edge the rule compares with costs. It is not a registered ablation.
  - Temperature scaling of those probabilities, fitted on the fold's last segment, is not built. No part of the encoder recalibrates them.
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

Two timescales follow: Matrix and the readout weights at every bar, gradient-free; the encoder and the head at monthly refits, plus early refits on alarm. The encoder built in §4 is fitted once a year. Its monthly refit, and a refit on alarm, are not built.

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
- 2026-10-29: iteration 3 builds the centred core.
  - §3.1: centred nodes now exist. They sit beside the raw tree, which keeps every raw read's bits.
  - §3.2: the centred ridge with a free intercept and its refusal bound.
  - §3.3: the batch readouts, their statuses and their measured speed.
- 2026-11-01: §4 and §5 now describe the encoder iteration 4 built.
  - §4: one token per hourly bar of one pair, in a window of 64. The OLS penalty is `diag(0, Tλ, …, Tλ)`, and that Cholesky is in fp32. The memory is a single 30-day window; its prefix sums and its 8 × 8 solve stay in fp64. Episodes are weeks, and fits are yearly.
  - §5: the head reads `[h, m]` (35) and the trained loss is cross-entropy. The Brier term remains H6; its registered run on the real panel has not been made. Label smoothing and temperature scaling are not built. One-year and all-history readouts stay the plan in §7.
  - §7 no longer says the protocol already refits the encoder every month.
