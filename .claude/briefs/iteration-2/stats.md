# Brief — Statistics of the bench (iteration 2)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:** a second `ml-engineer` (baselines), two `data-engineer`s, two `rust-engineer`s, and a `scout` who is checking the formulas below against the papers right now. If the orchestrator sends a correction, apply it.
- **Motto:** we should be more ambitious: every number in `docs/RESULTS.md` will carry an honest interval, and the tests prove the intervals cover.

## Why

`docs/PROTOCOL.md` says that PnL has little statistical power and that every Sharpe ratio comes with the number of variants tried and a deflated Sharpe ratio. The iteration 1 audit logged that no Sharpe had an interval yet. This module closes that gap, before any row reaches `RESULTS.md`.

## The module: `python/olsprime/eval/stats.py`

Pure numpy and scipy; no I/O. Every random function takes a `numpy.random.Generator` or a seed. Conventions:
- r is a 1-D array of per-period simple returns;
- SR is the per-period Sharpe ratio, `mean / std` with ddof 1;
- γ₃ is the skewness and γ₄ the **non-excess** kurtosis (normal = 3), both as population moments;
- T is the number of returns.

**Sharpe ratios.**
- `sharpe(r)`, and `annualized(sr, periods_per_year) = sr·√ppy`.
- `sharpe_se(r)`: for non-normal iid returns (Mertens 2002), `√((1 − γ₃·SR + (γ₄ − 1)/4·SR²)/(T − 1))`. This is the form inside PSR; with normal returns it is Lo's `√((1 + SR²/2)/T)`, up to T − 1. Document both.
- `sharpe_ci(r, ppy, level=0.95)`: the analytic interval, annualized.
- `sharpe_ci_bootstrap(r, ppy, level, n_boot=2000, seed, mean_block=None)`: the stationary bootstrap (Politis and Romano 1994). With `mean_block=None`, the block length comes from `optimal_block_length(r)`, Politis and White (2004) as corrected by Patton, Politis and White (2009). Percentile interval.
- `sharpe_diff_ci_bootstrap(r_a, r_b, ...)`: the paired version, with the same resampled indices for both series. H9 uses it.

**Probabilistic and deflated Sharpe.**
- `psr(r, sr_star=0.0)` (Bailey and López de Prado 2012): `Φ[(SR̂ − SR*)·√(T − 1) / √(1 − γ₃·SR̂ + (γ₄ − 1)/4·SR̂²)]`.
- `expected_max_sharpe(n_trials, var_sr)` (Bailey and López de Prado 2014): `√var_sr·((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))`, with γ ≈ 0.5772156649 (Euler's constant).
- `dsr(r, n_trials, var_sr)`: `psr(r, expected_max_sharpe(n_trials, var_sr))`. At N = 1 it is `psr(r, 0)`; document it. `var_sr` is the variance of the per-period Sharpe ratios across the trials (ddof 1). Document that it is noisy when N is small.

**Forecasts.** Probabilities are arrays of shape (n, 3); labels are in {0, 1, 2}.
- `log_loss(p, y, eps=1e-15)`, the mean.
- `brier(p, y)`, the mean multi-class Brier: `Σ_k (p_k − 1{y = k})²`.
- `brier_skill(p, p_ref, y)`: `1 − B/B_ref`.
- `ece(p, y, n_bins=10, strategy="width" | "mass")`: top-label ECE, plus `reliability_table(...)` for diagrams.
- `per_row_losses(p, y)`: the per-row log-loss and Brier, for Diebold-Mariano and for the bootstrap.
- `bss_ci_bootstrap(p, p_ref, y, ...)`: a paired stationary bootstrap on the per-row Brier losses. H2 needs it in iteration 3.

**Comparisons.**
- `diebold_mariano(loss_a, loss_b, h=1)`, with d = loss_a − loss_b:
  - the HAC (Newey-West, Bartlett) variance of the mean of d, with h − 1 lags;
  - the Harvey, Leybourne and Newbold (1997) correction, `√((T + 1 − 2h + h(h − 1)/T)/T)`;
  - a Student t with T − 1 degrees of freedom.
  - It returns the statistic, a two-sided p-value and the mean of d.
  - If the HAC variance is ≤ 0, fall back to `h = 1`, and say so in the result.

## Tests (the part that convinces a sceptic)

- **Closed forms.**
  - With normal returns, `sharpe_se` is within 2 % of `√((1 + SR²/2)/T)` for large T.
  - Skewed and fat-tailed inputs move the standard error in the right direction.
  - DSR at N = 1 equals PSR(0).
  - `expected_max_sharpe` matches a Monte Carlo estimate of E[max of N normal Sharpes]: N = 10, 100; within 3 %.
- **Coverage, by Monte Carlo.** For iid normal returns with a known SR, the analytic and bootstrap 95 % intervals each cover the true SR in 95 % ± 2.5 % of 400 to 1000 runs.
  - For an AR(1) series with ρ = 0.3, the stationary bootstrap covers better than the iid analytic interval.
  - Keep each test under about 20 s. Mark longer ones `@pytest.mark.slow` and skip them by default.
- **Diebold-Mariano.**
  - Equal-accuracy forecasts give a p-value roughly uniform: check the rejection rate at 5 % over 400 simulations is within 2–9 %.
  - A clearly better forecast gives p < 0.01.
  - With h = 7, the result agrees with a hand-computed Newey-West.
- **Forecast metrics.**
  - `log_loss` and `brier` match hand computations.
  - The ECE of perfectly calibrated synthetic probabilities is near 0; that of overconfident ones is large.
  - BSS of the reference against itself is 0.
- **Block length.** `optimal_block_length` grows with autocorrelation and is about 1 for white noise.
- Run the whole `uv run pytest` at the end. If another agent's test fails, report it; do not fix it.

## Scope and constraints

- **Files.** Only `python/olsprime/eval/stats.py`, `python/olsprime/eval/__init__.py` (exports; keep its docstring) and `python/tests/test_stats.py`.
- **Not yours.** `eval/splits.py` belongs to a data-engineer working right now. Everything else in `python/` is someone else's. Add no dependency: `scipy` is installed.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\stats\`.
- **Git.** Run no git command that changes state, and do not commit.

## Report (short)

- The API.
- Tests, with counts and the coverage figures you measured.
- Every formula you changed from this brief, with the source.
- One bolder next step.
