//! Trading metrics of a [`SimResult`].
//!
//! All ratios are computed on per-bar simple returns `rₜ = eₜ / eₜ₋₁ − 1` and
//! annualized with [`periods_per_year`]. The risk-free rate is 0.
//!
//! **Regular grid.** Annualizing by a fixed number of bars per year assumes
//! that consecutive bars are exactly one bar length apart. [`crate::simulate`]
//! only requires strictly increasing `open_time`, so it accepts series with
//! gaps or off-grid bars; their returns span more (or less) than one period
//! and every annualized figure here would be silently biased. Callers must
//! re-align series to the grid and cut them at gaps before scoring them.
//!
//! Read them with `docs/PROTOCOL.md` in mind: over short windows PnL metrics
//! have little statistical power, and every reported Sharpe ratio must come
//! with the number of variants tried and a deflated Sharpe ratio, which this
//! module does not compute.

use crate::types::SimResult;

/// Bars per year for a bar length of `bar_ms` milliseconds. Crypto trades
/// every day of the year, so a year is 365 × 86 400 000 ms (no trading-day
/// calendar). It assumes a regular grid: see the module docs.
///
/// # Panics
///
/// Panics if `bar_ms ≤ 0`: a bar length is a programmer-supplied constant.
pub fn periods_per_year(bar_ms: i64) -> f64 {
    assert!(bar_ms > 0, "bar length must be positive, got {bar_ms} ms");
    365.0 * 86_400_000.0 / bar_ms as f64
}

/// Largest peak-to-trough loss, `max_t (1 − eₜ / max_{s≤t} eₛ)`, in [0, 1].
/// Returns 0 for an empty or never-falling series.
pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut mdd: f64 = 0.0;
    for &e in equity {
        peak = peak.max(e);
        if peak > 0.0 {
            mdd = mdd.max(1.0 - e / peak);
        }
    }
    mdd.clamp(0.0, 1.0)
}

/// Summary of one simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Number of per-bar returns: bars − 1.
    pub periods: usize,
    /// `equity[last] / equity[0] − 1`.
    pub total_return: f64,
    /// Geometric: `(1 + total)^(ppy / periods) − 1`; 0 when `periods == 0`.
    pub annualized_return: f64,
    /// Sample standard deviation (n − 1) of per-bar returns × √ppy; 0 when
    /// fewer than two returns exist or all returns are identical.
    pub annualized_volatility: f64,
    /// `mean(r) / std(r) · √ppy`, with the sample standard deviation above.
    /// `None` with fewer than two returns or a zero standard deviation.
    pub sharpe: Option<f64>,
    /// `mean(r) / dd · √ppy`, with downside deviation
    /// `dd = √(Σ min(r, 0)² / n)` (target return 0). `None` with fewer than two
    /// returns or no negative return.
    pub sortino: Option<f64>,
    pub max_drawdown: f64,
    /// Traded notional over mean equity, scaled to one year:
    /// `Σ notional / mean(equity) · ppy / periods`; 0 when `periods == 0`.
    /// Buys and sells both count, so a round trip of the whole equity adds 2,
    /// and trading the whole equity once per bar gives `ppy`. The mean runs
    /// over every bar, bar 0 included.
    pub turnover: f64,
    /// Fraction of bars ending with a position > 0, bar 0 included.
    pub exposure: f64,
    pub fees_paid: f64,
}

/// Computes [`Metrics`] for `result`, annualizing with `periods_per_year`.
///
/// # Panics
///
/// Panics if `result.equity` is empty (`simulate` never returns one) or if
/// `periods_per_year` is not finite and > 0.
pub fn compute(result: &SimResult, periods_per_year: f64) -> Metrics {
    let eq = &result.equity;
    assert!(!eq.is_empty(), "metrics of an empty equity curve");
    assert!(
        periods_per_year.is_finite() && periods_per_year > 0.0,
        "periods_per_year must be finite and > 0, got {periods_per_year}"
    );
    let ppy = periods_per_year;
    let periods = eq.len() - 1;
    let returns: Vec<f64> = eq.windows(2).map(|w| w[1] / w[0] - 1.0).collect();

    let total_return = eq[periods] / eq[0] - 1.0;
    let annualized_return = if periods == 0 {
        0.0
    } else {
        (1.0 + total_return).powf(ppy / periods as f64) - 1.0
    };

    let (annualized_volatility, sharpe, sortino) = if returns.len() < 2 {
        (0.0, None, None)
    } else {
        let n = returns.len() as f64;
        // Identical returns have a standard deviation of exactly 0. Computed
        // naively, the rounding of `mean` leaves a variance of a few ulps
        // squared and a meaningless Sharpe ratio of order 10¹⁵, so this case
        // is decided exactly.
        let (mean, std) = if returns.iter().all(|&r| r == returns[0]) {
            (returns[0], 0.0)
        } else {
            let mean = returns.iter().sum::<f64>() / n;
            let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, var.sqrt())
        };
        let down = (returns.iter().map(|r| r.min(0.0).powi(2)).sum::<f64>() / n).sqrt();
        let ratio = |den: f64| {
            let x = mean / den * ppy.sqrt();
            (den > 0.0 && x.is_finite()).then_some(x)
        };
        (std * ppy.sqrt(), ratio(std), ratio(down))
    };

    let mean_equity = eq.iter().sum::<f64>() / eq.len() as f64;
    let traded: f64 = result.fills.iter().map(|f| f.notional).sum();
    let turnover = if periods == 0 || mean_equity.is_nan() || mean_equity <= 0.0 {
        0.0
    } else {
        traded / mean_equity * ppy / periods as f64
    };
    let exposure = if result.position.is_empty() {
        0.0
    } else {
        result.position.iter().filter(|&&p| p > 0.0).count() as f64 / result.position.len() as f64
    };

    Metrics {
        periods,
        total_return,
        annualized_return,
        annualized_volatility,
        sharpe,
        sortino,
        max_drawdown: max_drawdown(eq),
        turnover,
        exposure,
        fees_paid: result.fees_paid,
    }
}
