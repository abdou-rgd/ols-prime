use nalgebra::{Cholesky, DMatrix, DVector, DVectorView, Dim, Dyn, Matrix, RawStorage};

use crate::OlsError;
use crate::stats::{SuffStats, check_finite, check_len};

/// The ridge penalty `Λ` and the intercept convention.
///
/// # Numerical singularity
///
/// Every solve factors `M = G + Λ` and refuses with [`OlsError::NotPositiveDefinite`] when `M`
/// lies within rounding of a singular matrix, so that a singular design never returns numbers.
///
/// A Cholesky factor that merely exists proves nothing: its k-th squared pivot is the Schur
/// complement `M_kk − m_kᵀ M_<k⁻¹ m_k` of the leading block, exactly 0 for a column in the span of
/// the previous ones, and rounding leaves a tiny value of either sign. A threshold on the pivots
/// is not enough either: the computed pivot of such a column errs by up to
/// `(1 + ‖M_<k⁻¹ m_k‖²)·‖ΔM‖`, which an ill-conditioned leading block inflates past any fixed
/// bound. The test is therefore on the spectrum, after diagonal scaling:
///
/// ```text
/// S = D^(-1/2) M D^(-1/2),  D = diag(M);   accept only if  λ_min(S) > τ = 4p(N + p + 2)ε,
/// ```
///
/// with `N` the number of entries summed into `G` ([`SuffStats::count`]) and `ε = 2⁻⁵²`.
///
/// - **Scaling.** `S` has a unit diagonal and `|sᵢⱼ| ≤ 1`, so `τ` needs no unit: the criterion
///   is the same whether a feature is in dollars or in millions of dollars. The accuracy of a
///   Cholesky solve is governed by `κ(S)` (van der Sluis), not by `κ(M)`.
/// - **Rounding of the data.** Each entry of `G` is a sum of `N` products, so
///   `|fl(G)ᵢⱼ − Gᵢⱼ| ≤ γ_(N+2) Σ w|xᵢ||xⱼ| ≤ γ_(N+2) √(Gᵢᵢ Gⱼⱼ)` by Cauchy–Schwarz, with
///   `γ_k = kε/(1 − kε)`. Cauchy–Schwarz needs `w ≥ 0`, which [`SuffStats::push`] enforces;
///   statistics written by hand into the public fields are outside this guarantee. The scaled
///   entry `mᵢⱼ·sᵢ·sⱼ`, with `s = 1/√d`, carries that `(N + 2)ε`, plus the error of both
///   diagonals through the square root, `(N + 3)ε/2` each with the `λ` add, plus one rounding
///   each for the two square roots, the two divisions and the two products. Every entry of `S`
///   therefore errs by at most `(2N + 11)ε` to first order, and the error `E` has
///   `‖E‖₂ ≤ ‖E‖_F ≤ p(2N + 11)ε`. The factor 4 in `τ` absorbs this.
/// - **Rounding of the eigensolver.** Householder tridiagonalization and implicit QR are
///   backward stable: the computed eigenvalues are exact for `S + F`, with `‖F‖₂ ≤ c ε ‖S‖₂` for
///   a small constant `c` (a modest multiple of `p`), and `‖S‖₂ ≤ tr S = p`.
/// - **Soundness.** If the design is singular in exact arithmetic (duplicate entries,
///   collinear columns, fewer entries than unpenalized coordinates), Weyl's inequality bounds
///   the computed `λ_min(S)` by `p(2N + 11)ε + c p ε`. Since `τ = p(4N + 4p + 8)ε`, that is at
///   most `τ` whenever `c ≤ 2N + 4p − 3`, so for every `N ≥ 1` any `c ≤ 4p − 1` will do. Such a
///   design is rejected whatever the rounding does.
/// - **What acceptance means.** An accepted `M` still has `κ(S)` up to `1/τ`: the criterion
///   certifies that the problem is not singular, not that the answer has many digits. Its
///   relative error is of order `κ(S)·(N + p)·ε`. For p = 16 and N = 10⁵, a design with `κ(S)`
///   above about 7·10⁸ is refused. A penalty `λ > 0` lifts the eigenvalues of the penalized
///   block of `S` by at least `λ / max Dᵢᵢ`; with a free intercept, coordinate 0 is lifted only
///   through its correlation with the penalized coordinates, so a penalty does not cure a
///   badly scaled intercept.
/// - **Centre and scale the features inside the fold.** An uncentred feature next to the
///   intercept is nearly collinear with it: with `r = |mean| / std`, the pair has
///   `λ_min(S) ≈ 1/(2r²)`. At λ = 0 and p = 33 (32 features and the intercept), such a feature
///   is refused from about `r ≈ 1.3·10⁴` at N = 10⁵ and `r ≈ 4·10³` at N = 10⁶. Standardized
///   features, with mean and scale fitted inside the fold, are far from that frontier
///   (`long_standardized_designs_are_accepted` in the tests).
///
/// The bound on the sums is the worst case of a sequential sum; Matrix's tree sums do better,
/// so the criterion is conservative there. The check costs one symmetric eigenvalue
/// decomposition, `O(p³)` like the factor itself. Every returned coefficient, direction,
/// weight and forecast is also checked to be finite ([`OlsError::NonFinite`] otherwise).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgeConfig {
    lambda: f64,
    intercept: bool,
}

impl RidgeConfig {
    /// Penalty `λ ≥ 0` on every coordinate: `Λ = λI`.
    pub fn new(lambda: f64) -> Result<Self, OlsError> {
        Self::build(lambda, false)
    }

    /// Penalty `λ ≥ 0` on every coordinate but the first, which the caller sets to the
    /// constant 1. The intercept is then free, and every row of dual weights sums to one
    /// (`docs/DESIGN.md` §1).
    pub fn with_intercept(lambda: f64) -> Result<Self, OlsError> {
        Self::build(lambda, true)
    }

    fn build(lambda: f64, intercept: bool) -> Result<Self, OlsError> {
        if lambda.is_finite() && lambda >= 0.0 {
            Ok(Self { lambda, intercept })
        } else {
            Err(OlsError::InvalidLambda(lambda))
        }
    }

    /// The penalty `λ`.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Whether coordinate 0 is an unpenalized intercept.
    pub fn intercept(&self) -> bool {
        self.intercept
    }

    /// The Cholesky factor of `M = G + Λ`, once `M` is certified to be numerically positive
    /// definite: `λ_min(D^(-1/2) M D^(-1/2)) > 4p(N + p + 2)ε`, where `N = count` is the number
    /// of entries summed into `gram` (see the type's docs for the derivation).
    fn factor(&self, gram: &DMatrix<f64>, count: u64) -> Result<Cholesky<f64, Dyn>, OlsError> {
        let p = gram.nrows();
        let mut m = gram.clone();
        for i in usize::from(self.intercept)..p {
            m[(i, i)] += self.lambda;
        }
        // An overflowed Gram is not a singular one: say so.
        check_finite(m.as_slice())?;
        let diag: Vec<f64> = (0..p).map(|i| m[(i, i)]).collect();
        // A zero diagonal is a coordinate that no entry and no penalty touches.
        if diag.iter().any(|&d| d <= 0.0) {
            return Err(OlsError::NotPositiveDefinite);
        }
        let scale: Vec<f64> = diag.iter().map(|d| d.sqrt().recip()).collect();
        let scaled = DMatrix::from_fn(p, p, |i, j| m[(i, j)] * scale[i] * scale[j]);
        if !scaled.iter().all(|v| v.is_finite()) {
            return Err(OlsError::NotPositiveDefinite);
        }
        // nalgebra 0.35 iterates without a cap here (`max_niter = 0`) and returns only once
        // every off-diagonal has deflated: no misconverged value comes back.
        let lambda_min = scaled.symmetric_eigenvalues().min();
        let tau = 4.0 * p as f64 * (count as f64 + p as f64 + 2.0) * f64::EPSILON;
        if lambda_min.is_nan() || lambda_min <= tau {
            return Err(OlsError::NotPositiveDefinite);
        }
        m.cholesky().ok_or(OlsError::NotPositiveDefinite)
    }
}

/// Primal ridge coefficients `β = C (G + Λ)⁻¹`, one row per outcome (q × p).
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeModel {
    pub beta: DMatrix<f64>,
}

impl RidgeModel {
    /// The forecast `β x`.
    pub fn predict(&self, x: &[f64]) -> Result<DVector<f64>, OlsError> {
        check_len(self.beta.ncols(), x.len())?;
        check_finite(x)?;
        finite(&self.beta * DVectorView::from_slice(x, x.len()))
    }
}

/// Fits ridge regression from sufficient statistics alone: `β = C (G + Λ)⁻¹`.
pub fn ridge_fit(stats: &SuffStats, cfg: &RidgeConfig) -> Result<RidgeModel, OlsError> {
    let chol = cfg.factor(&stats.gram, stats.count)?;
    // G + Λ is symmetric, so βᵀ = (G + Λ)⁻¹ Cᵀ.
    let beta = finite(chol.solve(&stats.cross.transpose()).transpose())?;
    Ok(RidgeModel { beta })
}

/// The dual direction `a = (G + Λ)⁻¹ x_q` of a query.
///
/// Everything about the query follows from `a`: entry i weighs `wᵢ = xᵢᵀ a`, the forecast is
/// `C a = Σᵢ wᵢ yᵢ`, and a subset S of entries contributes `C_S a`.
pub fn solve_query(
    stats: &SuffStats,
    cfg: &RidgeConfig,
    x_q: &[f64],
) -> Result<DVector<f64>, OlsError> {
    check_len(stats.p(), x_q.len())?;
    check_finite(x_q)?;
    let chol = cfg.factor(&stats.gram, stats.count)?;
    finite(chol.solve(&DVector::from_column_slice(x_q)))
}

/// In-sample residual sum of squares of `model` over the entries behind `stats`, per outcome.
///
/// Expanding the square gives `Σ w (yⱼ − βⱼ x)² = Σ w yⱼ² − 2 βⱼ·Cⱼ + βⱼ G βⱼᵀ`: it is read off
/// the statistics, without the entries. Cancellation can push an almost perfect fit a hair below
/// zero, so the result is clamped at zero; a non-finite sum is an error, not a zero.
pub fn residual_sum_squares(
    stats: &SuffStats,
    model: &RidgeModel,
) -> Result<DVector<f64>, OlsError> {
    check_len(stats.q(), model.beta.nrows())?;
    check_len(stats.p(), model.beta.ncols())?;
    let beta_gram = &model.beta * &stats.gram;
    let rss = DVector::from_fn(stats.q(), |j, _| {
        let beta = model.beta.row(j);
        stats.sum_yy[j] - 2.0 * beta.dot(&stats.cross.row(j)) + beta.dot(&beta_gram.row(j))
    });
    // `f64::max` would turn a NaN into 0: check before clamping.
    Ok(finite(rss)?.map(|r| r.max(0.0)))
}

/// Dense OLS attention over a batch of queries.
#[derive(Debug, Clone, PartialEq)]
pub struct Attention {
    /// `W = X_q (XᵀX + Λ)⁻¹ Xᵀ` (m × n): row j holds the signed weights of query j over the keys.
    pub weights: DMatrix<f64>,
    /// `Ŷ = W Y` (m × q).
    pub forecast: DMatrix<f64>,
}

/// OLS as attention (Goulet Coulombe): `keys` (n × p) are the past states, `values` (n × q) their
/// outcomes and `queries` (m × p) the states to forecast, one per row.
///
/// The weights come from the space whitened by `(XᵀX + Λ)⁻¹` instead of a softmax, so they can be
/// negative. They must be computed from past rows only: `XᵀX` is part of the fit.
pub fn ols_attention(
    keys: &DMatrix<f64>,
    values: &DMatrix<f64>,
    queries: &DMatrix<f64>,
    cfg: &RidgeConfig,
) -> Result<Attention, OlsError> {
    check_len(keys.nrows(), values.nrows())?;
    check_len(keys.ncols(), queries.ncols())?;
    for m in [keys, values, queries] {
        check_finite(m.as_slice())?;
    }
    let chol = cfg.factor(&keys.tr_mul(keys), keys.nrows() as u64)?;
    // A = (XᵀX + Λ)⁻¹ X_qᵀ, hence W = X_q (XᵀX + Λ)⁻¹ Xᵀ = (X A)ᵀ.
    let a = chol.solve(&queries.transpose());
    let weights = finite((keys * a).transpose())?;
    let forecast = finite(&weights * values)?;
    Ok(Attention { weights, forecast })
}

/// `m` itself if every value is finite.
pub(crate) fn finite<R: Dim, C: Dim, S: RawStorage<f64, R, C>>(
    m: Matrix<f64, R, C, S>,
) -> Result<Matrix<f64, R, C, S>, OlsError> {
    if m.iter().all(|v| v.is_finite()) {
        Ok(m)
    } else {
        Err(OlsError::NonFinite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penalties_must_be_finite_and_non_negative() {
        assert!(RidgeConfig::new(0.0).is_ok());
        for bad in [-1e-12, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                RidgeConfig::new(bad),
                Err(OlsError::InvalidLambda(_))
            ));
            assert!(RidgeConfig::with_intercept(bad).is_err());
        }
    }

    #[test]
    fn a_free_intercept_needs_at_least_one_entry() {
        let empty = SuffStats::zeros(2, 1);
        let cfg = RidgeConfig::with_intercept(1.0).unwrap();
        assert_eq!(ridge_fit(&empty, &cfg), Err(OlsError::NotPositiveDefinite));
        // Penalizing every coordinate makes even the empty fit well defined: β = 0.
        let fit = ridge_fit(&empty, &RidgeConfig::new(1.0).unwrap()).unwrap();
        assert_eq!(fit.beta, DMatrix::zeros(1, 2));
    }

    #[test]
    fn attention_checks_its_shapes() {
        let keys = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let values = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let cfg = RidgeConfig::new(1.0).unwrap();
        assert!(matches!(
            ols_attention(&keys, &values, &keys, &cfg),
            Err(OlsError::Dimension { .. })
        ));
    }
}
