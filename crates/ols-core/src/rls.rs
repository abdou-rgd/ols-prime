use nalgebra::{DMatrix, DVector};

use crate::OlsError;
use crate::stats::{check_finite, check_len};

/// Recursive least squares with forgetting factor `γ`: the online, primal twin of Matrix, with
/// no stored history and an exponential memory.
///
/// It keeps `β` (q × p) and `P = (G + γⁿλI)⁻¹`, where `G = Σᵢ γ^(n−1−i) xᵢxᵢᵀ`, and updates both
/// in O(p²) per observation. After n updates `β` equals weighted ridge with weights `γ^(n−1−i)`
/// and penalty `γⁿλ` on every coordinate, the intercept included (`docs/DESIGN.md` §2).
///
/// With `γ < 1` and inputs that stop exploring some direction, `P` grows like `γ⁻ⁿ` along it
/// (covariance wind-up). Keep `γ` close to 1 or the inputs rich.
///
/// A small `λ` costs digits: the first updates cancel entries of `P₀ = I/λ`, so `β` carries a
/// relative error of order `ε·max‖x‖²/λ` on top of the batch problem's `ε·κ`. With unit-scale
/// states, `λ = 10⁻¹²` leaves about four correct digits. Unlike the batch solvers, RLS has no
/// singularity check: it never refuses, so keep `λ` well above `ε·max‖x‖²`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rls {
    gamma: f64,
    beta: DMatrix<f64>,
    inv_gram: DMatrix<f64>,
    updates: u64,
}

impl Rls {
    /// Starts from `β = 0` and `P = I/λ`, with `λ > 0` and `γ ∈ (0, 1]`.
    pub fn new(p: usize, q: usize, lambda: f64, gamma: f64) -> Result<Self, OlsError> {
        if p == 0 || q == 0 {
            return Err(OlsError::InvalidShape("p and q must be at least 1"));
        }
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(OlsError::InvalidLambda(lambda));
        }
        if !(gamma > 0.0 && gamma <= 1.0) {
            return Err(OlsError::InvalidForgetting(gamma));
        }
        Ok(Self {
            gamma,
            beta: DMatrix::zeros(q, p),
            inv_gram: DMatrix::identity(p, p) / lambda,
            updates: 0,
        })
    }

    /// Absorbs one observation:
    ///
    /// ```text
    /// k = P x / (γ + xᵀ P x),   β ← β + (y − β x) kᵀ,   P ← (P − k (P x)ᵀ) / γ.
    /// ```
    pub fn update(&mut self, x: &[f64], y: &[f64]) -> Result<(), OlsError> {
        check_len(self.beta.ncols(), x.len())?;
        check_len(self.beta.nrows(), y.len())?;
        check_finite(x)?;
        check_finite(y)?;
        let x = DVector::from_column_slice(x);
        let px = &self.inv_gram * &x;
        let gain = &px / (self.gamma + x.dot(&px));
        let err = DVector::from_column_slice(y) - &self.beta * &x;
        self.beta.ger(1.0, &err, &gain, 1.0);
        self.inv_gram.ger(-1.0, &gain, &px, 1.0);
        self.inv_gram /= self.gamma;
        // Rounding breaks the symmetry of P a little at every step; restoring it keeps long runs
        // stable.
        self.inv_gram = (&self.inv_gram + self.inv_gram.transpose()) * 0.5;
        self.updates += 1;
        Ok(())
    }

    /// The forecast `β x`.
    pub fn predict(&self, x: &[f64]) -> Result<DVector<f64>, OlsError> {
        check_len(self.beta.ncols(), x.len())?;
        check_finite(x)?;
        Ok(&self.beta * DVector::from_column_slice(x))
    }

    /// The current coefficients (q × p).
    pub fn beta(&self) -> &DMatrix<f64> {
        &self.beta
    }

    /// Number of observations absorbed.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// The forgetting factor `γ`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_are_validated() {
        assert!(Rls::new(2, 1, 1.0, 1.0).is_ok());
        assert!(matches!(
            Rls::new(2, 1, 0.0, 1.0),
            Err(OlsError::InvalidLambda(_))
        ));
        for gamma in [0.0, 1.0 + 1e-12, f64::NAN] {
            assert!(matches!(
                Rls::new(2, 1, 1.0, gamma),
                Err(OlsError::InvalidForgetting(_))
            ));
        }
        assert!(Rls::new(0, 1, 1.0, 1.0).is_err());
    }

    #[test]
    fn a_noiseless_line_is_learned() {
        let mut rls = Rls::new(2, 1, 1e-6, 1.0).unwrap();
        for i in 0..50 {
            let x = f64::from(i) / 10.0;
            rls.update(&[1.0, x], &[0.5 - 2.0 * x]).unwrap();
        }
        assert_eq!(rls.updates(), 50);
        let forecast = rls.predict(&[1.0, 10.0]).unwrap()[0];
        assert!((forecast - (0.5 - 20.0)).abs() < 1e-4, "{forecast}");
        assert!(rls.update(&[1.0], &[0.0]).is_err());
        assert_eq!(rls.updates(), 50, "a rejected update changes nothing");
    }
}
