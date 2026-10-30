//! The primal side of a centred readout (iteration 4): `CentredReadout::beta` and `mean_y`, the
//! coefficients and the intercept that `olsprime.explain why` splits a forecast with.
//!
//! - `ȳ + β x̃`, with `x̃ = x_q − x̄`, reproduces the forecast **to the bit** when it repeats the
//!   forecast's own operations, and within the inner-product bound of Higham (Thm 3.1) when it
//!   runs in another order (nalgebra's `gemv`).
//! - `β` and `ȳ` are the bits of `CentredFit` on the same statistics: one solve, not two.
//! - `β` is the closed form `M_yx (M_xx + Λ)⁻¹`, solved independently by LU, within first-order
//!   backward-error bounds; and `β x̃ = M_yx a` (primal against dual).
//! - At `ρ = 0` on a well-posed design, `β` is the slope block of ordinary least squares with an
//!   intercept, solved by an SVD of the raw design `[1, X]`.

// The references are written index by index, as the formulas are.
#![allow(clippy::needless_range_loop)]

use nalgebra::{DMatrix, DVector};
use ols_core::{CentredFit, CentredReadout, CentredRidge, CentredStats, Matrix, OlsError};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;

/// The unit roundoff, `u = ε/2 = 2⁻⁵³`.
const U: f64 = f64::EPSILON / 2.0;

/// Higham's `γ_k = k·u / (1 − k·u)`.
fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

#[derive(Debug, Clone)]
struct Data {
    p: usize,
    q: usize,
    xs: Vec<f64>,
    ys: Vec<f64>,
    x_q: Vec<f64>,
}

impl Data {
    fn n(&self) -> usize {
        self.xs.len() / self.p
    }

    fn x(&self, i: usize) -> &[f64] {
        &self.xs[i * self.p..(i + 1) * self.p]
    }

    fn y(&self, i: usize) -> &[f64] {
        &self.ys[i * self.q..(i + 1) * self.q]
    }

    fn matrix(&self, block_log2: u32) -> Matrix {
        let mut m = Matrix::new(self.p, self.q, block_log2).unwrap();
        for i in 0..self.n() {
            m.push(i as i64, self.x(i), self.y(i)).unwrap();
        }
        m
    }

    /// Adds `shift` to feature `j` of every entry and of the query.
    fn shifted(mut self, j: usize, shift: f64) -> Self {
        for row in self.xs.chunks_exact_mut(self.p) {
            row[j] += shift;
        }
        self.x_q[j] += shift;
        self
    }
}

fn data(max_p: usize, max_q: usize, min_n: usize, max_n: usize) -> impl Strategy<Value = Data> {
    (1..=max_p, 1..=max_q, min_n..=max_n).prop_flat_map(|(p, q, n)| {
        (
            vec(-1.0..1.0, n * p),
            vec(-1.0..1.0, n * q),
            vec(-1.0..1.0, p),
        )
            .prop_map(move |(xs, ys, x_q)| Data { p, q, xs, ys, x_q })
    })
}

fn cfg(rho: f64, scale: &[f64]) -> CentredRidge {
    CentredRidge::new(rho, scale).unwrap()
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// `x̃ = x_q − x̄`, one IEEE subtraction per coordinate, as the forecast computed it.
fn x_tilde(r: &CentredReadout, x_q: &[f64]) -> Vec<f64> {
    x_q.iter().zip(r.mean_x()).map(|(x, m)| x - m).collect()
}

/// `A = M_xx + Λ`, `Λ = (ρ·n)·diag(s)`, rounded as `CentredFit` rounds it.
fn penalized(stats: &CentredStats, rho: f64, scale: &[f64]) -> DMatrix<f64> {
    let mut a = stats.m_xx();
    let rho_n = rho * stats.n();
    for j in 0..stats.p() {
        a[(j, j)] += rho_n * scale[j];
    }
    a
}

/// `κ₂(A)` and `‖A⁻¹‖₂` of a symmetric positive definite `A`, from its spectrum; `None` when
/// the computed spectrum is not positive (the bounds below then say nothing).
fn spectrum(a: &DMatrix<f64>) -> Option<(f64, f64)> {
    let ev = a.clone().symmetric_eigenvalues();
    let (lo, hi) = (ev.min(), ev.max());
    (lo > 0.0).then(|| (hi / lo, 1.0 / lo))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// On any window, at any penalty and block size, near the origin or 10⁶ away from it:
    ///
    /// - `ȳ` and `β` are the bits of `CentredFit` on the window's statistics, and `ȳ` is the
    ///   statistics' own mean outcome;
    /// - `ȳ_c + Σ_j β_cj x̃_j`, summed in the forecast's order (products rounded one by one,
    ///   left to right, `ȳ_c` last), is the forecast to the bit;
    /// - computed in another order (nalgebra's `ȳ + β x̃`, or the sum reversed with `ȳ_c`
    ///   first), it is the forecast within `2γ_{p+1}(|ȳ_c| + Σ_j |β_cj x̃_j|)`: two inner products of length
    ///   p + 1 over the same rounded terms, each within `γ_{p+1}` of the exact sum of its terms
    ///   (Higham, Thm 3.1). `x̃` is the same in both, one exact-rounded subtraction per
    ///   coordinate. The bound holds without underflow; `(p + 1)·2⁻¹⁰²²` allows for it.
    #[test]
    fn mean_y_plus_beta_x_tilde_is_the_forecast(
        d in data(5, 3, 1, 70), a in any::<Index>(), b in any::<Index>(),
        rho in prop_oneof![Just(0.0), 1e-6..1e-3, 1e-3..2.0],
        scale in vec(prop_oneof![Just(0.0), 0.1..2.0], 5), block_log2 in 0u32..=3,
        j in any::<Index>(), shift in prop_oneof![Just(0.0), Just(0.0), Just(1e6)],
    ) {
        let d = d.clone().shifted(j.index(d.p), shift);
        let (p, q, n) = (d.p, d.q, d.n());
        let scale = &scale[..p];
        let (s, e) = ordered(a.index(n + 1), b.index(n + 1));
        let m = d.matrix(block_log2);
        let r = match m.centred_query(&d.x_q, s, e, &cfg(rho, scale)) {
            Ok(r) => r,
            Err(OlsError::EmptyRange | OlsError::NotPositiveDefinite) => return Ok(()),
            Err(err) => return Err(TestCaseError::fail(format!("unexpected {err:?}"))),
        };
        prop_assert_eq!(r.beta().shape(), (q, p));
        prop_assert_eq!(r.mean_y().len(), q);

        // One solve: the readout's ȳ and β are the fit's, bit for bit.
        let stats = m.centred_range_stats(s, e).unwrap();
        let fit = CentredFit::new(&stats, &cfg(rho, scale)).unwrap();
        prop_assert!(same_bits(r.mean_y().as_slice(), fit.mean_y()));
        prop_assert!(same_bits(r.mean_y().as_slice(), stats.mean_y()));
        prop_assert!(same_bits(r.beta().as_slice(), fit.beta().as_slice()));

        // The forecast's own order: the bits.
        let xt = x_tilde(&r, &d.x_q);
        let beta = r.beta();
        let ordered_sum: Vec<f64> = (0..q)
            .map(|c| {
                let mut dot = beta[(c, 0)] * xt[0];
                for k in 1..p {
                    dot += beta[(c, k)] * xt[k];
                }
                r.mean_y()[c] + dot
            })
            .collect();
        prop_assert!(same_bits(&ordered_sum, r.forecast().as_slice()),
            "{ordered_sum:?} vs {:?}", r.forecast());

        // Other orders: within the inner-product bound. nalgebra's `gemv` happens to sum in the
        // same order (it gave the same bits in every measured case), so the reversed sum, with
        // ȳ first, is what exercises the bound: it differs in the last bits about once in four.
        let xt_v = DVector::from_column_slice(&xt);
        let gemv = r.mean_y() + beta * &xt_v;
        for c in 0..q {
            let reversed = (0..p).rev().fold(r.mean_y()[c], |acc, k| acc + beta[(c, k)] * xt[k]);
            let terms: f64 = (0..p).map(|k| (beta[(c, k)] * xt[k]).abs()).sum();
            let tol = 2.0 * gamma(p + 1) * (r.mean_y()[c].abs() + terms)
                + (p + 1) as f64 * f64::MIN_POSITIVE;
            for (name, other) in [("gemv", gemv[c]), ("reversed", reversed)] {
                let err = (other - r.forecast()[c]).abs();
                prop_assert!(err <= tol, "outcome {c}, {name}: |Δ| = {err:e} > {tol:e}");
            }
        }
    }

    /// `β` is the closed form `M_yx (M_xx + Λ)⁻¹`, with `A = M_xx + Λ` built from the window's
    /// `centred_range_stats` and solved by LU with partial pivoting (the library uses Cholesky).
    ///
    /// Tolerance, per outcome c, to first order in u: both solves are backward stable,
    /// `(A + ΔA) x̂ = b` with `‖ΔA‖₂ ≤ c_p·u·‖A‖₂` (Higham, Thm 10.4 for Cholesky, `c_p ≈ 3p²`;
    /// Thm 9.4 for LU, with a growth factor of at most `2^(p−1)`), so each is within
    /// `κ₂(A)·c_p·u·‖β_c‖₂` of the exact `β_c`. The test allows `2·(3p² + 3p²·2^(p−1))·u·κ₂(A)`
    /// times the larger of the two norms. A conservative first-order bound, not a proof.
    ///
    /// On the same window, primal against dual: `β_c x̃ = M_yx,c A⁻¹ x̃ = M_yx,c a` exactly, so
    /// the two computed sides differ by the errors of `β` and `a` through the dot products, plus
    /// the rounding of the dots themselves.
    #[test]
    fn beta_is_the_closed_form_and_the_dual(
        d in data(4, 3, 1, 70), a in any::<Index>(), b in any::<Index>(),
        rho in prop_oneof![Just(0.0), 1e-3..2.0], scale in vec(prop_oneof![Just(0.0), 0.1..2.0], 4),
        block_log2 in 0u32..=3, j in any::<Index>(), shift in prop_oneof![Just(0.0), Just(1e6)],
    ) {
        let d = d.clone().shifted(j.index(d.p), shift);
        let (p, q, n) = (d.p, d.q, d.n());
        let scale = &scale[..p];
        let (s, e) = ordered(a.index(n + 1), b.index(n + 1));
        let m = d.matrix(block_log2);
        let Ok(r) = m.centred_query(&d.x_q, s, e, &cfg(rho, scale)) else {
            return Ok(());
        };
        let stats = m.centred_range_stats(s, e).unwrap();
        let am = penalized(&stats, rho, scale);
        let Some((kappa, inv_norm)) = spectrum(&am) else {
            return Ok(());
        };
        // β_cᵀ solves A β_cᵀ = M_yx,cᵀ (A is symmetric): all q rows at once.
        let myx = stats.m_yx();
        let reference = am.clone().lu().solve(&myx.transpose())
            .ok_or_else(|| TestCaseError::fail("LU found an accepted A singular"))?
            .transpose();
        let pf = p as f64;
        let c_p = 3.0 * pf * pf * (1.0 + 2f64.powi(p as i32 - 1));
        let rel = 2.0 * c_p * U * kappa;
        let beta = r.beta();
        for c in 0..q {
            let (got, want) = (beta.row(c), reference.row(c));
            let tol = rel * got.norm().max(want.norm()) + f64::MIN_POSITIVE;
            let err = (got - want).norm();
            prop_assert!(err <= tol, "outcome {c}: ‖Δβ‖ = {err:e} > {tol:e} (κ = {kappa:e})");
        }

        // Primal against dual.
        let xt = DVector::from_column_slice(&x_tilde(&r, &d.x_q));
        let a_dir = r.a();
        let primal = beta * &xt;
        let dual = &myx * a_dir;
        for c in 0..q {
            let b_terms: f64 = (0..p).map(|k| (beta[(c, k)] * xt[k]).abs()).sum();
            let m_terms: f64 = (0..p).map(|k| (myx[(c, k)] * a_dir[k]).abs()).sum();
            let solves = c_p * U * kappa
                * (beta.row(c).norm() * xt.norm() + myx.row(c).norm() * a_dir.norm());
            let tol = solves + 2.0 * gamma(p) * (b_terms + m_terms) + f64::MIN_POSITIVE;
            let err = (primal[c] - dual[c]).abs();
            prop_assert!(err <= tol, "outcome {c}: |β x̃ − M_yx a| = {err:e} > {tol:e}");
        }
        // ‖a‖ ≤ ‖A⁻¹‖‖x̃‖ up to the solve's error: a sanity check on the spectrum used above.
        prop_assert!(a_dir.norm() <= inv_norm * xt.norm() * (1.0 + rel) + f64::MIN_POSITIVE);
    }

    /// At `ρ = 0` on a well-posed design (n ≥ 12 ≥ 3p uniform entries), `β` is the slope block
    /// of ordinary least squares with an intercept, `[b, β] = argmin ‖Y − [1, X] [b, β]ᵀ‖`,
    /// solved by an SVD of the raw design: no centring, no normal equations.
    ///
    /// Tolerance, per outcome c, first order in u. The library's route errs through the
    /// statistics, `|ΔM_jk| ≤ hε√(M_jj M_kk)` with `h = n + 3 + 3L` (`docs/DESIGN.md` §3.2),
    /// hence `‖ΔM_xx‖₂ ≤ p·h·ε·‖M_xx‖₂` and `‖ΔM_yx,c‖₂ ≤ √p·h·ε·√(M_yy,cc ‖M_xx‖₂)`, then
    /// through the Cholesky solve; the SVD route errs by `O((n + p)·u·(κ_D + κ_D²·‖r‖/(‖D‖‖β‖)))`
    /// (Higham, Thm 20.3). With `κ = κ₂(M_xx)` both are covered by
    /// `16·h·p·u·κ·(‖β_c‖₂ + √(M_yy,cc / λ_min(M_xx)))`, conservative constants, not a proof.
    #[test]
    fn at_rho_zero_beta_is_ols(d in data(4, 3, 12, 60), block_log2 in 0u32..=3) {
        let (p, q, n) = (d.p, d.q, d.n());
        let m = d.matrix(block_log2);
        let r = m.centred_query(&d.x_q, 0, n, &cfg(0.0, &vec![0.0; p]))
            .map_err(|e| TestCaseError::fail(format!("a well-posed design was refused: {e:?}")))?;
        let design = DMatrix::from_fn(n, p + 1, |i, k| if k == 0 { 1.0 } else { d.x(i)[k - 1] });
        let ys = DMatrix::from_fn(n, q, |i, c| d.y(i)[c]);
        let coef = design.svd(true, true).solve(&ys, 0.0)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
        let stats = m.centred_range_stats(0, n).unwrap();
        let Some((kappa, inv_norm)) = spectrum(&stats.m_xx()) else {
            return Err(TestCaseError::fail("M_xx of a well-posed design is not positive"));
        };
        let h = n as f64 + 3.0 + 3.0 * f64::from(stats.depth());
        let myy = stats.m_yy();
        for c in 0..q {
            let want = DVector::from_fn(p, |k, _| coef[(k + 1, c)]);
            let got = r.beta().row(c).transpose();
            let tol = 16.0 * h * p as f64 * U * kappa
                * (got.norm().max(want.norm()) + (myy[(c, c)] * inv_norm).sqrt());
            let err = (&got - &want).norm();
            prop_assert!(err <= tol, "outcome {c}: ‖Δβ‖ = {err:e} > {tol:e} (κ = {kappa:e})");
        }
        // The intercept the readout implies, ȳ − β x̄, is OLS's too.
        for c in 0..q {
            let b: f64 = r.mean_y()[c]
                - (0..p).map(|k| r.beta()[(c, k)] * r.mean_x()[k]).sum::<f64>();
            let tol = 16.0 * h * p as f64 * U * kappa
                * (1.0 + r.beta().row(c).norm() + (myy[(c, c)] * inv_norm).sqrt());
            prop_assert!((b - coef[(0, c)]).abs() <= tol, "intercept {c}: {b} vs {}", coef[(0, c)]);
        }
    }
}

/// A hand-sized case: y = 2·x₀ − x₁ + 3 exactly, on a design of small integers, so `β`, `ȳ` and
/// the forecast are known in closed form.
#[test]
fn an_exact_plane_gives_its_coefficients() {
    let xs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ];
    let mut m = Matrix::new(2, 1, 1).unwrap();
    for (i, x) in xs.iter().enumerate() {
        m.push(i as i64, x, &[2.0 * x[0] - x[1] + 3.0]).unwrap();
    }
    let r = m
        .centred_query(&[4.0, -2.0], 0, xs.len(), &cfg(0.0, &[0.0, 0.0]))
        .unwrap();
    let beta = r.beta();
    assert!((beta[(0, 0)] - 2.0).abs() < 1e-14, "{beta}");
    assert!((beta[(0, 1)] + 1.0).abs() < 1e-14, "{beta}");
    assert!((r.mean_y()[0] - 23.0 / 6.0).abs() < 1e-15, "{}", r.mean_y());
    assert!((r.forecast()[0] - 13.0).abs() < 1e-13, "{}", r.forecast());
}

/// A window of one entry under a penalty: `β = 0` exactly (`M_yx = 0`), and the forecast is `ȳ`.
#[test]
fn one_penalized_entry_has_zero_beta() {
    let mut m = Matrix::new(3, 2, 2).unwrap();
    m.push(0, &[0.3, -1.0, 2.0], &[0.25, -4.0]).unwrap();
    let r = m
        .centred_query(&[9.0, 9.0, 9.0], 0, 1, &cfg(0.5, &[1.0, 1.0, 1.0]))
        .unwrap();
    assert_eq!(r.beta(), &DMatrix::zeros(2, 3));
    assert_eq!(r.mean_y().as_slice(), &[0.25, -4.0]);
    assert_eq!(r.forecast().as_slice(), &[0.25, -4.0]);
}
