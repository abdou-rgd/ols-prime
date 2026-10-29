//! Centred sufficient statistics and the centred ridge solve (`docs/DESIGN.md` §3 and §7).
//!
//! A set of unit-weight entries `(xᵢ, yᵢ)` is summarized by
//!
//! ```text
//! n,   x̄ = Σx/n,   M_xx = Σ(x − x̄)(x − x̄)ᵀ,   ȳ = Σy/n,   M_yx = Σ(y − ȳ)(x − x̄)ᵀ,
//! M_yy = Σ(y − ȳ)(y − ȳ)ᵀ.
//! ```
//!
//! Two disjoint sets merge exactly with the pairwise update of Chan, Golub and LeVeque (1979):
//! with `δ = x̄_B − x̄_A` and `f = n_A n_B / n`,
//!
//! ```text
//! x̄ = x̄_A + δ·n_B/n,   M_xx = M_xx,A + M_xx,B + f·δδᵀ,
//! ```
//!
//! and likewise for the outcome blocks. Unlike the raw sums of [`crate::SuffStats`], no field
//! ever holds `Σxxᵀ`, whose cancellation against `n x̄ x̄ᵀ` costs digits when a feature's mean is
//! large against its spread: here the mean enters only through differences of means.
//!
//! The ridge with a free intercept and a penalty per coordinate is then solved in centred form
//! ([`CentredFit`]): `β = M_yx (M_xx + Λ)⁻¹` and `ŷ = ȳ + β (x_q − x̄)`, with
//! `Λ = ρ·n·diag(s)`.

use nalgebra::{DMatrix, DVector};

use crate::OlsError;
use crate::stats::{check_finite, check_len};

/// Centred sufficient statistics of a set of unit-weight entries (see the module docs).
///
/// Only the constructors of this module produce them: [`CentredStats::from_rows`], a merge, or
/// [`crate::Matrix::centred_range_stats`]. The fields are read-only because the refusal criterion
/// of [`CentredFit`] relies on how they were computed, which two counters record:
/// - `longest_scan`: the most entries ever summed sequentially behind any field;
/// - `depth`: the depth of the tree of merges behind the statistics (0 for a single scan).
///
/// **Exactness.** A scan is the corrected two-pass algorithm: the mean of the scanned entries,
/// then the centred sums and the correction `−(Σd)(Σd)ᵀ/n`, which removes the first-order effect
/// of the rounded mean exactly. The mean is then refined to `x̄ + Σd/n`. A rounded diagonal that
/// comes out a hair below zero (a constant column) is set to zero, its exact value's side.
#[derive(Debug, Clone, PartialEq)]
pub struct CentredStats {
    p: usize,
    q: usize,
    count: u64,
    longest_scan: u64,
    depth: u32,
    /// `[x̄ (p) | M_xx (p×p) | ȳ (q) | M_yx (q×p, row-major) | M_yy (q×q)]`. The symmetric blocks
    /// are exactly symmetric: only one triangle is computed, then mirrored.
    data: Vec<f64>,
}

impl CentredStats {
    /// The statistics of the empty set.
    pub fn zeros(p: usize, q: usize) -> Self {
        Self {
            p,
            q,
            count: 0,
            longest_scan: 0,
            depth: 0,
            data: vec![0.0; Self::len_for(p, q)],
        }
    }

    /// The statistics of unit-weight rows stored row-major: `xs` holds n × p values and `ys`
    /// n × q values, scanned in one corrected two-pass.
    pub fn from_rows(xs: &[f64], ys: &[f64], p: usize, q: usize) -> Result<Self, OlsError> {
        if p == 0 || q == 0 {
            return Err(OlsError::InvalidShape("p and q must be at least 1"));
        }
        let n = xs.len() / p;
        check_len(n * p, xs.len())?;
        check_len(n * q, ys.len())?;
        check_finite(xs)?;
        check_finite(ys)?;
        Ok(Self::scan(p, q, xs, ys))
    }

    fn len_for(p: usize, q: usize) -> usize {
        p + p * p + q + q * p + q * q
    }

    /// Dimension of the states.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Dimension of the outcomes.
    pub fn q(&self) -> usize {
        self.q
    }

    /// Number of entries.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Number of entries, as a float.
    pub fn n(&self) -> f64 {
        self.count as f64
    }

    /// The most entries summed sequentially behind any field: the `h` of [`CentredFit`]'s
    /// criterion.
    pub fn longest_scan(&self) -> u64 {
        self.longest_scan
    }

    /// The depth of the tree of merges behind the statistics: 0 for a single scan.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// `x̄` (length p). Zeros for the empty set.
    pub fn mean_x(&self) -> &[f64] {
        &self.data[..self.p]
    }

    /// `M_xx = Σ(x − x̄)(x − x̄)ᵀ` (p × p, symmetric).
    pub fn m_xx(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.p, self.p, self.mxx())
    }

    /// `ȳ` (length q). Zeros for the empty set.
    pub fn mean_y(&self) -> &[f64] {
        let o = self.p + self.p * self.p;
        &self.data[o..o + self.q]
    }

    /// `M_yx = Σ(y − ȳ)(x − x̄)ᵀ` (q × p).
    pub fn m_yx(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.q, self.p, self.myx())
    }

    /// `M_yy = Σ(y − ȳ)(y − ȳ)ᵀ` (q × q, symmetric).
    pub fn m_yy(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.q, self.q, self.myy())
    }

    /// Adds the statistics of a set of entries disjoint from this one (Chan, Golub and
    /// LeVeque's pairwise update).
    pub fn merge(&mut self, other: &Self) -> Result<(), OlsError> {
        check_len(self.p, other.p)?;
        check_len(self.q, other.q)?;
        self.merge_unchecked(other);
        Ok(())
    }

    /// The statistics of the union of two disjoint sets.
    pub fn merged(&self, other: &Self) -> Result<Self, OlsError> {
        let mut out = self.clone();
        out.merge(other)?;
        Ok(out)
    }

    pub(crate) fn mxx(&self) -> &[f64] {
        let p = self.p;
        &self.data[p..p + p * p]
    }

    pub(crate) fn myx(&self) -> &[f64] {
        let o = self.p + self.p * self.p + self.q;
        &self.data[o..o + self.q * self.p]
    }

    pub(crate) fn myy(&self) -> &[f64] {
        let o = self.p + self.p * self.p + self.q + self.q * self.p;
        &self.data[o..o + self.q * self.q]
    }

    /// The corrected two-pass scan of rows validated upstream.
    pub(crate) fn scan(p: usize, q: usize, xs: &[f64], ys: &[f64]) -> Self {
        let mut s = Self::zeros(p, q);
        let n = xs.len() / p;
        if n == 0 {
            return s;
        }
        let nf = n as f64;
        let mut mx = vec![0.0; p];
        let mut my = vec![0.0; q];
        for (x, y) in xs.chunks_exact(p).zip(ys.chunks_exact(q)) {
            for (m, v) in mx.iter_mut().zip(x) {
                *m += v;
            }
            for (m, v) in my.iter_mut().zip(y) {
                *m += v;
            }
        }
        for m in mx.iter_mut().chain(my.iter_mut()) {
            *m /= nf;
        }
        let (mut sd, mut se) = (vec![0.0; p], vec![0.0; q]);
        let (mut d, mut e) = (vec![0.0; p], vec![0.0; q]);
        let (mut mxx, mut myx, mut myy) = (vec![0.0; p * p], vec![0.0; q * p], vec![0.0; q * q]);
        for (x, y) in xs.chunks_exact(p).zip(ys.chunks_exact(q)) {
            for j in 0..p {
                d[j] = x[j] - mx[j];
                sd[j] += d[j];
            }
            for c in 0..q {
                e[c] = y[c] - my[c];
                se[c] += e[c];
            }
            for i in 0..p {
                let row = &mut mxx[i * p..i * p + i + 1];
                for (acc, dj) in row.iter_mut().zip(&d[..=i]) {
                    *acc += d[i] * dj;
                }
            }
            for c in 0..q {
                for (acc, dj) in myx[c * p..(c + 1) * p].iter_mut().zip(&d) {
                    *acc += e[c] * dj;
                }
                for k in 0..=c {
                    myy[c * q + k] += e[c] * e[k];
                }
            }
        }
        // The correction removes the rounded mean's contribution, (Σd)(Σd)ᵀ/n, which is
        // exactly zero for the exact mean.
        for i in 0..p {
            for j in 0..=i {
                mxx[i * p + j] -= sd[i] * sd[j] / nf;
            }
            mxx[i * p + i] = mxx[i * p + i].max(0.0);
        }
        for c in 0..q {
            for j in 0..p {
                myx[c * p + j] -= se[c] * sd[j] / nf;
            }
            for k in 0..=c {
                myy[c * q + k] -= se[c] * se[k] / nf;
            }
            myy[c * q + c] = myy[c * q + c].max(0.0);
        }
        mirror(&mut mxx, p);
        mirror(&mut myy, q);
        for j in 0..p {
            mx[j] += sd[j] / nf;
        }
        for c in 0..q {
            my[c] += se[c] / nf;
        }
        s.count = n as u64;
        s.longest_scan = n as u64;
        s.data.clear();
        s.data.extend_from_slice(&mx);
        s.data.extend_from_slice(&mxx);
        s.data.extend_from_slice(&my);
        s.data.extend_from_slice(&myx);
        s.data.extend_from_slice(&myy);
        s
    }

    /// [`CentredStats::merge`] for statistics of matching dimensions.
    pub(crate) fn merge_unchecked(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            // The empty set adds no rounding: keep the other side's bookkeeping too.
            self.clone_from(other);
            return;
        }
        let (p, q) = (self.p, self.q);
        let (na, nb) = (self.n(), other.n());
        let n = na + nb;
        let f = na * nb / n;
        let wb = nb / n;
        let (a, b) = (&mut self.data, &other.data);
        let dx: Vec<f64> = (0..p).map(|j| b[j] - a[j]).collect();
        let oy = p + p * p;
        let dy: Vec<f64> = (0..q).map(|c| b[oy + c] - a[oy + c]).collect();
        for j in 0..p {
            a[j] += dx[j] * wb;
        }
        for c in 0..q {
            a[oy + c] += dy[c] * wb;
        }
        // Each block gets `(M_A + M_B) + (f·δᵢ)·δⱼ`, on one triangle for the symmetric ones.
        let add = |a: &mut [f64], b: &[f64], left: &[f64], right: &[f64], lower: bool| {
            let cols = right.len();
            for (i, &li) in left.iter().enumerate() {
                let fi = f * li;
                let width = if lower { i + 1 } else { cols };
                let (ra, rb) = (
                    &mut a[i * cols..i * cols + width],
                    &b[i * cols..i * cols + width],
                );
                for ((va, vb), rj) in ra.iter_mut().zip(rb).zip(right) {
                    *va = (*va + vb) + fi * rj;
                }
            }
        };
        let (oxx, oyx) = (p, oy + q);
        let oyy = oyx + q * p;
        add(&mut a[oxx..oy], &b[oxx..oy], &dx, &dx, true);
        mirror(&mut a[oxx..oy], p);
        add(&mut a[oyx..oyy], &b[oyx..oyy], &dy, &dx, false);
        add(&mut a[oyy..], &b[oyy..], &dy, &dy, true);
        mirror(&mut a[oyy..], q);
        self.count += other.count;
        self.longest_scan = self.longest_scan.max(other.longest_scan);
        self.depth = self.depth.max(other.depth) + 1;
    }
}

/// Copies the lower triangle of a row-major `k × k` matrix onto its upper triangle.
fn mirror(m: &mut [f64], k: usize) {
    for i in 0..k {
        for j in 0..i {
            m[j * k + i] = m[i * k + j];
        }
    }
}

/// The penalty of the centred ridge: `Λ = ρ·n·diag(s)`, with `n` the entry count of the
/// statistics it is applied to.
///
/// With `s` the per-feature variances fitted inside the fold, this is ridge with penalty `ρ·n`
/// on the features standardized by those variances: the penalty grows with the window like the
/// data term, so `ρ` keeps one meaning across windows of different lengths. The intercept is
/// free: it is solved apart, as `ȳ − β x̄`.
#[derive(Debug, Clone, PartialEq)]
pub struct CentredRidge {
    rho: f64,
    scale: Vec<f64>,
}

impl CentredRidge {
    /// `ρ ≥ 0` finite, and every `sⱼ ≥ 0` finite. A zero scale leaves coordinate j unpenalized.
    pub fn new(rho: f64, scale: &[f64]) -> Result<Self, OlsError> {
        check_rho(rho)?;
        check_scale(scale)?;
        Ok(Self {
            rho,
            scale: scale.to_vec(),
        })
    }

    /// The scalar `ρ`.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// The per-coordinate scale `s`.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }
}

pub(crate) fn check_rho(rho: f64) -> Result<(), OlsError> {
    if rho.is_finite() && rho >= 0.0 {
        Ok(())
    } else {
        Err(OlsError::InvalidLambda(rho))
    }
}

pub(crate) fn check_scale(scale: &[f64]) -> Result<(), OlsError> {
    match scale.iter().position(|s| !(s.is_finite() && *s >= 0.0)) {
        None => Ok(()),
        Some(index) => Err(OlsError::InvalidScale {
            index,
            value: scale[index],
        }),
    }
}

/// The centred ridge of one set of statistics at one penalty, factored once and read for any
/// number of queries.
///
/// ```text
/// A = M_xx + Λ,   Λ = ρ·n·diag(s),   β = M_yx A⁻¹,   ŷ(x_q) = ȳ + β (x_q − x̄).
/// ```
///
/// This is ridge with a free intercept: minimizing `Σ‖yᵢ − b − β xᵢ‖² + Σⱼ Λⱼⱼ βⱼ²` over the
/// intercept b gives `b = ȳ − β x̄`, and what remains is the ridge of the centred data.
///
/// # Refusal
///
/// The solve refuses with [`OlsError::NotPositiveDefinite`] when `A` lies within rounding of a
/// singular matrix, so that a singular window never returns numbers. As for
/// [`crate::RidgeConfig`], the test is on the spectrum after diagonal scaling,
///
/// ```text
/// S = D^(-1/2) A D^(-1/2),  D = diag(A);   accept only if  λ_min(S) > τ,
/// τ = 4p(h + p + 2)ε · (1 + 2(L + 1)·r),
/// ```
///
/// with `ε = 2⁻⁵²`, `h = longest_scan + 3 + 3L`, `L` the merge depth of the statistics and
/// `r = maxⱼ √((M_jj + n x̄ⱼ²) / D_jj)`, the largest raw root-mean-square of a coordinate against
/// its scaled diagonal. Derivation, to first order in ε:
///
/// - **Scans.** The corrected two-pass of b entries computes `M_jk` with an error of at most
///   about `(b + 3)ε Σ|dⱼ||dₖ| ≤ (b + 3)ε √(M_jj M_kk)` (Cauchy–Schwarz, unit weights). The
///   rounded mean contributes nothing at first order: the correction term cancels it exactly.
/// - **Merges.** Each merge adds two roundings to `M_A + M_B + fδδᵀ`, each bounded by
///   `√(M_jj M_kk)` of the merged node, since `M_A`, `M_B` and `fδδᵀ` are positive
///   semidefinite parts of it. Nodes at one depth of the merge tree are disjoint, so by
///   Cauchy–Schwarz their errors sum to at most `3ε√(M_jj M_kk)` per depth. Hence the term
///   `hε√(M_jj M_kk)` with `h = b + 3 + 3L`.
/// - **Rounded means.** The means of the operands err by up to `hε` times their root-mean-square
///   `uⱼ = √(Σxⱼ²/n)`: the error is relative to the values, not to their spread. Through `fδδᵀ`
///   it costs `f|δₖ||Δδⱼ| ≤ 2hε √(M_kk Σxⱼ²)` per merge (with `f δₖ² ≤ M_kk` and
///   `f/n_B ≤ 1`), and again the disjoint nodes of one depth add up to that bound. Hence a term
///   `2Lhε(√(M_kk Σxⱼ²) + √(M_jj Σxₖ²))`, which after scaling is `2Lhε(rⱼ + rₖ)`, with
///   `Σxⱼ² = M_jj + n x̄ⱼ²`. This is the only place where a large mean against the spread
///   enters, and it enters linearly in `r`, where the raw Gram pays `r²`.
/// - **Spectrum.** With `E` the error of `S`, `‖E‖₂ ≤ ‖E‖_F ≤ phε + 4Lhε·p·r`, plus the
///   rounding of the scaling and of the backward-stable eigensolver, `O(pε)` as in
///   [`crate::RidgeConfig`]. `τ` covers this with the factor 4 and one extra depth.
///
/// By Weyl's inequality a design that is singular in exact arithmetic then has a computed
/// `λ_min(S) ≤ τ`, and is refused whatever the rounding does. This is a first-order bound with
/// conservative constants, not a machine-checked proof; the property tests check it against an
/// independent SVD on exactly singular designs, shifted far from the origin included.
///
/// - **What `r` does.** A standardized feature has `r ≈ 1`. A feature whose mean is 10⁶ times
///   its spread has `r ≈ 10⁶`: at 2.6·10⁵ entries in blocks of 64 (`h ≈ 200`, `L ≈ 38`), `τ`
///   is then about 10⁻⁴, so such a window is still accepted unless its scaled spectrum is nearly
///   flat. The raw path refuses it outright (`λ_min ≈ 1/(2r²) = 5·10⁻¹³`).
/// - **What the penalty does.** `Λⱼⱼ > 0` lifts `D_jj`, which lowers `rⱼ`, and lifts the
///   spectrum of `S`: with every `sⱼ > 0` and `ρ > 0`, a window of one entry is accepted
///   (`β = 0`, `ŷ = ȳ`).
/// - **A constant column** has `M_jj` at rounding level. Unpenalized, its `rⱼ` is huge or its
///   diagonal zero, and it is refused: the coordinate is not identified.
#[derive(Debug, Clone, PartialEq)]
pub struct CentredFit {
    p: usize,
    q: usize,
    n: f64,
    mean_x: Vec<f64>,
    mean_y: Vec<f64>,
    /// Lower Cholesky factor of `A`, row-major p × p.
    chol: Vec<f64>,
    /// β, row-major q × p.
    beta: Vec<f64>,
    rss: Vec<f64>,
}

impl CentredFit {
    /// Factors `A = M_xx + ρ·n·diag(s)` and solves for β.
    ///
    /// # Errors
    ///
    /// [`OlsError::EmptyRange`] for no entry, [`OlsError::NotPositiveDefinite`] for a singular
    /// `A` (see the type's docs), [`OlsError::NonFinite`] for an overflow, and
    /// [`OlsError::Dimension`] if `s` does not have length p.
    pub fn new(stats: &CentredStats, cfg: &CentredRidge) -> Result<Self, OlsError> {
        check_len(stats.p, cfg.scale.len())?;
        Self::fit(stats, cfg.rho, &cfg.scale)
    }

    /// [`CentredFit::new`] for a penalty validated upstream.
    pub(crate) fn fit(stats: &CentredStats, rho: f64, scale: &[f64]) -> Result<Self, OlsError> {
        let (p, q) = (stats.p, stats.q);
        if stats.count == 0 {
            return Err(OlsError::EmptyRange);
        }
        let n = stats.n();
        let mxx = stats.mxx();
        let mut a = mxx.to_vec();
        let rho_n = rho * n;
        let lambda: Vec<f64> = scale.iter().map(|s| rho_n * s).collect();
        for j in 0..p {
            a[j * p + j] += lambda[j];
        }
        check_finite(&a)?;
        let diag: Vec<f64> = (0..p).map(|j| a[j * p + j]).collect();
        // A zero diagonal is a coordinate that neither the data nor the penalty identifies.
        if diag.iter().any(|&d| d <= 0.0) {
            return Err(OlsError::NotPositiveDefinite);
        }
        let mean_x = stats.mean_x();
        let mut r_max: f64 = 0.0;
        for j in 0..p {
            let raw = mxx[j * p + j] + n * mean_x[j] * mean_x[j];
            r_max = r_max.max((raw / diag[j]).sqrt());
        }
        if !r_max.is_finite() {
            return Err(OlsError::NonFinite);
        }
        let sc: Vec<f64> = diag.iter().map(|d| d.sqrt().recip()).collect();
        let scaled = DMatrix::from_fn(p, p, |i, j| a[i * p + j] * sc[i] * sc[j]);
        if !scaled.iter().all(|v| v.is_finite()) {
            return Err(OlsError::NotPositiveDefinite);
        }
        let lambda_min = scaled.symmetric_eigenvalues().min();
        if lambda_min.is_nan() || lambda_min <= threshold(stats, r_max) {
            return Err(OlsError::NotPositiveDefinite);
        }
        let chol = cholesky(&a, p).ok_or(OlsError::NotPositiveDefinite)?;
        let myx = stats.myx();
        let mut beta = vec![0.0; q * p];
        let mut z = vec![0.0; p];
        for c in 0..q {
            let row = &myx[c * p..(c + 1) * p];
            solve_lower(&chol, p, row, &mut z);
            solve_upper(&chol, p, &z, &mut beta[c * p..(c + 1) * p]);
        }
        check_finite(&beta)?;
        // RSS = M_yy − 2βM_yxᵀ + βM_xxβᵀ, and βM_xxβᵀ = βAβᵀ − βΛβᵀ = βM_yxᵀ − βΛβᵀ.
        let myy = stats.myy();
        let rss: Vec<f64> = (0..q)
            .map(|c| {
                let b = &beta[c * p..(c + 1) * p];
                let fit: f64 = b
                    .iter()
                    .zip(&myx[c * p..(c + 1) * p])
                    .map(|(b, m)| b * m)
                    .sum();
                let pen: f64 = b.iter().zip(&lambda).map(|(b, l)| l * b * b).sum();
                myy[c * q + c] - fit - pen
            })
            .collect();
        check_finite(&rss)?;
        Ok(Self {
            p,
            q,
            n,
            mean_x: mean_x.to_vec(),
            mean_y: stats.mean_y().to_vec(),
            chol,
            beta,
            rss: rss.into_iter().map(|r| r.max(0.0)).collect(),
        })
    }

    /// Number of entries behind the fit.
    pub fn n(&self) -> f64 {
        self.n
    }

    /// `x̄` of the entries (length p).
    pub fn mean_x(&self) -> &[f64] {
        &self.mean_x
    }

    /// `ȳ` of the entries (length q).
    pub fn mean_y(&self) -> &[f64] {
        &self.mean_y
    }

    /// `β = M_yx A⁻¹` (q × p).
    pub fn beta(&self) -> DMatrix<f64> {
        DMatrix::from_row_slice(self.q, self.p, &self.beta)
    }

    /// The in-sample residual sum of squares at this penalty, per outcome:
    /// `Σᵢ (yᵢ − ŷ(xᵢ))² = M_yy,cc − β_c·M_yx,c − β_c Λ β_cᵀ`.
    ///
    /// It is read off the statistics, and cancels when the fit is close: its absolute error is
    /// of order `(h + p)ε·M_yy,cc`, so it is meaningful while `RSS ≫ (h + p)ε·M_yy,cc`, i.e. for
    /// any `R² < 1 − 10⁻¹⁰` or so. A value rounded below zero is returned as zero.
    pub fn rss(&self) -> &[f64] {
        &self.rss
    }

    /// The forecast `ŷ = ȳ + β (x_q − x̄)` (length q).
    pub fn forecast(&self, x_q: &[f64]) -> Result<DVector<f64>, OlsError> {
        check_len(self.p, x_q.len())?;
        check_finite(x_q)?;
        let mut out = vec![0.0; self.q];
        let mut xt = vec![0.0; self.p];
        self.forecast_into(x_q, &mut xt, &mut out);
        check_finite(&out)?;
        Ok(DVector::from_vec(out))
    }

    /// The leverage `x̃ᵀ A⁻¹ x̃` of the query, with `x̃ = x_q − x̄`.
    pub fn leverage(&self, x_q: &[f64]) -> Result<f64, OlsError> {
        check_len(self.p, x_q.len())?;
        check_finite(x_q)?;
        let mut xt = vec![0.0; self.p];
        let mut z = vec![0.0; self.p];
        let h = self.leverage_into(x_q, &mut xt, &mut z);
        check_finite(&[h])?;
        Ok(h)
    }

    /// The dual direction `a = A⁻¹ x̃`: entry i of the window weighs `wᵢ = 1/n + (xᵢ − x̄)ᵀ a`,
    /// and `Σᵢ wᵢ yᵢ = ȳ + M_yx a = ŷ`.
    pub fn direction(&self, x_q: &[f64]) -> Result<DVector<f64>, OlsError> {
        check_len(self.p, x_q.len())?;
        check_finite(x_q)?;
        let xt: Vec<f64> = x_q.iter().zip(&self.mean_x).map(|(x, m)| x - m).collect();
        let mut z = vec![0.0; self.p];
        let mut a = vec![0.0; self.p];
        solve_lower(&self.chol, self.p, &xt, &mut z);
        solve_upper(&self.chol, self.p, &z, &mut a);
        check_finite(&a)?;
        Ok(DVector::from_vec(a))
    }

    /// `out = ȳ + β (x_q − x̄)`, with `xt` as scratch for `x̃`. The sums run over j in order.
    pub(crate) fn forecast_into(&self, x_q: &[f64], xt: &mut [f64], out: &mut [f64]) {
        for ((t, x), m) in xt.iter_mut().zip(x_q).zip(&self.mean_x) {
            *t = x - m;
        }
        for (c, o) in out.iter_mut().enumerate() {
            let b = &self.beta[c * self.p..(c + 1) * self.p];
            let dot: f64 = b.iter().zip(xt.iter()).map(|(b, t)| b * t).sum();
            *o = self.mean_y[c] + dot;
        }
    }

    /// `‖L⁻¹ x̃‖²`, with `xt` and `z` as scratch.
    pub(crate) fn leverage_into(&self, x_q: &[f64], xt: &mut [f64], z: &mut [f64]) -> f64 {
        for ((t, x), m) in xt.iter_mut().zip(x_q).zip(&self.mean_x) {
            *t = x - m;
        }
        solve_lower(&self.chol, self.p, xt, z);
        z.iter().map(|v| v * v).sum()
    }
}

/// The `τ` of [`CentredFit`]'s docs.
fn threshold(stats: &CentredStats, r_max: f64) -> f64 {
    let p = stats.p as f64;
    let depth = f64::from(stats.depth);
    let h = stats.longest_scan as f64 + 3.0 + 3.0 * depth;
    4.0 * p * (h + p + 2.0) * f64::EPSILON * (1.0 + 2.0 * (depth + 1.0) * r_max)
}

/// The lower Cholesky factor of a row-major symmetric `p × p` matrix, or `None` at a pivot that
/// is not positive.
fn cholesky(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut d = a[j * p + j];
        for k in 0..j {
            d -= l[j * p + k] * l[j * p + k];
        }
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let ljj = d.sqrt();
        l[j * p + j] = ljj;
        for i in j + 1..p {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = s / ljj;
        }
    }
    Some(l)
}

/// Solves `L z = b` for a lower-triangular row-major `L`.
fn solve_lower(l: &[f64], p: usize, b: &[f64], z: &mut [f64]) {
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * p + k] * z[k];
        }
        z[i] = s / l[i * p + i];
    }
}

/// Solves `Lᵀ x = z` for a lower-triangular row-major `L`.
fn solve_upper(l: &[f64], p: usize, z: &[f64], x: &mut [f64]) {
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in i + 1..p {
            s -= l[k * p + i] * x[k];
        }
        x[i] = s / l[i * p + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scan_of_small_integers_is_exact() {
        // x = 0..4 and y = 2x + 1: x̄ = 1.5, M_xx = 5, M_yx = 10, M_yy = 20, all exact.
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [1.0, 3.0, 5.0, 7.0];
        let s = CentredStats::from_rows(&xs, &ys, 1, 1).unwrap();
        assert_eq!(
            (s.count(), s.mean_x(), s.mean_y()),
            (4, &[1.5][..], &[4.0][..])
        );
        assert_eq!(s.mxx(), &[5.0]);
        assert_eq!(s.myx(), &[10.0]);
        assert_eq!(s.myy(), &[20.0]);
        let left = CentredStats::from_rows(&xs[..1], &ys[..1], 1, 1).unwrap();
        let right = CentredStats::from_rows(&xs[1..], &ys[1..], 1, 1).unwrap();
        let m = left.merged(&right).unwrap();
        assert_eq!(
            (m.mxx(), m.myx(), m.myy()),
            (&[5.0][..], &[10.0][..], &[20.0][..])
        );
        assert_eq!((m.depth(), m.longest_scan()), (1, 3));
    }

    #[test]
    fn the_empty_set_merges_as_the_identity() {
        let s = CentredStats::from_rows(&[1.0, 2.0, 4.0, 3.0], &[1.0, 0.0], 2, 1).unwrap();
        let mut e = CentredStats::zeros(2, 1);
        e.merge(&s).unwrap();
        assert_eq!(e, s);
        assert_eq!(s.merged(&CentredStats::zeros(2, 1)).unwrap(), s);
        assert!(s.merged(&CentredStats::zeros(3, 1)).is_err());
    }

    #[test]
    fn a_constant_column_is_refused_unpenalized_and_inert_penalized() {
        let xs: Vec<f64> = (0..50).flat_map(|i| [0.1, f64::from(i).sin()]).collect();
        let ys: Vec<f64> = (0..50).map(|i| f64::from(i).cos()).collect();
        let s = CentredStats::from_rows(&xs, &ys, 2, 1).unwrap();
        let free = CentredRidge::new(0.0, &[0.0, 0.0]).unwrap();
        assert_eq!(
            CentredFit::new(&s, &free),
            Err(OlsError::NotPositiveDefinite)
        );
        let pen = CentredRidge::new(1e-3, &[1.0, 1.0]).unwrap();
        let fit = CentredFit::new(&s, &pen).unwrap();
        assert!(fit.beta()[(0, 0)].abs() < 1e-6, "{}", fit.beta());
    }

    #[test]
    fn penalties_and_scales_are_validated() {
        assert!(CentredRidge::new(0.0, &[0.0]).is_ok());
        for bad in [-1e-300, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                CentredRidge::new(bad, &[1.0]),
                Err(OlsError::InvalidLambda(_))
            ));
            assert!(matches!(
                CentredRidge::new(1.0, &[1.0, bad]),
                Err(OlsError::InvalidScale { index: 1, .. })
            ));
        }
        let s = CentredStats::from_rows(&[1.0, 2.0], &[0.0, 1.0], 1, 1).unwrap();
        let cfg = CentredRidge::new(1.0, &[1.0, 1.0]).unwrap();
        assert!(matches!(
            CentredFit::new(&s, &cfg),
            Err(OlsError::Dimension { .. })
        ));
        let empty = CentredStats::zeros(1, 1);
        let cfg = CentredRidge::new(1.0, &[1.0]).unwrap();
        assert_eq!(CentredFit::new(&empty, &cfg), Err(OlsError::EmptyRange));
    }

    #[test]
    fn one_entry_with_a_penalty_forecasts_its_outcome() {
        let s = CentredStats::from_rows(&[3.0, -1.0], &[0.25, 2.0], 2, 2).unwrap();
        let fit = CentredFit::new(&s, &CentredRidge::new(0.5, &[1.0, 2.0]).unwrap()).unwrap();
        assert_eq!(fit.forecast(&[10.0, 7.0]).unwrap().as_slice(), &[0.25, 2.0]);
        assert_eq!(fit.rss(), &[0.0, 0.0]);
        assert_eq!(fit.leverage(&[3.0, -1.0]).unwrap(), 0.0);
    }
}
