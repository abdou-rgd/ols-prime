//! The centred form of Matrix (`docs/DESIGN.md` §3, iteration 3), as property tests:
//! Chan–Golub–LeVeque merges against a two-pass reference, the centred ridge against the raw
//! ridge with a free intercept, shift invariance, the soundness of the refusal criterion, the
//! window boundaries, and the batch readouts against one-by-one reads, bit for bit, whatever the
//! thread count and whatever is pushed later.

// The references are written index by index, as the formulas are.
#![allow(clippy::needless_range_loop)]

use nalgebra::DMatrix;
use ols_core::{
    CentredFit, CentredRidge, CentredStats, Matrix, OlsError, ReadStatus, RidgeConfig, RidgeGrid,
    SuffStats, Window, solve_query,
};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;

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

/// The textbook two-pass over rows `[s, e)`, independent of the library: mean, then the sums of
/// centred products. Returns `(x̄, M_xx, ȳ, M_yx, M_yy)`, row-major.
type Moments = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn two_pass(d: &Data, s: usize, e: usize) -> Moments {
    let (p, q) = (d.p, d.q);
    let n = (e - s) as f64;
    let mut mx = vec![0.0; p];
    let mut my = vec![0.0; q];
    for i in s..e {
        for j in 0..p {
            mx[j] += d.x(i)[j] / n;
        }
        for c in 0..q {
            my[c] += d.y(i)[c] / n;
        }
    }
    let (mut mxx, mut myx, mut myy) = (vec![0.0; p * p], vec![0.0; q * p], vec![0.0; q * q]);
    for i in s..e {
        let dx: Vec<f64> = (0..p).map(|j| d.x(i)[j] - mx[j]).collect();
        let dy: Vec<f64> = (0..q).map(|c| d.y(i)[c] - my[c]).collect();
        for a in 0..p {
            for b in 0..p {
                mxx[a * p + b] += dx[a] * dx[b];
            }
        }
        for c in 0..q {
            for b in 0..p {
                myx[c * p + b] += dy[c] * dx[b];
            }
            for k in 0..q {
                myy[c * q + k] += dy[c] * dy[k];
            }
        }
    }
    (mx, mxx, my, myx, myy)
}

fn row_major(m: &DMatrix<f64>) -> Vec<f64> {
    m.transpose().as_slice().to_vec()
}

/// `|a − b| ≤ tol·(1 + scale)` element-wise.
fn within(a: &[f64], b: &[f64], scale: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .zip(scale)
            .all(|((x, y), s)| (x - y).abs() <= tol * (1.0 + s))
}

/// `√(M_aa M_bb)` for a square row-major `M`: the scale of entry `(a, b)`.
fn cs_scale(m: &[f64], k: usize) -> Vec<f64> {
    (0..k * k)
        .map(|i| (m[(i / k) * k + i / k] * m[(i % k) * k + i % k]).sqrt())
        .collect()
}

/// `√(M_yy,cc M_xx,bb)` for the q × p cross block.
fn cross_scale(myy: &[f64], mxx: &[f64], q: usize, p: usize) -> Vec<f64> {
    (0..q * p)
        .map(|i| (myy[(i / p) * q + i / p] * mxx[(i % p) * p + i % p]).sqrt())
        .collect()
}

fn assert_moments(got: &CentredStats, want: &Moments, tol: f64) -> Result<(), TestCaseError> {
    let (mx, mxx, my, myx, myy) = want;
    let (p, q) = (got.p(), got.q());
    prop_assert!(
        within(got.mean_x(), mx, &vec![0.0; p], tol),
        "x̄ {:?} vs {mx:?}",
        got.mean_x()
    );
    prop_assert!(within(got.mean_y(), my, &vec![0.0; q], tol));
    prop_assert!(
        within(&row_major(&got.m_xx()), mxx, &cs_scale(mxx, p), tol),
        "M_xx"
    );
    prop_assert!(
        within(&row_major(&got.m_yy()), myy, &cs_scale(myy, q), tol),
        "M_yy"
    );
    prop_assert!(
        within(
            &row_major(&got.m_yx()),
            myx,
            &cross_scale(myy, mxx, q, p),
            tol
        ),
        "M_yx"
    );
    let sym = |m: DMatrix<f64>| m == m.transpose();
    prop_assert!(sym(got.m_xx()) && sym(got.m_yy()), "exactly symmetric");
    Ok(())
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Two batches agree bit for bit, NaNs included.
fn same_batch(a: &ols_core::RidgeBatch, b: &ols_core::RidgeBatch) -> bool {
    let opt = |x: &Option<Vec<f64>>, y: &Option<Vec<f64>>| match (x, y) {
        (Some(x), Some(y)) => same_bits(x, y),
        (None, None) => true,
        _ => false,
    };
    (a.queries, a.windows, a.rhos, a.q) == (b.queries, b.windows, b.rhos, b.q)
        && same_bits(&a.forecast, &b.forecast)
        && a.start == b.start
        && a.end == b.end
        && a.status == b.status
        && opt(&a.leverage, &b.leverage)
        && opt(&a.rss, &b.rss)
}

fn cfg(rho: f64, scale: &[f64]) -> CentredRidge {
    CentredRidge::new(rho, scale).unwrap()
}

/// `λ_min` of the scaled `M_xx + Λ`, from an SVD of the centred data stacked over `√Λ`: an
/// independent reference for the refusal criterion, like `scaled_lambda_min` of `identities.rs`.
fn scaled_lambda_min(d: &Data, lambda: &[f64]) -> f64 {
    let (p, n) = (d.p, d.n());
    let mean: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| d.x(i)[j]).sum::<f64>() / n as f64)
        .collect();
    let mut a = DMatrix::zeros(n + p, p);
    for i in 0..n {
        for j in 0..p {
            a[(i, j)] = d.x(i)[j] - mean[j];
        }
    }
    for j in 0..p {
        a[(n + j, j)] = lambda[j].sqrt();
    }
    for j in 0..p {
        let norm = a.column(j).norm();
        // A constant column centres to rounding noise, whose direction means nothing: it is a
        // zero column. Unit-scale data puts every other column many decades above this.
        if norm <= 1e-9 {
            return 0.0;
        }
        a.column_mut(j).unscale_mut(norm);
    }
    a.singular_values().min().powi(2)
}

/// The library's `τ` of `CentredFit`, restated from its documentation.
fn tau(stats: &CentredStats, lambda: &[f64]) -> f64 {
    let (p, n) = (stats.p(), stats.n());
    let mxx = stats.m_xx();
    let r = (0..p)
        .map(|j| {
            let x = stats.mean_x()[j];
            ((mxx[(j, j)] + n * x * x) / (mxx[(j, j)] + lambda[j])).sqrt()
        })
        .fold(0.0, f64::max);
    let depth = f64::from(stats.depth());
    let h = stats.longest_scan() as f64 + 3.0 + 3.0 * depth;
    4.0 * p as f64 * (h + p as f64 + 2.0) * f64::EPSILON * (1.0 + 2.0 * (depth + 1.0) * r)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Merging the statistics of consecutive pieces, in any split, equals the two-pass of their
    /// union; so does the tree's centred read of any range, at every block size.
    #[test]
    fn merges_equal_the_two_pass_reference(
        d in data(4, 3, 1, 60), cuts in vec(any::<Index>(), 0..6),
        shift in prop_oneof![Just(0.0), Just(1e3), Just(-7.5)], block_log2 in 0u32..=3,
        a in any::<Index>(), b in any::<Index>(),
    ) {
        let d = d.shifted(0, shift);
        let n = d.n();
        let mut bounds: Vec<usize> = cuts.iter().map(|c| c.index(n + 1)).collect();
        bounds.extend([0, n]);
        bounds.sort_unstable();
        let mut acc = CentredStats::zeros(d.p, d.q);
        for w in bounds.windows(2) {
            let piece = CentredStats::from_rows(
                &d.xs[w[0] * d.p..w[1] * d.p], &d.ys[w[0] * d.q..w[1] * d.q], d.p, d.q,
            ).unwrap();
            acc.merge(&piece).unwrap();
        }
        let tol = 1e-12 * (1.0 + shift.abs());
        prop_assert_eq!(acc.count(), n as u64);
        assert_moments(&acc, &two_pass(&d, 0, n), tol)?;
        let (s, e) = ordered(a.index(n + 1), b.index(n + 1));
        let tree = d.matrix(block_log2).centred_range_stats(s, e).unwrap();
        prop_assert_eq!(tree.count(), (e - s) as u64);
        if e > s {
            assert_moments(&tree, &two_pass(&d, s, e), tol)?;
        }
    }

    /// With every penalty and scale, the centred solve is the raw ridge with a free intercept:
    /// `Λ = ρ·n·diag(s)` on x is `ρ·n·I` on `x/√s`, beside an unpenalized constant column.
    #[test]
    fn the_centred_solve_is_ridge_with_a_free_intercept(
        d in data(4, 3, 12, 60), rho in prop_oneof![Just(0.0), 1e-3..1.0],
        scale in vec(0.1..2.0, 4), block_log2 in 0u32..=3,
    ) {
        let (p, q, n) = (d.p, d.q, d.n());
        let scale = &scale[..p];
        let m = d.matrix(block_log2);
        let r = m.centred_query(&d.x_q, 0, n, &cfg(rho, scale));
        let mut raw_x = Vec::with_capacity(n * (p + 1));
        for i in 0..n {
            raw_x.push(1.0);
            raw_x.extend(d.x(i).iter().zip(scale).map(|(x, s)| x / s.sqrt()));
        }
        let raw = SuffStats::from_rows(&raw_x, &d.ys, p + 1, q).unwrap();
        let mut raw_q = vec![1.0];
        raw_q.extend(d.x_q.iter().zip(scale).map(|(x, s)| x / s.sqrt()));
        let raw_cfg = RidgeConfig::with_intercept(rho * n as f64).unwrap();
        let (Ok(r), Ok(a)) = (r, solve_query(&raw, &raw_cfg, &raw_q)) else {
            // Uniform features on [−1, 1] with n ≥ 12 ≥ 3p are far from singular.
            return Err(TestCaseError::fail("a well-conditioned design was refused"));
        };
        let want: Vec<f64> = (0..q)
            .map(|c| (0..=p).map(|j| raw.cross[(c, j)] * a[j]).sum())
            .collect();
        prop_assert!(within(r.forecast().as_slice(), &want, &vec![1.0; q], 1e-9),
            "{:?} vs {want:?}", r.forecast());
        // The dual weights sum to one, reproduce the forecast, and partition it.
        let w = m.centred_weights(&r).unwrap();
        let total: f64 = w.iter().sum();
        prop_assert!((total - 1.0).abs() < 1e-9, "Σw = {total}");
        let dual: Vec<f64> = (0..q).map(|c| (0..n).map(|i| w[i] * d.y(i)[c]).sum()).collect();
        prop_assert!(within(&dual, r.forecast().as_slice(), &vec![1.0; q], 1e-9));
        let cut = n / 3;
        let parts: Vec<f64> = m.centred_contribution(&r, 0, cut).unwrap().iter()
            .zip(m.centred_contribution(&r, cut, n).unwrap().iter())
            .map(|(a, b)| a + b).collect();
        prop_assert!(within(&parts, r.forecast().as_slice(), &vec![1.0; q], 1e-9));
        let older: Vec<f64> = (0..q).map(|c| (0..cut).map(|i| w[i] * d.y(i)[c]).sum()).collect();
        prop_assert!(within(m.centred_contribution(&r, 0, cut).unwrap().as_slice(), &older,
            &vec![1.0; q], 1e-9));
    }

    /// The leverage and the residual sums match a direct computation over the entries.
    #[test]
    fn leverage_and_rss_match_a_direct_pass(
        d in data(4, 2, 10, 50), rho in 1e-3..1.0, scale in vec(0.1..2.0, 4),
    ) {
        let (p, q, n) = (d.p, d.q, d.n());
        let scale = &scale[..p];
        let m = d.matrix(2);
        let r = m.centred_query(&d.x_q, 0, n, &cfg(rho, scale)).unwrap();
        let (mx, mxx, ..) = two_pass(&d, 0, n);
        let mut a = DMatrix::from_row_slice(p, p, &mxx);
        for j in 0..p {
            a[(j, j)] += rho * n as f64 * scale[j];
        }
        let xt = nalgebra::DVector::from_iterator(p, d.x_q.iter().zip(&mx).map(|(x, m)| x - m));
        let h = xt.dot(&a.clone().cholesky().unwrap().solve(&xt));
        prop_assert!((r.leverage() - h).abs() <= 1e-10 * (1.0 + h), "{} vs {h}", r.leverage());
        let fit = CentredFit::new(&m.centred_range_stats(0, n).unwrap(), &cfg(rho, scale)).unwrap();
        for c in 0..q {
            let rss: f64 = (0..n).map(|i| (d.y(i)[c] - fit.forecast(d.x(i)).unwrap()[c]).powi(2)).sum();
            prop_assert!((r.rss()[c] - rss).abs() <= 1e-10 * (1.0 + rss), "{} vs {rss}", r.rss()[c]);
        }
    }

    /// Adding 10⁶ to a feature moves the centred forecasts by rounding only. The bound is the
    /// rounding of the shifted data, `ε·10⁶` on unit-scale features, times a factor 10⁴ for the
    /// conditioning and the sums.
    #[test]
    fn a_shift_moves_the_centred_forecast_by_rounding_only(
        d in data(4, 2, 12, 60), j in any::<Index>(), rho in prop_oneof![Just(0.0), 1e-3..1.0],
        block_log2 in 0u32..=3,
    ) {
        let n = d.n();
        let j = j.index(d.p);
        let scale = vec![1.0; d.p];
        let base = d.matrix(block_log2).centred_query(&d.x_q, 0, n, &cfg(rho, &scale)).unwrap();
        let moved = d.clone().shifted(j, 1e6);
        let r = moved.matrix(block_log2).centred_query(&moved.x_q, 0, n, &cfg(rho, &scale)).unwrap();
        let tol = 1e4 * 1e6 * f64::EPSILON;
        prop_assert!(within(r.forecast().as_slice(), base.forecast().as_slice(),
            &vec![1.0; d.q], tol), "{:?} vs {:?}", r.forecast(), base.forecast());
    }

    /// Designs singular in exact arithmetic after centring (duplicated rows, a feature affine in
    /// another, a constant feature, fewer than p + 1 entries) are refused at every block size,
    /// near the origin or 10⁶ away from it; and on unshifted data the verdict agrees with an
    /// independent SVD.
    #[test]
    fn singular_centred_designs_are_always_refused(
        d in data(4, 2, 2, 40).prop_filter("p ≥ 2", |d| d.p >= 2), kind in 0usize..4,
        i in any::<Index>(), j in any::<Index>(),
        c in prop_oneof![Just(2.0), Just(-0.7), Just(1e3), Just(1e-3)],
        shift in prop_oneof![Just(0.0), Just(1e6)], block_log2 in 0u32..=3,
    ) {
        let (p, n) = (d.p, d.n());
        let mut d = d;
        let a = i.index(p);
        let b = (a + 1 + j.index(p - 1)) % p;
        match kind {
            0 => {
                let rank = 1 + i.index(p);
                for r in rank..n {
                    for k in 0..p {
                        d.xs[r * p + k] = d.xs[(r % rank) * p + k];
                    }
                }
            }
            1 => {
                for r in 0..n {
                    d.xs[r * p + b] = c * d.xs[r * p + a] + 0.25;
                }
            }
            2 => {
                for r in 0..n {
                    d.xs[r * p + b] = 0.1;
                }
            }
            _ => {
                let keep = (1 + i.index(p)).min(n);
                d.xs.truncate(keep * p);
                d.ys.truncate(keep * d.q);
            }
        }
        let d = d.shifted(a, shift);
        let m = d.matrix(block_log2);
        let free = vec![0.0; p];
        prop_assert_eq!(
            m.centred_query(&d.x_q, 0, d.n(), &cfg(0.0, &free)).err(),
            Some(OlsError::NotPositiveDefinite)
        );
        // A penalty on every coordinate but the constant one does not identify it either.
        let mut partial = vec![1.0; p];
        partial[b] = 0.0;
        if kind == 2 {
            prop_assert_eq!(
                m.centred_query(&d.x_q, 0, d.n(), &cfg(0.1, &partial)).err(),
                Some(OlsError::NotPositiveDefinite)
            );
        }
        if shift == 0.0 {
            prop_assert!(scaled_lambda_min(&d, &free) <= 1e-12);
        }
    }

    /// On unshifted data the verdict agrees with an independent SVD: an accepted design has
    /// `λ_min(S) ≥ τ/4`, a refused one `λ_min(S) ≤ 4τ`, as `assert_criterion` of
    /// `identities.rs` checks for the raw path. Exactly singular designs come from duplicated
    /// rows, at the singular edge `ρ = 0` and near it.
    #[test]
    fn the_centred_verdict_agrees_with_an_svd(
        d in data(4, 1, 1, 30), rho in prop_oneof![Just(0.0), Just(1e-14), 0.01..1.0],
        scale in vec(prop_oneof![Just(0.0), 0.1..2.0], 4), duplicate in any::<bool>(),
        block_log2 in 0u32..=3,
    ) {
        let (p, n) = (d.p, d.n());
        let mut d = d;
        if duplicate {
            for r in p.min(n)..n {
                for k in 0..p {
                    d.xs[r * p + k] = d.xs[(r % p) * p + k];
                }
            }
        }
        let scale = &scale[..p];
        let stats = d.matrix(block_log2).centred_range_stats(0, n).unwrap();
        let verdict = CentredFit::new(&stats, &cfg(rho, scale));
        let lambda: Vec<f64> = scale.iter().map(|s| rho * n as f64 * s).collect();
        let lmin = scaled_lambda_min(&d, &lambda);
        let t = tau(&stats, &lambda);
        match verdict {
            Ok(_) => prop_assert!(lmin >= t / 4.0, "accepted λ_min = {lmin:e}, τ = {t:e}"),
            Err(OlsError::NotPositiveDefinite) => {
                prop_assert!(lmin <= 4.0 * t, "refused λ_min = {lmin:e}, τ = {t:e}");
            }
            Err(e) => return Err(TestCaseError::fail(format!("unexpected {e:?}"))),
        }
    }

    /// One-hot outcomes give class forecasts that sum to one: `Σ_c M_yx,c = 0`.
    #[test]
    fn one_hot_centred_forecasts_sum_to_one(
        d in data(4, 1, 8, 40), classes in vec(0usize..3, 40), rho in 1e-3..1.0,
    ) {
        let mut m = Matrix::new(d.p, 3, 1).unwrap();
        for i in 0..d.n() {
            let mut y = [0.0; 3];
            y[classes[i]] = 1.0;
            m.push(i as i64, d.x(i), &y).unwrap();
        }
        let r = m.centred_query(&d.x_q, 0, d.n(), &cfg(rho, &vec![1.0; d.p])).unwrap();
        prop_assert!((r.forecast().sum() - 1.0).abs() < 1e-12, "{:?}", r.forecast());
    }

    /// `window_range` holds exactly the entries with `as_of − span < time ≤ as_of`, ties and
    /// boundaries included.
    #[test]
    fn windows_are_left_open_and_right_closed(
        steps in vec(0i64..3, 0..40), as_of in -2i64..90, span in 1i64..30, all in any::<bool>(),
    ) {
        let mut m = Matrix::new(1, 1, 1).unwrap();
        let mut t = 0;
        for s in &steps {
            t += s;
            m.push(t, &[0.0], &[0.0]).unwrap();
        }
        let window = if all { Window::ALL } else { Window::last(span).unwrap() };
        let (s, e) = m.window_range(as_of, window);
        let inside = |time: i64| time <= as_of && (all || time > as_of - span);
        let expected: Vec<usize> = (0..m.len()).filter(|&i| inside(m.time(i))).collect();
        prop_assert_eq!((s..e).collect::<Vec<_>>(), expected);
    }

    /// A batch cell is the one-by-one read of its window at its penalty, bit for bit: forecast,
    /// leverage, residual sums and status. The result does not depend on the thread count, nor
    /// on entries pushed after every decision time.
    #[test]
    fn batch_reads_are_one_by_one_reads_bit_for_bit(
        d in data(4, 3, 1, 80), steps in vec(0i64..3, 80), later in data(4, 3, 1, 30),
        queries in vec((-3i64..170, vec(-1.0..1.0f64, 4)), 0..40), sorted in any::<bool>(),
        block_log2 in 0u32..=3, threads in 2usize..6,
    ) {
        let (p, q, n) = (d.p, d.q, d.n());
        let mut times = Vec::with_capacity(n);
        let mut t = 0i64;
        for s in &steps[..n] {
            t += s;
            times.push(t);
        }
        let mut m = Matrix::new(p, q, block_log2).unwrap();
        for i in 0..n {
            m.push(times[i], d.x(i), d.y(i)).unwrap();
        }
        let mut queries = queries;
        if sorted {
            queries.sort_by_key(|q| q.0);
        }
        let as_of: Vec<i64> = queries.iter().map(|q| q.0.min(t)).collect();
        let x_q: Vec<f64> = queries.iter().flat_map(|q| q.1[..p].to_vec()).collect();
        let windows = [Window::last(4).unwrap(), Window::last(40).unwrap(), Window::ALL];
        let rhos = [0.0, 1e-3, 0.5];
        let scale: Vec<f64> = (0..p).map(|j| 0.5 + j as f64).collect();
        let grid = RidgeGrid::new(&windows, &rhos, &scale).unwrap().with_leverage(true).with_rss(true);
        let batch = m.ridge_batch(&x_q, &as_of, &grid, 1).unwrap();
        let mut statuses = [0usize; 5];
        for (i, &d_i) in as_of.iter().enumerate() {
            for (wi, &w) in windows.iter().enumerate() {
                let (s, e) = m.window_range(d_i, w);
                prop_assert_eq!((batch.start[i * 3 + wi], batch.end[i * 3 + wi]), (s, e));
                for (ri, &rho) in rhos.iter().enumerate() {
                    let cell = (i * 3 + wi) * 3 + ri;
                    let status = batch.status[cell];
                    statuses[status.code() as usize] += 1;
                    let f = &batch.forecast[cell * q..(cell + 1) * q];
                    match m.centred_query(&x_q[i * p..(i + 1) * p], s, e, &cfg(rho, &scale)) {
                        Ok(r) => {
                            prop_assert_eq!(status, ReadStatus::Ok);
                            prop_assert!(same_bits(f, r.forecast().as_slice()));
                            prop_assert_eq!(batch.leverage.as_ref().unwrap()[cell].to_bits(),
                                r.leverage().to_bits());
                            prop_assert!(same_bits(&batch.rss.as_ref().unwrap()[cell * q..(cell + 1) * q],
                                r.rss().as_slice()));
                        }
                        Err(e) => {
                            let want = match e {
                                OlsError::EmptyRange => ReadStatus::Empty,
                                OlsError::NotPositiveDefinite => ReadStatus::Singular,
                                _ => ReadStatus::NonFinite,
                            };
                            prop_assert_eq!(status, want);
                            prop_assert!(f.iter().all(|v| v.is_nan()));
                            prop_assert!(batch.leverage.as_ref().unwrap()[cell].is_nan());
                        }
                    }
                }
            }
        }
        // Any thread count, the same bits.
        let parallel = m.ridge_batch(&x_q, &as_of, &grid, threads).unwrap();
        prop_assert!(same_batch(&parallel, &batch));
        // Entries pushed after every decision time change nothing, to the bit.
        for i in 0..later.n() {
            let x: Vec<f64> = (0..p).map(|j| later.xs[(i * later.p + j) % later.xs.len()]).collect();
            let y: Vec<f64> = (0..q).map(|c| later.ys[(i * later.q + c) % later.ys.len()]).collect();
            m.push(t + 1 + i as i64, &x, &y).unwrap();
        }
        let after = m.ridge_batch(&x_q, &as_of, &grid, threads).unwrap();
        prop_assert!(same_batch(&after, &batch));
    }
}

/// The raw path on the same shifted design: beside an intercept, a feature whose mean is 10⁷
/// times its spread makes the raw Gram singular to working precision, and the raw solve refuses;
/// the centred solve answers, and its forecast is the unshifted one up to rounding.
#[test]
fn a_shift_the_raw_path_refuses_is_read_by_the_centred_path() {
    let n = 200;
    let rows: Vec<(f64, f64, f64)> = (0..n)
        .map(|i| {
            let t = f64::from(i);
            (
                (t * 0.37).sin(),
                (t * 0.91).cos(),
                0.3 * (t * 0.37).sin() - 0.2,
            )
        })
        .collect();
    let read = |shift: f64| {
        let mut m = Matrix::new(2, 1, 3).unwrap();
        let mut raw_x = Vec::new();
        for (i, &(a, b, y)) in rows.iter().enumerate() {
            m.push(i as i64, &[a + shift, b], &[y]).unwrap();
            raw_x.extend([1.0, a + shift, b]);
        }
        let ys: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let raw = SuffStats::from_rows(&raw_x, &ys, 3, 1).unwrap();
        let x_q = [0.4 + shift, -0.1];
        let centred = m
            .centred_query(&x_q, 0, n as usize, &cfg(0.0, &[1.0, 1.0]))
            .map(|r| r.forecast()[0]);
        let raw = solve_query(
            &raw,
            &RidgeConfig::with_intercept(0.0).unwrap(),
            &[1.0, x_q[0], x_q[1]],
        )
        .map(|a| (0..3).map(|j| raw_cross(&raw, j) * a[j]).sum::<f64>());
        (centred, raw)
    };
    fn raw_cross(s: &SuffStats, j: usize) -> f64 {
        s.cross[(0, j)]
    }
    let (base, raw_base) = read(0.0);
    let base = base.unwrap();
    assert!((raw_base.unwrap() - base).abs() < 1e-12);
    // The outcome is exactly linear in the first feature: ŷ = 0.3·0.4 − 0.2.
    assert!((base - (0.3 * 0.4 - 0.2)).abs() < 1e-12, "{base}");
    let (centred, raw) = read(1e7);
    assert_eq!(raw, Err(OlsError::NotPositiveDefinite));
    let centred = centred.unwrap();
    assert!((centred - base).abs() < 1e-7, "{centred} vs {base}");
}

/// The status codes of a batch: an empty window, a singular one, an overflow, and numbers.
#[test]
fn every_status_has_its_cause() {
    let mut m = Matrix::new(1, 1, 1).unwrap();
    for (t, x) in [(10, 0.5), (20, 0.5), (30, 1e160), (40, -1e160), (50, 1.0)] {
        m.push(t, &[x], &[1.0]).unwrap();
    }
    let windows = [Window::last(5).unwrap(), Window::last(15).unwrap()];
    let grid = RidgeGrid::new(&windows, &[0.0], &[1.0])
        .unwrap()
        .with_leverage(true);
    // At 5, nothing is known. At 25, (20, 25] is empty and (10, 25] holds one entry: singular
    // unpenalized. At 40, (35, 40] holds one entry, and (25, 40] the two 1e160 entries, whose
    // squares overflow M_xx. At 55, (50, 55] is empty and (40, 55] holds one entry.
    let b = m
        .ridge_batch(&[0.0; 4], &[5, 25, 40, 55], &grid, 0)
        .unwrap();
    use ReadStatus::*;
    assert_eq!(
        b.status,
        [
            Empty, Empty, Empty, Singular, Singular, NonFinite, Empty, Singular
        ]
    );
    assert!(b.forecast.iter().all(|v| v.is_nan()));
    assert!(b.leverage.unwrap().iter().all(|v| v.is_nan()));
    assert_eq!(
        [Ok, Empty, Singular, NonFinite, TooFew].map(ReadStatus::code),
        [0, 1, 2, 3, 4]
    );
    let grid = RidgeGrid::new(&windows, &[0.1], &[1.0]).unwrap();
    let b = m.ridge_batch(&[0.0], &[55], &grid, 0).unwrap();
    assert_eq!(b.status, [Empty, Ok], "a penalty identifies a lone entry");
    assert_eq!(b.forecast[1], 1.0);
}

#[test]
fn bad_batch_inputs_are_errors_not_statuses() {
    let m = Matrix::new(2, 1, 1).unwrap();
    let w = [Window::ALL];
    assert!(RidgeGrid::new(&[], &[1.0], &[1.0, 1.0]).is_err());
    assert!(RidgeGrid::new(&w, &[], &[1.0, 1.0]).is_err());
    assert!(matches!(
        RidgeGrid::new(&w, &[-1.0], &[1.0]),
        Err(OlsError::InvalidLambda(_))
    ));
    assert!(matches!(
        RidgeGrid::new(&w, &[1.0], &[1.0, f64::NAN]),
        Err(OlsError::InvalidScale { index: 1, .. })
    ));
    assert_eq!(Window::last(0), Err(OlsError::InvalidWindow(0)));
    let grid = RidgeGrid::new(&w, &[1.0], &[1.0]).unwrap();
    assert!(matches!(
        m.ridge_batch(&[0.0, 0.0], &[0], &grid, 1),
        Err(OlsError::Dimension { .. })
    ));
    let grid = RidgeGrid::new(&w, &[1.0], &[1.0, 1.0]).unwrap();
    assert!(matches!(
        m.ridge_batch(&[0.0], &[0], &grid, 1),
        Err(OlsError::Dimension { .. })
    ));
    assert_eq!(
        m.ridge_batch(&[0.0, f64::NAN], &[0], &grid, 1).err(),
        Some(OlsError::NonFinite)
    );
    let empty = m.ridge_batch(&[], &[], &grid, 0).unwrap();
    assert!(empty.forecast.is_empty() && empty.status.is_empty());
}

#[test]
fn centred_readouts_belong_to_their_memory() {
    let mut a = Matrix::new(1, 1, 1).unwrap();
    for i in 0..6 {
        a.push(i, &[f64::from(i as i32)], &[1.0]).unwrap();
    }
    let b = a.clone();
    let r = a.centred_query(&[2.0], 0, 6, &cfg(0.1, &[1.0])).unwrap();
    assert_eq!(b.centred_weights(&r), Err(OlsError::ForeignReadout));
    assert_eq!(
        b.centred_contribution(&r, 0, 2),
        Err(OlsError::ForeignReadout)
    );
    assert!(matches!(
        a.centred_contribution(&r, 0, 7),
        Err(OlsError::OutOfRange { .. })
    ));
    assert_eq!(a.centred_contribution(&r, 3, 3).unwrap().as_slice(), &[0.0]);
}
