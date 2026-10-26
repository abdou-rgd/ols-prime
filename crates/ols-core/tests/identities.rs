//! The identities of `docs/DESIGN.md` §1–3, as property tests.
//!
//! **Penalties.** Every ridge identity draws `λ` from {0, 10⁻¹², [0.05, 10]}: the singular edge,
//! a penalty too small to rescue a singular design, and the well-posed range. At the first two,
//! many designs are singular or nearly so. Each case must then either be rejected with
//! `NotPositiveDefinite`, or satisfy the identity within the tolerance below. A rejection is
//! itself checked against an independent SVD ([`assert_criterion`]), so neither a silently wrong
//! answer nor a spurious refusal passes.
//!
//! **Tolerance.** An identity `lhs = rhs` is checked as `|lhs − rhs| ≤ TOL·(1 + mag)`, where
//! `mag` is the magnitude of the terms before cancellation, e.g. `Σᵢ |yᵢ|·Σₖ |xᵢₖ aₖ|` for a
//! dual sum `Σᵢ wᵢ yᵢ`. This is where conditioning enters: `‖a‖` grows like
//! `‖x_q‖ / λ_min(G + Λ)`, so a near-singular design makes `mag` large, and with it the
//! absolute error the identity may carry. Over 2·10⁵ random cases with `κ(S)` up to 10¹³, the
//! worst observed `|lhs − rhs| / (1 + mag)` was 7·10⁻¹⁶; `TOL = 10⁻¹²` leaves three decades.

use nalgebra::DMatrix;
use ols_core::{
    Matrix, OlsError, RidgeConfig, Rls, SuffStats, ols_attention, residual_sum_squares, ridge_fit,
    solve_query,
};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;

/// See the module docs.
const TOL: f64 = 1e-12;

/// Row-major entries: n states in Rᵖ with outcomes in Rᵠ, plus one query state.
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

    /// Sets coordinate 0 of every state, the query included, to the constant 1.
    fn with_intercept(mut self) -> Self {
        for row in self.xs.chunks_exact_mut(self.p) {
            row[0] = 1.0;
        }
        self.x_q[0] = 1.0;
        self
    }

    fn matrix(&self, block_log2: u32) -> Matrix {
        let mut m = Matrix::new(self.p, self.q, block_log2).unwrap();
        for i in 0..self.n() {
            m.push(i as i64, self.x(i), self.y(i)).unwrap();
        }
        m
    }

    /// Statistics of entries [start, end) from a direct pass.
    fn stats(&self, start: usize, end: usize) -> SuffStats {
        SuffStats::from_rows(
            &self.xs[start * self.p..end * self.p],
            &self.ys[start * self.q..end * self.q],
            self.p,
            self.q,
        )
        .unwrap()
    }

    /// `Σᵢ |yᵢⱼ|·Σₖ |xᵢₖ aₖ|` over entries [start, end), per outcome: the magnitude of the dual
    /// sums `Σᵢ wᵢ yᵢⱼ` before cancellation.
    fn dual_magnitude(&self, start: usize, end: usize, a: &[f64]) -> Vec<f64> {
        let mut mag = vec![0.0; self.q];
        for i in start..end {
            let w = abs_dot(self.x(i), a);
            for (m, y) in mag.iter_mut().zip(self.y(i)) {
                *m += w * y.abs();
            }
        }
        mag
    }

    /// `Σᵢ Σₖ |xᵢₖ aₖ|` over entries [start, end): the magnitude of `Σᵢ wᵢ`.
    fn weight_magnitude(&self, start: usize, end: usize, a: &[f64]) -> f64 {
        (start..end).map(|i| abs_dot(self.x(i), a)).sum()
    }
}

fn data(max_p: usize, max_q: usize, max_n: usize) -> impl Strategy<Value = Data> {
    (1..=max_p, 1..=max_q, 1..=max_n).prop_flat_map(|(p, q, n)| {
        (
            vec(-1.0..1.0, n * p),
            vec(-1.0..1.0, n * q),
            vec(-1.0..1.0, p),
        )
            .prop_map(move |(xs, ys, x_q)| Data { p, q, xs, ys, x_q })
    })
}

/// The singular edge, a near-singular penalty, and the well-posed range.
fn lambdas() -> impl Strategy<Value = f64> {
    prop_oneof![1 => Just(0.0), 1 => Just(1e-12), 3 => 0.05..10.0]
}

/// Element-wise `|a − b| ≤ tol · (1 + max(|a|, |b|))`.
fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())))
}

/// Element-wise `|a − b| ≤ TOL · (1 + mag)`: an identity up to the rounding of its terms.
fn holds(a: &[f64], b: &[f64], mag: &[f64]) -> bool {
    a.len() == b.len()
        && a.len() == mag.len()
        && a.iter()
            .zip(b)
            .zip(mag)
            .all(|((x, y), m)| (x - y).abs() <= TOL * (1.0 + m))
}

fn stats_close(a: &SuffStats, b: &SuffStats, tol: f64) -> bool {
    a.count == b.count
        && close(&[a.n], &[b.n], tol)
        && close(a.sum_x.as_slice(), b.sum_x.as_slice(), tol)
        && close(a.gram.as_slice(), b.gram.as_slice(), tol)
        && close(a.cross.as_slice(), b.cross.as_slice(), tol)
        && close(a.sum_y.as_slice(), b.sum_y.as_slice(), tol)
        && close(a.sum_yy.as_slice(), b.sum_yy.as_slice(), tol)
}

fn config(lambda: f64, intercept: bool) -> RidgeConfig {
    if intercept {
        RidgeConfig::with_intercept(lambda)
    } else {
        RidgeConfig::new(lambda)
    }
    .unwrap()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn abs_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum()
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// The library's threshold `τ = 4p(N + p + 2)ε` on `λ_min(D^(-1/2)(G + Λ)D^(-1/2))`, restated
/// here from its documentation.
fn tau(p: usize, n: usize) -> f64 {
    4.0 * p as f64 * (n + p + 2) as f64 * f64::EPSILON
}

/// `λ_min(D^(-1/2)(XᵀX + Λ)D^(-1/2))` computed without forming the Gram matrix: it is
/// `σ_min(A D^(-1/2))²` for the stacked design `A = [X; √Λ]`, whose columns have squared norms
/// `D`. The SVD of the data is accurate to `ε·‖A D^(-1/2)‖` in `σ`, far below `τ` in `σ²`, so
/// this is an independent and sharper reference than the library's eigenvalues of a rounded `G`.
fn scaled_lambda_min(xs: &[f64], p: usize, lambda: f64, intercept: bool) -> f64 {
    let n = xs.len() / p;
    let penalized: Vec<usize> = (usize::from(intercept)..p).collect();
    let rows = n + penalized.len();
    let mut a = DMatrix::zeros(rows, p);
    for i in 0..n {
        for k in 0..p {
            a[(i, k)] = xs[i * p + k];
        }
    }
    for (r, &k) in penalized.iter().enumerate() {
        a[(n + r, k)] = lambda.sqrt();
    }
    for k in 0..p {
        let norm = a.column(k).norm();
        if norm == 0.0 {
            return 0.0;
        }
        a.column_mut(k).unscale_mut(norm);
    }
    if rows < p {
        return 0.0;
    }
    a.singular_values().min().powi(2)
}

/// The library's verdict on a design agrees with the independent SVD: a rejected design has
/// `λ_min(S) ≤ 4τ`, an accepted one `λ_min(S) ≥ τ/4`. The library computes `λ_min` from a
/// rounded Gram, which moves it by at most about `τ/4` (its docs), hence the factor 4 of
/// slack each way. A penalty in the well-posed range is never refused.
fn assert_criterion(
    xs: &[f64],
    p: usize,
    lambda: f64,
    intercept: bool,
    accepted: bool,
) -> Result<(), TestCaseError> {
    let n = xs.len() / p;
    let lambda_min = scaled_lambda_min(xs, p, lambda, intercept);
    let t = tau(p, n);
    if accepted {
        prop_assert!(
            lambda_min >= t / 4.0,
            "accepted a singular design: λ_min(S) = {lambda_min:e}, τ = {t:e}"
        );
    } else {
        prop_assert!(
            lambda_min <= 4.0 * t,
            "refused a regular design: λ_min(S) = {lambda_min:e}, τ = {t:e}"
        );
        prop_assert!(
            lambda < 0.05 || (intercept && n == 0),
            "refused λ = {lambda}"
        );
    }
    Ok(())
}

/// A refusal must be `NotPositiveDefinite`; anything else is a bug.
fn accepted<T: std::fmt::Debug>(r: &Result<T, OlsError>) -> Result<bool, TestCaseError> {
    match r {
        Ok(_) => Ok(true),
        Err(OlsError::NotPositiveDefinite) => Ok(false),
        Err(e) => Err(TestCaseError::fail(format!("unexpected error {e:?}"))),
    }
}

/// A design that is singular in exact arithmetic, built from `d` in one of five ways.
fn degenerate(mut d: Data, kind: usize, i: Index, j: Index, c: f64) -> Data {
    let (p, n) = (d.p, d.n());
    match kind {
        // Duplicates: every row is a copy of one of the first `rank < p` rows.
        0 => {
            let rank = 1 + i.index(p - 1);
            for r in rank..n {
                let src = r % rank;
                for k in 0..p {
                    d.xs[r * p + k] = d.xs[src * p + k];
                }
            }
        }
        // Collinear columns: column `b = c · column a`, rounded.
        1 => {
            let a = i.index(p);
            let b = (a + 1 + j.index(p - 1)) % p;
            for r in 0..n {
                d.xs[r * p + b] = c * d.xs[r * p + a];
            }
        }
        // Fewer entries than coordinates.
        2 => {
            let keep = 1 + i.index(p - 1);
            d.xs.truncate(keep.min(n) * p);
            d.ys.truncate(keep.min(n) * d.q);
        }
        // A column that is the sum of two others, rounded (needs p ≥ 3, else a zero column).
        3 if p >= 3 => {
            let b = i.index(p);
            for r in 0..n {
                d.xs[r * p + b] = d.xs[r * p + (b + 1) % p] + d.xs[r * p + (b + 2) % p];
            }
        }
        // A column that no entry touches.
        _ => {
            let b = i.index(p);
            for r in 0..n {
                d.xs[r * p + b] = 0.0;
            }
        }
    }
    d
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Primal `β x_q` equals dual `Σᵢ wᵢ yᵢ` with `wᵢ = xᵢᵀ a`.
    #[test]
    fn dual_equals_primal(d in data(4, 3, 40), lambda in lambdas(), intercept in any::<bool>()) {
        let d = if intercept { d.with_intercept() } else { d };
        let cfg = config(lambda, intercept);
        let stats = d.stats(0, d.n());
        let (fit, a) = (ridge_fit(&stats, &cfg), solve_query(&stats, &cfg, &d.x_q));
        let ok = accepted(&fit)?;
        prop_assert_eq!(ok, accepted(&a)?, "one factor, one verdict");
        assert_criterion(&d.xs, d.p, lambda, intercept, ok)?;
        let (Ok(fit), Ok(a)) = (fit, a) else { return Ok(()) };
        let primal = fit.predict(&d.x_q).unwrap();
        let mut dual = vec![0.0; d.q];
        for i in 0..d.n() {
            let w = dot(d.x(i), a.as_slice());
            for (acc, y) in dual.iter_mut().zip(d.y(i)) {
                *acc += w * y;
            }
        }
        let mut mag = d.dual_magnitude(0, d.n(), a.as_slice());
        for (j, m) in mag.iter_mut().enumerate() {
            let row: Vec<f64> = fit.beta.row(j).iter().copied().collect();
            *m = m.max(abs_dot(&row, &d.x_q));
        }
        prop_assert!(holds(primal.as_slice(), &dual, &mag), "{primal:?} vs {dual:?}, mag {mag:?}");
    }

    /// With a free intercept, the weights of a read sum to one.
    #[test]
    fn weights_sum_to_one_with_a_free_intercept(
        d in data(4, 2, 40), lambda in lambdas(), block_log2 in 0u32..=3,
    ) {
        let d = d.with_intercept();
        let m = d.matrix(block_log2);
        let r = m.query(&d.x_q, 0, d.n(), &config(lambda, true));
        assert_criterion(&d.xs, d.p, lambda, true, accepted(&r)?)?;
        let Ok(r) = r else { return Ok(()) };
        let total: f64 = m.weights(&r).unwrap().iter().sum();
        let mag = d.weight_magnitude(0, d.n(), r.a().as_slice());
        prop_assert!(holds(&[total], &[1.0], &[mag]), "Σw = {total}, mag {mag:e}");
    }

    /// The weights reproduce the forecast: `ŷ = Σᵢ wᵢ yᵢ`.
    #[test]
    fn weights_reproduce_the_forecast(
        d in data(4, 3, 50), lambda in lambdas(), intercept in any::<bool>(),
        block_log2 in 0u32..=3, a in any::<Index>(), b in any::<Index>(),
    ) {
        let d = if intercept { d.with_intercept() } else { d };
        let m = d.matrix(block_log2);
        let (s, e) = ordered(a.index(d.n() + 1), b.index(d.n() + 1));
        let r = m.query(&d.x_q, s, e, &config(lambda, intercept));
        assert_criterion(&d.xs[s * d.p..e * d.p], d.p, lambda, intercept, accepted(&r)?)?;
        let Ok(r) = r else { return Ok(()) };
        let mut total = vec![0.0; d.q];
        for (i, w) in (s..e).zip(m.weights(&r).unwrap()) {
            for (acc, y) in total.iter_mut().zip(d.y(i)) {
                *acc += w * y;
            }
        }
        let mag = d.dual_magnitude(s, e, r.a().as_slice());
        prop_assert!(holds(&total, r.forecast().as_slice(), &mag));
    }

    /// Dense OLS attention reproduces ridge forecasts, in sample and out, and its rows sum to one
    /// with a free intercept. The dense path forms `XᵀX` in another order than the statistics,
    /// so near the threshold the two may disagree on rejection; each verdict is checked on its
    /// own, and the identity wherever both accept.
    #[test]
    fn attention_matches_ridge(d in data(4, 3, 30), lambda in lambdas(), intercept in any::<bool>()) {
        let d = if intercept { d.with_intercept() } else { d };
        let cfg = config(lambda, intercept);
        let keys = DMatrix::from_row_slice(d.n(), d.p, &d.xs);
        let values = DMatrix::from_row_slice(d.n(), d.q, &d.ys);
        let mut rows = d.xs.clone();
        rows.extend_from_slice(&d.x_q);
        let queries = DMatrix::from_row_slice(d.n() + 1, d.p, &rows);
        let attention = ols_attention(&keys, &values, &queries, &cfg);
        let stats = d.stats(0, d.n());
        let model = ridge_fit(&stats, &cfg);
        assert_criterion(&d.xs, d.p, lambda, intercept, accepted(&attention)?)?;
        assert_criterion(&d.xs, d.p, lambda, intercept, accepted(&model)?)?;
        let (Ok(attention), Ok(model)) = (attention, model) else { return Ok(()) };
        for (j, query) in rows.chunks_exact(d.p).enumerate() {
            let expected = model.predict(query).unwrap();
            let got: Vec<f64> = attention.forecast.row(j).iter().copied().collect();
            let a = solve_query(&stats, &cfg, query).unwrap();
            let mut mag = d.dual_magnitude(0, d.n(), a.as_slice());
            for (k, m) in mag.iter_mut().enumerate() {
                let beta: Vec<f64> = model.beta.row(k).iter().copied().collect();
                *m = m.max(abs_dot(&beta, query));
            }
            prop_assert!(holds(&got, expected.as_slice(), &mag), "{got:?} vs {expected:?}");
            if intercept {
                let mag = d.weight_magnitude(0, d.n(), a.as_slice());
                prop_assert!(holds(&[attention.weights.row(j).sum()], &[1.0], &[mag]));
            }
        }
    }

    /// Exactly singular designs (duplicates, collinear columns, fewer entries than coordinates,
    /// a column that is a sum of others, an untouched column) are refused at λ = 0 by every
    /// entry point, whatever the rounding does.
    #[test]
    fn exactly_singular_designs_are_always_rejected(
        d in data(4, 2, 40).prop_filter("p ≥ 2", |d| d.p >= 2),
        intercept in any::<bool>(), kind in 0usize..5, i in any::<Index>(), j in any::<Index>(),
        c in prop_oneof![Just(2.0), Just(3.0), Just(-0.7), Just(1e3), Just(1e-3)],
        block_log2 in 0u32..=3,
    ) {
        let d = degenerate(if intercept { d.with_intercept() } else { d }, kind, i, j, c);
        let cfg = config(0.0, intercept);
        let refused = Some(OlsError::NotPositiveDefinite);
        let stats = d.stats(0, d.n());
        prop_assert_eq!(ridge_fit(&stats, &cfg).err(), refused.clone());
        prop_assert_eq!(solve_query(&stats, &cfg, &d.x_q).err(), refused.clone());
        let m = d.matrix(block_log2);
        prop_assert_eq!(m.query(&d.x_q, 0, d.n(), &cfg).err(), refused.clone());
        prop_assert_eq!(m.summarize(0, d.n(), &cfg).err(), refused.clone());
        let keys = DMatrix::from_row_slice(d.n(), d.p, &d.xs);
        let values = DMatrix::from_row_slice(d.n(), d.q, &d.ys);
        let queries = DMatrix::from_row_slice(1, d.p, &d.x_q);
        prop_assert_eq!(ols_attention(&keys, &values, &queries, &cfg).err(), refused);
        prop_assert!(scaled_lambda_min(&d.xs, d.p, 0.0, intercept) <= tau(d.p, d.n()));
    }

    /// The tree read of any range equals a direct pass over its entries.
    #[test]
    fn tree_equals_batch(
        d in data(3, 2, 70), block_log2 in 0u32..=3, a in any::<Index>(), b in any::<Index>(),
    ) {
        let m = d.matrix(block_log2);
        let (s, e) = ordered(a.index(d.n() + 1), b.index(d.n() + 1));
        prop_assert!(stats_close(&m.range_stats(s, e).unwrap(), &d.stats(s, e), 1e-12));
    }

    /// A read depends on its range alone: later pushes never change it, to the last bit, and a
    /// refusal stays a refusal.
    #[test]
    fn reads_ignore_later_entries(
        d in data(3, 2, 70), block_log2 in 0u32..=3, lambda in lambdas(),
        cut in any::<Index>(), a in any::<Index>(), b in any::<Index>(),
    ) {
        let c = cut.index(d.n() + 1);
        let mut m = Matrix::new(d.p, d.q, block_log2).unwrap();
        for i in 0..c {
            m.push(i as i64, d.x(i), d.y(i)).unwrap();
        }
        let (s, e) = ordered(a.index(c + 1), b.index(c + 1));
        let cfg = config(lambda, false);
        let before = (m.range_stats(s, e).unwrap(), m.query(&d.x_q, s, e, &cfg));
        for i in c..d.n() {
            m.push(i as i64, d.x(i), d.y(i)).unwrap();
        }
        let after = (m.range_stats(s, e).unwrap(), m.query(&d.x_q, s, e, &cfg));
        prop_assert_eq!(before, after);
    }

    /// The contributions of a partition of the range add up to the forecast.
    #[test]
    fn contributions_partition_the_forecast(
        d in data(3, 2, 60), block_log2 in 0u32..=3, lambda in lambdas(),
        cuts in vec(any::<Index>(), 0..6),
    ) {
        let d = d.with_intercept();
        let m = d.matrix(block_log2);
        let r = m.query(&d.x_q, 0, d.n(), &config(lambda, true));
        assert_criterion(&d.xs, d.p, lambda, true, accepted(&r)?)?;
        let Ok(r) = r else { return Ok(()) };
        let mut bounds: Vec<usize> = cuts.iter().map(|c| c.index(d.n() + 1)).collect();
        bounds.extend([0, d.n()]);
        bounds.sort_unstable();
        let mut total = vec![0.0; d.q];
        for pair in bounds.windows(2) {
            let part = m.contribution(&r, pair[0], pair[1]).unwrap();
            for (acc, c) in total.iter_mut().zip(part.iter()) {
                *acc += c;
            }
        }
        let mag = d.dual_magnitude(0, d.n(), r.a().as_slice());
        prop_assert!(holds(&total, r.forecast().as_slice(), &mag));
    }

    /// One-hot outcomes with a free intercept give class forecasts that sum to one: the decision
    /// head's options come out as signed probabilities.
    #[test]
    fn one_hot_forecasts_sum_to_one(
        d in data(4, 1, 40), classes in vec(0usize..3, 40), lambda in lambdas(),
        block_log2 in 0u32..=3,
    ) {
        let d = d.with_intercept();
        let mut m = Matrix::new(d.p, 3, block_log2).unwrap();
        for (i, &class) in classes.iter().enumerate().take(d.n()) {
            let mut y = [0.0; 3];
            y[class] = 1.0;
            m.push(i as i64, d.x(i), &y).unwrap();
        }
        let r = m.query(&d.x_q, 0, d.n(), &config(lambda, true));
        assert_criterion(&d.xs, d.p, lambda, true, accepted(&r)?)?;
        let Ok(r) = r else { return Ok(()) };
        let mag = d.weight_magnitude(0, d.n(), r.a().as_slice());
        prop_assert!(holds(&[r.forecast().sum()], &[1.0], &[mag]), "{:?}", r.forecast());
    }

    /// RLS equals weighted ridge with weights `γ^(n−1−i)` and penalty `γⁿλ`.
    ///
    /// RLS starts from `P = I/λ`, and its first updates cancel entries of that size, so it
    /// carries an error of order `ε·max‖xᵢ‖²/λ` on top of the batch's `ε·κ(S)`. The tolerance is
    /// `10⁻⁷ + 8ε(max‖xᵢ‖²/λ + κ(S))`, relative. The constant 8 is empirical, not a proof: over
    /// 10⁵ random cases at λ = 10⁻¹² the worst error was 1.13 times the bracket. Where the batch
    /// refuses the design there is no reference, and the case only checks the refusal.
    #[test]
    fn rls_equals_weighted_ridge(
        d in data(4, 2, 40), lambda in prop_oneof![1 => Just(1e-12), 3 => 0.1..10.0],
        gamma in 0.9..=1.0,
    ) {
        let n = d.n();
        let mut rls = Rls::new(d.p, d.q, lambda, gamma).unwrap();
        let mut stats = SuffStats::zeros(d.p, d.q);
        let mut weighted = Vec::with_capacity(d.xs.len());
        for i in 0..n {
            let w = gamma.powi((n - 1 - i) as i32);
            rls.update(d.x(i), d.y(i)).unwrap();
            stats.push(d.x(i), d.y(i), w).unwrap();
            weighted.extend(d.x(i).iter().map(|x| x * w.sqrt()));
        }
        let penalty = gamma.powi(n as i32) * lambda;
        let batch = ridge_fit(&stats, &RidgeConfig::new(penalty).unwrap());
        assert_criterion(&weighted, d.p, penalty, false, accepted(&batch)?)?;
        let Ok(batch) = batch else { return Ok(()) };
        let lambda_min = scaled_lambda_min(&weighted, d.p, penalty, false);
        let kappa = d.p as f64 / lambda_min;
        let x2 = (0..n).map(|i| dot(d.x(i), d.x(i))).fold(0.0, f64::max);
        let tol = 1e-7 + 8.0 * f64::EPSILON * (x2 / lambda + kappa);
        prop_assert!(
            close(rls.beta().as_slice(), batch.beta.as_slice(), tol),
            "{} vs {}, tol {tol:e}", rls.beta(), batch.beta
        );
    }

    /// A period's summary, read off the statistics, matches a direct pass over its entries.
    #[test]
    fn summary_matches_a_direct_pass(
        d in data(3, 2, 40), lambda in lambdas(), block_log2 in 0u32..=3,
    ) {
        let d = d.with_intercept();
        let m = d.matrix(block_log2);
        let summary = m.summarize(0, d.n(), &config(lambda, true));
        assert_criterion(&d.xs, d.p, lambda, true, accepted(&summary)?)?;
        let Ok(summary) = summary else { return Ok(()) };
        let n = d.n() as f64;
        let mut rss = vec![0.0; d.q];
        let mut mean = vec![0.0; d.q];
        // `Σᵢ (|yᵢⱼ| + Σₖ |βⱼₖ xᵢₖ|)²` bounds every term of both ways of computing the RSS.
        let mut mag = vec![0.0; d.q];
        for i in 0..d.n() {
            let fit = summary.model.predict(d.x(i)).unwrap();
            for (j, &y) in d.y(i).iter().enumerate() {
                rss[j] += (y - fit[j]).powi(2);
                mean[j] += y / n;
                let beta: Vec<f64> = summary.model.beta.row(j).iter().copied().collect();
                mag[j] += (y.abs() + abs_dot(&beta, d.x(i))).powi(2);
            }
        }
        let var: Vec<f64> = summary.residual_var.iter().map(|v| v * n).collect();
        prop_assert!(holds(&var, &rss, &mag), "{var:?} vs {rss:?}, mag {mag:?}");
        prop_assert!(close(summary.mean_y.as_slice(), &mean, 1e-12));
        let direct = residual_sum_squares(&d.stats(0, d.n()), &summary.model).unwrap();
        prop_assert!(holds(direct.as_slice(), &rss, &mag));
    }
}

/// Simple regression, by hand: x = 0..5, so x̄ = 2 and Sxx = 10, and at x_q = 6 the weights are
/// `wᵢ = 1/5 + (6 − 2)(xᵢ − 2)/10`. The two oldest points lie on the far side of the mean from
/// the query, so they count against their own outcomes.
#[test]
fn simple_regression_weights_match_the_textbook() {
    let ys = [0.3, -0.1, 0.4, 0.2, 0.9];
    let mut m = Matrix::new(2, 1, 1).unwrap();
    for (i, &y) in ys.iter().enumerate() {
        m.push(i as i64, &[1.0, i as f64], &[y]).unwrap();
    }
    let r = m
        .query(
            &[1.0, 6.0],
            0,
            5,
            &RidgeConfig::with_intercept(0.0).unwrap(),
        )
        .unwrap();
    let weights = m.weights(&r).unwrap();
    for (w, expected) in weights.iter().zip([-0.6, -0.2, 0.2, 0.6, 1.0]) {
        assert!((w - expected).abs() < 1e-12, "{weights:?}");
    }
    // ȳ = 0.34 and the slope is Sxy / Sxx = 1.5 / 10, so the line gives 0.34 + 0.15 · 4 at x = 6.
    assert!((r.forecast()[0] - 0.94).abs() < 1e-12);
}

/// Probe P4 of the iteration-1 audit: two identical entries make `G` singular at λ = 0, and
/// rounding used to leave a tiny positive pivot, so `query` returned `Ok(1.5)` although every
/// outcome is 1.0.
#[test]
fn duplicate_entries_at_zero_penalty_are_rejected() {
    let mut m = Matrix::new(2, 1, 1).unwrap();
    for t in 0..2 {
        m.push(t, &[1.0, 0.3], &[1.0]).unwrap();
    }
    let cfg = RidgeConfig::with_intercept(0.0).unwrap();
    assert_eq!(
        m.query(&[1.0, 0.7], 0, 2, &cfg).map(|r| r.forecast()[0]),
        Err(OlsError::NotPositiveDefinite)
    );
}

/// Probe P2 of the audit: a third column equal to twice the second up to 10⁻⁹ noise. At λ = 0
/// the design is singular to working precision (`λ_min(S) ≈ 10⁻¹⁹`, below what a rounded `G`
/// can resolve) and is refused; the audit saw `Σw = 0.999999999345` there. From λ = 10⁻⁶ the
/// read is accepted and its weights sum to one to 10⁻¹².
#[test]
fn near_collinear_designs_are_refused_until_a_penalty_resolves_them() {
    let mut m = Matrix::new(3, 1, 1).unwrap();
    for i in 0..40 {
        let x = 0.05 * f64::from(i) - 1.0;
        let noise = f64::from((i * 7919) % 13) / 13.0 - 0.5;
        m.push(i64::from(i), &[1.0, x, 2.0 * x + 1e-9 * noise], &[x.sin()])
            .unwrap();
    }
    let x_q = [1.0, 0.5, 1.0 + 1e-7];
    let cfg = |l| RidgeConfig::with_intercept(l).unwrap();
    assert_eq!(
        m.query(&x_q, 0, 40, &cfg(0.0)),
        Err(OlsError::NotPositiveDefinite)
    );
    for lambda in [1e-6, 0.05] {
        let r = m.query(&x_q, 0, 40, &cfg(lambda)).unwrap();
        let total: f64 = m.weights(&r).unwrap().iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "λ = {lambda}: Σw = {total}");
    }
}

/// Probe P5 of the audit: a Gram matrix that overflows is reported as non-finite, and one that
/// barely fits never returns a non-finite number.
#[test]
fn overflow_is_an_error_not_a_number() {
    let cfg = RidgeConfig::with_intercept(0.05).unwrap();
    let memory = |scale: f64| {
        let mut m = Matrix::new(2, 1, 1).unwrap();
        for i in 0..5 {
            m.push(i, &[1.0, scale * (i as f64 + 1.0)], &[1.0]).unwrap();
        }
        m
    };
    let m = memory(1e160);
    assert_eq!(m.query(&[1.0, 1e160], 0, 5, &cfg), Err(OlsError::NonFinite));
    let m = memory(1e153);
    match m.query(&[1.0, 1e153], 0, 5, &cfg) {
        Ok(r) => {
            assert!(
                r.a()
                    .iter()
                    .chain(r.forecast().iter())
                    .all(|v| v.is_finite())
            );
            assert!(m.weights(&r).unwrap().iter().all(|w| w.is_finite()));
        }
        Err(e) => assert!(matches!(
            e,
            OlsError::NotPositiveDefinite | OlsError::NonFinite
        )),
    }
}

fn memory_of(n: usize, slope: f64) -> Matrix {
    let mut m = Matrix::new(2, 1, 1).unwrap();
    for i in 0..n {
        let x = i as f64 / 10.0;
        m.push(i as i64, &[1.0, x], &[slope * x]).unwrap();
    }
    m
}

/// Audit finding §3(a): two independent memories with the same `p`, even with the same entries,
/// never accept each other's readouts.
#[test]
fn a_readout_is_refused_by_an_independent_memory() {
    let cfg = RidgeConfig::with_intercept(0.1).unwrap();
    let (m1, m2, m3) = (memory_of(12, 1.0), memory_of(12, 1.0), memory_of(30, -3.0));
    let r = m1.query(&[1.0, 0.5], 2, 10, &cfg).unwrap();
    assert!(m1.weights(&r).is_ok() && m1.contribution(&r, 2, 5).is_ok());
    for other in [&m2, &m3] {
        assert_eq!(other.weights(&r), Err(OlsError::ForeignReadout));
        assert_eq!(other.contribution(&r, 2, 5), Err(OlsError::ForeignReadout));
    }
    // The same query on the twin gives the same numbers, under its own readout.
    let twin = m2.query(&[1.0, 0.5], 2, 10, &cfg).unwrap();
    assert_eq!(
        (twin.a(), twin.forecast()),
        (r.a(), r.forecast()),
        "same entries, same read"
    );
    assert_eq!(m2.weights(&twin).unwrap(), m1.weights(&r).unwrap());
}

/// Audit finding §3(b): a memory cloned before a readout was taken, which then receives other
/// entries, refuses it, and the original refuses the clone's. The original keeps accepting its
/// own readout after later pushes, with the same weights to the bit, since reads are pure.
#[test]
fn a_readout_is_refused_by_a_diverged_clone() {
    let cfg = RidgeConfig::with_intercept(0.1).unwrap();
    let mut m = memory_of(12, 1.0);
    let mut c = m.clone();
    let r = m.query(&[1.0, 0.5], 0, 12, &cfg).unwrap();
    let before = m.weights(&r).unwrap();
    c.push(100, &[1.0, 9.0], &[-50.0]).unwrap();
    m.push(100, &[1.0, 2.0], &[2.0]).unwrap();
    assert_eq!(c.weights(&r), Err(OlsError::ForeignReadout));
    assert_eq!(c.contribution(&r, 0, 3), Err(OlsError::ForeignReadout));
    let rc = c.query(&[1.0, 0.5], 0, 13, &cfg).unwrap();
    assert_eq!(m.weights(&rc), Err(OlsError::ForeignReadout));
    assert_eq!(m.weights(&r).unwrap(), before);
    // A clone that has not diverged refuses too: the id is the whole check.
    assert_eq!(m.clone().weights(&r), Err(OlsError::ForeignReadout));
}

/// The `N` in `τ = 4p(N + p + 2)ε`: a single state `(1, x)` repeated 10⁵ times is exactly
/// singular, but a sequential sum of 10⁵ copies of an inexact `x` drifts by up to `Nε` relative,
/// far above a threshold blind to `N`. Read through the statistics and through a Matrix whose one
/// block is scanned sequentially, it is refused for every `x` tried.
#[test]
fn long_runs_of_one_state_are_refused() {
    let n = 100_000;
    let cfg = RidgeConfig::with_intercept(0.0).unwrap();
    for x in [0.1, 0.3, 1.0 / 3.0, 0.7, std::f64::consts::PI, 1e-3] {
        let xs: Vec<f64> = (0..n).flat_map(|_| [1.0, x]).collect();
        let ys = vec![1.0; n];
        let stats = SuffStats::from_rows(&xs, &ys, 2, 1).unwrap();
        assert_eq!(
            solve_query(&stats, &cfg, &[1.0, 0.5]),
            Err(OlsError::NotPositiveDefinite),
            "x = {x}"
        );
        let mut m = Matrix::new(2, 1, 20).unwrap();
        for i in 0..n {
            m.push(i as i64, &[1.0, x], &[1.0]).unwrap();
        }
        assert_eq!(
            m.query(&[1.0, 0.5], 0, n, &cfg).err(),
            Some(OlsError::NotPositiveDefinite),
            "x = {x}"
        );
    }
}

/// A forecast that overflows is an error, even when the direction `a` is finite: here
/// `a = 10¹⁰/3` and `ŷ = 10³⁰⁰ · 10¹⁰`.
#[test]
fn an_overflowing_forecast_is_an_error() {
    let mut m = Matrix::new(1, 1, 1).unwrap();
    for t in 0..3 {
        m.push(t, &[1.0], &[1e300]).unwrap();
    }
    let cfg = RidgeConfig::new(0.0).unwrap();
    assert_eq!(
        m.query(&[1e10], 0, 3, &cfg).err(),
        Some(OlsError::NonFinite)
    );
    let keys = DMatrix::from_element(3, 1, 1.0);
    let values = DMatrix::from_element(3, 1, 1e300);
    let queries = DMatrix::from_element(1, 1, 1e10);
    assert_eq!(
        ols_attention(&keys, &values, &queries, &cfg).err(),
        Some(OlsError::NonFinite)
    );
    let model = ridge_fit(&m.range_stats(0, 3).unwrap(), &cfg).unwrap();
    assert_eq!(model.predict(&[1e10]).err(), Some(OlsError::NonFinite));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// At λ = 0, ridge is ordinary least squares, which does not care about units: rescaling a
    /// coordinate of every state and of the query by `2ᵏ` changes no verdict and no forecast, to
    /// the bit. Powers of two scale every float exactly, so this holds in floating point and not
    /// just in exact arithmetic, which pins the diagonal scaling of the singularity test.
    #[test]
    fn units_do_not_matter_at_zero_penalty(
        d in data(4, 2, 30), col in any::<Index>(), k in -60i32..=60,
    ) {
        let c = col.index(d.p);
        let scale = 2f64.powi(k);
        let mut scaled = d.clone();
        for row in scaled.xs.chunks_exact_mut(d.p) {
            row[c] *= scale;
        }
        scaled.x_q[c] *= scale;
        let cfg = config(0.0, false);
        let read = |d: &Data| d.matrix(2).query(&d.x_q, 0, d.n(), &cfg).map(|r| r.forecast().clone());
        prop_assert_eq!(read(&d), read(&scaled));
    }
}

/// Deterministic xorshift64*, so the long designs below need no extra dependency.
struct Rng(u64);

impl Rng {
    /// Uniform on [−√3, √3): mean 0, variance 1.
    fn unit_variance(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let u = (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64;
        (2.0 * u - 1.0) * 3f64.sqrt()
    }
}

/// Centres and scales every column but the intercept with its own sample mean and standard
/// deviation, as the pipeline does inside a fold.
fn standardize(xs: &mut [f64], p: usize) {
    let n = (xs.len() / p) as f64;
    for k in 1..p {
        let mean = xs.iter().skip(k).step_by(p).sum::<f64>() / n;
        let var = xs
            .iter()
            .skip(k)
            .step_by(p)
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n;
        let sd = var.sqrt();
        for x in xs.iter_mut().skip(k).step_by(p) {
            *x = (*x - mean) / sd;
        }
    }
}

/// The intended use is far from the refusal frontier: 32 standardized features plus an
/// intercept, N = 10⁵ entries, λ = 0, is accepted.
/// The outcome is an exact linear function `y = xᵀβ*`, so the answer is known without
/// another solver: the fit recovers `β*`, the forecast at a fresh state is `x_qᵀβ*`, and the
/// weights sum to one.
#[test]
fn long_standardized_designs_are_accepted() {
    let (p, n) = (33, 100_000);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut xs: Vec<f64> = (0..n * p)
        .map(|i| if i % p == 0 { 1.0 } else { rng.unit_variance() })
        .collect();
    standardize(&mut xs, p);
    let beta: Vec<f64> = (0..p).map(|_| rng.unit_variance()).collect();
    let ys: Vec<f64> = xs.chunks_exact(p).map(|x| dot(x, &beta)).collect();
    let mut x_q: Vec<f64> = (0..p).map(|_| rng.unit_variance()).collect();
    x_q[0] = 1.0;
    let expected = dot(&x_q, &beta);
    let cfg = RidgeConfig::with_intercept(0.0).unwrap();

    // One accumulation (about 7 s in a debug build): blocks of 2¹⁰ scanned in order, then the
    // tree.
    let mut m = Matrix::new(p, 1, 10).unwrap();
    for (i, (x, y)) in xs.chunks_exact(p).zip(&ys).enumerate() {
        m.push(i as i64, x, &[*y]).unwrap();
    }
    let fit = ridge_fit(&m.range_stats(0, n).unwrap(), &cfg).unwrap();
    let got: Vec<f64> = fit.beta.iter().copied().collect();
    assert!(close(&got, &beta, 1e-9), "β = {got:?}, β* = {beta:?}");
    let r = m.query(&x_q, 0, n, &cfg).unwrap();
    assert!(close(r.forecast().as_slice(), &[expected], 1e-9));
    let total: f64 = m.weights(&r).unwrap().iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "Σw = {total}");
}

/// The frontier of the docs, on both sides: a feature with `r = |mean|/std` beside the
/// intercept has `λ_min(S) ≈ 1/(2r²)`, against `τ ≈ 1.8·10⁻¹⁰` for p = 2 and N = 10⁵. At
/// r = 10³ it is accepted; at r = 10⁵ it is refused, and standardizing it inside the fold makes
/// it acceptable again.
#[test]
fn uncentred_features_meet_the_frontier_where_the_docs_say() {
    let n = 100_000;
    let cfg = RidgeConfig::with_intercept(0.0).unwrap();
    let design = |mean: f64| {
        let mut rng = Rng(0xd1b5_4a32_d192_ed03);
        let xs: Vec<f64> = (0..n)
            .flat_map(|_| [1.0, mean + rng.unit_variance()])
            .collect();
        let ys: Vec<f64> = xs
            .iter()
            .skip(1)
            .step_by(2)
            .map(|x| 0.5 - x * 1e-6)
            .collect();
        (xs, ys)
    };
    let verdict = |xs: &[f64], ys: &[f64]| {
        let stats = SuffStats::from_rows(xs, ys, 2, 1).unwrap();
        solve_query(&stats, &cfg, &[1.0, xs[1]]).err()
    };
    let (xs, ys) = design(1e3);
    assert_eq!(verdict(&xs, &ys), None, "r = 10³ is accepted");
    let (mut xs, ys) = design(1e5);
    assert_eq!(
        verdict(&xs, &ys),
        Some(OlsError::NotPositiveDefinite),
        "r = 10⁵ is refused"
    );
    standardize(&mut xs, 2);
    assert_eq!(verdict(&xs, &ys), None, "standardized, it is accepted");
}
