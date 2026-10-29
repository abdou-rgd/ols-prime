//! The exact kernel readouts of H3 against a brute-force reference: kNN bit for bit under the
//! tie rule, softmax to 10⁻¹² relative, statuses, and the same bits whatever the thread count,
//! the other queries of the batch, or the entries pushed later.

// The references are written index by index, as the formulas are.
#![allow(clippy::needless_range_loop)]

use ols_core::{KernelBatch, KernelSpec, Matrix, OlsError, ReadStatus, Window};
use proptest::collection::vec;
use proptest::prelude::*;

#[derive(Debug, Clone)]
struct Case {
    p: usize,
    q: usize,
    times: Vec<i64>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// (as_of, state) per query.
    queries: Vec<(i64, Vec<f64>)>,
    window: Window,
    sigma: Vec<f64>,
    taus: Vec<f64>,
    ks: Vec<usize>,
}

impl Case {
    fn n(&self) -> usize {
        self.times.len()
    }

    fn matrix(&self) -> Matrix {
        let mut m = Matrix::new(self.p, self.q, 2).unwrap();
        for i in 0..self.n() {
            m.push(
                self.times[i],
                &self.xs[i * self.p..(i + 1) * self.p],
                &self.ys[i * self.q..(i + 1) * self.q],
            )
            .unwrap();
        }
        m
    }

    fn spec(&self) -> KernelSpec {
        KernelSpec::new(self.window, &self.sigma, &self.taus, &self.ks).unwrap()
    }

    fn x_q(&self) -> Vec<f64> {
        self.queries.iter().flat_map(|q| q.1.clone()).collect()
    }

    fn as_of(&self) -> Vec<i64> {
        self.queries.iter().map(|q| q.0).collect()
    }
}

/// Features either continuous or on a coarse grid, where exact distance ties are common.
fn feature(grid: bool) -> BoxedStrategy<f64> {
    if grid {
        prop_oneof![Just(-1.0), Just(-0.5), Just(0.0), Just(0.5), Just(1.0)].boxed()
    } else {
        (-1.0..1.0).boxed()
    }
}

fn case() -> impl Strategy<Value = Case> {
    (1usize..=4, 1usize..=3, 0usize..700, any::<bool>()).prop_flat_map(|(p, q, n, grid)| {
        (
            vec(0i64..3, n),
            vec(feature(grid), n * p),
            vec(-1.0..1.0f64, n * q),
            vec((-5i64..800, vec(feature(grid), p)), 0..12),
            prop_oneof![Just(None), (1i64..900).prop_map(Some)],
            vec(0.5..2.0f64, p),
            vec(prop_oneof![0.05..3.0f64, Just(0.5 * 15f64.sqrt())], 0..4),
            vec(prop_oneof![1usize..5, 1usize..800], 0..4),
        )
            .prop_map(move |(steps, xs, ys, queries, span, sigma, taus, ks)| {
                let mut t = 0;
                let times = steps
                    .iter()
                    .map(|s| {
                        t += s;
                        t
                    })
                    .collect();
                Case {
                    p,
                    q,
                    times,
                    xs,
                    ys,
                    queries,
                    window: span.map_or(Window::ALL, |s| Window::last(s).unwrap()),
                    sigma,
                    taus,
                    ks,
                }
            })
    })
}

/// `Σⱼ (x_q,j/σⱼ − x_i,j/σⱼ)²`, over j in order: the documented distance.
fn distance(c: &Case, xq: &[f64], i: usize) -> f64 {
    let mut d = 0.0;
    for j in 0..c.p {
        let t = xq[j] / c.sigma[j] - c.xs[i * c.p + j] / c.sigma[j];
        d += t * t;
    }
    d
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_kernel(a: &KernelBatch, b: &KernelBatch) -> bool {
    same_bits(&a.softmax, &b.softmax)
        && same_bits(&a.knn, &b.knn)
        && a.softmax_status == b.softmax_status
        && a.knn_status == b.knn_status
        && a.start == b.start
        && a.end == b.end
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    /// Every cell against a direct computation over its window, entry by entry.
    #[test]
    fn kernels_match_the_brute_force_reference(c in case()) {
        let m = c.matrix();
        let b = m.kernel_batch(&c.x_q(), &c.as_of(), &c.spec(), 0).unwrap();
        let (q, nt, nk) = (c.q, c.taus.len(), c.ks.len());
        for (qi, (as_of, xq)) in c.queries.iter().enumerate() {
            let (s, e) = m.window_range(*as_of, c.window);
            prop_assert_eq!((b.start[qi], b.end[qi]), (s, e));
            let n = e - s;
            let d: Vec<f64> = (s..e).map(|i| distance(&c, xq, i)).collect();
            // kNN: sort by (distance, index), sum the first k outcomes in that order.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| d[a].total_cmp(&d[b]).then(a.cmp(&b)));
            for (kk, &k) in c.ks.iter().enumerate() {
                let got = &b.knn[(qi * nk + kk) * q..(qi * nk + kk + 1) * q];
                let status = b.knn_status[qi * nk + kk];
                if n == 0 {
                    prop_assert_eq!(status, ReadStatus::Empty);
                } else if k > n {
                    prop_assert_eq!(status, ReadStatus::TooFew);
                } else {
                    prop_assert_eq!(status, ReadStatus::Ok);
                    let want: Vec<f64> = (0..q)
                        .map(|cc| {
                            let mut sum = 0.0;
                            for &r in &order[..k] {
                                sum += c.ys[(s + r) * q + cc];
                            }
                            sum / k as f64
                        })
                        .collect();
                    prop_assert!(same_bits(got, &want), "k = {k}: {got:?} vs {want:?}");
                    continue;
                }
                prop_assert!(got.iter().all(|v| v.is_nan()));
            }
            // Softmax: shift by the smallest distance, weights by libm's exp.
            let dmin = d.iter().copied().fold(f64::INFINITY, f64::min);
            for (ti, &tau) in c.taus.iter().enumerate() {
                let got = &b.softmax[(qi * nt + ti) * q..(qi * nt + ti + 1) * q];
                let status = b.softmax_status[qi * nt + ti];
                if n == 0 {
                    prop_assert_eq!(status, ReadStatus::Empty);
                    prop_assert!(got.iter().all(|v| v.is_nan()));
                    continue;
                }
                prop_assert_eq!(status, ReadStatus::Ok);
                let w: Vec<f64> = d.iter().map(|di| (-(di - dmin) / (2.0 * tau * tau)).exp()).collect();
                let sw: f64 = w.iter().sum();
                for cc in 0..q {
                    let want = (0..n).map(|r| w[r] * c.ys[(s + r) * q + cc]).sum::<f64>() / sw;
                    prop_assert!((got[cc] - want).abs() <= 1e-12 * (1.0 + want.abs()),
                        "τ = {tau}: {} vs {want}", got[cc]);
                }
            }
        }
    }

    /// The bits of a query do not depend on the thread count, on the other queries of the batch,
    /// nor on entries pushed after every decision time.
    #[test]
    fn kernel_reads_are_pure(c in case(), threads in 2usize..7) {
        let mut m = c.matrix();
        let spec = c.spec();
        let (x_q, as_of) = (c.x_q(), c.as_of());
        let base = m.kernel_batch(&x_q, &as_of, &spec, 1).unwrap();
        prop_assert!(same_kernel(&m.kernel_batch(&x_q, &as_of, &spec, threads).unwrap(), &base));
        for (qi, &d) in as_of.iter().enumerate() {
            let alone = m.kernel_batch(&x_q[qi * c.p..(qi + 1) * c.p], &[d], &spec, 1).unwrap();
            let (q, nt, nk) = (c.q, c.taus.len(), c.ks.len());
            prop_assert!(same_bits(&alone.softmax, &base.softmax[qi * nt * q..(qi + 1) * nt * q]));
            prop_assert!(same_bits(&alone.knn, &base.knn[qi * nk * q..(qi + 1) * nk * q]));
        }
        let last = c.times.last().copied().unwrap_or(0).max(800);
        for i in 0..300 {
            let x: Vec<f64> = (0..c.p).map(|j| ((i * 7 + j) as f64).sin()).collect();
            let y: Vec<f64> = (0..c.q).map(|k| ((i + k) as f64).cos()).collect();
            m.push(last + 1 + i as i64, &x, &y).unwrap();
        }
        prop_assert!(same_kernel(&m.kernel_batch(&x_q, &as_of, &spec, threads).unwrap(), &base));
    }
}

/// Every entry at the same state: every distance ties, and the k nearest are the k first.
#[test]
fn ties_go_to_the_lowest_index() {
    let mut m = Matrix::new(2, 1, 3).unwrap();
    for i in 0..1000 {
        m.push(i, &[0.25, -0.5], &[i as f64]).unwrap();
    }
    let spec = KernelSpec::new(Window::ALL, &[1.0, 1.0], &[1.0], &[1, 10, 1000, 1001]).unwrap();
    let b = m.kernel_batch(&[3.0, 3.0], &[999], &spec, 0).unwrap();
    assert_eq!(&b.knn[..3], &[0.0, 4.5, 499.5]);
    assert!(b.knn[3].is_nan());
    assert_eq!(
        b.knn_status,
        [
            ReadStatus::Ok,
            ReadStatus::Ok,
            ReadStatus::Ok,
            ReadStatus::TooFew
        ]
    );
    // Equal weights: the softmax is the plain mean, 499.5, to rounding.
    assert!((b.softmax[0] - 499.5).abs() < 1e-9);
}

/// A far query: every weight relative to the nearest entry, so nothing underflows to 0/0.
#[test]
fn a_far_query_does_not_underflow() {
    let mut m = Matrix::new(1, 1, 2).unwrap();
    for i in 0..600 {
        m.push(i, &[f64::from(i as i32) * 0.01], &[f64::from(i as i32)])
            .unwrap();
    }
    let spec = KernelSpec::new(Window::ALL, &[1.0], &[0.01], &[]).unwrap();
    // At x = 1000 every raw weight is exp(−10¹⁰): only the shift keeps the ratio defined.
    let b = m.kernel_batch(&[1000.0], &[599], &spec, 0).unwrap();
    assert_eq!(b.softmax_status, [ReadStatus::Ok]);
    assert!((b.softmax[0] - 599.0).abs() < 1e-9, "{}", b.softmax[0]);
}

#[test]
fn bad_kernel_inputs_are_errors() {
    let mut m = Matrix::new(2, 1, 1).unwrap();
    m.push(0, &[1e300, 0.0], &[0.0]).unwrap();
    let spec = KernelSpec::new(Window::ALL, &[1e-10, 1.0], &[1.0], &[1]).unwrap();
    assert_eq!(
        m.kernel_batch(&[0.0, 0.0], &[0], &spec, 1).err(),
        Some(OlsError::NonFinite)
    );
    let spec = KernelSpec::new(Window::ALL, &[1.0, 1.0], &[1.0], &[1]).unwrap();
    assert!(matches!(
        m.kernel_batch(&[0.0], &[0], &spec, 1),
        Err(OlsError::Dimension { .. })
    ));
    let spec = KernelSpec::new(Window::ALL, &[1.0], &[1.0], &[1]).unwrap();
    assert!(matches!(
        m.kernel_batch(&[0.0, 0.0], &[0], &spec, 1),
        Err(OlsError::Dimension { .. })
    ));
    let empty = m.kernel_batch(
        &[],
        &[],
        &KernelSpec::new(Window::ALL, &[1.0, 1.0], &[1.0], &[1]).unwrap(),
        0,
    );
    assert!(empty.unwrap().softmax.is_empty());
}

/// The footprint covers the union of the windows in whole blocks, and grows with the threads.
#[test]
fn footprints_bound_the_union_and_the_threads() {
    let mut m = Matrix::new(3, 2, 2).unwrap();
    for i in 0..1000 {
        m.push(i, &[0.0, 1.0, 2.0], &[0.0, 1.0]).unwrap();
    }
    let spec = KernelSpec::new(Window::last(100).unwrap(), &[1.0; 3], &[1.0], &[50]).unwrap();
    let as_of: Vec<i64> = (0..300).map(|i| 300 + i).collect();
    let one = m.kernel_footprint(&as_of, &spec, 1);
    // Windows (200, 300] to (499, 599]: entries 201..600, from block 200 to block 600.
    assert_eq!(one.union_entries, 400);
    assert_eq!((one.u_words, one.y_words, one.threads), (1200, 800, 1));
    let three = m.kernel_footprint(&as_of, &spec, 3);
    assert_eq!(three.threads, 3);
    assert_eq!(three.state_words, 3 * one.state_words);
    assert_eq!(m.kernel_footprint(&[], &spec, 0).union_entries, 0);
}
