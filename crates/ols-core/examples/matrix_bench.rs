//! The Matrix workload of iteration 3, on synthetic data of the same shape.
//!
//! ```text
//! cargo run --release -p ols-core --example matrix_bench [-- --threads 12 --block-log2 6]
//! ```
//!
//! - 2.0·10⁶ entries, p = 15, q = 4 (a one-hot class of 3, then a return), stamped hourly, 30
//!   per hour (one per member pair), pushed in time order;
//! - ridge batch readouts: 10⁵ queries × W = 3 windows (30 days, 365 days, all history) × R = 6
//!   penalties, 30 queries per decision hour;
//! - kernel readouts: 10⁴ queries over the 365-day window (262,800 candidates), τ ∈ {0.5, 1,
//!   2}·√15 and k ∈ {50, 200, 1000}.
//!
//! Times are wall-clock, best of one run each; the totals extrapolate linearly to 2·10⁶
//! queries. The bit-identity across thread counts is checked on a slice of each readout.

use std::time::Instant;

use ols_core::{KernelSpec, Matrix, ReadStatus, RidgeGrid, Window};

const HOUR: i64 = 3_600_000;
const DAY: i64 = 24 * HOUR;
const PAIRS: usize = 30;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately N(0, 1): the centred sum of four uniforms, scaled.
    fn normal(&mut self) -> f64 {
        let s: f64 = (0..4).map(|_| self.uniform()).sum();
        (s - 2.0) * 3f64.sqrt()
    }
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// FNV-1a over the bit patterns, NaNs included.
fn digest(v: &[f64]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn main() {
    let entries = arg("--entries", 2_000_000);
    let ridge_queries = arg("--ridge-queries", 100_000);
    let kernel_queries = arg("--kernel-queries", 10_000);
    let threads = arg("--threads", 12);
    let block_log2 = arg("--block-log2", 6) as u32;
    let (p, q) = (15, 4);
    println!(
        "entries {entries}, p {p}, q {q}, block 2^{block_log2}, threads {threads}, \
         ridge queries {ridge_queries}, kernel queries {kernel_queries}"
    );

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let t0: i64 = 1_514_764_800_000; // 2018-01-01
    let mut times = Vec::with_capacity(entries);
    let mut xs = Vec::with_capacity(entries * p);
    let mut ys = Vec::with_capacity(entries * q);
    for i in 0..entries {
        times.push(t0 + (i / PAIRS) as i64 * HOUR);
        let x: Vec<f64> = (0..p).map(|_| rng.normal()).collect();
        let signal = 0.1 * x[0] - 0.05 * x[3];
        let ret = 0.01 * (signal + rng.normal());
        let class = if ret < -0.004 {
            0
        } else if ret > 0.004 {
            2
        } else {
            1
        };
        xs.extend_from_slice(&x);
        let mut y = [0.0; 4];
        y[class] = 1.0;
        y[3] = ret;
        ys.extend_from_slice(&y);
    }

    let start = Instant::now();
    let mut m = Matrix::new(p, q, block_log2).unwrap();
    for i in 0..entries {
        m.push(times[i], &xs[i * p..(i + 1) * p], &ys[i * q..(i + 1) * q])
            .unwrap();
    }
    let push = start.elapsed().as_secs_f64();
    println!("push: {push:.2} s for {entries} entries");

    // Decisions 30 per hour, starting once 400 days of history exist.
    let hours = entries / PAIRS;
    let first_hour = (400 * 24).min(hours.saturating_sub(1));
    let make_queries = |count: usize, rng: &mut Rng| {
        let mut as_of = Vec::with_capacity(count);
        let mut x_q = Vec::with_capacity(count * p);
        for k in 0..count {
            let hour = (first_hour + k / PAIRS).min(hours - 1);
            as_of.push(t0 + hour as i64 * HOUR);
            x_q.extend((0..p).map(|_| rng.normal()));
        }
        (as_of, x_q)
    };

    let windows = [
        Window::last(30 * DAY).unwrap(),
        Window::last(365 * DAY).unwrap(),
        Window::ALL,
    ];
    let rhos = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0];
    let scale = vec![1.0; p];
    let grid = RidgeGrid::new(&windows, &rhos, &scale)
        .unwrap()
        .with_leverage(true)
        .with_rss(true);
    let (as_of, x_q) = make_queries(ridge_queries, &mut rng);
    let start = Instant::now();
    let batch = m.ridge_batch(&x_q, &as_of, &grid, threads).unwrap();
    let ridge = start.elapsed().as_secs_f64();
    let mut counts = [0usize; 5];
    for s in &batch.status {
        counts[s.code() as usize] += 1;
    }
    println!(
        "ridge batch: {ridge:.2} s for {ridge_queries} queries x 3 windows x 6 penalties \
         ({:.2} s per 1e5 queries; statuses ok/empty/singular/nonfinite = {:?})",
        ridge * 1e5 / ridge_queries as f64,
        &counts[..4]
    );
    let slice = ridge_queries.min(10_000);
    let one = m
        .ridge_batch(&x_q[..slice * p], &as_of[..slice], &grid, 1)
        .unwrap();
    let cells = slice * 3 * 6;
    println!(
        "ridge batch, 1 thread vs {threads} on {slice} queries: bit-identical = {}",
        same_bits(&one.forecast, &batch.forecast[..cells * q])
            && one.status[..] == batch.status[..cells]
            && same_bits(
                one.leverage.as_ref().unwrap(),
                &batch.leverage.as_ref().unwrap()[..cells]
            )
    );

    let sigma = vec![1.0; p];
    let t = 15f64.sqrt();
    // `--taus n` and `--ks n` keep the first n of each list, to time the parts apart.
    let taus = [0.5 * t, t, 2.0 * t];
    let ks = [50, 200, 1000];
    let spec = KernelSpec::new(
        Window::last(365 * DAY).unwrap(),
        &sigma,
        &taus[..arg("--taus", 3).min(3)],
        &ks[..arg("--ks", 3).min(3)],
    )
    .unwrap();
    let (nt, nk) = (spec.taus().len(), spec.ks().len());
    let (as_of, x_q) = make_queries(kernel_queries, &mut rng);
    let start = Instant::now();
    let kb = m.kernel_batch(&x_q, &as_of, &spec, threads).unwrap();
    let kernel = start.elapsed().as_secs_f64();
    let candidates: usize = kb.start.iter().zip(&kb.end).map(|(s, e)| e - s).sum();
    let ok = kb
        .knn_status
        .iter()
        .filter(|s| **s == ReadStatus::Ok)
        .count();
    println!(
        "kernel batch: {kernel:.2} s for {kernel_queries} queries, {:.0} candidates each \
         ({:.2} s per 1e4 queries; kNN ok {ok} of {})",
        candidates as f64 / kernel_queries.max(1) as f64,
        kernel * 1e4 / kernel_queries as f64,
        kb.knn_status.len()
    );
    let slice = kernel_queries.min(256);
    let one = m
        .kernel_batch(&x_q[..slice * p], &as_of[..slice], &spec, 1)
        .unwrap();
    println!(
        "kernel batch, 1 thread vs {threads} on {slice} queries: bit-identical = {}",
        same_bits(&one.softmax, &kb.softmax[..slice * nt * q])
            && same_bits(&one.knn, &kb.knn[..slice * nk * q])
    );
    println!(
        "digests (FNV-1a of the output bits; equal digests across builds mean equal bits): \
         ridge {:016x}, softmax {:016x}, knn {:016x}",
        digest(&batch.forecast),
        digest(&kb.softmax),
        digest(&kb.knn)
    );
    let full = 2_000_000.0;
    println!(
        "extrapolated to 2e6 queries: ridge {:.0} s, kernel {:.0} s, total with push {:.0} s",
        ridge * full / ridge_queries as f64,
        kernel * full / kernel_queries as f64,
        push + (ridge / ridge_queries as f64 + kernel / kernel_queries as f64) * full
    );
}
