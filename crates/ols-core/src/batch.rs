//! Batch readouts of Matrix: M queries × W windows × R penalties of the centred ridge, read in
//! parallel, with a status per cell.
//!
//! **Determinism.** Every cell is computed by one thread with the same sequential code as
//! [`Matrix::centred_query`], from the statistics of its own range. Threads only decide which
//! cells a thread computes, never how, and nothing is reduced across threads, so the output is
//! bit-identical whatever the thread count, and bit-identical to the one-by-one reads.
//!
//! **Sharing.** Queries that see the same range (the members deciding at the same hour) share
//! its statistics and its factorizations: a range is read and factored once per run of
//! consecutive queries with the same range in a chunk. This changes the cost, not the bits.

use std::sync::{Mutex, PoisonError};

use crate::OlsError;
use crate::centred::{CentredFit, check_rho, check_scale};
use crate::matrix::{Matrix, Window};
use crate::stats::{check_finite, check_len};

/// The outcome of one cell of a batch readout.
///
/// A cell that is not [`ReadStatus::Ok`] holds NaN in every one of its outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ReadStatus {
    /// A number.
    Ok = 0,
    /// The window holds no entry.
    Empty = 1,
    /// The solver refused the window as numerically singular ([`CentredFit`]).
    Singular = 2,
    /// An intermediate or the result overflowed.
    NonFinite = 3,
    /// A kNN readout asked for more neighbours than the window holds.
    TooFew = 4,
}

impl ReadStatus {
    /// The code exported to Python: 0 to 4, in the order of the variants.
    pub fn code(self) -> u8 {
        self as u8
    }

    pub(crate) fn of_error(e: &OlsError) -> Self {
        match e {
            OlsError::EmptyRange => Self::Empty,
            OlsError::NotPositiveDefinite => Self::Singular,
            // Dimensions and domains are validated before any cell is read, so the only error
            // left is an overflow.
            _ => Self::NonFinite,
        }
    }
}

/// What [`Matrix::ridge_batch`] reads for every query: W windows, R penalties `ρ`, the scale
/// `s` of `Λ = ρ·n·diag(s)`, and which optional outputs to compute.
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeGrid {
    windows: Vec<Window>,
    rhos: Vec<f64>,
    scale: Vec<f64>,
    leverage: bool,
    rss: bool,
}

impl RidgeGrid {
    /// At least one window and one penalty; every `ρ ≥ 0` and `sⱼ ≥ 0` finite.
    pub fn new(windows: &[Window], rhos: &[f64], scale: &[f64]) -> Result<Self, OlsError> {
        if windows.is_empty() || rhos.is_empty() {
            return Err(OlsError::InvalidShape(
                "a ridge batch needs at least one window and one penalty",
            ));
        }
        for &rho in rhos {
            check_rho(rho)?;
        }
        check_scale(scale)?;
        Ok(Self {
            windows: windows.to_vec(),
            rhos: rhos.to_vec(),
            scale: scale.to_vec(),
            leverage: false,
            rss: false,
        })
    }

    /// Also return the leverage `x̃ᵀ (M_xx + Λ)⁻¹ x̃` of every cell.
    pub fn with_leverage(mut self, on: bool) -> Self {
        self.leverage = on;
        self
    }

    /// Also return the window's residual sum of squares of every cell, per outcome.
    pub fn with_rss(mut self, on: bool) -> Self {
        self.rss = on;
        self
    }

    pub fn windows(&self) -> &[Window] {
        &self.windows
    }

    pub fn rhos(&self) -> &[f64] {
        &self.rhos
    }

    pub fn scale(&self) -> &[f64] {
        &self.scale
    }
}

/// The output of [`Matrix::ridge_batch`], row-major, query first.
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeBatch {
    /// M.
    pub queries: usize,
    /// W.
    pub windows: usize,
    /// R.
    pub rhos: usize,
    /// q.
    pub q: usize,
    /// `ŷ`, M × W × R × q.
    pub forecast: Vec<f64>,
    /// First entry of each window, M × W.
    pub start: Vec<usize>,
    /// One past the last entry of each window, M × W. `end − start` is `n_w`.
    pub end: Vec<usize>,
    /// M × W × R.
    pub status: Vec<ReadStatus>,
    /// `x̃ᵀ (M_xx + Λ)⁻¹ x̃`, M × W × R, if asked for.
    pub leverage: Option<Vec<f64>>,
    /// The window's residual sum of squares at the cell's penalty, M × W × R × q, if asked for.
    pub rss: Option<Vec<f64>>,
}

/// The thread count to use: `threads`, or every available core for 0, never more than the
/// chunks of work.
pub(crate) fn resolve_threads(threads: usize, chunks: usize) -> usize {
    let t = if threads == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    } else {
        threads
    };
    t.min(chunks).max(1)
}

/// `[i0, i1)` chunks of `0..m` of `size` queries each, the last one shorter.
pub(crate) fn chunk_bounds(m: usize, size: usize) -> Vec<(usize, usize)> {
    let size = size.max(1);
    (0..m.div_ceil(size))
        .map(|c| (c * size, ((c + 1) * size).min(m)))
        .collect()
}

/// `buf` cut into one slice of `(i1 − i0)·stride` values per chunk, in order. A stride of 0
/// gives empty slices, for an output that was not asked for.
pub(crate) fn split<'a, T>(
    mut buf: &'a mut [T],
    stride: usize,
    bounds: &[(usize, usize)],
) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(bounds.len());
    for &(i0, i1) in bounds {
        let (head, tail) = std::mem::take(&mut buf).split_at_mut((i1 - i0) * stride);
        out.push(head);
        buf = tail;
    }
    out
}

/// Runs `work` on every chunk, on the calling thread and `threads − 1` scoped threads, all
/// pulling chunks from a shared queue. Which thread takes which chunk varies from run to run;
/// what a chunk computes does not.
///
/// A thread the system refuses to spawn is not an error: the threads already running, the
/// calling one at least, drain the queue, and the result is the same.
pub(crate) fn run_chunks<T: Send, F: Fn(T) + Sync>(chunks: Vec<T>, threads: usize, work: F) {
    if threads <= 1 {
        chunks.into_iter().for_each(work);
        return;
    }
    let queue = Mutex::new(chunks.into_iter());
    let drain = || {
        loop {
            // The lock is held for `next` alone, which cannot panic: no poisoning.
            let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
            match next {
                Some(chunk) => work(chunk),
                None => break,
            }
        }
    };
    std::thread::scope(|s| {
        for _ in 1..threads {
            if std::thread::Builder::new().spawn_scoped(s, drain).is_err() {
                break;
            }
        }
        drain();
    });
}

/// The chunks of a ridge batch of `m` queries, about eight per thread and at least 256 queries
/// each, and the threads that run them.
fn ridge_plan(m: usize, threads: usize) -> (Vec<(usize, usize)>, usize) {
    let wanted = resolve_threads(threads, m.max(1));
    let bounds = chunk_bounds(m, (m / (wanted * 8)).max(256));
    let threads = wanted.min(bounds.len()).max(1);
    (bounds, threads)
}

/// What [`Matrix::ridge_batch`] allocates besides its outputs, in 8-byte words: an upper bound
/// for a caller who must refuse a call before it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RidgeFootprint {
    /// The threads that run.
    pub threads: usize,
    /// Every thread's factorizations at once (one per window and penalty: a p × p factor, β,
    /// the means and the RSS) and the transients of one fit (the statistics of a range, `A`,
    /// its scaled copy, the eigenvalue workspace and the solve vectors).
    pub cache_words: u128,
}

impl RidgeGrid {
    /// The [`RidgeFootprint`] of a [`Matrix::ridge_batch`] of `queries` queries for states in
    /// Rᵖ and outcomes in Rᵠ.
    pub fn footprint(&self, p: usize, q: usize, queries: usize, threads: usize) -> RidgeFootprint {
        let (_, threads) = ridge_plan(queries, threads);
        let (p, q) = (p as u128, q as u128);
        let fit = p * p + q * p + p + 2 * q;
        let transient = (p * p + q * p + q * q + p + q) + 6 * p * p + 4 * p;
        let cells = self.windows.len() as u128 * self.rhos.len() as u128;
        RidgeFootprint {
            threads,
            cache_words: threads as u128 * (cells * fit + transient),
        }
    }
}

/// The outputs of one chunk of queries.
struct RidgeChunk<'a> {
    i0: usize,
    i1: usize,
    forecast: &'a mut [f64],
    start: &'a mut [usize],
    end: &'a mut [usize],
    status: &'a mut [ReadStatus],
    leverage: &'a mut [f64],
    rss: &'a mut [f64],
}

/// One window's last range and its fits, one per penalty.
struct Cached {
    range: (usize, usize),
    fits: Vec<Result<CentredFit, ReadStatus>>,
}

impl Matrix {
    /// Reads every query under every window and penalty of `grid`: `x_q` holds M states
    /// row-major (M × p), and query i decides at `as_of[i]`.
    ///
    /// Cell `(i, w, r)` is [`Matrix::centred_query`] over [`Matrix::window_range`]`(as_of[i], w)`
    /// at penalty `ρ_r`, bit for bit. A window that is empty, refused as singular or that
    /// overflows gives a status and NaNs, never an error: only bad inputs are errors. The
    /// queries may come in any order; queries in time order with shared decision times run
    /// fastest.
    ///
    /// `threads = 0` uses every available core. The result does not depend on it.
    ///
    /// # Errors
    ///
    /// [`OlsError::Dimension`] if `x_q` is not `as_of.len() × p` or the scale not of length p,
    /// and [`OlsError::NonFinite`] for a non-finite query state.
    pub fn ridge_batch(
        &self,
        x_q: &[f64],
        as_of: &[i64],
        grid: &RidgeGrid,
        threads: usize,
    ) -> Result<RidgeBatch, OlsError> {
        let (p, q) = self.dims();
        let m = as_of.len();
        check_len(p, grid.scale.len())?;
        check_len(m * p, x_q.len())?;
        check_finite(x_q)?;
        let (nw, nr) = (grid.windows.len(), grid.rhos.len());
        let cells = m * nw * nr;
        let mut out = RidgeBatch {
            queries: m,
            windows: nw,
            rhos: nr,
            q,
            forecast: vec![f64::NAN; cells * q],
            start: vec![0; m * nw],
            end: vec![0; m * nw],
            status: vec![ReadStatus::Ok; cells],
            leverage: grid.leverage.then(|| vec![f64::NAN; cells]),
            rss: grid.rss.then(|| vec![f64::NAN; cells * q]),
        };
        let (bounds, threads) = ridge_plan(m, threads);
        let mut no_leverage = Vec::new();
        let mut no_rss = Vec::new();
        let (lev_buf, lev_stride) = match out.leverage.as_mut() {
            Some(v) => (v.as_mut_slice(), nw * nr),
            None => (no_leverage.as_mut_slice(), 0),
        };
        let (rss_buf, rss_stride) = match out.rss.as_mut() {
            Some(v) => (v.as_mut_slice(), nw * nr * q),
            None => (no_rss.as_mut_slice(), 0),
        };
        let chunks: Vec<RidgeChunk<'_>> = bounds
            .iter()
            .zip(split(&mut out.forecast, nw * nr * q, &bounds))
            .zip(split(&mut out.start, nw, &bounds))
            .zip(split(&mut out.end, nw, &bounds))
            .zip(split(&mut out.status, nw * nr, &bounds))
            .zip(split(lev_buf, lev_stride, &bounds))
            .zip(split(rss_buf, rss_stride, &bounds))
            .map(
                |((((((&(i0, i1), forecast), start), end), status), leverage), rss)| RidgeChunk {
                    i0,
                    i1,
                    forecast,
                    start,
                    end,
                    status,
                    leverage,
                    rss,
                },
            )
            .collect();
        run_chunks(chunks, threads, |chunk| {
            self.ridge_chunk(chunk, x_q, as_of, grid);
        });
        Ok(out)
    }

    fn ridge_chunk(&self, c: RidgeChunk<'_>, x_q: &[f64], as_of: &[i64], grid: &RidgeGrid) {
        let (p, q) = self.dims();
        let (nw, nr) = (grid.windows.len(), grid.rhos.len());
        let mut cache: Vec<Option<Cached>> = (0..nw).map(|_| None).collect();
        let (mut xt, mut z) = (vec![0.0; p], vec![0.0; p]);
        for (local, i) in (c.i0..c.i1).enumerate() {
            let xq = &x_q[i * p..(i + 1) * p];
            for (wi, &window) in grid.windows.iter().enumerate() {
                let range = self.window_range(as_of[i], window);
                c.start[local * nw + wi] = range.0;
                c.end[local * nw + wi] = range.1;
                let slot = &mut cache[wi];
                if slot.as_ref().is_none_or(|cached| cached.range != range) {
                    *slot = Some(self.fit_range(range, grid));
                }
                let Some(cached) = slot.as_ref() else {
                    continue;
                };
                for (ri, fit) in cached.fits.iter().enumerate() {
                    let cell = (local * nw + wi) * nr + ri;
                    let forecast = &mut c.forecast[cell * q..(cell + 1) * q];
                    let fit = match fit {
                        Ok(fit) => fit,
                        Err(status) => {
                            c.status[cell] = *status;
                            continue;
                        }
                    };
                    fit.forecast_into(xq, &mut xt, forecast);
                    let mut ok = forecast.iter().all(|v| v.is_finite());
                    let leverage = grid
                        .leverage
                        .then(|| fit.leverage_into(xq, &mut xt, &mut z));
                    ok &= leverage.is_none_or(f64::is_finite);
                    if !ok {
                        forecast.fill(f64::NAN);
                        c.status[cell] = ReadStatus::NonFinite;
                        continue;
                    }
                    if let Some(h) = leverage {
                        c.leverage[cell] = h;
                    }
                    if grid.rss {
                        c.rss[cell * q..(cell + 1) * q].copy_from_slice(fit.rss());
                    }
                }
            }
        }
    }

    /// The statistics of `range`, factored at every penalty of the grid.
    fn fit_range(&self, range: (usize, usize), grid: &RidgeGrid) -> Cached {
        let fits = match self.centred_range_stats(range.0, range.1) {
            Ok(stats) => grid
                .rhos
                .iter()
                .map(|&rho| {
                    CentredFit::fit(&stats, rho, &grid.scale).map_err(|e| ReadStatus::of_error(&e))
                })
                .collect(),
            // A window range is always within the memory; kept total rather than unwrapped.
            Err(e) => vec![Err(ReadStatus::of_error(&e)); grid.rhos.len()],
        };
        Cached { range, fits }
    }
}
