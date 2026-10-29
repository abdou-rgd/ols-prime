//! Exact kernel readouts over a window of Matrix: Nadaraya-Watson (softmax attention) and k
//! nearest neighbours, the competitors of the OLS readout in H3 (`docs/PROTOCOL.md`).
//!
//! Distances are Euclidean on `u = x / σ`, with σ the fold's per-feature scale:
//!
//! ```text
//! dᵢ = Σⱼ (u_q,j − u_i,j)²,  summed over j in order,  u_i,j = x_i,j / σⱼ.
//! ```
//!
//! - **Softmax.** `ŷ(τ) = Σᵢ wᵢ yᵢ / Σᵢ wᵢ` with `wᵢ = exp(−(dᵢ − d_min)·c)`, `c = 1/(2τ²)`.
//!   The shift by the smallest distance keeps every weight in (0, 1] and the largest at exactly
//!   1, so nothing overflows and the sum never underflows to zero.
//! - **kNN.** The mean `y` of the k entries with the smallest `(dᵢ, i)`: ties in distance go to
//!   the lowest entry index. The k outcomes are summed in that order, then divided by k.
//!
//! Every entry of the window is visited and gets its exact distance: nothing is approximated or
//! sampled. The kNN keeps a candidate buffer that it cuts back to the `k_max` nearest whenever
//! it doubles (see `knn_update`), which drops only entries that provably cannot be among them.
//!
//! **Determinism.** A query's result is a pure function of its state, its window's entries and
//! the parameters: not of the other queries of the batch, the chunking or the thread count.
//! Entries are visited in index order, in tiles of [`TILE`] entries aligned on absolute indices.
//! The softmax keeps a running minimum, updated once per tile, and rescales its sums by
//! `exp(−(old − new)·c)` when it drops; its sums run in [`LANES`] interleaved accumulators, lane
//! `i mod LANES` for entry i, added in lane order at the end. Both depend on the entry indices
//! alone. The kNN sums its k outcomes in `(distance, index)` order, whatever order the buffer
//! held them in. The exponential is computed here ([`exp_neg`]) rather than by the platform's
//! libm, so the bits do not depend on the C runtime either; and since Rust never contracts
//! `a·b + c` into a fused multiply-add, a build for AVX2 (`-C target-cpu=x86-64-v3`) gives the
//! same bits as the SSE2 baseline (checked by the digests of `examples/matrix_bench.rs`).
//!
//! **Layout.** The batch first copies `u = x / σ` of the union of its windows into blocks of 8
//! entries (p rows of 8 values each, aligned on absolute indices), and `y` into one column per
//! outcome, so that a block's distances read one contiguous run of memory. That copy costs
//! `8·(p + q)` bytes per entry of the union.

use std::cmp::Ordering;

use crate::OlsError;
use crate::batch::{ReadStatus, chunk_bounds, resolve_threads, run_chunks, split};
use crate::matrix::{Matrix, Window};
use crate::stats::{check_finite, check_len};

/// Entries per tile. The tile of a chunk of queries is read once from memory and reused by
/// every query of the chunk.
pub const TILE: usize = 128;
/// Interleaved accumulators of the softmax sums.
pub const LANES: usize = 4;
/// Queries per chunk: one chunk streams its windows' union once.
const GROUP: usize = 128;

/// What [`Matrix::kernel_batch`] reads for every query.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    window: Window,
    sigma: Vec<f64>,
    taus: Vec<f64>,
    ks: Vec<usize>,
}

impl KernelSpec {
    /// The window, the per-feature scale σ (each finite and > 0), the softmax widths τ (each
    /// finite and > 0) and the neighbour counts k (each ≥ 1). Either list may be empty.
    pub fn new(
        window: Window,
        sigma: &[f64],
        taus: &[f64],
        ks: &[usize],
    ) -> Result<Self, OlsError> {
        if let Some(index) = sigma.iter().position(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(OlsError::InvalidSigma {
                index,
                value: sigma[index],
            });
        }
        if let Some(&tau) = taus.iter().find(|t| !(t.is_finite() && **t > 0.0)) {
            return Err(OlsError::InvalidBandwidth(tau));
        }
        if ks.contains(&0) {
            return Err(OlsError::InvalidNeighbours(0));
        }
        Ok(Self {
            window,
            sigma: sigma.to_vec(),
            taus: taus.to_vec(),
            ks: ks.to_vec(),
        })
    }

    pub fn window(&self) -> Window {
        self.window
    }

    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    pub fn taus(&self) -> &[f64] {
        &self.taus
    }

    pub fn ks(&self) -> &[usize] {
        &self.ks
    }
}

/// The output of [`Matrix::kernel_batch`], row-major, query first.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelBatch {
    /// M.
    pub queries: usize,
    /// T, the number of widths.
    pub taus: usize,
    /// K, the number of neighbour counts.
    pub ks: usize,
    /// q.
    pub q: usize,
    /// Softmax forecasts, M × T × q.
    pub softmax: Vec<f64>,
    /// M × T: [`ReadStatus::Ok`], [`ReadStatus::Empty`] or [`ReadStatus::NonFinite`].
    pub softmax_status: Vec<ReadStatus>,
    /// kNN means, M × K × q.
    pub knn: Vec<f64>,
    /// M × K: [`ReadStatus::Ok`], [`ReadStatus::Empty`], [`ReadStatus::TooFew`] or
    /// [`ReadStatus::NonFinite`].
    pub knn_status: Vec<ReadStatus>,
    /// First entry of each window, M.
    pub start: Vec<usize>,
    /// One past the last entry of each window, M.
    pub end: Vec<usize>,
}

/// A kNN candidate, ordered by `(distance, index)`.
#[derive(Debug, Clone, Copy)]
struct Cand {
    d: f64,
    i: usize,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.d.total_cmp(&other.d).then(self.i.cmp(&other.i))
    }
}

/// `e^(−x)` for `x ≥ 0`, without libm, written so that a loop over it vectorizes.
///
/// Cody-Waite reduction `−x = k·ln 2 + r` with `|r| ≤ ln 2 / 2` (fdlibm's split of ln 2, so
/// `k·ln2_hi` is exact), then the Taylor polynomial of degree 13, whose truncation error is
/// below `0.35¹⁴/14! ≈ 4·10⁻¹⁸`, and a scaling by `2^k` in two exact steps. The polynomial is
/// evaluated by Estrin's scheme, whose dependency chain is 4 multiply-adds deep instead of
/// Horner's 13, so that a vectorized loop is not bound by latency. The result is within a few
/// ulp of `e^(−x)` wherever that is a normal float (`exp_neg_matches_libm_within_a_few_ulp`);
/// below about `e^(−708)` it degrades to the absolute accuracy of the subnormals, and from
/// `x ≈ 745.1` it is 0.
#[inline(always)]
pub fn exp_neg(x: f64) -> f64 {
    const SHIFT: f64 = 6_755_399_441_055_744.0; // 1.5·2⁵²: adding it rounds to an integer
    // fdlibm's ln2_hi (32 significant bits, so k·LN2_HI is exact for |k| < 2¹¹) and ln2_lo.
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
    // 1/n! for n = 0..=13.
    const C: [f64; 14] = [
        1.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5_040.0,
        1.0 / 40_320.0,
        1.0 / 362_880.0,
        1.0 / 3_628_800.0,
        1.0 / 39_916_800.0,
        1.0 / 479_001_600.0,
        1.0 / 6_227_020_800.0,
    ];
    let y = (-x).max(-746.0);
    let kf = y * std::f64::consts::LOG2_E + SHIFT;
    // The low bits of kf hold k ∈ [−1077, 0], as a difference of bit patterns.
    let k = kf.to_bits().wrapping_sub(SHIFT.to_bits()) as i64;
    let kr = kf - SHIFT;
    let r = (y - kr * LN2_HI) - kr * LN2_LO;
    let (r2, r4) = (r * r, (r * r) * (r * r));
    let pair = |i: usize| C[i] + C[i + 1] * r;
    let low = (pair(0) + pair(2) * r2) + (pair(4) + pair(6) * r2) * r4;
    let high = (pair(8) + pair(10) * r2) + pair(12) * r4;
    let poly = low + high * (r4 * r4);
    // 2^k = 2^k1 · 2^k2 with both halves normal (k1, k2 ≥ −539).
    let k1 = -((k.wrapping_neg() as u64 >> 1) as i64);
    let k2 = k - k1;
    let s1 = f64::from_bits(((k1 + 1023) as u64) << 52);
    let s2 = f64::from_bits(((k2 + 1023) as u64) << 52);
    poly * s1 * s2
}

/// The read-only inputs shared by every chunk.
struct Shared<'a> {
    spec: &'a KernelSpec,
    /// The first entry of the first block of the union of all windows: offsets start there.
    lo: usize,
    /// Entries of the union's blocks, a multiple of [`BLOCK`].
    len: usize,
    /// `u = x / σ` of the union's blocks: block b holds p rows of [`BLOCK`] values, one per
    /// feature, so a block's distances read one contiguous run of memory.
    u: Vec<f64>,
    /// `y` of the union, one column of `len` values per outcome.
    y: Vec<f64>,
    /// `u_q` of every query, M × p.
    uq: Vec<f64>,
    /// `1/(2τ²)` per width.
    inv: Vec<f64>,
    kmax: usize,
    ranges: &'a [(usize, usize)],
}

/// The running state of one query.
struct State {
    s: usize,
    e: usize,
    /// Candidates for the `kcap` nearest: a superset of them among the entries seen so far.
    cands: Vec<Cand>,
    /// Only entries strictly nearer than this can still be among the `kcap` nearest.
    worst: f64,
    kcap: usize,
    min: f64,
    /// `[t][c][lane]`.
    acc: Vec<f64>,
    /// `[t][lane]`.
    accw: Vec<f64>,
}

/// What [`Matrix::kernel_batch`] allocates besides its outputs, in 8-byte words, so that a
/// caller who must bound every allocation (the Python bindings) can refuse a call before it
/// starts. Upper bounds, exact in their arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelFootprint {
    /// Entries of the copy of the windows' union, rounded up to whole blocks.
    pub union_entries: usize,
    /// The copy of `u = x / σ`: `union_entries · p` words.
    pub u_words: u128,
    /// The copy of `y`: `union_entries · q` words.
    pub y_words: u128,
    /// The threads that run.
    pub threads: usize,
    /// Every thread's working state at once: its chunk's candidate buffers (two words per
    /// candidate, at most `min(2·k_max, n)` per query) and softmax accumulators, plus the
    /// cumulative sums of one query.
    pub state_words: u128,
}

impl Matrix {
    /// The [`KernelFootprint`] of `self.kernel_batch(_, as_of, spec, threads)`; it reads the
    /// window ranges only.
    pub fn kernel_footprint(
        &self,
        as_of: &[i64],
        spec: &KernelSpec,
        threads: usize,
    ) -> KernelFootprint {
        let (p, q) = self.dims();
        let mut nmax = 0;
        let (mut lo, mut hi) = (usize::MAX, 0);
        for &d in as_of {
            let (s, e) = self.window_range(d, spec.window);
            if s < e {
                nmax = nmax.max(e - s);
                lo = lo.min(s);
                hi = hi.max(e);
            }
        }
        let union_entries = if lo < hi {
            (hi - lo / BLOCK * BLOCK).div_ceil(BLOCK) * BLOCK
        } else {
            0
        };
        let kcap = spec.ks.iter().copied().max().unwrap_or(0).min(nmax) as u128;
        let group = GROUP.min(as_of.len()) as u128;
        let (nt, q128) = (spec.taus.len() as u128, q as u128);
        let per_query =
            2 * (PRUNE_AT as u128 * kcap).min(nmax as u128) + nt * (q128 + 1) * LANES as u128;
        let threads = resolve_threads(threads, as_of.len().div_ceil(GROUP).max(1));
        KernelFootprint {
            union_entries,
            u_words: union_entries as u128 * p as u128,
            y_words: union_entries as u128 * q128,
            threads,
            state_words: threads as u128 * (group * per_query + kcap * q128 + q128),
        }
    }
}

/// The outputs of one chunk of queries.
struct KernelChunk<'a> {
    i0: usize,
    i1: usize,
    softmax: &'a mut [f64],
    softmax_status: &'a mut [ReadStatus],
    knn: &'a mut [f64],
    knn_status: &'a mut [ReadStatus],
}

impl Matrix {
    /// The exact softmax and kNN readouts of every query over `spec`'s window: `x_q` holds M
    /// states row-major (M × p), and query i decides at `as_of[i]`. See the module docs for the
    /// definitions and the determinism.
    ///
    /// A window with no entry gives [`ReadStatus::Empty`], and a k above the window's entry
    /// count [`ReadStatus::TooFew`], with NaNs, never a guess. `threads = 0` uses every
    /// available core; the result does not depend on it.
    ///
    /// # Errors
    ///
    /// [`OlsError::Dimension`] if `x_q` is not `as_of.len() × p` or σ not of length p, and
    /// [`OlsError::NonFinite`] for a non-finite query state, or a `x / σ` that overflows.
    pub fn kernel_batch(
        &self,
        x_q: &[f64],
        as_of: &[i64],
        spec: &KernelSpec,
        threads: usize,
    ) -> Result<KernelBatch, OlsError> {
        let (p, q) = self.dims();
        let m = as_of.len();
        check_len(p, spec.sigma.len())?;
        check_len(m * p, x_q.len())?;
        check_finite(x_q)?;
        let ranges: Vec<(usize, usize)> = as_of
            .iter()
            .map(|&d| self.window_range(d, spec.window))
            .collect();
        let (lo, hi) = union_of(&ranges);
        // Blocks of BLOCK entries aligned on absolute indices; the last one padded with zeros,
        // which no query reads back.
        let lo = lo / BLOCK * BLOCK;
        let len = (hi - lo).div_ceil(BLOCK) * BLOCK;
        let mut u = vec![0.0; p * len];
        let mut y = vec![0.0; q * len];
        for i in lo..hi {
            let k = i - lo;
            let block = (k / BLOCK) * p * BLOCK + k % BLOCK;
            for (j, (x, s)) in self.x(i).iter().zip(&spec.sigma).enumerate() {
                u[block + j * BLOCK] = x / s;
            }
            for (c, v) in self.y(i).iter().enumerate() {
                y[c * len + k] = *v;
            }
        }
        let uq: Vec<f64> = x_q
            .chunks_exact(p.max(1))
            .flat_map(|x| x.iter().zip(&spec.sigma).map(|(x, s)| x / s))
            .collect();
        check_finite(&u)?;
        check_finite(&uq)?;
        let shared = Shared {
            spec,
            lo,
            len,
            u,
            y,
            uq,
            inv: spec.taus.iter().map(|t| 1.0 / (2.0 * t * t)).collect(),
            kmax: spec.ks.iter().copied().max().unwrap_or(0),
            ranges: &ranges,
        };
        let (nt, nk) = (spec.taus.len(), spec.ks.len());
        let mut out = KernelBatch {
            queries: m,
            taus: nt,
            ks: nk,
            q,
            softmax: vec![f64::NAN; m * nt * q],
            softmax_status: vec![ReadStatus::Ok; m * nt],
            knn: vec![f64::NAN; m * nk * q],
            knn_status: vec![ReadStatus::Ok; m * nk],
            start: ranges.iter().map(|r| r.0).collect(),
            end: ranges.iter().map(|r| r.1).collect(),
        };
        let bounds = chunk_bounds(m, GROUP);
        let threads = resolve_threads(threads, bounds.len());
        let chunks: Vec<KernelChunk<'_>> = bounds
            .iter()
            .zip(split(&mut out.softmax, nt * q, &bounds))
            .zip(split(&mut out.softmax_status, nt, &bounds))
            .zip(split(&mut out.knn, nk * q, &bounds))
            .zip(split(&mut out.knn_status, nk, &bounds))
            .map(
                |((((&(i0, i1), softmax), softmax_status), knn), knn_status)| KernelChunk {
                    i0,
                    i1,
                    softmax,
                    softmax_status,
                    knn,
                    knn_status,
                },
            )
            .collect();
        run_chunks(chunks, threads, |chunk| kernel_chunk(&shared, p, q, chunk));
        Ok(out)
    }
}

/// `[min start, max end)` over the non-empty ranges, or `(0, 0)` if there is none.
fn union_of(ranges: &[(usize, usize)]) -> (usize, usize) {
    let mut iter = ranges.iter().filter(|r| r.0 < r.1);
    let Some(&first) = iter.next() else {
        return (0, 0);
    };
    iter.fold(first, |(lo, hi), &(s, e)| (lo.min(s), hi.max(e)))
}

fn kernel_chunk(sh: &Shared<'_>, p: usize, q: usize, c: KernelChunk<'_>) {
    let (nt, nk) = (sh.spec.taus.len(), sh.spec.ks.len());
    let mut states: Vec<State> = (c.i0..c.i1)
        .map(|i| {
            let (s, e) = sh.ranges[i];
            let kcap = sh.kmax.min(e - s);
            State {
                s,
                e,
                // No more candidates than entries, whatever k_max.
                cands: Vec::with_capacity((PRUNE_AT * kcap).min(e - s)),
                worst: f64::INFINITY,
                kcap,
                min: f64::INFINITY,
                acc: vec![0.0; nt * q * LANES],
                accw: vec![0.0; nt * LANES],
            }
        })
        .collect();
    let (lo_c, hi_c) = union_of(&states.iter().map(|s| (s.s, s.e)).collect::<Vec<_>>());
    let mut dist = [0.0f64; TILE];
    let mut w = [0.0f64; TILE];
    let mut tile = lo_c / TILE * TILE;
    while tile < hi_c {
        let (t0, t1) = (tile.max(lo_c), (tile + TILE).min(hi_c));
        for (local, st) in states.iter_mut().enumerate() {
            let (a, b) = (t0.max(st.s), t1.min(st.e));
            if a >= b {
                continue;
            }
            let n = b - a;
            let off = a - sh.lo;
            let uq = &sh.uq[(c.i0 + local) * p..(c.i0 + local + 1) * p];
            // Whole blocks from the one holding `a`: the tile is block-aligned, so they fit.
            let lead = off % BLOCK;
            let blocks = (lead + n).div_ceil(BLOCK);
            distances(sh, p, uq, off / BLOCK, &mut dist[..blocks * BLOCK]);
            let d = &dist[lead..lead + n];
            knn_update(st, d, a);
            if nt > 0 {
                softmax_update(st, sh, q, d, &mut w[..n], a, off);
            }
        }
        tile += TILE;
    }
    for (local, st) in states.into_iter().enumerate() {
        let n = st.e - st.s;
        let so = &mut c.softmax[local * nt * q..(local + 1) * nt * q];
        let ss = &mut c.softmax_status[local * nt..(local + 1) * nt];
        finish_softmax(&st, q, n, so, ss);
        let ko = &mut c.knn[local * nk * q..(local + 1) * nk * q];
        let ks = &mut c.knn_status[local * nk..(local + 1) * nk];
        finish_knn(st, sh, q, n, ko, ks);
    }
}

/// Entries per block of the layout of `u`, and per register block of the distance loop.
const BLOCK: usize = 8;
const _: () = assert!(TILE.is_multiple_of(BLOCK) && BLOCK.is_multiple_of(LANES));

/// The distances of the whole blocks `first..first + d.len()/BLOCK` of `u`:
/// `d[k] = Σⱼ (u_q,j − u_k,j)²` over j in order. A block's sums stay in registers across the p
/// coordinates; every sum runs in the same order as one entry at a time, so the bits are the
/// same.
fn distances(sh: &Shared<'_>, p: usize, uq: &[f64], first: usize, d: &mut [f64]) {
    for (b, out) in d.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
        let start = (first + b) * p * BLOCK;
        let block = &sh.u[start..start + p * BLOCK];
        let mut acc = [0.0f64; BLOCK];
        for (&uqj, col) in uq.iter().zip(block.as_chunks::<BLOCK>().0) {
            for (a, &v) in acc.iter_mut().zip(col) {
                let t = uqj - v;
                *a += t * t;
            }
        }
        *out = acc;
    }
}

/// A query's candidate list is cut back to its `kcap` nearest when it reaches this many times
/// `kcap`.
const PRUNE_AT: usize = 2;

/// Offers the entries `a..a + d.len()` to the query's candidates.
///
/// The candidates are appended, and cut back to the `kcap` smallest by `(distance, index)`
/// whenever they reach `PRUNE_AT·kcap` (a linear-time selection). After a cut, `worst` is the
/// distance of the `kcap`-th: an entry at that distance or more cannot enter, since entries come
/// in increasing index and lose ties, as the tie rule says. So the candidates always contain the
/// `kcap` nearest of the entries seen, and the final cut is exact.
fn knn_update(st: &mut State, d: &[f64], a: usize) {
    if st.kcap == 0 {
        return;
    }
    for (b, block) in d.chunks(BLOCK).enumerate() {
        if min_of(block) >= st.worst {
            continue;
        }
        for (l, &x) in block.iter().enumerate() {
            // Distances are never NaN, so `<` agrees with the total order.
            if x < st.worst {
                st.cands.push(Cand {
                    d: x,
                    i: a + b * BLOCK + l,
                });
                if st.cands.len() == PRUNE_AT * st.kcap {
                    prune(st);
                }
            }
        }
    }
}

/// Keeps the `kcap` smallest candidates, in no particular order, and lowers `worst` to the
/// largest of them.
fn prune(st: &mut State) {
    if st.cands.len() > st.kcap {
        st.cands.select_nth_unstable(st.kcap - 1);
        st.cands.truncate(st.kcap);
        st.worst = st.cands[st.kcap - 1].d;
    }
}

/// The smallest of distances that are never NaN, in lanes so that it vectorizes. A minimum is
/// exact in any order, so the lanes change nothing but the speed.
fn min_of(d: &[f64]) -> f64 {
    let (chunks, rest) = d.as_chunks::<LANES>();
    let mut lanes = [f64::INFINITY; LANES];
    for chunk in chunks {
        for (m, &x) in lanes.iter_mut().zip(chunk) {
            *m = if x < *m { x } else { *m };
        }
    }
    lanes
        .iter()
        .chain(rest)
        .fold(f64::INFINITY, |m, &x| if x < m { x } else { m })
}

fn softmax_update(
    st: &mut State,
    sh: &Shared<'_>,
    q: usize,
    d: &[f64],
    w: &mut [f64],
    a: usize,
    off: usize,
) {
    let tile_min = min_of(d);
    if tile_min < st.min {
        if st.min.is_finite() {
            for (t, &inv) in sh.inv.iter().enumerate() {
                let f = exp_neg((st.min - tile_min) * inv);
                for v in &mut st.acc[t * q * LANES..(t + 1) * q * LANES] {
                    *v *= f;
                }
                for v in &mut st.accw[t * LANES..(t + 1) * LANES] {
                    *v *= f;
                }
            }
        }
        st.min = tile_min;
    }
    let n = d.len();
    // Entries before the first multiple of LANES, and after the last, go to their lanes one by
    // one; the aligned body in between, LANES at a time. Each lane sees its entries in order.
    let head = ((LANES - a % LANES) % LANES).min(n);
    let body = head..head + (n - head) / LANES * LANES;
    for (t, &inv) in sh.inv.iter().enumerate() {
        let min = st.min;
        for (wk, &dk) in w.iter_mut().zip(d) {
            *wk = exp_neg((dk - min) * inv);
        }
        let accw = &mut st.accw[t * LANES..(t + 1) * LANES];
        let acc = &mut st.acc[t * q * LANES..(t + 1) * q * LANES];
        let one_by_one = |accw: &mut [f64], acc: &mut [f64], ks: std::ops::Range<usize>| {
            for k in ks {
                let lane = (a + k) % LANES;
                accw[lane] += w[k];
                for cc in 0..q {
                    acc[cc * LANES + lane] += w[k] * sh.y[cc * sh.len + off + k];
                }
            }
        };
        one_by_one(accw, acc, 0..head);
        let wbody = w[body.clone()].as_chunks::<LANES>().0;
        let mut lanes = [0.0; LANES];
        lanes.copy_from_slice(accw);
        for chunk in wbody {
            for (l, wl) in lanes.iter_mut().zip(chunk) {
                *l += wl;
            }
        }
        accw.copy_from_slice(&lanes);
        for cc in 0..q {
            let ycol = &sh.y[cc * sh.len + off + body.start..cc * sh.len + off + body.end];
            let mut lanes = [0.0; LANES];
            lanes.copy_from_slice(&acc[cc * LANES..(cc + 1) * LANES]);
            for (wc, yc) in wbody.iter().zip(ycol.as_chunks::<LANES>().0) {
                for ((l, wl), yl) in lanes.iter_mut().zip(wc).zip(yc) {
                    *l += wl * yl;
                }
            }
            acc[cc * LANES..(cc + 1) * LANES].copy_from_slice(&lanes);
        }
        one_by_one(accw, acc, body.end..n);
    }
}

fn finish_softmax(st: &State, q: usize, n: usize, out: &mut [f64], status: &mut [ReadStatus]) {
    for (t, s) in status.iter_mut().enumerate() {
        let o = &mut out[t * q..(t + 1) * q];
        if n == 0 {
            *s = ReadStatus::Empty;
            continue;
        }
        let lanes = |v: &[f64]| v.iter().fold(0.0, |acc, x| acc + x);
        let sw = lanes(&st.accw[t * LANES..(t + 1) * LANES]);
        for (cc, oc) in o.iter_mut().enumerate() {
            let base = (t * q + cc) * LANES;
            *oc = lanes(&st.acc[base..base + LANES]) / sw;
        }
        if !o.iter().all(|v| v.is_finite()) {
            o.fill(f64::NAN);
            *s = ReadStatus::NonFinite;
        }
    }
}

fn finish_knn(
    st: State,
    sh: &Shared<'_>,
    q: usize,
    n: usize,
    out: &mut [f64],
    status: &mut [ReadStatus],
) {
    let mut st = st;
    prune(&mut st);
    let mut sorted = st.cands;
    sorted.sort_unstable();
    // cum[r·q + c]: the sum of y_c over the r + 1 nearest, in order.
    let mut cum = vec![0.0; sorted.len() * q];
    let mut run = vec![0.0; q];
    for (r, cand) in sorted.iter().enumerate() {
        for (cc, acc) in run.iter_mut().enumerate() {
            *acc += sh.y[cc * sh.len + cand.i - sh.lo];
        }
        cum[r * q..(r + 1) * q].copy_from_slice(&run);
    }
    for (kk, (&k, s)) in sh.spec.ks.iter().zip(status.iter_mut()).enumerate() {
        let o = &mut out[kk * q..(kk + 1) * q];
        if n == 0 {
            *s = ReadStatus::Empty;
        } else if k > n {
            *s = ReadStatus::TooFew;
        } else {
            for (cc, oc) in o.iter_mut().enumerate() {
                *oc = cum[(k - 1) * q + cc] / k as f64;
            }
            if !o.iter().all(|v| v.is_finite()) {
                o.fill(f64::NAN);
                *s = ReadStatus::NonFinite;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_neg_matches_libm_within_a_few_ulp() {
        let mut x = 0.0f64;
        while x < 760.0 {
            let (got, want) = (exp_neg(x), (-x).exp());
            if want >= f64::MIN_POSITIVE {
                assert!(
                    (got - want).abs() <= 4.0 * f64::EPSILON * want,
                    "x = {x}: {got:e} vs {want:e}"
                );
            } else {
                assert!((got - want).abs() <= 4.0 * f64::from_bits(1), "x = {x}");
            }
            x += 0.000_731;
        }
        assert_eq!(exp_neg(0.0), 1.0);
        assert_eq!(exp_neg(-0.0), 1.0);
        assert_eq!(exp_neg(746.0), 0.0);
        assert_eq!(exp_neg(f64::INFINITY), 0.0);
        assert_eq!(exp_neg(f64::MAX), 0.0);
    }

    #[test]
    fn candidates_order_by_distance_then_index() {
        let a = Cand { d: 1.0, i: 5 };
        let b = Cand { d: 1.0, i: 7 };
        let c = Cand { d: 0.5, i: 9 };
        assert!(a < b && c < a);
        let mut v = [b, a, c];
        v.sort();
        assert_eq!(v.iter().map(|c| c.i).collect::<Vec<_>>(), [9, 5, 7]);
    }

    #[test]
    fn specs_are_validated() {
        let w = Window::ALL;
        assert!(KernelSpec::new(w, &[1.0], &[], &[]).is_ok());
        assert!(matches!(
            KernelSpec::new(w, &[1.0, 0.0], &[1.0], &[1]),
            Err(OlsError::InvalidSigma { index: 1, .. })
        ));
        assert!(matches!(
            KernelSpec::new(w, &[1.0], &[f64::NAN], &[1]),
            Err(OlsError::InvalidBandwidth(_))
        ));
        assert_eq!(
            KernelSpec::new(w, &[1.0], &[1.0], &[3, 0]),
            Err(OlsError::InvalidNeighbours(0))
        );
    }
}
