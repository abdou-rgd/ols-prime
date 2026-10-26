use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::DVector;

use crate::OlsError;
use crate::ridge::{RidgeConfig, RidgeModel, finite, residual_sum_squares, ridge_fit, solve_query};
use crate::stats::{SuffStats, check_finite, check_len};

/// Blocks hold at most 2^20 entries.
const MAX_BLOCK_LOG2: u32 = 20;

/// The next instance id. Every `Matrix`, original or clone, takes a fresh one.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A process-wide unique id. `Relaxed` is enough: the read-modify-write of one atomic is totally
/// ordered, so no two calls get the same value, and nothing else is synchronized through it.
///
/// # Panics
///
/// After 2⁶⁴ instances, rather than reuse an id.
fn fresh_id() -> u64 {
    // A compare-and-swap loop rather than `fetch_add`, which would wrap and reuse id 0.
    let mut id = NEXT_ID.load(Ordering::Relaxed);
    loop {
        let next = id
            .checked_add(1)
            .expect("2^64 Matrix instances: ids exhausted");
        match NEXT_ID.compare_exchange_weak(id, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return id,
            Err(current) => id = current,
        }
    }
}

/// Matrix: an append-only memory of `(state, outcome, time)` entries, read through a binary tree
/// of sufficient statistics (`docs/DESIGN.md` §3).
///
/// Entries are grouped into blocks of `B = 2^b`. A finished block becomes a leaf holding its
/// [`SuffStats`], and leaves roll up into parents like the carries of a binary counter. A range is
/// then read from O(log n) nodes plus the at most two partial blocks at its ends, scanned
/// directly. This is OptMem's log of summaries, with summaries that are exact.
///
/// **Causality.** An entry joins Matrix when its outcome becomes known, and `time` records that
/// moment. A reader deciding at `t` may use the entries with `time <= t` ([`Matrix::time_range`]),
/// and must push every entry known at `t` before reading at `t`.
///
/// **Which timestamp, for bars.** Times are milliseconds since the Unix epoch, UTC. An outcome
/// realized by a bar (its return, the class of its move) is pushed with that bar's
///
/// ```text
/// known_at = max(close_time, open_time + interval − 1 ms),
/// ```
///
/// which the data pipeline provides as a `known_at` column. The stored `close_time` alone is
/// wrong: on truncated bars it ends early, and once it even precedes the open. A decision taken
/// at a bar's close reads with `as_of` = that bar's `known_at`, and `time <= as_of` is inclusive
/// on purpose: the outcome realized by the decision bar itself is known at its close, and the
/// decision only executes at the next bar's open. The example on [`Matrix::time_range`] encodes
/// this on two bars.
///
/// **Identity.** Every instance has a private id, and a clone gets a fresh one. A [`Readout`]
/// remembers the id of the memory that produced it, and only that memory accepts it: a readout
/// from an independent memory with the same `p`, or from a clone that has since received other
/// entries, is rejected with [`OlsError::ForeignReadout`] instead of being read against foreign
/// entries. The id is the whole check, so a faithful clone refuses it too: query the clone.
#[derive(Debug)]
pub struct Matrix {
    id: u64,
    p: usize,
    q: usize,
    block_log2: u32,
    xs: Vec<f64>,
    ys: Vec<f64>,
    times: Vec<i64>,
    /// `levels[k][j]` holds the statistics of blocks `[j·2^k, (j+1)·2^k)`.
    levels: Vec<Vec<SuffStats>>,
}

impl Clone for Matrix {
    /// A copy of the entries and the tree under a fresh id, so that the copy and the original
    /// never accept each other's readouts once they diverge.
    fn clone(&self) -> Self {
        Self {
            id: fresh_id(),
            p: self.p,
            q: self.q,
            block_log2: self.block_log2,
            xs: self.xs.clone(),
            ys: self.ys.clone(),
            times: self.times.clone(),
            levels: self.levels.clone(),
        }
    }
}

/// A read of Matrix for one query state over a range of entries.
///
/// Only [`Matrix::query`] makes one, and its fields are read-only, so a readout always holds
/// the direction and forecast its memory computed for its range.
///
/// Equality includes the owning memory: two readouts with the same range, direction and
/// forecast, taken from two different instances (a clone included), compare unequal. To
/// compare the numbers alone, compare [`Readout::a`] and [`Readout::forecast`].
#[derive(Debug, Clone, PartialEq)]
pub struct Readout {
    owner: u64,
    start: usize,
    end: usize,
    a: DVector<f64>,
    forecast: DVector<f64>,
}

impl Readout {
    /// First entry of the range.
    pub fn start(&self) -> usize {
        self.start
    }

    /// One past the last entry of the range.
    pub fn end(&self) -> usize {
        self.end
    }

    /// The dual direction `a = (G_R + Λ)⁻¹ x_q`, from which every weight and contribution
    /// follows.
    pub fn a(&self) -> &DVector<f64> {
        &self.a
    }

    /// The forecast `ŷ = C_R a`.
    pub fn forecast(&self) -> &DVector<f64> {
        &self.forecast
    }
}

/// The one-line summary of a period: its mean outcome, its own regression and how well that
/// regression fits.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// First entry of the period.
    pub start: usize,
    /// One past the last entry of the period.
    pub end: usize,
    /// Number of entries.
    pub n: f64,
    /// Mean outcome `Σ y / n`.
    pub mean_y: DVector<f64>,
    /// The period's own ridge regression `β_R`.
    pub model: RidgeModel,
    /// Mean squared residual of `β_R` over the period, per outcome.
    pub residual_var: DVector<f64>,
}

impl Matrix {
    /// An empty memory for states in `Rᵖ` and outcomes in `Rᵠ`, with blocks of `2^block_log2`
    /// entries.
    ///
    /// Larger blocks mean fewer tree nodes, each costing p² + pq + p + 2q + 1 floats, but longer
    /// scans at the ends of a range.
    pub fn new(p: usize, q: usize, block_log2: u32) -> Result<Self, OlsError> {
        if p == 0 || q == 0 {
            return Err(OlsError::InvalidShape("p and q must be at least 1"));
        }
        if block_log2 > MAX_BLOCK_LOG2 {
            return Err(OlsError::InvalidShape("block_log2 must be at most 20"));
        }
        Ok(Self {
            id: fresh_id(),
            p,
            q,
            block_log2,
            xs: Vec::new(),
            ys: Vec::new(),
            times: Vec::new(),
            levels: Vec::new(),
        })
    }

    /// Appends an entry whose outcome `y` for state `x` became known at `time`.
    ///
    /// # Errors
    ///
    /// Wrong lengths, non-finite values, or a `time` earlier than the last entry's. A rejected
    /// entry leaves the memory unchanged.
    pub fn push(&mut self, time: i64, x: &[f64], y: &[f64]) -> Result<(), OlsError> {
        check_len(self.p, x.len())?;
        check_len(self.q, y.len())?;
        check_finite(x)?;
        check_finite(y)?;
        if let Some(&last) = self.times.last()
            && time < last
        {
            return Err(OlsError::NonMonotonicTime { last, got: time });
        }
        self.times.push(time);
        self.xs.extend_from_slice(x);
        self.ys.extend_from_slice(y);
        if self.len().is_multiple_of(self.block_size()) {
            self.close_block();
        }
        Ok(())
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether the memory holds no entry.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// `(p, q)`: the dimensions of states and outcomes.
    pub fn dims(&self) -> (usize, usize) {
        (self.p, self.q)
    }

    /// Entries per block, `B = 2^b`.
    pub fn block_size(&self) -> usize {
        1 << self.block_log2
    }

    /// State of entry `i`.
    ///
    /// # Panics
    ///
    /// If `i >= self.len()`.
    pub fn x(&self, i: usize) -> &[f64] {
        &self.xs[i * self.p..(i + 1) * self.p]
    }

    /// Outcome of entry `i`.
    ///
    /// # Panics
    ///
    /// If `i >= self.len()`.
    pub fn y(&self, i: usize) -> &[f64] {
        &self.ys[i * self.q..(i + 1) * self.q]
    }

    /// Time at which the outcome of entry `i` became known.
    ///
    /// # Panics
    ///
    /// If `i >= self.len()`.
    pub fn time(&self, i: usize) -> i64 {
        self.times[i]
    }

    /// Index range `[start, end)` of the entries with `from <= time <= as_of`: what a reader
    /// deciding at `as_of` may use, over a lookback that starts at `from`.
    ///
    /// For bars, `time` and `as_of` are both a bar's `known_at` (see [`Matrix`]). Below, hourly
    /// bar 0 realizes outcome 0 and bar 1 outcome 1; bar 1 is the truncated bar of 2020-12-21
    /// whose stored `close_time` precedes its open. The decision at the close of bar 0 sees
    /// outcome 0 only; the one at the close of bar 1 sees both, its own bar's outcome included.
    /// Stamped with its stored `close_time`, outcome 1 would claim to be known before bar 0
    /// closed, a look-ahead, and Matrix refuses it as time going backwards.
    ///
    /// ```
    /// use ols_core::{Matrix, OlsError};
    ///
    /// const HOUR: i64 = 3_600_000;
    /// let known_at = |open_time: i64, close_time: i64| close_time.max(open_time + HOUR - 1);
    /// let t0 = 1_608_559_200_000; // 2020-12-21 14:00 UTC
    /// let close1 = t0 - 759_479; // bar 1's stored close_time, 12.7 minutes before its open
    /// let bar0 = known_at(t0 - HOUR, t0 - 1); // a regular bar: its close_time
    /// let bar1 = known_at(t0, close1); // the end of its grid slot
    /// assert_eq!((bar0, bar1), (t0 - 1, t0 + HOUR - 1));
    ///
    /// let mut memory = Matrix::new(1, 1, 0)?;
    /// memory.push(bar0, &[1.0], &[0.01])?;
    /// let mut stamped_by_close_time = memory.clone();
    /// memory.push(bar1, &[1.0], &[-0.02])?;
    /// assert_eq!(memory.time_range(i64::MIN, bar0), (0, 1));
    /// assert_eq!(memory.time_range(i64::MIN, bar1), (0, 2));
    /// assert_eq!(
    ///     stamped_by_close_time.push(close1, &[1.0], &[-0.02]),
    ///     Err(OlsError::NonMonotonicTime { last: bar0, got: close1 })
    /// );
    /// # Ok::<(), ols_core::OlsError>(())
    /// ```
    pub fn time_range(&self, from: i64, as_of: i64) -> (usize, usize) {
        let start = self.times.partition_point(|&t| t < from);
        let end = self.times.partition_point(|&t| t <= as_of);
        (start, end.max(start))
    }

    /// The statistics of entries `[start, end)`.
    ///
    /// The blocks lying entirely inside the range come from the tree; the partial blocks at its
    /// ends are scanned. The decomposition depends on `(start, end)` alone, so a read is a pure
    /// function of its entries: later pushes never change it, not even in the last bit.
    pub fn range_stats(&self, start: usize, end: usize) -> Result<SuffStats, OlsError> {
        self.check_range(start, end)?;
        let b = self.block_size();
        let mut acc = SuffStats::zeros(self.p, self.q);
        let (first_block, last_block) = (start.div_ceil(b), end / b);
        if first_block >= last_block {
            self.scan_into(&mut acc, start, end);
            return Ok(acc);
        }
        self.scan_into(&mut acc, start, first_block * b);
        // Bottom-up segment-tree walk over the blocks [first_block, last_block). A left bound
        // that is a right child, or a right bound that closes a left child, is taken at this
        // level; the rest is covered by the parents one level up.
        let (mut lo, mut hi) = (first_block, last_block);
        let mut level = 0;
        while lo < hi {
            if lo % 2 == 1 {
                acc.add_stats(&self.levels[level][lo]);
                lo += 1;
            }
            if hi % 2 == 1 {
                hi -= 1;
                acc.add_stats(&self.levels[level][hi]);
            }
            lo /= 2;
            hi /= 2;
            level += 1;
        }
        self.scan_into(&mut acc, last_block * b, end);
        Ok(acc)
    }

    /// Reads entries `[start, end)` for the query state `x_q`: `a = (G_R + Λ)⁻¹ x_q` and
    /// `ŷ = C_R a`.
    pub fn query(
        &self,
        x_q: &[f64],
        start: usize,
        end: usize,
        cfg: &RidgeConfig,
    ) -> Result<Readout, OlsError> {
        let stats = self.range_stats(start, end)?;
        let a = solve_query(&stats, cfg, x_q)?;
        let forecast = finite(&stats.cross * &a)?;
        Ok(Readout {
            owner: self.id,
            start,
            end,
            a,
            forecast,
        })
    }

    /// The signed weight `wᵢ = xᵢᵀ a` of every entry in the readout's range, in order. The
    /// forecast is `Σᵢ wᵢ yᵢ`; a negative weight means the entry counts against its own outcome.
    ///
    /// # Errors
    ///
    /// [`OlsError::ForeignReadout`] if another instance produced the readout, and
    /// [`OlsError::NonFinite`] if a weight overflows.
    pub fn weights(&self, readout: &Readout) -> Result<Vec<f64>, OlsError> {
        self.check_readout(readout)?;
        let a = readout.a.as_slice();
        let weights: Vec<f64> = (readout.start..readout.end)
            .map(|i| self.x(i).iter().zip(a).map(|(x, a)| x * a).sum())
            .collect();
        check_finite(&weights)?;
        Ok(weights)
    }

    /// The exact signed contribution `C_S a = Σ_{i∈S} wᵢ yᵢ` of the entries `S = [start, end)` to
    /// the readout's forecast. The contributions of a partition of the readout's range add up to
    /// the forecast, so any period, from a single day to years, can be credited or blamed.
    ///
    /// # Errors
    ///
    /// [`OlsError::ForeignReadout`] if another instance produced the readout,
    /// [`OlsError::OutOfRange`] if `S` is not inside the readout's range, and
    /// [`OlsError::NonFinite`] if the contribution overflows.
    pub fn contribution(
        &self,
        readout: &Readout,
        start: usize,
        end: usize,
    ) -> Result<DVector<f64>, OlsError> {
        self.check_readout(readout)?;
        if !(readout.start <= start && start <= end && end <= readout.end) {
            return Err(OlsError::OutOfRange {
                start,
                end,
                bound_start: readout.start,
                bound_end: readout.end,
            });
        }
        finite(self.range_stats(start, end)?.cross * &readout.a)
    }

    /// The one-line summary of entries `[start, end)`, computed from the statistics alone.
    pub fn summarize(
        &self,
        start: usize,
        end: usize,
        cfg: &RidgeConfig,
    ) -> Result<Summary, OlsError> {
        let stats = self.range_stats(start, end)?;
        let mean_y = stats.mean_y().ok_or(OlsError::EmptyRange)?;
        let model = ridge_fit(&stats, cfg)?;
        let residual_var = residual_sum_squares(&stats, &model)? / stats.n;
        Ok(Summary {
            start,
            end,
            n: stats.n,
            mean_y,
            model,
            residual_var,
        })
    }

    /// Turns the block that just filled up into a leaf, then carries like a binary counter:
    /// whenever a level holds an even number of nodes, its last two merge into a parent.
    fn close_block(&mut self) {
        let end = self.len();
        let mut node = SuffStats::zeros(self.p, self.q);
        self.scan_into(&mut node, end - self.block_size(), end);
        for level in 0.. {
            if level == self.levels.len() {
                self.levels.push(Vec::new());
            }
            let nodes = &mut self.levels[level];
            nodes.push(node);
            if nodes.len() % 2 == 1 {
                return;
            }
            node = nodes[nodes.len() - 2].clone();
            node.add_stats(&nodes[nodes.len() - 1]);
        }
    }

    fn scan_into(&self, acc: &mut SuffStats, start: usize, end: usize) {
        for i in start..end {
            acc.add_entry(self.x(i), self.y(i), 1.0);
        }
    }

    fn check_range(&self, start: usize, end: usize) -> Result<(), OlsError> {
        if start <= end && end <= self.len() {
            Ok(())
        } else {
            Err(OlsError::OutOfRange {
                start,
                end,
                bound_start: 0,
                bound_end: self.len(),
            })
        }
    }

    /// A readout is valid only on the instance that produced it. Matrix is append-only and a
    /// read is a pure function of its range ([`Matrix::range_stats`]), so it stays valid on that
    /// instance after later pushes; the shape and range checks are then invariants, kept as
    /// cheap defence.
    fn check_readout(&self, readout: &Readout) -> Result<(), OlsError> {
        if readout.owner != self.id {
            return Err(OlsError::ForeignReadout);
        }
        check_len(self.p, readout.a.len())?;
        self.check_range(readout.start, readout.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(entries: usize, block_log2: u32) -> Matrix {
        let mut m = Matrix::new(2, 1, block_log2).unwrap();
        for i in 0..entries {
            let v = i as f64;
            m.push(i as i64, &[1.0, v], &[v.sin()]).unwrap();
        }
        m
    }

    #[test]
    fn levels_follow_the_binary_counter() {
        for blocks in 1..=37usize {
            let m = filled(blocks * 2, 1);
            let sizes: Vec<usize> = m.levels.iter().map(Vec::len).collect();
            let expected: Vec<usize> = (0..=blocks.ilog2()).map(|k| blocks >> k).collect();
            assert_eq!(sizes, expected, "{blocks} blocks");
        }
    }

    #[test]
    fn every_node_sums_its_blocks() {
        let m = filled(48, 2);
        for (level, nodes) in m.levels.iter().enumerate() {
            let span = m.block_size() << level;
            for (j, node) in nodes.iter().enumerate() {
                let mut direct = SuffStats::zeros(2, 1);
                m.scan_into(&mut direct, j * span, (j + 1) * span);
                assert_eq!(node.n, direct.n);
                assert!((&node.gram - &direct.gram).abs().max() < 1e-9);
            }
        }
    }

    #[test]
    fn bad_entries_are_rejected_without_a_trace() {
        let mut m = filled(5, 1);
        let before = (m.len(), m.levels.clone());
        assert_eq!(
            m.push(3, &[1.0, 0.0], &[0.0]),
            Err(OlsError::NonMonotonicTime { last: 4, got: 3 })
        );
        assert!(matches!(
            m.push(9, &[1.0], &[0.0]),
            Err(OlsError::Dimension { .. })
        ));
        assert_eq!(
            m.push(9, &[1.0, f64::NAN], &[0.0]),
            Err(OlsError::NonFinite)
        );
        assert_eq!((m.len(), m.levels.clone()), before);
        assert!(
            m.push(4, &[1.0, 0.0], &[0.0]).is_ok(),
            "equal times are allowed"
        );
    }

    #[test]
    fn shapes_are_validated() {
        assert!(Matrix::new(0, 1, 2).is_err());
        assert!(Matrix::new(1, 0, 2).is_err());
        assert!(Matrix::new(1, 1, MAX_BLOCK_LOG2 + 1).is_err());
        assert!(Matrix::new(1, 1, 0).is_ok());
    }

    #[test]
    fn time_range_is_inclusive_and_causal() {
        let mut m = Matrix::new(1, 1, 1).unwrap();
        for t in [10, 20, 20, 30, 40] {
            m.push(t, &[1.0], &[0.0]).unwrap();
        }
        assert_eq!(m.time_range(i64::MIN, 20), (0, 3));
        assert_eq!(m.time_range(20, 30), (1, 4));
        assert_eq!(m.time_range(21, 29), (3, 3));
        assert_eq!(m.time_range(35, 5), (4, 4), "an inverted window is empty");
        assert_eq!(
            m.time_range(i64::MIN, 9),
            (0, 0),
            "nothing is known before the first entry"
        );
    }

    #[test]
    fn reads_are_bounds_checked() {
        let m = filled(10, 1);
        let cfg = RidgeConfig::with_intercept(0.1).unwrap();
        assert!(matches!(
            m.range_stats(3, 11),
            Err(OlsError::OutOfRange { .. })
        ));
        assert!(m.range_stats(7, 6).is_err());
        let r = m.query(&[1.0, 2.0], 2, 8, &cfg).unwrap();
        assert!(m.contribution(&r, 1, 4).is_err(), "outside the readout");
        assert!(m.contribution(&r, 2, 8).is_ok());
        assert_eq!(m.summarize(4, 4, &cfg), Err(OlsError::EmptyRange));
        let other = filled(4, 1);
        assert!(other.weights(&r).is_err(), "a readout from a longer memory");
    }

    #[test]
    fn clones_get_fresh_ids() {
        let m = filled(3, 1);
        let c = m.clone();
        assert_ne!(m.id, c.id);
        assert_ne!(c.id, c.clone().id);
        assert_eq!(
            (c.xs, c.ys, c.times, c.levels),
            (m.xs, m.ys, m.times, m.levels)
        );
    }
}
