use nalgebra::{DMatrix, DVector, DVectorView};

use crate::OlsError;

/// The sufficient statistics of a set of weighted entries `(xᵢ, yᵢ, wᵢ)`, with `x ∈ Rᵖ` and
/// `y ∈ Rᵠ`:
///
/// ```text
/// n = Σ w,   Σ w x,   G = Σ w x xᵀ,   C = Σ w y xᵀ,   Σ w y,   Σ w y∘y,
/// ```
///
/// plus `count`, the number of entries whatever their weights.
///
/// Every field is a sum over entries, so the statistics of a union of disjoint sets are the sum
/// of their statistics ([`SuffStats::merge`]). This is what lets Matrix roll blocks up into a
/// tree and read any range with O(log n) merges. Ridge needs only `G` and `C`; the other fields
/// give a period's mean outcome and residual variance.
#[derive(Debug, Clone, PartialEq)]
pub struct SuffStats {
    /// Number of entries summed, whatever their weights. It is the length of the rounding-error
    /// chain behind every other field, which is what decides when `G + Λ` is numerically
    /// singular (see [`crate::RidgeConfig`]).
    pub count: u64,
    /// Total weight: the number of entries when every weight is 1.
    pub n: f64,
    /// `Σ w x` (length p).
    pub sum_x: DVector<f64>,
    /// `G = Σ w x xᵀ` (p × p).
    pub gram: DMatrix<f64>,
    /// `C = Σ w y xᵀ` (q × p).
    pub cross: DMatrix<f64>,
    /// `Σ w y` (length q).
    pub sum_y: DVector<f64>,
    /// `Σ w y∘y`, the element-wise squares (length q).
    pub sum_yy: DVector<f64>,
}

impl SuffStats {
    /// The statistics of the empty set.
    pub fn zeros(p: usize, q: usize) -> Self {
        Self {
            count: 0,
            n: 0.0,
            sum_x: DVector::zeros(p),
            gram: DMatrix::zeros(p, p),
            cross: DMatrix::zeros(q, p),
            sum_y: DVector::zeros(q),
            sum_yy: DVector::zeros(q),
        }
    }

    /// The statistics of unit-weight rows stored row-major: `xs` holds n × p values and `ys`
    /// n × q values.
    pub fn from_rows(xs: &[f64], ys: &[f64], p: usize, q: usize) -> Result<Self, OlsError> {
        if p == 0 || q == 0 {
            return Err(OlsError::InvalidShape("p and q must be at least 1"));
        }
        let n = xs.len() / p;
        check_len(n * p, xs.len())?;
        check_len(n * q, ys.len())?;
        check_finite(xs)?;
        check_finite(ys)?;
        let mut stats = Self::zeros(p, q);
        for (x, y) in xs.chunks_exact(p).zip(ys.chunks_exact(q)) {
            stats.add_entry(x, y, 1.0);
        }
        Ok(stats)
    }

    /// Dimension of the states.
    pub fn p(&self) -> usize {
        self.sum_x.len()
    }

    /// Dimension of the outcomes.
    pub fn q(&self) -> usize {
        self.sum_y.len()
    }

    /// Adds one entry with weight `w ≥ 0`. A weight of 0 adds nothing but the count.
    ///
    /// # Errors
    ///
    /// Wrong lengths, non-finite values, or [`OlsError::NegativeWeight`]: the singularity test of
    /// [`crate::RidgeConfig`] is sound only for statistics with non-negative weights. A rejected
    /// entry leaves the statistics unchanged.
    pub fn push(&mut self, x: &[f64], y: &[f64], w: f64) -> Result<(), OlsError> {
        check_len(self.p(), x.len())?;
        check_len(self.q(), y.len())?;
        check_finite(x)?;
        check_finite(y)?;
        check_finite(&[w])?;
        if w < 0.0 {
            return Err(OlsError::NegativeWeight(w));
        }
        self.add_entry(x, y, w);
        Ok(())
    }

    /// Adds the statistics of a set of entries disjoint from this one.
    pub fn merge(&mut self, other: &Self) -> Result<(), OlsError> {
        check_len(self.p(), other.p())?;
        check_len(self.q(), other.q())?;
        self.add_stats(other);
        Ok(())
    }

    /// The statistics of the union of two disjoint sets.
    pub fn merged(&self, other: &Self) -> Result<Self, OlsError> {
        let mut out = self.clone();
        out.merge(other)?;
        Ok(out)
    }

    /// The mean outcome `Σ w y / n`, or `None` for an empty set.
    pub fn mean_y(&self) -> Option<DVector<f64>> {
        (self.n > 0.0).then(|| &self.sum_y / self.n)
    }

    /// [`SuffStats::push`] without the checks, for entries validated upstream.
    pub(crate) fn add_entry(&mut self, x: &[f64], y: &[f64], w: f64) {
        let x = DVectorView::from_slice(x, x.len());
        let y = DVectorView::from_slice(y, y.len());
        self.count += 1;
        self.n += w;
        self.sum_x.axpy(w, &x, 1.0);
        self.gram.ger(w, &x, &x, 1.0);
        self.cross.ger(w, &y, &x, 1.0);
        self.sum_y.axpy(w, &y, 1.0);
        for (acc, v) in self.sum_yy.iter_mut().zip(y.iter()) {
            *acc += w * v * v;
        }
    }

    /// [`SuffStats::merge`] without the checks, for statistics of matching dimensions.
    pub(crate) fn add_stats(&mut self, other: &Self) {
        self.count += other.count;
        self.n += other.n;
        self.sum_x += &other.sum_x;
        self.gram += &other.gram;
        self.cross += &other.cross;
        self.sum_y += &other.sum_y;
        self.sum_yy += &other.sum_yy;
    }
}

pub(crate) fn check_len(expected: usize, got: usize) -> Result<(), OlsError> {
    if expected == got {
        Ok(())
    } else {
        Err(OlsError::Dimension { expected, got })
    }
}

pub(crate) fn check_finite(values: &[f64]) -> Result<(), OlsError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(OlsError::NonFinite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_disjoint_sets_equals_pushing_their_union() {
        let rows = [
            ([1.0, 2.0], [0.5]),
            ([1.0, -1.0], [0.25]),
            ([0.0, 3.0], [-1.0]),
        ];
        let mut left = SuffStats::zeros(2, 1);
        let mut right = SuffStats::zeros(2, 1);
        let mut all = SuffStats::zeros(2, 1);
        for (i, (x, y)) in rows.iter().enumerate() {
            let part = if i < 1 { &mut left } else { &mut right };
            part.push(x, y, 1.0).unwrap();
            all.push(x, y, 1.0).unwrap();
        }
        assert_eq!(left.merged(&right).unwrap(), all);
        assert_eq!((all.count, all.n), (3, 3.0));
        assert_eq!(all.gram[(1, 1)], 4.0 + 1.0 + 9.0);
        assert_eq!(all.cross[(0, 1)], 1.0 - 0.25 - 3.0);
    }

    #[test]
    fn bad_inputs_are_rejected() {
        let mut s = SuffStats::zeros(2, 1);
        assert_eq!(
            s.push(&[1.0], &[0.0], 1.0),
            Err(OlsError::Dimension {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            s.push(&[1.0, f64::NAN], &[0.0], 1.0),
            Err(OlsError::NonFinite)
        );
        assert_eq!(
            s.push(&[1.0, 2.0], &[0.0], f64::INFINITY),
            Err(OlsError::NonFinite)
        );
        assert_eq!(
            s.push(&[1.0, 2.0], &[0.0], -1e-300),
            Err(OlsError::NegativeWeight(-1e-300))
        );
        assert!(s.merge(&SuffStats::zeros(3, 1)).is_err());
        assert!(SuffStats::from_rows(&[1.0, 2.0, 3.0], &[0.0], 2, 1).is_err());
        assert!(SuffStats::from_rows(&[], &[], 0, 1).is_err());
        assert_eq!(s, SuffStats::zeros(2, 1), "failed pushes leave no trace");
        assert_eq!(s.mean_y(), None);
        // Zero weights stay allowed, negative zero included: they count but weigh nothing.
        s.push(&[1.0, 2.0], &[0.5], 0.0).unwrap();
        s.push(&[1.0, 2.0], &[0.5], -0.0).unwrap();
        assert_eq!((s.count, s.n), (2, 0.0));
        assert_eq!(s.gram, DMatrix::zeros(2, 2));
    }
}
