use thiserror::Error;

/// Everything that can go wrong in `ols-core`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OlsError {
    /// `G + Λ` is singular within rounding: too few entries for the penalty, or a degenerate
    /// design. See [`crate::RidgeConfig`] for the exact criterion.
    #[error("the regularized Gram matrix is not positive definite")]
    NotPositiveDefinite,
    /// The ridge penalty must be finite and non-negative; RLS needs it strictly positive.
    #[error("invalid ridge penalty {0}")]
    InvalidLambda(f64),
    /// The RLS forgetting factor must lie in (0, 1].
    #[error("invalid forgetting factor {0}: expected a value in (0, 1]")]
    InvalidForgetting(f64),
    /// An input has the wrong length.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dimension { expected: usize, got: usize },
    /// A range of entries does not lie within the bounds it must respect.
    #[error("range [{start}, {end}) is not within [{bound_start}, {bound_end})")]
    OutOfRange {
        start: usize,
        end: usize,
        bound_start: usize,
        bound_end: usize,
    },
    /// The operation needs at least one entry.
    #[error("the range holds no entry")]
    EmptyRange,
    /// Entries must arrive in time order: an outcome cannot become known before an earlier one.
    #[error("time went backwards: last entry at {last}, new entry at {got}")]
    NonMonotonicTime { last: i64, got: i64 },
    /// Entry weights must be non-negative: the singularity test of [`crate::RidgeConfig`]
    /// bounds the rounding of `G` by Cauchy–Schwarz on `Σ w|xᵢ||xⱼ|`, which needs `w ≥ 0`.
    #[error("negative weight {0}")]
    NegativeWeight(f64),
    /// A NaN or an infinity would silently poison every sum it enters.
    #[error("non-finite value in input")]
    NonFinite,
    /// A [`crate::Readout`] was given to a Matrix instance other than the one that produced it.
    #[error("the readout was produced by another Matrix instance")]
    ForeignReadout,
    /// A shape parameter is out of its domain.
    #[error("invalid shape: {0}")]
    InvalidShape(&'static str),
}
