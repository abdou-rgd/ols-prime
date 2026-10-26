//! Errors for bad inputs. Every variant points at the offending item.

use thiserror::Error;

/// Why [`crate::simulate`] refused its inputs. The simulator never runs on
/// partially valid data: all inputs are validated before the first bar.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SimError {
    #[error("no bars")]
    Empty,
    /// `targets` must hold exactly one value per bar.
    #[error("{bars} bars but {targets} targets")]
    LengthMismatch { bars: usize, targets: usize },
    /// A bar with a non-finite or non-positive price, inconsistent OHLC, or
    /// a negative or non-finite volume. A volume of 0 is valid.
    #[error("invalid bar {index}: {reason}")]
    InvalidBar { index: usize, reason: &'static str },
    /// `open_time` must increase strictly. No grid is enforced: gaps and
    /// off-grid bars are accepted (see [`crate::metrics`] for why callers
    /// should still re-align series).
    #[error("open_time of bar {index} is not strictly after the previous bar's")]
    NonIncreasingTime { index: usize },
    /// A target that is not finite or lies outside [0, 1].
    #[error("invalid target {value} at {index}: must be finite and in [0, 1]")]
    InvalidTarget { index: usize, value: f64 },
    /// A [`crate::SimConfig`] field outside its domain; `field` names it.
    #[error("invalid config: {field} = {value}")]
    InvalidConfig { field: &'static str, value: f64 },
}
