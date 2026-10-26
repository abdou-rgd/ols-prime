//! OLS as attention.
//!
//! A ridge forecast is a weighted sum of past outcomes:
//!
//! ```text
//! ŷ = C (G + Λ)⁻¹ x_q = Σᵢ wᵢ yᵢ,     wᵢ = xᵢᵀ a,     a = (G + Λ)⁻¹ x_q,
//! ```
//!
//! with `G = Σ xxᵀ` and `C = Σ yxᵀ`. The query is `x_q`, the keys are the `xᵢ` and the values
//! the `yᵢ`; unlike softmax attention, the weights are signed. This crate implements the identity
//! three ways:
//!
//! - [`ridge_fit`], [`solve_query`] and [`ols_attention`]: the closed form, primal and dual;
//! - [`Rls`]: recursive least squares with forgetting, the online primal form;
//! - [`Matrix`]: an append-only memory whose ranges are read through a binary tree of
//!   [`SuffStats`], with exact signed attributions.
//!
//! The mathematics behind each property test is in `docs/DESIGN.md`.
//!
//! ```
//! use ols_core::{Matrix, RidgeConfig};
//!
//! // States carry a leading 1; each entry is pushed once its outcome is known.
//! let mut memory = Matrix::new(2, 1, 1)?;
//! for (t, (x, y)) in [(0.0, 0.3), (1.0, -0.1), (2.0, 0.4), (3.0, 0.2), (4.0, 0.9)]
//!     .into_iter()
//!     .enumerate()
//! {
//!     memory.push(t as i64, &[1.0, x], &[y])?;
//! }
//! let cfg = RidgeConfig::with_intercept(0.0)?;
//! let readout = memory.query(&[1.0, 6.0], 0, memory.len(), &cfg)?;
//! let weights = memory.weights(&readout)?;
//!
//! // The textbook line through the five points, read at x = 6, as a weighted sum of outcomes.
//! assert!((readout.forecast()[0] - 0.94).abs() < 1e-12);
//! // With a free intercept the weights sum to one, and the oldest entries count against
//! // their own outcomes.
//! assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! assert!(weights[0] < 0.0 && weights[1] < 0.0);
//! # Ok::<(), ols_core::OlsError>(())
//! ```

mod error;
mod matrix;
mod ridge;
mod rls;
mod stats;

pub use error::OlsError;
pub use matrix::{Matrix, Readout, Summary};
pub use nalgebra;
pub use ridge::{
    Attention, RidgeConfig, RidgeModel, ols_attention, residual_sum_squares, ridge_fit, solve_query,
};
pub use rls::Rls;
pub use stats::SuffStats;
