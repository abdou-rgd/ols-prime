//! Python bindings for OLS Prime: the module `olsprime_rs`.
//!
//! The bindings add no arithmetic. Every number the module returns is the one the Rust crate
//! computed, copied bit for bit into a fresh numpy array:
//!
//! - [`backtest::simulate`] and [`backtest::metrics`];
//! - the ridge, OLS-attention, RLS and Matrix types of [`ols_core`];
//! - [`risk::check_order`].
//!
//! The crate has two layers.
//!
//! - [`convert`] is plain Rust: column layouts, name tables and the atomic batch push into
//!   Matrix. `cargo test -p olsprime-py` tests it with no Python anywhere.
//! - `py` (feature `python`, which maturin enables) is the PyO3 layer: argument extraction,
//!   exceptions and classes. It is tested from Python, by `python/tests/test_rust_bindings.py`.
//!
//! The feature is off by default so that the workspace's `cargo test`, its MSRV check and
//! `cargo kani -p risk` never compile PyO3 nor link a Python shared library.

pub mod convert;

#[cfg(feature = "python")]
mod py;
