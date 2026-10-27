//! The PyO3 layer: the module, its exceptions and the numpy plumbing shared by its parts.
//!
//! Conventions, everywhere in the module:
//! - **Inputs.** Arrays are numpy arrays of the exact dtype (float64, or int64 for times) and
//!   dimensionality. Anything else, lists included, is a `TypeError` raised by rust-numpy
//!   before any value is read. Strided and reversed views are accepted and read in their
//!   logical order; byte-swapped arrays are refused by the dtype check.
//! - **Outputs.** Every array returned is a fresh copy that the caller owns, so no result can be
//!   changed through an array it handed out.
//! - **Errors.** Bad data raises `SimError` or `OlsError` (both `ValueError`s) with the Rust
//!   message, or a `ValueError` / `TypeError` from the bindings' own checks. Nothing reachable
//!   from Python panics: every `# Panics` precondition of the crates is checked here first.

mod ols;
mod risk;
mod sim;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use ols_core::nalgebra::{DMatrix, DVector};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(
    olsprime_rs,
    SimError,
    PyValueError,
    "The simulator refused its inputs (backtest::SimError); the message is Rust's."
);
create_exception!(
    olsprime_rs,
    OlsError,
    PyValueError,
    "An ols-core operation failed (ols_core::OlsError); the message is Rust's."
);

pub(crate) fn sim_error(e: backtest::SimError) -> PyErr {
    SimError::new_err(e.to_string())
}

pub(crate) fn ols_error(e: ols_core::OlsError) -> PyErr {
    OlsError::new_err(e.to_string())
}

/// The values of a 1-D array in logical order, whatever its strides.
pub(crate) fn vec1<T: numpy::Element + Copy>(a: &PyReadonlyArray1<'_, T>) -> Vec<T> {
    match a.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => a.as_array().iter().copied().collect(),
    }
}

/// `(rows, cols, values)` of a 2-D array, values in row-major logical order whatever its
/// strides.
pub(crate) fn rows2(a: &PyReadonlyArray2<'_, f64>) -> (usize, usize, Vec<f64>) {
    let v = a.as_array();
    (v.nrows(), v.ncols(), v.iter().copied().collect())
}

/// A 2-D array as a `nalgebra` matrix.
pub(crate) fn dmatrix(a: &PyReadonlyArray2<'_, f64>) -> DMatrix<f64> {
    let (r, c, data) = rows2(a);
    crate::convert::dmatrix_from_row_major(r, c, &data)
}

pub(crate) fn dvector<'py>(py: Python<'py>, v: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v.as_slice())
}

/// A `nalgebra` matrix as a C-ordered numpy array of the same shape.
pub(crate) fn array2<'py>(
    py: Python<'py>,
    m: &DMatrix<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray1::from_vec(py, crate::convert::row_major(m)).reshape([m.nrows(), m.ncols()])
}

/// The extension module.
#[pymodule]
fn olsprime_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SimError", py.get_type::<SimError>())?;
    m.add("OlsError", py.get_type::<OlsError>())?;
    sim::register(m)?;
    ols::register(m)?;
    risk::register(m)?;
    Ok(())
}
