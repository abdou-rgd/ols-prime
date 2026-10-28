//! The PyO3 layer: the module, its exceptions and the numpy plumbing shared by its parts.
//!
//! Conventions, everywhere in the module:
//! - **Inputs.** Arrays are numpy arrays of the exact dtype (float64, or int64 for times) and
//!   dimensionality. Anything else, lists included, is a `TypeError` raised by rust-numpy
//!   before any value is read. Strided and reversed views are accepted and read in their
//!   logical order; byte-swapped arrays are refused by the dtype check.
//! - **Outputs.** Every array returned is a fresh copy that the caller owns, so no result can be
//!   changed through an array it handed out.
//! - **Scalars.** Integer arguments are [`Int`]s and float arguments [`Float`]s, never PyO3's
//!   built-in `usize`, `i64` or `f64` extraction: those raise `OverflowError` for an `int`
//!   outside the Rust type, which is not a `ValueError`. A `bool` is refused for both.
//! - **Errors.** Bad data raises `SimError` or `OlsError` (both `ValueError`s) with the Rust
//!   message, or a `ValueError` / `TypeError` from the bindings' own checks, naming the
//!   argument. Nothing else is raised for bad input, `OverflowError` included. Nothing
//!   reachable from Python panics: every `# Panics` precondition of the crates is checked here
//!   first.
//! - **Sizes.** A failed allocation aborts the process rather than raising, so every allocation
//!   whose size a caller chooses (the p × p and q × p statistics, the attention matrices, and
//!   the copy of every input array, whose logical size a zero-stride view makes free) is sized
//!   first in exact arithmetic. Above [`MAX_ALLOC_BYTES`] it is a `ValueError` naming the
//!   argument, raised before anything is allocated.

mod ols;
mod risk;
mod sim;

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use ols_core::nalgebra::{DMatrix, DVector};
use pyo3::create_exception;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::convert::{self, MAX_ALLOC_BYTES, MICROSECOND_TIMES_FROM};

/// The Rust integer types an [`Int`] converts to, with their ranges for the error message.
pub(crate) trait RustInt: TryFrom<i128> {
    const MIN: i128;
    const MAX: i128;
}

macro_rules! rust_int {
    ($($t:ty),*) => {$(
        impl RustInt for $t {
            // Widening casts: every value of these types is an i128.
            const MIN: i128 = <$t>::MIN as i128;
            const MAX: i128 = <$t>::MAX as i128;
        }
    )*};
}
rust_int!(i64, u32, u64, usize);

/// The type name of `obj`, for a `TypeError`.
fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |n| n.to_string())
}

/// An integer argument as Python passed it: any `int`, or any object with `__index__`
/// (numpy's integers included), but not a `bool`.
///
/// Extraction never fails. The check against the Rust type is [`Int::get`]'s, which knows the
/// argument's name: PyO3 0.30 attaches the name of a failed argument as an exception *note*,
/// which `str(e)` does not show, and keeps its `OverflowError`.
#[derive(Debug, Clone)]
pub(crate) enum Int {
    /// Every Python integer of magnitude below 2¹²⁷: wider than every target type.
    Exact(i128),
    /// An integer beyond ±2¹²⁷, of this many bits.
    Huge { negative: bool, bits: u64 },
    /// Not an integer: the object's type name.
    WrongType(String),
}

impl Int {
    /// The value as a `T`, or a `ValueError` (out of range) or `TypeError` (not an int) naming
    /// `name`.
    pub(crate) fn get<T: RustInt>(&self, name: &str) -> PyResult<T> {
        let range = || format!("[{}, {}]", T::MIN, T::MAX);
        match *self {
            Self::Exact(v) => T::try_from(v).map_err(|_| {
                PyValueError::new_err(format!("{name} = {v} is out of range {}", range()))
            }),
            Self::Huge { negative, bits } => Err(PyValueError::new_err(format!(
                "{name} is out of range {}: got a {} int of {bits} bits",
                range(),
                if negative { "negative" } else { "positive" }
            ))),
            Self::WrongType(ref ty) => Err(PyTypeError::new_err(format!(
                "{name} must be an int, got {ty}"
            ))),
        }
    }
}

/// `bit_length()` of a Python int, or 0 if it cannot be read.
fn bit_length(obj: &Bound<'_, PyAny>) -> u64 {
    obj.call_method0("__index__")
        .and_then(|i| i.call_method0("bit_length"))
        .and_then(|b| b.extract())
        .unwrap_or(0)
}

/// Whether `obj < 0`, for an object that is known to be an integer.
fn is_negative(obj: &Bound<'_, PyAny>) -> bool {
    obj.lt(0).unwrap_or(false)
}

impl<'a, 'py> FromPyObject<'a, 'py> for Int {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj: &Bound<'py, PyAny> = &obj;
        if obj.is_instance_of::<PyBool>() {
            return Ok(Self::WrongType("bool".to_owned()));
        }
        Ok(match obj.extract::<i128>() {
            Ok(v) => Self::Exact(v),
            Err(e) if e.is_instance_of::<PyOverflowError>(obj.py()) => Self::Huge {
                negative: is_negative(obj),
                bits: bit_length(obj),
            },
            Err(_) => Self::WrongType(type_name(obj)),
        })
    }
}

impl From<i128> for Int {
    /// A default value in a `#[pyo3(signature)]`.
    fn from(v: i128) -> Self {
        Self::Exact(v)
    }
}

/// A float argument as Python passed it: a `float`, an `int`, or any object with `__float__`
/// (numpy's floats included), but not a `bool`. Like [`Int`], extraction never fails and
/// [`Float::get`] names the argument: an `int` too large for a float is a `ValueError`, not
/// PyO3's `OverflowError`.
#[derive(Debug, Clone)]
pub(crate) enum Float {
    Value(f64),
    /// An `int` beyond the largest finite float, of this many bits.
    Huge {
        bits: u64,
    },
    /// Not a number: the object's type name.
    WrongType(String),
}

impl Float {
    /// The value, or a `ValueError` (out of range) or `TypeError` (not a number) naming `name`.
    /// NaN and infinities pass: their domain is the callee's to check.
    pub(crate) fn get(&self, name: &str) -> PyResult<f64> {
        match *self {
            Self::Value(v) => Ok(v),
            Self::Huge { bits } => Err(PyValueError::new_err(format!(
                "{name} is out of range for a float: got an int of {bits} bits"
            ))),
            Self::WrongType(ref ty) => Err(PyTypeError::new_err(format!(
                "{name} must be a float, got {ty}"
            ))),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Float {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj: &Bound<'py, PyAny> = &obj;
        if obj.is_instance_of::<PyBool>() {
            return Ok(Self::WrongType("bool".to_owned()));
        }
        Ok(match obj.extract::<f64>() {
            Ok(v) => Self::Value(v),
            Err(e) if e.is_instance_of::<PyOverflowError>(obj.py()) => Self::Huge {
                bits: bit_length(obj),
            },
            Err(_) => Self::WrongType(type_name(obj)),
        })
    }
}

impl From<f64> for Float {
    /// A default value in a `#[pyo3(signature)]`.
    fn from(v: f64) -> Self {
        Self::Value(v)
    }
}

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

/// [`convert::check_alloc`] as a `ValueError`.
pub(crate) fn check_alloc(
    what: impl FnOnce() -> String,
    blocks: &[(usize, usize)],
) -> PyResult<()> {
    convert::check_alloc(what, blocks).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// 8-byte words per value of `T`, for [`check_alloc`].
const fn words<T>() -> usize {
    size_of::<T>().div_ceil(8)
}

/// The values of a 1-D array in logical order, whatever its strides, once the copy is known to
/// fit [`MAX_ALLOC_BYTES`]; `name` is the argument's, for the error.
pub(crate) fn vec1<T: numpy::Element + Copy>(
    a: &PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<Vec<T>> {
    let n = a.len();
    check_alloc(|| format!("{name} ({n} values)"), &[(n, words::<T>())])?;
    Ok(match a.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => a.as_array().iter().copied().collect(),
    })
}

/// The shape of a 2-D array, read without copying.
pub(crate) fn shape2(a: &PyReadonlyArray2<'_, f64>) -> (usize, usize) {
    let s = a.shape();
    (s[0], s[1])
}

/// `(rows, cols, values)` of a 2-D array, values in row-major logical order whatever its
/// strides, once the copy is known to fit [`MAX_ALLOC_BYTES`].
pub(crate) fn rows2(
    a: &PyReadonlyArray2<'_, f64>,
    name: &str,
) -> PyResult<(usize, usize, Vec<f64>)> {
    let (r, c) = shape2(a);
    check_alloc(|| format!("{name} of shape ({r}, {c})"), &[(r, c)])?;
    let v = a.as_array();
    Ok((r, c, v.iter().copied().collect()))
}

/// A 2-D array as a `nalgebra` matrix (a second copy of the same size as [`rows2`]'s).
pub(crate) fn dmatrix(a: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<DMatrix<f64>> {
    let (r, c, data) = rows2(a, name)?;
    Ok(convert::dmatrix_from_row_major(r, c, &data))
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
    m.add("MAX_ALLOC_BYTES", MAX_ALLOC_BYTES)?;
    m.add("MICROSECOND_TIMES_FROM", MICROSECOND_TIMES_FROM)?;
    sim::register(m)?;
    ols::register(m)?;
    risk::register(m)?;
    Ok(())
}
