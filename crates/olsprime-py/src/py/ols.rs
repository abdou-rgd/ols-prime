//! The OLS core: sufficient statistics, ridge, OLS attention, RLS and Matrix.
//!
//! States `x` are 1-D float64 arrays of length p, outcomes `y` of length q; batches are 2-D
//! arrays with one entry per row. Matrices come back C-ordered: `beta` is (q, p), attention
//! weights (queries, keys).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ols_core::OlsError as RsOlsError;
use pyo3::prelude::*;

use super::{array2, dmatrix, dvector, ols_error, rows2, vec1};
use crate::convert;

/// `x` and `p` must agree before a slice reaches `ols-core`; it checks too, but the batch
/// shapes are the bindings' to check.
fn shape_error(expected: usize, got: usize) -> PyErr {
    ols_error(RsOlsError::Dimension { expected, got })
}

/// The ridge penalty `λ ≥ 0` and the intercept convention (`ols_core::RidgeConfig`).
///
/// With `intercept=True`, coordinate 0 is unpenalized and the caller sets it to the constant 1.
#[pyclass(module = "olsprime_rs", frozen, eq)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RidgeConfig {
    inner: ols_core::RidgeConfig,
}

#[pymethods]
impl RidgeConfig {
    #[new]
    #[pyo3(signature = (lam, *, intercept = false))]
    fn new(lam: f64, intercept: bool) -> PyResult<Self> {
        let inner = if intercept {
            ols_core::RidgeConfig::with_intercept(lam)
        } else {
            ols_core::RidgeConfig::new(lam)
        }
        .map_err(ols_error)?;
        Ok(Self { inner })
    }

    /// `RidgeConfig(lam, intercept=True)`.
    #[staticmethod]
    fn with_intercept(lam: f64) -> PyResult<Self> {
        Self::new(lam, true)
    }

    /// The penalty λ.
    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lambda()
    }

    #[getter]
    fn intercept(&self) -> bool {
        self.inner.intercept()
    }

    fn __repr__(&self) -> String {
        format!(
            "RidgeConfig({:?}, intercept={})",
            self.inner.lambda(),
            if self.inner.intercept() {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Sufficient statistics of weighted entries (`ols_core::SuffStats`). Read-only from Python
/// except through `push` and `merge`, which keep every weight ≥ 0: the singularity test of
/// `ridge_fit` relies on it.
#[pyclass(module = "olsprime_rs", eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct SuffStats {
    pub(crate) inner: ols_core::SuffStats,
}

#[pymethods]
impl SuffStats {
    /// The statistics of the empty set, for states in Rᵖ and outcomes in Rᵠ.
    #[new]
    fn new(p: usize, q: usize) -> PyResult<Self> {
        if p == 0 || q == 0 {
            return Err(ols_error(RsOlsError::InvalidShape(
                "p and q must be at least 1",
            )));
        }
        Ok(Self {
            inner: ols_core::SuffStats::zeros(p, q),
        })
    }

    /// The statistics of unit-weight rows: `xs` is (n, p), `ys` is (n, q).
    #[staticmethod]
    fn from_rows(xs: PyReadonlyArray2<'_, f64>, ys: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let (n, p, xs) = rows2(&xs);
        let (m, q, ys) = rows2(&ys);
        if n != m {
            return Err(shape_error(n, m));
        }
        ols_core::SuffStats::from_rows(&xs, &ys, p, q)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// Adds one entry with weight `w ≥ 0`; a rejected entry changes nothing.
    #[pyo3(signature = (x, y, w = 1.0))]
    fn push(
        &mut self,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        w: f64,
    ) -> PyResult<()> {
        self.inner.push(&vec1(&x), &vec1(&y), w).map_err(ols_error)
    }

    /// Adds the statistics of a disjoint set. `s.merge(s)` doubles `s`.
    fn merge(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        // Copy first: `other` may be `slf` itself, which is about to be borrowed mutably.
        let other = other.borrow().inner.clone();
        slf.borrow_mut().inner.merge(&other).map_err(ols_error)
    }

    /// The statistics of the union, leaving both operands unchanged.
    fn merged(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .merged(&other.inner)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// A copy.
    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// `Σ w y / n`, or `None` for an empty set.
    fn mean_y<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.mean_y().map(|m| dvector(py, &m))
    }

    #[getter]
    fn p(&self) -> usize {
        self.inner.p()
    }

    #[getter]
    fn q(&self) -> usize {
        self.inner.q()
    }

    /// Number of entries, whatever their weights.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count
    }

    /// Total weight.
    #[getter]
    fn n(&self) -> f64 {
        self.inner.n
    }

    /// `Σ w x`, shape (p,).
    #[getter]
    fn sum_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, &self.inner.sum_x)
    }

    /// `G = Σ w x xᵀ`, shape (p, p).
    #[getter]
    fn gram<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.gram)
    }

    /// `C = Σ w y xᵀ`, shape (q, p).
    #[getter]
    fn cross<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.cross)
    }

    /// `Σ w y`, shape (q,).
    #[getter]
    fn sum_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, &self.inner.sum_y)
    }

    /// `Σ w y∘y`, shape (q,).
    #[getter]
    fn sum_yy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, &self.inner.sum_yy)
    }

    fn __repr__(&self) -> String {
        format!(
            "SuffStats(p={}, q={}, count={}, n={:?})",
            self.inner.p(),
            self.inner.q(),
            self.inner.count,
            self.inner.n
        )
    }
}

/// Primal ridge coefficients `β = C (G + Λ)⁻¹`, made by `ridge_fit`.
#[pyclass(module = "olsprime_rs", frozen, eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct RidgeModel {
    inner: ols_core::RidgeModel,
}

#[pymethods]
impl RidgeModel {
    /// β, shape (q, p).
    #[getter]
    fn beta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.beta)
    }

    /// The forecast `β x`, shape (q,).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.inner.predict(&vec1(&x)).map_err(ols_error)?;
        Ok(dvector(py, &f))
    }

    fn __repr__(&self) -> String {
        format!(
            "RidgeModel(q={}, p={})",
            self.inner.beta.nrows(),
            self.inner.beta.ncols()
        )
    }
}

/// Ridge from sufficient statistics alone.
#[pyfunction]
fn ridge_fit(stats: &SuffStats, cfg: &RidgeConfig) -> PyResult<RidgeModel> {
    ols_core::ridge_fit(&stats.inner, &cfg.inner)
        .map(|inner| RidgeModel { inner })
        .map_err(ols_error)
}

/// The dual direction `a = (G + Λ)⁻¹ x_q`, shape (p,).
#[pyfunction]
fn solve_query<'py>(
    py: Python<'py>,
    stats: &SuffStats,
    cfg: &RidgeConfig,
    x_q: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = ols_core::solve_query(&stats.inner, &cfg.inner, &vec1(&x_q)).map_err(ols_error)?;
    Ok(dvector(py, &a))
}

/// In-sample residual sum of squares of `model` over the entries behind `stats`, shape (q,).
#[pyfunction]
fn residual_sum_squares<'py>(
    py: Python<'py>,
    stats: &SuffStats,
    model: &RidgeModel,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r = ols_core::residual_sum_squares(&stats.inner, &model.inner).map_err(ols_error)?;
    Ok(dvector(py, &r))
}

/// Dense OLS attention: `weights` (m, n) and `forecast = weights @ values` (m, q).
#[pyclass(module = "olsprime_rs", frozen)]
pub struct Attention {
    inner: ols_core::Attention,
}

#[pymethods]
impl Attention {
    /// `W = X_q (XᵀX + Λ)⁻¹ Xᵀ`, shape (queries, keys): signed weights.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.weights)
    }

    /// `Ŷ = W Y`, shape (queries, q).
    #[getter]
    fn forecast<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.forecast)
    }
}

/// OLS as attention: `keys` (n, p) are past states, `values` (n, q) their outcomes and
/// `queries` (m, p) the states to forecast.
#[pyfunction]
fn ols_attention(
    keys: PyReadonlyArray2<'_, f64>,
    values: PyReadonlyArray2<'_, f64>,
    queries: PyReadonlyArray2<'_, f64>,
    cfg: &RidgeConfig,
) -> PyResult<Attention> {
    let (keys, values, queries) = (dmatrix(&keys), dmatrix(&values), dmatrix(&queries));
    // ols-core never builds a zero-width state or outcome elsewhere; numpy can (shape (n, 0)).
    if keys.ncols() == 0 || values.ncols() == 0 {
        return Err(ols_error(RsOlsError::InvalidShape(
            "p and q must be at least 1",
        )));
    }
    ols_core::ols_attention(&keys, &values, &queries, &cfg.inner)
        .map(|inner| Attention { inner })
        .map_err(ols_error)
}

/// Recursive least squares with forgetting factor γ (`ols_core::Rls`).
#[pyclass(module = "olsprime_rs")]
pub struct Rls {
    inner: ols_core::Rls,
}

#[pymethods]
impl Rls {
    /// Starts from β = 0 and P = I/λ, with λ > 0 and γ in (0, 1].
    #[new]
    #[pyo3(signature = (p, q, lam, gamma = 1.0))]
    fn new(p: usize, q: usize, lam: f64, gamma: f64) -> PyResult<Self> {
        ols_core::Rls::new(p, q, lam, gamma)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// Absorbs one observation; a rejected one changes nothing.
    fn update(
        &mut self,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        self.inner.update(&vec1(&x), &vec1(&y)).map_err(ols_error)
    }

    /// The forecast `β x`, shape (q,).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.inner.predict(&vec1(&x)).map_err(ols_error)?;
        Ok(dvector(py, &f))
    }

    /// β, shape (q, p).
    #[getter]
    fn beta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, self.inner.beta())
    }

    /// Number of observations absorbed.
    #[getter]
    fn updates(&self) -> u64 {
        self.inner.updates()
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }
}

/// A read of Matrix for one query over a range; only `Matrix.query` makes one, and only the
/// memory that made it accepts it.
#[pyclass(module = "olsprime_rs", frozen)]
pub struct Readout {
    inner: ols_core::Readout,
}

#[pymethods]
impl Readout {
    #[getter]
    fn start(&self) -> usize {
        self.inner.start()
    }

    #[getter]
    fn end(&self) -> usize {
        self.inner.end()
    }

    /// The dual direction `a = (G_R + Λ)⁻¹ x_q`, shape (p,).
    #[getter]
    fn a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, self.inner.a())
    }

    /// The forecast `C_R a`, shape (q,).
    #[getter]
    fn forecast<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, self.inner.forecast())
    }

    fn __repr__(&self) -> String {
        format!(
            "Readout(start={}, end={}, forecast={:?})",
            self.inner.start(),
            self.inner.end(),
            self.inner.forecast().as_slice()
        )
    }
}

/// The summary of a period: mean outcome, own regression and its residual variance.
#[pyclass(module = "olsprime_rs", frozen)]
pub struct Summary {
    inner: ols_core::Summary,
}

#[pymethods]
impl Summary {
    #[getter]
    fn start(&self) -> usize {
        self.inner.start
    }

    #[getter]
    fn end(&self) -> usize {
        self.inner.end
    }

    #[getter]
    fn n(&self) -> f64 {
        self.inner.n
    }

    /// Mean outcome, shape (q,).
    #[getter]
    fn mean_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, &self.inner.mean_y)
    }

    /// The period's own ridge regression.
    #[getter]
    fn model(&self) -> RidgeModel {
        RidgeModel {
            inner: self.inner.model.clone(),
        }
    }

    /// Mean squared residual of the period's regression, shape (q,).
    #[getter]
    fn residual_var<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, &self.inner.residual_var)
    }
}

/// Matrix: an append-only memory of `(time, state, outcome)` entries read through a tree of
/// sufficient statistics (`ols_core::Matrix`). Push an entry only once its outcome is known,
/// stamped with the time it became known.
#[pyclass(module = "olsprime_rs")]
pub struct Matrix {
    inner: ols_core::Matrix,
}

#[pymethods]
impl Matrix {
    /// An empty memory for states in Rᵖ and outcomes in Rᵠ, with blocks of `2**block_log2`
    /// entries. The block size only trades memory for scan length; reads are exact either way.
    #[new]
    #[pyo3(signature = (p, q, *, block_log2 = 6))]
    fn new(p: usize, q: usize, block_log2: u32) -> PyResult<Self> {
        ols_core::Matrix::new(p, q, block_log2)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// Appends one entry whose outcome `y` for state `x` became known at `time` (ms, UTC).
    fn push(
        &mut self,
        time: i64,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        self.inner
            .push(time, &vec1(&x), &vec1(&y))
            .map_err(ols_error)
    }

    /// Appends n entries at once, or none: `times` (n,), `xs` (n, p), `ys` (n, q). The error is
    /// the one the first failing `push` would raise.
    fn extend(
        &mut self,
        times: PyReadonlyArray1<'_, i64>,
        xs: PyReadonlyArray2<'_, f64>,
        ys: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let times = vec1(&times);
        let (p, q) = self.inner.dims();
        let (nx, px, xs) = rows2(&xs);
        let (ny, qy, ys) = rows2(&ys);
        for (expected, got) in [(p, px), (q, qy), (times.len(), nx), (times.len(), ny)] {
            if expected != got {
                return Err(shape_error(expected, got));
            }
        }
        convert::extend_matrix(&mut self.inner, &times, &xs, &ys).map_err(ols_error)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn p(&self) -> usize {
        self.inner.dims().0
    }

    #[getter]
    fn q(&self) -> usize {
        self.inner.dims().1
    }

    /// Entries per block.
    #[getter]
    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    /// The time of every entry, int64, in push order.
    #[getter]
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let t: Vec<i64> = (0..self.inner.len()).map(|i| self.inner.time(i)).collect();
        PyArray1::from_vec(py, t)
    }

    /// Index range `(start, end)` of the entries with `from_time <= time <= as_of`.
    fn time_range(&self, from_time: i64, as_of: i64) -> (usize, usize) {
        self.inner.time_range(from_time, as_of)
    }

    /// The statistics of entries `[start, end)`.
    fn range_stats(&self, start: usize, end: usize) -> PyResult<SuffStats> {
        self.inner
            .range_stats(start, end)
            .map(|inner| SuffStats { inner })
            .map_err(ols_error)
    }

    /// Reads entries `[start, end)` for the query state `x_q`.
    fn query(
        &self,
        x_q: PyReadonlyArray1<'_, f64>,
        start: usize,
        end: usize,
        cfg: &RidgeConfig,
    ) -> PyResult<Readout> {
        self.inner
            .query(&vec1(&x_q), start, end, &cfg.inner)
            .map(|inner| Readout { inner })
            .map_err(ols_error)
    }

    /// The signed weight of every entry of the readout's range, shape (end − start,).
    fn weights<'py>(
        &self,
        py: Python<'py>,
        readout: &Readout,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = self.inner.weights(&readout.inner).map_err(ols_error)?;
        Ok(PyArray1::from_vec(py, w))
    }

    /// The exact contribution of entries `[start, end)` to the readout's forecast, shape (q,).
    fn contribution<'py>(
        &self,
        py: Python<'py>,
        readout: &Readout,
        start: usize,
        end: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self
            .inner
            .contribution(&readout.inner, start, end)
            .map_err(ols_error)?;
        Ok(dvector(py, &c))
    }

    /// The summary of entries `[start, end)`.
    fn summarize(&self, start: usize, end: usize, cfg: &RidgeConfig) -> PyResult<Summary> {
        self.inner
            .summarize(start, end, &cfg.inner)
            .map(|inner| Summary { inner })
            .map_err(ols_error)
    }

    fn __repr__(&self) -> String {
        let (p, q) = self.inner.dims();
        format!(
            "Matrix(p={p}, q={q}, len={}, block_size={})",
            self.inner.len(),
            self.inner.block_size()
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RidgeConfig>()?;
    m.add_class::<SuffStats>()?;
    m.add_class::<RidgeModel>()?;
    m.add_class::<Attention>()?;
    m.add_class::<Rls>()?;
    m.add_class::<Readout>()?;
    m.add_class::<Summary>()?;
    m.add_class::<Matrix>()?;
    m.add_function(wrap_pyfunction!(ridge_fit, m)?)?;
    m.add_function(wrap_pyfunction!(solve_query, m)?)?;
    m.add_function(wrap_pyfunction!(residual_sum_squares, m)?)?;
    m.add_function(wrap_pyfunction!(ols_attention, m)?)?;
    Ok(())
}
