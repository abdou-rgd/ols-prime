//! The OLS core: sufficient statistics, ridge, OLS attention, RLS and Matrix.
//!
//! States `x` are 1-D float64 arrays of length p, outcomes `y` of length q; batches are 2-D
//! arrays with one entry per row. Matrices come back C-ordered: `beta` is (q, p), attention
//! weights (queries, keys).

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use ols_core::OlsError as RsOlsError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Float, Int, array2, check_alloc, dmatrix, dvector, ols_error, rows2, shape2, vec1};
use crate::convert::{self, MICROSECOND_TIMES_FROM, stats_blocks};

/// A Matrix time in microseconds, refused as `simulate` refuses a bar time: see
/// [`MICROSECOND_TIMES_FROM`].
fn microseconds(name: &str, time: i64) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{name} = {time} looks like microseconds: Matrix times are milliseconds since the \
         epoch, and every time >= 10^14 (the year 5138 in ms, 1973 in us) is refused"
    ))
}

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
    fn new(lam: Float, intercept: bool) -> PyResult<Self> {
        let lam = lam.get("lam")?;
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
    fn with_intercept(lam: Float) -> PyResult<Self> {
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
    fn new(p: Int, q: Int) -> PyResult<Self> {
        let (p, q): (usize, usize) = (p.get("p")?, q.get("q")?);
        if p == 0 || q == 0 {
            return Err(ols_error(RsOlsError::InvalidShape(
                "p and q must be at least 1",
            )));
        }
        check_alloc(|| format!("SuffStats(p={p}, q={q})"), &stats_blocks(p, q))?;
        Ok(Self {
            inner: ols_core::SuffStats::zeros(p, q),
        })
    }

    /// The statistics of unit-weight rows: `xs` is (n, p), `ys` is (n, q).
    #[staticmethod]
    fn from_rows(xs: PyReadonlyArray2<'_, f64>, ys: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let ((n, p), (m, q)) = (shape2(&xs), shape2(&ys));
        if n != m {
            return Err(shape_error(n, m));
        }
        check_alloc(
            || {
                format!(
                    "SuffStats(p={p}, q={q}) from xs of shape ({n}, {p}) and ys of shape ({m}, {q})"
                )
            },
            &stats_blocks(p, q),
        )?;
        let (_, _, xs) = rows2(&xs, "xs")?;
        let (_, _, ys) = rows2(&ys, "ys")?;
        ols_core::SuffStats::from_rows(&xs, &ys, p, q)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// Adds one entry with weight `w ≥ 0`; a rejected entry changes nothing.
    #[pyo3(signature = (x, y, w = Float::from(1.0)))]
    fn push(
        &mut self,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        w: Float,
    ) -> PyResult<()> {
        let w = w.get("w")?;
        self.inner
            .push(&vec1(&x, "x")?, &vec1(&y, "y")?, w)
            .map_err(ols_error)
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
        let f = self.inner.predict(&vec1(&x, "x")?).map_err(ols_error)?;
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
    let a =
        ols_core::solve_query(&stats.inner, &cfg.inner, &vec1(&x_q, "x_q")?).map_err(ols_error)?;
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
    let ((n, p), (nv, q), (m, pq)) = (shape2(&keys), shape2(&values), shape2(&queries));
    // ols-core never builds a zero-width state or outcome elsewhere; numpy can (shape (n, 0)).
    if p == 0 || q == 0 {
        return Err(ols_error(RsOlsError::InvalidShape(
            "p and q must be at least 1",
        )));
    }
    // ols-core's own shape checks, in its order, made here so that the sizes below are real.
    if n != nv {
        return Err(shape_error(n, nv));
    }
    if p != pq {
        return Err(shape_error(p, pq));
    }
    // What ols-core allocates from these shapes: XᵀX and its factor (p × p),
    // A = (XᵀX + Λ)⁻¹ X_qᵀ (p × m), X A and W (n × m, m × n), and Ŷ (m × q).
    for (what, block) in [
        ("the Gram matrix of keys", (p, p)),
        ("the solve of queries", (p, m)),
        ("the weights", (m, n)),
        ("the forecast", (m, q)),
    ] {
        check_alloc(
            || {
                format!(
                    "ols_attention: {what} of shape {block:?}, for keys ({n}, {p}), values \
                     ({nv}, {q}) and queries ({m}, {pq}),"
                )
            },
            &[block],
        )?;
    }
    let (keys, values, queries) = (
        dmatrix(&keys, "keys")?,
        dmatrix(&values, "values")?,
        dmatrix(&queries, "queries")?,
    );
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
    #[pyo3(signature = (p, q, lam, gamma = Float::from(1.0)))]
    fn new(p: Int, q: Int, lam: Float, gamma: Float) -> PyResult<Self> {
        let (p, q) = (p.get("p")?, q.get("q")?);
        let (lam, gamma) = (lam.get("lam")?, gamma.get("gamma")?);
        // β (q × p) and P (p × p).
        check_alloc(|| format!("Rls(p={p}, q={q})"), &[(p, p), (q, p)])?;
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
        self.inner
            .update(&vec1(&x, "x")?, &vec1(&y, "y")?)
            .map_err(ols_error)
    }

    /// The forecast `β x`, shape (q,).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let f = self.inner.predict(&vec1(&x, "x")?).map_err(ols_error)?;
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

/// The float64 blocks of one [`ols_core::CentredStats`]: `M_xx`, `M_yx`, `M_yy`, `x̄`, `ȳ`.
fn centred_blocks(p: usize, q: usize) -> [(usize, usize); 5] {
    [(p, p), (q, p), (q, q), (p, 1), (q, 1)]
}

/// Centred statistics of unit-weight entries (`ols_core::CentredStats`): the count, the means
/// `x̄`, `ȳ` and the centred sums `M_xx = Σ (x − x̄)(x − x̄)ᵀ`, `M_yx`, `M_yy`. Immutable;
/// `merged` returns the statistics of a union (Chan–Golub–LeVeque).
#[pyclass(module = "olsprime_rs", frozen, eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct CentredStats {
    inner: ols_core::CentredStats,
}

#[pymethods]
impl CentredStats {
    /// The statistics of rows `xs` (n, p) and `ys` (n, q), in one corrected two-pass scan.
    #[staticmethod]
    fn from_rows(xs: PyReadonlyArray2<'_, f64>, ys: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let ((n, p), (m, q)) = (shape2(&xs), shape2(&ys));
        if n != m {
            return Err(shape_error(n, m));
        }
        if p == 0 || q == 0 {
            return Err(ols_error(RsOlsError::InvalidShape(
                "p and q must be at least 1",
            )));
        }
        check_alloc(
            || format!("CentredStats(p={p}, q={q})"),
            &centred_blocks(p, q),
        )?;
        let (_, _, xs) = rows2(&xs, "xs")?;
        let (_, _, ys) = rows2(&ys, "ys")?;
        ols_core::CentredStats::from_rows(&xs, &ys, p, q)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// The statistics of the union of two disjoint sets of entries.
    fn merged(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .merged(&other.inner)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    #[getter]
    fn p(&self) -> usize {
        self.inner.p()
    }

    #[getter]
    fn q(&self) -> usize {
        self.inner.q()
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// The count as a float, the `n` of the formulas.
    #[getter]
    fn n(&self) -> f64 {
        self.inner.n()
    }

    /// The most entries summed by one scan behind these statistics, a term of the refusal
    /// threshold.
    #[getter]
    fn longest_scan(&self) -> u64 {
        self.inner.longest_scan()
    }

    /// The depth of the tree of merges behind these statistics (0 for one scan).
    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth()
    }

    /// `x̄`, shape (p,); zeros for the empty set.
    #[getter]
    fn mean_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.mean_x())
    }

    /// `M_xx`, shape (p, p), exactly symmetric.
    #[getter]
    fn m_xx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.m_xx())
    }

    /// `ȳ`, shape (q,).
    #[getter]
    fn mean_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.mean_y())
    }

    /// `M_yx = Σ (y − ȳ)(x − x̄)ᵀ`, shape (q, p).
    #[getter]
    fn m_yx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.m_yx())
    }

    /// `M_yy`, shape (q, q), exactly symmetric.
    #[getter]
    fn m_yy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        array2(py, &self.inner.m_yy())
    }

    fn __repr__(&self) -> String {
        format!(
            "CentredStats(p={}, q={}, count={})",
            self.inner.p(),
            self.inner.q(),
            self.inner.count()
        )
    }
}

/// A centred ridge read of Matrix for one query over a range; only `Matrix.centred_query`
/// makes one, and only the memory that made it accepts it.
#[pyclass(module = "olsprime_rs", frozen)]
pub struct CentredReadout {
    inner: ols_core::CentredReadout,
}

#[pymethods]
impl CentredReadout {
    #[getter]
    fn start(&self) -> usize {
        self.inner.start()
    }

    #[getter]
    fn end(&self) -> usize {
        self.inner.end()
    }

    /// The range's mean state `x̄`, shape (p,).
    #[getter]
    fn mean_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.mean_x())
    }

    /// The dual direction `a = (M_xx + Λ)⁻¹ (x_q − x̄)`, shape (p,).
    #[getter]
    fn a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, self.inner.a())
    }

    /// `ŷ = ȳ + β (x_q − x̄)`, shape (q,): the bits `ridge_batch` gives for the same cell.
    #[getter]
    fn forecast<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, self.inner.forecast())
    }

    /// `(x_q − x̄)ᵀ (M_xx + Λ)⁻¹ (x_q − x̄)`.
    #[getter]
    fn leverage(&self) -> f64 {
        self.inner.leverage()
    }

    /// The range's residual sum of squares at this penalty, per outcome, shape (q,).
    #[getter]
    fn rss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        dvector(py, self.inner.rss())
    }

    fn __repr__(&self) -> String {
        format!(
            "CentredReadout(start={}, end={}, forecast={:?})",
            self.inner.start(),
            self.inner.end(),
            self.inner.forecast().as_slice()
        )
    }
}

/// A window argument: `None` for every entry up to `as_of`, else the span in ms, > 0.
fn window_arg(window: Option<&Int>, name: &str) -> PyResult<ols_core::Window> {
    match window {
        None => Ok(ols_core::Window::ALL),
        Some(span) => ols_core::Window::last(span.get(name)?).map_err(ols_error),
    }
}

/// The decision times of a batch, refused at or above `MICROSECOND_TIMES_FROM` as in `push`.
fn as_of_ms(as_of: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<i64>> {
    let as_of = vec1(as_of, "as_of")?;
    if let Some((i, t)) = convert::first_microsecond_time(&as_of) {
        return Err(microseconds(&format!("as_of[{i}]"), t));
    }
    Ok(as_of)
}

/// `x_q` must be (len(as_of), p).
fn check_queries(x_q: &PyReadonlyArray2<'_, f64>, m: usize, p: usize) -> PyResult<()> {
    let (rows, cols) = shape2(x_q);
    if cols != p {
        return Err(shape_error(p, cols));
    }
    if rows != m {
        return Err(shape_error(m, rows));
    }
    Ok(())
}

/// `a · b`, saturated: a saturated count is refused by [`check_alloc`] anyway.
fn mul(a: usize, b: usize) -> usize {
    a.saturating_mul(b)
}

/// A word count from `ols-core`'s footprints as one [`check_alloc`] block.
fn words(w: u128) -> [(usize, usize); 1] {
    [(usize::try_from(w).unwrap_or(usize::MAX), 1)]
}

/// Values row-major, as a fresh numpy array of `shape`.
fn shaped<'py, T: numpy::Element>(
    py: Python<'py>,
    values: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, numpy::PyArrayDyn<T>>> {
    PyArray1::from_vec(py, values).reshape(shape)
}

fn codes(status: &[ols_core::ReadStatus]) -> Vec<u8> {
    status.iter().map(|s| s.code()).collect()
}

/// Entry indices as int64; an index is below the entry count, which fits.
fn indices(v: &[usize]) -> Vec<i64> {
    v.iter()
        .map(|&i| i64::try_from(i).unwrap_or(i64::MAX))
        .collect()
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
    ///
    /// `MAX_ALLOC_BYTES` bounds one tree node (p² + qp + p + 2q + 1 floats), not the memory:
    /// that grows by p + q floats per entry and by one node per closed block, as long as
    /// entries are pushed.
    #[new]
    #[pyo3(signature = (p, q, *, block_log2 = Int::from(6)))]
    fn new(p: Int, q: Int, block_log2: Int) -> PyResult<Self> {
        let (p, q) = (p.get("p")?, q.get("q")?);
        let block_log2 = block_log2.get("block_log2")?;
        // Every closed block adds tree nodes of this size: refuse the shape now, not at the
        // push that closes the first block.
        check_alloc(
            || format!("Matrix(p={p}, q={q}): one tree node"),
            &stats_blocks(p, q),
        )?;
        ols_core::Matrix::new(p, q, block_log2)
            .map(|inner| Self { inner })
            .map_err(ols_error)
    }

    /// Appends one entry whose outcome `y` for state `x` became known at `time` (ms, UTC). A
    /// time at or above `MICROSECOND_TIMES_FROM` is refused first, as `simulate` refuses one.
    fn push(
        &mut self,
        time: Int,
        x: PyReadonlyArray1<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let time: i64 = time.get("time")?;
        if time >= MICROSECOND_TIMES_FROM {
            return Err(microseconds("time", time));
        }
        self.inner
            .push(time, &vec1(&x, "x")?, &vec1(&y, "y")?)
            .map_err(ols_error)
    }

    /// Appends n entries at once, or none: `times` (n,), `xs` (n, p), `ys` (n, q). The shapes
    /// are checked first, then the unit of every time as in `push`; past those, the error is
    /// the one the first failing `push` would raise.
    fn extend(
        &mut self,
        times: PyReadonlyArray1<'_, i64>,
        xs: PyReadonlyArray2<'_, f64>,
        ys: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let (p, q) = self.inner.dims();
        let ((nx, px), (ny, qy)) = (shape2(&xs), shape2(&ys));
        let nt = times.len();
        for (expected, got) in [(p, px), (q, qy), (nt, nx), (nt, ny)] {
            if expected != got {
                return Err(shape_error(expected, got));
            }
        }
        let times = vec1(&times, "times")?;
        if let Some((i, t)) = convert::first_microsecond_time(&times) {
            return Err(microseconds(&format!("times[{i}]"), t));
        }
        let (_, _, xs) = rows2(&xs, "xs")?;
        let (_, _, ys) = rows2(&ys, "ys")?;
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

    /// Index range `(start, end)` of the entries with `from_time <= time <= as_of`. Both bounds
    /// are milliseconds, refused at or above `MICROSECOND_TIMES_FROM` as in `push`: an `as_of`
    /// in microseconds would silently expose every entry, a look-ahead. `i64::MIN` stays a valid
    /// `from_time`.
    fn time_range(&self, from_time: Int, as_of: Int) -> PyResult<(usize, usize)> {
        let (from_time, as_of): (i64, i64) = (from_time.get("from_time")?, as_of.get("as_of")?);
        for (name, t) in [("from_time", from_time), ("as_of", as_of)] {
            if t >= MICROSECOND_TIMES_FROM {
                return Err(microseconds(name, t));
            }
        }
        Ok(self.inner.time_range(from_time, as_of))
    }

    /// The statistics of entries `[start, end)`.
    fn range_stats(&self, start: Int, end: Int) -> PyResult<SuffStats> {
        self.inner
            .range_stats(start.get("start")?, end.get("end")?)
            .map(|inner| SuffStats { inner })
            .map_err(ols_error)
    }

    /// Reads entries `[start, end)` for the query state `x_q`.
    fn query(
        &self,
        x_q: PyReadonlyArray1<'_, f64>,
        start: Int,
        end: Int,
        cfg: &RidgeConfig,
    ) -> PyResult<Readout> {
        let (start, end) = (start.get("start")?, end.get("end")?);
        self.inner
            .query(&vec1(&x_q, "x_q")?, start, end, &cfg.inner)
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
        start: Int,
        end: Int,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (start, end) = (start.get("start")?, end.get("end")?);
        let c = self
            .inner
            .contribution(&readout.inner, start, end)
            .map_err(ols_error)?;
        Ok(dvector(py, &c))
    }

    /// The summary of entries `[start, end)`.
    fn summarize(&self, start: Int, end: Int, cfg: &RidgeConfig) -> PyResult<Summary> {
        self.inner
            .summarize(start.get("start")?, end.get("end")?, &cfg.inner)
            .map(|inner| Summary { inner })
            .map_err(ols_error)
    }

    /// Index range `(start, end)` of the entries a decision at `as_of` reads through `window`:
    /// `as_of − window < time <= as_of`, or every `time <= as_of` for `window=None`. `as_of`
    /// is refused at or above `MICROSECOND_TIMES_FROM` as in `time_range`; `window` must be
    /// > 0 ms.
    #[pyo3(signature = (as_of, window = None))]
    fn window_range(&self, as_of: Int, window: Option<Int>) -> PyResult<(usize, usize)> {
        let as_of: i64 = as_of.get("as_of")?;
        if as_of >= MICROSECOND_TIMES_FROM {
            return Err(microseconds("as_of", as_of));
        }
        let window = window_arg(window.as_ref(), "window")?;
        Ok(self.inner.window_range(as_of, window))
    }

    /// The centred statistics of entries `[start, end)`, read from the tree of centred nodes.
    fn centred_stats(&self, start: Int, end: Int) -> PyResult<CentredStats> {
        let (p, q) = self.inner.dims();
        check_alloc(
            || format!("CentredStats(p={p}, q={q})"),
            &centred_blocks(p, q),
        )?;
        self.inner
            .centred_range_stats(start.get("start")?, end.get("end")?)
            .map(|inner| CentredStats { inner })
            .map_err(ols_error)
    }

    /// The centred ridge read of entries `[start, end)` for `x_q` (p,): a free intercept and
    /// `Λ = rho · n · diag(scale)`, with `rho >= 0` and `scale` (p,) finite and >= 0.
    fn centred_query(
        &self,
        x_q: PyReadonlyArray1<'_, f64>,
        start: Int,
        end: Int,
        rho: Float,
        scale: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<CentredReadout> {
        let (start, end) = (start.get("start")?, end.get("end")?);
        let rho = rho.get("rho")?;
        let (x_q, scale) = (vec1(&x_q, "x_q")?, vec1(&scale, "scale")?);
        let (p, _) = self.inner.dims();
        if scale.len() != p {
            return Err(shape_error(p, scale.len()));
        }
        let cfg = ols_core::CentredRidge::new(rho, &scale).map_err(ols_error)?;
        self.inner
            .centred_query(&x_q, start, end, &cfg)
            .map(|inner| CentredReadout { inner })
            .map_err(ols_error)
    }

    /// The signed weight `1/n + (x_i − x̄)ᵀ a` of every entry of the readout's range, shape
    /// (end − start,). They sum to one, and `weights @ ys` is the forecast.
    fn centred_weights<'py>(
        &self,
        py: Python<'py>,
        readout: &CentredReadout,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = self
            .inner
            .centred_weights(&readout.inner)
            .map_err(ols_error)?;
        Ok(PyArray1::from_vec(py, w))
    }

    /// The exact contribution of entries `[start, end)` (inside the readout's range) to its
    /// forecast, shape (q,): the sum of their `weight · y`, from their statistics alone.
    fn centred_contribution<'py>(
        &self,
        py: Python<'py>,
        readout: &CentredReadout,
        start: Int,
        end: Int,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (start, end) = (start.get("start")?, end.get("end")?);
        let c = self
            .inner
            .centred_contribution(&readout.inner, start, end)
            .map_err(ols_error)?;
        Ok(dvector(py, &c))
    }

    /// The centred ridge read of M queries under W windows and R penalties, in parallel.
    ///
    /// `x_q` (M, p) and `as_of` (M,) int64 ms; `windows` a list of spans in ms (> 0) or `None`
    /// for all history; `rhos` (R,) and `scale` (p,) as in `centred_query`. Returns a dict:
    /// `forecast` (M, W, R, q), `status` (M, W, R) uint8, `start`, `end` and `count`
    /// (M, W) int64, and `leverage` (M, W, R) and `rss` (M, W, R, q) when asked for. A cell
    /// whose status is not `STATUS_OK` is NaN in every output. Bit-identical to
    /// `centred_query` cell by cell, whatever `threads` (0: every core). Runs without the GIL.
    #[pyo3(signature = (x_q, as_of, windows, rhos, scale, *, leverage = false, rss = false,
                        threads = Int::from(0)))]
    #[allow(clippy::too_many_arguments)]
    fn ridge_batch<'py>(
        &self,
        py: Python<'py>,
        x_q: PyReadonlyArray2<'_, f64>,
        as_of: PyReadonlyArray1<'_, i64>,
        windows: Vec<Option<Int>>,
        rhos: PyReadonlyArray1<'_, f64>,
        scale: PyReadonlyArray1<'_, f64>,
        leverage: bool,
        rss: bool,
        threads: Int,
    ) -> PyResult<Bound<'py, PyDict>> {
        let threads: usize = threads.get("threads")?;
        let (p, q) = self.inner.dims();
        let m = as_of.len();
        check_queries(&x_q, m, p)?;
        let windows = windows
            .iter()
            .enumerate()
            .map(|(i, w)| window_arg(w.as_ref(), &format!("windows[{i}]")))
            .collect::<PyResult<Vec<_>>>()?;
        let rhos = vec1(&rhos, "rhos")?;
        let scale = vec1(&scale, "scale")?;
        if scale.len() != p {
            return Err(shape_error(p, scale.len()));
        }
        let grid = ols_core::RidgeGrid::new(&windows, &rhos, &scale)
            .map_err(ols_error)?
            .with_leverage(leverage)
            .with_rss(rss);
        let (nw, nr) = (windows.len(), rhos.len());
        let what = |part: &'static str| {
            move || format!("ridge_batch: {part}, for {m} queries x {nw} windows x {nr} penalties")
        };
        let cells = mul(nw, nr);
        let mut arrays = vec![
            ("the forecast", (m, mul(cells, q))),
            ("the status", (m, cells)),
            ("each of start, end and count", (m, nw)),
        ];
        if leverage {
            arrays.push(("the leverage", (m, cells)));
        }
        if rss {
            arrays.push(("the rss", (m, mul(cells, q))));
        }
        for (part, block) in arrays {
            check_alloc(what(part), &[block])?;
        }
        let footprint = grid.footprint(p, q, m, threads);
        check_alloc(
            what("the factorizations of every thread"),
            &words(footprint.cache_words),
        )?;
        let (_, _, x_q) = rows2(&x_q, "x_q")?;
        let as_of = as_of_ms(&as_of)?;
        let inner = &self.inner;
        let b = py
            .detach(move || inner.ridge_batch(&x_q, &as_of, &grid, threads))
            .map_err(ols_error)?;
        let count: Vec<usize> = b.end.iter().zip(&b.start).map(|(e, s)| e - s).collect();
        let out = PyDict::new(py);
        out.set_item("forecast", shaped(py, b.forecast, &[m, nw, nr, q])?)?;
        out.set_item("status", shaped(py, codes(&b.status), &[m, nw, nr])?)?;
        out.set_item("start", shaped(py, indices(&b.start), &[m, nw])?)?;
        out.set_item("end", shaped(py, indices(&b.end), &[m, nw])?)?;
        out.set_item("count", shaped(py, indices(&count), &[m, nw])?)?;
        if let Some(h) = b.leverage {
            out.set_item("leverage", shaped(py, h, &[m, nw, nr])?)?;
        }
        if let Some(r) = b.rss {
            out.set_item("rss", shaped(py, r, &[m, nw, nr, q])?)?;
        }
        Ok(out)
    }

    /// The exact softmax (Nadaraya-Watson) and kNN reads of M queries over one window, in
    /// parallel.
    ///
    /// `x_q` (M, p) and `as_of` (M,) int64 ms; `window` a span in ms (> 0) or `None`; `sigma`
    /// (p,) the per-feature scale (> 0) of `u = x / sigma`; `taus` (T,) the widths (> 0) of
    /// `w = exp(−‖u_q − u_i‖² / (2 tau²))`; `ks` (K,) int64 neighbour counts (>= 1), ties to
    /// the lowest entry index. Returns a dict: `softmax` (M, T, q), `softmax_status` (M, T)
    /// uint8, `knn` (M, K, q), `knn_status` (M, K) uint8, and `start`, `end` and `count` (M,)
    /// int64. A cell whose status is not `STATUS_OK` is NaN. Bit-identical whatever `threads`
    /// (0: every core) and whatever the other queries. Runs without the GIL.
    #[pyo3(signature = (x_q, as_of, window, sigma, taus, ks, *, threads = Int::from(0)))]
    #[allow(clippy::too_many_arguments)]
    fn kernel_batch<'py>(
        &self,
        py: Python<'py>,
        x_q: PyReadonlyArray2<'_, f64>,
        as_of: PyReadonlyArray1<'_, i64>,
        window: Option<Int>,
        sigma: PyReadonlyArray1<'_, f64>,
        taus: PyReadonlyArray1<'_, f64>,
        ks: PyReadonlyArray1<'_, i64>,
        threads: Int,
    ) -> PyResult<Bound<'py, PyDict>> {
        let threads: usize = threads.get("threads")?;
        let (p, q) = self.inner.dims();
        let m = as_of.len();
        check_queries(&x_q, m, p)?;
        let window = window_arg(window.as_ref(), "window")?;
        let sigma = vec1(&sigma, "sigma")?;
        if sigma.len() != p {
            return Err(shape_error(p, sigma.len()));
        }
        let taus = vec1(&taus, "taus")?;
        let ks = vec1(&ks, "ks")?
            .iter()
            .enumerate()
            .map(|(i, &k)| match usize::try_from(k) {
                Ok(k) if k >= 1 => Ok(k),
                _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "ks[{i}] = {k}: expected a neighbour count >= 1"
                ))),
            })
            .collect::<PyResult<Vec<usize>>>()?;
        let spec = ols_core::KernelSpec::new(window, &sigma, &taus, &ks).map_err(ols_error)?;
        let (nt, nk) = (taus.len(), ks.len());
        let what = |part: &'static str| {
            move || format!("kernel_batch: {part}, for {m} queries, {nt} widths and {nk} counts")
        };
        for (part, block) in [
            ("the softmax", (m, mul(nt, q))),
            ("the softmax status", (m, nt)),
            ("the knn", (m, mul(nk, q))),
            ("the knn status", (m, nk)),
            ("each of start, end and count", (m, 1)),
            ("the window ranges", (m, 2)),
        ] {
            check_alloc(what(part), &[block])?;
        }
        let as_of = as_of_ms(&as_of)?;
        let footprint = self.inner.kernel_footprint(&as_of, &spec, threads);
        let union = footprint.union_entries;
        for (part, w) in [
            (
                "the copy of u = x / sigma over the windows' union",
                footprint.u_words,
            ),
            ("the copy of y over the windows' union", footprint.y_words),
            (
                "the candidates and sums of every thread",
                footprint.state_words,
            ),
        ] {
            check_alloc(
                move || format!("kernel_batch: {part} ({union} entries)"),
                &words(w),
            )?;
        }
        let (_, _, x_q) = rows2(&x_q, "x_q")?;
        let inner = &self.inner;
        let b = py
            .detach(move || inner.kernel_batch(&x_q, &as_of, &spec, threads))
            .map_err(ols_error)?;
        let count: Vec<usize> = b.end.iter().zip(&b.start).map(|(e, s)| e - s).collect();
        let out = PyDict::new(py);
        out.set_item("softmax", shaped(py, b.softmax, &[m, nt, q])?)?;
        out.set_item(
            "softmax_status",
            shaped(py, codes(&b.softmax_status), &[m, nt])?,
        )?;
        out.set_item("knn", shaped(py, b.knn, &[m, nk, q])?)?;
        out.set_item("knn_status", shaped(py, codes(&b.knn_status), &[m, nk])?)?;
        out.set_item("start", PyArray1::from_vec(py, indices(&b.start)))?;
        out.set_item("end", PyArray1::from_vec(py, indices(&b.end)))?;
        out.set_item("count", PyArray1::from_vec(py, indices(&count)))?;
        Ok(out)
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
    m.add_class::<CentredStats>()?;
    m.add_class::<CentredReadout>()?;
    for (name, status) in [
        ("STATUS_OK", ols_core::ReadStatus::Ok),
        ("STATUS_EMPTY", ols_core::ReadStatus::Empty),
        ("STATUS_SINGULAR", ols_core::ReadStatus::Singular),
        ("STATUS_NON_FINITE", ols_core::ReadStatus::NonFinite),
        ("STATUS_TOO_FEW", ols_core::ReadStatus::TooFew),
    ] {
        m.add(name, status.code())?;
    }
    m.add_function(wrap_pyfunction!(ridge_fit, m)?)?;
    m.add_function(wrap_pyfunction!(solve_query, m)?)?;
    m.add_function(wrap_pyfunction!(residual_sum_squares, m)?)?;
    m.add_function(wrap_pyfunction!(ols_attention, m)?)?;
    Ok(())
}
