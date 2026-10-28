//! `simulate`, its result, its metrics and the fee schedules.

use backtest::metrics as bm;
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use super::{Float, Int, check_alloc, sim_error, vec1};
use crate::convert::{self, BarColumns, FillColumns};

/// A venue's spot fee schedule. Built-in schedules come from `FeeSchedule.named` or the class
/// attributes; a custom one is `FeeSchedule(name, maker, taker)`.
#[pyclass(module = "olsprime_rs", frozen, eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct FeeSchedule {
    name: String,
    /// What the simulator receives. A custom schedule carries [`convert::CUSTOM_FEE_NAME`]
    /// there; the name plays no part in the simulation.
    inner: backtest::FeeSchedule,
}

impl FeeSchedule {
    fn builtin(inner: backtest::FeeSchedule) -> Self {
        Self {
            name: inner.name.to_owned(),
            inner,
        }
    }

    /// `fees` as `simulate` accepts it: a `FeeSchedule` or a built-in name.
    fn resolve(fees: &Bound<'_, PyAny>) -> PyResult<backtest::FeeSchedule> {
        if let Ok(f) = fees.cast::<FeeSchedule>() {
            return Ok(f.get().inner);
        }
        if let Ok(name) = fees.cast::<PyString>() {
            return Ok(Self::named(name.to_str()?)?.inner);
        }
        Err(PyTypeError::new_err(format!(
            "fees must be a FeeSchedule or the name of one, got {}",
            fees.get_type().name()?
        )))
    }
}

#[pymethods]
impl FeeSchedule {
    /// A custom schedule. Rates are fractions of notional in [0, 1); built-in names are
    /// reserved.
    #[new]
    fn new(name: String, maker: Float, taker: Float) -> PyResult<Self> {
        let inner = convert::custom_fee(&name, maker.get("maker")?, taker.get("taker")?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { name, inner })
    }

    /// The built-in schedule with this exact name.
    #[staticmethod]
    fn named(name: &str) -> PyResult<Self> {
        backtest::FeeSchedule::named(name)
            .map(Self::builtin)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown fee schedule {name:?}; expected one of {:?}",
                    convert::fee_names()
                ))
            })
    }

    /// The names of the built-in schedules.
    #[staticmethod]
    fn names() -> Vec<&'static str> {
        convert::fee_names()
    }

    #[classattr]
    #[pyo3(name = "ZERO")]
    fn zero() -> Self {
        Self::builtin(backtest::FeeSchedule::ZERO)
    }

    #[classattr]
    #[pyo3(name = "BINANCE_SPOT")]
    fn binance_spot() -> Self {
        Self::builtin(backtest::FeeSchedule::BINANCE_SPOT)
    }

    #[classattr]
    #[pyo3(name = "BINANCE_SPOT_BNB")]
    fn binance_spot_bnb() -> Self {
        Self::builtin(backtest::FeeSchedule::BINANCE_SPOT_BNB)
    }

    #[classattr]
    #[pyo3(name = "COINBASE_ADVANCED_ENTRY")]
    fn coinbase_advanced_entry() -> Self {
        Self::builtin(backtest::FeeSchedule::COINBASE_ADVANCED_ENTRY)
    }

    #[classattr]
    #[pyo3(name = "KRAKEN_PRO_ENTRY")]
    fn kraken_pro_entry() -> Self {
        Self::builtin(backtest::FeeSchedule::KRAKEN_PRO_ENTRY)
    }

    #[classattr]
    #[pyo3(name = "BITVAVO_ENTRY")]
    fn bitvavo_entry() -> Self {
        Self::builtin(backtest::FeeSchedule::BITVAVO_ENTRY)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn maker(&self) -> f64 {
        self.inner.maker
    }

    #[getter]
    fn taker(&self) -> f64 {
        self.inner.taker
    }

    /// The rate charged for `liquidity`, `"taker"` or `"maker"`.
    fn rate(&self, liquidity: &str) -> PyResult<f64> {
        Ok(self.inner.rate(liquidity_arg(liquidity)?))
    }

    fn __repr__(&self) -> String {
        format!(
            "FeeSchedule({:?}, maker={:?}, taker={:?})",
            self.name, self.inner.maker, self.inner.taker
        )
    }
}

fn liquidity_arg(name: &str) -> PyResult<backtest::Liquidity> {
    convert::parse_liquidity(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown liquidity {name:?}; expected \"taker\" or \"maker\""
        ))
    })
}

/// The output of `simulate`. It holds the Rust result itself, so `metrics` reads exactly what
/// the simulator produced; every array property returns a fresh copy.
#[pyclass(module = "olsprime_rs", frozen)]
pub struct SimResult {
    inner: backtest::SimResult,
}

#[pymethods]
impl SimResult {
    /// Equity at each bar's close, after any trade at its open (quote).
    #[getter]
    fn equity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.equity)
    }

    /// Base units held at each bar's close.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.position)
    }

    /// Cash at each bar's close (quote).
    #[getter]
    fn cash<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.cash)
    }

    /// The fills as columns.
    #[getter]
    fn fills(&self) -> Fills {
        Fills {
            cols: FillColumns::from_fills(&self.inner.fills),
        }
    }

    /// Sum of the fills' fees, in fill order (quote).
    #[getter]
    fn fees_paid(&self) -> f64 {
        self.inner.fees_paid
    }

    /// Number of bars.
    fn __len__(&self) -> usize {
        self.inner.equity.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SimResult(bars={}, fills={}, fees_paid={:?})",
            self.inner.equity.len(),
            self.inner.fills.len(),
            self.inner.fees_paid
        )
    }
}

/// The fills of a simulation, one numpy column per field. `side` is `BUY` (+1) or `SELL` (−1),
/// so `side * qty` is the signed change of the position.
#[pyclass(module = "olsprime_rs", frozen)]
pub struct Fills {
    cols: FillColumns,
}

#[pymethods]
impl Fills {
    /// Index of the bar whose open executed the fill (int64).
    #[getter]
    fn bar<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.cols.bar)
    }

    /// `open_time` of that bar, ms since the epoch (int64).
    #[getter]
    fn time<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.cols.time)
    }

    /// `BUY` (+1) or `SELL` (−1), int8.
    #[getter]
    fn side<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        PyArray1::from_slice(py, &self.cols.side)
    }

    #[getter]
    fn qty<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.cols.qty)
    }

    #[getter]
    fn price<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.cols.price)
    }

    #[getter]
    fn notional<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.cols.notional)
    }

    #[getter]
    fn fee<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.cols.fee)
    }

    /// The columns as a dict of arrays, ready for `polars.DataFrame` or `pandas.DataFrame`.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("bar", self.bar(py))?;
        d.set_item("time", self.time(py))?;
        d.set_item("side", self.side(py))?;
        d.set_item("qty", self.qty(py))?;
        d.set_item("price", self.price(py))?;
        d.set_item("notional", self.notional(py))?;
        d.set_item("fee", self.fee(py))?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.cols.len()
    }

    fn __repr__(&self) -> String {
        format!("Fills(len={})", self.cols.len())
    }
}

/// Runs a long/flat spot strategy; see `backtest::simulate` for the execution model.
///
/// Before the simulator sees the bars, the columns must have one length and every `open_time`
/// must be below [`convert::MICROSECOND_TIMES_FROM`]: a microsecond series is refused rather
/// than simulated with bar lengths 10³ too long.
#[pyfunction]
#[pyo3(signature = (
    open_time, open, high, low, close, volume, targets, *,
    initial_cash, fees, liquidity = "taker", slippage_bps = Float::from(2.0),
    min_notional = Float::from(0.0), step_size = Float::from(0.0),
    rebalance_threshold = Float::from(0.0),
))]
#[allow(clippy::too_many_arguments)]
fn simulate(
    py: Python<'_>,
    open_time: PyReadonlyArray1<'_, i64>,
    open: PyReadonlyArray1<'_, f64>,
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    close: PyReadonlyArray1<'_, f64>,
    volume: PyReadonlyArray1<'_, f64>,
    targets: PyReadonlyArray1<'_, f64>,
    initial_cash: Float,
    fees: &Bound<'_, PyAny>,
    liquidity: &str,
    slippage_bps: Float,
    min_notional: Float,
    step_size: Float,
    rebalance_threshold: Float,
) -> PyResult<SimResult> {
    let mut cfg = backtest::SimConfig::new(
        initial_cash.get("initial_cash")?,
        FeeSchedule::resolve(fees)?,
    );
    cfg.liquidity = liquidity_arg(liquidity)?;
    cfg.slippage_bps = slippage_bps.get("slippage_bps")?;
    cfg.filters = backtest::SymbolFilters {
        min_notional: min_notional.get("min_notional")?,
        step_size: step_size.get("step_size")?,
    };
    cfg.rebalance_threshold = rebalance_threshold.get("rebalance_threshold")?;

    // The rows of bars are the largest copy, six words per bar: size it before any copy.
    let n = open_time.len();
    check_alloc(
        || format!("open_time ({n} bars)"),
        &[(n, size_of::<backtest::Bar>().div_ceil(8))],
    )?;
    let (t, o, h, l, c, v) = (
        vec1(&open_time, "open_time")?,
        vec1(&open, "open")?,
        vec1(&high, "high")?,
        vec1(&low, "low")?,
        vec1(&close, "close")?,
        vec1(&volume, "volume")?,
    );
    let bars = BarColumns {
        open_time: &t,
        open: &o,
        high: &h,
        low: &l,
        close: &c,
        volume: &v,
    }
    .to_bars()
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let targets = vec1(&targets, "targets")?;
    // The inputs are owned copies now: the loop runs without the interpreter lock.
    let inner = py
        .detach(move || backtest::simulate(&bars, &targets, &cfg))
        .map_err(sim_error)?;
    Ok(SimResult { inner })
}

/// `backtest::metrics::Metrics`, field for field.
#[pyclass(module = "olsprime_rs", frozen, eq, get_all)]
#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    periods: usize,
    total_return: f64,
    annualized_return: f64,
    annualized_volatility: f64,
    sharpe: Option<f64>,
    sortino: Option<f64>,
    max_drawdown: f64,
    turnover: f64,
    exposure: f64,
    fees_paid: f64,
}

impl From<bm::Metrics> for Metrics {
    fn from(m: bm::Metrics) -> Self {
        let bm::Metrics {
            periods,
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe,
            sortino,
            max_drawdown,
            turnover,
            exposure,
            fees_paid,
        } = m;
        Self {
            periods,
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe,
            sortino,
            max_drawdown,
            turnover,
            exposure,
            fees_paid,
        }
    }
}

#[pymethods]
impl Metrics {
    /// The fields as a dict, for a results table.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("periods", self.periods)?;
        d.set_item("total_return", self.total_return)?;
        d.set_item("annualized_return", self.annualized_return)?;
        d.set_item("annualized_volatility", self.annualized_volatility)?;
        d.set_item("sharpe", self.sharpe)?;
        d.set_item("sortino", self.sortino)?;
        d.set_item("max_drawdown", self.max_drawdown)?;
        d.set_item("turnover", self.turnover)?;
        d.set_item("exposure", self.exposure)?;
        d.set_item("fees_paid", self.fees_paid)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// `backtest::metrics::compute`, annualizing with `periods_per_year`.
#[pyfunction]
fn metrics(result: &SimResult, periods_per_year: Float) -> PyResult<Metrics> {
    let periods_per_year = periods_per_year.get("periods_per_year")?;
    // `compute` asserts this; a panic must not reach Python.
    if !(periods_per_year.is_finite() && periods_per_year > 0.0) {
        return Err(PyValueError::new_err(format!(
            "periods_per_year must be finite and > 0, got {periods_per_year}"
        )));
    }
    Ok(bm::compute(&result.inner, periods_per_year).into())
}

/// Bars per year for a bar length of `bar_ms` milliseconds (365-day years).
#[pyfunction]
fn periods_per_year(bar_ms: Int) -> PyResult<f64> {
    let bar_ms: i64 = bar_ms.get("bar_ms")?;
    if bar_ms <= 0 {
        return Err(PyValueError::new_err(format!(
            "bar length must be positive, got {bar_ms} ms"
        )));
    }
    Ok(bm::periods_per_year(bar_ms))
}

/// Largest peak-to-trough loss of an equity curve, in [0, 1].
#[pyfunction]
fn max_drawdown(equity: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(bm::max_drawdown(&vec1(&equity, "equity")?))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FeeSchedule>()?;
    m.add_class::<SimResult>()?;
    m.add_class::<Fills>()?;
    m.add_class::<Metrics>()?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(metrics, m)?)?;
    m.add_function(wrap_pyfunction!(periods_per_year, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add("BUY", convert::BUY)?;
    m.add("SELL", convert::SELL)?;
    Ok(())
}
