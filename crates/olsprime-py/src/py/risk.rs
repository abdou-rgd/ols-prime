//! The risk kernel, integers only.
//!
//! Every field is an `int` in the kernel's fixed-point units (`QTY_SCALE` qty units per base
//! unit, `QUOTE_SCALE` quote units per quote unit, prices in quote units per whole base unit).
//! A float, a `bool` or a value outside the field's Rust type is refused, with the field's name:
//! a silent rounding or wrap-around here would be a wrong order.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

/// An exact integer of type `T` from a Python `int` (or any `__index__` integer, numpy's
/// included).
fn int_field<T>(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<T>
where
    T: TryFrom<i128>,
{
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{field} must be an int, got bool"
        )));
    }
    let wide: i128 = obj.extract().map_err(|e: PyErr| {
        let py = obj.py();
        if e.is_instance_of::<PyOverflowError>(py) {
            PyValueError::new_err(format!("{field} is out of range: {obj}"))
        } else {
            let ty = obj
                .get_type()
                .name()
                .map_or_else(|_| "?".to_owned(), |n| n.to_string());
            PyTypeError::new_err(format!("{field} must be an int, got {ty}"))
        }
    })?;
    T::try_from(wide).map_err(|_| {
        PyValueError::new_err(format!(
            "{field} = {wide} is out of range for {}",
            std::any::type_name::<T>()
        ))
    })
}

/// Order direction.
#[pyclass(module = "olsprime_rs", frozen, eq, eq_int, hash, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    #[pyo3(name = "BUY")]
    Buy,
    #[pyo3(name = "SELL")]
    Sell,
}

impl From<Side> for risk::Side {
    fn from(s: Side) -> Self {
        match s {
            Side::Buy => Self::Buy,
            Side::Sell => Self::Sell,
        }
    }
}

impl From<risk::Side> for Side {
    fn from(s: risk::Side) -> Self {
        match s {
            risk::Side::Buy => Self::Buy,
            risk::Side::Sell => Self::Sell,
        }
    }
}

/// Why an order was refused (`risk::Rejection`).
#[pyclass(module = "olsprime_rs", frozen, eq, eq_int, hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rejection {
    #[pyo3(name = "INVALID_INPUT")]
    InvalidInput,
    #[pyo3(name = "KILL_SWITCH")]
    KillSwitch,
    #[pyo3(name = "DAILY_LOSS")]
    DailyLoss,
    #[pyo3(name = "COOLDOWN")]
    Cooldown,
    #[pyo3(name = "POSITION_CAP")]
    PositionCap,
    #[pyo3(name = "NO_POSITION")]
    NoPosition,
    #[pyo3(name = "BELOW_MINIMUM")]
    BelowMinimum,
    #[pyo3(name = "OVERFLOW")]
    Overflow,
}

impl From<risk::Rejection> for Rejection {
    // Exhaustive on purpose: a new kernel rejection must fail this build, not reach Python as
    // something else.
    fn from(r: risk::Rejection) -> Self {
        match r {
            risk::Rejection::InvalidInput => Self::InvalidInput,
            risk::Rejection::KillSwitch => Self::KillSwitch,
            risk::Rejection::DailyLoss => Self::DailyLoss,
            risk::Rejection::Cooldown => Self::Cooldown,
            risk::Rejection::PositionCap => Self::PositionCap,
            risk::Rejection::NoPosition => Self::NoPosition,
            risk::Rejection::BelowMinimum => Self::BelowMinimum,
            risk::Rejection::Overflow => Self::Overflow,
        }
    }
}

/// Account state at the decision bar's close. Keyword-only, so that no two quote amounts can
/// be swapped by position.
#[pyclass(module = "olsprime_rs", frozen, eq, hash, get_all)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Account {
    equity: i64,
    peak_equity: i64,
    day_start_equity: i64,
    position: i64,
    bars_since_order: u32,
}

#[pymethods]
impl Account {
    #[new]
    #[pyo3(signature = (*, equity, peak_equity, day_start_equity, position, bars_since_order))]
    fn new(
        equity: &Bound<'_, PyAny>,
        peak_equity: &Bound<'_, PyAny>,
        day_start_equity: &Bound<'_, PyAny>,
        position: &Bound<'_, PyAny>,
        bars_since_order: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            equity: int_field(equity, "equity")?,
            peak_equity: int_field(peak_equity, "peak_equity")?,
            day_start_equity: int_field(day_start_equity, "day_start_equity")?,
            position: int_field(position, "position")?,
            bars_since_order: int_field(bars_since_order, "bars_since_order")?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Account(equity={}, peak_equity={}, day_start_equity={}, position={}, bars_since_order={})",
            self.equity,
            self.peak_equity,
            self.day_start_equity,
            self.position,
            self.bars_since_order
        )
    }
}

impl From<&Account> for risk::Account {
    fn from(a: &Account) -> Self {
        Self {
            equity: risk::Quote(a.equity),
            peak_equity: risk::Quote(a.peak_equity),
            day_start_equity: risk::Quote(a.day_start_equity),
            position: risk::Qty(a.position),
            bars_since_order: a.bars_since_order,
        }
    }
}

/// Static risk limits. Keyword-only.
#[pyclass(module = "olsprime_rs", frozen, eq, hash, get_all)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Limits {
    max_position: i64,
    max_order: i64,
    min_order: i64,
    lot_step: i64,
    max_daily_loss: i64,
    max_drawdown_bps: u32,
    cooldown_bars: u32,
}

#[pymethods]
impl Limits {
    #[new]
    #[pyo3(signature = (
        *, max_position, max_order, min_order, lot_step, max_daily_loss, max_drawdown_bps,
        cooldown_bars,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_position: &Bound<'_, PyAny>,
        max_order: &Bound<'_, PyAny>,
        min_order: &Bound<'_, PyAny>,
        lot_step: &Bound<'_, PyAny>,
        max_daily_loss: &Bound<'_, PyAny>,
        max_drawdown_bps: &Bound<'_, PyAny>,
        cooldown_bars: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            max_position: int_field(max_position, "max_position")?,
            max_order: int_field(max_order, "max_order")?,
            min_order: int_field(min_order, "min_order")?,
            lot_step: int_field(lot_step, "lot_step")?,
            max_daily_loss: int_field(max_daily_loss, "max_daily_loss")?,
            max_drawdown_bps: int_field(max_drawdown_bps, "max_drawdown_bps")?,
            cooldown_bars: int_field(cooldown_bars, "cooldown_bars")?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Limits(max_position={}, max_order={}, min_order={}, lot_step={}, max_daily_loss={}, max_drawdown_bps={}, cooldown_bars={})",
            self.max_position,
            self.max_order,
            self.min_order,
            self.lot_step,
            self.max_daily_loss,
            self.max_drawdown_bps,
            self.cooldown_bars
        )
    }
}

impl From<&Limits> for risk::Limits {
    fn from(l: &Limits) -> Self {
        Self {
            max_position: risk::Quote(l.max_position),
            max_order: risk::Quote(l.max_order),
            min_order: risk::Quote(l.min_order),
            lot_step: risk::Qty(l.lot_step),
            max_daily_loss: risk::Quote(l.max_daily_loss),
            max_drawdown_bps: l.max_drawdown_bps,
            cooldown_bars: l.cooldown_bars,
        }
    }
}

/// What a model asks for. Keyword-only.
#[pyclass(module = "olsprime_rs", frozen, eq, hash, get_all)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrderRequest {
    side: Side,
    qty: i64,
    price: i64,
}

#[pymethods]
impl OrderRequest {
    #[new]
    #[pyo3(signature = (*, side, qty, price))]
    fn new(side: Side, qty: &Bound<'_, PyAny>, price: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            side,
            qty: int_field(qty, "qty")?,
            price: int_field(price, "price")?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderRequest(side={:?}, qty={}, price={})",
            self.side, self.qty, self.price
        )
    }
}

impl From<&OrderRequest> for risk::OrderRequest {
    fn from(r: &OrderRequest) -> Self {
        Self {
            side: r.side.into(),
            qty: risk::Qty(r.qty),
            price: risk::Price(r.price),
        }
    }
}

/// An order that passed every check. Python cannot build one: only `check_order` returns it.
#[pyclass(module = "olsprime_rs", frozen, eq, hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Approved {
    inner: risk::Approved,
}

#[pymethods]
impl Approved {
    #[getter]
    fn side(&self) -> Side {
        self.inner.side().into()
    }

    /// Approved quantity, qty units: positive, a lot multiple, at most the request.
    #[getter]
    fn qty(&self) -> i64 {
        self.inner.qty().0
    }

    /// The rounded-up notional, quote units.
    #[getter]
    fn notional(&self) -> i64 {
        self.inner.notional().0
    }

    fn __repr__(&self) -> String {
        format!(
            "Approved(side={:?}, qty={}, notional={})",
            self.inner.side(),
            self.inner.qty().0,
            self.inner.notional().0
        )
    }
}

/// A refused order: the kernel's reason and its message.
#[pyclass(module = "olsprime_rs", frozen, eq, hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rejected {
    inner: risk::Rejection,
}

#[pymethods]
impl Rejected {
    #[getter]
    fn reason(&self) -> Rejection {
        self.inner.into()
    }

    /// The kernel's own description of the reason.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Rejected({:?}: {})", self.inner, self.inner)
    }
}

#[derive(IntoPyObject)]
enum Decision {
    Approved(Approved),
    Rejected(Rejected),
}

/// `risk::check_order`: an `Approved` order, or a `Rejected` one with its reason.
#[pyfunction]
fn check_order(account: &Account, limits: &Limits, request: &OrderRequest) -> Decision {
    match risk::check_order(&account.into(), &limits.into(), &request.into()) {
        Ok(inner) => Decision::Approved(Approved { inner }),
        Err(inner) => Decision::Rejected(Rejected { inner }),
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Side>()?;
    m.add_class::<Rejection>()?;
    m.add_class::<Account>()?;
    m.add_class::<Limits>()?;
    m.add_class::<OrderRequest>()?;
    m.add_class::<Approved>()?;
    m.add_class::<Rejected>()?;
    m.add_function(wrap_pyfunction!(check_order, m)?)?;
    m.add("QTY_SCALE", risk::QTY_SCALE)?;
    m.add("QUOTE_SCALE", risk::QUOTE_SCALE)?;
    m.add("BPS_SCALE", risk::BPS_SCALE)?;
    Ok(())
}
