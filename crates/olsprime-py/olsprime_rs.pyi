"""Type stubs for olsprime_rs, the Rust core of OLS Prime.

Conventions:
- Arrays in are numpy arrays of the exact dtype (float64; int64 for times) and number of
  dimensions; anything else raises TypeError before a value is read. Strided views are read
  in their logical order.
- Scalars: an `int` parameter takes any integer (numpy's included) and a `float` parameter any
  real number; a bool is refused for both.
- Arrays out are fresh copies owned by the caller.
- Errors: bad input raises TypeError (wrong type) or ValueError (wrong value), and nothing
  else. An integer outside its Rust range is a ValueError naming the parameter, never an
  OverflowError. SimError and OlsError are ValueErrors carrying the Rust message.
- Sizes: a shape whose allocation would exceed MAX_ALLOC_BYTES (a p x p Gram matrix, an
  attention matrix, or the copy of an input array, zero-stride views included) is a ValueError
  naming the arguments, raised before anything is allocated. A failed allocation would abort
  the interpreter instead of raising.
"""

from typing import Final, Literal, NotRequired, TypedDict, final

import numpy as np
import numpy.typing as npt

__version__: Final[str]

_F64 = npt.NDArray[np.float64]
_I64 = npt.NDArray[np.int64]
_I8 = npt.NDArray[np.int8]
_U8 = npt.NDArray[np.uint8]

# Scalar parameters of the centred and batch readouts (iteration 3). `_Int` is `int` and
# `_Float` is `float`, with the same conversion rules; the aliases keep these parameters out
# of the inventory of tests/test_rust_bindings.py, and tests/test_matrix_batch.py runs the
# same range and type checks on every parameter annotated with them.
_Int = int
_Float = float

# --- errors -------------------------------------------------------------------------------

class SimError(ValueError):
    """backtest::SimError: the simulator refused its inputs."""

class OlsError(ValueError):
    """ols_core::OlsError: an OLS-core operation failed."""

# --- limits -------------------------------------------------------------------------------

MAX_ALLOC_BYTES: Final[int]
"""2**30 (1 GiB): the most bytes one object or array copy sized by a caller's shape may take."""
MICROSECOND_TIMES_FROM: Final[int]
"""10**14: the first time refused as microseconds by simulate (open_time) and Matrix (push,
extend, time_range). Every time here is in milliseconds; 10**14 ms is the year 5138, 10**14 us
is 1973."""

# --- backtest -----------------------------------------------------------------------------

BUY: Final[int]
"""Fills.side code of a buy: +1."""
SELL: Final[int]
"""Fills.side code of a sell: -1."""

_FeeName = Literal[
    "zero",
    "binance-spot",
    "binance-spot-bnb",
    "coinbase-advanced-entry",
    "kraken-pro-entry",
    "bitvavo-entry",
]

@final
class FeeSchedule:
    """A venue's spot fee schedule, as fractions of notional."""

    ZERO: Final[FeeSchedule]
    BINANCE_SPOT: Final[FeeSchedule]
    BINANCE_SPOT_BNB: Final[FeeSchedule]
    COINBASE_ADVANCED_ENTRY: Final[FeeSchedule]
    KRAKEN_PRO_ENTRY: Final[FeeSchedule]
    BITVAVO_ENTRY: Final[FeeSchedule]

    def __init__(self, name: str, maker: float, taker: float) -> None:
        """A custom schedule: rates in [0, 1), built-in names reserved (ValueError)."""
    @staticmethod
    def named(name: str) -> FeeSchedule:
        """The built-in schedule with this exact name (ValueError if unknown)."""
    @staticmethod
    def names() -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def maker(self) -> float: ...
    @property
    def taker(self) -> float: ...
    def rate(self, liquidity: Literal["taker", "maker"]) -> float: ...
    def __eq__(self, other: object) -> bool: ...

@final
class Fills:
    """The fills of a simulation, one column per field, in fill order."""

    @property
    def bar(self) -> _I64:
        """Index of the bar whose open executed the fill."""
    @property
    def time(self) -> _I64:
        """open_time of that bar, ms since the epoch, UTC."""
    @property
    def side(self) -> _I8:
        """BUY (+1) or SELL (-1)."""
    @property
    def qty(self) -> _F64: ...
    @property
    def price(self) -> _F64:
        """Executed price, slippage included."""
    @property
    def notional(self) -> _F64: ...
    @property
    def fee(self) -> _F64: ...
    def as_dict(self) -> dict[str, _I64 | _I8 | _F64]: ...
    def __len__(self) -> int: ...

@final
class SimResult:
    """The output of simulate: one value per bar, marked at the close."""

    @property
    def equity(self) -> _F64: ...
    @property
    def position(self) -> _F64: ...
    @property
    def cash(self) -> _F64: ...
    @property
    def fills(self) -> Fills: ...
    @property
    def fees_paid(self) -> float: ...
    def __len__(self) -> int: ...

def simulate(
    open_time: _I64,
    open: _F64,
    high: _F64,
    low: _F64,
    close: _F64,
    volume: _F64,
    targets: _F64,
    *,
    initial_cash: float,
    fees: FeeSchedule | _FeeName,
    liquidity: Literal["taker", "maker"] = "taker",
    slippage_bps: float = 2.0,
    min_notional: float = 0.0,
    step_size: float = 0.0,
    rebalance_threshold: float = 0.0,
) -> SimResult:
    """backtest::simulate. targets[t] executes at the open of bar t + 1; targets[-1] never.

    open_time is in milliseconds since the epoch, UTC. A time >= 10**14 (the year 5138 in ms,
    1973 in microseconds) is refused as a microsecond time.

    Raises SimError for invalid bars, targets or config, ValueError for bar columns of unequal
    length, a microsecond time, or an unknown fee or liquidity name, TypeError for wrong dtypes.
    """

@final
class Metrics:
    """backtest::metrics::Metrics."""

    @property
    def periods(self) -> int: ...
    @property
    def total_return(self) -> float: ...
    @property
    def annualized_return(self) -> float: ...
    @property
    def annualized_volatility(self) -> float: ...
    @property
    def sharpe(self) -> float | None: ...
    @property
    def sortino(self) -> float | None: ...
    @property
    def max_drawdown(self) -> float: ...
    @property
    def turnover(self) -> float: ...
    @property
    def exposure(self) -> float: ...
    @property
    def fees_paid(self) -> float: ...
    def as_dict(self) -> dict[str, int | float | None]: ...
    def __eq__(self, other: object) -> bool: ...

def metrics(result: SimResult, periods_per_year: float) -> Metrics:
    """backtest::metrics::compute. ValueError unless periods_per_year is finite and > 0."""

def periods_per_year(bar_ms: int) -> float:
    """365 * 86_400_000 / bar_ms. ValueError if bar_ms <= 0."""

def max_drawdown(equity: _F64) -> float: ...

# --- ols-core -----------------------------------------------------------------------------

@final
class RidgeConfig:
    """Penalty lam >= 0 on every coordinate, or on all but coordinate 0 with intercept=True."""

    def __init__(self, lam: float, *, intercept: bool = False) -> None: ...
    @staticmethod
    def with_intercept(lam: float) -> RidgeConfig: ...
    @property
    def lam(self) -> float: ...
    @property
    def intercept(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...

class SuffStats:
    """Sufficient statistics of weighted entries; mutable only through push and merge."""

    def __init__(self, p: int, q: int) -> None:
        """The statistics of the empty set."""
    @staticmethod
    def from_rows(xs: _F64, ys: _F64) -> SuffStats:
        """Unit-weight rows: xs is (n, p), ys is (n, q)."""
    def push(self, x: _F64, y: _F64, w: float = 1.0) -> None: ...
    def merge(self, other: SuffStats) -> None: ...
    def merged(self, other: SuffStats) -> SuffStats: ...
    def copy(self) -> SuffStats: ...
    def __copy__(self) -> SuffStats: ...
    def mean_y(self) -> _F64 | None: ...
    @property
    def p(self) -> int: ...
    @property
    def q(self) -> int: ...
    @property
    def count(self) -> int: ...
    @property
    def n(self) -> float: ...
    @property
    def sum_x(self) -> _F64: ...
    @property
    def gram(self) -> _F64:
        """G = sum w x x^T, shape (p, p)."""
    @property
    def cross(self) -> _F64:
        """C = sum w y x^T, shape (q, p)."""
    @property
    def sum_y(self) -> _F64: ...
    @property
    def sum_yy(self) -> _F64: ...
    def __eq__(self, other: object) -> bool: ...

@final
class RidgeModel:
    @property
    def beta(self) -> _F64:
        """beta = C (G + Lambda)^-1, shape (q, p)."""
    def predict(self, x: _F64) -> _F64: ...
    def __eq__(self, other: object) -> bool: ...

def ridge_fit(stats: SuffStats, cfg: RidgeConfig) -> RidgeModel: ...
def solve_query(stats: SuffStats, cfg: RidgeConfig, x_q: _F64) -> _F64:
    """The dual direction a = (G + Lambda)^-1 x_q."""

def residual_sum_squares(stats: SuffStats, model: RidgeModel) -> _F64: ...

@final
class Attention:
    @property
    def weights(self) -> _F64:
        """Signed weights, shape (queries, keys)."""
    @property
    def forecast(self) -> _F64:
        """weights @ values, shape (queries, q)."""

def ols_attention(keys: _F64, values: _F64, queries: _F64, cfg: RidgeConfig) -> Attention: ...

class Rls:
    """Recursive least squares with forgetting factor gamma in (0, 1]."""

    def __init__(self, p: int, q: int, lam: float, gamma: float = 1.0) -> None: ...
    def update(self, x: _F64, y: _F64) -> None: ...
    def predict(self, x: _F64) -> _F64: ...
    @property
    def beta(self) -> _F64: ...
    @property
    def updates(self) -> int: ...
    @property
    def gamma(self) -> float: ...

@final
class Readout:
    """A read of Matrix; only the Matrix that made it accepts it."""

    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def a(self) -> _F64: ...
    @property
    def forecast(self) -> _F64: ...

@final
class Summary:
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def n(self) -> float: ...
    @property
    def mean_y(self) -> _F64: ...
    @property
    def model(self) -> RidgeModel: ...
    @property
    def residual_var(self) -> _F64: ...

@final
class CentredStats:
    """Centred statistics of unit-weight entries: the count, the means and the centred sums
    M_xx = sum (x - mean_x)(x - mean_x)^T, M_yx and M_yy. Immutable."""

    @staticmethod
    def from_rows(xs: _F64, ys: _F64) -> CentredStats:
        """xs (n, p), ys (n, q), in one corrected two-pass scan."""
    def merged(self, other: CentredStats) -> CentredStats:
        """The statistics of the union of two disjoint sets (Chan-Golub-LeVeque)."""
    @property
    def p(self) -> int: ...
    @property
    def q(self) -> int: ...
    @property
    def count(self) -> int: ...
    @property
    def n(self) -> float: ...
    @property
    def longest_scan(self) -> int: ...
    @property
    def depth(self) -> int: ...
    @property
    def mean_x(self) -> _F64: ...
    @property
    def m_xx(self) -> _F64: ...
    @property
    def mean_y(self) -> _F64: ...
    @property
    def m_yx(self) -> _F64: ...
    @property
    def m_yy(self) -> _F64: ...

@final
class CentredReadout:
    """A centred ridge read of Matrix; only the Matrix that made it accepts it."""

    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def mean_x(self) -> _F64:
        """The range's mean state, (p,)."""
    @property
    def mean_y(self) -> _F64:
        """The range's mean outcome, (q,) float64: the forecast at x_q = mean_x, and the bits
        the forecast added."""
    @property
    def beta(self) -> _F64:
        """beta = M_yx (M_xx + Lambda)^-1, (q, p) float64, C-ordered: the coefficients of the
        same solve as the forecast, so that forecast = mean_y + beta @ (x_q - mean_x), and
        beta[c, j] * (x_q[j] - mean_x[j]) is feature j's part of outcome c.

        Bits: the forecast rounded each product beta[c, j] * xt[j] on its own (no fused
        multiply-add), summed them over j = 0, 1, ... in order, then added mean_y[c] last, with
        xt = x_q - mean_x. A Python loop over floats in that order reproduces it to the bit.
        numpy's `beta @ xt` sums in its own order (BLAS, possibly with FMA), so it agrees only
        within 2 * gamma(p + 1) * (|mean_y[c]| + sum_j |beta[c, j] * xt[j]|), with
        gamma(k) = k u / (1 - k u) and u = 2**-53."""
    @property
    def a(self) -> _F64:
        """(M_xx + Lambda)^-1 (x_q - mean_x), (p,)."""
    @property
    def forecast(self) -> _F64:
        """mean_y + beta (x_q - mean_x), (q,): the bits ridge_batch gives for the same cell."""
    @property
    def leverage(self) -> float:
        """(x_q - mean_x)^T (M_xx + Lambda)^-1 (x_q - mean_x)."""
    @property
    def rss(self) -> _F64:
        """The range's residual sum of squares at this penalty, (q,)."""

STATUS_OK: Final[int]
"""0: a number."""
STATUS_EMPTY: Final[int]
"""1: the window holds no entry."""
STATUS_SINGULAR: Final[int]
"""2: the centred solve refused the window as numerically singular."""
STATUS_NON_FINITE: Final[int]
"""3: an intermediate or the result overflowed."""
STATUS_TOO_FEW: Final[int]
"""4: a kNN read asked for more neighbours than the window holds."""

# The dicts that ridge_batch and kernel_batch return (stub-only types). Shapes:
# - ridge: forecast (M, W, R, q); status (M, W, R) uint8; start, end and count (M, W) int64,
#   count = end - start, the n_w of the penalty; leverage (M, W, R) with leverage=True; rss
#   (M, W, R, q) with rss=True.
# - kernel: softmax (M, T, q); softmax_status (M, T) uint8; knn (M, K, q); knn_status (M, K)
#   uint8; start, end and count (M,) int64.
_RidgeBatchResult = TypedDict(
    "_RidgeBatchResult",
    {
        "forecast": _F64,
        "status": _U8,
        "start": _I64,
        "end": _I64,
        "count": _I64,
        "leverage": NotRequired[_F64],
        "rss": NotRequired[_F64],
    },
)
_KernelBatchResult = TypedDict(
    "_KernelBatchResult",
    {
        "softmax": _F64,
        "softmax_status": _U8,
        "knn": _F64,
        "knn_status": _U8,
        "start": _I64,
        "end": _I64,
        "count": _I64,
    },
)

class Matrix:
    """Append-only memory of (time, state, outcome) entries. Push an entry only once its
    outcome is known, stamped with the time it became known (ms, UTC)."""

    def __init__(self, p: int, q: int, *, block_log2: int = 6) -> None:
        """MAX_ALLOC_BYTES bounds one tree node (p*p + q*p + p + 2*q + 1 floats), not the
        memory, which grows by p + q floats per entry and one node per closed block."""
    def push(self, time: int, x: _F64, y: _F64) -> None:
        """Push one entry. A time >= MICROSECOND_TIMES_FROM is a ValueError."""
    def extend(self, times: _I64, xs: _F64, ys: _F64) -> None:
        """Push n entries at once, or none. Shapes are checked first, then the unit of every
        time (a time >= MICROSECOND_TIMES_FROM is a ValueError), then each entry in order."""
    def __len__(self) -> int: ...
    @property
    def p(self) -> int: ...
    @property
    def q(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def times(self) -> _I64: ...
    def time_range(self, from_time: int, as_of: int) -> tuple[int, int]:
        """(start, end) of the entries with from_time <= time <= as_of. Both are milliseconds:
        a bound >= MICROSECOND_TIMES_FROM is a ValueError, since an as_of in microseconds would
        expose every entry. -2**63 stays a valid from_time."""
    def range_stats(self, start: int, end: int) -> SuffStats: ...
    def query(self, x_q: _F64, start: int, end: int, cfg: RidgeConfig) -> Readout: ...
    def weights(self, readout: Readout) -> _F64: ...
    def contribution(self, readout: Readout, start: int, end: int) -> _F64: ...
    def summarize(self, start: int, end: int, cfg: RidgeConfig) -> Summary: ...
    def window_range(self, as_of: _Int, window: _Int | None = None) -> tuple[int, int]:
        """(start, end) of the entries with as_of - window < time <= as_of (every
        time <= as_of for window=None). window > 0 ms; as_of refused as in time_range."""
    def centred_stats(self, start: _Int, end: _Int) -> CentredStats:
        """The centred statistics of entries [start, end), from the tree of centred nodes."""
    def centred_query(
        self, x_q: _F64, start: _Int, end: _Int, rho: _Float, scale: _F64
    ) -> CentredReadout:
        """Ridge with a free intercept over [start, end): Lambda = rho * n * diag(scale),
        rho >= 0, scale (p,) finite and >= 0. An empty or singular range is an OlsError."""
    def centred_weights(self, readout: CentredReadout) -> _F64:
        """1/n + (x_i - mean_x)^T a for every entry of the range, (end - start,); they sum
        to one and weights @ ys is the forecast."""
    def centred_contribution(self, readout: CentredReadout, start: _Int, end: _Int) -> _F64:
        """The exact contribution of entries [start, end) to the readout's forecast, (q,)."""
    def ridge_batch(
        self,
        x_q: _F64,
        as_of: _I64,
        windows: list[_Int | None],
        rhos: _F64,
        scale: _F64,
        *,
        leverage: bool = False,
        rss: bool = False,
        threads: _Int = 0,
    ) -> _RidgeBatchResult:
        """centred_query for every query (x_q (M, p), as_of (M,) ms), window (a span in ms,
        or None for all history) and penalty (rhos (R,)), in parallel without the GIL. A cell
        that is not STATUS_OK is NaN in every output; bit-identical to centred_query cell by
        cell, whatever threads (0: every core)."""
    def kernel_batch(
        self,
        x_q: _F64,
        as_of: _I64,
        window: _Int | None,
        sigma: _F64,
        taus: _F64,
        ks: _I64,
        *,
        threads: _Int = 0,
    ) -> _KernelBatchResult:
        """Exact softmax (weights exp(-|u_q - u_i|^2 / (2 tau^2)), u = x / sigma) and kNN
        (mean y of the k nearest, ties to the lowest index) over one window, in parallel
        without the GIL. Bit-identical whatever threads and whatever the other queries."""

# --- risk ---------------------------------------------------------------------------------

QTY_SCALE: Final[int]
"""Qty units per whole base unit (1e8)."""
QUOTE_SCALE: Final[int]
"""Quote units per whole quote unit (1e6)."""
BPS_SCALE: Final[int]

@final
class Side:
    BUY: Final[Side]
    SELL: Final[Side]
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

@final
class Rejection:
    INVALID_INPUT: Final[Rejection]
    KILL_SWITCH: Final[Rejection]
    DAILY_LOSS: Final[Rejection]
    COOLDOWN: Final[Rejection]
    POSITION_CAP: Final[Rejection]
    NO_POSITION: Final[Rejection]
    BELOW_MINIMUM: Final[Rejection]
    OVERFLOW: Final[Rejection]
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

@final
class Account:
    """Account state at the decision bar's close, in quote and qty units."""

    def __init__(
        self,
        *,
        equity: int,
        peak_equity: int,
        day_start_equity: int,
        position: int,
        bars_since_order: int,
    ) -> None: ...
    @property
    def equity(self) -> int: ...
    @property
    def peak_equity(self) -> int: ...
    @property
    def day_start_equity(self) -> int: ...
    @property
    def position(self) -> int: ...
    @property
    def bars_since_order(self) -> int: ...

@final
class Limits:
    def __init__(
        self,
        *,
        max_position: int,
        max_order: int,
        min_order: int,
        lot_step: int,
        max_daily_loss: int,
        max_drawdown_bps: int,
        cooldown_bars: int,
    ) -> None: ...
    @property
    def max_position(self) -> int: ...
    @property
    def max_order(self) -> int: ...
    @property
    def min_order(self) -> int: ...
    @property
    def lot_step(self) -> int: ...
    @property
    def max_daily_loss(self) -> int: ...
    @property
    def max_drawdown_bps(self) -> int: ...
    @property
    def cooldown_bars(self) -> int: ...

@final
class OrderRequest:
    def __init__(self, *, side: Side, qty: int, price: int) -> None: ...
    @property
    def side(self) -> Side: ...
    @property
    def qty(self) -> int: ...
    @property
    def price(self) -> int: ...

@final
class Approved:
    """An order that passed every check; only check_order makes one."""

    @property
    def side(self) -> Side: ...
    @property
    def qty(self) -> int: ...
    @property
    def notional(self) -> int: ...

@final
class Rejected:
    @property
    def reason(self) -> Rejection: ...
    @property
    def message(self) -> str: ...

def check_order(account: Account, limits: Limits, request: OrderRequest) -> Approved | Rejected: ...
