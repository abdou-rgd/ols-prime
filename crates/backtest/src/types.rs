//! Plain data types of the simulator: bars, venue costs, exchange filters,
//! configuration and results.

/// One OHLCV bar. Prices are in quote currency per unit of base, `volume` in
/// base units.
///
/// [`crate::simulate`] requires finite prices > 0 with
/// `low ≤ min(open, close)` and `high ≥ max(open, close)`, a finite volume
/// ≥ 0 (0 is valid: exchanges publish truncated bars with no trade), and
/// strictly increasing `open_time` across bars. No time grid is enforced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bar {
    /// Opening time of the bar, milliseconds since the Unix epoch, UTC.
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Exchange order filters that bind a long/flat spot strategy.
///
/// The constants are a **snapshot** of Binance spot `exchangeInfo`
/// (`NOTIONAL.minNotional` and `LOT_SIZE.stepSize`) as of October 2026.
/// Exchanges change filters without notice: re-check against `exchangeInfo`
/// (or the target venue's equivalent) before relying on them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolFilters {
    /// Minimum executed notional of an order, in quote. An order below it is
    /// skipped.
    pub min_notional: f64,
    /// Lot step in base units. Every fill quantity is a whole number of steps.
    /// `0.0` means quantities are not rounded.
    pub step_size: f64,
}

impl SymbolFilters {
    /// No minimum notional, no lot rounding.
    pub const NONE: Self = Self {
        min_notional: 0.0,
        step_size: 0.0,
    };
    /// Binance spot BTCUSDT (snapshot; re-check against `exchangeInfo`).
    pub const BINANCE_BTCUSDT: Self = Self {
        min_notional: 5.0,
        step_size: 0.000_01,
    };
    /// Binance spot ETHUSDT (snapshot; re-check against `exchangeInfo`).
    pub const BINANCE_ETHUSDT: Self = Self {
        min_notional: 5.0,
        step_size: 0.000_1,
    };
}

/// Which side of the book an order takes, which selects its fee rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Liquidity {
    Maker,
    Taker,
}

/// A venue's spot fee schedule, as fractions of notional (0.001 = 0.10 %).
///
/// The constants follow the cost table of `docs/PROTOCOL.md` (October 2026,
/// re-check before use). Binance is history only: it is closed to French
/// residents, so live costs come from the venue actually traded on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeeSchedule {
    pub name: &'static str,
    pub maker: f64,
    pub taker: f64,
}

impl FeeSchedule {
    /// No fees: for identities and tests only.
    pub const ZERO: Self = Self {
        name: "zero",
        maker: 0.0,
        taker: 0.0,
    };
    /// Binance spot, regular tier: 0.10 % / 0.10 %.
    pub const BINANCE_SPOT: Self = Self {
        name: "binance-spot",
        maker: 0.001,
        taker: 0.001,
    };
    /// Binance spot paying fees in BNB: 0.075 % / 0.075 %.
    pub const BINANCE_SPOT_BNB: Self = Self {
        name: "binance-spot-bnb",
        maker: 0.000_75,
        taker: 0.000_75,
    };
    /// Coinbase Advanced, entry tier: 0.25 % maker / 0.50 % taker.
    pub const COINBASE_ADVANCED_ENTRY: Self = Self {
        name: "coinbase-advanced-entry",
        maker: 0.002_5,
        taker: 0.005,
    };
    /// Kraken Pro, entry tier: 0.40 % maker / 0.80 % taker.
    pub const KRAKEN_PRO_ENTRY: Self = Self {
        name: "kraken-pro-entry",
        maker: 0.004,
        taker: 0.008,
    };

    /// Fee rate charged for the given liquidity.
    pub fn rate(&self, liquidity: Liquidity) -> f64 {
        match liquidity {
            Liquidity::Maker => self.maker,
            Liquidity::Taker => self.taker,
        }
    }
}

/// Simulation parameters. All fields are public; [`SimConfig::new`] gives the
/// protocol defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimConfig {
    /// Starting cash in quote. Must be finite and > 0.
    pub initial_cash: f64,
    pub fees: FeeSchedule,
    /// Liquidity every order is charged at. Taker is the honest default for
    /// orders placed at the open.
    pub liquidity: Liquidity,
    /// Adverse slippage per side, in basis points of the open.
    pub slippage_bps: f64,
    pub filters: SymbolFilters,
    /// Minimum |target − current weight| that triggers a rebalance. A target
    /// of 0 always exits in full, whatever the threshold.
    pub rebalance_threshold: f64,
}

impl SimConfig {
    /// Taker liquidity, 2 bps slippage per side (the protocol's value), no
    /// filters and no rebalance threshold.
    pub fn new(initial_cash: f64, fees: FeeSchedule) -> Self {
        Self {
            initial_cash,
            fees,
            liquidity: Liquidity::Taker,
            slippage_bps: 2.0,
            filters: SymbolFilters::NONE,
            rebalance_threshold: 0.0,
        }
    }
}

/// Direction of a fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

/// One executed order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fill {
    /// Index of the bar whose open executed the order. A fill at bar `b`
    /// carries the target decided at the close of bar `b − 1`.
    pub bar: usize,
    /// `open_time` of that bar.
    pub time: i64,
    pub side: Side,
    /// Base quantity, > 0.
    pub qty: f64,
    /// Executed price, slippage included: `open·(1 ± s)`.
    pub price: f64,
    /// `qty · price`, in quote.
    pub notional: f64,
    /// `notional · rate`, in quote.
    pub fee: f64,
}

/// Output of [`crate::simulate`]: one value per bar for each series, marked at
/// the bar's close after any trade at its open.
#[derive(Debug, Clone, PartialEq)]
pub struct SimResult {
    pub equity: Vec<f64>,
    pub position: Vec<f64>,
    pub cash: Vec<f64>,
    pub fills: Vec<Fill>,
    /// Sum of `fill.fee` over all fills, in fill order.
    pub fees_paid: f64,
}
