//! The Python-free half of the bindings: layouts, names and batch checks.
//!
//! numpy hands the bindings columns (one array per field) and row-major matrices, while the
//! crates speak in rows of structs and column-major `nalgebra` matrices. Every conversion here is
//! a permutation of values, never an arithmetic operation, so a round trip is the identity bit
//! for bit; the tests below prove it on random inputs.

use backtest::{Bar, FeeSchedule, Fill, Liquidity, Side};
use ols_core::nalgebra::DMatrix;
use ols_core::{Matrix, OlsError};
use thiserror::Error;

/// The `side` code of a buy fill in [`FillColumns::side`]. Signed so that `side · qty` is the
/// signed change of the position.
pub const BUY: i8 = 1;
/// The `side` code of a sell fill in [`FillColumns::side`].
pub const SELL: i8 = -1;

/// The Python default of `liquidity`, which must be [`backtest::SimConfig::new`]'s.
pub const DEFAULT_LIQUIDITY: &str = "taker";
/// The Python default of `slippage_bps`, which must be [`backtest::SimConfig::new`]'s: the
/// protocol's 2 bps per side.
pub const DEFAULT_SLIPPAGE_BPS: f64 = 2.0;

/// The name a custom schedule carries inside [`backtest::FeeSchedule`]. That field is a
/// `&'static str` and plays no part in the simulation; the Python object keeps the caller's name.
pub const CUSTOM_FEE_NAME: &str = "custom";

/// `"taker"` or `"maker"`, exactly.
pub fn parse_liquidity(name: &str) -> Option<Liquidity> {
    match name {
        "taker" => Some(Liquidity::Taker),
        "maker" => Some(Liquidity::Maker),
        _ => None,
    }
}

/// The names accepted for `fees`, in the order of [`FeeSchedule::ALL`].
pub fn fee_names() -> Vec<&'static str> {
    FeeSchedule::ALL.iter().map(|f| f.name).collect()
}

/// Why a custom fee schedule was refused.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FeeError {
    #[error("a fee schedule needs a non-empty name")]
    EmptyName,
    /// A custom schedule may not borrow a built-in name: `"binance-spot"` must always mean the
    /// built-in rates, or a report could print a venue's name next to someone else's costs.
    #[error("{0:?} is a built-in schedule; use FeeSchedule.named({0:?}) instead")]
    ReservedName(String),
    /// The simulator's own domain for a rate: finite, in [0, 1).
    #[error("invalid fee rate: {field} = {value}, expected a value in [0, 1)")]
    InvalidRate { field: &'static str, value: f64 },
}

/// A custom schedule, checked against the same domain as [`backtest::simulate`] so that a bad
/// rate fails where it is written rather than at the first simulation.
pub fn custom_fee(name: &str, maker: f64, taker: f64) -> Result<FeeSchedule, FeeError> {
    if name.is_empty() {
        return Err(FeeError::EmptyName);
    }
    if FeeSchedule::named(name).is_some() {
        return Err(FeeError::ReservedName(name.to_owned()));
    }
    for (field, value) in [("maker", maker), ("taker", taker)] {
        if !(0.0..1.0).contains(&value) {
            return Err(FeeError::InvalidRate { field, value });
        }
    }
    Ok(FeeSchedule {
        name: CUSTOM_FEE_NAME,
        maker,
        taker,
    })
}

/// Bar columns of unequal length. The simulator's own [`backtest::SimError::LengthMismatch`]
/// covers targets against bars; this covers the columns of the bars themselves, which the
/// simulator never sees as columns.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("column {column} has {got} values but open_time has {expected}")]
pub struct ColumnError {
    pub column: &'static str,
    pub expected: usize,
    pub got: usize,
}

/// The six columns of a bar series, borrowed.
#[derive(Debug, Clone, Copy)]
pub struct BarColumns<'a> {
    pub open_time: &'a [i64],
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: &'a [f64],
}

impl BarColumns<'_> {
    /// The bars, row by row. Only the lengths are checked here; prices, volumes and times are
    /// the simulator's to validate, so that its error is the one the caller sees.
    pub fn to_bars(&self) -> Result<Vec<Bar>, ColumnError> {
        let n = self.open_time.len();
        for (column, values) in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ] {
            if values.len() != n {
                return Err(ColumnError {
                    column,
                    expected: n,
                    got: values.len(),
                });
            }
        }
        Ok((0..n)
            .map(|i| Bar {
                open_time: self.open_time[i],
                open: self.open[i],
                high: self.high[i],
                low: self.low[i],
                close: self.close[i],
                volume: self.volume[i],
            })
            .collect())
    }
}

/// The fills of a simulation as columns, one entry per fill.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FillColumns {
    /// [`Fill::bar`], as `i64` for numpy.
    pub bar: Vec<i64>,
    pub time: Vec<i64>,
    /// [`BUY`] or [`SELL`].
    pub side: Vec<i8>,
    pub qty: Vec<f64>,
    pub price: Vec<f64>,
    pub notional: Vec<f64>,
    pub fee: Vec<f64>,
}

impl FillColumns {
    /// Transposes `fills` into columns, in fill order.
    pub fn from_fills(fills: &[Fill]) -> Self {
        let mut out = Self::default();
        for f in fills {
            // A bar index is below the length of a `Vec`, at most `isize::MAX`.
            out.bar.push(i64::try_from(f.bar).unwrap_or(i64::MAX));
            out.time.push(f.time);
            out.side.push(match f.side {
                Side::Buy => BUY,
                Side::Sell => SELL,
            });
            out.qty.push(f.qty);
            out.price.push(f.price);
            out.notional.push(f.notional);
            out.fee.push(f.fee);
        }
        out
    }

    /// Number of fills.
    pub fn len(&self) -> usize {
        self.bar.len()
    }

    /// Whether there is no fill.
    pub fn is_empty(&self) -> bool {
        self.bar.is_empty()
    }

    /// The inverse of [`FillColumns::from_fills`], for the round-trip tests. `None` if a column
    /// is short, a side code is unknown or a bar index is negative.
    pub fn to_fills(&self) -> Option<Vec<Fill>> {
        let n = self.len();
        let lens = [
            self.time.len(),
            self.side.len(),
            self.qty.len(),
            self.price.len(),
            self.notional.len(),
            self.fee.len(),
        ];
        if lens.iter().any(|&l| l != n) {
            return None;
        }
        (0..n)
            .map(|i| {
                Some(Fill {
                    bar: usize::try_from(self.bar[i]).ok()?,
                    time: self.time[i],
                    side: match self.side[i] {
                        BUY => Side::Buy,
                        SELL => Side::Sell,
                        _ => return None,
                    },
                    qty: self.qty[i],
                    price: self.price[i],
                    notional: self.notional[i],
                    fee: self.fee[i],
                })
            })
            .collect()
    }
}

/// A `rows × cols` matrix from values in row-major order, numpy's default layout.
///
/// # Panics
///
/// If `data.len() != rows · cols`: the caller derives all three from one array's shape.
pub fn dmatrix_from_row_major(rows: usize, cols: usize, data: &[f64]) -> DMatrix<f64> {
    assert_eq!(
        data.len(),
        rows * cols,
        "row-major data of the wrong length"
    );
    DMatrix::from_row_slice(rows, cols, data)
}

/// The values of `m` in row-major order, for a numpy array of shape `(nrows, ncols)`.
pub fn row_major(m: &DMatrix<f64>) -> Vec<f64> {
    // `transpose` of a column-major matrix lays its rows out contiguously.
    m.transpose().as_slice().to_vec()
}

/// Pushes `n` entries into `memory` at once, or none: `times` has n values, `xs` the n states
/// row-major (n × p) and `ys` the n outcomes (n × q).
///
/// Every entry is checked first, with the same checks and in the same order as
/// [`Matrix::push`], time order against the memory's last entry and within the batch included.
/// Only then are the entries pushed, so the pushes cannot fail and a rejected batch leaves the
/// memory unchanged. The error is the one the first failing [`Matrix::push`] would have
/// returned.
pub fn extend_matrix(
    memory: &mut Matrix,
    times: &[i64],
    xs: &[f64],
    ys: &[f64],
) -> Result<(), OlsError> {
    let (p, q) = memory.dims();
    let n = times.len();
    if xs.len() != n * p {
        return Err(OlsError::Dimension {
            expected: n * p,
            got: xs.len(),
        });
    }
    if ys.len() != n * q {
        return Err(OlsError::Dimension {
            expected: n * q,
            got: ys.len(),
        });
    }
    let mut last = memory.len().checked_sub(1).map(|i| memory.time(i));
    for (i, &time) in times.iter().enumerate() {
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        if !finite(&xs[i * p..(i + 1) * p]) || !finite(&ys[i * q..(i + 1) * q]) {
            return Err(OlsError::NonFinite);
        }
        if let Some(last) = last
            && time < last
        {
            return Err(OlsError::NonMonotonicTime { last, got: time });
        }
        last = Some(time);
    }
    for (i, &time) in times.iter().enumerate() {
        memory.push(time, &xs[i * p..(i + 1) * p], &ys[i * q..(i + 1) * q])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use backtest::SimConfig;
    use ols_core::SuffStats;
    use proptest::prelude::*;

    #[test]
    fn python_defaults_are_the_simulator_defaults() {
        let cfg = SimConfig::new(1.0, FeeSchedule::ZERO);
        assert_eq!(cfg.slippage_bps.to_bits(), DEFAULT_SLIPPAGE_BPS.to_bits());
        assert_eq!(parse_liquidity(DEFAULT_LIQUIDITY), Some(cfg.liquidity));
        assert_eq!(cfg.filters, backtest::SymbolFilters::NONE);
        assert_eq!(cfg.rebalance_threshold, 0.0);
    }

    #[test]
    fn names_resolve_strictly() {
        assert_eq!(parse_liquidity("maker"), Some(Liquidity::Maker));
        for bad in ["Taker", "TAKER", " taker", "", "market"] {
            assert_eq!(parse_liquidity(bad), None, "{bad:?}");
        }
        assert_eq!(
            fee_names(),
            [
                "zero",
                "binance-spot",
                "binance-spot-bnb",
                "coinbase-advanced-entry",
                "kraken-pro-entry",
                "bitvavo-entry"
            ]
        );
        assert_eq!(
            FeeSchedule::named("bitvavo-entry"),
            Some(FeeSchedule::BITVAVO_ENTRY)
        );
    }

    #[test]
    fn custom_fees_are_checked_where_they_are_written() {
        let f = custom_fee("my-venue", 0.001, 0.002).unwrap();
        assert_eq!((f.name, f.maker, f.taker), (CUSTOM_FEE_NAME, 0.001, 0.002));
        assert_eq!(custom_fee("", 0.0, 0.0), Err(FeeError::EmptyName));
        assert_eq!(
            custom_fee("binance-spot", 0.0, 0.0),
            Err(FeeError::ReservedName("binance-spot".into()))
        );
        for bad in [-1e-300, 1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                custom_fee("v", bad, 0.0),
                Err(FeeError::InvalidRate { field: "maker", .. })
            ));
            assert!(matches!(
                custom_fee("v", 0.0, bad),
                Err(FeeError::InvalidRate { field: "taker", .. })
            ));
        }
    }

    #[test]
    fn bar_columns_must_have_one_length() {
        let t = [1, 2, 3];
        let ok = [1.0, 2.0, 3.0];
        let short = [1.0, 2.0];
        let cols = BarColumns {
            open_time: &t,
            open: &ok,
            high: &ok,
            low: &short,
            close: &ok,
            volume: &ok,
        };
        assert_eq!(
            cols.to_bars(),
            Err(ColumnError {
                column: "low",
                expected: 3,
                got: 2
            })
        );
        let empty = BarColumns {
            open_time: &[],
            open: &[],
            high: &[],
            low: &[],
            close: &[],
            volume: &[],
        };
        assert_eq!(empty.to_bars(), Ok(vec![]));
    }

    fn any_f64() -> impl Strategy<Value = f64> {
        // Every bit pattern: NaNs with payloads, infinities, subnormals, -0.0.
        any::<u64>().prop_map(f64::from_bits)
    }

    fn any_fill() -> impl Strategy<Value = Fill> {
        (
            0usize..1 << 40,
            any::<i64>(),
            any::<bool>(),
            any_f64(),
            any_f64(),
            any_f64(),
            any_f64(),
        )
            .prop_map(|(bar, time, buy, qty, price, notional, fee)| Fill {
                bar,
                time,
                side: if buy { Side::Buy } else { Side::Sell },
                qty,
                price,
                notional,
                fee,
            })
    }

    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    proptest! {
        #[test]
        fn bars_keep_every_bit(rows in prop::collection::vec(
            (any::<i64>(), any_f64(), any_f64(), any_f64(), any_f64(), any_f64()), 0..40)
        ) {
            let col = |k: usize| -> Vec<f64> {
                rows.iter().map(|r| [r.1, r.2, r.3, r.4, r.5][k]).collect()
            };
            let t: Vec<i64> = rows.iter().map(|r| r.0).collect();
            let (o, h, l, c, v) = (col(0), col(1), col(2), col(3), col(4));
            let bars = BarColumns { open_time: &t, open: &o, high: &h, low: &l, close: &c, volume: &v }
                .to_bars()
                .unwrap();
            prop_assert_eq!(bars.len(), rows.len());
            for (b, r) in bars.iter().zip(&rows) {
                prop_assert_eq!(b.open_time, r.0);
                prop_assert!(same_bits(
                    &[b.open, b.high, b.low, b.close, b.volume],
                    &[r.1, r.2, r.3, r.4, r.5]
                ));
            }
        }

        #[test]
        fn fill_columns_round_trip_bit_for_bit(fills in prop::collection::vec(any_fill(), 0..40)) {
            let cols = FillColumns::from_fills(&fills);
            prop_assert_eq!(cols.len(), fills.len());
            let back = cols.to_fills().unwrap();
            for (a, b) in fills.iter().zip(&back) {
                prop_assert_eq!((a.bar, a.time, a.side), (b.bar, b.time, b.side));
                prop_assert!(same_bits(
                    &[a.qty, a.price, a.notional, a.fee],
                    &[b.qty, b.price, b.notional, b.fee]
                ));
            }
            for (i, f) in fills.iter().enumerate() {
                prop_assert_eq!(cols.side[i] == BUY, f.side == Side::Buy);
                prop_assert!(cols.side[i] == BUY || cols.side[i] == SELL);
            }
        }

        #[test]
        fn row_major_is_numpy_order(rows in 1usize..6, cols in 1usize..6, seed in any::<u64>()) {
            let data: Vec<f64> = (0..rows * cols)
                .map(|k| f64::from_bits(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k as u64)))
                .collect();
            let m = dmatrix_from_row_major(rows, cols, &data);
            for i in 0..rows {
                for j in 0..cols {
                    prop_assert_eq!(m[(i, j)].to_bits(), data[i * cols + j].to_bits());
                }
            }
            prop_assert!(same_bits(&row_major(&m), &data));
        }

        /// `extend_matrix` is all-or-nothing, and when it accepts it is exactly the sequence of
        /// pushes: same entries, same tree, same statistics on every range, bit for bit. When it
        /// refuses, the error is the first one the pushes would have hit.
        #[test]
        fn a_batch_push_is_the_pushes_or_nothing(
            prefix in prop::collection::vec((-5i64..5, -3.0..3.0f64, -3.0..3.0f64), 0..12),
            batch in prop::collection::vec((-5i64..5, -3.0..3.0f64, -3.0..3.0f64, 0u8..20), 0..24),
        ) {
            let (p, q) = (2, 1);
            let mut base = Matrix::new(p, q, 1).unwrap();
            let mut t0 = i64::MIN;
            for &(dt, a, y) in &prefix {
                t0 = t0.max(dt);
                base.push(t0, &[1.0, a], &[y]).unwrap();
            }
            let mut times = Vec::new();
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut t = base.len().checked_sub(1).map_or(0, |i| base.time(i));
            for &(dt, a, y, poison) in &batch {
                // Mostly forward in time, sometimes backwards; rarely a non-finite value.
                t += if poison == 0 { -1 } else { dt.abs() };
                times.push(t);
                xs.extend([1.0, if poison == 1 { f64::NAN } else { a }]);
                ys.push(if poison == 2 { f64::INFINITY } else { y });
            }

            let mut sequential = base.clone();
            let expected = times.iter().enumerate().try_for_each(|(i, &time)| {
                sequential.push(time, &xs[i * p..(i + 1) * p], &ys[i..=i])
            });
            let mut batched = base.clone();
            let got = extend_matrix(&mut batched, &times, &xs, &ys);
            prop_assert_eq!(&got, &expected);
            let reference = if got.is_ok() { &sequential } else { &base };
            prop_assert_eq!(batched.len(), reference.len());
            for i in 0..batched.len() {
                prop_assert_eq!(batched.time(i), reference.time(i));
                prop_assert!(same_bits(batched.x(i), reference.x(i)));
                prop_assert!(same_bits(batched.y(i), reference.y(i)));
            }
            for start in 0..=batched.len() {
                for end in start..=batched.len() {
                    let a: SuffStats = batched.range_stats(start, end).unwrap();
                    let b = reference.range_stats(start, end).unwrap();
                    prop_assert!(same_bits(a.gram.as_slice(), b.gram.as_slice()));
                    prop_assert!(same_bits(a.cross.as_slice(), b.cross.as_slice()));
                    prop_assert!(same_bits(a.sum_yy.as_slice(), b.sum_yy.as_slice()));
                    prop_assert_eq!(a.count, b.count);
                }
            }
        }
    }

    #[test]
    fn a_batch_of_the_wrong_shape_is_refused_whole() {
        let mut m = Matrix::new(2, 1, 0).unwrap();
        assert_eq!(
            extend_matrix(&mut m, &[0, 1], &[1.0, 2.0, 3.0], &[0.0, 0.0]),
            Err(OlsError::Dimension {
                expected: 4,
                got: 3
            })
        );
        assert_eq!(
            extend_matrix(&mut m, &[0, 1], &[1.0; 4], &[0.0]),
            Err(OlsError::Dimension {
                expected: 2,
                got: 1
            })
        );
        assert!(m.is_empty());
        extend_matrix(&mut m, &[], &[], &[]).unwrap();
        assert!(m.is_empty());
    }
}
