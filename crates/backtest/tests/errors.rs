//! Every `SimError` variant is reachable, and validation runs before any bar.

mod common;

use backtest::{FeeSchedule, SimConfig, SimError, simulate};
use common::*;

fn cfg() -> SimConfig {
    SimConfig::new(1_000.0, FeeSchedule::BINANCE_SPOT)
}

#[test]
fn empty_input() {
    assert_eq!(simulate(&[], &[], &cfg()), Err(SimError::Empty));
}

#[test]
fn length_mismatch() {
    let b = flat_bars(3, 10.0);
    assert_eq!(
        simulate(&b, &[0.0, 1.0], &cfg()),
        Err(SimError::LengthMismatch {
            bars: 3,
            targets: 2
        })
    );
}

#[test]
fn invalid_bars() {
    type Edit = fn(&mut backtest::Bar);
    let cases: [(Edit, &str); 8] = [
        (|b| b.open = 0.0, "OHLC"),
        (|b| b.close = -1.0, "OHLC"),
        (|b| b.high = f64::INFINITY, "OHLC"),
        (|b| b.low = f64::NAN, "OHLC"),
        (|b| b.low = b.open.min(b.close) * 1.001, "low"),
        (|b| b.high = b.open.max(b.close) * 0.999, "high"),
        (|b| b.volume = -1.0, "volume"),
        (|b| b.volume = f64::NAN, "volume"),
    ];
    for (edit, needle) in cases {
        let mut b = bars(&[(10.0, 11.0), (11.0, 10.5), (10.4, 10.9)]);
        edit(&mut b[2]);
        match simulate(&b, &[0.0; 3], &cfg()) {
            Err(SimError::InvalidBar { index: 2, reason }) => {
                assert!(reason.contains(needle), "{reason}")
            }
            other => panic!("expected InvalidBar at 2, got {other:?}"),
        }
    }
}

#[test]
fn non_increasing_time() {
    let mut b = flat_bars(4, 10.0);
    b[2].open_time = b[1].open_time;
    assert_eq!(
        simulate(&b, &[0.0; 4], &cfg()),
        Err(SimError::NonIncreasingTime { index: 2 })
    );
    b[2].open_time = b[1].open_time - 1;
    assert_eq!(
        simulate(&b, &[0.0; 4], &cfg()),
        Err(SimError::NonIncreasingTime { index: 2 })
    );
}

#[test]
fn invalid_targets() {
    let b = flat_bars(3, 10.0);
    for bad in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
        match simulate(&b, &[0.0, 0.5, bad], &cfg()) {
            Err(SimError::InvalidTarget { index: 2, value }) => {
                assert!(value.to_bits() == bad.to_bits())
            }
            other => panic!("expected InvalidTarget for {bad}, got {other:?}"),
        }
    }
    // The bounds themselves, and -0.0, are valid.
    assert!(simulate(&b, &[0.0, 1.0, -0.0], &cfg()).is_ok());
}

#[test]
fn invalid_configs() {
    let b = flat_bars(3, 10.0);
    let t = [0.0; 3];
    type Edit = fn(&mut SimConfig);
    let cases: [(Edit, &str); 13] = [
        (|c| c.initial_cash = 0.0, "initial_cash"),
        (|c| c.initial_cash = f64::NAN, "initial_cash"),
        (|c| c.initial_cash = f64::INFINITY, "initial_cash"),
        (|c| c.fees.maker = -1e-9, "fees.maker"),
        (|c| c.fees.taker = 1.0, "fees.taker"),
        (|c| c.fees.taker = f64::NAN, "fees.taker"),
        (|c| c.slippage_bps = -0.1, "slippage_bps"),
        (|c| c.slippage_bps = 10_000.0, "slippage_bps"),
        (|c| c.rebalance_threshold = 1.0, "rebalance_threshold"),
        (|c| c.rebalance_threshold = -0.1, "rebalance_threshold"),
        (|c| c.filters.min_notional = -5.0, "filters.min_notional"),
        (|c| c.filters.step_size = f64::INFINITY, "filters.step_size"),
        (|c| c.filters.step_size = f64::NAN, "filters.step_size"),
    ];
    for (edit, name) in cases {
        let mut c = cfg();
        edit(&mut c);
        match simulate(&b, &t, &c) {
            Err(SimError::InvalidConfig { field, .. }) => assert_eq!(field, name),
            other => panic!("expected InvalidConfig({name}), got {other:?}"),
        }
    }
}

#[test]
fn errors_render() {
    assert_eq!(SimError::Empty.to_string(), "no bars");
    assert!(
        SimError::NonIncreasingTime { index: 4 }
            .to_string()
            .contains("bar 4")
    );
    // Index 0 never comes out of `simulate`, but rendering it must not panic.
    assert!(
        SimError::NonIncreasingTime { index: 0 }
            .to_string()
            .contains("bar 0")
    );
    let target = SimError::InvalidTarget {
        index: 1,
        value: 2.0,
    };
    assert!(target.to_string().contains("[0, 1]"));
    let config = SimError::InvalidConfig {
        field: "slippage_bps",
        value: -1.0,
    };
    assert_eq!(config.to_string(), "invalid config: slippage_bps = -1");
}

/// The real Binance series have truncated bars with volume 0, gaps and
/// off-grid bars. None of these is an error: only strict time order is.
#[test]
fn zero_volume_gaps_and_off_grid_bars_are_valid() {
    let mut b = bars(&[(10.0, 11.0), (11.0, 10.5), (10.4, 10.9), (10.9, 10.9)]);
    b[1].volume = 0.0;
    b[2].open_time += 5 * HOUR; // a gap of five missing bars
    b[3].open_time = b[2].open_time + 28 * 60_000; // 28 min off the grid
    b[3].volume = 0.0;
    let r = simulate(&b, &[1.0, 0.0, 1.0, 0.0], &cfg()).unwrap();
    // Across the gap, the decision at close 1 executes at the next available
    // open, bar 2.
    let bars_filled: Vec<_> = r.fills.iter().map(|f| (f.bar, f.time)).collect();
    assert_eq!(
        bars_filled,
        [
            (1, b[1].open_time),
            (2, b[2].open_time),
            (3, b[3].open_time)
        ]
    );
}

/// All inputs are validated before the first bar: a bad target at the very
/// end is reported even though the bars before it are fine, and the config
/// is checked first.
#[test]
fn validation_precedes_simulation() {
    let b = flat_bars(50, 10.0);
    let mut t = vec![0.5; 50];
    t[49] = 2.0;
    assert_eq!(
        simulate(&b, &t, &cfg()),
        Err(SimError::InvalidTarget {
            index: 49,
            value: 2.0
        })
    );
    let mut c = cfg();
    c.slippage_bps = f64::NAN;
    assert!(matches!(
        simulate(&[], &t, &c),
        Err(SimError::InvalidConfig {
            field: "slippage_bps",
            ..
        })
    ));
}
