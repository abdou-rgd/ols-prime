//! Metrics on hand-computed series.

use backtest::metrics::{compute, max_drawdown, periods_per_year};
use backtest::{Fill, Side, SimResult};

fn result(equity: Vec<f64>, position: Vec<f64>, fills: Vec<Fill>) -> SimResult {
    let fees_paid = fills.iter().map(|f| f.fee).sum();
    SimResult {
        cash: equity.clone(),
        equity,
        position,
        fills,
        fees_paid,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

#[test]
fn max_drawdown_hand_cases() {
    assert_eq!(max_drawdown(&[100.0, 120.0, 90.0, 130.0]), 0.25);
    assert_eq!(max_drawdown(&[]), 0.0);
    assert_eq!(max_drawdown(&[1.0, 2.0, 3.0]), 0.0);
    assert_eq!(max_drawdown(&[100.0, 50.0, 200.0, 150.0]), 0.5);
}

#[test]
fn periods_per_year_counts_every_day() {
    assert_eq!(periods_per_year(3_600_000), 8760.0);
    assert_eq!(periods_per_year(86_400_000), 365.0);
    assert_eq!(periods_per_year(60_000), 525_600.0);
}

#[test]
#[should_panic(expected = "bar length must be positive")]
fn periods_per_year_rejects_zero() {
    periods_per_year(0);
}

#[test]
fn constant_equity_has_no_sharpe() {
    let m = compute(&result(vec![100.0; 10], vec![0.0; 10], vec![]), 365.0);
    assert_eq!(m.sharpe, None);
    assert_eq!(m.sortino, None);
    assert_eq!(m.annualized_volatility, 0.0);
    assert_eq!(m.total_return, 0.0);
    assert_eq!(m.annualized_return, 0.0);
    assert_eq!(m.max_drawdown, 0.0);
    assert_eq!(m.exposure, 0.0);
    assert_eq!(m.periods, 9);
}

#[test]
fn hand_computed_ratios() {
    // Returns +10 %, −10 %, +10 %: mean 1/30, sample std 1/√75, downside
    // deviation √(0.01/3).
    let fill = Fill {
        bar: 1,
        time: 0,
        side: Side::Buy,
        qty: 1.0,
        price: 100.0,
        notional: 100.0,
        fee: 0.1,
    };
    let eq = vec![100.0, 110.0, 99.0, 108.9];
    let m = compute(&result(eq, vec![0.0, 1.0, 1.0, 0.0], vec![fill]), 4.0);
    assert_eq!(m.periods, 3);
    assert!(close(m.total_return, 0.089));
    assert!(close(m.annualized_return, 1.089f64.powf(4.0 / 3.0) - 1.0));
    assert!(close(m.annualized_volatility, 2.0 / 75f64.sqrt()));
    assert!(close(m.sharpe.unwrap(), 2.0 * 75f64.sqrt() / 30.0));
    assert!(close(m.sortino.unwrap(), 2.0 * 3f64.sqrt() / 3.0));
    assert!(close(m.max_drawdown, 0.1));
    // Notional 100 over mean equity 104.475, over 3 periods, × 4 per year.
    assert!(close(m.turnover, 100.0 / 104.475 * 4.0 / 3.0));
    assert_eq!(m.exposure, 0.5);
    assert_eq!(m.fees_paid, 0.1);
}

#[test]
fn no_losing_period_means_no_sortino() {
    let m = compute(
        &result(vec![100.0, 101.0, 103.0], vec![1.0; 3], vec![]),
        365.0,
    );
    assert!(m.sharpe.is_some());
    assert_eq!(m.sortino, None);
    assert_eq!(m.exposure, 1.0);
}

#[test]
fn one_bar_is_degenerate_but_defined() {
    let m = compute(&result(vec![100.0], vec![0.0], vec![]), 365.0);
    assert_eq!(m.periods, 0);
    assert_eq!((m.annualized_return, m.turnover), (0.0, 0.0));
    assert_eq!((m.sharpe, m.sortino), (None, None));
}

/// Identical returns have no standard deviation, so no Sharpe ratio, even
/// where the rounding of their mean would leave a spurious variance.
#[test]
fn identical_nonzero_returns_have_no_sharpe() {
    // Build equity curves whose per-bar returns are bitwise identical, then
    // keep one where the naive two-pass variance is not 0.
    let naive_var = |r: &[f64]| {
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        r.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
    };
    let mut found = None;
    'search: for i in 1..1_000 {
        let growth = 1.0 + i as f64 * 1e-3;
        let mut eq = vec![100.0, 100.0 * growth];
        let r0 = eq[1] / eq[0] - 1.0;
        for _ in 0..5 {
            let prev = *eq.last().unwrap();
            let mut x = prev * growth;
            let mut tries = 0;
            while x / prev - 1.0 != r0 {
                x = if x / prev - 1.0 < r0 {
                    x.next_up()
                } else {
                    x.next_down()
                };
                tries += 1;
                if tries > 64 {
                    continue 'search;
                }
            }
            eq.push(x);
        }
        let r: Vec<f64> = eq.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
        if naive_var(&r) > 0.0 {
            found = Some(eq);
            break;
        }
    }
    let eq = found.expect("no curve with a spurious naive variance: the test is vacuous");
    let n = eq.len();
    let m = compute(&result(eq, vec![1.0; n], vec![]), 365.0);
    assert_eq!(m.sharpe, None);
    assert_eq!(m.annualized_volatility, 0.0);
    assert_eq!(m.sortino, None);
}

#[test]
fn sortino_divides_by_all_periods() {
    // Returns +20 %, −10 %, −5 %, +10 %: downside deviation
    // √((0.01 + 0.0025) / 4), not over the two losing periods only.
    let eq = vec![100.0, 120.0, 108.0, 102.6, 112.86];
    let r = [0.2, -0.1, -0.05, 0.1];
    let mean = r.iter().sum::<f64>() / 4.0;
    let m = compute(&result(eq, vec![1.0; 5], vec![]), 52.0);
    let dd = ((0.01f64 + 0.0025) / 4.0).sqrt();
    assert!(close(m.sortino.unwrap(), mean / dd * 52f64.sqrt()));
    assert_eq!(m.exposure, 1.0);
}

#[test]
fn periods_per_year_common_bars() {
    assert_eq!(periods_per_year(4 * 3_600_000), 2190.0);
    assert_eq!(periods_per_year(15 * 60_000), 35_040.0);
    assert_eq!(periods_per_year(7 * 86_400_000), 365.0 / 7.0);
}

mod props {
    use super::result;
    use backtest::metrics::{compute, max_drawdown};
    use backtest::{Fill, Side};
    use proptest::prelude::*;

    fn arb_equity() -> impl Strategy<Value = Vec<f64>> {
        (1usize..=64).prop_flat_map(|n| {
            (
                1.0f64..1e6,
                prop::collection::vec(prop_oneof![Just(0.0), -0.3f64..0.3], n - 1),
            )
                .prop_map(|(e0, steps)| {
                    let mut e = vec![e0];
                    for s in steps {
                        let last = *e.last().unwrap();
                        e.push(last * s.exp());
                    }
                    e
                })
        })
    }

    fn fill(notional: f64) -> Fill {
        Fill {
            bar: 1,
            time: 0,
            side: Side::Buy,
            qty: 1.0,
            price: notional,
            notional,
            fee: 0.0,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        /// The running-peak drawdown equals the O(n²) definition
        /// `max_{i ≤ j} (1 − e_j / e_i)` bit for bit: `1 − a / b` is monotone
        /// in `b` under correctly rounded division, so the maximum over `i`
        /// is reached at the running peak.
        #[test]
        fn drawdown_matches_its_definition(eq in arb_equity()) {
            let mut brute: f64 = 0.0;
            for j in 0..eq.len() {
                for i in 0..=j {
                    brute = brute.max(1.0 - eq[j] / eq[i]);
                }
            }
            let mdd = max_drawdown(&eq);
            prop_assert_eq!(mdd.to_bits(), brute.to_bits());
            prop_assert!((0.0..=1.0).contains(&mdd));
        }

        /// Scaling the equity curve and the traded notional by 2ʲ changes no
        /// ratio, bit for bit: the metrics have no hidden level dependence.
        #[test]
        fn ratios_are_scale_free(eq in arb_equity(), notional in 0.0f64..1e6, j in -20i32..=20) {
            let k = 2f64.powi(j);
            let n = eq.len();
            let a = compute(&result(eq.clone(), vec![1.0; n], vec![fill(notional)]), 8760.0);
            let scaled: Vec<f64> = eq.iter().map(|e| e * k).collect();
            let b = compute(&result(scaled, vec![1.0; n], vec![fill(notional * k)]), 8760.0);
            prop_assert_eq!(a.total_return.to_bits(), b.total_return.to_bits());
            prop_assert_eq!(a.annualized_return.to_bits(), b.annualized_return.to_bits());
            prop_assert_eq!(a.annualized_volatility.to_bits(), b.annualized_volatility.to_bits());
            prop_assert_eq!(a.sharpe.map(f64::to_bits), b.sharpe.map(f64::to_bits));
            prop_assert_eq!(a.sortino.map(f64::to_bits), b.sortino.map(f64::to_bits));
            prop_assert_eq!(a.max_drawdown.to_bits(), b.max_drawdown.to_bits());
            prop_assert_eq!(a.turnover.to_bits(), b.turnover.to_bits());
        }

        /// Annualization: ×4ʲ periods per year scales volatility, Sharpe and
        /// Sortino by 2ʲ and turnover by 4ʲ, bit for bit (√(4ʲx) = 2ʲ√x
        /// exactly), and the annualized return is the geometric
        /// `(1 + total)^(ppy / periods) − 1`.
        #[test]
        fn annualization_scales_exactly(eq in arb_equity(), notional in 0.0f64..1e6, j in -3i32..=3) {
            let n = eq.len();
            let r = result(eq, vec![0.0; n], vec![fill(notional)]);
            let a = compute(&r, 365.0);
            let b = compute(&r, 365.0 * 4f64.powi(j));
            let two = 2f64.powi(j);
            prop_assert_eq!((a.annualized_volatility * two).to_bits(), b.annualized_volatility.to_bits());
            prop_assert_eq!(a.sharpe.map(|x| (x * two).to_bits()), b.sharpe.map(f64::to_bits));
            prop_assert_eq!(a.sortino.map(|x| (x * two).to_bits()), b.sortino.map(f64::to_bits));
            prop_assert_eq!((a.turnover * two * two).to_bits(), b.turnover.to_bits());
            prop_assert_eq!(a.total_return.to_bits(), b.total_return.to_bits());
            if a.periods > 0 {
                let expect = (1.0 + a.total_return).powf(365.0 / a.periods as f64) - 1.0;
                prop_assert_eq!(a.annualized_return.to_bits(), expect.to_bits());
            }
            prop_assert_eq!(a.exposure, 0.0);
        }
    }
}
