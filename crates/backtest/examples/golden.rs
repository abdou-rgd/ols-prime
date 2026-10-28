//! Golden cases for [`backtest::simulate`]: inputs and outputs as exact bits.
//!
//! The Python binding claims to return exactly what the simulator computes. A Python port of
//! the simulator can share a bug with it; the compiled crate cannot. So this example runs the
//! crate itself on fixed cases and writes everything to a committed text file, and
//! `python/tests/test_rust_bindings.py` replays the inputs through `olsprime_rs.simulate` and
//! compares every output bit for bit.
//!
//! ```text
//! cargo run -p backtest --example golden              # regenerate the committed file
//! cargo run -p backtest --example golden -- --check   # compare without writing
//! cargo run -p backtest --example golden -- <path>    # write elsewhere
//! ```
//!
//! `cargo test -p backtest --test golden` regenerates the cases and compares them with the
//! committed file, so any change to the simulator, intended or not, fails until the file is
//! regenerated on purpose and the diff reviewed.
//!
//! # Portability
//!
//! The file must come out identical on every platform. The cases are built from SplitMix64 and
//! IEEE-754 basic operations only (`+ − × /`, comparisons, `floor`, `min`, `max`), which are
//! correctly rounded or exact everywhere; nothing calls `exp`, `ln` or `powf`, whose last bit
//! depends on the platform's libm. The simulator itself uses only the same operations.
//!
//! # Format
//!
//! Line-oriented and versioned. `#` starts a comment line. Every float is the 16 hex digits of
//! its binary64 bit pattern (so NaN, −0.0 and subnormals survive), times and indices are
//! decimal. Per case:
//!
//! ```text
//! case <name>
//! fees <name> <maker> <taker>       a built-in name, or a custom one
//! liquidity <taker|maker>
//! initial_cash <f>
//! slippage_bps <f>
//! min_notional <f>
//! step_size <f>
//! rebalance_threshold <f>
//! bars <n>
//! b <open_time> <open> <high> <low> <close> <volume>        n lines
//! targets <k>
//! t <target>                                                k lines
//! error <message>                   simulate refused: the SimError's Display, or
//! ok
//! r <equity> <position> <cash>                              n lines
//! fills <m>
//! f <bar> <time> <buy|sell> <qty> <price> <notional> <fee>  m lines
//! fees_paid <f>
//! end
//! ```

use std::fmt::Write as _;

use backtest::{Bar, FeeSchedule, Liquidity, Side, SimConfig, SymbolFilters, simulate};

/// The committed file.
pub const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/tests/golden/simulate.txt"
);

/// The format version, the first non-comment line of the file.
pub const FORMAT: &str = "format olsprime-golden-simulate 1";

/// The seed of every random case.
pub const SEED: u64 = 0x0150_5052_494D_4501;

/// The name a custom fee schedule carries inside [`FeeSchedule`]; [`Case::fee_name`] keeps the
/// file's. The name plays no part in the simulation.
const CUSTOM: &str = "custom";

/// One call of [`simulate`]: its inputs only. The outputs are recomputed by [`render`].
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub name: String,
    /// The schedule's name in the file: a built-in name, or a custom one.
    pub fee_name: String,
    pub cfg: SimConfig,
    pub bars: Vec<Bar>,
    pub targets: Vec<f64>,
}

/// SplitMix64 (Steele, Lea and Flood, 2014): tiny, and the same on every platform.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1): the top 53 bits, scaled exactly.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (with a negligible modulo bias).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

const T0: i64 = 1_700_000_000_000;
const MINUTE: i64 = 60_000;

/// A valid random walk of `n` bars from `p0`, with moves of relative size up to `vol` (< 1),
/// wicks, a few zero volumes and, with `gaps`, missing bars on the minute grid.
fn random_bars(rng: &mut SplitMix64, n: usize, p0: f64, vol: f64, gaps: bool) -> Vec<Bar> {
    let mut prev = p0;
    let mut t = T0;
    (0..n)
        .map(|_| {
            let open = prev * (1.0 + 0.2 * vol * (rng.unit() - 0.5));
            let close = open * (1.0 + 2.0 * vol * (rng.unit() - 0.5));
            let high = open.max(close) * (1.0 + 0.5 * vol * rng.unit());
            let low = open.min(close) * (1.0 - 0.5 * vol * rng.unit());
            let volume = if rng.below(10) == 0 {
                0.0
            } else {
                100.0 * rng.unit()
            };
            let bar = Bar {
                open_time: t,
                open,
                high,
                low,
                close,
                volume,
            };
            let skip = if gaps && rng.below(6) == 0 {
                1 + rng.below(4) as i64
            } else {
                0
            };
            t += MINUTE * (1 + skip);
            prev = close;
            bar
        })
        .collect()
}

/// Flat, full, held and fractional targets, with the bit-level edges of [0, 1] now and then.
fn random_targets(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    let mut prev = 0.0;
    (0..n)
        .map(|_| {
            let t = match rng.below(16) {
                0..=3 => 0.0,
                4..=7 => 1.0,
                8..=10 => prev,
                11..=13 => rng.unit(),
                14 => rng.pick(&[-0.0, f64::from_bits(1), 1.0 - f64::EPSILON / 2.0, 0.5]),
                _ => 0.01 * rng.unit(),
            };
            prev = t;
            t
        })
        .collect()
}

/// Bars on the minute grid from `(open, high, low, close)`, volume 1.
fn ohlc(rows: &[(f64, f64, f64, f64)]) -> Vec<Bar> {
    rows.iter()
        .enumerate()
        .map(|(i, &(open, high, low, close))| Bar {
            open_time: T0 + i as i64 * MINUTE,
            open,
            high,
            low,
            close,
            volume: 1.0,
        })
        .collect()
}

fn flat(n: usize, price: f64) -> Vec<Bar> {
    ohlc(&vec![(price, price, price, price); n])
}

fn case(
    name: &str,
    fees: FeeSchedule,
    bars: Vec<Bar>,
    targets: Vec<f64>,
    tweak: impl FnOnce(&mut SimConfig),
) -> Case {
    let mut cfg = SimConfig::new(10_000.0, fees);
    tweak(&mut cfg);
    Case {
        name: name.to_owned(),
        fee_name: fees.name.to_owned(),
        cfg,
        bars,
        targets,
    }
}

fn custom_fees(maker: f64, taker: f64) -> FeeSchedule {
    FeeSchedule {
        name: CUSTOM,
        maker,
        taker,
    }
}

/// Hand-made cases, each aimed at one rule of the execution model.
fn edge_cases(rng: &mut SplitMix64) -> Vec<Case> {
    let bin = FeeSchedule::BINANCE_SPOT;
    let zero = FeeSchedule::ZERO;
    let mut out = vec![
        // The worked example of the binding's tests: buy at open 1, exit at open 2.
        case(
            "three-bars-round-trip",
            bin,
            {
                let mut b = ohlc(&[
                    (100.0, 101.0, 99.0, 100.0),
                    (125.0, 131.0, 124.0, 130.0),
                    (140.0, 151.0, 139.0, 150.0),
                ]);
                for (i, bar) in b.iter_mut().enumerate() {
                    bar.open_time = i as i64 * MINUTE;
                }
                b
            },
            vec![0.5, 0.0, 1.0],
            |c| {
                c.initial_cash = 1000.0;
                c.slippage_bps = 0.0;
            },
        ),
        // targets[last] never executes, so a single bar never trades.
        case("one-bar", bin, flat(1, 100.0), vec![1.0], |_| {}),
        // A cash-capped buy leaves a few ulps of cash; the dust floor stops it being re-bought
        // at every bar.
        case(
            "all-in-dust",
            bin,
            random_bars(rng, 25, 100.0, 0.02, false),
            vec![1.0; 25],
            |_| {},
        ),
        case(
            "all-in-dust-lots",
            bin,
            random_bars(rng, 25, 67_000.0, 0.02, false),
            vec![1.0; 25],
            |c| c.filters = SymbolFilters::BINANCE_BTCUSDT,
        ),
        // A buy beyond 2⁵⁰ lots is clamped, and a held clamp cannot buy more.
        case(
            "max-lots-clamp",
            zero,
            flat(7, 1.0),
            vec![1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 1.0],
            |c| {
                c.initial_cash = 1e15;
                c.slippage_bps = 0.0;
                c.filters.step_size = 1e-8;
            },
        ),
        // A full exit below the minimum notional is skipped: the position stays.
        case(
            "exit-below-min-notional",
            zero,
            ohlc(&[
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 100.0, 100.0, 100.0),
                (40.0, 40.0, 40.0, 40.0),
                (40.0, 40.0, 40.0, 40.0),
                (40.0, 40.0, 40.0, 40.0),
            ]),
            vec![0.5, 0.5, 0.0, 0.0, 0.0],
            |c| {
                c.initial_cash = 100.0;
                c.slippage_bps = 0.0;
                c.filters.min_notional = 30.0;
            },
        ),
        // A lot worth more than the account: every order floors to 0 lots.
        case(
            "coarse-lot-never-fills",
            bin,
            random_bars(rng, 15, 67_000.0, 0.01, false),
            random_targets(rng, 15),
            |c| c.filters.step_size = 1.0,
        ),
        // −0.0 is a valid target equal to 0 (a full exit); the smallest subnormal is a valid
        // target whose order is dust; 1 − 2⁻⁵³ is the largest target below 1.
        case(
            "signed-zero-and-subnormal-targets",
            bin,
            random_bars(rng, 11, 3_500.0, 0.02, false),
            vec![
                f64::from_bits(1),
                1.0,
                -0.0,
                1.0 - f64::EPSILON / 2.0,
                f64::from_bits(1),
                0.5,
                -0.0,
                1.0,
                0.0,
                1.0,
                -0.0,
            ],
            |_| {},
        ),
        // No grid is enforced: gaps and zero volumes are valid.
        case(
            "gaps-and-zero-volume",
            bin,
            {
                let mut b = random_bars(rng, 20, 250.0, 0.03, true);
                for bar in b.iter_mut().step_by(2) {
                    bar.volume = 0.0;
                }
                b
            },
            random_targets(rng, 20),
            |_| {},
        ),
        case(
            "tiny-prices-whole-tokens",
            FeeSchedule::KRAKEN_PRO_ENTRY,
            random_bars(rng, 20, 1.234e-8, 0.05, false),
            random_targets(rng, 20),
            |c| {
                c.filters = SymbolFilters {
                    min_notional: 5.0,
                    step_size: 1.0,
                }
            },
        ),
        case(
            "huge-prices",
            FeeSchedule::BITVAVO_ENTRY,
            random_bars(rng, 15, 1e9, 0.03, false),
            random_targets(rng, 15),
            |c| c.initial_cash = 1e6,
        ),
        // A target of 0 exits in full whatever the threshold; others wait for the threshold.
        case(
            "threshold-and-full-exit",
            bin,
            flat(6, 2_000.0),
            vec![0.95, 0.1, 0.5, 0.0, 0.95, 0.0],
            |c| {
                c.initial_cash = 1000.0;
                c.rebalance_threshold = 0.9;
            },
        ),
        case(
            "maker-coinbase",
            FeeSchedule::COINBASE_ADVANCED_ENTRY,
            random_bars(rng, 15, 3_500.0, 0.02, false),
            random_targets(rng, 15),
            |c| {
                c.liquidity = Liquidity::Maker;
                c.slippage_bps = 5.0;
                c.filters = SymbolFilters::BINANCE_ETHUSDT;
            },
        ),
    ];
    let mut custom = case(
        "custom-fees",
        custom_fees(0.000_3, 0.000_7),
        random_bars(rng, 15, 42.0, 0.02, false),
        random_targets(rng, 15),
        |c| c.filters.min_notional = 50.0,
    );
    custom.fee_name = "golden-venue".to_owned();
    out.push(custom);
    out
}

/// Cases [`simulate`] must refuse, each with one defect, all reachable through the binding.
fn error_cases() -> Vec<Case> {
    let bin = FeeSchedule::BINANCE_SPOT;
    let base = || {
        ohlc(&[
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 102.0, 98.0, 101.0),
            (101.0, 103.0, 100.0, 102.0),
        ])
    };
    let bad_bar = |name: &str, edit: fn(&mut Bar)| {
        let mut b = base();
        edit(&mut b[1]);
        case(name, bin, b, vec![0.0; 3], |_| {})
    };
    let bad_cfg =
        |name: &str, edit: fn(&mut SimConfig)| case(name, bin, base(), vec![0.0; 3], edit);
    let bad_target = |name: &str, t: f64| case(name, bin, base(), vec![0.5, t, 0.0], |_| {});
    vec![
        case("err-no-bars", bin, vec![], vec![], |_| {}),
        case("err-length-mismatch", bin, base(), vec![0.0; 2], |_| {}),
        bad_bar("err-nonpositive-open", |b| b.open = 0.0),
        bad_bar("err-infinite-high", |b| b.high = f64::INFINITY),
        bad_bar("err-nan-close", |b| b.close = f64::NAN),
        bad_bar("err-low-above-body", |b| b.low = 100.5),
        bad_bar("err-high-below-body", |b| b.high = 100.5),
        bad_bar("err-negative-volume", |b| b.volume = -1.0),
        bad_bar("err-nan-volume", |b| b.volume = f64::NAN),
        bad_bar("err-equal-times", |b| b.open_time = T0),
        bad_bar("err-decreasing-times", |b| b.open_time = T0 - 1),
        bad_target("err-target-nan", f64::NAN),
        bad_target("err-target-above-one", 1.0 + f64::EPSILON),
        bad_target("err-target-negative", -0.5),
        bad_cfg("err-cash-zero", |c| c.initial_cash = 0.0),
        bad_cfg("err-cash-nan", |c| c.initial_cash = f64::NAN),
        bad_cfg("err-cash-infinite", |c| c.initial_cash = f64::INFINITY),
        bad_cfg("err-slippage-10000", |c| c.slippage_bps = 10_000.0),
        bad_cfg("err-slippage-negative", |c| c.slippage_bps = -0.5),
        bad_cfg("err-threshold-one", |c| c.rebalance_threshold = 1.0),
        bad_cfg("err-min-notional-negative", |c| {
            c.filters.min_notional = -1.0
        }),
        bad_cfg("err-step-nan", |c| c.filters.step_size = f64::NAN),
        // The config is validated before the bars, and the bars before the targets.
        {
            let mut b = base();
            b[0].open = -1.0;
            case("err-config-before-bars", bin, b, vec![2.0; 3], |c| {
                c.initial_cash = -1.0
            })
        },
        {
            let mut b = base();
            b[2].open_time = T0;
            case("err-bars-before-targets", bin, b, vec![2.0; 3], |_| {})
        },
    ]
}

/// Random configurations on random walks.
fn random_cases(rng: &mut SplitMix64, count: usize) -> Vec<Case> {
    (0..count)
        .map(|i| {
            let n = 15 + rng.below(30) as usize;
            let p0 = rng.pick(&[1.234e-5, 0.5, 1.0, 3_500.0, 67_000.0]);
            let vol = rng.pick(&[0.005, 0.02, 0.05]);
            let gaps = rng.below(3) == 0;
            let bars = random_bars(rng, n, p0, vol, gaps);
            let targets = random_targets(rng, n);
            let custom = rng.below(7) == 0;
            let fees = if custom {
                custom_fees(0.003 * rng.unit(), 0.006 * rng.unit())
            } else {
                rng.pick(&FeeSchedule::ALL)
            };
            let mut c = case(&format!("random-{i:02}"), fees, bars, targets, |c| {
                c.initial_cash = rng.pick(&[10_000.0, 1_000.0, 1e6, 123.456, 50.0]);
                if rng.below(4) == 0 {
                    c.liquidity = Liquidity::Maker;
                }
                let u = rng.unit();
                c.slippage_bps = rng.pick(&[0.0, 2.0, 7.5, 30.0 * u]);
                c.filters = match rng.below(4) {
                    0 => SymbolFilters::NONE,
                    1 => SymbolFilters::BINANCE_BTCUSDT,
                    2 => SymbolFilters::BINANCE_ETHUSDT,
                    _ => SymbolFilters {
                        min_notional: 50.0 * rng.unit(),
                        step_size: rng.pick(&[1e-8, 0.001, 0.3, 1.0]),
                    },
                };
                let u = rng.unit();
                c.rebalance_threshold = rng.pick(&[0.0, 0.0, 0.05, 0.1, 0.2 * u]);
            });
            if custom {
                c.fee_name = format!("golden-venue-{i:02}");
            }
            c
        })
        .collect()
}

/// Every case of the committed file, in order.
pub fn cases() -> Vec<Case> {
    let mut rng = SplitMix64::new(SEED);
    let mut out = edge_cases(&mut rng);
    out.extend(error_cases());
    out.extend(random_cases(&mut rng, 10));
    out
}

fn hex(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn liquidity_name(l: Liquidity) -> &'static str {
    match l {
        Liquidity::Taker => "taker",
        Liquidity::Maker => "maker",
    }
}

/// The file: every case's inputs, and the outputs [`simulate`] computes for them now.
pub fn render(cases: &[Case]) -> String {
    let mut s = String::new();
    // `write!` into a String cannot fail.
    let mut w = |args: std::fmt::Arguments<'_>| {
        s.write_fmt(args).expect("writing to a String");
        s.push('\n');
    };
    w(format_args!(
        "# Golden cases for backtest::simulate, written by crates/backtest/examples/golden.rs."
    ));
    w(format_args!(
        "# Do not edit. Regenerate with `cargo run -p backtest --example golden`."
    ));
    w(format_args!(
        "# Floats are the 16 hex digits of their IEEE-754 binary64 bits; the example's docs give the format."
    ));
    w(format_args!("{FORMAT}"));
    w(format_args!("cases {}", cases.len()));
    for c in cases {
        let cfg = &c.cfg;
        w(format_args!("case {}", c.name));
        w(format_args!(
            "fees {} {} {}",
            c.fee_name,
            hex(cfg.fees.maker),
            hex(cfg.fees.taker)
        ));
        w(format_args!("liquidity {}", liquidity_name(cfg.liquidity)));
        w(format_args!("initial_cash {}", hex(cfg.initial_cash)));
        w(format_args!("slippage_bps {}", hex(cfg.slippage_bps)));
        w(format_args!(
            "min_notional {}",
            hex(cfg.filters.min_notional)
        ));
        w(format_args!("step_size {}", hex(cfg.filters.step_size)));
        w(format_args!(
            "rebalance_threshold {}",
            hex(cfg.rebalance_threshold)
        ));
        w(format_args!("bars {}", c.bars.len()));
        for b in &c.bars {
            w(format_args!(
                "b {} {} {} {} {} {}",
                b.open_time,
                hex(b.open),
                hex(b.high),
                hex(b.low),
                hex(b.close),
                hex(b.volume)
            ));
        }
        w(format_args!("targets {}", c.targets.len()));
        for &t in &c.targets {
            w(format_args!("t {}", hex(t)));
        }
        match simulate(&c.bars, &c.targets, cfg) {
            Err(e) => w(format_args!("error {e}")),
            Ok(r) => {
                w(format_args!("ok"));
                for i in 0..r.equity.len() {
                    w(format_args!(
                        "r {} {} {}",
                        hex(r.equity[i]),
                        hex(r.position[i]),
                        hex(r.cash[i])
                    ));
                }
                w(format_args!("fills {}", r.fills.len()));
                for f in &r.fills {
                    w(format_args!(
                        "f {} {} {} {} {} {} {}",
                        f.bar,
                        f.time,
                        match f.side {
                            Side::Buy => "buy",
                            Side::Sell => "sell",
                        },
                        hex(f.qty),
                        hex(f.price),
                        hex(f.notional),
                        hex(f.fee)
                    ));
                }
                w(format_args!("fees_paid {}", hex(r.fees_paid)));
            }
        }
        w(format_args!("end"));
    }
    s
}

/// The non-comment lines of the file, numbered from 1, split on single spaces.
struct Lines<'a> {
    lines: std::iter::Peekable<Box<dyn Iterator<Item = (usize, &'a str)> + 'a>>,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        let it: Box<dyn Iterator<Item = (usize, &'a str)> + 'a> = Box::new(
            text.lines()
                .enumerate()
                .map(|(i, l)| (i + 1, l))
                .filter(|(_, l)| !l.starts_with('#') && !l.is_empty()),
        );
        Self {
            lines: it.peekable(),
        }
    }

    /// The next line, which must start with `key`, and the rest of it after one space.
    fn rest(&mut self, key: &str) -> Result<(usize, &'a str), String> {
        let (n, line) = self
            .lines
            .next()
            .ok_or_else(|| format!("unexpected end of file, expected {key:?}"))?;
        if line == key {
            return Ok((n, ""));
        }
        line.strip_prefix(key)
            .and_then(|r| r.strip_prefix(' '))
            .map(|r| (n, r))
            .ok_or_else(|| format!("line {n}: expected {key:?}, got {line:?}"))
    }

    /// The next line, `key` and exactly `count` fields.
    fn fields(&mut self, key: &str, count: usize) -> Result<(usize, Vec<&'a str>), String> {
        let (n, rest) = self.rest(key)?;
        let f: Vec<&str> = rest.split(' ').collect();
        if f.len() != count {
            return Err(format!(
                "line {n}: {key} needs {count} fields, got {}",
                f.len()
            ));
        }
        Ok((n, f))
    }

    fn float(&mut self, key: &str) -> Result<f64, String> {
        let (n, f) = self.fields(key, 1)?;
        unhex(f[0]).map_err(|e| format!("line {n}: {e}"))
    }

    fn count(&mut self, key: &str) -> Result<usize, String> {
        let (n, f) = self.fields(key, 1)?;
        f[0].parse().map_err(|e| format!("line {n}: {e}"))
    }
}

fn unhex(s: &str) -> Result<f64, String> {
    if s.len() != 16 {
        return Err(format!("{s:?} is not 16 hex digits"));
    }
    u64::from_str_radix(s, 16)
        .map(f64::from_bits)
        .map_err(|e| format!("{s:?}: {e}"))
}

/// The cases of a file written by [`render`]: inputs only, the expected outputs are skipped.
///
/// # Errors
///
/// A line that does not follow the format, with its number.
pub fn parse(text: &str) -> Result<Vec<Case>, String> {
    let mut l = Lines::new(text);
    let (n, header) = l.rest("format")?;
    if format!("format {header}") != FORMAT {
        return Err(format!("line {n}: unknown format {header:?}"));
    }
    let count = l.count("cases")?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (_, name) = l.rest("case")?;
        let (n, f) = l.fields("fees", 3)?;
        let (maker, taker) = (unhex(f[1])?, unhex(f[2])?);
        let fees = match FeeSchedule::named(f[0]) {
            Some(b)
                if b.maker.to_bits() == maker.to_bits() && b.taker.to_bits() == taker.to_bits() =>
            {
                b
            }
            Some(_) => return Err(format!("line {n}: {} has other rates in the crate", f[0])),
            None => custom_fees(maker, taker),
        };
        let fee_name = f[0].to_owned();
        let (n, f) = l.fields("liquidity", 1)?;
        let liquidity = match f[0] {
            "taker" => Liquidity::Taker,
            "maker" => Liquidity::Maker,
            other => return Err(format!("line {n}: unknown liquidity {other:?}")),
        };
        let mut cfg = SimConfig::new(l.float("initial_cash")?, fees);
        cfg.liquidity = liquidity;
        cfg.slippage_bps = l.float("slippage_bps")?;
        cfg.filters.min_notional = l.float("min_notional")?;
        cfg.filters.step_size = l.float("step_size")?;
        cfg.rebalance_threshold = l.float("rebalance_threshold")?;
        let bars = (0..l.count("bars")?)
            .map(|_| {
                let (n, f) = l.fields("b", 6)?;
                let at = |e: String| format!("line {n}: {e}");
                Ok(Bar {
                    open_time: f[0].parse().map_err(|e| at(format!("{e}")))?,
                    open: unhex(f[1]).map_err(at)?,
                    high: unhex(f[2]).map_err(at)?,
                    low: unhex(f[3]).map_err(at)?,
                    close: unhex(f[4]).map_err(at)?,
                    volume: unhex(f[5]).map_err(at)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let targets = (0..l.count("targets")?)
            .map(|_| l.float("t"))
            .collect::<Result<Vec<_>, String>>()?;
        // Skip the expected outputs: `render` recomputes them.
        loop {
            match l.lines.next() {
                Some((_, "end")) => break,
                Some(_) => {}
                None => return Err(format!("case {name}: no \"end\"")),
            }
        }
        out.push(Case {
            name: name.to_owned(),
            fee_name,
            cfg,
            bars,
            targets,
        });
    }
    if let Some((n, line)) = l.lines.next() {
        return Err(format!("line {n}: trailing {line:?}"));
    }
    Ok(out)
}

/// The committed file, with line endings normalized (a Windows checkout may hold CRLF).
pub fn read_committed() -> std::io::Result<String> {
    std::fs::read_to_string(PATH).map(|s| s.replace("\r\n", "\n"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = render(&cases());
    match args.as_slice() {
        [flag] if flag == "--check" => {
            let committed = read_committed().unwrap_or_else(|e| panic!("reading {PATH}: {e}"));
            if committed == text {
                println!("{PATH} is up to date ({} bytes)", text.len());
            } else {
                eprintln!("{PATH} is stale: run `cargo run -p backtest --example golden`");
                std::process::exit(1);
            }
        }
        [] | [_] => {
            let path = args.first().map_or(PATH, String::as_str);
            if let Some(dir) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("creating {dir:?}: {e}"));
            }
            std::fs::write(path, &text).unwrap_or_else(|e| panic!("writing {path}: {e}"));
            println!(
                "wrote {path}: {} cases, {} bytes",
                cases().len(),
                text.len()
            );
        }
        _ => {
            eprintln!("usage: golden [--check | <path>]");
            std::process::exit(2);
        }
    }
}
