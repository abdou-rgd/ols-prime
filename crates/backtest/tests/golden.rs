//! The committed golden file of `backtest::simulate` (`python/tests/golden/simulate.txt`).
//!
//! The Python binding is checked against that file bit for bit, so the file must be exactly
//! what the simulator computes today. Any change to the simulator fails here until the file is
//! regenerated on purpose (`cargo run -p backtest --example golden`) and its diff reviewed.

#[path = "../examples/golden.rs"]
#[allow(dead_code)]
mod golden;

use backtest::{SimError, simulate};

fn committed() -> String {
    golden::read_committed().unwrap_or_else(|e| {
        panic!(
            "reading {}: {e}; run `cargo run -p backtest --example golden`",
            golden::PATH
        )
    })
}

/// The first differing line, not a 100 kB dump.
fn assert_same_text(got: &str, want: &str, what: &str) {
    if got == want {
        return;
    }
    let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    let i = g.iter().zip(&w).take_while(|(a, b)| a == b).count();
    panic!(
        "{what}: first difference at line {} (generated {:?}, committed {:?}). If the simulator \
         changed on purpose, run `cargo run -p backtest --example golden` and review the diff.",
        i + 1,
        g.get(i),
        w.get(i)
    );
}

#[test]
fn the_committed_file_is_what_the_generator_writes() {
    assert_same_text(
        &golden::render(&golden::cases()),
        &committed(),
        "the generator and the committed file differ",
    );
}

/// The file is self-contained: its inputs alone reproduce its outputs. This is exactly what
/// the Python test does through the binding, so a disagreement there is the binding's.
#[test]
fn the_committed_inputs_reproduce_the_committed_outputs() {
    let text = committed();
    let cases = golden::parse(&text).expect("the committed file parses");
    // `render` prints every input as bits, so this also proves that `parse` recovers them
    // exactly; `==` on the cases could not, since some inputs are NaN.
    assert_same_text(
        &golden::render(&cases),
        &text,
        "re-rendering the parsed file",
    );
}

#[test]
fn the_file_stays_small() {
    let n = committed().len();
    assert!(n < 200_000, "{n} bytes");
}

#[test]
fn the_parser_refuses_a_damaged_file() {
    let text = committed();
    for (from, to) in [
        (
            "format olsprime-golden-simulate 1",
            "format olsprime-golden-simulate 2",
        ),
        ("liquidity taker", "liquidity Taker"),
        (
            "fees binance-spot 3f50624dd2f1a9fc",
            "fees binance-spot 3f50624dd2f1a9fd",
        ),
        ("\nend\n", "\n"),
    ] {
        assert!(text.contains(from), "{from:?}");
        let damaged = text.replacen(from, to, 1);
        assert!(golden::parse(&damaged).is_err(), "{from:?} -> {to:?}");
    }
    assert!(golden::parse(&format!("{text}case extra\n")).is_err());
}

/// The cases keep covering what they were written for: every refusal of the simulator, and the
/// edges of the execution model. Tuning the generator must not lose them silently.
#[test]
fn the_cases_cover_every_refusal_and_the_edges() {
    let cases = golden::cases();
    let run = |name: &str| {
        let c = cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case {name}"));
        (c, simulate(&c.bars, &c.targets, &c.cfg))
    };
    let mut kinds = [false; 6];
    for c in &cases {
        if let Err(e) = simulate(&c.bars, &c.targets, &c.cfg) {
            assert!(c.name.starts_with("err-"), "{} fails: {e}", c.name);
            kinds[match e {
                SimError::Empty => 0,
                SimError::LengthMismatch { .. } => 1,
                SimError::InvalidBar { .. } => 2,
                SimError::NonIncreasingTime { .. } => 3,
                SimError::InvalidTarget { .. } => 4,
                SimError::InvalidConfig { .. } => 5,
            }] = true;
        } else {
            assert!(!c.name.starts_with("err-"), "{} succeeds", c.name);
        }
    }
    assert_eq!(kinds, [true; 6], "every SimError variant");
    assert!(matches!(
        run("err-config-before-bars").1,
        Err(SimError::InvalidConfig {
            field: "initial_cash",
            ..
        })
    ));
    assert!(matches!(
        run("err-bars-before-targets").1,
        Err(SimError::NonIncreasingTime { index: 2 })
    ));

    // One buy, then the cash residue is dust at every bar.
    for name in ["all-in-dust", "all-in-dust-lots"] {
        let r = run(name).1.unwrap();
        assert_eq!(r.fills.len(), 1, "{name}");
        assert!(r.cash.last().unwrap() < &1.0, "{name}");
    }
    // The clamp at 2⁵⁰ lots binds.
    let (c, r) = run("max-lots-clamp");
    let r = r.unwrap();
    let cap = (1u64 << 50) as f64 * c.cfg.filters.step_size;
    assert_eq!(r.position[1].to_bits(), cap.to_bits());
    assert_eq!(r.fills.len(), 2);
    // A full exit below the minimum notional leaves the position.
    let r = run("exit-below-min-notional").1.unwrap();
    assert_eq!(r.fills.len(), 1);
    assert!(r.position.last().unwrap() > &0.0);
    assert!(run("coarse-lot-never-fills").1.unwrap().fills.is_empty());
    assert!(run("one-bar").1.unwrap().fills.is_empty());
    // −0.0 and the smallest subnormal are in the inputs, as bits.
    let (c, _) = run("signed-zero-and-subnormal-targets");
    assert!(c.targets.iter().any(|t| t.to_bits() == (-0.0f64).to_bits()));
    assert!(c.targets.iter().any(|t| t.to_bits() == 1));
    // Most random cases trade, with lots and without.
    let random: Vec<_> = cases
        .iter()
        .filter(|c| c.name.starts_with("random-"))
        .collect();
    let trading = random
        .iter()
        .filter(|c| {
            !simulate(&c.bars, &c.targets, &c.cfg)
                .unwrap()
                .fills
                .is_empty()
        })
        .count();
    assert!(trading >= random.len() - 2, "{trading} of {}", random.len());
    assert!(random.iter().any(|c| c.cfg.filters.step_size > 0.0));
    assert!(random.iter().any(|c| c.cfg.filters.step_size == 0.0));
}

#[test]
fn splitmix64_is_the_reference_sequence() {
    // The first outputs for seed 0 from Vigna's reference implementation (splitmix64.c).
    let mut r = golden::SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
    assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    assert_eq!(r.next_u64(), 0x06C4_5D18_8009_454F);
}
