# Brief — Bench fixes, round 4: colour × size, and exact floors (iteration 2)

- **Agent:** the `ml-engineer` of rounds 1 to 3, resumed.
- **When:** 2026-10-28, after Audit A's third re-check.
  - **Verdict:** pass with fixes, no Blocker.
  - **What held.** The `?` mechanism is sound now:
    - `unknown_reason` reads the original bars only;
    - two more sabotages (a weak trades push, and a ÷3 bar rebuilt without the fill and close-location needs) both fail H1 loudly;
    - every band and floor you stated holds by construction and empirically, on 16 cuts × 3 seeds, except the fill floor;
    - the auditor's scripts reproduce your counts.
  - **The auditor's files** are in the orchestrator's scratch folder: `scratchpad\audit-a\round3.py`, with `round3.log`, `leaks2-r3.log` and `leaks3-r3.log`.
- **This round is final.** The official runs follow it. Anything the re-check finds after this goes to LIMIT, or to iteration 3, unless it is a Blocker.

## Major — colour × size conjunctions are one-sided

**Where.** `perturb.py:226-252`. The two runs realise exactly two of the four colour × size cells of bar t+1: (green, small) in the ×3 run and (red, big) in the ÷3 run. A leak that selects one of the other two cells is false in both runs, so it flips only where it was true on the original bars.

**Evidence, BTCUSDT, the bench's 8 cuts.**

| leak | flagged |
|---|---|
| red and body < 1 % | **0/8** |
| green and body > 1 % | 4/8 |
| green and body > body[t] | 4/8 |

A candle-pattern feature read one bar early, such as "yesterday was a small red bar", is caught with a probability of about 0.9 per model, not 1. LIMIT does not name this class.

**Fix, by construction.** Alternate the colour by cut, so that each of the four cells is realised at half the cuts.
- **Even cuts:** as now.
  - ×3: small, green, body-dominant, close location ≥ 0.75.
  - ÷3: big, red, upper-wick-dominant, close location ≤ 0.25.
- **Odd cuts:** the mirror image.
  - ×3: small, red, body-dominant, close location ≤ 0.25.
  - ÷3: big, green, lower-wick-dominant (lower wick ≥ 3 × upper), close location ≥ 0.75.
  - Use the mirror in close location, in price units, as you do now. In log space it is not exactly 1 − CL.
- **What must still hold at every cut.**
  - The cheater flips in exactly one run, because one run is green and the other red.
  - Fill and close location cross bar t's in exactly one run. At odd cuts the close location's direction is reversed.
  - Every one-quantity push is unchanged: levels, VWAPs, volumes, trades, sizes, shares, body, wicks, range.
  - Only colour and close location change direction with the cut's parity.
  - The parity is the cut's index in the bench's cut list, so it is fixed and deterministic.
- **New permanent controls:** `red-and-small` (red and body < 1 %) and `green-and-big` (green and body > 1 %).
  - Compute each predicted pattern exactly, per cut, from the cell each run realises and the probe's value on the original bars. A cut is predicted X when at least one run differs from the original.
  - Do the same for any probe whose prediction now depends on parity.
- **`h1.md`.** Add a small per-cut table: the cell, (colour, size), that each run realised.
- **LIMIT** names what stays probabilistic: conjunctions of colour or size with other shape features, such as wick ratio × colour.

## Minor 1 — the fill floor is a target

**Where.** `perturb.py:241, 246-247`. `fill` is the target, but the upper wick becomes `max(window, 3·lw, fill need, uw_clv) × U(1, 1.5)`. The realised ÷3 fill therefore goes down to about 0.0134: 9 of the auditor's 48 ÷3 bars had a fill in [0.0144, 0.0182]. `h1.md` line 11 and `perturb.py:32, 112-113` say "never below 0.02", which is false.

**Fix.** Apply the random factor to the non-fill needs only: `uw = max(other_needs × U, fill_need)`. The floor is then exact. Keep the declared `?` for a reference below the floor.

## Minor 2 — no floor on the close location

**Where.** `clv = min(0.25, _bounded_down(wc, c0, 1.5))` has no floor. When bar t, or the original bar t+1, closed within 1e-5 of its low, `uw_clv` reaches 7 to 28 log units. At ETH cut 1994 the ÷3 bar's range was 28.8 log units, with prices down to 1e-12 of the level: valid, but a bar no venue could print. It also causes the fill-floor breach above.

**Fix.**
- Add `CLV_FLOOR = 0.02`, with a declared `?` for a reference close location below it. At odd cuts, mirror it: a reference above 0.98.
- This caps `uw_clv` at about `ln(lw / 0.02)`, two to three log units.
- State the realised maximum range of the ÷3 bar in LIMIT.

## Nits

1. ×3 fill = 0.8 − 1 ulp at 9 of 48 bars. Aim slightly above 0.8, or say "strictly below 0.8" for the band's upper end.
2. `_bounded_up` and `_bounded_down` skip window values at 0 or 1. So for shares, LIMIT should say "against the window's interior values".

## Constraints

- **Files.** Only `python/olsprime/bench/**` and its three test files.
- **Python.** `uv run --no-sync`.
- **Off limits.**
  - The guard files: round 3 is done, and they must not change now.
  - `crates/`.
  - `docs/`.
- **The check run.** `--out runs/2026-10-28-h1-fix4`, a new folder.
- **The auditor's scripts.** Run `leaks2.py`, `leaks3.py` and `round3.py` unmodified, from the orchestrator's `scratchpad\audit-a\`, and give their flags.
  - `round3.py` measures the conjunction leaks and the floors.
  - If a script reaches into a helper you renamed, report it; do not edit it.
- **Time.** Report by 04:40 UTC.

## Done when

- **Tests.** The three bench test files pass, and so does the whole suite except `test_rust_bindings.py`.
  - Tests cover the parity: the cells, the close-location direction, the cheater.
  - Tests cover the exact fill floor, the close-location floor and its `?`.
- **The run.** `runs/2026-10-28-h1-fix4` holds:
  - every control is on its predicted pattern, the two new ones included;
  - every `?` is listed with its reason.
- **Conjunctions.** `red and body < 1 %` and `green and body > 1 %` are flagged at every cut where a run realises the opposite cell from the original, measured by `round3.py` or by your new probes.

## Report

- The files changed.
- The parity design.
- The H1 table, with ×3/÷3 counts.
- The `?` cuts and their reasons.
- The realised ranges.
- The flags of the three scripts.
- The test count and the run time.
- One bolder next step.
