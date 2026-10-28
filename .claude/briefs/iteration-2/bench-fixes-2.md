# Brief — Bench fixes, round 2: ratios (iteration 2)

- **Agent:** the `ml-engineer` who wrote `bench-fixes.md`'s fixes, resumed.
- **When:** 2026-10-28, after Audit A's focused re-check of the bench: pass with fixes, with one Major and three Minor.
- **Motto:** we should be more ambitious. The perturbation now moves every value. Make it move **every function of one or two values** of a later bar, and make every one-bar comparison against the past flip, **by construction**.

## M2 (Major): H1 is blind to the ratios inside a bar

**Where.**
- `python/olsprime/bench/perturb.py:123` sets `steps[0] = steps[1] = 0`.
- `:130`: one factor for the open and the close of bar t+1.
- `:137-138`: high and low get the max and the min of two equal factors.
- `:142` and `:160`: one `v_factor` for `volume` and `quote_volume`, and the same for the taker pair.

**Which ratios never move.** They are never perturbed, in either direction, at any cut.
- On bar t+1:
  - the range `high/low`;
  - the body `|ln(close/open)|`.
- On every later bar:
  - the VWAP `quote_volume/volume`;
  - the taker VWAP `taker_buy_quote/taker_buy_base`;
  - the trade size `volume/trades`, up to rounding.

`check_coverage` checks one column at a time, so it misses them.

**Evidence.** The auditor ran the bench's own cuts on real BTCUSDT 1d, in both directions:

| planted leak | flagged ×3 | flagged ÷3 |
|---|---|---|
| `ln(high/low)[t+1] > [t]`, an `f_range_1` read one bar early | 0/8 | 0/8 |
| `ln(high/low)[t+1] > 3 %` | 0/8 | 0/8 |
| `abs(ln(close/open))[t+1] > 2 %` | 0/8 | 0/8 |
| `VWAP[t+1] > VWAP[t]` | 0/8 | 0/8 |
| taker-VWAP peeker | 0/8 | 0/8 |
| `volume/trades`, `[t+1] > [t]` | 0/8 | 0/8 |

- The continuous versions are flagged only by floating-point noise, and only at some cuts.
- The auditor's scripts are `leaks2.py` and `luck.py`, with `leaks2.log`, in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\audit-a\`. Read them first.

## The fix

### 1. Every later bar after t+1: independent factors per column

- **Prices.**
  - Open and close already come from the walk.
  - High and low get their own wick factors: draw a new upper wick and a new lower wick around the new body, so the range changes.
- **Volumes and trades.** `volume`, `quote_volume`, `trades` and each taker column get independent factors.
- **Plausibility.** Keep the bars plausible. VWAP and taker VWAP inside the bar's new `[low, high]` is preferred, for example by placing them uniformly in it.
- **Hard constraints.** Taker ≤ total is required. The bar validity of `_check_valid` stays. Exact zeros of a bar without volume stay.

### 2. Bar t+1: designed per direction, against the recent past

In the up run (×3), push each quantity f of bar t+1 above the past; in the down run (÷3), below it. A comparison of f between bar t+1 and the past then flips in exactly one direction, whatever its original sign. Do this for:
- every price;
- the body `|ln(close/open)|`, the range `ln(high/low)`, and the upper and lower wicks;
- the VWAP and the taker VWAP;
- `volume`, `quote_volume` and `trades`;
- the taker share, of base and of quote;
- the trade size `volume/trades`.

**"The past" means a window, not only bar t.** Use bar t and the K bars before it, with K = 90, the longest feature window. Up: at least 1.5 × the window's maximum, and at least ×3 the original. Down: at most the window's minimum / 1.5, and at most ÷3 the original. This also closes the auditor's case of a comparison with bar t−k.

**Thresholds.** A threshold comparison `f[t+1] > c` flips for every c between the down value and the up value.
- Give the up values floors that beat the usual thresholds: for example, a range of at least 20 %, a body of at least 10 %.
- State the band in the docstring.

**When two requirements conflict.** Examples:
- `trades` must stay an integer of at least 1 when there is volume;
- taker ≤ total;
- a taker VWAP inside the range, while both taker shares are pushed.

Keep the bars valid and the crossing of levels first. Mark the other control's cut `?`, and say why in the docstring. Never bend a validity constraint to make a control flip.

**The doji (Minor).** With `open == close` the swap flips nothing. Give the body a nonzero sign in the up run, so `close > open` there, and keep the cheater on its pattern.

**The taker-share boundaries (Minor).** When no cross is possible, for example a share of 1 at t and at t+1, or a bar with zero volume:
- mark that control's expected flag at that cut as unknown, `?`, in the pattern;
- leave it out of the verdict, and report it.
Never count it as a pass.

### 3. Coverage of ratios, asserted at run time

Extend `check_coverage`. For every later bar and every pair (a, b) of perturbed columns that are nonzero in the original and in the perturbed bars, the ratio a/b must change. The exact zeros stay the only exception. Pin it in `tests/test_invariance.py`.

### 4. New permanent controls in `leaks.py`

Each control has an exact predicted pattern of `flag-every-cut`, except at `?` cuts.
- **Binary peekers:**
  - the range;
  - the body magnitude;
  - the VWAP;
  - the taker VWAP;
  - the trade size;
  - the range above 3 %;
  - the body above 2 %.
- **Continuous:** `exp_return = ln(high/low)[t+1]` and `exp_return = VWAP[t+1] / close[t] − 1`.
- **Comparison with an older bar:** `close[t+1] > close[t−5]`.

### 5. The LIMIT paragraph of h1.md

Say plainly:
- Two-bar-ahead comparisons, such as `close[t+2] > close[t+1]`, are caught with high probability, not by construction. The auditor measured 7/8.
- Thresholds outside the band of §2 are not guaranteed to flip.
- The `?` cuts are reported.
- The time-stamp limit stays.

## Minor, already handled

- The recorded run `runs/2026-10-28-h1-fix` has no provenance. The official run on the committed tree will have it, so nothing to do.
- **Cap dates.** The auditor accepts your definition. Its own dates came from a proxy.

## Constraints

- **Files.** Only `python/olsprime/bench/**` and its three test files.
- **Python.** Use `uv run --no-sync`. The extension must not rebuild: `tests/test_rust_bindings.py` fails until the orchestrator rebuilds it, which is expected.
- **The check run.** Run H1 once on the real data, to check every control's pattern, with `--out` pointing to a new folder: `runs/2026-10-28-h1-fix2`.
  - Never overwrite an existing run folder.
  - Do not run Report n°1: the perturbation is not part of it.
- **Never** touch `docs/`, the vault or `OLSPRIME_OPEN_VAULT`.
- **Time.** Report by 01:40 UTC. If something is not finished by then, report what is in.

## Done when

- The three bench test files pass.
- In `runs/2026-10-28-h1-fix2/h1.md`:
  - every honest model has 0 violations;
  - every control is exactly on its predicted pattern, with `?` cuts listed;
  - the positive control passes.
- The auditor's six leaks from the table above, run against the new perturbation, are flagged.

## Report

- The files changed.
- The new H1 table.
- The `?` cuts, if any.
- The bands of §2.
- The run time.
- The test count.
- One bolder next step.
