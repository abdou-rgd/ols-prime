# Brief — Bench fixes, round 3: the candle's shape, and a sound `?` (iteration 2)

- **Agent:** the `ml-engineer` of rounds 1 and 2, resumed.
- **When:** 2026-10-28, after Audit A's re-check of round 2.
  - **Verdict:** pass with fixes. M2 is fixed: the six ratio leaks go from 0/8 to 8/8, with ×3 + ÷3 = 8 at every cut.
  - **Unchanged:** honest models stay at 0 violations, xgboost included, and the pushed bars break no pipeline.
  - **Left:** two Major findings and one Minor.
- **This is the last design round of the iteration.** With two runs per cut, not every function of a bar can flip by construction. Control what can be controlled, randomize the rest, and **state the rest plainly** in LIMIT. The official runs follow this round.

## Major 1 — The candle's shape is not controlled

**Where.** `python/olsprime/bench/perturb.py:171-188`. Body and wicks are pushed by the same rule in both runs. When the floors or caps bind, body : upper wick : lower wick is 2 : 1 : 1 in both directions. So functions of three or four prices of bar t+1 do not flip by construction.

**Evidence.** The auditor's script is `scratchpad\audit-a\leaks3.py`, with its log `leaks3.log`, on the bench's 8 cuts of BTCUSDT:

| leak on bar t+1 | flagged (either) |
|---|---|
| close location `(close − low)/(high − low)` > bar t's | 3/8 |
| fill `body/range` > bar t's | 2/8 |
| hammer: lower wick > 2 × body | 3/8 |
| upper wick / lower wick > bar t's | 5/8 |
| close location > 0.5 | 6/8 |
| `abs(ret)[t+1] > abs(ret)[t]`, close to close | 4/8 |

**Fix.**
1. **Give the two runs opposite shapes, by construction.**
   - Up run: body-dominant. The fill `body/range` is at least 0.8, so each wick is small next to a body of at least 4 × (upper + lower), and the close location is high, since the bar is green.
   - Down run: wick-dominant, with an upper wick at least 3 × the lower. The fill is at most 0.2 and the close location at most 0.25.
   - Keep everything round 2 guarantees: levels, VWAP, volumes, shares, colour, and the bands, adjusted if needed. State the new bands.
   - When the window makes a guarantee impossible, the cut becomes a **declared** `?` (Major 2), never a silent pass.
2. **Randomize per run** whatever shape freedom remains, so that other shape functions flip with high probability.
3. **New permanent controls in `leaks.py`:**
   - the close-location peeker, against bar t;
   - the fill peeker, against bar t;
   - close location > 0.5.
   Each has an exact predicted pattern.
4. **Close-to-close magnitude.** `abs(ln(close[t+1]/close[t]))` against `abs(ln(close[t]/close[t−1]))`: make it flip by construction if the level pushes allow it. The ×3 run's push is far beyond, while the ÷3 run's push is also large in magnitude. Otherwise, state it as probabilistic.

## Major 2 — `?` is read from the outcome, so a broken push would pass as `?`

**Where.** `invariance.py:312-316` sets `crossable` from the perturbed bars. `verdict:344` turns any `crossable is False` into `None`, `ok` counts known cuts only (`:346-358`), and `h1_holds:435` ignores `unknown_cuts`.

**Scenario.** A regression that stops pushing the VWAP gives `vwap-peeker` the pattern `????????` on both pairs, and H1 "holds".

**Fix.**
- **Predict `?` before perturbing.** Compute it from a predicate on the **original** bars, per probe, naming only the documented reasons:
  - a share at its bound in the window;
  - zero volume or zero trades at t+1 or at the reference bar;
  - a count of 1 that must go down;
  - any reason the new shape guarantees add.
- **A failure to cross without such a reason is a failure.** The cut counts as off pattern and `ok = False`.
- **A control that is `?` at every cut of a pair fails `h1_holds`.**
- **Guard `crossable:119`.** For `cut < back`, `values(original)[cut - self.back]` wraps to the end of the array and reads a future bar. Refuse such a cut, or treat it as a declared `?`.
- **Tests:**
  - a sabotaged push (monkeypatch the VWAP push off) fails H1 instead of reading `?`;
  - a declared reason gives `?`;
  - `cut < back` is handled.

## Minor — the LIMIT paragraph (`invariance.py:553`)

Add:
1. Comparisons with bars older than t−90 flip with high probability only. The auditor measured 8/8 for t−120 and t−364, but because 3 × original happened to exceed the old high, not by construction.
2. For a share, the up value is `1 − (1 − max_W s)/1.5`, so `share > c` is guaranteed only for `c` below it. State the guaranteed band per share as well.
3. Shape functions:
   - the two controlled by construction, fill and close location;
   - the others, which are probabilistic: wick ratio, hammer-like patterns, and functions of several bars.
4. Drop "under the previous perturbation" from the 7/8 for `close[t+2] > close[t+1]`: it holds under the new one too (`XXX.XXXX`).

## Constraints

- **Files.** Only `python/olsprime/bench/**` and its three test files.
- **Python.** Use `uv run --no-sync`.
- **Off limits.**
  - The guard files under `python/tests`: a data-engineer is editing them.
  - `crates/`.
  - `docs/`.
- **The check run.** Run H1 once with `--out runs/2026-10-28-h1-fix3`, a new folder. Never overwrite one.
- **The auditor's scripts.** Run `leaks2.py` and `leaks3.py` unmodified against the new perturbation, and give their flags.
- **Time.** Report by 04:00 UTC.

## Done when

- The three bench test files pass.
- `runs/2026-10-28-h1-fix3`:
  - holds;
  - has every control on its pattern;
  - lists every `?` with its declared reason.
- `leaks3.py`:
  - the close-location and fill leaks are flagged at 8/8;
  - every other leak's count is reported as it is.
- The sabotage test fails H1.

## Report

- The files changed.
- The shape design, with its new bands.
- The H1 table.
- The `?` cuts and their reasons.
- The flags of `leaks2.py` and `leaks3.py`.
- The test count and the run time.
- One bolder next step.
