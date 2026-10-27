# Brief — Features, labels and walk-forward splits (iteration 2)

- **Agent:** `data-engineer` (sonnet, high).
- **Runs in parallel with:**
  - a second `data-engineer` (`universe-manifest-tripwire.md`), who downloads the universe and edits `protocol.py`, `conftest.py` and `data/`;
  - two `ml-engineer`s who build the baselines and the statistics on top of your frames;
  - a `rust-engineer` building `crates/olsprime-py`.
- **Motto:** we should be more ambitious: every timestamp traceable, every leak impossible by construction.

## Why this matters

Report n°1 scores every baseline on these frames, and H1 (`docs/HYPOTHESES.md`) runs the future-invariance test through them. A feature that peeks one bar ahead, or a label used before it is known, would invalidate every row of `docs/RESULTS.md`. The ml-engineers code against the contract below *now*, in parallel: keep the names and types exactly.

## The contract

Input everywhere: the bars of one pair as `olsprime.protocol.load_klines` returns them. They are sealed, and carry `known_at = max(close_time, open_time + interval − 1 ms)`, which is when bar t's values are final. Times are `Datetime("ms", "UTC")` (`olsprime.data.klines.TIMESTAMP`). Bar t is "on the grid" when its `open_time` is exactly one interval after bar t − 1's.

### 1. `olsprime/features/basic.py`: `basic_features(bars, interval) -> pl.DataFrame`

One row per bar, with these columns:
- `symbol`, `open_time`;
- `decision_time`: bar t's `known_at`, the close at which a decision uses these features;
- the features, all Float64, computed only from bars ≤ t:

| Column | Definition |
|---|---|
| `f_ret_{k}`, k ∈ {1, 3, 7, 14, 30, 90} | `ln(close_t / close_{t−k})` |
| `f_vol_{k}`, k ∈ {7, 30} | sample std (ddof 1) of the last k one-bar log returns `ln(close_s / close_{s−1})`, s ∈ (t − k, t] |
| `f_ma_gap_{k}`, k ∈ {10, 30, 90} | `ln(close_t / mean(close_{t−k+1..t}))` |
| `f_range_1`, `f_range_7` | `ln(high_t / low_t)`, and its mean over the last 7 bars |
| `f_qvol_z_30` | z-score of `ln(quote_volume_t)` against the previous 30 bars' mean and std (t − 30 … t − 1) |
| `f_taker_7` | mean over the last 7 bars of `taker_buy_quote_volume / quote_volume` |

- **Gaps.** A window is valid only if every bar in it is on the grid: the bar k steps back opens exactly k intervals earlier. Otherwise the value is null. Never fill.
- **Zero volume.** A bar of volume 0 makes `f_qvol_z_30` and `f_taker_7` null where they would divide by 0 or take `ln 0`.
- **Warm-up.** Features are null until their window is full.
- **No scaling.** Standardizing is a model's job, inside the fold.

### 2. `olsprime/labels/forward.py`: `forward_labels(bars, interval, h, *, vol_window=30, k=0.5, floor=0.0024) -> pl.DataFrame`

The decision at bar t's close executes at bar t + 1's open, and a position held for h bars exits at bar t + 1 + h's open. One row per bar t:

| Column | Definition |
|---|---|
| `symbol`, `open_time`, `decision_time` | as above |
| `entry_time` | `open_time[t+1]` |
| `exit_time` | `open_time[t+1+h]` |
| `label_known_at` | `exit_time`: the instant the exit price prints. The splitter purges on it. |
| `fwd_return` | `ln(open[t+1+h] / open[t+1])`, the return a long decision earns before costs |
| `threshold` | `max(k · σ̂_t · √h, floor)`, with σ̂_t = `f_vol_{vol_window}` at t. It is known at t. |
| `label` | UInt8: 0 (down) if `fwd_return < −threshold`, 2 (up) if `> threshold`, else 1 (neutral) |

- **Nulls.** Everything but the first three columns and `threshold` is null when bar t + 1 or bar t + 1 + h is missing, or when either one is not exactly 1 or 1 + h intervals after t. That covers the end of the data and the vault: the seal has removed those bars, so the label cannot exist.
- **Cost floor.** 0.0024 is Binance's taker round trip with 2 bps of slippage per side. `docs/DESIGN.md` §5 sets thresholds by fees and volatility; record this rule in a docstring.
- **Constants.** Export `LABEL_COLUMNS = ("entry_time", "exit_time", "label_known_at", "fwd_return", "label")`. These are the columns a model must never see at prediction time. `threshold` is not one of them: it is known at t.
- Re-export `basic_features` and `forward_labels` from the packages' `__init__.py`.

### 3. `olsprime/eval/splits.py`: walk-forward with purge and embargo

```python
@dataclass(frozen=True)
class Fold:
    fit_time: datetime          # = test_start: the model is fitted at 00:00 UTC on the month's first day
    test_start: datetime
    test_end: datetime          # exclusive, the next month's start
    train_cutoff: datetime      # fit_time − embargo
    train_idx: np.ndarray       # row indices into the frame
    test_idx: np.ndarray

def walk_forward(frame, *, interval, h, start: Month, end: Month,
                 embargo_bars: int | None = None, min_train_rows: int = 365,
                 period: Period = DEVELOPMENT) -> list[Fold]
```

- **Frame.** Features joined to labels on `(symbol, open_time)`. It may hold several symbols: folds cut by time, across all symbols.
- **Test rows.** `decision_time ∈ [test_start, test_end)`, one fold per month from `start` to `end`.
- **Training rows.** Expanding window from the first row. A row trains only if:
  - its label is not null;
  - `label_known_at ≤ train_cutoff`, with `train_cutoff = fit_time − embargo_bars × interval` and `embargo_bars = h` by default (protocol: purge and embargo equal the horizon).

  This is the purge: a label not yet known when the model is fitted never trains it. The embargo widens the gap by h more bars.
- **Small folds.** A fold with fewer than `min_train_rows` training rows is skipped, and the skip is logged.
- **Guards.**
  - Raise `VaultSealed` if any test month or training label reaches `VAULT.start`, whatever `period` says.
  - Raise `ValueError` if a test month falls outside `period`. The default is DEVELOPMENT, 2017-08 → 2024-12.
- `describe(folds) -> pl.DataFrame` gives one row per fold: dates, train and test counts, and the last `label_known_at` used. Reports log it.

## Tests

Offline and synthetic. `python/tests/fakes.py` has helpers; add your own fixtures next to your tests.
- **Features.**
  - Each feature equals a direct numpy computation on a hand-built series.
  - Gaps make windows null, and nothing is filled.
  - **Future invariance:** perturb every bar after t, and every feature row ≤ t stays identical, bit for bit. This is H1 at the feature level.
- **Labels.**
  - `fwd_return` uses `open[t+1]` and `open[t+1+h]`, checked on a series with distinct opens and closes.
  - Shifting by one bar breaks the test.
  - Labels are null at the end, across gaps and when the next bar is missing.
  - `threshold` at t is unchanged when bars after t change.
- **Splits.**
  - No training row has `label_known_at > train_cutoff`.
  - Train and test rows never overlap.
  - An h = 7 test shows the purge drops exactly the expected rows.
  - The vault guard and the period guard raise.
  - Multi-symbol frames work.
- Run the whole `uv run pytest` at the end. If another agent's test fails, report it; do not fix it.

## On real data, at the end

Build features and labels (h = 1) for BTCUSDT and ETHUSDT 1d through `load_klines`, and folds for 2018-01 → 2024-12. Report the rows, the null counts per column, the label class frequencies and the number of folds. Keep any script in your scratch folder.

## Scope and constraints

- **Files.** Only these:
  - `python/olsprime/features/**`, `python/olsprime/labels/**`, `python/olsprime/eval/splits.py`;
  - `python/tests/test_features.py`, `test_labels.py`, `test_splits.py`.
- **Not yours.** `eval/__init__.py` and `eval/stats.py` belong to an ml-engineer. `protocol.py`, `conftest.py` and `data/` belong to the other data-engineer. `pyproject.toml` and `uv.lock` are not yours either: add no dependency.
- **Reads.** Read data only through `olsprime.protocol` (`load_klines`). The static guard in `tests/test_protocol.py` enforces it, and a runtime tripwire is coming.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\features\`.
- **Git.** Run no git command that changes state, and do not commit.

## Report (short)

- Modules and public API.
- Tests, with counts.
- The summary on real data.
- Every deviation from this contract, and why.
- One bolder next step.
