# Brief — Baselines, decision rule and cheater (iteration 2)

- **Agent:** `ml-engineer` (opus, high).
- **Runs in parallel with:**
  - a `data-engineer` writing the features, labels and splits you consume (`features-labels-splits.md`: read its contract section now, and code against it);
  - a second `ml-engineer` writing `eval/stats.py` (`stats.md`);
  - a `rust-engineer` writing the PyO3 bindings (`pyo3.md`).
- **Motto:** we should be more ambitious: baselines a sceptic would accept as fair, so that beating them later means something.

## Why

Report n°1 asks what fees do to simple strategies. H9 (`docs/HYPOTHESES.md`, registered today) says that at Coinbase and Kraken entry fees no active baseline beats buy and hold on BTC or ETH. These baselines are also the bar every later model must clear. Read `docs/DESIGN.md` §5 (decision rule) and `docs/PROTOCOL.md` (baselines, costs) first.

## The interface

In `python/olsprime/models/baselines.py`:

```python
class Baseline(Protocol):
    name: str            # unique, e.g. "ma-momentum-30"; it names the variant
    config: Any          # a frozen dataclass, saved with every run
    def fit(self, train: pl.DataFrame) -> None: ...
    def predict(self, test: pl.DataFrame) -> pl.DataFrame: ...
```

- **`fit`** receives the training rows of one fold: features joined to labels, as in the data contract, with every label column.
- **`predict`** receives the test rows **without** `LABEL_COLUMNS`. It sees `symbol`, `open_time`, `decision_time`, `threshold` and the `f_*` features. It returns, per row:
  - `symbol` and `decision_time`;
  - `p_down`, `p_neutral`, `p_up`, null for a non-probabilistic baseline;
  - `exp_return`, the expected `fwd_return`, null if not modelled;
  - `target`, the position in [0, 1], null when the decision rule sets it.
- **Strip the labels in code.** `predict` raises if a label column is present: the bench will strip them, but the baseline must refuse them anyway.
- **Pure.** No baseline reads data or files. Randomness comes only from a seed in its config.

## The baselines (each config one variant; the list is fixed)

| name | rule |
|---|---|
| `flat` | target 0 |
| `buy-and-hold` | target 1 |
| `ma-momentum-{10,30,90}` | target 1 iff `f_ma_gap_L > 0` (close above its L-bar moving average), else 0; null feature → 0 |
| `random-turnover` | a long/flat Markov chain with seed s. Its switch probability is ma-momentum-30's flip rate **measured on the training fold**. The bench runs 100 seeds and reports their distribution as one variant. |
| `climatology` | class frequencies of the training labels → constant probabilities; `exp_return = Σ_k p_k·m_k`, with m_k the class-mean `fwd_return` in training |
| `ridge` | standardize the features with training means and stds, ridge on `fwd_return` with an unpenalized intercept. λ is chosen inside the fold on the last 20 % of training rows, purged by h, over {1, 10, 100, 10³, 10⁴}, then refitted on all training rows. `exp_return = μ̂`. Gaussian probabilities, with σ̂ the training residual std: `p_up = 1 − Φ((θ − μ̂)/σ̂)`, `p_down = Φ((−θ − μ̂)/σ̂)` |
| `xgboost` | `multi:softprob` on the 3 classes. Fixed hyperparameters: max_depth 3, 300 trees, learning rate 0.05, subsample 0.8, colsample 0.8, min_child_weight 5, seed 0, fixed `nthread`, `tree_method="hist"`. `exp_return = Σ_k p_k·m_k` |
| `cheater` | target 1 iff `close[t+1] > open[t+1]`. It is built with the bars and reads the *next* bar on purpose. It exists so that H1 can catch it: label it loudly in the code. |

- Rows with null features: ridge and xgboost return null probabilities and `exp_return`, and the rule then keeps the model flat.
- Use `numpy` for the ridge closed form now. The bench will cross-check it against `olsprime_rs.ridge_fit` from the Rust core once the bindings land.

## The decision rule

In `python/olsprime/models/decision.py`:

```python
long_flat(exp_return: np.ndarray, cost_round_trip: float) -> np.ndarray
```

A sequential rule with hysteresis:
- flat → long when `exp_return > cost_round_trip`;
- long → flat when `exp_return < 0`;
- otherwise keep the state;
- null → flat.

The cost is `cost_round_trip = 2·(taker + slippage)`, with slippage 0.0002, so the targets of `climatology`, `ridge` and `xgboost` depend on the venue. Export `round_trip_cost(fee_schedule_name)`, with the four schedules of `crates/backtest` (`zero`, `binance-spot`, `coinbase-advanced-entry`, `kraken-pro-entry`) as numbers in one table.

## Tests (offline, synthetic, fast)

- **Interface.** Every baseline refuses label columns in `predict`, and its output has the contract's columns and types.
- **Determinism.** Same seed and data give identical outputs, bit for bit. That includes xgboost: check that it really is deterministic with your `nthread`, and say so.
- **Ridge.**
  - It equals a closed form you write independently.
  - The Gaussian probabilities sum to 1.
  - On data with a planted linear signal, ridge recovers it and its log-loss beats climatology's.
- **Rule.** Hysteresis cases: enter, hold inside the band, exit, null.
- **Cheater.** It wins on any synthetic series with up-bars: its gross return beats buy and hold's.
- **Overfit sanity.** xgboost reaches near-zero log-loss on one tiny training set: it can learn.
- Run the whole `uv run pytest` at the end. If another agent's test fails, report it; do not fix it.

## Scope and constraints

- **Files.** Only `python/olsprime/models/**` (but not `models/__init__.py`'s docstring intent: you may add exports) and `python/tests/test_baselines.py`, `python/tests/test_decision.py`.
- **Not yours.** `features/`, `labels/`, `eval/`, `data/`, `protocol.py`, `pyproject.toml`, `uv.lock`. `scipy` and `xgboost-cpu` are installed already.
- **Data.** If you need the frames before the data-engineer finishes, build synthetic ones with the contract's columns. Never read Parquet.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\baselines\`.
- **Git.** Run no git command that changes state, and do not commit.

## Report (short)

- Modules and API.
- Tests, with counts.
- Sanity checks.
- Every deviation from this brief or from `docs/DESIGN.md`, and why.
- One bolder next step.
