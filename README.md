# OLS Prime

**A from-scratch decision model whose attention is ordinary least squares, tested on crypto markets without fooling ourselves.**

OLS Prime is a local-AI lab. Crypto trading is the playground, not the goal. Every component is a hypothesis: it is registered before it is tested, then measured by ablation against simple baselines on a leak-free backtest bench. A clean negative result counts as a result.

> Status: iteration 4, the encoder. The vault months are out of `data/`, stage 1 of a physical seal whose encryption waits for an offline key. On the hourly bars of 233 pairs, the encoder learns at about Matrix's level ([Report n°3](docs/reports/report-3.md)): a Brier skill score of 0.0110 for `enc-ols`, against 0.0130 for `matrix-select`. H4 holds as non-inferiority. H5 does not hold. H13 has no verdict: the measurement is invalid. H6 was not run. These are forecasts, before costs. Nothing here trades yet, and nothing here is investment advice.

## The idea

- **OLS is attention.** Goulet Coulombe (2025) shows that an OLS forecast is an attention mechanism. The forecast `ŷ = x_qᵀ (XᵀX)⁻¹ Xᵀ y` equals `Σᵢ wᵢ yᵢ`, and the weights `wᵢ = xᵢᵀ (XᵀX)⁻¹ x_q` are *signed*. OLS Prime's encoder heads solve a small ridge regression instead of applying a softmax. The trained arms are `enc-ols`, with that head, and `enc-softmax` and `enc-linear`, which share every weight but the attention operator.
- **Matrix remembers.** Past states and their realized outcomes go into an append-only log. Older entries are rolled up into a binary tree, as in Victor Taelin's OptMem, except that each node holds the sufficient statistics of a regression (`n, Σz, Σzzᵀ, Σyzᵀ`).
  - Those statistics add up, so any period is the sum of O(log n) nodes.
  - Every forecast splits *exactly* into signed contributions of past days and periods.
  - The name is a nod to Optimus Prime's Matrix of Leadership, which carries the wisdom of the Primes before him.
- **One pass, typed and calibrated.** Like Jev-style decision models, the output is a probability over a fixed set of options: down, neutral or up over a horizon. The encoder is trained with cross-entropy on the softmax output, before the probability floor. The Brier term is the pre-registered H6 ablation, and H6 was not run. Expected calibration error is reported with the scores. A fixed-point Rust risk kernel then turns those probabilities into long/flat positions.

## Why from scratch

A pretrained LLM has already read the news of the years we would backtest on, so its backtest is contaminated by look-ahead bias. Masking names or dates does not remove that bias ([Lopez-Lira, Tang and Zhu, 2025](https://arxiv.org/abs/2504.14765)). A small model trained only on point-in-time data avoids the problem by construction, and it fits on a single RTX 3060.

## Layout

| Path | What |
|---|---|
| `crates/ols-core` | closed-form ridge, dual (attention) weights, recursive least squares with forgetting; Matrix, with centred nodes, batch ridge readouts and exact softmax and kNN readouts |
| `crates/backtest` | bar simulator (decide at the close, execute at the next open), venue fee schedules, metrics |
| `crates/risk` | fixed-point risk kernel, with Kani proof harnesses and function contracts |
| `crates/olsprime-py` | Python bindings of the Rust core (PyO3), built by `uv sync` |
| `python/olsprime/data`, `universe.py`, `vault.py` | data pipeline (data.binance.vision), point-in-time universe, the data manifest, and the physical seal that keeps the vault months outside `data/` |
| `python/olsprime/features`, `labels` | point-in-time features and the hourly panel of the universe members; three-class forward labels |
| `python/olsprime/eval` | walk-forward splits with purge and embargo; statistics (Sharpe intervals, PSR, DSR, stationary bootstrap, Diebold-Mariano, calibration) |
| `python/olsprime/models` | baselines and the long/flat decision rule; the eight Matrix arms of Report n°2; the encoder of Report n°3 (`enc-ols`, `enc-softmax`, `enc-linear`) and the Matrix memory over its learned keys |
| `python/olsprime/train` | weekly episodes, yearly fits, checkpoints and the deterministic settings of the encoder arms |
| `python/olsprime/explain` | `why`, `series` and `sample` for a Report n°2 forecast, and `head` for the OLS heads of a Report n°3 run; every call is logged |
| `python/olsprime/bench` | the future-invariance test (H1) on daily bars and on the panel, its adversary, and the reports, one command each; Report n°3 and the size and power of the pass rule |
| `docs/` | [design](docs/DESIGN.md), [evaluation protocol](docs/PROTOCOL.md), [pre-registered hypotheses](docs/HYPOTHESES.md), [results](docs/RESULTS.md), [reports](docs/reports/) |

## Getting started

```bash
cargo test --workspace
cd python
uv sync
uv run pytest
```

`uv sync` also builds the Rust core into the Python environment. The bench reads the klines of data.binance.vision, which are never committed:

```bash
uv run python -m olsprime.data.binance_vision --symbols BTCUSDT ETHUSDT --intervals 1d --start 2017-08 --end 2025-12
uv run python -m olsprime.bench.invariance
uv run python -m olsprime.bench.report1
```

The first command downloads and checks the archives. On the development machine, the second runs H1, the future-invariance test, in seven to eight minutes with its default four processes. The third writes [Report n°1](docs/reports/report-1.md) in about 75 seconds. `report1 --check <run folder>` re-runs a report and compares it byte for byte.

Report n°2 reads the hourly bars of every universe member: [`docs/DATA.md`](docs/DATA.md) gives the commands that rank the universe and download them, and the vault move that takes the sealed months out of `data/`. Then:

```bash
uv run python -m olsprime.bench.panel_invariance
uv run python -m olsprime.bench.adversary
uv run python -m olsprime.bench.report2
```

The first runs H1 on a 10-member sub-panel, with five planted leaks, in about 35 minutes. The second sends 10,000 random leaks through H1's perturbation in 3 to 5 minutes. The third writes [Report n°2](docs/reports/report-2.md) in about 20 minutes; `report2 --check <run folder>` takes as long again.

Report n°3 trains the three encoder arms. It reads a Report n°2 `series.npz` and the SHA-256 that run records, and the fits and the timing need CUDA:

```bash
uv run python -m olsprime.bench.report3 --series <series.npz> --sha256 <digest>
```

The official fits took 12 h 51 min of wall time, from 2026-10-30 09:00:54 UTC to 21:51:41 UTC. The 12.69 training GPU-hours are upper bounds, because the fits shared the GPU. `report3 --check <run folder>` re-scores a run and needs the same `--series` and `--sha256`. The generated report prints H13 as "holds". The verdict on record, in [docs/RESULTS.md](docs/RESULTS.md), is no verdict: the measurement is invalid.

The size and power of the pass rule read that series and refit nothing. The run on the Windows series took 5,198 s:

```bash
uv run python -m olsprime.bench.power --series <series.npz> --sha256 <digest>
```

`uv run python -m olsprime.explain` says why a Matrix forecast is what it is. Its `head` command is built, and it has not been called.

The Kani proofs need Linux (WSL2 works). Five of the twenty-one harnesses finish:
- the two of iteration 1: every buy gate blocks every buy, and sells are never gated;
- the function contracts of `product` and of `max_qty_within`;
- a lemma that the specifications' 128-bit products never wrap.

On the development machine, run four at a time (`-j 4 --output-format=terse`), the five take about two hours. The contract of `max_qty_within` takes most of that: 122 minutes on 2026-10-29, under load from other jobs, and 72.5 minutes alone on a GitHub runner. The other four take 14 to 18 minutes together. On GitHub, the `proven` job runs the five in 74 to 91 minutes. On 2026-10-31 the five passed again in WSL2 with `-j 2`; `max_qty_within_contract` took 5,313 s, and the risk crate was unchanged since `9c414ca`.

```bash
cargo kani -p risk --exact -Z function-contracts -Z stubbing --no-assert-contracts --harness proofs::buy_gates_block_every_buy --harness proofs::sells_are_never_gated --harness proofs::product_contract --harness proofs::exact_never_wraps --harness proofs::max_qty_within_contract
```

`cargo kani -p risk`, with the same flags, runs all of them.
- The contracts of the three other helpers that divide (`notional_floor`, `notional_ceil`, `round_down_to_lot`) are written in `crates/risk/src/spec.rs`. They are not proven yet. Each gave no verdict in 340 minutes on a GitHub runner, and Z3 and Bitwuzla gave none in 45 minutes either. Property tests are the evidence for them for now.
- So did the `check_order` harnesses that use those contracts as stubs.
- Never pass `--cbmc-args --bitwuzla` as is: CBMC 6.11's Bitwuzla back end gives false proofs on Rust's overflow checks ([diffblue/cbmc#9144](https://github.com/diffblue/cbmc/issues/9144), fixed on CBMC's `develop` but not yet released; [model-checking/kani#5096](https://github.com/model-checking/kani/issues/5096)).

[`kani.yml`](.github/workflows/kani.yml) keeps both lists.

## Roadmap

0. **Foundations:** repository, data pipeline, CI. Done in iteration 1.
1. **Honest bench**, done in iteration 2:
   - Rust backtest with real venue fees;
   - risk kernel;
   - walk-forward protocol and baselines.

   Report: what fees do to simple strategies.
2. **Matrix on hand-made features**, the paper's OLS attention extended to time series, done in iteration 3:
   - Matrix on the hourly bars of 233 pairs, 2019 to 2024, with the vault months moved out of `data/`;
   - an adversary that measures how many random leaks H1 catches: 98 % of the 10,000 it planted.

   [Report n°2](docs/reports/report-2.md): a Brier skill score of 0.013 over climatology, so H2 holds. Matrix beats softmax attention on the same memory, but not measurably kNN, so H3 does not hold. Refitted on a CUSUM's alarms, Matrix needs 13 fits instead of 72, and is no worse within the registered margin, so H12 holds. H10 and H11 do not hold. These are forecasts, before costs.
3. **OLS Prime encoder, from scratch**, done in iteration 4:
   - `enc-ols`, `enc-softmax` and `enc-linear` on the hourly panel, each reading a Matrix of learned keys;
   - 21,003 parameters for `enc-ols` and 20,995 for each of the other two; three seeds and six yearly fits, 54 fits;
   - the size and power of the pass rule, read from Report n°2's hourly series, with nothing refitted.

   [Report n°3](docs/reports/report-3.md): over 50,626 test hours, Brier skill scores of 0.0110 for `enc-ols`, 0.0079 for `enc-softmax` and 0.0030 for `enc-linear`, against 0.0130 for `matrix-select`. H4 holds as non-inferiority. `enc-ols` minus `enc-softmax` is -0.00269 (-0.2498 % of climatology's log-loss); the margin is 0.00538, and the one-sided upper bound -0.00182 and the bootstrap's 95th percentile -0.00181 both lie below it. H5 does not hold. `enc-ols` minus `matrix-select` is +0.00082, Diebold-Mariano p 0.5198, bootstrap interval [-0.00147, +0.00313]. The encoder refits yearly and `matrix-select` monthly. H13 has no verdict: the measurement is invalid. The registered ratio 0.4031 measures the timing stage's GPU cache, so OLS attention is not shown to be cheaper to run than softmax attention. The generated report still prints "holds" for that row; the verdict is the one in [docs/RESULTS.md](docs/RESULTS.md). H6 was not run. Its code sits on `iteration-5-ablations` at `5be70a3` and is not merged, and the two LFM3 ablations were not run either. These are forecasts, before costs.

   On Report n°2's comparisons, the pass rule's two-sided size is 2.2 % to 6.4 %, none above 7.5 %. `matrix-1y` minus `knn-1y` is -0.094 %, at 19 % to 24 % power: a difference too small to see, not evidence of none.
4. **Rust inference engine and decision journal:** a prefill-only engine and an OptMem-style decision journal.
5. **News:** through a small point-in-time text encoder, trained from scratch.
6. **Live loop:** paper trading on public data, then a 50 € test on a MiCA-authorized venue.

## Credits

- Philippe Goulet Coulombe, [Ordinary Least Squares as an Attention Mechanism](https://arxiv.org/abs/2504.09663) ([code](https://github.com/philgoucou/ols-attention)).
- Johannes von Oswald et al., the [mesa-layer](https://arxiv.org/abs/2309.05858) and [MesaNet](https://arxiv.org/abs/2506.05233): least squares solved inside a sequence model.
- Victor Taelin, [OptMem](https://github.com/VictorTaelin/OptMem): the append-only log with rolled-up summaries that inspired Matrix. No code is reused.
- Jev (TypeSafe AI), [Clef](https://blog.cloudflare.com/clef-decision-models) (Cloudflare), [Tev1](https://github.com/togethercomputer/tev1) (Together AI) and [Microsoft-Decision-1](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/): typed decision outputs. Microsoft-Decision-1 also measures how often its decisions flip under small perturbations of the inputs. All four date from autumn 2026 (Tev1's repository was created on 2026-09-23), and their results are self-reported.
- Paster, McIlraith and Ba, [You Can't Count on Luck](https://arxiv.org/abs/2205.15967): why return-to-go conditioning fails in stochastic environments.
- Market data: [Binance public data](https://github.com/binance/binance-public-data).

## License

MIT. No data is redistributed; each source keeps its own terms.
