# OLS Prime

**A from-scratch decision model whose attention is ordinary least squares, tested on crypto markets without fooling ourselves.**

OLS Prime is a local-AI lab. Crypto trading is the playground, not the goal. Every component is a hypothesis: it is registered before it is tested, then measured by ablation against simple baselines on a leak-free backtest bench. A clean negative result counts as a result.

> Status: iteration 1, foundations. Nothing here trades yet, and nothing here is investment advice.

## The idea

- **OLS is attention.** Goulet Coulombe (2025) shows that an OLS forecast is an attention mechanism. The forecast `ŷ = x_qᵀ (XᵀX)⁻¹ Xᵀ y` equals `Σᵢ wᵢ yᵢ`, and the weights `wᵢ = xᵢᵀ (XᵀX)⁻¹ x_q` are *signed*. OLS Prime's encoder heads solve a small ridge regression instead of applying a softmax.
- **Matrix remembers.** Past states and their realized outcomes go into an append-only log. Older entries are rolled up into a binary tree, as in Victor Taelin's OptMem, except that each node holds the sufficient statistics of a regression (`n, Σz, Σzzᵀ, Σyzᵀ`).
  - Those statistics add up, so any period is the sum of O(log n) nodes.
  - Every forecast splits *exactly* into signed contributions of past days and periods.
  - The name is a nod to Optimus Prime's Matrix of Leadership, which carries the wisdom of the Primes before him.
- **One pass, typed and calibrated.** Like Jev-style decision models, the output is a probability over a fixed set of options: down, neutral or up over a horizon. It is trained with cross-entropy plus a Brier term. A fixed-point Rust risk kernel then turns those probabilities into long/flat positions.

## Why from scratch

A pretrained LLM has already read the news of the years we would backtest on, so its backtest is contaminated by look-ahead bias. Masking names or dates does not remove that bias ([Lopez-Lira, Tang and Zhu, 2025](https://arxiv.org/abs/2504.14765)). A small model trained only on point-in-time data avoids the problem by construction, and it fits on a single RTX 3060.

## Layout

| Path | What |
|---|---|
| `crates/ols-core` | closed-form ridge, dual (attention) weights, recursive least squares with forgetting, Matrix |
| `crates/backtest` | bar simulator (decide at the close, execute at the next open), venue fee schedules, metrics |
| `crates/risk` | fixed-point risk kernel with Kani proof harnesses |
| `python/olsprime` | data pipeline (data.binance.vision), point-in-time universe; models and training later |
| `docs/` | [design](docs/DESIGN.md), [evaluation protocol](docs/PROTOCOL.md), [pre-registered hypotheses](docs/HYPOTHESES.md), [results](docs/RESULTS.md) |

## Getting started

```bash
cargo test --workspace
```

The Kani proofs need Linux (WSL2 works). Two of the fourteen harnesses finish, in about 11 minutes together on the development machine:

```bash
cargo kani -p risk --exact --harness proofs::buy_gates_block_every_buy --harness proofs::sells_are_never_gated
```

`cargo kani -p risk` runs all of them. The other twelve ran out of time at one hour each, on a shared machine. [`kani.yml`](.github/workflows/kani.yml) keeps both lists.

## Roadmap

0. **Foundations:** repository, data pipeline, CI.
1. **Honest bench:**
   - Rust backtest with real venue fees;
   - risk kernel;
   - walk-forward protocol and baselines.

   Report: what fees do to simple strategies.
2. **Matrix on hand-made features:** the paper's OLS attention, extended to time series.
3. **OLS Prime encoder, from scratch:** ablations against softmax and linear attention.
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
