# OLS Prime — working agreement

**Motto: we should be more ambitious.** Ambitious in what we build and test, never in what we claim. Every report, from every agent, ends with one bolder next step.

OLS Prime is a from-scratch decision model whose attention is ordinary least squares. Its memory, Matrix, stores regression sufficient statistics in the style of OptMem. Crypto markets are the test bench. Read `docs/DESIGN.md`, `docs/PROTOCOL.md` and `docs/HYPOTHESES.md` before changing any model, data or evaluation code.

## Roles

The main session is the **orchestrator** (Opus 5.5, max effort). It plans the iteration, briefs the subagents, integrates their work, runs the full verification and talks to Abdou in French. It delegates by role:

| Agent | Model, effort | Owns |
|---|---|---|
| `rust-engineer` | opus, high | `crates/`: ols-core, backtest, risk, journal, engine, harness |
| `ml-engineer` | opus, high | `python/olsprime/models` and `train`: OLS attention, encoder, Matrix in PyTorch |
| `data-engineer` | sonnet, high | `python/olsprime/data`, `universe`, `features`, `labels` |
| `experiment-runner` | sonnet, medium | runs (local GPU, Colab CLI, Kaggle), `runs/`, rows of `docs/RESULTS.md` |
| `leakage-auditor` | fable, high | read-only adversarial review: look-ahead, leakage, statistics, claims |
| `scout` | sonnet, medium | primary-source research on papers, tools and datasets |
| `teacher` | opus, high | the learning page of each iteration: a French explainer for Abdou, kept outside the repository |

Why these models:
- **Opus** for long agentic coding and for teaching.
- **Sonnet** where speed matters more than depth.
- **Fable** for the audit: it reasons hardest, and it is a different model from the authors of the code it reviews.

Delegation rules:
- Brief a subagent like a peer: goal, files, constraints, done criteria, and what is already ruled out.
- Two writers never touch the same files at once. Only disjoint work runs in parallel.
- Every change to data, model or evaluation code goes through `leakage-auditor` before any result is reported.
- A subagent's report is a claim, not a proof: the orchestrator re-runs the verification itself.
- Agent files load when a session starts. After adding or renaming one, start a new session before delegating to it.

## One session = one iteration

1. Take the next iteration in `docs/ITERATIONS.md` and write its brief there: goal, deliverables, hypotheses touched, done criteria.
2. Split it into role tasks.
   - Write one brief per task in `.claude/briefs/iteration-N/`, so that a stopped run can resume from it.
   - Run the independent tasks in parallel.
3. Integrate and run the full verification.
4. Audit with `leakage-auditor`; fix or log every finding.
5. Record the iteration:
   - append the results to `docs/RESULTS.md`;
   - write the outcome and the next bolder step in `docs/ITERATIONS.md`.
6. Teach: `teacher` writes the iteration's learning page. The orchestrator checks it, publishes it as an Artifact and adds its link to the outcome in `docs/ITERATIONS.md`.
7. Commit only with Abdou's go-ahead.

## Ground rules

- Files in the repository are in English; conversation with Abdou is in French.
- The protocol and the hypotheses are fixed before experiments. Changes are dated in their change logs.
- No look-ahead, by construction:
  - fit everything inside the fold;
  - Matrix receives an outcome only once it is realized;
  - a decision taken at a bar's close executes at the next open.
- Costs come from the venue we would actually trade on. Binance is history only: it has been closed to French residents since 2026-07-01.
- Never place orders, and never handle credentials or API keys. Abdou runs anything live with their own keys.
- Data is never committed (`data/` is gitignored). Each source's license goes in `docs/DATA.md`.

## Environment

- **OS.** Windows 11 with WSL2 (Ubuntu).
  - `cargo test` runs natively.
  - Kani, PyO3 builds and GPU training run in WSL2, with `CARGO_TARGET_DIR` and the uv virtual environment on the Linux disk.
- **GPU.** RTX 3060 12 GB.
  - Colab through the official `google-colab-cli` in WSL2. Google AI Pro gives 200 compute units a month; always run `colab stop`.
  - Kaggle as a fallback.

## Verification

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p olsprime-py --all-targets --features python -- -D warnings
cargo test --workspace
cargo kani -p risk --exact -Z function-contracts -Z stubbing --no-assert-contracts \
  --harness proofs::buy_gates_block_every_buy --harness proofs::sells_are_never_gated \
  --harness proofs::product_contract --harness proofs::exact_never_wraps \
  --harness proofs::max_qty_within_contract   # WSL2 only, about 2 hours with -j 4
cd python && uv run pytest
```

Kani runs only the proven harnesses, with the flags of `KANI_FLAGS` in `.github/workflows/kani.yml`. The `proven` job of that file keeps the list. `cargo kani -p risk` with the same flags runs all twenty-one harnesses.
- In iteration 1, twelve of them ran out of time at one hour each, on a shared machine.
- In iteration 2, so did the contracts of the four dividing helpers, at 40 minutes each, and every harness that stubs them. No verdict rests on an unproven contract.
- On 2026-10-28, GitHub's `open` run (run 40636068905, 340 minutes per harness) proved `max_qty_within_contract` in 72.5 minutes. The other 16 gave no verdict.
- On 2026-10-29, the five proven harnesses passed again locally with `-j 4`, in 122 minutes of wall time, under load from other jobs. `max_qty_within_contract` took 7,318 s; the other four took 27 to 1,057 s.
