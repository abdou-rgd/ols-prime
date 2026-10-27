# Iterations

One Claude Code session delivers one iteration. The orchestrator writes the brief before the work starts and the outcome when it ends. The motto applies: each outcome names one bolder next step.

## Iteration 1 — Foundations (2026-10-25)

**Goal.** Repository, protocol and hypotheses, the Rust core crates, the data pipeline, CI.

**Deliverables.**
- Docs: `DESIGN`, `PROTOCOL`, `HYPOTHESES`, `RESULTS`, `ITERATIONS`; `CLAUDE.md` and the six agent roles.
- `crates/ols-core`, with property tests:
  - `SuffStats`;
  - ridge in primal and dual (attention) form;
  - RLS with forgetting;
  - Matrix.
- `crates/backtest`: bar simulator, venue fee schedules, metrics. Tests include future invariance.
- `crates/risk`: fixed-point kernel, with property tests and Kani harnesses.
- `python/olsprime`:
  - data.binance.vision downloader, quality report, point-in-time universe, all tested offline;
  - BTC and ETH, 1 h and 1 d, downloaded.
- CI: fmt, clippy, tests, Kani, pytest.

**Done when.**
- Every test is green.
- `leakage-auditor` passes.
- Abdou approves the commit.

**State at hand-off (2026-10-25).** A setup session wrote the docs, `CLAUDE.md` and the agent roles. It then handed the iteration over to a fresh session, where the agents load.

| Deliverable | State | Brief |
|---|---|---|
| Docs, `CLAUDE.md`, agents | done | |
| `crates/ols-core` | Written by the orchestrator. 25 tests green; clippy and fmt clean. Not yet audited. | |
| `crates/backtest` | Draft from a stopped run. 37/37 tests green. Not yet reviewed against its brief. | `.claude/briefs/iteration-1/backtest.md` |
| `crates/risk` | Draft from a stopped run. 28/29 tests green. Kani never run. | `.claude/briefs/iteration-1/risk.md` |
| `python/olsprime` | Built in its own session. 35 offline tests green, re-run by the orchestrator. BTCUSDT and ETHUSDT, 1 h and 1 d, 2017-08 → 2026-09: downloaded, checksums verified, stored as Parquet. The full universe has not been downloaded. Not yet audited. | |
| CI, `docs/DATA.md` | not started | |
| Audit | not started | `.claude/briefs/iteration-1/audit.md` |

**Plan for the session that takes over.**
1. Run two `rust-engineer` tasks in parallel: one on `backtest.md`, one on `risk.md`. Meanwhile the orchestrator writes:
   - `.github/workflows/ci.yml`, with fmt, clippy, test, Kani and pytest jobs. Check the current inputs of the Kani and uv actions first.
   - `docs/DATA.md`, listing the sources and their licenses.
2. The data-pipeline session has reported. Review its diff and re-run `uv run pytest`.
3. Run the full verification in `CLAUDE.md`.
4. Run `leakage-auditor` on `audit.md`. Fix or log every finding.
5. Write the outcome below, with one bolder next step. Abdou approves the first commit.

**Outcome (2026-10-26).** Done.
- Every check is green.
- The audit passed with fixes after four passes, and every finding was fixed or logged.
- Abdou authorized the first commit.
- No experiment ran, so `RESULTS.md` has no row.

| Deliverable | State | Brief |
|---|---|---|
| Docs, `CLAUDE.md`, agents | Done. A seventh role, `teacher`, now writes each iteration's learning page (step 6 of the loop). | |
| `crates/ols-core` | 38 tests. After the audit:<ul><li>A solve is refused unless the Gram matrix, scaled to a unit diagonal, is numerically positive definite: `λ_min(D^-½ M D^-½) > 4p(N+p+2)ε`. The derivation is in the `RidgeConfig` docs.</li><li>Every output is checked finite. Negative weights are refused.</li><li>A `Readout` is valid only on the Matrix instance that produced it.</li><li>`time` is documented as the bar's `known_at`.</li></ul> | `audit-fixes-rust.md` |
| `crates/backtest` | 59 tests. The dust floor is pinned from both sides. Mutant B9 now fails `rebalances_above_the_dust_floor_fill`; the orchestrator re-ran it. | `backtest.md`, `audit-fixes-rust.md` |
| `crates/risk` | 40 tests. Kani 0.68.0 ran the 14 harnesses one per process, each with a 60-minute cap. Up to 12 processes shared the 12-thread, 15 GB machine:<ul><li>2 are proven, with every cover satisfied: `buy_gates_block_every_buy` and `sells_are_never_gated`. They took 795 s and 29 s in the shared queue, and 611 s and 14 s when re-run alone on the final code.</li><li>12 ran out of time without a verdict, so they prove nothing yet. The 21 property tests check P1–P7 on random inputs, against a reference implementation.</li></ul> | `risk.md`, `kani-narrow.md` |
| `python/olsprime` | 146 offline tests. The vault is sealed in code (`protocol.py`).<ul><li>Every read seals on `known_at`: BTCUSDT 1 h returns 73,285 of its 79,837 bars.</li><li>Two keys open it, and every open read is logged. Git tracks the key file and the log.</li><li>An AST guard walks every `.py` and `.ipynb` file under `python/`. Its coverage is measured by cheater snippets, each of which it must flag.</li></ul> | `vault-lock.md`, `vault-guard-fixes.md` |
| CI, `docs/DATA.md` | Written. All pins were verified. Rust 1.89 (the MSRV) was checked locally. Kani has its own workflow, `kani.yml`:<ul><li>the two proven harnesses run on every push and pull request that touches `crates/risk`;</li><li>the twelve open ones run only when started by hand, each within a time limit. Running out of time is a warning; a counterexample fails.</li></ul> | |
| Audit | Four passes, each "pass with fixes" and none with a Blocker:<ul><li>pass 1: 2 Major, 6 Minor;</li><li>re-audit of the vault and the docs: 7 Minor;</li><li>re-audit of the Rust fixes and the guard: 7 Minor;</li><li>a focused check of the Kani results, `kani.yml` and the claims about Kani: 5 Minor, all on wording or unmeasured detail. Each is now fixed or measured.</li></ul> | `audit.md`, `audit-kani-ci.md` |

**Audit findings and what became of them.**
- **M1, a vault lock was needed now.** The Parquet files hold the vault months, and `universe --end` defaulted to the current month. Fixed, see `python/olsprime`.
- **M2, the dust floor was untested.** Mutant B9 survived the whole suite. Fixed by three tests.
- **A singular solve could succeed silently.** Probe P4 gave `Ok(1.5)` for two identical entries at λ = 0. Fixed by the eigenvalue criterion, with λ ∈ {0, 1e-12} in every identity generator and an SVD cross-check.
  - The auditor's pivot test was not used: it accepts a rank-1 design.
  - The re-audit found an undercounted rounding term. The final tally is (2N + 11)ε rather than (N + 7)ε. The factor 4 in τ absorbs it, and the docs now say so.
  - The re-audit also mapped the refusal frontier. A feature beside the intercept with `|mean|/std` above about 4·10³ is refused at λ = 0 once N reaches 10⁶. So features must be centred inside the fold (`docs/DESIGN.md` §1).
- **A Readout was accepted by a foreign Matrix.** Fixed with instance ids.
- **Which timestamp Matrix's `time` is.** It is now documented, with a doctest on the truncated bar of 2020-12-21 14:00.
- **Kani future-incompat warnings.** Fixed. Five `=> assert!(…)` match arms in `proofs.rs` became blocks, as rustc's `semicolon_in_expressions_from_non_local_macros` lint asks (rust-lang/rust#79813). On the final code, with the `proven` job's command, both proven harnesses pass again (611 s and 14 s), and the warning is gone.
- **Sharpe has no interval.** Logged for iteration 2, before any row reaches `RESULTS.md`.
- **Wording in `docs/DATA.md` and `README.md`.** Fixed.
- **Rejected: eigensolver misconvergence.** nalgebra 0.35's `symmetric_eigenvalues` iterates until convergence (`max_niter = 0`), so a misconverged value is never returned.
- **Claims the audit did not check.** A `scout` checked them.
  - Kraken and Binance fees and the ESMA register rows are confirmed.
  - The Coinbase EU fees are unverified at the source (HTTP 403 on every page), though two news reports agree.
  - That Binance left France on 2026-07-01 rests on press reports.
  - All four decision models date from autumn 2026.

**Lessons.**
- **One scratch folder per parallel agent.** The backtest and risk agents shared one, and a mutation script was overwritten.
- **A summarizing fetch tool paraphrases numbers.** A page read through it counts as read, not verified. The scout found no date for Tev1; the GitHub API gives 2026-09-23.
- **Python's text mode on Windows writes CRLF.** Agents' edit scripts converted files. `.gitattributes` now normalizes line endings.
- **Kani's cost is in the arithmetic.** CBMC bit-blasts every 64- and 128-bit product and quotient. Facts as simple as `(q / l)·l mod l = 0` are hard for a SAT solver, and `kani::assume` on magnitudes does not shrink the circuit.
  - Lot rounding builds about 20,000 variables and 99,000 clauses both on the full `i64` range and at every bound from 2⁸ to 2³².
  - The `check_order` circuits are close in size: 1.43 M variables for `_full`, 1.32 M for MARKET and 1.21 M for TINY, whose harnesses also check fewer covers. None of them finishes.
  - Narrow inputs do not shrink it either. Drawn from sign-extended narrow integers, every circuit stayed within 0.1 % of its size, and every run was as slow.
  - On lot rounding over b-bit inputs, verification took 0.85 s at 8 bits, 3.1 s at 12 and 73 s at 16. At 20 bits it ran out of a 45-minute cap.
  - Kissat (`--solver kissat`, which CBMC runs as an external SAT solver) did no better than CaDiCaL: both of its harnesses timed out at 40 minutes.
- **WSL2 needs the Virtual Machine Platform and SVM in the firmware.**
- **Never pass LaTeX through a shell heredoc.** It ate the learning page's backslashes.

**Bolder next step.** Make Report n°1 a single reproducible command.
- It runs from the verified archives, through `load_klines`, the Rust simulator (through PyO3) and the risk kernel, to the table in `RESULTS.md`.
- It prints each Parquet file's SHA-256 and sealed row count.
- It includes a cheater baseline, built from `close[t+1] > open[t+1]`, which H1's detector must catch and the vault seal must refuse.
- It puts a Sharpe interval and a deflated Sharpe on every row.

**Learning page.** [Comprendre OLS Prime](https://claude.ai/artifact/386T8BiAjdUfbmjMZ1niGw), in French, written by `teacher` and checked by the orchestrator. It is private to Abdou's account.

## Iteration 2 — Honest bench (2026-10-27)

**Goal.** One command, from the verified archives to the rows of Report n°1: "what fees do to simple strategies". The bench must catch a model that reads its own label.

**Deliverables.**
- `crates/olsprime-py`: PyO3 bindings for `backtest`, `ols-core` and `risk`, importable as `olsprime_rs` from the uv environment, on Windows and in CI.
- Point-in-time hand-made features, and forward labels that carry the instant they become known.
- A walk-forward splitter with monthly refits, purge and embargo.
- Baselines: climatology, always flat, buy and hold, momentum, random positions with matched turnover, ridge, XGBoost. Plus a cheater that reads the next bar.
- Statistics on every reported number:
  - Sharpe standard error for non-normal returns (Mertens 2002, Opdyke 2007) and a stationary block-bootstrap interval;
  - probabilistic and deflated Sharpe ratios (Bailey and López de Prado 2012, 2014);
  - log-loss, Brier skill score, expected calibration error, Diebold-Mariano with HAC variance.
- H1: the future-invariance test, run on every model, and the cheater flagged.
- Report n°1 as a single command, `uv run python -m olsprime.bench.report1`, deterministic. It prints the vault manifest: each Parquet file's SHA-256 and sealed row count.
- The full universe: daily bars of every eligible USDT pair, the monthly top 30 from 2017-10 to 2025-12, and hourly bars of every member.
- Carried over from iteration 1:
  - the vault manifest above;
  - a runtime tripwire in `python/tests/conftest.py`: a Parquet read outside the allowed files fails, whatever the indirection;
  - Kani function contracts for the arithmetic helpers, used by the `check_order` harnesses through `stub_verified`, with all 14 harnesses in the `proven` job as the aim;
  - the venues' fee pages re-read at the source before Report n°1 uses them.

**Hypotheses touched.** H1 is tested. H9 is registered before Report n°1 runs.

**Done when.**
- Every check of `CLAUDE.md` is green, locally and in GitHub Actions.
- Report n°1 runs as one command, and a second run reproduces its table exactly.
- `RESULTS.md` has the H1 row and the Report n°1 rows, each Sharpe with its interval, its deflated value and the variant count.
- `leakage-auditor` passes.
- The pull request is merged into `main` and pushed. Abdou authorized commit, merge and push for this iteration on 2026-10-27.

**Plan.** Briefs are in `.claude/briefs/iteration-2/`.

| Wave | Task | Agent | Brief |
|---|---|---|---|
| 1 | PyO3 bindings | `rust-engineer` | `pyo3.md` |
| 1 | Kani function contracts | `rust-engineer` | `kani-contracts.md` |
| 1 | Universe download, vault manifest, runtime tripwire | `data-engineer` | `universe-manifest-tripwire.md` |
| 1 | Fees, formulas and tool facts at the source | `scout` | `sources.md` |
| 2 | Features, labels, walk-forward splits | `data-engineer` | `features-labels-splits.md` |
| 2 | Baselines, decision rule, cheater | `ml-engineer` | `baselines.md` |
| 2 | Statistics | `ml-engineer` | `stats.md` |
| 3 | H1 bench and the Report n°1 command | `ml-engineer` | `bench.md` |
| 3 | Run Report n°1 twice, record the rows | `experiment-runner` | `report1-run.md` |
| 4 | Audit | `leakage-auditor` | `audit.md` |
| 5 | Learning page | `teacher` | `learning-page.md` |

**Set up by the orchestrator before wave 1.**
- The public repository [abdou-rgd/ols-prime](https://github.com/abdou-rgd/ols-prime), created at Abdou's request, with iteration 1 pushed to `main`. That commit now carries Abdou's GitHub noreply address, again at their request. Work happens on the branch `iteration-2`.
- The `open` Kani job was started on GitHub runners for the twelve unproven harnesses, 340 minutes each, on the iteration 1 code (run 40596921451). Public repositories pay no Actions minutes.
- `scipy` and `xgboost-cpu` were added to `python/pyproject.toml`. The empty packages `features`, `labels`, `eval`, `models` and `bench` were created, so that parallel agents never create the same file.
