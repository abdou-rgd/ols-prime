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

**Outcome (2026-10-28).** Done.
- Every check is green, locally and in GitHub Actions, on Windows and Linux.
- The audit passed after thirteen passes. Every finding was fixed or stated as a limit.
- H1 holds, on its second official run (commit 881a63c).
  - The 11 honest models have 0 violations.
  - The 27 controls are each on their predicted pattern, and each is caught above rounding.
  - The cheater is caught at 16 of 16 cuts.
- H9 holds at Coinbase and at Kraken. No active baseline's paired interval against buy and hold lies above 0.
- `RESULTS.md` has its first five rows, one of them an erratum.
- Abdou authorized commit, merge and push.

The first official run of H1 (commit 9c414ca) also held. CI on Linux then showed that one of its controls was caught by rounding alone. The bench was fixed and H1 was run again (see the Bench and Audit rows). Report n°1 does not use the changed code. Re-run at 881a63c with `--check`, it reproduces every number: only the commit and code digest lines differ, in `config.json` and `report-1.md`.

| Part | Result | Briefs |
|---|---|---|
| `crates/olsprime-py` | PyO3 0.30.1 and rust-numpy 0.30.0 bindings for `backtest`, `ols-core` and `risk`, importable as `olsprime_rs`. They are abi3 for Python 3.12, behind the non-default `python` feature, and built by maturin 1.15.0 through a uv path source.<ul><li>Conversions are proven to be permutations of the bits by property tests.</li><li>Strided, reversed and Fortran-ordered inputs give bit-identical results. Wrong dtypes raise before any value is read, and no panic crosses into Python.</li><li>uv rebuilds the extension after any Rust edit.</li><li>After Audit B, a shape the caller chooses can no longer abort the interpreter: allocations above 1 GiB (`MAX_ALLOC_BYTES`) are refused with a `ValueError` before anything is allocated. Integer overflow surfaces as `ValueError`, and times of 10¹⁴ or more (microseconds) are refused everywhere, `Matrix` included.</li><li>The oracle for the bindings is now a golden file written by the compiled crate: `python/tests/golden/simulate.txt`, 48 cases. `cargo run -p backtest --example golden -- --check` keeps it current, so a port of the simulator is never compared with itself.</li></ul> | `pyo3.md`, `pyo3-fixes.md`, `pyo3-fixes-2.md` |
| Features, labels, splits | 15 point-in-time features (`FEATURE_COLUMNS`). Each needs k consecutive on-grid bars and gives null across gaps; none is ever filled.<ul><li>Labels have three classes, from `ln(open[t+1+h]/open[t+1])`, found by time rather than by row offset. `label_known_at` is the exit bar's open.</li><li>Walk-forward with monthly refits. A training row needs `label_known_at ≤ fit_time − h·interval`. Guards refuse the validation year and the vault.</li></ul> | `features-labels-splits.md` |
| Baselines, decision rule | Flat, buy and hold, moving-average momentum (10, 30, 90), random with matched turnover, climatology, ridge, XGBoost, and a cheater that reads the next bar.<ul><li>`predict` refuses label columns. Scalers and λ are fitted inside the fold, with a purged inner validation.</li><li>The long/flat rule enters above the venue's round trip, 2·(taker + 2 bps), and exits below 0.</li></ul> | `baselines.md` |
| Statistics | Mertens/Opdyke Sharpe SE, PSR, DSR (N and V from the candidates), stationary bootstrap with the Politis-White block length, paired bootstrap of Sharpe differences, Diebold-Mariano with HLN, log-loss, Brier skill score, top-label and class-wise ECE.<ul><li>DM uses the rectangular kernel. Its measured size at a nominal 5 % with h = 7 is 5.25 %, against 8.75 % for Bartlett.</li></ul> | `stats.md` |
| Bench | `python -m olsprime.bench.invariance` (H1) and `python -m olsprime.bench.report1` (Report n°1), each one command and deterministic. `--check` re-runs a report and compares its 14 files byte for byte, on the same machine and library versions.<ul><li>The provenance block gives the commit, a code digest, the data manifest digest and every file's SHA-256 with its stored and sealed row counts.</li><li>After four rounds of Audit A, H1's perturbation moves every value and every ratio of every later bar, and every cut runs in both directions, ×3 and ÷3. Each run asserts this coverage.<ul><li>Bar t+1 is built against the last 91 bars, so any comparison with them flips in exactly one run.</li><li>Its candle takes opposite shapes in the two runs, and the colour alternates from cut to cut, so all four pairings of colour and size occur.</li><li>The bands where a threshold flips by construction, and what stays probabilistic, are printed in every report.</li></ul></li><li>26 leaky controls run with every H1, beside the cheater, each with an exact predicted pattern: 21 planted leaks, a return shifted the wrong way, a scaler fitted on the whole test month, and three unpurged models.</li><li>A `?` cut is predicted from the original bars, for a declared reason. A failure to cross without one fails H1, and so does a control that is `?` at every cut.</li><li>A positive control fits ridge on labels shifted one bar early. It reaches a BSS of +0.97, so the pipeline sees a real signal when there is one.</li><li>Report n°1 gains two sensitivities: longer bootstrap blocks for H9, and a Newey-West Diebold-Mariano. No verdict changes.</li><li>After the first official run, CI on Linux showed a control caught by rounding alone. The transductive scaler's ridge standardized the features again and undid the leak exactly, so its outputs moved by 10⁻¹⁶ at most. Now a control is caught only where an output moves by more than 10⁻⁹, and the report gives each control's smallest catch. The scaler's ridge keeps the scale it is given, and its catches move an output by 8·10⁻³ or more on the synthetic bars. H1 was run again on the corrected code.</li></ul> | `bench.md`, `bench-fixes.md` to `bench-fixes-4.md`, `rounding-fix.md` |
| Data | The full universe from data.binance.vision.<ul><li>99 monthly top-30 rebalances (2017-10 → 2025-12), with 274 distinct members.</li><li>1h and 1d bars of every member, and 1d bars of 568 eligible pairs: 35,189 verified archives and 843 Parquet files, 989 MB in all.</li><li>Manifest digest `230e28e8…`, with 9,977,671 rows stored and 9,964,021 after the seal.</li><li>A runtime tripwire goes up before any test module is imported. It wraps 53 read entry points, and it watches the seal.<ul><li>Under pytest, only the seal's loaders may read the repository's `data/`.</li><li>Every test gets an empty data root of its own.</li><li>A raw archive is read only when every file on the stack holds the allowance.</li><li>A check after every test ends the session if a guard was taken down.</li></ul></li><li>The static guard also flags `sys.modules`, edits of `sys.path`, patches of the seal, and the attributes that take a guard apart. It bans `runpy`, `gc`, `ctypes` and `zipimport`. Its corpus has 137 cheater snippets, each of which it must flag, and 25 controls, none of which it may.</li><li>Every evasion the audits found now trips at run time or is flagged statically, apart from the limits `docs/DATA.md` states. In plain words: these guards stop accidents, not sabotage.</li><li>`docs/DATA.md` states the survivorship evidence and the guards' remaining limits.</li></ul> | `universe-manifest-tripwire.md`, `guard-fixes.md` to `guard-fixes-4.md` |
| `crates/risk` | Kani function contracts for the arithmetic helpers, in `src/spec.rs`. They are complete specifications that pin each quotient without dividing.<ul><li>**Proven:** the contract of `product`, and `exact_never_wraps` (the specifications' 128-bit products never wrap). With the two harnesses of iteration 1, the `proven` job now has four. Re-run twice by the orchestrator in WSL2, the second time on the final crate: all four VERIFICATION SUCCESSFUL, in 23 s, 87 s, 143 s and 945 s, 16 minutes in all at `-j 4`.</li><li>**Assumed, not proven:** the contracts of `notional_floor`, `notional_ceil`, `max_qty_within` and `round_down_to_lot`. Each ran out of time at 40 minutes. Small experiments point at nonlinear integer arithmetic (lot multiples, monotone and distributive products) as what CaDiCaL cannot finish.</li><li>**Stubbed and still open:** the `check_order`, `approval` and `sell_outcome` harnesses now stub the helpers with `stub_verified`. That shrinks them from 1.2–1.4 M to 0.5–0.7 M variables, but they still run out of time at 30 to 40 minutes. No verdict rests on an unproven contract.</li><li>Five property tests, 4,096 cases each, check that each predicate accepts the helper's result and rejects its neighbours.</li><li>Every harness runs with `-Z function-contracts -Z stubbing --no-assert-contracts`. The 17 open harnesses go to GitHub's `open` job, 340 minutes each.</li></ul> | `kani-contracts.md` |
| Audit | Two `leakage-auditor`s (Fable), read-only, over thirteen passes. Every finding was fixed or stated as a limit before any result was reported.<ul><li>**Audit A, the bench.** Phase 1 found one Major: H1 left `trades` still, so a planted peeker on it was flagged at 0 of 8 cuts. It also found nine Minor points, all fixed. Phase 2 broke the vault guards twice (D1, D2) by reading vault rows inside a test, an exposure recorded in `PROTOCOL.md`.</li><li>**Re-checks of the bench.** The first found that ratios inside a bar did not move: six planted leaks at 0/8, now 8/8. The second found that the candle's shape did not flip, and that a `?` was read from the outcome, so a broken push would have passed. The third found colour tied to size: a "small red bar" leak was flagged at 0 of 8 cuts. Rounds 3 and 4 fixed all three. The final verification passed: every band holds by construction, and sabotaging the trades, fill or mirror push makes H1 fail, as it must. Its last two Minor points were fixed by the orchestrator before the commit: the close-location `?` now holds at both parities, and LIMIT gives the ÷3 bar's range as about 20 log units.</li><li>**Round 5 of Audit A, the rounding fix.** Pass with fixes. The diagnosis held, and the 10⁻⁹ rule loosens nothing that matters: on the real data every honest model moves by 0 at every cut, no model has a rounding-only cut, and the weakest catch of any control moves an output by 7.1·10⁻⁴ (climatology-unpurged). Its two Minor points were fixed: the table shows a control's caught rows, and `RESULTS.md` gets an erratum row instead of a silent replacement.</li><li>**The guards.** A dedicated re-audit found three more routes and eight smaller holes, fixed in round 2. Its re-check ran 56 evasion cases on a fake tree. All of round 1 held, and it found two more ordinary routes: a test allowed to read Parquet could name the real `data/`, and `zipimport` read the raw archives. Round 3 closes both, and also refuses forged keys on a copy of a vault file. A last check found Minor points only: rarer spellings of a path, such as a `file://` URI, and a transitive exemption. Round 4 closes or states them.</li><li>**Audit B, Rust.** Phase 1: six Minor points. Phase 2: the four proven Kani harnesses are genuine, and a mutation of `product` makes `product_contract` fail. One PyO3 Major (a shape the caller chooses aborted the interpreter) and four Minor points, all fixed and re-checked.</li></ul> | `audit.md`, `bench-fixes*.md`, `guard-fixes*.md`, `pyo3-fixes*.md` |

**Other results.** GitHub's `open` Kani job (run 40596921451, on the iteration 1 code) ended without a verdict: none of the twelve harnesses finished within 340 minutes. So there is no counterexample, and no proof.

**Lessons.**
- **A detector sees only what its perturbation moves.** Four rounds of the audit each found a blind spot in H1, by planting a leak the bench had never seen:
  - columns: `trades`, the volume level, the taker share;
  - ratios inside a bar;
  - the candle's shape;
  - colour tied to size.

  "Every value moves" is not "every function of values moves". Comparisons with the past can be made to flip by construction. The rest is probabilistic, and the report says so.
- **A control must be checked too.**
  - The transductive scaler fed its scaled features to a ridge that standardized them again, which undid the leak exactly. A bit-for-bit detector still "caught" it, through rounding, at every cut on Windows.
  - It took a second platform to show it.
  - A catch now has to move an output by more than 10⁻⁹, and every report gives each control's smallest catch.
- **A verdict must never be computable vacuously.** A `?` read from the outcome would have turned a broken push into a pass.
- **In-process guards stop accidents, not sabotage.**
  - Four rounds of red-teaming found ordinary routes: forged keys, a raw reader in an allowed module, a direct path to the real `data/`, `zipimport`, and rare spellings of a path.
  - Each fix added a rule about paths.
  - The structural fix is to make the vault unreadable, not watched.
- **A proof of a hole must run on fakes.** Audit A's evasion tests read vault rows, an exposure now recorded in `PROTOCOL.md`.
- **PyO3: "nothing else is raised" must hold for every input a caller chooses.** An oversize shape aborted the interpreter before the bound check.
- **Kani: nonlinear integer arithmetic is the wall.**
  - Lot multiples and monotone and distributive products defeat the SAT solver.
  - Contracts and stubs shrink the formulas two- to threefold, which is not enough.
  - 340 minutes on GitHub's runners gave no verdict either.
- **Process.**
  - Official runs wait until CI is green on every platform. The first H1 run started while CI was still running.
  - A WIP commit caught files mid-edit and turned CI red.
  - `uv run` rebuilds the extension after a Rust edit, so use `--no-sync`.
  - Heredocs eat backslashes, twice in this iteration. Write scripts with a file tool.
  - Exact equality between two libraries is not portable. After the merge, the push to `main` failed on one Linux runner: a label test compared `np.log` with `math.log` at zero tolerance, and they differed by one ulp. The pull request's runners had passed the same tree. The test now allows 1e-14 relative (pull request #2).
  - Headless Edge will not render narrower than about 477 px.
  - Several Fable auditors in parallel hit the session's usage limit, so run at most three agents at once.

**Bolder next step.** Put Matrix on the whole universe, behind a seal the process cannot open, and give H1 an adversary.
- **The physical seal.**
  - The vault months move out of `data/`, both the Parquet rows and the raw archives, into a tree that opens only with both keys.
  - The in-process guards then no longer stand between a test and the vault, and their path rules can go.
- **An adversary for H1.**
  - A generator plants thousands of random leaks: functions of the next bars' levels, ratios and shapes.
  - H1 reports the share it catches, with an interval, and the size of each catch.
  - The hand-picked list of planted leaks becomes a measured detection rate. The misses show where the next designed run should go.
- **Matrix at scale.** Matrix and the OLS-attention readout run on the hourly bars of all 274 members, and H2, H3 and H10–H12 are tested on the development period.

**Learning page.** [Le banc honnête](https://claude.ai/artifact/VDdujVgLyyi2PcMBnaM1Sd), in French, written by `teacher` and checked by the orchestrator. It is private to Abdou's account.

## Iteration 3 — Matrix at scale, behind a physical seal (2026-10-29)

**Goal.** This is roadmap phase 2, "Matrix on hand-made features", made bolder as iteration 2's outcome proposes:
- Matrix and its OLS-attention readout on the hourly bars of every universe member, testing H2, H3 and H10–H12;
- an adversary that measures how many random leaks H1 catches;
- the vault moved out of `data/`.

**Deliverables.**
1. **A physical seal.**
   - `olsprime.vault` moves the vault months out of `data/` into a vault tree outside the repository: the raw archives, and the Parquet files that hold vault rows.
   - It rebuilds those Parquet files from the remaining archives, and checks that every sealed read is unchanged.
   - The downloader and `build_parquet` refuse vault months.
   - Encryption of the vault tree to an age recipient is written and tested on fake trees.
   - The move is designed and tested on fake trees first. The real move runs once `leakage-auditor` has reviewed the tool.
   - Encrypting the real vault needs a private key that never touches this machine, so that step waits for Abdou.
2. **An adversary for H1.** `python -m olsprime.bench.adversary` runs a seeded generator of random leaks: functions of the next bars' levels, ratios and shapes. Each leak goes through H1's perturbation, at H1's cuts and at a longer schedule.
   - It reports the share caught, with a Wilson interval, per family and per cut; the size of each catch; and the misses.
   - Every leak has an honest twin, which must never be caught.
3. **Matrix at scale.**
   - `ols-core`: nodes in centred form (Chan, Golub and LeVeque), and the ridge solved in centred form with a free intercept and a penalty per coordinate. Batch readouts over many queries, windows and penalties, and exact kernel readouts (softmax and kNN) for H3, all with Python bindings.
   - A point-in-time hourly panel of the universe members: rows only in member months, features on each pair's own history, labels with h = 24 hours, and the validation year and the vault left out.
   - The Matrix arms of H2, H3, H10, H11 and H12, and a future-invariance test on the panel with planted online leaks.
   - Report n°2 as one command.
4. **Kani.** The `open` run of 2026-10-28 proved `max_qty_within_contract`. It moves to `proven`.

**Scope, fixed before any result.** `docs/PROTOCOL.md`, "Matrix on hourly bars (iteration 3)", holds the full pre-registration.
- **Data.** All 274 members, in their member months only: at most 30 pairs in any hour. About 1.8 M labelled rows were expected in the development period. The built panel has 1.63 M, from the 233 members that have a member month before 2025 (`docs/DATA.md`).
  - The workload was sized first: about 2 M entries and 2 M queries, each read under 3 windows and 6 penalties, plus exact kernels over about 263 k candidates per query. It fits in the Rust core in well under an hour, so no subset of members is needed.
  - The fallback that keeps it within the session is fixed in advance: if the one-year kernels cannot run in time, H3 runs at the 30-day window.
- **Design.** Hourly decisions, h = 24 bars, test months 2019-01 → 2024-12, as for H9. One pooled memory over all members.
- **Metrics.** Forecast quality only, the protocol's primary metric. Trading metrics of the Matrix arms wait for a portfolio simulator, and each arm still counts as a variant.
- **Priority if time runs out.** H2 and H3 first, then H10, H11 and H12. A hypothesis that was not run is reported as not run.

**Hypotheses touched.** H1 (the adversary, and invariance of the Matrix arms on the panel), H2, H3, H10, H11, H12.

**Done when.**
- Every check of `CLAUDE.md` is green, locally and in GitHub Actions, on Windows and Linux.
- On the panel, H1 holds for every Matrix arm, and its planted online leaks are caught where predicted.
- The adversary and Report n°2 each run as one command, and a second run reproduces each of them.
- `RESULTS.md` has the rows of the adversary, the panel H1 and H2, H3, H10–H12, each with its variant count.
- `leakage-auditor` passes.
- The vault months are out of `data/` with every sealed read unchanged, or the reason they are not is recorded.
- The pull request is merged into `main` with a local merge commit and pushed. Abdou authorized commit, merge, push and installs for this iteration on 2026-10-29.

**Plan.** Briefs are in `.claude/briefs/iteration-3/`. At most three agents run at once.

| Wave | Task | Agent | Brief |
|---|---|---|---|
| 1 | Centred nodes, batch and kernel readouts, bindings | `rust-engineer` | `matrix-core.md` |
| 1 | Hourly panel of the members | `data-engineer` | `panel.md` |
| 1 | H1 adversary | `ml-engineer` | `adversary.md` |
| 2 | Physical seal | `data-engineer` | `seal.md` |
| 2 | Matrix arms, panel H1, Report n°2 | `ml-engineer` | `matrix-arms.md` |
| 3 | Official runs and rows | `experiment-runner` | `runs.md` |
| 4 | Audit | `leakage-auditor` | `audit.md` |
| 5 | Learning page | `teacher` | `learning-page.md` |

**Set up by the orchestrator before wave 1.**
- `main` holds iteration 2 (merge commit `ae716a0`, pull request abdou-rgd/ols-prime#1) and the label-test fix (pull request abdou-rgd/ols-prime#2). Work happens on the branch `iteration-3`.
- The Kani `open` run of 2026-10-28 (run 40636068905, 17 harnesses, 340 minutes each) is recorded in the outcome below. Its one proof moves to the `proven` job of `kani.yml`, to the Verification section of `CLAUDE.md` and to `README.md`. The `proven` job's time limit rises from 60 to 150 minutes.
- A `scout` read Liquid AI's LFM3 announcement at Abdou's request. See the outcome.
