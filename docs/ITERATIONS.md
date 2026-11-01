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

**Outcome (2026-10-29).** Done.
- Every check is green, locally and in GitHub Actions, on Windows and Linux (pull request abdou-rgd/ols-prime#3).
  - On GitHub, for each commit under test, `3a22daa` and `31c8005`, before its runs: rustfmt, clippy and rustdoc, MSRV, and cargo test and pytest on both platforms.
  - Kani's `proven` job, now five harnesses, passed on GitHub for `3a22daa` in 91 minutes (run 40842249385, 08:27 → 09:58 UTC) and for `31c8005` in 74 minutes (run 40854754950, 10:05 → 11:20 UTC). Its inputs, the risk crate, the workspace manifests and `kani.yml`, have not changed since `3a22daa`.
  - Locally, all exit 0: the same checks on Windows, and in WSL2 Linux on a clean clone of `3a22daa` without `data/` (04:34 → 04:41 UTC). After the ECE fix, pytest on Windows again: 1,262 passed, 3 skipped.
- The audit's phase 1 took six passes before any run, 1a to 1f, and ended with GO. Phase 2, on the results, gives GO to record and merge, with no Blocker and no Major point. Every finding of both phases was fixed or logged (see the Audit row).
- The vault months are out of `data/`, with every sealed read unchanged: stage 1 of the physical seal. Stage 2, encryption, waits for Abdou's offline key.
  - Until then the vault tree is plaintext, and only the in-process guards stand between it and our code.
  - The vault months are public on data.binance.vision anyway: the seal stops our own pipeline and agents, not a person.
- **H1 holds on the panel** at `3a22daa`. The eight honest arms have 0 violations at every cut. Each of the five planted online leaks is caught on its predicted pattern of cuts.
- **The adversary catches 9,795 of 10,000 random leaks at H1's cuts**: 98.0 %, Wilson [97.7, 98.2] %. Its honest twins: 0 caught.
  - A second run with 4 workers instead of 8 gives byte-identical files.
  - Of the 205 leaks caught at no cut, 163 are constant at the compared rows. The audit's phase 2 classified the other 42, at Abdou's request. None shows a hole beyond the limits the protocol already states.
    - 11 are thresholds outside the bands where H1's perturbation flips by construction, such as a fill above 0.97. The long schedule catches each of them at 1 to 12 of its 128 cuts, when a real extreme bar comes along.
    - 18 are aggregates of future bars compared with a past bar or window. The comparison is so lopsided that H1's perturbation does not flip it at H1's cuts. The long schedule catches 16 of them, at 1 to 19 of its cuts.
    - 13 are combinations. 2 of them depend on the past alone and are no leak: their XOR cancels the future bar. The headline counts them as misses, which is conservative.
- **Report n°2: Matrix forecasts better than climatology on hourly bars, by a small margin. H2 holds.**
  - The Brier skill score of `matrix-select` is 0.0130 [0.0108, 0.0156], over 50,626 test hours of 233 pairs, 2019-01 → 2024-12.
  - On the 54 test months that no smoke run had seen, it is 0.0108 [0.0083, 0.0132], against 0.0201 [0.0143, 0.0265] on the 18 that smoke runs had scored. That split is a description fixed before the run.
- **H3 does not hold.** On the same one-year memory, the OLS readout beats softmax attention by 0.83 % of climatology's log-loss (Diebold-Mariano p < 0.0001, bootstrap interval below 0). It does not beat kNN: −0.09 %, p 0.29, interval [−0.26 %, +0.08 %]. H3 needs both.
- **H10 and H11 do not hold.**
  - Choosing ρ at each decision from its prequential error is worse than the monthly selection: +0.12 %, p 0.006, interval above 0. The registered rule is directional and the test two-sided, so the significance in the other direction is a description, not a registered finding.
  - Stacking the three windows is better at the point, −0.06 %, but p is 0.07 and the interval ends just above 0, at +0.00001.
- **H12 holds.** Refitting `matrix-select` on CUSUM alarms is no worse than refitting it every month, with 13 fits instead of 72. The difference is −0.00002; its one-sided upper bound is +0.00053, against a margin of 0.00538.
- These are forecasts, not trades. Costs are in none of these numbers, and no Matrix arm has a trading result yet.
- Kani proves a fifth harness, `max_qty_within_contract`.

| Part | Result | Briefs |
|---|---|---|
| `crates/ols-core` | Matrix keeps a centred tree beside the raw one: each node holds n, x̄, M_xx, ȳ, M_yx and M_yy, merged by Chan, Golub and LeVeque.<ul><li>`centred_query` solves ridge with a free intercept and a penalty ρ·n_w·σⱼ² per feature, which is ridge on features standardized inside the fold.</li><li>A cell is refused, and falls back to climatology, when the smallest eigenvalue of the scaled normal matrix is at most τ = 4p(h+p+2)ε(1+2(L+1)r). τ is a first-order bound, not a proof (`docs/DESIGN.md` §3.2).</li><li>`ridge_batch` and `kernel_batch` read many queries, windows and penalties at once, without the GIL, bit-identical whatever the thread count. Kernel readouts are exact softmax and kNN over every candidate.</li><li>Every cell has a status: OK, EMPTY, SINGULAR, NON_FINITE or TOO_FEW. After audit pass 1b, a distance that overflows to +∞ gives NON_FINITE instead of a panic.</li><li>Measured under load: 2 M pushes in 0.64 s; ridge at 0.19 s per 10⁵ queries × 3 windows × 6 penalties; one-year kernels at 7.55 s per 10⁴ queries over 262,800 candidates, about 25 minutes for the panel with the portable SSE2 build, by that estimate. In the official run, the whole arms stage, kernels included, took 1,025 s on a quiet machine. An x86-64-v3 build gives the same bits in about 14 minutes; the portable build was kept, and H3 runs at 365 days as registered, with no fallback.</li><li>192 Rust tests in the workspace, 40 binding tests in `test_matrix_batch.py`.</li></ul> | `matrix-core.md` |
| Panel | `olsprime.features.panel`: the point-in-time hourly panel of the universe members.<ul><li>1,655,881 rows in 2,273 member months of 233 pairs. The other 41 of the 274 members first join in 2025, after the development period.</li><li>1,631,784 rows are labelled, and 1,591,462 have all 15 features. Classes: down 29.4 %, neutral 43.6 %, up 27.1 %.</li><li>Features use each pair's own history. Labels have h = 24 hours, found by time. A row exists only in a member month, and a row's month is the month of its bar's open (after audit pass 1a).</li><li>The validation year and the vault are left out. Built in 12 to 41 s; fingerprint `e64b39d8…`, unchanged by the vault move.</li><li>22 tests.</li></ul> | `panel.md` |
| Adversary | `python -m olsprime.bench.adversary`: a seeded generator of random leaks, functions of the next bars' levels, ratios and shapes, each with an honest twin, run through H1's perturbation at H1's 16 cuts and at a longer schedule.<ul><li>It reports two headline rates, caught at any cut and the mean per-cut rate, each with its interval; the rate per family and per cut, with Wilson intervals; the size of each catch; and the misses.</li><li>After audit pass 1a: the attenuation curve over ε ∈ {1, 10⁻², 10⁻⁴, 10⁻⁶, 10⁻⁸, 10⁻¹⁰}, the share of twins that are constant on the data, and the families the generator never produces.</li><li>The official run, N = 10,000: the mean per-cut rate is 86.9 % [86.4, 87.3] %, and 89.0 % over the long schedule of 128 cuts. All 1,247 leaks that the protocol's by-construction list covers are caught. 98.0 % are still caught at ε = 10⁻⁶, 97.8 % at 10⁻⁸ and 2.4 % at 10⁻¹⁰. 1,166 twins (11.7 %) are constant at the compared rows, so they could not be caught. This is the detection rate of this generator, not of all leaks.</li><li>28 tests.</li></ul> | `adversary.md` |
| Physical seal | `olsprime.vault`, stage 1 run for real on 2026-10-29 after the audit's GO.<ul><li>The move took 7 s: 72 files renamed out of `data/`, and 4 Parquet tables rebuilt from the remaining archives. The vault tree outside the repository holds 80 files, 8,981,586 bytes.</li><li>`data/` keeps 843 files, and every sealed read is unchanged: 9,964,021 rows stored, the same number sealed, the panel's fingerprint unchanged, and Report n°1's `--check` reproducing every number. Only the provenance lines differ.</li><li>The data manifest digest went 230e28e8 → a9c8c088 (the body gained `vault_tree`, file entries identical) → 7c5f2397 (after the move) → 0ee7ac0f, the digest of today's runs. The last step is `deae747`'s new `vault_tree` fields. The snapshot taken just after the move, given today's `vault_tree`, has today's digest exactly, so no file changed after the move.</li><li>A journal pins each archive's and each CHECKSUM file's SHA-256, so an interrupted move resumes and a changed file stops it. A file locked by another process gives one refusal line.</li><li>Stage 2, tested on fake trees only: `encrypt` to an age recipient, `verify-decrypt` with the identity on its removable medium, which stamps the manifest, then `encrypt --remove-plaintext`, which refuses without that stamp.</li><li>54 tests in `test_vault.py` and `test_manifest.py`.</li></ul> | `seal.md`, `seal-fixes.md` |
| Matrix arms | `olsprime.models.matrix`: the eight pre-registered arms over one pooled Matrix of the panel: `climatology-panel`, `matrix-select`, `matrix-1y`, `softmax-1y`, `knn-1y`, `matrix-adaptive`, `matrix-stack` and `matrix-cusum`.<ul><li>Entries are stamped at `label_known_at` and read at decision time. The prequential record of all 24 candidates runs from 2017-11, with the σ of each row's month.</li><li>Fit-time choices read only the selection rows, purged at fit − 24 h. A selection with fewer than 1,000 rows takes the registered defaults, and is counted.</li><li>The CUSUM threshold for an ARL of 365 days is h = 4.761 (simulated ARL 365.03, SE 0.36, 10⁶ runs).</li><li>`bench.panel_invariance` runs H1 on a 10-member sub-panel, 8 cuts in both directions, with five planted online leaks. `bench.report2` is Report n°2 as one deterministic command with `--check`.</li><li>After audit pass 1c, the CUSUM's s is read with the fit's own σ, as its live days are. Report n°2 lists the day each refit leaves out, the Diebold-Mariano kernel of each comparison and its coverage in hours.</li><li>After audit pass 1d, Report n°2 also shows the 18 test months that smoke runs scored before the run apart from the other 54, as a description fixed in advance. It says so under its verdicts.</li><li>After audit pass 1e, each comparison gives its share of identical hours, where both arms' hourly log-losses are equal, and its count of informative hours.</li><li>54 tests in `test_matrix_arms.py`, `test_panel_invariance.py` and `test_report2.py`.</li></ul> | `matrix-arms.md`, `arms-fixes-1c.md` |
| Runs | `experiment-runner`, on a quiet machine, with explicit run folders `runs/2026-10-29-*`. Each run's log records its start, exit code and end.<ul><li>**Report n°1's check**, 80 s: every number reproduces after the vault move, and only provenance lines differ.</li><li>**The panel H1**, 34 min 48 s: it holds. It compares 2.7 to 14.7 million values per cut.</li><li>**The adversary**, 3 min 15 s with 8 workers, then 4 min 44 s with 4. The two runs are byte-identical, which also tests that the results do not depend on the worker count.</li><li>**Report n°2** first stopped after 17 minutes, at `3a22daa`, on its ECE cross-check: rounding of 1.3 × 10⁻⁹ relative against a fixed tolerance of 10⁻⁹. The fix, `31c8005`, sets the tolerance from the rounding bound, 4·n·ε, and changes nothing else. It passed CI, and the report ran again in 20 min 15 s. Its `--check` reproduced its three files byte for byte in 19 min 34 s.</li><li>Report n°2 scores 96.2 % of the test hours. The rest sit around exchange outages, where the features' 90-bar windows refill. The 3-bar outage of 2020-06-28 leaves 94 hours unscored in a row, from 2020-06-28T01 to 2020-07-01T22. That is 1 row whose entry bar falls in the gap, the 3 bars of the gap, and the 90-bar refill. By the label rule, 3 more hours on 2020-06-27 are unscored too, since their exit bars fall in the gap. Every arm has 0 fallbacks and 0 defaults. By point estimate, every other arm is less calibrated than climatology: ECE 0.0083 to 0.0138, against 0.0029. The intervals overlap.</li><li>Variants: 10 before this iteration, plus the 8 arms, 18 in all.</li></ul> | `runs.md` |
| `crates/risk` | GitHub's `open` run of 2026-10-28 (run 40636068905, 340 minutes per harness) proved `max_qty_within_contract` in 72.5 minutes; the other 16 open harnesses gave no verdict.<ul><li>It moved to the `proven` job, whose time limit rose from 60 to 150 minutes.</li><li>Re-run by the orchestrator in WSL2 at `-j 4`, under load from other sessions: all five VERIFICATION SUCCESSFUL in 122 minutes, `max_qty_within_contract` alone 7,318 s.</li></ul> | — |
| Audit | `leakage-auditor` (Fable), read-only.<ul><li>**Pass 1a**, on the pre-registration: three Major points. The CUSUM's registered ARL of 30 days gave about 72 false alarms over the test period, a coin flip on "fewer than 72 refits". The first fit's selections had no record to read. The CUSUM's timing let a refit act an hour before its labels were known. All three, and nine Minor points, were fixed in `PROTOCOL.md` and `HYPOTHESES.md` before any Matrix run.</li><li>**Pass 1b**, on the core, the seal and the pre-registration: eight Minor points, all fixed. Among them, kNN panicked on an overflowing distance, and `encrypt --remove-plaintext` could run without a decrypt round trip. GO for the real move, with conditions, all met.</li><li>**Pass 1c**, on the arms, the panel H1 and Report n°2's statistics: GO, no look-ahead, six Minor points, all fixed. The CUSUM's s was read with the σ of each selection row's month instead of the fit's. A cut in a month with no row decided by it flagged the honest system. The reports left out the days each refit skips, the Diebold-Mariano kernel and the coverage.</li><li>**Pass 1d**, on those fixes: GO kept. Two Minor points, both fixed: the list of contacts with the real data missed the passes over the full panel, and the smoke runs' 18 test months needed a line where the verdicts are read. Its bolder step, those months shown apart, was built before the run.</li><li>**Pass 1e**, on the split: GO kept. Two Minor points, both fixed. A part's comparison said "passes", the verdict's word, so it now says "rule met". A comparison where the arms differ at only a few hours showed an ordinary p-value, so each comparison now gives its share of identical hours. On the synthetic store, H10 had 1,404 identical hours out of 1,416.</li><li>**Pass 1f**, on those fixes: GO kept, no Minor point. Its one Nit, the count of informative hours beside the share, was folded in before the commit.</li><li>**Phase 2**, on the results and the claims: GO to record and merge. No Blocker, no Major, three Minor points and five Nits. It recomputed every number of the eight rows and every verdict from the registered rules. The previewed part of Report n°2 is bit-identical to the 18-month smoke run in every mean, difference, p-value and fit the two `report2.json` files hold.<ul><li>Minor 1: report-2.md's Coverage paragraph misstates the label rule. A gap of g bars unlabels 1 + g rows of each member, not "the 25 bars before it", and the hole around 2020-07-01 is the 2020-06-28 outage plus the 90-bar refill. `RESULTS.md` records the correction beside Report n°2's row; the run's copy stays byte-identical.</li><li>Minor 2: the ECE cross-check was said to catch a row put in another bin, which it cannot, since both paths bin the same way. The PROTOCOL change log now says what it guards, and states its rounding bound (Nit 1).</li><li>Minor 3: two of the 42 misses depend on the past alone.</li><li>Nits: H12's parts carry the generic pass rule as `rule_met: false` in the JSON, which is not H12's rule; three rows said "pass rule met", now "passes"; `diff_runs` truncates its log at 40 lines; the post-move check's output was read in the session but not saved, and the audit rebuilt its facts from the manifests.</li><li>Logged for the next change to `bench/report2.py`: the Coverage sentence, the docstring of `ece_tolerance`, H12's rule in the parts, and a per-file summary in `diff_runs`.</li><li>A second pass read this outcome, `README.md`, the pull request's description and the corrections. GO to commit and merge. It found three sentences that overstated and five Nits, all reworded before the commit. The pull request claimed a byte-for-byte reproduction for every run, and the PROTOCOL entry called its bound "exact". The H10 bullet called the test one-sided. The Nits were the 94-hour sum, the measured kernel time, ECE intervals that overlap, the README's seal wording and the constant twins.</li></ul></li></ul> | `audit.md` |

**Other results.**
- **LFM3**, read by `scout` at Abdou's request, through a summarizing fetch tool and not verified at the source. Liquid AI's model is gated short convolutions with a few attention layers, a sigmoid gate on each attention head, and a Pareto search over quality and latency. It has no test-time regression layer. What we may take is candidate ablation arms for H4: a gate per OLS head, and a conv-first hybrid with a few OLS-attention layers.
- **SMT solvers on the four dividing contracts**, tried by the iteration 2 session. Z3 5.3.2 and a repaired Bitwuzla 0.9.1 gave no verdict in 45 minutes on any of them.
- **A false-proof bug in CBMC 6.11.** Its `--bitwuzla` path writes an overflow flag in one order and reads it in the other, so it proves false overflow checks: 4 false proofs out of 8 micro-harnesses. Kani's own `#[kani::solver(bitwuzla)]` does it too. Never pass `--cbmc-args --bitwuzla` as is; Z3 and CaDiCaL are sound.
  - Upstream knew the cause: diffblue/cbmc#9144, fixed on `develop` by #9145 on 2026-10-12. The fix is not in 6.11.0, the release Kani 0.68.0 bundles.
  - With Abdou's OK, the iteration 2 session posted from Abdou's account on 2026-10-29. It commented on diffblue/cbmc#8723 to link it to #9144, and opened model-checking/kani#5096 with three harnesses that reproduce it.

**Lessons.**
- **The prior expectations were half right.** They said H2 "may well fail at daily horizons on crypto", and that each adaptive loop "may just as well chase noise", H11 being the most likely to hold.
  - H2 held, on forecast quality. Whether that skill survives costs, the prior's reason, is not tested yet.
  - The per-decision ρ did worse than the monthly choice, as if it chased noise: H10 failed, with a significant difference the other way.
  - H11 came closest without holding.
  - H12 held. It is the one loop whose false-alarm rate was fixed in advance, and also the only one tested for non-inferiority rather than for a gain.
- **A "does not hold" says nothing without the test's power.** Against kNN, matrix-1y is ahead by 0.09 %, with an interval from −0.26 % to +0.08 %. That fits "no difference" as well as "a small gain the rule cannot see". The next iteration measures the rule's size and power.
- **Show apart what was previewed.** The 18 months the smoke runs had scored show almost twice the skill of the other 54: a BSS of 0.020 against 0.011. Read alone, they would have flattered Matrix. No rule changed after the preview, so the gap may come from the period itself, 2019 to mid-2020. The split cannot tell which.
- **A tolerance is a claim about rounding.** A fixed 10⁻⁹ looked safe. Then climatology, which repeats one forecast within each fit, put 1.5 million nearly equal forecasts into the same bins, and their rounding errors added up instead of cancelling. The tolerance now comes from the bound, 4·n·ε, and is checked against the defect it guards: a lost row of climatology moves the ECE 220 to 285 times more.
- **A threshold is a claim about false alarms.** An ARL of 30 days looked conservative, until it was counted against the decision rule: 72 false alarms against a bar of 72 refits.
- **A selection needs a record before its first use.** The prequential record of every candidate now starts in 2017-11, so the first fit reads a real history.
- **Time rules need their edge cases written down.** The day that straddles a refit, and a late close at a month's last hour, each needed their own sentence.
- **A proof is only as sound as its toolchain.** A solver bug turned four false overflow checks into proofs.
- **Every contact with the data counts, not only the runs.** Smoke runs scored 18 of the 72 test months before the official run, and two passes read the whole panel to size it. The protocol lists each one, and the report shows those months apart and says so where its verdicts are read.
- **Size the workload first.** The panel's 2 M queries fit in the Rust core, so no subset of members was needed.
- **Process.**
  - Two `python3 -` processes, fed empty heredocs through the WindowsApps alias by earlier agents, each burned a full core from 01:10 on. Write scripts with a file tool, always.
  - gh's token expired mid-iteration, so nothing could be pushed and CI could not run. Commit `3a22daa` was ready at 04:34 UTC and was pushed at 08:27, after Abdou renewed the login: the official runs waited about four hours.
  - An engineer's report described a window the code does not use. The code was right; the report was checked against it.
  - Background agents' reports did not always arrive as messages. The orchestrator verified their work from the diffs.
  - `leakage-auditor` has no file-writing tool, so its phase 2 probes went through quoted heredocs, against the project's rule. It said so. Its tools should match the rule: a writing tool limited by its brief to its scratch folder.

**Bolder next step.** Learn the features, and know what a verdict is worth.
- **The encoder, roadmap phase 3.** OLS-attention heads trained from scratch on the hourly panel, on the RTX 3060 and Colab.
  - They read Matrix as their memory.
  - H4 compares them with softmax and linear attention at equal parameter count, and H5 with Matrix alone.
  - LFM3's two ideas become ablation arms: a sigmoid gate per OLS head, and a conv-first hybrid with a few OLS layers.
- **The size and power of the pass rule**, pre-registered before any run, so that "does not hold" reads as "no difference" or as "too weak to see".
  - Report n°2 saves its hourly series, with their digest, so neither study refits an arm.
  - The size: circular block sign flips of each hourly difference series, with blocks of 1, 7 and 30 days.
  - The power: a signal planted at several strengths into a panel shaped like ours.
- **H1's blind spots, measured and closed** (audit phase 2's bolder step).
  - Give each random leak a future-shuffle twin, re-evaluated on bars where every bar after s is resampled from the past. Then the misses split into three kinds: constant, no real dependence on the future, and true blind spots.
  - Add an extreme-shape cut to H1, where bar t+1 takes a fill and a close location of 0.99 and 0.01 in the two runs. Planted controls such as `fill[s+1] > 0.95` are predicted `........` under today's design and `XXXXXXXX` under the new cut.
- **From forecasts to trades.** A portfolio simulator turns the Matrix arms' probabilities into long/flat positions through the risk kernel, at Coinbase's and Kraken's fees. Does a BSS of 0.013 survive a round trip of about 1 %?

**Learning page.** [L'arbre et le coffre](https://claude.ai/artifact/8AkKGG3q97u8jU2Z9BjpbK), in French, written by `teacher` and checked by the orchestrator. It is private to Abdou's account.

## Iteration 4 — Learned features, and what a verdict is worth (2026-10-30)

**Goal.** This is roadmap phase 3, the OLS Prime encoder, made bolder as iteration 3's outcome proposes: learn the features, and know what a verdict is worth.
- The encoder: OLS-attention heads trained from scratch on the hourly panel, reading Matrix as their memory.
  - H4 compares them with softmax and linear attention at equal parameter count.
  - H5 compares them with Matrix alone.
  - H13 asks whether they are also cheaper to run.
- The size and power of the pass rule, so that "does not hold" reads as "no difference" or as "too weak to see".
- An instrument, `olsprime.explain`, that says why a forecast is what it is.
- A bolder check of H1: two cheaters that leak at training time must be caught on GPU, where bits are harder to keep.

**Deliverables.**
1. **Report n°2's series.** `series.npz` holds its hourly series and scored rows. Its SHA-256 goes in `report2.json`, so that no later study refits an arm.
   - Audit phase 2's four logged items are folded in.
   - Report n°2 is re-run once and compared with iteration 3's run. Only named lines may differ.
2. **The size and power of the pass rule.** `python -m olsprime.bench.power` reads `series.npz` by its digest and refits nothing.
3. **`olsprime.explain`.**
   - `why` breaks a Matrix forecast into the memory's tree nodes and into per-feature parts, with two loud checks. For it, `CentredReadout` gains `beta` and `mean_y`.
   - `head` shows the OLS heads' W, once the encoder exists.
   - Every call is logged in `runs/explain/CONTACTS.log`. Its first use is the five whys below.
4. **The encoder.**
   - torch is added to the lock without moving a locked package: CPU on Windows and CI, CUDA in WSL2.
   - The arms `enc-ols`, `enc-softmax` and `enc-linear`, with the memory over learned keys, training in weekly episodes, yearly fits and three seeds.
   - Report n°3: H4, H5, H13 and the cost table.
   - The panel H1 for the encoder arms on GPU, with `cheat-stamp` and `cheat-value`. After audits 4a and 4b, it also has two purge cuts with `cheat-purge` and `cheat-slice`, and a positive control, `pos-window`.
   - LFM3's ablations and H6, if time remains.
5. **H1's blind spots.** The adversary's future-shuffle twins classify its misses.
6. **Registered now, run in iteration 5:** H14, the portfolio simulator's first claim, and H1's extreme-shape cut.

**Scope, fixed before any result.** `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)", holds the full pre-registration.
- **Priority if time runs out**, in the protocol's order:
  1. the audited pre-registration;
  2. Report n°2's series, then the size and power study;
  3. `explain`;
  4. the encoder's core arms with H5, H13 and the cost table, then LFM3's ablations and H6;
  5. the twins.

  A hypothesis that was not run is reported as not run.
- **The first finding, if it happens.** If the deterministic H1 cannot catch `cheat-stamp` bit for bit on GPU, that is reported before any H4 number.
- **Sizing, before the schedule.**
  - The machine: the RTX 3060 12 GB in WSL2.
  - Colab: 194.04 compute units on 2026-10-29, used only if the sizing rules the 3060 out.
  - The time of one fit of `enc-ols` on the 2024 fit's training set, measured on a synthetic panel of the real shape in wave 1. The numbers, and the schedule they allow, are written here before any encoder run.
  - **Measured on 2026-10-30, synthetic data only.**
    - The panel: 30 members, hourly from 2017-10. The 2024 fit trains on 1,558,575 rows over 323 weeks, an upper bound, since real members list later.
    - The build: RTX 3060 12 GB in WSL2, driver 617.42, CUDA 13.4, cuDNN 9.26.0, torch 2.15.0+cu134, with the deterministic settings on.

    | | `enc-ols` | `enc-softmax` | `enc-linear` |
    |---|---|---|---|
    | Training step, median | 0.471 s | 0.442 s | 0.300 s |
    | Peak memory in training | 1.85 GB | 2.68 GB | 2.09 GB |
    | Time per 10⁴ decisions, H13's conditions | 0.142 s | 0.134 s | 0.093 s |

    - One whole 2024 fit of `enc-ols` took 945 s: 6 epochs, 1,938 steps. Forecasting its 253,193 rows took 5.1 s.
    - **Determinism.** Two such fits were bit-identical: every epoch's parameters, the slice losses, the epoch kept and every forecast. The scope test passed at every prefix. So the registered H1 runs on the GPU. Its own pre-check still runs first, on the real sub-panel.
    - **Two signs, before any real run.**
      - On these synthetic rows, `enc-ols` takes 1.06 times `enc-softmax`'s time. H13 needs 0.9 or less. Its registered measurement is the one on the real rows.
      - On a small synthetic panel, `pos-window` with `enc-ols` reached a Brier skill score of 0.53, against the 0.5 it needs, and kept its last epoch.
  - **Projections.**
    - H4 and H5's 54 fits (3 arms × 3 seeds × 6 fits) take about 7.4 GPU-hours.
    - The panel H1, with its pre-check and every control, takes about 8.7 hours. That is under the 12-hour budget, so every cut runs.
    - Each of LFM3's ablations and H6 takes about 2.9 GPU-hours.
  - **The schedule.** Everything runs on the 3060, so Colab is not needed. In order:
    1. the pre-check on the real sub-panel;
    2. H1 with its controls;
    3. the 54 fits;
    4. H13's timing and the cost table;
    5. then, while time remains, LFM3's ablations and H6.

**The five whys**, written before the first call of `explain`.
- Every call on 2019–2024 comes after the frozen design is committed.
- What a call shows on those months is exploratory: no hypothesis of this iteration is tested on it.

1. Why do `matrix-1y` and `knn-1y` tie?
2. Does the Brier skill of the 18 previewed months come from a few hours, or from the whole 2019–2020 regime?
3. Are hours with a high negative mass worse calibrated?
4. Which feature carries the gain over climatology?
5. What did the model do around the 2020-06-28 outage?

(1), (2) and (5) get numbers in the outcome. (3) and (4) are answered only once the encoder is frozen, and are labelled "exploratory, test months".

**Hypotheses touched.** H1 (the encoder arms on GPU, the training-time cheaters, the twins), H4, H5 and H13; H6 if time remains. H14 is registered.

**Done when.**
- The pre-registration was audited before any run, and every finding was fixed or logged.
- Every check of `CLAUDE.md` is green, locally and in GitHub Actions, on Windows and Linux.
- Every change to data, model or evaluation code has passed `leakage-auditor`.
- `RESULTS.md` has the new rows with their variant counts. Every item not run is reported as not run.
- The outcome and one bolder next step are recorded here, with the learning page's link.
- The pull request is merged into `main` with a local merge commit and pushed. Abdou authorized commit, merge, push and installs for this iteration on 2026-10-29.

**Plan.** Briefs are in `.claude/briefs/iteration-4/`. At most three agents run at once.

| Wave | Task | Agent | Brief |
|---|---|---|---|
| 1 | `beta` and `mean_y` on `CentredReadout` | `rust-engineer` | `beta-getter.md` |
| 1 | Report n°2's series; the size and power study | `ml-engineer` | `series-power.md` |
| 1 | The encoder's core, torch and the sizing | `ml-engineer` | `encoder-core.md` |
| 1 | Audit pass 4a, on the pre-registration | `leakage-auditor` | `audit-prereg.md` |
| 2 | `olsprime.explain` | `ml-engineer` | `explain.md` |
| 2 | The future-shuffle twins | `ml-engineer` | `twins.md` |
| 2 | Audit pass 4c, on wave 1's evaluation code and the twins | `leakage-auditor` | `audit-code-4c.md` |
| 2 | Audit pass 4d, on `olsprime.explain`, in three looks | `leakage-auditor` | `audit-code-4d.md` |
| 2 | The encoder's panel H1 on GPU, with its controls | `ml-engineer` | `encoder-h1.md` |
| 2 | Report n°3: the fits, H4, H5, H13 and the cost table | `ml-engineer` | `encoder-report3.md` |
| 2 | Audit pass 4e, on the encoder's core and its panel H1 | `leakage-auditor` | `audit-code-4e.md` |
| 2 | Audit pass 4f, on Report n°3's runner | `leakage-auditor` | `audit-code-4f.md` |
| 3 | `explain head`, on Report n°3's checkpoints | `ml-engineer` | `explain-head.md` |
| 3 | Audit pass 4g, on `explain head` | `leakage-auditor` | `audit-code-4g.md` |
| 3 | Official runs and rows | orchestrator, `experiment-runner` | `runs.md` |
| 3 | The `explain` twin's failure at one row: a diagnosis, no code change | `ml-engineer` | `twin-gap.md` |
| 3 | LFM3's ablations and H6, built while the GPU runs | `ml-engineer` | `ablations.md` |
| 3 | Audit pass 4h, on the ablations | `leakage-auditor` | `audit-code-4h.md` |
| 3 | Audit pass 4h's fixes, and the seed-split control beside H6 | `ml-engineer` | `ablations-fixes-4h.md` |
| 3 | H13's gap on synthetic data, then the rows of `docs/RESULTS.md` | `experiment-runner` | `h13-gap-and-rows.md` |
| 4 | Audit of the code and the results | `leakage-auditor` | `audit.md` |
| 5 | Learning page | `teacher` | `learning-page.md` |

**Set up by the orchestrator before wave 1.**
- `main` holds iteration 3 (merge commit `6e05906`, pull request abdou-rgd/ols-prime#3).
- Work happens on the branch `iteration-4`. It was made from `origin/main` without upstream tracking, so that no push can land on `main` by mistake.
- The Colab CLI, `google-colab-cli`, is installed with `uv tool` in WSL2. Abdou signed in on 2026-10-29.
- The machine stays awake, and shuts down at the end of the session unless something blocks.

**Before wave 2** (after audit pass 4a). The audited pre-registration is committed, and wave 2 starts only after that commit.

**Outcome (2026-10-31).** Done, with LFM3's ablations and H6 not run, and H13 without a verdict, its measurement invalid.
- Every check is green, locally and in GitHub Actions, on Windows and Linux (pull request abdou-rgd/ols-prime#4).
  - On GitHub, all 7 jobs passed for `5f17a76`, the commit of the official runs (run 41019531179), and for `da0433e`, the last code commit of this branch (run 41035843930). Later commits change only documents and briefs.
  - Locally, on Windows at `67cd8f3` with the record's documents in progress: rustfmt, both clippy runs and cargo test exit 0, and pytest gives 1,498 passed, 4 skipped. In WSL2 at `5f17a76`: 1,480 passed, 2 skipped.
- The audit took eight passes on the code before any result, 4a to 4h. Phase 2, on the results and the claims, gives GO to record and merge, with its Major point fixed before the commit (see the Audit row).
- **Contacts with the real data:** 17 entries in the protocol's change log; `runs/explain/CONTACTS.log` holds 24 lines.
- **The first finding, the GPU H1: it holds.** The pre-check passed: each arm, trained twice on the RTX 3060 at each of the four pre-check years, gave the same bits. `enc-ols` 0 differing values of 5,232,329 (2020), 7,796,766 (2021), 7,630,828 (2022) and 10,004,070 (2024); `enc-softmax` and `enc-linear` 0 of 5,232,281, 7,796,718, 7,630,780 and 10,004,022. The registered scope test, at the first cut, moved 0 of 1,758,980 values for each arm; at the eight regular cuts, as a description, 0 too.
  - **H1: pass.** The three honest arms have 0 violations at 8 cuts and 2 purge cuts. Every control is caught on its predicted pattern, in each direction: `cheat-stamp`, `cheat-value` and `cheat-stamp-day` at the 8 cuts and neither purge cut, `cheat-purge` at both purge cuts, and `cheat-slice` at F − 30 days alone.
  - **The skill check: FAIL, on `cheat-stamp` alone.** Its Brier skill score is 0.0064 [0.0029, 0.0100], against the 0.1 it needs and `enc-ols`'s 0.0052 [0.0018, 0.0089] on the same rows. `cheat-value` reaches 0.9997 [0.9994, 1.0000] and `cheat-stamp-day` 0.4376 [0.4244, 0.4515]. So H1 caught a stamp leak that gives almost no skill, bit for bit, and one that gives much. A likely reading, not tested: in a 30-day memory, the last day's entries are about a thirtieth of what the readout sees, and `cheat-stamp-day` keeps only those.
  - **`pos-window`: pass.** 0.6825 [0.6428, 0.7209], against the 0.5 it needs: the training loop learns a signal that is there.
  - Exit code 3 is the code for "H1 holds and a skill check fails". The chain's log line "the H1 did not pass" is its own wording.
  - The honest runs' forecasts are a first look at the encoder's skill on the sub-panel's cut years, listed as a contact: `enc-ols` 0.0052, `enc-softmax` −0.0004 [−0.0034, 0.0025], `enc-linear` −0.0024 [−0.0052, 0.0002].
- **Report n°3: the encoder learns, at about Matrix's level, on 54 fits of 21,003 parameters.** Over 50,626 test hours, 2019-01 → 2024-12, the Brier skill scores are 0.0110 [0.0087, 0.0133] for `enc-ols`, 0.0079 [0.0059, 0.0098] for `enc-softmax`, 0.0030 [0.0016, 0.0044] for `enc-linear`, and 0.0130 [0.0108, 0.0156] for `matrix-select`.
- **H4 holds, as non-inferiority.** `enc-ols` − `enc-softmax` is −0.00269, or −0.25 % of climatology's log-loss. Its one-sided upper bound, −0.00182, and the bootstrap's 95th percentile, −0.00181, lie below the margin, +0.00538. It is testable: both arms have a skill interval above 0. The difference favours `enc-ols` (p 3 × 10⁻⁷), in each seed pair too, but the registered test is non-inferiority: superiority is a description.
- **H5 does not hold.** `enc-ols` − `matrix-select` is +0.00082, or +0.08 %, with p 0.52 and an interval of [−0.14 %, +0.29 %] of climatology's log-loss. No difference is shown in either direction. The power study covers Report n°2's comparisons, not this one, so the rule's detectable size for H5 is not measured. It compares two systems: the encoder refits yearly, `matrix-select` monthly.
- **H13: no verdict, the measurement is invalid.** The rule's arithmetic gives 0.4031 ≤ 0.9, but the number measures the timing stage's own GPU cache, not the arms. The diagnosis was triggered by the gap with the 1.06 on record before the run, not by the sign of the result.
  - The stage keys the memory over the panel before timing, and leaves torch's cache at 12,926 MiB reserved on a card of 12,288 MiB. Synthetic rows, with a far smaller panel, had given 1.06.
  - On the same rows, the ratio is 0.3739 as the code stands, and **1.0732** with the cache emptied before each protocol call, above the bound. Emptied, the three operators fall to their synthetic times.
  - That repeat is a description, not a registered measurement, and the official stage logged no memory state: its cache is inferred from identical code and data and from repeat 1 reproducing its numbers. Over repeats 5 to 10, past the allocator's refill, the ratio is 1.05. OLS attention is not shown to be cheaper to run than softmax attention, on this kernel. The GPU wall times of Report n°3's cost table carry the same artefact.
- **What a verdict is worth.** The pass rule keeps its size: 2.2 % to 6.4 % two-sided, none above 7.5 %. Its smallest δ with 80 % power is 0.23 % to 0.25 % of climatology for H3 against kNN (0.26 % when concentrated), so H3's −0.094 %, at 19 % to 24 % power, is too weak to see, not evidence of no difference. H11's −0.056 % has 42 % to 51 % power. The second run, for byte identity, stopped at start-up and is not re-run.
- **H1's blind spots.** 205 misses: constant 193, unchanged 5, blind spot 7; caught leaks moved by a twin 98.3 % [98.0, 98.5]; honest twins 0.
- **Report n°2, twice and on two platforms.** The Windows re-run equals iteration 3's run but for five allowed lines. The WSL2 re-run, Report n°3's base, differs on the platform's last bits only, by at most 5.2 × 10⁻¹⁴ relative in `report2.json`: every verdict and rule is the same.
- **The five whys** (exploratory, test months; numbers for 1, 2 and 5):
  1. They tie because large hourly wins and losses nearly cancel. Over 50,626 hours the mean difference is −0.00101, and no hour is identical. The gross of the hourly differences is 39 times the net: the hours where `matrix-1y` wins sum to about 20 times the net, those where it loses to about −19 times. Its worst months are 2020-03 and 2022-05, the COVID crash and the LUNA collapse. At BTCUSDT's rows of 2020-03-12T15:00 and 18:00, the two hours of largest d (+3.17 and +3.29), the ridge readout extrapolates. 53 to 59 % of its weight mass is negative, Σ|w| is 5.6 to 8.0, the effective sample size is 1,430 to 3,255 against 41,259 at a calm hour, and the leverage is 17 to 37 times the calm hour's. So it puts 0.81 to 0.84 on "neutral" and 0.001 to 0.013 on "down", at the floor at 15:00. `knn-1y`, an average of its 1,000 nearest entries, stays at 0.61 to 0.66 and 0.06 to 0.11. The hourly log-loss over the 30 rows is 5.13 and 6.05 for `matrix-1y`, against 1.96 and 2.77. The same extrapolation also wins: at 2020-03-14T07:00 it gave 0.918 to "neutral" against kNN's 0.558, and won by 0.51. At the calm hour, 2024-04-12T23:00, where it also won by 0.52, its readout is close to an average. In short, signed weights gain a little in ordinary hours and lose a lot in the few hours where the query lies far outside the memory; the floor caps each loss at −ln 0.001 ≈ 6.9.
  2. The previewed months' higher skill is the whole 2019 – mid-2020 regime, not a few hours. All 18 months gain (median monthly BSS 0.019); without March 2020, the largest month, the BSS is still 0.0184, against 0.0108 on the other 54 months, of which 49 gain (median 0.011). Within months, half the net gain comes from 3.3 % and 2.2 % of the hours, crash hours of both signs (2020-03-12/14, 2021-05-19/20, 2022-05-11/12): the hourly gains mostly cancel (gross / |net| 3.1 and 4.9).
  3. No, not in this sample (exploratory, test months). Over 1,000 seeded rows of `matrix-select`, cut into deciles of 100 by negative mass, the mean log-loss is 1.07 to 1.11 in the first nine deciles and 0.93 in the tenth, whose negative mass runs from 1.07 to 8.7. The calibration gap, mean top-label confidence minus hit rate, ranges from −0.04 to +0.10 with no trend: with 100 rows a decile, its standard error is about 0.05. The first run of this call stopped on the twin's loud check at one row; the re-run, whole and at one BLAS thread, passed every check on all 1,000 rows (twin's largest gap 4 × 10⁻⁶ of its bound).
  4. The 30-hour volatility, mostly through the "neutral" readout (exploratory, test months). On the hourly panel the feature windows count hours. Per row, each feature's parts sum to zero over the three classes, so its part on the realized class says whether it moved probability toward what happened. Over the 1,000 rows, all features together moved +0.0098 toward the realized class (positive in 57.6 % of rows). `f_vol_30` alone moved +0.0122 ± 0.0030 (naive standard error), and its mean |part| on "neutral", 0.072, is the largest of any feature and class. The 90-hour moving-average gap (`f_ma_gap_90`) and the 7-hour volatility (`f_vol_7`) add about +0.001 each. The 7-hour range (`f_range_7`) takes back −0.0042 ± 0.0018, and the 90-hour return (`f_ret_90`) −0.0011 ± 0.0005. The other features stay within about one standard error of 0.
  5. Nothing special: `matrix-select` kept the same cell (window all, ρ 0.01, every cell OK, no fallback) and forecast from a memory the outage had frozen. At 2020-07-01T23, its newest entry had been known since 2020-06-29T01:00, 70 hours earlier; from the last row before the outage (2020-06-27T21, 438,481 entries) only 750 entries joined, where about 30 an hour join normally. Against 439,231 entries, the missing ones barely move a forecast: BTCUSDT's neutral probability ranged from 0.345 to 0.390 over the four hours of 2020-06-27 and was 0.367 after; ETHUSDT's ranged from 0.318 to 0.383, then 0.362. The July fit read the same memory (σ and climatology from 439,231 entries known by 2020-06-30T00:00). Every scored forecast equals `series.npz` to the bit; every stamp check passed; the twin's gaps were about 2 × 10⁻¹⁵.
- **LFM3's ablations and H6: not run.** They are built, and audit pass 4h gave GO with fixes. The fixes are unverified, and no ablation was fitted on real data. The code is on the branch `iteration-5-ablations` at `5be70a3`, not merged. `explain head` is built and audited, and was never called. The seed-split control beside H6 is registered and not run.
- **Registered, run in iteration 5:** H14 and H1's extreme-shape cut.
- Kani: the five proven harnesses passed again locally, in WSL2 with `-j 2`, at `c2e6381` (risk crate unchanged since `9c414ca`): `max_qty_within_contract` 5,313 s; end 04:59:32 UTC.
- These are forecasts, not trades. Costs are in none of these numbers.

| Part | Result | Briefs |
|---|---|---|
| Report n°2's series | `series.npz` holds the hourly series and the scored rows, its SHA-256 in `report2.json`, so that no later study refits an arm. Windows `03916fda…`; WSL2 `691e7ea7…`, the base of Report n°3. Each run's `--check` reproduces its four files. | `series-power.md` |
| Size and power | `python -m olsprime.bench.power`: 244,000 replicates in 5,198 s with 8 workers and no pool restart, on the Windows series. | `series-power.md` |
| `olsprime.explain` | `why`, `series`, `sample` and `head`, every call logged. Each `why` closes on two loud checks, and a Matrix-free twin recomputes the forecast. The twin failed once at one row of the first `sample`. A diagnosis then held the core exact to within 4.6 × 10⁻¹⁷ of an exact rational reference, so no scored forecast is affected; the failure's mechanism is not identified, and later calls run at one BLAS thread. | `explain.md`, `explain-head.md`, `twin-gap.md` |
| Encoder | `enc-ols`, 21,003 parameters; `enc-softmax` and `enc-linear`, 20,995. Training in weekly episodes with causal memory, yearly fits, three seeds. | `encoder-core.md` |
| Panel H1 for the encoder | 170 fits, 220,920 steps, from 2026-10-30 07:38:33 UTC. It reached its 16-hour timeout and resumed once with the same folder, to 00:30:02 UTC. Run `runs/2026-10-30-encoder-h1/`. | `encoder-h1.md` |
| Report n°3 | 54 fits in 12 h 51 min, beside the H1 on one GPU, so the 12.69 training GPU-hours are upper bounds. `--check`: 4 files identical, `run.json` excluded, after a first refusal from a script error. Run `runs/2026-10-30-report3/`, [docs/reports/report-3.md](reports/report-3.md). | `encoder-report3.md` |
| H13's gap | On synthetic rows, `experiment-runner` ruled out a cold start, tensor values and the size of the bars table. The orchestrator's two repeats on real rows found the cause. Run `runs/2026-10-31-h13-repeat/`. | `h13-gap-and-rows.md` |
| Twins | `python -m olsprime.bench.twins`, two runs, byte-identical. | `twins.md` |
| Ablations | `enc-ols-gate` 23,051 parameters, `enc-ols-conv` 20,007, `enc-ols-brier` 21,003, and `bench/ablations.py`. Not fitted on real data. | `ablations.md`, `ablations-fixes-4h.md` |
| Runs | From clean clones at `221591c` and `5f17a76`. Colab: 0 units; the RTX 3060 carried every fit. Variants: 18 before this iteration, plus 3 encoder arms, 21. | `runs.md` |
| Audit | 4a: five Major (vacuous H4; batch-size bits on GPU; fallback under one draw of noise; H14 confounded with drift; explain as a channel into the design), nine Minor, seven Nits. 4b: GO, four Minor, five Nits (H14's random positions, second purge cut and cheat-slice, the CPU budget, pos-window's stamp). 4c (series, power, twins, getters): three Minor, three Nits; second look one Minor, three Nits (one BLAS thread per pool worker; pool rebuilt on WinError 6; refusal above 10⁻⁹). 4d (explain), three looks: two Major (a residual zero by construction; a stale extension), seven Minor, six Nits; the Matrix-free twin. 4e (encoder core, panel H1): three Minor, four Nits; second look one Minor, five Nits (resume refused on a changed build; `cumsum` on CUDA probed). 4f (Report n°3): one Major (a copied part scored as another arm), seven Minor, five Nits; second look three Minor, three Nits (tree frozen at the fits' commit). 4g (explain head): one Major (head bound only to its own folder), two Minor, two Nits; second look three Nits. 4h (ablations, H6): one Major (a fitted arm reported as not run), six Minor (five test gaps; a reader that could not load the Brier arm), six Nits; the seed-split control. Phase 2, on the results and the claims: GO to record and merge. No Blocker, one Major, two Minor, three Nits. The Major: H13's Verdict column recorded a verdict the project itself said was not measured, so it now reads no verdict. The set-aside itself was judged justified, because the trigger was a reference on record before the run. The Minor points: the evidence row now gives the card's state in each repeat and repeat 2's transient, and the re-measure's status is defined. Two Nits fixed; the third, a detectable size for H5 from its standard error, is optional and left out. It recomputed H4 and H5, and read the panel H1's verdicts, the twins' counts and the power study's alerts from their JSON. It did not check the five whys, the coverage of `compare_platforms.py`, the `--check` logs, the fits table, the δ grid beyond its alerts or the GPU-hours against the per-fit times | `audit-prereg.md`, `audit-code-4c.md` … `audit-code-4h.md`, `audit.md` |

**Lessons.**
- **A surprising number gets a diagnosis before a record.** The rule met, the ratio stable within 3.4 % over ten repeats, and the wrong story fitted: a cold GPU, then a slow memory clock. A number that moved from 1.06 to 0.40 on the same code was the only alarm, and two repeats settled it.
- **A timing is a claim about the machine's state, and the instrument is part of the machine.** The protocol fixed every shape, setting and repeat, and did not fix what the process held in memory. The stage's own preparation set the result.
- **Stability is not validity.** Ten repeats within 3.4 % showed that the measurement was repeatable, not that it measured the arms.
- **A control can fail its skill check and still do its job.** `cheat-stamp` was caught bit for bit and gives no useful skill. `cheat-stamp-day` was added before any run to separate the two questions, and each verdict stayed apart.
- A bit is a property of a build: `np.log` and `np.exp` differ in their last bits between Windows and Linux, so the same commit gives two panels.
- Documentation drifts: torch's list of deterministic CUDA ops moved `cumsum` between releases. Check the claim on the machine.
- An `explain` call refused before contact costs nothing: `series --top 100` was refused (range 1..50) and logged nothing.
- **Process.**
  - Chaining the long runs into one script kept the GPU working while the weekly limit waited. The script's error on `--check` and its wording on exit 3 were both read and corrected, not taken at face value.
  - A pipeline's last exit code hides the first: `wsl ...; tail ...` reported 0 when the timing never ran, and the "repeat" then read as identical to the official run to the last bit. That identity was the tell.
  - Git Bash rewrites `/mnt/c/...` before WSL sees it. Launch WSL scripts from PowerShell, in a login shell.

**Bolder next step.** Put a price on the skill, on an instrument that cannot fool itself.
- **H14, the first trades.** The long/flat portfolio of `matrix-select` at Coinbase's and Kraken's taker fees, against always-flat and against random chains matched on exposure and entries. Does a Brier skill score of 0.013 survive a round trip of about 1 %?
- **The timing stage rebuilt.** It empties the cache, and records the reserved memory, the card's total and the clocks beside every ratio. A placebo, a fixed reference kernel timed before and after each protocol call, refuses to write a ratio when the reference drifts: that would have refused 0.4031 at 00:36 UTC, with nobody awake. Then a fused OLS-head kernel as a new registered variant: can OLS attention earn H13 on a kernel worth the name?
- **The deferred work.** The second look at audit 4h's fixes, the ablations' 54 fits, H6 and its seed-split control, the first `explain head` calls and H1's extreme-shape cut.

**Learning page.** « Le nombre qui mentait », in French, written by `teacher` and checked by the orchestrator. At Abdou's choice it stays a local HTML page, `C:\Users\abdou\ols-prime-pages\iteration-4\iteration-4.html`, read in a browser, and is not published as an Artifact, so this iteration has no link.
