# Brief — finish the iteration 1 learning page (teacher)

**Role.** `teacher`. Your first run. **The page:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\explainer\comprendre-ols-prime.html`. Edit it in place; keep a backup copy beside it first. It lives outside the repository.

A general-purpose agent wrote this page before the `teacher` role existed (brief: `learning-page.md`). The orchestrator has since corrected it, for example the Sharpe standard error in §7. Its design, demos and typography stay. Your job is to bring it to the iteration's final state. It still says that the audit and Kani are running and that nothing is committed. Since then the audit ran three passes, the findings were fixed, claims were checked, and the vault was sealed and hardened.

## What to change

1. **§12, the deliverables table and its "À retenir".** Use the final states and counts below. The `#kani` and `#audit` placeholders become real sections.
2. **`#audit`: teach the audit.** Cover the verdicts of the three passes, each finding and what became of it, and how mutants and probes work.
   - **The centrepiece: the singularity criterion.** Give the intuition: a Cholesky factor that merely exists proves nothing. Derive it from the facts below, add a worked example with the real numbers, and add a small demo if it fits.
     - Example demo: a 2 × 2 design [intercept, x], with sliders for r = |mean|/std and N. It shows λ_min(S) ≈ 1/(2r²) against τ, and where the solve is refused.
   - **The dust floor**, pinned from both sides.
   - **The foreign readout.**
3. **`#kani`: teach what Kani proved and why the rest is slow.**
   - Bit-blasting.
   - Why CDCL SAT solvers choke on 64-bit products and quotients.
   - Why `kani::assume` on magnitudes does not shrink the circuit.
   - The CI split.
   - The results table arrives by message from the orchestrator when the runs end. Write everything else now, and leave a clearly marked spot for the table.
4. **§7, the vault.** It is sealed, and now hardened:
   - the guard rules;
   - the cheater snippets that measure the guard's coverage;
   - the OPEN format, and the audit trail that git tracks.
5. **§8:** the backtest crate now has 59 tests, and the dust floor is pinned.
6. **§10 and "Ce que l'itération a appris":** the scout's claim checks and the lessons below.
7. **§13 "La suite":** the iteration 2 additions and the bolder next step below.
8. **§14 and §15:** glossary terms for the new concepts, and two or three new questions.
9. **The status line at the top and the footer.** The first commit happens after you finish. Write `Premier commit : ⟨hash⟩` in the status line and the footer. The orchestrator fills in the hash.

Keep:
- the narrow no-break space (U+202F) before French colons and inside « », as the page does;
- no control characters, and every tag closed;
- MathJax `\(...\)` with backslashes intact. Write files with your editor tools, never through a shell heredoc, which ate backslashes once already.

Run the checks of your workflow: `node --check` on each inline script, every tag closed, every URL answers 200, and a read at 400 px. Then report. Do not publish.

## Facts (verified by the orchestrator unless marked)

**Audit, pass 1** (`leakage-auditor`, Fable; brief `audit.md`). Verdict: pass with fixes. No result was produced, so nothing was invalidated.
- **Major.**
  - **M1, a vault lock was needed.** The Parquet files hold the vault months: 6,552 rows per 1 h file and 273 per 1 d file. `universe --end` defaulted to the current month.
  - **M2, the dust floor was untested.** Mutant B9 (`DUST_FRACTION` raised from 1e-12 to 1e-3) survived the whole suite.
- **Minor.**
  - A singular solve could succeed silently. Probe P4: two identical entries at λ = 0 gave `Ok(1.5)` for outcomes all equal to 1.0. Probe P2, near-collinear at λ = 0: Σw = 0.999999999345.
  - A Readout was accepted by a foreign Matrix (P10).
  - Future-incompat warnings in `crates/risk/src/proofs.rs` (`assert!` in match arms).
  - Sharpe has no interval: logged for iteration 2.
  - The DATA.md "same gaps" wording: the ETH outage ends 11 ms later.
  - Which timestamp Matrix's `time` is, for bars.
- **Clean.**
  - Mutants: 13 of 13 look-ahead and cost mutants caught in backtest (B1–B8, B10–B14), and B15 is equivalent. 10 of 10 in risk (R1–R10).
  - Data: 440 archives, every SHA-256 matches, and every DATA.md count was verified on disk.
  - CI: every action pin was verified.

**Fixes.**
- **The vault** (`data-engineer`; briefs `vault-lock.md` and `vault-guard-fixes.md`; `python/olsprime/protocol.py`):
  - `load_klines` seals on `known_at = max(close_time, open_time + interval − 1 ms)`. BTCUSDT 1 h goes from 79,837 to 73,285 bars, the last opening on 2025-12-31 23:00; 1 d goes from 3,332 to 3,059.
  - Two keys: the env `OLSPRIME_OPEN_VAULT=1`, and `runs/vault/OPEN`, whose first line must be an ISO date. Otherwise `VaultSealed` is raised, so a bare `touch` never opens it.
  - Each open read leaves a line in `ACCESS.log`. Git tracks `OPEN` and `ACCESS.log`, and nothing else under `runs/`.
  - `universe` defaults to `--end 2025-12` and refuses vault months.
  - `Period` accepts only true UTC; Europe/London in winter is refused.
  - `seal` also cross-checks `open_time`, which defeats a forged `known_at`.
  - The AST guard walks every `.py` and `.ipynb` under `python/`, tests included. It has rule ids and a per-file allow-list.
  - The bolder check, from the auditor: the guard's coverage is measured, not assumed. Cheater snippets must each be flagged and controls must pass. All of them are flagged (32 in the first round, then eval/exec, subprocess, pickle and np.load, shutil, notebook magics), and every control passes.
  - A test forbids parquet, arrow and csv crates in `Cargo.lock`.
  - Still uncaught, by design of a static filter: indirection through a variable, computed paths, `torch.load`.
- **M2** (`rust-engineer`; brief `audit-fixes-rust.md`):
  - three tests pin the floor inside [5e-13, 2e-12) of equity;
  - a full exit is never dust, down to a position worth 1e-296 of equity;
  - B9 now fails `rebalances_above_the_dust_floor_fill`. The orchestrator re-applied it in a scratch copy to check.
- **The singularity criterion** (`crates/ols-core/src/ridge.rs`, `RidgeConfig` docs). Accept the solve only if `λ_min(S) > τ = 4p(N+p+2)ε`, where:
  - S = D^-½ M D^-½, M = G + Λ and D = diag(M);
  - N is the number of summed entries (`SuffStats::count`), and ε = 2^-52.

  Why it holds:
  - **Rounding of G.** Each entry of G is a sum of N products. After scaling, each entry of S errs by at most about (2N+9)ε, to first order. So ‖E‖₂ ≤ ‖E‖_F ≤ p(2N+9)ε.
  - **The eigensolver.** It is backward stable, with an error of at most c·p·ε for c ≤ 4p (a textbook bound, not a proof).
  - **Weyl's inequality.** For an exactly singular design λ_min = 0, so the computed value is at most p(2N+9)ε + 4p²ε. That is at most τ, because τ minus that sum equals p(2N−1)ε ≥ 0.
  - **Scaling.** It makes the test unit-free (van der Sluis): dollars or millions of dollars give the same verdict.

  Why not a pivot threshold:
  - the computed pivot of a dependent column errs by up to (1 + ‖M_<k⁻¹ m_k‖²)·‖ΔM‖;
  - the rust-engineer found that the auditor's pivot test `p·ε·max diag` accepts three copies of (−0.0075, 0.90) at λ = 0, a rank-1 design.

  What it means in practice:
  - The cost is one symmetric eigendecomposition, O(p³): 40 to 120 µs per solve at p = 33 in release builds (auditor's probe).
  - Every output is checked finite. λ ∈ {0, 1e-12} was added to every identity generator, with an SVD cross-check.
  - Features must be centred inside the fold. With r = |mean|/std beside the intercept, λ_min ≈ 1/(2r²). At λ = 0 and p = 33, the solve is refused from r ≈ 1.3·10⁴ when N = 10⁵, and from r ≈ 4·10³ when N = 10⁶.
  - Auditor's probe: standardized iid features at p = 32 are accepted at N = 10⁵ and 10⁶. A pair with correlation 1 − 1e-8 is refused at N = 10⁶ for λ ≤ 1e-6. A pair at 1 − 1e-10 is refused until λ = 1e-2.
  - A penalty lifts the penalized block by at least λ / max Dᵢᵢ. The free intercept is lifted only through its correlation with the rest.
  - In progress when you start: negative weights are refused, and a deterministic acceptance test (p = 32, N = 10⁵, λ = 0) is added. Read `ridge.rs` again before you finish.
- **The foreign readout.**
  - Each Matrix instance gets an id from an atomic counter, and a clone gets a fresh one.
  - `Readout` fields are private.
  - A readout from another instance fails with `OlsError::ForeignReadout`.
- **known_at in Matrix.** It is documented, with a doctest on the real truncated bar of 2020-12-21 14:00, whose close is 12.7 minutes before its open. Stamped with its stored close, that outcome would be refused as time going backwards.

**Re-audit, part 1** (the vault and the docs): pass with fixes, 7 Minor, all fixed.

**Re-audit, part 2** (the Rust fixes, the guard and the docs): pass with fixes, 7 Minor.
- Negative weights: in progress.
- The (2N+9)ε count: fixed in the docs.
- The λ-lift caveat for the free intercept.
- The refusal frontier: documented, and an acceptance test is coming.
- Eigensolver misconvergence: rejected. nalgebra 0.35's `symmetric_eigenvalues` passes `max_niter = 0` and iterates until convergence. The orchestrator read the source.
- Readout equality includes the owning memory: documented.
- The guard's open routes: closed.

**Claims checked by `scout`** (brief `claims-check.md`; read through a summarizing fetch tool):
- Kraken Pro tier 1, 0.40 % / 0.80 %: confirmed.
- Binance VIP 0, 0.10 % / 0.10 % (0.075 % with BNB): confirmed.
- Coinbase Advanced EU, 0.25 % / 0.50 %: unverified at the source. Every Coinbase page answered HTTP 403. Two news reports agree, effective 2026-09-16, with the first tier cut from 25,000 to 10,000 USD.
- ESMA register:
  - Coinbase Luxembourg (CSSF), Payward Europe (Central Bank of Ireland), Bitstamp Europe (CSSF) and Bitvavo (AFM) are passported to France. Payward Global is not.
  - No Binance entity is listed. Binance's exit from France on 2026-07-01 rests on press reports.
  - Three rows carry a later `ac_lastupdate` date. It is the row's last update, not its authorization.
- The four decision models:
  - all four date from autumn 2026. For Tev1, the scout found no date on the page; the auditor read the GitHub API, which gives created on 2026-09-23. The orchestrator re-checked it.
  - all four are self-reported;
  - Microsoft-Decision-1 flips on 1.3 % of perturbations on average, across eight perturbation types.

**Counts.**
- Rust: 135 tests (backtest 59, ols-core 36, risk 40), plus the follow-up in progress.
- Python: 146 offline tests.
- fmt, clippy `-D warnings` and rustdoc `-D warnings` are clean. Rust 1.89 (the MSRV) compiles the workspace.

**Kani so far** (0.68.0, WSL2, 12 threads, 4 to 7 harnesses at a time, CaDiCaL):
- `buy_gates_block_every_buy` is SUCCESSFUL in 795 s, with 5 of 5 covers satisfied.
- The other 13 harnesses had each run for more than 50 minutes, with a 60-minute cap per harness.
- A Kissat run on `round_down_to_lot_full` and `tripwires_full` is under way.
- That lot rounding is trivial for a human (`r % lot == 0` with `r = qty/lot·lot`), yet hard for SAT. That is the teaching hook.

**Lessons.**
- Each parallel agent needs its own scratch folder: a mutation script was overwritten.
- A summarizing fetch tool paraphrases numbers, so a page read through it is "read", not "verified".
- Python's text mode on Windows writes CRLF. Agents' edit scripts converted files, and a `.gitattributes` now normalizes them.
- WSL2 needed SVM in the firmware.
- Shell heredocs ate the page's LaTeX backslashes.
- The `teacher` role was born at the end of this iteration, at Abdou's request.

**Iteration 2 additions:**
- a Sharpe standard error (Lo 2002, with skewness and kurtosis) and the deflated Sharpe ratio (Bailey and López de Prado 2014), on every reported Sharpe, before any row reaches RESULTS.md;
- a vault manifest: the SHA-256 and the sealed row count of each Parquet file;
- making the slow Kani harnesses tractable, for example with function contracts and stubs that cut the 128-bit division out, or with narrower-width models;
- a runtime tripwire in `conftest.py` that catches what the static guard cannot see.

**The bolder next step.** Report n°1 as one reproducible command:
- it runs from the verified archives, through `load_klines`, the Rust simulator (PyO3) and the risk kernel, to the table in RESULTS.md;
- it prints the manifest hashes;
- it includes a cheater baseline, which H1's detector must catch and the vault seal must refuse;
- it puts a Sharpe interval and a deflated Sharpe on every row.

## Report (short)

- the sections you changed and the demo you added;
- your checks;
- anything you could not ground;
- the spot where the Kani table goes;
- one bolder next step.
