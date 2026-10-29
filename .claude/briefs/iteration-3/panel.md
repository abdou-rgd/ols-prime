# Brief — Hourly panel of the universe members (iteration 3)

- **Agent:** `data-engineer` (sonnet, high).
- **Runs in parallel with:** a `rust-engineer` on `crates/` (`matrix-core.md`) and an `ml-engineer` on the H1 adversary (`adversary.md`).
- **Motto:** we should be more ambitious, in the scale of what we build, never in what we claim.

## Goal

Build the point-in-time panel that Matrix reads in iteration 3: every hourly decision of every universe member during the months it is a member, with its features and its one-day forward label, over the development period.

Read first:
- `docs/PROTOCOL.md`, the section "Matrix on hourly bars (iteration 3)", "Panel" part. It is the pre-registration, so do not deviate from it. If something in it cannot be built as written, stop and report.
- `docs/DATA.md`, on the universe, the hourly quirks and the guards.
- The code you build on: `python/olsprime/features/basic.py`, `python/olsprime/labels/forward.py`, `python/olsprime/universe.py`, `python/olsprime/protocol.py` (`load_klines`, `load_universe`, `DEVELOPMENT`) and `python/olsprime/bench/pipeline.py` (`build_frame`, the daily equivalent).

## Deliverables

1. **`python/olsprime/features/panel.py`.**
   - `member_months(universe)`: one row per (symbol, month) of membership, from the sealed universe file. A member of month M is a member of the rebalance at M's first day.
   - `hourly_panel(...)` returns one row per (member, hour) in member months, sorted by `(decision_time, symbol)`, with:
     - `symbol`, `open_time`, `decision_time`, `month`;
     - the 15 `FEATURE_COLUMNS` and `threshold`;
     - the label columns `entry_time`, `exit_time`, `label_known_at`, `fwd_return`, `label`;
     - a boolean `complete`: every feature and the label are non-null.
     - Defaults: h = 24, months 2017-10 → 2024-12, every member, `data_dir` from `olsprime.paths`.
   - How each row is built:
     - Bars are read through `load_klines(symbol, "1h", ...)`, then cut to `known_at < DEVELOPMENT.end` **before** features and labels. So a label that needs a 2025 bar is null: the validation year never reaches a label.
     - Features and labels are computed on each pair's whole hourly history, with `basic_features` and `forward_labels(bars, "1h", 24)`. They are joined on `(symbol, open_time)` as `build_frame` does, then restricted to member months, using the month of `decision_time`.
     - Rows from non-member months are dropped. Above all, a pair's months before it first joins are dropped: using them would leak its later selection (survivorship).
   - Refuse any month at or after `DEVELOPMENT.end`, like the splitter's guards.
   - `h1_subpanel_members(universe)`: the 10 members with the most member months in the development period, ties broken by symbol. The panel future-invariance test uses them.
2. **A summary command,** `uv run --no-sync python -m olsprime.features.panel --summary`.
   - It prints, per month: members, rows, labelled rows, complete rows, and the class shares of the complete rows.
   - It also prints the totals and the 10 sub-panel members.
   - It writes the same as JSON to `runs/<UTC date>-panel/summary.json` (suffixed if taken).
   - Measure and report its wall time and peak memory on the real store.
3. **No cache in Parquet.** Only `olsprime/protocol.py` may open a Parquet file, and the static guard enforces it. Build the panel in memory. If a full build takes more than about 3 minutes, propose a cache design that goes through the seal, and ask before building it.
4. **Tests** in a new file `python/tests/test_panel.py`, on fake stores under a temporary data root, as the other tests do (see `tests/fakes.py`):
   - rows exist only in member months;
   - a pair contributes nothing before its first member month;
   - the cut at `DEVELOPMENT.end` nulls the labels that would need later bars;
   - the cut keeps every earlier label exactly as `forward_labels` computes it on the full bars;
   - h = 24 is found by time, across gaps, as in `forward_labels`;
   - determinism and ordering;
   - `complete` is right;
   - the sub-panel's tie rule.
   - Every test must pass under the tripwire, so read nothing from the repository's `data/`.
5. **Docs input.** In your report, a short paragraph for `docs/DATA.md`, section "Hourly panel (iteration 3)", with the totals from the real run. The orchestrator writes the docs.

## Constraints

- **Files.** `python/olsprime/features/panel.py`, `python/olsprime/features/__init__.py` (exports only) and `python/tests/test_panel.py`. Do not change `basic.py`, `forward.py`, `protocol.py` or the guards. If one needs a change, ask.
- **Python.** Always `uv run --no-sync ...`: a `rust-engineer` is editing the extension. Run the whole suite at the end with `cd python && uv run --no-sync pytest`.
- **Scripts.** Write scripts with a file tool, never through shell heredocs.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\panel\`.
- **Git.** Run no git command that changes state, and do not commit.
- **The vault.** Never set `OLSPRIME_OPEN_VAULT` and never write `runs/vault/OPEN`.

## Done when

- `hourly_panel` and the summary command work on the real store.
- `test_panel.py` passes, and so does the whole Python suite.
- The summary JSON exists, and its totals are in your report.

## Report (short)

- The API, with signatures.
- The real totals: rows, complete rows, rows per month at the extremes, and class shares.
- Wall time and memory.
- Anything in the pre-registration that was hard to follow.
- One bolder next step.
