# Brief — Guard fixes, round 3 (iteration 2)

- **Agent:** the `data-engineer` of guard rounds 1 and 2, resumed.
- **When:** 2026-10-28, after the guard re-audit of round 2.
  - **Verdict:** pass with fixes. Every round 1 finding is closed as claimed. On the fake tree, the auditor's 43 old cases and 13 new ones either trip, fail because what they reach no longer exists, or land on a stated limit. The exceptions are below.
  - **Where the auditor worked:** the harness and its cases are in `scratchpad\audit-guards\` (`test_round2.py`, `evasions4.py`), under the orchestrator's scratch folder.
- **This is the last guard round of the iteration.** Close the two ordinary routes, state the rest plainly, and stop. The structural fix, the physical seal, is iteration 3's.

## R1 — Major: a file allowed `reader` can read the repository's `data/` directly

**Where.** `python/tests/tripwire.py:503-515` (`gate`). The `reader` branch has no rule for the repository's data. `allowlist.py:39-45` gives `reader` to `tests/test_universe.py`, `tests/test_binance_vision.py` and `tests/test_protocol.py`, so that they can read back files they just wrote.

**Scenario.** Judged as `tests/test_universe.py`, this line passes both guards and returns the 2026 rebalances:
```python
pl.read_parquet(REPO_ROOT / "data" / "parquet" / "universe" / "spot_usdt_top30.parquet")
```
The empty `OLSPRIME_DATA_DIR` does not help, because the path is explicit.

**Fix.**
- In `gate`, a path under `repo_data_roots()` (the repository's own `data/`, not the configured root) trips for every rule, unless the calling file is `olsprime/protocol.py`.
- The raw all-frames rule stays as it is.
- Nothing legitimate reads the repository's `data/` under pytest except through the seal. Check this, and say so in the report: `provenance.current_manifest` uses `data_dir()`, which is the empty test root.

**Test, on a fake only.** Use a path under the repository's `data/` that does not exist, for example `REPO_ROOT / "data" / "parquet" / "__tripwire_probe__.parquet"`.
- From a file that holds `reader`, the call must trip before anything is read.
- If the gate ever fails to trip, the read fails with `FileNotFoundError`, so no real file is ever touched.
- Repeat for `pq.read_table` and for `pl.scan_parquet(...).collect()`.

## R2 — Major: `zipimport` reads zip members, and the tripwire never sees it

**Scenario.** `zipimport.zipimporter(str(archive)).get_data("BTCUSDT-1d-2026-03.csv")` returns the bytes of a member of a raw archive, under `data/raw/` or in a copy. It does not go through `zipfile.ZipFile`, so no gate runs, and `zipimport` is not banned statically.

`tripwire.py:22-24` and `:198` mention `zipimport` only as frames to skip, which reads as if it were handled.

**Fix.**
- **Static.** Add `zipimport` to the banned modules in `test_protocol.py`, with a cheater snippet.
- **Runtime.** Gate `zipimport.zipimporter.__init__` in place as a `raw-reader` target with the path rule only (trigger off), as `ZipFile` is. In Python 3.12 it is a pure-Python class, frozen.
  - The import system builds zipimporters for the zips on `sys.path`, such as `python312.zip`, through `sys.path_hooks`, which holds the same class object.
  - So the gate must never trip on a path outside `data/`.
  - Check that importing from `python312.zip` still works under the tripwire.
- **Docstrings.** Fix `:22-24` and `:198`, so that `zipimport` is no longer described only as "acting for its caller".

**Tests.**
- A zip under the fake data root's `raw/` (the configured root that `sealed_vault` gives each test), opened with `zipimporter` from a file without `raw-reader`, trips.
- The same zip outside `data/` does not trip.

## R3 — Minor: forged keys plus a copy of a vault file outside `data/`

`copy.write_bytes(VAULT_PQ.read_bytes())`, then `load_klines_file(copy, "1d")` with both keys forged, returns vault rows with no trip at run time. Only the forge is flagged statically, as `seal-patch`.
- **Docs.** `docs/DATA.md:151` must say "a copy of a vault file outside `data/`" in so many words.
- **Optional.** With both keys present, the two loaders trip unless some file of the stack holds `seal` or `seal-patch`. Do it only if no legitimate test breaks: `test_manifest` and `test_protocol` keep their fake-vault tests. Otherwise leave it as a stated limit.

## R4 — Minor: child processes have no tripwire

`subprocess` is banned, but `multiprocessing` and `concurrent.futures.ProcessPoolExecutor` are not.
- `olsprime/bench/report1.py:584` and `invariance.py:534` use `ProcessPoolExecutor` legitimately, through `load_klines`. That read is sealed in the child, and the child inherits the empty data root.
- **Docs.** State this as a limit in `docs/DATA.md`.
- **Do not flag the pool statically.** The bench is another agent's, and its tests use the pool.
- **Also state the evidence that no test needs the real store.** GitHub Actions runs the whole suite on a runner that has no `data/` at all. Check that no test skips or branches on whether real data exist (a `.exists()` on a data path, a skip marker). Report what you find.

## R5 — Minor, docs only

1. `DATA.md:148` gives `pl.read_parquet.__closure__` as its example, but that one trips: polars' `read_parquet` calls the wrapped `scan_parquet`. The live example is `pl.scan_parquet.__closure__`.
2. `DATA.md:142` should add that a `MonkeyPatch.undo()` of the session patches, run from a test, now trips at the `__class__` restore, and not only at teardown.
3. `tripwire.py:22-24` (see R2).

## Constraints

- **Files.** Only these:
  - `python/tests/tripwire.py`, `conftest.py`, `allowlist.py`;
  - `python/tests/test_tripwire.py`, `test_protocol.py`;
  - `docs/DATA.md`, the tripwire section only.
- **Off limits.**
  - `python/olsprime/bench/**` and its three test files: an `ml-engineer` is finishing there.
  - `crates/`.
  - Every other doc.
- **Python.** Use `uv run --no-sync`. Never rebuild the venv.
- **Data.**
  - Fakes only. Never read the real `data/` or the vault, and never set `OLSPRIME_OPEN_VAULT` outside a fake tree.
  - Every R1 test uses a path that does not exist.
- **Time.** Report by 04:15 UTC.

## Done when

- `uv run --no-sync pytest tests/test_tripwire.py tests/test_protocol.py` passes.
- The whole suite passes, except `test_rust_bindings.py`.
  - The bench's three test files may be mid-edit by the `ml-engineer`.
  - If one of them fails, show that the failure is not yours: run the same test with your changes stashed, or read its traceback.
- The auditor's cases n1, n2 and the R1 scenario trip. Re-run them from `scratchpad\audit-guards\` on the fake tree, with the auditor's harness, read-only.

## Report

- The files changed.
- The R1 and R2 designs.
- The test counts.
- The R3 choice.
- What you found for R4: tests that depend on whether data exist.
- One bolder next step.
