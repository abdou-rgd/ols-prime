# Brief — Guard fixes, round 4: path spellings (iteration 2)

- **Agent:** the `data-engineer` of guard rounds 1 to 3, resumed.
- **When:** 2026-10-28, after the guard re-check of round 3.
  - **Verdict:** pass with fixes, Minor only.
  - **What held.** R1, R2 and R3 behave as claimed. The repository-data rule caught 21 path forms: relative paths, `..`, `os.PathLike`, `filesystem=LocalFileSystem()`, globs, 8.3 short names, junctions, `np.load`, `ds.dataset(dir)` and open handles. The R3 tightening breaks nothing, and DATA.md's R4 and R5 text is accurate.
  - **The auditor's files.** Its cases are in the orchestrator's scratch folder, `scratchpad\audit-guards\`: `evasions5.py`, `evasions6.py`, `test_round3.py` and `test_round3b.py`.
- **A short round, about 20 minutes.** Close the cheap spellings, state the rest, stop.

## F1 — Minor: path spellings the repository-data rule does not recognise

From a frame of a file that holds `reader` (`tests/test_universe.py`), each of these returned vault rows with no trip. From any other file they still trip, through the caller rule.

1. `pl.read_parquet(VAULT_PQ.as_uri())`: a `file:///C:/...` URI. `_norm` treats it as a relative path (`tripwire.py:296-301`).
2. `"\\\\?\\" + path`: the extended-length prefix. The auditor reasoned this case and did not re-run it.
3. `ds.dataset("parquet/spot/klines/1d/BTCUSDT.parquet", filesystem=fs.SubTreeFileSystem(<data dir>, LocalFileSystem()))`: the path is relative to the filesystem's `base_path`.
4. `pq.read_table(fs.LocalFileSystem().open_input_file(str(VAULT_PQ)))`: a pyarrow `NativeFile` has no `.name`.

**Fix.**
- **In `_collect` / `data_paths`:**
  - map `file:` URIs to paths (`urllib.parse.urlparse` plus `urllib.request.url2pathname`);
  - strip `\\?\` and `\\?\UNC\`;
  - when `kwargs["filesystem"]` has a `base_path` (a `SubTreeFileSystem`), join it to each relative candidate.
- **`NativeFile`:** state it as a limit in `docs/DATA.md`, next to `pyarrow._parquet`. Wrapping the cython filesystem methods is not worth it in this iteration.
- **Tests.** Fakes only, as in round 3: a nonexistent path under the real `data/` for the URI and `\\?\` forms, and a fake data root for the `SubTreeFileSystem` form. Each must trip from a file that holds `reader`.

## F2 — Minor: R3's exemption is transitive through helpers of exempt files

`from tests.test_tripwire import run_source` puts a `test_tripwire.py` frame on the stack. Forge the keys, run a helper that calls `load_klines_file(copy)`, and the loader is exempt: it returns vault rows with no trip. The forge is flagged statically (`seal-patch`), but the import of `tests.test_tripwire` is not.

**Fix.**
- Add imports of `tests.test_tripwire`, `tests.test_protocol` and `tests.test_manifest` to the static `disarm` rule, as `tests.conftest`, `tests.tripwire` and `tests.allowlist` already are.
- First check that no test imports another test module today; if one does, report it instead of breaking it.
- State the transitive case in `DATA.md`, under the forged-keys limit.

## Constraints

- **Files.** Only:
  - `python/tests/tripwire.py`, `test_tripwire.py`, `test_protocol.py`;
  - the tripwire section of `docs/DATA.md`.
- **Off limits.** An `ml-engineer` is in `python/olsprime/bench/**` and its three test files.
- **Python.** `uv run --no-sync`.
- **Data.** Fakes only.
- **Time.** Report by 04:20 UTC.

## Done when

- `test_tripwire.py` and `test_protocol.py` pass.
- The whole suite passes, except `test_rust_bindings.py`. If a bench test fails while the `ml-engineer` is mid-edit, show that it is not yours.
- The auditor's `test_round3b.py`, re-run read-only on the fake tree: the URI, `\\?\` and `SubTreeFileSystem` forms trip, and the `NativeFile` form lands on its stated limit.

## Report

- The files changed.
- The test counts.
- What `test_round3b.py` gives.
- One bolder next step.
