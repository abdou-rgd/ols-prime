# Brief — Guard fixes after Audit A, phase 2 (iteration 2)

- **Agent:** the `data-engineer` who built the manifest and the tripwire (`universe-manifest-tripwire.md`), resumed.
- **When:** 2026-10-28, after Audit A's data verdict, "pass with fixes": no Blocker, 2 Major, 4 Minor. The data, the manifest and the seal are sound. The holes are in the guards.
- **Motto:** we should be more ambitious. The deliverable says that a read outside the allowed files fails "whatever the indirection". Make that true for every path the auditor found, and keep the red-team corpus growing.

The auditor's evasion tests are in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\audit-a\`: `bypass/test_bypass.py` and `bypass2/test_evasion.py`. Read them first.

## D1 — Two read paths pass both guards

`bypass2/test_evasion.py` gives no static findings, judged as `tests/test_evasion.py`. Under the installed tripwire, it read 31 vault bars and the full BTCUSDT 1d Parquet, with no trip.

- **Path A, the raw archives.**
  - A helper module is written to `tmp_path`, put on `sys.path` and imported under an unbanned name.
  - It calls `zipfile.ZipFile` and `pl.read_csv` on `data/raw/.../BTCUSDT-1d-2026-03.zip`.
  - The static guard never walks files outside `python/`. The tripwire wraps neither `zipfile.ZipFile` nor polars' CSV readers.
- **Path B, Parquet through pyarrow.dataset.** The call is `sys.modules["pyarrow.dataset"].ParquetFileFormat().make_fragment(path, filesystem=sys.modules["pyarrow.fs"].LocalFileSystem()).to_table()`.
  - `make_fragment` and `Fragment.to_table` are not in `TARGETS`.
  - `sys.modules[...]` is not in `DYNAMIC_NAMES` (`tests/test_protocol.py:401`).

**Fix.**
1. **Tripwire, more entry points.** Wrap:
   - `zipfile.ZipFile`;
   - polars' `read_csv`, `scan_csv` and `read_csv_batched`, under the rule `raw-reader`;
   - `pyarrow.dataset.FileFormat.make_fragment`, `ParquetFileFormat.make_fragment`, `FileSystemDataset.from_paths` and `Fragment.to_table`.
2. **Tripwire, a path rule.** Any wrapped reader whose path argument resolves under the repository's `data/` trips, whatever the entry point, unless the calling frame holds the matching allowance. Resolve symlinks, and handle `str`, `Path`, file-like objects with `.name`, and lists of paths.
3. **Static guard.**
   - Flag `sys.modules[...]` and `sys.modules.get(...)` as `dynamic`.
   - Flag `sys.path.insert`, `sys.path.append` and `sys.path.extend` as `dynamic`.
   - Give allowances, with reasons, to any existing file that needs them.

## D2 — One line switches the seal off

**What fails.** `monkeypatch.setattr(olsprime.protocol, "vault_open", lambda: True)`, or the same on `"VAULT"` or `"seal"`, or with a computed name such as `"vault_" + "open"`, gives no static finding.
- `protocol.py:138` consults the module global `vault_open()` at read time, so the next `load_klines` returns the vault rows.
- The tripwire sees `protocol.py` as the calling frame, which holds the `reader` allowance, and stays silent.

**Fix.**
1. **Static rule.** Flag `setattr`, `monkeypatch.setattr`, `mock.patch` and `mock.patch.object` when their target is `olsprime.protocol`, or a module object imported from it. Also flag them when the attribute string is `seal`, `vault_open`, `VAULT`, `load_klines`, `load_klines_file` or `load_universe`. A computed attribute string on that target counts. Files that hold the `seal` allowance are exempt; `test_protocol.py:246` does this legitimately.
2. **Runtime check.** The tripwire session watches `protocol.seal` through the autouse fixture. A test fails if any call returned a row at or after `VAULT.start` from a test file without the `seal` allowance. Without the vault keys, no test should ever see a vault row.

## The red-team corpus

- Add the auditor's two evasion tests to `tests/test_tripwire.py`, as snippets that must trip at runtime, and to the static guard's cheater corpus in `tests/test_protocol.py`, where they must be flagged.
- Add one snippet per new rule: `sys.modules`, the `sys.path` helper, a `zipfile` read, a `read_csv` of an archive, `make_fragment`, and a `monkeypatch` of the seal.
- Every snippet must work against fakes in `tmp_path`, or against a fake `data/` tree. The tests stay offline, and never read the real `data/`.

## Minor, in `docs/DATA.md`

- **D3, the survivorship evidence.**
  - 61 of the 274 members stop trading before 2025-11-30, and 355 of the 568 daily files end before 2025-12.
  - The 2019 test months have fewer than 30 members: 2019-01 has 19, 2019-02 has 19, 2019-03 has 22 and 2019-04 has 28.
  - Check these figures with your survey before writing them.
- **D4, the residual risk.** A pair whose archives Binance removed from the bucket altogether is invisible to the listing, and nothing in the pipeline can detect it.
- **D5 and D6 are already stated.** Keep them.
- **Keep the orchestrator's edits to `docs/DATA.md` §2:** the Bitvavo row, and the sentence on the 2026-10-27 re-read.
- **State the tripwire's remaining limits plainly:** `pyarrow._parquet`, `ctypes`, a subprocess, and a parser over `open()`.

## Not now: the physical seal

The auditor proposes the stronger fix: keep vault rows in a separate tree (`data/vault/`), so that no reader can reach them without both keys. The orchestrator records it as iteration 3's bolder data step. Do not start it tonight.

## Constraints

- **Files.**
  - `python/tests/tripwire.py`, `tests/conftest.py`, `tests/allowlist.py`, `tests/test_tripwire.py` and `tests/test_protocol.py` (the static guard and its corpus);
  - `python/olsprime/protocol.py`, only if the runtime check needs a hook there, and without changing the seal's semantics;
  - `docs/DATA.md`.
- **Out of bounds.**
  - `python/olsprime/bench/**` and its tests: the `ml-engineer` is fixing those right now.
  - `crates/**`, and `python/tests/test_rust_bindings.py`.
- **Python.** Run it with `uv run --no-sync`: a `rust-engineer` is editing crates that the extension compiles.
- **Never** open the vault. Never set `OLSPRIME_OPEN_VAULT`.
- **Time.** Report by 00:55 UTC. If a fix is not finished by then, report what is in and what is not.

## Done when

- `uv run --no-sync pytest` is green.
- Every snippet in the corpus is flagged by the guard it targets.
- The auditor's two evasion tests, run as they are, now trip or are flagged.

## Report

- What changed, file by file.
- The new count of entry points and rules.
- The corpus, with each snippet and the guard that catches it.
- The remaining limits.
- The test count.
- One bolder next step.
