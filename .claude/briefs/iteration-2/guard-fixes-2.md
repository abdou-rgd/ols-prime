# Brief — Guard fixes, round 2 (iteration 2)

- **Agent:** the `data-engineer` who built the tripwire and made the round 1 guard fixes (`guard-fixes.md`), resumed.
- **When:** 2026-10-28, after an independent re-audit of the guards: pass with fixes, with 3 Major and 8 Minor. The D1 and D2 fixes hold against what they targeted. The new holes are elsewhere.
- **Motto:** we should be more ambitious. Close every ordinary route, and say plainly what in-process guards cannot do.

The auditor's harness and corpus are in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\audit-guards\`. Read them first:
- `conftest.py` installs the real `tests/tripwire.py` over a fake `data/` tree;
- the 43 cases are in `evasions.py`, `evasions2.py`, `evasions3.py` and `test_zz_undo.py`.

## Major

### G1 — Forging the two keys opens the real vault from any test

`from tests.test_protocol import open_vault; open_vault(monkeypatch); load_klines("BTCUSDT", "1d", caller="x")`
- **What happens.** It returns vault rows with no trip and no static finding. `install_seal_watch` accepts it, because `real_vault_open()` is true. ACCESS.log goes to the tmp runs folder and is lost.
- **Where.** `protocol.py:95-116`, `tests/conftest.py:21-27`, `tests/test_protocol.py:58-62`. Cases `e16_forge_keys_by_hand` and `e16b_open_vault_helper`.

**Fix.**
1. **Data folder.** In `sealed_vault`, also set `OLSPRIME_DATA_DIR` to an empty tmp folder. A default `load_klines(symbol, interval)` then never finds the real store under pytest. Check that no test needs the real data: the auditor found none.
2. **Seal watch.** Also wrap `load_klines_file` and `load_universe`; both are already in `SEAL_NAMES`.
   - A path that resolves under the **repository's** `data/` (`REPO_ROOT / "data"`, not `data_dir()`) while the vault is open trips with rule `seal`.
   - The exception is a calling file that holds a new `vault-open` allowance, which no file holds today.
   - These wrappers also check their returned frames for vault rows while the real keys are absent. The result check then no longer depends on the `seal` global (G3d).
3. **Static rule.** Flag the string `"OLSPRIME_OPEN_VAULT"` and the names `open_vault` and `vault_key_file` as `seal-patch`, outside `test_protocol.py`, `test_manifest.py` and `test_tripwire.py`.

### G2 — An allowed module exports an unsealed raw reader

- **What happens.**
  - `binance_vision.read_klines_zip(VAULT_ZIP, "BTCUSDT")` returns 31 vault bars with no trip; the static guard does flag it as `raw-reader`.
  - `getattr(bv, "read_" + "klines_zip")(...)` does the same, with no trip and no static finding.
- **Where.** `python/olsprime/data/binance_vision.py:408`, `tests/allowlist.py:22-25`, `tests/tripwire.py:352`. Cases e10 and e10b.

**Fix.**
1. **Runtime.** When the path is under `data/raw/` and the rule is `raw-reader`, require **every** non-infrastructure frame on the stack, the test file included, to hold `raw-reader`, not only the innermost one. `test_binance_vision.py` builds from tmp raw folders, so it should be unaffected: check.
2. **Static.** Extend the `getattr`, `vars` and `__dict__` rule to every module of `olsprime.data`.

### G3 — The tripwire can be disarmed from inside a test, and `runpy` hides the disarm from the static guard

All of these return vault rows with zero findings from either guard, on the fake tree:
- `from tests.conftest import PATCHES; PATCHES.undo()`, then `runpy.run_path(helper)`;
- `monkeypatch.setattr(tests.tripwire, "is_allowed", lambda f, r: True)` plus a runpy helper;
- `ALLOWED["tests/test_x.py"] = {...}`, which silences both guards for that file;
- `load_klines.__globals__["seal"] = ...__wrapped__` and `__globals__["vault_open"] = ...`;
- `importlib.reload(olsprime.protocol)`, then `vars(protocol)`.

**Fix.**
1. Ban `runpy` (`import:runpy`), with an allowance for `test_tripwire.py` if it needs one.
2. Freeze `ALLOWED`: a `MappingProxyType` over frozen inner maps. The per-test fixture compares it at teardown with a snapshot taken in `pytest_configure`.
3. Check, at the teardown of every test:
   - every patch that `install` applied is still bound;
   - `type(protocol)` is `GuardedProtocol`;
   - for each of the six names, `protocol.__dict__[name]` is in `permitted[name]`;
   - `tests.tripwire.is_allowed`, `calling_file` and `data_roots` are the originals.
4. Make the result check independent of the `seal` global. This is G1, fix 2.
5. **Static.** Flag the following outside the guard files, as `dynamic` or a new rule, your choice:
   - `__globals__`, `__wrapped__`, `__closure__`, `__bases__`, `__mro__`;
   - `import tests.conftest` and `from tests.conftest`;
   - `import tests.tripwire` and `from tests import tripwire`;
   - `.undo(`.

## Minor

- **M1. "Putting the original back is always allowed" is the hole.** Set `permitted["seal"] = [watched_seal]` only, and `del watched_seal.__wrapped__`. Restoring at unconfigure stays possible: undo the module class first, LIFO.
- **M2. `__class__` and `__dict__` are not protected.** Refuse both in `GuardedProtocol.__setattr__`. In `pytest_unconfigure`, restore the class through `types.ModuleType.__setattr__(protocol, "__class__", types.ModuleType)` before `PATCHES.undo()`.
- **M3. `__wrapped__` on the reader wrappers.** `pl.scan_parquet.__wrapped__(P).collect()` reads with no trip. Delete `__wrapped__` from every wrapper, and state `__closure__` as a remaining limit.
- **M4 and M5. Subclass wrapping leaves the original one attribute away.**
  - `pq.ParquetFile.__bases__[0]` and `zipfile.ZipFile.__bases__[0]` both read.
  - `zipfile.Path` reads with the original `ZipFile`.
  - For the pure-Python classes, `ParquetFile` and `ZipFile`, patch `__init__` in place instead of subclassing.
  - The two cython classes cannot be fixed: state them as limits.
- **M6. Static blind spots.** Add `operator.attrgetter`, `__getattribute__`, and `getattr` passed as a value without being called (as in `functools.reduce(getattr, ...)`) to the `dynamic` rule.
- **M7. More readers.** Wrap `polars.read_database` and `read_database_uri`, and add `duckdb` to the optional targets. State `polars._plr.PyLazyFrame.new_from_parquet` as a limit, polars' analogue of `pyarrow._parquet`.
- **M8. Vault metadata in the quality reports.** `{SYMBOL}.quality.json` is computed on the unsealed rows, so for BTCUSDT and ETHUSDT its `rows`, `last_open_time` and `gaps` include 2026. Do not change the files tonight. State it in `docs/DATA.md`, and say that the survey's figures count those rows.

## Docs, in `docs/DATA.md`

- **Two wrong sentences.**
  - "`Fragment.to_table` … is reached only through the wrapped constructors" is wrong: `__bases__[0]` of the wrapped cython classes reaches it.
  - "the result check on `seal` catches its effect" was true only while the `seal` global was the watched one. Rewrite both after your fixes.
- **The missing limits.** List every one that remains after your fixes, among:
  - forged keys;
  - a raw reader from an allowed module;
  - in-process disarm;
  - `__closure__`;
  - the cython bases;
  - `zipfile.Path`;
  - `polars._plr`;
  - `runpy`;
  - the quality-report metadata.
- **Say it plainly.** Every in-process guard can be disarmed from inside the process. These guards stop accidents and careless code, not sabotage. The structural fix is iteration 3's physical seal.
- **D3.** "the other 213 trade to the end of 2025" overstates. The criterion is a last daily bar on or after 2025-11-30: say that instead.

## The corpus

- Add the auditor's 43 cases, at least every one that gave vault rows, to `tests/test_tripwire.py` as runtime snippets that must trip, against fakes in `tmp_path`.
- Add their sources to the static corpus of `tests/test_protocol.py` wherever a static rule now applies.
- Every snippet runs offline against a fake `data/` tree. Never read the real `data/`.

## Constraints

- **Files.**
  - `python/tests/tripwire.py`, `tests/conftest.py`, `tests/allowlist.py`, `tests/test_tripwire.py` and `tests/test_protocol.py`;
  - `python/olsprime/protocol.py`, only for a hook, without changing the seal's semantics;
  - `python/olsprime/data/binance_vision.py`, only if G2 needs it;
  - `docs/DATA.md`.
- **Out of bounds.** `python/olsprime/bench/**` and its tests: the `ml-engineer` is finishing there. Also `crates/**`.
- **Python.** Use `uv run --no-sync`. `tests/test_rust_bindings.py` fails in the shared venv until the orchestrator rebuilds it, which is expected.
- **Never** open the vault. Never set `OLSPRIME_OPEN_VAULT` outside a fake tree. Never read real files under `data/`.
- **Time.** Report by 03:45 UTC. If something is not finished by then, report what is in and what is not.

## Done when

- `uv run --no-sync pytest --ignore=tests/test_rust_bindings.py` is green.
- Every case of the auditor's corpus that gave vault rows now trips, or is flagged statically, or is stated as a limit in `docs/DATA.md` with a reason.

## Report

- The files changed.
- For each finding (G1, G2, G3, M1 to M8): fixed, partly fixed or stated as a limit.
- The new counts of entry points and rules.
- The test count.
- The remaining limits, verbatim from `docs/DATA.md`.
- One bolder next step.
