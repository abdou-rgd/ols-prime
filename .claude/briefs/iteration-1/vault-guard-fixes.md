# Brief — harden the vault guard after the re-audit (iteration 1)

**Role.** `data-engineer`. **Scratch folder:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\data-engineer-guard\` (yours alone).

The `leakage-auditor` re-audited the vault lock you built (`vault-lock.md`). Its verdict is pass with fixes, and every finding is Minor. Its probe log is `...\scratchpad\audit\vault_probe.log`. The orchestrator handles the docs and `.gitignore`. This brief covers the code.

## 1. The AST guard is a name filter

The guard is in `python/tests/test_protocol.py:307-338`. It catches aliases, attribute calls and the banned imports, but the auditor's probes found these misses:
- `getattr(pl, 'read_' + 'parquet')`;
- `importlib.import_module('duckdb')`;
- `pl.read_csv_batched`, `pl.read_database`, `pl.scan_pyarrow_dataset`, `pl.from_pandas`;
- `zipfile` plus `csv.reader` on the raw cache;
- `pa.ipc.open_file`;
- `open('x.parquet', 'rb')` followed by `pl.DataFrame.deserialize`;
- `np.genfromtxt`;
- a forged time column: `seal(frame.with_columns(known_at=lit(2020)), time_col='known_at')`.

None of them matters today, because no such code exists. They all matter once feature and model code exists. The guard stops carelessness, not intent, and that is the right bar. Extend it:
- **Names:** add `read_csv_batched`, `read_database`, `scan_pyarrow_dataset`, `from_pandas`, `from_arrow`, `deserialize`, `ZipFile`, `genfromtxt`, `loadtxt` and `import_module`.
- **Imports:** ban `csv`, `zipfile`, `importlib` and top-level `pyarrow` outside the modules that need them today. Read the code to find which ones do, for example `binance_vision.py`, and allow-list them exactly.
- **Dynamic access:** flag `getattr` on a polars, pyarrow, pandas or numpy alias, and `open(...)` on a string literal ending in `.parquet` or `.zip`.
- **Scope:** walk every `.py` file under `python/` except `.venv`, including `python/tests`. Allow-list the test files that legitimately write or read a temporary Parquet file, by name, each with a one-line reason.
- **Rust:** add a test that `Cargo.lock` at the repository root lists no crate named `parquet*`, `arrow*` or `csv*`. No Rust code may read data until a Rust seal exists.
- **The forged time column:** when the frame has an `open_time` column, `seal` also drops every row whose `open_time` is in the vault. This never drops a row the `known_at` seal keeps, since a bar that opens in the vault is final in the vault. It only defeats a forged time column. Test it with the auditor's probe.

## 2. `OPEN` may be empty (`protocol.py:80`)

When the environment key is set and `runs/vault/OPEN` exists, require its first line to be a non-empty ISO date (`date.fromisoformat`). Otherwise raise `VaultSealed("OPEN file has no date")`. With the environment key unset the vault stays sealed, whatever the file holds. Test the empty file, a non-date first line and a valid one.

## 3. UTC only means UTC (`protocol.py:47`)

`Period` accepts any zone with a zero offset, for example Europe/London in winter. Accept only `datetime.timezone.utc` and `zoneinfo.ZoneInfo("UTC")`, and reject the rest with a clear error. Test both accepted forms, Europe/London in January and a naive datetime.

## 4. The bolder check: measure the guard

The guard's coverage should be measured, not assumed. Add a parametrized test of cheater snippets: one source string per bypass listed in §1, each passed to `violations(source, filename)`, and each must be flagged.
- Add negative controls, for example `from olsprime.protocol import load_klines` and an ordinary polars expression. These must pass.
- Print or keep a small table of which rule flags which snippet, so that the report can show the coverage.

## Constraints

- Own `python/olsprime/protocol.py`, `python/tests/test_protocol.py` and, if needed, `python/tests/conftest.py`. Touch nothing else: not `crates/`, not `docs/`, not `.gitignore`.
- No new dependency.
- `cd python && uv run --locked pytest` must be green. Today it passes 56 tests.
- Do not commit.

## Report (short)

What changed, the new test count, the coverage table of the cheater snippets, anything you could not catch and why, and one bolder next step.
