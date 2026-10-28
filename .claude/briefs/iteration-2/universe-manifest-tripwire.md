# Brief — Universe download, vault manifest, runtime tripwire (iteration 2)

- **Agent:** `data-engineer` (sonnet, high).
- **Runs in parallel with:**
  - a second `data-engineer` (`features-labels-splits.md`) in `features/`, `labels/` and `eval/splits.py`;
  - two `ml-engineer`s in `models/` and `eval/stats.py`;
  - a `rust-engineer` building `crates/olsprime-py`, who adds one dependency to `python/pyproject.toml` and writes `python/tests/test_rust_bindings.py`.
- **Motto:** we should be more ambitious: a data layer someone else could audit in an afternoon.

## 1. Download the full universe

Abdou approved this download on 2026-10-27: public archives of data.binance.vision, under 1 GB, checksums verified.

1. Rank the universe: `uv run python -m olsprime.universe --top 30 --start 2017-10 --end 2025-12`.
   - It downloads the daily archives of every eligible USDT pair over the ranking windows. Read `universe.run` first: it fetches up to `end.shift(-1)`.
2. Download the bars of every pair that is a member in at least one month: `uv run python -m olsprime.data.binance_vision --symbols ... --intervals 1h 1d --start 2017-08 --end 2025-12`.
   - BTCUSDT and ETHUSDT are already there through 2026-09. The new pairs stop at 2025-12: the vault months are not needed before the vault opens.
3. Run the downloads in the background, with logs in your scratch folder. Keep the CLI's default of 16 concurrent requests, and retry failed archives once.
4. Flag likely ticker reuse, as with `LUNAUSDT`: a gap longer than 7 days followed by a price level that changes by more than 50 %. List these pairs in `docs/DATA.md`; do not repair them.
5. Update `docs/DATA.md`, with the date:
   - what was downloaded: pairs, intervals, months;
   - eligible pairs; members per month (minimum, median, maximum); distinct members;
   - archives fetched, missing and failed; sizes of `data/raw` and `data/parquet`;
   - the quality summary per interval (gaps, duplicates, off-grid bars, bad close times), in aggregate and for the ten worst pairs.

## 2. Vault manifest

- **Module.** `python/olsprime/manifest.py`, with `build_manifest(data_dir) -> dict` and a CLI, `python -m olsprime.manifest [--out PATH]`. The default output is `runs/manifest/<UTC timestamp>.json`.
- **Contents.** For every Parquet file under `data/parquet/` (the klines and the universe file):
  - its relative path, size and the SHA-256 of its bytes;
  - its stored row count, from the Parquet metadata;
  - its sealed row count, through the protocol's loaders;
  - the first and last sealed `open_time` (or `rebalance_time` for the universe).
- **One digest.** The JSON is canonical (sorted keys, sorted paths), and the manifest carries the SHA-256 of its own canonical body. A report then cites one hash.
- **The seal.** Reading Parquet metadata or rows is a Parquet read, so it belongs in `protocol.py`.
  - Add the helpers there: `stored_row_count(path)` and a sealed count built on `load_klines_file`.
  - Add `load_universe(path)`, which seals on `rebalance_time < VAULT.start`, unless the vault is open, and logs like `seal`.
  - `manifest.py` calls only these helpers.
  - The seal's semantics do not change.
- **Static guard.** `tests/test_protocol.py` (`ALLOWED`) must still pass: `manifest.py` reads no Parquet itself.
- **Tests.** Offline, with fake Parquet files (`tests/fakes.py` has helpers).

## 3. Runtime tripwire

The static AST guard of iteration 1 cannot see indirection: computed names, `getattr`, `exec`, a helper module built at run time. The tripwire catches them at run time, in every test.

- **Where.** An autouse fixture in `python/tests/conftest.py`.
- **What it patches.**
  - Polars: `read_parquet`, `scan_parquet`, `read_parquet_schema`, and any other Parquet entry point your installed version has. Polars here is 2.1.0; list its entry points with `dir(polars)`.
  - PyArrow: `parquet.read_table`, `parquet.ParquetFile`, `parquet.read_metadata`, `parquet.read_schema`, `parquet.ParquetDataset`, `dataset.dataset`.
  - `pandas.read_parquet`, if pandas is importable.
  - `numpy.load` with `allow_pickle=True`, `pickle.load` and `pickle.loads`.
  - `torch.load`, if torch is importable.
- **Who may read.** Walk the stack to the first frame outside polars, pyarrow, pandas, numpy, importlib and the tripwire itself. The read is allowed only if that frame's file holds the matching allowance in `ALLOWED`.
  - Keep one source of truth: move `ALLOWED` into a small module that both the static guard and the tripwire import.
- **Cheater tests.** Each bypass must trip:
  - `getattr(pl, "read_" + "parquet")`;
  - a computed path;
  - a helper module written to `tmp_path` and imported;
  - `pl.scan_parquet(...).collect()`;
  - `pyarrow.dataset`;
  - `exec` of a string.
- **Limits.** The tripwire runs only under pytest. Say so in `docs/DATA.md` or in the docstring, and the orchestrator adds a line to the protocol's change log.

## Scope and constraints

- **Files.** Only these:
  - `python/olsprime/data/**` and `python/olsprime/universe.py`;
  - `python/olsprime/manifest.py` (new);
  - `python/olsprime/protocol.py`: the read helpers above, and nothing that changes the seal;
  - `python/tests/conftest.py`, `test_manifest.py` and `test_tripwire.py`;
  - `python/tests/test_protocol.py` and the new shared allow-list module, only to share `ALLOWED`, or to add an allowance with its reason;
  - `docs/DATA.md`;
  - `data/**`, which git ignores.
- **Not yours.** `features/`, `labels/`, `eval/`, `models/`, `bench/`, `pyproject.toml` and `uv.lock`. Add no dependency: pyarrow is already there.
- **Tests.** Run `uv run pytest` at the end. The whole suite must be green. If another agent's test fails, report it; do not fix it.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou--claude-spares-spare-9EqB4p\b947edd7-985a-4415-b997-8738d276bea9\scratchpad\data-download\`.
- **Git.** Run no git command that changes state, and do not commit.

## Report (short)

- Pipelines and their tests, with counts.
- The data summary: rows, ranges, gaps, repairs, sizes.
- The manifest of the real data: number of files, its digest, BTCUSDT 1h's stored and sealed row counts. Iteration 1 found 79,837 and 73,285.
- One bolder next step.
