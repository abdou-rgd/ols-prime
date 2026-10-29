# Data

Every source OLS Prime reads, with its license and what that license lets us do. A source is listed here before its first download.

Data is never committed: `data/` is gitignored. The code is MIT. That license covers neither the data nor what is derived from it, which keep their source's terms.

## 1. Binance Vision: spot klines

**What we read**
- Monthly spot kline archives from `https://data.binance.vision`:
  - `data/spot/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM}.zip`;
  - each with a `.zip.CHECKSUM` file beside it, holding its SHA-256.
- The bucket listing at `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision` (S3 ListObjects v1), to enumerate every USDT pair, delisted ones included.
- Code: `python/olsprime/data/binance_vision.py`, `python/olsprime/universe.py`, `python/olsprime/manifest.py` and `python/olsprime/data/survey.py`.

**Terms.** The *Binance Vision Dataset Terms*, version 1.0, last updated 2026-08-26. They are published as [`TERMS_AND_CONDITIONS.md`](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) in `binance/binance-public-data`; we read them on 2026-10-25, at commit `d3357a5c33ba`. Our summary, which is not legal advice:

| Clause | What it says |
|---|---|
| 1.2 | The terms supplement the Binance Terms of Use. |
| 1.4 | Amendments apply only to data made available after them. Data already downloaded stays under the terms in force at download. |
| 2.4 | Anything derived from the data falls under these terms: charts, models, indicators. |
| 3 | The datasets are licensed under CC BY-NC-SA 4.0. Where these terms and the license both bind, the stricter obligation applies. Commercial use needs a separate written license. |
| 4.1 | The allowed non-commercial uses: academic research; non-monetized open educational projects; historical backtesting for purely personal, non-production research; non-commercial open-source evaluations. |
| 4.2–4.4 | Forbidden: live proprietary trading; automated commercial order generation; selling signals; reselling or hosting the data; building it into commercial products. |
| 4.5 | A redistributed derivative work must credit Binance Vision and be licensed under CC BY-NC-SA 4.0. |
| 7 | Respect the posted rate limits, and never script around them. On 2026-10-25, neither the site nor the repository posted any limit. |
| 8 | Never imply that Binance sponsors or endorses the project. |

The repository's own `MIT` license covers its scripts, not the datasets.

**What this means for OLS Prime**
- **Allowed use.** OLS Prime is personal, non-monetized, open-source research on historical data, which clause 4.1 allows.
- **Data.** We never commit, host or redistribute it.
- **Derived outputs.** Results, figures, reports and trained weights computed from this data may be published only:
  - with the credit "Source: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0";
  - and under CC BY-NC-SA 4.0 themselves.

  The code stays MIT.
- **Live trading (roadmap phase 6).** Clause 2.4 extends the terms to models derived from the data, and clause 4.2 forbids live proprietary trading. A real-money test of a model trained on this data, even 50 €, may therefore fall outside what the terms allow. Settle one of these before phase 6:
  - train the live model on market data whose terms allow live use, such as the chosen venue's own data;
  - or get Binance's written answer.
- **Terms of Use.** Binance has been closed to French residents since 2026-07-01. Clause 2.3 of the dataset terms grants access to any individual, but the Terms of Use they supplement (clause 1.2) have not been read yet.
- **Access.** The downloader:
  - caps concurrency: 8 workers for klines by default, 16 for the universe (the 2026-10-27 download of the members' bars also used 16);
  - retries 429, 500, 502, 503 and 504 answers with exponential backoff;
  - never fetches a verified archive twice.

**On disk (2026-10-25)**

| Pair | Interval | Archives | Bars | Open times (UTC) | Gaps |
|---|---|---|---|---|---|
| BTCUSDT | 1h | 110 | 79,837 | 2017-08-17 04:00 → 2026-09-30 23:00 | 28 gaps, 127 missing bars |
| ETHUSDT | 1h | 110 | 79,837 | same | 28 gaps, 127 missing bars: the same exchange outages, though the 2018-02-08 outage ends 11 ms later |
| BTCUSDT | 1d | 110 | 3,332 | 2017-08-17 → 2026-09-30 | none |
| ETHUSDT | 1d | 110 | 3,332 | same | none |

- All 440 archives match their published SHA-256.
- 36 of them are vault months (2026-01 → 2026-09, see [PROTOCOL.md](PROTOCOL.md)). They stay in the raw cache and in the Parquet files, and the integrity checks cover them. Every read for modelling goes through `olsprime.protocol.load_klines`, which seals them: BTCUSDT 1 h then returns 73,285 bars, the last opening on 2025-12-31 23:00.
- Binance may replace an archive later. Its repository lists the replacements in `updates/`. The cache keeps the first verified copy, so a rebuild is reproducible, but it misses later corrections until the archive is downloaded again with `--force`.

**The universe download (2026-10-27 and 28).** Abdou approved the download of the full universe on 2026-10-27: public archives, checksums verified, under 1 GB.

- **Ranking.** `uv run python -m olsprime.universe --top 30 --start 2017-10 --end 2025-12`, 16 workers.
  - The bucket lists 757 USDT pairs. 681 remain without stablecoins and leveraged tokens (the eligible pairs).
  - It fetched the daily archives of all 681 over 2017-09 → 2025-11, the windows the 99 rebalances read: 22,047 archives, none failed. 568 of the 681 have at least one such archive; the other 113 have none in that window (presumably listed after 2025-11; not checked one by one).
  - Result: `data/parquet/universe/spot_usdt_top30.parquet`, 2,633 rows, rebalances from 2017-10 to 2025-12.
- **Members.** Per rebalance: minimum 2 (2017-10, when few USDT pairs existed), median 30, maximum 30. 274 distinct members over the 99 months.
- **Bars of the members.** `python -m olsprime.data.binance_vision --symbols <the 274> --intervals 1h 1d --start 2017-08 --end 2025-12`, 16 workers: 55,348 archives asked, 12,900 downloaded, 12,882 already cached from the ranking, 29,566 missing (before the listing or after the delisting), **0 failed**, so no retry was needed. BTCUSDT and ETHUSDT keep their months through 2026-09.
- **Totals.** 35,189 archives on disk, all with a verified SHA-256 (35,189 CHECKSUM files). `data/raw` is 584 MB and `data/parquet` 405 MB: 989 MB, just under the 1 GB the approval allowed.
- **Files.** 843 Parquet files: 274 of 1 h (members only), 568 of 1 d (the 274 members through 2025-12, the 294 other eligible pairs that traded through 2025-11 stop there), one universe file.
- **Rows.** 9,977,671 stored, 9,964,021 sealed. The 13,650 rows cut are the vault months of BTCUSDT and ETHUSDT (2026-01 → 2026-09).

**Vault manifest.** `uv run python -m olsprime.manifest` writes `runs/manifest/<UTC timestamp>.json`: for every Parquet file its path, bytes, SHA-256, stored row count (Parquet metadata), sealed row count, and first and last sealed time, with one `digest` over the canonical body (sorted keys, no `generated_at`). The reads go through `olsprime.protocol` (`stored_row_count`, `sealed_klines_span`, `sealed_universe_span`), the only module that opens a Parquet file. On 2026-10-28:

| | |
|---|---|
| Files | 843 (420,120,956 bytes) |
| Digest | `230e28e830f728ec7b3974c359f944e7d7bc28f8268694ef00448e672f52f3cd` |
| BTCUSDT 1 h | 79,837 stored, 73,285 sealed, last sealed bar 2025-12-31 23:00 |
| Universe file | 2,633 stored, 2,633 sealed (last rebalance 2025-12-01) |

**Quality summary.** Produced by `uv run python -m olsprime.data.survey`. The per-pair `{SYMBOL}.quality.json` reports are written by `build_parquet` on the raw concatenation of the archives, vault months of BTCUSDT and ETHUSDT included (structural checks, not reads for modelling). "Bad close" is a close time other than open + interval − 1 ms. No pair has a duplicate, a non-monotonic time, an inconsistent OHLC row or an invalid volume at 1 h; repairs: none, nothing was changed.

| Interval | Pairs | Rows | Gaps (pairs with) | Missing bars | Duplicates | Non-monotonic | OHLC | Off-grid | Bad close | Volume | Clean pairs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h, members | 274 | 9,307,346 | 2,076 (198) | 13,927 | 0 | 0 | 0 | 258 | 1,095 | 0 | 76 |
| 1 d, members | 274 | 388,196 | 7 (7) | 349 | 0 | 0 | 0 | 0 | 70 | 5 | 208 |
| 1 d, all 568 stored pairs | 568 | 667,692 | 15 (15) | 652 | 0 | 0 | 0 | 0 | 156 | 5 | 420 |

Ten worst pairs at 1 h, by problems in all (gaps count missing bars):

| Pair | Rows | Gaps | Missing bars | Off-grid | Bad close |
|---|---|---|---|---|---|
| FTTUSDT | 45,396 | 15 | 7,497 | 0 | 8 |
| LUNAUSDT | 46,550 | 11 | 456 | 0 | 7 |
| STRAXUSDT | 44,661 | 11 | 216 | 0 | 6 |
| BCCUSDT | 8,805 | 8 | 166 | 43 | 7 |
| BTCUSDT | 79,837 | 28 | 127 | 43 | 15 |
| ETHUSDT | 79,837 | 28 | 127 | 43 | 15 |
| BNBUSDT | 71,349 | 27 | 120 | 43 | 14 |
| LTCUSDT | 70,461 | 27 | 120 | 43 | 14 |
| NEOUSDT | 71,013 | 27 | 120 | 43 | 14 |
| BNXUSDT | 29,362 | 2 | 150 | 0 | 2 |

Ten worst pairs at 1 d (members): FTTUSDT (310 missing days), LUNAUSDT (17), STRAXUSDT (7), BNXUSDT (5), BCCUSDT (4), BTTUSDT (5 invalid volumes), BTCSTUSDT (3), SUNUSDT (3), KLAYUSDT and AGIXUSDT (bad close times only). The 43 off-grid bars of BTCUSDT are the 2018-02 outage described below; the older pairs share it. The full tables are in the output of the survey command.

**Likely ticker reuse.** The rule: a daily gap longer than 7 days after which the price level changes by more than 50 % (open after the gap over the close before it). Flagged, not repaired. Over the 274 members the survey flags one pair:

| Pair | Gap starts | Gap days | Close before | Open after |
|---|---|---|---|---|
| LUNAUSDT | 2022-05-14 | 17 | 5e-05 | 1 |

`FTTUSDT` has a gap of 310 days (2022-11-16 → 2023-09-22) but resumes within 50 % of its last price, so the rule does not flag it. The rule sees only gaps; a ticker handed to a new asset without a gap passes. A price-level check across the whole series is a possible next step.

**Survivorship, in numbers (2026-10-28, from the survey and the quality reports).** The universe keeps delisted pairs, and the evidence is in the files:
- 61 of the 274 members stop trading before 2025-11-30 (their last daily bar is earlier). The other 213 have a last daily bar on or after 2025-11-30; that is the criterion, and it says nothing about whether they trade to the end of 2025.
- 355 of the 568 daily files end before 2025-12-01. Read this with care: the 294 non-members were downloaded only through 2025-11 by construction, so only 140 of the 568 end before 2025-11-30 (61 members and 79 non-members).
- The early universe is thin. 19 of the 99 rebalances have fewer than 30 members, all of them up to 2019-04: 2019-01 has 19, 2019-02 has 19, 2019-03 has 22 and 2019-04 has 28 (the first months have as few as 2). From 2019-05 on every month has 30. A test window that starts before 2019-05 ranks fewer pairs.

**Residual risk of the listing.** The universe is built from what the bucket lists. A pair whose archives Binance removed from the bucket altogether is invisible to the listing, and nothing in the pipeline can detect it: it would be a survivorship bias we cannot see. The bias that remains is the one the bucket itself carries.

**Vault rows in the quality reports (known, not changed).** `{SYMBOL}.quality.json` is computed on the unsealed rows, so for BTCUSDT and ETHUSDT (the two pairs downloaded past 2025-11) its `rows`, `last_open_time` and `gaps` include 2026 bars. The survey's per-pair figures read those reports and count the same rows. The reports are metadata about the files, not model input, and no model code reads them, but a person reading them sees vault bars counted. The files were not changed in round 2 of the guard fixes; a sealed report (computed on `known_at < VAULT.start`) is a next step.

**Runtime tripwire.** `python/tests/tripwire.py` is put up by `pytest_configure` in `python/tests/conftest.py`, before any test module is imported, so a reader captured at import time (`from polars import read_parquet`, a dict of readers, a `functools.partial`) is already the wrapper. It wraps 53 entry points (plus 3 for pandas and torch, and 5 for duckdb, each when the library is installed):
- the Parquet, IPC, Feather, Delta, Iceberg and Lance readers of polars, `read_database` and `read_database_uri`, `DataFrame.deserialize`, `LazyFrame.deserialize`, and the pyarrow parquet, dataset, IPC and Feather readers (rule `reader`);
- the pyarrow dataset constructors a fragment comes from: `make_fragment` of `FileFormat` and `ParquetFileFormat`, `FileSystemDataset.from_paths`, `FileSystemDatasetFactory`, `ParquetDatasetFactory` (rule `reader`);
- the CSV, JSON, NDJSON, Avro, Excel and lines readers of polars, `zipfile.ZipFile` opened for reading, and `zipimport.zipimporter` (rule `raw-reader`; `zipimporter.get_data` returns the members of a zip without `ZipFile`, so it is gated by the path rule alone, and the zips of `sys.path` still import);
- `numpy.load(allow_pickle=True)` (rule `binary-load`), `pickle.load(s)` and `torch.load` (rule `import:pickle`), and pandas' Parquet and Feather readers when pandas is installed.

How a call is judged:
- A call trips unless the first frame outside those libraries, the standard library and pytest sits in a file that holds the matching allowance in `python/tests/allowlist.py`, the list the static guard uses too. The standard library counts as infrastructure like polars and pyarrow: `zipfile.Path` and `importlib.metadata` act for their caller, and the read is judged at the first frame outside them. An `importlib.metadata` or `importlib.resources` read of a zip that is not under `data/` (numpy does it for `sys.path` entries such as `python312.zip`) is not stored data and is not judged.
- For a raw reader (`zipfile`, CSV) handed a path under `data/raw/`, **every** frame of the stack outside the libraries must hold the allowance, the test file included. An allowed module that exports a raw reader (`binance_vision.read_klines_zip`) is therefore no way round it: the test that calls it must hold `raw-reader` too.
- The repository's own `data/` is read through `olsprime/protocol.py` and nothing else. A wrapped reader handed a path under it trips for every rule, even from a file that holds `reader` (`test_universe.py`, `test_binance_vision.py` and `test_protocol.py` may read back what they wrote in a temporary folder, not the repository's store). Nothing legitimate needs it under pytest: every test gets an empty data root of its own, and `provenance.current_manifest` reads `data_dir()`, which is that root.
- A path rule backs it: a wrapped reader handed a path under the repository's `data/` (symlinks resolved; `str`, `Path`, `file:` URIs, the `\\?\` and `\\?\UNC\` prefixes, file-like objects with a name, lists, and a path relative to the `base_path` of a `SubTreeFileSystem` among the arguments) is checked even where its own trigger is off, and the hit names the path.
- Pure-Python classes (`pyarrow.parquet.ParquetFile`, `ParquetDataset`, `zipfile.ZipFile`) are patched in place, through `__init__`: no copy of the original is left to reach through `__bases__`, and `zipfile.Path`, which builds a `ZipFile`, is gated. Wrapped functions carry no `__wrapped__`. The gates bind their helpers (`is_allowed`, `calling_file`, `data_paths`, ...) when they are built, so patching `tests.tripwire.is_allowed` afterwards changes nothing for them.

The vault's seal is watched, in three layers:
- `olsprime.protocol` refuses, from a file without the `seal` allowance, to have `seal`, `vault_open`, `VAULT`, `load_klines`, `load_klines_file` or `load_universe` rebound (`monkeypatch.setattr`, `mock.patch`, `setattr`, plain assignment, `del`), and refuses `__class__` and `__dict__` for every file. Four of the names (`seal`, `load_klines`, `load_klines_file`, `load_universe`) are watching wrappers, and only the wrapper may stand under its name: putting the original back is refused, so is `protocol.seal = protocol.seal.__wrapped__` (there is none).
- Each of the four wrappers checks what it returns. Without both keys, a frame with a row at or after `VAULT.start` fails the test. The check judges the keys with the original `vault_open` and does not use the `seal` global of the module, so a seal swapped in the module dict does not blind it.
- The two file loaders refuse, with both keys in place, a path under the repository's `data/`, unless every file of the stack holds the new `vault-open` allowance (no file holds it today). Forging both keys (`open_vault` of `test_protocol`, a hand-written `OPEN` file with the variable) then opens nothing of the real vault from a test. `conftest.py` also gives every test an empty data root of its own (`OLSPRIME_DATA_DIR`), so a default `load_klines(symbol, interval)` never finds the real store under pytest. With both keys in place, a path outside the repository's `data/` (a copy of a vault file) is refused too unless some file of the stack besides `protocol.py` holds `seal`, `seal-patch` or `vault-open`, the files that may open a fake vault. A fake vault in a temporary tree stays testable, as `test_manifest.py` does.

Disarming is noticed. After every test, once its fixtures are gone, a check verifies that every patch `install` applied is still bound, that `olsprime.protocol` still wears the guard class and holds only the permitted values under the six names, that the allow-list (a read-only mapping of read-only mappings) is the one taken at the start, and that the tripwire's own functions are the originals. A breach ends the session (`pytest.exit`, return code 3): nothing after it is trustworthy. A `MonkeyPatch.undo()` of the session patches, run from a test, trips at once, at the restore of the module class (the last patch to go), and not only at the teardown. At the end of the session the guard class is taken off before the patches are undone. The static guard flags, outside the files that hold `disarm`: `__globals__`, `__wrapped__`, `__closure__`, `__code__`, `__bases__`, `__base__`, `__mro__`, `__subclasses__`, `.undo(`, `reload`; any import of `tests.conftest`, `tests.tripwire`, `tests.allowlist`; the names `open_vault`, `vault_key_file`, `OPEN_ENV` and the string `OLSPRIME_OPEN_VAULT` (also built by concatenation) as `seal-patch`; `runpy`, `zipimport`, `gc` and `ctypes` as banned imports; `attrgetter`, `methodcaller`, `__getattribute__`, `__builtins__` and a bare `getattr` passed as a value as `dynamic`; and `getattr`, `vars` and `__dict__` on any module of `olsprime.data` as `dynamic`.

**The plain statement.** Every guard that runs inside the process can be taken down from inside the process. These guards stop accidents and careless code, and the ordinary routes an honest model author or an automated agent might stumble into. They do not stop sabotage: someone who wants to read the vault and is willing to write code for it can. The structural fix is not another guard but the physical seal of iteration 3: the vault rows in a separate tree that the process cannot read without both keys, outside the repository's `data/`.

Remaining limits, stated plainly:
- **It runs only under pytest.** Scripts, notebooks and the CLIs are outside it, and the static guard alone covers them.
- **In-process disarm in general.** A wrapper's `__closure__` holds the original function (`pl.scan_parquet.__closure__` gives it back; `pl.read_parquet.__closure__` does not, because polars' `read_parquet` calls the wrapped `scan_parquet`. `load_klines.__closure__` gives back the real loader, whose `__globals__` then reach the real module). Frame objects, `gc`, `ctypes` and the `__code__` of a wrapper can be rewritten. A disarm that is undone before the test ends (a `MonkeyPatch.undo()` in a `finally`) leaves no trace for the teardown check. Statically these are flagged (`disarm`, `import:gc`, `import:ctypes`); at run time they are not stopped.
- **The cython types.** `pyarrow.dataset.FileFormat`, `ParquetFileFormat`, `FileSystemDataset` and the dataset factories are cython types that cannot be patched in place: they are wrapped by a subclass, and the original is one attribute away (`ParquetFileFormat.__bases__[0]`, whose `make_fragment(...).to_table()` reads). `Fragment.to_table` itself is a cython method. The static guard flags `__bases__`, `__base__`, `__mro__` and `__subclasses__`; the run-time tripwire does not stop it.
- **A pyarrow `NativeFile` has no path.** `LocalFileSystem().open_input_file(path)` returns a handle with no `.name`, so a reader handed that handle cannot be tied to a path under `data/`. It is judged by the caller rule alone, so it trips from a file without `reader` and passes from a file that holds it. Wrapping the cython file system methods is left for iteration 3.
- **Below the Python entry points.** A read through `pyarrow._parquet` or `pyarrow.lib` directly, through `polars._plr` (for instance `PyLazyFrame.new_from_parquet`, polars' own analogue of `pyarrow._parquet`), through `ctypes`, through a subprocess, or with the `open` built-in and a parser written by hand, passes the run-time tripwire. The static guard flags the imports of `ctypes` and `subprocess` and the names of the readers, not a hand-written parser.
- **Forged keys, partly.** Forging both keys opens a fake vault in a temporary tree, by design. It no longer opens the repository's `data/` through the loaders, and a copy of a vault file outside `data/` loaded with both keys forged trips unless a file of the stack holds `seal`, `seal-patch` or `vault-open`. A file that does hold `seal-patch` (`conftest.py`, `test_manifest.py`, `test_tripwire.py`) can still open a fake vault, by design. Its exemption is transitive through its helpers: a helper imported from such a file (`from tests.test_tripwire import run_source`) puts that file on the stack of the call, and the loader's exemption covers the caller. The static guard closes that route as `disarm`: it flags any import of `tests.test_tripwire` or `tests.test_manifest`, and from `tests.test_protocol` of anything but the data builders `bars`, `store`, `utc`, `access_lines` and `open_vault` (the last is a `seal-patch` of its own). Every test file imports from `test_protocol` that way today, so nothing was broken. At run time the case is a limit, and a copy of a vault file outside `data/` read without the loaders is judged by the `reader` rule alone (the loaders seal it). A reader that does not go through the loaders (a direct `pl.read_parquet` of a vault file) is judged by the `reader` rule alone, as before.
- **A raw reader from an allowed module, partly.** A module that holds `raw-reader` can export a function that returns raw rows; a test that holds `raw-reader` too can call it. The all-frames rule closes the case of a test that does not hold it.
- **`runpy` and the like.** `runpy` is a banned import, but a file that holds the `import:runpy` allowance (`test_tripwire.py`, which runs its evasion snippets as files) can run any file; the file it runs is outside `python/` and so holds no allowance of its own.
- **`zipfile.Path` over the standard library's own readers.** Judged at the first frame outside the standard library; an `importlib.metadata` read of a zip outside `data/` is not judged at all (see above).
- **`zipimport` on a copy.** `zipimporter` is judged by the path rule only, so it trips on an archive under `data/` and not on a copy of an archive outside `data/` (`zipimporter(copy).get_data(member)`). The static guard bans the import of `zipimport`; at run time the copy passes, like a copy read with a hand-written parser.
- **Child processes have no tripwire.** `multiprocessing` and `concurrent.futures.ProcessPoolExecutor` are not banned: the bench uses a process pool, through `load_klines`. A child process does not run `conftest.py`, so none of the run-time guards stand in it. What it reads through `load_klines` is sealed in the child, and the child inherits the empty test data root; anything else it reads is judged by the static guard only.
- **No test needs the real store, and CI is the evidence.** GitHub Actions runs the whole suite (`uv run --locked pytest`, `.github/workflows/ci.yml`) on runners that have no `data/` at all. No test skips or branches on whether real data exist: the only skips in `python/tests` are `test_decision.py` (a Rust source file), `test_stats.py` (the slow flag) and a symbolic-link check in `test_tripwire.py`, and none tests a `.exists()` on a data path.
- A reader captured before `conftest.py` is loaded (by a plugin, say) is not wrapped.
- `pandas`, `torch` and `duckdb` are not installed in this environment, so those wrappers are untested against the real libraries.
- Until stage 1 of the physical seal runs (below), the vault rows sit in the same files as the rest, and in the quality reports of the two pairs that have them (see above).

**The physical seal (iteration 3).** The vault months leave `data/` in two stages, with `uv run python -m olsprime.vault {plan,move,encrypt,keygen,status}` (`python/olsprime/vault.py`).
- **Stage 1, out of `data/`.** `move` works in two phases.
  - First it rebuilds, into a staging folder (`data/parquet.vault-stage/`), every Parquet file that holds vault rows, from that pair's non-vault archives. Each rebuild is checked before any table is touched. Its sealed read must equal the old file's, frame for frame, schema included. Its stored row count must equal the old file's sealed count, so no vault row is left.
  - Then the old file and its quality report are hard-linked into the vault tree, and the new file is renamed over the old path. The path therefore always holds the old file or the new one.
  - The vault archives and their `.CHECKSUM` files are renamed into the tree once `is_verified` passes on each.
  - Nothing is deleted. A journal makes the move resumable, and `MANIFEST.json` lists every file with its size, SHA-256 and original path.
  - The vault tree is `$OLSPRIME_VAULT_DIR`, else `<parent of the repository>/ols-prime-vault`: outside the repository and on the same volume, so that renames and links work. A root inside the repository, nested with the data root or on another volume is refused.
- **Stage 2, encrypted.** `encrypt --recipient age1…` writes each file of the tree to `<name>.age`, and records each ciphertext's SHA-256. It uses age X25519 through `pyrage` 1.4.0, which binds Rust's `age` crate.
  - Plaintext goes only with `--remove-plaintext`, which is Abdou's to pass.
  - The identity (`keygen --out`, on a medium taken away) must never be stored on this machine.
  - `encrypt` cannot check that the recipient matches an identity Abdou holds. So run it first without `--remove-plaintext`, decrypt one file with the real identity, and only then remove plaintext.
- **Refusals.** `download_month`, the downloader's command and the archive downloads refuse a vault month (`VaultMonthRefused`) before any request. `build_parquet` leaves out every archive of a vault month, with a warning that names it. Fetching the vault months for the final evaluation will go through the vault module, a later step.
- **Guards.**
  - The runtime tripwire treats the vault root like the repository's `data/`. The static guard flags the names and strings that reach the vault tree (rule `vault-tree`) outside `vault.py`, `protocol.py`, `paths.py` and the test guards.
  - Comparing the tripwire's targets with the static guard's lists found 11 polars readers the static guard did not flag, such as `read_parquet_metadata`, `scan_iceberg` and `read_lines`. They are flagged now.
- **Limits, in plain words.**
  - The vault months are public on data.binance.vision. The seal stops our own pipeline and our agents, not a person who downloads them.
  - Until stage 2, the vault tree is plaintext, outside `data/`, and guarded only in-process.
  - Until the identity leaves this machine, stage 2 is only as safe as that file.
  - The tool reads no vault row, but the process that runs it sees file names, sizes and the Parquet footers' statistics. A table whose footer lacks statistics is not scanned for stray vault bars.
- **Status, 2026-10-29: stage 0.** `plan` found 36 archives (788 kB), only BTCUSDT and ETHUSDT, 1 h and 1 d, 2026-01 → 2026-09, all verified. It found 4 tables holding 13,650 vault rows, and no other table with a bar at or after 2026-01-01. `plan --verify` rebuilt the four tables in a temporary folder and found every sealed read identical. The real move runs after the audit.

**Quirks to handle downstream.** The quality report (`{SYMBOL}.quality.json` beside each Parquet file) counts them.
- **Timestamps.** From 2025-01-01, spot files use microseconds. The parser reads the unit from each value's magnitude and stores milliseconds UTC.
- **Off-grid bars.** The 1 h series have 43 of them, from 2018-02-09 09:28 to 2018-02-11 03:28. They are shifted by about 28 minutes after the largest outage: 32 bars missing from 2018-02-08 01:00.
- **Bad close times.** The 1 h series have 15 bars whose close time is not open + 1 h − 1 ms:
  - 12 end early, from 1 second to 60 minutes before the hour, around outages and maintenance windows;
  - the bar of 2020-12-21 14:00 closes 12.7 minutes before it opens;
  - the last off-grid bar (2018-02-11 03:28) closes at 04:00:00.999, and the bar of 2017-09-06 15:00 closes 1 ms late;
  - four of the 15 have volume 0.

  So a bar's values count as final at `known_at = max(close_time, open_time + interval − 1 ms)`, which `load_klines` adds, and never at the stored `close_time` alone. The vault seal cuts on `known_at`, and any Matrix entry built from a bar must use it as its time.
- **The last gap** in the 1 h series starts on 2023-03-24 at 13:00.
- **Ticker reuse.** `LUNAUSDT` is the old LUNA until 2022-05-13 and the new LUNA from 2022-05-31. Cut every series at its gaps before computing returns. The survey above lists the likely cases among the members.

**Hourly panel (iteration 3).** `uv run python -m olsprime.features.panel --summary` builds the panel that Matrix reads (`python/olsprime/features/panel.py`) in memory, and writes `runs/<UTC date>-panel/summary.json`. On 2026-10-29:
- **Runtime.** 12 to 41 s, depending on the load of the machine, with peak memory near 2 GB. The command does not measure memory on Windows: `peak_rss_mb` is null there.
- **Rows.** Rows exist only in the months when a pair is a member of the universe: 2,273 member months, 1,655,881 hourly rows from 2017-10 to 2024-12.
- **Pairs.** 233 distinct pairs. The other 41 of the 274 members first join in 2025, so they have no row in the development period. Months have between 2 members (1,440 rows, 2017-11) and 30 (22,320 rows).
- **The cut.** Bars come through `load_klines`, which keeps the vault out. They are then cut to `known_at < 2025-01-01` before features and labels, so a label that needs a 2025 bar is null. The validation year's bars are loaded into memory and dropped there: none reaches a feature, a label or a model.
- **Labels.** 1,631,784 rows are labelled (24,097 have no exit bar), and 1,591,462 are complete (every feature and the label). The complete rows split into down 29.4 %, neutral 43.6 % and up 27.1 %.
- **The sub-panel** of the panel's future-invariance test: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, LTCUSDT, TRXUSDT, LINKUSDT, MATICUSDT and DOGEUSDT.
- **Months.** A row's month is the month of its bar's open, after audit pass 1a. Before, it was the month of the decision time, which a late close at a month's last hour would move into the next month. Rebuilt with the new rule, the panel has the same fingerprint: no member's bar closes late at a month's last hour.
- **Fingerprint** (SHA-256 over every column): `e64b39d8784af6039dc4810a1247fd957af94a8fe76240563855c45ba0442508`.
- **Files.** Nothing is cached in Parquet, so only `olsprime.protocol` opens a file.
- **License.** The panel is derived from Binance Vision, so it keeps CC BY-NC-SA 4.0: non-commercial, never redistributed.

## 2. Reference parameters

These are not datasets. They are the venue figures that `crates/backtest` hard-codes. A `scout` agent checked them on 2026-10-25 through a summarizing fetch tool, so re-read the official page before relying on any one of them. A second `scout` re-read the fees on 2026-10-27, before Report n°1 used them: Kraken is confirmed at the source, and Coinbase still rests on secondary sources (`docs/PROTOCOL.md`, change log).

| Parameter | Value in the code | Source | Check on 2026-10-25 |
|---|---|---|---|
| Binance spot, VIP 0, maker and taker | 0.10 % / 0.10 %; 0.075 % with BNB | [binance.com fee schedule](https://www.binance.com/en/fee/schedule) | matches |
| Coinbase Advanced, entry tier | 0.25 % / 0.50 % | coinbase.com (every fee page answered HTTP 403, on two passes) | **unverified at the source**. Two news reports give 0.25 % / 0.50 % for the EU from 2026-09-16, with the first tier cut from 25,000 to 10,000 USD of 30-day volume. The 0.40 % / 0.60 % of an older help page predates that change. |
| Kraken Pro, tier 1 | 0.40 % / 0.80 % | [kraken.com fee schedule](https://www.kraken.com/features/fee-schedule) | matches; the page shows no separate EU schedule |
| Bitvavo, entry tier | 0.15 % / 0.25 % | Bitvavo's API documentation, and a search snippet of its fee page | added on 2026-10-27, read through those two sources only: the fee page itself was not read |
| BTCUSDT filters | `NOTIONAL` minNotional 5 USDT; `LOT_SIZE` stepSize 0.00001 | [`exchangeInfo`](https://data-api.binance.vision/api/v3/exchangeInfo?symbols=%5B%22BTCUSDT%22,%22ETHUSDT%22%5D) | matches |
| ETHUSDT filters | `NOTIONAL` minNotional 5 USDT; `LOT_SIZE` stepSize 0.0001 | same | matches |

**MiCA.** ESMA's interim register of crypto-asset service providers ([CSV](https://www.esma.europa.eu/sites/default/files/2024-12/CASPS.csv)) lists these venues as authorized and passported to France, as read on 2026-10-25:

| Venue | Entity | Authority | Date |
|---|---|---|---|
| Coinbase | Coinbase Luxembourg S.A. | CSSF | 2025-06-20 |
| Kraken | Payward Europe Solutions Limited | Central Bank of Ireland | 2025-06-25 |
| Bitstamp | Bitstamp Europe S.A. | CSSF | 2025-05-15 |
| Bitvavo | Bitvavo B.V. | AFM | 2025-06-26 |

Caveats:
- The authorization dates are our reading of the file's first date column. The Coinbase, Bitstamp and Bitvavo rows also carry a later date in the `ac_lastupdate` column (2026-01-22, 2025-05-16 and 2025-06-30), which is the row's last update, not its authorization.
- For Kraken the entity matters: Payward Global Solutions Limited is not passported to France.
- No Binance entity is in the register. That Binance stopped serving French residents on 2026-07-01 rests on press reports of its e-mail to users; we found no primary page.

## 3. Not used yet

Each of these gets an entry above before its first download:
- the news source for H7 (phase 5): its license must allow point-in-time research use;
- the market data of the venue chosen for the live loop (phase 6).
