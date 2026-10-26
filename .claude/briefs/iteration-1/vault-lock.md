# Brief — seal the vault in code (iteration 1, audit finding M1)

**Role.** `data-engineer`. **Scratch folder:** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\data-engineer\` (yours alone).

## Why

`docs/PROTOCOL.md` reserves 2026-01 → 2026-10 as the vault: "opened once, at the end, for all final models together". Today nothing enforces it:
- `build_parquet` (`python/olsprime/data/binance_vision.py:420-448`) writes every verified archive into one Parquet file per pair. On disk, each 1 h file holds 6,552 vault rows and each 1 d file 273, from 2026-01-01 00:00 UTC.
- `python -m olsprime.universe` defaults `--end` to the current month (`universe.py:324`), so a plain run ranks 2026-01 → 2026-10 on vault bars.
- `quality.py:223` and `universe.py:301` call `pl.read_parquet` directly.

Iteration 2 adds feature, label and training code. The lock must exist before any of it, so that a careless reader gets an error or a sealed frame, never vault rows.

## Deliverables

1. **`python/olsprime/protocol.py`**, the single source of truth for the splits.
   - `Period(start, end)`: a frozen dataclass of UTC datetimes, half-open `[start, end)`.
   - `DEVELOPMENT = Period(2017-08-01, 2025-01-01)`, `VALIDATION = Period(2025-01-01, 2026-01-01)`, `VAULT = Period(2026-01-01, 2026-11-01)`.
   - `class VaultSealed(RuntimeError)`.
   - `vault_open() -> bool`. It is true only if the environment variable `OLSPRIME_OPEN_VAULT` equals `"1"` **and** the file `<runs>/vault/OPEN` exists. A person writes that file by hand at the final evaluation, with the date and the ids of the final models. Neither key alone opens the vault.
   - `seal(frame, *, time_col="open_time", caller) -> pl.DataFrame`.
     - Raise `TypeError` unless `time_col` has dtype `pl.Datetime("ms", "UTC")`, so that an integer or string column can never be compared by mistake.
     - Sealed: return the rows with `time_col < VAULT.start`.
     - Open: return every row, and append one line to `<runs>/vault/ACCESS.log`: UTC time, caller, rows returned, vault rows among them.
   - `load_klines(symbol, interval, *, data_dir=None, columns=None) -> pl.DataFrame`.
     - Read `parquet_path(...)`, then `seal(...)`. Always read `open_time` for the seal, even when `columns` leaves it out, and drop it afterwards if it was not asked for.
     - Add a column `known_at`: the time a bar's values are final, `max(close_time, open_time + interval − 1 ms)`. The stored `close_time` alone is wrong for 15 bars of the 1 h series, and once even earlier than the open (`docs/DATA.md`, "Bad close times"); `open_time + interval − 1 ms` alone is 1 ms early for the bar of 2017-09-06 15:00. Taking the later of the two is never early. Compute it only when `close_time` and `open_time` are both available (read them for it); document it in the docstring.
2. **`python/olsprime/paths.py`**: `runs_dir()`, `$OLSPRIME_RUNS_DIR` when set, else `<repo>/runs`, like `data_dir()`.
3. **Every reader goes through `load_klines`.**
   - `universe.run` reads the daily files through it.
   - `universe.main`: the default `--end` becomes `Month(2025, 12)`; fix the help text and the module docstring example (`--end 2026-10`).
   - `build_universe` raises `VaultSealed` when `end >= Month(2026, 1)` while the vault is sealed. With the vault open it behaves as today.
   - The quality CLI (`quality.py` `main`) reads through `load_klines`, so its report on a stored file is sealed too.
   - `build_parquet` keeps writing all verified archives, and its `quality.json` keeps covering every row. Downloading and integrity checks are not reads for modelling: they compute structural checks only (gaps, duplicates, ordering, OHLC consistency, close times).
4. **Tests**, offline, with `tmp_path`. Point `OLSPRIME_RUNS_DIR` at a temporary folder with `monkeypatch`: a test must never create `runs/vault/` in the repository.
   - `seal` keeps the bar of 2025-12-31 23:00 and drops the bar of 2026-01-01 00:00, for both 1 h and 1 d fixtures.
   - `seal` raises `TypeError` on an integer time column.
   - The vault stays sealed with the environment variable alone and with the file alone; with both, every row comes back and `ACCESS.log` gains exactly one line per read.
   - `load_klines` with `columns=["close"]` still seals and returns only `close` (plus `known_at` if you decide it always comes; say which).
   - `known_at` on five fixture bars: on the grid, an early close, a close before the open, a close 1 ms late, an off-grid bar.
   - `build_universe(..., end=Month(2026, 1))` raises `VaultSealed` when sealed; the CLI default is 2025-12.
   - A static guard: no module under `python/olsprime/` other than `protocol.py` contains `read_parquet(` or `scan_parquet(`, and `read_klines_zip(` appears only in `binance_vision.py`. The guard reads the source files; it is not a mock.
   - A sync guard: `docs/PROTOCOL.md`'s split table names the same months as `DEVELOPMENT`, `VALIDATION` and `VAULT`.
5. **A real-data check, read-only.** `load_klines("BTCUSDT", "1h")` on the files in `data/` returns 79,837 − 6,552 = 73,285 rows, the last at 2025-12-31 23:00 UTC; the 1 d file returns 3,332 − 273 = 3,059 rows. Give the numbers you get.

## Constraints

- Own only `python/olsprime/` and `python/tests/`. Do not edit `docs/` (the orchestrator updates `PROTOCOL.md` and `DATA.md` from your report), the Rust crates, or anything under `data/`.
- No new dependency. No network: no download, no listing.
- `cd python; uv run --locked pytest` must pass: 35 tests today, plus yours.
- Do not commit.

## Report (short)

The API as built, the tests with counts, the real-data numbers, anything you decided differently from this brief and why, and one bolder next step.
