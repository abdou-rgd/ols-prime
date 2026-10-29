# Brief — The physical seal (iteration 3)

- **Agent:** `data-engineer` (sonnet, high).
- **Runs in parallel with:** a `rust-engineer` finishing the Matrix core (`crates/`), an `ml-engineer` finishing the H1 adversary (`adversary.md`), and later an `ml-engineer` on the Matrix arms (`matrix-arms.md`). None of them touches your files.
- **Motto:** we should be more ambitious, in how hard the vault is to reach, never in what we claim the seal stops.

## Goal

Today the vault months sit in the same files as everything else:
- the raw archives of BTCUSDT and ETHUSDT for 2026-01 → 2026-09, under `data/raw/`;
- their rows, in the same two pairs' Parquet files, 1 h and 1 d, under `data/parquet/`.

By file name, on 2026-10-29: only BTCUSDT and ETHUSDT hold archives of vault months, 1 d and 1 h, nine months each (2026-01 → 2026-09). That is 36 archives and their 36 `.CHECKSUM` files. No other pair of the 568 with 1 d archives, or of the 274 with 1 h archives, holds one. `plan` must still find every vault archive from the names alone, not from this list.

Guards inside the process stand between a test and those rows, and four rounds of audit showed that such guards can be taken down from inside the process. Iteration 2's lesson: make the vault unreadable, not watched.

Move the vault out of `data/`, in two stages:
1. **Out of `data/`.** Rename the vault archives and the Parquet files that hold vault rows into a vault tree outside the repository. Rebuild those Parquet files from the remaining archives, and check that every sealed read is unchanged.
2. **Encrypted.** Encrypt the vault tree to an [age](https://age-encryption.org) recipient whose private key never touches this machine. Without that key, no process can read the vault, whatever it does to the guards.

Read first:
- `docs/PROTOCOL.md`, "Splits" and the change log, on the two keys;
- `docs/DATA.md`, the guards and "The plain statement";
- `python/olsprime/protocol.py`, `paths.py`, `data/binance_vision.py`, `manifest.py`;
- `python/tests/tripwire.py`, `conftest.py`, `allowlist.py` and the static guard's test.

## Never open the vault

Never set `OLSPRIME_OPEN_VAULT`, never write `runs/vault/OPEN`, never parse a vault archive or a vault row. Moving a file, and hashing its bytes the way `is_verified` already does, are not reads of rows. Proofs that something works run on **fake trees only**, under temporary directories, with throwaway age identities.

## Deliverables

1. **`python/olsprime/vault.py`**, with a command `uv run --no-sync python -m olsprime.vault {plan,move,encrypt,keygen,status}`.
   - **Vault root.** `$OLSPRIME_VAULT_DIR` when set, else `<parent of the repository>/ols-prime-vault`. It is outside the repository, so a `git add`, a glob of `data/` or a test's empty data root never reaches it. Refuse a vault root inside the repository or inside the data root.
   - **`plan`.** A dry run that changes nothing. It lists:
     - every archive under `data/raw/` whose month, read from the file name, is at or after `VAULT.start`, with its size, plus its `.CHECKSUM`;
     - every Parquet file that holds vault rows, found without reading a row: a pair with a vault archive, cross-checked with the stored count (`protocol.stored_row_count`) against the sealed count;
     - its `.quality.json`;
     - the rebuild each file would get.
   - **`move`** (stage 1). For each affected pair and interval:
     1. Rebuild the Parquet file from its non-vault archives into a temporary file beside it.
     2. Check that the old file's sealed read equals the new file's, frame for frame: `protocol.load_klines_file` on both, same rows, same values, same dtypes. Also check that the new file's stored row count equals the old file's sealed row count, so no vault row is left in `data/`.
     3. Rename the old Parquet file and its quality report into the vault tree, mirroring their paths. Then rename the new file into place, with the quality report the rebuild wrote.
     4. Rename every vault archive and its `.CHECKSUM` into the vault tree, mirroring paths, once `is_verified` passes on it. Record its SHA-256.
     5. Write a manifest into the vault tree: every file, its size, its SHA-256 and its original path. Then print a summary.
     - Everything is a rename, on the same volume. Nothing is deleted.
     - A journal makes the move resumable: a crash at any step leaves either the old or the new file in place, never neither. A second run finishes the job or does nothing.
   - **`encrypt --recipient age1…`** (stage 2).
     - It encrypts every plaintext file of the vault tree to `<name>.age` and records the ciphertext SHA-256 in the vault manifest.
     - It removes a plaintext file **only** with `--remove-plaintext`, after its ciphertext is written, flushed and re-hashed. That flag is for Abdou, who runs this step: never pass it outside a test.
   - **`keygen --out PATH`.**
     - It writes a new age identity to PATH and prints only its public recipient.
     - It refuses a PATH inside the repository, the data root or the vault root, and a PATH that exists.
     - It is for Abdou, to run with PATH on a medium they take away, such as a USB key. Never run it outside a test.
   - **`status`.** Where the vault stands: stage 0 (in `data/`), stage 1 (moved, plaintext) or stage 2 (encrypted, plaintext removed). It also gives file counts and bytes, and any vault month still under `data/`.
   - **Encryption library.**
     - Use `pyrage` (age X25519, from Rust's `age` crate) if it is maintained and ships wheels for Windows (win_amd64) and Linux for Python 3.12.
     - Check its current version and its wheels on PyPI's JSON API, and its repository's activity, before adding it with `uv add`. Record what you checked.
     - If it does not qualify, stop and report: do not write cryptography by hand.
2. **The open path, for the final evaluation only.** With both keys, `protocol.load_klines` must still return the vault rows: after stage 1, from the vault tree's Parquet files; after stage 2, decrypted in memory with an identity named by `$OLSPRIME_VAULT_IDENTITY`.
   - Every such read is logged in `ACCESS.log`, as today.
   - Without both keys nothing reads the vault tree. Without the identity, stage 2 opens nothing even with both keys forged.
   - This path is tested on fake trees only.
3. **Downloads and rebuilds refuse the vault.**
   - `binance_vision.download_month` and the command refuse a month at or after `VAULT.start`. The final evaluation will fetch those months through the vault module, a later step that is out of scope here. Say so in the error message.
   - `build_parquet` leaves out any archive of a vault month, with a warning that names it.
   - Check that `universe` and `survey` still work.
4. **The guards cover the vault tree.**
   - The runtime tripwire treats a path under the vault root like a path under the repository's `data/`: every wrapped reader trips on it.
   - The static guard flags the names that reach the vault tree outside `vault.py` and `protocol.py`.
   - Keep every existing guard. They become the second line of defence. Their path rules are not removed in this iteration.
5. **The manifest.** `olsprime.manifest` also reports the vault tree: its stage, file count, bytes, and the digest of its own manifest. It reads no vault row.
6. **Tests** in a new file `python/tests/test_vault.py`, on fake trees: fake monthly archives from 2025-11 to 2026-02, Parquet files built from them, a throwaway identity.
   - `plan` lists exactly the vault archives and files.
   - After `move`:
     - no archive of a vault month is under `data/`;
     - the sealed reads are identical before and after;
     - the moved files are byte-identical in the vault tree;
     - the new Parquet files hold no vault row;
     - the quality reports no longer count vault bars.
   - A crash at each step of `move`, injected, still lets a second run finish without losing a file.
   - `encrypt` and decrypt round-trip byte for byte. A wrong identity fails, and so does a tampered ciphertext.
   - Plaintext is removed only with the flag.
   - The open path needs both keys and, at stage 2, the identity. Forged keys without the identity read nothing.
   - The downloader and `build_parquet` refuse vault months.
   - The tripwire trips on a read under the vault root.
7. **Dry run on the real tree.** Run `plan` on the real store and put its output in your report. **Do not run `move` on the real store.** The orchestrator runs it after the audit has reviewed the tool.
8. **Docs input.** In your report, the paragraphs for `docs/DATA.md` (the physical seal, the two stages and their limits) and for `docs/PROTOCOL.md` (how the vault opens now).
   - Include the limits in plain words. The vault months are public on data.binance.vision: the seal stops our own pipeline and our agents, not a person who downloads them.
   - Until stage 2, the vault tree is plaintext, outside `data/` and guarded only in-process.
   - Until the identity leaves the machine, stage 2 is only as safe as that file.

## Constraints

- **Files.**
  - `python/olsprime/vault.py` (new), `python/olsprime/protocol.py`, `python/olsprime/paths.py`, `python/olsprime/data/binance_vision.py`, `python/olsprime/manifest.py`.
  - `python/tests/test_vault.py` (new), `python/tests/tripwire.py`, `python/tests/allowlist.py`, `python/tests/conftest.py`, and the static guard and its corpus (find them under `python/tests/`).
  - `python/pyproject.toml` and `uv.lock`, for the new dependency.
  - Ask before touching anything else.
- **Python.** Use `uv run --no-sync` for tests: a `rust-engineer` may be rebuilding the extension.
  - Never let uv sync the environment. A plain `uv add` or `uv sync` rebuilds the Rust extension from the `rust-engineer`'s sources while they are being edited.
  - Add the dependency with `uv add --no-sync pyrage==<version>`, which writes `pyproject.toml` and `uv.lock` only. Then install it with `uv pip install pyrage==<version>`. The orchestrator syncs once the Rust work has landed.
  - If a file is locked, wait and retry, and kill nothing.
- **The panel.** `features/panel.py` reads its bars through `protocol.load_klines`. `tests/test_panel.py` must stay green, and the panel's fingerprint must not change: `uv run --no-sync python -m olsprime.features.panel --summary --no-write` prints it, `e64b39d8784af6039dc4810a1247fd957af94a8fe76240563855c45ba0442508`.
- **Scripts.** Write scripts with a file tool, never through shell heredocs.
- **Scratch.** `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\seal\`.
- **Git.** Run no git command that changes state, and do not commit.

## Done when

- The module, the guards and the tests are in place, and the whole suite is green: `cd python && uv run --no-sync pytest`.
- `plan` has run on the real store, and its output is in your report.

## Report (short)

- The design in ten lines.
- What `plan` says about the real store.
- The library check.
- The docs paragraphs.
- What is not done.
- One bolder next step.
