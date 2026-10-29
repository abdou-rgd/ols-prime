# Brief — Seal fixes after audit pass 1b (iteration 3)

- **Agent:** `data-engineer` (sonnet, high).
- **Runs after:** the real `vault move` (stage 1), which the orchestrator runs on the audited code of commit `767e2c0`. Do not start before the orchestrator says the move is done.
- **Runs in parallel with:** an `ml-engineer` on the Matrix arms (`python/olsprime/models/matrix.py`, `bench/panel_invariance.py`, `bench/report2.py`) and the `rust-engineer` in `crates/ols-core`. Touch none of their files.

## Goal

Close the four Minor findings of audit pass 1b on the physical seal. None blocked stage 1, and one must be closed before stage 2. Read first:
- `python/olsprime/vault.py` and `python/tests/test_vault.py`;
- `docs/DATA.md`, "The physical seal (iteration 3)";
- `docs/PROTOCOL.md`, "Splits", the paragraph on the physical seal.

## The findings

1. **A file lock gives a traceback, not a refusal.** `vault.py:582`, the `os.replace` in `_swap_in`, raises `PermissionError` when another process holds the Parquet file open. `main()` (`vault.py:964`) catches `VaultError` only, so the command exits 1 with a traceback.
   - The auditor showed that nothing is lost and that `move` resumes to the byte-identical end state.
   - Fix: catch `OSError` around each step of `move`. Print one refusal line saying which file was locked, that nothing was lost, and that running `move` again resumes. Exit 2.
   - Test it on a fake tree, with the lock simulated by monkeypatching `os.replace` to raise `PermissionError` once.
2. **`encrypt --remove-plaintext` deletes plaintext without a decrypt round-trip.** `vault.py:798-801` re-hashes the ciphertext but never checks that it decrypts to the plaintext's SHA-256. A valid but wrong `age1…` recipient, such as a stale key, would make the local copy unrecoverable.
   - Add a subcommand `verify-decrypt`. Abdou runs it once, with the identity on a medium he mounts (`OLSPRIME_VAULT_IDENTITY`). It decrypts each `.age` file in memory, compares the SHA-256 with `files[].sha256` in `MANIFEST.json`, and on success writes a stamp into the manifest: the time, the count, and the digest of the identity's public recipient, never the identity.
   - `encrypt --remove-plaintext` then refuses unless the stamp covers every encrypted file.
   - Test both on fake trees with throwaway identities, as the suite already does: a right recipient passes, a wrong one is refused, and removal without the stamp is refused.
   - Never generate, read or print a real key. `keygen` and `encrypt` on the real store are Abdou's.
3. **`months_still_in_data` counts items, not months.** `tree_summary` (`vault.py:883`) returns 40 on the real store: 36 archives plus 4 tables.
   - Replace the field with `archives_still_in_data` and `tables_still_in_data`.
   - Update `status`'s text and `manifest.py`'s print line.
   - Update the tests.
4. **The journal pins the archive's SHA-256 at plan time, but not its CHECKSUM file's.** `vault.py:553` writes `checksum_sha256: None`, and `vault.py:656-660` hashes whatever is there at move time.
   - Record the CHECKSUM file's SHA-256 in `_new_journal`, and refuse at move time if it changed, as is done for the archive.
   - Test with an edited CHECKSUM file.
   - A journal written before this change, with `None`, must still resume; the real store's move is already finished.

## Constraints

- **Files:** `python/olsprime/vault.py`, `python/olsprime/manifest.py` (its print line only) and `python/tests/test_vault.py`. Ask before touching any other file. The orchestrator updates the docs.
- **The vault.**
  - Never read a vault row.
  - Never set `OLSPRIME_OPEN_VAULT`, and never write `runs/vault/OPEN`.
  - Never set `OLSPRIME_VAULT_IDENTITY` to anything but a throwaway test identity.
  - Never run `move`, `encrypt`, `keygen` or `verify-decrypt` on the real store. `plan` and `status` are fine.
- **Python:** use `uv run --no-sync`. Never rebuild the extension.
- **Scripts:** write them with a file tool, never through shell heredocs. Put scratch files in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\ddf94fa1-adbc-447c-8cf7-b41f53dfa29f\scratchpad\seal-fixes\`.
- **Git:** no git command that changes state.

## Done when

- `uv run --no-sync pytest tests/test_vault.py tests/test_manifest.py -q` passes.
- `uv run --no-sync python -m olsprime.vault status` on the real store shows the new fields and stage 1.

## Report (short)

- Each fix, with its file and line.
- The new tests.
- The real `status` output.
- One bolder next step.
