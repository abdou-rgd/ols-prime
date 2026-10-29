"""The physical seal: plan, move, encrypt, the open path, and the refusals, all on fake trees.

Every tree here lives under ``tmp_path``: fake monthly archives from 2025-11 to 2026-02 (microseconds from
2025-01, as on data.binance.vision), Parquet files built from them as they were before the seal, a duplicate bar,
a pair listed late, a pair delisted before the vault, a pair with a vault archive and no table. Identities are
throwaway. Nothing here touches the real store or the real vault tree.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyrage
import pytest

from olsprime import manifest, protocol, universe, vault
from olsprime.data import quality, survey
from olsprime.data.binance_vision import (
    Month,
    VaultMonthRefused,
    archive_relpath,
    build_parquet,
    download_archives,
    download_klines,
    download_month,
    parquet_path,
    parse_klines_csv,
    run,
)
from olsprime.paths import REPO_ROOT, default_vault_dir
from olsprime.protocol import VAULT, VaultSealed, load_klines
from tests.fakes import DAY_MS, HOUR_MS, FakeBucket, klines_csv, month_rows, zip_archive
from tests.test_protocol import access_lines, open_vault

NOV, DEC, JAN, FEB, MAR = Month(2025, 11), Month(2025, 12), Month(2026, 1), Month(2026, 2), Month(2026, 3)
STEP = {"1d": DAY_MS, "1h": HOUR_MS}
# symbol -> interval -> archive months
LAYOUT: dict[str, dict[str, list[Month]]] = {
    "AAAUSDT": {"1d": [NOV, DEC, JAN, FEB], "1h": [NOV, DEC, JAN, FEB]},  # duplicate bars across archive edges
    "CCCUSDT": {"1d": [DEC, JAN]},  # listed late: no archive for 2025-11
    "BBBUSDT": {"1d": [NOV, DEC]},  # nothing in the vault
    "DDDUSDT": {"1d": [Month(2025, 9), Month(2025, 10)]},  # delisted before the vault
}
ORPHAN = ("EEEUSDT", "1d", MAR)  # an archive of a vault month and no table at all
VAULT_PAIRS = {("AAAUSDT", "1d"), ("AAAUSDT", "1h"), ("CCCUSDT", "1d")}
PAIRS = [(symbol, interval) for symbol, by_interval in LAYOUT.items() for interval in by_interval]


def month_data(symbol: str, interval: str, month: Month) -> list[list[str]]:
    """Rows of one fake archive; AAAUSDT repeats the last bar of the previous archive at each edge."""
    unit = "us" if month >= Month(2025, 1) else "ms"  # the switch of data.binance.vision, read by magnitude
    price = 100.0 + 50 * list(LAYOUT).index(symbol) if symbol in LAYOUT else 7.0
    rows = month_rows(month, STEP[interval], unit=unit, price=price)
    if symbol == "AAAUSDT" and interval == "1d" and month in (DEC, JAN):
        before = month_rows(month.shift(-1), STEP[interval], unit=unit, price=price)
        rows = [before[-1], *rows]
    return rows


@functools.cache
def cached_zip(member: str, csv: bytes) -> bytes:
    """One zip per content, for the whole session: zip members carry the time they were written, and the
    crash test compares trees built at different seconds byte for byte."""
    return zip_archive(member, csv)


def write_archive(raw: Path, symbol: str, interval: str, month: Month, rows: list[list[str]]) -> None:
    name = f"{symbol}-{interval}-{month}"
    archive = cached_zip(f"{name}.csv", klines_csv(rows))
    path = raw / archive_relpath(symbol, interval, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive)
    path.with_name(path.name + ".CHECKSUM").write_text(f"{hashlib.sha256(archive).hexdigest()}  {name}.zip\n")


@dataclass(frozen=True)
class World:
    data: Path
    vault: Path


def make_world(root: Path, *, orphan: bool = True) -> World:
    """A store as it was before the seal: every archive in the cache, Parquet files built from all of them."""
    data, tree = root / "data", root / "vault"
    raw = data / "raw"
    for symbol, by_interval in LAYOUT.items():
        for interval, months in by_interval.items():
            frames = []
            for month in months:
                rows = month_data(symbol, interval, month)
                write_archive(raw, symbol, interval, month, rows)
                frames.append(parse_klines_csv(klines_csv(rows), symbol))
            concat = pl.concat(frames)
            report = quality.check_quality(concat, interval, symbol=symbol)
            bars = concat.unique(subset="open_time", keep="first", maintain_order=True).sort("open_time")
            path = parquet_path(data / "parquet", symbol, interval)
            path.parent.mkdir(parents=True, exist_ok=True)
            bars.write_parquet(path, compression="zstd", statistics=True)
            path.with_name(f"{symbol}.quality.json").write_text(json.dumps(report.to_dict(), indent=2))
    if orphan:
        symbol, interval, month = ORPHAN
        write_archive(raw, symbol, interval, month, month_data(symbol, interval, month))
    return World(data, tree)


def hashes(root: Path, *, skip: frozenset[str] = frozenset()) -> dict[str, str]:
    """``{relative path: SHA-256}`` of every file under ``root`` (empty if it does not exist)."""
    if not root.exists():
        return {}
    found = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in skip:
            found[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def sealed_reads(world: World) -> dict[tuple[str, str], pl.DataFrame]:
    return {pair: load_klines(*pair, data_dir=world.data) for pair in PAIRS}


def vault_files(world: World) -> dict[str, str]:
    """The files that ``move`` must put in the vault tree, by their path in the vault tree, with their SHA-256 before the move."""
    found = {}
    for rel, sha in hashes(world.data).items():
        parts = vault.archive_name_parts(Path(rel).name.removesuffix(".CHECKSUM")) if rel.startswith("raw/") else None
        if parts is not None and parts[2].start() >= VAULT.start:
            found[rel] = sha
        if rel.startswith("parquet/spot/klines/") and (Path(rel).stem.split(".")[0], Path(rel).parent.name) in VAULT_PAIRS:
            found[rel] = sha
    return found


BOOKKEEPING = frozenset({vault.MANIFEST_NAME, vault.JOURNAL_NAME})


# --- plan --------------------------------------------------------------------------------------------------


def test_plan_lists_exactly_the_vault_archives_and_files_and_changes_nothing(tmp_path):
    world = make_world(tmp_path)
    before = hashes(world.data)
    plan = vault.build_plan(world.data, world.vault)
    archives = {a.zip_rel for a in plan.archives}
    expected = {
        archive_relpath(symbol, interval, month)
        for symbol, by_interval in LAYOUT.items()
        for interval, months in by_interval.items()
        for month in months
        if Month.of(VAULT.start) <= month
    } | {archive_relpath(*ORPHAN)}
    assert archives == expected and len(plan.archives) == 2 + 2 + 1 + 1  # AAA 1d, AAA 1h, CCC 1d, and the orphan
    assert all(a.verified and a.checksum_bytes for a in plan.archives)
    assert {(i.symbol, i.interval) for i in plan.parquet} == VAULT_PAIRS
    assert all(i.vault_rows > 0 and i.sealed_rows + i.vault_rows == i.stored_rows and i.quality_rel for i in plan.parquet)
    assert plan.left_alone == [f"{ORPHAN[0]} {ORPHAN[1]}: vault archives but no Parquet file"]
    assert plan.anomalies == [] and plan.has_work
    item = next(i for i in plan.parquet if (i.symbol, i.interval) == ("CCCUSDT", "1d"))
    assert (item.rebuild_archives, item.first_month, item.last_month) == (1, DEC, DEC)
    text = vault.format_plan(plan)
    assert "dry run" in text and "AAAUSDT-1h-2026-02.zip" in text and "Anomalies: 0" in text
    assert hashes(world.data) == before and not world.vault.exists()  # a dry run


def test_plan_verify_rebuilds_in_a_temporary_folder(tmp_path):
    world = make_world(tmp_path)
    before = hashes(world.data)
    plan = vault.build_plan(world.data, world.vault)
    checks = vault.verify_rebuilds(plan)
    assert len(checks) == 3 and all(c.ok for c in checks), checks
    assert "verified in a temporary folder: OK" in vault.format_plan(plan, checks)
    assert hashes(world.data) == before and not world.vault.exists()


def test_plan_finds_vault_months_from_the_names_alone_even_in_a_pair_nobody_listed(tmp_path):
    world = make_world(tmp_path, orphan=False)
    write_archive(world.data / "raw", "ZZZUSDT", "1w", Month(2026, 9), month_data("DDDUSDT", "1d", Month(2026, 9)))
    plan = vault.build_plan(world.data, world.vault)
    assert archive_relpath("ZZZUSDT", "1w", Month(2026, 9)) in {a.zip_rel for a in plan.archives}


def test_plan_flags_an_archive_that_fails_its_checksum_and_move_refuses(tmp_path):
    world = make_world(tmp_path)
    broken = world.data / "raw" / archive_relpath("AAAUSDT", "1d", FEB)
    broken.write_bytes(broken.read_bytes() + b"x")
    plan = vault.build_plan(world.data, world.vault)
    assert any("does not match its CHECKSUM" in text for text in plan.anomalies)
    before = hashes(world.data)
    with pytest.raises(vault.VaultError, match="anomalies"):
        vault.move(world.data, world.vault)
    assert hashes(world.data) == before and not world.vault.exists()


def test_plan_flags_vault_bars_in_a_table_without_a_vault_archive(tmp_path):
    world = make_world(tmp_path, orphan=False)
    # BBBUSDT has no vault archive, but its table was written with a vault bar (by some other route).
    extra = parse_klines_csv(klines_csv(month_rows(JAN, DAY_MS)[:3]), "BBBUSDT")
    path = parquet_path(world.data / "parquet", "BBBUSDT", "1d")
    extra.write_parquet(path, compression="zstd", statistics=True)
    plan = vault.build_plan(world.data, world.vault)
    assert any("BBBUSDT 1d" in text and "footer" in text for text in plan.anomalies)
    with pytest.raises(vault.VaultError, match="BBBUSDT"):
        vault.move(world.data, world.vault)


@pytest.mark.parametrize("where", ["inside the repository", "inside the data root", "holding the data root"])
def test_the_vault_root_must_be_apart_from_the_repository_and_the_data(tmp_path, where):
    world = make_world(tmp_path)
    root = {
        "inside the repository": REPO_ROOT / "ols-prime-vault-test",
        "inside the data root": world.data / "vault",
        "holding the data root": tmp_path,
    }[where]
    with pytest.raises(vault.VaultError):
        vault.build_plan(world.data, root)
    with pytest.raises(vault.VaultError):
        vault.move(world.data, root)
    assert not (REPO_ROOT / "ols-prime-vault-test").exists()


def test_the_default_vault_root_is_beside_the_repository_and_outside_it():
    assert default_vault_dir() == REPO_ROOT.parent / "ols-prime-vault"
    assert not default_vault_dir().resolve().is_relative_to(REPO_ROOT.resolve())


def test_the_tool_refuses_to_run_with_the_vault_open(tmp_path, monkeypatch):
    world = make_world(tmp_path)
    open_vault(monkeypatch)
    with pytest.raises(vault.VaultError, match="open"):
        vault.build_plan(world.data, world.vault)
    with pytest.raises(vault.VaultError, match="open"):
        vault.move(world.data, world.vault)
    assert access_lines() == []  # and it read nothing


# --- move --------------------------------------------------------------------------------------------------


def test_move_takes_the_vault_months_out_of_data_and_changes_no_sealed_read(tmp_path):
    world = make_world(tmp_path)
    reads = sealed_reads(world)
    moving = vault_files(world)
    assert len(moving) == 6 * 2 + 3 * 2  # 6 archives and their CHECKSUM files, 3 tables and their quality reports
    old_quality = {
        pair: json.loads(parquet_path(world.data / "parquet", *pair).with_name(f"{pair[0]}.quality.json").read_text()) for pair in VAULT_PAIRS
    }
    result = vault.move(world.data, world.vault)
    assert not result.nothing_to_do and not result.resumed

    # No archive of a vault month, and no table row at or after the vault start, is left under data/.
    assert vault.scan_vault_archives(world.data / "raw") == ([], [])
    for pair in VAULT_PAIRS:
        path = parquet_path(world.data / "parquet", *pair)
        assert protocol.stored_row_count(path) == load_klines(*pair, data_dir=world.data).height
        assert protocol.stored_open_time_bounds(path)[1] < VAULT.start
    # The sealed reads are identical, frame for frame, dtypes included.
    after = sealed_reads(world)
    for pair, frame in reads.items():
        assert after[pair].schema == frame.schema and after[pair].equals(frame), pair
    # The moved files are byte-identical in the vault tree, at the mirrored path.
    assert moving
    for rel, sha in moving.items():
        assert hashes(world.vault)[rel] == sha, rel
    # The quality reports no longer count vault bars.
    for pair in VAULT_PAIRS:
        report = json.loads(parquet_path(world.data / "parquet", *pair).with_name(f"{pair[0]}.quality.json").read_text())
        assert report["rows"] < old_quality[pair]["rows"] and report["last_open_time"] < VAULT.start.isoformat()
    # The manifest of the tree: every file, size, hash, original path.
    listed = json.loads((world.vault / vault.MANIFEST_NAME).read_text())
    assert {e["path"]: e["sha256"] for e in listed["files"]} == moving
    assert all(e["original"] == f"data/{e['path']}" and e["bytes"] > 0 for e in listed["files"])
    assert listed["digest"] == vault.manifest_digest(listed)
    # Stage 1, and a second run has nothing to do.
    state = vault.status(world.data, world.vault)
    assert state["stage"] == 1 and state["vault_months_in_data"] == [] and state["plaintext_files"] == len(moving)
    again = hashes(world.data), hashes(world.vault, skip=frozenset({vault.MANIFEST_NAME}))
    assert vault.move(world.data, world.vault).nothing_to_do
    assert (hashes(world.data), hashes(world.vault, skip=frozenset({vault.MANIFEST_NAME}))) == again
    assert not (world.data / vault.STAGING_DIR).exists()


def test_move_leaves_the_pairs_without_vault_months_alone(tmp_path):
    world = make_world(tmp_path)
    untouched = {rel: sha for rel, sha in hashes(world.data).items() if "BBBUSDT" in rel or "DDDUSDT" in rel}
    vault.move(world.data, world.vault)
    now = hashes(world.data)
    assert all(now[rel] == sha for rel, sha in untouched.items())
    assert not any("BBBUSDT" in rel or "DDDUSDT" in rel for rel in hashes(world.vault))


def test_move_is_resumable_after_a_crash_at_every_step_and_never_loses_a_file(tmp_path):
    reference = make_world(tmp_path / "reference")
    labels: list[str] = []
    vault.move(reference.data, reference.vault, checkpoint=labels.append)
    final = hashes(reference.data), hashes(reference.vault, skip=BOOKKEEPING)
    digest = json.loads((reference.vault / vault.MANIFEST_NAME).read_text())["digest"]
    assert len(labels) >= 20 and labels[0] == "journal-written" and labels[-1] == "manifest-written"

    class Crash(Exception):
        pass

    for step in range(len(labels)):
        world = make_world(tmp_path / f"crash{step}")
        original = hashes(world.data)
        count = 0

        def boom(label: str) -> None:
            nonlocal count
            count += 1
            if count == step + 1:
                raise Crash(label)

        with pytest.raises(Crash):
            vault.move(world.data, world.vault, checkpoint=boom)
        # At the crash: every original file is in data/ or in the vault tree, and no table path is empty.
        now_data, now_vault = hashes(world.data), hashes(world.vault)
        for rel, sha in original.items():
            assert now_data.get(rel) == sha or now_vault.get(rel) == sha, (labels[step], rel)
            if rel.startswith("parquet/"):
                assert rel in now_data, (labels[step], rel)
        # A second run finishes the job; the end state is the uninterrupted run's.
        resumed = vault.move(world.data, world.vault)
        assert resumed.resumed and not resumed.nothing_to_do, labels[step]
        assert (hashes(world.data), hashes(world.vault, skip=BOOKKEEPING)) == final, labels[step]
        assert json.loads((world.vault / vault.MANIFEST_NAME).read_text())["digest"] == digest, labels[step]
        assert vault.move(world.data, world.vault).nothing_to_do


def test_move_stops_if_the_rebuild_differs_from_the_old_sealed_read(tmp_path):
    world = make_world(tmp_path)
    # The December archive is replaced by one with other prices: the old table no longer matches the archives.
    write_archive(
        world.data / "raw", "CCCUSDT", "1d", DEC, month_rows(DEC, DAY_MS, unit="us", price=999.0)
    )
    before = hashes(world.data)
    with pytest.raises(vault.VaultError, match="differs"):
        vault.move(world.data, world.vault)
    now = hashes(world.data)
    staged = {rel for rel in now if rel.startswith(vault.STAGING_DIR)}
    assert {rel: sha for rel, sha in now.items() if rel not in staged} == before  # no archive, no table, no report changed
    assert staged  # only the staging folder holds the verified rebuilds, outside parquet/
    assert not any(rel.startswith(("raw/", "parquet/")) for rel in hashes(world.vault))  # and nothing was moved


def test_a_move_for_another_data_root_does_not_finish_a_stranger_journal(tmp_path):
    world = make_world(tmp_path / "one")
    other = make_world(tmp_path / "two")

    class Crash(Exception):
        pass

    def boom(label: str) -> None:
        raise Crash(label)

    with pytest.raises(Crash):
        vault.move(world.data, world.vault, checkpoint=boom)
    with pytest.raises(vault.VaultError, match="unfinished move"):
        vault.move(other.data, world.vault)


# --- encrypt -----------------------------------------------------------------------------------------------


def throwaway_identity(tmp_path: Path, name: str = "id.txt") -> tuple[pyrage.x25519.Identity, str, Path]:
    identity = pyrage.x25519.Identity.generate()
    path = tmp_path / "usb" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# throwaway\n# public key: {identity.to_public()}\n{identity}\n")
    return identity, str(identity.to_public()), path


def test_encrypt_round_trips_byte_for_byte_and_removes_plaintext_only_with_the_flag(tmp_path):
    world = make_world(tmp_path)
    vault.move(world.data, world.vault)
    plain = hashes(world.vault, skip=BOOKKEEPING)
    identity, recipient, _ = throwaway_identity(tmp_path)

    first = vault.encrypt(world.vault, recipient)
    assert (first.encrypted, first.reused, first.plaintext_removed) == (len(plain), 0, 0)
    assert all((world.vault / rel).is_file() for rel in plain)  # no flag: every plaintext file stays
    listed = {e["path"]: e for e in json.loads((world.vault / vault.MANIFEST_NAME).read_text())["files"]}
    assert vault.status(world.data, world.vault)["stage"] == 1  # plaintext is still there
    for rel, sha in plain.items():
        cipher = world.vault / (rel + ".age")
        blob = cipher.read_bytes()
        assert listed[rel]["ciphertext"]["sha256"] == hashlib.sha256(blob).hexdigest() and listed[rel]["sha256"] == sha
        assert hashlib.sha256(pyrage.decrypt(blob, [identity])).hexdigest() == sha  # byte for byte
        assert blob[:21] == b"age-encryption.org/v1"

    second = vault.encrypt(world.vault, recipient, remove_plaintext=True)
    assert (second.encrypted, second.reused, second.plaintext_removed) == (0, len(plain), len(plain))
    assert not any((world.vault / rel).exists() for rel in plain)
    assert all((world.vault / (rel + ".age")).is_file() for rel in plain)
    state = vault.status(world.data, world.vault)
    assert state["stage"] == 2 and state["plaintext_files"] == 0 and state["encrypted_files"] == len(plain) and state["manifest_ok"]
    assert vault.encrypt(world.vault, recipient, remove_plaintext=True).plaintext_removed == 0  # idempotent


def test_a_wrong_identity_and_a_tampered_ciphertext_both_fail(tmp_path):
    world = make_world(tmp_path)
    vault.move(world.data, world.vault)
    identity, recipient, _ = throwaway_identity(tmp_path)
    vault.encrypt(world.vault, recipient)
    cipher = next(p for p in sorted(world.vault.rglob("*.age")))
    blob = cipher.read_bytes()
    assert pyrage.decrypt(blob, [identity])
    stranger = pyrage.x25519.Identity.generate()
    with pytest.raises(pyrage.DecryptError):
        pyrage.decrypt(blob, [stranger])
    for position in (len(blob) // 2, len(blob) - 5):  # the body, and the last bytes
        tampered = bytearray(blob)
        tampered[position] ^= 0x01
        with pytest.raises(pyrage.DecryptError):
            pyrage.decrypt(bytes(tampered), [identity])


def test_encrypt_refuses_a_bad_recipient_an_unlisted_file_and_a_changed_file(tmp_path):
    world = make_world(tmp_path)
    vault.move(world.data, world.vault)
    _, recipient, _ = throwaway_identity(tmp_path)
    with pytest.raises(vault.VaultError, match="recipient"):
        vault.encrypt(world.vault, "age1notarecipient")
    with pytest.raises(vault.VaultError, match="recipient"):
        vault.encrypt(world.vault, "AGE-SECRET-KEY-1ABC")
    stray = world.vault / "raw" / "stray.txt"
    stray.write_text("not in the manifest")
    with pytest.raises(vault.VaultError, match="does not list"):
        vault.encrypt(world.vault, recipient, remove_plaintext=True)
    stray.unlink()
    victim = next(p for p in sorted((world.vault / "raw").rglob("*.zip")))
    victim.write_bytes(victim.read_bytes() + b"!")
    with pytest.raises(vault.VaultError, match="do not match the SHA-256"):
        vault.encrypt(world.vault, recipient, remove_plaintext=True)
    assert victim.is_file() and not list(world.vault.rglob("*.age"))  # refused before anything was encrypted or removed


def test_encrypt_waits_for_an_unfinished_move(tmp_path):
    world = make_world(tmp_path)
    _, recipient, _ = throwaway_identity(tmp_path)

    class Crash(Exception):
        pass

    def boom(label: str) -> None:
        raise Crash(label)

    with pytest.raises(Crash):
        vault.move(world.data, world.vault, checkpoint=boom)
    with pytest.raises(vault.VaultError, match="unfinished"):
        vault.encrypt(world.vault, recipient)


# --- keygen ------------------------------------------------------------------------------------------------


def test_keygen_writes_an_identity_and_prints_only_the_recipient(tmp_path, capsys):
    world = make_world(tmp_path)
    out = tmp_path / "usb" / "age.key"
    out.parent.mkdir()
    argv = ["--data-dir", str(world.data), "--vault-dir", str(world.vault), "keygen", "--out", str(out)]
    assert vault.main(argv) == 0
    printed = capsys.readouterr().out
    recipient = printed.strip()
    assert recipient.startswith("age1") and "\n" not in recipient and "AGE-SECRET-KEY" not in printed
    secret = out.read_text()
    assert "AGE-SECRET-KEY-1" in secret and recipient in secret
    # The recipient and the identity match: a ciphertext for the one opens with the other.
    identity = protocol.read_identity(out)
    assert pyrage.decrypt(pyrage.encrypt(b"x", [pyrage.x25519.Recipient.from_str(recipient)]), [identity]) == b"x"
    assert vault.main(argv) == 2  # never overwrites
    assert out.read_text() == secret


@pytest.mark.parametrize("where", ["repository", "data root", "vault root", "missing medium"])
def test_keygen_refuses_a_path_where_the_key_would_stay_on_this_machine(tmp_path, where):
    world = make_world(tmp_path)
    world.vault.mkdir()
    out = {
        "repository": REPO_ROOT / "age.key",
        "data root": world.data / "age.key",
        "vault root": world.vault / "age.key",
        "missing medium": tmp_path / "no-usb" / "age.key",
    }[where]
    with pytest.raises(vault.VaultError):
        vault.keygen(out, world.data, world.vault)
    assert not out.exists()


# --- the open path ------------------------------------------------------------------------------------------


def open_reads(world: World, monkeypatch) -> dict[tuple[str, str], pl.DataFrame]:
    """Every pair read with both keys forged (a fake vault only: the keys are in a temporary folder)."""
    open_vault(monkeypatch)
    try:
        return {pair: load_klines(*pair, data_dir=world.data, caller="vault test") for pair in PAIRS}
    finally:
        monkeypatch.delenv("OLSPRIME_OPEN_VAULT")


def test_after_stage_1_the_open_path_returns_the_same_rows_as_before_the_move_and_logs_each_read(tmp_path, monkeypatch):
    world = make_world(tmp_path)
    monkeypatch.setenv("OLSPRIME_VAULT_DIR", str(world.vault))
    before = open_reads(world, monkeypatch)
    assert max(before[("AAAUSDT", "1d")]["open_time"]) >= VAULT.start  # the vault rows are in it
    vault.move(world.data, world.vault)
    logged = len(access_lines())
    after = open_reads(world, monkeypatch)
    for pair, frame in before.items():
        assert after[pair].schema == frame.schema and after[pair].equals(frame), pair
    lines = access_lines()[logged:]
    assert any("[vault tree]" in line and "vault_rows=" in line for line in lines)
    tree_reads = [line for line in lines if "[vault tree]" in line]
    assert len(tree_reads) == len(VAULT_PAIRS) and all("vault test" in line for line in lines)
    # A subset of columns keeps its order, and an open read without the vault tree's pair is the data file alone.
    open_vault(monkeypatch)
    narrow = load_klines("AAAUSDT", "1d", data_dir=world.data, columns=["close", "open_time"])
    monkeypatch.delenv("OLSPRIME_OPEN_VAULT")
    assert narrow.columns == ["close", "open_time"] and narrow.equals(before[("AAAUSDT", "1d")].select("close", "open_time"))


def test_without_both_keys_nothing_reads_the_vault_tree(tmp_path, monkeypatch):
    world = make_world(tmp_path)
    monkeypatch.setenv("OLSPRIME_VAULT_DIR", str(world.vault))
    vault.move(world.data, world.vault)
    for rel in hashes(world.vault):  # make every file of the tree unreadable as a table
        if rel.endswith(".parquet"):
            (world.vault / rel).write_bytes(b"this is not a parquet file")
    sealed = load_klines("AAAUSDT", "1d", data_dir=world.data)
    assert sealed["open_time"].max() < VAULT.start
    monkeypatch.setenv("OLSPRIME_OPEN_VAULT", "1")  # the variable alone
    assert load_klines("AAAUSDT", "1d", data_dir=world.data).equals(sealed)
    monkeypatch.delenv("OLSPRIME_OPEN_VAULT")
    key = Path(protocol.vault_key_file())
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text("2026-11-01\nfinal models: test\n")  # the file alone
    assert load_klines("AAAUSDT", "1d", data_dir=world.data).equals(sealed)
    assert access_lines() == []
    monkeypatch.setenv("OLSPRIME_OPEN_VAULT", "1")  # both keys: now the tree is read, and it is garbage
    with pytest.raises(pl.exceptions.PolarsError):
        load_klines("AAAUSDT", "1d", data_dir=world.data)


def test_at_stage_2_the_open_path_needs_the_identity_and_forged_keys_alone_read_nothing(tmp_path, monkeypatch):
    world = make_world(tmp_path)
    monkeypatch.setenv("OLSPRIME_VAULT_DIR", str(world.vault))
    before = open_reads(world, monkeypatch)
    vault.move(world.data, world.vault)
    _, recipient, identity_file = throwaway_identity(tmp_path)
    vault.encrypt(world.vault, recipient, remove_plaintext=True)
    assert vault.status(world.data, world.vault)["stage"] == 2
    logged = len(access_lines())

    # Both keys forged, no identity: nothing opens.
    open_vault(monkeypatch)
    with pytest.raises(VaultSealed, match="identity"):
        load_klines("AAAUSDT", "1d", data_dir=world.data)
    # Both keys forged, another identity: nothing opens.
    _, _, stranger_file = throwaway_identity(tmp_path, "stranger.txt")
    monkeypatch.setenv("OLSPRIME_VAULT_IDENTITY", str(stranger_file))
    with pytest.raises(VaultSealed, match="does not open"):
        load_klines("AAAUSDT", "1d", data_dir=world.data)
    # A file that is no identity, and a missing one.
    junk = tmp_path / "usb" / "junk.txt"
    junk.write_text("not a key\n")
    monkeypatch.setenv("OLSPRIME_VAULT_IDENTITY", str(junk))
    with pytest.raises(VaultSealed, match="no age identity"):
        load_klines("AAAUSDT", "1d", data_dir=world.data)
    monkeypatch.setenv("OLSPRIME_VAULT_IDENTITY", str(tmp_path / "usb" / "absent.txt"))
    with pytest.raises(VaultSealed, match="cannot read"):
        load_klines("AAAUSDT", "1d", data_dir=world.data)
    assert not any("vault_rows" in line and "[vault tree]" in line for line in access_lines()[logged:])  # no rows came out
    # The identity: every pair opens, and equals what was read before the move.
    monkeypatch.setenv("OLSPRIME_VAULT_IDENTITY", str(identity_file))
    after = {pair: load_klines(*pair, data_dir=world.data, caller="vault test") for pair in PAIRS}
    monkeypatch.delenv("OLSPRIME_OPEN_VAULT")
    for pair, frame in before.items():
        assert after[pair].schema == frame.schema and after[pair].equals(frame), pair
    # The identity alone (keys not both in place) opens nothing: the sealed read has no vault row.
    assert load_klines("AAAUSDT", "1d", data_dir=world.data)["open_time"].max() < VAULT.start


# --- downloads and rebuilds refuse the vault ------------------------------------------------------------------


def test_the_downloader_refuses_a_vault_month_before_any_request(tmp_path):
    bucket = FakeBucket()
    for month in (DEC, JAN):
        bucket.add_month("BTCUSDT", "1d", month, month_rows(month, DAY_MS))
    with bucket.client() as client:
        with pytest.raises(VaultMonthRefused, match="olsprime.vault"):
            download_month(client, "BTCUSDT", "1d", JAN, raw_dir=tmp_path)
        with pytest.raises(VaultMonthRefused, match="2026-01"):
            download_klines(["BTCUSDT"], ["1d"], [DEC, JAN], raw_dir=tmp_path, client=client)
        with pytest.raises(VaultMonthRefused):
            download_archives([("BTCUSDT", "1d", MAR)], raw_dir=tmp_path, client=client)
        with pytest.raises(VaultMonthRefused, match="later step"):
            run(["BTCUSDT"], ["1d"], DEC, FEB, data_dir=tmp_path, client=client)
        assert bucket.requests == []  # not one request, not even for the December archive
        run(["BTCUSDT"], ["1d"], DEC, DEC, data_dir=tmp_path, client=client)  # the month before the vault is fine
    assert not list(tmp_path.rglob("*2026*"))


def test_the_download_command_refuses_a_vault_month_and_defaults_before_the_vault(tmp_path, capsys):
    from olsprime.data import binance_vision

    with pytest.raises(SystemExit) as stop:
        binance_vision.main(["--symbols", "BTCUSDT", "--start", "2025-12", "--end", "2026-02", "--data-dir", str(tmp_path)])
    assert stop.value.code == 2 and "olsprime.vault" in capsys.readouterr().err
    assert binance_vision.vault_first_month() == JAN and binance_vision.is_vault_month(JAN) and not binance_vision.is_vault_month(DEC)


def test_build_parquet_leaves_a_vault_archive_out_with_a_warning_that_names_it(tmp_path, caplog):
    world = make_world(tmp_path)
    with caplog.at_level("WARNING"):
        report = build_parquet("AAAUSDT", "1d", raw_dir=world.data / "raw", parquet_dir=tmp_path / "rebuilt")
    named = [r.getMessage() for r in caplog.records if "vault month" in r.getMessage()]
    assert any("AAAUSDT-1d-2026-01.zip" in text for text in named) and any("AAAUSDT-1d-2026-02.zip" in text for text in named)
    assert report.last_open_time < VAULT.start
    assert protocol.stored_open_time_bounds(parquet_path(tmp_path / "rebuilt", "AAAUSDT", "1d"))[1] < VAULT.start


def test_universe_and_survey_still_work_after_the_move(tmp_path):
    world = make_world(tmp_path)
    vault.move(world.data, world.vault)
    ranked = universe.run(data_dir=world.data, start=DEC, end=DEC, top_n=2, min_days=5, download=False)
    assert ranked.height > 0 and set(ranked["symbol"]) <= {"AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"}
    reports = survey.quality_reports(world.data / "parquet", "1d", {"AAAUSDT"})
    assert reports[0]["rows"] > 0 and reports[0]["last_open_time"] < VAULT.start.isoformat()
    assert survey.ticker_reuse(world.data, ["AAAUSDT"]) == []


# --- the guards cover the tree ----------------------------------------------------------------------------------


def test_the_tripwire_trips_on_a_read_of_the_real_vault_tree_even_with_both_keys(tripwire, monkeypatch):
    # The default vault tree sits beside the repository. The path is only named, never opened.
    path = default_vault_dir() / "parquet" / "spot" / "klines" / "1d" / "AAAUSDT.parquet"
    open_vault(monkeypatch)
    with tripwire.expect_trip("seal"):
        protocol.load_klines_file(path, "1d", caller="vault test")
    assert not path.exists()


def test_the_tripwire_judges_a_fake_vault_tree_by_the_ordinary_rules(tmp_path, monkeypatch):
    # A table of the fake tree read through the protocol is allowed (its file holds the allowance); a read
    # from a file without the `reader` allowance would trip (that is test_tripwire's business).
    world = make_world(tmp_path)
    monkeypatch.setenv("OLSPRIME_VAULT_DIR", str(world.vault))
    vault.move(world.data, world.vault)
    assert protocol.stored_row_count(protocol.vault_tree_klines_path("AAAUSDT", "1d")) > 0


# --- the manifest reports the tree ----------------------------------------------------------------------------------


def test_the_manifest_reports_the_vault_tree_at_each_stage_and_reads_no_vault_row(tmp_path, monkeypatch):
    world = make_world(tmp_path)
    monkeypatch.setenv("OLSPRIME_VAULT_DIR", str(world.vault))
    stage0 = manifest.build_manifest(world.data)
    assert stage0["vault_tree"] == {
        "stage": 0,
        "files": 0,
        "bytes": 0,
        "months_still_in_data": 6 + len(VAULT_PAIRS),
        "manifest_digest": None,
        "manifest_ok": None,
    }
    vault.move(world.data, world.vault)
    stage1 = manifest.build_manifest(world.data)
    tree = stage1["vault_tree"]
    listed = json.loads((world.vault / vault.MANIFEST_NAME).read_text())
    assert (tree["stage"], tree["files"], tree["months_still_in_data"]) == (1, len(listed["files"]), 0)
    assert tree["manifest_digest"] == listed["digest"] and tree["manifest_ok"] is True
    assert tree["bytes"] == sum(e["bytes"] for e in listed["files"])
    _, recipient, _ = throwaway_identity(tmp_path)
    vault.encrypt(world.vault, recipient, remove_plaintext=True)
    stage2 = manifest.build_manifest(world.data)
    assert stage2["vault_tree"]["stage"] == 2 and stage2["vault_tree"]["files"] == len(listed["files"])
    assert len({stage0["digest"], stage1["digest"], stage2["digest"]}) == 3  # the stage is part of the body
    assert all(manifest.verify_manifest(m) for m in (stage0, stage1, stage2))
    assert access_lines() == []  # not one vault row was read


def test_status_names_what_is_still_under_data(tmp_path):
    world = make_world(tmp_path)
    state = vault.status(world.data, world.vault)
    assert state["stage"] == 0 and state["tree_files"] == 0 and not state["unfinished_move"]
    named = "\n".join(state["vault_months_in_data"])
    assert "raw/spot/monthly/klines/AAAUSDT/1d/AAAUSDT-1d-2026-01.zip" in named and "CCCUSDT 1d (Parquet footer)" in named
    assert "stage 0" in vault.format_status(state)


def test_the_command_line_runs_each_stage(tmp_path, capsys):
    world = make_world(tmp_path)
    base = ["--data-dir", str(world.data), "--vault-dir", str(world.vault)]
    assert vault.main([*base, "plan"]) == 0
    assert "Archives of vault months under data/raw/: 6" in capsys.readouterr().out
    assert vault.main([*base, "move"]) == 0 and "finished" in capsys.readouterr().out
    assert vault.main([*base, "move"]) == 0 and "nothing to do" in capsys.readouterr().out
    assert vault.main([*base, "status"]) == 0 and "stage 1" in capsys.readouterr().out
    _, recipient, _ = throwaway_identity(tmp_path)
    assert vault.main([*base, "encrypt", "--recipient", recipient]) == 0 and "encrypted" in capsys.readouterr().out
    assert vault.main([*base, "status"]) == 0 and "stage 1" in capsys.readouterr().out
    assert vault.main([*base, "encrypt", "--recipient", recipient, "--remove-plaintext"]) == 0
    capsys.readouterr()
    assert vault.main([*base, "status"]) == 0 and "stage 2" in capsys.readouterr().out
    assert vault.main([*base, "encrypt", "--recipient", "nonsense"]) == 2
