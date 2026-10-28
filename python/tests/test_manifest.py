"""The vault manifest: checksums, stored and sealed counts, one canonical digest."""

import hashlib
import json
import random
from datetime import UTC, datetime

import polars as pl
import pytest

from olsprime import manifest
from olsprime.manifest import body_digest, build_manifest, verify_manifest
from olsprime.paths import runs_dir
from olsprime.protocol import VAULT, load_klines, load_universe
from olsprime.universe import UNIVERSE_SCHEMA, universe_path
from tests.fakes import DAY_MS, HOUR_MS
from tests.test_protocol import access_lines, bars, open_vault, store, utc

FIXED = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
HOURLY = "parquet/spot/klines/1h/AAAUSDT.parquet"
DAILY = "parquet/spot/klines/1d/AAAUSDT.parquet"
UNIVERSE = "parquet/universe/spot_usdt_top2.parquet"


def store_universe(data_dir, rebalances: list[datetime]) -> None:
    rows = [(t, i + 1, f"S{i}USDT", 1e6 - i, 30) for t in rebalances for i in range(2)]
    frame = pl.DataFrame(rows, schema=list(UNIVERSE_SCHEMA), orient="row").cast(UNIVERSE_SCHEMA)
    path = universe_path(data_dir / "parquet", 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


@pytest.fixture
def data(tmp_path):
    store(tmp_path, bars(utc(2025, 12, 31, 20), 8, HOUR_MS, "AAAUSDT"), "AAAUSDT", "1h")  # 4 sealed, 4 in the vault
    store(tmp_path, bars(utc(2025, 12, 29), 6, DAY_MS, "AAAUSDT"), "AAAUSDT", "1d")  # 3 sealed, 3 in the vault
    store(tmp_path, bars(utc(2026, 2, 1), 5, DAY_MS, "BBBUSDT"), "BBBUSDT", "1d")  # all in the vault
    store_universe(tmp_path, [utc(2025, 11, 1), utc(2025, 12, 1), utc(2026, 1, 1)])
    return tmp_path


def entries(built):
    return {e["path"]: e for e in built["files"]}


def test_every_parquet_file_is_listed_in_path_order_with_its_checksum(data):
    built = build_manifest(data, generated_at=FIXED)
    paths = [e["path"] for e in built["files"]]
    assert paths == sorted(paths) and len(paths) == 4
    assert paths[0] == DAILY and paths[-1] == UNIVERSE
    for entry in built["files"]:
        raw = (data / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest() and entry["bytes"] == len(raw)
    assert built["totals"] == {
        "files": 4,
        "bytes": sum(e["bytes"] for e in built["files"]),
        "stored_rows": 8 + 6 + 5 + 6,
        "sealed_rows": 4 + 3 + 0 + 4,
    }


def test_stored_and_sealed_counts_and_spans(data):
    by_path = entries(build_manifest(data, generated_at=FIXED))
    hourly = by_path[HOURLY]
    assert (hourly["stored_rows"], hourly["sealed_rows"]) == (8, 4)
    assert hourly["first_sealed"] == "2025-12-31T20:00:00.000+00:00"
    assert hourly["last_sealed"] == "2025-12-31T23:00:00.000+00:00"
    assert (hourly["kind"], hourly["symbol"], hourly["interval"], hourly["time_column"]) == (
        "klines",
        "AAAUSDT",
        "1h",
        "open_time",
    )
    daily = by_path[DAILY]
    assert (daily["stored_rows"], daily["sealed_rows"], daily["last_sealed"]) == (6, 3, "2025-12-31T00:00:00.000+00:00")
    gone = by_path["parquet/spot/klines/1d/BBBUSDT.parquet"]  # the whole file is in the vault
    assert (gone["stored_rows"], gone["sealed_rows"], gone["first_sealed"], gone["last_sealed"]) == (5, 0, None, None)
    universe = by_path[UNIVERSE]
    assert (universe["kind"], universe["time_column"]) == ("universe", "rebalance_time")
    assert (universe["stored_rows"], universe["sealed_rows"]) == (6, 4)  # the 2026-01-01 rebalance is the vault's
    assert universe["last_sealed"] == "2025-12-01T00:00:00.000+00:00" and "symbol" not in universe


def test_the_sealed_count_matches_the_protocol_loaders(data):
    by_path = entries(build_manifest(data, generated_at=FIXED))
    for symbol, interval in (("AAAUSDT", "1h"), ("AAAUSDT", "1d"), ("BBBUSDT", "1d")):
        entry = by_path[f"parquet/spot/klines/{interval}/{symbol}.parquet"]
        assert entry["sealed_rows"] == load_klines(symbol, interval, data_dir=data, columns=["open_time"]).height
    kept = load_universe(universe_path(data / "parquet", 2), caller="test")
    assert kept.height == by_path[UNIVERSE]["sealed_rows"]
    assert kept["rebalance_time"].max() < VAULT.start
    assert load_universe(universe_path(data / "parquet", 2), columns=["symbol"], caller="test").columns == ["symbol"]


def test_the_digest_is_of_the_canonical_body_and_survives_the_clock(data):
    one = build_manifest(data, generated_at=FIXED)
    two = build_manifest(data, generated_at=datetime(2030, 1, 1, tzinfo=UTC))
    assert one["generated_at"] != two["generated_at"] and one["digest"] == two["digest"]
    assert verify_manifest(one) and len(one["digest"]) == 64
    # Key order in the file does not matter: the digest is of the canonical form.
    shuffled = {k: one[k] for k in random.Random(3).sample(list(one), len(one))}
    assert body_digest(shuffled) == one["digest"]
    written = json.loads(manifest.write_manifest(one, data / "m.json").read_text())
    assert verify_manifest(written) and written == one


def test_any_change_moves_the_digest(data):
    base = build_manifest(data, generated_at=FIXED)
    tampered = json.loads(json.dumps(base))
    tampered["files"][0]["sealed_rows"] += 1
    assert not verify_manifest(tampered)
    # A table that changes by one value is a different manifest, though no count moves.
    store(data, bars(utc(2025, 12, 29), 6, DAY_MS, "AAAUSDT").with_columns(close=pl.col("close") + 1.0), "AAAUSDT", "1d")
    changed = build_manifest(data, generated_at=FIXED)
    assert changed["digest"] != base["digest"]
    assert entries(changed)[DAILY]["sha256"] != entries(base)[DAILY]["sha256"]
    assert entries(changed)[DAILY]["stored_rows"] == entries(base)[DAILY]["stored_rows"]
    assert entries(changed)[HOURLY] == entries(base)[HOURLY]  # the other entries are untouched


def test_opening_the_vault_changes_the_sealed_counts_and_logs_the_reads(data, monkeypatch):
    sealed = build_manifest(data, generated_at=FIXED)
    assert sealed["vault_open"] is False and access_lines() == []
    open_vault(monkeypatch)
    opened = build_manifest(data, generated_at=FIXED)
    assert opened["vault_open"] is True and opened["digest"] != sealed["digest"]
    assert opened["totals"]["sealed_rows"] == opened["totals"]["stored_rows"] == 25
    assert len(access_lines()) == 4 and all("manifest.build_manifest" in line for line in access_lines())


def test_an_empty_folder_gives_an_empty_manifest(tmp_path):
    built = build_manifest(tmp_path, generated_at=FIXED)
    assert built["files"] == [] and built["totals"]["files"] == 0 and verify_manifest(built)


def test_a_parquet_file_of_unknown_kind_is_refused(data):
    stray = data / "parquet" / "elsewhere" / "table.parquet"
    stray.parent.mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(stray)
    with pytest.raises(ValueError, match="does not know its seal"):
        build_manifest(data)


def test_the_cli_writes_the_manifest_and_prints_its_digest(data, tmp_path_factory, capsys):
    out = tmp_path_factory.mktemp("out") / "nested" / "m.json"
    assert manifest.main(["--data-dir", str(data), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert verify_manifest(written) and written["totals"]["files"] == 4
    text = capsys.readouterr().out
    assert written["digest"] in text and "4 files" in text and "vault sealed" in text


def test_the_default_output_is_a_timestamped_file_under_runs(data, capsys):
    assert manifest.main(["--data-dir", str(data)]) == 0
    (written,) = (runs_dir() / "manifest").glob("*.json")
    assert len(written.stem) == len("20261027T120000Z") and written.stem.endswith("Z")
    assert verify_manifest(json.loads(written.read_text()))
    assert manifest.default_output(FIXED).name == "20261027T120000Z.json"
