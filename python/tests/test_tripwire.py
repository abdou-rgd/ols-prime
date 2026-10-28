"""The run-time tripwire: each bypass the static guard cannot see must trip, and allowed readers must not."""

import functools
import importlib
import importlib.util
import io
import json
import os
import runpy
import sys
import threading
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from olsprime.data.binance_vision import Month, parquet_path
from olsprime.paths import REPO_ROOT, data_dir
from olsprime.protocol import load_klines, load_klines_file, stored_row_count
from tests import tripwire as wire
from tests.allowlist import ALLOWED
from tests.fakes import DAY_MS, HOUR_MS, klines_csv, month_rows, zip_archive
from tests.test_protocol import bars, open_vault, store, utc
from tests.tripwire import Hit, Target, Tripwire, TripwireError, is_allowed

PYTHON_ROOT = Path(wire.__file__).resolve().parents[1]

# Readers captured while this module is imported, in a dict and in a partial: the first audit's evasions.
# The tripwire goes up in pytest_configure, before collection, so what is captured here is the wrapper.
EARLY_READERS = {"table": getattr(pl, "read_" + "parquet")}
EARLY_PARTIAL = functools.partial(getattr(pl, "scan_" + "parquet"), "table.parquet")


@pytest.fixture
def stored(tmp_path) -> Path:
    """A real Parquet file (and an Arrow IPC file beside it), written with the unpatched writers."""
    path = tmp_path / "data.parquet"
    frame = bars(utc(2025, 12, 31, 20), 8, HOUR_MS)  # 4 bars before the vault, 4 inside
    frame.write_parquet(path)
    frame.write_ipc(tmp_path / "data.arrow")
    np.save(tmp_path / "plain.npy", np.arange(3))
    np.save(tmp_path / "boxed.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    (tmp_path / "rows.csv").write_text("a,b\n1,2\n")
    (tmp_path / "rows.ndjson").write_text('{"a": 1}\n')
    # An archive, made by a helper that only writes: opening a zip for writing is not a read.
    maker = "import zipfile\nwith zipfile.ZipFile(FOLDER + '/archive.zip', 'w') as z:\n    z.writestr('rows.csv', '1,2\\n')\n"
    run_source(tmp_path, maker, path)
    return path


def run_source(tmp_path: Path, source: str, stored: Path, **extra: str) -> dict:
    """Run ``source`` as a file in ``tmp_path`` (outside python/), with ``PATH`` bound to ``stored``."""
    script = tmp_path / "helper_script.py"
    script.write_text(source, encoding="utf-8")
    names = {"PATH": str(stored), "FOLDER": str(stored.parent), **extra}
    return runpy.run_path(str(script), init_globals=names)


# --- Each bypass trips ----------------------------------------------------------------------

BYPASSES = [
    ("direct", "import polars as pl\npl.read_parquet(PATH)", "reader"),
    ("getattr-string-concat", "import polars as pl\ngetattr(pl, 'read_' + 'parquet')(PATH)", "reader"),
    ("vars-of-polars", "import polars as pl\nvars(pl)['read_' + 'parquet'](PATH)", "reader"),
    (
        "computed-path",
        "import os, polars as pl\nname = ''.join(['da', 'ta', '.par', 'quet'])\npl.read_parquet(os.path.join(FOLDER, name))",
        "reader",
    ),
    ("scan-then-collect", "import polars as pl\npl.scan_parquet(PATH).collect()", "reader"),
    ("read_parquet_schema", "import polars as pl\npl.read_parquet_schema(PATH)", "reader"),
    ("read_parquet_metadata", "import polars as pl\npl.read_parquet_metadata(PATH)", "reader"),
    ("alias-from-import", "from polars import read_parquet as rp\nrp(PATH)", "reader"),
    ("polars-ipc", "import polars as pl\npl.read_ipc(FOLDER + '/data.arrow')", "reader"),
    ("polars-scan-ipc", "import polars as pl\npl.scan_ipc(FOLDER + '/data.arrow').collect()", "reader"),
    (
        "polars-deserialize",
        "import polars as pl\nimport io\npl.DataFrame.deserialize(io.BytesIO(pl.DataFrame({'a': [1]}).serialize()))",
        "reader",
    ),
    ("lazy-deserialize", "import polars as pl\nimport io\npl.LazyFrame.deserialize(io.BytesIO(pl.LazyFrame({'a': [1]}).serialize()))", "reader"),
    ("pyarrow-dataset", "import pyarrow.dataset as ds\nds.dataset(PATH).to_table()", "reader"),
    ("pyarrow-dataset-module-getattr", "import pyarrow\nimport pyarrow.dataset\ngetattr(pyarrow.dataset, 'data' + 'set')(PATH)", "reader"),
    ("pyarrow-read_table", "import pyarrow.parquet as pq\npq.read_table(PATH)", "reader"),
    ("pyarrow-ParquetFile", "import pyarrow.parquet as pq\npq.ParquetFile(PATH).read()", "reader"),
    ("pyarrow-ParquetDataset", "import pyarrow.parquet as pq\npq.ParquetDataset(PATH)", "reader"),
    ("pyarrow-read_metadata", "import pyarrow.parquet as pq\npq.read_metadata(PATH)", "reader"),
    ("pyarrow-read_schema", "import pyarrow.parquet as pq\npq.read_schema(PATH)", "reader"),
    ("pyarrow-make_fragment", "import pyarrow.dataset as ds\nds.ParquetFileFormat().make_fragment(PATH)", "reader"),
    ("pyarrow-make_fragment-to_table", "import pyarrow.dataset as ds\nds.ParquetFileFormat().make_fragment(PATH).to_table()", "reader"),
    (
        "audit-a-sys-modules-fragment",  # bypass2/test_evasion.py, test 2, as the auditor wrote it
        "import sys\nds = sys.modules['pyarrow.dataset']\nfs = sys.modules['pyarrow.fs']\n"
        "ds.ParquetFileFormat().make_fragment(PATH, filesystem=fs.LocalFileSystem()).to_table()",
        "reader",
    ),
    ("pyarrow-from_paths", "import pyarrow.dataset as ds\nds.FileSystemDataset.from_paths([PATH], format=ds.ParquetFileFormat())", "reader"),
    (
        "pyarrow-dataset-factory",
        "import pyarrow.dataset as ds, pyarrow.fs as fs\nds.FileSystemDatasetFactory(fs.LocalFileSystem(), [PATH], ds.ParquetFileFormat())",
        "reader",
    ),
    ("zipfile-read", "import zipfile\nzipfile.ZipFile(FOLDER + '/archive.zip').read('rows.csv')", "raw-reader"),
    ("zipfile-append-mode", "import zipfile\nzipfile.ZipFile(FOLDER + '/archive.zip', 'a')", "raw-reader"),
    ("zipfile-mode-by-keyword", "import zipfile\nzipfile.ZipFile(FOLDER + '/archive.zip', mode='r')", "raw-reader"),
    ("polars-read_csv-of-a-file", "import polars as pl\npl.read_csv(FOLDER + '/rows.csv')", "raw-reader"),
    ("polars-read_csv-of-bytes", "import io, polars as pl\npl.read_csv(io.BytesIO(b'a\\n1\\n'), has_header=False)", "raw-reader"),
    ("polars-scan_csv", "import polars as pl\npl.scan_csv(FOLDER + '/rows.csv').collect()", "raw-reader"),
    ("polars-read_ndjson", "import polars as pl\npl.read_ndjson(FOLDER + '/rows.ndjson')", "raw-reader"),
    ("polars-scan_ndjson", "import polars as pl\npl.scan_ndjson(FOLDER + '/rows.ndjson').collect()", "raw-reader"),
    (
        "audit-a-zip-and-csv-in-a-helper",  # the helper of bypass2/test_evasion.py, run from a script
        "import io, zipfile\nimport polars as pl\nwith zipfile.ZipFile(FOLDER + '/archive.zip') as z:\n"
        "    pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), has_header=False)",
        "raw-reader",
    ),
    ("pyarrow-ipc", "import pyarrow.ipc as ipc\nipc.open_file(FOLDER + '/data.arrow')", "reader"),
    ("pyarrow-feather", "import pyarrow.feather as feather\nfeather.read_table(FOLDER + '/data.arrow')", "reader"),
    ("numpy-load-allow-pickle", "import numpy as np\nnp.load(FOLDER + '/boxed.npy', allow_pickle=True)", "binary-load"),
    ("numpy-load-positional", "import numpy as np\nnp.load(FOLDER + '/boxed.npy', None, True)", "binary-load"),
    ("pickle-loads", "import pickle\npickle.loads(pickle.dumps([1]))", "import:pickle"),
    ("pickle-load", "import pickle, io\npickle.load(io.BytesIO(pickle.dumps([1])))", "import:pickle"),
    ("pickle-pure-python-loads", "import pickle\npickle._loads(pickle.dumps([1]))", "import:pickle"),
    (
        "exec-of-a-string-in-a-helper",
        "exec(\"import polars as pl\\npl.read_parquet(PATH)\", {'PATH': PATH})",
        "reader",
    ),
]


@pytest.mark.parametrize(("name", "source", "rule"), BYPASSES, ids=[b[0] for b in BYPASSES])
def test_each_bypass_trips(name, source, rule, tripwire, tmp_path, stored):
    with tripwire.expect_trip(rule):
        run_source(tmp_path, source, stored)


def test_a_helper_module_written_to_tmp_path_and_imported_trips(tripwire, tmp_path, stored):
    helper = tmp_path / "sneaky_reader.py"
    helper.write_text("import polars as pl\n\ndef grab(path):\n    return pl.read_parquet(path)\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("sneaky_reader", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # importing is harmless; reading is not
    with tripwire.expect_trip("reader") as hits:
        module.grab(str(stored))
    assert hits[0].caller == str(helper)


def test_exec_of_a_string_trips_even_from_a_file_that_may_exec(tripwire, stored):
    with tripwire.expect_trip("reader") as hits:
        exec("import polars as pl\npl.read_parquet(PATH)", {"PATH": str(stored)})
    assert hits[0].caller == "<string>"


def test_this_test_file_holds_no_reader_allowance(tripwire, tmp_path, stored):
    assert "reader" not in ALLOWED["tests/test_tripwire.py"] and not is_allowed(__file__, "reader")
    with tripwire.expect_trip("reader") as hits:
        run_source(tmp_path, "import polars as pl\npl.read_parquet(PATH)", stored)
    assert Path(hits[0].caller) == tmp_path / "helper_script.py"
    assert "tests/allowlist.py" in str(hits[0])


def test_a_swallowed_trip_is_still_recorded(tripwire, tmp_path, stored):
    source = "import polars as pl\ntry:\n    pl.read_parquet(PATH)\nexcept Exception:\n    pass\n"
    with tripwire.expect_trip("reader"):  # the helper hid the error; the record remains for the teardown check
        run_source(tmp_path, source, stored)


def test_the_error_is_an_assertion_error_that_names_the_caller(tripwire, tmp_path, stored):
    with pytest.raises(TripwireError, match=r"read_parquet called from .*helper_script\.py") as raised:
        run_source(tmp_path, "import polars as pl\npl.read_parquet(PATH)", stored)
    assert isinstance(raised.value, AssertionError)
    tripwire.hits.clear()  # this test provoked the trip on purpose


# --- What must not trip -----------------------------------------------------------------------


def test_the_protocol_reads_through_its_allowance(tripwire, stored):
    sealed = load_klines_file(stored, "1h", columns=["open_time"], caller="test")
    assert sealed.height == 4
    assert stored_row_count(stored) == 8
    assert tripwire.hits == []


def test_writers_constructors_and_plain_numpy_do_not_trip(tripwire, tmp_path, stored):
    frame = pl.DataFrame({"a": [1, 2, 3]})
    frame.write_parquet(tmp_path / "out.parquet")
    frame.write_ipc(tmp_path / "out.arrow")
    assert pl.DataFrame({"a": [1]}).lazy().collect().height == 1
    # No pickle: numpy.load is not watched, from a helper outside python/ or anywhere else.
    run_source(tmp_path, "import numpy as np\nnp.load(FOLDER + '/plain.npy')\nnp.load(FOLDER + '/plain.npy', allow_pickle=False)", stored)
    assert tripwire.hits == []


# --- Who may read ----------------------------------------------------------------------------


def test_an_allowance_is_per_file_and_per_rule():
    assert is_allowed(str(PYTHON_ROOT / "olsprime" / "protocol.py"), "reader")
    assert not is_allowed(str(PYTHON_ROOT / "olsprime" / "protocol.py"), "binary-load")
    assert not is_allowed(str(PYTHON_ROOT / "olsprime" / "protocol.py"), "import:pickle")
    assert is_allowed(str(PYTHON_ROOT / "tests" / "test_universe.py"), "reader")
    assert not is_allowed(str(PYTHON_ROOT / "tests" / "test_models.py"), "reader")
    assert not is_allowed(str(PYTHON_ROOT / "olsprime" / "features" / "elsewhere.py"), "reader")
    assert not is_allowed("<string>", "reader")
    assert not is_allowed(str(PYTHON_ROOT.parent / "protocol.py"), "reader")  # same name, outside python/


def test_every_rule_the_tripwire_uses_is_a_rule_of_the_static_guard():
    static_rules = {"reader", "raw-reader", "binary-load", "import:pickle"}  # as test_protocol.findings names them
    used = {t.rule for t in wire.TARGETS} | {t.rule for ts in wire.OPTIONAL_TARGETS.values() for t in ts}
    assert used == static_rules
    assert wire.SEAL == "seal"  # the rule that guards olsprime.protocol; test_protocol.SEAL is the same name


def test_library_frames_are_skipped_and_run_time_code_is_not():
    skipped = wire.infrastructure()
    assert wire.is_infrastructure(pl.__file__, skipped)
    assert wire.is_infrastructure(np.__file__, skipped)
    assert wire.is_infrastructure(wire.__file__, skipped)
    assert wire.is_infrastructure("<frozen importlib._bootstrap>", skipped)
    assert wire.is_infrastructure(str(Path(importlib.util.__file__)), skipped)
    assert not wire.is_infrastructure("<string>", skipped)
    assert not wire.is_infrastructure(__file__, skipped)
    assert not wire.is_infrastructure(str(Path(__file__).with_name("conftest.py")), skipped)


def test_the_entry_points_of_this_polars_are_all_wrapped():
    wrapped = {t.label for t in wire.TARGETS}
    for expected in ("polars.read_parquet", "polars.scan_parquet", "polars.read_parquet_schema", "polars.read_parquet_metadata"):
        assert expected in wrapped
    # Anything polars calls a Parquet reader is found by name, so a new version is covered without an edit.
    by_name = {n for n in dir(pl) if n.startswith(("read_", "scan_")) and "parquet" in n}
    assert by_name <= {t.attribute for t in wire.TARGETS if t.module == "polars"}


# --- The mechanism, on a stand-in library -----------------------------------------------------


@pytest.fixture
def standin(monkeypatch):
    """A fake library with one reader, an alias of it bound before the tripwire goes up, and a reader class."""
    library = types.ModuleType("standin_frames")

    def read_frame(path):
        return f"read {path}"

    class Dataset:
        def __init__(self, path):
            self.path = path

    library.read_frame, library.Dataset = read_frame, Dataset
    holder = types.ModuleType("standin_holder")
    holder.early_alias = read_frame  # from standin_frames import read_frame as early_alias
    monkeypatch.setitem(sys.modules, "standin_frames", library)
    monkeypatch.setitem(sys.modules, "standin_holder", holder)
    return library, holder


def test_install_wraps_every_alias_and_keeps_classes_usable(standin, monkeypatch):
    library, holder = standin
    local = Tripwire()
    targets = [Target("standin_frames", "read_frame", "reader"), Target("standin_frames", "Dataset", "reader")]
    monkeypatch.setattr(wire, "TARGETS", targets)
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        assert wire.install(patches, local) == ["standin_frames.read_frame", "standin_frames.Dataset"]
        with pytest.raises(TripwireError):
            library.read_frame("x")
        with pytest.raises(TripwireError):
            holder.early_alias("x")  # the alias bound before the tripwire is wrapped too
        with pytest.raises(TripwireError):
            library.Dataset("x")
        assert [h.what for h in local.hits] == ["standin_frames.read_frame"] * 2 + ["standin_frames.Dataset"]
    assert library.read_frame("x") == "read x" and holder.early_alias("x") == "read x"  # undone
    assert library.Dataset("x").path == "x"


def test_a_missing_entry_point_is_skipped_not_an_error(standin, monkeypatch):
    local = Tripwire()
    monkeypatch.setattr(wire, "TARGETS", [Target("standin_frames", "read_nothing", "reader"), Target("no_such_lib", "f", "reader")])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        assert wire.install(patches, local) == []


def test_a_python_class_is_patched_in_place_so_no_copy_of_the_original_stays_reachable(standin, monkeypatch):
    library, _ = standin
    original = library.Dataset
    local = Tripwire()
    monkeypatch.setattr(wire, "TARGETS", [Target("standin_frames", "Dataset", "reader")])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        wire.install(patches, local)
        assert library.Dataset is original and original.__bases__ == (object,)  # nothing to walk back to
        assert not hasattr(original.__init__, "__wrapped__")
        with pytest.raises(TripwireError):
            library.Dataset("x")
        class Child(library.Dataset):  # a subclass (zipfile.Path over ZipFile) is guarded too
            pass
        with pytest.raises(TripwireError):
            Child("y")
    assert library.Dataset("x").path == "x" and Child("y").path == "y"  # undone


def test_a_native_class_becomes_a_subclass_and_the_original_stays_a_stated_limit(monkeypatch):
    native = type("Native", (), {})  # no __init__ of its own: stands for a cython type
    library = types.ModuleType("standin_native")
    library.Native = native
    monkeypatch.setitem(sys.modules, "standin_native", library)
    local = Tripwire()
    monkeypatch.setattr(wire, "TARGETS", [Target("standin_native", "Native", "reader")])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        wire.install(patches, local)
        assert library.Native is not native and issubclass(library.Native, native) and library.Native.__name__ == "Native"
        with pytest.raises(TripwireError):
            library.Native()
        assert library.Native.__bases__[0] is native  # the limit: one attribute away, and ungated


def test_a_wrapped_function_hands_back_no_original(standin, monkeypatch):
    library, _ = standin
    local = Tripwire()
    monkeypatch.setattr(wire, "TARGETS", [Target("standin_frames", "read_frame", "reader")])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        wire.install(patches, local)
        assert not hasattr(library.read_frame, "__wrapped__") and library.read_frame.__name__ == "read_frame"


def test_only_if_decides_whether_a_call_is_checked(standin, monkeypatch):
    library, _ = standin
    local = Tripwire()
    target = Target("standin_frames", "read_frame", "reader", only_if=lambda path: path.endswith(".secret"))
    monkeypatch.setattr(wire, "TARGETS", [target])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        wire.install(patches, local)
        assert library.read_frame("a.csv") == "read a.csv"
        with pytest.raises(TripwireError):
            library.read_frame("a.secret")
    assert len(local.hits) == 1 and isinstance(local.hits[0], Hit)


# --- Audit A, D1: the helper on sys.path, and the path rule ----------------------------------------


@pytest.fixture
def fake_data(tmp_path, monkeypatch, stored):
    """A fake ``data/`` tree (never the real one) that the tripwire takes for the repository's."""
    root = tmp_path / "fake_repo" / "data"
    for sub in ("raw/spot/monthly/klines/BTCUSDT/1d", "parquet/spot/klines/1d"):
        (root / sub).mkdir(parents=True)
    archive = root / "raw/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2026-03.zip"
    archive.write_bytes((stored.parent / "archive.zip").read_bytes())
    table = root / "parquet/spot/klines/1d/BTCUSDT.parquet"
    table.write_bytes(stored.read_bytes())
    plain = root / "plain.npy"
    plain.write_bytes((stored.parent / "plain.npy").read_bytes())
    monkeypatch.setattr(wire, "data_roots", lambda: (os.path.normcase(os.path.realpath(root)) + os.sep,))
    return types.SimpleNamespace(root=root, archive=archive, table=table, plain=plain)


AUDIT_A_HELPER = """
import io, zipfile
import polars as pl
import pyarrow.dataset as ds
import pyarrow.fs as pafs

def raw(path):
    with zipfile.ZipFile(path) as z:
        return pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), has_header=False)

def fragment(path):
    return ds.ParquetFileFormat().make_fragment(path, filesystem=pafs.LocalFileSystem()).to_table()
"""


@pytest.fixture
def helper_on_sys_path(tmp_path, monkeypatch):
    """The auditor's trick: a helper written to ``tmp_path``, put on ``sys.path``, imported under a plain name."""
    (tmp_path / "helper_audit_a.py").write_text(AUDIT_A_HELPER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "helper_audit_a", raising=False)
    yield importlib.import_module("helper_audit_a")
    sys.modules.pop("helper_audit_a", None)


def test_audit_a_a_helper_on_sys_path_cannot_read_the_raw_archives(tripwire, helper_on_sys_path, fake_data):
    with tripwire.expect_trip("raw-reader") as hits:
        helper_on_sys_path.raw(str(fake_data.archive))
    assert hits[0].what == "zipfile.ZipFile" and hits[0].path == str(fake_data.archive)
    assert Path(hits[0].caller).name == "helper_audit_a.py"


def test_audit_a_a_helper_on_sys_path_cannot_read_parquet_through_a_fragment(tripwire, helper_on_sys_path, fake_data):
    with tripwire.expect_trip("reader") as hits:
        helper_on_sys_path.fragment(str(fake_data.table))
    assert hits[0].path == str(fake_data.table) and "make_fragment" in hits[0].what


def test_a_path_under_data_is_checked_even_when_the_entry_point_would_not_be(tripwire, tmp_path, stored, fake_data):
    # numpy.load without allow_pickle is not watched on a path outside data/ ...
    run_source(tmp_path, "import numpy as np\nnp.load(FOLDER + '/plain.npy')", stored)
    # ... but under data/ it is, whatever its arguments say, and the hit names the path.
    with tripwire.expect_trip("binary-load") as hits:
        run_source(tmp_path, "import numpy as np\nnp.load(PLAIN)", stored, PLAIN=str(fake_data.plain))
    assert hits[0].path == str(fake_data.plain) and "on " in str(hits[0])


def test_writing_a_zip_is_not_a_read_unless_it_lands_under_data(tripwire, tmp_path, stored, fake_data):
    maker = "import zipfile\nzipfile.ZipFile(OUT, 'w').close()"
    run_source(tmp_path, maker, stored, OUT=str(tmp_path / "elsewhere.zip"))
    with tripwire.expect_trip("raw-reader"):
        run_source(tmp_path, maker, stored, OUT=str(fake_data.root / "planted.zip"))


def test_data_paths_resolve_every_form_of_path(tmp_path, fake_data):
    inside, outside = fake_data.table, tmp_path / "elsewhere.parquet"
    assert wire.data_paths((str(inside),), {}) == [str(inside)]
    assert wire.data_paths((inside,), {}) == [str(inside)]  # a Path
    assert wire.data_paths(([str(outside), inside],), {}) == [str(inside)]  # a list of paths
    assert wire.data_paths((), {"source": inside}) == [str(inside)]  # a path keyword
    assert wire.data_paths((), {"columns": [str(inside)]}) == []  # other keywords are not paths
    assert wire.data_paths((str(outside),), {}) == []
    assert wire.data_paths((str(fake_data.root / "x" / ".." / "plain.npy"),), {}) != []  # .. is resolved
    assert wire.data_paths((str(fake_data.root.parent / "data_not"),), {}) == []  # a sibling with the same prefix
    with inside.open("rb") as handle:  # a file-like object remembers its name
        assert wire.data_paths((handle,), {}) == [str(inside)]
    assert wire.data_paths((io.BytesIO(b"x"),), {}) == []
    assert wire.data_paths(("bad\0name",), {}) == []  # a path that cannot exist is not a crash


def test_data_paths_follow_symlinks(tmp_path, fake_data):
    link = tmp_path / "innocent_name.parquet"
    try:
        link.symlink_to(fake_data.table)
    except (OSError, NotImplementedError):
        pytest.skip("this account may not create symbolic links")
    assert wire.data_paths((str(link),), {}) == [str(link)]


def test_the_default_data_roots_are_the_repository_data_and_the_configured_root(monkeypatch, tmp_path):
    monkeypatch.setenv("OLSPRIME_DATA_DIR", str(tmp_path / "elsewhere"))
    roots = wire.data_roots()
    assert any(r.rstrip(os.sep).endswith(os.sep + "data") for r in roots)
    assert os.path.normcase(os.path.realpath(tmp_path / "elsewhere")) + os.sep in roots


# --- Audit A, D2: the seal cannot be switched off ------------------------------------------------

SEAL_PATCHES = [
    ("monkeypatch-vault_open", "mp.setattr(protocol, 'vault_open', lambda: True)"),
    ("monkeypatch-computed-name", "mp.setattr(protocol, 'vault_' + 'open', lambda: True)"),
    ("monkeypatch-VAULT", "mp.setattr(protocol, 'VAULT', protocol.Period(protocol.VAULT.end, protocol.VAULT.end.replace(year=2099)))"),
    ("monkeypatch-the-seal", "mp.setattr(protocol, 'seal', lambda frame, **kwargs: frame)"),
    ("monkeypatch-load_klines", "mp.setattr(protocol, 'load_klines', lambda *a, **k: None)"),
    ("monkeypatch-load_klines_file", "mp.setattr(protocol, 'load_klines_file', lambda *a, **k: None)"),
    ("monkeypatch-load_universe", "mp.setattr(protocol, 'load_universe', lambda *a, **k: None)"),
    ("monkeypatch-string-target", "mp.setattr('olsprime.protocol.vault_open', lambda: True)"),
    ("builtin-setattr", "setattr(protocol, 'vault_open', lambda: True)"),
    ("assignment", "protocol.vault_open = lambda: True"),
    ("assignment-VAULT", "protocol.VAULT = None"),
    ("delattr", "delattr(protocol, 'vault_open')"),
    ("mock-patch-object", "mock.patch.object(protocol, 'vault_open', return_value=True).start()"),
    ("mock-patch-string", "mock.patch('olsprime.protocol.vault_open', return_value=True).start()"),
]


@pytest.mark.parametrize(("name", "statement"), SEAL_PATCHES, ids=[p[0] for p in SEAL_PATCHES])
def test_nothing_can_rebind_the_seal_from_a_file_without_the_seal_allowance(name, statement, tripwire, tmp_path, stored):
    source = (
        "import pytest\nfrom unittest import mock\nfrom olsprime import protocol\n"
        "mp = pytest.MonkeyPatch()\n" + statement + "\n"
    )
    module = importlib.import_module("olsprime.protocol")
    before = {n: getattr(module, n) for n in wire.SEAL_NAMES}
    with tripwire.expect_trip("seal"):
        run_source(tmp_path, source, stored)
    assert {n: getattr(module, n) for n in wire.SEAL_NAMES} == before  # nothing was changed


def test_putting_the_original_back_is_always_allowed_and_other_names_are_free(tripwire, tmp_path, stored):
    source = (
        "from olsprime import protocol\n"
        "protocol.vault_open = protocol.vault_open\n"  # what an undo does
        "protocol.SOMETHING_ELSE = 1\ndel protocol.SOMETHING_ELSE\n"
    )
    run_source(tmp_path, source, stored)
    assert tripwire.hits == []


def test_the_seal_allowance_is_what_lets_a_file_patch_the_seal():
    assert is_allowed(str(PYTHON_ROOT / "tests" / "test_protocol.py"), "seal")
    assert is_allowed(str(PYTHON_ROOT / "olsprime" / "protocol.py"), "seal")
    assert not is_allowed(str(PYTHON_ROOT / "tests" / "test_tripwire.py"), "seal")
    assert not is_allowed(str(PYTHON_ROOT / "olsprime" / "features" / "x.py"), "seal")


def test_a_seal_forced_open_by_the_module_dict_is_caught_by_its_result(tripwire, tmp_path, stored):
    """``vars(protocol)[...] = ...`` is not refused, but the sealed read returns vault rows and fails the test."""
    source = (
        "from olsprime import protocol\nfrom pathlib import Path\n"
        "table = vars(protocol)\nreal = table['vault_open']\n"
        "table['vault_open'] = lambda: True\n"
        "try:\n    protocol.load_klines_file(Path(PATH), '1h', columns=['open_time'], caller='bypass')\n"
        "finally:\n    table['vault_open'] = real\n"
    )
    with tripwire.expect_trip("seal") as hits:
        run_source(tmp_path, source, stored)
    assert "VAULT.start" in hits[0].what


def test_a_moved_vault_start_is_caught_by_its_result_too(tripwire, tmp_path, stored):
    source = (
        "from datetime import UTC, datetime\nfrom pathlib import Path\nfrom olsprime import protocol\n"
        "table = vars(protocol)\nreal = table['VAULT']\n"
        "table['VAULT'] = protocol.Period(datetime(2030, 1, 1, tzinfo=UTC), datetime(2031, 1, 1, tzinfo=UTC))\n"
        "try:\n    protocol.load_klines_file(Path(PATH), '1h', columns=['open_time'], caller='bypass')\n"
        "finally:\n    table['VAULT'] = real\n"
    )
    with tripwire.expect_trip("seal"):
        run_source(tmp_path, source, stored)


def test_with_both_keys_the_vault_rows_come_back_and_nothing_trips(tripwire, stored, monkeypatch):
    open_vault(monkeypatch)
    assert load_klines_file(stored, "1h", columns=["open_time"], caller="final").height == 8
    assert tripwire.hits == []


def test_the_sealed_read_of_the_real_loaders_is_untouched(tripwire, stored):
    assert load_klines_file(stored, "1h", columns=["open_time"], caller="test").height == 4
    assert tripwire.hits == []
    assert wire.SEAL_NAMES == ("seal", "vault_open", "VAULT", "load_klines", "load_klines_file", "load_universe")


# --- The mechanism: methods of a class, on a stand-in library --------------------------------------


def test_a_class_with_wrapped_methods_is_checked_at_the_method_not_at_construction(standin, monkeypatch):
    library, _ = standin

    class Format:
        def make_fragment(self, path):
            return f"fragment {path}"

        @classmethod
        def from_paths(cls, paths):
            return f"dataset {paths}"

    library.Format = Format
    local = Tripwire()
    target = Target("standin_frames", "Format", "reader", methods=("make_fragment", "from_paths"), gate_init=False)
    monkeypatch.setattr(wire, "TARGETS", [target])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        wire.install(patches, local)
        fmt = library.Format()  # constructing reads nothing
        assert isinstance(fmt, Format) and local.hits == []
        with pytest.raises(TripwireError):
            fmt.make_fragment("x")
        with pytest.raises(TripwireError):
            library.Format.from_paths(["x"])
        assert [h.what for h in local.hits] == ["standin_frames.Format.make_fragment", "standin_frames.Format.from_paths"]
    assert library.Format is Format and Format().make_fragment("x") == "fragment x"


def test_zip_files_are_gated_only_for_reading():
    assert wire._zip_reads("a.zip") and wire._zip_reads("a.zip", "r") and wire._zip_reads("a.zip", mode="a")
    assert not wire._zip_reads("a.zip", "w") and not wire._zip_reads("a.zip", mode="x")


def test_every_new_entry_point_is_wrapped():
    wrapped = {t.label for t in wire.TARGETS}
    for label in (
        "zipfile.ZipFile",
        "polars.read_csv",
        "polars.scan_csv",
        "polars.read_ndjson",
        "pyarrow.dataset.FileFormat",
        "pyarrow.dataset.ParquetFileFormat",
        "pyarrow.dataset.FileSystemDataset",
        "pyarrow.dataset.FileSystemDatasetFactory",
        "pyarrow.dataset.ParquetDatasetFactory",
    ):
        assert label in wrapped, label
    methods = {t.attribute: t.methods for t in wire.TARGETS if t.module == "pyarrow.dataset"}
    assert methods["FileFormat"] == methods["ParquetFileFormat"] == ("make_fragment",)
    assert methods["FileSystemDataset"] == ("from_paths",)


def test_a_reader_captured_at_import_time_in_a_dict_or_a_partial_is_already_wrapped(tripwire, stored):
    with tripwire.expect_trip("reader") as hits:
        EARLY_READERS["table"](stored)
    assert hits[0].what == "polars.read_parquet"
    with tripwire.expect_trip("reader") as hits:
        EARLY_PARTIAL()
    assert hits[0].what == "polars.scan_parquet"


# --- Round 2: forged keys, raw readers of allowed modules, a tripwire taken down from inside ------------


@pytest.fixture
def vault_tree(tmp_path, monkeypatch, stored):
    """A fake repository ``data/`` holding a vault: 4 sealed and 10 vault daily bars, a raw March 2026 archive, a universe."""
    root = tmp_path / "vault_repo" / "data"
    store(root, bars(utc(2025, 12, 28), 14, DAY_MS, "BTCUSDT"), "BTCUSDT", "1d")
    raw = root / "raw" / "spot" / "monthly" / "klines" / "BTCUSDT" / "1d"
    raw.mkdir(parents=True)
    archive = raw / "BTCUSDT-1d-2026-03.zip"
    archive.write_bytes(zip_archive("BTCUSDT-1d-2026-03.csv", klines_csv(month_rows(Month(2026, 3), DAY_MS))))
    universe = root / "parquet" / "universe" / "spot_usdt_top30.parquet"
    universe.parent.mkdir(parents=True)
    times = [utc(2025, 11, 1), utc(2025, 12, 1), utc(2026, 1, 1), utc(2026, 2, 1)]
    pl.DataFrame({"rebalance_time": times, "symbol": ["BTCUSDT"] * 4, "rank": [1] * 4}).with_columns(
        pl.col("rebalance_time").dt.cast_time_unit("ms").dt.replace_time_zone("UTC")
    ).write_parquet(universe)
    resolved = (os.path.normcase(os.path.realpath(root)) + os.sep,)
    monkeypatch.setenv("OLSPRIME_DATA_DIR", str(root))
    monkeypatch.setattr(wire, "data_roots", lambda: resolved)
    monkeypatch.setattr(wire, "repo_data_roots", lambda: resolved)  # this tree stands for the repository's data/
    return types.SimpleNamespace(root=root, table=parquet_path(root / "parquet", "BTCUSDT", "1d"), archive=archive, universe=universe, stored=stored)


PRELUDE = """
import polars as pl
def rows(frame):
    column = 'open_time' if 'open_time' in frame.columns else frame.columns[0]
    return frame.filter(pl.col(column) >= pl.datetime(2026, 1, 1, time_zone='UTC')).height
"""

# The auditor's evasions of round 2 (scratchpad audit-guards), each as a file in tmp_path (outside python/).
# "closed": the route no longer exists (an attribute is gone, an object is frozen): the snippet fails and
# yields nothing. Otherwise the rule that must trip.
AUDIT_B_CASES = [
    # G2: raw readers of an allowed module, all the frames of the stack must hold the allowance
    ("e10-read_klines_zip", "from olsprime.data import binance_vision as bv\nOUT = rows(bv.read_klines_zip(Z, 'BTCUSDT'))", "raw-reader"),
    ("e10b-by-getattr", "from olsprime.data import binance_vision as bv\nOUT = rows(getattr(bv, 'read_' + 'klines_zip')(Z, 'BTCUSDT'))", "raw-reader"),
    ("e9-zipfile.Path", "import zipfile\nOUT = len(zipfile.Path(Z, at='BTCUSDT-1d-2026-03.csv').read_bytes())", "raw-reader"),
    (
        "e9b-zipfile.Path-then-csv",
        "import io, zipfile\nOUT = pl.read_csv(io.BytesIO(zipfile.Path(Z, at='BTCUSDT-1d-2026-03.csv').read_bytes()), has_header=False).height",
        "raw-reader",
    ),
    ("f3-zipfile-base", "import zipfile\nbase = zipfile.ZipFile.__bases__[0]\nOUT = base(Z).namelist()", "closed"),
    # G1: forged keys
    (
        "e16-forge-by-hand",
        "from olsprime.paths import runs_dir\nfrom olsprime.protocol import load_klines\nkey = runs_dir() / 'vault' / 'OPEN'\n"
        "key.parent.mkdir(parents=True, exist_ok=True)\nkey.write_text('2026-11-01\\nmy model\\n')\n"
        "mp.setenv('OLSPRIME_OPEN_VAULT', '1')\nOUT = rows(load_klines('BTCUSDT', '1d', caller='e16'))",
        "seal",
    ),
    (
        "e16b-open_vault-helper",
        "from tests.test_protocol import open_vault\nfrom olsprime.protocol import load_klines\nopen_vault(mp)\nOUT = rows(load_klines('BTCUSDT', '1d', caller='e16b'))",
        "seal",
    ),
    (
        "e16c-forged-universe",
        "from tests.test_protocol import open_vault\nfrom olsprime.protocol import load_universe\nopen_vault(mp)\nOUT = load_universe(U, caller='e16c').height",
        "seal",
    ),
    (
        "e16d-forged-file",
        "from tests.test_protocol import open_vault\nfrom olsprime.protocol import load_klines_file\nopen_vault(mp)\nOUT = rows(load_klines_file(P, '1d', caller='e16d'))",
        "seal",
    ),
    # G3 and the Minor ones: the originals are no longer one attribute away
    ("e1-seal-wrapped", "from olsprime import protocol\nprotocol.seal = protocol.seal.__wrapped__", "closed"),
    ("e2-globals-seal", "from olsprime.protocol import load_klines\ng = load_klines.__globals__\ng['seal'] = g['seal'].__wrapped__", "closed"),
    ("e3-reset-module-class", "import types\nfrom olsprime import protocol\nprotocol.__class__ = types.ModuleType", "seal"),
    ("e3b-delete-module-class", "from olsprime import protocol\ndel protocol.__class__", "seal"),
    (
        "e4-patch-is_allowed",
        "import pytest\nfrom tests import tripwire\nmp2 = pytest.MonkeyPatch()\nmp2.setattr(tripwire, 'is_allowed', lambda f, r: True)\n"
        "try:\n    OUT = rows(pl.read_parquet(P))\nfinally:\n    mp2.undo()",
        "reader",
    ),
    ("e4b-edit-the-allow-list", "from tests.allowlist import ALLOWED\nALLOWED['tests/test_models.py'] = {'reader': 'mine'}", "closed"),
    ("e4c-edit-an-inner-map", "from tests.allowlist import ALLOWED\nALLOWED['tests/test_tripwire.py']['reader'] = 'mine'", "closed"),
    ("e6-wrapped-reader", "OUT = rows(pl.read_parquet.__wrapped__(P))", "closed"),
    ("f1-scan_parquet-wrapped", "OUT = rows(pl.scan_parquet.__wrapped__(P).collect())", "closed"),
    ("f2-read_table-wrapped", "import pyarrow.parquet as pq\nOUT = rows(pl.from_arrow(pq.read_table.__wrapped__(P)))", "closed"),
    ("e7-ParquetFile-base", "import pyarrow.parquet as pq\nbase = pq.ParquetFile.__bases__[0]\nOUT = rows(pl.from_arrow(base(P).read()))", "closed"),
    (
        "f6-spoof-then-runpy",
        "import pytest, runpy\nfrom tests import tripwire\nmp2 = pytest.MonkeyPatch()\nmp2.setattr(tripwire, 'is_allowed', lambda f, r: True)\n"
        "open(HELPER, 'w').write('import polars as pl\\nDF = pl.read_parquet(P)\\n')\n"
        "try:\n    OUT = rows(runpy.run_path(HELPER, init_globals={'P': P})['DF'])\nfinally:\n    mp2.undo()",
        "reader",
    ),
    # Readers the tripwire already caught, kept as regressions
    ("e11-bytes-then-bytesio", "import io\nOUT = rows(pl.read_parquet(io.BytesIO(open(P, 'rb').read())))", "reader"),
    (
        "e11b-bufferreader",
        "import pyarrow as pa, pyarrow.parquet as pq\nOUT = rows(pl.from_arrow(pq.ParquetFile(pa.BufferReader(open(P, 'rb').read())).read()))",
        "reader",
    ),
    (
        "e11c-numpy-fromfile",
        "import io\nimport numpy as np\nOUT = rows(pl.read_parquet(io.BytesIO(np.fromfile(P, dtype=np.uint8).tobytes())))",
        "reader",
    ),
    ("e12-copy-then-read", "import shutil\nshutil.copy(P, COPY)\nOUT = rows(pl.read_parquet(COPY))", "reader"),
    ("e13-scan_pyarrow_dataset", "import pyarrow.dataset as ds\nOUT = rows(pl.scan_pyarrow_dataset(ds.dataset(P)).collect())", "reader"),
    (
        "e14-os-read",
        "import io, os\nfd = os.open(P, os.O_RDONLY | getattr(os, 'O_BINARY', 0))\nbuf = os.read(fd, 10**6)\nOUT = rows(pl.read_parquet(io.BytesIO(buf)))",
        "reader",
    ),
    ("s1-attrgetter", "import operator\nOUT = rows(operator.attrgetter('read_' + 'parquet')(pl)(P))", "reader"),
    ("s2-reduce-getattr", "import functools\nOUT = rows(functools.reduce(getattr, ['read_' + 'parquet'], pl)(P))", "reader"),
    ("s3-getattribute", "OUT = rows(pl.__getattribute__('read_' + 'parquet')(P))", "reader"),
    ("s4-builtins-import", "zf = __builtins__['__import__']('zip' + 'file')\nOUT = zf.ZipFile(Z).namelist()", "raw-reader"),
    ("s5-import-in-a-function", "def lib():\n    import polars\n    return polars\nOUT = rows(lib().read_parquet(P))", "reader"),
]


@pytest.mark.parametrize(("name", "source", "expect"), AUDIT_B_CASES, ids=[c[0] for c in AUDIT_B_CASES])
def test_the_second_audit_evasions_trip_or_are_closed(name, source, expect, tripwire, tmp_path, vault_tree, monkeypatch):
    names = {
        "P": str(vault_tree.table), "Z": str(vault_tree.archive), "U": str(vault_tree.universe),
        "COPY": str(tmp_path / "innocent.parquet"), "HELPER": str(tmp_path / "inner_helper.py"), "mp": monkeypatch,
    }  # fmt: skip
    protocol = importlib.import_module("olsprime.protocol")
    before = {n: protocol.__dict__[n] for n in wire.SEAL_NAMES}
    if expect == "closed":
        with pytest.raises((AttributeError, KeyError, TypeError)):
            run_source(tmp_path, PRELUDE + source, vault_tree.stored, **names)
        assert tripwire.hits == []
    else:
        outcome = {}
        with tripwire.expect_trip(expect):
            outcome = run_source(tmp_path, PRELUDE + source, vault_tree.stored, **names)
        assert not outcome.get("OUT"), "the evasion got vault rows out"
    assert {n: protocol.__dict__[n] for n in wire.SEAL_NAMES} == before


def test_a_sealed_read_from_a_copy_of_the_file_comes_back_sealed(tripwire, tmp_path, vault_tree):
    """e12b: copying the vault file and loading the copy through the loader gives the sealed frame, and no trip."""
    copy = tmp_path / "innocent.parquet"
    copy.write_bytes(vault_tree.table.read_bytes())
    frame = load_klines_file(copy, "1d", caller="e12b")
    assert frame.height == 4 and tripwire.hits == []


def test_load_klines_never_finds_the_repository_data_under_pytest(tripwire):
    """G1, step 1: the data root of every test is an empty folder of its own."""
    root = Path(data_dir())
    assert root != REPO_ROOT / "data" and not any(root.iterdir())
    assert not str(root).startswith(str(REPO_ROOT / "data"))


def test_forged_keys_open_nothing_of_the_repository_vault_but_the_vault_open_allowance_does(tripwire, vault_tree, monkeypatch):
    open_vault(monkeypatch)  # both keys, forged
    with tripwire.expect_trip("seal"):
        load_klines("BTCUSDT", "1d", caller="forged")
    with tripwire.expect_trip("seal"):
        load_klines_file(vault_tree.table, "1d", caller="forged")
    # A file that holds the vault-open allowance (no file does today) may read it, and the read is logged.
    monkeypatch.setattr(wire, "is_allowed", lambda filename, rule: rule == wire.VAULT_OPEN)
    assert load_klines("BTCUSDT", "1d", caller="final").height == 14
    assert tripwire.hits == []


def test_forged_keys_in_a_tree_that_is_not_the_repository_data_are_not_refused(tripwire, tmp_path, monkeypatch, stored):
    """The path rule is about the repository's data/: a fake vault in tmp_path stays testable (test_manifest does)."""
    open_vault(monkeypatch)
    assert load_klines_file(stored, "1h", columns=["open_time"], caller="final").height == 8
    assert tripwire.hits == []


def test_the_result_check_does_not_depend_on_the_seal_global(tripwire, tmp_path, vault_tree):
    """The seal replaced in the module dict (no setattr), without the keys: the loader's result still trips."""
    source = (
        "from olsprime import protocol\ntable = vars(protocol)\nreal = table['seal']\n"
        "table['seal'] = lambda frame, **kwargs: frame\n"
        "try:\n    protocol.load_klines_file(P, '1d', caller='bypass')\nfinally:\n    table['seal'] = real\n"
    )
    with tripwire.expect_trip("seal") as hits:
        run_source(tmp_path, source, vault_tree.stored, P=str(vault_tree.table))
    assert "VAULT.start" in hits[0].what


def test_the_universe_loader_is_watched_too(tripwire, tmp_path, vault_tree):
    source = (
        "from olsprime import protocol\ntable = vars(protocol)\nreal = table['seal']\n"
        "table['seal'] = lambda frame, **kwargs: frame\n"
        "try:\n    protocol.load_universe(U, caller='bypass')\nfinally:\n    table['seal'] = real\n"
    )
    with tripwire.expect_trip("seal"):
        run_source(tmp_path, source, vault_tree.stored, U=str(vault_tree.universe))


# --- The stdlib acts for its caller; the path rule stays (M5) ---------------------------------------


def test_a_metadata_style_read_of_a_zip_outside_data_does_not_trip(tripwire, tmp_path, vault_tree, monkeypatch):
    """importlib.metadata opens the zips of sys.path with zipfile.Path: not stored data, and not the test's doing."""
    import importlib.metadata

    folder = tmp_path / "eggs"
    folder.mkdir()
    meta = b"Metadata-Version: 2.1\nName: zzmeta\nVersion: 1.0\n"
    (folder / "zzmeta.zip").write_bytes(zip_archive("zzmeta-1.0.dist-info/METADATA", meta))
    monkeypatch.syspath_prepend(str(folder / "zzmeta.zip"))
    assert importlib.metadata.version("zzmeta") == "1.0"
    assert tripwire.hits == []


def test_zipfile_path_over_an_archive_under_data_trips(tripwire, tmp_path, vault_tree):
    run = "import zipfile\nzipfile.Path(Z, at='BTCUSDT-1d-2026-03.csv').read_bytes()"
    with tripwire.expect_trip("raw-reader") as hits:
        run_source(tmp_path, run, vault_tree.stored, Z=str(vault_tree.archive))
    assert hits[0].path == str(vault_tree.archive) and Path(hits[0].caller).name == "helper_script.py"


def test_a_zipfile_path_outside_data_from_an_ordinary_file_still_trips_at_that_file(tripwire, tmp_path, stored):
    """Not through importlib.metadata, the caller is the file that asked: no allowance, a trip (the stdlib is not blamed)."""
    with tripwire.expect_trip("raw-reader") as hits:
        run_source(tmp_path, "import zipfile\nzipfile.Path(FOLDER + '/archive.zip')", stored)
    assert Path(hits[0].caller).name == "helper_script.py"


def test_stdlib_frames_are_infrastructure_but_site_packages_are_not():
    skipped = wire.infrastructure()
    assert wire.is_infrastructure(json.__file__, skipped)
    assert wire.is_infrastructure(str(Path(importlib.__file__)), skipped)
    assert not wire.is_infrastructure(str(PYTHON_ROOT / "tests" / "test_x.py"), skipped)
    assert not wire.is_infrastructure("<string>", skipped)


# --- A tripwire taken down from inside is noticed (G3) -------------------------------------------


def test_the_allow_list_is_frozen_at_both_levels():
    with pytest.raises(TypeError):
        ALLOWED["tests/test_models.py"] = {"reader": "mine"}
    with pytest.raises(TypeError):
        ALLOWED["tests/test_tripwire.py"]["reader"] = "mine"
    with pytest.raises(TypeError):
        del ALLOWED["tests/test_tripwire.py"]
    assert wire.is_allowed.__defaults__ == (ALLOWED,)  # bound when the module was imported


def test_patching_the_allow_check_changes_nothing_for_the_gates_already_up(tripwire, tmp_path, stored, monkeypatch):
    monkeypatch.setattr(wire, "is_allowed", lambda filename, rule: True)
    monkeypatch.setattr(wire, "calling_file", lambda skipped, depth=1: str(PYTHON_ROOT / "olsprime" / "protocol.py"))
    with tripwire.expect_trip("reader"):
        run_source(tmp_path, "import polars as pl\npl.read_parquet(PATH)", stored)


@pytest.fixture
def clone_guard(monkeypatch):
    """A copy of ``olsprime.protocol`` under a local Tripwire: the guard class and the watching wrappers, on a module of its own."""
    protocol = importlib.import_module("olsprime.protocol")
    clone = types.ModuleType("olsprime.protocol")
    clone.__dict__.update(protocol.__dict__)

    def fresh(function):  # distinct objects, so that only the clone is found as an alias
        def stand_in(*args, **kwargs):
            return function(*args, **kwargs)

        stand_in.__name__ = function.__name__
        return stand_in

    for name in wire.WATCHED_NAMES:
        clone.__dict__[name] = fresh(protocol.__dict__[name])
    local = Tripwire()
    with pytest.MonkeyPatch.context() as patches:
        patches.setitem(sys.modules, "olsprime.protocol", clone)
        wire.install_seal_watch(patches, local)
        yield clone, local
        wire.restore_protocol_class(local)


def test_a_standing_guard_has_no_integrity_problem(clone_guard):
    clone, local = clone_guard
    assert wire.integrity_problems(local) == []


@pytest.mark.parametrize("name", wire.WATCHED_NAMES)
def test_only_the_watching_wrapper_may_stand_under_a_watched_name(clone_guard, name):
    clone, local = clone_guard
    wrapper = clone.__dict__[name]
    with pytest.raises(TripwireError):
        setattr(clone, name, lambda *a, **k: None)
    with pytest.raises(TripwireError):
        delattr(clone, name)
    assert clone.__dict__[name] is wrapper and local.hits


def test_the_original_seal_cannot_be_put_back(clone_guard):
    """M1: the wrapper has no __wrapped__, and the permitted values are the wrappers only."""
    clone, local = clone_guard
    assert not hasattr(clone.__dict__["seal"], "__wrapped__")
    assert local.permitted["seal"] == [clone.__dict__["seal"]]


@pytest.mark.parametrize("dunder", ["__class__", "__dict__"])
def test_the_module_class_and_dict_are_protected(clone_guard, dunder):
    clone, local = clone_guard
    with pytest.raises(TripwireError):
        setattr(clone, dunder, types.ModuleType if dunder == "__class__" else {})
    with pytest.raises(TripwireError):
        delattr(clone, dunder)
    assert type(clone) is local.guard_class


def test_integrity_notices_a_class_reset_a_dict_write_and_an_edited_guard(clone_guard):
    clone, local = clone_guard
    types.ModuleType.__setattr__(clone, "__class__", types.ModuleType)  # the route the guard itself refuses
    assert any("guard class" in p for p in wire.integrity_problems(local))
    types.ModuleType.__setattr__(clone, "__class__", local.guard_class)
    assert wire.integrity_problems(local) == []
    original = clone.__dict__["seal"]
    clone.__dict__["seal"] = lambda *a, **k: None  # what a reload does: the dict, not setattr
    assert any("protocol.seal" in p for p in wire.integrity_problems(local))
    clone.__dict__["seal"] = original
    assert wire.integrity_problems(local) == []


def test_integrity_notices_undone_patches_a_replaced_function_and_a_changed_allow_list(monkeypatch, standin):
    local = Tripwire()
    targets = [Target("standin_frames", "read_frame", "reader"), Target("standin_frames", "Dataset", "reader")]
    monkeypatch.setattr(wire, "TARGETS", targets)
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    patches = pytest.MonkeyPatch()
    wire.install(patches, local)
    assert wire.integrity_problems(local) == []
    patches.undo()  # what tests.conftest.PATCHES.undo() does, on the local tripwire
    problems = wire.integrity_problems(local)
    assert any("read_frame" in p for p in problems) and any("__init__" in p for p in problems)
    with pytest.MonkeyPatch.context() as inner:
        inner.setattr(wire, "is_allowed", lambda filename, rule: True)
        assert "tests.tripwire.is_allowed was replaced" in wire.integrity_problems(Tripwire())
    assert wire.integrity_problems(Tripwire()) == []
    local.allowed_snapshot = {}
    assert "the allow-list changed" in wire.integrity_problems(local)


def test_the_globals_of_a_watched_loader_are_the_tripwires_not_the_protocols(tripwire, vault_tree):
    """e2b: writing 'vault_open' there changes nothing of the seal (and the name is removed again)."""
    table = load_klines.__globals__
    assert table is vars(wire) and "vault_open" not in table
    table["vault_open"] = lambda: True
    try:
        assert load_klines("BTCUSDT", "1d", caller="e2b").height == 4
    finally:
        del table["vault_open"]


# --- Round 3: the repository's data/ is read through protocol.py only; zipimport; a copy of a vault file ---------

# A path under the repository's real data/ that does not exist: a gate that misses ends in FileNotFoundError,
# and no real file is ever read.
PROBE = REPO_ROOT / "data" / "parquet" / "__tripwire_probe__.parquet"


def run_as(filename: str, source: str, **names) -> None:
    """Run ``source`` as if it were written in ``python/tests/<filename>`` (the frame's file decides the allowance)."""
    exec(compile(source, str(PYTHON_ROOT / "tests" / filename), "exec"), dict(names))


R1_READS = [
    ("polars-read_parquet", "import polars as pl\npl.read_parquet(PROBE)"),
    ("polars-scan-collect", "import polars as pl\npl.scan_parquet(PROBE).collect()"),
    ("pyarrow-read_table", "import pyarrow.parquet as pq\npq.read_table(PROBE)"),
    ("pyarrow-ParquetFile", "import pyarrow.parquet as pq\npq.ParquetFile(PROBE)"),
    ("polars-read_ipc", "import polars as pl\npl.read_ipc(PROBE)"),
    ("numpy-load-without-pickle", "import numpy as np\nnp.load(PROBE)"),
    ("pyarrow-dataset", "import pyarrow.dataset as ds\nds.dataset(str(PROBE))"),
]


@pytest.mark.parametrize(("name", "source"), R1_READS, ids=[r[0] for r in R1_READS])
@pytest.mark.parametrize("holder", ["test_universe.py", "test_binance_vision.py", "test_protocol.py"])
def test_a_file_allowed_reader_may_not_read_the_repositorys_data(name, source, holder, tripwire):
    assert not PROBE.exists()
    with tripwire.expect_trip() as hits:
        run_as(holder, source, PROBE=PROBE)
    assert hits[0].path == str(PROBE) and Path(hits[0].caller).name == holder


def test_the_same_file_reads_back_what_it_wrote_elsewhere(tripwire, tmp_path):
    """Control: from the same file, a path outside the repository's data/ is judged by the allowance as before."""
    missing = tmp_path / "written_here.parquet"
    with pytest.raises(FileNotFoundError):  # allowed, so the read happens (and finds nothing)
        run_as("test_universe.py", "import polars as pl\npl.read_parquet(P)", P=missing)
    assert tripwire.hits == []


def test_protocol_py_is_the_one_caller_that_may_name_the_repositorys_data(tripwire):
    """The loaders read through protocol.py: the read is not refused there (it fails on the missing file)."""
    with pytest.raises(FileNotFoundError):
        load_klines_file(PROBE, "1d", caller="probe")
    assert tripwire.hits == []


# --- zipimport ------------------------------------------------------------------------------------


def test_zipimporter_over_a_raw_archive_under_the_data_root_trips(tripwire, tmp_path, stored):
    raw = Path(data_dir()) / "raw" / "spot" / "monthly" / "klines" / "BTCUSDT" / "1d"
    raw.mkdir(parents=True)
    archive = raw / "BTCUSDT-1d-2026-03.zip"
    archive.write_bytes(zip_archive("BTCUSDT-1d-2026-03.csv", b"1,2,3\n"))
    source = "import zipimport\nOUT = zipimport.zipimporter(ARCHIVE).get_data('BTCUSDT-1d-2026-03.csv')"
    with tripwire.expect_trip("raw-reader") as hits:
        run_source(tmp_path, source, stored, ARCHIVE=str(archive))
    assert hits[0].what == "zipimport.zipimporter" and hits[0].path == str(archive)


def test_zipimporter_over_a_copy_outside_data_is_not_judged(tripwire, tmp_path, stored):
    copy = tmp_path / "copy.zip"
    copy.write_bytes(zip_archive("member.csv", b"1,2,3\n"))
    out = run_source(tmp_path, "import zipimport\nOUT = zipimport.zipimporter(COPY).get_data('member.csv')", stored, COPY=str(copy))
    assert out["OUT"] == b"1,2,3\n" and tripwire.hits == []


def test_modules_still_import_from_a_zip_on_sys_path(tripwire, tmp_path, monkeypatch):
    """The import system builds zipimporters from the same class (sys.path_hooks): the gate must not stand in its way."""
    archive = tmp_path / "libs.zip"
    archive.write_bytes(zip_archive("zz_from_a_zip.py", b"VALUE = 41 + 1\n"))
    monkeypatch.syspath_prepend(str(archive))
    monkeypatch.delitem(sys.modules, "zz_from_a_zip", raising=False)
    importlib.invalidate_caches()
    module = importlib.import_module("zz_from_a_zip")
    assert module.VALUE == 42 and tripwire.hits == []
    sys.modules.pop("zz_from_a_zip", None)
    # and the zips of sys.path that do not exist (python312.zip on Windows) are tried without a trip
    missing = str(tmp_path / "python_missing.zip")
    monkeypatch.syspath_prepend(missing)
    sys.path_importer_cache.pop(missing, None)
    assert importlib.util.find_spec("zz_not_there_at_all") is None and tripwire.hits == []


# --- A copy of a vault file with forged keys (R3) -------------------------------------------------


def run_alone_as(filename: str, source: str, **names) -> None:
    """Like ``run_as``, in a thread of its own, so that this test file is not on the stack: the caller is ``filename`` alone."""
    errors: list[BaseException] = []
    lines = ["try:", *("    " + line for line in source.splitlines()), "except BaseException as error:", "    ERRORS.append(error)"]
    code = compile("\n".join(lines) + "\n", str(PYTHON_ROOT / "tests" / filename), "exec")
    # exec is the thread's target itself: no frame of this file is under the snippet's
    thread = threading.Thread(target=exec, args=(code, {**names, 'ERRORS': errors}))
    thread.start()
    thread.join()
    if errors:
        raise errors[0]


def test_forged_keys_and_a_copy_of_a_vault_file_outside_data_trip_from_a_file_that_may_not_touch_the_seal(
    tripwire, tmp_path, vault_tree, monkeypatch
):
    copy = tmp_path / "innocent.parquet"
    copy.write_bytes(vault_tree.table.read_bytes())
    open_vault(monkeypatch)
    source = "from olsprime.protocol import load_klines_file\nOUT = load_klines_file(P, '1d', caller='r3').height"
    with tripwire.expect_trip("seal") as hits:
        run_alone_as("test_universe.py", source, P=str(copy))
    assert "forged" in hits[0].what and Path(hits[0].caller).name == "test_universe.py"
    # the files that may open a fake vault (this one holds seal-patch) keep their fake-vault tests
    assert load_klines_file(copy, "1d", caller="fake vault").height == 14 and tripwire.hits == []
    run_alone_as("test_manifest.py", source, P=str(copy))  # test_manifest holds seal-patch
    assert tripwire.hits == []


def test_undoing_the_seal_patches_from_a_test_trips_at_the_class_restore(tripwire):
    """The session's MonkeyPatch undone from a test: the last patch to go (the module class) is refused, at once."""
    protocol = importlib.import_module("olsprime.protocol")
    clone = types.ModuleType("olsprime.protocol")
    clone.__dict__.update(protocol.__dict__)
    local = Tripwire()
    patches = pytest.MonkeyPatch()
    patches.setitem(sys.modules, "olsprime.protocol", clone)
    wire.install_seal_watch(patches, local)
    try:
        with pytest.raises(TripwireError):
            patches.undo()
        assert type(clone) is local.guard_class and local.hits
    finally:
        wire.restore_protocol_class(local)
        patches.undo()


# --- Round 4: path spellings the repository-data rule must recognise ---------------------------------

R4_SPELLINGS = [
    ("file-uri", "import polars as pl\npl.read_parquet(PROBE.as_uri())"),
    ("extended-length-prefix", "import polars as pl\npl.read_parquet('\\\\\\\\?\\\\' + str(PROBE))"),
    ("extended-length-forward", "import polars as pl\npl.read_parquet('//?/' + str(PROBE).replace(chr(92), '/'))"),
    ("uri-with-the-drive-as-host", "import polars as pl\npl.scan_parquet('file://' + str(PROBE).replace(chr(92), '/')).collect()"),
    ("uri-through-pyarrow", "import pyarrow.parquet as pq\npq.read_table(PROBE.as_uri())"),
    (
        "relative-to-a-subtree-filesystem",
        "import pyarrow.dataset as ds, pyarrow.fs as fs\n"
        "ds.dataset(PROBE.name, filesystem=fs.SubTreeFileSystem(str(PROBE.parent), fs.LocalFileSystem()))",
    ),
    (
        "subtree-filesystem-positional",
        "import pyarrow.dataset as ds, pyarrow.fs as fs\n"
        "ds.dataset(PROBE.name, None, None, fs.SubTreeFileSystem(str(PROBE.parent), fs.LocalFileSystem()))",
    ),
]
EXTENDED = "\\\\?\\"  # the extended-length prefix, \\?\


@pytest.mark.parametrize(("name", "source"), R4_SPELLINGS, ids=[r[0] for r in R4_SPELLINGS])
def test_a_file_allowed_reader_may_not_read_the_repositorys_data_by_another_spelling(name, source, tripwire):
    assert not PROBE.exists()
    with tripwire.expect_trip() as hits:
        run_as("test_universe.py", source, PROBE=PROBE)
    assert Path(hits[0].caller).name == "test_universe.py"


def test_spellings_of_a_path_outside_the_roots_stay_outside(tmp_path):
    outside = tmp_path / "x.parquet"
    roots = (os.path.normcase(os.path.realpath(tmp_path / "elsewhere")) + os.sep,)
    assert wire.data_paths((outside.as_uri(),), {}, roots) == []
    assert wire.data_paths((EXTENDED + str(outside),), {}, roots) == []
    assert wire.data_paths(("x.parquet",), {"filesystem": types.SimpleNamespace(base_path=str(tmp_path))}, roots) == []


def test_spelled_paths_map_onto_the_file_system_paths(tmp_path):
    inside = tmp_path / "data" / "x.parquet"
    roots = (os.path.normcase(os.path.realpath(tmp_path / "data")) + os.sep,)
    assert wire.data_paths((inside.as_uri(),), {}, roots) == [inside.as_uri()]
    assert wire.data_paths((EXTENDED + str(inside),), {}, roots) == [EXTENDED + str(inside)]
    # The UNC root takes the platform's separator. Off Windows the prefix is stripped too, which errs towards a trip.
    assert wire.spelled_path(EXTENDED + "UNC\\server\\share\\f") == os.sep * 2 + "server\\share\\f"
    assert wire.spelled_path("plain/relative.parquet") == "plain/relative.parquet"
    if os.name == "nt":
        assert wire.spelled_path("file:///C:/a%20b/c") == "C:\\a b\\c"


def test_a_pyarrow_native_file_has_no_name_and_stays_a_stated_limit(tmp_path):
    """The handle of ``LocalFileSystem().open_input_file`` carries no path: the path rule cannot see where it came from."""
    pafs = importlib.import_module("pyarrow.fs")

    target = tmp_path / "data" / "x.bin"
    target.parent.mkdir()
    target.write_bytes(b"x")
    roots = (os.path.normcase(os.path.realpath(tmp_path / "data")) + os.sep,)
    with pafs.LocalFileSystem().open_input_file(str(target)) as handle:
        assert wire.data_paths((handle,), {}, roots) == []
