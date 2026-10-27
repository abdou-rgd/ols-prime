"""The run-time tripwire: each bypass the static guard cannot see must trip, and allowed readers must not."""

import importlib.util
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from olsprime.data.binance_vision import parquet_path
from olsprime.protocol import load_klines_file, stored_row_count
from tests import tripwire as wire
from tests.allowlist import ALLOWED
from tests.fakes import HOUR_MS
from tests.test_protocol import bars, utc
from tests.tripwire import Hit, Target, Tripwire, TripwireError, is_allowed

PYTHON_ROOT = Path(wire.__file__).resolve().parents[1]


@pytest.fixture
def stored(tmp_path) -> Path:
    """A real Parquet file (and an Arrow IPC file beside it), written with the unpatched writers."""
    path = tmp_path / "data.parquet"
    frame = bars(utc(2025, 12, 31, 20), 8, HOUR_MS)  # 4 bars before the vault, 4 inside
    frame.write_parquet(path)
    frame.write_ipc(tmp_path / "data.arrow")
    np.save(tmp_path / "plain.npy", np.arange(3))
    np.save(tmp_path / "boxed.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    return path


def run_source(tmp_path: Path, source: str, stored: Path) -> dict:
    """Run ``source`` as a file in ``tmp_path`` (outside python/), with ``PATH`` bound to ``stored``."""
    script = tmp_path / "helper_script.py"
    script.write_text(source, encoding="utf-8")
    return runpy.run_path(str(script), init_globals={"PATH": str(stored), "FOLDER": str(stored.parent)})


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
    static_rules = {"reader", "binary-load", "import:pickle"}  # spelled as test_protocol.findings names them
    used = {t.rule for t in wire.TARGETS} | {t.rule for ts in wire.OPTIONAL_TARGETS.values() for t in ts}
    assert used == static_rules


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


def test_a_wrapped_class_is_a_subclass_of_the_original(standin, monkeypatch):
    library, _ = standin
    original = library.Dataset
    local = Tripwire()
    monkeypatch.setattr(wire, "TARGETS", [Target("standin_frames", "Dataset", "reader")])
    monkeypatch.setattr(wire, "OPTIONAL_TARGETS", {})
    with pytest.MonkeyPatch.context() as patches:
        wire.install(patches, local)
        assert issubclass(library.Dataset, original) and library.Dataset.__name__ == "Dataset"
        with pytest.raises(TripwireError):
            library.Dataset("x")
        assert isinstance(library.Dataset.__mro__[1], type) and library.Dataset.__mro__[1] is original


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
