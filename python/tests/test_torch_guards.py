"""The guards against the real torch (iteration 4): ``torch.load`` unpickles, so it is watched like ``pickle.load``.

The run-time tripwire judges the file of the first frame outside the libraries. The probes below are
compiled as if written in this file, which holds no ``import:pickle`` allowance, so the static guard
(which flags ``torch.load`` itself, ``tests/test_protocol.py``) never sees a call in this file's own
code, and the tripwire still judges this file. The static side is in ``test_protocol.py``'s cheaters.
"""

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from olsprime.paths import REPO_ROOT
from olsprime.train import checkpoint

# A path under the repository's real data/ that does not exist: a gate that misses ends in FileNotFoundError,
# and no real file is ever read.
PROBE = REPO_ROOT / "data" / "__torch_probe__.pt"
HERE = str(Path(__file__).resolve())


def run_here(source: str, **names) -> None:
    """Run ``source`` with this file as its frame's file: the tripwire judges this file's allowances."""
    exec(compile(source, HERE, "exec"), {"torch": torch, "io": io, **names})


def saved_tensor() -> bytes:
    buffer = io.BytesIO()
    torch.save({"w": torch.arange(4.0)}, buffer)
    return buffer.getvalue()


def test_torch_is_the_real_library_and_its_load_is_wrapped():
    assert torch.__version__.startswith("2.") and hasattr(torch, "serialization")
    # The wrapper replaced both names of the same function, without a handle to reach behind it.
    # Written as a probe: the static guard flags these names in this file's own code.
    run_here("assert torch.load is torch.serialization.load\n"
             "assert torch.load.__module__ == 'torch.serialization' and not hasattr(torch.load, '__wrapped__')")  # fmt: skip


def test_torch_load_of_a_file_under_the_repositorys_data_trips(tripwire):
    assert not PROBE.exists()
    with tripwire.expect_trip("import:pickle") as hits:
        run_here("torch.load(PROBE)", PROBE=PROBE)
    assert hits[0].path == str(PROBE) and Path(hits[0].caller).resolve() == Path(HERE)
    with tripwire.expect_trip("import:pickle") as hits:
        run_here("torch.load(str(PROBE), weights_only=True)", PROBE=PROBE)
    assert hits[0].path == str(PROBE)


@pytest.mark.parametrize("call", ["torch.load(io.BytesIO(DATA))", "torch.load(io.BytesIO(DATA), weights_only=True)",
                                  "torch.serialization.load(io.BytesIO(DATA))", "torch.load(io.BytesIO(DATA), weights_only=False)"])  # fmt: skip
def test_torch_load_from_a_file_without_the_pickle_allowance_trips(tripwire, call):
    with tripwire.expect_trip("import:pickle") as hits:
        run_here(call, DATA=saved_tensor())
    assert Path(hits[0].caller).resolve() == Path(HERE)


def test_a_checkpoint_loads_without_pickle_and_trips_nothing(tripwire, tmp_path):
    arrays = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.array([1.5])}
    checkpoint.save(tmp_path / "fit.safetensors", arrays, {"config": "{}"})
    back, meta = checkpoint.load(tmp_path / "fit.safetensors")
    assert all(np.array_equal(back[k], v) and back[k].dtype == v.dtype for k, v in arrays.items()) and meta == {"config": "{}"}
    assert tripwire.hits == []
