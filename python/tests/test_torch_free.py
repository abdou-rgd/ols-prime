"""The CPU evaluation code stays free of torch.

``olsprime.models`` is imported by report2, power, the explainer, the twins and the adversary, and by
every worker of their process pools. On Windows those pools spawn their workers, which import the
package again. An eager ``import torch`` there would load torch in each of them: seconds and hundreds
of MB per process, and a dependency of the CPU evaluation on torch. The encoder lives in
``olsprime.models.encoder``, ``olsprime.models.memory`` and ``olsprime.train``, imported directly.

Each check runs in a fresh interpreter, a process started with ``spawn``, since this test session
may already hold torch. The child reports through its exit code only, so nothing is unpickled here.
This module itself imports no torch: the child imports it to find its target.
"""

import multiprocessing
import sys

import pytest

TORCH_FREE = ("olsprime.models", "olsprime.bench.report2", "olsprime.bench.power")
ABSENT, PRESENT = 0, 3


def _child(module: str) -> None:
    """In the fresh interpreter: import the modules of TORCH_FREE (plain statements: the static guard
    refuses imports by name), plus the encoder for the positive control, then exit PRESENT if torch
    came with them, ABSENT if not."""
    import olsprime.bench.power  # noqa: F401
    import olsprime.bench.report2  # noqa: F401
    import olsprime.models  # noqa: F401

    if module == "olsprime.models.encoder":
        import olsprime.models.encoder  # noqa: F401
    sys.exit(PRESENT if "torch" in sys.modules else ABSENT)


def _exit_code(module: str) -> int | None:
    process = multiprocessing.get_context("spawn").Process(target=_child, args=(module,))
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail(f"the fresh interpreter importing {module} did not finish in 300 s")
    return process.exitcode


def test_the_evaluation_modules_import_no_torch() -> None:
    assert _exit_code("evaluation") == ABSENT, f"importing {', '.join(TORCH_FREE)} loads torch"


def test_the_check_sees_torch_when_it_is_imported() -> None:
    """Positive control: the same child, also importing the encoder, must report torch."""
    assert _exit_code("olsprime.models.encoder") == PRESENT
