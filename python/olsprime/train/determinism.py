"""The registered determinism settings (``docs/PROTOCOL.md``, iteration 4, "Determinism").

``torch.use_deterministic_algorithms(True)``, ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, TF32 off,
cuDNN benchmark off, and fixed seeds. :func:`deterministic` applies them and restores the
previous state on exit, so that a test leaves the process as it found it.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _state() -> dict[str, Any]:
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def apply() -> None:
    """Put the registered settings in place (for a whole process)."""
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG is {current!r}, the registered value is {CUBLAS_WORKSPACE_CONFIG!r}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG  # read by cuBLAS when its handle is made
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def deterministic() -> Iterator[None]:
    """The registered settings inside the block; the previous ones after it."""
    before = _state()
    apply()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before["deterministic"], warn_only=before["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
        if before["cublas"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def build(device: torch.device | str) -> dict[str, Any]:
    """The machine and build a run records: torch, CUDA, cuDNN, the device's name, and the settings in force."""
    device = torch.device(device)
    out: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "settings": _state(),
    }
    if device.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(device)
        out["cudnn"] = torch.backends.cudnn.version()
        out["capability"] = list(torch.cuda.get_device_capability(device))
    else:
        out["cpu_threads"] = torch.get_num_threads()
    return out
