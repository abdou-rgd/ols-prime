"""Checkpoints without pickle: named arrays in the safetensors layout, written deterministically.

The file is ``u64 little-endian header length`` + a JSON header + the raw little-endian bytes of
every array, in the layout of the safetensors format
(https://github.com/huggingface/safetensors#format), written by hand so that no new dependency
is needed. The header lists the arrays sorted by name, with ``dtype``, ``shape`` and
``data_offsets``, and a ``__metadata__`` map of strings (the fit's config as JSON). It holds no
date, so the same arrays and metadata give the same bytes. Reading uses ``numpy.frombuffer``: no
pickle, no ``allow_pickle``, no code runs.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_DTYPES = {"F64": np.dtype("<f8"), "F32": np.dtype("<f4"), "I64": np.dtype("<i8"), "I32": np.dtype("<i4"), "U8": np.dtype("u1"), "BOOL": np.dtype("?")}
_NAMES = {v: k for k, v in _DTYPES.items()}
MAX_HEADER = 100_000_000


def to_bytes(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, str] | None = None) -> bytes:
    """The file's bytes for ``arrays`` (sorted by name) and string ``metadata``."""
    header: dict[str, object] = {}
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise TypeError("metadata keys and values must be strings")
        header["__metadata__"] = dict(sorted(metadata.items()))
    blobs, offset = [], 0
    for name in sorted(arrays):
        if name == "__metadata__":
            raise ValueError("'__metadata__' is reserved")
        a = np.asarray(arrays[name])
        dt = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
        if dt not in _NAMES:
            raise TypeError(f"{name}: dtype {a.dtype} is not stored")
        raw = np.ascontiguousarray(a, dtype=dt).tobytes()
        header[name] = {"dtype": _NAMES[dt], "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + b"".join(blobs)


def from_bytes(data: bytes) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """``(arrays, metadata)`` from a file's bytes; every array is a fresh copy."""
    if len(data) < 8:
        raise ValueError("not a checkpoint: too short")
    (size,) = struct.unpack("<Q", data[:8])
    if size > MAX_HEADER or 8 + size > len(data):
        raise ValueError("not a checkpoint: bad header length")
    header = json.loads(data[8 : 8 + size].decode("utf-8"))
    body = memoryview(data)[8 + size :]
    metadata = header.pop("__metadata__", {})
    arrays = {}
    for name, info in header.items():
        dt = _DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        count = int(np.prod(info["shape"], dtype=np.int64))
        if end - start != count * dt.itemsize or end > len(body):
            raise ValueError(f"{name}: offsets do not match its shape")
        arrays[name] = np.frombuffer(body[start:end], dtype=dt).reshape(info["shape"]).copy()
    return arrays, metadata


def save(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, str] | None = None) -> str:
    """Write the checkpoint and return its SHA-256."""
    import hashlib

    data = to_bytes(arrays, metadata)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    return from_bytes(Path(path).read_bytes())
