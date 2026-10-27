"""Run-time tripwire: a read of stored data from a file that holds no allowance fails the test.

The static guard of ``test_protocol.py`` reads syntax. It cannot follow ``getattr(pl,
"read_" + "parquet")``, a helper module written at run time, or ``exec`` of a string. This
module watches the calls instead. It wraps the entry points that read Parquet (and IPC,
Feather, Delta, pickles) in polars, pyarrow, pandas, numpy, pickle and torch. At each
call it walks the stack to the first frame that is not inside those libraries (or
``importlib``, or this module); the read is allowed only if the file of that frame holds
the matching rule in :data:`tests.allowlist.ALLOWED`. A frame from ``exec`` of a string
has the file ``<string>``, and a helper in a temporary folder is outside ``python/``:
neither holds an allowance.

Every entry point is patched wherever the original function is bound, so an alias made by
``from polars import read_parquet as rp`` before the tripwire went up is patched too.

What it does not see (limits, to keep in mind):

- It runs only under pytest, in the autouse fixtures of ``conftest.py``. A script run by
  hand, a notebook and the CLIs are outside it, and only the static guard covers them.
- It sees Python-level entry points. A read through ``pyarrow._parquet`` or
  ``pyarrow.lib`` directly, through ``ctypes``, through a subprocess or by the ``open``
  built-in followed by ``read_bytes`` and a parser written by hand, passes.
- A read is judged by the file of the calling frame, not by the path it reads: an allowed
  file that reads a vault file by a wrong path still goes through ``load_klines`` or fails
  the static guard, not this one.
- Whoever is allowed ``reader`` may read; the seal itself is checked by the protocol tests.

A trip raises :class:`TripwireError` (an ``AssertionError``) and is also recorded, so a
test that swallows the exception with ``except Exception`` still fails at teardown.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
import polars
import pyarrow
import pyarrow.dataset
import pyarrow.feather
import pyarrow.ipc
import pyarrow.parquet

from tests.allowlist import ALLOWED

PYTHON_ROOT = Path(__file__).resolve().parents[1]

# The rules of ``tests/allowlist.py`` that the triggers map onto.
READER, BINARY_LOAD, PICKLE = "reader", "binary-load", "import:pickle"


class TripwireError(AssertionError):
    """A stored table was read by a file that holds no allowance."""


@dataclass(frozen=True)
class Hit:
    rule: str
    what: str
    caller: str  # file of the first frame outside the data libraries

    def __str__(self) -> str:
        return (
            f"tripwire: {self.what} called from {self.caller}, which holds no {self.rule!r} allowance "
            f"in tests/allowlist.py"
        )


# --- Who is calling -------------------------------------------------------------------


def _real(path: str | Path) -> str:
    return os.path.realpath(path)


def _norm(path: str | Path) -> str:
    return os.path.normcase(_real(path))


def _package_dir(module: Any) -> str:
    return _norm(os.path.dirname(module.__file__)) + os.sep


def infrastructure() -> tuple[str, ...]:
    """Directories (and files) whose frames are skipped when looking for the caller."""
    stdlib_importlib = _norm(os.path.dirname(importlib.__file__)) + os.sep
    skipped = [_package_dir(m) for m in (polars, pyarrow, numpy, importlib)]
    skipped += [stdlib_importlib, _norm(__file__)]
    for name in ("pandas", "torch"):
        if name in sys.modules:
            skipped.append(_package_dir(sys.modules[name]))
    return tuple(skipped)


def is_infrastructure(filename: str, skipped: tuple[str, ...]) -> bool:
    if filename.startswith("<frozen importlib"):
        return True
    if filename.startswith("<"):  # <string>, <stdin>: code built at run time is never infrastructure
        return False
    path = _norm(filename)
    return any(path == s or path.startswith(s) for s in skipped)


def calling_file(skipped: tuple[str, ...], depth: int = 1) -> str:
    """File of the first frame, from ``depth`` frames up, that is not inside the data libraries."""
    frame: Any = sys._getframe(depth)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not is_infrastructure(filename, skipped):
            return filename
        frame = frame.f_back
    return "<no frame outside the data libraries>"


def allowance_key(filename: str) -> str | None:
    """``filename`` as a key of ALLOWED (relative to python/, forward slashes), or ``None`` outside python/."""
    if filename.startswith("<"):
        return None
    real, root = _real(filename), _real(PYTHON_ROOT) + os.sep
    return real[len(root) :].replace(os.sep, "/") if _norm(real).startswith(os.path.normcase(root)) else None


def is_allowed(filename: str, rule: str) -> bool:
    key = allowance_key(filename)
    return key is not None and rule in ALLOWED.get(key, {})


# --- The tripwire ---------------------------------------------------------------------


@dataclass
class Tripwire:
    hits: list[Hit] = field(default_factory=list)
    skipped: tuple[str, ...] = ()

    def check(self, rule: str, what: str) -> None:
        """Raise if the caller of the wrapped function ``what`` holds no ``rule`` allowance."""
        filename = calling_file(self.skipped)
        if is_allowed(filename, rule):
            return
        hit = Hit(rule, what, filename)
        self.hits.append(hit)
        raise TripwireError(str(hit))

    @contextmanager
    def expect_trip(self, rule: str | None = None) -> Iterator[list[Hit]]:
        """For the tripwire's own tests: the block must trip; the hits are consumed, not failures."""
        before = len(self.hits)
        new: list[Hit] = []
        try:
            yield new
        except TripwireError:
            pass  # the expected trip; it is in self.hits
        new.extend(self.hits[before:])
        del self.hits[before:]
        assert new, "the tripwire did not trip"
        if rule is not None:
            assert {h.rule for h in new} == {rule}, new

    def unconsumed(self) -> list[Hit]:
        return list(self.hits)


@dataclass(frozen=True)
class Target:
    """An entry point: ``module.path`` and the attribute (``Class.method`` allowed) to wrap."""

    module: str
    attribute: str
    rule: str
    only_if: Callable[..., bool] | None = None  # called with the arguments; the check runs when it is true

    @property
    def label(self) -> str:
        return f"{self.module}.{self.attribute}"


def _allow_pickle(*args: Any, **kwargs: Any) -> bool:
    """For numpy.load(file, mmap_mode=None, allow_pickle=False, ...): whether pickles are allowed."""
    if "allow_pickle" in kwargs:
        return bool(kwargs["allow_pickle"])
    return len(args) > 2 and bool(args[2])


def polars_targets() -> list[Target]:
    """Every polars reader of Parquet, IPC, Delta, Iceberg, Lance or a pyarrow dataset, found by name."""
    markers = ("parquet", "ipc", "delta", "iceberg", "lance", "pyarrow_dataset", "arrow_c_stream", "external_reader")
    names = sorted(n for n in dir(polars) if n.startswith(("read_", "scan_")) and any(m in n for m in markers))
    targets = [Target("polars", name, READER) for name in names]
    targets += [Target("polars", f"{cls}.deserialize", READER) for cls in ("DataFrame", "LazyFrame")]
    return targets


TARGETS: list[Target] = [
    *polars_targets(),
    Target("pyarrow.parquet", "read_table", READER),
    Target("pyarrow.parquet", "read_pandas", READER),
    Target("pyarrow.parquet", "read_metadata", READER),
    Target("pyarrow.parquet", "read_schema", READER),
    Target("pyarrow.parquet", "ParquetFile", READER),
    Target("pyarrow.parquet", "ParquetDataset", READER),
    Target("pyarrow.dataset", "dataset", READER),
    Target("pyarrow.dataset", "parquet_dataset", READER),
    Target("pyarrow.ipc", "open_file", READER),
    Target("pyarrow.ipc", "open_stream", READER),
    Target("pyarrow.feather", "read_table", READER),
    Target("pyarrow.feather", "read_feather", READER),
    Target("numpy", "load", BINARY_LOAD, only_if=_allow_pickle),
    Target("pickle", "load", PICKLE),
    Target("pickle", "loads", PICKLE),
    Target("pickle", "_load", PICKLE),  # the pure-Python loaders, reachable by name
    Target("pickle", "_loads", PICKLE),
]
OPTIONAL_TARGETS: dict[str, list[Target]] = {
    "pandas": [Target("pandas", "read_parquet", READER), Target("pandas", "read_feather", READER)],
    "torch": [Target("torch", "load", PICKLE)],  # torch.load unpickles
}


def _resolve(module_name: str, attribute: str) -> tuple[Any, str, Any, Any]:
    """``(owner, name, raw attribute, resolved value)``; the raw one keeps classmethods as they are."""
    owner: Any = importlib.import_module(module_name)
    *path, name = attribute.split(".")
    for part in path:
        owner = getattr(owner, part)
    raw = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    return owner, name, raw, getattr(owner, name)


def guard(target: Target, original: Any, tripwire: Tripwire) -> Any:
    """``original`` wrapped to check the caller first; classes become subclasses so isinstance holds."""

    def gate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if target.only_if is None or target.only_if(*args, **kwargs):
            tripwire.check(target.rule, target.label)

    if isinstance(original, type):

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            gate(args, kwargs)
            original.__init__(self, *args, **kwargs)

        return type(original.__name__, (original,), {"__init__": __init__, "__module__": original.__module__,
                                                       "__doc__": original.__doc__})  # fmt: skip

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        gate(args, kwargs)
        return original(*args, **kwargs)

    return wrapper


def _aliases(original: Any) -> list[tuple[Any, str]]:
    """Every loaded module attribute that is ``original``: ``from x import f`` makes more than one."""
    found = []
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        found += [(module, name) for name, value in list(namespace.items()) if value is original]
    return found


def install(monkeypatch: Any, tripwire: Tripwire) -> list[str]:
    """Wrap every target (and every alias of it) with ``monkeypatch``; return the labels wrapped."""
    targets_to_wrap = list(TARGETS)
    for name, targets in OPTIONAL_TARGETS.items():
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            importlib.import_module(name)
            targets_to_wrap += targets
    tripwire.skipped = infrastructure()
    wrapped = []
    for target in targets_to_wrap:
        try:
            owner, name, raw, original = _resolve(target.module, target.attribute)
        except (ImportError, AttributeError, KeyError):
            continue  # this version of the library does not have it
        replacement = guard(target, original, tripwire)
        if isinstance(raw, classmethod):
            replacement = classmethod(guard(target, raw.__func__, tripwire))
        elif isinstance(raw, staticmethod):
            replacement = staticmethod(guard(target, raw.__func__, tripwire))
        for holder, attribute in _aliases(original) if not isinstance(owner, type) else [(owner, name)]:
            monkeypatch.setattr(holder, attribute, replacement)
        wrapped.append(target.label)
    return wrapped

