"""Run-time tripwire: a read of stored data from a file that holds no allowance fails the test.

The static guard of ``test_protocol.py`` reads syntax. It cannot follow ``getattr(pl,
"read_" + "parquet")``, a helper module written at run time, or ``exec`` of a string. This
module watches the calls instead. It wraps the entry points that read Parquet, IPC, Feather,
Delta, CSV and JSON tables, zip archives, pyarrow dataset fragments and pickles in polars,
pyarrow, pandas, numpy, zipfile, pickle and torch. At each call it walks the stack to the
first frame that is not inside those libraries (or ``importlib``, pytest, or this module);
the read is allowed only if the file of that frame holds the matching rule in
:data:`tests.allowlist.ALLOWED`. A frame from ``exec`` of a string has the file
``<string>``, and a helper in a temporary folder is outside ``python/``: neither holds an
allowance.

Every entry point is patched wherever the original function is bound, so an alias made by
``from polars import read_parquet as rp`` before the tripwire went up is patched too. A
path rule backs this up: any wrapped reader that is handed a path under the repository's
``data/`` (symlinks resolved; ``str``, ``Path``, file-like objects with ``.name``, lists of
them) is checked even if its own trigger (``numpy.load`` without ``allow_pickle``) is off.

Round 2 of the guard fixes added:

- The standard library is infrastructure like polars: ``zipfile.Path`` and ``importlib.metadata``
  act for their caller. A zip opened through ``importlib.metadata``/``resources`` or ``zipimport``
  and not under ``data/`` is not judged (numpy does it for ``python312.zip``). ``zipimport.zipimporter``
  is a gated reader itself (its ``get_data`` returns the members of a zip, no ``ZipFile`` involved), by
  the path rule only: the zips of ``sys.path`` import as before.
- The repository's own ``data/`` is read through ``olsprime/protocol.py`` only. A wrapped reader handed
  a path under it trips for every rule, even from a file that holds ``reader`` (such a file may read back
  what it wrote in a temporary folder, not the repository's store).
- A raw reader (``zipfile``, CSV) handed a path under ``data/raw/`` needs the ``raw-reader``
  allowance on **every** frame of the stack, so an allowed module that exports a raw reader
  (``binance_vision.read_klines_zip``) is no way round it for a test that does not hold it.
- Pure-Python classes (``ParquetFile``, ``ParquetDataset``, ``ZipFile``) are patched in place, through
  ``__init__``; cython types become subclasses (their original stays at ``__bases__[0]``, a stated
  limit). No wrapper has ``__wrapped__``. The gates bind their helpers as defaults when they are built.

The vault tree of the physical seal (``python -m olsprime.vault``, outside ``data/``) is covered like ``data/``:
any wrapped reader handed a path under the configured vault root (``$OLSPRIME_VAULT_DIR``, an empty folder
of its own under pytest) needs the ordinary allowance, and one handed a path under the default vault root
(``<parent of the repository>/ols-prime-vault``, the real one) trips for every file but ``olsprime/protocol.py``,
like the repository's ``data/``. Those guards are the second line: the first is that the vault tree is
encrypted (stage 2) or, at stage 1, outside the repository.

The seal is watched too (the ``seal`` and ``vault-open`` rules of ``tests/allowlist.py``):

- ``olsprime.protocol`` gets a module class that refuses to rebind ``seal``, ``vault_open``,
  ``VAULT``, ``load_klines``, ``load_klines_file`` or ``load_universe`` from a file without the
  ``seal`` allowance: ``monkeypatch.setattr``, ``mock.patch``, ``setattr`` and ``protocol.x = y``
  all end in it. It refuses ``__class__`` and ``__dict__`` for every file. Four names hold watching
  wrappers, and only the wrapper may stand under its name (the original cannot be put back).
- ``seal``, ``load_klines``, ``load_klines_file`` and ``load_universe`` are wrapped. Without the two
  real keys, a returned frame with a row at or after ``VAULT.start`` fails the test, however the seal
  was switched off (``vars(protocol)[...] = ...`` included): the keys are judged by the original
  ``vault_open`` and the module's ``seal`` global is not consulted.
- The file loaders refuse, with both keys in place, a path under the repository's ``data/`` unless
  every file of the stack holds ``vault-open`` (none does). Forged keys open nothing of the real vault.
  For any other path (a copy of a vault file) with both keys in place, some file of the stack besides
  ``protocol.py`` must hold ``seal``, ``seal-patch`` or ``vault-open``, the files that may open a fake vault.
- :func:`integrity_problems` is run after every test, once its fixtures are gone: patches still bound,
  guard class on the module, permitted values under the six names, allow-list unchanged, this module's
  functions the originals. A breach ends the session.

What it does not see (limits, to keep in mind):

- It runs only under pytest, in the autouse fixtures of ``conftest.py``. A script run by
  hand, a notebook and the CLIs are outside it, and only the static guard covers them.
- Every in-process guard can be disarmed from inside the process: the ``__closure__`` of a wrapper
  holds the original, ``gc`` and ``ctypes`` reach the rest, and a disarm undone before the test ends
  leaves no trace for the teardown check. The static guard flags those names. Stops accidents and
  careless code, not sabotage; the structural fix is the physical seal of iteration 3: the vault rows
  outside ``data/`` and, at stage 2, encrypted with a key that is not on this machine.
- It sees Python-level entry points. A read through ``pyarrow._parquet``, ``pyarrow.lib`` or
  ``polars._plr`` directly, through ``ctypes``, through a subprocess, or by the ``open``
  built-in followed by a parser written by hand, passes. So does the original behind a wrapped
  cython type (``ParquetFileFormat.__bases__[0]``, and ``Fragment.to_table`` on its fragment).
- A read is judged by the file of the calling frame (every frame, for the raw archives); the path
  matters only for the rules above.
- Whoever is allowed ``reader`` may read; the seal itself is checked by the protocol tests.

A trip raises :class:`TripwireError` (an ``AssertionError``) and is also recorded, so a
test that swallows the exception with ``except Exception`` still fails at teardown.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import os
import sys
import sysconfig
import types
import unittest.mock
import urllib.parse
import urllib.request
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
import pluggy
import _pytest

from olsprime.paths import REPO_ROOT, data_dir, default_vault_dir, vault_dir
from tests import allowlist as allowlist_module
from tests.allowlist import ALLOWED

PYTHON_ROOT = Path(__file__).resolve().parents[1]

# The rules of ``tests/allowlist.py`` that the triggers map onto.
READER, RAW_READER, BINARY_LOAD, PICKLE, SEAL = "reader", "raw-reader", "binary-load", "import:pickle", "seal"
# A file that may read the repository's vault with both keys in place; no file holds it today.
VAULT_OPEN = "vault-open"

# What olsprime.protocol refuses to rebind from a file without the ``seal`` allowance.
SEAL_NAMES = ("seal", "vault_open", "VAULT", "load_klines", "load_klines_file", "load_universe")
# Of those, the ones the tripwire replaces by a watching wrapper; only the wrapper may stand in the module.
WATCHED_NAMES = ("seal", "load_klines", "load_klines_file", "load_universe")
# Besides SEAL_NAMES, these attributes of the module may not be rebound either.
PROTECTED_DUNDERS = ("__class__", "__dict__")


class TripwireError(AssertionError):
    """A stored table was read, or the seal was touched, by a file that holds no allowance."""


@dataclass(frozen=True)
class Hit:
    rule: str
    what: str
    caller: str  # file of the first frame outside the data libraries
    path: str = ""  # the path under data/ the call was handed, if any

    def __str__(self) -> str:
        where = f" on {self.path}" if self.path else ""
        return (
            f"tripwire: {self.what}{where} called from {self.caller}, which holds no {self.rule!r} allowance "
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
    skipped = [_package_dir(m) for m in (polars, pyarrow, numpy, importlib, pluggy, _pytest)]
    # unittest.mock.patch and monkeypatch.setattr act for their caller: the test file is the actor.
    skipped += [stdlib_importlib, _norm(__file__), _norm(sys.modules["unittest.mock"].__file__)]
    for name in ("pandas", "torch", "duckdb"):
        if getattr(sys.modules.get(name), "__file__", None):
            skipped.append(_package_dir(sys.modules[name]))
    return tuple(skipped)


def is_infrastructure(filename: str, skipped: tuple[str, ...]) -> bool:
    if filename.startswith("<frozen importlib"):
        return True
    if filename.startswith("<"):  # <string>, <stdin>: code built at run time is never infrastructure
        return False
    path = _norm(filename)
    if any(path == s or path.startswith(s) for s in skipped):
        return True
    # The standard library acts for its caller too: ``zipfile.Path`` opens a ``ZipFile`` (importlib.metadata
    # does it for every zip on sys.path), and the read belongs to the first frame outside the library.
    return path.startswith(_stdlib_dirs()) and "site-packages" not in path


def calling_file(skipped: tuple[str, ...], depth: int = 1) -> str:
    """File of the first frame, from ``depth`` frames up, that is not inside the data libraries."""
    frame: Any = sys._getframe(depth)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not is_infrastructure(filename, skipped):
            return filename
        frame = frame.f_back
    return "<no frame outside the data libraries>"


@functools.cache
def _stdlib_dirs() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    return tuple({_norm(paths[k]) + os.sep for k in ("stdlib", "platstdlib") if k in paths})


@functools.cache
def _metadata_dirs() -> tuple[str, ...]:
    base = os.path.dirname(importlib.__file__)
    return tuple(_norm(os.path.join(base, sub)) + os.sep for sub in ("metadata", "resources"))


def through_importlib(depth: int = 1) -> bool:
    """Whether ``importlib.metadata``, ``importlib.resources`` or the frozen ``zipimport`` is on the stack.

    Not ``importlib`` as a whole: pytest imports every test module through ``importlib.import_module``,
    and a read made while a module is imported must keep its caller.

    It opens the zip files of ``sys.path`` (``python312.zip``, eggs) for a library that looks up a
    package's metadata. Such a read is not a read of stored data, and it has no caller to blame.
    """
    frame: Any = sys._getframe(depth)
    metadata_dirs = _metadata_dirs()
    while frame is not None:
        name = frame.f_code.co_filename
        if name.startswith("<frozen zipimport") or (not name.startswith("<") and _norm(name).startswith(metadata_dirs)):
            return True
        frame = frame.f_back
    return False


def stack_files(skipped: tuple[str, ...], depth: int = 1) -> list[str]:
    """Files of every frame that is neither a data library nor the standard library, innermost first.

    A call from ``read_klines_zip`` in an allowed module reaches this through a test file: the
    test file is on the stack too, and for the raw archives it must hold the allowance as well.
    """
    files: list[str] = []
    frame: Any = sys._getframe(depth)
    while frame is not None:
        name = frame.f_code.co_filename
        if not is_infrastructure(name, skipped) and name not in files:
            files.append(name)
        frame = frame.f_back
    return files


def allowance_key(filename: str) -> str | None:
    """``filename`` as a key of ALLOWED (relative to python/, forward slashes), or ``None`` outside python/."""
    if filename.startswith("<"):
        return None
    real, root = _real(filename), _real(PYTHON_ROOT) + os.sep
    return real[len(root) :].replace(os.sep, "/") if _norm(real).startswith(os.path.normcase(root)) else None


def is_allowed(filename: str, rule: str, _allowed: Any = ALLOWED) -> bool:
    """Whether the file holds the rule. The allow-list is bound here, when the module is imported: rebinding
    ``tests.tripwire.ALLOWED`` later changes nothing."""
    key = allowance_key(filename)
    return key is not None and rule in _allowed.get(key, {})


# --- Paths under data/ ------------------------------------------------------------------


def data_roots() -> tuple[str, ...]:
    """The folders that hold stored data, normalised.

    The repository's ``data/``, the configured data root, and the vault tree (the configured root and the
    default one beside the repository, ``physical seal`` of iteration 3): a path under the vault tree is a path
    under ``data/`` for every wrapped reader, and ``<vault>/raw`` counts as a ``raw/`` folder.
    """
    return tuple(
        {
            _norm(REPO_ROOT / "data") + os.sep,
            _norm(data_dir()) + os.sep,
            _norm(default_vault_dir()) + os.sep,
            _norm(vault_dir()) + os.sep,
        }
    )


PATH_KEYWORDS = frozenset(
    {"source", "sources", "path", "paths", "file", "files", "fname", "filename", "where", "path_or_buffer", "file_or_filename"}
)


def _collect(value: Any, found: list[str], budget: list[int]) -> None:
    budget[0] -= 1
    if budget[0] < 0:
        return
    if isinstance(value, str | os.PathLike):
        found.append(os.fspath(value))
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            _collect(item, found, budget)
    else:  # a file-like object remembers its name
        name = getattr(value, "name", None)
        if isinstance(name, str | os.PathLike):
            found.append(os.fspath(name))


def spelled_path(text: str) -> str:
    """A path as the file system sees it, from the spellings that ``_norm`` alone would take for a relative path.

    ``file:///C:/x`` (a URI), ``\\\\?\\C:\\x`` (the extended-length prefix) and ``\\\\?\\UNC\\server\\share\\x``.
    """
    if text[:5].lower() == "file:":
        parsed = urllib.parse.urlparse(text)
        path = urllib.request.url2pathname(parsed.path)
        if len(parsed.netloc) == 2 and parsed.netloc[1] == ":":  # file://C:/x, a drive written as the host
            return urllib.request.url2pathname(parsed.netloc + parsed.path)
        return (os.sep * 2 + parsed.netloc + path) if parsed.netloc and parsed.netloc != "localhost" else path
    for prefix, unc in (("\\\\?\\UNC\\", True), ("//?/UNC/", True), ("\\\\?\\", False), ("//?/", False)):
        if text.upper().startswith(prefix.upper()):
            rest = text[len(prefix) :]
            return (os.sep * 2 + rest) if unc else rest
    return text


def _base_paths(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """``base_path`` of every file system among the arguments: a ``SubTreeFileSystem`` reads paths relative to it."""
    bases = []
    for value in [*args[1:], *kwargs.values()]:
        base = getattr(value, "base_path", None) if "FileSystem" in type(value).__name__ else None
        if isinstance(base, str) and base:
            bases.append(base)
    return bases


def repo_data_roots() -> tuple[str, ...]:
    """The real store: the repository's own ``data/`` and the default vault tree beside the repository.

    Not the configured data root or vault root: under pytest those are empty folders of the test's own (a fake
    vault tree), and a path under them is judged by the ordinary rules of :func:`data_roots`.
    """
    return (_norm(REPO_ROOT / "data") + os.sep, _norm(default_vault_dir()) + os.sep)


def raw_roots() -> tuple[str, ...]:
    """The ``raw/`` folders of the data roots: the downloaded archives."""
    return tuple(root + "raw" + os.sep for root in data_roots())


def data_paths(args: tuple[Any, ...], kwargs: dict[str, Any], roots: tuple[str, ...] | None = None) -> list[str]:
    """Paths among the first positional argument and the path-like keywords that lie under ``roots`` (``data/``)."""
    candidates: list[str] = []
    budget = [10_000]
    if args:
        _collect(args[0], candidates, budget)
    for key, value in kwargs.items():
        if key in PATH_KEYWORDS:
            _collect(value, candidates, budget)
    roots = data_roots() if roots is None else roots
    bases = _base_paths(args, kwargs)
    inside = []
    for candidate in candidates:
        try:
            spelled = spelled_path(candidate)
            variants = [spelled]
            if not os.path.isabs(spelled):  # relative to a SubTreeFileSystem's base_path, if one is given
                variants += [os.path.join(base, spelled) for base in bases]
            resolved = [_norm(variant) + os.sep for variant in variants]
        except (ValueError, OSError):
            continue
        if any(r.startswith(root) for r in resolved for root in roots):
            inside.append(candidate)
    return inside


# --- The tripwire ---------------------------------------------------------------------


@dataclass
class Tripwire:
    hits: list[Hit] = field(default_factory=list)
    skipped: tuple[str, ...] = ()
    permitted: dict[str, list[Any]] = field(default_factory=dict)  # the values olsprime.protocol may hold
    bindings: list[tuple[Any, str, Any]] = field(default_factory=list)  # (holder, name, wrapper) that install applied
    protocol: Any = None
    guard_class: Any = None
    guard_methods: tuple[Any, Any] | None = None  # (__setattr__, __delattr__) of the guard class
    allowed_snapshot: dict[str, dict[str, str]] | None = None  # taken by install; ``None``: not yet checked

    def trip(self, rule: str, what: str, paths: list[str] | None = None) -> None:
        """Record and raise: the caller of the wrapped function ``what`` holds no ``rule`` allowance."""
        hit = Hit(rule, what, calling_file(self.skipped), paths[0] if paths else "")
        self.hits.append(hit)
        raise TripwireError(str(hit))

    def trip_at(self, filename: str, rule: str, what: str, paths: list[str] | None = None) -> None:
        """Record and raise, naming ``filename`` as the offender (a frame further out than the innermost)."""
        hit = Hit(rule, what, filename, paths[0] if paths else "")
        self.hits.append(hit)
        raise TripwireError(str(hit))

    def check(self, rule: str, what: str, paths: list[str] | None = None) -> None:
        """Raise if the caller of the wrapped function ``what`` holds no ``rule`` allowance."""
        if not is_allowed(calling_file(self.skipped), rule):
            self.trip(rule, what, paths)

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
    """An entry point: ``module.path`` and the attribute (``Class.method`` allowed) to wrap.

    For a class, ``methods`` lists methods to wrap as well (a subclass overrides them: the
    cython types of pyarrow cannot be patched in place) and ``gate_init`` says whether
    constructing the class is itself a read.
    """

    module: str
    attribute: str
    rule: str
    only_if: Callable[..., bool] | None = None  # called with the arguments; the check runs when it is true
    methods: tuple[str, ...] = ()
    gate_init: bool = True

    @property
    def label(self) -> str:
        return f"{self.module}.{self.attribute}"


def _allow_pickle(*args: Any, **kwargs: Any) -> bool:
    """For numpy.load(file, mmap_mode=None, allow_pickle=False, ...): whether pickles are allowed."""
    if "allow_pickle" in kwargs:
        return bool(kwargs["allow_pickle"])
    return len(args) > 2 and bool(args[2])


def _zip_reads(*args: Any, **kwargs: Any) -> bool:
    """For ZipFile(file, mode="r", ...): whether the archive is opened for reading ('r' or 'a'), not created."""
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    return mode in ("r", "a")


def polars_targets() -> list[Target]:
    """Every polars reader of tables, found by name: Parquet and friends (``reader``), CSV and JSON (``raw-reader``)."""
    tables = ("parquet", "ipc", "delta", "iceberg", "lance", "pyarrow_dataset", "arrow_c_stream", "external_reader", "database")
    raw = ("csv", "json", "avro", "excel", "ods", "lines")
    readers = (n for n in sorted(dir(polars)) if n.startswith(("read_", "scan_")))
    targets = []
    for name in readers:
        if any(m in name for m in tables):
            targets.append(Target("polars", name, READER))
        elif any(m in name for m in raw):
            targets.append(Target("polars", name, RAW_READER))
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
    # The cython types cannot be patched in place: a subclass overrides the entry that makes a fragment.
    Target("pyarrow.dataset", "FileFormat", READER, methods=("make_fragment",), gate_init=False),
    Target("pyarrow.dataset", "ParquetFileFormat", READER, methods=("make_fragment",), gate_init=False),
    Target("pyarrow.dataset", "FileSystemDataset", READER, methods=("from_paths",), gate_init=False),
    Target("pyarrow.dataset", "FileSystemDatasetFactory", READER),
    Target("pyarrow.dataset", "ParquetDatasetFactory", READER),
    Target("pyarrow.ipc", "open_file", READER),
    Target("pyarrow.ipc", "open_stream", READER),
    Target("pyarrow.feather", "read_table", READER),
    Target("pyarrow.feather", "read_feather", READER),
    Target("zipfile", "ZipFile", RAW_READER, only_if=_zip_reads),
    # zipimport reads the members of a zip (``get_data``) without ZipFile. The import system builds one for each
    # zip on sys.path (python312.zip) from the same class: the trigger is off, so only a path under data/ is judged.
    Target("zipimport", "zipimporter", RAW_READER, only_if=lambda *args, **kwargs: False),
    Target("numpy", "load", BINARY_LOAD, only_if=_allow_pickle),
    Target("pickle", "load", PICKLE),
    Target("pickle", "loads", PICKLE),
    Target("pickle", "_load", PICKLE),  # the pure-Python loaders, reachable by name
    Target("pickle", "_loads", PICKLE),
]
OPTIONAL_TARGETS: dict[str, list[Target]] = {
    "pandas": [Target("pandas", "read_parquet", READER), Target("pandas", "read_feather", READER)],
    "torch": [Target("torch", "load", PICKLE)],  # torch.load unpickles
    # duckdb reads files by SQL; none of this is installed here, so it is untested against the real library
    "duckdb": [
        Target("duckdb", "read_parquet", READER),
        Target("duckdb", "read_csv", RAW_READER),
        Target("duckdb", "sql", READER),
        Target("duckdb", "query", READER),
        Target("duckdb", "execute", READER),
    ],
}


def _resolve(module_name: str, attribute: str) -> tuple[Any, str, Any, Any]:
    """``(owner, name, raw attribute, resolved value)``; the raw one keeps classmethods as they are."""
    owner: Any = importlib.import_module(module_name)
    *path, name = attribute.split(".")
    for part in path:
        owner = getattr(owner, part)
    raw = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    return owner, name, raw, getattr(owner, name)


def _named(wrapper: Any, original: Any) -> Any:
    """Give ``wrapper`` the name and the doc of ``original``, but not ``__wrapped__``: that would hand the original back."""
    for attribute in ("__module__", "__name__", "__qualname__", "__doc__"):
        try:
            setattr(wrapper, attribute, getattr(original, attribute))
        except (AttributeError, TypeError):
            pass
    return wrapper


def _under_raw(paths: list[str]) -> bool:
    roots = raw_roots()
    for path in paths:
        try:
            if any((_norm(path) + os.sep).startswith(root) for root in roots):
                return True
        except (ValueError, OSError):
            continue
    return False


def is_python_class(cls: type) -> bool:
    """Whether the class is written in Python (its ``__init__`` can be patched in place), not a cython type."""
    return isinstance(cls.__dict__.get("__init__"), types.FunctionType)


def guard(target: Target, original: Any, tripwire: Tripwire) -> Any:
    """``original`` wrapped to check the caller first.

    A pure-Python class is patched in place (its ``__init__``), so no copy of the original class stays
    reachable through ``__bases__`` or a subclass such as ``zipfile.Path``. A cython class cannot be:
    it becomes a subclass, and its original stays one attribute away (``__bases__[0]``, a stated limit).
    """

    # The helpers are bound as defaults, when the target is wrapped: patching ``tests.tripwire.is_allowed``
    # (or any of these names) later changes nothing for the gates already up.
    def gate(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        label: str = target.label,
        is_allowed: Any = is_allowed,
        calling_file: Any = calling_file,
        stack_files: Any = stack_files,
        data_paths: Any = data_paths,
        _under_raw: Any = _under_raw,
        through_importlib: Any = through_importlib,
        repo_data_roots: Any = repo_data_roots,
        allowance_key: Any = allowance_key,
    ) -> None:
        paths = data_paths(args, kwargs)
        # The repository's own data/ (the real store and the vault) is read through olsprime/protocol.py and
        # nothing else, whatever the rule and whoever holds it: a file allowed ``reader`` to read back what it
        # wrote in a temporary folder may not read the repository's files. Under pytest the configured data
        # root is an empty folder of its own, so no test has a reason to name the repository's data/.
        if data_paths(args, kwargs, repo_data_roots()) and allowance_key(calling_file(tripwire.skipped)) != "olsprime/protocol.py":
            tripwire.trip(target.rule, label, paths)
            return
        if target.rule == RAW_READER and not paths and through_importlib():
            return  # importlib.metadata opening a zip of sys.path: no stored data, and nobody to blame
        if not (target.only_if is None or target.only_if(*args, **kwargs) or paths):
            return
        skipped = tripwire.skipped
        if target.rule == RAW_READER and _under_raw(paths):
            # The downloaded archives: every frame on the stack, the test file included, must hold the allowance.
            for filename in stack_files(skipped):
                if not is_allowed(filename, target.rule):
                    tripwire.trip_at(filename, target.rule, label, paths)
        elif not is_allowed(calling_file(skipped), target.rule):
            tripwire.trip(target.rule, label, paths)

    if isinstance(original, type):
        if target.gate_init and is_python_class(original):
            real_init = original.__init__

            def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
                gate(args, kwargs)
                real_init(self, *args, **kwargs)

            _named(__init__, real_init)
            return _InPlace(original, __init__)
        namespace: dict[str, Any] = {"__module__": original.__module__, "__doc__": original.__doc__}
        if target.gate_init:

            def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
                gate(args, kwargs)
                original.__init__(self, *args, **kwargs)

            namespace["__init__"] = __init__
        subclass = type(original.__name__, (original,), namespace)
        for method in target.methods:
            if isinstance(inspect.getattr_static(original, method), classmethod):

                def class_method(cls: Any, *args: Any, _name: str = method, **kwargs: Any) -> Any:
                    gate(args, kwargs, f"{target.label}.{_name}")
                    return getattr(super(subclass, cls), _name)(*args, **kwargs)

                setattr(subclass, method, classmethod(class_method))
            else:

                def instance_method(self: Any, *args: Any, _name: str = method, **kwargs: Any) -> Any:
                    gate(args, kwargs, f"{target.label}.{_name}")
                    return getattr(super(subclass, self), _name)(*args, **kwargs)

                setattr(subclass, method, instance_method)
        return subclass

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        gate(args, kwargs)
        return original(*args, **kwargs)

    return _named(wrapper, original)


@dataclass(frozen=True)
class _InPlace:
    """A guarded ``__init__`` to put on ``cls`` itself."""

    cls: type
    init: Any


def _aliases(original: Any) -> list[tuple[Any, str]]:
    """Every loaded module attribute that is ``original``: ``from x import f`` makes more than one."""
    found = []
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        found += [(module, name) for name, value in list(namespace.items()) if value is original]
    return found


def bind(monkeypatch: Any, tripwire: Tripwire, holder: Any, attribute: str, value: Any) -> None:
    """``monkeypatch.setattr`` that remembers the binding, so that the teardown can check it still stands."""
    monkeypatch.setattr(holder, attribute, value)
    tripwire.bindings.append((holder, attribute, value))


def install(monkeypatch: Any, tripwire: Tripwire) -> list[str]:
    """Wrap every target (and every alias of it) with ``monkeypatch``; return the labels wrapped."""
    targets_to_wrap = list(TARGETS)
    for name, targets in OPTIONAL_TARGETS.items():
        spec = importlib.util.find_spec(name)
        if name in sys.modules or (spec is not None and spec.origin):  # a real package, not a namespace folder
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
        if isinstance(replacement, _InPlace):
            bind(monkeypatch, tripwire, replacement.cls, "__init__", replacement.init)
            wrapped.append(target.label)
            continue
        if isinstance(raw, classmethod):
            replacement = classmethod(guard(target, raw.__func__, tripwire))
        elif isinstance(raw, staticmethod):
            replacement = staticmethod(guard(target, raw.__func__, tripwire))
        for holder, attribute in _aliases(original) if not isinstance(owner, type) else [(owner, name)]:
            bind(monkeypatch, tripwire, holder, attribute, replacement)
        wrapped.append(target.label)
    tripwire.allowed_snapshot = {key: dict(rules) for key, rules in ALLOWED.items()}
    return wrapped


# --- The seal ---------------------------------------------------------------------------


def _has_vault_rows(frame: Any, vault_start: Any) -> bool:
    """Whether a sealed frame still holds a row at or after ``vault_start`` in a time column."""
    if not isinstance(frame, polars.DataFrame):
        return False
    for column in ("known_at", "open_time", "rebalance_time"):
        if column in frame.columns and isinstance(frame.schema[column], polars.Datetime):
            if bool((frame[column] >= vault_start).any()):
                return True
    return False


def install_seal_watch(monkeypatch: Any, tripwire: Tripwire) -> list[str]:
    """Guard ``olsprime.protocol``: refuse to rebind the seal, and fail a read that lets the vault out.

    Four functions are replaced by watching wrappers (``seal``, ``load_klines``,
    ``load_klines_file``, ``load_universe``), each of which checks what it returns:

    - without both keys of the vault, no returned frame may hold a row at or after ``VAULT.start``.
      The keys are judged by the original ``vault_open``, not by whatever the module holds, and the
      check does not depend on the ``seal`` global the loaders call;
    - the two file loaders refuse a path under the repository's ``data/`` while both keys are in
      place, unless every file on the stack holds the ``vault-open`` allowance. Forged keys
      (``open_vault`` in a test) then do not open the real vault.

    Only the wrappers may stand in the module under those names: putting the original back is refused.
    """
    protocol = importlib.import_module("olsprime.protocol")
    tripwire.skipped = tripwire.skipped or infrastructure()
    originals = {name: getattr(protocol, name) for name in SEAL_NAMES}
    real_vault_open, vault_start = originals["vault_open"], originals["VAULT"].start

    def keys_in_place() -> bool:
        try:
            return bool(real_vault_open())
        except Exception:  # a bad OPEN file: the loader raises its own error
            return False

    def watch(name: str) -> Any:
        original = originals[name]
        loads_a_file = name in ("load_klines_file", "load_universe")

        def watched(*args: Any, **kwargs: Any) -> Any:
            if loads_a_file and keys_in_place():
                paths = data_paths(args[:1], {k: v for k, v in kwargs.items() if k == "path"}, repo_data_roots())
                if paths:
                    for filename in stack_files(tripwire.skipped):
                        if not is_allowed(filename, VAULT_OPEN) and allowance_key(filename) != "olsprime/protocol.py":
                            tripwire.trip_at(filename, SEAL, f"protocol.{name} of the repository's vault with both keys", paths)
                else:
                    # Both keys, a file outside the repository's data/ (a copy of a vault file): the keys must have
                    # been put there by a file that may touch the seal; forging them is a ``seal-patch`` for the
                    # static guard, so a file that holds none of these rules is not who opened the vault.
                    outside = [f for f in stack_files(tripwire.skipped) if allowance_key(f) != "olsprime/protocol.py"]
                    if outside and not any(is_allowed(f, rule) for f in outside for rule in (SEAL, "seal-patch", VAULT_OPEN)):
                        tripwire.trip_at(outside[0], SEAL, f"protocol.{name} of a file with both keys forged", [str(args[0])] if args else [])
            result = original(*args, **kwargs)
            if _has_vault_rows(result, vault_start) and not keys_in_place():
                tripwire.trip(SEAL, f"protocol.{name} returned a row at or after VAULT.start without both vault keys")
            return result

        return _named(watched, original)

    wrappers = {name: watch(name) for name in WATCHED_NAMES}
    # Only the wrapper may stand under a watched name; the other two names keep their original.
    tripwire.permitted = {name: [wrappers[name]] if name in wrappers else [originals[name]] for name in SEAL_NAMES}
    tripwire.protocol = protocol
    tripwire.allowed_snapshot = {key: dict(rules) for key, rules in ALLOWED.items()}

    class GuardedProtocol(types.ModuleType):
        def __setattr__(self, name: str, value: Any) -> None:
            if name in PROTECTED_DUNDERS:
                tripwire.check(SEAL, f"setattr(olsprime.protocol, {name!r})")
            elif name in SEAL_NAMES and not any(value is ok for ok in tripwire.permitted[name]):
                tripwire.check(SEAL, f"setattr(olsprime.protocol, {name!r})")
            super().__setattr__(name, value)

        def __delattr__(self, name: str) -> None:
            if name in SEAL_NAMES or name in PROTECTED_DUNDERS:
                tripwire.check(SEAL, f"delattr(olsprime.protocol, {name!r})")
            super().__delattr__(name)

    tripwire.guard_class = GuardedProtocol
    tripwire.guard_methods = (GuardedProtocol.__setattr__, GuardedProtocol.__delattr__)
    for name, wrapper in wrappers.items():
        for holder, attribute in _aliases(originals[name]):
            bind(monkeypatch, tripwire, holder, attribute, wrapper)
    monkeypatch.setattr(protocol, "__class__", GuardedProtocol)
    return ["olsprime.protocol.seal, load_klines, load_klines_file, load_universe (results watched)", "olsprime.protocol (rebinding refused)"]


def restore_protocol_class(tripwire: Tripwire) -> None:
    """Take the guard class off the module (the unconfigure step, before the patches are undone, last in first out)."""
    if tripwire.protocol is not None:
        types.ModuleType.__setattr__(tripwire.protocol, "__class__", types.ModuleType)


def integrity_problems(tripwire: Tripwire) -> list[str]:
    """What has been undone since ``install``: empty while the guards stand, one line per breach otherwise.

    Checked at the teardown of every test: every patch still bound, the protocol module still wearing
    its guard class and holding only the watching wrappers, the allow-list unchanged, and the
    tripwire's own functions still the ones defined here.
    """
    problems = []
    for holder, attribute, value in tripwire.bindings:
        if getattr(holder, "__dict__", {}).get(attribute) is not value:
            problems.append(f"{getattr(holder, '__name__', holder)!s}.{attribute} is no longer the tripwire's wrapper")
    protocol = tripwire.protocol
    if protocol is not None:
        if type(protocol) is not tripwire.guard_class:
            problems.append("olsprime.protocol no longer wears the guard class")
        for name in SEAL_NAMES:
            if not any(protocol.__dict__.get(name) is ok for ok in tripwire.permitted.get(name, [])):
                problems.append(f"olsprime.protocol.{name} is not the value the tripwire permits")
        if tripwire.guard_methods and (
            tripwire.guard_class.__dict__.get("__setattr__"), tripwire.guard_class.__dict__.get("__delattr__")
        ) != tripwire.guard_methods:
            problems.append("the guard class was edited")
    snapshot = {k: dict(v) for k, v in ALLOWED.items()}
    if allowlist_module.ALLOWED is not ALLOWED or (tripwire.allowed_snapshot is not None and snapshot != tripwire.allowed_snapshot):
        problems.append("the allow-list changed")
    for name, original in _ORIGINAL_FUNCTIONS.items():
        if getattr(_SELF, name, None) is not original:
            problems.append(f"tests.tripwire.{name} was replaced")
    for name, original in _ORIGINAL_METHODS.items():
        if Tripwire.__dict__.get(name) is not original:
            problems.append(f"Tripwire.{name} was replaced")
    return problems


# What the tripwire's own functions were when this module was imported.
_SELF = sys.modules[__name__]
_ORIGINAL_FUNCTIONS = {
    name: getattr(_SELF, name)
    for name in (
        "is_allowed", "calling_file", "stack_files", "allowance_key", "data_roots", "repo_data_roots", "raw_roots",
        "data_paths", "is_infrastructure", "guard", "bind", "install", "install_seal_watch", "integrity_problems",
    )  # fmt: skip
    if name in globals()
}
_ORIGINAL_METHODS = {name: Tripwire.__dict__[name] for name in ("trip", "trip_at", "check")}
