import socket
from collections.abc import Iterator

import pytest

from tests.tripwire import Tripwire, install, install_seal_watch, integrity_problems, restore_protocol_class


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a real connection."""

    def refuse(*args, **kwargs):
        raise OSError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def sealed_vault(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every test starts with the vault sealed, its key files and the data root in temporary folders.

    A test must never read or create ``runs/vault/`` in the repository, and a default
    ``load_klines(symbol, interval)`` must never find the repository's ``data/``: the data root is an
    empty folder of its own (not inside ``tmp_path``, which tests fill). The vault tree
    (``olsprime.vault``) gets an empty folder of its own too, and no identity is named, so a default
    test never finds the real ``ols-prime-vault`` beside the repository nor a private key.
    """
    monkeypatch.delenv("OLSPRIME_OPEN_VAULT", raising=False)
    monkeypatch.delenv("OLSPRIME_VAULT_IDENTITY", raising=False)
    monkeypatch.setenv("OLSPRIME_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))
    monkeypatch.setenv("OLSPRIME_DATA_DIR", str(tmp_path_factory.mktemp("datadir")))
    monkeypatch.setenv("OLSPRIME_VAULT_DIR", str(tmp_path_factory.mktemp("vaultdir")))


WIRE = Tripwire()
PATCHES = pytest.MonkeyPatch()


def pytest_configure(config: pytest.Config) -> None:
    """Put the tripwire up before any test module is imported.

    A reader captured at import time (``from polars import read_parquet``, a dict of readers, a
    ``functools.partial``) then holds the wrapper. The tripwire runs only under pytest: scripts,
    notebooks and the CLIs are outside it (see ``tests/tripwire.py``).
    """
    install(PATCHES, WIRE)
    install_seal_watch(PATCHES, WIRE)


def pytest_unconfigure(config: pytest.Config) -> None:
    restore_protocol_class(WIRE)  # the module class first: the patches are undone last in first out
    PATCHES.undo()


@pytest.fixture(scope="session", autouse=True)
def _tripwire_installed() -> Tripwire:
    return WIRE


@pytest.fixture(autouse=True)
def tripwire(_tripwire_installed: Tripwire) -> Iterator[Tripwire]:
    """Fail a test whose reads of stored data came from a file without an allowance, even if it swallowed the error."""
    wire = _tripwire_installed
    wire.hits.clear()
    yield wire
    hits, wire.hits[:] = list(wire.hits), []
    if hits:
        pytest.fail("\n".join(str(hit) for hit in hits), pytrace=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """After every fixture is gone (the test's own monkeypatches included), check that the guards still stand.

    Every patch still bound, the guard class on ``olsprime.protocol`` holding only the watching wrappers,
    the allow-list unchanged, the tripwire's functions the originals. A breach ends the session: nothing
    after it is trustworthy.
    """
    result = yield
    problems = integrity_problems(WIRE)
    if problems:
        pytest.exit("the tripwire was disarmed:\n" + "\n".join(problems), returncode=3)
    return result
