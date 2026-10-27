import socket
from collections.abc import Iterator

import pytest

from tests.tripwire import Tripwire, install


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a real connection."""

    def refuse(*args, **kwargs):
        raise OSError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def sealed_vault(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every test starts with the vault sealed and its key files in a temporary folder.

    A test must never read or create ``runs/vault/`` in the repository.
    """
    monkeypatch.delenv("OLSPRIME_OPEN_VAULT", raising=False)
    monkeypatch.setenv("OLSPRIME_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))


@pytest.fixture(scope="session", autouse=True)
def _tripwire_installed() -> Iterator[Tripwire]:
    """Wrap every reader of stored data for the whole session (see ``tests/tripwire.py``).

    The tripwire runs only under pytest: scripts, notebooks and the CLIs are outside it.
    """
    wire = Tripwire()
    with pytest.MonkeyPatch.context() as patches:
        install(patches, wire)
        yield wire


@pytest.fixture(autouse=True)
def tripwire(_tripwire_installed: Tripwire) -> Iterator[Tripwire]:
    """Fail a test whose reads of stored data came from a file without an allowance, even if it swallowed the error."""
    wire = _tripwire_installed
    wire.hits.clear()
    yield wire
    hits, wire.hits[:] = list(wire.hits), []
    if hits:
        pytest.fail("\n".join(str(hit) for hit in hits), pytrace=False)
