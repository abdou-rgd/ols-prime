import socket

import pytest


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
