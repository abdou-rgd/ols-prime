"""Where the guards allow a rule, by file: one source of truth for two guards.

``tests/test_protocol.py`` reads it statically (an AST walk of every source file) and
``tests/tripwire.py`` reads it at run time (the file of the first frame outside the data
libraries that calls a reader). A file that may read data is therefore named here once,
with a reason, and both guards agree.

Keys are paths relative to ``python/``, exact and never patterns. A rule is a name of
``findings`` in ``test_protocol.py``. The tripwire maps its triggers onto these:

- ``reader``: Parquet, IPC, Feather and Delta entry points of polars, pyarrow, pandas;
- ``raw-reader``: CSV, JSON and zip readers (the downloaded archives);
- ``binary-load``: ``numpy.load(allow_pickle=True)``;
- ``import:pickle``: ``pickle.load``, ``pickle.loads`` and ``torch.load`` (which unpickles);
- ``vault-tree``: the names and strings that find the vault tree of the physical seal (the function that
  returns its root, the two variables that name its folder and its identity, the default folder, the
  identity reader; the list is ``VAULT_TREE_NAMES`` and ``VAULT_TREE_STRINGS`` in ``test_protocol.py``).
  Static only: the tripwire judges the paths instead (``tests/tripwire.py``).
"""

from collections.abc import Mapping
from types import MappingProxyType

_RULES: dict[str, dict[str, str]] = {
    "olsprime/protocol.py": {
        "reader": "the one reader of stored klines and universe tables, always followed by seal",
        "seal": "defines and calls the seal",
        "import:pyarrow": "stored_row_count reads the row count from the Parquet metadata, no row",
        "vault-tree": "with both keys, finds the vault tree and decrypts its files in memory with the named identity",
    },
    "olsprime/vault.py": {
        "vault-tree": "the vault tool: plans, moves, encrypts and reports the vault tree; it reads no row",
    },
    "olsprime/paths.py": {
        "vault-tree": "defines the vault root and the variable that overrides it",
    },
    "olsprime/data/binance_vision.py": {
        "raw-reader": "parses the downloaded archives into the stored schema",
        "import:zipfile": "opens the downloaded archives",
    },
    "tests/fakes.py": {
        "raw-reader": "writes fake archives",
        "import:zipfile": "writes fake archives",
    },
    "tests/conftest.py": {
        "seal-patch": "seals the vault for every test: it unsets the key variable and moves the key folder and the data root",
        "disarm": "installs the tripwire, checks it at each teardown and takes it down at the end of the session",
        "vault-tree": "gives every test an empty vault root of its own and no identity",
    },
    "tests/test_manifest.py": {"seal-patch": "opens the vault in a fake tree to show that the manifest counts and logs it"},
    "tests/test_vault.py": {
        "seal-patch": "forges both keys in a fake tree to show that the open path needs both keys and, at stage 2, the identity",
        "vault-tree": "builds fake vault trees and points the vault root and the identity at them",
    },
    "tests/test_binance_vision.py": {"reader": "reads back the Parquet file the build just wrote"},
    "tests/test_universe.py": {"reader": "reads back the universe file the run just wrote"},
    "tests/test_protocol.py": {
        "reader": "shows that the stored file keeps the vault rows",
        "seal": "tests the seal itself",
        "vault-tree": "its cheater snippets name the vault tree",
        "disarm": "reads the allow-list and checks that the guard names and the tripwire's agree",
    },
    "tests/tripwire.py": {
        "dynamic": "imports the module of each patched entry point by name",
        "import:pyarrow": "patches pyarrow's Parquet, dataset, IPC and Feather entry points",
        "import:importlib": "finds and imports the optional libraries (pandas, torch) when installed",
        "disarm": "reads the allow-list it enforces, and unwraps classmethod and staticmethod entry points to wrap them",
        "vault-tree": "computes the vault roots that the readers are judged against",
    },
    "tests/test_tripwire.py": {
        "exec": "executes a string to show that the tripwire sees code built at run time",
        "dynamic": "puts stand-in libraries in sys.modules to test the patching mechanism",
        "import:importlib": "imports a helper module written to a temporary folder",
        "import:runpy": "runs the evasion snippets as files in a temporary folder, outside python/",
        "seal-patch": "forges the vault keys in a fake tree to show that the tripwire refuses the repository's vault",
        "disarm": "tests the tripwire from inside: its integrity check, its patches, its guard class",
    },
}

# Frozen: a test that edits the allow-list would silence both guards for its own file. The tripwire
# compares this object and a snapshot with the originals at the teardown of every test.
ALLOWED: Mapping[str, Mapping[str, str]] = MappingProxyType({key: MappingProxyType(dict(rules)) for key, rules in _RULES.items()})
del _RULES
