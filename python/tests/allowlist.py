"""Where the guards allow a rule, by file: one source of truth for two guards.

``tests/test_protocol.py`` reads it statically (an AST walk of every source file) and
``tests/tripwire.py`` reads it at run time (the file of the first frame outside the data
libraries that calls a reader). A file that may read data is therefore named here once,
with a reason, and both guards agree.

Keys are paths relative to ``python/``, exact and never patterns. A rule is a name of
``findings`` in ``test_protocol.py``. The tripwire maps its triggers onto three of them:

- ``reader``: Parquet, IPC, Feather and Delta entry points of polars, pyarrow, pandas;
- ``binary-load``: ``numpy.load(allow_pickle=True)``;
- ``import:pickle``: ``pickle.load``, ``pickle.loads`` and ``torch.load`` (which unpickles).
"""

ALLOWED: dict[str, dict[str, str]] = {
    "olsprime/protocol.py": {
        "reader": "the one reader of stored klines and universe tables, always followed by seal",
        "seal": "defines and calls the seal",
        "import:pyarrow": "stored_row_count reads the row count from the Parquet metadata, no row",
    },
    "olsprime/data/binance_vision.py": {
        "raw-reader": "parses the downloaded archives into the stored schema",
        "import:zipfile": "opens the downloaded archives",
    },
    "tests/fakes.py": {
        "raw-reader": "writes fake archives",
        "import:zipfile": "writes fake archives",
    },
    "tests/test_binance_vision.py": {"reader": "reads back the Parquet file the build just wrote"},
    "tests/test_universe.py": {"reader": "reads back the universe file the run just wrote"},
    "tests/test_protocol.py": {
        "reader": "shows that the stored file keeps the vault rows",
        "seal": "tests the seal itself",
    },
    "tests/tripwire.py": {
        "dynamic": "imports the module of each patched entry point by name",
        "import:pyarrow": "patches pyarrow's Parquet, dataset, IPC and Feather entry points",
        "import:importlib": "finds and imports the optional libraries (pandas, torch) when installed",
    },
    "tests/test_tripwire.py": {
        "exec": "executes a string to show that the tripwire sees code built at run time",
        "import:importlib": "imports a helper module written to a temporary folder",
    },
}
