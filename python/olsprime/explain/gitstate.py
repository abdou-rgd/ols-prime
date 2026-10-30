"""The dirty flag of ``explain``'s provenance: do the code files differ from git's index?

Read without git (the static guard forbids shelling out): ``.git/index`` (versions 2 and 3) gives the
blob SHA-1 of every tracked file, and each code file of :data:`olsprime.bench.provenance.CODE_GLOBS` is
hashed as git hashes a blob. A file is clean when its bytes, or its bytes with CRLF turned into LF
(``.gitattributes`` stores text with LF; a Windows checkout may write CRLF), give the index's blob.

The flag is set when a code file is modified, untracked or deleted with respect to the index, or in a
merge conflict. Its limit: a change staged but not committed (index different from ``HEAD``) is not
seen, since reading ``HEAD``'s tree would mean reading git's packed objects. The code digest beside it
in every answer does not depend on git at all.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Any

from olsprime.bench.provenance import CODE_GLOBS
from olsprime.paths import REPO_ROOT


def _glob_regex(glob: str) -> re.Pattern[str]:
    """The regular expression of a ``pathlib`` glob over POSIX paths: ``**/`` any directories, ``*`` within one."""
    out, i = "", 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out, i = out + "(?:[^/]+/)*", i + 3
        elif glob[i] == "*":
            out, i = out + "[^/]*", i + 1
        else:
            out, i = out + re.escape(glob[i]), i + 1
    return re.compile(out + r"\Z")


def index_entries(root: Path = REPO_ROOT) -> dict[str, bytes] | None:
    """``{path: blob SHA-1}`` of git's index (stage 0; a conflict maps to ``b""``), or None if it cannot be read."""
    git = root / ".git"
    if not git.is_dir():
        return None
    try:
        data = (git / "index").read_bytes()
    except OSError:
        return None
    if len(data) < 12 or data[:4] != b"DIRC":
        return None
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3):  # version 4 compresses the paths: not read here
        return None
    pos, out = 12, {}
    for _ in range(count):
        sha = data[pos + 40 : pos + 60]
        (flags,) = struct.unpack(">H", data[pos + 60 : pos + 62])
        head = 62 + (2 if version >= 3 and flags & 0x4000 else 0)
        end = data.index(b"\0", pos + head)
        name = data[pos + head : end].decode("utf-8")
        out[name] = sha if (flags >> 12) & 3 == 0 else b""
        pos += (end - pos + 8) // 8 * 8  # the entry is padded with 1 to 8 NULs to a multiple of 8
    return out


def _blob(data: bytes) -> bytes:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).digest()


def code_state(root: Path = REPO_ROOT) -> dict[str, Any]:
    """The dirty flag over the code files, with its counts and its scope."""
    scope = "the code files of CODE_GLOBS against git's index (a change staged but not committed is not seen)"
    entries = index_entries(root)
    if entries is None:
        return {"dirty": None, "scope": scope, "note": "git's index could not be read (no .git folder, or index version 4)"}
    patterns = [_glob_regex(g) for g in CODE_GLOBS]
    files = {p.relative_to(root).as_posix() for g in CODE_GLOBS for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    tracked = {name for name in entries if any(r.match(name) for r in patterns)}
    untracked, deleted = sorted(files - tracked), sorted(tracked - files)
    modified = []
    for name in sorted(files & tracked):
        data = (root / name).read_bytes()
        want = entries[name]
        if not want or (_blob(data) != want and _blob(data.replace(b"\r\n", b"\n")) != want):
            modified.append(name)
    return {"dirty": bool(untracked or deleted or modified), "scope": scope,
            "modified": len(modified), "untracked": len(untracked), "deleted": len(deleted)}  # fmt: skip
