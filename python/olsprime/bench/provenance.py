"""Provenance of a bench run: the commit, a digest of the source and the digest of the data manifest.

Shared by ``report1`` and ``invariance`` so that every official run is tied to its code and data the same way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from olsprime.paths import REPO_ROOT

PROVENANCE_KEYS = ("git_commit", "code", "manifest_digest")


def git_commit(root: Path = REPO_ROOT) -> str | None:
    """HEAD's commit, read from ``.git`` (no subprocess: the static guard forbids shelling out)."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="ascii").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = git / ref
        if loose.is_file():
            return loose.read_text(encoding="ascii").strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        return None
    return None


CODE_GLOBS = ("python/olsprime/**/*.py", "crates/*/src/**/*.rs", "crates/*/Cargo.toml", "Cargo.lock", "python/uv.lock", "python/pyproject.toml")


def code_digest(root: Path = REPO_ROOT) -> dict[str, Any]:
    """SHA-256 of the source that produced the numbers: the Python package, the Rust crates, the lock files.

    The commit alone does not say whether the worktree was clean; this digest does not depend on git.
    """
    files = sorted({p for g in CODE_GLOBS for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts})
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return {"sha256": h.hexdigest(), "files": len(files), "globs": list(CODE_GLOBS)}


def current_manifest() -> dict[str, Any]:
    """The data manifest as a run records it: ``build_manifest(data_dir())`` without its wall clock."""
    from olsprime.manifest import build_manifest
    from olsprime.paths import data_dir

    return strip_wall_clock(build_manifest(data_dir()))


def strip_wall_clock(manifest: dict[str, Any]) -> dict[str, Any]:
    """Drop ``generated_at``: the wall clock belongs in ``run.json``, not in compared outputs."""
    return {k: v for k, v in manifest.items() if k != "generated_at"}


def record(manifest: dict[str, Any], root: Path = REPO_ROOT) -> dict[str, Any]:
    """The provenance block of a run: exactly the keys of ``PROVENANCE_KEYS``."""
    return {"git_commit": git_commit(root), "code": code_digest(root), "manifest_digest": manifest.get("digest")}


def render_md(prov: dict[str, Any]) -> list[str]:
    """The "Provenance" lines of a run's Markdown report."""
    return [
        "## Provenance",
        "",
        f"- Git commit: `{prov['git_commit']}`.",
        f"- Code digest (SHA-256 of {prov['code']['files']} source and lock files, independent of git): `{prov['code']['sha256']}`.",
        f"- Data manifest digest: `{prov['manifest_digest']}`.",
    ]
