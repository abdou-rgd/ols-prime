"""What every subcommand of ``explain`` shares: the run folder, the periods, the provenance, the contacts, the output.

- **The run folder.** A Report n°2 run, opened only through :func:`olsprime.bench.power.load_series`, the
  package's one reader of ``series.npz``: the file's SHA-256 must be the one given, and the one in the
  run's ``report2.json``. The run's configuration is rebuilt from its ``config.json`` and must write the
  same JSON back.
- **The periods** (``docs/PROTOCOL.md``, "``olsprime.explain``"). The default is 2017-11 -> 2018-12, the
  only development hours that are not test hours. Any other period, or an ``as_of`` outside it, must be
  named with ``--test-months``. Nothing at or after 2025-01-01 is in reach, flag or not.
- **The contacts.** Every call that opens a run folder appends one JSON line to
  ``<runs>/explain/CONTACTS.log`` first: UTC time, subcommand, period touched, arm, commit, dirty flag and
  the series digest given. Before it, the arguments are checked and ``series.npz`` is hashed (bytes only)
  against the digest given and its run's record, so a call refused for those logs nothing. It is the
  only output with a wall clock. ``--jsonl`` is resolved, refused under ``<runs>/explain`` and
  ``<runs>/vault``, and created, never overwritten.
- **The output.** JSON with sorted keys, no wall clock, NaN written as null: two identical calls give
  identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from olsprime.bench import power
from olsprime.bench import provenance as prov_mod
from olsprime.bench.report2 import SERIES_FILE, SERIES_KEYS, Report2Config, sha256_file
from olsprime.data.binance_vision import Month
from olsprime.explain.gitstate import code_state
from olsprime.models.matrix import MatrixConfig, iso, month_start_ms, next_month_ms
from olsprime.paths import runs_dir
from olsprime.protocol import DEVELOPMENT

MAX_TOP = 50
MAX_SAMPLE = 20_000
TEST_FLAG = "--test-months"
EXPLORATORY = ("exploratory, test months: a pattern seen through explain on 2019-2024 hours is never tested as a hypothesis "
               "on those months; it waits for the validation year's scheduled opening, or for the vault")  # fmt: skip


class Refused(ValueError):
    """The call is refused before any result: a period out of reach, a missing flag, a wrong digest, a changed store."""


class LoudCheckFailed(AssertionError):
    """One of ``why``'s two loud checks failed: the call stops, and nothing is printed as a result."""


# --- Periods ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Period:
    """Calendar months ``first .. last``, both included."""

    first: Month
    last: Month

    @property
    def start_ms(self) -> int:
        return month_start_ms(self.first.year, self.first.month)

    @property
    def end_ms(self) -> int:
        return next_month_ms(month_start_ms(self.last.year, self.last.month))

    def __str__(self) -> str:
        return f"{self.first}..{self.last}"


DEFAULT_PERIOD = Period(Month(2017, 11), Month(2018, 12))
REACH = Period(Month(2017, 10), Month(2024, 12))  # the development panel; the validation year and the vault are out of reach
REACH_END_MS = int(DEVELOPMENT.end.timestamp() * 1000)


def parse_period(text: str) -> Period:
    """``YYYY-MM:YYYY-MM`` (or one ``YYYY-MM``), both months included."""
    first, _, last = text.partition(":")
    try:
        period = Period(Month.parse(first), Month.parse(last or first))
    except ValueError as exc:
        raise Refused(f"period {text!r}: {exc}") from None
    if period.last < period.first:
        raise Refused(f"period {text!r}: the last month is before the first")
    return period


def check_period(period: Period, flag: bool) -> None:
    """Refuse a period out of reach (2025 and later, before 2017-10), or other than the default without the flag."""
    if period.end_ms > REACH_END_MS:
        raise Refused(f"period {period} reaches 2025 or later: the validation year and the vault are out of explain's reach")
    if period.start_ms < REACH.start_ms:
        raise Refused(f"period {period} starts before the development panel ({REACH.first})")
    if period != DEFAULT_PERIOD and not flag:
        raise Refused(f"period {period} is not the default ({DEFAULT_PERIOD}): name it with {TEST_FLAG}")


def check_as_of(as_of_ms: int, flag: bool) -> None:
    """Refuse an ``as_of`` out of reach, or outside the default period without the flag."""
    if as_of_ms >= REACH_END_MS:
        raise Refused(f"as_of {iso(as_of_ms)} is in 2025 or later: the validation year and the vault are out of explain's reach")
    if as_of_ms < REACH.start_ms:
        raise Refused(f"as_of {iso(as_of_ms)} is before the development panel")
    if not DEFAULT_PERIOD.start_ms <= as_of_ms < DEFAULT_PERIOD.end_ms and not flag:
        raise Refused(f"as_of {iso(as_of_ms)} is outside the default period ({DEFAULT_PERIOD}): name it with {TEST_FLAG}")


def parse_time(text: str) -> int:
    """An ISO 8601 instant in epoch milliseconds; a time without a zone is UTC."""
    try:
        when = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise Refused(f"time {text!r}: {exc}") from None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    ms = when.timestamp() * 1000
    if ms != int(ms):
        raise Refused(f"time {text!r} is finer than a millisecond")
    return int(ms)


def month_label(ms: int) -> str:
    return iso(ms)[:7]


# --- The run folder ---------------------------------------------------------------------------


@dataclass(eq=False)
class RunFolder:
    """A Report n°2 run, opened by the SHA-256 of its ``series.npz``."""

    path: Path
    series_sha256: str
    report: dict[str, Any]
    config: dict[str, Any]
    report_sha256: str
    arrays: dict[str, np.ndarray]
    cfg: Report2Config

    @property
    def arms(self) -> list[str]:
        return [str(a) for a in self.arrays["arms"]]


def rebuild_config(config: dict[str, Any]) -> Report2Config:
    """The run's ``Report2Config`` from its ``config.json``; it must write back the same JSON."""

    def tup(v: Any) -> Any:
        return tuple(tup(x) for x in v) if isinstance(v, list) else v

    raw = config.get("config")
    if not isinstance(raw, dict) or not isinstance(raw.get("matrix"), dict):
        raise Refused("config.json has no config with a matrix: not a Report n°2 run")
    rest = {k: tup(v) for k, v in raw.items() if k != "matrix"}
    try:
        cfg = Report2Config(matrix=MatrixConfig(**{k: tup(v) for k, v in raw["matrix"].items()}), **rest)
    except TypeError as exc:
        raise Refused(f"config.json's config does not rebuild a Report2Config: {exc}") from None
    if json.loads(json.dumps(asdict(cfg))) != raw:
        raise Refused("config.json's config does not round-trip through Report2Config: the run was made by other code")
    return cfg


def verify_digest(run_dir: Path, sha256: str) -> None:
    """Before any contact line: hash ``series.npz`` (bytes only, nothing parsed) and match the digest given and its run's record."""
    path = Path(run_dir) / SERIES_FILE
    want = sha256.strip().lower()
    if not path.is_file():
        raise Refused(f"{path}: no such file")
    got = sha256_file(path)
    if got != want:
        raise Refused(f"{path}: SHA-256 {got}, not the {want} given")
    report_path = path.parent / "report2.json"
    if not report_path.is_file():
        raise Refused(f"{path.parent}: no report2.json beside the series, so its run cannot vouch for it")
    try:
        rec = json.loads(report_path.read_bytes()).get("series")
    except (ValueError, AttributeError):
        raise Refused(f"{report_path}: not a JSON object") from None
    if not isinstance(rec, dict) or rec.get("file") != path.name or rec.get("sha256") != got:
        raise Refused(f"{path}: its run's report2.json records {rec!r}, not this file's SHA-256 {got}")


def open_run(run_dir: Path, sha256: str, keys: tuple[str, ...] | None = None) -> RunFolder:
    """Open a run folder by its series digest (refused unless the file and the run's report2.json both agree)."""
    run_dir = Path(run_dir)
    keys = tuple(k for k, _, _ in SERIES_KEYS) if keys is None else keys
    try:
        arrays, report, config, report_sha = power.load_series(run_dir / SERIES_FILE, sha256, keys)
    except power.SeriesRefused as exc:
        raise Refused(str(exc)) from None
    except FileNotFoundError as exc:
        raise Refused(f"{run_dir}: {exc.strerror}: {exc.filename}") from None
    cfg = rebuild_config(config)
    return RunFolder(run_dir, sha256.strip().lower(), report, config, report_sha, arrays, cfg)


# --- Provenance and contacts ------------------------------------------------------------------


def provenance(run: RunFolder, *, panel_verified: bool, contacts: Path) -> dict[str, Any]:
    """Every answer's provenance: this code's commit, dirty flag and digest; the run's data manifest, panel and series; the contacts log."""
    state = code_state()
    return {
        "contacts_file": str(contacts),
        "commit": prov_mod.git_commit(),
        "dirty": state["dirty"],
        "dirty_scope": state["scope"],
        "code_sha256": prov_mod.code_digest()["sha256"],
        "manifest_digest": run.config.get("provenance", {}).get("manifest_digest"),
        "panel_fingerprint": run.report["panel"]["fingerprint_sha256"],
        "panel_fingerprint_verified": panel_verified,
        "series_sha256": run.series_sha256,
        "report2_json_sha256": run.report_sha256,
        "run": run.path.name,
        "run_commit": run.config.get("provenance", {}).get("git_commit"),
        "registered_run": bool(run.report.get("registered")),
    }


def contacts_file(path: Path | None = None) -> Path:
    """The log of contacts, resolved (symbolic links included): ``<runs>/explain/CONTACTS.log`` unless a test names another."""
    return (runs_dir() / "explain" / "CONTACTS.log" if path is None else Path(path)).resolve()


def check_jsonl(path: Path | None) -> Path | None:
    """``--jsonl``, resolved (symbolic links included); refused under ``<runs>/explain`` or ``<runs>/vault``, or if it exists."""
    if path is None:
        return None
    target = Path(path).resolve()
    for name in ("explain", "vault"):
        guarded = (runs_dir() / name).resolve()
        if target == guarded or guarded in target.parents:
            raise Refused(f"--jsonl {path} resolves to {target}, under {guarded}: explain never writes there")
    if target.exists():
        raise Refused(f"--jsonl {target} exists: explain writes a new file only")
    return target


def log_contact(path: Path | None, *, subcommand: str, period: str, arm: str, series_sha256: str) -> None:
    """Append one line to the log of contacts: UTC time, subcommand, period touched, arm, commit and the series digest."""
    path = contacts_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "subcommand": subcommand, "period": period, "arm": arm,
            "commit": prov_mod.git_commit(), "dirty": code_state()["dirty"], "series_sha256": series_sha256.strip().lower()}  # fmt: skip
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


# --- Output -----------------------------------------------------------------------------------


def clean(value: Any) -> Any:
    """JSON-ready: numpy scalars and arrays to Python, NaN and infinities to None, -0.0 to 0.0."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


def json_line(obj: Any) -> str:
    return json.dumps(clean(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
