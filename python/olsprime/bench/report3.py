"""Report n°3: the encoder arms on the panel, H4, H5, H13 and the cost table (``docs/PROTOCOL.md``, iteration 4).

Usage::

    uv run --no-sync python -m olsprime.bench.report3 --series <Report n°2 run>/series.npz --sha256 <digest>
        [--out DIR] [--stage fits|timing|matrix-timing|score|all] [--arms ARM ...] [--device cuda|cpu]
        [--driver VERSION] [--cpu-name NAME]
    uv run --no-sync python -m olsprime.bench.report3 --series ... --sha256 ... --check DIR
    uv run --no-sync python -m olsprime.bench.report3 --synthetic [--out DIR] [--stage ...]

Every choice is the registered one ("The encoder arms", "H4, H5 and H13", "Variants"); the build's
readings are those of the change-log entry "2026-10-30, wave 1". The stages:

- **fits** (GPU, WSL2). The panel is Report n°2's own: built as ``report2`` builds it (member months
  of the sealed universe, ``load_klines``, ``pair_rows``, ``assemble``), refused unless its
  fingerprint is the one Report n°2's ``report2.json`` records, and unless its scored rows are
  exactly the rows of Report n°2's ``series.npz`` (symbol, decision time, open time and label). The
  windows come from the same bars (:func:`olsprime.train.fit.encoder_data`). For each arm, seed and
  fit year it trains the fit (:class:`olsprime.train.fit.EncoderSystem`) and forecasts every scored
  row of that year (a row's year is that of its month). One part per (arm, seed, fit) under
  ``<out>/parts/fits/<arm>/s<seed>/<year>/``: ``fit.safetensors`` (the checkpoint, every epoch's
  parameters, the fit's inputs), ``forecast.safetensors`` (the probabilities before the floor and
  the fallback flag of each row, keyed by symbol and decision time, with its open time and label),
  ``timing.json`` (wall clock, peak memory) and, written last, ``part.json`` (both files' SHA-256,
  every epoch's digest, the slice losses, the epoch kept). Each file sits beside its SHA-256. A
  restart verifies every part and skips it; it refuses to start if the configuration, the code
  digest, the data manifest, the series or the panel changed (``parts/IDENTITY.json``). ``--arms``
  restricts the stage, so that the arms can run in turn.
- **timing** (GPU, WSL2), H13's protocol: 12,288 scored rows with a valid window drawn with the seed
  20261030 (:func:`timing_sample`), 3 batches of 4,096; per repeat 3 untimed warm-up batches, then
  the 3 timed batches with ``torch.cuda.synchronize()`` before and after each; 10 repeats cycling the
  arms (``enc-ols``, ``enc-softmax``, ``enc-linear``, ...: ABAB for the two of H13); plain fp32 eager,
  the softmax on the math backend, the deterministic settings on. A decision is the whole pass of
  :meth:`EncoderSystem.predict` on a batch: the encoder, the memory readout (the prefix sums of the
  memory's statistics are built before, untimed) and the head. Each arm runs its first seed's fit of
  the last year. Two parts of that pass are timed the same way, as descriptions: the attention
  operator alone on the Q, K and V of the same batches, and the memory readout alone on the same
  batches' cached keys.
- **matrix-timing** (CPU; the plan runs it in WSL2 with every other stage, and the platform is
  recorded): the eight Matrix arms' readouts on the same sample, with the fit-time choices of
  Report n°2's last test month, the same protocol, the whole process's peak memory where the
  platform gives it, and the memory's own analytic footprint.

The device (``--device``) is operational: it is no part of a folder's identity, so the stages that
do not train (matrix-timing, score, ``--check``) accept any. Every fit of a folder must run on one
device, and each part records it. Every stage binds the configuration, the code digest, the data
manifest and the series of the fits (``parts/IDENTITY.json``): none runs on other code than theirs.
- **score** (CPU, deterministic): reads the saved forecasts and Report n°2's ``series.npz`` by its
  digest (:func:`olsprime.bench.power.load_series`). The ensemble of an arm is the mean of its seeds'
  probabilities before the floor, then the floor (:func:`olsprime.train.fit.ensemble`); a row whose
  window failed takes the arm's climatology, and is counted. The encoder arms' rows must be Report
  n°2's scored rows, one for one, so their hourly series pair hour by hour; any mismatch is refused
  (:class:`PairingRefused`). Then Report n°2's statistics, unchanged (:mod:`olsprime.bench.report2`):
  hourly means, Brier skill intervals, ECE, Diebold-Mariano with HLN and the stationary bootstrap with
  the registered seeds, and the rules of H4, H5 and H13 (:func:`verdicts`).

``all`` runs fits, timing and score. Outputs, to ``runs/<UTC date>-report3/`` or ``--out``:
``report3.json`` (sorted keys, no wall clock, the digest of Report n°2's ``series.npz``),
``report-3.md``, ``series.npz`` (the encoder arms' hourly series and scored rows, written as
``report2`` writes its own, its digest in ``report3.json``), ``config.json`` (configuration and
provenance) and ``run.json`` (wall clock; GPU, driver, CUDA and torch versions). ``--check DIR``
verifies every saved part by its digest (the fits are never re-run), re-runs the score stage from
the saved forecasts into a temporary folder and compares the outputs byte for byte.

``--synthetic`` is the smoke: a synthetic panel and a synthetic Report n°2 run built in memory and
written under ``<out>/report2-synthetic/``. It never reads stored data.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from olsprime.bench import provenance as prov_mod
from olsprime.bench import report2 as r2
from olsprime.bench.power import load_series
from olsprime.eval import stats
from olsprime.features.panel import _BAR_COLUMNS as PANEL_BAR_COLUMNS
from olsprime.features.panel import INTERVAL, TOP_N, assemble, fingerprint, member_months, pair_rows
from olsprime.models.encoder import (
    ARMS,
    EncoderConfig,
    EncoderModel,
    Standardizer,
    gather_windows,
    linear_attention,
    ols_attention,
    softmax_attention,
)
from olsprime.models.matrix import N_CLASSES, P, Q, MatrixConfig, MatrixSystem, PanelArrays, month_start_ms, prob_map
from olsprime.models.matrix import ARMS as MATRIX_ARMS
from olsprime.models.memory import MemoryConfig, prefix_moments, readout, window_bounds
from olsprime.paths import runs_dir
from olsprime.train import checkpoint
from olsprime.train.determinism import build as build_info
from olsprime.train.determinism import deterministic
from olsprime.train.fit import FIT_YEARS, SEEDS, EncoderData, EncoderSystem, FitConfig, encoder_data, ensemble, load_state, week_start, year_start_ms

REPORT_FILE = "report-3.md"
SERIES_FILE = r2.SERIES_FILE
COMPARED = ("config.json", "report3.json", REPORT_FILE, SERIES_FILE)
CLIMATOLOGY = r2.CLIMATOLOGY
MATRIX_SELECT = "matrix-select"
STAGES = ("fits", "timing", "matrix-timing", "score", "all")
VARIANTS_BEFORE = 18  # Report n°1's 10 baselines and Report n°2's 8 arms
# The comparisons, difference a - b: (name, a, b, kind). "h4" non-inferiority, "h5" superiority, "described" no rule.
COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("H4", "enc-ols", "enc-softmax", "h4"),
    ("H5", "enc-ols", MATRIX_SELECT, "h5"),
    ("enc-linear vs enc-ols", "enc-linear", "enc-ols", "described"),
    ("enc-linear vs enc-softmax", "enc-linear", "enc-softmax", "described"),
)
# What score reads of Report n°2's series.npz (load_series adds arms, hours, count and log_loss).
SERIES_READ = ("brier", "label", "fwd_return", "symbol", "symbols", "open_time", "decision_time", "label_known_at", "hour")
TIMING_FILE = "timing.json"
MATRIX_TIMING_FILE = "matrix-timing.json"


class PairingRefused(ValueError):
    """The encoder arms' rows are not Report n°2's scored rows, one for one."""


class ResumeRefused(RuntimeError):
    """The output folder holds parts of another run."""


class Interrupted(RuntimeError):
    """Raised after ``stop_after`` new parts (tests of the resume)."""


class PartRefused(RuntimeError):
    """A saved file fails its digest, or a part is not the part of its folder (another arm, seed, year or configuration)."""


# --- Configuration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Report3Config:
    """Everything Report n°3 fixes. Saved with every run; a resume needs the same."""

    arms: tuple[str, ...] = ARMS
    seeds: tuple[int, ...] = SEEDS
    years: tuple[int, ...] = FIT_YEARS
    device: str = "cuda"  # operational: outside config_json and the identity; each fit records its own
    encoder: tuple[tuple[str, Any], ...] = ()  # EncoderConfig overrides: tests and smoke runs only
    fit: tuple[tuple[str, Any], ...] = ()  # FitConfig overrides: tests and smoke runs only
    h4_margin: float = 0.005  # times climatology-panel's mean hourly log-loss
    h13_ratio: float = 0.9
    timing_seed: int = 20261030
    timing_batch: int = 4096
    timing_batches: int = 3
    timing_warmups: int = 3
    timing_repeats: int = 10
    per_decisions: int = 10_000
    variants_before: int = VARIANTS_BEFORE
    synthetic: str | None = None  # a description of the synthetic panel; None: the real one

    def __post_init__(self) -> None:
        if set(self.arms) != set(ARMS) or len(self.arms) != len(ARMS):
            raise ValueError(f"Report n°3 has the three arms {ARMS}; --arms restricts a stage, not the configuration")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("the device is cuda or cpu")
        if not self.seeds or not self.years or list(self.years) != sorted(set(self.years)):
            raise ValueError("seeds and years must be non-empty, the years increasing")

    @property
    def registered(self) -> bool:
        """The registered configuration; the device is checked on the fits themselves (:func:`score`)."""
        return dataclasses.replace(self, device="cuda") == Report3Config()

    @property
    def timing_rows(self) -> int:
        return self.timing_batch * self.timing_batches


def config_json(cfg: Report3Config) -> dict[str, Any]:
    """The configuration as saved and bound by the identity: without the device, which is operational."""
    out = dataclasses.asdict(cfg)
    del out["device"]
    out["registered"] = cfg.registered
    return json.loads(json.dumps(r2._clean(out)))


def fit_config(cfg: Report3Config, arm: str, seed: int, year: int) -> FitConfig:
    return FitConfig(fit_ms=year_start_ms(year), seed=seed, encoder=EncoderConfig(arm=arm, **dict(cfg.encoder)), **dict(cfg.fit))


def rule_config(report2_config: Mapping[str, Any]) -> r2.Report2Config:
    """Report n°2's statistics, with the parameters its run used (``config.json``): resamples, level, DM lags, seed, ECE bins."""
    c = report2_config["config"]
    return r2.Report2Config(n_boot=int(c["n_boot"]), level=float(c["level"]), dm_h=int(c["dm_h"]), bootstrap_seed=int(c["bootstrap_seed"]),
                            ece_bins=int(c["ece_bins"]))  # fmt: skip


def matrix_config(report2_config: Mapping[str, Any]) -> MatrixConfig:
    """Report n°2's MatrixConfig, from its ``config.json`` (lists back to tuples)."""
    m = dict(report2_config["config"]["matrix"])
    return MatrixConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in m.items()})


# --- Report n°2's run, read by its digest --------------------------------------------------------


@dataclass(eq=False)
class Report2Run:
    """Report n°2's ``series.npz`` (the arrays score reads), its ``report2.json`` and ``config.json``."""

    path: Path
    sha256: str
    series: dict[str, np.ndarray]
    report: dict[str, Any]
    config: dict[str, Any]
    report_sha256: str

    @property
    def fingerprint(self) -> str:
        return str(self.report["panel"]["fingerprint_sha256"])

    def symbol_names(self) -> np.ndarray:
        return np.asarray(self.series["symbols"])[self.series["symbol"]]

    def ll(self, arm: str) -> np.ndarray:
        return self.series["log_loss"][list(self.series["arms"]).index(arm)]

    def brier(self, arm: str) -> np.ndarray:
        return self.series["brier"][list(self.series["arms"]).index(arm)]


def read_report2(path: Path, sha256: str) -> Report2Run:
    series, report, config, report_sha = load_series(Path(path), sha256, keys=SERIES_READ)
    run = Report2Run(Path(path), sha256.strip().lower(), series, report, config, report_sha)
    dec = series["decision_time"]
    if dec.size and np.any(np.diff(dec) < 0):
        raise PairingRefused(f"{path}: the scored rows are not in panel order")
    if CLIMATOLOGY not in list(series["arms"]) or MATRIX_SELECT not in list(series["arms"]):
        raise PairingRefused(f"{path}: no {CLIMATOLOGY} or {MATRIX_SELECT} series")
    return run


# --- The panel and its windows --------------------------------------------------------------------


def load_panel(report2_config: Mapping[str, Any], *, data_dir: Path | None = None, universe: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Report n°2's panel, built as :func:`olsprime.features.panel.hourly_panel` builds it, and the bars of its pairs.

    The member months of the sealed universe between Report n°2's first and last month, each
    member's hourly bars through ``load_klines`` (the seal), ``pair_rows`` and ``assemble``. The
    bars are kept: the windows read them.
    """
    from olsprime import paths
    from olsprime.data.binance_vision import Month
    from olsprime.protocol import load_klines, load_universe

    c = report2_config["config"]
    root = data_dir if data_dir is not None else paths.data_dir()
    if universe is None:
        from olsprime.universe import universe_path

        universe = load_universe(universe_path(root / "parquet", TOP_N), caller="bench.report3")
    members = member_months(universe, first=Month(*c["first_month"]), last=Month(*c["last_month"]))
    if c.get("symbols") is not None:
        members = members.filter(pl.col("symbol").is_in(list(c["symbols"])))
    parts, bars = [], {}
    for symbol, rows in members.group_by("symbol", maintain_order=True):
        frame = load_klines(symbol[0], INTERVAL, data_dir=root, columns=PANEL_BAR_COLUMNS, caller="bench.report3")
        bars[symbol[0]] = frame
        parts.append(pair_rows(frame, rows["month"].to_list()))
    return assemble(parts), bars


def scored_panel_rows(arrays: PanelArrays, run: Report2Run) -> np.ndarray:
    """The panel rows of Report n°2's scored rows, in its order; refused unless they match one for one.

    Each scored row must be a complete panel row with the same symbol, decision time, open time and
    label, and every complete row of the test months must be a scored row.
    """
    s = run.series
    k = len(arrays.symbols) + 1
    code = {name: i for i, name in enumerate(arrays.symbols)}
    names = run.symbol_names()
    unknown = sorted({str(n) for n in np.unique(names)} - set(code))
    if unknown:
        raise PairingRefused(f"Report n°2's scored rows name symbols the panel lacks: {unknown[:5]}")
    sym = np.array([code[str(n)] for n in np.asarray(s["symbols"])], dtype=np.int64)[s["symbol"]]
    pk = arrays.decision_ms * k + arrays.symbol
    if pk.size > 1 and np.any(np.diff(pk) <= 0):
        raise PairingRefused("the panel is not sorted by (decision time, symbol) without repeats")
    sk = s["decision_time"].astype(np.int64) * k + sym
    pos = np.searchsorted(pk, sk)
    found = pos < pk.size
    found[found] &= pk[pos[found]] == sk[found]
    if not found.all():
        i = int(np.flatnonzero(~found)[0])
        raise PairingRefused(f"scored row {i} ({names[i]}, decision {r2.iso(int(s['decision_time'][i]))}) is not a panel row")
    bad = ~(arrays.complete[pos] & (arrays.open_ms[pos] == s["open_time"]) & (arrays.label[pos] == s["label"]))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise PairingRefused(f"scored row {i} ({names[i]}, decision {r2.iso(int(s['decision_time'][i]))}) differs from its panel row")
    mc = matrix_config(run.config)
    first, end = month_start_ms(*mc.first_test), r2.next_month_ms(month_start_ms(*mc.last_test))
    in_test = arrays.complete & (arrays.month_ms >= first) & (arrays.month_ms < end)
    if int(in_test.sum()) != pos.size or not in_test[pos].all():
        raise PairingRefused(f"the panel has {int(in_test.sum())} complete rows in the test months, Report n°2 scored {pos.size}")
    return pos


def check_panel(panel: pl.DataFrame, run: Report2Run) -> str:
    fp = fingerprint(panel)
    if fp != run.fingerprint:
        raise PairingRefused(f"the panel's fingerprint {fp} is not Report n°2's {run.fingerprint}")
    return fp


def year_of_month(month_ms: np.ndarray) -> np.ndarray:
    return np.array([datetime.fromtimestamp(int(m) / 1000, tz=UTC).year for m in month_ms], dtype=np.int64)


def rows_by_year(arrays: PanelArrays, rows: np.ndarray, years: Sequence[int]) -> dict[int, np.ndarray]:
    """The scored rows of each fit year (a row's year is that of its month), panel order; every scored row in a fit year."""
    year = year_of_month(arrays.month_ms[rows]) if rows.size else np.zeros(0, dtype=np.int64)
    missing = sorted(set(year.tolist()) - set(years))
    if missing:
        raise PairingRefused(f"scored rows fall in years without a fit: {missing}")
    if rows.size > 1 and np.any(np.diff(rows) <= 0):
        raise PairingRefused("the scored rows are not in panel order")
    return {y: rows[year == y] for y in years}


# --- Parts ------------------------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_text(obj: Any) -> bytes:
    return (json.dumps(r2._clean(obj), sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")


class Parts:
    """Saved parts under ``<out>/parts/``, each file beside its SHA-256; written atomically, verified when read.

    ``identity`` (the fits) must equal ``parts/IDENTITY.json`` or the folder is refused; ``None``
    (the other stages) checks only the keys in ``require``.
    """

    def __init__(self, out: Path, identity: dict[str, Any] | None, *, require: Mapping[str, Any] | None = None,
                 create: dict[str, Any] | None = None, log: Callable[[str], None] = lambda s: None,
                 stop_after: int | None = None) -> None:  # fmt: skip
        self.dir = out / "parts"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log, self.stop_after, self.saved = log, stop_after, 0
        path = self.dir / "IDENTITY.json"
        if identity is not None:
            text = _json_text(identity)
            if path.exists():
                if path.read_bytes() != text:
                    old = json.loads(path.read_text(encoding="utf-8"))
                    new = json.loads(text)
                    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
                    raise ResumeRefused(f"{self.dir} holds parts of another run: {', '.join(changed)} changed. Use another --out.")
            else:
                _write_atomic(path, text)
        if require and not path.exists() and create is not None:
            _write_atomic(path, _json_text(create))  # the Matrix CPU stage may run first, on its own machine
        if require:
            if not path.exists():
                raise ResumeRefused(f"{self.dir}: no IDENTITY.json; run the fits stage first")
            old = json.loads(path.read_text(encoding="utf-8"))
            new = json.loads(_json_text(dict(require)))
            changed = sorted(k for k in new if old.get(k) != new[k])
            if changed:
                raise ResumeRefused(f"{self.dir} holds parts of another run: {', '.join(changed)} differ")

    def identity(self) -> dict[str, Any]:
        return json.loads((self.dir / "IDENTITY.json").read_text(encoding="utf-8"))

    def put(self, key: str, data: bytes) -> str:
        sha = _sha(data)
        _write_atomic(self.dir / key, data)
        _write_atomic(self.dir / (key + ".sha256"), (sha + "\n").encode("ascii"))
        return sha

    def get(self, key: str, *, unsealed_ok: bool = False) -> bytes | None:
        """The file's bytes, verified against its digest; None if absent.

        A file that exists and fails its digest is refused (:class:`PartRefused`), never read as absent.
        A file without its digest beside it is refused too, unless ``unsealed_ok``: a part interrupted
        between its two writes, which the fits stage then redoes.
        """
        path, digest = self.dir / key, self.dir / (key + ".sha256")
        if not path.is_file():
            return None
        if not digest.is_file():
            if unsealed_ok:
                return None
            raise PartRefused(f"parts/{key}: exists without its digest")
        data = path.read_bytes()
        if _sha(data) != digest.read_text(encoding="ascii").strip():
            raise PartRefused(f"parts/{key}: fails its digest")
        return data

    def put_json(self, key: str, obj: Any) -> str:
        return self.put(key, _json_text(obj))

    def get_json(self, key: str, *, unsealed_ok: bool = False) -> Any:
        data = self.get(key, unsealed_ok=unsealed_ok)
        return None if data is None else json.loads(data)


def part_key(arm: str, seed: int, year: int) -> str:
    return f"fits/{arm}/s{seed}/{year}"


def parse_key(key: str) -> tuple[str, int, int]:
    """``(arm, seed, year)`` of a part key ``fits/<arm>/s<seed>/<year>``."""
    head, arm, seed, year = key.split("/")
    if head != "fits" or arm not in ARMS or not seed.startswith("s"):
        raise ValueError(f"not a part key: {key!r}")
    return arm, int(seed[1:]), int(year)


def _metadata(data: bytes) -> dict[str, str]:
    """A checkpoint's string metadata, from its header alone (the arrays are not read)."""
    (size,) = struct.unpack("<Q", data[:8])
    if size > checkpoint.MAX_HEADER or 8 + size > len(data):
        raise PartRefused("not a checkpoint: bad header length")
    return dict(json.loads(data[8 : 8 + size].decode("utf-8")).get("__metadata__", {}))


def _expect(where: str, what: str, got: Any, want: Any) -> None:
    if got != want:
        raise PartRefused(f"{where}: {what} is {got!r}, the folder's is {want!r}: not this part's file")


def verified_part(parts: Parts, key: str, cfg: Report3Config) -> dict[str, Any] | None:
    """The part's record, every file verified against its digest and against the part's own place.

    None if the part is not finished (no sealed ``part.json``: the fits stage redoes it). Otherwise
    refused (:class:`PartRefused`) unless both arrays files match ``part.json``'s digests, and unless
    ``part.json``, the checkpoint's config and the forecast's metadata all name the key's arm, seed,
    fit year and fit instant, the checkpoint's config being the folder's for that fit: a part copied
    into another arm's or seed's folder verifies its digests but is refused here.
    """
    arm, seed, year = parse_key(key)
    fc = fit_config(cfg, arm, seed, year)
    want_config = json.loads(fc.to_json())
    rec = parts.get_json(f"{key}/part.json", unsealed_ok=True)
    if rec is None:
        return None
    where = f"parts/{key}/part.json"
    for name, want in (("arm", arm), ("seed", seed), ("year", year), ("fit_ms", fc.fit_ms), ("fit_config", want_config)):
        _expect(where, name, rec.get(name), want)
    files = {}
    for name, sha in (("fit.safetensors", rec.get("fit_sha256")), ("forecast.safetensors", rec.get("forecast_sha256"))):
        data = parts.get(f"{key}/{name}")
        if data is None or _sha(data) != sha:
            raise PartRefused(f"parts/{key}/{name}: missing or not the file part.json records")
        files[name] = data
    where = f"parts/{key}/fit.safetensors"
    c = json.loads(_metadata(files["fit.safetensors"])["config"])
    _expect(where, "encoder.arm", c["encoder"]["arm"], arm)
    _expect(where, "seed", c["seed"], seed)
    _expect(where, "fit_ms", c["fit_ms"], fc.fit_ms)
    _expect(where, "config", c, want_config)
    where = f"parts/{key}/forecast.safetensors"
    meta = _metadata(files["forecast.safetensors"])
    for name, want in (("arm", arm), ("seed", str(seed)), ("year", str(year)), ("fit_ms", str(fc.fit_ms)), ("fit_sha256", rec["fit_sha256"])):
        _expect(where, name, meta.get(name), want)
    return rec


# --- The fits stage -------------------------------------------------------------------------------


def _peak_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gpu(device: torch.device) -> int | None:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def process_peak_bytes() -> int | None:
    """The process's peak resident memory, where the platform gives it without a subprocess (Linux: ``ru_maxrss``)."""
    try:
        import resource
    except ImportError:  # Windows: neither resource nor a stdlib route (psutil is not a dependency; ctypes is banned)
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def forecast_arrays(data: EncoderData, out: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
    a = data.arrays
    rows = out["rows"]
    return {"raw": np.ascontiguousarray(out["raw"], dtype=np.float64), "fallback": np.asarray(out["fallback"], dtype=bool),
            "symbol": a.symbol[rows].astype(np.int64), "decision_time": a.decision_ms[rows].astype(np.int64),
            "open_time": a.open_ms[rows].astype(np.int64), "label": y[rows].astype(np.int64)}  # fmt: skip


def plan_fits(data: EncoderData, cfg: Report3Config, arms: Sequence[str]) -> dict[str, Any]:
    """Each fit's training weeks and steps, from the training masks alone (no label value is read)."""
    a = data.arrays
    out = []
    for year in cfg.years:
        fc = fit_config(cfg, arms[0], cfg.seeds[0], year)
        mask = data.usable & (a.lka_ms <= fc.fit_ms - fc.train_gap_ms)
        weeks = int(np.unique(week_start(a.decision_ms[mask])).size)
        out.append({"year": year, "training_rows": int(mask.sum()), "weeks": weeks, "steps": weeks * fc.epochs})
    per_seed = sum(f["steps"] for f in out)
    return {"fits": out, "steps_per_arm_and_seed": per_seed, "steps": per_seed * len(cfg.seeds) * len(arms)}


def run_fits(data: EncoderData, rows: dict[int, np.ndarray], cfg: Report3Config, parts: Parts, arms: Sequence[str], *,
             log: Callable[[str], None] = lambda s: None, driver: str | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Train and forecast every (arm, seed, year) not already saved; returns their records."""
    device = torch.device(cfg.device)
    records = []
    symbols = list(data.arrays.symbols)
    for arm in arms:
        for seed in cfg.seeds:
            for year in cfg.years:
                key = part_key(arm, seed, year)
                rec = verified_part(parts, key, cfg)
                if rec is not None:
                    if rec.get("device") != cfg.device:
                        raise ResumeRefused(f"{key} was fitted on {rec.get('device')!r}, this run is on {cfg.device!r}: "
                                            "every fit of a folder runs on one device")  # fmt: skip
                    log(f"{key}: saved, verified, skipped")
                    records.append(rec)
                    continue
                if parts.stop_after is not None and parts.saved >= parts.stop_after:
                    raise Interrupted(f"stopped after {parts.saved} new parts")
                fc = fit_config(cfg, arm, seed, year)
                system = EncoderSystem(data, fc, device=device)
                _peak_reset(device)
                started = time.perf_counter()
                res = system.fit()
                peak_train = _peak_gpu(device)
                t = time.perf_counter()
                out = system.forecast(res, rows[year])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forecast_s = time.perf_counter() - t
                seconds = time.perf_counter() - started
                folder = parts.dir / key
                fit_sha = res.save(folder)
                fit_bytes = (folder / "fit.safetensors").read_bytes()
                parts.put(f"{key}/fit.safetensors", fit_bytes)
                arrays = forecast_arrays(data, out, system.y)
                meta = {"arm": arm, "seed": str(seed), "year": str(year), "fit_ms": str(fc.fit_ms), "symbols": json.dumps(symbols),
                        "fit_sha256": fit_sha}  # fmt: skip
                forecast_sha = parts.put(f"{key}/forecast.safetensors", checkpoint.to_bytes(arrays, meta))
                parts.put_json(f"{key}/timing.json", {"seconds": seconds, "train_s": res.timing.get("train_s"), "slice_s": res.timing.get("slice_s"),
                                                      "forecast_s": forecast_s, "steps": res.steps, "peak_gpu_bytes_training": peak_train,
                                                      "device": str(device), "gpu": res.build.get("gpu"), "driver": driver,
                                                      "finished_at": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
                rec = {
                    "arm": arm, "seed": seed, "year": year, "fit_ms": fc.fit_ms, "fit_config": json.loads(fc.to_json()),
                    "fit_sha256": fit_sha, "forecast_sha256": forecast_sha, "epoch_digests": res.epoch_digests(),
                    "slice_losses": list(res.slice_losses), "kept_epoch": res.kept_epoch, "steps": res.steps,
                    "n_parameters": res.n_parameters, "training_rows": int(res.train_rows.size), "slice_rows": int(res.slice_rows.size),
                    "climatology": res.climatology.tolist(), "climatology_rows": res.climatology_rows,
                    "complete_without_window": res.complete_without_window, "forecast_rows": int(rows[year].size),
                    "fallbacks": int(arrays["fallback"].sum()), "build": res.build, "device": cfg.device,
                }  # fmt: skip
                parts.put_json(f"{key}/part.json", rec)
                parts.saved += 1
                records.append(rec)
                log(f"{key}: {seconds:.1f} s, {res.steps} steps ({res.timing.get('train_s', 0.0) / max(res.steps, 1):.3f} s/step), "
                    f"kept epoch {res.kept_epoch}, slice losses {', '.join(f'{v:.4f}' for v in res.slice_losses)}")  # fmt: skip
    return records


# --- Forecasts and the ensemble ------------------------------------------------------------------------


@dataclass(eq=False)
class Forecasts:
    """One arm's forecasts of every scored row, per seed, in the order of Report n°2's scored rows."""

    raw: dict[int, np.ndarray]  # seed -> (N, 3), before the floor
    fallback: np.ndarray  # (N,) the row's window failed: the arm's climatology


def read_forecast(parts: Parts, key: str) -> dict[str, Any]:
    """The forecast of a part, refused unless its metadata names the key's arm, seed and year (and its digest verifies)."""
    data = parts.get(f"{key}/forecast.safetensors")
    if data is None:
        raise PairingRefused(f"{key}: no verified forecast")
    arrays, meta = checkpoint.from_bytes(data)
    arm, seed, year = parse_key(key)
    for name, want in (("arm", arm), ("seed", str(seed)), ("year", str(year)), ("fit_ms", str(year_start_ms(year)))):
        _expect(f"parts/{key}/forecast.safetensors", name, meta.get(name), want)
    arrays["symbols"] = json.loads(meta["symbols"])
    return arrays


def pair_forecasts(parts_list: Sequence[dict[str, Any]], run: Report2Run, where: str) -> dict[str, np.ndarray]:
    """One (arm, seed)'s forecasts over every fit, in Report n°2's order; refused unless the rows are its scored rows, one for one.

    Each forecast's key is its symbol (by name, coded as Report n°2 codes it) and decision time; its open
    time and label must then be the scored row's too.
    """
    s = run.series
    code = {str(n): i for i, n in enumerate(np.asarray(s["symbols"]))}
    codes = []
    for p in parts_list:
        names = [str(n) for n in p["symbols"]]
        unknown = sorted(set(names) - set(code))
        if unknown:
            raise PairingRefused(f"{where}: forecasts of symbols Report n°2 never scored: {unknown[:5]}")
        codes.append(np.array([code[n] for n in names], dtype=np.int64)[p["symbol"]])
    sym = np.concatenate(codes) if codes else np.zeros(0, dtype=np.int64)
    cat = {k: np.concatenate([p[k] for p in parts_list]) for k in ("raw", "fallback", "decision_time", "open_time", "label")}
    order = np.lexsort((sym, cat["decision_time"]))
    sym, cat = sym[order], {k: v[order] for k, v in cat.items()}
    if sym.size != s["symbol"].size or not (np.array_equal(cat["decision_time"], s["decision_time"]) and np.array_equal(sym, s["symbol"])):
        have = set(zip(sym.tolist(), cat["decision_time"].tolist(), strict=True))
        need = set(zip(s["symbol"].tolist(), s["decision_time"].tolist(), strict=True))
        lost = sorted(d for _, d in need - have)
        extra = sorted(d for _, d in have - need)
        raise PairingRefused(f"{where}: {len(need - have)} scored rows have no forecast (first decision {r2.iso(lost[0]) if lost else '-'}), "
                             f"{len(have - need)} forecasts are not scored rows (first decision {r2.iso(extra[0]) if extra else '-'}), "
                             f"{sym.size} forecasts for {s['symbol'].size} rows")  # fmt: skip
    if not np.array_equal(cat["open_time"], s["open_time"]) or not np.array_equal(cat["label"], s["label"]):
        i = int(np.flatnonzero((cat["open_time"] != s["open_time"]) | (cat["label"] != s["label"]))[0])
        raise PairingRefused(f"{where}: row {i} has another open time or label than Report n°2's scored row")
    return cat


def load_forecasts(parts: Parts, cfg: Report3Config, run: Report2Run) -> dict[str, Forecasts]:
    out = {}
    for arm in cfg.arms:
        raw, fallback = {}, None
        for seed in cfg.seeds:
            keys = [part_key(arm, seed, y) for y in cfg.years]
            missing = [k for k in keys if verified_part(parts, k, cfg) is None]
            if missing:
                raise PairingRefused(f"{arm} seed {seed}: no verified part {missing[0]}; run the fits stage")
            cat = pair_forecasts([read_forecast(parts, k) for k in keys], run, f"{arm} seed {seed}")
            raw[seed] = cat["raw"]
            if fallback is not None and not np.array_equal(fallback, cat["fallback"]):
                raise PairingRefused(f"{arm}: the seeds disagree on the rows whose window failed")
            fallback = cat["fallback"]
        out[arm] = Forecasts(raw, fallback)
    return out


def arm_probabilities(fc: Forecasts, floor: float) -> np.ndarray:
    """The arm's forecast: the mean of its seeds' probabilities before the floor, then the floor."""
    return ensemble([fc.raw[s] for s in sorted(fc.raw)], floor)


# --- The timing protocol ---------------------------------------------------------------------------


def timing_sample(usable: np.ndarray, cfg: Report3Config) -> np.ndarray:
    """Positions, among Report n°2's scored rows, of the timed rows: ``cfg.timing_rows`` drawn without replacement
    with ``cfg.timing_seed`` from the scored rows with a valid window, then sorted (panel order)."""
    pool = np.flatnonzero(usable)
    if pool.size < cfg.timing_rows:
        raise ValueError(f"{pool.size} scored rows with a valid window, the timing needs {cfg.timing_rows}")
    rng = np.random.default_rng(cfg.timing_seed)
    return np.sort(pool[rng.choice(pool.size, size=cfg.timing_rows, replace=False)])


def run_protocol(runners: Mapping[str, Callable[[int], None]], *, n_batches: int, warmups: int, repeats: int,
                 sync: Callable[[], None], clock: Callable[[], float] = time.perf_counter,
                 events: list[tuple[int, str, str, int]] | None = None) -> dict[str, list[float]]:  # fmt: skip
    """H13's protocol: ``repeats`` repeats cycling the arms in the order given (ABAB); in each, an arm runs
    ``warmups`` untimed batches, then each of its ``n_batches`` timed batches with ``sync()`` before and after.

    Returns, per arm, the seconds of each repeat (the sum over its timed batches). ``events`` gets
    ``(repeat, arm, "warmup" | "timed", batch)`` in order.
    """
    out: dict[str, list[float]] = {arm: [] for arm in runners}
    for r in range(repeats):
        for arm, run in runners.items():
            for w in range(warmups):
                run(w % n_batches)
                if events is not None:
                    events.append((r, arm, "warmup", w % n_batches))
            sync()
            total = 0.0
            for b in range(n_batches):
                sync()
                t0 = clock()
                run(b)
                sync()
                total += clock() - t0
                if events is not None:
                    events.append((r, arm, "timed", b))
            out[arm].append(total)
    return out


def per_decisions(seconds: Sequence[float], rows: int, per: int) -> list[float]:
    """Seconds per ``per`` decisions: each repeat's seconds over its ``rows`` timed rows, times ``per``."""
    return [s / rows * per for s in seconds]


def summary(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def load_fit(parts: Parts, key: str, cfg: Report3Config) -> tuple[dict[str, np.ndarray], Standardizer, FitConfig]:
    """A part's kept parameters, standardizer and fit config, the part verified first (:func:`verified_part`)."""
    if verified_part(parts, key, cfg) is None:
        raise PairingRefused(f"{key}: no finished part; run the fits stage")
    data = parts.get(f"{key}/fit.safetensors")
    if data is None:
        raise PairingRefused(f"{key}: no verified checkpoint")
    arrays, meta = checkpoint.from_bytes(data)
    state = {k.removeprefix("param."): v for k, v in arrays.items() if k.startswith("param.")}
    std = Standardizer(arrays["std.mean"], arrays["std.std"], int(meta["standardizer_rows"]))
    c = json.loads(meta["config"])
    fc = FitConfig(**{**c, "encoder": EncoderConfig(**c["encoder"]), "memory": MemoryConfig(**c["memory"]), "betas": tuple(c["betas"])})
    return state, std, fc


class EncoderTimer:
    """One arm's timed pass over the sample's batches: the encoder, the memory readout and the head, as
    :meth:`EncoderSystem.predict` runs them on a batch, with the memory's prefix sums built before."""

    def __init__(self, data: EncoderData, fc: FitConfig, state: Mapping[str, np.ndarray], std: Standardizer, rows: np.ndarray, cfg: Report3Config) -> None:
        self.device = torch.device(cfg.device)
        self.system = EncoderSystem(data, fc, device=self.device)
        model = EncoderModel(fc.encoder)
        load_state(model, state)
        self.model = model.to(self.device).eval()
        self.mem = fc.memory
        with torch.no_grad():
            self.bars = self.system.standardized_bars(std)
            pool_rows, pool_stamps = self.system.sorted_pool(np.flatnonzero(data.usable))
            as_of = self.system.readout_as_of(rows).astype(np.int64)
            lo, hi = self.system._entries_for(pool_stamps, as_of)
            e_rows, e_stamps = pool_rows[lo:hi], pool_stamps[lo:hi]
            ordered = np.sort(e_rows)
            _, z_e = self.system._encode_fixed(self.model, self.bars, ordered)
            z_e = z_e.to("cpu", torch.float64)[torch.from_numpy(np.searchsorted(ordered, e_rows))]
            y_e = torch.nn.functional.one_hot(torch.from_numpy(self.system.y[e_rows]), N_CLASSES).to(torch.float64)
            self.prefix = prefix_moments(z_e, y_e, causal=True)
            lo_q, hi_q = window_bounds(torch.from_numpy(e_stamps), torch.from_numpy(as_of), self.mem.window_ms)
        b = cfg.timing_batch
        self.ends = [torch.from_numpy(data.windows.end[rows[i : i + b]]).to(self.device) for i in range(0, rows.size, b)]
        self.lo = [lo_q[i : i + b] for i in range(0, rows.size, b)]
        self.hi = [hi_q[i : i + b] for i in range(0, rows.size, b)]
        self.ops: list[list[tuple[torch.Tensor, ...]]] | None = None
        self.keys: list[torch.Tensor] | None = None

    @torch.no_grad()
    def run(self, b: int) -> None:
        h, z = self.model.encode(gather_windows(self.bars, self.ends[b]))
        ro = readout(self.prefix, self.lo[b], self.hi[b], z.to("cpu", torch.float64), self.mem)
        torch.softmax(self.model.decide(h, ro.m.to(self.device, h.dtype)), dim=-1)

    @torch.no_grad()
    def prepare_readout(self) -> None:
        """Each batch's keys, encoded once and untimed, on the CPU in fp64 as the whole pass reads them."""
        self.keys = [self.model.encode(gather_windows(self.bars, ends))[1].to("cpu", torch.float64) for ends in self.ends]

    @torch.no_grad()
    def run_readout(self, b: int) -> None:
        """The memory readout alone on a batch's cached keys: the part of the whole pass that runs on the CPU.

        The whole pass also copies the keys z from the device to the host and the readout m back; those two
        copies belong to the whole pass but to neither this timing nor the attention operator's.
        """
        assert self.keys is not None
        readout(self.prefix, self.lo[b], self.hi[b], self.keys[b], self.mem)

    @torch.no_grad()
    def prepare_ops(self) -> None:
        """The Q, K and V of every layer for every batch, computed once and untimed, for the operator's own time."""
        enc = self.model.encoder
        self.ops = []
        for ends in self.ends:
            hidden = enc.embed(gather_windows(self.bars, ends)) + enc.position
            layers = []
            for block in enc.blocks:
                q, k, v = block.attn.project(block.ln1(hidden))
                lam = block.attn.lam() if enc.cfg.attention == "ols" else None
                layers.append((q.contiguous(), k.contiguous(), v.contiguous(), lam))
                hidden = block(hidden)
            self.ops.append(layers)

    @torch.no_grad()
    def run_op(self, b: int) -> None:
        assert self.ops is not None
        kind, order = self.model.cfg.attention, self.model.cfg.ols_order
        for q, k, v, lam in self.ops[b]:
            if kind == "ols":
                ols_attention(q, k, v, lam, order=order)
            elif kind == "softmax":
                softmax_attention(q, k, v)
            else:
                linear_attention(q, k, v)


def run_timing(data: EncoderData, run: Report2Run, scored: np.ndarray, cfg: Report3Config, parts: Parts, *,
               log: Callable[[str], None] = lambda s: None, driver: str | None = None) -> dict[str, Any]:  # fmt: skip
    """H13's timing on the device of ``cfg``; saved as ``parts/timing.json``."""
    device = torch.device(cfg.device)
    sample = timing_sample(data.usable[scored], cfg)
    rows = scored[sample]
    year, seed = cfg.years[-1], cfg.seeds[0]
    timers: dict[str, EncoderTimer] = {}
    for arm in cfg.arms:
        state, std, fc = load_fit(parts, part_key(arm, seed, year), cfg)
        timers[arm] = EncoderTimer(data, fc, state, std, rows, cfg)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    peaks = {}  # each arm's peak over one untimed pass of the sample, before the protocol
    for arm, t in timers.items():
        _peak_reset(device)
        for b in range(cfg.timing_batches):
            t.run(b)
        sync()
        peaks[arm] = _peak_gpu(device)
    log(f"timing {cfg.timing_repeats} repeats of {cfg.timing_batches} batches of {cfg.timing_batch} rows, arms cycled {', '.join(cfg.arms)}")
    events: list[tuple[int, str, str, int]] = []
    whole = run_protocol({a: t.run for a, t in timers.items()}, n_batches=cfg.timing_batches, warmups=cfg.timing_warmups,
                         repeats=cfg.timing_repeats, sync=sync, events=events)  # fmt: skip
    for t in timers.values():
        t.prepare_ops()
    op = run_protocol({a: t.run_op for a, t in timers.items()}, n_batches=cfg.timing_batches, warmups=cfg.timing_warmups,
                      repeats=cfg.timing_repeats, sync=sync)  # fmt: skip
    for t in timers.values():
        t.prepare_readout()
    rd = run_protocol({a: t.run_readout for a, t in timers.items()}, n_batches=cfg.timing_batches, warmups=cfg.timing_warmups,
                      repeats=cfg.timing_repeats, sync=sync)  # fmt: skip
    n = cfg.timing_rows
    arms = {}
    for arm in cfg.arms:
        w, o, r = (per_decisions(x[arm], n, cfg.per_decisions) for x in (whole, op, rd))
        arms[arm] = {"seconds": whole[arm], "operator_seconds": op[arm], "readout_seconds": rd[arm], "per_decisions_s": w,
                     "operator_per_decisions_s": o, "readout_per_decisions_s": r, "whole": summary(w), "operator": summary(o),
                     "readout": summary(r), "peak_gpu_bytes": peaks[arm], "fit": part_key(arm, seed, year)}  # fmt: skip
    info = build_info(device)
    out = {
        "device": str(device), "gpu": info.get("gpu"), "driver": driver, "cuda": info.get("cuda_build"), "torch": info.get("torch"),
        "settings": info.get("settings"), "rows": n, "batch": cfg.timing_batch, "batches": cfg.timing_batches, "warmups": cfg.timing_warmups,
        "repeats": cfg.timing_repeats, "per_decisions": cfg.per_decisions, "seed": cfg.timing_seed, "order": list(cfg.arms),
        "sample_sha256": _sha(np.ascontiguousarray(rows, dtype=np.int64).tobytes()), "arms": arms,
        "first_repeat_events": [list(e) for e in events if e[0] == 0],
    }  # fmt: skip
    parts.put_json(TIMING_FILE, out)
    log("timing per 10^4 decisions (median; whole pass, operator, readout): "
        + ", ".join(f"{a} {arms[a]['whole']['median']:.4f}, {arms[a]['operator']['median']:.4f}, {arms[a]['readout']['median']:.4f} s" for a in cfg.arms))  # fmt: skip
    return out


# --- The Matrix arms' readouts on CPU -----------------------------------------------------------------


def matrix_choices(run: Report2Run) -> dict[str, Any]:
    """The fit-time choices of Report n°2's last test month (and matrix-cusum's last fit), as Report n°2 records them."""
    last = run.report["fits"][-1]
    cus = run.report["cusum"]["fits"][-1]
    return {"month": last["month"], "select": last["select"], "rho_1y": last["rho_1y"], "tau_factor": last["tau_factor"], "k": last["k"],
            "stack_rho": last["stack_rho"], "alpha": last["alpha"], "cusum": {"window": cus["window"], "rho": cus["rho"], "fit": cus["fit"]}}  # fmt: skip


def _window_ms(days: int | None) -> int | None:
    return None if days is None else int(days) * 86_400_000


def matrix_runners(system: MatrixSystem, rows: np.ndarray, ch: dict[str, Any], batch: int) -> tuple[dict[str, Callable[[int], None]], dict[str, float]]:
    """Each Matrix arm's readout of a batch with the choices ``ch`` (and the mean entries of each window read, for the FLOPs).

    ``matrix-adaptive`` is timed as one ridge readout at matrix-select's window and rho: its rho is a lookup
    per decision, a constant whose trailing means are computed once per month and amortized over its
    decisions, so the per-decision cost is that readout.
    """
    d, mc = system.data, system.cfg
    year, month = (int(x) for x in ch["month"].split("-"))
    st = system.month_stats(month_start_ms(year, month))
    var, sigma = st.var, np.sqrt(st.var)
    xs = [np.ascontiguousarray(d.x[rows[i : i + batch]]) for i in range(0, rows.size, batch)]
    at = [np.ascontiguousarray(d.decision_ms[rows[i : i + batch]], dtype=np.int64) for i in range(0, rows.size, batch)]
    floor, threads = mc.prob_floor, mc.threads
    m = system.matrix

    def ridge(window: int | None, rho: float) -> Callable[[int], None]:
        def go(b: int) -> None:
            out = m.ridge_batch(xs[b], at[b], [_window_ms(window)], np.array([rho]), var, threads=threads)
            prob_map(out["forecast"][:, 0, 0, :N_CLASSES], floor)

        return go

    def stack(b: int) -> None:
        parts = []
        for w, rho in zip(mc.windows_days, ch["stack_rho"], strict=True):
            out = m.ridge_batch(xs[b], at[b], [_window_ms(w)], np.array([rho]), var, threads=threads)
            parts.append(prob_map(out["forecast"][:, 0, 0, :N_CLASSES], floor))
        sum(a * p for a, p in zip(ch["alpha"], parts, strict=True))

    def kernel(taus: list[float], ks: list[int]) -> Callable[[int], None]:
        def go(b: int) -> None:
            out = m.kernel_batch(xs[b], at[b], mc.kernel_window_days * 86_400_000, sigma, np.array(taus, dtype=np.float64), np.array(ks, dtype=np.int64),
                                 threads=threads)  # fmt: skip
            prob_map((out["softmax"] if taus else out["knn"])[:, 0, :N_CLASSES], floor)

        return go

    def clim(b: int) -> None:
        np.tile(st.clim, (xs[b].shape[0], 1))

    sel_w, sel_rho = ch["select"]["window"], ch["select"]["rho"]
    tau = ch["tau_factor"] * math.sqrt(xs[0].shape[1])
    runners = {
        CLIMATOLOGY: clim,
        MATRIX_SELECT: ridge(sel_w, sel_rho),
        "matrix-1y": ridge(mc.kernel_window_days, ch["rho_1y"]),
        "softmax-1y": kernel([tau], []),
        "knn-1y": kernel([], [int(ch["k"])]),
        "matrix-adaptive": ridge(sel_w, sel_rho),  # its window; the rho of a decision is a lookup, timed at the selected one
        "matrix-stack": stack,
        "matrix-cusum": ridge(ch["cusum"]["window"], ch["cusum"]["rho"]),
    }
    counts = {}
    for days in {*mc.windows_days, mc.kernel_window_days, ch["cusum"]["window"]}:
        n = [m.window_range(int(t), _window_ms(days)) for t in np.concatenate(at)]
        counts["all" if days is None else str(days)] = float(np.mean([e - s for s, e in n]))
    return runners, counts


def matrix_memory_bytes(n: int, p: int = P, q: int = Q, block_log2: int = 6) -> dict[str, int]:
    """The analytic footprint of :class:`olsprime_rs.Matrix` holding ``n`` entries (``crates/ols-core/src/matrix.rs``).

    Entries: ``8 (p + q + 1) n`` bytes (key, outcome, stamp). Two trees of block statistics, raw and
    centred, each of ``p + p^2 + q + qp + q^2`` float64 per node, over ``ceil(n / 2^b)`` blocks and
    every level above (``ceil`` halvings to 1). The struct's own fields and the allocator's slack are
    not counted.

    Right within about 5 %, not exactly: the raw nodes (``SuffStats``) hold ``2q`` floats for the outcome
    sums, not ``q + q^2``; the vectors' heap headers are not counted; and a partial last block is counted
    as a node although the crate scans it instead of storing it.
    """
    blocks, nodes = -(-n // (1 << block_log2)), 0
    while blocks > 0:
        nodes += blocks
        blocks = 0 if blocks == 1 else -(-blocks // 2)
    entries = 8 * (p + q + 1) * n
    trees = 2 * nodes * 8 * (p + p * p + q + q * p + q * q)
    return {"entries": entries, "trees": trees, "total": entries + trees, "nodes_per_tree": nodes,
            "bytes_per_entry": (entries + trees) // n if n else 0}  # fmt: skip


def run_matrix_timing(panel: pl.DataFrame, data: EncoderData, run: Report2Run, scored: np.ndarray, cfg: Report3Config, parts: Parts, *,
                      log: Callable[[str], None] = lambda s: None, cpu_name: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The eight Matrix arms' readouts on the CPU, on the timing sample; saved as ``parts/matrix-timing.json``."""
    import olsprime_rs

    sample = timing_sample(data.usable[scored], cfg)
    rows = scored[sample]
    t = time.perf_counter()
    system = MatrixSystem(panel, matrix_config(run.config))
    build_s = time.perf_counter() - t
    ch = matrix_choices(run)
    runners, counts = matrix_runners(system, rows, ch, cfg.timing_batch)
    runners = {a: runners[a] for a in MATRIX_ARMS}
    log(f"matrix timing: memory of {system.entry_rows.size} entries built in {build_s:.1f} s; {cfg.timing_repeats} repeats")
    secs = run_protocol(runners, n_batches=cfg.timing_batches, warmups=cfg.timing_warmups, repeats=cfg.timing_repeats, sync=lambda: None)
    arms = {}
    for arm in MATRIX_ARMS:
        w = per_decisions(secs[arm], cfg.timing_rows, cfg.per_decisions)
        arms[arm] = {"seconds": secs[arm], "per_decisions_s": w, "whole": summary(w)}
    out = {"machine": cpu_name or platform.processor() or None, "platform": platform.platform(), "python": platform.python_version(),
           "olsprime_rs": olsprime_rs.__version__, "threads": system.cfg.threads, "cpu_count": os.cpu_count(), "rows": cfg.timing_rows,
           "batch": cfg.timing_batch, "batches": cfg.timing_batches, "warmups": cfg.timing_warmups, "repeats": cfg.timing_repeats,
           "per_decisions": cfg.per_decisions, "seed": cfg.timing_seed, "choices": ch, "window_entries_mean": counts,
           "memory_entries": int(system.entry_rows.size), "block_log2": system.cfg.block_log2,
           "memory_bytes": matrix_memory_bytes(int(system.entry_rows.size), block_log2=system.cfg.block_log2),
           "process_peak_bytes": process_peak_bytes(),
           "sample_sha256": _sha(np.ascontiguousarray(rows, dtype=np.int64).tobytes()), "arms": arms}  # fmt: skip
    parts.put_json(MATRIX_TIMING_FILE, out)
    log("matrix timing per 10^4 decisions (median): " + ", ".join(f"{a} {arms[a]['whole']['median']:.4f} s" for a in MATRIX_ARMS))
    return out


# --- Analytic FLOPs --------------------------------------------------------------------------------


GELU_FLOPS = 8  # tanh or erf form, counted as 8 floating-point operations per element
LAYERNORM_FLOPS = 8  # per element: mean, centre, variance (2), normalize (2), affine (2)


def attention_flops(kind: str, t: int, e: int) -> int:
    """FLOPs of one head's operator on Q, K, V of shape (T, e); a multiply-add counts 2, an exp or a division 1.

    - softmax: ``Q K^T`` 2 T^2 e, the scale T^2, the softmax 4 T^2 (max, subtract and exp, sum, divide), ``A V`` 2 T^2 e:
      ``4 T^2 e + 5 T^2``.
    - linear: phi(Q), phi(K) 2 T e each (elu + 1), ``phi(K)^T V`` 2 T e^2, its column sum T e, ``phi(Q) (phi(K)^T V)``
      2 T e^2, the normaliser 2 T e, the division T e: ``4 T e^2 + 8 T e``.
    - ols (``assoc``, the registered order), with e' = e + 1 for the constant coordinate: the RMS scaling of Q and K
      2 (3 T e + T), ``K~^T K~`` 2 T e'^2, Lambda e, the Cholesky factor e'^3 / 3, ``K~^T V`` 2 T e' e, the two
      triangular solves 2 e'^2 e, ``Q~ M`` 2 T e' e: ``2 T e'^2 + 4 T e' e + e'^3 / 3 + 2 e'^2 e + 6 T e + 2 T + e``.
    """
    if kind == "softmax":
        return 4 * t * t * e + 5 * t * t
    if kind == "linear":
        return 4 * t * e * e + 8 * t * e
    if kind == "ols":
        f = e + 1
        return 2 * t * f * f + 4 * t * f * e + round(f**3 / 3) + 2 * f * f * e + 6 * t * e + 2 * t + e
    raise ValueError(kind)


def memory_readout_flops(p: int, q: int) -> int:
    """FLOPs of one memory readout from built prefix sums, keys in R^p, values in R^q.

    The window's raw moments by difference ``1 + p + p^2 + q + q p``; the means p + q; ``M_zz`` 3 p^2 and its
    symmetrization 2 p^2; ``M_yz`` 3 q p; Lambda 3 p (``rho diag + jitter n``) and its addition p; the Cholesky
    factor p^3 / 3; ``beta`` by two triangular solves with q right-hand sides 2 p^2 q; ``m = ybar + (z - zbar) beta``
    p + 2 p q + q.
    """
    return (1 + p + p * p + q + q * p) + (p + q) + 5 * p * p + 3 * q * p + 4 * p + round(p**3 / 3) + 2 * p * p * q + (p + 2 * p * q + q)


def encoder_flops(cfg: EncoderConfig, mem: MemoryConfig = MemoryConfig()) -> dict[str, int]:
    """FLOPs of one decision of an encoder arm, by part (a multiply-add counts 2).

    T = window, d = d_model, H heads of e = d_head, L layers, m = d_mlp, F = quantities, k = key_dim,
    K = classes, g = head_hidden. Every position is computed in every layer, as the implementation does.

    - input: ``T d (2 F + 2)`` (the linear map, its bias and the position embedding);
    - per layer: two LayerNorms ``2 * 8 T d``; Q, K, V ``2 T d 3d + 3 T d``; the operator ``H * attention_flops``;
      the output projection ``2 T d^2 + T d``; the MLP ``2 T d m + T m + 8 T m + 2 T m d + T d``; two residuals ``2 T d``;
    - the final LayerNorm ``8 T d``; the key ``2 d k``; the memory readout (:func:`memory_readout_flops`, p = k, q = K);
    - the head ``2 (d + K) g + g + 8 g + 2 g K + K``, its softmax ``4 K`` and the floor ``3 K``.
    """
    t, d, h, e, layers, m = cfg.window, cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_layers, cfg.d_mlp
    f, k, kk, g = cfg.n_quantities, cfg.key_dim, cfg.n_classes, cfg.head_hidden
    op = h * attention_flops(cfg.attention, t, e)
    layer_rest = 2 * LAYERNORM_FLOPS * t * d + (2 * t * d * 3 * d + 3 * t * d) + (2 * t * d * d + t * d) + (
        2 * t * d * m + t * m + GELU_FLOPS * t * m + 2 * t * m * d + t * d) + 2 * t * d  # fmt: skip
    out = {
        "input": t * d * (2 * f + 2),
        "attention_operator": layers * op,
        "layers_other": layers * layer_rest,
        "final_norm_and_key": LAYERNORM_FLOPS * t * d + 2 * d * k,
        "memory_readout": memory_readout_flops(k, kk),
        "head": 2 * (d + kk) * g + g + GELU_FLOPS * g + 2 * g * kk + kk + 4 * kk + 3 * kk,
    }
    out["total"] = sum(out.values())
    return out


def ridge_readout_flops(n_w: float, p: int = 15, q: int = 4, block: int = 64) -> float:
    """Dominant FLOPs of one centred ridge readout of Matrix over a window of ``n_w`` entries (``olsprime_rs``).

    The window's statistics come from about ``log2(n_w / B)`` tree nodes, each merged by the update of Chan,
    Golub and LeVeque, ``3 (p^2 + q p + q^2) + 4 (p + q)``, and from the partial blocks at its two ends,
    about B entries scanned by the corrected two-pass, ``2 (p + q) + p (p + 1) + 2 q p + q (q + 1)`` each.
    Then Lambda and A 2 p, the Cholesky factor p^3 / 3, beta 2 p^2 q, the forecast p + 2 p q + q, the floor 9.
    p = 15 features, q = 4 outputs (three classes and ``fwd_return``), B = 2^block_log2.
    """
    merges = math.log2(max(n_w / block, 1.0))
    merge = 3 * (p * p + q * p + q * q) + 4 * (p + q)
    scan = 2 * (p + q) + p * (p + 1) + 2 * q * p + q * (q + 1)
    return merges * merge + min(block, n_w) * scan + 2 * p + p**3 / 3 + 2 * p * p * q + p + 2 * p * q + q + 9


def matrix_flops(arm: str, n_w: Mapping[str, float], ch: Mapping[str, Any], mc: MatrixConfig) -> float:
    """Dominant FLOPs of one decision of a Matrix arm, with the fit-time choices ``ch`` and the mean window sizes ``n_w``.

    - ``climatology-panel``: the stored distribution, 0;
    - ``matrix-select``, ``matrix-1y``, ``matrix-cusum``: one ridge readout (:func:`ridge_readout_flops`) at the arm's
      window; ``matrix-adaptive``: the same at its window, plus the rho lookup ``3 R`` over the R = 6 trailing means;
    - ``matrix-stack``: three ridge readouts, one per window, and the mix ``2 * 3 * 3``;
    - ``softmax-1y``: per entry of the 365-day window, the scaled difference and its square ``4 p``, the exponent and
      exp ``3``, the weighted sums ``2 q + 1``; then q divisions: ``n_w (4 p + 2 q + 4) + q + 9``;
    - ``knn-1y``: the distances ``4 p n_w``, a linear-time selection ``n_w``, the mean of the k labels ``k q + q``, the floor 9.
    """
    p, q, b = 15, 4, 2**mc.block_log2

    def nw(days: int | None) -> float:
        return float(n_w["all" if days is None else str(days)])

    if arm == CLIMATOLOGY:
        return 0.0
    if arm in (MATRIX_SELECT, "matrix-adaptive"):
        extra = 3 * len(mc.rhos) if arm == "matrix-adaptive" else 0
        return ridge_readout_flops(nw(ch["select"]["window"]), p, q, b) + extra
    if arm == "matrix-1y":
        return ridge_readout_flops(nw(mc.kernel_window_days), p, q, b)
    if arm == "matrix-cusum":
        return ridge_readout_flops(nw(ch["cusum"]["window"]), p, q, b)
    if arm == "matrix-stack":
        return sum(ridge_readout_flops(nw(w), p, q, b) for w in mc.windows_days) + 18
    n = nw(mc.kernel_window_days)
    if arm == "softmax-1y":
        return n * (4 * p + 2 * q + 4) + q + 9
    if arm == "knn-1y":
        return n * (4 * p + 1) + int(ch["k"]) * q + q + 9
    raise ValueError(arm)


# --- Scoring -----------------------------------------------------------------------------------------


def _compare(ll: Mapping[str, np.ndarray], a: str, b: str, clim: float, rcfg: r2.Report2Config) -> dict[str, Any]:
    d = ll[a] - ll[b]
    if np.all(d == d[0]):
        return {"a": a, "b": b, "mean_diff": float(d.mean()), "hours": int(d.size), **r2.identical_hours(d), "undefined": "the difference is the same at every hour",
                "passes": False, "upper_95_one_sided": None, "bootstrap": None}  # fmt: skip
    return r2.compare_arms(dict(ll), a, b, clim, rcfg)


def verdicts(bss: Mapping[str, dict[str, Any]], comparisons: Mapping[str, dict[str, Any]], margin: float, timing: Mapping[str, Any] | None,
             cfg: Report3Config) -> dict[str, Any]:  # fmt: skip
    """H4, H5 and H13 by their registered rules.

    - H4: testable only if ``enc-ols`` or ``enc-softmax`` has a BSS against climatology whose 95 % interval lies above 0.
      Then it holds if the one-sided 95 % upper bound of mean(ll_ols - ll_softmax) (mean + t_0.95(T - 1) x the
      HLN-corrected DM standard error) and the stationary bootstrap's 95th percentile both lie below the margin.
    - H5: ``enc-ols`` below ``matrix-select``, the comparison passing (DM p < 0.05, the bootstrap interval below 0).
    - H13: not testable when H4 is not; does not hold when H4 does not; not run without the timing; else holds if the
      median time per 10^4 decisions of ``enc-ols`` over the repeats is at most 0.9 x ``enc-softmax``'s.
    """
    skill = {a: bool(bss[a]["estimate"] > 0 and bss[a]["low"] > 0) for a in ("enc-ols", "enc-softmax")}
    testable = any(skill.values())
    h4c = comparisons["H4"]
    upper = h4c.get("upper_95_one_sided")
    q95 = h4c["bootstrap"]["q95"] if h4c.get("bootstrap") else None
    bounds = upper is not None and q95 is not None and r2.h12_rule_met(upper, q95, margin)
    h4_status = "not testable" if not testable else ("holds" if bounds else "does not hold")
    h4 = {"status": h4_status, "holds": h4_status == "holds", "testable": testable, "skill_above_0": skill, "margin": margin,
          "upper_95_one_sided": upper, "bootstrap_q95": q95, "bounds_below_margin": bool(bounds),
          "rule": "testable if enc-ols or enc-softmax has a BSS interval above 0; then the one-sided 95 % upper bound and the bootstrap's "
                  "95th percentile of mean(ll_ols - ll_softmax) both below 0.005 x climatology-panel's log-loss"}  # fmt: skip
    h5c = comparisons["H5"]
    h5 = {"status": "holds" if h5c["passes"] else "does not hold", "holds": bool(h5c["passes"]),
          "rule": "enc-ols below matrix-select: DM p < 0.05 and the bootstrap interval below 0",
          "limit": "the encoder refits yearly and matrix-select monthly: H5 compares two systems, not two attentions at equal cadence"}  # fmt: skip
    ratio = paired = None
    if timing is not None:
        ratio = timing["arms"]["enc-ols"]["whole"]["median"] / timing["arms"]["enc-softmax"]["whole"]["median"]
        # a description: each repeat's enc-ols over the same repeat's enc-softmax (ABAB), their median
        paired = float(np.median(np.asarray(timing["arms"]["enc-ols"]["seconds"]) / np.asarray(timing["arms"]["enc-softmax"]["seconds"])))
    if h4_status == "not testable":
        h13_status = "not testable"
    elif h4_status == "does not hold":
        h13_status = "does not hold"
    elif ratio is None:
        h13_status = "not run"
    elif not timing_registered(timing):
        h13_status = "not run under the registered conditions"
    else:
        h13_status = "holds" if ratio <= cfg.h13_ratio else "does not hold"
    h13 = {"status": h13_status, "holds": h13_status == "holds", "median_ratio": ratio, "paired_median_ratio": paired, "max_ratio": cfg.h13_ratio,
           "registered_conditions": None if timing is None else timing_registered(timing),
           "rule": "H4 holds and the median wall time per 10^4 decisions of enc-ols over the repeats is at most 0.9 x enc-softmax's, "
                   "timed on the GPU (the RTX 3060)"}  # fmt: skip
    return {"H4": h4, "H5": h5, "H13": h13}


def registered_run(cfg: Report3Config, run: Report2Run, devices: Sequence[str], builds: Sequence[Mapping[str, Any]],
                   timing: Mapping[str, Any] | None) -> bool:  # fmt: skip
    """The registered run: the registered configuration, paired with a registered Report n°2, every fit on CUDA in one
    build, and the timing (if any) on CUDA naming its GPU."""
    return bool(cfg.registered and run.report.get("registered", False) and list(devices) == ["cuda"] and len(builds) == 1
                and timing_registered(timing))  # fmt: skip


def timing_registered(timing: Mapping[str, Any] | None) -> bool:
    """H13's registered conditions: no timing yet, or a timing on CUDA that names its GPU."""
    return timing is None or bool(timing.get("device") == "cuda" and timing.get("gpu"))


def score_hourly(ll: Mapping[str, np.ndarray], br: Mapping[str, np.ndarray], rcfg: r2.Report2Config, cfg: Report3Config,
                 timing: Mapping[str, Any] | None) -> dict[str, Any]:  # fmt: skip
    """The comparisons, the encoder arms' log-loss, Brier and BSS intervals, and the verdicts, from hourly series
    (``climatology-panel`` and ``matrix-select`` among them)."""
    clim = float(ll[CLIMATOLOGY].mean())
    margin = cfg.h4_margin * clim
    arms = {}
    for arm in cfg.arms:
        arms[arm] = {"log_loss": r2.mean_interval(ll[arm], rcfg, "ll", arm), "brier": r2.mean_interval(br[arm], rcfg, "brier", arm),
                     "bss": r2.bss_interval(br[arm], br[CLIMATOLOGY], rcfg, "bss", arm)}  # fmt: skip
    comparisons = {name: dict(_compare(ll, a, b, clim, rcfg), name=name, kind=kind) for name, a, b, kind in COMPARISONS}
    return {"climatology_log_loss": clim, "h4_margin": margin, "arms": arms, "comparisons": comparisons,
            "verdicts": verdicts({a: arms[a]["bss"] for a in cfg.arms}, comparisons, margin, timing, cfg)}  # fmt: skip


def _bounds(series: Mapping[str, np.ndarray], a: str, b: str, clim: float, margin: float, rcfg: r2.Report2Config) -> dict[str, Any]:
    c = _compare(series, a, b, clim, rcfg)
    upper = c.get("upper_95_one_sided")
    q95 = c["bootstrap"]["q95"] if c.get("bootstrap") else None
    met = upper is not None and q95 is not None and r2.h12_rule_met(upper, q95, margin)
    return {"a": a, "b": b, "mean_diff": c["mean_diff"], "upper_95_one_sided": upper, "bootstrap_q95": q95, "bounds_below_margin": bool(met)}


def h4_descriptions(fcs: Mapping[str, Forecasts], sc: r2.Scored, ll: Mapping[str, np.ndarray], floor: float, margin: float,
                    rcfg: r2.Report2Config) -> dict[str, Any]:  # fmt: skip
    """H4's two bounds beside its verdict, descriptions only, from the saved forecasts (no new fit).

    - ``seed_pairs``: enc-ols seed i against enc-softmax seed i, each seed's own floored probabilities;
    - ``reversed``: enc-softmax non-inferior to enc-ols, on the ensembles (difference softmax - ols).

    Same margin, same statistics; the verdict stays the registered one, on the ensembles.
    """
    clim = float(ll[CLIMATOLOGY].mean())
    seeds = {}
    for s in sorted(set(fcs["enc-ols"].raw) & set(fcs["enc-softmax"].raw)):
        series = {}
        for arm in ("enc-ols", "enc-softmax"):
            p = prob_map(fcs[arm].raw[s], floor)
            series[f"{arm} seed {s}"] = r2.hourly(stats.per_row_losses(p, sc.label).log_loss, sc)
        seeds[str(s)] = _bounds(series, f"enc-ols seed {s}", f"enc-softmax seed {s}", clim, margin, rcfg)
    return {"seed_pairs": seeds, "reversed": _bounds(ll, "enc-softmax", "enc-ols", clim, margin, rcfg),
            "note": "descriptions beside H4's registered verdict, which reads the ensembles in the direction ols - softmax"}  # fmt: skip


def per_seed(fc: Forecasts, sc: r2.Scored, floor: float) -> dict[str, Any]:
    """Each seed's mean hourly log-loss (its own floored probabilities) and their range: a description."""
    vals = {}
    for seed in sorted(fc.raw):
        p = prob_map(fc.raw[seed], floor)
        vals[str(seed)] = float(r2.hourly(stats.per_row_losses(p, sc.label).log_loss, sc).mean())
    v = list(vals.values())
    return {"log_loss": vals, "range": float(max(v) - min(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}


def cost_table(cfg: Report3Config, fits: Sequence[dict[str, Any]], timings: Mapping[str, dict[str, Any]], timing: Mapping[str, Any] | None,
               matrix_timing: Mapping[str, Any] | None, mc: MatrixConfig) -> dict[str, Any]:  # fmt: skip
    """The five descriptive columns for the encoder arms (the RTX 3060) and the eight Matrix arms (the CPU), never mixed."""
    rows = {}
    for arm in cfg.arms:
        enc = EncoderConfig(arm=arm, **dict(cfg.encoder))
        n_params = sorted({f["n_parameters"] for f in fits if f["arm"] == arm})
        mine = [timings[k] for k in timings if k.split("/")[1] == arm]
        gpu = sorted({t.get("gpu") or t.get("device") for t in mine})
        rows[arm] = {
            "parameters": n_params[0] if len(n_params) == 1 else n_params,
            "flops": encoder_flops(enc),
            "gpu_per_decisions_s": timing["arms"][arm]["whole"]["median"] if timing else None,
            "gpu_operator_per_decisions_s": timing["arms"][arm]["operator"]["median"] if timing else None,
            "gpu_peak_bytes": timing["arms"][arm]["peak_gpu_bytes"] if timing else None,
            "gpu_peak_bytes_training": max((t["peak_gpu_bytes_training"] or 0 for t in mine), default=None) or None,
            "training_hours": sum(t["seconds"] for t in mine) / 3600.0,
            "training_device": gpu,
            "fits_timed": len(mine),
        }
    for arm in MATRIX_ARMS:
        mt = matrix_timing
        rows[arm] = {
            "parameters": 0,
            "flops": {"total": matrix_flops(arm, mt["window_entries_mean"], mt["choices"], mc)} if mt else None,
            "cpu_per_decisions_s": mt["arms"][arm]["whole"]["median"] if mt else None,
            "cpu_peak_bytes": mt["process_peak_bytes"] if mt else None,  # the whole process: panel, windows, torch, Matrix
            "memory_bytes": mt["memory_bytes"]["total"] if mt and arm != CLIMATOLOGY else None,  # the memory's own, analytic
            "training_hours": None,
        }
    return {"arms": rows, "gpu": timing.get("gpu") if timing else None, "gpu_device": timing.get("device") if timing else None,
            "cpu": (matrix_timing or {}).get("machine"), "cpu_platform": (matrix_timing or {}).get("platform"),
            "olsprime_rs": (matrix_timing or {}).get("olsprime_rs")}  # fmt: skip


def series_out(run: Report2Run, arms: Sequence[str], probs: Mapping[str, np.ndarray], ll: Mapping[str, np.ndarray], br: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The encoder arms' ``series.npz``, keys and dtypes of Report n°2's (:data:`olsprime.bench.report2.SERIES_KEYS`)."""
    s = run.series
    out = {"arms": np.array(arms), "hours": s["hours"], "count": s["count"], "log_loss": np.stack([ll[a] for a in arms]),
           "brier": np.stack([br[a] for a in arms]), "probs": np.stack([probs[a] for a in arms]), "label": s["label"],
           "fwd_return": s["fwd_return"], "symbol": s["symbol"], "symbols": s["symbols"], "open_time": s["open_time"],
           "decision_time": s["decision_time"], "label_known_at": s["label_known_at"], "hour": s["hour"]}  # fmt: skip
    return {k: np.ascontiguousarray(out[k], dtype=out[k].dtype if kind == "<U" else np.dtype(kind)) for k, kind, _ in r2.SERIES_KEYS}


def score(run: Report2Run, cfg: Report3Config, parts: Parts) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Every number of Report n°3 from the saved forecasts and Report n°2's series, and the arrays of its ``series.npz``."""
    rcfg = rule_config(run.config)
    s = run.series
    hours, count = s["hours"], s["count"]
    sc = r2.Scored(np.arange(s["label"].size), s["label"], s["hour"], hours, count)
    if not np.array_equal(np.bincount(sc.hour, minlength=hours.size), count):
        raise PairingRefused("Report n°2's hour index does not give its counts")
    fcs = load_forecasts(parts, cfg, run)  # every part verified, so every fit's config is fit_config's
    floors = {fit_config(cfg, a, sd, y).prob_floor for a in cfg.arms for sd in cfg.seeds for y in cfg.years}
    if len(floors) != 1:
        raise PairingRefused(f"the fits use several probability floors: {sorted(floors)}")
    floor = floors.pop()
    probs = {arm: arm_probabilities(fcs[arm], floor) for arm in cfg.arms}
    ll = {CLIMATOLOGY: run.ll(CLIMATOLOGY), MATRIX_SELECT: run.ll(MATRIX_SELECT)}
    br = {CLIMATOLOGY: run.brier(CLIMATOLOGY), MATRIX_SELECT: run.brier(MATRIX_SELECT)}
    for arm in cfg.arms:
        losses = stats.per_row_losses(probs[arm], sc.label)
        ll[arm], br[arm] = r2.hourly(losses.log_loss, sc), r2.hourly(losses.brier, sc)
    recorded = run.report["arms"][CLIMATOLOGY]["log_loss"]["estimate"]
    if float(ll[CLIMATOLOGY].mean()) != recorded:
        raise PairingRefused(f"climatology-panel's mean hourly log-loss {ll[CLIMATOLOGY].mean()!r} is not Report n°2's {recorded!r}")
    timing = parts.get_json(TIMING_FILE)  # a file that fails its digest is refused, never read as "not run"
    matrix_timing = parts.get_json(MATRIX_TIMING_FILE)
    if timing is not None and matrix_timing is not None and timing["sample_sha256"] != matrix_timing["sample_sha256"]:
        raise PairingRefused(f"{TIMING_FILE} and {MATRIX_TIMING_FILE} timed different samples "
                             f"({timing['sample_sha256'][:16]} and {matrix_timing['sample_sha256'][:16]})")  # fmt: skip
    out = score_hourly(ll, br, rcfg, cfg, timing)
    out["h4_descriptions"] = h4_descriptions(fcs, sc, ll, floor, out["h4_margin"], rcfg)
    for arm in cfg.arms:
        out["arms"][arm]["ece"] = r2.ece_interval(probs[arm], sc.label, sc, br[arm], rcfg, "ece", arm)
        out["arms"][arm]["fallbacks"] = int(fcs[arm].fallback.sum())
        out["arms"][arm]["seeds"] = per_seed(fcs[arm], sc, floor)
    out["reference_arms"] = {a: {k: run.report["arms"][a][k] for k in ("log_loss", "brier", "bss", "ece")} for a in (CLIMATOLOGY, MATRIX_SELECT)}
    fits = [verified_part(parts, part_key(a, sd, y), cfg) for a in cfg.arms for sd in cfg.seeds for y in cfg.years]
    timings = {part_key(a, sd, y): parts.get_json(f"{part_key(a, sd, y)}/timing.json") or {} for a in cfg.arms for sd in cfg.seeds for y in cfg.years}
    timings = {k: v for k, v in timings.items() if v}
    out["fits"] = [{k: f[k] for k in ("arm", "seed", "year", "fit_sha256", "forecast_sha256", "kept_epoch", "slice_losses", "epoch_digests",
                                      "steps", "n_parameters", "training_rows", "slice_rows", "climatology_rows", "forecast_rows", "fallbacks")}
                   for f in fits if f is not None]  # fmt: skip
    out["timing"] = None if timing is None else {k: v for k, v in timing.items() if k not in ("finished_at",)}
    out["matrix_timing"] = matrix_timing
    out["cost_table"] = cost_table(cfg, [f for f in fits if f is not None], timings, timing, matrix_timing, matrix_config(run.config))
    out["scored"] = {"rows": int(sc.label.size), "hours": int(hours.size), "first_hour": r2.iso(int(hours[0])), "last_hour": r2.iso(int(hours[-1]))}
    out["fallbacks"] = {arm: out["arms"][arm]["fallbacks"] for arm in cfg.arms}
    out["variants"] = {"before": cfg.variants_before, "this_report": len(cfg.arms), "total": cfg.variants_before + len(cfg.arms)}
    out["fits_devices"] = sorted({str(f.get("device")) for f in fits if f is not None})
    out["fits_builds"] = fits_builds(parts, cfg)
    out["registered"] = registered_run(cfg, run, out["fits_devices"], out["fits_builds"], timing)
    out["report2"] = {"series_sha256": run.sha256, "report2_json_sha256": run.report_sha256, "panel_fingerprint": run.fingerprint,
                      "registered": bool(run.report.get("registered", False)), "git_commit": run.config.get("provenance", {}).get("git_commit")}  # fmt: skip
    return out, series_out(run, cfg.arms, probs, ll, br)


# --- The report ------------------------------------------------------------------------------------------


def _ci(ci: Mapping[str, Any] | None, digits: int = 4) -> str:
    return r2._ci(ci, digits)


def _num(x: float | None, fmt: str = ".4f") -> str:
    return "-" if x is None else format(x, fmt)


def _bytes(x: int | None) -> str:
    return "-" if not x else f"{x / 2**30:.2f} GiB" if x >= 2**30 else f"{x / 2**20:.0f} MiB"


ORDER: tuple[str, ...] = (*ARMS, *MATRIX_ARMS, *(c[0] for c in COMPARISONS))


def _ordered(d: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The items of a reloaded JSON object (keys sorted) in the report's order: the encoder arms, the Matrix arms, the comparisons."""
    known = [(k, d[k]) for k in ORDER if k in d]
    return known + [(k, v) for k, v in d.items() if k not in ORDER]


def render(m: dict[str, Any], config: dict[str, Any]) -> str:
    v = m["verdicts"]
    lines = ["# Report n°3: the encoder arms on the panel", ""]
    if not m["registered"]:
        lines += ["**NOT THE REGISTERED RUN.** This run's configuration, or that of the Report n°2 run it pairs with, differs from "
                  "the pre-registered one (a smoke run or a test): its numbers are not results.", ""]  # fmt: skip
    lines += [
        "Pre-registered in `docs/PROTOCOL.md`, \"The encoder, and what a verdict is worth (iteration 4)\". Development period only; "
        "the validation year and the vault stay closed.",
        "",
        "## Verdicts",
        "",
        "| hypothesis | verdict | rule |",
        "|---|---|---|",
        *(f"| {h} | {v[h]['status']} | {v[h]['rule']} |" for h in ("H4", "H5", "H13")),
        "",
        f"Scored: {m['scored']['rows']} rows over {m['scored']['hours']} decision hours ({m['scored']['first_hour'][:13]} -> "
        f"{m['scored']['last_hour'][:13]}), Report n°2's scored rows one for one, paired hour by hour with its series "
        f"(SHA-256 `{m['report2']['series_sha256']}`). Every mean is a mean over hours of the hourly means; intervals are "
        "stationary-bootstrap 95 % percentile intervals with Report n°2's seeds.",
        "",
        "## Arms",
        "",
        "Each encoder arm is the mean of its seeds' probabilities before the floor, then the floor. A row whose window failed "
        "takes the arm's climatology (fallbacks).",
        "",
        "| arm | log-loss | Brier | BSS vs climatology | ECE | fallbacks |",
        "|---|---|---|---|---|---|",
    ]
    for arm, a in _ordered(m["arms"]):
        lines.append(f"| {arm} | {_ci(a['log_loss'])} | {_ci(a['brier'])} | {_ci(a['bss'])} | {_ci(a['ece'])} | {a['fallbacks']} |")
    for arm, a in _ordered(m["reference_arms"]):
        lines.append(f"| {arm} (Report n°2) | {_ci(a['log_loss'])} | {_ci(a['brier'])} | {_ci(a['bss'])} | {_ci(a['ece'])} | - |")
    lines += ["", "Each seed's log-loss, a description (the model is the ensemble):", "", "| arm | " + " | ".join(
        f"seed {s}" for s in next(iter(m["arms"].values()))["seeds"]["log_loss"]) + " | range |",
              "|---|" + "---|" * (len(next(iter(m["arms"].values()))["seeds"]["log_loss"]) + 1)]  # fmt: skip
    for arm, a in _ordered(m["arms"]):
        lines.append(f"| {arm} | " + " | ".join(f"{x:.5f}" for x in a["seeds"]["log_loss"].values()) + f" | {a['seeds']['range']:.5f} |")
    lines += ["", "## Comparisons", "", "Difference = a - b in mean hourly log-loss (negative: a is better).", "",
              "| comparison | a | b | difference | share of climatology | DM p | one-sided upper 95 % | bootstrap 95 % | bootstrap q95 | identical hours |",
              "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, c in _ordered(m["comparisons"]):
        if "undefined" in c:
            lines.append(f"| {name} | {c['a']} | {c['b']} | {c['mean_diff']:+.5f} | - | - | - | - | - | no test: {c['undefined']} |")
            continue
        b = c["bootstrap"]
        lines.append(f"| {name} | {c['a']} | {c['b']} | {c['mean_diff']:+.5f} | {c['share_of_climatology']:+.4%} | {c['dm_p']:.4f} | "
                     f"{c['upper_95_one_sided']:+.5f} | [{b['low']:+.5f}, {b['high']:+.5f}] | {b['q95']:+.5f} | {r2._informative(c)} |")  # fmt: skip
    h4, h13 = v["H4"], v["H13"]
    lines += [
        "",
        "## H4",
        "",
        f"Margin 0.005 x climatology-panel's mean hourly log-loss ({m['climatology_log_loss']:.5f}) = {h4['margin']:.6f}. Testable: "
        f"{'yes' if h4['testable'] else 'no'} (BSS interval above 0: enc-ols {h4['skill_above_0']['enc-ols']}, enc-softmax "
        f"{h4['skill_above_0']['enc-softmax']}). One-sided 95 % upper bound of mean(ols - softmax): {_num(h4['upper_95_one_sided'], '+.6f')}; "
        f"bootstrap 95th percentile: {_num(h4['bootstrap_q95'], '+.6f')}. Verdict: {h4['status']}."
        + (" Two arms that both sit at climatology would meet the bounds vacuously, so without that skill H4 and H13 are not testable."
           if not h4["testable"] else ""),
        "",
        "Descriptions beside the verdict, not verdicts: the same two bounds and margin on each seed-matched pair (enc-ols seed i "
        "against enc-softmax seed i), and on the ensembles in the reversed direction (enc-softmax non-inferior to enc-ols, "
        "difference softmax - ols). The verdict above is the registered one, on the ensembles.",
        "",
        "| description | a | b | difference | one-sided upper 95 % | bootstrap q95 | both below the margin |",
        "|---|---|---|---|---|---|---|",
        *(f"| {name} | {d['a']} | {d['b']} | {d['mean_diff']:+.6f} | {_num(d['upper_95_one_sided'], '+.6f')} | {_num(d['bootstrap_q95'], '+.6f')} | "
          f"{'yes' if d['bounds_below_margin'] else 'no'} |"
          for name, d in [*((f"seed {s}", d) for s, d in m["h4_descriptions"]["seed_pairs"].items()), ("reversed", m["h4_descriptions"]["reversed"])]),
        "",
        "## H5",
        "",
        f"enc-ols against matrix-select (Report n°2's series): {v['H5']['status']}. A limit: {v['H5']['limit']}.",
        "",
        "## H13 and the timing",
        "",
    ]  # fmt: skip
    t = m["timing"]
    if t is None:
        lines.append("The timing stage has not run: H13 is " + h13["status"] + ".")
    else:
        lines += [f"{t['repeats']} repeats cycling {', '.join(t['order'])}; each: {t['warmups']} untimed warm-up batches, then "
                  f"{t['batches']} timed batches of {t['batch']} rows ({t['rows']} rows drawn with the seed {t['seed']}), synchronized "
                  f"before and after each, scaled to {t['per_decisions']:,} decisions. Device: {t['gpu'] or t['device']}; driver "
                  f"{t['driver']}, CUDA {t['cuda']}, torch {t['torch']}; plain fp32 eager, softmax on the math backend, deterministic settings.",
                  "", "| arm | whole pass per 10^4 decisions: min, median, max (s) | attention operator alone, median (s) | "
                  "memory readout alone, median (s) |", "|---|---|---|---|"]  # fmt: skip
        for arm, a in _ordered(t["arms"]):
            w = a["whole"]
            lines.append(f"| {arm} | {w['min']:.4f}, {w['median']:.4f}, {w['max']:.4f} | {a['operator']['median']:.4f} | {a['readout']['median']:.4f} |")
        lines += ["", f"Median ratio enc-ols / enc-softmax (the registered rule, a ratio of medians): {_num(h13['median_ratio'])} (at most "
                  f"{h13['max_ratio']}). H13: {h13['status']}. Median of the per-repeat paired ratios, a description: "
                  f"{_num(h13['paired_median_ratio'])}. The operator's and the readout's own times are descriptions of the whole pass's split: "
                  "the operator runs on the GPU on cached Q, K and V, the readout on the CPU in fp64 on cached keys, as the whole pass "
                  "runs it. The copies of the keys z from the device to the host and of the readout m back belong to the whole pass but "
                  "to neither of these two timings, nor do the embedding, the layer norms, the MLPs and the head; analytic FLOPs favour "
                  "OLS by construction when T >> d_head."]  # fmt: skip
    ct = m["cost_table"]
    lines += ["", "## The cost table", "", f"Descriptive. GPU columns: {ct['gpu'] or ct['gpu_device'] or 'not measured'}. CPU columns: "
              f"{ct['cpu'] or 'not measured'}{' (' + ct['cpu_platform'] + ', olsprime_rs ' + ct['olsprime_rs'] + ')' if ct['cpu_platform'] else ''}. "
              "Columns never mix machines.", "",
              "| arm | parameters | FLOPs per decision | GPU s per 10^4 | GPU peak memory | CPU s per 10^4 | CPU peak memory, whole process | "
              "memory's own footprint (analytic) | training GPU-hours |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for arm, r in _ordered(ct["arms"]):
        enc = arm in m["arms"]
        flops = r["flops"]["total"] if r.get("flops") else None
        params = f"{r['parameters']:,}" if isinstance(r["parameters"], int) else str(r["parameters"])
        lines.append(f"| {arm} | {params} | {_num(flops, ',.0f')} | {_num(r.get('gpu_per_decisions_s'))} | "
                     f"{_bytes(r.get('gpu_peak_bytes')) if enc else '-'} | {_num(r.get('cpu_per_decisions_s'))} | "
                     f"{'-' if enc else _bytes(r.get('cpu_peak_bytes'))} | {'-' if enc else _bytes(r.get('memory_bytes'))} | "
                     f"{_num(r.get('training_hours'), '.3f') if enc else '-'} |")  # fmt: skip
    mt = m["matrix_timing"]
    mem = mt["memory_bytes"] if mt else None
    lines += ["", "FLOPs: a multiply-add counts 2. The encoder's formulas are in `report3.encoder_flops` and `attention_flops`, the Matrix "
              "arms' in `report3.matrix_flops` (dominant terms, with the mean window sizes of the timed rows). Matrix has no trained "
              "parameter: its fit-time choices are selections. Its CPU peak memory is the peak of the whole process, panel and windows "
              "included (the panel, the bars, the encoder's windows, torch, the Matrix memory and every arm), where the platform gives it "
              "without a subprocess; on Windows it does not. The memory's own footprint is analytic (`report3.matrix_memory_bytes`): "
              + (f"{mt['memory_entries']:,} entries x {mem['bytes_per_entry']:,} bytes per entry = {_bytes(mem['total'])} (entries "
                 f"{_bytes(mem['entries'])}, two trees of {mem['nodes_per_tree']:,} nodes {_bytes(mem['trees'])}). " if mem else "not measured. ")
              + "Training GPU-hours sum the fits' wall time; the encoder's peak GPU memory in training: "
              + ", ".join(f"{a} {_bytes(ct['arms'][a]['gpu_peak_bytes_training'])}" for a in m["arms"]) + "."]  # fmt: skip
    builds = "; ".join(f"{b['gpu'] or b['device']}, driver {b['driver']}, CUDA {b['cuda']}, torch {b['torch']}" for b in m["fits_builds"])
    lines += ["", "## Fits", "", f"Fitted on: {', '.join(m['fits_devices']) or '-'}, in {len(m['fits_builds'])} build(s): {builds or '-'}"
              + (" (more than one: not the registered run)" if len(m["fits_builds"]) > 1 else "") + ". The slice losses are the registered early-stopping "
              "statistic, not results: each epoch's log-loss on the fit's slice, the rows decided in the 30 days before the fit instant "
              "(December rows, the last 30 days of the year before the fit year, at the end of the training years), their labels known "
              "24 h before it; the kept epoch is the first with the lowest.", "", "| arm | seed | year | kept epoch | slice losses | checkpoint SHA-256 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for f in m["fits"]:
        lines.append(f"| {f['arm']} | {f['seed']} | {f['year']} | {f['kept_epoch']} | {', '.join(f'{x:.4f}' for x in f['slice_losses'])} | "
                     f"`{f['fit_sha256'][:16]}` |")  # fmt: skip
    lines += [
        "",
        "## Limits",
        "",
        "- H5 compares two systems: the encoder refits yearly and matrix-select monthly.",
        "- The panel H1 of the encoder arms forecasts the sub-panel's cut years with seed 1 before this run: a first look at the "
        "encoder's skill there, listed in the change log when it happens.",
        "- Each seed's log-loss is a description; the verdicts read the ensemble.",
        "",
        "## Variants",
        "",
        f"Variants tried so far: {m['variants']['before']} before this report, plus the {m['variants']['this_report']} encoder arms: "
        f"{m['variants']['total']}. A seed ensemble counts once.",
        "",
        *prov_mod.render_md(config["provenance"]),
        f"- Report n°2's series: `{m['report2']['series_sha256']}`; its report2.json: `{m['report2']['report2_json_sha256']}`; panel "
        f"fingerprint `{m['report2']['panel_fingerprint']}`.",
        f"- This report's series (`{m['series']['file']}`), SHA-256: `{m['series']['sha256']}`." if "series" in m else "",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def config_record(cfg: Report3Config, run: Report2Run, prov: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    import scipy

    return {
        "report": "report-3",
        "config": config_json(cfg),
        "registered": cfg.registered,
        "report2": {"series_sha256": run.sha256, "report2_json_sha256": run.report_sha256, "panel_fingerprint": run.fingerprint,
                    "provenance": run.config.get("provenance")},  # fmt: skip
        "fits_identity": identity,
        "versions": {"numpy": np.__version__, "polars": pl.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]},
        "provenance": prov,
    }


def write_score(run: Report2Run, cfg: Report3Config, parts: Parts, out: Path, prov: dict[str, Any]) -> dict[str, Any]:
    metrics, series = score(run, cfg, parts)
    out.mkdir(parents=True, exist_ok=True)
    r2.write_npz(out / SERIES_FILE, series)
    metrics["series"] = {"file": SERIES_FILE, "sha256": r2.sha256_file(out / SERIES_FILE)}
    r2._write_json(out / "config.json", config_record(cfg, run, prov, parts.identity()))
    r2._write_json(out / "report3.json", metrics)
    metrics, config = (json.loads((out / n).read_text(encoding="utf-8")) for n in ("report3.json", "config.json"))
    (out / REPORT_FILE).write_text(render(metrics, config), encoding="utf-8", newline="\n")
    return metrics


def fits_builds(parts: Parts, cfg: Report3Config) -> list[dict[str, Any]]:
    """The distinct builds the fits ran on (GPU, driver, CUDA, torch, device), sorted: one in a registered run."""
    builds = []
    for a in cfg.arms:
        for sd in cfg.seeds:
            for y in cfg.years:
                rec = parts.get_json(f"{part_key(a, sd, y)}/part.json")
                tim = parts.get_json(f"{part_key(a, sd, y)}/timing.json")
                if rec:
                    b = rec["build"]
                    builds.append({"gpu": b.get("gpu"), "cuda": b.get("cuda_build"), "torch": b.get("torch"), "device": b.get("device"),
                                   "driver": (tim or {}).get("driver")})  # fmt: skip
    uniq = {json.dumps(b, sort_keys=True) for b in builds}
    return [json.loads(t) for t in sorted(uniq)]


def run_record(parts: Parts, cfg: Report3Config, started_at: datetime, seconds: float, stages: dict[str, float]) -> dict[str, Any]:
    """The wall clock and the machines: the fits' GPU, driver, CUDA and torch (from the parts), the timing's, the CPU stage's."""
    uniq = fits_builds(parts, cfg)
    tm, mt = parts.get_json(TIMING_FILE), parts.get_json(MATRIX_TIMING_FILE)
    return {"started_at": started_at.isoformat(timespec="seconds"), "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds": round(seconds, 1), "stages_seconds": {k: round(v, 1) for k, v in stages.items()}, "fits_builds": uniq,
            "timing": None if tm is None else {k: tm.get(k) for k in ("gpu", "driver", "cuda", "torch", "device")},
            "matrix_timing": None if mt is None else {k: mt.get(k) for k in ("machine", "platform", "olsprime_rs", "python")}}  # fmt: skip


# --- Entry points ------------------------------------------------------------------------------------------


@dataclass(eq=False)
class Inputs:
    """What every stage reads: Report n°2's run, and (for fits, timing and matrix-timing) the panel and its windows."""

    run: Report2Run
    panel: pl.DataFrame | None = None
    data: EncoderData | None = None
    scored: np.ndarray | None = None
    years: dict[int, np.ndarray] = field(default_factory=dict)


def build_inputs(run: Report2Run, panel: pl.DataFrame, bars: Mapping[str, pl.DataFrame], cfg: Report3Config) -> Inputs:
    """The panel checked against Report n°2's fingerprint, its windows, and the scored rows checked one for one."""
    check_panel(panel, run)
    data = encoder_data(panel, dict(bars))
    scored = scored_panel_rows(data.arrays, run)
    return Inputs(run, panel, data, scored, rows_by_year(data.arrays, scored, cfg.years))


def identity_of(cfg: Report3Config, run: Report2Run, prov: dict[str, Any]) -> dict[str, Any]:
    return {"config": config_json(cfg), "code_sha256": prov.get("code", {}).get("sha256"), "manifest_digest": prov.get("manifest_digest"),
            "series_sha256": run.sha256, "panel_fingerprint": run.fingerprint}  # fmt: skip


def run(inputs: Inputs, cfg: Report3Config, out: Path, stage: str, *, prov: dict[str, Any], arms: Sequence[str] | None = None,
        log: Callable[[str], None] = lambda s: None, stop_after: int | None = None, driver: str | None = None,
        cpu_name: str | None = None) -> dict[str, Any] | None:  # fmt: skip
    """Run ``stage`` into ``out`` (resuming the fits from ``out/parts``). Returns the metrics when the score stage ran."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    arms = tuple(arms or cfg.arms)
    if set(arms) - set(cfg.arms):
        raise ValueError(f"unknown arms {sorted(set(arms) - set(cfg.arms))}")
    started, started_at, stages = time.perf_counter(), datetime.now(UTC), {}
    identity = identity_of(cfg, inputs.run, prov)
    # every stage binds the fits' configuration, code, data manifest and series; the device is operational, not identity
    require = {k: identity[k] for k in ("config", "code_sha256", "manifest_digest", "series_sha256")}
    metrics = None
    with deterministic():
        if stage in ("fits", "all"):
            assert inputs.data is not None and inputs.scored is not None
            parts = Parts(out, identity, log=log, stop_after=stop_after)
            plan = plan_fits(inputs.data, cfg, arms)
            log(f"fits: {len(arms)} arms x {len(cfg.seeds)} seeds x {len(cfg.years)} years, {plan['steps']} training steps in all "
                "(saved parts are skipped)")  # fmt: skip
            t = time.perf_counter()
            run_fits(inputs.data, inputs.years, cfg, parts, arms, log=log, driver=driver)
            stages["fits"] = time.perf_counter() - t
        if stage in ("timing", "all"):
            assert inputs.data is not None and inputs.scored is not None
            parts = Parts(out, None, require=require, log=log)
            t = time.perf_counter()
            run_timing(inputs.data, inputs.run, inputs.scored, cfg, parts, log=log, driver=driver)
            stages["timing"] = time.perf_counter() - t
        if stage == "matrix-timing":
            assert inputs.panel is not None and inputs.data is not None and inputs.scored is not None
            parts = Parts(out, None, require=require, create=identity, log=log)
            t = time.perf_counter()
            run_matrix_timing(inputs.panel, inputs.data, inputs.run, inputs.scored, cfg, parts, log=log, cpu_name=cpu_name)
            stages["matrix-timing"] = time.perf_counter() - t
    if stage in ("score", "all"):
        parts = Parts(out, None, require=require, log=log)
        t = time.perf_counter()
        metrics = write_score(inputs.run, cfg, parts, out, prov)
        stages["score"] = time.perf_counter() - t
        r2._write_json(out / "run.json", run_record(parts, cfg, started_at, time.perf_counter() - started, stages))
    return metrics


def verify_parts(out: Path, cfg: Report3Config) -> list[str]:
    """Every saved part checked against its digest: the fits are compared by their digests, never re-run."""
    problems = []
    parts = Parts(out, None)
    for a in cfg.arms:
        for sd in cfg.seeds:
            for y in cfg.years:
                try:
                    if verified_part(parts, part_key(a, sd, y), cfg) is None:
                        problems.append(f"{part_key(a, sd, y)}: not finished (no sealed part.json)")
                except PartRefused as e:
                    problems.append(str(e))
    for name in (TIMING_FILE, MATRIX_TIMING_FILE):
        try:
            parts.get(name)
        except PartRefused as e:
            problems.append(str(e))
    return problems


def check(expected: Path, run_: Report2Run, cfg: Report3Config, prov: dict[str, Any]) -> int:
    """Verify the parts by their digests, re-run the score stage from them and compare every output byte for byte."""
    if not (expected / "report3.json").is_file():
        print(f"{expected}: not a report3 run folder (no report3.json)")
        return 2
    problems = verify_parts(expected, cfg)
    ident = Parts(expected, None).identity()
    code = prov.get("code", {}).get("sha256")
    manifest = prov.get("manifest_digest")
    if ident.get("code_sha256") != code or ident.get("manifest_digest") != manifest:  # still re-scored, so that the outputs' differences show too
        problems.append(f"the fits ran at code digest {ident.get('code_sha256')} and data manifest {ident.get('manifest_digest')}, "
                        f"this check at code digest {code} and data manifest {manifest}")  # fmt: skip
    scratch = Path(tempfile.mkdtemp(prefix="report3-check-"))
    try:
        if not problems or all(p.startswith("the fits ran at code digest") for p in problems):
            # read only: the re-score writes to scratch; the code digest is reported above, not refused here
            parts = Parts(expected, None, require={"config": config_json(cfg), "series_sha256": run_.sha256})
            write_score(run_, cfg, parts, scratch, prov)
            for name in COMPARED:
                x, y = r2._read(expected, name), r2._read(scratch, name)
                if x != y:
                    problems.append(r2._file_summary(name, x, y, first="the run", second="the re-score"))
                    if name != SERIES_FILE and x is not None and y is not None:
                        problems.append(r2._unified(x, y, f"run/{name}", f"re-score/{name}", 40))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if problems:
        print(f"CHECK FAILED: {len(problems)} problem(s) in {expected}")
        for p in problems:
            r2._print(p)
        return 1
    print(f"check passed: every part verifies, and {len(COMPARED)} files are identical to {expected} (run.json excluded)")
    return 0


# --- The synthetic smoke ------------------------------------------------------------------------------------


SMOKE_START = datetime(2018, 9, 1, tzinfo=UTC)
SMOKE_MEMBERS = 10
SMOKE_HOURS = 24 * 182  # through 2019-03-01: the labels of February's last rows


def smoke_config(device: str) -> Report3Config:
    """The registered encoder, fit and timing on a synthetic panel: test months 2019-01 and 2019-02, the 2019 fit."""
    return Report3Config(years=(2019,), device=device, synthetic=f"{SMOKE_MEMBERS} synthetic members, hourly from 2018-09-01 for "
                         f"{SMOKE_HOURS // 24} days (seed 0); test months 2019-01 and 2019-02; the 2019 fit")  # fmt: skip


def synthetic_report2(out: Path, *, members: int = SMOKE_MEMBERS, start: datetime = SMOKE_START, hours: int = SMOKE_HOURS,
                      r2cfg: r2.Report2Config | None = None) -> tuple[pl.DataFrame, dict[str, pl.DataFrame], Path, str]:  # fmt: skip
    """A synthetic panel and a Report n°2 run on it (``r2.compute``), written to ``out`` as ``run_report`` writes one. Reads no data."""
    from olsprime.bench.synthetic import synthetic_bars

    r2cfg = r2cfg or r2.Report2Config(matrix=MatrixConfig(first_test=(2019, 1), last_test=(2019, 2)), first_month=(2018, 10), last_month=(2019, 2),
                                      readout_sample=50, previewed_through=(2019, 1))  # fmt: skip
    lo = datetime(*r2cfg.first_month, 1, tzinfo=UTC)
    hi = datetime(*r2cfg.last_month, 1, tzinfo=UTC)
    bars, parts = {}, []
    for i in range(members):
        symbol = f"S{i:02d}USDT"
        frame = synthetic_bars(symbol, start=start, days=hours, interval="1h", seed=i, drift=0.0, vol=0.01)
        bars[symbol] = frame
        months = [m for m in sorted(set(frame["open_time"].dt.truncate("1mo").to_list())) if lo <= m <= hi]
        parts.append(pair_rows(frame, months))
    panel = assemble(parts)
    metrics, _, series = r2.compute(panel, r2cfg)
    out.mkdir(parents=True, exist_ok=True)
    r2.write_npz(out / SERIES_FILE, series)
    metrics["series"] = {"file": SERIES_FILE, "sha256": r2.sha256_file(out / SERIES_FILE)}
    prov = prov_mod.record({})
    r2._write_json(out / "config.json", r2.config_record(r2cfg, prov, {}))
    r2._write_json(out / "report2.json", metrics)
    return panel, bars, out / SERIES_FILE, metrics["series"]["sha256"]


def _default_out(suffix: str = "report3") -> Path:
    return runs_dir() / f"{datetime.now(UTC):%Y-%m-%d}-{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m olsprime.bench.report3", description="Report no. 3: the encoder arms on the panel.")
    parser.add_argument("--series", type=Path, default=None, help="Report n°2's series.npz")
    parser.add_argument("--sha256", default=None, help="its SHA-256, as its report2.json records it")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default: <runs>/<UTC date>-report3; a restart resumes it)")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=None, help="restrict the fits stage to these arms")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--driver", default=None, help="the GPU driver's version, recorded (no subprocess may ask nvidia-smi)")
    parser.add_argument("--cpu-name", default=None, help="the CPU's name for the Matrix timing (e.g. 'AMD Ryzen 5 3600')")
    parser.add_argument("--check", type=Path, default=None, metavar="DIR", help="verify the parts and re-score DIR, byte for byte")
    parser.add_argument("--synthetic", action="store_true", help="the smoke: a synthetic panel and Report n°2 run, never stored data")
    args = parser.parse_args(argv)
    log = lambda s: print(f"{datetime.now(UTC):%H:%M:%S} {s}", flush=True)  # noqa: E731
    if args.synthetic:
        if args.series is not None:
            parser.error("--synthetic builds its own Report n°2 run")
        cfg = smoke_config(args.device)
        prov = prov_mod.record({})
        if args.check is not None:  # the smoke folder's own synthetic Report n°2 run, by the digest it records
            series = args.check / "report2-synthetic" / SERIES_FILE
            sha = json.loads((series.parent / "report2.json").read_text(encoding="utf-8"))["series"]["sha256"]
            return check(args.check, read_report2(series, sha), cfg, prov)
        out = args.out or _default_out("report3-smoke")
        t = time.perf_counter()
        panel, bars, series, sha = synthetic_report2(out / "report2-synthetic")
        log(f"synthetic panel of {panel.height} rows and its Report n°2 run in {time.perf_counter() - t:.1f} s")
    else:
        if args.series is None or args.sha256 is None:
            parser.error("--series and --sha256 are required (or --synthetic)")
        cfg = Report3Config(device=args.device)
        series, sha = args.series, args.sha256
        out = args.out or _default_out()
    if args.device == "cuda" and args.stage in ("fits", "timing", "all") and not torch.cuda.is_available():
        parser.error("CUDA is not available: run in WSL2 with the CUDA environment, or pass --device cpu")
    from olsprime.train.determinism import apply

    apply()
    run_ = read_report2(series, sha)
    if args.check is not None:
        prov = prov_mod.record(prov_mod.current_manifest())
        return check(args.check, run_, cfg, prov)
    if not args.synthetic:
        prov = prov_mod.record(prov_mod.current_manifest())  # before any data is read
        if args.stage == "score":
            inputs = Inputs(run_)
        else:
            t = time.perf_counter()
            panel, bars = load_panel(run_.config)
            log(f"panel of {panel.height} rows loaded in {time.perf_counter() - t:.1f} s")
    if args.synthetic or args.stage != "score":
        t = time.perf_counter()
        inputs = build_inputs(run_, panel, bars, cfg)
        log(f"windows built, {inputs.scored.size} scored rows paired with Report n°2's in {time.perf_counter() - t:.1f} s")
    metrics = run(inputs, cfg, out, args.stage, prov=prov, arms=args.arms, log=log, driver=args.driver, cpu_name=args.cpu_name)
    if metrics is not None:
        v = metrics["verdicts"]
        log(f"report3 written to {out}: " + ", ".join(f"{h}: {v[h]['status']}" for h in ("H4", "H5", "H13")))
    else:
        log(f"stage {args.stage} done in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
