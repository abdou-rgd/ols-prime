"""``head``: what each OLS head of a fitted ``enc-ols`` attends to, at one scored row (``docs/PROTOCOL.md``, "``olsprime.explain``").

The readings fixed by the orchestrator on 2026-10-30, before any encoder run on real data:

1. **The call.** ``head PAIR AS_OF --run DIR --sha256 HEX [--seed 1|2|3] --test-months [--jsonl FILE]``. ``DIR`` is a
   Report n°3 run folder and ``HEX`` the SHA-256 of its ``series.npz``, which its ``report3.json`` must agree with.
   Without ``--seed``, every seed of the run is given, each apart (one answer per seed).
2. **The rows it explains.** Only scored rows of 2019-2024, which Report n°3 forecasts: every call needs
   ``--test-months`` and is a logged contact. A row before 2019-01-01 is refused: no fit is in force there.
3. **The model.** The ``enc-ols`` checkpoint of the seed's fit of the row's year. A row's year is that of its
   month, the month of its bar's open (``docs/PROTOCOL.md``, Fits; :func:`olsprime.bench.report3.rows_by_year`):
   that is the fit Report n°3 forecast the row with. Its part is verified by its digests and against its folder
   (arm, seed, year, ``fit_ms``) by :func:`olsprime.bench.report3.verified_part` before any value is read.
4. **W.** For each block and head, ``W = Q~ (K~^T K~ + Lambda)^-1 K~^T`` over the row's 64-bar window (64 x 64,
   rank at most d_head + 1 = 9), from the Q, K and lambda of the forward pass that gave the row's forecast. That
   pass evaluates ``O = Q~ (A^-1 (K~^T V))`` and never forms W (change log, 2026-10-30, wave 1). So the input of
   each attention is captured in that very pass (the row's batch of ``eval_batch`` rows, padded as the fits pad
   it), the attention is recomputed from it and must reproduce the pass's output to the bit, and W is then formed
   from the same Q~, K~ and lambda by :func:`olsprime.models.encoder.ols_weights` (Cholesky, fp32, on the row).
5. **The row of W** is the decision token's (the window's last bar): 64 signed weights, oldest bar first, with
   their bars' open times.
6. **The diagonal mass** is ``trace(W) / 64``; the decision token's own weight ``W[64, 64]`` is given beside it.
7. **The effective rank** is ``exp`` of the Shannon entropy of W's normalized singular values (Roy and Vetterli,
   2007), over all 64 of them (SVD in fp64 of the fp32 W). W's rank is at most 9 by construction: the nine
   leading singular values are given, and the tail's share of the sum, which is rounding.
8. **The row sum.** Every row of every head must sum to 1 within :func:`olsprime.models.encoder.row_sum_tolerance`
   (the sums are taken in fp64 over the fp32 weights). A failure stops the call (exit 3). That registered fp32
   bound is loose (1e3 to 1e4 times the measured gaps, audit pass 4g), so a tight check runs beside it: W formed in
   fp64 from the same Q, K and lambda must have every row within ``row_sum_tolerance`` taken in fp64. Both bounds
   are reported, and either failure stops the call.
9. **The head's own state:** lambda and the condition number of ``K~^T K~ + Lambda`` (fp64 eigenvalues).
10. **The model check**, loud. The row's forecast is recomputed from the checkpoint and must equal the seed's
    saved forecast of that row (``forecast.safetensors``, before the floor) to the bit. It is computed as the fits
    computed it: :meth:`olsprime.train.fit.EncoderSystem.predict` over the very rows the part forecast (the
    year's scored rows with a valid window), so the same fixed batches of ``eval_batch`` rows in panel order, the
    same union of queries and memory entries, and the memory's prefix sums as ``predict`` builds them; on the
    device the fit ran on, whose build (torch, CUDA, GPU, cuDNN, CPU threads) must be the fit's. A call on
    another device or build is refused rather than held to a bound: no bound is derived for a whole forward pass
    across builds, so none is claimed.

**Identity and fingerprint** (audit pass 4g). The check above is a fingerprint, not an identity: a checkpoint
changed by one ulp and re-sealed with its forecast can still reproduce a re-sealed forecast to the bit. What makes
the part the one Report n°3 scored is its digests bound to ``report3.json``, whose ``series.npz`` digest the
caller gives: the entry of ``report3.json["fits"]`` for the part's arm, seed and year must record the part's
``fit_sha256``, ``forecast_sha256``, ``forecast_rows`` and ``kept_epoch``, and the forecast file must hold
``forecast_rows`` rows, or the call is refused (exit 2). ``report3.json`` itself is bound only through the series
digest it records; a forged ``report3.json`` that keeps that record is beyond this check.

Order of a call: the arguments are checked (``--test-months``, ``AS_OF`` in reach and not before 2019,
``--jsonl``), ``series.npz`` is hashed against the digest given and ``report3.json``'s record; then one contact
line is written for the row, naming what the call reads: the forecasts and labels of the fit year of ``AS_OF``'s
row, and the memory's labels through ``AS_OF``. Only then is the run folder opened (its configuration, parts and
checkpoints), the panel rebuilt and checked against Report n°2's fingerprint that ``report3.json`` records. A row
whose fit year is not the one logged is refused before its part is read.

The panel: a run paired with the registered Report n°2 has it rebuilt from the store as Report n°2 built it
(:func:`olsprime.bench.report3.load_panel` with the registered ``Report2Config``). Any other run (a synthetic smoke,
a test) must hand the panel and its bars in (``panel_source``); the fingerprint is required either way.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from olsprime.bench import provenance as prov_mod
from olsprime.bench import report2 as r2
from olsprime.bench import report3 as r3
from olsprime.explain.common import (
    EXPLORATORY,
    TEST_FLAG,
    LoudCheckFailed,
    Refused,
    check_as_of,
    log_contact,
    month_label,
    parse_time,
)
from olsprime.explain.gitstate import code_state
from olsprime.features.panel import fingerprint
from olsprime.models.encoder import EncoderModel, ols_attention, ols_condition, ols_weights, row_sum_tolerance
from olsprime.models.matrix import iso
from olsprime.train.determinism import build as build_info
from olsprime.train.determinism import deterministic
from olsprime.train.fit import FIT_YEARS, EncoderData, EncoderSystem, encoder_data, load_state, year_start_ms

HEAD_ARM = "enc-ols"
HOUR_MS = 3_600_000
FIRST_FIT_MS = year_start_ms(FIT_YEARS[0])  # 2019-01-01: no fit of Report n°3 is in force before
REPORT_FILE = "report3.json"
SERIES_FILE = r3.SERIES_FILE
TOP_BARS = 5
# What must be the fit's for its forecast to be recomputed to the bit (determinism.build's keys).
BUILD_KEYS = ("device", "torch", "cuda_build", "gpu", "cudnn", "capability", "cpu_threads")
DEFINITIONS = {
    "row": "the decision token's row of W (the window's last bar): 64 signed weights, oldest bar first",
    "diagonal_mass": "trace(W) / 64, the mean weight a token puts on itself; 1 for a head that copies its input",
    "effective_rank": ("exp of the Shannon entropy of W's normalized singular values (Roy and Vetterli, 2007), over all 64; "
                       "W's rank is at most d_head + 1 by construction"),  # fmt: skip
    "row_sum": "every row of W sums to 1 by construction (the constant key coordinate); sums in fp64 of the fp32 weights against "
               "the registered bound, and of W formed in fp64 against the same bound taken in fp64 (audit pass 4g)",
    "condition": "2-norm condition number of K~^T K~ + Lambda, from its fp64 eigenvalues",
    "wv_gap": "max |W V - O| over the row's 64 tokens: W V in fp64 from the fp32 W and V, O the forward pass's (fp32): a description",
    "w_fp64_gap": "max |W - W64| over the 64 x 64 weights, W64 formed in fp64 from the same Q, K and lambda: the fp32 W's own error",
}
YEAR_RULE = "the fit of the row's year (the year of its month, the month of its bar's open), the one Report n°3 forecast it with"


# --- The run folder ---------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_head_digest(run_dir: Path, sha256: str) -> None:
    """Before any contact line: hash ``series.npz`` (bytes only) against the digest given and ``report3.json``'s record."""
    path = Path(run_dir) / SERIES_FILE
    want = sha256.strip().lower()
    if not path.is_file():
        raise Refused(f"{path}: no such file")
    got = r2.sha256_file(path)
    if got != want:
        raise Refused(f"{path}: SHA-256 {got}, not the {want} given")
    report_path = path.parent / REPORT_FILE
    if not report_path.is_file():
        raise Refused(f"{path.parent}: no {REPORT_FILE} beside the series: not a Report n°3 run folder")
    try:
        rec = json.loads(report_path.read_bytes()).get("series")
    except (ValueError, AttributeError):
        raise Refused(f"{report_path}: not a JSON object") from None
    if not isinstance(rec, dict) or rec.get("file") != path.name or rec.get("sha256") != got:
        raise Refused(f"{path}: its run's {REPORT_FILE} records {rec!r}, not this file's SHA-256 {got}")


def rebuild_config(recorded: Mapping[str, Any]) -> r3.Report3Config:
    """The run's ``Report3Config`` from its ``config.json``; it must write back the same JSON (the device is operational)."""

    def tup(v: Any) -> Any:
        return tuple(tup(x) for x in v) if isinstance(v, list) else v

    if not isinstance(recorded, Mapping):
        raise Refused("config.json has no config: not a Report n°3 run")
    fields = {k: tup(v) for k, v in recorded.items() if k != "registered"}
    try:
        cfg = r3.Report3Config(**fields, device="cpu")
    except (TypeError, ValueError) as exc:
        raise Refused(f"config.json's config does not rebuild a Report3Config: {exc}") from None
    if r3.config_json(cfg) != json.loads(json.dumps(dict(recorded))):
        raise Refused("config.json's config does not round-trip through Report3Config: the run was made by other code")
    return cfg


@dataclass(eq=False)
class HeadRun:
    """A Report n°3 run folder, opened by the SHA-256 of its ``series.npz``."""

    path: Path
    series_sha256: str
    report: dict[str, Any]
    report_sha256: str
    config: dict[str, Any]
    cfg: r3.Report3Config
    parts: r3.Parts
    identity: dict[str, Any]


def open_head_run(run_dir: Path, sha256: str) -> HeadRun:
    """Open the run folder (after the contact line): ``report3.json``, ``config.json`` and the parts' identity."""
    run_dir = Path(run_dir)
    verify_head_digest(run_dir, sha256)  # again: the folder could have changed since the check before the contact
    report_bytes = (run_dir / REPORT_FILE).read_bytes()
    report = json.loads(report_bytes)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise Refused(f"{run_dir}: no config.json")
    config = json.loads(config_path.read_bytes())
    cfg = rebuild_config(config.get("config"))
    if HEAD_ARM not in cfg.arms:
        raise Refused(f"the run has no {HEAD_ARM} arm")
    if not (run_dir / "parts" / "IDENTITY.json").is_file():  # checked first: Parts would create the folder
        raise Refused(f"{run_dir}: no parts/IDENTITY.json, so no fit to read")
    rep2 = report.get("report2") or {}
    try:
        parts = r3.Parts(run_dir, None, require={"config": r3.config_json(cfg), "series_sha256": rep2.get("series_sha256")})
    except r3.ResumeRefused as exc:
        raise Refused(f"the parts are not this run's: {exc}") from None
    return HeadRun(run_dir, sha256.strip().lower(), report, _sha(report_bytes), config, cfg, parts, parts.identity())


def registered_panel(run: HeadRun, *, data_dir: Path | None, universe: pl.DataFrame | None) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Report n°2's registered panel and its bars, from the store, as Report n°3's fits read them."""
    if not run.report.get("report2", {}).get("registered"):
        raise Refused("this run is not paired with the registered Report n°2, the only panel head rebuilds from the store; "
                      "a synthetic or test run must hand its panel in")  # fmt: skip
    return r3.load_panel({"config": dataclasses.asdict(r2.Report2Config())}, data_dir=data_dir, universe=universe)


# --- The forward pass, captured ---------------------------------------------------------------


@dataclass(eq=False)
class Captured:
    """The batch that held the row in the forecast's forward pass: its index, the row's place, and each attention's input and output."""

    batch: int
    offset: int
    batch_rows: np.ndarray
    layers: list[tuple[torch.Tensor, torch.Tensor]]


class CapturingSystem(EncoderSystem):
    """The honest system, unchanged, that also keeps the attention inputs and outputs of the batch holding ``target``.

    Only :meth:`_encode` is wrapped: it runs as before, with forward hooks on the attention modules for the one
    batch whose rows contain ``target``. Hooks read; they change nothing.
    """

    def __init__(self, data: EncoderData, cfg: Any, device: torch.device, target: int) -> None:
        super().__init__(data, cfg, device=device)
        self.target = int(target)
        self.batches = 0
        self.captured: Captured | None = None

    def _encode(self, model: EncoderModel, bars: torch.Tensor, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.batches
        self.batches += 1
        hit = np.flatnonzero(np.asarray(rows) == self.target)
        if hit.size == 0:
            return super()._encode(model, bars, rows)
        if self.captured is not None:
            raise LoudCheckFailed(f"the row was encoded in two batches ({self.captured.batch} and {index}): not the fits' fixed batches")
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        def keep(_module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
            layers.append((args[0].detach().clone(), output.detach().clone()))

        handles = [block.attn.register_forward_hook(keep) for block in model.encoder.blocks]
        try:
            out = super()._encode(model, bars, rows)
        finally:
            for h in handles:
                h.remove()
        self.captured = Captured(index, int(hit[0]), np.asarray(rows).copy(), layers)
        return out


# --- One head's numbers -----------------------------------------------------------------------


def effective_rank(singular_values: np.ndarray) -> float:
    """exp of the Shannon entropy of the normalized singular values (Roy and Vetterli, 2007); 0 log 0 = 0."""
    s = np.asarray(singular_values, dtype=np.float64)
    total = float(s.sum())
    if not total > 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def head_stats(w: np.ndarray, d_head: int) -> dict[str, Any]:
    """The numbers of one head's W (T x T): diagonal mass, the decision token's own weight, effective rank, singular values."""
    w = np.asarray(w, dtype=np.float64)
    t = w.shape[0]
    s = np.linalg.svd(w, compute_uv=False)
    cap = d_head + 1
    total = float(s.sum())
    return {
        "diagonal_mass": float(np.trace(w)) / t,
        "self_weight": float(w[-1, -1]),
        "effective_rank": effective_rank(s),
        "singular_values": [float(x) for x in s[:cap]],
        "singular_tail_share": float(s[cap:].sum()) / total if total > 0 else 0.0,
        "rank_cap": cap,
    }


def row_view(row: np.ndarray, opens: np.ndarray, top: int = TOP_BARS) -> dict[str, Any]:
    """The decision token's row: its weights, sum |w|, negative mass and the bars with the largest |w|."""
    row = np.asarray(row, dtype=np.float64)
    t = row.size
    order = np.argsort(-np.abs(row), kind="stable")[:top]
    return {
        "weights": [float(x) for x in row],
        "sum_abs": float(np.abs(row).sum()),
        "negative_mass": float(row[row < 0].sum()),
        "top": [{"bar": iso(int(opens[j])), "lag_hours": int(t - 1 - j), "w": float(row[j])} for j in order],
    }


# --- The call ---------------------------------------------------------------------------------


def check_call(args: argparse.Namespace) -> int:
    """The arguments, before any file is read: ``--test-months``, ``AS_OF`` in reach and not before 2019. Returns ``AS_OF``."""
    if not args.test_months:
        raise Refused(f"head explains scored rows of 2019-2024 only, which are test hours: name that contact with {TEST_FLAG}")
    t = parse_time(args.as_of)
    check_as_of(t, True)
    if t < FIRST_FIT_MS:
        raise Refused(f"as_of {iso(t)} is before {iso(FIRST_FIT_MS)}: no fit of Report n°3 is in force there")
    return t


def logged_fit_years(t: int) -> tuple[int, ...]:
    """The fit years ``AS_OF`` can name, before the panel is read: the year of a bar opening at ``t``, and of a bar
    decided at ``t`` (opened at ``t - 1 h + 1 ms``, its known_at's nominal lag). One year, except at a year's first
    instant."""
    return tuple(sorted({int(iso(t)[:4]), int(iso(t - HOUR_MS + 1)[:4])}))


def _find_row(data: EncoderData, pair: str, t: int) -> int:
    a = data.arrays
    if pair not in a.symbols:
        raise Refused(f"{pair} is not a member of the run's panel")
    code = list(a.symbols).index(pair)
    hit = np.flatnonzero((a.symbol == code) & ((a.decision_ms == t) | (a.open_ms == t)))
    if hit.size > 1:  # a bar that closed late is decided at the next bar's open
        raise Refused(f"{pair} at {iso(t)} names two rows, one decided and one opening at that instant: give the other instant of the row")
    if hit.size != 1:
        raise Refused(f"{pair} has no row whose decision time or bar open is {iso(t)}")
    row = int(hit[0])
    if int(a.decision_ms[row]) < FIRST_FIT_MS or int(a.month_ms[row]) < FIRST_FIT_MS:
        raise Refused(f"{pair} at {iso(t)}: the row's month is {month_label(int(a.month_ms[row]))}, before 2019: no fit is in force there")
    return row


def _forecast_rows(data: EncoderData, fc: Mapping[str, Any], key: str) -> np.ndarray:
    """The panel rows of a part's forecast, in its order (panel order): the rows the fit forecast."""
    a = data.arrays
    names = [str(n) for n in fc["symbols"]]
    if names != list(a.symbols):
        raise Refused(f"{key}: the forecast's symbols are not the rebuilt panel's")
    k = len(names) + 1
    pk = a.decision_ms.astype(np.int64) * k + a.symbol
    fk = fc["decision_time"].astype(np.int64) * k + fc["symbol"].astype(np.int64)
    pos = np.searchsorted(pk, fk)
    found = pos < pk.size
    found[found] &= pk[pos[found]] == fk[found]
    if not found.all():
        raise Refused(f"{key}: {int((~found).sum())} forecast rows are not rows of the rebuilt panel")
    if not (np.array_equal(a.open_ms[pos], fc["open_time"]) and np.array_equal(a.label[pos], fc["label"])):
        raise Refused(f"{key}: the forecast rows' open times or labels are not the rebuilt panel's")
    if pos.size > 1 and np.any(np.diff(pos) <= 0):
        raise Refused(f"{key}: the forecast rows are not in panel order")
    return pos


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def _tensor_bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))) if a.dtype == torch.float32 else bool(torch.equal(a, b))


def _build_check(rec: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Refused("the fit ran on CUDA and CUDA is not available here: its forecast can only be recomputed to the bit on its device")
    fit_build, here = rec.get("build") or {}, build_info(device)
    differ = {k: (fit_build.get(k), here.get(k)) for k in BUILD_KEYS if fit_build.get(k) != here.get(k)}
    if differ:
        raise Refused("this machine's build is not the fit's, so the forecast cannot be held to the bit here (no bound is derived "
                      "for another build): " + "; ".join(f"{k} {a!r} at the fit, {b!r} here" for k, (a, b) in differ.items()))  # fmt: skip
    return {k: fit_build.get(k) for k in BUILD_KEYS if k in fit_build}


IDENTITY_KEYS = ("fit_sha256", "forecast_sha256", "forecast_rows", "kept_epoch")


def scored_record(run: HeadRun, key: str, rec: Mapping[str, Any], forecast: Mapping[str, Any]) -> dict[str, Any]:
    """The part's entry in ``report3.json["fits"]``: refused unless it records this part's digests, rows and kept epoch.

    ``verified_part`` checks the parts folder against itself; this binds the part to what Report n°3 scored.
    """
    arm, seed, year = r3.parse_key(key)
    entries = [f for f in run.report.get("fits") or [] if f.get("arm") == arm and f.get("seed") == seed and f.get("year") == year]
    if len(entries) != 1:
        raise Refused(f"{key}: {len(entries)} entries for it in {REPORT_FILE}'s fits: not a part Report n°3 scored")
    entry = entries[0]
    differ = [k for k in IDENTITY_KEYS if entry.get(k) != rec.get(k)]
    if differ:
        raise Refused(f"{key}: the part's {', '.join(differ)} are not the ones {REPORT_FILE} records for it "
                      "(a part re-sealed after Report n°3 scored it)")  # fmt: skip
    if int(forecast["raw"].shape[0]) != int(entry["forecast_rows"]):
        raise Refused(f"{key}: the forecast holds {forecast['raw'].shape[0]} rows, {REPORT_FILE} records {entry['forecast_rows']}")
    return entry


def explain_seed(run: HeadRun, data: EncoderData, row: int, seed: int) -> dict[str, Any]:
    """One seed's answer at ``row``: the part verified, the forecast recomputed to the bit, every head's W and its numbers."""
    a, cfg = data.arrays, run.cfg
    year = int(r3.year_of_month(a.month_ms[[row]])[0])
    if year not in cfg.years:
        raise Refused(f"the row's year {year} has no fit in this run (fits: {list(cfg.years)})")
    if seed not in cfg.seeds:
        raise Refused(f"seed {seed} is not one of the run's seeds {list(cfg.seeds)}")
    key = r3.part_key(HEAD_ARM, seed, year)
    try:
        rec = r3.verified_part(run.parts, key, cfg)
        if rec is None:
            raise Refused(f"{key}: no finished part (no sealed part.json)")
        state, std, fc = r3.load_fit(run.parts, key, cfg)
        forecast = r3.read_forecast(run.parts, key)
    except (r3.PartRefused, r3.PairingRefused) as exc:
        raise Refused(f"{key}: {exc}") from None
    scored_record(run, key, rec, forecast)
    enc = fc.encoder
    if enc.arm != HEAD_ARM or enc.attention != "ols" or enc.value_leak:
        raise Refused(f"{key}: the checkpoint is not an honest {HEAD_ARM} (arm {enc.arm}, value_leak {enc.value_leak})")
    device = torch.device(str(rec["device"]))
    build = _build_check(rec, device)
    rows = _forecast_rows(data, forecast, key)
    at = np.flatnonzero(rows == row)
    if at.size != 1:
        raise Refused(f"{a.symbols[a.symbol[row]]} decided {iso(int(a.decision_ms[row]))} is not a row {key} forecast: not a scored row")
    j = int(at[0])
    ok = data.usable[rows]
    if not np.array_equal(~ok, forecast["fallback"].astype(bool)):
        raise LoudCheckFailed(f"{key}: the rebuilt windows do not flag the rows the fit gave the climatology to")
    if not ok[j]:
        raise Refused(f"{key}: this row's window failed, so the fit gave it the climatology: no head saw it")
    queries = rows[ok]
    i = int(np.flatnonzero(queries == row)[0])

    system = CapturingSystem(data, fc, device, row)
    model = EncoderModel(enc)
    load_state(model, state)
    model = model.to(device).eval()
    with torch.no_grad():
        bars = system.standardized_bars(std)
        out = system.predict(model, bars, queries)
        cap = system.captured
        if cap is None or int(cap.batch_rows[cap.offset]) != row or len(cap.layers) != enc.n_layers:
            raise LoudCheckFailed(f"{key}: the forward pass of the row's batch was not captured")
        got, saved = out["raw"][i], forecast["raw"][j]
        if not _bits_equal(got, saved):
            ulps = np.abs(np.ascontiguousarray(got, dtype=np.float64).view(np.int64) - np.ascontiguousarray(saved, dtype=np.float64).view(np.int64))
            raise LoudCheckFailed(f"{key}: the forecast recomputed from the checkpoint, {got.tolist()}, is not the saved forecast "
                                  f"{saved.tolist()} to the bit (up to {int(ulps.max())} ulps), on {device} as the fit ran")  # fmt: skip
        heads, failures, recomputed = _heads(model, cap, enc)
    if failures:
        raise LoudCheckFailed(f"{key}: {len(failures)} rows of W do not sum to 1 within row_sum_tolerance; first: " + "; ".join(failures[:3]))
    if not recomputed:
        raise LoudCheckFailed(f"{key}: the attention recomputed from its captured input does not reproduce the forward pass to the bit")

    end = int(data.windows.end[row])
    opens = data.windows.open_ms[end - enc.window + 1 : end + 1]
    worst = max(heads, key=lambda h: h["row_sum"]["max_dev_over_bound"])
    for h in heads:
        h["row"] = row_view(h.pop("_row"), opens)
    return {
        "kind": "head",
        "arm": HEAD_ARM,
        "seed": seed,
        "row": {"pair": a.symbols[a.symbol[row]], "decision_time": iso(int(a.decision_ms[row])), "open_time": iso(int(a.open_ms[row])),
                "month": month_label(int(a.month_ms[row])), "year": year,
                "window": {"bars": [iso(int(x)) for x in opens], "first_bar": iso(int(opens[0])), "last_bar": iso(int(opens[-1]))}},  # fmt: skip
        "fit": {"part": key, "year": year, "fit_time": iso(int(fc.fit_ms)), "rule": YEAR_RULE, "fit_sha256": rec["fit_sha256"],
                "forecast_sha256": rec["forecast_sha256"], "kept_epoch": rec["kept_epoch"], "device": str(device), "build": build},  # fmt: skip
        "forecast": {
            "raw": [float(x) for x in got],
            "saved_raw": [float(x) for x in saved],
            "equal_to_the_bit": True,
            "how": {"device": str(device), "eval_batch": int(fc.eval_batch), "batch": int(cap.batch), "batches": int(system.batches),
                    "position_in_batch": int(cap.offset), "batch_rows": int(cap.batch_rows.size), "queries": int(queries.size),
                    "computation": ("EncoderSystem.predict over the rows the part forecast (the year's scored rows with a valid window): "
                                    "the same union of queries and memory entries in fixed padded batches in panel order, the memory's "
                                    "prefix sums on the CPU in fp64 as predict builds them")},  # fmt: skip
        },
        "heads": heads,
        "checks": {
            "forecast": "equal to the saved forecast of the seed, to the bit",
            "identity": f"the part's {', '.join(IDENTITY_KEYS)} equal the ones {REPORT_FILE} records for it",
            "attention_recomputed": ("the Q, K and lambda behind W come from inputs captured by hooks inside the very forward pass whose "
                                     "forecast matched the saved one to the bit; recomputing each block's attention from them gives that "
                                     "pass's output to the bit, which shows the operator is deterministic on this device"),  # fmt: skip
            "row_sums": {"passed": True, "rows_checked": len(heads) * enc.window, "max_dev": max(h["row_sum"]["max_dev"] for h in heads),
                         "max_dev_over_bound": worst["row_sum"]["max_dev_over_bound"], "worst_head": [worst["block"], worst["head"]],
                         "fp64_max_dev": max(h["row_sum"]["fp64"]["max_dev"] for h in heads),
                         "fp64_max_dev_over_bound": max(h["row_sum"]["fp64"]["max_dev_over_bound"] for h in heads)},  # fmt: skip
        },
        "definitions": DEFINITIONS,
        "exploratory": EXPLORATORY,
    }


def _heads(model: EncoderModel, cap: Captured, enc: Any) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Every head's W at the captured row, its numbers and its row-sum check; and whether every attention was reproduced."""
    r = cap.offset
    heads, failures, recomputed = [], [], True
    for b, (x, y) in enumerate(cap.layers):
        attn = model.encoder.blocks[b].attn
        recomputed &= _tensor_bits_equal(attn(x), y)
        q, k, v = attn.project(x)
        lam = attn.lam()
        o = ols_attention(q, k, v, lam, order=enc.ols_order)[r : r + 1]
        qr, kr, vr = q[r : r + 1], k[r : r + 1], v[r : r + 1]
        w = ols_weights(qr, kr, lam)  # (1, H, T, T), fp32
        exact = ols_weights(qr.double(), kr.double(), lam.double())  # the same W in fp64, from the same Q, K and lambda
        w_gap = (w[0].double() - exact[0]).abs().amax(dim=(-1, -2)).cpu().numpy()
        tol = row_sum_tolerance(w, kr, lam)[0].double().cpu().numpy()
        tol_exact = row_sum_tolerance(exact, kr.double(), lam.double())[0].cpu().numpy()  # the same bound, in fp64: tight
        sums_exact = exact[0].sum(dim=-1).cpu().numpy()
        cond = ols_condition(kr, lam)[0].cpu().numpy()
        w64 = w[0].double().cpu().numpy()
        gap = (w[0].double() @ vr[0].double() - o[0].double()).abs().amax(dim=(-1, -2)).cpu().numpy()
        scale = o[0].double().abs().amax(dim=(-1, -2)).cpu().numpy()
        lam_np = lam.detach().double().cpu().numpy()
        for h in range(enc.n_heads):
            sums = w64[h].sum(axis=-1)
            dev = np.abs(sums - 1.0)
            bad = np.flatnonzero(~(dev <= tol[h]))
            failures += [f"block {b + 1} head {h + 1} row {int(i) + 1}: |sum - 1| = {dev[i]:.3g} > {tol[h][i]:.3g}" for i in bad]
            ratio = dev / tol[h]
            dev_x = np.abs(sums_exact[h] - 1.0)
            bad_x = np.flatnonzero(~(dev_x <= tol_exact[h]))
            failures += [f"block {b + 1} head {h + 1} row {int(i) + 1} (fp64 W): |sum - 1| = {dev_x[i]:.3g} > {tol_exact[h][i]:.3g}"
                         for i in bad_x]  # fmt: skip
            heads.append({
                "block": b + 1,
                "head": h + 1,
                "lambda": float(lam_np[h]),
                "condition": float(cond[h]),
                **head_stats(w64[h], enc.d_head),
                "row_sum": {"decision_token": float(sums[-1]), "decision_token_bound": float(tol[h][-1]), "max_dev": float(dev.max()),
                            "max_dev_over_bound": float(ratio.max()), "bound_at_max_dev": float(tol[h][int(np.argmax(dev))]),
                            "fp64": {"max_dev": float(dev_x.max()), "max_dev_over_bound": float((dev_x / tol_exact[h]).max()),
                                     "bound_at_max_dev": float(tol_exact[h][int(np.argmax(dev_x))]),
                                     "decision_token_bound": float(tol_exact[h][-1])}},  # fmt: skip
                "wv_gap": {"max_abs": float(gap[h]), "output_scale": float(scale[h])},
                "w_fp64_gap": float(w_gap[h]),
                "_row": w64[h][-1],
            })  # fmt: skip
    return heads, failures, recomputed


def head_provenance(run: HeadRun, *, contacts: Path, panel_fingerprint: str) -> dict[str, Any]:
    """Every answer's provenance: this code's commit and digest, the run's manifest, panel and series, the fits' code digest."""
    state = code_state()
    prov = run.config.get("provenance", {})
    rep2 = run.report.get("report2", {})
    return {
        "contacts_file": str(contacts),
        "commit": prov_mod.git_commit(),
        "dirty": state["dirty"],
        "dirty_scope": state["scope"],
        "code_sha256": prov_mod.code_digest()["sha256"],
        "fits_code_sha256": run.identity.get("code_sha256"),
        "manifest_digest": prov.get("manifest_digest"),
        "panel_fingerprint": panel_fingerprint,
        "panel_fingerprint_verified": True,
        "series_sha256": run.series_sha256,
        "report3_json_sha256": run.report_sha256,
        "report2_series_sha256": rep2.get("series_sha256"),
        "run": run.path.name,
        "run_commit": prov.get("git_commit"),
        "registered_run": bool(run.report.get("registered")),
    }


PanelSource = Callable[[], tuple[pl.DataFrame, Mapping[str, pl.DataFrame]]]


def run_head(args: argparse.Namespace, *, contacts: Path, data_dir: Path | None = None, universe: pl.DataFrame | None = None,
             panel_source: PanelSource | None = None) -> tuple[list[str], list[dict[str, Any]]]:  # fmt: skip
    """The subcommand: checks, the digest, the contact line, then the run folder, the panel and one answer per seed."""
    t = check_call(args)
    verify_head_digest(args.run, args.sha256)
    seeds = "every seed" if args.seed is None else f"seed {args.seed}"
    years = logged_fit_years(t)
    log_contact(contacts, subcommand="head", arm=f"{HEAD_ARM} {seeds}", series_sha256=args.sha256,
                period=(f"as_of {iso(t)}; the forecasts and labels of fit year {' or '.join(map(str, years))}; "
                        f"memory through as_of"))  # fmt: skip
    run = open_head_run(args.run, args.sha256)
    if panel_source is not None:
        panel, bars = panel_source()
    else:
        panel, bars = registered_panel(run, data_dir=data_dir, universe=universe)
    want = run.report.get("report2", {}).get("panel_fingerprint")
    fp = fingerprint(panel)
    if fp != want:
        raise Refused(f"the panel rebuilt for head has fingerprint {fp}, not Report n°2's {want} that {REPORT_FILE} records")
    data = encoder_data(panel, dict(bars))
    row = _find_row(data, args.pair, t)
    year = int(r3.year_of_month(data.arrays.month_ms[[row]])[0])
    if year not in years:  # before any part is read: the log names the fit year the call reads
        raise Refused(f"the row's fit year {year} is not the one logged for as_of {iso(t)} ({years}): give its bar's open instead")
    prov = head_provenance(run, contacts=contacts, panel_fingerprint=fp)
    results = []
    with deterministic():  # the registered settings, as the fits ran under them, before any device call
        for seed in run.cfg.seeds if args.seed is None else (args.seed,):
            res = explain_seed(run, data, row, int(seed))
            res["provenance"] = prov
            results.append(res)
    return [head_text(r) for r in results], results


# --- Text -------------------------------------------------------------------------------------


def head_text(r: dict[str, Any]) -> str:
    row, fit, fc = r["row"], r["fit"], r["forecast"]
    how = fc["how"]
    lines = [
        f"head {r['arm']} seed {r['seed']}: {row['pair']} decided {row['decision_time']} (bar {row['open_time'][:16]}, month {row['month']})",
        (f"fit {fit['part']} at {fit['fit_time'][:10]}, kept epoch {fit['kept_epoch']}, on {fit['device']}; window "
         f"{row['window']['first_bar'][:16]} -> {row['window']['last_bar'][:16]}"),
        (f"forecast check: raw [{', '.join(f'{x:.6f}' for x in fc['raw'])}] equal to the saved forecast to the bit (batch {how['batch'] + 1} "
         f"of {how['batches']}, {how['eval_batch']} rows each, position {how['position_in_batch']}; {how['queries']} queries, as predict ran them)"),
        (f"part bound to {REPORT_FILE} (digests, rows, kept epoch); attention recomputed to the bit in every block; row sums: "
         f"{r['checks']['row_sums']['rows_checked']} rows within their bounds (fp32 W: largest |sum - 1| {r['checks']['row_sums']['max_dev']:.2g}, "
         f"{r['checks']['row_sums']['max_dev_over_bound']:.2g} of its bound; fp64 W: {r['checks']['row_sums']['fp64_max_dev']:.2g}, "
         f"{r['checks']['row_sums']['fp64_max_dev_over_bound']:.2g} of its bound)"),
        "block.head: lambda, cond, diag mass, self w, eff. rank, row sum, sum|w|, neg mass; top bars (lag h: w)",
    ]
    for h in r["heads"]:
        top = ", ".join(f"{t['lag_hours']}: {t['w']:+.3f}" for t in h["row"]["top"][:4])
        lines.append(f"  {h['block']}.{h['head']}: {h['lambda']:.4g}, {h['condition']:.3g}, {h['diagonal_mass']:.4f}, {h['self_weight']:+.4f}, "
                     f"{h['effective_rank']:.3f}, {h['row_sum']['decision_token']:.9f}, {h['row']['sum_abs']:.3f}, "
                     f"{h['row']['negative_mass']:+.3f}; {top}")  # fmt: skip
    gap = max(h["wv_gap"]["max_abs"] for h in r["heads"])
    w_gap = max(h["w_fp64_gap"] for h in r["heads"])
    lines.append(f"descriptions: W V against the forward pass's O, largest gap {gap:.2g} (the pass never forms W); the fp32 W against W "
                 f"formed in fp64 from the same Q, K and lambda, largest gap {w_gap:.2g}")  # fmt: skip
    p = r["provenance"]
    lines.append(f"provenance: commit {str(p['commit'])[:12]} dirty {p['dirty']}, manifest {str(p['manifest_digest'])[:12]}, panel "
                 f"{p['panel_fingerprint'][:12]} (verified), series {p['series_sha256'][:12]}, run {p['run']}")  # fmt: skip
    lines.append(f"contact logged in {p['contacts_file']}")
    lines.append("EXPLORATORY, test months: never tested as a hypothesis on them")
    return "\n".join(lines)
