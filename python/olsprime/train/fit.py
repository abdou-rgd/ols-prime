"""Training the encoder arms in weekly episodes, and their yearly fits (``docs/PROTOCOL.md``, iteration 4).

A fit at F (00:00 UTC on 1 January of a test year):

- **trains** on the complete rows with a valid window whose labels are known by F - 30 days;
- **stops early** on a slice: the complete rows with a valid window decided in [F - 30 days, F)
  whose labels are known by F - 24 h, never trained on;
- **fits** the inputs' standardization on the training rows, and its climatology on the labelled
  rows known by F - 24 h (as ``climatology-panel``);
- **runs** 6 epochs; an epoch takes every week of training rows once (Monday 00:00 UTC to the
  next), in an order drawn from the seed; after each epoch the slice's mean log-loss (floored
  probabilities) is computed in evaluation mode with the memory as at test time; the fit keeps the
  epoch with the lowest, the first on ties;
- **optimizes** with AdamW (lr 10^-3, betas (0.9, 0.999), weight decay 0.01 on the weight matrices
  only), a linear warm-up over the first 100 steps then a cosine decay to 10^-4 at the end of
  epoch 6, gradient norm clipped at 1;
- **forecasts** every scored row of its year; the memory takes every entry once its label is
  known, test rows included, keyed by the frozen encoder.

One step is one episode: the training rows decided in one week, over every member. Inside it, a
query at decision time d reads exactly the entries with ``d - 30 days < stamp <= d`` of the
training rows: prefix sums over the entries ordered by stamp, so an outcome joins the memory only
once it is known. The episode's own rows are keyed with gradient; the older entries the windows
need are keyed with the current weights without gradient (the cache).

Hooks
-----
:class:`EncoderSystem` exposes where a leak or a control could enter, as ``MatrixSystem`` does.
Nothing here turns a control on; they live in the bench.

- **Stamp.** :meth:`~EncoderSystem.entry_stamps` (``label_known_at``); ``cheat-stamp`` overrides it
  with the decision time, which then rules training, the slice and the forecasts alike.
- **Value path.** ``FitConfig(encoder=EncoderConfig(value_leak=True))`` is ``cheat-value``.
- **Training horizon.** ``FitConfig.train_gap_ms``, any offset from F: 30 days honest, 24 h
  ``cheat-slice``, 0 ``cheat-purge`` (read by :meth:`~EncoderSystem.training_mask`).
- **Label.** :meth:`~EncoderSystem.label_array` gives the label of every row, the one trained on,
  stored in the memory, used by the climatology and scored; ``pos-window`` overrides it with
  :func:`past_window_labels`. Its entries keep the stamp hook's ``label_known_at``.
- Also :meth:`~EncoderSystem.readout_as_of` and :meth:`~EncoderSystem.slice_mask`.
- **Read-only.** :meth:`~EncoderSystem.on_readout` sees the memory's prefix sums and window
  bounds of every evaluation readout (the panel H1 records the window moments); it returns nothing
  and changes nothing.

Every epoch's parameters are kept (:attr:`FitResult.epoch_states`, :meth:`FitResult.epoch_digests`)
beside the kept epoch's.

Bits
----
Forecasts and slice losses (the outputs the panel H1 compares) encode rows in batches of a fixed
size (padded), so a row's state does not depend on how many rows share its batch, and run the
memory's prefix sums sequentially on the CPU (:mod:`olsprime.models.memory`). Training reads
inputs that all precede the fit; its bits on a GPU are the subject of the determinism pre-check.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from olsprime.models.encoder import EncoderConfig, EncoderModel, Standardizer, Windows, build_windows, count_parameters, decay_groups, gather_windows, one_hot
from olsprime.models.matrix import N_CLASSES, PanelArrays, panel_arrays, prob_map, seqsum
from olsprime.models.memory import MemoryConfig, Readout, prefix_moments, readout, window_bounds
from olsprime.train import checkpoint

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
WEEK_MS = 7 * DAY_MS
MONDAY_MS = 4 * DAY_MS  # 1970-01-05 00:00 UTC, a Monday
FIT_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024)
SEEDS: tuple[int, ...] = (1, 2, 3)


def year_start_ms(year: int) -> int:
    return int(datetime(year, 1, 1, tzinfo=UTC).timestamp() * 1000)


def week_start(ms: np.ndarray) -> np.ndarray:
    """The Monday 00:00 UTC at or before each instant."""
    ms = np.asarray(ms, dtype=np.int64)
    return MONDAY_MS + ((ms - MONDAY_MS) // WEEK_MS) * WEEK_MS


@dataclass(frozen=True)
class FitConfig:
    """One fit of one arm with one seed. Frozen and saved with the fit."""

    fit_ms: int
    seed: int
    encoder: EncoderConfig = EncoderConfig()
    memory: MemoryConfig = MemoryConfig()
    epochs: int = 6
    lr: float = 1e-3
    lr_final: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    train_gap_ms: int = 30 * DAY_MS  # the training horizon: labels known by F - train_gap_ms (honest 30 days; cheat-slice 24 h; cheat-purge 0)
    slice_ms: int = 30 * DAY_MS  # the slice: decided in [F - 30 days, F)
    purge_ms: int = 24 * HOUR_MS  # the slice's and the climatology's labels known by F - 24 h
    prob_floor: float = 1e-3
    eval_batch: int = 4096  # the fixed batch of every evaluation (inference batches of 4,096 rows)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def lr_at(step: int, total: int, cfg: FitConfig) -> float:
    """The learning rate of step ``step`` (0-based) of ``total``.

    Linear warm-up ``lr (s + 1) / warmup`` for the first ``warmup`` steps, then a cosine from
    ``lr`` to ``lr_final``, reached at the last step: ``lr_final + (lr - lr_final)(1 + cos(pi r)) / 2``
    with ``r = (s - warmup + 1) / (total - warmup)``.
    """
    w = cfg.warmup_steps
    if step < w:
        return cfg.lr * (step + 1) / w
    if total <= w:
        return cfg.lr
    r = (step - w + 1) / (total - w)
    return cfg.lr_final + (cfg.lr - cfg.lr_final) * 0.5 * (1.0 + math.cos(math.pi * r))


# --- Data -------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EncoderData:
    """The panel as arrays and the windows of its rows, in the panel's order, with each row's label threshold."""

    arrays: PanelArrays
    windows: Windows
    threshold: np.ndarray | None = None  # the panel's ``threshold`` column (NaN where null), for label controls

    @property
    def usable(self) -> np.ndarray:
        """Complete rows with a valid window: the rows that train, key the memory and are forecast by the encoder."""
        return self.arrays.complete & self.windows.ok


def encoder_data(panel: pl.DataFrame, bars: Mapping[str, pl.DataFrame]) -> EncoderData:
    """From the panel (:func:`olsprime.features.panel.assemble`) and the bars its pairs were built from."""
    arrays = panel_arrays(panel)
    windows = build_windows(bars, panel)
    if windows.symbols != arrays.symbols:
        raise ValueError("the windows and the panel disagree on the symbols")
    threshold = panel["threshold"].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float64)
    return EncoderData(arrays, windows, threshold)


def rows_of_year(arrays: PanelArrays, year: int) -> np.ndarray:
    """The complete rows a fit of ``year`` forecasts: a row's year is that of its month, the month of its bar's open."""
    return np.flatnonzero(arrays.complete & (arrays.month_ms >= year_start_ms(year)) & (arrays.month_ms < year_start_ms(year + 1)))


def past_window_labels(data: EncoderData, hours: int = 24) -> np.ndarray:
    """The positive control ``pos-window``'s label: the class of the ``hours`` that end at bar t.

    ``r = ln(c_t / c_{t-24})``, the sequential sum of the window's last 24 one-bar returns, classed
    with the panel's own rule and threshold: 0 if ``r < -threshold``, 2 if ``r > threshold``, 1
    otherwise; -1 where the window fails or the threshold is null. The window holds it, so the
    encoder can learn it. Never used by an honest arm.
    """
    if data.threshold is None:
        raise ValueError("the data carry no threshold")
    w = data.windows
    out = np.full(w.n_rows, -1, dtype=np.int64)
    rows = np.flatnonzero(w.ok & np.isfinite(data.threshold))
    idx = w.end[rows][:, None] + np.arange(-hours + 1, 1)[None, :]
    r = np.add.accumulate(w.bars[idx, 0], axis=1)[:, -1]
    thr = data.threshold[rows]
    out[rows] = np.where(r < -thr, 0, np.where(r > thr, 2, 1))
    return out


# --- Results ----------------------------------------------------------------------------------


@dataclass(eq=False)
class FitResult:
    """Everything a fit computed, its inputs included, for the forecasts and the panel H1."""

    config: FitConfig
    state: dict[str, np.ndarray]  # the kept epoch's parameters
    standardizer: Standardizer
    climatology: np.ndarray
    climatology_rows: int
    train_rows: np.ndarray
    slice_rows: np.ndarray
    complete_without_window: int  # complete rows whose label is known by F - 30 days but whose window failed
    weeks: np.ndarray  # Monday 00:00 of each episode
    orders: list[np.ndarray]  # the episode order of each epoch (indices into weeks)
    slice_losses: list[float]
    kept_epoch: int  # 1-based
    steps: int
    n_parameters: int
    epoch_states: list[dict[str, np.ndarray]] = field(default_factory=list)  # the parameters after each epoch
    build: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)  # not compared: wall times

    def epoch_digests(self) -> list[str]:
        """SHA-256 of every parameter after each epoch (names sorted, dtype, shape and bytes): the purge cut's comparison."""
        return [state_digest(s) for s in self.epoch_states]

    def save(self, folder: Path) -> str:
        """The kept parameters and the fit's inputs as a checkpoint, with the config as metadata; returns its SHA-256."""
        arrays = {f"param.{k}": v for k, v in self.state.items()}
        arrays |= {"std.mean": self.standardizer.mean, "std.std": self.standardizer.std, "climatology": self.climatology,
                   "train_rows": self.train_rows, "slice_rows": self.slice_rows, "weeks": self.weeks,
                   "slice_losses": np.asarray(self.slice_losses, dtype=np.float64)}  # fmt: skip
        arrays |= {f"order.{i}": o.astype(np.int64) for i, o in enumerate(self.orders)}
        arrays |= {f"epoch{i + 1}.{k}": v for i, s in enumerate(self.epoch_states) for k, v in s.items()}
        meta = {"config": self.config.to_json(), "kept_epoch": str(self.kept_epoch), "steps": str(self.steps),
                "epoch_digests": json.dumps(self.epoch_digests()),
                "n_parameters": str(self.n_parameters), "standardizer_rows": str(self.standardizer.rows),
                "climatology_rows": str(self.climatology_rows), "complete_without_window": str(self.complete_without_window),
                "build": json.dumps(self.build, sort_keys=True)}  # fmt: skip
        folder.mkdir(parents=True, exist_ok=True)
        return checkpoint.save(folder / "fit.safetensors", arrays, meta)


def state_digest(state: Mapping[str, np.ndarray]) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(state):
        a = np.ascontiguousarray(state[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_arrays(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().to("cpu").numpy().copy() for k, v in model.state_dict().items()}


def load_state(model: torch.nn.Module, state: Mapping[str, np.ndarray]) -> None:
    model.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in state.items()})


# --- The system -------------------------------------------------------------------------------


class EncoderSystem:
    """One arm's fit at F over a panel, and its forecasts."""

    name = "honest"

    def __init__(self, data: EncoderData, cfg: FitConfig, device: str | torch.device = "cpu") -> None:
        self.data = data
        self.cfg = cfg
        self.device = torch.device(device)
        self.y = np.asarray(self.label_array(), dtype=np.int64)
        if self.y.shape != (data.arrays.n,):
            raise ValueError("label_array must give one label per panel row")
        if np.any((self.y < 0)[data.usable]) or np.any(self.y >= N_CLASSES):
            raise ValueError("every usable row needs a label in {0, 1, 2}")

    # --- hooks: the honest definitions (the bench plants its leaks and controls by overriding them) ---

    def label_array(self) -> np.ndarray:
        """The label of every panel row, the one the fit trains on, the memory stores and the scores read
        (-1 where there is none): the panel's forward label. The positive control ``pos-window``
        replaces it with the class of the 24 hours that end at bar t (:func:`past_window_labels`)."""
        return self.data.arrays.label

    def entry_stamps(self, rows: np.ndarray) -> np.ndarray:
        """The stamp of each memory entry: its ``label_known_at``."""
        return self.data.arrays.lka_ms[rows]

    def readout_as_of(self, rows: np.ndarray) -> np.ndarray:
        """The instant a row's memory readout reads up to: its decision time."""
        return self.data.arrays.decision_ms[rows]

    def on_readout(self, prefix: Any, lo: torch.Tensor, hi: torch.Tensor) -> None:
        """Read-only: called by :meth:`predict` with the prefix sums and window bounds of its queries. Honest: nothing."""

    def training_mask(self) -> np.ndarray:
        """The training rows: usable, label known by the training horizon ``F - train_gap_ms`` (30 days).

        Any offset from F: ``cheat-slice`` sets ``train_gap_ms = 24 h`` (its own slice trained on),
        ``cheat-purge`` sets ``train_gap_ms = 0`` (labels known by F, no purge).
        """
        a = self.data.arrays
        return self.data.usable & (a.lka_ms <= self.cfg.fit_ms - self.cfg.train_gap_ms)

    def slice_mask(self) -> np.ndarray:
        """The slice: usable, decided in [F - 30 days, F), label known by F - 24 h."""
        a, f = self.data.arrays, self.cfg.fit_ms
        return self.data.usable & (a.decision_ms >= f - self.cfg.slice_ms) & (a.decision_ms < f) & (a.lka_ms <= f - self.cfg.purge_ms)

    def climatology(self) -> tuple[np.ndarray, int]:
        """Class frequencies of the labelled rows known by F - 24 h, through the probability floor."""
        a = self.data.arrays
        rows = a.labelled & (self.y >= 0) & (a.lka_ms <= self.cfg.fit_ms - self.cfg.purge_ms)
        n = int(rows.sum())
        if n == 0:
            return np.full(N_CLASSES, 1.0 / N_CLASSES), 0
        counts = np.bincount(self.y[rows], minlength=N_CLASSES).astype(np.float64)
        return prob_map(counts / float(n), self.cfg.prob_floor), n

    # --- pieces ---

    def new_model(self) -> EncoderModel:
        torch.manual_seed(self.cfg.seed)
        return EncoderModel(self.cfg.encoder).to(self.device)

    def standardized_bars(self, std: Standardizer) -> torch.Tensor:
        return torch.from_numpy(std.apply(self.data.windows.bars)).to(self.device)

    def _labels(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self.y[rows]).to(self.device)

    def _leak(self, rows: np.ndarray) -> torch.Tensor | None:
        return one_hot(self._labels(rows)) if self.cfg.encoder.value_leak else None

    def _encode(self, model: EncoderModel, bars: torch.Tensor, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        ends = torch.from_numpy(self.data.windows.end[rows]).to(self.device)
        return model.encode(gather_windows(bars, ends), self._leak(rows))

    def sorted_pool(self, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The pool's rows in Matrix's push order (stamp, symbol, open time) and their stamps."""
        a = self.data.arrays
        stamps = self.entry_stamps(pool).astype(np.int64)
        order = np.lexsort((a.open_ms[pool], a.symbol[pool], stamps))
        return pool[order], np.ascontiguousarray(stamps[order])

    def _entries_for(self, stamps_sorted: np.ndarray, as_of: np.ndarray) -> tuple[int, int]:
        """The slice of the sorted pool that the windows of these decisions can reach."""
        w = self.cfg.memory.window_ms
        return int(np.searchsorted(stamps_sorted, as_of.min() - w, side="right")), int(np.searchsorted(stamps_sorted, as_of.max(), side="right"))

    # --- training ---

    def episode_loss(self, model: EncoderModel, bars: torch.Tensor, q_rows: np.ndarray, pool_rows: np.ndarray, pool_stamps: np.ndarray) -> torch.Tensor:
        """The mean cross-entropy of one episode, with its memory as registered."""
        a, mem = self.data.arrays, self.cfg.memory
        as_of = self.readout_as_of(q_rows).astype(np.int64)
        h_q, z_q = self._encode(model, bars, q_rows)
        lo, hi = self._entries_for(pool_stamps, as_of)
        e_rows, e_stamps = pool_rows[lo:hi], pool_stamps[lo:hi]
        pos = np.searchsorted(q_rows, e_rows)
        in_ep = (pos < q_rows.size) & (q_rows[np.minimum(pos, q_rows.size - 1)] == e_rows)
        cache_rows = e_rows[~in_ep]
        z_parts, idx = [], np.empty(e_rows.size, dtype=np.int64)
        n_cache = cache_rows.size
        if n_cache:
            # The cache: keyed with the current weights, no gradient, in fixed batches in panel order.
            cache_sorted = np.sort(cache_rows)
            z_parts.append(self._encode_fixed(model, bars, cache_sorted)[1])
            idx[~in_ep] = np.searchsorted(cache_sorted, cache_rows)
        z_parts.append(z_q[torch.from_numpy(pos[in_ep]).to(self.device)])
        idx[in_ep] = n_cache + np.arange(int(in_ep.sum()))
        z_e = torch.cat(z_parts)[torch.from_numpy(idx).to(self.device)]
        y_e = one_hot(self._labels(e_rows), dtype=torch.float64)
        prefix = prefix_moments(z_e, y_e, causal=False)
        stamps_t = torch.from_numpy(e_stamps).to(self.device)
        lo_q, hi_q = window_bounds(stamps_t, torch.from_numpy(as_of).to(self.device), mem.window_ms)
        m = readout(prefix, lo_q, hi_q, z_q, mem).m.to(h_q.dtype)
        logits = model.decide(h_q, m)
        return F.cross_entropy(logits, self._labels(q_rows))

    def fit(self, *, on_epoch: Any = None) -> FitResult:
        cfg, a = self.cfg, self.data.arrays
        started = time.perf_counter()
        train_mask = self.training_mask()
        train_rows = np.flatnonzero(train_mask)
        slice_rows = np.flatnonzero(self.slice_mask())
        if train_rows.size == 0 or slice_rows.size == 0:
            raise ValueError(f"the fit at {_iso(cfg.fit_ms)} has {train_rows.size} training rows and {slice_rows.size} slice rows")
        no_window = int((a.complete & ~self.data.windows.ok & (a.lka_ms <= cfg.fit_ms - cfg.train_gap_ms)).sum())
        std = Standardizer.fit(self.data.windows, train_rows)
        clim, clim_rows = self.climatology()
        bars = self.standardized_bars(std)
        model = self.new_model()
        n_params = count_parameters(model)
        opt = torch.optim.AdamW(decay_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
        week_of = week_start(a.decision_ms[train_rows])
        weeks = np.unique(week_of)
        episodes = [train_rows[week_of == w] for w in weeks]
        pool_rows, pool_stamps = self.sorted_pool(train_rows)
        rng = np.random.default_rng(cfg.seed)
        total = cfg.epochs * weeks.size
        orders, losses, step = [], [], 0
        best_loss, best_epoch, epoch_states = math.inf, 0, []
        timing = {"train_s": 0.0, "slice_s": 0.0}
        for epoch in range(1, cfg.epochs + 1):
            order = rng.permutation(weeks.size)
            orders.append(order)
            model.train()
            t0 = time.perf_counter()
            for wi in order:
                for group in opt.param_groups:
                    group["lr"] = lr_at(step, total, cfg)
                loss = self.episode_loss(model, bars, episodes[wi], pool_rows, pool_stamps)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                step += 1
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timing["train_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            out = self.predict(model, bars, slice_rows)
            slice_loss = float(seqsum(-np.log(out["probs"][np.arange(slice_rows.size), self.y[slice_rows]])) / slice_rows.size)
            timing["slice_s"] += time.perf_counter() - t0
            losses.append(slice_loss)
            epoch_states.append(state_arrays(model))
            if slice_loss < best_loss:
                best_loss, best_epoch = slice_loss, epoch
            if on_epoch is not None:
                on_epoch(epoch, slice_loss, time.perf_counter() - started)
        timing["fit_s"] = time.perf_counter() - started
        from olsprime.train.determinism import build

        return FitResult(cfg, epoch_states[best_epoch - 1], std, clim, clim_rows, train_rows, slice_rows, no_window, weeks, orders, losses,
                         best_epoch, step, n_params, epoch_states, build(self.device), timing)  # fmt: skip

    # --- evaluation and forecasts ---

    def model_from(self, result: FitResult) -> EncoderModel:
        model = EncoderModel(result.config.encoder)
        load_state(model, result.state)
        return model.to(self.device).eval()

    @torch.no_grad()
    def _encode_fixed(self, model: EncoderModel, bars: torch.Tensor, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """``(h, z)`` of ``rows`` (panel order) in batches of exactly ``eval_batch`` rows.

        The last batch is padded with copies of its last row (``docs/PROTOCOL.md``, "Batches"), so
        BLAS sees one problem size and a row's bits do not depend on how many rows the run holds.
        """
        if rows.size and np.any(np.diff(rows) <= 0):
            raise ValueError("fixed batches take rows in panel order, without repeats")
        b = self.cfg.eval_batch
        hs, zs = [], []
        for i in range(0, rows.size, b):
            chunk = rows[i : i + b]
            padded = np.concatenate([chunk, np.full(b - chunk.size, chunk[-1], dtype=chunk.dtype)])
            h, z = self._encode(model, bars, padded)
            hs.append(h[: chunk.size])
            zs.append(z[: chunk.size])
        if not hs:
            d, k = self.cfg.encoder.d_model, self.cfg.encoder.key_dim
            return torch.zeros(0, d, device=self.device), torch.zeros(0, k, device=self.device)
        return torch.cat(hs), torch.cat(zs)

    @torch.no_grad()
    def _readout_fixed(self, prefix: Any, lo: torch.Tensor, hi: torch.Tensor, z_q: torch.Tensor) -> Readout:
        """The memory's readouts in fixed batches of ``eval_batch`` queries, the last padded with copies of its last query."""
        b, n = self.cfg.eval_batch, int(lo.shape[0])
        parts = []
        for i in range(0, n, b):
            k = min(b, n - i)
            pad = torch.full((b,), i + k - 1, dtype=torch.long)
            pad[:k] = torch.arange(i, i + k)
            r = readout(prefix, lo[pad], hi[pad], z_q[pad], self.cfg.memory)
            parts.append([t[:k] for t in (r.m, r.count, r.mean_z, r.mean_y, r.beta)])
        if not parts:
            p, q = self.cfg.memory.key_dim, self.cfg.memory.n_classes
            f64 = torch.float64
            return Readout(torch.zeros(0, q, dtype=f64), torch.zeros(0, dtype=f64), torch.zeros(0, p, dtype=f64), torch.zeros(0, q, dtype=f64), torch.zeros(0, q, p, dtype=f64))
        return Readout(*(torch.cat(cols) for cols in zip(*parts, strict=True)))

    @torch.no_grad()
    def _decide_fixed(self, model: EncoderModel, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        b = self.cfg.eval_batch
        out = []
        for i in range(0, h.shape[0], b):
            n = min(b, h.shape[0] - i)
            pad = torch.full((b,), i + n - 1, dtype=torch.long, device=self.device)
            pad[:n] = torch.arange(i, i + n, device=self.device)
            out.append(torch.softmax(model.decide(h[pad], m[pad]), dim=-1)[:n])
        return torch.cat(out) if out else torch.zeros(0, self.cfg.encoder.n_classes, device=self.device)

    @torch.no_grad()
    def predict(self, model: EncoderModel, bars: torch.Tensor, rows: np.ndarray, *, pool: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Forecasts of usable ``rows`` (panel order) with the memory as at test time.

        The memory holds every usable row (``pool``, default all) once its stamp is reached, keyed by
        the frozen model. Returns the probabilities before the floor (``raw``) and after it
        (``probs``), and per query the memory's count, mean key, mean label and beta, and the
        window's entry range in the sorted pool.
        """
        model.eval()
        rows = np.asarray(rows, dtype=np.int64)
        if not self.data.usable[rows].all():
            raise ValueError("predict takes usable rows; a failed window takes the climatology instead")
        if rows.size and np.any(np.diff(rows) <= 0):
            raise ValueError("rows must be in panel order, without repeats")
        mem = self.cfg.memory
        pool = np.flatnonzero(self.data.usable) if pool is None else pool
        pool_rows, pool_stamps = self.sorted_pool(pool)
        as_of = self.readout_as_of(rows).astype(np.int64)
        lo, hi = self._entries_for(pool_stamps, as_of) if rows.size else (0, 0)
        e_rows, e_stamps = pool_rows[lo:hi], pool_stamps[lo:hi]
        union = np.union1d(rows, e_rows)
        h_all, z_all = self._encode_fixed(model, bars, union)
        z_cpu = z_all.to("cpu", torch.float64)
        q_at = torch.from_numpy(np.searchsorted(union, rows))
        e_at = torch.from_numpy(np.searchsorted(union, e_rows))
        y_e = one_hot(torch.from_numpy(self.y[e_rows]), dtype=torch.float64)
        prefix = prefix_moments(z_cpu[e_at], y_e, causal=True)
        lo_q, hi_q = window_bounds(torch.from_numpy(e_stamps), torch.from_numpy(as_of), mem.window_ms)
        self.on_readout(prefix, lo_q, hi_q)
        ro = self._readout_fixed(prefix, lo_q, hi_q, z_cpu[q_at])
        h_q = h_all[q_at.to(self.device)]
        raw = self._decide_fixed(model, h_q, ro.m.to(self.device, h_q.dtype)).to("cpu", torch.float64).numpy()
        return {
            "rows": rows,
            "raw": raw,
            "probs": prob_map(raw, self.cfg.prob_floor),
            "m": ro.m.numpy(),
            "count": ro.count.numpy(),
            "mean_z": ro.mean_z.numpy(),
            "mean_y": ro.mean_y.numpy(),
            "beta": ro.beta.numpy(),
            "window_lo": lo_q.numpy() + lo,
            "window_hi": hi_q.numpy() + lo,
        }

    def forecast(self, result: FitResult, rows: np.ndarray) -> dict[str, np.ndarray]:
        """The arm's forecasts of ``rows`` with the fit's frozen encoder; a failed window takes the climatology."""
        rows = np.asarray(rows, dtype=np.int64)
        model = self.model_from(result)
        bars = self.standardized_bars(result.standardizer)
        ok = self.data.usable[rows]
        out = self.predict(model, bars, rows[ok])
        raw = np.tile(result.climatology, (rows.size, 1))
        raw[ok] = out["raw"]
        return {**out, "rows": rows, "raw": raw, "probs": prob_map(raw, self.cfg.prob_floor), "fallback": ~ok}


def ensemble(raws: Sequence[np.ndarray], floor: float = 1e-3) -> np.ndarray:
    """The arm's forecast: the mean of its seeds' probabilities before the floor, then floored."""
    stack = np.stack(raws)
    mean = np.asarray(seqsum(stack, axis=0)) / float(len(raws))
    return prob_map(mean, floor)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="milliseconds")
