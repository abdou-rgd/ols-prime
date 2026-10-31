"""The encoder of iteration 4 (``docs/PROTOCOL.md``, "The encoder, and what a verdict is worth", "The encoder arms").

Three arms share every weight but the attention operator:

============== ==============================================================================
``enc-ols``     W = Q~ (K~^T K~ + Lambda)^-1 K~^T and O = W V (``docs/DESIGN.md`` §1 and §4)
``enc-softmax`` softmax(Q K^T / sqrt 8) V, PyTorch's scaled dot-product attention (math backend)
``enc-linear``  phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1), phi(x) = elu(x) + 1
============== ==============================================================================

Inputs: a window of bars
------------------------
Row (s, t) reads the 64 hourly bars of pair s that end with bar t. Each bar j gives eight
quantities (:data:`QUANTITIES`): ``ret = ln(c_j / c_{j-1})``, ``gap = ln(o_j / c_{j-1})``,
``range = ln(h_j / l_j)``, ``body = ln(c_j / o_j)``, ``upper = ln(h_j / max(o_j, c_j))``,
``lower = ln(min(o_j, c_j) / l_j)``, ``ln(1 + quote volume)`` and the taker share of the quote
volume minus 1/2 (0 for a bar without volume). The window's 65 bars, from t - 64 for the first
return, must be consecutive hours, and every quantity of its 64 bars finite; otherwise the window
is flagged (:attr:`Windows.ok`), never filled. :func:`build_windows` reads the same bars that
:func:`olsprime.features.panel.pair_rows` receives, cut the same way, so the panel H1 rebuilds the
windows from perturbed bars with the same call.

Each quantity is standardized with its mean and standard deviation over the last bar of every
training row of the fit, then clipped to +-8 (:class:`Standardizer`). The statistics are a
function of those rows' bars alone: sequential fp64 sums in panel order.

Encoder
-------
A linear map of the eight quantities to d_model = 32 plus a learned embedding of the 64
positions; 2 pre-LayerNorm blocks (attention with 4 heads of d_head = 8, with Q, K, V and output
projections; an MLP 32 -> 64 -> 32 with GELU; a residual around each; no dropout); a final
LayerNorm. The state h is the output at the last position (bar t). The memory's key is
``z = W_z h`` (8, no bias: the memory's ridge is centred, so a bias would cancel) and the decision
head maps ``[h, m]`` (35) -> 32 -> GELU -> 3 logits (:class:`EncoderModel`).

The OLS head
------------
Queries and keys are scaled to unit root mean square at each position, with no parameter.
K~ = [1 | K] and q~ = [1 | q]; Lambda = diag(0, T lambda, ..., T lambda), lambda = softplus(theta)
+ 10^-3 per head, theta initialised so that lambda = 0.1. ``A = K~^T K~ + Lambda`` is factorized by
Cholesky in fp32 (never inverted). The forward pass evaluates ``O = Q~ (A^-1 (K~^T V))``, which is
``W V`` reassociated (:data:`OLS_ORDERS`); :func:`ols_weights` gives W itself for ``head``, whose
rows sum to 1 (``docs/DESIGN.md`` §1) within a rounding bound (:func:`row_sum_tolerance`).

Hooks
-----
``value_leak`` (:class:`EncoderConfig`) is the planted ``cheat-value`` control of the panel H1: a
learned projection of the row's own one-hot label, added to the value of the last position in
every head. It is off by default and nothing in ``olsprime.models`` turns it on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from olsprime.models.matrix import seqsum
from olsprime.protocol import DEVELOPMENT

HOUR_MS = 3_600_000
WINDOW = 64  # bars per window: t - 63 .. t
QUANTITIES: tuple[str, ...] = ("ret", "gap", "range", "body", "upper", "lower", "log1p_quote_volume", "taker_share")
N_QUANTITIES = len(QUANTITIES)
CLIP = 8.0

ARMS: tuple[str, ...] = ("enc-ols", "enc-softmax", "enc-linear")
ATTENTION_OF_ARM: Mapping[str, str] = {"enc-ols": "ols", "enc-softmax": "softmax", "enc-linear": "linear"}
# How the OLS head evaluates O = W V: "assoc" is Q~ (A^-1 (K~^T V)), O(T d^2); "explicit" forms W
# (T x T) and then W V, O(T^2 d). Same function, different rounding and cost.
OLS_ORDERS: tuple[str, ...] = ("assoc", "explicit")
_BAR_COLUMNS = ("open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume", "known_at")


# --- Windows ---------------------------------------------------------------------------------


def bar_quantities(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, quote: np.ndarray, taker_quote: np.ndarray) -> np.ndarray:
    """The eight quantities of every bar, (n, 8) float64, in the order of :data:`QUANTITIES`.

    ``ret`` and ``gap`` of bar j use the close of row j - 1 whatever its time (the window's grid
    check decides whether that row is bar j - 1); they are NaN for the first row. A quantity whose
    logarithm is undefined (a price <= 0) is NaN, never filled.
    """
    o, h, lo, c = (np.asarray(a, dtype=np.float64) for a in (open_, high, low, close))
    q, b = np.asarray(quote, dtype=np.float64), np.asarray(taker_quote, dtype=np.float64)
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(all="ignore"):
        out = np.stack(
            [
                np.log(c / prev),
                np.log(o / prev),
                np.log(h / lo),
                np.log(c / o),
                np.log(h / np.maximum(o, c)),
                np.log(np.minimum(o, c) / lo),
                np.log1p(q),
                np.where(q > 0, b / np.where(q > 0, q, 1.0) - 0.5, 0.0),
            ],
            axis=1,
        )
    return np.where(np.isfinite(out), out, np.nan)


def window_flags(open_ms: np.ndarray, quantities: np.ndarray, *, window: int = WINDOW, step_ms: int = HOUR_MS) -> np.ndarray:
    """``ok[j]``: rows j - window .. j are consecutive bars ``step_ms`` apart, and every quantity of rows j - window + 1 .. j is finite."""
    n = len(open_ms)
    ok = np.zeros(n, dtype=bool)
    if n <= window:
        return ok
    on_grid = np.zeros(n, dtype=np.int64)
    on_grid[1:] = np.diff(open_ms) == step_ms
    grid = np.concatenate([[0], np.cumsum(on_grid)])  # grid[j + 1] = gaps on the grid among rows 1..j
    finite = np.concatenate([[0], np.cumsum(np.isfinite(quantities).all(axis=1).astype(np.int64))])
    j = np.arange(window, n)
    ok[window:] = (grid[j + 1] - grid[j + 1 - window] == window) & (finite[j + 1] - finite[j + 1 - window] == window)
    return ok


@dataclass(frozen=True, eq=False)
class Windows:
    """Every bar the panel's windows read, and each panel row's window.

    ``bars`` (n_bars, 8) float64 are the raw quantities of every pair, pair after pair in the
    order of ``symbols``; ``end`` (n_rows,) is the global index of bar t of each panel row (-1 when
    the row's bar is not among the bars); ``ok`` flags the rows whose window is valid. A row's
    window is ``bars[end - 63 : end + 1]``.
    """

    symbols: tuple[str, ...]
    bars: np.ndarray
    open_ms: np.ndarray
    end: np.ndarray
    ok: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.end.size)

    def failed(self, rows: np.ndarray) -> int:
        """How many of ``rows`` have no valid window (each takes the arm's climatology)."""
        return int((~self.ok[rows]).sum())


def build_windows(bars: Mapping[str, pl.DataFrame], panel: pl.DataFrame, *, window: int = WINDOW) -> Windows:
    """The windows of every row of ``panel`` from the bars of its pairs.

    ``bars[symbol]`` is one pair's hourly bars as :func:`olsprime.protocol.load_klines` returns
    them, the frame :func:`olsprime.features.panel.pair_rows` receives. They are cut to
    ``known_at < DEVELOPMENT.end`` here, as ``pair_rows`` cuts them, and must be strictly
    increasing in ``open_time``. Each pair keeps only the bars its rows' windows can reach.
    """
    symbols = tuple(sorted(panel["symbol"].unique().to_list()))
    missing = [s for s in symbols if s not in bars]
    if missing:
        raise ValueError(f"no bars for {missing}")
    row_symbol = panel["symbol"].to_numpy()
    row_open = panel["open_time"].dt.epoch("ms").to_numpy().astype(np.int64)
    end = np.full(panel.height, -1, dtype=np.int64)
    ok = np.zeros(panel.height, dtype=bool)
    parts, opens, offset = [], [], 0
    for symbol in symbols:
        frame = bars[symbol].select(_BAR_COLUMNS).filter(pl.col("known_at") < DEVELOPMENT.end)
        open_ms = frame["open_time"].dt.epoch("ms").to_numpy().astype(np.int64)
        if open_ms.size > 1 and not np.all(np.diff(open_ms) > 0):
            raise ValueError(f"{symbol}: open_time is not strictly increasing")
        rows = np.flatnonzero(row_symbol == symbol)
        if open_ms.size == 0:
            continue
        at = np.searchsorted(open_ms, row_open[rows])
        found = (at < open_ms.size) & (open_ms[np.minimum(at, open_ms.size - 1)] == row_open[rows])
        if not found.any():
            continue
        lo, hi = max(int(at[found].min()) - window, 0), int(at[found].max()) + 1
        cols = [frame[c].to_numpy()[lo:hi] for c in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")]
        q = bar_quantities(*cols)
        flags = window_flags(open_ms[lo:hi], q, window=window)
        local = at[found] - lo
        end[rows[found]] = offset + local
        ok[rows[found]] = flags[local]
        parts.append(q)
        opens.append(open_ms[lo:hi])
        offset += hi - lo
    all_bars = np.concatenate(parts) if parts else np.zeros((0, N_QUANTITIES))
    all_open = np.concatenate(opens) if opens else np.zeros(0, dtype=np.int64)
    return Windows(symbols, np.ascontiguousarray(all_bars), all_open, end, ok)


@dataclass(frozen=True, eq=False)
class Standardizer:
    """Mean and standard deviation (divisor n) of each quantity over the last bar of the fit's training rows."""

    mean: np.ndarray
    std: np.ndarray
    rows: int

    @staticmethod
    def fit(windows: Windows, rows: np.ndarray) -> Standardizer:
        """From the rows given (panel order), by sequential fp64 sums: a function of their last bars alone."""
        rows = np.asarray(rows)
        if rows.size == 0:
            raise ValueError("no training row to standardize on")
        if not windows.ok[rows].all():
            raise ValueError("a training row has no valid window")
        last = windows.bars[windows.end[rows]]
        n = float(rows.size)
        mean = np.asarray(seqsum(last, axis=0)) / n
        var = np.asarray(seqsum((last - mean) ** 2, axis=0)) / n
        std = np.sqrt(var)
        if not np.all(std > 0):
            bad = [QUANTITIES[j] for j in np.flatnonzero(~(std > 0))]
            raise ValueError(f"a quantity is constant over the training rows: {bad}")
        return Standardizer(mean, std, int(rows.size))

    def apply(self, bars: np.ndarray) -> np.ndarray:
        """``clip((bars - mean) / std, -8, 8)`` as float32; a NaN stays NaN (its window is flagged)."""
        return np.clip((bars - self.mean) / self.std, -CLIP, CLIP).astype(np.float32)


def gather_windows(std_bars: torch.Tensor, ends: torch.Tensor, window: int = WINDOW) -> torch.Tensor:
    """``(B, window, 8)``: rows ``end - window + 1 .. end`` of the standardized bars, for each end."""
    offsets = torch.arange(-window + 1, 1, device=ends.device, dtype=ends.dtype)
    return std_bars[ends[:, None] + offsets[None, :]]


# --- The attention operators ----------------------------------------------------------------


def rms_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale each vector (last axis) to unit root mean square; no parameter."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _with_constant(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones_like(x[..., :1]), x], dim=-1)


def ols_penalty(lam: torch.Tensor, d_head: int, t: int) -> torch.Tensor:
    """``diag(0, T lambda, ..., T lambda)`` per head, ``(H, d + 1, d + 1)``: the constant coordinate is free."""
    diag = torch.cat([torch.zeros_like(lam)[:, None], (t * lam)[:, None].expand(-1, d_head)], dim=1)
    return torch.diag_embed(diag)


def _ols_factor(q: torch.Tensor, k: torch.Tensor, lam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(Q~, K~, L)`` with ``L L^T = K~^T K~ + Lambda``; q, k are ``(B, H, T, d)``, lam ``(H,)``."""
    t, d = k.shape[-2], k.shape[-1]
    qt, kt = _with_constant(rms_normalize(q)), _with_constant(rms_normalize(k))
    gram = kt.transpose(-1, -2) @ kt + ols_penalty(lam.to(kt.dtype), d, t)
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        raise FloatingPointError("the OLS head's K~^T K~ + Lambda is not positive definite in its compute precision")
    return qt, kt, chol


def ols_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lam: torch.Tensor, *, order: str = "assoc") -> torch.Tensor:
    """The OLS head: ``O = W V`` with ``W = Q~ (K~^T K~ + Lambda)^-1 K~^T``, solved by Cholesky in the inputs' precision.

    ``q``, ``k``, ``v`` are ``(B, H, T, d)``, ``lam`` is ``(H,)``. ``order`` picks the evaluation
    (:data:`OLS_ORDERS`); the explicit inverse is never formed.
    """
    if order not in OLS_ORDERS:
        raise ValueError(f"unknown OLS order {order!r}")
    qt, kt, chol = _ols_factor(q, k, lam)
    if order == "assoc":
        return qt @ torch.cholesky_solve(kt.transpose(-1, -2) @ v, chol)
    return (qt @ torch.cholesky_solve(kt.transpose(-1, -2), chol)) @ v


def ols_weights(q: torch.Tensor, k: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """W = Q~ (K~^T K~ + Lambda)^-1 K~^T, ``(B, H, T, T)``: the signed attention weights of the OLS head."""
    qt, kt, chol = _ols_factor(q, k, lam)
    return qt @ torch.cholesky_solve(kt.transpose(-1, -2), chol)


def ols_condition(k: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """The 2-norm condition number of ``K~^T K~ + Lambda`` per (batch, head), in float64."""
    t, d = k.shape[-2], k.shape[-1]
    kt = _with_constant(rms_normalize(k.double()))
    gram = kt.transpose(-1, -2) @ kt + ols_penalty(lam.double(), d, t)
    ev = torch.linalg.eigvalsh(gram)
    return ev[..., -1] / ev[..., 0]


def row_sum_tolerance(w: torch.Tensor, k: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """A first-order bound on ``|sum_j W_ij - 1|`` for W computed in the precision of ``w``.

    Forming ``A = K~^T K~ + Lambda`` errs by ``gamma_T |K~|^T |K~|``; the Cholesky solve is backward
    stable with ``gamma_{3(d+1)}``; the products ``Q~ M`` add ``gamma_{d+1}``. So each weight errs
    by about ``kappa(A) (T + 3(d + 1)) u |W_ij| + (d + 1) u |W_ij|``, and the row sum by that times
    ``sum_j |W_ij|`` plus the T additions of the sum, ``T u sum_j |W_ij|``. Returned with a factor 2:
    ``2 u (kappa (T + 3(d+1)) + (d + 1) + T) sum_j |W_ij|``, ``(B, H, T)``.
    """
    t, d = k.shape[-2], k.shape[-1]
    u = torch.finfo(w.dtype).eps / 2
    kappa = ols_condition(k, lam).to(w.dtype)[..., None]
    return 2 * u * (kappa * (t + 3 * (d + 1)) + (d + 1) + t) * w.abs().sum(dim=-1)


def softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``softmax(Q K^T / sqrt d) V`` by PyTorch's scaled dot-product attention, on its math backend (no fused kernel)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v)


def elu_feature(x: torch.Tensor) -> torch.Tensor:
    """phi(x) = elu(x) + 1 (Katharopoulos et al., 2020)."""
    return F.elu(x) + 1.0


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``phi(Q) (phi(K)^T V) / (phi(Q) phi(K)^T 1)``, bidirectional; phi > 0, so the denominator is positive."""
    fq, fk = elu_feature(q), elu_feature(k)
    kv = fk.transpose(-1, -2) @ v  # (B, H, d, d)
    norm = fq @ fk.sum(dim=-2, keepdim=True).transpose(-1, -2)  # (B, H, T, 1)
    return (fq @ kv) / norm


# --- The modules ----------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderConfig:
    """Everything the encoder fixes. Saved with every fit (``dataclasses.asdict``)."""

    arm: str = "enc-ols"
    window: int = WINDOW
    n_quantities: int = N_QUANTITIES
    d_model: int = 32
    n_heads: int = 4
    d_head: int = 8
    n_layers: int = 2
    d_mlp: int = 64
    key_dim: int = 8
    head_hidden: int = 32
    n_classes: int = 3
    lambda_init: float = 0.1
    lambda_floor: float = 1e-3
    min_window_per_head: int = 8  # T / d_head; below it K (K^T K)^-1 K^T ~ I and the head copies its input
    ols_order: str = "assoc"
    value_leak: bool = False  # cheat-value: a planted control of the panel H1, never an honest arm

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; the arms are {ARMS}")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model")
        if self.ols_order not in OLS_ORDERS:
            raise ValueError(f"unknown OLS order {self.ols_order!r}")
        if self.arm == "enc-ols" and self.window < self.min_window_per_head * self.d_head:
            raise ValueError(
                f"T = {self.window} is not >> d_head = {self.d_head} (T / d_head < {self.min_window_per_head}): "
                "the OLS head would collapse towards the identity"
            )
        if not (self.lambda_init > self.lambda_floor > 0):
            raise ValueError("lambda_init must exceed lambda_floor > 0")

    @property
    def attention(self) -> str:
        return ATTENTION_OF_ARM[self.arm]


def inverse_softplus(y: float) -> float:
    """theta with softplus(theta) = y, for y > 0."""
    return y + math.log(-math.expm1(-y))


class Attention(nn.Module):
    """Multi-head attention with the arm's operator; Q, K, V and the output projection are shared weights."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        if cfg.attention == "ols":
            theta = inverse_softplus(cfg.lambda_init - cfg.lambda_floor)
            self.theta = nn.Parameter(torch.full((cfg.n_heads,), theta))
        if cfg.value_leak:
            self.leak = nn.Linear(cfg.n_classes, cfg.d_model, bias=False)

    def lam(self) -> torch.Tensor:
        """lambda = softplus(theta) + floor, one per head."""
        return F.softplus(self.theta) + self.cfg.lambda_floor

    def project(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        h, d = self.cfg.n_heads, self.cfg.d_head
        q, k, v = self.qkv(x).view(b, t, 3, h, d).permute(2, 0, 3, 1, 4)  # each (B, H, T, d)
        if self.cfg.value_leak:
            if leak_labels is None:
                raise ValueError("value_leak needs the rows' one-hot labels")
            v = v.clone()
            v[:, :, -1, :] = v[:, :, -1, :] + self.leak(leak_labels).view(b, h, d)
        elif leak_labels is not None:
            raise ValueError("labels reach the encoder only in the planted cheat-value control")
        return q, k, v

    def forward(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self.project(x, leak_labels)
        kind = self.cfg.attention
        if kind == "ols":
            o = ols_attention(q, k, v, self.lam(), order=self.cfg.ols_order)
        elif kind == "softmax":
            o = softmax_attention(q, k, v)
        else:
            o = linear_attention(q, k, v)
        b, _, t, _ = o.shape
        return self.out(o.transpose(1, 2).reshape(b, t, self.cfg.d_model))

    def weights(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> torch.Tensor:
        """The OLS head's W, ``(B, H, T, T)``, for ``head``."""
        if self.cfg.attention != "ols":
            raise ValueError("only the OLS head exposes its W")
        q, k, _ = self.project(x, leak_labels)
        return ols_weights(q, k, self.lam())


class Block(nn.Module):
    """Pre-LayerNorm: x + attention(LN(x)), then x + MLP(LN(x)); no dropout."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), leak_labels)
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    """Windows ``(B, T, 8)`` -> the state h ``(B, d_model)`` at the last position."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.n_quantities, cfg.d_model)
        self.position = nn.Parameter(torch.zeros(cfg.window, cfg.d_model))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1:] != (self.cfg.window, self.cfg.n_quantities):
            raise ValueError(f"windows must be (B, {self.cfg.window}, {self.cfg.n_quantities}), got {tuple(x.shape)}")
        hidden = self.embed(x) + self.position
        for block in self.blocks:
            hidden = block(hidden, leak_labels)
        return self.ln_f(hidden)[:, -1, :]

    def ols_weights(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> list[torch.Tensor]:
        """W of every OLS head, one ``(B, H, T, T)`` per layer, with the activations of the forward pass."""
        hidden = self.embed(x) + self.position
        out = []
        for block in self.blocks:
            out.append(block.attn.weights(block.ln1(hidden), leak_labels))
            hidden = block(hidden, leak_labels)
        return out


class DecisionHead(nn.Module):
    """``[h, m]`` -> 32 -> GELU -> 3 logits."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.d_model + cfg.n_classes, cfg.head_hidden), nn.GELU(), nn.Linear(cfg.head_hidden, cfg.n_classes))

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, m], dim=-1))


class EncoderModel(nn.Module):
    """The encoder, the memory's key map ``z = W_z h`` and the decision head of one arm."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.key = nn.Linear(cfg.d_model, cfg.key_dim, bias=False)
        self.head = DecisionHead(cfg)

    def encode(self, x: torch.Tensor, leak_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(h, z)`` of each window."""
        h = self.encoder(x, leak_labels)
        return h, self.key(h)

    def decide(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Logits of the three classes from the state and the memory's readout."""
        return self.head(h, m)


def count_parameters(model: nn.Module) -> int:
    """Every trainable scalar: lambda's theta counted, the constant key coordinate (an input) not."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_table(model: nn.Module) -> dict[str, int]:
    return {name: p.numel() for name, p in model.named_parameters()}


def expected_parameters(cfg: EncoderConfig) -> int:
    """The closed-form count of :class:`EncoderModel` for ``cfg``."""
    d, m = cfg.d_model, cfg.d_mlp
    embed = cfg.n_quantities * d + d + cfg.window * d
    attn = (d * 3 * d + 3 * d) + (d * d + d) + (cfg.n_heads if cfg.attention == "ols" else 0)
    attn += cfg.n_classes * d if cfg.value_leak else 0
    block = 2 * (2 * d) + attn + (d * m + m) + (m * d + d)
    head = (d + cfg.n_classes) * cfg.head_hidden + cfg.head_hidden + cfg.head_hidden * cfg.n_classes + cfg.n_classes
    return embed + cfg.n_layers * block + 2 * d + d * cfg.key_dim + head


def decay_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: weight decay on the weight matrices of the linear maps only.

    Not on biases, LayerNorm, the position embedding or lambda's theta.
    """
    decay, rest = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            (decay if isinstance(module, nn.Linear) and name == "weight" else rest).append(p)
    if len(decay) + len(rest) != len(list(model.parameters())):
        raise AssertionError("a parameter fell out of the AdamW groups")
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


def one_hot(labels: torch.Tensor, n_classes: int = 3, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return F.one_hot(labels.long(), n_classes).to(dtype)


def arm_configs(arms: Sequence[str] = ARMS, **overrides: Any) -> dict[str, EncoderConfig]:
    return {arm: EncoderConfig(arm=arm, **overrides) for arm in arms}
