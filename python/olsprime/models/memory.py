"""Matrix over learned keys (``docs/PROTOCOL.md``, iteration 4, "Memory: Matrix over learned keys").

Every encoder arm reads a memory whose keys it learns. Each entry is a row: its key
``z = W_z h`` (8), its value the one-hot label (3), its stamp the row's ``label_known_at``
(the stamp is a hook: :class:`olsprime.train.fit.EncoderSystem.entry_stamps`). A readout at
decision time d uses the entries with ``d - 30 days < stamp <= d``:

    m = ybar + beta (z_q - zbar),   beta = M_yz (M_zz + Lambda)^-1,
    Lambda = rho diag(M_zz) + 10^-6 n_w I,   rho = 10^-2,

the centred ridge with a free intercept of ``docs/DESIGN.md`` §3.2, on keys standardized inside the
window. An empty window gives (1/3, 1/3, 1/3). In the Rust core's terms this is
``Matrix.centred_query(z_q, start, end, rho, scale)`` with ``scale = var_w + 10^-6 / rho``, where
``var_w = diag(M_zz) / n_w``: ``rho n_w scale = rho diag(M_zz) + 10^-6 n_w``.

Causality and bits
------------------
- The entries are sorted by stamp (ties: symbol, then open time, as Matrix's entries are). The
  statistics of a window are the difference of two prefix sums of the raw moments
  ``(n, sum z, sum z z^T, sum y, sum y z^T)``, accumulated and differenced in fp64 before centring
  (:func:`prefix_moments`, :func:`window_bounds`, :func:`readout`). The 8 x 8 solve is in fp64.
- A prefix sum at position i is a sequential sum of entries 0..i: on the CPU, ``torch.cumsum`` runs
  sequentially along the summed axis, so the value at i depends on entries 0..i only, in a fixed
  order, and never on the array's length or on later entries. :func:`prefix_moments` checks that
  this holds (``causal=True``, the default, refuses a CUDA tensor). Training (where every input
  precedes the fit) may run its prefix sums on the GPU with ``causal=False``. Torch 2.15's docs
  list ``cumsum`` on a CUDA floating tensor among the ops that act deterministically under
  ``use_deterministic_algorithms(True)`` (older releases listed it among those that raise). On
  torch 2.15.0+cu134 and the RTX 3060 it does not raise, is bit-stable across calls and prefix
  lengths, and matched the CPU's sequential sum to the bit on a 5,000 × 100 fp64 probe (audit 4e).
  The pre-check remains the gate.
- Each query's solve depends on its own window's statistics and its own key only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DAY_MS = 86_400_000
N_CLASSES = 3


@dataclass(frozen=True)
class MemoryConfig:
    """The memory's registered constants. Saved with every fit."""

    window_ms: int = 30 * DAY_MS
    rho: float = 1e-2
    jitter: float = 1e-6  # the 10^-6 n_w I of Lambda
    key_dim: int = 8
    n_classes: int = N_CLASSES

    def __post_init__(self) -> None:
        if self.window_ms <= 0 or self.rho < 0 or self.jitter <= 0:
            raise ValueError("the window must be positive, rho >= 0 and the jitter > 0")


@dataclass(frozen=True, eq=False)
class Prefix:
    """Prefix sums of the raw moments, each with a leading row of zeros: entry i covers entries 0..i-1."""

    n: torch.Tensor  # (E + 1,)
    s_z: torch.Tensor  # (E + 1, p)
    s_zz: torch.Tensor  # (E + 1, p, p)
    s_y: torch.Tensor  # (E + 1, q)
    s_yz: torch.Tensor  # (E + 1, q, p)


def prefix_moments(z: torch.Tensor, y: torch.Tensor, *, causal: bool = True) -> Prefix:
    """The prefix sums of ``(1, z, z z^T, y, y z^T)`` over entries already sorted by stamp, in fp64.

    ``causal=True`` keeps the sums on the CPU, where ``cumsum`` is sequential: the prefix at i is
    then the same bits whatever follows entry i.
    """
    if causal and z.device.type != "cpu":
        raise ValueError("causal prefix sums run on the CPU, where cumsum is sequential")
    z, y = z.to(torch.float64), y.to(torch.float64)
    e, p = z.shape
    q = y.shape[1]
    moments = torch.cat([torch.ones(e, 1, dtype=torch.float64, device=z.device), z, (z[:, :, None] * z[:, None, :]).reshape(e, p * p), y,
                         (y[:, :, None] * z[:, None, :]).reshape(e, q * p)], dim=1)  # fmt: skip
    csum = torch.cat([torch.zeros(1, moments.shape[1], dtype=torch.float64, device=z.device), torch.cumsum(moments, dim=0)], dim=0)
    a = 1
    out = [csum[:, 0]]
    for width in (p, p * p, q, q * p):
        out.append(csum[:, a : a + width])
        a += width
    n, s_z, s_zz, s_y, s_yz = out
    return Prefix(n, s_z, s_zz.reshape(e + 1, p, p), s_y, s_yz.reshape(e + 1, q, p))


def window_bounds(stamps: torch.Tensor, as_of: torch.Tensor, window_ms: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lo, hi)``: the entries with ``as_of - window < stamp <= as_of`` are ``lo .. hi - 1`` of the sorted stamps.

    This is ``Matrix.window_range(as_of, window)`` of the Rust core.
    """
    if stamps.numel() > 1 and bool((stamps[1:] < stamps[:-1]).any()):
        raise ValueError("the entries must be sorted by stamp")
    hi = torch.searchsorted(stamps, as_of, right=True)
    lo = torch.searchsorted(stamps, as_of - window_ms, right=True)
    return lo, hi


@dataclass(frozen=True, eq=False)
class Readout:
    """The memory's readout per query: the forecast m (fp64) and the window's count and mean."""

    m: torch.Tensor  # (Q, q)
    count: torch.Tensor  # (Q,)
    mean_z: torch.Tensor  # (Q, p)
    mean_y: torch.Tensor  # (Q, q)
    beta: torch.Tensor  # (Q, q, p)


def window_stats(prefix: Prefix, lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Raw moments of each window, differenced in fp64: ``(n, sum z, sum z z^T, sum y, sum y z^T)``."""
    return tuple(t[hi] - t[lo] for t in (prefix.n, prefix.s_z, prefix.s_zz, prefix.s_y, prefix.s_yz))


def readout(prefix: Prefix, lo: torch.Tensor, hi: torch.Tensor, z_q: torch.Tensor, cfg: MemoryConfig = MemoryConfig()) -> Readout:
    """The centred ridge readout of every query over its window; an empty window gives 1/3 each.

    Centring after the difference: ``zbar = S_z / n``, ``M_zz = S_zz - n zbar zbar^T``,
    ``ybar = S_y / n``, ``M_yz = S_yz - n ybar zbar^T``; ``Lambda = rho diag(M_zz)^+ + jitter n I``
    (a diagonal that rounds below zero counts as zero, as in the Rust core); then
    ``beta = M_yz A^-1`` by a Cholesky solve in fp64.
    """
    n, s_z, s_zz, s_y, s_yz = window_stats(prefix, lo, hi)
    z_q = z_q.to(torch.float64)
    p, q = s_z.shape[1], s_y.shape[1]
    empty = n == 0
    safe_n = torch.where(empty, torch.ones_like(n), n)
    zbar = s_z / safe_n[:, None]
    ybar = s_y / safe_n[:, None]
    m_zz = s_zz - safe_n[:, None, None] * zbar[:, :, None] * zbar[:, None, :]
    m_zz = 0.5 * (m_zz + m_zz.transpose(1, 2))
    m_yz = s_yz - safe_n[:, None, None] * ybar[:, :, None] * zbar[:, None, :]
    diag = torch.diagonal(m_zz, dim1=1, dim2=2).clamp_min(0.0)
    lam = cfg.rho * diag + cfg.jitter * safe_n[:, None]
    eye = torch.eye(p, dtype=torch.float64, device=z_q.device)
    a = m_zz + lam[:, :, None] * eye
    chol, info = torch.linalg.cholesky_ex(a)
    if bool((info[~empty] != 0).any()):
        raise FloatingPointError("a memory window is not positive definite in fp64")
    # beta^T = A^-1 M_yz^T, (Q, p, q)
    beta_t = torch.cholesky_solve(m_yz.transpose(1, 2), chol)
    m = ybar + ((z_q - zbar)[:, None, :] @ beta_t)[:, 0, :]
    uniform = torch.full_like(m, 1.0 / q)
    m = torch.where(empty[:, None], uniform, m)
    return Readout(m, n, zbar, ybar, beta_t.transpose(1, 2))


def sort_entries(stamps: torch.Tensor, tie1: torch.Tensor, tie2: torch.Tensor) -> torch.Tensor:
    """The order of entries by ``(stamp, tie1, tie2)``: Matrix's push order (stamp, symbol, open time)."""
    order = torch.argsort(tie2, stable=True)
    order = order[torch.argsort(tie1[order], stable=True)]
    return order[torch.argsort(stamps[order], stable=True)]
