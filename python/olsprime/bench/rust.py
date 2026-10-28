"""The bench's calls into the Rust core (``olsprime_rs``, built from ``crates/olsprime-py``).

Imported lazily, so that the parts of the bench that do not simulate (the invariance test
runs on targets) import without the extension.
"""

from __future__ import annotations

import numpy as np


def ridge_fit(z: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """``(intercept, beta)`` of ridge on ``[1, z]`` with an unpenalized intercept, by ``ols_core``.

    ``olsprime_rs.RidgeConfig(lam, intercept=True)`` penalizes every coordinate but the first,
    the constant column prepended here. ``SuffStats.from_rows`` builds ``G = XᵀX`` and
    ``C = yᵀX``; ``RidgeModel.beta`` is ``C (G + Λ)⁻¹``, shape ``(1, 1 + p)``.
    """
    import olsprime_rs

    x = np.ascontiguousarray(np.column_stack([np.ones(z.shape[0]), z]), dtype=np.float64)
    ys = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1, 1))
    stats = olsprime_rs.SuffStats.from_rows(x, ys)
    model = olsprime_rs.ridge_fit(stats, olsprime_rs.RidgeConfig(float(lam), intercept=True))
    beta = np.asarray(model.beta, dtype=np.float64)
    if beta.shape != (1, x.shape[1]):
        raise ValueError(f"olsprime_rs.ridge_fit returned beta of shape {beta.shape}, expected {(1, x.shape[1])}")
    return float(beta[0, 0]), beta[0, 1:].copy()


def version() -> str:
    import olsprime_rs

    return str(olsprime_rs.__version__)
