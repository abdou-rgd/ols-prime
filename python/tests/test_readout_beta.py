"""The primal side of a centred readout (iteration 4): ``CentredReadout.beta`` and ``mean_y``.

``olsprime.explain why`` splits a Matrix forecast into per-feature parts ``beta[c, j] * xt[j]``,
with ``xt = x_q - mean_x``. What is checked here, through the bindings, on synthetic data only:

- shapes (q, p) and (q,), dtype float64, C order, fresh copies;
- ``mean_y`` is the range's ``centred_stats(...).mean_y``, bit for bit, and ``beta`` is the
  closed form ``M_yx (M_xx + Lambda)^-1`` of those statistics, within a solve's error;
- ``mean_y + beta @ xt`` reproduces ``forecast``: **to the bit** when summed in the forecast's own
  order (each product rounded on its own, j = 0, 1, ... in order, ``mean_y`` added last), and
  within ``2 gamma(p + 1) (|mean_y| + sum_j |beta xt|)`` through numpy's ``@`` (Higham, Thm 3.1);
- at ``rho = 0`` on a well-posed design, ``beta`` is the OLS slope block of ``lstsq([1, X], Y)``.
"""

import numpy as np
import pytest

import olsprime_rs as rs

HOUR = 3_600_000
U = 2.0**-53  # the unit roundoff


def gamma(k: int) -> float:
    """Higham's gamma_k = k u / (1 - k u)."""
    return k * U / (1 - k * U)


def bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def memory(p: int, q: int, n: int, seed: int, shift: float = 0.0, block_log2: int = 3):
    """A Matrix of n synthetic entries, four per hour, with its arrays."""
    rng = np.random.default_rng(seed)
    times = (np.arange(n, dtype=np.int64) // 4) * HOUR
    xs = rng.normal(size=(n, p)) + shift
    ys = (xs - shift) @ rng.normal(size=(p, q)) + rng.normal(size=(n, q))
    m = rs.Matrix(p, q, block_log2=block_log2)
    m.extend(times, xs, ys)
    return m, xs, ys, rng


def in_order(readout: rs.CentredReadout, x_q: np.ndarray) -> np.ndarray:
    """mean_y + beta xt, with the forecast's own operations, in Python floats (IEEE double, no
    FMA): xt_j = x_q_j - mean_x_j, the products summed left to right from the first, then
    mean_y last."""
    beta, mean_y = readout.beta, readout.mean_y
    xt = [float(a) - float(b) for a, b in zip(x_q, readout.mean_x, strict=True)]
    out = []
    for c in range(beta.shape[0]):
        dot = float(beta[c, 0]) * xt[0]
        for j in range(1, len(xt)):
            dot += float(beta[c, j]) * xt[j]
        out.append(float(mean_y[c]) + dot)
    return np.array(out, dtype=np.float64)


CASES = [
    # p, q, n, start, end, rho, shift, block_log2
    (3, 2, 400, 0, 400, 0.0, 0.0, 3),
    (3, 2, 400, 37, 301, 1e-3, 0.0, 3),
    (1, 1, 50, 5, 45, 0.5, 0.0, 0),
    (6, 4, 900, 100, 900, 1e-2, 0.0, 6),
    (15, 4, 2000, 0, 2000, 1e-3, 0.0, 6),
    (4, 3, 300, 11, 250, 0.0, 1e6, 2),
    (5, 1, 9, 2, 9, 0.1, 0.0, 1),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "p{}q{}n{}[{}:{}]rho{}shift{}b{}".format(*c))
def test_mean_y_plus_beta_xt_is_the_forecast(case) -> None:
    p, q, n, start, end, rho, shift, block_log2 = case
    m, xs, _, rng = memory(p, q, n, seed=p * 1000 + n, shift=shift, block_log2=block_log2)
    scale = rng.uniform(0.5, 2.0, size=p)
    checked = 0
    for _ in range(50):
        x_q = xs[rng.integers(start, end)] + rng.normal(size=p)
        r = m.centred_query(x_q, start, end, rho, scale)
        beta, mean_y, forecast = r.beta, r.mean_y, r.forecast

        assert beta.shape == (q, p) and beta.dtype == np.float64 and beta.flags.c_contiguous
        assert mean_y.shape == (q,) and mean_y.dtype == np.float64
        # Fresh copies: writing to one changes nothing in the readout.
        beta[...] = np.nan
        mean_y[...] = np.nan
        beta, mean_y = r.beta, r.mean_y
        assert np.isfinite(beta).all() and np.isfinite(mean_y).all()

        # To the bit, in the forecast's own order.
        assert same_bits(in_order(r, x_q), forecast)

        # Within rounding, through numpy's matmul.
        xt = x_q - r.mean_x
        via_numpy = mean_y + beta @ xt
        terms = np.abs(beta * xt).sum(axis=1)
        tol = 2 * gamma(p + 1) * (np.abs(mean_y) + terms) + (p + 1) * np.finfo(np.float64).tiny
        assert (np.abs(via_numpy - forecast) <= tol).all(), (via_numpy - forecast, tol)
        checked += 1
    assert checked == 50


@pytest.mark.parametrize("case", CASES, ids=lambda c: "p{}q{}n{}[{}:{}]rho{}shift{}b{}".format(*c))
def test_beta_is_the_closed_form_of_the_range_statistics(case) -> None:
    """beta = M_yx A^-1 with A = M_xx + rho n diag(scale), solved by numpy (LU) from the bindings'
    own ``centred_stats``. Each side is a backward-stable solve, within
    ``kappa_2(A) c_p u ||beta_c||`` of the exact value to first order (Higham, Thm 9.4 and 10.4);
    the test allows ``2 * 3p^2 (1 + 2^(p-1)) u kappa_2(A)`` times the larger norm."""
    p, q, n, start, end, rho, shift, block_log2 = case
    m, xs, _, rng = memory(p, q, n, seed=p * 1000 + n, shift=shift, block_log2=block_log2)
    scale = rng.uniform(0.5, 2.0, size=p)
    r = m.centred_query(xs[start], start, end, rho, scale)
    s = m.centred_stats(start, end)
    assert same_bits(r.mean_y, s.mean_y)
    assert same_bits(r.mean_x, s.mean_x)
    a = s.m_xx.copy()
    a[np.diag_indices(p)] += (rho * (end - start)) * scale
    want = np.linalg.solve(a, s.m_yx.T).T
    eig = np.linalg.eigvalsh(a)
    kappa = eig.max() / eig.min()
    c_p = 3 * p * p * (1 + 2.0 ** (p - 1))
    for c in range(q):
        tol = 2 * c_p * U * kappa * max(np.linalg.norm(r.beta[c]), np.linalg.norm(want[c]))
        assert np.linalg.norm(r.beta[c] - want[c]) <= tol, (c, r.beta[c] - want[c], tol)


@pytest.mark.parametrize("p,q,n", [(1, 1, 30), (3, 2, 400), (8, 3, 200)])
def test_at_rho_zero_beta_is_ols(p: int, q: int, n: int) -> None:
    """At rho = 0, beta is the slope block of least squares on the raw design [1, X] (an SVD,
    no centring); the intercept mean_y - beta mean_x is OLS's too. The tolerance is the one of
    ``at_rho_zero_beta_is_ols`` in ``crates/ols-core/tests/readout_beta.rs``."""
    m, xs, ys, _ = memory(p, q, n, seed=7 + p)
    r = m.centred_query(np.zeros(p), 0, n, 0.0, np.zeros(p))
    design = np.column_stack([np.ones(n), xs])
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    s = m.centred_stats(0, n)
    eig = np.linalg.eigvalsh(s.m_xx)
    kappa, inv_norm = eig.max() / eig.min(), 1 / eig.min()
    h = n + 3 + 3 * s.depth
    for c in range(q):
        extra = np.sqrt(s.m_yy[c, c] * inv_norm)
        tol = 16 * h * p * U * kappa * (np.linalg.norm(coef[1:, c]) + extra)
        assert np.linalg.norm(r.beta[c] - coef[1:, c]) <= tol, c
        intercept = r.mean_y[c] - r.beta[c] @ r.mean_x
        assert abs(intercept - coef[0, c]) <= 16 * h * p * U * kappa * (
            1 + np.linalg.norm(r.beta[c]) + extra
        ), c
