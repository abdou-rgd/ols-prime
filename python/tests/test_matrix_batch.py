"""The centred and batch readouts of Matrix (iteration 3), through the Python bindings.

What is checked here, against numpy references written from the formulas of ``docs/DESIGN.md``:

- the centred statistics of a range, and their merge;
- the centred ridge with a free intercept, ``Lambda = rho * n_w * diag(scale)``: forecast,
  leverage and in-window RSS;
- ``ridge_batch`` equals ``centred_query`` cell by cell, bit for bit, whatever the thread count
  and whatever is pushed later, with a status per cell;
- the window boundaries ``as_of - w < time <= as_of``;
- ``kernel_batch`` against brute force: kNN bit for bit under the tie rule, softmax to 1e-12;
- the bindings' rules: dtypes before values, sizes before allocations, microsecond times,
  scalar ranges and types, and the GIL released during the long calls.
"""

import ast
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import olsprime_rs as rs

HOUR = 3_600_000
P, Q = 3, 2


def bits(a: np.ndarray) -> np.ndarray:
    """The bit patterns of a float64 array, so that NaN == NaN and -0.0 != 0.0."""
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def memory(n: int = 400, seed: int = 0, block_log2: int = 3, shift: float = 0.0):
    """A Matrix of n entries, four per hour, with its arrays."""
    rng = np.random.default_rng(seed)
    times = (np.arange(n, dtype=np.int64) // 4) * HOUR
    xs = rng.normal(size=(n, P)) + shift
    ys = np.column_stack([xs @ np.array([0.5, -0.2, 0.1]), rng.normal(size=n)]) + rng.normal(
        scale=0.1, size=(n, Q)
    )
    m = rs.Matrix(P, Q, block_log2=block_log2)
    m.extend(times, xs, ys)
    return m, times, xs, ys


def ref_ridge(xs, ys, x_q, rho, scale):
    """Ridge with an unpenalized intercept, in centred form, by numpy in one pass."""
    n = len(xs)
    mx, my = xs.mean(axis=0), ys.mean(axis=0)
    xc, yc = xs - mx, ys - my
    a = xc.T @ xc + rho * n * np.diag(scale)
    beta = np.linalg.solve(a, xc.T @ yc).T  # (q, p)
    xt = x_q - mx
    forecast = my + beta @ xt
    leverage = xt @ np.linalg.solve(a, xt)
    rss = ((yc - xc @ beta.T) ** 2).sum(axis=0)
    return forecast, leverage, rss


# --- centred statistics ---------------------------------------------------------------------


@pytest.mark.parametrize("shift", [0.0, 1e6])
def test_centred_stats_match_a_two_pass_reference(shift: float) -> None:
    m, _, xs, ys = memory(shift=shift)
    for s, e in [(0, 400), (3, 397), (17, 18), (8, 264)]:
        st = m.centred_stats(s, e)
        x, y = xs[s:e], ys[s:e]
        mx, my = x.mean(axis=0), y.mean(axis=0)
        assert st.count == e - s and st.n == float(e - s)
        # A mean near zero cancels: its error scales with the data, not with itself.
        np.testing.assert_allclose(st.mean_x, mx, rtol=0, atol=1e-14 * (1 + abs(shift)))
        np.testing.assert_allclose(st.mean_y, my, rtol=1e-12, atol=1e-12)
        scale = max(1.0, (e - s))
        np.testing.assert_allclose(st.m_xx, (x - mx).T @ (x - mx), rtol=1e-10, atol=1e-10 * scale)
        np.testing.assert_allclose(st.m_yx, (y - my).T @ (x - mx), rtol=1e-10, atol=1e-10 * scale)
        np.testing.assert_allclose(st.m_yy, (y - my).T @ (y - my), rtol=1e-10, atol=1e-10 * scale)
        assert np.array_equal(st.m_xx, st.m_xx.T) and np.array_equal(st.m_yy, st.m_yy.T)


def test_centred_stats_merge_like_their_union() -> None:
    _, _, xs, ys = memory()
    a = rs.CentredStats.from_rows(xs[:150], ys[:150])
    b = rs.CentredStats.from_rows(xs[150:], ys[150:])
    whole = rs.CentredStats.from_rows(xs, ys)
    ab = a.merged(b)
    assert ab.count == 400 and ab.depth == 1
    for name in ("mean_x", "mean_y", "m_xx", "m_yx", "m_yy"):
        np.testing.assert_allclose(getattr(ab, name), getattr(whole, name), rtol=1e-12, atol=1e-11)
    with pytest.raises(rs.OlsError):
        a.merged(rs.CentredStats.from_rows(np.ones((2, 1)), np.ones((2, 1))))


# --- the centred ridge ----------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 1e-3, 1.0])
def test_centred_query_is_ridge_with_a_free_intercept(rho: float) -> None:
    m, _, xs, ys = memory()
    scale = np.array([1.0, 2.0, 0.5])
    x_q = np.array([0.3, -1.2, 2.0])
    for s, e in [(0, 400), (40, 120), (5, 9)]:
        r = m.centred_query(x_q, s, e, rho, scale)
        f, h, rss = ref_ridge(xs[s:e], ys[s:e], x_q, rho, scale)
        assert (r.start, r.end) == (s, e)
        np.testing.assert_allclose(r.forecast, f, rtol=1e-9, atol=1e-9)
        assert r.leverage == pytest.approx(h, rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(r.rss, rss, rtol=1e-8, atol=1e-9)
        w = m.centred_weights(r)
        assert w.shape == (e - s,)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(w @ ys[s:e], r.forecast, rtol=1e-9, atol=1e-9)
        cut = (s + e) // 2
        parts = m.centred_contribution(r, s, cut) + m.centred_contribution(r, cut, e)
        np.testing.assert_allclose(parts, r.forecast, rtol=1e-9, atol=1e-9)


def test_a_shift_of_a_million_is_read_by_the_centred_path() -> None:
    """The raw path pays the mean squared; the centred one only the rounding of the data."""
    base, times, xs, ys = memory()
    far = rs.Matrix(P, Q, block_log2=3)
    far.extend(times, xs + 1e6, ys)  # the same outcomes, every state shifted
    x_q = np.array([0.3, -1.2, 2.0])
    near = base.centred_query(x_q, 0, 400, 0.0, np.ones(P)).forecast
    shifted = far.centred_query(x_q + 1e6, 0, 400, 0.0, np.ones(P)).forecast
    # The shift rounds each state at ulp(1e6)/2 ~ 6e-11; the forecasts move by that order.
    np.testing.assert_allclose(shifted, near, rtol=0, atol=1e-7)


def test_centred_readouts_belong_to_their_memory() -> None:
    m, _, _, _ = memory()
    other, _, _, _ = memory(seed=1)
    r = m.centred_query(np.zeros(P), 0, 100, 1.0, np.ones(P))
    with pytest.raises(rs.OlsError):
        other.centred_weights(r)
    with pytest.raises(rs.OlsError):
        other.centred_contribution(r, 0, 10)


def test_bad_centred_reads_are_ols_errors() -> None:
    m, _, _, _ = memory()
    with pytest.raises(rs.OlsError):
        m.centred_query(np.zeros(P), 5, 5, 1.0, np.ones(P))  # empty
    with pytest.raises(rs.OlsError):
        m.centred_query(np.zeros(P), 0, 3, 0.0, np.ones(P))  # n <= p, unpenalized
    with pytest.raises(rs.OlsError):
        m.centred_query(np.zeros(P), 0, 100, -1.0, np.ones(P))
    with pytest.raises(rs.OlsError):
        m.centred_query(np.zeros(P), 0, 100, 1.0, np.array([1.0, np.nan, 1.0]))
    with pytest.raises(rs.OlsError, match="expected 3, got 2"):
        m.centred_query(np.zeros(P), 0, 100, 1.0, np.ones(2))
    with pytest.raises(rs.OlsError):
        m.centred_query(np.zeros(P), 0, 401, 1.0, np.ones(P))


# --- windows --------------------------------------------------------------------------------


def test_windows_are_left_open_and_right_closed() -> None:
    m, times, _, _ = memory()
    for as_of in [-1, 0, 1, HOUR - 1, HOUR, 50 * HOUR, 99 * HOUR, 10**12]:
        for window in [None, 1, HOUR, 10 * HOUR, 10 * HOUR + 1]:
            end = int(np.searchsorted(times, as_of, side="right"))
            start = 0 if window is None else int(np.searchsorted(times, as_of - window, "right"))
            start = min(start, end)
            assert m.window_range(as_of, window) == (start, end), (as_of, window)
    for bad in [0, -HOUR]:
        with pytest.raises(rs.OlsError, match="window"):
            m.window_range(0, bad)


# --- ridge batch ----------------------------------------------------------------------------

WINDOWS = [10 * HOUR, 40 * HOUR, None]
RHOS = np.array([0.0, 1e-3, 1.0])


def batch_queries(n: int = 120, seed: int = 7):
    rng = np.random.default_rng(seed)
    as_of = np.sort(rng.integers(-HOUR, 110 * HOUR, size=n)).astype(np.int64)
    as_of[::5] = as_of[::5] // HOUR * HOUR  # many on the hour, where entries are stamped
    return rng.normal(size=(n, P)), as_of


def test_ridge_batch_is_centred_query_cell_by_cell() -> None:
    m, times, xs, ys = memory()
    x_q, as_of = batch_queries()
    scale = np.array([1.0, 0.5, 2.0])
    b = m.ridge_batch(x_q, as_of, WINDOWS, RHOS, scale, leverage=True, rss=True, threads=3)
    seen = set()
    for i in range(len(as_of)):
        for w, window in enumerate(WINDOWS):
            s, e = m.window_range(int(as_of[i]), window)
            assert (b["start"][i, w], b["end"][i, w], b["count"][i, w]) == (s, e, e - s)
            for r, rho in enumerate(RHOS):
                status = b["status"][i, w, r]
                seen.add(int(status))
                try:
                    one = m.centred_query(x_q[i], s, e, float(rho), scale)
                except rs.OlsError:
                    assert status in (rs.STATUS_EMPTY, rs.STATUS_SINGULAR)
                    assert np.isnan(b["forecast"][i, w, r]).all()
                    assert np.isnan(b["leverage"][i, w, r]) and np.isnan(b["rss"][i, w, r]).all()
                    continue
                assert status == rs.STATUS_OK
                assert same_bits(b["forecast"][i, w, r], one.forecast)
                assert same_bits(b["leverage"][i, w, r], np.float64(one.leverage))
                assert same_bits(b["rss"][i, w, r], one.rss)
    # Singular cells are in test_every_status_has_its_cause.
    assert {rs.STATUS_OK, rs.STATUS_EMPTY} <= seen


def test_ridge_batch_matches_numpy() -> None:
    m, times, xs, ys = memory()
    x_q, as_of = batch_queries(30)
    scale = np.ones(P)
    b = m.ridge_batch(x_q, as_of, WINDOWS, RHOS, scale)
    assert set(b) == {"forecast", "status", "start", "end", "count"}
    assert b["forecast"].shape == (30, 3, 3, Q) and b["status"].dtype == np.uint8
    for i in range(30):
        for w in range(3):
            s, e = b["start"][i, w], b["end"][i, w]
            for r, rho in enumerate(RHOS):
                if b["status"][i, w, r] != rs.STATUS_OK:
                    continue
                f, _, _ = ref_ridge(xs[s:e], ys[s:e], x_q[i], rho, scale)
                np.testing.assert_allclose(b["forecast"][i, w, r], f, rtol=1e-8, atol=1e-8)


def same_dict(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        same_bits(a[k], b[k]) if a[k].dtype == np.float64 else np.array_equal(a[k], b[k]) for k in a
    )


def test_batches_do_not_depend_on_threads_order_or_later_pushes() -> None:
    m, times, xs, ys = memory()
    x_q, as_of = batch_queries(700)
    scale = np.ones(P)
    taus, ks = np.array([0.7, 2.0]), np.array([1, 7, 64], dtype=np.int64)
    ridge = m.ridge_batch(x_q, as_of, WINDOWS, RHOS, scale, leverage=True, rss=True, threads=1)
    kern = m.kernel_batch(x_q, as_of, 30 * HOUR, np.ones(P), taus, ks, threads=1)
    for threads in (0, 2, 5):
        assert same_dict(m.ridge_batch(x_q, as_of, WINDOWS, RHOS, scale, leverage=True, rss=True,
                                       threads=threads), ridge)
        assert same_dict(m.kernel_batch(x_q, as_of, 30 * HOUR, np.ones(P), taus, ks,
                                        threads=threads), kern)
    # Reversed order: each query's cells move with it, unchanged.
    rev = m.ridge_batch(x_q[::-1], as_of[::-1], WINDOWS, RHOS, scale, leverage=True, rss=True)
    assert same_dict({k: v[::-1] for k, v in rev.items()}, ridge)
    krev = m.kernel_batch(x_q[::-1], as_of[::-1], 30 * HOUR, np.ones(P), taus, ks)
    assert same_dict({k: v[::-1] for k, v in krev.items()}, kern)
    # Entries pushed after every decision time change nothing a decision reads.
    rng = np.random.default_rng(3)
    later = 200 * HOUR + np.arange(300, dtype=np.int64)
    m.extend(later, rng.normal(size=(300, P)), rng.normal(size=(300, Q)))
    again = m.ridge_batch(x_q, as_of, WINDOWS, RHOS, scale, leverage=True, rss=True)
    assert same_dict(again, ridge)
    assert same_dict(m.kernel_batch(x_q, as_of, 30 * HOUR, np.ones(P), taus, ks), kern)


def test_every_status_has_its_cause() -> None:
    m = rs.Matrix(2, 1, block_log2=2)
    rng = np.random.default_rng(0)
    m.extend(np.arange(10, dtype=np.int64) * 10, rng.normal(size=(10, 2)), rng.normal(size=(10, 1)))
    x_q = np.zeros((3, 2))
    as_of = np.array([-5, 15, 95], dtype=np.int64)  # nothing yet, two entries, all ten
    b = m.ridge_batch(x_q, as_of, [None], np.array([0.0, 1.0]), np.ones(2))
    assert b["status"][:, 0].tolist() == [
        [rs.STATUS_EMPTY, rs.STATUS_EMPTY],
        [rs.STATUS_SINGULAR, rs.STATUS_OK],  # n = 2 <= p unpenalized; the penalty identifies it
        [rs.STATUS_OK, rs.STATUS_OK],
    ]
    ks = np.array([2, 3], dtype=np.int64)
    k = m.kernel_batch(x_q, as_of, None, np.ones(2), np.array([1.0]), ks)
    assert k["knn_status"].tolist() == [
        [rs.STATUS_EMPTY, rs.STATUS_EMPTY],
        [rs.STATUS_OK, rs.STATUS_TOO_FEW],
        [rs.STATUS_OK, rs.STATUS_OK],
    ]
    assert k["softmax_status"][:, 0].tolist() == [rs.STATUS_EMPTY, rs.STATUS_OK, rs.STATUS_OK]
    assert np.isnan(k["knn"][1, 1]).all() and np.isnan(k["softmax"][0]).all()
    assert [rs.STATUS_OK, rs.STATUS_EMPTY, rs.STATUS_SINGULAR, rs.STATUS_NON_FINITE,
            rs.STATUS_TOO_FEW] == [0, 1, 2, 3, 4]


def test_an_overflowing_cell_is_non_finite() -> None:
    m = rs.Matrix(1, 1, block_log2=2)
    m.extend(np.arange(4, dtype=np.int64), np.array([[0.0], [1.0], [2.0], [3.0]]),
             np.array([[1e308], [-1e308], [1e308], [-1e308]]))
    b = m.ridge_batch(np.zeros((1, 1)), np.array([3], dtype=np.int64), [None], np.array([1.0]),
                      np.ones(1), rss=True)
    assert b["status"][0, 0, 0] == rs.STATUS_NON_FINITE
    assert np.isnan(b["forecast"][0, 0, 0]).all() and np.isnan(b["rss"][0, 0, 0]).all()


# --- kernels --------------------------------------------------------------------------------


def brute_force(xs, ys, s, e, xq, sigma, taus, ks):
    """The documented kernels, entry by entry: distances summed over j in order."""
    u, uq = xs[s:e] / sigma, xq / sigma
    d = np.zeros(e - s)
    for j in range(len(sigma)):
        d = d + (uq[j] - u[:, j]) ** 2
    y = ys[s:e]
    order = np.lexsort((np.arange(e - s), d))  # by distance, then by index
    knn = []
    for k in ks:
        if k > e - s:
            knn.append(np.full(y.shape[1], np.nan))
        else:  # summed in order, then divided
            knn.append(np.cumsum(y[order[:k]], axis=0)[-1] / k)
    soft = []
    for tau in taus:
        w = np.exp(-(d - d.min()) / (2 * tau * tau))
        soft.append((w[:, None] * y).sum(axis=0) / w.sum())
    return np.array(soft), np.array(knn)


@pytest.mark.parametrize("grid", [False, True])
def test_kernels_match_brute_force(grid: bool) -> None:
    rng = np.random.default_rng(11)
    n = 900
    xs = rng.normal(size=(n, P))
    if grid:  # coarse features: many exact distance ties
        xs = np.round(xs * 2) / 2
    ys = rng.normal(size=(n, Q))
    times = (np.arange(n, dtype=np.int64) // 3) * HOUR
    m = rs.Matrix(P, Q, block_log2=4)
    m.extend(times, xs, ys)
    sigma = np.array([1.0, 0.5, 2.0])
    taus = np.array([0.3, 1.0, 0.5 * np.sqrt(15)])
    ks = np.array([1, 5, 50, 299, 300, 301], dtype=np.int64)
    x_q = np.round(rng.normal(size=(40, P)) * 2) / 2 if grid else rng.normal(size=(40, P))
    as_of = rng.integers(0, 320 * HOUR, size=40).astype(np.int64)
    b = m.kernel_batch(x_q, as_of, 100 * HOUR, sigma, taus, ks)
    for i in range(40):
        s, e = int(b["start"][i]), int(b["end"][i])
        assert (s, e) == m.window_range(int(as_of[i]), 100 * HOUR) and b["count"][i] == e - s
        if s == e:
            assert (b["knn_status"][i] == rs.STATUS_EMPTY).all()
            continue
        soft, knn = brute_force(xs, ys, s, e, x_q[i], sigma, taus, ks)
        assert same_bits(b["knn"][i], knn), i
        too_few = ks > e - s
        assert (b["knn_status"][i] == np.where(too_few, rs.STATUS_TOO_FEW, rs.STATUS_OK)).all()
        np.testing.assert_allclose(b["softmax"][i], soft, rtol=1e-12, atol=1e-12)
        assert (b["softmax_status"][i] == rs.STATUS_OK).all()


def test_kernel_inputs_are_validated() -> None:
    m, _, _, _ = memory()
    x_q, as_of = np.zeros((1, P)), np.array([HOUR], dtype=np.int64)
    ok = dict(window=None, sigma=np.ones(P), taus=np.array([1.0]), ks=np.array([1], dtype=np.int64))
    m.kernel_batch(x_q, as_of, **ok)
    for k in (0, -1):
        with pytest.raises(ValueError, match=rf"^ks\[0\] = {k}: expected a neighbour count >= 1"):
            m.kernel_batch(x_q, as_of, **{**ok, "ks": np.array([k], dtype=np.int64)})
    with pytest.raises(rs.OlsError):
        m.kernel_batch(x_q, as_of, **{**ok, "sigma": np.array([1.0, 0.0, 1.0])})
    with pytest.raises(rs.OlsError):
        m.kernel_batch(x_q, as_of, **{**ok, "taus": np.array([np.inf])})
    with pytest.raises(rs.OlsError, match="expected 3, got 2"):
        m.kernel_batch(x_q, as_of, **{**ok, "sigma": np.ones(2)})
    with pytest.raises(rs.OlsError):
        m.kernel_batch(np.full((1, P), np.nan), as_of, **ok)
    with pytest.raises(rs.OlsError, match="window"):
        m.kernel_batch(x_q, as_of, **{**ok, "window": 0})


# --- the bindings' rules --------------------------------------------------------------------


def test_dtypes_are_checked_before_any_value() -> None:
    m, _, _, _ = memory()
    x_q, as_of = np.zeros((2, P)), np.array([0, HOUR], dtype=np.int64)
    good = dict(windows=[None], rhos=np.array([1.0]), scale=np.ones(P))
    m.ridge_batch(x_q, as_of, **good)
    for bad in [
        lambda: m.ridge_batch(x_q.astype(np.float32), as_of, **good),
        lambda: m.ridge_batch(x_q.tolist(), as_of, **good),
        lambda: m.ridge_batch(x_q, as_of.astype(np.int32), **good),
        lambda: m.ridge_batch(x_q, as_of.astype(np.float64), **good),
        lambda: m.ridge_batch(x_q[0], as_of, **good),
        lambda: m.ridge_batch(x_q, as_of, **{**good, "rhos": [1.0]}),
        lambda: m.ridge_batch(x_q, as_of, **{**good, "scale": np.ones(P, dtype=np.float32)}),
        lambda: m.ridge_batch(x_q, as_of, **{**good, "windows": "abc"}),
        lambda: m.ridge_batch(x_q, as_of, **{**good, "windows": None}),
        lambda: m.kernel_batch(x_q, as_of, None, np.ones(P), np.ones(1), np.ones(1)),
        lambda: m.kernel_batch(x_q, as_of, None, np.ones(P), np.ones(1, dtype=np.int64),
                               np.ones(1, dtype=np.int64)),
        lambda: m.centred_query(np.zeros(P, dtype=np.float32), 0, 10, 1.0, np.ones(P)),
        lambda: rs.CentredStats.from_rows(np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1))),
    ]:
        with pytest.raises(TypeError):
            bad()


def test_shapes_are_checked() -> None:
    m, _, _, _ = memory()
    as_of = np.array([0, HOUR], dtype=np.int64)
    good = dict(windows=[None], rhos=np.array([1.0]), scale=np.ones(P))
    with pytest.raises(rs.OlsError, match="expected 3, got 2"):
        m.ridge_batch(np.zeros((2, 2)), as_of, **good)
    with pytest.raises(rs.OlsError, match="expected 2, got 3"):
        m.ridge_batch(np.zeros((3, P)), as_of, **good)
    with pytest.raises(rs.OlsError, match="expected 3, got 1"):
        m.ridge_batch(np.zeros((2, P)), as_of, **{**good, "scale": np.ones(1)})
    with pytest.raises(rs.OlsError, match="at least one window and one penalty"):
        m.ridge_batch(np.zeros((2, P)), as_of, **{**good, "windows": []})
    with pytest.raises(rs.OlsError, match="at least one window and one penalty"):
        m.ridge_batch(np.zeros((2, P)), as_of, **{**good, "rhos": np.zeros(0)})
    empty = m.ridge_batch(np.zeros((0, P)), np.zeros(0, dtype=np.int64), **good)
    assert empty["forecast"].shape == (0, 1, 1, Q)
    kempty = m.kernel_batch(np.zeros((0, P)), np.zeros(0, dtype=np.int64), None, np.ones(P),
                            np.ones(1), np.ones(1, dtype=np.int64))
    assert kempty["knn"].shape == (0, 1, Q)


def test_strided_inputs_are_read_in_logical_order() -> None:
    m, _, _, _ = memory()
    x_q, as_of = batch_queries(50)
    wide = np.zeros((50, 2 * P))
    wide[:, ::2] = x_q
    good = dict(windows=WINDOWS, rhos=RHOS, scale=np.ones(P))
    assert same_dict(m.ridge_batch(wide[:, ::2], as_of, **good), m.ridge_batch(x_q, as_of, **good))
    rx, ra = x_q[::-1], as_of[::-1]  # negative strides
    assert same_dict(m.ridge_batch(rx, ra, **good), m.ridge_batch(rx.copy(), ra.copy(), **good))


def test_outputs_are_fresh_copies() -> None:
    m, _, _, _ = memory()
    x_q, as_of = batch_queries(10)
    b = m.ridge_batch(x_q, as_of, WINDOWS, RHOS, np.ones(P))
    for v in b.values():
        assert v.flags.writeable and v.flags.c_contiguous
    b["forecast"][:] = 0.0
    assert not (m.ridge_batch(x_q, as_of, WINDOWS, RHOS, np.ones(P))["forecast"] == 0.0).all()


def test_oversized_batches_are_refused_before_allocating() -> None:
    m, _, _, _ = memory()
    huge = 2**27  # zero-stride views: free to build, 2**27 queries
    x_q = np.broadcast_to(np.zeros(P), (huge, P))
    as_of = np.broadcast_to(np.int64(0), (huge,))
    with pytest.raises(ValueError, match=r"^ridge_batch: the forecast, for 134217728 queries x 3 "
                                         r"windows x 3 penalties would take \d+ bytes, above the "
                                         r"limit MAX_ALLOC_BYTES") as info:
        m.ridge_batch(x_q, as_of, WINDOWS, RHOS, np.ones(P))
    assert type(info.value) is ValueError
    with pytest.raises(ValueError, match=r"^kernel_batch: the softmax, for 134217728 queries"):
        m.kernel_batch(x_q, as_of, None, np.ones(P), np.ones(2), np.ones(1, dtype=np.int64))
    # Penalties by the million: the factorizations every thread caches are refused.
    with pytest.raises(ValueError, match=r"^ridge_batch: the factorizations of every thread"):
        m.ridge_batch(np.zeros((1, P)), np.zeros(1, dtype=np.int64), WINDOWS,
                      np.broadcast_to(1.0, (10**7,)), np.ones(P))
    # A neighbour count near the window: the candidate buffers of 12 threads would take 1.2 GB.
    big = rs.Matrix(1, 1, block_log2=6)
    n = 50_000
    big.extend(np.arange(n, dtype=np.int64), np.zeros((n, 1)), np.zeros((n, 1)))
    many = 12 * 128
    with pytest.raises(ValueError, match=r"^kernel_batch: the candidates and sums of every thread"):
        big.kernel_batch(np.zeros((many, 1)), np.full(many, n, dtype=np.int64), None, np.ones(1),
                         np.ones(1), np.array([n], dtype=np.int64), threads=12)


def test_microsecond_decision_times_are_refused() -> None:
    m, _, _, _ = memory()
    us = rs.MICROSECOND_TIMES_FROM
    m.window_range(us - 1)
    with pytest.raises(ValueError, match=rf"^as_of = {us} looks like microseconds") as info:
        m.window_range(us)
    assert type(info.value) is ValueError
    as_of = np.array([0, us, us + 1], dtype=np.int64)
    with pytest.raises(ValueError, match=rf"^as_of\[1\] = {us} looks like microseconds"):
        m.ridge_batch(np.zeros((3, P)), as_of, [None], np.ones(1), np.ones(P))
    with pytest.raises(ValueError, match=rf"^as_of\[1\] = {us} looks like microseconds"):
        m.kernel_batch(np.zeros((3, P)), as_of, None, np.ones(P), np.ones(1),
                       np.ones(1, dtype=np.int64))


def test_the_gil_is_released_during_a_batch() -> None:
    """Another Python thread runs while a long batch computes."""
    rng = np.random.default_rng(5)
    n = 30_000
    m = rs.Matrix(8, 1, block_log2=6)
    m.extend(np.arange(n, dtype=np.int64), rng.normal(size=(n, 8)), rng.normal(size=(n, 1)))
    x_q = rng.normal(size=(3000, 8))
    as_of = np.full(3000, n, dtype=np.int64)
    args = (x_q, as_of, None, np.ones(8), np.array([1.0]), np.array([10], dtype=np.int64))

    def spin(seconds: float) -> int:
        count, stop = 0, time.perf_counter() + seconds
        while time.perf_counter() < stop:
            count += 1
        return count

    rate = spin(0.05) / 0.05  # iterations per second with the GIL to ourselves
    done = threading.Event()
    elapsed = []

    def work() -> None:
        t0 = time.perf_counter()
        m.kernel_batch(*args, threads=1)
        elapsed.append(time.perf_counter() - t0)
        done.set()

    worker = threading.Thread(target=work)
    count = 0
    worker.start()
    while not done.is_set():
        count += 1
    worker.join()
    # Holding the GIL, the call would leave this loop a switch interval (5 ms) at most.
    assert elapsed[0] > 0.05, "the batch was too quick to tell"
    assert count > 0.2 * rate * elapsed[0], (count, rate, elapsed)


# --- scalar parameters ----------------------------------------------------------------------


def _m() -> rs.Matrix:
    return memory(n=40)[0]


def _readout(m: rs.Matrix) -> rs.CentredReadout:
    return m.centred_query(np.zeros(P), 0, 40, 1.0, np.ones(P))


_X1, _T1 = np.zeros((1, P)), np.array([HOUR], dtype=np.int64)
_KERNEL = dict(sigma=np.ones(P), taus=np.ones(1), ks=np.ones(1, dtype=np.int64))

# Every parameter annotated `_Int` / `_Float` in the stubs, as a call passing `v` there (and
# valid values elsewhere), and the name the error must start with.
ALIAS_PARAMS = {
    ("Matrix.window_range", "as_of"): (lambda v: _m().window_range(v), "as_of"),
    ("Matrix.window_range", "window"): (lambda v: _m().window_range(HOUR, v), "window"),
    **{("Matrix.centred_stats", k): ((lambda v, k=k: _m().centred_stats(
        **{"start": 0, "end": 40, k: v})), k) for k in ("start", "end")},
    **{("Matrix.centred_query", k): ((lambda v, k=k: _m().centred_query(
        np.zeros(P), **{"start": 0, "end": 40, "rho": 1.0, k: v}, scale=np.ones(P))), k)
       for k in ("start", "end", "rho")},
    **{("Matrix.centred_contribution", k): ((lambda v, k=k: (lambda m: m.centred_contribution(
        _readout(m), **{"start": 0, "end": 40, k: v}))(_m())), k) for k in ("start", "end")},
    ("Matrix.ridge_batch", "windows"): (
        lambda v: _m().ridge_batch(_X1, _T1, [v], np.ones(1), np.ones(P)), "windows[0]"),
    ("Matrix.ridge_batch", "threads"): (
        lambda v: _m().ridge_batch(_X1, _T1, [None], np.ones(1), np.ones(P), threads=v), "threads"),
    ("Matrix.kernel_batch", "window"): (
        lambda v: _m().kernel_batch(_X1, _T1, v, **_KERNEL), "window"),
    ("Matrix.kernel_batch", "threads"): (
        lambda v: _m().kernel_batch(_X1, _T1, None, **_KERNEL, threads=v), "threads"),
}


def _alias_params() -> dict[tuple[str, str], str]:
    """(qualified name, parameter) -> "int" or "float", for every alias-annotated parameter."""
    tree = ast.parse((Path(rs.__file__).parent / "__init__.pyi").read_text(encoding="utf-8"))
    out = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            a = fn.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                note = ast.unparse(arg.annotation) if arg.annotation is not None else ""
                for alias, kind in (("_Int", "int"), ("_Float", "float")):
                    if alias in note:
                        out[(f"{cls.name}.{fn.name}", arg.arg)] = kind
    return out


ALIAS_SCALARS = _alias_params()


def test_every_aliased_scalar_parameter_is_checked_below() -> None:
    assert ALIAS_SCALARS and set(ALIAS_PARAMS) == set(ALIAS_SCALARS)


@pytest.mark.parametrize("key", sorted(ALIAS_SCALARS), ids="{0[0]}:{0[1]}".format)
def test_aliased_scalars_out_of_range_are_value_errors_naming_the_argument(key) -> None:
    (call, name), kind = ALIAS_PARAMS[key], ALIAS_SCALARS[key]
    call(1 if kind == "int" else 0.5)  # the other arguments are valid
    for bad in [2**64, -(2**64), 2**200] if kind == "int" else [2**1024, 10**400]:
        with pytest.raises(ValueError, match=rf"^{re.escape(name)} ") as info:
            call(bad)
        assert type(info.value) is ValueError, (bad, type(info.value))
    # None is a valid window.
    wrong = (True, "1", [1]) + ((1.0, np.float64(1.0)) if kind == "int" else ())
    for bad in wrong:
        message = rf"^{re.escape(name)} must be an? {kind}, got "
        with pytest.raises(TypeError, match=message) as info:
            call(bad)
        assert type(info.value) is TypeError, (bad, type(info.value))


def test_numpy_scalars_are_accepted() -> None:
    m = _m()
    assert m.window_range(np.int64(HOUR), np.int32(HOUR)) == m.window_range(HOUR, HOUR)
    r = m.centred_query(np.zeros(P), np.intp(0), np.uint16(40), np.float32(1.0), np.ones(P))
    assert r.end == 40
