"""Tests of ``olsprime.eval.stats``: closed forms, and Monte Carlo coverage of every interval.

Coverage tests use fixed seeds, so they are deterministic; their tolerances are about two Monte
Carlo standard errors of a coverage rate (1.1 points at 400 runs, 0.7 at 1000). Tests that would
take longer than about 20 s are skipped unless ``OLSPRIME_SLOW=1``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from scipy import stats as st

from olsprime.eval import stats as S

slow = pytest.mark.skipif(os.environ.get("OLSPRIME_SLOW") != "1", reason="slow; set OLSPRIME_SLOW=1")


def lo_se(sr: float, t: int) -> float:
    return math.sqrt((1.0 + sr * sr / 2.0) / t)


def ar1(rng: np.random.Generator, t: int, rho: float, mu: float) -> tuple[np.ndarray, float]:
    """An AR(1) series with unit innovations, mean ``mu``, and its true Sharpe ratio."""
    e = rng.normal(size=t + 200)
    x = np.empty_like(e)
    x[0] = e[0] / math.sqrt(1.0 - rho * rho)
    for i in range(1, e.size):
        x[i] = rho * x[i - 1] + e[i]
    return mu + x[200:], mu * math.sqrt(1.0 - rho * rho)


def calibrated(rng: np.random.Generator, n: int, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    p = rng.dirichlet(np.ones(k), size=n)
    u = rng.random(n)[:, None]
    y = (u > np.cumsum(p, axis=1)).sum(axis=1)
    return p, np.minimum(y, k - 1)


# ---------------------------------------------------------------------------------------------
# Sharpe ratios
# ---------------------------------------------------------------------------------------------


def test_sharpe_and_annualized_follow_their_definitions() -> None:
    r = np.array([0.01, -0.02, 0.03, 0.005, -0.001])
    assert S.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert S.annualized(0.1, 365) == pytest.approx(0.1 * math.sqrt(365))
    with pytest.raises(ValueError):
        S.sharpe(np.ones(10))
    with pytest.raises(ValueError):
        S.sharpe(np.array([0.1, np.nan, 0.2]))


def test_sharpe_se_matches_the_hand_formula() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, size=300) * 0.02 + 0.004
    t = r.size
    sr = r.mean() / r.std(ddof=1)
    c = r - r.mean()
    m2 = np.mean(c**2)
    g3 = np.mean(c**3) / m2**1.5
    g4 = np.mean(c**4) / m2**2
    assert S.moments(r) == pytest.approx((g3, g4))
    expected = math.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr * sr) / (t - 1))
    assert S.sharpe_se(r) == pytest.approx(expected)


def test_sharpe_se_with_normal_returns_is_lo() -> None:
    rng = np.random.default_rng(1)
    for sr_true in (0.0, 0.1, 0.5):
        r = rng.normal(sr_true, 1.0, size=200_000)
        se = S.sharpe_se(r)
        assert se == pytest.approx(lo_se(S.sharpe(r), r.size), rel=0.02)


def test_skew_and_fat_tails_move_the_se_the_right_way() -> None:
    rng = np.random.default_rng(2)
    t = 50_000
    # Positive mean in every case; compare with Lo's normal formula at the same SR and T - 1.
    z = rng.lognormal(0.0, 0.6, size=t)
    z = (z - z.mean()) / z.std()
    neg_skew = 0.3 - z
    pos_skew = 0.3 + z
    fat = 0.3 + rng.standard_t(4, size=t) / math.sqrt(2.0)
    def ratio(r: np.ndarray) -> float:
        return S.sharpe_se(r) / lo_se(S.sharpe(r), r.size - 1)
    assert S.moments(neg_skew)[0] < -1.0
    assert ratio(neg_skew) > 1.1
    assert ratio(pos_skew) < 0.9
    assert ratio(fat) > 1.02
    assert ratio(0.3 + rng.normal(size=t)) == pytest.approx(1.0, abs=0.01)


def test_analytic_interval_is_symmetric_and_annualized() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.02, size=1000)
    iv = S.sharpe_ci(r, 365, 0.95)
    se = S.sharpe_se(r) * math.sqrt(365)
    assert iv.estimate == pytest.approx(S.sharpe(r) * math.sqrt(365))
    assert iv.high - iv.estimate == pytest.approx(1.959963985 * se)
    assert iv.estimate - iv.low == pytest.approx(1.959963985 * se)


def test_analytic_interval_covers_iid_normal_sharpe() -> None:
    rng = np.random.default_rng(10)
    runs, hit = 1000, 0
    for _ in range(runs):
        iv = S.sharpe_ci(rng.normal(0.1, 1.0, size=500))
        hit += iv.low <= 0.1 <= iv.high
    assert abs(hit / runs - 0.95) <= 0.025, hit / runs


def test_bootstrap_interval_covers_iid_normal_sharpe() -> None:
    rng = np.random.default_rng(11)
    runs, hit = 400, 0
    for _ in range(runs):
        iv = S.sharpe_ci_bootstrap(rng.normal(0.1, 1.0, size=500), n_boot=500, seed=rng)
        hit += iv.low <= 0.1 <= iv.high
    assert abs(hit / runs - 0.95) <= 0.025, hit / runs


def test_stationary_bootstrap_covers_ar1_better_than_iid_interval() -> None:
    rng = np.random.default_rng(12)
    runs, hit_a, hit_b = 400, 0, 0
    for _ in range(runs):
        r, sr = ar1(rng, 1000, 0.3, 0.1)
        a = S.sharpe_ci(r)
        b = S.sharpe_ci_bootstrap(r, n_boot=500, seed=rng)
        hit_a += a.low <= sr <= a.high
        hit_b += b.low <= sr <= b.high
    cov_a, cov_b = hit_a / runs, hit_b / runs
    # Theory: the iid interval is too short by sqrt((1 + rho) / (1 - rho)) ~ 1.36, about 85 %.
    assert cov_a < 0.92, cov_a
    assert cov_b >= 0.91, cov_b
    assert cov_b - cov_a >= 0.03, (cov_a, cov_b)


def test_bootstrap_is_reproducible_and_reports_its_block() -> None:
    r = np.random.default_rng(4).normal(0.05, 1.0, size=400)
    a = S.sharpe_ci_bootstrap(r, 365, n_boot=300, seed=7)
    b = S.sharpe_ci_bootstrap(r, 365, n_boot=300, seed=7)
    assert a == b
    assert a.mean_block == S.optimal_block_length(r)
    assert a.low < a.estimate < a.high
    assert S.sharpe_ci_bootstrap(r, n_boot=300, seed=7, mean_block=5).mean_block == 5
    with pytest.raises(TypeError):
        S.sharpe_ci_bootstrap(r, n_boot=10, seed=1.5)  # type: ignore[arg-type]


def test_paired_bootstrap_uses_the_same_indices() -> None:
    r = np.random.default_rng(5).normal(0.05, 1.0, size=500)
    # A rescaled copy has the same Sharpe ratio on every resample only if the indices are shared.
    iv = S.sharpe_diff_ci_bootstrap(r, 3.0 * r, n_boot=500, seed=1)
    assert iv.low == pytest.approx(0.0, abs=1e-12)
    assert iv.high == pytest.approx(0.0, abs=1e-12)
    assert iv.p_value == 1.0
    with pytest.raises(ValueError):
        S.sharpe_diff_ci_bootstrap(r, r[:-1], seed=1)


def test_paired_bootstrap_detects_a_clear_difference() -> None:
    rng = np.random.default_rng(6)
    e = rng.normal(size=2000)
    a = 0.15 + e
    b = 0.0 + 0.9 * e + math.sqrt(1 - 0.81) * rng.normal(size=2000)
    iv = S.sharpe_diff_ci_bootstrap(a, b, 365, n_boot=1000, seed=2)
    assert iv.low > 0.0
    assert iv.p_value is not None and iv.p_value < 0.01


def test_paired_bootstrap_covers_a_known_sharpe_difference() -> None:
    rng = np.random.default_rng(13)
    runs, hit = 400, 0
    rho = 0.7
    for _ in range(runs):
        e = rng.normal(size=500)
        a = 0.10 + e
        b = 0.02 + rho * e + math.sqrt(1 - rho * rho) * rng.normal(size=500)
        iv = S.sharpe_diff_ci_bootstrap(a, b, n_boot=500, seed=rng)
        hit += iv.low <= 0.08 <= iv.high
    assert abs(hit / runs - 0.95) <= 0.025, hit / runs


# ---------------------------------------------------------------------------------------------
# PSR and DSR
# ---------------------------------------------------------------------------------------------


def test_psr_is_phi_of_the_standardized_sharpe() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0.05, 1.0, size=750)
    sr, se = S.sharpe(r), S.sharpe_se(r)
    assert S.psr(r, sr) == pytest.approx(0.5)
    assert S.psr(r, 0.0) == pytest.approx(st.norm.cdf(sr / se))
    assert S.psr(r, 0.02) > S.psr(r, 0.04)


def test_dsr_with_one_trial_is_psr_at_zero() -> None:
    r = np.random.default_rng(8).normal(0.05, 1.0, size=500)
    assert S.expected_max_sharpe(1, None) == 0.0
    assert S.dsr(r, 1) == S.psr(r, 0.0)
    assert S.dsr(r, 1, var_sr=0.3) == S.psr(r, 0.0)


def test_dsr_needs_a_variance_and_falls_with_more_trials() -> None:
    r = np.random.default_rng(9).normal(0.08, 1.0, size=500)
    with pytest.raises(ValueError):
        S.dsr(r, 5)
    with pytest.raises(ValueError):
        S.expected_max_sharpe(0, 1.0)
    v = 1.0 / 500
    values = [S.dsr(r, n, v) for n in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert S.dsr(r, 10, v) == pytest.approx(S.psr(r, S.expected_max_sharpe(10, v)))
    assert S.trials_var_sr(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        S.trials_var_sr(np.array([0.1]))


def test_expected_max_sharpe_closed_form() -> None:
    g = S.EULER_GAMMA
    for n in (2, 10, 100):
        expected = (1 - g) * st.norm.ppf(1 - 1 / n) + g * st.norm.ppf(1 - 1 / (n * math.e))
        assert S.expected_max_sharpe(n, 4.0) == pytest.approx(2.0 * expected)


@pytest.mark.parametrize("n_trials", [10, 100])
def test_expected_max_sharpe_matches_monte_carlo(n_trials: int) -> None:
    """The maximum of N estimated Sharpe ratios of skill-less strategies, with V their pooled
    cross-sectional variance, as in Bailey and López de Prado (2014)."""
    rng = np.random.default_rng(100 + n_trials)
    t, reps = 250, 4000
    maxima, sharpes = [], []
    for _ in range(reps // 200):
        r = rng.normal(size=(200, n_trials, t))
        sr = r.mean(axis=2) / r.std(axis=2, ddof=1)
        maxima.append(sr.max(axis=1))
        sharpes.append(sr)
    mc = float(np.concatenate(maxima).mean())
    var_sr = float(np.concatenate(sharpes).var(ddof=1))
    assert S.expected_max_sharpe(n_trials, var_sr) == pytest.approx(mc, rel=0.03)


# ---------------------------------------------------------------------------------------------
# Block length and the stationary bootstrap
# ---------------------------------------------------------------------------------------------


def test_block_length_is_about_one_for_white_noise() -> None:
    rng = np.random.default_rng(20)
    blocks = [S.optimal_block_length(rng.normal(size=2000)) for _ in range(30)]
    assert np.median(blocks) <= 2
    assert max(blocks) <= 5


def test_block_length_grows_with_autocorrelation_and_matches_ar1_theory() -> None:
    rng = np.random.default_rng(21)
    t = 2000
    medians = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        blocks = [S.optimal_block_length(ar1(rng, t, rho, 0.0)[0]) for _ in range(15)]
        medians.append(float(np.median(blocks)))
        if rho in (0.3, 0.6):
            # For AR(1), B* = (2 rho / (1 - rho^2))^(2/3) T^(1/3).
            theory = (2 * rho / (1 - rho * rho)) ** (2 / 3) * t ** (1 / 3)
            assert np.median(blocks) == pytest.approx(theory, rel=0.3)
    assert all(a < b for a, b in zip(medians, medians[1:])), medians


def test_block_length_is_capped_and_handles_constants() -> None:
    rng = np.random.default_rng(22)
    x, _ = ar1(rng, 100, 0.99, 0.0)
    assert 1 <= S.optimal_block_length(x) <= math.ceil(min(3 * math.sqrt(100), 100 / 3))
    assert S.optimal_block_length(np.ones(50)) == 1


def test_stationary_bootstrap_indices_have_geometric_blocks() -> None:
    t = 1000
    for mean_block in (1.0, 5.0, 20.0):
        idx = S.stationary_bootstrap_indices(t, mean_block, 200, seed=3)
        assert idx.shape == (200, t)
        assert idx.min() >= 0 and idx.max() < t
        breaks = np.mean(idx[:, 1:] != (idx[:, :-1] + 1) % t)
        # A new block starts w.p. 1/b, and lands on the next index anyway w.p. 1/t.
        assert breaks == pytest.approx((1 / mean_block) * (1 - 1 / t), rel=0.05)
    a = S.stationary_bootstrap_indices(t, 5.0, 10, seed=4)
    assert np.array_equal(a, S.stationary_bootstrap_indices(t, 5.0, 10, seed=4))
    with pytest.raises(ValueError):
        S.stationary_bootstrap_indices(t, 0.5, 10, seed=4)


# ---------------------------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------------------------

P = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [1 / 3, 1 / 3, 1 / 3]])
Y = np.array([0, 1, 2])


def test_log_loss_and_brier_match_hand_computation() -> None:
    ll = -(math.log(0.7) + math.log(0.3) + math.log(1 / 3)) / 3
    br = ((0.09 + 0.04 + 0.01) + (0.01 + 0.49 + 0.36) + (1 / 9 + 1 / 9 + 4 / 9)) / 3
    assert S.log_loss(P, Y) == pytest.approx(ll)
    assert S.brier(P, Y) == pytest.approx(br)
    rows = S.per_row_losses(P, Y)
    assert rows.log_loss == pytest.approx([-math.log(0.7), -math.log(0.3), math.log(3)])
    assert rows.brier == pytest.approx([0.14, 0.86, 2 / 3])
    assert S.log_loss(np.array([[1.0, 0.0, 0.0]]), np.array([1])) == pytest.approx(-math.log(1e-15))


def test_forecast_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        S.brier(np.array([[0.5, 0.6, 0.1]]), np.array([0]))
    with pytest.raises(ValueError):
        S.brier(P, np.array([0, 1, 3]))
    with pytest.raises(ValueError):
        S.brier(P, np.array([0, 1]))
    with pytest.raises(ValueError):
        S.brier(np.array([[-0.1, 0.6, 0.5]]), np.array([0]))


def test_brier_skill_of_the_reference_is_zero() -> None:
    rng = np.random.default_rng(30)
    p, y = calibrated(rng, 2000)
    clim = np.bincount(y, minlength=3) / y.size
    assert S.brier_skill(p, p, y) == 0.0
    assert S.brier_skill(np.tile(clim, (y.size, 1)), clim, y) == pytest.approx(0.0, abs=1e-15)
    assert S.brier_skill(p, clim, y) > 0.0
    iv = S.bss_ci_bootstrap(p, p, y, n_boot=200, seed=1)
    assert (iv.estimate, iv.low, iv.high) == (0.0, 0.0, 0.0)


def test_bss_interval_covers_the_population_skill() -> None:
    """Truth p ~ Dirichlet(2,2,2); the model shrinks p halfway to uniform; the reference is
    uniform. Expected Brier of q against y ~ p is sum q^2 - 2 q.p + 1."""
    rng = np.random.default_rng(31)
    u = np.full(3, 1 / 3)
    big = rng.dirichlet([2, 2, 2], size=500_000)
    q_big = 0.5 * big + 0.5 * u
    e_model = np.mean(np.sum(q_big**2, 1) - 2 * np.sum(q_big * big, 1) + 1)
    e_ref = np.mean(np.sum(u**2) - 2 * big @ u + 1)
    truth = 1 - e_model / e_ref
    runs, hit = 400, 0
    for _ in range(runs):
        p = rng.dirichlet([2, 2, 2], size=1000)
        y = (rng.random(1000)[:, None] > np.cumsum(p, 1)).sum(1).clip(max=2)
        iv = S.bss_ci_bootstrap(0.5 * p + 0.5 * u, u, y, n_boot=500, seed=rng)
        hit += iv.low <= truth <= iv.high
    assert abs(hit / runs - 0.95) <= 0.025, (hit / runs, truth)


def test_ece_matches_a_hand_computation() -> None:
    p = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.1, 0.45, 0.45]])
    y = np.array([0, 1, 2, 1])
    # Top-label scores 0.9, 0.8, 0.5, 0.45 with outcomes 1, 0, 1, 1 (argmax of row 4 is class 1).
    # Width bins of 0.1: {0.9}, {0.8}, {0.5}, {0.45}: |1-.9| + |0-.8| + |1-.5| + |1-.45|, over 4.
    assert S.ece(p, y, 10, "width") == pytest.approx((0.1 + 0.8 + 0.5 + 0.55) / 4)
    # Two equal-mass bins: {0.45, 0.5} -> conf 0.475, acc 1; {0.8, 0.9} -> conf 0.85, acc 0.5.
    assert S.ece(p, y, 2, "mass") == pytest.approx(0.5 * 0.525 + 0.5 * 0.35)
    table = S.reliability_table(p, y, 2, "mass")
    assert table.count.tolist() == [2, 2]
    assert table.ece == pytest.approx(S.ece(p, y, 2, "mass"))


@pytest.mark.parametrize("strategy", ["width", "mass"])
def test_ece_is_small_when_calibrated_and_large_when_overconfident(strategy: str) -> None:
    rng = np.random.default_rng(32)
    p, y = calibrated(rng, 30_000)
    sharp = p**4 / np.sum(p**4, axis=1, keepdims=True)
    assert S.ece(p, y, 10, strategy) < 0.015
    assert S.classwise_ece(p, y, 10, strategy) < 0.015
    assert S.ece(sharp, y, 10, strategy) > 0.1
    assert S.classwise_ece(sharp, y, 10, strategy) > 0.05


def test_ece_is_biased_upward_at_small_n() -> None:
    rng = np.random.default_rng(33)
    small = np.mean([S.ece(*calibrated(rng, 200)) for _ in range(50)])
    large = np.mean([S.ece(*calibrated(rng, 20_000)) for _ in range(5)])
    assert small > 3 * large


def test_classwise_ece_sees_miscalibration_that_top_label_misses() -> None:
    """Calibrated top label, but the two minority classes are swapped."""
    rng = np.random.default_rng(34)
    n = 30_000
    top = rng.uniform(0.6, 0.9, size=n)
    rest = 1 - top
    p_true = np.column_stack([top, 0.8 * rest, 0.2 * rest])
    y = (rng.random(n)[:, None] > np.cumsum(p_true, 1)).sum(1).clip(max=2)
    p_model = np.column_stack([top, 0.2 * rest, 0.8 * rest])
    assert S.ece(p_model, y) < 0.015
    assert S.classwise_ece(p_model, y) > 0.05


def test_ece_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(35)
    p, y = calibrated(rng, 2000)
    for kind in ("top", "classwise"):
        iv = S.ece_ci_bootstrap(p, y, kind=kind, n_boot=200, seed=1)
        assert 0.0 <= iv.low <= iv.high
        assert iv.mean_block is not None and iv.mean_block >= 1


# ---------------------------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------------------------


def test_dm_matches_a_hand_computed_newey_west_at_h7() -> None:
    rng = np.random.default_rng(40)
    t, h = 300, 7
    la = rng.normal(size=t) ** 2
    lb = np.convolve(rng.normal(size=t + h - 1), np.ones(h) / h, "valid") ** 2 + 0.1
    d = [la[i] - lb[i] for i in range(t)]
    mean = sum(d) / t
    gamma = [sum((d[i] - mean) * (d[i - k] - mean) for i in range(k, t)) / t for k in range(h)]
    lrv = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    dm = mean / math.sqrt(lrv / t)
    hln = dm * math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    p = 2 * st.t.sf(abs(hln), t - 1)
    res = S.diebold_mariano(la, lb, h, "bartlett")
    assert res.statistic == pytest.approx(hln)
    assert res.p_value == pytest.approx(p)
    assert res.mean_diff == pytest.approx(mean)
    assert (res.h, res.h_used, res.n, res.note, res.kernel_used) == (7, 7, t, "", "bartlett")
    # The default is the rectangular kernel of Diebold and Mariano (1995).
    rect = gamma[0] + 2 * sum(gamma[1:])
    assert rect > 0
    hln_rect = mean / math.sqrt(rect / t) * math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    default = S.diebold_mariano(la, lb, h)
    assert default.statistic == pytest.approx(hln_rect)
    assert default.p_value == pytest.approx(2 * st.t.sf(abs(hln_rect), t - 1))
    assert (default.kernel_used, default.h_used, default.note) == ("rectangular", 7, "")


def test_dm_kernels_agree_at_h1() -> None:
    rng = np.random.default_rng(45)
    la, lb = rng.normal(size=200) ** 2, rng.normal(size=200) ** 2
    assert S.diebold_mariano(la, lb) == S.diebold_mariano(la, lb, kernel="rectangular")
    rect = S.diebold_mariano(la, lb, kernel="rectangular")
    bart = S.diebold_mariano(la, lb, kernel="bartlett")
    assert (rect.statistic, rect.p_value) == (bart.statistic, bart.p_value)


def test_dm_sign_and_power() -> None:
    rng = np.random.default_rng(41)
    truth = rng.normal(size=1000)
    good = (truth - (truth + 0.3 * rng.normal(size=1000))) ** 2
    bad = (truth - rng.normal(size=1000)) ** 2
    res = S.diebold_mariano(good, bad)
    assert res.statistic < 0 and res.mean_diff < 0
    assert res.p_value < 0.01
    assert S.diebold_mariano(bad, good).statistic == pytest.approx(-res.statistic)


def _dm_rejection_rate(seed: int, t: int, h: int, kernel: str) -> float:
    rng = np.random.default_rng(seed)
    rejections, sims = 0, 400
    for _ in range(sims):
        # Equal accuracy; with h > 1, h-step errors overlap: an MA(h - 1) loss differential.
        ea = np.convolve(rng.normal(size=t + h - 1), np.ones(h), "valid")
        eb = np.convolve(rng.normal(size=t + h - 1), np.ones(h), "valid")
        rejections += S.diebold_mariano(ea**2, eb**2, h, kernel).p_value < 0.05  # type: ignore[arg-type]
    return rejections / sims


def test_dm_has_nominal_size_under_equal_accuracy() -> None:
    rate = _dm_rejection_rate(42, 500, 1, "rectangular")
    assert 0.02 <= rate <= 0.09, rate


def test_dm_rectangular_kernel_has_nominal_size_for_overlapping_horizons() -> None:
    rate = _dm_rejection_rate(43, 1000, 7, "rectangular")
    assert 0.02 <= rate <= 0.09, rate


def test_dm_bartlett_over_rejects_for_overlapping_horizons() -> None:
    """Pins the size distortion documented in ``diebold_mariano``: Newey-West with h - 1 lags
    misses a third of an MA(6) long-run variance."""
    rate = _dm_rejection_rate(43, 1000, 7, "bartlett")
    assert 0.07 <= rate <= 0.14, rate


def test_dm_falls_back_from_rectangular_to_bartlett() -> None:
    """Lag-1 autocorrelation near -0.8: rectangular 1 + 2 rho < 0, Bartlett 1 + rho > 0."""
    rng = np.random.default_rng(44)
    d = np.tile([1.0, -1.0], 100) + 0.5 * rng.normal(size=200)
    t, h = d.size, 2
    c = d - d.mean()
    g0, g1 = c @ c / t, c[1:] @ c[:-1] / t
    assert g0 + 2 * g1 <= 0 < g0 + g1
    res = S.diebold_mariano(d, np.zeros_like(d), h=h)
    assert (res.kernel_used, res.h_used) == ("bartlett", 2)
    assert "fell back to Bartlett" in res.note and "h = 1" not in res.note
    expected = d.mean() / math.sqrt((g0 + g1) / t) * math.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
    assert res.statistic == pytest.approx(expected)


def test_dm_falls_back_to_h1_when_bartlett_is_not_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bartlett is positive semi-definite, so a non-positive value needs a degenerate input; force it.
    real = S._hac
    monkeypatch.setattr(S, "_hac", lambda d, lags, kernel: -1.0 if lags > 0 else real(d, lags, kernel))
    rng = np.random.default_rng(46)
    la, lb = rng.normal(size=100) ** 2, rng.normal(size=100) ** 2
    res = S.diebold_mariano(la, lb, h=3)
    assert (res.kernel_used, res.h_used) == ("bartlett", 1)
    assert "fell back to Bartlett" in res.note and "fell back to h = 1" in res.note
    assert res.statistic == pytest.approx(S.diebold_mariano(la, lb, h=1).statistic)
    only = S.diebold_mariano(la, lb, h=3, kernel="bartlett")
    assert only.h_used == 1 and "rectangular" not in only.note


def test_dm_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        S.diebold_mariano(np.ones(50), np.zeros(50))
    with pytest.raises(ValueError):
        S.diebold_mariano(np.ones(50), np.zeros(49))


# ---------------------------------------------------------------------------------------------
# Longer checks
# ---------------------------------------------------------------------------------------------


@slow
def test_bootstrap_coverage_at_1000_runs_and_2000_draws() -> None:
    rng = np.random.default_rng(50)
    runs, hit = 1000, 0
    for _ in range(runs):
        iv = S.sharpe_ci_bootstrap(rng.normal(0.1, 1.0, size=500), n_boot=2000, seed=rng)
        hit += iv.low <= 0.1 <= iv.high
    assert abs(hit / runs - 0.95) <= 0.02, hit / runs


@slow
def test_ar1_bootstrap_coverage_at_1000_runs() -> None:
    rng = np.random.default_rng(51)
    runs, hit = 1000, 0
    for _ in range(runs):
        r, sr = ar1(rng, 2000, 0.3, 0.1)
        iv = S.sharpe_ci_bootstrap(r, n_boot=1000, seed=rng)
        hit += iv.low <= sr <= iv.high
    assert hit / runs >= 0.925, hit / runs
