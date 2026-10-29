"""The eight Matrix arms (``olsprime.models.matrix``) on a synthetic panel, against numpy references.

- the record's reads (centred ridge with a free intercept and the per-coordinate penalty, the
  windows, the softmax and kNN kernels) agree with a numpy reference written from the formulas;
- the sigma schedule and the climatology agree with numpy;
- each arm's selection logic is right on hand-built losses, defaults included;
- matrix-adaptive's prefix sums agree with a brute force;
- the CUSUM: its simulation is reproducible, the registered threshold gives an ARL of 365 days,
  and its timing (value known at the exit open, alarm acting at the next 00:00 UTC) holds.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from olsprime.bench.synthetic import synthetic_bars
from olsprime.features.panel import assemble, pair_rows
from olsprime.models import matrix as mx
from olsprime.models.matrix import DAY_MS, HOUR_MS, MatrixConfig, MatrixSystem, Scope

START = datetime(2018, 9, 1, tzinfo=UTC)
MONTHS = [datetime(y, m, 1, tzinfo=UTC) for y, m in [(2018, 10), (2018, 11), (2018, 12), (2019, 1), (2019, 2)]]
CFG = MatrixConfig(first_test=(2019, 1), last_test=(2019, 2))
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")


def synthetic_panel(hours: dict[str, int] | None = None) -> pl.DataFrame:
    hours = hours or {"AAAUSDT": 24 * 200, "BBBUSDT": 24 * 200, "CCCUSDT": 24 * 160}
    parts = []
    for i, s in enumerate(SYMBOLS):
        b = synthetic_bars(s, start=START, days=hours[s], interval="1h", seed=i, vol=0.004, drift=0.0)
        parts.append(pair_rows(b, MONTHS))
    return assemble(parts)


@pytest.fixture(scope="module")
def system() -> MatrixSystem:
    return MatrixSystem(synthetic_panel(), CFG)


@pytest.fixture(scope="module")
def result(system: MatrixSystem) -> mx.ArmsResult:
    return system.run()


@pytest.fixture(scope="module")
def record(system: MatrixSystem, result: mx.ArmsResult) -> mx.Record:
    return system.last_record


# --- numpy references -----------------------------------------------------------------------


def entries(system: MatrixSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every complete row in push order: stamps, x, y (one-hot class and fwd_return)."""
    d = system.data
    rows = np.flatnonzero(d.complete)
    order = np.lexsort((d.open_ms[rows], d.symbol[rows], d.lka_ms[rows]))
    rows = rows[order]
    y = np.zeros((rows.size, 4))
    y[np.arange(rows.size), d.label[rows]] = 1.0
    y[:, 3] = d.fwd[rows]
    return d.lka_ms[rows], d.x[rows], y


def ref_ridge(xs: np.ndarray, ys: np.ndarray, x_q: np.ndarray, rho: float, var: np.ndarray) -> np.ndarray:
    n = len(xs)
    mxs, my = xs.mean(axis=0), ys.mean(axis=0)
    xc, yc = xs - mxs, ys - my
    a = xc.T @ xc + rho * n * np.diag(var)
    beta = np.linalg.solve(a, xc.T @ yc).T
    return my + beta @ (x_q - mxs)


def ref_kernels(xs, ys, x_q, sigma, tau, k):
    u, uq = xs / sigma, x_q / sigma
    d2 = ((u - uq) ** 2).sum(axis=1)
    w = np.exp(-(d2 - d2.min()) / (2 * tau * tau))
    soft = (w[:, None] * ys).sum(axis=0) / w.sum()
    near = np.argsort(d2, kind="stable")[:k]
    return soft, ys[near].mean(axis=0)


def ref_var(system: MatrixSystem, month_ms: int) -> np.ndarray:
    t, x, _ = entries(system)
    return x[t <= month_ms - CFG.horizon_ms].var(axis=0)


def test_sigma_and_climatology_follow_the_schedule(system: MatrixSystem, record: mx.Record) -> None:
    d = system.data
    for m in np.unique(record.month):
        st = system.month_stats(int(m))
        np.testing.assert_allclose(st.var, ref_var(system, int(m)), rtol=1e-11)
        known = d.labelled & (d.lka_ms <= m - CFG.horizon_ms)
        freq = np.bincount(d.label[known], minlength=3) / known.sum()
        np.testing.assert_allclose(st.clim, mx.prob_map(freq, CFG.prob_floor), rtol=1e-12)
    first = int(np.unique(record.month)[0])
    assert datetime.fromtimestamp(first / 1000, tz=UTC) == MONTHS[1]  # 2018-10 has no sigma: the record starts in 2018-11
    assert system.month_stats(int(MONTHS[0].timestamp() * 1000)).var is None


def test_the_record_matches_numpy_ridge_and_kernels(system: MatrixSystem, record: mx.Record) -> None:
    d = system.data
    t, xs, ys = entries(system)
    rng = np.random.default_rng(0)
    picks = rng.choice(record.rows.size, size=24, replace=False)
    W, R = len(CFG.windows_days), len(CFG.rhos)
    checked = 0
    for i in picks:
        row = record.rows[i]
        at = d.decision_ms[row]
        var = system.month_stats(int(record.month[i])).var
        for w, days in enumerate(CFG.windows_days):
            inside = (t <= at) & ((t > at - days * DAY_MS) if days is not None else True)
            for r, rho in enumerate(CFG.rhos):
                c = w * R + r
                if not record.ok[i, c]:
                    continue
                ref = mx.prob_map(ref_ridge(xs[inside], ys[inside], d.x[row], rho, var)[:3], CFG.prob_floor)
                np.testing.assert_allclose(record.probs[i, c], ref, rtol=1e-9, atol=1e-11)
                checked += 1
        inside = (t <= at) & (t > at - CFG.kernel_window_days * DAY_MS)
        for j, f in enumerate(CFG.tau_factors):
            for kk, k in enumerate(CFG.ks):
                soft, near = ref_kernels(xs[inside], ys[inside], d.x[row], np.sqrt(var), f * math.sqrt(15), k)
                np.testing.assert_allclose(record.probs[i, system.softmax0 + j], mx.prob_map(soft[:3], CFG.prob_floor), rtol=1e-12)
                if inside.sum() >= k:
                    np.testing.assert_allclose(record.probs[i, system.knn0 + kk], mx.prob_map(near[:3], CFG.prob_floor), rtol=1e-14)
                else:
                    assert not record.ok[i, system.knn0 + kk]
    assert checked > 24 * W * R // 2
    # the losses are -log p of the label
    lab = d.label[record.rows]
    np.testing.assert_array_equal(record.loss[:, 0], -np.log(record.probs[np.arange(lab.size), 0, lab]))


class _ReadsBeforeEveryEntry(MatrixSystem):
    """Every ridge window ends before the first entry: every ridge cell is empty."""

    def ridge_as_of(self, rows: np.ndarray) -> np.ndarray:
        return np.full(rows.size, int(START.timestamp() * 1000), dtype=np.int64)


def test_a_refused_cell_falls_back_to_the_arm_climatology_and_is_counted(system: MatrixSystem) -> None:
    s2 = _ReadsBeforeEveryEntry(system.data, CFG)
    res = s2.run(Scope(arms=("climatology-panel", "matrix-select", "matrix-stack", "matrix-cusum")))
    rec = s2.last_record
    assert not rec.ok[:, : s2.n_ridge].any()
    for i in (0, rec.rows.size - 1):
        np.testing.assert_array_equal(rec.probs[i, 0], s2.month_stats(int(rec.month[i])).clim)
    n = res.rows.size
    assert res.fallback_counts() == {"climatology-panel": 0, "matrix-select": n, "matrix-stack": n, "matrix-cusum": n}
    np.testing.assert_array_equal(res.probs["matrix-select"], res.probs["climatology-panel"])
    assert res.record_fallbacks[s2.candidates[0]] == rec.rows.size


def test_mid_rows_carry_the_leverage_and_rss(system: MatrixSystem, record: mx.Record) -> None:
    d = system.data
    t, xs, ys = entries(system)
    k = len(record.mid) // 2
    j = record.mid[k]
    row = record.rows[j]
    assert system.data.open_ms[row] % DAY_MS == 0
    var = system.month_stats(int(record.month[j])).var
    inside = (t <= d.decision_ms[row]) & (t > d.decision_ms[row] - 365 * DAY_MS)
    x, y = xs[inside], ys[inside, 3]
    n = len(x)
    xc, yc = x - x.mean(axis=0), y - y.mean()
    a = xc.T @ xc + 1e-2 * n * np.diag(var)
    beta = np.linalg.solve(a, xc.T @ yc)
    xt = d.x[row] - x.mean(axis=0)
    assert record.mid_lev[k, 1, 2] == pytest.approx(xt @ np.linalg.solve(a, xt), rel=1e-9)
    assert record.mid_fwd[k, 1, 2] == pytest.approx(y.mean() + beta @ xt, rel=1e-9, abs=1e-14)
    rss = ((yc - xc @ beta) ** 2).sum()
    assert record.mid_rss[k, 1, 2] == pytest.approx(rss, rel=1e-8)
    assert record.mid_n[k, 1] == n


def test_the_status_of_every_output_is_counted(result: mx.ArmsResult, record: mx.Record) -> None:
    for arm, counts in result.status_counts.items():
        assert sum(counts.values()) == result.rows.size, arm
        assert counts["ok"] + sum(v for k, v in counts.items() if k != "ok") == result.rows.size
        assert result.fallback[arm].sum() == result.rows.size - counts["ok"]
    for c, counts in result.record_status.items():
        assert set(counts) == {"empty", "singular", "non-finite", "too-few"}
    # k = 1000 at the record's first hours, and nothing else, may find too few entries
    assert all(v == 0 for c, cnt in result.record_status.items() for k, v in cnt.items() if k == "singular")


# --- loud failures: time units and zero variance (audit pass 1b) -----------------------------


def _with(data: mx.PanelArrays, **changes: np.ndarray) -> mx.PanelArrays:
    return mx.PanelArrays(**{**data.__dict__, **changes})


@pytest.mark.parametrize("unit", [1000, 1_000_000])
def test_times_in_another_unit_fail_loudly(system: MatrixSystem, unit: int) -> None:
    d = system.data
    seconds = d.decision_ms // 1000 if unit == 1000 else d.decision_ms * 1000
    with pytest.raises(ValueError, match="millisecond epochs"):
        MatrixSystem(_with(d, decision_ms=seconds), CFG)
    with pytest.raises(ValueError, match="millisecond epochs"):
        MatrixSystem(_with(d, lka_ms=np.where(d.lka_ms >= 0, d.lka_ms // 1000, -1)), CFG)


class _SecondsAsOf(MatrixSystem):
    def ridge_as_of(self, rows: np.ndarray) -> np.ndarray:
        return self.data.decision_ms[rows] // 1000


def test_a_read_in_seconds_fails_loudly_instead_of_emptying_every_window(system: MatrixSystem) -> None:
    with pytest.raises(ValueError, match="ridge as_of"):
        _SecondsAsOf(system.data, CFG).run(Scope(arms=("matrix-select",)))
    with pytest.raises(TypeError, match="int64"):
        mx.check_ms("x", np.array([1.5e12]))


def test_a_zero_fold_variance_fails_loudly_with_the_month_and_the_feature(system: MatrixSystem) -> None:
    d = system.data
    x = d.x.copy()
    x[:, 5] = np.where(np.isfinite(x[:, 5]), 0.25, np.nan)
    with pytest.raises(ValueError, match=r"2018-11-01.*" + mx.FEATURE_COLUMNS[5]):
        MatrixSystem(_with(d, x=x), CFG).run()


def test_feature_ratios_are_mean_over_std() -> None:
    x = np.zeros((4, 15))
    x[:, 0] = [9.0, 11.0, 9.0, 11.0]
    r = mx.feature_ratios(x)
    assert r[mx.FEATURE_COLUMNS[0]] == pytest.approx(10.0) and math.isnan(r[mx.FEATURE_COLUMNS[1]])


# --- the probability map and the residuals ---------------------------------------------------


def test_the_probability_map_floors_and_renormalizes() -> None:
    p = mx.prob_map(np.array([[0.7, -0.2, 0.5], [0.2, 0.3, 0.5]]), 1e-3)
    np.testing.assert_allclose(p[0], np.array([0.7, 1e-3, 0.5]) / 1.201)
    np.testing.assert_allclose(p[1], [0.2, 0.3, 0.5])
    assert np.isnan(mx.prob_map(np.array([np.nan, 0.5, 0.5]), 1e-3)).all()


def test_standardized_residuals_use_rss_over_n_minus_p_minus_one() -> None:
    z = mx.standardized_residuals(np.array([0.03]), np.array([0.01]), np.array([0.2]), np.array([0.5]), np.array([116]))
    sd = math.sqrt(0.5 / 100) * math.sqrt(1 + 1 / 116 + 0.2)
    assert z[0] == pytest.approx(0.02 / sd, rel=1e-14)
    assert np.isnan(mx.standardized_residuals(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([16])))[0]


def test_seqsum_is_sequential_and_argmin_takes_the_first() -> None:
    a = np.array([1e16, 1.0, -1e16, 1.0])
    assert mx.seqsum(a) == ((1e16 + 1.0) - 1e16) + 1.0
    assert mx.argmin_first(np.array([3.0, 1.0, 1.0, np.nan])) == 1
    assert mx.argmin_first(np.array([np.nan, 2.0])) == 1


def test_the_stack_grid_is_the_simplex_in_steps_of_005() -> None:
    a = mx.stack_alphas(20)
    assert a.shape == (231, 3)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert (a >= 0).all() and np.allclose(a * 20, np.round(a * 20))
    p = np.array([[0.2, 0.3, 0.5]])
    assert np.array_equal(mx.mix(np.array([0.0, 1.0, 0.0]), [p * 0, p, p * 0]), p)


# --- selections on hand-built losses ---------------------------------------------------------


def _handbuilt(system: MatrixSystem, record: mx.Record, best: dict[str, int]) -> mx.Record:
    """A copy of the record whose losses make the given candidates the best on every row."""
    loss = np.full_like(record.loss, 2.0)
    loss[:, : system.n_ridge] += np.arange(system.n_ridge) * 1e-3  # a strict order inside the ridge block
    for c, v in best.items():
        loss[:, c] = v
    return mx.Record(**{**record.__dict__, "loss": loss})


def test_each_monthly_arm_selects_the_lowest_mean_loss(system: MatrixSystem, record: mx.Record) -> None:
    R = len(CFG.rhos)
    m = int(MONTHS[3].timestamp() * 1000)
    # matrix-select's best: all history at rho = 1e-3; inside the 365-day window rho = 1; softmax tau index 2, knn k index 0
    rec = _handbuilt(system, record, {2 * R + 1: 0.1, 1 * R + 4: 0.2, system.softmax0 + 2: 0.5, system.knn0: 0.5})
    fit = system.monthly_fit(rec, m, kernels=True)
    assert not fit["default"]
    assert fit["select"] == (2, 1) and fit["rho_1y"] == 4 and fit["tau"] == 2 and fit["k"] == 0
    assert fit["rho_by_window"] == (0, 4, 1)
    sel = system.selection_mask(m, rec)
    np.testing.assert_allclose(fit["scores"], rec.loss[sel].mean(axis=0), rtol=1e-12)


def test_the_selection_rows_are_purged_and_within_the_year(system: MatrixSystem, record: mx.Record) -> None:
    d = system.data
    m = int(MONTHS[4].timestamp() * 1000)
    sel = system.selection_mask(m, record)
    rows = record.rows[sel]
    assert (d.lka_ms[rows] <= m - 24 * HOUR_MS).all()
    assert (d.decision_ms[rows] > m - 365 * DAY_MS).all() and (d.decision_ms[rows] < m).all()
    late = record.rows[(d.decision_ms[record.rows] < m) & (d.lka_ms[record.rows] > m - 24 * HOUR_MS)]
    assert late.size > 0 and not np.isin(late, rows).any()


def test_the_stack_weights_follow_the_label_probabilities(system: MatrixSystem, record: mx.Record) -> None:
    R = len(CFG.rhos)
    m = int(MONTHS[3].timestamp() * 1000)
    probs = record.probs.copy()
    lab = system.data.label[record.rows]
    rows = np.arange(lab.size)
    for w, good in enumerate((0.9, 0.1, 0.1)):  # the 30-day window is right, the others wrong
        c = w * R + 2
        probs[:, c] = (1 - good) / 2
        probs[rows, c, lab] = good
    rec = mx.Record(**{**_handbuilt(system, record, {2: 0.1, R + 2: 0.1, 2 * R + 2: 0.1}).__dict__, "probs": probs})
    fit = system.monthly_fit(rec, m, kernels=True)
    assert fit["rho_by_window"] == (2, 2, 2)
    np.testing.assert_array_equal(fit["alpha"], [1.0, 0.0, 0.0])


def test_a_selection_on_too_few_rows_takes_the_defaults(system: MatrixSystem, record: mx.Record) -> None:
    cfg = MatrixConfig(first_test=(2019, 1), last_test=(2019, 2), min_selection_rows=10**9)
    s2 = MatrixSystem(system.data, cfg)
    fit = s2.monthly_fit(record, int(MONTHS[3].timestamp() * 1000), kernels=True)
    assert fit["default"]
    assert fit["select"] == (1, 2) and fit["rho_1y"] == 2 and fit["rho_by_window"] == (2, 2, 2)
    assert cfg.tau_factors[fit["tau"]] == 1.0 and cfg.ks[fit["k"]] == 200
    np.testing.assert_array_equal(fit["alpha"], [0.0, 1.0, 0.0])
    res = s2.run()
    assert res.defaults["matrix-select"] == 2 and res.defaults["matrix-adaptive"] == res.adaptive["choice"].size
    assert (res.adaptive["choice"] == 2).all()


def test_adaptive_prefix_sums_match_a_brute_force(system: MatrixSystem, result: mx.ArmsResult, record: mx.Record) -> None:
    d = system.data
    R = len(CFG.rhos)
    fit = result.fits[0]
    w = fit["select"][0]
    m = fit["month_ms"]
    decisions = np.unique(d.decision_ms[record.rows[record.month == m]])[::37]
    ch = system.adaptive_choices(record, m, w, decisions)
    lka, dec = d.lka_ms[record.rows], d.decision_ms[record.rows]
    for i, at in enumerate(decisions):
        inside = (lka <= at) & (dec > at - 30 * DAY_MS)
        assert ch["count"][i] == inside.sum()
        mean = record.loss[inside, w * R : (w + 1) * R].mean(axis=0)
        np.testing.assert_allclose(ch["scores"][i], mean, rtol=1e-10)
        assert ch["choice"][i] == (int(np.argmin(mean)) if inside.sum() >= CFG.min_selection_rows else 2)


def test_outputs_read_the_selected_candidates(system: MatrixSystem, result: mx.ArmsResult, record: mx.Record) -> None:
    R = len(CFG.rhos)
    pos = {int(r): i for i, r in enumerate(record.rows)}
    for fit in result.fits:
        at = np.flatnonzero(system.data.month_ms[result.rows] == fit["month_ms"])
        rp = np.array([pos[int(r)] for r in result.rows[at]])
        w, r = fit["select"]
        np.testing.assert_array_equal(result.probs["matrix-select"][at], record.probs[rp, w * R + r])
        np.testing.assert_array_equal(result.probs["matrix-1y"][at], record.probs[rp, R + fit["rho_1y"]])
        np.testing.assert_array_equal(result.probs["softmax-1y"][at], record.probs[rp, system.softmax0 + fit["tau"]])
        np.testing.assert_array_equal(result.probs["knn-1y"][at], record.probs[rp, system.knn0 + fit["k"]])
        np.testing.assert_array_equal(result.probs["climatology-panel"][at], np.broadcast_to(fit["clim"], (at.size, 3)))
        parts = [record.probs[rp, v * R + fit["rho_by_window"][v]] for v in range(3)]
        np.testing.assert_array_equal(result.probs["matrix-stack"][at], mx.mix(fit["alpha"], parts))
    for arm, p in result.probs.items():
        assert np.isfinite(p).all(), arm
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_a_scoped_run_has_the_bits_of_the_full_run(system: MatrixSystem, result: mx.ArmsResult) -> None:
    until = int(datetime(2019, 2, 10, 5, tzinfo=UTC).timestamp() * 1000) - 1
    month = int(MONTHS[4].timestamp() * 1000)
    part = MatrixSystem(system.data, CFG).run(Scope(until_ms=until, kernel_from_ms=month - 365 * DAY_MS, kernel_months=(month,)))
    dec = system.data.decision_ms[result.rows]
    assert np.array_equal(part.rows, result.rows[dec <= until])
    for arm in mx.ARMS:
        a, b = result.probs[arm][dec <= until], part.probs[arm]
        if arm in mx.KERNEL_ARMS:
            keep = system.data.month_ms[part.rows] == month
            a, b = a[keep], b[keep]
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), arm


# --- the CUSUM -------------------------------------------------------------------------------


def test_the_cusum_simulation_is_reproducible() -> None:
    a, b = mx.cusum_simulate(2000, 5), mx.cusum_simulate(2000, 5)
    for name in ("path", "time", "value"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert mx.cusum_arl(a, 4.0) == mx.cusum_arl(b, 4.0)
    assert mx.cusum_arl(mx.cusum_simulate(2000, 6), 4.0) != mx.cusum_arl(a, 4.0)
    # the run length is monotone in h
    arls = [mx.cusum_arl(a, h)[0] for h in (2.0, 3.0, 4.0, 5.0)]
    assert arls == sorted(arls)


def test_the_registered_threshold_gives_an_arl_of_365_days() -> None:
    rec = mx.cusum_simulate(100_000, mx.CUSUM_SEED)
    arl, se = mx.cusum_arl(rec, mx.CUSUM_THRESHOLD)
    assert abs(arl - 365.0) < 4 * se
    assert mx.cusum_arl(rec, mx.CUSUM_THRESHOLD - 0.05)[0] < 365.0 < mx.cusum_arl(rec, mx.CUSUM_THRESHOLD + 0.05)[0]
    assert mx.CUSUM_THRESHOLD == round(mx.CUSUM_THRESHOLD, 3) and CFG.cusum_threshold == mx.CUSUM_THRESHOLD


def test_page_cusum_on_a_hand_built_sequence() -> None:
    up, down, alarm = mx.page_cusum([1.0, 1.0, 1.0, -3.0, 0.0], k=0.5, h=1.2)
    assert up == [0.5, 1.0, 1.5, 0.0, 0.0] and alarm == [False, False, True, True, False]
    assert down == [0.0, 0.0, 0.0, 2.5, 0.0]  # restarted at 0 after the alarm at the third value


def test_the_cusum_timing_and_its_refits(system: MatrixSystem) -> None:
    cfg = MatrixConfig(first_test=(2019, 1), last_test=(2019, 2), cusum_threshold=1.0)
    res = MatrixSystem(system.data, cfg).run(Scope(arms=("matrix-cusum",)))
    path, fits = res.cusum["path"], res.cusum["fits"]
    assert len(fits) > 2, "a low threshold must alarm"
    assert fits[0]["fit_ms"] == int(MONTHS[3].timestamp() * 1000)
    assert (path["enters_ms"] == path["day_ms"] + DAY_MS + HOUR_MS).all()
    for f in fits[1:]:
        alarm_days = path["day_ms"][path["alarm"]]
        assert f["fit_ms"] - 2 * DAY_MS in alarm_days  # an alarm on day D acts at D + 2 days, 00:00 UTC
        assert f["fit_ms"] % DAY_MS == 0
    # each day's value is made entirely by the fit in force at its decisions, its s included (audit pass 1b)
    times = np.array([f["fit_ms"] for f in fits])
    for i in range(path["day_ms"].size):
        in_force = np.searchsorted(times, path["day_ms"][i], side="right") - 1
        assert path["fit"][i] == in_force and path["s"][i] == fits[in_force]["s"]
        # no fit acted between the day's decisions and the instant its value entered
        assert not np.any((times > path["day_ms"][i]) & (times <= path["enters_ms"][i]))
    # the day decided just before each refit is left out of the restarted CUSUM, and only that day
    skipped = res.cusum["skipped_days"]
    assert set(skipped.tolist()) == {int(f["fit_ms"] - DAY_MS) for f in fits[1:]}
    assert not np.isin(skipped, path["day_ms"]).any()
    # the CUSUM restarts at 0 after every alarm
    for i in np.flatnonzero(path["alarm"][:-1]):
        z = path["z"][i + 1]
        assert path["up"][i + 1] == max(0.0, z - cfg.cusum_k) and path["down"][i + 1] == max(0.0, -z - cfg.cusum_k)
    # its outputs come from its own readouts: before the first refit they equal matrix-select's of January
    full = MatrixSystem(system.data, cfg).run(Scope(arms=("matrix-select", "matrix-cusum")))
    dec = system.data.decision_ms[full.rows]
    early = (dec < fits[1]["fit_ms"]) & (system.data.month_ms[full.rows] == fits[0]["fit_ms"])
    assert early.sum() > 0
    assert (full.cusum["fits"][0]["window"], full.cusum["fits"][0]["rho"]) == full.fits[0]["select"]
    assert np.array_equal(full.probs["matrix-cusum"][early].view(np.uint64), full.probs["matrix-select"][early].view(np.uint64))
    later = dec >= fits[1]["fit_ms"]
    assert not np.array_equal(full.probs["matrix-cusum"][later], full.probs["matrix-select"][later])
