"""The memory over learned keys against the Rust core and its closed form (CPU torch)."""

import numpy as np
import pytest
import torch

import olsprime_rs as rs
from olsprime.models.memory import DAY_MS, MemoryConfig, prefix_moments, readout, sort_entries, window_bounds

HOUR_MS = 3_600_000
U64 = 2.0**-53
CFG = MemoryConfig()


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One intra-op thread: spinning OpenMP threads slow down by orders of magnitude when another process shares the cores."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def episode(seed: int = 0, members: int = 5, hours: int = 24 * 75, burst: int | None = None):
    """Entries of a synthetic episode: one row per member and hour, decided at the hour's last ms, known 24 h + 1 ms later.

    Keys have means far from zero (the centring matters) and correlated coordinates. ``burst`` keeps
    only the rows decided in the first ``burst`` hours.
    """
    rng = np.random.default_rng(seed)
    t0 = 1_514_764_800_000  # 2018-01-01
    open_ms = np.repeat(t0 + HOUR_MS * np.arange(hours, dtype=np.int64), members)
    symbol = np.tile(np.arange(members, dtype=np.int64), hours)
    if burst is not None:
        keep = open_ms < t0 + burst * HOUR_MS
        open_ms, symbol = open_ms[keep], symbol[keep]
    n = open_ms.size
    decision = open_ms + HOUR_MS - 1
    lka = open_ms + 25 * HOUR_MS
    mix = rng.normal(size=(8, 8)) / 3 + np.eye(8)
    z = rng.normal(size=(n, 8)) @ mix + rng.uniform(-3, 3, size=8)
    label = rng.integers(0, 3, size=n)
    return {"open": open_ms, "symbol": symbol, "decision": decision, "lka": lka, "z": z, "label": label, "t0": t0, "hours": hours}


def sorted_by(ep, stamps):
    order = sort_entries(torch.from_numpy(stamps), torch.from_numpy(ep["symbol"]), torch.from_numpy(ep["open"])).numpy()
    return order, stamps[order]


def onehot(label):
    y = np.zeros((label.size, 3))
    y[np.arange(label.size), label] = 1.0
    return y


def torch_readouts(ep, stamps, queries_at, z_q):
    order, st = sorted_by(ep, stamps)
    prefix = prefix_moments(torch.from_numpy(ep["z"][order]), torch.from_numpy(onehot(ep["label"][order])))
    lo, hi = window_bounds(torch.from_numpy(st), torch.from_numpy(queries_at), CFG.window_ms)
    return order, st, lo, hi, readout(prefix, lo, hi, torch.from_numpy(z_q), CFG)


def test_the_memory_holds_the_entries_and_gives_the_readout_of_the_rust_core():
    ep = episode()
    order, st = sorted_by(ep, ep["lka"])
    matrix = rs.Matrix(8, 3)
    matrix.extend(np.ascontiguousarray(st), np.ascontiguousarray(ep["z"][order]), np.ascontiguousarray(onehot(ep["label"][order])))
    decisions = np.unique(ep["decision"])
    rng = np.random.default_rng(1)
    z_q = ep["z"][rng.integers(0, ep["z"].shape[0], decisions.size)] + rng.normal(size=(decisions.size, 8))
    _, _, lo, hi, ro = torch_readouts(ep, ep["lka"], decisions, z_q)
    worst = 0.0
    for i, d in enumerate(decisions):
        start, end = matrix.window_range(int(d), CFG.window_ms)
        assert (int(lo[i]), int(hi[i])) == (start, end)  # the same entries, at every decision time
        if end == start:
            assert np.array_equal(ro.m[i].numpy(), np.full(3, 1 / 3))
            continue
        stats = matrix.centred_stats(start, end)
        var = np.diag(stats.m_xx) / stats.n
        scale = var + CFG.jitter / CFG.rho  # rho n scale = rho diag(M_zz) + 1e-6 n
        rust = matrix.centred_query(np.ascontiguousarray(z_q[i]), start, end, CFG.rho, np.ascontiguousarray(scale))
        # The bound: the prefix difference over E entries errs by about E u max|raw moment|, the centring
        # multiplies it by (1 + r^2) relative to the centred moments (r = |zbar| / sd), the solve by kappa(A).
        n = end - start
        a = stats.m_xx + np.diag(CFG.rho * n * scale)
        kappa = np.linalg.cond(a)
        r2 = float(np.max(stats.mean_x**2 / var))
        size = np.abs(rust.mean_y) + np.abs(rust.beta) @ np.abs(z_q[i] - stats.mean_x)
        tol = 64 * U64 * kappa * (end / n) * (1 + r2) * size
        err = np.abs(ro.m[i].numpy() - rust.forecast)
        assert np.all(err <= tol), (i, err, tol)
        worst = max(worst, float((err / tol).max()))
    assert decisions.size > 1000 and worst > 0


def test_an_empty_window_gives_one_third_each():
    ep = episode(hours=24 * 3)
    early = np.array([ep["t0"] - 1, ep["t0"] + 10 * HOUR_MS])  # before any label is known
    *_, ro = torch_readouts(ep, ep["lka"], early, np.ones((2, 8)))
    assert torch.equal(ro.m, torch.full((2, 3), 1 / 3, dtype=torch.float64)) and torch.equal(ro.count, torch.zeros(2, dtype=torch.float64))


def test_prefix_sums_are_sequential_and_never_see_later_entries():
    ep = episode(hours=24 * 10)
    z, y = torch.from_numpy(ep["z"]), torch.from_numpy(onehot(ep["label"]))
    full = prefix_moments(z, y)
    for k in (1, 17, 500, z.shape[0] - 1):
        part = prefix_moments(z[:k], y[:k])
        assert torch.equal(part.s_zz, full.s_zz[: k + 1]) and torch.equal(part.s_yz, full.s_yz[: k + 1]) and torch.equal(part.s_z, full.s_z[: k + 1])
    # bit for bit the sequential sum
    zz = (ep["z"][:, :, None] * ep["z"][:, None, :]).reshape(-1, 64)
    assert np.array_equal(full.s_zz.reshape(-1, 64)[1:].numpy(), np.add.accumulate(zz, axis=0))
    assert np.array_equal(full.s_z[1:].numpy(), np.add.accumulate(ep["z"], axis=0))
    with pytest.raises(ValueError, match="CPU"):
        prefix_moments(z.to("meta"), y.to("meta"))


def test_a_readout_at_d_does_not_move_when_the_entries_after_d_change():
    ep = episode(hours=24 * 50)
    decisions = np.unique(ep["decision"])
    z_q = np.random.default_rng(2).normal(size=(decisions.size, 8))
    order, st, _, _, ro = torch_readouts(ep, ep["lka"], decisions, z_q)
    cut = decisions[decisions.size // 2]
    later = ep["lka"] > cut
    other = dict(ep)
    other["z"] = np.where(later[:, None], ep["z"] * 3.0 + 5.0, ep["z"])
    other["label"] = np.where(later, (ep["label"] + 1) % 3, ep["label"])
    *_, ro2 = torch_readouts(other, other["lka"], decisions, z_q)
    before = decisions <= cut
    assert torch.equal(ro.m[before], ro2.m[before])  # to the bit
    assert not torch.equal(ro.m[~before], ro2.m[~before])


def test_a_query_reads_the_same_bits_alone_or_in_a_batch():
    ep = episode(hours=24 * 40)
    decisions = np.unique(ep["decision"])
    z_q = np.random.default_rng(3).normal(size=(decisions.size, 8))
    *_, ro = torch_readouts(ep, ep["lka"], decisions, z_q)
    for i in (0, 100, decisions.size - 1):
        *_, one = torch_readouts(ep, ep["lka"], decisions[i : i + 1], z_q[i : i + 1])
        assert torch.equal(one.m[0], ro.m[i])


def test_cheat_stamp_differs_from_the_honest_memory_exactly_where_the_entry_sets_differ():
    ep = episode(members=4, hours=24 * 3)  # rows only in the first three days: later, both memories hold every row
    decisions = ep["t0"] + HOUR_MS * np.arange(24 * 40, dtype=np.int64) + HOUR_MS - 1
    z_q = np.random.default_rng(4).normal(size=(decisions.size, 8))
    oh, sh, loh, hih, honest = torch_readouts(ep, ep["lka"], decisions, z_q)
    oc, sc, loc, hic, cheat = torch_readouts(ep, ep["decision"], decisions, z_q)
    w = CFG.window_ms
    differs = (honest.m != cheat.m).any(dim=1).numpy()
    for i, d in enumerate(decisions):
        h_set = set(oh[int(loh[i]) : int(hih[i])].tolist())
        c_set = set(oc[int(loc[i]) : int(hic[i])].tolist())
        # cheat-stamp holds, beyond the honest set, exactly the rows decided by d whose label is known after d
        extra = {j for j in range(ep["z"].shape[0]) if d - w < ep["decision"][j] <= d and ep["lka"][j] > d}
        missing = {j for j in range(ep["z"].shape[0]) if d - w < ep["lka"][j] <= d and ep["decision"][j] <= d - w}
        assert c_set - h_set == extra and h_set - c_set == missing
        assert bool(differs[i]) == (c_set != h_set)
    assert differs.any() and (~differs).any()  # both cases occur


def test_gradcheck_through_the_memory_readout():
    rng = np.random.default_rng(5)
    e, p = 12, 3
    z = torch.tensor(rng.normal(size=(e, p)) + 1.0, requires_grad=True)
    y = torch.from_numpy(onehot(rng.integers(0, 3, e)))
    z_q = torch.tensor(rng.normal(size=(4, p)), requires_grad=True)
    stamps = torch.arange(e, dtype=torch.int64) * 10
    as_of = torch.tensor([35, 60, 95, 119])
    cfg = MemoryConfig(window_ms=80, key_dim=p)

    def f(zz, qq):
        prefix = prefix_moments(zz, y)
        lo, hi = window_bounds(stamps, as_of, cfg.window_ms)
        return readout(prefix, lo, hi, qq, cfg).m

    assert torch.autograd.gradcheck(f, (z, z_q))


def test_unsorted_entries_are_refused():
    with pytest.raises(ValueError, match="sorted"):
        window_bounds(torch.tensor([3, 1, 2]), torch.tensor([2]), 10)


def test_the_window_is_thirty_days_and_rho_one_hundredth():
    assert CFG.window_ms == 30 * DAY_MS and CFG.rho == 1e-2 and CFG.jitter == 1e-6 and CFG.key_dim == 8
