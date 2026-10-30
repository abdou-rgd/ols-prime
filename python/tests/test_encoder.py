"""The encoder arms of iteration 4 against closed forms (CPU torch, so that CI runs them)."""

import math

import numpy as np
import polars as pl
import pytest
import torch

from olsprime.models.encoder import (
    ARMS,
    QUANTITIES,
    WINDOW,
    EncoderConfig,
    EncoderModel,
    Standardizer,
    bar_quantities,
    build_windows,
    count_parameters,
    decay_groups,
    expected_parameters,
    linear_attention,
    ols_attention,
    ols_condition,
    ols_weights,
    rms_normalize,
    row_sum_tolerance,
    softmax_attention,
    window_flags,
)
from olsprime.train.synthetic import small_panel

U32 = 2.0**-24


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One intra-op thread: spinning OpenMP threads slow down by orders of magnitude when another process shares the cores."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def qkv(seed: int, b: int = 2, h: int = 4, t: int = 64, d: int = 8, dtype=torch.float32, mean: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(b, h, t, d, generator=g, dtype=torch.float64) + mean).to(dtype) for _ in range(3)]


def numpy_ridge_forecast(q: np.ndarray, k: np.ndarray, v: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge with an unpenalized constant coordinate, in fp64: keys and queries RMS-normalized, Lambda = diag(0, T lam)."""
    b, h, t, d = q.shape
    out, weights = np.empty_like(v), np.empty((b, h, t, t))
    for i in range(b):
        for j in range(h):
            kk = k[i, j] / np.sqrt((k[i, j] ** 2).mean(axis=1, keepdims=True))
            qq = q[i, j] / np.sqrt((q[i, j] ** 2).mean(axis=1, keepdims=True))
            x = np.hstack([np.ones((t, 1)), kk])
            xq = np.hstack([np.ones((t, 1)), qq])
            penalty = np.diag([0.0] + [t * lam[j]] * d)
            beta = np.linalg.solve(x.T @ x + penalty, x.T @ v[i, j]).T  # (d_out, d + 1): the ridge coefficients
            out[i, j] = xq @ beta.T
            weights[i, j] = xq @ np.linalg.solve(x.T @ x + penalty, x.T)
    return out, weights


def ols_tolerance(w_ref: np.ndarray, v: np.ndarray, kappa: np.ndarray, t: int, d: int) -> np.ndarray:
    """First-order bound on |O - O_ref| for the fp32 head.

    Forming A errs by gamma_T, the Cholesky solve by gamma_{3(d+1)}, the RMS scaling of q and k by
    gamma_{d+3} (twice through A), the products by gamma_{d+1} and gamma_T: each output errs by about
    u (kappa (T + 3(d + 1) + 2(d + 3)) + 2(d + 1) + T) sum_j |W_ij| |V_j|. Factor 2 of margin.
    """
    scale = np.einsum("bhij,bhjc->bhic", np.abs(w_ref), np.abs(v))
    return 2 * U32 * (kappa[..., None, None] * (t + 3 * (d + 1) + 2 * (d + 3)) + 2 * (d + 1) + t) * scale


# --- The OLS head -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_ols_head_is_the_ridge_forecast_with_a_free_constant(seed):
    q, k, v = qkv(seed, mean=0.3 * seed)
    lam = torch.tensor([0.1, 0.01, 1.0, 0.05])
    out = ols_attention(q, k, v, lam).double().numpy()
    ref, w_ref = numpy_ridge_forecast(q.double().numpy(), k.double().numpy(), v.double().numpy(), lam.double().numpy())
    kappa = ols_condition(k, lam).numpy()
    tol = ols_tolerance(w_ref, v.double().numpy(), kappa, 64, 8)
    err = np.abs(out - ref)
    assert np.all(err <= tol), float((err / tol).max())
    assert err.max() > 0  # the comparison is not vacuous: fp32 against fp64


@pytest.mark.parametrize("order", ["assoc", "explicit"])
def test_both_evaluation_orders_give_the_ridge_forecast(order):
    q, k, v = qkv(7)
    lam = torch.full((4,), 0.1)
    out = ols_attention(q, k, v, lam, order=order).double().numpy()
    ref, w_ref = numpy_ridge_forecast(q.double().numpy(), k.double().numpy(), v.double().numpy(), lam.double().numpy())
    tol = ols_tolerance(w_ref, v.double().numpy(), ols_condition(k, lam).numpy(), 64, 8)
    assert np.all(np.abs(out - ref) <= tol)


@pytest.mark.parametrize(("seed", "lam_value", "collinear"), [(0, 0.1, False), (1, 1e-3, False), (2, 1e-3, True)])
def test_every_row_of_w_sums_to_one_within_its_derived_bound(seed, lam_value, collinear):
    q, k, _ = qkv(seed)
    if collinear:  # two nearly equal key coordinates: a larger condition number, a looser bound, still met
        k[..., 1] = k[..., 0] + 1e-3 * k[..., 1]
    lam = torch.full((4,), lam_value)
    w = ols_weights(q, k, lam)
    tol = row_sum_tolerance(w, k, lam)
    err = (w.sum(dim=-1) - 1).abs()
    assert torch.all(err <= tol), float((err / tol).max())
    # the bound is not loose by orders of magnitude: it stays within 10^4 of the observed error
    assert float(tol.max()) < 1e4 * max(float(err.max()), U32)


def test_w_is_the_numpy_ridge_weight_matrix():
    q, k, v = qkv(3, b=1)
    lam = torch.full((4,), 0.1)
    w = ols_weights(q, k, lam).double().numpy()
    _, w_ref = numpy_ridge_forecast(q.double().numpy(), k.double().numpy(), v.double().numpy(), lam.double().numpy())
    kappa = ols_condition(k, lam).numpy()[..., None, None]
    assert np.all(np.abs(w - w_ref) <= 2 * U32 * (kappa * (64 + 3 * 9 + 22) + 18 + 64) * np.abs(w_ref).sum(axis=-1, keepdims=True))


def test_keys_and_queries_are_scaled_to_unit_rms_without_a_parameter():
    x = torch.randn(3, 5, 8) * 7 + 2
    y = rms_normalize(x)
    assert torch.allclose(y.pow(2).mean(dim=-1), torch.ones(3, 5), atol=1e-6)
    # the operator does not depend on the keys' or queries' scale
    q, k, v = qkv(4)
    lam = torch.full((4,), 0.1)
    assert torch.allclose(ols_attention(q, k, v, lam), ols_attention(3 * q, 0.5 * k, v, lam), atol=1e-5)


def test_the_ols_head_solves_with_cholesky_and_never_an_inverse(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("an explicit inverse")

    monkeypatch.setattr(torch.linalg, "inv", refuse)
    monkeypatch.setattr(torch, "inverse", refuse)
    q, k, v = qkv(5)
    ols_attention(q, k, v, torch.full((4,), 0.1))
    ols_weights(q, k, torch.full((4,), 0.1))


def test_gradcheck_through_the_ols_solve():
    q, k, v = (t.requires_grad_() for t in qkv(6, b=1, h=1, t=12, d=2, dtype=torch.float64))
    lam = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b, c, l: ols_attention(a, b, c, l), (q, k, v, lam))
    assert torch.autograd.gradcheck(lambda a, b, c, l: ols_attention(a, b, c, l, order="explicit"), (q, k, v, lam))


def test_a_window_too_short_for_the_head_is_refused():
    with pytest.raises(ValueError, match="collapse"):
        EncoderConfig(arm="enc-ols", window=32)
    EncoderConfig(arm="enc-softmax", window=32)  # the other operators have no such constraint


def test_lambda_starts_at_one_tenth_and_stays_above_its_floor():
    model = EncoderModel(EncoderConfig())
    for block in model.encoder.blocks:
        assert torch.allclose(block.attn.lam(), torch.full((4,), 0.1), atol=1e-7)
        block.attn.theta.data.fill_(-100.0)
        assert torch.all(block.attn.lam() >= 1e-3)


# --- The other operators -------------------------------------------------------------------------


def test_the_softmax_operator_is_scaled_dot_product_attention():
    q, k, v = qkv(8, dtype=torch.float64)
    ref = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(8), dim=-1) @ v
    assert torch.allclose(softmax_attention(q, k, v), ref, atol=1e-12)


def test_the_linear_operator_is_katharopoulos():
    q, k, v = (t.numpy() for t in qkv(9, dtype=torch.float64))
    phi = lambda x: np.where(x > 0, x + 1, np.exp(x))  # noqa: E731  elu(x) + 1
    num = phi(q) @ (np.swapaxes(phi(k), -1, -2) @ v)
    den = phi(q) @ np.swapaxes(phi(k), -1, -2) @ np.ones(k.shape[:-1] + (1,))
    out = linear_attention(*(torch.from_numpy(t) for t in (q, k, v))).numpy()
    assert np.allclose(out, num / den, rtol=1e-12, atol=0)


# --- The arms ------------------------------------------------------------------------------------


def test_parameter_counts_equal_but_for_the_eight_lambdas():
    counts = {arm: count_parameters(EncoderModel(EncoderConfig(arm=arm))) for arm in ARMS}
    assert counts["enc-softmax"] == counts["enc-linear"] == 20995
    assert counts["enc-ols"] == counts["enc-softmax"] + 8
    assert (counts["enc-ols"] - counts["enc-softmax"]) / counts["enc-softmax"] < 1e-3
    for arm in ARMS:
        assert counts[arm] == expected_parameters(EncoderConfig(arm=arm))
    names = dict(EncoderModel(EncoderConfig()).named_parameters())
    assert sum(p.numel() for n, p in names.items() if n.endswith("theta")) == 8


def test_the_arms_share_every_weight_but_the_operator():
    shapes = {arm: {n: tuple(p.shape) for n, p in EncoderModel(EncoderConfig(arm=arm)).named_parameters()} for arm in ARMS}
    ols_only = {n for n in shapes["enc-ols"] if n.endswith("theta")}
    assert {n: s for n, s in shapes["enc-ols"].items() if n not in ols_only} == shapes["enc-softmax"] == shapes["enc-linear"]


def test_the_same_seed_gives_every_arm_the_same_shared_weights():
    states = {}
    for arm in ARMS:
        torch.manual_seed(1)
        states[arm] = EncoderModel(EncoderConfig(arm=arm)).state_dict()
    for name, value in states["enc-softmax"].items():
        assert torch.equal(value, states["enc-linear"][name]) and torch.equal(value, states["enc-ols"][name])


def test_weight_decay_applies_to_the_weight_matrices_only():
    model = EncoderModel(EncoderConfig())
    decay, rest = decay_groups(model, 0.01)
    named = {id(p): n for n, p in model.named_parameters()}
    assert decay["weight_decay"] == 0.01 and rest["weight_decay"] == 0.0
    assert all(named[id(p)].endswith("weight") and p.dim() == 2 for p in decay["params"])
    rest_names = {named[id(p)] for p in rest["params"]}
    assert "encoder.position" in rest_names and any(n.endswith("theta") for n in rest_names)
    assert not any(n.endswith("weight") and "ln" not in n for n in rest_names)


@pytest.mark.parametrize("arm", ARMS)
def test_every_parameter_receives_a_finite_gradient(arm):
    torch.manual_seed(0)
    model = EncoderModel(EncoderConfig(arm=arm))
    x = torch.randn(16, WINDOW, 8)
    h, z = model.encode(x)
    m = torch.softmax(z[:, :3], dim=-1)  # the memory's readout depends on z
    loss = torch.nn.functional.cross_entropy(model.decide(h, m), torch.arange(16) % 3)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_the_ols_encoder_exposes_w_and_its_rows_sum_to_one():
    torch.manual_seed(0)
    model = EncoderModel(EncoderConfig())
    x = torch.randn(4, WINDOW, 8)
    with torch.no_grad():
        ws = model.encoder.ols_weights(x)
    assert len(ws) == 2 and ws[0].shape == (4, 4, WINDOW, WINDOW)
    hidden = model.encoder.embed(x) + model.encoder.position
    with torch.no_grad():
        for block, w in zip(model.encoder.blocks, ws, strict=True):
            _, k, _ = block.attn.project(block.ln1(hidden))
            err = (w.sum(-1) - 1).abs()
            assert torch.all(err <= row_sum_tolerance(w, k, block.attn.lam()))
            hidden = block(hidden)
    with pytest.raises(ValueError, match="only the OLS"):
        EncoderModel(EncoderConfig(arm="enc-softmax")).encoder.ols_weights(x)


def test_cheat_value_is_off_by_default_and_reads_the_label_when_planted():
    honest = EncoderModel(EncoderConfig())
    assert EncoderConfig().value_leak is False
    assert not any("leak" in n for n, _ in honest.named_parameters())
    x = torch.randn(6, WINDOW, 8)
    labels = torch.nn.functional.one_hot(torch.arange(6) % 3, 3).float()
    with pytest.raises(ValueError, match="cheat-value"):
        honest.encode(x, labels)
    torch.manual_seed(0)
    leaky = EncoderModel(EncoderConfig(value_leak=True))
    assert count_parameters(leaky) == expected_parameters(EncoderConfig(value_leak=True)) == 21003 + 2 * 3 * 32
    with pytest.raises(ValueError, match="one-hot labels"):
        leaky.encode(x)
    h1, _ = leaky.encode(x, labels)
    h2, _ = leaky.encode(x, labels.roll(1, dims=1))
    assert not torch.allclose(h1, h2)


# --- Windows -------------------------------------------------------------------------------------


def test_the_eight_bar_quantities_have_their_closed_form():
    o, h, l, c = np.array([10.0, 11.0]), np.array([12.0, 12.5]), np.array([9.0, 10.5]), np.array([11.0, 10.8])
    q, b = np.array([100.0, 0.0]), np.array([70.0, 0.0])
    out = bar_quantities(o, h, l, c, q, b)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    expected = [math.log(10.8 / 11.0), math.log(11.0 / 11.0), math.log(12.5 / 10.5), math.log(10.8 / 11.0),
                math.log(12.5 / 11.0), math.log(10.8 / 10.5), math.log(1.0), 0.0]  # fmt: skip
    assert np.allclose(out[1], expected, rtol=0, atol=1e-15)
    assert out[0, 7] == pytest.approx(0.2) and out[0, 6] == pytest.approx(math.log(101.0))
    assert len(QUANTITIES) == 8


def test_a_gap_or_a_bad_price_flags_the_windows_that_reach_it():
    n, step = 300, 3_600_000
    open_ms = np.arange(n, dtype=np.int64) * step
    open_ms[150:] += step  # a missing hour between rows 149 and 150
    q = np.zeros((n, 8))
    q[0, :2] = np.nan
    ok = window_flags(open_ms, q)
    assert not ok[:64].any()  # no full window yet
    assert ok[64:150].all()
    assert not ok[150:214].any()  # every window whose 65 bars straddle the gap
    assert ok[214:].all()
    q[250, 3] = np.nan  # a bar with an undefined quantity
    ok = window_flags(open_ms, q)
    assert not ok[250:314].any() and ok[249]


def test_windows_are_built_from_the_bars_pair_rows_receives():
    panel, bars = small_panel(hours=24 * 20, gap_at=200)
    win = build_windows(bars, panel)
    rows = np.flatnonzero(win.ok)
    assert rows.size > 0 and (~win.ok).sum() > 0
    rng = np.random.default_rng(0)
    for r in rng.choice(rows, 10, replace=False):
        symbol = panel["symbol"][int(r)]
        frame = bars[symbol]
        t = int(np.searchsorted(frame["open_time"].dt.epoch("ms").to_numpy(), panel["open_time"].dt.epoch("ms")[int(r)]))
        cols = [frame[c].to_numpy()[t - 64 : t + 1] for c in ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")]
        direct = bar_quantities(*cols)[1:]
        assert np.array_equal(win.bars[win.end[r] - 63 : win.end[r] + 1], direct)
    # the rows next to the planted gap are flagged, never filled
    first = panel.filter(pl.col("symbol") == sorted(bars)[0])
    gap_open = bars[sorted(bars)[0]]["open_time"][200]
    after = np.flatnonzero((panel["symbol"] == sorted(bars)[0]).to_numpy() & (panel["open_time"] >= gap_open).to_numpy())[:64]
    assert first.height > 0 and not win.ok[after].any()


def test_standardization_is_a_function_of_the_training_rows_alone():
    panel, bars = small_panel(hours=24 * 20)
    win = build_windows(bars, panel)
    rows = np.flatnonzero(win.ok)
    train = rows[: rows.size // 2]
    std = Standardizer.fit(win, train)
    last = win.bars[win.end[train]]
    assert np.allclose(std.mean, last.mean(axis=0), rtol=1e-12) and np.allclose(std.std, last.std(axis=0), rtol=1e-12)
    # another row's bars do not move it, by a single bit
    other = win.bars.copy()
    untouched = np.setdiff1d(np.arange(win.bars.shape[0]), win.end[train])
    other[untouched] *= 3.0
    again = Standardizer.fit(type(win)(win.symbols, other, win.open_ms, win.end, win.ok), train)
    assert np.array_equal(again.mean, std.mean) and np.array_equal(again.std, std.std)
    clipped = std.apply(np.vstack([std.mean + 100 * std.std, std.mean - 100 * std.std]))
    assert np.array_equal(clipped, np.array([[8.0] * 8, [-8.0] * 8], dtype=np.float32))
