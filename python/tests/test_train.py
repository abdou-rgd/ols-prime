"""Training of the encoder arms: the fit's rows and limits, the schedule, determinism, batches and checkpoints (CPU torch)."""

import dataclasses
import json
import math
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
import torch

from olsprime.models.encoder import EncoderConfig, Windows
from olsprime.models.matrix import prob_map
from olsprime.train import checkpoint
from olsprime.train.determinism import CUBLAS_WORKSPACE_CONFIG, deterministic
from olsprime.train.fit import (
    DAY_MS,
    HOUR_MS,
    EncoderData,
    EncoderSystem,
    FitConfig,
    encoder_data,
    ensemble,
    lr_at,
    past_window_labels,
    rows_of_year,
    week_start,
    year_start_ms,
)
from olsprime.train.synthetic import small_panel

F = int(datetime(2018, 4, 1, tzinfo=UTC).timestamp() * 1000)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """One intra-op thread: spinning OpenMP threads slow down by orders of magnitude when another process shares the cores."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def ms(*args) -> int:
    return int(datetime(*args, tzinfo=UTC).timestamp() * 1000)


@pytest.fixture(scope="module")
def panel_and_bars():
    return small_panel(hours=24 * 100)


@pytest.fixture(scope="module")
def data(panel_and_bars) -> EncoderData:
    return encoder_data(*panel_and_bars)


def small_cfg(**overrides) -> FitConfig:
    base = {"fit_ms": F, "seed": 1, "encoder": EncoderConfig(arm="enc-ols"), "epochs": 2, "warmup_steps": 5, "eval_batch": 256}
    return FitConfig(**(base | overrides))


@pytest.fixture(scope="module")
def honest_fit(data):
    with deterministic():
        system = EncoderSystem(data, small_cfg())
        return system, system.fit()


class Planted(EncoderSystem):
    """Labels flipped on the rows whose label becomes known in (lo, hi]: an extreme change of the outcome."""

    lo, hi = 0, 0

    def label_array(self):
        a = self.data.arrays
        hit = (a.lka_ms > self.lo) & (a.lka_ms <= self.hi) & (a.label >= 0)
        return np.where(hit, (a.label + 1) % 3, a.label)


def planted_fit(data, lo, hi):
    cls = type("P", (Planted,), {"lo": lo, "hi": hi})
    with deterministic():
        return cls(data, small_cfg()).fit()


def same_states(r1, r2) -> bool:
    return all(all(np.array_equal(a[k], b[k]) for k in a) for a, b in zip(r1.epoch_states, r2.epoch_states, strict=True))


# --- Calendar and schedule -------------------------------------------------------------------------


def test_episodes_are_calendar_weeks_from_monday_midnight_utc():
    monday = ms(2018, 1, 1)  # a Monday
    assert datetime(2018, 1, 1).weekday() == 0
    times = np.array([monday, monday + 7 * DAY_MS - 1, monday + 7 * DAY_MS, monday - 1])
    assert week_start(times).tolist() == [monday, monday, monday + 7 * DAY_MS, monday - 7 * DAY_MS]


def test_the_learning_rate_warms_up_for_100_steps_then_decays_to_1e_4_at_the_last_step():
    cfg = FitConfig(fit_ms=F, seed=1)
    total = 6 * 300
    assert lr_at(0, total, cfg) == pytest.approx(1e-5) and lr_at(99, total, cfg) == pytest.approx(1e-3)
    assert lr_at(total - 1, total, cfg) == pytest.approx(1e-4, rel=1e-12)
    mid = 100 + (total - 100) // 2 - 1
    assert lr_at(mid, total, cfg) == pytest.approx(1e-4 + 0.9e-3 * 0.5 * (1 + math.cos(math.pi * (mid - 99) / (total - 100))))
    rates = [lr_at(s, total, cfg) for s in range(100, total)]
    assert all(b <= a for a, b in zip(rates, rates[1:]))
    assert cfg.betas == (0.9, 0.999) and cfg.weight_decay == 0.01 and cfg.grad_clip == 1.0 and cfg.epochs == 6


def test_the_registered_fits_and_the_rows_of_a_year():
    from olsprime.train.fit import FIT_YEARS, SEEDS

    assert FIT_YEARS == (2019, 2020, 2021, 2022, 2023, 2024) and SEEDS == (1, 2, 3)
    panel, bars = small_panel(start=datetime(2018, 12, 20, tzinfo=UTC), hours=24 * 30)
    d = encoder_data(panel, bars)
    rows = rows_of_year(d.arrays, 2019)
    assert np.all(d.arrays.month_ms[rows] >= year_start_ms(2019)) and np.all(d.arrays.complete[rows])
    # a row's year follows its month: the bar opening at 2018-12-31 23:00 is decided in 2019 but belongs to 2018
    last_2018 = np.flatnonzero(d.arrays.open_ms == ms(2018, 12, 31, 23))
    assert last_2018.size and np.all(d.arrays.decision_ms[last_2018] < year_start_ms(2019))
    assert not np.isin(last_2018, rows).any()


# --- The fit's rows and limits ---------------------------------------------------------------------


def test_the_fit_trains_before_f_minus_30_days_and_stops_on_a_purged_slice(honest_fit, data):
    system, res = honest_fit
    a = data.arrays
    assert a.lka_ms[res.train_rows].max() <= F - 30 * DAY_MS
    assert np.all(a.decision_ms[res.slice_rows] >= F - 30 * DAY_MS) and np.all(a.decision_ms[res.slice_rows] < F)
    assert a.lka_ms[res.slice_rows].max() <= F - 24 * HOUR_MS
    assert not np.intersect1d(res.train_rows, res.slice_rows).size
    assert np.all(data.usable[res.train_rows]) and np.all(data.usable[res.slice_rows])
    assert len(res.slice_losses) == 2 and res.kept_epoch == 1 + int(np.argmin(res.slice_losses))
    assert res.steps == 2 * res.weeks.size and all(sorted(o.tolist()) == list(range(res.weeks.size)) for o in res.orders)
    clim, n = system.climatology()
    known = a.labelled & (a.lka_ms <= F - 24 * HOUR_MS)
    assert n == int(known.sum()) and np.allclose(clim, prob_map(np.bincount(a.label[known], minlength=3) / n, 1e-3))


def test_a_label_known_just_after_f_minus_30_days_never_reaches_the_training(data, honest_fit):
    _, base = honest_fit
    late = planted_fit(data, F - 30 * DAY_MS, F - 30 * DAY_MS + 6 * HOUR_MS)
    assert same_states(base, late)  # every parameter of every epoch, to the bit
    early = planted_fit(data, F - 30 * DAY_MS - 6 * HOUR_MS, F - 30 * DAY_MS)  # positive control: just inside
    assert not same_states(base, early)


def test_a_label_known_just_after_f_minus_24_hours_never_reaches_the_slice(data, honest_fit):
    _, base = honest_fit
    late = planted_fit(data, F - 24 * HOUR_MS, F - 24 * HOUR_MS + 6 * HOUR_MS)
    assert late.slice_losses == base.slice_losses and late.kept_epoch == base.kept_epoch
    assert np.array_equal(late.climatology, base.climatology) and same_states(base, late)
    early = planted_fit(data, F - 24 * HOUR_MS - 6 * HOUR_MS, F - 24 * HOUR_MS)  # positive control: just inside
    assert early.slice_losses != base.slice_losses


def test_cheat_purge_is_a_parameter_of_the_training_horizon(data):
    honest = EncoderSystem(data, small_cfg()).training_mask()
    purge = EncoderSystem(data, small_cfg(train_gap_ms=0)).training_mask()
    a = data.arrays
    assert a.lka_ms[purge].max() > F - 30 * DAY_MS and a.lka_ms[purge].max() <= F
    assert np.all(purge[honest]) and FitConfig(fit_ms=F, seed=1).train_gap_ms == 30 * DAY_MS


# --- Determinism, batches and the future -----------------------------------------------------------


def test_two_fits_with_the_same_seed_give_the_same_bits(data, honest_fit):
    system, first = honest_fit
    with deterministic():
        second = EncoderSystem(data, small_cfg()).fit()
        other = EncoderSystem(data, small_cfg(seed=2)).fit()
    assert same_states(first, second) and first.slice_losses == second.slice_losses
    assert all(np.array_equal(a, b) for a, b in zip(first.orders, second.orders, strict=True))
    rows = np.flatnonzero(data.usable & (data.arrays.decision_ms >= F))
    f1, f2 = system.forecast(first, rows), system.forecast(second, rows)
    assert np.array_equal(f1["raw"], f2["raw"]) and np.array_equal(f1["m"], f2["m"])
    assert not same_states(first, other) and not all(np.array_equal(a, b) for a, b in zip(first.orders, other.orders, strict=True))


@pytest.mark.parametrize(("arm", "batch"), [("enc-ols", 256), ("enc-linear", 4096)])
def test_the_forecasts_of_a_row_set_equal_those_of_every_prefix_to_the_bit(data, arm, batch):
    """The scope test of audit pass 4a: a row's outputs never depend on which other rows are in the run."""
    with deterministic():
        system = EncoderSystem(data, small_cfg(encoder=EncoderConfig(arm=arm), epochs=1, eval_batch=batch))
        res = system.fit()
        rows = np.flatnonzero(data.usable & (data.arrays.decision_ms >= F - 20 * DAY_MS))
        full = system.forecast(res, rows)
        for k in (1, 37, batch - 1, batch + 1, rows.size - 1):
            if k >= rows.size:
                continue
            part = system.forecast(res, rows[:k])
            for key in ("raw", "m", "count", "mean_z", "beta", "window_lo", "window_hi"):
                assert np.array_equal(part[key], full[key][:k]), (k, key)


def perturbed_data(panel_and_bars, cut_ms: int, factor: float) -> EncoderData:
    """Every bar opening after the cut: prices and volumes multiplied by ``factor`` (the future moves)."""
    _, bars = panel_and_bars
    from olsprime.features.panel import assemble, pair_rows

    new = {}
    for symbol, frame in bars.items():
        after = pl.col("open_time").dt.epoch("ms") > cut_ms
        cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
        new[symbol] = frame.with_columns([pl.when(after).then(pl.col(c) * factor).otherwise(pl.col(c)).alias(c) for c in cols])
    parts = [pair_rows(f, sorted(set(f["open_time"].dt.truncate("1mo").to_list()))) for f in new.values()]
    return encoder_data(assemble(parts), new)


class StampedAtDecision(EncoderSystem):
    """cheat-stamp: an entry joins the memory at its row's decision time, before its label is known."""

    def entry_stamps(self, rows):
        return self.data.arrays.decision_ms[rows]


@pytest.mark.parametrize(("system_cls", "honest"), [(EncoderSystem, True), (StampedAtDecision, False)])
def test_a_future_that_moves_leaves_every_output_up_to_the_cut_unchanged_unless_the_memory_cheats(panel_and_bars, data, system_cls, honest):
    cut = F + 4 * DAY_MS + 5 * HOUR_MS  # the data end on 2018-04-11
    outs = []
    for d in (data, perturbed_data(panel_and_bars, cut, 3.0)):
        with deterministic():
            system = system_cls(d, small_cfg(epochs=1))
            res = system.fit()
            rows = np.flatnonzero(d.usable & (d.arrays.decision_ms >= F) & (d.arrays.decision_ms <= cut))
            outs.append((res, rows, system.forecast(res, rows)))
    (r1, rows1, f1), (r2, rows2, f2) = outs
    assert np.array_equal(rows1, rows2) and same_states(r1, r2)
    same = np.array_equal(f1["raw"], f2["raw"]) and np.array_equal(f1["m"], f2["m"])
    assert same == honest


# --- The pieces ------------------------------------------------------------------------------------


def test_a_row_without_a_valid_window_takes_the_climatology(data, honest_fit):
    system, res = honest_fit
    rows = np.flatnonzero(data.usable & (data.arrays.decision_ms >= F))[:50]
    ok = data.windows.ok.copy()
    ok[rows[::7]] = False
    broken = EncoderData(data.arrays, dataclasses.replace(data.windows, ok=ok), data.threshold)
    out = EncoderSystem(broken, system.cfg).forecast(res, rows)
    assert out["fallback"].sum() == rows[::7].size
    assert np.array_equal(out["raw"][out["fallback"]], np.tile(res.climatology, (rows[::7].size, 1)))


def test_the_seed_ensemble_averages_before_the_floor():
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.5, 0.5]])
    assert np.allclose(ensemble([a, b]), prob_map(np.array([[0.5, 0.25, 0.25]]), 1e-3))


def test_one_episode_can_be_overfitted(data):
    system = EncoderSystem(data, small_cfg())
    a = data.arrays
    train = np.flatnonzero(system.training_mask())
    week = week_start(a.decision_ms[train])
    rows = train[week == np.unique(week)[2]][:96]  # four days of one week: the batch to memorize
    pool_rows, pool_stamps = system.sorted_pool(train)
    from olsprime.models.encoder import Standardizer

    bars = system.standardized_bars(Standardizer.fit(data.windows, train))
    torch.manual_seed(0)
    model = system.new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for _ in range(150):
        loss = system.episode_loss(model, bars, rows, pool_rows, pool_stamps)
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first > 0.9 and float(loss.detach()) < 0.1, (first, float(loss.detach()))


class PosWindow(EncoderSystem):
    """The positive control's label hook, as the bench would plant it."""

    def label_array(self):
        return past_window_labels(self.data)


def test_the_positive_control_label_is_the_class_of_the_last_24_hours(data, panel_and_bars):
    panel, bars = panel_and_bars
    y = past_window_labels(data)
    rows = np.flatnonzero(data.usable)[::97]
    for r in rows:
        frame = bars[panel["symbol"][int(r)]]
        t = int(np.searchsorted(frame["open_time"].dt.epoch("ms").to_numpy(), data.arrays.open_ms[r]))
        ret = math.log(frame["close"][t] / frame["close"][t - 24])
        thr = data.threshold[r]
        assert y[r] == (0 if ret < -thr else 2 if ret > thr else 1)
    system = PosWindow(data, small_cfg())
    assert np.array_equal(system.y, y) and np.array_equal(system.entry_stamps(rows), data.arrays.lka_ms[rows])  # entries keep label_known_at


def test_the_encoder_learns_the_positive_control_from_its_window(data):
    """pos-window: the class of the 24 hours that end at bar t is in the window, so a fit must find it.

    On this small CPU panel the softmax arm learns it within 6 epochs at lr 10^-2; the OLS arm does not
    (see the sizing report). The test checks that the label hook reaches training, memory and scores.
    """
    with deterministic():
        system = PosWindow(data, small_cfg(encoder=EncoderConfig(arm="enc-softmax"), epochs=6, lr=1e-2))
        res = system.fit()
    y = system.y[res.slice_rows]
    clim_loss = float(-np.log(res.climatology[y]).mean())
    assert min(res.slice_losses) < 0.85 * clim_loss, (res.slice_losses, clim_loss)


def test_every_epochs_parameters_are_kept_with_their_digests(honest_fit):
    _, res = honest_fit
    assert len(res.epoch_states) == len(res.epoch_digests()) == 2
    assert res.state is res.epoch_states[res.kept_epoch - 1]
    assert res.epoch_digests()[0] != res.epoch_digests()[1]


def test_cheat_slice_trains_on_its_own_slice(data):
    rows = EncoderSystem(data, small_cfg(train_gap_ms=24 * HOUR_MS)).training_mask()
    slice_rows = EncoderSystem(data, small_cfg()).slice_mask()
    assert np.all(rows[slice_rows])


def test_a_fit_saves_without_pickle_to_the_same_bytes(honest_fit, tmp_path):
    _, res = honest_fit
    h1 = res.save(tmp_path / "a")
    h2 = res.save(tmp_path / "b")
    raw = (tmp_path / "a" / "fit.safetensors").read_bytes()
    assert h1 == h2 and raw == (tmp_path / "b" / "fit.safetensors").read_bytes()
    arrays, meta = checkpoint.load(tmp_path / "a" / "fit.safetensors")
    for name, value in res.state.items():
        assert np.array_equal(arrays[f"param.{name}"], value) and arrays[f"param.{name}"].dtype == value.dtype
    assert json.loads(meta["config"]) == dataclasses.asdict(res.config) | {"betas": list(res.config.betas)}
    assert int(meta["kept_epoch"]) == res.kept_epoch
    with pytest.raises(ValueError):
        checkpoint.from_bytes(raw[:20])


def test_the_determinism_settings_are_the_registered_ones_and_are_restored():
    before = torch.are_deterministic_algorithms_enabled()
    with deterministic():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cudnn.benchmark
        import os

        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled() == before


def test_windows_type_is_shared():
    assert Windows.__dataclass_fields__.keys() >= {"bars", "end", "ok"}
