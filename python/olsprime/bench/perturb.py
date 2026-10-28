"""The perturbation of the future-invariance test: change every value, and every ratio of two values, after a cut.

For a cut at row ``t`` of one pair's bars, every bar ``s > t`` is changed and every bar
``s <= t`` is returned untouched, bit for bit. **Every column a model could read moves, and so
does every ratio of two columns of a bar**; :func:`perturb_after` asserts both at run time
(:func:`check_coverage`).

**Bar t + 1, designed against the recent past.** The bar a one-bar-ahead leak reads is not
drawn but built from the window ``W`` of bar ``t`` and the ``WINDOW = 90`` bars before it (the
longest feature window). Each quantity ``f`` of bar t + 1 below has a **high run** and a **low
run**:

- high: ``f >= PUSH * max_W f`` (``PUSH = 1.5``) and ``f >= 3 * f_original``;
- low: ``f <= min_W f / PUSH`` (over the window's positive values) and ``f <= f_original / 3``.

So a comparison ``f[t+1] > f[t-k]``, for any ``0 <= k <= 90``, is true in one run and false in
the other, **whatever its original sign and its size**:

- high in the up run (level > 1): every price (the up bar sits above every high of the window,
  the down bar below every low), the VWAP ``quote_volume/volume``, the taker VWAP
  ``taker_buy_quote/taker_buy_base``, ``volume``, ``quote_volume``, ``trades``, the trade size
  ``volume/trades`` and the taker shares of base and of quote (bounded by 1: high means
  ``1 - s <= (1 - max s) / PUSH`` over the window's values inside (0, 1));
- high in the down run: the body ``|ln(close/open)|``, the upper wick ``ln(high/max(open,
  close))``, the lower wick ``ln(min(open, close)/low)``, and with them the range ``ln(high/low)``;
- high in the up run, **against bar t alone**: the fill ``body / range`` and the close location
  ``(close - low) / (high - low)``, pushed like shares (a near-doji anywhere in the window would
  otherwise ask for an unbounded range).

**The shape, and the cut's parity.** At even cuts of the bench (``Perturbation.mirror`` false)
the up bar is small, green and body-dominant: fill ``>= UP_FILL = 0.8`` (aimed just above), close
location ``>= UP_CLV = 0.75``, body ``<= 0.4 %``, each wick ``<= 0.05 %``; the down bar is big,
red and upper-wick-dominant: fill ``<= DOWN_FILL = 0.2`` and never below ``FILL_FLOOR = 0.02``
(exact: the body grows if the wicks would take the fill below it), close location
``<= DOWN_CLV = 0.25``, upper wick ``>= 3 x`` the lower, body ``>= 10 %``, each wick ``>= 1 %``.
At odd cuts (``mirror`` true) colour and close location swap: the up bar is small and red,
closing near its low (close location ``<= 0.25``); the down bar is big and green,
lower-wick-dominant (lower ``>= 3 x`` upper), closing near its high (``>= 0.75``). Its close
location tends to ``exp(-upper wick)``, so its upper wick is capped at ``UW_CAP_SHARE * -ln C``
(:func:`mirrored_upper_wick_cap`): against bar t it still crosses, or the cut is a declared
``?``; against older bars it is not guaranteed. The close-location target of the big bar never
goes past ``CLV_FLOOR = 0.02`` (``0.98`` mirrored), which bounds the wick it asks for. The four
colour x size cells (:data:`CELLS`) are each realised at half the cuts. The freedom left (the
split of the up run's wick budget, the factors of the down run's body and wicks) is drawn per
run, so other shape functions (the wick ratio, a hammer) flip with high probability only.

**The bands of the thresholds.** A threshold comparison ``f[t+1] > c`` flips for every ``c``
between the low run's value and the high run's: body ``[0.4 %, 10 %)``, each wick
``[0.05 %, 1 %)``, range ``[0.5 %, 50 %)`` (log units), fill ``[0.2, 0.8)``, close location
``[0.25, 0.75)``. For a share the band depends on the window: ``(min s / 1.5, 1 - (1 - max s) / 1.5)``.
Thresholds outside the bands are not guaranteed to flip.

**Conflicts.** Validity first, then the crossing of levels: every bar stays valid
(``low <= open, close <= high``, ``low > 0``, taker ``<=`` total, ``trades`` an integer of at
least 1 when there is volume) and every price, volume and count crosses. What cannot cross, and
is therefore a **declared** ``?`` of :meth:`olsprime.bench.leaks.Probe.unknown_reason`, read on
the original bars before perturbing:

- a reference at a bound of its quantity: a share, a fill or a close location at 0 or 1; a zero
  body, wick, range, volume or count; an undefined one (zero volume, zero trades, a flat bar);
- a reference fill below ``FILL_FLOOR``, or a reference close location below ``CLV_FLOOR``
  (even cuts) or above ``1 - CLV_FLOOR`` (odd cuts);
- at odd cuts, bar t's upper wick at or above :func:`mirrored_upper_wick_cap`;
- a count of 1 at t + 1 (it cannot go lower: it moves up by one instead);
- a bar t + 1 without volume, whose zeros stay.

The taker VWAP gives way to the taker quote share's bound if the two conflict (the quote share
is capped below 1). No declared reason covers that case, so if it ever happened the taker-VWAP
control would fall off its pattern and H1 would fail, loudly.

**Every later bar after t + 1: independent factors per column.**

- **Prices.** A positive random walk ``W`` is sampled at every open and every close (two
  log-normal steps per bar), times one common factor ``level_shift``; ``open * W(open_s)`` and
  ``close * W(close_s)`` change the body and its colour. The wicks are redrawn around the new
  body, ``(wick + 0.001) * exp(N(0, wick_sigma))`` above and below, so the range moves.
- **Level.** The common factor (3 or 1/3; the H1 run uses both at every cut) moves any
  statistic of the whole sample, such as a full-sample mean or a scaler fitted on every row.
- **Volumes and trades.** ``volume`` and ``trades`` get independent factors (the level times a
  log-normal draw; ``trades`` rounded, moved by one if the rounding gave it back, at least 1
  when there is volume). The VWAP is redrawn log-uniformly in the bar's new ``[low, high]`` and
  ``quote_volume = volume * VWAP``. The taker base share is redrawn uniformly in (0, 1) and the
  taker VWAP log-uniformly in ``[low, high]``; the taker quote share follows, kept below 1.

**What stays.** ``open_time``, ``close_time``, ``known_at`` and ``symbol``: the grid, and with
it every null pattern, must stay the same, so that each row of the original pipeline has its
counterpart in the perturbed one. For the same reason a bar of zero volume keeps its zero
volumes (``ln 0`` nulls a feature) and, if it had none, its zero trades and taker volumes:
those exact zeros are the only values after the cut that may stay.

A column the perturbation does not know raises: a new input cannot escape it silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl

PRICE_COLUMNS = ("open", "high", "low", "close")
TOTAL_VOLUME_COLUMNS = ("volume", "quote_volume")
TAKER_VOLUME_COLUMNS = ("taker_buy_base_volume", "taker_buy_quote_volume")
COUNT_COLUMNS = ("trades",)
# The grid and the key: never perturbed, so that every row keeps its counterpart and its nulls.
UNCHANGED_COLUMNS = ("symbol", "open_time", "close_time", "known_at")
PERTURBED_COLUMNS = (*PRICE_COLUMNS, *TOTAL_VOLUME_COLUMNS, *TAKER_VOLUME_COLUMNS, *COUNT_COLUMNS)
_TAKER_TOTAL = {"taker_buy_base_volume": "volume", "taker_buy_quote_volume": "quote_volume"}

WINDOW = 90  # bars before t compared with bar t + 1: the longest feature window
PUSH = 1.5  # bar t + 1 beats the window's extreme by this factor
# The shape of bar t + 1 (log units for body and wicks): small, green and body-dominant in the up run;
# big, red and wick-dominant in the down run, upper wick at least 3 x the lower.
UP_CAP = {"body": 0.004, "wick": 0.0005}  # up run: body <= 0.4 %, each wick <= 0.05 %, range <= 0.5 %
DOWN_FLOOR = {"body": 0.10, "wick": 0.01}  # down run: body >= 10 %, each wick >= 1 %, range >= 50 %
UP_FILL, DOWN_FILL = 0.8, 0.2  # fill = body / range (log units): >= 0.8 up, <= 0.2 down
UP_CLV, DOWN_CLV = 0.75, 0.25  # close location (close - low) / (high - low): >= 0.75 up, <= 0.25 down
UPPER_OVER_LOWER = 3.0  # down run: upper wick >= 3 x lower wick
# Fill and close location are pushed past bar t alone (a near-doji in the window would otherwise need
# an unbounded range); the down run's fill never goes below FILL_FLOOR, so a bar t with a smaller
# fill cannot be undercut: a declared ``?`` for the fill peeker.
FILL_FLOOR = 0.02
# The close location of the big bar never goes past CLV_FLOOR (even cuts) or 1 - CLV_FLOOR (odd cuts):
# that bounds its wicks (a few log units). A reference past it is a declared ``?``.
CLV_FLOOR = 0.02
# Odd cuts, big green bar: its close location tends to exp(-upper wick), so the upper wick is capped
# at UW_CAP_SHARE * -ln(target close location); the lower wick does the rest.
UW_CAP_SHARE = 0.8
_EDGE = 1e-9  # aim just inside the shape bands, so that rounding never lands on the edge
SMALL_BODY = 0.01  # the size boundary of the colour x size cells (log body)
# The (colour, size) cell each run gives bar t + 1: even cuts, then the mirror image at odd cuts.
CELLS = {
    False: {"x3": ("green", "small"), "/3": ("red", "big")},
    True: {"x3": ("red", "small"), "/3": ("green", "big")},
}


@dataclass(frozen=True)
class Perturbation:
    """Scales of the random walk, of the wicks and of the volume factors (log-normal sigmas), and the level shift."""

    price_sigma: float = 0.05
    volume_sigma: float = 0.3
    wick_sigma: float = 0.5
    level_shift: float = 3.0
    mirror: bool = False  # odd cuts of the bench: bar t + 1 takes the mirror shape (red small up, green big down)


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _up(window: np.ndarray, original: float, mag: float, floor: float = 0.0) -> float:
    """At least ``PUSH`` times the window's maximum, ``mag`` times the original, and the floor."""
    w = _finite(window)
    out = max(mag * original if math.isfinite(original) else 0.0, floor)
    return max(out, PUSH * float(w.max())) if w.size else out


def _down(window: np.ndarray, original: float, mag: float, cap: float = math.inf) -> float:
    """At most the window's smallest positive value over ``PUSH``, the original over ``mag``, and the cap.

    Zeros are left out: a positive quantity (a price, a volume, a VWAP) must stay positive, and a
    comparison with a zero reference is a declared ``?``.
    """
    w = _finite(window)
    w = w[w > 0]
    out = min(original / mag if math.isfinite(original) and original > 0 else cap, cap)
    return min(out, float(w.min()) / PUSH) if w.size else out


def _interior(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = _finite(x)
    return x[(x > lo) & (x < hi)]


def _bounded_up(window: np.ndarray, original: float, mag: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """A value in ``(lo, hi)`` above every interior value of the window: the gap to ``hi`` shrinks by ``PUSH``.

    Window values at a bound cannot be beaten; a comparison with one is a declared ``?``
    (:meth:`olsprime.bench.leaks.Probe.unknown_reason`). Never equal to the original.
    """
    inner = _interior(window, lo, hi)
    gaps = [(hi - float(inner.max())) / PUSH] if inner.size else []
    if math.isfinite(original) and original < hi:
        gaps.append((hi - max(original, lo)) / mag)
    return hi - (min(gaps) if gaps else (hi - lo) / 4.0)


def _bounded_down(window: np.ndarray, original: float, mag: float, lo: float = 0.0, hi: float = 1.0) -> float:
    inner = _interior(window, lo, hi)
    gaps = [(float(inner.min()) - lo) / PUSH] if inner.size else []
    if math.isfinite(original) and original > lo:
        gaps.append((min(original, hi) - lo) / mag)
    return lo + (min(gaps) if gaps else (hi - lo) / 4.0)


def _small(window: np.ndarray, original: float, mag: float, cap: float) -> float:
    """Positive, at most ``cap``, below every positive value of the window by ``PUSH`` and the original by ``mag``."""
    out = cap
    pos = _interior(window, 0.0, math.inf)
    if pos.size:
        out = min(out, float(pos.min()) / PUSH)
    if math.isfinite(original) and original > 0:
        out = min(out, original / mag)
    return out


def _shape(o, h, lo, c):  # noqa: ANN001, ANN202
    """Fill (body / range, log units) and close location ((close - low) / (high - low)); NaN on a flat bar."""
    body, uw, lw = _candle(o, h, lo, c)
    rng = body + uw + lw
    with np.errstate(divide="ignore", invalid="ignore"):
        fill = np.where(rng > 0, body / rng, np.nan)
        clv = np.where(np.asarray(h) > np.asarray(lo), (np.asarray(c) - lo) / (np.asarray(h) - lo), np.nan)
    return fill, clv


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def _candle(o, h, lo, c):  # noqa: ANN001, ANN202
    """Body, upper wick and lower wick, in log units."""
    with np.errstate(divide="ignore", invalid="ignore"):
        top, bot = np.maximum(o, c), np.minimum(o, c)
        return np.abs(np.log(c / o)), np.log(h / top), np.log(bot / lo)


def _targets_clv(wc: np.ndarray, c0: float, high_side: bool) -> float:
    """The close location the bar must reach: past bar t's, inside the band, never past the floors."""
    if high_side:
        return max(UP_CLV + _EDGE, min(1.0 - CLV_FLOOR, _bounded_up(wc, c0, PUSH)))
    return min(DOWN_CLV - _EDGE, max(CLV_FLOOR, _bounded_down(wc, c0, PUSH)))


def mirrored_upper_wick_cap(bars: pl.DataFrame, cut: int) -> float:
    """Odd cuts, /3 run: the largest upper wick of the big green bar, from the original bars t and t + 1.

    Its close location must reach the target ``C`` (at least 0.75 and past bar t's), and a close
    location tends to ``exp(-upper wick)`` however long the lower wick: so the upper wick is
    capped at ``UW_CAP_SHARE * -ln C``. Read on the original bars only, so the bench can declare
    an upper-wick comparison ``?`` before perturbing when bar t's upper wick is above the cap.
    """
    o, h, lo, c = (bars[k].cast(pl.Float64).to_numpy() for k in PRICE_COLUMNS)
    _, wc = _shape(o[cut : cut + 1], h[cut : cut + 1], lo[cut : cut + 1], c[cut : cut + 1])
    _, c0 = _shape(o[cut + 1], h[cut + 1], lo[cut + 1], c[cut + 1])
    return -UW_CAP_SHARE * math.log(_targets_clv(wc, float(c0), high_side=True))


def first_bar(
    bars: pl.DataFrame, cut: int, level: float, rng: np.random.Generator | None = None, *, mirror: bool = False
) -> dict[str, float]:
    """The values of bar ``cut + 1`` in the run of this level: pushed past the window (module docstring).

    ``mirror`` (odd cuts of the bench) swaps the colours: the up bar is small and red, closing
    near its low; the down bar is big and green, lower-wick-dominant, closing near its high.
    """
    up = level > 1.0
    mag = level if up else 1.0 / level
    w = slice(max(0, cut - WINDOW), cut + 1)
    col = {c: bars[c].cast(pl.Float64).to_numpy() for c in PERTURBED_COLUMNS}
    win = {c: v[w] for c, v in col.items()}
    now = {c: float(v[cut + 1]) for c, v in col.items()}

    rng = rng if rng is not None else np.random.default_rng(cut)
    wb, wu, wl = _candle(win["open"], win["high"], win["low"], win["close"])
    b0, u0, l0 = (float(x) for x in _candle(now["open"], now["high"], now["low"], now["close"]))
    wf, wc = _shape(*(win[k][-1:] for k in ("open", "high", "low", "close")))  # bar t alone
    f0, c0 = (float(x) for x in _shape(now["open"], now["high"], now["low"], now["close"]))
    wr = wb + wu + wl
    r0 = b0 + u0 + l0
    out: dict[str, float] = {}
    if up:  # small and body-dominant: green closing near its high, or (mirror) red closing near its low
        fill = max(UP_FILL + _EDGE, _bounded_up(wf, f0, PUSH))
        range_cap = _small(wr, r0, mag, UP_CAP["body"] + 2 * UP_CAP["wick"])
        body = min(_small(wb, b0, mag, UP_CAP["body"]), UP_FILL * range_cap) * rng.uniform(0.5, 1.0)
        budget = body * (1.0 - fill) / fill  # upper + lower wick: range <= body / fill <= range_cap
        split = rng.uniform(0.1, 0.9)  # the free part of the shape, drawn per run
        clv = _targets_clv(wc, c0, high_side=not mirror)
        if not mirror:
            lw = min(_small(wl, l0, mag, UP_CAP["wick"]), budget * (1.0 - split))
            uw_clv = math.log(math.expm1(lw + body) / clv + 1.0) - lw - body  # the largest upper wick keeping clv
            uw = min(_small(wu, u0, mag, UP_CAP["wick"]), budget * split, uw_clv)
        else:
            uw = min(_small(wu, u0, mag, UP_CAP["wick"]), budget * split)
            lw_clv = math.log((1.0 - clv) / (1.0 - clv * math.exp(body + uw)))  # the largest lower wick keeping clv
            lw = min(_small(wl, l0, mag, UP_CAP["wick"]), budget * (1.0 - split), lw_clv)
        low = max(PUSH * float(win["high"].max()), mag * now["high"])
        out["low"] = low
        if not mirror:  # green
            out["open"] = low * math.exp(lw)
            out["close"] = out["open"] * math.exp(body)
            out["high"] = out["close"] * math.exp(uw)
        else:  # red
            out["close"] = low * math.exp(lw)
            out["open"] = out["close"] * math.exp(body)
            out["high"] = out["open"] * math.exp(uw)
    else:  # big and wick-dominant: red closing near its low, or (mirror) green closing near its high
        fill = min(DOWN_FILL - _EDGE, max(FILL_FLOOR, _bounded_down(wf, f0, PUSH)))
        body = _up(wb, b0, mag, DOWN_FLOOR["body"]) * rng.uniform(1.0, 1.5)
        clv = _targets_clv(wc, c0, high_side=mirror)
        if not mirror:  # upper wick >= 3 x lower
            lw = _up(wl, l0, mag, DOWN_FLOOR["wick"]) * rng.uniform(1.0, 1.5)
            uw_clv = math.log(math.expm1(lw) / clv + 1.0) - lw - body  # the smallest upper wick giving clv
            other = max(_up(wu, u0, mag, DOWN_FLOOR["wick"]), UPPER_OVER_LOWER * lw, uw_clv) * rng.uniform(1.0, 1.5)
            uw = max(other, body * (1.0 - fill) / fill - lw)  # the fill need is exact: no random factor
        else:  # lower wick >= 3 x upper, upper wick capped by the close location
            cap = -UW_CAP_SHARE * math.log(clv)
            uw = min(_up(wu, u0, mag, DOWN_FLOOR["wick"]) * rng.uniform(1.0, 1.5), cap)
            lw_clv = math.log((1.0 - clv) / (1.0 - clv * math.exp(uw))) - body  # the smallest lower wick giving clv
            range_need = _up(wr, r0, mag) - body - uw
            other = max(_up(wl, l0, mag, DOWN_FLOOR["wick"]), UPPER_OVER_LOWER * uw, range_need, lw_clv)
            lw = max(other * rng.uniform(1.0, 1.5), body * (1.0 - fill) / fill - uw)
        body = max(body, FILL_FLOOR * (uw + lw) / (1.0 - FILL_FLOOR) * (1.0 + _EDGE))  # the fill floor, exact
        high = min(float(win["low"].min()) / PUSH, now["low"] / mag)
        out["high"] = high
        if not mirror:  # red
            out["open"] = high * math.exp(-uw)
            out["close"] = out["open"] * math.exp(-body)
            out["low"] = out["close"] * math.exp(-lw)
        else:  # green
            out["close"] = high * math.exp(-uw)
            out["open"] = out["close"] * math.exp(-body)
            out["low"] = out["open"] * math.exp(-lw)

    v0, n0 = now["volume"], now["trades"]
    if v0 == 0.0 and n0 == 0.0:
        out["trades"] = 0.0
    elif up:
        out["trades"] = float(max(math.ceil(PUSH * float(win["trades"].max())), math.ceil(mag * n0), 1))
    else:
        n = max(math.floor(_down(win["trades"], n0, mag)), 1)
        out["trades"] = float(n if n != n0 else n0 + 1)  # a single trade cannot go lower: it moves up
    if v0 == 0.0:  # a bar without volume keeps its zeros
        for c in (*TOTAL_VOLUME_COLUMNS, *TAKER_VOLUME_COLUMNS):
            out[c] = 0.0
        return out

    vw_w = _ratio(win["quote_volume"], win["volume"])
    vw0 = now["quote_volume"] / v0
    mid = math.sqrt(out["low"] * out["high"])
    vwap = max(mid, _up(vw_w, vw0, mag)) if up else min(mid, _down(vw_w, vw0, mag))
    n = out["trades"]
    size_w = _ratio(win["volume"], win["trades"])
    size0 = v0 / n0 if n0 > 0 else math.nan
    pos_v = win["volume"][win["volume"] > 0]
    if up:
        v = max(_up(pos_v, v0, mag), _up(size_w, size0, mag) * n)
    else:
        v = min(_down(pos_v, v0, mag), _down(size_w, size0, mag) * n)
    out["volume"] = v
    out["quote_volume"] = v * vwap

    sb_w, sq_w = _ratio(win["taker_buy_base_volume"], win["volume"]), _ratio(win["taker_buy_quote_volume"], win["quote_volume"])
    tv_w = _ratio(win["taker_buy_quote_volume"], win["taker_buy_base_volume"])
    q0, tb0 = now["quote_volume"], now["taker_buy_base_volume"]
    sb0 = min(max(tb0 / v0, 0.0), 1.0)
    sq0 = min(max(now["taker_buy_quote_volume"] / q0, 0.0), 1.0) if q0 > 0 else sb0
    tv0 = now["taker_buy_quote_volume"] / tb0 if tb0 > 0 else math.nan
    if up:
        sb = _bounded_up(sb_w, sb0, mag)
        sq_req = _bounded_up(sq_w, sq0, mag)
        cap = 1.0 - (1.0 - sq_req) / 2.0 if sq_req < 1.0 else sq_req
        sq = min(max(sq_req, sb * _up(tv_w, tv0, mag) / vwap), cap)  # the taker VWAP gives way to the bound
    else:
        sb = _bounded_down(sb_w, sb0, mag)
        sq_req = _bounded_down(sq_w, sq0, mag)
        tv = _down(tv_w, tv0, mag)
        sq = min(sq_req, sb * tv / vwap) if math.isfinite(tv) else sq_req
    out["taker_buy_base_volume"] = sb * v
    out["taker_buy_quote_volume"] = sq * out["quote_volume"]
    return out


def perturb_after(
    bars: pl.DataFrame,
    cut: int,
    *,
    seed: int,
    config: Perturbation = Perturbation(),
) -> pl.DataFrame:
    """``bars`` (one symbol, time order) with every value of every row after position ``cut`` changed.

    ``cut`` is a row position, ``0 <= cut < height - 1`` (at least one bar must follow it).
    Rows ``0 .. cut`` are returned bit for bit. See the module docstring; the coverage of values
    and ratios is checked before returning (:func:`check_coverage`).
    """
    if bars["symbol"].n_unique() != 1:
        raise ValueError("perturb one symbol at a time")
    if not bars["open_time"].is_sorted() or bars["open_time"].n_unique() != bars.height:
        raise ValueError("bars must be strictly increasing in open_time")
    unknown = [c for c in bars.columns if c not in UNCHANGED_COLUMNS and c not in PERTURBED_COLUMNS]
    if unknown:
        raise ValueError(f"the perturbation does not know columns {unknown}: decide how they move first")
    missing = [c for c in PERTURBED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"the perturbation needs every bar column; missing {missing}")
    n = bars.height
    if not 0 <= cut < n - 1:
        raise ValueError(f"cut must leave at least one later bar: 0 <= cut < {n - 1}, got {cut}")
    level = float(config.level_shift)
    if not (level > 0 and level != 1.0):
        raise ValueError(f"level_shift must be positive and not 1, got {level}")
    rng = np.random.default_rng(seed)
    u = cut + 1
    rest = slice(cut + 2, n)  # the bars after t + 1
    m = n - cut - 2
    col = {c: bars[c].cast(pl.Float64).to_numpy().copy() for c in PERTURBED_COLUMNS}

    rng1 = np.random.default_rng([seed, 1])
    if config.mirror:  # bar t + 1: designed against the window (the mirror shape at odd cuts)
        first = first_bar(bars, cut, level, rng1, mirror=True)
    else:
        first = first_bar(bars, cut, level, rng1)

    # Later bars: the walk at opens and closes, new wicks, independent volume factors.
    o, h, lo, c = (col[k][rest].copy() for k in PRICE_COLUMNS)
    walk = level * np.exp(np.cumsum(rng.normal(0.0, config.price_sigma, size=2 * m)))
    o2, c2 = o * walk[0::2], c * walk[1::2]
    _, uw, lw = _candle(o, h, lo, c)
    uw2 = (np.nan_to_num(uw, nan=0.0) + 1e-3) * np.exp(rng.normal(0.0, config.wick_sigma, size=m))
    lw2 = (np.nan_to_num(lw, nan=0.0) + 1e-3) * np.exp(rng.normal(0.0, config.wick_sigma, size=m))
    h2 = np.maximum(o2, c2) * np.exp(uw2)
    lo2 = np.minimum(o2, c2) * np.exp(-lw2)
    vol = col["volume"][rest]
    v2 = vol * level * np.exp(rng.normal(0.0, config.volume_sigma, size=m))
    vwap = lo2 * (h2 / lo2) ** rng.uniform(0.0, 1.0, size=m)
    q2 = v2 * vwap
    sb = rng.uniform(0.0, 1.0, size=m)
    sb = np.where(sb > 0.0, sb, 0.5)
    tvwap = lo2 * (h2 / lo2) ** rng.uniform(0.0, 1.0, size=m)
    refill = rng.uniform(0.5, 1.0, size=m)
    sb = np.where(sb * tvwap / vwap >= 1.0, refill * vwap / tvwap, sb)  # keep the taker quote below the total
    sq = sb * tvwap / vwap
    tb2, tq2 = sb * v2, sq * q2
    trades = col["trades"][rest]
    t2 = np.rint(trades * level * np.exp(rng.normal(0.0, config.volume_sigma, size=m)))
    t2 = np.where(vol > 0, np.maximum(t2, 1.0), t2)
    t2 = np.where(t2 == trades, trades + np.where((trades > 1) & (level < 1.0), -1.0, 1.0), t2)
    t2 = np.where((vol == 0) & (trades == 0), 0.0, t2)  # a bar without volume or trades keeps its zeros

    new = {"open": o2, "high": h2, "low": lo2, "close": c2, "volume": v2, "quote_volume": q2,
           "taker_buy_base_volume": tb2, "taker_buy_quote_volume": tq2, "trades": t2}  # fmt: skip
    columns = []
    for k, v in col.items():
        v[rest] = new[k]
        v[u] = first[k]
        if k == "trades":
            columns.append(pl.Series(k, np.rint(v).astype(np.int64), dtype=bars.schema[k]))
        else:
            columns.append(pl.Series(k, v, dtype=bars.schema[k]))
    out = bars.with_columns(columns)
    _check_valid(out, cut)
    check_coverage(bars, out, cut)
    return out


def check_coverage(original: pl.DataFrame, perturbed: pl.DataFrame, cut: int) -> None:
    """Assert that rows ``<= cut`` are identical, and that every perturbed value and ratio after the cut differs.

    The only values after the cut allowed to stay are exact zeros of the volume, taker and
    ``trades`` columns on a bar of zero volume (see the module docstring). For every later bar
    and every pair ``(a, b)`` of perturbed columns that are nonzero in the original and in the
    perturbed bar, ``a / b`` must change. The grid columns must be identical everywhere.
    """
    if original.columns != perturbed.columns or original.height != perturbed.height:
        raise AssertionError("the perturbation changed the shape of the bars")
    for c in UNCHANGED_COLUMNS:
        if c in original.columns and not original[c].equals(perturbed[c]):
            raise AssertionError(f"the perturbation moved {c!r}, which must stay")
    later = slice(cut + 1, original.height)
    zero_volume = np.zeros(original.height - cut - 1, dtype=bool)
    for name in TOTAL_VOLUME_COLUMNS:
        if name in original.columns:
            zero_volume |= original[name].to_numpy()[later] == 0
    for c in original.columns:
        if c in UNCHANGED_COLUMNS:
            continue
        if not original[c][: cut + 1].equals(perturbed[c][: cut + 1]):
            raise AssertionError(f"the perturbation moved {c!r} at or before the cut")
        a, b = original[c].to_numpy()[later], perturbed[c].to_numpy()[later]
        stayed = a == b
        allowed = stayed & zero_volume & (a == 0) if c not in PRICE_COLUMNS else np.zeros_like(stayed)
        if np.any(stayed & ~allowed):
            first = int(np.flatnonzero(stayed & ~allowed)[0]) + cut + 1
            raise AssertionError(f"the perturbation left {c!r} unchanged at row {first}, after the cut")
    present = [c for c in PERTURBED_COLUMNS if c in original.columns]
    a_ = {c: original[c].cast(pl.Float64).to_numpy()[later] for c in present}
    b_ = {c: perturbed[c].cast(pl.Float64).to_numpy()[later] for c in present}
    for x, y in combinations(present, 2):
        both = (a_[x] != 0) & (a_[y] != 0) & (b_[x] != 0) & (b_[y] != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            stayed = both & (a_[x] / a_[y] == b_[x] / b_[y])
        if np.any(stayed):
            first = int(np.flatnonzero(stayed)[0]) + cut + 1
            raise AssertionError(f"the perturbation left the ratio {x}/{y} unchanged at row {first}, after the cut")


def _check_valid(bars: pl.DataFrame, cut: int) -> None:
    """The perturbed bars (after the cut) are valid; the stored history is not judged here."""
    bars = bars[cut + 1 :]
    o, h, lo, c = (bars[k].to_numpy() for k in PRICE_COLUMNS)
    if not (np.all(lo <= np.minimum(o, c)) and np.all(h >= np.maximum(o, c)) and np.all(lo > 0)):
        raise AssertionError("the perturbation produced an invalid bar")
    for name, total in _TAKER_TOTAL.items():
        if name in bars.columns and total in bars.columns:
            if np.any(bars[name].to_numpy() > bars[total].to_numpy() * (1 + 1e-12)):
                raise AssertionError(f"the perturbation produced {name} > {total}")
    if np.any((bars["volume"].to_numpy() > 0) & (bars["trades"].to_numpy() < 1)):
        raise AssertionError("the perturbation produced a bar with volume and no trade")
