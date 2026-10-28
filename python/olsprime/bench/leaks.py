"""!!! LEAKY MODELS, ON PURPOSE. NOT MODELS. !!!

The planted leaks of the H1 run (:mod:`olsprime.bench.invariance`), ported from Audit A
(2026-10-28). Each one reads the future through a different column or a different mistake, so
that the future-invariance test must show it can see that column or that mistake. They are
permanent controls, beside :class:`olsprime.models.baselines.Cheater`:

- :class:`ProbePeeker`, one per :data:`PROBES`: a function of one or two values of bar t + 1
  (a price, a volume, a ratio inside the bar: range, body, VWAP, taker VWAP, trade size, taker
  share), compared with bar t, with bar t - 5 or with a threshold, or returned as ``exp_return``.
  Flagged at every cut, except at the cuts where a documented reason, read on the original bars
  before perturbing (:meth:`Probe.unknown_reason`), says the comparison cannot cross: those are
  ``?``, reported with their reason. Any other failure to cross puts the control off its pattern.
- :class:`OffByOneReturn`: ``exp_return = ln(close[t+1] / close[t])``, a feature shifted the
  wrong way. Flagged at every cut.
- :class:`TransductiveScaler`: ridge on features standardized with the training rows **and the
  whole test month**. Flagged at every cut whose month goes on after it, and never at the last
  bar of a month, where the month's features are all in the past. Its ridge centres the features
  but does not rescale them, so the test month's spread reaches the penalty.

Never report one as a result, never copy one into a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from olsprime.bench.perturb import CELLS, CLV_FLOOR, FILL_FLOOR, SMALL_BODY, mirrored_upper_wick_cap
from olsprime.models.baselines import Ridge, check_test_frame

TAKER_SHARE = "taker_share"


def _output(test: pl.DataFrame, *, exp_return: np.ndarray | None = None, target: np.ndarray | None = None) -> pl.DataFrame:
    n = test.height
    nulls = pl.Series([None] * n, dtype=pl.Float64)
    return pl.DataFrame(
        {
            "symbol": test["symbol"],
            "decision_time": test["decision_time"],
            "p_down": nulls,
            "p_neutral": nulls,
            "p_up": nulls,
            "exp_return": nulls if exp_return is None else pl.Series(exp_return, dtype=pl.Float64, nan_to_null=True),
            "target": nulls if target is None else pl.Series(target, dtype=pl.Float64),
        }
    )


@dataclass(frozen=True)
class LeakConfig:
    leaks: str


QUANTITIES: dict[str, pl.Expr] = {
    "close": pl.col("close").cast(pl.Float64),
    "trades": pl.col("trades").cast(pl.Float64),
    "quote_volume": pl.col("quote_volume").cast(pl.Float64),
    TAKER_SHARE: pl.col("taker_buy_quote_volume") / pl.col("quote_volume"),
    "taker_base_share": pl.col("taker_buy_base_volume") / pl.col("volume"),
    "range": (pl.col("high") / pl.col("low")).log(),
    "body": (pl.col("close") / pl.col("open")).log().abs(),
    "vwap": pl.col("quote_volume") / pl.col("volume"),
    "taker_vwap": pl.col("taker_buy_quote_volume") / pl.col("taker_buy_base_volume"),
    "trade_size": pl.col("volume") / pl.col("trades"),
    "fill": (pl.col("close") / pl.col("open")).log().abs() / (pl.col("high") / pl.col("low")).log(),
    "close_location": (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")),
    "upper_wick": (pl.col("high") / pl.max_horizontal("open", "close")).log(),
    "lower_wick": (pl.min_horizontal("open", "close") / pl.col("low")).log(),
    # Colour x size cells of a bar (1.0 or 0.0): a candle pattern read one bar early.
    "red_small": ((pl.col("close") < pl.col("open")) & ((pl.col("close") / pl.col("open")).log().abs() < SMALL_BODY)),
    "green_big": ((pl.col("close") > pl.col("open")) & ((pl.col("close") / pl.col("open")).log().abs() > SMALL_BODY)),
}
# Quantities that need volume: a bar without volume keeps its zeros, so they cannot cross there.
VOLUME_BASED = frozenset({"trades", "quote_volume", TAKER_SHARE, "taker_base_share", "vwap", "taker_vwap", "trade_size"})
# The open interval a quantity lives in: a reference at a bound cannot be crossed from both sides.
BOUNDS: dict[str, tuple[float, float]] = {
    TAKER_SHARE: (0.0, 1.0),
    "taker_base_share": (0.0, 1.0),
    "fill": (0.0, 1.0),
    "close_location": (0.0, 1.0),
}
NEXT_VS_REF = "next-vs-ref"  # long iff q[t+1] > q[t - back]
THRESHOLD = "threshold"  # long iff q[t+1] > threshold
CONTINUOUS = "continuous"  # exp_return = q[t+1]
VS_CLOSE = "continuous-vs-close"  # exp_return = q[t+1] / close[t] - 1


@dataclass(frozen=True)
class Probe:
    """!!! A LEAK, ON PURPOSE: what a peeker reads of bar t + 1, and how it decides. !!!"""

    name: str
    quantity: str
    mode: str = NEXT_VS_REF
    back: int = 0
    threshold: float = 0.0
    cell: tuple[str, str] | None = None  # a colour x size cell: its pattern follows the cells the runs realise

    def predicted_flag(self, original: pl.DataFrame, cut: int, mirror: bool) -> bool | None:
        """For a cell probe: flagged iff a run gives bar t + 1 a cell that differs from the original's on this cell.

        Computed from the cells the design gives each run (:data:`olsprime.bench.perturb.CELLS`)
        and the original bars only. ``None`` for the other probes, whose prediction is fixed.
        """
        if self.cell is None:
            return None
        before = bool(self.values(original)[cut + 1] > 0.5)
        return any((cell == self.cell) != before for cell in CELLS[mirror].values())

    def values(self, bars: pl.DataFrame) -> np.ndarray:
        """The quantity on every row (time order); NaN where it is undefined (a zero denominator)."""
        q = bars.sort("open_time").select(QUANTITIES[self.quantity].cast(pl.Float64).alias("q"))["q"]
        q = q.to_numpy().astype(np.float64)
        return np.where(np.isfinite(q), q, np.nan)

    def decide(self, bars: pl.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """``(target, exp_return)`` for every row t, from row t + 1 (and row t - back)."""
        q = self.values(bars)
        nxt = np.append(q[1:], np.nan)
        with np.errstate(invalid="ignore"):
            if self.mode == NEXT_VS_REF:
                ref = np.concatenate([np.full(self.back, np.nan), q[: len(q) - self.back]]) if self.back else q
                return np.where(nxt > ref, 1.0, 0.0), None
            if self.mode == THRESHOLD:
                return np.where(nxt > self.threshold, 1.0, 0.0), None
            if self.mode == CONTINUOUS:
                return None, nxt
            if self.mode == VS_CLOSE:
                close = bars.sort("open_time")["close"].cast(pl.Float64).to_numpy()
                return None, nxt / close - 1.0
        raise ValueError(f"unknown probe mode {self.mode!r}")

    def unknown_reason(self, original: pl.DataFrame, cut: int, mirror: bool = False) -> str | None:
        """Why this probe cannot cross at ``cut``, predicted from the **original** bars; None when it must.

        Only the documented reasons of :mod:`olsprime.bench.perturb` are named here: a bar
        without volume (its zeros stay), a reference undefined or at a bound of its quantity (a
        share or a shape at 0 or 1, a zero body, wick, range or count), a count of 1 at t + 1 that
        cannot go lower, and a reference before the first bar. Any other failure to cross is a
        failure of the perturbation: the cut stays predicted, and the control falls off its pattern.
        """
        b = original.sort("open_time")
        if cut + 1 >= b.height:
            return "no bar after the cut"
        volume = b["volume"].cast(pl.Float64).to_numpy()
        if self.quantity in VOLUME_BASED and volume[cut + 1] == 0:
            return "zero volume at t+1"
        if self.mode != NEXT_VS_REF:
            return None
        if cut - self.back < 0:
            return "reference before the first bar"
        ref = self.values(b)[cut - self.back]
        if not np.isfinite(ref):
            return "reference undefined (zero volume, zero trades or a flat bar)"
        lo, hi = BOUNDS.get(self.quantity, (0.0, np.inf))
        if self.quantity != "close" and not lo < ref < hi:
            return f"reference at a bound of {self.quantity} ({ref:g})"
        if self.quantity == "fill" and ref <= FILL_FLOOR:
            return f"a reference fill at or below the floor {FILL_FLOOR:g} of the /3 run ({ref:g})"
        # At every cut one run aims below bar t's close location and the other above it, whatever the parity,
        # and neither goes past the floor or the ceiling: a reference past either cannot be crossed.
        if self.quantity == "close_location" and ref <= CLV_FLOOR:
            return f"a reference close location at or below the floor {CLV_FLOOR:g} of the runs ({ref:g})"
        if self.quantity == "close_location" and ref >= 1.0 - CLV_FLOOR:
            return f"a reference close location at or above the ceiling {1.0 - CLV_FLOOR:g} of the runs ({ref:g})"
        if self.quantity == "upper_wick" and mirror and self.back == 0:
            cap = mirrored_upper_wick_cap(b, cut)
            if ref >= cap:
                return f"bar t's upper wick {ref:.4g} at or above the mirrored /3 run's cap {cap:.4g}"
        if self.quantity == "trades" and float(b["trades"][cut + 1]) == 1.0:
            return "a count of 1 at t+1 that must go down"
        return None


PROBES: tuple[Probe, ...] = (
    Probe("trades-peeker", "trades"),
    Probe("quote-volume-peeker", "quote_volume"),
    Probe("taker-share-peeker", TAKER_SHARE),
    Probe("taker-base-share-peeker", "taker_base_share"),
    Probe("range-peeker", "range"),
    Probe("body-peeker", "body"),
    Probe("vwap-peeker", "vwap"),
    Probe("taker-vwap-peeker", "taker_vwap"),
    Probe("trade-size-peeker", "trade_size"),
    Probe("range-above-3pct", "range", THRESHOLD, threshold=0.03),
    Probe("body-above-2pct", "body", THRESHOLD, threshold=0.02),
    Probe("range-next-continuous", "range", CONTINUOUS),
    Probe("vwap-next-continuous", "vwap", VS_CLOSE),
    Probe("close-vs-6-bars-back", "close", back=5),
    Probe("close-location-peeker", "close_location"),
    Probe("fill-peeker", "fill"),
    Probe("close-location-above-half", "close_location", THRESHOLD, threshold=0.5),
    Probe("upper-wick-peeker", "upper_wick"),
    Probe("lower-wick-peeker", "lower_wick"),
    Probe("red-and-small", "red_small", THRESHOLD, threshold=0.5, cell=("red", "small")),
    Probe("green-and-big", "green_big", THRESHOLD, threshold=0.5, cell=("green", "big")),
)


class ProbePeeker:
    """!!! LEAKS: decides at t from bar t + 1, read from the bars it is built with (see :class:`Probe`). !!!"""

    def __init__(self, bars: pl.DataFrame, probe: Probe) -> None:
        self.name = probe.name
        self.config = LeakConfig(leaks=f"reads {probe.quantity} of the next bar ({probe.mode})")
        target, exp_return = probe.decide(bars)
        times = bars.sort("open_time")["open_time"].to_list()
        self._target = None if target is None else dict(zip(times, target.tolist(), strict=True))
        self._exp = None if exp_return is None else dict(zip(times, exp_return.tolist(), strict=True))

    def fit(self, train: pl.DataFrame) -> None:
        pass

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        times = test["open_time"].to_list()
        target = None if self._target is None else np.array([self._target[t] for t in times])
        exp_return = None if self._exp is None else np.array([self._exp[t] for t in times])
        return _output(test, target=target, exp_return=exp_return)


class OffByOneReturn:
    """!!! LEAKS: ``exp_return = ln(close[t+1] / close[t])``, a return feature shifted the wrong way. !!!"""

    name = "off-by-one-return"

    def __init__(self, bars: pl.DataFrame) -> None:
        self.config = LeakConfig(leaks="exp_return is the next bar's close-to-close log return")
        b = bars.sort("open_time")
        r = b.select("open_time", (pl.col("close").shift(-1) / pl.col("close")).log().alias("_r"))
        self._r = dict(zip(r["open_time"].to_list(), r["_r"].to_list(), strict=True))

    def fit(self, train: pl.DataFrame) -> None:
        pass

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        r = np.array([np.nan if self._r[t] is None else self._r[t] for t in test["open_time"].to_list()])
        return _output(test, exp_return=r)


class _RidgeAtGivenScale(Ridge):
    """Ridge that centres its features on its training rows but keeps the scale it is given."""

    @staticmethod
    def _standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x.mean(axis=0), np.ones(x.shape[1])


class TransductiveScaler:
    """!!! LEAKS: ridge on features standardized on the training rows plus the whole test month. !!!

    The inner ridge centres the features on its training rows but does not rescale them, so the
    test month's spread reaches its penalty, as in any pipeline that scales before a penalized fit.
    The baseline :class:`Ridge` standardizes again on its training rows, which undoes this scaling
    exactly: the leak then showed only as rounding, which H1 does not count as a catch (iteration 2,
    caught at every cut on Windows and missed at one on Linux).
    """

    name = "transductive-scaler"

    def __init__(self) -> None:
        self.config = LeakConfig(leaks="feature means and stds include the fold's whole test month")
        self._train: pl.DataFrame | None = None

    def fit(self, train: pl.DataFrame) -> None:
        self._train = train

    def predict(self, test: pl.DataFrame) -> pl.DataFrame:
        check_test_frame(test)
        assert self._train is not None
        feats = [c for c in test.columns if c.startswith("f_")]
        both = pl.concat([self._train.select(feats), test.select(feats)])
        mean = both.select([pl.col(c).mean() for c in feats]).row(0)
        std = both.select([pl.col(c).std(ddof=0) for c in feats]).row(0)

        def scale(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                [(pl.col(c) - m) / (s if s and s > 0 else 1.0) for c, m, s in zip(feats, mean, std, strict=True)]
            )

        inner = _RidgeAtGivenScale()
        inner.fit(scale(self._train))
        return inner.predict(scale(test))
