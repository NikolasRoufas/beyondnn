"""Uncertainty statements for aggregate audit quantities (ADR-049;
docs/research/PHASE_7_5_LITERATURE.md §1).

* :func:`wilson`: a Wilson score interval for a proportion k of n (Brown, Cai &
  DasGupta 2001). Deterministic.
* :func:`bootstrap`: a percentile bootstrap interval of a statistic of per-sample values;
  the resampling unit is the input sample (Efron & Tibshirani; Koehn 2004).
* :func:`paired_bootstrap`: the same for the difference of two statistics on the same
  samples (the pairing is kept by resampling sample indices).

Every :class:`Interval` names its method, level, resampling unit, n, and (for the
bootstrap) seed and number of draws. An interval describes sampling variability over
inputs drawn like the declared samples. It is never a confidence that an explanation,
a claim, or a model is correct, and nothing here combines intervals into a score.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["STATISTICS", "Interval", "bootstrap", "paired_bootstrap", "wilson"]

STATISTICS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": statistics.fmean,
    "median": statistics.median,
}


@dataclass(frozen=True, slots=True)
class Interval:
    """An interval estimate with everything needed to interpret and reproduce it."""

    quantity: str
    estimate: float
    low: float
    high: float
    level: float
    method: str
    unit: str
    n: int
    k: int | None = None
    draws: int | None = None
    seed: int | None = None

    def describe(self) -> str:
        count = f"{self.k} of {self.n}; " if self.k is not None else f"n = {self.n}; "
        rng = f", {self.draws} draws, seed {self.seed}" if self.draws is not None else ""
        return (
            f"{self.quantity}: {self.estimate:.4g} ({count}{self.level * 100:g}% {self.method} "
            f"[{self.low:.4g}, {self.high:.4g}] over {self.unit}{rng})"
        )


def _z(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    return statistics.NormalDist().inv_cdf(0.5 + level / 2)


def wilson(k: int, n: int, *, quantity: str, unit: str, level: float = 0.95) -> Interval:
    """Wilson score interval for the proportion ``k / n``."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n and n > 0")
    z = _z(level)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(
        quantity=quantity,
        estimate=p,
        low=0.0 if k == 0 else max(0.0, centre - half),  # exact at the boundaries
        high=1.0 if k == n else min(1.0, centre + half),
        level=level,
        method="wilson",
        unit=unit,
        n=n,
        k=k,
    )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method)."""
    position = q * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def _check(draws: int, seed: int) -> None:
    if draws < 1000:
        raise ValueError("use at least 1000 bootstrap draws")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("the bootstrap seed must be an int (it is recorded)")


def bootstrap(
    values: Sequence[float],
    *,
    quantity: str,
    unit: str,
    statistic: str = "mean",
    draws: int = 10_000,
    seed: int,
    level: float = 0.95,
) -> Interval:
    """Percentile bootstrap of ``statistic`` over ``values`` (one value per sample)."""
    _check(draws, seed)
    values = [float(v) for v in values]
    if len(values) < 2:
        raise ValueError("the bootstrap needs at least 2 samples")
    fn = STATISTICS[statistic]
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    alpha = (1 - level) / 2
    return Interval(
        quantity=quantity,
        estimate=fn(values),
        low=_percentile(stats, alpha),
        high=_percentile(stats, 1 - alpha),
        level=level,
        method=f"percentile bootstrap of the {statistic}",
        unit=unit,
        n=n,
        draws=draws,
        seed=seed,
    )


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    quantity: str,
    unit: str,
    statistic: str = "mean",
    draws: int = 10_000,
    seed: int,
    level: float = 0.95,
) -> Interval:
    """Percentile bootstrap of ``statistic(a) - statistic(b)``, where ``a[i]`` and
    ``b[i]`` are measured on the same sample ``i`` (the pairing is kept)."""
    _check(draws, seed)
    if len(a) != len(b):
        raise ValueError("paired values must have the same length (one pair per sample)")
    if len(a) < 2:
        raise ValueError("the bootstrap needs at least 2 samples")
    fa = [float(v) for v in a]
    fb = [float(v) for v in b]
    fn = STATISTICS[statistic]
    rng = random.Random(seed)
    n = len(fa)
    stats = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(fn([fa[i] for i in idx]) - fn([fb[i] for i in idx]))
    stats.sort()
    alpha = (1 - level) / 2
    return Interval(
        quantity=quantity,
        estimate=fn(fa) - fn(fb),
        low=_percentile(stats, alpha),
        high=_percentile(stats, 1 - alpha),
        level=level,
        method=f"paired percentile bootstrap of the difference of {statistic}s",
        unit=unit,
        n=n,
        draws=draws,
        seed=seed,
    )
