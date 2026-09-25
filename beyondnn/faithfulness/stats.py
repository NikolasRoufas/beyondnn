"""Statistics for faithfulness controls (Phase 5; ADR-033). Pure Python, deterministic.

Nothing here decides a claim. These are descriptive comparisons with a stated null:

* control fractions: the share of matched random controls whose statistic is below,
  tied with, or above the observed one;
* ``mc_p_value``: (1 + #{control >= observed}) / (N + 1), a one-sided Monte-Carlo
  estimate of "a matched random set does at least as well" (the +1 form recommended
  by Phipson & Smyth 2010, "Permutation p-values should never be zero"); resolution
  1/(N+1); never called "significant";
* ``sign_flip_p``: a paired, one-sided sign-flip permutation test of mean(d) > 0 over
  per-sample paired differences d, with a seeded generator;
* rank agreement: Spearman correlation of two *ordinal* rankings (ties already broken
  by the declared rule) and Jaccard overlap of top-k sets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

__all__ = [
    "control_fractions",
    "jaccard",
    "mc_p_value",
    "quantile",
    "rank_order",
    "sign_flip_p",
    "spearman_of_orders",
    "uniform_permutations",
    "uniform_subsets",
]


def control_fractions(observed: float, controls: Sequence[float]) -> dict[str, float]:
    n = len(controls)
    if n == 0:
        raise ValueError("no controls")
    below = sum(1 for c in controls if c < observed)
    tied = sum(1 for c in controls if c == observed)
    return {
        "fraction_below": below / n,
        "fraction_tied": tied / n,
        "fraction_above": (n - below - tied) / n,
    }


def mc_p_value(observed: float, controls: Sequence[float]) -> float:
    """P(control >= observed) estimated as (1 + #{control >= observed}) / (N + 1)."""
    return (1 + sum(1 for c in controls if c >= observed)) / (len(controls) + 1)


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default 'linear' rule)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def uniform_subsets(n_units: int, k: int, draws: int, seed: int) -> list[tuple[int, ...]]:
    """``draws`` subsets of size ``k``, uniform without replacement, from a local seeded
    generator (never the global RNG). Returned sorted."""
    if not 1 <= k <= n_units:
        raise ValueError(f"k={k} must be in [1, {n_units}]")
    g = _generator(seed)
    return [tuple(sorted(torch.randperm(n_units, generator=g)[:k].tolist())) for _ in range(draws)]


def uniform_permutations(n_units: int, draws: int, seed: int) -> list[tuple[int, ...]]:
    g = _generator(seed)
    return [tuple(torch.randperm(n_units, generator=g).tolist()) for _ in range(draws)]


def rank_order(scores: Sequence[float], *, by: str) -> tuple[int, ...]:
    """Units ordered by descending ``|score|`` (``by="abs"``) or descending signed score
    (``by="signed"``); ties broken by lower unit index."""
    if by not in ("abs", "signed"):
        raise ValueError("by must be 'abs' or 'signed'")
    key = [abs(s) if by == "abs" else s for s in scores]
    return tuple(sorted(range(len(scores)), key=lambda i: (-key[i], i)))


def spearman_of_orders(a: Sequence[int], b: Sequence[int]) -> float:
    """Spearman correlation between two complete rankings of the same units."""
    if sorted(a) != sorted(b) or len(a) < 2:
        raise ValueError("rankings must order the same units (at least two)")
    n = len(a)
    rank_a = {u: i for i, u in enumerate(a)}
    rank_b = {u: i for i, u in enumerate(b)}
    d2 = sum((rank_a[u] - rank_b[u]) ** 2 for u in a)
    return 1 - 6 * d2 / (n * (n * n - 1))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def sign_flip_p(differences: Sequence[float], draws: int, seed: int) -> float:
    """One-sided paired sign-flip permutation p for mean(d) > 0:
    (1 + #{flipped mean >= observed mean}) / (draws + 1)."""
    if not differences:
        raise ValueError("no paired differences")
    observed = sum(differences) / len(differences)
    g = _generator(seed)
    count = 0
    for _ in range(draws):
        signs = torch.randint(0, 2, (len(differences),), generator=g).tolist()
        flipped = sum(d if s else -d for d, s in zip(differences, signs, strict=True))
        if flipped / len(differences) >= observed:
            count += 1
    return (1 + count) / (draws + 1)


def magnitude_strata(magnitudes: Sequence[float], strata: int) -> list[list[int]]:
    """Units ordered by magnitude (descending; ties by lower index), split into
    ``strata`` contiguous groups whose sizes differ by at most one."""
    order = sorted(range(len(magnitudes)), key=lambda u: (-magnitudes[u], u))
    k = min(strata, len(order))
    base, extra = divmod(len(order), k)
    out, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append(order[start : start + size])
        start += size
    return out


def stratified_subsets(
    magnitudes: Sequence[float], selected: Sequence[int], draws: int, seed: int, strata: int
) -> list[tuple[int, ...]]:
    """``draws`` random sets matched to ``selected`` by perturbation-magnitude stratum
    (ADR-035): each selected unit is replaced by a uniform draw, without replacement,
    from its own stratum. Local seeded generator; returned sorted."""
    groups = magnitude_strata(magnitudes, strata)
    where = {u: i for i, g in enumerate(groups) for u in g}
    need: dict[int, int] = {}
    for u in selected:
        need[where[u]] = need.get(where[u], 0) + 1
    g = _generator(seed)
    out = []
    for _ in range(draws):
        chosen: list[int] = []
        for stratum in sorted(need):
            members = groups[stratum]
            pick = torch.randperm(len(members), generator=g)[: need[stratum]].tolist()
            chosen += [members[i] for i in pick]
        out.append(tuple(sorted(chosen)))
    return out
