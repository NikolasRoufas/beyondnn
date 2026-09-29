"""Phase 7.75 analysis helpers: pure functions over result JSON (no torch, no model).

Unit-tested in ``tests/test_phase775_analysis.py`` (the mutation targets of request §32:
independence of ground truth, confusion cells, the pre-declared concept threshold, shortcut
stratification kept as analysis only).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

KNOWN_TRUE, KNOWN_FALSE, EMPTY = "known_true", "known_false", "empty"
STANDINGS = ("supported", "contradicted", "unsupported", "inconclusive", "other")
C_REL = 0.2  # ADR-055


def truth_is_independent(truth_operation: str, audit_operation: str) -> bool:
    """Ground truth defined by the audited operation itself is an implementation-consistency
    check, never an independent validation (request §6)."""
    if not truth_operation or not audit_operation:
        raise ValueError("declare both the ground-truth and the audited operation")
    return truth_operation.strip().lower() != audit_operation.strip().lower()


def truth_table(
    rows: Iterable[Mapping[str, Any]], *, kinds: Sequence[str] = ("head", "mlp")
) -> dict[str, dict[str, int]]:
    """Every truth x standing cell (request §8); nothing collapsed into one metric."""
    out = {t: dict.fromkeys(STANDINGS, 0) for t in (KNOWN_TRUE, KNOWN_FALSE, EMPTY)}
    for r in rows:
        if r["kind"] not in kinds:
            continue
        truth = r["truth"]
        if truth not in out:
            raise ValueError(f"unknown truth label {truth!r}")
        standing = r["standing"] if r["standing"] in STANDINGS else "other"
        out[truth][standing] += 1
    return out


def component_support(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per (program, component): its truth and how many samples are PRIMARY-SUPPORTED."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["kind"] == "selection":
            continue
        key = f"{r['program']}/{r['component']}"
        cell = out.setdefault(key, {"truth": r["truth"], "supported": 0, "n": 0})
        cell["n"] += 1
        cell["supported"] += r["standing"] == "supported"
    return out


def configuration_rates(
    rows: Iterable[Mapping[str, Any]], config: str, *, truth: str
) -> tuple[int, int]:
    """(SUPPORTS, instances) of one configuration on instances of ``truth``."""
    k = n = 0
    for r in rows:
        if r["kind"] == "selection" or r["truth"] != truth or config not in r["configs"]:
            continue
        n += 1
        k += r["configs"][config] == "supports"
    return k, n


def rel_min_change(train_target_values: Sequence[float], c: float = C_REL) -> float:
    """ADR-055: ``c`` x SD of the clean target over the train split. Takes train values only,
    so a held-out outcome can never enter the threshold."""
    if len(train_target_values) < 2:
        raise ValueError("need at least two train-split target values")
    return c * statistics.stdev(train_target_values)


def stratify(
    per_sample: Sequence[str], strata: Sequence[bool], *, labels: tuple[str, str]
) -> dict[str, dict[str, int]]:
    """Standings split by a boolean stratum (e.g. empty-premise predictable). Analysis only:
    every sample stays in exactly one stratum; none is removed."""
    if len(per_sample) != len(strata):
        raise ValueError("one stratum per sample")
    out: dict[str, dict[str, int]] = {labels[0]: {}, labels[1]: {}}
    for standing, flag in zip(per_sample, strata, strict=True):
        cell = out[labels[0] if flag else labels[1]]
        cell[standing] = cell.get(standing, 0) + 1
    return out
