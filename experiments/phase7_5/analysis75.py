"""Phase 7.5 analysis helpers: pure functions over raw result JSON (no torch, no model).

Used by ``figure_data.py`` and the report tables; unit-tested in ``tests/test_phase75_analysis.py``
(including the mutation targets of plan §32: ground-truth labels, FP/FN counts, benchmark
scope, perturbation labels, and the plausibility / faithfulness separation).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Ground-truth labels of the external benchmark (frozen policy §8).
NECESSARY, NOT_NECESSARY, AMBIGUOUS = "necessary", "not_necessary", "ambiguous"
#: Token strategies of the OOD probes and whether each is model-native (literature §5).
NATIVE = {"zero": False, "mask": True, "pad": False, "unk": True, "delete": True}


def confusion(
    rows: Iterable[Mapping[str, Any]], *, kinds: Sequence[str] = ("head", "mlp")
) -> dict[str, int]:
    """Confusion of PRIMARY standings against the external ground truth (frozen policy §9).

    Ambiguous instances are counted separately and never enter TP/FN/FP/TN; standings other
    than SUPPORTED / CONTRADICTED on a clear instance are 'other' (never counted as correct).
    """
    out = {"tp": 0, "fn": 0, "pos_other": 0, "tn": 0, "fp": 0, "neg_other": 0, "ambiguous": 0}
    for r in rows:
        if r["kind"] not in kinds:
            continue
        gt, standing = r["gt"], r["standing"]
        if gt == AMBIGUOUS:
            out["ambiguous"] += 1
        elif gt == NECESSARY:
            key = {"supported": "tp", "contradicted": "fn"}.get(standing, "pos_other")
            out[key] += 1
        elif gt == NOT_NECESSARY:
            key = {"contradicted": "tn", "supported": "fp"}.get(standing, "neg_other")
            out[key] += 1
        else:
            raise ValueError(f"unknown ground-truth label {gt!r}")
    return out


def configuration_confusion(
    rows: Iterable[Mapping[str, Any]], *, kinds: Sequence[str] = ("head", "mlp")
) -> dict[str, dict[str, int]]:
    """For every single configuration, treated as if it were the only one: SUPPORTS on
    necessary (tp) / not-necessary (fp) instances, among clear instances where it was run."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["kind"] not in kinds or r["gt"] == AMBIGUOUS:
            continue
        for config, outcome in r["configs"].items():
            cell = out.setdefault(config, {"tp": 0, "pos": 0, "fp": 0, "neg": 0})
            if r["gt"] == NECESSARY:
                cell["pos"] += 1
                cell["tp"] += outcome == "supports"
            else:
                cell["neg"] += 1
                cell["fp"] += outcome == "supports"
    return out


def reliable_rows(cases: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows of benchmark-reliable cases only (frozen policy §8), tagged with their case."""
    out = []
    for c, r in cases.items():
        if "error" in r or not r.get("benchmark_reliable", False):
            continue
        out += [dict(row, case=c) for row in r["rows"]]
    return out


def iou(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def token_f1(pred: set[int], gold: set[int]) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))


def rationale_summary(
    samples: Sequence[Mapping[str, Any]], audits: Mapping[str, Any]
) -> dict[str, Any]:
    """Plausibility (agreement with human highlights) and faithfulness (audited PRIMARY
    standings) side by side -- never merged: human highlights are not causal ground truth."""
    plausibility = {
        "ig_token_f1": [s["plausibility"]["ig_token_f1"] for s in samples],
        "random_token_f1": [s["plausibility"]["random_token_f1"] for s in samples],
    }
    faithfulness = {
        approach: {name: claim["distribution"] for name, claim in claims.items()}
        for approach, claims in audits.items()
    }
    return {
        "plausibility": plausibility,
        "faithfulness": faithfulness,
        "human_is_ground_truth": False,
    }


def per_sample_support(per_sample: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """(supporting configurations, tested configurations) per sample, from central75 output."""
    out = []
    for s in per_sample:
        sup = sum(n for _role, outcome, n in s["outcomes"] if outcome == "supports")
        out.append((sup, s["tested"]))
    return out
