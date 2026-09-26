"""Scientific audits of interpretability evidence (Phase 7; ADR-044..047).

``bnn.audit(evidence, plan=plan)`` classifies the claims and concepts an
:class:`~beyondnn.schema.AuditPlan` declares from recorded, re-derived, in-scope
evidence: SUPPORTED, CONTRADICTED, MIXED, ASSUMPTION_SENSITIVE, INCONCLUSIVE,
UNSUPPORTED or NOT_EVALUATED, with the findings behind each standing. It runs no
model, computes no score (ADR-007), resolves no contradiction, and never states that
an explanation is trustworthy. Missing evidence is NOT_EVALUATED.

Build plans with :func:`plan`, :func:`claim`, :func:`selection`, :func:`requirement`,
:func:`concept`, :func:`invariance`, :func:`alternative`, :func:`counterexample_rule`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from beyondnn.schema import AuditPlan

from .engine import AuditPlanError, audit
from .plan import (
    alternative,
    checkpoint_of,
    claim,
    concept,
    counterexample_rule,
    invariance,
    plan,
    requirement,
    selection,
)
from .report import (
    AuditReport,
    ClaimAudit,
    ConceptAudit,
    Coverage,
    Finding,
    FindingKind,
    FindingSeverity,
    GroupAudit,
    Standing,
    load_report,
)

__all__ = [
    "AuditMismatchError",
    "AuditPlanError",
    "AuditReport",
    "ClaimAudit",
    "ConceptAudit",
    "Coverage",
    "Finding",
    "FindingKind",
    "FindingSeverity",
    "GroupAudit",
    "Standing",
    "alternative",
    "audit",
    "checkpoint_of",
    "claim",
    "concept",
    "counterexample_rule",
    "invariance",
    "load_report",
    "plan",
    "requirement",
    "selection",
    "verify_report",
]


class AuditMismatchError(ValueError):
    """A stored audit report does not follow from its evidence and plan."""


def verify_report(
    document: dict[str, Any] | AuditReport, evidence: Iterable[Any], plan: AuditPlan
) -> AuditReport:
    """Re-run the audit and require the stored report to equal it exactly (plan §26).

    On any difference raises :class:`AuditMismatchError` naming the differing paths;
    the stored report is never corrected."""
    stored = document.to_dict() if isinstance(document, AuditReport) else document
    stored = json.loads(json.dumps(stored))
    fresh = audit(evidence, plan=plan)
    expected = json.loads(fresh.to_json())
    if stored != expected:
        paths = list(_diff(stored, expected, "$"))
        raise AuditMismatchError(
            f"the stored report differs from the re-derived audit at {len(paths)} path(s): "
            + ", ".join(paths[:12])
            + (" ..." if len(paths) > 12 else "")
        )
    return fresh


def _diff(a: Any, b: Any, path: str) -> Iterable[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield f"{path}.{key}"
            else:
                yield from _diff(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}[len {len(a)} != {len(b)}]"
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            yield from _diff(x, y, f"{path}[{i}]")
    elif a != b:
        yield path
