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
import os
from collections.abc import Iterable
from pathlib import Path
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
    role,
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
    SensitivityProfile,
    Standing,
    load_report,
)
from .uncertainty import Interval, bootstrap, paired_bootstrap, wilson

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
    "Interval",
    "SensitivityProfile",
    "Standing",
    "alternative",
    "audit",
    "bootstrap",
    "checkpoint_of",
    "claim",
    "concept",
    "counterexample_rule",
    "invariance",
    "load_evidence",
    "load_report",
    "paired_bootstrap",
    "plan",
    "requirement",
    "role",
    "sample_id",
    "save_evidence",
    "selection",
    "traces_of",
    "verify_report",
    "wilson",
]


def sample_id(*inputs: Any, model_kwargs: dict[str, Any] | None = None) -> str:
    """The exact sample identity BeyondNN records for these inputs (``plan(samples=...)``,
    ``claim(sample_targets=...)``); the same function the recorders use."""
    from beyondnn.core.samples import sample_id as _sample_id

    return _sample_id(*inputs, model_kwargs=model_kwargs)


def traces_of(evidence: Iterable[Any]) -> list[Any]:
    """Exactly the traces :func:`audit` ingests from ``evidence`` (traces, saved trace paths,
    or result objects with their nested traces), deduplicated, in a deterministic order."""
    from .evidence import collect_traces

    return collect_traces(evidence)


def save_evidence(evidence: Iterable[Any], directory: str | os.PathLike[str]) -> list[Path]:
    """Save :func:`traces_of` ``evidence`` into the new directory ``directory`` (one trace
    directory each, ``trace0000``...), so ``audit(load_evidence(directory), plan=...)`` in
    another process sees exactly the same evidence. Never overwrites."""
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    target.mkdir(parents=True)
    paths = []
    for i, trace in enumerate(traces_of(evidence)):
        path = target / f"trace{i:04d}"
        trace.save(path)
        paths.append(path)
    return paths


def load_evidence(directory: str | os.PathLike[str]) -> list[Path]:
    """The trace directories written by :func:`save_evidence`, in their saved order (pass
    them to :func:`audit` or ``concepts.load_validation``; each is fully re-validated when
    loaded)."""
    paths = sorted(
        p for p in Path(directory).iterdir() if p.is_dir() and p.name.startswith("trace")
    )
    if not paths:
        raise FileNotFoundError(f"no saved traces in {directory}")
    return paths


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
    # the scientific content must match exactly; who produced it is reported, not compared
    envelope = ("producer", "format_version")
    content = {k: v for k, v in stored.items() if k not in envelope}
    if content != {k: v for k, v in expected.items() if k not in envelope}:
        paths = list(_diff(content, {k: v for k, v in expected.items() if k not in envelope}, "$"))
        produced = (stored.get("producer") or {}).get("audit_semantics")
        note = (
            ""
            if produced == expected["producer"]["audit_semantics"]
            else f"; the stored report was produced under audit semantics "
            f"{produced if produced is not None else '<= 2 (not recorded)'}, this BeyondNN "
            f"applies {expected['producer']['audit_semantics']}, so a difference may be a "
            "change of audit rules rather than of evidence"
        )
        raise AuditMismatchError(
            f"the stored report differs from the re-derived audit at {len(paths)} path(s): "
            + ", ".join(paths[:12])
            + (" ..." if len(paths) > 12 else "")
            + note
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
