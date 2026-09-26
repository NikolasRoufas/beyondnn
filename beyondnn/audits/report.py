"""The audit report: standings, findings, and their structured presentation (ADR-044).

Everything here is data derived by :func:`beyondnn.audits.audit`. Prose is rendered
only from these structures (:meth:`AuditReport.render`). There is no score, no
confidence, and no statement that an explanation is trustworthy: standings say what
the recorded evidence establishes *under the declared plan*, nothing more.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from beyondnn.schema import AuditedClaim, AuditedConcept, AuditPlan, to_dict

__all__ = [
    "REPORT_FORMAT",
    "REPORT_FORMAT_VERSION",
    "AuditReport",
    "ClaimAudit",
    "ConceptAudit",
    "Coverage",
    "Diagnostic",
    "EvidenceSummary",
    "Finding",
    "FindingKind",
    "FindingSeverity",
    "GroupAudit",
    "InventoryEntry",
    "Standing",
    "TestEntry",
    "load_report",
]

REPORT_FORMAT = "beyondnn.audit_report"
REPORT_FORMAT_VERSION = 1


class Standing(Enum):
    """What the in-scope, re-derived evidence establishes about a claim under the plan
    (plan §22). Not a probability, not a quality grade, and never "trustworthy"."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    ASSUMPTION_SENSITIVE = "assumption_sensitive"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    NOT_EVALUATED = "not_evaluated"


class FindingKind(Enum):
    """What a finding is about (plan §22; LIMITATION added, see plan Deviations)."""

    CONTRADICTION = "contradiction"
    ASSUMPTION_SENSITIVE = "assumption_sensitive"
    PROTOCOL_DISAGREEMENT = "protocol_disagreement"
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_CONTROL = "missing_control"
    EVIDENCE_TYPE_MISMATCH = "evidence_type_mismatch"
    SCOPE_MISMATCH = "scope_mismatch"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    COUNTEREXAMPLE_FOUND = "counterexample_found"
    INCONCLUSIVE_EVIDENCE = "inconclusive_evidence"
    INTEGRITY_FAILURE = "integrity_failure"
    NOT_EVALUATED = "not_evaluated"
    LIMITATION = "limitation"


class FindingSeverity(Enum):
    """Categorical, never summed or compared across claims (plan §23).

    BLOCKING: the claim as stated is not established by the supplied evidence under
    the plan. QUALIFYING: a stated claim must carry this qualification.
    INFORMATIONAL: context."""

    BLOCKING = "blocking"
    QUALIFYING = "qualifying"
    INFORMATIONAL = "informational"


@dataclass(frozen=True, slots=True)
class Finding:
    """One finding. ``values`` pairs an outcome or side with an axis value (for
    sensitivity findings); ``records`` / ``samples`` identify what it is about."""

    kind: FindingKind
    code: str
    severity: FindingSeverity
    subject: str
    detail: str
    axis: str | None = None
    values: tuple[tuple[str, str], ...] = ()
    records: tuple[str, ...] = ()
    samples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TestEntry:
    """A matched result (or a re-evaluation of one under declared alternative
    criteria, ``alternative`` set; never part of a verdict)."""

    result: str
    protocol: str
    outcome: str
    axes: tuple[tuple[str, str], ...]
    alternative: str | None = None


@dataclass(frozen=True, slots=True)
class GroupAudit:
    """A claim audited on one group: one sample (per-sample claims) or the claim's
    declared estimand (``sample=None``)."""

    sample: str | None
    standing: Standing
    verdict: str
    tests: tuple[TestEntry, ...]
    findings: tuple[Finding, ...]
    axis_values: tuple[tuple[str, tuple[str, ...]], ...]
    related: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimAudit:
    """One plan claim. Per-sample claims have ``standing=None``: their outcome is the
    distribution of per-sample standings (plan §21), never a single truth value."""

    claim: AuditedClaim
    per_sample: bool
    standing: Standing | None
    groups: tuple[GroupAudit, ...]
    distribution: tuple[tuple[str, int], ...]
    counterexamples: tuple[str, ...]
    findings: tuple[Finding, ...]
    limitations: tuple[tuple[str, int], ...]

    @property
    def name(self) -> str:
        return self.claim.name

    def group(self, sample: str | None) -> GroupAudit:
        for g in self.groups:
            if g.sample == sample:
                return g
        raise KeyError(sample)

    def count(self, standing: Standing) -> int:
        return dict(self.distribution).get(standing.value, 0)


@dataclass(frozen=True, slots=True)
class ConceptAudit:
    """One plan concept: its validations and tests in scope, standing, findings, and
    the counterexample identities (false positives / negatives)."""

    concept: AuditedConcept
    label: str | None
    label_source: str | None
    standing: Standing
    validations: tuple[tuple[str, str, tuple[str, ...]], ...]
    tests: tuple[TestEntry, ...]
    findings: tuple[Finding, ...]
    false_positives: tuple[str, ...]
    false_negatives: tuple[str, ...]
    limitations: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """A claim record found in the evidence (plan §6)."""

    claim: str
    relation: str
    subject: str
    target: str
    scope: str
    sample: str | None
    results: tuple[tuple[str, str, str, str], ...]
    assessments: tuple[tuple[str, str], ...]
    matched: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A diagnostic protocol result (context; never decides a standing)."""

    record: str
    protocol: str
    samples: int
    verified: bool
    outcomes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    digest: str
    traces: int
    records: int
    results: int
    results_included: int
    results_excluded: int


@dataclass(frozen=True, slots=True)
class Coverage:
    """What was and was not evaluated. Coverage, not confidence (plan §20)."""

    claims: int
    claims_by_standing: tuple[tuple[str, int], ...]
    per_sample_claims: tuple[tuple[str, int, int], ...]
    concepts: int
    concepts_evaluated: int
    plan_samples: int
    samples_with_evidence: int
    evidence_statuses: tuple[str, ...]
    protocols_run: tuple[str, ...]
    protocols_required_not_run: tuple[str, ...]
    untested_invariances: tuple[tuple[str, str], ...]
    counterexample_caps: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True, eq=False)
class AuditReport:
    """The result of :func:`beyondnn.audits.audit` (plan §5)."""

    plan: AuditPlan
    evidence: EvidenceSummary
    inventory: tuple[InventoryEntry, ...]
    claims: tuple[ClaimAudit, ...]
    concepts: tuple[ConceptAudit, ...]
    findings: tuple[Finding, ...]
    diagnostics: tuple[Diagnostic, ...]
    coverage: Coverage
    limitations: tuple[tuple[str, int], ...]

    # ------------------------------------------------------------------ lookups

    def claim(self, name: str) -> ClaimAudit:
        for c in self.claims:
            if c.name == name:
                return c
        raise KeyError(name)

    def concept(self, concept_id: str) -> ConceptAudit:
        for c in self.concepts:
            if c.concept.concept == concept_id:
                return c
        raise KeyError(concept_id)

    # ------------------------------------------------------------------ sections

    def _standing(self, standing: Standing) -> tuple[ClaimAudit, ...]:
        return tuple(c for c in self.claims if c.standing is standing)

    @property
    def supported(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.SUPPORTED)

    @property
    def contradicted(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.CONTRADICTED)

    @property
    def mixed(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.MIXED)

    @property
    def assumption_sensitive(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.ASSUMPTION_SENSITIVE)

    @property
    def inconclusive(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.INCONCLUSIVE)

    @property
    def unsupported(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.UNSUPPORTED)

    @property
    def not_evaluated(self) -> tuple[ClaimAudit, ...]:
        return self._standing(Standing.NOT_EVALUATED)

    @property
    def per_sample(self) -> tuple[ClaimAudit, ...]:
        return tuple(c for c in self.claims if c.per_sample)

    def all_findings(self) -> tuple[Finding, ...]:
        """Report-level, claim-level (per-sample findings aggregated) and concept findings."""
        out = list(self.findings)
        for c in self.claims:
            out += c.findings
        for k in self.concepts:
            out += k.findings
        return tuple(out)

    def _kind(self, *kinds: FindingKind) -> tuple[Finding, ...]:
        return tuple(f for f in self.all_findings() if f.kind in kinds)

    @property
    def missing_evidence(self) -> tuple[Finding, ...]:
        return self._kind(FindingKind.MISSING_EVIDENCE, FindingKind.NOT_EVALUATED)

    @property
    def missing_controls(self) -> tuple[Finding, ...]:
        return self._kind(FindingKind.MISSING_CONTROL)

    @property
    def counterexamples(self) -> tuple[Finding, ...]:
        return self._kind(FindingKind.COUNTEREXAMPLE_FOUND)

    @property
    def protocol_disagreements(self) -> tuple[Finding, ...]:
        return self._kind(FindingKind.PROTOCOL_DISAGREEMENT)

    @property
    def contradictions(self) -> tuple[Finding, ...]:
        return self._kind(FindingKind.CONTRADICTION)

    @property
    def overclaims(self) -> tuple[Finding, ...]:
        """Structural overclaims (plan §10): type, scope, untested invariance, controls."""
        return tuple(
            f
            for f in self.all_findings()
            if f.kind in (FindingKind.EVIDENCE_TYPE_MISMATCH, FindingKind.MISSING_CONTROL)
            or f.code in ("narrower_estimand", "invariance_untested")
        )

    def sensitivity(self, axis: str) -> tuple[Finding, ...]:
        """ASSUMPTION_SENSITIVE findings on ``axis`` (replacement, threshold, k, null, ...)."""
        return tuple(f for f in self._kind(FindingKind.ASSUMPTION_SENSITIVE) if f.axis == axis)

    @property
    def provenance_findings(self) -> tuple[Finding, ...]:
        return self._kind(
            FindingKind.PROVENANCE_MISMATCH,
            FindingKind.INTEGRITY_FAILURE,
            FindingKind.SCOPE_MISMATCH,
        )

    @property
    def concept_findings(self) -> tuple[Finding, ...]:
        return tuple(f for k in self.concepts for f in k.findings)

    @property
    def faithfulness_findings(self) -> tuple[Finding, ...]:
        return tuple(f for c in self.claims if c.claim.selection is not None for f in c.findings)

    # ------------------------------------------------------------------ output

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-ready (plan §25)."""
        return {
            "format": REPORT_FORMAT,
            "format_version": REPORT_FORMAT_VERSION,
            "plan": to_dict(self.plan),
            "plan_id": self.plan.id,
            "evidence": _plain(self.evidence),
            "inventory": _plain(self.inventory),
            "claims": _plain(self.claims),
            "concepts": _plain(self.concepts),
            "findings": _plain(self.findings),
            "diagnostics": _plain(self.diagnostics),
            "coverage": _plain(self.coverage),
            "limitations": _plain(self.limitations),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1, allow_nan=False)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the report JSON to the new file ``path`` (atomic; never overwrites)."""
        target = Path(path)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"{target} already exists")
        text = self.to_json() + "\n"
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.link(tmp, target)  # fails if target appeared meanwhile
        finally:
            os.unlink(tmp)

    def render(self) -> str:
        """Prose rendered only from the structured report."""
        from .render import render

        return render(self)


def load_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """A saved report as a dict (verify it with :func:`beyondnn.audits.verify_report`)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != REPORT_FORMAT:
        raise ValueError(f"{path} is not a BeyondNN audit report")
    if document.get("format_version") != REPORT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported audit report format_version {document.get('format_version')!r}"
        )
    return document


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, AuditedClaim):
        return value.name  # the plan (included in full) holds the declaration
    if isinstance(value, AuditedConcept):
        return value.concept
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def aggregate(findings: Iterable[Finding], subject: str) -> tuple[Finding, ...]:
    """Merge per-group findings into claim-level findings by (kind, code, severity,
    axis): samples and records are unioned (sorted), details kept if identical."""
    groups: dict[tuple[str, str, str, str], list[Finding]] = {}
    for f in findings:
        key = (f.kind.value, f.code, f.severity.value, f.axis or "")
        groups.setdefault(key, []).append(f)
    out = []
    for key in sorted(groups):
        items = groups[key]
        first = items[0]
        details = sorted({f.detail for f in items})
        samples = sorted({s for f in items for s in f.samples})
        values = sorted({v for f in items for v in f.values})
        detail = details[0] if len(details) == 1 else f"{first.code} on {len(samples)} sample(s)"
        out.append(
            Finding(
                kind=first.kind,
                code=first.code,
                severity=first.severity,
                subject=subject,
                detail=detail,
                axis=first.axis,
                values=tuple(values),
                records=(),
                samples=tuple(samples),
            )
        )
    return tuple(out)
