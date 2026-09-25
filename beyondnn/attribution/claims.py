"""Attribution-backed claim tests (Phase 3; ADR-030): ``attribution_threshold`` v1.

A claim "S is ATTRIBUTED_TO target M on input X" is decided by the
``AttributionRecord`` of a declared method (spec and baseline) at S (and the
declared module call) for M on exactly X:

* score = ``sum |attribution|`` over the claim subject (every element, or the
  listed units along the last dimension);
* SUPPORTS iff ``score >= min_abs_attribution`` (declared before running), else
  CONTRADICTS;
* any mismatch of method, baseline, call, site/leaf, target, or input:
  NOT_APPLICABLE.

This is a statement about an attribution method's output. It is not evidence that
S is necessary, sufficient, or a cause: ATTRIBUTED_TO is the only relation this
protocol justifies, and the schema refuses attribution evidence for causal
relations.
"""

from __future__ import annotations

from typing import Any

import torch

from beyondnn.interventions.metrics import Metric
from beyondnn.interventions.runner import sample_id
from beyondnn.protocols import ATTRIBUTION_THRESHOLD, PROTOCOLS
from beyondnn.schema import (
    AssessmentPolicy,
    AttributionBaseline,
    AttributionRecord,
    Claim,
    ClaimSource,
    ClaimSourceKind,
    ClaimTestResult,
    ClaimTestSpec,
    Estimand,
    JsonMap,
    Outcome,
    PolicyRequirement,
    Relation,
    Subject,
)

from .runner import _effective_spec
from .spec import At, Method

__all__ = ["ATTRIBUTION_POLICY", "evaluate_claim", "make_claim", "threshold_spec"]

#: ATTRIBUTED_TO needs a SUPPORTS result from ``attribution_threshold``. No causal
#: relation appears here, and none can: the registry does not allow it.
ATTRIBUTION_POLICY = AssessmentPolicy(
    name="attribution_threshold_policy",
    version=1,
    requirements=(
        PolicyRequirement(relation=Relation.ATTRIBUTED_TO, protocols=(ATTRIBUTION_THRESHOLD,)),
    ),
)


def _record_baseline(baseline: AttributionBaseline | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    out: dict[str, Any] = {"kind": baseline.kind.value}
    if baseline.value is not None:
        out["digest"] = baseline.value.content_digest
    if baseline.input_path is not None:
        out["input_path"] = baseline.input_path
    return out


def threshold_spec(
    method: Method, *, at: At | None = None, min_abs_attribution: float
) -> ClaimTestSpec:
    """Declare an ``attribution_threshold`` test for ``method`` at ``at`` before running."""
    if not isinstance(method, Method):
        raise TypeError("method must come from beyondnn.attribution constructors")
    if isinstance(min_abs_attribution, bool) or not min_abs_attribution > 0:
        raise ValueError("min_abs_attribution must be > 0")
    at = at if at is not None else At()
    return ClaimTestSpec(
        protocol=ATTRIBUTION_THRESHOLD,
        protocol_version=1,
        applicable_relations=(Relation.ATTRIBUTED_TO,),
        criteria=JsonMap({"min_abs_attribution": float(min_abs_attribution)}),
        params=JsonMap(
            {
                "method": _effective_spec(method, at).params.to_plain()
                | {
                    "name": method.spec.name,
                    "implementation": method.spec.implementation,
                    "implementation_version": method.spec.implementation_version,
                },
                "baseline": None if method.baseline is None else method.baseline.identity(),
                "call_index": at.call_index,
                "score": "abs_sum",
            }
        ),
    )


def make_claim(
    at: At,
    target: Metric,
    *inputs: Any,
    statement: str,
    units: tuple[int, ...] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    source: ClaimSource | None = None,
) -> Claim:
    """An ATTRIBUTED_TO claim about ``at`` and ``target`` on exactly this input."""
    return Claim(
        statement=statement,
        relation=Relation.ATTRIBUTED_TO,
        subject=Subject(site=at.site(), units=units),
        target=target.spec.target(),
        estimand=Estimand.instance(sample_id(*inputs, model_kwargs=model_kwargs)),
        source=source or ClaimSource(kind=ClaimSourceKind.USER),
    )


def evaluate_claim(
    claim: Claim, spec: ClaimTestSpec, record: AttributionRecord, attribution: torch.Tensor
) -> ClaimTestResult:
    """Decide ``claim`` from ``record`` (whose retained tensor is ``attribution``)."""
    if spec.protocol != ATTRIBUTION_THRESHOLD or spec.protocol_version != 1:
        raise ValueError("evaluate_claim implements attribution_threshold v1 only")
    threshold = spec.criteria.get("min_abs_attribution")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError("spec criteria must declare min_abs_attribution > 0")
    declared_method = spec.params.get("method")
    method = record.method
    recorded_method = method.params.to_plain() | {
        "name": method.name,
        "implementation": method.implementation,
        "implementation_version": method.implementation_version,
    }
    units = claim.subject.units
    applicable = (
        claim.relation in spec.applicable_relations
        and claim.relation in PROTOCOLS[ATTRIBUTION_THRESHOLD]
        and declared_method == JsonMap(recorded_method)
        and spec.params.get("baseline") == _jsonish(_record_baseline(record.baseline))
        and spec.params.get("call_index") == record.call_index
        and claim.subject.site == record.site
        and claim.target == record.target.target()
        and claim.estimand == Estimand.instance(record.sample_id)
        and (units is None or (attribution.dim() > 0 and max(units) < attribution.shape[-1]))
    )
    if not applicable:
        return ClaimTestResult.for_claim(
            claim, spec, outcome=Outcome.NOT_APPLICABLE, provenance_id=record.provenance_id or ""
        )
    selected = attribution if units is None else attribution[..., list(units)]
    score = float(selected.double().abs().sum())
    t = float(threshold)
    return ClaimTestResult.for_claim(
        claim,
        spec,
        outcome=Outcome.SUPPORTS if score >= t else Outcome.CONTRADICTS,
        evidence=(record,),
        statistics=JsonMap({"abs_attribution": score, "min_abs_attribution": t}),
        provenance_id=record.provenance_id or "",
    )


def _jsonish(value: Any) -> Any:
    """``value`` as it reads back from a JsonMap (for equality with spec params)."""
    return JsonMap({"v": value}).get("v")
