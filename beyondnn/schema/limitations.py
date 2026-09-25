"""Trace limitations: first-class statements of what a result does not cover.

Only codes emitted by the current design are registered. A code is added in the
milestone that first emits it, with its meaning and emission rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ._types import require
from .base import BaseRecord, is_record_id, record_kind
from .errors import UnknownLimitationCodeError

__all__ = ["LIMITATIONS", "LimitationDef", "Severity", "TraceLimitation"]


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class LimitationDef:
    code: str
    meaning: str
    emitted_when: str
    severity: Severity


def _defs(*defs: LimitationDef) -> Mapping[str, LimitationDef]:
    table = {d.code: d for d in defs}
    assert len(table) == len(defs), "duplicate limitation code"
    return MappingProxyType(table)


#: The closed registry of limitation codes.
LIMITATIONS: Mapping[str, LimitationDef] = _defs(
    LimitationDef(
        "FUNCTIONAL_OPS_UNOBSERVED",
        "Computation performed between module boundaries (functional calls, tensor "
        "arithmetic inside forward) is not observed.",
        "Always, by trace() and recording(): hooks only see module inputs and outputs.",
        Severity.INFO,
    ),
    LimitationDef(
        "PARTIAL_SITE_COVERAGE",
        "Only a subset of modules was recorded; unrecorded modules may carry relevant computation.",
        "By trace()/recording() when the output sites selected do not cover every named "
        "(non-root) module of the model.",
        Severity.WARNING,
    ),
    LimitationDef(
        "SELECTED_SITE_NOT_EXECUTED",
        "A selected site was valid but never executed during the trace, so it produced "
        "no evidence; this is not coverage.",
        "By trace()/recording() for every selected (module, input|output) site with no "
        "observed execution in any root pass (e.g. an iterated ModuleList).",
        Severity.WARNING,
    ),
    LimitationDef(
        "NON_TENSOR_LEAVES_IGNORED",
        "Some observed inputs/outputs contained non-tensor values (numbers, strings, "
        "objects); they were not recorded as evidence.",
        "By trace()/recording() when a walked value has leaves other than tensors or None.",
        Severity.INFO,
    ),
    LimitationDef(
        "ZERO_ABLATION_MAY_BE_OOD",
        "The intervention replaced an activation with zeros; zero is not a neutral or "
        "in-distribution value, so the effect may reflect an off-distribution state.",
        "By intervene() for ZERO interventions.",
        Severity.WARNING,
    ),
    LimitationDef(
        "CONSTANT_REPLACEMENT_MAY_BE_OOD",
        "The intervention replaced an activation with a caller-chosen constant, which "
        "may not be a value the model produces; the effect may reflect an off-"
        "distribution state.",
        "By intervene() for CONSTANT interventions.",
        Severity.WARNING,
    ),
    LimitationDef(
        "PATCH_SOURCE_CONTEXT_DIFFERS",
        "The patched activation came from an execution on a different input, so it is "
        "placed into a context it was not computed in.",
        "By intervene() for PATCH interventions whose source input differs from the target.",
        Severity.INFO,
    ),
    LimitationDef(
        "CUSTOM_METRIC_UNVERIFIED",
        "The outcome metric is caller-supplied code; BeyondNN recorded only its name and "
        "the caller-declared revision/config (ADR-029), and cannot verify that the code "
        "matches that declaration or is deterministic and free of side effects.",
        "By intervene() when the metric is not a BeyondNN built-in.",
        Severity.WARNING,
    ),
    LimitationDef(
        "ATTRIBUTION_BASELINE_ASSUMPTION",
        "The attribution is relative to a chosen baseline; a different baseline (zero is "
        "not neutral) gives different attributions. Scores describe the path from the "
        "baseline to the input, not the input alone.",
        "By attribute() for every Integrated Gradients attribution (the detail names the "
        "baseline).",
        Severity.WARNING,
    ),
    LimitationDef(
        "ATTRIBUTION_NUMERICAL_APPROXIMATION",
        "The attribution is a numerical approximation of an integral (finite steps of a "
        "stated rule); the completeness delta is a diagnostic of that approximation, not "
        "a confidence score.",
        "By attribute() for every Integrated Gradients attribution.",
        Severity.INFO,
    ),
    LimitationDef(
        "DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION",
        "The model has discrete (integer) inputs, which have no gradients; the attribution "
        "is to a module's output representation (e.g. embedding dimensions per position), "
        "not to the discrete ids themselves.",
        "By attribute() for layer attributions when any model input tensor is not floating point.",
        Severity.INFO,
    ),
    LimitationDef(
        "LAYER_ATTRIBUTION_PARTIAL_COVERAGE",
        "The attribution is to one leaf of one call of one module output; computation that "
        "bypasses it (residual connections, functional operations, other calls or modules) "
        "is not attributed, so these scores need not account for the target.",
        "By attribute() for every module (layer) attribution.",
        Severity.INFO,
    ),
    LimitationDef(
        "SITE_RELATIVE_SUFFICIENCY",
        "Retention was applied within one site only: units of that site outside the "
        "selection were replaced, but computation that bypasses the site (other modules, "
        "residual paths, other calls) was left intact. The result is sufficiency at this "
        "site, not sufficiency for the model's computation.",
        "By faithfulness sufficiency tests and retention curves at internal (module) sites.",
        Severity.WARNING,
    ),
    LimitationDef(
        "SELECTION_TIE_AT_BOUNDARY",
        "The k-th and (k+1)-th ranked units have equal scores; which of them was selected "
        "is decided only by the declared tie-break (lower unit index), not by the method.",
        "By faithfulness runs whose attribution selection has a tie across the k boundary.",
        Severity.WARNING,
    ),
    LimitationDef(
        "DECLARED_TRANSFORMATION_UNVERIFIED",
        "The invariance transformation is caller code, declared by name and revision; "
        "BeyondNN cannot verify that it preserves what matters for the task or that the "
        "code matches its declaration.",
        "By stability tests.",
        Severity.WARNING,
    ),
    LimitationDef(
        "NO_ATTRIBUTION",
        "No attribution method was run; no input or unit importance scores exist.",
        "By explain() when its plan contains no attribution method.",
        Severity.INFO,
    ),
    LimitationDef(
        "NO_CAUSAL_EVIDENCE",
        "No intervention was performed; nothing in the result is causal evidence.",
        "By explain() when no INTERVENTIONAL or ESTIMATED_CAUSAL evidence is present.",
        Severity.WARNING,
    ),
    LimitationDef(
        "NO_CLAIMS_TESTED",
        "No claim about the computation has a test result; there are no assessments.",
        "By explain() when no claim test result is present.",
        Severity.INFO,
    ),
)


@record_kind("limitation")
@dataclass(frozen=True, slots=True, kw_only=True)
class TraceLimitation(BaseRecord):
    """A registered limitation, optionally scoped to specific records.

    ``applies_to`` lists record ids (sorted on construction); empty means the whole
    result. ``detail`` adds instance-specific context to the registered meaning.
    """

    code: str
    detail: str | None = None
    applies_to: tuple[str, ...] = ()

    def _validate(self) -> None:
        if self.code not in LIMITATIONS:
            raise UnknownLimitationCodeError(
                f"unknown limitation code {self.code!r}; registered: {sorted(LIMITATIONS)}"
            )
        require(all(is_record_id(i) for i in self.applies_to), "applies_to has a malformed id")
        require(len(set(self.applies_to)) == len(self.applies_to), "applies_to has duplicate ids")
        object.__setattr__(self, "applies_to", tuple(sorted(self.applies_to)))

    @property
    def definition(self) -> LimitationDef:
        return LIMITATIONS[self.code]

    @property
    def severity(self) -> Severity:
        return LIMITATIONS[self.code].severity
