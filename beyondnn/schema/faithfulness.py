"""Faithfulness-test records (Phase 5; ADR-033).

Pure data. The runtime lives in :mod:`beyondnn.faithfulness`. Neither kind is
evidence (status ``None``): the raw measurements of a faithfulness test are the
INTERVENTIONAL ``CausalEffect`` records of its perturbations (and the OBSERVED /
ATTRIBUTED records they rest on). These records say what was selected and what
was derived:

* :class:`EvidenceSelection`: which units of one site were selected and by what
  rule: an attribution ranking (with the scores used), a declared set, or a seeded
  random draw. It makes "what evidence was tested" explicit and re-derivable.
* :class:`ProtocolResult`: the output of a *diagnostic* protocol that does not
  decide a claim (curves, stability, counterexample summaries, paired controls,
  method diagnostics). ``params``/``criteria`` were declared before running;
  ``measurements`` are numbers derived from referenced records; ``outcomes`` are
  per-aspect checks against the declared criteria. There is never a single score.

Claim-deciding protocols (``comprehensiveness``, ``sufficiency``) produce ordinary
``ClaimTestResult`` records (ADR-012): raw effect -> protocol -> result -> assessment.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .base import BaseRecord, record_kind, register_migration
from .interventions import MetricSpec
from .values import Site, SiteIO

__all__ = [
    "AspectOutcome",
    "CheckOutcome",
    "EvidenceSelection",
    "ProtocolResult",
    "SelectionSource",
]

_TOKEN_RE = re.compile(r"^\S+$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_INPUT_LEAF_RE = re.compile(r"^args\[(0|[1-9][0-9]*)\]$")
_RULES = {
    "attribution": frozenset({"abs_desc", "signed_desc"}),
    "declared": frozenset({"declared"}),
    "random": frozenset({"uniform_random_permutation"}),
}


class SelectionSource(Enum):
    ATTRIBUTION = "attribution"
    DECLARED = "declared"
    RANDOM = "random"


@record_kind("evidence_selection", version=3)
@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceSelection(BaseRecord):
    """An ordered selection of units (last-dimension indices) of one site.

    ``order`` is a full ranking of all ``n_units`` units (attribution or random
    source) or the declared units (declared source). ``k`` is the selected prefix
    size (``None``: a ranking used for curves). Attribution selections carry the
    ``scores`` they ranked (by unit index) and the id of the source
    ``AttributionRecord`` (which may live in another trace; composition verifies it).
    Ties are broken by lower unit index.

    ``eligible`` / ``eligibility`` (ADR-053): the units the claim is about, declared by
    the caller with a name (e.g. ``"content_tokens"``). With them, a ranking orders only
    the eligible units and random controls are drawn from them; ``None`` means every unit.
    """

    REQUIRES_PROVENANCE: ClassVar[bool] = True

    site: Site
    call_index: int = 0
    sample_id: str
    source: SelectionSource
    rule: str
    order: tuple[int, ...]
    n_units: int
    k: int | None = None
    scores: tuple[float, ...] | None = None
    source_record: str | None = None
    seed: int | None = None
    target: MetricSpec | None = None
    unit_axes: tuple[int, ...] | None = None
    unit_reduction: str | None = None
    eligible: tuple[int, ...] | None = None
    eligibility: str | None = None

    @property
    def population(self) -> tuple[int, ...]:
        """The units a ranking orders and controls are drawn from (ADR-053)."""
        return tuple(range(self.n_units)) if self.eligible is None else self.eligible

    @property
    def selected(self) -> tuple[int, ...]:
        """The selected units, sorted (``order[:k]``)."""
        assert self.k is not None
        return tuple(sorted(self.order[: self.k]))

    def _validate(self) -> None:
        if self.site.module == "":
            require(
                self.site.io is SiteIO.INPUT and bool(_INPUT_LEAF_RE.match(self.site.output_path)),
                "an input selection names a positional input leaf ('args[i]')",
            )
        else:
            require(self.site.io is SiteIO.OUTPUT, "internal selections are of module OUTPUTs")
        require(self.call_index >= 0 and self.n_units >= 1, "invalid call_index / n_units")
        require(bool(_TOKEN_RE.match(self.sample_id)), "sample_id must be a non-empty token")
        require(
            self.rule in _RULES[self.source.value],
            f"rule {self.rule!r} is not a {self.source.value} rule",
        )
        require(len(set(self.order)) == len(self.order), "order has duplicate units")
        require(all(0 <= u < self.n_units for u in self.order), "order has out-of-range units")
        require(
            (self.eligible is None) == (self.eligibility is None),
            "eligible units and their eligibility name are declared together (ADR-053)",
        )
        if self.eligible is not None:
            assert self.eligibility is not None
            require(bool(_NAME_RE.match(self.eligibility)), "eligibility is a name token")
            require(
                len(self.eligible) >= 1
                and list(self.eligible) == sorted(set(self.eligible))
                and all(0 <= u < self.n_units for u in self.eligible),
                "eligible units are sorted, unique and in range",
            )
            require(
                set(self.order) <= set(self.eligible),
                "a selection orders only eligible units (ADR-053)",
            )
        if self.source is SelectionSource.DECLARED:
            require(len(self.order) >= 1, "a declared selection names at least one unit")
            require(self.k == len(self.order), "a declared selection selects all its units")
        else:
            require(
                len(self.order) == len(self.population),
                "a ranking orders every unit (every eligible unit when declared; ADR-053)",
            )
        if self.k is not None:
            require(1 <= self.k <= len(self.order), "k must be in [1, number of ranked units]")
        if self.source is SelectionSource.ATTRIBUTION:
            require(
                self.scores is not None and len(self.scores) == self.n_units,
                "an attribution selection records the score of every unit",
            )
            assert self.scores is not None
            require(all(math.isfinite(v) for v in self.scores), "scores must be finite")
            require(self.source_record is not None, "an attribution selection names its record")
            require(self.target is not None, "an attribution selection records its target")
        else:
            require(
                self.scores is None and self.source_record is None,
                "only attribution selections carry scores and a source record",
            )
        require(
            (self.seed is not None) == (self.source is SelectionSource.RANDOM),
            "exactly the random selections carry a seed",
        )
        if self.unit_axes is not None:
            require(
                len(self.unit_axes) > 0
                and all(a >= 0 for a in self.unit_axes)
                and list(self.unit_axes) == sorted(set(self.unit_axes)),
                "unit_axes must be non-negative, sorted and unique (ADR-034)",
            )
        require(
            self.unit_reduction in (None, "sum", "abs_sum", "l2"),
            f"unknown unit_reduction {self.unit_reduction!r}",
        )
        require(
            self.unit_reduction is None or self.source is SelectionSource.ATTRIBUTION,
            "only attribution selections reduce scores to units",
        )


class CheckOutcome(Enum):
    """The outcome of one declared check of a diagnostic protocol."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True, kw_only=True)
class AspectOutcome(Value):
    aspect: str
    outcome: CheckOutcome
    detail: str | None = None

    def _validate(self) -> None:
        require(bool(_NAME_RE.match(self.aspect)), f"invalid aspect name {self.aspect!r}")


@record_kind("protocol_result")
@dataclass(frozen=True, slots=True, kw_only=True)
class ProtocolResult(BaseRecord):
    """Declared diagnostic protocol output (see module docstring).

    ``samples`` are the sample ids involved (the first is the primary sample).
    Outcomes are per aspect; PASS/FAIL require a declared criterion for that aspect,
    so an undeclared check can only be NOT_APPLICABLE or INDETERMINATE.
    """

    REQUIRES_PROVENANCE: ClassVar[bool] = True

    protocol: str
    protocol_version: int
    params: JsonMap = EMPTY_JSON
    criteria: JsonMap = EMPTY_JSON
    measurements: JsonMap
    outcomes: tuple[AspectOutcome, ...] = ()
    samples: tuple[str, ...]
    target: MetricSpec | None = None

    def _validate(self) -> None:
        require(bool(_NAME_RE.match(self.protocol)), f"invalid protocol {self.protocol!r}")
        require(self.protocol_version >= 1, "protocol_version must be >= 1")
        require(len(self.samples) >= 1, "a protocol result names its samples")
        require(all(_TOKEN_RE.match(s) for s in self.samples), "sample ids must be tokens")
        aspects = [o.aspect for o in self.outcomes]
        require(len(set(aspects)) == len(aspects), "one outcome per aspect")
        for o in self.outcomes:
            if o.outcome in (CheckOutcome.PASS, CheckOutcome.FAIL):
                require(
                    o.aspect in self.criteria,
                    f"aspect {o.aspect!r} has a pass/fail outcome but no declared criterion",
                )


@register_migration("evidence_selection", 1)
def _selection_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v1 selections used last-axis vector units (ADR-034)."""
    if "unit_axes" in data or "unit_reduction" in data:
        raise ValueError("an evidence_selection v1 payload cannot contain unit axes")
    return data | {"unit_axes": None, "unit_reduction": None}


@register_migration("evidence_selection", 2)
def _selection_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """v2 selections were about every unit (ADR-053)."""
    if "eligible" in data or "eligibility" in data:
        raise ValueError("an evidence_selection v2 payload cannot declare eligibility")
    return data | {"eligible": None, "eligibility": None}
