"""The Phase-1 INPUT -> WHY -> OUTPUT response (M1.8): a presentation view over a trace.

Phase-1 WHY answers "what internal evidence was measured while this output was
produced?" It does NOT answer "which internal state caused the output?": no
attribution, intervention, causal test, feature or concept interpretation has
run, and none is implied. Activations are never ranked or called important,
supporting, causal, decisive, or reasons.

The :class:`~beyondnn.core.trace.TraceResult` is the scientific source of truth:
``ExplainResponse`` and ``Why`` hold a reference to it and expose its own record
objects; nothing is copied. They persist through the trace (``trace.save`` /
``load_trace`` then :meth:`ExplainResponse.from_trace`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from beyondnn.core.trace import TraceResult
from beyondnn.schema import (
    CAUSAL_EVIDENCE_STATUSES,
    ActivationRecord,
    AttributionRecord,
    ClaimTestResult,
    EvidenceStatus,
    InputRecord,
    NamedTensor,
    OutputRecord,
    ProvenanceRecord,
    TensorRef,
    TraceLimitation,
)

__all__ = ["EXPLANATION_LIMITATIONS", "ExplainResponse", "Why"]

#: Emitted by an explanation whose trace lacks that kind of evidence (always, for
#: ``explain()``, which runs none of these methods).
EXPLANATION_LIMITATIONS = ("NO_ATTRIBUTION", "NO_CAUSAL_EVIDENCE", "NO_CLAIMS_TESTED")


@dataclass(frozen=True, slots=True, eq=False)
class Why:
    """Structured Phase-1 WHY: measured internal evidence and its limitations.

    Answers: "what internal evidence was measured while this output was produced?"
    Does NOT answer: "which internal state caused the output?"
    """

    QUESTION: ClassVar[str] = "What internal evidence was measured while this output was produced?"
    NOT_ANSWERED: ClassVar[str] = "Which internal state caused the output? (no causal evidence yet)"

    trace: TraceResult

    @property
    def activations(self) -> tuple[ActivationRecord, ...]:
        """MEASURED activation records, in execution order (the trace's own objects)."""
        return self.trace.activations

    @property
    def attributions(self) -> tuple[AttributionRecord, ...]:
        """ATTRIBUTED records present in the trace (only if an attribution was explicitly
        run, e.g. ``ExplainResponse.from_trace(bnn.attribute(...).trace)``; ``explain()``
        never runs one). Method-relative scores, not causes."""
        return tuple(r for r in self.trace.records if isinstance(r, AttributionRecord))

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        """The trace's limitations, then the explanation-level ones that apply (each is
        emitted only when its kind of evidence is absent from the trace)."""
        own = self.trace.limitations
        codes = {lim.code for lim in own}
        statuses = self.evidence_statuses
        present = {
            "NO_ATTRIBUTION": bool(self.attributions),
            "NO_CAUSAL_EVIDENCE": bool(statuses & CAUSAL_EVIDENCE_STATUSES),
            "NO_CLAIMS_TESTED": any(isinstance(r, ClaimTestResult) for r in self.trace.records),
        }
        extra = tuple(
            TraceLimitation(code=c)
            for c in EXPLANATION_LIMITATIONS
            if c not in codes and not present[c]
        )
        return own + extra

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return self.trace.provenance

    @property
    def evidence_statuses(self) -> frozenset[EvidenceStatus]:
        """Evidence statuses present in the trace (Phase 1: OBSERVED and MEASURED)."""
        return frozenset(r.status for r in self.trace.records if r.status is not None)


def _describe(ref: TensorRef) -> str:
    text = f"{ref.dtype} {tuple(ref.shape)} on {ref.device}"
    if ref.stats is not None:
        s = ref.stats
        text += f"  mean={s.mean:.6g} std={s.std:.6g} min={s.min:.6g} max={s.max:.6g}"
    return text


def _tensors(tensors: tuple[NamedTensor, ...]) -> list[str]:
    return [f"  {t.path}: {_describe(t.ref)}" for t in tensors] or ["  (no tensor leaves)"]


@dataclass(frozen=True, slots=True, eq=False)
class ExplainResponse:
    """INPUT -> WHY -> OUTPUT for one root invocation, backed by a single-pass trace."""

    trace: TraceResult

    def __post_init__(self) -> None:
        if not isinstance(self.trace, TraceResult):
            raise TypeError("ExplainResponse requires a TraceResult")
        if self.trace.passes != 1:
            raise ValueError(
                f"an explanation covers exactly one root invocation; this trace has "
                f"{self.trace.passes} passes"
            )

    @classmethod
    def from_trace(cls, trace: TraceResult) -> ExplainResponse:
        """Rebuild the response from a (e.g. loaded) single-pass trace."""
        return cls(trace)

    @property
    def input(self) -> InputRecord:
        return self.trace.input

    @property
    def output(self) -> OutputRecord:
        return self.trace.output

    @property
    def why(self) -> Why:
        return Why(self.trace)

    def render(self) -> str:
        """Deterministic plain-text view derived only from the structured records.

        Not generated prose and not reasoning: every line restates a record.
        """
        why = self.why
        sites = sorted({(a.site.module, a.site.io.value) for a in why.activations})
        lines = ["INPUT  [observed]", *_tensors(self.input.tensors), ""]
        lines.append("WHY  [measured internal evidence; not a causal or attributed explanation]")
        lines.append(f"  answers: {Why.QUESTION}")
        lines.append(f"  does not answer: {Why.NOT_ANSWERED}")
        lines.append(
            f"  {len(why.activations)} measured activation records at {len(sites)} site(s)"
        )
        for a in why.activations:
            leaf = f" {a.site.output_path}" if a.site.output_path else ""
            where = f"{a.site.module} {a.site.io.value}{leaf}"
            lines.append(
                f"    {where}  pass {a.pass_index} call {a.call_index}: {_describe(a.value)}"
            )
        statuses = ", ".join(sorted(s.value for s in why.evidence_statuses))
        lines.append(f"  evidence statuses present: {statuses}")
        lines.append("  limitations:")
        for lim in why.limitations:
            detail = f" [{lim.detail}]" if lim.detail else ""
            lines.append(
                f"    - {lim.code} ({lim.severity.value}): {lim.definition.meaning}{detail}"
            )
        lines += ["", "OUTPUT  [observed]", *_tensors(self.output.tensors)]
        return "\n".join(lines)
