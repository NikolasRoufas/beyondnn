"""Evidence ingestion for audits (plan §7, §11-§14, §18-§19).

Builds one index over every supplied trace, checks integrity (ids match content,
references resolve, one id never carries two contents), re-derives every claim-test
result from the raw records it cites with its registered protocol, and computes each
result's assumption axes. Nothing is trusted because BeyondNN produced it, and
nothing here runs a model. Problems are recorded (:class:`Problem`), never raised,
so one bad record cannot hide the rest (plan §24).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beyondnn.core.trace import TraceResult
from beyondnn.protocols import (
    ATTRIBUTION_THRESHOLD,
    COMPREHENSIVENESS,
    CONCEPT_ENCODING,
    CONCEPT_INTERVENTION,
    DIAGNOSTIC_PROTOCOLS,
    INTERVENTION_THRESHOLD,
    PROTOCOL_VERSIONS,
    PROTOCOLS,
    SUFFICIENCY,
)
from beyondnn.schema import (
    AlternativeCriteria,
    Assessment,
    AttributionRecord,
    BaseRecord,
    CausalEffect,
    Claim,
    ClaimTestResult,
    ClaimTestSpec,
    ConceptDataset,
    ConceptValidation,
    EstimandScope,
    EvidenceSelection,
    FeatureRecord,
    InterventionRecord,
    JsonMap,
    ModelDeclaration,
    Outcome,
    ProtocolResult,
    ProvenanceRecord,
    SelectionSource,
    to_json,
)

__all__ = ["EvidenceSet", "Problem", "ResultEntry", "ValidationEntry", "collect_traces"]

#: Criteria keys that belong to the null (control) declaration, not the threshold.
CONTROL_CRITERIA = frozenset(
    {"min_fraction_below", "min_fraction_above", "min_fraction_beyond_controls"}
)
_STRATEGIES = {
    "uniform_without_replacement_same_site_same_size": "count",
    "perturbation_magnitude_stratified_same_site_same_size": "magnitude",
}
AXES = ("protocol", "threshold", "replacement", "k", "null", "method", "dataset")


@dataclass(frozen=True, slots=True)
class Problem:
    """Why a record cannot be used: ``kind`` is a FindingKind value."""

    kind: str
    code: str
    detail: str


@dataclass(eq=False, slots=True)
class ResultEntry:
    """One ClaimTestResult with everything the audit needs about it."""

    result: ClaimTestResult
    claim: Claim
    spec: ClaimTestSpec
    trace: TraceResult
    checkpoint: str | None
    declared_model: ModelDeclaration | None
    axes: dict[str, str]
    controlled: bool
    selection: EvidenceSelection | None = None
    method: str | None = None
    problem: Problem | None = None
    exclusion: Problem | None = None  # plan-relative (provenance / scope), set by the engine
    # ADR-054: the highest fraction of controls the selection could beat or match, given
    # control sets identical to the selection (which tie by construction); None: no controls
    max_control_fraction: float | None = None
    absolute_criterion_met: bool | None = None

    @property
    def id(self) -> str:
        return self.result.id

    def control_unattainable(self, criteria: Any = None) -> bool:
        """Whether the declared control criterion could not be met by any model (ADR-054)."""
        if self.max_control_fraction is None:
            return False
        crit = self.spec.criteria if criteria is None else criteria
        need = crit.get("min_fraction_below", crit.get("min_fraction_above"))
        return isinstance(need, (int, float)) and self.max_control_fraction < float(need)

    @property
    def outcome(self) -> Outcome:
        """The recorded outcome, except that a CONTRADICTS whose control criterion was
        unattainable by construction is uninformative: INCONCLUSIVE (ADR-054)."""
        if self.result.outcome is Outcome.CONTRADICTS and self.control_unattainable():
            return Outcome.INCONCLUSIVE
        return self.result.outcome

    @property
    def usable(self) -> bool:
        return self.problem is None and self.exclusion is None

    @property
    def sample(self) -> str | None:
        e = self.claim.estimand
        return e.sample_id if e.scope is EstimandScope.INSTANCE else None

    @property
    def dataset(self) -> str | None:
        value = self.spec.params.get("dataset")
        return value if isinstance(value, str) else None

    @property
    def concept(self) -> str | None:
        value = self.spec.params.get("concept")
        return value if isinstance(value, str) else None


@dataclass(eq=False, slots=True)
class ValidationEntry:
    record: ConceptValidation
    trace: TraceResult
    encoding: TraceResult | None
    uses: tuple[TraceResult, ...]
    problem: Problem | None
    feature: str  # "rederived" | "declared" | "not_rederived: ..." | "failed: ..."
    exclusion: Problem | None = None

    @property
    def usable(self) -> bool:
        return self.problem is None and self.exclusion is None


@dataclass(eq=False)
class EvidenceSet:
    traces: list[TraceResult]
    index: dict[str, tuple[BaseRecord, TraceResult]]
    failed_traces: list[tuple[int, Problem]]
    results: list[ResultEntry]
    validations: list[ValidationEntry]
    attributions: list[tuple[AttributionRecord, TraceResult]]
    diagnostics: list[tuple[ProtocolResult, bool]]
    records: int = 0
    _alternatives: dict[tuple[str, str], Outcome | None] = field(default_factory=dict)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, traces: Sequence[TraceResult]) -> EvidenceSet:
        from beyondnn.explain.bundle import _check_source

        index: dict[str, tuple[BaseRecord, TraceResult]] = {}
        failed: list[tuple[int, Problem]] = []
        included: list[TraceResult] = []
        for position, trace in enumerate(traces):
            try:
                _check_source(trace)
            except Exception as exc:
                failed.append((position, Problem("integrity_failure", "trace_integrity", str(exc))))
                continue
            conflict = next(
                (
                    r.id
                    for r in trace.records
                    if r.id in index
                    and index[r.id][0] is not r
                    and to_json(index[r.id][0]) != to_json(r)
                ),
                None,
            )
            if conflict is not None:
                failed.append(
                    (
                        position,
                        Problem(
                            "integrity_failure",
                            "id_content_conflict",
                            f"record id {conflict} carries different content in two traces",
                        ),
                    )
                )
                continue
            included.append(trace)
            for record in trace.records:
                if record.id not in index:
                    index[record.id] = (record, trace)
        ev = cls(
            traces=included,
            index=index,
            failed_traces=failed,
            results=[],
            validations=[],
            attributions=[],
            diagnostics=[],
            records=len(index),
        )
        ev._ingest()
        return ev

    def lookup(self, record_id: str) -> tuple[BaseRecord, TraceResult]:
        return self.index[record_id]

    def digest(self) -> str:
        text = "\n".join(sorted(self.index))
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest()

    # ------------------------------------------------------------------ ingest

    def _ingest(self) -> None:
        claim_ids = {rid for rid, (r, _) in self.index.items() if isinstance(r, Claim)}
        concept_checked: dict[int, Problem | None] = {}
        seen: set[str] = set()
        for trace in self.traces:
            for record in trace.records:
                if record.id in seen:
                    continue
                seen.add(record.id)
                if isinstance(record, ClaimTestResult):
                    self.results.append(self._entry(record, trace, claim_ids, concept_checked))
                elif isinstance(record, AttributionRecord):
                    self.attributions.append((record, trace))
                elif isinstance(record, ConceptValidation):
                    self.validations.append(self._validation(record, trace))
                elif isinstance(record, ProtocolResult) and record.protocol in DIAGNOSTIC_PROTOCOLS:
                    self.diagnostics.append((record, self._verify_diagnostic(record)))

    def _entry(
        self,
        result: ClaimTestResult,
        trace: TraceResult,
        claim_ids: set[str],
        concept_checked: dict[int, Problem | None],
    ) -> ResultEntry:
        claim, _ = self.index[result.claim.claim_id]
        spec, _ = self.index[result.spec.spec_id]
        assert isinstance(claim, Claim)
        assert isinstance(spec, ClaimTestSpec)
        checkpoint, declared, problem = self._provenance(result, trace)
        entry = ResultEntry(
            result=result,
            claim=claim,
            spec=spec,
            trace=trace,
            checkpoint=checkpoint,
            declared_model=declared,
            axes={},
            controlled=_controlled(spec),
            problem=problem,
        )
        protocol = spec.protocol
        if protocol in (COMPREHENSIVENESS, SUFFICIENCY):
            self._selection(entry)
            self._control_attainability(entry)
        if entry.problem is None:
            entry.problem = self._rederive(entry, claim_ids, concept_checked)
        entry.axes = self._axes(entry)
        return entry

    def _provenance(
        self, result: ClaimTestResult, trace: TraceResult
    ) -> tuple[str | None, ModelDeclaration | None, Problem | None]:
        try:
            origin = trace.origin(result)
        except Exception as exc:
            return None, None, Problem("integrity_failure", "provenance_unresolved", str(exc))
        digests = {origin.model.state_digest}
        declared = {to_json_value(origin.declared_model)}
        for ev in result.evidence:
            found = self.index.get(ev.record_id)
            if found is None:
                continue
            record = found[0]
            if record.provenance_id is None:
                continue
            p, _ = self.index.get(record.provenance_id, (None, None))
            if isinstance(p, ProvenanceRecord):
                digests.add(p.model.state_digest)
                declared.add(to_json_value(p.declared_model))
        if len(digests) > 1 or len(declared) > 1:
            return (
                origin.model.state_digest,
                origin.declared_model,
                Problem(
                    "provenance_mismatch",
                    "mixed_checkpoints",
                    f"{result.id} cites evidence from {len(digests)} checkpoints / "
                    f"{len(declared)} model declarations",
                ),
            )
        return origin.model.state_digest, origin.declared_model, None

    def _control_attainability(self, entry: ResultEntry) -> None:
        """ADR-054: control sets identical to the selected set tie with it by construction,
        so a selection can beat at most the non-identical fraction of its controls."""
        stats = entry.result.statistics
        controls = stats.get("control_effects")
        main = stats.get("selected_effect")
        if not isinstance(controls, tuple) or not controls or not isinstance(main, str):
            return
        try:
            units = [self._effect_units(str(i)) for i in (main, *controls)]
        except (KeyError, AssertionError):
            return  # unresolvable evidence is reported by re-derivation
        identical = sum(1 for u in units[1:] if u == units[0])
        entry.max_control_fraction = 1.0 - identical / len(units[1:])
        drop = stats.get("drop")
        crit = entry.spec.criteria
        if isinstance(drop, (int, float)):
            if "min_drop" in crit:
                entry.absolute_criterion_met = float(drop) >= float(crit["min_drop"])  # type: ignore[arg-type]
            elif "max_drop" in crit:
                entry.absolute_criterion_met = float(drop) <= float(crit["max_drop"])  # type: ignore[arg-type]

    def _effect_units(self, effect_id: str) -> tuple[int, ...] | None:
        effect = self.index[effect_id][0]
        assert isinstance(effect, CausalEffect)
        record = self.index[effect.interventions[0].record_id][0]
        assert isinstance(record, InterventionRecord)
        return record.units

    def _selection(self, entry: ResultEntry) -> None:
        from beyondnn.faithfulness.verify import verify_selection

        found = [
            r
            for r in entry.trace.records
            if isinstance(r, EvidenceSelection) and r.provenance_id == entry.result.provenance_id
        ]
        if len(found) != 1:
            entry.problem = Problem(
                "integrity_failure", "selection_unresolved", f"{entry.id}: {len(found)} selections"
            )
            return
        selection = found[0]
        entry.selection = selection
        declared = entry.spec.params.get("eligible")
        if (None if declared is None else tuple(declared)) != selection.eligible or (  # type: ignore[arg-type]
            entry.spec.params.get("eligibility") != selection.eligibility
        ):
            entry.problem = Problem(
                "integrity_failure",
                "eligibility_mismatch",
                f"{entry.id}: the test's declared eligible units differ from its selection's",
            )
            return
        if selection.source is SelectionSource.ATTRIBUTION:
            source = self.index.get(selection.source_record or "")
            if source is None:
                entry.problem = Problem(
                    "missing_evidence",
                    "selection_source_not_supplied",
                    f"{entry.id}: the attribution {selection.source_record} its units were "
                    "selected from was not supplied; the selection method cannot be verified",
                )
                return
            record = source[0]
            assert isinstance(record, AttributionRecord)
            entry.method = record.method.name
            try:
                verify_selection(selection, self.lookup)
            except Exception as exc:
                entry.problem = Problem("integrity_failure", "selection_rederivation", str(exc))
        else:
            entry.method = selection.source.value

    def _rederive(
        self, entry: ResultEntry, claim_ids: set[str], concept_checked: dict[int, Problem | None]
    ) -> Problem | None:
        from beyondnn.explain.bundle import _revalidate

        protocol = entry.spec.protocol
        if protocol in PROTOCOLS and entry.spec.protocol_version != PROTOCOL_VERSIONS[protocol]:
            return Problem(
                "scope_mismatch",
                "unsupported_protocol_version",
                f"{entry.id}: {protocol} v{entry.spec.protocol_version} was recorded; this "
                f"BeyondNN implements v{PROTOCOL_VERSIONS[protocol]} and cannot re-derive it",
            )
        if protocol not in PROTOCOLS:
            return Problem(
                "integrity_failure",
                "unregistered_protocol",
                f"{entry.id}: protocol {protocol!r} is not registered; it cannot be re-derived",
            )
        if protocol in (CONCEPT_ENCODING, CONCEPT_INTERVENTION):
            key = id(entry.trace)
            if key not in concept_checked:
                from beyondnn.concepts.verify import verify_encoding_trace, verify_use_trace

                verify = verify_encoding_trace if protocol == CONCEPT_ENCODING else verify_use_trace
                try:
                    verify(entry.trace)
                    concept_checked[key] = None
                except Exception as exc:
                    concept_checked[key] = Problem(
                        "integrity_failure", "rederivation_failed", f"{entry.id}: {exc}"
                    )
            return concept_checked[key]
        try:
            _revalidate(entry.result, self.index, claim_ids)
        except Exception as exc:
            return Problem("integrity_failure", "rederivation_failed", f"{entry.id}: {exc}")
        return None

    def _validation(self, record: ConceptValidation, trace: TraceResult) -> ValidationEntry:
        from beyondnn.concepts._core import ConceptError
        from beyondnn.concepts.verify import (
            ConceptVerificationError,
            needs_derivation,
            verify_feature_record,
            verify_validation_trace,
        )

        def locate(assessment_id: str) -> TraceResult:
            found = self.index.get(assessment_id)
            if found is None or not isinstance(found[0], Assessment):
                raise KeyError(assessment_id)
            return found[1]

        try:
            encoding, uses = verify_validation_trace(trace, locate)
        except Exception as exc:
            return ValidationEntry(
                record,
                trace,
                None,
                (),
                Problem("integrity_failure", "rederivation_failed", f"{record.id}: {exc}"),
                "not checked",
            )
        feature, _ = self.index[record.feature.record_id]
        data, _ = self.index[record.dataset.record_id]
        assert isinstance(feature, FeatureRecord)
        assert isinstance(data, ConceptDataset)
        if not needs_derivation(feature):
            status = "declared"
        else:
            status = "not_rederived: no supplied recording holds the train-split activations"
            failures = []
            for candidate in self._recordings(data):
                try:
                    verify_feature_record(feature, candidate, data)
                    status = "rederived"
                    break
                except ConceptVerificationError as exc:
                    failures.append(str(exc))
                except ConceptError:
                    continue
            if status != "rederived" and failures:
                status = f"failed: {failures[0]} ({len(failures)} recording(s) tried)"
        problem = None
        if status.startswith("failed"):
            problem = Problem("integrity_failure", "feature_rederivation_failed", status)
        return ValidationEntry(record, trace, encoding, uses, problem, status)

    def _recordings(self, data: ConceptDataset) -> Iterable[TraceResult]:
        """Traces that recorded every train sample of ``data`` (encoding traces do)."""
        train = {data.samples[i] for i in data.indices("train")}
        for trace in self.traces:
            ids = {i.sample_id for i in trace.inputs}
            if train <= ids:
                yield trace

    def _verify_diagnostic(self, record: ProtocolResult) -> bool:
        if record.protocol == "concept_counterexamples":
            return True  # re-derived with its encoding test (concepts.verify)
        from beyondnn.faithfulness.verify import verify_protocol_result

        try:
            verify_protocol_result(record, self.lookup)
        except Exception:
            return False
        return True

    # ------------------------------------------------------------------ axes

    def _axes(self, entry: ResultEntry) -> dict[str, str]:
        spec = entry.spec
        protocol = spec.protocol
        params = spec.params
        axes = dict.fromkeys(AXES, "-")
        axes["protocol"] = protocol
        axes["threshold"] = _key(
            {k: v for k, v in spec.criteria.to_plain().items() if k not in CONTROL_CRITERIA}
        )
        axes["null"] = "none"
        if protocol == INTERVENTION_THRESHOLD:
            axes["replacement"] = self._intervention_key(entry)
        elif protocol == ATTRIBUTION_THRESHOLD:
            axes["method"] = _key(
                {"method": params.get("method"), "baseline": params.get("baseline")}
            )
        elif protocol in (COMPREHENSIVENESS, SUFFICIENCY):
            axes["replacement"] = _replacement_key(params.get("replacement"))
            axes["k"] = str(params.get("k"))
            axes["method"] = entry.method or "-"
            axes["null"] = _faithfulness_null(params.get("controls"), spec.criteria)
        elif protocol in (CONCEPT_ENCODING, CONCEPT_INTERVENTION):
            axes["null"] = _concept_null(params.get("controls"), spec.criteria)
            axes["dataset"] = entry.dataset or "-"
            if protocol == CONCEPT_INTERVENTION:
                intervention = params.get("intervention")
                if isinstance(intervention, JsonMap):
                    axes["replacement"] = (
                        f"{intervention.get('mode')}:"
                        f"{_replacement_key(intervention.get('reference'))}"
                    )
        return axes

    def _intervention_key(self, entry: ResultEntry) -> str:
        for ev in entry.result.evidence:
            found = self.index.get(ev.record_id)
            if found is None or not isinstance(found[0], CausalEffect):
                continue
            (ref,) = found[0].interventions[:1]
            record = self.index.get(ref.record_id, (None, None))[0]
            if isinstance(record, InterventionRecord):
                key = record.operation.value
                if record.constant is not None:
                    key += f":{record.constant!r}"
                if record.value is not None:
                    key += f":{(record.value.content_digest or '')[7:23]}"
                if record.source is not None:
                    key += f":source={record.source.record_id}"
                return key
        operation = entry.spec.params.get("operation")
        return str(operation) if operation is not None else "-"

    # ------------------------------------------------------------------ alternatives

    def reevaluate(self, entry: ResultEntry, alternative: AlternativeCriteria) -> Outcome | None:
        """The outcome of ``entry`` under ``alternative`` criteria, re-evaluated from its
        recorded evidence (pure; no model). ``None``: not applicable to this result."""
        key = (entry.id, repr(alternative))
        if key in self._alternatives:
            return self._alternatives[key]
        outcome = self._reevaluate(entry, alternative)
        if outcome is Outcome.CONTRADICTS and alternative.protocol in (
            COMPREHENSIVENESS,
            SUFFICIENCY,
        ):
            criteria = entry.spec.criteria.to_plain()
            current = criteria.get(alternative.key)
            if isinstance(current, (int, float)) and not isinstance(current, bool):
                criteria[alternative.key] = (
                    float(current) * alternative.factor
                    if alternative.factor is not None
                    else alternative.value
                )
            if entry.control_unattainable(criteria):
                outcome = Outcome.INCONCLUSIVE  # ADR-054
        self._alternatives[key] = outcome
        return outcome

    def _reevaluate(self, entry: ResultEntry, alt: AlternativeCriteria) -> Outcome | None:
        spec, result, claim = entry.spec, entry.result, entry.claim
        if (
            spec.protocol != alt.protocol
            or alt.key not in spec.criteria
            or result.outcome not in (Outcome.SUPPORTS, Outcome.CONTRADICTS)
        ):
            return None
        current = spec.criteria[alt.key]
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            return None
        value = float(current) * alt.factor if alt.factor is not None else alt.value
        criteria = JsonMap(spec.criteria.to_plain() | {alt.key: value})
        new = ClaimTestSpec(
            protocol=spec.protocol,
            protocol_version=spec.protocol_version,
            applicable_relations=spec.applicable_relations,
            criteria=criteria,
            params=spec.params,
        )
        cited = [self.index[e.record_id][0] for e in result.evidence]
        pid = result.provenance_id or ""
        if spec.protocol == INTERVENTION_THRESHOLD:
            from beyondnn.interventions import evaluate_claim

            effect = next(r for r in cited if isinstance(r, CausalEffect))
            intervention = self.index[effect.interventions[0].record_id][0]
            assert isinstance(intervention, InterventionRecord)
            return evaluate_claim(claim, new, effect, intervention, provenance_id=pid).outcome
        if spec.protocol == ATTRIBUTION_THRESHOLD:
            from beyondnn.attribution import evaluate_claim as evaluate_attribution

            record = next(r for r in cited if isinstance(r, AttributionRecord))
            tensor = self.index[record.id][1].tensor(record.value)
            return evaluate_attribution(claim, new, record, tensor).outcome
        if spec.protocol in (COMPREHENSIVENESS, SUFFICIENCY):
            from beyondnn.faithfulness.claims import evaluate

            stats = result.statistics
            primary = self.index[str(stats["selected_effect"])][0]
            controls = [self.index[str(c)][0] for c in stats.get("control_effects") or ()]  # type: ignore[union-attr]
            assert isinstance(primary, CausalEffect)
            effects = [primary, *controls]
            records = {}
            for e in effects:
                assert isinstance(e, CausalEffect)
                rid = e.interventions[0].record_id
                records[rid] = self.index[rid][0]
            return evaluate(
                claim,
                new,
                primary,
                controls,  # type: ignore[arg-type]
                records,  # type: ignore[arg-type]
                no_op=bool(stats.get("no_op")),
                provenance_id=pid,
            ).outcome
        if spec.protocol == CONCEPT_INTERVENTION:
            from beyondnn.concepts.use import UseCriteria, evaluate_use

            stats = result.statistics
            primary = self.index[str(stats["primary_effect"])][0]
            assert isinstance(primary, CausalEffect)
            control_effects = [
                self.index[str(c)][0]
                for c in stats.get("control_effects") or ()  # type: ignore[union-attr]
            ]
            plain = criteria.to_plain()
            crit = UseCriteria(
                plain["min_fraction_beyond_controls"],
                plain.get("min_change"),
                plain.get("max_change"),
            )
            outcome, _ = evaluate_use(
                claim.relation,
                primary.effect,
                [c.effect for c in control_effects],  # type: ignore[attr-defined]
                crit,
            )
            return outcome
        return None


# ------------------------------------------------------------------ helpers


def to_json_value(value: Any) -> str:
    if value is None:
        return "null"
    return str(value.to_json()) if hasattr(value, "to_json") else repr(value)


def _key(value: Any) -> str:
    return json.dumps(_plainish(value), sort_keys=True, separators=(",", ":"), default=str)


def _plainish(value: Any) -> Any:
    if isinstance(value, JsonMap):
        return value.to_plain()
    if isinstance(value, Mapping):
        return {str(k): _plainish(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plainish(v) for v in value]
    return value


def _replacement_key(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "-" if value is None else str(value)
    kind = value.get("kind")
    if kind == "zero":
        return "zero"
    digest = value.get("digest")
    name = value.get("name")
    label = f"{kind}" + (f"/{name}" if isinstance(name, str) else "")
    return label + (f":{str(digest)[7:23]}" if isinstance(digest, str) else "")


def _controlled(spec: ClaimTestSpec) -> bool:
    return bool(spec.params.get("controls")) and bool(CONTROL_CRITERIA & set(spec.criteria))


def _faithfulness_null(controls: Any, criteria: JsonMap) -> str:
    if not isinstance(controls, Mapping):
        return "none"
    strategy = _STRATEGIES.get(str(controls.get("strategy")), str(controls.get("strategy")))
    fraction = [f"{k}={criteria[k]!r}" for k in sorted(CONTROL_CRITERIA) if k in criteria]
    return strategy + (f"@{','.join(fraction)}" if fraction else "@not_decisive")


def _concept_null(controls: Any, criteria: JsonMap) -> str:
    if not isinstance(controls, tuple) or not controls:
        return "none"
    kinds = sorted(
        f"{c.get('kind')}" + (f"/{c.get('distribution')}" if c.get("distribution") else "")
        for c in controls
        if isinstance(c, Mapping)
    )
    fraction = [f"{k}={criteria[k]!r}" for k in sorted(CONTROL_CRITERIA) if k in criteria]
    return "+".join(kinds) + (f"@{','.join(fraction)}" if fraction else "@not_decisive")


def collect_traces(evidence: Iterable[Any]) -> list[TraceResult]:
    """Traces from traces, saved trace paths, and Phase 3-6 result objects (their own
    and nested traces only; nothing else about an object is trusted)."""
    from beyondnn.core.persistence import load_trace

    out: list[TraceResult] = []

    def add(trace: TraceResult) -> None:
        if not any(trace is t for t in out):
            out.append(trace)

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 6 or obj is None:
            return
        if isinstance(obj, TraceResult):
            add(obj)
            return
        if isinstance(obj, (str, os.PathLike)):
            add(load_trace(Path(obj)))
            return
        found = False
        for name in ("trace", "store", "fit_trace"):
            value = getattr(obj, name, None)
            if isinstance(value, TraceResult):
                add(value)
                found = True
        for name in ("attributions", "tests", "use", "additional"):
            for item in getattr(obj, name, ()) or ():
                walk(item, depth + 1)
                found = True
        for name in ("encoding", "concept", "feature", "generated", "data"):
            value = getattr(obj, name, None)
            if value is not None and not isinstance(value, (str, int, float)):
                walk(value, depth + 1)
                found = found or value is not None
        if not found and depth == 0:
            raise TypeError(
                f"cannot audit {type(obj).__name__}: evidence must be traces, saved trace "
                "paths, or BeyondNN result objects"
            )

    for item in evidence:
        walk(item)
    return out
