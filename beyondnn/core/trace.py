"""Trace recording: ``trace()``, ``recording()``, and the ``TraceResult`` container (M1.6).

A trace is evidence from one recording context (ADR-015): per root invocation
(*pass*), an OBSERVED ``InputRecord`` and ``OutputRecord`` for the root model,
MEASURED ``ActivationRecord``s for every tensor leaf observed at the selected
sites, one ``ProvenanceRecord`` per distinct set of conditions (the model is
fingerprinted at the start of every root invocation, so state changes between
passes, e.g. BatchNorm running statistics, give later passes their own
provenance), an ``ExecutionOccurrence`` per pass (timestamps live only there),
and ``TraceLimitation``s.

Every record entering a ``TraceResult`` is validated: registered kind, id
recomputed from content, same-id records must have identical canonical content
(never compared with ``==``: NaN), every reference (``derived_from``, evidence,
claim/spec/result refs) must resolve to a record already in the trace and pass
:func:`~beyondnn.schema.verify_ref`, every ``applies_to`` id must exist, and
every ``provenance_id`` must name a ``ProvenanceRecord`` in the trace.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch import nn

from beyondnn.provenance import collect_environment, fingerprint_model, make_provenance
from beyondnn.provenance import record_occurrence as _record_occurrence
from beyondnn.schema import (
    ActivationRecord,
    BaseRecord,
    ClaimRef,
    EvidenceRef,
    ExecutionContext,
    ExecutionMode,
    ExecutionOccurrence,
    InputRecord,
    MethodIdentity,
    ModelDeclaration,
    NamedTensor,
    OutputRecord,
    ProvenanceRecord,
    Randomness,
    RecordRef,
    ResultRef,
    Site,
    SiteIO,
    SpecRef,
    TensorRef,
    TraceLimitation,
    to_json,
    verify_ref,
)
from beyondnn.schema._types import Value
from beyondnn.schema.base import _compute_id, registered_kinds

from .hooks import HookEvent, HookSession
from .sites import resolve_sites
from .tensors import RETENTIONS, tensor_ref, walk

__all__ = [
    "ActivationLookupError",
    "ExternalForwardHooksError",
    "OutOfPassExecutionError",
    "Recording",
    "RecordingError",
    "TraceConfig",
    "TraceError",
    "TraceIntegrityError",
    "TraceResult",
    "UnsupportedExecutionError",
    "recording",
    "trace",
]

#: The method identity of Mode-A hook tracing. Retention is capture configuration,
#: not part of the measurement method, and is kept in ``TraceConfig``.
TRACE_METHOD = MethodIdentity(name="forward_hook", version="1")

_REF_TARGET = {RecordRef: "record_id", EvidenceRef: "record_id", ClaimRef: "claim_id"}
_REF_TARGET.update({SpecRef: "spec_id", ResultRef: "result_id"})


class TraceError(RuntimeError):
    """Base class for tracing errors."""


class RecordingError(TraceError):
    """The recording failed or its result is not (yet) available."""


class ExternalForwardHooksError(RecordingError):
    """Forward (pre-)hooks not installed by BeyondNN are present. They can change what a
    module receives or returns without being represented in provenance, so public
    trace recording refuses to run (HookSession itself tolerates them)."""


class UnsupportedExecutionError(RecordingError):
    """The execution cannot be represented faithfully in Phase 1 (e.g. several devices)."""


class OutOfPassExecutionError(TraceError):
    """A selected module executed outside any root invocation (v0 refuses to record it)."""


class TraceIntegrityError(TraceError):
    """A record or reference failed validation on entering a trace."""


class ActivationLookupError(LookupError):
    """No record, or more than one, matched an activation lookup."""


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """How a trace was captured (not part of any record identity)."""

    sites: tuple[str, ...]
    input_sites: tuple[str, ...]
    retention: str

    def to_json(self) -> dict[str, Any]:
        return {
            "sites": list(self.sites),
            "input_sites": list(self.input_sites),
            "retention": self.retention,
        }

    @classmethod
    def from_json(cls, data: Any) -> TraceConfig:
        if not isinstance(data, dict) or set(data) != {"sites", "input_sites", "retention"}:
            raise TraceIntegrityError("invalid trace config")
        sites, input_sites, retention = data["sites"], data["input_sites"], data["retention"]
        for value in (sites, input_sites):
            if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
                raise TraceIntegrityError("invalid trace config site list")
        if retention not in RETENTIONS:
            raise TraceIntegrityError(f"invalid retention {retention!r}")
        return cls(tuple(sites), tuple(input_sites), retention)


# ------------------------------------------------------------------ container


def _refs(value: Any) -> Iterator[Value]:
    """All reference values inside a record or value (recursively)."""
    if isinstance(value, tuple):
        for item in value:
            yield from _refs(item)
    elif type(value) in _REF_TARGET:
        yield value
    elif isinstance(value, (Value, BaseRecord)) and dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            if f.init:
                yield from _refs(getattr(value, f.name))


class TraceResult:
    """Validated, execution-ordered evidence from one recording context.

    Built by :func:`recording` / :func:`trace` or :func:`~beyondnn.load_trace`;
    read-only once finalised.
    """

    def __init__(self, config: TraceConfig) -> None:
        self._config = config
        self._records: dict[str, BaseRecord] = {}
        self._canonical: dict[str, str] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        self._sealed = False

    # ------------------------------------------------------------ building (internal)

    def _add(self, record: BaseRecord) -> BaseRecord:
        """Validate and add ``record``; return the stored record (deduplicated by id)."""
        if self._sealed:
            raise TraceIntegrityError("a finalised trace cannot be modified")
        if not isinstance(record, BaseRecord) or registered_kinds().get(record.KIND) is not type(
            record
        ):
            raise TraceIntegrityError(f"not a registered record: {type(record).__name__}")
        if _compute_id(record) != record.id:
            raise TraceIntegrityError(f"record {record.id} does not match its content")
        canonical = to_json(record)
        existing = self._canonical.get(record.id)
        if existing is not None:
            if existing != canonical:
                raise TraceIntegrityError(f"conflicting content for record id {record.id}")
            return self._records[record.id]
        self._check_references(record)
        self._records[record.id] = record
        self._canonical[record.id] = canonical
        return record

    def _check_references(self, record: BaseRecord) -> None:
        for ref in _refs(record):
            target_id = getattr(ref, _REF_TARGET[type(ref)])
            target = self._records.get(target_id)
            if target is None:
                raise TraceIntegrityError(
                    f"{record.id}: {type(ref).__name__} points to {target_id}, not in trace"
                )
            try:
                verify_ref(ref, target)
            except Exception as exc:
                raise TraceIntegrityError(f"{record.id}: {exc}") from exc
        if isinstance(record, TraceLimitation):
            for target_id in record.applies_to:
                if target_id not in self._records:
                    raise TraceIntegrityError(
                        f"{record.id}: applies_to {target_id}, which is not in the trace"
                    )
        if record.provenance_id is not None and not isinstance(
            self._records.get(record.provenance_id), ProvenanceRecord
        ):
            raise TraceIntegrityError(
                f"{record.id}: provenance_id {record.provenance_id} does not name a "
                "ProvenanceRecord in this trace"
            )

    def _add_tensor(self, key: str, tensor: torch.Tensor) -> None:
        if self._sealed:
            raise TraceIntegrityError("a finalised trace cannot be modified")
        self._tensors.setdefault(key, tensor)

    def _seal(self) -> None:
        referenced = {r.storage_key for r in self._tensor_refs() if r.storage_key is not None}
        if referenced != set(self._tensors):
            raise TraceIntegrityError("retained tensors do not match tensor references")
        self._sealed = True

    def _tensor_refs(self) -> Iterator[TensorRef]:
        for record in self._records.values():
            if isinstance(record, ActivationRecord):
                yield record.value
            elif isinstance(record, (InputRecord, OutputRecord)):
                yield from (t.ref for t in record.tensors)

    # ------------------------------------------------------------ reading

    @property
    def config(self) -> TraceConfig:
        return self._config

    @property
    def records(self) -> tuple[BaseRecord, ...]:
        """All records, in the order they entered the trace (execution order)."""
        return tuple(self._records.values())

    def get(self, record_id: str) -> BaseRecord:
        try:
            return self._records[record_id]
        except KeyError:
            raise KeyError(f"no record {record_id!r} in this trace") from None

    def _of(self, cls: type[Any]) -> tuple[Any, ...]:
        return tuple(r for r in self._records.values() if isinstance(r, cls))

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return self._of(ProvenanceRecord)

    @property
    def occurrences(self) -> tuple[ExecutionOccurrence, ...]:
        return self._of(ExecutionOccurrence)

    @property
    def inputs(self) -> tuple[InputRecord, ...]:
        return self._of(InputRecord)

    @property
    def outputs(self) -> tuple[OutputRecord, ...]:
        return self._of(OutputRecord)

    @property
    def passes(self) -> int:
        return len(self.inputs)

    @property
    def input(self) -> InputRecord:
        """The root input of a single-pass trace (see ``inputs`` otherwise)."""
        record: InputRecord = self._single(self.inputs, "input")
        return record

    @property
    def output(self) -> OutputRecord:
        """The root output of a single-pass trace (see ``outputs`` otherwise)."""
        record: OutputRecord = self._single(self.outputs, "output")
        return record

    @staticmethod
    def _single(records: tuple[Any, ...], name: str) -> Any:
        if len(records) != 1:
            raise ActivationLookupError(
                f"trace has {len(records)} {name} records (one per pass); use '{name}s'"
            )
        return records[0]

    @property
    def activations(self) -> tuple[ActivationRecord, ...]:
        """MEASURED activation records in execution order."""
        return self._of(ActivationRecord)

    @property
    def limitations(self) -> tuple[TraceLimitation, ...]:
        return self._of(TraceLimitation)

    def activation(
        self,
        module: str,
        *,
        io: SiteIO = SiteIO.OUTPUT,
        output_path: str = "",
        pass_index: int | None = None,
        call_index: int | None = None,
    ) -> ActivationRecord:
        """The single activation matching the arguments; raises if none or several match."""
        site = Site(module=module, io=io, output_path=output_path)
        matches = [
            a
            for a in self.activations
            if a.site == site
            and (pass_index is None or a.pass_index == pass_index)
            and (call_index is None or a.call_index == call_index)
        ]
        if len(matches) != 1:
            found = [(a.pass_index, a.call_index) for a in matches]
            what = "no activation" if not matches else f"{len(matches)} activations"
            raise ActivationLookupError(
                f"{what} for {module!r} ({io.value}, output_path={output_path!r}, "
                f"pass={pass_index}, call={call_index}); (pass, call) found: {found}"
            )
        return matches[0]

    def origin(self, record: BaseRecord | str) -> ProvenanceRecord:
        """The provenance under which ``record`` was produced."""
        rec = self.get(record) if isinstance(record, str) else record
        if rec.provenance_id is None:
            raise KeyError(f"{rec.id} has no provenance")
        prov = self._records[rec.provenance_id]
        assert isinstance(prov, ProvenanceRecord)
        return prov

    def tensor(self, item: TensorRef | ActivationRecord) -> torch.Tensor:
        """A copy of a retained tensor (``retention="cpu"`` only)."""
        ref = item.value if isinstance(item, ActivationRecord) else item
        if ref.storage_key is None or ref.storage_key not in self._tensors:
            raise KeyError("tensor values were not retained (use retention='cpu')")
        return self._tensors[ref.storage_key].clone()

    def save(self, path: Any) -> None:
        """Save to the new directory ``path``: ``trace.json`` (+ ``tensors.pt`` if
        tensors were retained). Atomic; never overwrites. See ``load_trace``."""
        from .persistence import save_trace

        save_trace(self, path)

    def __repr__(self) -> str:
        return (
            f"TraceResult(passes={self.passes}, activations={len(self.activations)}, "
            f"records={len(self._records)}, retention={self._config.retention!r})"
        )


# ------------------------------------------------------------------ recording


@dataclass
class _Pass:
    provenance_id: str
    input_ref: RecordRef
    completed: bool = False


def _as_patterns(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of patterns, not a str")
    patterns = tuple(value)
    if "" in patterns:
        raise ValueError(
            f"{name} may not select the root (''): root inputs/outputs are always recorded "
            "as OBSERVED InputRecord/OutputRecord"
        )
    return patterns


def _device(model: nn.Module, args: Any) -> str:
    """The single execution device. Several distinct devices are refused (Phase 1
    records one device and has only verified CPU)."""
    devices = {str(t.device) for t in (*model.parameters(), *model.buffers())}
    if not devices:
        leaves, _ = walk(args)
        devices = {str(leaf.tensor.device) for leaf in leaves} or {"cpu"}
    if len(devices) > 1:
        raise UnsupportedExecutionError(
            f"model tensors are on several devices {sorted(devices)}; Phase 1 traces record "
            "a single execution device"
        )
    return next(iter(devices))


def _foreign_forward_hooks(model: nn.Module, owned: frozenset[int]) -> list[str]:
    """Where forward/forward-pre hooks not owned by BeyondNN are registered."""
    import torch.nn.modules.module as module_impl

    found: list[str] = []
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks"):
        registry = getattr(module_impl, name, None)
        if registry is None:
            raise ExternalForwardHooksError(f"cannot inspect torch global hooks ({name})")
        if len(registry):
            found.append(f"<global {name.removeprefix('_global_')}>")
    seen: set[int] = set()
    for path, module in model.named_modules(remove_duplicate=False):
        if id(module) in seen:
            continue
        seen.add(id(module))
        for kind, hooks in (
            ("forward_pre", module._forward_pre_hooks),
            ("forward", module._forward_hooks),
        ):
            if set(hooks) - owned:
                found.append(f"{path or '<root>'} ({kind})")
    return found


def _refuse_foreign_hooks(model: nn.Module, owned: frozenset[int], when: str) -> None:
    found = _foreign_forward_hooks(model, owned)
    if found:
        raise ExternalForwardHooksError(
            f"forward hooks not installed by BeyondNN are present {when}: {', '.join(found)}. "
            "They could alter module inputs/outputs without being represented in provenance; "
            "remove them before recording."
        )


class Recording:
    """Context manager behind :func:`recording`. Single use; ``result`` after a clean exit."""

    def __init__(
        self,
        model: nn.Module,
        *,
        sites: Sequence[str] = (),
        input_sites: Sequence[str] = (),
        retention: str = "summary",
        declared_model: ModelDeclaration | None = None,
        randomness: Randomness | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("recording() requires a torch.nn.Module")
        if retention not in RETENTIONS:
            raise ValueError(f"retention must be one of {RETENTIONS}, got {retention!r}")
        if declared_model is not None and not isinstance(declared_model, ModelDeclaration):
            raise TypeError("declared_model must be a ModelDeclaration")
        if randomness is not None and not isinstance(randomness, Randomness):
            raise TypeError("randomness must be a Randomness")
        self._model = model
        self._config = TraceConfig(
            _as_patterns(sites, "sites"), _as_patterns(input_sites, "input_sites"), retention
        )
        self._declared = declared_model
        self._randomness = randomness
        self._state = "new"
        self._failure = ""
        self._result: TraceResult | None = None

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> Recording:
        if self._state != "new":
            raise RecordingError("a recording context can be entered only once")
        self._state = "active"
        try:
            config = self._config
            self._selected: list[tuple[str, SiteIO]] = []
            self._bound: dict[str, nn.Module] = {}  # the module objects hooked at entry
            for patterns, io in ((config.input_sites, SiteIO.INPUT), (config.sites, SiteIO.OUTPUT)):
                if patterns:
                    for resolved in resolve_sites(self._model, patterns, io=io):
                        self._selected.append((resolved.path, io))
                        self._bound[resolved.path] = resolved.module
            self._named = [p for p, _ in self._model.named_modules(remove_duplicate=False) if p]
            _refuse_foreign_hooks(self._model, frozenset(), "before recording")
            self._environment = collect_environment()
            self._trace = TraceResult(config)
            self._passes: dict[int, _Pass] = {}
            self._executed: set[tuple[str, SiteIO]] = set()
            self._structure: str | None = None
            self._ignored = 0
            self._session = HookSession(
                self._model,
                sink=self._on_event,
                outputs=["", *config.sites],
                inputs=["", *config.input_sites],
            )
            self._session.__enter__()
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._session.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            self._fail(f"{exc_type.__name__}: {exc}")
            return
        try:
            self._finalize()
        except BaseException as error:
            self._fail(f"{type(error).__name__}: {error}")
            raise

    def _fail(self, reason: str) -> None:
        # Keep only a message: no exception/traceback (which could pin tensors).
        self._state = "failed"
        self._failure = reason
        self._trace = None  # type: ignore[assignment]
        self._passes = {}

    @property
    def result(self) -> TraceResult:
        if self._state == "done" and self._result is not None:
            return self._result
        if self._state == "failed":
            raise RecordingError(f"the recording failed and has no result ({self._failure})")
        raise RecordingError("the result is available only after the recording context exits")

    # ------------------------------------------------------------ events

    def _on_event(self, event: HookEvent) -> None:
        if event.paths == ("",):
            if event.call_index != 0:
                raise RecordingError("re-entrant root model calls are not supported in v0")
            if event.io is SiteIO.INPUT:
                self._begin_pass(event)
            else:
                self._end_pass(event)
            return
        if not event.in_root_pass:
            raise OutOfPassExecutionError(
                f"selected module {event.paths[0]!r} executed outside any root invocation "
                "(e.g. called directly); v0 traces only record executions within a root pass"
            )
        self._record_activations(event)

    def _refs_for(self, value: Any, prefix: str = "") -> list[tuple[str, TensorRef]]:
        leaves, ignored = walk(value, prefix)
        self._ignored += ignored
        store: dict[str, torch.Tensor] = {}
        refs = [
            (leaf.path, tensor_ref(leaf.tensor, self._config.retention, store)) for leaf in leaves
        ]
        for key, tensor in store.items():
            self._trace._add_tensor(key, tensor)
        return refs

    def _input_refs(self, event: HookEvent) -> list[tuple[str, TensorRef]]:
        return self._refs_for(event.args, "args") + self._refs_for(
            dict(event.kwargs or {}), "kwargs"
        )

    def _begin_pass(self, event: HookEvent) -> None:
        model = self._model
        _refuse_foreign_hooks(model, self._session.owned_hook_ids, "at the start of a root pass")
        execution = ExecutionContext(
            mode=ExecutionMode.CLEAN,
            device=_device(model, event.args),
            training=model.training,
            grad_enabled=torch.is_grad_enabled(),
            randomness=self._randomness,
        )
        self._check_bindings()
        identity = fingerprint_model(model)  # model state at the start of this root invocation
        if self._structure is None:
            self._structure = identity.structure_digest
        elif identity.structure_digest != self._structure:
            raise UnsupportedExecutionError(
                "the model's module structure changed during recording; hooks were placed on "
                "the structure at entry, so later passes could silently miss selected sites"
            )
        provenance = make_provenance(
            identity,
            method=TRACE_METHOD,
            execution=execution,
            environment=self._environment,
            declared_model=self._declared,
        )
        self._trace._add(provenance)
        self._trace._add(_record_occurrence(provenance, pass_index=event.pass_index))
        record = InputRecord(
            tensors=tuple(NamedTensor(path=p, ref=r) for p, r in self._input_refs(event)),
            pass_index=event.pass_index,
            provenance_id=provenance.id,
        )
        self._trace._add(record)
        self._passes[event.pass_index] = _Pass(provenance.id, RecordRef.to(record))

    def _check_bindings(self) -> None:
        for path, module in self._bound.items():
            try:
                current: nn.Module | None = self._model.get_submodule(path)
            except AttributeError:
                current = None
            if current is not module:
                raise UnsupportedExecutionError(
                    f"the module at selected site {path!r} was replaced or removed during "
                    "recording; its hooks are on the original object, so evidence would be "
                    "silently missing"
                )

    def _end_pass(self, event: HookEvent) -> None:
        _refuse_foreign_hooks(
            self._model, self._session.owned_hook_ids, "at the end of a root pass"
        )
        state = self._passes[event.pass_index]
        refs = self._refs_for(event.output, "output")
        record = OutputRecord(
            tensors=tuple(NamedTensor(path=p, ref=r) for p, r in refs),
            pass_index=event.pass_index,
            provenance_id=state.provenance_id,
            derived_from=(state.input_ref,),
        )
        self._trace._add(record)
        state.completed = True

    def _record_activations(self, event: HookEvent) -> None:
        state = self._passes[event.pass_index]
        site_path = event.site.module  # raises for alias groups (never path-specific-less)
        self._executed.add((site_path, event.io))
        refs = self._input_refs(event) if event.io is SiteIO.INPUT else self._refs_for(event.output)
        for leaf_path, ref in refs:
            self._trace._add(
                ActivationRecord(
                    site=Site(module=site_path, io=event.io, output_path=leaf_path),
                    value=ref,
                    call_index=event.call_index,
                    pass_index=event.pass_index,
                    provenance_id=state.provenance_id,
                    derived_from=(state.input_ref,),
                )
            )

    # ------------------------------------------------------------ finalisation

    def _finalize(self) -> None:
        if not self._passes:
            raise RecordingError("no root invocation of the model happened inside recording()")
        incomplete = sorted(p for p, s in self._passes.items() if not s.completed)
        if incomplete:
            raise RecordingError(
                f"root invocation(s) {incomplete} did not complete (the model raised); "
                "no partial trace is produced"
            )
        trace = self._trace
        trace._add(TraceLimitation(code="FUNCTIONAL_OPS_UNOBSERVED"))
        output_paths = {p for p, io in self._selected if io is SiteIO.OUTPUT}
        if not set(self._named) <= output_paths:
            trace._add(
                TraceLimitation(
                    code="PARTIAL_SITE_COVERAGE",
                    detail=f"{len(output_paths)} of {len(self._named)} named modules selected "
                    "for output",
                )
            )
        missing = [f"{p} ({io.value})" for p, io in self._selected if (p, io) not in self._executed]
        if missing:
            trace._add(
                TraceLimitation(code="SELECTED_SITE_NOT_EXECUTED", detail=", ".join(missing))
            )
        if self._ignored:
            trace._add(
                TraceLimitation(
                    code="NON_TENSOR_LEAVES_IGNORED",
                    detail=f"{self._ignored} non-tensor leaves were not recorded",
                )
            )
        trace._seal()
        self._result = trace
        self._state = "done"


def recording(
    model: nn.Module,
    *,
    sites: Sequence[str] = (),
    input_sites: Sequence[str] = (),
    retention: str = "summary",
    declared_model: ModelDeclaration | None = None,
    randomness: Randomness | None = None,
) -> Recording:
    """Record every root invocation of ``model`` inside a ``with`` block.

    ``sites`` select module OUTPUTS (patterns, see ``beyondnn.core.sites``);
    ``input_sites`` select module INPUTS. Root inputs/outputs are always recorded.
    ``retention`` is ``"summary"`` (default), ``"cpu"``, or ``"none"``.
    ``declared_model`` / ``randomness`` are caller declarations for provenance.
    After a clean exit ``ctx.result`` is the :class:`TraceResult`; if anything
    failed there is no result.
    """
    return Recording(
        model,
        sites=sites,
        input_sites=input_sites,
        retention=retention,
        declared_model=declared_model,
        randomness=randomness,
    )


def trace(
    model: nn.Module,
    *inputs: Any,
    sites: Sequence[str] = (),
    input_sites: Sequence[str] = (),
    retention: str = "summary",
    declared_model: ModelDeclaration | None = None,
    randomness: Randomness | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> TraceResult:
    """Run ``model(*inputs, **model_kwargs)`` once under :func:`recording`; return the trace.

    Exactly equivalent to a one-pass ``recording()``; the live model output is not
    returned or retained (use ``recording()`` to keep it, or ``retention="cpu"``).
    """
    with recording(
        model,
        sites=sites,
        input_sites=input_sites,
        retention=retention,
        declared_model=declared_model,
        randomness=randomness,
    ) as ctx:
        model(*inputs, **(model_kwargs or {}))
    return ctx.result
