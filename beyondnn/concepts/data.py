"""Concept datasets, concept proposals, and generated labels (plan §5, §17).

* :func:`dataset`: the operational extension of a concept (exact inputs, 0/1 labels,
  train/val/test splits, a scope name).
* :func:`propose`: a concept hypothesis about a feature: always PROPOSED_CONCEPT.
* :func:`generated_label`: a label produced by a model/LLM, recorded as GENERATED
  (never evidence); it can seed a proposal, never a validation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import nn

from beyondnn.core.samples import sample_id
from beyondnn.core.trace import TraceResult
from beyondnn.provenance import collect_environment, fingerprint_model
from beyondnn.schema import (
    ConceptDataset,
    ConceptRecord,
    ExecutionContext,
    ExecutionMode,
    GeneratedLabel,
    LabelSource,
    MethodIdentity,
    ProvenanceRecord,
    SemanticStatus,
    TraceLimitation,
)

from ._core import ConceptError, copy_record, new_store

if TYPE_CHECKING:
    from .features import Feature

__all__ = [
    "Concept",
    "ConceptData",
    "GeneratedLabelResult",
    "dataset",
    "generated_label",
    "propose",
]

_GENERATOR_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


@dataclass(frozen=True, slots=True, eq=False)
class ConceptData:
    """A concept dataset record with the exact inputs it identifies."""

    record: ConceptDataset
    inputs: tuple[tuple[Any, ...], ...]
    kwargs: tuple[dict[str, Any], ...]
    store: TraceResult

    @property
    def id(self) -> str:
        return self.record.id


def dataset(
    samples: Sequence[Any],
    labels: Sequence[int],
    splits: Sequence[str],
    *,
    name: str,
    label_source: str,
    model_kwargs: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
) -> ConceptData:
    """Declare a concept's extension: ``samples`` (tensors or tuples of positional
    inputs), 0/1 ``labels``, and ``splits`` (``train``/``val``/``test``). ``name`` is
    the scope shown with any validation (e.g. ``"sst2-validation[seed 1234]"``);
    ``label_source`` states how labels were obtained. Exact sample identities are
    recorded, so every later test is bound to exactly these inputs."""
    items = [s if isinstance(s, tuple) else (s,) for s in samples]
    if isinstance(model_kwargs, dict) or model_kwargs is None:
        kwargs = [dict(model_kwargs or {}) for _ in items]
    else:
        kwargs = [dict(k) for k in model_kwargs]
    if not (len(items) == len(labels) == len(splits) == len(kwargs)):
        raise ConceptError("samples, labels, splits (and model_kwargs) must align")
    ids = [sample_id(*s, model_kwargs=k) for s, k in zip(items, kwargs, strict=True)]
    record = ConceptDataset(
        name=name,
        label_source=label_source,
        samples=tuple(ids),
        labels=tuple(int(v) for v in labels),
        splits=tuple(splits),
    )
    store = new_store()
    store._add(record)
    store._seal()
    return ConceptData(record, tuple(items), tuple(kwargs), store)


@dataclass(frozen=True, slots=True, eq=False)
class GeneratedLabelResult:
    record: GeneratedLabel
    store: TraceResult


def generated_label(
    model: nn.Module,
    feature: Feature,
    text: str,
    *,
    generator: str,
    revision: str,
    prompt_digest: str | None = None,
) -> GeneratedLabelResult:
    """Record a label produced by a model or LLM (plan §17). Status GENERATED: it is
    never evidence and never validated by existing. Its provenance names the model the
    label is *about* and, as the method, the caller-declared generator (unverified,
    ``GENERATED_LABEL_UNVERIFIED``). BeyondNN calls no generator."""
    if not _GENERATOR_RE.match(generator):
        raise ConceptError("generator must be a name token ([A-Za-z0-9_.-])")
    provenance = ProvenanceRecord(
        model=fingerprint_model(model),
        environment=collect_environment(),
        execution=ExecutionContext(
            mode=ExecutionMode.CLEAN, device="cpu", training=False, grad_enabled=False
        ),
        method=MethodIdentity(name=f"generated_label:{generator}", version=revision),
    )
    record = GeneratedLabel(
        text=text,
        feature=feature.id,
        generator=generator,
        revision=revision,
        prompt_digest=prompt_digest,
        provenance_id=provenance.id,
    )
    store = new_store()
    store._add(provenance)
    store._add(record)
    store._add(
        TraceLimitation(
            code="GENERATED_LABEL_UNVERIFIED",
            detail=f"{generator}@{revision}",
            applies_to=(record.id,),
        )
    )
    store._seal()
    return GeneratedLabelResult(record, store)


@dataclass(frozen=True, slots=True, eq=False)
class Concept:
    """A concept hypothesis (PROPOSED_CONCEPT) with its feature and, if generated, the
    GENERATED label it came from. Its standing is only ever a derived validation."""

    record: ConceptRecord
    feature: Feature
    store: TraceResult
    generated: GeneratedLabelResult | None = None

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def semantic_status(self) -> SemanticStatus:
        return SemanticStatus.PROPOSED_CONCEPT


def propose(
    feature: Feature,
    *,
    label: str | None = None,
    definition: str,
    label_source: str = "user",
    generated: GeneratedLabelResult | None = None,
) -> Concept:
    """Propose that ``feature`` carries a concept. The result is PROPOSED_CONCEPT, and
    stays so until :func:`beyondnn.concepts.validate` derives otherwise. A generated
    label is proposed with ``generated=`` (its text is the label; status GENERATED)."""
    if generated is not None:
        if generated.record.feature != feature.id:
            raise ConceptError("the generated label is about another feature")
        if label is not None and label != generated.record.text:
            raise ConceptError("a generated proposal uses the generated text as its label")
        text, source = generated.record.text, LabelSource.GENERATED
    else:
        if label is None:
            raise ConceptError("a proposal needs a label (or a generated label)")
        if label_source not in ("user", "dataset"):
            raise ConceptError(
                "label_source is 'user' or 'dataset' (generated labels use generated=)"
            )
        text, source = label, LabelSource(label_source)
    record = ConceptRecord(
        label=text,
        definition=definition,
        feature=feature.id,
        label_source=source,
        generated_label=None if generated is None else generated.record.id,
    )
    store = new_store()
    copy_record(store, feature.store, feature.record)
    if generated is not None:
        copy_record(store, generated.store, generated.record)
    store._add(record)
    store._seal()
    return Concept(record, feature, store, generated)
