"""Features, concept hypotheses, and concept validation (Phase 6; ADR-039..042).

Pure data. The runtime lives in :mod:`beyondnn.concepts`.

* :class:`FeatureRecord`: a structural coordinate of one site's activation (a neuron,
  a direction, or an SAE latent) with its activation rule and provenance of how it
  was obtained. A feature has no meaning (semantic status UNLABELED_FEATURE).
* :class:`GeneratedLabel`: a label produced by a model/LLM. Status GENERATED; it can
  never be evidence (``EvidenceRef`` refuses GENERATED records).
* :class:`ConceptRecord`: a semantic hypothesis (label + operational definition)
  about a feature. Always PROPOSED_CONCEPT; it stores no status.
* :class:`ConceptDataset`: the operational extension of a concept: exact sample
  identities, binary labels, and train/val/test assignment, with a scope name.
* :class:`ConceptValidation`: the derived semantic standing of one concept under a
  declared policy and scope, computed from assessment summaries; a stored status
  that does not follow from them is refused at construction. Composition re-derives
  the summaries themselves from the underlying records.
* :class:`ConceptActivation`: a feature activation on one input of a concept that
  has a VALIDATED validation (status VALIDATED_CONCEPT, ADR-009).

Nothing here is a concept score (ADR-007).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from ._canonical import EMPTY_JSON, JsonMap
from ._types import Value, require
from .base import BaseRecord, RecordRef, is_record_id, record_kind
from .status import EvidenceStatus, Relation, SemanticStatus, Verdict
from .values import Site, SiteIO, TensorRef

__all__ = [
    "AssessmentSummary",
    "ConceptActivation",
    "ConceptDataset",
    "ConceptPolicy",
    "ConceptRecord",
    "ConceptValidation",
    "FeatureBasis",
    "FeatureRecord",
    "FeatureSource",
    "GeneratedLabel",
    "LabelSource",
    "SAEIdentity",
    "derive_semantic_status",
]

_TOKEN_RE = re.compile(r"^\S+$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_POOLINGS = frozenset({"none", "mean"})
_SPLITS = ("train", "val", "test")
ENCODING_PROTOCOL = "concept_encoding"
USE_PROTOCOL = "concept_intervention"
USE_RELATIONS = frozenset({Relation.DECREASES, Relation.INCREASES})


def _retained_vector(ref: TensorRef | None, what: str) -> None:
    require(
        ref is not None
        and len(ref.shape) == 1
        and ref.shape[0] >= 1
        and ref.content_digest is not None
        and ref.storage_key is not None,
        f"{what} must be a retained 1-D tensor with a content digest",
    )


# ------------------------------------------------------------------ features


class FeatureBasis(Enum):
    NEURON = "neuron"
    DIRECTION = "direction"
    SAE = "sae"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureSource(Value):
    """How a feature was obtained (never evidence about its meaning).

    ``declared``: given by the caller. ``fit``: fitted by BeyondNN on the ``train``
    split of a concept dataset (``method`` names the fitter). ``search``: chosen among
    ``params["candidate_count"]`` candidates on the ``train`` split by ``method``.
    ``sae``: a latent of an external SAE (see :class:`SAEIdentity`).
    """

    kind: str
    method: str | None = None
    params: JsonMap = EMPTY_JSON
    dataset: str | None = None
    split: str | None = None

    def _validate(self) -> None:
        require(self.kind in ("declared", "fit", "search", "sae"), f"unknown source {self.kind}")
        if self.kind in ("fit", "search"):
            require(
                self.method is not None and bool(_NAME_RE.match(self.method)),
                f"a {self.kind} source names its method",
            )
            require(
                self.dataset is not None and is_record_id(self.dataset, "concept_dataset"),
                f"a {self.kind} source names the concept dataset it used",
            )
            require(self.split == "train", "features are fitted/searched on the train split only")
        else:
            require(
                self.dataset is None and self.split is None,
                f"a {self.kind} source uses no concept dataset",
            )
        if self.kind == "search":
            count = self.params.get("candidate_count")
            require(
                isinstance(count, int) and not isinstance(count, bool) and count >= 1,
                "a search source records its candidate_count",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SAEIdentity(Value):
    """An external sparse-autoencoder latent (adapter; ADR-041).

    Activation: ``relu(<x - b_dec, encoder> + b_enc)`` along the feature axis; the
    feature's ``direction`` is the latent's decoder row. ``checkpoint`` identifies the
    SAE (e.g. a digest or ``"hf:org/sae@rev"``); ``reconstruction`` holds summary
    statistics reported by the caller (never verified)."""

    checkpoint: str
    latent: int
    d_sae: int
    activation: str
    encoder: TensorRef
    b_dec: TensorRef
    b_enc: float
    reconstruction: JsonMap = EMPTY_JSON

    def _validate(self) -> None:
        require(bool(_TOKEN_RE.match(self.checkpoint)), "SAEIdentity.checkpoint must be a token")
        require(0 <= self.latent < self.d_sae, "SAE latent index out of range")
        require(self.activation == "relu", "only relu SAE encoders are supported (v0)")
        _retained_vector(self.encoder, "SAE encoder column")
        _retained_vector(self.b_dec, "SAE decoder bias")
        require(self.encoder.shape == self.b_dec.shape, "encoder/b_dec dimensions differ")
        require(math.isfinite(self.b_enc), "SAE encoder bias must be finite")


@record_kind("feature")
@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureRecord(BaseRecord):
    """A feature of one module-output leaf: where it lives and how it is computed.

    ``axis`` is the feature axis of the leaf (non-batch); ``pooling`` reduces the
    remaining non-batch axes (``none``: they must have size 1; ``mean``). Activation:
    neuron ``x[..., index, ...]``; direction ``<x, v̂>``; SAE see :class:`SAEIdentity`.
    ``model_state_digest`` binds fitted/searched/SAE features to the checkpoint they
    were derived on. The semantic status of a feature is UNLABELED_FEATURE.
    """

    basis: FeatureBasis
    site: Site
    call_index: int = 0
    axis: int
    pooling: str
    index: int | None = None
    direction: TensorRef | None = None
    source: FeatureSource
    model_state_digest: str | None = None
    sae: SAEIdentity | None = None

    semantic_status: ClassVar[SemanticStatus] = SemanticStatus.UNLABELED_FEATURE

    def retained_tensors(self) -> Iterator[TensorRef]:
        if self.direction is not None:
            yield self.direction
        if self.sae is not None:
            yield self.sae.encoder
            yield self.sae.b_dec

    @property
    def dimension(self) -> int | None:
        return None if self.direction is None else self.direction.shape[0]

    def _validate(self) -> None:
        require(bool(self.site.module), "features live at module outputs (not model inputs)")
        require(self.site.io is SiteIO.OUTPUT, "features live at module OUTPUT leaves")
        require(self.call_index >= 0, "call_index must be >= 0")
        require(self.axis >= 1, "the feature axis is a non-batch axis (>= 1)")
        require(self.pooling in _POOLINGS, f"pooling must be one of {sorted(_POOLINGS)}")
        if self.basis is FeatureBasis.NEURON:
            require(self.index is not None and self.index >= 0, "a neuron feature has an index")
            require(self.direction is None and self.sae is None, "a neuron has no direction/SAE")
            require(self.source.kind in ("declared", "search"), "neurons are declared or searched")
        elif self.basis is FeatureBasis.DIRECTION:
            require(self.index is None and self.sae is None, "a direction has no index/SAE")
            _retained_vector(self.direction, "a direction feature's vector")
            require(self.source.kind in ("declared", "fit"), "directions are declared or fitted")
        else:
            require(self.index is None and self.sae is not None, "an SAE feature has SAEIdentity")
            _retained_vector(self.direction, "an SAE feature's decoder direction")
            assert self.sae is not None
            assert self.direction is not None
            require(
                self.sae.encoder.shape == self.direction.shape,
                "SAE encoder/decoder dimensions differ",
            )
            require(self.source.kind in ("sae", "search"), "SAE features come from an SAE")
        if self.source.kind in ("fit", "search", "sae"):
            require(
                self.model_state_digest is not None
                and bool(_DIGEST_RE.match(self.model_state_digest)),
                "a fitted/searched/SAE feature records the model state digest it was derived on",
            )


# ------------------------------------------------------------------ labels and concepts


@record_kind("generated_label")
@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratedLabel(BaseRecord):
    """A label produced by a model or LLM. GENERATED: never evidence, never validation.

    ``generator``/``revision`` are the caller's declaration of what produced the text
    (never verified); ``prompt_digest`` optionally identifies the prompt."""

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.GENERATED

    text: str
    feature: str
    generator: str
    revision: str
    prompt_digest: str | None = None

    def _validate(self) -> None:
        require(bool(self.text.strip()), "a generated label has text")
        require(is_record_id(self.feature, "feature"), "a generated label names its feature")
        require(bool(_TOKEN_RE.match(self.generator)), "generator must be a token")
        require(bool(_TOKEN_RE.match(self.revision)), "revision must be a token")
        if self.prompt_digest is not None:
            require(bool(_DIGEST_RE.match(self.prompt_digest)), "prompt_digest must be sha256")


class LabelSource(Enum):
    USER = "user"
    DATASET = "dataset"
    GENERATED = "generated"


@record_kind("concept")
@dataclass(frozen=True, slots=True, kw_only=True)
class ConceptRecord(BaseRecord):
    """A concept hypothesis: ``label`` (+ ``definition``) proposed for ``feature``.

    Every concept record is a proposal (PROPOSED_CONCEPT); its standing is derived by
    :class:`ConceptValidation`, never stored here. A GENERATED label names its
    ``generated_label`` record."""

    label: str
    definition: str
    feature: str
    label_source: LabelSource
    generated_label: str | None = None

    semantic_status: ClassVar[SemanticStatus] = SemanticStatus.PROPOSED_CONCEPT

    def _validate(self) -> None:
        require(bool(self.label.strip()), "a concept has a label")
        require(bool(self.definition.strip()), "a concept states an operational definition")
        require(is_record_id(self.feature, "feature"), "a concept names its feature record")
        if self.label_source is LabelSource.GENERATED:
            require(
                self.generated_label is not None
                and is_record_id(self.generated_label, "generated_label"),
                "a GENERATED label source names its generated_label record",
            )
        else:
            require(self.generated_label is None, "only generated labels name a generator record")


@record_kind("concept_dataset")
@dataclass(frozen=True, slots=True, kw_only=True)
class ConceptDataset(BaseRecord):
    """The operational extension of a concept: which exact inputs (``samples``, exact
    sample identities) carry the concept (``labels``: 1/0), and how they are split.

    ``name`` is the validation scope shown wherever a status is shown; ``label_source``
    states how labels were obtained. Nothing about other data is implied."""

    name: str
    label_source: str
    samples: tuple[str, ...]
    labels: tuple[int, ...]
    splits: tuple[str, ...]

    def _validate(self) -> None:
        require(bool(self.name.strip()), "a concept dataset has a scope name")
        require(bool(self.label_source.strip()), "a concept dataset states its label source")
        n = len(self.samples)
        require(n >= 2 and len(self.labels) == n == len(self.splits), "misaligned dataset fields")
        require(len(set(self.samples)) == n, "duplicate samples in a concept dataset")
        require(all(_TOKEN_RE.match(s) for s in self.samples), "sample ids must be tokens")
        require(all(v in (0, 1) for v in self.labels), "concept labels are 0 or 1")
        require(all(s in _SPLITS for s in self.splits), f"splits must be in {_SPLITS}")

    def indices(self, split: str, label: int | None = None) -> tuple[int, ...]:
        return tuple(
            i
            for i, (s, v) in enumerate(zip(self.splits, self.labels, strict=True))
            if s == split and (label is None or v == label)
        )


# ------------------------------------------------------------------ validation


@dataclass(frozen=True, slots=True, kw_only=True)
class ConceptPolicy(Value):
    """A declared rule turning concept assessments into a semantic status.

    Fixed requirements (v0; cannot be weakened): a SUPPORTED ``concept_encoding``
    assessment of an ENCODES claim, at least ``min_use_claims`` (>= 1) use claims, each
    SUPPORTED by ``concept_intervention``, every spec with a declared control
    criterion, and a counterexample record. Optional caps on the recorded
    counterexample rates."""

    name: str
    version: int
    min_use_claims: int = 1
    max_false_positive_rate: float | None = None
    max_false_negative_rate: float | None = None

    def _validate(self) -> None:
        require(bool(_NAME_RE.match(self.name)), f"invalid policy name {self.name!r}")
        require(self.version >= 1, "policy version must be >= 1")
        require(self.min_use_claims >= 1, "a concept validation requires >= 1 use claim")
        for cap in (self.max_false_positive_rate, self.max_false_negative_rate):
            require(cap is None or 0.0 <= cap <= 1.0, "rate caps are in [0, 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class AssessmentSummary(Value):
    """What a validation relies on from one assessment (cross-trace; composition
    re-derives it from the Assessment, its results, specs and evidence)."""

    assessment_id: str
    claim_id: str
    relation: Relation
    protocol: str
    verdict: Verdict
    controls_declared: bool

    def _validate(self) -> None:
        require(is_record_id(self.assessment_id, "assessment"), "malformed assessment id")
        require(is_record_id(self.claim_id, "claim"), "malformed claim id")


def derive_semantic_status(
    policy: ConceptPolicy,
    encoding: AssessmentSummary,
    use: tuple[AssessmentSummary, ...],
    false_positive_rate: float,
    false_negative_rate: float,
) -> tuple[SemanticStatus, tuple[str, ...]]:
    """``(status, unmet requirements)``. VALIDATED_CONCEPT only if every requirement of
    the policy is met; otherwise PROPOSED_CONCEPT with the reasons listed."""
    unmet: list[str] = []
    if not (encoding.relation is Relation.ENCODES and encoding.protocol == ENCODING_PROTOCOL):
        unmet.append("encoding assessment is not an ENCODES/concept_encoding assessment")
    elif encoding.verdict is not Verdict.SUPPORTED:
        unmet.append(f"encoding claim {encoding.verdict.value}")
    if not encoding.controls_declared:
        unmet.append("encoding test declared no control criterion")
    if len(use) < policy.min_use_claims:
        unmet.append(f"{len(use)} use claim(s); the policy requires {policy.min_use_claims}")
    for u in use:
        if not (u.relation in USE_RELATIONS and u.protocol == USE_PROTOCOL):
            unmet.append(f"{u.claim_id} is not a concept_intervention use claim")
        elif u.verdict is not Verdict.SUPPORTED:
            unmet.append(f"use claim {u.claim_id} {u.verdict.value}")
        if not u.controls_declared:
            unmet.append(f"use claim {u.claim_id} declared no control criterion")
    cap = policy.max_false_positive_rate
    if cap is not None and false_positive_rate > cap:
        unmet.append(f"false-positive rate {false_positive_rate:.3g} > {cap:.3g}")
    cap = policy.max_false_negative_rate
    if cap is not None and false_negative_rate > cap:
        unmet.append(f"false-negative rate {false_negative_rate:.3g} > {cap:.3g}")
    status = SemanticStatus.PROPOSED_CONCEPT if unmet else SemanticStatus.VALIDATED_CONCEPT
    return status, tuple(unmet)


@record_kind("concept_validation")
@dataclass(frozen=True, slots=True, kw_only=True)
class ConceptValidation(BaseRecord):
    """The derived semantic standing of ``concept`` under ``policy``, within ``scope``.

    ``semantic_status`` and ``unmet`` must equal :func:`derive_semantic_status` of the
    stored summaries (checked at construction). ``counterexamples`` names the
    ``concept_counterexamples`` protocol result; ``model_state_digest`` binds the
    validation to one checkpoint. ``additional`` lists further assessments recorded
    with the validation but not required by the policy (e.g. SUFFICIENT_FOR under a
    retention intervention); they never change the status. Not evidence."""

    concept: RecordRef
    feature: RecordRef
    dataset: RecordRef
    policy: ConceptPolicy
    encoding: AssessmentSummary
    use: tuple[AssessmentSummary, ...]
    counterexamples: str
    false_positive_rate: float
    false_negative_rate: float
    model_state_digest: str
    scope: str
    semantic_status: SemanticStatus
    unmet: tuple[str, ...] = ()
    additional: tuple[AssessmentSummary, ...] = ()

    def _validate(self) -> None:
        require(self.concept.kind == "concept", "validation.concept must reference a concept")
        require(self.feature.kind == "feature", "validation.feature must reference a feature")
        require(self.dataset.kind == "concept_dataset", "validation.dataset must be a dataset")
        require(is_record_id(self.counterexamples, "protocol_result"), "malformed counterexamples")
        require(
            bool(_DIGEST_RE.match(self.model_state_digest)), "model_state_digest must be sha256"
        )
        require(bool(self.scope.strip()), "a validation states its scope")
        for rate in (self.false_positive_rate, self.false_negative_rate):
            require(0.0 <= rate <= 1.0, "rates are in [0, 1]")
        ids = [u.assessment_id for u in self.use]
        require(len(set(ids)) == len(ids), "duplicate use assessments")
        object.__setattr__(self, "use", tuple(sorted(self.use, key=lambda u: u.assessment_id)))
        object.__setattr__(
            self, "additional", tuple(sorted(self.additional, key=lambda u: u.assessment_id))
        )
        extra = [u.assessment_id for u in self.additional]
        require(
            len(set(extra)) == len(extra) and not set(extra) & set(ids),
            "additional assessments are distinct from each other and from the use claims",
        )
        status, unmet = derive_semantic_status(
            self.policy,
            self.encoding,
            self.use,
            self.false_positive_rate,
            self.false_negative_rate,
        )
        require(
            (status, unmet) == (self.semantic_status, tuple(self.unmet)),
            f"semantic status {self.semantic_status.value} does not follow from the assessments "
            f"under {self.policy.name}/v{self.policy.version} (derived {status.value})",
        )


@record_kind("concept_activation")
@dataclass(frozen=True, slots=True, kw_only=True)
class ConceptActivation(BaseRecord):
    """The value of a VALIDATED concept's feature on one input (ADR-009).

    Status VALIDATED_CONCEPT; it derives from exactly one MEASURED activation (the
    site on this input) and names the validation it relies on (cross-trace;
    composition checks that the validation is VALIDATED and matches model/site/
    feature). It does not say what the input "means"; only that this validated
    feature took this value here."""

    STATUS: ClassVar[EvidenceStatus | None] = EvidenceStatus.VALIDATED_CONCEPT

    validation: str
    concept: str
    feature: str
    sample_id: str
    value: float

    def _validate(self) -> None:
        require(is_record_id(self.validation, "concept_validation"), "malformed validation id")
        require(is_record_id(self.concept, "concept"), "malformed concept id")
        require(is_record_id(self.feature, "feature"), "malformed feature id")
        require(bool(_TOKEN_RE.match(self.sample_id)), "sample_id must be a token")
        require(math.isfinite(self.value), "a concept activation is finite")
        parents = [r for r in self.derived_from if r.kind == "activation"]
        require(
            len(parents) == 1 and len(self.derived_from) == 1,
            "a concept activation derives from exactly one MEASURED activation",
        )
