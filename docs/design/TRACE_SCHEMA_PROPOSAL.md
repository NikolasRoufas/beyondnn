# Trace Schema Proposal (schema_version 0.1)

Status: **design, pending implementation.**

Per **ADR-014**, schema 0.1 (Phase 1) stabilises only:
- the record mechanism (identity, status, provenance reference, lineage, registry);
- the serialisation envelope;
- `InputRecord`, `OutputRecord`, `ActivationRecord`, `TensorRef`, `TraceLimitation`;
- the claim types (ADR-012).

Records for later phases that appear below are **proposals**. They arrive in later schema versions, with their own `record_version`, once their semantics are validated.

Relevant decisions:
- ADR-001: frozen dataclasses;
- ADR-006: `INTERVENTIONAL` vs `ESTIMATED_CAUSAL`;
- ADR-007: no global confidence;
- ADR-009: semantic states;
- ADR-012: claims (accepted);
- ADR-013: estimand scope;
- ADR-014: incremental schema versioning;
- ADR-015: a trace is a single execution.

## Design rules

1. **Everything is a record.** Every record has `id`, `kind` (the codec discriminator, derived from the class), and `provenance_id` where it was produced by running something. Evidence-bearing records have a `status`.
2. **Records are immutable** (`@dataclass(frozen=True, slots=True)`). Derived results are new records that list their parents in `ProvenanceRecord.derived_from`.
3. **Provenance is shared, not copied.** Many records reference one `ProvenanceRecord` by id.
4. **IDs are deterministic within a trace** (`"act:0007"`, `"eff:0002"`), so repeated runs give diff-able JSON. `TraceResult.trace_id` is a uuid4.
5. **Status is set by the producer and cannot be upgraded by consumers.** The only upgrade path in the library is concept validation (`PROPOSED_CONCEPT → VALIDATED_CONCEPT | REJECTED_CONCEPT`), and it requires attached evidence.
6. **No bare floats for claims.** A number that means something carries its metric, target, and baseline.
7. **Claims carry no status.** Standing is an `Assessment` derived from test results (ADR-012).
8. **Generated content is never support.** No `supporting` / `produced` field may reference a `GeneratedText` record. The codec and `TraceResult.add()` both check this.

## Enums

```python
class EvidenceStatus(str, Enum):
    OBSERVED          = "observed"           # model input/output, exact
    MEASURED          = "measured"           # internal value from an unmodified pass, exact
    ATTRIBUTED        = "attributed"         # method-relative score (gradient, IG, …)
    INTERVENTIONAL    = "interventional"     # exact effect of a specified intervention on given inputs (ADR-006)
    ESTIMATED_CAUSAL  = "estimated_causal"   # approximation/aggregate of an interventional quantity (ADR-006)
    VALIDATED_CONCEPT = "validated_concept"  # ConceptActivation of a VALIDATED_CONCEPT only
    GENERATED         = "generated"          # produced by a model/LLM/template; never evidence

class SemanticStatus(str, Enum):             # ADR-009
    UNLABELED_FEATURE = "unlabeled_feature"
    PROPOSED_CONCEPT  = "proposed_concept"
    VALIDATED_CONCEPT = "validated_concept"
    REJECTED_CONCEPT  = "rejected_concept"

class Severity(str, Enum):
    INFO = "info"; WARNING = "warning"; CRITICAL = "critical"

class Relation(str, Enum):                   # what a claim asserts (ADR-012)
    NECESSARY_FOR  = "necessary_for"         # intervening on subject reduces target by ≥ expectation
    SUFFICIENT_FOR = "sufficient_for"        # restoring subject (alone) into a corrupted run restores target
    INCREASES      = "increases"             # amplifying subject increases target
    DECREASES      = "decreases"             # amplifying subject decreases target
    ATTRIBUTED_TO  = "attributed_to"         # subject ranks in top-k under an attribution method (non-causal)
    ENCODES        = "encodes"               # subject linearly decodes a property (non-causal)

class Outcome(str, Enum):                    # one test on one claim
    SUPPORTS = "supports"; CONTRADICTS = "contradicts"; INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"; ERRORED = "errored"

class Verdict(str, Enum):                    # a claim's derived standing under a policy
    UNTESTED = "untested"; SUPPORTED = "supported"; CONTRADICTED = "contradicted"
    MIXED = "mixed"; INCONCLUSIVE = "inconclusive"
```

**Causal relations** are `NECESSARY_FOR`, `SUFFICIENT_FOR`, `INCREASES`, and `DECREASES`. A `ClaimTestResult` with `outcome=SUPPORTS` on a causal relation must reference at least one `CausalEffect` in `produced`. This is enforced in `__post_init__`. It is the schema-level guard against presenting correlation as causation.

## Core value types

```python
@dataclass(frozen=True, slots=True)
class TensorRef:
    shape: tuple[int, ...]
    dtype: str                          # "float32"
    device: str                         # "cpu", "cuda:0", "mps"
    stats: TensorStats | None           # None under retain="none"
    storage_key: str | None             # key into TraceResult.tensors / sidecar; None = not retained
    _tensor: torch.Tensor | None = field(default=None, compare=False, repr=False)  # runtime only; not serialised

@dataclass(frozen=True, slots=True)
class TensorStats:
    mean: float; std: float; min: float; max: float; l2: float; frac_zero: float; numel: int

@dataclass(frozen=True, slots=True)
class Site:
    """Where in the model a value lives. Time is not part of Site (see ActivationRecord)."""
    module: str                         # canonical path; "" = model root
    io: Literal["input", "output"]
    output_path: str = ""               # pytree path within tuple/dict outputs, e.g. "[0]", "['logits']"
    selector: Selector | None = None    # sub-selection (used by claims/interventions; Phase 1 type only)

@dataclass(frozen=True, slots=True)
class Selector:
    index: tuple[IndexItem, ...]        # IndexItem = int | SliceSpec | Ellipsis marker; JSON-safe
    basis_id: str | None = None         # indices in a FeatureBasis rather than raw units

@dataclass(frozen=True, slots=True)
class TargetSpec:
    metric: str                         # "logit" | "prob" | "logit_diff" | "log_prob" | "loss" | "output" | "custom:<name>"
    params: dict[str, JSON]             # {"class": 3} / {"a": 12, "b": 40} / {"token_id": 5, "position": -1}
    defaulted: bool = False             # chosen implicitly → DEFAULT_TARGET_ARGMAX limitation
    reconstructible: bool = True        # False for custom metrics (callable not serialisable)
```

## Provenance

```python
@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    id: str
    method: str                         # "forward_hook", "gradient", "zero_ablation", "claim_test:ablation/v1", …
    method_params: dict[str, JSON]
    site: Site | None
    seed: int | None
    beyondnn_version: str
    torch_version: str
    python_version: str
    device: str
    model: ModelIdentity
    created_at: str                     # ISO-8601 UTC
    derived_from: tuple[str, ...] = ()  # parent record ids

@dataclass(frozen=True, slots=True)
class ModelIdentity:
    class_name: str                     # fully qualified
    param_fingerprint: str              # "sha256:" over sorted (name, shape, dtype, bytes) of state_dict
    num_params: int
    training: bool                      # model.training at run time
    user_label: str | None = None
```

- `trace.origin(record_or_id) -> ProvenanceTree` gives the provenance record plus the transitive `derived_from` chain.
- **Open issue:** fingerprinting hashes every parameter byte. That is fine for tiny models and costly for large ones. See the Phase 1 plan, M1.2 risk.

## Measurement records (Phase 1 unless noted)

```python
@dataclass(frozen=True, slots=True)
class InputRecord:                      # status OBSERVED
    id: str; provenance_id: str
    tensors: dict[str, TensorRef]       # flattened pytree of positional+keyword inputs ("args[0]", "kwargs['mask']")
    display: str | None = None
    tokens: tuple[str, ...] | None = None

@dataclass(frozen=True, slots=True)
class OutputRecord:                     # status OBSERVED
    id: str; provenance_id: str
    tensors: dict[str, TensorRef]
    target: TargetSpec | None
    target_value: tuple[float, ...] | None    # per batch item

@dataclass(frozen=True, slots=True)
class ActivationRecord:                 # status MEASURED
    id: str; provenance_id: str
    site: Site
    value: TensorRef
    call_index: int = 0                 # n-th call of this module within the pass (shared modules)
    pass_index: int = 0                 # n-th top-level forward pass within one recording()
    forward_index: int = 0              # global hook firing order within the recording
    aliases: tuple[str, ...] = ()       # other paths to the same module object
    mutated_after_capture: bool = False # tensor._version changed before finalisation → INPLACE_MUTATION_RISK

@dataclass(frozen=True, slots=True)
class AttributionRecord:                # status ATTRIBUTED — Phase 3
    id: str; provenance_id: str
    method: str                         # "gradient" | "input_x_gradient" | "captum:IntegratedGradients"
    target: TargetSpec
    attributed_to: Site
    scores: TensorRef
    baseline: str | None
    convergence_delta: float | None = None

@dataclass(frozen=True, slots=True)
class EvidenceSpan:                     # status ATTRIBUTED (derived) — Phase 3/4
    id: str; provenance_id: str
    input_key: str
    region: tuple[int | tuple[int, int], ...]
    label: str | None
    score: float
    attribution_id: str

@dataclass(frozen=True, slots=True)
class InterventionSpec:                 # Phase 2 (type defined Phase 1)
    site: Site
    op: str                             # "zero" | "mean" | "constant" | "patch" | "scale" | "feature_delta"
    params: dict[str, JSON]             # {"ref_id": …} / {"value": …} / {"source_record_id": …} / {"k": …}

@dataclass(frozen=True, slots=True)
class InterventionRecord:               # Phase 2
    id: str; provenance_id: str
    spec: InterventionSpec
    input_id: str
    distribution_check: DistributionCheck | None   # None = unchecked → limitation

@dataclass(frozen=True, slots=True)
class DistributionCheck:
    statistic: str                      # "norm_ratio_vs_reference"
    value: float
    reference_range: tuple[float, float]
    in_range: bool

@dataclass(frozen=True, slots=True)
class CausalEffect:                     # Phase 2
    id: str; provenance_id: str
    intervention_ids: tuple[str, ...]
    target: TargetSpec
    baseline_value: float
    intervened_value: float
    effect: float                       # intervened − baseline
    relative_effect: float | None
    n_inputs: int
    estimator: str = "exact"            # "exact" | "mean_over_inputs" | "attribution_patching" | …
    ci: tuple[float, float] | None = None
    # status property: INTERVENTIONAL iff estimator == "exact", else ESTIMATED_CAUSAL (ADR-006).
    # __post_init__: estimator == "exact" requires n_inputs == 1 (under the current proposal; see open issue).

@dataclass(frozen=True, slots=True)
class CausalEdge:                       # Phase 2+ targeted paths only
    id: str; provenance_id: str
    source: Site; target: Site
    effect_id: str
    kind: Literal["total", "direct", "path"]
```

**Resolved by ADR-013.** Status depends on the estimand scope and the estimator, not on `n`:
- An exact effect on one input (`INSTANCE`), or an exact aggregate over exactly the measured inputs (`FINITE_SAMPLE`, with `n`, `aggregation`, and references to the individual effects), is `INTERVENTIONAL`.
- Anything used as a claim about a `POPULATION`, or produced by an approximate estimator, is `ESTIMATED_CAUSAL`.

The `CausalEffect` sketch above predates this rule. It will be redesigned around `Estimand` when it enters the schema in Phase 2.

## Features and concepts (types Phase 1, logic Phase 6)

```python
@dataclass(frozen=True, slots=True)
class FeatureActivation:                # status MEASURED
    id: str; provenance_id: str
    site: Site
    basis_id: str
    index: int
    value: float
    reconstruction_error: float | None

@dataclass(frozen=True, slots=True)
class Concept:
    id: str
    label: str
    status: SemanticStatus              # UNLABELED_FEATURE not valid here (a Concept has a label by definition)
    basis_id: str | None
    feature_indices: tuple[int, ...]
    label_source: Literal["human", "auto_interp", "native_declared", "probe"]
    validation_ids: tuple[str, ...] = ()   # required non-empty iff VALIDATED_CONCEPT or REJECTED_CONCEPT

@dataclass(frozen=True, slots=True)
class ValidationResult:
    id: str; provenance_id: str
    concept_id: str
    protocol: str                       # "detection+causal/v1"
    dataset_id: str
    passed: bool
    detection_result_id: str            # ClaimTestResult on an ENCODES claim
    causal_result_id: str               # ClaimTestResult on a causal claim
    statistics: dict[str, float]
    criteria: dict[str, float]

@dataclass(frozen=True, slots=True)
class ConceptActivation:
    id: str; provenance_id: str
    concept_id: str
    value: float
    status: EvidenceStatus              # VALIDATED_CONCEPT only if the concept is VALIDATED_CONCEPT, else MEASURED
```

A feature with no `Concept` referencing it is `UNLABELED_FEATURE` by definition, so there is no record to store.

`ValidationResult` reuses the claim machinery. Validating a concept means testing two claims: `ENCODES` (detection) and a causal relation. This removes a parallel validation system.

## Claim records (types Phase 1, runners Phase 2+, ADR-012)

```python
@dataclass(frozen=True, slots=True)
class Subject:
    kind: Literal["site", "feature", "concept", "input_region", "set"]
    site: Site | None = None            # for site/feature (feature = site + selector with basis_id)
    concept_id: str | None = None
    input_key: str | None = None; region: tuple | None = None
    members: tuple["Subject", ...] = () # for "set" (e.g. top-k units jointly)

@dataclass(frozen=True, slots=True)
class Scope:
    input_set_id: str                   # content hash of the input tensor(s) or a registered dataset id
    n_inputs: int
    description: str | None = None

@dataclass(frozen=True, slots=True)
class Expectation:
    direction: Literal["decrease", "increase", "any"]
    min_effect: float | None            # absolute or relative threshold
    relative: bool = False

@dataclass(frozen=True, slots=True)
class ClaimSource:
    kind: Literal["user", "method", "generated"]
    detail: str | None = None           # e.g. "top-5 of input_x_gradient", "llm:<model>"
    derived_from: tuple[str, ...] = ()  # record ids that motivated the claim (never counted as support)

@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    statement: str                      # human-readable; descriptive only, never evidence
    subject: Subject
    relation: Relation
    target: TargetSpec
    scope: Scope
    expectation: Expectation
    source: ClaimSource
    # no status field — see Assessment

@dataclass(frozen=True, slots=True)
class ClaimTestSpec:
    id: str
    kind: str                           # "ablation_necessity/v1", "random_baseline/v1", "sufficiency_patch/v1",
                                        # "comprehensiveness/v1", "stability/v1", "counterexample/v1", "detection/v1"
    params: dict[str, JSON]             # intervention op, n_random, alpha, perturbation id…
    criteria: dict[str, JSON]           # decision thresholds, declared before running
    applicable_relations: tuple[Relation, ...]
    spec_hash: str                      # sha256 of canonical (kind, params, criteria); computed in __post_init__

@dataclass(frozen=True, slots=True)
class ClaimTestResult:
    id: str; provenance_id: str
    claim_id: str
    spec_id: str
    spec_hash: str                      # must equal the spec's hash at run time
    outcome: Outcome
    evidence_status: EvidenceStatus     # strongest status among produced records
    statistics: dict[str, float]        # effect, random-baseline mean/std, z, p, jaccard, …
    produced: tuple[str, ...]           # measurement record ids created by the test
    limitation_codes: tuple[str, ...] = ()
    error: str | None = None            # for ERRORED

@dataclass(frozen=True, slots=True)
class Assessment:
    id: str
    claim_id: str
    policy: str                         # "default/v1" — versioned, documented
    verdict: Verdict
    result_ids: tuple[str, ...]
    components: dict[str, float]        # per-dimension evidence, e.g. effect_vs_random_z, stability_jaccard,
                                        # method_agreement_topk — no aggregate (ADR-007)
    required_but_missing: tuple[str, ...]   # test kinds the policy requires for this relation but were not run
    limitation_codes: tuple[str, ...] = ()
```

**Invariants (checked in `__post_init__` or in `TraceResult.add`):**
- `ClaimTestResult.outcome == SUPPORTS` with a causal `relation` requires a `CausalEffect` in `produced`, and `evidence_status ∈ {INTERVENTIONAL, ESTIMATED_CAUSAL}`.
- A spec whose `applicable_relations` excludes the claim's relation can only yield `NOT_APPLICABLE`.
- `Assessment.verdict == SUPPORTED` requires `required_but_missing == ()`, at least one `SUPPORTS`, and no `CONTRADICTS`.
- `Assessment.verdict == UNTESTED` iff `result_ids == ()`.
- `ClaimSource.derived_from` records are never counted as support by any policy.

**Assessment policy `default/v1`** (Phase 2, documented as its own page):

| Relation | Required tests | SUPPORTED when |
|---|---|---|
| `NECESSARY_FOR` | ablation_necessity, random_baseline | both SUPPORTS |
| `SUFFICIENT_FOR` | sufficiency_patch, random_baseline | both SUPPORTS |
| `INCREASES` / `DECREASES` | scaling_direction, random_baseline | both SUPPORTS |
| `ATTRIBUTED_TO` | (the attribution itself) + method_agreement | SUPPORTS; renders as "scored", never "caused" |
| `ENCODES` | detection (with counterexamples), random_direction_baseline | both SUPPORTS; renders as "decodable", never "used" |

- `CONTRADICTS` from any test (including `counterexample/v1`) gives `CONTRADICTED` if nothing supports, and `MIXED` otherwise.

## Presentation records

```python
@dataclass(frozen=True, slots=True)
class TraceLimitation:
    code: str                           # from the registry below; unknown codes rejected
    message: str
    severity: Severity
    applies_to: tuple[str, ...] = ()    # record ids; empty = whole trace

@dataclass(frozen=True, slots=True)
class GeneratedText:                    # status GENERATED — Phase 4
    id: str; provenance_id: str
    text: str
    generator: str                      # "template:why/v1" | "llm:<model>"
    supporting_records: tuple[str, ...]
```

## Container

```python
@dataclass(slots=True)
class TraceResult:                      # mutable container of immutable records
    schema_version: str                 # "0.1"
    trace_id: str
    model: ModelIdentity
    config: TraceConfig                 # sites, retention, seed, dtype policy
    input: InputRecord
    output: OutputRecord
    records: dict[str, Record]
    provenance: dict[str, ProvenanceRecord]
    limitations: list[TraceLimitation]
    tensors: TensorStore

    activations -> ActivationView
    def activation(self, module: str, *, io="output", output_path="", call_index=0, pass_index=0) -> ActivationRecord
    def tensor(self, ref: TensorRef) -> torch.Tensor
    def origin(self, record_or_id) -> ProvenanceTree
    def by_status(self, status: EvidenceStatus) -> list[Record]
    def add(self, record) -> None       # checks id uniqueness, provenance existence, cross-record invariants
    def to_json(self) -> str; @classmethod from_json(cls, s) -> TraceResult
    def save(self, path, *, tensors=True) -> None; @classmethod load(cls, path, *, model=None) -> TraceResult
```

- `records` holds claims, specs, results, and assessments as well. One trace is the unit of reproducibility.
- **Resolved by ADR-015:** a `TraceResult` holds evidence from one execution. Multi-input claims, test results, and dataset-level assessments belong in a `Study` container (Phase 2). Claim types are container-neutral.

## Why (presentation view, Phase 4; Phase 1 minimal)

```python
@dataclass(frozen=True, slots=True)
class Why:
    trace: TraceResult
    evidence: tuple[EvidenceSpan, ...]
    activations: tuple[ActivationRecord, ...]
    features: tuple[FeatureActivation, ...]
    concepts: tuple[ConceptActivation, ...]
    attributions: tuple[AttributionRecord, ...]
    interventions: tuple[InterventionRecord, ...]
    causal_effects: tuple[CausalEffect, ...]
    claims: tuple[Claim, ...]
    assessments: tuple[Assessment, ...]          # replaces "confidence" (ADR-007)
    limitations: tuple[TraceLimitation, ...]
    def render(self) -> str                       # deterministic, status-bound vocabulary
    def summary(self) -> GeneratedText            # status GENERATED; supporting_records populated

@dataclass(frozen=True, slots=True)
class Explanation:
    input: InputRecord
    why: Why
    output: OutputRecord
```

## Limitation code registry (initial)

| Code | Emitted when | Phase |
|---|---|---|
| `NO_ATTRIBUTION` | explain ran without attribution | 1 |
| `NO_CAUSAL_EVIDENCE` | no interventions were run | 1 |
| `NO_CLAIMS_TESTED` | no claim has a test result | 1 |
| `PARTIAL_SITE_COVERAGE` | only a subset of modules traced | 1 |
| `FUNCTIONAL_OPS_UNOBSERVED` | always: computation between module boundaries is not observed | 1 |
| `TENSORS_NOT_RETAINED` | summaries only | 1 |
| `MODEL_IN_TRAIN_MODE` | `model.training` was True | 1 |
| `DEFAULT_TARGET_ARGMAX` | target inferred | 1 |
| `INPLACE_MUTATION_RISK` | a captured tensor's `_version` changed before finalisation | 1 |
| `MODEL_MISMATCH` | loaded trace fingerprint ≠ given model | 1 |
| `NON_RECONSTRUCTIBLE_TARGET` | custom metric; cannot be re-run from JSON | 1 |
| `OFF_DISTRIBUTION_INTERVENTION` | distribution check out of range | 2 |
| `UNCHECKED_INTERVENTION_DISTRIBUTION` | no reference available for a check | 2 |
| `WEAK_INTERVENTION_EFFECT` | effect within random-baseline range | 2 |
| `SINGLE_INPUT_EFFECT` | causal effect measured on one input only | 2 |
| `APPROXIMATE_CAUSAL_ESTIMATE` | estimator ≠ exact | 2 |
| `REQUIRED_TESTS_MISSING` | an assessment has `required_but_missing` | 2 |
| `ATTRIBUTION_METHOD_DISAGREEMENT` | top-k overlap across methods below threshold | 3 |
| `UNSTABLE_EXPLANATION` | stability test failed | 5 |
| `UNVALIDATED_CONCEPT` | concept not `VALIDATED_CONCEPT` | 6 |
| `BASIS_RECONSTRUCTION_ERROR` | basis error above threshold | 6 |
| `GENERATED_SUMMARY` | a generated summary is present | 4 |

## Serialisation format

```json
{
  "schema_version": "0.1",
  "trace_id": "5b0c…",
  "model": {"class_name": "beyondnn.models.tiny_mlp.TinyMLP", "param_fingerprint": "sha256:…", "num_params": 354, "training": false},
  "config": {"sites": ["layers.*"], "retain": "summary", "seed": 0},
  "provenance": {"prov:0001": {"method": "forward_hook", "...": "..."}},
  "records": {
    "act:0001": {"kind": "activation", "status": "measured", "provenance_id": "prov:0001",
                 "site": {"module": "layers.0", "io": "output", "output_path": ""},
                 "call_index": 0, "pass_index": 0, "forward_index": 0,
                 "value": {"shape": [8, 16], "dtype": "float32", "device": "cpu",
                           "stats": {"mean": 0.12, "...": 0}, "storage_key": null}}
  },
  "limitations": [{"code": "FUNCTIONAL_OPS_UNOBSERVED", "severity": "info", "message": "…", "applies_to": []}]
}
```

- Unknown `kind`s from a newer minor schema are preserved as opaque dicts with a warning.
- Float serialisation uses `repr` round-trip. NaN and Inf are encoded as strings (`"NaN"`, `"Infinity"`), because strict JSON forbids them.
- Schema versioning: minor = additive, major = breaking.
