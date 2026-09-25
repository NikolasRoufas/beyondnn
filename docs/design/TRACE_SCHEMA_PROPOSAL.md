# Trace Schema (schema_version 0.1)

This document has two parts:

- **Part A: Schema 0.1 as implemented.** Status: **Accepted** (ADR-012 to ADR-018; M1.1 reviewed 2026-09-25). Provenance types (§A.10, ADR-019/020) were added in M1.2 and await review. Code: `beyondnn/schema/`.
- **Part B: Proposals for later schema versions.** The original design sketches, kept verbatim for history. Where Part B differs from Part A, **Part A wins**. Part B record types enter the schema only in later versions, once their semantics are validated (ADR-014).

Relevant decisions: ADR-001 (frozen dataclasses), ADR-006/013 (interventional vs estimated causal; estimand scope), ADR-007 (no global confidence), ADR-009 (semantic states), ADR-012 (claims), ADR-014 (incremental versioning), ADR-015 (a trace is one execution), ADR-016 (identity), ADR-017 (status mechanism), ADR-018 (assessment policies).

---

## Part A: Schema 0.1 as implemented (M1.1)

### A.1 Modules

| Module | Contents |
|---|---|
| `status.py` | `EvidenceStatus`, `EstimandScope`, `Relation`, `Outcome`, `Verdict`, `CAUSAL_RELATIONS`, `CAUSAL_EVIDENCE_STATUSES`, `ALLOWED_PARENT_STATUSES`, `check_derivation` |
| `values.py` | Value types: `TensorStats`, `TensorRef`, `NamedTensor`, `Site`, `SiteIO`, `TargetSpec`, `Estimand`, `Subject`, `ClaimSource`, `ClaimSourceKind` |
| `base.py` | `BaseRecord`, `RecordRef`, `EvidenceRef`, the kind registry (`record_kind`), migrations, `verify_ref`, id computation |
| `records.py` | `InputRecord`, `OutputRecord`, `ActivationRecord` |
| `limitations.py` | `Severity`, `LimitationDef`, `LIMITATIONS`, `TraceLimitation` |
| `claims.py` | `Claim`, `ClaimTestSpec`, `ClaimTestResult`, `Assessment`, `AssessmentPolicy`, `PolicyRequirement`, the refs `ClaimRef`/`SpecRef`/`ResultRef`, `derive_verdict` |
| `codec.py` | `SCHEMA_VERSION`, `to_dict`, `from_dict`, `to_json`, `from_json` |
| `_types.py`, `_canonical.py` | Runtime type checking and field codec; canonical JSON, digests, `JsonMap` (internal) |
| `errors.py` | The `SchemaError` family |

`import beyondnn.schema` does not import `torch` (tested).

### A.2 Records and values

- **Records.** Every record is `@dataclass(frozen=True, slots=True, kw_only=True)`, a subclass of `BaseRecord`, and registered with `@record_kind(kind, version=n)`. Common fields:
  - `provenance_id: str | None`: a reference only. Provenance records and generation are M1.2.
  - `derived_from: tuple[RecordRef, ...]`: lineage.
  - `id: str`: computed. Not an init field, and excluded from equality.
- **Values.** Value types (`Value` subclasses) are frozen, slotted, keyword-only dataclasses without identity of their own.
- **Construction.** Every field is type-checked at construction against a small annotation vocabulary: `str`, `int`, `float`, `bool`, `Enum`, `X | None`, `tuple[X, ...]`, `JsonMap`, `Value`.
  - Lists are rejected where tuples are expected.
  - `bool` is rejected where `int` is expected.
  - An `int` passed for a `float` field is normalised to `float`.
  - Mappings passed for `JsonMap` fields are deep-frozen.
- **Unsupported annotations** fail when the kind is registered (at import).
- **The runtime checker is internal** (`_types.py`) and is not public API. It complements mypy rather than duplicating it: mypy protects typed development, and the runtime checker protects decoded files, untyped callers, and corrupted or malicious payloads.

| Kind | Class | Status | Provenance | Fields (besides common) |
|---|---|---|---|---|
| `input` | `InputRecord` | OBSERVED | required | `tensors: tuple[NamedTensor, ...]`, `display: str \| None` |
| `output` | `OutputRecord` | OBSERVED | required | `tensors` |
| `activation` | `ActivationRecord` | MEASURED | required | `site: Site`, `value: TensorRef`, `call_index`, `pass_index` |
| `limitation` | `TraceLimitation` | — | optional | `code`, `detail`, `applies_to: tuple[str, ...]` |
| `claim` | `Claim` | — | optional | `statement`, `relation`, `subject`, `target`, `estimand`, `source` |
| `claim_test_spec` | `ClaimTestSpec` | — | optional | `protocol`, `protocol_version`, `applicable_relations`, `criteria`, `params` |
| `claim_test_result` | `ClaimTestResult` | — | **required** | `claim: ClaimRef`, `spec: SpecRef`, `outcome`, `evidence: tuple[EvidenceRef, ...]`, `statistics`, `error` |
| `assessment` | `Assessment` | — | optional | `claim`, `policy`, `results: tuple[ResultRef, ...]`, `verdict`, `required_but_missing` |

**`TensorRef`** holds `shape`, `dtype`, `device` (the original device), optional `stats: TensorStats`, optional `storage_key`, and optional `content_digest` (`sha256:<hex>`).
- It never holds tensor data or a runtime tensor handle. The containing trace resolves `storage_key` (M1.6/M1.7).
- `stats.numel` must match `shape`.
- Non-finite stats are allowed, and are encoded as `"NaN"`, `"Infinity"`, or `"-Infinity"`.

**`Site`** holds `module` (a dotted path; `""` is the root), `io` (`INPUT`/`OUTPUT`, default `OUTPUT`), and `output_path`. Time (`call_index`, `pass_index`) lives on the record, not in the site.

**Deferred from 0.1 (ADR-014):**
- all Phase 2–6 record kinds;
- `SemanticStatus` and concept records (Phase 6; ADR-009 still governs them);
- `TargetSpec.defaulted` (arrives with `explain()` in M1.8);
- `Selector` (replaced for now by `Subject.units`);
- opaque pass-through of unknown kinds (M1.7).

### A.3 Record identity (ADR-016)

```
id = f"{kind}:{sha256(canonical_json({'kind': kind, 'record_version': v, 'data': encoded_init_fields}))[:32]}"
```

- **What contributes to identity:** every init field, including `provenance_id`, `derived_from`, `statement` (for claims), `criteria` and `params` (for specs), and all tensor *metadata* (shape, dtype, device, stats, storage key, content digest).
- **What does not:**
  - `id` itself;
  - `schema_version` (the envelope version);
  - tensor values, which are never in records. `content_digest` is the opt-in way to bind a record to exact bytes.
- **Provenance changes:** the same content measured under a different `provenance_id` is a different record. M1.2 must therefore derive provenance ids deterministically, excluding timestamps.
- **Canonical JSON:**
  - sorted keys, compact separators, UTF-8;
  - floats in shortest round-trip form;
  - `1`, `1.0`, `true`, and `"1"` are distinct;
  - non-finite floats are allowed only in typed float fields (as strings), and rejected in free-form JSON.
- **Order normalisation:** order-insensitive collections (`derived_from`, `ClaimTestResult.evidence`, `Subject.units`, `applicable_relations`, `PolicyRequirement.protocols`, `AssessmentPolicy.requirements`, `Assessment.results`, `TraceLimitation.applies_to`) are sorted at construction, so equal content yields equal ids. `-0.0` is normalised to `0.0`.
- **Identity is not semantic equivalence.** An id identifies one artifact, including its wording and provenance. Two formally equivalent claims can have different ids. Semantic equivalence must be decided from the formal claim structure available in the schema (in 0.1: subject, relation, target, estimand/scope; the direction is carried by the relation and quantitative criteria by `ClaimTestSpec`), never from `id`. Nothing in 0.1 does so. A future version may add an explicit expectation field if Phase 2 shows it is needed.
- **No Unicode normalisation.** Strings enter identity exactly as given. `"é"` (one code point) and `"e"` plus a combining accent are distinct content (ADR-016 amendment).
- **Collisions:** 128-bit truncation makes accidental collisions negligible. Containers must still reject two different records with one id (M1.6). Python's `hash()` is never used for identity. Golden ids in `tests/test_record_identity.py` pin the algorithm, and a cross-process test runs under different `PYTHONHASHSEED` values.
- **Schema migration may change record ids.** `record_version` is part of identity, so a migrated record gets a new id. M1.7 must remap references explicitly during migration, and must test it.

### A.4 Epistemic status (ADR-017)

- `EvidenceStatus` is a plain `Enum`: **not ordered** (`<` raises `TypeError`) and not a `str`.
- Status is **never supplied by callers**: it is never a constructor argument, and it cannot be reassigned. `dataclasses.replace(rec, status=…)` raises `TypeError`.
- Status is **intrinsic to a kind where the semantics guarantee it** (`STATUS`: input/output are OBSERVED, activation is MEASURED). It is **not universally fixed per kind**: kinds may derive status from content through the `status` property (e.g. a future effect record whose status follows its estimand; see the test-only `FakeEffect`).
- **Measured state under intervention ≠ intervention effect.**
  - Model state observed during an intervened execution is **MEASURED**, and the intervention is recorded in provenance / execution context (M1.2).
  - **INTERVENTIONAL** is reserved for an intervention-derived *effect*: intervened compared against baseline (e.g. 0.91 → 0.34, effect −0.57).
- **Derivation rules.** `derived_from` means "values computed from". Evidence records may only derive from evidence records whose status is allowed below. Context that *selected* what to compute (e.g. an attribution used to choose which unit to ablate) goes in provenance parameters, not lineage.

| Child status | Allowed parent statuses |
|---|---|
| OBSERVED | OBSERVED |
| MEASURED | OBSERVED, MEASURED |
| ATTRIBUTED | OBSERVED, MEASURED, ATTRIBUTED |
| INTERVENTIONAL | OBSERVED, MEASURED, INTERVENTIONAL |
| ESTIMATED_CAUSAL | OBSERVED, MEASURED, ATTRIBUTED, INTERVENTIONAL, ESTIMATED_CAUSAL |
| VALIDATED_CONCEPT | OBSERVED, MEASURED, INTERVENTIONAL, ESTIMATED_CAUSAL, VALIDATED_CONCEPT |
| GENERATED | any |
| *(no status: claims, specs, results, assessments, limitations)* | any, including GENERATED and status-less records |

Consequently, nothing except GENERATED can derive from GENERATED.

### A.5 Evidence references and the generated-evidence rule

- **`EvidenceRef(record_id, kind, status, estimand)`** is the only way to cite evidence. Its rules:
  - `status == GENERATED` raises `EvidenceRuleError`.
  - `INTERVENTIONAL`/`ESTIMATED_CAUSAL` must carry an `Estimand`.
  - `INTERVENTIONAL` with a `POPULATION` estimand raises.
  - Non-causal evidence carries no estimand in 0.1.
- `EvidenceRef.to(record)` builds a ref from a real record, and refuses status-less records (claims, specs, …).
- **TODO (re-evaluate after M1.6):** keep the five typed references for now, because they enforce invariants locally. Reconsider them only if real traces show substantial API or maintenance problems.
- **Refs carry the attributes needed for local checks.** This applies to `RecordRef`, `EvidenceRef`, `ClaimRef`, `SpecRef`, and `ResultRef`. Containers verify them against the referenced records with `verify_ref(ref, record)`: `verify_ref(ref, record)` holds exactly when `type(ref).to(record) == ref`.

### A.6 Estimand (ADR-013)

| Scope | Required | Forbidden | Meaning |
|---|---|---|---|
| `INSTANCE` | `sample_id`, `n == 1` | `aggregation`, `population` | one identified input |
| `FINITE_SAMPLE` | `sample_id`, `n ≥ 1`, `aggregation` | `population` | exactly those `n` inputs; `INTERVENTIONAL` if exact |
| `POPULATION` | `population`, `aggregation` | — (`n`, `sample_id` optional) | beyond the measured inputs; always `ESTIMATED_CAUSAL` |

- Scope is never inferred from `n`.
- `claim_estimand.covers(evidence_estimand)` requires the same scope. For `INSTANCE` and `FINITE_SAMPLE` it also requires the same `(sample_id, n, aggregation)`. For `POPULATION` it requires the same `(population, aggregation)`.

### A.7 Claims, tests, results, assessments (ADR-012, ADR-018)

- **`Claim`** has no status, verdict, or confidence field. Its standing is only ever an `Assessment`.
- **`ClaimTestSpec`:**
  - `criteria` is required and non-empty, and its keys may not overlap `params`.
  - `criteria_digest = "sha256:" + sha256(canonical_json(criteria))`.
  - The spec id covers protocol, version, relations, criteria, and params. Changing any criterion yields a different spec, and results bind to the spec id.
  - BeyondNN defines **no thresholds** in 0.1.
- **`ClaimTestResult`** invariants:
  1. `ERRORED` if and only if `error` is set.
  2. A spec whose `applicable_relations` excludes the claim's relation yields only `NOT_APPLICABLE` or `ERRORED`.
  3. `SUPPORTS`/`CONTRADICTS` must cite ≥ 1 evidence ref.
  4. For causal relations (`NECESSARY_FOR`, `SUFFICIENT_FOR`, `INCREASES`, `DECREASES`), `SUPPORTS`/`CONTRADICTS` require ≥ 1 `INTERVENTIONAL`/`ESTIMATED_CAUSAL` ref whose estimand the claim's estimand `covers`. Therefore finite-sample evidence cannot decide a population claim, and attribution or measurement cannot decide a causal claim.
- **`AssessmentPolicy(name, version, requirements)`:** each `PolicyRequirement` names protocols that must each have a `SUPPORTS` result. `derive_verdict` refuses to assess a **causal** relation under a policy that names no protocol for it. BeyondNN ships no policy in 0.1.
- **`Assessment`:** `derive(claim, results, policy)` checks that every result is about the claim. Direct construction (used by decoding) recomputes the verdict and rejects any stored `verdict`/`required_but_missing` that does not follow. There is no numeric confidence.

| Results (ignoring NOT_APPLICABLE) | Verdict |
|---|---|
| none | UNTESTED |
| SUPPORTS and CONTRADICTS | MIXED |
| CONTRADICTS only | CONTRADICTED |
| SUPPORTS, nothing contradicting, all required protocols supported | SUPPORTED |
| anything else (INCONCLUSIVE, ERRORED, required protocol missing) | INCONCLUSIVE |

**Known gap (Phase 2):** a policy names protocols, but nothing yet checks that a named protocol genuinely justifies the relation's wording (e.g. that "necessary" is backed by an ablation-style protocol). The Phase 2 protocol registry will declare which relations each protocol justifies, and policies will be validated against it.

### A.8 Limitations

| Code | Severity | Meaning | Emitted when |
|---|---|---|---|
| `FUNCTIONAL_OPS_UNOBSERVED` | info | Computation between module boundaries is not observed. | Always, by `trace()`/`recording()` (M1.6). |
| `PARTIAL_SITE_COVERAGE` | warning | Only a subset of modules was recorded. | Site patterns select fewer than all leaf modules (M1.6). |
| `NO_ATTRIBUTION` | info | No attribution method was run. | `explain()` without attribution (M1.8). |
| `NO_CAUSAL_EVIDENCE` | warning | No intervention was performed; nothing is causal evidence. | `explain()` without causal evidence (M1.8). |
| `NO_CLAIMS_TESTED` | info | No claim has a test result. | `explain()` without results (M1.8). |

- Unknown codes raise `UnknownLimitationCodeError`. The registry is read-only.
- A code is added in the milestone that first emits it. For example, `TENSORS_NOT_RETAINED` and `MODEL_IN_TRAIN_MODE` arrive in M1.6.

### A.9 Serialisation envelope and decoding (ADR-014)

```json
{"schema_version": "0.1", "kind": "activation", "record_version": 1,
 "id": "activation:bf06…", "status": "measured", "data": {"call_index": 0, "...": "..."}}
```

`from_dict` / `from_json` are strict:
- The envelope has exactly these six keys.
- `schema_version` must equal `0.1` (while below 1.0, minors may break).
- The kind must be registered, or `UnknownRecordKindError` is raised.
- A newer `record_version` raises `UnsupportedVersionError`. An older one needs a registered migration chain.
- Every field must be present, and no unknown field is allowed.
- Values must have the right JSON types.
- Construction invariants are re-run, wrapped in `DecodeError` with the original error as `__cause__`.
- The stored `id` and `status` must match the recomputed ones (`IntegrityError`).
- `NaN`/`Infinity` JSON literals are rejected.

`to_json` output is canonical. Tensor sidecars are M1.7.

### A.10 Provenance (M1.2; ADR-019, ADR-020)

Types (torch-free, in `beyondnn/schema/provenance.py`):

| Type | Fields | Notes |
|---|---|---|
| `ModelIdentity` | `model_class`, `method` (`FULL`), `algorithm_version` (1), `structure_digest`, `state_digest`, `parameter_tensors`, `parameter_elements`, `buffer_tensors`, `buffer_elements` | Counts are over distinct tensors (tied counted once), including all non-`None` buffers |
| `EnvironmentIdentity` | `python_implementation`, `python_version`, `torch_version`, `beyondnn_version`, `platform_system`, `platform_machine` | No hostname, username, paths, or hardware ids |
| `Randomness` | `declared_seed`, `rng_generator`, `rng_state_digest` | A seed and a captured state are distinct facts; at least one is required, and the generator is named iff a digest is given |
| `ExecutionMode` | `CLEAN`, `INTERVENTION` | |
| `ExecutionContext` | `mode`, `device`, `training`, `grad_enabled`, `intervention_id`, `randomness` | `intervention_id` is absent iff CLEAN |
| `MethodIdentity` | `name` (`forward_hook`, `captum:IntegratedGradients`, …), `version`, `params` (`JsonMap`) | |
| `ProvenanceRecord` (kind `provenance`) | `model`, `environment`, `execution`, `method` | Its id is the `provenance_id` of evidence. It has no `provenance_id` and no lineage of its own |
| `ExecutionOccurrence` (kind `execution_occurrence`) | `provenance_id` (required), `started_at` (UTC, `…Z`) | Occurrence identity. Timestamps live only here |

- **What `provenance_id` means:** the reproducible conditions of an execution. It is the ADR-016 hash of exactly `{model, environment, execution, method}`. Different timestamps never change it. Any change in model state or structure, environment, execution conditions (mode, intervention, device, train/eval, grad mode, randomness), or method name, version or params does change it.
- **What a model fingerprint means:** the FULL v1 algorithm in ADR-020 and the `beyondnn/provenance/fingerprint.py` docstring.
  - Tied parameters and shared modules are represented as sorted name groups.
  - Parameters and **all** registered buffers (persistent and non-persistent) are hashed bitwise, with dtype, shape, name, role, and buffer persistence.
  - Device, `requires_grad`, `None` slots, and Python source code are excluded. FULL means full *supported topology and registered tensor state*, not the identity of arbitrary Python behaviour (ADR-020 correction).
  - Unsupported tensor types raise instead of being skipped.

**Collection** (torch-dependent, in `beyondnn/provenance/`, never imported by `beyondnn.schema`):
- `fingerprint_model(model)`;
- `collect_environment()`;
- `capture_cpu_rng(declared_seed=None)`;
- `make_provenance(model_or_identity, method=…, execution=…, environment=None)`;
- `record_occurrence(provenance, started_at=None)`;
- `FingerprintError`.

---

## Part B: Proposals for later schema versions (original design, kept for history)

> These sketches predate M1.1. They are **not** part of schema 0.1. Where they conflict with Part A (for example, sequential ids like `act:0007`, `str` enums, a status field on records, `ConfidenceEstimate`, and `spec_hash` as a separate field), Part A is authoritative.

### Design rules (original proposal)

1. **Everything is a record.** Every record has `id`, `kind` (the codec discriminator, derived from the class), and `provenance_id` where it was produced by running something. Evidence-bearing records have a `status`.
2. **Records are immutable** (`@dataclass(frozen=True, slots=True)`). Derived results are new records that list their parents in `ProvenanceRecord.derived_from`.
3. **Provenance is shared, not copied.** Many records reference one `ProvenanceRecord` by id.
4. **IDs are deterministic within a trace** (`"act:0007"`, `"eff:0002"`), so repeated runs give diff-able JSON. `TraceResult.trace_id` is a uuid4.
5. **Status is set by the producer and cannot be upgraded by consumers.** The only upgrade path in the library is concept validation (`PROPOSED_CONCEPT → VALIDATED_CONCEPT | REJECTED_CONCEPT`), and it requires attached evidence.
6. **No bare floats for claims.** A number that means something carries its metric, target, and baseline.
7. **Claims carry no status.** Standing is an `Assessment` derived from test results (ADR-012).
8. **Generated content is never support.** No `supporting` / `produced` field may reference a `GeneratedText` record. The codec and `TraceResult.add()` both check this.

### Enums

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

### Core value types

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

### Provenance

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

### Measurement records (Phase 1 unless noted)

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

### Features and concepts (types Phase 1, logic Phase 6)

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

### Claim records (types Phase 1, runners Phase 2+, ADR-012)

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

### Presentation records

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

### Container

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

### Why (presentation view, Phase 4; Phase 1 minimal)

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

### Limitation code registry (initial)

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

### Serialisation format

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
