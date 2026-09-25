# Architecture Decision Records

**Rules:**
- This file is append-only.
- Never edit the substance of an accepted decision. To change one, append a new ADR that says `Supersedes ADR-NNN`, and set the old ADR's status line to `Superseded by ADR-MMM`. The status line is the only permitted edit to an old record.
- Status values: `Proposed`, `Accepted`, `Superseded by ADR-NNN`, `Rejected`.

---

## ADR-001: Frozen stdlib dataclasses for the trace schema

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- All schema records are `@dataclass(frozen=True, slots=True)`.
- Serialisation is a hand-written codec (`to_dict` / `from_dict`) keyed by a `kind` discriminator and a `schema_version`.
- Invariants are enforced in `__post_init__`.

**Reason:**
- The base install stays `torch` + stdlib.
- Research environments often pin Pydantic versions indirectly, through vLLM, TransformerLens, SAELens and web stacks, so adding it invites conflicts.
- Tensor-typed fields are awkward in Pydantic.
- Immutability supports the rule that consumers can never upgrade a record's epistemic status.

**Alternatives considered:**
- *Pydantic v2*: free validation and JSON Schema, but a heavy dependency, version-conflict risk, and awkward tensor fields.
- *attrs*: good, but an extra dependency for little gain over dataclasses on Python ≥ 3.10.
- *msgspec*: fast, but niche, and one more dependency.
- *Protobuf / JSON Schema first*: gives cross-language interop, but has high ceremony too early. It can be generated later from the dataclasses.

**Consequences:**
- We own the codec and must test round-trips exhaustively.
- Validation is only as good as our `__post_init__` checks.
- A JSON Schema, if wanted, must be generated.
- `slots=True` forbids ad-hoc attributes. Records needing a runtime-only field (e.g. the in-memory tensor on `TensorRef`) declare it with `field(compare=False, repr=False)` and exclude it in the codec.

---

## ADR-002: `instrument()` returns a pass-through handle, not a wrapping `nn.Module`

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- Free functions (`bnn.trace`, `bnn.recording`, later `bnn.intervene`, `bnn.explain`) are the primitives.
- `bnn.instrument(model)` returns an `Instrumented` handle. It is not an `nn.Module`. It:
  - forwards `__call__` and attribute access to the wrapped model;
  - exposes the model as `.module`;
  - offers `trace` / `recording` / `explain` methods that call the free functions.
- The model object is never modified. It has no hooks at rest, `__class__` is unchanged, and `state_dict` keys are unchanged.

**Reason:** users keep training, saving and loading the plain model with zero behavioural change. All instrumentation is visibly scoped.

**Alternatives considered:**
- *Wrapping `nn.Module`*: adds a `model.` prefix to `state_dict` keys. Fixing that with state-dict hooks is fragile.
- *Swapping `model.__class__` to a dynamic mixin subclass*: preserves `isinstance`, but it mutates the user's object, surprises pickling and `torch.compile`, and is "magic".
- *Free functions only*: the simplest option, and kept as the primitive layer. The handle is sugar on top.

**Consequences:**
- `isinstance(ix, nn.Module)` is `False`. Code that type-checks for `nn.Module` must use `ix.module`.
- `ix.parameters()`, `ix.state_dict()`, `ix.train()`, `ix.to()` work through delegation.
- `ix.to(device)` returns the underlying module, not the handle, unless we special-case it. Special-casing `to`/`train`/`eval`/`cpu`/`cuda` to return the handle is part of Milestone 1.8.

---

## ADR-003: `trace()` for eager one-shot, `recording()` for the context form

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- `bnn.trace(model, *args, **kwargs_and_options) -> TraceResult` runs one forward pass and returns the finished trace.
- `with bnn.recording(model, sites=[...]) as ctx:` records every forward pass that happens inside the block. `ctx.result` is finalised on exit.

**Reason:**
- One name per meaning. An overloaded `trace()` (eager when given inputs, a context manager otherwise) is a known source of confusion.
- `recording()` covers code that calls the model itself: loss computation, custom loops, `generate()`.

**Alternatives considered:**
- `with bnn.trace(model) as t:` (overloaded), `bnn.tracing(...)`, and nnsight-style deferred `model.trace(x)`.

**Consequences:**
- Option handling and site selection must be shared: `trace()` is implemented on top of `recording()`.
- `recording()` can observe several forward passes, so records carry a `pass_index`.

---

## ADR-004: License is Apache-2.0

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:** Apache License 2.0.

**Reason:**
- It is permissive, includes an explicit patent grant, and is compatible with the ecosystem: PyTorch is BSD-3, and Captum and TransformerLens are permissive.
- It is acceptable to industrial and academic users.

**Alternatives considered:** MIT (no patent grant), BSD-3, and copyleft licenses (would hinder adoption as a library).

**Consequences:** the canonical `LICENSE` text sits at the repo root. A `NOTICE` file is only needed if we incorporate third-party Apache-licensed code that requires one.

---

## ADR-005: Positioning as an evidence and claim-testing framework

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:** BeyondNN is *an interpretability evidence framework for PyTorch that turns claims about neural-network computation into structured, provenance-aware, testable objects.* It is not positioned as a replacement for Captum, nnsight, TransformerLens, or SAELens.

**Reason:**
- The ecosystem audit (`docs/research/ECOSYSTEM_AUDIT.md`) found that tracing, interventions, attribution and SAE features are commodity capabilities.
- No existing tool records epistemic status or provenance, or models claims and their tests.

**Alternatives considered:**
- *"Beyond black-box networks" as a general interpretability framework*: rejected as undifferentiated and overclaiming.
- *A new tracing/intervention engine competing with nnsight*: rejected as duplication.

**Consequences:**
- The internal hook engine stays deliberately minimal, targeting under about 1k LOC.
- Adapters that convert other tools' outputs into BeyondNN records are core to adoption, not marketing.
- Scientific discipline is part of the API: status can't be upgraded, limitations are mandatory, and generated text is never evidence. It is not just documentation.

---

## ADR-006: Separate `INTERVENTIONAL` and `ESTIMATED_CAUSAL` evidence statuses

- **Date:** 2026-09-24
- **Status:** Accepted. The status-derivation consequence is refined by ADR-013.

**Decision:** `EvidenceStatus` has both:
- `INTERVENTIONAL`: the directly measured metric change from a specified intervention on specified inputs.
- `ESTIMATED_CAUSAL`: approximations of interventional quantities (e.g. attribution patching), or statistical aggregates over sampled inputs that carry sampling uncertainty.

**Reason:** a single-input ablation result is an exact do-operation on a deterministic program, while a gradient linearisation of it is not. The two differ substantially in reliability and should not share a label.

**Alternatives considered:** one `ESTIMATED_CAUSAL` label with an `estimator` field. Rejected because consumers filtering by status would conflate the two.

**Consequences:**
- `CausalEffect.status` is derived from `estimator`: `"exact"` gives `INTERVENTIONAL`, anything else gives `ESTIMATED_CAUSAL`. It cannot be set independently.
- `INTERVENTIONAL` does **not** mean "the component represents X", or that the effect generalises. The limitations `SINGLE_INPUT_EFFECT` and `OFF_DISTRIBUTION_INTERVENTION` exist for that reason.

---

## ADR-007: No global explanation-confidence score in v0

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- BeyondNN does not emit a single scalar "explanation confidence".
- It exposes component-level evidence per claim: intervention effect vs random baseline, stability, method agreement, concept validation evidence, and coverage limitations.
- `Assessment` (see the schema) carries a categorical verdict plus components. Any aggregate field is `None` unless tied to a registered, validated definition.

**Reason:**
- "Confidence" is undefined without a proposition, a calibration target, and a validation study.
- A number without a definition is exactly the kind of false precision BeyondNN exists to prevent.

**Alternatives considered:**
- A weighted average of components: arbitrary weights, uncalibrated.
- A learned calibrator: needs ground truth that only synthetic models provide, and generalisation to real models is unknown.

**Consequences:**
- Renders show components, not percentages.
- Research Question 8 (`docs/research/RESEARCH_QUESTIONS.md`) defines what evidence would justify introducing an aggregate later. That requires a superseding ADR.

---

## ADR-008: Mode B (`InterpretableModule`) deferred; extension point preserved

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- No native interpretable-module API is built during Phase 1, or before the tracing, evidence, intervention and claim-testing layers are mature.
- The extension point is preserved. `Site` resolution will consult an optional `__bnn_sites__` declaration on modules, so native modules can later declare exact intervention sites without a schema change.

**Reason:**
- Concept-bottleneck leakage research shows that declaring a unit "concept X" guarantees neither meaning nor exclusive use.
- The only distinctive Mode B guarantee is exact, declared intervention sites. That is valuable but not urgent.

**Alternatives considered:** building Mode A and Mode B in parallel. Rejected, because it doubles the surface before the evidence model is validated.

**Consequences:** native concept slots, when they exist, still start at `PROPOSED_CONCEPT` and pass the same validation protocol (ADR-009).

---

## ADR-009: Conservative feature/concept semantic states

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- Semantic status is tracked separately from evidence status in a `SemanticStatus` enum: `UNLABELED_FEATURE`, `PROPOSED_CONCEPT`, `VALIDATED_CONCEPT`, `REJECTED_CONCEPT`.
- No code path converts a feature to a concept automatically.
- `VALIDATED_CONCEPT` requires attached `ValidationResult` records from a declared protocol that includes **both** a detection test with counterexamples **and** a causal-relevance test against random-direction controls.

**Reason:**
- A readable feature is not a concept.
- A decodable concept is not necessarily used: probe accuracy is not causal use.

**Alternatives considered:**
- Treating concept status as part of `EvidenceStatus` would conflate how evidence was produced with where a hypothesis is in its lifecycle.
- A three-state lifecycle without `REJECTED_CONCEPT` would silently lose negative results.

**Consequences:**
- Implemented in Phase 6. The enum is defined in Phase 1 so the schema is stable.
- `EvidenceStatus.VALIDATED_CONCEPT` is assignable only to a `ConceptActivation` whose concept is `VALIDATED_CONCEPT`. This is checked at construction.

---

## ADR-010: Architecture independence of the core

- **Date:** 2026-09-24
- **Status:** Accepted

**Decision:**
- `beyondnn` core code contains no architecture-specific branches. It understands `nn.Module` trees, module boundaries, and pytree outputs only.
- Phase 1 must pass the same test suite on TinyMLP, TinyCNN and TinyTransformer.

**Reason:**
- Transformer-only tools already exist (TransformerLens, circuit-tracer).
- An abstraction that only works on one architecture is not a general evidence schema.

**Alternatives considered:** a transformer-first design with generalisation later. Rejected, because the abstractions ossify around the first architecture.

**Consequences:**
- Architecture-specific conveniences (canonical transformer site names, token-level evidence rendering) live in adapters or optional helpers, never in `core`.
- A CI test asserts that the shared test suite is parameterised over all three tiny models.

---

## ADR-011: Flat package layout

- **Date:** 2026-09-24
- **Status:** Accepted (confirmed in review of 2026-09-25).

**Decision:** the package lives at `beyondnn/` in the repository root (flat layout), as specified in the repository plan.

**Reason:** the repository plan specified it, and it's simpler to navigate for new contributors.

**Alternatives considered:**
- *`src/beyondnn/` (src layout)*: proposed in the first architecture draft. It prevents tests from accidentally importing the working tree instead of the installed package.

**Consequences:**
- The src layout's main protection is lost. It is mitigated by running pytest with `--import-mode=importlib`, and by CI installing the package (`pip install -e .`) before running tests.
- This is revisitable by a superseding ADR if packaging bugs appear.

---

## ADR-012: Claims are immutable data; tests are declared specs; status is derived

- **Date:** 2026-09-24
- **Status:** Accepted (review of 2026-09-25). Schema types land in Phase 1 (M1.1), runners in Phase 2.

**Decision:**
- `Claim` is an immutable record with no status field.
- Tests are `ClaimTestSpec` records with pre-declared criteria and a `spec_hash`.
- Running a test produces a `ClaimTestResult`.
- A claim's standing is an `Assessment`, computed by a pure, versioned policy from (claim, results).
- Testing is a free function, `bnn.test_claim(model, claim, inputs, tests)`. It is not a method on `Claim`.

**Reason:**
- A status stored on the claim can go stale, or be set without evidence.
- Pre-declared criteria with a hash make post-hoc threshold tuning visible.
- Keeping claims as pure data lets them be serialised, diffed, and shared without model references.

**Alternatives considered:**
- `Claim(status=…, evidence=[…], tests=[…])` with a `claim.test()` method (the initial sketch): mutable or stale status, and schema objects coupled to runtime.
- Status stored on claims, with an audit log of changes: more state and more ways to be wrong.

**Consequences:**
- Users read verdicts from `Assessment`, not from `Claim`.
- Relation–test compatibility is part of the test registry, so an inapplicable test yields `NOT_APPLICABLE` rather than support.
- Class names avoid pytest's `Test*` collection prefix.

---

## ADR-013: Estimand scope separates instance, finite-sample, and population causal results

- **Date:** 2026-09-25
- **Status:** Accepted. Refines the status-derivation consequence of ADR-006.

**Decision:** every causal result and every claim carries an explicit **estimand**, with a scope of `INSTANCE`, `FINITE_SAMPLE`, or `POPULATION`.

- **`INSTANCE`**: an intervention on exactly one input. `n = 1`, and the input is identified by `sample_id`. Exact measurements have status `INTERVENTIONAL`.
- **`FINITE_SAMPLE`**: interventions on exactly the `n` identified inputs (`sample_id`), summarised by a named `aggregation` (e.g. `"mean"`). The aggregate is still `INTERVENTIONAL`, because it summarises direct measurements and makes no claim beyond those inputs. The effect record (Phase 2) must reference the individual per-input effects.
- **`POPULATION`**: a claim about a distribution, future inputs, or a dataset beyond the measured examples. It names the `population` and the `aggregation` / estimator. It is always `ESTIMATED_CAUSAL`.

**Rules:**
- `INTERVENTIONAL` evidence can never have `POPULATION` scope.
- A `POPULATION` causal claim can only be decided (supported or contradicted) by `ESTIMATED_CAUSAL` evidence with `POPULATION` scope for the same population.
- A `FINITE_SAMPLE` claim can only be decided by evidence on the same sample (same `sample_id` and `n`).
- An `INSTANCE` claim can only be decided by evidence on the same instance.
- **Sample size never changes scope.** A `FINITE_SAMPLE` estimand with `n = 10⁶` is still finite-sample. There is no automatic promotion.
- Approximate estimators (e.g. attribution patching) give `ESTIMATED_CAUSAL` at any scope.

**Reason:**
- The same averaged number means different things depending on whether it describes the measured set or is used to generalise beyond it.
- Tying status to (estimator, scope) avoids both over-claiming (a finite mean presented as a population fact) and under-claiming (an exact finite summary labelled as an estimate).

**Alternatives considered:**
- *Status derived from `n` alone (`n > 1` gives `ESTIMATED_CAUSAL`)*: the first proposal. It mislabels exact finite summaries.
- *Always `INTERVENTIONAL` for exact estimators*: lets finite results silently support population claims.

**Consequences:**
- `Claim` carries an `Estimand`, and so do causal `EvidenceRef`s. Scope compatibility is checked at `ClaimTestResult` construction.
- Moving from a finite-sample result to a population claim requires an explicit estimation step (Phase 2+) that produces a new `ESTIMATED_CAUSAL` record.

---

## ADR-014: Incremental, versioned schema instead of freezing future record types

- **Date:** 2026-09-25
- **Status:** Accepted

**Decision:** Phase 1 stabilises only:
- the record mechanism: base record, identity, status mechanism, provenance reference, lineage, and a registry of record kinds;
- the serialisation envelope and decoding rules;
- the records Phase 1 needs: `InputRecord`, `OutputRecord`, `ActivationRecord`, `TensorRef`, `TraceLimitation`, plus the claim types approved in ADR-012.

Later-phase, algorithm-specific records (intervention, causal effect, attribution, feature, concept) arrive in later schema versions, when their semantics have been validated.

**Versioning mechanism:**
- Every serialised record carries `schema_version` (the global envelope version), `kind`, and `record_version` (per kind).
- Decoding is strict. It rejects:
  - unknown kinds;
  - unknown fields;
  - missing required fields;
  - a newer `record_version` than the installed library supports;
  - an incompatible `schema_version`.
- Each rejection raises a specific error.
- Older record versions are accepted only through explicitly registered migrations.
- Before the first public release (`0.1.0`), `record_version = 1` may still change without migrations, because no persisted traces exist. After release, any field change bumps `record_version` and requires a migration.

**Reason:** freezing speculative record types now would lock in semantics that have not been tested experimentally. Compatibility should come from explicit versions and migrations, not from guessing future types.

**Alternatives considered:**
- *Define every Phase 2–6 type in Phase 1*: the earlier plan, now rejected.
- *Permissive decoding (ignore unknown fields or kinds)*: silently drops information that may carry epistemic meaning.

**Consequences:**
- The schema doc separates **accepted (schema 0.1)** types from **proposed (future)** types.
- An opt-in pass-through for unknown kinds (preserving them verbatim on re-save) is deferred to M1.7, where save/load exists. Until then, unknown kinds always raise.

---

## ADR-015: `TraceResult` is single-execution evidence; multi-input work goes in a later `Study`

- **Date:** 2026-09-25
- **Status:** Accepted

**Decision:**
- A `TraceResult` holds evidence from one execution / trace context.
- Dataset-level work goes in a lightweight `Study` container, planned for Phase 2 (not implemented in Phase 1): multiple traces, multi-input claim test results, dataset-level assessments, population estimates, and cross-input statistics.
- Claim types are container-neutral. They reference records by id and never assume that everything they reference lives in one trace.

**Reason:** claims typically range over many inputs, while a trace describes one run. Mixing the two would either bloat traces or force claims into a single-trace shape.

**Consequences:**
- `TraceResult.add()` (M1.6) validates only records belonging to that execution. It does not own claim/test/assessment bookkeeping for multi-input claims.
- References between records carry the attributes needed for *local* invariant checks, such as an evidence reference carrying the referenced record's status and estimand. Containers (`TraceResult` now, `Study` later) verify that references match the records they point to.
- `Study` may move earlier if implementation evidence shows it is needed.

---

## ADR-016: Content-derived record identity

- **Date:** 2026-09-25
- **Status:** Accepted (review of 2026-09-25). Proposed during M1.1; the review added the identity/equivalence note below.

**Decision:** `record.id = f"{kind}:{sha256(canonical_json({kind, record_version, data}))[:32]}"`.
- `data` is every init field, encoded.
- Canonical JSON uses sorted keys, compact separators, UTF-8, and shortest round-trip floats.
- Order-insensitive collections are sorted at construction.
- The id is computed at construction, excluded from equality, and verified on decode.

**Reason:**
- Deterministic across processes and machines. Python's randomised `hash()` is never used.
- Equal content gives an equal id, which makes deduplication and diffs trivial.
- The id doubles as an integrity check for payloads.
- References (`derived_from`, evidence) cannot silently point at modified content.

**Alternatives considered:**
- *Sequential per-trace ids (`act:0007`)*: the original proposal. Not stable across containers, so they conflict with container-neutral claims (ADR-015).
- *uuid4*: not deterministic.
- *Full 64-hex digests*: longer ids for negligible gain at this scale.
- *Excluding `provenance_id` or `statement` from identity*: rejected for uniformity. One rule for all kinds.

**Consequences:**
- The same content under a different provenance is a different record, so provenance ids must be deterministic (constraint on M1.2).
- Rewording a claim's statement yields a new claim, and results attached to the old one stay attached to the old one.
- `record_version` is part of identity, so migrations change ids, and containers must remap references when migrating (M1.7).
- Tensor bytes do not contribute unless `TensorRef.content_digest` is set.

**Record identity is NOT semantic equivalence** (added at acceptance).
- `record.id` identifies a specific artifact: exactly this content, under this provenance, with this wording.
- Two formally equivalent claims can have different ids, because their statements, provenance, or other non-semantic metadata differ.
- Later phases must **never** use `record.id` to decide whether two claims mean the same thing. Semantic equivalence is based on the formal claim structure available in the schema: in schema 0.1, subject, relation, target and estimand/scope. The qualitative direction is carried by the relation; quantitative decision criteria belong to `ClaimTestSpec`. Future schema versions may add an explicit expectation field if Phase 2 experiments show it is necessary.
- No semantic-equivalence mechanism is built now.

Revision note (2026-09-25, during review, before acceptance):
- `derived_from` and `ClaimTestResult.evidence` are also sorted at construction, so parent and evidence order cannot change ids.
- `-0.0` is normalised to `0.0` in float fields and JSON data, so records that compare equal have equal ids.

**Amendment (2026-09-25, M1.2 kickoff review).**
- *Expectation wording.* The equivalence sentence above originally listed "expectation" as if a field existed. It was corrected to describe the schema-0.1 structure. No schema change was made.
- *Unicode.* BeyondNN does **not** Unicode-normalise strings that enter record identity. `"é"` as one code point and `"e"` followed by a combining accent are distinct content, and may yield different ids. This is intentional: identity is about the artifact exactly as produced. Semantic-equivalence layers may normalise text later, if a specific protocol requires it. User content is never silently NFC/NFKC-normalised.

---

## ADR-017: Status not caller-supplied; intrinsic to kind where semantics guarantee it; explicit derivation rules

- **Date:** 2026-09-25
- **Status:** Accepted with corrected measurement/intervention semantics (review of 2026-09-25). See the revision note.

**Decision:**
- `EvidenceStatus` is a plain, unordered `Enum`.
- Evidence status is never supplied by callers: it is never a constructor argument, and it cannot be reassigned.
- **Status may be intrinsic to a record kind when the semantics guarantee it**: `InputRecord` and `OutputRecord` are OBSERVED, and `ActivationRecord` is MEASURED.
- **Status is not universally determined by kind.** Future kinds, such as causal effects, may derive status from their content (estimand, estimation method) through a pure `status` property. There is no rule of the form "one record kind = exactly one status forever".
- **Measured state under an intervention is not an intervention effect:**
  - Model state directly observed during an intervened execution is **MEASURED**. The intervention (execution mode, intervention id, …) is recorded in provenance / execution context.
  - **INTERVENTIONAL** is reserved for evidence that represents an intervention-derived **effect**: a comparison of intervened against baseline behaviour (e.g. baseline metric 0.91, intervened 0.34, effect −0.57).
- `ALLOWED_PARENT_STATUSES` states which parent statuses each evidence status may be derived from. Status-less records (claims, specs, results, assessments, limitations) may derive from anything, and evidence records may not derive from status-less records.

**Reason:**
- An ordering would imply a hierarchy that does not exist (ATTRIBUTED is not "more" or "less" than MEASURED).
- Binding status to the kind makes relabelling impossible without defining a new kind, and that is visible in review.

**Alternatives considered:**
- A `status` field validated against an allowed set per kind: still allows choosing a status at construction.
- An `IntEnum` hierarchy: rejected as scientifically wrong.

**Consequences:**
- `derived_from` means "values computed from". Selection context goes in provenance / experimental design. For example, an attribution used to choose which site to ablate does not make the resulting activation derived from the attribution. A later claim test result may still reference that attribution when it is scientifically relevant; that machinery is not built yet.
- `EvidenceRef` refuses GENERATED and status-less records, so generated text cannot be cited as evidence anywhere.
- M1.2 provenance must be able to express execution context (e.g. `execution_mode`, `intervention_id`), because activations measured under intervention are distinguished only there.

Revision note (2026-09-25, review, before acceptance):
- Removed the proposed rule *"A measurement taken during an intervened pass is not MEASURED. Its status (likely INTERVENTIONAL) will be defined with the Phase 2 records."*
- Replaced "status is a property of its kind" with the intrinsic-where-guaranteed wording above.

---

## ADR-018: Explicit assessment policies; causal relations need named protocols

- **Date:** 2026-09-25
- **Status:** Accepted (review of 2026-09-25).

**Decision:**
- An `Assessment` is always computed under an explicit, versioned `AssessmentPolicy`, which is embedded in the assessment and so part of its identity.
- A policy lists, per relation, the protocols that must each have a `SUPPORTS` result.
- Assessing a causal relation under a policy that names no protocol for it raises.
- The verdict follows a fixed outcome table (see the schema §A.7), and a stored verdict must equal the recomputed one.
- BeyondNN ships no policy and no thresholds in schema 0.1.

**Reason:**
- This makes it impossible to reach `SUPPORTED` for `NECESSARY_FOR`/`SUFFICIENT_FOR` (or any causal relation) without an explicitly recorded statement of which protocols justify that wording.
- It avoids inventing thresholds before the protocols exist.

**Alternatives considered:**
- *A built-in default policy*: would encode arbitrary requirements before Phase 2 evidence exists.
- *Verdict from outcomes only, with no policy*: a single supporting test could make "necessary" SUPPORTED.

**Consequences:**
- Until Phase 2 adds a protocol registry that declares which relations each protocol justifies, policies are trusted as written. This is a documented gap.
- Non-causal relations can be assessed under a policy with no requirements.
- **No universal numeric thresholds.** Policies and specs may declare thresholds as their own criteria, but BeyondNN does not present hand-picked defaults as scientifically validated. Any future default policy must be versioned, documented, and tested on controlled ground-truth models before researchers are encouraged to rely on it (added at acceptance).
- The v0 verdict rules are kept: support plus contradiction gives MIXED, and a missing required protocol gives INCONCLUSIVE.

---

## ADR-019: Provenance identity is reproducible conditions; occurrence is separate

- **Date:** 2026-09-25
- **Status:** Accepted (M1.2 review of 2026-09-25). `ExecutionOccurrence` stays a separate type for now.

**Decision:**
- A `ProvenanceRecord` answers **how** evidence was produced, never what it means. It holds exactly four immutable values:
  - `ModelIdentity`;
  - `EnvironmentIdentity`: Python implementation and version, torch version, BeyondNN version, OS family, CPU architecture;
  - `ExecutionContext`: mode CLEAN/INTERVENTION, `intervention_id` (absent iff CLEAN), device, `training`, `grad_enabled`, and optional `Randomness`;
  - `MethodIdentity`: name, version, and canonical params.
- Its id (`provenance:<hash>`, per ADR-016) is what evidence records store in `provenance_id`. All four values contribute, and nothing else does.
- **Occurrence is separate.** When an execution happened is recorded in a separate `ExecutionOccurrence` record (`provenance_id`, `started_at` in UTC ISO-8601 with `Z`). Its id is an occurrence identity. Timestamps therefore never change `provenance_id`, and two executions under identical conditions share it.
- **Randomness** distinguishes a *declared seed* (as stated by the caller, unverified) from a *captured RNG-state digest* (SHA-256 of `torch.get_rng_state()`, read without mutation; the raw state is not stored). RNG capture is opt-in, because a captured state legitimately changes the conditions after every random draw.
- **Privacy.** Automatically collected provenance never contains hostname, username, home directory, working directory, file paths, or hardware identifiers. Machine identity is never part of model identity.

**Reason:**
- M1.1 record ids include `provenance_id`, so provenance ids must be deterministic.
- Separating conditions from occurrences keeps useful timestamps without making every record id run-specific.

**Alternatives considered:**
- *A timestamp inside `ProvenanceRecord`, excluded from identity*: two records with the same id but different content would violate ADR-016's collision rule.
- *No timestamps*: loses useful operational metadata.
- *uuid run ids in provenance*: turns provenance ids into run ids.

**Consequences:**
- Identical conditions on different machines differ only if `EnvironmentIdentity` differs (e.g. OS or architecture). Machine identity itself is not recorded.
- `ExecutionOccurrence` is a minimal placeholder for run-level metadata. M1.6 decides how a `TraceResult` references it.
- Evidence records' `provenance_id` is still only token-checked locally. M1.6 containers must verify that it names a `ProvenanceRecord` they hold.

---

## ADR-020: FULL model fingerprint, algorithm v1

- **Date:** 2026-09-25
- **Status:** Accepted with a required correction (M1.2 review of 2026-09-25). See the correction note at the end of this ADR: all buffers are hashed.

**Decision:** `ModelIdentity` carries two SHA-256 digests plus counts, computed by `beyondnn.provenance.fingerprint_model` with method `FULL`, `algorithm_version = 1`.

- **Structure digest.** Canonical JSON of:
  - every module path from `named_modules(remove_duplicate=False)`, with its fully qualified class;
  - *module alias groups*: paths that are the same module object;
  - every parameter and persistent buffer: name, role, dtype, shape;
  - the names of non-persistent buffers;
  - *tensor alias groups*: names that are the same tensor object, i.e. tied parameters.

  Groups are sorted lists of names, never object ids or addresses.
- **State digest.** SHA-256 over a version tag, then, for each distinct parameter or persistent buffer ordered by representative name (the smallest name in its alias group), a length-prefixed canonical header (name, role, dtype, shape, nbytes) followed by the raw bytes. The bytes come from a contiguous CPU copy with lazy conj/neg bits resolved. Values are hashed bitwise and never converted between dtypes.
- **Excluded:** device placement (for device-independent identity), `requires_grad`, train/eval mode (execution context), non-persistent buffer values, and `None` parameter/buffer slots.
- **Unsupported, failing with `FingerprintError` rather than hashing partially:**
  - tensor subclasses other than `nn.Parameter`, including lazy/uninitialised parameters;
  - meta, sparse and other non-strided, quantized, and nested tensors;
  - modules with a custom `get_extra_state`;
  - big-endian hosts.

  Non-persistent buffers are never read, so they cannot trigger this.
- **No sampled/partial/fast modes and no caching.** Every call re-reads every value, so in-place mutation is always seen.

**Reason:**
- The digest must change for any value, dtype, shape, structure, or aliasing change, and stay identical across processes and hash seeds.
- Tied versus equal-but-copied weights are different models: training behaviour differs.
- Device independence lets the same state on CPU, CUDA, or MPS share one identity. Only CPU is tested in v0.

**Alternatives considered:**
- `torch.save`, pickle, or `repr` hashing: nondeterministic or not content-based.
- `state_dict()` iteration: loses aliasing topology, and is subject to user state-dict hooks.
- Hashing with object ids: not reproducible.
- Sampled hashing for scale: rejected. A sample must never be presented as full identity.

**Consequences:**
- Module classes are identified by qualified name (`module.qualname`). If a class moves between modules, for example across torch versions, the structure digest changes even though the model is the same.
- Uses the private-but-stable `nn.Module._non_persistent_buffers_set`. Its absence raises.
- The measured cost is about 1.8 ms per ~1M float32 parameters on Apple arm64 CPU (see the experiment log). The earlier `bytes(untyped_storage())` path took about 3 s and was replaced by a byte-identical `ctypes.string_at` read.
- Any algorithm change requires `algorithm_version` 2. Golden digests in `tests/test_fingerprint.py` pin v1.

**Correction (2026-09-25, M1.2 review, before any release):**
- **Buffers.** v1 as first implemented hashed only the names of non-persistent buffers. That was wrong for a FULL fingerprint: `persistent=False` only removes a buffer from `state_dict`, and the buffer can still take part in `forward`. v1 now:
  - hashes the values of **all** registered non-`None` buffers;
  - records `persistent` as buffer metadata in both the structure and the state headers;
  - applies the unsupported-tensor checks to non-persistent buffers too.

  Buffer counts cover all buffers. The golden digests were updated. Because no fingerprint had been persisted or released, the fix keeps `algorithm_version = 1`; any later change requires version 2.
- **What FULL means.** The full *supported* PyTorch module topology and registered tensor state, under BeyondNN's v1 fingerprint specification. It is **not** a cryptographic identity of every behaviour the Python object can exhibit. Python code is not hashed: changing `forward` from `x + 1` to `x + 2` while keeping the qualified class name, module tree, parameters and buffers leaves the fingerprint unchanged. Automatic source hashing is rejected as fragile (dynamic classes, monkeypatching, compiled extensions, interactive definitions, decorators, generated functions, unavailable source). A caller-supplied implementation/code/model revision is a roadmap item.
- **Class paths.** The qualified class names stay in the structure. A class-path change can change structure identity even when numerical behaviour is unchanged. This is acceptable because fingerprint identity is artifact identity, not semantic equivalence.
- **The private persistence field** is read only in `_non_persistent_buffer_names`. It fails clearly if unavailable, and is checked against `state_dict()` keys on the tested torch versions (2.12, 2.14).
- **Approved for Phase 1:**
  - `ExecutionOccurrence` stays separate;
  - train/eval, grad mode and device belong to `ExecutionContext`;
  - `requires_grad` is excluded;
  - CPU RNG capture is opt-in;
  - the six-field environment;
  - FULL is the only fingerprint mode;
  - unsupported tensors fail explicitly;
  - the privacy boundary is unchanged.

---

## ADR-021: HookSession semantics: physical hooks, indices, aliases, no retention

- **Date:** 2026-09-25
- **Status:** Accepted after the invocation-guard fix (M1.5 review of 2026-09-25). See the amendment at the end of this ADR.

**Decision** (`beyondnn/core/hooks.py`, internal):
- **Scope.** `HookSession(model, sink=…, outputs=[…], inputs=[…], alias_policy=REFUSE)` is a single-use context manager.
  - Patterns are resolved against the live model on `__enter__`. Changing the module tree while the session is active is unsupported.
  - Hooks exist only between `__enter__` and `__exit__`. Only BeyondNN's own handles are removed. Removal happens on normal exit, on any exception (including `BaseException`), and after a partially failed installation. A second exit is harmless.
- **Physical hooks.** Per observed module *object*:
  - one pre-hook, which claims the call index and emits INPUT if selected;
  - one `always_call` cleanup forward hook, which releases the invocation and never raises or emits;
  - one ordinary forward hook iff OUTPUT is selected, which runs only on success and emits OUTPUT.

  The root additionally gets a prepended pass-start pre-hook and an `always_call` pass-end hook. Overlapping patterns, alias paths, and INPUT+OUTPUT never duplicate hooks.
- **Indices.**
  - **`pass_index`** is the n-th root invocation in the session. A root invocation that raises consumes its index. Re-entrant root calls don't start a pass. Executions outside a root invocation get `-1`.
  - **`call_index`** is the n-th invocation of that physical module in the pass, reset per pass. It is claimed by the pre-hook, so a failed invocation consumes its index, and INPUT and OUTPUT share it. A per-module stack pairs recursive calls. Out-of-pass calls are counted separately and never reset.
- **Aliases.** Hooks attach to objects, so a module registered under several paths cannot be attributed to one path.
  - **`REFUSE`** (default) raises `AliasSiteAmbiguityError` when such a module is selected.
  - **`GROUP`** (internal) observes it once per execution, with `paths` holding all aliases and `path_specific=False`. `event.site` raises for such events.
- **Values.** Hooks return `None`, so arguments and outputs are never replaced. Values are not detached, cloned, or moved. `kwargs` is exposed as a read-only view. Events are ephemeral (`HookEvent`, no id, not a record), and the session keeps no reference to them or their values. The sink decides retention (M1.6).
- **User hooks.** User hooks registered earlier run before BeyondNN's, except the prepended pass-start hook. BeyondNN therefore observes the values the module actually receives and returns.

**Reason:**
- What a hook proves is narrow: this object executed and this value crossed its boundary. The design must never overstate it. Attributing an aliased execution to one path would be fabricated evidence.
- Using ordinary hooks for emission and `always_call` hooks for bookkeeping keeps indices consistent when `forward`, a user hook, or the sink raises. PyTorch passes `output=None` *or* a real output on failure paths, so `always_call` hooks cannot tell success from failure.

**Alternatives considered:**
- One hook per role, counting in the output hook: failed calls would not consume indices, and recursion would mispair.
- Prepending BeyondNN pre-hooks: this would guard the recursion edge case below, but observe arguments *before* user pre-hooks modify them.
- Deduplicating by path: this double-hooks aliased objects.

**Consequences / limits:**
- Not thread-safe, and not for `torch.compile`d modules. Functional ops (`F.gelu`, residual `+`) remain unobserved (`FUNCTIONAL_OPS_UNOBSERVED`).
- Edge case: if a user or global pre-hook that runs *before* BeyondNN's raises during a *recursive* call of the same module, the cleanup hook can release the enclosing call's index.
- A selected module that never executes (e.g. an iterated `ModuleList`) yields no events.
- `pass_index = -1` events cannot become `ActivationRecord`s under schema 0.1 (`pass_index >= 0`). M1.6 must decide how to represent them, or refuse them, with a limitation.

**Amendment (2026-09-25, M1.5 review): invocation guard.**
- The recursive edge case above was not acceptable, because it could silently misattribute later evidence. Invocation bookkeeping now has two stages:
  - a *guard* pre-hook, prepended so it runs before user pre-hooks, claims the call index and pushes the frame, and never emits;
  - the *observation* pre-hook, which still runs after user pre-hooks, emits INPUT with the arguments `forward` actually receives.
- If a user pre-hook raises, the attempt has consumed its own call index, no INPUT event exists, and the cleanup pops exactly that attempt's frame. The enclosing recursive frame stays intact (regression test, mutation-checked).
- **Attempt identity ≠ observed input.**
- Global module pre-hooks run before any per-module hook, so a session **refuses to start** while any are registered (`HookSessionError`).
- Hook count per observed module: guard + observation + cleanup, plus an output hook when OUTPUT is selected.
- The edge-case bullet above is superseded.

---

## ADR-022: Caller-declared model context in provenance (ProvenanceRecord v2)

- **Date:** 2026-09-25
- **Status:** Accepted for the M1.6 gate. Awaiting review.

**Decision:**
- New value `ModelDeclaration(config: JsonMap = {}, implementation_revision: str | None, checkpoint_revision: str | None)`. At least one field must be set, and revisions are non-empty strings without whitespace.
- It is entirely caller-supplied and optional. It is never inferred or scraped from Python attributes, never verified (e.g. no git or hub lookups), and carries no generated interpretation. `config` keys are not prescribed.
- `ProvenanceRecord` gains `declared_model: ModelDeclaration | None = None`, kept separate from the automatically measured `model: ModelIdentity`, so the two epistemic origins stay inspectable and distinct. The FULL v1 fingerprint is unchanged.
- **Versioning:** `ProvenanceRecord` moves to **record_version 2**. A registered migration upgrades v1 payloads by adding `declared_model = None`.
  - A v1 payload that already contains the field is rejected.
  - Migrated records get new ids (ADR-016). Remapping references stays M1.7's job.
  - Migration errors are now reported as `DecodeError` (a small codec fix).
- `make_provenance(…, declared_model=None)`. When nothing is declared, nothing is fabricated.

**Reason:**
- Closes the M1.3 finding: `TinyTransformer(n_heads=2)` and `n_heads=4` share a FULL v1 fingerprint. With `declared_model=ModelDeclaration(config={"n_heads": 2})` versus `{"n_heads": 4}`, their `provenance_id`s differ.
- The fingerprint measures what BeyondNN can observe. The declaration records what only the experimenter knows. Merging the two would disguise claims as measurements.

**Alternatives considered:**
- *Auto-hashing module attributes or source*: rejected as fragile and unscoped (ADR-020).
- *Folding the declaration into `ModelIdentity`*: mixes measured and declared data.
- *Changing record_version 1 in place, since nothing was released*: rejected in favour of an explicit, tested version transition.

**Consequences:**
- `provenance_id` now covers `model`, `environment`, `execution`, `method`, and `declared_model`. With no declaration, the value is `null`.
- The **M1.6 hard gate is resolved.** M1.6 decides whether traces without a declaration get a limitation (none is added now).
- All existing provenance ids changed with the version bump. That is acceptable before any release.

---

## ADR-023: The trace pipeline: trace(), recording(), TraceResult

- **Date:** 2026-09-25
- **Status:** Accepted for M1.6 implementation. Awaiting review.

**Decision** (`beyondnn/core/trace.py`, `beyondnn/core/tensors.py`; public `bnn.trace`, `bnn.recording`, `bnn.TraceResult`):

- **API.**
  - `recording(model, *, sites=(), input_sites=(), retention="summary", declared_model=None, randomness=None)` is a single-use context.
  - `trace(model, *inputs, …, model_kwargs=None)` is exactly `recording()` plus one root call. There is one engine.
  - `sites` selects module OUTPUTS and `input_sites` selects INPUTS. The root (`""`) may not be selected: root inputs and outputs are always recorded, as OBSERVED `InputRecord`/`OutputRecord`.
- **Result states.** `ctx.result` is only available after a clean exit. It raises `RecordingError` before exit, and after any failure, including a root call that raised even if the caller swallowed it, and a recording with no root call. There are no partial traces. A failed recording keeps only a message; tensors and traceback are dropped.
- **Per-pass provenance.** The model is fingerprinted at the start of every root invocation. Each pass's evidence references the `ProvenanceRecord` for the state at its start, so state changes (BatchNorm running statistics, mutable buffers) yield new provenance. Unchanged state deduplicates to one record.
  - Execution context: CLEAN; device (the first parameter/buffer, else the first input tensor, else cpu); `model.training`; `torch.is_grad_enabled()`; and the caller's declared `randomness`, if any.
  - Method: `forward_hook` v1. Retention is not part of the method or model identity.
  - One `ExecutionOccurrence` per pass.
- **Records.**
  - One `ActivationRecord` (MEASURED) per tensor leaf at each selected site. It carries `Site(module, io, output_path)`, `pass_index`, `call_index`, a `TensorRef`, `provenance_id`, and `derived_from` = the pass's `InputRecord`.
  - `OutputRecord` derives from its pass's `InputRecord`.
  - Records are kept in execution order.
- **Tensor paths.**
  - `""` for a bare tensor, `[i]` for sequences, `["key"]` (JSON-quoted) for str keys, `[k]` for int keys.
  - Root and module-input leaves are prefixed `args`/`kwargs`; root outputs are prefixed `output`.
  - Mappings are walked in insertion order.
  - Non-tensor leaves are never stringified or pickled: they are counted, and reported as `NON_TENSOR_LEAVES_IGNORED`. `None` is skipped.
- **Retention.**
  - `none` (shape/dtype/device only);
  - `summary` (default; plus float64 scalar stats computed without autograd, `None` for complex);
  - `cpu` (plus a detached contiguous CPU clone, stored under key `sha256:<digest>` = `content_digest`; identical contents are stored once).
  - There is no live/device retention.
- **Refusals.**
  - A selected module executing outside a root call raises `OutOfPassExecutionError`. `-1` is never stored or turned into 0.
  - Aliased modules raise `AliasSiteAmbiguityError` (the group mode is not public).
  - Re-entrant root calls raise `RecordingError`.
- **Limitations emitted by traces:**
  - `FUNCTIONAL_OPS_UNOBSERVED`, always;
  - `PARTIAL_SITE_COVERAGE`, when output sites do not cover every named module;
  - `SELECTED_SITE_NOT_EXECUTED`, which names every silent selected site;
  - `NON_TENSOR_LEAVES_IGNORED`.

  `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE` and `NO_CLAIMS_TESTED` are emitted by `explain()` (M1.8), not by plain traces.
- **Container integrity** (`TraceResult._add`): every record entering a trace must pass all of these checks:
  - the kind is registered and the id is recomputed from content;
  - a duplicate id is deduplicated by canonical JSON, never `==` (NaN); conflicting content is rejected;
  - every `RecordRef`/`EvidenceRef`/`ClaimRef`/`SpecRef`/`ResultRef` resolves to a record already in the trace and passes `verify_ref`;
  - every `applies_to` id exists;
  - every `provenance_id` names a `ProvenanceRecord` in the trace.

  Finalised traces are read-only.
- **Schema.** `InputRecord`/`OutputRecord` move to record_version 2 with `pass_index` (migration v1→v2 sets `None` = unknown, never an invented 0). There are two new limitation codes.

**Typed-reference review (deferred from M1.1).** Keep all five reference types.
- In real traces, `RecordRef` is used on every activation and output. The container verifies all five generically, with a single `verify_ref` path.
- Their embedded attributes (status, estimand, relation, protocol) let schema invariants run locally *and* let the container prove the references true.
- No duplication burden was found.

**Consequences / caveats:**
- An `ExecutionOccurrence` id includes a microsecond timestamp, so two passes starting in the same microsecond share one occurrence record. Occurrences are operational metadata and are not linked to passes.
- Fingerprinting once per pass costs about 2 ms per ~1M float32 parameters.

---

## ADR-024: Trace persistence, sidecar security, migration with reference remapping

- **Date:** 2026-09-25
- **Status:** Accepted for M1.7 implementation. Awaiting review.

**Decision** (`beyondnn/core/persistence.py`; public `TraceResult.save(path)`, `bnn.load_trace(path)`):

- **Format.** A directory containing `trace.json` and, only if tensors were retained, `tensors.pt`. The file names are fixed constants, and **no path is ever read from the JSON**.
- **`trace.json`** has exactly the keys `format` (`"beyondnn.trace"`), `format_version` (1), `schema_version`, `config`, `records` (envelopes in execution order), and `tensors` (sorted storage keys).
  - It is deterministic: sorted keys, 2-space indent, UTF-8, no NaN literals.
  - It contains no absolute or home paths.
- **Sidecar.**
  - It is a plain `{storage_key: Tensor}` dict. Keys are content digests (`sha256:<hex>`), so identical contents are stored once.
  - It is loaded **only** with `torch.load(weights_only=True, map_location="cpu")`.
  - Keys must equal the declared keys, and every value must be exactly `torch.Tensor`.
  - Symlinked `trace.json`/`tensors.pt` are refused.
- **Save atomicity.** Everything is written into a sibling temporary directory (tensors, then `trace.json` last, both fsynced), then renamed to the target in one `os.rename`. An existing target is never overwritten. On any failure, the temporary directory is removed. **Guarantee:** the target either does not exist or is complete.
- **Load validation.**
  - The document's keys and versions are checked.
  - Every record goes through the strict codec. The stored id is now verified against the payload's **own** kind, version, and data *before* migration; previously, migrated payloads skipped the id check.
  - Every record also goes through `TraceResult._add` (references, provenance, dedup).
  - Every retained tensor must match its `TensorRef` dtype, shape, and `content_digest`.
  - Any unknown key, kind, or version fails the load. Nothing is dropped, weakened, or invented.
- **Migration and remapping.**
  - Records are processed in order. When a migration changes an id (old → new), the new id is recorded in a map.
  - Every later record has all string occurrences of mapped ids rewritten before decoding (`derived_from`, evidence/claim/spec/result refs, `applies_to`, `provenance_id`). Its *original* stored id is verified first, and it gets its new id, so the rewriting cascades.
  - A missing migration fails the load.

**Reason:** loading must not trust a file because it parses. Arbitrary pickle execution, path traversal, and silently dropping or relabelling data are unacceptable for evidence.

**Consequences:**
- The file format is versioned separately from the schema. Old traces load only through registered migrations.
- `torch.save` output is not byte-deterministic; `trace.json` is.

---

## ADR-025: Occurrence per pass; public tracing refuses foreign forward hooks and multi-device models

- **Date:** 2026-09-25
- **Status:** Accepted (required by the M1.6/M1.7 review).

**Decisions:**
1. **Occurrences.** `ExecutionOccurrence` moves to record_version 2 with `pass_index: int | None`.
   - New traces set it to the root pass, so one occurrence corresponds to exactly one root invocation, even when two passes start within the clock resolution.
   - v1 records migrate to `None` (unknown, never invented). References to changed ids are remapped by the M1.7 loader.
2. **External forward hooks.** Public `trace()`/`recording()` produce provenance-bearing evidence. Forward or forward-pre hooks that BeyondNN did not install could change module inputs or outputs without being represented in `ModelIdentity`, `ModelDeclaration`, or `ProvenanceRecord`.
   - So public recording raises `ExternalForwardHooksError` if any such hook (on any module of the tree, or registered globally) is present. The check runs at context entry, and again at the start and end of every root pass.
   - BeyondNN's own handles are recognised by hook id.
   - Callback source code is not inspected or fingerprinted.
   - `HookSession` is a general internal observation primitive and keeps its tested coexistence behaviour; **public trace recording has stricter reproducibility requirements.**
   - A hook added and removed entirely inside one forward call cannot be detected (documented).
3. **Devices.** The execution device is the set of devices of the model's parameters and buffers (or of the input tensors if the model has none). More than one distinct device raises `UnsupportedExecutionError`; one device is never silently chosen. Phase 1 has verified CPU only.

**Consequence:** users who rely on their own forward hooks must remove them before recording, until a future phase can represent hook-modified computation in provenance.

