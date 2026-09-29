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

---

## ADR-026: instrument() handle and the Phase-1 INPUT → WHY → OUTPUT view

- **Date:** 2026-09-25
- **Status:** Accepted for M1.8. Refines ADR-002, whose attribute-delegating handle is **not** implemented. **Amended by ADR-031 (Phase 4)** — see "Amendment" at the end of this ADR.

**Decision** (`beyondnn/explain/`):
- **`instrument(model)`** returns `Instrumented`: a frozen handle holding only `model`.
  - It is not an `nn.Module` and not callable.
  - It does not delegate module attributes, register, copy, patch, or hook the model.
  - `handle.model` is the original object.
  - `handle.trace(...)` and `handle.recording(...)` are exactly `bnn.trace(model, ...)` and `bnn.recording(model, ...)`.
- **`handle.explain(*inputs, sites=(), input_sites=(), retention="summary", declared_model=None, randomness=None, model_kwargs=None)`** runs one trace and returns `ExplainResponse`. There is **no `target` parameter**, because it would have no effect until attribution or interventions exist.
- **`ExplainResponse(trace)`**
  - It is frozen, and requires a single-pass trace.
  - `input` and `output` are the trace's OBSERVED records.
  - `why` is `Why(trace)`.
  - `render()` is a deterministic text restatement of the records: not generated prose, no LLM.
  - `from_trace(trace)` rebuilds the response after `load_trace`. There is no second persistence format.
- **`Why(trace)`** is a structured view:
  - `activations`: the trace's own MEASURED records, not copies;
  - `limitations`: the trace's limitations, plus `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE`, `NO_CLAIMS_TESTED`;
  - `provenance`;
  - `evidence_statuses`;
  - the constants `QUESTION` and `NOT_ANSWERED`.
- **Scientific meaning:** Phase-1 WHY answers *what internal evidence was measured while this output was produced*, and **not** *which internal state caused the output*. Nothing is ranked or called important, supporting, causal, decisive, or a reason. Phase 1 produces only OBSERVED and MEASURED evidence.

**Reason:** the user-facing shape exists now without claiming knowledge that later phases must earn. The trace remains the single source of truth.

**Top-level API:** `trace`, `recording`, `TraceResult`, `load_trace`, `instrument`, plus the status enums and `schema`. `ExplainResponse` and `Why` are importable from `beyondnn.explain`.

**Amendment (Phase 4, ADR-031, 2026-09-25).** What changes, and what stays the same:
- **Unchanged:**
  - `handle.explain(...)`, `ExplainResponse(trace)`, `from_trace`, `input`, `output`, `Why(trace)`, `Why.activations`, `QUESTION` and `NOT_ANSWERED`.
  - The measured-only meaning.
  - The measured-only render lines: verified to be identical to the Phase-3 renderer output for the same traces.
  - The limitations and evidence statuses of a measured-only view.
- **Changed (additive):**
  - `ExplainResponse` has an optional `bundle` (an `EvidenceBundle`). `ExplainResponse(trace)` composes a trace-only bundle.
  - `from_evidence(...)` and `bnn.compose(...)` present explicitly computed attributions, interventions, and declared claims in separate status sections.
  - `Why` gains `observations`, `measurements`, `attributions`/`attribution_views`, `effects`/`intervention_views`, `estimated_causal`, `claims`, `assessments`, `by_status`, `origin`, `coverage`, `unanswered`, `target`, and `question`/`not_answered`.
  - The measured-only render appends a NOT EVALUATED block (faithfulness, comprehensiveness, sufficiency, concept validation).
- **Still true:** `explain()` never runs attribution, interventions, or claim tests.

---

## ADR-027: Refuse module replacement and structural change during a recording

- **Date:** 2026-09-25
- **Status:** Accepted (found during the M1.10 audit).

**Problem:** hooks are placed at `recording()` entry. If the user replaced a selected module with an equivalent object (same class and shape, so the same structure digest) between passes, later passes silently produced no evidence for that site. The site had executed in pass 0, so `SELECTED_SITE_NOT_EXECUTED` did not fire.

**Decision:** at the start of every root pass, public recording checks two things, and raises `UnsupportedExecutionError` if either fails:
1. every selected path still resolves to the *same module object* hooked at entry;
2. the model's FULL structure digest equals the first pass's.

Parameter and buffer *value* changes remain allowed; they get per-pass provenance.

**Alternative considered:** re-resolving and re-hooking mid-session. Rejected: it adds complexity and makes pass semantics ambiguous.

---

## ADR-028: Phase-2 interventions: one comparison = one recording; INTERVENTIONAL is reserved for effects

- **Date:** 2026-09-25
- **Status:** Accepted. Phase 2 closed (GO WITH EXPLICIT LIMITATIONS). Custom metric identity amended by ADR-029.

**Decision:**
- **Records** (`schema/interventions.py`):
  - **`InterventionRecord`** (kind `intervention`, no status) specifies the replacement of one tensor leaf (`Site.output_path`) of one call (`call_index`) of a module OUTPUT. The operation is ZERO, CONSTANT (a finite scalar, or a retained exact-shape tensor), or PATCH (a retained value plus a `RecordRef` to the MEASURED source `ActivationRecord` in the same trace). Its content-derived id is the `intervention_id` in `ExecutionContext`, and the trace verifies that it resolves.
  - **`CausalEffect`** (kind `causal_effect`) has fields `interventions`, `metric: MetricSpec`, `estimand`, `baseline_value`, `intervention_value`, `effect`, and `estimator`.
  - **Sign convention:** `effect = intervention_value − baseline_value`, for every metric.
  - **Status** is derived: INTERVENTIONAL for INSTANCE (exact, from the two paired `OutputRecord`s) and FINITE_SAMPLE (mean over exactly `n` instance effects). ESTIMATED_CAUSAL only for POPULATION with a non-exact estimator, which Phase 2 never produces.
- **Runtime** (`beyondnn.interventions`): `intervene()` / `intervene_sample()` run one `recording()` per comparison. The passes are an optional CLEAN patch-source pass, a CLEAN baseline pass, and an INTERVENTION pass, all under `torch.no_grad()`.
  - The replacement is done by a BeyondNN-owned forward hook registered through `HookSession.owned_forward_hook`. It counts as owned (foreign-hook protection unchanged) and runs before output observation, so the recorded activation is the replaced one. It stays **MEASURED** (ADR-017).
  - **The comparison is refused** if any module is in training mode, if the CPU RNG state changes in either pass, if the model state fingerprint differs between the two passes, or if the declared call did not run exactly once.
- **Metrics:** built-ins `select`, `difference`, `mean` (scalar, pure, recorded as `MetricSpec`). `custom(name, fn, implementation_revision=...)` is never serialised; it emits `CUSTOM_METRIC_UNVERIFIED`. Identity per ADR-029.
- **Claims:**
  - The protocol registry is `interventions.PROTOCOLS`: `intervention_threshold` v1 justifies NECESSARY_FOR, DECREASES, and INCREASES. It closes the ADR-018 gap for this protocol, and `check_policy` validates policies against the registry.
  - The criteria (`min_effect`) are declared before running.
  - Any mismatch of site, metric, estimand, or operation gives NOT_APPLICABLE.
  - `INTERVENTION_POLICY` covers the three relations. SUFFICIENT_FOR is covered by no protocol, so it cannot be assessed.
- **Limitations:** `ZERO_ABLATION_MAY_BE_OOD`, `CONSTANT_REPLACEMENT_MAY_BE_OOD`, `PATCH_SOURCE_CONTEXT_DIFFERS`, `CUSTOM_METRIC_UNVERIFIED`, each scoped to the effect.

**Reason:**
- Keeping all passes of a comparison in one trace reuses Phase 1's record validation, per-pass provenance, and persistence with no new container or file format. The patch source is therefore a verifiable record, not an unexplained tensor.
- Refusing unpaired comparisons prevents confounded differences from being reported as intervention effects.

**Alternatives considered:**
- A `Study` container (ADR-015): still deferred, because a controlled comparison is one recording context.
- Patching from external saved traces: deferred. It needs cross-trace provenance.
- RNG replay to allow stochastic models: rejected for Phase 2. Refusal is the conservative choice.

**Consequences / limits:**
- Output interventions only.
- Finite-sample comparisons support ZERO/CONSTANT only.
- Only CPU RNG is checked (CPU-only phase).
- Effects are scoped to exactly the compared inputs. Nothing generalises to a population.

---

## ADR-029: Caller metrics carry a declared identity; undeclared caller metrics are refused

- **Date:** 2026-09-25
- **Status:** Accepted (Phase 2 hardening, before Phase 3).

**Problem:** a Phase 2 caller metric was identified only by its friendly name (`custom:<name>`). Two different functions with the same name therefore had the same `MetricSpec`, so their effects and claim targets were indistinguishable.

**Decision:**
- New value `MetricDeclaration(implementation_revision: str, config: JsonMap)`. It is **declared** by the caller, recorded as stated, never verified, and kept apart from anything BeyondNN measures. This is the same principle as `ModelDeclaration` (ADR-022).
- `MetricSpec.declaration` is **required** for caller metrics and forbidden for built-ins. Built-ins are identified automatically by name and `params`. Caller metrics keep their parameters in `declaration.config`, never in `params`.
- `metrics.custom(name, fn, *, implementation_revision, config=None)`: the revision is a required keyword. An unversioned caller metric is **refused** (option A) rather than allowed with a limitation, because a limitation cannot remove the identity ambiguity. `CUSTOM_METRIC_UNVERIFIED` still applies, because a declaration is not verification.
- The function is never serialised. Source code is never scraped, there is no bytecode hashing, and there is no repository lookup.
- `MetricSpec.target()` is the single deterministic mapping from a metric to a claim `TargetSpec`. For caller metrics it includes the declared revision and config, so a claim about revision A is NOT_APPLICABLE to evidence from revision B.
- `causal_effect` is now record version 2. v1 payloads migrate with `declaration = None`. A v1 effect on a caller metric therefore **fails to load** instead of gaining an invented revision. Nothing was ever published, so no real data is affected.

**Consequences:** declared identity is only as good as the caller's discipline: changing the function without changing the revision is undetectable. This is documented and not claimed as a guarantee.

---

## ADR-030: Phase-3 attribution: ATTRIBUTED by kind, explicit scalar targets and reductions, native references plus a Captum adapter

- **Date:** 2026-09-25
- **Status:** Accepted. Phase 3 reviewed and closed (GO WITH EXPLICIT LIMITATIONS).

**Decision:**
- **Records** (`schema/attribution.py`):
  - `AttributionRecord` (kind `attribution`) and `AttributionReduction` (kind `attribution_reduction`) have class-level status ATTRIBUTED. The status is never a constructor argument, and neither record is ever MEASURED or INTERVENTIONAL.
  - Identity covers the method spec (name, implementation, implementation version, params), the target `MetricSpec`, the baseline (kind plus tensor digest), the site, call, pass, and sample, and the attribution tensor's digest.
- **Targets** reuse the Phase-2 metric language. Built-in `select`, `difference`, and `mean` gained a differentiable form (`Metric.tensor`). A target that does not select exactly one element is refused, never summed. Caller metrics cannot be attribution targets in Phase 3.
- **Attributed tensor:**
  - A floating-point positional input `args[i]`. Integer inputs are refused (`DiscreteInputError`).
  - Or one leaf of call `k` of one module's OUTPUT, targeted by a temporary hook that replaces exactly that call's leaf with the evaluation point.
  - Alias paths are refused. The container verifies that the referenced activation has exactly the declared site, call, and pass.
- **Methods:**
  - Native reference implementations: `gradient`, `input_x_gradient`, `integrated_gradients`. IG uses rules `riemann_left`/`riemann_right`/`riemann_middle`/`trapezoid`, with float64 accumulation and a required baseline, which is ZERO only when explicitly requested.
  - Captum 0.9.x adapters: `Saliency(abs=False)`, `InputXGradient`, `IntegratedGradients`, and `LayerIntegratedGradients` (single-call layers, input-space baselines).
  - The adapters use `internal_batch_size=1` and a leading dimension of 1, so both implementations evaluate identical tensors.
  - The adapter refuses Captum versions it was not verified against.
- **Diagnostics:** IG records `completeness_delta = Σ attribution − (F(x) − F(x'))`. For layer IG, `F(x')` has the layer output replaced by the layer baseline. The delta is a numerical diagnostic, never a confidence score.
- **Reductions** are only explicit (`reduce(kind, dims)`), and each is recorded as its own derived record.
- **Execution:**
  - Attribution passes run outside tracing, on detached clones, with `torch.autograd.grad` only.
  - Before/after guards cover the model fingerprint, train/eval flags, parameter `.grad` (restored and refused), caller tensors (values, `requires_grad`, `.grad`), hook counts, and CPU RNG.
  - Foreign forward, backward, and parameter-tensor hooks are refused up front.
  - One CLEAN traced reference pass records provenance, the observed input/output, and the MEASURED attributed activation. The target value must agree with the attribution passes.
- **Claims:** a central protocol registry (`beyondnn.protocols`). `attribution_threshold` v1 justifies ATTRIBUTED_TO only: Σ|attribution| over the subject ≥ a declared threshold under the declared method, baseline, and call. The existing `ClaimTestResult` rule already refuses non-causal evidence for causal relations.
- **Limitations:** `ATTRIBUTION_BASELINE_ASSUMPTION` and `ATTRIBUTION_NUMERICAL_APPROXIMATION` (IG), `LAYER_ATTRIBUTION_PARTIAL_COVERAGE` (layer), and `DISCRETE_INPUT_ATTRIBUTED_VIA_REPRESENTATION` (layer attribution on a model with integer inputs).
- **Deferred:** custom attribution methods, attribution to module inputs, multi-sample attribution, and faithfulness metrics.

**Reason:**
- Attribution is method-relative evidence. Making every choice part of its identity, and keeping its status fixed, prevents it from being mistaken for measurement or causation.
- Reusing the trace, persistence, and claim machinery avoids a second evidence system.
- Native references make the Captum integration testable against analytic ground truth.

**Alternatives considered:**
- Tracing every IG step: rejected, because it would add n_steps passes of records that describe method internals, not evidence.
- An `Estimand` on attribution records: rejected, because it would make attribution look like causal evidence.
- Accepting Captum's default `Saliency(abs=True)` or its `riemann_trapezoid` as equivalent to the native rules: rejected. Both are documented convention differences.

---

## ADR-031: Phase-4 evidence synthesis: EvidenceBundle, structured WHY, exact input identity

- **Date:** 2026-09-25
- **Status:** Accepted and final. Phase 4 reviewed and closed (GO WITH EXPLICIT LIMITATIONS); `InputRecord.sample_id` explicitly approved by the owner as the exact input-identity mechanism (2026-09-25). The privacy limitation stands: the hash may confirm a guessed low-entropy input and is not an anonymisation mechanism.

**Decision:**
- **Input identity.** `InputRecord.sample_id` (record version 3) holds the exact identity of the root input: the same SHA-256 algorithm used by effect estimands, attributions, and claims, now in `beyondnn.core.samples`.
  - v2 records migrate with `None`.
  - Summary statistics cannot identify an input (they cannot tell (3, 5) from (5, 3)), so without this field composition could not avoid silent sample misassignment.
  - Privacy: the digest can confirm a guessed low-entropy input. It was already present in Phase 2/3 records.
- **`EvidenceBundle`** (`beyondnn.explain`): the deferred multi-evidence container (ADR-015), in its smallest form.
  - It is built only by `EvidenceBundle.compose(trace, *, attributions, interventions, claims, policies)`. It is immutable and is **not evidence**.
  - It covers **one explanation context**: one reference single-pass trace, one automatic `ModelIdentity` and one `ModelDeclaration` across every provenance record, one exact input sample, INSTANCE scope only, and at most one scalar target (`MetricSpec`; claims must match `target()`).
  - Incompatible evidence is refused (`CompositionError` subclasses), never merged or grouped silently.
  - It revalidates everything it is given:
    - record ids against content;
    - references within each source trace;
    - conflicting content under one id;
    - evidence cited by claim-test results;
    - every decisive result, re-derived with its registered protocol's evaluator and required to be identical. An unregistered protocol is refused.
- **Assessments** are derived with `Assessment.derive` under caller-supplied, registry-checked policies (`check_policy`).
  - A claim is assessed only under policies that name a protocol for its relation, using every recorded result about it.
  - There is no default or universal Phase-4 policy.
- **Claims** are only those declared by the caller or recorded with the evidence. None are generated.
- **Presentation:**
  - `bnn.compose(...)` returns an `ExplainResponse`. `Why` groups evidence by status (OBSERVED, MEASURED, ATTRIBUTED, INTERVENTIONAL, and a separate ESTIMATED_CAUSAL section).
  - Views (`AttributionView`, `InterventionView`, `ClaimView`) reference the original record objects.
  - Limitations are the deduplicated union of every source trace's limitations (details preserved), plus `NO_*` codes only where that kind of evidence is absent.
  - `Coverage` states which evidence exists. `faithfulness_evaluated` and `concepts_validated` are `False` and cannot be set otherwise.
  - `render()` is deterministic text. `to_dict()` is a deterministic presentation summary; the codec remains the serialisation.
- **No computation:** composition and rendering never execute the model, use autograd or Captum, install hooks, touch RNG, or write tensors. This is verified with a tripwire model and patched autograd/hook entry points.
- **No persistence format:** an explanation is reconstructed from its persisted traces by composing again.
- **Naming:** the top-level function is `compose`, not `explain`, because `beyondnn.explain` is the subpackage: importing it would rebind a top-level `explain` function.

**Not done (by design):**
- no confidence, agreement, or combined score;
- no ranking or "most important";
- no `EVIDENCE_CONFLICT` for attribution/intervention differences (they answer different questions);
- no generated text or claims;
- no multi-target or multi-sample views;
- no faithfulness, sufficiency, or concepts (Phases 5 and 6).

---

## ADR-032: Unit-level and model-input interventions, and comparison families, as Phase-2 extensions

- **Date:** 2026-09-25
- **Status:** Accepted for Phase 5 implementation. Awaiting Phase 5 review.

**Problem:** faithfulness protocols perturb *parts* of a tensor (the top-k attributed input features, selected internal units) and need many perturbations per input, plus random controls. Phase 2 could only replace a whole module-output leaf, one intervention per recording. Recording unit-level perturbations as opaque exact-shape CONSTANT tensors would hide what was selected. Building a second perturbation engine for faithfulness would duplicate the pairing and refusal logic.

**Decision** (extensions, not a new engine):
- **`InterventionRecord` v2** adds `units: tuple[int, ...] | None` (indices along the leaf's last dimension; sorted, unique, non-empty) and `retain: bool`.
  - With `retain=False` exactly the units are replaced (removal).
  - With `retain=True` every *other* unit of the leaf is replaced (retention within that leaf only).
  - `units=None` is the Phase-2 whole-leaf replacement.
  - The replacement values come from the operation: ZERO, CONSTANT (a scalar or exact-shape tensor), or PATCH (source activation).
  - v1 payloads migrate with `units=None, retain=False`; nothing is invented.
- **Model-input interventions.** The site may be a positional model input leaf (`module=""`, `io=INPUT`, `args[i]`; ZERO or CONSTANT only). The intervention pass runs the model on the perturbed input.
  - It is still a paired, CLEAN-baseline vs INTERVENTION comparison, with every Phase-2 refusal.
  - The intervention pass's `InputRecord.sample_id` identifies the perturbed input exactly, so the perturbation is verifiable.
  - The effect is an INSTANCE `CausalEffect` about the original input: do(x_S := b). Root *outputs* can never be intervened on.
- **`compare_family(model, groups, metric, *, extend=...)`** runs all comparisons in **one recording**: one CLEAN baseline pass per input, then one INTERVENTION pass per intervention, each checked against that baseline.
  - It returns `ComparisonFamily` (effects and records per group).
  - `extend` adds records to the same trace during finalisation (used by faithfulness runs).
  - N controls therefore cost N passes and one trace, not N traces.
- Builders: `zero(..., units=, retain=)`, `constant(...)`, `patch(...)`, `zero_input(index, units=, retain=)`, `constant_input(value, index, units=, retain=)`.

**Consequences:**
- Perturbations are explicit in the record: which units, remove vs retain, and replacement values (digest).
- A unit is a last-dimension index. Leaves with non-singleton leading dimensions are perturbed at those units across all leading positions; faithfulness selection refuses such tensors (see ADR-033).
- The existing OOD limitations (`ZERO_ABLATION_MAY_BE_OOD`, `CONSTANT_REPLACEMENT_MAY_BE_OOD`) apply unchanged to input-level perturbations.

---

## ADR-033: Phase-5 faithfulness tests: claim protocols, diagnostic results, controls, and no new status

- **Date:** 2026-09-25
- **Status:** Accepted; the "Units" bullet is superseded by ADR-034 (in part).

**Decision:**
- **No new `EvidenceStatus`.**
  - The raw measurements of every faithfulness test are INTERVENTIONAL `CausalEffect`s of perturbations (ADR-032), resting on OBSERVED inputs/outputs and, for selections, ATTRIBUTED records.
  - A test result is not evidence of a new kind; it is a test result.
- **Layering: raw effect → protocol → result → assessment.**
  - `comprehensiveness` v1 decides NECESSARY_FOR/DECREASES claims.
  - `sufficiency` v1 decides SUFFICIENT_FOR claims. It is the first protocol that can, and only in its declared sense: retention within one site, with a declared replacement.
  - Both produce ordinary `ClaimTestResult`s. Their statistics are derived from the cited effects; criteria are declared in the spec before running; assessments need explicit policies (`COMPREHENSIVENESS_POLICY`, `SUFFICIENCY_POLICY`).
  - Each evaluator hard-codes its decidable relations, independent of the registry (defence in depth, as in ADR-031).
  - A perturbation that changed nothing gives INCONCLUSIVE, never CONTRADICTS.
- **New non-evidence records:**
  - `EvidenceSelection` records which units were tested: source, rule, full ranking, the scores used, k, seed, and the source attribution id.
  - `ProtocolResult` records diagnostic protocols that never decide claims: removal/retention curves, stability, counterexample and paired-control summaries, method agreement, baseline and IG-step sensitivity. It holds declared params and criteria, derived measurements, and per-aspect outcomes (PASS/FAIL/NOT_APPLICABLE/INDETERMINATE). PASS/FAIL requires a declared criterion for that aspect, and there is never a single score.
  - Diagnostic protocols are a separate registry (`DIAGNOSTIC_PROTOCOLS`).
- **Units** are last-dimension indices. Selection is refused unless every other dimension has size 1; per-position selection needs an explicit reduction and is out of scope. Ties are broken by lower index, and a tie across the k boundary is flagged (`SELECTION_TIE_AT_BOUNDARY`).
- **Controls:**
  - matched random selections: same site, same size, uniform without replacement; random permutations for curves;
  - drawn from a seeded **local** `torch.Generator` (never the global RNG), so they are reproducible and re-derivable;
  - statistics: control fractions below/tied/above, and a one-sided Monte-Carlo p `(1 + b)/(N + 1)` described as "a matched random set does at least as well", never as "significant";
  - dataset level: per-sample paired differences with a seeded sign-flip test.
  - There is no numpy/scipy dependency.
- **Curves:** "deletion" and "progressive ablation" are both `removal_curve`; "insertion" and "progressive retention" are both `retention_curve`.
  - The level is carried by the site. The full curve is always stored.
  - `aopc_mean_drop` is optional and its normalisation is recorded: the mean drop over the declared points, anchors included.
- **Stability:** the caller declares the transformation (name, revision, config, unit map; the function is never serialised, and `DECLARED_TRANSFORMATION_UNVERIFIED` applies).
  - Four aspects are reported separately: prediction, ranking (Spearman on ordinal rankings), top-k Jaccard, and claim outcome.
- **Samples vs datasets:**
  - An instance run is one sample.
  - `run_dataset` runs a declared sample set in **one** trace, with per-sample claims and results. It adds a `counterexample` summary (every CONTRADICTS kept; "held on a of n") and, with controls, a `paired_control` summary. Both link to the per-sample results.
  - Universally quantified claims are not first-class claims; the counterexample summary is how "holds on the declared set" is represented.
  - Dataset results are refused in an instance-level WHY.
- **Verification:**
  - every perturbation is checked against the recorded execution: the perturbed input's `sample_id`, or the retained site activation;
  - composition re-derives claim results, including the no-op status and the seed-reproduced controls, as well as selections and diagnostic results from the raw records (`faithfulness.verify`).
- **Coverage:** `faithfulness_evaluated` becomes true only with composed faithfulness results, and `faithfulness_protocols` names exactly which. It never means "faithful"; `NOT_EVALUATED` then lists the protocols not run. `concepts_validated` stays false.
- **Not built:**
  - global faithfulness/aggregate scores (ADR-007; RQ8 deferred, too few ground-truth tasks for a held-out evaluation);
  - ROAR retraining;
  - Quantus/Captum-metric adapters;
  - probes;
  - universally quantified claims.

**Consequences:**
- A passed protocol supports only its claim, under its declared replacement and scope.
- Removal/retention inputs are off-distribution (recorded via the Phase-2 OOD limitations); BeyondNN documents this confound but cannot remove it.
- With few units, matched controls often coincide with the selection, so a perfect selection can have a Monte-Carlo p above 0.05. This is reported, not hidden.


## ADR-034: Declared unit axes for images, channels and token positions

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 5.5. Awaiting Phase 5.5 review.
- **Supersedes ADR-033 in part:** only its "Units" bullet (last-dimension units only). Everything else in ADR-033 stands.

**Observed realistic failure** (Phase 5.5, before any change; `experiments/phase5_5/results/abstraction_probe.json`):
- Model B (trained CNN, digits): `F.top_k` over the pixel attribution `(1, 1, 8, 8)` and over the conv-channel attribution `(1, 16, 4, 4)` was refused ("non-last dimensions").
- A declared pixel/channel selection was refused only **after** every perturbation pass had run (ergonomics).
- Model C (BERT-tiny, SST-2): the same for token positions of the word-embedding output `(1, T, 128)`.
- Interventions could address only the last axis, so "pixel (h, w) in every channel", "channel c everywhere" and "token t (all hidden dims)" were not expressible without reshaping the model (an architecture-specific hack).
- This was predicted in the pre-registration (H8) and confirmed.

**Decision:**
- An optional `unit_axes` (non-negative, sorted, unique tensor axes) is added to:
  - interventions: `Intervention` and every builder; `InterventionRecord` v3;
  - selections: `Selection`, `EvidenceSelection` v2, with `unit_reduction`;
  - claim subjects: `Subject.unit_axes`, `Claim` v2.
- A unit is a **row-major index into the sub-grid of the declared axes**. Every other axis lies within the unit and is perturbed together: pixels `(2, 3)`, channels `(1,)`, token positions `(1,)`.
- `unit_axes=None` keeps the Phase-5 last-axis meaning exactly.
  - Legacy records migrate with `unit_axes=None` and no invented axes (intervention v2→v3, claim v1→v2, selection v1→v2).
  - Phase-5 records created without `unit_axes` keep their content, except `Claim`, whose v2 serialisation adds `subject.unit_axes`. The golden claim id therefore changed, deliberately (`tests/test_record_identity.py`).
- **Scores per unit need an explicit `reduce`** (`sum`, `abs_sum`, `l2`) whenever a unit spans more than one element. Nothing is aggregated implicitly. The reduction is recorded in the `EvidenceSelection` and re-derived by the verifier.
- Diagnostics (`stability`, `method_agreement`, `baseline_sensitivity`, `ig_step_sensitivity`) take the same `unit_axes`/`reduce`. They are recorded in `ProtocolResult.params` only when declared, so Phase-5 diagnostic ids are unchanged.
- **Input units are checked before any pass runs.**
- The faithfulness evaluator is applicable only if the claim's `unit_axes` equal the intervention's, and every control's.
- `attribution_threshold` (Phase 3) remains last-axis only and is NOT_APPLICABLE to a subject with `unit_axes`.

**Alternatives considered:**
- *Flatten in user code* (reshape the input or wrap the model): works only at inputs, not at internal sites. It changes the recorded site, loses the unit ↔ tensor-position link, and is architecture-specific.
- *An arbitrary boolean mask per unit:* maximally general, but not auditable in a record (masks would have to be stored), and it invites overlapping units.
- *A named-dimension API* (for example "C", "H", "W"): readable, but model-specific, and it needs dimension names that PyTorch modules do not carry.
- *Implicit reduction* (for example sum by default): rejected. The reduction changes rankings, and silently choosing one is what ADR-030 forbids for targets.

**Consequences:**
- CNN pixel/channel units and transformer token units are expressible with general abstractions and verified in composition.
- Units are always axis-aligned grids. Superpixels, spans, and other irregular groups remain unsupported (a documented limitation).
- The migration changes the `Claim` serialisation, so claim ids made before Phase 5.5 differ from those made after. A loaded v1 claim is migrated, gets a new id, and references are remapped (as for every migration).

## ADR-035: Perturbation-magnitude-matched random controls

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 5.5. Awaiting Phase 5.5 review.

**Context:** Count-matched controls (ADR-033) answer "does the selection beat a random set of the same size?". On trained models, input × gradient and similar methods favour units with large |x − b|. A selection can beat count-matched controls just because it perturbs the input more (the ROAD/Blücher confound; `docs/research/PHASE_5_5_LITERATURE.md`).

**Decision:**
- `faithfulness.controls(n, seed=, match="magnitude", strata=4)`:
  - each unit's perturbation magnitude is ‖x_u − b_u‖₂ over its within-unit elements, where b is the declared replacement at the same site;
  - units are split into `strata` equal-count strata by magnitude (descending, ties by lower index);
  - each control replaces every selected unit by a uniform draw, without replacement, from the same stratum;
  - draws come from the seeded local generator, as in ADR-033.
- The per-unit magnitudes and `strata` are recorded in the spec's `controls` identity (strategy `perturbation_magnitude_stratified_same_site_same_size`).
- **Verification re-derives the magnitudes** from the retained clean site tensor and the recorded replacement, then re-draws the controls. Forged magnitudes are refused.
- For internal sites the magnitudes come from one public traced pass (`bnn.trace`, CPU retention).
- `count` remains the default.
- Magnitude-matched controls are refused for curves (random rankings have no per-point stratum).

**Alternatives considered:**
- *Stronger nulls with more structure* (random sets matched on attribution-score percentiles): circular, because they condition on the method under test.
- *Spatially contiguous random masks* (RISE-like): relevant for pixels only, and not general across A/B/C.
- *ROAR retraining:* out of scope. It is CPU-heavy and changes the model under test (literature §ROAR).
- *Distribution-matched replacement* (ROAD noisy linear imputation): addresses a different confound (OOD), handled by the replacement sensitivity (r1–r3), not by controls.

**Consequences:**
- The magnitude null is available wherever count controls are, at the cost of one extra traced pass for internal sites.
- Magnitudes depend on the replacement. With a zero replacement at a standardised input they equal |x_u|.
- A selection that beats count-matched controls but not magnitude-matched ones is reported as such; neither result is hidden.

## ADR-036: Attribution reproducibility tolerance scaled to output precision

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 5.5. Awaiting Phase 5.5 review.

**Observed realistic failure** (Phase 5.5, model B, trained CNN on digits, held-out sample 2 of 60):
- `attribute(..., method=gradient())` raised `StochasticAttributionError`: "the target is 0.8396041989326477 in the traced pass but 0.8396055698394775 in the attribution passes; not reproducible".
- The model is deterministic. The grad-enabled and no-grad forwards differ by 9.5e-7 on logits of magnitude 11.6, about one float32 rounding unit, because different conv kernels run.
- The target is the margin (a difference of two logits). Cancellation turns that into a 1.6e-6 *relative* error, above the Phase-3 `rel_tol=1e-6`.
- A realistic, correctly classified, low-margin sample was therefore refused as "stochastic". This is a BUG (a false refusal).

**Decision:**
- The traced-pass and attribution-pass targets must agree within the larger of:
  - 1e-6 relative to the target, as before;
  - 16 × `finfo(dtype).eps` × the largest finite output magnitude (`attribution.runner.reproducibility_tolerance`).
- The refusal message states the tolerance.

**Alternatives considered:**
- *Run the reference pass with grad enabled:* removes this kernel difference only. Other benign differences, such as batch-size-dependent kernels in Captum's internal batching, would remain. It also changes what the OBSERVED reference pass is.
- *A user-settable tolerance:* invites silencing a real stochasticity signal per call, and adds a parameter every caller has to reason about.
- *Compare the output tensors instead of the target:* the attribution passes expose only the target value, not the full output. It would need a larger change to the native/Captum engines.
- *A larger fixed relative tolerance* (e.g. 1e-4): still wrong for small margins, which are exactly the realistic cases, and too loose for large targets.

**Consequences:**
- Rounding-level kernel differences are accepted.
- State drift and randomness still change the output by orders of magnitude more than 16 ulps and are still refused (`test_state_and_randomness_are_refused_not_averaged`).
- A drift of 1000 ulps is refused (`test_rounding_level_kernel_differences_are_not_randomness`).
- A model whose randomness is below 16 ulps of its largest output would no longer be detected by this check. The RNG-state check is unchanged and still detects the use of the global generator.

## ADR-037: Keyword model inputs in faithfulness diagnostics

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 5.5. Awaiting Phase 5.5 review.

**Observed realistic failure** (Phase 5.5, model C, BERT-tiny, SST-2):
- `method_agreement` and `ig_step_sensitivity` were refused on all 40 held-out sentences (160 refusals): "the attributions are not about this input".
- The attributions were computed with `model_kwargs` (`attention_mask`, `token_type_ids`), and the sample identity includes them (ADR-031). The diagnostics could not be given the kwargs at all, unlike `run` and `curve` (API review F-11).

**Decision:**
- `stability`, `method_agreement`, `baseline_sensitivity` and `ig_step_sensitivity` take `model_kwargs`.
- They are passed unchanged to every attribution, claim test and anchoring pass, and to the sample-identity check. A stability transformation applies to `x` only.
- Diagnostic parameters are unchanged, so existing diagnostic record ids are unchanged.

**Alternatives considered:**
- *Drop keyword inputs from sample identity:* breaks ADR-031 (exact input identity), because two different masks would look like the same input.
- *Callers wrap the model to bind the kwargs:* changes the model under test and its fingerprint, and hides an input from the record.
- *Document as a limitation only:* leaves the transformer setting without any library diagnostic, although the fix is the same parameter `run` and `curve` already take.

**Consequences:**
- The diagnostics are expressible for models with keyword inputs. Regression tests: `tests/test_diagnostics_kwargs.py`.

## ADR-038: Magnitude re-derivation within rounding tolerance

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 5.5. Awaiting Phase 5.5 review.
- **Refines ADR-035:** only its verification step. The recorded magnitudes and the exact re-drawing of the controls are unchanged.

**Observed realistic failure** (Phase 5.5, model B, conv-channel site `relu2`, held-out sample 1):
- Composition refused all 15 magnitude-matched results: "declared control magnitudes do not re-derive from the trace".
- The magnitudes are computed from a separate traced pass, taken before the comparison family runs. The conv activation of that pass differs from the family's clean pass by 9.5e-7, one float32 ulp; the magnitudes differ by up to 1.2e-6.
- The records were correct, but verification used exact float equality. This is a BUG in the ADR-035 verification. It is invisible on the MLP and the embedding site.

**Decision:**
- Verification accepts recorded magnitudes that differ from the re-derived ones by at most 16 × `eps(dtype)` × √(elements per unit) × the largest |site| or |replacement| value.
- The controls are still re-drawn **exactly** from the recorded magnitudes, so the tolerance bounds only how far the record may differ from the trace, never which controls were drawn.

**Alternatives considered:**
- *Compute the magnitudes from the family's own clean pass:* the magnitudes are needed to choose the controls before the family runs. That would need two families, and so two different clean passes to reconcile.
- *Record the site tensor instead of the magnitudes:* larger records, and the same rounding question when comparing it with the family's retained activation.
- *Keep exact equality:* rejects correct records on realistic CNNs.

**Consequences:**
- A forged magnitude vector is still refused: 1% inflation and reordering are both refused by tests.
- A forgery smaller than the tolerance, and small enough not to change the magnitude strata, is not detected. Such a forgery cannot change which controls were drawn from the recorded values.

## ADR-039: Three-state semantic lifecycle; concept status is derived, never stored

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 6. Awaiting Phase 6 review.
- **Supersedes ADR-009 in part:** its four-state enum (REJECTED_CONCEPT) and its `ValidationResult` sketch. The separation of semantic and evidence status, the "no automatic promotion" rule, and the requirement of detection plus causal tests with random controls all stand.

**Decision:**
- `SemanticStatus` has exactly three values: `UNLABELED_FEATURE`, `PROPOSED_CONCEPT`, `VALIDATED_CONCEPT`.
- **Where each status comes from:**
  - A `feature` record is UNLABELED_FEATURE.
  - A `concept` record (label, definition, feature, label source) is always PROPOSED_CONCEPT and stores no status.
  - VALIDATED_CONCEPT exists only as the **derived** status of a `concept_validation` record. Construction refuses a status (or list of unmet requirements) that does not follow from the stored assessment summaries under the declared `ConceptPolicy`. The v1 policy cannot be weakened: it requires encoding, at least one use claim, controls, and counterexamples.
- **No global REJECTED state.** A failed validation is a `concept_validation` with PROPOSED_CONCEPT and its unmet requirements (for example "use claim … contradicted"), plus CONTRADICTED assessments. Rejection, like validation, is relative to a scope (dataset, split, intervention, target, checkpoint). Phase 5.5 showed that such scopes change conclusions.
- **Generated labels** are `generated_label` records with status **GENERATED**. They are never evidence (the existing `EvidenceRef` rule), carry the limitation `GENERATED_LABEL_UNVERIFIED`, and can only seed a proposal. Their provenance names:
  - the model the label is *about*, as `model`;
  - the caller-declared generator, as `method` (`generated_label:<generator>`, version = the declared revision). BeyondNN never calls a generator.
- **Validation and evidence status:** `EvidenceStatus.VALIDATED_CONCEPT` is used by exactly one kind, `concept_activation`, created only for a VALIDATED validation on the same checkpoint (ADR-009's rule, kept). Composition checks this.

**Alternatives considered:**
- *Keep REJECTED_CONCEPT:* makes a scoped negative result look global, and invites "rejected forever" readings.
- *A status field on the concept record:* can go stale, and can be set without evidence (the ADR-012 argument).
- *A generated label as a plain string with a flag:* loses provenance and lets the text be quoted as if validated.

**Consequences:**
- Negative results remain visible as data, not as a lifecycle state.
- A generated label's provenance `model` describes the subject of the label, not the generator. This is documented in the record docstring.

## ADR-040: Feature bases, feature subjects, and direction interventions

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 6. Awaiting Phase 6 review.

**Decision:**
- **Feature records.** A `feature` record describes a structural coordinate of one module-output leaf:
  - `basis ∈ {neuron, direction, sae}`, a non-batch feature `axis`, and `pooling ∈ {none, mean}` over the other non-batch axes;
  - the activation rule: neuron = value at `index`; direction = ⟨x, v/‖v‖⟩; SAE = ADR-041;
  - the direction is a retained tensor whose digest is part of the identity, so a changed vector or norm is a different feature;
  - `source ∈ {declared, fit, search, sae}`. Fitted and searched features name the concept dataset, and may only use its **train** split;
  - fitted, searched and SAE features record the `model_state_digest` they were derived on.
- **Feature subjects.**
  - `Subject.feature` (a feature record id) makes a claim about a feature, not about raw units; it excludes `units`.
  - **`Claim` becomes record version 3.** Migration v2→v3 sets `subject.feature = None`. **The golden claim id changes** (`tests/test_record_identity.py`, documented).
- **Directions are not units.** A direction is a vector, not an index. Direction interventions are a new operation rather than another `unit_axes` case:
  - **`InterventionRecord` v4** adds `direction` (a retained 1-D tensor) and `direction_axis`. Migration v3→v4 sets both to `None`.
  - Operation `DIRECTION`, with a reference r (a `value` tensor of the leaf's exact shape, or zeros):
    - **removal** (`retain=False`): x′ = x − ⟨x − r, v̂⟩ v̂, i.e. the coordinate is replaced by the reference's;
    - **retention** (`retain=True`): x′ = r + ⟨x − r, v̂⟩ v̂, i.e. only the coordinate is kept.
  - Applied at every position of the leaf. No units. Emits `DIRECTION_REPLACEMENT_MAY_BE_OOD`.
  - Builder: `interventions.direction(site, v, axis=, reference=, retain=)`.
- **Not implemented:** direction addition and scaling (steering). They are off-distribution by construction and conflate influence with use (literature §3–4).

**Alternatives considered:**
- *Treat a direction as a rotated unit basis (DAS-style):* needs a learned rotation, which is out of scope, and still needs projection semantics to intervene.
- *Encode the feature id in `Subject.site` or in the target:* ambiguous, and composition could not refuse a wrong feature.
- *Arbitrary callable interventions:* not serialisable, and not verifiable.
- *Only removal:* cannot express the sufficiency side, which is needed to expose "sufficient but redundant" (plan §20).

**Consequences:**
- Neuron features reuse the Phase-2/5.5 unit interventions (`unit_axes=(axis,)`). Direction and SAE features use DIRECTION.
- Old traces load. Nothing is invented by the migrations.

## ADR-041: SAE features through a tensor-only adapter

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 6. Awaiting Phase 6 review.

**Decision:**
- `concepts.sae_feature(model, site, encoder=, decoder=, b_enc=, b_dec=, latent=, checkpoint=, reconstruction=)` takes raw tensors. There is no SAELens or other import, and BeyondNN does not train SAEs.
- The record (`SAEIdentity`) stores:
  - the checkpoint token;
  - the latent index and dictionary size;
  - the relu activation rule;
  - the retained encoder column and decoder bias, and the scalar encoder bias;
  - the caller-reported reconstruction statistics (unverified);
  - the decoder row, as the feature's direction.
- **Activation:** relu(⟨x − b_dec, W_enc[:, i]⟩ + b_enc[i]), pooled.
- **Intervention:** along the **normalised decoder direction** (projection semantics, ADR-040). This is *not* SAE-native clamping or ablation, and is recorded as `SAE_INTERVENTION_IS_PROJECTION`.
- **Limitations always attached:** splitting, absorption, polysemanticity not excluded, reconstruction error, missing features.

**Alternatives considered:**
- *A SAELens dependency:* heavy, and the core would track a fast-moving API.
- *SAE-native ablation* (subtract a_i·d_i, keep the error term): needs the full SAE forward inside the hook. It is a candidate for a later adapter.
- *Trusting SAE features as monosemantic:* rejected by the literature (Chanin 2025; Kantamneni 2025; Wu 2025).

**Consequences:**
- An SAE latent is validated exactly like any other feature, with no privileged status.

## ADR-042: Concept claim protocols with mandatory controls, and concept validation

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 6. Awaiting Phase 6 review.

**Decision:**
- **`concept_encoding` v1** decides ENCODES only:
  - The observed statistic is the AUROC of the feature's fixed 1-D readout (declared sign) on the concept dataset's held-out **test** split. The evidence is MEASURED activations from one clean recording; nothing is trained at evaluation time.
  - **At least one control with a declared criterion is required** (`min_fraction_below` on the superiority over each control):
    - random directions with a declared distribution (`isotropic` or `covariance`), scored sign-free;
    - random neurons, scored sign-free;
    - permutations of the held-out labels.
  - A diagnostic `concept_counterexamples` result lists every held-out false positive and false negative at a threshold fixed on the **val** split.
  - Limitations `DECODABILITY_NOT_USE` and `FEATURE_MAY_CARRY_OTHER_INFORMATION`.
- **`concept_intervention` v1** decides:
  - DECREASES/INCREASES under **removal** of the feature;
  - SUFFICIENT_FOR under **retention** (site-relative).
  - The intervention and its reference are **required** (no default).
  - The statistic is the FINITE_SAMPLE mean effect over a declared evaluation subset (by default the concept-positive test samples), from one Phase-2 comparison family.
  - **At least one random-feature control with a declared criterion is required** (`min_fraction_beyond_controls`).
  - Limitations: `SINGLE_FEATURE_TEST_MISSES_REDUNDANCY` and `DIRECTION_INTERVENTION_MAY_ACTIVATE_DORMANT_PATHWAYS`.
  - It is the second protocol that can decide SUFFICIENT_FOR, and only in its declared, site-relative sense.
- **Assessments** use explicit policies: `concept_encoding_v1` and `concept_use_v1`.
- **`concepts.validate(concept, encoding=, use=[...], additional=[...], policy=POLICY_V1)`** runs nothing:
  - it refuses results about another concept, feature, dataset or checkpoint;
  - it summarises the assessments and builds the `concept_validation` record (ADR-039);
  - its scope string names the dataset, split and size, the model digest, the site and axis, and each claim's relation, target and intervention.
- **Per-sample targets are not needed** (plan §10). Use claims are stated over concept-conditioned subsets with one fixed target.
- **No concept score** (ADR-007).

**Alternatives considered:**
- *Trained probes* (logistic, MLP) at evaluation time: add flexibility that must then be controlled (Hewitt & Liang). A fixed readout plus train-only fitting avoids it.
- *TCAV scores as use evidence:* gradients are not interventions.
- *Optional controls:* Phase 5.5 F-21 showed random selections passing absolute criteria.
- *Requiring only encoding for VALIDATED:* contradicts the central rule (decodability is not use).

**Consequences:**
- A decodable-but-unused feature shows ENCODES SUPPORTED and use CONTRADICTED, and stays PROPOSED_CONCEPT.
- Validation is scoped; it never generalises.

## ADR-043: Per-kind and per-pass indexes in `TraceResult`

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 6. Awaiting Phase 6 review.

**Observed failure:**
- A concept use test on the toy model (88 samples × 51 interventions, about 4,600 passes in one comparison family) took 54 s.
- 85% of the time went to `TraceResult._of` and the engine's pass lookups, which scan every record on every call: O(N²) in the number of passes.
- Phase 5.5 traces (≤ 52 passes) never exposed it.

**Decision:**
- `TraceResult` keeps an insertion-ordered list per record type, and a (type, pass) index for input and output records, both maintained in `_add`.
- `_of` returns the per-type list when exactly one registered type matches (the same result and order as the scan), and falls back to the scan otherwise.
- New `input_of_pass` / `output_of_pass` replace the engine's linear lookups.

**Alternatives considered:**
- *Cache invalidated on every add:* the engine interleaves adds and reads, so it stays quadratic.
- *Smaller concept tests:* hides the problem, and realistic tests need thousands of passes.

**Consequences:**
- The same use test takes 6.6 s; the cost is now linear, about 1.4 ms per paired pass on the toy model.
- The results are unchanged: the full suite passes unmodified.

## ADR-044: The audit classifies claims from re-derived records, with standings and findings and no score

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7. Awaiting Phase 7 review.

**Context:**
- Phases 1–6 produce typed, provenance-bound, re-derivable evidence, but nothing reads a whole body of it.
- Phase 6's validations were instance-WHY context only (P6-5).
- Users need to know which claims the recorded evidence establishes, which it contradicts, and where the answer depends on an assumption or a missing test.
- ADR-007 forbids a global confidence scalar.

**Decision:**
- `bnn.audit(evidence, plan=plan)` is a deterministic, **model-free** function of recorded traces and a declared plan (ADR-045).
- **Evidence handling.** Every trace is integrity-checked, and every claim-test result is re-derived from its cited records with its registered protocol. Evidence that fails, or that is out of the plan's checkpoint, declared model, sample or dataset scope, is **excluded and reported**, never merged or corrected.
- **Standings.** Each plan claim or concept receives one of 7 standings (SUPPORTED, CONTRADICTED, MIXED, ASSUMPTION_SENSITIVE, INCONCLUSIVE, UNSUPPORTED, NOT_EVALUATED), derived by the fixed precedence in `docs/PHASE_7_PLAN.md` §22 and deviation D1. The verdict part reuses `derive_verdict`.
- **Findings.** 13 finding kinds, each with a closed-vocabulary code and a categorical severity (BLOCKING / QUALIFYING / INFORMATIONAL). Severities are never summed or compared.
- **Disagreement is explained, never resolved.** SUPPORTS/CONTRADICTS pairs are explained by the assumption axes (protocol, threshold, replacement, k, null, method, dataset) they differ in. A disagreement without an axis is MIXED.
- **Per-sample claims** yield a distribution and counterexample identities, never a claim-level truth value.
- **Missing evidence** is NOT_EVALUATED.

**Alternatives considered:**
- *A weighted evidence score:* violates ADR-007 and hides which assumption a conclusion rests on.
- *Auto-resolving disagreements by majority or by protocol priority:* hides exactly the finding the audit exists to report.
- *Auditing live result objects only:* ties audits to one Python session (P6-4); the audit reads traces.

**Consequences:**
- Structural overclaims are caught mechanically: attribution → causal, probe → use, generated → validated, narrower estimand, untested invariance, missing controls.
- An audit can only see what was recorded and supplied. Selective *recording* is invisible; only an untested *declared* invariance is visible.

## ADR-045: `AuditPlan` is a content-addressed record with no invisible defaults

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7. Awaiting Phase 7 review.

**Decision:**
- `AuditPlan` (record kind `audit_plan` v1) declares:
  - the checkpoint digest and the declared model;
  - the sample ids and concept-dataset ids;
  - the claims (formal structure; subject **or** a selection by an attribution method; scope; named requirement; asserted invariances);
  - the evidence requirements (an assessment policy, whether controls are required, and alternative criteria for threshold sensitivity);
  - the concepts asserted to be validated;
  - the counterexample caps and the naive-AUROC reference.
- **No defaults.** Every constructor argument is required. `None` means "not declared", and the report states it (for example "no cap declared").
- **Identity.** The plan's id covers everything it declares, and the report embeds the full plan.
- **Refusals.** A causal claim whose requirement names no protocol for its relation is refused at construction. Unregistered or unjustified policies, and alternatives that cannot be re-evaluated (`concept_encoding`), are refused by the audit.

**Consequences:**
- Plans are serialisable with the schema codec and can be pre-registered, committed and re-used verbatim.
- A dishonest plan (one that declares no invariance, say) is not detected, but it is visible to every reader of the report.

## ADR-046: Trace-level re-derivation makes audits reload-safe (partial fix of P6-4)

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7. Awaiting Phase 7 review.

**Context:**
- Phase-6 verification took live result objects (`EncodingResult`, `UseResult`, `ConceptValidationResult`). Those cannot be rebuilt from saved traces (P6-4).

**Decision:**
- `concepts.verify` gains `verify_encoding_trace`, `verify_use_trace`, `verify_validation_trace(trace, locate)` and `verify_feature_record(record, recording, dataset)`. They use records and retained tensors only; the object-level verifiers delegate to them.
- The audit re-derives a fitted or searched feature from **any** supplied recording holding the train-split activations. Encoding traces do; if none does, the audit reports `feature_derivation_not_rederived`.
- `AuditReport.to_dict()` is deterministic.
- `verify_report(document, evidence, plan)` re-runs the audit and raises `AuditMismatchError` (listing paths) on any difference; it never corrects.
- `report.save` is atomic and never overwrites.

**Consequences:**
- The loop "save traces → restart → load → audit" gives byte-identical reports, without recomputation.
- `compose` still needs live Phase-6 objects; P6-4 remains open there.

## ADR-047: The WHY shows an AUDIT section, restricted to the reference input

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7. Awaiting Phase 7 review.

**Decision:**
- `bnn.compose(..., audit=report)` accepts one `AuditReport`. It **refuses** a report about another checkpoint (`ModelMismatchError`), and one whose per-sample claims do not declare the reference sample (`SampleMismatchError`).
- `why.audit` gives an `AuditView`:
  - the per-sample standing and findings of every per-sample plan claim for this input, with the distribution across the declared samples;
  - finite-sample and population claims, and concept audits, as dataset-scoped context.
- `render()` adds an AUDIT section, which states that the standings come from the audited records, not from the evidence composed into this WHY.
- `to_dict()` gains an `audit` key only when an audit is composed, so earlier outputs are unchanged.

**Consequences:**
- An instance explanation can show, for example, "IG top-k necessary on this input: ASSUMPTION_SENSITIVE (replacement)" next to its own evidence, without merging the two.

## ADR-048: Declared configuration roles, sensitivity profiles, and configuration-level disagreement

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7.5. Awaiting Phase 7.5 review.

**Context:**
- In Phase 7, ASSUMPTION_SENSITIVE collapsed "supported in 20 of 21 configurations" and "in 1 of 21".
- Every configuration counted equally, so a claim had to survive every stress test.
- The standing-level necessity/sufficiency check fired on 0 samples, while configuration-level disagreements were common (Phase 7 report §32).

**Decision:**
- **Roles.** `RoleRule(axis, pattern, role, sample=None)` on `AuditedClaim.roles` and `AuditedConcept.roles` (`audit_plan` v2, with a migration: v1 means no roles).
  - Roles are PRIMARY, ALTERNATIVE or STRESS_TEST, and patterns are `fnmatch` globs over axis keys.
  - A configuration's role is the worst over the declared axes; on a single axis, the best matching rule counts. A value on a declared axis that no rule matches makes the configuration UNDECLARED (listed, never in a standing).
- **Standings with roles.**
  - The standing and verdict use PRIMARY configurations only.
  - ALTERNATIVE reversals of the primary conclusion are `alternative_reverses` (QUALIFYING); STRESS_TEST reversals are `stress_test_reverses` (INFORMATIONAL).
  - If no PRIMARY configuration was tested, the result is `primary_untested`.
  - Claims without roles keep Phase-7 behaviour.
- **`SensitivityProfile` per group and per concept.** It holds the tested configurations (id, role, outcome, axes); counts by role and outcome; supporting, contradicting and inconclusive configuration ids; sensitive and stable axes; minimal reversals; and untested declared values.
- **Configuration-level PROTOCOL_DISAGREEMENT** (`configuration_level_disagreement`): configurations of two claims with the same subject/selection and target that agree on replacement, k, method and dataset, where one SUPPORTS and the other CONTRADICTS. It is reported whatever the standings are.
- **Replacement names.** `faithfulness.replacement(tensor, name=...)` labels a replacement whose tensor differs per sample, so a role can refer to it. The identity is unchanged without a name.
- **Report format version 2.**

**Alternatives considered:**
- A robustness fraction or threshold ("≥ 80% of configurations"): rejected by ADR-007; the counts are descriptive only.
- Requiring survival of every configuration: that was Phase 7's behaviour, and it confounded plan breadth with evidence.

**Consequences:**
- A claim reads "SUPPORTED under the pre-registered primary configuration; reversed by these alternatives; fails these stress tests".
- Roles must be declared before evaluation. They are part of the plan id.

## ADR-049: Uncertainty statements for aggregate audit quantities

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7.5. Awaiting Phase 7.5 review.

**Decision** (`beyondnn.audits.uncertainty`; literature in `docs/research/PHASE_7_5_LITERATURE.md` §1):
- `wilson(k, n)` for proportions (Brown, Cai & DasGupta 2001), exact at k = 0 and k = n;
- `bootstrap` (percentile) and `paired_bootstrap` over per-sample values, with at least 1,000 draws and a required integer seed (Efron & Tibshirani; Koehn 2004);
- every `Interval` records its quantity, estimate, bounds, level, method, resampling unit, n, k, draws and seed;
- per-sample claims carry Wilson intervals for each standing's share of the declared samples.

**Not a confidence:** an interval describes sampling variability over inputs drawn like the declared samples. It is never a confidence in an explanation, and intervals are never combined into a score.

## ADR-050: Concept validations from saved traces; explicit protocol versions

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7.5. Awaiting Phase 7.5 review.

**Context:** `compose` needed live Phase-6 result objects (P6-4). A saved concept validation could be audited (ADR-046), but not shown in a WHY after a restart.

**Decision:**
- **`concepts.load_validation(traces)`** rebuilds a `ConceptValidationResult` from saved traces (objects or paths), after re-deriving everything (`verify_validation_trace`; the fitted feature from the encoding trace).
  - The rebuilt dataset has no inputs, and its references have no tensors.
  - `encoding_test`, `use_test` and `Reference.identity()` refuse to run with them.
- **`PROTOCOL_VERSIONS`** declares the implemented version of each claim-test protocol. Evidence recorded under another version is excluded from audits (`unsupported_protocol_version`) instead of failing re-derivation opaquely.

**Consequences:**
- The loop "run → save → new process → load → audit → compose WHY" works with no live objects. P6-4 is closed for concept validations.
- Other Phase 3–5 results were already composable from traces through their own result classes' `trace` attribute.

## ADR-051: A builtin `margin` metric (predicted class vs best other class)

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7.5. Awaiting Phase 7.5 review.

**Context:** on the InterpBench development cases, claims of the form "node n is necessary for the prediction" were operationalised with `difference(pred, runner-up)`.
- When an intervention moved the argmax to a *third* class, the margin to the clean runner-up stayed positive, and the test read CONTRADICTS: 14 of 32 ground-truth-necessary head instances in case 7.
- Caller metrics cannot be attribution targets (no differentiable form), and selections must share their test's target.

**Decision:** `interventions.metrics.margin(index, path="")` is the element at `index` minus the largest other element on the same last axis. It is builtin and differentiable, and "drop ≥ clean margin" means exactly "the declared class is no longer the argmax".
- `difference` is unchanged and remains the right metric for a declared pair.

**Consequences:**
- New Phase-7.5 experiments use `margin`.
- The Phase-7 re-runs keep `difference` for comparability. On the 10-class model B this can under-detect prediction changes, and it is reported as a limitation.

## ADR-052: Faithfulness replacements are always declared (no implicit zero)

- **Date:** 2026-09-26
- **Status:** Accepted for Phase 7.5 (API freeze review). Awaiting Phase 7.5 review.

**Context:**
- Phase-5.5 API review F-22: `comprehensiveness`, `sufficiency` and `curve` defaulted to the zero replacement when none was given.
- Phase-7.5 development evidence (InterpBench cases 7 and 13) shows why that default is dangerous. Under zero ablation, non-circuit nodes appeared necessary on 63 of 381 (case 7) and 31 of 370 (case 13) ground-truth-not-necessary instances, against 0 under the benchmark's resample ablation.

**Decision:** `replacement` is a required keyword of `faithfulness.comprehensiveness`, `faithfulness.sufficiency` and `faithfulness.curve`. Passing `None` raises `TypeError` with guidance.

**Consequences:**
- A breaking change before v0.1. 22 test, benchmark and Phase-5.5 script call sites now pass `F.zero()` explicitly.
- Recorded identities and results are unchanged (zero was the default).

## ADR-053: Declared unit eligibility for selections and selection claims

- **Date:** 2026-09-29
- **Status:** Accepted for Phase 7.75 (scientific fix under the API freeze; additive).

**Context:**
- Phase 7.5, model D (BERT-base SST-2): IG top-k contained [SEP] on 30/40 samples and on 10/11 PRIMARY-supported necessity samples. A direct probe showed that [SEP] perturbation moves D's margin more than an interior token does.
- "The top-k tokens are necessary" therefore mixed two claims:
  - "the most important tokens, model-control tokens included";
  - "the most important lexical/content tokens".
- Nothing in a selection, a control draw or an audit plan recorded which one was meant, and random controls were always drawn from every position.

**Decision:**
- **Runtime selections:** `faithfulness.ranking` / `top_k` / `units` take optional `eligible=` units and an `eligibility=` name (e.g. `"content_tokens"`), declared together by the caller. Nothing BERT-specific lives in the framework: the caller supplies the mask (e.g. from the tokenizer's special ids).
- **Rankings** order only eligible units. Declared units must be eligible. Count- and magnitude-matched **controls are drawn from the eligible units only**. Curves refuse eligibility-restricted rankings.
- **Selection records:** `EvidenceSelection` v3 adds `eligible` / `eligibility`, with a v2 → v3 migration (`None`: every unit). The claim-test spec records `eligible` / `eligibility` in `params` only when declared, so specs without eligibility are unchanged. Re-derivation (rankings, control sets) uses the eligible population.
- **Audit plans:** `audits.selection(..., eligibility=)`; `SelectionSubject.eligibility`; `audit_plan` v3 with a v2 → v3 migration (`None`).
- **Matching:** an all-units claim and an eligible-units claim are different claims.
  - Evidence counts only for the claim with the same eligibility.
  - Evidence that matches except for eligibility yields a QUALIFYING `eligibility_mismatch` finding. It is never used silently.
  - A spec whose declared eligibility differs from its selection's is an integrity failure.
- **Special tokens are never removed automatically.** An all-units claim still includes them.

**Consequences:**
- Additive: every existing call and record means what it meant before, with new selection ids after the version bump (as ADR-034).
- Phase-7.5 evidence remains valid for the all-units claim.

## ADR-054: The audit treats unattainable control criteria as uninformative, and separates "effect present, not competitive" from "no effect"

- **Date:** 2026-09-29
- **Status:** Accepted for Phase 7.75 (scientific fix; audit semantics only, no evidence format change).

**Context:**
- Phase 7.5, EH4: the count-matched null rejected 98.6% of known-necessary InterpBench heads.
- **Diagnosis** (independent of any outcome): `uniform_subsets` draws random sets from *all* units, including the selected set. A draw identical to the selection ties with it, and ties never count as "below".
  - For one head among 4 at a layer, about 25% of draws are the selected head itself.
  - The maximum attainable fraction below is then about 0.75 < 0.95. The criterion was unmeetable by any model, yet the audit reported CONTRADICTS.
- **Development confirmation:** on InterpBench dev case 7, all 32 count-null tests of known-necessary heads are unattainable by this criterion.
- The request (§23) also requires that failure against a *competitive* control is never equated with failure of the causal effect itself.

**Decision:**
- **Attainability:** for each faithfulness result with controls, the audit computes the fraction of control sets identical to the selected set, from the recorded intervention units. The maximum attainable fraction is `1 - identical / n_controls`.
  - If that is below the declared `min_fraction_below` / `min_fraction_above`, a CONTRADICTS outcome is **INCONCLUSIVE** for standings, profiles and verdicts (also for re-evaluations under alternative criteria).
  - The group gets a QUALIFYING `control_criterion_unattainable` finding.
- **Competitive failure:** a PRIMARY CONTRADICTS whose declared absolute effect criterion *was* met, and which failed only its control criterion, keeps the standing (the claim was declared "beyond controls"). It gets a QUALIFYING `effect_without_competitive_advantage` finding: the effect is present, and matched random sets do as well.
- **No new control ontology** (negative / competitive / stress) is introduced. The declared controls are all competitive, and this finding says exactly what their failure means.

**Consequences:**
- Recorded evidence and faithfulness outcomes are unchanged; only audit reports change.
- Phase-7.5 audit reports stay as recorded (immutable). Affected audits are re-run in Phase 7.75, and the changes are listed in `PHASE_7_75_SCIENTIFIC_FIXES.md`.

## ADR-055: Concept use criteria are declared in the target's own scale

- **Date:** 2026-09-29
- **Status:** Accepted for Phase 7.75 (policy decision; no API change).

**Context:**
- Phase 7.5 rejected the known-used Tracr variable (case 39): use effect about 0.06 < `min_change` 0.1.
- Case 39's target is a *fraction* whose train-split values are small, so the absolute 0.1 was barely attainable on that target's scale.
- Phase 6 used `min_change` 0.25 and Phase 7.5 used 0.1, with no scale argument for either.
- **Calibration** on constructed models (`experiments/phase7_75/concept_calibration.py`; no Tracr or held-out model): rescaling the output by 0.01–100 changes the absolute rule's verdict for the same used concept, both ways.

**Decision:**
- **Rule:** `concepts.use_criteria(min_change=...)` stays the API. Every Phase-7.75 (and later) concept policy declares `min_change` **in target units, derived before the use test**: `min_change = 0.2 × SD(clean target over the concept dataset's train split)`.
- **Why 0.2:** Cohen's conventional "small" standardised effect. The use test's claim is *use*, not *strong use*, and specificity is carried by the mandatory control criterion (≥ 95% beyond random directions). The value is chosen from the literature convention and the calibration grid, **not** from any Tracr result.
- **Operational meaning:** in the calibration, uses with weight ≥ 0.5 relative to unit background variation are validated; weight ≤ 0.25 is not.
- **The derivation is recorded** with each result (the train SD and the resulting `min_change`).

**Consequences:**
- The Phase-7.5 concept results remain as recorded under the absolute rule, and both rules are reported in Phase 7.75.
- The Phase-7.5 held-out case 39 cannot confirm this rule, because its effect was known when the rule was chosen. Confirmation uses new held-out TD programs.
