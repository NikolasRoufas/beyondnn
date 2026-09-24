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
