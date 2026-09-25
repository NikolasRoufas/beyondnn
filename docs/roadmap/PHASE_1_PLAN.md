# Phase 1 Implementation Plan: Foundations

Status: **planned, not started.** Implementation begins only after the architecture review is approved.

**References:**
- `../design/ARCHITECTURE_PROPOSAL.md`
- `../design/TRACE_SCHEMA_PROPOSAL.md`
- `../decisions/ARCHITECTURE_DECISIONS.md`

**Phase goal:**
- A dependable measurement layer (schema, provenance, hooks, `trace()`/`recording()`, serialisation, `instrument()`, a minimal honest `explain()`).
- Every later record type, through `Assessment`, is defined and round-trip tested.
- Everything is demonstrated identically on MLP, CNN, and tiny-transformer models.

**Scope unit:** approximate library LOC and test LOC. No time estimates.

**Global rules for every milestone:**
- `ruff check`, `ruff format --check`, `mypy --strict beyondnn/schema beyondnn/core`, and `pytest` are green on CPU.
- No architecture-specific branches in `schema/` or `core/` (ADR-010).
- An autouse hook-leak fixture is active from M1.5 onward.
- Every public symbol has a docstring stating what it measures and what it does **not** establish.

---

## M1.0: Scaffolding (done in the repository-organisation step, except as noted)

- **Goal:** installable empty package, CI, and community files.
- **Files:** `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `beyondnn/__init__.py`, `beyondnn/py.typed`, `tests/test_package.py`, and the root community files.
- **Public API:** `beyondnn.__version__`.
- **Internal API:** none.
- **Tests:** `test_package.py` (import, version string is PEP 440).
- **Exit criteria:**
  - `pip install -e ".[dev]"` works.
  - CI is green on Python 3.10–3.14, CPU torch.
  - pre-commit hooks pass.
- **Known risk:**
  - CI has not run yet, because nothing has been pushed.
  - Torch CPU wheels for 3.14 must be available on the pinned index.
- **Scope:** about 150 lines of config. Done except for the first CI run.

## M1.1: Schema types and invariants: **done (approved and closed 2026-09-25)**

- **Goal (revised per ADR-014):** the smallest rigorous schema foundation:
  - the record mechanism (identity, status rules, provenance reference, lineage, registry);
  - the serialisation envelope with strict, versioned decoding;
  - `InputRecord`, `OutputRecord`, `ActivationRecord`, `TensorRef`, `TraceLimitation`;
  - the approved claim types (ADR-012) with the estimand rule (ADR-013).

  Later-phase record types (intervention, effect, attribution, feature, concept) are **not** defined.
- **As implemented:**
  - Files: `beyondnn/schema/{__init__,status,values,base,records,limitations,claims,codec,errors,_types,_canonical}.py`.
  - Tests: `tests/{conftest,test_trace_schema,test_record_identity,test_evidence_status,test_claims,test_estimand,test_limitations,test_serialization}.py`.
  - The concept types and `SemanticStatus` were **not** implemented (deferred to Phase 6 per ADR-014).
  - Implementation choices awaiting review are ADR-016 to ADR-018.
- **Files (original plan):**
  - `beyondnn/schema/__init__.py`
  - `status.py` (EvidenceStatus, SemanticStatus, Severity, Relation, Outcome, Verdict)
  - `records.py` (measurement records, Site, Selector, TargetSpec, intervention/effect types)
  - `concepts.py` (FeatureActivation, Concept, ValidationResult, ConceptActivation)
  - `claims.py` (Subject, Scope, Expectation, ClaimSource, Claim, ClaimTestSpec, ClaimTestResult, Assessment)
  - `limitations.py` (TraceLimitation, a code registry with severities and default messages)
  - `presentation.py` (GeneratedText)
  - `_ids.py` (deterministic id allocator)
- **Public API:** `bnn.EvidenceStatus`, `bnn.SemanticStatus`, `bnn.Relation`, `bnn.Outcome`, `bnn.Verdict`, `bnn.Site`, `bnn.TargetSpec`, `bnn.Claim` (plus its parts), `bnn.TraceLimitation`. Record classes are importable from `beyondnn.schema`.
- **Internal API:** `IdAllocator(prefix_counts)`, `LIMITATION_REGISTRY: dict[str, LimitationDef]`, `canonical_hash(obj) -> str`.
- **Tests:**
  - `test_trace_schema.py`: construction, immutability, `kind` uniqueness, and every invariant violation raises `SchemaError`. That includes:
    - causal SUPPORTS without a CausalEffect;
    - SUPPORTED with missing required tests;
    - `VALIDATED_CONCEPT` without validation ids;
    - `VALIDATED_CONCEPT` ConceptActivation on an unvalidated concept;
    - unknown limitation code;
    - `spec_hash` mismatch.
  - `test_concepts.py` (schema part).
  - `test_feature_concept_distinction.py`: no function in the package converts a `FeatureActivation` into a `Concept`. The test introspects the public API for any callable returning `Concept` besides constructors, plus the planned `concepts.validate`.
- **Exit criteria:** every invariant listed in the schema doc has a failing-case test, and the schema package imports without importing `torch.nn`.
- **Known risk:**
  - Invariants that need *other* records (e.g. "produced contains a CausalEffect") cannot be checked in a single record's `__post_init__`. They are split into local checks (`__post_init__`) and cross-record checks (`TraceResult.add`, M1.6).
  - Designing claim types before any runner exists may need revision in Phase 2. That is accepted, and handled by bumping `schema_version` to 0.2 if needed.
- **Scope:** about 600 LOC library, about 500 LOC tests.

## M1.2: Provenance and model identity: **done (approved 2026-09-25 after the buffer-coverage correction)**

- **As implemented:**
  - schema types in `beyondnn/schema/provenance.py`;
  - collection in `beyondnn/provenance/{fingerprint,collect}.py`;
  - tests in `tests/test_fingerprint.py` and `tests/test_provenance.py`;
  - decisions in ADR-019 and ADR-020.
- **Public API** (in `beyondnn.provenance`, not top-level): `fingerprint_model`, `collect_environment`, `capture_cpu_rng`, `make_provenance`, `record_occurrence`, `FingerprintError`. The originally planned `bnn.model_identity` was not added at top level.
- **Timing:** about 1.8 ms per ~1M float32 parameters (experiment log).

**Original plan:**

- **Goal:** every record can answer "how was this produced?".
- **Files:** `beyondnn/schema/provenance.py` (ProvenanceRecord, ModelIdentity, ProvenanceTree), `beyondnn/core/fingerprint.py`.
- **Public API:** `bnn.model_identity(model, label=None) -> ModelIdentity`.
- **Internal API:** `make_provenance(method, params, site, model_identity, derived_from) -> ProvenanceRecord`, `environment_info() -> dict`.
- **Tests:** `test_provenance.py`:
  - the fingerprint is stable across two constructions with the same seed;
  - it changes when one weight element changes;
  - it ignores non-persistent buffers (a documented choice);
  - `origin()` returns the complete `derived_from` chain;
  - versions are populated;
  - timestamps are UTC ISO-8601.
- **Exit criteria:** all of the above pass, and fingerprinting a 1M-parameter model takes under about 1 s on CPU. That number is recorded, not gated.
- **Constraint from M1.1 (ADR-017):** provenance must be able to express execution context, e.g. `execution_mode` (clean / intervention) and `intervention_id`. An activation measured under intervention is still MEASURED; only its provenance says an intervention was active.
- **Constraint from M1.1:** record ids are content-derived and include `provenance_id`. Provenance ids must therefore also be deterministic: derived from content, excluding the timestamp. Otherwise record ids change on every run.
- **Known risk:** full-byte hashing does not scale to billions of parameters. Mitigation (not built now): a `fingerprint="full" | "shapes" | "sampled"` option, with the choice recorded in ModelIdentity.
- **Scope:** about 200 LOC, about 150 LOC tests.

## M1.3: Tiny models: **done (approved 2026-09-25)**

- **As implemented:** `beyondnn/_testing/models.py` (internal namespace, not `beyondnn.models`, not exported) and `tests/test_reference_models.py`. All three models are built inside `seeded_init(seed)`, which forks and restores the CPU generator, so the caller's RNG is unchanged.

| Model | Params | Exists to test |
|---|---|---|
| `TinyMLP` | 139 | `shared` (a `_SharedBlock` with nested `shared.linear`) is called twice per forward, so there are repeated calls at one path; a functional `F.gelu` invisible to hooks; keyword-only `scale` |
| `TinyCNN` | 396 | Conv2d, BatchNorm buffers (train ≠ eval), functional ReLU, `pool`/`gap` modules, nested `stem`/`block`; `return_features=True` gives a `(logits, features)` tuple |
| `TinyTransformer` | 5,120 | embeddings; attention module returning `(out, weights)`; LayerNorm, MLP, residuals; `lm_head.weight is token_embedding.weight`; non-persistent `position_ids` and `causal_mask` used in forward; int token inputs; `return_dict=True` gives `{"logits", "hidden_states"}` |

- **Pinned paths** are listed in the module docstring. They are only paths these models own.
- **Known limitation found:** plain hyperparameters (e.g. `n_heads`) are not fingerprinted. It is pinned by a test (see experiment log and roadmap).

**Original plan:**

- **Goal:** deterministic, CPU-fast test subjects that exercise the edge cases the hook engine must handle.
- **Files:** `beyondnn/models/__init__.py`, `tiny_mlp.py`, `tiny_cnn.py`, `tiny_transformer.py`.
- **Public API:** `bnn.models.TinyMLP`, `TinyCNN`, `TinyTransformer`. Each takes `seed=`.
- **Internal API:** none.
- **Required edge cases:**
  - TinyMLP: an `nn.Sequential` plus one **module called twice** in forward (shared block).
  - TinyCNN: a conv, batchnorm (train/eval difference), pooling, and a **functional op** (`F.relu`) that hooks cannot see.
  - TinyTransformer: attention returning a **tuple** (`out, weights`), a module returning a **dict**, **weight tying** (the same parameter object under two paths), and **keyword-argument** forward (`mask=`).
  - Each model is at most 50k params.
- **Tests:** `test_tiny_mlp.py`, `test_tiny_cnn.py`, `test_tiny_transformer.py` (model-only part): determinism by seed, output shapes, and parameter counts pinned.
- **Exit criteria:** bit-identical outputs across two constructions with the same seed on CPU.
- **Known risk:** these are *engine* test models, not the Phase 2 ground-truth models. Do not conflate the two.
- **Scope:** about 250 LOC, about 150 LOC tests.

## M1.4: Site resolution: **done (approved 2026-09-25)**

- **As implemented:** `beyondnn/core/sites.py` and `tests/test_sites.py`.
  - Public within `beyondnn.core.sites` only: `resolve_sites`, `parse_pattern`, `ResolvedSite`, `SiteResolutionError`, `InvalidPatternError`, `UnmatchedPatternError`.
  - Grammar and semantics are in `docs/design/ARCHITECTURE_PROPOSAL.md` §Module site resolution.
  - `**` does **not** select the root (the original plan below said `""` via exact match only, which is unchanged).
  - `list_sites` / suggestions: `UnmatchedPatternError` includes up to 5 similar paths (via difflib). No separate `list_sites` was added.

**Original plan:**

- **Goal:** turn user site selectors into concrete module paths, handling nesting, sharing, and pytree outputs.
- **Files:** `beyondnn/core/sites.py`, `beyondnn/core/pytree.py`.
- **Public API:** `bnn.list_sites(model) -> list[SiteInfo]` (for discoverability). Site patterns are accepted in `trace`/`recording`.
- **Internal API:**
  - `resolve_sites(model, patterns) -> list[ResolvedSite]`: glob with `*` = one path segment and `**` = any depth. Supports an exact match, `io=`, and output paths.
  - `canonical_paths(model) -> dict[int(id(module)), (path, aliases)]`
  - `flatten(obj) -> list[(path, Tensor)]`
  - `unflatten`
  - Reads optional `__bnn_sites__` (ADR-008), which is otherwise ignored.
- **Tests:**
  - `test_nested_modules.py`: deep nesting, glob semantics, the root module `""`, and a nonexistent pattern raising `SiteNotFoundError` with suggestions.
  - `test_shared_modules.py`: one module under two paths resolves to one canonical path plus aliases, and tied weights don't duplicate.
  - Pytree tests for tuple, list, dict, namedtuple, and dataclass outputs, including non-tensor leaves being skipped.
- **Exit criteria:** all patterns documented in the API doc have tests. No resolution depends on module *types*.
- **Known risk:**
  - Glob semantics must be unambiguous and documented. `**` vs `*` bugs are common.
  - Models that construct submodules lazily at first forward are unsupported in v0. They raise a clear error.
- **Scope:** about 250 LOC, about 250 LOC tests.

## M1.5: HookSession: **done (approved 2026-09-25 after the invocation-guard fix)**

- **As implemented:** `beyondnn/core/hooks.py` (`HookSession`, `HookEvent`, `AliasPolicy`, `AliasSiteAmbiguityError`, `HookSessionError`; internal, not exported) and `tests/test_hooks.py`. Semantics are in ADR-021.

**Original plan:**

- **Goal:** the only place hooks are registered. Leak-proof by construction.
- **Files:** `beyondnn/core/hooks.py`, `tests/conftest.py` (leak fixture, model fixtures parameterised over the 3 tiny models).
- **Public API:** none (internal).
- **Internal API:**
  - `HookSession(model)` as a context manager.
  - `.add_forward(module, fn, *, with_kwargs, always_call=True)`, `.add_pre(module, fn, *, with_kwargs)`.
  - Tracks handles in an `ExitStack`. A session-stack check raises `HookSessionOrderError` on out-of-order close, *after* removing its own hooks.
- **Tests:** `test_hooks.py` and `test_hook_cleanup.py`:
  - normal exit;
  - exception in forward (hooks removed, exception propagates unchanged);
  - exception in the user block;
  - `KeyboardInterrupt` simulated via a raising hook;
  - nested sessions, closed in order and out of order;
  - 1,000 repeated sessions, after which the hook dicts are identical by identity to their pre-state;
  - pre-existing user hooks are preserved and still fire in their original order;
  - `prepend=False` is respected;
  - no reference to captured tensors survives session exit when `retain="none"` (a `weakref` test).
- **Exit criteria:** all pass on all three tiny models. The leak fixture is enabled for the whole suite.
- **Known risk:**
  - Generators and async code that exit a context in another frame. Documented as unsupported.
  - Backward hooks are deliberately not included in Phase 1.
- **Scope:** about 200 LOC, about 350 LOC tests.

## M1.6: `trace()` and `recording()`

- **Goal:** produce a `TraceResult` of OBSERVED and MEASURED records with correct limitations.
- **Files:** `beyondnn/core/trace.py`, `beyondnn/core/result.py` (TraceResult, ActivationView, TensorStore), `beyondnn/schema/tensors.py` (TensorRef, TensorStats).
- **Public API:**
  - `bnn.trace(model, *args, sites=None, retain="summary", seed=None, **kwargs) -> TraceResult`
  - `bnn.recording(model, *, sites=None, retain="summary") -> RecordingContext` (`.result`)
  - `TraceResult.activations`, `.activation(...)`, `.tensor(ref)`, `.origin(...)`, `.by_status(...)`, `.limitations`, `.add(record)`
- **Internal API:**
  - `Recorder` (hook callbacks → pending captures → records at finalisation)
  - `summarise(tensor) -> TensorStats` (computed in float64 on the tensor's device, then transferred)
  - `RetentionPolicy`
- **Behaviour:**
  - `trace` = `recording` + one call.
  - `sites=None` means *all leaf modules* in summary mode. That is documented as potentially expensive, and `PARTIAL_SITE_COVERAGE` is emitted when patterns are given.
  - Mandatory limitations:
    - `FUNCTIONAL_OPS_UNOBSERVED` always;
    - `TENSORS_NOT_RETAINED` under summary/none;
    - `MODEL_IN_TRAIN_MODE` when training.
  - In-place mutation detection compares `tensor._version` at capture vs finalisation. Because `_version` is a private attribute, it is feature-detected: if unavailable, the check is skipped and recorded.
- **Tests:**
  - `test_cpu.py` and `test_tiny_*.py` (trace part): traced output is **bit-identical** to untraced output for all three models;
  - call_index and pass_index correctness;
  - kwargs inputs captured;
  - retention modes (`device` keeps device, `cpu` copies, `summary` holds no tensors);
  - batchnorm in train mode yields the limitation;
  - in-place mutation is flagged;
  - cross-record invariants enforced by `TraceResult.add`.
- **Review item from M1.1:** re-evaluate the five typed reference types (`RecordRef`, `EvidenceRef`, `ClaimRef`, `SpecRef`, `ResultRef`) once real traces exercise them. Keep them unless there are substantial API or maintenance problems.
- **HARD BLOCKER: RESOLVED (2026-09-25, ADR-022)** by caller-declared `ModelDeclaration` in provenance; the `n_heads` regression passes. Original text follows. **HARD BLOCKER (M1.2/M1.3 review):** before M1.6 may create provenance-bearing `TraceResult`s, BeyondNN must decide and implement a **caller-declared model configuration / implementation revision** mechanism. Behaviourally different models with the same FULL v1 automatic fingerprint (e.g. `TinyTransformer(n_heads=2)` vs `n_heads=4`) must not silently share provenance identity when the experimenter knows the distinction. Candidate inputs: model/constructor config, repository commit, checkpoint revision, implementation or model revision (e.g. `declared_model_config=…`, `implementation_revision=…`). The design is decided after M1.5 and before trace containers; it is **not** implemented in M1.4.
- **From ADR-022:** decide whether traces without a `declared_model` get a limitation, and how `trace()` accepts a declaration.
- **Requirements carried from M1.5:**
  - Convert `HookEvent`s to `ActivationRecord`s in the sink. The sink owns retention; the session must not.
  - Decide how to handle `pass_index = -1` (out-of-pass) events. The schema requires `>= 0`: either refuse them or record them with a limitation, but never invent a pass number.
  - Alias-group events (`path_specific=False`) have no `Site`: refuse them or represent them explicitly.
  - Record a limitation for selected sites that never executed.
  - Keep `FUNCTIONAL_OPS_UNOBSERVED`.
- **Requirements carried from M1.1/M1.2 review:**
  - **Reference integrity (exit criterion):** `TraceResult` must call `verify_ref` for **every** reference entering it (`derived_from`, evidence, claim/spec/result refs). It must also check that every `provenance_id` names a `ProvenanceRecord` it holds. Required tests: a forged `RecordRef` status, a forged `EvidenceRef`, and a dangling `provenance_id` are each rejected.
  - **Identity, not `==`:** containers compare records by id and verify content and reference integrity explicitly. They must not rely on dataclass equality, because NaN-bearing values (e.g. `TensorStats`) are not reflexively equal. Required test: a record with NaN stats is found, deduplicated, and verified by id.
  - Decide how a trace references its `ExecutionOccurrence` (run identity), and keep it out of `provenance_id` (ADR-019).
  - Build `ExecutionContext` from the live run (`model.training`, `torch.is_grad_enabled()`, device).
- **Exit criteria:** all pass. `recording()` over two forward passes yields distinct `pass_index` values. No tensor is retained under `summary` (weakref).
- **Known risk:**
  - Defining "top-level forward pass" inside `recording()`: counting root-module calls fails if the user calls a submodule directly. The rule is documented as "a pass = one call of the root module's forward"; submodule-only calls get `pass_index=-1` plus a limitation.
  - Stats on huge tensors cost time. Summary-mode overhead is measured in M1.9.
- **Scope:** about 500 LOC, about 500 LOC tests.

## M1.7: Serialisation

- **Goal:** lossless semantic round-trip. Tensor round-trip when retained.
- **Files:** `beyondnn/schema/codec.py`, `beyondnn/core/io.py` (save/load, sidecar).
- **Public API:** `TraceResult.to_json()`, `TraceResult.from_json()`, `TraceResult.save(path, tensors=True)`, `TraceResult.load(path, model=None)`.
- **Internal API:** `encode(record) -> dict`, `decode(dict) -> record` (kind registry), float/NaN handling, and unknown-kind passthrough (`OpaqueRecord`).
- **Tests:** `test_serialization.py`:
  - a property-style round-trip for every record kind (with hand-written generators, so no hypothesis dependency; that is reconsidered if coverage is weak);
  - a full trace round-trip for each tiny model;
  - bit-exact tensor round-trip through `weights_only=True`;
  - a `MODEL_MISMATCH` limitation on load with a different model;
  - `OpaqueRecord` preserved on re-save;
  - `schema_version` major mismatch raises;
  - JSON output is deterministic (sorted keys) for identical traces, apart from `trace_id` and timestamps.
- **Exit criteria:** every kind in the registry has a round-trip test, enforced by a test that iterates the registry.
- **Required by M1.1 (ADR-014/016):** schema migration may change record ids, so migration must remap every reference (`derived_from`, evidence, claim/spec/result refs, limitation `applies_to`) explicitly. Required test: migrate a record that other records reference, and verify that all references point at the new id and that `verify_ref` passes.
- **Known risk:** determinism vs timestamps and uuids. A `normalize_for_diff()` helper that masks volatile fields is used in tests and documented for users.
- **Scope:** about 350 LOC, about 350 LOC tests.

## M1.8: `instrument()`, targets, minimal `explain()`, examples

- **Goal:** the pass-through handle (ADR-002) and an honest Phase-1 explanation.
- **Files:** `beyondnn/core/instrument.py`, `beyondnn/core/targets.py`, `beyondnn/explain/__init__.py`, `why.py`, `render.py`, `examples/01_trace_mlp.py`, `02_trace_cnn.py`, `03_trace_transformer.py`.
- **Public API:**
  - `bnn.instrument(model) -> Instrumented` (`.module`, `__call__`, delegation, `.trace`, `.recording`, `.explain`; `train`/`eval`/`to`/`cpu`/`cuda` return the handle)
  - `bnn.targets.Logit`, `Prob`, `LogitDiff`, `LogProb`, `Loss`, `Output`
  - `bnn.explain(model, x, target=None, sites=None) -> Explanation`
  - `Why.render()`
- **Internal API:** the `Metric` protocol, `metric.spec() -> TargetSpec`, and a render template registry with status-bound verbs.
- **Tests:**
  - `test_instrument.py`: the model object is unchanged (same `__class__`, no hooks, identical `state_dict` keys and values), `optimizer(ix.parameters())` trains, and `isinstance(ix.module, nn.Module)`.
  - `test_explain_minimal.py`:
    - `explain` without a target on a classifier sets `defaulted=True` and emits `DEFAULT_TARGET_ARGMAX`;
    - missing evidence categories always produce `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE`, and `NO_CLAIMS_TESTED`;
    - render output is deterministic (golden file);
    - the render vocabulary lint: "caused" appears only next to INTERVENTIONAL records.
  - Examples run in CI as smoke tests.
- **Exit criteria:** all pass. Examples produce the documented output on all three models.
- **Known risk:**
  - `__getattr__` delegation can shadow attributes, and `__setattr__` semantics are subtle (setting on the handle vs the module). The decision is that setting attributes on the handle raises, and users must set them on `.module`.
  - `explain()` in Phase 1 may be perceived as underwhelming. That is intentional, and the docs must say so.
- **Scope:** about 400 LOC, about 350 LOC tests, about 120 LOC examples.

## M1.9: Overhead benchmark

- **Goal:** baseline cost numbers before interventions exist.
- **Files:** `benchmarks/bench_trace_overhead.py`, `benchmarks/README.md`.
- **Public API:** none.
- **Measures:** for each tiny model plus one scaled-up MLP (~10M params, CPU):
  - wall time for baseline, `trace(retain="none")`, `"summary"`, and `"cpu"`, over all leaf sites and over 10% of sites;
  - peak memory (`tracemalloc` plus RSS delta);
  - `trace.json` size and sidecar size.
  - Median of N runs, with warmup.
- **Tests:** the script runs in CI in `--quick` mode (1 repetition), as a smoke test only.
- **Exit criteria:** numbers are recorded in `EXPERIMENT_LOG.md` and the Phase 1 report, with hardware and commit. They are **not** pass/fail gates.
- **Known risk:** CPU timing noise. Report the spread, not just the median.
- **Scope:** about 200 LOC.

## M1.10: Phase 1 report and go/no-go

- **Goal:** an honest assessment before Phase 2.
- **Files:** `docs/roadmap/PHASE_1_REPORT.md`, `CHANGELOG.md`, `docs/experiments/EXPERIMENT_LOG.md`.
- **Content:**
  - implemented vs planned;
  - the test inventory;
  - known limitations;
  - benchmark numbers;
  - failures and their fixes;
  - design concerns, including whether the claim types still look right with Phase 2 in view;
  - a go/no-go recommendation.
- **Exit criteria:** you review it and decide.
- **Scope:** documentation only.

---

## Phase 1 test inventory (mapped to the originally requested files)

| Requested file | Phase 1 milestone | Notes |
|---|---|---|
| `test_trace_schema.py` | M1.1 | |
| `test_provenance.py` | M1.2 | |
| `test_hooks.py`, `test_hook_cleanup.py` | M1.5 | |
| `test_nested_modules.py`, `test_shared_modules.py` | M1.4 (+M1.6 trace behaviour) | |
| `test_serialization.py` | M1.7 | |
| `test_cpu.py` | M1.6 | |
| `test_tiny_mlp.py`, `test_tiny_cnn.py`, `test_tiny_transformer.py` | M1.3 + M1.6 | |
| `test_concepts.py`, `test_feature_concept_distinction.py` | Phase 6 | concept types deferred by ADR-014; not created in M1.1 |
| `test_interventions.py`, `test_intervention_cleanup.py`, `test_ablation.py`, `test_causal_effect.py` | Phase 2 | not created in Phase 1 |
| `test_attribution.py` | Phase 3 | |
| `test_audit.py` | Phase 7 | |

Estimated Phase 1 total: **about 3.0k LOC library**, **about 2.5–3k LOC tests**. That is larger than the first estimate, because the claim, concept, and effect *types* moved into Phase 1.
