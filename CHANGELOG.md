# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project will use
[Semantic Versioning](https://semver.org/) once released. The trace `schema_version` is versioned separately
(see `docs/design/TRACE_SCHEMA_PROPOSAL.md`).

## [Unreleased]

## [0.1.0] - 2026-10-01

**The first public release.** Pre-1.0 research software; the public API is frozen ([`docs/API_FREEZE.md`](docs/API_FREEZE.md)). Not yet published on PyPI at the time of tagging.

### Added
- **Provenance-aware tracing** (`bnn.trace`, `bnn.recording`, `bnn.instrument`): structured, content-addressed, versioned records bound to the model checkpoint, the sample identity and the software environment; persistence with full re-validation on load.
- **Explicit evidence statuses** on every record: OBSERVED, MEASURED, ATTRIBUTED, INTERVENTIONAL, ESTIMATED_CAUSAL (reserved), VALIDATED_CONCEPT, GENERATED.
- **Attribution** (`bnn.attribute`): gradient, input × gradient and integrated gradients (native, and via an optional Captum adapter), recorded as ATTRIBUTED evidence with baselines and diagnostics.
- **Controlled interventions** (`bnn.intervene`): INTERVENTIONAL effects with declared claims and threshold tests.
- **Faithfulness protocols** (`beyondnn.faithfulness`): comprehensiveness and sufficiency with explicit replacements and matched random controls; curves, stability and diagnostics.
- **Concepts** (`beyondnn.concepts`): features (neurons, directions, SAE latents), concept datasets, proposals, encoding and use tests, and validation under a declared policy (UNLABELED_FEATURE → PROPOSED_CONCEPT → VALIDATED_CONCEPT).
- **Scientific audits** (`bnn.audit`, `beyondnn.audits`):
  - deterministic, model-free classification of declared claims from re-derived, in-scope evidence: standings and typed findings;
  - PRIMARY / ALTERNATIVE / STRESS_TEST configuration roles;
  - sensitivity profiles; uncertainty intervals;
  - declared unit eligibility (e.g. content tokens vs all tokens);
  - control-attainability checks.
- **Save / reload / re-audit / verify:** `audits.save_evidence`, `load_evidence`, `verify_report`; reports record the BeyondNN version and audit semantics that produced them.
- **Structured WHY** (`bnn.compose`): recorded evidence arranged by kind, never merged into a narrative or a score, with what was not evaluated.
- **Reproducibility and migrations:** tested migrations for every older record version; `uv.lock`.

### Scientific safeguards
- **Attribution is not causal evidence:** attribution-only causal claims are UNSUPPORTED.
- **Decodability is not causal use:** decodable-but-unused concepts stay PROPOSED, and generated labels are never upgraded.
- **Replacements are always explicit;** there is no implicit zero.
- **Evidence from another checkpoint, sample or dataset scope is excluded and reported,** never silently counted.
- **Alternative and stress-test configurations never rewrite the PRIMARY standing.**
- **No global explanation, interpretability, trust or confidence score.**

### Documentation and release
- **Documentation:** a public README (concepts, quickstart, end-to-end example, validation with its scope, limitations); a topic-organised `docs/`; a reproducibility guide; the full research record (pre-registered plans, reports, negative results).
- **Examples:** numbered, public-API examples (`examples/01_quickstart.py` … `07_save_reload.py`), all executed in CI.
- **CI:** tests on Python 3.10 / 3.12 / 3.14 (Linux), a Captum job, lint, mypy --strict, build and a clean-wheel check.
- **Release:** a manual-only PyPI Trusted Publishing workflow ([`docs/release/PYPI_RELEASE.md`](docs/release/PYPI_RELEASE.md)).
- **Community files:** `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.

## Development history before 0.1.0

The entries below record the pre-registered development stages in detail, including negative results and deviations (see [`docs/README.md`](docs/README.md#development-history)).

### Public release preparation
- **Documentation:**
  - README rewritten for external users: concepts, installation, quickstart, end-to-end example, validation (scoped) and limitations;
  - documentation index reorganised by topic;
  - `docs/REPRODUCIBILITY.md` and `experiments/README.md` added.
- **Examples:** numbered public-API examples `01_quickstart.py` … `07_save_reload.py`, all executed by the test suite. The earlier examples were renamed: `phase4_redundant_path.py` → `03_intervention.py`, `phase6_concepts.py` → `05_concepts.py`, `phase7_audit.py` → `06_audit.py`.
- **Packaging:**
  - project URLs, author, classifiers;
  - a `dev` dependency group for `uv sync`;
  - `uv.lock`;
  - the wheel contains only the `beyondnn` package;
  - the sdist include list is anchored.
- **Community files:** `CONTRIBUTING.md` (uv workflow, scientific rules, invariants, migrations), `SECURITY.md` (GitHub private vulnerability reporting), Code of Conduct contact route, `CITATION.cff` (software).
- **GitHub:** CI (tests on Python 3.10 / 3.12 / 3.14, a Captum job, lint, mypy, build and clean-wheel checks), a manual build-only release workflow, issue and PR templates, Dependabot.

### Changed (pre-Phase-8 hardening)
- **Audit reports (format 3)** record their `producer` (BeyondNN version, audit semantics; ADR-056). Format 2 still loads. `verify_report` compares scientific content and names a semantics difference.
- **Trace load errors** name the trace directory.

### Added (pre-Phase-8 hardening)
- A permanent golden scientific-workflow test, a migration-matrix test, provenance-completeness and named-claim-lookup tests.
- `docs/PRE_PHASE8_INVARIANTS.md`.

### Added
- **Phase 7.75: scientific fixes** (ADR-053 to ADR-055):
  - declared unit eligibility for selections, controls and selection claims (`eligible=` / `eligibility=`; `evidence_selection` v3, `audit_plan` v3, with migrations);
  - the audit treats unattainable control criteria as inconclusive (`control_criterion_unattainable`);
  - the audit distinguishes competitive-only failure (`effect_without_competitive_advantage`) and never uses evidence of another eligibility (`eligibility_mismatch`);
  - concept `min_change` is declared in target units (policy);
  - experiments in `experiments/phase7_75/` (TD programs with program-defined truth, concept calibration, D1/D2 and C1/C2, InterpBench re-audit, NLI stratification).

### Added (Phase 7.5)
- **Phase 7.5: audit refinement, uncertainty, external validation, API freeze** (ADR-048 to ADR-052):
  - declared configuration roles (`audits.role`; PRIMARY / ALTERNATIVE / STRESS_TEST) with PRIMARY-only standings;
  - `alternative_reverses` / `stress_test_reverses` / `undeclared_configuration` / `primary_untested` findings;
  - a `SensitivityProfile` per group; `configuration_level_disagreement` findings;
  - `audit_plan` v2 (v1 migrates); report format v2;
  - uncertainty: `audits.wilson` / `bootstrap` / `paired_bootstrap` and `Interval`; per-sample claims carry Wilson intervals;
  - evidence helpers `audits.sample_id` / `traces_of` / `save_evidence` / `load_evidence`;
  - `concepts.load_validation` (a WHY and an audit from saved traces); `protocols.PROTOCOL_VERSIONS`;
  - `interventions.metrics.margin`; named replacements (`faithfulness.replacement(t, name=...)`);
  - `docs/API_FREEZE.md`.
- **Experiments:** `experiments/phase7_5/`: InterpBench / Tracr external validation, central A–D, e-SNLI rationales, token probes, the external-researcher workflow, mutations, performance, hypothesis evaluation and figure data.

### Changed
- **Breaking (ADR-052):** `faithfulness.comprehensiveness`, `sufficiency` and `curve` require `replacement=`; there is no implicit zero replacement.
- `Interval.describe()` prints non-integer levels as declared (98.75%, not 99%).

### Added (Phase 7 and earlier)
- **Phase 7: scientific audits** (`bnn.audit`, `bnn.audits`, ADR-044 to ADR-047):
  - a deterministic, model-free audit of recorded traces (or saved trace paths) under a pre-declared `AuditPlan` (record kind `audit_plan` v1);
  - integrity and re-derivation of every result; provenance and scope exclusion;
  - 7 standings and 13 finding kinds with categorical severities; no score;
  - assumption-axis sensitivity (protocol, threshold, replacement, k, null, method, dataset);
  - re-evaluation under declared alternative thresholds;
  - structural overclaims (attribution → causal, decodable → used, generated → validated, narrower estimand, untested invariance, missing controls);
  - per-sample distributions with counterexample identities and caps;
  - concept audits; coverage;
  - deterministic report JSON, `verify_report`, and an AUDIT section in the WHY (`bnn.compose(..., audit=report)`).
- **Trace-level concept re-derivation** (`concepts.verify.verify_*_trace`, `verify_feature_record`): a partial fix of P6-4 (audits work from saved traces).
- **Documentation and experiments:** Phase 7 literature review, differentiation, plan, report, `docs/audit/`, `examples/phase7_audit.py`, `benchmarks/bench_audit.py`, `experiments/phase7/`, and `docs/roadmap/TRACE3B_FUTURE.md` (a boundary note only).
- **Phase 6: concepts and concept validation** (`bnn.concepts`, ADR-039 to ADR-043):
  - features: neurons, directions, SAE latents (a tensor-only adapter);
  - train-only discovery (`fit_direction`, `search_neurons`);
  - concept datasets and proposals (always PROPOSED), and GENERATED labels;
  - `encoding_test` (ENCODES; held-out AUROC against mandatory controls; counterexamples);
  - `use_test` (DECREASES/INCREASES by removal, SUFFICIENT_FOR by retention; the intervention and reference are required; mandatory random-feature controls);
  - `validate` (derived semantic status under `POLICY_V1`; no global REJECTED state);
  - `activation` (VALIDATED_CONCEPT records only for validated concepts);
  - full re-derivation (`concepts.verify`) and a CONCEPTS section in the structured WHY (`bnn.compose(concepts=[...])`);
  - `Coverage.concepts_evaluated`.
- **DIRECTION interventions** (`interventions.direction`; `InterventionRecord` v4): projection removal and retention against a declared reference.
- **`Subject.feature`** (`Claim` v3; the golden claim id changes).
- **Documentation:** Phase 6 literature review, plan, report, `docs/concepts/`, and `examples/phase6_concepts.py`.
- **Phase 5.5: realistic faithfulness validation.** Pre-registered experiments (`docs/PHASE_5_5_PLAN.md`) on a trained MLP, a trained CNN and BERT-tiny/SST-2 (`experiments/phase5_5/`, with its own pinned requirements; nothing added to the core install). Also `docs/PHASE_5_5_API_REVIEW.md` and `docs/PHASE_5_5_REPORT.md`.
- **Declared unit axes (ADR-034):**
  - `unit_axes` on interventions, selections (`ranking`, `top_k`, `units`, `selector`) and claim subjects, for pixels, channels and token positions;
  - an explicit `reduce` (`sum`/`abs_sum`/`l2`) for per-unit attribution scores;
  - diagnostics accept the same.
  - Record versions: `InterventionRecord` v3, `Claim` v2, `EvidenceSelection` v2, with migrations. The golden claim id changes.
- **Perturbation-magnitude-matched controls (ADR-035):** `faithfulness.controls(n, seed=, match="magnitude", strata=4)`. The magnitudes are recorded and re-derived in composition.
- **Keyword model inputs** in `stability`, `method_agreement`, `baseline_sensitivity` and `ig_step_sensitivity` (ADR-037).
- **Phase 5: faithfulness tests** (`bnn.faithfulness`, ADR-032, ADR-033):
  - unit-level and model-input interventions and comparison families (`InterventionRecord` v2);
  - `comprehensiveness`/`sufficiency` claim tests with matched random controls;
  - removal/retention curves;
  - `stability` under caller-declared transformations;
  - dataset runs with `counterexample` and `paired_control` summaries;
  - method diagnostics;
  - `EvidenceSelection` and `ProtocolResult` records; no new evidence status;
  - composition into the structured WHY, with re-derivation of every result;
  - `Coverage.faithfulness_protocols`;
  - ground-truth models;
  - protocol documentation;
  - a faithfulness benchmark and the Phase 5 report.
- **Phase 4: structured WHY / evidence synthesis** (`bnn.compose`, ADR-031, ADR-026 amended):
  - `EvidenceBundle` validates one explanation context (model, declaration, exact input sample, instance scope, one target) and refuses anything incompatible;
  - recorded claim-test results are re-derived from their evidence;
  - `Why` sections by epistemic status over the original records, with claims, tests, and policy assessments (MIXED preserved), the limitations union, `Coverage` (faithfulness and concepts never evaluated), `by_status`, and `origin`;
  - deterministic `render()` and `to_dict()`;
  - README walk-through, `examples/phase4_redundant_path.py`, a synthesis benchmark, and the Phase 4 report (gate: GO WITH EXPLICIT LIMITATIONS).
- ADR-030 and ADR-031 finalized (owner approval of `InputRecord.sample_id`).
- `InputRecord.sample_id` (record version 3): the exact input identity, with a migration from v2.

### Fixed
- `TraceResult` lookups by record kind and pass were linear scans, which made comparison families of thousands of passes quadratic (ADR-043; 8× faster on a concept use test).
- Attribution no longer refuses deterministic models whose grad-enabled and no-grad forwards differ by rounding: the tolerance is scaled to the output precision (ADR-036; found on a trained CNN).
- Curves over declared units keep `unit_axes`/`unit_reduction` in their selection record, and a declared reduction is recorded even when each unit is one element. Both were found by composition on realistic models.
- `intervention_threshold` can decide only NECESSARY_FOR/DECREASES/INCREASES, independent of the protocol registry. Composition refuses decisive results from protocols that do not justify the claim's relation (found by the Phase 4 mutation audit).
- **Phase 3: attribution** (`bnn.attribute`, `bnn.attribution`, ADR-030):
  - ATTRIBUTED `AttributionRecord` and `AttributionReduction` records, with container checks for site/call/pass and shape;
  - native gradient, input × gradient, and Integrated Gradients (explicit baseline, declared rule, completeness diagnostic) on float inputs or one call of a module output;
  - explicit scalar targets (built-in metrics, never an implicit sum) and explicit reductions;
  - guards on model state, gradients, caller tensors, hooks, and RNG;
  - Captum 0.9 adapters (optional extra `beyondnn[captum]`) cross-checked against native;
  - the `attribution_threshold` protocol (ATTRIBUTED_TO only) in a central protocol registry (`beyondnn.protocols`);
  - four attribution limitations;
  - `Why.attributions`;
  - an attribution benchmark and the Phase 3 report (gate: GO WITH EXPLICIT LIMITATIONS).
- **Phase 2: causal interventions** (`bnn.intervene`, `bnn.interventions`, ADR-028):
  - `InterventionRecord`/`CausalEffect` schema;
  - zero, constant, and patch interventions on module outputs, via BeyondNN-owned hooks inside one recording;
  - paired baseline/intervention passes with refusal of training mode, RNG consumption, and state drift;
  - INSTANCE and FINITE_SAMPLE INTERVENTIONAL effects (`intervention − baseline`);
  - scalar metrics;
  - the `intervention_threshold` claim protocol with a protocol registry;
  - ground-truth causal models;
  - four intervention limitations.
- Caller metrics carry a declared identity (ADR-029): `metrics.custom(name, fn, *, implementation_revision, config)`, `MetricDeclaration`, and `MetricSpec.target()`. Unversioned caller metrics are refused. `causal_effect` is now record version 2, with a migration.
- Phase 2 benchmark (`benchmarks/bench_interventions.py`), a README intervention example executed by tests, and the Phase 2 report (gate: GO WITH EXPLICIT LIMITATIONS).
- Comparisons also refuse input modification in place and drift in execution conditions between the paired passes.
- **M1.1: trace schema 0.1** (`beyondnn.schema`):
  - immutable records with content-derived ids;
  - evidence status bound to record kind, with explicit derivation rules;
  - `InputRecord`, `OutputRecord`, `ActivationRecord`, `TensorRef`, `TraceLimitation`;
  - `Claim`, `ClaimTestSpec`, `ClaimTestResult`, and derived `Assessment` under explicit policies;
  - estimand scopes (instance / finite sample / population);
  - a strict, versioned JSON envelope.
- ADR-016 to ADR-018, accepted after review: record identity (identity ≠ semantic equivalence), status mechanism (measured state under intervention stays MEASURED; INTERVENTIONAL is reserved for effects), assessment policies (no universal thresholds).

- **M1.2: provenance and model fingerprinting**:
  - schema types `ModelIdentity`, `EnvironmentIdentity`, `ExecutionContext` (CLEAN/INTERVENTION), `Randomness`, `MethodIdentity`, `ProvenanceRecord`, `ExecutionOccurrence`;
  - `beyondnn.provenance` with the FULL v1 model fingerprint (structure plus bitwise state, tied-parameter and shared-module topology), environment collection without machine identity, read-only CPU RNG digest, and provenance and occurrence builders;
  - ADR-019 and ADR-020;
  - ADR-016 amendment: expectation wording and the no-Unicode-normalisation decision.

- **M1.10: Phase 1 audit and gate report** (`docs/PHASE_1_REPORT.md`: GO WITH EXPLICIT LIMITATIONS).
  - Public recording now refuses module replacement or structural change between passes (ADR-027).
  - Audit tests (`tests/test_phase1_audit.py`) guard the public surface and the Phase-1 evidence statuses.
  - README corrected to state that only observed and measured evidence exist today.
  - Installed-wheel smoke test run in a clean virtualenv.
- **M1.9: `benchmarks/bench_trace_overhead.py`**: CPU characterisation of baseline, trace none/summary/cpu, fingerprint, save/load, and exact byte sizes. Results are in the experiment log. There is no pass/fail threshold and no optimisation.
- **M1.8: `bnn.instrument(model)`**: a frozen handle referencing the original model.
  - `handle.trace`/`recording` are exactly the public pipeline.
  - `handle.explain(...)` gives `ExplainResponse` (INPUT → WHY → OUTPUT) over one single-pass trace. Phase-1 `why` is measured evidence (activations, limitations, provenance) with `NO_ATTRIBUTION`/`NO_CAUSAL_EVIDENCE`/`NO_CLAIMS_TESTED`, never a causal explanation.
  - A deterministic `render()`.
  - README examples are executed by the test suite. ADR-026.
- **Trace occurrence and hook provenance fixes (ADR-025):**
  - `ExecutionOccurrence` v2 carries `pass_index`, one per pass, with v1 migrating to `None`.
  - Public `trace()`/`recording()` refuse foreign forward/forward-pre hooks (local or global), checked at entry and at every pass start/end.
  - Models whose tensors are on several devices are refused instead of being labelled with one device.
- **M1.7: trace persistence:** `TraceResult.save(dir)` and `bnn.load_trace(dir)`.
  - Deterministic `trace.json` with an optional `tensors.pt`, read only with `weights_only=True`. No paths are taken from JSON, and symlinks are refused.
  - Atomic save; the target is never overwritten.
  - Full re-validation on load: ids, references, provenance, and tensor dtype/shape/digest.
  - Migrations with deterministic reference remapping.
  - The codec now verifies stored ids before migrating. ADR-024.
- **M1.6: trace pipeline:** public `bnn.trace`, `bnn.recording`, `bnn.TraceResult`.
  - Per-pass provenance, with the model fingerprinted at the start of each root call.
  - OBSERVED root input/output and MEASURED activations, one per tensor leaf, with deterministic paths.
  - Retention `summary`/`cpu`/`none`; no live tensors are retained.
  - Container validation of every record and reference.
  - Refusal of out-of-pass and aliased executions; honest limitations (`SELECTED_SITE_NOT_EXECUTED`, `NON_TENSOR_LEAVES_IGNORED`).
  - `InputRecord`/`OutputRecord` v2 (`pass_index`). ADR-023.
- **M1.5 fix: recursive pre-hook invocation bookkeeping.** A prepended guard pre-hook claims each attempt's call index, and the observation pre-hook (after user pre-hooks) emits INPUT. A user pre-hook raising inside recursion can no longer release the enclosing frame. Sessions refuse to start while global module forward pre-hooks are registered.
- **Caller-declared model provenance** (M1.6 gate, ADR-022):
  - `ModelDeclaration` (config, implementation revision, checkpoint revision) in `ProvenanceRecord.declared_model`;
  - `ProvenanceRecord` record_version 2, with a v1→v2 migration;
  - `make_provenance(declared_model=…)`;
  - migration errors now raise `DecodeError`.

  Resolves the M1.3 `n_heads` identity gap without changing FULL v1.
- **M1.5: safe hook session** (`beyondnn.core.hooks`, internal):
  - scoped hooks, removed on every exit path (normal exit, exceptions, sink errors, partial install);
  - physical-module dedup;
  - pass/call indices with failure consumption, out-of-pass `-1`, and stack-paired INPUT/OUTPUT;
  - aliased modules refused by default (internal `GROUP` mode);
  - values neither modified nor retained;
  - autograd, outputs, and BatchNorm state identical to baseline. ADR-021.
- **M1.4: deterministic module site resolution** in `beyondnn.core.sites` (not exported):
  - segment-based `*` / `**` pattern language; the root is selected only by `""`;
  - strict (unmatched or invalid patterns raise);
  - traversal-order results, deduplicated by exact path;
  - alias paths preserved as distinct sites;
  - no forward, hooks, or RNG use.

  Also: a hard roadmap gate before M1.6 for caller-declared model config/revision.
- **M1.3: deterministic tiny reference models** in the internal, unstable `beyondnn._testing.models`:
  - `TinyMLP` (139 params; shared module called twice; functional op; keyword-only `scale`);
  - `TinyCNN` (396; BatchNorm buffers; tuple output);
  - `TinyTransformer` (5,120; tied `lm_head`/`token_embedding`; non-persistent `position_ids`/`causal_mask` used in forward; dict output).

  Construction is seeded without changing the caller's global RNG. The models are not exported from `beyondnn`.

### Fixed (M1.2 review)
- The FULL fingerprint now hashes the values of **all** registered buffers, including non-persistent ones, which can affect `forward`. Buffer persistence is recorded as metadata. The v1 golden digests were updated before any release (ADR-020 correction).
- Access to PyTorch's private buffer-persistence field is isolated in one helper, with a compatibility test.
- Docs: the definition of FULL (Python code is not hashed); `schema` is stdlib-only; an implementation-revision roadmap item.

### Changed (M1.1 review fixes)
- `ActivationRecord` / `MEASURED` semantics: directly observed state, including during intervened executions.
- `derived_from` and `ClaimTestResult.evidence` are sorted at construction, and `-0.0` is normalised, so ids no longer depend on order or on the sign of zero.
- The codec no longer imports the schema package from inside itself (the apparent import cycle is removed). Dead code is removed.
- Second, independent guard tests added for derivation rules, population matching, policy coverage, and id integrity.
- Design documentation: ecosystem audit, differentiation, interpretability definition, architecture proposal,
  trace schema proposal, roadmap, Phase 1 plan, research questions, experiment log.
- Architecture decision records ADR-001 to ADR-015 (ADR-012 claims, ADR-013 estimand scope, ADR-014 incremental schema versioning, ADR-015 single-execution traces).
- Project scaffolding: `pyproject.toml`, CI workflow, pre-commit configuration, package placeholder.
