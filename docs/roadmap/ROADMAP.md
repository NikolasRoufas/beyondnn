# Roadmap

**Rules for advancing:**
- Phases advance only through a written go/no-go in `docs/roadmap/PHASE_<N>_REPORT.md`.
- Each report covers: what was implemented, tests, known limitations, performance, failures, design concerns, and a recommendation for the next phase.
- Decisions made along the way are appended to `docs/decisions/ARCHITECTURE_DECISIONS.md`.
- Experiments go in `docs/experiments/EXPERIMENT_LOG.md`, and negative results stay.

The conceptual pipeline every phase serves (see `docs/design/ARCHITECTURE_PROPOSAL.md`):

```
Measurement → Claim → Test → Evidence → Assessment → WHY presentation
```

| Phase | Adds to the pipeline |
|---|---|
| 1 | Measurement (observe/measure), plus the *types* for every later stage |
| 2 | Interventional measurement, the first claim tests, the Assessment policy |
| 3 | Attribution measurement plus `ATTRIBUTED_TO` claims |
| 4 | WHY presentation |
| 5 | More claim tests (faithfulness suite) |
| 6 | Concepts, validated via `ENCODES` + causal claims |
| 7 | Audit (runs claim tests over datasets) |
| 8 | External comparison |

## Phase 1: Foundations: **COMPLETE, GO WITH EXPLICIT LIMITATIONS** (2026-09-25; see `docs/PHASE_1_REPORT.md`)

Detailed plan: [`PHASE_1_PLAN.md`](PHASE_1_PLAN.md) (milestones M1.0–M1.10).

**Goal:** a trace schema and hook engine that researchers can trust.
- Hooks never leak.
- Results serialise losslessly.
- Every record type through `Assessment` is defined and round-trips.
- The same test suite passes on an MLP, a CNN, and a tiny transformer.

**Out of scope:** interventions, attribution, test runners, concepts logic, adapters, GPU-specific code, visualisation, `torch.compile`, and Mode B.

## Phase 2: Interventions, causal effects, first claim tests: **IMPLEMENTED, gate GO WITH EXPLICIT LIMITATIONS** (2026-09-25; `docs/PHASE_2_REPORT.md`, ADR-028)

- **As built:**
  - one comparison is one recording, so there is no `Study` yet;
  - output interventions: zero, constant, and patch from a source pass;
  - INSTANCE and FINITE_SAMPLE INTERVENTIONAL effects;
  - the `intervention_threshold` protocol with a registry.
- **Not built from the original list below:** mean/resample ablation, feature-delta interventions, bootstrap CIs, `Study`, and the synthetic A/B/C correlated-feature suite. Replaced for now by analytic ground-truth models.

**Original plan:**
- **`Study` container (ADR-015):** multiple `TraceResult`s, multi-input claim test results, dataset-level assessments, population estimates, and cross-input statistics.
- The `CausalEffect` record, designed around `Estimand` (ADR-013). Finite-sample aggregates reference their individual per-input effects. Population estimates are a separate `ESTIMATED_CAUSAL` step.
- `InterventionSpec` operations: zero, mean (explicit reference set), constant, patch-from-trace, scale, and neuron-basis feature delta.
- A `bnn.intervene(model, spec)` context, with the same cleanup test matrix as hooks.
- `Metric` evaluation, `measure_effect`, and `CausalEffect` (single-input exact, and aggregate with bootstrap CI).
- Distribution checks leading to `OFF_DISTRIBUTION_INTERVENTION`.
- **Claims:** `bnn.test_claim`, the runner registry with relation compatibility, the tests `ablation_necessity/v1`, `random_baseline/v1`, `sufficiency_patch/v1`, and `scaling_direction/v1`, and assessment policy `default/v1` with its own documentation page.
- **First ground-truth suite:** synthetic models where A is causal, B is correlated but unused, and C is noise.
  - Built with hand-set weights (exact ground truth) and also trained (does training preserve it?).
  - Expected results are written *before* running, and outcomes are logged whatever they are.

## Phase 3: Attribution: **IMPLEMENTED, gate GO WITH EXPLICIT LIMITATIONS** (2026-09-25; `docs/PHASE_3_REPORT.md`, ADR-029, ADR-030)

- **As built:**
  - ATTRIBUTED `AttributionRecord`/`AttributionReduction`;
  - native gradient, input × gradient, and IG on inputs or one call of a module output;
  - Captum 0.9 adapters (Saliency, InputXGradient, IntegratedGradients, LayerIntegratedGradients) cross-checked against native;
  - explicit scalar targets and reductions;
  - `attribution_threshold` (ATTRIBUTED_TO only);
  - analytic ground truth, including the redundant-path negative example.
- **Not built from the original list below:** FeatureAblation/Occlusion adapters, `EvidenceSpan`, `method_agreement/v1`, and the top-attributed-vs-random ablation experiment.

**Original plan:**
- Native gradient and input × gradient.
- A Captum adapter (IG, FeatureAblation, Occlusion, LayerIntegratedGradients). It must be numerically equal to direct Captum calls.
- `EvidenceSpan` derivation. `ATTRIBUTED_TO` claims and the `method_agreement/v1` test.
- **Experiment:** do top-attributed units pass `ablation_necessity` more often than random units on the Phase 2 suite? Logged, not assumed.

## Phase 4: WHY: **IMPLEMENTED, gate GO WITH EXPLICIT LIMITATIONS** (2026-09-25; `docs/PHASE_4_REPORT.md`, ADR-031)

- **As built:**
  - `bnn.compose` over an immutable, validated `EvidenceBundle` (one model, declaration, exact input sample, target; instance scope);
  - `Why` sections by epistemic status, over the original records;
  - declared claims with re-derived test results and policy-based assessments;
  - the limitations union;
  - `Coverage` (faithfulness and concepts never evaluated);
  - a deterministic renderer and `to_dict`;
  - `InputRecord.sample_id`.
- **Not built from the original list below:**
  - status-bound template summaries (`summary()`) and the optional LLM summariser: deliberately not built, since no generated text is allowed;
  - `explain(x, target, plan)` that runs tests: methods run explicitly, and composition only reads;
  - candidate-claim proposal: claims are only declared.

**Original plan:**
- `Why` as a view and `render()` with status-bound vocabulary, including a lint test on templates.
- A template `summary()` with `supporting_records`. An optional LLM summariser goes behind an extra and is always `GENERATED`.
- `explain(x, target, plan)`: proposes candidate claims (`ClaimSource.method`), runs the plan's tests, and assesses.

## Phase 5: Faithfulness tests: **IMPLEMENTED, gate GO WITH EXPLICIT LIMITATIONS** (2026-09-25; `docs/PHASE_5_REPORT.md`, ADR-032, ADR-033)

- **As built:**
  - `comprehensiveness` and `sufficiency` claim tests, at input and internal level;
  - removal and retention curves;
  - matched random controls;
  - `stability` under declared transformations;
  - `counterexample` and `paired_control` over declared sample sets;
  - method diagnostics;
  - a ground-truth validation suite;
  - protocol docs (`docs/protocols/`).
- **Not built:**
  - RQ8, deferred with reason: too few ground-truth tasks for a held-out evaluation;
  - ROAR retraining;
  - Quantus/Captum-metric adapters.

**Original plan:**
- New test kinds: `comprehensiveness/v1`, `sufficiency/v1` (input and internal), `stability/v1` (user-declared invariances only), and `counterexample/v1`.
- Assessment components are exposed. There is no aggregate score (ADR-007).
- Every test gets a documentation page covering its formal definition, implementation, interpretation, and limitations.
- Also Research Question 8: can any aggregate predict held-out outcomes on ground-truth models?

## Phase 5.5: Realistic faithfulness validation: **gate READY FOR PHASE 6 WITH EXPLICIT LIMITATIONS** (2026-09-26; `docs/PHASE_5_5_REPORT.md`, ADR-034 to ADR-038)

- Pre-registered validation on a trained MLP (breast_cancer), a trained CNN (digits) and BERT-tiny (SST-2), with 60/60/40 held-out samples.
- **Framework changes forced by the realistic runs:**
  - declared unit axes (pixels, channels, tokens);
  - magnitude-matched controls;
  - an output-precision reproducibility tolerance;
  - keyword inputs in diagnostics;
  - two fixes to the unit-axes change itself.
- **Open, bounded** (API review):
  - `run_dataset` supports only one target and one criterion, and takes no kwargs;
  - there is no selection source for intervention-derived rankings;
  - curves refuse declared rankings;
  - a diagnostic takes one reduction for both methods;
  - `faithfulness.stats` is not exported.

## Phase 6: Concepts: **gate READY FOR PHASE 7 WITH EXPLICIT LIMITATIONS** (2026-09-26; `docs/PHASE_6_REPORT.md`, ADR-039 to ADR-043)

- **As built:**
  - feature records (neurons, directions, SAE latents through a tensor-only adapter) with train-only discovery;
  - concept datasets and proposals, and GENERATED labels;
  - the three-state semantic lifecycle (no global REJECTED);
  - ENCODES (`concept_encoding`) and use (`concept_intervention`) claim tests with mandatory controls, and counterexamples;
  - DIRECTION interventions;
  - derived validation (`POLICY_V1`) with full re-derivation;
  - a CONCEPTS section in the WHY;
  - trace indexes.
- **Results:**
  - the pre-registered ground truth (A–J) behaved as predicted;
  - realistic runs found natural decodable-but-unused features and strong intervention-semantics dependence;
  - only 1 of 9 realistic concepts validated.
- **Open (bounded; see the report):**
  - the covariance null can be over-matched;
  - v1 has no counterexample-rate caps;
  - label text is not checked against the extension;
  - result objects are not reconstructable from saved traces;
  - concept validations appear only as context in the instance WHY;
  - there is no dataset-level WHY;
  - SUFFICIENT_FOR is shown but not required.

**Original plan:**
- `FeatureBasis` (neuron, direction; SAE via adapter).
- The `SemanticStatus` lifecycle, and `concepts.validate()` implemented as testing an `ENCODES` claim plus a causal claim, with counterexamples and random-direction controls.

## Phase 7: Audit: **gate READY FOR PHASE 7.5 WITH EXPLICIT LIMITATIONS** (2026-09-26; `docs/PHASE_7_REPORT.md`, ADR-044 to ADR-047)

- **As built:**
  - `bnn.audit(evidence, plan=plan)`: model-free; it re-derives every result, excludes evidence by provenance and scope, and yields 7 standings and 13 finding kinds with no score;
  - `AuditPlan` records, including per-sample targets;
  - reload-safe audits and `verify_report`;
  - an AUDIT section in the WHY.
- **Results:**
  - scenarios A–N 14/14;
  - on the realistic models, the IG necessity claim is SUPPORTED on 1/280 samples under the declared invariances, vs 84/280 for a single configuration;
  - random selections are mostly CONTRADICTED;
  - the concept audit reproduces Phase 6.
- **Open (Phase 7.5):**
  - ASSUMPTION_SENSITIVE is coarse (conditional standings are needed);
  - configuration-level protocol disagreement;
  - uncertainty statements;
  - streaming or compact evidence storage;
  - `compose` still needs live Phase-6 objects (P6-4).

**Original plan:**
- `bnn.audit(model, dataset, plan) -> AuditReport` with `metadata` (versions, seed, device, config, methods).
- It uses only tests and metrics that already exist with documentation.

## Phase 7.5: External validation, audit refinement, API freeze: **gate READY FOR PHASE 8 WITH EXPLICIT LIMITATIONS** (2026-09-29; `docs/PHASE_7_5_REPORT.md`, ADR-048 to ADR-052)

**Done:**
- configuration roles, sensitivity profiles, configuration-level disagreement, uncertainty intervals;
- evidence persistence and `load_validation`; `metrics.margin`; required replacements;
- InterpBench / Tracr external validation; central A–D held-out; e-SNLI rationales; token probes;
- the external-researcher workflow;
- the API freeze (`docs/API_FREEZE.md`).

**Open (must-fix before the corresponding paper claims; `research/PHASE_7_5_PAPER_READINESS.md`):**
- external ground truth independent of the PRIMARY intervention (E1 is partly circular);
- a concept known positive validated without tuning (the frozen policy rejects Tracr `is_x`);
- declared special-token eligibility (D's IG supports mostly select [SEP]).

Trace 3B stays out of scope.

## Phase 7.75: Scientific problem resolution and evidence freeze: **gate SCIENTIFICALLY READY FOR PHASE 8 WITH PAPER LIMITATIONS** (2026-09-30; `docs/PHASE_7_75_REPORT.md`, ADR-053 to ADR-055)

**Done:**
- independent known-mechanism benchmark (TD: Tracr programs with program-defined truth; 0/400 decoys supported);
- declared unit eligibility (content vs all tokens);
- unattainable control criteria are inconclusive; competitive-only failures flagged;
- scale-relative concept use threshold (4/4 held-out positives, 0/8 negatives);
- D1 / D2, C1 / C2; InterpBench and central re-audits;
- final paper-claim freeze (`research/PAPER_EVIDENCE_LEDGER.md`).

**Paper limitations:**
- Claims F and G hold only with qualifiers (compiled programs; programs with known use).
- There is no independent truth on trained models.

**Phase 8 must stay boring:** packaging, CI, docs, release metadata, reproducibility. No methodology changes.

## Phase 8: Release engineering (not started; awaits approval)

- **Scope (Phase 7.75 request §43; `docs/PRE_PHASE8_INVARIANTS.md`):** packaging, CI, GitHub, documentation polish, security / contributing / code of conduct, release metadata, TestPyPI / PyPI, `uv` lock, reproducibility instructions, benchmark scripts.
- **No scientific methodology changes.**
- The earlier "Benchmark" scope below is not part of Phase 8 unless re-approved.

### Earlier plan (superseded scope): Benchmark
- Capability and correctness vs nnsight, Captum, TransformerLens, and pyvene on the tiny models, plus runtime and memory.
- Ease of use is assessed through task scripts and reported qualitatively.

## Hard gates

- **RESOLVED (2026-09-25, ADR-022): before M1.6 (trace containers)**, resolve a caller-declared model configuration / implementation revision in provenance. Now `ModelDeclaration` in `ProvenanceRecord.declared_model`, record_version 2. Original gate text: so that behaviourally different models with the same FULL v1 fingerprint cannot silently share provenance identity when the distinction is known (see the M1.3 finding below and `PHASE_1_PLAN.md` §M1.6). FULL v1 itself stays unchanged; the `n_heads` case is an accepted, documented limitation of automatic fingerprinting.

## Open investigation items

- **Tolerance-aware ties in control comparisons** (found by the first GitHub CI run, 2026-10-01).
  - **Current behaviour:** control fractions compare recomputed effects by exact float equality, so a mathematically equal control set can tie on one platform and not on another.
  - **Impact today:** none on recorded evidence. Re-derivation uses recorded values, and the test fixture was made exact.
  - **Proposed change:** a tolerance would change the protocol's statistic. That needs an ADR and a protocol version bump, and is not a release-engineering change.

- **Implementation revision in provenance** (from the M1.2 review). The FULL fingerprint does not hash Python code (ADR-020). Investigate an optional, caller-supplied `implementation_revision` / code revision / repository commit / model revision for reproducibility. Automatic source hashing is out of scope. There is no schema change until implementation evidence shows it is needed.
  - **Evidence from M1.3:** `TinyTransformer(n_heads=2)` and `n_heads=4` with the same seed have identical FULL fingerprints but compute different functions, because `n_heads` is a plain Python attribute and parameter shapes are unchanged. Plain constructor hyperparameters are therefore also outside v1 identity. Candidate remedies: a caller-declared model config or revision in provenance, or an opt-in declared-hyperparameter hook. This needs a decision before evidence from differently configured models is compared.

## Release blockers (`BLOCKS_PUBLIC_RELEASE`): all resolved

These had to be resolved before BeyondNN was published (public repository, PyPI). All were resolved for v0.1.0 (2026-10-01).

| ID | Item | Resolution |
|---|---|---|
| RB-1 | Code of Conduct enforcement contact | `CODE_OF_CONDUCT.md` points to the maintainer through GitHub private reporting (no personal address). |
| RB-2 | Security contact | `SECURITY.md` uses GitHub private vulnerability reporting, which is enabled on the repository. |
| RB-3 | CI had never run on GitHub | CI runs on every push and pull request; green on Linux for Python 3.10, 3.12 and 3.14 (`docs/PHASE_8_REPORT.md`). |

## Explicitly not planned
Large model training, dashboards, transformer-only APIs, auto-labelled features presented as concepts, and a global confidence scalar (without a superseding ADR).
