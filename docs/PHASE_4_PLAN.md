# Phase 4 Plan: Structured WHY / Evidence Synthesis

- **Status:** plan, written before implementation. The expected structured outputs below were written before any scenario ran.
- **Baseline:** Phase 3 closed at `3bc3f41` (gate: GO WITH EXPLICIT LIMITATIONS). Clean tree: 770 passed plus 1 reported skip without Captum; 791 passed with Captum 0.9.0.
- **Purpose:** combine compatible evidence, claims, assessments, provenance, and limitations into one structured WHY view, while preserving exactly what each piece of evidence can and cannot establish.
- **What this is not:** Phase 4 is synthesis and presentation. It is **not** a new interpretability method, and it produces **no new evidence status**.

## Principles (fixed before code)

1. **No collapse.**
   - OBSERVED, MEASURED, ATTRIBUTED, INTERVENTIONAL, and ESTIMATED_CAUSAL stay separate sections.
   - Numbers from different methods are never combined, averaged, normalised, ranked, or turned into a score.
   - There is no confidence, trust, or agreement score, and no automatic "most important" anything.
   - Attribution and intervention disagreeing is not an `EVIDENCE_CONFLICT`: they answer different questions.
2. **Composition runs nothing.** Composing and rendering never:
   - execute the model;
   - use autograd or Captum;
   - install hooks;
   - touch RNG;
   - modify tensors.

   Only existing records are read and validated. Assessments are derived deterministically from existing results under caller-supplied, registry-checked policies.
3. **Trace remains the source of truth.**
   - Views reference the original record objects, and nothing is copied into weaker duplicates.
   - Evidence from several traces goes into an explicit, immutable composition container.
   - The container is not evidence and is not a second database.
4. **One explanation context.** One model identity (automatic and declared), one input sample (exact identity), one output, and at most one scalar target. Anything else is **refused**, never merged.
5. **Claims are only ever declared.**
   - Nothing generates a claim from a score or an effect.
   - Every test result is re-derived from its cited evidence before it is shown. A forged or inapplicable result is refused.
6. **Coverage, not confidence.** The view states which kinds of evidence exist and which evaluations were never performed: faithfulness, comprehensiveness, sufficiency, and concept validation are always `False` / not evaluated in Phase 4.
7. **Backward compatibility.**
   - `handle.explain(x, sites=...)` and `ExplainResponse(trace)` keep their Phase-1 meaning and fields.
   - Measured-only rendering keeps every Phase-1 line and only appends a NOT EVALUATED block.

## A required identity fix: the observed input carries its exact sample identity

- **Problem.** Target-specific evidence (`AttributionRecord.sample_id`, `CausalEffect.estimand.sample_id`, claim estimands) identifies its input exactly: SHA-256 over tensor bytes plus JSON scalars. A Phase-1 `InputRecord` with default `summary` retention does not.
  - Summary statistics cannot distinguish (3, 5) from (5, 3).
  - So "is this attribution about the traced input?" would be unanswerable. Accepting a statistics match would allow silent misassignment.
- **Decision.**
  - `InputRecord` gains `sample_id` (record version 3; v2 payloads migrate with `sample_id = None`, never invented).
  - Recording computes it for every root pass with the same algorithm as `interventions.sample_id`, which moves to `beyondnn.core.samples` unchanged. When an input cannot be identified, the value is `None`.
  - Composition with target-specific evidence requires exact equality, and refuses when an identity is missing.
- **Privacy note.** The digest can confirm a guessed low-entropy input. Phases 2 and 3 already recorded the same digest in effects, attributions, and claims; it now also appears in plain traces. This is documented in ADR-031.

## Design

| Piece | Where | Notes |
|---|---|---|
| `EvidenceBundle` | `beyondnn/explain/bundle.py` | Frozen and validated on construction; not evidence. Holds the reference `TraceResult` (single pass: INPUT/OUTPUT/MEASURED), `AttributionResult`s, `InterventionResult`s, claims, registry-checked `AssessmentPolicy`s, and derived `Assessment`s. |
| Composition | `EvidenceBundle.compose(trace, *, attributions=(), interventions=(), claims=(), policies=())`; `bnn.compose(...) -> ExplainResponse` | `bnn.explain` cannot be a function: `beyondnn.explain` is the subpackage, and importing it would rebind the name. `compose` also says what happens: composition, not method execution. |
| `ExplainResponse` | amended (ADR-026 amendment) | `ExplainResponse(trace)` still works and composes a trace-only bundle. `from_evidence(...)` and `bundle` are new. `input`/`output` are unchanged. |
| `Why` | amended | `observations`, `measurements` (= `activations`), `attributions` (records) plus `attribution_views`, `effects` plus `intervention_views`, `estimated_causal`, `claims` (`ClaimView`), `assessments`, `limitations`, `provenance`, `evidence_statuses`, `by_status()`, `origin()`, `coverage`, `unanswered`, `target` |
| Views | `AttributionView`, `InterventionView`, `ClaimView`, `Coverage` | Frozen dataclasses that reference original records only |
| Rendering | `render()` / `to_dict()` | Deterministic and presentation only. `to_dict` exposes ids and values, not a second codec. |

## Validation on composition (all refusals, `CompositionError` subclasses)

- **Records.** Every record's id is recomputed from its content. Every reference resolves inside its own trace. The same id carrying different content across traces is refused.
- **Reference.** The trace must have exactly one pass.
- **Model.** Every provenance of every included trace must have the reference's automatic `ModelIdentity` and an equal `ModelDeclaration`.
  - Two passes of one controlled comparison are covered by this, because Phase 2 already requires equal fingerprints.
- **Sample.**
  - Every attribution's `sample_id`, every effect's INSTANCE estimand, every attribution reference pass, every intervention baseline pass, and every claim's estimand must equal the reference input's `sample_id`.
  - FINITE_SAMPLE and POPULATION evidence and claims are refused in an instance-level WHY.
  - A patch-source pass has its own sample and is presented as **source context**.
- **Target.** All attributions and effects must share one `MetricSpec`, and every claim's target must equal `MetricSpec.target()`. With no target-specific evidence the target is `None` (measured-only), or the claims' common target.
- **Claim tests.**
  - Every `ClaimTestResult` in an included trace must use a registered protocol, and must cite evidence that is present with the stated status.
  - Its outcome is re-derived with the protocol's own evaluator (`intervention_threshold`: effect and intervention; `attribution_threshold`: record and retained tensor) and must match exactly.
  - NOT_APPLICABLE results cite nothing and cannot support anything, so they are accepted as recorded.
- **Policies.**
  - Each policy passes `check_policy` (registry).
  - A claim is assessed only under policies that name a protocol for its relation, using every result about the claim (no cherry-picking).
  - Without such a policy the claim is shown as "not assessed".

## Expected structured outputs (written before running)

Common setting: inputs `x = (x0, x1)` of shape (1, 2) and target `select([0, 0])` unless stated otherwise.

| Scenario | Setup | Expected |
|---|---|---|
| **A: measured only** | `TinyMLP`, `handle.explain(x, sites=["**"])` | Statuses {observed, measured}. Target `None`. `attributions == effects == claims == ()`. Limitations contain `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE`, `NO_CLAIMS_TESTED`. Coverage: measured yes, attribution no, intervention no, faithfulness_evaluated False, concepts_validated False. Render: every Phase-1 line, then NOT EVALUATED. |
| **B: attribution only** | `Linear` (2·x0 + 3·x1), x = (3, 5), IG zero baseline (n = 16) at the input. Claims: "x ATTRIBUTED_TO y, units (0,), min 5" and "x NECESSARY_FOR y", both tested with `attribution_threshold`. Policies: attribution and intervention policies. | Attribution value [6, 15], delta 0. No effects. The ATTRIBUTED_TO test SUPPORTS (6 ≥ 5), assessment SUPPORTED. The NECESSARY_FOR test is NOT_APPLICABLE, assessment UNTESTED with `required_but_missing = (intervention_threshold,)`. Coverage: `causal_claim_tested` False. `NO_CAUSAL_EVIDENCE` present, `NO_ATTRIBUTION` absent. |
| **C: intervention only** | `Additive`, x = (3, 5), zero `a` | Effect −3 (baseline 8, intervention 5), INTERVENTIONAL, INSTANCE. `ZERO_ABLATION_MAY_BE_OOD` present. `NO_ATTRIBUTION` present, `NO_CAUSAL_EVIDENCE` absent. No attributions. |
| **D: both** | `Additive`, x = (3, 5): IG zero-baseline attribution at `layer("a")`, plus zero `a` | Attribution [[3.0]] and effect −3.0, in separate sections. No combined number, no generated claim (`claims == ()`). |
| **E: redundant path (flagship)** | `Redundant`, x = (3, 5): IG at `layer("p")`, plus zero `p`. Claims: "p ATTRIBUTED_TO y, min 2" (`attribution_threshold`) and "p NECESSARY_FOR y, min_effect 6" (`intervention_threshold`). | Attribution 3.0 → ATTRIBUTED_TO SUPPORTED. Effect −3.0 with the output staying 3.0 → NECESSARY_FOR CONTRADICTED. Both shown, neither replaced. No `EVIDENCE_CONFLICT` code exists. Limitations include `LAYER_ATTRIBUTION_PARTIAL_COVERAGE`, `ATTRIBUTION_BASELINE_ASSUMPTION`, `ATTRIBUTION_NUMERICAL_APPROXIMATION`, `ZERO_ABLATION_MAY_BE_OOD`. |
| **F: mismatched sample** | Attribution on (3, 5), intervention on (1, 1), reference (3, 5) | Refused (`SampleMismatchError`) |
| **G: mismatched target** | `TinyMLP`: attribution on `select([0, 0])`, intervention on `select([0, 1])` | Refused (`TargetMismatchError`) |
| **H: mixed** | `Additive`, x = (3, 5); claim "a DECREASES y". Zero `a` (effect −3, min 1: SUPPORTS) and constant `a := 10` (effect +7: CONTRADICTS), under the intervention policy. | Assessment MIXED, and both results are shown with their own evidence |
| Extra refusals | A different model (other weights); a different `ModelDeclaration` revision; a finite-sample effect; a claim about another input; a forged ClaimTestResult; a loaded pre-Phase-4 trace with no input identity plus target evidence | All refused |

## Milestones

1. Plan, ADR-031, and the ADR-026 amendment.
2. Input identity (`InputRecord` v3), the bundle, validation, and the Why sections.
3. Claim/test/assessment presentation and revalidation.
4. Renderer, `to_dict`, README, and examples.
5. Benchmark (composition only), language audit, mutation tests, and report.
