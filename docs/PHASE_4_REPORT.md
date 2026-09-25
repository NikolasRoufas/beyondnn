# Phase 4 Gate Report: Structured WHY / Evidence Synthesis

- **Date:** 2026-09-25
- **Baseline:** Phase 3 closed at `3bc3f41` (770 passed + 1 reported skip without Captum; 791 with Captum; clean tree).
- **Plan:** `docs/PHASE_4_PLAN.md` (`e4f2d9e`, written with the expected scenario outputs before implementation).
- **Decisions:** ADR-031; ADR-026 amended.
- **Repository:** local only, never pushed.

## 1. Executive summary

Phase 4 composes already-computed evidence into one structured INPUT → TARGET → WHY → OUTPUT view.

- OBSERVED, MEASURED, ATTRIBUTED, INTERVENTIONAL and ESTIMATED_CAUSAL remain separate sections.
- Only declared claims appear, each with every recorded test and the assessments of explicit policies.
- Limitations are the union of every source.
- Coverage states what exists and what was never evaluated: faithfulness, comprehensiveness, sufficiency, concepts.
- Composition runs nothing, and it refuses evidence that cannot honestly belong to one model/input/target context.
- No new evidence status, no score, no ranking, no generated text.

**Gate: GO WITH EXPLICIT LIMITATIONS** (§14).

## 2. What was implemented

| Piece | Details |
|---|---|
| Input identity | `InputRecord.sample_id` (record v3, migration from v2 = `None`); `beyondnn.core.samples.sample_id` (algorithm unchanged from Phase 2) |
| `EvidenceBundle` | Immutable and validated. Built only by `compose(trace, *, attributions, interventions, claims, policies)`. Not evidence. |
| API | `bnn.compose(...) -> ExplainResponse`; `ExplainResponse.from_evidence(...)`; `ExplainResponse(trace)` / `handle.explain(...)` unchanged |
| `Why` | `observations`, `activations`/`measurements`, `attributions` + `attribution_views`, `effects` + `intervention_views`, `estimated_causal`, `claims` (`ClaimView`), `assessments`, `limitations`, `provenance`, `evidence_statuses`, `by_status()`, `origin()`, `coverage` (`Coverage`), `unanswered`, `target`/`target_spec`, `question`/`not_answered` |
| Rendering | Deterministic `render()`; `to_dict()` presentation summary |
| Guards | Protocol/relation check on decisive results; the intervention evaluator hard-codes its decidable relations |
| Examples / benchmark | README walk-through (runs in tests); `examples/phase4_redundant_path.py`; `benchmarks/bench_synthesis.py` |

## 3. Semantics

### Compatibility (refused, never merged; `CompositionError` subclasses)

- **Model** (`ModelMismatchError`): every provenance record in every source trace must have the reference pass's automatic `ModelIdentity` **and** an equal `ModelDeclaration`. Declared vs. undeclared counts as different. Equal weights do not make two declared models the same.
- **Sample** (`SampleMismatchError`): the following must equal the reference input's exact `sample_id`:
  - every attribution's `sample_id` and its reference pass input;
  - every effect's INSTANCE estimand and its baseline pass input;
  - every claim's estimand.

  A reference with no identity (a pre-ADR-031 trace, or an unidentifiable input) cannot anchor target-specific evidence. A patch source keeps its own sample and is shown as source context.
- **Scope** (`UnsupportedScopeError`): FINITE_SAMPLE and POPULATION effects and claims are refused in an instance-level WHY.
- **Target** (`TargetMismatchError`): all attributions and effects share one `MetricSpec`, and every claim's target equals `MetricSpec.target()`.
  - With no target-specific evidence the target is `None` (measured-only) or the claims' single target.
  - A different target is refused, not grouped.
- **Integrity** (`EvidenceIntegrityError`):
  - each record's id is recomputed from its content;
  - every reference must resolve in its own source trace;
  - the same id with different content is refused;
  - cited evidence must be present with the stated status;
  - a decisive result must come from a protocol that justifies the claim's relation;
  - every decisive result is **re-derived** with its registered protocol's evaluator and must have the identical id;
  - unregistered protocols are refused.

### Grouping

- The sections are exactly the statuses. `ESTIMATED_CAUSAL` is its own (empty) section.
- `by_status(s)` returns the presented records of status `s`: the reference input/output, reference activations, attributions and reductions, and effects.
- Nothing is converted, averaged, or normalised across sections.

### Attribution presentation

Each attribution shows:
- method, implementation and version, parameters;
- target and its value;
- baseline (kind, digest, replaced input);
- site, leaf, call, pass;
- the raw attribution in index order (never ranked);
- explicit reductions only;
- diagnostics;
- the limitations scoped to the record.

### Intervention presentation

Each intervention shows:
- operation (and constant), site, leaf, call;
- patch-source sample (if any);
- target;
- "under this intervention the target changed by Δ (baseline → intervention)";
- scope;
- scoped limitations.

It never says "caused".

### Claims, tests, assessments

- For each claim: statement, relation, subject, estimand, and every recorded test (protocol, version, criteria, outcome, cited evidence with status).
- A causal claim without a decisive test is marked "no decisive causal test was performed for this causal claim".
- Assessments are shown per policy. If no policy was supplied: "not assessed".
- Outcomes and verdicts keep the schema vocabulary; nothing is reduced to a boolean.
- **Assessment rules:**
  - A claim is assessed only under supplied policies that pass `check_policy` and name a protocol for its relation, using every recorded result about it.
  - There is no default or universal policy.
  - For a contradicted or untested claim, "required but missing" is shown as "required protocols without a supports result".

### MIXED

Additive model, x = (3, 5), claim "a DECREASES y":
- zero `a`: effect −3, `min_effect` 1 → SUPPORTS;
- `a := 10`: effect +7 → CONTRADICTS.

The assessment is **MIXED** and both tests are rendered. If contradicting results were dropped (mutation), four tests fail.

### Limitations

- The deduplicated union (by record identity) of every source trace's limitations, in source order: reference first, then attributions, then interventions. Details are preserved: for example, two `PARTIAL_SITE_COVERAGE` records with different details both stay.
- `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE` and `NO_CLAIMS_TESTED` are added only when that kind of evidence is absent.
- There is no `EVIDENCE_CONFLICT` code, and no limitation is removed because another method exists.

### Coverage and unknowns

`Coverage` has these fields:
- `measured_internal_states`, `attribution_available`, `intervention_effect_available`, `estimated_causal_available`;
- `claims_declared`, `claims_tested`, `causal_claim_tested`;
- `faithfulness_evaluated` and `concepts_validated`: typed `Literal[False]`, and construction with anything else raises.

`Coverage.NOT_EVALUATED` lists faithfulness, comprehensiveness, sufficiency, and concept validation. `Why.unanswered` is a deterministic list of unanswered questions.

**No global confidence score:**
- no field, method, or section is named or computed as confidence, score, trust, agreement, combined, or importance;
- `test_scenario_d` asserts that none of these appear anywhere in `to_dict()`;
- the rendered text contains no `confidence:`/`=` and no percentages;
- the only uses of "confidence" are the disclaimers "not a confidence".

**No faithfulness claim:** `faithfulness_evaluated` cannot be true, and every render lists faithfulness under NOT EVALUATED. The rendered text never says anything "is faithful" (language test); the only mentions are questions marked "(not evaluated)".

**No concepts:** `concepts_validated` cannot be true. Sites are shown only as module paths, leaves, calls, and input positions. No record carries a semantic label, and VALIDATED_CONCEPT cannot occur.

## 4. Scenario results (expected values pre-registered in PHASE_4_PLAN.md; all met)

| Scenario | Result |
|---|---|
| A: measured only | statuses {observed, measured}; target `None`; no attributions, effects, or claims; `NO_ATTRIBUTION`/`NO_CAUSAL_EVIDENCE`/`NO_CLAIMS_TESTED`; faithfulness/concepts False. The render equals the Phase-3 renderer's output **exactly** (checked against `3bc3f41`'s code on three traces) plus the NOT EVALUATED block. |
| B: attribution only | IG [6, 15]; ATTRIBUTED_TO SUPPORTS → SUPPORTED; NECESSARY_FOR → NOT_APPLICABLE → **UNTESTED** (missing `intervention_threshold`); "no decisive causal test was performed" is shown |
| C: intervention only | effect −3 (8 → 5), INTERVENTIONAL, instance; `ZERO_ABLATION_MAY_BE_OOD`; `NO_ATTRIBUTION` |
| D: both | attribution [[3.0]] and effect −3.0 in separate sections; no combined value; no generated claim |
| **E: redundant path** | IG at `p` = 3.0 → "p ATTRIBUTED_TO y ≥ 2" **SUPPORTED**; zeroing `p`: 6 → 3 (effect −3) → "p NECESSARY_FOR y, min_effect 6" **CONTRADICTED**. Both are shown side by side and neither replaces the other. No conflict code exists. |
| F: mismatched sample | refused. The reference (5, 3) vs attribution (3, 5), which have identical summary statistics, is also refused thanks to exact identity. |
| G: mismatched target | refused (evidence vs evidence, evidence vs claim, attribution vs attribution) |
| H: mixed | MIXED, both tests shown |

## 5. Refusal tests

The following are all refused, each by at least two independent tests:
- different sample;
- different target;
- different model weights (attribution; intervention);
- different `ModelDeclaration`, and declared vs undeclared (attribution; intervention);
- finite-sample effect;
- claim about another input;
- a forged intervention result, and a forged attribution result;
- a record mutated after creation (effect; attribution);
- a pre-ADR-031 trace anchoring target evidence;
- a multi-pass reference;
- wrong types.

## 6. Persistence and reconstruction

- There is no explanation file format (no pickle, no callbacks).
- An explanation is **reconstructed**: save the traces, reload them (`load_trace`, `AttributionResult.from_trace`, `InterventionResult.from_trace`), and compose again. `to_dict()` and `render()` are identical to the original (tested, and in the clean-wheel smoke).
- Claims recorded with the evidence persist in their traces. Extra caller claims and policies are the caller's to supply again.
- `InputRecord` v2 → v3 migration: `sample_id = None`. Old traces still load and still give measured-only responses.

## 7. No computation during synthesis

- `EvidenceBundle` never receives a model: result objects carry traces, not models. So there is no path from composition to execution.
- **Tests:**
  - a tripwire model that raises if run;
  - patched `torch.autograd.grad`, `Tensor.backward` and `nn.Module.register_forward_hook` that raise;
  - an RNG-state check;
  - hook counts;
  - trace records and retained tensors unchanged by identity;
  - Captum never imported (clean wheel).

## 8. Deterministic rendering

- Two independent builds of the same scenario give identical `render()` and `to_dict()`.
- Ordering: reference input/output, execution-order activations, caller order for attributions and interventions, caller claims then recorded claims, tests in recorded order, limitations in source then trace order.
- No set iteration.

## 9. Validation

| Check | Result |
|---|---|
| Python 3.10 / 3.12 / 3.14, without Captum (`-W error -rsxX`) | **808 passed**, 1 skipped (Captum cross-check module, reported) |
| Python 3.10 / 3.12 / 3.14, with Captum 0.9.0 | **829 passed**, 0 skipped |
| ruff, ruff format, mypy `--strict` (with and without Captum) | clean |
| sdist (now includes `examples/`) + wheel | built |
| Clean venv (Python 3.12, torch 2.14), wheel only, without and then with Captum | Phase 1–4 smokes and all 4 README examples pass; composition does not import Captum |

## 10. Benchmark (composition only; evidence computed beforehand)

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64; warm-up 3, 30 iterations; median / p90 ms. Evidence computed beforehand (not timed).

| composition | source records | compose | render | rendered chars |
|---|---|---|---|---|
| measured only | 7 | 0.173 / 0.177 | 0.046 / 0.048 | 1559 |
| trace + attribution | 19 | 0.447 / 0.452 | 0.074 / 0.075 | 3793 |
| trace + intervention | 23 | 0.553 / 0.559 | 0.062 / 0.064 | 2890 |
| trace + attribution + intervention + claims | 41 | 1.146 / 1.154 | 0.092 / 0.093 | 4609 |

Synthesis is about 1 ms for the full case and grows with the number of records. Nothing is pathological, and nothing was optimised.

## 11. Mutation tests (all caught; scratch copies)

| Mutation | Tests that fail |
|---|---|
| ATTRIBUTED and INTERVENTIONAL merged in `by_status` | 3 |
| attributions merged into the effects section | 3 |
| attribution evidence may support NECESSARY_FOR | 23 |
| claims assessed under policies without a protocol for their relation | 5 |
| intervention protocol justifies ATTRIBUTED_TO (registry widened) | 1, plus a second defence (below) |
| mismatched sample accepted | 2 |
| mismatched model accepted | 2 |
| mismatched target accepted | 2 |
| different ModelDeclaration accepted | 2 |
| WHY copies records instead of referencing them | 2 |
| limitations dropped (reference only) | 2 |
| MIXED collapsed (contradicting results dropped) | 4 |
| missing causal test shown as performed | 2 |
| `faithfulness_evaluated` becomes true | 2 |
| `concepts_validated` becomes true | 2 |
| renderer says "caused" for attribution | 2 |
| forged claim result accepted (no re-derivation) | 2 |
| synthesis uses autograd (stand-in for execution; see §7) | 2 |
| synthesis changes RNG | 2 |
| tampered record id accepted | 2 |

- **Survivor found and fixed:** the first run had one survivor. Widening the protocol registry so that `intervention_threshold` justified ATTRIBUTED_TO was caught by nothing; with a hand-built spec, intervention evidence could then have decided an attribution claim.
- **Fixes:**
  - the bundle now refuses a decisive result whose protocol does not justify the claim's relation;
  - the intervention evaluator hard-codes its decidable relations, independent of the registry;
  - new tests pin both.
- **Remaining single catch:** the registry mutation itself is still caught by one test, but its consequence is now blocked by the evaluator's independent defence (tested with a monkeypatched registry).
- The first run also had many single-test catches; each got an independent second test.

## 12. Language audit

- **Searched:** the whole repository (code, docstrings, errors, README, docs, examples, benchmarks) for why, reason, cause/caused, important, faithful, confidence, explanation, decision, thought, understand, attention, and "the model used/thought/decided/focused/understood".
- **Result:** every remaining occurrence is one of the following:
  - a disclaimer or negation ("does not answer what caused the output", "not a confidence score", "not the reason");
  - a licensed technical term ("causal effect", "causal_mask", attention layers, "decision criteria", "decided by" a named protocol);
  - an internal variable name.
- There are no unlicensed statements.
- **Changes made:**
  - the new "does not answer" wording was changed to avoid "is faithful";
  - a README sentence using "because" to describe the toy model's construction was reworded.
- **Automated guard:** a test forbids caused/causes/because/reason/important/"the model thought/used/decided/focused/understood"/"is faithful"/"explanation is"/confidence values/percentages in every rendered output, except lines that explicitly list unanswered questions.

## 13. Backward compatibility

- Phase-1 `handle.explain(x, sites=...)`, `ExplainResponse(trace)`, `from_trace`, `Why(trace)`, `why.activations`, and the Phase-1 README example all work unchanged.
- The measured-only render and limitations are verified identical to the Phase-3 renderer, plus the NOT EVALUATED block.
- All Phase 1–3 tests pass. The only changed test is the persistence downgrade helper, which now also removes the new v3 field, plus the updated API/audit inventories.
- No evidence status changed meaning, and Captum remains optional.
- The top-level API gained exactly one name, `compose`. The full list, 17 names:
  - `EstimandScope`, `EvidenceStatus`, `Outcome`, `Relation`, `TraceResult`, `Verdict`, `__version__`;
  - `attribute`, `attribution`, `compose`, `instrument`, `intervene`, `interventions`, `load_trace`, `recording`, `schema`, `trace`.

## 14. Limitations

- Instance-level, single-target, single-model only; no multi-target, multi-sample, or cross-model study views.
- Reference-pass measurements only: activations measured during intervention passes and attribution reference passes are reachable via `origin` and the source traces, but they are not listed as MEASURED sections.
- Pre-ADR-031 traces cannot anchor target-specific evidence; they must be re-traced.
- Declared identity (model, caller metrics) is only as good as the caller's declarations.
- **Revalidation boundaries:**
  - revalidation covers the two registered protocols; any future protocol needs its own revalidator, or composition refuses it;
  - NOT_APPLICABLE results that cite nothing are accepted as recorded (they can support nothing).
- `to_dict()` is a presentation summary, not a serialisation format.
- The exact-input digest can confirm a guessed low-entropy input (privacy note, ADR-031).
- CPU-only, as in all phases.

## 15. Gate

**GO WITH EXPLICIT LIMITATIONS.**

| Criterion | Status |
|---|---|
| Measured-only behaviour remains correct | ✅ (identical to Phase 3 plus NOT EVALUATED) |
| ATTRIBUTED and INTERVENTIONAL compose without semantic collapse | ✅ |
| Sample/model/target compatibility enforced | ✅ (exact identity, declaration included) |
| Claim protocol boundaries enforced | ✅ (re-derivation, relation check, evaluator defence) |
| MIXED preserved | ✅ |
| Limitations preserved | ✅ |
| No global confidence score | ✅ |
| Faithfulness explicitly unevaluated | ✅ |
| Concepts explicitly unvalidated | ✅ |
| Synthesis runs no model/scientific computation | ✅ |
| Deterministic renderer | ✅ |
| All validation green | ✅ |
| Fresh-wheel examples work (with and without Captum) | ✅ |
| No known silent evidence misrepresentation | ✅ |

**Reasons:**
- Every required property holds and is covered by tests and mutations.
- The one mutation survivor exposed a real boundary gap, which was fixed with two independent defences.
- The limitations in §14 are refused or explicitly stated scope boundaries, not silent collapses. That is why the gate is not an unqualified GO.

## 16. Before Phase 5

- Owner review of this report and ADR-031, including the `InputRecord` v3 identity change and its privacy note.
- **What Phase 5 must add** (none of it is implemented here):
  - declared faithfulness protocols with formal definitions: comprehensiveness, sufficiency (input and internal), and deletion/insertion or ablation curves;
  - controlled ground-truth faithfulness tasks, where the true mechanism is known;
  - method-specific diagnostics beyond IG completeness;
  - `Coverage.faithfulness_evaluated` may become true only through such a protocol's recorded results, which needs a schema decision for faithfulness evidence;
  - multi-sample (FINITE_SAMPLE) composition if faithfulness is evaluated over datasets.

**Public release** remains blocked by RB-1 (Code of Conduct contact placeholder), RB-2 (security contact placeholder), and RB-3 (GitHub CI never run). Nothing has been pushed or published.
