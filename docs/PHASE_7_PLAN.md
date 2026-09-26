# Phase 7 Plan: Scientific Audit Framework (pre-registered)

- **Status:** committed **before** any Phase-7 implementation or experiment. Hypotheses, thresholds, scenarios, sample selections and gate criteria below are fixed. Anything changed after results are seen is recorded under Deviations at the end with its reason; nothing is edited in place.
- **Start:** HEAD `25fe462` (Phase 6 gate) → `6faca63` (Phase 7 literature and differentiation).
- **Baseline:** 934 passed + 1 skipped without Captum and 955 with Captum, on Python 3.10, 3.12 and 3.14; ruff, format and mypy --strict clean.
- **Prior knowledge that is not blind.** Phase 5.5 and Phase 6 results are known to the author:
  - SUPPORTS shares per replacement;
  - concept outcomes and FP/FN rates;
  - K4 is the only validated concept;
  - BERT pooler length/negation are decodable but unused;
  - the covariance null over-matches K7.

  Where a Phase-7 hypothesis restates a known Phase-5.5/6 fact in audit terms, it is marked **[consistency]**, not **[prediction]**. Its value is that the audit reproduces a hand analysis mechanically, not that it predicts something new.

## 1. Audit definition

- **What an audit is.** An audit is a deterministic, model-free function:

  ```
  audit(evidence: recorded traces, plan: AuditPlan) -> AuditReport
  ```

- **For every claim and concept the plan declares, it reports:**
  - a **standing** derived by fixed rules from re-derived, in-scope evidence;
  - the **findings** that qualify or block that standing;
  - the distributions, counterexamples, sensitivities, coverage and limitations behind it.
- **What an audit never does:**
  - runs a model;
  - computes a score (ADR-007);
  - resolves a contradiction;
  - fills a gap with an assumption;
  - says an explanation is trustworthy, reliable or correct.
- **What missing evidence means:** NOT_EVALUATED. It never means "probably supported" or "probably false".

## 2. Scope

**In scope:**
- claims decided by the six registered protocols (`intervention_threshold`, `attribution_threshold`, `comprehensiveness`, `sufficiency`, `concept_encoding`, `concept_intervention`);
- faithfulness diagnostics;
- concept validations and generated labels;
- instance, finite-sample and population claims;
- per-sample claim families over a declared sample set.

**Out of scope:**
- uncertainty intervals and confidence sequences;
- DAS / interchange training;
- external ground-truth suites;
- models above BERT-tiny;
- Trace 3B (see `docs/roadmap/TRACE3B_FUTURE.md`);
- anything requiring a model run during the audit.

## 3. Threat model

The audit must not be fooled by:
- **T1** an edited or forged result, assessment, counterexample list or validation → re-derivation from raw records;
- **T2** evidence from another checkpoint, declared model, sample or dataset → provenance and scope checks with exclusion and a finding;
- **T3** a claim stated more broadly than its evidence (type, scope, invariance) → structural overclaim checks;
- **T4** selective reporting *within the supplied evidence* (only the favourable k, replacement, threshold or null is emphasised) → every matched result counts, and disagreement is a finding;
- **T5** missing evidence treated as positive or negative → NOT_EVALUATED;
- **T6** an audit report edited after the fact → `verify_report` re-runs and compares.

**Not defended:**
- **N1** evidence that was never recorded or not supplied. The audit sees only what it is given. Selective *recording* is invisible; only a declared invariance that was not tested is visible (the MISSING_EVIDENCE finding).
- **N2** a dishonest plan, for example one that declares no invariance. The plan is part of the report and is itself auditable by a reader.
- **N3** bugs in BeyondNN's own protocols (mitigated by tests and mutation runs, not by the audit).

## 4. Inputs

```python
bnn.audit(evidence, *, plan, model=None) -> AuditReport
```

- **`evidence`:** a sequence of any of:
  - `TraceResult` objects;
  - paths to saved trace directories (loaded with `load_trace`);
  - Phase-3–6 result objects (their traces, including nested attribution / test / encoding / use traces, are taken; nothing else about the object is trusted).
- **`plan`:** an `AuditPlan` (§28).
- **`model`:** optional. If given, it is fingerprinted and must equal `plan.checkpoint`, or the audit refuses (`ModelMismatchError`). The model is never run.
- **Why this differs from the suggested `bnn.audit(model, dataset, plan=plan)`:** the audit is model-free, and datasets and samples are declared in the plan (sample ids, concept-dataset ids), matching `compose` (which also takes records, not a model).

## 5. Outputs

- **`AuditReport`** (immutable), with these sections:
  - **scope:** plan, plan id, checkpoint, declared model, samples, datasets;
  - **evidence:**
    - digest over sorted record ids;
    - trace count, record count;
    - included, excluded and failed records with reasons;
  - **claim inventory:** every claim record found in the evidence, plus every plan claim;
  - **claim audits:** per plan claim:
    - standing, or a per-sample distribution;
    - findings;
    - matched and related evidence;
    - axis values;
    - counterexamples;
    - limitations;
  - **concept audits:** per plan concept;
  - **report-level findings:** provenance, integrity, scope;
  - **coverage;**
  - **limitations:** code → occurrences.
- **Derived views (properties):**
  - `supported`, `contradicted`, `mixed`, `assumption_sensitive`, `inconclusive`, `unsupported`, `not_evaluated`;
  - `missing_evidence`, `missing_controls`, `counterexamples`, `protocol_disagreements`;
  - `sensitivity(axis)`;
  - `concept_findings`, `faithfulness_findings`, `provenance_findings`.
- **Serialization and verification:**
  - `report.to_dict()` gives deterministic JSON (sorted keys, no floats beyond what records hold);
  - `report.render()` gives prose rendered **only** from the structured data;
  - `report.save(path)`, `load_report(path)`, `verify_report(document, evidence, plan)`.

## 6. Claim inventory

- **Inventory entries.** Every `Claim` record in the included evidence is inventoried with:
  - id, relation, subject (site / units / unit axes / feature), target, estimand scope and sample;
  - protocols tested, recorded outcomes, and re-derived verdicts under the policies recorded with it;
  - the plan claims it matched (by formal structure, ADR-016; never by id or statement text).
- **Plan claims with no match** appear with standing NOT_EVALUATED.
- **Evidence claims matching no plan claim** are listed as `unaudited` (inventory only). They are not judged.

## 7. Evidence inventory

- **Ingestion:**
  - Every trace is integrity-checked (`_check_source`: ids match content, references resolve, provenance resolves).
  - One id carrying two contents across traces is an integrity failure.
- **What each ClaimTestResult carries:**
  - **evidence statuses** it cites;
  - **provenance:** checkpoint digest and declared model;
  - **sample scope:** instance sample id, finite-sample set id, or concept dataset id;
  - **axis coordinates** (§11–§14);
  - **controls** declared or absent;
  - **re-derivation status:** OK / FAILED (reason) / NOT_DECISIVE (no evidence cited).
- **Also inventoried:**
  - attribution records (method, sample, site);
  - intervention records (operation, replacement digest);
  - evidence selections;
  - protocol results (diagnostics);
  - concept records: feature, concept, dataset, generated label, validation, activation.

## 8. Contradiction detection

- **Contradiction.** For one plan claim (per sample, for per-sample claims): its matched decisive results include both SUPPORTS and CONTRADICTS.
- **Assumption-sensitive.** The disagreement is **explained** by an axis if some SUPPORTS result and some CONTRADICTS result differ in exactly that axis and agree on all others. If every SUPPORTS/CONTRADICTS pair is explained by one or more axes, the standing is ASSUMPTION_SENSITIVE with one finding per axis, listing the values on each side.
- **Mixed.** A pair that differs in no axis is an **unexplained** CONTRADICTION finding, and the standing is MIXED. Pairs that differ only in several axes at once are listed as "combined: a+b".
- **No auto-resolution.** Nothing is resolved automatically: both sides are listed with record ids.
- **Cross-relation disagreement (PROTOCOL_DISAGREEMENT, QUALIFYING):**
  - **(a)** a causal plan claim is CONTRADICTED while related ATTRIBUTED_TO results on the same subject, target and sample SUPPORT (attribution vs intervention);
  - **(b)** two plan claims with the same subject or selection and the same sample but different relations (for example NECESSARY_FOR via comprehensiveness vs SUFFICIENT_FOR via sufficiency) have per-sample standings SUPPORTED vs CONTRADICTED.

## 9. Missing-evidence detection

MISSING_EVIDENCE is raised for:
- **(a)** a protocol the claim's requirement policy requires, with no recorded result (verdict INCONCLUSIVE with `required_but_missing`);
- **(b)** an axis the claim declares invariance over with fewer distinct tested values than required, or a required value that was never tested (BLOCKING);
- **(c)** plan samples of a per-sample claim with no matched evidence: counted as NOT_EVALUATED in the distribution, never as SUPPORTED or CONTRADICTED;
- **(d)** a concept with no validation (NOT_EVALUATED);
- **(e)** a fitted feature whose fitting cannot be re-derived from the supplied evidence (QUALIFYING; reported, not assumed).

## 10. Unsupported-claim detection (structural overclaims)

Checked mechanically. **All are BLOCKING and give the standing UNSUPPORTED.** Each check below applies when no deciding evidence exists and the named non-deciding evidence does.

- **O1, ATTRIBUTED → causal.** Causal plan claim; the related evidence is ATTRIBUTED (ATTRIBUTED_TO results, or attribution records for the sample, site and method of a selection claim).
  Finding: EVIDENCE_TYPE_MISMATCH `attribution_is_not_intervention`.
- **O2, probe → use.** Use claim (DECREASES / INCREASES / SUFFICIENT_FOR on a feature); the related evidence is ENCODES results on that feature.
  Finding: EVIDENCE_TYPE_MISMATCH `decodability_is_not_use`.
- **O3, GENERATED → validated.** A plan concept asserted VALIDATED_CONCEPT with a GENERATED label and no VALIDATED validation in scope.
  Finding: EVIDENCE_TYPE_MISMATCH `generated_label_is_not_validation`.
- **O4, instance → population / finite sample.** The plan claim has POPULATION or FINITE_SAMPLE scope; the related evidence has a narrower estimand (instance or another sample set) on the same structure.
  Finding: SCOPE_MISMATCH `narrower_estimand`, with the instance distribution attached as context.
- **O5, one replacement / threshold / k / null / dataset → universal.** The claim declares `invariant_over` an axis, but fewer than the required values were tested.
  Finding: MISSING_EVIDENCE `invariance_untested` (§9b).
- **O6, missing required controls.** The requirement demands controls, and no SUPPORTS result with declared controls exists.
  Finding: MISSING_CONTROL.

## 11. Assumption sensitivity

The axis coordinates of a result are string keys computed from records only. Axes: `protocol`, `threshold`, `replacement`, `k`, `null`, `method`, `dataset`.

| protocol | threshold | replacement | k | null | method | dataset |
|---|---|---|---|---|---|---|
| intervention_threshold | criteria digest | intervention operation + value digest (+ patch source sample) | — | `none` | — | — |
| attribution_threshold | criteria digest | — | — | `none` | method identity + baseline | — |
| comprehensiveness / sufficiency | criteria digest | `spec.params.replacement` | selection k | control strategy (`count` / `magnitude` / `none`) | selection source method | — |
| concept_encoding | criteria digest | — | — | sorted control kinds + distributions | — | concept dataset id |
| concept_intervention | criteria digest | mode + reference identity | — | sorted control kinds + distributions | — | concept dataset id |

Assumption sensitivity is §8 over all axes.

## 12. Replacement sensitivity

The `replacement` axis (table §11): interventions, faithfulness replacements and concept references.

## 13. Threshold sensitivity

Threshold sensitivity is covered two ways:

- recorded results with different criteria (the `threshold` axis);
- **re-evaluation under declared alternative criteria.** An `EvidenceRequirement` may declare `alternatives` (protocol + criteria). Every matched result of that protocol is re-evaluated from its recorded evidence (pure functions: `interventions.evaluate_claim`, `faithfulness.evaluate`, `concepts.evaluate_use`, and `evaluate_encoding` on re-derived activations). Re-evaluations are labelled as such; they never enter the verdict. A flip is a disagreement on the `threshold` axis for §8.

## 14. Null sensitivity

The `null` axis. Examples: count- vs magnitude-matched controls; isotropic vs covariance random directions; controls vs no controls (`none`).

## 15. Counterexample policy

- **The rule.** `CounterexampleRule` declares:
  - `max_counterexample_fraction`: per-sample claims, the fraction of evaluated samples with standing CONTRADICTED;
  - `max_false_positive_rate` and `max_false_negative_rate`: concepts.

  Each may be `None`, meaning "no cap declared". This is stated in the report, never defaulted.
- **Counterexamples are always listed** with sample identities: per-sample CONTRADICTED samples, and concept FP/FN sample ids from the re-derived `concept_counterexamples` record.
- **Severity:**
  - Any counterexample: COUNTEREXAMPLE_FOUND (QUALIFYING).
  - A cap exceeded: BLOCKING. For a concept this gives UNSUPPORTED.
  - A per-sample claim has no single standing (§21); a cap breach is reported as a BLOCKING claim-level finding.

## 16. Concept audit

**Per plan concept** (concept record id, asserted status, concept policy name/version), over in-scope validations (plan checkpoint, plan datasets, the named policy; others are SCOPE_MISMATCH and excluded):

- **Re-derivation** at trace level:
  - encoding and use results, counterexamples and assessments via `concepts.verify` trace-level functions;
  - validation summaries and derived status;
  - fitted features from any supplied trace that holds the train-split activations (the encoding trace does). If none: MISSING_EVIDENCE `feature_derivation_not_rederived`.
- **Standing (never "rejected", per ADR-039):**
  - SUPPORTED: every in-scope validation is VALIDATED_CONCEPT and no cap is exceeded;
  - ASSUMPTION_SENSITIVE: validations disagree and the disagreement is explained by an axis (reference / null / dataset);
  - UNSUPPORTED: validations exist, none is VALIDATED, or a cap is exceeded;
  - NOT_EVALUATED: no in-scope validation.
- **Findings (codes):**
  - `decodable_not_used`: ENCODES supported, every use claim not supported;
  - `generated_label_unverified`: label source GENERATED. INFORMATIONAL if validated, BLOCKING (O3) if not;
  - `null_sensitive`: ENCODES results for the same feature, concept and dataset disagree across control distributions;
  - `replacement_sensitive`: use results disagree across references;
  - `counterexample_heavy`: a cap is exceeded, with ids;
  - `controls_defeat_encoding`: encoding CONTRADICTS although its AUROC ≥ `plan.naive_auroc`, if declared. It shows that a naive reading would have accepted what the controls reject;
  - `feature_encodes_several_concepts`: the same feature has SUPPORTED ENCODES for ≥ 2 concepts (a polysemanticity indicator, not a proof);
  - SAE limitation codes, surfaced.

## 17. Faithfulness audit

Faithfulness claims are per-sample claims whose subject is a **selection** (site; source method, or `declared`; optional fixed k). They match comprehensiveness / sufficiency results whose trace holds an `EvidenceSelection` with:
- the same provenance as the result;
- the same sample and site;
- a source attribution record of that method (or source DECLARED);
- the k given in the plan, if one was given;
- selected units equal to the claim's units.

Findings:
- **favourable k or replacement only:** `invariant_over` k / replacement untested (O5), or disagreement across k / replacement (§8);
- **absolute pass vs controls:** SUPPORTS without declared controls → MISSING_CONTROL if required. Uncontrolled SUPPORTS against controlled CONTRADICTS → disagreement on the `null` axis (`none` vs strategy);
- **comprehensiveness vs sufficiency:** §8(b) PROTOCOL_DISAGREEMENT;
- **diagnostics** (curves, stability, method agreement): re-verified and listed as context. They never decide a standing.

## 18. Provenance audit

- **Checkpoint and declared model.** Every result's provenance checkpoint must equal `plan.checkpoint`, and its declared model must equal `plan.declared_model`. The declared model is explicit; `None` means "no declaration".
- **Mismatches** are excluded with PROVENANCE_MISMATCH, which is:
  - report-level;
  - attached to each plan claim whose structure they match;
  - BLOCKING if nothing in-scope remains, QUALIFYING otherwise.
- **No cross-checkpoint composition.** The audit never composes evidence across checkpoints.

## 19. Identity audit

- **Record identity:**
  - ids are recomputed from content;
  - one id with two contents fails;
  - references must resolve.
- **Semantic identity** is by formal structure: relation, subject, target and estimand.
- **Sample identity:**
  - instance evidence must be about a plan sample (else SCOPE_MISMATCH `sample_out_of_scope`);
  - finite-sample evidence must be about the declared set id;
  - concept evidence must be about a plan dataset.

## 20. Coverage

- **Reported:**
  - plan claims by standing (counts);
  - per-sample claims: samples with any matched evidence vs plan samples;
  - evidence statuses present;
  - protocols run vs the protocols requirements name;
  - per claim, the distinct values per axis;
  - declared invariances untested;
  - concepts evaluated.
- **Coverage, not confidence:** no ratio is combined into a number about the whole audit.

## 21. Dataset aggregation

- **Per-sample (INSTANCE-scope) plan claims** are audited per plan sample. The claim-level output is:
  - the distribution of per-sample standings, for example `SUPPORTED 12 / ASSUMPTION_SENSITIVE 31 / CONTRADICTED 9 / NOT_EVALUATED 8 of 60`;
  - counterexample ids;
  - finding counts by kind, with sample ids;
  - the §15 cap check.
- **No single claim-level standing** is derived from a distribution. "37/60" is shown as "37 of 60 samples", never as "supported".

## 22. Finding taxonomy

**Standings (7).** They reuse the `Verdict` semantics where they coincide.

| Standing | Meaning |
|---|---|
| SUPPORTED | Verdict SUPPORTED under the requirement policy; no disagreement; no BLOCKING finding |
| CONTRADICTED | Verdict CONTRADICTED; no disagreement |
| MIXED | Decisive results disagree, and some disagreement is unexplained by any axis |
| ASSUMPTION_SENSITIVE | Decisive results (recorded or re-evaluated) disagree, and every disagreement is explained by assumption axes |
| INCONCLUSIVE | Verdict INCONCLUSIVE (inconclusive / errored results, or a required protocol missing); no BLOCKING finding |
| UNSUPPORTED | Evidence exists but, under the plan, cannot establish the claim as stated (O1–O6, cap exceeded) |
| NOT_EVALUATED | No matched or related in-scope evidence |

**Precedence:**
1. disagreement → ASSUMPTION_SENSITIVE / MIXED;
2. CONTRADICTED;
3. any BLOCKING finding → UNSUPPORTED;
4. SUPPORTED / INCONCLUSIVE per the verdict;
5. UNTESTED with no related evidence → NOT_EVALUATED.

**Finding kinds (12):** CONTRADICTION, ASSUMPTION_SENSITIVE, PROTOCOL_DISAGREEMENT, MISSING_EVIDENCE, MISSING_CONTROL, EVIDENCE_TYPE_MISMATCH, SCOPE_MISMATCH, PROVENANCE_MISMATCH, COUNTEREXAMPLE_FOUND, INCONCLUSIVE_EVIDENCE, INTEGRITY_FAILURE, NOT_EVALUATED.

- Each finding has: kind, a `code` from a closed list documented in `docs/audit/`, severity, subject (plan claim / concept / report), axis, detail, record ids, and sample ids.
- **INCONCLUSIVE_EVIDENCE** is raised whenever a matched result is INCONCLUSIVE, ERRORED or NOT_APPLICABLE. It is QUALIFYING; inconclusive evidence is never ignored.

## 23. Severity semantics

Severities are categorical labels, never summed, averaged or compared across claims:
- **BLOCKING:** the claim as stated is not established by the supplied evidence under the plan.
- **QUALIFYING:** the claim, if stated, must carry this qualification.
- **INFORMATIONAL:** context (coverage notes, validated generated labels).

## 24. Refusal behaviour

The audit **raises** on:
- a malformed plan (unknown requirement name, policy not registry-valid, duplicate claim names);
- `model` given and not equal to the plan's checkpoint;
- evidence that is not a trace, a path or a known result object;
- an unreadable trace path.

Everything about the **evidence content** (forgery, mismatch, scope) is **flagged, not raised**, so one bad record cannot hide the rest. Flagged records are excluded from every standing.

## 25. Serialization

- **`AuditPlan`** is a registered record kind (`audit_plan` v1): content-derived id, canonical JSON, `to_json` / `from_json` via the schema codec.
- **`AuditReport.to_dict()`** is deterministic. It carries `format: "beyondnn.audit_report"`, `format_version: 1`, the plan (full), the plan id, the evidence digest and the BeyondNN version.

## 26. Persistence and reload

- **Round trip:** traces saved with `TraceResult.save` → new process → `load_trace` → `audit` → a report whose `to_dict()` equals the in-memory audit's exactly.
- **P6-4:** fixed *for auditing* by trace-level concept verification (it stays open for `compose`, which still needs live objects).
- **Reports:** `report.save(path)` (JSON, atomic, never overwrites) and `load_report(path)` (a dict). `verify_report(doc, evidence, plan)` re-runs the audit and compares canonical JSON. On disagreement it raises `AuditMismatchError` listing the differing paths; it never corrects.

## 27. WHY integration

- **Composing:** `bnn.compose(..., audit=report)` accepts one `AuditReport`. It refuses if:
  - the report's checkpoint differs from the reference model's digest;
  - the reference sample is not in `plan.samples` (when any per-sample claim exists).
- **The view:** `why.audit` gives an `AuditView` with:
  - the per-sample standings and findings of the plan claims for the reference sample;
  - finite-sample and population claim standings as dataset-scoped context;
  - concept audits as context.
- **Rendering:** `render()` gains an AUDIT section. The coverage line names what the audit did not evaluate.

## 28. API

```python
import beyondnn as bnn
A = bnn.audits
plan = A.plan(
    name="...", checkpoint=A.checkpoint_of(model), declared_model=None,
    samples=[...sample ids...], datasets=[...concept dataset ids...],
    requirements=[A.requirement("necessity_v1", policy=COMPREHENSIVENESS_POLICY, controls=True,
                                alternatives=[A.alternative("comprehensiveness", {...})])],
    claims=[A.claim("ig_necessary", relation="necessary_for", target=metric,
                    selection=A.selection(site, method="integrated_gradients"),
                    scope="instance", requirement="necessity_v1",
                    invariant_over=[A.invariance("replacement", min_values=3), ...])],
    concepts=[A.concept(concept_id, asserted="validated_concept", policy=POLICY_V1)],
    counterexamples=A.counterexample_rule(max_counterexample_fraction=None, ...),
    naive_auroc=None,
)
report = bnn.audit(evidence, plan=plan)
```

Every plan field is explicit; constructors have no hidden defaults except `None`, meaning "not declared", which the report states.

## 29. Realistic experiments (pre-registered)

### 29.1 Central experiment

- **Models (reused, not retrained):**
  - A = MLP / breast_cancer (`model_A.pt`), sites `input` and `net.1`;
  - B = CNN / digits (`model_B.pt`), sites `pixels` and `relu2`;
  - C = BERT-tiny / SST-2 (pinned revision), site `tokens` (word embeddings).
- **Samples:**
  - exactly the Phase-5.5 held-out selections (60 / 60 / 40, `SELECTION_SEED` 1234);
  - no re-selection;
  - every sample is reported.
- **Evidence per sample and site.** Approaches: gradient (G), integrated gradients (IG; Captum for input sites as in 5.5), and a seeded random declared selection (R; seed 30000 + index).
  - Replacements r1–r3 exactly as in 5.5.
  - k at p ∈ {5%, 10%, 20%}. For each method and k, the tests are:
    - comprehensiveness with count-matched controls (N=50, seed 10000 + index) under r1, r2, r3;
    - sufficiency (count) under r1, r2, r3;
    - comprehensiveness with magnitude-matched controls under r1.
  - Thresholds as 5.5: `min_drop` = `max_drop` = 0.5 · margin.
  - Alternative thresholds declared in the plan: 0.25 · margin and 0.75 · margin.
  - Traces are saved and audited after reload.
- **Plan per model and site.** For each method M ∈ {G, IG, R}:
  - `M_necessary`: NECESSARY_FOR, selection (site, M), per-sample, requirement `necessity_v1` (comprehensiveness required, controls required), `invariant_over` replacement (3 values), k (3 values), null (2 values);
  - `M_sufficient`: SUFFICIENT_FOR, requirement `sufficiency_v1` (sufficiency required, controls required), `invariant_over` replacement (3) and k (3).
- **Attribution-only audit.** The same plan claims audited over the attribution traces alone (α).
- **Naive single-configuration reference** (what a typical report states): the fraction of samples whose IG comprehensiveness result SUPPORTS under r1, k at p=10%, count controls, threshold 0.5 · margin. It is computed from the same records.
- **Hypotheses** (evaluated per model/site; 5 model/sites):
  - **CH1 [prediction]:** for IG_necessary, the number of samples with per-sample standing SUPPORTED is ≤ ½ of the naive single-configuration SUPPORTS count in ≥ 3 of 5 model/sites. Falsified otherwise.
  - **CH2 [prediction]:** R_necessary is SUPPORTED on ≤ 5% of samples at every model/site.
  - **CH3 [prediction]:** the IG_necessary count of samples in {SUPPORTED, ASSUMPTION_SENSITIVE} exceeds R_necessary's at every model/site.
  - **CH4 [structural]:** in the attribution-only audit, every per-sample standing of every `*_necessary` / `*_sufficient` claim of G and IG is UNSUPPORTED with EVIDENCE_TYPE_MISMATCH. R has no attribution evidence, so it is NOT_EVALUATED.
  - **CH5 [prediction]:** ASSUMPTION_SENSITIVE is the most frequent IG_necessary standing in ≥ 3 of 5 model/sites.
  - **CH6 [engineering]:**
    - 0 integrity failures on untampered evidence;
    - reloaded audit equal to the in-memory audit;
    - `verify_report` passes.
- **Reported regardless of outcome:**
  - the full distributions for every claim;
  - finding counts by kind and axis;
  - sensitivity counts per axis (replacement / threshold / k / null);
  - the protocol-disagreement count (necessary vs sufficient).

### 29.2 Concept audit experiment (realistic)

- **Re-run** the Phase-6 realistic concept tests for A (K1–K4), B (K5, K6) and C (K7–K9), code unchanged, saving traces:
  - encoding with covariance and with isotropic controls;
  - use with zero and train-mean references;
  - validations under POLICY_V1.
- **Plan:** each (concept, feature kind) asserted VALIDATED_CONCEPT under POLICY_V1. ENCODES claims `invariant_over` null (2); use claims `invariant_over` replacement (2). Two counterexample rules are declared up front and both reported: strict (FP and FN ≤ 0.10) and lenient (≤ 0.30). `naive_auroc` = 0.6.
- **Hypotheses:**
  - KH1 [consistency]: only K4-direction has any VALIDATED validation;
  - KH2 [consistency]: under the strict rule, K4-direction is UNSUPPORTED (`counterexample_heavy`, FP 0.28);
  - KH3 [consistency]: the K1-direction use claim is ASSUMPTION_SENSITIVE (replacement);
  - KH4 [consistency]: ≥ 1 ENCODES claim is ASSUMPTION_SENSITIVE (null) (K1-direction, K5, K7);
  - KH5 [prediction]: no concept is SUPPORTED under the strict rule.

### 29.3 Falsification scenarios A–N

- **Status:** pre-registered expected audit output.
- **How they are built:** from real runs on BeyondNN's `_testing` models, whose ground truth is known by construction. They are *tests of the audit*, not findings about models; the central experiment (§29.1) uses no toy models.
- **Evidence:** each scenario's evidence is produced by BeyondNN runs; nothing is hand-written.

| | Scenario | Expected standing | Expected finding(s) |
|---|---|---|---|
| A | necessary unit, intervention + comprehensiveness with controls, 3 replacements, all SUPPORTS | SUPPORTED | none BLOCKING |
| B | ATTRIBUTED_TO supported only; causal claim declared | UNSUPPORTED | EVIDENCE_TYPE_MISMATCH `attribution_is_not_intervention` |
| C | attribution high, intervention on a redundant path shows no effect | CONTRADICTED | PROTOCOL_DISAGREEMENT `attribution_intervention_disagree` |
| D | selection necessary but not sufficient (comp SUPPORTS, suff CONTRADICTS) | per claim: SUPPORTED / CONTRADICTED | PROTOCOL_DISAGREEMENT `necessary_not_sufficient` on both |
| E | effect reverses between replacements | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE axis=replacement |
| F | uncontrolled comprehensiveness SUPPORTS, the same with controls CONTRADICTS | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE axis=null |
| G | feature decodable, not used | concept UNSUPPORTED; use claim CONTRADICTED | `decodable_not_used` |
| H | generated label, no validation, asserted validated | UNSUPPORTED | EVIDENCE_TYPE_MISMATCH `generated_label_is_not_validation` |
| I | concept validated but FP rate above the declared cap | UNSUPPORTED | COUNTEREXAMPLE_FOUND `counterexample_heavy` BLOCKING, ids listed |
| J | all evidence from another checkpoint | NOT_EVALUATED | PROVENANCE_MISMATCH BLOCKING |
| K | evidence about a sample not in the plan | NOT_EVALUATED for that sample | SCOPE_MISMATCH `sample_out_of_scope` |
| L | encoding SUPPORTS under one null and CONTRADICTS under the other | ASSUMPTION_SENSITIVE | ASSUMPTION_SENSITIVE axis=null |
| M | SUPPORTS without controls, requirement demands controls | UNSUPPORTED | MISSING_CONTROL |
| N | declared claim, no evidence at all | NOT_EVALUATED | NOT_EVALUATED (not CONTRADICTED) |

A tampered variant of A (a result's outcome edited in the saved JSON, with its id recomputed) must give INTEGRITY_FAILURE and exclusion.

## 30. NLP experiment (ACL case study)

- **Data:** model C (BERT-tiny / SST-2) from §29, plus the Phase-6 concept runs K7–K9 re-run with traces saved (§29.2).
- **Additional pre-registered cross-tabulations** of IG_necessary per-sample standings (descriptive only; no hypothesis):
  - by whether the IG top-k selection (p = 10%) includes a special token ([CLS] / [SEP]);
  - by sentence length (tertiles);
  - by the presence of a negation token (Phase-6 list);
  - by position of the top-1 token (first / last / interior).
- **Token-removal OOD:** replacement sensitivity (zero vs [MASK] vs [PAD] embedding) is the recorded proxy. The audit reports it; it does not measure OOD-ness.
- **Documented, not tested:**
  - padding: batch size 1, no padding occurs; padding effects are not evaluated;
  - label leakage: the checkpoint was fine-tuned by a third party on SST-2 train; validation-set use is standard, but training data cannot be verified;
  - lexical shortcuts: the K7 (positive-word) and K9 (negation) concept audits.
- **Concept hypotheses [consistency]** with Phase 6:
  - NH1: K8 (length) and K9 (negation) neurons are UNSUPPORTED with `decodable_not_used`;
  - NH2: K7 direction is ASSUMPTION_SENSITIVE (replacement), because its use test flips between zero and train-mean references.

## 31. Mutations

**Evidence and report mutations** (tests; each must be detected):
- drop a contradicting result;
- flip an assessment or result outcome (id recomputed);
- change the checkpoint;
- change the sample;
- remove controls;
- change the replacement;
- change the threshold;
- change the null;
- change the concept, feature or target;
- change the dataset scope;
- remove counterexamples;
- promote a GENERATED label;
- relabel INCONCLUSIVE as SUPPORTS;
- supply only the favourable k or replacement for an invariance claim;
- edit a saved report's standing or finding.

"Treat missing evidence as positive/negative" is a code mutation.

**Code mutations** (a scratch script mutates `beyondnn/audits` and requires a test failure):
- the precedence order;
- the axis-explanation rule;
- the invariance count;
- the cap comparison;
- the scope checks;
- the type-mismatch checks;
- missing → SUPPORTED;
- missing → CONTRADICTED;
- ignoring INCONCLUSIVE;
- dropping re-derivation;
- dropping the provenance check.

Survivors are fixed with tests or documented.

## 32. Performance

On the MLP, CNN and BERT central evidence bodies, one torch thread, measure:
- ingestion and integrity;
- re-derivation;
- aggregation (claims, concepts);
- serialization;
- reload (load traces + audit);
- peak memory (tracemalloc);
- trace directory and report sizes.

The first measurement sets the reference; there are no targets.

## 33. Publication evidence outputs

- `docs/research/PAPER_EVIDENCE_LEDGER.md`;
- `experiments/phase7/results/RESULT_MANIFEST.json`;
- `docs/research/PHASE_7_FIGURE_CANDIDATES.md` (only figures whose data exist);
- `docs/research/PHASE_7_REVIEWER_RISK.md`;
- `docs/roadmap/TRACE3B_FUTURE.md` (≈1 page).

The paper is not written.

## 34. Gate criteria

**READY FOR PHASE 7.5** requires all of:
- the full test matrix green;
- ruff, format and mypy --strict clean;
- build and clean-wheel smokes;
- README examples run;
- scenarios A–N and the tampered A give exactly the §29.3 expectations;
- every evidence mutation detected;
- every code-mutation survivor fixed or explained;
- CH6 holds;
- the central, NLP and concept experiments are run as registered and reported in full.

- **"WITH EXPLICIT LIMITATIONS"** if the engineering criteria hold but scientific limitations materially bound the claims (for example, CH1–CH5 mostly fail, or the separation is weak).
- **REQUIRES CHANGES** if any scenario, reload or mutation criterion fails at the end of the phase.

## 35. NO-GO criteria

**NO-GO** if any of:
- the audit needs a global score or auto-resolution to produce its output;
- a scenario's expected standing cannot be produced without special-casing it;
- the audit reports SUPPORTED where the evidence does not decide the claim (a correctness failure that cannot be fixed within the phase);
- the literature review is found to have missed a system implementing this record-level audit.

## Deviations

All of the following were recorded on 2026-09-26, during implementation and before any §29 experiment was run.

- **D1 (§22 precedence vs §29.3 J/K).** Exclusion findings (PROVENANCE_MISMATCH, SCOPE_MISMATCH `sample_out_of_scope` / `dataset_out_of_scope` / `other_sample_set`, INTEGRITY_FAILURE) are BLOCKING but do **not** set UNSUPPORTED. They remove evidence; the standing is then computed from what remains.
  - Why: this resolves a conflict inside the plan. Step 3 of the §22 precedence ("any BLOCKING → UNSUPPORTED") contradicted the pre-registered J/K expectations (NOT_EVALUATED) and the §22 definition of UNSUPPORTED ("O1–O6, cap exceeded").
  - The UNSUPPORTED-triggering codes are: `attribution_is_not_intervention`, `decodability_is_not_use`, `generated_label_is_not_validation`, `narrower_estimand`, `invariance_untested`, `missing_required_controls`, `counterexample_cap_exceeded`, `counterexample_heavy`, `decodable_not_used`.
- **D2 (§22).** A 13th finding kind, LIMITATION, was added for limitation-type notes that are not evidence problems: a validated generated label (`generated_label_unverified`, INFORMATIONAL), the polysemanticity indicator (`feature_encodes_several_concepts`), and SAE limitation codes.
- **D3 (§13).**
  - Alternative criteria are declared as `(protocol, key, factor | value)`, so "0.25 · margin / 0.75 · margin" is declared as factors 0.5 / 1.5 of the recorded `min_drop` / `max_drop`.
  - Re-evaluation is implemented for `intervention_threshold`, `attribution_threshold`, `comprehensiveness`, `sufficiency` and `concept_intervention`.
  - Alternatives for `concept_encoding` are **refused** (AuditPlanError), because they would need the activations re-derived. Declare a second recorded encoding test instead.
- **D4 (§29.3 A).** `intervention_threshold` cannot decide a unit subset (it requires `subject.units = None`). Scenario A therefore has two plan claims:
  - a unit-level comprehensiveness claim (3 replacements, controls);
  - a site-level `intervention_threshold` claim.

  Both are expected SUPPORTED.
- **D5 (§16).** Finding kinds for codes the plan left unspecified:
  - `decodable_not_used` is EVIDENCE_TYPE_MISMATCH (BLOCKING);
  - `controls_defeat_encoding` is ASSUMPTION_SENSITIVE on the `null` axis (QUALIFYING).
- **D6 (§11).** The `threshold` axis key excludes control-fraction criteria (`min_fraction_below`, `min_fraction_above`, `min_fraction_beyond_controls`); they are part of the `null` key. Without this, "controls vs none" would differ in two axes and could never be explained by the null axis alone (scenario F).
- **D7 (§29.3 G, scenario construction).** The first version of scenario G declared the use claim over the whole test split, while `use_test` evaluates the positive subset by default. The audit correctly refused that claim (`other_sample_set`). The scenario was corrected to declare the positive subset. This behaviour is kept as a test of the audit, not reported as a result.
- **D8 (§9a).** `required_protocol_missing` is raised only for required protocols with **no recorded result**. `derive_verdict`'s `required_but_missing` also lists protocols that ran and contradicted, which would have been reported as "missing". It is not raised when nothing matched at all; the standing already says so, and coverage lists "required protocols never run".
- **D9 (§29.1, before the central experiment was run).** Phase-5.5 tests *record* matched controls but decide by the absolute threshold only. With those tests, the plan's `null` invariance (count vs magnitude) could never change an outcome, and every SUPPORTS would fail "controls required" (§10 O6, where the implementation counts controls as present only when a control criterion enters the decision; D6).
  - **Control criterion added.** The central tests declare the Phase-6 control criterion: comprehensiveness `min_fraction_below = 0.95`, sufficiency `min_fraction_above = 0.95`.
  - **Uncontrolled test added.** One comprehensiveness test without controls under r1 per method and k (null value `none`), so that "absolute pass vs controls" (§17) can be observed on realistic evidence.
  - **Unchanged:** thresholds (0.5 · margin), replacements, seeds, samples, and methods G / IG / R.
  - **Caveat:** Phase-5.5 superiority medians are known (≈0.97–1.0 for IG, ≈0.5 for random), so the effect of the 0.95 criterion is not blind.
- **D10 (§26, §29.1 CH6).** Measured trace sizes are ≈0.5 MB per test trace on disk (0.9 MB for BERT) and ≈0.7 MB in memory. The full central evidence (≈20k test traces, ≈11 GB) does not fit the 6 GB of free disk. Therefore:
  - **Full audits** run in memory, one model/site at a time, and `verify_report` is checked on each.
  - **Save → fresh process → load → audit** is checked on a subset declared now: the first 3 samples (in selection order) of every model/site, with the same plan restricted to those samples.
  - **Artifacts** are git-ignored; they are deleted after the check if disk is needed, and that is reported.
- **D11 (§29.1).** The k invariance is declared with `min_values = 2`, not 3. p ∈ {5, 10, 20}% collapses to two distinct k on short inputs (BERT T ≤ 10: k = 1, 1, 2). Every declared p-level is still run on every sample, so all distinct k values are tested and enter the disagreement analysis. With min_values = 3, short sentences would be UNSUPPORTED only because of rounding.
- **D12 (implementation, before experiments).** Evidence ingestion no longer keeps a canonical JSON copy of every record (memory). Id/content conflicts are checked by serialising only ids that appear in more than one trace; the check itself is unchanged.
