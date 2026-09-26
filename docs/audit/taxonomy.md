# Audit taxonomy

## Standings

A standing is derived by the precedence in `docs/PHASE_7_PLAN.md` §22 (with deviation D1). It describes what the in-scope, re-derived evidence establishes *under the declared plan*. It is not a probability and not a quality grade.

| Standing | When |
|---|---|
| SUPPORTED | The verdict is SUPPORTED under the requirement policy, there is no disagreement, and no UNSUPPORTED-triggering finding |
| CONTRADICTED | The verdict is CONTRADICTED and there is no disagreement |
| ASSUMPTION_SENSITIVE | Decisive results (recorded, or re-evaluated under declared alternative criteria) disagree, and every disagreement is explained by assumption axes |
| MIXED | Decisive results disagree, and at least one pair differs in no recorded assumption (an unexplained CONTRADICTION) |
| INCONCLUSIVE | The verdict is INCONCLUSIVE (inconclusive / errored / not-applicable results, or a required protocol never ran) |
| UNSUPPORTED | Evidence exists but cannot establish the claim *as stated*: a finding with an UNSUPPORTED-triggering code |
| NOT_EVALUATED | No matched or related in-scope evidence |

**Precedence:**
1. disagreement (ASSUMPTION_SENSITIVE / MIXED);
2. CONTRADICTED;
3. an UNSUPPORTED-triggering code;
4. SUPPORTED / INCONCLUSIVE;
5. NOT_EVALUATED.

**Excluded evidence** (provenance, scope, integrity) is reported with BLOCKING or QUALIFYING severity. It is removed before classification and never sets a standing by itself.

**Per-sample (INSTANCE-scope) claims** have no claim-level standing. Their result is the distribution of per-sample standings plus the counterexample sample ids.

**Concept standings:**
- SUPPORTED: every in-scope validation is VALIDATED_CONCEPT and no cap is exceeded;
- ASSUMPTION_SENSITIVE / MIXED: tests or validations disagree;
- UNSUPPORTED: validations exist and none is validated, or a cap is exceeded, or a generated label is asserted without a validation;
- NOT_EVALUATED: no validation.

There is no "rejected" (ADR-039).

## Assumption axes

Keys are computed from records only.

| axis | intervention_threshold | attribution_threshold | comprehensiveness / sufficiency | concept_encoding | concept_intervention |
|---|---|---|---|---|---|
| protocol | ✓ | ✓ | ✓ | ✓ | ✓ |
| threshold | criteria (excl. control fractions) | criteria | criteria (excl. control fractions) | criteria (excl. control fraction) | criteria (excl. control fraction) |
| replacement | operation + value digest (+ patch source) | – | declared replacement | – | mode + reference |
| k | – | – | selection k | – | – |
| null | `none` | `none` | control strategy @ control criterion, or `none` | control kinds / distributions @ criterion | control kinds @ criterion |
| method | – | method + baseline | selection method | – | – |
| dataset | – | – | – | concept dataset | concept dataset |

A SUPPORTS/CONTRADICTS pair that differs in exactly one axis is *explained* by it (finding ASSUMPTION_SENSITIVE, axis = that axis). If only multi-axis pairs exist, one `combined:a+b` finding is reported.

## Finding kinds and codes

| Kind | Codes | Default severity |
|---|---|---|
| CONTRADICTION | `unexplained_contradiction` | BLOCKING |
| ASSUMPTION_SENSITIVE | `assumption_sensitive`, `<axis>_sensitive` (concepts), `combined[_sensitive]`, `controls_defeat_encoding` | QUALIFYING |
| PROTOCOL_DISAGREEMENT | `attribution_intervention_disagree`, `necessary_not_sufficient`, `sufficient_not_necessary`, `relations_disagree` | QUALIFYING |
| MISSING_EVIDENCE | `required_protocol_missing` (QUALIFYING), `invariance_untested` (BLOCKING†), `selection_source_not_supplied`, `feature_derivation_not_rederived` (QUALIFYING) | see codes |
| MISSING_CONTROL | `missing_required_controls` | BLOCKING† |
| EVIDENCE_TYPE_MISMATCH | `attribution_is_not_intervention`†, `decodability_is_not_use`†, `generated_label_is_not_validation`†, `decodable_not_used`† | BLOCKING |
| SCOPE_MISMATCH | `narrower_estimand`† (BLOCKING), `sample_out_of_scope`, `dataset_out_of_scope`, `other_sample_set`, `policy_differs` | BLOCKING if nothing in scope remains, else QUALIFYING |
| PROVENANCE_MISMATCH | `other_checkpoint`, `other_model_declaration`, `mixed_checkpoints` | same rule |
| COUNTEREXAMPLE_FOUND | `counterexamples_present` (QUALIFYING), `counterexample_cap_exceeded`† / `counterexample_heavy`† (BLOCKING) | see codes |
| INCONCLUSIVE_EVIDENCE | `inconclusive_results` | QUALIFYING |
| INTEGRITY_FAILURE | `trace_integrity`, `id_content_conflict`, `rederivation_failed`, `selection_rederivation`, `selection_unresolved`, `provenance_unresolved`, `unregistered_protocol`, `feature_rederivation_failed` | BLOCKING |
| NOT_EVALUATED | `no_evidence`, `no_validation`, `concept_not_supplied` | BLOCKING |
| LIMITATION | `generated_label_unverified` (INFORMATIONAL), `feature_encodes_several_concepts`, `sae_*` (QUALIFYING) | see codes |

† UNSUPPORTED-triggering codes.

## Severities

Severities are categorical and never summed, averaged or compared across claims.
- **BLOCKING:** the claim as stated is not established by the supplied evidence under the plan.
- **QUALIFYING:** a stated claim must carry this qualification.
- **INFORMATIONAL:** context.
