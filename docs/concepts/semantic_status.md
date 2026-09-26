# Semantic status (ADR-009 amended by ADR-039)

| Status | Held by | How it arises |
|---|---|---|
| UNLABELED_FEATURE | a `feature` record | always, for a feature without a proposal |
| PROPOSED_CONCEPT | every `concept` record | `propose(...)`, from a user, a dataset or a generated label |
| VALIDATED_CONCEPT | a `concept_validation` record, and only there | derived under the declared policy, when every requirement holds |

## Derived, never stored

- A `concept` record never stores a status.
- A `concept_validation` stores a status *and its unmet requirements*. Construction refuses both unless they are exactly what `derive_semantic_status(policy, summaries, rates)` computes.
- Composition then re-derives the summaries from the underlying assessments, results and raw evidence.

## No global REJECTED state

- A failed validation stays PROPOSED_CONCEPT, with the reasons listed (for example "use claim … contradicted") and the CONTRADICTED assessments kept.
- Rejection, like validation, is relative to a scope. Phase 5.5 showed that the intervention and the replacement change conclusions.

## Semantic status vs evidence status

These are different vocabularies:
- Evidence status says how a record was obtained (MEASURED, INTERVENTIONAL, GENERATED, …).
- Semantic status says where a hypothesis stands.

They meet in exactly two places:
- **A `generated_label`** has evidence status **GENERATED**. It can never be cited as evidence, and it can only seed a proposal.
- **A `concept_activation`** (the value of a validated concept's feature on one input) has evidence status **VALIDATED_CONCEPT**. It exists only for a VALIDATED validation on the same checkpoint (ADR-009). Otherwise the WHY shows a MEASURED feature activation of a PROPOSED concept.

**VALIDATED_CONCEPT is always scoped** to the model checkpoint, site, concept dataset and split, intervention and target. It never means universal validity.
