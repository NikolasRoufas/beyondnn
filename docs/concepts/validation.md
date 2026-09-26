# Concept validation (ADR-042)

## The two claims

| | ENCODES (`concept_encoding` v1) | Use (`concept_intervention` v1) |
|---|---|---|
| Question | Is C decodable from F on held-out data, above the declared nulls? | Does intervening on F change the target Y beyond the same intervention on random features? |
| Relation | ENCODES | DECREASES / INCREASES (removal); SUFFICIENT_FOR (retention, site-relative) |
| Statistic | AUROC of F's fixed 1-D readout on the **test** split (declared sign) | FINITE_SAMPLE mean effect over a declared test subset (default: concept-positive) |
| Evidence | MEASURED activations (one clean pass per sample) | INTERVENTIONAL effects (one Phase-2 comparison family) |
| Mandatory | ≥ 1 control with `min_fraction_below` | an intervention and its reference, and ≥ 1 control with `min_fraction_beyond_controls` |
| Limitations | `DECODABILITY_NOT_USE`, `FEATURE_MAY_CARRY_OTHER_INFORMATION` | `SINGLE_FEATURE_TEST_MISSES_REDUNDANCY`, and `DIRECTION_INTERVENTION_MAY_ACTIVATE_DORMANT_PATHWAYS` for directions |

- **ENCODES never implies use.** Ground-truth case B (a copied but unused feature) gives ENCODES SUPPORTED and use CONTRADICTED, with an effect of exactly 0.
- **Per-sample targets are not needed.** A use claim is stated over a concept-conditioned subset with one fixed, declared target.

## Counterexamples

- The threshold is fixed on the **val** split (maximum balanced accuracy).
- **Every** held-out false positive and false negative is listed, with sample ids and values, and the rates are recorded.
- Counterexamples are never dropped: composition re-derives them.

## Policy `concept_validation_v1`

VALIDATED_CONCEPT requires all of:
- a SUPPORTED encoding assessment;
- at least one SUPPORTED use claim (DECREASES/INCREASES);
- a declared control criterion in every spec;
- a counterexample record.

- **Not requirements:** SUFFICIENT_FOR results may be attached as `additional`. They are shown but never change the status.
- **Rate caps:** v1 declares none; the rates are shown next to the status. *Consequence (ground truth H): a polysemantic feature with a 26% false-positive rate is VALIDATED under v1; its counterexamples are listed in full.* A policy may declare `max_false_positive_rate` / `max_false_negative_rate`.

## Scope

The validation's `scope` names:
- the dataset, split and size;
- the model digest;
- the site and axis;
- each claim's relation, target and intervention.

It is printed wherever the status is printed.

## In the structured WHY

`bnn.compose(trace, concepts=[validation, ...])` adds a **CONCEPTS** section with:
- the encoding and use outcomes on separate lines;
- the unmet requirements;
- the counterexample rates;
- the feature's value on this input (VALIDATED_CONCEPT or MEASURED);
- the limitations.

Composition refuses:
- a validation from another checkpoint;
- a forged or re-attached validation (full re-derivation);
- a concept activation about another input, or without its validation.

Concept validations are dataset-scoped *context*. They never override the instance evidence, and their claims are never merged with it.
