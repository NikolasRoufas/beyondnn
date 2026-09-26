# Concepts (Phase 6)

Phase 6 makes semantic claims about neural representations explicit, provenance-aware, testable, controlled and causally scoped. It does not make it easy to attach names to neurons.

**The central rule:** *a concept being decodable from an activation does not establish that the model uses it.*

## The seven steps

| Step | API | Produces | Status |
|---|---|---|---|
| 1. Feature definition | `concepts.neuron`, `direction`, `sae_feature`; discovery on the train split only: `fit_direction`, `search_neurons` | `feature` record | UNLABELED_FEATURE |
| 2. Concept extension | `concepts.dataset(samples, labels, splits, name=, label_source=)` | `concept_dataset` record | — |
| 3. Semantic proposal | `concepts.propose(feature, label=, definition=)`, or from a `generated_label` | `concept` record (and `generated_label`, GENERATED) | PROPOSED_CONCEPT |
| 4. Representation test | `concepts.encoding_test(...)` | ENCODES claim, test result, counterexamples, assessment | — |
| 5. Causal test | `concepts.use_test(..., intervention=remove(ref) / retain(ref))` | DECREASES / INCREASES / SUFFICIENT_FOR claim, INTERVENTIONAL effects, assessment | — |
| 6. Assessment | `concepts.validate(concept, encoding=, use=[...], policy=POLICY_V1)` | `concept_validation` record | derived: VALIDATED_CONCEPT or PROPOSED_CONCEPT |
| 7. Presentation | `bnn.compose(trace, concepts=[validation, activation])` | the CONCEPTS section of the structured WHY | — |

Nothing is promoted automatically, and there is no concept score (ADR-007).

## Pages

- [features.md](features.md): neurons, directions, SAE latents; activation rules; discovery; direction interventions.
- [semantic_status.md](semantic_status.md): the lifecycle, and how semantic status relates to evidence status.
- [validation.md](validation.md): the two claim protocols, the policy, counterexamples, scope, and the WHY.
- [controls.md](controls.md): the mandatory controls and their nulls.

## What a result may and may not say

**Allowed:**
- "On the declared held-out split of D, feature F supported decoding C above covariance-matched random directions and label permutations."
- "Removing the projection onto v (reference R) decreased Y on the concept-positive test subset beyond matched random directions."
- "These results support the declared concept-use claim under this intervention, within the stated scope."

**Not allowed** (no API produces these):
- "The neuron understands C."
- "The model thinks about C."
- "F is C."
- "The model reasons using C."
