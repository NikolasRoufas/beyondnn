"""Phase-6 concepts: features, concept hypotheses, and controlled concept validation.

The steps are separate and visible (plan §18; ADR-039..042):

1. **feature definition**: :func:`neuron`, :func:`direction`, :func:`sae_feature`, or
   train-only discovery with :func:`fit_direction` / :func:`search_neurons`. A feature
   has no meaning (UNLABELED_FEATURE).
2. **concept extension**: :func:`dataset` (exact inputs, 0/1 labels, splits, scope).
3. **semantic proposal**: :func:`propose` (PROPOSED_CONCEPT), optionally from a
   :func:`generated_label` (GENERATED, never evidence).
4. **representation test**: :func:`encoding_test` (ENCODES; held-out AUROC of a fixed
   readout against mandatory controls; counterexamples).
5. **causal test**: :func:`use_test` (DECREASES/INCREASES with :func:`remove`, or
   SUFFICIENT_FOR with :func:`retain`; the intervention and its reference are
   required; mandatory random-feature controls).
6. **assessment**: :func:`validate` derives the semantic status under a declared
   policy (:data:`POLICY_V1`); VALIDATED_CONCEPT only if every requirement holds, and
   only within the recorded scope.
7. **presentation**: ``bnn.compose(trace, concepts=[...])`` re-derives everything and
   shows encoding and use separately.

Decodability is never use; a validated concept is never "the model understands C".
There is no concept score (ADR-007).
"""

from ._core import ConceptError, Control
from .data import Concept, ConceptData, GeneratedLabelResult, dataset, generated_label, propose
from .encoding import (
    ENCODING_POLICY,
    EncodingCriteria,
    EncodingResult,
    encoding_criteria,
    encoding_test,
    label_permutation,
    random_directions,
    random_neurons,
)
from .features import (
    Feature,
    direction,
    fit_direction,
    neuron,
    sae_feature,
    search_neurons,
)
from .use import (
    USE_POLICY,
    FeatureIntervention,
    Reference,
    UseCriteria,
    UseResult,
    reference,
    remove,
    retain,
    use_criteria,
    use_test,
    zero,
)
from .validate import (
    POLICY_V1,
    ConceptActivationResult,
    ConceptValidationResult,
    activation,
    load_validation,
    validate,
)

__all__ = [
    "ENCODING_POLICY",
    "POLICY_V1",
    "USE_POLICY",
    "Concept",
    "ConceptActivationResult",
    "ConceptData",
    "ConceptError",
    "ConceptValidationResult",
    "Control",
    "EncodingCriteria",
    "EncodingResult",
    "Feature",
    "FeatureIntervention",
    "GeneratedLabelResult",
    "Reference",
    "UseCriteria",
    "UseResult",
    "activation",
    "dataset",
    "direction",
    "encoding_criteria",
    "encoding_test",
    "fit_direction",
    "generated_label",
    "label_permutation",
    "load_validation",
    "neuron",
    "propose",
    "random_directions",
    "random_neurons",
    "reference",
    "remove",
    "retain",
    "sae_feature",
    "search_neurons",
    "use_criteria",
    "use_test",
    "validate",
    "zero",
]
