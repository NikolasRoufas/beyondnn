"""Phase 6: concepts that are decodable are not necessarily used.

Run: python examples/phase6_concepts.py

A hand-built model (``ConceptToy``) copies x0 and x1 into its hidden layer; the output
uses h0 = x0 but ignores h1 = x1. Both concepts are perfectly *decodable*; only one is
*used*. BeyondNN keeps the two claims separate, requires controls and an explicit
intervention, and derives the semantic status under a declared policy. It never
prints "the model understands C".

Covered: a neuron concept, a direction concept, a decodable-but-unused concept, a
validated causal concept, a random direction that fails its controls, a generated
label that stays proposed, and concept evidence in the structured WHY.
"""

from __future__ import annotations

import torch

import beyondnn as bnn
from beyondnn._testing.concept_models import ConceptToy, concept_inputs

C, iv = bnn.concepts, bnn.interventions


def main() -> None:
    model = ConceptToy().eval()
    x = concept_inputs(300, seed=0)
    samples = [x[i : i + 1] for i in range(300)]
    splits = ["train"] * 150 + ["val"] * 50 + ["test"] * 100

    def concept_data(col: int) -> C.ConceptData:
        labels = (x[:, col] > 0).long().tolist()
        return C.dataset(samples, labels, splits, name=f"toy x{col}", label_source=f"x{col} > 0")

    enc_controls = [
        C.random_directions(100, seed=1, distribution="isotropic"),
        C.label_permutation(100, seed=2),
    ]
    enc_criteria = C.encoding_criteria(min_fraction_below=0.95)
    use_criteria = C.use_criteria(min_change=0.25, min_fraction_beyond_controls=0.9)
    target = iv.metrics.select([0, 0])  # output y0 = 3 * h0

    def validate(concept: C.Concept, data: C.ConceptData) -> C.ConceptValidationResult:
        encoding = C.encoding_test(
            model, concept, data, controls=enc_controls, criteria=enc_criteria
        )
        feature = concept.feature.record
        controls = (
            [C.random_neurons(20, seed=3)]
            if feature.basis.value == "neuron"
            else [C.random_directions(20, seed=3, distribution="isotropic")]
        )
        use = C.use_test(
            model,
            concept,
            data,
            target=target,
            relation="decreases",
            intervention=C.remove(C.zero()),  # explicit: zero is declared, never a default
            controls=controls,
            criteria=use_criteria,
        )
        return C.validate(concept, encoding=encoding, use=[use])

    # 1. a neuron concept that is encoded AND used -> VALIDATED_CONCEPT (scoped)
    used = validate(
        C.propose(C.neuron("hidden", 0), label="x0 is positive", definition="x0 > 0"),
        concept_data(0),
    )
    # 2. a direction concept that is decodable but UNUSED -> stays PROPOSED_CONCEPT
    unit = torch.zeros(12)
    unit[1] = 1.0
    unused = validate(
        C.propose(C.direction("hidden", unit), label="x1 is positive", definition="x1 > 0"),
        concept_data(1),
    )
    for v in (used, unused):
        print(v.describe(), "\n")

    # 3. a random direction proposed as a concept fails its controls
    random = C.propose(
        C.direction("hidden", torch.randn(12, generator=torch.Generator().manual_seed(7))),
        label="x0 is positive",
        definition="random direction",
    )
    enc = C.encoding_test(
        model, random, concept_data(0), controls=enc_controls, criteria=enc_criteria
    )
    print(
        f"random direction: AUROC {enc.auroc:.3f} -> "
        f"ENCODES {enc.outcome.value.upper()} (controls decide)\n"
    )

    # 4. a generated label is GENERATED and at most a proposal
    gen = C.generated_label(
        model, C.neuron("hidden", 1), "x0 is positive", generator="example-labeler", revision="v1"
    )
    proposed = C.propose(C.neuron("hidden", 1), definition="x0 > 0", generated=gen)
    print(
        f"generated label {gen.record.text!r}: evidence status {gen.record.status.value}, "
        f"semantic status {proposed.semantic_status.value}"
    )
    wrong = C.encoding_test(
        model, proposed, concept_data(0), controls=enc_controls, criteria=enc_criteria
    )
    print(f"  tested: ENCODES {wrong.outcome.value.upper()} (AUROC {wrong.auroc:.3f})\n")

    # 5. concept evidence in the structured WHY (dataset-scoped context)
    point = torch.tensor([[1.5, 2.0] + [0.0] * 10])
    activation = C.activation(model, point, validation=used)
    reference = bnn.trace(model, point, sites=["hidden"], retention="cpu")
    print(bnn.compose(reference, concepts=[used, unused, activation]).render())


if __name__ == "__main__":
    main()
