# What BeyondNN means by "interpretable"

BeyondNN does not treat "interpretable" as a property a model simply has. It uses an **operational** definition:

> A model is interpretable **with respect to a claim language L and a test suite T** to the degree that claims about its computation, expressed in L, can be stated, tested by T, and survive.

This makes "how interpretable is this model?" meaningless on its own. "Which claims about this model survived which tests?" is always answerable. Everything below defines the vocabulary.

## Terms

**Observation (OBSERVED).** A value that crossed the model boundary unchanged: the input, the output, the logits. It is exact for this run.

**Measurement (MEASURED).** Directly observed internal model state, such as a module output captured by a hook. It is exact for this run, at this module boundary. It says nothing about what the value *means* or whether it *matters*.
- It stays MEASURED when read during an **intervened** execution. The intervention is execution context and is recorded in provenance (ADR-017).
- Measured state under intervention is not an intervention effect: see INTERVENTIONAL below.

**Attribution (ATTRIBUTED).** A score that assigns output sensitivity to inputs or internal units, computed by a specific method (gradient, IG, …) relative to a specific baseline. It is method-relative, not a fact about the model. Different methods legitimately disagree.

**Intervention.** A specified modification of an internal or input value during a forward pass. It always comes with: target (module path + selector), operation (zero / mean / constant / patch / scale), reference data (for mean or patch), and the inputs it was applied to.

**Interventional effect (INTERVENTIONAL).** The measured change in a metric caused by an intervention, on specified inputs: intervened behaviour compared against a baseline (e.g. baseline 0.91, intervened 0.34, effect −0.57). A raw tensor read while an intervention is active is MEASURED, not INTERVENTIONAL. An exact summary over exactly the measured inputs (finite sample) stays INTERVENTIONAL; a claim about a population is ESTIMATED_CAUSAL (ADR-013). *For that model, those inputs, that intervention, and that metric,* it is an exact causal quantity (a do-operation on a deterministic program). What it does **not** establish:
- that the component "represents" anything;
- that the effect generalises to other inputs;
- that the intervention was in-distribution (zero ablation often is not);
- that the component is the only route (redundancy and self-repair can hide necessity).

**Estimated causal effect (ESTIMATED_CAUSAL).** An approximation of an interventional quantity (attribution patching, linearised effects), or an aggregate over a sample of inputs with sampling uncertainty. It must carry its estimator and, where possible, its error.

> **Accepted (ADR-006):** the original `ESTIMATED_CAUSAL` is split into `INTERVENTIONAL` (exact, local) and `ESTIMATED_CAUSAL` (approximate or aggregated). Otherwise a single-input ablation and a gradient approximation of it would carry the same label, even though their reliability differs greatly.

**Feature.** A measurable scalar function of an activation, defined **relative to a basis**:
- a neuron (standard basis);
- a direction (probe, CAV, difference of means);
- a dictionary element (SAE / transcoder latent);
- a native bottleneck unit (Mode B).

"Feature 41 at layer 5" is undefined until the basis is named. A feature has no semantic meaning by default.

**Concept.** A *hypothesis* that a feature (or set of features) tracks a human-meaningful property. It has a lifecycle (`SemanticStatus`, ADR-009):
- `UNLABELED_FEATURE`: a feature exists, and there is no semantic claim.
- `PROPOSED_CONCEPT`: a label has been suggested by a human, an auto-interp model, or a heuristic. The label is data, not truth.
- `VALIDATED_CONCEPT`: the label passed a *declared* validation protocol, with results attached. The minimum protocol proposed for v1 has two parts:
  1. **Detection:** on held-out labelled positives *and counterexamples*, the feature separates them better than random directions or features, with a stated statistic.
  2. **Causal relevance:** intervening on the feature changes a behaviour that depends on the property, more than random-direction interventions of matched norm do.

  Passing detection alone gives `PROPOSED_CONCEPT` plus a detection result, never `VALIDATED_CONCEPT`, because decodability does not imply use.
- `REJECTED_CONCEPT`: failed validation. Keep this, since negative results matter.

  Validation is always relative to a dataset and protocol, so `VALIDATED_CONCEPT` means "validated under protocol P on dataset D". It never means "true".

**Generated (GENERATED).** Text or labels produced by a model (including an LLM describing a feature, or the model under study explaining itself). A generated item may *cite* records, but it is never evidence for them.

**Claim.** A proposition about the model's computation with a testable form, for example "ablating component S reduces metric M on input set X by at least δ". A claim has a *relation* (`NECESSARY_FOR`, `SUFFICIENT_FOR`, `INCREASES`, `DECREASES`, `ATTRIBUTED_TO`, `ENCODES`), and only some tests can bear on each relation. Only the first four are causal. A claim carries no status. Its standing is an **Assessment** derived from test results under a versioned policy (ADR-012). Explanations consist of claims plus the evidence for and against them.

**Faithfulness.** The degree to which an explanation's claims hold under the tests their form implies. Faithfulness belongs to claims and is measured by tests. It is not a property of a heat-map.

**Explanation confidence.** Deliberately **not** defined as a scalar in v0 (ADR-007). BeyondNN exposes *component* scores (effect size vs random baseline, stability across perturbations, agreement across methods, reproducibility across seeds, concept validation status). A single aggregate number is a research question (see `../research/RESEARCH_QUESTIONS.md`, RQ 8). It will be introduced only if it is shown to predict something checkable, such as held-out intervention outcomes on ground-truth models.

## Non-claims (what BeyondNN never asserts)

- That a trace is the complete computation. Coverage is bounded by module boundaries and the chosen basis.
- That attention weights are explanations.
- That a model's self-explanation (chain of thought) reflects its computation.
- That a label is correct because it is plausible.
- That a correlation (probe, attribution, co-activation) is causal.
