# Phase 6 Plan: Concepts and Concept Validation (pre-registration)

- **Status:** written and committed **before** any Phase-6 implementation.
  - Hypotheses, expected ground-truth outcomes and analysis rules must not change after results are seen.
  - Corrections are appended to §26 with the original text kept.
- **Baseline** (verified 2026-09-26):
  - HEAD `d3c22b0`, clean tree.
  - Python 3.10 / 3.12 / 3.14: 905 passed plus 1 reported skip without Captum; 926 passed with Captum 0.9.0.
  - ruff, format and mypy `--strict` clean.
- **Literature:** `docs/research/PHASE_6_LITERATURE.md`.
- **Central rule:** *decodability does not establish use.*

## 1. Problem definition

BeyondNN can record measurements, attributions, interventions, faithfulness tests and claims. It cannot yet express a *semantic* hypothesis about a representation ("this direction carries sentiment") in a form that is:
- explicit;
- bound to a model, a site and a data scope;
- tested separately for representation and for use;
- controlled against declared nulls;
- re-derivable from its records;
- impossible to promote without evidence.

Phase 6 adds that, on top of the existing Measurement → Claim → Test → Evidence → Assessment → Presentation pipeline. It is not a second evidence system.

## 2. Terminology

| Term | Meaning | Record |
|---|---|---|
| **Feature** | A structural or numerical coordinate of one site's activation: a neuron (one index along a declared feature axis), a direction (a vector along that axis), or an SAE latent (encoder rule plus decoder direction). It has no meaning. | `feature` (new; not evidence) |
| **Feature activation** | A feature's scalar value on one input, computed from a MEASURED site activation under the feature's activation rule and declared pooling. | computed; MEASURED parents |
| **Concept hypothesis / proposed concept** | A label and an operational definition attached to a feature, with its label source (user / dataset / GENERATED). | `concept` (new; not evidence) |
| **Generated label** | A label produced by a model or LLM. Always GENERATED; never evidence. | `generated_label` (new; status GENERATED) |
| **Concept dataset** | The operational definition of a concept *extension*: exact sample identities, binary labels, split assignment (train / val / test), label provenance, and a scope name. | `concept_dataset` (new; not evidence) |
| **ENCODES claim** | "Feature F encodes concept C": under the declared concept dataset, held-out split, readout and controls, C is decodable from F above the declared nulls. | `Claim` (relation ENCODES, existing) |
| **Causal-use claim** | "Intervening on F by intervention R DECREASES (or INCREASES) target Y, on the declared evaluation subset, beyond matched random-feature controls." | `Claim` (relation DECREASES / INCREASES, existing) |
| **Concept validation** | The derived semantic standing of one concept under a declared policy and scope, computed from assessments. | `concept_validation` (new; not evidence) |
| **Validated concept (semantic status)** | The status of a concept whose validation passed its policy, *within that validation's scope*. | derived |
| **Concept activation** | A feature activation on one input, tagged `VALIDATED_CONCEPT` only when its concept has a VALIDATED validation for the same model and site. | `concept_activation` (new; VALIDATED_CONCEPT or MEASURED) |

## 3. Scientific claims Phase 6 will support

1. On declared held-out data, feature F's activation separates concept-C examples from non-C examples (AUROC) above:
   - matched random features (random directions or neurons at the same site);
   - label permutations.
2. Under a declared intervention on F (neuron replacement or direction projection removal/retention, with a declared reference), a declared target Y changes (finite-sample mean over a declared evaluation subset) beyond the same intervention on matched random features.
3. The *combination* of (1), (2), recorded counterexamples and a declared policy yields a semantic status of VALIDATED_CONCEPT, scoped to:
   - the model checkpoint;
   - the site;
   - the concept dataset and split;
   - the intervention;
   - the target.

## 4. Claims Phase 6 explicitly will NOT support

- "The model uses / understands / thinks about / reasons with C."
- "F *is* C", "C is localised in F", or "F is monosemantic".
- Anything beyond the concept dataset's scope: no generalisation to other data, tasks, or checkpoints.
- Causal use from probe accuracy, TCAV-style gradients, attribution, or feature-description plausibility.
- Semantic validity of an SAE latent from sparsity or top-activating examples.
- A concept score, confidence or interpretability number (ADR-007).
- REJECTED as a global state: a failed validation is a scoped negative result (§6).

## 5. Concept representation

- **`FeatureRecord`**:
  - `basis ∈ {neuron, direction, sae}`;
  - `site` (module OUTPUT leaf), `call_index`;
  - `axis`: the feature axis of the leaf (non-batch);
  - `pooling ∈ {none, mean}` over the remaining non-batch axes. `none` requires them to be of size 1;
  - neuron: `index`;
  - direction / SAE: a retained `direction` tensor. The content digest is part of the identity, so any change of vector or norm is a different feature. The activation uses the normalised direction v̂ = v/‖v‖;
  - SAE: `SAEIdentity` with the checkpoint digest, encoder/decoder/bias digests, latent index, activation rule (`relu`), the pre-encoder bias convention, and optional reconstruction summary statistics. The activation is the SAE encoder activation of that latent;
  - `source`: declared, `fit:mean_difference`, `sae`, or `search`, with parameters (seed, split digest, candidate count, selection criterion) and the id of the fitting trace when fitted;
  - `model_state_digest`: the checkpoint the feature was derived on. Required for fitted and SAE features.
- **Activation rule:**
  - neuron: the value at `index` on `axis`;
  - direction: ⟨x, v̂⟩ along `axis`;
  - SAE: relu(⟨x − b_dec, W_enc[:, i]⟩ + b_enc[i]);
  - then pooled.
- **`ConceptRecord`:** `label`, `definition`, `feature` (a reference to the feature record), and `label_source ∈ {user, dataset, generated}`. A generated source references a `generated_label` record id.

## 6. Semantic-status lifecycle

- `SemanticStatus` has **three** values: `UNLABELED_FEATURE`, `PROPOSED_CONCEPT`, `VALIDATED_CONCEPT`.
  - A feature record without a concept is UNLABELED_FEATURE.
  - A concept record is always PROPOSED_CONCEPT. The record never stores a status.
  - VALIDATED_CONCEPT exists only as the derived status of a `concept_validation` record, and only when its policy passes.
- **REJECTED_CONCEPT (from ADR-009) is not implemented; a superseding ADR is required.**
  - Rejection, like validation, is scope-relative. A concept contradicted under one intervention or dataset may be supported under another (Phase 5.5 H4).
  - A global REJECTED state would overgeneralise exactly as a global VALIDATED would.
  - Negative results are *not lost*: they are CONTRADICTED assessments and a `concept_validation` with status PROPOSED_CONCEPT and its failures listed.
- **No automatic promotion.** The only path to VALIDATED_CONCEPT is `concepts.validate(...)` passing the policy, and composition re-derives it.

## 7. Concept evidence model

Everything reuses the existing pipeline.

- **Representation evidence:**
  - MEASURED `ActivationRecord`s of the site on every concept-dataset sample used (one multi-pass recording).
  - The ENCODES `ClaimTestResult` (protocol `concept_encoding` v1) cites them; its statistics hold the AUROC and control distributions.
- **Causal evidence:**
  - INTERVENTIONAL `CausalEffect`s from one `compare_family` (Phase 2): per-sample instance effects, plus a FINITE_SAMPLE mean effect for the primary intervention and for each control.
  - The DECREASES/INCREASES `ClaimTestResult` (protocol `concept_intervention` v1) cites the FINITE_SAMPLE effects.
- **Counterexamples:** a `ProtocolResult` (`concept_counterexamples` v1, diagnostic) lists every held-out false positive and false negative at a threshold fixed on the val split.
- **Assessments:** `Assessment.derive` with explicit concept policies.
- **Validation:** `concept_validation` references the concept, the feature, the dataset, the assessments, the counterexample result, the policy and the scope, and stores the derived status. Construction refuses a status that does not follow from the referenced verdicts. Composition re-derives everything from raw records.

## 8. Concept-discovery boundary

- **Discovery is outside validation.**
- **Discovery methods provided:**
  - `mean_difference` direction fitting on the **train** split (Zou 2023; Wu 2025 support it as a strong simple baseline);
  - a recorded `search` over candidate neurons or SAE latents on the **train** split, by a declared criterion (train AUROC). The candidate count and criterion are stored.
- **Not provided:** logistic probes, INLP, DAS, ACE-style clustering, automated labelling.
- **The val and test splits are never used for discovery.** Validation refuses a feature whose fit or search used test samples (the split digests are compared).

## 9. Probe policy

- The readout is **fixed and 1-dimensional**: the feature activation itself. No probe is trained at evaluation time.
- **Fitting:** a direction is fitted only on train (mean difference of the class means).
- **Metric:** AUROC on the held-out test split. The observed feature uses a *declared* sign (fitted directions: positive = concept).
- **Recorded:** readout family, fitting method, split sizes and digests, class balance, feature dimension, seeds, and control counts.
- **Rationale:** the Hewitt & Liang selectivity concern (probe flexibility) is removed by having no evaluation-time training. Small-sample variability is handled by the controls (§11).

## 10. Causal-validation policy

A concept is validated for use only with a causal-use claim:
- relation DECREASES or INCREASES on a declared target metric;
- over a declared evaluation subset of the concept dataset (by default the concept-positive **test** samples);
- under a declared intervention on the feature;
- against random-feature controls with a declared criterion.

| Relation | SUPPORTS iff |
|---|---|
| DECREASES | mean effect ≤ −`min_change` **and** the fraction of controls whose mean effect is larger (less decreasing) ≥ `min_fraction_beyond_controls` |
| INCREASES | symmetric |

- Ties count half.
- The claim's estimand is FINITE_SAMPLE over exactly the evaluation subset (aggregation `mean`).
- **Per-sample targets are not needed** (request §31): use claims are stated over concept-conditioned subsets with one fixed declared target. A concrete per-sample-target failure would be logged in §26 before any extension.

## 11. Control policy (mandatory)

**No ENCODES or causal-use spec can be built without at least one control with a declared criterion** (a `ValueError` at construction). Control statistics are never merely descriptive in concept protocols.

- **Encoding controls:**
  - (a) `random_directions(n, seed, distribution)`: n random directions at the same site and axis. `isotropic` means v ~ N(0, I); `covariance` means v ~ N(0, Σ_train), with Σ from the train-split activations and retained. Each is normalised, pooled identically, and scored **sign-free**: max(AUC, 1 − AUC). This is conservative, because random directions get their best sign.
  - (b) `random_neurons(n, seed)`: uniform other indices on the same axis (neuron features), also sign-free.
  - (c) `label_permutation(n, seed)`: the observed feature against permuted test labels, signed as declared.
  - Criterion: per control, the superiority (fraction of controls strictly below the observed statistic + ½ · fraction tied) must be ≥ the declared `min_fraction_below`. Optionally also `min_auroc`.
- **Causal controls:** the same intervention (same operation and reference) on n random features of the same kind (random directions with a declared distribution, or random neurons) on the same samples and target. Criterion: `min_fraction_beyond_controls`.
- **Seeds:** local generators only. The global RNG is never used.

## 12. Replacement policy

- **The intervention is required.** There is no default: `concepts.intervention(...)` has no zero default, and specs refuse a missing intervention.
- **Neuron features:** `replace_unit(reference=...)` with a reference of `zero()` or a tensor of the site's exact shape (e.g. the train mean or a resampled activation), using the existing CONSTANT/ZERO unit interventions (`unit_axes` for non-last axes).
- **Direction and SAE features (new DIRECTION operation, InterventionRecord v4):**
  - **removal** (`retain=False`): x′ = x − ⟨x − r, v̂⟩ v̂. The coordinate along v̂ is replaced by the reference's.
  - **retention** (`retain=True`): x′ = r + ⟨x − r, v̂⟩ v̂. Only the coordinate along v̂ is kept; everything else comes from the reference.
  - r is zero or a site-shaped reference tensor.
  - Addition and scaling (steering) are **not** implemented: they are off-distribution and conflate influence with use (literature §4).
- **Recorded:** the operation, reference kind and digest, direction digest and axis, and the scope (every position of the leaf).
- **Limitations attached:** `DIRECTION_INTERVENTION_MAY_ACTIVATE_DORMANT_PATHWAYS` (Makelov 2024), plus the existing OOD codes.
- **SAE latents:** they are intervened on along their normalised decoder direction. This is *projection* semantics, not SAE-native clamping or ablation, and is recorded as the limitation `SAE_INTERVENTION_IS_PROJECTION`.

## 13. Counterexample policy

- The threshold is fixed on the **val** split, as the midpoint that maximises balanced accuracy (ties to the lower threshold). It is never tuned on test.
- On the test split, list **every** false positive (non-C with activation ≥ threshold) and false negative (C below threshold), with sample ids and activations, sorted.
- The counts and rates are recorded.
- The validation policy **requires** a counterexample record. The counterexamples are always shown and never dropped.
- A policy may declare `max_false_positive_rate` / `max_false_negative_rate`. The default v1 policy declares none, and records the rates as scope-weakening information.

## 14. Random-direction policy

- Same site, same axis, same pooling, and unit norm (norm matching is implicit, because every direction is normalised before use).
- The distribution is declared: `isotropic` or `covariance`.
- Random directions are **not** matched on concept correlation or on sparsity; overmatching would make the null meaningless (literature §3).
- Recorded: seed, n, distribution, the covariance digest (retained), normalisation, and the matching rule.

## 15. Dataset-level validation

- A concept dataset is a finite, identified set: sample ids, labels and splits.
- Every concept statistic is FINITE_SAMPLE over an identified subset. There is no POPULATION estimate.
- The validation's scope string names the dataset (for example `"sst2-validation[seeded 50/20/30]"`) and is shown wherever the status is shown.

## 16. SAE adapter policy

- `concepts.sae_feature(site, encoder=..., decoder=..., b_enc=..., b_dec=..., index=..., checkpoint=..., reconstruction=...)` takes **tensors**, so there is no SAELens or other dependency.
- The adapter records the digests, the activation rule, the index and the reconstruction summary if given.
- **SAE limitations always attached:**
  - `SAE_FEATURE_SPLITTING`;
  - `SAE_FEATURE_ABSORPTION`;
  - `SAE_POLYSEMANTICITY_NOT_EXCLUDED`;
  - `SAE_RECONSTRUCTION_ERROR`;
  - `SAE_MISSING_FEATURES`;
  - `SAE_INTERVENTION_IS_PROJECTION`.

## 17. Generated-label policy

- `concepts.generated_label(feature, text, generator=..., revision=...)` creates a GENERATED record. It can seed `propose(..., label_source=generated)`, which gives PROPOSED_CONCEPT.
- A generated label cannot be cited as evidence (existing `EvidenceRef` rule).
- Validation of a concept with a generated label follows the same policy. The WHY shows "label: GENERATED" alongside any status.
- There is no LLM dependency.

## 18. User / API workflow (separate, visible steps)

```python
import beyondnn as bnn
C = bnn.concepts
data = C.dataset(samples, labels, splits, name="...", label_source="dataset")        # extension
feat = C.fit_direction(model, data, site="net.1", method="mean_difference")          # discovery (train only)
# or: C.neuron("net.1", 5), C.direction("net.1", v), C.sae_feature(...)
concept = C.propose(feat, label="large tumour", definition="mean radius > train median")  # PROPOSED
enc = C.test_encoding(model, concept, data, controls=[C.random_directions(200, seed=0, distribution="covariance"),
                                                      C.label_permutation(200, seed=1)],
                      criteria=C.encoding_criteria(min_fraction_below=0.95))        # ENCODES claim test
use = C.test_use(model, concept, data, target=m, relation="decreases",
                 intervention=C.remove_direction(reference=C.zero()),                  # explicit, no default
                 controls=[C.random_directions(50, seed=2, distribution="covariance")],
                 criteria=C.use_criteria(min_change=0.5, min_fraction_beyond_controls=0.95))
val = C.validate(concept, encoding=enc, use=[use], policy=C.POLICY_V1)                 # derived status
bnn.compose(bnn.trace(model, x, sites=["net.1"]), concepts=[val])                      # WHY
```

- Names may change during implementation if the architecture suggests better ones (logged in §26).
- The separation of the steps may not change.

## 19. Backwards compatibility

- **Schema changes** (each with a migration and no invented content):
  - **`InterventionRecord` v4:** adds `direction: TensorRef | None` and `direction_axis: int | None`. Migration v3→v4 sets both to `None`.
  - **`Subject` gains `feature: str | None`** (a feature record id), so `Claim` becomes v3. Migration v2→v3 sets `subject.feature = None`. This **changes the golden claim id again**; the change is documented.
  - **New kinds:** `feature`, `concept`, `generated_label`, `concept_dataset`, `concept_validation`, `concept_activation`.
- **Unchanged:** `Coverage.concepts_validated` keeps meaning "no validated concept composed" when no concept is composed. A new `concepts_evaluated` field is added.
- **Old traces:** Phase 1–5.5 traces load unchanged, or migrate. Nothing semantic is invented.

## 20. Expected failure cases (predicted before implementation)

1. Realistic "use" effects may be small relative to covariance-matched random directions, because random directions inside the data subspace are a strong null. Many realistic use claims may fail.
2. Encoding of dataset-derived concepts from hidden layers may be near-universal (information is preserved), so ENCODES alone will rarely discriminate.
3. Zero-reference projection removal may be off-distribution for sites with a large mean (BERT pooler, tanh). Mean-reference removal may give different conclusions.
4. Label-aligned concepts (sentiment for an SST-2 model) will validate trivially, and must be labelled as label-aligned rather than cited as evidence of framework power.
5. Composition may refuse realistic records for rounding reasons (Phase 5.5 precedent). Tolerances will be justified by ADR if needed.

## 21. Pre-registered synthetic ground-truth cases

**Model `ConceptToy`** (hand-built, `beyondnn/_testing/concept_models.py`):
- **Input:** x ∈ ℝ¹², drawn N(0, I) with a local seed, except where a case induces correlation.
- **Hidden module `hidden`**, 12 units:
  - h0 = x0; h1 = x1;
  - h2 = x2 + x3; h3 = x2 − x3;
  - h4 = h5 = x4 (duplicate);
  - h6 = x6;
  - h7 = x7 + x8;
  - h8 = relu(x9)·[x10 > 0]; h9 = relu(x9)·[x10 ≤ 0];
  - h10 = x10; h11 = x11.
- **Readout `readout`** (6 outputs):
  - y0 = 3·h0;
  - y1 = h2 + h3;
  - y2 = max(h4, h5);
  - y3 = h6;
  - y4 = h7;
  - y5 = h8 + h9.
  - h1 feeds no output.

**Data:** n = 600, split 300 train / 100 val / 200 test (seed 0).

**Defaults unless stated:**
- Encoding: controls `random_directions(200, isotropic, seed 1)` and `label_permutation(200, seed 2)`, both with `min_fraction_below = 0.95`.
- Use: relation DECREASES; evaluation subset = concept-positive test samples; intervention = zero-reference projection removal for directions or zero replacement for neurons; controls `random_directions(50, isotropic, seed 3)` or `random_neurons(50, seed 3)`; `min_change = 0.25`; `min_fraction_beyond_controls = 0.9`.

**Pre-registered outcomes** (E = ENCODES outcome, U = use outcome, S = resulting semantic status):

| Case | Concept (label) | Feature | Target | Expected |
|---|---|---|---|---|
| A: encoded and used | x0 > 0 | neuron h0; direction e0 | y0 | E SUPPORTS (AUROC 1.0); U SUPPORTS (mean drop = 3·E[x0 \| x0 > 0] ≈ 2.4); S VALIDATED |
| B: decodable, unused (mandatory) | x1 > 0 | neuron h1; direction e1 | y0 | E SUPPORTS (AUROC 1.0); U CONTRADICTS (effect exactly 0); S PROPOSED. The WHY shows "ENCODES: supported / use: contradicted" |
| C: distributed | x2 > 0 | neuron h2; direction u = (e2 + e3)/√2 | y1 | Neuron: E SUPPORTS (AUROC ≈ 0.83, above controls); U: zeroing h2 drops y1 by x2 + x3, mean ≈ 0.80, which is half the full effect. It SUPPORTS at `min_change` 0.25 but is visibly partial. Direction: ⟨h, u⟩ = √2·x2, so removal sets h2 and h3 to h2 − x2 and h3 − x2, and y1 drops by 2·x2 (mean ≈ 1.60): E SUPPORTS (AUROC 1.0), U SUPPORTS. *Prediction: the direction effect ≈ 2 × the neuron effect.* |
| D: redundant | x4 > 0 | neuron h4; direction (e4 + e5)/√2 | y2 | Neuron: E SUPPORTS; U CONTRADICTS (effect 0, because max(0, h5) = x4 when x4 > 0). Direction: U SUPPORTS (both copies removed). The single-neuron failure must *not* turn into "concept unused" in the WHY wording |
| E: correlated proxy | x5 > 0, with x6 = x5 + 0.3·ε in dataset E1 and x6 independent in dataset E2 | neuron h6 | y3 | E1: E SUPPORTS, U SUPPORTS → VALIDATED *scoped to E1*. E2: E CONTRADICTS. The scope strings differ, and the E1 validation never appears as the E2 status |
| F: random direction, small data | x0 > 0, test n = 16 | 50 seeded random directions proposed as the concept | — | The naive criterion (AUROC ≥ 0.7, no controls) passes for ≥ 10% of the directions. With the mandatory controls, ENCODES SUPPORTS for ≤ 5 of 50 (a random direction is itself a draw from the random-direction null, so ≈ α = 5% is expected; 10% allows for binomial noise) |
| G: label permutation | labels = a seeded permutation of (x0 > 0) | neuron h0 | y0 | E CONTRADICTS (AUROC ≈ 0.5 within controls); S PROPOSED |
| H: polysemantic | x7 > 0 | neuron h7 | y4 | E SUPPORTS (AUROC ≈ 0.83); the counterexample false-positive rate on test is ≥ 20% (driven by x8), listed in full |
| I: feature splitting | x9 > 0 | neuron h8; direction (e8 + e9)/√2 | y5 | Neuron h8: false-negative rate ≈ 50% of concept positives (the other half is carried by h9); U partial. Direction: AUROC 1.0, U SUPPORTS |
| J: generated but wrong | GENERATED label "x0 > 0" attached to neuron h1 | neuron h1 | y0 | The generated label record has status GENERATED; S PROPOSED at proposal; E CONTRADICTS (h1 is independent of x0); S stays PROPOSED. A correct generated label on h0 stays PROPOSED until `validate` passes |

**Refusals that must happen:**
- validation without controls;
- use test without an intervention;
- a feature fitted on test samples;
- evidence of concept X attached to concept Y;
- a validation composed with the wrong model, site, feature, dataset or target;
- a generated label promoted without validation.

## 22. Realistic validation cases

Phase-5.5 models, reused unchanged; experiment code in `experiments/phase6/`; checkpoints verified by state digest.
- **Concept datasets:** each concept is a binary label derived by a stated rule from the input or dataset annotation (no invented semantics).
- **Splits:** seeded (1234) 50% train / 20% val / 30% test over the stated pool.
- **Features:**
  - (i) the `mean_difference` direction on train;
  - (ii) the best single neuron by train AUROC (search; the candidate count is recorded).
- **Encoding controls:** `random_directions(200, covariance, seed 11)` (primary) plus the isotropic variant (reported), and `label_permutation(200, seed 12)`, each at `min_fraction_below = 0.95`.
- **Use test:**
  - relation DECREASES;
  - target: the log-softmax of the class with the highest train-split lift P(class | C = 1)/P(class), declared by rule;
  - evaluation subset: concept-positive test samples;
  - `min_change = 0.1` nats; `min_fraction_beyond_controls = 0.95`;
  - controls: `random_directions(50, covariance, seed 13)`, or random neurons for neuron features;
  - interventions (§R4): projection removal with a **zero** reference and with a **train-mean** reference (both run; both reported); neurons with zero and train-mean replacement.

**Sites and concepts:**

| Model | Site (pooling) | Concepts (label rule) |
|---|---|---|
| A: MLP / breast_cancer | `net.1` (64; none) | K1 size: mean radius > train median. K2 texture: mean texture > median. K3 smoothness: mean smoothness > median. K4 fractal: mean fractal dimension > median |
| B: CNN / digits | `relu2`, axis 1 (16 channels; mean over H, W) | K5 ink: total ink > train median. K6 centre: ink in the central 4×4 > median |
| C: BERT-tiny / SST-2 validation | `bert.pooler` (128; none) | K7 positive: the dataset label (label-aligned; a *positive-control* case). K8 length: token count > train median. K9 negation: contains one of {not, n't, no, never, nothing, nobody} |

**Hypotheses** (pre-registered; the verdicts are reported whatever they are):
- **R1:** ENCODES is SUPPORTED for ≥ 7 of the 9 concepts with the direction feature (information is preserved).
- **R2 (primary falsification, request §42):** at least one concept has ENCODES SUPPORTED and use NOT SUPPORTED under both references. Predicted: K8 (length) and/or K4 (fractal).
- **R3:** K7 (label-aligned) is VALIDATED under both references.
- **R4 (request §44):** for at least one concept, the use outcome differs between the zero and the train-mean reference.
- **R5:** the best single neuron has a lower test AUROC than the mean-difference direction for ≥ 6 of 9 concepts.
- **R6 (request §43):** at every site, 20 seeded random directions proposed as "concept K1 / K5 / K7" are evaluated on a seeded 40-sample subset of the test split. A naive AUROC ≥ 0.6 criterion passes for ≥ 1 of the 20 at each site; with the mandatory controls, ≤ 2 of 20 pass.
- **R7:** covariance-matched random directions are a stronger null than isotropic ones. The median control AUROC is higher for covariance directions at every site.

## 23. Mutation / adversarial tests

Each mutation must make at least one test fail or produce a refusal:
1. a wrong concept label attached to the evidence;
2. a wrong feature index;
3. a wrong direction vector;
4. a changed direction norm (a different digest);
5. a wrong model checkpoint;
6. a wrong dataset split (test used for fitting);
7. train/test leakage in the counterexample threshold;
8. random labels accidentally pass;
9. a random direction treated as validated;
10. a probe result accepted without controls;
11. causal evidence omitted from a validation;
12. the replacement silently changed;
13. counterexamples dropped;
14. a generated label promoted automatically;
15. evidence from one concept attached to another;
16. a wrong SAE checkpoint;
17. validation surviving deletion of its causal evidence;
18. controls computed but ignored in the outcome;
19. sign-free control scoring removed;
20. a validation status that does not follow from its assessments.

## 24. Performance requirements

- **Budgets** (CPU, one thread):
  - an encoding test with 200 + 200 controls on ≤ 300 samples: under 60 s for BERT-tiny, under 10 s for A/B;
  - a use test with 50 controls on ≤ 150 samples: under 5 min for BERT-tiny;
  - composition with full re-derivation: under 10 s.
- **Measured:** feature-activation extraction, the probe (AUROC) with controls, random-direction scaling (n = 50/200/800), the use test, dataset-level validation, and composition.

## 25. Gate criteria

- **NO-GO:** a validation record misstates its evidence and cannot be corrected; or the concept distinction (encodes vs uses) cannot be represented without breaking Phase 1–5.5 semantics.
- **REQUIRES CHANGES BEFORE PHASE 7:** a substantive issue found and not fixed, or ground-truth case B cannot be distinguished.
- **READY WITH EXPLICIT LIMITATIONS:** all of the following, with remaining issues bounded and documented:
  - all ground-truth cases behave as pre-registered, or deviations are explained and logged;
  - B, F and J are handled exactly;
  - the realistic experiments complete;
  - the mutation tests are caught;
  - composition re-derives everything.
- **READY:** as above, with no substantive limitation.
- **Positive realistic results are not required.**

## 26. Corrections log

(Empty at pre-registration. Corrections are appended with the original text kept.)
