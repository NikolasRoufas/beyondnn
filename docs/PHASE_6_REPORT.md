# Phase 6 Report: Concepts and Concept Validation

- **Gate:** **READY FOR PHASE 7 WITH EXPLICIT LIMITATIONS** (§47).
- **Central rule:** decodability does not establish use. Phase 6 makes that distinction a property of the records, not of the prose.
- **Pre-registration:** `docs/PHASE_6_PLAN.md`. Its §26 logs corrections and operational details made before the realistic runs.
- **Results:** `experiments/phase6/results/`.

## 1. Starting commit

`d3c22b0` (Phase 5.5 report), clean tree. Verified: not `28d4026` and not `39499da`.

## 2. Ending commit

The commit that adds this report (hash in the final summary; `git log` is authoritative). All commits are listed in §50.

## 3. Baseline

Run at `d3c22b0`:
- Python 3.10 / 3.12 / 3.14: 905 passed plus 1 reported skip without Captum; 926 passed with Captum 0.9.0.
- ruff, format and mypy `--strict` clean.

## 4. Literature reviewed

`docs/research/PHASE_6_LITERATURE.md`: 30+ sources. [V] = verified on the web on 2026-09-26; [P] = prior knowledge.

| Area | Sources |
|---|---|
| Concept methods | TCAV, CBM, CEM, post-hoc CBM, Network Dissection, ACE, Concept Whitening |
| Probing | linear probes, control tasks/selectivity, MDL, Belinkov's survey, "probing the probing paradigm", INLP/amnesic probing, counterfactual embeddings |
| Causal and representation work | causal mediation, ROME/causal tracing, DAS, the subspace-patching illusion, RepE, the linear representation hypothesis |
| SAEs | Towards/Scaling Monosemanticity, Huben/Cunningham, absorption, dark matter, sparse probing, AxBench, SAEBench |
| Automated labels | Bills et al. [P], Huang et al. 2023 |
| Dataset factors | Ramaswamy et al. 2023 |
| Libraries | Captum TCAV, SAELens, TransformerLens, NNsight, pyvene |

The review includes a comparison matrix and answers to the design questions.

**What it implied for the design:**
- decodability is not use;
- probes need controls;
- subspace interventions can mislead, so causal claims must be scoped;
- labels, including generated ones, are hypotheses;
- SAE features get no privileged status;
- validation is dataset-scoped.

## 5. Concept terminology

- **Feature:** a structural coordinate of a site (neuron, direction, SAE latent). It has no meaning.
- **Concept hypothesis:** a label plus an operational definition attached to a feature. It is PROPOSED.
- **Concept dataset:** the concept's extension (exact samples, 0/1 labels, splits, and a scope name).
- **ENCODES claim:** representation.
- **Use claim:** DECREASES/INCREASES under removal, or SUFFICIENT_FOR under retention.
- **Concept validation:** a derived semantic status under a declared policy and scope.
- **Concept activation:** the value of a validated concept's feature on one input.

## 6. Feature representation

- `feature` records carry:
  - basis, site, call, feature axis, pooling (none/mean);
  - the neuron index, or a retained direction whose digest is part of the identity;
  - the source (declared/fit/search/sae, method, parameters, dataset, split = train);
  - the model state digest (fit/search/sae);
  - an SAE identity.
- Activation rules are in `docs/concepts/features.md`. A changed vector or norm, or a changed SAE checkpoint, is a different feature (tested).

## 7. Semantic-status lifecycle

- **Three states:** UNLABELED_FEATURE → PROPOSED_CONCEPT → VALIDATED_CONCEPT (ADR-039, which supersedes ADR-009's REJECTED_CONCEPT).
- The status is **derived, never stored**:
  - a concept record is always PROPOSED;
  - a `concept_validation` refuses at construction a status or unmet-list that does not follow from its assessment summaries.
- There is no automatic promotion and no global REJECTED state. Negative results remain as CONTRADICTED assessments with their unmet requirements listed.

## 8. Schema changes

**New kinds:**
- `feature`
- `generated_label` (GENERATED)
- `concept`
- `concept_dataset`
- `concept_validation`
- `concept_activation` (VALIDATED_CONCEPT)

**Version bumps:**
- `intervention` v4: `direction` and `direction_axis`; the DIRECTION operation.
- `claim` v3: `Subject.feature`. The golden claim id changed deliberately and is documented.

**Other additions:**
- `SemanticStatus`;
- 13 limitation codes;
- the protocols `concept_encoding`, `concept_intervention` and the diagnostic `concept_counterexamples`;
- `Coverage.concepts_evaluated`, while `concepts_validated` became a real, derived boolean;
- `TraceResult` per-kind and per-pass indexes.

## 9. Public API

`bnn.concepts` exposes:
- **features and data:** `neuron`, `direction`, `sae_feature`, `fit_direction`, `search_neurons`, `dataset`;
- **proposals:** `propose`, `generated_label`;
- **encoding tests:** `encoding_test` with `random_directions`, `random_neurons`, `label_permutation` and `encoding_criteria`;
- **use tests:** `use_test` with `remove` / `retain`, `zero` / `reference` and `use_criteria`;
- **assessment:** `validate` with `POLICY_V1`, and `activation`.

Also: `bnn.interventions.direction`, and `bnn.compose(..., concepts=[...])`.

The steps are separate calls. No call does everything, and every intervention and control is declared.

## 10. Neuron features

- `neuron(site, index, axis=, pooling=)`.
- Interventions reuse unit replacement (`unit_axes=(axis,)`) with a zero or tensor reference.
- The controls are random other neurons of the same axis.
- `search_neurons` records its candidate count, criterion (sign-free train AUROC) and sign.

## 11. Direction features

- `direction(site, v, axis=, pooling=)`, or `fit_direction` (mean difference on the train split only; re-derived in composition).
- Interventions use the new DIRECTION operation:
  - **removal:** x − ⟨x − r, v̂⟩ v̂;
  - **retention:** r + ⟨x − r, v̂⟩ v̂.
- Steering (addition and scaling) was deliberately not built.

## 12. SAE adapter

- `sae_feature(model, site, encoder=, decoder=, b_enc=, b_dec=, latent=, checkpoint=, reconstruction=, search=, data=)` takes tensors only. There is no SAELens dependency.
- **Activation:** the relu SAE encoder. **Intervention:** along the normalised decoder direction (projection semantics, recorded as a limitation).
- **Six SAE limitation codes** are always attached.
- **Case study (§31):** a local SAE, a searched latent, a GENERATED label, and full validation.

## 13. Concept proposal flow

- `propose(feature, label=, definition=, label_source="user"|"dataset")`, or `propose(feature, definition=, generated=generated_label(...))`.
- A generated label must be about the same feature, and its text becomes the label. The claim source becomes GENERATED.

## 14. Representation / ENCODES testing

`encoding_test`:
- runs one clean recording over the train, val and test splits;
- computes the AUROC of the fixed 1-D readout on **test** with the declared sign (fitted directions: positive = concept);
- controls are mandatory;
- records counterexamples with a val-fixed threshold;
- builds the claim, spec, result, assessment (`concept_encoding_v1`) and limitations (`DECODABILITY_NOT_USE`, …).

## 15. Probe policy

- Nothing is trained at evaluation time. Directions are fitted on train only, and fitted features can be evaluated only on their own dataset.
- There are no MLP or logistic probes (literature: probe flexibility).
- **Recorded:** readout, fitting method, split sets (digests), sizes, class balance, dimension, seeds, and controls.

## 16. Control policy

**Mandatory:** encoding and use specs refuse to exist without at least one control and a criterion, and the criterion enters the outcome (mutations 10 and 17 are caught).

| Test | Available controls | Criterion |
|---|---|---|
| Encoding | random directions (isotropic or covariance), random neurons, label permutations | `min_fraction_below` |
| Use | random directions or neurons under the same intervention | `min_fraction_beyond_controls` |

## 17. Random-direction controls

- Same site, axis and pooling; unit norm; sign-free scoring.
- The distribution is declared: isotropic N(0, I), or covariance N(0, Σ_train), using the symmetric Σ^½.
- **Recorded:** seed, n, distribution and matching rule.
- **Composition regenerates them:**
  - exactly for isotropic controls;
  - from the recorded train activations, within 1e-4, for covariance controls (ADR-038 argument).

## 18. Causal-validation policy

VALIDATED_CONCEPT requires all of:
- a SUPPORTED encoding assessment;
- at least one SUPPORTED DECREASES/INCREASES use claim;
- a control criterion in every spec;
- a counterexample record.

SUFFICIENT_FOR (retention) results may be attached as `additional`; they are shown but never required. Use claims are FINITE_SAMPLE over a declared test subset, with one fixed target.

**Per-sample targets were not needed** (request §31). This is recorded in ADR-042.

## 19. Replacement policy

- `remove(reference)` and `retain(reference)`, with `zero()` or `reference(tensor, name=)`, are required. There is no default.
- The reference's kind, name and digest are in the spec. Verification checks that the executed intervention used exactly that reference (mutation 12 is caught).
- **Limitations:** zero/constant OOD, `DIRECTION_REPLACEMENT_MAY_BE_OOD`, and `DIRECTION_INTERVENTION_MAY_ACTIVATE_DORMANT_PATHWAYS`.

## 20. Counterexample handling

- The threshold is fixed on val (maximum balanced accuracy).
- **All** test false positives and false negatives are listed with sample ids and values, and the rates are recorded.
- Verification recomputes them, so dropped counterexamples are refused (mutation 13).
- The rates are shown next to every status.

## 21. Semantic-status promotion rules

`derive_semantic_status(policy, encoding, use, fp_rate, fn_rate)` is the only path. Its output is compared at construction, then re-derived in composition from:
- the assessments;
- the results, re-derived from the activations and effects;
- the controls, regenerated.

## 22. Structured-WHY integration

`bnn.compose(trace, concepts=[validation, activation])` adds a **CONCEPTS** section with:
- the encoding and use outcomes on separate lines;
- the unmet requirements;
- the scope;
- the counterexample rates;
- the feature's value on this input, labelled VALIDATED_CONCEPT or MEASURED;
- the limitations.

**Concept evidence is context:** it is not merged with the instance claims or target, and it never overrides attribution or intervention evidence.

**Composition refuses:**
- another checkpoint;
- forged or re-attached validations;
- activations about another input;
- VALIDATED activations without their validation.

## 23. Controlled ground-truth results

All ten pre-registered cases (full N) behaved as predicted. The table is in `EXPERIMENT_LOG.md` and `ground_truth.json`.

| Case | Encoding | Use | Status |
|---|---|---|---|
| A | 1.0 | −2.62 | VALIDATED |
| B | 1.0 | 0.0 (CONTRADICTED) | PROPOSED |
| C | neuron 0.83; direction 1.0 | neuron −0.82; direction −1.71 | both VALIDATED |
| D | — | neuron 0 (CONTRADICTED); direction −0.82 | direction VALIDATED |
| E1 / E2 | E1 validated; E2 0.53 (CONTRADICTED) | — | E1 VALIDATED |
| F | 5/50 naive vs 2/50 controlled | — | — |
| G | 0.49 (CONTRADICTED) | — | — |
| H | 0.85; false-positive rate 0.26 | — | VALIDATED (v1 has no caps) |
| I | neuron false-negative rate 0.58 | — | direction VALIDATED |
| J | wrong label CONTRADICTED | — | GENERATED stays PROPOSED |

## 24. Decodable-but-unused result

- **Ground truth B:** h1 = x1 feeds no output. ENCODES SUPPORTED (AUROC 1.0 above both controls); removal effect exactly 0.0; use CONTRADICTED; status PROPOSED_CONCEPT. The WHY prints "ENCODING CLAIM: SUPPORTED" and "USE CLAIM …: CONTRADICTED".
- **Realistic:** BERT pooler neurons
  - for sentence length: test AUROC 0.75; use effect −0.006 nats (zero) and −0.005 (train-mean);
  - for negation: AUROC 0.87; effects −0.015 and −0.001.
  - Both are ENCODES SUPPORTED under both nulls and use CONTRADICTED.
  - MLP neurons for size, texture and fractal dimension behave the same way (AUROC 0.84–0.93, effects |≤ 0.11|).

## 25. Distributed and redundant results

- **C (distributed):** the single neuron carries half of the effect (−0.82 vs −1.71 for the spanning direction). Both validate: partial, not absent.
- **D (redundant):** the single neuron's removal changes nothing (CONTRADICTED), while the spanning direction validates (−0.82).
- **Sufficiency:** a unit test shows the single neuron is *sufficient* under retention (SUFFICIENT_FOR SUPPORTED, `additional`).
- The WHY records `SINGLE_FEATURE_TEST_MISSES_REDUNDANCY` rather than "unused".

## 26. Random-direction result

- **Ground truth F** (test n = 16): 5/50 random directions pass a naive AUROC ≥ 0.7; with the mandatory controls, 2/50 pass (≈ α, as expected for draws from the null).
- **Realistic R6** (test subsample of 40): naive AUROC ≥ 0.6 in 13/20 (MLP), 11/20 (CNN) and 6/20 (BERT); **0/20** pass the controlled test at every site.

## 27. Generated-label result

- **J:** the GENERATED label "x0 > 0" on h1 has evidence status GENERATED. It is refused as evidence, ENCODES is CONTRADICTED (AUROC 0.46), and it stays PROPOSED.
- The same label on h0 becomes VALIDATED only through `validate` with passing tests; the label record stays GENERATED.
- **SAE case:** a GENERATED label stays PROPOSED (§31).

## 28. Realistic MLP result

Breast cancer; `net.1`; 569 samples; test split n = 172.

| Concept | Direction AUROC / encoding (covariance null) | Use (zero / train-mean) | Status |
|---|---|---|---|
| K1 size | 0.92 / CONTRADICTED | −2.99 S / +1.60 C | PROPOSED |
| K2 texture | 0.82 / SUPPORTED | −1.01 C / +3.22 C | PROPOSED |
| K3 smoothness | 0.81 / CONTRADICTED | C / C | PROPOSED |
| K4 fractal | 0.90 / SUPPORTED | −0.44 S / −0.26 S | **VALIDATED under both references** |

The best neurons encode K1, K2 and K4 but are causally inert.

## 29. Realistic CNN result

Digits; `relu2` channels, mean-pooled; 1,797 samples.

| Concept | Direction AUROC / encoding (covariance null) | Use | Status |
|---|---|---|---|
| K5 ink | 0.97 / CONTRADICTED (covariance median 0.963) | — | PROPOSED |
| K6 centre | 0.79 / CONTRADICTED | — | PROPOSED |

- The best neuron for K6 encodes (0.79) but is not used.
- 7 of 8 use effects were *positive*: the intervention raised the lift class's log-probability.
- **No concept validated.**

## 30. Realistic transformer result

BERT-tiny; `bert.pooler`; SST-2 validation, 872 sentences.

| Concept | Direction AUROC / encoding (covariance null) | Use | Status |
|---|---|---|---|
| K7 positive (label-aligned) | 0.88 / CONTRADICTED (covariance median 0.878) | zero CONTRADICTED; train-mean SUPPORTED (−0.11) | PROPOSED |
| K8 length | 0.61 / SUPPORTED | CONTRADICTED under both | PROPOSED |
| K9 negation | 0.76 / CONTRADICTED | zero SUPPORTED (−0.48); train-mean CONTRADICTED | PROPOSED |

The best neurons for K8 and K9 decode but are unused (§24). **No concept validated.**

## 31. SAE case-study result

- A 64→128 SAE was trained locally on K1-train activations of `net.1`: validation explained variance 0.986, L0 ≈ 69/128 (barely sparse: the L1 penalty was weak), no dead latents.
- Latent 115 was chosen by search from 128 candidates (train AUROC 0.917, sign −1).
- The GENERATED label ("responds to low mean perimeter", from a rule-based stand-in labeller) was tested against the K1 extension:
  - ENCODES CONTRADICTED: AUROC 0.866 against a covariance median of 0.851;
  - use CONTRADICTED under both references;
  - status PROPOSED.
- The adapter represented everything required: source site, checkpoint digest, encoder/decoder, activation, label, probe evidence, intervention, controls, counterexamples and limitations.
- **Finding:** the label's polarity is the opposite of the dataset concept. BeyondNN tests the extension, not the label text (§45).

## 32. Replacement sensitivity

**R4 held.** The reference changes the use outcome for MLP K1, BERT K7 and BERT K9. The most extreme case: MLP K1 direction removal gives −2.99 nats with a zero reference and **+1.60** with the train-mean coordinate (both recomputed by hand, exactly). The network is non-monotonic along the direction once the orthogonal components are held fixed.

## 33. Control sensitivity

- **R7 not held.** The covariance null is stronger than the isotropic one for 6/9 concepts and weaker for K4, K8 and K9.
- The null decides the encoding outcome in 7 of 18 feature tests: the K1, K5, K7 and K8 directions, and the K5, K6 and K7 neurons.
- **The covariance null can be over-matched:** at the BERT pooler it contains the sentiment axis, so it rejects the task label itself (R3 not held).
- Recommendation: declare both nulls, with explicit question semantics. The isotropic null asks "better than an arbitrary direction?"; the covariance null asks "better than a typical high-variance direction?".

## 34. Stability findings

- **Implemented and tested:**
  - control-draw reproducibility (seeded, local; re-derived in composition);
  - run-to-run identity (the ground truth was re-run with identical results);
  - Phase-5.5 compatibility (identical values).
- **Not implemented in v0** (plan §23; request §23 allows scoping): cross-seed direction stability, cross-checkpoint transfer (refused by design, not measured), and label stability.

## 35. Counterexamples

Every realistic encoding test lists its counterexamples. False-positive rates range from 0.09 to 0.50 and false-negative rates from 0.07 to 0.65. The K9 direction has a false-positive rate of 0.50; the K8 direction a false-negative rate of 0.65. Ground truth H: 0.26. None is dropped (verification refuses that).

## 36. Failures and inconclusive results

**Realistic pre-registered hypotheses not held:**
- R1 (3/9 directions encode);
- R3 (the label-aligned concept is not validated);
- R5 (the best neurons beat mean-difference directions in 7/9);
- R7 (the covariance null is not always stronger).

**Other outcomes:**
- Only 1 of 9 realistic concepts validated.
- Ground-truth case I's single neuron: use CONTRADICTED. The plan said "partial"; this is consistent, but the outcome was not predicted explicitly.

## 37. API issues

| # | Issue | Class | Status |
|---|---|---|---|
| P6-1 | O(N²) trace lookups at concept scale | BUG (performance) | **fixed** (ADR-043) |
| P6-2 | A fitted or searched feature cannot be evaluated on a relabelled dataset with the same split (refused to guarantee disjoint splits) | API ERGONOMICS | open (bounded) |
| P6-3 | The label text is not checked against the concept extension (SAE case: opposite polarity) | SCIENTIFIC DESIGN ISSUE | open; the user's responsibility; documented |
| P6-4 | Result objects (`EncodingResult`, …) cannot be rebuilt from saved traces; composition needs the live objects | MISSING CAPABILITY | open (traces persist and reload) |
| P6-5 | Concept validations appear only as context in the instance WHY; there is no dataset-level WHY | MISSING CAPABILITY | open (Phase 7 audit territory) |
| P6-6 | v1 policy has no counterexample-rate caps (H validated at FP 0.26) | SCIENTIFIC DESIGN ISSUE | open; caps are available |
| P6-7 | The covariance null can be over-matched | SCIENTIFIC DESIGN ISSUE | open; documented; both nulls are available |
| P6-8 | Mean references are undefined for variable-shape sites (token sequences); the BERT pooler was used instead | EXPECTED LIMITATION | documented |
| P6-9 | Readouts are 1-D; a distributed concept needs an explicit direction (C, D, I) | EXPECTED LIMITATION | documented |

## 38. Framework changes

| Change | ADR |
|---|---|
| Semantic lifecycle and concept records | ADR-039 |
| Features, `Subject.feature`, DIRECTION interventions | ADR-040 |
| SAE adapter | ADR-041 |
| Concept protocols, controls, validation, WHY integration | ADR-042 |
| Trace indexes | ADR-043 |

The base install is still torch only. Experiment dependencies are in `experiments/phase5_5/requirements.txt`.

## 39. Changes considered but rejected

- **A global REJECTED state:** scope-relative (ADR-039).
- **A stored status on concepts.**
- **Trained probes at evaluation time.**
- **TCAV scores as use evidence.**
- **Steering (direction addition and scaling).**
- **SAE-native ablation:** needs the full SAE forward; a candidate for later.
- **SAELens dependency.**
- **Per-sample targets in `run_dataset`:** not needed (ADR-042).
- **Logistic, INLP and DAS fitters:** out of scope; mean difference is verifiable exactly.
- **Optional controls.**
- **Encoding-only validation.**
- **REJECTED_CONCEPT from ADR-009.**
- **An aggregate concept score** (ADR-007).

## 40. Schema migrations

- **intervention v3→v4:** `direction`/`direction_axis` = None.
- **claim v2→v3:** `subject.feature` = None.
- The v1/v2 chains are preserved and tested (loading a forged v3 intervention and a v2 claim).
- **Phase 1–5.5 traces load.** Nothing is invented: no labels, statuses, controls or unit axes.
- **Record ids of newly produced claims and interventions change** (the Phase-5.5 re-run is identical in values, with different ids).

## 41. Performance

Idle machine, one thread (`performance.json`):

| | Toy (600 samples) | MLP (569) | BERT-tiny (872) |
|---|---|---|---|
| activation extraction | 0.28 s | 0.24 s | 7.9 s |
| encoding with 200 + 200 controls | 0.31 s | 0.40 s | 8.1 s |
| random directions, 50 → 800 | 0.27 → 0.32 s | 0.29 → 0.33 s | 8.06 → 8.21 s |
| use test with 50 controls | 3.2 s (88 samples) | 2.8 s (70 samples) | 79 s (143 samples) |
| validate | 0.006 s | 0.006 s | 0.011 s |
| composition with full re-derivation | 0.66 s | 0.54 s | 1.33 s |

- All plan §24 budgets are met.
- **Realistic full-grid runtime** (3 concurrent processes): MLP 67 s, CNN 132 s, BERT 1030 s.

## 42. Mutation / adversarial tests

- **26 mutations of the Phase-6 code**, covering:
  - the wrong concept, feature index, direction, norm, checkpoint, split, or counterexample threshold;
  - ignored permutations;
  - a failed encoding validated;
  - no controls, or no use evidence;
  - the replacement changed;
  - counterexamples dropped;
  - a generated label hidden;
  - the SAE checkpoint dropped;
  - summaries not verified;
  - controls ignored;
  - sign-free scoring removed;
  - a forged status;
  - removal and retention swapped;
  - composition skipping re-derivation or the checkpoint check;
  - a wrong trace index;
  - `min_change` ignored;
  - the relation check removed;
  - the control-direction check disabled.
- **First run: 24 caught.** Two survivors (`min_change` ignored; control-direction check disabled) exposed test gaps. Tests were added (`test_a_small_effect_beating_controls_is_not_enough`, `test_recorded_control_directions_must_match_their_declaration`), and both are now caught. **26/26.**

## 43. Final test counts

| Environment | Result |
|---|---|
| Python 3.10.16 / 3.12.13 / 3.14.3 (torch 2.14.0), without Captum | **934 passed, 1 skipped** (the Captum cross-check module, reported) |
| same, with Captum 0.9.0 | **955 passed** |

The baseline was 905 + 1 / 926. There are 29 new concept tests (`tests/test_concepts.py`). Existing guard tests were updated deliberately:
- registered kinds;
- limitation codes;
- the top-level API (`concepts`);
- the status audit (`generated_label` GENERATED, `concept_activation` VALIDATED_CONCEPT);
- the golden claim id (Claim v3);
- migration fixtures.

There are no other regressions.

## 44. Lint / type / build results

- ruff check: clean. ruff format: 117 files already formatted (`beyondnn`, `tests`, `experiments`, `benchmarks`, `examples`).
- mypy `--strict`: clean with and without Captum (100 source files).
- sdist and wheel build (`uv build`).
- **Fresh wheel install** (Python 3.12), without and with Captum:
  - the Phase 1, 2, 3, 4, 5, 5.5 and 6 smokes pass (the Phase-6 smoke gives `['validated_concept', 'proposed_concept']`);
  - all 6 README examples pass, including the concept example ("supports contradicts proposed_concept").

## 45. Remaining scientific weaknesses

1. **Choosing the null is a scientific decision** that changes outcomes (R7); the covariance null can be over-matched (K7).
2. **Use conclusions depend on the intervention** (R4). Projection removal can activate off-manifold behaviour (MLP K1 +1.60), and dormant pathways are not excluded.
3. **Encoding is decided on one held-out split** of modest size (60–262). There is no cross-seed or cross-dataset stability yet.
4. **v1 validates features with high counterexample rates** (H), unless caps are declared.
5. **Label text is not checked** against the operational extension (SAE case).
6. **1-D readouts only.** Distributed concepts need an explicit direction.
7. **Realistic concepts are dataset-derived properties**, not rich human semantics. The SST-2 "label-aligned" case shows how easily task-aligned concepts behave differently.
8. **The SAE case used a barely sparse, locally trained SAE.**
9. **Ground truth is hand-built.** Realistic models provide no ground truth for use.

## 46. Remaining engineering limitations

- Results are not reconstructable from saved traces (P6-4).
- There is no dataset-level WHY (P6-5).
- Fitted features are bound to their dataset (P6-2).
- Mean references are unavailable for variable-shape sites (P6-8).
- A BERT use test costs about 11 ms per paired pass, with 51 passes per sample (79 s for 143 samples).
- Release blockers RB-1 to RB-3 remain (contact placeholders; CI never run). Nothing was pushed.

## 47. Gate decision

**READY FOR PHASE 7 WITH EXPLICIT LIMITATIONS.**

**Why not NO-GO:** the encodes/uses distinction is represented, derived and verified end to end. Ground-truth case B is handled exactly. No validation record misstates its evidence: composition re-derives features, statistics, controls, interventions, counterexamples, assessments and status, and 26/26 mutations are caught.

**Why not REQUIRES CHANGES:**
- The substantive framework problem found (the O(N²) trace scaling) was fixed with an ADR.
- The ground-truth suite behaved as pre-registered.
- The open items (P6-2 to P6-9) are bounded: they are documented limitations or user-declared choices, and none makes a record wrong.

**Why not READY without limitations:**
- Realistic results show that concept conclusions depend strongly on the declared null and the intervention semantics.
- Only 1 of 9 realistic concepts validated, and the label-aligned positive control did not.
- v1 lacks counterexample caps.
- There is no dataset-level presentation.

## 48. What Phase 7 may assume

- Features (neuron, direction, SAE) and concept hypotheses are first-class, provenance-bound records. Their semantic status is derived under a declared policy and re-derivable.
- The ENCODES and use claims are separate, with mandatory controls and a declared intervention. Decodable-but-unused features are reported as such.
- Concept validations are bound to a checkpoint and scope. Composition refuses mismatches and forgeries.
- DIRECTION interventions and trace indexes are available for large comparison families.

## 49. What Phase 7 must NOT assume

- That VALIDATED_CONCEPT means the model "uses" or "understands" a concept beyond the recorded scope, intervention and target.
- That any single null or reference is neutral; audits must declare both, and report the dependence.
- That counterexample rates are bounded under v1; audits should declare caps.
- That concept results can be reloaded as objects, or composed at the dataset level.
- That label text matches the tested extension.
- That SAE latents are monosemantic.
- That results transfer across checkpoints or datasets.

## 50. Commits

Phase 6 commits after `d3c22b0`, in order:

1. `c48d30c` Literature review and pre-registered plan
2. `068102a` Concept features, proposals and controlled validation (ADR-039..043)
3. `eff0ef8` Pre-registered ground-truth suite and results
4. `f733063` Plan §26: implementation names and realistic-run details (before realistic runs)
5. `72a7e9d` Realistic concept validation and SAE case study
6. `4764e92` Documentation
7. `4cc5780` Tests for the two mutation survivors
8. `90cb054` Performance measurements and flagship-example test
9. This report (with the experiment log, roadmap and docs index)

All commits are local. Nothing was pushed or published, and no commit has a co-author trailer.

## 51. Git status

Clean after the report commit. The ignored `experiments/phase6/artifacts/` holds the run logs.

## 52. Uncommitted work

None.
