# Phase 5.5 Plan: Realistic Faithfulness Validation (pre-registration)

- **Status:** written and committed **before** any main experiment ran.
  - Hypotheses and analysis rules below must not change after results are seen.
  - Any correction is appended in §18 with the original text kept.
- **Baseline** (verified 2026-09-26):
  - HEAD `28d4026`, clean tree.
  - Python 3.10.16 and 3.12.13 (torch 2.14.0), and 3.14.3 (torch 2.12.0).
  - Without Captum: 877 passed plus 1 reported skip. With Captum 0.9.0: 898 passed.
  - ruff, format, and mypy `--strict` (both ways) clean; sdist and wheel build.
  - Clean-wheel smokes (Phases 1–5) and all 5 README examples pass, with and without Captum.
- **Literature:** `docs/research/PHASE_5_5_LITERATURE.md`.
- **Principle:** try to falsify BeyondNN's premise; do not try to prove it.

## 1. Research questions

- **RQ1.** Do attribution-selected units produce larger intervention effects than matched random controls on real trained models?
- **RQ2.** Which attribution methods, if any, better predict intervention-sensitive units (scoped; no universal ranking)?
- **RQ3.** How often do comprehensiveness and sufficiency disagree?
- **RQ4.** How sensitive are conclusions to the replacement strategy?
- **RQ5.** How sensitive are outcomes to k and to the threshold?
- **RQ6.** Does attribution-method disagreement (rank/top-k) correspond to faithfulness disagreement?
- **RQ7.** Are results consistent across declared held-out samples, rather than isolated examples?
- **RQ8.** Do the current abstractions work without architecture-specific hacks across MLP, CNN, and transformer?
- **RQ9.** Does BeyondNN reveal disagreement that a single explanation or confidence score would hide?

## 2. Hypotheses (directional; fixed now)

- **H1 (RQ1).** Setting: models A and B, k = 10%, replacement r1.
  - Both the IG-selected and the ablation-ranked selections beat count-matched controls: median **superiority** (fraction_below + ½·fraction_tied) ≥ 0.8.
  - For C, the same holds at a weaker level: median superiority ≥ 0.6.
- **H1b (RQ1, controls).** Magnitude-matched controls lower the superiority of input × gradient more than that of IG, because input × gradient favours units with large |x − b|.
- **H2 (RQ2).** Median comprehensiveness drop at k = 1 is ordered ablation ≥ IG ≥ input × gradient ≥ gradient.
  - For ablation at k = 1 this is circular (it ranks by exactly that perturbation) and is reported as such.
  - No ordering is predicted for k ≥ 5%.
- **H3 (RQ3).** Comprehensiveness and sufficiency outcomes (at the primary threshold) disagree in ≥ 10% of (sample, method, k, replacement) cells.
- **H4 (RQ4).** ≥ 10% of (sample, method, k) comprehensiveness outcomes change between replacements, and at least one median method ordering reverses between replacements in at least one model.
- **H5 (RQ5).** ≥ 20% of cells change outcome as the threshold moves across t ∈ {0.25, 0.5, 0.75}.
- **H6 (RQ6).** Across (sample, method-pair, k) cells, top-k Jaccard is negatively but weakly associated with |Δ comprehensiveness drop|: Spearman ρ < 0 with |ρ| < 0.5.
- **H7 (RQ7 / control sanity).** A seeded random ranking treated as a "method" has median superiority in [0.35, 0.65] in every model.
- **H8 (RQ8).** Model A is expressible with the Phase-5 abstractions. B (pixels/channels) and C (tokens) are **not**: selection refuses tensors with non-singleton non-last dimensions. A general unit-axes abstraction will be required (predicted ADR-034).
- **H9 (RQ9).** Every model shows cells where the same method both SUPPORTS and CONTRADICTS across samples, and cells where protocols disagree. A single aggregate would hide ≥ 10% of disagreeing cells.

## 3. Models

All are trained or pretrained realistically; none are hand-constructed.

| ID | Model | Training | Why |
|---|---|---|---|
| A | MLP 30 → 64 → 32 → 2 (ReLU) | Adam lr 1e-3, CE loss, full batch, 300 epochs, torch seed 0; final epoch kept | real tabular classifier, vector units |
| B | CNN: Conv(1, 8, 3, p1)–ReLU–Conv(8, 16, 3, p1)–ReLU–MaxPool(2)–Flatten–Linear(256, 10) | Adam lr 1e-3, CE loss, batch 64, 30 epochs, torch seed 0; final epoch kept | real image classifier; pixels and channels |
| C | `M-FAC/bert-tiny-finetuned-sst2` (4.4M params), revision `41ad6709ec46b414749b37daf49cf5ca1c7dba7c` | pretrained (external) | real transformer; tokens |

Checkpoints are recorded by SHA-256. A and B are re-trainable deterministically on CPU.

## 4. Datasets

| ID | Dataset | Details |
|---|---|---|
| A | scikit-learn `load_breast_cancer` (bundled with scikit-learn 1.9.0; 569 × 30) | stratified split 60/20/20 (`train_test_split`, random_state 0, then 0 again for val/test); standardised with training mean/std |
| B | scikit-learn `load_digits` (bundled; 1797 × 8 × 8, 17 grey levels) | scaled /16; the same split rule |
| C | GLUE SST-2 validation (872 sentences), `nyu-mll/glue` revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` (parquet) | tokenised with the model's own tokenizer |

## 5. Sample selection (no cherry-picking)

- **A and B:** from the **test** split, keep the correctly classified samples, then take a seeded random 60 (`torch.Generator().manual_seed(1234)` permutation, first 60).
- **C:** from validation, keep the correctly classified sentences with 8–24 tokens including [CLS]/[SEP] (a pre-declared compute bound), then take a seeded random 40 (seed 1234).
- **Target** (per sample, fixed from the clean pass before any perturbation): **margin** = logit[predicted] − logit[runner-up], via `metrics.difference`.
- Case studies may be shown later and are labelled "case study". They never replace dataset results.

## 6. Attribution methods

The same site is used for attribution and perturbation.

| Method | Implementation | Unit score |
|---|---|---|
| gradient | BeyondNN native | for sites with within-unit axes: l2 over within-unit axes |
| input × gradient | native | sum over within-unit axes |
| IG, n = 32, riemann_middle | Captum 0.9 adapter for A/B input sites (exercises the adapter); native for internal sites; Captum LayerIG for C | sum over within-unit axes |
| ablation ranking | BeyondNN interventions: the single-unit removal drop for each unit (INTERVENTIONAL, not ATTRIBUTED) | the drop itself |
| random | seeded permutation (a control "method") | — |

- **IG baselines:** zero (r1) for A/B; layer-space zero for C.
- **Ranking:** by |unit score|, ties broken by lower index.
- Attention weights are not used.

## 7. Sites and intervention strategies

| Model | Input-level site | Internal site |
|---|---|---|
| A | `args[0]`, 30 features | first hidden ReLU output, 64 neurons |
| B | `args[0]` pixels, 64 (unit axes H×W; needs H8's change) | second conv ReLU output, 16 channels (unit axis C) |
| C | none (integer ids) | `bert.embeddings.word_embeddings` output, one unit per token position (needs H8's change) |

All positions, including [CLS]/[SEP], are units for C; this is recorded as a limitation. Perturbations are BeyondNN interventions (removal or retention of units), replaced as in §10.

## 8. Faithfulness protocols

- **Tests:** `comprehensiveness` and `sufficiency` at every (sample, method, k, replacement).
- **Primary criterion:** t = 0.5 × margin.
  - Comprehensiveness SUPPORTS iff drop ≥ 0.5·margin.
  - Sufficiency SUPPORTS iff drop ≤ 0.5·margin.
- **Curves:** removal and retention along each method's ranking, at points {0, k₁, …, k₄} (plus n for retention), under r1.
- **Stability (model B only):** declared transformation "roll the image right by one pixel (circular)", with the matching unit map. IG with k = 10%, and every aspect reported.

## 9. Random and control baselines

- **Count-matched** (Phase 5): N = 50 per test, seed = 10 000 + sample index. The null is "a random set of the same size at the same site".
- **Perturbation-magnitude-matched** (new, predicted ADR-035): N = 50, stratified on |x_u − b_u| (l2 over within-unit axes).
  - Units are split into 4 equal-count magnitude strata. Each selected unit is matched by a random unit from its stratum, without replacement.
  - The null is "a random set perturbed by similar amounts". It addresses the confound "large perturbations change the output regardless of the method".
  - Run for comprehensiveness under r1 only (compute bound).
- **Random-ranking method** (H7).

## 10. Replacement strategies (what, with what, why, OOD caveat)

| Site | r1 | r2 | r3 |
|---|---|---|---|
| A input (standardised) | zero, which equals the training mean (by construction) | resample: the features of one seeded random training row | training minimum per feature ("absence-like"; strongly OOD) |
| A hidden | zero | per-neuron training mean activation | resample: the activation of a random training row |
| B pixels | zero (black) | per-pixel training mean image | resample: a random training image |
| B channels | zero | per-channel training-mean activation map | resample: the activation of a random training image |
| C tokens | zero embedding | the [MASK] token embedding at every position (the same as masking the input ids) | the [PAD] token embedding |

- **Every replacement is OOD** to some degree and is recorded as such.
- **Resample seed:** 20 000 + sample index.

## 11. k policy

- p ∈ {1%, 5%, 10%, 20%}, with k = max(1, ceil(p·n_units)). Duplicates collapse.
- The same k values are used for every method.
- All k are reported; there is no best-k selection.

## 12. Threshold policy

- **Primary:** t = 0.5 (fraction of the clean margin).
- **Sensitivity grid:** t ∈ {0.1, 0.25, 0.5, 0.75, 0.9}, computed from the recorded raw drops. The outcome is a deterministic function of drop and threshold; the raw drops are always reported.

## 13. Statistical analysis

- **Per cell:** median and IQR of drops.
- **Outcome shares:** fraction SUPPORTS / CONTRADICTS / INCONCLUSIVE.
- **Superiority distribution:** median and IQR, plus the Monte-Carlo p distribution.
- **Paired method comparisons:** per-sample drop differences, median, win rate, and a seeded sign-flip p (5 000 draws, seed 0).
- **RQ6:** Spearman ρ between Jaccard and |Δdrop|.
- **Language:** no "significant". Statistics stay separate from assessments.

## 14. Expected failure modes

- IG and gradient saturation in trained ReLU nets.
- Redundant features in breast-cancer (highly correlated features), so single-feature comprehensiveness will be weak.
- Pixel mask-shape leakage (ROAD).
- The [CLS]/[SEP] embeddings dominate in C.
- The resample replacement is sometimes a no-op-like change.
- Compute: C with many controls.

## 15. API stress test

- Only public APIs are used. Every private access, workaround, confusing message, or misuse-inviting default is logged in `docs/PHASE_5_5_API_REVIEW.md` with a class: BUG / SCIENTIFIC DESIGN ISSUE / API ERGONOMICS / MISSING CAPABILITY / DOCUMENTATION ISSUE / EXPECTED LIMITATION.
- The first attempt at B and C is made with the unchanged Phase-5 API, and the failure is recorded before any change.

## 16. Performance

Per model:
- plain forward, trace, and each attribution method;
- one intervention;
- one faithfulness test with N = 50;
- the dataset runtime;
- the trace record count.

Hardware and all package versions are recorded.

## 17. Gate criteria

- **NO-GO:** a realistic experiment exposes a record that misstates what was done (sample, target, replacement, or unit association) and cannot be corrected within the phase; or the evidence structure cannot be represented without breaking Phase 1–5 semantics.
- **REQUIRES CHANGES BEFORE PHASE 6:** a substantive issue is found and not fixed in-phase.
- **READY WITH EXPLICIT LIMITATIONS:** all three settings are expressible through general abstractions, the experiments complete, and remaining issues are fixed or bounded and documented.
- **READY:** as above, with no substantive limitation remaining.
- **Positive results are not required** for any gate.

## 18. Corrections log

(Empty at pre-registration. Corrections are appended here, with the original text kept.)

**Appended 2026-09-26, before the analysis script was run on any full result file.**
- **What had been seen by then:**
  - progress logs;
  - the two-sample model-A smoke run (outcome counts for one sample);
  - no aggregate over samples.

**Corrections (the original text above is unchanged):**

- **§4 / environment.** scikit-learn **1.9.1** was installed, not 1.9.0. The bundled `load_breast_cancer` and `load_digits` data are the same files across these versions. Recorded in every result's `environment`.
- **§6, model C, IG.** Captum's `LayerIntegratedGradients` accepts only *input-space* baselines (token ids), so it cannot express the pre-registered *layer-space zero* baseline. The baseline is kept and the implementation changes: BeyondNN's native IG at `bert.embeddings.word_embeddings`. Models A and B use the Captum adapter at the input as planned.
- **§6, random method.** The seeded permutation uses seed 30 000 + sample index (the seed was not stated).
- **§6, ablation ranking.** It is computed per replacement, as single-unit removals under the same replacement as the test (circular, as §2 H2 says), and ranked by |drop| as in the general §6 rule.
- **§8, curves.**
  - `faithfulness.curve` refuses declared rankings (API review). The random reference is therefore the curve's own seeded random-ranking controls (N = 20, seed 10 000 + index).
  - The ablation ranking's curve is built from public interventions in one `compare_family` (zero replacement).
  - The library diagnostics `method_agreement` and `ig_step_sensitivity` take one reduction for both attributions, so they use `sum` for every method. H6 is computed from the per-method rankings (the §6 reductions).

**Clarifications of the analysis rules (no hypothesis changed):**

- **H1.** Evaluated per (model, site). "Holds" requires both IG and ablation to reach the bar. There is no cross-site aggregate.
- **H1b, H7.** Evaluated in the H1 setting (p = 10%, r1, comprehensiveness); all k are also reported.
- **H2.** k = 1 is the literal unit count. It is evaluated per replacement; r1 is primary.
- **H3.** A cell "disagrees" if one test SUPPORTS and the other CONTRADICTS. Cells with an INCONCLUSIVE (no-op) outcome are excluded and counted. Reported per site and pooled.
- **H5.** Cells are (sample, method, k, replacement, test) with count controls, no-ops excluded.
- **H6.** Primary: the pairs of attribution methods (gradient, input × gradient, IG), r1, comprehensiveness. With ablation added as a secondary analysis.
- **H9.** The aggregate is the majority outcome per (method, k, replacement, test) cell across samples. "Hidden" means the per-sample results that differ from the majority, as a share of the per-sample results in cells containing both SUPPORTS and CONTRADICTS.

**Provenance of the runs:**

- A framework BUG found by the realistic runs was fixed during the phase: ADR-036, "reproducibility tolerance" (`4862125`). Two defects in the ADR-034 implementation were also fixed: curve units (`2e4e662`) and the recorded reduction (`a5768ec`).
- Model B was re-run from the start after those fixes.
- The model-A and model-C main runs started at `622150d`. They are unaffected:
  - A uses last-axis units;
  - C's units always span 128 elements, so their reduction was always recorded;
  - neither run uses curves;
  - neither hit the tolerance refusal.
- The model-C diagnostics started at `2e4e662`, before `a5768ec`, which does not affect C for the same reason. Each result file records its commit.


**Appended 2026-09-26, after the runs (provenance only; no hypothesis or rule changed):**

- **Model B:**
  - The model-B main run was repeated once more at `39bdd75`, after ADR-038 (a magnitude-verification tolerance: composition had refused the correct magnitude-matched results at the conv-channel site).
  - It reproduced all 14,700 rows of the previous run exactly, and every first-sample result composed.
- **Model C:**
  - The model-C diagnostics were re-run at `3c7650b`, after ADR-037: the diagnostics had refused every BERT sample for lack of `model_kwargs`.
  - The model-C main run (at `622150d`) was checked by re-running its first 3 sentences at HEAD (`8a2a536`): 315 of 315 rows are identical (`faithfulness_C_limit3.json.gz`).
- **§2 H3/H9 wording:** H3 and H9 are evaluated per site and, for H3, pooled, as clarified above.
