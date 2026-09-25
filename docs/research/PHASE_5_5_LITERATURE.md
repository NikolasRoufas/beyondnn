# Phase 5.5 Literature Review: Realistic Faithfulness Evaluation

- **Date:** 2026-09-26.
- **Purpose:** inform the design of Phase 5.5 (`docs/PHASE_5_5_PLAN.md`) before any realistic experiment runs.
- **Verification level**, per source:
  - **[V]** checked against the original paper page (arXiv/ACL Anthology) in this or the previous session;
  - **[P]** from prior knowledge of the original paper, not re-fetched in this session.

  Claims are limited to what the source states.

## Summary of what the literature implies for Phase 5.5

1. **The removal value is part of the experiment.**
   - Zero, mean, resample, blur, inpainting, and "optimal" ablation give different rankings and conclusions (Blücher 2024 [V]; Li & Janson 2024 [V]; Zhang & Nanda 2024 [V]; Miller 2024 [V]; Hase 2021 [V]; Sturmfels 2020 [P]).
   - Phase 5.5 must vary the replacement and report reversals.
2. **Removal creates off-distribution inputs.**
   - The fixes are retraining (ROAR, Hooker 2019 [V]), debiased imputation (ROAD, Rong 2022 [V]), or realistic occlusion (Blücher 2024).
   - None removes the confound for arbitrary internal sites. BeyondNN should keep recording it rather than claim to fix it.
3. **Evaluation metrics disagree with each other and are statistically fragile** (Tomsett 2020 [V]; Hedström 2023 MetaQuantus [V]; Krishna 2022 [V] for explanation methods themselves). Any single aggregate would hide this, which supports ADR-007.
4. **Random baselines are essential and informative.**
   - Many methods are no better than random importance on realistic models (Hooker 2019 [V]).
   - Explanations can be model-independent (Adebayo 2018 [V]) or fooled by input shifts (Kindermans 2019 [V]).
5. **Mask shape leaks information** (Rong 2022 [V]). For image pixel removal, the removed pattern can carry class signal independent of the pixel values.
6. **Ground truth is available only in constructed settings** (Zhou 2022 [V]; InterpBench / MIB [P]). Realistic evaluation must therefore rely on interventions plus controls, not on known answers.
7. **Activation patching and ablation choices** (corruption method, metric, noising vs denoising) change localisation results (Zhang & Nanda 2024 [V]; Heimersheim & Nanda 2024 [V]).

## Sources

### Faithfulness definitions and perturbation metrics

**DeYoung et al. 2020, "ERASER: A Benchmark to Evaluate Rationalized NLP Models", ACL. [V]**
- **Problem:** do rationales influence predictions (faithfulness), beyond agreeing with humans?
- **Models/data:** BERT-style NLP models on 7 rationale datasets.
- **Definition:** comprehensiveness = m(x)ⱼ − m(x∖r)ⱼ and sufficiency = m(x)ⱼ − m(r)ⱼ, in predicted-class probability. AOPC averages them over the top 1/5/10/20/50% token bins.
- **Perturbation:** token deletion.
- **Control:** one random-score ordering.
- **Finding:** provides the metrics.
- **Limitation:** deletion is off-distribution; there is one random reference, not a null distribution.
- **Relevance:** BeyondNN's `comprehensiveness`/`sufficiency` follow this orientation.
- **Adopt:** the percentage-based k grid (1/5/10/20%) as the pre-registered k policy.

**Samek et al. 2017, "Evaluating the Visualization of What a Deep Neural Network Has Learned", IEEE TNNLS (arXiv 2015). [V]**
- **Problem:** objective heatmap evaluation.
- **Models/data:** ImageNet, SUN397, and Places CNNs.
- **Definition:** region perturbation (MoRF) and AOPC.
- **Perturbation:** random-value patches.
- **Control:** random ordering.
- **Finding:** LRP beat sensitivity and deconvolution under their protocol.
- **Limitation:** conclusions depend on the perturbation.
- **Relevance:** removal curves.
- **Adopt:** curves over the declared k grid.

**Petsiuk et al. 2018, "RISE", BMVC (arXiv). [V]**
- **Definition:** deletion and insertion metrics (area under the class-probability curve as pixels are removed or inserted).
- **Perturbation:** zero (deletion) or blur (insertion).
- **Relevance:** BeyondNN's removal and retention curves.
- **Adopt:** report both curve directions; they can disagree.

**Hooker et al. 2019, "A Benchmark for Interpretability Methods in Deep Neural Networks" (ROAR), NeurIPS. [V]**
- **Problem:** removal without retraining confounds distribution shift with importance.
- **Models/data:** ResNet-50 on ImageNet (plus other image datasets).
- **Perturbation:** remove the top-t% pixels, retrain, measure accuracy.
- **Control:** random assignment of importance.
- **Finding:** many popular methods are no better than random; only some ensembles (SmoothGrad-Squared, VarGrad) beat random.
- **Limitation:** expensive, and it evaluates a retrained model, not the original.
- **Relevance:** random controls are necessary.
- **Adopt:** matched random controls. Retraining is **not adopted** (out of scope); the confound is recorded instead.

**Rong et al. 2022, "A Consistent and Efficient Evaluation Strategy for Attribution Methods" (ROAD), ICML. [V]**
- **Finding:** information leaks through the *shape* of removed pixels, so different evaluation strategies rank methods inconsistently.
- **Remedy:** noisy linear imputation (Remove And Debias), about 99% cheaper than ROAR.
- **Relevance:** pixel-level removal on images inherits this leakage.
- **Adopt:** record it as a limitation for the CNN experiment. Implementing ROAD imputation is **not adopted**: it is a replacement strategy the caller could supply.

**Blücher et al. 2024, "Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks", TMLR. [V]**
- **Finding:** occlusion strategies (from mean to diffusion inpainting) change rankings, and MIF/LIF (most/least-influential-first) orderings depend inversely on occlusion reliability.
- **Proposals:** a symmetric measure (SRG) and a realism score (R-OMS).
- **Relevance:** directly motivates RQ4 (replacement sensitivity) and reporting removal *and* retention.
- **Adopt:** the replacement-strategy comparison.

**Yeh et al. 2019, "On the (In)fidelity and Sensitivity for Explanations", NeurIPS. [V]**
- **Definitions:** infidelity = expected squared error between the explanation·perturbation and the output change, under a perturbation distribution; sensitivity = explanation change under small input perturbations.
- **Relevance:** random-perturbation evaluation of input maps (implemented in Captum).
- **Adopt:** not adopted in Phase 5.5; it is not claim-level, and Captum already implements it.

**Bhatt et al. 2020, "Evaluating and Aggregating Feature-based Model Explanations", IJCAI. [V]**
- **Definitions:** faithfulness (correlation between attribution sums and output change over random subsets), sensitivity, complexity. It also aggregates explanations.
- **Relevance:** "faithfulness correlation" is a single number per explanation.
- **Adopt:** not adopted. The subset-sampling idea resembles matched controls, but BeyondNN keeps components separate (ADR-007).

### Reliability of explanations and of evaluation

**Adebayo et al. 2018, "Sanity Checks for Saliency Maps", NeurIPS. [V]**
- **Tests:** model-parameter and data randomisation.
- **Finding:** some saliency methods are independent of the model and data (edge-detector-like).
- **Relevance:** a method can look plausible and be model-independent.
- **Adopt:** a random-ranking "method" as an explicit control in every experiment. Model randomisation is recorded as future work.

**Kindermans et al. 2019, "The (Un)reliability of Saliency Methods", in *Explainable AI* (LNCS; arXiv 2017). [V]**
- **Finding:** a constant input shift that does not change the model's function changes many methods' attributions (input invariance).
- **Relevance:** baselines and reference points matter; this is the motivation for `stability`.
- **Adopt:** baseline sensitivity is measured.

**Tomsett et al. 2020, "Sanity Checks for Saliency Metrics", AAAI. [V]**
- **Finding:** saliency *metrics* are statistically unreliable and inconsistent, both across metrics and for individual maps.
- **Relevance:** the metrics BeyondNN computes may disagree, and per-sample results are noisy.
- **Adopt:** report distributions and disagreement, never a single metric.

**Hedström et al. 2023, "The Meta-Evaluation Problem in Explainable AI" (MetaQuantus), TMLR [V]; Hedström et al. 2023, "Quantus", JMLR [P].**
- **Finding:** competing quality estimators give conflicting method rankings; it proposes resilience/reactivity meta-evaluation.
- **Relevance:** supports keeping metrics separate.
- **Adopt:** not adopted (Quantus has heavy dependencies and covers input maps only).

**Krishna et al. 2022, "The Disagreement Problem in Explainable Machine Learning", TMLR. [V]**
- **Models/data:** 6 explanation methods, 6 models, 4 real datasets.
- **Measures:** top-k feature agreement, rank agreement, sign agreement, and others.
- **Finding:** methods frequently disagree, and practitioners resolve this ad hoc.
- **Relevance:** RQ6 (does attribution disagreement predict faithfulness disagreement?).
- **Adopt:** the top-k Jaccard and rank correlation measures (already in BeyondNN).

**Hase et al. 2021, "The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations", NeurIPS. [V]**
- **Finding:** feature removal creates OOD counterfactuals that misalign explanations. It compares removal methods and proposes search-based explanations.
- **Adopt:** a data-derived (resample) replacement alongside zero/mean, with OOD limitations recorded.

**Zhou et al. 2022, "Do Feature Attribution Methods Correctly Attribute Features?", AAAI. [V]**
- **Method:** modify datasets so that ground-truth attribution is known.
- **Finding:** saliency, rationales, and attention show deficiencies against ground truth.
- **Relevance:** realistic data has no ground truth, so Phase 5.5 cannot score "correctness", only intervention consistency.
- **Adopt:** the distinction between "consistent with interventions" and "correct" is stated in the report.

**Jain & Wallace 2019, "Attention is not Explanation", NAACL. [P]**
- **Finding:** attention weights often do not correlate with gradient or leave-one-out importance, and alternative attention distributions give the same predictions.
- **Relevance:** the transformer experiment must not use attention weights as explanations.
- **Adopt:** attention weights are **not** used.

### Baselines and integrated gradients

**Sundararajan et al. 2017, "Axiomatic Attribution for Deep Networks" (IG), ICML. [P]**
- **Content:** completeness and implementation invariance; the baseline is a required choice.
- **Adopt:** IG with declared baselines (already in Phase 3).

**Sturmfels et al. 2020, "Visualizing the Impact of Feature Attribution Baselines", Distill. [P]**
- **Finding:** IG results depend strongly on the baseline (black, blurred, uniform, Gaussian, training distribution), and no baseline is universally right.
- **Adopt:** a data-derived IG baseline where meaningful (training mean) and a zero baseline, reported separately.

### Ablation and activation patching (internal components)

**Zhang & Nanda 2024, "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods", ICLR. [V]**
- **Finding:** the corruption method and the metric (for example logit difference vs probability) change localisation conclusions. The paper gives best-practice recommendations.
- **Adopt:** logit-difference (margin) targets; replacement strategy varied and reported.

**Heimersheim & Nanda 2024, "How to use and interpret activation patching" (arXiv). [V]**
- **Pitfalls:** noising vs denoising (removal vs retention), metric choice, and interpretation of circuits.
- **Adopt:** both removal (noising) and retention (denoising) directions.

**Li & Janson 2024, "Optimal Ablation for Interpretability" (arXiv). [V]**
- **Finding:** zero, mean, and resample ablation each have theoretical and practical problems. It proposes optimal ablation (the constant that minimises the loss increase).
- **Adopt:** the problem is acknowledged; optimal ablation itself is **not adopted** (it needs training data and an optimisation per component).

**Miller et al. 2024, "Transformer Circuit Faithfulness Metrics are not Robust", CoLM. [V]**
- **Finding:** circuit faithfulness scores are highly sensitive to seemingly insignificant ablation choices.
- **Adopt:** RQ4 and RQ5 test exactly this sensitivity.

**Wang et al. 2023, "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small", ICLR. [P]**
- **Definitions:** faithfulness, completeness, and minimality of circuits under mean ablation.
- **Relevance:** "mean over a reference distribution" as a data-derived replacement.
- **Adopt:** the training-mean replacement.

**Chan et al. 2022, "Causal Scrubbing" (Redwood Research / Alignment Forum). [P]**
- **Content:** hypothesis testing by resampling activations from inputs the hypothesis says are equivalent.
- **Relevance:** resample ablation as a data-derived replacement.
- **Adopt:** the "resample from a random training example" replacement.

### Ground-truth benchmarks

**Gupta et al. 2024, "InterpBench" [P]; Mueller et al. 2025, "MIB: A Mechanistic Interpretability Benchmark" [P]; Lindner et al. 2023, "Tracr" [P].**
- **Content:** models with known circuits (compiled or trained to match), and a benchmark of circuit and causal-variable localisation.
- **Relevance:** ground truth exists only there.
- **Adopt:** out of scope for 5.5 (which uses realistic models without ground truth); candidates for later.

### Implementations

**Captum 0.9 `FeatureAblation` / `Occlusion` [P, API not re-verified]; Captum `infidelity` / `sensitivity_max` [V, inspected in Phase 5].**
- **Content:** feature ablation replaces each feature group with a baseline and reports the output change.
- **Relevance:** this is an *intervention* measurement, not a gradient attribution.
- **Adopt:** single-unit ablation effects are computed with BeyondNN's own intervention engine, so they stay INTERVENTIONAL rather than being mislabelled ATTRIBUTED. The epistemic question is recorded in the API review.
