# Phase 7.5 Literature Review: methods for the Phase-7 weaknesses

- **Status:** written **before** the Phase-7.5 policy was frozen and before any Phase-7.5 audit result existed.
- **Markers:** [V] means verified on the web on 2026-09-26 in this phase; [V-7] means verified in `PHASE_7_LITERATURE.md`; [P] means prior knowledge, not re-verified.
- **Purpose:** for each Phase-7 weakness (Phase 7 report §40–§44), choose a defensible method, not an exhaustive survey.

## 1. Uncertainty for aggregate audit findings

### 1.1 Proportions over samples (fraction SUPPORTED, fraction showing a replacement reversal, FP/FN rates)

- **Source:** Brown, Cai & DasGupta, "Interval Estimation for a Binomial Proportion", *Statistical Science* 16(2), 2001 [V].
- **Problem it solves:** the Wald interval under-covers badly for small n and for p near 0 or 1. Both are common here (for example 1/280 SUPPORTED, 0/60 CONTRADICTED).
- **Assumptions:** exchangeable Bernoulli outcomes (samples drawn independently by the declared selection procedure).
- **Why appropriate:** Brown et al. recommend Wilson (or Jeffreys) for n ≤ 40, and Wilson remains reasonable at n = 60 to 280. It is closed-form, deterministic, and needs no seed.
- **Limitations:**
  - it describes sampling variability *over inputs drawn like the declared samples*, nothing else;
  - it ignores dependence between claims on the same sample;
  - coverage is nominal only.
- **How BeyondNN uses it:** `audits.uncertainty.wilson(k, n, level=0.95)` for every reported proportion. The report shows the raw count "k of n" next to the interval.

### 1.2 Means, medians and paired differences (intervention effects, selection-vs-control differences, differences between methods on the same samples)

- **Sources:**
  - Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 [P];
  - Koehn, "Statistical Significance Tests for Machine Translation Evaluation", EMNLP 2004, paired bootstrap [V];
  - Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in NLP", ACL 2018 [V].
- **Problem it solves:** the distributions of effects are skewed or heavy-tailed, with no parametric model.
- **Assumptions:** samples are i.i.d. from the declared selection; the resampling unit is the **input sample** (not configurations, not records).
- **Why appropriate:** the standard nonparametric choice in NLP evaluation (Koehn; Dror et al.). *Paired* resampling keeps the within-sample pairing when two methods or two configurations are evaluated on the same inputs.
- **Limitations:**
  - the percentile bootstrap can under-cover for small n and skewed statistics. BCa was considered and not adopted: with n = 40–60 its acceleration estimate is itself noisy, and percentile is simpler to audit;
  - it describes inputs like these, not a population of models, seeds or datasets.
- **How BeyondNN uses it:** `audits.uncertainty.bootstrap(values, statistic, draws=10_000, seed=…, level=0.95)`, and a paired version over per-sample differences. Every call records the seed, the number of draws, the level and the resampling unit in the output.

### 1.3 Monte Carlo error of control comparisons

- **Source:** the finite number of random controls (N = 50 or 200) makes "fraction of controls below" a Monte Carlo estimate. Standard MC error analysis [P]; Phase 5 already reports `mc_p_value`.
- **How BeyondNN uses it:** a Wilson interval on the fraction of controls below the selection, per test. It is reported as MC uncertainty of that one comparison, never as uncertainty about the claim.

### 1.4 Multiple comparisons

- **Sources:** Holm, "A simple sequentially rejective multiple test procedure", *Scand. J. Statistics* 1979 [P]; Dror et al. 2018 [V] on multiplicity in NLP.
- **Decision:**
  - the pre-registered **confirmatory** questions (external RQ1–RQ5 and the held-out hypotheses) are few and fixed; where a test is attached, Holm–Bonferroni is applied across them;
  - all other intervals are **descriptive and unadjusted**, labelled as such, with no claim of significance.
- **Limitation:** descriptive intervals over many cells will include chance extremes. The report says so.

### 1.5 Statistical-validity work on interpretability

- Méloux et al. 2025/2026 (MI as statistical estimation) and CIF 2026 (anytime-valid confidence sequences) [V-7] are the stronger standard.
- **Not adopted:** confidence sequences are valid under optional stopping, but Phase 7.5 has fixed-n designs, where Wilson and bootstrap intervals are adequate. Adopting CIF is left as future work and recorded as a gap.

## 2. Reporting sensitivity without a score

**Sources:**
- Steegen et al., "Increasing Transparency Through a Multiverse Analysis", *Perspectives on Psychological Science* 2016 [V];
- Simonsohn, Simmons & Nelson, "Specification curve analysis", *Nature Human Behaviour* 2020 [V].

**Problem it solves:** Phase-7 ASSUMPTION_SENSITIVE collapsed "supported in 20 of 21 configurations" and "in 1 of 21".

**What is adopted:**
- the multiverse idea of reporting **every** declared configuration's outcome;
- the specification-curve practice of showing the full distribution of outcomes across specifications.

**What is not adopted:** their summary inference (for example a permutation test on the share of significant specifications), because it becomes a de facto robustness score (ADR-007).

**How BeyondNN uses it:** a `SensitivityProfile` per claim and sample, listing:
- every tested configuration (with its declared role);
- the outcome counts, and the identities of supporting and contradicting configurations;
- the axes that flip the outcome, and the minimal assumption changes that reverse it.

Counts and fractions are descriptive only.

**Configuration roles:** the multiverse literature separates "reasonable" from arbitrary specifications. BeyondNN makes this a **declared role** per configuration value:
- PRIMARY: the pre-registered analysis;
- ALTERNATIVE: a reasonable alternative;
- STRESS_TEST: deliberately extreme or out-of-distribution.

The roles are fixed before evaluation.

## 3. Known-mechanism benchmarks

**InterpBench**
- **Source:** Gupta, Arcuschin, Kwa & Garriga-Alonso, NeurIPS 2024 D&B [V].
- **What it is:** 86 released semi-synthetic transformers trained with Strict IIT (SIIT), pinned at HF `cybershiptrooper/InterpBench@a1242a84`, CC-BY-4.0. Each has:
  - a ground-truth circuit (edges at attention-head / MLP granularity);
  - a high-level Tracr model;
  - a correspondence between high- and low-level nodes.
- **Ground truth:** SIIT trains non-circuit nodes to have no effect under resample ablation, and circuit nodes to implement the high-level variables under interchange.
- **Why appropriate:**
  - the mechanism is fixed *independently of BeyondNN*;
  - the models are trained (not hand-set), in PyTorch / TransformerLens, and small enough for CPU;
  - the official `circuits-benchmark` code (commit `220791e4`) supplies inputs, the high-level model and the correspondence.
- **Limitations:**
  - SIIT is approximate: the low-level model can disagree with the high-level one;
  - the ground truth is defined under **resample/interchange** ablation only; zero or mean ablation has no benchmark ground truth;
  - the tasks are algorithmic.

**Tracr**
- **Source:** Lindner et al., NeurIPS 2023 [V-7] (FlyingPumba fork, as used by circuits-benchmark).
- **What it is:** compiled RASP programs with exact weights, where each RASP variable occupies known residual dimensions. The high-level models of InterpBench *are* Tracr models.
- **Why appropriate:** it gives exact, independently known *representational* ground truth (which variable is encoded where) for concept-style claims, which InterpBench's circuit labels do not.
- **Limitation:** compiled weights are unlike trained weights (one-hot residual encodings).

**MIB**
- **Source:** Mueller et al., ICML 2025 [V].
- **What it is:** circuit-localisation and causal-variable-localisation tracks on GPT-2 / Qwen / Gemma / Llama.
- **Not adopted:**
  - MIB scores *methods* against faithfulness-based metrics and counterfactual datasets; it does not provide mechanism labels independent of those metrics;
  - its models (≥ 124M parameters, with many counterfactual passes) exceed the CPU budget for full audits.

  This is recorded as a gap.

**RAVEL (ACL 2024) and CausalGym (ACL 2024)** [V-7]
- These evaluate featurisers and interchange interventions on LMs.
- **Not adopted:** they need DAS-style trained featurisers, which BeyondNN does not implement (no feature creep).

## 4. Human rationales, plausibility vs faithfulness

- **Sources:**
  - DeYoung et al., "ERASER", ACL 2020 [V];
  - Camburu et al., "e-SNLI", NeurIPS 2018 [P];
  - Jacovi & Goldberg, ACL 2020 [V-7];
  - Carton et al., "Evaluating and Characterizing Human Rationales", EMNLP 2020 [V], which finds that human rationales are not necessarily sufficient or comprehensive for models.
- **Adopted:** e-SNLI dev and test with three annotators' highlighted words per pair, raw CSVs from the authors' repository pinned at commit `7b585a3f`.
- **How it is used:**
  - human highlights are a **plausibility reference only**;
  - agreement between an attribution's top-k and the human highlights is reported separately from the comprehensiveness / sufficiency audit of the same tokens;
  - human rationales are never labelled causal ground truth.
- **Limitations:**
  - highlights mark words a human found relevant to the explanation, not what the model uses;
  - three annotators may disagree;
  - the tokenisation mismatch between words and WordPieces is handled by marking every WordPiece of a highlighted word.

## 5. Token perturbation and OOD

- **Sources:**
  - Hase, Xie & Bansal, "The Out-of-Distribution Problem in Explainability…", NeurIPS 2021 [V]: removal produces OOD inputs, and explanations then reflect the model's OOD behaviour;
  - Hooker et al., ROAR, NeurIPS 2019 [P];
  - Kim et al., "Interpretation of NLP models through input marginalization", EMNLP 2020 [P]: counterfactual token replacement drawn from an MLM.
- **Adopted, as a measurement and not a fix:** several token interventions, each characterised by:
  - (a) prediction shift under *random* (non-selected) token replacement;
  - (b) representation shift: the final-layer [CLS] distance to the nearest clean reference [CLS], compared with clean-to-clean nearest-neighbour distances;
  - (c) whether the intervention is model-native: [MASK] was seen in pretraining; deletion yields a natural shorter input; zero or [PAD] embeddings mid-sentence were never seen.
- **Strategies:** zero embedding; [MASK] embedding; [PAD] embedding; [UNK] embedding; deletion (a model-level measurement outside BeyondNN's site-intervention protocol).
- **Not adopted:** MLM-marginalisation replacement. It is feasible, but it is a new perturbation method (feature creep); it is recorded as a gap.

## 6. Leakage and shortcuts

- **Sources:** Gururangan et al., NAACL 2018; Poliak et al., *SEM 2018 [V]. A hypothesis-only model reaches about 67–71% on SNLI, far above chance (33%).
- **Adopted checks:**
  - the NLI model's accuracy with the premise replaced by [MASK] tokens (the hypothesis-only behaviour of the *given* model);
  - SST-2 train/validation exact-duplicate overlap;
  - word-order shuffling;
  - [CLS] / [SEP] ablation;
  - padded vs unpadded equivalence with and without attention masks.

## 7. Decisions recorded for the frozen policy

1. Wilson intervals for proportions, and the percentile (paired) bootstrap with B = 10,000 over input samples for means and differences. Level 95%. Every seed is recorded.
2. Holm–Bonferroni only across the pre-registered confirmatory tests; everything else is descriptive and unadjusted.
3. `SensitivityProfile` with declared PRIMARY / ALTERNATIVE / STRESS_TEST roles. No summary statistic across configurations.
4. External validation on InterpBench (SIIT) circuit claims and on Tracr representational claims, using the official artifacts. MIB, RAVEL and CausalGym are not used, for the reasons given.
5. NLP on BERT-base SST-2 (a moderate held-out model) and BERT-base SNLI with e-SNLI human rationales (a held-out task). Human rationales are a plausibility reference only.
6. Token OOD characterised by prediction shift, representation shift and nativeness. No MLM marginalisation.
