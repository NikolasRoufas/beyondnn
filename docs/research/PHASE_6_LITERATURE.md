# Phase 6 Literature Review: Concepts and Concept Validation

- **Status:** written **before** any Phase-6 design or code (request §5–§6).
- **Markers:**
  - **[V]:** the source's existence, venue and main claim were verified on the web on 2026-09-26, from the publisher page, the proceedings, or arXiv.
  - **[P]:** prior knowledge that was not re-verified this session.
  - Details that are not in an abstract (exact controls, sample counts) are marked [P] even when the paper is [V].
- **Scope:** what each line of work calls a concept, how the concept is represented, discovered, named and tested, which controls it uses, its failure modes, and what BeyondNN should reuse or avoid.

## 0. Summary: what the literature implies for Phase 6

1. **Decodability is not use.** A property can be linearly decodable from a representation that the model does not use for its task: Ravichander et al. 2021 [V]; Elazar et al. 2021 [V]; Belinkov 2022 [V]. *Representation claims and use claims must be separate claims, tested by different protocols.* The Phase-5.5 finding "attribution agreement can be circular" has a direct analogue: "the probe found it, so the model uses it" is circular.
2. **Probe accuracy depends on the probe, not only on the representation.**
   - Flexible probes memorise. Hewitt & Liang 2019 [V] (control tasks, selectivity) and Voita & Titov 2020 [V] (MDL) show that raw accuracy is uninformative without a control.
   - *Every representation test needs a declared null:* random directions, permuted labels, or both, *evaluated on held-out data*.
3. **Interventions on subspaces can create illusions.**
   - Makelov et al. 2024 [V]: patching a subspace can change behaviour through a dormant parallel pathway, even when the subspace is causally disconnected in normal operation.
   - Consequence: *a causal effect of a direction intervention is evidence about that intervention, not proof that the model "uses the concept" in normal operation.* Claims must be scoped to the declared intervention.
4. **The intervention choice matters.**
   - Amnesic probing removes linear information with INLP (Ravfogel 2020; Elazar 2021 [V]). Causal tracing uses noise corruption (Meng 2022 [V]). DAS learns a rotation (Geiger 2024 [V]).
   - These are different interventions and can support different conclusions. This agrees with Phase 5.5 (replacement choice changed up to 49% of outcomes). *The intervention must be declared and recorded; no default.*
5. **Named units and features are hypotheses, and automatic labels are often wrong.**
   - Network Dissection (Bau 2017 [V]) aligns units to labelled concepts, but is limited to concepts in its probe dataset.
   - Automated neuron explanations (Bills et al. 2023 [P]) were found by Huang et al. 2023 [V] to have high error rates and little to no causal efficacy, even for the most confident explanations. *A generated label is GENERATED and at most a proposal.*
6. **The probe dataset shapes the explanation.** Ramaswamy et al. 2023 [V]: different probe datasets give very different concept explanations, and concepts are often harder to learn than the classes they explain. *Validation is scoped to its dataset; VALIDATED never means universal.*
7. **SAE features are useful hypotheses, not ground truth.**
   - Bricken 2023 [V], Huben/Cunningham et al. 2024 [V] and Templeton 2024 [V] report interpretable SAE features.
   - Feature splitting and absorption (Chanin et al. 2025 [V]) mean a seemingly monosemantic latent can fail to fire where it should.
   - Reconstruction error ("dark matter") is substantial and structured (Engels 2024 [V]).
   - SAEs do not consistently beat simple baselines in probing (Kantamneni 2025 [V]) or steering (Wu et al. 2025, AxBench [V]); proxy metrics do not predict downstream usefulness (Karvonen 2025, SAEBench [V]).
   - *An SAE feature enters BeyondNN as a feature basis with provenance and explicit limitations, and is validated like any other feature.*
8. **Concept bottlenecks leak.** Concept representations in CBM-style models encode information beyond the named concepts (Mahinpei 2021 [V]), so "the slot is named C" does not imply "the slot only carries C". This is consistent with ADR-008/009.
9. **Random-concept baselines exist but are easy to skip.**
   - TCAV (Kim 2018 [V]) proposes testing CAVs against CAVs trained on random concept sets [P for the exact test].
   - The Captum TCAV API documents experimental sets but *no* built-in random-concept significance test (Captum docs, checked 2026-09-26 [V]).
   - *In BeyondNN the control criterion is mandatory for validation* (request §16; Phase-5.5 F-21).
10. **Distributed and redundant representations defeat single-unit necessity tests.**
    - Causal mediation (Vig 2020 [V]) found effects that are sparse but also synergistic.
    - DAS (Geiger 2024 [V]) exists because variables are distributed across neurons.
    - *A failed single-feature necessity test is not evidence that the concept is unused*, which is the Phase-5 redundancy lesson again.

## 1. Sources

Each entry covers: what it calls a concept, representation, discovery, naming, testing, correlational vs causal testing, controls, label origin, failure modes, relevance, what to reuse, and what to avoid.

### 1.1 Concept-based interpretability

**Kim et al. 2018, "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)", ICML (PMLR 80). [V]**
- *Concept:* a human-chosen set of example inputs, e.g. "striped" images.
- *Representation:* the CAV, the normal of a linear classifier separating concept examples from random examples at a layer.
- *Discovery:* none; concepts are user-supplied.
- *Naming:* human.
- *Testing:* the TCAV score is the fraction of class inputs whose directional derivative along the CAV is positive. It is gradient-based, *not an intervention*: a sensitivity measure (correlational/attributional).
- *Controls:* CAVs from random concept sets and a statistical test across repeated runs [P for the exact procedure].
- *Failure modes:*
  - the CAV depends on the negative set and on the classifier;
  - a directional derivative does not show that intervening on the direction changes the output;
  - dataset dependence (Ramaswamy 2023).
- *BeyondNN:*
  - **reuse** the linear concept direction fitted on concept vs non-concept examples, and random-direction baselines;
  - **avoid** treating gradient alignment as causal use (it is ATTRIBUTED-like evidence at most) and allowing the random test to be optional.

**Koh et al. 2020, "Concept Bottleneck Models", ICML (PMLR 119). [V]**
- *Concept:* annotated attributes predicted in a bottleneck layer.
- *Representation:* a dedicated neuron per concept, trained with concept supervision. *Naming:* dataset annotations.
- *Testing:* concept accuracy, plus test-time intervention (editing the concept values). Editing is causal *within the designed architecture*.
- *Failure modes:* leakage (Mahinpei 2021 [V]); needs concept annotations.
- *BeyondNN:*
  - relevant for a future Mode B (ADR-008); concept slots still start as PROPOSED;
  - **reuse** the idea that an intervention on the concept representation must change the output;
  - **avoid** equating "the slot is named C" with "the slot means only C".

**Mahinpei et al. 2021, "Promises and Pitfalls of Black-Box Concept Learning Models" (arXiv 2106.13314). [V]**
- *Finding:* learned concept representations encode information beyond the predefined concepts ("leakage"), and natural mitigations do not fully work.
- *BeyondNN:* **avoid** assuming a concept feature is exclusive. Record the limitation `FEATURE_MAY_CARRY_OTHER_INFORMATION`.

**Espinosa Zarlenga et al. 2022, "Concept Embedding Models", NeurIPS. [V]**
- *Concept:* a pair of positive and negative embeddings per concept.
- *Testing:* concept accuracy and intervention efficacy.
- *BeyondNN:* shows that a concept need not be a single scalar unit. **Reuse:** concepts may be represented by directions and subspaces, not only by neurons.

**Yuksekgonul et al. 2023, "Post-hoc Concept Bottleneck Models", ICLR. [V]**
- Concepts are CAV-like directions, or come from multimodal text embeddings, projected post hoc; a residual path recovers accuracy.
- *Failure mode:* the residual can carry the actual decision information.
- *BeyondNN:* **avoid** calling a post-hoc bottleneck explanation "the model's reasoning". Residual or unexplained paths belong in limitations.

**Bau et al. 2017, "Network Dissection: Quantifying Interpretability of Deep Visual Representations", CVPR. [V]**
- *Concept:* a labelled visual concept (object, part, texture, colour) from the Broden dataset [P].
- *Representation:* a single unit, with thresholded activation maps [P].
- *Testing:* IoU between the unit's activation mask and the concept segmentation. Correlational.
- *Controls:* comparison across units and training regimes [P].
- *Failure modes:*
  - limited to concepts present in the probe dataset;
  - a high-IoU unit may be polysemantic;
  - no causal test.
- *BeyondNN:* **reuse** the feature/concept alignment measured on held-out labelled data. **Avoid** naming a unit from alignment alone.

**Ghorbani, Wexler, Zou & Kim 2019, "Towards Automatic Concept-based Explanations" (ACE), NeurIPS. [V]**
- *Discovery:* segments images, clusters segments in activation space, and scores clusters with TCAV. *Naming:* by humans after discovery.
- *Failure modes:* clusters may reflect segmentation artefacts, and naming happens post hoc.
- *BeyondNN:* discovery is *outside* the validation boundary. A discovered cluster or direction is an UNLABELED_FEATURE until someone proposes a label. The search procedure (candidate count, criterion) must be recorded to avoid cherry-picking.

**Chen, Bei & Rudin 2020, "Concept Whitening for Interpretable Image Recognition", Nature Machine Intelligence. [V]**
- *Representation:* whitened latent axes aligned to concepts during training.
- *Testing:* concept purity and axis activation on concept examples.
- *BeyondNN:* a training-time intervention, so out of scope for post-hoc validation. It shows that axis alignment is a design choice, not a discovered fact.

### 1.2 Probing and representation analysis

**Alain & Bengio 2016, "Understanding intermediate layers using linear classifier probes" (arXiv). [P]**
- Linear probes per layer.
- *BeyondNN:* **reuse** linear readouts of a declared site. **Avoid** reading layer-wise probe accuracy as use.

**Hewitt & Liang 2019, "Designing and Interpreting Probes with Control Tasks", EMNLP. [V]**
- *Controls:* control tasks assign random outputs to word types. Selectivity = task accuracy − control accuracy.
- *Finding:* popular probe designs have high control accuracy, i.e. they memorise. Complexity control (small probes) improves selectivity; ordinary regularisation does not.
- *BeyondNN:*
  - **reuse** a label-randomisation control evaluated with the *same probe family on held-out data*;
  - keep the probe family minimal: 1-D projections and linear directions, not MLP probes;
  - record the probe family, dimension, sample counts and seed.

**Voita & Titov 2020, "Information-Theoretic Probing with Minimum Description Length", EMNLP. [V]**
- The measure becomes the description length of labels given representations, which accounts for probe complexity and data size.
- *BeyondNN:* not implemented in v0. The same concern (effort and flexibility) is addressed by fixed, low-capacity probe families plus random-direction and permutation controls. MDL is a candidate for later.

**Belinkov 2022, "Probing Classifiers: Promises, Shortcomings, and Advances", Computational Linguistics 48(1). [V]**
- Survey: the choice of probe, datasets and baselines, correlation vs causation, and the need for intervention-based analyses.
- *BeyondNN:* the rationale for separating ENCODES from causal claims.

**Ravichander, Belinkov & Hovy 2021, "Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?", EACL. [V]**
- *Finding:* models encode properties that are not needed for their task, so probing accuracy does not entail task relevance.
- *BeyondNN:* the direct literature basis for **mandatory ground-truth case B** (decodable but unused).

**Ravfogel et al. 2020, "Null It Out" (INLP), ACL [P]; Elazar, Ravfogel, Jacovi & Goldberg 2021, "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals", TACL 9. [V]**
- *Intervention:* remove linearly decodable information by iterative nullspace projection, then measure the change in *task behaviour*.
- *Finding:* probing results do not license behavioural conclusions; amnesic interventions measure use.
- *Failure modes:*
  - the projection is off-distribution;
  - repeated projections may remove more than the concept;
  - linear removal leaves non-linear information.
- *BeyondNN:*
  - **reuse** projection removal as the direction intervention, with the reference coordinate declared;
  - **avoid** calling a null effect "the concept is not used" when information may be redundant or non-linear (case D; the limitations).

**Tucker, Qian & Levy 2021, "What if This Modified That? Syntactic Interventions with Counterfactual Embeddings", Findings of ACL. [V]**
- Counterfactual embeddings are produced by gradient-editing probe outputs, and the effects on behaviour are measured.
- *BeyondNN:* supports scoping causal claims to the intervention that was performed.

### 1.3 Mechanistic and causal representation work

**Vig et al. 2020, "Investigating Gender Bias in Language Models Using Causal Mediation Analysis", NeurIPS. [V]**
- Direct and indirect effects through neurons and attention heads. The effects are sparse, *synergistic*, and decomposable.
- *BeyondNN:* distributed effects are expected (case C), and a single-unit test can understate them.

**Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (ROME; causal tracing), NeurIPS. [V]**
- Corrupt the input with noise, then restore activations to locate decisive states.
- *Failure mode:* localisation vs editability can disagree (Makelov 2024; Hase 2023 [P]).
- *BeyondNN:* **reuse** the principle that the intervention and its reference must be explicit. Noise corruption is not implemented.

**Geiger, Wu, Potts, Icard & Goodman 2024, "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (DAS), CLeaR (PMLR 236). [V]**
- Learns a rotation in which a high-level causal variable aligns with a subspace; tests it with interchange interventions.
- *BeyondNN:* concepts can be directions, not neurons. DAS itself (a learned rotation plus interchange training) is out of scope for v0.
- **Avoid:** evaluating a learned subspace on the data used to learn it (a leakage risk that Makelov 2024 sharpens).

**Makelov, Lange & Nanda 2024, "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching", ICLR. [V]**
- Subspace interventions can change outputs via dormant parallel pathways, so the "feature" is not what the model uses in normal operation.
- *BeyondNN:* causal concept claims are **scoped to the declared intervention**; the limitation `DIRECTION_INTERVENTION_MAY_ACTIVATE_DORMANT_PATHWAYS` is recorded; and the plan includes an intervention-sensitivity experiment (request §44).

**Zou et al. 2023, "Representation Engineering: A Top-Down Approach to AI Transparency" (arXiv). [V]**
- "Reading vectors" (LAT) from contrasting prompts, then steering by adding vectors.
- *BeyondNN:* **reuse** mean-difference and contrast directions as a *discovery* method. **Avoid** treating steering success as validation of meaning: addition is off-distribution and not implemented in v0.

**Park, Choe & Veitch 2024, "The Linear Representation Hypothesis and the Geometry of Large Language Models", ICML (PMLR 235). [V]**
- Formalises the linear representation of concepts via counterfactual pairs and identifies a "causal inner product".
- *BeyondNN:* the direction geometry is not canonical (the inner product matters). Random-direction controls therefore need a declared distribution (isotropic vs covariance-matched).

### 1.4 Sparse autoencoders and dictionary learning

**Bricken et al. 2023, "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", Transformer Circuits Thread. [V]**
- SAEs on MLP activations yield interpretable features. The validation includes specificity, causal effect on logits, and automated interpretability.
- *Concept:* an SAE latent. *Naming:* human and automated.
- *Failure modes:* feature splitting as the dictionary grows; interpretability judged on top activations.
- *BeyondNN:* an SAE latent is a **feature basis** (encoder rule, decoder direction). Its label is a proposal.

**Huben, Cunningham, Smith, Ewart & Sharkey 2024, "Sparse Autoencoders Find Highly Interpretable Features in Language Models", ICLR. [V]**
- The motivation is superposition and polysemanticity; SAEs recover feature directions. Automated interpretability scoring [P].
- *BeyondNN:* same as above.

**Templeton et al. 2024, "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", Transformer Circuits Thread. [V]**
- SAEs with 1M/4M/34M features; abstract, multilingual features; steering by clamping features.
- *BeyondNN:* scale is out of scope. It shows that SAE causal tests are *clamping* interventions, a different intervention from projection removal, which must be recorded as such.

**Chanin et al. 2025, "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders", NeurIPS. [V]**
- Hierarchical features split; seemingly monosemantic latents fail to fire where they should (absorption); changing SAE size or sparsity does not fix it.
- *BeyondNN:* **counterexample search is mandatory**, in both directions: the feature fires on non-concept inputs, and it fails to fire on concept inputs. Record the limitations `SAE_FEATURE_SPLITTING` / `SAE_FEATURE_ABSORPTION`.

**Engels et al. 2024, "Decomposing The Dark Matter of Sparse Autoencoders" (arXiv 2410.14670). [V]**
- SAE reconstruction error is substantial and partly linearly predictable.
- *BeyondNN:* record the reconstruction information in the SAE adapter when available, together with `SAE_RECONSTRUCTION_ERROR`.

**Kantamneni et al. 2025, "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing", ICML (PMLR 267). [V]**
- SAE probes do not consistently beat baselines under data scarcity or label noise.
- *BeyondNN:* SAE features get no privileged status. They are compared against the same controls.

**Wu et al. 2025, "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders", ICML (PMLR 267). [V]**
- On concept detection and steering, simple baselines (probes, difference-in-means) are competitive with or beat SAEs.
- *BeyondNN:* difference-in-means is a legitimate, cheap direction-discovery baseline, which supports keeping `mean_difference` as the default fitter.

**Karvonen et al. 2025, "SAEBench", ICML (PMLR 267). [V]**
- Eight metrics; proxy metrics do not reliably predict practical performance.
- *BeyondNN:* no single "SAE quality" or "concept" score (ADR-007).

### 1.5 Automated labels and neuron explanations

**Bills et al. 2023, "Language models can explain neurons in language models" (OpenAI). [P]**
- An LLM writes explanations from top-activating examples and scores them by simulation.
- *BeyondNN:* this is the prototype of a **GENERATED** label.

**Huang, Geiger, D'Oosterlinck, Wu & Potts 2023, "Rigorously Assessing Natural Language Explanations of Neurons", BlackboxNLP. [V]**
- *Observational mode:* does the neuron activate on all and only the inputs the explanation picks out? *Interventional mode:* is it a causal mediator?
- *Finding:* even the most confident GPT-4 explanations had high error rates and little to no causal efficacy.
- *BeyondNN:* the structure of Phase 6 follows this: an observational encoding test with counterexamples in both directions, plus an interventional test. A generated label must pass both, like any label. Mandatory case **J**.

### 1.6 Dataset and evaluation factors

**Ramaswamy, Kim, Fong & Russakovsky 2023, "Overlooked Factors in Concept-Based Explanations: Dataset Choice, Concept Learnability, and Human Capability", CVPR. [V]**
- Probe-dataset choice profoundly changes the explanations; concepts are often harder to learn than the target classes.
- *BeyondNN:* validation scope must be recorded, as data, split and population, and never generalised.

### 1.7 Libraries (inspected; none added as a dependency)

| Library | What it offers | BeyondNN decision |
|---|---|---|
| Captum `captum.concept` (TCAV) [V, docs] | CAVs via a trained classifier; sign and magnitude TCAV scores; experimental sets; no documented random-concept significance test | Not a dependency. The TCAV score is attribution-like; BeyondNN's ENCODES test is held-out decodability against controls, and its causal test is an intervention |
| SAELens [V] | SAE training and pretrained SAEs | Optional future adapter source. The adapter takes raw encoder/decoder tensors, so no SAELens import is needed |
| TransformerLens [P] | hooked transformers | Not needed; BeyondNN has its own hooks |
| NNsight [V] | delayed-execution intervention graphs | Not needed |
| pyvene [V] | configurable interventions, including trainable ones | Not needed; DAS-style trainable interventions are out of scope |
| Concept-bottleneck repositories (e.g. `yewsiang/ConceptBottleneck`, `mateoespinosa/cem`) [V] | CBM/CEM training | Mode-B territory (ADR-008); not used |

## 2. Comparison matrix

| Work | Concept = | Representation | Discovery | Naming | Test | Corr./causal | Controls | Label origin |
|---|---|---|---|---|---|---|---|---|
| TCAV | example set | linear direction (CAV) | user | human | directional-derivative sign | correlational (gradient) | random CAVs [P] | human |
| CBM | annotated attribute | dedicated unit | supervised | dataset | concept accuracy + edit | causal within the design | — | human |
| CEM | attribute | ± embeddings | supervised | dataset | accuracy + intervention | causal within the design | — | human |
| Post-hoc CBM | attribute / text | projections | CAV / CLIP | human / text | accuracy | correlational | — | human / generated |
| Network Dissection | labelled visual concept | single unit | alignment search | dataset | IoU | correlational | cross-unit | human |
| ACE | segment cluster | cluster in activation space | automatic | human post hoc | TCAV | correlational | random [P] | human |
| Concept Whitening | annotated concept | whitened axis | training | dataset | purity | correlational | — | human |
| Linear probes | property | readout | supervised | human | accuracy | correlational | often none | human |
| Control tasks | property | readout | supervised | human | selectivity | correlational | random-label control | human |
| MDL probing | property | readout | supervised | human | description length | correlational | complexity-aware | human |
| Amnesic / INLP | property | nullspace | supervised removal | human | behaviour change after removal | **causal** (removal) | random-direction removal [P] | human |
| Causal mediation | variable | neuron / head | enumeration | human | indirect effect | **causal** | — | human |
| ROME / causal tracing | fact | MLP states | corruption + restore | human | restoration effect | **causal** | noise-level variants | human |
| DAS | high-level variable | learned subspace | gradient search | human | interchange accuracy | **causal** (interchange) | — | human |
| Subspace-illusion work | — | subspace | — | — | patching vs pathway analysis | **causal**, with a caveat | — | — |
| RepE | cognitive property | reading vector | contrast prompts | human | steering / reading | correlational + steering | — | human |
| SAEs (Bricken, Huben, Templeton) | latent | encoder / decoder direction | unsupervised | human / automated | top activations, logit effect, clamping | mixed | — | human / **generated** |
| Absorption | — | SAE latent | — | — | first-letter ground truth | correlational | — | — |
| Automated neuron labels | explanation | neuron | top activations | **generated** | simulation score | correlational | — | **generated** |
| Huang 2023 | explanation | neuron | — | generated | observational + interventional | both | — | generated |

## 3. Answers to the review questions

- **Can arbitrary labels be decoded from representations?**
  - With flexible probes and small data, yes: control tasks are learned (Hewitt & Liang).
  - With a fixed 1-D readout (a declared direction or neuron) evaluated on held-out data, a random label is decoded only at chance, up to sampling variability. That variability is exactly what random-direction and permutation controls quantify.
  - A direction *fitted* on the evaluation data will decode anything; hence the train/held-out separation (§30 of the request).
- **Does probe accuracy demonstrate model use?** No (Ravichander 2021; Elazar 2021; Belinkov 2022).
- **What makes a direction meaningful?**
  - Operationally: it is defined without the evaluation data; it decodes the concept on held-out data better than a declared null of random directions and permuted labels; its counterexamples are recorded; and, for use claims, intervening on it under a declared intervention changes a declared target more than intervening on matched random directions.
  - None of this makes it "the" concept.
- **What evidence supports naming a feature?** A proposal needs only a label and a feature. Retaining the name as *validated* requires, under a declared policy: held-out encoding evidence above controls, interventional evidence above controls, and recorded counterexamples. Generated names carry GENERATED provenance throughout.
- **When does a concept deserve causal language?** Only for the declared intervention on the declared data and target, and only when the effect beats matched random-intervention controls. Causal language is always scoped ("removing projection onto v reduced Y under reference R on dataset D").
- **What controls are needed against random directions?**
  - Random directions at the same site and dimensionality, with a declared distribution: isotropic, or matched to the activation covariance. Covariance matching is the stronger, more conservative null.
  - The same intervention operation, and the same data and target.
  - Sign-free scoring for random directions, because a random direction has no preferred sign.
  - Do not overmatch (e.g. matching the concept-label correlation would make the null meaningless).
- **What does an SAE feature establish?** That an SAE with a given checkpoint assigns a sparse code; nothing about monosemanticity, completeness, uniqueness or causal use (Chanin; Engels; Kantamneni; Wu).
- **How do splitting and polysemanticity affect claims?**
  - A split concept makes each latent look weak or partial (case I).
  - Absorption creates false negatives.
  - Polysemanticity creates false positives on non-concept inputs (case H).
  - Counterexample reporting in both directions exposes these; aggregates hide them.
- **What does intervention on a direction mean?**
  - It replaces the activation's coordinate along the (normalised) direction with a reference coordinate: removal, which is necessity-style.
  - Or it keeps only that coordinate and takes the rest from a reference: retention, which is sufficiency-style.
  - The reference (zero, a dataset mean, or a resampled activation) is part of the intervention.
  - Adding a direction (steering) is a different, off-distribution intervention and is not implemented in v0.
- **Representation vs utilisation:** representation means information is present and decodable. Utilisation means the model's declared behaviour depends on that information under a declared intervention. The first never implies the second (case B).

## 4. What Phase 6 adopts, and what it avoids

**Adopt:**
- the claim split: ENCODES (representation) vs DECREASES under a declared feature intervention (use);
- held-out evaluation;
- fixed low-capacity readouts, with the direction fitted on train only;
- random-direction and label-permutation controls, with a mandatory criterion;
- counterexamples in both directions;
- projection removal and retention with a declared reference;
- scope recorded as model, dataset, split and intervention;
- generated labels are GENERATED;
- SAE latents as feature bases with limitations.

**Avoid:**
- TCAV-style gradient scores as use evidence;
- MLP probes;
- training and testing probes on the same data;
- a single concept score;
- automatic promotion of labels;
- treating a null single-feature effect as "unused" without considering redundancy;
- steering-as-validation;
- SAE-specific trust.

**Not in v0 (considered):**
- MDL probing;
- INLP (multi-direction nullspace removal);
- DAS;
- causal tracing with noise;
- CBM/Mode B;
- automated labelling (only an optional record type for externally generated labels).
