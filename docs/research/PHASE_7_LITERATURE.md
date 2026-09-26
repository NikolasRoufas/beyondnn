# Phase 7 Literature Review: Auditing Interpretability Evidence

- **Status:** written **before** any Phase-7 design or code (request §3).
- **Markers:**
  - **[V]:** the source's existence, venue and main claim were verified on the web on 2026-09-26.
  - **[P]:** prior knowledge, not re-verified.
  - **[V-5.5] / [V-6]:** verified in the Phase-5.5 or Phase-6 reviews (`PHASE_5_5_LITERATURE.md`, `PHASE_6_LITERATURE.md`), which hold the per-source details. They are summarised here only for the audit question.
- **Question:** which systems or papers already do something close to Claim → Test → Evidence → Provenance → Controls → Limitations → Contradictions → Audit, and what exactly does BeyondNN add?

## 0. Summary

1. **The conceptual thesis is not new.** Several papers argue that interpretability claims must be matched to the kind of evidence they rest on:
   - Doshi-Velez & Kim 2017 [V]: evaluation must fit the claim;
   - Jacovi & Goldberg 2020 [V]: faithfulness must be defined and evaluated, not assumed;
   - Joshi, Mueller, Klindt, Brendel, Reizinger & Sridhar 2026, "Causality is Key for Interpretability Claims to Generalise" [V]: map claims onto Pearl's observational / interventional / counterfactual levels and match evidence to the level claimed.

   A BeyondNN paper cannot claim that "claims must be matched to evidence" as its contribution.
2. **Existing evaluation frameworks are metric suites or benchmarks, not claim-audit systems.** They evaluate *methods* (explainers, circuit finders, featurisers) with scores and leaderboards. None represents an individual interpretability *claim* as a structured object with an evidence status, provenance, scope, controls, contradictions, and a derived verdict:
   - Quantus [V], OpenXAI [V], XAI-Bench [V];
   - ERASER, ROAR, ROAD [V-5.5];
   - MIB [V], InterpBench [V], Tracr [V], RAVEL [V], CausalGym [V], AxBench and SAEBench [V-6].
3. **Documentation frameworks describe methods or models in prose:** Model Cards [V], Explainability Fact Sheets [V], Saliency Cards [V]. They are templates for humans; they do not bind to execution records and are not re-derived.
4. **Causal scrubbing** (Chan et al. 2022 [V]) and **causal abstraction** (Geiger et al., JMLR 2025 [V]) are the closest *methodological* precedents: an explicit hypothesis, a declared intervention (resampling), and a test of whether behaviour is preserved.
   - They are single-method frameworks, not multi-method evidence records.
   - Causal scrubbing's *hypothesis → test* structure is conceptually close to BeyondNN's Claim → Spec → Result.
5. **Claim-level audit with provenance and contradiction transparency** exists in a *different domain*: AAR, "From Fluent to Verifiable: Claim-Level Auditability for Deep Research Agents", Rasheed et al. 2026 [V]. It measures provenance coverage, soundness, contradiction transparency, and audit effort for text claims written by research agents, not for claims about model internals. It shows that "claim-level auditability" as a vocabulary is not unique to BeyondNN.
6. **Statistical-validity work shows that interpretability numbers are unstable:**
   - Méloux, Portet & Peyrard 2025/2026 [V]: causal-mediation scores have high intrinsic variance;
   - "Certified Interventional Fidelity" 2026 [V]: anytime-valid confidence sequences for interventional estimands;
   - Tomsett 2020 [V-5.5], MetaQuantus 2023 [V-5.5]: metric reliability.

   BeyondNN records controls and distributions but does **not** provide confidence sequences. This is a gap relative to CIF.
7. **Tooling libraries** (Captum, TransformerLens, NNsight, pyvene, SAELens [V-6 / V]) provide measurement and intervention *mechanics*. None records epistemic status, claims or verdicts.

**Consequence for Phase 7:** the defensible contribution is *an implemented, verifiable claim-audit layer over heterogeneous interpretability evidence*. That means:
- typed epistemic status;
- estimand scope;
- provenance binding and refusal;
- mandatory controls;
- re-derivation from raw records;
- explicit contradiction, sensitivity, and missing-evidence findings.

It is not the idea that claims should match evidence.

## 1. Sources

Each entry covers: purpose; what is evaluated; explicit claims?; evidence types distinguished?; provenance?; interventions?; controls?; counterexamples?; contradictions?; limitations?; re-derivation?; audit/report abstraction?; overlap; differentiation.

### 1.1 Foundations of interpretability evaluation

**Doshi-Velez & Kim 2017, "Towards a Rigorous Science of Interpretable Machine Learning" (arXiv 1702.08608). [V]**
- **Purpose and scope:** a taxonomy of evaluation (application-, human-, and functionally-grounded) and a call to match evaluation to claims. Conceptual only.
- **What it has:** explicit claims as data, no; evidence types, informally; provenance, controls, contradictions and re-derivation, no.
- **Overlap:** the motivation.
- **BeyondNN differs** in implementing the matching as typed records, with enforced rules (for example, a causal relation needs causal evidence).

**Jacovi & Goldberg 2020, "Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness?", ACL. [V]**
- **Purpose and scope:** defines faithfulness, argues against conflating plausibility with faithfulness, and proposes a graded view. Conceptual.
- **What it has:** explicit claims, no; plausibility vs faithfulness separated, yes (conceptually).
- **Overlap:** the "plausible ≠ faithful" principle.
- **BeyondNN differs** in encoding it structurally: ATTRIBUTED ≠ INTERVENTIONAL; GENERATED is never evidence.

**Joshi, Mueller, Klindt, Brendel, Reizinger & Sridhar 2026, "Causality is Key for Interpretability Claims to Generalise" (arXiv 2602.16698). [V]**
- **Purpose:** uses Pearl's hierarchy (observational, interventional, counterfactual) plus causal representation learning to decide which claims each method or evaluation supports, and gives practitioners a diagnostic framework. Conceptual; no software.
- **What it has:** explicit claims, yes (conceptually); evidence types, yes (by causal level); provenance, no; interventions, discussed; controls, contradictions and re-derivation, no; no audit object.
- **Overlap:** **substantial at the level of the thesis.** BeyondNN's split of ATTRIBUTED / INTERVENTIONAL / ESTIMATED_CAUSAL, and ENCODES vs use, is an operationalisation of the same hierarchy.
- **BeyondNN differs** in making the hierarchy *enforced by the data model*:
  - claim tests refuse non-causal evidence for causal relations;
  - estimand scope prevents instance → population leaps;
  - composition refuses mismatched provenance.

  BeyondNN does not implement the counterfactual level or causal representation learning.

**Lipton 2018, "The Mythos of Model Interpretability", CACM [P]; Rudin 2019, "Stop explaining black box models…", Nature MI [P].**
- Context: interpretability claims are often under-specified.

### 1.2 Faithfulness and attribution evaluation

**ERASER; ROAR; ROAD; Samek 2017; RISE; sanity checks (Adebayo 2018); infidelity and sensitivity (Yeh 2019); faithfulness correlation (Bhatt 2020); Tomsett 2020; MetaQuantus. [V-5.5]**
- **Purpose:** metrics or benchmarks for attribution faithfulness.
- **What they evaluate:** explanation methods (maps and rationales), by scores.
- **What they have:**
  - explicit claims, no (the implicit claim is "method M is faithful");
  - evidence types, no;
  - provenance, no;
  - interventions, yes (removal and perturbation, in ROAR, ROAD and ERASER);
  - controls, sometimes (random baselines in ROAR and sanity checks);
  - counterexamples, per-sample scores at most;
  - contradictions, metric disagreement studied (Tomsett, MetaQuantus) but not represented;
  - re-derivation, audit and report objects, no.
- **Overlap:** BeyondNN's comprehensiveness, sufficiency and curves come from here (Phase 5).
- **BeyondNN differs** in running them as *claim tests*, with declared criteria, controls and scope, and keeping the protocol disagreements (Phase 5.5 H3).

**Quantus (Hedström et al. 2023, JMLR) [V]; MetaQuantus 2023 [V-5.5].**
- **Purpose and scope:** a toolkit of 35+ explanation-evaluation metrics in 6 categories, for image, time-series and tabular data; MetaQuantus meta-evaluates the metrics.
- **What it has:** explicit claims, no; evidence types, no (metric categories); provenance, no; interventions, perturbation-based; controls, randomisation metrics; contradictions, meta-evaluation of metric reliability; re-derivation, no; audit, no (scores).
- **Overlap:** faithfulness metrics.
- **BeyondNN differs** in having no scalar summaries, internal-unit tests, claim/verdict records, provenance, and refusal.

**OpenXAI (Agarwal et al. 2022, NeurIPS D&B). [V]**
- **Purpose and scope:** 22 metrics (faithfulness, stability, fairness), synthetic data with ground truth, and public leaderboards. Tabular first.
- **What it has:** explicit claims, no; evidence types, no; provenance, partly (a benchmark protocol); controls, ground truth on synthetic data; contradictions, no; re-derivation, no; audit, leaderboards.
- **Overlap:** the synthetic ground-truth idea (BeyondNN Phases 5 and 6).
- **BeyondNN differs** in auditing claims about one model's computation, not ranking explainers.

**XAI-Bench (Liu et al. 2021, NeurIPS D&B) [V]; ExSum (Zhou, Ribeiro & Shah 2022, NAACL) [V].**
- **XAI-Bench:** synthetic data with ground-truth Shapley values.
- **ExSum:** turns local explanations into global rules, with coverage, validity and sharpness metrics. It is the closest "summary of local evidence with quantified coverage", but has no provenance, interventions or claims-as-records.
- **BeyondNN differs** in presenting dataset-level summaries as evidence distributions, keeping counterexamples, with no summary score.

### 1.3 Concepts and probing

**TCAV; concept bottlenecks (CBM, CEM, post-hoc CBM); Network Dissection; ACE; control tasks; MDL probing; amnesic probing / INLP; "probing the probing paradigm"; Huang et al. 2023. [V-6]**
- These evaluate concepts or probes by accuracy, TCAV score, IoU or selectivity, and some use random-concept controls.
- They have no claim objects and no provenance.
- Their contradictions are discussed (probe ≠ use) but not represented.
- BeyondNN's ENCODES vs use split (Phase 6) is the operational form.

### 1.4 Mechanistic interpretability evaluation

**Causal scrubbing (Chan et al. 2022, Alignment Forum / Redwood Research). [V]**
- **Purpose:** test an interpretability hypothesis (a correspondence between a model and a high-level graph) with behaviour-preserving resampling ablations.
- **What it has:**
  - explicit claims, **yes** (the hypothesis is explicit);
  - evidence types, interventional;
  - provenance, no;
  - interventions, yes (resampling);
  - controls, the hypothesis-specific resampling acts as its own null;
  - counterexamples, no;
  - contradictions, no;
  - re-derivation, no;
  - audit, a scalar "fraction of loss recovered".
- **Overlap:** **closest to Claim → Test.**
- **BeyondNN differs** in being multi-protocol and multi-evidence-type, with typed statuses, scope, refusal, and no single scalar.

**Causal abstraction (Geiger et al., JMLR 26, 2025) [V]; DAS [V-6]; MIB (Mueller et al. 2025, ICML) [V]; RAVEL (Huang et al. 2024, ACL) [V]; CausalGym (Arora, Jurafsky & Potts 2024, ACL) [V].**
- **Purpose:**
  - causal abstraction unifies patching, mediation, scrubbing and SAEs as abstraction tests;
  - MIB and RAVEL are benchmarks (circuit localisation, causal-variable localisation, disentanglement), with leaderboards;
  - CausalGym benchmarks interpretability methods' *causal efficacy* on SyntaxGym-derived linguistic tasks (Pythia 14M–6.9B), and finds that DAS outperforms probing.
- **What they have:**
  - explicit claims, as hypotheses or benchmark tasks;
  - evidence types, interventional (interchange interventions);
  - provenance, benchmark protocol;
  - controls, task baselines;
  - contradictions, no;
  - audit, scores.
- **Overlap:**
  - CausalGym's "probes vs causal efficacy" is *the same scientific question* as BeyondNN's ENCODES vs use, at larger scale and with trained featurisers;
  - MIB's causal-variable track evaluates featurisers by intervention.
- **BeyondNN differs** in being a record/audit layer, not a benchmark. It does **not** implement DAS or interchange training.

**Tracr (Lindner et al. 2023, NeurIPS) [V]; InterpBench (Gupta et al. 2024, NeurIPS D&B) [V].**
- **What they are:** ground-truth transformers (compiled, or trained with strict IIT) for evaluating mechanistic-interpretability methods.
- **Overlap:** ground truth for validation, which BeyondNN builds by hand at small scale.
- **Gap for BeyondNN:** it has not been evaluated on Tracr or InterpBench models. That is external validation Phase 7.5 could add.

**Makelov et al. 2024 (subspace illusion); Méloux et al. 2025/2026 (MI as statistical estimation) [V]; Certified Interventional Fidelity 2026 (arXiv 2607.08349) [V].**
- **Méloux et al.:** causal-mediation scores are high-variance, and aggregation is fragile, so stability metrics should be reported.
- **CIF:** turns interventional metrics into explicit causal estimands with anytime-valid confidence sequences.
- **Overlap:** BeyondNN also writes causal quantities as estimands (ADR-013) and records finite-sample scope.
- **Gap:** BeyondNN has *no confidence sequences or intervals*. It uses control fractions, Monte-Carlo p and sign-flip p.

### 1.5 Documentation and audit frameworks

**Model Cards (Mitchell et al. 2019, FAT\*) [V]; Explainability Fact Sheets (Sokol & Flach 2020, FAT\*) [V]; Saliency Cards (Boggust et al. 2023, FAccT) [V].**
- **Purpose:** structured *prose* documentation:
  - Model Cards: model evaluation across conditions;
  - Fact Sheets: explainers along functional, operational, usability, safety and validation dimensions;
  - Saliency Cards: saliency-method attributes such as hyperparameter dependence and sensitivity.
- **What they have:** explicit claims, informal; evidence types, no; provenance, no execution binding; limitations, **yes** (the prose's central feature); re-derivation, no; audit, human-read documents.
- **Overlap:** the idea of first-class limitations.
- **BeyondNN differs** in making limitations machine-readable records attached to specific evidence, and bound to execution provenance.

**Casper et al. 2024, "Black-Box Access is Insufficient for Rigorous AI Audits", FAccT. [V]**
- **Purpose:** argues that white-box and outside-the-box access are needed for rigorous audits. This is policy.
- **Overlap:** motivation for internals-based audits.
- **BeyondNN is not** a governance audit and does not claim regulatory adequacy.

**AAR (Rasheed et al. 2026, arXiv 2602.13855). [V]**
- **Purpose:** claim-level auditability of deep-research-agent *reports*, measured by provenance coverage, provenance soundness, contradiction transparency, and audit effort.
- **Overlap:** the vocabulary of claim-level provenance and contradiction transparency.
- **Differentiation:** its domain is text claims about sources, not claims about neural computation. It has no interventions or epistemic statuses of model evidence.

**Merry, Riddle & Warren 2025, "Explanation Beyond Intuition: A Testable Criterion for Inherent Explainability" (arXiv 2512.17316). [V]**
- **Purpose:** a graph decomposition of models with "annotations" as hypothesis-evidence structures, and verification by intervening on subgraphs, applied to a clinical Cox model. Conceptual and formal.
- **Overlap:** hypothesis-evidence pairs, verified by intervention.
- **Differentiation:** a criterion for inherent explainability of (mostly interpretable) models; no general PyTorch evidence layer, no statuses, and no dataset-level audit.

### 1.6 Tools (measurement and intervention mechanics)

| Library | What it provides | Claims, statuses, provenance, audit? |
|---|---|---|
| Captum [V-6] | attribution, TCAV, LayerIG | none |
| TransformerLens [V] | a hooked GPT-style model library | none |
| NNsight / NDIF (Fiotto-Kaufman et al.) [V] | deferred and remote intervention graphs | none |
| pyvene [V-6] | configurable, trainable interventions | none |
| SAELens [V-6] | SAE training and loading | none |
| Quantus / OpenXAI | metrics and leaderboards | scores only |

BeyondNN integrates Captum through an adapter. It does not replace these libraries (ADR-005).

### 1.7 Scientific claim verification (text)

**SciFact (Wadden et al. 2020, EMNLP). [V]**
- Verifies natural-language scientific claims against abstracts (SUPPORTS / REFUTES with rationales).
- **Relevance:** the SUPPORTS / CONTRADICTS vocabulary and rationale requirement.
- **Not overlap:** its domain is text, not model internals.

## 2. Comparison matrix (audit-relevant properties)

| | explicit claim objects | evidence-type distinction | provenance binding | interventions | controls | counterexamples kept | contradictions represented | re-derivation | audit/report object | scalar score |
|---|---|---|---|---|---|---|---|---|---|---|
| Quantus / MetaQuantus | no | metric categories | no | perturbation | randomisation | no | meta-eval (not per claim) | no | no | yes |
| OpenXAI | no | no | benchmark protocol | perturbation | synthetic GT | no | no | no | leaderboard | yes |
| ERASER / ROAR / ROAD | no | no | no | removal | random (ROAR) | no | no | no | no | yes |
| Causal scrubbing | **yes** (hypothesis) | interventional | no | resampling | self-null | no | no | no | no | yes (loss recovered) |
| MIB / RAVEL / CausalGym | task-level | interventional | benchmark | interchange | baselines | no | no | no | leaderboard | yes |
| Tracr / InterpBench | ground truth | — | — | — | — | — | — | — | — | — |
| Model / Fact / Saliency Cards | prose | prose | no | no | no | no | no | no | document | no |
| Joshi et al. 2026 | conceptual | causal levels | no | discussed | no | no | no | no | no | no |
| AAR (text agents) | yes (text) | source-based | yes (text) | no | no | no | yes (text) | no | yes | metrics |
| CIF | estimands | interventional | no | yes | CIs | no | no | no | no | no (CIs) |
| **BeyondNN (Phases 1–6)** | **yes (records)** | **yes (7 statuses)** | **yes (fingerprints, sample ids)** | **yes** | **mandatory for concepts** | **yes** | **per assessment (MIXED)** | **yes (composition)** | **WHY view (instance); no dataset audit yet** | **no (ADR-007)** |

## 3. What this implies for Phase 7

- **Build what no source has:** an audit over *records*, not over prose or scores. It should:
  - inventory claims;
  - classify each claim's standing from re-derived assessments;
  - surface contradictions, assumption sensitivity, missing evidence and missing controls, overclaims, and provenance and identity mismatches;
  - keep counterexamples;
  - present dataset-level distributions.
- **Do not claim** conceptual novelty for "matching claims to evidence" (Joshi 2026; Jacovi & Goldberg 2020; Doshi-Velez & Kim 2017), or for claim-level auditability as an idea (AAR in another domain).
- **Honest gaps to record** for Phase 7.5:
  - no confidence intervals or sequences (CIF, Méloux);
  - no evaluation on external ground-truth suites (Tracr, InterpBench, MIB);
  - no DAS or trained featurisers (CausalGym, RAVEL);
  - small models only.
