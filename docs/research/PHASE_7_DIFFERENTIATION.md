# Phase 7: Differentiation (factual comparison)

- **Source:** `PHASE_7_LITERATURE.md` (verification markers there).
- **Written before** any Phase-7 design or implementation.
- **Style:** comparisons are statements of what each system represents or does. No ranking is implied; nothing here says one system is "better" than another.

## 1. What overlaps (and is therefore not a contribution)

| Idea | Prior work that already has it | Consequence |
|---|---|---|
| Interpretability claims must be matched to the kind of evidence they rest on (correlational vs interventional vs counterfactual) | Joshi et al. 2026 (Pearl hierarchy for interpretability claims); Jacovi & Goldberg 2020; Doshi-Velez & Kim 2017 | Not claimable as novel. BeyondNN operationalises it; the idea itself is theirs. |
| Plausibility is not faithfulness; an attribution is not a causal effect | Jacovi & Goldberg 2020; Adebayo et al. 2018; ROAR, ERASER | Not novel. |
| Probing accuracy is not use | Hewitt & Liang 2019; Elazar et al. 2021 (amnesic probing); Belinkov 2022; CausalGym 2024 (probes vs causal efficacy) | Not novel. CausalGym evaluates the same question at larger scale with trained featurisers. |
| Hypothesis → intervention → behaviour test | Causal scrubbing 2022; causal abstraction (Geiger et al. 2025); interchange interventions / DAS | Not novel as a method. |
| Faithfulness metrics disagree; results depend on replacement / baseline | ROAD, Tomsett 2020, MetaQuantus 2023, Saliency Cards 2023 (hyperparameter dependence) | Not novel as an observation. |
| Interpretability estimates vary; report stability | Méloux et al. 2025/2026; CIF 2026 (anytime-valid confidence sequences) | Not novel. BeyondNN has **less** here (no CIs or confidence sequences). |
| Claim-level provenance, contradiction transparency, auditability | AAR 2026 (deep-research-agent reports) | The vocabulary exists in another domain (text claims about sources). |
| Documented limitations per method or model | Model Cards, Explainability Fact Sheets, Saliency Cards | Not novel as an idea. |
| Ground truth to validate interpretability tooling | Tracr, InterpBench, OpenXAI, XAI-Bench | Not novel. BeyondNN's ground truths are small and hand-made. |

## 2. What BeyondNN does that none of the reviewed systems does (as implemented, not as an idea)

Each item was checked against the matrix in `PHASE_7_LITERATURE.md` §2.

1. **Claims are typed, immutable records with content-derived identity.**
   - A claim has relation, subject (site / units / feature), target, estimand scope and source. Its standing is *not stored*: it is re-derived from test results under an explicit, registry-checked policy.
   - Causal scrubbing has explicit hypotheses but no record system. AAR has claim records, but for text.
2. **Evidence status is a closed vocabulary enforced by the data model.** The seven statuses are OBSERVED / MEASURED / ATTRIBUTED / INTERVENTIONAL / ESTIMATED_CAUSAL / VALIDATED_CONCEPT / GENERATED.
   - A causal relation cannot be decided by non-causal evidence.
   - GENERATED content cannot be cited as evidence.
   - Finite-sample evidence cannot decide a population claim (ADR-013).
   - Joshi et al. describe such a hierarchy; no reviewed tool *enforces* it.
3. **Provenance binding with refusal.** Evidence carries a model-state fingerprint, declared model and exact sample identity. Composition *refuses* mixed checkpoints, samples, targets and scopes instead of merging them.
   - No reviewed toolkit (Quantus, OpenXAI, Captum, TransformerLens, NNsight, pyvene) refuses composition across checkpoints.
4. **Re-derivation.** Decisive results are recomputed from their cited raw records (attribution tensors, causal effects, activations, controls) when they are composed. A forged or edited result is rejected.
   - No reviewed metric suite re-derives stored results.
5. **Mandatory controls in the protocol itself:**
   - count- and magnitude-matched random unit sets (Phase 5 / 5.5);
   - random directions, neurons and label permutations for concepts (Phase 6).

   Quantus and ROAR have random baselines as metrics, not as preconditions of a claim test.
6. **Counterexamples are kept as identified samples, not averaged away:**
   - `counterexample` ProtocolResults (Phase 5);
   - FP/FN sample identities for concepts (Phase 6).
7. **No global score (ADR-007).** Every reviewed benchmark or suite produces scalar scores or leaderboards; BeyondNN deliberately does not.

## 3. What Phase 7 would add, and how much of it is new

Phase 7 proposes an **audit over the records above**. It would:
- inventory the claims;
- apply structural overclaim checks;
- surface contradictions, assumption sensitivity (replacement / threshold / k / null / protocol), missing evidence and missing controls;
- flag provenance, identity and scope mismatches;
- apply counterexample caps;
- present dataset-level distributions and coverage;
- re-derive everything from saved traces.

An honest assessment of each part:

- **Structural overclaim checks** (ATTRIBUTED → causal, probe → use, GENERATED → validated, instance → population) are the Joshi / Jacovi principle applied to records. They are **mechanical given BeyondNN's schema**. Their value is enforcement and completeness, not insight. A reviewer can fairly say they are "type checking".
- **Sensitivity and contradiction surfacing** over heterogeneous evidence (the same claim tested under several replacements, thresholds, k or nulls) has no direct precedent as a *per-claim audit output* among the reviewed systems. Metric-disagreement studies (Tomsett, MetaQuantus, ROAD) report it as aggregate statistics about *methods*, not as findings attached to individual claims.
- **Persistence and re-derivation of an audit** from saved evidence without re-running the model has no precedent among the reviewed systems.
- **Weak spots**, stated plainly:
  - **No uncertainty quantification** over estimates (CIF and Méloux are ahead).
  - **Small models only** (BERT-tiny is the largest).
  - **No DAS / interchange training** (CausalGym, RAVEL and MIB are ahead).
  - **No external ground-truth suites** (Tracr, InterpBench).
  - **The audit can only audit what was recorded.** It cannot detect a favourable choice that was made *before* recording, for example a result selected from runs that were never saved. Only a declared invariance that was never tested is visible.

## 4. Is the differentiation strong enough to proceed?

- **Yes, with a narrow claim:** "an implemented, verifiable audit layer that classifies interpretability claims from typed, provenance-bound, re-derivable evidence records without a global score". That is not "a new principle of interpretability evaluation".
- **The overlap with Joshi et al. 2026 is serious at the level of framing.** A paper must present BeyondNN as an *operationalisation* of that line of argument, not as a competing theory.
- **No reviewed system implements this layer,** so a NO-GO on novelty grounds is not warranted. Whether it is a *research* contribution rather than a *tooling* contribution depends on the Phase-7 experiments. The key question is whether the audit reveals, on realistic models, something that single-configuration reporting hides. That is what the central experiment (plan §29) is designed to test, and it can come out negative.
