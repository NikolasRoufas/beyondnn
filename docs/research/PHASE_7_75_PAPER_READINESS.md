# Phase 7.75 Paper Readiness (NeurIPS and ACL, assessed separately)

- **Status:** an assessment of the evidence; no acceptance prediction and no scores.
- **Claim labels:** see `PAPER_EVIDENCE_LEDGER.md`, "Final paper-claim freeze (Phase 7.75)".
- **Gap labels:** must-fix, desirable, acceptable.

## 1. What changed since Phase 7.5

- **Independent known-mechanism validation (TD):** zero false positives on 400 decoy instances, 160 perfectly correlated decoys and 57 IG-selected decoys.
- **Must-fix G-N1** (circular external validation) is **resolved for compiled programs**.
- **Concept criterion (ADR-055):** 4/4 held-out known positives validated, 0/8 negatives.
- **Must-fix G-N2** (no external concept true positive) is **resolved, with a qualifier.** The held-out programs did not discriminate between absolute and relative thresholds.
- **Special tokens (ADR-053):** content-token claims are expressible.
- **Must-fix G-A7** is **resolved:** D2 gives 6/40, with 3 of D1's 11 supports surviving.
- **Count null (ADR-054):** diagnosed as an unattainable criterion. Its rejections were never evidence.

## 2. NeurIPS

| question | answer |
|---|---|
| Can we now claim independent known-mechanism validation? | **Yes, for compiled Tracr programs with program-defined truth.** Not for trained models. |
| Can we claim the audit separates some correct and incorrect mechanisms? | **Yes, with that scope:** 0/400 decoy instances supported; every used component supported on 28–38 of 40 samples. The per-instance single-counterfactual false-negative rate (22%) must be reported. |
| Can we claim concept *validation* rather than concept *testing*? | **Validation on programs with known use**, under a scale-relative criterion calibrated on constructed models. No validated concept in a trained model has independent truth. |
| Is configuration sensitivity supported across architecture classes? | **Yes:** MLP (2 sites), CNN (2), BERT-tiny, BERT-base. 47 of 52 IG supports reversed by a declared alternative. |
| Are uncertainty and negative results adequately represented? | **Yes for sampling uncertainty** (Wilson / bootstrap, Bonferroni where confirmatory), with failed predictions reported (N1, N3-D). **No uncertainty over seeds or checkpoints.** |
| What would a skeptical reviewer still attack? | (1) Independent truth only on tiny compiled models; (2) the role declarations determine what counts as a reversal; (3) one machine, one seed; (4) no comparison with an alternative auditing approach (multiverse reporting, ROAR, confidence sequences); (5) single-counterfactual necessity tests miss 22% of known-true instances. |

**Remaining gaps:**

| gap | label |
|---|---|
| Independent truth on a *trained* model (e.g. SIIT models evaluated against HL-only labels with a non-interchange audit, or planted-mechanism training) | desirable. The paper can state the compiled-program scope honestly. |
| Comparison with an alternative (multiverse summary, ROAR retraining, CIF confidence sequences) | desirable |
| Seed / checkpoint variation on at least one central model | desirable |
| Multiple resample sources per sample (reduces single-counterfactual false negatives) | desirable (future method; would change evidence cost) |
| Tiny-model and CPU scale | acceptable (stated) |

**Must-fix for NeurIPS:** none remaining, *provided* claims F and G carry their qualifiers exactly as in the ledger.

## 3. ACL

| question | answer |
|---|---|
| Are lexical / content-token claims now separated from special-token effects? | **Yes:** declared eligibility (ADR-053). On BERT-base SST-2, 3 of 11 all-token IG supports survive as content-token supports; D2 finds 6/40, all with alternative reversals. |
| Does the human-rationale experiment support a useful plausibility-vs-faithfulness result? | **Yes, qualified:** IG is more human-aligned than random (N2 held), yet less often necessary than the human highlight (N1 failed; the interval includes 0). One annotator, n = 40. |
| Are OOD perturbation limitations explicit? | **Yes:** the zero-vs-[MASK] ordering reverses between two BERT-base fine-tunes. No universal safe replacement is claimed. |
| Are shortcut artefacts explicit? | **Yes:** empty-premise accuracy 0.456 (chance 1/3). 17/40 e-SNLI samples are empty-premise-predictable, and IG's supports concentrate there (5 of 7; exploratory). |
| Is one sentiment task plus one NLI task enough for the intended ACL framing? | **For a framework / methodology paper with an NLP case study: adequate.** For a paper whose central claim is about NLP explanations: thin. |
| If not, what exact additional task is required? | A span-rationale task on a different task type, e.g. extractive QA (SQuAD-style evidence spans) or a token-classification task with gold rationale spans (e.g. HateXplain rationales). The ERASER-style rationale datasets are the natural source. Not added in this phase (request §39). |

**Remaining gaps:**

| gap | label |
|---|---|
| A third task type with gold rationales | desirable (**must-fix only if** the paper's headline is an NLP-explanations claim) |
| Multiple annotators / inter-annotator-aware plausibility | desirable |
| Control of hypothesis-only shortcuts (a shortcut-controlled NLI subset) | desirable |
| Padding inside BeyondNN protocols | acceptable (probe shows mask sensitivity; stated) |

## 4. Summary

**The framework contribution (Claims A–E) is supported.**
- Claim F (independent mechanism validation) and Claim G (concept validation) are supported **only with their qualifiers**.
- There is no remaining must-fix for a framework paper that states those qualifiers.
- The most important desirable addition is independent truth on a trained model.
