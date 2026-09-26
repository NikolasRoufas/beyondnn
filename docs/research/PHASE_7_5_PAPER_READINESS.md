# Phase 7.5 Paper Readiness (NeurIPS and ACL, assessed separately)

- **Status:** an assessment of the evidence, not a paper and not an acceptance prediction.
- **No scores:** there are no readiness scores or grades.
- **Gap labels:** each gap is marked **must-fix** (a claim the paper would need cannot be made honestly without it), **desirable** (strengthens the paper), or **acceptable** (a limitation that can be stated plainly).
- **Sources:** `docs/PHASE_7_5_REPORT.md`, `docs/PHASE_7_5_EXTERNAL_VALIDATION.md`, `results/hypotheses75.json`, and the Phase-7 ledger.

## 1. What the evidence currently supports (both venues)

1. **Structural guarantees:**
   - attribution-only evidence never supports a causal claim (280/280 in Phase 7; 1,960/1,960 IG selection claims on InterpBench);
   - a known-wrong attribution selection is contradicted when intervention evidence is supplied (1,571/1,571).
2. **Configuration dependence is real and large, and the audit surfaces it without a score:**
   - on InterpBench, zero ablation supports 13.0% [12.3, 13.7] of known-unnecessary nodes, where the benchmark's own interchange semantics support 0%;
   - on the central models, nearly every PRIMARY-supported IG sample is reversed by a declared alternative (CH7 held at all 3 qualifying sites);
   - configuration-level disagreements appear on 57–94% of IG samples per site.
3. **Implementation correctness:**
   - BeyondNN's intervention and audit pipeline reproduces an independent TransformerLens computation on 9,782 of 9,782 clear InterpBench instances;
   - persisted audits re-derive identically;
   - 24/24 audit mutations are killed.
4. **Pre-registered, held-out evaluation with a frozen policy and one documented deviation.**

## 2. NeurIPS (framework / methodology track)

| gap | label | why |
|---|---|---|
| G-N1: external validation is partly circular. The PRIMARY configuration equals the ground-truth-defining intervention on clear instances (External Validation §2), so EH1/EH2 cannot show that the audit *discovers* correct mechanisms | **must-fix** | the headline "does the audit separate correct from incorrect mechanisms?" needs ground truth independent of the audited intervention: e.g. HL-only labels without LL filtering, circuits whose necessity is established by a different operation, or planted mechanisms with a known distribution shift |
| G-N2: the concept policy has no demonstrated external true positive. The known-used Tracr variable is rejected on dev and held-out (use effect 0.06 < 0.1) | **must-fix** if concept validation is a headline contribution; acceptable if concepts are presented as conservative-by-design with this failure stated | a validation procedure that rejects every known positive tested cannot be claimed to validate concepts |
| G-N3: one machine, single runs, CPU-only; models ≤ 110M parameters | acceptable (stated) | the claims are about audit logic, not scale |
| G-N4: no comparison with an alternative auditing or robustness approach (e.g. multiverse summaries, ROAR-style retraining, CIF confidence sequences) | desirable | reviewers will ask what the audit adds over reporting the raw configuration grid |
| G-N5: statistical validity. Wilson / bootstrap intervals over samples only; no uncertainty over seeds, checkpoints or datasets | acceptable (stated); desirable to add seed variation for at least one model | intervals describe inputs like these, nothing broader |
| G-N6: the count null is over-restrictive on small multi-head models (EH4, 98.6%) | acceptable (stated; it is an ALTERNATIVE here) | shows why roles matter; must not be presented as a property of controls in general |
| G-N7: the central results are only on the Phase-5.5 models plus one BERT-base | desirable | wider model families would strengthen generality |

## 3. ACL (interpretability / NLP track)

| gap | label | why |
|---|---|---|
| G-A1: NLP scope is SST-2 (BERT-tiny, BERT-base) and e-SNLI (BERT-base SNLI), 40 held-out samples each | desirable | an ACL paper would want ≥ 1 more task type (e.g. QA or NER with span rationales) |
| G-A2: human-rationale comparison. Plausibility and faithfulness are reported separately (N1, N2) | acceptable as designed; see the NLP results in the report | the claim can only be "agreement with annotator 1 and audited necessity differ in X way", never "humans are ground truth" |
| G-A3: token-replacement OOD. Measured with nativeness labels (N3). The PRIMARY [MASK] replacement is itself a distribution shift for a fine-tuned classifier | acceptable (stated) | no replacement is in-distribution for token removal; the audit reports the dependence instead of hiding it |
| G-A4: annotation artefacts. The e-SNLI model's hypothesis-only behaviour (N4) is measured, not controlled | desirable | a faithfulness claim about a model that solves NLI from the hypothesis alone concerns the shortcut, not NLI |
| G-A5: G-N1 (circular external validation) applies equally | **must-fix** for any claim that the audit identifies correct explanations | |
| G-A6: G-N2 applies if concepts are claimed | as NeurIPS | |

## 4. Must-fix before a framework paper, summarised

1. **(G-N1 / G-A5):** an external validation whose ground truth is not the PRIMARY intervention, or else the paper's claim must be narrowed to "the audit reproduces a known mechanism's intervention results and exposes configuration dependence", without "separates correct from incorrect mechanisms".
2. **(G-N2):** either a concept known-positive that the frozen concept policy validates on a *new* held-out case (no tuning), or concept validation is not a headline claim.

Everything else is desirable or acceptable when stated plainly.
