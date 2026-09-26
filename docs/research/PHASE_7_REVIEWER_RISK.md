# Phase 7 reviewer risk

- **Purpose:** an honest pre-mortem of how a careful reviewer would read the Phase-7 evidence.
- **Not included:** no acceptance prediction and no scores.

## Three strongest contributions

1. **An implemented, verifiable audit layer over typed interpretability evidence.**
   - **What it does:** it re-derives every result from raw records, excludes out-of-scope and forged evidence, reports standings and findings without a score, and survives a save → restart → load cycle byte-identically.
   - **Evidence:**
     - scenarios A–N 14/14, plus the tampered case;
     - 25/25 code mutations killed;
     - `verify_report` passed on 11 audits;
     - 20,112 realistic results re-derived with 0 integrity failures.
   - **Novelty:** no reviewed system implements this layer (`PHASE_7_DIFFERENTIATION.md`).
2. **Empirical evidence that single-configuration faithfulness claims do not survive declared invariance on trained models.** The IG necessity claim was supported on 84/280 samples under one configuration and on 1/280 when the claim is required to hold across the replacements, k, nulls and thresholds it implicitly generalises over. Random selections are separated from IG by the count of CONTRADICTED samples (E3, E4).
3. **Typed structural overclaim detection and a concept audit that mechanically reproduces a hand analysis.**
   - All attribution-only causal claims are UNSUPPORTED (E2).
   - The concept audit reproduced the Phase-6 conclusions (decodable-but-unused, null and replacement sensitivity) without re-running the models.
   - It shows that the one validated concept's status depends on a declared counterexample cap (E8).

## Three strongest concerns

1. **ASSUMPTION_SENSITIVE saturates, and the headline contrast is partly by construction.**
   - 33–60 IG samples per site are ASSUMPTION_SENSITIVE.
   - The standing does not distinguish "supported in 20 of 21 configurations" from "in 1 of 21". The descriptive supporting-share analysis does, but it is not part of the audit.
   - Declaring more axes, including the hypothetical ×1.5 threshold, mechanically lowers the SUPPORTED count, so "84 → 1" partly measures the plan.
   - The cross-claim necessity/sufficiency check never fired on realistic data, although configuration-level disagreements were common (E6).
   - A reviewer can fairly call the realistic output "everything is sensitive".
   - **Needed:** conditional standings (for example "SUPPORTED given replacement = r1, for all k"), and configuration-level disagreement findings.
2. **The conceptual novelty is thin, and the paper risks reading as a tool paper.**
   - "Match claims to evidence type" is Joshi et al. 2026, Jacovi & Goldberg 2020 and Doshi-Velez & Kim 2017.
   - Faithfulness-metric disagreement is ROAD, MetaQuantus and Saliency Cards.
   - Probes-vs-use is CausalGym.
   - The contribution is the implementation, the verification and one empirical demonstration on small models: an MLP, a small CNN, and BERT-tiny.
   - There is no evaluation on external ground truth (Tracr, InterpBench, MIB) and no trained featurisers.
3. **Statistical and procedural weaknesses:**
   - no confidence intervals or sequences, and single deterministic runs (CIF and Méloux et al. set a higher bar);
   - the author knew the Phase-5.5/6 results when choosing the invariances, the control criterion (D9) and the concept caps;
   - a design defect (one target per claim, D13) was found only by the first realistic run and fixed mid-phase; the run was discarded and repeated, and this is disclosed;
   - the audit can only see recorded evidence: selective recording is invisible.
