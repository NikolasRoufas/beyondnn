# Faithfulness protocols

Each page covers:
- the definition and formal quantity;
- required inputs and perturbation semantics;
- the output and its interpretation;
- what the protocol can and cannot support;
- failure modes and distribution-shift concerns;
- baselines, an example, limitations, and references.

| Protocol | Kind | Decides | Page |
|---|---|---|---|
| `comprehensiveness` v1 | claim test | NECESSARY_FOR, DECREASES | [comprehensiveness.md](comprehensiveness.md) |
| `sufficiency` v1 | claim test | SUFFICIENT_FOR (site-relative) | [sufficiency.md](sufficiency.md) |
| `removal_curve` v1, `retention_curve` v1 | diagnostic | nothing | [curves.md](curves.md) |
| `stability` v1 | diagnostic | nothing | [stability.md](stability.md) |
| `counterexample` v1, `paired_control` v1 | dataset diagnostics | nothing | [counterexample.md](counterexample.md) |
| `method_agreement`, `baseline_sensitivity`, `ig_step_sensitivity` v1 | method diagnostics | nothing | [diagnostics.md](diagnostics.md) |

## Shared rules

- **Perturbations.** Every perturbation is a Phase-2 intervention (ADR-032): a paired CLEAN baseline pass and an INTERVENTION pass, with every pairing refusal.
  - The raw measurement is an INTERVENTIONAL `CausalEffect` about the original input.
  - "Removing" a unit means replacing it with a declared value (zero or an explicit tensor), and that replacement is recorded. Nothing is ever deleted: a network always sees *some* value.
- **Controls** are matched random selections: same site, same size, uniform without replacement (random permutations for curves), drawn from a seeded local generator.
  - The observed statistic is compared with the control distribution as fractions below/tied/above and as `P(a matched random set does at least as well) = (1 + b)/(N + 1)` (Phipson & Smyth 2010).
  - These are descriptive comparisons with a stated null. They are never labelled "significant", and a small p is not a faithfulness verdict.
- **Magnitude-matched controls** (Phase 5.5, ADR-035), `controls(n, seed=, match="magnitude", strata=4)`: each selected unit is replaced by a random unit from the same stratum of perturbation magnitude ‖x_u − b_u‖₂. The null becomes "a random set perturbed by similar amounts". The magnitudes are recorded and re-derived in composition. Not available for curves.
- **Units** (Phase 5.5, ADR-034): by default, indices along the last axis, with every other dimension of size 1. With declared `unit_axes`, a unit is a row-major index into the sub-grid of those axes: pixels `(2, 3)`, channels `(1,)`, token positions `(1,)`. Attribution scores per unit need an explicit `reduce` (`sum`, `abs_sum`, `l2`) whenever a unit spans more than one element.
- **Outcomes** come only from criteria declared before running (claim tests) or per-aspect criteria (diagnostics). Assessments come only from explicit policies.
- **There is no aggregate faithfulness score** (ADR-007). RQ8 is deferred (see the Phase-5 plan §15).

## Literature and ecosystem: why these protocols, and what BeyondNN adds

| Idea | Origin | Existing implementations | BeyondNN decision |
|---|---|---|---|
| Comprehensiveness / sufficiency | DeYoung et al. 2020 (ERASER, ACL): comprehensiveness = m(x)ⱼ − m(x∖r)ⱼ, sufficiency = m(x)ⱼ − m(r)ⱼ, AOPC over the top 1/5/10/20/50% bins; a single random-score reference | ERASER code (NLP rationales); Quantus (input maps) | Implemented natively as **claim tests**, because BeyondNN needs them for *internal* units, which Quantus does not cover. Added: declared replacement, repeated matched controls, verified perturbations, and scoped claims. |
| Deletion / insertion, region perturbation, AOPC | Samek et al. 2017 (IEEE TNNLS); Petsiuk et al. 2018 (RISE, BMVC) | Quantus (pixel flipping, region perturbation), Captum (no curves) | Two curve protocols (removal, retention), not four (§4 of the plan). The full curve is kept, and AOPC is optional with its normalisation recorded. |
| ROAR (remove and retrain) | Hooker et al. 2019 (NeurIPS) | research code | **Not implemented** (retraining is out of scope). The confound it addresses (distribution shift) is recorded as a limitation. |
| ROAD (mask-shape leakage) | Rong et al. 2022 (ICML) | Quantus | Not implemented. It is a known risk for image inputs (report §weaknesses). |
| OOD effects of removal | Hase et al. 2021 (NeurIPS) | — | Recorded (`*_MAY_BE_OOD`), and replacement is explicit. RQ9 shows the replacement choice alone can erase the signal. |
| Sanity checks / randomisation | Adebayo et al. 2018 (NeurIPS); Tomsett et al. 2020 (AAAI) | Quantus (randomisation metrics) | Random-method control (scenario H) and matched random controls. Model-randomisation checks are deferred. |
| Faithfulness correlation, infidelity, sensitivity | Bhatt et al. 2020 (IJCAI); Yeh et al. 2019 (NeurIPS) | Quantus; Captum `infidelity` / `sensitivity_max` | Not re-implemented. A Captum-metrics adapter is a candidate for later; these evaluate input maps under random perturbations, not claims. |
| Circuit faithfulness under ablation choice | Wang et al. 2022 (IOI); Conmy et al. 2023 (ACDC); Miller et al. 2024 (CoLM) | TransformerLens-based research code | This is why every internal result names its replacement and site, and why sufficiency is site-relative. |
| Ground-truth interpretability benchmarks | Lindner et al. 2023 (Tracr); Gupta et al. 2024 (InterpBench); Mueller et al. 2025 (MIB) | benchmark suites | Phase 5 has a small analytic *validation suite*, not a benchmark. |

**Quantus 0.6.0** (checked 2026-09-25) needs opencv, scikit-image, pandas, scikit-learn, and matplotlib, and evaluates input saliency maps. It is not added as a dependency. An adapter would be optional and would not replace claim-level tests.
