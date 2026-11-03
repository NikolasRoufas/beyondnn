# Baseline capability matrix (Stage 3, E3.1)

- **Date:** 2026-11-03.
- **Sources:** each tool's primary paper and its current documentation or release notes. The versions checked on PyPI on 2026-11-03 are Captum 0.9.0, Quantus 0.6.0, nnsight 0.8.2, pyvene 0.1.8 and TransformerLens 4.3.0. The repository's own ecosystem audit (`docs/research/ECOSYSTEM_AUDIT.md`, September 2026) is the starting point.
- **What a cell means:** what the tool provides **as a built-in, documented capability**. Anything a user can build on top of a general library is marked "user code", not "no".
- **No ranking:** the tools solve different problems. The matrix exists to position BeyondNN, not to rank them.

**Legend:**
- **✓:** built-in and documented.
- **partial:** some of it is built in.
- **user code:** possible, but the user must write it.
- **—:** outside the tool's scope.

| capability | Captum [Kokhlikyan 2020] | Quantus [Hedström 2023] | causal scrubbing [Chan 2022] | NNsight [Fiotto-Kaufman 2025] | pyvene [Wu 2024] | TransformerLens | CIF [Asiaee 2026] | Certified Circuits [Anani 2026] | BeyondNN 0.1.0 |
|---|---|---|---|---|---|---|---|---|---|
| measurement generation (attribution / activations / interventions) | ✓ attribution; layer/neuron methods | uses external explainers | ✓ resampling ablations (research code) | ✓ activations and interventions, deferred/remote | ✓ configurable and trainable interventions | ✓ hooks, patching utilities | consumes interventional metrics | wraps a circuit-discovery method | partial: native gradient/IG and interventions; Captum adapter; not a performance engine |
| explicit, typed claim object (relation, subject, target, scope) | — | — | partial: a hypothesis as a correspondence graph | — | — | — | partial: a claim as an estimand with a threshold | — (circuit-inclusion decisions) | ✓ |
| evidence-type admissibility (e.g. attribution cannot decide a causal relation) | — | — | n/a (only interventional evidence) | — | — | — | n/a (only interventional evidence) | — | ✓ enforced by rule |
| checkpoint / sample provenance on every record | user code | user code | user code | user code | partial: intervention configs serializable | user code | user code | user code | ✓ (state digest, input hash, versions) |
| refusal to combine mismatched checkpoint / sample / scope | user code | user code | user code | user code | user code | user code | user code | user code | ✓ (excluded and reported) |
| eligibility / estimand matching (e.g. content tokens vs all tokens; instance vs population) | — | — | partial: the hypothesis fixes the variables | — | — | — | ✓ the estimand is explicit | — | ✓ |
| mandatory matched controls as a precondition of support | — | random baselines as metrics | — | — | — | — | — | — | ✓ (count- / magnitude-matched; unattainable → inconclusive) |
| declared alternative configurations with roles | — | — | — | — | — | — | — | partial: stability over data subsamples | ✓ PRIMARY / ALTERNATIVE / STRESS_TEST |
| counterevidence retained per sample | — | per-sample scores returned | aggregate (fraction of loss recovered) | — | — | — | — | abstains on unstable components | ✓ counterexamples and findings |
| deterministic re-audit of saved evidence | — | — | — | — | — | — | — | — | ✓ `verify_report` |
| statistical inference / certification | — | — | — | — | — | — | ✓ anytime-valid confidence sequences | ✓ stability certificate | partial: Wilson / bootstrap helpers only; no population inference |
| global score | metric values | metric scores | scalar (loss recovered) | — | interchange accuracy (in its analyses) | — | certified bounds | certified circuit | none, by design |

## Reading the matrix

1. **Measurement libraries** (Captum, NNsight, pyvene, TransformerLens) produce the evidence. BeyondNN depends on, or overlaps with, them for that part and claims no advantage there.
2. **Quantus** and the benchmarks (MIB, InterpBench, OpenXAI) score *methods*. BeyondNN audits *a particular claim*. It is therefore complementary to them, not a better metric.
3. **The closest conceptual precedents** are causal scrubbing (an explicit hypothesis tested by declared interventions) and CIF (the claim as an estimand, with statistical certification). Neither records heterogeneous evidence types with admissibility rules, provenance refusal or declared alternative roles. BeyondNN, in turn, has none of CIF's inferential guarantees.
4. **Everything in the BeyondNN column marked ✓ can also be written as user code** on top of a measurement library with careful logging. Whether the representation adds value beyond that is an empirical question. Stage 3 S3-CMP (`COMPARATIVE_EVALUATION.md`) tests it on the same measurements.

## Sources

- Captum: arXiv:2009.07896; captum.ai documentation (attribution and `captum.metrics`: infidelity, sensitivity).
- Quantus: JMLR 24(34), 2023; quantus documentation (metric categories).
- Causal scrubbing: Chan et al., Alignment Forum, 2022.
- NNsight / NDIF: ICLR 2025.
- pyvene: NAACL 2024 (System Demonstrations).
- TransformerLens: GitHub `TransformerLensOrg/TransformerLens`.
- CIF: Asiaee, UAI 2026, PMLR 337 ([proceedings page](https://proceedings.mlr.press/v337/asiaee26d.html)).
- Certified Circuits: [arXiv:2602.22968](https://arxiv.org/abs/2602.22968).
- Position papers framing the claim/evidence problem, which supply no implementation: Joshi et al. ([arXiv:2602.16698](https://arxiv.org/abs/2602.16698), ICML 2026); Lin & Liu ([arXiv:2605.08012](https://arxiv.org/abs/2605.08012)); Lan et al. ([ACL 2026](https://aclanthology.org/2026.acl-long.159/)).
