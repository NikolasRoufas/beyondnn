# Independent validation (Stage 3): protocol, benchmark audit and results

- **Date:** 2026-11-03.
- **Scope:** reference mechanisms that are *not* defined by the intervention being audited, in trained networks where possible.
- **The four questions kept apart:**
  - (a) software agreement;
  - (b) evidence admissibility;
  - (c) correct identification of the reference mechanism;
  - (d) population-level validity.

## 1. Benchmark and literature audit (before any new experiment)

| candidate | trained network? | reference truth | independent of the audited intervention? | decision |
|---|---|---|---|---|
| Tracr compiled programs (TD, Phase 7.75) | no (compiled) | program text + compiled weights | yes | **existing evidence**; tiny, one-hot |
| InterpBench SIIT models, Phase-7.5 labels | yes (SIIT-trained) | specification **filtered by the trained model's own resample ablation** | **no** (circular) | implementation-consistency check only |
| **InterpBench SIIT models, specification labels** (this stage, S3-IBSPEC) | **yes** | circuit membership + interchange on the *compiled high-level program* (never on the trained model) | **yes for the label**. Partial for the training: SIIT trains the network with interchange losses toward that specification | **run (artifact-only)** |
| MIB circuit-localisation track [Mueller et al. 2025] | yes | reference circuits for GPT-2-class models, partly derived from interventions on the same models | partly | NOT JUSTIFIED now: GPU-scale; its labels share the patching family |
| Planted structural decoys by function-preserving surgery on trained MLPs (duplicate a unit with zero outgoing weight; split a unit in two) | yes | exact, from the weights | yes | **NO-GO without compute:** see below |
| Backdoor / trigger planted in training data | yes | data-generating process | the label is about data, not about the model's computation; verifying use needs an input intervention of the audited family | NOT JUSTIFIED |

**Why the planted-surgery design was not run:**
- A zero-outgoing-weight copy has an *exactly* zero effect under every replacement, so it repeats the Tracr decoy result with no new information. IG also gives it zero attribution, so it is never selected.
- A split unit tests only the documented limitation that single-set operational necessity is not structural use (manuscript §VII). It does not test the audit.
- Neither design would change a manuscript claim, so compute was not spent on it.

## 2. S3-IBSPEC: protocol

- **Experiment ID:** S3-IBSPEC.
- **Script:** `experiments/tnnls_revision/stage3/interpbench_specification_labels.py` (artifact-only, standard library).
- **Research question:** on 18 SIIT-trained transformers, does the audit's PRIMARY standing agree with reference labels that use no intervention on the trained network? Where it disagrees, why?
- **Hypothesis:** stated after the Phase 7.5 results had been seen, so **this is not a blind test.** The 18 "ambiguous" instances were known; this re-analysis re-labels them.
- **Reference labels:**
  - **SPEC-INPUT** (per node and input): *necessary* iff the node is in the benchmark circuit *and* interchanging the corresponding node of the compiled high-level program changes that program's output.
  - **SPEC-STRUCT:** node-level circuit membership.
- **Recoverability:** both labels are recoverable exactly from the committed rows (verified: every Phase-7.5 "ambiguous" instance is a non-circuit node).
- **Data:** 18 held-out cases, 40 inputs each; 9,800 node instances (245 nodes).
- **Audit:** the committed Phase-7.5 standings (PRIMARY = resample, i.e. interchange with the next input; no controls) and the per-configuration outcomes.
- **Statistics:**
  - exact counts, with unclustered Wilson intervals (descriptive);
  - instances are nested in nodes and in cases.

## 3. S3-IBSPEC: results

Source: `experiments/tnnls_revision/stage3/results/interpbench_specification_labels.json`.

| configuration (as if it were primary) | SPEC-INPUT necessary: supported | SPEC-INPUT not necessary: supported |
|---|---|---|
| resample (PRIMARY) | **942 / 942** [0.996, 1] | **18 / 8,858** (0.20%) [0.13%, 0.32%] |
| mean (ALTERNATIVE) | 645 / 942 (68.5%) | 121 / 8,858 (1.4%) |
| zero (STRESS_TEST) | 659 / 942 (70.0%) | 1,155 / 8,858 (13.0%) |

**PRIMARY standings on the 8,858 not-necessary instances:**

| standing | count |
|---|---|
| CONTRADICTED | 8,831 |
| SUPPORTED | 18 |
| INCONCLUSIVE | 9 |

**The 18 disagreements:**
- All are **non-circuit nodes**, 17 of them in a single model (case 21).
- In each, the trained network's own resample ablation flips its prediction, and an independent TransformerLens resample of the trained model agrees (that is why Phase 7.5 called them ambiguous).
- These are therefore instances where **the trained network does not implement its specification exactly**, and the audit reports the network's behaviour, not the specification's.
- Under the mean replacement 5 of the 18 remain supported; under zero, 5.

**By circuit membership (SPEC-STRUCT):** PRIMARY supported 942 of 2,520 in-circuit instances (63 nodes) and 18 of 7,280 out-of-circuit instances (182 nodes). An in-circuit node is necessary only on inputs where the program's interchange changes the output (1,578 in-circuit instances have SPEC-INPUT "not necessary").

## 4. Interpretation (layers kept apart)

- **(a) Software agreement:** the PRIMARY test reproduces the benchmark's interchange semantics. It shares them, so agreement on 942/942 is expected and is not validation.
- **(b) Admissibility:** the attribution-only audits stay UNSUPPORTED on all 1,960 IG selections (Phase 7.5; recalculated in Stage 1).
- **(c) Mechanism identification against an independent specification:**
  - The PRIMARY standing agrees with SPEC-INPUT on 9,782 of 9,800 instances.
  - Every disagreement is a case where the trained model uses a node its specification excludes.
  - Neither the audit nor the specification is "the truth" there. The audit's conclusion is about the trained network under the declared intervention.
  - This is the first evidence in this project on *trained* networks with labels that no intervention on the trained network defines. It is still limited to small SIIT-trained transformers whose training objective used interchange interventions.
- **(d) Population validity:** not addressed. The cases are a fixed benchmark set, and the intervals are unclustered.

## 5. What remains

- **No trained network with an independently specified mechanism outside the interchange family.** SIIT training uses interchange interventions, so the label is independent of the audited *evaluation*, but not of the *intervention family* that shaped the network.
- **No mechanism evidence for the central models** (MLP, CNN, BERT).
- A stronger test would use a trained model whose mechanism is fixed by construction but not by interchange training (for example, provably constrained architectures). We did not identify one that avoids the triviality in §1, and no such experiment was run.
