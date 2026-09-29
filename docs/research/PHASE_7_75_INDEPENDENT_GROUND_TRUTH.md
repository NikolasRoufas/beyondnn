# Phase 7.75: Independent Ground Truth for Known-Mechanism Validation

- **Status:** written before the Phase-7.75 plan was frozen, and before any held-out Phase-7.75 result existed.
- **Markers:** [V] verified on the web on 2026-09-29; [V-7.5] verified in `PHASE_7_5_LITERATURE.md`; [P] prior knowledge.

## 1. The problem

On the Phase-7.5 E1 benchmark (InterpBench, 18 held-out cases), clear instances were those where an independent TransformerLens **resample** of the trained model agreed with the Tracr label. The PRIMARY audit configuration was a BeyondNN **resample** of the same node with the same source.

So the 942/942 and 0/8,840 PRIMARY results show **implementation agreement** (question A in the request). They do not show that the audit identifies correct mechanisms (question C).

**What is needed:** a confirmatory test whose ground truth is defined **without** the intervention the audit uses.

## 2. Candidates

| candidate | what the ground truth is | how it was constructed | same operation as the audit? | independent true and false claims? | BeyondNN can run it without reconstructing it wrongly? | feasible here? |
|---|---|---|---|---|---|---|
| **InterpBench (SIIT) circuits** [V-7.5] | nodes of the Tracr circuit, mapped to the trained model | SIIT training aligns LL nodes to HL nodes. Non-circuit nodes are trained to be interchange-invariant (strict IIT loss) | **yes, partly**: the "not necessary" premise is itself an interchange-invariance training objective, and Phase 7.5's ambiguity filter used resample | true: circuit nodes; false: non-circuit nodes, but only as far as training succeeded (18 non-circuit nodes are used in held-out cases 21 and 58) | yes | yes (done in Phase 7.5) |
| **MIB, circuit-localization track** [V] | none as a node list. Methods are scored by faithfulness metrics computed with interchange / ablation interventions | intervention-based metrics on IOI, MCQA, arithmetic, ARC | **yes**: the score is an intervention outcome | no: there is no independent label of individual components | yes | models are GPT-2 small and larger; CPU-heavy |
| **MIB, causal-variable track; RAVEL** [V] | a high-level variable is localised if interchange interventions on the features reproduce the counterfactual behaviour (Cause / Iso / Disentangle scores) | interchange interventions on base/source pairs | **yes** | only relative to interchange success | yes | LLM scale; CPU-heavy |
| **CausalGym** [V] | SyntaxGym minimal pairs; a feature is causal if an interchange intervention changes the syntactic prediction | interchange (DAS and others) | **yes** | no | yes | pythia models; heavy |
| **Tracr-compiled models of programs written for this purpose (TD)** | the RASP program. The compiled model reproduces the program exactly on valid inputs (compiler guarantee; checked per sample) | we write the program: each intermediate variable is **used** (the output function depends on it injectively) or a **decoy** (read by the output function but ignored in its source). Components are mapped to variables through their output weights (which residual dimensions they write) | **no**: the label comes from the program text and the weights; no intervention defines it | yes: used vs decoy components, including decoys that are *perfectly correlated* with the used variable (the plausible-but-wrong mechanism) | yes: BeyondNN runs on the compiled `HookedTransformer` like any model | yes: tiny models, seconds per program |

**Sources:**
- MIB: Mueller et al., ICML 2025 ([arXiv 2504.13151](https://arxiv.org/abs/2504.13151)).
- RAVEL: Huang et al., ACL 2024 ([arXiv 2402.17700](https://arxiv.org/abs/2402.17700)).
- CausalGym: Arora et al., ACL 2024 ([ACL Anthology](https://aclanthology.org/2024.acl-long.785.pdf)).
- Tracr: Lindner et al., NeurIPS 2023 [V-7.5].
- InterpBench: Gupta et al. 2024 [V-7.5].

## 3. Choice

**TD** is the confirmatory benchmark: compiled Tracr programs with decoys, written and split into development and held-out programs **before** any held-out run.

**Why TD's truth is independent:**
1. **Labels are defined by the program source.** A decoy is read by the output `SequenceMap` whose function (`lambda x, y: x`) ignores it. A used variable determines the output injectively.
2. **Components are mapped structurally.** A head or MLP is known-true iff every variable it writes (through non-zero rows of W_O / W_out into labelled residual dimensions) is used or is the output. It is known-false iff every variable it writes is a decoy; empty iff it writes nothing. No held-out component is structurally mixed (`td_structure.py`).
3. **Compiled model checked:** the compiled model's prediction equals the RASP interpreter's on every selected sample (checked, recorded).
4. **No intervention defines any label.** The audit's interventions (resample, mean, zero) are then an **independent** measurement.

**Implementation agreement (question A)** is still measured separately: BeyondNN's PRIMARY resample vs an independent TransformerLens interchange of the same node.

**What TD does not solve:**
- **Idealised models.** TD models are compiled, not trained: small, exact, one-hot / numeric encodings.
  - They test whether the audit separates used from decoy mechanisms when the truth is known.
  - They do not test messy trained representations. InterpBench (partly circular) and the central A–D models (no ground truth) cover the trained side.
- **Mechanism-level truth vs per-sample tests.** A used component is necessary for the *task*. A single counterfactual (one resample source per sample) may leave its value unchanged at the position that matters.
  - Development showed this: 11/40 used-head instances were CONTRADICTED because the source carried the same value.
  - Per-instance and per-component results are therefore reported separately (plan §3). The per-instance CONTRADICTED on a known-true component is labelled a single-counterfactual false negative, not hidden.
- **Categorical decoys are not usable for concept tests.** In Tracr's categorical lookups, removing one coordinate of a decoy's one-hot takes the input off-manifold, so the lookup fails.
  - Development reasoning showed that concept *removal* tests on categorical decoys would therefore report use.
  - Concept tests use **numeric** linear programs, in which the decoy is read with coefficient exactly 0 (plan §4).

## 4. Secondary evidence kept

- **InterpBench E1:** stays as the implementation-consistency check and as trained-model evidence. It is re-audited under ADR-054, with the Phase-7.5 results unchanged on disk.
- **Phase-7.5 Tracr case 39 concept rerun:** reported as **not blind**. Its use effect was known when the concept criterion was chosen.
