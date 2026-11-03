# Revision experiment plan (TNNLS revision, proposed after Stage 1)

- **Status: PROPOSAL FOR REVIEW.** Nothing here is authorised or frozen. Every experiment needs a committed protocol (plan §9 fields) before it runs.
- **Labels:**
  - **ESSENTIAL:** closes a central validity gap.
  - **HIGH VALUE:** materially strengthens the methodological contribution.
  - **OPTIONAL:** useful but not submission-blocking.
  - **NOT JUSTIFIED:** unlikely to strengthen the paper enough to merit compute.
- **Ranking criteria:** methodological contribution, TNNLS relevance, validity of the comparison, need for independent evidence, compute cost, and risk of an inconclusive result.
- **Compute tiers:**
  - T0: artifact only
  - T1: CPU, under 1 h
  - T2: CPU, hours, or a small GPU pilot
  - T3: GPU, a day or more

## Summary table

| ID | stage | experiment | label | tier | needs new execution? |
|---|---|---|---|---|---|
| P0 | – | Repository hygiene: pin Phase 7.75 hashes; obtain and version the manuscript; publish a small evidence bundle (traces + plans + reports) | ESSENTIAL | T0–T1 | bundle needs traces regenerated (T1 for A/B; T2 for C/D) |
| E2.0 | 2 | Attainability and monotonicity checks of every declared control / threshold / role rule, from per-configuration records | HIGH VALUE | T0 | no |
| E2.1 | 2 | 47/52 decomposition from existing per-sample outcomes: separate k (subject change) from fixed-subject axes as far as artifacts allow; share-of-alternatives distribution; R-selection reference rate | ESSENTIAL | T0 | no |
| E2.2 | 2 | Fixed-selection factorial re-audit (replacement × null × threshold at fixed units; k reported separately) on A, B, C, from regenerated evidence | ESSENTIAL | T1–T2 (CPU) | yes (A/B: minutes; C: about 30 min) |
| E2.3 | 2 | Same for D (BERT-base) | OPTIONAL | T2–T3 (about 19 min/sample on CPU; GPU unverified) | yes |
| E3.1 | 3 | Source-verified capability matrix (Captum, Quantus, causal scrubbing, pyvene, nnsight, TransformerLens, CIF, Certified Circuits, MIB/InterpBench) | ESSENTIAL | T0 (literature/docs) | no |
| E3.2 | 3 | Comparative error-detection study: a fair, strong Captum + patching + logging baseline vs BeyondNN on the same evidence, with seeded realistic workflow faults (wrong checkpoint, mixed sample provenance, mismatched selection/eligibility, attribution-as-causal, omitted control, single-configuration reporting) | ESSENTIAL | T1 | yes (small) |
| E3.3 | 3 | Independent validation on a *trained* network with planted / independently specified mechanisms (dev / held-out split; truth not defined by the audited intervention) | ESSENTIAL | T2 (CPU or small GPU) | yes |
| E3.4 | 3 | External trained-model benchmark with independent truth (e.g. MIB circuit tracks, if their truth is not interchange-defined) | OPTIONAL | T2–T3 | yes |
| E4.1 | 4 | Checkpoint replication: retrain A and B with 5 independent seeds; repeat the central comparison on fresh held-out samples | HIGH VALUE | T1–T2 (CPU) | yes |
| E4.2 | 4 | Overhead: matched with/without-audit workflows; separate timing of evidence, audit build, serialize, reload, verify, storage, peak RAM; repeated runs | HIGH VALUE | T1 | yes |
| E4.3 | 4 | Scalability: synthetic evidence stores scaled in samples / records / claims / configurations (model-free) | HIGH VALUE | T1 | yes (cheap) |
| E4.4 | 4 | Reproducibility: save → reload → integrity → provenance → re-audit → WHY on published bundles, plus controlled evidence mutations | HIGH VALUE | T1 | yes (cheap) |
| E4.5 | 4 | BERT-base checkpoint replication (several independent fine-tunes) | OPTIONAL | T3 | yes |
| X1 | – | Scaling to 3B/7B decoder LMs | NOT JUSTIFIED (now) | T3 | – |
| X2 | – | More NLP tasks / annotators for e-SNLI | NOT JUSTIFIED for TNNLS | T2 | – |
| X3 | – | Multi-counterfactual necessity estimand (fixes W9) | OPTIONAL (method change; affects API freeze) | T1–T2 | – |

## Details and rationale

### P0. Hygiene (ESSENTIAL, cheap)

- Add a hash-pin test for `experiments/phase7_75/results/*`, like `tests/test_phase75_immutable.py`.
- Obtain the manuscript LaTeX source and the supplementary appendices, and confirm which file is the original version to preserve. The PDF consistency check is done (`MANUSCRIPT_CONSISTENCY.md`).
- Publish a small evidence bundle so `verify_report` can be repeated by reviewers (W11).

### E2.1. Artifact-only decomposition of 47/52 (ESSENTIAL, T0)

- **Question:** how much of 47/52 survives when subject-changing k alternatives are excluded? How concentrated are reversals?
- **Feasible now:**
  - per-sample role×outcome counts (`outcomes`);
  - per-configuration outcomes for TD and InterpBench;
  - `fd5_central_profiles_per_sample.csv`.
- **Not feasible from artifacts:** central per-configuration outcomes with their axis values. The central files store only counts per role, and `reversal_axes` mixes ALTERNATIVE and STRESS_TEST.
- So E2.1 can bound but **not isolate** the effects. That limit is the justification for E2.2.
- **Reference rate:** apply the same "any alternative reverses" rule to R-selections and to G-selections, using the existing records.
- **Risk:** low. Inconclusive is possible and must be reported.

### E2.2. Fixed-selection factorial (ESSENTIAL, CPU)

- Regenerate evidence for A/input, A/net.1, B/pixels, B/relu2 and C on CPU (deterministic training; pinned HF revision for C).
- Write full per-configuration records.
- Audit with the PRIMARY units held fixed, crossing replacement × null × threshold.
- Report k as a separate *subject-change* analysis.
- **Units, denominators and dependence:** per-sample paired comparisons within a site; per-site reporting; McNemar-type paired tests only where decisive outcomes exist; inconclusive outcomes reported as a third category, never dropped.
- **Held-out:** use the existing held-out samples for comparability with 47/52 (labelled as a re-audit), *and* a fresh held-out draw for any confirmatory statement.
- **Compute:** minutes for A/B; about 30 min for C. No GPU needed.

### E3.1. Capability matrix (ESSENTIAL, T0)

- Verify each cell against primary papers and current documentation (Captum 0.9.0, Quantus 0.6.0, nnsight 0.8.2, pyvene 0.1.8, TransformerLens 4.3.0 as of 2026-11-03).
- Cover the newly identified works: arXiv 2606.00033, 2602.22968, 2510.00845, 2605.08012, CIF.
- Never rank tools; state what each represents.

### E3.2. Comparative error-detection (ESSENTIAL, small)

- **This is the single most important new experiment** (W1).
- **Baseline:** a careful researcher's script. Captum IG, explicit patching with the same replacements, matched random controls, and a JSON log containing checkpoint hash and sample IDs.
- **Do not handicap the baseline.** Give it the logging a competent researcher would add.
- **Faults:** inject realistic faults into evidence and workflow at known rates.
- **Measure per workflow:**
  - detected / missed / false alerts;
  - lines of configuration;
  - runtime.
- **Pre-register:** the fault list and how detection is scored.
- **Honest expected result:**
  - a careful baseline will catch some faults (wrong checkpoint, if it hashes);
  - BeyondNN's advantage should be in composition refusal, eligibility/estimand mismatch and configuration-dependent standing.
- **A null result is possible and must be reported.**

### E3.3. Trained network with planted mechanisms (ESSENTIAL)

- **Example design:** small MLPs/transformers trained on tasks where some input features or intermediate variables are provably irrelevant or relevant, by data construction and verified by retraining-free tests independent of the audited intervention. Candidates:
  - (a) planted-feature tasks where the label depends on a known subset of inputs, plus perfectly correlated decoy inputs that are decorrelated only on held-out probes;
  - (b) SIIT-trained models evaluated on HL-only labels with a non-interchange criterion (already listed as desirable in `PHASE_7_75_PAPER_READINESS.md`).
- **Literature/benchmark novelty check before implementation:** MIB, InterpBench, and planted-mechanism literature.
- **Development and held-out programs must be separate.**
- **Compute:** CPU or a small GPU pilot.
- **Risk:** medium. Truth in trained models is never perfect; the protocol must state what "independent" means.

### E4.1. Checkpoint replication (HIGH VALUE, CPU)

- Five seeds for A and B (CPU, minutes each).
- Repeat the 47/52-type comparison and the R-vs-IG separation per checkpoint.
- Report between-checkpoint spread.
- Intervention seeds are **not** model replications.

### E4.2–E4.4. Overhead, scalability, reproducibility (HIGH VALUE, cheap)

- Mostly model-free.
- Must compare equivalent workflows, and must not attribute model inference time to BeyondNN.

### Not justified now

- **X1:** large-LM scaling would not address W1–W3 and would consume GPU time on a server whose GPU 1 is occupied and whose root filesystem is full.
- **X2:** more NLP tasks or annotators serve an ACL framing, not TNNLS.

## Server dependence

- E2.1, E3.1 and P0 (except the bundle traces) need no compute.
- E2.2, E3.2, E4.1–E4.4 fit on a CPU, either locally or on the server's CPUs.
- Only E2.3, E3.4 and E4.5 would need a GPU.
- As of 2026-11-03 the server is **SERVER COMPUTE BLOCKED** for GPU work (see `SERVER_READINESS.md`).

## Requested decision from Prof. Kanavos

1. Approve, change or reject the ESSENTIAL set: P0, E2.1, E2.2, E3.1, E3.2, E3.3.
2. Decide whether D (BERT-base) re-runs (E2.3) are worth CPU-hours or GPU time.
3. Decide whether to keep TNNLS as the target pending E3.2/E3.3 results, or to consider TAI now (W5).
