# Trace 3B: a future boundary note (not a design)

- **Status:** placeholder only, written in Phase 7 because the request asked for it.
- **Nothing is designed here.** Nothing was trained, downloaded, measured, or changed in the code for Trace 3B. This page records only what BeyondNN would *have to establish* before any work on a ~3B-parameter model could begin. It defines no API, architecture, experiment or schedule.

## Why it is out of scope now

- Every BeyondNN result so far is on small models: the hand-built toys, a 2-layer MLP, a small CNN, and BERT-tiny (4.4M parameters).
- **Scale.** Phase-7 measurements show that evidence volume scales with passes × records. A single BERT-tiny faithfulness test with 50 controls produces a ~0.9 MB trace, and the full Phase-7 central experiment (~20k test traces) did not fit the 6 GB of free disk on the development machine. A 3B model multiplies both the per-pass cost and the site sizes.
- **Unsolved at small scale:**
  - no uncertainty intervals or sequences;
  - no trained featurisers (DAS);
  - no external ground-truth suite;
  - the covariance-null over-matching (P6-7).

## Preconditions that would have to hold first (not commitments)

1. **Storage.** A trace storage format or retention policy that keeps re-derivation possible without storing every control pass in full (for example, content-addressed tensor sidecars or sampled retention with declared limitations). It needs its own ADR.
2. **Execution.** Remote or streamed execution: the record and provenance model has to work when the model runs elsewhere. NNsight/NDIF-style deferred execution is the relevant precedent.
3. **Uncertainty.** Sample efficiency: finite-sample claims with uncertainty statements, so fewer controls and samples suffice (CIF / Méloux et al. are the relevant literature).
4. **Audit cost.** Audit cost linear in the evidence, with streaming ingestion. The Phase-7 audit holds every record in memory.
5. **External validation.** External validation of the audit on a benchmark with known mechanisms (for example Tracr or InterpBench) *before* any large-model claim.

## What must not happen

- No Trace 3B paper and no Trace 3B experiments until these preconditions have their own plans, pre-registrations and gates.
- No claim that Phase-7 results transfer to large models.
