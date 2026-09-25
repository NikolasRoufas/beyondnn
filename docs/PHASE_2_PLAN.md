# Phase 2 Plan: Causal Interventions

- **Status:** plan, written before implementation.
- **Baseline:** Phase 1 commit `34cfa89` (gate: GO WITH EXPLICIT LIMITATIONS). 638 tests green at the start of Phase 2.
- **Central question:** what changes when we deliberately alter an internal model state while holding the rest of the experimental setup fixed?

## Principles (fixed before code)

1. **Measured state ≠ intervention ≠ intervention effect (ADR-017).**
   - Activations recorded during an intervened pass are MEASURED, with provenance `execution_mode = INTERVENTION` and `intervention_id`.
   - Only the *effect* (intervened metric − baseline metric, under a controlled pairing) is INTERVENTIONAL.
2. **An effect is always "of an intervention, on a metric, over a scope".** A causal claim still needs a declared test criterion. One nonzero effect never makes a claim SUPPORTED.
3. **No silent confounding.**
   - A comparison where randomness or model state differs between baseline and intervention is refused, not reported.
   - An intervention that did not actually apply is refused.
4. **No new tracing or persistence system.** Every comparison is one `recording()`: an optional patch-source pass, a CLEAN baseline pass, and an INTERVENTION pass. The intervention spec, per-pass provenance, `CausalEffect`, and claim records all live in that one validated `TraceResult`, and persist with M1.7.
5. **Sufficiency is not implemented.** `SUFFICIENT_FOR` stays unassessable: no policy names a protocol for it, so assessment refuses it.

## Scope

**In scope:**
- module-OUTPUT interventions on one tensor leaf of one call: zero ablation, constant replacement (scalar or exact-shape tensor), and activation patching from a source pass in the same experiment;
- scalar built-in metrics, plus clearly marked caller metrics;
- `CausalEffect` records (INSTANCE, FINITE_SAMPLE);
- the `intervention_threshold` claim-test protocol, with a protocol registry;
- ground-truth causal models;
- a benchmark.

**Out of scope:** input/parameter/attention-head/functional-op interventions, POPULATION estimates, sufficiency, attribution, concepts, large models, and patching from external saved traces.

## Design

| Piece | Where | Notes |
|---|---|---|
| `InterventionRecord` (kind `intervention`) | `schema/interventions.py` | site (OUTPUT) + leaf path, call index, operation ZERO/CONSTANT/PATCH, scalar constant or `TensorRef` value (content digest, stored in the trace), PATCH source `RecordRef` to the source `ActivationRecord`. Content-derived id = `intervention_id` |
| `CausalEffect` (kind `causal_effect`) | same | intervention refs, `MetricSpec`, estimand, baseline/intervention values, `effect = intervention − baseline`, estimator `exact`. Status derived: INTERVENTIONAL (instance/finite sample, exact), ESTIMATED_CAUSAL only for POPULATION with a non-exact estimator (not produced in Phase 2) |
| Container checks | `core/trace.py` | a provenance record with `INTERVENTION` must name an `InterventionRecord` in the trace; intervention tensors count as retained tensors |
| Runtime | `interventions/` | builders `zero`/`constant`/`patch`; `intervene()`, `intervene_sample()`; metrics; claim protocol; limitations |
| Hooks | `core/hooks.py` | BeyondNN-owned temporary intervention hooks registered *through the session*, so the foreign-hook check recognises them; the replacement runs before output observation, so intervened activations are what downstream receives (and are MEASURED) |

**Pairing checks (refuse on failure):**
- no module in training mode;
- the CPU RNG state is unchanged by the baseline pass (a changed state means randomness was consumed);
- the model state fingerprint is identical at the start of the baseline and intervention passes;
- the intervention applied exactly once, to the declared call.

## Ground-truth expectations (written before running)

The models are fixed-weight, with input `x = (x0, x1)` and scalar output `y`.

| Model | Definition | Intervention | Expected effect |
|---|---|---|---|
| Additive | `y = a(x) + b(x)`, `a = x0`, `b = x1` | zero `a` | `−x0` |
| Additive | same | constant `a := c` | `c − x0` |
| Additive | same | patch `a` from source `x'` | `x'0 − x0` |
| Additive | same | patch `a` from identical source | `0` |
| Additive | same | replace `a` with its own original value | `0` |
| Gated | `y = gate(x)·signal(x)`, `gate = x0`, `signal = x1` | zero `gate` | `−x0·x1` (output → 0) |
| Redundant | `y = p(x) + q(x)`, `p = q = x0` | zero `p` | `−x0`, output `x0 ≠ 0`: ablating one path does not remove the output |
| Interaction | `y = a(x)·b(x)`, `a = x0`, `b = x1` | zero `a` | `−x0·x1`: `0` when `x1 = 0` (context-dependent) |

## Milestones

1. Plan and ADR.
2. Schema: intervention and effect records, container checks, limitations.
3. Runtime: operations, hooks, `intervene`, metrics, pairing checks, ground-truth models, tests.
4. Claim integration: protocol registry, `intervention_threshold`, policy.
5. Finite-sample comparisons, persistence round trip, benchmark, report.

Each milestone ends green on Python 3.10, 3.12, and 3.14, with mutation checks.
