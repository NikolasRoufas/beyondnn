# Protocol P2: comparative evaluation against conventional workflows (S3-CMP)

**Status:** written and committed before execution (2026-11-03). Script: `experiments/tnnls_revision/stage3/compare_workflows.py`.

## Research question

On the same model, samples, attributions, interventions and controls, which workflow errors does the BeyondNN audit detect, and which do conventional workflows detect? At what configuration burden and computational cost?

## Hypotheses (stated before running)

- **H-CMP1:** single-configuration reporting (B0) detects none of the injected faults.
- **H-CMP2:** a careful conventional checker (B1) can detect each fault for which it implements an explicit check. Detection then depends on the checks its author writes; it does not come from the representation.
- **H-CMP3:** the BeyondNN audit flags every injected fault from its declared plan, without fault-specific code.

The comparison is about *which checks exist by default*, not about better measurements. Both workflows see identical numbers.

## Design

| field | value |
|---|---|
| model / checkpoint | A (MLP, breast cancer), frozen seed-0 checkpoint (digest `4c19a8da…1c24`); fault F2 uses the independently trained seed-1 checkpoint from protocol P1 |
| site / samples | `input`; the first 20 frozen held-out samples (development of this protocol used no outcomes); 5 further held-out samples serve as substitute inputs for F3 |
| evidence | the frozen central grid (`central775.run_sample`): G / IG / R × k ∈ {5, 10, 20}% × 3 replacements × comprehensiveness / sufficiency with count controls, plus magnitude and no-control comprehensiveness under PRIMARY |
| BeyondNN | `bnn.audit` with the frozen central plan (`central775.plan_for`) |
| baseline log (same measurements) | one flat JSON entry per result, written in the evidence loop as a careful researcher would: checkpoint SHA-256 of `state_dict` bytes, input SHA-256, sample index, method, k, replacement, test, control type, number of controls, threshold τ, drop, control drops, outcome |
| B0 | "single-configuration report": the conclusion for a sample is the outcome of its PRIMARY IG comprehensiveness entry (method IG, k at 10%, PRIMARY replacement, count controls) |
| B1 | B0 plus explicit checks written for this study: (c1) entry checkpoint hash = declared; (c2) entry input hash = declared input of that sample; (c3) a necessity conclusion requires an intervention entry (not an attribution); (c4) the PRIMARY entry has ≥ 1 control; (c5) k = ⌈0.1 n⌉; (c6) the outcome is recomputed from the logged drop, control drops, τ and q = 0.95 and must equal the logged outcome; (c7) multiverse: flag if any logged ALTERNATIVE configuration contradicts a supported PRIMARY. Its non-blank code lines are counted. |

## Faults

Each fault is injected into *both* representations identically.

| id | fault | affected samples | injection |
|---|---|---|---|
| F0 | none (clean) | — | the false-alert reference |
| F1 | attribution taken as necessity | all 20 | only attribution evidence; B0/B1 conclude from the IG ranking |
| F2 | wrong checkpoint | 5 | those samples' results come from the seed-1 checkpoint |
| F3 | wrong input | 5 | those samples' results are computed on 5 other held-out inputs but filed under the declared sample |
| F4 | wrong selection size | 5 | the PRIMARY-k results are removed and the k = 5% results are filed as the PRIMARY claim |
| F5 | omitted controls | all 20 | only the uncontrolled PRIMARY comprehensiveness results |
| F6 | configuration dependence (single-configuration reporting) | samples PRIMARY-supported with ≥ 1 ALTERNATIVE contradiction | clean evidence; the question is whether the qualification is surfaced |
| F7 | tampered outcome | up to 5 samples whose PRIMARY IG outcome is `contradicts` | the outcome is changed to `supports` (flat log: field edited; BeyondNN: the saved trace JSON is edited and reloaded) |

## Metrics

- **Detected:** the workflow flags the affected sample, through a BeyondNN standing other than SUPPORTED, an exclusion/integrity finding, a qualifying finding, or a load failure. For B1, a failed check.
- **Missed:** the erroneous conclusion is reported without a flag.
- **False alerts:** flags on unaffected samples, and on F0.
- **Burden:** non-blank lines of B1 check code versus the declared plan (lines of `central75.plan_for`, which is reused unchanged).
- **Cost:** wall time of the PRIMARY IG test per sample. BeyondNN path: `A.attribute` + `F.top_k` + `F.run`. Plain path: Captum IG + `topk` + 51 forward passes. Median of 3 repeats; drop agreement and outcome agreement are reported.

## Statistics and limits

- **Statistics:** exact counts per fault, without significance tests. The faults are designed by the authors, so detection rates describe this fault set, not real-world error rates.
- **Compute:** CPU, a single process, at most 30 min.
- **Stopping rule:** none needed. A failed hypothesis is reported as such.
