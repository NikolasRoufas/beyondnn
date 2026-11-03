# Comparative evaluation (Stage 3, S3-CMP)

- **Protocol:** `protocols/P2_COMPARATIVE_EVALUATION_PROTOCOL.md`, committed at `84e91df` before execution.
- **Script:** `experiments/tnnls_revision/stage3/compare_workflows.py`, with the final tally in `stage3/retally.py`.
- **Results:** `experiments/tnnls_revision/results/S3_CMP_v2/` and `stage3/results/comparative_tally.json`.
- **Execution:** `interstellar`, CPU, one process, 2026-11-03; model A, frozen seed-0 checkpoint `4c19a8da…`, 20 held-out samples at the input site.

## 1. A failure, kept on record

**Run v1** (`runs/S3_CMP`, not used for results):
- The plain-PyTorch baseline ranked IG scores by their *signed* value, whereas `F.top_k` ranks by absolute value (ties broken by lower index).
- As a result, the two workflows selected different features on 4 of 20 samples, with outcomes disagreeing on 2.

**Run v2** (fix in commit `601d640`) uses the same ranking rule. The two paths are then measurement-equivalent:
- identical selections on 20/20 samples;
- identical outcomes on 20/20;
- maximum drop difference 2.9 × 10⁻⁶.

Every result below comes from v2.

## 2. Same measurements, three workflows

| workflow | what it is |
|---|---|
| **B0**, single-configuration report | the conclusion is the outcome of the PRIMARY IG comprehensiveness entry; no checks |
| **B1**, careful conventional checker | the same flat log, plus 7 explicit checks written for this study (54 non-blank lines): checkpoint hash, input hash, intervention required, controls present, k, outcome recomputation, multiverse disagreement |
| **BeyondNN** | the frozen central plan (`central75.plan_for` + `roles_for`, 98 non-blank lines, reused unchanged) and `bnn.audit` |

## 3. Results

Detections are counted per affected sample.

| fault | affected | erroneous B0 "supported" conclusions | B0 | B1: detected / false alerts | BeyondNN: detected / false alerts | BeyondNN standing on affected samples |
|---|---|---|---|---|---|---|
| F1: attribution taken as necessity | 20 | 20 | 0 | 20 / 0 (c3) | 20 / 0 | UNSUPPORTED ×20 (`attribution_is_not_intervention`) |
| F2: wrong checkpoint | 5 | 1 | 0 | 5 / 0 (c1) | 5 / 0 | NOT_EVALUATED ×5 (`other_checkpoint`) |
| F3: wrong input filed under a declared sample | 5 | 0 | 0 | 5 / 0 (c2) | 5 / 0 | NOT_EVALUATED ×5 (`no_evidence`) |
| F4: wrong selection size | 5 | 1 | 0 | 5 / 0 (c5) | 5 / 0 | NOT_EVALUATED ×5 (`primary_untested`, `no_evidence`) |
| F5: omitted controls | 20 | 6 | 0 | 20 / 0 (c4) | 20 / 0 | UNSUPPORTED ×20 (`primary_untested`, `required_protocol_missing`, `invariance_untested`): the declared PRIMARY null, count-matched controls, was never run |
| F6: configuration dependence | 4 | 4 | 0 | 4 / **1** (c7) | 4 / 0 | SUPPORTED + `alternative_reverses` ×4 |
| F7: tampered outcome | 5 | 5 | 0 | 5 / 0 (c6) | 5 (**study-level**) / — | evidence refused on load (`TracePersistenceError`: stored id does not match content) |

On the clean study (F0), B1 raised no fault flag. Its multiverse check flagged 5 samples as configuration-dependent; BeyondNN's role findings flagged 4.

## 4. What the comparison shows

1. **H-CMP1 holds.** Single-configuration reporting surfaced none of the injected faults. It would have reported 37 erroneous "supported" conclusions across F1–F7 (20 + 1 + 0 + 1 + 6 + 4 + 5).
2. **H-CMP2 holds, and it is the important negative result.** A careful researcher who writes the right checks detects every fault in this set. **BeyondNN does not detect anything that a careful, fault-aware conventional checker cannot.**
3. **H-CMP3 holds.** The BeyondNN audit flagged every fault from its declared plan, with no fault-specific code and no false alert on unaffected samples.
4. **Where the two differ:**
   - **Semantics of uninformative tests.** The one B1 false alert (F6, sample 47) is a test whose control criterion was unattainable. The flat log records it as `contradicts`, so the multiverse check counts it as counterevidence. BeyondNN treats it as uninformative (ADR-054). A hand-written rule silently mis-scores exactly this case, unless its author knows the attainability problem.
   - **Localisation of tampering.** B1 localises a tampered outcome to the sample. BeyondNN refuses the whole evidence set, because any edit breaks the content hash. BeyondNN's refusal is stronger, since it also covers edits to raw values that a recomputation check could not see. It is less informative about *where* the edit happened.
   - **Burden.** By lines of code, BeyondNN is *not* cheaper here: its plan (98 lines) is longer than B1's checks (54 lines). The difference is reuse and coverage. B1's checks are bound to one log schema and one study, and were written knowing the fault list. BeyondNN's are part of the representation and apply to any plan.
5. **Cost of the evidence layer**, for one PRIMARY IG necessity test with 50 controls on the MLP (30 input features), median of 3 repeats over 20 samples:
   - BeyondNN path: **55.3 ms**;
   - plain PyTorch + Captum: **2.38 ms**, about 23× faster.

   For a model this small, record construction, hashing and provenance dominate. The relative overhead shrinks as model execution grows; Stage 4 (`COMPUTATIONAL_EVALUATION.md`) reports the per-phase breakdown for all models.

## 5. Limits

- The faults were designed by the authors, and B1 was written knowing them. Detection rates describe this fault set, not real-world error frequencies.
- Model A only, 20 samples, one site.
- B1 is one reasonable conventional workflow, not the strongest possible one.
- No user study: real researchers' error rates and effort were not measured.

## 6. Recommended manuscript statement

> "On identical measurements with seven injected workflow faults, single-configuration reporting surfaced none. A conventional checker detected all seven once its author wrote a dedicated check for each (54 lines). The BeyondNN audit detected all seven from its declared plan without fault-specific code. It differed from the hand-written checks in treating unattainable-control tests as uninformative rather than as counterevidence, and in refusing, rather than localising, tampered evidence. The evidence layer added about 53 ms per test on a 30-feature MLP."
