# Experiment Log

Append-only. Negative results stay. Each entry records: date, git commit, experiment, model, dataset, seed, hardware, configuration, result, conclusion, and next action.

---

## 2026-09-24: Ecosystem audit (desk research, no code)
- **Commit:** none (repo not yet committed)
- **Experiment:** survey of PyTorch 2.14, Captum 0.9.0, TransformerLens 4.0.0, nnsight 0.7.0, SAELens 6.51.2, pyvene 0.1.8, Quantus 0.6.0, circuit-tracer, and Inseq.
- **Result:** tracing, intervention, and attribution are commodity capabilities. There is no existing epistemic-status or provenance schema, and claim-level faithfulness testing of internal components across architectures is thin.
- **Conclusion:** position BeyondNN as an evidence-accounting and claim-testing layer, not as a new tracing engine.
- **Next action:** architecture review with the project owner before any code.

---

## 2026-09-25: M1.1 invariant mutation check (verification, not a research experiment)
- **Commit:** the M1.1 schema commit (parent `90b9124`)
- **Experiment:** disable one invariant at a time in a scratch copy, then run the full suite, to check that the tests actually guard each rule.
- **Model / dataset / seed:** none (schema only).
- **Hardware:** macOS (Darwin 24.5), CPU, Python 3.14.3.
- **Configuration:** the invariants disabled were:
  - the causal-evidence check in `ClaimTestResult`;
  - GENERATED rejection in `EvidenceRef`;
  - assessment verdict recomputation;
  - derivation rules;
  - population estimand matching;
  - the requirement that a causal relation's policy names a protocol;
  - decode id integrity.
- **Result:** every mutation was caught. Failing tests per mutation: 14, 2, 2, 1, 1, 1, 1.
- **Conclusion:**
  - The rules are guarded.
  - Derivation rules, population matching, the policy requirement, and id integrity are each guarded by a single test. That is thin but sufficient for now.
  - Separately, running the suite on Python 3.10 found a real bug: there, `list[int]` passes `isinstance(tp, type)`. It was fixed before commit.
- **Next action:** review M1.1. Consider adding mutation testing (e.g. mutmut) to CI once the codebase grows.

