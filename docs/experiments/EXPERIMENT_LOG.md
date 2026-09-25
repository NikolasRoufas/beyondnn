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

---

## 2026-09-25: M1.1 review — mutation re-check after added guard tests (verification)
- **Commit:** the M1.1 review-fix commit (parent `3349dec`)
- **Experiment:** re-ran the invariant mutation check after adding independent guard tests. Added three mutations for the new id-stability fixes: unsorted lineage, unsorted evidence, and un-normalised `-0.0`.
- **Model / dataset / seed:** none. **Hardware:** macOS (Darwin 24.5), CPU, Python 3.14.3.
- **Result:** failing tests per mutation:
  - causal-evidence check: 16;
  - GENERATED rejection: 2;
  - assessment recomputation: 2;
  - derivation rules: 2;
  - population matching: 2;
  - policy protocol requirement: 2;
  - id integrity: 3;
  - lineage sorting: 1;
  - evidence sorting: 1;
  - `-0.0` normalisation: 1.

  The first `-0.0` guard test **failed to catch its mutation**: it compared values with `==`, which `-0.0` passes. It was rewritten to compare record ids.
- **Conclusion:** every scientific invariant now has at least two independent catching tests. The three id-stability normalisations have one each, which is acceptable for mechanical canonicalisation.
- **Next action:** final M1.1 checkpoint, then M1.2.

---

## 2026-09-25: M1.2 model fingerprint timing (~1M parameters)
- **Commit:** the M1.2 commit (parent `90dc715`)
- **Experiment:** time `fingerprint_model` (FULL v1) on a dense CPU model.
- **Model:** `Sequential(Linear(1000,1000), ReLU, Linear(1000,1), BatchNorm1d(1))`: 1,002,003 float32 parameters (4.0 MB) plus 3 BatchNorm buffers. Seed 0.
- **Hardware:** Apple arm64, macOS (Darwin 24.5), CPU, 4 torch threads. Python 3.14.3, torch 2.12.0.
- **Configuration:** one warm-up call, then N timed calls with `time.perf_counter`.
- **Result 1 (negative, kept):** the first implementation read bytes with `bytes(tensor.untyped_storage())`. Median **3,073 ms** (min 2,997, max 3,138; N=20). Diagnosis: storage iteration happens element by element in Python (3.7 s for 4 MB in isolation).
- **Result 2:** reading the same bytes with `ctypes.string_at` from a contiguous CPU copy took median **1.83 ms** (min 1.73, max 2.05; N=50). Byte equality with the reference path is asserted for 10 dtypes in `test_fast_byte_path_equals_reference_storage_bytes`, and an independent re-implementation of the documented algorithm reproduces the digest.
- **Conclusion:** FULL hashing is cheap at this scale. No sampled mode is needed for Phase 1. The cost scales roughly linearly with model bytes, plus one transient copy per tensor.
- **Next action:** re-measure inside tracing overhead (M1.9). Check peak memory for large tensors (the transient copy doubles the per-tensor memory).

---

## 2026-09-25: M1.2 fingerprint determinism across processes, Python and torch versions
- **Commit:** the M1.2 commit (parent `90dc715`)
- **Experiment:** fingerprint and provenance of one fixed model in independent processes.
- **Model:** `Sequential(Linear(4,8), ReLU, Linear(8,4,bias=False), BatchNorm1d(4))` with a copied (untied) transposed weight. Seed 0.
- **Hardware:** Apple arm64, macOS, CPU.
- **Configuration:** Python 3.14.3 + torch 2.12.0 under `PYTHONHASHSEED` 0, 1, 12345 and random. Python 3.10 + torch 2.14.0 and Python 3.12 + torch 2.14.0 (uv environments).
- **Result:**
  - Structure digest `d669006d29df6b13…` and state digest `7b6a3e65c6f64a0f…` were identical in all 6 runs.
  - `provenance_id` was identical across hash seeds, and differed between Python/torch versions (`51099b9a…`, `375a0c0d…`, `e4262e45…`).
- **Conclusion:**
  - Model identity is independent of process, hash seed, Python version, and torch version (2.12 → 2.14) for this model.
  - Provenance correctly changes with the environment (ADR-019).
  - Not tested: other OS/architectures, and non-CPU devices.
- **Side findings:**
  - torch 2.14 deprecates quantized tensor creation (UserWarning). The test setup now silences that one warning.
  - A torch install without numpy warns at import, which `-W error` turns into collection errors. BeyondNN itself does not use numpy.

