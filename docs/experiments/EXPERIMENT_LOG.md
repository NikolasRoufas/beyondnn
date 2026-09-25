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

---

## 2026-09-25: M1.3 reference models × M1.2 fingerprint cross-check
- **Commit:** the M1.3 commit (parent `628630a`)
- **Experiment:** use the new reference models to cross-check fingerprint behaviour and model contracts.
- **Models:** `TinyMLP` (139 params), `TinyCNN` (396), `TinyTransformer` (5,120; tied `lm_head` counted once).
- **Hardware:** Apple arm64 CPU. Python 3.14.3 / torch 2.12.0; also Python 3.10 and 3.12 with torch 2.14.0.
- **Seeds:** 0, 1, 2, 7, 123.
- **Results:**
  - For all three models, the same seed gives identical `state_dict`, outputs and fingerprint.
  - A different seed gives the same structure digest, a different state digest, and different outputs.
  - Construction leaves `torch.get_rng_state()` unchanged, including when construction raises.
  - A `state_dict` round trip reproduces outputs and fingerprint, and the tie survives `load_state_dict` and `.to(float64)`.
  - Tie vs equal copy: identical outputs, different structure digest, one fewer parameter tensor.
  - Changing the non-persistent `causal_mask` changes both the outputs and the state digest.
  - Running `TinyCNN` in train mode updates its BatchNorm running stats, which changes the state digest.
- **Negative finding (kept):** `TinyTransformer(n_heads=2)` and `n_heads=4` with the same seed have **identical** FULL fingerprints but different outputs. Plain Python hyperparameters are outside v1 identity (consistent with ADR-020's "code and attributes are not hashed", but a more everyday case than editing `forward`). It is pinned by `test_known_limitation_plain_hyperparameters_are_not_fingerprinted`.
- **Mutation check:** removing RNG isolation, the tie, the causal mask, or the second shared call is caught by 5, 4, 1, and 1 tests respectively.
- **Next action:** decide on caller-declared config/revision in provenance (roadmap investigation item) before comparing differently configured models. M1.4 site resolution next.

---

## 2026-09-25: M1.4 site-resolution guard (mutation) checks
- **Commit:** the M1.4 commit (parent `ff34413`)
- **Experiment:** break the resolver in a scratch copy, one defect at a time, and run `tests/test_sites.py` (59 tests).
- **Hardware:** Apple arm64 CPU, Python 3.14.3, torch 2.12.0.
- **Results (failing tests per broken implementation):**

  | Broken implementation | Failing tests |
  |---|---|
  | `*` matches multiple segments | 8 |
  | alias modules deduplicated by object identity | 1, then 2 after adding an independent wildcard-alias test |
  | pattern order controls result order | 3 |
  | no-match patterns silently ignored | 3 |
  | parameterless modules skipped | 11 |
  | `**` matches the root | 9 |
  | modules with a tied weight collapsed | 4 |
- **Conclusion:** every required guard is caught by at least 2 tests.
- **Torch versions:** traversal order and alias preservation from `named_modules(remove_duplicate=False)` behaved identically on torch 2.12 (Python 3.14) and torch 2.14 (Python 3.10, 3.12).
- **Next action:** M1.5 hook lifecycle.

---

## 2026-09-25: M1.5 hook-session compatibility probes and mutation checks
- **Commit:** the M1.5 commit (parent `daa2a46`)
- **Probe (torch 2.12 source and behaviour; the tests also pass on torch 2.14):**
  - `always_call=True` forward hooks run once per invocation, both when `forward` raises and when a later pre-hook raises.
  - On failure they receive the current `result` variable: `None`, or a real output if a later hook failed. So `output is None` cannot detect failure.
  - Ordinary forward hooks do not run on failure.
  - An `always_call` hook that raises during a failure produces a `UserWarning` (an error under `-W error`), so BeyondNN's cleanup hooks never raise.
- **Mutation results (`tests/test_hooks.py`, 50 tests; failing tests per broken implementation):**

  | Broken implementation | Failing tests |
  |---|---|
  | hooks not removed on normal exit | 8 |
  | not removed on any exception | 4 |
  | not removed only for forward errors | 1 (a targeted variant) |
  | not removed only for sink errors | 1 (a targeted variant) |
  | call index never reset | 5 |
  | INPUT and OUTPUT claim separate indices | 10 |
  | user hooks removed | 2 |
  | hook returns a modified output | 22 |
  | output values retained by the session | 2 |
  | alias treated as path-specific | 6 |
  | alias module hooked twice | 2 |
  | no cleanup after partial install | 1 |
- **Conclusion:** all required guards are caught.
- **Next action:** caller-declared model provenance (M1.6 gate), then M1.5 review.

---

## 2026-09-25: M1.6 gate: declared model context closes the n_heads gap
- **Commit:** the declared-model provenance commit (parent: the M1.5 commit)
- **Experiment:** repeat the M1.3 negative finding with a caller declaration.
- **Model:** `TinyTransformer(n_heads=2, seed=0)` vs `TinyTransformer(n_heads=4, seed=0)`.
- **Hardware:** Apple arm64 CPU, Python 3.14.3, torch 2.12.0.
- **Result:**
  - The FULL v1 fingerprints are still equal (limitation unchanged by design).
  - Without a declaration, `provenance_id` is equal.
  - With `declared_model=ModelDeclaration(config={"n_heads": 2})` vs `{"n_heads": 4}`, the `provenance_id`s **differ**, and `ModelIdentity` is identical in both.
  - Implementation and checkpoint revisions also separate the ids.
  - The v1→v2 migration decodes an old payload with `declared_model = None`.
- **Mutation checks:** dropping the declaration in `make_provenance` is caught by 6 tests; removing the migration by 2.
- **Conclusion:** the M1.6 hard gate is resolved.
- **Next action:** M1.5 and gate review, then M1.6.

---

## 2026-09-25: M1.5 invocation-guard fix: regression and mutation check
- **Commit:** "Fix recursive hook invocation bookkeeping" (parent `5c96212`)
- **Experiment:** a recursive module whose inner attempt (`depth=1`) is rejected by a user pre-hook. The outer call catches the error and makes another inner call. Tested both with the module as the root and as a child.
- **Result:** the events are outer INPUT (call 0); inner attempt with call 1 consumed and no INPUT event; inner INPUT/OUTPUT (call 2); outer OUTPUT (call 0). Pass 1 repeats this exactly, and all hooks are removed afterwards.
- **Mutation:** restoring the pre-fix design (claiming in the observation pre-hook, no guard) is caught by both regression tests. Before the fix, this scenario would have popped the outer frame.
- **Conclusion:** fixed. Remaining pre-guard entry point: global module pre-hooks, which sessions now refuse.

