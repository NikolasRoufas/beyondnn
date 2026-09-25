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

---

## 2026-09-25: M1.6 trace pipeline mutation checks
- **Commit:** "Add trace recording pipeline (M1.6)" (parent `b1e5234`)
- **Results (tests/test_trace.py; failing tests per broken implementation):**

  | Broken implementation | Failing tests |
  |---|---|
  | recording keeps the original tensors alive | 0, then **3** |
  | `pass_index` flattened to 0 | 2 |
  | `call_index` ignored | 4 |
  | selected-not-executed limitation removed | 2 |
  | provenance check removed | 2 |
  | forged `EvidenceRef` accepted | 2 |
  | `trace()` diverging from `recording()` | 4 |
- **Negative finding (kept):** the first retention mutation survived. The weakref tests only used `trace()`, whose internal recording object is discarded on return. A test that keeps a live `recording()` context and its result now catches it.
- **Conclusion:** all required guards are caught.

---

## 2026-09-25: M1.7 persistence mutation checks and corruption matrix
- **Commit:** "Add trace persistence and migration (M1.7)" (parent `4aae85d`)
- **Corruption matrix (all rejected with `TracePersistenceError`):**
  - missing, malformed, or NaN-literal `trace.json`;
  - unknown kind; unsupported schema version (document- and record-level);
  - tampered id; dangling input or provenance; reordered records;
  - unknown document key (e.g. `tensor_file: ../../etc/passwd`); a `storage_key` rewritten to a path (the id no longer matches);
  - wrong format or format_version; unsorted keys; invalid config;
  - missing sidecar or key; wrong shape, dtype, or content;
  - pickled object (never executed); non-tensor value; unexpected sidecar; symlinked sidecar.
- **Mutations (tests/test_persistence.py; failing tests per broken implementation):**

  | Broken implementation | Failing tests |
  |---|---|
  | `weights_only=False` | 1 |
  | remapping removed | 2 |
  | shape validation removed | 1 |
  | dtype validation removed | 1 |
  | extra document keys accepted, with the sidecar name read from JSON | 1 |
  | record-id integrity skipped | 3 |

  With `weights_only=False`, the pickled payload (a harmless `echo`) really executed in the scratch copy, so the guard matters. With shape or dtype validation removed, the content-digest check still rejects the file, but with a different error; the tests pin the specific check.
- **Conclusion:** all required guards are caught.

---

## 2026-09-25: M1.9 Phase-1 tracing overhead characterisation
- **Commit:** "Add Phase 1 performance benchmarks (M1.9)" (parent `a969d4a`)
- **Script:** `benchmarks/bench_trace_overhead.py` (rerunnable: `python benchmarks/bench_trace_overhead.py`).
- **Models and sites:**
  - TinyMLP: sites `**`, input 32×4;
  - TinyCNN (eval): sites `**`, input 8×1×8×8;
  - TinyTransformer: sites `blocks.*.attn`, `blocks.*.mlp`, `lm_head`, input 4×8 tokens;
  - Dense~1M: `Linear(1000,1000)–ReLU–Linear(1000,2)`, sites `**`, input 16×1000.
- **Seeds:** fixed (model seed 0; input generators seed 0).
- **Methodology:** everything runs under `torch.no_grad()`, with 3 warm-up and 30 measured iterations; save/load use 1 warm-up. Median and p90 are reported. Sizes are exact bytes. No tracemalloc.

**Final run** (the script as committed):

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64, 4 torch threads; warm-up 3, 30 measured iterations; times are median / p90 in ms, under no_grad.

| model | params | input | sites | act. records | baseline | fingerprint | trace none | trace summary | trace cpu | fingerprint share of summary overhead | save (cpu) | load (cpu) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyMLP | 139 | (32, 4) | 4 | 6 | 0.021 / 0.024 | 0.067 / 0.070 | 0.786 / 0.816 | 1.028 / 1.053 | 1.080 / 1.111 | 7% | 0.616 / 0.670 | 1.330 / 1.373 |
| TinyCNN | 396 | (8, 1, 8, 8) | 9 | 9 | 0.108 / 0.114 | 0.155 / 0.160 | 1.282 / 1.347 | 1.626 / 1.651 | 1.737 / 1.788 | 10% | 0.624 / 0.656 | 1.776 / 1.828 |
| TinyTransformer | 5,120 | (4, 8) | 5 | 7 | 0.255 / 0.263 | 0.297 / 0.306 | 1.592 / 1.662 | 1.864 / 1.924 | 1.965 / 2.036 | 18% | 0.652 / 0.792 | 1.558 / 1.593 |
| Dense~1M | 1,003,002 | (16, 1000) | 3 | 3 | 0.272 / 0.317 | 1.922 / 2.077 | 2.847 / 3.021 | 3.100 / 3.295 | 3.174 / 3.407 | 68% | 0.707 / 0.855 | 1.239 / 1.452 |

| model | records | retained tensor bytes (cpu) | trace.json bytes | tensors.pt bytes |
|---|---|---|---|---|
| TinyMLP | 11 | 6,016 | 12,803 | 9,233 |
| TinyCNN | 14 | 41,344 | 16,985 | 45,381 |
| TinyTransformer | 13 | 16,640 | 14,654 | 20,173 |
| Dense~1M | 8 | 192,128 | 8,794 | 194,589 |

**Earlier run** (same logic, before a lint-driven refactor that changed the measurement order): Dense~1M fingerprint 1.83 ms (58% of summary overhead); tiny-model trace times within about 3% of the final run. Both runs are kept, to show the run-to-run spread.

- **Findings:**
  1. **Fixed per-trace cost of about 0.8–1.3 ms** on the tiny models. In relative terms that is large (~5–40× a 0.02–0.25 ms forward); in absolute terms it is small. A profile of TinyMLP attributes about a third of it to per-record integrity work (content hashing at construction, plus id recomputation and canonical JSON in `TraceResult._add`). The rest is tensor summaries and per-pass provenance.
  2. **Fingerprint cost is re-measured, not copied:** 1.92 ms median (p90 2.08) for the 1,003,002-parameter dense model (~4 MB float32), versus 1.83 ms in the earlier run. That is 58–68% of its summary-trace overhead; for the tiny models it is 7–19%. This is consistent with the earlier ~1.8 ms/1M figure *on this machine only*.
  3. **Retention mode barely changes time** at these sizes: none < summary ≈ cpu. `cpu` retention stores exactly the retained tensor bytes shown, and `tensors.pt` adds ~2–4 KB of container overhead.
  4. **Save takes ~0.6 ms and load ~1.3–1.8 ms**, dominated by full re-validation (by design).
- **Conclusion:** nothing pathological, so there is **no optimisation in Phase 1**.
- **Candidates for later:** skip the redundant id recomputation for records the container itself just built; batch statistics.
- **Not measured:** GPU, large models, peak process memory.

---

## 2026-09-25: M1.10 final audit
- **Commit:** "Complete Phase 1 audit and gate report (M1.10)" (parent `d99e026`)
- **Validation:** 638 tests on Python 3.14/torch 2.12, 3.10/torch 2.14, and 3.12/torch 2.14 (`-W error`, 0 skipped); ruff, mypy `--strict`, and sdist/wheel builds are clean.
- **Clean environment:** a fresh uv virtualenv (Python 3.12.13, torch 2.14.0) with the built wheel installed, run outside the checkout with `-W error`. The smoke test (trace, recording, save/load, instrument/explain) and the README runnable example both pass. `beyondnn` resolved to site-packages.
- **Final mutations (all caught; failing tests per broken implementation):**

  | Broken implementation | Failing tests |
  |---|---|
  | occurrence `pass_index` removed | 1 |
  | external hook allowed | 5 |
  | WHY labelled causal | 2 |
  | `NO_CAUSAL_EVIDENCE` omitted | 2 |
  | `instrument` copies the model | 1 |
  | `handle.trace` diverges | 1 |
  | WHY copies records | 1 |
  | multi-device silently accepted | 1 |
  | binding check removed | 1 |
  | structure check removed | 1 |
- **Audit finding (fixed):** a selected module replaced by an identical-shape object between passes silently lost evidence. The first fix attempt, using only the structure digest, **failed its test**: an identical-shape replacement leaves the digest unchanged. Checking object identity closed the gap (ADR-027).
- **Scientific-language audit:** the README intro implied interventions and claim testing were available, and that all evidence statuses were produced; it was corrected. Package docstrings and render strings contain only disclaimers or schema vocabulary.
- **Gate:** GO WITH EXPLICIT LIMITATIONS.

---

## 2026-09-25: Phase 2 ground-truth intervention effects and mutation checks
- **Commit:** "Add causal intervention records and paired comparisons" (parent `b6889c4`)
- **Models:** `beyondnn/_testing/causal_models.py` (fixed weights, input `(x0, x1)`).
- **Hardware:** CPU, Python 3.14.3, torch 2.12.0.
- **Expectations:** written in `docs/PHASE_2_PLAN.md` before running.
- **Results:** all reproduced exactly, with float equality:
  - additive, zero `a`: `−x0` for (3, 5), (−2, 0.5), (0, 1);
  - additive, constant 10: `10 − x0`;
  - additive, patch from (7, 1): `+4`;
  - identical-source patch: `0`; own-value constant: `0`;
  - gated, zero gate: `−15` (output `0`);
  - redundant, zero `p`: `−3`, with the output still `3`;
  - interaction: `0` at `x1 = 0`, `−15` at (3, 5);
  - finite sample {1, 2, 6}: mean `−3`, from 3 instance effects.
- **Claim semantics:** on the redundant model, "p necessary with `min_effect = 6`" is CONTRADICTED (effect −3). A nonzero effect did not imply support.
- **Mutations (failing tests per broken implementation):**

  | Broken implementation | Failing tests |
  |---|---|
  | hook leak | 34 |
  | baseline run under intervention mode | 1, then 2 after adding an independent test |
  | intervention pass marked CLEAN | 2 |
  | activations labelled INTERVENTIONAL | 23 |
  | CausalEffect labelled MEASURED | 7 |
  | sign reversed | 28 |
  | patch source ignored | 4 |
  | zero ablation not applied | 11 |
  | stochastic comparison accepted | 1, then 2 |
  | state-mutating passes paired | 1, then 2 |
  | causal result without a CausalEffect | 10 |
  | GENERATED as causal support | 2 |

  All are caught.

