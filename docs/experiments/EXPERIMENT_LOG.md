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

---

## 2026-09-25: Phase 2 intervention overhead, review findings, gate
- **Commit:** "Add Phase 2 benchmark, review fixes and report" (parent `1fd495b`)
- **Benchmark** (`benchmarks/bench_interventions.py`: eval mode, 3 warm-up, 30 iterations, median/p90):

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64, 4 threads; warm-up 3, 30 iterations; median / p90 ms.

| model | site | forward | baseline trace | zero-ablation comparison | patch comparison |
|---|---|---|---|---|---|
| TinyMLP | shared | 0.023 / 0.023 | 0.669 / 0.752 | 1.460 / 1.502 | 2.034 / 2.092 |
| TinyTransformer | blocks.0.attn | 0.251 / 0.261 | 1.311 / 1.352 | 2.661 / 2.698 | 3.810 / 3.843 |

  A comparison costs about two traced passes (three for patching) plus effect records; there was no optimisation.
- **Review findings (fixed before the gate):**
  - A model that modifies its input in place would have given the intervention pass a different input, silently. The input fingerprint is now checked after each pass.
  - Execution conditions (training/grad/device/randomness) were not compared between the paired passes; they now are.
  - Both mutations are caught.
- **Validation:** 699 tests on Python 3.10, 3.12, and 3.14. The clean-venv wheel smoke tests and README examples pass.
- **Gate:** GO WITH EXPLICIT LIMITATIONS.

---

## 2026-09-25: Phase 3 attribution: ground truth, Captum cross-check, mutations, benchmark

- **Pre-registration:** `docs/PHASE_3_PLAN.md` (`dd1bffc`) holds every expected value and tolerance, written before any attribution code.
- **Ground truth:** every expectation was met; see `docs/PHASE_3_REPORT.md` §5.
  - Linear, irrelevant-input, and product models: exact.
  - Saturating IG error vs tanh 3: 2.9e-5 / 1.8e-6 / 1.5e-7 at n = 16 / 64 / 256. The declared bound 2.25/n² was never approached.
  - One-sided-rule completeness deltas: exactly ∓15/n.
  - Per-call gradients on Twice: 2 and 1.
- **Negative example:** on the redundant model, `p` receives attribution 3.0 (ATTRIBUTED_TO supported), while zero-ablating `p` leaves the output at 3 (necessity contradicted).
- **Captum 0.9.0 cross-check:** max abs difference 1.2e-7 for IG (three rules, four models), for gradient/Saliency(abs=False) and IxG, and for layer IG on TinyTransformer embeddings.
  - Found by inspection before implementation, then confirmed: Captum's `riemann_trapezoid` weights sum to (n−1)/n (6.75 vs 7.5 on Product, n = 10).
- **Mutations:** 21 applied to scratch copies; all caught (report §9).
  - First pass: two survivors, neither a defect. One mutation was not actually a broken state; the other was masked by the second alias defence. Both were replaced by real or independent checks.
  - Seven single-test catches each got a second, independent test.
- **Validation:** 770 tests plus 1 reported skip without Captum, and 791 with Captum, on Python 3.10, 3.12, and 3.14. mypy is clean with and without Captum; after Captum pulled in numpy, numpy's stubs needed a `follow_imports=skip` override.
- **Benchmark** (`benchmarks/bench_attribution.py`, batch-1 inputs, 3 warm-up, 30 iterations):

Environment: Python 3.14.3, torch 2.12.0, Captum 0.9.0, Darwin arm64, 4 threads; warm-up 3, 30 iterations; median / p90 ms. IG: zero baseline (input ids 0 for TinyTransformer), rule riemann_middle.

| model | attributed | forward | trace | gradient | input x grad | IG 16 | IG 64 | Captum IG 16 | Captum IG 64 |
|---|---|---|---|---|---|---|---|---|---|
| Product (analytic) | input | 0.004 / 0.004 | 0.333 / 0.341 | 0.648 / 0.669 | 0.643 / 0.663 | 1.450 / 1.489 | 3.570 / 3.643 | 1.866 / 1.911 | 5.074 / 5.175 |
| TinyMLP | input | 0.020 / 0.020 | 0.456 / 0.471 | 0.957 / 1.000 | 0.954 / 0.988 | 2.297 / 2.354 | 6.190 / 7.189 | 2.823 / 3.139 | 7.646 / 8.343 |
| TinyCNN | input | 0.071 / 0.084 | 0.683 / 0.717 | 1.532 / 1.563 | 1.553 / 1.690 | 5.232 / 6.373 | 14.969 / 15.911 | 5.505 / 5.582 | 16.934 / 17.389 |
| TinyTransformer | token_embedding | 0.185 / 0.199 | 1.188 / 1.223 | 3.128 / 3.188 | 3.140 / 3.473 | 14.511 / 16.530 | 40.464 / 41.236 | 14.014 / 14.239 | 44.319 / 49.170 |

- **Gate:** GO WITH EXPLICIT LIMITATIONS.

---

## 2026-09-25: Phase 4 evidence synthesis: scenarios, mutations, benchmark

- **Pre-registration:** `docs/PHASE_4_PLAN.md` (`e4f2d9e`), with scenarios A–H and their expected structured outputs.
- **Design finding before code:** summary statistics cannot identify an input ((3, 5) and (5, 3) have equal statistics). This led to `InputRecord.sample_id` (ADR-031); scenario F now also refuses exactly that case.
- **Scenarios:** all expectations were met.
  - E (flagship): IG at `p` = 3.0 → ATTRIBUTED_TO SUPPORTED; zeroing `p` 6 → 3 → NECESSARY_FOR CONTRADICTED; shown side by side.
  - B: NECESSARY_FOR with attribution evidence only → UNTESTED.
  - H: MIXED.
- **Backward compatibility:** the measured-only render equals the output of the Phase-3 renderer (`3bc3f41`) on three traces, plus the NOT EVALUATED block.
- **Mutations:** 20 on scratch copies, all caught.
  - The first run had one survivor: a widened registry letting `intervention_threshold` justify ATTRIBUTED_TO. It was fixed with two independent defences.
  - Single-catch guards got second tests (report §11).
- **Language audit:** repository-wide; no unlicensed statements remain. Two wordings were changed.
- **Validation:** 808 passed + 1 reported skip without Captum, and 829 with Captum, on Python 3.10, 3.12, and 3.14. The clean wheel works with and without Captum.
- **Benchmark** (`benchmarks/bench_synthesis.py`):

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64; warm-up 3, 30 iterations; median / p90 ms. Evidence computed beforehand (not timed).

| composition | source records | compose | render | rendered chars |
|---|---|---|---|---|
| measured only | 7 | 0.173 / 0.177 | 0.046 / 0.048 | 1559 |
| trace + attribution | 19 | 0.447 / 0.452 | 0.074 / 0.075 | 3793 |
| trace + intervention | 23 | 0.553 / 0.559 | 0.062 / 0.064 | 2890 |
| trace + attribution + intervention + claims | 41 | 1.146 / 1.154 | 0.092 / 0.093 | 4609 |

- **Gate:** GO WITH EXPLICIT LIMITATIONS.

---

## 2026-09-25: Phase 5 faithfulness tests: pre-registered scenarios, premise test, mutations, benchmark

- **Commits:** `b7e1073` (plan) through the Phase-5 report commit. CPU; Python 3.14.3 / torch 2.12.0 for the numbers below.
- **Pre-registration:** `docs/PHASE_5_PLAN.md` §17 (scenarios A–N, RQ9), committed before any Phase-5 code.
- **Seeds:** controls use `F.controls(n, seed)` with the seed in each test: A/K seed 0; H seeds 0–39 for controls and 1000–1039 for the random methods.

| Scenario | Configuration | Result | Pre-registered | Conclusion |
|---|---|---|---|---|
| A: true causal selection | `Weighted8`, x = 1, IG top-2, N = 200 | drop 12; fraction_below 0.945; tied 0.055; P = 0.060 | drop 12; 0.964 ± 0.04; P in [0.006, 0.066] | Met. **Negative/limiting result:** a perfect selection does not reach P < 0.05 with 8 units, because 1/28 of random pairs coincide with it. |
| B: plausible distractor | `ProxyDistractor`, IG vs correlation ranking | IG {x0}: drop 3, SUPPORTS; correlation {x2}: drop 0, CONTRADICTS | same | Met. **The premise holds here:** only the protocol exposes the correlation ranking. |
| C: redundancy | `RedundantMax` (3, 3) | {x0}: 0 (CONTRADICTS); {x0, x1}: 3 | same | Met. Necessity is not causal relevance. |
| D: insufficient set | `EqualSum4` | retain {x0}: drop 3 (CONTRADICTS); remove {x0}: 1 (SUPPORTS) | same | Met. |
| E: invariance | `Additive`, swap values, IxG | prediction PASS; rho = -1 FAIL; Jaccard 0 FAIL | same | Met; not collapsed. |
| F: counterexample | `Interaction`, internal `a` | 3 of 4 held; counterexample (3, 0) | same | Met; recorded. |
| G: wrong sample | selection from (3, 5) on (1, 1) | refused (run and composition) | refused | Met. |
| H: random method | 40 random rankings vs N = 200 | mean superiority within [0.40, 0.60] (asserted); IG >= 0.95 | same | Met. |
| I: misleading gradient | `SaturatingPlus` | gradient top-1 {x1}: drop 0.2 CONTRADICTS; IG {x0}: 1.0 SUPPORTS; agreement rho = -1 FAIL | gradient about 6e-10 | Met. **Deviation:** the float32 gradient of tanh(12) underflows to exactly 0. |
| J: pre-ADR-031 trace | reference without sample_id | measured-only works; faithfulness refused | same | Met. |
| K: metrics disagree | `RedundantMax` {x0} | comprehensiveness CONTRADICTED, sufficiency SUPPORTED, both in WHY | same | Met. |
| L: readable but unused | `ProbeReadable` | h1 = x1 (readable); drops 6 / 0; retain {0} SUPPORTS with the site-relative limitation | same | Met. |
| M: curves | `Weighted8`, IG ranking | removal 0, 8, 12, 14, 15x5 (aopc 109/9); retention 15, 7, 3, 1, 0x5 (26/9) | same | Met, exact. |
| N: diagnostics | Product baselines; Saturating steps | baseline_sensitivity rho = -1 FAIL; step max diff 2.1e-5 PASS | same | Met. |
| RQ9: replacement choice | `Weighted8` x = (1, 1, 1, 1, 5, 5, 5, 5) | zero: drops 8, 4, 2, 1, 0, 0, 0, 0; "mean" (= x): all INCONCLUSIVE (no-op) | same | Met. The replacement choice alone can erase the signal. |

- **Mutation audit** (plan §13, 20 mutations on scratch copies). The first run had two survivors, both test gaps that were fixed:
  - a composition target check masked by claim targets;
  - retention orientation of paired controls.

  After the fixes, every guard is caught by at least 2 tests:

```
   2  1 selection sample-id check skipped
  11  2 ranking direction reversed
   6  3 controls of a different size
   2  4a controls from the global RNG
   2  4b control seed ignored
  68  5 perturbation not applied
  17  6 retain replaces S instead of its complement
   2  7 faithfulness target not checked in composition
   2  8 site-relative sufficiency limitation dropped
   2  9 paired orientation ignored for retention
   2  10 paired with another sample's controls
   2  11 faithfulness_evaluated without results
   2  12 sufficiency decides NECESSARY_FOR
   2  13 counterexample dropped from the summary
   2  14 no-op reported as CONTRADICTS
   2  15 stability aspects collapsed
   2  16 forged faithfulness result accepted
   3  17 curve k=0 anchor dropped
   2  18 aopc normalisation changed
   2  19 selection re-derivation skipped in composition
   2  20 concepts_validated becomes true
ALL CAUGHT
```

- **Benchmark** (`benchmarks/bench_faithfulness.py`): cost is linear in perturbation passes (about 0.6–0.9 ms/pass, about 6 records/pass) and nearly flat in the number of units.

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64, 4 threads; warm-up 1, 10 iterations; median / p90 ms.

| case | varied | value | passes | records | time |
|---|---|---|---|---|---|
| comprehensiveness, 50 controls | units d | 8 | 52 | 289 | 30.1 / 30.6 |
| comprehensiveness, 50 controls | units d | 32 | 52 | 343 | 31.0 / 31.1 |
| comprehensiveness, 50 controls | units d | 128 | 52 | 388 | 31.9 / 32.1 |
| comprehensiveness, d=16 | controls N | 0 | 2 | 18 | 1.3 / 1.3 |
| comprehensiveness, d=16 | controls N | 25 | 27 | 209 | 15.9 / 16.0 |
| comprehensiveness, d=16 | controls N | 100 | 102 | 716 | 67.0 / 67.2 |
| comprehensiveness, d=16 | controls N | 400 | 402 | 2369 | 360.5 / 369.7 |
| removal curve, 10 control rankings, d=16 | curve points | 5 | 45 | 324 | 27.8 / 30.3 |
| removal curve, 10 control rankings, d=16 | curve points | 9 | 89 | 673 | 59.4 / 60.6 |
| removal curve, 10 control rankings, d=16 | curve points | 17 | 177 | 1362 | 135.1 / 141.0 |
| dataset comprehensiveness, 20 controls, d=16 | samples | 1 | 22 | 151 | 13.8 / 13.9 |
| dataset comprehensiveness, 20 controls, d=16 | samples | 4 | 88 | 484 | 57.6 / 58.1 |
| dataset comprehensiveness, 20 controls, d=16 | samples | 16 | 352 | 1816 | 287.9 / 339.8 |
| internal comprehensiveness, 50 controls, d=16 | units d | 16 | 52 | 363 | 40.7 / 41.2 |

- **RQ8:** not attempted. There are 8 ground-truth tasks, which is too few to split into calibration and held-out sets. Logged as a deliberate negative decision.
- **Next action:** owner review of the Phase 5 report.


## 2026-09-26: Phase 5.5 realistic faithfulness validation (pre-registered)

- **Commits:** `23c0047` (literature + plan) through the Phase 5.5 report commit.
- **Pre-registration:** `docs/PHASE_5_5_PLAN.md`. §18 records the corrections and analysis-rule clarifications, made before any aggregate was computed, and the run provenance.
- **Environment:** Python 3.12.13, torch 2.14.0, Captum 0.9.0, transformers 5.17.0, scikit-learn 1.9.1, numpy 2.5.3; macOS arm64 (8 cores), CPU only, one torch thread per run. The pinned requirements are in `experiments/phase5_5/requirements.txt`, outside the core install.
- **Models** (none hand-constructed):

| ID | Model | Data | Held-out samples | Test accuracy | Checkpoint |
|---|---|---|---|---|---|
| A | MLP 30→64→32→2, trained (seed 0) | breast_cancer | 60 correct test rows (seed 1234) | 0.974 | sha256:4c19a8dafa3c0b38… |
| B | CNN (2 conv + head), trained (seed 0) | digits 8×8 | 60 correct test images (seed 1234) | 0.964 | sha256:d799d1e89efd31f2… |
| C | BERT-tiny fine-tuned on SST-2, `41ad6709…` | SST-2 validation (`bcdcba79…`) | 40 of 368 eligible sentences (seed 1234) | — (pretrained) | sha256:e2b44a0891e8b46f… |

- **Grid** per sample and site:
  - methods: gradient, input × gradient, IG n = 32, ablation ranking, random ranking;
  - k ∈ {1, 5, 10, 20}%;
  - three replacements;
  - comprehensiveness and sufficiency with N = 50 count-matched controls, plus comprehensiveness with N = 50 magnitude-matched controls under r1.
  - Totals: 36,155 claim-test runs (A 16,800; B 14,700; C 4,655), all kept (`results/faithfulness_*.json.gz`).

**Pre-registered hypotheses (per site; analysis rules in plan §18):**

| H | Prediction | Result | Verdict |
|---|---|---|---|
| H1 | IG and ablation median superiority ≥ 0.8 (A/B), ≥ 0.6 (C), p = 10%, r1 | A 1.00/1.00, 1.00/1.00; B 0.99/1.00, 0.995/0.995; C 0.945/0.92 | **held** at every site |
| H1b | magnitude-matched controls lower IxG's superiority more than IG's | decrease IxG vs IG: A-in 0.01 vs 0.005; A-hid 0 vs 0; B-pix 0.15 vs 0.14; B-ch 0.04 vs 0.04; C 0.08 vs 0.105 | **mostly not held** (2 of 5, both by ≤ 0.01) |
| H2 | median drop at k = 1: ablation ≥ IG ≥ IxG ≥ gradient | held under r1 in 4 of 5 sites (A-input fails: IxG 1.71 > IG 1.69); under r2/r3 it fails for A-input and B-pixels | **partly held**; replacement-dependent |
| H3 | comprehensiveness and sufficiency disagree in ≥ 10% of cells | 24–52% per site; 36.6% pooled | **held**, far above |
| H4 | ≥ 10% of outcomes change across replacements, plus ≥ 1 ordering reversal | A 33%/28%, B 49%/31%, C 5.0%; reversals at every site | **held** for A and B; **not held** for C (5%) |
| H5 | ≥ 20% of cells change outcome across t ∈ {0.25, 0.5, 0.75} | A 25%/32%, B 28%/34%, C 17% | **held** for A and B; **not held** for C |
| H6 | ρ(Jaccard, \|Δdrop\|) < 0 and \|ρ\| < 0.5 | −0.56 to −0.86 | **not held**: the association is *stronger* than predicted (partly mechanical: Jaccard 1 ⇒ Δ = 0) |
| H7 | random method median superiority in [0.35, 0.65] | 0.45–0.54 | **held** |
| H8 | A expressible; B and C not, without a unit-axes abstraction | the pre-change probe refused pixels, channels and tokens | **held** (confirmed before any change); ADR-034 |
| H9 | mixed SUPPORTS/CONTRADICTS cells exist, and a majority aggregate hides ≥ 10% | mixed cells 68–100% per site; hidden 20–29% | **held** |

**Negative and limiting results (kept):**

- **Circularity.** Under r1 (zero replacement) in the piecewise-linear A and B networks, the IxG, IG and ablation selections were identical at the hidden and channel sites (top-k Jaccard 1, identical drops). Their agreement with interventions under zero ablation is partly by construction, not independent evidence.
- **Relative vs absolute.** Beating controls is not the same as supporting the claim. IG's selections beat count-matched controls (median superiority ≥ 0.94 everywhere), but IG comprehensiveness SUPPORTS at t = 0.5 held for only 6/60 (A-input), 19/60 (A-hidden), 42/60 (B-pixels), 21/60 (B-channels), and 16/40 (C).
- **Stronger controls weaken the pixel result.** On B pixels, magnitude-matched controls cut superiority from 0.99 to 0.85 (IG) and from 0.98 to 0.83 (IxG) at p = 10%, and to 0.75/0.69 at p = 20%. In the worst single case (IG, sample 493, k = 13) superiority fell from 0.94 to 0.42.
- **Gradient.** On B pixels under zero replacement gradient is near chance at small k (superiority 0.43 at p = 1%), because it ranks black pixels, whose zeroing is a no-op (22 no-op top-k selections). Under the mean-image replacement it becomes the best attribution method (0.98 vs IG 0.91): the ordering reverses.
- **Dead ReLUs.** On A's hidden layer, 39 gradient top-k selections (sample × k, r1 comprehensiveness) consisted only of dead ReLU units: zero removal is an exact no-op, reported as INCONCLUSIVE.
- **Replacement flips.** The same IG selection changed outcome with the replacement in 88 (A-input), 71 (A-hidden), 131 (B-pixels), 73 (B-channels) and 7 (C) (sample, k) cases.
- **Redundancy in C.** In C, removing the top-2 tokens rarely halves the margin, while *retaining* only them keeps it: IG sufficiency drop 5.6% of the margin vs 97% for random retention. 972 of 975 disagreeing C cells are "not necessary but sufficient".
- **Stability (B, 1-pixel roll).** Median prediction change 2.56 logits (45% of samples change by more than half their margin); ranking ρ 0.95 but top-k Jaccard 0.40; claim outcome the same in 77%. 17 samples have IG–ablation agreement ≥ 0.5 but stability Jaccard ≤ 0.3.
- **Curves.** Removal and retention curves rank method pairs differently in 18% (A-input), 13% (A-hidden), 9% (B-pixels), 0% (B-channels) and 33% (C) of decided pairs.
- **Random SUPPORTS.** The seeded random method received comprehensiveness SUPPORTS 218 times (API review F-21).
- **Pre-registration errors.** The plan's predicted failure "the [CLS]/[SEP] embeddings dominate in C" did not occur: every attribution top-1 was a word token. H1b's premise did not hold in general.

**API failures found by the realistic runs (all four BUGs fixed in-phase, regression tests fail before the fixes):**

- the pre-change probe could not express pixels, channels or tokens (ADR-034);
- a false stochasticity refusal on the CNN (ADR-036);
- curve records dropped `unit_axes`, and a declared reduction was dropped for single-element units (ADR-034 fixes);
- diagnostics refused every BERT sample for lack of `model_kwargs` (ADR-037);
- magnitude re-derivation refused correct results at a conv site (ADR-038).

Open items are listed in `docs/PHASE_5_5_API_REVIEW.md`.

**Reproducibility:**

- Model B's re-run reproduced all 14,700 rows exactly, including content-derived result ids.
- C's first 3 sentences reproduced 315/315 rows at HEAD.
- Composition re-derived every first-sample result at every site: 630 claim-test results, plus curves and diagnostics.

**Mutation checks:** 20 mutations of the new code (unit mapping, reduction axes, strata, magnitudes, the evaluator's unit-axes checks, curve unit fields, tolerances, keyword inputs). **All caught.**

**Performance** (idle, one thread; `results/performance.json`):
- Forward pass: A 0.012 ms, B 0.027 ms, C 0.28 ms.
- One faithfulness test with N = 50: A 32–39 ms, B 33–39 ms, C 0.51 s.
- Controls scale linearly (C: N = 10, 50, 200 → 0.12, 0.51, 2.04 s).
- Dataset runtime (full grid, 3 or 4 concurrent runs on 8 cores): A 29 min, B 19 min, C 75 min.

## 2026-09-26: Phase 6 concepts and concept validation (pre-registered)

- **Commits:** `c48d30c` (literature + plan) through the Phase 6 report commit.
- **Pre-registration:** `docs/PHASE_6_PLAN.md`. §26 records the implementation names and the realistic-run operational details, written after the ground-truth run and before any realistic run.
- **Environment:**
  - Ground truth and tests: Python 3.14.3, torch 2.14.0.
  - Realistic runs: the Phase-5.5 experiment environment (Python 3.12.13, torch 2.14.0, transformers 5.17.0, scikit-learn 1.9.1); CPU, one thread.

**Ground truth** (hand-built `ConceptToy`; `experiments/phase6/results/ground_truth.json`; full pre-registered N): **every case behaved as pre-registered.**

| Case | Result |
|---|---|
| A: encoded and used | ENCODES AUROC 1.0; use −2.62; VALIDATED (neuron and direction) |
| B: decodable, unused | ENCODES AUROC 1.0 SUPPORTED; use effect **exactly 0**, CONTRADICTED; PROPOSED |
| C: distributed | neuron AUROC 0.83, use −0.82; direction AUROC 1.0, use −1.71 (≈ 2×) |
| D: redundant | neuron use 0, CONTRADICTED; spanning direction −0.82, VALIDATED |
| E: proxy | E1 (correlated) VALIDATED; E2 (independent) ENCODES CONTRADICTED (AUROC 0.53) |
| F: random directions, test n = 16 | naive AUROC ≥ 0.7 in 5/50; controlled ENCODES SUPPORTED in 2/50 |
| G: permuted labels | CONTRADICTED (AUROC 0.49) |
| H: polysemantic | AUROC 0.85; false-positive rate 0.26 (listed); **VALIDATED under v1**: no rate caps; the rates are shown |
| I: split | neuron false-negative rate 0.58, use CONTRADICTED; direction VALIDATED |
| J: generated labels | wrong label: GENERATED, ENCODES CONTRADICTED, PROPOSED; right label: VALIDATED only through `validate` |

**Realistic** (`experiments/phase6/results/realistic_{A,B,C}.json`): 9 concepts × (fitted direction, train-searched neuron) × 2 references.
- Encoding: covariance-matched random directions (N = 200) plus label permutation (N = 200) at 0.95.
- Use: removal with zero and with train-mean references; N = 50 matched controls; `min_change` 0.1 nats; 0.95.

| Hypothesis | Result | Verdict |
|---|---|---|
| R1: ENCODES for ≥ 7/9 directions | 3/9 directions (6/9 neurons) | **not held** |
| R2: a concept encoded but not used under both references | the neurons of K1, K2 and K4 (MLP), K8 and K9 (BERT) | held |
| R3: label-aligned K7 validated | not validated under either reference: the direction's AUROC 0.88 is **contradicted by the covariance null** (median 0.878) | **not held** |
| R4: the reference changes the use outcome | K1 (−2.99 zero vs +1.60 train-mean), K7, K9 directions | held |
| R5: the best neuron is worse than the direction for ≥ 6/9 | the neuron is worse in 2/9 | **not held** |
| R6: random directions pass a naive test but not the controlled one | naive AUROC ≥ 0.6 in 13 / 11 / 6 of 20; controlled 0/20 at every site | held |
| R7: the covariance null is stronger everywhere | stronger in 6/9 concepts; weaker for K4, K8, K9 | **not held** |

- **Validated:** only MLP K4 (fractal-dimension direction) is VALIDATED, under both references.
- **Decodable but unused:** natural cases include the BERT pooler neurons for length (AUROC 0.75) and negation (0.87), both ENCODES SUPPORTED, with use effects of −0.006 and −0.015 nats.
- **Wrong-direction effects:** 7 of the 8 CNN use effects were *positive* (the intervention raised the lift class's log-probability). The exception is the K5 neuron under zero removal (−0.47), which still failed its controls.

**Negative and limiting results (kept):**

- **The covariance null can be over-matched.** At the BERT pooler, the dominant variance *is* sentiment, so random covariance directions decode sentiment (median AUROC 0.878) almost as well as the fitted direction (0.881). ENCODES for the task label is contradicted.
- **Intervention semantics decide use.** On MLP K1 the same direction's removal lowers log p(malignant) by 2.99 nats with a zero reference and *raises* it by 1.60 with the train-mean reference. This was recomputed by hand and is exact.
- **v1 validates polysemantic features.** Case H is VALIDATED under v1 with a 26% false-positive rate (listed). Rate caps are available but not in v1.
- **Label text is not checked.** In the SAE case the GENERATED label ("responds to low mean perimeter") has the opposite polarity to the K1 dataset concept. BeyondNN tests the dataset extension, not the text.

**SAE case** (`sae_case.json`):
- A 64→128 SAE was trained locally on K1-train activations: validation explained variance 0.986; 0 dead latents; **L0 ≈ 69 of 128**. The L1 penalty (3e-3) was too weak, so this SAE is barely sparse. The case exercises the adapter and pipeline, not a realistic sparse dictionary.
- Latent 115 was chosen by search from 128 candidates (train AUROC 0.917).
- Its GENERATED label stays PROPOSED: ENCODES CONTRADICTED under the covariance null (test AUROC 0.866); use CONTRADICTED under both references.

**Framework failure found and fixed:** trace lookups were O(N²) at concept scale. A use test took 54 s, which fell to 6.6 s after the fix (ADR-043).

**Mutation checks:**
- 26 mutations of the Phase-6 code.
- First run: 24 caught; two survivors (`min_change` ignored; control-direction check disabled).
- Tests were added for both, and both are now caught: **26/26**.

**Performance** (idle, one thread; `performance.json`):
- MLP: encoding test with 400 controls 0.40 s; use test with 50 controls 2.8 s.
- BERT-tiny: encoding 8.1 s; use 79 s; composition with full re-derivation 1.3 s.
- Random-direction scaling from n = 50 to 800 adds under 0.2 s.

**Phase-5.5 compatibility:** re-running MLP sample 1 at HEAD reproduces all 280 rows' values, outcomes and statistics. Every record id differs because of Claim v3 and InterventionRecord v4.

## 2026-09-26: Phase 7 scientific audits (pre-registered)

- **Commits:** `6faca63` (literature + differentiation), `4786efc` (plan), then implementation through `a1b62ef`. Plan deviations D1–D13 and execution note E1 are listed in `docs/PHASE_7_PLAN.md`.
- **Environment:** `experiments/phase5_5/requirements.txt` pins, Python 3.12, one torch thread. No file under `beyondnn/` changed between `27d027c` and the last run.

**Scenarios A–N** (`scenarios.json`): 14/14 matched their pre-registration; the tampered A gave INTEGRITY_FAILURE with 1 result excluded.

**Central experiment** (`central_{A,B,C}.json`): 280 samples (60/60/60/60/40 over A/input, A/net.1, B/pixels, B/relu2, C/tokens); 20,112 re-derived results with 0 integrity failures; `verify_report` passed on all 5.
- The naive single configuration (IG, r1, p = 10%, count controls at 0.95, 0.5 · margin) SUPPORTS on 5/19/35/16/9 = 84/280.
- The audited per-sample IG_necessary is SUPPORTED on 0/0/0/0/1 = 1/280; ASSUMPTION_SENSITIVE on 55/60/54/59/33; CONTRADICTED on 5/0/6/1/6.
- R_necessary is CONTRADICTED on 50/43/23/29/33; ASSUMPTION_SENSITIVE on 10/17/37/31/7.
- The attribution-only audit is UNSUPPORTED for G and IG on 280/280 and NOT_EVALUATED for R.
- Hypotheses: CH1–CH6 held. CH1 holds essentially by construction: SUPPORTED is almost never reached under the declared invariances.

**Negative and limiting results (kept):**
- **Coarse standings.** ASSUMPTION_SENSITIVE covers both "supported in most configurations" and "supported in one"; `analysis.json` gives the descriptive supporting shares.
- **The cross-claim necessity/sufficiency check fired on 0 samples,** while configuration-level disagreements occurred on 22–52 IG samples per site (`analysis.json`).
- **Absolute pass, controls reject:** on B/pixels, B/relu2 and C/tokens, 11–18 IG and 3–21 R samples pass an uncontrolled removal test that their matched random controls reject.
- **D13:** the first central run declared one target for margin claims whose targets differ per sample. The audit left 22/60 (A/input) and 55/60 (B/pixels) NOT_EVALUATED. The run was discarded and repeated with per-sample targets.
- **E1:** concurrent runs thrashed memory; three runs were stopped and restarted.

**NLP** (`nlp.json`, descriptive):
- IG top-k selections never included [CLS] / [SEP].
- The top-1 tokens were sentiment-bearing words.
- Replacement (token-removal) sensitivity appeared on 11/40 samples.

**Concept audits** (`concepts_{A,B,C}.json`):
- The Phase-6 hand analysis was reproduced (KH1–KH4, NH1–NH2).
- No concept is SUPPORTED under the strict caps (KH5). K4-direction is UNSUPPORTED under strict (FP 0.277) and SUPPORTED under lenient.
- Concept A's audit reloaded from 76 saved traces was byte-identical.

**Reload** (`performance.json`): in a fresh process, 54/54 per-sample groups (the first 3 samples of 3 sites) were equal to the in-memory full audits.

**Mutations** (`mutations.json`): 25 code mutations. 6 were not killed in the first run (5 survivors plus 1 pattern that did not apply). 5 tests were added; the final count is 25/25 killed.

**Performance:** full central audits took 36–68 s for 2,832–4,320 results (about 12–16 ms per result), and `verify_report` costs the same again. The evidence itself is about 0.5 MB per test trace on disk (0.9 MB for BERT).

**Compatibility:**
- A Phase-5.5 re-run of 2 MLP samples reproduces 560/560 rows (outcomes and drops).
- A Phase-6 ground-truth re-run reproduces every outcome and status. One claim id in `J-wrong`'s `unmet` text differs from the committed Phase-6 JSON; it is identical at `25fe462` and at HEAD (pre-existing: generated-label provenance includes the environment).

## 2026-09-27: Phase 7.5 external validation, NLP expansion, API freeze (pre-registered, frozen policy `ee3f91e`)

**Setup:** see `docs/PHASE_7_5_PLAN.md` and `docs/PHASE_7_5_FROZEN_POLICY.md`; results in `experiments/phase7_5/results/`, evaluated by `evaluate75.py` into `hypotheses75.json`.

**External, InterpBench** (18 held-out SIIT models; case 124 re-run after DV-1):
- PRIMARY: TP 942/942, FP 0/8,840, 18 ambiguous (all non-circuit nodes the trained model uses), 9 INCONCLUSIVE (resample no-ops, case 110).
- IG top-1 wrong-head selections CONTRADICTED 1,571/1,571; attribution-only IG selection claims UNSUPPORTED 1,960/1,960.
- Zero ablation FP 13.0%; the count null rejects 98.6% of necessary heads (ALTERNATIVE, predicted).
- EH1–EH6 held.
- **Negative finding (analysis time, no rule changed):** EH1/EH2 are largely by construction. PRIMARY uses the ground truth's own resample semantics, and "clear" is defined by agreement with that resample.

**Concepts, Tracr** (held-out case 39): known negatives are never validated (KE1). The known-used variable is **not** validated either (use effect about 0.06 < 0.1; KE2 held as predicted; KE3: SIIT likewise).

**Central held-out** (A/B/C; D pending in this entry, see the report):
- IG_necessary PRIMARY SUPPORTED: 8/51, 4/51, 14/60, 1/60, 14/40; `alternative_reverses` on 38/41 of them.
- R ≤ 1 per site. CH7 held; CH8 and CH9 held on A/B/C.

**NLP:**
- **e-SNLI (BERT-base SNLI, 40):** human necessary 10/40, IG 7/40, random 3/40 (N1 failed); IG − random F1 +0.149, Bonferroni [0.031, 0.270] (N2 held).
- **OOD:** zero − [MASK] percentile: D −5.5 (N3 failed; reversed), E +1.5 (held).
- **Shortcuts and padding:** empty-premise accuracy 0.456 (N4 held). Without the attention mask, padding flips D on 13/40; word shuffles keep the SST-2 predictions on 83–94% of shuffles.

**Engineering:**
- The external-researcher workflow first failed (3 blocking API defects), then passed after the fixes.
- 24/24 mutations killed.
- The regression matrix is green: 1014 + 1 / 1035 tests on 3.10 / 3.12 / 3.14 (1015 + 1 / 1036 after the Interval fix).
- Re-runs of Phase 5.5 / 6 / 7 experiments reproduce.
- BERT-base evidence costs about 10–19 min and about 225 MB per sample on this CPU.

**Operational:**
- Model D's 40-sample held-out run takes many hours. The owner chose to keep the frozen N = 40 (no deviation).
- An earlier ETA estimate was wrong, because the BERT-base per-sample cost was unknown until measured.
