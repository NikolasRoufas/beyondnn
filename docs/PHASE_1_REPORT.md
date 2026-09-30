# Phase 1 Gate Report: Foundations

- **Date:** 2026-09-25
- **Scope:** milestones M1.0–M1.10.
- **Repository state:** local only, never pushed.
- **Reviewer inputs:** `docs/decisions/ARCHITECTURE_DECISIONS.md` (ADR-001–027), `docs/experiments/EXPERIMENT_LOG.md`, `docs/roadmap/PHASE_1_PLAN.md`.

## 1. Executive summary

Phase 1 set out to deliver a trustworthy *evidence-capture* foundation: typed, provenance-bearing, validated records of what a PyTorch model observably did, with honest limits. That is implemented and tested:
- the schema;
- provenance and fingerprinting;
- site resolution and hook sessions;
- trace recording and persistence;
- a minimal INPUT → WHY → OUTPUT view.

It passes on Python 3.10, 3.12 and 3.14 with no known silent evidence misattribution. Phase-1 WHY is **measured evidence only**: it makes no attributional, causal, or conceptual claims.

**Gate decision: GO WITH EXPLICIT LIMITATIONS** (§19).

## 2. What was implemented

| Milestone | Deliverable | Key decisions |
|---|---|---|
| M1.1 | Trace schema 0.1: immutable records, content-derived ids, unordered epistemic status with derivation rules, claims/tests/assessments, estimand scopes, strict versioned codec | ADR-012–018 |
| M1.2 | Provenance (model/environment/execution/method), FULL v1 model fingerprint (all buffers), caller-declared model context | ADR-019, 020, 022 |
| M1.3 | Internal deterministic reference models (`beyondnn._testing`) | — |
| M1.4 | Strict segment-based site resolution | — |
| M1.5 | `HookSession`: physical-module hooks, pass/call indices, invocation guard, alias refusal, no retention | ADR-021 |
| M1.6 | `trace()`, `recording()`, `TraceResult` with full record/reference validation; per-pass provenance | ADR-023, 025 |
| M1.7 | Persistence: `trace.json` + `tensors.pt` (weights_only), atomic save, migration with reference remapping | ADR-024 |
| M1.8 | `instrument()` handle, `ExplainResponse`/`Why` view | ADR-026 |
| M1.9 | CPU overhead characterisation | — |
| M1.10 | Audit (this report), refusal of structural change during recording | ADR-027 |

**Size:** about 3,350 package code lines, 4,550 test code lines, and 638 tests (docstrings and comments excluded).

## 3. Public API (audited)

| Name | Why public | Stable for pre-alpha? | Documented | Tested |
|---|---|---|---|---|
| `trace` | primary one-shot capture | yes | README, ADR-023 | `test_trace.py` |
| `recording` | multi-pass capture | yes | README, ADR-023 | `test_trace.py` |
| `TraceResult` | the result type users read and save | yes | schema doc §A.11 | `test_trace.py`, `test_persistence.py` |
| `load_trace` | persistence counterpart of `TraceResult.save` | yes | ADR-024 | `test_persistence.py` |
| `instrument` | Phase-1 INPUT→WHY→OUTPUT entry point | yes (the surface is minimal on purpose) | README, ADR-026 | `test_explain.py` |
| `EvidenceStatus`, `Relation`, `Outcome`, `Verdict`, `EstimandScope` | vocabulary users filter on | yes | schema doc | schema tests |
| `schema` | the pure-data package | yes | schema doc | schema tests |
| `__version__` | standard | yes | — | `test_package.py` |

- **Not public:** `HookSession`, `HookEvent`, `resolve_sites`, the tensor walkers, persistence internals, and fingerprint internals. They are importable from submodules only; `test_package.py` and `test_phase1_audit.py` guard this.
- **No accidental exports were found.**
- `import beyondnn` does not import torch; the tracing names load lazily.

## 4. Architecture

- **`beyondnn.schema`**: stdlib only (verified by subprocess test).
- **`beyondnn.provenance`**: torch-dependent collection.
- **`beyondnn.core`**: sites, hooks, trace, tensors, persistence.
- **`beyondnn.explain`**: presentation.

Dependencies point one way: schema ← provenance ← core ← explain. The trace is the single source of truth; `Why` is a view over it, and there is no second store and no second capture engine.

## 5. Scientific invariants (enforced in code, mutation-checked)

1. Evidence status is never caller-supplied, and cannot be upgraded. `EvidenceStatus` is unordered, and derivation rules are explicit.
2. GENERATED content can never be cited as evidence.
3. A decisive result on a causal relation requires INTERVENTIONAL/ESTIMATED_CAUSAL evidence about the claim's estimand. Finite-sample evidence cannot decide a population claim.
4. Assessments are derived. A stored verdict that doesn't follow from its results is rejected, and there is no confidence number.
5. Hook evidence proves only that "this module object executed and this value crossed its boundary". Aliased modules and out-of-pass executions are refused rather than attributed.
6. Measured state under an intervention would still be MEASURED. INTERVENTIONAL is reserved for effects (ADR-017; no interventions exist yet).

## 6. Epistemic-status guarantees

**Phase 1 produces only OBSERVED (root input/output) and MEASURED (activations) evidence.** Records without a status are provenance, occurrences, and limitations. This is verified three ways:
- programmatically on all reference models (`test_no_stronger_evidence_status_is_ever_produced`);
- statically: every registered kind's status is OBSERVED, MEASURED, or none, and no kind overrides `status`;
- by source scan: `core`/`explain`/`provenance`/`_testing` never name a stronger status (`test_phase1_audit.py`).

The deterministic `render()` returns plain text. It is not a record and carries no status.

## 7. Provenance guarantees

- **Per root pass**, the provenance record holds:
  - `ModelIdentity`: FULL v1 fingerprint at the *start of that pass*;
  - `EnvironmentIdentity`: Python, torch, BeyondNN, OS family, architecture, with no host, user, or path;
  - `ExecutionContext`: CLEAN, device, `model.training`, grad mode, declared randomness;
  - `MethodIdentity` `forward_hook` v1;
  - the optional caller `ModelDeclaration`.
- State changes between passes give new provenance (tested with BatchNorm).
- Timestamps live only in `ExecutionOccurrence`, one per pass (`pass_index`), and never affect `provenance_id`.
- Public tracing refuses foreign forward/pre hooks (local or global; checked at entry and at every pass start and end), multi-device models, and module replacement or structural change during a recording (ADR-025, 027).

## 8. Trace semantics

- One `InputRecord`/`OutputRecord` (OBSERVED) per pass, plus one `ActivationRecord` (MEASURED) per tensor leaf per selected site invocation, in execution order.
- Deterministic leaf paths (`""`, `[i]`, `["key"]`, with `args`/`kwargs`/`output` prefixes).
- Retention is none, summary, or cpu, and never keeps live tensors or graphs.
- `trace()` is exactly `recording()` plus one call.
- **Limitations emitted by traces:** `FUNCTIONAL_OPS_UNOBSERVED` (always), `PARTIAL_SITE_COVERAGE`, `SELECTED_SITE_NOT_EXECUTED`, `NON_TENSOR_LEAVES_IGNORED`.
- **Explanations** add `NO_ATTRIBUTION`, `NO_CAUSAL_EVIDENCE`, `NO_CLAIMS_TESTED`.

## 9. Persistence and security guarantees

- **Fixed file names**, and no paths read from JSON. Symlinks are refused.
- **The sidecar** is only `torch.load(weights_only=True)`, must be a plain `{key: Tensor}` dict, and is checked for dtype, shape, and SHA-256 content digest.
- **Every record is re-validated on load:** ids verified against the payload's own version before migration, then all references and provenance links.
- **Migrations** remap every reference to changed ids, and the remapping cascades.
- **Saving is atomic:** a temporary directory plus one rename. Existing targets are never overwritten.
- **Corruption matrix:** 20+ corruptions are all rejected (see the experiment log). With the `weights_only=False` mutation, a malicious pickle actually executed, which confirms the guard is load-bearing.

## 10. Test matrix (final)

| Environment | Result |
|---|---|
| Python 3.14.3 / torch 2.12.0 (macOS arm64) | **638 passed** (`-W error`, 0 skipped, 0 xfail) |
| Python 3.10 / torch 2.14.0 | 638 passed |
| Python 3.12 / torch 2.14.0 | 638 passed |
| ruff check / ruff format | clean |
| mypy `--strict` (package + tests) | clean |
| sdist + wheel build | OK |
| Clean venv (Python 3.12, torch 2.14) install of the built wheel, run outside the checkout with `-W error` | smoke test OK: trace, recording, save/load, instrument/explain; import resolved to site-packages |
| README runnable example | executed by `tests/test_readme.py` and in the clean venv: OK |

## 11. Mutation-test summary

Every required guard was broken in a scratch copy and was caught by at least one test. Highlights:
- **schema:** causal-evidence check (16 tests); GENERATED rejection; assessment recomputation; derivation rules; population matching (≥ 2 each);
- **fingerprint:** buffer coverage (5); dtype/shape/aliasing/persistence guards;
- **sites:** 7/7 guards;
- **hooks:** 11/11, plus the invocation guard;
- **trace:** 7/7, plus retention with a live context (initially missed, then fixed);
- **persistence:** 6/6;
- **final checks:** 8/8 (occurrence `pass_index`, external hooks, causal labelling, missing `NO_CAUSAL_EVIDENCE`, model replacement by `instrument`, divergent `handle.trace`, copied WHY records, multi-device), plus structure and binding guards (2/2).

**Mutations that were initially missed** are logged honestly: `-0.0` normalisation, live-context retention, and the module-alias group. Each got a stronger test.

## 12. Performance benchmark (M1.9; this machine only)

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

- **Fixed cost:** about 0.8–1.3 ms per trace on tiny models, a third of it per-record integrity hashing.
- **Fingerprint:** 1.8–1.9 ms per ~1M float32 parameters, 58–68% of the 1M model's overhead.
- **Retained bytes:** exactly what `cpu` retention stores.
- **Characterisation only:** no thresholds.

## 13. Known limitations (surfaced as limitations, errors, or documentation)

- **Functional ops** (`F.gelu`, residual `+`, SDPA internals) are unobserved (`FUNCTIONAL_OPS_UNOBSERVED`).
- **FULL v1 does not hash Python code or plain attributes.** `n_heads=2` vs `n_heads=4` fingerprint equal; use `ModelDeclaration`.
- **Structured values:** only Tensor/tuple/list/dict are decomposed. Dataclasses, custom output classes, and custom pytree nodes count as non-tensor leaves. Namedtuples are handled as tuples, so field names are lost.
- **Sidecars** are loaded fully into memory. There is no sharding, mmap, or lazy loading.
- **Hardware:** CPU is the only verified device. One execution device per model; several are refused.
- **User forward hooks** must be removed before public tracing. A hook added *and* removed inside a single forward call cannot be detected.
- **Unsupported:** re-entrant root calls, out-of-pass selected executions, aliased selected modules, structural change during a recording, `torch.compile`, and concurrent threads.
- **Old records:** v1 input/output/occurrence records migrate with `pass_index = None`, so their pass association is unknown.
- **Fingerprint dependencies:** it depends on qualified class paths (moving a class changes structure identity) and on the private `nn.Module._non_persistent_buffers_set`.

## 14. Unsupported cases (explicit errors)

- `OutOfPassExecutionError`;
- `AliasSiteAmbiguityError`;
- `ExternalForwardHooksError`;
- `UnsupportedExecutionError` (several devices; replaced module; structural change);
- `RecordingError` (failed or partial recording, no root call, re-entrant root);
- `HookSessionError` (global pre-hooks);
- `FingerprintError` (meta, sparse, quantized, nested tensors, tensor subclasses, extra state, big-endian);
- `TracePersistenceError`.

## 15. Scientific risks

1. **Users may still read MEASURED activations as reasons.** Mitigated by the `NO_*` limitations, the render header, and README warnings, but not preventable in code.
2. **Coverage is module-boundary coverage.** Traces with every module selected still miss functional computation.
3. **Identity stops at registered tensors and class paths.** Behaviour hidden in code or attributes is invisible without caller declarations.
4. **Phase 2 is where causal claims enter.** The schema invariants (ADR-012/013/017/018) are untested against real intervention runners until then.

## 16. Engineering risks

1. **Per-record hashing overhead** grows with the number of records. Fingerprinting per pass scales with model bytes. Neither has been measured on large models.
2. **Private torch internals are relied upon:** `_non_persistent_buffers_set`, `_global_forward(_pre)_hooks`, and hook dicts. There are compatibility tests on torch 2.12 and 2.14 only.
3. **CI has never run on GitHub** (release blocker RB-3). Only local multi-version runs exist.
4. **The first three commits** carry a co-author trailer; history was left unchanged at the owner's request.

## 17. Deferred work

- Interventions and `CausalEffect` records;
- claim-test runners and the protocol registry;
- `Study` containers;
- attribution (Captum adapter);
- features and concepts;
- `audit()`;
- performance work (redundant hashing, batched statistics);
- large-sidecar handling;
- non-CPU verification;
- representing user-hook-modified computation in provenance.

## 18. Phase 2 prerequisites

1. Owner review of this report and ADR-023–027.
2. The Phase 2 design: intervention specs and records (zero/mean/constant ablation, activation replacement/patching), how an intervention is represented in `ExecutionContext` (`INTERVENTION` + `intervention_id`, already in the schema), the `CausalEffect` record with estimand-derived status (ADR-013), a protocol registry that validates `AssessmentPolicy` protocol claims (ADR-018 gap), and `Study` (ADR-015).
3. The ground-truth synthetic suite (A causal / B correlated / C noise), with expected results written *before* running.

## 19. Final gate decision

**GO WITH EXPLICIT LIMITATIONS.**

| Criterion | Status |
|---|---|
| All validation green (3.10 / 3.12 / 3.14, ruff, mypy, build) | ✅ |
| README example runs | ✅ (test + clean venv) |
| Clean installed-wheel smoke test | ✅ |
| No known silent evidence misattribution | ✅ Known paths are refused or flagged. The last one found in this audit (module replaced mid-recording) was fixed. The remaining undetectable case (a hook added and removed within one forward) is documented. |
| No live graph retention | ✅ (weakref tests, live context) |
| Record/reference integrity enforced | ✅ (container + load) |
| Safe persistence load path | ✅ |
| Per-pass provenance correct | ✅ |
| Public trace refuses unrepresented external hooks | ✅ |
| Known limitations surfaced/documented | ✅ (§13) |
| Public API coherent | ✅ (§3) |

**Why "with explicit limitations" and not plain GO:**
- §13 contains real scope boundaries that a Phase-2 user will hit: CPU-only, module-boundary coverage, no code identity, and no user hooks during tracing.
- CI has not run remotely.

None of these is a silent correctness failure.

**PHASE 1 COMPLETE.** Phase completion is **not** release approval. The public-release blockers remain:
- **RB-1:** Code of Conduct contact placeholder;
- **RB-2:** security contact placeholder;
- **RB-3:** CI never run on GitHub.

Nothing has been pushed or published.
