# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project will use
[Semantic Versioning](https://semver.org/) once released. The trace `schema_version` is versioned separately
(see `docs/design/TRACE_SCHEMA_PROPOSAL.md`).

## [Unreleased]

### Added
- **M1.1: trace schema 0.1** (`beyondnn.schema`):
  - immutable records with content-derived ids;
  - evidence status bound to record kind, with explicit derivation rules;
  - `InputRecord`, `OutputRecord`, `ActivationRecord`, `TensorRef`, `TraceLimitation`;
  - `Claim`, `ClaimTestSpec`, `ClaimTestResult`, and derived `Assessment` under explicit policies;
  - estimand scopes (instance / finite sample / population);
  - a strict, versioned JSON envelope.
- ADR-016 to ADR-018, accepted after review: record identity (identity ≠ semantic equivalence), status mechanism (measured state under intervention stays MEASURED; INTERVENTIONAL is reserved for effects), assessment policies (no universal thresholds).

- **M1.2: provenance and model fingerprinting**:
  - schema types `ModelIdentity`, `EnvironmentIdentity`, `ExecutionContext` (CLEAN/INTERVENTION), `Randomness`, `MethodIdentity`, `ProvenanceRecord`, `ExecutionOccurrence`;
  - `beyondnn.provenance` with the FULL v1 model fingerprint (structure plus bitwise state, tied-parameter and shared-module topology), environment collection without machine identity, read-only CPU RNG digest, and provenance and occurrence builders;
  - ADR-019 and ADR-020;
  - ADR-016 amendment: expectation wording and the no-Unicode-normalisation decision.

- **M1.10: Phase 1 audit and gate report** (`docs/PHASE_1_REPORT.md`: GO WITH EXPLICIT LIMITATIONS).
  - Public recording now refuses module replacement or structural change between passes (ADR-027).
  - Audit tests (`tests/test_phase1_audit.py`) guard the public surface and the Phase-1 evidence statuses.
  - README corrected to state that only observed and measured evidence exist today.
  - Installed-wheel smoke test run in a clean virtualenv.
- **M1.9: `benchmarks/bench_trace_overhead.py`**: CPU characterisation of baseline, trace none/summary/cpu, fingerprint, save/load, and exact byte sizes. Results are in the experiment log. There is no pass/fail threshold and no optimisation.
- **M1.8: `bnn.instrument(model)`**: a frozen handle referencing the original model.
  - `handle.trace`/`recording` are exactly the public pipeline.
  - `handle.explain(...)` gives `ExplainResponse` (INPUT → WHY → OUTPUT) over one single-pass trace. Phase-1 `why` is measured evidence (activations, limitations, provenance) with `NO_ATTRIBUTION`/`NO_CAUSAL_EVIDENCE`/`NO_CLAIMS_TESTED`, never a causal explanation.
  - A deterministic `render()`.
  - README examples are executed by the test suite. ADR-026.
- **Trace occurrence and hook provenance fixes (ADR-025):**
  - `ExecutionOccurrence` v2 carries `pass_index`, one per pass, with v1 migrating to `None`.
  - Public `trace()`/`recording()` refuse foreign forward/forward-pre hooks (local or global), checked at entry and at every pass start/end.
  - Models whose tensors are on several devices are refused instead of being labelled with one device.
- **M1.7: trace persistence:** `TraceResult.save(dir)` and `bnn.load_trace(dir)`.
  - Deterministic `trace.json` with an optional `tensors.pt`, read only with `weights_only=True`. No paths are taken from JSON, and symlinks are refused.
  - Atomic save; the target is never overwritten.
  - Full re-validation on load: ids, references, provenance, and tensor dtype/shape/digest.
  - Migrations with deterministic reference remapping.
  - The codec now verifies stored ids before migrating. ADR-024.
- **M1.6: trace pipeline:** public `bnn.trace`, `bnn.recording`, `bnn.TraceResult`.
  - Per-pass provenance, with the model fingerprinted at the start of each root call.
  - OBSERVED root input/output and MEASURED activations, one per tensor leaf, with deterministic paths.
  - Retention `summary`/`cpu`/`none`; no live tensors are retained.
  - Container validation of every record and reference.
  - Refusal of out-of-pass and aliased executions; honest limitations (`SELECTED_SITE_NOT_EXECUTED`, `NON_TENSOR_LEAVES_IGNORED`).
  - `InputRecord`/`OutputRecord` v2 (`pass_index`). ADR-023.
- **M1.5 fix: recursive pre-hook invocation bookkeeping.** A prepended guard pre-hook claims each attempt's call index, and the observation pre-hook (after user pre-hooks) emits INPUT. A user pre-hook raising inside recursion can no longer release the enclosing frame. Sessions refuse to start while global module forward pre-hooks are registered.
- **Caller-declared model provenance** (M1.6 gate, ADR-022):
  - `ModelDeclaration` (config, implementation revision, checkpoint revision) in `ProvenanceRecord.declared_model`;
  - `ProvenanceRecord` record_version 2, with a v1→v2 migration;
  - `make_provenance(declared_model=…)`;
  - migration errors now raise `DecodeError`.

  Resolves the M1.3 `n_heads` identity gap without changing FULL v1.
- **M1.5: safe hook session** (`beyondnn.core.hooks`, internal):
  - scoped hooks, removed on every exit path (normal exit, exceptions, sink errors, partial install);
  - physical-module dedup;
  - pass/call indices with failure consumption, out-of-pass `-1`, and stack-paired INPUT/OUTPUT;
  - aliased modules refused by default (internal `GROUP` mode);
  - values neither modified nor retained;
  - autograd, outputs, and BatchNorm state identical to baseline. ADR-021.
- **M1.4: deterministic module site resolution** in `beyondnn.core.sites` (not exported):
  - segment-based `*` / `**` pattern language; the root is selected only by `""`;
  - strict (unmatched or invalid patterns raise);
  - traversal-order results, deduplicated by exact path;
  - alias paths preserved as distinct sites;
  - no forward, hooks, or RNG use.

  Also: a hard roadmap gate before M1.6 for caller-declared model config/revision.
- **M1.3: deterministic tiny reference models** in the internal, unstable `beyondnn._testing.models`:
  - `TinyMLP` (139 params; shared module called twice; functional op; keyword-only `scale`);
  - `TinyCNN` (396; BatchNorm buffers; tuple output);
  - `TinyTransformer` (5,120; tied `lm_head`/`token_embedding`; non-persistent `position_ids`/`causal_mask` used in forward; dict output).

  Construction is seeded without changing the caller's global RNG. The models are not exported from `beyondnn`.

### Fixed (M1.2 review)
- The FULL fingerprint now hashes the values of **all** registered buffers, including non-persistent ones, which can affect `forward`. Buffer persistence is recorded as metadata. The v1 golden digests were updated before any release (ADR-020 correction).
- Access to PyTorch's private buffer-persistence field is isolated in one helper, with a compatibility test.
- Docs: the definition of FULL (Python code is not hashed); `schema` is stdlib-only; an implementation-revision roadmap item.

### Changed (M1.1 review fixes)
- `ActivationRecord` / `MEASURED` semantics: directly observed state, including during intervened executions.
- `derived_from` and `ClaimTestResult.evidence` are sorted at construction, and `-0.0` is normalised, so ids no longer depend on order or on the sign of zero.
- The codec no longer imports the schema package from inside itself (the apparent import cycle is removed). Dead code is removed.
- Second, independent guard tests added for derivation rules, population matching, policy coverage, and id integrity.
- Design documentation: ecosystem audit, differentiation, interpretability definition, architecture proposal,
  trace schema proposal, roadmap, Phase 1 plan, research questions, experiment log.
- Architecture decision records ADR-001 to ADR-015 (ADR-012 claims, ADR-013 estimand scope, ADR-014 incremental schema versioning, ADR-015 single-execution traces).
- Project scaffolding: `pyproject.toml`, CI workflow, pre-commit configuration, package placeholder.
