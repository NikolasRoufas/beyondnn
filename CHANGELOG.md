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
