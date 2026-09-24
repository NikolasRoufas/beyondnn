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
- ADR-016 to ADR-018 (proposed): record identity, status mechanism, assessment policies.
- Design documentation: ecosystem audit, differentiation, interpretability definition, architecture proposal,
  trace schema proposal, roadmap, Phase 1 plan, research questions, experiment log.
- Architecture decision records ADR-001 to ADR-015 (ADR-012 claims, ADR-013 estimand scope, ADR-014 incremental schema versioning, ADR-015 single-execution traces).
- Project scaffolding: `pyproject.toml`, CI workflow, pre-commit configuration, package placeholder.
