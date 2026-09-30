# Contributing to BeyondNN

Thanks for your interest. BeyondNN is research software whose value depends on *not* overclaiming. Contributions are welcome, and the rules below keep that property intact.

## Development setup

BeyondNN uses [uv](https://docs.astral.sh/uv/). `uv.lock` pins the development environment.

```bash
git clone https://github.com/NikolasRoufas/beyondnn.git
cd beyondnn
uv sync                       # installs the package and the dev group (pytest, ruff, mypy, pre-commit)
uv run pre-commit install     # optional: run ruff on every commit
```

Optional Captum adapter: `uv sync --extra captum`.

## Checks (all must pass; CI runs them)

```bash
uv run pytest -q                   # the full suite, a few minutes on CPU
uv run ruff check .
uv run ruff format --check .
uv run mypy                        # --strict, configured in pyproject.toml
```

Useful subsets:
- `tests/test_golden_workflow.py`: the permanent end-to-end scientific workflow;
- `tests/test_migration_matrix.py`: record migrations;
- `tests/test_readme.py` and `tests/test_examples.py`: the README code and `examples/`.

## Scientific rules (enforced in review)

1. **Never upgrade epistemic status.** Attribution is not causal, a probe is not use, and a label is not a validated concept. Code, docstrings and rendered text must not relabel a result as stronger evidence than the method that produced it.
2. **Causal language requires an intervention.** "Causes", "necessary" and "sufficient" appear only for INTERVENTIONAL (or ESTIMATED_CAUSAL) evidence.
3. **No undefined numbers.** A new metric or test needs a formal definition, an implementation, tests, documentation, an interpretation and its limitations, all in the same PR.
4. **No global score.** No trust, confidence, robustness or interpretability number, anywhere.
5. **Limitations are part of the output.** A known failure mode must emit a `TraceLimitation` or an audit finding.
6. **Negative results stay.** `docs/experiments/EXPERIMENT_LOG.md` and the phase reports are append-only records.
7. **Architecture-independent core.** No transformer- or CNN-specific code in `beyondnn/schema` or `beyondnn/core`. For example, special tokens are declared by the caller, never hard-coded.
8. **Hooks never leak.** Every test runs under a leak-checking fixture. Don't disable it.

## Scientific invariants and the API freeze

- **Invariants:** [`docs/PRE_PHASE8_INVARIANTS.md`](docs/PRE_PHASE8_INVARIANTS.md) lists the invariants that ordinary maintenance must preserve, each mapped to the tests that protect it.
- **API freeze:** [`docs/API_FREEZE.md`](docs/API_FREEZE.md) lists what is frozen, experimental or internal.

> **A change that requires editing a scientific invariant, or a test that protects one, is not ordinary maintenance. It needs explicit scientific review and an ADR.**

## Architecture decisions

- **When one is needed:** significant design or semantic changes need an ADR, appended to [`docs/decisions/ARCHITECTURE_DECISIONS.md`](docs/decisions/ARCHITECTURE_DECISIONS.md).
- **Accepted ADRs** are never edited. Supersede them with a new one.

## Adding a protocol

- **The protocol itself:** register it in `beyondnn/protocols.py` (name, the relations it can justify, a version). Implement it so its result can be **re-derived** from recorded evidence (the audit re-derives every result).
- **Documentation:** add a page under `docs/protocols/` covering the definition, what it can and cannot support, failure modes and limitations.
- **Tests:** add tests, including a scenario where the protocol must *not* support a claim.

## Changing a record schema

- **Version and migration:** bump the record's version in `@record_kind(..., version=N)` and register a migration from N−1 with `@register_migration`. Old records must keep their **old meaning**: new fields default to "unknown", "every unit" or "undeclared", and nothing is invented.
- **Tests:** add the migration to `tests/test_migration_matrix.py`. It fails if a registered migration is not covered.

## Documentation

- User-facing guides live in `docs/` (see [`docs/README.md`](docs/README.md)).
- Runnable README blocks start with `# runnable example` and are executed by `tests/test_readme.py`.
- Examples in `examples/` are executed by `tests/test_examples.py`. They must use the public API only.

## Pull requests

- Keep PRs small and focused, with tests.
- All checks above must pass.
- Update `CHANGELOG.md` under *Unreleased*.
- Commit messages describe the change in plain, professional language.
- Avoid test helper classes named `Test*` unless they are pytest test classes.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
