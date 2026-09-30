## What this changes

<!-- A short description of the change and why. -->

## Checklist

- [ ] Tests added or updated; `uv run pytest -q`, `ruff check`, `ruff format --check` and `mypy` pass
- [ ] Documentation updated (README / `docs/` / docstrings) where behaviour changed
- [ ] `CHANGELOG.md` updated under *Unreleased*
- [ ] Scientific invariants preserved ([`docs/PRE_PHASE8_INVARIANTS.md`](../docs/PRE_PHASE8_INVARIANTS.md)); no protecting test was weakened
- [ ] No change to public API semantics, or an ADR is included
- [ ] If a record schema changed: version bumped, migration registered, `tests/test_migration_matrix.py` updated
