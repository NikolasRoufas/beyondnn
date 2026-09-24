# Contributing to BeyondNN

BeyondNN is in its design phase. The most useful contributions right now are critical reviews of the
documents in `docs/design/` and `docs/research/`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your platform's build
pip install -e ".[dev]"
pre-commit install
pytest
```

## Ground rules

These are scientific rules, and they are enforced in review.

1. **Never upgrade epistemic status.** Code must not relabel a result as stronger evidence than the
   method that produced it: attribution is not causal, a probe is not use, and a label is not a validated concept.
2. **Causal language requires an intervention.** "Causes", "necessary", and "sufficient" only appear for
   `INTERVENTIONAL` / `ESTIMATED_CAUSAL` evidence, and that includes docstrings and render templates.
3. **No undefined numbers.** Every new metric or test needs a formal definition, an implementation, tests,
   documentation, an interpretation, and its limitations, all in the same PR.
4. **Limitations are part of the output.** If a method has a known failure mode, it must emit a
   `TraceLimitation`.
5. **Negative results stay.** `docs/experiments/EXPERIMENT_LOG.md` is append-only.
6. **Architecture-independent core.** No transformer- (or CNN-) specific code in `beyondnn/schema` or
   `beyondnn/core`.
7. **Hooks never leak.** Every test runs under a leak-checking fixture. Don't disable it.

## Design changes

Significant design changes need an ADR. Append one to `docs/decisions/ARCHITECTURE_DECISIONS.md` (never edit
an accepted ADR; supersede it instead).

## Pull requests

- Keep PRs small and focused. Include tests.
- `ruff check`, `ruff format --check`, `mypy`, and `pytest` must pass.
- Update `CHANGELOG.md` under *Unreleased*.
- Avoid test helper classes named `Test*` unless they are pytest test classes.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
