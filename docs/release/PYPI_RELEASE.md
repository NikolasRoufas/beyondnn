# Publishing a release to PyPI

The PyPI Trusted Publisher configured for this project:

```text
PyPI Project Name: beyondnn
Owner: NikolasRoufas
Repository name: beyondnn
Workflow name: publish.yml
Environment name: pypi
```

Publication is a **deliberate, manual** maintainer action:
- **Manual only:** `.github/workflows/publish.yml` runs only when triggered by hand (`workflow_dispatch`). Pushing a tag or creating a GitHub release never publishes.
- **No credentials:** publication uses PyPI Trusted Publishing (GitHub OIDC). No password, API token or GitHub secret exists for PyPI.
- **Immutable:** a version uploaded to PyPI cannot be replaced or re-uploaded. Review everything before triggering the workflow.

## Released versions

| version | tag | GitHub release | PyPI | authentication | clean PyPI install |
|---|---|---|---|---|---|
| 0.1.0 | `v0.1.0` | [published](https://github.com/NikolasRoufas/beyondnn/releases/tag/v0.1.0) | [published](https://pypi.org/project/beyondnn/0.1.0/) | Trusted Publishing (GitHub OIDC) | verified (Python 3.14, macOS; README examples and `examples/` pass) |

## Releasing version X.Y.Z

Below, `X.Y.Z` is the new version and `vX.Y.Z` its tag.

### Prepare

1. **Set the version** in `beyondnn/__init__.py` (`__version__ = "X.Y.Z"`; the single source).
2. **Update the release text:**
   - `CHANGELOG.md`: move the `[Unreleased]` entries to `## [X.Y.Z] - YYYY-MM-DD`;
   - `CITATION.cff`: `version` and `date-released`;
   - the pinned GitHub install line in `README.md` (`@vX.Y.Z`).
3. **Validate locally:**

   ```bash
   uv run pytest -q -W error
   uv run ruff check . && uv run ruff format --check .
   uv run mypy
   rm -rf dist && uv build && uvx twine check --strict dist/*
   ```

   The build is a check only; never upload it by hand.
4. **Commit and push** to `main`, then wait until *Actions → CI* is green for that commit.

### Tag and publish

5. **Tag the green commit** and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "BeyondNN X.Y.Z"
   git push origin vX.Y.Z
   git ls-remote --tags origin vX.Y.Z
   ```

6. **Create the GitHub release** from the tag (optional, but recommended). It does not publish to PyPI.
7. **Open the workflow:** GitHub → *Actions → Publish to PyPI* → *Run workflow*. Keep branch `main`; the input decides what is published.
8. **Enter the tag:** `vX.Y.Z`.
9. **Review the run.** The build job:
   - validates the tag format;
   - checks out **exactly** that tag (it fails if the tag does not exist; it never falls back to `main`);
   - verifies tag == `v` + package version and prints `Publishing BeyondNN X.Y.Z from tag vX.Y.Z`;
   - builds fresh distributions with `uv build`;
   - runs `twine check`;
   - checks that the wheel contains only the package.

   The publish job then uploads through the `pypi` environment with OIDC. If the environment has required reviewers, approve it there.

### Verify

10. **Check the PyPI page:** https://pypi.org/project/beyondnn/ shows X.Y.Z, with the README rendered.
11. **Fresh-install from PyPI**, in a new environment outside the repository:

    ```bash
    python -m venv /tmp/bnn && . /tmp/bnn/bin/activate
    pip install --no-cache-dir beyondnn==X.Y.Z
    python -c "import beyondnn; print(beyondnn.__version__, beyondnn.__file__)"   # X.Y.Z, inside /tmp/bnn
    ```

## First-time setup (already done for `beyondnn`)

The Trusted Publisher was registered before 0.1.0 as a *pending publisher*: on PyPI, *Account settings → Publishing → Add a new pending publisher → GitHub*, then the five values above. Re-register it only if the repository, workflow file name or environment name changes.

## Optional: protect the `pypi` environment

*Repository → Settings → Environments → `pypi`*: add yourself under *Required reviewers*, so every publish run waits for an explicit approval. Do **not** add any secret.

## If something goes wrong

**PyPI versions are immutable:**
- **Before the upload:** fix it and release a new patch version with a new tag. Never move a published tag.
- **After the upload:** a bad release can be *yanked* on PyPI (it stays installable only by exact pin), but the version number is spent.
- **The PyPI project page shows the README of the uploaded version.** README changes on `main` appear on PyPI only with the next release.
