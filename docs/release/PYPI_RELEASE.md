# Publishing a release to PyPI

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
- **Immutable:** a version uploaded to PyPI cannot be replaced or re-uploaded. Review everything before step 6.

## Steps

1. **Go to PyPI Trusted Publishers.** Log in at https://pypi.org, then *Account settings → Publishing → Add a new pending publisher → GitHub*. A pending publisher creates the project on its first upload.
2. **Enter the five values** above, exactly.
3. **Save the publisher.**
4. **Check the tag.** `v0.1.0` must exist and point to the release commit whose CI is green:

   ```bash
   git ls-remote --tags origin v0.1.0
   ```

   (On GitHub: *Actions → CI*, the run for that commit.)
5. **Open the workflow:** GitHub → *Actions → Publish to PyPI*.
6. **Click *Run workflow*** (branch: `main`; the input decides what is published).
7. **Enter the tag:** `v0.1.0`.
8. **Review the run before considering the release complete.** The build job:
   - validates the tag format;
   - checks out **exactly** that tag (it fails if the tag does not exist; it never falls back to `main`);
   - verifies tag == `v` + package version and prints `Publishing BeyondNN 0.1.0 from tag v0.1.0`;
   - builds fresh distributions with `uv build`;
   - runs `twine check`;
   - checks that the wheel contains only the package.

   The publish job then uploads through the `pypi` environment with OIDC. If the environment has required reviewers, approve it there.
9. **Verify the PyPI page:** https://pypi.org/project/beyondnn/ shows 0.1.0, with the README rendered.
10. **Fresh-install from PyPI**, in a new environment:

    ```bash
    python -m venv /tmp/bnn && . /tmp/bnn/bin/activate
    pip install beyondnn
    ```

11. **Verify the version:**

    ```bash
    python -c "import beyondnn; print(beyondnn.__version__)"   # 0.1.0
    ```

## After publication

- README: replace "PyPI publication is pending" with the published state, and add a PyPI badge.
- CHANGELOG: drop "Not yet published on PyPI at the time of tagging".

## Optional: protect the `pypi` environment

*Repository → Settings → Environments → `pypi`*: add yourself under *Required reviewers*, so every publish run waits for an explicit approval. Do **not** add any secret.

## If something goes wrong

**PyPI versions are immutable:**
- **Before the upload:** fix it and release a new patch version (e.g. `0.1.1`) with a new tag. Never move a published tag.
- **After the upload:** a bad release can be *yanked* on PyPI (it stays installable only by exact pin), but the version number is spent.
