# Publishing a release to PyPI

Publication is a **deliberate, manual** maintainer action:
- **Manual only:** `.github/workflows/publish.yml` runs only when triggered by hand (`workflow_dispatch`). Pushes, tags and GitHub releases never publish.
- **No credentials:** publication uses PyPI Trusted Publishing (GitHub OIDC). No password or API token is stored anywhere.
- **Immutable:** a version uploaded to PyPI cannot be replaced. Check everything before step 4.

## One-time setup

1. **Log in to PyPI** (https://pypi.org) with the account that will own the project.
2. **Add a pending Trusted Publisher:** Account → *Publishing* → *Add a new pending publisher* → GitHub:

   | field | value |
   |---|---|
   | PyPI project name | `beyondnn` |
   | Owner | `NikolasRoufas` |
   | Repository name | `beyondnn` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

3. **Create the GitHub environment.** On GitHub: *Settings → Environments → New environment* → `pypi`. Recommended: add yourself under *Required reviewers*, so every publish run waits for an explicit approval. Do **not** add any secret.

## Publishing a version (e.g. 0.1.0)

1. **Check the release:**
   - `main` CI is green for the release commit;
   - the tag `v0.1.0` exists and points to that commit;
   - `beyondnn/__init__.py` says `0.1.0`;
   - the GitHub release exists.
2. **Trigger the workflow:** on GitHub, *Actions → Publish to PyPI → Run workflow*. Enter the tag: `v0.1.0`.
3. **Watch the build job.** It validates the tag format, checks out exactly that tag, verifies that the tag equals `v<package version>`, builds fresh distributions, runs `twine check --strict`, and checks that the wheel contains only the package.
4. **Approve the `pypi` environment** (if you configured a reviewer). The publish job then uploads with OIDC.
5. **Verify on PyPI:** https://pypi.org/project/beyondnn/ shows `0.1.0`, with the README rendered.
6. **Fresh-install test**, in a new environment:

   ```bash
   python -m venv /tmp/bnn && . /tmp/bnn/bin/activate
   pip install beyondnn
   python -c "import beyondnn; print(beyondnn.__version__)"   # 0.1.0
   ```

7. **After publication:**
   - README: replace "PyPI publication is pending" with the published state, and add a PyPI badge;
   - CHANGELOG: drop "Not yet published on PyPI at the time of tagging".

## If something goes wrong

- **Before the upload:** fix it, create a new patch version (e.g. `0.1.1`) with a new tag. Never move a published tag.
- **After the upload:** a bad release can be *yanked* on PyPI (it stays installable only by exact pin), but the version number is spent.
