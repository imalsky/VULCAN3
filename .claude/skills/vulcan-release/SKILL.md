---
name: vulcan-release
description: Use when asked to release the VULCAN packages (vulcan3, vulcan-forward, vulcan-retrieval, jwst-transit-authority) to TestPyPI, tag a release, or redeploy the JWST Transit Authority Hugging Face Spaces.
---

# VULCAN release: TestPyPI, tags, JWST Spaces

Work from the workspace root, the folder that holds the four repositories.
Credentials must already be in place: `~/.pypirc` with a `[testpypi]` section,
and a Hugging Face login that can write to the Space. `uv` and `uvx` must be
on the PATH, which a non-interactive shell may not set up.

The user names the expected head SHA and version of each repo. Stop and ask
if git shows anything else. Never edit code, bump a version or run
`release.sh` (it bumps the patch version) as part of a release.

Repos: `VULCAN-JAX` (GitHub `imalsky/VULCAN3`), `vulcan-forward`,
`vulcan-retrieval`, `vulcan-jwst-tool` (GitHub `imalsky/jwst-transit-authority`).

## 1. Heads

Per repo: `git checkout main && git pull --ff-only`, confirm the SHA and an
empty `git status --porcelain`. An untracked leftover cannot reach a
`git archive` build: report it, do not delete it. Before leaving another
branch, `git cherry origin/main <branch>` shows whether its work is on main.
Versions: `git show HEAD:<path>` of `src/vulcan3/_version.py`,
`src/vulcan_forward/_version.py`, `retrieval_framework/_version.py`,
`jwst_tool/_version.py`.

## 2. Build from a clean export

```bash
W=$PWD; S=$(mktemp -d)
for r in VULCAN-JAX vulcan-forward vulcan-retrieval vulcan-jwst-tool; do
  mkdir -p $S/src/$r; git -C $r archive HEAD | tar -x -C $S/src/$r
  (cd $S/src/$r && $W/.venv/bin/python -m build --outdir $S/dist/$r .)
done
.venv/bin/python -m twine check $S/dist/*/*
unzip -p $S/dist/vulcan-forward/*.whl '*/METADATA' | grep Requires-Dist   # must list radis<0.17
```

Check that no target version exists yet: `curl -s https://test.pypi.org/pypi/<name>/json`.

## 3. Upload, one at a time

Order: vulcan3, vulcan-forward, vulcan-retrieval, jwst-transit-authority.

```bash
.venv/bin/python -m twine upload --non-interactive -r testpypi $S/dist/<repo>/*
```

Never `--repository-url`: it ignores `~/.pypirc`. Before the next upload,
confirm both files are listed at `https://test.pypi.org/pypi/<name>/<ver>/json`
and `https://test.pypi.org/simple/<name>/`. The HTML project page is behind a
JavaScript "Client Challenge" and cannot be read by a script. A 403 "Invalid
or non-existent authentication information" means the token is bad: check
that `~/.pypirc` holds a ~185-character `pypi-` token, not a placeholder.

## 4. README install tests

One fresh venv per README: `uv venv --seed --python 3.12 --no-project <dir>`,
activate it, `export PIP_LOG=<file>`, then paste the README install block
verbatim.

- VULCAN-JAX: `import vulcan3`, then run the README "Example" block
  (extract it from README.md, do not retype it) with `MPLBACKEND=Agg` and
  print `int(rs.params.count), int(rs.params.end_case)` after it. Expect
  1495 and 1 (about 50 s on a laptop CPU).
- vulcan-forward: `from vulcan_forward import vulcan_chem`, then
  `from vulcan_forward import exojax_rt`.
- Files pip took from TestPyPI:
  `grep -E '(Downloading link|Obtaining dependency information for .* from) https://test-files' $PIP_LOG | grep -oE 'test-files\.pythonhosted\.org/[^ #)]*' | sed 's#.*/##; s#\.metadata$##' | sort -u`

Known at vulcan3 0.18.0 / vulcan-forward 0.29.0 (2026-10-16); report again if still true:
- The plain `vulcan3` install has no matplotlib (it is in the `[plot]`
  extra), so the README example fails at `import matplotlib.pyplot`.
- TestPyPI serves `python-dateutil 3.9.0` to both installs. Its code is
  identical to PyPI's 2.9.0.post0 apart from the version string and the
  tz-data tarball.
- TestPyPI serves `psutil 8.0.0` to the vulcan-forward install (seen
  2026-10-16), ahead of PyPI's psutil.
- The unpinned README installs resolve jax 0.11.x (CI pins 0.6.2); the
  README example still gives 1495 / 1 on it.
- TestPyPI serves `radis 0.16.4`, which is not on PyPI. It requires vaex
  below Python 3.13, so vaex installs on 3.12 despite `radis<0.17`.

## 5. Tags: ask the user first

Annotated, like the earlier tags:

```bash
git -C <repo> tag -a v<ver> <sha> -m "Release v<ver>"
git -C <repo> push origin refs/tags/v<ver>
git -C <repo> ls-remote origin "refs/tags/v<ver>^{}"   # must print <sha>
```

VULCAN-JAX's `tests` workflow also runs on a tag push. Wait for it
(`gh run list --repo imalsky/VULCAN3`) and read `gh run view <id> --log-failed`
on anything but success.

## 6. Redeploy the Space

From `vulcan-jwst-tool` at the release commit; `deploy/hf-space` must hold
only tracked files.

```bash
uvx --from huggingface_hub python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='imalsky/jwst-transit-authority', repo_type='space',
    folder_path='deploy/hf-space', ignore_patterns=['__pycache__*'],
    commit_message='Pin to <sha>')"
```

Verify (`s` = `imalsky/jwst-transit-authority`):
- `HfApi().get_space_runtime(s).stage` is exactly `RUNNING` AND
  `runtime.raw['sha'] == HfApi().space_info(s).sha`. The old instance
  reports `RUNNING_BUILDING` / `RUNNING_APP_STARTING` while the new one builds.
- The build log (`https://huggingface.co/api/spaces/<s>/logs/build`, header
  `Authorization: Bearer <huggingface_hub.get_token()>`, an SSE stream of
  `data:` JSON lines) prints the three `BUILD_INFO` lines
  `VULCAN-JAX <sha>`, `vulcan-forward <sha>`, `jwst-transit-authority <sha>`.
  The echoed `RUN set -e` command also contains the SHAs; that is not evidence.
- `https://imalsky-jwst-transit-authority.hf.space/_stcore/health` returns `ok`.

A pin-bump rebuild takes about 2 minutes. On a failure, read the build log
and stop. Never edit the Dockerfile.

## 7. Report

Name, version and TestPyPI URL of each upload; install-test results with the
versions pip resolved and the files it took from TestPyPI; tag SHAs; each
Space's stage, live commit, build-log SHAs and health.
