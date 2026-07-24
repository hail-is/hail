# Bumping the supported Python version range

This doc covers how to add or remove a Python version from the set that Hail builds client images for, and how to shift the *default* service Python version (i.e. what runs inside the Batch/Auth/CI pods).

The two operations are separable: you can add a new client image (e.g. for EMR users on Python 3.11) without changing the service default.

---

## Adding a new client image (non-default Python version)

A "client image" means `hailgenetics/hail:{version}-py{X.Y}` and `hailgenetics/python-dill:{X.Y}[-slim]`. Users on that Python version point their Batch jobs at these images.

### 1. `build.yaml` — three new stanzas

Copy the `hail_ubuntu_image_python_3_13` / `hailgenetics_hail_image_python_3_13` / `test_hailgenetics_hail_image_python_3_13` stanza trio and replace `3_13` / `3.13` / `3-13` throughout.

- `hail_ubuntu_image_python_3_XY` — builds the base image via `ARG PYTHON_VERSION=3.XY` in `docker/hail-ubuntu/Dockerfile`. No extra flags needed unless the Python version predates `google-cloud-profiler` support (3.13+ needed `SKIP_GCLOUD_PROFILER=true`).
- `hailgenetics_hail_image_python_3_XY` — builds the hailgenetics image on top of the ubuntu base.
- `test_hailgenetics_hail_image_python_3_XY` — smoke-tests import and a trivial computation.

Also wire the new image into the release step:
- Add `HAIL_GENETICS_HAIL_IMAGE_PY_3_XY=docker://{{ hailgenetics_hail_image_python_3_XY.image }} \` to the env block.
- Add `- hailgenetics_hail_image_python_3_XY` to the `dependsOn` list.

### 2. `hail/scripts/release.sh`

Four places to update (search for the `3_13` / `3.13` pattern to find them all):
- Usage comment block
- `arguments=` required-var list
- `skopeo inspect` preflight checks
- `skopeo copy` publish calls (two lines: DockerHub + GCP AR)

### 3. `hail/python/hailtop/batch/hail_genetics_images.py`

Add the new minor version to the accepted set and update the error message's version range description.

### 4. `docker/hailgenetics/python-dill/push.sh`

Add `3.XY` and `3.XY-slim` to the version loop.

### 5. `docker/hailgenetics/mirror_images.sh`

Add `"python-dill:3.XY"` and `"python-dill:3.XY-slim"` to the `images=()` array.

### 6. `infra/gcp-broad/gcp-ar-cleanup-policy.txt`

Add `"hail-ubuntu-python-3-XY"` to the keep-list so CI builds of that image are not garbage-collected. The `publishAs:` value in `build.yaml` determines the repository name — make sure they match.

---

## Shifting the default service Python version

The *default* is the version used by the Batch/Auth/CI/Gear services and by `hailgenetics/hail:{version}` (no `-pyX.Y` suffix). Shifting the default is a bigger change.

### Additional files beyond the client-image list above

| File | What to change |
|---|---|
| `docker/hail-ubuntu/Dockerfile` | `ARG PYTHON_VERSION=3.XY` default |
| `hail/python/hail/__init__.py` | `version_info < (3, XY)` runtime guard and its error message |
| `hail/python/setup.py` | `python_requires=">=3.XY"` |
| `hail/python/setup-hailtop.py` | `python_requires=">=3.XY"` |
| `hail/python/hailtop/batch/hail_genetics_images.py` | The branch that returns the untagged image (`if version.minor == XY`) |
| `hail/python/hailtop/batch/docker.py` | `minor_version < XY` floor check and its error message |
| `generate-pip-lockfile.sh` | `--python-version 3.XY` |
| `check_pip_requirements.sh` | `--python-version 3.XY` |
| `build.yaml` `test_install_in_isolation` step | `uv venv --python 3.XY` and step name |
| All `pinned-requirements.txt` files | Regenerate with `generate-pip-lockfile.sh` for each service |
| `.github/workflows/static-and-unit-tests.yml` | Python version matrix |
| `.github/workflows/pip-npm-bump.yml` | Python version |
| `.github/workflows/validate-build-yaml.yml` | Python version |
| `ci/test/resources/build.yaml` | Python version references |
| `hail/Makefile` | Python version in test targets |
| Docs: `hail/python/hail/docs/install/*.rst` | Stated minimum Python version |
| User-facing strings in `hail_genetics_images.py` error message | Version range |
| Docstring examples in `batch.py`, `job.py` | `python-dill:3.XY-slim` image refs |
| Test helpers: `hail/python/test/hailtop/batch/utils.py` | `PYTHON_DILL_IMAGE` |
| Test helpers: `hail/python/test/hailtop/hailctl/batch/test_submit.py` | Default image |
| `batch/test/test_worker_image_security.py` | Hardcoded `python:3.XY-slim` image |
| Dev docs / install guides that mention a specific Python version | e.g. `dev-docs/hail-for-new-engineers.md`, `user-scripts/manifold/install_hail_kernel.sh` |

---

## Dropping an old client image

Mirror the add steps in reverse:
- Remove the three `build.yaml` stanzas and their release-step references.
- Remove from `release.sh`.
- Remove the minor version from `hail_genetics_images.py`.
- Remove from `push.sh` and `mirror_images.sh`.
- No need to remove the version from `gcp-ar-cleanup-policy.txt` — entries are free, and removing them while old images are still aging out risks leaving behind zombie artifacts with no retention policy.
- Delete the corresponding `pinned-requirements-py3XY.txt` if one exists.

---

## Gotchas from the 3.10→3.12 bump (July 2026)

- **`generate-pip-lockfile.sh` has a hardcoded `--python-version`** that was missed in the initial commit; the lockfiles were re-pinned against the wrong version until a followup fixed it.

- **`check_pip_requirements.sh` also has a hardcoded `--python-version`** — same oversight, same fix.

- **`build.yaml` step names contain the version** (`test_install_in_isolation_python3_11`). The step itself was renamed but its `dependsOn` references in downstream steps were not, causing a broken reference caught only after the initial commit.

- **Docstring examples in `batch.py` and `job.py`** hardcode a `python-dill:3.XY-slim` image tag. Easy to grep-miss because they live in docstrings rather than code.

- **`batch/test/test_worker_image_security.py`** uses a hardcoded `python:3.XY-slim` image (not `hailgenetics/python-dill`). Grep for `python:3.` not just `python-dill:3.`.

- **`gcp-ar-cleanup-policy.txt` needs updating** for each new non-default ubuntu image repository name (`hail-ubuntu-python-3-XY`). The `publishAs:` value in `build.yaml` determines the name — double-check they match. **Do not remove old entries when dropping a version** — cleanup policy entries are free, and removing them before old images age out leaves zombie artifacts with no retention policy.

- **`hail/python/test/hailtop/hailctl/batch/test_submit.py`** has a fallback default image used when no `--image` flag is set in tests; it's not near the other image constants and is easy to miss.

- **`hailtop/batch/docker.py`** has a `minor_version < XY` floor check (for `build_python_image`) that also needs updating. It's separate from `hail_genetics_images.py` and easy to miss because it guards a different code path (local Docker builds, not hailgenetics image selection).

- **`hail/python/hail/__init__.py`** has a runtime `version_info < (3, XY)` guard at import time. It must match `python_requires` in `setup.py` / `setup-hailtop.py`. Easy to miss because it's in the package `__init__` rather than near any build or setup file.

- **EMR consideration**: EMR 7.x ships Python 3.11 and upgrades slowly. Before dropping 3.11 support in a future bump, check current EMR release notes — keeping a non-default 3.11 client image is low cost and avoids breaking EMR users.
