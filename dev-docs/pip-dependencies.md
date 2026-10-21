# Updating dependencies

Pip dependencies for a particular python module should be listed in a
`requirements.txt` file, e.g. `$HAIL/hail/python/requirements.txt`.
The version should be the most permissive version range compatible with the hail
package. In each directory that contains a `requirements.txt` file there should
also be a `pinned-requirements.txt` file that contains a fully resolved
dependency tree with each pip dependency pinned to a specific version.
This allows deterministic builds in our Dockerfiles. When adding a dependency
to a `requirements.txt` file, run

```bash
make generate-pip-lockfiles
```

to regenerate the `pinnned-requirements.txt` files in the repository.
Note that the full dependency tree for a pip package can
differ on different operating systems. All services and docker images in CI
use the fully pinned requirements on a Linux platform, so the pinned requirements
files are only generated for Linux.

## Hashes

The `pinned-requirements.txt` files include a `--hash` for every published file of each
pinned version, so pip refuses to install anything whose contents don't match what was
recorded when the lockfile was generated. Pip turns on hash checking for a whole
`pip install` call as soon as any requirement *or constraint* has a hash, so:

- Don't mix a pinned file with unpinned packages in a single `pip install`; install
  the extras in a separate call.
- When using a pinned file only as a `--constraint` for an unpinned package, pass
  `--no-require-hashes`.

If an install fails with a hash mismatch, don't just regenerate the lockfile.
PyPI files are immutable, so a mismatch means the downloaded file is not the one we
pinned (a tampered download, a bad mirror, or a corrupted cache). Investigate first.
