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
make generate-pip-lockfile
```

to regenerate the `pinnned-requirements.txt` file to one compatible with the
new requirements.
