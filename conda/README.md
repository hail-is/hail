# Conda package

This folder contains a conda recipe to build the `hail` package for
the [`cpg` Anaconda channel](https://anaconda.org/cpg/hail).

Note that there is also a `hail` package in the
[`bioconda` channel](https://github.com/bioconda/bioconda-recipes/tree/master/recipes/hail)
synced with the [official PyPI release](https://pypi.org/project/hail). However, having
a separate conda package in the `cpg` channel allows us to build it against the codebase
in our fork.

Because we don't control versioning of Hail project, the `cpg` conda package is
versioned specifically: we append the git commit hash to the official version tag,
e.g. `0.2.62.dev289c163`.

[GitHub Actions CI](../.github/workflows/condarise.yaml) is set up to build the package
using this recipe and push it to Anaconda on every push event to the `main` branch in
the
[CPG hail fork](https://github.com/populationgenomics/hail).

When installing the package, list the `cpg` channel before `bioconda` to prioritize it
in the channel order:

```
conda create --name hail -c cpg -c bioconda -c conda-forge hail
conda activate hail
```

You can also install Hail into an existing environment. However, note that Hail requires
Python of versions 3.6 or 3.7, so conda might downgrade Python in that environment,
which may affect other installed packages.

Note that if you don't have `conda` installed, here are handy commands to do that:

```
if [[ "$OSTYPE" == "darwin"* ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi
bash miniconda.sh
```
