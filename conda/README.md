# Conda package

This folder contains a Conda recipe to build [the `hail` package for the `cpg` Anaconda channel]((https://anaconda.org/cpg/hail)).

Note that there is also [a `hail` package in the `bioconda` channel](https://github.com/bioconda/bioconda-recipes/tree/master/recipes/hail), which is synced with the [official PyPI release](https://pypi.org/project/hail). The idea of having a separate package in the `cpg` channel is to be able to build it from the most recent development codebase, that would reflect changes that were not yet released, or not to be propagated to the upstream repository at all.

[GitHub Actions CI](../.github/workflows/main.yaml)) is set up to build the package using this recipe and push it to Anaconda on every push to the `main` branch in the [CPG hail fork](https://github.com/populationgenomics/hail).

To install the package, set up miniconda first:

```
if [[ "$OSTYPE" == "darwin"* ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi
bash miniconda.sh
```

When installing, to prioritize the CPG package, list the `cpg` channel before `bioconda`:

```
conda create --name hail -c cpg -c bioconda -c conda-forge hail
conda activate hail
```

You can also install Hail into an existing enviornment; however note that Hail requires Python of versions 3.6 or 3.7, so Conda might downgrade Python in that environment, which may affect other installed packages.
