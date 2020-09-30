import math

from typing import List

from ..utils.utils import grouped
from .utils import BatchException
from .batch import Batch
from .resource import ResourceGroup, ResourceFile


def concatenate(b: Batch, files: List[ResourceFile], branching_factor: int = 100):
    """
    Concatenate files using tree aggregation.

    Notes
    -----
    Runs the `cat` command aggregating files using the `branching_factor`.

    Examples
    --------
    Create and execute a batch that concatenates output files:

    >>> b = Batch()
    >>> j1 = b.new_job()
    >>> j1.command(f'touch {j1.ofile}')
    >>> j2 = b.new_job()
    >>> j2.command(f'touch {j2.ofile}')
    >>> j3 = b.new_job()
    >>> j3.command(f'touch {j3.ofile}')
    >>> files = [j1.ofile, j2.ofile, j3.ofile]
    >>> ofile = concatenate(b, files, branching_factor=2)
    >>> b.run()

    Parameters
    ----------
    b:
        Batch to add concatenation jobs to.
    files:
        List of files to concatenate.
    branching_factor:
        Grouping factor when concatenating files.

    Returns
    -------
    Concatenated output file.
    """

    def _concatenate(b, name, xs):
        j = b.new_job(name=name)
        j.command(f'cat {" ".join(xs)} > {j.ofile}')
        return j.ofile

    if len(files) == 0:
        raise BatchException(f'Must have at least one file to concatenate.')

    if not all([isinstance(f, ResourceFile) for f in files]):
        raise BatchException(f'Invalid input file(s) - all inputs must be resource files.')

    return _combine(_concatenate, b, 'concatenate', files, branching_factor=branching_factor)


def plink_merge(b: Batch, bfiles: List[ResourceGroup], branching_factor: int = 100):
    """
    Merge binary PLINK files using tree aggregation.

    Notes
    -----
    Runs the `plink --bmerge-list` command aggregating files using the `branching_factor`.

    Parameters
    ----------
    b:
        Batch to add merge jobs to.
    bfiles:
        List of binary PLINK file roots to merge.
    branching_factor:
        Grouping factor when merging files.

    Returns
    -------
    Concatenated binary PLINK file.
    """

    def _plink_merge(b, name, xs):
        assert xs
        if len(xs) == 1:
            return xs[0]
        j = b.new_job(name=name)
        for f in xs[1:]:
            j.command(f'echo "{f.bed} {f.bim} {f.fam}" >> merge_list')
        j.command(f'plink --bfile {xs[0]} --merge-list merge_list --out {j.ofile}')
        return j.ofile

    if len(bfiles) == 0:
        raise BatchException(f'Must have at least one binary PLINK file to merge.')

    if not all([isinstance(bf, ResourceGroup) for bf in bfiles]):
        raise BatchException(f'Invalid input file(s) - all inputs must be resource groups.')

    return _combine(_plink_merge, b, 'plink-merge', bfiles, branching_factor=branching_factor)


def _combine(combop, b, name, xs, branching_factor=100):
    n_levels = math.log(len(xs), branching_factor)
    level = 0
    while level < n_levels:
        grouped_xs = grouped(branching_factor, xs)
        xs = [combop(b, f'{name}-{level}-{i}', xs) for i, xs in enumerate(grouped_xs)]
        level += 1
    assert len(xs) == 1
    return xs[0]
