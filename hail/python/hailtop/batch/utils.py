import math

from typing import List, Optional

from ..utils.utils import grouped, digits_needed
from ..config.deploy_config import TerraDeployConfig, get_deploy_config
from .batch import Batch
from .exceptions import BatchException
from .resource import ResourceGroup, ResourceFile


def concatenate(b: Batch, files: List[ResourceFile], image: Optional[str] = None, branching_factor: int = 100) -> ResourceFile:
    """
    Concatenate files using tree aggregation.

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
    image:
        Image to use. Must have the cat command.

    Returns
    -------
    Concatenated output file.
    """

    def _concatenate(b, name, xs):
        j = b.new_job(name=name)
        if image:
            j.image(image)
        j.command(f'cat {" ".join(xs)} > {j.ofile}')
        return j.ofile

    if len(files) == 0:
        raise BatchException('Must have at least one file to concatenate.')

    if not all(isinstance(f, ResourceFile) for f in files):
        raise BatchException('Invalid input file(s) - all inputs must be resource files.')

    return _combine(_concatenate, b, 'concatenate', files, branching_factor=branching_factor)


def plink_merge(b: Batch, bfiles: List[ResourceGroup],
                image: Optional[str] = None, branching_factor: int = 100) -> ResourceGroup:
    """
    Merge binary PLINK files using tree aggregation.

    Parameters
    ----------
    b:
        Batch to add merge jobs to.
    bfiles:
        List of binary PLINK file roots to merge.
    image:
        Image name that contains PLINK.
    branching_factor:
        Grouping factor when merging files.

    Returns
    -------
    Merged binary PLINK file.
    """

    def _plink_merge(b, name, xs):
        assert xs
        if len(xs) == 1:
            return xs[0]
        j = b.new_job(name=name)
        if image:
            j.image(image)
        for f in xs[1:]:
            j.command(f'echo "{f.bed} {f.bim} {f.fam}" >> merge_list')
        j.command(f'plink --bfile {xs[0]} --merge-list merge_list --out {j.ofile}')
        return j.ofile

    if len(bfiles) == 0:
        raise BatchException('Must have at least one binary PLINK file to merge.')

    if not all(isinstance(bf, ResourceGroup) for bf in bfiles):
        raise BatchException('Invalid input file(s) - all inputs must be resource groups.')

    return _combine(_plink_merge, b, 'plink-merge', bfiles, branching_factor=branching_factor)


def _combine(combop, b, name, xs, branching_factor=100):
    assert isinstance(branching_factor, int) and branching_factor >= 1
    n_levels = math.ceil(math.log(len(xs), branching_factor))
    level_digits = digits_needed(n_levels)

    level = 0
    while level < n_levels:
        branch_digits = digits_needed(len(xs) // branching_factor + min(len(xs) % branching_factor, 1))
        grouped_xs = grouped(branching_factor, xs)
        xs = [combop(b, f'{name}-{level:0{level_digits}}-{i:0{branch_digits}}', xs) for i, xs in enumerate(grouped_xs)]
        level += 1
    assert len(xs) == 1
    return xs[0]


def needs_tokens_mounted() -> bool:
    return not isinstance(get_deploy_config(), TerraDeployConfig)
