from numpy import linspace
from typing import Optional

import hail as hl
from .misc import check_nonnegative_and_in_range, check_positive_and_in_range
from ..genetics.reference_genome import reference_genome_type
from ..typecheck import typecheck, nullable


@typecheck(n=int, n_partitions=nullable(int), reference_genome=nullable(reference_genome_type))
def genomic_range_table(n: int, n_partitions: Optional[int] = None, reference_genome='default') -> 'hl.Table':
    """Construct a table with a locus and no other fields.

    Examples
    --------

    >>> ht = hl.utils.genomic_range_table(100)
    >>> ht.count()
    100

    Notes
    -----
    The resulting table contains one field:

     - `locus` (:py:data:`.tlocus`) - Row index (key).

    The loci appear in sequential ascending order.

    Parameters
    ----------
    n : int
        Number of loci. Must be less than 2 ** 31.
    n_partitions : int, optional
        Number of partitions [default: 8].
    reference_genome : :class:`str` or :class:`.ReferenceGenome`
        Reference genome to use for creating the loci.

    Returns
    -------
    :class:`.Table`
    """
    check_nonnegative_and_in_range('range_table', 'n', n)
    if n_partitions is not None:
        check_positive_and_in_range('range_table', 'n_partitions', n_partitions)
    if n >= (1 << 31):
        raise ValueError(f'`n`, {n}, must be less than 2 ** 31.')

    n_partitions = n_partitions or 8
    start_idxs = [int(x) for x in linspace(0, n, n_partitions + 1)]
    idx_bounds = list(zip(start_idxs, start_idxs[1:]))

    return hl.Table._generate(
        contexts=idx_bounds,
        partitions=[
            hl.Interval(**{
                endpoint: hl.Struct(locus=reference_genome.locus_from_global_position(idx))
                for endpoint, idx in [('start', lo), ('end', hi)]
            })
            for (lo, hi) in idx_bounds
        ],
        rowfn=lambda idx_range, _: hl.range(idx_range[0], idx_range[1]).map(
            lambda idx: hl.struct(locus=hl.locus_from_global_position(idx, reference_genome))
        ),
    )
