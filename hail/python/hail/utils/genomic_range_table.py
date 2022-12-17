from typing import Optional

import hail as hl
from .misc import check_nonnegative_and_in_range, check_positive_and_in_range
from ..genetics.reference_genome import reference_genome_type
from ..typecheck import typecheck, nullable


@typecheck(n=int, n_partitions=nullable(int), reference_genome=nullable(reference_genome_type))
def genomic_range_table(n: int,
                        n_partitions: Optional[int] = None,
                        reference_genome='default'
                        ) -> 'hl.Table':
    """Construct a table with a locus and no other fields.

    Examples
    --------

    >>> ht = hl.utils.range_table(100)
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
        Number of partitions (uses Spark default parallelism if None).
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

    return hl.Table(hl.ir.TableGenomicRange(n, n_partitions, reference_genome))
