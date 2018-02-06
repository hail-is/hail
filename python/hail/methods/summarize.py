from hail.typecheck import *
from hail.matrixtable import MatrixTable
from hail.utils.java import handle_py4j
from hail.utils import Summary

@handle_py4j
@typecheck(ds=MatrixTable)
def summarize(ds):
    """Returns a summary of useful information about the dataset.

    .. include:: ../_templates/experimental.rst

    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------
    >>> s = methods.summarize(dataset)
    >>> print(s.contigs)
    >>> print('call rate is %.2f' % s.call_rate)
    >>> s.report()

    Notes
    -----
    The following information is contained in the summary:

     - **samples** (*int*) - Number of samples.
     - **variants** (*int*) - Number of variants.
     - **call_rate** (*float*) - Fraction of all genotypes called.
     - **contigs** (*list of str*) - List of all unique contigs found in the dataset.
     - **multiallelics** (*int*) - Number of multiallelic variants.
     - **snps** (*int*) - Number of SNP alternate alleles.
     - **mnps** (*int*) - Number of MNP alternate alleles.
     - **insertions** (*int*) - Number of insertion alternate alleles.
     - **deletions** (*int*) - Number of deletions alternate alleles.
     - **complex** (*int*) - Number of complex alternate alleles.
     - **star** (*int*) - Number of star (upstream deletion) alternate alleles.
     - **max_alleles** (*int*) - The highest number of alleles at any variant.

    Returns
    -------
    :class:`~hail.utils.Summary`
        Summary of the dataset.
    """
    return Summary._from_java(ds._jvds.summarize())
