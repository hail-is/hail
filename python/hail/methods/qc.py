from hail.typecheck import *
from hail.utils.java import Env, handle_py4j
from hail.api2 import MatrixTable
from .misc import require_biallelic


@handle_py4j
@require_biallelic
@typecheck(dataset=MatrixTable, name=strlike)
def sample_qc(dataset, name='sample_qc'):
    """Compute per-sample metrics useful for quality control.

    .. include:: ../_templates/req_tvariant.rst

    **Examples**

    .. testsetup::

        dataset = vds.annotate_samples_expr('sa = drop(sa, qc)').to_hail2()
        from hail.methods import sample_qc

    Compute sample QC metrics and remove low-quality samples:

    >>> dataset = sample_qc(dataset, name='sample_qc')
    >>> filtered_dataset = dataset.filter_cols((dataset.sample_qc.dpMean > 20) & (dataset.sample_qc.rTiTv > 1.5))

    **Notes**:

    This method computes summary statistics per sample from a genetic matrix and stores the results as
    a new column-indexed field in the matrix, named based on the ``name`` parameter.

    +------------------------+-------+-+----------------------------------------------------------+
    | Name                   | Type    | Description                                              |
    +========================+=========+==========================================================+
    | ``callRate``           | Float64 | Fraction of calls non-missing                            |
    +------------------------+---------+----------------------------------------------------------+
    | ``nHomRef``            | Int64   | Number of homozygous reference calls                     |
    +------------------------+---------+----------------------------------------------------------+
    | ``nHet``               | Int64   | Number of heterozygous calls                             |
    +------------------------+---------+----------------------------------------------------------+
    | ``nHomVar``            | Int64   | Number of homozygous alternate calls                     |
    +------------------------+---------+----------------------------------------------------------+
    | ``nCalled``            | Int64   | Sum of ``nHomRef`` + ``nHet`` + ``nHomVar``              |
    +------------------------+---------+----------------------------------------------------------+
    | ``nNotCalled``         | Int64   | Number of missing calls                                  |
    +------------------------+---------+----------------------------------------------------------+
    | ``nSNP``               | Int64   | Number of SNP alternate alleles                          |
    +------------------------+---------+----------------------------------------------------------+
    | ``nInsertion``         | Int64   | Number of insertion alternate alleles                    |
    +------------------------+---------+----------------------------------------------------------+
    | ``nDeletion``          | Int64   | Number of deletion alternate alleles                     |
    +------------------------+---------+----------------------------------------------------------+
    | ``nSingleton``         | Int64   | Number of private alleles                                |
    +------------------------+---------+----------------------------------------------------------+
    | ``nTransition``        | Int64   | Number of transition (A-G, C-T) alternate alleles        |
    +------------------------+---------+----------------------------------------------------------+
    | ``nTransversion``      | Int64   | Number of transversion alternate alleles                 |
    +------------------------+---------+----------------------------------------------------------+
    | ``nStar``              | Int64   | Number of star (upstream deletion) alleles               |
    +------------------------+---------+----------------------------------------------------------+
    | ``nNonRef``            | Int64   | Sum of ``nHet`` and ``nHomVar``                          |
    +------------------------+---------+----------------------------------------------------------+
    | ``rTiTv``              | Float64 | Transition/Transversion ratio                            |
    +------------------------+---------+----------------------------------------------------------+
    | ``rHetHomVar``         | Float64 | Het/HomVar call ratio                                    |
    +------------------------+---------+----------------------------------------------------------+
    | ``rInsertionDeletion`` | Float64 | Insertion/Deletion allele ratio                          |
    +------------------------+---------+----------------------------------------------------------+
    | ``dpMean``             | Float64 | Depth mean across all calls                              |
    +------------------------+---------+----------------------------------------------------------+
    | ``dpStDev``            | Float64 | Depth standard deviation across all calls                |
    +------------------------+---------+----------------------------------------------------------+
    | ``gqMean``             | Float64 | The average genotype quality across all calls            |
    +------------------------+---------+----------------------------------------------------------+
    | ``gqStDev``            | Float64 | Genotype quality standard deviation across all calls     |
    +------------------------+---------+----------------------------------------------------------+

    Missing values ``NA`` may result from division by zero. The empirical
    standard deviation is computed with zero degrees of freedom.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    name : :obj:`str`
        Name for resulting field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with a new column-indexed field `name`.
    """

    return MatrixTable(Env.hail().methods.SampleQC.apply(dataset._jvds, 'sa.`{}`'.format(name)))

@handle_py4j
@require_biallelic
@typecheck(dataset=MatrixTable, name=strlike)
def variant_qc(dataset, name='variant_qc'):
    """Compute common variant statistics (quality control metrics).

    .. include:: ../_templates/req_biallelic.rst
    .. include:: ../_templates/req_tvariant.rst

    Examples
    --------

    >>> dataset_result = methods.variant_qc(dataset)

    Notes
    -----
    This method computes 18 variant statistics from the genotype data,
    returning a new struct field `name` with the following metrics:

    +---------------------------+---------+--------------------------------------------------------+
    | Name                      | Type    | Description                                            |
    +===========================+=========+========================================================+
    | ``callRate``              | Float64 | Fraction of samples with called genotypes              |
    +---------------------------+---------+--------------------------------------------------------+
    | ``AF``                    | Float64 | Calculated alternate allele frequency (q)              |
    +---------------------------+---------+--------------------------------------------------------+
    | ``AC``                    | Int32   | Count of alternate alleles                             |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rHeterozygosity``       | Float64 | Proportion of heterozygotes                            |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rHetHomVar``            | Float64 | Ratio of heterozygotes to homozygous alternates        |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rExpectedHetFrequency`` | Float64 | Expected rHeterozygosity based on HWE                  |
    +---------------------------+---------+--------------------------------------------------------+
    | ``pHWE``                  | Float64 | p-value from Hardy Weinberg Equilibrium null model     |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nHomRef``               | Int32   | Number of homozygous reference samples                 |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nHet``                  | Int32   | Number of heterozygous samples                         |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nHomVar``               | Int32   | Number of homozygous alternate samples                 |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nCalled``               | Int32   | Sum of ``nHomRef``, ``nHet``, and ``nHomVar``          |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nNotCalled``            | Int32   | Number of uncalled samples                             |
    +---------------------------+---------+--------------------------------------------------------+
    | ``nNonRef``               | Int32   | Sum of ``nHet`` and ``nHomVar``                        |
    +---------------------------+---------+--------------------------------------------------------+
    | ``rHetHomVar``            | Float64 | Het/HomVar ratio across all samples                    |
    +---------------------------+---------+--------------------------------------------------------+
    | ``dpMean``                | Float64 | Depth mean across all samples                          |
    +---------------------------+---------+--------------------------------------------------------+
    | ``dpStDev``               | Float64 | Depth standard deviation across all samples            |
    +---------------------------+---------+--------------------------------------------------------+
    | ``gqMean``                | Float64 | The average genotype quality across all samples        |
    +---------------------------+---------+--------------------------------------------------------+
    | ``gqStDev``               | Float64 | Genotype quality standard deviation across all samples |
    +---------------------------+---------+--------------------------------------------------------+

    Missing values ``NA`` may result from division by zero. The empirical
    standard deviation is computed with zero degrees of freedom.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    name : :obj:`str`
        Name for resulting field.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with a new row-indexed field `name`.
    """

    return MatrixTable(Env.hail().methods.VariantQC.apply(dataset._jvds, 'va.`{}`'.format(name)))
