from hail.typecheck import *
from hail.utils.java import handle_py4j
from hail.api2 import MatrixTable

@handle_py4j
@typecheck_method(dataset=MatrixTable, root=strlike)
def sample_qc(dataset, name='sample_qc'):
    """Compute per-sample metrics useful for quality control.

    .. include:: ../_templates/req_tvariant.rst

    **Examples**

    .. testsetup ::
        dataset = vds1.to_hail2()
        from hail2.genetics import sample_qc

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

    The empirical standard deviation is computed with zero degrees of freedom.

    :param str name: Field name for the computed struct.

    :return: Annotated matrix table.
    :rtype: :class:`.MatrixTable`
    """

    return MatrixTable(dataset.hc, dataset._jvds.sampleQC('sa.`{}`'.format(name)))
