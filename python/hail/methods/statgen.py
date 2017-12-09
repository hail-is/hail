from hail.api2.matrixtable import MatrixTable
from hail.expr.expression import *
from hail.genetics.ldMatrix import LDMatrix
from hail.typecheck import *
from hail.utils import wrap_to_list
from hail.utils.java import handle_py4j


@typecheck(dataset=MatrixTable,
           ys=oneof(Expression, listof(Expression)),
           x=Expression,
           covariates=listof(Expression),
           root=strlike,
           block_size=integral)
def linreg(dataset, ys, x, covariates=[], root='linreg', block_size=16):
    """Test each variant for association with multiple phenotypes using linear regression.

    .. warning::

        :py:meth:`.linreg` uses the same set of samples for each phenotype,
        namely the set of samples for which **all** phenotypes and covariates are defined.

    **Annotations**

    With the default root, the following four variant annotations are added.
    The indexing of the array annotations corresponds to that of ``y``.

    - **va.linreg.nCompleteSamples** (*Int*) -- number of samples used
    - **va.linreg.AC** (*Double*) -- sum of input values ``x``
    - **va.linreg.ytx** (*Array[Double]*) -- array of dot products of each response vector ``y`` with the input vector ``x``
    - **va.linreg.beta** (*Array[Double]*) -- array of fit effect coefficients, :math:`\hat\beta_1`
    - **va.linreg.se** (*Array[Double]*) -- array of estimated standard errors, :math:`\widehat{\mathrm{se}}`
    - **va.linreg.tstat** (*Array[Double]*) -- array of :math:`t`-statistics, equal to :math:`\hat\beta_1 / \widehat{\mathrm{se}}`
    - **va.linreg.pval** (*Array[Double]*) -- array of :math:`p`-values

    :param ys: list of one or more response expressions.
    :type ys: list of str

    :param str x: expression for input variable

    :param covariates: list of covariate expressions.
    :type covariates: list of str

    :param str root: Variant annotation path to store result of linear regression.

    :param int variant_block_size: Number of variant regressions to perform simultaneously.  Larger block size requires more memmory.

    :return: Variant dataset with linear regression variant annotations.
    :rtype: :py:class:`.VariantDataset`

    """
    all_exprs = [x]

    ys = wrap_to_list(ys)

    # x is entry-indexed
    analyze(x, dataset._entry_indices, set(), set(dataset._fields.keys()))

    # ys and covariates are col-indexed
    for e in (tuple(wrap_to_list(ys)) + tuple(covariates)):
        all_exprs.append(e)
        analyze(e, dataset._col_indices, set(), set(dataset._fields.keys()))

    base, cleanup = dataset._process_joins(*all_exprs)

    jm = base._jvds.linreg(
        jarray(Env.jvm().java.lang.String, [y._ast.to_hql() for y in ys]),
        x._ast.to_hql(),
        jarray(Env.jvm().java.lang.String, [cov._ast.to_hql() for cov in covariates]),
        'va.`{}`'.format(root),
        block_size
    )

    return cleanup(MatrixTable(dataset._hc, jm))


@handle_py4j
@typecheck(dataset=anytype, force_local=bool)
def ld_matrix(dataset, force_local=False):
    """Computes the linkage disequilibrium (correlation) matrix for the variants in this VDS.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    .. testsetup::

        dataset = vds.annotate_samples_expr('sa = drop(sa, qc)').to_hail2()
        from hail.methods import ld_matrix

    **Examples**

    >>> ld_matrix = ld_matrix(dataset)

    **Notes**

    Each entry (i, j) in the LD matrix gives the :math:`r` value between variants i and j, defined as
    `Pearson's correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
    :math:`\\rho_{x_i,x_j}` between the two genotype vectors :math:`x_i` and :math:`x_j`.

    .. math::

        \\rho_{x_i,x_j} = \\frac{\\mathrm{Cov}(X_i,X_j)}{\\sigma_{X_i} \\sigma_{X_j}}

    Also note that variants with zero variance (:math:`\\sigma = 0`) will be dropped from the matrix.

    .. caution::

        The matrix returned by this function can easily be very large with most entries near zero
        (for example, entries between variants on different chromosomes in a homogenous population).
        Most likely you'll want to reduce the number of variants with methods like
        :py:meth:`.sample_variants`, :py:meth:`.filter_variants_expr`, or :py:meth:`.ld_prune` before
        calling this unless your dataset is very small.

    :param dataset: Variant-keyed dataset.
    :type dataset: :py:class:`.MatrixTable`

    :param bool force_local: If true, the LD matrix is computed using local matrix multiplication on the Spark driver.
        This may improve performance when the genotype matrix is small enough to easily fit in local memory.
        If false, the LD matrix is computed using distributed matrix multiplication if the number of entries
        exceeds :math:`5000^2` and locally otherwise.

    :return: Matrix of r values between pairs of variants.
    :rtype: :py:class:`LDMatrix`
    """

    jldm = dataset._jvdf.ldMatrix(force_local)
    return LDMatrix(jldm)
