from hail.api2.matrixtable import MatrixTable, Table
from hail.expr.expression import *
from hail.genetics.ldMatrix import LDMatrix
from hail.typecheck import *
from hail.utils import wrap_to_list
from hail.utils.java import handle_py4j
from .misc import require_biallelic
from hail.expr import functions
import hail.expr.aggregators as agg


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

    return cleanup(MatrixTable(jm))


@handle_py4j
@typecheck(dataset=MatrixTable, force_local=bool)
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

    jldm = Env.hail().methods.LDMatrix.apply(dataset._jvds, force_local)
    return LDMatrix(jldm)


@handle_py4j
@require_biallelic
@typecheck(dataset=MatrixTable,
           k=integral,
           compute_loadings=bool,
           as_array=bool)
def hwe_normalized_pca(dataset, k=10, compute_loadings=False, as_array=False):
    """Run principal component analysis (PCA) on the Hardy-Weinberg-normalized call matrix.

    Examples
    --------

    >>> eigenvalues, scores, loadings = methods.hwe_normalized_pca(dataset, k=15)

    Notes
    -----
    Variants that are all homozygous reference or all homozygous variant are removed before evaluation.

    Parameters
    ----------
    dataset : :class:`MatrixTable`
        Dataset.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.
    as_array : :obj:`bool`
        If ``True``, return scores and loadings as an array field. If ``False``, return
        one field per element (`PC1`, `PC2`, ... `PCk`).

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`Table`, :class:`Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """

    dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.num_alt_alleles()),
                                    n_called=agg.count_where(functions.is_defined(dataset.GT)))
    dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called)).persist()

    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError(
            "Cannot run PCA: found 0 variants after filtering out monomorphic sites.")
    info("Running PCA using {} variants.".format(n_variants))

    # FIXME: use bind
    mean_gt = dataset.AC / dataset.n_called
    entry_expr = functions.cond(functions.is_defined(dataset.GT),
                                (dataset.GT.num_alt_alleles() - mean_gt) /
                                functions.sqrt(mean_gt * (2 - mean_gt) * n_variants / 2),
                                0)
    result = pca(entry_expr,
                 k,
                 compute_loadings,
                 as_array)
    dataset.unpersist()
    return result


@handle_py4j
@typecheck(entry_expr=Expression,
           k=integral,
           compute_loadings=bool,
           as_array=bool)
def pca(entry_expr, k=10, compute_loadings=False, as_array=False):
    """Run principal Component Analysis (PCA) on a matrix table, using `entry_expr` as the numerical entry.

    Examples
    --------

    Compute the top 2 principal component scores and eigenvalues of the call missingness matrix.

    >>> eigenvalues, scores, _ = methods.pca(functions.is_defined(dataset.GT).to_int32(),
    ...                                      k=2)

    Notes
    -----

    PCA computes the SVD

    .. math::

      M = USV^T

    where columns of :math:`U` are left singular vectors (orthonormal in
    :math:`\mathbb{R}^n`), columns of :math:`V` are right singular vectors
    (orthonormal in :math:`\mathbb{R}^m`), and :math:`S=\mathrm{diag}(s_1, s_2,
    \ldots)` with ordered singular values :math:`s_1 \ge s_2 \ge \cdots \ge 0`.
    Typically one computes only the first :math:`k` singular vectors and values,
    yielding the best rank :math:`k` approximation :math:`U_k S_k V_k^T` of
    :math:`M`; the truncations :math:`U_k`, :math:`S_k` and :math:`V_k` are
    :math:`n \\times k`, :math:`k \\times k` and :math:`m \\times k`
    respectively.

    From the perspective of the samples or rows of :math:`M` as data,
    :math:`V_k` contains the variant loadings for the first :math:`k` PCs while
    :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each
    sample. The loadings represent a new basis of features while the scores
    represent the projected data on those features. The eigenvalues of the GRM
    :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2,
    \ldots`, which represent the variances carried by the respective PCs. By
    default, Hail only computes the loadings if the ``loadings`` parameter is
    specified.

    Note
    ----
    In PLINK/GCTA the GRM is taken as the starting point and it is
    computed slightly differently with regard to missing data. Here the
    :math:`ij` entry of :math:`MM^T` is simply the dot product of rows :math:`i`
    and :math:`j` of :math:`M`; in terms of :math:`C` it is

    .. math::

      \\frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}

    where :math:`\mathcal{C}_i = \{l \mid C_{il} \\text{ is non-missing}\}`. In
    PLINK/GCTA the denominator :math:`m` is replaced with the number of terms in
    the sum :math:`\\lvert\mathcal{C}_i\cap\\mathcal{C}_j\\rvert`, i.e. the
    number of variants where both samples have non-missing genotypes. While this
    is arguably a better estimator of the true GRM (trading shrinkage for
    noise), it has the drawback that one loses the clean interpretation of the
    loadings and scores as features and projections.

    Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM; even
    ignoring the above discrepancy that means the left singular vectors
    :math:`U_k` instead of the component scores :math:`U_k S_k`. While this is
    just a matter of the scale on each PC, the scores have the advantage of
    representing true projections of the data onto features with the variance of
    a score reflecting the variance explained by the corresponding feature. (In
    PC bi-plots this amounts to a change in aspect ratio; for use of PCs as
    covariates in regression it is immaterial.)

    Scores are stored in a :class:`Table` with the following fields:

     - **s**: Column key of the dataset.

     - **pcaScores**: The principal component scores. This can have two different
       structures, depending on the value of the `as_array` flag.

    If `as_array` is ``False`` (default), then `pcaScores` is a ``Struct`` with
    fields `PC1`, `PC2`, etc. If `as_array` is ``True``, then `pcaScores` is a
    field of type ``Array[Float64]`` containing the principal component scores.

    Loadings are stored in a :class:`Table` with a structure similar to the scores
    table:

     - **v**: Row key of the dataset.

     - **pcaLoadings**: Row loadings (same type as the scores)

    Parameters
    ----------
    dataset : :class:`MatrixTable`
        Dataset.
    entry_expr : :class:`Expression`
        Numeric expression for matrix entries.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.
    as_array : :obj:`bool`
        If ``True``, return scores and loadings as an array field. If ``False``, return
        one field per element (`PC1`, `PC2`, ... `PCk`).

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`Table`, :class:`Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    source = entry_expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    dataset = source
    base, _ = dataset._process_joins(entry_expr)
    analyze(entry_expr, dataset._entry_indices, set(), set(dataset._fields.keys()))

    r = Env.hail().methods.PCA.apply(dataset._jvds, to_expr(entry_expr)._ast.to_hql(), k, compute_loadings, as_array)
    scores = Table(
        Env.hc(),
        Env.hail().methods.PCA.scoresTable(dataset._jvds, as_array, r._2()))
    loadings = from_option(r._3())
    if loadings:
        loadings = Table(Env.hc(), loadings)
    return (jiterable_to_list(r._1()), scores, loadings)
