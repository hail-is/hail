from hail.api2.matrixtable import MatrixTable, Table
from hail.expr.expression import *
from hail.genetics import KinshipMatrix
from hail.genetics.ldMatrix import LDMatrix
from hail.linalg import BlockMatrix
from hail.typecheck import *
from hail.utils import wrap_to_list, new_temp_file, info
from hail.utils.java import handle_py4j, joption
from .misc import require_biallelic, require_unique_samples
from hail.expr import functions
import hail.expr.aggregators as agg
from math import sqrt


@handle_py4j
@typecheck(dataset=MatrixTable,
           maf=nullable(oneof(Float32Expression, Float64Expression)),
           bounded=bool,
           min=nullable(numeric),
           max=nullable(numeric))
def ibd(dataset, maf=None, bounded=True, min=None, max=None):
    """Compute matrix of identity-by-descent estimations.

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    To calculate a full IBD matrix, using minor allele frequencies computed
    from the dataset itself:

    >>> methods.ibd(dataset)

    To calculate an IBD matrix containing only pairs of samples with
    ``PI_HAT`` in :math:`[0.2, 0.9]`, using minor allele frequencies stored in
    the row field `panel_maf`:

    >>> methods.ibd(dataset, maf=dataset['panel_maf'], min=0.2, max=0.9)

    Notes
    -----

    The implementation is based on the IBD algorithm described in the `PLINK
    paper <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838>`__.

    :meth:`ibd` requires the dataset to be biallelic and does not perform LD
    pruning. Linkage disequilibrium may bias the result so consider filtering
    variants first.

    The resulting :class:`.Table` entries have the type: *{ i: String,
    j: String, ibd: { Z0: Double, Z1: Double, Z2: Double, PI_HAT: Double },
    ibs0: Long, ibs1: Long, ibs2: Long }*. The key list is: `*i: String, j:
    String*`.

    Conceptually, the output is a symmetric, sample-by-sample matrix. The
    output table has the following form

    .. code-block:: text

        i		j	ibd.Z0	ibd.Z1	ibd.Z2	ibd.PI_HAT ibs0	ibs1	ibs2
        sample1	sample2	1.0000	0.0000	0.0000	0.0000 ...
        sample1	sample3	1.0000	0.0000	0.0000	0.0000 ...
        sample1	sample4	0.6807	0.0000	0.3193	0.3193 ...
        sample1	sample5	0.1966	0.0000	0.8034	0.8034 ...

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        A variant-keyed :class:`.MatrixTable` containing genotype information.

    maf : :class:`.Float32Expression`, :class:`.Float64Expression` or :obj:`None`
        (optional) expression on `dataset` for the minor allele frequency.

    bounded : :obj:`bool`
        Forces the estimations for `Z0``, ``Z1``, ``Z2``, and ``PI_HAT`` to take
        on biologically meaningful values (in the range [0,1]).

    min : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` below this value will
        not be included in the output. Must be in :math:`[0,1]`.

    max : :obj:`float` or :obj:`None`
        Sample pairs with a ``PI_HAT`` above this value will
        not be included in the output. Must be in :math:`[0,1]`.

    Return
    ------
    :class:`.Table`
        A table which maps pairs of samples to their IBD statistics
    """

    if maf is not None:
        analyze('ibd/maf', maf, dataset._row_indices)
        dataset, _ = dataset._process_joins(maf)
        maf = maf._ast.to_hql()

    return Table(Env.hail().methods.IBD.apply(require_biallelic(dataset, 'ibd')._jvds,
                                              joption(maf),
                                              bounded,
                                              joption(min),
                                              joption(max)))

@typecheck(dataset=MatrixTable,
           ys=oneof(Expression, listof(Expression)),
           x=Expression,
           covariates=listof(Expression),
           root=strlike,
           block_size=integral)
def linreg(dataset, ys, x, covariates=[], root='linreg', block_size=16):
    """For each row, test a derived input variable for association with response variables using linear regression.

    Examples
    --------

    >>> dataset_result = methods.linreg(dataset, [dataset.pheno.height], dataset.GT.num_alt_alleles(),
    ...                                 covariates=[dataset.pheno.age, dataset.pheno.isFemale])

    Warning
    -------
    :meth:`.linreg` considers the same set of columns (i.e., samples, points) for every response variable and row,
    namely those columns for which **all** response variables and covariates are defined.
    For each row, missing values of ``x`` are mean-imputed over these columns.

    Notes
    -----

    With the default root, the following row-indexed fields are added.
    The indexing of the array fields corresponds to that of ``ys``.

    - **linreg.nCompleteSamples** (*Int32*) -- number of columns used
    - **linreg.AC** (*Float64*) -- sum of input values ``x``
    - **linreg.ytx** (*Array[Float64]*) -- array of dot products of each response vector ``y`` with the input vector ``x``
    - **linreg.beta** (*Array[Float64]*) -- array of fit effect coefficients of ``x``, :math:`\hat\\beta_1` below
    - **linreg.se** (*Array[Float64]*) -- array of estimated standard errors, :math:`\widehat{\mathrm{se}}_1`
    - **linreg.tstat** (*Array[Float64]*) -- array of :math:`t`-statistics, equal to :math:`\hat\\beta_1 / \widehat{\mathrm{se}}_1`
    - **linreg.pval** (*Array[Float64]*) -- array of :math:`p`-values

    In the statistical genetics example above, the input variable ``x`` encodes genotype
    as the number of alternate alleles (0, 1, or 2). For each variant (row), genotype is tested for association
    with height controlling for age and sex, by fitting the linear regression model:

    .. math::

        \mathrm{height} = \\beta_0 + \\beta_1 \, \mathrm{genotype} + \\beta_2 \, \mathrm{age} + \\beta_3 \, \mathrm{isFemale} + \\varepsilon, \quad \\varepsilon \sim \mathrm{N}(0, \sigma^2)

    Boolean covariates like :math:`\mathrm{isFemale}` are encoded as 1 for true and 0 for false.
    The null model sets :math:`\\beta_1 = 0`.

    The standard least-squares linear regression model is derived in Section
    3.2 of `The Elements of Statistical Learning, 2nd Edition
    <http://statweb.stanford.edu/~tibs/ElemStatLearn/printings/ESLII_print10.pdf>`__. See
    equation 3.12 for the t-statistic which follows the t-distribution with
    :math:`n - k - 2` degrees of freedom, under the null hypothesis of no
    effect, with :math:`n` samples and :math:`k` covariates in addition to
    ``x`` and the intercept.

    Parameters
    ----------
    ys : :obj:`list` of :class:`.Expression`
        One or more response expressions.
    x : :class:`.Expression`
        Input variable.
    covariates : :obj:`list` of :class:`.Expression`
        Covariate expressions.
    root : :obj:`str`
        Name of resulting row-indexed field.
    block_size : :obj:`int`
        Number of row regressions to perform simultaneously per core. Larger blocks
        require more memory but may improve performance.

    Returns
    -------
    :class:`.MatrixTable`
        Dataset with regression results in a new row-indexed field.
    """

    all_exprs = [x]

    ys = wrap_to_list(ys)

    # x is entry-indexed
    analyze('linreg/x', x, dataset._entry_indices)

    # ys and covariates are col-indexed
    ys = wrap_to_list(ys)
    for e in ys:
        all_exprs.append(e)
        analyze('linreg/ys', e, dataset._col_indices)
    for e in covariates:
        all_exprs.append(e)
        analyze('linreg/covariates', e, dataset._col_indices)

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
        :meth:`.sample_variants`, :meth:`.filter_variants_expr`, or :meth:`.ld_prune` before
        calling this unless your dataset is very small.

    :param dataset: Variant-keyed dataset.
    :type dataset: :class:`.MatrixTable`

    :param bool force_local: If true, the LD matrix is computed using local matrix multiplication on the Spark driver.
        This may improve performance when the genotype matrix is small enough to easily fit in local memory.
        If false, the LD matrix is computed using distributed matrix multiplication if the number of entries
        exceeds :math:`5000^2` and locally otherwise.

    :return: Matrix of r values between pairs of variants.
    :rtype: :class:`.LDMatrix`
    """

    jldm = Env.hail().methods.LDMatrix.apply(require_biallelic(dataset, 'ld_matrix')._jvds, force_local)
    return LDMatrix(jldm)


@handle_py4j
@typecheck(dataset=MatrixTable,
           k=integral,
           compute_loadings=bool,
           as_array=bool)
def hwe_normalized_pca(dataset, k=10, compute_loadings=False, as_array=False):
    """Run principal component analysis (PCA) on the Hardy-Weinberg-normalized call matrix.

    Examples
    --------

    >>> eigenvalues, scores, loadings = methods.hwe_normalized_pca(dataset, k=5)

    Notes
    -----
    Variants that are all homozygous reference or all homozygous variant are removed before evaluation.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
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
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    dataset = require_biallelic(dataset, 'hwe_normalized_pca')
    dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.num_alt_alleles()),
                                    n_called=agg.count_where(functions.is_defined(dataset.GT)))
    dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called))

    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError(
            "Cannot run PCA: found 0 variants after filtering out monomorphic sites.")
    info("Running PCA using {} variants.".format(n_variants))

    dataset = dataset.annotate_rows(mean_gt = dataset.AC / dataset.n_called)
    dataset = dataset.annotate_rows(hwe_std_dev = functions.sqrt(dataset.mean_gt * (2 - dataset.mean_gt) * n_variants / 2))

    entry_expr = functions.or_else((dataset.GT.num_alt_alleles() - dataset.mean_gt) /
                                   dataset.hwe_std_dev,
                                   0.0)
    result = pca(entry_expr,
                 k,
                 compute_loadings,
                 as_array)
    return result


@handle_py4j
@typecheck(entry_expr=expr_numeric,
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

    Scores are stored in a :class:`.Table` with the following fields:

     - **s**: Column key of the dataset.

     - **pcaScores**: The principal component scores. This can have two different
       structures, depending on the value of the `as_array` flag.

    If `as_array` is ``False`` (default), then `pcaScores` is a ``Struct`` with
    fields `PC1`, `PC2`, etc. If `as_array` is ``True``, then `pcaScores` is a
    field of type ``Array[Float64]`` containing the principal component scores.

    Loadings are stored in a :class:`.Table` with a structure similar to the scores
    table:

     - **v**: Row key of the dataset.

     - **pcaLoadings**: Row loadings (same type as the scores)

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    entry_expr : :class:`.Expression`
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
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    source = entry_expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    dataset = source
    base, _ = dataset._process_joins(entry_expr)
    analyze('pca', entry_expr, dataset._entry_indices)

    r = Env.hail().methods.PCA.apply(dataset._jvds, to_expr(entry_expr)._ast.to_hql(), k, compute_loadings, as_array)
    scores = Table(Env.hail().methods.PCA.scoresTable(dataset._jvds, as_array, r._2()))
    loadings = from_option(r._3())
    if loadings:
        loadings = Table(loadings)
    return (jiterable_to_list(r._1()), scores, loadings)

@handle_py4j
@typecheck(ds=MatrixTable,
           k=integral,
           maf=numeric,
           path=nullable(strlike),
           block_size=integral,
           min_kinship=numeric,
           statistics=enumeration("phi", "phik2", "phik2k0", "all"),
           pca_partitions=nullable(integral))
def pc_relate(ds, k, maf, path=None, block_size=512, min_kinship=-float("inf"), statistics="all", pca_partitions=None):
    """Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/experimental.rst

    .. include:: ../_templates/req_tvariant.rst

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    Estimate kinship, identity-by-descent two, identity-by-descent one, and
    identity-by-descent zero for every pair of samples, using 10 prinicpal
    components to correct for ancestral populations, and a minimum minor
    allele frequency filter of 0.01:

    >>> rel = vds.pc_relate(10, 0.01)

    Calculate values as above, but when performing distributed matrix
    multiplications use a matrix-block-size of 1024 by 1024.

    >>> rel = vds.pc_relate(10, 0.01, 1024)

    Calculate values as above, excluding sample-pairs with kinship less
    than 0.1. This is more efficient than producing the full table and
    filtering using :meth:`.Table.filter`.

    >>> rel = vds.pc_relate(5, 0.01, min_kinship=0.1)


    The traditional estimator for kinship between a pair of individuals
    :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
    single-nucleotide variants, from a population with allele frequencies
    :math:`p_s`, is given by:

    .. math::

      \\widehat{\\phi_{ij}} :=
        \\frac{1}{|S_{ij}|}
        \\sum_{s \\in S_{ij}}
          \\frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}
                {4 * \\sum_{s \\in S_{ij} p_s (1 - p_s)}}

    This estimator is true under the model that the sharing of common
    (relative to the population) alleles is not very informative to
    relatedness (because they're common) and the sharing of rare alleles
    suggests a recent common ancestor from which the allele was inherited by
    descent.

    When multiple ancestry groups are mixed in a sample, this model breaks
    down. Alleles that are rare in all but one ancestry group are treated as
    very informative to relatedness. However, these alleles are simply
    markers of the ancestry group. The PC-Relate method corrects for this
    situation and the related situation of admixed individuals.

    PC-Relate slightly modifies the usual estimator for relatedness:
    occurences of population allele frequency are replaced with an
    "individual-specific allele frequency". This modification allows the
    method to correctly weight an allele according to an individual's unique
    ancestry profile.

    The "individual-specific allele frequency" at a given genetic locus is
    modeled by PC-Relate as a linear function of their first ``k`` principal
    component coordinates. As such, the efficacy of this method rests on two
    assumptions:

     - an individual's first ``k`` principal component coordinates fully
       describe their allele-frequency-relevant ancestry, and

     - the relationship between ancestry (as described by principal
       component coordinates) and population allele frequency is linear

    The estimators for kinship, and identity-by-descent zero, one, and two
    follow. Let:

     - :math:`S_{ij}` be the set of genetic loci at which both individuals
       :math:`i` and :math:`j` have a defined genotype

     - :math:`g_{is} \\in {0, 1, 2}` be the number of alternate alleles that
       individual :math:`i` has at gentic locus :math:`s`

     - :math:`\\widehat{\\mu_{is}} \\in [0, 1]` be the individual-specific allele
       frequency for individual :math:`i` at genetic locus :math:`s`

     - :math:`{\\widehat{\\sigma^2_{is}}} := \\widehat{\\mu_{is}} (1 - \\widehat{\\mu_{is}})`,
       the binomial variance of :math:`\\widehat{\\mu_{is}}`

     - :math:`\\widehat{\\sigma_{is}} := \\sqrt{\\widehat{\\sigma^2_{is}}}`,
       the binomial standard deviation of :math:`\\widehat{\\mu_{is}}`

     - :math:`\\text{IBS}^{(0)}_{ij} := \\sum_{s \\in S_{ij}} \\mathbb{1}_{||g_{is} - g_{js} = 2||}`,
       the number of genetic loci at which individuals :math:`i` and :math:`j`
       share no alleles

     - :math:`\\widehat{f_i} := 2 \\widehat{\\phi_{ii}} - 1`, the inbreeding
       coefficient for individual :math:`i`

     - :math:`g^D_{is}` be a dominance encoding of the genotype matrix, and
       :math:`X_{is}` be a normalized dominance-coded genotype matrix

    .. math::

        g^D_{is} :=
          \\begin{cases}
            \\widehat{\\mu_{is}}     & g_{is} = 0 \\\\
            0                        & g_{is} = 1 \\\\
            1 - \\widehat{\\mu_{is}} & g_{is} = 2
          \\end{cases}

        X_{is} := g^D_{is} - \\widehat{\\sigma^2_{is}} (1 - \\widehat{f_i})

    The estimator for kinship is given by:

    .. math::

      \\widehat{\phi_{ij}} :=
        \\frac{\sum_{s \\in S_{ij}}(g - 2 \\mu)_{is} (g - 2 \\mu)_{js}}
              {4 * \\sum_{s \\in S_{ij}}
                            \\widehat{\\sigma_{is}} \\widehat{\\sigma_{js}}}

    The estimator for identity-by-descent two is given by:

    .. math::

      \\widehat{k^{(2)}_{ij}} :=
        \\frac{\\sum_{s \\in S_{ij}}X_{is} X_{js}}{\sum_{s \\in S_{ij}}
          \\widehat{\\sigma^2_{is}} \\widehat{\\sigma^2_{js}}}

    The estimator for identity-by-descent zero is given by:

    .. math::

      \\widehat{k^{(0)}_{ij}} :=
        \\begin{cases}
          \\frac{\\text{IBS}^{(0)}_{ij}}
                {\\sum_{s \\in S_{ij}}
                       \\widehat{\\mu_{is}}^2(1 - \\widehat{\\mu_{js}})^2
                       + (1 - \\widehat{\\mu_{is}})^2\\widehat{\\mu_{js}}^2}
            & \\widehat{\\phi_{ij}} > 2^{-5/2} \\\\
          1 - 4 \\widehat{\\phi_{ij}} + k^{(2)}_{ij}
            & \\widehat{\\phi_{ij}} \\le 2^{-5/2}
        \\end{cases}

    The estimator for identity-by-descent one is given by:

    .. math::

      \\widehat{k^{(1)}_{ij}} :=
        1 - \\widehat{k^{(2)}_{ij}} - \\widehat{k^{(0)}_{ij}}

    Notes
    -----
    The PC-Relate method is described in "Model-free Estimation of Recent
    Genetic Relatedness". Conomos MP, Reiner AP, Weir BS, Thornton TA. in
    American Journal of Human Genetics. 2016 Jan 7. The reference
    implementation is available in the `GENESIS Bioconductor package
    <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`_ .

    :func:`methods.pc_relate` differs from the reference
    implementation in a couple key ways:

     - the principal components analysis does not use an unrelated set of
       individuals

     - the estimators do not perform small sample correction

     - the algorithm does not provide an option to use population-wide
       allele frequency estimates

     - the algorithm does not provide an option to not use "overall
       standardization" (see R ``pcrelate`` documentation)

    Note
    ----
    The ``block_size`` controls memory usage and parallelism. If it is large
    enough to hold an entire sample-by-sample matrix of 64-bit doubles in
    memory, then only one Spark worker node can be used to compute matrix
    operations. If it is too small, communication overhead will begin to
    dominate the computation's time. The author has found that on Google
    Dataproc (where each core has about 3.75GB of memory), setting
    ``block_size`` larger than 512 tends to cause memory exhaustion errors.

    Note
    ----
    The minimum allele frequency filter is applied per-pair: if either of
    the two individual's individual-specific minor allele frequency is below
    the threshold, then the variant's contribution to relatedness estimates
    is zero.

    Note
    ----
    Under the PC-Relate model, kinship, :math:`\\phi_{ij}`, ranges from 0 to
    0.5, and is precisely half of the
    fraction-of-genetic-material-shared. Listed below are the statistics for
    a few pairings:

     - Monozygotic twins share all their genetic material so their kinship
       statistic is 0.5 in expection.

     - Parent-child and sibling pairs both have kinship 0.25 in expectation
       and are separated by the identity-by-descent-zero, :math:`k^{(2)}_{ij}`,
       statistic which is zero for parent-child pairs and 0.25 for sibling
       pairs.

     - Avuncular pairs and grand-parent/-child pairs both have kinship 0.125
       in expectation and both have identity-by-descent-zero 0.5 in expectation

     - "Third degree relatives" are those pairs sharing
       :math:`2^{-3} = 12.5 %` of their genetic material, the results of
       PCRelate are often too noisy to reliably distinguish these pairs from
       higher-degree-relative-pairs or unrelated pairs.



    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset in which to compute relatedness
    k : :obj:`int`
        The number of principal components to use to distinguish ancestries.
    maf : :obj:`float`
        The minimum individual-specific allele frequency for an allele used to
        measure relatedness.
    path : :obj:`str` or :obj:`None`
        A temporary directory to store intermediate matrices. Storing the matrices
        to a file system is necessary for reliable execution of this method.
    block_size : :obj:`int`
        the side length of the blocks of the block-distributed matrices; this
        should be set such that at least three of these matrices fit in memory
        (in addition to all other objects necessary for Spark and Hail).
    min_kinship : :obj:`float`
        Pairs of samples with kinship lower than ``min_kinship`` are excluded
        from the results.
    statistics : :obj:`str`
        the set of statistics to compute, `phi` will only compute the
        kinship statistic, `phik2` will compute the kinship and
        identity-by-descent two statistics, `phik2k0` will compute the
        kinship statistics and both identity-by-descent two and zero, `all`
        computes the kinship statistic and all three identity-by-descent
        statistics.
    pca_partitions : :obj:`int` or :obj:`None`
        the number of partitions to use when performing PCA, smaller is
        better. If you are unsure how many to use, try one-tenth of the number
        of partitions in the ``ds``

    Returns
    -------
    :class:`.Table`
        A :class:`.Table` mapping pairs of samples to estimations of their
        kinship and identity-by-descent zero, one, and two.

        The fields of the resulting :class:`.Table` entries are of types:
        `i`: ``ds.colkey_schema``, `j`: ``ds.colkey_schema``, `kin`: `Double`, `k2`: `Double`,
        `k1`: `Double`, `k0`: `Double`. The table is keyed by `i` and `j`.

    """
    _, scores, _ = hwe_normalized_pca(ds if pca_partitions is None else ds.repartition(pca_partitions), k, False, True)

    return pc_relate_with_scores(ds, scores, maf, path, block_size, min_kinship, statistics)

@handle_py4j
@typecheck(ds=MatrixTable,
           scores=Table,
           maf=numeric,
           path=nullable(strlike),
           block_size=integral,
           min_kinship=numeric,
           statistics=enumeration("phi", "phik2", "phik2k0", "all"))
def pc_relate_with_scores(ds, scores, maf, path=None, block_size=512, min_kinship=-float("inf"), statistics="all"):
    ds = require_biallelic(ds, 'pc_relate')
    ds = require_unique_samples(ds, 'pc_relate')
    ds = ds.annotate_rows(mean_gt =
                          agg.sum(ds.GT.num_alt_alleles()).to_float64() /
                          agg.count_where(functions.is_defined(ds.GT)))

    mean_imputed_gt = functions.or_else(ds.GT.num_alt_alleles().to_float64(), ds.mean_gt)

    g = BlockMatrix.from_matrix_table(mean_imputed_gt, path=path, block_size=block_size)

    int_statistics = {"phi": 0, "phik2": 1, "phik2k0": 2, "all": 3}[statistics]
    return Table(
        scala_object(Env.hail().methods, 'PCRelate')
        .apply(Env.hc()._jhc,
               g._jbm,
               jarray(Env.jvm().java.lang.Object, [ds.colkey_schema._convert_to_j(x) for x in ds.col_keys()]),
               ds.colkey_schema._jtype,
               scores._jt,
               maf,
               block_size,
               min_kinship,
               int_statistics))

@handle_py4j
@typecheck(dataset=MatrixTable,
           fraction=numeric,
           seed=integral)
def sample_rows(dataset, fraction, seed=1):
    """Downsample rows to a given fraction of the dataset.

    Examples
    --------

    >>> small_dataset = methods.sample_rows(dataset, 0.01)

    Notes
    -----

    This method may not sample exactly ``(fraction * n_rows)`` rows from
    the dataset.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset to sample from.
    fraction : :obj:`float`
        (Expected) fraction of rows to keep.
    seed : :obj:`int`
        Random seed.

    Returns
    ------
    :class:`.MatrixTable`
        Downsampled matrix table.
    """

    return MatrixTable(dataset._jvds.sampleVariants(fraction, seed))

@handle_py4j
@typecheck(ds=MatrixTable,
           keep_star=bool,
           left_aligned=bool)
def split_multi_hts(ds, keep_star=False, left_aligned=False):
    """Split multiallelic variants for HTS :meth:`.MatrixTable.entry_schema`:

    .. code-block:: text

      Struct {
        GT: Call,
        AD: Array[!Int32],
        DP: Int32,
        GQ: Int32,
        PL: Array[!Int32].
      }

    For generic genotype schema, use :meth:`.split_multi`.

    Examples
    --------

    >>> methods.split_multi_hts(dataset).write('output/split.vds')

    Notes
    -----

    We will explain by example. Consider a hypothetical 3-allelic
    variant:

    .. code-block:: text

      A   C,T 0/2:7,2,6:15:45:99,50,99,0,45,99

    split_multi will create two biallelic variants (one for each
    alternate allele) at the same position

    .. code-block:: text

      A   C   0/0:13,2:15:45:0,45,99
      A   T   0/1:9,6:15:50:50,0,99

    Each multiallelic `GT` field is downcoded once for each alternate allele. A
    call for an alternate allele maps to 1 in the biallelic variant
    corresponding to itself and 0 otherwise. For example, in the example above,
    0/2 maps to 0/0 and 0/1. The genotype 1/2 maps to 0/1 and 0/1.

    The biallelic alt `AD` entry is just the multiallelic `AD` entry
    corresponding to the alternate allele. The ref AD entry is the sum of the
    other multiallelic entries.

    The biallelic `DP` is the same as the multiallelic `DP`.

    The biallelic `PL` entry for a genotype g is the minimum over `PL` entries
    for multiallelic genotypes that downcode to g. For example, the `PL` for (A,
    T) at 0/1 is the minimum of the PLs for 0/1 (50) and 1/2 (45), and thus 45.

    Fixing an alternate allele and biallelic variant, downcoding gives a map
    from multiallelic to biallelic alleles and genotypes. The biallelic `AD` entry
    for an allele is just the sum of the multiallelic `AD` entries for alleles
    that map to that allele. Similarly, the biallelic `PL` entry for a genotype is
    the minimum over multiallelic `PL` entries for genotypes that map to that
    genotype.

    `GQ` is recomputed from `PL`.

    Here is a second example for a het non-ref

    .. code-block:: text

      A   C,T 1/2:2,8,6:16:45:99,50,99,45,0,99

    splits as

    .. code-block:: text

      A   C   0/1:8,8:16:45:45,0,99
      A   T   0/1:10,6:16:50:50,0,99

    **VCF Info Fields**

    Hail does not split fields in the info field. This means that if a
    multiallelic site with `info.AC` value ``[10, 2]`` is split, each split
    site will contain the same array ``[10, 2]``. The provided allele index
    field `aIndex` can be used to select the value corresponding to the split
    allele's position:

    >>> ds = methods.split_multi_hts(dataset)
    >>> ds = ds.filter_rows(ds.info.AC[ds.aIndex - 1] < 10, keep = False)

    VCFs split by Hail and exported to new VCFs may be
    incompatible with other tools, if action is not taken
    first. Since the "Number" of the arrays in split multiallelic
    sites no longer matches the structure on import ("A" for 1 per
    allele, for example), Hail will export these fields with
    number ".".

    If the desired output is one value per site, then it is
    possible to use annotate_variants_expr to remap these
    values. Here is an example:

    >>> ds = methods.split_multi_hts(dataset)
    >>> ds = ds.annotate_rows(info = Struct(AC=ds.info.AC[ds.aIndex - 1], **ds.info)) # doctest: +SKIP
    >>> methods.export_vcf(ds, 'output/export.vcf') # doctest: +SKIP

    The info field AC in *data/export.vcf* will have ``Number=1``.

    **New Fields**

    :meth:`.split_multi_hts` adds the following fields:

     - `wasSplit` (*Boolean*) -- ``True`` if this variant was originally
       multiallelic, otherwise ``False``.

     - `aIndex` (*Int*) -- The original index of this alternate allele in the
       multiallelic representation (NB: 1 is the first alternate allele or the
       only alternate allele in a biallelic variant). For example, 1:100:A:T,C
       splits into two variants: 1:100:A:T with ``aIndex = 1`` and 1:100:A:C
       with ``aIndex = 2``.

    Parameters
    ----------
    keep_star : :obj:`bool`
        Do not filter out * alleles.
    left_aligned : :obj:`bool`
        If ``True``, variants are assumed to be left
        aligned and have unique loci. This avoids a shuffle. If the assumption
        is violated, an error is generated.

    Returns
    -------
    :class:`.MatrixTable`
        A biallelic variant dataset.

    """

    hts_genotype_schema = TStruct._from_java(Env.hail().variant.Genotype.htsGenotypeType().deepOptional())
    if ds.entry_schema._deep_optional() != hts_genotype_schema:
        raise FatalError("""
split_multi_hts: entry schema must be the HTS genotype schema
  found: {}
  expected: {}

hint: Use `split_multi` to split entries with a non-HTS genotype schema.
""".format(ds.entry_schema, hts_genotype_schema))

    variant_expr = 'va.aIndex = aIndex, va.wasSplit = wasSplit'
    genotype_expr = '''
g = let
  newgt = downcode(g.GT, aIndex) and
  newad = if (isDefined(g.AD))
      let sum = g.AD.sum() and adi = g.AD[aIndex] in [sum - adi, adi]
    else
      NA: Array[Int] and
  newpl = if (isDefined(g.PL))
      range(3).map(i => range(g.PL.length).filter(j => downcode(Call(j), aIndex) == Call(i)).map(j => g.PL[j]).min())
    else
      NA: Array[Int] and
  newgq = gqFromPL(newpl)
in { GT: newgt, AD: newad, DP: g.DP, GQ: newgq, PL: newpl }
'''
    jds = scala_object(Env.hail().methods, 'SplitMulti').apply(
        ds._jvds, variant_expr, genotype_expr, keep_star, left_aligned)
    return MatrixTable(jds)

@typecheck(dataset=MatrixTable)
def grm(dataset):
    """Compute the Genetic Relatedness Matrix (GRM).

    .. include:: ../_templates/req_tvariant.rst
    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    >>> km = methods.grm(dataset)

    Notes
    -----

    The genetic relationship matrix (GRM) :math:`G` encodes genetic correlation
    between each pair of samples. It is defined by :math:`G = MM^T` where
    :math:`M` is a standardized version of the genotype matrix, computed as
    follows. Let :math:`C` be the :math:`n \\times m` matrix of raw genotypes
    in the variant dataset, with rows indexed by :math:`n` samples and columns
    indexed by :math:`m` bialellic autosomal variants; :math:`C_{ij}` is the
    number of alternate alleles of variant :math:`j` carried by sample
    :math:`i`, which can be 0, 1, 2, or missing. For each variant :math:`j`,
    the sample alternate allele frequency :math:`p_j` is computed as half the
    mean of the non-missing entries of column :math:`j`. Entries of :math:`M`
    are then mean-centered and variance-normalized as

    .. math::

        M_{ij} = \\frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},

    with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype
    imputation). This scaling normalizes genotype variances to a common value
    :math:`1/m` for variants in Hardy-Weinberg equilibrium and is further
    motivated in the paper `Patterson, Price and Reich, 2006
    <http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190>`__.
    (The resulting amplification of signal from the low end of the allele
    frequency spectrum will also introduce noise for rare variants; common
    practice is to filter out variants with minor allele frequency below some
    cutoff.) The factor :math:`1/m` gives each sample row approximately unit
    total variance (assuming linkage equilibrium) so that the diagonal entries
    of the GRM are approximately 1. Equivalently,

    .. math::

        G_{ik} = \\frac{1}{m} \\sum_{j=1}^m \\frac{(C_{ij}-2p_j)(C_{kj}-2p_j)}{2 p_j (1-p_j)}

    Warning
    -------
    Since Hardy-Weinberg normalization cannot be applied to variants that
    contain only reference alleles or only alternate alleles, all such variants
    are removed prior to calcularing the GRM.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset to sample from.

    Returns
    -------
    :class:`.genetics.KinshipMatrix`
        Genetic Relatedness Matrix for all samples.
    """

    dataset = require_biallelic(dataset, "grm")
    dataset = dataset.annotate_rows(AC=agg.sum(dataset.GT.num_alt_alleles()),
                                    n_called=agg.count_where(functions.is_defined(dataset.GT)))
    dataset = dataset.filter_rows((dataset.AC > 0) & (dataset.AC < 2 * dataset.n_called)).persist()

    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError("Cannot run GRM: found 0 variants after filtering out monomorphic sites.")
    info("Computing GRM using {} variants.".format(n_variants))

    normalized_genotype_expr = functions.bind(
        dataset.AC / dataset.n_called,
        lambda mean_gt: functions.cond(functions.is_defined(dataset.GT),
                                       (dataset.GT.num_alt_alleles() - mean_gt) /
                                       functions.sqrt(mean_gt * (2 - mean_gt) * n_variants / 2),
                                       0))

    bm = BlockMatrix.from_matrix_table(normalized_genotype_expr)
    dataset.unpersist()
    grm = bm.T.dot(bm)

    return KinshipMatrix._from_block_matrix(dataset.colkey_schema,
                                      grm,
                                      dataset.col_keys(),
                                      n_variants)

@handle_py4j
@typecheck(call_expr=CallExpression)
def rrm(call_expr):
    """Computes the Realized Relationship Matrix (RRM).

    .. include:: ../_templates/req_biallelic.rst

    Examples
    --------

    >>> kinship_matrix = methods.rrm(dataset['GT'])

    Notes
    -----

    The Realized Relationship Matrix is defined as follows. Consider the
    :math:`n \\times m` matrix :math:`C` of raw genotypes, with rows indexed by
    :math:`n` samples and columns indexed by the :math:`m` bialellic autosomal
    variants; :math:`C_{ij}` is the number of alternate alleles of variant
    :math:`j` carried by sample :math:`i`, which can be 0, 1, 2, or missing. For
    each variant :math:`j`, the sample alternate allele frequency :math:`p_j` is
    computed as half the mean of the non-missing entries of column :math:`j`.
    Entries of :math:`M` are then mean-centered and variance-normalized as

    .. math::

        M_{ij} =
          \\frac{C_{ij}-2p_j}
                {\sqrt{\\frac{m}{n} \\sum_{k=1}^n (C_{ij}-2p_j)^2}},

    with :math:`M_{ij} = 0` for :math:`C_{ij}` missing (i.e. mean genotype
    imputation). This scaling normalizes each variant column to have empirical
    variance :math:`1/m`, which gives each sample row approximately unit total
    variance (assuming linkage equilibrium) and yields the :math:`n \\times n`
    sample correlation or realized relationship matrix (RRM) :math:`K` as simply

    .. math::

        K = MM^T

    Note that the only difference between the Realized Relationship Matrix and
    the Genetic Relationship Matrix (GRM) used in :func:`.methods.grm` is the
    variant (column) normalization: where RRM uses empirical variance, GRM uses
    expected variance under Hardy-Weinberg Equilibrium.


    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Expression on a :class:`.MatrixTable` that gives the genotype call.

    Returns
        :return: Realized Relationship Matrix for all samples.
        :rtype: :class:`.KinshipMatrix`
        """
    source = call_expr._indices.source
    if not isinstance(source, MatrixTable):
        raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
            "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))
    dataset = source
    base, _ = dataset._process_joins(call_expr)
    analyze('rrm', call_expr, dataset._entry_indices)

    dataset = require_biallelic(dataset, 'rrm')

    call_expr._indices = dataset._entry_indices
    gt_expr = call_expr.num_alt_alleles()
    dataset = dataset.annotate_rows(AC=agg.sum(gt_expr),
                                    ACsq=agg.sum(gt_expr * gt_expr),
                                    n_called=agg.count_where(functions.is_defined(call_expr)))

    dataset = dataset.filter_rows((dataset.AC > 0) &
                                  (dataset.AC < 2 * dataset.n_called) &
                                  ((dataset.AC != dataset.n_called) |
                                   (dataset.ACsq != dataset.n_called)))

    n_samples = dataset.count_cols()
    n_variants = dataset.count_rows()
    if n_variants == 0:
        raise FatalError("Cannot run RRM: found 0 variants after filtering out monomorphic sites.")
    info("Computing RRM using {} variants.".format(n_variants))

    call_expr._indices = dataset._entry_indices
    gt_expr = call_expr.num_alt_alleles()
    normalized_genotype_expr = functions.bind(
        dataset.AC / dataset.n_called,
        lambda mean_gt: functions.bind(
            functions.sqrt((dataset.ACsq +
                            (n_samples - dataset.n_called) * mean_gt ** 2) /
                           n_samples - mean_gt ** 2),
            lambda stddev: functions.cond(functions.is_defined(call_expr),
                                          (gt_expr - mean_gt) / stddev, 0)))

    bm = BlockMatrix.from_matrix_table(normalized_genotype_expr)
    dataset.unpersist()
    rrm = bm.T.dot(bm) / n_variants

    return KinshipMatrix._from_block_matrix(dataset.colkey_schema,
                                            rrm,
                                            dataset.col_keys(),
                                            n_variants)
