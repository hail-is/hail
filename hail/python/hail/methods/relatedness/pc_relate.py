from typing import Optional

import hail as hl
import hail.expr.aggregators as agg
from hail import ir
from hail.backend.spark_backend import SparkBackend
from hail.expr import (ArrayNumericExpression, BooleanExpression, CallExpression,
                       Float64Expression, analyze, expr_array, expr_call,
                       expr_float64, matrix_table_source)
from hail.expr.types import tarray
from hail.linalg import BlockMatrix
from hail.table import Table
from hail.typecheck import enumeration, nullable, numeric, typecheck
from hail.utils import new_temp_file
from hail.utils.java import Env
from ..pca import _hwe_normalized_blanczos, hwe_normalized_pca


@typecheck(call_expr=expr_call,
           min_individual_maf=numeric,
           k=nullable(int),
           scores_expr=nullable(expr_array(expr_float64)),
           min_kinship=nullable(numeric),
           statistics=enumeration('kin', 'kin2', 'kin20', 'all'),
           block_size=nullable(int),
           include_self_kinship=bool)
def pc_relate(call_expr: CallExpression,
              min_individual_maf: float,
              *,
              k: Optional[int] = None,
              scores_expr: Optional[ArrayNumericExpression] = None,
              min_kinship: Optional[float] = None,
              statistics: str = 'all',
              block_size: Optional[int] = None,
              include_self_kinship: bool = False) -> Table:
    r"""Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/req_diploid_gt.rst

    Examples
    --------
    Estimate kinship, identity-by-descent two, identity-by-descent one, and
    identity-by-descent zero for every pair of samples, using a minimum minor
    allele frequency filter of 0.01 and 10 principal components to control
    for population structure.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10) # doctest: +SKIP

    Only compute the kinship statistic. This is more efficient than
    computing all statistics.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10, statistics='kin') # doctest: +SKIP

    Compute all statistics, excluding sample-pairs with kinship less
    than 0.1. This is more efficient than producing the full table and
    then filtering using :meth:`.Table.filter`.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10, min_kinship=0.1) # doctest: +SKIP

    One can also pass in pre-computed principal component scores.
    To produce the same results as in the previous example:

    >>> _, scores_table, _ = hl.hwe_normalized_pca(dataset.GT,
    ...                                            k=10,
    ...                                            compute_loadings=False)
    >>> rel = hl.pc_relate(dataset.GT,
    ...                    0.01,
    ...                    scores_expr=scores_table[dataset.col_key].scores,
    ...                    min_kinship=0.1) # doctest: +SKIP

    Notes
    -----
    The traditional estimator for kinship between a pair of individuals
    :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
    single-nucleotide variants, from a population with estimated allele
    frequencies :math:`\widehat{p}_{s}` at SNP :math:`s`, is given by:

    .. math::

      \widehat{\psi}_{ij} \coloneqq
        \frac{1}{\left|\mathcal{S}_{ij}\right|}
        \sum_{s \in \mathcal{S}_{ij}}
          \frac{\left(g_{is} - 2\hat{p}_{s}\right)\left(g_{js} - 2\widehat{p}_{s}\right)}
                {4 \widehat{p}_{s}\left(1-\widehat{p}_{s}\right)}

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
    occurrences of population allele frequency are replaced with an
    "individual-specific allele frequency". This modification allows the
    method to correctly weight an allele according to an individual's unique
    ancestry profile.

    The "individual-specific allele frequency" at a given genetic locus is
    modeled by PC-Relate as a linear function of a sample's first ``k``
    principal component coordinates. As such, the efficacy of this method
    rests on two assumptions:

     - an individual's first ``k`` principal component coordinates fully
       describe their allele-frequency-relevant ancestry, and

     - the relationship between ancestry (as described by principal
       component coordinates) and population allele frequency is linear

    The estimators for kinship, and identity-by-descent zero, one, and two
    follow. Let:

     - :math:`S_{ij}` be the set of genetic loci at which both individuals
       :math:`i` and :math:`j` have a defined genotype

     - :math:`g_{is} \in {0, 1, 2}` be the number of alternate alleles that
       individual :math:`i` has at genetic locus :math:`s`

     - :math:`\widehat{\mu_{is}} \in [0, 1]` be the individual-specific allele
       frequency for individual :math:`i` at genetic locus :math:`s`

     - :math:`{\widehat{\sigma^2_{is}}} \coloneqq \widehat{\mu_{is}} (1 - \widehat{\mu_{is}})`,
       the binomial variance of :math:`\widehat{\mu_{is}}`

     - :math:`\widehat{\sigma_{is}} \coloneqq \sqrt{\widehat{\sigma^2_{is}}}`,
       the binomial standard deviation of :math:`\widehat{\mu_{is}}`

     - :math:`\text{IBS}^{(0)}_{ij} \coloneqq \sum_{s \in S_{ij}} \mathbb{1}_{||g_{is} - g_{js} = 2||}`,
       the number of genetic loci at which individuals :math:`i` and :math:`j`
       share no alleles

     - :math:`\widehat{f_i} \coloneqq 2 \widehat{\phi_{ii}} - 1`, the inbreeding
       coefficient for individual :math:`i`

     - :math:`g^D_{is}` be a dominance encoding of the genotype matrix, and
       :math:`X_{is}` be a normalized dominance-coded genotype matrix

    .. math::

        g^D_{is} \coloneqq
          \begin{cases}
            \widehat{\mu_{is}}     & g_{is} = 0 \\
            0                        & g_{is} = 1 \\
            1 - \widehat{\mu_{is}} & g_{is} = 2
          \end{cases}

        \qquad
        X_{is} \coloneqq g^D_{is} - \widehat{\sigma^2_{is}} (1 + \widehat{f_i})

    The estimator for kinship is given by:

    .. math::

      \widehat{\phi_{ij}} \coloneqq
        \frac{\sum_{s \in S_{ij}}(g - 2 \mu)_{is} (g - 2 \mu)_{js}}
              {4 * \sum_{s \in S_{ij}}
                            \widehat{\sigma_{is}} \widehat{\sigma_{js}}}

    The estimator for identity-by-descent two is given by:

    .. math::

      \widehat{k^{(2)}_{ij}} \coloneqq
        \frac{\sum_{s \in S_{ij}}X_{is} X_{js}}{\sum_{s \in S_{ij}}
          \widehat{\sigma^2_{is}} \widehat{\sigma^2_{js}}}

    The estimator for identity-by-descent zero is given by:

    .. math::

      \widehat{k^{(0)}_{ij}} \coloneqq
        \begin{cases}
          \frac{\text{IBS}^{(0)}_{ij}}
                {\sum_{s \in S_{ij}}
                       \widehat{\mu_{is}}^2(1 - \widehat{\mu_{js}})^2
                       + (1 - \widehat{\mu_{is}})^2\widehat{\mu_{js}}^2}
            & \widehat{\phi_{ij}} > 2^{-5/2} \\
          1 - 4 \widehat{\phi_{ij}} + k^{(2)}_{ij}
            & \widehat{\phi_{ij}} \le 2^{-5/2}
        \end{cases}

    The estimator for identity-by-descent one is given by:

    .. math::

      \widehat{k^{(1)}_{ij}} \coloneqq
        1 - \widehat{k^{(2)}_{ij}} - \widehat{k^{(0)}_{ij}}

    Note that, even if present, phase information is ignored by this method.

    The PC-Relate method is described in "Model-free Estimation of Recent
    Genetic Relatedness". Conomos MP, Reiner AP, Weir BS, Thornton TA. in
    American Journal of Human Genetics. 2016 Jan 7. The reference
    implementation is available in the `GENESIS Bioconductor package
    <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`_ .

    :func:`.pc_relate` differs from the reference implementation in a few
    ways:

     - if ``k`` is supplied, samples scores are computed via PCA on all samples,
       not a specified subset of genetically unrelated samples. The latter
       can be achieved by filtering samples, computing PCA variant loadings,
       and using these loadings to compute and pass in scores for all samples.

     - the estimators do not perform small sample correction

     - the algorithm does not provide an option to use population-wide
       allele frequency estimates

     - the algorithm does not provide an option to not use "overall
       standardization" (see R ``pcrelate`` documentation)

    Under the PC-Relate model, kinship, :math:`\phi_{ij}`, ranges from 0 to
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

    Note that :math:`g_{is}` is the number of alternate alleles. Hence, for
    multi-allelic variants, a value of 2 may indicate two distinct alternative
    alleles rather than a homozygous variant genotype. To enforce the latter,
    either filter or split multi-allelic variants first.

    The resulting table has the first 3, 4, 5, or 6 fields below, depending on
    the `statistics` parameter:

     - `i` (``col_key.dtype``) -- First sample. (key field)
     - `j` (``col_key.dtype``) -- Second sample. (key field)
     - `kin` (:py:data:`.tfloat64`) -- Kinship estimate, :math:`\widehat{\phi_{ij}}`.
     - `ibd2` (:py:data:`.tfloat64`) -- IBD2 estimate, :math:`\widehat{k^{(2)}_{ij}}`.
     - `ibd0` (:py:data:`.tfloat64`) -- IBD0 estimate, :math:`\widehat{k^{(0)}_{ij}}`.
     - `ibd1` (:py:data:`.tfloat64`) -- IBD1 estimate, :math:`\widehat{k^{(1)}_{ij}}`.

    Here ``col_key`` refers to the column key of the source matrix table,
    and ``col_key.dtype`` is a struct containing the column key fields.

    There is one row for each pair of distinct samples (columns), where `i`
    corresponds to the column of smaller column index. In particular, if the
    same column key value exists for :math:`n` columns, then the resulting
    table will have :math:`\binom{n-1}{2}` rows with both key fields equal to
    that column key value. This may result in unexpected behavior in downstream
    processing.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    min_individual_maf : :obj:`float`
        The minimum individual-specific minor allele frequency.
        If either individual-specific minor allele frequency for a pair of
        individuals is below this threshold, then the variant will not
        be used to estimate relatedness for the pair.
    k : :obj:`int`, optional
        If set, `k` principal component scores are computed and used.
        Exactly one of `k` and `scores_expr` must be specified.
    scores_expr : :class:`.ArrayNumericExpression`, optional
        Column-indexed expression of principal component scores, with the same
        source as `call_expr`. All array values must have the same positive length,
        corresponding to the number of principal components, and all scores must
        be non-missing. Exactly one of `k` and `scores_expr` must be specified.
    min_kinship : :obj:`float`, optional
        If set, pairs of samples with kinship lower than `min_kinship` are excluded
        from the results.
    statistics : :class:`str`
        Set of statistics to compute.
        If ``'kin'``, only estimate the kinship statistic.
        If ``'kin2'``, estimate the above and IBD2.
        If ``'kin20'``, estimate the above and IBD0.
        If ``'all'``, estimate the above and IBD1.
    block_size : :obj:`int`, optional
        Block size of block matrices used in the algorithm.
        Default given by :meth:`.BlockMatrix.default_block_size`.
    include_self_kinship: :obj:`bool`
        If ``True``, include entries for an individual's estimated kinship with
        themselves. Defaults to ``False``.

    Returns
    -------
    :class:`.Table`
        A :class:`.Table` mapping pairs of samples to their pair-wise statistics.
    """
    if not isinstance(Env.backend(), SparkBackend):
        return _pc_relate_bm(call_expr,
                             min_individual_maf,
                             k=k,
                             scores_expr=scores_expr,
                             min_kinship=min_kinship,
                             statistics=statistics,
                             block_size=block_size,
                             include_self_kinship=include_self_kinship)

    mt = matrix_table_source('pc_relate/call_expr', call_expr)

    if k and scores_expr is None:
        _, scores, _ = hwe_normalized_pca(call_expr, k, compute_loadings=False)
        scores_expr = scores[mt.col_key].scores
    elif not k and scores_expr is not None:
        analyze('pc_relate/scores_expr', scores_expr, mt._col_indices)
    elif k and scores_expr is not None:
        raise ValueError("pc_relate: exactly one of 'k' and 'scores_expr' must be set, found both")
    else:
        raise ValueError("pc_relate: exactly one of 'k' and 'scores_expr' must be set, found neither")

    scores_table = mt.select_cols(__scores=scores_expr) \
        .key_cols_by().select_cols('__scores').cols()

    n_missing = scores_table.aggregate(agg.count_where(hl.is_missing(scores_table.__scores)))
    if n_missing > 0:
        raise ValueError(f'Found {n_missing} columns with missing scores array.')

    mt = mt.select_entries(__gt=call_expr.n_alt_alleles()).unfilter_entries()
    mt = mt.annotate_rows(__mean_gt=agg.mean(mt.__gt))
    mean_imputed_gt = hl.or_else(hl.float64(mt.__gt), mt.__mean_gt)

    if not block_size:
        block_size = BlockMatrix.default_block_size()

    g = BlockMatrix.from_entry_expr(mean_imputed_gt,
                                    block_size=block_size)

    pcs = scores_table.collect(_localize=False).map(lambda x: x.__scores)

    ht = Table(ir.BlockMatrixToTableApply(g._bmir, pcs._ir, {
        'name': 'PCRelate',
        'maf': min_individual_maf,
        'blockSize': block_size,
        'minKinship': min_kinship,
        'statistics': {'kin': 0, 'kin2': 1, 'kin20': 2, 'all': 3}[statistics]})).persist()

    if statistics == 'kin':
        ht = ht.drop('ibd0', 'ibd1', 'ibd2')
    elif statistics == 'kin2':
        ht = ht.drop('ibd0', 'ibd1')
    elif statistics == 'kin20':
        ht = ht.drop('ibd1')

    if not include_self_kinship:
        ht = ht.filter(ht.i == ht.j, keep=False)

    col_keys = hl.literal(mt.select_cols().key_cols_by().cols().collect(), dtype=tarray(mt.col_key.dtype))
    return ht.key_by(i=col_keys[ht.i], j=col_keys[ht.j]).persist()


def _bad_mu(mu: Float64Expression, maf: float) -> BooleanExpression:
    """Check if computed value for estimated individual-specific allele
    frequency (mu) is not valid for estimating relatedness.

    Parameters
    ----------
    mu : :class:`.Float64Expression`
        Estimated individual-specific allele frequency.
    maf : :obj:`float`
        Minimum individual-specific minor allele frequency.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `mu` is not valid for relatedness estimation, else ``False``.
    """
    return (mu <= maf) | (mu >= (1.0 - maf)) | (mu <= 0.0) | (mu >= 1.0)


def _gram(M: BlockMatrix) -> BlockMatrix:
    """Compute Gram matrix, `M.T @ M`.

    Parameters
    ----------
    M : :class:`.BlockMatrix`

    Returns
    -------
    :class:`.BlockMatrix`
        `M.T @ M`
    """
    return (M.T @ M).checkpoint(new_temp_file('pc_relate_bm/gram', 'bm'))


def _dominance_encoding(g: Float64Expression, mu: Float64Expression) -> Float64Expression:
    """Compute value for a single entry in dominance encoding of genotype matrix,
    given the number of alternate alleles from the genotype matrix and the
    estimated individual-specific allele frequency.

    Parameters
    ----------
    g : :class:`.Float64Expression`
        Alternate allele count.
    mu : :class:`.Float64Expression`
        Estimated individual-specific allele frequency.

    Returns
    -------
    gd : :class:`.Float64Expression`
        Dominance-coded entry for dominance-coded genotype matrix.
    """
    gd = hl.case() \
        .when(hl.is_nan(mu), 0.0) \
        .when(g == 0.0, mu) \
        .when(g == 1.0, 0.0) \
        .when(g == 2.0, 1 - mu) \
        .or_error('entries in genotype matrix must be 0.0, 1.0, or 2.0')
    return gd


def _AtB_plus_BtA(A: BlockMatrix, B: BlockMatrix) -> BlockMatrix:
    """Compute `(A.T @ B) + (B.T @ A)`, used in estimating IBD0 (k0).

    Parameters
    ----------
    A : :class:`.BlockMatrix`
    B : :class:`.BlockMatrix`

    Returns
    -------
    :class:`.BlockMatrix`
        `(A.T @ B) + (B.T @ A)`
    """
    temp = (A.T @ B).checkpoint(new_temp_file())
    return temp + temp.T


def _replace_nan(M: BlockMatrix, value: float) -> BlockMatrix:
    """Replace NaN entries in a dense :class:`.BlockMatrix` with provided value.

    Parameters
    ----------
    M: :class:`.BlockMatrix`
    value: :obj:`float`
        Value to replace NaN entries with.

    Returns
    -------
    :class:`.BlockMatrix`
    """
    return M._map_dense(lambda x: hl.if_else(hl.is_nan(x), value, x))


@typecheck(call_expr=expr_call,
           min_individual_maf=numeric,
           k=nullable(int),
           scores_expr=nullable(expr_array(expr_float64)),
           min_kinship=nullable(numeric),
           statistics=enumeration('kin', 'kin2', 'kin20', 'all'),
           block_size=nullable(int),
           include_self_kinship=bool)
def _pc_relate_bm(call_expr: CallExpression,
                  min_individual_maf: float,
                  *,
                  k: Optional[int] = None,
                  scores_expr: Optional[ArrayNumericExpression] = None,
                  min_kinship: Optional[float] = None,
                  statistics: str = "all",
                  block_size: Optional[int] = None,
                  include_self_kinship: bool = False) -> Table:
    assert (0.0 <= min_individual_maf <= 1.0), \
        f'invalid argument: min_individual_maf={min_individual_maf}. ' \
        f'Must have min_individual_maf on interval [0.0, 1.0].'
    mt = matrix_table_source('pc_relate_bm/call_expr', call_expr)
    if k and scores_expr is None:
        eigens, scores, _ = _hwe_normalized_blanczos(call_expr, k, compute_loadings=False, q_iterations=10)
        scores_table = scores.select(__scores=scores.scores).key_by().select('__scores')
        compute_S0 = False
    elif not k and scores_expr is not None:
        analyze('pc_relate_bm/scores_expr', scores_expr, mt._col_indices)
        eigens = None
        scores_table = mt.select_cols(__scores=scores_expr).key_cols_by().select_cols('__scores').cols()
        compute_S0 = True
    elif k and scores_expr is not None:
        raise ValueError("pc_relate_bm: exactly one of 'k' and 'scores_expr' "
                         "must be set, found both")
    else:
        raise ValueError("pc_relate_bm: exactly one of 'k' and 'scores_expr' "
                         "must be set, found neither")

    n_missing = scores_table.aggregate(agg.count_where(hl.is_missing(scores_table.__scores)))
    if n_missing > 0:
        raise ValueError(f'Found {n_missing} columns with missing scores array.')
    pc_scores = hl.nd.array(scores_table.collect(_localize=False).map(lambda x: x.__scores))

    # Define NaN for missing values, otherwise cannot convert expr to block matrix
    nan = hl.float64(float('NaN'))

    # Create genotype matrix, set missing GT entries to NaN
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles()).unfilter_entries()
    gt_with_nan_expr = hl.or_else(hl.float64(mt.__gt), nan)
    if not block_size:
        block_size = BlockMatrix.default_block_size()
    g = BlockMatrix.from_entry_expr(gt_with_nan_expr, block_size=block_size)
    g = g.checkpoint(new_temp_file('pc_relate_bm/g', 'bm'))
    sqrt_n_samples = hl.nd.array([hl.sqrt(g.shape[1])])

    # Recover singular values, S0, as vector of column norms of pc_scores if necessary
    if compute_S0:
        S0 = (pc_scores ** hl.int32(2)).sum(0).map(lambda x: hl.sqrt(x))
    else:
        S0 = hl.nd.array(eigens).map(lambda x: hl.sqrt(x))
    # Set first entry of S to sqrt(n), for intercept term in beta
    S = hl.nd.hstack((sqrt_n_samples, S0))._persist()
    # Recover V from pc_scores with inv(S0)
    V0 = (pc_scores * (1 / S0))._persist()
    # Set all entries in first column of V to 1/sqrt(n), for intercept term in beta
    ones_normalized = hl.nd.full((V0.shape[0], 1), (1 / S[0]))
    V = hl.nd.hstack((ones_normalized, V0))

    # Compute matrix of regression coefficients for PCs (beta), shape (k, m)
    beta = BlockMatrix.from_ndarray(((1 / S) * V).T, block_size=block_size) @ g.T
    beta = beta.checkpoint(new_temp_file('pc_relate_bm/beta', 'bm'))

    # Compute matrix of individual-specific AF estimates (mu), shape (m, n)
    mu = 0.5 * (BlockMatrix.from_ndarray(V * S, block_size=block_size) @ beta).T
    # Replace entries in mu with NaN if invalid or if corresponding GT is missing (no contribution from that variant)
    mu = mu._apply_map2(lambda _mu, _g: hl.if_else(_bad_mu(_mu, min_individual_maf) | hl.is_nan(_g), nan, _mu),
                        g,
                        sparsity_strategy='NeedsDense')
    mu = mu.checkpoint(new_temp_file('pc_relate_bm/mu', 'bm'))

    # Compute kinship matrix (phi), shape (n, n)
    # Where mu is NaN (missing), set variance and centered AF to 0 (no contribution from that variant)
    variance = _replace_nan(mu * (1.0 - mu), 0.0).checkpoint(new_temp_file('pc_relate_bm/variance', 'bm'))
    centered_af = _replace_nan(g - (2.0 * mu), 0.0)
    phi = _gram(centered_af) / (4.0 * _gram(variance.sqrt()))
    phi = phi.checkpoint(new_temp_file('pc_relate_bm/phi', 'bm'))
    ht = phi.entries().rename({'entry': 'kin'})
    ht = ht.annotate(k0=hl.missing(hl.tfloat64),
                     k1=hl.missing(hl.tfloat64),
                     k2=hl.missing(hl.tfloat64))

    if statistics in ['kin2', 'kin20', 'all']:
        # Compute inbreeding coefficient and dominance encoding of GT matrix
        f_i = (2.0 * phi.diagonal()) - 1.0
        gd = g._apply_map2(lambda _g, _mu: _dominance_encoding(_g, _mu), mu, sparsity_strategy='NeedsDense')
        normalized_gd = gd - (variance * (1.0 + f_i))

        # Compute IBD2 (k2) estimate
        k2 = _gram(normalized_gd) / _gram(variance)
        ht = ht.annotate(k2=k2.entries()[ht.i, ht.j].entry)

        if statistics in ['kin20', 'all']:
            # Get the numerator used in IBD0 (k0) computation (IBS0), compute indicator matrices for homozygotes
            hom_alt = g._apply_map2(lambda _g, _mu: hl.if_else((_g != 2.0) | hl.is_nan(_mu), 0.0, 1.0),
                                    mu,
                                    sparsity_strategy='NeedsDense')
            hom_ref = g._apply_map2(lambda _g, _mu: hl.if_else((_g != 0.0) | hl.is_nan(_mu), 0.0, 1.0),
                                    mu,
                                    sparsity_strategy='NeedsDense')
            ibs0 = _AtB_plus_BtA(hom_alt, hom_ref)

            # Get the denominator used in IBD0 (k0) computation
            mu2 = _replace_nan(mu ** 2.0, 0.0)
            one_minus_mu2 = _replace_nan((1.0 - mu) ** 2.0, 0.0)
            k0_denom = _AtB_plus_BtA(mu2, one_minus_mu2)

            # Compute IBD0 (k0) estimates, correct the estimates where phi <= k0_cutoff
            k0 = ibs0 / k0_denom
            k0_cutoff = 2.0 ** (-5.0 / 2.0)
            ht = ht.annotate(k0=k0.entries()[ht.i, ht.j].entry)
            ht = ht.annotate(k0=hl.if_else(ht.kin <= k0_cutoff, 1.0 - (4.0 * ht.kin) + ht.k2, ht.k0))

            if statistics == 'all':
                # Finally, compute IBD1 (k1) estimate
                ht = ht.annotate(k1=1.0 - (ht.k2 + ht.k0))

    # Filter table to only have one row for each distinct pair of samples
    ht = ht.filter(ht.i <= ht.j)
    ht = ht.rename({'k0': 'ibd0', 'k1': 'ibd1', 'k2': 'ibd2'})

    if min_kinship is not None:
        ht = ht.filter(ht.kin >= min_kinship)

    if statistics != 'all':
        fields_to_drop = {
            'kin': ['ibd0', 'ibd1', 'ibd2'],
            'kin2': ['ibd0', 'ibd1'],
            'kin20': ['ibd1']}
        ht = ht.drop(*fields_to_drop[statistics])

    if not include_self_kinship:
        ht = ht.filter(ht.i == ht.j, keep=False)

    col_keys = hl.literal(mt.select_cols().key_cols_by().cols().collect(),
                          dtype=hl.tarray(mt.col_key.dtype))
    return ht.key_by(i=col_keys[hl.int32(ht.i)], j=col_keys[hl.int32(ht.j)])
