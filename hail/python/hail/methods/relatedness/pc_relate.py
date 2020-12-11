import hail as hl
import hail.expr.aggregators as agg
from hail.expr import (expr_float64, expr_call, expr_array, analyze,
                       matrix_table_source)
from hail.expr.types import tarray
from hail import ir
from hail.linalg import BlockMatrix
from hail.table import Table
from hail.typecheck import typecheck, nullable, numeric, enumeration

from ..pca import hwe_normalized_pca


@typecheck(call_expr=expr_call,
           min_individual_maf=numeric,
           k=nullable(int),
           scores_expr=nullable(expr_array(expr_float64)),
           min_kinship=nullable(numeric),
           statistics=enumeration('kin', 'kin2', 'kin20', 'all'),
           block_size=nullable(int),
           include_self_kinship=bool)
def pc_relate(call_expr, min_individual_maf, *, k=None, scores_expr=None,
              min_kinship=None, statistics="all", block_size=None,
              include_self_kinship=False) -> Table:
    r"""Compute relatedness estimates between individuals using a variant of the
    PC-Relate method.

    .. include:: ../_templates/req_diploid_gt.rst

    Examples
    --------
    Estimate kinship, identity-by-descent two, identity-by-descent one, and
    identity-by-descent zero for every pair of samples, using a minimum minor
    allele frequency filter of 0.01 and 10 principal components to control
    for population structure.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10)

    Only compute the kinship statistic. This is more efficient than
    computing all statistics.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10, statistics='kin')

    Compute all statistics, excluding sample-pairs with kinship less
    than 0.1. This is more efficient than producing the full table and
    then filtering using :meth:`.Table.filter`.

    >>> rel = hl.pc_relate(dataset.GT, 0.01, k=10, min_kinship=0.1)

    One can also pass in pre-computed principal component scores.
    To produce the same results as in the previous example:

    >>> _, scores_table, _ = hl.hwe_normalized_pca(dataset.GT,
    ...                                      k=10,
    ...                                      compute_loadings=False)
    >>> rel = hl.pc_relate(dataset.GT,
    ...                    0.01,
    ...                    scores_expr=scores_table[dataset.col_key].scores,
    ...                    min_kinship=0.1)

    Notes
    -----
    The traditional estimator for kinship between a pair of individuals
    :math:`i` and :math:`j`, sharing the set :math:`S_{ij}` of
    single-nucleotide variants, from a population with allele frequencies
    :math:`p_s`, is given by:

    .. math::

      \widehat{\phi_{ij}} \coloneqq
        \frac{1}{|S_{ij}|}
        \sum_{s \in S_{ij}}
          \frac{(g_{is} - 2 p_s) (g_{js} - 2 p_s)}
                {4 \sum_{s \in S_{ij}} p_s (1 - p_s)}

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

     - an individual's first `k` principal component coordinates fully
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
        X_{is} \coloneqq g^D_{is} - \widehat{\sigma^2_{is}} (1 - \widehat{f_i})

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

     - if `k` is supplied, samples scores are computed via PCA on all samples,
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

    scores_table = mt.select_cols(__scores=scores_expr)\
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
        'statistics': {'kin': 0, 'kin2': 1, 'kin20': 2, 'all': 3}[statistics]
    }))

    if statistics == 'kin':
        ht = ht.drop('ibd0', 'ibd1', 'ibd2')
    elif statistics == 'kin2':
        ht = ht.drop('ibd0', 'ibd1')
    elif statistics == 'kin20':
        ht = ht.drop('ibd1')

    if not include_self_kinship:
        ht = ht.filter(ht.i == ht.j, keep=False)

    col_keys = hl.literal(mt.select_cols().key_cols_by().cols().collect(), dtype=tarray(mt.col_key.dtype))
    return ht.key_by(i=col_keys[ht.i], j=col_keys[ht.j])
