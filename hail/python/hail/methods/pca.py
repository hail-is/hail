from typing import List, Tuple

import hail as hl
import hail.expr.aggregators as agg
from hail.expr import (expr_float64, expr_call, check_entry_indexed,
                       matrix_table_source)
from hail import ir
from hail.table import Table
from hail.typecheck import typecheck
from hail.utils import FatalError
from hail.utils.java import Env, info
from hail.experimental import mt_to_table_of_ndarray


def hwe_normalize(call_expr):
    mt = matrix_table_source('hwe_normalized_pca/call_expr', call_expr)
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__AC=agg.sum(mt.__gt),
                          __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

    n_variants = mt.count_rows()
    if n_variants == 0:
        raise FatalError("hwe_normalized_pca: found 0 variants after filtering out monomorphic sites.")
    info("hwe_normalized_pca: running PCA using {} variants.".format(n_variants))

    mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(
        __hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) * n_variants / 2))
    mt = mt.unfilter_entries()

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__hwe_scaled_std_dev, 0.0)
    return normalized_gt


@typecheck(call_expr=expr_call,
           k=int,
           compute_loadings=bool)
def hwe_normalized_pca(call_expr, k=10, compute_loadings=False) -> Tuple[List[float], Table, Table]:
    r"""Run principal component analysis (PCA) on the Hardy-Weinberg-normalized
    genotype call matrix.

    Examples
    --------

    >>> eigenvalues, scores, loadings = hl.hwe_normalized_pca(dataset.GT, k=5)

    Notes
    -----
    This method specializes :func:`.pca` for the common use case
    of PCA in statistical genetics, that of projecting samples to a small
    number of ancestry coordinates. Variants that are all homozygous reference
    or all homozygous alternate are unnormalizable and removed before
    evaluation. See :func:`.pca` for more details.

    Users of PLINK/GCTA should be aware that Hail computes the GRM slightly
    differently with regard to missing data. In Hail, the
    :math:`ij` entry of the GRM :math:`MM^T` is simply the dot product of rows
    :math:`i` and :math:`j` of :math:`M`; in terms of :math:`C` it is

    .. math::

      \frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}

    where :math:`\mathcal{C}_i = \{l \mid C_{il} \text{ is non-missing}\}`. In
    PLINK/GCTA the denominator :math:`m` is replaced with the number of terms in
    the sum :math:`\lvert\mathcal{C}_i\cap\mathcal{C}_j\rvert`, i.e. the
    number of variants where both samples have non-missing genotypes. While this
    is arguably a better estimator of the true GRM (trading shrinkage for
    noise), it has the drawback that one loses the clean interpretation of the
    loadings and scores as features and projections

    Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM, i.e.
    the left singular vectors :math:`U_k` instead of the component scores
    :math:`U_k S_k`. The scores have the advantage of representing true
    projections of the data onto features with the variance of a score
    reflecting the variance explained by the corresponding feature. In PC
    bi-plots this amounts to a change in aspect ratio; for use of PCs as
    covariates in regression it is immaterial.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """

    return pca(hwe_normalize(call_expr),
               k,
               compute_loadings)


@typecheck(entry_expr=expr_float64,
           k=int,
           compute_loadings=bool)
def pca(entry_expr, k=10, compute_loadings=False) -> Tuple[List[float], Table, Table]:
    r"""Run principal component analysis (PCA) on numeric columns derived from a
    matrix table.

    Examples
    --------

    For a matrix table with variant rows, sample columns, and genotype entries,
    compute the top 2 PC sample scores and eigenvalues of the matrix of 0s and
    1s encoding missingness of genotype calls.

    >>> eigenvalues, scores, _ = hl.pca(hl.int(hl.is_defined(dataset.GT)),
    ...                                 k=2)

    Warning
    -------
      This method does **not** automatically mean-center or normalize each column.
      If desired, such transformations should be incorporated in `entry_expr`.

      Hail will return an error if `entry_expr` evaluates to missing, nan, or
      infinity on any entry.

    Notes
    -----

    PCA is run on the columns of the numeric matrix obtained by evaluating
    `entry_expr` on each entry of the matrix table, or equivalently on the rows
    of the **transposed** numeric matrix :math:`M` referenced below.

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
    :math:`n \times k`, :math:`k \times k` and :math:`m \times k`
    respectively.

    From the perspective of the rows of :math:`M` as samples (data points),
    :math:`V_k` contains the loadings for the first :math:`k` PCs while
    :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each
    sample. The loadings represent a new basis of features while the scores
    represent the projected data on those features. The eigenvalues of the Gramian
    :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2,
    \ldots`, which represent the variances carried by the respective PCs. By
    default, Hail only computes the loadings if the ``loadings`` parameter is
    specified.

    Scores are stored in a :class:`.Table` with the column key of the matrix
    table as key and a field `scores` of type ``array<float64>`` containing
    the principal component scores.

    Loadings are stored in a :class:`.Table` with the row key of the matrix
    table as key and a field `loadings` of type ``array<float64>`` containing
    the principal component loadings.

    The eigenvalues are returned in descending order, with scores and loadings
    given the corresponding array order.

    Parameters
    ----------
    entry_expr : :class:`.Expression`
        Numeric expression for matrix entries.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    check_entry_indexed('pca/entry_expr', entry_expr)

    mt = matrix_table_source('pca/entry_expr', entry_expr)

    #  FIXME: remove once select_entries on a field is free
    if entry_expr in mt._fields_inverse:
        field = mt._fields_inverse[entry_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_cols().select_rows().select_globals()

    t = (Table(ir.MatrixToTableApply(mt._mir, {
        'name': 'PCA',
        'entryField': field,
        'k': k,
        'computeLoadings': compute_loadings
    })).persist())

    g = t.index_globals()
    scores = hl.Table.parallelize(g.scores, key=list(mt.col_key))
    if not compute_loadings:
        t = None
    return hl.eval(g.eigenvalues), scores, None if t is None else t.drop('eigenvalues', 'scores')


@typecheck(entry_expr=expr_float64,
           k=int,
           compute_loadings=bool,
           q_iterations=int,
           oversampling_param=int,
           block_size=int)
def _blanczos_pca(entry_expr, k=10, compute_loadings=False, q_iterations=2, oversampling_param=2, block_size=128):
    r"""Run randomized principal component analysis approximation (PCA)
    on numeric columns derived from a matrix table.

    Implements the Blanczos algorithm found by Rokhlin, Szlam, and Tygert.

    Examples
    --------

    For a matrix table with variant rows, sample columns, and genotype entries,
    compute the top 2 PC sample scores and eigenvalues of the matrix of 0s and
    1s encoding missingness of genotype calls.

    >>> eigenvalues, scores, _ = hl._blanczos_pca(hl.int(hl.is_defined(dataset.GT)),
    ...                                 k=2)

    Warning
    -------
      This method does **not** automatically mean-center or normalize each column.
      If desired, such transformations should be incorporated in `entry_expr`.

      Hail will return an error if `entry_expr` evaluates to missing, nan, or
      infinity on any entry.

    Notes
    -----

    PCA is run on the columns of the numeric matrix obtained by evaluating
    `entry_expr` on each entry of the matrix table, or equivalently on the rows
    of the **transposed** numeric matrix :math:`M` referenced below.

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
    :math:`n \times k`, :math:`k \times k` and :math:`m \times k`
    respectively.

    From the perspective of the rows of :math:`M` as samples (data points),
    :math:`V_k` contains the loadings for the first :math:`k` PCs while
    :math:`MV_k = U_k S_k` contains the first :math:`k` PC scores of each
    sample. The loadings represent a new basis of features while the scores
    represent the projected data on those features. The eigenvalues of the Gramian
    :math:`MM^T` are the squares of the singular values :math:`s_1^2, s_2^2,
    \ldots`, which represent the variances carried by the respective PCs. By
    default, Hail only computes the loadings if the ``loadings`` parameter is
    specified.

    Scores are stored in a :class:`.Table` with the column key of the matrix
    table as key and a field `scores` of type ``array<float64>`` containing
    the principal component scores.

    Loadings are stored in a :class:`.Table` with the row key of the matrix
    table as key and a field `loadings` of type ``array<float64>`` containing
    the principal component loadings.

    The eigenvalues are returned in descending order, with scores and loadings
    given the corresponding array order.

    Parameters
    ----------
    entry_expr : :class:`.Expression`
        Numeric expression for matrix entries.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.
    q_iterations : :obj:`int`
        Number of rounds of power iteration to amplify singular values.
    oversampling_param : :obj:`int`
        Amount of oversampling to use when approximating the singular values.
        Usually a value between `0 <= oversampling_param <= k`.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    check_entry_indexed('mt_to_table_of_ndarray/entry_expr', entry_expr)
    mt = matrix_table_source('pca/entry_expr', entry_expr)

    A, ht = mt_to_table_of_ndarray(entry_expr, block_size, return_checkpointed_table_also=True)
    A = A.persist()

    # Set Parameters

    q = q_iterations
    L = k + oversampling_param
    n = A.take(1)[0].ndarray.shape[1]

    # Generate random matrix G
    G = hl.nd.zeros((n, L)).map(lambda n: hl.rand_norm(0, 1))

    def hailBlanczos(A, G, k, q, compute_U=False):

        G_i = hl.nd.qr(G)[0]
        g_list = [G_i]

        for j in range(0, q):
            info(f"blanczos_pca: Beginning iteration {j+1}/{q}")
            G_i = A.aggregate(hl.agg.ndarray_sum(A.ndarray.T @ (A.ndarray @ G_i)), _localize=False)
            G_i = hl.nd.qr(G_i)[0]._persist()
            g_list.append(G_i)

        info("blanczos_pca: Iterations complete. Computing local QR")
        G = hl.nd.hstack(g_list)
        V = hl.nd.qr(G)[0]._persist()

        AV = A.select(ndarray=A.ndarray @ V)

        if compute_U:
            AV_local = hl.nd.vstack(AV.aggregate(hl.agg.collect(AV.ndarray), _localize=False))
            U, R = hl.nd.qr(AV_local)._persist()
            return U, R, V
        else:
            Rs = hl.nd.vstack(AV.aggregate(hl.agg.collect(hl.nd.qr(AV.ndarray)[1]), _localize=False))
            R = hl.nd.qr(Rs)[1]._persist()
            return R, V

    if compute_loadings:
        U0, R, V0 = hailBlanczos(A, G, k, q, compute_U=True)
    else:
        R, V0 = hailBlanczos(A, G, k, q, compute_U=False)

    info("blanczos_pca: QR Complete. Computing local SVD")
    U1, S, V1t = hl.nd.svd(R, full_matrices=False)._persist()

    S = S[:k]
    V = V0 @ V1t.T[:, :k]

    scores = V * S
    eigens = hl.eval(S * S)
    info("blanczos_pca: SVD Complete. Computing conversion to PCs.")

    hail_array_scores = scores._data_array()
    cols_and_scores = hl.zip(A.index_globals().cols, hail_array_scores).map(lambda tup: tup[0].annotate(scores=tup[1]))
    st = hl.Table.parallelize(cols_and_scores, key=list(mt.col_key))

    if compute_loadings:
        U = U0 @ U1[:, :k]
        lt = ht.select()
        lt = lt.annotate_globals(U=U)
        idx_name = '_tmp_pca_loading_index'
        lt = lt.add_index(idx_name)
        lt = lt.annotate(loadings=lt.U[lt[idx_name], :]._data_array()).select_globals()
        lt = lt.drop(lt[idx_name])
        return eigens, st, lt
    else:
        return eigens, st, None


@typecheck(call_expr=expr_call,
           k=int,
           compute_loadings=bool,
           q_iterations=int,
           oversampling_param=int,
           block_size=int)
def _hwe_normalized_blanczos(call_expr, k=10, compute_loadings=False, q_iterations=2, oversampling_param=2, block_size=128):
    r"""Run randomized principal component analysis approximation (PCA) on the
    Hardy-Weinberg-normalized genotype call matrix.

    Implements the Blanczos algorithm found by Rokhlin, Szlam, and Tygert.

    Examples
    --------

    >>> eigenvalues, scores, loadings = hl._hwe_normalized_blanczos(dataset.GT, k=5)

    Notes
    -----
    This method specializes :func:`._blanczos_pca` for the common use case
    of PCA in statistical genetics, that of projecting samples to a small
    number of ancestry coordinates. Variants that are all homozygous reference
    or all homozygous alternate are unnormalizable and removed before
    evaluation. See :func:`._blanczos_pca` for more details.

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression.
    k : :obj:`int`
        Number of principal components.
    compute_loadings : :obj:`bool`
        If ``True``, compute row loadings.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """

    return _blanczos_pca(hwe_normalize(call_expr), k, compute_loadings=compute_loadings, q_iterations=q_iterations,
                         oversampling_param=oversampling_param, block_size=block_size)
