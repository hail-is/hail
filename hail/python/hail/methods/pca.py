from typing import List, Tuple

import hail as hl
import hail.expr.aggregators as agg
from hail.expr.expressions import construct_expr
from hail.expr import expr_float64, expr_call, raise_unless_entry_indexed, matrix_table_source
from hail import ir
from hail.table import Table
from hail.typecheck import typecheck, oneof, nullable
from hail.utils import FatalError
from hail.utils.java import Env, info
from hail.experimental import mt_to_table_of_ndarray


def hwe_normalize(call_expr):
    mt = matrix_table_source('hwe_normalize/call_expr', call_expr)
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__AC=agg.sum(mt.__gt), __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

    n_variants = mt.count_rows()
    if n_variants == 0:
        raise FatalError("hwe_normalize: found 0 variants after filtering out monomorphic sites.")
    info(f"hwe_normalize: found {n_variants} variants after filtering out monomorphic sites.")

    mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
    mt = mt.annotate_rows(__hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) * n_variants / 2))
    mt = mt.unfilter_entries()

    normalized_gt = hl.or_else((mt.__gt - mt.__mean_gt) / mt.__hwe_scaled_std_dev, 0.0)
    return normalized_gt


@typecheck(call_expr=expr_call, k=int, compute_loadings=bool)
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
    from hail.backend.service_backend import ServiceBackend

    if isinstance(hl.current_backend(), ServiceBackend):
        return _hwe_normalized_blanczos(call_expr, k, compute_loadings)

    return pca(hwe_normalize(call_expr), k, compute_loadings)


@typecheck(entry_expr=expr_float64, k=int, compute_loadings=bool)
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
    from hail.backend.service_backend import ServiceBackend

    if isinstance(hl.current_backend(), ServiceBackend):
        return _blanczos_pca(entry_expr, k, compute_loadings)

    raise_unless_entry_indexed('pca/entry_expr', entry_expr)

    mt = matrix_table_source('pca/entry_expr', entry_expr)

    #  FIXME: remove once select_entries on a field is free
    if entry_expr in mt._fields_inverse:
        field = mt._fields_inverse[entry_expr]
    else:
        field = Env.get_uid()
        mt = mt.select_entries(**{field: entry_expr})
    mt = mt.select_cols().select_rows().select_globals()

    t = Table(
        ir.MatrixToTableApply(
            mt._mir, {'name': 'PCA', 'entryField': field, 'k': k, 'computeLoadings': compute_loadings}
        )
    ).persist()

    g = t.index_globals()
    scores = hl.Table.parallelize(g.scores, key=list(mt.col_key))
    if not compute_loadings:
        t = None
    return hl.eval(g.eigenvalues), scores, None if t is None else t.drop('eigenvalues', 'scores')


class TallSkinnyMatrix:
    def __init__(self, block_table, block_expr, source_table, col_key):
        self.col_key = col_key
        first_block = block_expr.take(1)[0]
        self.ncols = first_block.shape[1]
        self.block_table = block_table
        self.block_expr = block_expr
        self.source_table = source_table


def _make_tsm(
    entry_expr,
    block_size,
    *,
    partition_size=None,
    whiten_window_size=None,
    whiten_block_size=64,
    normalize_after_whiten=False,
):
    mt = matrix_table_source('_make_tsm/entry_expr', entry_expr)

    if whiten_window_size is None:
        A, ht = mt_to_table_of_ndarray(entry_expr, block_size, return_checkpointed_table_also=True)
        return TallSkinnyMatrix(A, A.ndarray, ht, list(mt.col_key))
    else:
        # FIXME: don't whiten across chromosome boundaries
        A, trailing_blocks_ht, ht = mt_to_table_of_ndarray(
            entry_expr,
            block_size,
            return_checkpointed_table_also=True,
            partition_size=partition_size,
            window_size=whiten_window_size,
        )

        A = A.annotate(ndarray=A.ndarray.T)
        vec_size = hl.eval(ht.take(1, _localize=False)[0].xs.length())
        if vec_size <= whiten_window_size:
            raise ValueError("whiten_window_size must be smaller than number of cols")
        joined = A.annotate(prev_window=trailing_blocks_ht[A.key].prev_window)

        def whiten_map_body(part_stream):
            stream_ir = ir.StreamWhiten(
                part_stream._ir,
                "ndarray",
                "prev_window",
                vec_size,
                whiten_window_size,
                block_size,
                whiten_block_size,
                normalize_after_whiten,
            )
            return construct_expr(stream_ir, part_stream.dtype)

        whitened = joined._map_partitions(whiten_map_body)
        whitened = whitened.annotate(ndarray=whitened.ndarray.T).persist()

        return TallSkinnyMatrix(whitened, whitened.ndarray, ht, list(mt.col_key))


def _make_tsm_from_call(
    call_expr,
    block_size,
    *,
    mean_center=False,
    hwe_normalize=False,
    partition_size=None,
    whiten_window_size=None,
    whiten_block_size=64,
    normalize_after_whiten=False,
):
    mt = matrix_table_source('_make_tsm/entry_expr', call_expr)
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    if mean_center or hwe_normalize:
        mt = mt.annotate_rows(__AC=agg.sum(mt.__gt), __n_called=agg.count_where(hl.is_defined(mt.__gt)))
        mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

        n_variants = mt.count_rows()
        if n_variants == 0:
            raise FatalError("_make_tsm: found 0 variants after filtering out monomorphic sites.")
        info(f"_make_tsm: found {n_variants} variants after filtering out monomorphic sites.")

        mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
        mt = mt.unfilter_entries()

        mt = mt.select_entries(__x=hl.or_else(mt.__gt - mt.__mean_gt, 0.0))

        if hwe_normalize:
            mt = mt.annotate_rows(__hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) / 2))
            mt = mt.select_entries(__x=mt.__x / mt.__hwe_scaled_std_dev)
    else:
        mt = mt.select_entries(__x=mt.__gt)

    return _make_tsm(
        mt.__x,
        block_size,
        partition_size=partition_size,
        whiten_window_size=whiten_window_size,
        whiten_block_size=whiten_block_size,
        normalize_after_whiten=normalize_after_whiten,
    )


class KrylovFactorization:
    # For now, all three factors are `NDArrayExpression`s, but eventually U and/or V should be
    # allowed to be distributed. All properties must be persisted.
    def __init__(self, U, R, V, k):
        self.U = U
        self.V = V
        self.k = k
        (self.U1, self.S, self.V1t) = hl.nd.svd(R, full_matrices=False)._persist()

    def reduced_svd(self, k):
        S = self.S[:k]._persist()

        if self.U is None:
            U = None
        else:
            U = (self.U @ self.U1[:, :k])._persist()

        if self.V is None:
            V = None
        else:
            V = (self.V @ self.V1t.T[:, :k])._persist()

        return U, S, V

    def spectral_moments(self, num_moments, R):
        eigval_powers = hl.nd.vstack([self.S.map(lambda x: x ** (2 * i)) for i in range(1, num_moments + 1)])
        moments = eigval_powers @ (self.V1t[:, : self.k] @ R).map(lambda x: x**2)
        means = moments.sum(1) / self.k
        variances = (moments - means.reshape(-1, 1)).map(lambda x: x**2).sum(1) / (self.k - 1)
        stdevs = variances.map(lambda x: hl.sqrt(x))
        return hl.struct(moments=means, stdevs=stdevs)


def _krylov_factorization(A: TallSkinnyMatrix, V0, p, compute_U=False, compute_V=True):
    r"""Computes matrices :math:`U`, :math:`R`, and :math:`V` satisfying the following properties:
    * :math:`U\in\mathbb{R}^{m\times (p+1)b` and :math:`V\in\mathbb{R}^{n\times (p+1)b` are
      orthonormal matrices (:math:`U^TU = I` and :math:`V^TV = I`)
    * :math:`\mathrm{span}(V) = \mathcal{K}_p(A^TA, V_0)`
    * :math:`UR=AV`, hence :math:`\mathrm{span}(U) = \mathcal{K}_p(AA^T, AV_0)`
    * :math:`V[:, :b] = V_0`
    * :math:`R\in\mathbb{R}^{b\times b}` is upper triangular
    where :math:`\mathcal{K}_p(X, Y)` is the block Krylov subspace
    :math:`\mathrm{span}(Y, XY, \dots, X^pY)`.

    Parameters
    ----------
    A_expr
    V0
    p
    compute_U

    Returns
    -------

    """
    t = A.block_table
    A_expr = A.block_expr

    g_list = [V0]
    G_i = V0
    k = hl.eval(V0.shape[1])

    for j in range(0, p):
        info(f"krylov_factorization: Beginning iteration {j+1}/{p}")
        G_i = t.aggregate(hl.agg.ndarray_sum(A_expr.T @ (A_expr @ G_i)), _localize=False)
        G_i = hl.nd.qr(G_i)[0]._persist()
        g_list.append(G_i)

    info("krylov_factorization: Iterations complete. Computing local QR")
    V0 = hl.nd.hstack(g_list)

    if compute_V:
        V = hl.nd.qr(V0)[0]._persist()
        t = t.annotate(AV=A_expr @ V)
    else:
        V = hl.nd.qr(V0)[0]
        t = t.annotate(AV=A_expr @ V)
        V = None

    if compute_U:
        temp_file_name = hl.utils.new_temp_file("_krylov_factorization_intermediate", "ht")
        t = t.checkpoint(temp_file_name)
        AV_local = t.aggregate(hl.nd.vstack(hl.agg.collect(t.AV)), _localize=False)
        U, R = hl.nd.qr(AV_local)._persist()
    else:
        Rs = t.aggregate(hl.nd.vstack(hl.agg.collect(hl.nd.qr(t.AV)[1])), _localize=False)
        R = hl.nd.qr(Rs)[1]._persist()
        U = None

    return KrylovFactorization(U, R, V, k)


def _reduced_svd(A: TallSkinnyMatrix, k=10, compute_U=False, iterations=2, iteration_size=None):
    # Set Parameters
    q = iterations
    if iteration_size is None:
        L = k + 2
    else:
        L = iteration_size
    assert (q + 1) * L >= k
    n = A.ncols

    # Generate random matrix G
    G = hl.rand_norm(0, 1, size=(n, L))
    G = hl.nd.qr(G)[0]._persist()

    fact = _krylov_factorization(A, G, q, compute_U)
    info("_reduced_svd: Computing local SVD")
    return fact.reduced_svd(k)


@typecheck(
    A=oneof(expr_float64, TallSkinnyMatrix), num_moments=int, p=nullable(int), moment_samples=int, block_size=int
)
def _spectral_moments(A, num_moments, p=None, moment_samples=500, block_size=128):
    if not isinstance(A, TallSkinnyMatrix):
        raise_unless_entry_indexed('_spectral_moments/entry_expr', A)
        A = _make_tsm(A, block_size)

    n = A.ncols

    if p is None:
        p = min(num_moments // 2, 10)

    # TODO: When moment_samples > n, we should just do a TSQR on A, and compute
    # the spectrum of R.
    assert moment_samples < n, '_spectral_moments: moment_samples must be smaller than num cols of A'
    G = hl.rand_unif(-1, 1, size=(n, moment_samples)).map(lambda x: hl.sign(x))
    Q1, R1 = hl.nd.qr(G)._persist()
    fact = _krylov_factorization(A, Q1, p, compute_U=False)
    moments_and_stdevs = hl.eval(fact.spectral_moments(num_moments, R1))
    moments = moments_and_stdevs.moments
    stdevs = moments_and_stdevs.stdevs
    return moments, stdevs


@typecheck(
    A=oneof(expr_float64, TallSkinnyMatrix),
    k=int,
    num_moments=int,
    compute_loadings=bool,
    q_iterations=int,
    oversampling_param=nullable(int),
    block_size=int,
    moment_samples=int,
)
def _pca_and_moments(
    A,
    k=10,
    num_moments=5,
    compute_loadings=False,
    q_iterations=10,
    oversampling_param=None,
    block_size=128,
    moment_samples=100,
):
    if not isinstance(A, TallSkinnyMatrix):
        raise_unless_entry_indexed('_spectral_moments/entry_expr', A)
        A = _make_tsm(A, block_size)

    if oversampling_param is None:
        oversampling_param = k

    # Set Parameters
    q = q_iterations
    L = k + oversampling_param
    n = A.ncols

    # Generate random matrix G
    G = hl.rand_norm(0, 1, size=(n, L))
    G = hl.nd.qr(G)[0]._persist()

    fact = _krylov_factorization(A, G, q, compute_loadings)
    info("_reduced_svd: Computing local SVD")
    U, S, V = fact.reduced_svd(k)

    p = min(num_moments // 2, 10)

    # Generate random matrix G2 for moment estimation
    G2 = hl.rand_unif(-1, 1, size=(n, moment_samples)).map(lambda x: hl.sign(x))
    # Project out components in subspace fact.V, which we can compute exactly
    G2 = G2 - fact.V @ (fact.V.T @ G2)
    Q1, R1 = hl.nd.qr(G2)._persist()
    fact2 = _krylov_factorization(A, Q1, p, compute_U=False)
    moments_and_stdevs = fact2.spectral_moments(num_moments, R1)
    # Add back exact moments
    moments = moments_and_stdevs.moments + hl.nd.array([
        fact.S.map(lambda x: x ** (2 * i)).sum() for i in range(1, num_moments + 1)
    ])
    moments_and_stdevs = hl.eval(hl.struct(moments=moments, stdevs=moments_and_stdevs.stdevs))
    moments = moments_and_stdevs.moments
    stdevs = moments_and_stdevs.stdevs

    scores = V * S
    eigens = hl.eval(S * S)
    info("blanczos_pca: SVD Complete. Computing conversion to PCs.")

    hail_array_scores = scores._data_array()
    cols_and_scores = hl.zip(A.source_table.index_globals().cols, hail_array_scores).map(
        lambda tup: tup[0].annotate(scores=tup[1])
    )
    st = hl.Table.parallelize(cols_and_scores, key=A.col_key)

    if compute_loadings:
        lt = A.source_table.select()
        lt = lt.annotate_globals(U=U)
        idx_name = '_tmp_pca_loading_index'
        lt = lt.add_index(idx_name)
        lt = lt.annotate(loadings=hl.array(lt.U[lt[idx_name], :])).select_globals()
        lt = lt.drop(lt[idx_name])
    else:
        lt = None

    return eigens, st, lt, moments, stdevs


@typecheck(
    A=oneof(expr_float64, TallSkinnyMatrix),
    k=int,
    compute_loadings=bool,
    q_iterations=int,
    oversampling_param=nullable(int),
    block_size=int,
    compute_scores=bool,
    transpose=bool,
)
def _blanczos_pca(
    A,
    k=10,
    compute_loadings=False,
    q_iterations=10,
    oversampling_param=None,
    block_size=128,
    compute_scores=True,
    transpose=False,
):
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
    if not isinstance(A, TallSkinnyMatrix):
        raise_unless_entry_indexed('_blanczos_pca/entry_expr', A)
        A = _make_tsm(A, block_size)

    if oversampling_param is None:
        oversampling_param = k

    compute_U = (not transpose and compute_loadings) or (transpose and compute_scores)
    U, S, V = _reduced_svd(A, k, compute_U, q_iterations, k + oversampling_param)
    info("blanczos_pca: SVD Complete. Computing conversion to PCs.")

    def numpy_to_rows_table(X, field_name):
        t = A.source_table.select()
        t = t.annotate_globals(X=X)
        idx_name = '_tmp_pca_loading_index'
        t = t.add_index(idx_name)
        t = t.annotate(**{field_name: hl.array(t.X[t[idx_name], :])}).select_globals()
        t = t.drop(t[idx_name])
        return t

    def numpy_to_cols_table(X, field_name):
        hail_array = X._data_array()
        cols_and_X = hl.zip(A.source_table.index_globals().cols, hail_array).map(
            lambda tup: tup[0].annotate(**{field_name: tup[1]})
        )
        t = hl.Table.parallelize(cols_and_X, key=A.col_key)
        return t

    st = None
    lt = None
    eigens = hl.eval(S * S)
    if transpose:
        if compute_loadings:
            lt = numpy_to_cols_table(V, 'loadings')
        if compute_scores:
            st = numpy_to_rows_table(U * S, 'scores')
    else:
        if compute_scores:
            st = numpy_to_cols_table(V * S, 'scores')
        if compute_loadings:
            lt = numpy_to_rows_table(U, 'loadings')

    return eigens, st, lt


@typecheck(
    call_expr=expr_call,
    k=int,
    compute_loadings=bool,
    q_iterations=int,
    oversampling_param=nullable(int),
    block_size=int,
)
def _hwe_normalized_blanczos(
    call_expr, k=10, compute_loadings=False, q_iterations=10, oversampling_param=None, block_size=128
):
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
    raise_unless_entry_indexed('_blanczos_pca/entry_expr', call_expr)
    A = _make_tsm_from_call(call_expr, block_size, hwe_normalize=True)

    return _blanczos_pca(
        A,
        k,
        compute_loadings=compute_loadings,
        q_iterations=q_iterations,
        oversampling_param=oversampling_param,
        block_size=block_size,
    )
