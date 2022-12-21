from typing import List, Tuple, Optional, Any

import hail as hl
import hail.expr.aggregators as agg
from hail.expr import (expr_float64, expr_call, check_entry_indexed,
                       matrix_table_source)
from hail import ir
from hail.table import Table
from hail.typecheck import typecheck, oneof, nullable, sized_tupleof
from hail._foundation.java import Env, info
from hail.errors import FatalError
from hail.experimental import mt_to_tsm, TallSkinnyMatrix


def hwe_normalize(mt: hl.MatrixTable,
                  call_expr: hl.CallExpression,
                  dimensions: Optional[Tuple[int, int]] = None
                  ):
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    mt = mt.annotate_rows(__AC=agg.sum(mt.__gt),
                          __n_called=agg.count_where(hl.is_defined(mt.__gt)))
    mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

    n_variants, _ = dimensions
    if n_variants == 0:
        raise FatalError("hwe_normalize: found 0 variants after filtering out monomorphic sites.")
    info(f"hwe_normalize: found {n_variants} variants after filtering out monomorphic sites.")

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
    mt = matrix_table_source('hwe_normalize/call_expr', call_expr)
    dimensions = mt.count()

    return pca(hwe_normalize(mt, call_expr, dimensions),
               k,
               compute_loadings,
               _dimensions=dimensions)


@typecheck(entry_expr=expr_float64,
           k=int,
           compute_loadings=bool,
           _dimensions=nullable(sized_tupleof(int, int)))
def pca(entry_expr,
        k: int = 10,
        compute_loadings: bool = False,
        *,
        _dimensions: Optional[Tuple[int, int]] = None
        ) -> Tuple[List[float], Table, Optional[Table]]:
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
        return _blanczos_pca(entry_expr, k, compute_loadings, _dimensions=_dimensions)

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


def _make_tsm_from_call(call_expr,
                        rows_per_block: Optional[int],
                        mean_center: bool = False,
                        hwe_normalize: bool = False,
                        dimensions: Optional[Tuple[int, int]] = None
                        ) -> TallSkinnyMatrix:
    mt = matrix_table_source('_make_tsm/entry_expr', call_expr)
    mt = mt.select_entries(__gt=call_expr.n_alt_alleles())
    if mean_center or hwe_normalize:
        mt = mt.annotate_rows(__AC=agg.sum(mt.__gt),
                              __n_called=agg.count_where(hl.is_defined(mt.__gt)))
        mt = mt.filter_rows((mt.__AC > 0) & (mt.__AC < 2 * mt.__n_called))

        if dimensions is None:
            dimensions = mt.count()
        n_variants = dimensions[0]
        if n_variants == 0:
            raise FatalError("_make_tsm: found 0 variants after filtering out monomorphic sites.")
        info(f"_make_tsm: found {n_variants} variants after filtering out monomorphic sites.")

        mt = mt.annotate_rows(__mean_gt=mt.__AC / mt.__n_called)
        mt = mt.unfilter_entries()

        mt = mt.select_entries(__x=hl.or_else(mt.__gt - mt.__mean_gt, 0.0))

        if hwe_normalize:
            mt = mt.annotate_rows(
                __hwe_scaled_std_dev=hl.sqrt(mt.__mean_gt * (2 - mt.__mean_gt) * n_variants / 2))
            mt = mt.select_entries(__x=mt.__x / mt.__hwe_scaled_std_dev)
    else:
        mt = mt.select_entries(__x=mt.__gt)

    return mt_to_tsm(mt.__x, rows_per_block, dimensions)


def tsm_reduced_svd(tsm: TallSkinnyMatrix, k: int):
    tsm = tsm.annotate_globals(S = tsm.S[:k])
    if 'U' in tsm.globals:
        tsm = tsm.annotate_globals(U = tsm.U @ tsm.U1[:, :k])
    if 'V' in tsm.globals:
        tsm = tsm.annotate_globals(V = tsm.V @ tsm.V1t.T[:, :k])
    return tsm


def tsm_spectral_moments(tsm: TallSkinnyMatrix,
                         num_moments: int,
                         k: int,
                         S_field: str,
                         R1_field: str,
                         V1t_field: str):
    def sqr(x):
        return x ** 2

    eigval_powers = hl.nd.vstack([
        tsm[S_field].map(lambda x: x**(2 * i))
        for i in range(1, num_moments + 1)
    ])
    xxx = (tsm[V1t_field][:, :k] @ tsm[R1_field]).map(sqr)
    moments = eigval_powers @ xxx
    means = moments.sum(1) / k
    variances = (moments - means.reshape(-1, 1)).map(sqr).sum(1) / (k - 1)

    tsm = tsm.annotate_globals(
        moments = means,
        stdevs = variances.map(hl.sqrt)
    )
    return tsm


def tsm_krylov_factorization(
        tsm: TallSkinnyMatrix,
        p: int,
        V0_field: str,
        *,
        compute_U: bool = False,
        compute_V: bool = True
) -> TallSkinnyMatrix:
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
    prev = tsm[V0_field]
    g_fields = [V0_field]

    for j in range(1, p + 1):
        tsm = tsm.annotate_globals(**{
            f'G_{j}': hl.nd.qr(
                hl.agg.ndarray_sum(tsm.block.T @ (tsm.block @ prev))
            )[0]
        })
        prev = tsm[f'G_{j}']
        g_fields.append(f'G_{j}')

    tsm = tsm.annotate_globals(
        V = hl.nd.qr(hl.nd.hstack([tsm[f] for f in g_fields]))[0]
    )
    tsm = tsm.annotate(AV = tsm.block @ tsm.V)

    if not compute_V:
        tsm = tsm.drop('V')

    if compute_U:
        tsm = tsm.annotate_globals(
            UandR = hl.nd.qr(hl.nd.vstack(hl.agg.collect(tsm.AV)))
        )
        U, R = tsm.UandR
        tsm = tsm.annotate_globals(U = U, R = R)
    else:
        Rs = hl.agg.collect(hl.nd.qr(tsm.AV)[1])
        tsm = tsm.annotate_globals(
            R = hl.nd.qr(hl.nd.vstack(Rs))[1]
        )

    U1, S, V1t = hl.nd.svd(tsm.R, full_matrices=False)
    tsm = tsm.annotate_globals(
        U1 = U1, S = S, V1t = V1t
    )
    return tsm


def _reduced_svd(tsm: TallSkinnyMatrix, k=10, compute_U=False, iterations=2, iteration_size=None):
    # Set Parameters
    q = iterations
    if iteration_size is None:
        L = k + 2
    else:
        L = iteration_size
    assert (q + 1) * L >= k

    # Generate random matrix G
    G = hl.rand_norm(0, 1, size=(tsm.n_cols, L))
    G = hl.nd.qr(G)[0]

    tsm = tsm.annotate_globals(V0 = G)
    tsm = tsm_krylov_factorization(tsm, q, 'V0', compute_U=compute_U)
    return tsm_reduced_svd(tsm, k)


@typecheck(tsm=oneof(expr_float64, TallSkinnyMatrix),
           num_moments=int,
           p=nullable(int),
           moment_samples=int,
           block_size=nullable(int))
def _spectral_moments(tsm,
                      num_moments: int,
                      p: Optional[int] = None,
                      moment_samples: int = 500,
                      block_size: Optional[int] = None
                      ) -> Tuple[Any, Any]:
    if not isinstance(tsm, TallSkinnyMatrix):
        check_entry_indexed('_spectral_moments/entry_expr', tsm)
        tsm = _make_tsm_from_call(tsm, block_size)

    n = tsm.n_cols

    if p is None:
        p = min(num_moments // 2, 10)

    # TODO: When moment_samples > n, we should just do a TSQR on A, and compute
    # the spectrum of R.
    assert moment_samples < n, '_spectral_moments: moment_samples must be smaller than num cols of A'
    G = hl.rand_unif(-1, 1, size=(n, moment_samples)).map(lambda x: hl.sign(x))
    Q1, R1 = hl.nd.qr(G)
    tsm = tsm.annotate_globals(V0 = Q1, R1 = R1)

    tsm = tsm_krylov_factorization(tsm, p, 'V0', compute_U=False)
    k = min(n, moment_samples)
    tsm = tsm_spectral_moments(tsm, num_moments, k, 'S', 'R1', 'V1t')
    return tuple(tsm.globals.select('moments', 'stdevs').collect()[0].values())


@typecheck(tsm=oneof(expr_float64, TallSkinnyMatrix),
           k=int,
           num_moments=int,
           compute_loadings=bool,
           q_iterations=int,
           oversampling_param=nullable(int),
           block_size=nullable(int),
           moment_samples=int)
def _pca_and_moments(tsm,
                     k=10,
                     num_moments=5,
                     compute_loadings=False,
                     q_iterations=10,
                     oversampling_param=None,
                     block_size: Optional[int] = None,
                     moment_samples=100):
    if not isinstance(tsm, TallSkinnyMatrix):
        check_entry_indexed('_spectral_moments/entry_expr', tsm)
        tsm = _make_tsm_from_call(tsm, block_size)

    if oversampling_param is None:
        oversampling_param = k

    # Set Parameters
    q = q_iterations
    L = k + oversampling_param
    n = tsm.n_cols

    # Generate random matrix G
    G = hl.rand_norm(0, 1, size=(n, L))
    G = hl.nd.qr(G)[0]
    tsm = tsm.annotate_globals(G = G)
    tsm = tsm_krylov_factorization(tsm, q, 'G', compute_U=compute_loadings)
    if compute_loadings:
        tsm = tsm.annotate_globals(
            krylov1_U = tsm.U,
        )
    tsm = tsm.annotate_globals(
        krylov1_S = tsm.S,
        krylov1_V = tsm.V
    )

    tsm = tsm_reduced_svd(tsm, k)
    if compute_loadings:
        tsm = tsm.annotate_globals(
            svd_U = tsm.U
        )
    tsm = tsm.annotate_globals(
        svd_S = tsm.S,
        svd_V = tsm.V
    )

    tsm = tsm.annotate_globals(
        scores = tsm.svd_V * tsm.svd_S,
        eigens = tsm.svd_S * tsm.svd_S
    )

    p = min(num_moments // 2, 10)

    # Generate random matrix G2 for moment estimation
    G2 = hl.rand_unif(-1, 1, size=(n, moment_samples)).map(lambda x: hl.sign(x))
    # Project out components in the subspace KrylovFactorization V (nb: NOT the V from SVD), which
    # we can compute exactly
    tsm = tsm.annotate_globals(
        G2 = G2 - tsm.krylov1_V @ (tsm.krylov1_V.T @ G2)
    )
    Q1, R1 = hl.nd.qr(G2)
    tsm = tsm.annotate_globals(Q1 = Q1, R1 = R1)
    tsm = tsm_krylov_factorization(tsm, p, 'Q1', compute_U=False)
    k = min(n, moment_samples)
    tsm = tsm_spectral_moments(tsm, num_moments, k, 'S', 'R1', 'V1t')
    # Add back exact moments
    tsm = tsm.annotate_globals(
        moments = tsm.moments + hl.nd.array([
            tsm.krylov1_S.map(lambda x: x**(2 * i)).sum()
            for i in range(1, num_moments + 1)
        ])
    )

    tsm = tsm.annotate_globals(
        real_cols = hl.range(tsm.n_cols).map(
            lambda i: hl.struct(
                **tsm.col_keys[i],
                scores = tsm.scores[i, :]._data_array(),
            )
        )
    )

    if compute_loadings:
        tsm = tsm.annotate(
            loading_vectors = hl.range(hl.len(tsm.row_keys)).map(
                lambda i: hl.struct(
                    row_key = tsm.row_keys[i],
                    loadings = tsm.svd_U[i, :]._data_array()
                )
            )
        )

    ht = tsm.to_table()

    if compute_loadings:
        ht = ht.explode(ht.loading_vectors)
        ht = ht.key_by(**ht.loading_vectors.row_key)
        ht = ht.select(loadings = ht.loading_vectors.loadings)
    else:
        ht = ht.head(0)

    ht = ht.annotate(
        fake_entries = hl.range(hl.len(ht.real_cols)).map(lambda _: hl.struct())
    )

    mt = ht._unlocalize_entries('fake_entries', 'real_cols', tsm.col_key())

    mt = mt.checkpoint(hl.utils.new_temp_file('_pca_and_moments', 'ht'))

    st = mt.cols().select_globals()
    eigens, moments, stdevs = hl.tuple((mt.eigens, mt.moments, mt.stdevs)).collect()[0]
    lt = None

    if compute_loadings:
        lt = mt.rows().select_globals()

    return eigens, st, lt, moments, stdevs


@typecheck(tsm=oneof(expr_float64, TallSkinnyMatrix),
           k=int,
           compute_loadings=bool,
           q_iterations=int,
           oversampling_param=nullable(int),
           rows_per_block=nullable(int),
           _dimensions=nullable(sized_tupleof(int, int)))
def _blanczos_pca(tsm,
                  k: int = 10,
                  compute_loadings: bool = False,
                  q_iterations: int = 10,
                  oversampling_param: Optional[int] = None,
                  rows_per_block: Optional[int] = None,
                  *,
                  _dimensions: Optional[Tuple[int, int]] = None):
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
    rows_per_block : :obj:`.int`
        This method internally uses a tall skinny matrix of blocks. Each block comprises at most
        rows_per_block. The last block may have fewer rows.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.

    """
    if not isinstance(tsm, TallSkinnyMatrix):
        check_entry_indexed('_blanczos_pca/entry_expr', tsm)
        tsm = mt_to_tsm(tsm, rows_per_block, dimensions=_dimensions)

    if oversampling_param is None:
        oversampling_param = k

    tsm = _reduced_svd(tsm,
                       k=k,
                       compute_U=compute_loadings,
                       iterations=q_iterations,
                       iteration_size=k + oversampling_param)
    tsm = tsm.annotate_globals(
        scores = tsm.V * tsm.S,
        eigens = tsm.S * tsm.S
    )
    tsm = tsm.annotate_globals(
        real_cols = hl.range(tsm.n_cols).map(
            lambda i: hl.struct(
                **tsm.col_keys[i],
                scores = tsm.scores[i, :]._data_array(),
            )
        )
    )
    if compute_loadings:
        tsm = tsm.annotate(
            loading_vectors = hl.range(hl.len(tsm.row_keys)).map(
                lambda i: hl.struct(
                    row_key = tsm.row_keys[i],
                    loadings = tsm.U[i, :]._data_array()
                )
            )
        )

    ht = tsm.to_table()

    if compute_loadings:
        ht = ht.explode(ht.loading_vectors)
        ht = ht.key_by(**ht.loading_vectors.row_key)
        ht = ht.select(loadings = ht.loading_vectors.loadings)
    else:
        ht = ht.head(0)

    ht = ht.annotate(
        fake_entries = hl.range(hl.len(ht.real_cols)).map(lambda _: hl.struct())
    )

    mt = ht._unlocalize_entries('fake_entries', 'real_cols', tsm.col_key())

    mt = mt.checkpoint(hl.utils.new_temp_file('_blanczos_pca', 'mt'))

    st = mt.cols().select_globals()
    eigens = mt.eigens.collect()[0]
    lt = None
    if compute_loadings:
        lt = mt.rows().select_globals()

    return eigens, st, lt


@typecheck(call_expr=expr_call,
           k=int,
           compute_loadings=bool,
           q_iterations=int,
           oversampling_param=nullable(int),
           rows_per_block=nullable(int))
def _hwe_normalized_blanczos(call_expr,
                             k: int = 10,
                             compute_loadings: int = False,
                             q_iterations: int = 10,
                             oversampling_param: Optional[int] = None,
                             rows_per_block: Optional[int] = None):
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
    rows_per_block : :obj:`.int`
        This method internally uses a tall skinny matrix of blocks. Each block comprises at most
        rows_per_block. The last block may have fewer rows.

    Returns
    -------
    (:obj:`list` of :obj:`float`, :class:`.Table`, :class:`.Table`)
        List of eigenvalues, table with column scores, table with row loadings.
    """
    check_entry_indexed('_blanczos_pca/entry_expr', call_expr)
    A = _make_tsm_from_call(call_expr, rows_per_block, hwe_normalize=True)

    return _blanczos_pca(A, k, compute_loadings=compute_loadings, q_iterations=q_iterations,
                         oversampling_param=oversampling_param, rows_per_block=rows_per_block)
