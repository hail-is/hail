import math
import pytest
import numpy as np

import hail as hl
from hail.methods.pca import _make_tsm
from ..helpers import resource, fails_local_backend, skip_when_service_backend, test_timeout


@fails_local_backend()
@test_timeout(batch=10 * 60)
def test_hwe_normalized_pca():
    mt = hl.balding_nichols_model(3, 100, 50)
    eigenvalues, scores, loadings = hl.hwe_normalized_pca(mt.GT, k=2, compute_loadings=True)

    assert len(eigenvalues) == 2
    assert isinstance(scores, hl.Table)
    assert scores.count() == 100
    assert isinstance(loadings, hl.Table)

    _, _, loadings = hl.hwe_normalized_pca(mt.GT, k=2, compute_loadings=False)
    assert loadings is None


@fails_local_backend()
@test_timeout(batch=10 * 60)
def test_pca_against_numpy():
    mt = hl.import_vcf(resource('tiny_m.vcf'))
    mt = mt.annotate_rows(AC=hl.agg.sum(mt.GT.n_alt_alleles()),
                          n_called=hl.agg.count_where(hl.is_defined(mt.GT)))
    n_rows = 3
    n_cols = 4
    k = 3

    mean = mt.AC / mt.n_called
    eigen, scores, loadings = hl.pca(
        hl.coalesce(
            (mt.GT.n_alt_alleles() - mean) / hl.sqrt(mean * (2 - mean) * n_rows / 2),
            0
        ),
        k=k,
        compute_loadings=True
    )

    hail_scores = scores.explode('scores').scores.collect()
    hail_loadings = loadings.explode('loadings').loadings.collect()

    assert len(hail_scores) == n_cols * k
    assert len(hail_loadings) == n_rows * k

    # compute PCA with numpy
    def normalize(a):
        ms = np.mean(a, axis=0, keepdims=True)
        return np.divide(np.subtract(a, ms), np.sqrt(2.0 * np.multiply(ms / 2.0, 1 - ms / 2.0) * a.shape[1]))

    g = np.pad(np.diag([1.0, 1, 2]), ((0, 1), (0, 0)), mode='constant')
    g[1, 0] = 1.0 / 3
    n = normalize(g)
    U, s, V = np.linalg.svd(n, full_matrices=0)
    np_scores = U.dot(np.diag(s)).flatten()
    np_loadings = V.transpose().flatten()
    np_eigenvalues = np.multiply(s, s).flatten()

    np.testing.assert_allclose(eigen, np_eigenvalues, rtol=1e-5)
    np.testing.assert_allclose(np.abs(hail_scores), np.abs(np_scores), rtol=1e-5)
    np.testing.assert_allclose(np.abs(hail_loadings), np.abs(np_loadings), rtol=1e-5)


@test_timeout(batch=10 * 60)
def test_blanczos_against_numpy():
    def concatToNumpy(blocks, horizontal=True):
        if horizontal:
            return np.concatenate(blocks, axis=0)
        else:
            return np.concatenate(blocks, axis=1)

    mt = hl.import_vcf(resource('tiny_m.vcf'))
    mt = mt.annotate_rows(AC=hl.agg.sum(mt.GT.n_alt_alleles()),
                          n_called=hl.agg.count_where(hl.is_defined(mt.GT)))
    n_rows = 3
    n_cols = 4
    k = 3

    mean = mt.AC / mt.n_called
    float_expr = hl.coalesce(
        (mt.GT.n_alt_alleles() - mean) / hl.sqrt(mean * (2 - mean) * n_rows / 2),
        0
    )

    eigens, scores_t, loadings_t = hl._blanczos_pca(float_expr, k=k, q_iterations=7, compute_loadings=True)
    A = np.array(float_expr.collect()).reshape((3, 4)).T
    scores_blocks = scores_t.scores.collect()
    scores = concatToNumpy(scores_blocks)
    scores = np.reshape(scores, (len(scores) // k, k))

    loadings_blocks = loadings_t.loadings.collect()
    loadings = concatToNumpy(loadings_blocks)
    loadings = np.reshape(loadings, (len(loadings) // k, k))

    assert len(eigens) == 3
    assert len(scores) == n_cols
    assert len(loadings) == n_rows
    np.testing.assert_almost_equal(A @ loadings, scores)

    assert len(scores_t.globals) == 0
    assert len(loadings_t.globals) == 0

    # compute PCA with numpy
    def normalize(a):
        ms = np.mean(a, axis=0, keepdims=True)
        return np.divide(np.subtract(a, ms), np.sqrt(2.0 * np.multiply(ms / 2.0, 1 - ms / 2.0) * a.shape[1]))

    g = np.pad(np.diag([1.0, 1, 2]), ((0, 1), (0, 0)), mode='constant')
    g[1, 0] = 1.0 / 3
    n = normalize(g)
    U, s, V = np.linalg.svd(n, full_matrices=0)
    np_loadings = V.transpose()
    np_eigenvalues = np.multiply(s, s)

    def bound(vs, us):  # equation 12 from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4827102/pdf/main.pdf
        return 1/k * sum([np.linalg.norm(us.T @ vs[:,i]) for i in range(k)])

    np.testing.assert_allclose(eigens, np_eigenvalues, rtol=0.05)
    assert bound(np_loadings, loadings) > 0.9


def matrix_table_from_numpy(np_mat):
    rows, cols = np_mat.shape
    mt = hl.utils.range_matrix_table(rows, cols)
    mt = mt.annotate_globals(entries_global = np_mat)
    mt = mt.annotate_entries(ent = mt.entries_global[mt.row_idx, mt.col_idx])
    return mt


# k, m, n
dim_triplets = [(20, 1000, 1000), (10, 100, 200)]


@test_timeout(batch=5 * 60)
def test_blanczos_T():
    k, m, n = 10, 100, 200
    sigma = np.diag([spec1(i + 1, k) for i in range(m)])
    seed = 1025
    np.random.seed(seed)
    U = np.linalg.qr(np.random.normal(0, 1, (m, m)))[0]
    V = np.linalg.qr(np.random.normal(0, 1, (n, m)))[0]
    A = U @ sigma @ V.T
    mt_A_T = matrix_table_from_numpy(A.T)

    eigenvalues, scores, loadings = hl._blanczos_pca(mt_A_T.ent, k=k, oversampling_param=k, q_iterations=4, compute_loadings=True, transpose=True)
    singulars = np.sqrt(eigenvalues)
    hail_V = (np.array(scores.scores.collect()) / singulars).T
    hail_U = np.array(loadings.loadings.collect())
    approx_A = hail_U @ np.diag(singulars) @ hail_V
    norm_of_diff = np.linalg.norm(A - approx_A, 2)
    np.testing.assert_allclose(norm_of_diff, spec1(k + 1, k), rtol=1e-02)
    np.testing.assert_allclose(singulars, np.diag(sigma)[:k], rtol=1e-01)

def spectra_helper(spec_func, triplet):
    k, m, n = triplet
    min_dim = min(m, n)
    sigma = np.diag([spec_func(i + 1, k) for i in range(min_dim)])
    seed = 1025
    np.random.seed(seed)
    U = np.linalg.qr(np.random.normal(0, 1, (m, min_dim)))[0]
    V = np.linalg.qr(np.random.normal(0, 1, (n, min_dim)))[0]
    A = U @ sigma @ V.T
    mt_A = matrix_table_from_numpy(A)

    eigenvalues, scores, loadings = hl._blanczos_pca(mt_A.ent, k=k, oversampling_param=k, compute_loadings=True, q_iterations=4)
    singulars = np.sqrt(eigenvalues)
    hail_V = (np.array(scores.scores.collect()) / singulars).T
    hail_U = np.array(loadings.loadings.collect())
    approx_A = hail_U @ np.diag(singulars) @ hail_V
    norm_of_diff = np.linalg.norm(A - approx_A, 2)
    np.testing.assert_allclose(norm_of_diff, spec_func(k + 1, k), rtol=1e-02, err_msg=f"Norm test failed on triplet {triplet} ")
    np.testing.assert_allclose(singulars, np.diag(sigma)[:k], rtol=1e-01, err_msg=f"Failed on triplet {triplet}")


def spec1(j, k):
    return 1/j


def spec2(j, k):
    if j == 1:
        return 1
    if j <= k:
        return 2 * 10**-5
    else:
        return (10**-5) * (k + 1)/j


def spec3(j, k):
    if j <= k:
        return 10**(-5*(j-1)/(k-1))
    else:
        return (10**-5)*(k+1)/j


def spec4(j, k):
    if j <= k:
        return 10**(-5*(j-1)/(k-1))
    elif j == (k + 1):
        return 10**-5
    else:
        return 0


def spec5(j, k):
    if j <= k:
        return 10**-5 + (1 - 10**-5)*(k - j)/(k - 1)
    else:
        return 10**-5 * math.sqrt((k + 1)/j)


@pytest.mark.parametrize("triplet", dim_triplets)
@test_timeout(5 * 60, batch=8 * 60)
def test_spectra_1(triplet):
    spectra_helper(spec1, triplet)


@pytest.mark.parametrize("triplet", dim_triplets)
@test_timeout(5 * 60, batch=8 * 60)
def test_spectra_2(triplet):
    spectra_helper(spec2, triplet)


@pytest.mark.parametrize("triplet", dim_triplets)
@test_timeout(5 * 60, batch=8 * 60)
def test_spectra_3(triplet):
    spectra_helper(spec3, triplet)


@pytest.mark.parametrize("triplet", dim_triplets)
@test_timeout(5 * 60, batch=8 * 60)
def test_spectra_4(triplet):
    spectra_helper(spec4, triplet)


@pytest.mark.parametrize("triplet", dim_triplets)
@test_timeout(5 * 60, batch=8 * 60)
def test_spectra_5(triplet):
    spectra_helper(spec5, triplet)


def spectral_moments_helper(spec_func):
    for triplet in [(20, 1000, 1000)]:
        k, m, n = triplet
        min_dim = min(m, n)
        sigma = np.diag([spec_func(i+1, k) for i in range(min_dim)])
        seed = 1025
        np.random.seed(seed)
        U = np.linalg.qr(np.random.normal(0, 1, (m, min_dim)))[0]
        V = np.linalg.qr(np.random.normal(0, 1, (n, min_dim)))[0]
        A = U @ sigma @ V.T
        mt_A = matrix_table_from_numpy(A)

        moments, stdevs = hl._spectral_moments(_make_tsm(mt_A.ent, 128), 7)
        true_moments = np.array([np.sum(np.power(sigma, 2*i)) for i in range(1, 8)])
        np.testing.assert_allclose(moments, true_moments, rtol=2e-01)


@skip_when_service_backend(reason='v slow & OOms')
@test_timeout(local=3 * 60, batch=8 * 60)
def test_spectral_moments_1():
    spectral_moments_helper(spec1)


@skip_when_service_backend(reason='v slow & OOms')
@test_timeout(local=3 * 60, batch=8 * 60)
def test_spectral_moments_2():
    spectral_moments_helper(spec2)


@skip_when_service_backend(reason='v slow & OOms')
@test_timeout(local=3 * 60, batch=8 * 60)
def test_spectral_moments_3():
    spectral_moments_helper(spec3)


@skip_when_service_backend(reason='v slow & OOms')
@test_timeout(local=3 * 60, batch=8 * 60)
def test_spectral_moments_4():
    spectral_moments_helper(spec4)


@skip_when_service_backend(reason='v slow & OOms')
@test_timeout(local=3 * 60, batch=8 * 60)
def test_spectral_moments_5():
    spectral_moments_helper(spec5)


def spectra_and_moments_helper(spec_func):
    for triplet in [(20, 1000, 1000)]:
        k, m, n = triplet
        min_dim = min(m, n)
        sigma = np.diag([spec_func(i+1, k) for i in range(min_dim)])
        seed = 1025
        np.random.seed(seed)
        U = np.linalg.qr(np.random.normal(0, 1, (m, min_dim)))[0]
        V = np.linalg.qr(np.random.normal(0, 1, (n, min_dim)))[0]
        A = U @ sigma @ V.T
        mt_A = matrix_table_from_numpy(A)

        eigenvalues, scores, loadings, moments, stdevs = hl._pca_and_moments(_make_tsm(mt_A.ent, 128), k=k, num_moments=7, oversampling_param=k, compute_loadings=True, q_iterations=4)
        singulars = np.sqrt(eigenvalues)
        hail_V = (np.array(scores.scores.collect()) / singulars).T
        hail_U = np.array(loadings.loadings.collect())
        approx_A = hail_U @ np.diag(singulars) @ hail_V
        norm_of_diff = np.linalg.norm(A - approx_A, 2)
        np.testing.assert_allclose(norm_of_diff, spec_func(k + 1, k), rtol=1e-02, err_msg=f"Norm test failed on triplet {triplet}")
        np.testing.assert_allclose(singulars, np.diag(sigma)[:k], rtol=1e-01, err_msg=f"Failed on triplet {triplet}")

        true_moments = np.array([np.sum(np.power(sigma, 2*i)) for i in range(1, 8)])
        np.testing.assert_allclose(moments, true_moments, rtol=1e-04)


@test_timeout(4 * 60, batch=8 * 60)
def test_spectra_and_moments_1():
    spectra_and_moments_helper(spec1)


@test_timeout(4 * 60, batch=8 * 60)
def test_spectra_and_moments_2():
    spectra_and_moments_helper(spec2)


@test_timeout(4 * 60, batch=8 * 60)
def test_spectra_and_moments_3():
    spectra_and_moments_helper(spec3)


@test_timeout(4 * 60, batch=8 * 60)
def test_spectra_and_moments_4():
    spectra_and_moments_helper(spec4)


@test_timeout(4 * 60, batch=8 * 60)
def test_spectra_and_moments_5():
    spectra_and_moments_helper(spec5)
