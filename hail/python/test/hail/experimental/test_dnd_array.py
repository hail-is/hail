import numpy as np

import hail as hl
from hail.utils import new_temp_file
from ..helpers import startTestHailContext, stopTestHailContext, fails_local_backend, fails_service_backend

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


@fails_service_backend()
def test_range_collect():
    n_variants = 10
    n_samples = 10
    block_size = 3
    mt = hl.utils.range_matrix_table(n_variants, n_samples)
    mt = mt.select_entries(x=mt.row_idx * mt.col_idx)

    da = hl.experimental.dnd.array(mt, 'x', block_size=block_size)
    a = np.array(mt.x.collect()).reshape(n_variants, n_samples)

    assert np.array_equal(da.collect(), a)


@fails_service_backend()
@fails_local_backend()
def test_range_matmul():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt = hl.utils.range_matrix_table(n_variants, n_samples)
    mt = mt.select_entries(x=mt.row_idx * mt.col_idx)

    da = hl.experimental.dnd.array(mt, 'x', block_size=block_size)
    da = (da @ da.T).checkpoint(new_temp_file())
    assert da._force_count_blocks() == n_blocks
    da_result = da.collect()

    a = np.array(mt.x.collect()).reshape(n_variants, n_samples)
    a_result = a @ a.T

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_small_collect():
    n_variants = 10
    n_samples = 10
    block_size = 3
    mt = hl.balding_nichols_model(n_populations=2,
                                  n_variants=n_variants,
                                  n_samples=n_samples)
    mt = mt.select_entries(dosage=hl.float(mt.GT.n_alt_alleles()))

    da = hl.experimental.dnd.array(mt, 'dosage', block_size=block_size)
    a = np.array(mt.dosage.collect()).reshape(n_variants, n_samples)

    assert np.array_equal(da.collect(), a)


@fails_service_backend()
@fails_local_backend()
def test_medium_collect():
    n_variants = 100
    n_samples = 100
    block_size = 32
    mt = hl.balding_nichols_model(n_populations=2,
                                  n_variants=n_variants,
                                  n_samples=n_samples)
    mt = mt.select_entries(dosage=hl.float(mt.GT.n_alt_alleles()))

    da = hl.experimental.dnd.array(mt, 'dosage', block_size=block_size)
    a = np.array(mt.dosage.collect()).reshape(n_variants, n_samples)

    assert np.array_equal(da.collect(), a)


@fails_service_backend()
@fails_local_backend()
def test_small_matmul():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt = hl.balding_nichols_model(n_populations=2,
                                  n_variants=n_variants,
                                  n_samples=n_samples)
    mt = mt.select_entries(dosage=hl.float(mt.GT.n_alt_alleles()))

    da = hl.experimental.dnd.array(mt, 'dosage', block_size=block_size)
    da = (da @ da.T).checkpoint(new_temp_file())
    assert da._force_count_blocks() == n_blocks
    da_result = da.collect()

    a = np.array(mt.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a @ a.T

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_medium_matmul():
    n_variants = 100
    n_samples = 100
    block_size = 32
    n_blocks = 16
    mt = hl.balding_nichols_model(n_populations=2,
                                  n_variants=n_variants,
                                  n_samples=n_samples)
    mt = mt.select_entries(dosage=hl.float(mt.GT.n_alt_alleles()))

    da = hl.experimental.dnd.array(mt, 'dosage', block_size=block_size)
    da = (da @ da.T).checkpoint(new_temp_file())
    assert da._force_count_blocks() == n_blocks
    da_result = da.collect()

    a = np.array(mt.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a @ a.T

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_matmul_via_inner_product():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt = hl.utils.range_matrix_table(n_variants, n_samples)
    mt = mt.select_entries(x=mt.row_idx * mt.col_idx)

    da = hl.experimental.dnd.array(mt, 'x', block_size=block_size)
    prod = (da @ da.T).checkpoint(new_temp_file())
    assert prod._force_count_blocks() == n_blocks
    prod_result = prod.collect()

    ip_result = da.inner_product(da.T,
                                 lambda l, r: l * r,
                                 lambda l, r: l + r,
                                 hl.float(0.0),
                                 lambda prod: hl.agg.sum(prod)
    ).collect()

    assert np.array_equal(prod_result, ip_result)


@fails_service_backend()
@fails_local_backend()
def test_king_homo_estimator():
    hl.set_global_seed(1)
    mt = hl.balding_nichols_model(2, 5, 5)
    mt = mt.select_entries(genotype_score=hl.float(mt.GT.n_alt_alleles()))
    da = hl.experimental.dnd.array(mt, 'genotype_score', block_size=3)

    def sqr(x):
        return x * x
    score_difference = da.T.inner_product(
        da,
        lambda l, r: sqr(l - r),
        lambda l, r: l + r,
        hl.float(0),
        hl.agg.sum
    ).checkpoint(new_temp_file())
    assert np.array_equal(
        score_difference.collect(),
        np.array([[0., 6., 4., 2., 4.],
                  [6., 0., 6., 4., 6.],
                  [4., 6., 0., 6., 0.],
                  [2., 4., 6., 0., 6.],
                  [4., 6., 0., 6., 0.]]))


@fails_service_backend()
@fails_local_backend()
def test_dndarray_sum():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))
    mt2 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt2 = mt2.select_entries(dosage=hl.float(mt2.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da2 = hl.experimental.dnd.array(mt2, 'dosage', block_size=block_size)
    da_sum = (da1 + da2).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a2 = np.array(mt2.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a1 + a2

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_dndarray_sum_scalar():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da_sum = (da1 + 10).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a1 + 10

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_dndarray_rsum_scalar():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da_sum = (10 + da1).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a_result = 10 + a1

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_dndarray_mul_scalar():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da_sum = (da1 * 10).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a1 * 10

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_dndarray_rmul_scalar():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da_sum = (10 * da1).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a_result = 10 * a1

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_dndarray_sub_scalar():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da_sum = (da1 - 10).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a1 - 10

    assert np.array_equal(da_result, a_result)


@fails_service_backend()
@fails_local_backend()
def test_dndarray_rsub_scalar():
    n_variants = 10
    n_samples = 10
    block_size = 3
    n_blocks = 16
    mt1 = hl.balding_nichols_model(n_populations=2,
                                   n_variants=n_variants,
                                   n_samples=n_samples)
    mt1 = mt1.select_entries(dosage=hl.float(mt1.GT.n_alt_alleles()))

    da1 = hl.experimental.dnd.array(mt1, 'dosage', block_size=block_size)
    da_sum = (10 - da1).checkpoint(new_temp_file())
    assert da_sum._force_count_blocks() == n_blocks
    da_result = da_sum.collect()

    a1 = np.array(mt1.dosage.collect()).reshape(n_variants, n_samples)
    a_result = 10 - a1

    assert np.array_equal(da_result, a_result)


def test_dndarray_errors_on_unsorted_columns():
    n_variants = 10
    n_samples = 10
    block_size = 3
    mt = hl.utils.range_matrix_table(n_variants, n_samples)
    mt = mt.key_cols_by(sampleid=hl.str('zyxwvutsrq')[mt.col_idx])
    mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
    try:
        hl.experimental.dnd.array(mt, 'x', block_size=block_size)
    except ValueError as err:
        assert 'columns are not in sorted order', err.args[0]
    else:
        assert False


@fails_service_backend()
@fails_local_backend()
def test_dndarray_sort_columns():
    n_variants = 10
    n_samples = 10
    block_size = 3
    disorder = [0, 9, 8, 7, 1, 2, 3, 4, 6, 5]
    order = [x[0]
             for x in sorted(enumerate(disorder),
                             key=lambda x: x[1])]
    mt = hl.utils.range_matrix_table(n_variants, n_samples)
    mt = mt.key_cols_by(sampleid=hl.literal(disorder)[mt.col_idx])
    mt = mt.select_entries(x=mt.row_idx * mt.col_idx)
    da = hl.experimental.dnd.array(mt, 'x', block_size=block_size, sort_columns=True)

    a = np.array(
        [r * order[c] for r in range(n_variants) for c in range(n_samples)]
    ).reshape((n_variants, n_samples))

    assert np.array_equal(da.collect(), a)

    result = (da.T @ da).collect()
    expected = a.T @ a
    assert np.array_equal(result, expected)
