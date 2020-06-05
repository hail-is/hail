import numpy as np

import hail as hl
from hail.utils import new_temp_file
from ..helpers import startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


def test_range_collect():
    n_variants = 10
    n_samples = 10
    block_size = 3
    mt = hl.utils.range_matrix_table(n_variants, n_samples)
    mt = mt.select_entries(x=mt.row_idx * mt.col_idx)

    da = hl.experimental.dnd.array(mt, 'x', block_size=block_size)
    a = np.array(mt.x.collect()).reshape(n_variants, n_samples)

    assert np.array_equal(da.collect(), a)


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
    da_result = da.collect().reshape(n_variants, n_variants)

    a = np.array(mt.x.collect()).reshape(n_variants, n_samples)
    a_result = a @ a.T

    assert np.array_equal(da_result, a_result)


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
    da_result = da.collect().reshape(n_variants, n_variants)

    a = np.array(mt.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a @ a.T

    assert np.array_equal(da_result, a_result)


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
    da_result = da.collect().reshape(n_variants, n_variants)

    a = np.array(mt.dosage.collect()).reshape(n_variants, n_samples)
    a_result = a @ a.T

    assert np.array_equal(da_result, a_result)
