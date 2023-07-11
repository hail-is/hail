import hail as hl

from ..helpers import run_in


@run_in('local')
def test_cdf():
    ht = hl.utils.range_table(100)
    hl.plot.cdf(ht.idx)


@run_in('local')
def test_histogram():
    ht = hl.utils.range_table(100)
    hl.plot.histogram(ht.idx, (0, 100), 10)
    hl.plot.histogram(ht.idx)


@run_in('local')
def test_pdf():
    ht = hl.utils.range_table(100)
    hl.plot.pdf(ht.idx)


@run_in('local')
def test_smoothed_pdf():
    ht = hl.utils.range_table(100)
    hl.plot.smoothed_pdf(ht.idx)


@run_in('local')
def test_cumulative_histogram():
    ht = hl.utils.range_table(100)
    hl.plot.cumulative_histogram(ht.idx, (0, 100), 10)
    hl.plot.cumulative_histogram(ht.idx)


@run_in('local')
def test_histogram2d():
    ht = hl.utils.range_matrix_table(100)
    hl.plot.histogram2d(ht.idx, ht.col_idx)


@run_in('local')
def test_histogram2d():
    ht = hl.utils.range_table(100)
    hl.plot.histogram2d(ht.idx, ht.idx * ht.idx)


@run_in('local')
def test_scatter():
    ht = hl.utils.range_table(100)
    hl.plot.scatter(ht.idx, ht.idx * ht.idx)


@run_in('local')
def test_joint_plot():
    ht = hl.utils.range_table(100)
    hl.plot.joint_plot(ht.idx, ht.idx * ht.idx)


@run_in('local')
def test_qq():
    ht = hl.utils.range_table(100)
    hl.plot.qq(ht.idx / 100)


@run_in('local')
def test_manhattan():
    ht = hl.balding_nichols_model(1, n_variants=100, n_samples=1).rows()
    ht = ht.add_index('idx')
    hl.plot.manhattan(ht.idx / 100)


@run_in('local')
def test_visualize_missingness():
    mt = hl.balding_nichols_model(1, n_variants=100, n_samples=1)
    mt = mt.key_rows_by('locus')
    hl.plot.visualize_missingness(hl.or_missing(hl.rand_bool(0.2), mt.GT))
