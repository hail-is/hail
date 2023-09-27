from unittest.mock import patch

import hail as hl


def test_cdf():
    ht = hl.utils.range_table(100)
    hl.plot.cdf(ht.idx)


def test_histogram():
    ht = hl.utils.range_table(100)
    hl.plot.histogram(ht.idx, (0, 100), 10)
    hl.plot.histogram(ht.idx)


def test_pdf():
    ht = hl.utils.range_table(100)
    hl.plot.pdf(ht.idx)


def test_smoothed_pdf():
    ht = hl.utils.range_table(100)
    hl.plot.smoothed_pdf(ht.idx)


def test_cumulative_histogram():
    ht = hl.utils.range_table(100)
    hl.plot.cumulative_histogram(ht.idx, (0, 100), 10)
    hl.plot.cumulative_histogram(ht.idx)


def test_histogram2d():
    ht = hl.utils.range_matrix_table(100)
    hl.plot.histogram2d(ht.idx, ht.col_idx)


def test_histogram2d():
    ht = hl.utils.range_table(100)
    hl.plot.histogram2d(ht.idx, ht.idx * ht.idx)


def test_scatter():
    ht = hl.utils.range_table(100)
    hl.plot.scatter(ht.idx, ht.idx * ht.idx)


def test_joint_plot():
    ht = hl.utils.range_table(100)
    hl.plot.joint_plot(ht.idx, ht.idx * ht.idx)


def test_qq():
    ht = hl.utils.range_table(100)
    hl.plot.qq(ht.idx / 100)

def test_manhattan():
    ht = hl.balding_nichols_model(1, n_variants=100, n_samples=1).rows()
    ht = ht.add_index('idx')

    with patch('warnings.warn') as mock:
        hl.plot.manhattan(ht.idx / 100)
        mock.assert_not_called()

    with patch('warnings.warn') as mock:
        hl.plot.manhattan(ht.idx / 100, collect_all=True)
        mock.assert_called()



def test_visualize_missingness():
    mt = hl.balding_nichols_model(1, n_variants=100, n_samples=1)
    mt = mt.key_rows_by('locus')
    hl.plot.visualize_missingness(hl.or_missing(hl.rand_bool(0.2), mt.GT))
