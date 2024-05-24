from unittest.mock import patch

import pytest

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
    hl.plot.manhattan(ht.idx / 100)


@pytest.mark.parametrize(
    'name, plot',
    [
        ('manhattan', hl.plot.manhattan),
        ('scatter', lambda x, **kwargs: hl.plot.scatter(x, x, **kwargs)),
        ('join_plot', lambda x, **kwargs: hl.plot.joint_plot(x, x, **kwargs)),
        ('qq', hl.plot.qq),
    ],
)
def test_plots_deprecated_collect_all(name, plot):
    ht = hl.balding_nichols_model(1, n_variants=100, n_samples=1).rows()
    ht = ht.add_index('idx')

    with pytest.raises(ValueError):
        plot(ht.idx, collect_all=True)

    with pytest.raises(ValueError):
        plot(ht.idx, n_divisions=0)

    with patch('warnings.warn') as mock:
        plot(ht.idx, n_divisions=None)
        mock.assert_not_called()

    with patch('warnings.warn') as mock:
        plot(ht.idx, n_divisions=None, collect_all=True)
        mock.assert_called()
        assert name in mock.call_args_list[0][0][0]


def test_visualize_missingness():
    mt = hl.balding_nichols_model(1, n_variants=100, n_samples=1)
    mt = mt.key_rows_by('locus')
    hl.plot.visualize_missingness(hl.or_missing(hl.rand_bool(0.2), mt.GT))
