import hail as hl
import scipy.stats as spst
import pytest


def test_deprecated_binom_test():
    assert hl.eval(hl.binom_test(2, 10, 0.5, 'two.sided')) == \
        pytest.approx(spst.binom_test(2, 10, 0.5, 'two-sided'))


def test_binom_test():
    arglists = [[2, 10, 0.5, 'two-sided'],
                [4, 10, 0.5, 'less'],
                [32, 50, 0.4, 'greater']]
    for args in arglists:
        assert hl.eval(hl.binom_test(*args)) == pytest.approx(spst.binom_test(*args)), args

def test_pchisqtail():
    def right_tail_from_scipy(x, df, ncp):
        if ncp:
            return 1 - spst.ncx2.cdf(x, df, ncp)
        else:
            return 1 - spst.chi2.cdf(x, df)

    arglists = [[3, 1, 2],
                [5, 1, None],
                [1, 3, 4],
                [1, 3, None],
                [3, 6, 0],
                [3, 6, None]]

    for args in arglists:
        assert hl.eval(hl.pchisqtail(*args)) == pytest.approx(right_tail_from_scipy(*args)), args


def test_shuffle():
    assert set(hl.eval(hl.shuffle(hl.range(5)))) == set(range(5))
