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
