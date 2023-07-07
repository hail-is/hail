import hail as hl
import scipy.stats as spst
import pytest
from ..helpers import resource


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


def test_pgenchisq():
    ht = hl.import_table(
        resource('davies-genchisq-tests.tsv'),
        types={
            'c': hl.tfloat64,
            'weights': hl.tarray(hl.tfloat64),
            'k': hl.tarray(hl.tint32),
            'lam': hl.tarray(hl.tfloat64),
            'sigma': hl.tfloat64,
            'lim': hl.tint32,
            'acc': hl.tfloat64,
            'expected': hl.tfloat64,
            'expected_n_iterations': hl.tint32
        }
    )
    ht = ht.add_index('line_number')
    ht = ht.annotate(line_number = ht.line_number + 1)
    ht = ht.annotate(genchisq_result = hl.pgenchisq(
       ht.c, ht.weights, ht.k, ht.lam, 0.0, ht.sigma, max_iterations=ht.lim, min_accuracy=ht.acc
    ))
    tests = ht.collect()
    for test in tests:
        assert abs(test.genchisq_result.value - test.expected) < 0.0000005, str(test)
        assert test.genchisq_result.fault == 0, str(test)
        assert test.genchisq_result.converged == True, str(test)
        assert test.genchisq_result.n_iterations == test.expected_n_iterations, str(test)


def test_array():
    actual = hl.eval((
        hl.array(hl.array([1, 2, 3, 3])),
        hl.array(hl.set([1, 2, 3])),
        hl.array(hl.dict({1: 5, 7: 4})),
        hl.array(hl.nd.array([1, 2, 3, 3])),
    ))

    expected = (
        [1, 2, 3, 3],
        [1, 2, 3],
        [(1, 5), (7, 4)],
        [1, 2, 3, 3]
    )

    assert actual == expected

    with pytest.raises(ValueError, match='array: only one dimensional ndarrays are supported: ndarray<float64, 2>'):
        hl.eval(hl.array(hl.nd.array([[1.0], [2.0]])))


def test_literal_free_vars():
    "Give better error messages in response to code written by ChatGPT"
    array = hl.literal([1, 2, 3])
    with pytest.raises(ValueError, match='expressions that depend on other expressions'):
        array.map(hl.literal)
