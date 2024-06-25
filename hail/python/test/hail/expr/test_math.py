import scipy.special as scsp

import hail as hl


def test_logit():
    assert hl.eval(hl.logit(0.5)) == 0.0
    assert hl.eval(hl.is_infinite(hl.logit(1.0)))
    assert hl.eval(hl.is_nan(hl.logit(1.01)))
    assert hl.eval(hl.logit(0.27)) == scsp.logit(0.27)


def test_expit():
    assert hl.eval(hl.expit(0.0)) == 0.5
    assert hl.eval(hl.expit(800)) == 1.0
    assert hl.eval(hl.expit(-920)) == 0.0
    assert hl.eval(hl.expit(0.75)) == scsp.expit(0.75)
