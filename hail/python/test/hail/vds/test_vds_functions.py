import hail as hl

from ..helpers import startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

def test_lgt_to_gt():
    call_0_0_f = hl.call(0, 0, phased=False)
    call_0_0_t = hl.call(0, 0, phased=True)
    call_0_1_f = hl.call(0, 1, phased=False)
    call_2_0_t = hl.call(2, 0, phased=True)

    call_1 = hl.call(1, phased=False)

    la = [0, 3, 5]

    assert hl.eval(tuple(hl.vds.lgt_to_gt(c, la) for c in [call_0_0_f, call_0_0_t, call_0_1_f, call_2_0_t, call_1])) == \
           tuple([hl.Call([0, 0], phased=False), hl.Call([0, 0], phased=True), hl.Call([0, 3], phased=False), hl.Call([5, 0], phased=True), hl.Call([3], phased=False)])


def test_lgt_to_gt_invalid():

    c1 = hl.call(1, 1)
    c2 = hl.call(1, 1, phased=True)
    assert hl.eval(hl.vds.lgt_to_gt(c1, [0, 17495])) == hl.Call([17495, 17495])
    # the below fails because phasing uses the sum of j and k for its second allele.
    # we cannot represent this allele index in 28 bits
    # assert hl.eval(hl.vds.lgt_to_gt(c2, [0, 17495])) == hl.Call([17495, 17495], phased=True)
