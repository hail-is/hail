import pytest

import hail as hl


def test_lgt_to_gt():
    call_0_0_f = hl.call(0, 0, phased=False)
    call_0_0_t = hl.call(0, 0, phased=True)
    call_0_1_f = hl.call(0, 1, phased=False)
    call_2_0_t = hl.call(2, 0, phased=True)

    call_1 = hl.call(1, phased=False)

    la = [0, 3, 5]

    assert hl.eval(
        tuple(hl.vds.lgt_to_gt(c, la) for c in [call_0_0_f, call_0_0_t, call_0_1_f, call_2_0_t, call_1])
    ) == tuple([
        hl.Call([0, 0], phased=False),
        hl.Call([0, 0], phased=True),
        hl.Call([0, 3], phased=False),
        hl.Call([5, 0], phased=True),
        hl.Call([3], phased=False),
    ])

    assert hl.eval(hl.vds.lgt_to_gt(call_0_0_f, hl.missing('array<int32>'))) == hl.Call([0, 0], phased=False)


def test_lgt_to_gt_invalid():
    c1 = hl.call(1, 1)
    assert hl.eval(hl.vds.lgt_to_gt(c1, [0, 17495])) == hl.Call([17495, 17495])
    # the below fails because phasing uses the sum of j and k for its second allele.
    # we cannot represent this allele index in 28 bits
    # c2 = hl.call(1, 1, phased=True)
    # assert hl.eval(hl.vds.lgt_to_gt(c2, [0, 17495])) == hl.Call([17495, 17495], phased=True)


def test_local_to_global():
    local_alleles = [0, 1, 3]
    lad = [1, 9, 10]
    lpl = [1001, 1002, 1003, 1004, 0, 1005]

    assert hl.eval(hl.vds.local_to_global(lad, local_alleles, 4, 0, number='R')) == [1, 9, 0, 10]
    assert hl.eval(hl.vds.local_to_global(lpl, local_alleles, 4, 999, number='G')) == [
        1001,
        1002,
        1003,
        999,
        999,
        999,
        1004,
        0,
        999,
        1005,
    ]
    assert hl.eval(hl.vds.local_to_global(lad, [0, 1, 2], 3, 0, number='R')) == lad
    assert hl.eval(hl.vds.local_to_global(lpl, [0, 1, 2], 3, 999, number='G')) == lpl


def test_local_to_global_alleles_non_increasing():
    local_alleles = [0, 3, 1]
    lad = [1, 10, 9]
    lpl = [1001, 1004, 0, 1002, 1003, 1005]

    assert hl.eval(hl.vds.local_to_global(lad, local_alleles, 4, 0, number='R')) == [1, 9, 0, 10]
    assert hl.eval(hl.vds.local_to_global(lpl, local_alleles, 4, 999, number='G')) == [
        1001,
        1002,
        1005,
        999,
        999,
        999,
        1004,
        1003,
        999,
        0,
    ]

    assert hl.eval(hl.vds.local_to_global([0, 1, 2, 3, 4, 5], [0, 2, 1], 3, 0, number='G')) == [0, 3, 5, 1, 4, 2]


def test_local_to_global_missing_fill():
    local_alleles = [0, 3, 1]
    lad = [1, 10, 9]
    assert hl.eval(hl.vds.local_to_global(lad, local_alleles, 4, hl.missing('int32'), number='R')) == [1, 9, None, 10]


def test_local_to_global_out_of_bounds():
    local_alleles = [0, 2]
    lad = [1, 9]
    lpl = [1001, 0, 1002]

    with pytest.raises(
        hl.utils.HailUserError, match='local_to_global: local allele of 2 out of bounds given n_total_alleles of 2'
    ):
        assert hl.eval(hl.vds.local_to_global(lad, local_alleles, 2, 0, number='R')) == [1, 0]

    with pytest.raises(
        hl.utils.HailUserError, match='local_to_global: local allele of 2 out of bounds given n_total_alleles of 2'
    ):
        assert hl.eval(hl.vds.local_to_global(lpl, local_alleles, 2, 10001, number='G')) == [
            1001,
            10001,
            0,
            10001,
            10001,
            1002,
        ]
