import pytest

import hail as hl

from ..helpers import fails_local_backend, fails_service_backend, resource


def assert_c_king_same_as_hail_king(c_king_path, hail_king_mt):
    actual = hail_king_mt.entries()
    expected = hl.import_table(c_king_path, types={'Kinship': hl.tfloat}, key=['ID1', 'ID2'])
    expected = expected.rename({'ID1': 's_1', 'ID2': 's', 'Kinship': 'phi'})
    expected = expected.key_by('s_1', 's')
    expected = expected.annotate(actual=actual[expected.key])
    expected = expected.select(
        expected=expected.phi, actual=expected.actual.phi, diff=expected.phi - expected.actual.phi
    )
    expected = expected.annotate(
        # KING prints 4 significant digits; but there are several instances
        # where we calculate 0.XXXX5 whereas KING outputs 0.XXXX
        failure=hl.abs(expected.diff) > 0.00006
    )
    expected = expected.filter(expected.failure)
    assert expected.count() == 0, expected.collect()


@fails_service_backend()
@fails_local_backend()
def test_king_small():
    plink_path = resource('balding-nichols-1024-variants-4-samples-3-populations')
    mt = hl.import_plink(bed=f'{plink_path}.bed', bim=f'{plink_path}.bim', fam=f'{plink_path}.fam')
    kinship = hl.king(mt.GT)
    assert_c_king_same_as_hail_king(resource('balding-nichols-1024-variants-4-samples-3-populations.kin0'), kinship)


@pytest.mark.unchecked_allocator
@fails_service_backend()
@fails_local_backend()
def test_king_large():
    plink_path = resource('fastlmmTest')
    mt = hl.import_plink(
        bed=f'{plink_path}.bed', bim=f'{plink_path}.bim', fam=f'{plink_path}.fam', reference_genome=None
    )
    kinship = hl.king(mt.GT)
    assert_c_king_same_as_hail_king(resource('fastlmmTest.kin0.bgz'), kinship)


@fails_service_backend()
@fails_local_backend()
def test_king_filtered_entries_no_error():
    plink_path = resource('balding-nichols-1024-variants-4-samples-3-populations')
    mt = hl.import_plink(bed=f'{plink_path}.bed', bim=f'{plink_path}.bim', fam=f'{plink_path}.fam')
    mt = mt.filter_entries(hl.rand_bool(0.5))
    hl.king(mt.GT)._force_count_rows()
