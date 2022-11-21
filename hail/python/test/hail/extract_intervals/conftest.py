import pytest
import hail as hl

from ..helpers import resource


@pytest.fixture(scope='session')
def mt():
    return hl.read_matrix_table(resource('sample.vcf-20-partitions.mt'))


@pytest.fixture(scope='session')
def ht(mt):
    return mt.rows()


@pytest.fixture(scope='session', params=[hl.locus, hl.Locus])
def probe_locus(request):
    return request.param('20', 17434581)


@pytest.fixture(scope='session', params=[hl.locus, hl.Locus])
def probe_variant(request):
    return hl.Struct(locus=request.param('20', 17434581), alleles=['A', 'G'])
