from itertools import pairwise
from pathlib import Path

import pytest

import hail as hl

# The benchmark package is not shipped with the test image, so these tests only
# run from a source checkout.
benchmark_utils = pytest.importorskip('benchmark.hail.utils')


def test_native_path_is_keyed_on_unstable_encodings():
    resource_dir = Path('/data')
    with hl._with_flags(use_unstable_encodings=None):
        stable = benchmark_utils.native_path(resource_dir, 'random_doubles/random_doubles_mt.mt')
    with hl._with_flags(use_unstable_encodings='1'):
        unstable = benchmark_utils.native_path(resource_dir, 'random_doubles/random_doubles_mt.mt')

    assert stable == Path('/data/random_doubles/random_doubles_mt.mt')
    assert unstable == Path('/data/unstable_encodings/random_doubles/random_doubles_mt.mt')


@pytest.mark.backend('local', 'spark')
@pytest.mark.parametrize(
    'unstable, locus_ptype',
    [(None, 'PCanonicalLocus'), ('1', 'PPackedLocus')],
)
def test_decoded_row_ptypes_reports_locus_representation(tmp_path, unstable, locus_ptype):
    if not benchmark_utils.packed_loci_supported():
        locus_ptype = 'PCanonicalLocus'

    path = str(tmp_path / 'loci.ht')
    with hl._with_flags(use_unstable_encodings=unstable):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(locus=hl.locus('1', ht.idx + 1))
        ht.write(path)

    assert benchmark_utils.decoded_row_ptypes(path)['locus'] == locus_ptype


def contig_switches(contigs):
    return sum(a != b for a, b in pairwise(contigs))


@pytest.mark.parametrize('rate', [0.001, 0.5, 1.0])
def test_multi_contig_table_switches_contig_at_rate(rate):
    n = 100_000
    ht = benchmark_utils.multi_contig_table(n, rate, n_partitions=4)
    assert len(ht.key) == 0
    contigs = ht.locus.contig.collect()
    assert len(contigs) == n
    assert contig_switches(contigs) / (n - 1) == pytest.approx(rate, rel=0.02)


def test_multi_contig_table_sorted_visits_each_grch38_contig_once_in_order():
    n = 100_000
    ht = benchmark_utils.multi_contig_table(n, 'sorted', n_partitions=4)
    assert len(ht.key) == 0
    loci = ht.locus.collect()
    assert len(loci) == n

    contigs = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    assert [loci[0].contig] + [b.contig for a, b in pairwise(loci) if a.contig != b.contig] == contigs
    assert all(a.position < b.position for a, b in pairwise(loci) if a.contig == b.contig)
