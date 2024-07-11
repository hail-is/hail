import tempfile

import pytest

import hail as hl
from benchmark.tools.impex import import_benchmarks
from benchmark.tools.statistics import analyze_benchmarks, laaber_mds, schultz_mds


def test_analyze_benchmarks(local_tmpdir, onethreetwo, onethreethree):
    tables = (import_benchmarks(v, local_tmpdir) for v in (onethreetwo, onethreethree))
    tables = (t.select(instances=t.instances.iterations.time) for t in tables)
    tables = (t._key_by_assert_sorted(*t.key.drop('version')) for t in tables)
    tables = (t.checkpoint(tempfile.mktemp(suffix='.ht', dir=local_tmpdir)) for t in tables)
    analyze_benchmarks(*tables, n_bootstrap_iterations=1000, confidence=0.95)._force_count()


@pytest.fixture(scope='session')
def _100_instances_100_iterations(resource_dir):
    rows = lambda n, _: [hl.struct(id=0, instances=hl.repeat(hl.repeat(1.0, n), n))]
    ht = hl.Table._generate(contexts=[100], partitions=1, rowfn=rows)
    ht = ht._key_by_assert_sorted(ht.id)
    return ht.checkpoint(f'{resource_dir}/100_instances_100_iterations.ht')


@pytest.mark.parametrize('mds', [laaber_mds, schultz_mds])
def test_minimal_detectable_slowdown(_100_instances_100_iterations, mds):
    mds(_100_instances_100_iterations, n_experiments=1)._force_count()
