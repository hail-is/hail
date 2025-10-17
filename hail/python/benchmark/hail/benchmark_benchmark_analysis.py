import pytest
import tempfile

from benchmark.tools.impex import import_benchmarks
from benchmark.tools.statistics import analyze_benchmarks, laaber_mds, schultz_mds

import hail as hl


def benchmark_analyze_benchmarks(local_tmpdir, onethreetwo, onethreethree):
    tables = (import_benchmarks(v, local_tmpdir) for v in (onethreetwo, onethreethree))
    tables = (t.select(instances=t.instances.iterations.time) for t in tables)
    tables = (t._key_by_assert_sorted(*t.key.drop('version')) for t in tables)
    tables = (t.checkpoint(tempfile.mktemp(suffix='.ht', dir=local_tmpdir)) for t in tables)

    results = analyze_benchmarks(
        *tables,
        n_bootstrap_iterations=1000,
        confidence=0.95,
    )

    results._force_count()


@pytest.fixture(scope='session')
def _100_instances_100_iterations(resource_dir):
    rows = lambda n, _: [hl.struct(key=0, instances=hl.repeat(hl.repeat(1.0, n), n))]
    ht = hl.Table._generate(contexts=[100],partitions=1, rowfn=rows)
    ht = ht._key_by_assert_sorted('key')
    return ht.checkpoint(f'{resource_dir}/100_instances_100_iterations.ht')


@pytest.mark.parametrize('mds', [laaber_mds, schultz_mds])
def benchmark_minimal_detectable_slowdown(_100_instances_100_iterations, mds):
    mds(_100_instances_100_iterations, n_bootstrap_iterations=100)._force_count()
