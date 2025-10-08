import tempfile

from benchmark.tools.impex import import_benchmarks
from benchmark.tools.statistics import analyze_benchmarks


def benchmark_analyze_benchmarks(local_tmpdir, onethreetwo, onethreethree):
    tables = (import_benchmarks(v, local_tmpdir) for v in (onethreetwo, onethreethree))
    tables = (t.select(instances=t.instances.iterations.time) for t in tables)
    tables = (t._key_by_assert_sorted(*t.key.drop('version')) for t in tables)
    tables = (t.checkpoint(tempfile.mktemp(suffix='.mt', dir=local_tmpdir)) for t in tables)

    results = analyze_benchmarks(
        *tables,
        n_bootstrap_iterations=1000,
        confidence=0.95,
    )

    results._force_count()
