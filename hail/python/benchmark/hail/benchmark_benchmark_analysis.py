import tempfile
from pathlib import Path

import pytest

from benchmark.tools.impex import dump_tsv, import_timings
from benchmark.tools.statistics import analyze_benchmarks


@pytest.mark.benchmark()
def benchmark_analyze_benchmarks(local_tmpdir, onethreetwo, onethreethree):
    inputs = (onethreetwo, onethreethree)
    inputs = ((v, Path(tempfile.mktemp(dir=local_tmpdir))) for v in inputs)
    inputs = ((dump_tsv(v, t), t)[-1] for v, t in inputs)

    tables = (import_timings(v) for v in inputs)
    tables = (t.select(instances=t.instances.trials.time) for t in tables)
    tables = (t._key_by_assert_sorted(*t.key.drop('version')) for t in tables)
    tables = (t.checkpoint(tempfile.mktemp(suffix='.mt', dir=local_tmpdir)) for t in tables)

    results = analyze_benchmarks(
        *tables,
        n_bootstrap_iterations=1000,
        confidence=0.95,
    )

    results._force_count()
