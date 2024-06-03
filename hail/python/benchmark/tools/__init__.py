import functools
import logging

import pytest


def benchmark(burn_in_iterations=1, iterations=5, batch_jobs=5):
    def wrap(benchmark_fn):
        @pytest.mark.benchmark(burn_in_iterations=burn_in_iterations, iterations=iterations, batch_jobs=batch_jobs)
        @functools.wraps(benchmark_fn)
        def runner(*args, **kwargs):
            return benchmark_fn(*args, **kwargs)

        return runner

    return wrap


def chunk(size, seq):
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]


def init_logging(file=None):
    logging.basicConfig(format="%(asctime)-15s: %(levelname)s: %(message)s", level=logging.INFO, filename=file)
