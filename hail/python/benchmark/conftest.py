import logging
import os

import pytest

from benchmark.tools import init_logging

# hooks that customise command line arguments must be implemented in the top-level conftest.py
# See https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest.hookspec.pytest_addoption


@pytest.hookimpl
def pytest_addoption(parser):
    group = parser.getgroup('benchmark')
    group.addoption("--log", type=str, help='Log file path', default=None)
    group.addoption("--output", type=str, help="Output file path.", default=None)
    group.addoption("--data-dir", type=str, help="Data directory.", default=os.getenv('HAIL_BENCHMARK_DIR'))
    group.addoption(
        "--burn-in-iterations", type=int, help="override number of burn-in iterations for all benchmarks", default=None
    )
    group.addoption('--iterations', type=int, help='override number of iterations for all benchmarks', default=None)
    group.addoption('--cores', type=int, help='Number of cores to use.', default=1)
    group.addoption(
        '--profile',
        choices=['cpu', 'alloc', 'itimer'],
        help='Run with async-profiler.',
        nargs='?',
        const='cpu',
        default=None,
    )
    group.addoption(
        '--max-duration',
        type=int,
        help='Maximum permitted duration for any benchmark trial in seconds, not to be confused with pytest-timeout',
        default=200,
    )
    group.addoption('--max-failures', type=int, help='Stop benchmarking item after this many failures', default=3)
    group.addoption(
        '--profiler-path', type=str, help='path to aysnc profiler', default=os.getenv('ASYNC_PROFILER_HOME')
    )
    group.addoption('--profiler-fmt', choices=['html', 'flame', 'jfr'], help='Choose profiler output.', default='html')


@pytest.hookimpl
def pytest_configure(config):
    init_logging(file=config.getoption('log'))

    if (nburn_in_iterations := config.getoption('burn_in_iterations')) is not None:
        logging.info(f'benchmark: using {nburn_in_iterations} burn-in iterations.')

    if (niterations := config.getoption('iterations')) is not None:
        logging.info(f'benchmark: using {niterations} iterations.')


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    max_duration = config.getoption('max_duration')

    xfail = pytest.mark.xfail(
        raises=TimeoutError,
        reason=f'Runtime exceeds maximum permitted duration of {max_duration}s',
    )

    for item in items:
        if (xtimeout := item.get_closest_marker('xtimeout')) is None:
            continue

        if len(xtimeout.args) == 0 or (len(xtimeout.args) == 1 and xtimeout.args[0] >= max_duration):
            item.add_marker(xfail)
