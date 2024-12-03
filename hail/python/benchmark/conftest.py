import os

import pytest

from benchmark.tools import init_logging

# hooks that customise command line arguments must be implemented in the top-level conftest.py
# See https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest.hookspec.pytest_addoption


@pytest.hookimpl
def pytest_addoption(parser):
    parser.addoption("--log", type=str, help='Log file path', default=None)
    parser.addoption("--output", type=str, help="Output file path.", default=None)
    parser.addoption("--data-dir", type=str, help="Data directory.", default=os.getenv('HAIL_BENCHMARK_DIR'))
    parser.addoption('--iterations', type=int, help='override number of iterations for all benchmarks', default=None)
    parser.addoption('--cores', type=int, help='Number of cores to use.', default=1)
    parser.addoption(
        '--profile',
        choices=['cpu', 'alloc', 'itimer'],
        help='Run with async-profiler.',
        nargs='?',
        const='cpu',
        default=None,
    )
    parser.addoption(
        '--max-duration',
        type=int,
        help='Maximum permitted duration for any benchmark trial in seconds, not to be confused with pytest-timeout',
        default=200,
    )
    parser.addoption('--max-failures', type=int, help='Stop benchmarking item after this many failures', default=3)
    parser.addoption(
        '--profiler-path', type=str, help='path to aysnc profiler', default=os.getenv('ASYNC_PROFILER_HOME')
    )
    parser.addoption('--profiler-fmt', choices=['html', 'flame', 'jfr'], help='Choose profiler output.', default='html')


@pytest.hookimpl
def pytest_configure(config):
    init_logging(file=config.getoption('log'))


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
