import os

import pytest

from benchmark.tools import init_logging

# hooks that customise command line arguments must be implemented in the top-level conftest.py
# See https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest.hookspec.pytest_addoption


@pytest.hookimpl
def pytest_addoption(parser):
    parser.addoption("--log", type=str, help='Log file path', default=None)
    parser.addoption("--output", type=str, help="Output file path.", default=None)
    parser.addoption("--data-dir", type=str, help="Data directory.", default=None)
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
    parser.addoption('--profiler-path', type=str, help='path to aysnc profiler', default=None)
    parser.addoption('--profiler-fmt', choices=['html', 'flame', 'jfr'], help='Choose profiler output.', default='html')


def run_config_from_pytest_config(pytest_config):
    return type(
        'RunConfig',
        (object,),
        {
            **{
                flag: pytest_config.getoption(flag) or default
                for flag, default in [
                    ('log', None),
                    ('output', None),
                    ('cores', 1),
                    ('data_dir', os.getenv('HAIL_BENCHMARK_DIR')),
                    ('iterations', None),
                    ('profile', None),
                    ('profiler_path', os.getenv('ASYNC_PROFILER_HOME')),
                    ('profiler_fmt', None),
                ]
            },
            'verbose': pytest_config.getoption('verbose') > 0,
            'quiet': pytest_config.getoption('verbose') < 0,
            'timeout': int(pytest_config.getoption('timeout') or 1800),
        },
    )


@pytest.hookimpl
def pytest_configure(config):
    config.run_config = run_config_from_pytest_config(config)
    init_logging(file=config.run_config.log)
