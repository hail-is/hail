import timeit

import contextlib
import hail as hl
import logging
import os
import re
import shutil
import signal
from py4j.protocol import Py4JError

from .resources import all_resources


class BenchmarkTimeoutError(KeyboardInterrupt):
    pass


_timeout_state = False
_init_args = {}


def recursive_delete(path):
    shutil.rmtree(path)


# https://stackoverflow.com/questions/492519/timeout-on-a-function-call/494273#494273
@contextlib.contextmanager
def timeout_signal(time_in_seconds):
    global _timeout_state
    _timeout_state = False

    def handler(signum, frame):
        global _timeout_state
        _timeout_state = True
        hl.stop()
        hl.init(**_init_args)
        raise BenchmarkTimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_in_seconds)

    try:
        yield
    finally:

        def no_op(signum, frame):
            pass

        signal.signal(signal.SIGALRM, no_op)
        signal.alarm(0)


def benchmark(args=()):
    if len(args) == 2 and callable(args[1]):
        args = (args,)

    groups = set(h[0] for h in args)
    fs = tuple(h[1] for h in args)

    def inner(f):
        _registry[f.__name__] = Benchmark(f, f.__name__, groups, fs)

    return inner


class Benchmark:
    def __init__(self, f, name, groups, args):
        self.name = name
        self.f = f
        self.groups = groups
        self.args = args

    def run(self, data_dir):
        return self.f(*(arg(data_dir) for arg in self.args))


class RunConfig:
    def __init__(
        self, n_iter, handler, noisy, timeout, dry_run, data_dir, cores, verbose, log, profiler_path, profile, prof_fmt
    ):
        self.n_iter = n_iter
        self.handler = handler
        self.noisy = noisy
        self.timeout = timeout
        self.dry_run = dry_run
        self.data_dir = data_dir
        self.cores = cores
        self.hail_verbose = verbose
        self.log = log
        self.profiler_path = profiler_path
        self.profile = profile
        self.prof_fmt = prof_fmt


_registry = {}
_initialized = False


def ensure_single_resource(data_dir, group):
    resources = [r for r in all_resources if r.name() == group]
    if not resources:
        raise RuntimeError(f"no group {group!r}")
    ensure_resources(data_dir, resources)


def ensure_resources(data_dir, resources):
    logging.info(f'using benchmark data directory {data_dir}')
    os.makedirs(data_dir, exist_ok=True)
    to_create = []
    for rg in resources:
        if not rg.exists(data_dir):
            to_create.append(rg)
    if to_create:
        hl.init()
        for rg in to_create:
            rg.create(data_dir)
        hl.stop()


def _ensure_initialized():
    if not _initialized:
        raise AssertionError(
            "Hail benchmark environment not initialized. " "Are you running benchmark from the main module?"
        )


def stop():
    global _initialized
    _initialized = False
    hl.stop()


def initialize(config):
    global _initialized, _mt, _init_args
    assert not _initialized
    _init_args = {'master': f'local[{config.cores}]', 'quiet': not config.hail_verbose, 'log': config.log}

    if config.profile is not None:
        if config.prof_fmt == 'html':
            filetype = 'html'
            fmt_arg = 'tree=total'
        elif config.prof_fmt == 'flame':
            filetype = 'svg'
            fmt_arg = 'svg=total'
        else:
            filetype = 'jfr'
            fmt_arg = 'jfr'

        prof_args = (
            f'-agentpath:{config.profiler_path}/build/libasyncProfiler.so=start,'
            f'event={config.profile},'
            f'{fmt_arg},'
            f'file=bench-profile-{config.profile}-%t.{filetype},'
            'interval=1ms,'
            'framebuf=15000000'
        )

        _init_args['spark_conf'] = {
            'spark.driver.extraJavaOptions': prof_args,
            'spark.executor.extraJavaOptions': prof_args,
        }

    hl.init(**_init_args)
    _initialized = True

    # make JVM do something to ensure that it is fresh
    hl.utils.range_table(1)._force_count()
    logging.getLogger('py4j').setLevel(logging.CRITICAL)
    logging.getLogger('py4j.java_gateway').setLevel(logging.CRITICAL)


def run_with_timeout(b, config):
    max_time = config.timeout
    with timeout_signal(max_time):
        try:
            cleanup_container = []

            def runner():
                result = b.run(config.data_dir)
                if result is not None:
                    assert callable(result)
                    cleanup_container.append(result)

            timer = timeit.Timer(runner).timeit(1)

            if cleanup_container:
                [cleanup] = cleanup_container
                assert callable(cleanup)
                cleanup()

            return timer, False
        except Py4JError as e:
            if _timeout_state:
                return max_time, True
            raise
        except BenchmarkTimeoutError as e:
            return max_time, True


def get_peak_task_memory(log_path) -> int:
    peak_memory_per_task = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(r'.*TaskReport:.*peakBytes=(\d+),.*', line)
            if match is not None:
                peak_memory_per_task.append(int(match.groups()[0]))
    if len(peak_memory_per_task) == 0:
        return 0
    return max(peak_memory_per_task)


def _run(benchmark: Benchmark, config: RunConfig, context):
    _ensure_initialized()
    if config.noisy:
        logging.info(f'{context}Running {benchmark.name}...')
    times = []

    timed_out = False
    try:
        burn_in_time, burn_in_timed_out = run_with_timeout(benchmark, config)
        if burn_in_timed_out:
            if config.noisy:
                logging.warning(f'burn in timed out after {burn_in_time:.2f}s')
            timed_out = True
            times.append(float(burn_in_time))
        elif config.noisy:
            logging.info(f'burn in: {burn_in_time:.2f}s')
    except Exception as e:  # pylint: disable=broad-except
        if config.noisy:
            logging.error('burn in: Caught exception')
            logging.exception(e)
        config.handler({'name': benchmark.name, 'failed': True})
        return

    for i in range(config.n_iter):
        if timed_out:
            continue
        try:
            t, run_timed_out = run_with_timeout(benchmark, config)
            times.append(t)
            if run_timed_out:
                if config.noisy:
                    logging.warning(f'run {i + 1} timed out after {t:.2f}s')
                    timed_out = True
            elif config.noisy:
                logging.info(f'run {i + 1}: {t:.2f}s')
        except Exception as e:  # pylint: disable=broad-except
            if config.noisy:
                logging.error(f'run ${i + 1}: Caught exception: {e}')
            config.handler({'name': benchmark.name, 'failed': True})
            return

    from hail.utils.java import Env

    peak_task_memory = get_peak_task_memory(Env.hc()._log)
    config.handler({
        'name': benchmark.name,
        'failed': False,
        'timed_out': timed_out,
        'times': times,
        'peak_task_memory': [peak_task_memory],
    })


def run_all(config: RunConfig):
    run_list(list(_registry), config)


def run_pattern(pattern, config: RunConfig):
    to_run = []
    regex = re.compile(pattern)
    for name in _registry:
        if regex.search(name):
            to_run.append(name)
    if not to_run:
        raise ValueError(f'pattern {pattern!r} matched no benchmarks')
    run_list(to_run, config)


def run_list(tests, config: RunConfig):
    n_tests = len(tests)
    to_run = []
    for i, name in enumerate(tests):
        b = _registry.get(name)
        if not b:
            raise ValueError(f'test {name!r} not found')
        if config.dry_run:
            logging.info(f'found benchmark {name}')
        else:
            to_run.append(b)
    resources = {rg for b in to_run for rg in b.groups}
    ensure_resources(config.data_dir, resources)
    for i, b in enumerate(to_run):
        initialize(config)
        _run(b, config, f'[{i + 1}/{n_tests}] ')
        try:
            stop()
        except Exception as e:  # pylint: disable=broad-except
            logging.error(f'error while stopping session: {e}')


def list_benchmarks():
    return list(_registry.values())
