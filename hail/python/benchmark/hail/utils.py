import contextlib
import logging
import re
import signal
import timeit
import traceback
from contextlib import contextmanager

import hail as hl


def init_hail_for_benchmarks(run_config):
    init_args = {
        'master': f'local[{run_config.cores}]',
        'quiet': not run_config.verbose,
        'log': run_config.log,
    }

    if run_config.profile is not None:
        if run_config.profile_fmt == 'html':
            filetype = 'html'
            fmt_arg = 'tree=total'
        elif run_config.profile_fmt == 'flame':
            filetype = 'svg'
            fmt_arg = 'svg=total'
        else:
            filetype = 'jfr'
            fmt_arg = 'jfr'

        prof_args = (
            f'-agentpath:{run_config.profiler_path}/build/libasyncProfiler.so=start,'
            f'event={run_config.profile},'
            f'{fmt_arg},'
            f'file=bench-profile-{run_config.profile}-%t.{filetype},'
            'interval=1ms,'
            'framebuf=15000000'
        )

        init_args['spark_conf'] = {
            'spark.driver.extraJavaOptions': prof_args,
            'spark.executor.extraJavaOptions': prof_args,
        }

    hl.init(**init_args)
    # make JVM do something to ensure that it is fresh
    hl.utils.range_table(1)._force_count()
    logging.getLogger('py4j').setLevel(logging.CRITICAL)
    logging.getLogger('py4j.java_gateway').setLevel(logging.CRITICAL)


__timeout_state = False


# https://stackoverflow.com/questions/492519/timeout-on-a-function-call/494273#494273
@contextlib.contextmanager
def timeout_signal(run_config):
    global __timeout_state
    __timeout_state = False

    def handler(signum, frame):
        global __timeout_state
        __timeout_state = True
        try:
            hl.stop()
            init_hail_for_benchmarks(run_config)
        except Exception:
            traceback.print_exc()  # we're fucked.

        raise TimeoutError(f'Timed out after {run_config.timeout}s')

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(run_config.timeout)

    try:
        yield
    finally:

        def no_op(signum, frame):
            pass

        signal.signal(signal.SIGALRM, no_op)
        signal.alarm(0)


@contextmanager
def run_with_timeout(run_config, fn, *args, **kwargs):
    with timeout_signal(run_config):
        try:
            timer = timeit.Timer(lambda: fn(*args, **kwargs)).timeit(1)
            yield timer, False, None
        except Exception as e:
            timed_out = isinstance(e, TimeoutError)
            yield (run_config.timeout if timed_out else None, timed_out, traceback.format_exc())
            raise e


__peak_mem_pattern = re.compile(r'.*TaskReport:.*peakBytes=(\d+),.*')


def get_peak_task_memory(log_path) -> int:
    with open(log_path, 'r') as f:
        task_peak_bytes = [
            int(match.groups()[0])
            for line in f
            if (match := __peak_mem_pattern.match(line)) is not None  #
        ]

    if len(task_peak_bytes) == 0:
        return 0

    return max(task_peak_bytes)


def select(keys, **kwargs):
    return (kwargs.get(k, None) for k in keys)
