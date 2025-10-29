import glob
import logging
import re
import signal
import timeit
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import hail as hl


def init_hail_for_benchmarks(config):
    init_args = {
        'master': f'local[{config.getoption("cores")}]',
        'quiet': config.getoption('verbose') < 0,
        'log': config.getoption('log'),
    }

    if (profile := config.getoption('profile')) is not None:
        if config.getoption('profile_fmt') == 'html':
            filetype = 'html'
            fmt_arg = 'tree=total'
        elif config.getoption('profile_fmt') == 'flame':
            filetype = 'svg'
            fmt_arg = 'svg=total'
        else:
            filetype = 'jfr'
            fmt_arg = 'jfr'

        prof_args = (
            f'-agentpath:{config.getoption("profiler_path")}/build/libasyncProfiler.so=start,'
            f'event={profile},'
            f'{fmt_arg},'
            f'file=bench-profile-{profile}-%t.{filetype},'
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


# Using a custom exception instead of TimeoutError allows explicit handling
class __Timeout(BaseException):
    pass


# https://stackoverflow.com/questions/492519/timeout-on-a-function-call/494273#494273
@contextmanager
def timeout_signal(duration):
    def handler(signum, frame) -> NoReturn:
        try:
            signal.siginterrupt(signal.SIGINT, True)
        except KeyboardInterrupt:
            pass
        finally:
            raise __Timeout()

    restore = signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, restore)
        signal.alarm(0)


# Spark exposes the configuration parameter 'spark.worker.cleanup.enabled'.
# When enabled, spark periodically cleans up temporary files. It's not clear
# how to trigger a clean-up manually or how to configure which directory gets
# used.
def __hack_cleanup_spark_tmpfiles():
    for tmpdir in glob.glob('/tmp/blockmgr*/**/*'):
        Path(tmpdir).unlink()


@contextmanager
def run_with_timeout(max_duration, fn, *args, **kwargs):
    try:
        try:
            timer = timeit.Timer(lambda: fn(*args, **kwargs))
            with timeout_signal(max_duration):
                duration = timer.timeit(1)
        except __Timeout as _:
            from hail.backend.spark_backend import SparkBackend

            if isinstance(b := hl.current_backend(), SparkBackend):
                b.sc.cancelAllJobs()

            yield (max_duration, True, traceback.format_exc())
            raise TimeoutError(f'Timed out after {max_duration}s')
        except Exception:
            yield (None, False, traceback.format_exc())
            raise

        yield duration, False, None
    finally:
        __hack_cleanup_spark_tmpfiles()


__peak_mem_pattern = re.compile(r'.*TaskReport:.*peakBytes=(\d+),.*')


def get_peak_task_memory(log_path) -> int:
    with open(log_path, 'r') as f:
        task_peak_bytes = [
            int(match.groups()[0]) for line in f if (match := __peak_mem_pattern.match(line)) is not None
        ]

    if len(task_peak_bytes) == 0:
        return 0

    return max(task_peak_bytes)
