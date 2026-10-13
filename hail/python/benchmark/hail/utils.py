import glob
import logging
import re
import signal
import timeit
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, NoReturn, Union

from py4j.java_gateway import JavaClass

import hail as hl
from hail.utils.java import Env


def use_unstable_encodings() -> bool:
    return hl._get_flags('use_unstable_encodings')['use_unstable_encodings'] is not None


def native_path(resource_dir: Path, relpath: Union[str, Path]) -> Path:
    """Path under `resource_dir` for a native file written by a fixture.

    Files written with unstable encodings are encoded differently (for example,
    loci decode as packed), so they live in their own tree and a run never reuses
    a file written under the other setting.
    """
    if use_unstable_encodings():
        return resource_dir / 'unstable_encodings' / relpath
    return resource_dir / relpath


GRCH38_CONTIGS = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']


def multi_contig_table(n_rows: int, contig_switch_rate: Union[str, float], n_partitions: int) -> hl.Table:
    """An unkeyed table of `n_rows` GRCh38 loci whose contig changes between
    consecutive rows at `contig_switch_rate`.

    Rows cycle through the contigs in runs of `1 / contig_switch_rate` rows. The
    rows of each contig get positions 1, 2, ... in row order. With 'sorted', each
    contig is one run, so the loci are in sorted order.
    """
    n_contigs = len(GRCH38_CONTIGS)
    if contig_switch_rate == 'sorted':
        run_length = -(-n_rows // n_contigs)  # ceil(n_rows / n_contigs)
    else:
        run_length = round(1 / contig_switch_rate)

    ht = hl.utils.range_table(n_rows, n_partitions)
    contig = hl.literal(GRCH38_CONTIGS)[(ht.idx // run_length) % n_contigs]
    position = (ht.idx // (run_length * n_contigs)) * run_length + ht.idx % run_length + 1
    return ht.select(locus=hl.locus(contig, position, reference_genome='GRCh38')).key_by()


def decoded_row_ptypes(table_path: Union[str, Path]) -> Dict[str, str]:
    """The JVM physical type (class name) each top-level row field of the native
    table at `table_path` decodes to, e.g. `{'locus': 'PPackedLocus'}`."""
    backend = Env.py4j_backend('decoded_row_ptypes')
    spec = backend.hail_package().rvd.AbstractRVDSpec.read(backend._jbackend.pyFs(), f'{table_path}/rows')
    fields = spec.typedCodecSpec().decodedPType().fields()
    return {(f := fields.apply(i)).name(): f.typ().getClass().getSimpleName() for i in range(fields.size())}


def packed_loci_supported() -> bool:
    """Whether this version of hail has packed loci. Versions without them write
    canonical loci whatever use_unstable_encodings says."""
    physical = Env.py4j_backend('packed_loci_supported').hail_package().types.physical
    return isinstance(physical.PPackedLocus, JavaClass)


def check_locus_encoding(table_path: Union[str, Path]) -> None:
    """Assert that the loci of the native table at `table_path` decode the way the
    active backend writes them, so a benchmark never measures loci encoded under
    the other setting of use_unstable_encodings."""
    expected = 'PPackedLocus' if packed_loci_supported() and use_unstable_encodings() else 'PCanonicalLocus'
    loci = {name: ptype for name, ptype in decoded_row_ptypes(table_path).items() if ptype.endswith('Locus')}
    assert loci and all(ptype == expected for ptype in loci.values()), (
        f'{table_path}: expected {expected} loci, found {loci}'
    )


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
                b._spark._sc.cancelAllJobs()

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
