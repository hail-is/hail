import functools
import glob
import gzip
import json
import logging
import re
import signal
import timeit
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, NoReturn, Tuple, Union

import hail as hl


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


_OPEN = {'{': '}', '[': ']', '(': ')', '<': '>'}
_CLOSE = set(_OPEN.values())


def _struct_fields(t: str) -> List[Tuple[str, str]]:
    """Split the top level of a printed struct type (`+EBaseStruct{a:EInt32,...}`
    or `Struct{a:Int32,...}`) into (field name, field type) pairs."""
    t = t.lstrip('+')
    head, brace, body = t.partition('{')
    assert brace and head in ('EBaseStruct', 'Struct') and body.endswith('}'), t
    body = body[:-1]

    fields, start, depth, quoted, i = [], 0, 0, False, 0
    while i <= len(body):
        c = body[i] if i < len(body) else ','
        if quoted:
            if c == '\\':
                i += 1
            elif c == '`':
                quoted = False
        elif c == '`':
            quoted = True
        elif c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == ',' and depth == 0:
            if i > start:
                fields.append(body[start:i])
            start = i + 1
        i += 1

    return [_split_field(f) for f in fields]


def _split_field(f: str) -> Tuple[str, str]:
    # Names are left as printed (quoted and escaped if need be): they are only
    # matched between the _vType and _eType of one spec, which print them alike.
    end = 0
    if f.startswith('`'):
        end = f.index('`', 1)
        while f[end - 1] == '\\':
            end = f.index('`', end + 1)
        end += 1
    end = f.index(':', end)
    return f[:end], f[end + 1 :]


def row_locus_etypes(table_path: Union[str, Path]) -> Dict[str, str]:
    """The encoded type of each top-level locus field in the rows of the native
    table at `table_path`, read from its metadata, e.g.
    `{'locus': 'EBaseStruct{contig:+EBinary,position:+EInt32}'}`."""
    with hl.current_backend().fs.open(f'{table_path}/rows/metadata.json.gz', 'rb') as f:
        with gzip.open(f, 'rt') as g:
            codec = json.load(g)['_codecSpec']
    vfields = _struct_fields(codec['_vType'])
    efields = dict(_struct_fields(codec['_eType']))
    return {name: efields[name].lstrip('+') for name, vtype in vfields if vtype.startswith('Locus(')}


@functools.cache
def _written_locus_etype(unstable: bool) -> str:
    # `unstable` only keys the cache; the write reads the flag itself.
    path = hl.utils.new_temp_file(extension='ht')
    ht = hl.utils.range_table(1)
    ht.annotate(locus=hl.locus('1', 1)).write(path)
    try:
        return row_locus_etypes(path)['locus']
    finally:
        hl.current_backend().fs.rmtree(path)


def written_locus_etype() -> str:
    """The encoded type the active backend gives a locus field, found by writing
    one."""
    return _written_locus_etype(use_unstable_encodings())


def _locus_shape(etype: str) -> str:
    # Struct-encoded loci (contig and position) decode as canonical whatever the
    # encodings of their fields, which vary across versions; others (EVarint)
    # decode as packed.
    return 'struct' if etype.startswith('EBaseStruct{') else etype


def check_locus_encoding(table_path: Union[str, Path]) -> None:
    """Assert that the loci of the native table at `table_path` are encoded the
    way the active backend writes them, so a benchmark never measures loci
    encoded under the other setting of use_unstable_encodings."""
    expected = _locus_shape(written_locus_etype())
    loci = row_locus_etypes(table_path)
    assert loci and all(_locus_shape(etype) == expected for etype in loci.values()), (
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
