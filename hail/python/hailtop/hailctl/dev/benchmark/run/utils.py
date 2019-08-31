import logging
import os
import timeit
from urllib.request import urlretrieve

import numpy as np
import re

import hail as hl
from .. import init_logging


def resource(filename):
    assert _initialized
    return os.path.join(_data_dir, filename)


def get_mt():
    return _mt


def benchmark(f):
    _registry[f.__name__] = Benchmark(f, f.__name__)
    return f


class Benchmark:
    def __init__(self, f, name):
        self.name = name
        self.f = f

    def run(self):
        self.f()


class RunConfig:
    def __init__(self, n_iter, handler, verbose):
        self.n_iter = n_iter
        self.handler = handler
        self.verbose = verbose


_registry = {}
_data_dir = ''
_mt = None
_initialized = False


def download_data(data_dir):
    global _data_dir, _mt
    _data_dir = data_dir or os.environ.get('HAIL_BENCHMARK_DIR') or '/tmp/hail_benchmark_data'
    logging.info(f'using benchmark data directory {_data_dir}')
    os.makedirs(_data_dir, exist_ok=True)

    files = map(lambda f: os.path.join(_data_dir, f), ['profile.vcf.bgz',
                                                       'profile.mt',
                                                       'table_10M_par_1000.ht',
                                                       'table_10M_par_100.ht',
                                                       'table_10M_par_10.ht',
                                                       'gnomad_dp_simulation.mt',
                                                       'many_strings_table.ht',
                                                       'many_ints_table.ht',
                                                       'sim_ukb.bgen'])
    if not all(os.path.exists(file) for file in files):
        hl.init()  # use all cores

        vcf = os.path.join(_data_dir, 'profile.vcf.bgz')
        logging.info('downloading profile.vcf.bgz...')
        urlretrieve('https://storage.googleapis.com/hail-common/benchmark/profile.vcf.bgz', vcf)
        logging.info('done downloading profile.vcf.bgz.')
        logging.info('importing profile.vcf.bgz...')
        hl.import_vcf(vcf, min_partitions=16).write(os.path.join(_data_dir, 'profile.mt'), overwrite=True)
        logging.info('done importing profile.vcf.bgz.')

        logging.info('writing 10M row partitioned tables...')

        ht = hl.utils.range_table(10_000_000, 1000).annotate(**{f'f_{i}': hl.rand_unif(0, 1) for i in range(5)})
        ht = ht.checkpoint(os.path.join(_data_dir, 'table_10M_par_1000.ht'), overwrite=True)
        ht = ht.naive_coalesce(100).checkpoint(os.path.join(_data_dir, 'table_10M_par_100.ht'), overwrite=True)
        ht.naive_coalesce(10).write(os.path.join(_data_dir, 'table_10M_par_10.ht'), overwrite=True)
        logging.info('done writing 10M row partitioned tables.')

        logging.info('creating gnomad_dp_simulation matrix table...')
        mt = hl.utils.range_matrix_table(n_rows=250_000, n_cols=1_000, n_partitions=32)
        mt = mt.annotate_entries(x=hl.int(hl.rand_unif(0, 4.5) ** 3))
        mt.write(os.path.join(_data_dir, 'gnomad_dp_simulation.mt'), overwrite=True)
        logging.info('done creating gnomad_dp_simulation matrix table.')

        logging.info('downloading many_strings_table.tsv.bgz...')
        mst_tsv = os.path.join(_data_dir, 'many_strings_table.tsv.bgz')
        mst_ht = os.path.join(_data_dir, 'many_strings_table.ht')
        urlretrieve('https://storage.googleapis.com/hail-common/benchmark/many_strings_table.tsv.bgz', mst_tsv)
        logging.info('done downloading many_strings_table.tsv.bgz.')
        logging.info('importing many_strings_table.tsv.bgz...')
        hl.import_table(mst_tsv).write(mst_ht, overwrite=True)
        logging.info('done importing many_strings_table.tsv.bgz.')

        logging.info('downloading many_ints_table.tsv.bgz...')
        mit_tsv = os.path.join(_data_dir, 'many_ints_table.tsv.bgz')
        mit_ht = os.path.join(_data_dir, 'many_ints_table.ht')
        urlretrieve('https://storage.googleapis.com/hail-common/benchmark/many_ints_table.tsv.bgz', mit_tsv)
        logging.info('done downloading many_ints_table.tsv.bgz.')
        logging.info('importing many_ints_table.tsv.bgz...')
        hl.import_table(mit_tsv,
                        types={'idx': 'int',
                               **{f'i{i}': 'int' for i in range(5)},
                               **{f'array{i}': 'array<int>' for i in range(2)}}
                        ).write(mit_ht, overwrite=True)
        logging.info('done importing many_ints_table.tsv.bgz.')

        bgen = 'sim_ukb.bgen'
        sample = 'sim_ukb.sample'
        logging.info(f'downloading {bgen}...')
        local_bgen = os.path.join(_data_dir, bgen)
        local_sample = os.path.join(_data_dir, sample)
        urlretrieve(f'https://storage.googleapis.com/hail-common/benchmark/{bgen}', local_bgen)
        urlretrieve(f'https://storage.googleapis.com/hail-common/benchmark/{sample}', local_sample)
        logging.info(f'done downloading {bgen}...')
        logging.info(f'indexing {bgen}...')
        hl.index_bgen(local_bgen)
        logging.info(f'done indexing {bgen}.')

        hl.stop()
    else:
        logging.info('all files found.')


def _ensure_initialized():
    if not _initialized:
        raise AssertionError("Hail benchmark environment not initialized. "
                             "Are you running benchmark from the main module?")


def initialize(args):
    global _initialized, _mt
    assert not _initialized
    init_logging()
    download_data(args.data_dir)
    hl.init(master=f'local[{args.cores}]', quiet=True, log=args.log)
    _initialized = True
    _mt = hl.read_matrix_table(resource('profile.mt'))

    # make JVM do something to ensure that it is fresh
    hl.utils.range_table(1)._force_count()


def _run(benchmark: Benchmark, config: RunConfig, context):
    if config.verbose:
        logging.info(f'{context}Running {benchmark.name}...')
    times = []

    try:
        burn_in_time = timeit.Timer(benchmark.run).timeit(1)
        if config.verbose:
            logging.info(f'    burn in: {burn_in_time:.2f}s')
    except Exception as e:  # pylint: disable=broad-except
        if config.verbose:
            logging.error(f'    burn in: Caught exception: {e}')
        config.handler({'name': benchmark.name,
                        'failed': True})
        return

    for i in range(config.n_iter):
        try:
            time = timeit.Timer(lambda: benchmark.run()).timeit(1)  # pylint: disable=unnecessary-lambda
            times.append(time)
            if config.verbose:
                logging.info(f'    run {i + 1}: {time:.2f}s')
        except Exception as e:  # pylint: disable=broad-except
            if config.verbose:
                logging.error(f'    run ${i + 1}: Caught exception: {e}')
            config.handler({'name': benchmark.name,
                            'failed': True})
            return
    config.handler({'name': benchmark.name,
                    'failed': False,
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'stdev': np.std(times),
                    'times': times})


def run_all(config: RunConfig):
    _ensure_initialized()
    run_list(list(_registry), config)


def run_pattern(pattern, config: RunConfig):
    _ensure_initialized()
    to_run = []
    regex = re.compile(pattern)
    for name in _registry:
        if regex.search(name):
            to_run.append(name)
    if not to_run:
        raise ValueError(f'pattern {pattern!r} matched no benchmarks')
    run_list(to_run, config)


def run_list(tests, config: RunConfig):
    _ensure_initialized()

    n_tests = len(tests)
    for i, name in enumerate(tests):
        if name not in _registry:
            raise ValueError(f'test {name!r} not found')
        _run(_registry[name], config, f'[{i + 1}/{n_tests}] ')


def list_benchmarks():
    return list(_registry)
