#!/usr/bin/env python3

import json
import logging
import random
import time
from argparse import ArgumentParser, Namespace
from getpass import getuser
from os import path as P
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

import benchmark
import pytest
from benchmark.tools import chunk, init_logging

from hailtop import batch as hb
from hailtop.batch import Batch, Resource
from hailtop.batch.job import Job
from hailtop.utils import sync_check_exec


class CollectBenchmarks:
    def __init__(self):
        self.items = []

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        self.items = [
            (item.location[0], item.name, marker.kwargs['batch_jobs'])
            for item in items
            if (marker := item.get_closest_marker('benchmark'))
        ]


# https://github.com/pytest-dev/pytest/discussions/2039
def list_benchmarks(include: str, exclude: str) -> List[Tuple[Path, str, int]]:
    collect = CollectBenchmarks()
    directory = Path(benchmark.__file__).parent.parent
    pytest.main(
        [
            '-qq',
            '--co',
            directory,
            '-m',
            'benchmark',
            *(['-k', include] if include is not None else []),
            *(['--ignore', exclude] if exclude is not None else []),
        ],
        plugins=[collect],
    )
    return collect.items


def run_list_benchmarks(args: Namespace) -> None:
    print(sep='\n', *[f'{path}::{name}' for path, name, _ in list_benchmarks(args.include, args.exclude)])


def build_and_push_benchmark_image(hail_dev_image: str, artifact_uri: str, tag: str) -> str:
    image_name = f'{artifact_uri}:{tag}'
    with NamedTemporaryFile(mode='w+', encoding='utf-8', dir=P.curdir, prefix='Dockerfile.benchmark.') as df:
        df.write(f"""\
# syntax=docker/dockerfile:1.7-labs
FROM --platform=linux/amd64 {hail_dev_image}

COPY python/pytest.ini .
COPY --exclude=**/__pycache__ --exclude=**/.pytest_cache python/benchmark benchmark
COPY --exclude=**/__pycache__ --exclude=**/.pytest_cache python/test test
""")
        df.flush()
        df.file.close()

        sync_check_exec('docker', 'build', '.', '-f', df.name, '-t', image_name)
        sync_check_exec('docker', 'push', image_name)
        print(f'built and pushed image {image_name}')
        return image_name


def make_test_storage_permissions_job(b: Batch, object_prefix: str, labelled_sha: str) -> Job:
    test_permissions = b.new_job('test permissions')
    permissions_test_file = P.join(object_prefix, f'{labelled_sha}-permissions-test')
    test_permissions.command(f'echo hello world > {test_permissions.permissions_test_file}')
    b.write_output(test_permissions.permissions_test_file, permissions_test_file)
    return test_permissions


def make_benchmark_trial(
    b: Batch,
    env: Dict[str, Optional[str]],
    path: Path,
    benchmark_name: str,
    trial: int,
    iterations: Optional[int],
    deps: List[Job],  # dont reformat
) -> Job:
    j = b.new_job(name=f'{path}::{benchmark_name}-{trial}')
    j.depends_on(*deps)
    j.always_copy_output()

    for varname, val in env.items():
        if val is not None:
            j.env(varname, val)

    j.env('BENCHMARK_TRIAL_ID', str(trial))

    # If the benchmarks fail, we always want this file to exist otherwise
    # later combine jobs will fail when localising.
    j.command(f'touch {j.ofile}')
    j.command('mkdir -p benchmark-resources')
    j.command(
        ' '.join([
            f'python3 -m pytest {path}',
            '-Werror:::hail -Werror:::hailtop -Werror::ResourceWarning',
            '--log-cli-level=ERROR',
            '-s',
            '-r A',
            '-vv',
            '--instafail',
            '--durations=50',
            '--timeout=1800',
            # pytest keeps 3 test sessions worth of temp files by default.
            # some benchmarks generate very large output files which can quickly
            # fill the tmpfs and so we'll make pytest always delete tmp files
            # immediately when tmp_path fixtures are torn-down.
            '--override-ini=tmp_path_retention_count=0',
            '--override-ini=tmp_path_retention_policy=failed',
            f'--output={j.ofile}',
            '--data-dir=benchmark-resources',
            f'--iterations={iterations}' if iterations is not None else '',
            f'-k {benchmark_name}',
        ])
    )

    return j.ofile


def read_jsonl(p: Path):
    logging.info(f'reading json lines from {p}.')
    with p.open(encoding='utf-8') as r:
        for line in r:
            if len(line) > 1:
                yield json.loads(line)


def combine(output: Resource, files: List[Resource]) -> None:
    init_logging()
    n_files = len(files)
    if n_files < 1:
        raise ValueError("'combine' requires at least 1 file to merge")

    logging.info(f'combining {len(files)} files')

    jsonl = [line for f in files for line in read_jsonl(Path(f))]
    jsonl.sort(key=lambda r: (r['path'], r['name'], r['trial']))

    logging.info(f'Writing combine output to {output}')
    logging.info(f'collected {len(jsonl)} benchmark jobs.')
    with open(output, encoding='utf-8', mode='w+') as out:
        for line in jsonl:
            json.dump(line, out)
            out.write('\n')


def run_combine(args: Namespace) -> None:
    combine(args.output, args.files)


def make_combine_job(b: Batch, context: str, files: List[Resource]) -> Resource:
    j = b.new_python_job(f'combine_output_{context}')
    j.call(combine, j.ofile, files)
    j.always_run()
    return j.ofile


def combine_results_as_tree(b: Batch, branch_factor: int, results: List[Resource]) -> Resource:
    phase_i = 1

    while len(results) > 1:
        results = [
            (make_combine_job(b, f'phase{phase_i}_job{job_id}', files) if len(files) > 1 else files[0])
            for job_id, files in enumerate(chunk(branch_factor, results), start=1)
        ]
        phase_i += 1

    return results[0]


def run_submit(args: Namespace) -> None:
    timestamp = time.strftime('%Y-%m-%d')
    labelled_sha = args.sha + (f'-{args.label}' if args.label is not None else '')
    object_prefix = getattr(args, 'object-prefix')
    output_file = P.join(object_prefix, f'{timestamp}-{labelled_sha}.jsonl')

    all_benchmarks = [
        (path, name, trial)
        for path, name, num_jobs in list_benchmarks(args.include, args.exclude)
        for trial in range(args.jobs or num_jobs)
    ]

    assert len(all_benchmarks) > 0

    image = build_and_push_benchmark_image(args.image, args.artifact_uri, labelled_sha)

    b = Batch(
        name=f'benchmark-{labelled_sha}',
        backend=hb.ServiceBackend(),
        default_image=image,
        default_python_image=image,
        default_cpu='2',
        default_storage='30G',
        attributes={
            'output_file': output_file,
            'sha': args.sha,
            'image': image,
        },
    )

    test_permissions = make_test_storage_permissions_job(b, object_prefix, labelled_sha)

    print(f'generating {len(all_benchmarks)} individual benchmark tasks')

    envvars = {
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'HAIL_DEV_LOWER': '1' if args.lower else None,
        'HAIL_DEV_LOWER_ONLY': '1' if args.lower_only else None,
        'PYSPARK_SUBMIT_ARGS': '--driver-memory 6G pyspark-shell',
        'TMPDIR': '/io/tmp',
    }

    random.shuffle(all_benchmarks)
    result = combine_results_as_tree(
        b,
        args.branch_factor,
        [
            make_benchmark_trial(b, envvars, path, name, trial, args.iterations, deps=[test_permissions])
            for path, name, trial in all_benchmarks
        ],
    )

    print(f'writing output to {output_file}')
    b.write_output(result, output_file)

    if args.wait:
        b.run()
    else:
        b = b.run(wait=False)
        print(f"Submitted  batch '{b.id}'.")


def nonempty(s: str):
    if not s:
        raise ValueError('must be non-empty')
    else:
        return s


if __name__ == '__main__':
    parser = ArgumentParser()
    subparser = parser.add_subparsers(title='commands')

    list_p = subparser.add_parser(
        'list',
        description='List known hail benchmarks',
    )
    list_p.add_argument('--include', type=nonempty, help="see pytest -k", default=None)
    list_p.add_argument('--exclude', type=nonempty, help='see pytest --ignore', default=None)
    list_p.set_defaults(main=run_list_benchmarks)

    combine_p = subparser.add_parser(
        'combine',
        description='Combine parallelized benchmark metrics.',
    )
    combine_p.add_argument("files", type=nonempty, nargs='+', help="JSONL files to combine.")
    combine_p.add_argument("--output", "-o", type=nonempty, help="Output file.", default='out.jsonl')
    combine_p.set_defaults(main=run_combine)

    submit_p = subparser.add_parser('submit', description='Submit hail benchmarks to the batch service')
    submit_p.add_argument('image', type=nonempty, help='hail dev docker image url')
    submit_p.add_argument('object-prefix', type=nonempty, help='GCS object prefix for results json')
    submit_p.add_argument('sha', type=nonempty, help='SHA-1 object name, possibly abbreviated.')
    submit_p.add_argument(
        '--artifact-uri',
        type=nonempty,
        help='GCS Artifact Repository URI to upload benchmark image',
        default=f'us-docker.pkg.dev/broad-ctsa/hail-benchmarks/{getuser()}',
    )
    submit_p.add_argument('--label', type=nonempty, help='batch job label', default=None)
    submit_p.add_argument('--branch-factor', type=int, help='number of benchmarks to combine at a time', default=32)
    submit_p.add_argument(
        '--iterations', type=int, help='override number of iterations for each benchmark', default=None
    )
    submit_p.add_argument(
        '--jobs', type=int, help='override number of batch jobs created for each benchmark', default=None
    )
    submit_p.add_argument('--include', type=nonempty, help="see pytest -k", default=None)
    submit_p.add_argument('--exclude', type=nonempty, help='see pytest --ignore', default=None)
    submit_p.add_argument('--lower', action='store_true')
    submit_p.add_argument('--lower-only', action='store_true')
    submit_p.add_argument('--wait', action='store_true', help='wait for batch to complete')
    submit_p.set_defaults(main=run_submit)

    args = parser.parse_args()
    args.main(args)
