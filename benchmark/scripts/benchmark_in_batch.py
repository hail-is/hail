#!/usr/bin/env python3

import random
import re
import sys
import time

from benchmark_hail.run.resources import all_resources
from benchmark_hail.run.utils import list_benchmarks
from hailtop import batch as hb
from os import (getenv, path as P)


CMDS = ['start', 'run']
CMDS_STR = f'[{ "|".join(CMDS) }]'

USAGE = ' '.join([
    f'usage: ./{P.basename(sys.argv[0])}',
    'DOCKER_IMAGE_URL',
    'BUCKET_BASE',
    'SHA',
    'N_REPLICATES',
    'N_ITERS',
    CMDS_STR
])

BILLING_PROJECT = getenv('HAIL_BATCH_BILLING_PROJECT', 'benchmark')
LABEL = getenv('BENCHMARK_LABEL')
REGEX_INCLUDE = getenv('BENCHMARK_REGEX_INCLUDE')
REGEX_EXCLUDE = getenv('BENCHMARK_REGEX_EXCLUDE')
LOWER = getenv('BENCHMARK_LOWER')
LOWER_ONLY = getenv('BENCHMARK_LOWER_ONLY')
BRANCH_FACTOR = int(getenv('BENCHMARK_BRANCH_FACTOR', 32))


def gen_benchmark_replicates(nreplicates, all_benchmarks):
    include = (
        (lambda t: re.match(REGEX_INCLUDE, t) is not None) if REGEX_INCLUDE
        else lambda t: True
    )

    exclude = (
        (lambda t: re.match(REGEX_EXCLUDE, t) is not None) if REGEX_EXCLUDE
        else lambda t: False
    )

    tasks = [
        (benchmark.name, replicate, benchmark.groups)
        for benchmark in all_benchmarks
        if include(benchmark.name) and not exclude(benchmark.name)
        for replicate in range(nreplicates)
    ]

    random.shuffle(tasks)

    return tasks


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise RuntimeError(USAGE)

    BENCHMARK_IMAGE = sys.argv[1]
    BUCKET_BASE = sys.argv[2]
    SHA = sys.argv[3]
    N_REPLICATES = int(sys.argv[4])
    N_ITERS = int(sys.argv[5])
    CMD = sys.argv[6]

    if not CMD in CMDS:
        raise RuntimeError(f'unknown comamnd: "{CMD}". choices={CMDS_STR}')

    timestamp = time.strftime('%Y-%m-%d')
    labeled_sha = SHA + ('' if LABEL is None else f'-{LABEL}')
    output_file = P.join(BUCKET_BASE, f'{timestamp}-{labeled_sha}.json')

    b = hb.Batch(
        name=f'benchmark-{labeled_sha}',
        backend=hb.ServiceBackend(billing_project=BILLING_PROJECT),
        default_image=BENCHMARK_IMAGE,
        default_cpu='2',
        default_storage='30G',
        attributes={
            'output_file': output_file,
            'n_replicates': str(N_REPLICATES),
            'n_iters': str(N_ITERS),
            'image': str(BENCHMARK_IMAGE),
        },
    )

    test_permissions = b.new_job(f'test permissions')
    permissions_test_file = P.join(BUCKET_BASE, f'permissions-test')
    test_permissions.command(f'echo hello world > {test_permissions.permissions_test_file}')
    b.write_output(test_permissions.permissions_test_file, permissions_test_file)

    resource_tasks = {}
    for r in all_resources:
        j = b.new_job(f'create_resource_{r.name()}').cpu(4)
        j.depends_on(test_permissions)
        j.command(f'hail-bench create-resources --data-dir benchmark-resources --group {r.name()}')
        j.command(f"time tar --exclude='*.crc' -cf {r.name()}.tar benchmark-resources/{r.name()}")
        j.command(f'ls -lh {r.name()}.tar')
        j.command(f'mv {r.name()}.tar {j.ofile}')
        resource_tasks[r] = j


    all_benchmarks = list_benchmarks()
    assert len(all_benchmarks) > 0
    tasks = gen_benchmark_replicates(N_REPLICATES, all_benchmarks)

    print(
        f'generating {len(tasks)} * {N_REPLICATES} = {len(tasks) * N_REPLICATES} individual benchmark tasks'
    )

    all_output = []
    for name, replicate, groups in tasks:
        j = b.new_job(name=f'{name}_{replicate}')
        j.command('mkdir -p benchmark-resources')

        for resource_group in groups:
            resource_task = resource_tasks[resource_group]
            j.command(f'mv {resource_task.ofile} benchmark-resources/{resource_group.name()}.tar')
            j.command(f'time tar -xf benchmark-resources/{resource_group.name()}.tar')

        j.command(
            f'MKL_NUM_THREADS=1 '
            f'OPENBLAS_NUM_THREADS=1 '
            f'OMP_NUM_THREADS=1 '
            f'VECLIB_MAXIMUM_THREADS=1 '
            f'{"HAIL_DEV_LOWER=1 " if LOWER else ""}'
            f'{"HAIL_DEV_LOWER_ONLY=1 " if LOWER_ONLY else ""}'
            f'PYSPARK_SUBMIT_ARGS="--driver-memory 6G pyspark-shell" '
            f'TMPDIR="/io/tmp" '
            f'hail-bench run -o {j.ofile} -n {N_ITERS} --data-dir benchmark-resources -t {name}'
        )

        all_output.append(j.ofile)

    phase_i = 1
    while len(all_output) > BRANCH_FACTOR:
        new_output = []

        job_i = 1
        i = 0
        while i < len(all_output):
            combine = b.new_job(f'combine_output_phase{phase_i}_job{job_i}')
            combine.command(
                f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output[i : i + BRANCH_FACTOR])
            )
            new_output.append(combine.ofile)
            i += BRANCH_FACTOR
            job_i += 1

        phase_i += 1
        all_output = new_output

    combine = b.new_job('final_combine_output')
    combine.command(f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output))
    combine.command(f'cat {combine.ofile}')

    print(f'writing output to {output_file}')

    b.write_output(combine.ofile, output_file)
    b.run(wait=CMD=='run')
