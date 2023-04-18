import os
import random
import re
import sys
import time

from benchmark_hail.run.resources import all_resources
from benchmark_hail.run.utils import list_benchmarks
from hailtop import batch as hb

if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise RuntimeError(f'usage: <script.py> DOCKER_IMAGE_URL BUCKET_BASE SHA N_REPLICATES N_ITERS')
    BENCHMARK_IMAGE = sys.argv[1]
    BUCKET_BASE = sys.argv[2]
    SHA = sys.argv[3]
    N_REPLICATES = int(sys.argv[4])
    N_ITERS = int(sys.argv[5])

    labeled_sha = SHA
    label = os.environ.get('BENCHMARK_LABEL')
    timestamp = time.strftime('%Y-%m-%d')
    if label:
        labeled_sha = f'{labeled_sha}-{label}'
    output_file = os.path.join(BUCKET_BASE, f'{timestamp}-{labeled_sha}.json')
    permissions_test_file = os.path.join(BUCKET_BASE, f'permissions-test')

    b = hb.Batch(
        name=f'benchmark-{labeled_sha}',
        backend=hb.ServiceBackend(billing_project='hail'),
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

    all_output = []

    task_filter_regex_include = os.environ.get('BENCHMARK_REGEX_INCLUDE')
    task_filter_regex_exclude = os.environ.get('BENCHMARK_REGEX_EXCLUDE')

    if task_filter_regex_include:
        include = lambda t: re.match(task_filter_regex_include, t) is not None
    else:
        include = lambda t: True

    if task_filter_regex_exclude:
        exclude = lambda t: re.match(task_filter_regex_exclude, t) is not None
    else:
        exclude = lambda t: False

    n_passed_filter = 0
    task_fs = []
    for benchmark in all_benchmarks:
        if include(benchmark.name):
            if not exclude(benchmark.name):
                n_passed_filter += 1
                for replicate in range(N_REPLICATES):
                    task_fs.append((benchmark.name, replicate, benchmark.groups))

    print(
        f'generating {n_passed_filter} * {N_REPLICATES} = {n_passed_filter * N_REPLICATES} individual benchmark tasks'
    )

    random.shuffle(task_fs)

    benchmark_lower_env_var = ''
    if os.environ.get('BENCHMARK_LOWER'):
        benchmark_lower_env_var = f' HAIL_DEV_LOWER="1" '
    if os.environ.get('BENCHMARK_LOWER_ONLY'):
        benchmark_lower_env_var = f'{benchmark_lower_env_var} HAIL_DEV_LOWER_ONLY="1" '

    for name, replicate, groups in task_fs:
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
            f'{benchmark_lower_env_var} '
            f'PYSPARK_SUBMIT_ARGS="--driver-memory 6G pyspark-shell" '
            f'hail-bench run -o {j.ofile} -n {N_ITERS} --data-dir benchmark-resources -t {name}'
        )
        all_output.append(j.ofile)

    combine_branch_factor = int(os.environ.get('BENCHMARK_BRANCH_FACTOR', 32))
    phase_i = 1
    while len(all_output) > combine_branch_factor:
        new_output = []

        job_i = 1
        i = 0
        while i < len(all_output):
            combine = b.new_job(f'combine_output_phase{phase_i}_job{job_i}')
            combine.command(
                f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output[i : i + combine_branch_factor])
            )
            new_output.append(combine.ofile)
            i += combine_branch_factor
            job_i += 1

        phase_i += 1
        all_output = new_output

    combine = b.new_job('final_combine_output')
    combine.command(f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output))
    combine.command(f'cat {combine.ofile}')

    print(f'writing output to {output_file}')

    b.write_output(combine.ofile, output_file)
    b.run()
