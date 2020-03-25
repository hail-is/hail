import os
import random
import sys

from benchmark_hail.run.resources import all_resources
from benchmark_hail.run.utils import list_benchmarks
from hailtop import pipeline as pl

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
    if label:
        labeled_sha = f'{labeled_sha}-{label}'
    p = pl.Pipeline(name=f'benchmark-{labeled_sha}',
                    backend=pl.BatchBackend(billing_project='hail'),
                    default_image=BENCHMARK_IMAGE,
                    default_storage='100G',
                    default_memory='7G',
                    default_cpu=2)

    resource_tasks = {}
    for r in all_resources:
        t = p.new_task(f'create_resource_{r.name()}').cpu(4)
        t.command(f'hail-bench create-resources --data-dir benchmark-resources --group {r.name()}')
        t.command(f"time tar -cf {r.name()}.tar benchmark-resources/{r.name()} --exclude='*.crc'")
        t.command(f'ls -lh {r.name()}.tar')
        t.command(f'mv {r.name()}.tar {t.ofile}')
        resource_tasks[r] = t

    all_benchmarks = list_benchmarks()
    assert len(all_benchmarks) > 0

    all_output = []

    print(f'generating {len(all_benchmarks)} * {N_REPLICATES} = '
          f'{len(all_benchmarks) * N_REPLICATES} individual benchmark tasks')

    task_fs = []
    for benchmark in all_benchmarks:
        for replicate in range(N_REPLICATES):
            task_fs.append((benchmark.name, replicate, benchmark.groups))

    random.shuffle(task_fs)

    for name, replicate, groups in task_fs:
        t = p.new_task(name=f'{name}_{replicate}')
        t.command('mkdir -p benchmark-resources')
        for resource_group in groups:
            resource_task = resource_tasks[resource_group]
            t.command(f'mv {resource_task.ofile} benchmark-resources/{resource_group.name()}.tar')
            t.command(f'time tar -xf benchmark-resources/{resource_group.name()}.tar')
        t.command(f'MKL_NUM_THREADS=1'
                  f'OPENBLAS_NUM_THREADS=1'
                  f'OMP_NUM_THREADS=1'
                  f'VECLIB_MAXIMUM_THREADS=1'
                  f'PYSPARK_SUBMIT_ARGS="--driver-memory 6G pyspark-shell" '
                  f'hail-bench run -o {t.ofile} -n {N_ITERS} --data-dir benchmark-resources -t {name}')
        all_output.append(t.ofile)

    combine = p.new_task('combine_output')
    combine.command(f'hail-bench combine -o {combine.ofile} ' + ' '.join(all_output))
    combine.command(f'cat {combine.ofile}')

    output_file = os.path.join(BUCKET_BASE, f'{labeled_sha}.json')
    print(f'writing output to {output_file}')

    p.write_output(combine.ofile, output_file)
    p.run()
