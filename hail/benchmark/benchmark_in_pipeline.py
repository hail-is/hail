import os
import sys

from hailtop import pipeline as pl
from hailtop.hailctl.dev.benchmark.run.utils import list_benchmarks

if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise RuntimeError(f'usage: <script.py> DOCKER_IMAGE_URL BUCKET_BASE SHA N_REPLICATES N_ITERS')
    BENCHMARK_IMAGE = sys.argv[1]
    BUCKET_BASE = sys.argv[2]
    SHA = sys.argv[3]
    N_REPLICATES = int(sys.argv[4])
    N_ITERS = int(sys.argv[5])

    p = pl.Pipeline(name='benchmark',
                    backend=pl.BatchBackend(),
                    default_image=BENCHMARK_IMAGE,
                    default_storage='10G',
                    default_memory='7G',
                    default_cpu=2)

    make_resources = p.new_task('create_resources').cpu(4)
    make_resources.command('hailctl dev benchmark create-resources --data-dir benchmark-resources')
    make_resources.command("time tar -czf benchmark-resources.tar.gz benchmark-resources --exclude='*.crc'")
    make_resources.command('ls -lh benchmark-resources.tar.gz')
    make_resources.command(f'mv benchmark-resources.tar.gz {make_resources.ofile}')

    all_benchmarks = list_benchmarks()
    assert len(all_benchmarks) > 0

    all_output = []

    print(f'generating {len(all_benchmarks)} * {N_REPLICATES} = '
          f'{len(all_benchmarks) * N_REPLICATES} individual benchmark tasks')

    for name in all_benchmarks:
        for replicate in range(N_REPLICATES):
            t = p.new_task(name=f'{name}_{replicate}')
            t.command(f'mv {make_resources.ofile} benchmark-resources.tar.gz')
            t.command('time tar -xf benchmark-resources.tar.gz')
            t.command(f'hailctl dev benchmark run '
                      f'-v -o {t.ofile} -n {N_ITERS} --data-dir benchmark-resources -t {name}')
            all_output.append(t.ofile)

    combine = p.new_task('combine_output')
    combine.command(f'hailctl dev benchmark combine -o {combine.ofile} ' + ' '.join(all_output))

    output_file = os.path.join(BUCKET_BASE, f'{SHA}.json')
    print(f'writing output to {output_file}')

    p.write_output(combine.ofile, output_file)
    p.run()
