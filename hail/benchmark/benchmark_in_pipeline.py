import os
import sys

from hailtop import pipeline as pl

if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise RuntimeError(f'usage: <script.py> DOCKER_IMAGE_URL BUCKET_BASE SHA N_REPLICATES N_ITERS')
    BENCHMARK_IMAGE = sys.argv[1]
    BUCKET_BASE = sys.argv[2]
    SHA = sys.argv[3]
    N_REPLICATES = int(sys.argv[4])
    N_ITERS = int(sys.argv[5])

    p = pl.Pipeline(name='benchmark',
                    backend=pl.BatchBackend(url='https://batch.hail.is'),
                    default_image=BENCHMARK_IMAGE,
                    default_storage='5G',
                    default_memory='7G',
                    default_cpu=2)

    print(f'writing files to f{os.path.join(BUCKET_BASE, SHA, f"benchmark_{{0-{N_REPLICATES}}}.json")}')
    for i in range(N_REPLICATES):
        t = p.new_task(name=f'replicate_{i}')
        t.command(f'hailctl dev benchmark run -v -o {t.ofile} -n {N_ITERS}')
        p.write_output(t.ofile, os.path.join(BUCKET_BASE, SHA, f'benchmark_{i}.json'))
    p.run()
