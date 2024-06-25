import gzip

from benchmark.tools import benchmark


@benchmark(iterations=15)
def benchmark_sentinel_read_gunzip(many_ints_tsv):
    with gzip.open(many_ints_tsv) as f:
        for _ in f:
            pass


@benchmark(iterations=15)
def benchmark_sentinel_cpu_hash_1():
    x = 0
    for _ in range(10_000):
        y = 0
        for j in range(25_000):
            y = hash(y + j)
        x += y
