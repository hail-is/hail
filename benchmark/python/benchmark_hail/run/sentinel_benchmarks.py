import hail as hl

from .resources import *
from .utils import benchmark
import gzip


def read_gunzip(path):
    with gzip.open(path) as f:
        for line in f:
            pass


@benchmark(args=many_ints_table.handle('tsv'))
def sentinel_read_gunzip_1(path):
    read_gunzip(path)


@benchmark(args=many_ints_table.handle('tsv'))
def sentinel_read_gunzip_2(path):
    read_gunzip(path)


@benchmark(args=many_ints_table.handle('tsv'))
def sentinel_read_gunzip_3(path):
    read_gunzip(path)


def iter_hash(m, n):
    x = 0
    for i in range(m):
        y = 0
        for j in range(n):
            y = hash(y + j)
        x += y


@benchmark()
def sentinel_cpu_hash_1():
    iter_hash(10000, 25000)


@benchmark()
def sentinel_cpu_hash_2():
    iter_hash(10000, 25000)


@benchmark()
def sentinel_cpu_hash_3():
    iter_hash(10000, 25000)
