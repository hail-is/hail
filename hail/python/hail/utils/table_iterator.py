from typing import Optional
import hail as hl
from concurrent.futures import ProcessPoolExecutor


ht: Optional[hl.Table] = None


def init_process(path):
    global ht
    assert ht is None
    hl.init()
    ht = hl.read_table(path)


def read_partition(i):
    assert ht is not None
    return ht._filter_partitions([i]).collect()


def n_partitions():
    assert ht is not None
    return ht.n_partitions()


def table_iterator(path):
    with ProcessPoolExecutor(max_workers=1, initializer=init_process, initargs=(path,)) as pool:
        n_parts = pool.submit(n_partitions).result()
        if n_parts == 0:
            return

        fut = pool.submit(read_partition, 0)
        for i in range(0, n_parts):
            part = fut.result()
            if i + 1 < n_parts:
                fut = pool.submit(read_partition, i + 1)
            for row in part:
                yield row
