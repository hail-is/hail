import hail as hl

from ..helpers import skip_unless_service_backend
from hail.backend.service_backend import ServiceBackend

@skip_unless_service_backend()
def test_tiny_driver_has_tiny_memory():
    try:
        hl.utils.range_table(100_000_000, 50).to_pandas()
    except Exception as exc:
        # Sometimes the JVM properly OOMs, sometimes it just dies.
        assert (
            'java.lang.OutOfMemoryError: Java heap space' in exc.args[0] or
            'batch.worker.jvm_entryway_protocol.EndOfStream' in exc.args[0]
        )
    else:
        assert False

@skip_unless_service_backend()
def test_big_driver_has_big_memory():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    old_driver_cores = backend.driver_cores
    old_driver_memory = backend.driver_memory
    try:
        backend.driver_cores = 8
        backend.driver_memory = 'highmem'
        t = hl.utils.range_table(100_000_000, 50)
        # The pytest (client-side) worker dies if we try to realize all 100M rows in memory.
        # Instead, we realize the 100M rows in memory on the driver and then take just the first 10M
        # rows back to the client.
        hl.eval(t.aggregate(hl.agg.collect(t.idx), _localize=False)[:10_000_000])
    finally:
        backend.driver_cores = old_driver_cores
        backend.driver_memory = old_driver_memory

@skip_unless_service_backend()
def test_tiny_worker_has_tiny_memory():
    try:
        t = hl.utils.range_table(2, n_partitions=2).annotate(nd=hl.nd.ones((30_000, 30_000)))
        t = t.annotate(nd_sum=t.nd.sum())
        t.aggregate(hl.agg.sum(t.nd_sum))
    except Exception as exc:
        assert 'HailException: Hail off-heap memory exceeded maximum threshold' in exc.args[0]
    else:
        assert False

@skip_unless_service_backend()
def test_big_worker_has_big_memory():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    old_driver_cores = backend.driver_cores
    old_driver_memory = backend.driver_memory
    try:
        backend.worker_cores = 8
        backend.worker_memory = 'highmem'
        t = hl.utils.range_table(2, n_partitions=2).annotate(nd=hl.nd.ones((30_000, 30_000)))
        t = t.annotate(nd_sum=t.nd.sum())
        # We only eval the small thing so that we trigger an OOM on the worker
        # but not the driver or client
        t.aggregate(hl.agg.sum(t.nd_sum))
    finally:
        backend.driver_cores = old_driver_cores
        backend.driver_memory = old_driver_memory
