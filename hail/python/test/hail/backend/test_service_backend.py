import os

import hail as hl

from ..helpers import skip_unless_service_backend, test_timeout
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
@test_timeout(batch=6 * 60)
def test_big_driver_has_big_memory():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    t = hl.utils.range_table(100_000_000, 50)
    # The pytest (client-side) worker dies if we try to realize all 100M rows in memory.
    # Instead, we realize the 100M rows in memory on the driver and then take just the first 10M
    # rows back to the client.
    hl.eval(
        t.aggregate(hl.agg.collect(t.idx), _localize=False)[:10_000_000],
        _execute_kwargs={'driver_cores': 8, 'driver_memory': 'highmem'}
    )


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
@test_timeout(batch=10 * 60)
def test_big_worker_has_big_memory():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    t = hl.utils.range_table(2, n_partitions=2).annotate(nd=hl.nd.ones((30_000, 30_000)))
    t = t.annotate(nd_sum=t.nd.sum())
    # We only eval the small thing so that we trigger an OOM on the worker
    # but not the driver or client
    hl.eval(
        t.aggregate(hl.agg.sum(t.nd_sum), _localize=False),
        _execute_kwargs={'worker_cores': 8, 'worker_memory': 'highmem'}
    )


@skip_unless_service_backend()
@test_timeout(batch=12 * 60)
def test_regions():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    old_regions = backend.regions
    CLOUD = os.environ['HAIL_CLOUD']
    try:
        if CLOUD == 'gcp':
            backend.regions = ['us-east1']
        else:
            assert CLOUD == 'azure'
            backend.regions = ['eastus']
        hl.utils.range_table(1, 1).to_pandas()
    finally:
        backend.regions = old_regions
