import os

import hail as hl
from hail.backend.service_backend import ServiceBackend
from hailtop.batch_client.client import Batch

from ..helpers import qobtest, skip_unless_service_backend, test_timeout


@qobtest
@skip_unless_service_backend()
def test_tiny_driver_has_tiny_memory():
    try:
        hl.eval(hl.range(1024 * 1024).map(lambda _: hl.range(1024 * 1024)))
    except hl.utils.FatalError as exc:
        assert "HailException: Hail off-heap memory exceeded maximum threshold: limit " in exc.args[0]
    else:
        assert False


@qobtest
@skip_unless_service_backend()
@test_timeout(batch=6 * 60)
def test_big_driver_has_big_memory():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    # A fresh backend is used for every test so this should only affect this method
    backend.driver_cores = 8
    backend.driver_memory = 'highmem'
    t = hl.utils.range_table(100_000_000, 50)
    # The pytest (client-side) worker dies if we try to realize all 100M rows in memory.
    # Instead, we realize the 100M rows in memory on the driver and then take just the first 10M
    # rows back to the client.
    hl.eval(t.aggregate(hl.agg.collect(t.idx), _localize=False)[:10_000_000])


@qobtest
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


@qobtest
@skip_unless_service_backend()
@test_timeout(batch=10 * 60)
def test_big_worker_has_big_memory():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    backend.worker_cores = 8
    backend.worker_memory = 'highmem'
    t = hl.utils.range_table(2, n_partitions=2).annotate(nd=hl.nd.ones((30_000, 30_000)))
    t = t.annotate(nd_sum=t.nd.sum())
    # We only eval the small thing so that we trigger an OOM on the worker
    # but not the driver or client
    hl.eval(t.aggregate(hl.agg.sum(t.nd_sum), _localize=False))


@qobtest
@skip_unless_service_backend()
@test_timeout(batch=24 * 60)
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


@qobtest
@skip_unless_service_backend()
def test_driver_and_worker_job_groups():
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    n_partitions = 2
    hl.utils.range_table(2, n_partitions=n_partitions)._force_count()

    sync_batch = Batch(backend._batch)
    root_job_group = sync_batch.get_job_group(0)

    action_job_groups = list(root_job_group.job_groups())
    assert len(action_job_groups) == 1
    action_jg = action_job_groups[0]
    assert action_jg.attributes()['name'] == 'execute(...)'

    action_drivers = list(action_jg.jobs(recursive=False))
    assert len(action_drivers) == 1
    driver = action_drivers[0]
    assert driver['name'] == 'execute(...)_driver'

    worker_stages = list(action_jg.job_groups())
    assert len(worker_stages) == 1
    worker_stage = worker_stages[0]
    assert worker_stage.attributes()['name'] == 'table_force_count'

    worker_jobs = list(worker_stage.jobs())
    assert len(worker_jobs) == n_partitions
    for i, partition in enumerate(worker_jobs):
        assert partition['name'] == f'execute(...)_stage0_table_force_count_job{i}'
