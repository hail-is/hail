import os

import hail as hl
from hail.backend.service_backend import ServiceBackend
from hailtop.batch_client.client import Batch

from ..helpers import qobtest, skip_unless_service_backend, test_timeout

import pytest

@qobtest
@skip_unless_service_backend()
def test_big_worker_has_big_memory(mocker):
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)
    backend.worker_cores = 8
    backend.worker_memory = 'highmem'
    t = hl.utils.range_table(2, n_partitions=2).annotate(nd=hl.nd.ones((30_000, 30_000)))
    t = t.annotate(nd_sum=t.nd.sum())
    spy = mocker.spy(backend, '_run_on_batch')
    spy.return_value = ('','')
    hl.eval(t.aggregate(hl.agg.sum(t.nd_sum), _localize=False))


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
