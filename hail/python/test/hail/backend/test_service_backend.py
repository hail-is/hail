import os
import struct
from dataclasses import dataclass
from random import randint

import orjson
import pytest

import hail as hl
from hail.backend.service_backend import ServiceBackend
from hailtop.batch_client.client import Batch, Job, JobGroup


@dataclass
class BatchServiceMocks:
    create_jvm_job = None
    create_job_group = None
    submit = None
    wait = None
    driver_config = None


@pytest.fixture
def run_on_batch_mocks(mocker):
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)

    def classname(c):
        return f'{c.__module__}.{c.__qualname__}'

    m = BatchServiceMocks()

    job = mocker.patch(classname(Job))
    job.job_id = randint(0, 999_999_999)

    outfile = type('Settable', (object,), {'value': None})

    def set_outdir(*args, **kwargs):
        outfile.value = kwargs['argv'][3]
        with backend.fs.open(kwargs['argv'][2]) as f:
            m.driver_config = orjson.loads(f.read())
            return job

    batch = backend._batch
    m.create_jvm_job = mocker.patch.object(batch, 'create_jvm_job')
    m.create_jvm_job.side_effect = set_outdir

    m.create_job_group = mocker.patch.object(batch, 'create_job_group')
    m.create_job_group.return_value = mocker.patch(classname(JobGroup))

    m.submit = mocker.patch.object(batch, 'submit')
    m.submit.return_value = None

    async def write_output(*args, **kwargs):
        async with await backend._async_fs.create(outfile.value) as f:
            payload = hl.ttuple(hl.tint64)._to_encoding([0])
            await f.write(struct.pack('<b', 1))
            await f.write(struct.pack('<i', len(payload)))
            await f.write(payload)

    m.wait = mocker.patch.object(batch, 'wait')
    m.wait.side_effect = write_output
    return m


@pytest.mark.backend('batch')
def test_big_worker_has_big_memory(run_on_batch_mocks):
    backend = hl.current_backend()
    assert isinstance(backend, ServiceBackend)

    backend.worker_cores = 8
    backend.worker_memory = 'highmem'

    hl.utils.range_table(10)._force_count()

    config = run_on_batch_mocks.driver_config['config']
    assert config['worker_cores'] == '8'
    assert config['worker_memory'] == 'highmem'


@pytest.mark.backend('batch')
def test_regions(run_on_batch_mocks):
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

        hl.utils.range_table(10)._force_count()

        driver_kwargs = run_on_batch_mocks.create_jvm_job.call_args.kwargs
        assert driver_kwargs['regions'] == backend.regions

    finally:
        backend.regions = old_regions


@pytest.mark.backend('batch')
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
