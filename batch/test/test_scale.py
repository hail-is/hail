import random
import pytest
from hailtop.batch_client.client import BatchClient

from .utils import batch_status_job_counter, legacy_batch_status


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_scale(client):
    n_jobs = 10
    batch = client.create_batch()
    for idx in range(n_jobs):
        sleep_time = random.uniform(0, 30)
        batch.create_job('alpine:3.8', command=['sleep', str(round(sleep_time))])

    batch = batch.submit()
    batch.wait()
    status = legacy_batch_status(batch)

    assert batch_status_job_counter(status, 'Success') == n_jobs, status
    assert all([j['exit_code'] == 0 for j in status['jobs']])
