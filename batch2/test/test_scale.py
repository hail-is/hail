import random
import pytest

from hailtop.batch_client.client import BatchClient


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def batch_status_job_counter(batch_status, job_state):
    return len([j for j in batch_status['jobs'] if j['state'] == job_state])


def batch_status_exit_codes(batch_status):
    return [j['exit_code'] for j in batch_status['jobs']]


def test_scale(client):
    n_jobs = 10
    batch = client.create_batch()
    for idx in range(n_jobs):
        sleep_time = random.uniform(0, 30)
        batch.create_job('alpine:3.8', command=['sleep', str(round(sleep_time))])

    batch = batch.submit()
    status = batch.wait()

    assert batch_status_job_counter(status, 'Success') == n_jobs, status

    exit_codes = [{'input': 0, 'main': 0, 'output': 0} for _ in range(n_jobs)]
    assert batch_status_exit_codes(status) == exit_codes, status
