import random
import time
import humanize
import pytest
from hailtop.batch_client.client import BatchClient

from .utils import batch_status_job_counter, \
    legacy_batch_status


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


def test_scale(client):
    now = time.time()
    n_batches = 100
    batches = []
    for _ in range(n_batches):
        n_jobs = 100
        batch = client.create_batch()
        for idx in range(n_jobs):
            sleep_time = random.uniform(0, 30)
            batch.create_job('alpine:3.8', command=['sleep', str(round(sleep_time))])

        batches.append(batch.submit())

    for batch in batches:
        status = batch.wait()
        jobs = batch.jobs()

        successful_jobs = [j for j in jobs if j['state'] == 'Success']
        assert len(successful_jobs) == len(jobs), str(jobs) + '\n\n' + str(status)
        assert all([j['exit_code'] == 0 for j in jobs])
    duration = time.time() - now
    print('duration: {humanize.time(duration)} ({duration}s)')
