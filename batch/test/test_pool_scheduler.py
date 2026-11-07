import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

# The driver modules read these at import time.
for _name, _value in {
    'CLOUD': 'gcp',
    'HAIL_DEFAULT_NAMESPACE': 'default',
    'HAIL_SCOPE': 'test',
    'HAIL_DOCKER_ROOT_IMAGE': 'ubuntu:24.04',
    'HAIL_DOCKER_PREFIX': 'docker.io',
    'KUBERNETES_SERVER_URL': 'https://kubernetes.default',
    'INTERNAL_GATEWAY_IP': '10.0.0.1',
    'HAIL_BATCH_STORAGE_URI': 'gs://test-bucket',
    'HAIL_SHA': 'test-sha',
    'HAIL_QUERY_STORAGE_URI': 'gs://test-bucket',
    'HAIL_QUERY_ACCEPTABLE_JAR_SUBFOLDER': '/jars',
    'HAIL_QUERY_SPARK_ARCHIVE_SUBFOLDER': '/spark-archives',
}.items():
    os.environ.setdefault(_name, _value)

from batch.driver.instance_collection import pool as pool_module  # noqa: E402
from batch.driver.instance_collection.pool import PoolScheduler  # noqa: E402
from batch.utils import ExceededSharesCounter  # noqa: E402

REGION = 'us-central1'
INSTANCE_CORES_MCPU = 16_000


class FakeInstance:
    region = REGION
    state = 'active'

    def adjust_free_cores_in_memory(self, delta_mcpu):
        pass


class FakePool:
    name = 'standard'

    def __init__(self):
        self.all_supported_regions = [REGION]
        self.inst_coll_manager = SimpleNamespace(regions=[REGION])

    def get_instance(self, cores_mcpu, _regions):
        return FakeInstance() if cores_mcpu <= INSTANCE_CORES_MCPU else None


class InlineWorkerPool:
    async def call(self, f, *args, **kwargs):
        await f(*args, **kwargs)


def make_scheduler(job_cores_mcpu_by_user: Dict[str, int], n_jobs_per_user: int, n_prior_attempts: int = 0):
    """Returns the scheduler and a per-user count of jobs it has read from the queue."""
    n_read = {user: 0 for user in job_cores_mcpu_by_user}

    async def compute_fair_share():
        return {user: {'allocated_cores_mcpu': 1_000_000} for user in job_cores_mcpu_by_user}

    async def user_runnable_jobs(user):
        for job_id in range(1, n_jobs_per_user + 1):
            n_read[user] += 1
            yield {
                'batch_id': 1,
                'job_group_id': 0,
                'job_id': job_id,
                'user': user,
                'cores_mcpu': job_cores_mcpu_by_user[user],
                'regions_bits_rep': None,
                'n_prior_attempts': n_prior_attempts,
                'n_max_attempts': 5,
                'format_version': 1,
            }

    scheduler: Any = PoolScheduler.__new__(PoolScheduler)
    scheduler.app = {'frozen': False}
    scheduler.pool = FakePool()
    scheduler.async_worker_pool = InlineWorkerPool()
    scheduler.exceeded_shares_counter = ExceededSharesCounter()
    scheduler.compute_fair_share = compute_fair_share
    scheduler._user_runnable_jobs = user_runnable_jobs
    return scheduler, n_read


@pytest.fixture
def scheduled_users(monkeypatch):
    users: List[str] = []

    async def schedule_job(_app, record, _instance):
        users.append(record['user'])

    monkeypatch.setattr(pool_module, 'schedule_job', schedule_job)
    return users


@pytest.fixture
def errored_users(monkeypatch):
    users: List[str] = []

    async def mark_job_errored(_app, _batch_id, _job_group_id, _job_id, _attempt_id, user, _format_version, _msg):
        users.append(user)

    monkeypatch.setattr(pool_module, 'mark_job_errored', mark_job_errored)
    return users


async def test_each_user_gets_their_share_in_one_pass(scheduled_users):
    # Equal allocations give each user a share of 150 jobs per pass.
    scheduler, _ = make_scheduler({'alice': 1000, 'bob': 1000}, n_jobs_per_user=200)

    should_wait = await scheduler.schedule_loop_body()

    assert scheduled_users.count('alice') == 150
    assert scheduled_users.count('bob') == 150
    assert should_wait is False


async def test_unplaceable_jobs_are_bounded_by_share(scheduled_users):
    # alice's jobs don't fit on any instance.
    scheduler, n_read = make_scheduler({'alice': 2 * INSTANCE_CORES_MCPU, 'bob': 1000}, n_jobs_per_user=200)

    await scheduler.schedule_loop_body()

    assert n_read['alice'] == 150
    assert scheduled_users.count('bob') == 150


async def test_waits_when_nothing_could_be_placed(scheduled_users):
    # A lone user's share is 300, so alice reaches it without placing anything.
    scheduler, _ = make_scheduler({'alice': 2 * INSTANCE_CORES_MCPU}, n_jobs_per_user=400)

    should_wait = await scheduler.schedule_loop_body()

    assert scheduled_users == []
    assert should_wait is True


async def test_errored_jobs_count_towards_share(errored_users):
    # A lone user's share is 300; every job is over its attempt limit.
    scheduler, n_read = make_scheduler({'alice': 1000}, n_jobs_per_user=400, n_prior_attempts=5)

    should_wait = await scheduler.schedule_loop_body()

    assert n_read['alice'] == 300
    assert len(errored_users) == 300
    assert should_wait is False
