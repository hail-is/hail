import hashlib
import logging
import os
import sys

import pytest
import pytest_asyncio

from hailtop.batch_client import aioclient
from hailtop.batch_client.client import BatchClient
from hailtop.config import get_remote_tmpdir

# The root that build.yaml, ci/ and batch/sql are read from when applying migrations. In CI the tests are
# mounted away from the rest of the repo, so the test step sets this explicitly.
_REPO_ROOT = os.environ.get(
    'HAIL_TEST_REPO_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_TEST_DB_NAME = 'test_batch_db'

log = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope='session')
async def db():
    """Spin up a real migrated batch DB and yield it; drop it on teardown."""
    # Imported here rather than at module level: other test steps share this conftest but run in images
    # without these packages.
    import warnings as _warnings  # pylint: disable=import-outside-toplevel

    import aiomysql  # pylint: disable=import-outside-toplevel

    from gear import Database  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, os.path.join(_REPO_ROOT, 'ci'))
    from create_local_database import async_main  # pylint: disable=import-outside-toplevel

    conn = await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')
    try:
        async with conn.cursor() as cur:
            # Both can warn (a missing database; the variable is deprecated in later 8.0 releases), and pytest.ini turns
            # warnings into errors.
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                await cur.execute(f'DROP DATABASE IF EXISTS `{_TEST_DB_NAME}`')
                await cur.execute('SET GLOBAL log_bin_trust_function_creators = 1')
        await conn.commit()
    finally:
        conn.close()

    orig_dir = os.getcwd()
    os.chdir(_REPO_ROOT)
    os.environ.pop('HAIL_SQL_DATABASE', None)
    try:
        await async_main('batch', _TEST_DB_NAME)
    finally:
        os.chdir(orig_dir)
    # async_main points HAIL_SQL_DATABASE at the new database as a side effect; set it here rather than rely on that.
    os.environ['HAIL_SQL_DATABASE'] = _TEST_DB_NAME
    database = Database()
    await database.async_init()
    yield database
    await database.async_close()

    conn = await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')
    try:
        async with conn.cursor() as cur:
            await cur.execute(f'DROP DATABASE IF EXISTS `{_TEST_DB_NAME}`')
        await conn.commit()
    finally:
        conn.close()


NOISE_BATCH_N_JOBS = 2000


@pytest_asyncio.fixture(scope='session')
async def noise_batch(db):
    """A large batch seeded once per session, so an unscoped read or full scan in a query under test shows up
    in its row-read count and makes MySQL's plans realistic. Tables are analyzed after seeding."""
    from .db_seed import (  # pylint: disable=import-outside-toplevel
        Attempt,
        Job,
        JobGroup,
        Update,
        analyze_tables,
        seed_batch,
    )

    states = ('Success', 'Failed', 'Error', 'Cancelled', 'Running', 'Ready', 'Pending')
    finished = ('Success', 'Failed', 'Error', 'Cancelled')
    groups = [JobGroup(), JobGroup(parent_id=1), JobGroup()]
    jobs = []
    for i in range(NOISE_BATCH_N_JOBS):
        job_id = i + 1
        state = states[i % len(states)]
        # Consistent dependencies: a Pending job waits on the Ready job before it; every fifth other job depends
        # on the most recent finished job.
        if state == 'Pending':
            parent_ids = [job_id - 1]
        elif i % 5 == 0:
            parent_ids = [p for p in range(job_id - 1, max(job_id - 8, 0), -1) if states[(p - 1) % 7] in finished][:1]
        else:
            parent_ids = []
        attempts = None
        if state == 'Success' and i % 3 == 0:
            attempts = [
                Attempt('pre-1', start_time=1_000 * job_id, end_time=1_000 * job_id + 500, reason='preempted'),
                Attempt('att-2', start_time=1_000 * job_id + 600, end_time=1_000 * job_id + 900, reason='completed'),
            ]
        jobs.append(
            Job(
                state=state,
                job_group_id=i % 4,
                attributes={'shard': str(i % 10)},
                parent_ids=parent_ids,
                attempts=attempts,
            )
        )
    # plus a pending update partly uploaded, so batch_updates and staging have cross-batch rows to (not) read
    pending = Update(jobs=[Job(state='Pending') for _ in range(200)], committed=False, n_reserved_jobs=300)
    seeded = await seed_batch(db, [Update(jobs=jobs, job_groups=groups), pending])
    await analyze_tables(db)
    return seeded


@pytest.fixture(autouse=True)
def log_before_after():
    log.info('starting test')
    yield
    log.info('ending test')


@pytest.fixture
def client():
    client = BatchClient('test')
    yield client
    client.close()


@pytest.fixture
async def async_client():
    client = await aioclient.BatchClient.create('test')
    yield client
    await client.close()


@pytest.fixture(scope='module')
def remote_tmpdir():
    return get_remote_tmpdir('batch_tests')


def pytest_collection_modifyitems(config, items):  # pylint: disable=unused-argument
    n_splits = int(os.environ.get('HAIL_RUN_IMAGE_SPLITS', '1'))
    split_index = int(os.environ.get('HAIL_RUN_IMAGE_SPLIT_INDEX', '-1'))
    if n_splits <= 1:
        return
    if not 0 <= split_index < n_splits:
        raise RuntimeError(f"invalid split_index: index={split_index}, n_splits={n_splits}\n  env={os.environ}")
    skip_this = pytest.mark.skip(reason="skipped in this round")

    def digest(s):
        return int.from_bytes(hashlib.md5(str(s).encode('utf-8')).digest(), 'little')

    for item in items:
        if not digest(item.name) % n_splits == split_index:
            item.add_marker(skip_this)
