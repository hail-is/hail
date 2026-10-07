import hashlib
import logging
import os
import sys
import warnings

import pytest
import pytest_asyncio

from hailtop.batch_client import aioclient
from hailtop.batch_client.client import BatchClient
from hailtop.config import get_remote_tmpdir

# Where migrations are read from. CI mounts the tests away from the repo, so it sets this.
_REPO_ROOT = os.environ.get(
    'HAIL_TEST_REPO_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_TEST_DB_NAME = 'test_batch_db'

log = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope='session')
async def db():
    """Spin up a real migrated batch DB and yield it; drop it on teardown."""
    # Other test steps share this conftest in images without these packages.
    import aiomysql  # pylint: disable=import-outside-toplevel

    from gear import Database  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, os.path.join(_REPO_ROOT, 'ci'))
    from create_local_database import async_main  # pylint: disable=import-outside-toplevel

    conn = await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')
    try:
        async with conn.cursor() as cur:
            # Both can warn, and pytest.ini turns warnings into errors.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
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
    # async_main sets this too, but only as a side effect.
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
    """Another batch's rows, so a query that reads beyond its own batch shows it, and so plans are realistic."""
    from .db_seed import (  # pylint: disable=import-outside-toplevel
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
        # Pending jobs wait on an unfinished parent; others only depend on finished ones.
        if state == 'Pending':
            parent_ids = [job_id - 1]
        elif i % 5 == 0:
            parent_ids = [p for p in range(job_id - 1, max(job_id - 8, 0), -1) if states[(p - 1) % 7] in finished][:1]
        else:
            parent_ids = []
        jobs.append(
            Job(
                state=state,
                job_group_id=i % 4,
                attributes={'shard': str(i % 10)},
                parent_ids=parent_ids,
                # every third Success job was preempted once first
                n_attempts=2 if state == 'Success' and i % 3 == 0 else None,
            )
        )
    # a partly uploaded pending update, so other batches' staging rows exist too
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
