import hashlib
import logging
import os
import sys

import aiomysql
import pytest
import pytest_asyncio

from gear import Database
from hailtop.batch_client import aioclient
from hailtop.batch_client.client import BatchClient
from hailtop.config import get_remote_tmpdir

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEST_DB_NAME = 'test_billing'

log = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope='session')
async def db():
    """Spin up a real migrated batch DB and yield it; drop it on teardown."""
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'ci'))
    import warnings as _warnings  # pylint: disable=import-outside-toplevel

    from create_local_database import async_main  # pylint: disable=import-outside-toplevel

    conn = await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')
    try:
        async with conn.cursor() as cur:
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
    database = Database()
    await database.async_init()
    yield database
    await database.async_exit_stack.aclose()

    conn = await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')
    try:
        async with conn.cursor() as cur:
            await cur.execute(f'DROP DATABASE IF EXISTS `{_TEST_DB_NAME}`')
        await conn.commit()
    finally:
        conn.close()


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
