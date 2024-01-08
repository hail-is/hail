import asyncio
import hashlib
import logging
import os

import pytest
from pytest_asyncio import is_async_test

from hailtop.config import get_remote_tmpdir

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def log_before_after():
    log.info('starting test')
    yield
    log.info('ending test')


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

    session_scope_marker = pytest.mark.asyncio(scope="session")
    for item in items:
        if not digest(item.name) % n_splits == split_index:
            item.add_marker(skip_this)
        if is_async_test(item):
            item.add_marker(session_scope_marker)
