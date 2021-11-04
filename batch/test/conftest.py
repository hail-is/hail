import hashlib
import os
import logging
import pytest

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def log_before_after():
    log.info('starting test')
    yield
    log.info('ending test')


def pytest_collection_modifyitems(config, items):
    n_splits = int(os.environ.get('PYTEST_SPLITS', '1'))
    split_index = int(os.environ.get('PYTEST_SPLIT_INDEX', '-1'))
    if n_splits <= 1:
        return
    if not (0 <= split_index < n_splits):
        raise RuntimeError(f"invalid split_index: index={split_index}, n_splits={n_splits}\n  env={os.environ}")
    skip_this = pytest.mark.skip(reason="skipped in this round")

    def digest(s):
        return int.from_bytes(hashlib.md5(str(s).encode('utf-8')).digest(), 'little')

    for item in items:
        if not digest(item.name) % n_splits == split_index:
            item.add_marker(skip_this)
