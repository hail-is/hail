import asyncio
import hashlib
import os

import pytest

from hail import current_backend, init, reset_global_randomness
from hail.backend.service_backend import ServiceBackend
from .helpers import startTestHailContext, stopTestHailContext


def pytest_collection_modifyitems(config, items):
    n_splits = int(os.environ.get('HAIL_RUN_IMAGE_SPLITS', '1'))
    split_index = int(os.environ.get('HAIL_RUN_IMAGE_SPLIT_INDEX', '-1'))
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


@pytest.fixture(scope="session", autouse=True)
def ensure_event_loop_is_initialized_in_test_thread():
    try:
        asyncio.get_running_loop()
    except RuntimeError as err:
        assert err.args[0] == "no running event loop"
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture(scope="session", autouse=True)
def init_hail():
    startTestHailContext()
    yield
    stopTestHailContext()


@pytest.fixture(autouse=True)
def reset_randomness(init_hail):
    reset_global_randomness()


@pytest.fixture(autouse=True)
def set_query_name(init_hail, request):
    backend = current_backend()
    if isinstance(backend, ServiceBackend):
        backend.batch_attributes = dict(name=request.node.name)
        yield
        backend.batch_attributes = dict()
        backend._batch = None
    else:
        yield
