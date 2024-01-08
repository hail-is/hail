from typing import Dict
import asyncio
import hashlib
import os
import logging

import pytest
from pytest import StashKey, CollectReport


from hail import current_backend, init, reset_global_randomness
from hail.backend.service_backend import ServiceBackend
from hailtop.hail_event_loop import hail_event_loop
from hailtop.utils import secret_alnum_string
from .helpers import hl_init_for_test, hl_stop_for_test


log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def pytest_collection_modifyitems(items):
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
def init_hail():
    hl_init_for_test()
    yield
    hl_stop_for_test()


@pytest.fixture(autouse=True)
def reset_randomness(init_hail):
    reset_global_randomness()


test_results_key = StashKey[Dict[str, CollectReport]]()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # from: https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
    report = yield
    item.stash.setdefault(test_results_key, {})[report.when] = report
    return report


@pytest.fixture(autouse=True)
def reinitialize_hail_for_each_qob_test(init_hail, request):
    if isinstance(current_backend(), ServiceBackend):
        hl_stop_for_test()
        hl_init_for_test(app_name=request.node.name)
        new_backend = current_backend()
        assert isinstance(new_backend, ServiceBackend)
        yield
        if new_backend._batch_was_submitted:
            batch = new_backend._batch
            report: Dict[str, CollectReport] = request.node.stash[test_results_key]
            if any(r.failed for r in report.values()):
                log.info(f'cancelling failed test batch {batch.id}')
                hail_event_loop().run_until_complete(batch.cancel())
    else:
        yield
