import asyncio
import hashlib
import logging
import os
from typing import Dict

import pytest
from pytest import CollectReport, StashKey

from hail import current_backend, reset_global_randomness
from hail.backend.service_backend import ServiceBackend
from hail.utils.java import choose_backend
from hailtop.hail_event_loop import hail_event_loop

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
    def digest(s):
        return int.from_bytes(hashlib.md5(str(s).encode('utf-8')).digest(), 'little')

    use_splits = False
    n_splits = int(os.getenv('HAIL_RUN_IMAGE_SPLITS', '1'))
    split_index = int(os.getenv('HAIL_RUN_IMAGE_SPLIT_INDEX', '-1'))
    if n_splits > 1:
        if not (0 <= split_index < n_splits):
            raise RuntimeError(f"invalid split_index: index={split_index}, n_splits={n_splits}\n  env={os.environ}")

        use_splits = True

    backend = choose_backend()
    cloud = os.getenv('HAIL_CLOUD')

    skip_not_in_split = pytest.mark.skip(reason=f'not included in split index {split_index}')
    for item in items:
        if use_splits and not digest(item.name) % n_splits == split_index:
            item.add_marker(skip_not_in_split)

        backend_mark = item.get_closest_marker('backend')
        if backend_mark is not None and backend_mark.args is not None:
            if backend not in backend_mark.args:
                reason = f'current backend "{backend}" not listed in "{backend_mark.args}"'
                item.add_marker(pytest.mark.skip(reason=reason))
            elif backend == 'batch':
                item.fixturenames.insert(0, 'reinitialize_hail_for_testing')

        cloud_mark = item.get_closest_marker('backend')
        if (
            cloud is not None
            and cloud_mark is not None
            and cloud_mark.args is not None
            and cloud not in cloud_mark.args
        ):
            reason = f'current cloud "{cloud}" not listed in "{cloud_mark.args}"'
            item.add_marker(pytest.mark.skip(reason=reason))


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


@pytest.fixture
def reinitialize_hail_for_testing(init_hail, request):
    hl_stop_for_test()
    hl_init_for_test(app_name=request.node.name)
    yield
    new_backend = current_backend()
    if isinstance(new_backend, ServiceBackend) and new_backend._batch_was_submitted:
        batch = new_backend._batch
        report: Dict[str, CollectReport] = request.node.stash[test_results_key]
        if any(r.failed for r in report.values()):
            log.info(f'cancelling failed test batch {batch.id}')
            hail_event_loop().run_until_complete(batch.cancel())
