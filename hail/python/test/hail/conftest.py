import asyncio
import hashlib
import logging
import os
from typing import Dict

import pytest
from pytest import CollectReport, StashKey

from hail import current_backend
from hail.backend.service_backend import ServiceBackend
from hail.utils.java import Env, choose_backend
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
        use_splits = True
        if not (0 <= split_index < n_splits):
            raise RuntimeError(f"invalid split_index: index={split_index}, n_splits={n_splits}\n  env={os.environ}")

    backend = choose_backend()
    cloud = os.getenv('HAIL_CLOUD')

    bins = [[] for _ in range(2)]

    for item in items:
        if use_splits and not digest(item.name) % n_splits == split_index:
            skip_reason = f'not included in split index {split_index} modulo {n_splits}'
            item.add_marker(pytest.mark.skip(reason=skip_reason))
            continue

        if (backend_mark := item.get_closest_marker('backend')) is not None and backend not in backend_mark.args:
            skip_reason = f'current backend "{backend}" not listed in "{backend_mark.args}"'
            item.add_marker(pytest.mark.skip(reason=skip_reason))
            continue

        if (
            (cloud_mark := item.get_closest_marker('cloud')) is not None
            and cloud is not None
            and cloud not in cloud_mark.args
        ):
            skip_reason = f'current cloud "{cloud}" not listed in "{cloud_mark.args}"'
            item.add_marker(pytest.mark.skip(reason=skip_reason))
            continue

        init_fixture_name, priority = (
            ('uninitialized', 0)
            if item.get_closest_marker('uninitialized') is not None
            else ('init_query_on_batch', 1)
            if backend == 'batch'
            else ('init_hail', 1)
        )

        item.fixturenames.insert(0, init_fixture_name)
        bins[priority].append(item)

    # Attempt to run tests that require no initialisation first
    items[:] = [item for group in bins for item in group]


@pytest.fixture(scope='function')
def uninitialized():
    assert not Env.is_fully_initialized()
    try:
        yield
    finally:
        hl_stop_for_test()


@pytest.fixture(scope='session')
def init_hail(request):
    hl_init_for_test(app_name=request.node.name)
    try:
        yield
    finally:
        hl_stop_for_test()


@pytest.fixture
def init_query_on_batch(request):
    hl_stop_for_test()
    hl_init_for_test(backend='batch', app_name=request.node.name)
    try:
        yield
    finally:
        new_backend = current_backend()
        assert isinstance(new_backend, ServiceBackend)
        batch = new_backend._batch
        report: Dict[str, CollectReport] = request.node.stash[test_results_key]
        if any(r.failed for r in report.values()):
            log.info(f'cancelling failed test batch {batch.id}')
            hail_event_loop().run_until_complete(batch.cancel())


@pytest.fixture(autouse=True)
def reset_global_randomness():
    Env.reset_global_randomness()


test_results_key = StashKey[Dict[str, CollectReport]]()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # from: https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
    report = yield
    item.stash.setdefault(test_results_key, {})[report.when] = report
    return report
