import asyncio
import hashlib
import os

import pytest
import pytest_timeout


from hail import current_backend, reset_global_randomness
from hail.backend.service_backend import ServiceBackend
from hail.utils.java import choose_backend
from .helpers import hl_init_for_test, hl_stop_for_test


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

    backend_name = choose_backend()

    for item in items:
        run_in_marker = [marker for marker in item.own_markers if marker.name == 'run_in']
        if not run_in_marker:
            raise RuntimeError(f'test {item.name} does not have run_in marker')
        run_in = run_in_marker[0]

        locations = run_in.args
        clouds = run_in.kwargs['clouds']
        current_cloud = os.environ.get('HAIL_CLOUD')

        def check_cloud(backend_name):
            if clouds and current_cloud not in clouds:
                item.add_marker(pytest.mark.skip(
                    reason=f'test is not configured to run with the {backend_name} in cloud {current_cloud}'))

        if backend_name == 'spark':
            if 'spark' not in locations and 'all' not in locations:
                item.add_marker(pytest.mark.skip(reason='test is not configured to run with the SparkBackend'))
            check_cloud('SparkBackend')
        elif backend_name == 'batch':
            if 'batch' not in locations and 'all' not in locations:
                item.add_marker(pytest.mark.skip(reason='test is not configured to run with the ServiceBackend'))
            check_cloud('ServiceBackend')
        else:
            assert backend_name == 'local'
            if 'local' not in locations and 'all' not in locations:
                item.add_marker(pytest.mark.skip(reason='test is not configured to run with the LocalBackend'))
            check_cloud('LocalBackend')

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
    hl_init_for_test()
    yield
    hl_stop_for_test()


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
