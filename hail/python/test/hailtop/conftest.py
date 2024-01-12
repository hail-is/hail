import asyncio
import hashlib
import os

import pytest


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def pytest_collection_modifyitems(items):
    n_splits = int(os.environ.get("HAIL_RUN_IMAGE_SPLITS", "1"))
    split_index = int(os.environ.get("HAIL_RUN_IMAGE_SPLIT_INDEX", "-1"))
    if n_splits <= 1:
        return
    if not (0 <= split_index < n_splits):
        raise RuntimeError(f"invalid split_index: index={split_index}, n_splits={n_splits}\n  env={os.environ}")
    skip_this = pytest.mark.skip(reason="skipped in this round")

    def digest(s):
        return int.from_bytes(hashlib.md5(str(s).encode("utf-8")).digest(), "little")

    for item in items:
        if not digest(item.name) % n_splits == split_index:
            item.add_marker(skip_this)
