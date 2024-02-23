from typing import AsyncIterator, Tuple
import asyncio

import pytest
import os

from hailtop.batch import ServiceBackend
from hailtop.utils import secret_alnum_string
from hailtop.config import get_remote_tmpdir
from hailtop.aiotools.router_fs import RouterAsyncFS


@pytest.fixture(scope="session")
async def service_backend() -> AsyncIterator[ServiceBackend]:
    sb = ServiceBackend()
    try:
        yield sb
    finally:
        await sb.async_close()


@pytest.fixture(scope="session")
async def fs() -> AsyncIterator[RouterAsyncFS]:
    fs = RouterAsyncFS()
    try:
        yield fs
    finally:
        await fs.close()


@pytest.fixture(scope="session")
def tmpdir() -> str:
    return os.path.join(
        get_remote_tmpdir('test_batch_service_backend.py::tmpdir'),
        secret_alnum_string(5),  # create a unique URL for each split of the tests
    )


@pytest.fixture
def output_tmpdir(tmpdir: str) -> str:
    return os.path.join(tmpdir, 'output', secret_alnum_string(5))


@pytest.fixture
def output_bucket_path(fs: RouterAsyncFS, output_tmpdir: str) -> Tuple[str, str, str]:
    url = fs.parse_url(output_tmpdir)
    bucket = '/'.join(url.bucket_parts)
    path = url.path
    path = '/' + os.path.join(bucket, path)
    return bucket, path, output_tmpdir


@pytest.fixture(scope="session")
async def upload_test_files(
    fs: RouterAsyncFS, tmpdir: str
) -> Tuple[Tuple[str, bytes], Tuple[str, bytes], Tuple[str, bytes]]:
    test_files = (
        (os.path.join(tmpdir, 'inputs/hello.txt'), b'hello world'),
        (os.path.join(tmpdir, 'inputs/hello spaces.txt'), b'hello'),
        (os.path.join(tmpdir, 'inputs/hello (foo) spaces.txt'), b'hello'),
    )
    await asyncio.gather(*(fs.write(url, data) for url, data in test_files))
    return test_files
