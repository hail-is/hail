from typing import AsyncIterator, Tuple
import asyncio
import inspect

import pytest
import os

from hailtop import pip_version
from hailtop.batch import Batch, ServiceBackend
from hailtop.utils import secret_alnum_string
from hailtop.config import get_remote_tmpdir
from hailtop.aiotools.router_fs import RouterAsyncFS


DOCKER_ROOT_IMAGE = os.environ.get('DOCKER_ROOT_IMAGE', 'ubuntu:22.04')
PYTHON_DILL_IMAGE = 'hailgenetics/python-dill:3.9-slim'
HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')
REQUESTER_PAYS_PROJECT = os.environ.get('GCS_REQUESTER_PAYS_PROJECT')


@pytest.fixture(scope="session")
async def backend() -> AsyncIterator[ServiceBackend]:
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


def batch(backend, **kwargs):
    name_of_test_method = inspect.stack()[1][3]
    return Batch(
        name=name_of_test_method,
        backend=backend,
        default_image=DOCKER_ROOT_IMAGE,
        attributes={'foo': 'a', 'bar': 'b'},
        **kwargs,
    )
