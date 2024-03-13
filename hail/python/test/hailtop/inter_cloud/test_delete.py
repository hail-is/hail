import asyncio
from typing import Dict, Tuple

import pytest

from hailtop.aiotools.delete import delete
from hailtop.aiotools.fs import AsyncFS

from .utils import fresh_dir


@pytest.fixture(params=['file', 'gs', 's3', 'azure-https'])
async def test_delete_one_file(request, router_filesystem: Tuple[asyncio.Semaphore, AsyncFS, Dict[str, str]]):
    sema, fs, bases = router_filesystem
    scheme = request.param
    dirname = await fresh_dir(fs, bases, scheme)

    url = f'{dirname}/file'
    await fs.write(url, b'hello world')
    assert await fs.isfile(url)
    await delete(iter([url]))
    assert not await fs.isfile(url)


@pytest.fixture(params=['file', 'gs', 's3', 'azure-https'])
async def test_delete_folder(request, router_filesystem: Tuple[asyncio.Semaphore, AsyncFS, Dict[str, str]]):
    sema, fs, bases = router_filesystem
    scheme = request.param
    dirname = await fresh_dir(fs, bases, scheme)

    url = f'{dirname}/folder'
    await asyncio.gather(
        fs.write(f'{url}/1', b'hello world'),
        fs.write(f'{url}/2', b'hello world'),
        fs.write(f'{url}/3', b'hello world'),
        fs.write(f'{url}/4', b'hello world'),
    )
    assert await fs.isdir(url)
    await delete(iter([url]))
    assert not await fs.isdir(url)
