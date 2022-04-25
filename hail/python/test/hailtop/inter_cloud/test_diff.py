from typing import Tuple, AsyncIterator, Dict
import secrets
import os
import asyncio
import pytest
import functools

from hailtop.aiotools.fs import AsyncFS
from hailtop.aiotools.diff import diff, DiffException
from hailtop.utils import bounded_gather2
from hailtop.aiotools.router_fs import RouterAsyncFS


@pytest.fixture(scope='module')
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(params=['gs', 's3', 'hail-az'])
async def cloud_scheme(request):
    yield request.param

@pytest.fixture(scope='module')
async def router_filesystem() -> AsyncIterator[Tuple[asyncio.Semaphore, AsyncFS, Dict[str, str]]]:
    token = secrets.token_hex(16)

    async with RouterAsyncFS('file') as fs:
        file_base = f'/tmp/{token}/'
        await fs.mkdir(file_base)

        gs_bucket = os.environ['HAIL_TEST_GCS_BUCKET']
        gs_base = f'gs://{gs_bucket}/tmp/{token}/'

        s3_bucket = os.environ['HAIL_TEST_S3_BUCKET']
        s3_base = f's3://{s3_bucket}/tmp/{token}/'

        azure_account = os.environ['HAIL_TEST_AZURE_ACCOUNT']
        azure_container = os.environ['HAIL_TEST_AZURE_CONTAINER']
        azure_base = f'hail-az://{azure_account}/{azure_container}/tmp/{token}/'

        bases = {
            'file': file_base,
            'gs': gs_base,
            's3': s3_base,
            'hail-az': azure_base
        }

        sema = asyncio.Semaphore(50)
        async with sema:
            yield (sema, fs, bases)
            await bounded_gather2(sema,
                                  functools.partial(fs.rmtree, sema, file_base),
                                  functools.partial(fs.rmtree, sema, gs_base),
                                  functools.partial(fs.rmtree, sema, s3_base),
                                  functools.partial(fs.rmtree, sema, azure_base))

        assert not await fs.isdir(file_base)
        assert not await fs.isdir(gs_base)
        assert not await fs.isdir(s3_base)
        assert not await fs.isdir(azure_base)


async def fresh_dir(fs, bases, scheme):
    token = secrets.token_hex(16)
    dir = f'{bases[scheme]}{token}/'
    await fs.mkdir(dir)
    return dir


@pytest.fixture(params=['file/file', 'file/gs', 'file/s3', 'file/hail-az',
                        'gs/file', 'gs/gs', 'gs/s3', 'gs/hail-az',
                        's3/file', 's3/gs', 's3/s3', 's3/hail-az',
                        'hail-az/file', 'hail-az/gs', 'hail-az/s3', 'hail-az/hail-az'])
async def diff_test_context(request, router_filesystem: Tuple[asyncio.Semaphore, AsyncFS, Dict[str, str]]):
    sema, fs, bases = router_filesystem

    [src_scheme, dest_scheme] = request.param.split('/')

    src_base = await fresh_dir(fs, bases, src_scheme)
    dest_base = await fresh_dir(fs, bases, dest_scheme)

    await asyncio.gather(
        fs.write(f'{src_base}same', b'123'),
        fs.write(f'{dest_base}same', b'123'),
        fs.write(f'{src_base}diff', b'123'),
        fs.write(f'{dest_base}diff', b'1'),
        fs.write(f'{src_base}src-only', b'123'),
        fs.write(f'{dest_base}dest-only', b'123'),

        fs.write(f'{src_base}a/same', b'123'),
        fs.write(f'{dest_base}a/same', b'123'),
        fs.write(f'{src_base}a/diff', b'123'),
        fs.write(f'{dest_base}a/diff', b'1'),
        fs.write(f'{src_base}a/src-only', b'123'),
        fs.write(f'{dest_base}a/dest-only', b'123'),

        fs.write(f'{src_base}b/same', b'123'),
        fs.write(f'{dest_base}b/same', b'123'),
        fs.write(f'{src_base}b/diff', b'123'),
        fs.write(f'{dest_base}b/diff', b'1'),
        fs.write(f'{src_base}b/src-only', b'123'),
        fs.write(f'{dest_base}b/dest-only', b'123'),
    )


    yield sema, fs, src_base, dest_base


@pytest.mark.asyncio
async def test_diff(diff_test_context):
    sema, fs, src_base, dest_base = diff_test_context

    expected = [
        {'from': f'{src_base}diff', 'to': f'{dest_base}diff', 'from_size': 3, 'to_size': 1},
        {'from': f'{src_base}src-only', 'to': f'{dest_base}src-only', 'from_size': 3, 'to_size': None},
        {'from': f'{src_base}a/diff', 'to': f'{dest_base}a/diff', 'from_size': 3, 'to_size': 1},
        {'from': f'{src_base}a/src-only', 'to': f'{dest_base}a/src-only', 'from_size': 3, 'to_size': None},
        {'from': f'{src_base}b/diff', 'to': f'{dest_base}b/diff', 'from_size': 3, 'to_size': 1},
        {'from': f'{src_base}b/src-only', 'to': f'{dest_base}b/src-only', 'from_size': 3, 'to_size': None},
    ]
    actual = diff(source=src_base, target=dest_base)
    assert actual == expected

    try:
        result = diff(source=f'{src_base}doesnotexist', target=dest_base)
    except DiffException as exc:
        assert 'Source URL refers to no files or directories' in exc.args[0]
    else:
        assert False, result

    expected = [
        {'from': f'{src_base}src-only', 'to': f'{dest_base}', 'from_size': 3, 'to_size': None}
    ]
    actual = diff(source=f'{src_base}src-only', target=f'{dest_base}')
    assert actual == expected

    expected = [
        {'from': f'{src_base}diff', 'to': f'{dest_base}diff', 'from_size': 3, 'to_size': 1}
    ]
    actual = diff(source=f'{src_base}diff', target=f'{dest_base}diff')
    assert actual == expected

    expected = []
    actual = diff(source=f'{src_base}same', target=f'{dest_base}same')
    assert actual == expected
