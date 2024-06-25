import asyncio
from typing import Dict, Tuple

import pytest

from hailtop.aiotools.diff import DiffException, diff
from hailtop.aiotools.fs import AsyncFS
from hailtop.frozendict import frozendict

from .utils import fresh_dir


@pytest.fixture(
    params=[
        'file/file',
        'file/gs',
        'file/s3',
        'file/azure-https',
        'gs/file',
        'gs/gs',
        'gs/s3',
        'gs/azure-https',
        's3/file',
        's3/gs',
        's3/s3',
        's3/azure-https',
        'azure-https/file',
        'azure-https/gs',
        'azure-https/s3',
        'azure-https/azure-https',
    ]
)
async def diff_test_context(request, router_filesystem: Tuple[asyncio.Semaphore, AsyncFS, Dict[str, str]]):
    sema, fs, bases = router_filesystem

    [src_scheme, dest_scheme] = request.param.split('/')

    src_base = await fresh_dir(fs, bases, src_scheme)
    dest_base = await fresh_dir(fs, bases, dest_scheme)

    await asyncio.gather(*[fs.mkdir(x) for x in [f'{src_base}a/', f'{src_base}b/', f'{dest_base}a/', f'{dest_base}b/']])

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


async def test_diff(diff_test_context):
    sema, fs, src_base, dest_base = diff_test_context

    expected = {
        frozendict({'from': f'{src_base}diff', 'to': f'{dest_base}diff', 'from_size': 3, 'to_size': 1}),
        frozendict({'from': f'{src_base}src-only', 'to': f'{dest_base}src-only', 'from_size': 3, 'to_size': None}),
        frozendict({'from': f'{src_base}a/diff', 'to': f'{dest_base}a/diff', 'from_size': 3, 'to_size': 1}),
        frozendict({'from': f'{src_base}a/src-only', 'to': f'{dest_base}a/src-only', 'from_size': 3, 'to_size': None}),
        frozendict({'from': f'{src_base}b/diff', 'to': f'{dest_base}b/diff', 'from_size': 3, 'to_size': 1}),
        frozendict({'from': f'{src_base}b/src-only', 'to': f'{dest_base}b/src-only', 'from_size': 3, 'to_size': None}),
    }
    actual = await diff(source=src_base, target=dest_base)
    actual_set = set(frozendict(x) for x in actual)
    assert actual_set == expected, str((actual, expected))

    try:
        result = await diff(source=f'{src_base}doesnotexist', target=dest_base)
    except DiffException as exc:
        assert 'Source URL refers to no files or directories' in exc.args[0]
    else:
        assert False, result

    expected = [{'from': f'{src_base}src-only', 'to': f'{dest_base}', 'from_size': 3, 'to_size': None}]
    actual = await diff(source=f'{src_base}src-only', target=f'{dest_base}')
    assert actual == expected

    expected = [{'from': f'{src_base}diff', 'to': f'{dest_base}diff', 'from_size': 3, 'to_size': 1}]
    actual = await diff(source=f'{src_base}diff', target=f'{dest_base}diff')
    assert actual == expected

    expected = []
    actual = await diff(source=f'{src_base}same', target=f'{dest_base}same')
    assert actual == expected
