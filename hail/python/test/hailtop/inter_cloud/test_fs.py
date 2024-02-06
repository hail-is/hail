import asyncio
import datetime
import functools
import os
import random
import secrets
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Tuple

import pytest

from hailtop.aiocloud.aioaws import S3AsyncFS
from hailtop.aiocloud.aioazure import AzureAsyncFS
from hailtop.aiocloud.aiogoogle import GoogleStorageAsyncFS
from hailtop.aiotools import AsyncFS, IsABucketError, LocalAsyncFS, UnexpectedEOFError
from hailtop.aiotools.fs.fs import AsyncFSURL
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import bounded_gather2, retry_transient_errors, secret_alnum_string


@pytest.fixture(
    params=[
        'file',
        'gs',
        's3',
        'azure-https',
        'router/file',
        'router/gs',
        'router/s3',
        'router/azure-https',
        'sas/azure-https',
    ]
)
async def filesystem(request) -> AsyncIterator[Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]]:
    token = secret_alnum_string()

    with ThreadPoolExecutor() as thread_pool:
        fs: AsyncFS
        if request.param.startswith('router/'):
            fs = RouterAsyncFS(
                local_kwargs={'thread_pool': thread_pool},
                s3_kwargs={'thread_pool': thread_pool},
            )
        elif request.param == 'file':
            fs = LocalAsyncFS(thread_pool)
        elif request.param.endswith('gs'):
            fs = GoogleStorageAsyncFS()
        elif request.param.endswith('s3'):
            fs = S3AsyncFS(thread_pool)
        else:
            assert request.param.endswith('azure-https')
            fs = AzureAsyncFS()

        async with fs:
            if request.param.endswith('file'):
                base = fs.parse_url(f'/tmp/{token}')
            elif request.param.endswith('gs'):
                bucket = os.environ['HAIL_TEST_GCS_BUCKET']
                base = fs.parse_url(f'gs://{bucket}/tmp/{token}/')
            elif request.param.endswith('s3'):
                bucket = os.environ['HAIL_TEST_S3_BUCKET']
                base = fs.parse_url(f's3://{bucket}/tmp/{token}/')
            else:
                assert request.param.endswith('azure-https')
                account = os.environ['HAIL_TEST_AZURE_ACCOUNT']
                container = os.environ['HAIL_TEST_AZURE_CONTAINER']
                if request.param.startswith('sas'):
                    sub_id = os.environ['HAIL_TEST_AZURE_SUBID']
                    res_grp = os.environ['HAIL_TEST_AZURE_RESGRP']
                    assert isinstance(fs, AzureAsyncFS)
                    sas_token = await fs.generate_sas_token(sub_id, res_grp, account, "rwdlc")
                    base = fs.parse_url(f'https://{account}.blob.core.windows.net/{container}/tmp/{token}/?{sas_token}')
                else:
                    base = fs.parse_url(f'https://{account}.blob.core.windows.net/{container}/tmp/{token}/')

            await fs.mkdir(str(base))
            sema = asyncio.Semaphore(50)
            async with sema:
                yield (sema, fs, base)
                await fs.rmtree(sema, str(base))
            assert not await fs.isdir(str(base))


@pytest.fixture
async def local_filesystem(request):
    token = secret_alnum_string()

    with ThreadPoolExecutor() as thread_pool:
        async with LocalAsyncFS(thread_pool) as fs:
            base = f'/tmp/{token}/'
            await fs.mkdir(base)
            sema = asyncio.Semaphore(50)
            async with sema:
                yield (sema, fs, base)
                await fs.rmtree(sema, base)
            assert not await fs.isdir(base)


@pytest.fixture(params=['small', 'multipart', 'large'])
def file_data(request):
    if request.param == 'small':
        return [b'foo']
    elif request.param == 'multipart':
        return [b'foo', b'bar', b'baz']
    else:
        assert request.param == 'large'
        return [secrets.token_bytes(1_000_000)]


async def test_write_read(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL], file_data):
    (
        _,
        fs,
        base,
    ) = filesystem

    file = str(base.with_new_path_component('foo'))

    async with await fs.create(file) as writer:
        for b in file_data:
            await writer.write(b)

    expected = b''.join(file_data)
    async with await fs.open(file) as reader:
        actual = await reader.read()

    assert expected == actual


async def test_open_from(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    async with await fs.create(file) as f:
        await f.write(b'abcde')

    async with await fs.open_from(file, 2) as f:
        r = await f.read()
        assert r == b'cde'


async def test_open_from_with_length(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    async with await fs.create(file) as f:
        await f.write(b'abcde')

    async with await fs.open_from(file, 2, length=2) as f:
        r = await f.read()
        assert r == b'cd'

    async with await fs.open_from(file, 2, length=1) as f:
        r = await f.read()
        assert r == b'c'

    async with await fs.open_from(file, 2, length=0) as f:
        r = await f.read()
        assert r == b''

    try:
        await fs.open_from(str(base.with_new_path_component('foodoesnotexist')), 2, length=0)
    except FileNotFoundError:
        pass
    else:
        assert False

    try:
        await fs.open_from(str(base), 2, length=0)
    except IsADirectoryError:
        pass
    else:
        assert False


async def test_open_empty(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    async with await fs.create(file) as f:
        await f.write(b'')

    async with await fs.open_from(file, 0, length=0) as f:
        r = await f.read()
        assert r == b''


async def test_open_nonexistent_file(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))
    with pytest.raises(FileNotFoundError):
        await fs.open(file)


async def test_open_from_nonexistent_file(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))
    with pytest.raises(FileNotFoundError):
        await fs.open_from(file, 2)


async def test_read_from(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    await fs.write(file, b'abcde')
    r = await fs.read_from(file, 2)
    assert r == b'cde'


async def test_read_range(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    await fs.write(file, b'abcde')

    r = await fs.read_range(file, 2, 2)
    assert r == b'c'

    r = await fs.read_range(file, 2, 4)
    assert r == b'cde'

    try:
        await fs.read_range(file, 2, 10)
    except UnexpectedEOFError:
        pass
    else:
        assert False


async def test_read_range_end_exclusive_empty_file(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    await fs.write(file, b'')

    assert await fs.read_range(file, 0, 0, end_inclusive=False) == b''


async def test_read_range_end_inclusive_empty_file_should_error(
    filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL],
):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    await fs.write(file, b'')

    try:
        assert await fs.read_range(file, 0, 0, end_inclusive=True) == b''
    except UnexpectedEOFError:
        pass
    else:
        assert False


async def test_read_range_end_exclusive_nonempty_file(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    await fs.write(file, b'abcde')

    assert await fs.read_range(file, 2, 4, end_inclusive=False) == b'cd'


async def test_write_read_range(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL], file_data):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    async with await fs.create(file) as f:
        for b in file_data:
            await f.write(b)

    pt1 = random.randint(0, len(file_data))
    pt2 = random.randint(0, len(file_data))
    start = min(pt1, pt2)
    end = max(pt1, pt2)

    expected = b''.join(file_data)[start : end + 1]
    actual = await fs.read_range(file, start, end)  # end is inclusive

    assert expected == actual


async def test_isfile(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    # doesn't exist yet
    assert not await fs.isfile(file)

    await fs.touch(file)

    assert await fs.isfile(file)


async def test_isdir(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    # mkdir with trailing slash
    dir = str(base.with_new_path_component('dir/'))
    await fs.mkdir(dir)

    file = str(base.with_new_path_component('dir/foo'))
    await fs.touch(file)

    # can't test this until after creating foo
    assert await fs.isdir(dir)

    # mkdir without trailing slash
    dir2 = str(base.with_new_path_component('dir2'))
    await fs.mkdir(dir2)

    file2 = str(base.with_new_path_component('dir2/foo'))
    await fs.touch(file2)

    assert await fs.isdir(dir)


async def test_isdir_subdir_only(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    dir = str(base.with_new_path_component('dir/'))
    await fs.mkdir(dir)

    subdir = str(base.with_new_path_component('dir/subdir/'))
    await fs.mkdir(subdir)

    file = str(base.with_new_path_component('dir/subdir/foo'))
    await fs.touch(file)

    # can't test this until after creating foo
    assert await fs.isdir(dir)
    assert await fs.isdir(subdir)


async def test_remove(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('foo'))

    await fs.touch(file)
    assert await fs.isfile(file)

    await fs.remove(file)

    assert not await fs.isfile(file)


async def test_rmtree(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    sema, fs, base = filesystem

    dir = base.with_new_path_component('foo/')
    subdir1 = base.with_new_path_component('foo/foo/')
    subdir1subdir1 = base.with_new_path_component('foo/foo/foo/')
    subdir1subdir2 = base.with_new_path_component('foo/foo/bar/')
    subdir1subdir3 = base.with_new_path_component('foo/foo/baz/')
    subdir1subdir4_empty = base.with_new_path_component('foo/foo/qux/')
    subdir2 = base.with_new_path_component('foo/bar/')
    subdir3 = base.with_new_path_component('foo/baz/')
    subdir4_empty = base.with_new_path_component('foo/qux/')

    await fs.mkdir(str(dir))
    await fs.touch(str(dir.with_new_path_component('a')))
    await fs.touch(str(dir.with_new_path_component('b')))

    await fs.mkdir(str(subdir1))
    await fs.mkdir(str(subdir1subdir1))
    await fs.mkdir(str(subdir1subdir2))
    await fs.mkdir(str(subdir1subdir3))
    await fs.mkdir(str(subdir1subdir4_empty))
    await fs.mkdir(str(subdir2))
    await fs.mkdir(str(subdir3))
    await fs.mkdir(str(subdir4_empty))

    sema = asyncio.Semaphore(100)
    await bounded_gather2(
        sema,
        *[
            functools.partial(fs.touch, str(subdir.with_new_path_component(f'a{i:02}')))
            for subdir in [dir, subdir1, subdir2, subdir3, subdir1subdir1, subdir1subdir2, subdir1subdir3]
            for i in range(30)
        ],
    )

    assert await fs.isdir(str(dir))
    assert await fs.isdir(str(subdir1))
    assert await fs.isdir(str(subdir1subdir1))
    assert await fs.isdir(str(subdir1subdir2))
    assert await fs.isdir(str(subdir1subdir3))
    # subdir1subdir4_empty: in cloud fses, empty dirs do not exist and thus are not dirs
    assert await fs.isdir(str(subdir2))
    assert await fs.isdir(str(subdir3))
    # subdir4_empty: in cloud fses, empty dirs do not exist and thus are not dirs

    await fs.rmtree(sema, str(subdir1subdir2))

    assert await fs.isdir(str(dir))
    assert await fs.isfile(str(dir.with_new_path_component('a')))
    assert await fs.isfile(str(dir.with_new_path_component('b')))

    assert await fs.isdir(str(subdir1))
    assert await fs.isfile(str(subdir1.with_new_path_component('a00')))

    assert await fs.isdir(str(subdir1subdir1))
    assert await fs.isfile(str(subdir1subdir1.with_new_path_component('a00')))

    assert not await fs.isdir(str(subdir1subdir2))
    assert not await fs.isfile(str(subdir1subdir2.with_new_path_component('a00')))

    assert await fs.isdir(str(subdir1subdir3))
    assert await fs.isfile(str(subdir1subdir3.with_new_path_component('a00')))

    assert await fs.isdir(str(subdir2))
    assert await fs.isfile(str(subdir2.with_new_path_component('a00')))
    assert await fs.isdir(str(subdir3))
    assert await fs.isfile(str(subdir3.with_new_path_component('a00')))

    await fs.rmtree(sema, str(dir))

    assert not await fs.isdir(str(dir))


async def test_rmtree_empty_dir(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    sema, fs, base = filesystem

    dir = str(base.with_new_path_component('bar/'))

    await fs.mkdir(dir)
    await fs.rmtree(sema, dir)
    assert not await fs.isdir(dir)


async def test_cloud_rmtree_file_ending_in_slash(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    sema, fs, base = filesystem

    if isinstance(fs, LocalAsyncFS) or base.scheme in ('', 'file'):
        return

    fname = str(base.with_new_path_component('bar/'))

    async with await fs.create(fname) as f:
        await f.write(b'test_rmtree_file_ending_in_slash')
    await fs.rmtree(sema, fname)
    assert not await fs.isdir(fname)
    assert not await fs.isfile(fname)
    assert not await fs.exists(fname)


async def test_statfile_nonexistent_file(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    with pytest.raises(FileNotFoundError):
        await fs.statfile(str(base.with_new_path_component('foo')))


async def test_statfile_directory(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    await fs.mkdir(str(base.with_new_path_component('dir/')))
    await fs.touch(str(base.with_new_path_component('dir/foo')))

    with pytest.raises(FileNotFoundError):
        # statfile raises FileNotFound on directories
        await fs.statfile(str(base.with_new_path_component('dir/')))


async def test_statfile(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    n = 37
    file = str(base.with_new_path_component('bar'))
    await fs.write(file, secrets.token_bytes(n))
    status = await fs.statfile(file)
    assert await status.size() == n


async def test_statfile_creation_and_modified_time(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('bar'))
    now = datetime.datetime.utcnow()
    await fs.write(file, b'abc123')
    status = await fs.statfile(file)

    if isinstance(fs, RouterAsyncFS):
        is_local = isinstance(await fs._get_fs(file), LocalAsyncFS)
    else:
        is_local = isinstance(fs, LocalAsyncFS)

    if is_local:
        try:
            status.time_created()
        except ValueError as err:
            assert err.args[0] == 'LocalFS does not support time created.'
        else:
            assert False

        modified_time = status.time_modified()
        assert modified_time.timestamp() == pytest.approx(now.timestamp(), abs=60)
    else:
        create_time = status.time_created()
        assert create_time.timestamp() == pytest.approx(now.timestamp(), abs=60)
        modified_time = status.time_modified()
        assert modified_time == create_time


async def test_file_can_contain_url_query_delimiter(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    file = str(base.with_new_path_component('bar?baz'))
    await fs.write(file, secrets.token_bytes(10))
    assert await fs.exists(file)
    async for f in await fs.listfiles(str(base)):
        if 'bar?baz' in f.basename():
            break
    else:
        assert False, 'File bar?baz not found'


async def test_basename_is_not_path(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    await fs.write(str(base.with_new_path_component('abc123')), b'foo')
    assert (await fs.statfile(str(base.with_new_path_component('abc123')))).basename() == 'abc123'


async def test_listfiles(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    with pytest.raises(FileNotFoundError):
        await fs.listfiles(str(base.with_new_path_component('does/not/exist')))

    with pytest.raises(FileNotFoundError):
        await fs.listfiles(str(base.with_new_path_component('does/not/exist')), recursive=True)

    # create the following directory structure in base:
    # foobar
    # foo/a
    # foo/b/c
    a = str(base.with_new_path_component('foo/a'))
    b = str(base.with_new_path_component('/foo/b/'))
    c = str(base.with_new_path_component('foo/b/c'))
    await fs.touch(str(base.with_new_path_component('foobar')))
    await fs.mkdir(str(base.with_new_path_component('foo/')))
    await fs.touch(a)
    await fs.mkdir(b)
    await fs.touch(c)

    async def listfiles(dir, recursive):
        return {(await entry.url_full(), await entry.is_file()) async for entry in await fs.listfiles(dir, recursive)}

    foo = str(base.with_new_path_component('foo/'))
    assert await listfiles(foo, recursive=True) == {(a, True), (c, True)}
    assert await listfiles(foo, recursive=False) == {(a, True), (b, False)}

    # without trailing slash
    assert await listfiles(foo, recursive=True) == {(a, True), (c, True)}
    assert await listfiles(foo, recursive=False) == {(a, True), (b, False)}

    # test FileListEntry.status raises on directory
    async for entry in await fs.listfiles(foo, recursive=False):
        if await entry.is_dir():
            with pytest.raises(IsADirectoryError):
                await entry.status()
        else:
            stat = await entry.status()
            assert await stat.size() == 0


@pytest.mark.parametrize("permutation", [None, [0, 1, 2], [0, 2, 1], [1, 2, 0], [2, 1, 0]])
async def test_multi_part_create(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL], permutation):
    sema, fs, base = filesystem

    # S3 has a minimum part size (except for the last part) of 5MiB
    if base.scheme == 's3':
        min_part_size = 5 * 1024 * 1024
        part_data_size = [min_part_size, min_part_size, min_part_size]
    else:
        part_data_size = [8192, 600, 20000]
    part_data = [secrets.token_bytes(s) for s in part_data_size]

    s = 0
    part_start = []
    for b in part_data:
        part_start.append(s)
        s += len(b)

    path = str(base.with_new_path_component('a'))
    async with await fs.multi_part_create(sema, path, len(part_data)) as c:

        async def create_part(i):
            async with await c.create_part(i, part_start[i]) as f:
                await f.write(part_data[i])

        if permutation:
            # do it in a fixed order
            for i in permutation:
                await retry_transient_errors(create_part, i)
        else:
            # do in parallel
            await asyncio.gather(*[retry_transient_errors(create_part, i) for i in range(len(part_data))])

    expected = b''.join(part_data)
    async with await fs.open(path) as f:
        actual = await f.read()
    assert expected == actual


async def test_rmtree_on_symlink_to_directory():
    token = secret_alnum_string()
    with ThreadPoolExecutor() as thread_pool:
        fs = LocalAsyncFS(thread_pool)
        base = fs.parse_url(f'/tmp/{token}')
        await fs.mkdir(str(base))
        sema = asyncio.Semaphore(50)
        try:
            os.mkdir(f'/tmp/{token}/subdir')
            os.symlink(f'/tmp/{token}/subdir', f'/tmp/{token}/subdir/loop')
            await fs.rmtree(sema, str(base.with_new_path_component('/subdir')))
        finally:
            await fs.rmtree(sema, str(base))
            assert not await fs.isdir(str(base))


async def test_operations_on_a_bucket_url_is_error(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    if base.scheme in ('', 'file'):
        return

    bucket_url = str(base.with_path(''))

    with pytest.raises(IsABucketError):
        await fs.isdir(bucket_url)

    assert await fs.isfile(bucket_url) is False

    with pytest.raises(IsABucketError):
        await fs.statfile(bucket_url)

    with pytest.raises(IsABucketError):
        await fs.remove(bucket_url)

    with pytest.raises(IsABucketError):
        await fs.create(bucket_url)

    with pytest.raises(IsABucketError):
        await fs.open(bucket_url)


async def test_hfs_ls_bucket_url_not_an_error(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, fs, base = filesystem

    if base.scheme in ('', 'file'):
        return

    await fs.write(str(base.with_new_path_component('abc123')), b'foo')  # ensure the bucket is non-empty

    bucket_url = str(base.with_path(''))
    with RouterFS() as fs:
        fs.ls(bucket_url)


async def test_with_new_path_component(filesystem: Tuple[asyncio.Semaphore, AsyncFS, AsyncFSURL]):
    _, _, base = filesystem

    assert str(base.with_path('').with_new_path_component('abc')) == str(base.with_path('abc'))
    assert str(base.with_path('abc').with_new_path_component('def')) == str(base.with_path('abc/def'))

    actual = base.with_path('abc').with_new_path_component('def').with_new_path_component('ghi')
    assert str(actual) == str(base.with_path('abc/def/ghi'))
