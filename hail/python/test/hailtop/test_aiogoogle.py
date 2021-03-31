import os
import secrets
import shutil
from itertools import accumulate
from concurrent.futures import ThreadPoolExecutor
import asyncio
import pytest
import concurrent
from hailtop.utils import secret_alnum_string, bounded_gather2
from hailtop.aiotools import LocalAsyncFS, RouterAsyncFS
from hailtop.aiogoogle import StorageClient, GoogleStorageAsyncFS


@pytest.fixture(params=['file', 'gs', 'router/file', 'router/gs'])
async def filesystem(request):
    token = secret_alnum_string()

    with ThreadPoolExecutor() as thread_pool:
        if request.param.startswith('router/'):
            fs = RouterAsyncFS(
                'file', [LocalAsyncFS(thread_pool), GoogleStorageAsyncFS()])
        elif request.param == 'file':
            fs = LocalAsyncFS(thread_pool)
        else:
            fs = GoogleStorageAsyncFS()
        async with fs:
            if request.param.endswith('file'):
                base = f'/tmp/{token}/'
            else:
                assert request.param.endswith('gs')
                bucket = os.environ['HAIL_TEST_BUCKET']
                base = f'gs://{bucket}/tmp/{token}/'

            await fs.mkdir(base)
            sema = asyncio.Semaphore(50)
            async with sema:
                yield (sema, fs, base)
                await fs.rmtree(sema, base)
            assert not await fs.isdir(base)


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
async def file_data(request):
    if request.param == 'small':
        return [b'foo']
    elif request.param == 'multipart':
        return [b'foo', b'bar', b'baz']
    else:
        assert request.param == 'large'
        return [secrets.token_bytes(1_000_000)]


@pytest.mark.asyncio
async def test_write_read(filesystem, file_data):
    sema, fs, base = filesystem

    file = f'{base}foo'

    async with await fs.create(file) as f:
        for b in file_data:
            await f.write(b)

    expected = b''.join(file_data)
    async with await fs.open(file) as f:
        actual = await f.read()

    assert expected == actual


@pytest.mark.asyncio
async def test_open_from(filesystem):
    sema, fs, base = filesystem

    file = f'{base}foo'

    async with await fs.create(file) as f:
        await f.write(b'abcde')

    async with await fs.open_from(file, 2) as f:
        r = await f.read()
        assert r == b'cde'


@pytest.mark.asyncio
async def test_isfile(filesystem):
    sema, fs, base = filesystem

    file = f'{base}foo'

    # doesn't exist yet
    assert not await fs.isfile(file)

    await fs.touch(file)

    assert await fs.isfile(file)


@pytest.mark.asyncio
async def test_isdir(filesystem):
    sema, fs, base = filesystem

    # mkdir with trailing slash
    dir = f'{base}dir/'
    await fs.mkdir(dir)

    await fs.touch(f'{dir}foo')

    # can't test this until after creating foo
    assert await fs.isdir(dir)

    # mkdir without trailing slash
    dir2 = f'{base}dir2'
    await fs.mkdir(dir2)

    await fs.touch(f'{dir2}/foo')

    assert await fs.isdir(dir)


@pytest.mark.asyncio
async def test_isdir_subdir_only(filesystem):
    sema, fs, base = filesystem

    dir = f'{base}dir/'
    await fs.mkdir(dir)

    subdir = f'{dir}subdir/'
    await fs.mkdir(subdir)

    await fs.touch(f'{subdir}foo')

    # can't test this until after creating foo
    assert await fs.isdir(dir)
    assert await fs.isdir(subdir)


@pytest.mark.asyncio
async def test_remove(filesystem):
    sema, fs, base = filesystem

    file = f'{base}foo'

    await fs.touch(file)
    assert await fs.isfile(file)

    await fs.remove(file)

    assert not await fs.isfile(file)


@pytest.mark.asyncio
async def test_rmtree(filesystem):
    sema, fs, base = filesystem

    dir = f'{base}foo/'

    await fs.mkdir(dir)
    await fs.touch(f'{dir}a')
    await fs.touch(f'{dir}b')

    assert await fs.isdir(dir)

    await fs.rmtree(sema, dir)

    assert not await fs.isdir(dir)


@pytest.mark.asyncio
async def test_get_object_metadata():
    bucket = os.environ['HAIL_TEST_BUCKET']
    file = secrets.token_hex(16)

    async with StorageClient() as client:
        async with await client.insert_object(bucket, file) as f:
            await f.write(b'foo')
        metadata = await client.get_object_metadata(bucket, file)
        assert 'etag' in metadata
        assert metadata['md5Hash'] == 'rL0Y20zC+Fzt72VPzMSk2A=='
        assert metadata['crc32c'] == 'z8SuHQ=='
        assert int(metadata['size']) == 3


@pytest.mark.asyncio
async def test_get_object_headers():
    bucket = os.environ['HAIL_TEST_BUCKET']
    file = secrets.token_hex(16)

    async with StorageClient() as client:
        async with await client.insert_object(bucket, file) as f:
            await f.write(b'foo')
        async with await client.get_object(bucket, file) as f:
            headers = f.headers()
            assert 'ETag' in headers
            assert headers['X-Goog-Hash'] == 'crc32c=z8SuHQ==,md5=rL0Y20zC+Fzt72VPzMSk2A=='
            assert await f.read() == b'foo'


@pytest.mark.asyncio
async def test_compose():
    bucket = os.environ['HAIL_TEST_BUCKET']
    token = secret_alnum_string()

    part_data = [b'a', b'bb', b'ccc']

    async with StorageClient() as client:
        for i, b in enumerate(part_data):
            async with await client.insert_object(bucket, f'{token}/{i}') as f:
                await f.write(b)
        await client.compose(bucket, [f'{token}/{i}' for i in range(len(part_data))], f'{token}/combined')

        expected = b''.join(part_data)
        async with await client.get_object(bucket, f'{token}/combined') as f:
            actual = await f.read()
        assert actual == expected


@pytest.mark.asyncio
async def test_statfile_nonexistent_file(filesystem):
    sema, fs, base = filesystem

    with pytest.raises(FileNotFoundError):
        await fs.statfile(f'{base}foo')


@pytest.mark.asyncio
async def test_statfile_directory(filesystem):
    sema, fs, base = filesystem

    await fs.mkdir(f'{base}dir/')
    await fs.touch(f'{base}dir/foo')

    with pytest.raises(FileNotFoundError):
        # statfile raises FileNotFound on directories
        await fs.statfile(f'{base}dir')


@pytest.mark.asyncio
async def test_statfile(filesystem):
    sema, fs, base = filesystem

    n = 37
    file = f'{base}bar'
    async with await fs.create(file) as f:
        await f.write(secrets.token_bytes(n))

    status = await fs.statfile(file)
    assert await status.size() == n

@pytest.mark.asyncio
async def test_listfiles(filesystem):
    sema, fs, base = filesystem

    with pytest.raises(FileNotFoundError):
        await fs.listfiles(f'{base}does/not/exist')

    with pytest.raises(FileNotFoundError):
        await fs.listfiles(f'{base}does/not/exist', recursive=True)

    # create the following directory structure in base:
    # foobar
    # foo/a
    # foo/b/c
    a = f'{base}foo/a'
    b = f'{base}foo/b/'
    c = f'{base}foo/b/c'
    await fs.touch(f'{base}foobar')
    await fs.mkdir(f'{base}foo/')
    await fs.touch(a)
    await fs.mkdir(b)
    await fs.touch(c)

    async def listfiles(dir, recursive):
        return {(await entry.url(), await entry.is_file()) async for entry in await fs.listfiles(dir, recursive)}

    assert await listfiles(f'{base}foo/', recursive=True) == {(a, True), (c, True)}
    assert await listfiles(f'{base}foo/', recursive=False) == {(a, True), (b, False)}

    # without trailing slash
    assert await listfiles(f'{base}foo', recursive=True) == {(a, True), (c, True)}
    assert await listfiles(f'{base}foo', recursive=False) == {(a, True), (b, False)}

    # test FileListEntry.status raises on directory
    async for entry in await fs.listfiles(f'{base}foo/', recursive=False):
        if await entry.is_dir():
            with pytest.raises(ValueError):
                await entry.status()
        else:
            stat = await entry.status()
            assert await stat.size() == 0

@pytest.mark.asyncio
@pytest.mark.parametrize("permutation", [
    None,
    [0, 1, 2],
    [0, 2, 1],
    [1, 2, 0],
    [2, 1, 0]
])
async def test_multi_part_create(filesystem, permutation):
    sema, fs, base = filesystem

    part_data = [secrets.token_bytes(s) for s in [8192, 600, 20000]]

    s = 0
    part_start = []
    for b in part_data:
        part_start.append(s)
        s += len(b)

    path = f'{base}a'
    async with await fs.multi_part_create(sema, path, len(part_data)) as c:
        async def create_part(i):
            async with await c.create_part(i, part_start[i]) as f:
                await f.write(part_data[i])

        if permutation:
            # do it in a fixed order
            for i in permutation:
                await create_part(i)
        else:
            # do in parallel
            await asyncio.gather(*[
                create_part(i) for i in range(len(part_data))])

    expected = b''.join(part_data)
    async with await fs.open(path) as f:
        actual = await f.read()
    assert expected == actual


@pytest.mark.asyncio
async def test_multi_part_create_many(filesystem):
    try:
        sema, fs, base = filesystem

        # > 32 so we perform at least 2 layers of merging
        part_data = [secrets.token_bytes(100) for _ in range(80)]

        s = 0
        part_start = []
        for b in part_data:
            part_start.append(s)
            s += len(b)

        path = f'{base}a'
        async with await fs.multi_part_create(sema, path, len(part_data)) as c:
            async def create_part(i):
                async with await c.create_part(i, part_start[i]) as f:
                    await f.write(part_data[i])

            # do in parallel
            await bounded_gather2(sema, *[
                create_part(i) for i in range(len(part_data))])

        expected = b''.join(part_data)
        async with await fs.open(path) as f:
            actual = await f.read()
        assert expected == actual
    except (concurrent.futures._base.CancelledError, asyncio.CancelledError) as err:
        raise AssertionError('uncaught cancelled error') from err
