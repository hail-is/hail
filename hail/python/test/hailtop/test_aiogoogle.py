import os
import secrets
import shutil
from concurrent.futures import ThreadPoolExecutor
import pytest
from hailtop.aiotools import LocalAsyncFS, RouterAsyncFS
from hailtop.aiogoogle import StorageClient, GoogleStorageAsyncFS


@pytest.fixture(params=['file', 'gs', 'router/file', 'router/gs'])
async def filesystem(request):
    token = secrets.token_hex(16)

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
                base = f'/tmp/{token}'
            else:
                assert request.param.endswith('gs')
                bucket = os.environ['HAIL_TEST_BUCKET']
                base = f'gs://{bucket}/{token}'

            await fs.mkdir(base)
            yield (fs, base)
            await fs.rmtree(base)
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
    fs, base = filesystem

    token = secrets.token_hex(16)
    file = f'{base}/{token}'

    async with await fs.create(file) as f:
        for b in file_data:
            await f.write(b)

    expected = b''.join(file_data)
    async with await fs.open(file) as f:
        actual = await f.read()

    assert expected == actual


@pytest.mark.asyncio
async def test_isfile(filesystem):
    fs, base = filesystem

    token = secrets.token_hex(16)
    file = f'{base}/{token}'

    # doesn't exist yet
    assert not await fs.isfile(file)

    await fs.touch(file)

    assert await fs.isfile(file)


@pytest.mark.asyncio
async def test_isdir(filesystem):
    fs, base = filesystem

    token = secrets.token_hex(16)
    dir = f'{base}/{token}'
    await fs.mkdir(dir)

    await fs.touch(f'{dir}/foo')

    # can't test this until after creating foo
    assert await fs.isdir(dir)


@pytest.mark.asyncio
async def test_isdir_subdir_only(filesystem):
    fs, base = filesystem

    token1 = secrets.token_hex(16)
    dir = f'{base}/{token1}'
    await fs.mkdir(dir)

    token2 = secrets.token_hex(16)
    subdir = f'{dir}/{token2}'
    await fs.mkdir(subdir)

    await fs.touch(f'{subdir}/foo')

    # can't test this until after creating foo
    assert await fs.isdir(dir)
    assert await fs.isdir(subdir)


@pytest.mark.asyncio
async def test_remove(filesystem):
    fs, base = filesystem

    token = secrets.token_hex(16)
    file = f'{base}/{token}'

    await fs.touch(file)
    assert await fs.isfile(file)

    await fs.remove(file)

    assert not await fs.isfile(file)


@pytest.mark.asyncio
async def test_rmtree(filesystem):
    fs, base = filesystem

    token = secrets.token_hex(16)
    dir = f'{base}/{token}'

    await fs.mkdir(dir)
    await fs.touch(f'{dir}/a')
    await fs.touch(f'{dir}/b')

    assert await fs.isdir(dir)

    await fs.rmtree(dir)

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
