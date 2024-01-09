import os
import secrets
from concurrent.futures import ThreadPoolExecutor
import asyncio
import pytest
import concurrent.futures
import functools
from hailtop.utils import secret_alnum_string, bounded_gather2, retry_transient_errors
from hailtop.aiotools import LocalAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiocloud.aiogoogle import GoogleStorageClient, GoogleStorageAsyncFS


@pytest.fixture(params=['gs', 'router/gs'])
async def gs_filesystem(request):
    token = secret_alnum_string()

    with ThreadPoolExecutor() as thread_pool:
        if request.param.startswith('router/'):
            fs = RouterAsyncFS(filesystems=[LocalAsyncFS(thread_pool), GoogleStorageAsyncFS()])
        else:
            assert request.param.endswith('gs')
            fs = GoogleStorageAsyncFS()
        async with fs:
            test_storage_uri = os.environ['HAIL_TEST_STORAGE_URI']
            protocol = 'gs://'
            assert test_storage_uri[: len(protocol)] == protocol
            base = f'{test_storage_uri}/tmp/{token}/'

            await fs.mkdir(base)
            sema = asyncio.Semaphore(50)
            async with sema:
                yield (sema, fs, base)
                await fs.rmtree(sema, base)
            assert not await fs.isdir(base)


@pytest.fixture
def bucket_and_temporary_file():
    bucket, prefix = GoogleStorageAsyncFS.get_bucket_and_name(os.environ['HAIL_TEST_STORAGE_URI'])
    return bucket, prefix + '/' + secrets.token_hex(16)


def test_bucket_path_parsing():
    bucket, prefix = GoogleStorageAsyncFS.get_bucket_and_name('gs://foo')
    assert bucket == 'foo' and prefix == ''

    bucket, prefix = GoogleStorageAsyncFS.get_bucket_and_name('gs://foo/bar/baz')
    assert bucket == 'foo' and prefix == 'bar/baz'


async def test_get_object_metadata(bucket_and_temporary_file):
    bucket, file = bucket_and_temporary_file

    async with GoogleStorageClient() as client:

        async def upload():
            async with await client.insert_object(bucket, file) as f:
                await f.write(b'foo')

        await retry_transient_errors(upload)
        metadata = await client.get_object_metadata(bucket, file)
        assert 'etag' in metadata
        assert metadata['md5Hash'] == 'rL0Y20zC+Fzt72VPzMSk2A=='
        assert metadata['crc32c'] == 'z8SuHQ=='
        assert int(metadata['size']) == 3


async def test_get_object_headers(bucket_and_temporary_file):
    bucket, file = bucket_and_temporary_file

    async with GoogleStorageClient() as client:

        async def upload():
            async with await client.insert_object(bucket, file) as f:
                await f.write(b'foo')

        await retry_transient_errors(upload)
        async with await client.get_object(bucket, file) as f:
            headers = f.headers()  # type: ignore
            assert 'ETag' in headers
            assert headers['X-Goog-Hash'] == 'crc32c=z8SuHQ==,md5=rL0Y20zC+Fzt72VPzMSk2A=='
            assert await f.read() == b'foo'


async def test_compose(bucket_and_temporary_file):
    bucket, file = bucket_and_temporary_file

    part_data = [b'a', b'bb', b'ccc']

    async with GoogleStorageClient() as client:

        async def upload(i, b):
            async with await client.insert_object(bucket, f'{file}/{i}') as f:
                await f.write(b)

        for i, b in enumerate(part_data):
            await retry_transient_errors(upload, i, b)
        await client.compose(bucket, [f'{file}/{i}' for i in range(len(part_data))], f'{file}/combined')

        expected = b''.join(part_data)
        async with await client.get_object(bucket, f'{file}/combined') as f:
            actual = await f.read()
        assert actual == expected


async def test_multi_part_create_many_two_level_merge(gs_filesystem):
    # This is a white-box test.  compose has a maximum of 32 inputs,
    # so if we're composing more than 32 parts, the
    # GoogleStorageAsyncFS does a multi-level hierarhical merge.
    try:
        sema, fs, base = gs_filesystem

        # > 32 so we perform at least 2 levels of merging
        part_data_size = [100 for _ in range(40)]
        part_data = [secrets.token_bytes(s) for s in part_data_size]

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
            await bounded_gather2(
                sema, *[functools.partial(retry_transient_errors, create_part, i) for i in range(len(part_data))]
            )

        expected = b''.join(part_data)
        actual = await fs.read(path)
        assert expected == actual
    except (concurrent.futures._base.CancelledError, asyncio.CancelledError) as err:
        raise AssertionError('uncaught cancelled error') from err


async def test_weird_urls(gs_filesystem):
    _, fs, base = gs_filesystem

    await fs.write(base + '?', b'contents of ?')
    assert await fs.read(base + '?') == b'contents of ?'

    await fs.write(base + '?a', b'contents of ?a')
    assert await fs.read(base + '?a') == b'contents of ?a'

    await fs.write(base + '?a#b', b'contents of ?a#b')
    assert await fs.read(base + '?a#b') == b'contents of ?a#b'

    await fs.write(base + '#b?a', b'contents of #b?a')
    assert await fs.read(base + '#b?a') == b'contents of #b?a'

    await fs.write(base + '?a#b@c', b'contents of ?a#b@c')
    assert await fs.read(base + '?a#b@c') == b'contents of ?a#b@c'

    await fs.write(base + '#b', b'contents of #b')
    assert await fs.read(base + '#b') == b'contents of #b'

    await fs.write(base + '???', b'contents of ???')
    assert await fs.read(base + '???') == b'contents of ???'
