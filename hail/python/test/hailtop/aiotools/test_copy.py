import os
import secrets
from concurrent.futures import ThreadPoolExecutor
import asyncio
import pytest
from hailtop.utils import url_scheme
from hailtop.aiotools import LocalAsyncFS, RouterAsyncFS, Transfer
from hailtop.aiogoogle import StorageClient, GoogleStorageAsyncFS

from .generate_copy_test_specs import (
    run_test_spec, create_test_file, create_test_dir)

from .copy_test_specs import COPY_TEST_SPECS


@pytest.fixture(scope='module')
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(params=COPY_TEST_SPECS)
async def test_spec(request):
    return request.param


@pytest.fixture(scope='module', params=['file/file', 'file/gs', 'gs/file', 'gs/gs'])
# @pytest.fixture(scope='module', params=['file/file'])
async def router_filesystem(request):
    token = secrets.token_hex(16)

    with ThreadPoolExecutor() as thread_pool:
        async with RouterAsyncFS(
                'file', [LocalAsyncFS(thread_pool), GoogleStorageAsyncFS()]) as fs:
            file_base = f'/tmp/{token}/'
            await fs.mkdir(file_base)

            bucket = os.environ['HAIL_TEST_BUCKET']
            gs_base = f'gs://{bucket}/tmp/{token}/'

            bases = {
                'file': file_base,
                'gs': gs_base
            }

            [src_scheme, dest_scheme] = request.param.split('/')

            src_base = f'{bases[src_scheme]}src/'
            dest_base = f'{bases[dest_scheme]}dest/'

            await fs.mkdir(src_base)
            await fs.mkdir(dest_base)

            yield (fs, src_base, dest_base)

            await fs.rmtree(file_base)
            assert not await fs.isdir(file_base)

            await fs.rmtree(gs_base)
            assert not await fs.isdir(gs_base)


@pytest.fixture
async def copy_test_context(router_filesystem):
    fs, src_base, dest_base = router_filesystem

    token = secrets.token_hex(16)
    src_base = f'{src_base}{token}/'
    dest_base = f'{dest_base}{token}/'

    await fs.mkdir(src_base)
    await fs.mkdir(dest_base)
    # make sure dest_base exists
    async with await fs.create(f'{dest_base}keep'):
        pass

    yield fs, src_base, dest_base


@pytest.mark.asyncio
async def test_copy_behavior(copy_test_context, test_spec):
    fs, src_base, dest_base = copy_test_context

    result = await run_test_spec(fs, test_spec, src_base, dest_base)
    try:
        expected = test_spec['result']

        dest_scheme = url_scheme(dest_base)
        if (dest_scheme == 'gs'
                and 'files' in result
                and expected.get('exception') in ('IsADirectoryError', 'NotADirectoryError')):
            return

        assert result == expected, (test_spec, result, expected)
    except Exception:
        print(test_spec)
        raise


async def expect_file(fs, path, expected):
    async with await fs.open(path) as f:
        actual = (await f.read()).decode('utf-8')
    assert actual == expected, (actual, expected)


@pytest.mark.asyncio
async def test_copy_doesnt_exist(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    with pytest.raises(FileNotFoundError):
        await fs.copy(Transfer(f'{src_base}a', dest_base))


@pytest.mark.asyncio
async def test_copy_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_rename_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', f'{dest_base}x'))

    await expect_file(fs, f'{dest_base}x', 'src/a')


@pytest.mark.asyncio
async def test_copy_rename_file_dest_target_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', f'{dest_base}x', treat_dest_as=Transfer.TARGET_FILE))

    await expect_file(fs, f'{dest_base}x', 'src/a')


@pytest.mark.asyncio
async def test_overwrite_rename_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'dest', dest_base, 'x')

    await fs.copy(Transfer(f'{src_base}a', f'{dest_base}x'))

    await expect_file(fs, f'{dest_base}x', 'src/a')


@pytest.mark.asyncio
async def test_copy_rename_dir(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')

    await fs.copy(Transfer(f'{src_base}a', f'{dest_base}x'))

    await expect_file(fs, f'{dest_base}x/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}x/subdir/file2', 'src/a/subdir/file2')


@pytest.mark.asyncio
async def test_copy_file_dest_trailing_slash_target_dir(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', dest_base, treat_dest_as=Transfer.TARGET_DIR))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_dest_target_dir(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', dest_base.rstrip('/'), treat_dest_as=Transfer.TARGET_DIR))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_dest_target_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', f'{dest_base}a', treat_dest_as=Transfer.TARGET_FILE))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_overwrite_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'dest', dest_base, 'a')

    await fs.copy(Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_src_trailing_slash(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    with pytest.raises(FileNotFoundError):
        await fs.copy(Transfer(f'{src_base}a/', dest_base))


@pytest.mark.asyncio
async def test_copy_dir(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')

    await fs.copy(Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}a/subdir/file2', 'src/a/subdir/file2')


@pytest.mark.asyncio
async def test_overwrite_dir(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')
    await create_test_dir(fs, 'dest', dest_base, 'a/')

    await fs.copy(Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}a/subdir/file2', 'src/a/subdir/file2')
    await expect_file(fs, f'{dest_base}a/file3', 'dest/a/file3')


@pytest.mark.asyncio
async def test_copy_multiple(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'b')

    await fs.copy(Transfer([f'{src_base}a', f'{src_base}b'], dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a', 'src/a')
    await expect_file(fs, f'{dest_base}b', 'src/b')


@pytest.mark.asyncio
async def test_copy_multiple_dest_target_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'b')

    with pytest.raises(NotADirectoryError):
        await fs.copy(Transfer([f'{src_base}a', f'{src_base}b'], dest_base.rstrip('/'), treat_dest_as=Transfer.TARGET_FILE))


@pytest.mark.asyncio
async def test_copy_multiple_dest_file(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'b')
    await create_test_file(fs, 'dest', dest_base, 'x')

    with pytest.raises(NotADirectoryError):
        await fs.copy(Transfer([f'{src_base}a', f'{src_base}b'], f'{dest_base}x'))


@pytest.mark.asyncio
async def test_file_overwrite_dir(copy_test_context):
    fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    with pytest.raises(IsADirectoryError):
        await fs.copy(Transfer(f'{src_base}a', dest_base.rstrip('/'), treat_dest_as=Transfer.TARGET_FILE))
