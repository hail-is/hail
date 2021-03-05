import os
import secrets
from concurrent.futures import ThreadPoolExecutor
import asyncio
import pytest
from hailtop.utils import url_scheme, bounded_gather2
from hailtop.aiotools import LocalAsyncFS, RouterAsyncFS, Transfer, FileAndDirectoryError
from hailtop.aiogoogle import StorageClient, GoogleStorageAsyncFS

from .generate_copy_test_specs import (
    run_test_spec, create_test_file, create_test_dir)

from .copy_test_specs import COPY_TEST_SPECS


@pytest.fixture(scope='module')
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# This fixture is for test_copy_behavior.  It runs a series of copy
# test "specifications" by calling run_test_spec.  The set of
# specifications is enumerated by
# generate_copy_test_specs.py::copy_test_configurations which are then
# run against the local file system.  This tests that (1) that copy
# runs without expected error for each enumerated spec, and that the
# semantics of each filesystem agree.
@pytest.fixture(params=COPY_TEST_SPECS)
async def test_spec(request):
    return request.param


@pytest.fixture(scope='module')
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

            sema = asyncio.Semaphore(50)
            async with sema:
                yield (sema, fs, bases)
                await bounded_gather2(sema,
                                      fs.rmtree(sema, file_base),
                                      fs.rmtree(sema, gs_base))

            assert not await fs.isdir(file_base)
            assert not await fs.isdir(gs_base)

async def fresh_dir(fs, bases, scheme):
    token = secrets.token_hex(16)
    dir = f'{bases[scheme]}{token}/'
    await fs.mkdir(dir)
    return dir


@pytest.fixture(params=['file/file', 'file/gs', 'gs/file', 'gs/gs'])
async def copy_test_context(request, router_filesystem):
    sema, fs, bases = router_filesystem

    [src_scheme, dest_scheme] = request.param.split('/')

    src_base = await fresh_dir(fs, bases, src_scheme)
    dest_base = await fresh_dir(fs, bases, dest_scheme)

    # make sure dest_base exists
    async with await fs.create(f'{dest_base}keep'):
        pass

    yield sema, fs, src_base, dest_base


@pytest.mark.asyncio
async def test_copy_behavior(copy_test_context, test_spec):
    sema, fs, src_base, dest_base = copy_test_context

    result = await run_test_spec(sema, fs, test_spec, src_base, dest_base)
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


class DidNotRaiseError(Exception):
    pass


class RaisedWrongExceptionError(Exception):
    pass


class RaisesOrGS:
    def __init__(self, dest_base, expected_type):
        self._gs = url_scheme(dest_base) == 'gs'
        self._expected_type = expected_type

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # gs can succeed or throw
        if type is None:
            if not self._gs:
                raise DidNotRaiseError()
        elif type != self._expected_type:
            raise RaisedWrongExceptionError(type)

        # suppress exception
        return True

@pytest.mark.asyncio
async def test_copy_doesnt_exist(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    with pytest.raises(FileNotFoundError):
        await fs.copy(sema, Transfer(f'{src_base}a', dest_base))


@pytest.mark.asyncio
async def test_copy_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_large_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    # mainly needs to be larger than the transfer block size (8K)
    contents = secrets.token_bytes(1_000_000)
    async with await fs.create(f'{src_base}a') as f:
        await f.write(contents)

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/')))

    async with await fs.open(f'{dest_base}a') as f:
        copy_contents = await f.read()
    assert copy_contents == contents


@pytest.mark.asyncio
async def test_copy_rename_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x'))

    await expect_file(fs, f'{dest_base}x', 'src/a')


@pytest.mark.asyncio
async def test_copy_rename_file_dest_target_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x', treat_dest_as=Transfer.DEST_IS_TARGET))

    await expect_file(fs, f'{dest_base}x', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_dest_target_directory_doesnt_exist(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    # SourceCopier._copy_file creates destination directories as needed
    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x', treat_dest_as=Transfer.DEST_DIR))
    await expect_file(fs, f'{dest_base}x/a', 'src/a')


@pytest.mark.asyncio
async def test_overwrite_rename_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'dest', dest_base, 'x')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x'))

    await expect_file(fs, f'{dest_base}x', 'src/a')


@pytest.mark.asyncio
async def test_copy_rename_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x'))

    await expect_file(fs, f'{dest_base}x/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}x/subdir/file2', 'src/a/subdir/file2')


@pytest.mark.asyncio
async def test_copy_rename_dir_dest_is_target(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x', treat_dest_as=Transfer.DEST_IS_TARGET))

    await expect_file(fs, f'{dest_base}x/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}x/subdir/file2', 'src/a/subdir/file2')


@pytest.mark.asyncio
async def test_overwrite_rename_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')
    await create_test_dir(fs, 'dest', dest_base, 'x/')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}x', treat_dest_as=Transfer.DEST_IS_TARGET))

    await expect_file(fs, f'{dest_base}x/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}x/subdir/file2', 'src/a/subdir/file2')
    await expect_file(fs, f'{dest_base}x/file3', 'dest/x/file3')


@pytest.mark.asyncio
async def test_copy_file_dest_trailing_slash_target_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base, treat_dest_as=Transfer.DEST_DIR))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_dest_target_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/'), treat_dest_as=Transfer.DEST_DIR))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_dest_target_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', f'{dest_base}a', treat_dest_as=Transfer.DEST_IS_TARGET))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_dest_target_file_is_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    with RaisesOrGS(dest_base, IsADirectoryError):
        await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/'), treat_dest_as=Transfer.DEST_IS_TARGET))


@pytest.mark.asyncio
async def test_overwrite_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'dest', dest_base, 'a')

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a', 'src/a')


@pytest.mark.asyncio
async def test_copy_file_src_trailing_slash(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    with pytest.raises(FileNotFoundError):
        await fs.copy(sema, Transfer(f'{src_base}a/', dest_base))


@pytest.mark.asyncio
async def test_copy_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}a/subdir/file2', 'src/a/subdir/file2')


@pytest.mark.asyncio
async def test_overwrite_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')
    await create_test_dir(fs, 'dest', dest_base, 'a/')

    await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a/file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}a/subdir/file2', 'src/a/subdir/file2')
    await expect_file(fs, f'{dest_base}a/file3', 'dest/a/file3')


@pytest.mark.asyncio
async def test_copy_multiple(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'b')

    await fs.copy(sema, Transfer([f'{src_base}a', f'{src_base}b'], dest_base.rstrip('/')))

    await expect_file(fs, f'{dest_base}a', 'src/a')
    await expect_file(fs, f'{dest_base}b', 'src/b')


@pytest.mark.asyncio
async def test_copy_multiple_dest_target_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'b')

    with RaisesOrGS(dest_base, NotADirectoryError):
        await fs.copy(sema, Transfer([f'{src_base}a', f'{src_base}b'], dest_base.rstrip('/'), treat_dest_as=Transfer.DEST_IS_TARGET))


@pytest.mark.asyncio
async def test_copy_multiple_dest_file(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'b')
    await create_test_file(fs, 'dest', dest_base, 'x')

    with RaisesOrGS(dest_base, NotADirectoryError):
        await fs.copy(sema, Transfer([f'{src_base}a', f'{src_base}b'], f'{dest_base}x'))


@pytest.mark.asyncio
async def test_file_overwrite_dir(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_file(fs, 'src', src_base, 'a')

    with RaisesOrGS(dest_base, IsADirectoryError):
        await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/'), treat_dest_as=Transfer.DEST_IS_TARGET))


@pytest.mark.asyncio
async def test_file_and_directory_error(router_filesystem):
    sema, fs, bases = router_filesystem

    src_base = await fresh_dir(fs, bases, 'gs')
    dest_base = await fresh_dir(fs, bases, 'file')

    await create_test_file(fs, 'src', src_base, 'a')
    await create_test_file(fs, 'src', src_base, 'a/subfile')

    with pytest.raises(FileAndDirectoryError):
        await fs.copy(sema, Transfer(f'{src_base}a', dest_base.rstrip('/')))


@pytest.mark.asyncio
async def test_copy_src_parts(copy_test_context):
    sema, fs, src_base, dest_base = copy_test_context

    await create_test_dir(fs, 'src', src_base, 'a/')

    await fs.copy(sema, Transfer([f'{src_base}a/file1', f'{src_base}a/subdir'], dest_base.rstrip('/'), treat_dest_as=Transfer.DEST_DIR))

    await expect_file(fs, f'{dest_base}file1', 'src/a/file1')
    await expect_file(fs, f'{dest_base}subdir/file2', 'src/a/subdir/file2')
