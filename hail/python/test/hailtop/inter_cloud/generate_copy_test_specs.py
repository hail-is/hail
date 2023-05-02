import secrets
from concurrent.futures import ThreadPoolExecutor
import pprint
import asyncio
from hailtop.aiotools import LocalAsyncFS, Transfer, Copier
from hailtop.aiotools.router_fs import RouterAsyncFS


def remove_prefix(s, prefix):
    assert s.startswith(prefix), (prefix, s)
    return s[len(prefix):]


async def create_test_file(fs, name, base, path):
    assert name in ('src', 'dest')
    assert not path.endswith('/')

    async with await fs.create(f'{base}{path}') as f:
        await f.write(f'{name}/{path}'.encode('utf-8'))


async def create_test_dir(fs, name, base, path):
    '''Create a directory of test data.

    The directory test data depends on the name (src or dest) so, when
    testing overwriting for example, there is a file in src which does
    not exist in dest, a file in dest that does not exist in src, and
    one that exists in both.

    The src configuration looks like:
     - {base}/src/a/file1
     - {base}/src/a/subdir/file2

    The dest configuration looks like:
     - {base}/dest/a/subdir/file2
     - {base}/dest/a/file3
    '''
    assert name in ('src', 'dest')
    assert path.endswith('/')

    await fs.mkdir(f'{base}{path}')
    if name == 'src':
        await create_test_file(fs, name, base, f'{path}file1')
    await fs.mkdir(f'{base}{path}subdir/')
    await create_test_file(fs, name, base, f'{path}subdir/file2')
    if name == 'dest':
        await create_test_file(fs, name, base, f'{path}file3')


async def create_test_data(fs, name, base, path, type):
    assert not path.endswith('/')
    if type == 'file':
        await create_test_file(fs, name, base, path)
    elif type == 'dir':
        await create_test_dir(fs, name, base, f'{path}/')
    else:
        assert type == 'noexist', type


def copy_test_configurations():
    for src_type in ['file', 'dir', 'noexist']:
        for dest_type in ['file', 'dir', 'noexist']:
            for dest_basename in [None, 'a', 'x']:
                for treat_dest_as in [Transfer.DEST_DIR, Transfer.DEST_IS_TARGET, Transfer.INFER_DEST]:
                    for src_trailing_slash in [True, False]:
                        for dest_trailing_slash in [True, False]:
                            yield {
                                'src_type': src_type,
                                'dest_type': dest_type,
                                'dest_basename': dest_basename,
                                'treat_dest_as': treat_dest_as,
                                'src_trailing_slash': src_trailing_slash,
                                'dest_trailing_slash': dest_trailing_slash
                            }


async def run_test_spec(sema, fs, spec, src_base, dest_base):
    await create_test_data(fs, 'src', src_base, 'a', spec['src_type'])
    await create_test_data(fs, 'dest', dest_base, 'a', spec['dest_type'])

    src = f'{src_base}a'
    if spec['src_trailing_slash']:
        src = src + '/'

    dest_basename = spec['dest_basename']
    if dest_basename:
        dest = f'{dest_base}{dest_basename}'
    else:
        dest = dest_base
    if spec['dest_trailing_slash']:
        if not dest.endswith('/'):
            dest = dest + '/'
    else:
        dest = dest.rstrip('/')

    result = None
    exc_type = None
    try:
        await Copier.copy(fs, sema, Transfer(src, dest, treat_dest_as=spec['treat_dest_as']))
    except Exception as e:
        exc_type = type(e)
        if exc_type not in (NotADirectoryError, IsADirectoryError, FileNotFoundError):
            raise
        result = {'exception': exc_type.__name__}

    if exc_type is None:
        files = {}
        async for entry in await fs.listfiles(dest_base, recursive=True):
            url = await entry.url()
            assert not url.endswith('/')
            file = remove_prefix(url, dest_base.rstrip('/'))
            async with await fs.open(url) as f:
                contents = (await f.read()).decode('utf-8')
            files[file] = contents
        result = {'files': files}

    return result


async def copy_test_specs():
    test_specs = []

    with ThreadPoolExecutor() as thread_pool:
        async with RouterAsyncFS(filesystems=[LocalAsyncFS(thread_pool)]) as fs:
            for config in copy_test_configurations():
                token = secrets.token_hex(16)

                base = f'/tmp/{token}/'
                src_base = f'{base}src/'
                dest_base = f'{base}dest/'

                await fs.mkdir(base)
                await fs.mkdir(src_base)
                await fs.mkdir(dest_base)
                # make sure dest_base exists
                async with await fs.create(f'{dest_base}keep'):
                    pass

                sema = asyncio.Semaphore(50)
                async with sema:
                    result = await run_test_spec(sema, fs, config, src_base, dest_base)
                    config['result'] = result

                    test_specs.append(config)

                    await fs.rmtree(sema, base)
                    assert not await fs.isdir(base)

    return test_specs


async def main():
    test_specs = await copy_test_specs()
    with open('test/hailtop/aiotools/copy_test_specs.py', 'w') as f:
        f.write(f'COPY_TEST_SPECS = ')
        pprint.pprint(test_specs, stream=f)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
