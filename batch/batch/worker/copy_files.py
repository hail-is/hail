import os
import asyncio
import argparse
import logging
import humanize
import concurrent
import uuid
import traceback
import google.oauth2.service_account

from hailtop.utils import AsyncWorkerPool, blocking_to_async, WaitableSharedPool, WaitableBunch, \
    AsyncOS, TimerBase, NamedLockStore
from hailtop.utils.os import contains_wildcard

from ..google_storage import GCS
from ..utils import parse_memory_in_bytes

log = logging.getLogger('copy_files')

process_pool = None
thread_pool = None
file_lock_store = None
gcs_client = None
async_os = None


class CopyFileTimer(TimerBase):
    def __init__(self, src, dest, size):
        self.src = src
        self.dest = dest
        self.size = humanize.naturalsize(size)
        super().__init__()

    async def __aenter__(self):
        print(f'copying {self.src} to {self.dest} of size {self.size}...')
        return await super().__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        await super().__aexit__(exc_type, exc, tb)
        total = (self.finish_time - self.start_time) / 1000
        if exc_type is None:
            msg = f'copied {self.src} to {self.dest} of size {self.size} in {total:.3f}s'
            if self.timing:
                msg += '\n  ' + '\n  '.join([f'{k}: {v / 1000:.3f}s' for k, v in self.timing.items()])
            print(msg)
        else:
            print(f'failed to copy {self.src} to {self.dest} of size {self.size} in {total:.3f}s due to {exc!r} {traceback.print_tb(tb)}')


def process_initializer(project, key_file):
    assert project is not None

    with Timer('setup process'):
        global gcs_client
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_file)
        gcs_client = GCS(blocking_pool=None, project=project, credentials=credentials)


def read_gs_file(uri, file_name, offset, *args, **kwargs):
    gcs_client._read_gs_file_to_file(uri, file_name, offset, *args, **kwargs)


def write_gs_file(uri, file_name, start, end, *args, **kwargs):
    gcs_client._write_gs_file_from_file(uri, file_name, start, end, *args, **kwargs)


def is_gcs_path(path):
    return path.startswith('gs://')


def get_dest_path(file, src, include_recurse_dir):
    src = src.rstrip('/').split('/')
    file = file.split('/')
    assert len(src) <= len(file)

    if len(src) == len(file):
        return file[-1]

    # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
    if include_recurse_dir:
        recurse_point = len(src) - 1
    else:
        recurse_point = len(src)

    return '/'.join(file[recurse_point:])


def partition_boundaries(file_size, min_partition_size, max_partitions):
    if file_size == 0:
        return ([0, 0], 1)

    n_partitions = (file_size + min_partition_size - 1) // min_partition_size

    if max_partitions:
        n_partitions = min(max_partitions, n_partitions)

    partition_size = file_size // n_partitions
    n_partitions_w_extra_byte = file_size % n_partitions

    partition_sizes = [partition_size + 1] * n_partitions_w_extra_byte + \
                      [partition_size] * (n_partitions - n_partitions_w_extra_byte)

    pos = 0
    boundaries = [0]
    for size in partition_sizes:
        pos += size
        boundaries.append(pos)

    return (boundaries, n_partitions)


async def glob_gcs(path, recursive=False):
    return [('gs://' + blob.bucket.name + '/' + blob.name, blob.size)
            for blob in await gcs_client.glob_gs_files(path, recursive=recursive)]


async def is_gcs_dir(path):
    return is_gcs_path(path) and \
           (path.endswith('/') or len(await gcs_client.glob_gs_files(path.rstrip('/') + '/*', recursive=True)) != 0)


async def is_local_dir(path):
    # gsutil does not copy empty local directories
    return not is_gcs_path(path) and \
           (path.endswith('/') or len(await async_os.glob(path.rstrip('/') + '/*', recursive=True)) != 0)


async def is_dir(path):
    if is_gcs_path(path):
        return await is_gcs_dir(path)
    return await is_local_dir(path)


async def file_exists(path):
    if is_gcs_path(path):
        return len(await glob_gcs(path, recursive=False)) == 1
    return os.path.isfile(path)


async def dir_exists(path):
    if is_gcs_path(path):
        return next(await gcs_client.list_gs_files(path.rstrip('/') + '/*', max_results=1), None) is not None
    return os.path.isdir(path)


def any_recursive_match(pattern, paths):
    pattern = pattern.rstrip('/').split('/')
    x = any([len(pattern) != len(path.split('/')) for path in paths])
    return x


async def add_destination_paths(src, glob_paths, dest):
    # gsutil throws an error if no files match
    if len(glob_paths) == 0:
        raise FileNotFoundError

    recursive_match = any_recursive_match(src, (p for p, _ in glob_paths))

    if is_gcs_path(src):
        is_dir_src = recursive_match
    else:
        is_dir_src = recursive_match

    if is_gcs_path(dest):
        is_dir_dest = dest.endswith('/') or await dir_exists(dest)
    else:
        is_dir_dest = os.path.isdir(dest)

    print(f'is_dir_src {is_dir_src} is_dir_dest {is_dir_dest}')

    if not is_dir_src and not is_dir_dest:
        if len(glob_paths) == 1:
            file, _ = glob_paths[0]
            if file.endswith('/'):
                print(f'cannot copy a remote file ending in a slash to a local directory, {file}')
                raise NotImplementedError
            return [(file, dest, size) for file, size in glob_paths]
        print('destination must name a directory when matching multiple files')
        raise NotADirectoryError

    filtered_glob_paths = []
    for path, size in glob_paths:
        if path.endswith('/'):
            print(f'ignoring file ending with slash of size {humanize.naturalsize(size)}, {path}')
        filtered_glob_paths.append((path, size))
    glob_paths = filtered_glob_paths

    if not is_dir_src and is_dir_dest:
        dest = dest.rstrip('/') + '/'
        return [(file, f'{dest}{os.path.basename(file)}', size) for file, size in glob_paths]

    assert is_dir_src
    # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
    include_recurse_dir = not is_gcs_path(dest) or is_dir_dest
    return [(file, dest.rstrip('/') + '/' + get_dest_path(file, src, include_recurse_dir), size) for file, size in glob_paths]


    # async def add_destination_paths(src, glob_paths, dest):
#     # gsutil throws an error if no files match
#     if len(glob_paths) == 0:
#         raise FileNotFoundError
#
#     recursive_match = any_recursive_match(src, (p for p, _ in glob_paths))
#
#     # local -> local
#     if not is_gcs_path(src) and not is_gcs_path(dest):
#         is_dir_src = recursive_match
#         is_dir_dest = os.path.isdir(dest)
#
#         if not is_dir_src and not is_dir_dest:
#             if len(glob_paths) == 1:
#                 return [(file, dest, size) for file, size in glob_paths]
#             print('destination must name a directory when matching multiple files')
#             raise NotADirectoryError
#
#         if not is_dir_src and is_dir_dest:
#             dest = dest.rstrip('/') + '/'
#             return [(file, f'{dest}{os.path.basename(file)}', size) for file, size in glob_paths]
#
#         assert is_dir_src
#         # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
#         include_recurse_dir = not is_gcs_path(dest) or dest.endswith('/') or await dir_exists(dest)
#         return [(path, dest.rstrip('/') + '/' + get_dest_path(path, src, include_recurse_dir), size) for path, size in glob_paths]
#
#     # remote -> local
#     if is_gcs_path(src) and not is_gcs_path(dest):
#         is_dir_src = recursive_match
#         is_dir_dest = os.path.isdir(dest)
#
#         if not is_dir_src and not is_dir_dest:
#             if len(glob_paths) == 1:
#                 file, _ = glob_paths[0]
#                 if file.endswith('/'):
#                     print(f'cannot copy a remote file ending in a slash to a local directory, {file}')
#                     raise NotImplementedError
#                 return [(file, dest, size) for file, size in glob_paths]
#             print('destination must name a directory when matching multiple files')
#             raise NotADirectoryError
#
#         filtered_glob_paths = []
#         for path, size in glob_paths:
#             if path.endswith('/'):
#                 print(f'ignoring file ending with slash of size {humanize.naturalsize(size)}, {path}')
#             filtered_glob_paths.append((path, size))
#         glob_paths = filtered_glob_paths
#
#         if not is_dir_src and is_dir_dest:
#             dest = dest.rstrip('/') + '/'
#             return [(file, f'{dest}{os.path.basename(file)}', size) for file, size in glob_paths]
#
#         assert is_dir_src
#         # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
#         include_recurse_dir = not is_gcs_path(dest) or dest.endswith('/') or await dir_exists(dest)
#         return [(path, dest.rstrip('/') + '/' + get_dest_path(path, src, include_recurse_dir), size) for path, size in glob_paths]
#
#     # local -> remote, remote -> remote
#     is_dir_src = recursive_match
#     is_dir_dest = dest.endswith('/') or await dir_exists(dest)
#
#     if not is_dir_src:
#         if len(glob_paths) == 1:
#             return [(file, dest, size) for file, size in glob_paths]
#         if not is_dir_dest:
#             print('destination must name a directory when matching multiple files')
#             raise NotADirectoryError
#
#     # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
#     return [(file, dest.rstrip('/') + '/' + get_dest_path(file, src, is_dir_dest), size) for file, size in glob_paths]





    # is_dir_src, is_dir_dest = await asyncio.gather(is_dir(src), is_dir(dest))
    # recursive_match = any_recursive_match(src, (p for p, _ in glob_paths))
    #
    # # local src files
    # if not is_gcs_path(src):
    #     if not is_dir_src and not is_dir_dest:
    #         assert not recursive_match
    #         if len(glob_paths) == 1:
    #             return [(file, dest, size) for file, size in glob_paths]
    #         print('destination must name a directory when matching multiple files')
    #         raise NotADirectoryError
    #     if not is_dir_src and is_dir_dest:
    #         dest = dest.rstrip('/') + '/'
    #         return [(file, f'{dest}{os.path.basename(file)}', size) for file, size in glob_paths]
    #     assert is_dir_src
    #     # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
    #     include_recurse_dir = not is_gcs_path(dest) or dest.endswith('/') or await dir_exists(dest)
    #     return [(path, dest.rstrip('/') + '/' + get_dest_path(path, src, include_recurse_dir), size) for path, size in glob_paths]


# is_file_dest, is_dir_dest = await asyncio.gather(file_exists(dest), dir_exists(dest))





# async def add_destination_paths(src, glob_paths, dest):
#     is_dir_src = await is_dir(src)
#     is_dir_dest = await is_dir(dest)
#
#     if len(glob_paths) == 0:
#         raise FileNotFoundError
#
#     if is_gcs_path(src) and not is_gcs_path(dest):
#         has_recursive_match = any_recursive_match(src, [p for p, _ in glob_paths])
#         if not has_recursive_match and dest.endswith('/') and not await dir_exists(dest):
#             if len(glob_paths) == 1:
#                 if is_gcs_path(src):
#                     print('skipping destination file ending with slash')
#                     return []
#                 print('destination is a directory')
#                 raise IsADirectoryError
#             if len(glob_paths) > 1:
#                 print('destination must name a directory when matching multiple files')
#                 raise NotADirectoryError
#
#     if is_gcs_path(src) and src.endswith('/'):
#         src_is_file = len(await gcs_client.glob_gs_files(src, recursive=False)) == 1
#         if src_is_file:
#             if not is_gcs_path(dest) and is_dir_dest:
#                 print(f'cannot copy a remote file ending in a slash to a local directory')
#                 raise NotImplementedError
#             return [(file, dest, size) for file, size in glob_paths]
#
#     if is_gcs_path(src) and not is_gcs_path(dest):
#         filtered_glob_paths = []
#         for path, size in glob_paths:
#             if path.endswith('/'):
#                 print(f'ignoring file ending with slash of size {humanize.naturalsize(size)}, {path}')
#             filtered_glob_paths.append((path, size))
#         glob_paths = filtered_glob_paths
#
#     if not is_dir_src and not is_dir_dest:
#         if len(glob_paths) == 1:
#             return [(file, dest, size) for file, size in glob_paths]
#         if is_gcs_path(dest):
#             include_recurse_dir = False
#             return [(path, dest.rstrip('/') + '/' + get_dest_path(path, src, include_recurse_dir), size)
#                     for path, size in glob_paths]
#         raise NotADirectoryError
#
#     if not is_dir_src and is_dir_dest:
#         dest = dest.rstrip('/') + '/'
#         return [(file, f'{dest}{os.path.basename(file)}', size) for file, size in glob_paths]
#
#     # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed_1
#     include_recurse_dir = not is_gcs_path(dest) or dest.endswith('/') or await dir_exists(dest)
#     return [(path, dest.rstrip('/') + '/' + get_dest_path(path, src, include_recurse_dir), size) for path, size in glob_paths]

import time
class Timer:
    def __init__(self, desc):
        self.start = None
        self.desc = desc

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'took {self.desc} {time.time() - self.start}s')


class Copier:
    def __init__(self, copy_pool, min_partition_size, max_upload_partitions, max_download_partitions):
        self.copy_pool = copy_pool
        self.min_partition_size = min_partition_size
        self.max_upload_partitions = max_upload_partitions
        self.max_download_partitions = max_download_partitions

    async def copy(self, src, dest):
        cp = self._copy_method_for_remoteness(is_gcs_path(src), is_gcs_path(dest))
        recursive = src.endswith('/') or contains_wildcard(src) or not await file_exists(src)
        with Timer('glob files'):
            glob_paths = await self._glob(src, recursive=recursive)
        with Timer('add dest paths'):
            tasks = await add_destination_paths(src, glob_paths, dest)
        for s, d, size in tasks:
            await cp(s, d, size)

    def _copy_method_for_remoteness(self, src_is_remote, dest_is_remote):
        if src_is_remote:
            if dest_is_remote:
                return self._copy_file_within_gcs
            return self._read_file_from_gcs
        if dest_is_remote:
            return self._write_file_to_gcs
        return self._copy_local_files

    async def _glob(self, path, recursive=False):
        if is_gcs_path(path):
            return await glob_gcs(path, recursive=recursive)
        return await async_os.glob(path, recursive=recursive)

    async def _copy_local_files(self, src, dest, size):
        async def _copy(src, dest):
            async with file_lock_store.get_lock(dest):
                async with CopyFileTimer(src, dest, size):
                    await async_os.cp(src, dest)

        await self.copy_pool.call(_copy, src, dest)

    async def _read_file_from_gcs(self, src, dest, size):
        async def _read(timer, start, end):
            async with timer.step(f'download [{start}, {end + 1})'):
                await blocking_to_async(process_pool, read_gs_file, src, dest, offset=start, start=start, end=end)

        async with file_lock_store.get_lock(dest):
            async with CopyFileTimer(src, dest, size) as timer:
                try:
                    async with timer.step('setup'):
                        await async_os.make_parent_dirs(dest)
                        await async_os.remove(dest)
                        await async_os.new_file(dest, size)

                    boundaries, n_partitions = partition_boundaries(size, self.min_partition_size, self.max_download_partitions)

                    wb = WaitableBunch(self.copy_pool)
                    for i in range(n_partitions):
                        await wb.call(_read, timer, boundaries[i], boundaries[i+1] - 1)
                    await wb.wait()
                except Exception:
                    if os.path.exists(dest):
                        os.remove(dest)
                    raise

    async def _write_file_to_gcs(self, src, dest, size):
        async def _write(timer, tmp_dest, start, end):
            async with timer.step(f'upload [{start}, {end})'):
                await blocking_to_async(process_pool, write_gs_file, tmp_dest, src, start, end)

        async with file_lock_store.get_lock(dest):
            async with CopyFileTimer(src, dest, size) as timer:
                boundaries, n_partitions = partition_boundaries(size, self.min_partition_size, max_partitions=self.max_upload_partitions)

                if n_partitions == 1:
                    assert len(boundaries) == 2
                    await self.copy_pool.call(_write, timer, dest, boundaries[0], boundaries[1])
                    return

                token = uuid.uuid4().hex[:8]
                tmp_dests = [dest + f'/{token}/{uuid.uuid4().hex[:8]}' for _ in range(n_partitions)]
                try:
                    wb = WaitableBunch(self.copy_pool)
                    for i in range(n_partitions):
                        await wb.call(_write, timer, tmp_dests[i], boundaries[i], boundaries[i+1])
                    await wb.wait()

                    async with timer.step('compose'):
                        await gcs_client.compose_gs_file(tmp_dests, dest)
                finally:
                    async with timer.step('delete temp files'):
                        await gcs_client.delete_gs_files(dest + f'/{token}/')

    async def _copy_file_within_gcs(self, src, dest, size):
        async def _copy(src, dest):
            async with file_lock_store.get_lock(dest):
                async with CopyFileTimer(src, dest, size):
                    await gcs_client.copy_gs_file(src, dest)
        await self.copy_pool.call(_copy, src, dest)


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--key-file', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--parallelism', type=int, default=1)
    parser.add_argument('--max-upload-partitions', type=int, default=32)
    parser.add_argument('--max-download-partitions', type=int, default=None)
    parser.add_argument('--min-partition-size', type=str, default='1Gi')
    parser.add_argument('-f', '--files', action='append', type=str, nargs=2, metavar=('src', 'dest'))

    with Timer('parse args'):
        args = parser.parse_args()

    global process_pool, thread_pool, gcs_client, file_lock_store, async_os

    with Timer('setup all global variables'):
        process_pool = concurrent.futures.ProcessPoolExecutor(
            initializer=process_initializer,
            initargs=(args.project, args.key_file))

        thread_pool = concurrent.futures.ThreadPoolExecutor()
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(args.key_file)
        gcs_client = GCS(thread_pool, project=args.project, credentials=credentials)

        async_os = AsyncOS(thread_pool)

        file_worker_pool = AsyncWorkerPool(10)  # max 10 simultaneous globbing of files
        file_pool = WaitableSharedPool(file_worker_pool, ignore_errors=False)

        copy_worker_pool = AsyncWorkerPool(args.parallelism)
        copy_pool = WaitableSharedPool(copy_worker_pool, ignore_errors=False)

        file_lock_store = NamedLockStore()

        copier = Copier(copy_pool, parse_memory_in_bytes(args.min_partition_size),
                        args.max_upload_partitions, args.max_download_partitions)

    try:
        for src, dest in args.files:
            if '**' in src:
                raise NotImplementedError(f'** not supported; got {src}')
            with Timer('glob and add to copy pool'):
                await file_pool.call(copier.copy, src, dest)
        with Timer('wait for globbing and copying to be done'):
            await file_pool.wait()
            await copy_pool.wait()
    finally:
        with Timer('cancel worker pools'):
            await file_worker_pool.cancel()
            await copy_worker_pool.cancel()

with Timer('time in event loop'):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
