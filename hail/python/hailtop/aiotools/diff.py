from typing import Optional, Tuple, TypeVar, List
import argparse
import asyncio
import orjson
import logging
import sys

from concurrent.futures import ThreadPoolExecutor

from ..utils.rich_progress_bar import SimpleCopyToolProgressBar, SimpleCopyToolProgressBarTask
from .router_fs import RouterAsyncFS
from .fs import AsyncFS, FileStatus

try:
    import uvloop

    uvloop_install = uvloop.install
except ImportError as e:
    if not sys.platform.startswith('win32'):
        raise e

    def uvloop_install():
        pass


async def diff(
    *,
    max_simultaneous: Optional[int] = None,
    local_kwargs: Optional[dict] = None,
    gcs_kwargs: Optional[dict] = None,
    azure_kwargs: Optional[dict] = None,
    s3_kwargs: Optional[dict] = None,
    source: str,
    target: str,
    verbose: bool = False,
) -> List[dict]:
    with ThreadPoolExecutor() as thread_pool:
        if max_simultaneous is None:
            max_simultaneous = 500
        if local_kwargs is None:
            local_kwargs = {}
        if 'thread_pool' not in local_kwargs:
            local_kwargs['thread_pool'] = thread_pool

        if s3_kwargs is None:
            s3_kwargs = {}
        if 'thread_pool' not in s3_kwargs:
            s3_kwargs['thread_pool'] = thread_pool
        if 'max_pool_connections' not in s3_kwargs:
            s3_kwargs['max_pool_connections'] = max_simultaneous * 2

        async with RouterAsyncFS(
            local_kwargs=local_kwargs, gcs_kwargs=gcs_kwargs, azure_kwargs=azure_kwargs, s3_kwargs=s3_kwargs
        ) as fs:
            with SimpleCopyToolProgressBar(description='files', transient=True, total=0, disable=not verbose) as pbar:
                return await do_diff(source, target, fs, max_simultaneous, pbar)


async def stat_and_size(fs: AsyncFS, url: str) -> Tuple[FileStatus, int]:
    stat = await fs.statfile(url)
    return (stat, await stat.size())


class DiffException(ValueError):
    pass


async def do_diff(
    top_source: str, top_target: str, fs: AsyncFS, max_simultaneous: int, pbar: SimpleCopyToolProgressBarTask
) -> List[dict]:
    if await fs.isfile(top_source):
        result = await diff_one(top_source, top_target, fs)
        if result is None:
            return []
        return [result]

    if not top_source.endswith('/'):
        top_source += '/'
    if not top_target.endswith('/'):
        top_target += '/'

    # The concurrency limit is the number of worker corooutines, not the queue size. Queue size must
    # be large because a single driver process is trying to feed max_simultaneous workers.
    active_tasks: asyncio.Queue[Optional[Tuple[str, str]]] = asyncio.Queue(max_simultaneous * 10)
    different = []

    async def create_work():
        try:
            filegenerator = await fs.listfiles(top_source, recursive=True)
        except FileNotFoundError as err:
            raise DiffException(f'Source URL refers to no files or directories: {top_source}') from err
        async for source_status in filegenerator:
            source_url = await source_status.url()
            assert source_url.startswith(top_source)
            suffix = source_url.removeprefix(top_source)
            target_url = top_target + suffix

            pbar.update(0, total=pbar.total() + 1)
            await active_tasks.put((source_url, target_url))

    async def worker():
        while True:
            urls = await active_tasks.get()
            if urls is None:
                return
            source_url, target_url = urls
            result = await diff_one(source_url, target_url, fs)
            if result is not None:
                different.append(result)
            pbar.update(1)
            active_tasks.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max_simultaneous)]

    try:
        await create_work()
        await active_tasks.join()
    finally:
        for _ in range(max_simultaneous):
            await active_tasks.put(None)
        await asyncio.gather(*workers)

    return different


T = TypeVar('T')


async def diff_one(source_url: str, target_url: str, fs: AsyncFS) -> Optional[dict]:
    source_task = asyncio.create_task(stat_and_size(fs, source_url))
    target_task = asyncio.create_task(stat_and_size(fs, target_url))
    await asyncio.wait([source_task, target_task])

    try:
        _, ssize = await source_task
    except FileNotFoundError as err:
        target_task.cancel()
        raise DiffException(f'Source file removed during diff: {source_url}') from err

    try:
        _, tsize = await target_task
    except FileNotFoundError:
        return {"from": source_url, "to": target_url, "from_size": ssize, "to_size": None}

    if ssize != tsize:
        return {"from": source_url, "to": target_url, "from_size": ssize, "to_size": tsize}

    return None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description='Hail size diff tool. Recursively finds files which differ in size or are entirely missing.'
    )
    parser.add_argument(
        '--requester-pays-project', type=str, nargs='?', help='The Google project to which to charge egress costs.'
    )
    parser.add_argument('source', type=str, help='The source of truth file or directory.')
    parser.add_argument('target', type=str, help='The target file or directory to which to compare.')
    parser.add_argument('--max-simultaneous', type=int, help='The limit on the number of simultaneous diff operations.')
    parser.add_argument(
        '-v', '--verbose', action='store_const', const=True, default=False, help='show logging information'
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig()
        logging.root.setLevel(logging.INFO)

    gcs_kwargs = {'gcs_requester_pays_configuration': args.requester_pays_project}

    try:
        different = await diff(
            max_simultaneous=args.max_simultaneous,
            gcs_kwargs=gcs_kwargs,
            source=args.source,
            target=args.target,
            verbose=args.verbose,
        )
    except DiffException as exc:
        print(exc.args[0], file=sys.stderr)
        sys.exit(1)
    else:
        sys.stdout.buffer.write(orjson.dumps(different))


if __name__ == '__main__':
    uvloop_install()
    asyncio.run(main())
