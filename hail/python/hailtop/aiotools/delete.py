import argparse
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from ..utils import grouped
from ..utils.rich_progress_bar import SimpleCopyToolProgressBar
from .router_fs import RouterAsyncFS


async def delete(paths: Iterator[str]) -> None:
    with ThreadPoolExecutor() as thread_pool:
        kwargs = {'thread_pool': thread_pool}
        async with RouterAsyncFS(local_kwargs=kwargs, s3_kwargs=kwargs) as fs:
            sema = asyncio.Semaphore(50)
            async with sema:
                with SimpleCopyToolProgressBar(description='files', transient=True, total=0) as file_pbar:
                    listener = file_pbar.make_listener()

                    async def remove(path):
                        try:
                            await fs.remove(path)
                        except FileNotFoundError:
                            await fs.rmtree(sema, path, listener=listener)
                        file_pbar.update(1)  # only advance if file or directory removal was successful, not on error

                    for grouped_paths in grouped(5_000, paths):
                        file_pbar.update(0, total=file_pbar.total() + len(grouped_paths))
                        await asyncio.gather(*[remove(path) for path in grouped_paths])


async def main() -> None:
    parser = argparse.ArgumentParser(
        description='Delete the given files and directories.',
        epilog="""Examples:

    python3 -m hailtop.aiotools.delete dir1/ file1 dir2/file1 dir2/file3 dir3

    python3 -m hailtop.aiotools.delete gs://bucket1/dir1 gs://bucket1/file1 gs://bucket2/abc/123
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'files', type=str, nargs='*', help='the paths (files or directories) to delete; if unspecified, read from stdin'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_const', const=True, default=False, help='show logging information'
    )
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig()
        logging.root.setLevel(logging.INFO)
    files = args.files
    if len(files) == 0:
        files = (x.strip() for x in sys.stdin if x != '')
    else:
        files = iter(files)

    await delete(files)


if __name__ == '__main__':
    asyncio.run(main())
