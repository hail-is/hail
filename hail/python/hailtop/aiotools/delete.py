from typing import List
import asyncio
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

from ..utils import tqdm
from .router_fs import RouterAsyncFS
from .utils import make_tqdm_listener


async def delete(paths: List[str]) -> None:
    with ThreadPoolExecutor() as thread_pool:
        kwargs = {'thread_pool': thread_pool}
        async with RouterAsyncFS(default_scheme='file', local_kwargs=kwargs, s3_kwargs=kwargs) as fs:
            sema = asyncio.Semaphore(50)
            async with sema:
                with tqdm(desc='files', leave=False, position=0, unit='file') as file_pbar:
                    await asyncio.gather(*[
                        fs.rmtree(sema, path, listener=make_tqdm_listener(file_pbar))
                        for path in paths
                    ])


async def main() -> None:
    parser = argparse.ArgumentParser(description='Delete the given files and directories.',
                                     epilog='''Examples:

    python3 -m hailtop.aiotools.delete dir1/ file1 dir2/file1 dir2/file3 dir3

    python3 -m hailtop.aiotools.delete gs://bucket1/dir1 gs://bucket1/file1 gs://bucket2/abc/123
''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', type=str, nargs='+',
                        help='the paths (files or directories) to delete')
    parser.add_argument('-v', '--verbose', action='store_const',
                        const=True, default=False,
                        help='show logging information')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig()
        logging.root.setLevel(logging.INFO)

    await delete(args.files)

if __name__ == '__main__':
    asyncio.run(main())
