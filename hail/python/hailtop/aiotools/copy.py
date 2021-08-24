from typing import List, Optional
import sys
import json
import urllib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hailtop.aiotools.fs import RouterAsyncFS, LocalAsyncFS, Transfer
from hailtop.aiogoogle import GoogleStorageAsyncFS
from hailtop.aiotools.s3asyncfs import S3AsyncFS
from hailtop.utils import tqdm

def referenced_schemes(transfers: List[Transfer]):
    def scheme_from_url(url):
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme
        if scheme == '':
            scheme = 'file'
        if scheme not in {'file', 'gs', 's3'}:
            raise ValueError(f'Unsupported scheme: {scheme}')
        return scheme
    return {
        scheme_from_url(url)
        for transfer in transfers
        for url in [transfer.src, transfer.dest]}


def filesystem_from_scheme(scheme, thread_pool=None, gcs_params=None):
    if scheme == 'file':
        assert thread_pool is not None
        return LocalAsyncFS(thread_pool)
    if scheme == 'gs':
        return GoogleStorageAsyncFS(params=gcs_params)
    if scheme == 's3':
        assert thread_pool is not None
        return S3AsyncFS(thread_pool)
    raise ValueError(f'Unsupported scheme: {scheme}')


async def copy(requester_pays_project: Optional[str],
               transfers: List[Transfer]
               ) -> None:
    gcs_params = {'userProject': requester_pays_project} if requester_pays_project else None
    schemes = referenced_schemes(transfers)
    default_scheme = 'file' if 'file' in schemes else None
    with ThreadPoolExecutor() as thread_pool:
        filesystems = [filesystem_from_scheme(s,
                                              thread_pool=thread_pool,
                                              gcs_params=gcs_params)
                       for s in schemes]
        async with RouterAsyncFS(default_scheme, filesystems) as fs:
            sema = asyncio.Semaphore(50)
            async with sema:
                with tqdm(desc='files', leave=False, position=0, unit='file') as tqdm_files, \
                     tqdm(desc='bytes', leave=False, position=1, unit='byte', unit_scale=True, smoothing=0.1) as tqdm_bytes:
                    copy_report = await fs.copy(sema, transfers, tqdm_files=tqdm_files, tqdm_bytes=tqdm_bytes)
                copy_report.summarize()


async def main() -> None:
    assert len(sys.argv) == 3
    requster_pays_project = json.loads(sys.argv[1])
    files = json.loads(sys.argv[2])

    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    collecting_stats = asyncio.Event()

    async def dump_stats():
        while True:
            done, pending = await asyncio.wait([asyncio.sleep(60), collecting_stats.wait()],
                                               return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                await t
            if collecting_stats.is_set():
                return
            ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
            ps.print_stats(10)
            pr.enable()

    stats_fut = asyncio.ensure_future(dump_stats())

    await copy(
        requster_pays_project,
        [Transfer(f['from'], f['to'], treat_dest_as=Transfer.DEST_IS_TARGET) for f in files]
    )
    pr.disable()
    collecting_stats.set()
    await stats_fut


if __name__ == '__main__':
    asyncio.run(main())
