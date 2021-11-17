from typing import List, Optional
import json
import urllib
import asyncio
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from hailtop.aiotools import RouterAsyncFS, Transfer, Copier
from hailtop.utils import tqdm


def referenced_schemes(transfers: List[Transfer]):
    def scheme_from_url(url):
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme
        if scheme == '':
            scheme = 'file'
        if scheme not in {'file', 'gs', 's3', 'hail-az'}:
            raise ValueError(f'Unsupported scheme: {scheme}')
        return scheme
    return {
        scheme_from_url(url)
        for transfer in transfers
        for url in [transfer.src, transfer.dest]}


def make_tqdm_listener(pbar):
    def listener(delta):
        if pbar.total is None:
            pbar.total = 0
        if delta > 0:
            pbar.total += delta
            pbar.refresh()
        if delta < 0:
            pbar.update(-delta)
    return listener


async def copy(*,
               local_kwargs: Optional[dict] = None,
               gcs_kwargs: Optional[dict] = None,
               azure_kwargs: Optional[dict] = None,
               s3_kwargs: Optional[dict] = None,
               transfers: List[Transfer]
               ) -> None:

    schemes = referenced_schemes(transfers)
    default_scheme = 'file' if 'file' in schemes else None

    with ThreadPoolExecutor() as thread_pool:
        if local_kwargs is None:
            local_kwargs = {}
        if 'thread_pool' not in local_kwargs:
            local_kwargs['thread_pool'] = thread_pool

        if s3_kwargs is None:
            s3_kwargs = {}
        if 'thread_pool' not in s3_kwargs:
            s3_kwargs['thread_pool'] = thread_pool

        async with RouterAsyncFS(default_scheme,
                                 local_kwargs=local_kwargs,
                                 gcs_kwargs=gcs_kwargs,
                                 azure_kwargs=azure_kwargs,
                                 s3_kwargs=s3_kwargs) as fs:
            sema = asyncio.Semaphore(50)
            async with sema:
                with tqdm(desc='files', leave=False, position=0, unit='file') as file_pbar, \
                     tqdm(desc='bytes', leave=False, position=1, unit='byte', unit_scale=True, smoothing=0.1) as byte_pbar:
                    copy_report = await Copier.copy(
                        fs,
                        sema,
                        transfers,
                        files_listener=make_tqdm_listener(file_pbar),
                        bytes_listener=make_tqdm_listener(byte_pbar))
                copy_report.summarize()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Hail copy tool')
    parser.add_argument('requester_pays_project', type=str,
                        help='a JSON string indicating the Google project to which to charge egress costs')
    parser.add_argument('files', type=str,
                        help='a JSON array of JSON objects indicating from where and to where to copy files')
    parser.add_argument('-v', '--verbose', action='store_const',
                        const=True, default=False,
                        help='show logging information')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig()
        logging.root.setLevel(logging.INFO)
    requester_pays_project = json.loads(args.requester_pays_project)
    files = json.loads(args.files)

    gcs_kwargs = {'project': requester_pays_project}

    await copy(
        gcs_kwargs=gcs_kwargs,
        transfers=[Transfer(f['from'], f['to'], treat_dest_as=Transfer.DEST_IS_TARGET) for f in files]
    )


if __name__ == '__main__':
    asyncio.run(main())
