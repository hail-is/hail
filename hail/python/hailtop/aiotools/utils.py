from typing import Dict, Optional, Any, Callable
import urllib
from concurrent.futures import ThreadPoolExecutor

from ..aiocloud.aiogoogle import GoogleStorageAsyncFS
from ..aiocloud.aioaws import S3AsyncFS
from ..aiocloud.aioazure import AzureAsyncFS

from .fs import AsyncFS, LocalAsyncFS


def scheme_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme
    if scheme == '':
        scheme = 'file'
    if scheme not in {'file', 'gs', 's3', 'hail-az'}:
        raise ValueError(f'Unsupported scheme: {scheme}')
    return scheme


def filesystem_from_scheme(scheme: str,
                           thread_pool: Optional[ThreadPoolExecutor] = None,
                           gcs_params: Optional[Dict[str, Any]] = None
                           ) -> AsyncFS:
    if scheme == 'file':
        assert thread_pool is not None
        return LocalAsyncFS(thread_pool)
    if scheme == 'gs':
        return GoogleStorageAsyncFS(params=gcs_params)
    if scheme == 's3':
        assert thread_pool is not None
        return S3AsyncFS(thread_pool)
    if scheme == 'hail-az':
        return AzureAsyncFS()
    raise ValueError(f'Unsupported scheme: {scheme}')


def make_tqdm_listener(pbar) -> Callable[[int], None]:
    def listener(delta):
        if pbar.total is None:
            pbar.total = 0
        if delta > 0:
            pbar.total += delta
            pbar.refresh()
        if delta < 0:
            pbar.update(-delta)
    return listener
