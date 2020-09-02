from typing import List
import asyncio
import urllib.parse
import aiohttp
from hailtop.aiotools import AsyncStream, AsyncFS, FeedableAsyncIterable
from .base_client import BaseClient


class InsertObjectStream(AsyncStream[bytes]):
    def __init__(self, it, request_task):
        super().__init__()
        self._it = it
        self._request_task = request_task
        self._value = None

    def writable(self):
        return True

    async def write(self, b):
        assert not self.closed
        await self._it.feed(b)
        return len(b)

    async def _wait_closed(self):
        await self._it.stop()
        async with await self._request_task as resp:
            self._value = await resp.json()


class GetObjectStream(AsyncStream[bytes]):
    def __init__(self, resp):
        super().__init__()
        self._resp = resp
        self._content = resp.content

    def readable(self) -> bool:
        return True

    async def read(self, n: int = -1) -> bytes:
        assert not self._closed
        return await self._content.read(n)

    async def _wait_closed(self) -> None:
        self._content = None
        self._resp.release()
        self._resp = None


class StorageClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__('https://storage.googleapis.com/storage/v1', **kwargs)

    # docs:
    # https://cloud.google.com/storage/docs/json_api/v1

    async def insert_object(self, bucket: str, name: str, **kwargs) -> InsertObjectStream:
        if 'params' in kwargs:
            params = kwargs['params']
        else:
            params = {}
            kwargs['params'] = params
        assert 'name' not in params
        params['name'] = name
        assert 'uploadType' not in params
        params['uploadType'] = 'media'

        assert 'data' not in kwargs
        it: FeedableAsyncIterable[bytes]  = FeedableAsyncIterable()
        kwargs['data'] = aiohttp.AsyncIterablePayload(it)
        request_task = asyncio.create_task(self._session.post(
            f'https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o',
            **kwargs))
        return InsertObjectStream(it, request_task)

    async def get_object(self, bucket: str, name: str, **kwargs) -> GetObjectStream:
        if 'params' in kwargs:
            params = kwargs['params']
        else:
            params = {}
            kwargs['params'] = params
        assert 'alt' not in params
        params['alt'] = 'media'

        resp = await self._session.get(f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}', **kwargs)
        return GetObjectStream(resp)

    async def get_object_metadata(self, bucket: str, name: str, **kwargs):
        params = kwargs.get('params')
        assert not params or 'alt' not in params
        return await self.get(f'/b/{bucket}/o/{name}', **kwargs)


class GoogleStorageAsyncFS(AsyncFS):
    def __init__(self, storage_client: StorageClient = None):
        if not storage_client:
            storage_client = StorageClient()
        self._storage_client = storage_client

    def schemes(self) -> List[str]:
        return ['gs']

    async def open(self, url: str, mode: str = 'r') -> AsyncStream[bytes]:
        if not all(c in 'rwxabt+' for c in mode):
            raise ValueError(f"invalid mode: {mode}")
        if 't' in mode and 'b' in mode:
            raise ValueError(f"can't have text and binary mode at once: {mode}")
        if 'b' not in mode:
            raise ValueError(f"text mode not supported: {mode}")
        if 'x' in mode:
            raise ValueError(f"exclusive creation not supported: {mode}")
        if 'a' in mode:
            raise ValueError(f"append mode not supported: {mode}")
        if '+' in mode:
            raise ValueError(f"updating not supported: {mode}")
        if ('r' in mode) + ('w' in mode) != 1:
            raise ValueError(f"must have exactly one of read/write mode: {mode}")

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'gs':
            raise ValueError(f"invalid scheme, expected gs: {parsed.scheme}")
        bucket = parsed.netloc

        name = parsed.path
        if name:
            assert name[0] == '/'
            name = name[1:]

        if 'r' in mode:
            return await self._storage_client.get_object(bucket, name)

        assert 'w' in mode
        return await self._storage_client.insert_object(bucket, name)

    async def close(self):
        await self._storage_client.close()
        self._storage_client = None
