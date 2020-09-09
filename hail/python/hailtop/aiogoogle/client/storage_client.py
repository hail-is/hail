from typing import Tuple, Any, Set, Optional
import asyncio
import urllib.parse
import aiohttp
from hailtop.aiotools import AsyncStream, AsyncFS, FeedableAsyncIterable
from multidict import CIMultiDictProxy
from .base_client import BaseClient


class InsertObjectStream(AsyncStream):
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
            print(self._value)


class GetObjectStream(AsyncStream):
    def __init__(self, resp):
        super().__init__()
        self._resp = resp
        self._content = resp.content

    def readable(self) -> bool:
        return True

    async def read(self, n: int = -1) -> bytes:
        assert not self._closed
        return await self._content.read(n)

    def headers(self) -> 'CIMultiDictProxy[str]':
        return self._resp.headers

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
        it: FeedableAsyncIterable[bytes] = FeedableAsyncIterable()
        kwargs['data'] = aiohttp.AsyncIterablePayload(it)
        request_task = asyncio.ensure_future(self._session.post(
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

        resp = await self._session.get(
            f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe="")}', **kwargs)
        return GetObjectStream(resp)

    async def get_object_metadata(self, bucket: str, name: str, **kwargs) -> Any:
        params = kwargs.get('params')
        assert not params or 'alt' not in params
        return await self.get(f'/b/{bucket}/o/{urllib.parse.quote(name, safe="")}', **kwargs)

    async def delete_object(self, bucket: str, name: str, **kwargs) -> None:
        await self.delete(f'/b/{bucket}/o/{urllib.parse.quote(name, safe="")}', **kwargs)

    async def _list_objects(self, bucket: str, **kwargs) -> Any:
        return await self.get(f'/b/{bucket}/o', **kwargs)


class GoogleStorageAsyncFS(AsyncFS):
    def __init__(self, storage_client: Optional[StorageClient] = None):
        if not storage_client:
            storage_client = StorageClient()
        self._storage_client = storage_client

    def schemes(self) -> Set[str]:
        return {'gs'}

    @staticmethod
    def _get_bucket_name(url: str) -> Tuple[str, str]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'gs':
            raise ValueError(f"invalid scheme, expected gs: {parsed.scheme}")

        name = parsed.path
        if name:
            assert name[0] == '/'
            name = name[1:]

        return (parsed.netloc, name)

    async def open(self, url: str, mode: str = 'r') -> AsyncStream:
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

        bucket, name = self._get_bucket_name(url)

        if 'r' in mode:
            return await self._storage_client.get_object(bucket, name)

        assert 'w' in mode
        return await self._storage_client.insert_object(bucket, name)

    async def mkdir(self, url: str) -> None:
        pass

    async def isfile(self, url: str) -> bool:
        try:
            bucket, name = self._get_bucket_name(url)
            await self._storage_client.get_object_metadata(bucket, name)
            return True
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return False
            raise

    async def isdir(self, url: str) -> bool:
        bucket, name = self._get_bucket_name(url)
        if not name.endswith('/'):
            name = f'{name}/'
        params = {
            'delimiter': '/',
            'includeTrailingDelimiter': 'true',
            'maxResults': 1,
            'prefix': name
        }
        data = await self._storage_client._list_objects(bucket, params=params)
        assert 'prefixes' not in data or data['prefixes'] is None
        items = data.get('items')
        return items is not None and len(items) > 0

    async def remove(self, url: str) -> None:
        bucket, name = self._get_bucket_name(url)
        await self._storage_client.delete_object(bucket, name)

    async def rmtree(self, url: str) -> None:
        bucket, name = self._get_bucket_name(url)
        if not name.endswith('/'):
            name = f'{name}/'
        params = {
            'prefix': name
        }
        done = False
        while not done:
            done = True
            data = await self._storage_client._list_objects(bucket, params=params)
            items = data.get('items')
            print(items)
            if items:
                for item in items:
                    await self._storage_client.delete_object(bucket, item['name'])
                    print('done: False')
                    done = False

    async def close(self) -> None:
        await self._storage_client.close()
        self._storage_client = None
