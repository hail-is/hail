import os
from typing import Tuple, Any, Set, Optional, MutableMapping, Dict, AsyncIterator, cast, Type, List, Coroutine, ClassVar
from types import TracebackType
from multidict import CIMultiDictProxy  # pylint: disable=unused-import
import sys
import logging
import asyncio
import urllib.parse
import aiohttp
import datetime
from hailtop import timex
from hailtop.utils import secret_alnum_string, OnlineBoundedGather2, TransientError, retry_transient_errors
from hailtop.aiotools.fs import (
    FileStatus,
    FileListEntry,
    ReadableStream,
    WritableStream,
    AsyncFS,
    AsyncFSURL,
    AsyncFSFactory,
    FileAndDirectoryError,
    MultiPartCreate,
    UnexpectedEOFError,
)
from hailtop.aiotools import FeedableAsyncIterable, WriteBuffer

from .base_client import GoogleBaseClient
from ..session import GoogleSession
from ..credentials import GoogleCredentials
from ..user_config import get_gcs_requester_pays_configuration, GCSRequesterPaysConfiguration

log = logging.getLogger(__name__)


class PageIterator:
    def __init__(self, client: 'GoogleBaseClient', path: str, request_kwargs: MutableMapping[str, Any]):
        if 'params' in request_kwargs:
            request_params = request_kwargs['params']
            del request_kwargs['params']
        else:
            request_params = {}
        self._client = client
        self._path = path
        self._request_params = request_params
        self._request_kwargs = request_kwargs
        self._page = None

    def __aiter__(self) -> 'PageIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            assert 'pageToken' not in self._request_params
            self._page = await retry_transient_errors(
                self._client.get, self._path, params=self._request_params, **self._request_kwargs
            )
            return self._page

        next_page_token = self._page.get('nextPageToken')
        if next_page_token is not None:
            self._request_params['pageToken'] = next_page_token
            self._page = await retry_transient_errors(
                self._client.get, self._path, params=self._request_params, **self._request_kwargs
            )
            return self._page

        raise StopAsyncIteration


class InsertObjectStream(WritableStream):
    def __init__(self, it: FeedableAsyncIterable[bytes], request_task: asyncio.Task[aiohttp.ClientResponse]):
        super().__init__()
        self._it = it
        self._request_task = request_task
        self._value = None

    async def write(self, b):
        assert not self.closed

        fut = asyncio.ensure_future(self._it.feed(b))
        try:
            await asyncio.wait([fut, self._request_task], return_when=asyncio.FIRST_COMPLETED)
            if fut.done() and not fut.cancelled():
                if exc := fut.exception():
                    raise exc
                return len(b)
            raise ValueError('request task finished early')
        finally:
            fut.cancel()

    async def _wait_closed(self):
        fut = asyncio.ensure_future(self._it.stop())
        try:
            await asyncio.wait([fut, self._request_task], return_when=asyncio.FIRST_COMPLETED)
            async with await self._request_task as resp:
                self._value = await resp.json()
        finally:
            if fut.done() and not fut.cancelled():
                if exc := fut.exception():
                    raise exc
            else:
                fut.cancel()


class _TaskManager:
    def __init__(self, coro: Coroutine[Any, Any, Any], closable: bool = False):
        self._coro = coro
        self._task: Optional[asyncio.Task[Any]] = None
        self._closable = closable

    async def __aenter__(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._coro)
        return self._task

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        assert self._task is not None

        if not self._task.done():
            if exc_val:
                self._task.cancel()
                try:
                    value = await self._task
                    if self._closable:
                        value.close()
                except:
                    _, exc, _ = sys.exc_info()
                    if exc is not exc_val:
                        log.warning('dropping preempted task exception', exc_info=True)
            else:
                value = await self._task
                if self._closable:
                    value.close()
        else:
            value = await self._task
            if self._closable:
                value.close()


class ResumableInsertObjectStream(WritableStream):
    def __init__(self, session: GoogleSession, session_url: str, chunk_size: int):
        super().__init__()
        self._session = session
        self._session_url = session_url
        self._write_buffer = WriteBuffer()
        self._broken = False
        self._done = False
        self._chunk_size = chunk_size

    @staticmethod
    def _range_upper(range):
        range = range.split('/', 1)[0]
        split_range = range.split('-')
        assert len(split_range) == 2
        return int(split_range[1])

    async def _write_chunk_1(self):
        assert not self._done
        assert self._closed or self._write_buffer.size() >= self._chunk_size

        if self._closed:
            total_size = self._write_buffer.offset() + self._write_buffer.size()
            total_size_str = str(total_size)
        else:
            total_size = None
            total_size_str = '*'

        if self._broken:
            # If the last request was unsuccessful, we are out of sync
            # with the server and we don't know what byte to send
            # next.  Perform a status check to find out.  See:
            # https://cloud.google.com/storage/docs/performing-resumable-uploads#status-check

            # note: this retries
            async with await self._session.put(
                self._session_url,
                headers={'Content-Length': '0', 'Content-Range': f'bytes */{total_size_str}'},
                raise_for_status=False,
            ) as resp:
                if resp.status >= 200 and resp.status < 300:
                    assert self._closed
                    assert total_size is not None
                    self._write_buffer.advance_offset(total_size)
                    assert self._write_buffer.size() == 0
                    self._done = True
                    return
                if resp.status == 308:
                    range = resp.headers.get('Range')
                    if range is not None:
                        new_offset = self._range_upper(range) + 1
                    else:
                        new_offset = 0
                    self._write_buffer.advance_offset(new_offset)
                    self._broken = False
                else:
                    assert resp.status >= 400
                    resp.raise_for_status()
                    assert False

        assert not self._broken
        self._broken = True

        offset = self._write_buffer.offset()
        if self._closed:
            n = self._write_buffer.size()
        # status check can advance the offset, so there might not be a
        # full chunk available to write
        elif self._write_buffer.size() < self._chunk_size:
            return
        else:
            n = self._chunk_size
        if n > 0:
            range = f'bytes {offset}-{offset + n - 1}/{total_size_str}'
        else:
            range = f'bytes */{total_size_str}'

        # Upload a single chunk.  See:
        # https://cloud.google.com/storage/docs/performing-resumable-uploads#chunked-upload
        it: FeedableAsyncIterable[bytes] = FeedableAsyncIterable()
        async with _TaskManager(
            self._session.put(
                self._session_url,
                data=aiohttp.AsyncIterablePayload(it),
                headers={'Content-Length': f'{n}', 'Content-Range': range},
                raise_for_status=False,
                retry=False,
            ),
            closable=True,
        ) as put_task:
            with self._write_buffer.chunks(n) as chunks:
                for chunk in chunks:
                    async with _TaskManager(it.feed(chunk)) as feed_task:
                        done, _ = await asyncio.wait([put_task, feed_task], return_when=asyncio.FIRST_COMPLETED)
                        if feed_task not in done:
                            msg = 'resumable upload chunk PUT request finished before writing data'
                            log.warning(msg)
                            raise TransientError(msg)

            await it.stop()

            resp = await put_task
            if resp.status >= 200 and resp.status < 300:
                assert self._closed
                assert total_size is not None
                self._write_buffer.advance_offset(total_size)
                assert self._write_buffer.size() == 0
                self._done = True
                return

            if resp.status == 308:
                range = resp.headers['Range']
                self._write_buffer.advance_offset(self._range_upper(range) + 1)
                self._broken = False
                return

            assert resp.status >= 400
            resp.raise_for_status()
            assert False

    async def _write_chunk(self):
        await retry_transient_errors(self._write_chunk_1)

    async def write(self, b):
        assert not self._closed
        assert self._write_buffer.size() < self._chunk_size
        self._write_buffer.append(b)
        while self._write_buffer.size() >= self._chunk_size:
            await self._write_chunk()
        assert self._write_buffer.size() < self._chunk_size
        return len(b)

    async def _wait_closed(self):
        assert self._closed
        assert self._write_buffer.size() < self._chunk_size
        while not self._done:
            await self._write_chunk()
        assert self._done and self._write_buffer.size() == 0


class GetObjectStream(ReadableStream):
    def __init__(self, resp: aiohttp.ClientResponse):
        super().__init__()
        self._resp: Optional[aiohttp.ClientResponse] = resp
        self._content: Optional[aiohttp.StreamReader] = resp.content

    # https://docs.aiohttp.org/en/stable/streams.html#aiohttp.StreamReader.read
    # Read up to n bytes. If n is not provided, or set to -1, read until EOF
    # and return all read bytes.
    async def read(self, n: int = -1) -> bytes:
        assert not self._closed and self._content is not None
        return await self._content.read(n)

    async def readexactly(self, n: int) -> bytes:
        assert not self._closed and n >= 0 and self._content is not None
        try:
            return await self._content.readexactly(n)
        except asyncio.IncompleteReadError as e:
            raise UnexpectedEOFError() from e

    def headers(self) -> 'CIMultiDictProxy[str]':
        assert self._resp is not None

        return self._resp.headers

    async def _wait_closed(self) -> None:
        assert self._resp is not None
        assert self._content is not None

        self._content = None
        self._resp.close()
        self._resp = None


class GoogleStorageClient(GoogleBaseClient):
    def __init__(self, gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None, **kwargs):
        if 'timeout' not in kwargs and 'http_session' not in kwargs:
            # Around May 2022, GCS started timing out a lot with our default 5s timeout
            kwargs['timeout'] = aiohttp.ClientTimeout(total=20)
        super().__init__('https://storage.googleapis.com/storage/v1', **kwargs)
        self._gcs_requester_pays_configuration = get_gcs_requester_pays_configuration(
            gcs_requester_pays_configuration=gcs_requester_pays_configuration
        )

    async def bucket_info(self, bucket: str) -> Dict[str, Any]:
        """
        See `the GCS API docs https://cloud.google.com/storage/docs/json_api/v1/buckets`_ for the list of bucket
        properties in the response.
        """
        kwargs: Dict[str, Any] = {}
        self._update_params_with_user_project(kwargs, bucket)
        return await self.get(f'/b/{bucket}', **kwargs)

    # docs:
    # https://cloud.google.com/storage/docs/json_api/v1

    async def insert_object(self, bucket: str, name: str, **kwargs) -> WritableStream:
        assert name
        # Insert an object.  See:
        # https://cloud.google.com/storage/docs/json_api/v1/objects/insert
        if 'params' in kwargs:
            params = kwargs['params']
        else:
            params = {}
            kwargs['params'] = params
        assert 'name' not in params
        params['name'] = name
        self._update_params_with_user_project(kwargs, bucket)

        assert 'data' not in params

        upload_type = params.get('uploadType')
        if not upload_type:
            upload_type = 'resumable'
            params['uploadType'] = upload_type

        if upload_type == 'media':
            it: FeedableAsyncIterable[bytes] = FeedableAsyncIterable()
            kwargs['data'] = aiohttp.AsyncIterablePayload(it)
            request_task = asyncio.create_task(
                self._session.post(
                    f'https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o', retry=False, **kwargs
                )
            )
            return InsertObjectStream(it, request_task)

        # Write using resumable uploads.  See:
        # https://cloud.google.com/storage/docs/performing-resumable-uploads
        assert upload_type == 'resumable'
        chunk_size = kwargs.get('bufsize', 8 * 1024 * 1024)

        async with await self._session.post(
            f'https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o', **kwargs
        ) as resp:
            session_url = resp.headers['Location']
        return ResumableInsertObjectStream(self._session, session_url, chunk_size)

    async def get_object(self, bucket: str, name: str, **kwargs) -> GetObjectStream:
        assert name
        if 'params' in kwargs:
            params = kwargs['params']
        else:
            params = {}
            kwargs['params'] = params
        assert 'alt' not in params
        params['alt'] = 'media'
        self._update_params_with_user_project(kwargs, bucket)

        try:
            resp = await self._session.get(
                f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe="")}', **kwargs
            )
            return GetObjectStream(resp)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise FileNotFoundError from e
            if e.status == 416:
                raise UnexpectedEOFError from e
            raise

    async def get_object_metadata(self, bucket: str, name: str, **kwargs) -> Dict[str, str]:
        assert name
        assert 'params' not in kwargs or 'alt' not in kwargs['params']
        self._update_params_with_user_project(kwargs, bucket)
        return cast(Dict[str, str], await self.get(f'/b/{bucket}/o/{urllib.parse.quote(name, safe="")}', **kwargs))

    async def delete_object(self, bucket: str, name: str, **kwargs) -> None:
        assert name
        self._update_params_with_user_project(kwargs, bucket)
        await self.delete(f'/b/{bucket}/o/{urllib.parse.quote(name, safe="")}', **kwargs)

    async def list_objects(self, bucket: str, **kwargs) -> PageIterator:
        self._update_params_with_user_project(kwargs, bucket)
        return PageIterator(self, f'/b/{bucket}/o', kwargs)

    async def compose(self, bucket: str, names: List[str], destination: str, **kwargs) -> None:
        assert destination
        n = len(names)
        if n == 0:
            raise ValueError('no components in compose')
        if n > 32:
            raise ValueError(f'too many components in compose, maximum of 32: {n}')
        assert 'json' not in kwargs
        assert 'body' not in kwargs
        kwargs['json'] = {'sourceObjects': [{'name': name} for name in names]}
        self._update_params_with_user_project(kwargs, bucket)
        await self.post(f'/b/{bucket}/o/{urllib.parse.quote(destination, safe="")}/compose', **kwargs)

    def _update_params_with_user_project(self, request_kwargs, bucket):
        if 'params' not in request_kwargs:
            request_kwargs['params'] = {}
        params = request_kwargs['params']
        config = self._gcs_requester_pays_configuration
        if isinstance(config, str):
            params.update({'userProject': config})
        elif isinstance(config, tuple):
            project, buckets = config
            if bucket in buckets:
                params.update({'userProject': project})


class GetObjectFileStatus(FileStatus):
    def __init__(self, items: Dict[str, str]):
        self._items = items

    async def size(self) -> int:
        return int(self._items['size'])

    def time_created(self) -> datetime.datetime:
        return timex.parse_rfc3339(self._items['timeCreated'])

    def time_modified(self) -> datetime.datetime:
        return timex.parse_rfc3339(self._items['updated'])

    async def __getitem__(self, key: str) -> str:
        return self._items[key]


class GoogleStorageFileListEntry(FileListEntry):
    def __init__(self, bucket: str, name: str, items: Optional[Dict[str, Any]]):
        self._bucket = bucket
        self._name = name
        self._items = items
        self._status: Optional[GetObjectFileStatus] = None

    def name(self) -> str:
        return os.path.basename(self._name)

    async def url(self) -> str:
        return f'gs://{self._bucket}/{self._name}'

    async def is_file(self) -> bool:
        return self._items is not None

    async def is_dir(self) -> bool:
        return self._items is None

    async def status(self) -> FileStatus:
        if self._status is None:
            if self._items is None:
                raise IsADirectoryError(await self.url())
            self._status = GetObjectFileStatus(self._items)
        return self._status


class GoogleStorageMultiPartCreate(MultiPartCreate):
    def __init__(self, sema: asyncio.Semaphore, fs: 'GoogleStorageAsyncFS', dest_url: str, num_parts: int):
        self._sema = sema
        self._fs = fs
        self._dest_url = dest_url
        self._num_parts = num_parts
        bucket, dest_name = fs.get_bucket_and_name(dest_url)
        self._bucket = bucket
        self._dest_name = dest_name

        # compute dest_dirname so gs://{bucket}/{dest_dirname}file
        # refers to a file in dest_dirname with no double slashes
        dest_dirname = os.path.dirname(dest_name)
        if dest_dirname:
            dest_dirname = dest_dirname + '/'
        self._dest_dirname = dest_dirname

        self._token = secret_alnum_string()

    def _tmp_name(self, filename: str) -> str:
        return f'{self._dest_dirname}_/{self._token}/{filename}'

    def _part_name(self, number: int) -> str:
        return self._tmp_name(f'part-{number}')

    async def create_part(
        self, number: int, start: int, size_hint: Optional[int] = None
    ) -> WritableStream:  # pylint: disable=unused-argument
        part_name = self._part_name(number)
        params = {'uploadType': 'media'}
        return await self._fs._storage_client.insert_object(self._bucket, part_name, params=params)

    async def __aenter__(self) -> 'GoogleStorageMultiPartCreate':
        return self

    async def _compose(self, names: List[str], dest_name: str):
        await self._fs._storage_client.compose(self._bucket, names, dest_name)

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        async with OnlineBoundedGather2(self._sema) as pool:
            try:
                if exc_val is not None:
                    return

                async def tree_compose(names, dest_name):
                    n = len(names)
                    assert n > 0
                    if n <= 32:
                        await self._compose(names, dest_name)
                        return

                    q, r = divmod(n, 32)
                    i = 0
                    p = 0
                    chunks = []
                    while i < 32:
                        # each chunk gets q, and the first r get one more
                        chunk_size = q
                        if i < r:
                            chunk_size += 1
                        chunks.append(names[p : p + chunk_size])
                        p += chunk_size
                        i += 1
                    assert p == n
                    assert len(chunks) == 32

                    chunk_names = [self._tmp_name(f'chunk-{secret_alnum_string()}') for _ in range(32)]

                    chunk_tasks = [pool.call(tree_compose, c, n) for c, n in zip(chunks, chunk_names)]

                    await pool.wait(chunk_tasks)

                    await self._compose(chunk_names, dest_name)

                    for name in chunk_names:
                        await pool.call(self._fs._remove_doesnt_exist_ok, f'gs://{self._bucket}/{name}')

                await tree_compose([self._part_name(i) for i in range(self._num_parts)], self._dest_name)
            finally:
                await self._fs.rmtree(self._sema, f'gs://{self._bucket}/{self._dest_dirname}_/{self._token}')


class GoogleStorageAsyncFSURL(AsyncFSURL):
    def __init__(self, bucket: str, path: str):
        self._bucket = bucket
        self._path = path

    @property
    def bucket_parts(self) -> List[str]:
        return [self._bucket]

    @property
    def path(self) -> str:
        return self._path

    @property
    def query(self) -> Optional[str]:
        return None

    @property
    def scheme(self) -> str:
        return 'gs'

    def with_path(self, path) -> 'GoogleStorageAsyncFSURL':
        return GoogleStorageAsyncFSURL(self._bucket, path)

    def __str__(self) -> str:
        return f'gs://{self._bucket}/{self._path}'


class GoogleStorageAsyncFS(AsyncFS):
    schemes: ClassVar[Set[str]] = {'gs'}

    def __init__(
        self,
        *,
        storage_client: Optional[GoogleStorageClient] = None,
        bucket_allow_list: Optional[List[str]] = None,
        **kwargs,
    ):
        if not storage_client:
            storage_client = GoogleStorageClient(**kwargs)
        self._storage_client = storage_client
        if bucket_allow_list is None:
            bucket_allow_list = []
        self.allowed_storage_locations = bucket_allow_list

    def storage_location(self, uri: str) -> str:
        return self.get_bucket_and_name(uri)[0]

    async def is_hot_storage(self, location: str) -> bool:
        """
        See `the GCS API docs https://cloud.google.com/storage/docs/storage-classes`_ for a list of possible storage
        classes.

        Raises
        ------
        :class:`aiohttp.ClientResponseError`
            If the specified bucket does not exist, or if the account being used to access GCS does not have permission
            to read the bucket's default storage policy.
        """
        return (await self._storage_client.bucket_info(location))["storageClass"].lower() in (
            "standard",
            "regional",
            "multi_regional",
        )

    @staticmethod
    def valid_url(url: str) -> bool:
        return url.startswith('gs://')

    @staticmethod
    def parse_url(url: str) -> GoogleStorageAsyncFSURL:
        return GoogleStorageAsyncFSURL(*GoogleStorageAsyncFS.get_bucket_and_name(url))

    @staticmethod
    def get_bucket_and_name(url: str) -> Tuple[str, str]:
        colon_index = url.find(':')
        if colon_index == -1:
            raise ValueError(f'invalid URL: {url}')

        scheme = url[:colon_index]
        if scheme != 'gs':
            raise ValueError(f'invalid scheme, expected gs: {scheme}')

        rest = url[(colon_index + 1) :]
        if not rest.startswith('//'):
            raise ValueError(f'Google Cloud Storage URI must be of the form: gs://bucket/path, found: {url}')

        end_of_bucket = rest.find('/', 2)
        if end_of_bucket != -1:
            bucket = rest[2:end_of_bucket]
            name = rest[(end_of_bucket + 1) :]
        else:
            bucket = rest[2:]
            name = ''

        return (bucket, name)

    async def open(self, url: str) -> GetObjectStream:
        bucket, name = self.get_bucket_and_name(url)
        return await self._storage_client.get_object(bucket, name)

    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> GetObjectStream:
        bucket, name = self.get_bucket_and_name(url)
        range_str = f'bytes={start}-'
        if length is not None:
            assert length >= 1
            range_str += str(start + length - 1)
        return await self._storage_client.get_object(bucket, name, headers={'Range': range_str})

    async def create(self, url: str, *, retry_writes: bool = True) -> WritableStream:
        bucket, name = self.get_bucket_and_name(url)
        params = {'uploadType': 'resumable' if retry_writes else 'media'}
        return await self._storage_client.insert_object(bucket, name, params=params)

    async def multi_part_create(
        self, sema: asyncio.Semaphore, url: str, num_parts: int
    ) -> GoogleStorageMultiPartCreate:
        return GoogleStorageMultiPartCreate(sema, self, url, num_parts)

    async def staturl(self, url: str) -> str:
        return await self._staturl_parallel_isfile_isdir(url)

    async def mkdir(self, url: str) -> None:
        pass

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        pass

    async def statfile(self, url: str) -> GetObjectFileStatus:
        try:
            bucket, name = self.get_bucket_and_name(url)
            return GetObjectFileStatus(await self._storage_client.get_object_metadata(bucket, name))
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise FileNotFoundError(url) from e
            raise

    async def _listfiles_recursive(self, bucket: str, name: str) -> AsyncIterator[FileListEntry]:
        assert not name or name.endswith('/')
        params = {'prefix': name}
        async for page in await self._storage_client.list_objects(bucket, params=params):
            prefixes = page.get('prefixes')
            assert not prefixes

            items = page.get('items')
            if items is not None:
                for item in page['items']:
                    yield GoogleStorageFileListEntry(bucket, item['name'], item)

    async def _listfiles_flat(self, bucket: str, name: str) -> AsyncIterator[FileListEntry]:
        assert not name or name.endswith('/')
        params = {'prefix': name, 'delimiter': '/', 'includeTrailingDelimiter': 'true'}
        async for page in await self._storage_client.list_objects(bucket, params=params):
            prefixes = page.get('prefixes')
            if prefixes:
                for prefix in prefixes:
                    assert prefix.endswith('/')
                    yield GoogleStorageFileListEntry(bucket, prefix, None)

            items = page.get('items')
            if items:
                for item in page['items']:
                    yield GoogleStorageFileListEntry(bucket, item['name'], item)

    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        bucket, name = self.get_bucket_and_name(url)
        if name and not name.endswith('/'):
            name = f'{name}/'

        if recursive:
            it = self._listfiles_recursive(bucket, name)
        else:
            it = self._listfiles_flat(bucket, name)

        it = it.__aiter__()
        try:
            first_entry = await it.__anext__()
        except StopAsyncIteration:
            raise FileNotFoundError(url)  # pylint: disable=raise-missing-from

        async def should_yield(entry):
            url = await entry.url()
            if url.endswith('/') and await entry.is_file():
                if not exclude_trailing_slash_files:
                    return True

                stat = await entry.status()
                if await stat.size() != 0:
                    raise FileAndDirectoryError(url)
                return False
            return True

        async def cons(first_entry, it) -> AsyncIterator[FileListEntry]:
            if await should_yield(first_entry):
                yield first_entry
            try:
                while True:
                    next_entry = await it.__anext__()
                    if await should_yield(next_entry):
                        yield next_entry
            except StopAsyncIteration:
                pass

        return cons(first_entry, it)

    async def isfile(self, url: str) -> bool:
        try:
            bucket, name = self.get_bucket_and_name(url)
            # if name is empty, get_object_metadata behaves like list objects
            # the urls are the same modulo the object name
            if not name:
                return False
            await self._storage_client.get_object_metadata(bucket, name)
            return True
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return False
            raise

    async def isdir(self, url: str) -> bool:
        bucket, name = self.get_bucket_and_name(url)
        assert not name or name.endswith('/'), name
        params = {'prefix': name, 'delimiter': '/', 'includeTrailingDelimiter': 'true', 'maxResults': 1}
        async for page in await self._storage_client.list_objects(bucket, params=params):
            prefixes = page.get('prefixes')
            items = page.get('items')
            return bool(prefixes or items)
        assert False  # unreachable

    async def remove(self, url: str) -> None:
        bucket, name = self.get_bucket_and_name(url)
        try:
            await self._storage_client.delete_object(bucket, name)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise FileNotFoundError(url) from e
            raise

    async def close(self) -> None:
        if hasattr(self, '_storage_client'):
            await self._storage_client.close()
            del self._storage_client


class GoogleStorageAsyncFSFactory(AsyncFSFactory[GoogleStorageAsyncFS]):
    def from_credentials_data(self, credentials_data: dict) -> GoogleStorageAsyncFS:
        return GoogleStorageAsyncFS(credentials=GoogleCredentials.from_credentials_data(credentials_data))

    def from_credentials_file(self, credentials_file: str) -> GoogleStorageAsyncFS:
        return GoogleStorageAsyncFS(credentials=GoogleCredentials.from_file(credentials_file))

    def from_default_credentials(self) -> GoogleStorageAsyncFS:
        return GoogleStorageAsyncFS(credentials=GoogleCredentials.default_credentials())
