from typing import Any, AsyncContextManager, AsyncIterator, Dict, List, Optional, Set, Tuple, Type
from types import TracebackType

import os
import re
import asyncio
import urllib
import secrets
import logging

from azure.storage.blob import BlobProperties
from azure.storage.blob.aio import BlobClient, ContainerClient, BlobServiceClient, StorageStreamDownloader
from azure.storage.blob.aio._list_blobs_helper import BlobPrefix
import azure.core.exceptions

from hailtop.utils import retry_transient_errors, flatten
from hailtop.aiotools import WriteBuffer
from hailtop.aiotools.fs import (AsyncFS, ReadableStream, WritableStream, MultiPartCreate, FileListEntry, FileStatus,
                                 FileAndDirectoryError, UnexpectedEOFError)

from .credentials import AzureCredentials


logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
logger.setLevel(logging.WARNING)


class AzureWritableStream(WritableStream):
    def __init__(self, client: BlobClient, block_ids: List[str], *, chunk_size: Optional[int] = None):
        super().__init__()
        self.client = client
        self._chunk_size = chunk_size or 256 * 1024
        self.block_ids = block_ids
        self._write_buffer = WriteBuffer()
        self._done = False

    async def _write_chunk(self):
        await retry_transient_errors(self._write_chunk_1)

    async def _write_chunk_1(self):
        if self._closed:
            n = self._write_buffer.size()
        elif self._write_buffer.size() < self._chunk_size:
            return
        else:
            n = self._chunk_size

        block_id = secrets.token_urlsafe(32)
        await self.client.stage_block(block_id, self._write_buffer.chunks(n))
        self.block_ids.append(block_id)

        new_offset = self._write_buffer.offset() + n
        self._write_buffer.advance_offset(new_offset)

    async def write(self, b: bytes) -> int:
        assert not self._closed
        assert self._write_buffer.size() < self._chunk_size
        self._write_buffer.append(b)
        while self._write_buffer.size() >= self._chunk_size:
            await self._write_chunk()
        assert self._write_buffer.size() < self._chunk_size
        return len(b)

    async def _wait_closed(self) -> None:
        assert self._closed
        assert self._write_buffer.size() < self._chunk_size
        while self._write_buffer.size() > 0:
            await self._write_chunk()


class AzureCreatePartManager(AsyncContextManager[WritableStream]):
    def __init__(self, client: BlobClient, block_ids: List[str]):
        self.client = client
        self.block_ids = block_ids
        self._writable_stream = AzureWritableStream(self.client, self.block_ids)

    async def __aenter__(self) -> 'AzureWritableStream':
        return self._writable_stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._writable_stream.wait_closed()


class AzureMultiPartCreate(MultiPartCreate):
    def __init__(self, sema: asyncio.Semaphore, client: BlobClient, num_parts: int):
        self._sema = sema
        self._client = client
        self._block_ids: List[List[str]] = [[] for _ in range(num_parts)]

    async def create_part(self, number: int, start: int, size_hint: Optional[int] = None) -> AsyncContextManager[WritableStream]:  # pylint: disable=unused-argument
        return AzureCreatePartManager(self._client, self._block_ids[number])

    async def __aenter__(self) -> 'AzureMultiPartCreate':
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        try:
            await self._client.commit_block_list(flatten(self._block_ids))
        except:
            try:
                await self._client.delete_blob()
            except azure.core.exceptions.ResourceNotFoundError:
                pass
            raise


class AzureCreateManager(AsyncContextManager[WritableStream]):
    def __init__(self, client: BlobClient):
        self._client = client
        self._block_ids: List[str] = []
        self._writable_stream: Optional[AzureWritableStream] = None

    async def __aenter__(self) -> WritableStream:
        self._writable_stream = AzureWritableStream(self._client, self._block_ids)
        return self._writable_stream

    async def __aexit__(
            self, exc_type: Optional[Type[BaseException]] = None,
            exc_value: Optional[BaseException] = None,
            exc_traceback: Optional[TracebackType] = None) -> None:
        if self._writable_stream:
            await self._writable_stream.wait_closed()

            try:
                await self._client.commit_block_list(self._block_ids)
            except:
                try:
                    await self._client.delete_blob()
                except azure.core.exceptions.ResourceNotFoundError:
                    pass
                raise


class AzureReadableStream(ReadableStream):
    def __init__(self, client: BlobClient, offset: Optional[int] = None):
        super().__init__()
        self._client = client
        self._buffer = bytearray()

        # cannot set the default to 0 because this will fail on an empty file
        # offset means to start at the first byte
        self._offset = offset

        self._eof = False
        self._downloader: Optional[StorageStreamDownloader] = None
        self._chunk_it: Optional[AsyncIterator[bytes]] = None

    async def read(self, n: int = -1) -> bytes:
        if self._eof:
            return b''

        if n == -1:
            try:
                downloader = await self._client.download_blob(offset=self._offset)
            except azure.core.exceptions.ResourceNotFoundError as e:
                raise FileNotFoundError from e
            data = await downloader.readall()
            self._eof = True
            return data

        if self._downloader is None:
            try:
                self._downloader = await self._client.download_blob(offset=self._offset)
            except azure.core.exceptions.ResourceNotFoundError as e:
                raise FileNotFoundError from e

        if self._chunk_it is None:
            self._chunk_it = self._downloader.chunks()

        while len(self._buffer) < n:
            try:
                chunk = await self._chunk_it.__anext__()
                self._buffer.extend(chunk)
            except StopAsyncIteration:
                break

        data = self._buffer[:n]
        self._buffer = self._buffer[n:]

        if self._offset is None:
            self._offset = 0
        self._offset += len(data)

        if len(data) < n:
            self._buffer = bytearray()
            self._downloader = None
            self._chunk_it = None
            self._eof = True

        return data

    async def readexactly(self, n: int) -> bytes:
        assert not self._closed and n >= 0
        data = await self.read(n)
        if len(data) != n:
            raise UnexpectedEOFError()
        return data

    async def _wait_closed(self) -> None:
        self._downloader = None
        self._chunk_it = None


class AzureFileListEntry(FileListEntry):
    def __init__(self, url: str, blob_props: Optional[BlobProperties]):
        self._url = url
        self._blob_props = blob_props
        self._status: Optional[AzureFileStatus] = None

    def name(self) -> str:
        parsed = urllib.parse.urlparse(self._url)
        return os.path.basename(parsed.path)

    async def url(self) -> str:
        return self._url

    def url_maybe_trailing_slash(self) -> str:
        return self._url

    async def is_file(self) -> bool:
        return self._blob_props is not None

    async def is_dir(self) -> bool:
        return self._blob_props is None

    async def status(self) -> FileStatus:
        if self._status is None:
            if self._blob_props is None:
                raise IsADirectoryError(self._url)
            self._status = AzureFileStatus(self._blob_props)
        return self._status


class AzureFileStatus(FileStatus):
    def __init__(self, blob_props: BlobProperties):
        self.blob_props = blob_props

    async def size(self) -> int:
        return self.blob_props.size

    async def __getitem__(self, key: str) -> Any:
        return self.blob_props.__dict__[key]


class AzureAsyncFS(AsyncFS):
    schemes: Set[str] = {'hail-az'}
    PATH_REGEX = re.compile('/(?P<container>[^/]+)(?P<name>.*)')

    def __init__(self, *, credential_file: Optional[str] = None, credentials: Optional[AzureCredentials] = None):
        if credentials is None:
            scopes = ['https://storage.azure.com/.default']
            if credential_file is not None:
                credentials = AzureCredentials.from_file(credential_file, scopes=scopes)
            else:
                credentials = AzureCredentials.default_credentials(scopes=scopes)
        else:
            if credential_file is not None:
                raise ValueError('credential and credential_file cannot both be defined')

        self._credential = credentials.credential
        self._blob_service_clients: Dict[str, BlobServiceClient] = {}

    @staticmethod
    def _get_account_container_name(url: str) -> Tuple[str, str, str]:
        parsed = urllib.parse.urlparse(url)

        if parsed.scheme != 'hail-az':
            raise ValueError(f'invalid scheme, expected hail-az: {parsed.scheme}')

        account = parsed.netloc

        match = AzureAsyncFS.PATH_REGEX.fullmatch(parsed.path)
        if match is None:
            raise ValueError(f'invalid path name, expected hail-az://account/container/blob_name: {parsed.path}')

        container = match.groupdict()['container']

        name = match.groupdict()['name']
        if name:
            assert name[0] == '/'
            name = name[1:]

        return (account, container, name)

    def get_blob_service_client(self, account: str) -> BlobServiceClient:
        if account not in self._blob_service_clients:
            self._blob_service_clients[account] = BlobServiceClient(f'https://{account}.blob.core.windows.net', credential=self._credential)
        return self._blob_service_clients[account]

    def get_blob_client(self, url: str) -> BlobClient:
        account, container, name = AzureAsyncFS._get_account_container_name(url)
        blob_service_client = self.get_blob_service_client(account)
        return blob_service_client.get_blob_client(container, name)

    def get_container_client(self, url: str) -> ContainerClient:
        account, container, _ = AzureAsyncFS._get_account_container_name(url)
        blob_service_client = self.get_blob_service_client(account)
        return blob_service_client.get_container_client(container)

    async def open(self, url: str) -> ReadableStream:
        client = self.get_blob_client(url)
        stream = AzureReadableStream(client)
        return stream

    async def open_from(self, url: str, start: int) -> ReadableStream:
        client = self.get_blob_client(url)
        stream = AzureReadableStream(client, offset=start)
        return stream

    async def create(self, url: str, *, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:  # pylint: disable=unused-argument
        client = self.get_blob_client(url)
        return AzureCreateManager(client)

    async def multi_part_create(
            self,
            sema: asyncio.Semaphore,
            url: str,
            num_parts: int) -> MultiPartCreate:
        client = self.get_blob_client(url)
        return AzureMultiPartCreate(sema, client, num_parts)

    async def isfile(self, url: str) -> bool:
        _, _, name = self._get_account_container_name(url)
        # if name is empty, get_object_metadata behaves like list objects
        # the urls are the same modulo the object name
        if not name:
            return False

        return await self.get_blob_client(url).exists()

    async def isdir(self, url: str) -> bool:
        _, _, name = self._get_account_container_name(url)
        assert not name or name.endswith('/'), name
        client = self.get_container_client(url)
        async for _ in client.walk_blobs(name_starts_with=name,
                                         include=['metadata'],
                                         delimiter='/'):
            return True
        return False

    async def mkdir(self, url: str) -> None:
        pass

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        pass

    async def statfile(self, url: str) -> FileStatus:
        try:
            blob_props = await self.get_blob_client(url).get_blob_properties()
            return AzureFileStatus(blob_props)
        except azure.core.exceptions.ResourceNotFoundError as e:
            raise FileNotFoundError(url) from e

    async def _listfiles_recursive(self, client: ContainerClient, name: str) -> AsyncIterator[FileListEntry]:
        assert not name or name.endswith('/')
        async for blob_props in client.list_blobs(name_starts_with=name,
                                                  include=['metadata']):
            url = f'hail-az://{client.account_name}/{client.container_name}/{blob_props.name}'
            yield AzureFileListEntry(url, blob_props)

    async def _listfiles_flat(self, client: ContainerClient, name: str) -> AsyncIterator[FileListEntry]:
        assert not name or name.endswith('/')
        async for item in client.walk_blobs(name_starts_with=name,
                                            include=['metadata'],
                                            delimiter='/'):
            if isinstance(item, BlobPrefix):
                url = f'hail-az://{client.account_name}/{client.container_name}/{item.prefix}'
                yield AzureFileListEntry(url, None)
            else:
                assert isinstance(item, BlobProperties)
                url = f'hail-az://{client.account_name}/{client.container_name}/{item.name}'
                yield AzureFileListEntry(url, item)

    async def listfiles(self,
                        url: str,
                        recursive: bool = False,
                        exclude_trailing_slash_files: bool = True
                        ) -> AsyncIterator[FileListEntry]:
        _, _, name = self._get_account_container_name(url)
        if name and not name.endswith('/'):
            name = f'{name}/'

        client = self.get_container_client(url)
        if recursive:
            it = self._listfiles_recursive(client, name)
        else:
            it = self._listfiles_flat(client, name)

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

        async def cons(first_entry, it):
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

    async def staturl(self, url: str) -> str:
        return await self._staturl_parallel_isfile_isdir(url)

    async def remove(self, url: str) -> None:
        try:
            await self.get_blob_client(url).delete_blob()
        except azure.core.exceptions.ResourceNotFoundError as e:
            raise FileNotFoundError(url) from e

    async def close(self) -> None:
        if self._credential:
            await self._credential.close()
            self._credential = None

        if self._blob_service_clients:
            await asyncio.wait([client.close() for client in self._blob_service_clients.values()])
