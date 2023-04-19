from typing import Any, AsyncContextManager, AsyncIterator, Dict, List, Optional, Set, Tuple, Type
from types import TracebackType

import re
import asyncio
from functools import wraps
import secrets
import logging
import datetime

from azure.storage.blob import BlobProperties
from azure.storage.blob.aio import BlobClient, ContainerClient, BlobServiceClient, StorageStreamDownloader
from azure.storage.blob.aio._list_blobs_helper import BlobPrefix
import azure.core.exceptions

from hailtop.utils import retry_transient_errors, flatten
from hailtop.aiotools import WriteBuffer
from hailtop.aiotools.fs import (AsyncFS, AsyncFSURL, AsyncFSFactory, ReadableStream,
                                 WritableStream, MultiPartCreate, FileListEntry, FileStatus,
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
        with self._write_buffer.chunks(n) as chunks:
            await self.client.stage_block(block_id, chunks)
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
            # azure allows both BlockBlob and the string id here, despite
            # only having BlockBlob annotations
            await self._client.commit_block_list(flatten(self._block_ids)  # type: ignore
                                                 )
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
                # azure allows both BlockBlob and the string id here, despite
                # only having BlockBlob annotations
                await self._client.commit_block_list(self._block_ids  # type: ignore
                                                     )
            except:
                try:
                    await self._client.delete_blob()
                except azure.core.exceptions.ResourceNotFoundError:
                    pass
                raise


class AzureReadableStream(ReadableStream):
    def __init__(self, client: BlobClient, url: str, offset: Optional[int] = None, length: Optional[int] = None):
        super().__init__()
        self._client = client
        self._buffer = bytearray()
        self._url = url

        # cannot set the default to 0 because this will fail on an empty file
        # offset means to start at the first byte
        self._offset = offset
        self._length = length

        self._eof = False
        self._downloader: Optional[StorageStreamDownloader] = None
        self._chunk_it: Optional[AsyncIterator[bytes]] = None

    async def read(self, n: int = -1) -> bytes:
        if self._eof:
            return b''

        if n == -1:
            try:
                downloader = await self._client.download_blob(offset=self._offset, length=self._length)  # type: ignore
            except azure.core.exceptions.ResourceNotFoundError as e:
                raise FileNotFoundError(self._url) from e
            data = await downloader.readall()
            self._eof = True
            return data

        if self._downloader is None:
            try:
                self._downloader = await self._client.download_blob(offset=self._offset)  # type: ignore
            except azure.core.exceptions.ResourceNotFoundError as e:
                raise FileNotFoundError(self._url) from e
            except azure.core.exceptions.HttpResponseError as e:
                if e.status_code == 416:
                    raise UnexpectedEOFError from e
                raise

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
    def __init__(self, account: str, container: str, name: str, blob_props: Optional[BlobProperties]):
        self._account = account
        self._container = container
        self._name = name
        self._blob_props = blob_props
        self._status: Optional[AzureFileStatus] = None

    def name(self) -> str:
        return self._name

    async def url(self) -> str:
        return f'hail-az://{self._account}/{self._container}/{self._name}'

    def url_maybe_trailing_slash(self) -> str:
        return f'hail-az://{self._account}/{self._container}/{self._name}'

    async def is_file(self) -> bool:
        return self._blob_props is not None

    async def is_dir(self) -> bool:
        return self._blob_props is None

    async def status(self) -> FileStatus:
        if self._status is None:
            if self._blob_props is None:
                raise IsADirectoryError(await self.url())
            self._status = AzureFileStatus(self._blob_props)
        return self._status


class AzureFileStatus(FileStatus):
    def __init__(self, blob_props: BlobProperties):
        self.blob_props = blob_props

    async def size(self) -> int:
        size = self.blob_props.size
        assert isinstance(size, int)
        return size

    def time_created(self) -> datetime.datetime:
        ct = self.blob_props.creation_time
        assert isinstance(ct, datetime.datetime)
        return ct

    def time_modified(self) -> datetime.datetime:
        lm = self.blob_props.last_modified
        assert isinstance(lm, datetime.datetime)
        return lm

    async def __getitem__(self, key: str) -> Any:
        return self.blob_props.__dict__[key]


class AzureAsyncFSURL(AsyncFSURL):
    def __init__(self, account: str, container: str, path: str):
        self._account = account
        self._container = container
        self._path = path

    @property
    def bucket_parts(self) -> List[str]:
        return [self._account, self._container]

    @property
    def path(self) -> str:
        return self._path

    @property
    def scheme(self) -> str:
        return 'hail-az'

    @property
    def account(self) -> str:
        return self._account

    @property
    def container(self) -> str:
        return self._container

    def with_path(self, path) -> 'AzureAsyncFSURL':
        return AzureAsyncFSURL(self._account, self._container, path)

    def __str__(self) -> str:
        return f'hail-az://{self._account}/{self._container}/{self._path}'


# ABS errors if you attempt credentialed access for a public container,
# so we try once with credentials, if that fails use anonymous access for
# that container going forward.
def handle_public_access_error(fun):
    @wraps(fun)
    async def wrapped(self, url, *args, **kwargs):
        try:
            return await fun(self, url, *args, **kwargs)
        except azure.core.exceptions.ClientAuthenticationError:
            fs_url = self.parse_url(url)
            anon_client = BlobServiceClient(f'https://{fs_url.account}.blob.core.windows.net', credential=None)
            self._blob_service_clients[(fs_url.account, fs_url.container)] = anon_client
            return await fun(self, url, *args, **kwargs)
    return wrapped


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
        self._blob_service_clients: Dict[Tuple[str, str], BlobServiceClient] = {}

    def parse_url(self, url: str) -> AzureAsyncFSURL:
        return AzureAsyncFSURL(*self.get_account_container_and_name(url))

    @staticmethod
    def get_account_container_and_name(url: str) -> Tuple[str, str, str]:
        colon_index = url.find(':')
        if colon_index == -1:
            raise ValueError(f'invalid URL: {url}')

        scheme = url[:colon_index]
        if scheme != 'hail-az':
            raise ValueError(f'invalid scheme, expected hail-az: {scheme}')

        rest = url[(colon_index + 1):]
        if not rest.startswith('//'):
            raise ValueError(f'invalid path name, expected hail-az://account/container/blob_name: {url}')

        end_of_account = rest.find('/', 2)
        account = rest[2:end_of_account]
        container_and_name = rest[end_of_account:]

        match = AzureAsyncFS.PATH_REGEX.fullmatch(container_and_name)
        if match is None:
            raise ValueError(f'invalid path name, expected hail-az://account/container/blob_name: {container_and_name}')

        container = match.groupdict()['container']

        name = match.groupdict()['name']
        if name:
            assert name[0] == '/'
            name = name[1:]

        return (account, container, name)

    def get_blob_service_client(self, account: str, container: str) -> BlobServiceClient:
        k = account, container
        if k not in self._blob_service_clients:
            self._blob_service_clients[k] = BlobServiceClient(f'https://{account}.blob.core.windows.net', credential=self._credential)
        return self._blob_service_clients[k]

    def get_blob_client(self, url: str) -> BlobClient:
        account, container, name = AzureAsyncFS.get_account_container_and_name(url)
        blob_service_client = self.get_blob_service_client(account, container)
        return blob_service_client.get_blob_client(container, name)

    def get_container_client(self, url: str) -> ContainerClient:
        account, container, _ = AzureAsyncFS.get_account_container_and_name(url)
        blob_service_client = self.get_blob_service_client(account, container)
        return blob_service_client.get_container_client(container)

    @handle_public_access_error
    async def open(self, url: str) -> ReadableStream:
        if not await self.exists(url):
            raise FileNotFoundError
        client = self.get_blob_client(url)
        return AzureReadableStream(client, url)

    @handle_public_access_error
    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        assert length is None or length >= 1
        if not await self.exists(url):
            raise FileNotFoundError
        client = self.get_blob_client(url)
        return AzureReadableStream(client, url, offset=start, length=length)

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

    @handle_public_access_error
    async def isfile(self, url: str) -> bool:
        _, _, name = self.get_account_container_and_name(url)
        # if name is empty, get_object_metadata behaves like list objects
        # the urls are the same modulo the object name
        if not name:
            return False

        return await self.get_blob_client(url).exists()

    @handle_public_access_error
    async def isdir(self, url: str) -> bool:
        _, _, name = self.get_account_container_and_name(url)
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

    @handle_public_access_error
    async def statfile(self, url: str) -> FileStatus:
        try:
            blob_props = await self.get_blob_client(url).get_blob_properties()
            return AzureFileStatus(blob_props)
        except azure.core.exceptions.ResourceNotFoundError as e:
            raise FileNotFoundError(url) from e

    @staticmethod
    async def _listfiles_recursive(client: ContainerClient, name: str) -> AsyncIterator[FileListEntry]:
        assert not name or name.endswith('/')
        async for blob_props in client.list_blobs(name_starts_with=name,
                                                  include=['metadata']):
            yield AzureFileListEntry(client.account_name, client.container_name, blob_props.name, blob_props)  # type: ignore

    @staticmethod
    async def _listfiles_flat(client: ContainerClient, name: str) -> AsyncIterator[FileListEntry]:
        assert not name or name.endswith('/')
        async for item in client.walk_blobs(name_starts_with=name,
                                            include=['metadata'],
                                            delimiter='/'):
            if isinstance(item, BlobPrefix):
                yield AzureFileListEntry(client.account_name, client.container_name, item.prefix, None)  # type: ignore
            else:
                assert isinstance(item, BlobProperties)
                yield AzureFileListEntry(client.account_name, client.container_name, item.name, item)  # type: ignore

    @handle_public_access_error
    async def listfiles(self,
                        url: str,
                        recursive: bool = False,
                        exclude_trailing_slash_files: bool = True
                        ) -> AsyncIterator[FileListEntry]:
        _, _, name = self.get_account_container_and_name(url)
        if name and not name.endswith('/'):
            name = f'{name}/'

        client = self.get_container_client(url)
        if recursive:
            it = AzureAsyncFS._listfiles_recursive(client, name)
        else:
            it = AzureAsyncFS._listfiles_flat(client, name)

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

    @handle_public_access_error
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


class AzureAsyncFSFactory(AsyncFSFactory[AzureAsyncFS]):
    def from_credentials_data(self, credentials_data: dict) -> AzureAsyncFS:
        return AzureAsyncFS(
            credentials=AzureCredentials.from_credentials_data(credentials_data))

    def from_credentials_file(self, credentials_file: str) -> AzureAsyncFS:
        return AzureAsyncFS(
            credentials=AzureCredentials.from_file(credentials_file))

    def from_default_credentials(self) -> AzureAsyncFS:
        return AzureAsyncFS(
            credentials=AzureCredentials.default_credentials())
