import asyncio
import datetime
import logging
import os.path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    BinaryIO,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

import aiohttp
import boto3
import botocore.config
import botocore.exceptions

from hailtop.aiotools.fs import (
    AsyncFS,
    AsyncFSURL,
    FileAndDirectoryError,
    FileListEntry,
    FileStatus,
    IsABucketError,
    MultiPartCreate,
    ReadableStream,
    WritableStream,
)
from hailtop.aiotools.fs.exceptions import UnexpectedEOFError
from hailtop.aiotools.fs.stream import (
    AsyncQueueWritableStream,
    async_writable_blocking_collect_pair,
    async_writable_blocking_readable_stream_pair,
    blocking_readable_stream_to_async,
)
from hailtop.utils import blocking_to_async

log = logging.getLogger(__name__)


class PageIterator:
    def __init__(self, fs: 'S3AsyncFS', bucket: str, prefix: str, delimiter: Optional[str] = None):
        self._fs = fs
        self._bucket = bucket
        self._prefix = prefix
        self._kwargs = {}
        if delimiter is not None:
            self._kwargs['Delimiter'] = delimiter
        self._page = None

    def __aiter__(self) -> 'PageIterator':
        return self

    async def __anext__(self):
        if self._page is None:
            self._page = await blocking_to_async(
                self._fs._thread_pool,
                self._fs._s3.list_objects_v2,  # type: ignore
                Bucket=self._bucket,
                Prefix=self._prefix,
                **self._kwargs,
            )
            return self._page

        next_continuation_token = self._page.get('NextContinuationToken')
        if next_continuation_token is not None:
            self._page = await blocking_to_async(
                self._fs._thread_pool,
                self._fs._s3.list_objects_v2,
                Bucket=self._bucket,
                Prefix=self._prefix,
                ContinuationToken=next_continuation_token,
                **self._kwargs,
            )
            return self._page

        raise StopAsyncIteration


class S3HeadObjectFileStatus(FileStatus):
    def __init__(self, head_object_resp, url: str):
        self.head_object_resp = head_object_resp
        self._url = url

    def basename(self) -> str:
        return os.path.basename(self._url.rstrip('/'))

    def url(self) -> str:
        return self._url

    async def size(self) -> int:
        return self.head_object_resp['ContentLength']

    def time_created(self) -> datetime.datetime:
        # https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html#API_HeadObject_ResponseSyntax
        # Misleading name: LastModified is creation time.
        # S3 Python library strips dashes from header names
        return self.head_object_resp['LastModified']

    def time_modified(self) -> datetime.datetime:
        # S3 objects are immutable, so creation == modified
        return self.head_object_resp['LastModified']

    async def __getitem__(self, key: str) -> Any:
        return self.head_object_resp[key]


class S3ListFilesFileStatus(FileStatus):
    def __init__(self, item: Dict[str, Any], url: str):
        self._item = item
        self._url = url

    def basename(self) -> str:
        return os.path.basename(self._url.rstrip('/'))

    def url(self) -> str:
        return self._url

    async def size(self) -> int:
        return self._item['Size']

    def time_created(self) -> datetime.datetime:
        # https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html#API_GetObject_ResponseSyntax
        # Misleading name: LastModified is creation time.
        # S3 Python library strips dashes from header names
        return self._item['LastModified']

    def time_modified(self) -> datetime.datetime:
        # S3 objects are immutable, so creation == modified
        print(repr(self._item))
        return self._item['LastModified']

    async def __getitem__(self, key: str) -> Any:
        return self._item[key]


class S3CreateManager(AsyncContextManager[WritableStream]):
    def __init__(self, fs: 'S3AsyncFS', bucket: str, name: str):
        self.fs: S3AsyncFS = fs
        self.bucket: str = bucket
        self.name: str = name
        self.async_writable: Optional[AsyncQueueWritableStream] = None
        self._put_thread: Optional[threading.Thread] = None
        self._value: Any = None
        self._exc: Optional[BaseException] = None

    async def __aenter__(self) -> WritableStream:
        async_writable, blocking_readable = async_writable_blocking_readable_stream_pair()
        self.async_writable = async_writable

        def put():
            try:
                self._value = self.fs._s3.upload_fileobj(blocking_readable, Bucket=self.bucket, Key=self.name)
            except BaseException as e:
                self._exc = e

        self._put_thread = threading.Thread(target=put)
        self._put_thread.start()
        return async_writable

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> None:
        assert self.async_writable
        assert self._put_thread
        await self.async_writable.wait_closed()
        try:
            await blocking_to_async(self.fs._thread_pool, self._put_thread.join)
        finally:
            if self._exc:
                _, exc, _ = sys.exc_info()
                if exc:
                    log.info('discarding exception', exc_info=True)
                raise self._exc


class S3FileListEntry(FileListEntry):
    def __init__(self, bucket: str, key: str, item: Optional[Dict[str, Any]]):
        self._bucket = bucket
        self._key = key
        self._item = item
        self._status: Optional[S3ListFilesFileStatus] = None

    def basename(self) -> str:
        return os.path.basename(self._key.rstrip('/'))

    async def url(self) -> str:
        return f's3://{self._bucket}/{self._key}'

    async def is_file(self) -> bool:
        return self._item is not None

    async def is_dir(self) -> bool:
        return self._item is None

    async def status(self) -> FileStatus:
        if self._status is None:
            if self._item is None:
                raise IsADirectoryError(f's3://{self._bucket}/{self._key}')
            self._status = S3ListFilesFileStatus(self._item, await self.url())
        return self._status


class S3CreatePartManager(AsyncContextManager[WritableStream]):
    def __init__(self, mpc, number: int, size_hint: int):
        self._mpc = mpc
        self._number = number
        self._size_hint = size_hint
        self._async_writable: Optional[AsyncQueueWritableStream] = None
        self._put_thread: Optional[threading.Thread] = None
        self._exc: Optional[BaseException] = None

    async def __aenter__(self) -> WritableStream:
        async_writable, blocking_collect = async_writable_blocking_collect_pair(self._size_hint)
        self._async_writable = async_writable

        def put():
            try:
                b = blocking_collect.get()
                resp = self._mpc._fs._s3.upload_part(
                    Bucket=self._mpc._bucket,
                    Key=self._mpc._name,
                    PartNumber=self._number + 1,
                    UploadId=self._mpc._upload_id,
                    Body=b,
                )
                self._mpc._etags[self._number] = resp['ETag']
            except BaseException as e:
                self._exc = e

        self._put_thread = threading.Thread(target=put)
        self._put_thread.start()
        return async_writable

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> None:
        assert self._async_writable is not None
        assert self._put_thread is not None
        await self._async_writable.wait_closed()
        try:
            await blocking_to_async(self._mpc._fs._thread_pool, self._put_thread.join)
        finally:
            if self._exc:
                _, exc, _ = sys.exc_info()
                if exc:
                    log.info('discarding exception', exc_info=True)
                raise self._exc


class S3MultiPartCreate(MultiPartCreate):
    def __init__(self, sema: asyncio.Semaphore, fs: 'S3AsyncFS', bucket: str, name: str, num_parts: int):
        assert num_parts <= 10000
        self._sema = sema
        self._fs = fs
        self._bucket = bucket
        self._name = name
        self._num_parts = num_parts
        self._upload_id = None
        self._etags: List[Optional[str]] = [None] * num_parts

    async def __aenter__(self) -> 'S3MultiPartCreate':
        resp = await blocking_to_async(
            self._fs._thread_pool, self._fs._s3.create_multipart_upload, Bucket=self._bucket, Key=self._name
        )
        self._upload_id = resp['UploadId']
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> None:
        if exc_value is not None:
            await blocking_to_async(
                self._fs._thread_pool,
                self._fs._s3.abort_multipart_upload,
                Bucket=self._bucket,
                Key=self._name,
                UploadId=self._upload_id,
            )
            return

        parts = []
        part_number = 1
        for etag in self._etags:
            assert etag is not None
            parts.append({'ETag': etag, 'PartNumber': part_number})
            part_number += 1

        await blocking_to_async(
            self._fs._thread_pool,
            self._fs._s3.complete_multipart_upload,
            Bucket=self._bucket,
            Key=self._name,
            MultipartUpload={'Parts': parts},
            UploadId=self._upload_id,
        )

    async def create_part(self, number: int, start: int, size_hint: Optional[int] = None) -> S3CreatePartManager:  # pylint: disable=unused-argument
        if size_hint is None:
            size_hint = 256 * 1024
        return S3CreatePartManager(self, number, size_hint)


class S3AsyncFSURL(AsyncFSURL):
    def __init__(self, bucket: str, path: str):
        self._bucket = bucket
        self._path = path

    def __repr__(self):
        return f'S3AsyncFSURL({self._bucket}, {self._path})'

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
        return 's3'

    def with_path(self, path) -> 'S3AsyncFSURL':
        return S3AsyncFSURL(self._bucket, path)

    def with_root_path(self) -> 'S3AsyncFSURL':
        return self.with_path('')

    def __str__(self) -> str:
        return f's3://{self._bucket}/{self._path}'


class S3AsyncFS(AsyncFS):
    def __init__(
        self,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        max_workers: Optional[int] = None,
        *,
        max_pool_connections: int = 10,
        timeout: Optional[Union[int, float, aiohttp.ClientTimeout]] = None,
    ):
        if not thread_pool:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_pool = thread_pool

        kwargs = {}
        if isinstance(timeout, aiohttp.ClientTimeout):
            if timeout.sock_read:
                kwargs['read_timeout'] = timeout.sock_read
            elif timeout.total:
                kwargs['read_timeout'] = timeout.total

            if timeout.sock_connect:
                kwargs['connect_timeout'] = timeout.sock_connect
            elif timeout.connect:
                kwargs['connect_timeout'] = timeout.connect
            elif timeout.total:
                kwargs['connect_timeout'] = timeout.total
        elif isinstance(timeout, (int, float)):
            kwargs['read_timeout'] = timeout
            kwargs['connect_timeout'] = timeout

        config = botocore.config.Config(
            max_pool_connections=max_pool_connections,
            **kwargs,
        )
        self._s3 = boto3.client('s3', config=config)

    @staticmethod
    def schemes() -> Set[str]:
        return {'s3'}

    @staticmethod
    def copy_part_size(url: str) -> int:  # pylint: disable=unused-argument
        # Because the S3 upload_part API call requires the entire part
        # be loaded into memory, use a smaller part size.
        return 32 * 1024 * 1024

    @staticmethod
    def valid_url(url: str) -> bool:
        return url.startswith('s3://')

    @staticmethod
    def parse_url(url: str, *, error_if_bucket: bool = False) -> S3AsyncFSURL:
        fsurl = S3AsyncFSURL(*S3AsyncFS.get_bucket_and_name(url))
        if error_if_bucket and fsurl._path == '':
            raise IsABucketError
        return fsurl

    @staticmethod
    def get_bucket_and_name(url: str) -> Tuple[str, str]:
        colon_index = url.find(':')
        if colon_index == -1:
            raise ValueError(f'invalid URL: {url}')

        scheme = url[:colon_index]
        if scheme != 's3':
            raise ValueError(f'invalid scheme, expected s3: {scheme}')

        rest = url[(colon_index + 1) :]
        if not rest.startswith('//'):
            raise ValueError(f's3 URI must be of the form: s3://bucket/key, found: {url}')

        end_of_bucket = rest.find('/', 2)
        bucket = rest[2:end_of_bucket]
        name = rest[(end_of_bucket + 1) :]

        return (bucket, name)

    async def open(self, url: str) -> ReadableStream:
        fsurl = self.parse_url(url, error_if_bucket=True)
        try:
            resp = await blocking_to_async(
                self._thread_pool, self._s3.get_object, Bucket=fsurl._bucket, Key=fsurl._path
            )
            return blocking_readable_stream_to_async(self._thread_pool, cast(BinaryIO, resp['Body']))
        except self._s3.exceptions.NoSuchKey as e:
            raise FileNotFoundError(url) from e

    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        fsurl = self.parse_url(url, error_if_bucket=True)
        range_str = f'bytes={start}-'
        if length is not None:
            assert length >= 1
            range_str += str(start + length - 1)
        try:
            resp = await blocking_to_async(
                self._thread_pool, self._s3.get_object, Bucket=fsurl._bucket, Key=fsurl._path, Range=range_str
            )
            return blocking_readable_stream_to_async(self._thread_pool, cast(BinaryIO, resp['Body']))
        except self._s3.exceptions.NoSuchKey as e:
            raise FileNotFoundError(url) from e
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidRange':
                raise UnexpectedEOFError from e
            raise

    async def create(self, url: str, *, retry_writes: bool = True) -> S3CreateManager:  # pylint: disable=unused-argument
        # It may be possible to write a more efficient version of this
        # that takes advantage of retry_writes=False.  Here's the
        # background information:
        #
        # There are essentially three options for implementing writes.
        # The first two handle retries:
        #
        #  1. Use some form of multipart uploads (which, in the case
        #     of GCS, we implement by writing temporary objects and
        #     then calling compose).
        #
        #  2. Use resumable uploads.  This is what the GCS backend
        #     does, although the performance is must worse than
        #     non-resumable uploads so in fact it may always be better
        #     to always use multipart uploads (1).
        #
        # The third does not handle failures:
        #
        #  3. Don't be failure/retry safe.  Just write the object, and
        #  if the API call fails, fail.  This is useful when you can
        #  retry at a higher level (this is what the copy code does).
        #
        # Unfortunately, I don't see how to do (3) with boto3, since
        # AWS APIs require a header that includes a hash of the
        # request body, and that needs to be computed up front.  In
        # terms of the boto3 interface, this contraint translates into
        # calls like `put_object` require bytes or a seekable stream
        # (so it can make two passes over the data, one to compute the
        # checksome, and the other to send the data).
        #
        # Here, we use S3CreateManager, which in turn uses boto3
        # `upload_fileobj` which is implemented in terms of multipart
        # uploads.
        #
        # Another possibility is to make an alternate `create` call
        # that takes bytes instead of returning a file-like object,
        # and then using `put_object`, and make copy use that
        # interface.  This has the disadvantage that the read must
        # complete before the write can begin (unlike the current
        # code, that copies 128MB parts in 256KB chunks).
        fsurl = self.parse_url(url, error_if_bucket=True)
        return S3CreateManager(self, fsurl._bucket, fsurl._path)

    async def multi_part_create(self, sema: asyncio.Semaphore, url: str, num_parts: int) -> MultiPartCreate:
        fsurl = self.parse_url(url, error_if_bucket=True)
        return S3MultiPartCreate(sema, self, fsurl._bucket, fsurl._path, num_parts)

    async def mkdir(self, url: str) -> None:
        pass

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        pass

    async def statfile(self, url: str) -> FileStatus:
        fsurl = self.parse_url(url, error_if_bucket=True)
        try:
            resp = await blocking_to_async(
                self._thread_pool, self._s3.head_object, Bucket=fsurl._bucket, Key=fsurl._path
            )
            return S3HeadObjectFileStatus(resp, url)
        except botocore.exceptions.ClientError as e:
            if e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                raise FileNotFoundError(url) from e
            raise e

    async def _listfiles_recursive(self, bucket: str, name: str) -> AsyncIterator[S3FileListEntry]:
        assert not name or name.endswith('/')
        async for page in PageIterator(self, bucket, name):
            assert 'CommonPrefixes' not in page
            contents = page.get('Contents')
            if contents:
                for item in contents:
                    yield S3FileListEntry(bucket, item['Key'], item)

    async def _listfiles_flat(self, bucket: str, name: str) -> AsyncIterator[S3FileListEntry]:
        assert not name or name.endswith('/')
        async for page in PageIterator(self, bucket, name, delimiter='/'):
            prefixes = page.get('CommonPrefixes')
            if prefixes is not None:
                for prefix in prefixes:
                    yield S3FileListEntry(bucket, prefix['Prefix'], None)
            contents = page.get('Contents')
            if contents:
                for item in contents:
                    yield S3FileListEntry(bucket, item['Key'], item)

    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        bucket, name = self.get_bucket_and_name(url)
        if name and not name.endswith('/'):
            name += '/'
        if recursive:
            it = self._listfiles_recursive(bucket, name)
        else:
            it = self._listfiles_flat(bucket, name)

        it = aiter(it)
        try:
            first_entry = await anext(it)
        except StopAsyncIteration:
            raise FileNotFoundError(url)  # pylint: disable=raise-missing-from

        async def should_yield(entry: S3FileListEntry):
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
                    next_entry = await anext(it)
                    if await should_yield(next_entry):
                        yield next_entry
            except StopAsyncIteration:
                pass

        return cons(first_entry, it)

    async def staturl(self, url: str) -> str:
        return await self._staturl_parallel_isfile_isdir(url)

    async def isfile(self, url: str) -> bool:
        bucket, name = self.get_bucket_and_name(url)
        if name == '':
            return False
        try:
            await blocking_to_async(self._thread_pool, self._s3.head_object, Bucket=bucket, Key=name)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                return False
            raise e

    async def isdir(self, url: str) -> bool:
        self.parse_url(url, error_if_bucket=True)
        try:
            async for _ in await self.listfiles(url, recursive=True):
                return True
            assert False  # unreachable
        except FileNotFoundError:
            return False

    async def remove(self, url: str) -> None:
        fsurl = self.parse_url(url, error_if_bucket=True)
        try:
            await blocking_to_async(self._thread_pool, self._s3.delete_object, Bucket=fsurl._bucket, Key=fsurl._path)
        except self._s3.exceptions.NoSuchKey as e:
            raise FileNotFoundError(url) from e

    async def close(self) -> None:
        del self._s3
