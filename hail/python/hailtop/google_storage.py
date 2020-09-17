from typing import Optional, IO, Callable, List
import os
import logging
import concurrent.futures
from functools import wraps

import google.api_core.exceptions
import google.oauth2.service_account
import google.cloud.storage
from google.cloud.storage.blob import Blob
from hailtop.utils import blocking_to_async, retry_transient_errors


logging.getLogger("google").setLevel(logging.WARNING)


class GCS:
    @staticmethod
    def _parse_uri(uri: str):
        assert uri.startswith('gs://'), uri
        uri_parts = uri.lstrip('gs://').split('/')
        bucket = uri_parts[0]
        path = '/'.join(uri_parts[1:])
        return bucket, path

    def __init__(self,
                 blocking_pool: concurrent.futures.Executor,
                 *,
                 project: Optional[str] = None,
                 key: Optional[str] = None,
                 credentials: Optional[google.oauth2.service_account.Credentials] = None):
        self.blocking_pool = blocking_pool
        # project=None doesn't mean default, it means no project:
        # https://github.com/googleapis/google-cloud-python/blob/master/storage/google/cloud/storage/client.py#L86
        if credentials is None:
            if key is not None:
                credentials = google.oauth2.service_account.Credentials.from_service_account_info(key)
            elif 'HAIL_GSA_KEY_FILE' in os.environ:
                key_file = os.environ['HAIL_GSA_KEY_FILE']
                credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_file)

        if project:
            self.gcs_client = google.cloud.storage.Client(
                project=project, credentials=credentials)
        else:
            self.gcs_client = google.cloud.storage.Client(
                credentials=credentials)
        self._wrapped_write_gs_file_from_string = self._wrap_network_call(GCS._write_gs_file_from_string)
        self._wrapped_write_gs_file_from_file_like_object = self._wrap_network_call(GCS._write_gs_file_from_file_like_object)
        self._wrapped_read_gs_file = self._wrap_network_call(GCS._read_gs_file)
        self._wrapped_read_binary_gs_file = self._wrap_network_call(GCS._read_binary_gs_file)
        self._wrapped_read_gs_file_to_file = self._wrap_network_call(GCS._read_gs_file_to_file)
        self._wrapped_delete_gs_file = self._wrap_network_call(GCS._delete_gs_file)
        self._wrapped_delete_gs_files = self._wrap_network_call(GCS._delete_gs_files)
        self._wrapped_copy_gs_file = self._wrap_network_call(GCS._copy_gs_file)
        self._wrapped_list_all_blobs_with_prefix = self._wrap_network_call(GCS._list_all_blobs_with_prefix)
        self._wrapped_compose_gs_file = self._wrap_network_call(GCS._compose_gs_file)
        self._wrapped_get_blob = self._wrap_network_call(GCS._get_blob)

    def shutdown(self, wait: bool = True):
        self.blocking_pool.shutdown(wait)

    async def get_etag(self, uri: str):
        return await retry_transient_errors(self._wrap_network_call(GCS._get_etag), self, uri)

    async def write_gs_file_from_string(self, uri: str, string: str, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_write_gs_file_from_string,
                                            self, uri, string, *args, **kwargs)

    async def write_gs_file_from_file_like_object(self, uri: str, file: IO, *args, start=None, end=None, **kwargs):
        return await retry_transient_errors(self._wrapped_write_gs_file_from_file_like_object,
                                            self, uri, file, start, end, *args, **kwargs)

    async def write_gs_file_from_file(self, uri: str, file_name: str, *args, start=None, end=None, **kwargs):
        with open(file_name, 'r') as file:
            await self.write_gs_file_from_file_like_object(uri, file, *args, start=start, end=end, **kwargs)

    async def read_gs_file(self, uri: str, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_read_gs_file,
                                            self, uri, *args, **kwargs)

    async def read_binary_gs_file(self, uri: str, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_read_binary_gs_file,
                                            self, uri, *args, **kwargs)

    async def read_gs_file_to_file(self, uri: str, file_name, offset, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_read_gs_file_to_file,
                                            self, uri, file_name, offset, *args, **kwargs)

    async def delete_gs_file(self, uri: str):
        return await retry_transient_errors(self._wrapped_delete_gs_file,
                                            self, uri)

    async def delete_gs_files(self, uri_prefix: str):
        return await retry_transient_errors(self._wrapped_delete_gs_files,
                                            self, uri_prefix)

    async def copy_gs_file(self, src: str, dest: str, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_copy_gs_file,
                                            self, src, dest, *args, **kwargs)

    async def compose_gs_file(self, sources: str, dest: str, *args, **kwargs):
        return await retry_transient_errors(self._wrapped_compose_gs_file,
                                            self, sources, dest, *args, **kwargs)

    async def list_all_blobs_with_prefix(self, uri: str, max_results: Optional[int] = None):
        return await retry_transient_errors(self._wrapped_list_all_blobs_with_prefix,
                                            self, uri, max_results=max_results)

    async def get_blob(self, uri: str):
        return await retry_transient_errors(self._wrapped_get_blob,
                                            self, uri)

    def _wrap_network_call(self, fun: Callable) -> Callable:
        @wraps(fun)
        async def wrapped(*args, **kwargs):
            return await blocking_to_async(self.blocking_pool,
                                           fun,
                                           *args,
                                           **kwargs)
        return wrapped

    def _get_etag(self, uri: str):
        b = self._get_blob(uri)
        b.reload()
        return b.etag

    def _write_gs_file_from_string(self, uri: str, string: str, *args, **kwargs):
        b = self._get_blob(uri)
        b.metadata = {'Cache-Control': 'no-cache'}
        b.upload_from_string(string, *args, **kwargs)

    def _write_gs_file_from_file_like_object(self, uri: str, file: IO, *args, **kwargs):
        b = self._get_blob(uri)
        b.metadata = {'Cache-Control': 'no-cache'}
        b.upload_from_file(file, *args, **kwargs)

    def _read_gs_file(self, uri: str, *args, **kwargs):
        content = self._read_binary_gs_file(uri, *args, **kwargs)
        return content.decode('utf-8')

    def _read_binary_gs_file(self, uri: str, *args, **kwargs):
        b = self._get_blob(uri)
        b.metadata = {'Cache-Control': 'no-cache'}
        content = b.download_as_string(*args, **kwargs)
        return content

    def _read_gs_file_to_file(self, uri: str, file_name: str, offset: int, *args, **kwargs):
        with open(file_name, 'r+b') as file:
            file.seek(offset)
            b = self._get_blob(uri)
            b.metadata = {'Cache-Control': 'no-cache'}
            b.download_to_file(file, *args, **kwargs)

    def _delete_gs_files(self, uri: str):
        for blob in self._list_all_blobs_with_prefix(uri):
            try:
                blob.delete()
            except google.api_core.exceptions.NotFound:
                continue

    def _delete_gs_file(self, uri: str):
        b = self._get_blob(uri)
        try:
            b.delete()
        except google.api_core.exceptions.NotFound:
            return

    def _copy_gs_file(self, src: str, dest: str, *args, **kwargs):
        src_bucket, src_path = GCS._parse_uri(src)
        src_bucket = self.gcs_client.bucket(src_bucket)
        dest_bucket, dest_path = GCS._parse_uri(dest)
        dest_bucket = self.gcs_client.bucket(dest_bucket)
        src_blob = src_bucket.blob(src_path)
        src_bucket.copy_blob(src_blob, dest_bucket, new_name=dest_path, *args, **kwargs)

    def _list_all_blobs_with_prefix(self, uri: str, max_results: Optional[int] = None):
        b = self._get_blob(uri)
        return iter(b.bucket.list_blobs(prefix=b.name, max_results=max_results))

    def _compose_gs_file(self, sources: List[str], dest: str, *args, **kwargs):
        assert sources
        sources = [self._get_blob(src) for src in sources]
        dest_blob = self._get_blob(dest)
        dest_blob.compose(sources, *args, **kwargs)

    def _get_blob(self, uri: str) -> Blob:
        bucket, path = GCS._parse_uri(uri)
        bucket = self.gcs_client.bucket(bucket)
        return bucket.blob(path)
