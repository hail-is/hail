import google.api_core.exceptions
import google.oauth2.service_account
import google.cloud.storage
from hailtop.utils import blocking_to_async


class GCS:
    @staticmethod
    def _parse_uri(uri):
        assert uri.startswith('gs://')
        uri = uri.lstrip('gs://').split('/')
        bucket = uri[0]
        path = '/'.join(uri[1:])
        return bucket, path

    def __init__(self, blocking_pool, credentials=None):
        self.blocking_pool = blocking_pool
        self.gcs_client = google.cloud.storage.Client(credentials=credentials)
        self._wrapped_write_gs_file = self._wrap_nonreturning_network_call(GCS._write_gs_file)
        self._wrapped_read_gs_file = self._wrap_returning_network_call(GCS._read_gs_file)
        self._wrapped_delete_gs_file = self._wrap_nonreturning_network_call(GCS._delete_gs_file)

    async def write_gs_file(self, uri, string):
        return await self._wrapped_write_gs_file(self, uri, string)

    async def read_gs_file(self, uri):
        return await self._wrapped_read_gs_file(self, uri)

    async def delete_gs_file(self, uri):
        return await self._wrapped_delete_gs_file(self, uri)

    def _wrap_returning_network_call(self, fun):
        async def wrapped(*args, **kwargs):
            try:
                return (await blocking_to_async(self.blocking_pool,
                                                fun,
                                                *args,
                                                **kwargs),
                        None)
            except google.api_core.exceptions.GoogleAPIError as err:
                return (None, err)
        wrapped.__name__ = fun.__name__
        return wrapped

    def _wrap_nonreturning_network_call(self, fun):
        fun = self._wrap_returning_network_call(fun)

        async def wrapped(*args, **kwargs):
            _, err = await fun(*args, *kwargs)
            return err
        wrapped.__name__ = fun.__name__
        return wrapped

    def _write_gs_file(self, uri, string):
        bucket, path = GCS._parse_uri(uri)
        bucket = self.gcs_client.bucket(bucket)
        f = bucket.blob(path)
        f.metadata = {'Cache-Control': 'no-cache'}
        f.upload_from_string(string)

    def _read_gs_file(self, uri):
        bucket, path = GCS._parse_uri(uri)
        bucket = self.gcs_client.bucket(bucket)
        f = bucket.blob(path)
        f.metadata = {'Cache-Control': 'no-cache'}
        content = f.download_as_string()
        return content.decode('utf-8')

    def _delete_gs_file(self, uri):
        bucket, path = GCS._parse_uri(uri)
        bucket = self.gcs_client.bucket(bucket)
        f = bucket.blob(path)
        f.delete()
