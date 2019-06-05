import os

import google.api_core.exceptions
import google.oauth2.service_account
import google.cloud.storage

from .blocking_to_async import blocking_to_async


class GCS:
    def __init__(self, blocking_pool, batch_gsa_key=None):
        self.blocking_pool = blocking_pool
        if batch_gsa_key is None:
            batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            batch_gsa_key)
        self.gcs_client = google.cloud.storage.Client(credentials=credentials)
        self._wrapped_upload_private_gs_file_from_string = self._wrap_nonreturning_network_call(
            GCS._upload_private_gs_file_from_string)
        self._wrapped_download_gs_file_as_string = self._wrap_returning_network_call(
            GCS._download_gs_file_as_string)
        self._wrapped_delete_gs_file = self._wrap_nonreturning_network_call(GCS._delete_gs_file)

    async def upload_private_gs_file_from_string(self, bucket, target_path, string):
        return await self._wrapped_upload_private_gs_file_from_string(
            self, bucket, target_path, string)

    async def download_gs_file_as_string(self, bucket, path):
        return await self._wrapped_download_gs_file_as_string(self, bucket, path)

    async def delete_gs_file(self, bucket, path):
        return await self._wrapped_delete_gs_file(self, bucket, path)

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

    def _upload_private_gs_file_from_string(self, bucket, target_path, string):
        bucket = self.gcs_client.bucket(bucket)
        f = bucket.blob(target_path)
        f.metadata = {'Cache-Control': 'no-cache'}
        f.upload_from_string(string)

    def _download_gs_file_as_string(self, bucket, path):
        bucket = self.gcs_client.bucket(bucket)
        f = bucket.blob(path)
        f.metadata = {'Cache-Control': 'no-cache'}
        content = f.download_as_string()
        return content.decode('utf-8')

    def _delete_gs_file(self, bucket, path):
        bucket = self.gcs_client.bucket(bucket)
        f = bucket.blob(path)
        f.delete()
