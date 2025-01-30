from textwrap import dedent

from hailtop.aiocloud.aiogoogle.client.storage_client import GoogleStorageAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS


async def validate_file(uri: str, router_async_fs: RouterAsyncFS) -> None:
    fs = await router_async_fs._get_fs(uri)
    if isinstance(fs, GoogleStorageAsyncFS):
        if not await fs.is_hot_storage(uri):
            location = fs.storage_location(uri)
            raise ValueError(
                dedent(f"""\
                    GCS Bucket '{location}' is configured to use cold storage by default. Accessing the blob
                    '{uri}' would incur egress charges. Either

                    * avoid the increased cost by changing the default storage policy for the bucket
                      (https://cloud.google.com/storage/docs/changing-default-storage-class) and the individual
                      blobs in it (https://cloud.google.com/storage/docs/changing-storage-classes) to 'Standard', or

                    * accept the increased cost by adding '{location}' to the 'gcs_bucket_allow_list' configuration
                      variable (https://hail.is/docs/0.2/configuration_reference.html).
                    """)
            )
