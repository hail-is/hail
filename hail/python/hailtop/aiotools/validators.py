from hailtop.aiocloud.aiogoogle.client.storage_client import GoogleStorageAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import async_to_blocking
from textwrap import dedent
from typing import Optional
from urllib.parse import urlparse


def validate_file(uri: str, router_async_fs: RouterAsyncFS, *, validate_scheme: Optional[bool] = False) -> None:
    """
    Validates a URI's scheme if a file scheme cache was provided, and its cloud location's default storage policy if
    the URI points to a cloud with an ``AsyncFS`` implementation that supports checking that policy.

    Raises
    ------
    :class:`ValueError`
        If one of the validation steps fails.
    """
    if validate_scheme:
        scheme = urlparse(uri).scheme
        if not scheme or scheme == "file":
            raise ValueError(
                f"Local filepath detected: '{uri}'. The Hail Batch Service does not support the use of local "
                "filepaths. Please specify a remote URI instead (e.g. 'gs://bucket/folder')."
            )
    fs = router_async_fs._get_fs(uri)
    if isinstance(fs, GoogleStorageAsyncFS):
        location = fs.storage_location(uri)
        if location not in fs.allowed_storage_locations:
            if not async_to_blocking(fs.is_hot_storage(location)):
                raise ValueError(
                    dedent(
                        f"""\
                        GCS Bucket '{location}' is configured to use cold storage by default. Accessing the blob
                        '{uri}' would incur egress charges. Either

                        * avoid the increased cost by changing the default storage policy for the bucket
                          (https://cloud.google.com/storage/docs/changing-default-storage-class) and the individual
                          blobs in it (https://cloud.google.com/storage/docs/changing-storage-classes) to 'Standard', or

                        * accept the increased cost by adding '{location}' to the 'gcs_bucket_allow_list' configuration
                          variable (https://hail.is/docs/0.2/configuration_reference.html).
                        """
                    )
                )
            fs.allowed_storage_locations.append(location)
