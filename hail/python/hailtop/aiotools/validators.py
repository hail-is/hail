from hailtop.hail_event_loop import hail_event_loop
from hailtop.aiocloud.aiogoogle.client.storage_client import GoogleStorageAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from typing import Optional
from urllib.parse import urlparse


def validate_file(uri: str, router_async_fs: RouterAsyncFS, *, validate_scheme: Optional[bool] = False) -> None:
    """
    Validates a URI's scheme if the ``validate_scheme`` flag was provided, and its cloud location's default storage
    policy if the URI points to a cloud with an ``AsyncFS`` implementation that supports checking that policy.

    Raises
    ------
    :class:`ValueError`
        If one of the validation steps fails.
    """
    return hail_event_loop().run_until_complete(
        _async_validate_file(uri, router_async_fs, validate_scheme=validate_scheme)
    )


async def _async_validate_file(
    uri: str, router_async_fs: RouterAsyncFS, *, validate_scheme: Optional[bool] = False
) -> None:
    if validate_scheme:
        scheme = urlparse(uri).scheme
        if not scheme or scheme == "file":
            raise ValueError(
                f"Local filepath detected: '{uri}'. The Hail Batch Service does not support the use of local "
                "filepaths. Please specify a remote URI instead (e.g. 'gs://bucket/folder')."
            )
    fs = await router_async_fs._get_fs(uri)
    if isinstance(fs, GoogleStorageAsyncFS):
        await fs.is_hot_storage(uri)
