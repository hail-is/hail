import warnings

from ..aiocloud.aiogoogle import (
    GCSRequesterPaysConfiguration,
    GoogleApplicationDefaultCredentials,
    GoogleBigQueryClient,
    GoogleBillingClient,
    GoogleComputeClient,
    GoogleContainerClient,
    GoogleCredentials,
    GoogleIAmClient,
    GoogleInstanceMetadataCredentials,
    GoogleLoggingClient,
    GoogleMetadataServerClient,
    GoogleServiceAccountCredentials,
    GoogleStorageAsyncFS,
    GoogleStorageAsyncFSFactory,
    GoogleStorageClient,
    get_gcs_requester_pays_configuration,
)

warnings.warn(
    "importing hailtop.aiogoogle is deprecated, please use hailtop.aiocloud.aiogoogle", DeprecationWarning, stacklevel=2
)

__all__ = [
    'GCSRequesterPaysConfiguration',
    'GoogleCredentials',
    'GoogleApplicationDefaultCredentials',
    'GoogleServiceAccountCredentials',
    'GoogleInstanceMetadataCredentials',
    'GoogleBigQueryClient',
    'GoogleBillingClient',
    'GoogleContainerClient',
    'GoogleComputeClient',
    'GoogleIAmClient',
    'GoogleLoggingClient',
    'GoogleMetadataServerClient',
    'GoogleStorageClient',
    'GoogleStorageAsyncFS',
    'GoogleStorageAsyncFSFactory',
    'get_gcs_requester_pays_configuration',
]
