import base64
import datetime
import hashlib
import urllib.parse
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from hailtop import httpx
from hailtop.aiocloud.aiogoogle.client.storage_client import GoogleStorageAsyncFS
from hailtop.aiocloud.aiogoogle.credentials import GoogleCredentials, GoogleServiceAccountCredentials
from hailtop.utils import retry_transient_errors

SIGNED_URL_EXPIRATION_SECONDS = 900  # 15 minutes


class GCSSignedURLGenerator:
    def __init__(self):
        self._credentials = GoogleCredentials.default_credentials()
        # Only created lazily for non-SA credentials (e.g. Workload Identity in dev/test)
        self._http_session: Optional[httpx.ClientSession] = None
        self._sa_email: Optional[str] = None

    async def _get_sa_email(self) -> str:
        if self._sa_email is not None:
            return self._sa_email
        if isinstance(self._credentials, GoogleServiceAccountCredentials):
            self._sa_email = self._credentials.email
        else:
            if self._http_session is None:
                self._http_session = httpx.ClientSession()
            email_bytes = await retry_transient_errors(
                self._http_session.get_read,
                'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email',
                headers={'Metadata-Flavor': 'Google'},
            )
            self._sa_email = email_bytes.decode().strip()
        return self._sa_email

    def _sign_bytes(self, data: bytes) -> bytes:
        """Sign bytes using the SA private key (local) or raise if unavailable."""
        if not isinstance(self._credentials, GoogleServiceAccountCredentials):
            raise NotImplementedError('Local signing requires a service account key file')
        private_key = serialization.load_pem_private_key(self._credentials.key['private_key'].encode(), password=None)
        assert isinstance(private_key, RSAPrivateKey), 'service account key must be RSA'
        return private_key.sign(data, padding.PKCS1v15(), hashes.SHA256())

    async def _sign_bytes_via_iam(self, sa_email: str, data: bytes) -> bytes:
        """Sign bytes via IAM signBlob API (for non-SA credentials)."""
        if self._http_session is None:
            self._http_session = httpx.ClientSession()
        access_token, _ = await self._credentials.access_token_with_expiration()
        response = await retry_transient_errors(
            self._http_session.post_read_json,
            f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa_email}:signBlob',
            headers={'Authorization': f'Bearer {access_token}', 'content-type': 'application/json'},
            json={'payload': base64.b64encode(data).decode()},
        )
        return base64.b64decode(response['signedBlob'])

    async def generate_signed_url(self, gcs_url: str) -> Tuple[str, datetime.datetime]:
        """Return (signed_url, expires_at) for a GCS object."""
        bucket, name = GoogleStorageAsyncFS.get_bucket_and_name(gcs_url)
        sa_email = await self._get_sa_email()

        now = datetime.datetime.now(datetime.timezone.utc)
        date_str = now.strftime('%Y%m%d')
        datetime_str = now.strftime('%Y%m%dT%H%M%SZ')
        expires_at = now + datetime.timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS)

        host = 'storage.googleapis.com'
        encoded_name = '/'.join(urllib.parse.quote(part, safe='') for part in name.split('/'))
        path = f'/{bucket}/{encoded_name}'
        credential = f'{sa_email}/{date_str}/auto/storage/goog4_request'

        query_params = sorted([
            ('X-Goog-Algorithm', 'GOOG4-RSA-SHA256'),
            ('X-Goog-Credential', credential),
            ('X-Goog-Date', datetime_str),
            ('X-Goog-Expires', str(SIGNED_URL_EXPIRATION_SECONDS)),
            ('X-Goog-SignedHeaders', 'host'),
        ])
        canonical_query_string = urllib.parse.urlencode(query_params)

        canonical_request = '\n'.join([
            'GET',
            path,
            canonical_query_string,
            f'host:{host}\n',
            'host',
            'UNSIGNED-PAYLOAD',
        ])

        string_to_sign = '\n'.join([
            'GOOG4-RSA-SHA256',
            datetime_str,
            f'{date_str}/auto/storage/goog4_request',
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ])

        if isinstance(self._credentials, GoogleServiceAccountCredentials):
            signature = self._sign_bytes(string_to_sign.encode())
        else:
            signature = await self._sign_bytes_via_iam(sa_email, string_to_sign.encode())

        signed_url = f'https://{host}{path}?{canonical_query_string}&X-Goog-Signature={signature.hex()}'
        return signed_url, expires_at

    async def close(self):
        await self._credentials.close()
        if self._http_session is not None:
            await self._http_session.close()
