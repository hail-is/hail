import os
import time
from typing import Dict, Optional, Tuple

from ..common.credentials import CloudCredentials
from hailtop.utils import retry_transient_errors


class AmazonExpiringAccessToken:
    @staticmethod
    def from_dict(data: dict) -> 'AmazonExpiringAccessToken':
        access_key = data['accessKeyId']
        secret_access_key = data['secretAccessKey']
        session_token = data['sessionToken']
        expiration = data['expiration']
        return AmazonExpiringAccessToken(access_key, secret_access_key, session_token, expiration)

    def __init__(self, access_key: str, secret_access_key: str, session_token: str, expiration: int):
        self._access_key = access_key
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._expiration = expiration

    def expired(self) -> bool:
        now = time.time()
        return self._expiration <= now

class AmazonPodServiceAccountCredentials(CloudCredentials):
    def __init__(
            self, http_session: Optional[httpx.ClientSession] = None, scopes: Optional[List[str]] = None, **kwargs
    ):
        self._access_token: Optional[AmazonExpiringAccessToken] = None
        if http_session is not None:
            assert len(kwargs) == 0
            self._http_session = http_session
        else:
            self._http_session = httpx.ClientSession(**kwargs)


    async def auth_headers_with_expiration(self) -> Tuple[Dict[str, str], Optional[float]]:
        """Return HTTP authentication headers and the time of expiration in seconds since the epoch (Unix time).

        None indicates a non-expiring credentials."""
        raise NotImplementedError

    async def access_token_with_expiration(self) -> Tuple[str, Optional[float]]:
        token_file = os.environ.get('AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE')
        if token_file:
            with open(token_file, 'r', encoding='utf-8') as f:
                authorization_token = f.read().strip()
                token_dict = await retry_transient_errors(
                    self._http_session.post_read_json,
                    os.environ.get('AWS_CONTAINER_CREDENTIALS_FULL_URI'),
                    headers={'content-type': 'application/x-www-form-urlencoded'},
                    data=urlencode({
                        'token': authorization_token,
                    }),
                )
                service_account_creds = AmazonExpiringAccessToken.from_dict(token_dict.get('credentials', {}))
                return service_account_creds._
        else:
            raise ValueError("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE environment variable is not set")


    async def auth_headers(self) -> Dict[str, str]:
        headers, _ = await self.auth_headers_with_expiration()
        return headers

    async def access_token(self) -> str:
        access_token, _ = await self.access_token_with_expiration()
        return access_token

    async def close(self):
        raise NotImplementedError
