import base64
import datetime
import logging
import os
import time
from typing import Dict, Optional, Tuple

from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from hailtop import httpx
from hailtop.utils import retry_transient_errors

from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)

HAIL_AWS_TOKEN_PREFIX = 'hail-aws-v1.'
HAIL_AWS_AUDIENCE_HEADER = 'x-hail-audience'


class AmazonExpiringCredentials:
    @staticmethod
    def from_dict(data: dict) -> 'AmazonExpiringCredentials':
        expiration = datetime.datetime.fromisoformat(data['Expiration']).timestamp()
        return AmazonExpiringCredentials(
            data['AccessKeyId'], data['SecretAccessKey'], data['Token'], expiration, data.get('AccountId')
        )

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        session_token: str,
        expiration: float,
        account_id: Optional[str] = None,
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.expiration = expiration
        self.account_id = account_id
        # refresh halfway to expiry, like GoogleExpiringAccessToken
        now = time.time()
        self._refresh_time = now + (expiration - now) / 2

    def expired(self) -> bool:
        return self._refresh_time <= time.time()

    def to_botocore(self) -> Credentials:
        return Credentials(self.access_key_id, self.secret_access_key, self.session_token)


# for Hail system credentials fetched using EKS Pod Identity
class AmazonPodServiceAccountCredentials(CloudCredentials):
    FULL_URI_ENV_VAR = 'AWS_CONTAINER_CREDENTIALS_FULL_URI'
    TOKEN_FILE_ENV_VAR = 'AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE'

    def __init__(
        self,
        http_session: Optional[httpx.ClientSession] = None,
        *,
        region: Optional[str] = None,
        audience: Optional[str] = None,
        token_expires_in: int = 60,
        **kwargs,
    ):
        if http_session is not None:
            assert len(kwargs) == 0
            self._http_session = http_session
        else:
            self._http_session = httpx.ClientSession(**kwargs)
        self._credentials: Optional[AmazonExpiringCredentials] = None
        region = region or os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
        if region is None:
            raise ValueError('AWS region is not set; pass region= or set AWS_REGION')
        self._region = region
        self._audience = audience
        self._token_expires_in = token_expires_in

    def __str__(self):
        return 'AmazonPodServiceAccountCredentials'

    @staticmethod
    def available() -> bool:
        return (
            AmazonPodServiceAccountCredentials.FULL_URI_ENV_VAR in os.environ
            and AmazonPodServiceAccountCredentials.TOKEN_FILE_ENV_VAR in os.environ
        )

    async def aws_credentials(self) -> AmazonExpiringCredentials:
        if self._credentials is None or self._credentials.expired():
            self._credentials = await self._get_credentials()
        return self._credentials

    async def _get_credentials(self) -> AmazonExpiringCredentials:
        full_uri = os.environ[self.FULL_URI_ENV_VAR]
        with open(os.environ[self.TOKEN_FILE_ENV_VAR], 'r', encoding='utf-8') as f:
            authorization_token = f.read()

        data = await retry_transient_errors(
            self._http_session.get_read_json,
            full_uri,
            headers={'Authorization': authorization_token, 'Accept': 'application/json'},
        )
        return AmazonExpiringCredentials.from_dict(data)

    async def access_token_with_expiration(self) -> Tuple[str, Optional[float]]:
        creds = await self.aws_credentials()
        headers = {HAIL_AWS_AUDIENCE_HEADER: self._audience} if self._audience else {}
        request = AWSRequest(
            method='GET',
            url=f'https://sts.{self._region}.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15',
            headers=headers,
        )
        SigV4QueryAuth(creds.to_botocore(), 'sts', self._region, expires=self._token_expires_in).add_auth(request)
        encoded = base64.urlsafe_b64encode(request.url.encode()).decode().rstrip('=')

        expiration = min(creds.expiration, time.time() + self._token_expires_in - 10)
        return HAIL_AWS_TOKEN_PREFIX + encoded, expiration

    async def auth_headers_with_expiration(self) -> Tuple[Dict[str, str], Optional[float]]:
        token, expiration = await self.access_token_with_expiration()
        return {'Authorization': f'Bearer {token}'}, expiration

    async def close(self):
        await self._http_session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
