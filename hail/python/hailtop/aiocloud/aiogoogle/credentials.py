from typing import Dict, Optional
import os
import json
import time
import logging
from urllib.parse import urlencode
import jwt
from hailtop.utils import request_retry_transient_errors
import hailtop.httpx
from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)


class GoogleExpiringAccessToken:
    @staticmethod
    def from_dict(data: dict) -> 'GoogleExpiringAccessToken':
        now = time.time()
        token = data['access_token']
        expiry_time = now + data['expires_in'] // 2
        return GoogleExpiringAccessToken(token, expiry_time)

    def __init__(self, token, expiry_time: int):
        self.token = token
        self._expiry_time = expiry_time

    def expired(self) -> bool:
        now = time.time()
        return self._expiry_time <= now


class GoogleCredentials(CloudCredentials):
    _http_session: hailtop.httpx.ClientSession

    def __init__(self,
                 http_session: Optional[hailtop.httpx.ClientSession] = None,
                 **kwargs):
        self._access_token: Optional[GoogleExpiringAccessToken] = None
        if http_session is not None:
            assert len(kwargs) == 0
            self._http_session = http_session
        else:
            self._http_session = hailtop.httpx.ClientSession(**kwargs)

    @staticmethod
    def from_file(credentials_file: str) -> 'GoogleCredentials':
        with open(credentials_file) as f:
            credentials = json.load(f)
        return GoogleCredentials.from_credentials_data(credentials)

    @staticmethod
    def from_credentials_data(credentials: dict, **kwargs) -> 'GoogleCredentials':
        credentials_type = credentials['type']
        if credentials_type == 'service_account':
            return GoogleServiceAccountCredentials(credentials, **kwargs)

        if credentials_type == 'authorized_user':
            return GoogleApplicationDefaultCredentials(credentials, **kwargs)

        raise ValueError(f'unknown Google Cloud credentials type {credentials_type}')

    @staticmethod
    def default_credentials() -> 'GoogleCredentials':
        credentials_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        if credentials_file is None:
            application_default_credentials_file = f'{os.environ["HOME"]}/.config/gcloud/application_default_credentials.json'
            if os.path.exists(application_default_credentials_file):
                credentials_file = application_default_credentials_file

        if credentials_file:
            creds = GoogleCredentials.from_file(credentials_file)
            log.info(f'using credentials file {credentials_file}: {creds}')
            return creds

        log.warning('unable to locate Google Cloud credentials file, will attempt to '
                    'use instance metadata server instead')

        return GoogleInstanceMetadataCredentials()

    async def auth_headers(self) -> Dict[str, str]:
        if self._access_token is None or self._access_token.expired():
            self._access_token = await self._get_access_token()
        return {'Authorization': f'Bearer {self._access_token.token}'}

    async def _get_access_token(self) -> GoogleExpiringAccessToken:
        raise NotImplementedError

    async def close(self):
        await self._http_session.close()


# protocol documented here:
# https://developers.google.com/identity/protocols/oauth2/web-server#offline
# studying `gcloud --log-http print-access-token` was also useful
class GoogleApplicationDefaultCredentials(GoogleCredentials):
    def __init__(self, credentials, **kwargs):
        super().__init__(**kwargs)
        self.credentials = credentials

    def __str__(self):
        return 'ApplicationDefaultCredentials'

    async def _get_access_token(self) -> GoogleExpiringAccessToken:
        async with await request_retry_transient_errors(
                self._http_session, 'POST',
                'https://www.googleapis.com/oauth2/v4/token',
                headers={
                    'content-type': 'application/x-www-form-urlencoded'
                },
                data=urlencode({
                    'grant_type': 'refresh_token',
                    'client_id': self.credentials['client_id'],
                    'client_secret': self.credentials['client_secret'],
                    'refresh_token': self.credentials['refresh_token']
                })) as resp:
            return GoogleExpiringAccessToken.from_dict(await resp.json())


# protocol documented here:
# https://developers.google.com/identity/protocols/oauth2/service-account
# studying `gcloud --log-http print-access-token` was also useful
class GoogleServiceAccountCredentials(GoogleCredentials):
    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def __str__(self):
        return f'GoogleServiceAccountCredentials for {self.key["client_email"]}'

    async def _get_access_token(self) -> GoogleExpiringAccessToken:
        now = int(time.time())
        scope = 'openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/appengine.admin https://www.googleapis.com/auth/compute'
        assertion = {
            "aud": "https://www.googleapis.com/oauth2/v4/token",
            "iat": now,
            "scope": scope,
            "exp": now + 300,  # 5m
            "iss": self.key['client_email']
        }
        encoded_assertion = jwt.encode(assertion, self.key['private_key'], algorithm='RS256')
        async with await request_retry_transient_errors(
                self._http_session, 'POST',
                'https://www.googleapis.com/oauth2/v4/token',
                headers={
                    'content-type': 'application/x-www-form-urlencoded'
                },
                data=urlencode({
                    'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                    'assertion': encoded_assertion
                })) as resp:
            return GoogleExpiringAccessToken.from_dict(await resp.json())


# https://cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances#applications
class GoogleInstanceMetadataCredentials(GoogleCredentials):
    async def _get_access_token(self) -> GoogleExpiringAccessToken:
        async with await request_retry_transient_errors(
                self._http_session, 'GET',
                'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                headers={'Metadata-Flavor': 'Google'}) as resp:
            return GoogleExpiringAccessToken.from_dict(await resp.json())
