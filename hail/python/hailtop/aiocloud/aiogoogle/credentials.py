import os
import json
import time
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode
import jwt
from hailtop.utils import request_retry_transient_errors
import hailtop.httpx
from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)


class GoogleCredentials(CloudCredentials):
    _session: hailtop.httpx.ClientSession
    _access_token: Optional[Dict[str, Any]]
    _expires_at: Optional[int]

    def __init__(self):
        self._access_token = None
        self._expires_at = None

    @staticmethod
    def from_file(credentials_file):
        with open(credentials_file) as f:
            credentials = json.load(f)
        return GoogleCredentials.from_credentials_data(credentials)

    @staticmethod
    def from_credentials_data(credentials):
        credentials_type = credentials['type']
        if credentials_type == 'service_account':
            return GoogleServiceAccountCredentials(credentials)

        if credentials_type == 'authorized_user':
            return GoogleApplicationDefaultCredentials(credentials)

        raise ValueError(f'unknown Google Cloud credentials type {credentials_type}')

    @staticmethod
    def default_credentials():
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

    async def auth_headers(self):
        now = time.time()
        if self._access_token is None or now > self._expires_at:
            self._access_token = await self.get_access_token()
            self._expires_at = now + self._access_token['expires_in'] // 2
        return {'Authorization': f'Bearer {self._access_token["access_token"]}'}

    async def get_access_token(self):
        raise NotImplementedError

    async def close(self):
        await self._session.close()


# protocol documented here:
# https://developers.google.com/identity/protocols/oauth2/web-server#offline
# studying `gcloud --log-http print-access-token` was also useful
class GoogleApplicationDefaultCredentials(GoogleCredentials):
    def __init__(self, credentials, **kwargs):
        super().__init__()
        self.credentials = credentials
        self._session = hailtop.httpx.ClientSession(**kwargs)

    def __str__(self):
        return 'ApplicationDefaultCredentials'

    async def get_access_token(self):
        async with await request_retry_transient_errors(
                self._session, 'POST',
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
            return await resp.json()


# protocol documented here:
# https://developers.google.com/identity/protocols/oauth2/service-account
# studying `gcloud --log-http print-access-token` was also useful
class GoogleServiceAccountCredentials(GoogleCredentials):
    def __init__(self, key, **kwargs):
        super().__init__()
        self.key = key
        self._session = hailtop.httpx.ClientSession(**kwargs)

    def __str__(self):
        return f'GoogleServiceAccountCredentials for {self.key["client_email"]}'

    async def get_access_token(self):
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
                self._session, 'POST',
                'https://www.googleapis.com/oauth2/v4/token',
                headers={
                    'content-type': 'application/x-www-form-urlencoded'
                },
                data=urlencode({
                    'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                    'assertion': encoded_assertion
                })) as resp:
            return await resp.json()


# https://cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances#applications
class GoogleInstanceMetadataCredentials(GoogleCredentials):
    def __init__(self, **kwargs):
        super().__init__()
        self._session = hailtop.httpx.ClientSession(**kwargs)

    async def get_access_token(self):
        async with await request_retry_transient_errors(
                self._session, 'GET',
                'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                headers={'Metadata-Flavor': 'Google'}) as resp:
            return await resp.json()
