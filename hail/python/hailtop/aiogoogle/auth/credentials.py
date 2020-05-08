import abc
import os
import json
import time
import logging
from urllib.parse import urlencode
import jwt
from hailtop.utils import request_retry_transient_errors

log = logging.getLogger(__name__)


class Credentials(abc.ABC):
    @staticmethod
    def from_file(credentials_file):
        with open(credentials_file) as f:
            credentials = json.loads(f.read())

        credentials_type = credentials['type']
        if credentials_type == 'service_account':
            return ServiceAccountCredentials(credentials)

        if credentials_type == 'authorized_user':
            return ApplicationDefaultCredentials(credentials)

        raise ValueError(f'unknown Google Cloud credentials type {credentials_type}')

    @staticmethod
    def default_credentials():
        credentials_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

        if credentials_file is None:
            application_default_credentials_file = f'{os.environ["HOME"]}/.config/gcloud/application_default_credentials.json'
            if os.path.exists(application_default_credentials_file):
                credentials_file = application_default_credentials_file

        if credentials_file is None:
            raise ValueError('unable to locate Google Cloud credentials')

        log.info(f'using credentials file {credentials_file}')

        return Credentials.from_file(credentials_file)

    async def get_access_token(self, session):
        pass


# protocol documented here:
# https://developers.google.com/identity/protocols/oauth2/web-server#offline
# studying `gcloud --log-http print-access-token` was also useful
class ApplicationDefaultCredentials(Credentials):
    def __init__(self, credentials):
        self.credentials = credentials

    async def get_access_token(self, session):
        async with await request_retry_transient_errors(
                session, 'POST',
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
class ServiceAccountCredentials(Credentials):
    def __init__(self, key):
        self.key = key

    async def get_access_token(self, session):
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
                session, 'POST',
                'https://www.googleapis.com/oauth2/v4/token',
                headers={
                    'content-type': 'application/x-www-form-urlencoded'
                },
                data=urlencode({
                    'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                    'assertion': encoded_assertion
                })) as resp:
            return await resp.json()
