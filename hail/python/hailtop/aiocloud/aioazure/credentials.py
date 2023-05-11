import base64
import os
import json
import jwt
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import time
import logging
from typing import Any, Dict, List, Optional, TypedDict, Union
from azure.identity.aio import DefaultAzureCredential, ClientSecretCredential

from hailtop import httpx
from hailtop.utils import first_extant_file

from ..common.credentials import CloudCredentials

log = logging.getLogger(__name__)


class AadJwk(TypedDict):
    kid: str
    x5c: List[str]


class AzureCredentials(CloudCredentials):
    _aad_keys: Optional[List[AadJwk]] = None

    @classmethod
    async def userinfo_from_access_token(cls, session: httpx.ClientSession, access_token) -> Dict[str, Any]:
        if cls._aad_keys is None:
            resp = await session.get_read_json('https://login.microsoftonline.com/common/discovery/keys')
            cls._aad_keys = resp['keys']

        assert cls._aad_keys
        kid = jwt.get_unverified_header(access_token)['kid']
        jwk = [key for key in cls._aad_keys if key['kid'] == kid][0]
        der_cert = base64.b64decode(jwk['x5c'][0])
        cert = x509.load_der_x509_certificate(der_cert, default_backend())
        pem_key = cert.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()

        return jwt.decode(access_token, pem_key, algorithms=['RS256'], options={'verify_aud': False})

    @staticmethod
    def from_credentials_data(credentials: dict, scopes: Optional[List[str]] = None):
        credential = ClientSecretCredential(tenant_id=credentials['tenant'],
                                            client_id=credentials['appId'],
                                            client_secret=credentials['password'])
        return AzureCredentials(credential, scopes)

    @staticmethod
    def from_file(credentials_file: str, scopes: Optional[List[str]] = None):
        with open(credentials_file, 'r', encoding='utf-8') as f:
            credentials = json.loads(f.read())
            return AzureCredentials.from_credentials_data(credentials, scopes)

    @staticmethod
    def default_credentials(scopes: Optional[List[str]] = None):
        credentials_file = first_extant_file(
            os.environ.get('AZURE_APPLICATION_CREDENTIALS'),
            '/azure-credentials/credentials.json',
            '/gsa-key/key.json'  # FIXME: make this file path cloud-agnostic
        )

        if credentials_file:
            log.info(f'using credentials file {credentials_file}')
            return AzureCredentials.from_file(credentials_file, scopes)

        return AzureCredentials(DefaultAzureCredential(), scopes)

    def __init__(self, credential, scopes: Optional[List[str]] = None):
        self.credential: Union[DefaultAzureCredential, ClientSecretCredential] = credential
        self._access_token = None
        self._expires_at = None

        if scopes is None:
            scopes = ['https://management.azure.com/.default']
        self.scopes = scopes

    async def auth_headers(self):
        return {'Authorization': f'Bearer {await self.access_token()}'}  # type: ignore

    async def access_token(self) -> str:
        now = time.time()
        if self._access_token is None or (self._expires_at is not None and now > self._expires_at):
            self._access_token = await self.get_access_token()
            self._expires_at = now + (self._access_token.expires_on - now) // 2   # type: ignore
        assert self._access_token
        return self._access_token.token

    async def email(self) -> str:
        async with httpx.client_session() as session:
            userinfo = await self.userinfo_from_access_token(session, await self.access_token())
            return userinfo['unique_name']

    @property
    def login_cli(self) -> str:
        return 'az'

    @property
    def login_command(self) -> str:
        return 'az login'

    async def get_access_token(self):
        return await self.credential.get_token(*self.scopes)

    async def close(self):
        await self.credential.close()
