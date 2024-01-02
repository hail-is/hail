import abc
import base64
from cryptography import x509
from cryptography.hazmat.primitives import serialization
import json
import logging
import os
import urllib.parse
from typing import Any, Dict, List, Mapping, Optional, TypedDict, ClassVar

import aiohttp.web
import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow
import jwt
import msal

from hailtop import httpx
from hailtop.utils import retry_transient_errors

log = logging.getLogger('auth')


class FlowResult:
    def __init__(self, login_id: str, unverified_email: str, organization_id: Optional[str], token: Mapping[Any, Any]):
        self.login_id = login_id
        self.unverified_email = unverified_email
        self.organization_id = organization_id  # In Azure, a Tenant ID. In Google, a domain name.
        self.token = token


class Flow(abc.ABC):
    @abc.abstractmethod
    def organization_id(self) -> str:
        """
        The unique identifier of the organization (e.g. Azure Tenant, Google Organization) in
        which this Hail Batch instance lives.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def initiate_flow(self, redirect_uri: str) -> dict:
        """
        Initiates the OAuth2 flow. Usually run in response to a user clicking a login button.
        The returned dict should be stored in a secure session so that the server can
        identify to which OAuth2 flow a client is responding. In particular, the server must
        pass this dict to :meth:`.receive_callback` in the OAuth2 callback.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def receive_callback(self, request: aiohttp.web.Request, flow_dict: dict) -> FlowResult:
        """Concludes the OAuth2 flow by returning the user's identity and credentials."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def perform_installed_app_login_flow(oauth2_client: Dict[str, Any]) -> Dict[str, Any]:
        """Performs an OAuth2 flow for credentials installed on the user's machine."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    async def logout_installed_app(oauth2_credentials: Dict[str, Any]):
        """Revokes the OAuth2 credentials on the user's machine."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    async def get_identity_uid_from_access_token(
        session: httpx.ClientSession, access_token: str, *, oauth2_client: dict
    ) -> Optional[str]:
        """
        Validate a user-provided access token. If the token is valid, return the identity
        to which it belongs. If it is not valid, return None.
        """
        raise NotImplementedError


class GoogleFlow(Flow):
    scopes: ClassVar[List[str]] = [
        'https://www.googleapis.com/auth/userinfo.profile',
        'https://www.googleapis.com/auth/userinfo.email',
        'openid',
    ]

    def __init__(self, credentials_file: str):
        self._credentials_file = credentials_file

    def organization_id(self) -> str:
        if organization_id := os.environ.get('HAIL_ORGANIZATION_DOMAIN'):
            return organization_id
        raise ValueError('Only available in the auth pod')

    def initiate_flow(self, redirect_uri: str) -> dict:
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self._credentials_file, scopes=GoogleFlow.scopes, state=None
        )
        flow.redirect_uri = redirect_uri
        authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')

        return {
            'authorization_url': authorization_url,
            'redirect_uri': redirect_uri,
            'state': state,
        }

    def receive_callback(self, request: aiohttp.web.Request, flow_dict: dict) -> FlowResult:
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self._credentials_file, scopes=GoogleFlow.scopes, state=flow_dict['state']
        )
        flow.redirect_uri = flow_dict['callback_uri']
        flow.fetch_token(code=request.query['code'])
        token = google.oauth2.id_token.verify_oauth2_token(
            flow.credentials.id_token, google.auth.transport.requests.Request()  # type: ignore
        )
        email = token['email']
        return FlowResult(email, email, token.get('hd'), token)

    @staticmethod
    def perform_installed_app_login_flow(oauth2_client: Dict[str, Any]) -> Dict[str, Any]:
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_config(oauth2_client, GoogleFlow.scopes)
        credentials = flow.run_local_server()
        return {
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'refresh_token': credentials.refresh_token,
            'type': 'authorized_user',
        }

    @staticmethod
    async def logout_installed_app(oauth2_credentials: Dict[str, Any]):
        async with httpx.client_session() as session:
            await session.post(
                'https://oauth2.googleapis.com/revoke',
                params={'token': oauth2_credentials['refresh_token']},
                headers={'content-type': 'application/x-www-form-urlencoded'},
            )

    @staticmethod
    async def get_identity_uid_from_access_token(
        session: httpx.ClientSession, access_token: str, *, oauth2_client: dict
    ) -> Optional[str]:
        oauth2_client_audience = oauth2_client['installed']['client_id']
        try:
            userinfo = await retry_transient_errors(
                session.get_read_json,
                'https://www.googleapis.com/oauth2/v3/tokeninfo',
                params={'access_token': access_token},
            )
            is_human_with_hail_audience = userinfo['aud'] == oauth2_client_audience
            is_service_account = userinfo['aud'] == userinfo['sub']
            if not (is_human_with_hail_audience or is_service_account):
                return None

            email = userinfo['email']
            if email.endswith('iam.gserviceaccount.com'):
                return userinfo['sub']
            # We don't currently track user's unique GCP IAM ID (sub) in the database, just their email,
            # but we should eventually use the sub as that is guaranteed to be unique to the user.
            return userinfo['email']
        except httpx.ClientResponseError as e:
            if e.status in (400, 401):
                return None
            raise


class AadJwk(TypedDict):
    kid: str
    x5c: List[str]


class AzureFlow(Flow):
    _aad_keys: Optional[List[AadJwk]] = None

    def __init__(self, credentials_file: str):
        with open(credentials_file, encoding='utf-8') as f:
            data = json.loads(f.read())

        tenant_id = data['tenant']
        authority = f'https://login.microsoftonline.com/{tenant_id}'
        self._client = msal.ConfidentialClientApplication(data['appId'], data['password'], authority)
        self._tenant_id = tenant_id

    def organization_id(self) -> str:
        return self._tenant_id

    def initiate_flow(self, redirect_uri: str) -> dict:
        flow = self._client.initiate_auth_code_flow(
            scopes=[],  # confusingly, scopes=[] is the only way to get the openid, profile, and
            # offline_access scopes
            # https://github.com/AzureAD/microsoft-authentication-library-for-python/blob/dev/msal/application.py#L568-L580
            redirect_uri=redirect_uri,
        )
        return {
            'flow': flow,
            'authorization_url': flow['auth_uri'],
            'state': flow['state'],
        }

    def receive_callback(self, request: aiohttp.web.Request, flow_dict: dict) -> FlowResult:
        query_key_to_list_of_values = urllib.parse.parse_qs(request.query_string)
        query_dict = {k: v[0] for k, v in query_key_to_list_of_values.items()}

        token = self._client.acquire_token_by_auth_code_flow(flow_dict['flow'], query_dict)

        if 'error' in token:
            raise ValueError(token)

        tid = token['id_token_claims']['tid']
        if tid != self._tenant_id:
            raise ValueError('invalid tenant id')

        return FlowResult(
            token['id_token_claims']['oid'],
            token['id_token_claims']['preferred_username'],
            token['id_token_claims']['tid'],
            token,
        )

    @staticmethod
    def perform_installed_app_login_flow(oauth2_client: Dict[str, Any]) -> Dict[str, Any]:
        tenant_id = oauth2_client['tenant']
        authority = f'https://login.microsoftonline.com/{tenant_id}'
        app = msal.PublicClientApplication(oauth2_client['appId'], authority=authority)
        credentials = app.acquire_token_interactive([oauth2_client['userOauthScope']])
        return {**oauth2_client, 'refreshToken': credentials['refresh_token']}

    @staticmethod
    async def logout_installed_app(_: Dict[str, Any]):
        # AAD does not support revocation of a single refresh token,
        # only all refresh tokens issued to all applications for a particular
        # user, which we neither wish nor should have the permissions
        # to perform.
        # See: https://learn.microsoft.com/en-us/answers/questions/1158831/invalidate-old-refresh-token-after-using-it-to-get
        pass

    @staticmethod
    async def get_identity_uid_from_access_token(
        session: httpx.ClientSession, access_token: str, *, oauth2_client: dict
    ) -> Optional[str]:
        audience = oauth2_client['appIdentifierUri']

        try:
            kid = jwt.get_unverified_header(access_token)['kid']

            if AzureFlow._aad_keys is None:
                resp = await session.get_read_json('https://login.microsoftonline.com/common/discovery/keys')
                AzureFlow._aad_keys = resp['keys']

            # This code is taken nearly verbatim from
            # https://github.com/AzureAD/microsoft-authentication-library-for-python/issues/147
            # At time of writing, the community response in that issue is the recommended way to validate
            # AAD access tokens in python as it is not a part of the MSAL library.

            jwk = next(key for key in AzureFlow._aad_keys if key['kid'] == kid)
            der_cert = base64.b64decode(jwk['x5c'][0])
            cert = x509.load_der_x509_certificate(der_cert)
            pem_key = (
                cert.public_key()
                .public_bytes(
                    encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                .decode()
            )

            decoded = jwt.decode(access_token, pem_key, algorithms=['RS256'], audience=audience)
            return decoded['oid']
        except jwt.InvalidTokenError:
            return None
