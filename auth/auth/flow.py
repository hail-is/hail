import abc
from typing import Optional, Tuple, Dict, Any
import msal
import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow
import aiohttp.web
import json
import urllib.parse

from gear.cloud_config import get_global_config, get_azure_config


class Flow(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def login_id_email_from_token(token: dict) -> Optional[Tuple[str, str]]:
        raise NotImplementedError

    @abc.abstractmethod
    def authorization_url_and_state(self) -> Tuple[str, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_and_verify_token(self, request: aiohttp.web.Request, flow_dict: Optional[dict]) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def as_dict(self):
        raise NotImplementedError


class GoogleFlow(Flow):
    @staticmethod
    def get_flow(credentials_file: str, redirect_uri: str, state: Optional[str] - None):
        scopes = [
            'https://www.googleapis.com/auth/userinfo.profile',
            'https://www.googleapis.com/auth/userinfo.email',
            'openid',
        ]

        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            credentials_file, scopes=scopes, state=state
        )
        flow.redirect_uri = redirect_uri

        return GoogleFlow(flow)

    @staticmethod
    def login_id_email_from_token(token: dict) -> Optional[Tuple[str, str]]:
        email = token['email']
        return (email, email)

    def __init__(self, flow: google_auth_oauthlib.flow.Flow):
        self._flow = flow

    def authorization_url_and_state(self) -> Tuple[str, str]:
        return self._flow.authorization_url(access_type='offline', include_granted_scopes='true')

    def fetch_and_verify_token(self, request: aiohttp.web.Request, flow_dict: Optional[dict]) -> Dict[str, Any]:  # pylint: disable=unused-argument
        self._flow.fetch_token(code=request.query['code'])
        token = google.oauth2.id_token.verify_oauth2_token(
            self._flow.credentials.id_token, google.auth.transport.requests.Request()
        )
        return token

    def as_dict(self):
        return None


class AzureFlow(Flow):
    @staticmethod
    def get_flow(credentials_file: str, redirect_uri: str, state: Optional[str] = None) -> 'AzureFlow':
        with open(credentials_file) as f:
            data = json.loads(f.read())

        tenant_id = data['tenant']
        authority = f'https://login.microsoftonline.com/{tenant_id}'
        client = msal.ConfidentialClientApplication(data['appId'], data['password'], authority)

        flow = client.initiate_auth_code_flow(
            scopes=[],
            redirect_uri=redirect_uri,
            state=state)
        return AzureFlow(client, flow, tenant_id)

    @staticmethod
    def login_id_email_from_token(token: dict) -> Optional[Tuple[str, str]]:
        return (token['id_token_claims']['oid'], token['id_token_claims']['preferred_username'])

    def __init__(self, client: msal.ClientApplication, flow_dict: dict, tenant_id: str):
        self._client = client
        self._flow_dict = flow_dict
        self._tenant_id = tenant_id

    def authorization_url_and_state(self) -> Tuple[str, str]:
        return (self._flow_dict['auth_uri'], self._flow_dict['state'])

    def fetch_and_verify_token(self, request: aiohttp.web.Request, flow_dict: Optional[dict]) -> Dict[str, Any]:
        query_dict = urllib.parse.parse_qs(request.query_string)
        query_dict = {k: v[0] for k, v in query_dict.items()}

        token = self._client.acquire_token_by_auth_code_flow(flow_dict, query_dict)

        if 'error' in token:
            raise Exception(f'{token}')

        tid = token['id_token_claims']['tid']
        if tid != self._tenant_id:
            raise Exception(f'invalid tenant id')

        return token

    def as_dict(self):
        return self._flow_dict


def get_flow(redirect_uri: str, state: Optional[str] = None, credentials_file: Optional[str] = None) -> Flow:
    if credentials_file is None:
        credentials_file = '/auth-oauth2-client-secret/client_secret.json'
    cloud = get_global_config()['cloud']
    if cloud == 'azure':
        return AzureFlow.get_flow(credentials_file, redirect_uri, state)
    assert cloud == 'gcp'
    return GoogleFlow.get_flow(credentials_file, redirect_uri, state)
