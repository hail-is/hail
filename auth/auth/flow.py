import abc
from typing import Optional
import msal
import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow
import aiohttp.web
import json
import urllib.parse

from gear.cloud_config import get_global_config


class FlowData:
    def __init__(self, authorization_url: str, state: str):
        self.authorization_url = authorization_url
        self.state = state


class FlowResult:
    def __init__(self, login_id: str, email: str, token: dict):
        self.login_id = login_id
        self.email = email
        self.token = token


class Flow(abc.ABC):
    @abc.abstractmethod
    def initialize_flow(self) -> FlowData:
        raise NotImplementedError

    @abc.abstractmethod
    def finish_flow(self, request: aiohttp.web.Request, flow_dict: Optional[dict]) -> FlowResult:
        raise NotImplementedError

    @abc.abstractmethod
    def as_dict(self):
        raise NotImplementedError


class GoogleFlow(Flow):
    @staticmethod
    def get_flow(credentials_file: str, redirect_uri: str, state: Optional[str] - None) -> 'GoogleFlow':
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

    def __init__(self, flow: google_auth_oauthlib.flow.Flow):
        self._flow = flow

    def initialize_flow(self) -> FlowData:
        authorization_url, state = self._flow.authorization_url(access_type='offline', include_granted_scopes='true')
        return FlowData(authorization_url, state)

    def finish_flow(self, request: aiohttp.web.Request, flow_dict: Optional[dict]) -> FlowResult:  # pylint: disable=unused-argument
        self._flow.fetch_token(code=request.query['code'])
        token = google.oauth2.id_token.verify_oauth2_token(
            self._flow.credentials.id_token, google.auth.transport.requests.Request()
        )
        email = token['email']
        return FlowResult(email, email, token)

    def as_dict(self) -> Optional[dict]:
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

    def __init__(self, client: msal.ClientApplication, flow_dict: dict, tenant_id: str):
        self._client = client
        self._flow_dict = flow_dict
        self._tenant_id = tenant_id

    def initialize_flow(self) -> FlowData:
        return FlowData(self._flow_dict['auth_uri'], self._flow_dict['state'])

    def finish_flow(self, request: aiohttp.web.Request, flow_dict: Optional[dict]) -> FlowResult:
        query_dict = urllib.parse.parse_qs(request.query_string)
        query_dict = {k: v[0] for k, v in query_dict.items()}

        token = self._client.acquire_token_by_auth_code_flow(flow_dict, query_dict)

        if 'error' in token:
            raise Exception(f'{token}')

        tid = token['id_token_claims']['tid']
        if tid != self._tenant_id:
            raise Exception('invalid tenant id')

        return FlowResult(token['id_token_claims']['oid'], token['id_token_claims']['preferred_username'], token)

    def as_dict(self) -> Optional[dict]:
        return self._flow_dict


def get_flow(redirect_uri: str, state: Optional[str] = None, credentials_file: Optional[str] = None) -> Flow:
    if credentials_file is None:
        credentials_file = '/auth-oauth2-client-secret/client_secret.json'
    cloud = get_global_config()['cloud']
    if cloud == 'azure':
        return AzureFlow.get_flow(credentials_file, redirect_uri, state)
    assert cloud == 'gcp'
    return GoogleFlow.get_flow(credentials_file, redirect_uri, state)
