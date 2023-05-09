import abc
import json
import urllib.parse

import aiohttp.web
import google.auth.transport.requests
import google.oauth2.id_token
import google_auth_oauthlib.flow
import msal

from gear.cloud_config import get_global_config


class FlowResult:
    def __init__(self, login_id: str, email: str, token: dict):
        self.login_id = login_id
        self.email = email
        self.token = token


class Flow(abc.ABC):
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


class GoogleFlow(Flow):
    scopes = [
        'https://www.googleapis.com/auth/userinfo.profile',
        'https://www.googleapis.com/auth/userinfo.email',
        'openid',
    ]

    def __init__(self, credentials_file: str):
        self._credentials_file = credentials_file

    def initiate_flow(self, redirect_uri: str) -> dict:
        flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
            self._credentials_file, scopes=self.scopes, state=None
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
            self._credentials_file, scopes=self.scopes, state=flow_dict['state']
        )
        flow.redirect_uri = flow_dict['callback_uri']
        flow.fetch_token(code=request.query['code'])
        token = google.oauth2.id_token.verify_oauth2_token(
            flow.credentials.id_token, google.auth.transport.requests.Request()
        )
        email = token['email']
        return FlowResult(email, email, token)


class AzureFlow(Flow):
    def __init__(self, credentials_file: str):
        with open(credentials_file, encoding='utf-8') as f:
            data = json.loads(f.read())

        tenant_id = data['tenant']
        authority = f'https://login.microsoftonline.com/{tenant_id}'
        client = msal.ConfidentialClientApplication(data['appId'], data['password'], authority)

        self._client = client
        self._tenant_id = tenant_id

    def initiate_flow(self, redirect_uri: str) -> dict:
        flow = self._client.initiate_auth_code_flow(scopes=[], redirect_uri=redirect_uri)
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

        return FlowResult(token['id_token_claims']['oid'], token['id_token_claims']['preferred_username'], token)


def get_flow_client(credentials_file: str) -> Flow:
    cloud = get_global_config()['cloud']
    if cloud == 'azure':
        return AzureFlow(credentials_file)
    assert cloud == 'gcp'
    return GoogleFlow(credentials_file)
