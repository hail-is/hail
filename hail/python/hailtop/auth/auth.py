from typing import Any, Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
import os
import json
import aiohttp

from hailtop import httpx
from hailtop.aiocloud.common.credentials import CloudCredentials
from hailtop.aiocloud.common import Session
from hailtop.aiocloud.aiogoogle import GoogleCredentials
from hailtop.aiocloud.aioazure import AzureCredentials
from hailtop.config import get_deploy_config, DeployConfig, get_user_identity_config_path
from hailtop.utils import async_to_blocking, retry_transient_errors

from .tokens import get_tokens, Tokens


class IdentityProvider(Enum):
    GOOGLE = 'Google'
    MICROSOFT = 'Microsoft'


@dataclass
class IdentityProviderSpec:
    idp: IdentityProvider
    # Absence of specific oauth credentials means Hail should use latent credentials
    oauth2_credentials: Optional[dict]

    @staticmethod
    def from_json(config: Dict[str, Any]):
        return IdentityProviderSpec(IdentityProvider(config['idp']), config.get('credentials'))


class HailCredentials(CloudCredentials):
    def __init__(self, tokens: Tokens, cloud_credentials: Optional[CloudCredentials], namespace: str, authorize_target: bool):
        self._tokens = tokens
        self._cloud_credentials = cloud_credentials
        self._namespace = namespace
        self._authorize_target = authorize_target

    async def auth_headers(self) -> Dict[str, str]:
        headers = {}
        if self._authorize_target:
            token = await self._get_idp_access_token_or_hail_token(self._namespace)
            headers['Authorization'] = f'Bearer {token}'
        if get_deploy_config().location() == 'external' and self._namespace != 'default':
            # We prefer an extant hail token to an access token for the internal auth token
            # during development of the idp access token feature because the production auth
            # is not yet configured to accept access tokens. This can be changed to always prefer
            # an idp access token when this change is in production.
            token = await self._get_hail_token_or_idp_access_token('default')
            headers['X-Hail-Internal-Authorization'] = f'Bearer {token}'
        return headers

    async def access_token(self) -> str:
        return await self._get_idp_access_token_or_hail_token(self._namespace)

    async def _get_idp_access_token_or_hail_token(self, namespace: str) -> str:
        if self._cloud_credentials is not None:
            return await self._cloud_credentials.access_token()
        return self._tokens.namespace_token_or_error(namespace)

    async def _get_hail_token_or_idp_access_token(self, namespace: str) -> str:
        if self._cloud_credentials is None:
            return self._tokens.namespace_token_or_error(namespace)
        return self._tokens.namespace_token(namespace) or await self._cloud_credentials.access_token()

    async def close(self):
        if self._cloud_credentials:
            await self._cloud_credentials.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


def hail_credentials(
    *,
    tokens_file: Optional[str] = None,
    namespace: Optional[str] = None,
    authorize_target: bool = True
) -> HailCredentials:
    tokens = get_tokens(tokens_file)
    deploy_config = get_deploy_config()
    ns = namespace or deploy_config.default_namespace()
    return HailCredentials(tokens, get_cloud_credentials_scoped_for_hail(), ns, authorize_target=authorize_target)


def get_cloud_credentials_scoped_for_hail() -> Optional[CloudCredentials]:
    scopes: Optional[List[str]]

    spec = load_identity_spec()
    if spec is None:
        return None

    if spec.idp == IdentityProvider.GOOGLE:
        scopes = ['email', 'openid', 'profile']
        if spec.oauth2_credentials is not None:
            return GoogleCredentials.from_credentials_data(spec.oauth2_credentials, scopes=scopes)
        return GoogleCredentials.default_credentials(scopes=scopes, anonymous_ok=False)

    assert spec.idp == IdentityProvider.MICROSOFT
    if spec.oauth2_credentials is not None:
        return AzureCredentials.from_credentials_data(spec.oauth2_credentials, scopes=[spec.oauth2_credentials['userOauthScope']])

    if 'HAIL_AZURE_OAUTH_SCOPE' in os.environ:
        scopes = [os.environ["HAIL_AZURE_OAUTH_SCOPE"]]
    else:
        scopes = None
    return AzureCredentials.default_credentials(scopes=scopes)


def load_identity_spec() -> Optional[IdentityProviderSpec]:
    if idp := os.environ.get('HAIL_IDENTITY_PROVIDER_JSON'):
        return IdentityProviderSpec.from_json(json.loads(idp))

    identity_file = get_user_identity_config_path()
    if os.path.exists(identity_file):
        with open(identity_file, 'r', encoding='utf-8') as f:
            return IdentityProviderSpec.from_json(json.loads(f.read()))

    return None


async def deploy_config_and_headers_from_namespace(namespace: Optional[str] = None, *, authorize_target: bool = True) -> Tuple[DeployConfig, Dict[str, str], str]:
    deploy_config = get_deploy_config()

    if namespace is not None:
        deploy_config = deploy_config.with_default_namespace(namespace)
    else:
        namespace = deploy_config.default_namespace()


    async with hail_credentials(namespace=namespace, authorize_target=authorize_target) as credentials:
        headers = await credentials.auth_headers()

    return (deploy_config, headers, namespace)


async def async_get_userinfo():
    deploy_config = get_deploy_config()
    userinfo_url = deploy_config.url('auth', '/api/v1alpha/userinfo')

    async with hail_credentials() as credentials:
        async with Session(credentials=credentials) as session:
            try:
                async with await session.get(userinfo_url) as resp:
                    return await resp.json()
            except aiohttp.ClientResponseError as err:
                if err.status == 401:
                    return None
                raise


def get_userinfo():
    return async_to_blocking(async_get_userinfo())


def copy_paste_login(copy_paste_token: str, namespace: Optional[str] = None):
    return async_to_blocking(async_copy_paste_login(copy_paste_token, namespace))


async def async_copy_paste_login(copy_paste_token: str, namespace: Optional[str] = None):
    deploy_config, headers, namespace = await deploy_config_and_headers_from_namespace(namespace, authorize_target=False)
    async with httpx.client_session(headers=headers) as session:
        data = await retry_transient_errors(
            session.post_read_json,
            deploy_config.url('auth', '/api/v1alpha/copy-paste-login'),
            params={'copy_paste_token': copy_paste_token}
        )
    token = data['token']
    username = data['username']

    tokens = get_tokens()
    tokens[namespace] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    return namespace, username


# TODO Logging out should revoke the refresh token and delete the credentials file
async def async_logout():
    deploy_config = get_deploy_config()

    auth_ns = deploy_config.service_ns('auth')
    tokens = get_tokens()
    if auth_ns not in tokens:
        print('Not logged in.')
        return

    headers = await hail_credentials().auth_headers()
    async with httpx.client_session(headers=headers) as session:
        async with session.post(deploy_config.url('auth', '/api/v1alpha/logout')):
            pass
    auth_ns = deploy_config.service_ns('auth')

    del tokens[auth_ns]
    tokens.write()


def get_user(username: str, namespace: Optional[str] = None) -> dict:
    return async_to_blocking(async_get_user(username, namespace))


async def async_get_user(username: str, namespace: Optional[str] = None) -> dict:
    deploy_config, headers, _ = await deploy_config_and_headers_from_namespace(namespace)

    async with httpx.client_session(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers) as session:
        return await retry_transient_errors(
            session.get_read_json,
            deploy_config.url('auth', f'/api/v1alpha/users/{username}')
        )


async def async_create_user(
    username: str,
    login_id: str,
    is_developer: bool,
    is_service_account: bool,
    hail_identity: Optional[str],
    hail_credentials_secret_name: Optional[str],
    *,
    namespace: Optional[str] = None
):
    deploy_config, headers, _ = await deploy_config_and_headers_from_namespace(namespace)

    body = {
        'login_id': login_id,
        'is_developer': is_developer,
        'is_service_account': is_service_account,
        'hail_identity': hail_identity,
        'hail_credentials_secret_name': hail_credentials_secret_name,
    }

    async with httpx.client_session(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers) as session:
        await retry_transient_errors(
            session.post,
            deploy_config.url('auth', f'/api/v1alpha/users/{username}/create'),
            json=body
        )


def delete_user(username: str, namespace: Optional[str] = None):
    return async_to_blocking(async_delete_user(username, namespace=namespace))


async def async_delete_user(username: str, namespace: Optional[str] = None):
    deploy_config, headers, _ = await deploy_config_and_headers_from_namespace(namespace)
    async with httpx.client_session(
            timeout=aiohttp.ClientTimeout(total=300),
            headers=headers) as session:
        await retry_transient_errors(
            session.delete,
            deploy_config.url('auth', f'/api/v1alpha/users/{username}')
        )
