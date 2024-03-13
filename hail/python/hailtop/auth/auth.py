import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from hailtop import httpx
from hailtop.aiocloud.aioazure import AzureCredentials
from hailtop.aiocloud.aiogoogle import GoogleCredentials
from hailtop.aiocloud.common import Session
from hailtop.aiocloud.common.credentials import CloudCredentials
from hailtop.config import DeployConfig, get_deploy_config, get_user_identity_config_path
from hailtop.utils import async_to_blocking, retry_transient_errors

from .flow import AzureFlow, GoogleFlow
from .tokens import Tokens, get_tokens


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
    def __init__(
        self,
        tokens: Tokens,
        cloud_credentials: Optional[CloudCredentials],
        deploy_config: DeployConfig,
        authorize_target: bool,
    ):
        self._tokens = tokens
        self._cloud_credentials = cloud_credentials
        self._deploy_config = deploy_config
        self._authorize_target = authorize_target

    async def auth_headers_with_expiration(self) -> Tuple[Dict[str, str], Optional[float]]:
        headers = {}
        expiration = None
        if self._authorize_target:
            token, expiration = await self._get_idp_access_token_or_hail_token(self._deploy_config.default_namespace())
            headers['Authorization'] = f'Bearer {token}'
        if get_deploy_config().location() == 'external' and self._deploy_config.default_namespace() != 'default':
            # We prefer an extant hail token to an access token for the internal auth token
            # during development of the idp access token feature because the production auth
            # is not yet configured to accept access tokens. This can be changed to always prefer
            # an idp access token when this change is in production.
            token, internal_expiration = await self._get_hail_token_or_idp_access_token('default')
            if internal_expiration:
                if not expiration:
                    expiration = internal_expiration
                else:
                    expiration = min(expiration, internal_expiration)
            headers['X-Hail-Internal-Authorization'] = f'Bearer {token}'
        return headers, expiration

    async def access_token_with_expiration(self) -> Tuple[str, Optional[float]]:
        return await self._get_idp_access_token_or_hail_token(self._deploy_config.default_namespace())

    async def _get_idp_access_token_or_hail_token(self, namespace: str) -> Tuple[str, Optional[float]]:
        if self._cloud_credentials is not None:
            return await self._cloud_credentials.access_token_with_expiration()
        return self._tokens.namespace_token_with_expiration_or_error(namespace)

    async def _get_hail_token_or_idp_access_token(self, namespace: str) -> Tuple[str, Optional[float]]:
        if self._cloud_credentials is None:
            return self._tokens.namespace_token_with_expiration_or_error(namespace)
        return (
            self._tokens.namespace_token_with_expiration(namespace)
            or await self._cloud_credentials.access_token_with_expiration()
        )

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
    cloud_credentials_file: Optional[str] = None,
    deploy_config: Optional[DeployConfig] = None,
    authorize_target: bool = True,
) -> HailCredentials:
    tokens = get_tokens(tokens_file)
    deploy_config = deploy_config or get_deploy_config()
    return HailCredentials(
        tokens,
        get_cloud_credentials_scoped_for_hail(credentials_file=cloud_credentials_file),
        deploy_config,
        authorize_target=authorize_target,
    )


def get_cloud_credentials_scoped_for_hail(credentials_file: Optional[str] = None) -> Optional[CloudCredentials]:
    scopes: Optional[List[str]]

    spec = load_identity_spec()
    if spec is None:
        return None

    if spec.idp == IdentityProvider.GOOGLE:
        scopes = ['email', 'openid', 'profile']
        if spec.oauth2_credentials is not None:
            return GoogleCredentials.from_credentials_data(spec.oauth2_credentials, scopes=scopes)
        if credentials_file is not None:
            return GoogleCredentials.from_file(credentials_file)
        return GoogleCredentials.default_credentials(scopes=scopes, anonymous_ok=False)

    assert spec.idp == IdentityProvider.MICROSOFT
    if spec.oauth2_credentials is not None:
        return AzureCredentials.from_credentials_data(
            spec.oauth2_credentials, scopes=[spec.oauth2_credentials['userOauthScope']]
        )

    if 'HAIL_AZURE_OAUTH_SCOPE' in os.environ:
        scopes = [os.environ["HAIL_AZURE_OAUTH_SCOPE"]]
    else:
        scopes = None

    if credentials_file is not None:
        return AzureCredentials.from_file(credentials_file, scopes=scopes)
    return AzureCredentials.default_credentials(scopes=scopes)


def load_identity_spec() -> Optional[IdentityProviderSpec]:
    if idp := os.environ.get('HAIL_IDENTITY_PROVIDER_JSON'):
        return IdentityProviderSpec.from_json(json.loads(idp))

    identity_file = get_user_identity_config_path()
    if os.path.exists(identity_file):
        with open(identity_file, 'r', encoding='utf-8') as f:
            return IdentityProviderSpec.from_json(json.loads(f.read()))

    return None


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


def copy_paste_login(copy_paste_token: str) -> str:
    return async_to_blocking(async_copy_paste_login(copy_paste_token))


async def async_copy_paste_login(copy_paste_token: str) -> str:
    deploy_config = get_deploy_config()
    async with httpx.client_session() as session:
        data = await retry_transient_errors(
            session.post_read_json,
            deploy_config.url('auth', '/api/v1alpha/copy-paste-login'),
            params={'copy_paste_token': copy_paste_token},
        )
    token = data['token']
    username = data['username']

    tokens = get_tokens()
    tokens[deploy_config.default_namespace()] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    return username


async def async_logout():
    deploy_config = get_deploy_config()

    # Logout any legacy auth tokens that might still exist
    auth_ns = deploy_config.default_namespace()
    tokens = get_tokens()
    if auth_ns in tokens:
        await logout_deprecated_token_credentials(deploy_config, tokens[auth_ns])
        del tokens[auth_ns]
        tokens.write()

    # Logout newer OAuth2-based credentials
    identity_spec = load_identity_spec()
    if identity_spec:
        await logout_oauth2_credentials(identity_spec)

    identity_config_path = get_user_identity_config_path()
    if os.path.exists(identity_config_path):
        os.remove(identity_config_path)


async def logout_deprecated_token_credentials(deploy_config, token):
    headers = {'Authorization': f'Bearer {token}'}
    async with httpx.client_session(headers=headers) as session:
        async with session.post(deploy_config.url('auth', '/api/v1alpha/logout')):
            pass


async def logout_oauth2_credentials(identity_spec: IdentityProviderSpec):
    if not identity_spec.oauth2_credentials:
        return

    if identity_spec.idp == IdentityProvider.GOOGLE:
        await GoogleFlow.logout_installed_app(identity_spec.oauth2_credentials)
    else:
        assert identity_spec.idp == IdentityProvider.MICROSOFT
        await AzureFlow.logout_installed_app(identity_spec.oauth2_credentials)


@asynccontextmanager
async def hail_session(**session_kwargs):
    async with hail_credentials() as credentials:
        async with Session(credentials=credentials, **session_kwargs) as session:
            yield session


def get_user(username: str) -> dict:
    return async_to_blocking(async_get_user(username))


async def async_get_user(username: str) -> dict:
    async with hail_session(timeout=aiohttp.ClientTimeout(total=30)) as session:
        url = get_deploy_config().url('auth', f'/api/v1alpha/users/{username}')
        async with await session.get(url) as resp:
            return await resp.json()


async def async_create_user(
    username: str,
    login_id: str,
    is_developer: bool,
    is_service_account: bool,
    hail_identity: Optional[str],
    hail_credentials_secret_name: Optional[str],
):
    body = {
        'login_id': login_id,
        'is_developer': is_developer,
        'is_service_account': is_service_account,
        'hail_identity': hail_identity,
        'hail_credentials_secret_name': hail_credentials_secret_name,
    }

    url = get_deploy_config().url('auth', f'/api/v1alpha/users/{username}/create')
    async with hail_session(timeout=aiohttp.ClientTimeout(total=30)) as session:
        await session.post(url, json=body)


def delete_user(username: str):
    return async_to_blocking(async_delete_user(username))


async def async_delete_user(username: str):
    url = get_deploy_config().url('auth', f'/api/v1alpha/users/{username}')
    async with hail_session(timeout=aiohttp.ClientTimeout(total=300)) as session:
        await session.delete(url)
