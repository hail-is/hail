from typing import Optional, Dict, Tuple, TypedDict
from enum import Enum
import os
import json
import sys
import aiohttp

from hailtop import httpx
from hailtop.aiocloud.common.credentials import CloudCredentials, Credentials
from hailtop.aiocloud.common import Session
from hailtop.aiocloud.aiogoogle import GoogleCredentials
from hailtop.aiocloud.aioazure import AzureCredentials
from hailtop.config import get_deploy_config, DeployConfig
from hailtop.utils import async_to_blocking, retry_transient_errors

from .tokens import get_tokens, Tokens
from ..utils import first_extant_file


class IdentityProvider(Enum):
    GOOGLE = 'Google'
    MICROSOFT = 'Microsoft'


class IdentityProviderSpec(TypedDict):
    idp: IdentityProvider
    email: Optional[str]


class HailCredentials(Credentials):
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
            # during development of the idp access token feature becausea the production auth
            # is not yet configured to accept access tokens. This can be changed to always prefer
            # an idp access token when this change is in production.
            token = await self._get_hail_token_or_idp_access_token('default')
            headers['X-Hail-Internal-Authorization'] = f'Bearer {token}'
        return headers

    async def _get_idp_access_token_or_hail_token(self, namespace: str) -> str:
        if self._cloud_credentials is not None:
            return await self._cloud_credentials.access_token()
        return self._tokens.namespace_token_or_error(namespace)

    async def _get_hail_token_or_idp_access_token(self, namespace: str) -> str:
        if self._cloud_credentials is None:
            return self._tokens.namespace_token_or_error(namespace)
        return self._tokens.namespace_token(namespace) or await self._cloud_credentials.access_token()

    async def verify_user_login(self):
        if self._cloud_credentials is None:
            return

        identity_spec = load_identity_spec()
        assert identity_spec
        authenticated_identity = identity_spec['email']
        identity_used = await self._cloud_credentials.email()

        if authenticated_identity is not None and identity_used != authenticated_identity:
            sys.stderr.write(f'''\
You are authenticated with hail using {authenticated_identity}
but logged into {self._cloud_credentials.login_cli} with {identity_used}.
Please log in with:

  $ {self._cloud_credentials.login_command}

to obtain new credentials.
''')
            sys.exit(1)

    async def close(self):
        if self._cloud_credentials:
            await self._cloud_credentials.close()


async def hail_credentials(*, tokens_file: Optional[str] = None, namespace: Optional[str] = None, authorize_target: bool = True) -> HailCredentials:
    tokens = get_tokens(tokens_file)
    deploy_config = get_deploy_config()
    ns = namespace or deploy_config.default_namespace()

    identity_spec = load_identity_spec()
    if identity_spec is not None:
        if identity_spec['idp'] == IdentityProvider.GOOGLE:
            cloud_credentials = GoogleCredentials.default_credentials()
        else:
            assert identity_spec['idp'] == IdentityProvider.MICROSOFT
            cloud_credentials = AzureCredentials.default_credentials()
    else:
        cloud_credentials = None

    return HailCredentials(tokens, cloud_credentials, ns, authorize_target=authorize_target)


def get_identity_spec_file() -> str:
    return first_extant_file(
        os.environ.get('HAIL_IDENTITY_FILE'),
    ) or os.path.expanduser('~/.hail/identity.json')


def load_identity_spec() -> Optional[IdentityProviderSpec]:
    idp = os.environ.get('HAIL_IDENTITY_PROVIDER')
    if idp is not None:
        return {'idp': IdentityProvider(idp), 'email': None}

    identity_file = get_identity_spec_file()
    if identity_file is not None and os.path.exists(identity_file):
        with open(identity_file, 'r', encoding='utf-8') as f:
            config = json.loads(f.read())
            return {'idp': IdentityProvider(config['idp']), 'email': config.get('email')}

    return None


async def deploy_config_and_headers_from_namespace(namespace: Optional[str] = None, *, authorize_target: bool = True) -> Tuple[DeployConfig, Dict[str, str], str]:
    deploy_config = get_deploy_config()

    if namespace is not None:
        deploy_config = deploy_config.with_default_namespace(namespace)
    else:
        namespace = deploy_config.default_namespace()

    credentials = await hail_credentials(namespace=namespace, authorize_target=authorize_target)
    headers = await credentials.auth_headers()

    return (deploy_config, headers, namespace)


async def async_get_userinfo():
    deploy_config = get_deploy_config()
    credentials = await hail_credentials()
    userinfo_url = deploy_config.url('auth', '/api/v1alpha/userinfo')

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


def create_user(username: str, login_id: str, is_developer: bool, is_service_account: bool, namespace: Optional[str] = None):
    return async_to_blocking(async_create_user(username, login_id, is_developer, is_service_account, namespace=namespace))


async def async_create_user(username: str, login_id: str, is_developer: bool, is_service_account: bool, namespace: Optional[str] = None):
    deploy_config, headers, _ = await deploy_config_and_headers_from_namespace(namespace)

    body = {
        'login_id': login_id,
        'is_developer': is_developer,
        'is_service_account': is_service_account,
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
