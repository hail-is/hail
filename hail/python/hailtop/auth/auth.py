from typing import Optional, Dict, Tuple
import os
import aiohttp

from hailtop import httpx
from hailtop.aiocloud.common.credentials import CloudCredentials
from hailtop.aiocloud.common import Session
from hailtop.config import get_deploy_config, DeployConfig
from hailtop.utils import async_to_blocking, retry_transient_errors

from .tokens import Tokens, get_tokens


class HailStoredTokenCredentials(CloudCredentials):
    def __init__(self, tokens: Tokens, namespace: Optional[str], authorize_target: bool):
        self._tokens = tokens
        self._namespace = namespace
        self._authorize_target = authorize_target

    @staticmethod
    def from_file(credentials_file: str, *, namespace: Optional[str] = None, authorize_target: bool = True):
        return HailStoredTokenCredentials(get_tokens(credentials_file), namespace, authorize_target)

    @staticmethod
    def default_credentials(*, namespace: Optional[str] = None, authorize_target: bool = True):
        return HailStoredTokenCredentials(get_tokens(), namespace, authorize_target)

    async def auth_headers(self) -> Dict[str, str]:
        deploy_config = get_deploy_config()
        ns = self._namespace or deploy_config.default_namespace()
        return namespace_auth_headers(deploy_config, ns, self._tokens, authorize_target=self._authorize_target)

    async def close(self):
        pass


def hail_credentials(*, credentials_file: Optional[str] = None, namespace: Optional[str] = None, authorize_target: bool = True) -> CloudCredentials:
    if credentials_file is not None:
        return HailStoredTokenCredentials.from_file(
            credentials_file,
            namespace=namespace,
            authorize_target=authorize_target
        )
    return HailStoredTokenCredentials.default_credentials(
        namespace=namespace,
        authorize_target=authorize_target
    )


def namespace_auth_headers(deploy_config: DeployConfig,
                           ns: str,
                           tokens: Tokens,
                           authorize_target: bool = True,
                           ) -> Dict[str, str]:
    headers = {}
    if authorize_target:
        headers['Authorization'] = f'Bearer {tokens.namespace_token_or_error(ns)}'
    if deploy_config.location() == 'external' and ns != 'default':
        headers['X-Hail-Internal-Authorization'] = f'Bearer {tokens.namespace_token_or_error("default")}'
    return headers


def deploy_config_and_headers_from_namespace(namespace: Optional[str] = None, *, authorize_target: bool = True) -> Tuple[DeployConfig, Dict[str, str], str]:
    deploy_config = get_deploy_config()

    if namespace is not None:
        deploy_config = deploy_config.with_default_namespace(namespace)
    else:
        namespace = deploy_config.default_namespace()

    headers = namespace_auth_headers(deploy_config, namespace, get_tokens(), authorize_target=authorize_target)

    return (deploy_config, headers, namespace)


async def async_get_userinfo():
    deploy_config = get_deploy_config()
    credentials = hail_credentials()
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
    deploy_config, headers, namespace = deploy_config_and_headers_from_namespace(namespace, authorize_target=False)
    async with httpx.client_session(headers=headers) as session:
        data = await retry_transient_errors(
            session.post_return_json,
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
    deploy_config, headers, _ = deploy_config_and_headers_from_namespace(namespace)

    async with httpx.client_session(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers) as session:
        return await retry_transient_errors(
            session.get_return_json,
            deploy_config.url('auth', f'/api/v1alpha/users/{username}')
        )


def create_user(username: str, login_id: str, is_developer: bool, is_service_account: bool, namespace: Optional[str] = None):
    return async_to_blocking(async_create_user(username, login_id, is_developer, is_service_account, namespace=namespace))


async def async_create_user(username: str, login_id: str, is_developer: bool, is_service_account: bool, namespace: Optional[str] = None):
    deploy_config, headers, _ = deploy_config_and_headers_from_namespace(namespace)

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
    deploy_config, headers, _ = deploy_config_and_headers_from_namespace(namespace)
    async with httpx.client_session(
            timeout=aiohttp.ClientTimeout(total=300),
            headers=headers) as session:
        await retry_transient_errors(
            session.delete,
            deploy_config.url('auth', f'/api/v1alpha/users/{username}')
        )
