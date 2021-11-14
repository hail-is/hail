from typing import Optional, Dict
import os
import aiohttp
from hailtop.config import get_deploy_config, DeployConfig
from hailtop.utils import async_to_blocking, request_retry_transient_errors
from hailtop import httpx

from .tokens import get_tokens


async def async_get_userinfo(*,
                             deploy_config: Optional[DeployConfig] = None,
                             session_id: Optional[str] = None,
                             client_session: Optional[httpx.ClientSession] = None):
    if deploy_config is None:
        deploy_config = get_deploy_config()
    if session_id is None:
        headers = service_auth_headers(deploy_config, 'auth')
    else:
        headers = {'Authorization': f'Bearer {session_id}'}

    userinfo_url = deploy_config.url('auth', '/api/v1alpha/userinfo')

    async def request(session):
        try:
            resp = await request_retry_transient_errors(
                session, 'GET', userinfo_url, headers=headers)
            return await resp.json()
        except aiohttp.client_exceptions.ClientResponseError as err:
            if err.status == 401:
                return None
            raise

    if client_session is None:
        async with httpx.client_session() as session:
            return await request(session)
    return await request(client_session)


def get_userinfo(deploy_config=None, session_id=None, client_session=None):
    return async_to_blocking(async_get_userinfo(
        deploy_config=deploy_config,
        session_id=session_id,
        client_session=client_session))


def namespace_auth_headers(deploy_config: DeployConfig,
                           ns: str,
                           authorize_target: bool = True,
                           *,
                           token_file: Optional[str] = None
                           ) -> Dict[str, str]:
    headers = {}
    if authorize_target:
        headers['Authorization'] = f'Bearer {get_tokens(token_file).namespace_token_or_error(ns)}'
    if deploy_config.location() == 'external' and ns != 'default':
        headers['X-Hail-Internal-Authorization'] = f'Bearer {get_tokens(token_file).namespace_token_or_error("default")}'
    return headers


def service_auth_headers(deploy_config: DeployConfig,
                         service: str,
                         authorize_target: bool = True,
                         *,
                         token_file: Optional[str] = None
                         ) -> Dict[str, str]:
    ns = deploy_config.service_ns(service)
    return namespace_auth_headers(deploy_config, ns, authorize_target, token_file=token_file)


def copy_paste_login(copy_paste_token, namespace=None):
    return async_to_blocking(async_copy_paste_login(copy_paste_token, namespace))


async def async_copy_paste_login(copy_paste_token, namespace=None):
    deploy_config = get_deploy_config()

    if namespace is not None:
        deploy_config = deploy_config.with_default_namespace(namespace)
    else:
        namespace = deploy_config.default_namespace()

    headers = namespace_auth_headers(deploy_config, namespace, authorize_target=False)

    async with aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=5),
            headers=headers) as session:
        resp = await request_retry_transient_errors(
            session, 'POST', deploy_config.url('auth', '/api/v1alpha/copy-paste-login'),
            params={'copy_paste_token': copy_paste_token})
        resp = await resp.json()
    token = resp['token']
    username = resp['username']

    tokens = get_tokens()
    tokens[namespace] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    return namespace, username
