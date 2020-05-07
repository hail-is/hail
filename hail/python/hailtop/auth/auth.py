import os
import aiohttp
from hailtop.config import get_deploy_config
from hailtop.utils import async_to_blocking, request_retry_transient_errors
from hailtop.tls import ssl_client_session

from .tokens import get_tokens


async def async_get_userinfo(deploy_config=None, headers=None):
    if deploy_config is None:
        deploy_config = get_deploy_config()
    if headers is None:
        headers = service_auth_headers(deploy_config, 'auth')
    userinfo_url = deploy_config.url('auth', '/api/v1alpha/userinfo')
    async with ssl_client_session(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
        resp = await request_retry_transient_errors(
            session, 'GET', userinfo_url, headers=headers)
        return await resp.json()


def get_userinfo(deploy_config=None):
    return async_to_blocking(async_get_userinfo(deploy_config))


def namespace_auth_headers(deploy_config, ns, authorize_target=True):
    tokens = get_tokens()
    headers = {}
    if authorize_target:
        headers['Authorization'] = f'Bearer {tokens.namespace_token_or_error(ns)}'
    if deploy_config.location() == 'external' and ns != 'default':
        headers['X-Hail-Internal-Authorization'] = f'Bearer {tokens.namespace_token_or_error("default")}'
    return headers


def service_auth_headers(deploy_config, service, authorize_target=True):
    ns = deploy_config.service_ns(service)
    return namespace_auth_headers(deploy_config, ns, authorize_target)


def copy_paste_login(copy_paste_token, namespace=None):
    return async_to_blocking(async_copy_paste_login(copy_paste_token, namespace))


async def async_copy_paste_login(copy_paste_token, namespace=None):
    deploy_config = get_deploy_config()
    if namespace is not None:
        auth_ns = namespace
        deploy_config = deploy_config.with_service('auth', auth_ns)
    else:
        auth_ns = deploy_config.service_ns('auth')
    headers = namespace_auth_headers(deploy_config, auth_ns, authorize_target=False)

    async with aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60),
            headers=headers) as session:
        resp = await request_retry_transient_errors(
            session, 'POST', deploy_config.url('auth', '/api/v1alpha/copy-paste-login'),
            params={'copy_paste_token': copy_paste_token})
        resp = await resp.json()
    token = resp['token']
    username = resp['username']

    tokens = get_tokens()
    tokens[auth_ns] = token
    dot_hail_dir = os.path.expanduser('~/.hail')
    if not os.path.exists(dot_hail_dir):
        os.mkdir(dot_hail_dir, mode=0o700)
    tokens.write()

    return auth_ns, username
