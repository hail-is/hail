import logging
from functools import wraps
import asyncio
import aiohttp
from aiohttp import web

from ..deploy_config import get_deploy_config
from .tokens import get_tokens

log = logging.getLogger('gear.auth')


async def async_get_userinfo():
    deploy_config = get_deploy_config()
    headers = auth_headers('auth')
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.get(
                deploy_config.url('auth', '/api/v1alpha/userinfo'), headers=headers) as resp:
            return await resp.json()


def get_userinfo():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_get_userinfo())

def _authenticated_users_only(rest):
    deploy_config = get_deploy_config()
    cookie_name = deploy_config.auth_session_cookie_name()
    def wrap(fun):
        @wraps(fun)
        async def wrapped(request, *args, **kwargs):
            headers = {}
            cookies = {}
            if rest:
                if 'Authorization' in request.headers:
                    headers['Authorization'] = request.headers['Authorization']
                else:
                    raise web.HTTPUnauthorized()
            else:
                # web
                if cookie_name in request.cookies:
                    cookies[cookie_name] = request.cookies[cookie_name]
                else:
                    raise web.HTTPUnauthorized()

            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    async with session.get(deploy_config.url('auth', '/api/v1alpha/userinfo'),
                                           headers=headers, cookies=cookies) as resp:
                        userdata = await resp.json()
            except Exception:
                log.exception('getting userinfo')
                raise web.HTTPUnauthorized()
            return await fun(request, userdata, *args, **kwargs)
        return wrapped
    return wrap


def _authenticated_developers_only(rest):
    def wrap(fun):
        @_authenticated_users_only(rest)
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            if ('developer' in userdata) and userdata['developer'] == 1:
                return await fun(request, *args, **kwargs)
            raise web.HTTPUnauthorized()
        return wrapped
    return wrap


rest_authenticated_users_only = _authenticated_users_only(True)
rest_authenticated_developers_only = _authenticated_developers_only(True)

web_authenticated_users_only = _authenticated_users_only(False)
web_authenticated_developers_only = _authenticated_developers_only(False)


def auth_headers(service):
    deploy_config = get_deploy_config()
    tokens = get_tokens()
    headers = {}
    if service:
        ns = deploy_config.service_ns(service)
        headers['Authorization'] = f'Bearer {tokens[ns]}'
    if deploy_config.location() == 'external' and ns != 'default':
        headers['X-Hail-Internal-Authorization'] = f'Bearer {tokens["default"]}'
    return headers
