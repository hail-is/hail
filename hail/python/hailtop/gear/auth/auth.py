import logging
from functools import wraps
import asyncio
import aiohttp
from aiohttp import web

from ..location import get_location
from ..deploy_config import get_deploy_config
from .tokens import get_tokens

log = logging.getLogger('gear.auth')


async def async_get_userinfo():
    deploy_config = get_deploy_config()
    async with aiohttp.ClientSession(
            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
        set_credentials(session, 'auth')
        async with session.get(
                deploy_config.url('auth', '/api/v1alpha/userinfo')) as resp:
            return await resp.json()


def get_userinfo():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_get_userinfo())


def authenticated_users_only(fun):
    deploy_config = get_deploy_config()
    cookie_name = deploy_config.auth_session_cookie_name()

    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        headers = {}
        cookies = {}
        if 'Authorization' in request.headers:
            headers['Authorization'] = request.headers['Authorization']
        elif cookie_name in request.cookies:
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


def authenticated_developers_only(fun):
    @authenticated_users_only
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        if ('developer' in userdata) and userdata['developer'] == 1:
            return await fun(request, *args, **kwargs)
        raise web.HTTPUnauthorized()
    return wrapped


def set_credentials(session, service):
    location = get_location()
    deploy_config = get_deploy_config()
    tokens = get_tokens()
    if service:
        ns = deploy_config.service_ns(service)
        session.headers['Authorization'] = f'Bearer {tokens[ns]}'
    if location == 'external' and ns != 'default':
        session.headers['X-Hail-Internal-Authorization'] = tokens['default']
