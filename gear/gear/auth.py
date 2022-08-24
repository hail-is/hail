import asyncio
import logging
import urllib.parse
from functools import wraps
from typing import Optional

import aiohttp
import aiohttp_session
from aiohttp import web

from hailtop import httpx
from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config

from .time_limited_max_size_cache import TimeLimitedMaxSizeCache

log = logging.getLogger('gear.auth')

deploy_config = get_deploy_config()

BEARER = 'Bearer '

TEN_SECONDS_IN_NANOSECONDS = int(1e10)

USERDATA_CACHE = None


def maybe_parse_bearer_header(value: str) -> Optional[str]:
    if value.startswith(BEARER):
        return value[len(BEARER) :]
    return None


def _init_userdata_cache(client_session: httpx.ClientSession):
    global USERDATA_CACHE

    async def load(session_id):
        return await _userdata_from_session_id(session_id, client_session)

    USERDATA_CACHE = TimeLimitedMaxSizeCache(load, TEN_SECONDS_IN_NANOSECONDS, 100, 'session_userdata_cache')


async def _userdata_from_session_id(session_id: str, client_session: httpx.ClientSession):
    try:
        return await async_get_userinfo(
            deploy_config=deploy_config, session_id=session_id, client_session=client_session
        )
    except asyncio.CancelledError:
        raise
    except aiohttp.ClientResponseError as e:
        log.exception('unknown exception getting userinfo')
        raise web.HTTPInternalServerError() from e
    except Exception as e:  # pylint: disable=broad-except
        log.exception('unknown exception getting userinfo')
        raise web.HTTPInternalServerError() from e


async def userdata_from_web_request(request):
    session = await aiohttp_session.get_session(request)
    if 'session_id' not in session:
        return None

    if USERDATA_CACHE is None:
        _init_userdata_cache(request.app['client_session'])
    return await USERDATA_CACHE.lookup(session['session_id'])


async def userdata_from_rest_request(request):
    if 'Authorization' not in request.headers:
        return None
    auth_header = request.headers['Authorization']
    session_id = maybe_parse_bearer_header(auth_header)
    if not session_id:
        return None

    if USERDATA_CACHE is None:
        _init_userdata_cache(request.app['client_session'])
    return await USERDATA_CACHE.lookup(session_id)


def rest_authenticated_users_only(fun):
    async def wrapped(request, *args, **kwargs):
        userdata = await userdata_from_rest_request(request)
        if not userdata:
            web_userdata = await userdata_from_web_request(request)
            if web_userdata:
                return web.HTTPUnauthorized(reason="provided web auth to REST endpoint")
            raise web.HTTPUnauthorized()
        return await fun(request, userdata, *args, **kwargs)

    return wrapped


def _web_unauthenticated(request, redirect):
    if not redirect:
        return web.HTTPUnauthorized()

    login_url = deploy_config.external_url('auth', '/login')

    # request.url is a yarl.URL
    request_url = request.url
    x_forwarded_host = request.headers.get('X-Forwarded-Host')
    if x_forwarded_host:
        request_url = request_url.with_host(x_forwarded_host)
    x_forwarded_proto = request.headers.get('X-Forwarded-Proto')
    if x_forwarded_proto:
        request_url = request_url.with_scheme(x_forwarded_proto)

    return web.HTTPFound(f'{login_url}?next={urllib.parse.quote(str(request_url))}')


def web_authenticated_users_only(redirect=True):
    def wrap(fun):
        @wraps(fun)
        async def wrapped(request, *args, **kwargs):
            userdata = await userdata_from_web_request(request)
            if not userdata:
                rest_userdata = await userdata_from_rest_request(request)
                if rest_userdata:
                    return web.HTTPUnauthorized(reason="provided REST auth to web endpoint")
                raise _web_unauthenticated(request, redirect)
            return await fun(request, userdata, *args, **kwargs)

        return wrapped

    return wrap


def web_maybe_authenticated_user(fun):
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        return await fun(request, await userdata_from_web_request(request), *args, **kwargs)

    return wrapped


def web_authenticated_developers_only(redirect=True):
    def wrap(fun):
        @web_authenticated_users_only(redirect)
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            if userdata['is_developer'] == 1:
                return await fun(request, userdata, *args, **kwargs)
            raise web.HTTPUnauthorized()

        return wrapped

    return wrap


def rest_authenticated_developers_only(fun):
    @rest_authenticated_users_only
    @wraps(fun)
    async def wrapped(request, userdata, *args, **kwargs):
        if userdata['is_developer'] == 1:
            return await fun(request, userdata, *args, **kwargs)
        raise web.HTTPUnauthorized()

    return wrapped
