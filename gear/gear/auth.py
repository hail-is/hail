import logging
from functools import wraps
import urllib.parse
import aiohttp
from aiohttp import web
import aiohttp_session
from hailtop.config import get_deploy_config
from hailtop.utils import request_retry_transient_errors
from hailtop.tls import in_cluster_ssl_client_session

log = logging.getLogger('gear.auth')

deploy_config = get_deploy_config()


async def _userdata_from_session_id(session_id):
    headers = {'Authorization': f'Bearer {session_id}'}
    try:
        async with in_cluster_ssl_client_session(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
            resp = await request_retry_transient_errors(
                session, 'GET', deploy_config.url('auth', '/api/v1alpha/userinfo'),
                headers=headers)
            assert resp.status == 200
            return await resp.json()
    except aiohttp.ClientResponseError as e:
        if e.status == 401:
            return None

        log.exception('unknown exception getting userinfo')
        raise web.HTTPInternalServerError()
    except Exception:  # pylint: disable=broad-except
        log.exception('unknown exception getting userinfo')
        raise web.HTTPInternalServerError()


async def userdata_from_web_request(request):
    session = await aiohttp_session.get_session(request)
    if 'session_id' not in session:
        return None
    return await _userdata_from_session_id(session['session_id'])


async def userdata_from_rest_request(request):
    if 'Authorization' not in request.headers:
        return None
    auth_header = request.headers['Authorization']
    if not auth_header.startswith('Bearer '):
        return None
    return await _userdata_from_session_id(auth_header[7:])


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


def _web_unauthorized(request, redirect):
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
                raise _web_unauthorized(request, redirect)
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
            raise _web_unauthorized(request, redirect)
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
