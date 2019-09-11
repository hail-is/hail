import logging
from functools import wraps
import urllib.parse
import aiohttp
from aiohttp import web
import aiohttp_session
from hailtop.config import get_deploy_config

log = logging.getLogger('gear.auth')


def _authenticated_users_only(rest, redirect):
    deploy_config = get_deploy_config()
    def wrap(fun):
        @wraps(fun)
        async def wrapped(request, *args, **kwargs):
            def unauth():
                if redirect:
                    login_url = deploy_config.external_url('auth', '/login')

                    # request.url is yarl.URL
                    request_url = request.url
                    x_forwarded_host = request.headers.get('X-Forwarded-Host')
                    if x_forwarded_host:
                        request_url = request_url.with_host(x_forwarded_host)
                    x_forwarded_proto = request.headers.get('X-Forwarded-Proto')
                    if x_forwarded_proto:
                        request_url = request_url.with_scheme(x_forwarded_proto)

                    raise web.HTTPFound(f'{login_url}?next={urllib.parse.quote(str(request_url))}')
                raise web.HTTPUnauthorized()

            if rest:
                if 'Authorization' not in request.headers:
                    unauth()
                auth_header = request.headers['Authorization']
                if not auth_header.startswith('Bearer '):
                    unauth()
                session_id = auth_header[7:]
            else:
                session = await aiohttp_session.get_session(request)
                if 'session_id' not in session:
                    unauth()
                session_id = session['session_id']

            headers = {'Authorization': f'Bearer {session_id}'}
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    async with session.get(deploy_config.url('auth', '/api/v1alpha/userinfo'),
                                           headers=headers) as resp:
                        userdata = await resp.json()
            except Exception:  # pylint: disable=broad-except
                log.exception('getting userinfo')
                unauth()
            return await fun(request, userdata, *args, **kwargs)
        return wrapped
    return wrap


def _authenticated_developers_only(rest, redirect):
    def wrap(fun):
        @_authenticated_users_only(rest, redirect)
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            if ('developer' in userdata) and userdata['developer'] == 1:
                return await fun(request, userdata, *args, **kwargs)
            raise web.HTTPUnauthorized()
        return wrapped
    return wrap


rest_authenticated_users_only = _authenticated_users_only(rest=True, redirect=False)
rest_authenticated_developers_only = _authenticated_developers_only(rest=True, redirect=False)


def web_authenticated_users_only(redirect=True):
    return _authenticated_users_only(False, redirect)


def web_authenticated_developers_only(redirect=True):
    return _authenticated_developers_only(False, redirect)


def web_maybe_authenticated_user(fun):
    deploy_config = get_deploy_config()
    @wraps(fun)
    async def wrapped(request, *args, **kwargs):
        userdata = None
        session = await aiohttp_session.get_session(request)
        if 'session_id' in session:
            session_id = session['session_id']
            headers = {'Authorization': f'Bearer {session_id}'}
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    async with session.get(deploy_config.url('auth', '/api/v1alpha/userinfo'),
                                           headers=headers) as resp:
                        userdata = await resp.json()
            except Exception:  # pylint: disable=broad-except
                log.exception('getting userinfo')
        return await fun(request, userdata, *args, **kwargs)
    return wrapped
