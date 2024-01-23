import abc
import asyncio
import logging
import os
import urllib.parse
from functools import wraps
from typing import Awaitable, Callable, Optional, Tuple, TypedDict, cast

import aiohttp
import aiohttp_session
from aiohttp import web

from hailtop import httpx
from hailtop.config import get_deploy_config
from hailtop.utils import retry_transient_errors

from .time_limited_max_size_cache import TimeLimitedMaxSizeCache

log = logging.getLogger('gear.auth')

deploy_config = get_deploy_config()

BEARER = 'Bearer '

TEN_SECONDS_IN_NANOSECONDS = int(1e10)


class CommonAiohttpAppKeys:
    CLIENT_SESSION = web.AppKey('client_session', httpx.ClientSession)


class UserData(TypedDict):
    id: int
    state: str
    username: str
    login_id: str
    namespace_name: str
    is_developer: bool
    is_service_account: bool
    hail_credentials_secret_name: str
    tokens_secret_name: str


def maybe_parse_bearer_header(value: str) -> Optional[str]:
    if value.startswith(BEARER):
        return value[len(BEARER) :]
    return None


AIOHTTPHandler = Callable[[web.Request], Awaitable[web.StreamResponse]]
AuthenticatedAIOHTTPHandler = Callable[[web.Request, UserData], Awaitable[web.StreamResponse]]
MaybeAuthenticatedAIOHTTPHandler = Callable[[web.Request, Optional[UserData]], Awaitable[web.StreamResponse]]


class Authenticator(abc.ABC):
    def authenticated_users_only(
        self, redirect: Optional[bool] = None
    ) -> Callable[[AuthenticatedAIOHTTPHandler], AIOHTTPHandler]:
        def wrap(fun: AuthenticatedAIOHTTPHandler):
            @wraps(fun)
            async def wrapped(request: web.Request) -> web.StreamResponse:
                userdata = await self._fetch_userdata(request)
                if not userdata:
                    # Only web routes should redirect by default
                    if redirect or (redirect is None and '/api/' not in request.path):
                        raise login_redirect(request)
                    raise web.HTTPUnauthorized()
                return await fun(request, userdata)

            return wrapped

        return wrap

    def maybe_authenticated_user(self, fun: MaybeAuthenticatedAIOHTTPHandler) -> AIOHTTPHandler:
        @wraps(fun)
        async def wrapped(request: web.Request) -> web.StreamResponse:
            return await fun(request, await self._fetch_userdata(request))

        return wrapped

    def authenticated_developers_only(self, redirect=True):
        def wrap(fun: AuthenticatedAIOHTTPHandler):
            @self.authenticated_users_only(redirect)
            @wraps(fun)
            async def wrapped(request: web.Request, userdata: UserData, *args, **kwargs):
                if userdata['is_developer'] == 1:
                    return await fun(request, userdata, *args, **kwargs)
                raise web.HTTPUnauthorized()

            return wrapped

        return wrap

    @abc.abstractmethod
    async def _fetch_userdata(self, request: web.Request) -> Optional[UserData]:
        raise NotImplementedError


class AuthServiceAuthenticator(Authenticator):
    def __init__(self):
        self._userdata_cache = TimeLimitedMaxSizeCache(
            self._fetch_userdata_from_auth_service, TEN_SECONDS_IN_NANOSECONDS, 100, 'session_userdata_cache'
        )

    async def _fetch_userdata(self, request: web.Request) -> Optional[UserData]:
        session_id = await get_session_id(request)
        if session_id is None:
            return None

        return await self._userdata_cache.lookup((session_id, request.app[CommonAiohttpAppKeys.CLIENT_SESSION]))

    @staticmethod
    async def _fetch_userdata_from_auth_service(session_id_and_session: Tuple[str, httpx.ClientSession]):
        session_id, client_session = session_id_and_session
        try:
            return await impersonate_user_and_get_info(session_id=session_id, client_session=client_session)
        except asyncio.CancelledError:
            raise
        except aiohttp.ClientResponseError as e:
            log.exception('unknown exception getting userinfo')
            raise web.HTTPInternalServerError() from e
        except Exception as e:  # pylint: disable=broad-except
            log.exception('unknown exception getting userinfo')
            raise web.HTTPInternalServerError() from e


class TrustedSingleTenantAuthenticator(Authenticator):
    async def _fetch_userdata(self, _: web.Request) -> Optional[UserData]:
        return cast(
            UserData,
            {
                'is_developer': True,
                'username': 'user',
                'hail_credentials_secret_name': 'dummy',
                'tokens_secret_name': 'dummy',
            },
        )


async def impersonate_user_and_get_info(session_id: str, client_session: httpx.ClientSession):
    userinfo_url = deploy_config.url('auth', '/api/v1alpha/userinfo')
    return await impersonate_user(session_id, client_session, userinfo_url)


async def impersonate_user(session_id: str, client_session: httpx.ClientSession, url: str):
    headers = {'Authorization': f'Bearer {session_id}'}
    try:
        return await retry_transient_errors(client_session.get_read_json, url, headers=headers)
    except aiohttp.ClientResponseError as err:
        if err.status == 401:
            return None
        raise


def get_authenticator() -> Authenticator:
    if os.environ.get('HAIL_TERRA'):
        return TrustedSingleTenantAuthenticator()
    return AuthServiceAuthenticator()


async def get_session_id(request: web.Request) -> Optional[str]:
    # Favor browser cookie to Bearer token auth
    session = await aiohttp_session.get_session(request)
    if 'session_id' in session:
        return session['session_id']

    if 'Authorization' not in request.headers:
        return None

    return maybe_parse_bearer_header(request.headers['Authorization'])


def login_redirect(request) -> web.HTTPFound:
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
