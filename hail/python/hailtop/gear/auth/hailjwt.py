import os
import logging
from functools import wraps
from aiohttp import web
import jwt

log = logging.getLogger('gear.auth')


class JWTClient:
    __ALGORITHM = 'HS256'

    @staticmethod
    def generate_key():
        import secrets
        return secrets.token_bytes(64)

    @staticmethod
    def unsafe_decode(token):
        return jwt.decode(token, verify=False)

    @staticmethod
    def _verify_key_preqrequisites(secret_key):
        if len(secret_key) < 32:
            raise ValueError(
                f'found secret key with {len(secret_key)} bytes, but secret '
                f'key must have at least 32 bytes (i.e. 256 bits)')

    def __init__(self, secret_key):
        assert isinstance(secret_key, bytes)
        JWTClient._verify_key_preqrequisites(secret_key)
        self.secret_key = secret_key

    def decode(self, token):
        return jwt.decode(
            token, self.secret_key, algorithms=[JWTClient.__ALGORITHM])

    def encode(self, payload):
        return (jwt.encode(
            payload, self.secret_key, algorithm=JWTClient.__ALGORITHM)
                .decode('ascii'))


jwtclient = None


def get_jwtclient():
    global jwtclient
    
    if not jwtclient:
        with open('/jwt-secret-key/secret-key', 'rb') as f:
            jwtclient = JWTClient(f.read())
    return jwtclient


async def get_web_session_id(request):
    session = await aiohttp_session.get_session(request)
    if not session:
        return None
    return session.get('session_id')


async def get_rest_session_id(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    if not auth_header.startswith('Bearer '):
        return None
    encoded_token = auth_header[7:]
    payload = get_jwtclient().decode(encoded_token)
    return payload.get('session_id')


def authenticated_users_only(get_session_id):
    def wrap(fun):
        @wraps(fun)
        async def wrapped(request, *args, **kwargs):
            session_id = await get_session_id(request)
            if session_id:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get('https://auth.hail.is/api/v1alpha/session/{session_id}') as resp:
                            userdata = await resp.json()
                            return await fun(request, userdata, *args, **kwargs)
                except jwt.exceptions.InvalidTokenError as exc:
                    log.info(f'could not decode token: {exc}')
            raise web.HTTPUnauthorized(headers={'WWW-Authenticate': 'Bearer'})
        return wrapped
    return wrap


def authenticated_developers_only(get_session_id):
    def wrap(fun):
        @authenticated_users_only(get_session_id)
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            if ('developer' in userdata) and userdata['developer'] is 1:
                return await fun(request, *args, **kwargs)
            raise web.HTTPNotFound()
        return wrapped
    return wrap


rest_authenticated_users_only = authenticated_users_only(get_rest_session_id)
rest_authenticated_developers_only = authenticated_developers_only(get_rest_session_id)

web_authenticated_users_only = authenticated_users_only(get_web_session_id)
web_authenticated_developers_only = authenticated_developers_only(get_web_session_id)
