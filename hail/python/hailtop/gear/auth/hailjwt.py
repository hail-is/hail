import abc
import os
import logging
from functools import wraps
from aiohttp import web
import jwt

log = logging.getLogger('gear.auth')

DEFAULT_NAMESPACE = os.environ['HAIL_DEFAULT_NAMESPACE']

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
        with open('/session-secret-keys/jwt-secret-key', 'rb') as f:
            jwtclient = JWTClient(f.read())
    return jwtclient


class Authentication(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_session_id(request):
        ...

    @staticmethod
    @abc.abstractmethod
    def unauthorized():
        ...

    @staticmethod
    def authenticated_users_only(fun):
        @wraps(fun)
        async def wrapped(request, *args, **kwargs):
            session_id = get_session_id(request)
            if session_id:
                try:
                    async with aiohttp.ClientSession() as session:
                        # FIXME parameterize default namespace for testing
                        async with session.get('http://auth.{DEFAULT_NAMESPACE}.svc.cluster.local/api/v1alpha/userinfo',
                                               headers={'Authentication': f'token {session_id}'}) as resp:
                            userdata = await resp.json()
                except Exception as e:
                    log.exception('getting userinfo')
                    raise unauthorized()
                return fun(request, userdata, *args, **kwargs)
            else:
                raise unauthorized()
        return wrapped

    @staticmethod
    def authenticated_developers_only(fun):
        @auth_users_only
        @wraps(fun)
        async def wrapped(request, userdata, *args, **kwargs):
            if ('developer' in userdata) and userdata['developer'] is 1:
                return await fun(request, *args, **kwargs)
            unauthorized()
        return wrapped


class RESTAuthentication(Authentication):
    def get_session_id(request):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None
        if not auth_header.startswith('Bearer '):
            return None
        encoded_token = auth_header[7:]
        payload = get_jwtclient().decode(encoded_token)
        return payload.get('sub')

    def rest_unauthorized():
        raise web.HTTPUnauthorized(headers={'WWW-Authenticate': 'Bearer'})


rest_get_session_id = RESTAuthentication.get_session_id
rest_authenticated_users_only = RESTAuthentication.authenticated_users_only
rest_authenticated_developers_only = RESTAuthentication.authenticated_developers_only


class WebAuthentication(Authentication):
    async def get_session_id(request):
        session = await aiohttp_session.get_session(request)
        if not session:
            return None
        return session.get('session_id')

    def web_unauthorized():
        raise web.HTTPUnauthorized()


web_authenticated_users_only = WebAuthentication.authenticated_users_only
web_authenticated_developers_only = WebAuthentication.authenticated_developers_only
