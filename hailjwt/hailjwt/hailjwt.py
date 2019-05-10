import os
import jwt


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
        if isinstance(secret_key, str):
            key_bytes = secret_key.encode('utf-8')
        else:
            assert isinstance(secret_key, bytes), type(secret_key)
            key_bytes = secret_key
        if len(key_bytes) < 32:
            raise ValueError(
                f'found secret key with {len(key_bytes)} bytes, but secret key '
                f'must have at least 32 bytes (i.e. 256 bits)')

    def __init__(self, secret_key):
        JWTClient._verify_key_preqrequisites(secret_key)
        self.secret_key = secret_key

    def decode(self, token):
        return jwt.decode(
            token, self.secret_key, algorithms=[JWTClient.__ALGORITHM])

    def encode(self, payload):
        return jwt.encode(
            payload, self.secret_key, algorithm=JWTClient.__ALGORITHM)


def get_domain(host):
    parts = host.split('.')
    return f"{parts[-2]}.{parts[-1]}"

jwtclient = None


def authenticated_users_only(fun):
    def wrapped(request, *args, **kwargs):
        global jwtclient

        encoded_token = request.cookies.get('user')
        if encoded_token is not None:
            try:
                if not jwtclient:
                    with open(os.environ.get('HAIL_JWT_SECRET_KEY_FILE', '/jwt-secret/secret-key')) as f:
                        jwtclient = JWTClient(f.read())

                userdata = jwtclient.decode(encoded_token)
                if 'userdata' in fun.__code__.co_varnames:
                    return fun(request, *args, userdata=userdata, **kwargs)
                return fun(request, *args, **kwargs)
            except jwt.exceptions.InvalidTokenError as exc:
                log.info(f'could not decode token: {exc}')
        raise web.HTTPUnauthorized(headers={'WWW-Authenticate': 'Bearer'})
    wrapped.__name__ = fun.__name__
    return wrapped
