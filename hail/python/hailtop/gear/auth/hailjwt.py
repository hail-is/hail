import secrets
import jwt


class JWTClient:
    ALGORITHM = 'HS256'
    SECRET_KEY_LEN = 64

    @staticmethod
    def generate_key():
        return secrets.token_bytes(JWTClient.SECRET_KEY_LEN)

    @staticmethod
    def unsafe_decode(token):
        return jwt.decode(token, verify=False)

    def __init__(self, _secret_key=None):
        if not _secret_key:
            with open('/session-secret-keys/jwt-secret-key', 'rb') as f:
                secret_key = f.read()
            assert isinstance(secret_key, bytes)
            assert len(secret_key) == self.SECRET_KEY_LEN
            self.secret_key = secret_key
        else:
            self.secret_key = _secret_key

    def decode(self, token):
        return jwt.decode(
            token, self.secret_key, algorithms=[JWTClient.ALGORITHM])

    def encode(self, payload):
        return (jwt
                .encode(payload, self.secret_key, algorithm=JWTClient.ALGORITHM)
                .decode('ascii'))


jwtclient = None


def get_jwtclient():
    global jwtclient

    if not jwtclient:
        jwtclient = JWTClient()
    return jwtclient
