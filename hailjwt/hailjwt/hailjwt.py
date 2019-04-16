import jwt


class JWTClient:
    __ALGORITHM = 'HS256'

    @staticmethod
    def generate_key():
        import secrets
        return secrets.token_bytes(64)

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

    def __init__(self, secret_key=None):
        if secret_key is not None:
            JWTClient._verify_key_preqrequisites(secret_key)
        self.secret_key = secret_key

    def unsafe_decode(self, token):
        return jwt.decode(token, verify=False)
        
    def decode(self, token):
        assert self.secret_key is not None
        return jwt.decode(
            token, self.secret_key, algorithms=[JWTClient.__ALGORITHM])

    def encode(self, payload):
        assert self.secret_key is not None        
        return jwt.encode(
            payload, self.secret_key, algorithm=JWTClient.__ALGORITHM)


def get_domain(host):
    parts = host.split('.')
    return f"{parts[-2]}.{parts[-1]}"
