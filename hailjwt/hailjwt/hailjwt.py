import jwt


class JWTClient:
    __ALGORITHM = 'HS256'

    def _verify_key_preqrequisites(self, secret_key):
        assert isinstance(secret_key, str)
        n_bytes = len(secret_key.encode('utf-8'))
        if n_bytes < 256:
            raise ValueError(
                f'found secret key with {n_bytes} bytes, but secret key must '
                f'have at least 256 bytes')

    def __init__(self, secret_key):
        self._verify_key_preqrequisites(secret_key)
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
