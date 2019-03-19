import jwt


class JWTClient:
    __ALGORITHM = 'HS256'

    def __init__(self, secret_key):
        self.secret_key = secret_key

    def decode(self, token):
        if token is None:
            return None
        return jwt.decode(
            token, self.secret_key, algorithms=[JWTClient.__ALGORITHM])

    def encode(self, payload):
        return jwt.encode(
            payload, self.secret_key, algorithm=JWTClient.__ALGORITHM)


def get_domain(host):
    parts = host.split('.')
    return f"{parts[-2]}.{parts[-1]}"
