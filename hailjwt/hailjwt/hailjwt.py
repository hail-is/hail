import jwt

class JWTClient:
    def __init__(self, secret_key):
        self.secret_key = secret_key

    def jwt_decode(self, token):
        if token is None:
            return None
        return jwt.decode(token, self.secret_key, algorithms=['HS256'])

    def jwt_encode(self, payload):
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

def get_domain(host):
    parts = host.split('.')
    p_len = len(parts)

    return f"{parts[p_len - 2]}.{parts[p_len - 1]}"
