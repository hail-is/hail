import base64
from globals import v1


def decode_secret(b64str):
    return base64.b64decode(b64str).decode('utf-8')


def fetch_secrets(secret_name, namespace):
    res = v1.read_namespaced_secret(secret_name, namespace)
    return res.data


def get_secrets(secret_name, namespace):
    secrets = {}
    for key, val in fetch_secrets(secret_name, namespace).items():
        secrets[key] = decode_secret(val)

    return secrets


def get_secret(secret_name, namespace, key):
    data = fetch_secrets(secret_name, namespace)

    if key not in data:
        return None

    return decode_secret(data[key])
