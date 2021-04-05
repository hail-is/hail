import base64
import collections.abc
import os
import sys
import json
import logging
from hailtop.config import get_deploy_config
from hailtop.utils import first_extant_file

log = logging.getLogger('gear')


def session_id_encode_to_str(session_id_bytes: bytes) -> str:
    return base64.urlsafe_b64encode(session_id_bytes).decode('ascii')


def session_id_decode_from_str(session_id_str: str) -> bytes:
    return base64.urlsafe_b64decode(session_id_str.encode('ascii'))


class Tokens(collections.abc.MutableMapping):
    @staticmethod
    def get_tokens_file():
        deploy_config = get_deploy_config()
        location = deploy_config.location()
        if location == 'external':
            return os.path.expanduser('~/.hail/tokens.json')
        return first_extant_file(
            os.environ.get('HAIL_TOKENS_FILE'),
            '/user-tokens/tokens.json'
        )

    @staticmethod
    def default_tokens():
        tokens_file = Tokens.get_tokens_file()
        if os.path.isfile(tokens_file):
            with open(tokens_file, 'r') as f:
                log.info(f'tokens loaded from {tokens_file}')
                return Tokens(json.load(f))
        else:
            log.info(f'tokens file not found: {tokens_file}')
            return Tokens({})

    @staticmethod
    def from_file(tokens_file):
        with open(tokens_file, 'r') as f:
            log.info(f'tokens loaded from {tokens_file}')
            return Tokens(json.load(f))

    def __init__(self, tokens):
        self._tokens = tokens

    def __setitem__(self, key, value):
        self._tokens[key] = value

    def __getitem__(self, key):
        return self._tokens[key]

    def namespace_token_or_error(self, ns):
        if ns in self._tokens:
            return self._tokens[ns]

        deploy_config = get_deploy_config()
        auth_ns = deploy_config.service_ns('auth')
        ns_arg = '' if ns == auth_ns else f'-n {ns}'
        sys.stderr.write(f'''\
You are not authenticated.  Please log in with:

  $ hailctl auth login {ns_arg}

to obtain new credentials.
''')
        sys.exit(1)

    def __delitem__(self, key):
        del self._tokens[key]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def write(self):
        # restrict permissions to user
        with os.fdopen(os.open(self.get_tokens_file(), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), 'w') as f:
            json.dump(self._tokens, f)


tokens = {}
default_tokens = None


def get_tokens(file=None):
    global tokens
    global default_tokens

    if file is None:
        if default_tokens is None:
            default_tokens = Tokens.default_tokens()
        return default_tokens
    if file not in tokens:
        tokens[file] = Tokens.from_file(file)
    return tokens[file]
