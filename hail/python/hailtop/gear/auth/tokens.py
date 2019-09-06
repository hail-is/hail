import collections.abc
import os
import json
import logging

from ..deploy_config import get_deploy_config

log = logging.getLogger('gear')


class Tokens(collections.abc.MutableMapping):
    @staticmethod
    def get_tokens_file():
        deploy_config = get_deploy_config()
        location = deploy_config.location()
        if location == 'external':
            return os.path.expanduser('~/.hail/tokens.json')
        return '/user-tokens/tokens.json'

    def __init__(self):
        tokens_file = self.get_tokens_file()
        if os.path.isfile(tokens_file):
            with open(tokens_file, 'r') as f:
                self._tokens = json.loads(f.read())
        else:
            log.info(f'tokens file not found: {tokens_file}')
            self._tokens = {}

    def __setitem__(self, key, value):
        self._tokens[key] = value

    def __getitem__(self, key):
        return self._tokens[key]

    def __delitem__(self, key):
        del self._tokens[key]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def write(self):
        # restrict permissions to user
        with os.fdopen(os.open(self.get_tokens_file(), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), 'w') as f:
            f.write(json.dumps(self._tokens))


tokens = None


def get_tokens():
    global tokens

    if not tokens:
        tokens = Tokens()
    return tokens
