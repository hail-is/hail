import collections.abc
import os
import json
import logging

from ..location import get_location

log = logging.getLogger('gear')


class Tokens(collections.abc.MutableMapping):
    @staticmethod
    def get_tokens_file():
        location = get_location()
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

    def write(self, tokens_file=None):
        if not tokens_file:
            tokens_file = self.get_tokens_file()
        with open(tokens_file, 'w', mode=0o600) as f:
            f.write(json.dumps(self._tokens))


tokens = None


def get_tokens():
    global tokens

    if not tokens:
        tokens = Tokens()
    return tokens
