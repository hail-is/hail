import random
import string
import shlex
import collections


def get_sha(k):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))


def escape_string(s):
    return shlex.quote(s)


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes, tuple)):
            yield from flatten(el)
        else:
            yield el