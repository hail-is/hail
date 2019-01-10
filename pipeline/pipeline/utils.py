import random, string, shlex


def get_sha(k):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))


def escape_string(s):
    return shlex.quote(s)
