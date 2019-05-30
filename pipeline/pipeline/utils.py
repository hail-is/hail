import shlex


def escape_string(s):
    return shlex.quote(s)
