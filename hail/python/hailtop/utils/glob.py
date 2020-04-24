import fnmatch

from .utils import flatten


wildcards = ('*', '?', '[', ']', '{', '}')


# need a custom escape because escaped characters are not
# treated properly with fnmatch
def escape(path):
    new_path = []
    n = len(path)
    i = 0
    while i < n:
        if i < n - 1 and path[i] == '\\' and path[i + 1] in wildcards:
            new_path.append('[')
            new_path.append(path[i + 1])
            new_path.append(']')
            i += 2
            continue
        if path[i] == '{' or  path[i] == '}':
            raise NotImplementedError
        new_path.append(path[i])
        i += 1
    return ''.join(new_path)


def contains_wildcard(c):
    i = 0
    n = len(c)
    while i < n:
        if i < n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            i += 2
            continue
        elif c[i] in wildcards:
            return True
        i += 1
    return False


def unescape_escaped_wildcards(c):
    new_c = []
    i = 0
    n = len(c)
    while i < n:
        if i < n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            new_c.append(c[i + 1])
            i += 2
            continue
        new_c.append(c[i])
        i += 1
    return ''.join(new_c)


def prefix_wout_wildcard(c):
    new_c = []
    i = 0
    n = len(c)
    while i < n:
        if i < n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            new_c.append(c[i + 1])
            i += 2
            continue
        elif c[i] in wildcards:
            return ''.join(new_c)
        new_c.append(c[i])
        i += 1
    return ''.join(new_c)


def list_files(fs, path):
    raise NotImplementedError


def match(string, pattern):
    return fnmatch.fnmatchcase(string, escape(pattern))


# recursive means match all downstream files with prefix before **
def glob(fs, pattern, recursive=False):
    if '**' in pattern and not recursive:
        raise NotImplementedError

    if recursive:
        raise NotImplementedError

    components = pattern.split('/')

    def _glob(prefix, i):
        if i == len(components):
            assert prefix is not None
            if fs._exists(prefix):
                return [prefix]
            return []

        c = components[i]
        prefix = prefix + '/' if prefix is not None else ''
        if contains_wildcard(c):
            return flatten([_glob(f'{prefix}{f}', i + 1)
                            for f in fs._listdir(prefix) if match(f, c)])

        return _glob(f'{prefix}{unescape_escaped_wildcards(c)}', i + 1)

    return _glob(None, 0)
