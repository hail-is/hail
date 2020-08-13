import re
import argparse
from shlex import quote as shq

from hailtop.utils import sync_check_shell


GCS_PATH_REGEX = re.compile('gs://(?P<bucket>[^/]+)/(?P<path>.+)')


wildcards = ('*', '?', '[', ']', '{', '}')


def contains_wildcard(c):
    i = 0
    n = len(c)
    while i < n:
        if i < n - 1 and c[i] == '\\' and c[i + 1] in wildcards:
            i += 2
            continue
        if c[i] in wildcards:
            return True
        i += 1
    return False


def check_call(script):
    return sync_check_shell(script, echo=True)


def copy(src, dst, requester_pays_project):
    if contains_wildcard(src):
        raise NotImplementedError(f'glob wildcards are not allowed in file paths, got source {src}')

    if contains_wildcard(dst):
        raise NotImplementedError(f'glob wildcards are not allowed in file paths, got destination {dst}')

    if not src.startswith('gs://'):
        raise NotImplementedError(f'cannot copy from sources that are not GCS paths, got source {src}')

    if not dst.startswith('/io/'):
        raise NotImplementedError(f'cannot copy to destinations that are not in /io/, got destination {dst}')

    if requester_pays_project:
        requester_pays_project = f'-u {requester_pays_project}'
    else:
        requester_pays_project = ''

    check_call(f'gsutil {requester_pays_project} -m cp -R {shq(src)} {shq(dst)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, required=True)
    parser.add_argument('--io-host-path', type=str, required=True)
    parser.add_argument('--cache-path', type=str, required=True)
    parser.add_argument('--requester-pays-project', type=str, required=False)
    parser.add_argument('-f', '--files', action='append', type=str, nargs=2, metavar=('src', 'dest'))
    args = parser.parse_args()

    for src, dest in args.files:
        copy(src, dest, args.requester_pays_project)
