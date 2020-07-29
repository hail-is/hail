import os
import re
import argparse
from shlex import quote as shq

from hailtop.utils import sync_check_shell

from .flock import Flock


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


def copy(src, dst, user, io_host_path, cache_path, requester_pays_project):
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

    cache_src = f'{cache_path}/{user}{src[4:]}'
    escaped_src = re.escape(os.path.basename(src))
    dst = f'/host{io_host_path}{dst[3:]}'

    match = GCS_PATH_REGEX.fullmatch(src)
    if not match:
        raise NotImplementedError(f'source is not a valid GCS path, got source {src}')
    src_path = match.groupdict()['path']
    src_tail_dir = src_path.rstrip('/').split('/')[-1]
    rsync_dst = f'{dst.rstrip("/")}/{src_tail_dir}/'

    rsync_file_cmd = f'''
gsutil {requester_pays_project} -m -q rsync -x '(?!^{escaped_src}$)' {shq(os.path.dirname(src))} {shq(os.path.dirname(dst))}
'''
    rsync_dir_cmd = f'''
mkdir -p {shq(rsync_dst)} && gsutil {requester_pays_project} -m -q rsync -r -d {shq(src)} {shq(rsync_dst)} || \
rm -rf {shq(rsync_dst)} && gsutil {requester_pays_project} -m -q cp -r {shq(src)} {shq(dst)}
'''

    with Flock(cache_src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.makedirs(os.path.dirname(cache_src), exist_ok=True)
        check_call(f'cp -p -R --reflink {shq(cache_src)} {shq(dst)} || true')
        if src.endswith('/'):
            check_call(rsync_dir_cmd)
        else:
            check_call(f'''
gsutil -q stat {shq(src)}
if [ $? = 0 ]; then
  {rsync_file_cmd};
else
  {rsync_dir_cmd};
fi
''')
        check_call(f'rm -Rf {shq(cache_src)}')
        check_call(f'cp -p -R --reflink {shq(dst)} {shq(cache_src)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, required=True)
    parser.add_argument('--io-host-path', type=str, required=True)
    parser.add_argument('--cache-path', type=str, required=True)
    parser.add_argument('--requester-pays-project', type=str, required=False)
    parser.add_argument('-f', '--files', action='append', type=str, nargs=2, metavar=('src', 'dest'))
    args = parser.parse_args()

    for src, dest in args.files:
        copy(src, dest, args.user, args.io_host_path, args.cache_path, args.requester_pays_project)
