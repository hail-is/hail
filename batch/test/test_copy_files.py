import os
import unittest
import tempfile
import subprocess as sp
import uuid
import copy
import random
import string
import json
from functools import wraps
import google.oauth2.service_account
from hailtop.auth import get_userinfo
from hailtop.utils.os import _glob
from hailtop.utils import async_to_blocking

from batch.google_storage import GCS
from batch.worker.copy_files import copy_files, OperationNotPermittedError

key_file = '/gsa-key/key.json'
project = os.environ['PROJECT']
user = get_userinfo()
tmp_bucket = f'gs://{user["bucket_name"]}/test_copy_files'

# key_file = '/Users/jigold/.hail/key.json'
# project = 'hail-vdc'
# tmp_bucket = f'gs://hail-jigold-59hi5/test_copy_files'

credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_file)
gcs_client = GCS(None, project=project, credentials=credentials)

rerun_gsutil = False
batch_dry_run = True

cd = os.path.dirname(os.path.abspath(__file__))
prerun_gsutil_result_file = f'{cd}/resources/prerun_gsutil_results.json'

if os.path.isfile(prerun_gsutil_result_file):
    with open(prerun_gsutil_result_file, 'r') as f:
        gsutil_results = json.loads(f.read())
else:
    with open(prerun_gsutil_result_file, 'w') as f:
        f.write(json.dumps({}))
    gsutil_results = {}


def tearDownModule():
    with open(prerun_gsutil_result_file, 'w') as f:
        f.write(json.dumps(gsutil_results))


class RemoteTemporaryDirectory:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        token = uuid.uuid4().hex[:6]
        self.name = tmp_bucket + f'/{token}'

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.dry_run:
            remove_remote_dir(self.name)


def upload_files(src, dest):
    sp.check_output(['gsutil', '-m', '-q', 'cp', '-r', src, dest])


def move_files(src, dest):
    sp.check_output(['gsutil', '-m', '-q', 'mv', '-r', src, dest])


def remove_remote_dir(path):
    path = path.rstrip('/') + '/'
    gcs_client._delete_gs_files(path)


def touch_file(path, data=None):
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    with open(path, 'w') as file:
        if data:
            file.write(data)


def strip_dest_path(files, root):
    return {f.replace(root, '') for f in files}


def glob_local_files(dir):
    try:
        files = [path for path, _ in _glob(dir, recursive=True)]
    except:
        files = []
    return strip_dest_path(files, dir)


def glob_remote_files(dir):
    try:
        blobs = gcs_client._glob_gs_files(dir, recursive=True)
    except:
        blobs = []
    files = {'gs://' + blob.bucket.name + '/' + blob.name for blob in blobs}
    return strip_dest_path(files, dir)


def cp_batch(src, dest, parallelism=1, min_partition_size='1Gi',
             max_upload_partitions=32, max_download_partitions=32,
             dry_run=False):
    files = [(src, dest)]
    coro = copy_files(files, key_file, project,
                      parallelism=parallelism,
                      min_partition_size=min_partition_size,
                      max_upload_partitions=max_upload_partitions,
                      max_download_partitions=max_download_partitions,
                      dry_run=dry_run)
    try:
        files = async_to_blocking(coro)[0]  # only one src, dest pair
        files = {dest for _, dest, _ in files}
        return (files, None)
    except Exception as err:
        return ({}, err)


def cp_gsutil(src, dest):
    cmd = ['gsutil', '-m', '-q', 'cp', '-r', src, dest]
    result = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
    return str(result.stdout), result.returncode, ' '.join(cmd)


def _get_batch_results(src, dest, local_dir, remote_dir, versions, dry_run=True):
    result = {}

    if 'rr' in versions:
        with RemoteTemporaryDirectory(dry_run=dry_run) as dest_dir:
            rr_files, rr_err = cp_batch(f'{remote_dir}{src}', dest_dir + dest, dry_run=dry_run)
            if not dry_run:
                rr_files = glob_remote_files(dest_dir)
            else:
                rr_files = strip_dest_path(rr_files, dest_dir)
            result['rr'] = {'files': rr_files, 'err': rr_err}

    if 'rl' in versions:
        with tempfile.TemporaryDirectory() as dest_dir:
            rl_files, rl_err = cp_batch(f'{remote_dir}{src}', dest_dir + dest, dry_run=dry_run)
            if not dry_run:
                rl_files = glob_local_files(dest_dir)
            else:
                rl_files = strip_dest_path(rl_files, dest_dir)
            result['rl'] = {'files': rl_files, 'err': rl_err}

    if 'lr' in versions:
        with RemoteTemporaryDirectory(dry_run=dry_run) as dest_dir:
            lr_files, lr_err = cp_batch(f'{local_dir}{src}', dest_dir + dest, dry_run=dry_run)
            if not dry_run:
                lr_files = glob_remote_files(dest_dir)
            else:
                lr_files = strip_dest_path(lr_files, dest_dir)
        result['lr'] = {'files': lr_files, 'err': lr_err}

    if 'll' in versions:
        with tempfile.TemporaryDirectory() as dest_dir:
            ll_files, ll_err = cp_batch(f'{local_dir}{src}', dest_dir + dest, dry_run=dry_run)
            if not dry_run:
                ll_files = glob_local_files(dest_dir)
            else:
                ll_files = strip_dest_path(ll_files, dest_dir)
        result['ll'] = {'files': ll_files, 'err': ll_err}

    return result


def _get_gsutil_results(test_name, src, dest, local_dir, remote_dir, versions, use_cache=True):
    result = {}

    if use_cache and not rerun_gsutil:
        test_result = gsutil_results.get(test_name)
        if test_result and test_result['src'] == src and test_result['dest'] == dest:
            tr = test_result['result']
            for version in tr:
                result[version] = copy.deepcopy(tr[version])
                result[version]['files'] = set(result[version]['files'])

    if 'rr' in versions and 'rr' not in result:
        with RemoteTemporaryDirectory() as dest_dir:
            rr_output, rr_rc, rr_cmd = cp_gsutil(f'{remote_dir}{src}', dest_dir + dest)
            rr_files = glob_remote_files(dest_dir)
            result['rr'] = {'files': rr_files, 'output': rr_output, 'rc': rr_rc, 'cmd': rr_cmd}

    if 'rl' in versions and 'rl' not in result:
        with tempfile.TemporaryDirectory() as dest_dir:
            rl_output, rl_rc, rl_cmd = cp_gsutil(f'{remote_dir}{src}', dest_dir + dest)
            rl_files = glob_local_files(dest_dir)
            result['rl'] = {'files': rl_files, 'output': rl_output, 'rc': rl_rc, 'cmd': rl_cmd}

    if 'lr' in versions and 'lr' not in result:
        with RemoteTemporaryDirectory() as dest_dir:
            lr_output, lr_rc, lr_cmd = cp_gsutil(f'{local_dir}{src}', dest_dir + dest)
            lr_files = glob_remote_files(dest_dir)
            result['lr'] = {'files': lr_files, 'output': lr_output, 'rc': lr_rc, 'cmd': lr_cmd}

    if 'll' in versions and 'll' not in result:
        with tempfile.TemporaryDirectory() as dest_dir:
            ll_output, ll_rc, ll_cmd = cp_gsutil(f'{local_dir}{src}', dest_dir + dest)
            ll_files = glob_local_files(dest_dir)
            result['ll'] = {'files': ll_files, 'output': ll_output, 'rc': ll_rc, 'cmd': ll_cmd}

    result2 = copy.deepcopy(result)
    for version in result2:
        result2[version]['files'] = list(result2[version]['files'])
    gsutil_results[test_name] = {
        'src': src,
        'dest': dest,
        'result': result2
    }

    return result


def get_results(src, dest, versions=('rr', 'rl', 'lr', 'll'), use_cache=True, dry_run=batch_dry_run):
    def wrap(fun):
        @wraps(fun)
        def wrapped(self, *args, **kwargs):
            test_name = fun.__qualname__
            g_result = _get_gsutil_results(test_name, src, dest, self.local_dir.name, self.remote_dir, versions, use_cache)
            b_result = _get_batch_results(src, dest, self.local_dir.name, self.remote_dir, versions, dry_run=dry_run)

            same = True
            for v in versions:
                g = g_result[v]
                b = b_result[v]
                same &= (g['files'] == b['files'])
                same &= ((g['rc'] == 0 and b['err'] is None) or
                         (g['rc'] != 0 and b['err'] is not None))

            return fun(self, b_result, g_result, same, *args, **kwargs)
        return wrapped
    return wrap


class TestEmptyDirectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        cls.remote_dir = None
        # FIXME: figure out how to get an empty directory in remote

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()

    @get_results('/', '/', ('ll', 'lr'))
    def test_download_directory(self, batch_results, gsutil_results, same):
        assert same
        for v in batch_results:
            assert isinstance(batch_results[v]['err'], FileNotFoundError)

    @get_results('/*', '/', ('ll', 'lr'))
    def test_download_asterisk(self, batch_results, gsutil_results, same):
        assert same
        for v in batch_results:
            assert isinstance(batch_results[v]['err'], FileNotFoundError)


class TestSingleFileTopLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/a')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/a', '/')
    def test_download_file_by_name(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/b', '/')
    def test_download_file_not_exists(self, batch_results, gsutil_results, same):
        assert same
        for v in batch_results:
            assert isinstance(batch_results[v]['err'], FileNotFoundError)

    @get_results('/data/a/', '/')
    def test_download_file_by_name_with_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/', '/')
    def test_download_directory(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data', '/')
    def test_download_directory_without_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*', '/')
    def test_download_single_wildcard(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*/*', '/')
    def test_download_multiple_wildcards(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/', '/', use_cache=False)
    def test_download_top_level_directory(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/a', '/b')
    def test_download_file_by_name_with_rename(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*', '/b')
    def test_download_file_by_wildcard_with_rename(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/a', '/foo/b')
    def test_download_file_by_name_to_nonexistent_subdir(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*', '/foo/b')
    def test_download_file_by_wildcard_to_nonexistent_subdir(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/**', '/', ('ll',))
    def test_double_asterisk(self, batch_results, gsutil_results, same):
        assert batch_results['ll']['files'] == set()
        err = batch_results['ll']['err']
        assert isinstance(err, NotImplementedError)
        assert '** not supported' in err.args[0]


class TestFileNestedInMultipleSubdirs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/a/b/c')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/a/b/c', '/')
    def test_download_file_by_name(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/a/b/c', '/foo')
    def test_download_file_by_name_with_rename(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/', '/')
    def test_download_directory_recursively(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*/b', '/')
    def test_download_wildcard_subdir_without_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*/b/', '/')
    def test_download_wildcard_subdir_with_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*/*/', '/')
    def test_download_double_wildcards(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*/*/*', '/')
    def test_download_double_wildcards_plus_file_wildcard(self, batch_results, gsutil_results, same):
        assert same


class TestDownloadMultipleFilesAtTopLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/a')
        touch_file(cls.local_dir.name + '/data/b')
        touch_file(cls.local_dir.name + '/data/c')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/*', '/')
    def test_download_file_asterisk(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/*/', '/')
    def test_download_file_asterisk_with_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/[ab]', '/')
    def test_download_file_match_brackets(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/?', '/')
    def test_download_file_question_mark(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/??', '/')
    def test_download_file_double_question_marks(self, batch_results, gsutil_results, same):
        assert same
        for v in batch_results:
            assert isinstance(batch_results[v]['err'], FileNotFoundError)

    @get_results('/data/[ab]', '/b')
    def test_download_multiple_files_to_single_file(self, batch_results, gsutil_results, same):
        assert same
        for v in batch_results:
            assert isinstance(batch_results[v]['err'], NotADirectoryError)

    @get_results('/data/a', '/b/')
    def test_download_file_invalid_dest_path_with_slash(self, batch_results, gsutil_results, same):
        assert same

        # assert 'skipping destination file ending with slash' in batch_results['rl']['err'].args(0)
        # assert 'skipping destination file ending with slash' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        #
        # # it's unclear why gsutil doesn't just create the directory like it does if the destination is remote
        # # it's also unclear why you don't get the same error as for the remote->local case
        # assert 'IsADirectoryError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')

    @get_results('/data/*', '/b/')
    def test_download_file_invalid_dest_dir_with_wildcard(self, batch_results, gsutil_results, same):
        assert same

        # assert 'destination must name a directory when matching multiple files' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        # assert 'NotADirectoryError' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        #
        # assert 'destination must name a directory when matching multiple files' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')
        # assert 'NotADirectoryError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')


class TestDownloadFileWithEscapedWildcards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/foo/bar/dog/a')
        touch_file(cls.local_dir.name + '/data/foo/baz/dog/h*llo')
        touch_file(cls.local_dir.name + '/data/foo/b?r/dog/b')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/foo/', '/')
    def test_download_all_files_recursively(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/foo/b\\?r/dog/', '/')
    def test_download_directory_with_escaped_question_mark(self, batch_results, gsutil_results, same):
        # gsutil does not have a mechanism for copying a path with an escaped wildcard
        # CommandException: No URLs matched: /var/folders/f_/ystbcjb13z78n85cyz6_jpl9sbv79d/T/tmpz9l9v9tf/data/foo/b\?r/dog/ CommandException: 1 file/object could not be transferred.
        expected = {'/dog/b'}
        for v in batch_results:
            assert batch_results[v]['files'] == expected
            assert batch_results[v]['err'] is None

    @get_results('/data/foo/b?r/dog/', '/')
    def test_download_directory_with_nonescaped_question_mark(self, batch_results, gsutil_results, same):
        # gsutil refuses to copy a path with a wildcard in it
        # Cloud folder gs://hail-jigold-59hi5/testing-suite/9f347b/data/foo/b\?r/ contains a wildcard; gsutil does not currently support objects with wildcards in their name.
        expected = {'/dog/a', '/dog/b'}
        for v in batch_results:
            assert batch_results[v]['files'] == expected
            assert batch_results[v]['err'] is None


class TestDownloadFileWithSpaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/foo/bar/dog/file with spaces.txt')
        touch_file(cls.local_dir.name + '/data/f o o/hello')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/', '/')
    def test_download_directory(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/foo/bar/dog/file with spaces.txt', '/')
    def test_download_file_with_spaces(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/f o o/hello', '/')
    def test_directory_with_spaces(self, batch_results, gsutil_results, same):
        assert same


class TestDownloadComplicatedDirectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/foo/a/data1')
        touch_file(cls.local_dir.name + '/data/bar/a')
        touch_file(cls.local_dir.name + '/data/baz')
        touch_file(cls.local_dir.name + '/data/dog/dog/dog')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/', '/')
    def test_download_all_files(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data', '/')
    def test_download_all_files_without_slash(self, batch_results, gsutil_results, same):
        assert same


class TestNonEmptyFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        cls.data = ''.join([random.choice(string.ascii_letters) for _ in range(16 * 1024)])

        with open(f'{cls.local_dir.name}/data', 'w') as f:
            f.write(cls.data)

        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    def test_download_multiple_partitions(self):
        with tempfile.TemporaryDirectory() as dest_dir:
            err = cp_batch(f'{self.remote_dir}/data', f'{dest_dir}/data', parallelism=4, min_partition_size='4Ki')
            with open(f'{dest_dir}/data', 'r') as f:
                assert f.read() == self.data, err

    def test_upload_multiple_partitions(self):
        with RemoteTemporaryDirectory() as remote_dest_dir:
            with tempfile.TemporaryDirectory() as local_dest_dir:
                err = cp_batch(f'{self.local_dir.name}/data', f'{remote_dest_dir}/data', parallelism=4, min_partition_size='4Ki')
                cp_batch(f'{remote_dest_dir}/data', f'{local_dest_dir}/data')
                with open(f'{local_dest_dir}/data', 'r') as f:
                    assert f.read() == self.data, err


class TestDownloadFileDirectoryWithSameName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        touch_file(cls.local_dir.name + '/data/foo/a')
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(cls.local_dir.name, cls.remote_dir)
        upload_files(cls.remote_dir + '/data/foo/a', cls.remote_dir + '/data/foo/a/b')

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/foo/a', '/')
    def test_download_file_by_name(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/foo/a/b', '/')
    def test_download_file_by_name_in_subdir(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/foo/a/', '/', ('rr', 'rl'))
    def test_download_directory_with_same_name_as_file(self, batch_results, gsutil_results, same):
        assert batch_results['rr']['err'] is None

        err = batch_results['rl']['err']
        assert isinstance(err, NotADirectoryError) or isinstance(err, FileExistsError)

    @get_results('/data/*/a', '/', ('rr', 'rl'))
    def test_download_file_with_wildcard(self, batch_results, gsutil_results, same):
        assert batch_results['rr']['err'] is None

        err = batch_results['rl']['err']
        assert isinstance(err, OperationNotPermittedError)


class TestEmptyFileSlashWithSameNameAsDirectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_dir = tempfile.TemporaryDirectory()
        token = uuid.uuid4().hex[:6]
        cls.remote_dir = f'{tmp_bucket}/{token}'
        gcs_client._write_gs_file_from_string(f'{cls.remote_dir}/data/bar/', 'bar')
        gcs_client._write_gs_file_from_string(f'{cls.remote_dir}/data/foo/', '')
        gcs_client._write_gs_file_from_string(f'{cls.remote_dir}/data/foo/a', 'a')
        gcs_client._write_gs_file_from_string(f'{cls.remote_dir}/data/foo/b', 'b')

    @classmethod
    def tearDownClass(cls):
        cls.local_dir.cleanup()
        remove_remote_dir(cls.remote_dir)

    @get_results('/data/foo/', '/', ('rr', 'rl'))
    def test_download_single_file_with_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/bar/', '/', ('rr', 'rl'))
    def test_download_single_nonempty_file_with_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/foo/a', '/', ('rr', 'rl'))
    def test_download_single_file_without_slash(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/', '/')
    def test_download_directory(self, batch_results, gsutil_results, same):
        # gsutil doesn't copy files that end in a slash
        # https://github.com/GoogleCloudPlatform/gsutil/issues/444
        expected = {'/data/foo/', '/data/foo/a', '/data/foo/b', '/data/bar/'}

        assert batch_results['rr']['files'] == expected
        assert batch_results['rr']['err'] is None

        assert batch_results['rl']['files'] == set()
        # We refuse to copy all the files because the file ends in a slash
        # gsutil ignores it with return code 0
        assert isinstance(batch_results['rl']['err'], OperationNotPermittedError)

    @get_results('/data/foo', '/', ('rr',))
    def test_download_directory_partial_name(self, batch_results, gsutil_results, same):
        assert same

    @get_results('/data/f*', '/', ('rr',))
    def test_download_wildcard(self, batch_results, gsutil_results, same):
        # I'm not sure the gsutil results are correct here
        # It seems like we should either copy foo/ or nothing
        # they copy /foo/a and /foo/b
        assert same
