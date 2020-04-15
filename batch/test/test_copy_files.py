import os
import unittest
import tempfile
import glob
import subprocess as sp
import uuid
from shlex import quote as shq
import google.oauth2.service_account
from hailtop.utils import flatten
from hailtop.auth import get_userinfo

from batch.google_storage import GCS

key_file = '/gsa-key/key.json'
project = os.environ['PROJECT']
user = get_userinfo()
tmp_bucket = user['bucket_name']

credentials = google.oauth2.service_account.Credentials.from_service_account_file(key_file)
gcs_client = GCS(None, project=project, credentials=credentials)


class RemoteTemporaryDirectory:
    def __init__(self):
        token = uuid.uuid4().hex[:6]
        self.name = tmp_bucket + f'/{token}'

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_remote_dir(self.name)


def upload_files(src, dest):
    os.system(f'gsutil -m -q cp -r {src} {dest}')


def move_files(src, dest):
    os.system(f'gsutil -m -q mv -r {src} {dest}')


def remove_remote_dir(path):
    os.system(f'gsutil -m -q rm -r {path}')


def touch_file(path, data=None):
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    with open(path, 'w') as file:
        if data:
            file.write(data)


def cp_batch(src, dest):
    cmd = f'python3 -u -m batch.worker.copy_files --key-file {key_file} --project {project} -f {shq(src)} {shq(dest)} 2>&1'
    result = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    return str(result.stdout), result.returncode


def cp_gsutil(src, dest):
    cmd = f'gsutil -m -q cp -r {shq(src)} {shq(dest)} 2>&1'
    result = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    return str(result.stdout), result.returncode


def _glob_local_files(path):
    path = os.path.abspath(path)
    paths = glob.glob(path, recursive=True)

    def listdir(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.isfile(path):
            return [path]
        # gsutil doesn't copy empty directories
        return flatten([listdir(path + '/' + f) for f in os.listdir(path)])

    return flatten([listdir(path) for path in paths])


def glob_local_files(dir):
    files = _glob_local_files(dir)
    return {f.replace(dir, '') for f in files}


def glob_remote_files(dir):
    blobs = gcs_client._list_gs_files(dir)
    files = {'gs://' + blob.bucket.name + '/' + blob.name for blob in blobs}
    return {f.replace(dir, '') for f in files}


def _copy_to_local(src, dest):
    with tempfile.TemporaryDirectory() as batch_dest_dir:
        batch_output, batch_rc = cp_batch(src, batch_dest_dir + dest)
        batch_files = glob_local_files(batch_dest_dir)

    with tempfile.TemporaryDirectory() as gsutil_dest_dir:
        gsutil_output, gsutil_rc = cp_gsutil(src, gsutil_dest_dir + dest)
        gsutil_files = glob_local_files(gsutil_dest_dir)

    return {
        'batch': {'files': batch_files, 'output': batch_output, 'rc': batch_rc},
        'gsutil': {'files': gsutil_files, 'output': gsutil_output, 'rc': gsutil_rc}
    }


def _copy_to_remote(src, dest):
    with RemoteTemporaryDirectory() as batch_dest_dir:
        batch_output, batch_rc = cp_batch(src, batch_dest_dir + dest)
        batch_files = glob_remote_files(batch_dest_dir)

    with RemoteTemporaryDirectory() as gsutil_dest_dir:
        gsutil_output, gsutil_rc = cp_gsutil(src, gsutil_dest_dir + dest)
        gsutil_files = glob_remote_files(gsutil_dest_dir)

    return {
        'batch': {'files': batch_files, 'output': batch_output, 'rc': batch_rc},
        'gsutil': {'files': gsutil_files, 'output': gsutil_output, 'rc': gsutil_rc}
    }


def _run_copy_step(cp, src, dest):
    result = cp(src, dest)
    batch = result['batch']
    gsutil = result['gsutil']
    batch_error = batch['rc'] != 0
    gsutil_error = gsutil['rc'] != 0
    success = batch['files'] == gsutil['files'] and batch_error == gsutil_error
    return {'success': success, 'result': result}


def run_local_to_remote(src, dest):
    assert not src.startswith('gs://')
    return _run_copy_step(_copy_to_remote, src, dest)


def run_remote_to_local(src, dest):
    assert src.startswith('gs://')
    return _run_copy_step(_copy_to_local, src, dest)


def run_local_to_local(src, dest):
    assert not src.startswith('gs://')
    return _run_copy_step(_copy_to_local, src, dest)


def run_remote_to_remote(src, dest):
    assert src.startswith('gs://')
    return _run_copy_step(_copy_to_remote, src, dest)


def _run_batch_same_as_gsutil(local_dir, remote_dir, src, dest):
    return {
        'rr': run_remote_to_remote(f'{remote_dir}{src}', dest),
        'rl': run_remote_to_local(f'{remote_dir}{src}', dest),
        'lr': run_local_to_remote(f'{local_dir}{src}', dest),
        'll': run_local_to_local(f'{local_dir}{src}', dest)
    }


def _get_output(result, version, method):
    return result[version]['result'][method]['output']


def _get_files(result, version, method):
    return result[version]['result'][method]['files']


def _get_rc(result, version, method):
    return result[version]['result'][method]['rc']


def _get_batch_gsutil_files_same(result):
    return [result['rr']['success'],
            result['rl']['success'],
            result['lr']['success'],
            result['ll']['success']]


class TestEmptyDirectory(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)
        return result

    def test_download_directory(self):
        self.assert_batch_same_as_gsutil('/', '/')

    def test_download_asterisk(self):
        self.assert_batch_same_as_gsutil('/*', '/')

    def test_download_double_asterisk(self):
        result = run_local_to_local(f'{self.local_dir.name}/**', '/')
        assert result['success'] != 0, str(result)
        assert '** not supported' in result['result']['batch']['output'], result['result']['batch']['output']


class TestSingleFileTopLevel(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/a')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)
        return result

    def test_download_file_by_name(self):
        self.assert_batch_same_as_gsutil('/data/a', '/')

    def test_download_file_not_exists(self):
        result = self.assert_batch_same_as_gsutil('/data/b', '/')
        assert 'FileNotFoundError' in _get_output(result, 'rr', 'batch'), _get_output(result, 'rr', 'batch')
        assert 'FileNotFoundError' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        assert 'FileNotFoundError' in _get_output(result, 'lr', 'batch'), _get_output(result, 'lr', 'batch')
        assert 'FileNotFoundError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')

    def test_download_file_by_name_with_slash(self):
        self.assert_batch_same_as_gsutil('/data/a/', '/')

    def test_download_directory(self):
        self.assert_batch_same_as_gsutil('/data/', '/')

    def test_download_single_wildcard(self):
        self.assert_batch_same_as_gsutil('/data/*', '/')

    def test_download_multiple_wildcards(self):
        self.assert_batch_same_as_gsutil('/*/*', '/')

    def test_download_top_level_directory(self):
        self.assert_batch_same_as_gsutil('/', '/')

    def test_download_file_by_name_with_rename(self):
        self.assert_batch_same_as_gsutil('/data/a', '/b')

    def test_download_file_by_wildcard_with_rename(self):
        self.assert_batch_same_as_gsutil('/data/*', '/b')

    def test_download_file_by_name_to_nonexistent_subdir(self):
        self.assert_batch_same_as_gsutil('/data/a', '/foo/b')

    def test_download_file_by_wildcard_to_nonexistent_subdir(self):
        self.assert_batch_same_as_gsutil('/data/*', '/foo/b')


class TestFileNestedInMultipleSubdirs(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/a/b/c')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)
        return result

    def test_download_file_by_name(self):
        self.assert_batch_same_as_gsutil('/data/a/b/c', '/')

    def test_download_file_by_name_with_rename(self):
        self.assert_batch_same_as_gsutil('/data/a/b/c', '/foo')

    def test_download_directory_recursively(self):
        self.assert_batch_same_as_gsutil('/data/', '/')

    def test_download_wildcard_subdir_without_slash(self):
        self.assert_batch_same_as_gsutil('/data/*/b', '/')

    def test_download_wildcard_subdir_with_slash(self):
        self.assert_batch_same_as_gsutil('/data/*/b/', '/')

    def test_download_double_wildcards(self):
        self.assert_batch_same_as_gsutil('/data/*/*/', '/')

    def test_download_double_wildcards_plus_file_wildcard(self):
        self.assert_batch_same_as_gsutil('/data/*/*/*', '/')


class TestDownloadMultipleFilesAtTopLevel(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/a')
        touch_file(self.local_dir.name + '/data/b')
        touch_file(self.local_dir.name + '/data/c')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)
        return result

    def test_download_file_by_name(self):
        self.assert_batch_same_as_gsutil('/data/a', '/')

    def test_download_file_with_rename(self):
        self.assert_batch_same_as_gsutil('/data/a', '/b')

    def test_download_file_asterisk(self):
        self.assert_batch_same_as_gsutil('/data/*', '/')

    def test_download_file_match_brackets(self):
        self.assert_batch_same_as_gsutil('/data/[ab]', '/b')

    def test_download_file_question_mark(self):
        self.assert_batch_same_as_gsutil('/data/?', '/')

    def test_download_file_double_question_marks(self):
        result = self.assert_batch_same_as_gsutil('/data/??', '/')
        assert 'FileNotFoundError' in _get_output(result, 'rr', 'batch'), _get_output(result, 'rr', 'batch')
        assert 'FileNotFoundError' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        assert 'FileNotFoundError' in _get_output(result, 'lr', 'batch'), _get_output(result, 'lr', 'batch')
        assert 'FileNotFoundError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')

    def test_download_multiple_files_to_single_file(self):
        result = self.assert_batch_same_as_gsutil('/data/[ab]', '/b')
        assert 'NotADirectoryError' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        assert 'NotADirectoryError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')

    def test_download_file_invalid_dest_path_with_slash(self):
        result = self.assert_batch_same_as_gsutil('/data/a', '/b/')
        assert 'skipping destination file ending with slash' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')

        # it's unclear why gsutil doesn't just create the directory like it does if the destination is remote
        # it's also unclear why you don't get the same error as for the remote->local case
        assert 'IsADirectoryError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')

    def test_download_file_invalid_dest_dir_with_wildcard(self):
        result = self.assert_batch_same_as_gsutil('/data/*', '/b/')

        assert 'destination must name a directory when matching multiple files' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')
        assert 'NotADirectoryError' in _get_output(result, 'rl', 'batch'), _get_output(result, 'rl', 'batch')

        assert 'destination must name a directory when matching multiple files' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')
        assert 'NotADirectoryError' in _get_output(result, 'll', 'batch'), _get_output(result, 'll', 'batch')


class TestDownloadFileDirectoryWithSameName(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/a')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)
        upload_files(self.remote_dir + '/data/a', self.remote_dir + '/data/a/b')

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)
        return result

    def test_download_file_by_name(self):
        self.assert_batch_same_as_gsutil('/data/a', '/')

    def test_download_file_by_name_in_subdir(self):
        self.assert_batch_same_as_gsutil('/data/a/b', '/')

    def test_download_directory_with_same_name_as_file(self):
        src = '/data/a/'
        dest = '/'

        rr = run_remote_to_remote(f'{self.remote_dir}{src}', dest)
        rl = run_remote_to_local(f'{self.remote_dir}{src}', dest)

        assert rr['success'] and not rl['success']

        rl_batch_output = rl['result']['batch']['output']
        assert 'NotADirectory' in rl_batch_output or 'FileExistsError' in rl_batch_output, rl_batch_output


class TestDownloadFileWithEscapedWildcards(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/foo/bar/dog/a')
        touch_file(self.local_dir.name + '/data/foo/b\\?r/dog/b')
        touch_file(self.local_dir.name + '/data/foo/bar/dog/h\\*llo')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)
        return result

    def test_download_all_files_recursively(self):
        self.assert_batch_same_as_gsutil('/data/foo/', '/')

    def test_download_directory_with_escaped_question_mark(self):
        self.assert_batch_same_as_gsutil('/data/foo/b\\?r/dog/', '/')

    def test_download_directory_with_nonescaped_question_mark(self):
        # gsutil refuses to copy a path with a wildcard in it
        # Cloud folder gs://hail-jigold-59hi5/testing-suite/9f347b/data/foo/b\?r/ contains a wildcard; gsutil does not currently support objects with wildcards in their name.
        # unclear why gsutil will work for other tests
        expected = {'/dog/a', '/dog/b', '/dog/h\\*llo'}

        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, '/data/foo/b?r/dog/', '/')
        assert not all(_get_batch_gsutil_files_same(result)), str(result)

        assert _get_files(result, 'rr', 'batch') == expected, _get_files(result, 'rr', 'batch')
        assert _get_rc(result, 'rr', 'batch') == 0, _get_rc(result, 'rr', 'batch')

        assert _get_files(result, 'rl', 'batch') == expected, _get_files(result, 'rl', 'batch')
        assert _get_rc(result, 'rl', 'batch') == 0, _get_rc(result, 'rl', 'batch')

        assert _get_files(result, 'lr', 'batch') == expected, _get_files(result, 'lr', 'batch')
        assert _get_rc(result, 'lr', 'batch') == 0, _get_rc(result, 'lr', 'batch')

        assert _get_files(result, 'll', 'batch') == expected, _get_files(result, 'll', 'batch')
        assert _get_rc(result, 'll', 'batch') == 0, _get_rc(result, 'll', 'batch')


class TestDownloadFileWithSpaces(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/foo/bar/dog/file with spaces.txt')
        touch_file(self.local_dir.name + '/data/f o o/hello')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)

    def test_download_file_with_spaces(self):
        self.assert_batch_same_as_gsutil('/data/foo/bar/dog/file with spaces.txt', '/')

    def test_directory_with_spaces(self):
        self.assert_batch_same_as_gsutil('/data/f o o/hello', '/')


class TestDownloadComplicatedDirectory(unittest.TestCase):
    def setUp(self):
        self.local_dir = tempfile.TemporaryDirectory()
        touch_file(self.local_dir.name + '/data/foo/a/data1')
        touch_file(self.local_dir.name + '/data/foo/a/data2')
        touch_file(self.local_dir.name + '/data/bar/a')
        touch_file(self.local_dir.name + '/data/baz')
        touch_file(self.local_dir.name + '/data/dog/dog/dog')
        token = uuid.uuid4().hex[:6]
        self.remote_dir = f'{tmp_bucket}/{token}'
        upload_files(self.local_dir.name, self.remote_dir)

    def tearDown(self):
        self.local_dir.cleanup()
        remove_remote_dir(self.remote_dir)

    def assert_batch_same_as_gsutil(self, src, dest):
        result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, src, dest)
        assert all(_get_batch_gsutil_files_same(result)), str(result)

    def test_download_all_files(self):
        # gsutil works for all cases here
        self.assert_batch_same_as_gsutil('/data/', '/')

    def test_download_all_files_without_slash(self):
        # I don't understand why gsutil is failing for the local->local case
        # /var/folders/f_/ystbcjb13z78n85cyz6_jpl9sbv79d/T/tmpfr0056u5/data/foo/a\ CommandException: 1 file/object could not be transferred.
        # The files it did copy are {'/data/baz', '/data/bar/a', '/data/foo/a/data2', '/data/dog/dog/dog'}
        self.assert_batch_same_as_gsutil('/data', '/')

        # result = _run_batch_same_as_gsutil(self.local_dir.name, self.remote_dir, '/data', '/')
        # assert result['rr']['success'] and result['rl']['success'] and result['lr']['success'], str(result)
        #
        # expected = {'/data/foo/a/data1', '/data/foo/a/data2', '/data/bar/a', '/data/baz', '/data/dog/dog/dog'}
        # assert _get_files(result, 'll', 'batch') == expected, _get_files(result, 'll', 'batch')
        # assert _get_rc(result, 'll', 'batch') == 0, _get_rc(result, 'll', 'batch')
