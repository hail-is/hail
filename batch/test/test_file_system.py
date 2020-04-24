import unittest
import tempfile
import concurrent.futures

from batch.file_system import LocalFileSystem

thread_pool = None


def setUpModule():
    global thread_pool
    thread_pool = concurrent.futures.ThreadPoolExecutor()


def tearDownModule():
    thread_pool.shutdown()


class TestFileSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = None
        cls.temp_dir = None
        raise unittest.SkipTest

    def test_empty_directory(self):
        with self.temp_dir() as tmpdir:
            assert self.fs._exists(tmpdir)
            assert self.fs._isdir(tmpdir)
            assert not self.fs._isfile(tmpdir)

            assert set(self.fs._listdir(f'{tmpdir}')) == set()
            assert set(self.fs._glob(f'{tmpdir}')) == {f'{tmpdir}'}
            assert set(self.fs._glob(f'{tmpdir}/')) == {f'{tmpdir}/'}

    def test_single_file(self):
        with self.temp_dir() as tmpdir:
            path = f'{tmpdir}/a'
            self.fs._touch(path, size=5)

            assert self.fs._exists(path)
            assert self.fs._isfile(path)
            assert not self.fs._isdir(path)

            assert set(self.fs._listdir(f'{tmpdir}')) == {'a'}
            assert set(self.fs._listdir(f'{tmpdir}/')) == {'a'}
            self.assertRaises(NotADirectoryError, self.fs._listdir, f'{tmpdir}/a')

            assert set(self.fs._glob(f'{tmpdir}/a')) == {f'{tmpdir}/a'}
            assert set(self.fs._glob(f'{tmpdir}/*')) == {f'{tmpdir}/a'}
            assert set(self.fs._glob(f'{tmpdir}/?')) == {f'{tmpdir}/a'}
            assert set(self.fs._glob(f'{tmpdir}/[b]')) == set()
            assert set(self.fs._glob(f'{tmpdir}/b')) == set()
            assert set(self.fs._glob(f'{tmpdir}/a/')) == set()

    def test_multiple_files_top_level(self):
        with self.temp_dir() as tmpdir:
            paths = {f'{tmpdir}/a{i}' for i in range(5)}
            for p in paths:
                self.fs._touch(p, size=0)
                assert self.fs._exists(p)
                assert self.fs._isfile(p)
                assert not self.fs._isdir(p)

            assert set(self.fs._listdir(f'{tmpdir}')) == {'a0', 'a1', 'a2', 'a3', 'a4'}
            assert set(self.fs._listdir(f'{tmpdir}/')) == {'a0', 'a1', 'a2', 'a3', 'a4'}
            self.assertRaises(NotADirectoryError, self.fs._listdir, f'{tmpdir}/a1')

            assert set(self.fs._glob(f'{tmpdir}/*')) == paths
            assert set(self.fs._glob(f'{tmpdir}/?')) == set()
            assert set(self.fs._glob(f'{tmpdir}/a?')) == paths
            assert set(self.fs._glob(f'{tmpdir}/a[1]')) == {f'{tmpdir}/a1'}
            assert set(self.fs._glob(f'{tmpdir}/a[!0124]')) == {f'{tmpdir}/a3'}
            assert set(self.fs._glob(f'{tmpdir}/a[01234]')) == paths

    def test_files_with_wildcards(self):
        with self.temp_dir() as tmpdir:
            dirs = {f'{tmpdir}/bar', f'{tmpdir}/b?r', f'{tmpdir}/b\\?r'}
            for dir in dirs:
                self.fs._mkdir(dir)
                assert self.fs._isdir(dir)

            assert set(self.fs._listdir(f'{tmpdir}')) == {'bar', 'b?r', 'b\\?r'}
            assert set(self.fs._listdir(f'{tmpdir}/')) == {'bar', 'b?r', 'b\\?r'}

            assert set(self.fs._glob(f'{tmpdir}/*')) == dirs
            assert set(self.fs._glob(f'{tmpdir}/b?r')) == {f'{tmpdir}/b?r', f'{tmpdir}/bar'}
            assert set(self.fs._glob(f'{tmpdir}/b\\?r')) == {f'{tmpdir}/b?r'}
            assert set(self.fs._glob(f'{tmpdir}/b\\\\?r')) == {f'{tmpdir}/b\\?r'}


class TestLocalFileSystem(TestFileSystem):
    @classmethod
    def setUpClass(cls):
        cls.fs = LocalFileSystem(thread_pool)
        cls.temp_dir = lambda x: tempfile.TemporaryDirectory()
