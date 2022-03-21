from typing import Optional
import gzip
import unittest
import secrets
import os

import hail as hl
from hail.context import _get_local_tmpdir
from hail.utils import hadoop_open, hadoop_copy
from hail.fs.local_fs import LocalFS
from hailtop.utils import secret_alnum_string
from hailtop.config import get_remote_tmpdir
from hail.utils.java import FatalError
from ..helpers import startTestHailContext, stopTestHailContext, _initialized


setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.remote_tmpdir = os.environ['HAIL_TEST_STORAGE_URI']
        if cls.remote_tmpdir[-1] == '/':
            cls.remote_tmpdir = cls.remote_tmpdir[:-1]

        local_tmpdir = _get_local_tmpdir(None)
        local_tmpdir = local_tmpdir[len('file://'):]
        cls.local_dir = os.path.join(local_tmpdir, secret_alnum_string(5))

        os.makedirs(cls.local_dir)

        with open(os.path.join(cls.local_dir, 'randomBytes'), 'wb') as f:
            f.write(secrets.token_bytes(2048))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.local_dir)

    def test_hadoop_methods(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.remote_tmpdir

        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))

        with hadoop_open(f'{prefix}/test_out.txt', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open(f'{prefix}/test_out.txt') as f:
            data2 = [line.strip() for line in f]

        self.assertEqual(data, data2)

        with hadoop_open(f'{prefix}/test_out.txt.gz', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open(f'{prefix}/test_out.txt.gz') as f:
            data3 = [line.strip() for line in f]

        self.assertEqual(data, data3)

        hadoop_copy(f'{prefix}/test_out.txt.gz',
                    f'{prefix}/test_out.copy.txt.gz')

        with hadoop_open(f'{prefix}/test_out.copy.txt.gz') as f:
            data4 = [line.strip() for line in f]

        self.assertEqual(data, data4)

        local_fs = LocalFS()
        with local_fs.open(os.path.join(self.local_dir, 'randomBytes'), 'rb', buffer_size=100) as f:
            with hadoop_open(f'{prefix}/randomBytesOut', 'wb', buffer_size=2**18) as out:
                b = f.read()
                out.write(b)

        with hadoop_open(f'{prefix}/randomBytesOut', 'rb', buffer_size=2**18) as f:
            b2 = f.read()

        self.assertEqual(b, b2)

    def test_hadoop_methods_local(self):
        self.test_hadoop_methods(self.local_dir)

    def test_hadoop_exists(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.remote_tmpdir

        with hadoop_open(f'{prefix}/test_exists.txt', 'w') as f:
            f.write("HELLO WORLD")

        r_exists = f'{prefix}/test_exists.txt'
        r_not_exists = f'{prefix}/not_exists.txt'
        self.assertTrue(hl.hadoop_exists(r_exists))
        self.assertFalse(hl.hadoop_exists(r_not_exists))

    def test_hadoop_exists_local(self):
        self.test_hadoop_exists(self.local_dir)

    def test_hadoop_is_file(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.remote_tmpdir

        a_file = f'{prefix}/test_hadoop_is_file.txt'
        with hadoop_open(a_file, 'w') as f:
            f.write("HELLO WORLD")

        self.assertTrue(hl.hadoop_is_file(a_file))
        self.assertFalse(hl.hadoop_is_file(f'{prefix}/'))
        self.assertFalse(hl.hadoop_is_file(f'{prefix}/invalid-path'))

    def test_hadoop_is_file_local(self):
        self.test_hadoop_is_file(self.local_dir)

    def test_hadoop_stat(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.remote_tmpdir

        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))
        with hadoop_open(f'{prefix}/test_hadoop_stat.txt.gz', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        stat1 = hl.hadoop_stat(f'{prefix}')
        self.assertEqual(stat1['is_dir'], True)

        hadoop_copy(f'{prefix}/test_hadoop_stat.txt.gz',
                    f'{prefix}/test_hadoop_stat.copy.txt.gz')

        stat2 = hl.hadoop_stat(f'{prefix}/test_hadoop_stat.copy.txt.gz')
        # The gzip format permits metadata which makes the compressed file's size unpredictable. In
        # practice, Hadoop creates a 175 byte file and gzip.GzipFile creates a 202 byte file. The 27
        # extra bytes appear to include at least the filename (20 bytes) and a modification timestamp.
        assert stat2['size_bytes'] == 175 or stat2['size_bytes'] == 202
        self.assertEqual(stat2['is_dir'], False)
        self.assertTrue('path' in stat2)

    def test_hadoop_stat_local(self):
        self.test_hadoop_stat(self.local_dir)

    def test_subdirs(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.remote_tmpdir

        fs = hl.current_backend().fs

        dir = f'{prefix}foo/'
        subdir1 = f'{dir}foo/'
        subdir1subdir1 = f'{subdir1}foo/'
        subdir1subdir2 = f'{subdir1}bar/'
        subdir1subdir3 = f'{subdir1}baz/'
        subdir1subdir4_empty = f'{subdir1}qux/'
        subdir2 = f'{dir}bar/'
        subdir3 = f'{dir}baz/'
        subdir4_empty = f'{dir}qux/'

        def touch(filename):
            with fs.open(filename, 'w') as fobj:
                fobj.write('hello world')

        fs.mkdir(dir)
        touch(f'{dir}a')
        touch(f'{dir}b')

        fs.mkdir(subdir1)
        fs.mkdir(subdir1subdir1)
        fs.mkdir(subdir1subdir2)
        fs.mkdir(subdir1subdir3)
        fs.mkdir(subdir1subdir4_empty)
        fs.mkdir(subdir2)
        fs.mkdir(subdir3)
        fs.mkdir(subdir4_empty)

        for subdir in [dir, subdir1, subdir2, subdir3, subdir1subdir1, subdir1subdir2, subdir1subdir3]:
            for i in range(30):
                touch(f'{subdir}a{i:02}')

        assert fs.is_dir(dir)
        assert fs.is_dir(subdir1)
        assert fs.is_dir(subdir1subdir1)
        assert fs.is_dir(subdir1subdir2)
        assert fs.is_dir(subdir1subdir3)
        # subdir1subdir4_empty: in cloud fses, empty dirs do not exist and thus are not dirs
        assert fs.is_dir(subdir2)
        assert fs.is_dir(subdir3)
        # subdir4_empty: in cloud fses, empty dirs do not exist and thus are not dirs

        fs.rmtree(subdir1subdir2)

        assert fs.is_dir(dir)
        assert fs.is_file(f'{dir}a')
        assert fs.is_file(f'{dir}b')

        assert fs.is_dir(subdir1)
        assert fs.is_file(f'{subdir1}a00')

        assert fs.is_dir(subdir1subdir1)
        assert fs.is_file(f'{subdir1subdir1}a00')

        assert not fs.is_dir(subdir1subdir2)
        assert not fs.is_file(f'{subdir1subdir2}a00')

        assert fs.is_dir(subdir1subdir3)
        assert fs.is_file(f'{subdir1subdir3}a00')

        assert fs.is_dir(subdir2)
        assert fs.is_file(f'{subdir2}a00')
        assert fs.is_dir(subdir3)
        assert fs.is_file(f'{subdir3}a00')

        fs.rmtree(dir)

        assert not fs.is_dir(dir)

    def test_subdirs_local(self):
        self.test_subdirs(self.local_dir)

    def test_remove_and_rmtree(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.remote_tmpdir

        fs = hl.current_backend().fs

        dir = f'{prefix}foo/'
        subdir1 = f'{dir}foo/'
        subdir1subdir1 = f'{subdir1}foo/'
        subdir1subdir2 = f'{subdir1}bar/'
        subdir1subdir3 = f'{subdir1}baz/'

        def touch(filename):
            with fs.open(filename, 'w') as fobj:
                fobj.write('hello world')

        fs.mkdir(dir)
        touch(f'{dir}a')
        touch(f'{dir}b')

        fs.mkdir(subdir1)
        touch(f'{subdir1}a')
        fs.mkdir(subdir1subdir1)
        touch(f'{subdir1subdir1}a')
        fs.mkdir(subdir1subdir2)
        touch(f'{subdir1subdir2}a')
        fs.mkdir(subdir1subdir3)
        touch(f'{subdir1subdir3}a')

        try:
            fs.remove(subdir1subdir2)
        except (FileNotFoundError, IsADirectoryError):
            pass
        except FatalError as err:
            java_nio_error_message = 'DirectoryNotEmptyException: Cannot delete a non-empty directory'
            hadoop_error_message = f'Directory {subdir1subdir2.rstrip("/")} is not empty'
            assert java_nio_error_message in err.args[0] or hadoop_error_message in err.args[0]
        else:
            assert False

        fs.remove(f'{subdir1subdir2}a')

        assert fs.exists(dir)
        assert fs.exists(f'{dir}a')
        assert fs.exists(f'{dir}b')
        assert fs.exists(subdir1)
        assert fs.exists(f'{subdir1}a')
        assert fs.exists(subdir1subdir1)
        assert fs.exists(f'{subdir1subdir1}a')
        # subdir1subdir2: will exist in cloud, but not local, so do not test for it
        assert not fs.exists(f'{subdir1subdir2}a')
        assert fs.exists(subdir1subdir3)
        assert fs.exists(f'{subdir1subdir3}a')

        fs.rmtree(subdir1subdir1)

        assert fs.exists(dir)
        assert fs.exists(f'{dir}a')
        assert fs.exists(f'{dir}b')
        assert fs.exists(subdir1)
        assert fs.exists(f'{subdir1}a')
        assert not fs.exists(subdir1subdir1)
        assert not fs.exists(f'{subdir1subdir1}a')
        # subdir1subdir2: will exist in cloud, but not local, so do not test for it
        assert not fs.exists(f'{subdir1subdir2}a')
        assert fs.exists(subdir1subdir3)
        assert fs.exists(f'{subdir1subdir3}a')

        fs.rmtree(subdir1)

        assert fs.exists(dir)
        assert fs.exists(f'{dir}a')
        assert fs.exists(f'{dir}b')
        assert not fs.exists(subdir1)
        assert not fs.exists(f'{subdir1}a')
        assert not fs.exists(subdir1subdir1)
        assert not fs.exists(f'{subdir1subdir1}a')
        assert not fs.exists(subdir1subdir2)
        assert not fs.exists(f'{subdir1subdir2}a')
        assert not fs.exists(subdir1subdir3)
        assert not fs.exists(f'{subdir1subdir3}a')

    def test_remove_and_rmtree_local(self):
        self.test_remove_and_rmtree(self.local_dir)
