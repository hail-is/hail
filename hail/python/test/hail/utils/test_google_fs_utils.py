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

        stat1 = hl.hadoop_stat(f'{prefix}')
        self.assertEqual(stat1['is_dir'], True)

        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))
        with hadoop_open(f'{prefix}/test_hadoop_stat.txt.gz', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        expected_uncompressed_bytes = ('\n'.join(data) + '\n').encode('utf-8')
        expected_compressed_length = len(gzip.compress(expected_uncompressed_bytes))
        assert expected_compressed_length == 175

        hadoop_copy(f'{prefix}/test_hadoop_stat.txt.gz',
                    f'{prefix}/test_hadoop_stat.copy.txt.gz')

        stat2 = hl.hadoop_stat(f'{prefix}/test_hadoop_stat.copy.txt.gz')
        self.assertEqual(stat2['size_bytes'], expected_compressed_length)
        self.assertEqual(stat2['is_dir'], False)
        self.assertTrue('path' in stat2)

    def test_hadoop_stat_local(self):
        self.test_hadoop_stat(self.local_dir)
