import unittest

import hail as hl
from hail.utils import hadoop_open, hadoop_copy
from hail.fs.hadoop_fs import HadoopFS
from ..helpers import startTestHailContext, stopTestHailContext, resource, _initialized
import os


setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bucket = os.environ.get("TEST_BUCKET_NAME", None)

        if bucket is None:
            raise unittest.case.SkipTest("TEST_BUCKET_NAME not set in env")
        if 'HAIL_APISERVER_URL' not in os.environ:
            raise unittest.case.SkipTest("HAIL_APISERVER_URL not set in env")

        import secrets

        with open('randomBytes', 'wb') as f:
            f.write(secrets.token_bytes(2048))

        if bucket.startswith('gs://'):
            cls.remote_bucket = bucket
            cls.local_dir = f"/tmp/{bucket[5:]}"
        else:
            cls.remote_bucket = f"gs://{bucket}"
            cls.local_dir = f"/tmp/{bucket}"

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.local_dir)

    def test_hadoop_methods(self, bucket=None):
        if bucket is None:
            bucket = self.remote_bucket

        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))

        with hadoop_open(f'{bucket}/test_out.txt', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open(f'{bucket}/test_out.txt') as f:
            data2 = [line.strip() for line in f]

        self.assertEqual(data, data2)

        with hadoop_open(f'{bucket}/test_out.txt.gz', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open(f'{bucket}/test_out.txt.gz') as f:
            data3 = [line.strip() for line in f]

        self.assertEqual(data, data3)

        hadoop_copy(f'{bucket}/test_out.txt.gz',
                    f'{bucket}/test_out.copy.txt.gz')

        with hadoop_open(f'{bucket}/test_out.copy.txt.gz') as f:
            data4 = [line.strip() for line in f]

        self.assertEqual(data, data4)

        local_fs = HadoopFS()
        with local_fs.open(resource('randomBytes'), buffer_size=100) as f:
            with hadoop_open(f'{bucket}/randomBytesOut', 'w', buffer_size=2**18) as out:
                b = f.read()
                out.write(b)

        with hadoop_open(f'{bucket}/randomBytesOut', buffer_size=2**18) as f:
            b2 = f.read()

        self.assertEqual(b, b2)

    def test_hadoop_methods_local(self):
        self.test_hadoop_methods(self.local_dir)

    def test_hadoop_exists(self, bucket=None):
        if bucket is None:
            bucket = self.remote_bucket

        with hadoop_open(f'{bucket}/test_exists.txt', 'w') as f:
            f.write("HELLO WORLD")

        r_exists = f'{bucket}/test_exists.txt'
        r_not_exists = f'{bucket}/not_exists.txt'
        self.assertTrue(hl.hadoop_exists(r_exists))
        self.assertFalse(hl.hadoop_exists(r_not_exists))

    def test_hadoop_exists_local(self):
        self.test_hadoop_exists(self.local_dir)

    def test_hadoop_is_file(self, bucket=None):
        if bucket is None:
            bucket = self.remote_bucket

        a_file = f'{bucket}/test_hadoop_is_file.txt'
        with hadoop_open(a_file, 'w') as f:
            f.write("HELLO WORLD")

        self.assertTrue(hl.hadoop_is_file(a_file))
        self.assertFalse(hl.hadoop_is_file(f'{bucket}/'))
        self.assertFalse(hl.hadoop_is_file(f'{bucket}/invalid-path'))

    def test_hadoop_is_file_local(self):
        self.test_hadoop_is_file(self.local_dir)

    def test_hadoop_stat(self, bucket=None):
        if bucket is None:
            bucket = self.remote_bucket

        stat1 = hl.hadoop_stat(f'{bucket}/')
        self.assertEqual(stat1['is_dir'], True)

        stat2 = hl.hadoop_stat(f'{bucket}/test_out.copy.txt.gz')
        self.assertEqual(stat2['size_bytes'], 302)
        self.assertEqual(stat2['is_dir'], False)
        self.assertTrue('path' in stat2)
        self.assertTrue('owner' in stat2)
        self.assertTrue('modification_time' in stat2)

    def test_hadoop_stat_local(self):
        self.test_hadoop_stat(self.local_dir)
