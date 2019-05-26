import unittest

import hail as hl
from hail.utils import *
from hail.utils.java import Env
from hail.utils.linkedlist import LinkedList
from hail.fs.hadoop_fs import HadoopFS
from ..helpers import *
import os

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

BUCKET = os.environ.get("TEST_BUCKET_NAME", None)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BUCKET is None:
            raise unittest.case.SkipTest("TEST_BUCKET_NAME not set in env")
        if 'HAIL_TEST_SERVICE_BACKEND_URL' not in os.environ:
            raise unittest.case.SkipTest("HAIL_TEST_SERVICE_BACKEND_URL not set in env")

    def test_hadoop_methods(self):
        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))

        with hadoop_open(f'{BUCKET}/test_out.txt', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open(f'{BUCKET}/test_out.txt') as f:
            data2 = [line.strip() for line in f]

        self.assertEqual(data, data2)

        with hadoop_open(f'{BUCKET}/test_out.txt.gz', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open(f'{BUCKET}/test_out.txt.gz') as f:
            data3 = [line.strip() for line in f]

        self.assertEqual(data, data3)

        hadoop_copy(f'{BUCKET}/test_out.txt.gz',
                    f'{BUCKET}/test_out.copy.txt.gz')

        with hadoop_open(f'{BUCKET}/test_out.copy.txt.gz') as f:
            data4 = [line.strip() for line in f]

        self.assertEqual(data, data4)

        local_fs = HadoopFS()
        with local_fs.open(resource('randomBytes'), buffer_size=100) as f:
            with hadoop_open(f'{BUCKET}/randomBytesOut', 'w', buffer_size=2**18) as out:
                b = f.read()
                out.write(b)

        with hadoop_open(f'{BUCKET}/randomBytesOut', buffer_size=2**18) as f:
            b2 = f.read()

        self.assertEqual(b, b2)

        with self.assertRaises(Exception):
            hadoop_open(f'{BUCKET}/randomBytesOut', 'xb')

    def test_hadoop_exists(self):
        with hadoop_open(f'{BUCKET}/test_exists.txt', 'w') as f:
            f.write("HELLO WORLD")

        r_exists = f'{BUCKET}/test_exists.txt'
        r_not_exists = f'{BUCKET}/not_exists.txt'
        self.assertTrue(hl.hadoop_exists(r_exists))
        self.assertFalse(hl.hadoop_exists(r_not_exists))

    def test_hadoop_is_file(self):
        a_file = f'{BUCKET}/test_hadoop_is_file.txt'
        with hadoop_open(a_file, 'w') as f:
            f.write("HELLO WORLD")

        self.assertTrue(hl.hadoop_is_file(a_file))
        self.assertFalse(hl.hadoop_is_file(f'{BUCKET}/'))
        self.assertFalse(hl.hadoop_is_file(f'{BUCKET}/invalid-path'))

    def test_hadoop_stat(self):
        stat1 = hl.hadoop_stat(f'{BUCKET}/')
        self.assertEqual(stat1['is_dir'], True)

        stat2 = hl.hadoop_stat(f'{BUCKET}/test_out.copy.txt.gz')
        self.assertEqual(stat2['size_bytes'], 302)
        self.assertEqual(stat2['is_dir'], False)
        self.assertTrue('path' in stat2)
        self.assertTrue('owner' in stat2)
        self.assertTrue('modification_time' in stat2)
