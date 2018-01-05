from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext
from hail.utils import *
from .queue import *

hc = None

def setUpModule():
    global hc
    hc = HailContext()  # master = 'local[2]')

def tearDownModule():
    global hc
    hc.stop()
    hc = None

class Tests(unittest.TestCase):
    def test_hadoop_methods(self):
        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))

        with hadoop_write('/tmp/test_out.txt') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_read('/tmp/test_out.txt') as f:
            data2 = [line.strip() for line in f]

        self.assertEqual(data, data2)

        with hadoop_write('/tmp/test_out.txt.gz') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_read('/tmp/test_out.txt.gz') as f:
            data3 = [line.strip() for line in f]

        self.assertEqual(data, data3)

        hadoop_copy('/tmp/test_out.txt.gz', '/tmp/test_out.copy.txt.gz')

        with hadoop_read('/tmp/test_out.copy.txt.gz') as f:
            data4 = [line.strip() for line in f]

        self.assertEqual(data, data4)

        with hadoop_read('src/test/resources/randomBytes', buffer_size=100) as f:
            with hadoop_write('/tmp/randomBytesOut', buffer_size=150) as out:
                b = f.read()
                out.write(b)

        with hadoop_read('/tmp/randomBytesOut', buffer_size=199) as f:
            b2 = f.read()

        self.assertEqual(b, b2)

    def test_queue(self):
        q = Queue()
        self.assertEqual(list(q), [])
        if q:
            self.fail('empty queue had an implicit boolean value of True')

        q = q.push(5).push(2)

        self.assertEqual(list(q), [5, 2])
        self.assertEqual(len(q), 2)

        if not q:
            self.fail('queue had an implicit boolean value of False')

        q2 = Queue().push(5, 2)
        self.assertEqual(list(q), list(q2))


        q3 = Queue().push(1)
        q4 = q3.push(2, 3)
        q5 = q3.push(4, 5)

        self.assertEqual(list(q4), [1, 2, 3])
        self.assertEqual(list(q5), [1, 4, 5])