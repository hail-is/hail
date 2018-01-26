from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext
from hail.utils import *
from .linkedlist import LinkedList
from hail2 import *

def setUpModule():
    init(master='local[2]', min_block_size=0)

def tearDownModule():
    stop()

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

    def test_linked_list(self):
        ll = LinkedList(int)
        self.assertEqual(list(ll), [])
        if ll:
            self.fail('empty linked list had an implicit boolean value of True')

        ll2 = ll.push(5).push(2)

        self.assertEqual(list(ll2), [2, 5])

        if not ll2:
            self.fail('populated linked list had an implicit boolean value of False')

        ll3 = ll.push(5, 2)
        self.assertEqual(list(ll2), list(ll3))
        self.assertEqual(ll2, ll3)

        ll4 = ll.push(1)
        ll5 = ll4.push(2, 3)
        ll6 = ll4.push(4, 5)

        self.assertEqual(list(ll5), [3, 2, 1])
        self.assertEqual(list(ll6), [5, 4, 1])

    def test_struct_ops(self):
        s = Struct(a=1, b=2, c=3)

        self.assertEqual(s.drop('c'), Struct(b=2, a=1))
        self.assertEqual(s.drop('b', 'c'), Struct(a=1))

        self.assertEqual(s.select('b', 'a'), Struct(b=2, a=1))
        self.assertEqual(s.select('a', b=5), Struct(a=1, b=5))

        self.assertEqual(s.annotate(), s)
        self.assertEqual(s.annotate(x=5), Struct(a=1, b=2, c=3, x=5))
        self.assertEqual(s.annotate(**{'a': 5, 'x': 10, 'y': 15}), Struct(a=5, b=2, c=3, x=10, y=15))
