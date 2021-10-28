import unittest

import hail as hl
from hail.utils import *
from hail.utils.misc import escape_str, escape_id
from hail.utils.java import Env
from hail.utils.linkedlist import LinkedList

from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):

    def test_hadoop_methods(self):
        data = ['foo', 'bar', 'baz']
        data.extend(map(str, range(100)))

        with hadoop_open('/tmp/test_out.txt', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open('/tmp/test_out.txt') as f:
            data2 = [line.strip() for line in f]

        self.assertEqual(data, data2)

        with hadoop_open('/tmp/test_out.txt.gz', 'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

        with hadoop_open('/tmp/test_out.txt.gz') as f:
            data3 = [line.strip() for line in f]

        self.assertEqual(data, data3)

        hadoop_copy('/tmp/test_out.txt.gz', '/tmp/test_out.copy.txt.gz')

        with hadoop_open('/tmp/test_out.copy.txt.gz') as f:
            data4 = [line.strip() for line in f]

        self.assertEqual(data, data4)

        with hadoop_open(resource('randomBytes'), mode='rb', buffer_size=100) as f:
            with hadoop_open('/tmp/randomBytesOut', mode='wb', buffer_size=150) as out:
                b = f.read()
                out.write(b)

        with hadoop_open('/tmp/randomBytesOut', mode='rb', buffer_size=199) as f:
            b2 = f.read()

        self.assertEqual(b, b2)

        with self.assertRaises(Exception):
            hadoop_open('/tmp/randomBytesOut', 'xb')

    def test_hadoop_exists(self):
        self.assertTrue(hl.hadoop_exists(resource('ls_test')))
        self.assertFalse(hl.hadoop_exists(resource('doesnt.exist')))

    def test_hadoop_mkdir_p(self):
        test_text = "HELLO WORLD"

        with hadoop_open(resource('./some/foo/bar.txt'), 'w') as out:
            out.write(test_text)

        self.assertTrue(hl.hadoop_exists(resource('./some/foo/bar.txt')))

        with hadoop_open(resource('./some/foo/bar.txt')) as f:
            assert(f.read() == test_text)

        import shutil
        shutil.rmtree(resource('./some'))

    def test_hadoop_mkdir_p(self):
        with self.assertRaises(Exception):
            hadoop_open(resource('./some2/foo/bar.txt'), 'r')

        self.assertFalse(hl.hadoop_exists(resource('./some2')))

    @skip_when_service_backend('service backend logs are not sent to a user-visible file')
    @fails_local_backend()
    def test_hadoop_copy_log(self):
        with with_local_temp_file('log') as r:
            hl.copy_log(r)
            stats = hl.hadoop_stat(r)
            self.assertTrue(stats['size_bytes'] > 0)

    def test_hadoop_is_file(self):
        self.assertTrue(hl.hadoop_is_file(resource('ls_test/f_50')))
        self.assertFalse(hl.hadoop_is_file(resource('ls_test/subdir')))
        self.assertFalse(hl.hadoop_is_file(resource('ls_test/invalid-path')))

    def test_hadoop_is_dir(self):
        self.assertTrue(hl.hadoop_is_dir(resource('ls_test/subdir')))
        self.assertFalse(hl.hadoop_is_dir(resource('ls_test/f_50')))
        self.assertFalse(hl.hadoop_is_dir(resource('ls_test/invalid-path')))

    def test_hadoop_stat(self):
        path1 = resource('ls_test')
        stat1 = hl.hadoop_stat(path1)
        self.assertEqual(stat1['is_dir'], True)

        path2 = resource('ls_test/f_50')
        stat2 = hl.hadoop_stat(path2)
        self.assertEqual(stat2['size_bytes'], 50)
        self.assertEqual(stat2['is_dir'], False)
        self.assertTrue('path' in stat2)
        self.assertTrue('owner' in stat2)
        self.assertTrue('modification_time' in stat2)

    @fails_service_backend()
    @fails_local_backend()
    def test_hadoop_ls(self):
        path1 = resource('ls_test/f_50')
        ls1 = hl.hadoop_ls(path1)
        self.assertEqual(len(ls1), 1)
        self.assertEqual(ls1[0]['size_bytes'], 50)
        self.assertEqual(ls1[0]['is_dir'], False)
        self.assertTrue('path' in ls1[0])
        self.assertTrue('owner' in ls1[0])
        self.assertTrue('modification_time' in ls1[0])

        path2 = resource('ls_test')
        ls2 = hl.hadoop_ls(path2)
        self.assertEqual(len(ls2), 3)
        ls2_dict = {x['path'].split("/")[-1]: x for x in ls2}
        self.assertEqual(ls2_dict['f_50']['size_bytes'], 50)
        self.assertEqual(ls2_dict['f_100']['size_bytes'], 100)
        self.assertEqual(ls2_dict['f_100']['is_dir'], False)
        self.assertEqual(ls2_dict['subdir']['is_dir'], True)
        self.assertTrue('owner' in ls2_dict['f_50'])
        self.assertTrue('modification_time' in ls2_dict['f_50'])

        path3 = resource('ls_test/f*')
        ls3 = hl.hadoop_ls(path3)
        assert len(ls3) == 2, ls3

        with self.assertRaisesRegex(Exception, "FileNotFound"):
            hl.hadoop_ls('a_file_that_does_not_exist')

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
        self.assertEqual(s.annotate(**{'a': 5, 'x': 10, 'y': 15}),
                         Struct(a=5, b=2, c=3, x=10, y=15))

    def test_expr_exception_results_in_hail_user_error(self):
        df = range_table(10)
        df = df.annotate(x=[1, 2])
        with self.assertRaises(HailUserError):
            df.filter(df.x[5] == 0).count()

    def test_interval_ops(self):
        interval1 = Interval(3, 22)
        interval2 = Interval(10, 20)

        self.assertTrue(interval1.start == 3)
        self.assertTrue(interval1.end == 22)
        self.assertTrue(interval1.includes_start)
        self.assertFalse(interval1.includes_end)
        self.assertTrue(interval1.point_type == hl.tint)

        self.assertTrue(interval1.contains(3))
        self.assertTrue(interval1.contains(13))
        self.assertFalse(interval1.contains(22))
        self.assertTrue(interval1.overlaps(interval2))

    def test_range_matrix_table_n_lt_partitions(self):
        hl.utils.range_matrix_table(1, 1)._force_count_rows()

    def test_seeding_is_consistent(self):
        hl.set_global_seed(0)
        a = [Env.next_seed() for _ in range(10)]
        hl.set_global_seed(0)
        b = [Env.next_seed() for _ in range(10)]

        self.assertEqual(len(set(a)), 10)
        self.assertEqual(a, b)

    def test_escape_string(self):
        self.assertEqual(escape_str("\""), "\\\"")
        self.assertEqual(escape_str("cat"), "cat")
        self.assertEqual(escape_str("my name is 名谦"), "my name is \\u540D\\u8C26")
        self.assertEqual(escape_str('"', backticked=True), '"')
        self.assertEqual(escape_str(chr(200)), '\\u00C8')
        self.assertEqual(escape_str(chr(500)), '\\u01F4')

    def test_escape_id(self):
        self.assertEqual(escape_id("`"), "`\\``")
        self.assertEqual(escape_id("cat"), "cat")
        self.assertEqual(escape_id("abc123"), "abc123")
        self.assertEqual(escape_id("123abc"), "`123abc`")

    def test_frozen_dict(self):
        self.assertEqual(frozendict({1:2, 4:7}), frozendict({1:2, 4:7}))
        my_frozen_dict = frozendict({"a": "apple", "h": "hail"})
        self.assertEqual(my_frozen_dict["a"], "apple")

        # Make sure mutating old dict doesn't change frozen counterpart.
        regular_dict = {"a": "b"}
        frozen_counterpart = frozendict(regular_dict)
        regular_dict["a"] = "d"
        self.assertEqual(frozen_counterpart["a"], "b")

        with pytest.raises(TypeError, match="does not support item assignment"):
            my_frozen_dict["a"] = "b"
