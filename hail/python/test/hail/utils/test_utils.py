import json
import unittest

import hail as hl
from hail.utils import *
from hail.utils.java import FatalError
from hail.utils.linkedlist import LinkedList
from hail.utils.misc import escape_id, escape_str

from ..helpers import *


def normalize_path(path: str) -> str:
    return hl.hadoop_stat(path)['path']


def touch(filename):
    with hl.current_backend().fs.open(filename, 'w') as fobj:
        fobj.write('hello world')


@qobtest
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
        self.assertTrue(hl.hadoop_exists(resource('ls_test/f_50')))
        self.assertFalse(hl.hadoop_exists(resource('doesnt.exist')))

    def test_hadoop_mkdir_p(self):
        test_text = "HELLO WORLD"

        with hadoop_open(resource('./some/foo/bar.txt'), 'w') as out:
            out.write(test_text)

        self.assertTrue(hl.hadoop_exists(resource('./some/foo/bar.txt')))

        with hadoop_open(resource('./some/foo/bar.txt')) as f:
            assert f.read() == test_text

        hl.current_backend().fs.rmtree(resource('./some'))

    def test_hadoop_mkdir_p_2(self):
        with self.assertRaises(Exception):
            hadoop_open(resource('./some2/foo/bar.txt'), 'r')

        self.assertFalse(hl.hadoop_exists(resource('./some2')))

    @fails_service_backend(reason='service backend logs are not sent to a user-visible file')
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
        self.assertTrue(hl.hadoop_is_dir(resource('ls_test')))
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

    @fails_local_backend()
    def test_hadoop_no_glob_in_bucket(self):
        test_dir_url = os.environ['HAIL_TEST_STORAGE_URI']
        scheme, rest = test_dir_url.split('://')
        bucket, path = rest.split('/', maxsplit=1)
        glob_in_bucket_url = f'{scheme}://glob*{bucket}/{path}'
        try:
            hl.hadoop_ls(glob_in_bucket_url)
        except ValueError as err:
            assert f'glob pattern only allowed in path (e.g. not in bucket): {glob_in_bucket_url}' in err.args[0]
        except FatalError as err:
            assert (
                f"Invalid GCS bucket name 'glob*{bucket}': bucket name must contain only 'a-z0-9_.-' characters."
                in err.args[0]
            )
        else:
            assert False

    def test_hadoop_ls_simple(self):
        with hl.TemporaryDirectory() as dirname:
            with hl.current_backend().fs.open(dirname + '/a', 'w') as fobj:
                fobj.write('hello world')
            dirname = normalize_path(dirname)

            results = hl.hadoop_ls(dirname + '/[a]')
            assert len(results) == 1
            assert results[0]['path'] == dirname + '/a'

    def test_hadoop_ls(self):
        path1 = resource('ls_test/f_50')
        ls1 = hl.hadoop_ls(path1)
        self.assertEqual(len(ls1), 1)
        self.assertEqual(ls1[0]['size_bytes'], 50)
        self.assertEqual(ls1[0]['is_dir'], False)
        self.assertTrue('path' in ls1[0])

        path2 = resource('ls_test')
        ls2 = hl.hadoop_ls(path2)
        self.assertEqual(len(ls2), 3)
        ls2_dict = {x['path'].split("/")[-1]: x for x in ls2}
        self.assertEqual(ls2_dict['f_50']['size_bytes'], 50)
        self.assertEqual(ls2_dict['f_100']['size_bytes'], 100)
        self.assertEqual(ls2_dict['f_100']['is_dir'], False)
        self.assertEqual(ls2_dict['subdir']['is_dir'], True)

        path3 = resource('ls_test/f*')
        ls3 = hl.hadoop_ls(path3)
        assert len(ls3) == 2, ls3

    def test_hadoop_ls_file_that_does_not_exist(self):
        try:
            hl.hadoop_ls('a_file_that_does_not_exist')
        except FileNotFoundError:
            pass
        except FatalError as err:
            assert 'FileNotFoundException: file:/io/a_file_that_does_not_exist' in err.args[0]
        else:
            assert False

    def test_hadoop_glob_heterogenous_structure(self):
        with hl.TemporaryDirectory() as dirname:
            touch(dirname + '/abc/cat')
            touch(dirname + '/abc/dog')
            touch(dirname + '/def/cat')
            touch(dirname + '/def/dog')
            touch(dirname + '/ghi/cat')
            touch(dirname + '/ghi/cat')
            dirname = normalize_path(dirname)

            actual = {x['path'] for x in hl.hadoop_ls(dirname + '/*/cat')}
            expected = {
                dirname + '/abc/cat',
                dirname + '/def/cat',
                dirname + '/ghi/cat',
            }
            assert actual == expected

            actual = {x['path'] for x in hl.hadoop_ls(dirname + '/*/dog')}
            expected = {
                dirname + '/abc/dog',
                dirname + '/def/dog',
            }
            assert actual == expected

    def test_hadoop_ls_glob_no_slash_in_group(self):
        try:
            hl.hadoop_ls(resource('foo[/]bar'))
        except ValueError as err:
            assert 'glob groups must not include forward slashes' in err.args[0]
        except FatalError as err:
            assert 'PatternSyntaxException: error parsing regexp: Unclosed character class at pos 4' in err.args[0]
        else:
            assert False

    def test_hadoop_ls_glob_1(self):
        expected = [normalize_path(resource('ls_test/f_100'))]
        actual = [x['path'] for x in hl.hadoop_ls(resource('l?_t?st/f*00'))]
        assert actual == expected

    def test_hadoop_ls_glob_2(self):
        expected = [normalize_path(resource('ls_test/f_50'))]
        actual = [x['path'] for x in hl.hadoop_ls(resource('ls_test/f_[51]0'))]
        assert actual == expected

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
        self.assertEqual(frozendict({1: 2, 4: 7}), frozendict({1: 2, 4: 7}))
        my_frozen_dict = frozendict({"a": "apple", "h": "hail"})
        self.assertEqual(my_frozen_dict["a"], "apple")

        # Make sure mutating old dict doesn't change frozen counterpart.
        regular_dict = {"a": "b"}
        frozen_counterpart = frozendict(regular_dict)
        regular_dict["a"] = "d"
        self.assertEqual(frozen_counterpart["a"], "b")

        with pytest.raises(TypeError, match="does not support item assignment"):
            my_frozen_dict["a"] = "b"

    def test_json_encoder(self):
        self.assertEqual(json.dumps(frozendict({"foo": "bar"}), cls=hl.utils.JSONEncoder), '{"foo": "bar"}')

        self.assertEqual(json.dumps(Struct(foo="bar"), cls=hl.utils.JSONEncoder), '{"foo": "bar"}')

        self.assertEqual(
            json.dumps(Interval(start=1, end=10), cls=hl.utils.JSONEncoder),
            '{"start": 1, "end": 10, "includes_start": true, "includes_end": false}',
        )

        self.assertEqual(
            json.dumps(hl.Locus(1, 100, "GRCh38"), cls=hl.utils.JSONEncoder),
            '{"contig": "1", "position": 100, "reference_genome": "GRCh38"}',
        )


@pytest.fixture(scope="module")
def glob_tests_directory(init_hail):
    with hl.TemporaryDirectory() as dirname:
        touch(dirname + '/abc/ghi/123')
        touch(dirname + '/abc/ghi/!23')
        touch(dirname + '/abc/ghi/?23')
        touch(dirname + '/abc/ghi/456')
        touch(dirname + '/abc/ghi/78')
        touch(dirname + '/abc/jkl/123')
        touch(dirname + '/abc/jkl/!23')
        touch(dirname + '/abc/jkl/?23')
        touch(dirname + '/abc/jkl/456')
        touch(dirname + '/abc/jkl/78')
        touch(dirname + '/def/ghi/123')
        touch(dirname + '/def/ghi/!23')
        touch(dirname + '/def/ghi/?23')
        touch(dirname + '/def/ghi/456')
        touch(dirname + '/def/ghi/78')
        touch(dirname + '/def/jkl/123')
        touch(dirname + '/def/jkl/!23')
        touch(dirname + '/def/jkl/?23')
        touch(dirname + '/def/jkl/456')
        touch(dirname + '/def/jkl/78')
        yield normalize_path(dirname)


def test_hadoop_ls_folder_glob(glob_tests_directory):
    expected = [glob_tests_directory + '/abc/ghi/123', glob_tests_directory + '/abc/jkl/123']
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/abc/*/123')]
    assert set(actual) == set(expected)


def test_hadoop_ls_prefix_folder_glob_qmarks(glob_tests_directory):
    expected = [glob_tests_directory + '/abc/ghi/78', glob_tests_directory + '/abc/jkl/78']
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/abc/*/??')]
    assert set(actual) == set(expected)


def test_hadoop_ls_two_folder_globs(glob_tests_directory):
    expected = [
        glob_tests_directory + '/abc/ghi/123',
        glob_tests_directory + '/abc/jkl/123',
        glob_tests_directory + '/def/ghi/123',
        glob_tests_directory + '/def/jkl/123',
    ]
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/*/*/123')]
    assert set(actual) == set(expected)


def test_hadoop_ls_two_folder_globs_and_two_qmarks(glob_tests_directory):
    expected = [
        glob_tests_directory + '/abc/ghi/78',
        glob_tests_directory + '/abc/jkl/78',
        glob_tests_directory + '/def/ghi/78',
        glob_tests_directory + '/def/jkl/78',
    ]
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/*/*/??')]
    assert set(actual) == set(expected)


def test_hadoop_ls_one_folder_glob_and_qmarks_in_multiple_components(glob_tests_directory):
    expected = [glob_tests_directory + '/abc/ghi/78', glob_tests_directory + '/def/ghi/78']
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/*/?h?/??')]
    assert set(actual) == set(expected)


def test_hadoop_ls_groups(glob_tests_directory):
    expected = [glob_tests_directory + '/abc/ghi/123']
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/abc/[ghi][ghi]i/123')]
    assert set(actual) == set(expected)


def test_hadoop_ls_size_one_groups(glob_tests_directory):
    expected = []
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/abc/[h][g]i/123')]
    assert set(actual) == set(expected)


def test_hadoop_ls_component_with_only_groups(glob_tests_directory):
    expected = [
        glob_tests_directory + '/abc/ghi/123',
        glob_tests_directory + '/abc/ghi/!23',
        glob_tests_directory + '/abc/ghi/?23',
        glob_tests_directory + '/abc/ghi/456',
        glob_tests_directory + '/abc/ghi/78',
    ]
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/abc/[g][h][i]/*')]
    assert set(actual) == set(expected)


def test_hadoop_ls_negated_group(glob_tests_directory):
    expected = [glob_tests_directory + '/abc/ghi/!23', glob_tests_directory + '/abc/ghi/?23']
    actual = [x['path'] for x in hl.hadoop_ls(glob_tests_directory + '/abc/ghi/[!1]23')]
    assert set(actual) == set(expected)


def test_struct_rich_comparison():
    """Asserts comparisons between structs and struct expressions are symmetric"""
    struct = hl.Struct(locus=hl.Locus(contig=10, position=60515, reference_genome='GRCh37'), alleles=['C', 'T'])

    expr = hl.struct(locus=hl.locus(contig='10', pos=60515, reference_genome='GRCh37'), alleles=['C', 'T'])

    assert hl.eval(struct == expr) and hl.eval(expr == struct)
    assert hl.eval(struct >= expr) and hl.eval(expr >= struct)
    assert hl.eval(struct <= expr) and hl.eval(expr <= struct)
    assert not (hl.eval(struct < expr) or hl.eval(expr < struct))
    assert not (hl.eval(struct > expr) or hl.eval(expr > struct))
