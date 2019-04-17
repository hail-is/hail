import unittest

import pandas as pd
import pyspark.sql
import pytest

import hail as hl
import hail.expr.aggregators as agg
from hail.utils import new_temp_file
from hail.utils.java import Env
from ..helpers import *
from test.hail.matrixtable.test_file_formats import create_all_values_datasets

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_annotate(self):
        schema = hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tint32, d=hl.tint32, e=hl.tstr, f=hl.tarray(hl.tint32))

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)

        self.assertTrue(kt.annotate()._same(kt))

        result1 = convert_struct_to_dict(kt.annotate(foo=kt.a + 1,
                                                     foo2=kt.a).take(1)[0])

        self.assertDictEqual(result1, {'a': 4,
                                       'b': 1,
                                       'c': 3,
                                       'd': 5,
                                       'e': "hello",
                                       'f': [1, 2, 3],
                                       'foo': 5,
                                       'foo2': 4})

        result3 = convert_struct_to_dict(kt.annotate(
            x1=kt.f.map(lambda x: x * 2),
            x2=kt.f.map(lambda x: [x, x + 1]).flatmap(lambda x: x),
            x3=hl.min(kt.f),
            x4=hl.max(kt.f),
            x5=hl.sum(kt.f),
            x6=hl.product(kt.f),
            x7=kt.f.length(),
            x8=kt.f.filter(lambda x: x == 3),
            x9=kt.f[1:],
            x10=kt.f[:],
            x11=kt.f[1:2],
            x12=kt.f.map(lambda x: [x, x + 1]),
            x13=kt.f.map(lambda x: [[x, x + 1], [x + 2]]).flatmap(lambda x: x),
            x14=hl.cond(kt.a < kt.b, kt.c, hl.null(hl.tint32)),
            x15={1, 2, 3}
        ).take(1)[0])

        self.assertDictEqual(result3, {'a': 4,
                                       'b': 1,
                                       'c': 3,
                                       'd': 5,
                                       'e': "hello",
                                       'f': [1, 2, 3],
                                       'x1': [2, 4, 6], 'x2': [1, 2, 2, 3, 3, 4],
                                       'x3': 1, 'x4': 3, 'x5': 6, 'x6': 6, 'x7': 3, 'x8': [3],
                                       'x9': [2, 3], 'x10': [1, 2, 3], 'x11': [2],
                                       'x12': [[1, 2], [2, 3], [3, 4]],
                                       'x13': [[1, 2], [3], [2, 3], [4], [3, 4], [5]],
                                       'x14': None, 'x15': set([1, 2, 3])})
        kt.annotate(
            x1=kt.a + 5,
            x2=5 + kt.a,
            x3=kt.a + kt.b,
            x4=kt.a - 5,
            x5=5 - kt.a,
            x6=kt.a - kt.b,
            x7=kt.a * 5,
            x8=5 * kt.a,
            x9=kt.a * kt.b,
            x10=kt.a / 5,
            x11=5 / kt.a,
            x12=kt.a / kt.b,
            x13=-kt.a,
            x14=+kt.a,
            x15=kt.a == kt.b,
            x16=kt.a == 5,
            x17=5 == kt.a,
            x18=kt.a != kt.b,
            x19=kt.a != 5,
            x20=5 != kt.a,
            x21=kt.a > kt.b,
            x22=kt.a > 5,
            x23=5 > kt.a,
            x24=kt.a >= kt.b,
            x25=kt.a >= 5,
            x26=5 >= kt.a,
            x27=kt.a < kt.b,
            x28=kt.a < 5,
            x29=5 < kt.a,
            x30=kt.a <= kt.b,
            x31=kt.a <= 5,
            x32=5 <= kt.a,
            x33=(kt.a == 0) & (kt.b == 5),
            x34=(kt.a == 0) | (kt.b == 5),
            x35=False,
            x36=True
        )

    def test_aggregate1(self):
        schema = hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tint32, d=hl.tint32, e=hl.tstr, f=hl.tarray(hl.tint32))

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)
        results = kt.aggregate(hl.Struct(q1=agg.sum(kt.b),
                                         q2=agg.count(),
                                         q3=agg.collect(kt.e),
                                         q4=agg.filter((kt.d >= 5) | (kt.a == 0), agg.collect(kt.e)),
                                         q5=agg.explode(lambda elt: agg.mean(elt), kt.f)))

        self.assertEqual(results.q1, 8)
        self.assertEqual(results.q2, 3)
        self.assertEqual(set(results.q3), {"hello", "cat", "dog"})
        self.assertEqual(set(results.q4), {"hello", "cat"})
        self.assertAlmostEqual(results.q5, 4)

    def test_aggregate2(self):
        schema = hl.tstruct(status=hl.tint32, GT=hl.tcall, qPheno=hl.tint32)

        rows = [{'status': 0, 'GT': hl.Call([0, 0]), 'qPheno': 3},
                {'status': 0, 'GT': hl.Call([0, 1]), 'qPheno': 13}]

        kt = hl.Table.parallelize(rows, schema)

        result = convert_struct_to_dict(
            kt.group_by(status=kt.status)
                .aggregate(
                x1=agg.collect(kt.qPheno * 2),
                x2=agg.explode(lambda elt: agg.collect(elt), [kt.qPheno, kt.qPheno + 1]),
                x3=agg.min(kt.qPheno),
                x4=agg.max(kt.qPheno),
                x5=agg.sum(kt.qPheno),
                x6=agg.product(hl.int64(kt.qPheno)),
                x7=agg.count(),
                x8=agg.count_where(kt.qPheno == 3),
                x9=agg.fraction(kt.qPheno == 1),
                x10=agg.stats(hl.float64(kt.qPheno)),
                x11=agg.hardy_weinberg_test(kt.GT),
                x13=agg.inbreeding(kt.GT, 0.1),
                x14=agg.call_stats(kt.GT, ["A", "T"]),
                x15=agg.collect(hl.Struct(a=5, b="foo", c=hl.Struct(banana='apple')))[0],
                x16=agg.collect(hl.Struct(a=5, b="foo", c=hl.Struct(banana='apple')).c.banana)[0],
                x17=agg.explode(lambda elt: agg.collect(elt), hl.null(hl.tarray(hl.tint32))),
                x18=agg.explode(lambda elt: agg.collect(elt), hl.null(hl.tset(hl.tint32))),
                x19=agg.take(kt.GT, 1, ordering=-kt.qPheno)
            ).take(1)[0])

        expected = {u'status': 0,
                    u'x13': {u'n_called': 2, u'expected_homs': 1.64, u'f_stat': -1.777777777777777,
                             u'observed_homs': 1},
                    u'x14': {u'AC': [3, 1], u'AF': [0.75, 0.25], u'AN': 4, u'homozygote_count': [1, 0]},
                    u'x15': {u'a': 5, u'c': {u'banana': u'apple'}, u'b': u'foo'},
                    u'x10': {u'min': 3.0, u'max': 13.0, u'sum': 16.0, u'stdev': 5.0, u'n': 2, u'mean': 8.0},
                    u'x8': 1, u'x9': 0.0, u'x16': u'apple',
                    u'x11': {u'het_freq_hwe': 0.5, u'p_value': 0.5},
                    u'x2': [3, 4, 13, 14], u'x3': 3, u'x1': [6, 26], u'x6': 39, u'x7': 2, u'x4': 13, u'x5': 16,
                    u'x17': [],
                    u'x18': [],
                    u'x19': [hl.Call([0, 1])]}

        self.maxDiff = None

        self.assertDictEqual(result, expected)

    def test_aggregate_ir(self):
        kt = hl.utils.range_table(10).annotate_globals(g1=5)
        r = kt.aggregate(hl.struct(x=agg.sum(kt.idx) + kt.g1,
                                   y=agg.filter(kt.idx % 2 != 0, agg.sum(kt.idx + 2)) + kt.g1,
                                   z=agg.sum(kt.g1 + kt.idx) + kt.g1))
        self.assertEqual(convert_struct_to_dict(r), {u'x': 50, u'y': 40, u'z': 100})

        r = kt.aggregate(5)
        self.assertEqual(r, 5)

        r = kt.aggregate(hl.null(hl.tint32))
        self.assertEqual(r, None)

        r = kt.aggregate(agg.filter(kt.idx % 2 != 0, agg.sum(kt.idx + 2)) + kt.g1)
        self.assertEqual(r, 40)

    def test_to_matrix_table(self):
        N, M = 50, 50
        mt = hl.utils.range_matrix_table(N, M)
        mt = mt.key_cols_by(s = 'Col' + hl.str(M - mt.col_idx))
        mt = mt.annotate_cols(c1 = hl.rand_bool(0.5))
        mt = mt.annotate_rows(r1 = hl.rand_bool(0.5))
        mt = mt.annotate_entries(e1 = hl.rand_bool(0.5))

        re_mt = mt.entries().to_matrix_table(['row_idx'], ['s'], ['r1'], ['col_idx', 'c1'])
        new_col_order = re_mt.col_idx.collect()
        mapping = [t[1] for t in sorted([(old, new) for new, old in enumerate(new_col_order)])]

        assert re_mt.choose_cols(mapping).drop('col_idx')._same(mt.drop('col_idx'))

    def test_to_matrix_table_row_major(self):
        t = hl.utils.range_table(10)
        t = t.annotate(foo=t.idx, bar=2 * t.idx, baz=3 * t.idx)
        mt = t.to_matrix_table_row_major(['bar', 'baz'], 'entry', 'col')
        round_trip = mt.localize_entries('entries', 'cols')
        round_trip = round_trip.transmute(**{col.col: round_trip.entries[i].entry for i, col in enumerate(hl.eval(round_trip.cols))})
        round_trip = round_trip.drop(round_trip.cols)

        self.assertTrue(t._same(round_trip))

        t = hl.utils.range_table(10)
        t = t.annotate(foo=t.idx, bar=hl.struct(val=2 * t.idx), baz=hl.struct(val=3 * t.idx))
        mt = t.to_matrix_table_row_major(['bar', 'baz'])
        round_trip = mt.localize_entries('entries', 'cols')
        round_trip = round_trip.transmute(**{col.col: round_trip.entries[i] for i, col in enumerate(hl.eval(round_trip.cols))})
        round_trip = round_trip.drop(round_trip.cols)

        self.assertTrue(t._same(round_trip))

        t = t.annotate(**{'a': 1, 'b': 2, 'c': 'v', 'd': hl.struct(e=2)})
        self.assertRaises(ValueError, lambda: t.to_matrix_table_row_major(['a', 'b']))
        self.assertRaises(ValueError, lambda: t.to_matrix_table_row_major(['a', 'd']))
        self.assertRaises(ValueError, lambda: t.to_matrix_table_row_major(['d'], entry_field_name='c'))
        self.assertRaises(ValueError, lambda: t.to_matrix_table_row_major([]))

    def test_group_by_field_lifetimes(self):
        ht = hl.utils.range_table(3)
        ht2 = (ht.group_by(idx='100')
               .aggregate(x=hl.agg.collect_as_set(ht.idx + 5)))
        assert (ht2.all(ht2.x == hl.set({5, 6, 7})))

    def test_group_aggregate_by_key(self):
        ht = hl.utils.range_table(100, n_partitions=10)

        r1 = ht.group_by(k = ht.idx % 5)._set_buffer_size(3).aggregate(n = hl.agg.count())
        r2 = ht.group_by(k = ht.idx // 20)._set_buffer_size(3).aggregate(n = hl.agg.count())
        assert r1.all(r1.n == 20)
        assert r2.all(r2.n == 20)

    def test_aggregate_by_key_partitioning(self):
        ht1 = hl.Table.parallelize([
            {'k': 'foo', 'b': 1},
            {'k': 'bar', 'b': 2},
            {'k': 'bar', 'b': 2}],
            hl.tstruct(k=hl.tstr, b=hl.tint32),
            key='k')
        self.assertEqual(
            set(ht1.group_by('k').aggregate(mean_b = hl.agg.mean(ht1.b)).collect()),
            {hl.Struct(k='foo', mean_b=1.0), hl.Struct(k='bar', mean_b=2.0)})

    def test_filter(self):
        schema = hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tint32, d=hl.tint32, e=hl.tstr, f=hl.tarray(hl.tint32))

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)

        self.assertEqual(kt.filter(kt.a == 4).count(), 2)
        self.assertEqual(kt.filter((kt.d == -1) | (kt.c == 20) | (kt.e == "hello")).count(), 3)
        self.assertEqual(kt.filter((kt.c != 20) & (kt.a == 4)).count(), 1)
        self.assertEqual(kt.filter(True).count(), 3)

    def test_filter_missing(self):
        ht = hl.utils.range_table(1, 1)

        self.assertEqual(ht.filter(hl.null(hl.tbool)).count(), 0)

    def test_transmute(self):
        schema = hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tint32, d=hl.tint32, e=hl.tstr, f=hl.tarray(hl.tint32),
                            g=hl.tstruct(x=hl.tbool, y=hl.tint32))

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3], 'g': {'x': True, 'y': 2}},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': [], 'g': {'x': True, 'y': 2}},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7], 'g': None}]
        df = hl.Table.parallelize(rows, schema)

        df = df.transmute(h=df.a + df.b + df.c + df.g.y)
        r = df.select('h').collect()

        self.assertEqual(list(df.row), ['d', 'e', 'f', 'h'])
        self.assertEqual(r, [hl.Struct(h=x) for x in [10, 20, None]])

    def test_transmute_globals(self):
        ht = hl.utils.range_table(1).annotate_globals(a=5, b=10)
        self.assertEqual(ht.transmute_globals(c=ht.a + 5).globals.dtype, hl.tstruct(b=hl.tint, c=hl.tint))

    def test_transmute_key(self):
        ht = hl.utils.range_table(10)
        self.assertEqual(ht.transmute(y = ht.idx + 2).row.dtype, hl.dtype('struct{idx: int32, y: int32}'))
        ht = ht.key_by()
        self.assertEqual(ht.transmute(y = ht.idx + 2).row.dtype, hl.dtype('struct{y: int32}'))

    def test_select(self):
        schema = hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tint32, d=hl.tint32, e=hl.tstr, f=hl.tarray(hl.tint32),
                            g=hl.tstruct(x=hl.tbool, y=hl.tint32))

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3], 'g': {'x': True, 'y': 2}},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': [], 'g': {'x': True, 'y': 2}},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7], 'g': None}]

        kt = hl.Table.parallelize(rows, schema)

        t1 = kt.select(kt.a, kt.e)
        self.assertEqual(list(t1.row), ['a', 'e'])
        self.assertEqual(list(t1.key), [])

        t2 = kt.key_by('e')
        t2 = t2.select(t2.a)
        self.assertEqual(list(t2.row), ['e', 'a'])
        self.assertEqual(list(t2.key), ['e'])

        self.assertEqual(list(kt.select(kt.a, foo=kt.a + kt.b - kt.c - kt.d).row), ['a', 'foo'])
        self.assertEqual(list(kt.select(kt.a, foo=kt.a + kt.b - kt.c - kt.d, **kt.g).row), ['a', 'foo', 'x', 'y'])

        # select no fields
        s = kt.select()
        self.assertEqual(list(s.row), [])
        self.assertEqual(list(s.key), [])

    def test_errors(self):
        schema = hl.tstruct(status=hl.tint32, gt=hl.tcall, qPheno=hl.tint32)

        rows = [{'status': 0, 'gt': hl.Call([0, 0]), 'qPheno': 3},
                {'status': 0, 'gt': hl.Call([0, 1]), 'qPheno': 13},
                {'status': 1, 'gt': hl.Call([0, 1]), 'qPheno': 20}]

        kt = hl.Table.parallelize(rows, schema)

        def f():
            kt.a = 5

        self.assertRaises(NotImplementedError, f)

    def test_semi_anti_join(self):
        ht = hl.utils.range_table(10)
        ht2 = ht.filter(ht.idx < 3)

        assert ht.semi_join(ht2).count() == 3
        assert ht.anti_join(ht2).count() == 7

    def test_indirected_joins(self):
        kt = hl.utils.range_table(1)
        kt = kt.annotate(a='foo')

        kt1 = hl.utils.range_table(1)
        kt1 = kt1.annotate(a='foo', b='bar').key_by('a')

        kt2 = hl.utils.range_table(1)
        kt2 = kt2.annotate(b='bar', c='baz').key_by('b')

        kt3 = hl.utils.range_table(1)
        kt3 = kt3.annotate(c='baz', d='qux').key_by('c')

        kt4 = hl.utils.range_table(1)
        kt4 = kt4.annotate(d='qux', e='quam').key_by('d')

        assert kt.aggregate(agg.collect(kt4[kt3[kt2[kt1[kt.a].b].c].d].e)) == ['quam']

    def test_table_matrix_join_combinations(self):
        m = hl.import_vcf(resource('sample.vcf'))
        vkt = m.rows()
        assert vkt.filter(vkt.qual != m.index_rows(vkt.key).qual).count() == 0

        assert m.filter_rows(m.qual != vkt.index(m.row_key).qual).count_rows() == 0

        assert m.filter_rows(m.qual != m.index_rows(m.row_key).qual).count_rows() == 0

        kt5 = hl.utils.range_table(1).annotate(s='C1589').key_by('s')
        n_C1589 = m.filter_cols(m.s[:5] == 'C1589').count_cols()
        assert n_C1589 > 1

        m2 = m.annotate_cols(foo=m.s[:5])
        assert m2.filter_cols(hl.is_defined(kt5[m2.foo].idx)).count_cols() == n_C1589

    def test_index_globals(self):
        ht = hl.utils.range_table(1).annotate_globals(foo=5)
        assert hl.eval(ht.index_globals().foo) == 5

    def test_interval_join(self):
        left = hl.utils.range_table(50, n_partitions=10)
        intervals = hl.utils.range_table(4)
        intervals = intervals.key_by(interval=hl.interval(intervals.idx * 10, intervals.idx * 10 + 5))
        left = left.annotate(interval_matches=intervals.index(left.key))
        self.assertTrue(left.all(hl.case()
                                 .when(left.idx % 10 < 5, left.interval_matches.idx == left.idx // 10)
                                 .default(hl.is_missing(left.interval_matches))))

    def test_join_with_empty(self):
        kt = hl.utils.range_table(10)
        kt2 = kt.head(0)
        kt.annotate(foo=hl.is_defined(kt2[kt.idx]))

    def test_join_with_key(self):
        ht = hl.utils.range_table(10)
        ht1 = ht.annotate(foo=5)
        self.assertTrue(ht.all(ht1[ht.key].foo == 5))

    def test_multiple_entry_joins(self):
        mt = hl.utils.range_matrix_table(4, 4)
        mt2 = hl.utils.range_matrix_table(4, 4)
        mt2 = mt2.annotate_entries(x=mt2.row_idx + mt2.col_idx)
        mt.select_entries(a=mt2[mt.row_idx, mt.col_idx].x,
                          b=mt2[mt.row_idx, mt.col_idx].x)

    def test_multi_way_zip_join(self):
        d1 = [{"id": 0, "name": "a", "data": 0.0},
              {"id": 1, "name": "b", "data": 3.14},
              {"id": 2, "name": "c", "data": 2.78}]
        d2 = [{"id": 0, "name": "d", "data": 1.1},
              {"id": 0, "name": "x", "data": 2.2},
              {"id": 2, "name": "v", "data": 7.89}]
        d3 = [{"id": 1, "name": "f", "data":  9.99},
              {"id": 2, "name": "g", "data": -1.0},
              {"id": 3, "name": "z", "data":  0.01}]
        s = hl.tstruct(id=hl.tint32, name=hl.tstr, data=hl.tfloat64)
        ts = [hl.Table.parallelize(r, schema=s, key='id') for r in [d1, d2, d3]]
        joined = hl.Table._multi_way_zip_join(ts, '__data', '__globals').drop('__globals')
        dexpected = [{"id": 0, "__data": [{"name": "a", "data": 0.0},
                                          {"name": "d", "data": 1.1},
                                          None]},
                     {"id": 0, "__data": [None,
                                          {"name": "x", "data": 2.2},
                                          None]},
                     {"id": 1, "__data": [{"name": "b", "data": 3.14},
                                          None,
                                          {"name": "f", "data":  9.99}]},
                     {"id": 2, "__data": [{"name": "c", "data": 2.78},
                                          {"name": "v", "data": 7.89},
                                          {"name": "g", "data": -1.0}]},
                     {"id": 3, "__data": [None,
                                          None,
                                          {"name": "z", "data":  0.01}]}]
        expected = hl.Table.parallelize(
            dexpected,
            schema=hl.tstruct(id=hl.tint32, __data=hl.tarray(hl.tstruct(name=hl.tstr, data=hl.tfloat64))),
            key='id')
        self.assertTrue(expected._same(joined))

        expected2 = expected.transmute(data=expected['__data'])
        joined_same_name = hl.Table._multi_way_zip_join(ts, 'data', 'globals').drop('globals')
        self.assertTrue(expected2._same(joined_same_name))

        joined_nothing = hl.Table._multi_way_zip_join(ts, 'data', 'globals').drop('data', 'globals')
        self.assertEqual(joined_nothing._force_count(), 5)

    def test_multi_way_zip_join_globals(self):
        t1 = hl.utils.range_table(1).annotate_globals(x=hl.null(hl.tint32))
        t2 = hl.utils.range_table(1).annotate_globals(x=5)
        t3 = hl.utils.range_table(1).annotate_globals(x=0)
        expected = hl.struct(__globals=hl.array([
            hl.struct(x=hl.null(hl.tint32)),
            hl.struct(x=5),
            hl.struct(x=0)]))
        joined = hl.Table._multi_way_zip_join([t1, t2, t3], '__data', '__globals')
        self.assertEqual(hl.eval(joined.globals), hl.eval(expected))

    def test_multi_way_zip_join_key_downcast(self):
        mt = hl.import_vcf(resource('sample.vcf.bgz'))
        mt = mt.key_rows_by('locus')
        ht = mt.rows()
        j = hl.Table._multi_way_zip_join([ht, ht], 'd', 'g')
        j._force_count()

    def test_index_maintains_count(self):
        t1 = hl.Table.parallelize([
            {'a': 'foo', 'b': 1},
            {'a': 'bar', 'b': 2},
            {'a': 'bar', 'b': 2}],
            hl.tstruct(a=hl.tstr, b=hl.tint32),
            key='a')
        t2 = hl.Table.parallelize([
            {'t': 'foo', 'x': 3.14},
            {'t': 'bar', 'x': 2.78},
            {'t': 'bar', 'x': -1},
            {'t': 'quam', 'x': 0}],
            hl.tstruct(t=hl.tstr, x=hl.tfloat64),
            key='t')

        j = t1.annotate(f=t2[t1.a].x)
        self.assertEqual(j.count(), t1.count())

    def test_aggregation_with_no_aggregators(self):
        ht = hl.utils.range_table(3)
        self.assertEqual(ht.group_by(ht.idx).aggregate().count(), 3)

    def test_drop(self):
        kt = hl.utils.range_table(10)
        kt = kt.annotate(sq=kt.idx ** 2, foo='foo', bar='bar').key_by('foo')

        ktd = kt.drop('idx')
        self.assertEqual(set(ktd.row), {'foo', 'sq', 'bar'})
        ktd = ktd.key_by().drop('foo')
        self.assertEqual(list(ktd.key), [])

        self.assertEqual(set(kt.drop(kt['idx']).row), {'foo', 'sq', 'bar'})

        d = kt.key_by().drop(*list(kt.row))
        self.assertEqual(list(d.row), [])
        self.assertEqual(list(d.key), [])

        self.assertTrue(kt.drop()._same(kt))
        self.assertTrue(kt.drop(*list(kt.row_value))._same(kt.select()))

    def test_weird_names(self):
        df = hl.utils.range_table(10)
        exprs = {'a': 5, '   a    ': 5, r'\%!^!@#&#&$%#$%': [5], '$': 5, 'ß': 5}

        df.annotate_globals(**exprs)
        df.select_globals(**exprs)

        df.annotate(**exprs)
        df.select(**exprs)
        df = df.transmute(**exprs)

        df.explode('\%!^!@#&#&$%#$%')
        df.explode(df['\%!^!@#&#&$%#$%'])

        df.drop('\%!^!@#&#&$%#$%')
        df.drop(df['\%!^!@#&#&$%#$%'])
        df.group_by(**{'*``81': df.a}).aggregate(c=agg.count())

    def test_sample(self):
        kt = hl.utils.range_table(10)
        kt_small = kt.sample(0.01)
        self.assertTrue(kt_small.count() < kt.count())

    @skip_unless_spark_backend()
    def test_from_spark_works(self):
        sql_context = Env.sql_context()
        df = sql_context.createDataFrame([pyspark.sql.Row(x=5, y='foo')])
        t = hl.Table.from_spark(df)
        rows = t.collect()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].x, 5)
        self.assertEqual(rows[0].y, 'foo')

    @skip_unless_spark_backend()
    def test_from_pandas_works(self):
        d = {'a': [1, 2], 'b': ['foo', 'bar']}
        df = pd.DataFrame(data=d)
        t = hl.Table.from_pandas(df, key='a')

        d2 = [hl.struct(a=hl.int64(1), b='foo'), hl.struct(a=hl.int64(2), b='bar')]
        t2 = hl.Table.parallelize(d2, key='a')

        self.assertTrue(t._same(t2))

    def test_rename(self):
        kt = hl.utils.range_table(10)
        kt = kt.annotate_globals(foo=5, fi=3)
        kt = kt.annotate(bar=45, baz=32).key_by('bar')
        renamed = kt.rename({'foo': 'foo2', 'bar': 'bar2'})
        renamed.count()

        self.assertEqual(list(renamed.key), ['bar2'])
        self.assertEqual(renamed['foo2'].dtype, kt['foo'].dtype)
        self.assertEqual(renamed['fi'].dtype, kt['fi'].dtype)
        self.assertEqual(renamed['bar2'].dtype, kt['bar'].dtype)
        self.assertEqual(renamed['baz'].dtype, kt['baz'].dtype)

        self.assertEqual(renamed['bar2']._indices, renamed._row_indices)

        self.assertFalse('foo' in renamed._fields)
        self.assertFalse('bar' in renamed._fields)

        with self.assertRaises(ValueError):
            kt.rename({'foo': 'bar'})

        with self.assertRaises(ValueError):
            kt.rename({'bar': 'a', 'baz': 'a'})

        with self.assertRaises(LookupError):
            kt.rename({'hello': 'a'})

    def test_distinct(self):
        t1 = hl.Table.parallelize([
            {'a': 'foo', 'b': 1},
            {'a': 'bar', 'b': 2},
            {'a': 'bar', 'b': 2},
            {'a': 'bar', 'b': 3},
            {'a': 'bar', 'b': 3},
            {'a': 'baz', 'b': 2},
            {'a': 'baz', 'b': 0},
            {'a': 'baz', 'b': 0},
            {'a': 'foo', 'b': 0},
            {'a': '1', 'b': 0},
            {'a': '2', 'b': 0},
            {'a': '3', 'b': 0}],
            hl.tstruct(a=hl.tstr, b=hl.tint32),
            key='a',
            n_partitions=4)

        dist = t1.distinct().collect_by_key()
        self.assertTrue(dist.all(hl.len(dist.values) == 1))
        self.assertEqual(dist.count(), len(t1.aggregate(hl.agg.collect_as_set(t1.a))))

    def test_group_by_key(self):
        t1 = hl.Table.parallelize([
            {'a': 'foo', 'b': 1},
            {'a': 'bar', 'b': 2},
            {'a': 'bar', 'b': 2},
            {'a': 'bar', 'b': 3},
            {'a': 'bar', 'b': 3},
            {'a': 'baz', 'b': 2},
            {'a': 'baz', 'b': 0},
            {'a': 'baz', 'b': 0},
            {'a': 'foo', 'b': 0},
            {'a': '1', 'b': 0},
            {'a': '2', 'b': 0},
            {'a': '3', 'b': 0}],
            hl.tstruct(a=hl.tstr, b=hl.tint32),
            key='a',
            n_partitions=4)
        g = t1.collect_by_key().explode('values')
        g = g.transmute(**g.values)
        self.assertTrue(g._same(t1))

    def test_str_annotation_regression(self):
        t = hl.Table.parallelize([{'alleles': ['A', 'T']}],
                                 hl.tstruct(alleles=hl.tarray(hl.tstr)))
        t = t.annotate(ref=t.alleles[0])
        t._force_count()

    def test_range_table(self):
        t = hl.utils.range_table(26, n_partitions=5)
        self.assertEqual(t.globals.dtype, hl.tstruct())
        self.assertEqual(t.row.dtype, hl.tstruct(idx=hl.tint32))
        self.assertEqual(t.row_value.dtype, hl.tstruct())
        self.assertEqual(list(t.key), ['idx'])

        self.assertEqual([r.idx for r in t.collect()], list(range(26)))

    def test_issue_3654(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(x=[1, 2])
        self.assertEqual(ht.aggregate(hl.agg.array_sum(ht.x) / [2, 2]), [5.0, 10.0])

    def test_explode_key_error(self):
        t = hl.utils.range_table(1)
        t = t.key_by(a=['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            t.explode('a')

    def test_explode_on_set(self):
        t = hl.utils.range_table(1)
        t = t.annotate(a=hl.set(['a', 'b', 'c']))
        t = t.explode('a')
        self.assertEqual(set(t.collect()),
                         hl.eval(hl.set([hl.struct(idx=0, a='a'),
                                         hl.struct(idx=0, a='b'),
                                         hl.struct(idx=0, a='c')])))

    def test_explode_nested(self):
        t = hl.utils.range_table(2)
        t = t.annotate(foo=hl.case().when(t.idx == 1, hl.struct(bar=[1, 2, 3])).or_missing())
        t = t.explode(t.foo.bar)
        assert t.foo.bar.collect() == [1, 2, 3]

    def test_value_error(self):
        t = hl.utils.range_table(2)
        t = t.annotate(foo=hl.struct(bar=[1, 2, 3]))
        with pytest.raises(ValueError):
            t.explode(t.foo.bar, name='baz')

    def test_export(self):
        t = hl.utils.range_table(1).annotate(foo=3)
        tmp_file = new_temp_file()
        t.export(tmp_file)

        with hl.hadoop_open(tmp_file, 'r') as f_in:
            assert f_in.read() == 'idx\tfoo\n0\t3\n'

    def test_export_delim(self):
        t = hl.utils.range_table(1).annotate(foo = 3)
        tmp_file = new_temp_file()
        t.export(tmp_file, delimiter=',')

        with hl.hadoop_open(tmp_file, 'r') as f_in:
            assert f_in.read() == 'idx,foo\n0,3\n'

    def test_write_stage_locally(self):
        t = hl.utils.range_table(5)
        f = new_temp_file(suffix='ht')
        t.write(f, stage_locally=True)
        t2 = hl.read_table(f)
        self.assertTrue(t._same(t2))

    def test_min_partitions(self):
        assert hl.import_table(resource('variantAnnotations.tsv'), min_partitions=50).n_partitions() == 50

    def test_read_back_same_as_exported(self):
        t, _ = create_all_values_datasets()
        tmp_file = new_temp_file(prefix="test", suffix=".tsv")
        t.export(tmp_file)
        t_read_back = hl.import_table(tmp_file, types=dict(t.row.dtype)).key_by('idx')
        self.assertTrue(t.select_globals()._same(t_read_back, tolerance=1e-4, absolute=True))

    def test_order_by_parsing(self):
        hl.utils.range_table(1).annotate(**{'a b c' : 5}).order_by('a b c')._force_count()

    def test_filter_partitions(self):
        ht = hl.utils.range_table(23, n_partitions=8)
        self.assertEqual(ht.n_partitions(), 8)
        self.assertEqual(ht._filter_partitions([0, 1, 4]).n_partitions(), 3)
        self.assertEqual(ht._filter_partitions(range(3)).n_partitions(), 3)
        self.assertEqual(ht._filter_partitions([4, 5, 7], keep=False).n_partitions(), 5)
        self.assertTrue(
            ht._same(hl.Table.union(
                ht._filter_partitions([0, 3, 7]),
                ht._filter_partitions([0, 3, 7], keep=False))))
        # ht = [0, 1, 2], [3, 4, 5], ..., [21, 22]
        self.assertEqual(
            ht._filter_partitions([0, 7]).idx.collect(),
            [0, 1, 2, 21, 22])

    def test_localize_entries(self):
        ref_schema = hl.tstruct(row_idx=hl.tint32,
                                __entries=hl.tarray(hl.tstruct(v=hl.tint32)))
        ref_data = [{'row_idx': i, '__entries': [{'v': i+j} for j in range(6)]}
                    for i in range(8)]
        ref_tab = hl.Table.parallelize(ref_data, ref_schema).key_by('row_idx')
        ref_tab = ref_tab.select_globals(__cols=[hl.struct(col_idx=i) for i in range(6)])
        mt = hl.utils.range_matrix_table(8, 6)
        mt = mt.annotate_entries(v=mt.row_idx+mt.col_idx)
        t = mt._localize_entries('__entries', '__cols')
        self.assertTrue(t._same(ref_tab))

    def test_localize_self_join(self):
        ref_schema = hl.tstruct(row_idx=hl.tint32,
                                __entries=hl.tarray(hl.tstruct(v=hl.tint32)))
        ref_data = [{'row_idx': i, '__entries': [{'v': i+j} for j in range(6)]}
                    for i in range(8)]
        ref_tab = hl.Table.parallelize(ref_data, ref_schema).key_by('row_idx')
        ref_tab = ref_tab.join(ref_tab, how='outer')
        mt = hl.utils.range_matrix_table(8, 6)
        mt = mt.annotate_entries(v=mt.row_idx+mt.col_idx)
        t = mt._localize_entries('__entries', '__cols').drop('__cols')
        t = t.join(t, how='outer')
        self.assertTrue(t._same(ref_tab))

    def test_union(self):
        t1 = hl.utils.range_table(5)

        t2 = hl.utils.range_table(5)
        t2 = t2.key_by(idx = t2.idx + 5)

        t3 = hl.utils.range_table(5)
        t3 = t3.key_by(idx = t3.idx + 10)

        self.assertTrue(t1.union(t2, t3)._same(hl.utils.range_table(15)))
        self.assertTrue(t1.key_by().union(t2.key_by(), t3.key_by())
                        ._same(hl.utils.range_table(15).key_by()))

    def test_union_unify(self):
        t1 = hl.utils.range_table(2)
        t2 = t1.annotate(x=hl.int32(1), y='A')
        t3 = t1.annotate(z=(1, 2, 3), x=hl.float64(1.5))
        t4 = t1.key_by(idx=t1.idx + 10)

        u = t1.union(t2, t3, t4, unify=True)

        assert u.x.dtype == hl.tfloat64
        assert list(u.row) == ['idx', 'x', 'y', 'z']

        assert u.collect() == [
            hl.utils.Struct(idx=0, x=None, y=None, z=None),
            hl.utils.Struct(idx=0, x=1.0, y='A', z=None),
            hl.utils.Struct(idx=0, x=1.5, y=None, z=(1, 2, 3)),
            hl.utils.Struct(idx=1, x=None, y=None, z=None),
            hl.utils.Struct(idx=1, x=1.0, y='A', z=None),
            hl.utils.Struct(idx=1, x=1.5, y=None, z=(1, 2, 3)),
            hl.utils.Struct(idx=10, x=None, y=None, z=None),
            hl.utils.Struct(idx=11, x=None, y=None, z=None),
        ]

    def test_table_head_returns_right_number(self):
        rt = hl.utils.range_table(10, 11)
        par = hl.Table.parallelize([hl.Struct(x=x) for x in range(10)], schema='struct{x: int32}', n_partitions=11)

        # test TableRange and TableParallelize rewrite rules
        tables = [rt, par, rt.cache()]
        for table in tables:
            self.assertEqual(table.head(10).count(), 10)
            self.assertEqual(table.head(10)._force_count(), 10)
            self.assertEqual(table.head(9).count(), 9)
            self.assertEqual(table.head(9)._force_count(), 9)
            self.assertEqual(table.head(11).count(), 10)
            self.assertEqual(table.head(11)._force_count(), 10)
            self.assertEqual(table.head(0).count(), 0)
            self.assertEqual(table.head(0)._force_count(), 0)

    def test_table_order_by_head_rewrite(self):
        rt = hl.utils.range_table(10, 2)
        rt = rt.annotate(x = 10 - rt.idx)
        expected = list(range(10))[::-1]
        self.assertEqual(rt.order_by('x').idx.take(10), expected)
        self.assertEqual(rt.order_by('x').idx.collect(), expected)

    def test_null_joins(self):
        tr = hl.utils.range_table(7, 1)
        table1 = tr.key_by(new_key=hl.cond((tr.idx == 3) | (tr.idx == 5),
                                           hl.null(hl.tint32), tr.idx),
                           key2=1)
        table1 = table1.select(idx1=table1.idx)
        table2 = tr.key_by(new_key=hl.cond((tr.idx == 4) | (tr.idx == 6),
                                           hl.null(hl.tint32), tr.idx),
                           key2=1)
        table2 = table2.select(idx2=table2.idx)

        left_join = table1.join(table2, 'left')
        right_join = table1.join(table2, 'right')
        inner_join = table1.join(table2, 'inner')
        outer_join = table1.join(table2, 'outer')

        def row(new_key, idx1, idx2):
            return hl.Struct(new_key=new_key, key2=1, idx1=idx1, idx2=idx2)

        left_join_expected = [row(0, 0, 0), row(1, 1, 1), row(2, 2, 2),
                              row(4, 4, None), row(6, 6, None),
                              row(None, 3, None), row(None, 5, None)]

        right_join_expected = [row(0, 0, 0), row(1, 1, 1), row(2, 2, 2),
                               row(3, None, 3), row(5, None, 5),
                               row(None, None, 4), row(None, None, 6)]

        inner_join_expected = [row(0, 0, 0), row(1, 1, 1), row(2, 2, 2)]

        outer_join_expected = [row(0, 0, 0), row(1, 1, 1), row(2, 2, 2),
                               row(3, None, 3), row(4, 4, None),
                               row(5, None, 5), row(6, 6, None),
                               row(None, 3, None), row(None, 5, None),
                               row(None, None, 4), row(None, None, 6)]

        self.assertEqual(left_join.collect(), left_join_expected)
        self.assertEqual(right_join.collect(), right_join_expected)
        self.assertEqual(inner_join.collect(), inner_join_expected)
        self.assertEqual(outer_join.collect(), outer_join_expected)

    def test_null_joins_2(self):
        tr = hl.utils.range_table(7, 1)
        table1 = tr.key_by(new_key=hl.cond((tr.idx == 3) | (tr.idx == 5),
                                           hl.null(hl.tint32), tr.idx),
                           key2=tr.idx)
        table1 = table1.select(idx1=table1.idx)
        table2 = tr.key_by(new_key=hl.cond((tr.idx == 4) | (tr.idx == 6),
                                           hl.null(hl.tint32), tr.idx),
                           key2=tr.idx)
        table2 = table2.select(idx2=table2.idx)

        left_join = table1.join(table2, 'left')
        right_join = table1.join(table2, 'right')
        inner_join = table1.join(table2, 'inner')
        outer_join = table1.join(table2, 'outer')

        def row(new_key, key2, idx1, idx2):
            return hl.Struct(new_key=new_key, key2=key2, idx1=idx1, idx2=idx2)

        left_join_expected = [row(0, 0, 0, 0), row(1, 1, 1, 1), row(2, 2, 2, 2),
                              row(4, 4, 4, None), row(6, 6, 6, None),
                              row(None, 3, 3, None), row(None, 5, 5, None)]

        right_join_expected = [row(0, 0, 0, 0), row(1, 1, 1, 1), row(2, 2, 2, 2),
                               row(3, 3, None, 3), row(5, 5, None, 5),
                               row(None, 4, None, 4), row(None, 6, None, 6)]

        inner_join_expected = [row(0, 0, 0, 0), row(1, 1, 1, 1), row(2, 2, 2, 2)]

        outer_join_expected = [row(0, 0, 0, 0), row(1, 1, 1, 1), row(2, 2, 2, 2),
                               row(3, 3, None, 3), row(4, 4, 4, None),
                               row(5, 5, None, 5), row(6, 6, 6, None),
                               row(None, 3, 3, None), row(None, 4, None, 4),
                               row(None, 5, 5, None), row(None, 6, None, 6)]

        self.assertEqual(left_join.collect(), left_join_expected)
        self.assertEqual(right_join.collect(), right_join_expected)
        self.assertEqual(inner_join.collect(), inner_join_expected)
        self.assertEqual(outer_join.collect(), outer_join_expected)

    def test_partitioning_rewrite(self):
        ht = hl.utils.range_table(10, 3)
        ht1 = ht.annotate(x=hl.rand_unif(0, 1))
        self.assertEqual(ht1.x.collect()[:5], ht1.head(5).x.collect())

    def test_flatten(self):
        t1 = hl.utils.range_table(10)
        t1 = t1.key_by(x = hl.struct(a=t1.idx, b=0)).flatten()
        t2 = hl.utils.range_table(10).key_by()
        t2 = t2.annotate(**{'x.a': t2.idx, 'x.b': 0})
        self.assertTrue(t1._same(t2))

    def test_expand_types(self):
        t1 = hl.utils.range_table(10)
        t1 = t1.key_by(x = hl.locus('1', t1.idx+1)).expand_types()
        t2 = hl.utils.range_table(10).key_by()
        t2 = t2.annotate(x=hl.struct(contig='1', position=t2.idx+1))
        self.assertTrue(t1._same(t2))

    def test_expand_types_on_all_types(self):
        t = create_all_values_table()
        t.expand_types()

    def test_join_mangling(self):
        t1 = hl.utils.range_table(10).annotate_globals(glob1=5).annotate(row1=5)
        j = t1.join(t1, 'inner')
        assert j.row.dtype == hl.tstruct(idx=hl.tint32, row1=hl.tint32, row1_1=hl.tint32)
        assert j.globals.dtype == hl.tstruct(glob1=hl.tint32, glob1_1=hl.tint32)
        j._force_count()

    def test_key_by_aggregate_rewriting(self):
        ht = hl.utils.range_table(10)
        ht = ht.group_by(x=ht.idx % 5).aggregate(aggr = hl.agg.count())
        assert(ht.count() == 5)

    def test_field_method_assignment(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(sample=1)
        ht = ht.annotate(_row=2)
        assert ht['sample'].dtype == hl.tint32
        assert ht['_row'].dtype == hl.tint32

    def test_refs_with_process_joins(self):
        ht = hl.utils.range_table(10).annotate(foo=5)
        ht.annotate(a_join=ht[ht.key],
                    a_literal=hl.literal(['a']),
                    the_row_failure=hl.cond(True, ht.row, hl.null(ht.row.dtype)),
                    the_global_failure=hl.cond(True, ht.globals, hl.null(ht.globals.dtype))).count()

    def test_aggregate_localize_false(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(y = ht.idx + ht.aggregate(hl.agg.max(ht.idx), _localize=False))
        assert ht.y.collect() == [x + 9 for x in range(10)]

    def test_collect_localize_false(self):
        ht = hl.utils.range_table(10)
        assert hl.eval(ht.collect(_localize=False)) == ht.collect()

    def test_take_localize_false(self):
        ht = hl.utils.range_table(10)
        assert hl.eval(ht.take(3, _localize=False)) == ht.take(3)

    def test_expr_collect_localize_false(self):
        ht = hl.utils.range_table(10)
        assert hl.eval(ht.idx.collect(_localize=False)) == ht.idx.collect()

    def test_expr_take_localize_false(self):
        ht = hl.utils.range_table(10)
        assert hl.eval(ht.idx.take(3, _localize=False)) == ht.idx.take(3)

    def test_empty_show(self):
        hl.utils.range_table(1).filter(False).show()

    def test_same_equal(self):
        t1 = hl.utils.range_table(1)
        self.assertTrue(t1._same(t1))

    def test_same_within_tolerance(self):
        t = hl.utils.range_table(1)
        t1 = t.annotate(x = 1.0)
        t2 = t.annotate(x = 1.0 + 1e-7)
        self.assertTrue(t1._same(t2))

    def test_same_different_type(self):
        t1 = hl.utils.range_table(1)

        t2 = t1.annotate_globals(x = 7)
        self.assertFalse(t1._same(t2))
        
        t3 = t1.annotate(x = 7)
        self.assertFalse(t1._same(t3))

        t4 = t1.key_by()
        self.assertFalse(t1._same(t4))

    def test_same_different_global(self):
        t1 = (hl.utils.range_table(1)
              .annotate_globals(x = 7))
        t2 = t1.annotate_globals(x = 8)
        self.assertFalse(t1._same(t2))

    def test_same_different_rows(self):
        t1 = (hl.utils.range_table(2)
              .annotate(x = 7))
        
        t2 = t1.annotate(x = 8)
        self.assertFalse(t1._same(t2))

        t3 = t1.filter(t1.idx == 0)
        self.assertFalse(t1._same(t3))

    def test_rvd_key_write(self):
        tempfile = new_temp_file(suffix='ht')
        ht1 = hl.utils.range_table(1).key_by(foo='a', bar='b')
        ht1.write(tempfile)  # write ensures that table is written with both key fields

        ht1 = hl.read_table(tempfile)
        ht2 = hl.utils.range_table(1).annotate(foo='a')
        assert ht2.annotate(x = ht1.key_by('foo')[ht2.foo])._force_count() == 1

    def test_show_long_field_names(self):
        hl.utils.range_table(1).annotate(**{'a' * 256: 5}).show()

    def test_show__various_types(self):
        ht = hl.utils.range_table(1)
        ht = ht.annotate(
            x1 = [1],
            x2 = [hl.struct(y=[1])],
            x3 = {1},
            x4 = {1: 'foo'},
            x5 = {hl.struct(foo=5): 'bar'},
            x6 = hl.tuple(()),
            x7 = hl.tuple(('3',)),
            x8 = hl.tuple(('3', 3)),
            x9 = 4.2,
            x10 = hl.dict({'hello': 3, 'bar': 5})
        )
        ht.show()

    def test_import_filter_replace(self):
        def assert_filter_equals(filter, find_replace, to):
            assert hl.import_table(resource('filter_replace.txt'),
                                   filter=filter,
                                   find_replace=find_replace)['HEADER1'].collect() == to

        assert_filter_equals('Foo', None, ['(Baz),(Qux)('])
        assert_filter_equals(None, (r',', ''), ['(Foo(Bar))', '(Baz)(Qux)('])
        assert_filter_equals(None, (r'\((\w+)\)', '$1'), ['(Foo,Bar)', 'Baz,Qux('])

    def test_import_multiple_missing(self):
        ht = hl.import_table(resource('global_list.txt'),
                             missing=['gene1', 'gene2'],
                             no_header=True)

        assert ht.f0.collect() == [None, None, 'gene5', 'gene4', 'gene3']

    def test_unicode_ordering(self):
        a = hl.literal(["é", "e"])
        ht = hl.utils.range_table(1, 1)
        ht = ht.annotate(fd=hl.sorted(a))
        assert ht.fd.collect()[0] == ["e", "é"]

    def test_physical_key_truncation(self):
        path = new_temp_file(suffix='ht')
        hl.import_vcf(resource('sample.vcf')).rows().key_by('locus').write(path)
        hl.read_table(path).select()._force_count()

def test_large_number_of_fields(tmpdir):
    ht = hl.utils.range_table(100)
    ht = ht.annotate(**{
        str(k): k for k in range(1000)
    })
    f = tmpdir.join("foo.ht")
    assert_time(lambda: ht.count(), 5)
    assert_time(lambda: ht.write(str(f)), 5)
    ht = assert_time(lambda: hl.read_table(str(f)), 5)
    assert_time(lambda: ht.count(), 5)

def test_import_many_fields():
    assert_time(lambda: hl.import_table(resource('many_cols.txt')), 5)

def test_segfault():
    t = hl.utils.range_table(1)
    t2 = hl.utils.range_table(3)
    t = t.annotate(foo = [0])
    t2 = t2.annotate(foo = [0])
    joined = t.key_by('foo').join(t2.key_by('foo'))
    joined = joined.filter(hl.is_missing(joined.idx))
    assert joined.collect() == []
