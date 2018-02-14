"""
Unit tests for Hail.
"""
from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest
import random
import hail as hl
import hail.expr.aggregators as agg
from hail.utils.misc import test_file, new_temp_file
import json



def setUpModule():
    hl.init(master='local[2]', min_block_size=0)


def tearDownModule():
    hl.stop()


def schema_eq(x, y):
    x_fds = dict([(fd.name, fd.typ) for fd in x.fields])
    y_fds = dict([(fd.name, fd.typ) for fd in y.fields])
    return x_fds == y_fds


def convert_struct_to_dict(x):
    if isinstance(x, hl.Struct):
        return {k: convert_struct_to_dict(v) for k, v in x._fields.iteritems()}
    elif isinstance(x, list):
        return [convert_struct_to_dict(elt) for elt in x]
    elif isinstance(x, tuple):
        return tuple([convert_struct_to_dict(elt) for elt in x])
    elif isinstance(x, dict):
        return {k: convert_struct_to_dict(v) for k, v in x.iteritems()}
    else:
        return x


class TableTests(unittest.TestCase):
    def test_annotate(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(), hl.TArray(hl.TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)

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
            x14=hl.cond(kt.a < kt.b, kt.c, hl.null(hl.TInt32())),
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

    def test_query(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(), hl.TArray(hl.TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)
        results = kt.aggregate(hl.Struct(q1=agg.sum(kt.b),
                                         q2=agg.count(),
                                         q3=agg.collect(kt.e),
                                         q4=agg.collect(agg.filter((kt.d >= 5) | (kt.a == 0), kt.e))))

        self.assertEqual(results.q1, 8)
        self.assertEqual(results.q2, 3)
        self.assertEqual(set(results.q3), {"hello", "cat", "dog"})
        self.assertEqual(set(results.q4), {"hello", "cat"})

    def test_filter(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(), hl.TArray(hl.TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)

        self.assertEqual(kt.filter(kt.a == 4).count(), 2)
        self.assertEqual(kt.filter((kt.d == -1) | (kt.c == 20) | (kt.e == "hello")).count(), 3)
        self.assertEqual(kt.filter((kt.c != 20) & (kt.a == 4)).count(), 1)
        self.assertEqual(kt.filter(True).count(), 3)

    def test_transmute(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(), hl.TArray(hl.TInt32()),
                             hl.TStruct(['x', 'y'], [hl.TBoolean(), hl.TInt32()])])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3], 'g': {'x': True, 'y': 2}},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': [], 'g': {'x': True, 'y': 2}},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7], 'g': None}]
        df = hl.Table.parallelize(rows, schema)

        df = df.transmute(h=df.a + df.b + df.c + df.g.y)
        r = df.select('h').collect()

        self.assertEqual(df.columns, ['d', 'e', 'f', 'h'])
        self.assertEqual(r, [hl.Struct(h=x) for x in [10, 20, None]])

    def test_select(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(), hl.TArray(hl.TInt32()),
                             hl.TStruct(['x', 'y'], [hl.TBoolean(), hl.TInt32()])])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3], 'g': {'x': True, 'y': 2}},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': [], 'g': {'x': True, 'y': 2}},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7], 'g': None}]

        kt = hl.Table.parallelize(rows, schema)

        self.assertEqual(kt.select(kt.a, kt.e).columns, ['a', 'e'])
        self.assertEqual(kt.select(*[kt.a, kt.e]).columns, ['a', 'e'])
        self.assertEqual(kt.select(kt.a, foo=kt.a + kt.b - kt.c - kt.d).columns, ['a', 'foo'])
        self.assertEqual(set(kt.select(kt.a, foo=kt.a + kt.b - kt.c - kt.d, **kt.g).columns), {'a', 'foo', 'x', 'y'})

    def test_aggregate(self):
        schema = hl.TStruct(['status', 'GT', 'qPheno'],
                            [hl.TInt32(), hl.TCall(), hl.TInt32()])

        rows = [{'status': 0, 'GT': hl.Call([0, 0]), 'qPheno': 3},
                {'status': 0, 'GT': hl.Call([0, 1]), 'qPheno': 13}]

        kt = hl.Table.parallelize(rows, schema)

        result = convert_struct_to_dict(
            kt.group_by(status=kt.status)
                .aggregate(x1=agg.collect(kt.qPheno * 2),
                           x2=agg.collect(agg.explode([kt.qPheno, kt.qPheno + 1])),
                           x3=agg.min(kt.qPheno),
                           x4=agg.max(kt.qPheno),
                           x5=agg.sum(kt.qPheno),
                           x6=agg.product(kt.qPheno.to_int64()),
                           x7=agg.count(),
                           x8=agg.count_where(kt.qPheno == 3),
                           x9=agg.fraction(kt.qPheno == 1),
                           x10=agg.stats(kt.qPheno.to_float64()),
                           x11=agg.hardy_weinberg(kt.GT),
                           x13=agg.inbreeding(kt.GT, 0.1),
                           x14=agg.call_stats(kt.GT, ["A", "T"]),
                           x15=agg.collect(hl.Struct(a=5, b="foo", c=hl.Struct(banana='apple')))[0],
                           x16=agg.collect(hl.Struct(a=5, b="foo", c=hl.Struct(banana='apple')).c.banana)[0]
                           ).take(1)[0])

        expected = {u'status': 0,
                    u'x13': {u'n_called': 2L, u'expected_homs': 1.64, u'f_stat': -1.777777777777777,
                             u'observed_homs': 1L},
                    u'x14': {u'AC': [3, 1], u'AF': [0.75, 0.25], u'GC': [1, 1, 0], u'AN': 4},
                    u'x15': {u'a': 5, u'c': {u'banana': u'apple'}, u'b': u'foo'},
                    u'x10': {u'min': 3.0, u'max': 13.0, u'sum': 16.0, u'stdev': 5.0, u'nNotMissing': 2L, u'mean': 8.0},
                    u'x8': 1L, u'x9': 0.0, u'x16': u'apple',
                    u'x11': {u'rExpectedHetFrequency': 0.5, u'pHWE': 0.5},
                    u'x2': [3, 4, 13, 14], u'x3': 3, u'x1': [6, 26], u'x6': 39L, u'x7': 2L, u'x4': 13, u'x5': 16}

        self.assertDictEqual(result, expected)

    def test_errors(self):
        schema = hl.TStruct(['status', 'gt', 'qPheno'],
                            [hl.TInt32(), hl.TCall(), hl.TInt32()])

        rows = [{'status': 0, 'gt': hl.Call([0, 0]), 'qPheno': 3},
                {'status': 0, 'gt': hl.Call([0, 1]), 'qPheno': 13},
                {'status': 1, 'gt': hl.Call([0, 1]), 'qPheno': 20}]

        kt = hl.Table.parallelize(rows, schema)

        def f():
            kt.a = 5

        self.assertRaises(NotImplementedError, f)

    def test_joins(self):
        kt = hl.Table.range(1).drop('idx')
        kt = kt.annotate(a='foo')

        kt1 = hl.Table.range(1).drop('idx')
        kt1 = kt1.annotate(a='foo', b='bar').key_by('a')

        kt2 = hl.Table.range(1).drop('idx')
        kt2 = kt2.annotate(b='bar', c='baz').key_by('b')

        kt3 = hl.Table.range(1).drop('idx')
        kt3 = kt3.annotate(c='baz', d='qux').key_by('c')

        kt4 = hl.Table.range(1).drop('idx')
        kt4 = kt4.annotate(d='qux', e='quam').key_by('d')

        ktr = kt.annotate(e=kt4[kt3[kt2[kt1[kt.a].b].c].d].e)
        self.assertTrue(ktr.aggregate(agg.collect(ktr.e))== ['quam'])

        ktr = kt.select(e=kt4[kt3[kt2[kt1[kt.a].b].c].d].e)
        self.assertTrue(ktr.aggregate(agg.collect(ktr.e)) == ['quam'])

        self.assertEqual(kt.filter(kt4[kt3[kt2[kt1[kt.a].b].c].d].e == 'quam').count(), 1)

        m = hl.import_vcf(test_file('sample.vcf'))
        vkt = m.rows_table()
        vkt = vkt.select(vkt.locus, vkt.alleles, vkt.qual)
        vkt = vkt.annotate(qual2=m[(vkt.locus, vkt.alleles), :].qual)
        self.assertTrue(vkt.filter(vkt.qual != vkt.qual2).count() == 0)

        m2 = m.annotate_rows(qual2=vkt[m.locus, m.alleles].qual)
        self.assertTrue(m2.filter_rows(m2.qual != m2.qual2).count_rows() == 0)

        m3 = m.annotate_rows(qual2=m[(m.locus, m.alleles), :].qual)
        self.assertTrue(m3.filter_rows(m3.qual != m3.qual2).count_rows() == 0)

        kt = hl.Table.range(1)
        kt = kt.annotate_globals(foo=5)

        kt2 = hl.Table.range(1)

        kt2 = kt2.annotate_globals(kt_foo=kt[:].foo)
        self.assertEqual(kt2.get_globals().kt_foo, 5)

    def test_drop(self):
        kt = hl.Table.range(10)
        kt = kt.annotate(sq=kt.idx ** 2, foo='foo', bar='bar')

        self.assertEqual(kt.drop('idx', 'foo').columns, ['sq', 'bar'])
        self.assertEqual(kt.drop(kt['idx'], kt['foo']).columns, ['sq', 'bar'])

    def test_weird_names(self):
        df = hl.Table.range(10)
        exprs = {'a': 5, '   a    ': 5, r'\%!^!@#&#&$%#$%': [5]}

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
        kt = hl.Table.range(10)
        kt_small = kt.sample(0.01)
        self.assertTrue(kt_small.count() < kt.count())


class MatrixTests(unittest.TestCase):
    def get_vds(self, min_partitions=None):
        return hl.import_vcf(test_file("sample.vcf"), min_partitions=min_partitions)

    def test_update(self):
        vds = self.get_vds()
        vds = vds.select_entries(dp=vds.DP, gq=vds.GQ)
        self.assertTrue(schema_eq(vds.entry_schema, hl.TStruct(['dp', 'gq'], [hl.TInt32(), hl.TInt32()])))

    def test_annotate(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)

        new_global_schema = vds.global_schema
        self.assertEqual(new_global_schema, hl.TStruct(['foo'], [hl.TInt32()]))

        orig_variant_schema = vds.row_schema
        vds = vds.annotate_rows(x1=agg.count(),
                                x2=agg.fraction(False),
                                x3=agg.count_where(True),
                                x4=vds.info.AC + vds.foo)

        expected_fields = [(fd.name, fd.typ) for fd in orig_variant_schema.fields] + \
                          [('x1', hl.TInt64()),
                           ('x2', hl.TFloat64()),
                           ('x3', hl.TInt64()),
                           ('x4', hl.TArray(hl.TInt32()))]

        self.assertTrue(orig_variant_schema, hl.TStruct(*[list(x) for x in zip(*expected_fields)]))

        vds = vds.annotate_cols(apple=6)
        vds = vds.annotate_cols(y1=agg.count(),
                                y2=agg.fraction(False),
                                y3=agg.count_where(True),
                                y4=vds.foo + vds.apple)

        expected_schema = hl.TStruct(['s', 'apple', 'y1', 'y2', 'y3', 'y4'],
                                     [hl.TString(),
                                      hl.TInt32(), hl.TInt64(), hl.TFloat64(), hl.TInt64(), hl.TInt32()])

        self.assertTrue(schema_eq(vds.col_schema, expected_schema),
                        "expected: " + str(vds.col_schema) + "\nactual: " + str(expected_schema))

        vds = vds.select_entries(z1=vds.x1 + vds.foo,
                                 z2=vds.x1 + vds.y1 + vds.foo)
        self.assertTrue(schema_eq(vds.entry_schema, hl.TStruct(['z1', 'z2'], [hl.TInt64(), hl.TInt64()])))

    def test_filter(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)
        vds = vds.annotate_rows(x1=agg.count())
        vds = vds.annotate_cols(y1=agg.count())
        vds = vds.annotate_entries(z1=vds.DP)

        vds = vds.filter_rows((vds.x1 == 5) & (agg.count() == 3) & (vds.foo == 2))
        vds = vds.filter_cols((vds.y1 == 5) & (agg.count() == 3) & (vds.foo == 2))
        vds = vds.filter_entries((vds.z1 < 5) & (vds.y1 == 3) & (vds.x1 == 5) & (vds.foo == 2))
        vds.count_rows()

    def test_query(self):
        vds = self.get_vds()

        vds = vds.annotate_globals(foo=5)
        vds = vds.annotate_rows(x1=agg.count())
        vds = vds.annotate_cols(y1=agg.count())
        vds = vds.annotate_entries(z1=vds.DP)

        qv = vds.aggregate_rows(agg.count())
        qs = vds.aggregate_cols(agg.count())
        qg = vds.aggregate_entries(agg.count())

        self.assertEqual(qv, 346)
        self.assertEqual(qs, 100)
        self.assertEqual(qg, qv * qs)

        qvs = vds.aggregate_rows(hl.Struct(x=agg.collect(vds.locus.contig),
                                           y=agg.collect(vds.x1)))

        qss = vds.aggregate_cols(hl.Struct(x=agg.collect(vds.s),
                                           y=agg.collect(vds.y1)))

        qgs = vds.aggregate_entries(hl.Struct(x=agg.collect(agg.filter(False, vds.y1)),
                                              y=agg.collect(agg.filter(hl.rand_bool(0.1), vds.GT))))

    def test_drop(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)
        vds = vds.annotate_cols(bar=5)

        vds1 = vds.drop('GT', 'info', 'foo', 'bar')
        self.assertTrue(not any(f.name == 'foo' for f in vds1.global_schema.fields))
        self.assertTrue(not any(f.name == 'info' for f in vds1.row_schema.fields))
        self.assertTrue(not any(f.name == 'bar' for f in vds1.col_schema.fields))
        self.assertTrue(not any(f.name == 'GT' for f in vds1.entry_schema.fields))

        vds2 = vds.drop(vds.GT, vds.info, vds.foo, vds.bar)
        self.assertTrue(not any(f.name == 'foo' for f in vds2.global_schema.fields))
        self.assertTrue(not any(f.name == 'info' for f in vds2.row_schema.fields))
        self.assertTrue(not any(f.name == 'bar' for f in vds2.col_schema.fields))
        self.assertTrue(not any(f.name == 'GT' for f in vds2.entry_schema.fields))

    def test_drop_rows(self):
        vds = self.get_vds()
        vds = vds.drop_rows()
        self.assertEqual(vds.count_rows(), 0)

    def test_drop_cols(self):
        vds = self.get_vds()
        vds = vds.drop_cols()
        self.assertEqual(vds.count_cols(), 0)

    def test_weird_names(self):
        ds = self.get_vds()
        exprs = {'a': 5, '   a    ': 5, r'\%!^!@#&#&$%#$%': [5]}

        ds.annotate_globals(**exprs)
        ds.select_globals(**exprs)

        ds.annotate_cols(**exprs)
        ds1 = ds.select_cols(**exprs)

        ds.annotate_rows(**exprs)
        ds2 = ds.select_rows(*ds.row_key, **exprs)

        ds.annotate_entries(**exprs)
        ds.select_entries(**exprs)

        ds1.explode_cols('\%!^!@#&#&$%#$%')
        ds1.explode_cols(ds1['\%!^!@#&#&$%#$%'])
        ds1.group_cols_by(ds1.a).aggregate(**{'*``81': agg.count()})

        ds1.drop('\%!^!@#&#&$%#$%')
        ds1.drop(ds1['\%!^!@#&#&$%#$%'])

        ds2.explode_rows('\%!^!@#&#&$%#$%')
        ds2.explode_rows(ds2['\%!^!@#&#&$%#$%'])
        ds2.group_rows_by(ds2.a).aggregate(**{'*``81': agg.count()})

    def test_joins(self):
        vds = self.get_vds().select_rows('locus', 'alleles', x1=1, y1=1)
        vds2 = vds.select_rows(*vds.row_key, x2=1, y2=2)
        vds2 = vds2.select_cols('s', c1=1, c2=2)

        vds = vds.annotate_rows(y2=vds2[(vds.locus, vds.alleles), :].y2)
        vds = vds.annotate_cols(c2=vds2[:, vds.s].c2)

        vds = vds.annotate_cols(c2=vds2[:, hl.str(vds.s)].c2)

        rt = vds.rows_table()
        ct = vds.cols_table()

        vds.annotate_rows(**rt[vds.locus, vds.alleles])

        self.assertTrue(rt.all(rt.y2 == 2))
        self.assertTrue(ct.all(ct.c2 == 2))

    def test_table_join(self):
        ds = self.get_vds()
        # test different row schemas
        self.assertTrue(ds.union_cols(ds.drop(ds.info))
                        .count_rows(), 346)

    def test_naive_coalesce(self):
        vds = self.get_vds(min_partitions=8)
        self.assertEqual(vds.num_partitions(), 8)
        repart = vds.naive_coalesce(2)
        self.assertTrue(vds._same(repart))

    def tests_unions(self):
        dataset = hl.import_vcf(test_file('sample2.vcf'))

        # test union_rows
        ds1 = dataset.filter_rows(dataset.locus.position % 2 == 1)
        ds2 = dataset.filter_rows(dataset.locus.position % 2 == 0)

        datasets = [ds1, ds2]
        r1 = ds1.union_rows(ds2)
        r2 = hl.MatrixTable.union_rows(*datasets)

        self.assertTrue(r1._same(r2))

        # test union_cols
        ds = dataset.union_cols(dataset).union_cols(dataset)
        for s, count in ds.aggregate_cols(agg.counter(ds.s)).items():
            self.assertEqual(count, 3)

    def test_index(self):
        ds = self.get_vds(min_partitions=8)
        self.assertEqual(ds.num_partitions(), 8)
        ds = ds.index_rows('rowidx').index_cols('colidx')

        for i, struct in enumerate(ds.cols_table().select('colidx').collect()):
            self.assertEqual(i, struct.colidx)
        for i, struct in enumerate(ds.rows_table().select('rowidx').collect()):
            self.assertEqual(i, struct.rowidx)

    def test_reorder_columns(self):
        ds = self.get_vds()
        new_sample_order = [x.s for x in ds.cols_table().select("s").collect()]
        random.shuffle(new_sample_order)
        self.assertEqual([x.s for x in ds.reorder_columns(new_sample_order).cols_table().select("s").collect()],
                         new_sample_order)

    def test_computed_key_join_1(self):
        ds = self.get_vds()
        kt = hl.Table.parallelize(
            [{'key': 0, 'value': True},
             {'key': 1, 'value': False}],
            hl.TStruct(['key', 'value'],
                       [hl.TInt32(), hl.TBoolean()]),
            key=['key'])
        ds = ds.annotate_rows(key=ds.locus.position % 2)
        ds = ds.annotate_rows(value=kt[ds['key']].value)
        rt = ds.rows_table()
        self.assertTrue(
            rt.all(((rt.locus.position % 2) == 0) == rt.value))

    def test_computed_key_join_2(self):
        # multiple keys
        # FIXME: reenable with Python 3
        # ds = self.get_vds()
        # kt =hl.Table.parallelize(
        #     [{'key1': 0, 'key2': 0, 'value': 0},
        #      {'key1': 1, 'key2': 0, 'value': 1},
        #      {'key1': 0, 'key2': 1, 'value': -2},
        #      {'key1': 1, 'key2': 1, 'value': -1}],
        #     hl.TStruct(['key1', 'key2', 'value'],
        #             [hl.TInt32(), hl.TInt32(), hl.TInt32()]),
        #     key=['key1', 'key2'])
        # ds = ds.annotate_rows(key1=ds.locus.position % 2, key2=ds.info.DP % 2)
        # ds = ds.annotate_rows(value=kt[ds.key1, ds.key2].value)
        # rt = ds.rows_table()
        # self.assertTrue(
        #     rt.forall((rt.locus.position % 2) - 2 * (rt.info.DP % 2) == rt.value))
        pass

    def test_computed_key_join_3(self):
        # duplicate row keys
        # FIXME: reenable with Python 3
        # ds = self.get_vds()
        # kt =hl.Table.parallelize(
        #     [{'culprit': 'InbreedingCoeff', 'foo': 'bar', 'value': 'IB'}],
        #     hl.TStruct(['culprit', 'foo', 'value'],
        #             [hl.TString(), hl.TString(), hl.TString()]),
        #     key=['culprit', 'foo'])
        # ds = ds.annotate_rows(
        #     dsfoo='bar',
        #     info=ds.info.annotate(culprit=[ds.info.culprit, "foo"]))
        # ds = ds.explode_rows(ds.info.culprit)
        # ds = ds.annotate_rows(value=kt[ds.info.culprit, ds.dsfoo].value)
        # rt = ds.rows_table()
        # self.assertTrue(
        #     rt.forall(hl.cond(
        #         rt.info.culprit == "InbreedingCoeff",
        #         rt.value == "IB",
        #         hl.is_missing(rt.value))))
        pass

    def test_vcf_regression(self):
        ds = hl.import_vcf(test_file('33alleles.vcf'))
        self.assertEqual(
            ds.filter_rows(ds.alleles.length() == 2).count_rows(), 0)

    def test_field_groups(self):
        ds = self.get_vds()

        df = ds.annotate_rows(row_struct=ds.row).rows_table()
        self.assertTrue(df.all((df.info == df.row_struct.info) & (df.qual == df.row_struct.qual)))

        ds2 = ds.index_cols()
        df = ds2.annotate_cols(col_struct=ds2.col).cols_table()
        self.assertTrue(df.all((df.col_idx == df.col_struct.col_idx)))

        df = ds.annotate_entries(entry_struct=ds.entry).entries_table()
        self.assertTrue(df.all(
            ((hl.is_missing(df.GT) |
              (df.GT == df.entry_struct.GT)) &
             (df.AD == df.entry_struct.AD))))

    def test_filter_partitions(self):
        ds = self.get_vds(min_partitions=8)
        self.assertEqual(ds.num_partitions(), 8)
        self.assertEqual(ds._filter_partitions([0, 1, 4]).num_partitions(), 3)
        self.assertEqual(ds._filter_partitions([4, 5, 7], keep=False).num_partitions(), 5)
        self.assertTrue(
            ds._same(hl.MatrixTable.union_rows(
                ds._filter_partitions([0, 3, 7]),
                ds._filter_partitions([0, 3, 7], keep=False))))

    def test_from_rows_table(self):
        ds = hl.import_vcf(test_file('sample.vcf'))
        rt = ds.rows_table()
        rm = hl.MatrixTable.from_rows_table(rt)
        # would be nice to compare rm to ds.drop_cols(), but the latter
        # preserves the col, entry types
        self.assertEqual(rm.count_cols(), 0)
        self.assertTrue(rm.rows_table()._same(rt))

    def test_sample_rows(self):
        ds = self.get_vds()
        ds_small = ds.sample_rows(0.01)
        self.assertTrue(ds_small.count_rows() < ds.count_rows())

    def test_read_stored_cols(self):
        ds = self.get_vds()
        ds = ds.annotate_globals(x = 'foo')
        f = new_temp_file(suffix='vds')
        ds.write(f)
        t = hl.read_table(f + '/cols')
        self.assertTrue(ds.cols_table()._same(t))

    def test_read_stored_rows(self):
        ds = self.get_vds()
        ds = ds.annotate_globals(x = 'foo')
        f = new_temp_file(suffix='vds')
        ds.write(f)
        t = hl.read_table(f + '/rows')
        self.assertTrue(ds.rows_table()._same(t))

    def test_read_stored_globals(self):
        ds = self.get_vds()
        ds = ds.annotate_globals(x = 5, baz = 'foo')
        f = new_temp_file(suffix='vds')
        ds.write(f)
        t = hl.read_table(f + '/globals')
        self.assertTrue(ds.globals_table()._same(t))

    def test_codecs_matrix(self):
        from hail.utils.java import Env, scala_object
        codecs = scala_object(Env.hail().io, 'CodecSpec').codecSpecs()
        ds = self.get_vds()
        temp = new_temp_file(suffix='hmt')
        for codec in codecs:
            ds.write(temp, overwrite=True, _codec_spec=codec.toString())
            ds2 = hl.read_matrix_table(temp)
            self.assertTrue(ds._same(ds2))

    def test_codecs_table(self):
        from hail.utils.java import Env, scala_object
        codecs = scala_object(Env.hail().io, 'CodecSpec').codecSpecs()
        rt = self.get_vds().rows_table()
        temp = new_temp_file(suffix='ht')
        for codec in codecs:
            rt.write(temp, overwrite=True, _codec_spec=codec.toString())
            rt2 = hl.read_table(temp)
            self.assertTrue(rt._same(rt2))

class FunctionsTests(unittest.TestCase):
    def test(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(),
                             hl.TArray(hl.TInt32()),
                             hl.TArray(hl.TStruct(['x', 'y', 'z'], [hl.TInt32(), hl.TInt32(), hl.TString()])),
                             hl.TStruct(['a', 'b', 'c'], [hl.TInt32(), hl.TInt32(), hl.TString()]),
                             hl.TBoolean(), hl.TStruct(['x', 'y', 'z'], [hl.TInt32(), hl.TInt32(), hl.TString()])])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5,
                 'e': "hello", 'f': [1, 2, 3],
                 'g': [hl.Struct(x=1, y=5, z='banana')],
                 'h': hl.Struct(a=5, b=3, c='winter'),
                 'i': True,
                 'j': hl.Struct(x=3, y=2, z='summer')}]

        kt = hl.Table.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            chisq=hl.chisq(kt.a, kt.b, kt.c, kt.d),
            ctt=hl.ctt(kt.a, kt.b, kt.c, kt.d, 5),
            Dict=hl.dict([kt.a, kt.b], [kt.c, kt.d]),
            dpois=hl.dpois(4, kt.a),
            drop=kt.h.drop('b', 'c'),
            exp=hl.exp(kt.c),
            fet=hl.fisher_exact_test(kt.a, kt.b, kt.c, kt.d),
            hwe=hl.hardy_weinberg_p(1, 2, 1),
            index=hl.index(kt.g, 'z'),
            is_defined=hl.is_defined(kt.i),
            is_missing=hl.is_missing(kt.i),
            is_nan=hl.is_nan(kt.a.to_float64()),
            json=hl.json(kt.g),
            log=hl.log(kt.a.to_float64(), kt.b.to_float64()),
            log10=hl.log10(kt.c.to_float64()),
            or_else=hl.or_else(kt.a, 5),
            or_missing=hl.or_missing(kt.i, kt.j),
            pchisqtail=hl.pchisqtail(kt.a.to_float64(), kt.b.to_float64()),
            pcoin=hl.rand_bool(0.5),
            pnorm=hl.pnorm(0.2),
            pow=2.0 ** kt.b,
            ppois=hl.ppois(kt.a.to_float64(), kt.b.to_float64()),
            qchisqtail=hl.qchisqtail(kt.a.to_float64(), kt.b.to_float64()),
            range=hl.range(0, 5, kt.b),
            rnorm=hl.rand_norm(0.0, kt.b),
            rpois=hl.rand_pois(kt.a),
            runif=hl.rand_unif(kt.b, kt.a),
            select=kt.h.select('c', 'b'),
            sqrt=hl.sqrt(kt.a),
            to_str=[hl.str(5), hl.str(kt.a), hl.str(kt.g)],
            where=hl.cond(kt.i, 5, 10)
        ).take(1)[0])

        # print(result) # Fixme: Add asserts


class ColumnTests(unittest.TestCase):
    def test_operators(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                            [hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TInt32(), hl.TString(), hl.TArray(hl.TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = hl.Table.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
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
        ).take(1)[0])

        expected = {'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3],
                    'x1': 9, 'x2': 9, 'x3': 5,
                    'x4': -1, 'x5': 1, 'x6': 3,
                    'x7': 20, 'x8': 20, 'x9': 4,
                    'x10': 4.0 / 5, 'x11': 5.0 / 4, 'x12': 4, 'x13': -4, 'x14': 4,
                    'x15': False, 'x16': False, 'x17': False,
                    'x18': True, 'x19': True, 'x20': True,
                    'x21': True, 'x22': False, 'x23': True,
                    'x24': True, 'x25': False, 'x26': True,
                    'x27': False, 'x28': True, 'x29': False,
                    'x30': False, 'x31': True, 'x32': False,
                    'x33': False, 'x34': False, 'x35': False, 'x36': True}

        self.maxDiff = 2000

        self.assertDictEqual(result, expected)

    def test_array_column(self):
        schema = hl.TStruct(['a'], [hl.TArray(hl.TInt32())])
        rows = [{'a': [1, 2, 3]}]
        kt = hl.Table.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            x1=kt.a[0],
            x2=kt.a[2],
            x3=kt.a[:],
            x4=kt.a[1:2],
            x5=kt.a[-1:2],
            x6=kt.a[:2]
        ).take(1)[0])

        expected = {'a': [1, 2, 3], 'x1': 1, 'x2': 3, 'x3': [1, 2, 3],
                    'x4': [2], 'x5': [], 'x6': [1, 2]}

        self.assertDictEqual(result, expected)

    def test_dict_column(self):
        schema = hl.TStruct(['x'], [hl.TFloat64()])
        rows = [{'x': 2.0}]
        kt = hl.Table.parallelize(rows, schema)

        kt = kt.annotate(a=hl.dict(['cat', 'dog'], [3, 7]))

        result = convert_struct_to_dict(kt.annotate(
            x1=kt.a['cat'],
            x2=kt.a['dog'],
            x3=kt.a.keys().contains('rabbit'),
            x4=kt.a.size() == 0,
            x5=kt.a.key_set(),
            x6=kt.a.keys(),
            x7=kt.a.values(),
            x8=kt.a.size(),
            x9=kt.a.map_values(lambda v: v * 2.0)
        ).take(1)[0])

        expected = {'a': {'cat': 3, 'dog': 7}, 'x': 2.0, 'x1': 3, 'x2': 7, 'x3': False,
                    'x4': False, 'x5': {'cat', 'dog'}, 'x6': ['cat', 'dog'],
                    'x7': [3, 7], 'x8': 2, 'x9': {'cat': 6.0, 'dog': 14.0}}

        self.assertDictEqual(result, expected)

    def test_numeric_conversion(self):
        schema = hl.TStruct(['a', 'b', 'c', 'd'], [hl.TFloat64(), hl.TFloat64(), hl.TInt32(), hl.TInt64()])
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
        kt = hl.Table.parallelize(rows, schema)

        kt = kt.annotate(x1=[1.0, kt.a, 1, long(1)],
                         x2=[1, 1.0],
                         x3=[kt.a, kt.c],
                         x4=[kt.c, kt.d],
                         x5=[1, kt.c, long(1)])

        expected_schema = {'a': hl.TFloat64(), 'b': hl.TFloat64(), 'c': hl.TInt32(), 'd': hl.TInt64(),
                           'x1': hl.TArray(hl.TFloat64()), 'x2': hl.TArray(hl.TFloat64()),
                           'x3': hl.TArray(hl.TFloat64()),
                           'x4': hl.TArray(hl.TInt64()), 'x5': hl.TArray(hl.TInt64())}

        self.assertTrue(all([expected_schema[fd.name] == fd.typ for fd in kt.schema.fields]))

    def test_constructors(self):
        rg = hl.GenomeReference("foo", ["1"], {"1": 100})

        schema = hl.TStruct(['a', 'b', 'c', 'd'], [hl.TFloat64(), hl.TFloat64(), hl.TInt32(), hl.TInt64()])
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
        kt = hl.Table.parallelize(rows, schema)

        kt = kt.annotate(l1=hl.parse_locus("1:51"),
                         l2=hl.locus("1", 51, reference_genome=rg),
                         i1=hl.parse_interval("1:51-56", reference_genome=rg),
                         i2=hl.interval(hl.locus("1", 51, reference_genome=rg),
                                        hl.locus("1", 56, reference_genome=rg)))

        expected_schema = {'a': hl.TFloat64(), 'b': hl.TFloat64(), 'c': hl.TInt32(), 'd': hl.TInt64(),
                           'l1': hl.TLocus(), 'l2': hl.TLocus(rg),
                           'i1': hl.TInterval(hl.TLocus(rg)), 'i2': hl.TInterval(hl.TLocus(rg))}

        self.assertTrue(all([expected_schema[fd.name] == fd.typ for fd in kt.schema.fields]))
