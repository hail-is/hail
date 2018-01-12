"""
Unit tests for Hail.
"""
from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest
from hail2 import *

hc = None


def setUpModule():
    global hc
    hc = HailContext(master='local[2]', min_block_size=0)


def tearDownModule():
    global hc
    hc.stop()
    hc = None


def schema_eq(x, y):
    x_fds = dict([(fd.name, fd.typ) for fd in x.fields])
    y_fds = dict([(fd.name, fd.typ) for fd in y.fields])
    return x_fds == y_fds


def convert_struct_to_dict(x):
    if isinstance(x, Struct):
        return {k: convert_struct_to_dict(v) for k, v in x._attrs.iteritems()}
    elif isinstance(x, list):
        return [convert_struct_to_dict(elt) for elt in x]
    elif isinstance(x, tuple):
        return tuple([convert_struct_to_dict(elt) for elt in x])
    elif isinstance(x, dict):
        return {k: convert_struct_to_dict(v) for k, v in x.iteritems()}
    else:
        return x


class TableTests(unittest.TestCase):
    def test_conversion(self):
        test_resources = 'src/test/resources'
        kt_old = hc.import_table(test_resources + '/sampleAnnotations.tsv', impute=True).to_hail1()
        kt_new = kt_old.to_hail2()
        kt_old2 = kt_new.to_hail1()
        self.assertListEqual(kt_new.columns, ['Sample', 'Status', 'qPhen'])
        self.assertTrue(kt_old.same(kt_old2))

    def test_annotate(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = Table.parallelize(rows, schema)

        result1 = convert_struct_to_dict(kt.annotate(foo=kt.a + 1,
                                                     foo2=kt.a).to_hail1().take(1)[0])

        self.assertDictEqual(result1, {'a': 4,
                                       'b': 1,
                                       'c': 3,
                                       'd': 5,
                                       'e': "hello",
                                       'f': [1, 2, 3],
                                       'foo': 5,
                                       'foo2': 4})

        result2 = convert_struct_to_dict(kt.annotate(**{'a.foo': 5,
                                                        'b.x': "hello",
                                                        'b.y': 23,
                                                        'b.z': True,
                                                        'b.q.hello': [1, 2, 3]}
                                                     ).to_hail1().take(1)[0])

        self.assertDictEqual(result2, {'a': {'foo': 5},
                                       'b': {'x': "hello", 'y': 23, 'z': True, 'q': {'hello': [1, 2, 3]}},
                                       'c': 3,
                                       'd': 5,
                                       'e': "hello",
                                       'f': [1, 2, 3]})

        result3 = convert_struct_to_dict(kt.annotate(
            x1=kt.f.map(lambda x: x * 2),
            x2=kt.f.map(lambda x: [x, x + 1]).flatmap(lambda x: x),
            x3=kt.f.min(),
            x4=kt.f.max(),
            x5=kt.f.sum(),
            x6=kt.f.product(),
            x7=kt.f.length(),
            x8=kt.f.filter(lambda x: x == 3),
            x9=kt.f[1:],
            x10=kt.f[:],
            x11=kt.f[1:2],
            x12=kt.f.map(lambda x: [x, x + 1]),
            x13=kt.f.map(lambda x: [[x, x + 1], [x + 2]]).flatmap(lambda x: x),
            x14=functions.cond(kt.a < kt.b, kt.c, functions.null(TInt32())),
            x15={1, 2, 3}
        ).to_hail1().take(1)[0])

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
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = Table.parallelize(rows, schema)
        results = kt.aggregate(q1=agg.sum(kt.b),
                               q2=agg.count(),
                               q3=agg.collect(kt.e),
                               q4=agg.collect(agg.filter((kt.d >= 5) | (kt.a == 0), kt.e)))

        self.assertEqual(results.q1, 8)
        self.assertEqual(results.q2, 3)
        self.assertEqual(set(results.q3), {"hello", "cat", "dog"})
        self.assertEqual(set(results.q4), {"hello", "cat"})

    def test_filter(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = Table.parallelize(rows, schema)

        self.assertEqual(kt.filter(kt.a == 4).count(), 2)
        self.assertEqual(kt.filter((kt.d == -1) | (kt.c == 20) | (kt.e == "hello")).count(), 3)
        self.assertEqual(kt.filter((kt.c != 20) & (kt.a == 4)).count(), 1)
        self.assertEqual(kt.filter(True).count(), 3)

    def test_transmute(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32()),
                          TStruct(['x', 'y'], [TBoolean(), TInt32()])])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3], 'g': {'x': True, 'y': 2}},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': [], 'g': {'x': True, 'y': 2}},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7], 'g': None}]
        df = Table.parallelize(rows, schema)

        df = df.transmute(h = df.a + df.b + df.c + df.g.y)
        r = df.select('h').collect()

        self.assertEqual(df.columns, ['d', 'e', 'f', 'h'])
        self.assertEqual(r, [Struct(h=x) for x in [10, 20, None]])


    def test_select(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32()),
                          TStruct(['x', 'y'], [TBoolean(), TInt32()])])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3], 'g': {'x': True, 'y': 2}},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': [], 'g': {'x': True, 'y': 2}},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7], 'g': None}]

        kt = Table.parallelize(rows, schema)

        self.assertEqual(kt.select(kt.a, kt.e).columns, ['a', 'e'])
        self.assertEqual(kt.select(*[kt.a, kt.e]).columns, ['a', 'e'])
        self.assertEqual(kt.select(kt.a, foo=kt.a + kt.b - kt.c - kt.d).columns, ['a', 'foo'])
        self.assertEqual(kt.select(kt.a, *kt.g, foo=kt.a + kt.b - kt.c - kt.d).columns, ['a', 'x', 'y', 'foo'])

    def test_aggregate(self):
        schema = TStruct(['status', 'GT', 'qPheno'],
                         [TInt32(), TCall(), TInt32()])

        rows = [{'status': 0, 'GT': Call(0), 'qPheno': 3},
                {'status': 0, 'GT': Call(1), 'qPheno': 13}]

        kt = Table.parallelize(rows, schema)

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
                           x14=agg.call_stats(kt.GT, Variant("1", 10000, "A", "T")),
                           x15=agg.collect(Struct(a=5, b="foo", c=Struct(banana='apple')))[0],
                           x16=agg.collect(Struct(a=5, b="foo", c=Struct(banana='apple')).c.banana)[0]
                           ).to_hail1().take(1)[0])

        expected = {u'status': 0,
                    u'x13': {u'nCalled': 2L, u'expectedHoms': 1.64, u'Fstat': -1.777777777777777, u'nTotal': 2L,
                             u'observedHoms': 1L},
                    u'x14': {u'AC': [3, 1], u'AF': [0.75, 0.25], u'GC': [1, 1, 0], u'AN': 4},
                    u'x15': {u'a': 5, u'c': {u'banana': u'apple'}, u'b': u'foo'},
                    u'x10': {u'min': 3.0, u'max': 13.0, u'sum': 16.0, u'stdev': 5.0, u'nNotMissing': 2L, u'mean': 8.0},
                    u'x8': 1L, u'x9': 0.0, u'x16': u'apple',
                    u'x11': {u'rExpectedHetFrequency': 0.5, u'pHWE': 0.5},
                    u'x2': [3, 4, 13, 14], u'x3': 3, u'x1': [6, 26], u'x6': 39L, u'x7': 2L, u'x4': 13, u'x5': 16}

        self.assertDictEqual(result, expected)

    def test_errors(self):
        schema = TStruct(['status', 'gt', 'qPheno'],
                         [TInt32(), TCall(), TInt32()])

        rows = [{'status': 0, 'gt': Call(0), 'qPheno': 3},
                {'status': 0, 'gt': Call(1), 'qPheno': 13},
                {'status': 1, 'gt': Call(1), 'qPheno': 20}]

        kt = Table.parallelize(rows, schema)

        def f():
            kt.a = 5

        self.assertRaises(NotImplementedError, f)

    def test_joins(self):
        kt = Table.range(1).drop('index')
        kt = kt.annotate(a='foo')

        kt1 = Table.range(1).drop('index')
        kt1 = kt1.annotate(a='foo', b='bar').key_by('a')

        kt2 = Table.range(1).drop('index')
        kt2 = kt2.annotate(b='bar', c='baz').key_by('b')

        kt3 = Table.range(1).drop('index')
        kt3 = kt3.annotate(c='baz', d='qux').key_by('c')

        kt4 = Table.range(1).drop('index')
        kt4 = kt4.annotate(d='qux', e='quam').key_by('d')

        ktr = kt.annotate(e=kt4[kt3[kt2[kt1[kt.a].b].c].d].e)
        self.assertTrue(ktr.aggregate(result=agg.collect(ktr.e)).result == ['quam'])

        self.assertEqual(kt.filter(kt4[kt3[kt2[kt1[kt.a].b].c].d].e == 'quam').count(), 1)

        m = hc.import_vcf('src/test/resources/sample.vcf')
        vkt = m.rows_table()
        vkt = vkt.select(vkt.v, vkt.qual)
        vkt = vkt.annotate(qual2=m[vkt.v, :].qual)
        self.assertTrue(vkt.filter(vkt.qual != vkt.qual2).count() == 0)

        m2 = m.annotate_rows(qual2=vkt[m.v].qual)
        self.assertTrue(m2.filter_rows(m2.qual != m2.qual2).count_rows() == 0)

        m3 = m.annotate_rows(qual2=m[m.v, :].qual)
        self.assertTrue(m3.filter_rows(m3.qual != m3.qual2).count_rows() == 0)

        kt = Table.range(1)
        kt = kt.annotate_globals(foo=5)

        kt2 = Table.range(1)

        kt2 = kt2.annotate_globals(kt_foo=kt[:].foo)
        self.assertEqual(kt2.globals().kt_foo, 5)

    def test_drop(self):
        kt = Table.range(10)
        kt = kt.annotate(sq=kt.index ** 2, foo='foo', bar='bar')

        self.assertEqual(kt.drop('index', 'foo').columns, ['sq', 'bar'])
        self.assertEqual(kt.drop(kt['index'], kt['foo']).columns, ['sq', 'bar'])


class MatrixTests(unittest.TestCase):
    def get_vds(self, min_partitions=None):
        test_resources = 'src/test/resources/'
        return hc.import_vcf(test_resources + "sample.vcf", min_partitions=min_partitions)

    def testConversion(self):
        vds = self.get_vds()
        vds_old = vds.to_hail1()
        vds_new = vds_old.to_hail2()
        self.assertTrue(vds._same(vds_new))
        vds_old2 = vds_new.to_hail1()
        self.assertTrue(vds_old.same(vds_old2))

    def test_update(self):
        vds = self.get_vds()
        vds = vds.select_entries(dp=vds.DP, gq=vds.GQ)
        self.assertTrue(schema_eq(vds.entry_schema, TStruct(['dp', 'gq'], [TInt32(), TInt32()])))

    def test_annotate(self):
        vds = self.get_vds()
        vds = vds.annotate_globals(foo=5)

        new_global_schema = vds.global_schema
        self.assertEqual(new_global_schema, TStruct(['foo'], [TInt32()]))

        orig_variant_schema = vds.row_schema
        vds = vds.annotate_rows(x1=agg.count(),
                                x2=agg.fraction(False),
                                x3=agg.count_where(True),
                                x4=vds.info.AC + vds.foo)

        expected_fields = [(fd.name, fd.typ) for fd in orig_variant_schema.fields] + \
                          [('x1', TInt64()),
                           ('x2', TFloat64()),
                           ('x3', TInt64()),
                           ('x4', TArray(TInt32()))]

        self.assertTrue(orig_variant_schema, TStruct(*[list(x) for x in zip(*expected_fields)]))

        vds = vds.annotate_cols(apple=6)
        vds = vds.annotate_cols(y1=agg.count(),
                                y2=agg.fraction(False),
                                y3=agg.count_where(True),
                                y4=vds.foo + vds.apple)

        expected_schema = TStruct(['apple', 'y1', 'y2', 'y3', 'y4'],
                                  [TInt32(), TInt64(), TFloat64(), TInt64(), TInt32()])

        self.assertTrue(schema_eq(vds.col_schema, expected_schema),
                        "expected: " + str(vds.col_schema) + "\nactual: " + str(expected_schema))

        vds = vds.select_entries(z1=vds.x1 + vds.foo,
                                 z2=vds.x1 + vds.y1 + vds.foo)
        self.assertTrue(schema_eq(vds.entry_schema, TStruct(['z1', 'z2'], [TInt64(), TInt64()])))

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

        qv = vds.aggregate_rows(x=agg.count()).x
        qs = vds.aggregate_cols(x=agg.count()).x
        qg = vds.aggregate_entries(x=agg.count()).x

        self.assertEqual(qv, 346)
        self.assertEqual(qs, 100)
        self.assertEqual(qg, qv * qs)

        qvs = vds.aggregate_rows(x=agg.collect(vds.v.contig),
                                 y=agg.collect(vds.x1))

        qss = vds.aggregate_cols(x=agg.collect(vds.s),
                                 y=agg.collect(vds.y1))

        qgs = vds.aggregate_entries(x=agg.collect(agg.filter(False, vds.y1)),
                                    y=agg.collect(agg.filter(functions.rand_bool(0.1), vds.GT)))

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

    def test_joins(self):
        vds = self.get_vds().select_rows(x1=1, y1=1)
        vds2 = vds.select_rows(x2=1, y2=2)
        vds2 = vds2.select_cols(c1 = 1, c2 = 2)

        vds = vds.annotate_rows(y2 = vds2[vds.v, :].y2)
        vds = vds.annotate_cols(c2 = vds2[:, vds.s].c2)

        vds = vds.annotate_rows(y2 = vds2[functions.parse_variant(functions.to_str(vds.v)), :].y2)
        vds = vds.annotate_cols(c2 = vds2[:, functions.to_str(vds.s)].c2)

        rt = vds.rows_table()
        ct = vds.cols_table()

        self.assertTrue(rt.forall(rt.y2 == 2))
        self.assertTrue(ct.forall(ct.c2 == 2))

    def test_naive_coalesce(self):
        vds = self.get_vds(min_partitions=8)
        self.assertEqual(vds.num_partitions(), 8)
        repart = vds.naive_coalesce(2)
        self.assertTrue(vds._same(repart))

    def tests_unions(self):
        dataset = hc.import_vcf('src/test/resources/sample2.vcf')

        # test union_rows
        ds1 = dataset.filter_rows(dataset.v.start % 2 == 1)
        ds2 = dataset.filter_rows(dataset.v.start % 2 == 0)

        datasets = [ds1, ds2]
        r1 = ds1.union_rows(ds2)
        r2 = MatrixTable.union_rows(*datasets)

        self.assertTrue(r1._same(r2))

        # test union_cols
        ds = dataset.union_cols(dataset).union_cols(dataset)
        for s, count in ds.aggregate_cols(counts=agg.counter(ds.s)).counts.items():
            self.assertEqual(count, 3)

class FunctionsTests(unittest.TestCase):
    def test(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(),
                          TArray(TInt32()), TArray(TStruct(['x', 'y', 'z'], [TInt32(), TInt32(), TString()])),
                          TStruct(['a', 'b', 'c'], [TInt32(), TInt32(), TString()]),
                          TBoolean(), TStruct(['x', 'y', 'z'], [TInt32(), TInt32(), TString()])])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5,
                 'e': "hello", 'f': [1, 2, 3],
                 'g': [Struct(x=1, y=5, z='banana')],
                 'h': Struct(a=5, b=3, c='winter'),
                 'i': True,
                 'j': Struct(x=3, y=2, z='summer')}]

        kt = Table.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            chisq=functions.chisq(kt.a, kt.b, kt.c, kt.d),
            combvar=functions.combine_variants(Variant.parse("1:2:A:T"), Variant.parse("1:2:A:C")),
            ctt=functions.ctt(kt.a, kt.b, kt.c, kt.d, 5),
            Dict=functions.Dict([kt.a, kt.b], [kt.c, kt.d]),
            dpois=functions.dpois(4, kt.a),
            drop=functions.drop(kt.h, 'b', 'c'),
            exp=functions.exp(kt.c),
            fet=functions.fisher_exact_test(kt.a, kt.b, kt.c, kt.d),
            gt_index=functions.gt_index(kt.a, kt.b),
            hwe=functions.hardy_weinberg_p(1, 2, 1),
            index=functions.index(kt.g, 'z'),
            is_defined=functions.is_defined(kt.i),
            is_missing=functions.is_missing(kt.i),
            is_nan=functions.is_nan(kt.a.to_float64()),
            json=functions.json(kt.g),
            log=functions.log(kt.a.to_float64(), kt.b.to_float64()),
            log10=functions.log10(kt.c.to_float64()),
            merge=functions.merge(kt.h, kt.j),
            or_else=functions.or_else(kt.a, 5),
            or_missing=functions.or_missing(kt.i, kt.j),
            pchisqtail=functions.pchisqtail(kt.a.to_float64(), kt.b.to_float64()),
            pcoin=functions.rand_bool(0.5),
            pnorm=functions.pnorm(0.2),
            pow=2.0 ** kt.b,
            ppois=functions.ppois(kt.a.to_float64(), kt.b.to_float64()),
            qchisqtail=functions.qchisqtail(kt.a.to_float64(), kt.b.to_float64()),
            range=functions.range(0, 5, kt.b),
            rnorm=functions.rand_norm(0.0, kt.b),
            rpois=functions.rand_pois(kt.a),
            runif=functions.rand_unif(kt.b, kt.a),
            select=functions.select(kt.h, 'c', 'b'),
            sqrt=functions.sqrt(kt.a),
            to_str=[functions.to_str(5), functions.to_str(kt.a), functions.to_str(kt.g)],
            where=functions.cond(kt.i, 5, 10)
        ).to_hail1().take(1)[0])

        # print(result) # Fixme: Add asserts


class ColumnTests(unittest.TestCase):
    def test_operators(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = Table.parallelize(rows, schema)

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
        ).to_hail1().take(1)[0])

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

        self.assertDictEqual(result, expected)

    def test_array_column(self):
        schema = TStruct(['a'], [TArray(TInt32())])
        rows = [{'a': [1, 2, 3]}]
        kt = Table.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            x1=kt.a[0],
            x2=kt.a[2],
            x3=kt.a[:],
            x4=kt.a[1:2],
            x5=kt.a[-1:2],
            x6=kt.a[:2]
        ).to_hail1().take(1)[0])

        expected = {'a': [1, 2, 3], 'x1': 1, 'x2': 3, 'x3': [1, 2, 3],
                    'x4': [2], 'x5': [], 'x6': [1, 2]}

        self.assertDictEqual(result, expected)

    def test_dict_column(self):
        schema = TStruct(['x'], [TFloat64()])
        rows = [{'x': 2.0}]
        kt = Table.parallelize(rows, schema)

        kt = kt.annotate(a=functions.Dict(['cat', 'dog'], [3, 7]))

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
        ).to_hail1().take(1)[0])

        expected = {'a': {'cat': 3, 'dog': 7}, 'x': 2.0, 'x1': 3, 'x2': 7, 'x3': False,
                    'x4': False, 'x5': {'cat', 'dog'}, 'x6': ['cat', 'dog'],
                    'x7': [3, 7], 'x8': 2, 'x9': {'cat': 6.0, 'dog': 14.0}}

        self.assertDictEqual(result, expected)

    def test_numeric_conversion(self):
        schema = TStruct(['a', 'b', 'c', 'd'], [TFloat64(), TFloat64(), TInt32(), TInt64()])
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
        kt = Table.parallelize(rows, schema)

        kt = kt.annotate(x1=[1.0, kt.a, 1, long(1)],
                         x2=[1, 1.0],
                         x3=[kt.a, kt.c],
                         x4=[kt.c, kt.d],
                         x5=[1, kt.c, long(1)])

        expected_schema = {'a': TFloat64(), 'b': TFloat64(), 'c': TInt32(), 'd': TInt64(),
                           'x1': TArray(TFloat64()), 'x2': TArray(TFloat64()), 'x3': TArray(TFloat64()),
                           'x4': TArray(TInt64()), 'x5': TArray(TInt64())}

        self.assertTrue(all([expected_schema[fd.name] == fd.typ for fd in kt.schema.fields]))

    def test_constructors(self):
        rg = GenomeReference("foo", ["1"], {"1": 100})

        schema = TStruct(['a', 'b', 'c', 'd'], [TFloat64(), TFloat64(), TInt32(), TInt64()])
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
        kt = Table.parallelize(rows, schema)

        kt = kt.annotate(v1=functions.parse_variant("1:500:A:T", reference_genome=rg),
                         v2=functions.variant("1", 23, "A", ["T"], reference_genome=rg),
                         v3=functions.variant("1", 23, "A", ["T", "G"], reference_genome=rg),
                         l1=functions.parse_locus("1:51"),
                         l2=functions.locus("1", 51, reference_genome=rg),
                         i1=functions.parse_interval("1:51-56", reference_genome=rg),
                         i2=functions.interval(functions.locus("1", 51, reference_genome=rg),
                                               functions.locus("1", 56, reference_genome=rg)))

        expected_schema = {'a': TFloat64(), 'b': TFloat64(), 'c': TInt32(), 'd': TInt64(), 'v1': TVariant(rg),
                           'v2': TVariant(rg), 'v3': TVariant(rg), 'l1': TLocus(), 'l2': TLocus(rg),
                           'i1': TInterval(TLocus(rg)), 'i2': TInterval(TLocus(rg))}

        self.assertTrue(all([expected_schema[fd.name] == fd.typ for fd in kt.schema.fields]))
