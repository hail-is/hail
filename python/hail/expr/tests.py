"""
Unit tests for Hail.
"""
from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext
from hail.expr.functions import *
from hail.htypes import *
from hail.expr import NewKeyTable
from hail.expr.column import VariantColumn, LocusColumn, IntervalColumn, GenotypeColumn

hc = None


def setUpModule():
    global hc
    hc = HailContext(master='local[2]')


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


class KeyTableTests(unittest.TestCase):
    def test_conversion(self):
        test_resources = 'src/test/resources'
        kt_old = hc.import_table(test_resources + '/sampleAnnotations.tsv', impute=True).key_by('Sample')
        kt_new = kt_old._to_new_keytable()
        kt_old2 = kt_new._to_old_keytable()
        self.assertListEqual(kt_new.columns, ['Sample', 'Status', 'qPhen'])
        self.assertTrue(kt_old.same(kt_old2))
        
    def test_annotate(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.parallelize(rows, schema)

        result1 = convert_struct_to_dict(kt.annotate(foo = kt.a + 1,
                                                     foo2 = kt.a)._to_old_keytable().take(1)[0])

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
                                                     )._to_old_keytable().take(1)[0])

        self.assertDictEqual(result2, {'a': {'foo': 5},
                                   'b': {'x': "hello", 'y': 23, 'z': True, 'q': {'hello': [1, 2, 3]}},
                                   'c': 3,
                                   'd': 5,
                                   'e': "hello",
                                   'f': [1, 2, 3]})

        result3 = convert_struct_to_dict(kt.annotate(
            x1 = kt.f.map(lambda x: x * 2),
            x2 = kt.f.map(lambda x: [x, x + 1]).flat_map(lambda x: x),
            x3 = kt.f.min(),
            x4 = kt.f.max(),
            x5 = kt.f.sum(),
            x6 = kt.f.product(),
            x7 = kt.f.length(),
            x8 = kt.f.filter(lambda x: x == 3),
            x9 = kt.f.tail(),
            x10 = kt.f[:],
            x11 = kt.f[1:2],
            x12 = kt.f.map(lambda x: [x, x + 1]),
            x13 = kt.f.map(lambda x: [[x, x + 1], [x + 2]]).flat_map(lambda x: x),
            x14 = where(kt.a < kt.b, kt.c, Column.null(TInt32()))
        )._to_old_keytable().take(1)[0])

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
                                   'x14': None})

    def test_query(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.parallelize(rows, schema)
        kt_agg = kt.aggregate()
        q1, q2 = kt_agg.query([kt_agg.b.sum(), kt_agg.b.count()])
        q3 = kt_agg.query(kt_agg.e.collect())

        self.assertEqual(q1, 8)
        self.assertEqual(q2, 3)
        self.assertEqual(set(q3), set(["hello", "cat", "dog"]))

    def test_filter(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.parallelize(rows, schema)

        self.assertEqual(kt.filter(kt.a == 4).count(), 2)
        self.assertEqual(kt.filter((kt.d == -1) | (kt.c == 20) | (kt.e == "hello")).count(), 3)
        self.assertEqual(kt.filter((kt.c != 20) & (kt.a == 4)).count(), 1)

    def test_select(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.parallelize(rows, schema)

        self.assertEqual(kt.select('a', 'e').columns, ['a', 'e'])
        self.assertEqual(kt.select(*['a', 'e']).columns, ['a', 'e'])

    def test_aggregate(self):
        schema = TStruct(['status', 'gt', 'qPheno'],
                         [TInt32(), TGenotype(), TInt32()])

        rows = [{'status':0, 'gt': Genotype(0), 'qPheno': 3},
                {'status':0, 'gt': Genotype(1), 'qPheno': 13},
                {'status':1, 'gt': Genotype(1), 'qPheno': 20}]

        kt = NewKeyTable.parallelize(rows, schema)

        g = kt.group_by(status = kt.status)
        result = convert_struct_to_dict(g.aggregate_by_key(
            x1 = g.qPheno.map(lambda x: x * 2).collect(),
            x2 = g.qPheno.flat_map(lambda x: [x, x + 1]).collect(),
            x3 = g.qPheno.min(),
            x4 = g.qPheno.max(),
            x5 = g.qPheno.sum(),
            x6 = g.qPheno.map(lambda x: x.to_int64()).product(),
            x7 = g.qPheno.count(),
            x8 = g.qPheno.filter(lambda x: x == 3).count(),
            x9 = g.qPheno.fraction(lambda x: x == 1),
            x10 = g.qPheno.map(lambda x: x.to_float64()).stats(),
            x11 = g.gt.hardy_weinberg(),
            x12 = g.gt.map(lambda x: x.gp).info_score(),
            x13 = g.gt.inbreeding(lambda x: 0.1),
            x14 = g.gt.call_stats(lambda g: Variant("1", 10000, "A", "T")),
            x15 = g.gt.map(lambda g: Struct({'a': 5, 'b': "foo", 'c': Struct({'banana': 'apple'})})).collect()[0],
            x16 = (g.gt.map(lambda g: Struct({'a': 5, 'b': "foo", 'c': Struct({'banana': 'apple'})}))
                   .map(lambda s: s.c.banana).collect()[0]),
            num_partitions=5
        )._to_old_keytable().take(1)[0])

        expected = {'status': 0, 'x1': [6, 26], 'x2': [3, 4, 13, 14],
                    'x3': 3, 'x4': 13, 'x5': 16, 'x6': 39, 'x7': 2, 'x8': 1,
                    'x9': 0.0, 'x10': {'mean': 8, 'stdev': 5, 'min': 3, 'max': 13, 'nNotMissing': 2, 'sum': 16},
                    'x11': {'rExpectedHetFrequency': 1.0, 'pHWE': 0.5},
                    'x12': {'score': None, 'nIncluded': 0},
                    'x13': {'nCalled': 1, 'expectedHoms': 0.82, 'Fstat': -4.5555555555555545, 'nTotal': 2, 'observedHoms': 0},
                    'x14': {'AC': [1, 1], "AF": [0.5, 0.5], "GC": [0, 1, 0], "AN": 2},
                    'x15': {'a': 5, 'b': 'foo', 'c': {'banana': 'apple'}},
                    'x16': 'apple'}

        self.assertDictEqual(result, expected)


class DatasetTests(unittest.TestCase):
    def get_vds(self):
        test_resources = 'src/test/resources/'
        return hc.import_vcf(test_resources + "sample.vcf")._to_new_variant_dataset()

    def test_update(self):
        vds = self.get_vds()
        vds = vds.update_genotypes(lambda g: Struct({'dp': vds.g.dp, 'gq': vds.g.gq}))
        vds_old = vds._to_old_variant_dataset()
        self.assertTrue(schema_eq(vds_old.genotype_schema, TStruct(['dp', 'gq'], [TInt32(), TInt32()])))

    def test_with(self):
        vds = self.get_vds()
        vds = vds.with_global(foo = 5)

        new_global_schema = vds.global_schema
        self.assertEqual(new_global_schema, TStruct(['foo'], [TInt32()]))

        orig_variant_schema = vds.variant_schema
        vds = (vds.with_variants(x1 = vds.gs.count(),
                              x2 = vds.gs.fraction(lambda g: False),
                              x3 = vds.gs.filter(lambda g: True).count(),
                              x4 = vds.va.info.AC + vds.globals.foo)
                 .with_alleles(propagate_gq=False, a1 = vds.gs.count()))

        expected_fields = [(fd.name, fd.typ) for fd in orig_variant_schema.fields] + \
                          [('x1', TInt64()),
                           ('x2', TFloat64()),
                           ('x3', TInt64()),
                           ('x4', TArray(TInt32())),
                           ('a1', TInt64())]

        self.assertTrue(orig_variant_schema, TStruct(*[list(x) for x in zip(*expected_fields)]))

        vds = vds.with_samples(apple = 6)
        vds = vds.with_samples(x1 = vds.gs.count(),
                             x2 = vds.gs.fraction(lambda g: False),
                             x3 = vds.gs.filter(lambda g: True).count(),
                             x4 = vds.globals.foo + vds.sa.apple)

        expected_schema = TStruct(['apple','x1', 'x2', 'x3', 'x4'],
                                  [TInt32(), TInt64(), TFloat64(), TInt64(), TInt32()])

        self.assertTrue(schema_eq(vds.sample_schema, expected_schema),
                        "expected: " + str(vds.sample_schema) + "\nactual: " + str(expected_schema))

        vds = vds.with_genotypes(x1 = vds.va.x1 + vds.globals.foo,
                                 x2 = vds.va.x1 + vds.sa.x1 + vds.globals.foo)
        self.assertTrue(schema_eq(vds.genotype_schema, TStruct(['x1', 'x2'], [TInt64(), TInt64()])))

    def test_filter(self):
        vds = self.get_vds()

        vds = (vds
               .with_global(foo = 5)
               .with_variants(x1 = vds.gs.count())
               .with_samples(x1 = vds.gs.count())
               .with_genotypes(x1 = vds.g.dp))

        (vds
         .filter_variants((vds.va.x1 == 5) & (vds.gs.count() == 3) & (vds.globals.foo == 2))
         .filter_samples((vds.sa.x1 == 5) & (vds.gs.count() == 3) & (vds.globals.foo == 2), keep=False)
         .filter_genotypes((vds.va.x1 == 5) & (vds.sa.x1 == 5) & (vds.globals.foo == 2) & (vds.g.x1 != 3))
         .count_variants())

    def test_query(self):
        vds = self.get_vds()

        vds = (vds
               .with_global(foo = 5)
               .with_variants(x1 = vds.gs.count())
               .with_samples(x1 = vds.gs.count())
               .with_genotypes(x1 = vds.g.dp))

        vds_agg = vds.aggregate()
        qv = vds_agg.query_variants(vds_agg.variants.map(lambda v: vds_agg.v).count())
        qs = vds_agg.query_samples(vds_agg.samples.map(lambda s: vds_agg.s).count())
        qg = vds_agg.query_genotypes(vds_agg.gs.map(lambda g: vds.g).count())

        self.assertEqual(qv, 346)
        self.assertEqual(qs, 100)
        self.assertEqual(qg, qv * qs)

        [qv1, qv2] = vds_agg.query_variants([vds_agg.variants.map(lambda v: vds_agg.v.contig).collect(),
                                             vds_agg.variants.map(lambda v: vds_agg.va.x1).collect()])

        [qs1, qs2] = vds_agg.query_samples([vds_agg.samples.map(lambda s: vds_agg.s).collect(),
                                            vds_agg.samples.map(lambda s: vds_agg.sa.x1).collect()])

        [qg1, qg2] = vds_agg.query_genotypes([vds_agg.gs.filter(lambda g: False).map(lambda g: vds_agg.sa.x1).collect(),
                                              vds_agg.gs.filter(lambda g: pcoin(0.1)).map(lambda g: vds_agg.g).collect()])


class FunctionsTests(unittest.TestCase):
    def test(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(),
                          TArray(TInt32()), TArray(TStruct(['x', 'y', 'z'], [TInt32(), TInt32(), TString()])),
                          TStruct(['a', 'b', 'c'], [TInt32(), TInt32(), TString()]),
                          TBoolean(), TStruct(['x', 'y', 'z'], [TInt32(), TInt32(), TString()])])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5,
                 'e': "hello", 'f': [1, 2, 3],
                 'g': [Struct({'x': 1, 'y': 5, 'z': "banana"})],
                 'h': Struct({'a': 5, 'b': 3, 'c': "winter"}),
                 'i': True,
                 'j': Struct({'x': 3, 'y': 2, 'z': "summer"})}]

        kt = NewKeyTable.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            chisq = chisq(kt.a, kt.b, kt.c, kt.d),
            combvar = combine_variants(Variant.parse("1:2:A:T"), Variant.parse("1:2:A:C")),
            ctt = ctt(kt.a, kt.b, kt.c, kt.d, 5),
            Dict = Dict([kt.a, kt.b], [kt.c, kt.d]),
            dpois = dpois(4, kt.a),
            drop = drop(kt.h, 'b', 'c'),
            exp = exp(kt.c),
            fet = fet(kt.a, kt.b, kt.c, kt.d),
            gt_index = gt_index(kt.a, kt.b),
            gtj = gtj(kt.a),
            gtk = gtk(kt.b),
            hwe = hwe(1, 2, 1),
            index = index(kt.g, 'z'),
            is_defined = is_defined(kt.i),
            is_missing = is_missing(kt.i),
            is_nan = is_nan(kt.a.to_float64()),
            json = json(kt.g),
            log = log(kt.a.to_float64(), kt.b.to_float64()),
            log10 = log10(kt.c.to_float64()),
            merge = merge(kt.h, kt.j),
            or_else = or_else(kt.a, 5),
            or_missing = or_missing(kt.i, kt.j),
            pchisqtail = pchisqtail(kt.a.to_float64(), kt.b.to_float64()),
            pcoin = pcoin(0.5),
            pnorm = pnorm(0.2),
            pow = pow(2.0, kt.b),
            ppois = ppois(kt.a.to_float64(), kt.b.to_float64()),
            qchisqtail = qchisqtail(kt.a.to_float64(), kt.b.to_float64()),
            range = range(0, 5, kt.b),
            rnorm = rnorm(0.0, kt.b),
            rpois = rpois(kt.a),
            runif = runif(kt.b, kt.a),
            select = select(kt.h, 'c', 'b'),
            sqrt = sqrt(kt.a),
            to_str = [to_str(5), to_str(kt.a), to_str(kt.g)],
            where = where(kt.i, 5, 10)
        )._to_old_keytable().take(1)[0])

        # print(result) # Fixme: Add asserts


class ColumnTests(unittest.TestCase):
    def test_operators(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt32(), TInt32(), TInt32(), TInt32(), TString(), TArray(TInt32())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            x1 = kt.a + 5,
            x2 = 5 + kt.a,
            x3 = kt.a + kt.b,
            x4 = kt.a - 5,
            x5 = 5 - kt.a,
            x6 = kt.a - kt.b,
            x7 = kt.a * 5,
            x8 = 5 * kt.a,
            x9 = kt.a * kt.b,
            x10 = kt.a / 5,
            x11 = 5 / kt.a,
            x12 = kt.a / kt.b,
            x13 = -kt.a,
            x14 = +kt.a,
            x15 = kt.a == kt.b,
            x16 = kt.a == 5,
            x17 = 5 == kt.a,
            x18 = kt.a != kt.b,
            x19 = kt.a != 5,
            x20 = 5 != kt.a,
            x21 = kt.a > kt.b,
            x22 = kt.a > 5,
            x23 = 5 > kt.a,
            x24 = kt.a >= kt.b,
            x25 = kt.a >= 5,
            x26 = 5 >= kt.a,
            x27 = kt.a < kt.b,
            x28 = kt.a < 5,
            x29 = 5 < kt.a,
            x30 = kt.a <= kt.b,
            x31 = kt.a <= 5,
            x32 = 5 <= kt.a,
            x33 = (kt.a == 0) & (kt.b == 5),
            x34 = (kt.a == 0) | (kt.b == 5),
            x35 = False,
            x36 = True
        )._to_old_keytable().take(1)[0])

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
        kt = NewKeyTable.parallelize(rows, schema)

        result = convert_struct_to_dict(kt.annotate(
            x1 = kt.a[0],
            x2 = kt.a[2],
            x3 = kt.a[:],
            x4 = kt.a[1:2],
            x5 = kt.a[-1:2],
            x6 = kt.a[:2]
        )._to_old_keytable().take(1)[0])

        expected = {'a': [1, 2, 3], 'x1': 1, 'x2': 3, 'x3': [1, 2, 3],
                    'x4': [2], 'x5': [], 'x6': [1, 2]}

        self.assertDictEqual(result, expected)

    def test_dict_column(self):
        schema = TStruct(['x'], [TFloat64()])
        rows = [{'x': 2.0}]
        kt = NewKeyTable.parallelize(rows, schema)

        kt = kt.annotate(a = Dict(['cat', 'dog'], [3, 7]))

        result = convert_struct_to_dict(kt.annotate(
            x1 = kt.a['cat'],
            x2 = kt.a['dog'],
            x3 = kt.a.contains('rabbit'),
            x4 = kt.a.is_empty(),
            x5 = kt.a.key_set(),
            x6 = kt.a.keys(),
            x7 = kt.a.values(),
            x8 = kt.a.size(),
            x9 = kt.a.map_values(lambda v: v.to_float64())
        )._to_old_keytable().take(1)[0])

        expected = {'a': {'cat': 3, 'dog': 7}, 'x': 2.0, 'x1': 3, 'x2': 7, 'x3': False,
                    'x4': False, 'x5': set(['cat', 'dog']), 'x6': ['cat', 'dog'],
                    'x7': [3, 7], 'x8': 2, 'x9': {'cat': 3.0, 'dog': 7.0}}

        self.assertDictEqual(result, expected)

    def test_numeric_conversion(self):
        schema = TStruct(['a', 'b', 'c', 'd'], [TFloat64(), TFloat64(), TInt32(), TInt64()])
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
        kt = NewKeyTable.parallelize(rows, schema)

        kt = kt.annotate(x1 = [1.0, kt.a, 1, long(1)],
                         x2 = [1, 1.0],
                         x3 = [kt.a, kt.c],
                         x4 = [kt.c, kt.d],
                         x5 = [1, kt.c, long(1)])

        expected_schema = {'a': TFloat64(), 'b': TFloat64(), 'c': TInt32(), 'd': TInt64(),
                           'x1': TArray(TFloat64()), 'x2': TArray(TFloat64()), 'x3': TArray(TFloat64()),
                           'x4': TArray(TInt64()), 'x5': TArray(TInt64())}

        self.assertTrue(all([expected_schema[fd.name] == fd.typ for fd in kt.schema.fields]))

    def test_constructors(self):
        schema = TStruct(['a', 'b', 'c', 'd'], [TFloat64(), TFloat64(), TInt32(), TInt64()])
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': long(5)}]
        kt = NewKeyTable.parallelize(rows, schema)

        kt = kt.annotate(v1 = VariantColumn.parse("1:500:A:T"),
                         v2 = VariantColumn.from_args("1", 23, "A", "T"),
                         v3 = VariantColumn.from_args("1", 23, "A", ["T", "G"]),
                         l1 = LocusColumn.parse("1:51"),
                         l2 = LocusColumn.from_args("1", 51),
                         i1 = IntervalColumn.parse("1:51-56"),
                         i2 = IntervalColumn.from_args("1", 51, 56),
                         i3 = IntervalColumn.from_loci(LocusColumn.from_args("1", 51), LocusColumn.from_args("1", 56)))

        kt = kt.annotate(g1 = GenotypeColumn.dosage_genotype(kt.v1, [0.0, 1.0, 0.0]),
                         g2 = GenotypeColumn.dosage_genotype(kt.v1, [0.0, 1.0, 0.0], call=CallColumn.from_int32(1)),
                         g3 = GenotypeColumn.from_call(CallColumn.from_int32(1)),
                         g4 = GenotypeColumn.pl_genotype(kt.v1, CallColumn.from_int32(1), [6, 7], 13, 20, [20, 0, 1000]))

        expected_schema = {'a': TFloat64(), 'b': TFloat64(), 'c': TInt32(), 'd': TInt64(), 'v1': TVariant(),
                           'v2': TVariant(), 'v3': TVariant(), 'l1': TLocus(), 'l2': TLocus(), 'i1': TInterval(),
                           'i2': TInterval(), 'i3': TInterval(), 'g1': TGenotype(), 'g2': TGenotype(), 'g3': TGenotype(),
                           'g4': TGenotype()}

        self.assertTrue(all([expected_schema[fd.name] == fd.typ for fd in kt.schema.fields]))
