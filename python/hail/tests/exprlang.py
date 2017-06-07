"""
Unit tests for Hail.
"""
from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext, KeyTable, VariantDataset
from hail.representation import *
from hail.representation.annotations import *
from hail.expr.functions import *

from hail.types import *
from hail.java import *
from hail.keytable import asc, desc
from hail.expr import NewKeyTable
from hail.utils import *
import time
from hail.keytable import desc

hc = None

def setUpModule():
    global hc
    hc = HailContext(master='local[1]')  # master = 'local[2]')

def tearDownModule():
    global hc
    hc.stop()
    hc = None

def schema_eq(x, y):
    x_fds = dict([(fd.name, fd.typ) for fd in x.fields])
    y_fds = dict([(fd.name, fd.typ) for fd in y.fields])
    return x_fds == y_fds


class KeyTableTests(unittest.TestCase):
    def test_annotate(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt(), TInt(), TInt(), TInt(), TString(), TArray(TInt())])

        rows = [{'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.from_py(hc, rows, schema)

        result1 = to_dict(kt.annotate(foo = kt.a + 1,
                                      foo2 = kt.a).toKeyTable().take(1)[0])

        self.assertEqual(result1, {'a': 4,
                                   'b': 1,
                                   'c': 3,
                                   'd': 5,
                                   'e': "hello",
                                   'f': [1, 2, 3],
                                   'foo': 5,
                                   'foo2': 4})

        result2 = to_dict(kt.annotate(**{'a.foo': 5,
                                         'b.x': "hello",
                                         'b.y': 23,
                                         'b.z': True,
                                         'b.q.hello': [1, 2, 3]}
                                      ).toKeyTable().take(1)[0])

        self.assertEqual(result2, {'a': {'foo': 5},
                                   'b': {'x': "hello", 'y': 23, 'z': True, 'q': {'hello': [1, 2, 3]}},
                                   'c': 3,
                                   'd': 5,
                                   'e': "hello",
                                   'f': [1, 2, 3]})

        result3 = to_dict(kt.annotate(
            x1 = kt.f.map(lambda x: x * 2),
            x2 = kt.f.map(lambda x: [x]).flatMap(lambda x: [x, x + 1]), # This is failing. return type of column from lambda in map step is wrong
            x3 = kt.f.min(),
            x4 = kt.f.max(),
            x5 = kt.f.sum(),
            x6 = kt.f.product(),
            x7 = kt.f.count(),
            x8 = kt.f.filter(lambda x: x == 3),
            x9 = kt.f.fraction(lambda x: x == 1),
            x10 = kt.f.collect(),
            x11 = kt.f.map(lambda x: x.to_double()).stats().mean
        ).toKeyTable().take(1)[0])

        self.assertEqual(result3, {'a': 4,
                                   'b': 1,
                                   'c': 3,
                                   'd': 5,
                                   'e': "hello",
                                   'f': [1, 2, 3],
                                   'x1': [2, 4, 6], 'x2': [1, 2, 2, 3, 3, 4],
                                   'x3': 1, 'x4': 3, 'x5': 6, 'x6': 6, 'x7': 3, 'x8': [3],
                                   'x9': 1.0 / 3, 'x10': [1, 2, 3], 'x11': 2.0})
        #
        # print(result)

    def test_query(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt(), TInt(), TInt(), TInt(), TString(), TArray(TInt())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.from_py(hc, rows, schema)
        kt_agg = kt.aggregate()
        q1, q2 = kt_agg.query([kt_agg.b.sum(), kt_agg.b.count()])
        q3 = kt_agg.query(kt_agg.e.collect())

        self.assertEqual(q1, 8)
        self.assertEqual(q2, 3)
        self.assertEqual(q3, ["hello", "cat", "dog"])

    def test_filter(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt(), TInt(), TInt(), TInt(), TString(), TArray(TInt())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.from_py(hc, rows, schema)

        self.assertEqual(kt.filter(kt.a == 4).count(), 2)
        self.assertEqual(kt.filter((kt.d == -1) | (kt.c == 20) | (kt.e == "hello")).count(), 3)
        self.assertEqual(kt.filter((kt.c != 20) & (kt.a == 4)).count(), 1)

    def test_select(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt(), TInt(), TInt(), TInt(), TString(), TArray(TInt())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.from_py(hc, rows, schema)

        self.assertEqual(kt.select('a', 'e').columns, ['a', 'e'])
        self.assertEqual(kt.select(*['a', 'e']).columns, ['a', 'e'])

    def test_operators(self):
        schema = TStruct(['a', 'b', 'c', 'd', 'e', 'f'],
                         [TInt(), TInt(), TInt(), TInt(), TString(), TArray(TInt())])

        rows = [{'a':4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
                {'a':0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
                {'a':4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]}]

        kt = NewKeyTable.from_py(hc, rows, schema)

        result = to_dict(kt.annotate(
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
        ).toKeyTable().take(1)[0])

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

        self.assertEqual(result, expected)

    def test_aggregate(self):
        schema = TStruct(['status', 'gt', 'qPheno'],
                         [TInt(), TGenotype(), TInt()])

        rows = [{'status':0, 'gt': Genotype(0), 'qPheno': 3},
                {'status':0, 'gt': Genotype(1), 'qPheno': 13},
                {'status':1, 'gt': Genotype(1), 'qPheno': 20}]

        kt = NewKeyTable.from_py(hc, rows, schema)

        g = kt.group_by(status = kt.status)
        result = to_dict(g.aggregate_by_key(
            x1 = g.qPheno.map(lambda x: x * 2).collect(),
            x2 = g.qPheno.map(lambda x: [x]).flatMap(lambda x: [x, x + 1]).collect(), # This is failing. return type of column from lambda in map step is wrong
            x3 = g.qPheno.min(),
            x4 = g.qPheno.max(),
            x5 = g.qPheno.sum(),
            x6 = g.qPheno.map(lambda x: x.to_long()).product(),
            x7 = g.qPheno.count(),
            x8 = g.qPheno.filter(lambda x: x == 3).count(),
            x9 = g.qPheno.fraction(lambda x: x == 1),
            x10 = g.qPheno.map(lambda x: x.to_double()).stats(),
            x11 = g.gt.hardy_weinberg(),
            x12 = g.gt.info_score(),
            x13 = g.gt.inbreeding(lambda x: 0.1),
            x14 = g.gt.call_stats(lambda g: Variant("1", 10000, "A", "T"))
        ).toKeyTable().take(1)[0])

        expected = {'status': 0, 'x1': [6, 26], 'x2': [3, 4, 13, 14],
                    'x3': 3, 'x4': 13, 'x5': 16, 'x6': 39, 'x7': 2, 'x8': 1,
                    'x9': 0.0, 'x10': {'mean': 8, 'stdev': 5, 'min': 3, 'max': 13, 'nNotMissing': 2, 'sum': 16},
                    'x11': {'rExpectedHetFrequency': 1.0, 'pHWE': 0.5},
                    'x12': {'score': None, 'nIncluded': 0},
                    'x13': {'nCalled': 1, 'expectedHoms': 0.82, 'Fstat': -4.5555555555555545, 'nTotal': 2, 'observedHoms': 0},
                    'x14': {'AC': [1, 1], "AF": [0.5, 0.5], "GC": [0, 1, 0], "AN": 2}}

        self.assertEqual(result, expected)

class DatasetTests(unittest.TestCase):
    def get_vds(self):
        test_resources = 'src/test/resources/'
        return hc.import_vcf(test_resources + "sample.vcf").toNewVariantDataset()

    def test_with(self):
        vds = self.get_vds()
        vds = vds.with_global(foo = 5)

        new_global_schema = vds.global_schema
        self.assertEqual(new_global_schema, TStruct(['foo'], [TInt()]))

        orig_variant_schema = vds.variant_schema
        vds = (vds.with_variants(x1 = vds.gs.count(),
                              x2 = vds.gs.fraction(lambda g: False),
                              x3 = vds.gs.filter(lambda g: True).count(),
                              x4 = vds.va.info.AC + vds.globals.foo)
                 .with_alleles(propagate_gq=False, a1 = vds.gs.count()))

        expected_fields = [(fd.name, fd.typ) for fd in orig_variant_schema.fields] + \
                          [('x1', TLong()),
                           ('x2', TDouble()),
                           ('x3', TLong()),
                           ('x4', TArray(TInt())),
                           ('a1', TLong())]

        self.assertTrue(orig_variant_schema, TStruct(*[list(x) for x in zip(*expected_fields)]))

        vds = vds.with_samples(apple = 6)
        vds = vds.with_samples(x1 = vds.gs.count(),
                             x2 = vds.gs.fraction(lambda g: False),
                             x3 = vds.gs.filter(lambda g: True).count(),
                             x4 = vds.globals.foo + vds.sa.apple)

        expected_schema = TStruct(['apple','x1', 'x2', 'x3', 'x4'],
                                  [TInt(), TLong(), TDouble(), TLong(), TInt()])

        self.assertTrue(schema_eq(vds.sample_schema, expected_schema),
                        "expected: " + str(vds.sample_schema) + "\nactual: " + str(expected_schema))

        vds = vds.with_genotypes(x1 = vds.va.x1 + vds.globals.foo,
                                 x2 = vds.va.x1 + vds.sa.x1 + vds.globals.foo)
        self.assertTrue(schema_eq(vds.genotype_schema, TStruct(['x1', 'x2'], [TLong(), TLong()])))

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

    def test_aggregate_by_key(self):
        vds = self.get_vds()

        vds_grouped = vds.group_by(s = vds.s, v = vds.v)
        vds_grouped.aggregate_by_key(nHet = vds_grouped.gs.filter(lambda g: g.isHet()).count())

