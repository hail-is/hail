import math
import random
from scipy.stats import pearsonr
import unittest

import hail as hl
import hail.expr.aggregators as agg
from hail.expr.types import *
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    def test_key_by_random(self):
        ht = hl.utils.range_table(10, 4)
        ht = ht.annotate(new_key=hl.rand_unif(0, 1))
        ht = ht.key_by('new_key')
        self.assertEqual(ht._force_count(), 10)

    def test_seeded_same(self):

        def test_random_function(rand_f):
            ht = hl.utils.range_table(10, 4)
            sample1 = rand_f()
            ht = ht.annotate(x=sample1, y=sample1, z=rand_f())
            self.assertTrue(ht.aggregate(agg.all((ht.x == ht.y)) & ~agg.all((ht.x == ht.z))))

        test_random_function(lambda: hl.rand_unif(0, 1))
        test_random_function(lambda: hl.rand_bool(0.5))
        test_random_function(lambda: hl.rand_norm(0, 1))
        test_random_function(lambda: hl.rand_pois(1))
        test_random_function(lambda: hl.rand_beta(1, 1))
        test_random_function(lambda: hl.rand_beta(1, 1, 0, 1))
        test_random_function(lambda: hl.rand_gamma(1, 1))
        test_random_function(lambda: hl.rand_cat(hl.array([1, 1, 1, 1])))
        test_random_function(lambda: hl.rand_dirichlet(hl.array([1, 1, 1, 1])))

    def test_seeded_sampling(self):
        sampled1 = hl.utils.range_table(50, 6).filter(hl.rand_bool(0.5))
        sampled2 = hl.utils.range_table(50, 5).filter(hl.rand_bool(0.5))

        set1 = set(sampled1.idx.collect())
        set2 = set(sampled2.idx.collect())
        expected = set1 & set2

        for i in range(10):
            s1 = sampled1.filter(hl.is_defined(sampled2[sampled1.idx]))
            s2 = sampled2.filter(hl.is_defined(sampled1[sampled2.idx]))
            self.assertEqual(set(s1.idx.collect()), expected)
            self.assertEqual(set(s2.idx.collect()), expected)

    def test_order_by_head_optimization_with_randomness(self):
        ht = hl.utils.range_table(10, 6).annotate(x=hl.rand_unif(0, 1))
        expected = sorted(ht.collect(), key=lambda x: x['x'])[:5]
        self.assertEqual(ht.order_by(ht.x).take(5), expected)

    def test_operators(self):
        schema = hl.tstruct(a=hl.tint32, b=hl.tint32, c=hl.tint32, d=hl.tint32, e=hl.tstr, f=hl.tarray(hl.tint32))

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

        for k, v in expected.items():
            if isinstance(v, float):
                self.assertAlmostEqual(v, result[k], msg=k)
            else:
                self.assertEqual(v, result[k], msg=k)

    def test_array_slicing(self):
        schema = hl.tstruct(a=hl.tarray(hl.tint32))
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

    def test_dict_methods(self):
        schema = hl.tstruct(x=hl.tfloat64)
        rows = [{'x': 2.0}]
        kt = hl.Table.parallelize(rows, schema)

        kt = kt.annotate(a={'cat': 3, 'dog': 7})

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
        schema = hl.tstruct(a=hl.tfloat64, b=hl.tfloat64, c=hl.tint32, d=hl.tint32)
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': 5}]
        kt = hl.Table.parallelize(rows, schema)
        kt = kt.annotate(d=hl.int64(kt.d))

        kt = kt.annotate(x1=[1.0, kt.a, 1],
                         x2=[1, 1.0],
                         x3=[kt.a, kt.c],
                         x4=[kt.c, kt.d],
                         x5=[1, kt.c])

        expected_schema = {'a': hl.tfloat64,
                           'b': hl.tfloat64,
                           'c': hl.tint32,
                           'd': hl.tint64,
                           'x1': hl.tarray(hl.tfloat64),
                           'x2': hl.tarray(hl.tfloat64),
                           'x3': hl.tarray(hl.tfloat64),
                           'x4': hl.tarray(hl.tint64),
                           'x5': hl.tarray(hl.tint32)}

        for f, t in kt.row.dtype.items():
            self.assertEqual(expected_schema[f], t)

    def test_genetics_constructors(self):
        rg = hl.ReferenceGenome("foo", ["1"], {"1": 100})

        schema = hl.tstruct(a=hl.tfloat64, b=hl.tfloat64, c=hl.tint32, d=hl.tint32)
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': 5}]
        kt = hl.Table.parallelize(rows, schema)
        kt = kt.annotate(d=hl.int64(kt.d))

        kt = kt.annotate(l1=hl.parse_locus("1:51"),
                         l2=hl.locus("1", 51, reference_genome=rg),
                         i1=hl.parse_locus_interval("1:51-56", reference_genome=rg),
                         i2=hl.interval(hl.locus("1", 51, reference_genome=rg),
                                        hl.locus("1", 56, reference_genome=rg)))

        expected_schema = {'a': hl.tfloat64, 'b': hl.tfloat64, 'c': hl.tint32, 'd': hl.tint64,
                           'l1': hl.tlocus(), 'l2': hl.tlocus(rg),
                           'i1': hl.tinterval(hl.tlocus(rg)), 'i2': hl.tinterval(hl.tlocus(rg))}

        self.assertTrue(all([expected_schema[f] == t for f, t in kt.row.dtype.items()]))

    def test_floating_point(self):
        self.assertEqual(hl.eval(1.1e-15), 1.1e-15)

    def test_bind_multiple(self):
        self.assertEqual(hl.eval(hl.bind(lambda x, y: x * y, 2, 3)), 6)
        self.assertEqual(hl.eval(hl.bind(lambda y: y * 2, 3)), 6)

    def test_bind_placement(self):
        self.assertEqual(hl.eval(5 / hl.bind(lambda x: x, 5)), 1.0)

    def test_matches(self):
        self.assertEqual(hl.eval('\d+'), '\d+')
        string = hl.literal('12345')
        self.assertTrue(hl.eval(string.matches('\d+')))
        self.assertFalse(hl.eval(string.matches(r'\\d+')))

    def test_first_match_in(self):
        string = hl.literal('1:25-100')
        self.assertTrue(hl.eval(string.first_match_in("([^:]*)[:\\t](\\d+)[\\-\\t](\\d+)")) == ['1', '25', '100'])
        self.assertIsNone(hl.eval(string.first_match_in("hello (\w+)!")))

    def test_cond(self):
        self.assertEqual(hl.eval('A' + hl.cond(True, 'A', 'B')), 'AA')

        self.assertEqual(hl.eval(hl.cond(True, hl.struct(), hl.null(hl.tstruct()))), hl.utils.Struct())
        self.assertEqual(hl.eval(hl.cond(hl.null(hl.tbool), 1, 2)), None)
        self.assertEqual(hl.eval(hl.cond(hl.null(hl.tbool), 1, 2, missing_false=True)), 2)

    def test_aggregators(self):
        table = hl.utils.range_table(10)
        r = table.aggregate(hl.struct(x=agg.count(),
                                      y=agg.count_where(table.idx % 2 == 0),
                                      z=agg.filter(table.idx % 2 == 0, agg.count()),
                                      arr_sum=agg.array_sum([1, 2, hl.null(tint32)]),
                                      bind_agg=agg.count_where(hl.bind(lambda x: x % 2 == 0, table.idx)),
                                      mean=agg.mean(table.idx),
                                      mean2=agg.mean(hl.cond(table.idx == 9, table.idx, hl.null(tint32))),
                                      foo=hl.min(3, agg.sum(table.idx))))

        self.assertEqual(r.x, 10)
        self.assertEqual(r.y, 5)
        self.assertEqual(r.z, 5)
        self.assertEqual(r.arr_sum, [10, 20, 0])
        self.assertEqual(r.mean, 4.5)
        self.assertEqual(r.mean2, 9)
        self.assertEqual(r.bind_agg, 5)
        self.assertEqual(r.foo, 3)

        a = hl.literal([1, 2], tarray(tint32))
        self.assertEqual(table.aggregate(agg.filter(True, agg.array_sum(a))), [10, 20])

        r = table.aggregate(hl.struct(fraction_odd=agg.fraction(table.idx % 2 == 0),
                                      lessthan6=agg.fraction(table.idx < 6),
                                      gt6=agg.fraction(table.idx > 6),
                                      assert1=agg.fraction(table.idx > 6) < 0.50,
                                      assert2=agg.fraction(table.idx < 6) >= 0.50))
        self.assertEqual(r.fraction_odd, 0.50)
        self.assertEqual(r.lessthan6, 0.60)
        self.assertEqual(r.gt6, 0.30)
        self.assertTrue(r.assert1)
        self.assertTrue(r.assert2)

    def test_new_aggregator_maps(self):
        t = hl.utils.range_table(10)

        tests = [(agg.filter(t.idx > 7,
                             agg.collect(t.idx + 1).append(0)),
                  [9, 10, 0]),
                 (agg.explode(lambda elt: agg.collect(elt + 1).append(0),
                              hl.cond(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32))),
                  [9, 10, 10, 11, 0]),
                 (agg.explode(lambda elt: agg.explode(lambda elt2: agg.collect(elt2 + 1).append(0),
                                                      [elt, elt + 1]),
                              hl.cond(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32))),
                  [9, 10, 10, 11, 10, 11, 11, 12, 0]),
                 (agg.explode(lambda elt: agg.filter(elt > 8,
                                                     agg.collect(elt + 1).append(0)),
                              hl.cond(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32))),
                  [10, 10, 11, 0]),
                 (agg.filter(t.idx > 7,
                             agg.explode(lambda elt: agg.collect(elt + 1).append(0),
                                         [t.idx, t.idx + 1])),
                  [9, 10, 10, 11, 0]),
                 (agg.group_by(t.idx % 2,
                               hl.array(agg.collect_as_set(t.idx + 1)).append(0)),
                  {0: [1, 3, 5, 7, 9, 0], 1: [2, 4, 6, 8, 10, 0]}),
                 (agg.group_by(t.idx % 3,
                               agg.filter(t.idx > 7,
                                          hl.array(agg.collect_as_set(t.idx + 1)).append(0))),
                  {0: [10, 0], 1: [0], 2: [9, 0]}),
                 (agg.filter(t.idx > 7,
                              agg.group_by(t.idx % 3,
                                            hl.array(agg.collect_as_set(t.idx + 1)).append(0))),
                  {0: [10, 0], 2: [9, 0]}),
                 (agg.group_by(t.idx % 3,
                               agg.explode(lambda elt: agg.collect(elt + 1).append(0),
                                           hl.cond(t.idx > 7,
                                                   [t.idx, t.idx + 1],
                                                   hl.empty_array(hl.tint32)))),
                  {0: [10, 11, 0], 1: [0], 2:[9, 10, 0]}),
                 (agg.explode(lambda elt: agg.group_by(elt % 3,
                                                       agg.collect(elt + 1).append(0)),
                                           hl.cond(t.idx > 7,
                                                   [t.idx, t.idx + 1],
                                                   hl.empty_array(hl.tint32))),
                  {0: [10, 10, 0], 1: [11, 0], 2:[9, 0]})
                 ]
        for aggregation, expected in tests:
            self.assertEqual(t.aggregate(aggregation), expected)

    def test_aggregators_with_randomness(self):
        t = hl.utils.range_table(10)
        res = t.aggregate(agg.filter(hl.rand_bool(0.5), hl.struct(collection=agg.collect(t.idx), sum=agg.sum(t.idx))))
        self.assertEqual(sum(res.collection), res.sum)

    def test_aggregator_scope(self):
        t = hl.utils.range_table(10)
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(agg.explode(lambda elt: agg.sum(elt) + elt, [t.idx, t.idx + 1]))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(agg.filter(t.idx > 7, agg.sum(t.idx) / t.idx))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(agg.group_by(t.idx % 3, agg.sum(t.idx) / t.idx))

        tests = [(agg.filter(t.idx > 7,
                             agg.explode(lambda x: agg.collect(hl.int64(x + 1)),
                                         [t.idx, t.idx + 1]).append(
                                 agg.group_by(t.idx % 3, agg.sum(t.idx))[0])
                             ),
                  [9, 10, 10, 11, 9]),
                 (agg.explode(lambda x:
                              agg.filter(x > 7,
                                         agg.collect(x)
                                         ).extend(agg.group_by(t.idx % 3,
                                                               hl.array(agg.collect_as_set(x)))[0]),
                              [t.idx, t.idx + 1]),
                  [8, 8, 9, 9, 10, 0, 1, 3, 4, 6, 7, 9, 10]),
                 (agg.group_by(t.idx % 3,
                               agg.filter(t.idx > 7,
                                          agg.collect(t.idx)
                                          ).extend(agg.explode(
                                   lambda x: hl.array(agg.collect_as_set(x)),
                                   [t.idx, t.idx + 34]))
                               ),
                  {0: [9, 0, 3, 6, 9, 34, 37, 40, 43],
                   1: [1, 4, 7, 35, 38, 41],
                   2: [8, 2, 5, 8, 36, 39, 42]})
                 ]
        for aggregation, expected in tests:
            self.assertEqual(t.aggregate(aggregation), expected)

    def test_scan(self):
        table = hl.utils.range_table(10)

        t = table.select(scan_count=hl.scan.count(),
                         scan_count_where=hl.scan.count_where(table.idx % 2 == 0),
                         scan_count_where2=hl.scan.filter(table.idx % 2 == 0, hl.scan.count()),
                         arr_sum=hl.scan.array_sum([1, 2, hl.null(tint32)]),
                         bind_agg=hl.scan.count_where(hl.bind(lambda x: x % 2 == 0, table.idx)),
                         mean=hl.scan.mean(table.idx),
                         foo=hl.min(3, hl.scan.sum(table.idx)),
                         fraction_odd=hl.scan.fraction(table.idx % 2 == 0))
        rows = t.collect()
        r = hl.Struct(**{n: [i[n] for i in rows] for n in t.row.keys()})

        self.assertEqual(r.scan_count, [i for i in range(10)])
        self.assertEqual(r.scan_count_where, [(i + 1) // 2 for i in range(10)])
        self.assertEqual(r.scan_count_where2, [(i + 1) // 2 for i in range(10)])
        self.assertEqual(r.arr_sum, [None] + [[i * 1, i * 2, 0] for i in range(1, 10)])
        self.assertEqual(r.bind_agg, [(i + 1) // 2 for i in range(10)])
        self.assertEqual(r.foo, [min(sum(range(i)), 3) for i in range(10)])
        for (x, y) in zip(r.fraction_odd, [None] + [((i + 1)//2)/i for i in range(1, 10)]):
            self.assertAlmostEqual(x, y)

        table = hl.utils.range_table(10)
        r = table.aggregate(hl.struct(x=agg.count()))

        self.assertEqual(r.x, 10)

    def test_scan_group_by(self):
        t = hl.utils.range_table(5)
        t = t.select(group_by=hl.scan.group_by(t.idx % 2 == 0, hl.scan.count()))
        rows = t.collect()
        r = hl.Struct(**{n: [i[n] for i in rows] for n in t.row.keys()})
        self.assertEqual(r.group_by, [{}, {True: 1}, {True: 1, False: 1}, {True: 2, False: 1}, {True: 2, False: 2}])

    def test_aggregators_max_min(self):
        table = hl.utils.range_table(10)
        # FIXME: add boolean when function registry is removed
        for (f, typ) in [(lambda x: hl.int32(x), tint32), (lambda x: hl.int64(x), tint64),
                  (lambda x: hl.float32(x), tfloat32), (lambda x: hl.float64(x), tfloat64)]:
            t = table.annotate(x=-1 * f(table.idx) - 5, y=hl.null(typ))
            r = t.aggregate(hl.struct(max=agg.max(t.x), max_empty=agg.max(t.y),
                                      min=agg.min(t.x), min_empty=agg.min(t.y)))
            self.assertTrue(r.max == -5 and r.max_empty is None and
                            r.min == -14 and r.min_empty is None)

    def test_aggregators_sum_product(self):
        table = hl.utils.range_table(5)
        for (f, typ) in [(lambda x: hl.int32(x), tint32), (lambda x: hl.int64(x), tint64),
                         (lambda x: hl.float32(x), tfloat32), (lambda x: hl.float64(x), tfloat64)]:
            t = table.annotate(x=-1 * f(table.idx) - 1, y=f(table.idx), z=hl.null(typ))
            r = t.aggregate(hl.struct(sum_x=agg.sum(t.x), sum_y=agg.sum(t.y), sum_empty=agg.sum(t.z),
                                      prod_x=agg.product(t.x), prod_y=agg.product(t.y), prod_empty=agg.product(t.z)))
            self.assertTrue(r.sum_x == -15 and r.sum_y == 10 and r.sum_empty == 0 and
                            r.prod_x == -120 and r.prod_y == 0 and r.prod_empty == 1)

    def test_aggregators_hist(self):
        table = hl.utils.range_table(11)
        r = table.aggregate(agg.hist(table.idx - 1, 0, 8, 4))
        self.assertTrue(r.bin_edges == [0, 2, 4, 6, 8] and r.bin_freq == [2, 2, 2, 3] and r.n_smaller == 1 and r.n_larger == 1)

    # Tested against R code
    # y = c(0.22848042, 0.09159706, -0.43881935, -0.99106171, 2.12823289)
    # x = c(0.2575928, -0.3445442, 1.6590146, -1.1688806, 0.5587043)
    # df = data.frame(y, x)
    # fit <- lm(y ~ x, data=df)
    # sumfit = summary(fit)
    # coef = sumfit$coefficients
    # mse = sumfit$sigma
    # r2 = sumfit$r.squared
    # r2adj = sumfit$adj.r.squared
    # f = sumfit$fstatistic
    # p = pf(f[1],f[2],f[3],lower.tail=F)
    def test_aggregators_linreg(self):
        t = hl.Table.parallelize([
            {"y": None, "x": 1.0},
            {"y": 0.0, "x": None},
            {"y": None, "x": None},
            {"y": 0.22848042, "x": 0.2575928},
            {"y": 0.09159706, "x": -0.3445442},
            {"y": -0.43881935, "x": 1.6590146},
            {"y": -0.99106171, "x": -1.1688806},
            {"y": 2.12823289, "x": 0.5587043}
        ], hl.tstruct(y=hl.tfloat64, x=hl.tfloat64), n_partitions=3)
        r = t.aggregate(hl.struct(linreg=hl.agg.linreg(t.y, [1, t.x]))).linreg
        self.assertAlmostEqual(r.beta[0], 0.14069227)
        self.assertAlmostEqual(r.beta[1], 0.32744807)
        self.assertAlmostEqual(r.standard_error[0], 0.59410817)
        self.assertAlmostEqual(r.standard_error[1], 0.61833778)
        self.assertAlmostEqual(r.t_stat[0], 0.23681254)
        self.assertAlmostEqual(r.t_stat[1], 0.52956181)
        self.assertAlmostEqual(r.p_value[0], 0.82805147)
        self.assertAlmostEqual(r.p_value[1], 0.63310173)
        self.assertAlmostEqual(r.multiple_standard_error, 1.3015652)
        self.assertAlmostEqual(r.multiple_r_squared, 0.08548734)
        self.assertAlmostEqual(r.adjusted_r_squared, -0.2193502)
        self.assertAlmostEqual(r.f_stat, 0.2804357)
        self.assertAlmostEqual(r.multiple_p_value, 0.6331017)
        self.assertAlmostEqual(r.n, 5)

        # weighted OLS
        t = t.add_index()
        r = t.aggregate(hl.struct(
            linreg=hl.agg.linreg(t.y, [1, t.x], weight=t.idx))).linreg
        self.assertAlmostEqual(r.beta[0], 0.2339059)
        self.assertAlmostEqual(r.beta[1], 0.4275577)
        self.assertAlmostEqual(r.standard_error[0], 0.6638324)
        self.assertAlmostEqual(r.standard_error[1], 0.6662581)
        self.assertAlmostEqual(r.t_stat[0], 0.3523569)
        self.assertAlmostEqual(r.t_stat[1], 0.6417299)
        self.assertAlmostEqual(r.p_value[0], 0.7478709)
        self.assertAlmostEqual(r.p_value[1], 0.5667139)
        self.assertAlmostEqual(r.multiple_standard_error, 3.26238997)
        self.assertAlmostEqual(r.multiple_r_squared, 0.12070321)
        self.assertAlmostEqual(r.adjusted_r_squared, -0.17239572)
        self.assertAlmostEqual(r.f_stat, 0.41181729)
        self.assertAlmostEqual(r.multiple_p_value, 0.56671386)
        self.assertAlmostEqual(r.n, 5)

    def test_aggregator_downsample(self):
        xs = [2, 6, 4, 9, 1, 8, 5, 10, 3, 7]
        ys = [2, 6, 4, 9, 1, 8, 5, 10, 3, 7]
        label1 = ["2", "6", "4", "9", "1", "8", "5", "10", "3", "7"]
        label2 = ["two", "six", "four", "nine", "one", "eight", "five", "ten", "three", "seven"]
        table = hl.Table.parallelize([hl.struct(x=x, y=y, label1=label1, label2=label2)
                                      for x, y, label1, label2 in zip(xs, ys, label1, label2)])
        r = table.aggregate(agg.downsample(table.x, table.y, label=hl.array([table.label1, table.label2]), n_divisions=10))
        xs = [x for (x, y, l) in r]
        ys = [y for (x, y, l) in r]
        label = [tuple(l) for (x, y, l) in r]
        expected = set([(1.0, 1.0, ('1', 'one')), (2.0, 2.0, ('2', 'two')), (3.0, 3.0, ('3', 'three')),
                        (4.0, 4.0, ('4', 'four')), (5.0, 5.0, ('5', 'five')), (6.0, 6.0, ('6', 'six')),
                        (7.0, 7.0, ('7', 'seven')), (8.0, 8.0, ('8', 'eight')), (9.0, 9.0, ('9', 'nine')),
                        (10.0, 10.0, ('10', 'ten'))])
        for point in zip(xs, ys, label):
            self.assertTrue(point in expected)

    def test_downsample_aggregator_on_empty_table(self):
        ht = hl.utils.range_table(1)
        ht = ht.annotate(y=ht.idx).filter(False)
        r = ht.aggregate(agg.downsample(ht.idx, ht.y, n_divisions=10))
        self.assertTrue(len(r) == 0)

    def test_aggregator_info_score(self):
        gen_file = resource('infoScoreTest.gen')
        sample_file = resource('infoScoreTest.sample')
        truth_result_file = resource('infoScoreTest.result')

        mt = hl.import_gen(gen_file, sample_file=sample_file)
        mt = mt.annotate_rows(info_score = hl.agg.info_score(mt.GP))

        truth = hl.import_table(truth_result_file, impute=True, delimiter=' ', no_header=True, missing='None')
        truth = truth.drop('f1', 'f2').rename({'f0': 'variant', 'f3': 'score', 'f4': 'n_included'})
        truth = truth.transmute(**hl.parse_variant(truth.variant)).key_by('locus', 'alleles')

        computed = mt.rows()

        joined = truth[computed.key]
        computed = computed.select(score = computed.info_score.score,
                                   score_truth = joined.score,
                                   n_included = computed.info_score.n_included,
                                   n_included_truth = joined.n_included)
        violations = computed.filter(
            (computed.n_included != computed.n_included_truth) |
            (hl.abs(computed.score - computed.score_truth) > 1e-3))
        if not violations.count() == 0:
            violations.show()
            self.fail("disagreement between computed info score and truth")

    def test_aggregator_info_score_works_with_bgen_import(self):
        sample_file = resource('random.sample')
        bgen_file = resource('random.bgen')
        hl.index_bgen(bgen_file)
        bgenmt = hl.import_bgen(bgen_file, ['GT', 'GP'], sample_file)
        result = bgenmt.annotate_rows(info=hl.agg.info_score(bgenmt.GP)).rows().take(1)
        result = result[0].info
        self.assertAlmostEqual(result.score, -0.235041090, places=3)
        self.assertEqual(result.n_included, 8)

    def test_aggregator_group_by(self):
        t = hl.Table.parallelize([
            {"cohort": None, "pop": "EUR", "GT": hl.Call([0, 0])},
            {"cohort": None, "pop": "ASN", "GT": hl.Call([0, 1])},
            {"cohort": None, "pop": None, "GT": hl.Call([0, 0])},
            {"cohort": "SIGMA", "pop": "AFR", "GT": hl.Call([0, 1])},
            {"cohort": "SIGMA", "pop": "EUR", "GT": hl.Call([1, 1])},
            {"cohort": "IBD", "pop": "EUR", "GT": None},
            {"cohort": "IBD", "pop": "EUR", "GT": hl.Call([0, 0])},
            {"cohort": "IBD", "pop": None, "GT": hl.Call([0, 1])}
        ], hl.tstruct(cohort=hl.tstr, pop=hl.tstr, GT=hl.tcall), n_partitions=3)

        r = t.aggregate(hl.struct(count=hl.agg.group_by(t.cohort, hl.agg.group_by(t.pop, hl.agg.count_where(hl.is_defined(t.GT)))),
                                  inbreeding=hl.agg.group_by(t.cohort, hl.agg.inbreeding(t.GT, 0.1))))

        expected_count = {None: {'EUR': 1, 'ASN': 1, None: 1},
                    'SIGMA': {'AFR': 1, 'EUR': 1},
                    'IBD': {'EUR': 1, None: 1}}

        self.assertEqual(r.count, expected_count)

        self.assertAlmostEqual(r.inbreeding[None].f_stat, -0.8518518518518517)
        self.assertEqual(r.inbreeding[None].n_called, 3)
        self.assertAlmostEqual(r.inbreeding[None].expected_homs, 2.46)
        self.assertEqual(r.inbreeding[None].observed_homs, 2)
        
        self.assertAlmostEqual(r.inbreeding['SIGMA'].f_stat, -1.777777777777777)
        self.assertEqual(r.inbreeding['SIGMA'].n_called, 2)
        self.assertAlmostEqual(r.inbreeding['SIGMA'].expected_homs, 1.64)
        self.assertEqual(r.inbreeding['SIGMA'].observed_homs, 1)

        self.assertAlmostEqual(r.inbreeding['IBD'].f_stat, -1.777777777777777)
        self.assertEqual(r.inbreeding['IBD'].n_called, 2)
        self.assertAlmostEqual(r.inbreeding['IBD'].expected_homs, 1.64)
        self.assertEqual(r.inbreeding['IBD'].observed_homs, 1)

    def test_aggregator_group_by_sorts_result(self):
        t = hl.Table.parallelize([ # the `s` key is stored before the `m` in java.util.HashMap
            {"group": "m", "x": 1},
            {"group": "s", "x": 2},
            {"group": "s", "x": 3},
            {"group": "m", "x": 4},
            {"group": "m", "x": 5}
        ], hl.tstruct(group=hl.tstr, x=hl.tint32), n_partitions=1)

        grouped_expr = t.aggregate(hl.array(hl.agg.group_by(t.group, hl.agg.sum(t.x))))
        self.assertEqual(grouped_expr, hl.eval(hl.sorted(grouped_expr)))

    def test_agg_corr(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(x=hl.rand_unif(-10, 10),
                         y=hl.rand_unif(-10, 10))
        c, xs, ys = ht.aggregate((hl.agg.corr(ht.x, ht.y), hl.agg.collect(ht.x), hl.agg.collect(ht.y)))

        scipy_corr, _ = pearsonr(xs, ys)
        self.assertAlmostEqual(c, scipy_corr)

    def test_joins_inside_aggregators(self):
        table = hl.utils.range_table(10)
        table2 = hl.utils.range_table(10)
        self.assertEqual(table.aggregate(agg.count_where(hl.is_defined(table2[table.idx]))), 10)

    def test_switch(self):
        x = hl.literal('1')
        na = hl.null(tint32)

        expr1 = (hl.switch(x)
            .when('123', 5)
            .when('1', 6)
            .when('0', 2)
            .or_missing())
        self.assertEqual(hl.eval(expr1), 6)

        expr2 = (hl.switch(x)
            .when('123', 5)
            .when('0', 2)
            .or_missing())
        self.assertEqual(hl.eval(expr2), None)

        expr3 = (hl.switch(x)
            .when('123', 5)
            .when('0', 2)
            .default(100))
        self.assertEqual(hl.eval(expr3), 100)

        expr4 = (hl.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(hl.null(tint32), 3)  # NA != NA
            .default(4))
        self.assertEqual(hl.eval(expr4), None)

        expr5 = (hl.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(hl.null(tint32), 3)  # NA != NA
            .when_missing(-1)
            .default(4))
        self.assertEqual(hl.eval(expr5), -1)

    def test_case(self):
        def make_case(x):
            x = hl.literal(x)
            return (hl.case()
                .when(x == 6, 'A')
                .when(x % 3 == 0, 'B')
                .when(x == 5, 'C')
                .when(x < 2, 'D')
                .or_missing())

        self.assertEqual(hl.eval(make_case(6)), 'A')
        self.assertEqual(hl.eval(make_case(12)), 'B')
        self.assertEqual(hl.eval(make_case(5)), 'C')
        self.assertEqual(hl.eval(make_case(-1)), 'D')
        self.assertEqual(hl.eval(make_case(2)), None)

        self.assertEqual(hl.eval(hl.case().when(hl.null(hl.tbool), 1).default(2)), None)
        self.assertEqual(hl.eval(hl.case(missing_false=True).when(hl.null(hl.tbool), 1).default(2)), 2)

        error_case = hl.case().when(False, 1).or_error("foo")
        self.assertRaises(hl.utils.java.FatalError, lambda: hl.eval(error_case))

    def test_struct_ops(self):
        s = hl.struct(f1=1, f2=2, f3=3)

        def assert_typed(expr, result, dtype):
            self.assertEqual(expr.dtype, dtype)
            r, t = hl.eval_typed(expr)
            self.assertEqual(t, dtype)
            self.assertEqual(result, r)

        assert_typed(s.drop('f3'),
                     hl.Struct(f1=1, f2=2),
                     tstruct(f1=tint32, f2=tint32))

        assert_typed(s.drop('f1'),
                     hl.Struct(f2=2, f3=3),
                     tstruct(f2=tint32, f3=tint32))

        assert_typed(s.drop(),
                     hl.Struct(f1=1, f2=2, f3=3),
                     tstruct(f1=tint32, f2=tint32, f3=tint32))

        assert_typed(s.select('f1', 'f2'),
                     hl.Struct(f1=1, f2=2),
                     tstruct(f1=tint32, f2=tint32))

        assert_typed(s.select('f2', 'f1', f4=5, f5=6),
                     hl.Struct(f2=2, f1=1, f4=5, f5=6),
                     tstruct(f2=tint32, f1=tint32, f4=tint32, f5=tint32))

        assert_typed(s.select(),
                     hl.Struct(),
                     tstruct())

        assert_typed(s.annotate(f1=5, f2=10, f4=15),
                     hl.Struct(f1=5, f2=10, f3=3, f4=15),
                     tstruct(f1=tint32, f2=tint32, f3=tint32, f4=tint32))

        assert_typed(s.annotate(f1=5),
                     hl.Struct(f1=5, f2=2, f3=3),
                     tstruct(f1=tint32, f2=tint32, f3=tint32))

        assert_typed(s.annotate(),
                     hl.Struct(f1=1, f2=2, f3=3),
                     tstruct(f1=tint32, f2=tint32, f3=tint32))

    def test_iter(self):
        a = hl.literal([1, 2, 3])
        self.assertRaises(hl.expr.ExpressionException, lambda: hl.eval(list(a)))

    def test_dict_get(self):
        d = hl.dict({'a': 1, 'b': 2, 'missing_value': hl.null(hl.tint32), hl.null(hl.tstr): 5})
        self.assertEqual(hl.eval(d.get('a')), 1)
        self.assertEqual(hl.eval(d['a']), 1)
        self.assertEqual(hl.eval(d.get('b')), 2)
        self.assertEqual(hl.eval(d['b']), 2)
        self.assertEqual(hl.eval(d.get('c')), None)
        self.assertEqual(hl.eval(d.get(hl.null(hl.tstr))), 5)
        self.assertEqual(hl.eval(d[hl.null(hl.tstr)]), 5)

        self.assertEqual(hl.eval(d.get('c', 5)), 5)
        self.assertEqual(hl.eval(d.get('a', 5)), 1)

        self.assertEqual(hl.eval(d.get('missing_values')), None)
        self.assertEqual(hl.eval(d.get('missing_values', hl.null(hl.tint32))), None)
        self.assertEqual(hl.eval(d.get('missing_values', 5)), 5)

    def test_aggregator_any_and_all(self):
        df = hl.utils.range_table(10)
        df = df.annotate(all_true=True,
                         all_false=False,
                         true_or_missing=hl.cond(df.idx % 2 == 0, True, hl.null(tbool)),
                         false_or_missing=hl.cond(df.idx % 2 == 0, False, hl.null(tbool)),
                         all_missing=hl.null(tbool),
                         mixed_true_false=hl.cond(df.idx % 2 == 0, True, False),
                         mixed_all=hl.switch(df.idx % 3)
                         .when(0, True)
                         .when(1, False)
                         .or_missing()).cache()

        self.assertEqual(df.aggregate(agg.any(df.all_true)), True)
        self.assertEqual(df.aggregate(agg.all(df.all_true)), True)
        self.assertEqual(df.aggregate(agg.any(df.all_false)), False)
        self.assertEqual(df.aggregate(agg.any(df.all_false)), False)
        self.assertEqual(df.aggregate(agg.any(df.true_or_missing)), True)
        self.assertEqual(df.aggregate(agg.all(df.true_or_missing)), True)
        self.assertEqual(df.aggregate(agg.any(df.false_or_missing)), False)
        self.assertEqual(df.aggregate(agg.all(df.false_or_missing)), False)
        self.assertEqual(df.aggregate(agg.any(df.all_missing)), False)
        self.assertEqual(df.aggregate(agg.all(df.all_missing)), True)
        self.assertEqual(df.aggregate(agg.any(df.mixed_true_false)), True)
        self.assertEqual(df.aggregate(agg.all(df.mixed_true_false)), False)
        self.assertEqual(df.aggregate(agg.any(df.mixed_all)), True)
        self.assertEqual(df.aggregate(agg.all(df.mixed_all)), False)

        self.assertEqual(df.aggregate(agg.filter(False, agg.any(df.all_true))), False)
        self.assertEqual(df.aggregate(agg.filter(False, agg.all(df.all_true))), True)

    def test_str_ops(self):
        s = hl.literal("123")
        self.assertEqual(hl.eval(hl.int32(s)), 123)

        s = hl.literal("123123123123")
        self.assertEqual(hl.eval(hl.int64(s)), 123123123123)

        s = hl.literal("1.5")
        self.assertEqual(hl.eval(hl.float32(s)), 1.5)
        self.assertEqual(hl.eval(hl.float64(s)), 1.5)

        s1 = hl.literal('true')
        s2 = hl.literal('True')
        s3 = hl.literal('TRUE')

        s4 = hl.literal('false')
        s5 = hl.literal('False')
        s6 = hl.literal('FALSE')

        self.assertTrue(hl.eval(hl.bool(s1)))
        self.assertTrue(hl.eval(hl.bool(s2)))
        self.assertTrue(hl.eval(hl.bool(s3)))

        self.assertFalse(hl.eval(hl.bool(s4)))
        self.assertFalse(hl.eval(hl.bool(s5)))
        self.assertFalse(hl.eval(hl.bool(s6)))

        s = hl.literal('abcABC123')
        self.assertEqual(hl.eval(s.lower()), 'abcabc123')
        self.assertEqual(hl.eval(s.upper()), 'ABCABC123')

        s_whitespace = hl.literal(' \t 1 2 3 \t\n')
        self.assertEqual(hl.eval(s_whitespace.strip()), '1 2 3')

        self.assertEqual(hl.eval(s.contains('ABC')), True)
        self.assertEqual(hl.eval(~s.contains('ABC')), False)
        self.assertEqual(hl.eval(s.contains('a')), True)
        self.assertEqual(hl.eval(s.contains('C123')), True)
        self.assertEqual(hl.eval(s.contains('')), True)
        self.assertEqual(hl.eval(s.contains('C1234')), False)
        self.assertEqual(hl.eval(s.contains(' ')), False)

        self.assertTrue(hl.eval(s_whitespace.startswith(' \t')))
        self.assertTrue(hl.eval(s_whitespace.endswith('\t\n')))
        self.assertFalse(hl.eval(s_whitespace.startswith('a')))
        self.assertFalse(hl.eval(s_whitespace.endswith('a')))

    def test_str_missingness(self):
        self.assertEqual(hl.eval(hl.str(1)), '1')
        self.assertEqual(hl.eval(hl.str(hl.null('int32'))), None)


    def check_expr(self, expr, expected, expected_type):
        self.assertEqual(expected_type, expr.dtype)
        self.assertEqual((expected, expected_type), hl.eval_typed(expr))

    def test_division(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int64_4 = hl.int64(4)
        int64_4s = int32_4s.map(lambda x: hl.int64(x))
        float32_4 = hl.float32(4)
        float32_4s = int32_4s.map(lambda x: hl.float32(x))
        float64_4 = hl.float64(4)
        float64_4s = int32_4s.map(lambda x: hl.float64(x))

        expected = [0.5, 1.0, 2.0, 4.0, None]
        expected_inv = [2.0, 1.0, 0.5, 0.25, None]

        self.check_expr(a_int32 / 4, expected, tarray(tfloat32))
        self.check_expr(a_int64 / 4, expected, tarray(tfloat32))
        self.check_expr(a_float32 / 4, expected, tarray(tfloat32))
        self.check_expr(a_float64 / 4, expected, tarray(tfloat64))

        self.check_expr(int32_4s / a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(int32_4s / a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(int32_4s / a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int32_4s / a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 / int32_4s, expected, tarray(tfloat32))
        self.check_expr(a_int64 / int32_4s, expected, tarray(tfloat32))
        self.check_expr(a_float32 / int32_4s, expected, tarray(tfloat32))
        self.check_expr(a_float64 / int32_4s, expected, tarray(tfloat64))

        self.check_expr(a_int32 / int64_4, expected, tarray(tfloat32))
        self.check_expr(a_int64 / int64_4, expected, tarray(tfloat32))
        self.check_expr(a_float32 / int64_4, expected, tarray(tfloat32))
        self.check_expr(a_float64 / int64_4, expected, tarray(tfloat64))

        self.check_expr(int64_4 / a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_4 / a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(int64_4 / a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_4 / a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 / int64_4s, expected, tarray(tfloat32))
        self.check_expr(a_int64 / int64_4s, expected, tarray(tfloat32))
        self.check_expr(a_float32 / int64_4s, expected, tarray(tfloat32))
        self.check_expr(a_float64 / int64_4s, expected, tarray(tfloat64))

        self.check_expr(a_int32 / float32_4, expected, tarray(tfloat32))
        self.check_expr(a_int64 / float32_4, expected, tarray(tfloat32))
        self.check_expr(a_float32 / float32_4, expected, tarray(tfloat32))
        self.check_expr(a_float64 / float32_4, expected, tarray(tfloat64))

        self.check_expr(float32_4 / a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_4 / a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(float32_4 / a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_4 / a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 / float32_4s, expected, tarray(tfloat32))
        self.check_expr(a_int64 / float32_4s, expected, tarray(tfloat32))
        self.check_expr(a_float32 / float32_4s, expected, tarray(tfloat32))
        self.check_expr(a_float64 / float32_4s, expected, tarray(tfloat64))

        self.check_expr(a_int32 / float64_4, expected, tarray(tfloat64))
        self.check_expr(a_int64 / float64_4, expected, tarray(tfloat64))
        self.check_expr(a_float32 / float64_4, expected, tarray(tfloat64))
        self.check_expr(a_float64 / float64_4, expected, tarray(tfloat64))

        self.check_expr(float64_4 / a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_4 / a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_4 / a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_4 / a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 / float64_4s, expected, tarray(tfloat64))
        self.check_expr(a_int64 / float64_4s, expected, tarray(tfloat64))
        self.check_expr(a_float32 / float64_4s, expected, tarray(tfloat64))
        self.check_expr(a_float64 / float64_4s, expected, tarray(tfloat64))

    def test_floor_division(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.null(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [0, 1, 2, 5, None]
        expected_inv = [1, 0, 0, 0, None]

        self.check_expr(a_int32 // 3, expected, tarray(tint32))
        self.check_expr(a_int64 // 3, expected, tarray(tint64))
        self.check_expr(a_float32 // 3, expected, tarray(tfloat32))
        self.check_expr(a_float64 // 3, expected, tarray(tfloat64))

        self.check_expr(3 // a_int32, expected_inv, tarray(tint32))
        self.check_expr(3 // a_int64, expected_inv, tarray(tint64))
        self.check_expr(3 // a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(3 // a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 // int32_3s, expected, tarray(tint32))
        self.check_expr(a_int64 // int32_3s, expected, tarray(tint64))
        self.check_expr(a_float32 // int32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 // int32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 // int64_3, expected, tarray(tint64))
        self.check_expr(a_int64 // int64_3, expected, tarray(tint64))
        self.check_expr(a_float32 // int64_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 // int64_3, expected, tarray(tfloat64))

        self.check_expr(int64_3 // a_int32, expected_inv, tarray(tint64))
        self.check_expr(int64_3 // a_int64, expected_inv, tarray(tint64))
        self.check_expr(int64_3 // a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_3 // a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 // int64_3s, expected, tarray(tint64))
        self.check_expr(a_int64 // int64_3s, expected, tarray(tint64))
        self.check_expr(a_float32 // int64_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 // int64_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 // float32_3, expected, tarray(tfloat32))
        self.check_expr(a_int64 // float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float32 // float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 // float32_3, expected, tarray(tfloat64))

        self.check_expr(float32_3 // a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 // a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 // a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 // a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 // float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_int64 // float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float32 // float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 // float32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 // float64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 // float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 // float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 // float64_3, expected, tarray(tfloat64))

        self.check_expr(float64_3 // a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 // a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 // a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 // a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 // float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 // float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 // float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 // float64_3s, expected, tarray(tfloat64))

    def test_addition(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.null(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [5, 7, 11, 19, None]
        expected_inv = expected

        self.check_expr(a_int32 + 3, expected, tarray(tint32))
        self.check_expr(a_int64 + 3, expected, tarray(tint64))
        self.check_expr(a_float32 + 3, expected, tarray(tfloat32))
        self.check_expr(a_float64 + 3, expected, tarray(tfloat64))

        self.check_expr(3 + a_int32, expected_inv, tarray(tint32))
        self.check_expr(3 + a_int64, expected_inv, tarray(tint64))
        self.check_expr(3 + a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(3 + a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 + int32_3s, expected, tarray(tint32))
        self.check_expr(a_int64 + int32_3s, expected, tarray(tint64))
        self.check_expr(a_float32 + int32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 + int32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 + int64_3, expected, tarray(tint64))
        self.check_expr(a_int64 + int64_3, expected, tarray(tint64))
        self.check_expr(a_float32 + int64_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 + int64_3, expected, tarray(tfloat64))

        self.check_expr(int64_3 + a_int32, expected_inv, tarray(tint64))
        self.check_expr(int64_3 + a_int64, expected_inv, tarray(tint64))
        self.check_expr(int64_3 + a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_3 + a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 + int64_3s, expected, tarray(tint64))
        self.check_expr(a_int64 + int64_3s, expected, tarray(tint64))
        self.check_expr(a_float32 + int64_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 + int64_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 + float32_3, expected, tarray(tfloat32))
        self.check_expr(a_int64 + float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float32 + float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 + float32_3, expected, tarray(tfloat64))

        self.check_expr(float32_3 + a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 + a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 + a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 + a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 + float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_int64 + float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float32 + float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 + float32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 + float64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 + float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 + float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 + float64_3, expected, tarray(tfloat64))

        self.check_expr(float64_3 + a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 + a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 + a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 + a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 + float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 + float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 + float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 + float64_3s, expected, tarray(tfloat64))

    def test_subtraction(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.null(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [-1, 1, 5, 13, None]
        expected_inv = [1, -1, -5, -13, None]

        self.check_expr(a_int32 - 3, expected, tarray(tint32))
        self.check_expr(a_int64 - 3, expected, tarray(tint64))
        self.check_expr(a_float32 - 3, expected, tarray(tfloat32))
        self.check_expr(a_float64 - 3, expected, tarray(tfloat64))

        self.check_expr(3 - a_int32, expected_inv, tarray(tint32))
        self.check_expr(3 - a_int64, expected_inv, tarray(tint64))
        self.check_expr(3 - a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(3 - a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 - int32_3s, expected, tarray(tint32))
        self.check_expr(a_int64 - int32_3s, expected, tarray(tint64))
        self.check_expr(a_float32 - int32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 - int32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 - int64_3, expected, tarray(tint64))
        self.check_expr(a_int64 - int64_3, expected, tarray(tint64))
        self.check_expr(a_float32 - int64_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 - int64_3, expected, tarray(tfloat64))

        self.check_expr(int64_3 - a_int32, expected_inv, tarray(tint64))
        self.check_expr(int64_3 - a_int64, expected_inv, tarray(tint64))
        self.check_expr(int64_3 - a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_3 - a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 - int64_3s, expected, tarray(tint64))
        self.check_expr(a_int64 - int64_3s, expected, tarray(tint64))
        self.check_expr(a_float32 - int64_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 - int64_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 - float32_3, expected, tarray(tfloat32))
        self.check_expr(a_int64 - float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float32 - float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 - float32_3, expected, tarray(tfloat64))

        self.check_expr(float32_3 - a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 - a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 - a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 - a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 - float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_int64 - float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float32 - float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 - float32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 - float64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 - float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 - float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 - float64_3, expected, tarray(tfloat64))

        self.check_expr(float64_3 - a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 - a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 - a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 - a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 - float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 - float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 - float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 - float64_3s, expected, tarray(tfloat64))

    def test_multiplication(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.null(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [6, 12, 24, 48, None]
        expected_inv = expected

        self.check_expr(a_int32 * 3, expected, tarray(tint32))
        self.check_expr(a_int64 * 3, expected, tarray(tint64))
        self.check_expr(a_float32 * 3, expected, tarray(tfloat32))
        self.check_expr(a_float64 * 3, expected, tarray(tfloat64))

        self.check_expr(3 * a_int32, expected_inv, tarray(tint32))
        self.check_expr(3 * a_int64, expected_inv, tarray(tint64))
        self.check_expr(3 * a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(3 * a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 * int32_3s, expected, tarray(tint32))
        self.check_expr(a_int64 * int32_3s, expected, tarray(tint64))
        self.check_expr(a_float32 * int32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 * int32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 * int64_3, expected, tarray(tint64))
        self.check_expr(a_int64 * int64_3, expected, tarray(tint64))
        self.check_expr(a_float32 * int64_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 * int64_3, expected, tarray(tfloat64))

        self.check_expr(int64_3 * a_int32, expected_inv, tarray(tint64))
        self.check_expr(int64_3 * a_int64, expected_inv, tarray(tint64))
        self.check_expr(int64_3 * a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_3 * a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 * int64_3s, expected, tarray(tint64))
        self.check_expr(a_int64 * int64_3s, expected, tarray(tint64))
        self.check_expr(a_float32 * int64_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 * int64_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 * float32_3, expected, tarray(tfloat32))
        self.check_expr(a_int64 * float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float32 * float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 * float32_3, expected, tarray(tfloat64))

        self.check_expr(float32_3 * a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 * a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 * a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 * a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 * float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_int64 * float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float32 * float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 * float32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 * float64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 * float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 * float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 * float64_3, expected, tarray(tfloat64))

        self.check_expr(float64_3 * a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 * a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 * a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 * a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 * float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 * float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 * float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 * float64_3s, expected, tarray(tfloat64))

    def test_exponentiation(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.null(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [8, 64, 512, 4096, None]
        expected_inv = [9.0, 81.0, 6561.0, 43046721.0, None]

        self.check_expr(a_int32 ** 3, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** 3, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** 3, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** 3, expected, tarray(tfloat64))

        self.check_expr(3 ** a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(3 ** a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(3 ** a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(3 ** a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 ** int32_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** int32_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** int32_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** int32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 ** int64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** int64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** int64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** int64_3, expected, tarray(tfloat64))

        self.check_expr(int64_3 ** a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(int64_3 ** a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(int64_3 ** a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(int64_3 ** a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 ** int64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** int64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** int64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** int64_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 ** float32_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** float32_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** float32_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** float32_3, expected, tarray(tfloat64))

        self.check_expr(float32_3 ** a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float32_3 ** a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float32_3 ** a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float32_3 ** a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 ** float32_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** float32_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** float32_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** float32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 ** float64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** float64_3, expected, tarray(tfloat64))

        self.check_expr(float64_3 ** a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 ** a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 ** a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 ** a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 ** float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 ** float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 ** float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 ** float64_3s, expected, tarray(tfloat64))

    def test_modulus(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.null(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [2, 1, 2, 1, None]
        expected_inv = [1, 3, 3, 3, None]

        self.check_expr(a_int32 % 3, expected, tarray(tint32))
        self.check_expr(a_int64 % 3, expected, tarray(tint64))
        self.check_expr(a_float32 % 3, expected, tarray(tfloat32))
        self.check_expr(a_float64 % 3, expected, tarray(tfloat64))

        self.check_expr(3 % a_int32, expected_inv, tarray(tint32))
        self.check_expr(3 % a_int64, expected_inv, tarray(tint64))
        self.check_expr(3 % a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(3 % a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 % int32_3s, expected, tarray(tint32))
        self.check_expr(a_int64 % int32_3s, expected, tarray(tint64))
        self.check_expr(a_float32 % int32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 % int32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 % int64_3, expected, tarray(tint64))
        self.check_expr(a_int64 % int64_3, expected, tarray(tint64))
        self.check_expr(a_float32 % int64_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 % int64_3, expected, tarray(tfloat64))

        self.check_expr(int64_3 % a_int32, expected_inv, tarray(tint64))
        self.check_expr(int64_3 % a_int64, expected_inv, tarray(tint64))
        self.check_expr(int64_3 % a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(int64_3 % a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 % int64_3s, expected, tarray(tint64))
        self.check_expr(a_int64 % int64_3s, expected, tarray(tint64))
        self.check_expr(a_float32 % int64_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 % int64_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 % float32_3, expected, tarray(tfloat32))
        self.check_expr(a_int64 % float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float32 % float32_3, expected, tarray(tfloat32))
        self.check_expr(a_float64 % float32_3, expected, tarray(tfloat64))

        self.check_expr(float32_3 % a_int32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 % a_int64, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 % a_float32, expected_inv, tarray(tfloat32))
        self.check_expr(float32_3 % a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 % float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_int64 % float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float32 % float32_3s, expected, tarray(tfloat32))
        self.check_expr(a_float64 % float32_3s, expected, tarray(tfloat64))

        self.check_expr(a_int32 % float64_3, expected, tarray(tfloat64))
        self.check_expr(a_int64 % float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float32 % float64_3, expected, tarray(tfloat64))
        self.check_expr(a_float64 % float64_3, expected, tarray(tfloat64))

        self.check_expr(float64_3 % a_int32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 % a_int64, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 % a_float32, expected_inv, tarray(tfloat64))
        self.check_expr(float64_3 % a_float64, expected_inv, tarray(tfloat64))

        self.check_expr(a_int32 % float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_int64 % float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float32 % float64_3s, expected, tarray(tfloat64))
        self.check_expr(a_float64 % float64_3s, expected, tarray(tfloat64))

    def test_comparisons(self):
        f0 = hl.float(0.0)
        fnull = hl.null(tfloat)
        finf = hl.float(float('inf'))
        fnan = hl.float(float('nan'))

        self.check_expr(f0 == fnull, None, tbool)
        self.check_expr(f0 < fnull, None, tbool)
        self.check_expr(f0 != fnull, None, tbool)

        self.check_expr(fnan == fnan, False, tbool)
        self.check_expr(f0 == f0, True, tbool)
        self.check_expr(finf == finf, True, tbool)

        self.check_expr(f0 < finf, True, tbool)
        self.check_expr(f0 > finf, False, tbool)

        self.check_expr(fnan <= finf, False, tbool)
        self.check_expr(fnan >= finf, False, tbool)

    def test_bools_can_math(self):
        b1 = hl.literal(True)
        b2 = hl.literal(False)

        b_array = hl.literal([True, False])
        f1 = hl.float64(5.5)
        f_array = hl.array([1.5, 2.5])

        self.assertEqual(hl.eval(hl.int32(b1)), 1)
        self.assertEqual(hl.eval(hl.int64(b1)), 1)
        self.assertEqual(hl.eval(hl.float32(b1)), 1.0)
        self.assertEqual(hl.eval(hl.float64(b1)), 1.0)
        self.assertEqual(hl.eval(b1 * b2), 0)
        self.assertEqual(hl.eval(b1 + b2), 1)
        self.assertEqual(hl.eval(b1 - b2), 1)
        self.assertEqual(hl.eval(b1 / b1), 1.0)
        self.assertEqual(hl.eval(f1 * b2), 0.0)
        self.assertEqual(hl.eval(b_array + f1), [6.5, 5.5])
        self.assertEqual(hl.eval(b_array + f_array), [2.5, 2.5])

    def test_int_typecheck(self):
        self.assertIsNone(hl.eval(hl.literal(None, dtype='int32')))
        self.assertIsNone(hl.eval(hl.literal(None, dtype='int64')))

    def test_is_transition(self):
        self.assertTrue(hl.eval(hl.is_transition("A", "G")))
        self.assertTrue(hl.eval(hl.is_transition("C", "T")))
        self.assertTrue(hl.eval(hl.is_transition("AA", "AG")))
        self.assertFalse(hl.eval(hl.is_transition("AA", "G")))
        self.assertFalse(hl.eval(hl.is_transition("ACA", "AGA")))
        self.assertFalse(hl.eval(hl.is_transition("A", "T")))

    def test_is_transversion(self):
        self.assertTrue(hl.eval(hl.is_transversion("A", "T")))
        self.assertFalse(hl.eval(hl.is_transversion("A", "G")))
        self.assertTrue(hl.eval(hl.is_transversion("AA", "AT")))
        self.assertFalse(hl.eval(hl.is_transversion("AA", "T")))
        self.assertFalse(hl.eval(hl.is_transversion("ACCC", "ACCT")))

    def test_is_snp(self):
        self.assertTrue(hl.eval(hl.is_snp("A", "T")))
        self.assertTrue(hl.eval(hl.is_snp("A", "G")))
        self.assertTrue(hl.eval(hl.is_snp("C", "G")))
        self.assertTrue(hl.eval(hl.is_snp("CC", "CG")))
        self.assertTrue(hl.eval(hl.is_snp("AT", "AG")))
        self.assertTrue(hl.eval(hl.is_snp("ATCCC", "AGCCC")))

    def test_is_mnp(self):
        self.assertTrue(hl.eval(hl.is_mnp("ACTGAC", "ATTGTT")))
        self.assertTrue(hl.eval(hl.is_mnp("CA", "TT")))

    def test_is_insertion(self):
        self.assertTrue(hl.eval(hl.is_insertion("A", "ATGC")))
        self.assertTrue(hl.eval(hl.is_insertion("ATT", "ATGCTT")))

    def test_is_deletion(self):
        self.assertTrue(hl.eval(hl.is_deletion("ATGC", "A")))
        self.assertTrue(hl.eval(hl.is_deletion("GTGTA", "GTA")))

    def test_is_indel(self):
        self.assertTrue(hl.eval(hl.is_indel("A", "ATGC")))
        self.assertTrue(hl.eval(hl.is_indel("ATT", "ATGCTT")))
        self.assertTrue(hl.eval(hl.is_indel("ATGC", "A")))
        self.assertTrue(hl.eval(hl.is_indel("GTGTA", "GTA")))

    def test_is_complex(self):
        self.assertTrue(hl.eval(hl.is_complex("CTA", "ATTT")))
        self.assertTrue(hl.eval(hl.is_complex("A", "TATGC")))

    def test_is_star(self):
        self.assertTrue(hl.eval(hl.is_star("ATC", "*")))
        self.assertTrue(hl.eval(hl.is_star("A", "*")))

    def test_is_strand_ambiguous(self):
        self.assertTrue(hl.eval(hl.is_strand_ambiguous("A", "T")))
        self.assertFalse(hl.eval(hl.is_strand_ambiguous("G", "T")))

    def test_allele_type(self):
        self.assertEqual(
            hl.eval(hl.tuple((
                hl.allele_type('A', 'C'),
                hl.allele_type('AC', 'CT'),
                hl.allele_type('C', 'CT'),
                hl.allele_type('CT', 'C'),
                hl.allele_type('CTCA', 'AAC'),
                hl.allele_type('CTCA', '*'),
                hl.allele_type('C', '<DEL>'),
                hl.allele_type('C', '<SYMBOLIC>'),
                hl.allele_type('C', 'H'),
                hl.allele_type('C', ''),
                hl.allele_type('A', 'A'),
                hl.allele_type('', 'CCT'),
                hl.allele_type('F', 'CCT'),
                hl.allele_type('A', '[ASDASD[A'),
                hl.allele_type('A', ']ASDASD]A'),
                hl.allele_type('A', 'T<ASDASD>]ASDASD]'),
                hl.allele_type('A', 'T<ASDASD>[ASDASD['),
                hl.allele_type('A', '.T'),
                hl.allele_type('A', 'T.'),
            ))),
            (
                'SNP',
                'MNP',
                'Insertion',
                'Deletion',
                'Complex',
                'Star',
                'Symbolic',
                'Symbolic',
                'Unknown',
                'Unknown',
                'Unknown',
                'Unknown',
                'Unknown',
                'Symbolic',
                'Symbolic',
                'Symbolic',
                'Symbolic',
                'Symbolic',
                'Symbolic',
            )
        )

    def test_hamming(self):
        self.assertEqual(hl.eval(hl.hamming('A', 'T')), 1)
        self.assertEqual(hl.eval(hl.hamming('AAAAA', 'AAAAT')), 1)
        self.assertEqual(hl.eval(hl.hamming('abcde', 'edcba')), 4)

    def test_gp_dosage(self):
        self.assertAlmostEqual(hl.eval(hl.gp_dosage([1.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(hl.eval(hl.gp_dosage([0.0, 1.0, 0.0])), 1.0)
        self.assertAlmostEqual(hl.eval(hl.gp_dosage([0.0, 0.0, 1.0])), 2.0)
        self.assertAlmostEqual(hl.eval(hl.gp_dosage([0.5, 0.5, 0.0])), 0.5)
        self.assertAlmostEqual(hl.eval(hl.gp_dosage([0.0, 0.5, 0.5])), 1.5)

    def test_call(self):
        c2_homref = hl.call(0, 0)
        c2_het = hl.call(1, 0, phased=True)
        c2_homvar = hl.call(1, 1)
        c2_hetvar = hl.call(2, 1, phased=True)
        c1 = hl.call(1)
        c0 = hl.call()
        cNull = hl.null(tcall)

        self.check_expr(c2_homref.ploidy, 2, tint32)
        self.check_expr(c2_homref[0], 0, tint32)
        self.check_expr(c2_homref[1], 0, tint32)
        self.check_expr(c2_homref.phased, False, tbool)
        self.check_expr(c2_homref.is_hom_ref(), True, tbool)

        self.check_expr(c2_het.ploidy, 2, tint32)
        self.check_expr(c2_het[0], 1, tint32)
        self.check_expr(c2_het[1], 0, tint32)
        self.check_expr(c2_het.phased, True, tbool)
        self.check_expr(c2_het.is_het(), True, tbool)

        self.check_expr(c2_homvar.ploidy, 2, tint32)
        self.check_expr(c2_homvar[0], 1, tint32)
        self.check_expr(c2_homvar[1], 1, tint32)
        self.check_expr(c2_homvar.phased, False, tbool)
        self.check_expr(c2_homvar.is_hom_var(), True, tbool)
        self.check_expr(c2_homvar.unphased_diploid_gt_index(), 2, tint32)

        self.check_expr(c2_hetvar.ploidy, 2, tint32)
        self.check_expr(c2_hetvar[0], 2, tint32)
        self.check_expr(c2_hetvar[1], 1, tint32)
        self.check_expr(c2_hetvar.phased, True, tbool)
        self.check_expr(c2_hetvar.is_hom_var(), False, tbool)
        self.check_expr(c2_hetvar.is_het_non_ref(), True, tbool)

        self.check_expr(c1.ploidy, 1, tint32)
        self.check_expr(c1[0], 1, tint32)
        self.check_expr(c1.phased, False, tbool)
        self.check_expr(c1.is_hom_var(), True, tbool)

        self.check_expr(c0.ploidy, 0, tint32)
        self.check_expr(c0.phased, False, tbool)
        self.check_expr(c0.is_hom_var(), False, tbool)

        self.check_expr(cNull.ploidy, None, tint32)
        self.check_expr(cNull[0], None, tint32)
        self.check_expr(cNull.phased, None, tbool)
        self.check_expr(cNull.is_hom_var(), None, tbool)

        call_expr = hl.call(1, 2, phased=True)
        self.check_expr(call_expr[0], 1, tint32)
        self.check_expr(call_expr[1], 2, tint32)
        self.check_expr(call_expr.ploidy, 2, tint32)

        a0 = hl.literal(1)
        a1 = 2
        phased = hl.literal(True)
        call_expr = hl.call(a0, a1, phased=phased)
        self.check_expr(call_expr[0], 1, tint32)
        self.check_expr(call_expr[1], 2, tint32)
        self.check_expr(call_expr.ploidy, 2, tint32)

        call_expr = hl.parse_call("1|2")
        self.check_expr(call_expr[0], 1, tint32)
        self.check_expr(call_expr[1], 2, tint32)
        self.check_expr(call_expr.ploidy, 2, tint32)

        call_expr = hl.unphased_diploid_gt_index_call(2)
        self.check_expr(call_expr[0], 1, tint32)
        self.check_expr(call_expr[1], 1, tint32)
        self.check_expr(call_expr.ploidy, 2, tint32)

    def test_parse_variant(self):
        self.assertEqual(hl.eval(hl.parse_variant('1:1:A:T')),
                         hl.Struct(locus=hl.Locus('1', 1), alleles=['A', 'T']))

    def test_locus_to_global_position(self):
        self.assertEqual(hl.eval(hl.locus('chr22', 1, 'GRCh38').global_position()), 2824183054)

    def test_locus_from_global_position(self):
        self.assertEqual(hl.eval(hl.locus_from_global_position(2824183054, 'GRCh38')),
                         hl.eval(hl.locus('chr22', 1, 'GRCh38')))

    def test_dict_conversions(self):
        self.assertEqual(sorted(hl.eval(hl.array({1: 1, 2: 2}))), [(1, 1), (2, 2)])
        self.assertEqual(hl.eval(hl.dict(hl.array({1: 1, 2: 2}))), {1: 1, 2: 2})

        self.assertEqual(hl.eval(hl.dict([('1', 2), ('2', 3)])), {'1': 2, '2': 3})
        self.assertEqual(hl.eval(hl.dict({('1', 2), ('2', 3)})), {'1': 2, '2': 3})
        self.assertEqual(hl.eval(hl.dict([('1', 2), (hl.null(tstr), 3)])), {'1': 2, None: 3})
        self.assertEqual(hl.eval(hl.dict({('1', 2), (hl.null(tstr), 3)})), {'1': 2, None: 3})

    def test_zip(self):
        a1 = [1,2,3]
        a2 = ['a', 'b']
        a3 = [[1]]

        self.assertEqual(hl.eval(hl.zip(a1, a2)), [(1, 'a'), (2, 'b')])
        self.assertEqual(hl.eval(hl.zip(a1, a2, fill_missing=True)), [(1, 'a'), (2, 'b'), (3, None)])

        self.assertEqual(hl.eval(hl.zip(a3, a2, a1, fill_missing=True)),
                         [([1], 'a', 1), (None, 'b', 2), (None, None, 3)])
        self.assertEqual(hl.eval(hl.zip(a3, a2, a1)),
                         [([1], 'a', 1)])

    def test_array_methods(self):
        self.assertEqual(hl.eval(hl.any(lambda x: x % 2 == 0, [1, 3, 5])), False)
        self.assertEqual(hl.eval(hl.any(lambda x: x % 2 == 0, [1, 3, 5, 6])), True)

        self.assertEqual(hl.eval(hl.all(lambda x: x % 2 == 0, [1, 3, 5, 6])), False)
        self.assertEqual(hl.eval(hl.all(lambda x: x % 2 == 0, [2, 6])), True)

        self.assertEqual(hl.eval(hl.map(lambda x: x % 2 == 0, [0, 1, 4, 6])), [True, False, True, True])

        self.assertEqual(hl.eval(hl.len([0, 1, 4, 6])), 4)

        self.assertTrue(math.isnan(hl.eval(hl.mean(hl.empty_array(hl.tint)))))
        self.assertEqual(hl.eval(hl.mean([0, 1, 4, 6, hl.null(tint32)])), 2.75)

        self.assertEqual(hl.eval(hl.median(hl.empty_array(hl.tint))), None)
        self.assertTrue(1 <= hl.eval(hl.median([0, 1, 4, 6])) <= 4)

        for f in [lambda x: hl.int32(x), lambda x: hl.int64(x), lambda x: hl.float32(x), lambda x: hl.float64(x)]:
            self.assertEqual(hl.eval(hl.product([f(x) for x in [1, 4, 6]])), 24)
            self.assertEqual(hl.eval(hl.sum([f(x) for x in [1, 4, 6]])), 11)

        self.assertEqual(hl.eval(hl.group_by(lambda x: x % 2 == 0, [0, 1, 4, 6])), {True: [0, 4, 6], False: [1]})

        self.assertEqual(hl.eval(hl.flatmap(lambda x: hl.range(0, x), [1, 2, 3])), [0, 0, 1, 0, 1, 2])
        fm = hl.flatmap(lambda x: hl.set(hl.range(0, x.length()).map(lambda i: x[i])), {"ABC", "AAa", "BD"})
        self.assertEqual(hl.eval(fm), {'A', 'a', 'B', 'C', 'D'})

    def test_array_corr(self):
        x1 = [random.uniform(-10, 10) for x in range(10)]
        x2 = [random.uniform(-10, 10) for x in range(10)]
        self.assertAlmostEqual(hl.eval(hl.corr(x1, x2)), pearsonr(x1, x2)[0])

    def test_array_corr_missingness(self):
        x1 = [None, None, 5.0] + [random.uniform(-10, 10) for x in range(15)]
        x2 = [None, 5.0, None] + [random.uniform(-10, 10) for x in range(15)]
        self.assertAlmostEqual(hl.eval(hl.corr(hl.literal(x1, 'array<float>'), hl.literal(x2, 'array<float>'))),
                               pearsonr(x1[3:], x2[3:])[0])

    def test_array_find(self):
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.null(hl.tarray(hl.tint32)))), None)
        self.assertEqual(hl.eval(hl.find(lambda x: hl.null(hl.tbool), [1, 0, -4, 6])), None)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, [1, 0, -4, 6])), -4)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, [1, 0, 4, 6])), None)

    def test_set_find(self):
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.null(hl.tset(hl.tint32)))), None)
        self.assertEqual(hl.eval(hl.find(lambda x: hl.null(hl.tbool), hl.set([1, 0, -4, 6]))), None)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.set([1, 0, -4, 6]))), -4)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.set([1, 0, 4, 6]))), None)

    def test_sorted(self):
        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, 3, 2], lambda x: x % 2)), [0, 4, 2, 1, 3])
        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, 3, 2], lambda x: x % 2, reverse=True)), [1, 3, 0, 4, 2])

        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, hl.null(tint), 3, 2], lambda x: x)), [0, 1, 2, 3, 4, None])
        # FIXME: this next line triggers a bug: None should be sorted last!
        # self.assertEqual(hl.sorted([0, 1, 4, hl.null(tint), 3, 2], lambda x: x, reverse=True).collect()[0], [4, 3, 2, 1, 0, None])
        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, hl.null(tint), 3, 2], lambda x: x, reverse=True)), [4, 3, 2, 1, 0, None])

    def test_bool_r_ops(self):
        self.assertTrue(hl.eval(hl.literal(True) & True))
        self.assertTrue(hl.eval(True & hl.literal(True)))
        self.assertTrue(hl.eval(hl.literal(False) | True))
        self.assertTrue(hl.eval(True | hl.literal(False)))

    def test_array_neg(self):
        self.assertEqual(hl.eval(-(hl.literal([1, 2, 3]))), [-1, -2, -3])

    def test_max(self):
        self.assertEqual(hl.eval(hl.max(1, 2)), 2)
        self.assertEqual(hl.eval(hl.max(1.0, 2)), 2.0)
        self.assertEqual(hl.eval(hl.max([1, 2])), 2)
        self.assertEqual(hl.eval(hl.max([1.0, 2])), 2.0)
        self.assertEqual(hl.eval(hl.max(0, 1.0, 2)), 2.0)
        self.assertEqual(hl.eval(hl.max(0, 1, 2)), 2)
        self.assertEqual(hl.eval(hl.max([0, 10, 2, 3, 4, 5, 6, ])), 10)
        self.assertEqual(hl.eval(hl.max(0, 10, 2, 3, 4, 5, 6)), 10)
        self.assert_evals_to(hl.max([-5, -4, hl.null(tint32), -3, -2, hl.null(tint32)]), -2)

    def test_min(self):
        self.assertEqual(hl.eval(hl.min(1, 2)), 1)
        self.assertEqual(hl.eval(hl.min(1.0, 2)), 1.0)
        self.assertEqual(hl.eval(hl.min([1, 2])), 1)
        self.assertEqual(hl.eval(hl.min([1.0, 2])), 1.0)
        self.assertEqual(hl.eval(hl.min(0, 1.0, 2)), 0.0)
        self.assertEqual(hl.eval(hl.min(0, 1, 2)), 0)
        self.assertEqual(hl.eval(hl.min([0, 10, 2, 3, 4, 5, 6, ])), 0)
        self.assertEqual(hl.eval(hl.min(4, 10, 2, 3, 4, 5, 6)), 2)
        self.assert_evals_to(hl.min([-5, -4, hl.null(tint32), -3, -2, hl.null(tint32)]), -5)

    def test_abs(self):
        self.assertEqual(hl.eval(hl.abs(-5)), 5)
        self.assertEqual(hl.eval(hl.abs(-5.5)), 5.5)
        self.assertEqual(hl.eval(hl.abs(5.5)), 5.5)
        self.assertEqual(hl.eval(hl.abs([5.5, -5.5])), [5.5, 5.5])

    def test_sign(self):
        self.assertEqual(hl.eval(hl.sign(-5)), -1)
        self.assertEqual(hl.eval(hl.sign(0.0)), 0.0)
        self.assertEqual(hl.eval(hl.sign(10.0)), 1.0)
        self.assertTrue(hl.eval(hl.is_nan(hl.sign(float('nan')))))
        self.assertEqual(hl.eval(hl.sign(float('inf'))), 1.0)
        self.assertEqual(hl.eval(hl.sign(float('-inf'))), -1.0)
        self.assertEqual(hl.eval(hl.sign([-2, 0, 2])), [-1, 0, 1])
        self.assertEqual(hl.eval(hl.sign([-2.0, 0.0, 2.0])), [-1.0, 0.0, 1.0])

    def test_argmin_and_argmax(self):
        a = hl.array([2, 1, 1, 4, 4, 3])
        self.assertEqual(hl.eval(hl.argmax(a)), 3)
        self.assertEqual(hl.eval(hl.argmax(a, unique=True)), None)
        self.assertEqual(hl.eval(hl.argmax(hl.empty_array(tint32))), None)
        self.assertEqual(hl.eval(hl.argmin(a)), 1)
        self.assertEqual(hl.eval(hl.argmin(a, unique=True)), None)
        self.assertEqual(hl.eval(hl.argmin(hl.empty_array(tint32))), None)

    def test_show_row_key_regression(self):
        ds = hl.utils.range_matrix_table(3, 3)
        ds.col_idx.show(3)

    def test_or_else_type_conversion(self):
        self.assertEqual(hl.eval(hl.or_else(0.5, 2)), 0.5)

    def test_coalesce(self):
        self.assertEqual(hl.eval(hl.coalesce(hl.null('int'), hl.null('int'), hl.null('int'))), None)
        self.assertEqual(hl.eval(hl.coalesce(hl.null('int'), hl.null('int'), 2)), 2)
        self.assertEqual(hl.eval(hl.coalesce(hl.null('int'), hl.null('int'), 2.5)), 2.5)
        self.assertEqual(hl.eval(hl.coalesce(2.5)), 2.5)
        with self.assertRaises(TypeError):
            hl.coalesce(2.5, 'hello')

    def test_tuple_ops(self):
        t0 = hl.literal(())
        t1 = hl.literal((1,))
        t2 = hl.literal((1, "hello"))
        tn1 = hl.literal((1, (2, (3, 4))))

        t = hl.tuple([1, t1, hl.dict(hl.zip(["a", "b"], [t2, t2])), [1, 5], tn1])

        self.assertTrue(hl.eval(t[0]) == 1)
        self.assertTrue(hl.eval(t[1][0]) == 1)
        self.assertTrue(hl.eval(t[2]["a"]) == (1, "hello"))
        self.assertTrue(hl.eval(t[2]["b"][1]) == "hello")
        self.assertTrue(hl.eval(t[3][1]) == 5)
        self.assertTrue(hl.eval(t[4][1][1][1]) == 4)

        self.assertTrue(hl.eval(hl.len(t0) == 0))
        self.assertTrue(hl.eval(hl.len(t2) == 2))
        self.assertTrue(hl.eval(hl.len(t)) == 5)

    def test_interval_ops(self):
        interval = hl.interval(3, 6)
        self.assertTrue(hl.eval_typed(interval.start) == (3, hl.tint))
        self.assertTrue(hl.eval_typed(interval.end) == (6, hl.tint))
        self.assertTrue(hl.eval_typed(interval.includes_start) == (True, hl.tbool))
        self.assertTrue(hl.eval_typed(interval.includes_end) == (False, hl.tbool))
        self.assertTrue(hl.eval_typed(interval.contains(5)) == (True, hl.tbool))
        self.assertTrue(hl.eval_typed(interval.overlaps(hl.interval(5, 9))) == (True, hl.tbool))

        li = hl.parse_locus_interval('1:100-110')
        self.assertEqual(hl.eval(li), hl.utils.Interval(hl.genetics.Locus("1", 100),
                                                     hl.genetics.Locus("1", 110)))
        self.assertTrue(li.dtype.point_type == hl.tlocus())
        self.assertTrue(hl.eval(li.contains(hl.locus("1", 100))))
        self.assertTrue(hl.eval(li.contains(hl.locus("1", 109))))
        self.assertFalse(hl.eval(li.contains(hl.locus("1", 110))))
    
        li2 = hl.parse_locus_interval("1:109-200")
        li3 = hl.parse_locus_interval("1:110-200")
        li4 = hl.parse_locus_interval("1:90-101")
        li5 = hl.parse_locus_interval("1:90-100")
    
        self.assertTrue(hl.eval(li.overlaps(li2)))
        self.assertTrue(hl.eval(li.overlaps(li4)))
        self.assertFalse(hl.eval(li.overlaps(li3)))
        self.assertFalse(hl.eval(li.overlaps(li5)))

    def test_reference_genome_fns(self):
        self.assertTrue(hl.eval(hl.is_valid_contig('1', 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_contig('chr1', 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_contig('1', 'GRCh38')))
        self.assertTrue(hl.eval(hl.is_valid_contig('chr1', 'GRCh38')))

        self.assertTrue(hl.eval(hl.is_valid_locus('1', 325423, 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_locus('1', 0, 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_locus('1', 249250622, 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_locus('chr1', 2645, 'GRCh37')))

    def test_initop(self):
        t = (hl.utils.range_table(5, 3)
             .annotate(GT=hl.call(0, 1))
             .annotate_globals(alleles=["A", "T"]))

        self.assertTrue(t.aggregate(agg.call_stats(t.GT, t.alleles)) ==
                        hl.Struct(AC=[5, 5], AF=[0.5, 0.5], AN=10, homozygote_count=[0, 0])) # Tests table.aggregate initOp

        mt = (hl.utils.range_matrix_table(10, 5, 5)
              .annotate_entries(GT=hl.call(0, 1))
              .annotate_rows(alleles=["A", "T"])
              .annotate_globals(alleles2=["G", "C"]))

        row_agg = mt.annotate_rows(call_stats=agg.call_stats(mt.GT, mt.alleles)).rows() # Tests MatrixMapRows initOp
        col_agg = mt.annotate_cols(call_stats=agg.call_stats(mt.GT, mt.alleles2)).cols() # Tests MatrixMapCols initOp

        # must test that call_stats isn't null, because equality doesn't test for that
        self.assertTrue(row_agg.all(
            hl.is_defined(row_agg.call_stats)
            & (row_agg.call_stats == hl.struct(AC=[5, 5], AF=[0.5, 0.5], AN=10, homozygote_count=[0, 0]))))
        self.assertTrue(col_agg.all(
            hl.is_defined(col_agg.call_stats)
            & (col_agg.call_stats == hl.struct(AC=[10, 10], AF=[0.5, 0.5], AN=20, homozygote_count=[0, 0]))))

        # test TableAggregateByKey initOp
        t2 = t.annotate(group=t.idx < 3)
        group_agg = t2.group_by(t2['group']).aggregate(call_stats=agg.call_stats(t2.GT, t2.alleles))

        self.assertTrue(group_agg.all(
            hl.cond(group_agg.group,
                    hl.is_defined(group_agg.call_stats)
                    & (group_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_agg.call_stats)
                    & (group_agg.call_stats == hl.struct(AC=[2, 2], AF=[0.5, 0.5], AN=4, homozygote_count=[0, 0])))))

        # test MatrixAggregateColsByKey entries initOp
        mt2 = mt.annotate_cols(group=mt.col_idx < 3)
        group_cols_agg = (mt2.group_cols_by(mt2['group'])
                          .aggregate(call_stats=agg.call_stats(mt2.GT, mt2.alleles2)).entries())

        self.assertTrue(group_cols_agg.all(
            hl.cond(group_cols_agg.group,
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[2, 2], AF=[0.5, 0.5], AN=4, homozygote_count=[0, 0])))))

        # test MatrixAggregateColsByKey cols initOp
        mt2 = mt.annotate_cols(group=mt.col_idx < 3, GT_col=hl.call(0, 1))
        group_cols_agg = (mt2.group_cols_by(mt2['group'])
                          .aggregate_cols(call_stats=agg.call_stats(mt2.GT_col, mt2.alleles2))
                          .result()
                          ).entries()

        self.assertTrue(group_cols_agg.all(
            hl.cond(group_cols_agg.group,
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[2, 2], AF=[0.5, 0.5], AN=4, homozygote_count=[0, 0])))))

        # test MatrixAggregateRowsByKey entries initOp
        mt2 = mt.annotate_rows(group=mt.row_idx < 3)
        group_rows_agg = (mt2.group_rows_by(mt2['group'])
                          .aggregate(call_stats=agg.call_stats(mt2.GT, mt2.alleles2)).entries())

        self.assertTrue(group_rows_agg.all(
            hl.cond(group_rows_agg.group,
                    hl.is_defined(group_rows_agg.call_stats)
                    & (group_rows_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_rows_agg.call_stats)
                    & (group_rows_agg.call_stats == hl.struct(AC=[7, 7], AF=[0.5, 0.5], AN=14, homozygote_count=[0, 0])))))

        # test MatrixAggregateRowsByKey rows initOp
        mt2 = mt.annotate_rows(group=mt.row_idx < 3, GT_row=hl.call(0, 1))
        group_rows_agg = (mt2.group_rows_by(mt2['group'])
                          .aggregate_rows(call_stats=agg.call_stats(mt2.GT_row, mt2.alleles2))
                          .result()
                          ).entries()

        self.assertTrue(group_rows_agg.all(
            hl.cond(group_rows_agg.group,
                    hl.is_defined(group_rows_agg.call_stats)
                    & (group_rows_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_rows_agg.call_stats)
                    & (group_rows_agg.call_stats == hl.struct(AC=[7, 7], AF=[0.5, 0.5], AN=14, homozygote_count=[0, 0])))))

    def test_mendel_error_code(self):
        locus_auto = hl.Locus('2', 20000000)
        locus_x_par = hl.get_reference('default').par[0].start
        locus_x_nonpar = hl.Locus(locus_x_par.contig, locus_x_par.position - 1)
        locus_y_nonpar = hl.Locus('Y', hl.get_reference('default').lengths['Y'] - 1)

        self.assertTrue(hl.eval(hl.all(lambda x: x, hl.array([
            hl.literal(locus_auto).in_autosome_or_par(),
            hl.literal(locus_auto).in_autosome_or_par(),
            ~hl.literal(locus_x_par).in_autosome(),
            hl.literal(locus_x_par).in_autosome_or_par(),
            ~hl.literal(locus_x_nonpar).in_autosome_or_par(),
            hl.literal(locus_x_nonpar).in_x_nonpar(),
            ~hl.literal(locus_y_nonpar).in_autosome_or_par(),
            hl.literal(locus_y_nonpar).in_y_nonpar()
        ]))))

        hr = hl.Call([0, 0])
        het = hl.Call([0, 1])
        hv = hl.Call([1, 1])
        nocall = None

        expected = {
            (locus_auto, True, hv, hv, het): 1,
            (locus_auto, False, hv, hv, het): 1,
            (locus_x_par, True, hv, hv, het): 1,
            (locus_x_par, False, hv, hv, het): 1,
            (locus_auto, True, hr, hr, het): 2,
            (locus_auto, None, hr, hr, het): 2,
            (locus_auto, True, hr, het, hv): 3,
            (locus_auto, True, hr, hv, hv): 3,
            (locus_auto, True, hr, nocall, hv): 3,
            (locus_auto, True, het, hr, hv): 4,
            (locus_auto, True, hv, hr, hv): 4,
            (locus_auto, True, nocall, hr, hv): 4,
            (locus_auto, True, hr, hr, hv): 5,
            (locus_auto, None, hr, hr, hv): 5,
            (locus_auto, False, hr, hr, hv): 5,
            (locus_x_par, False, hr, hr, hv): 5,
            (locus_x_par, False, hv, het, hr): 6,
            (locus_x_par, False, hv, hr, hr): 6,
            (locus_x_par, False, hv, nocall, hr): 6,
            (locus_x_par, False, het, hv, hr): 7,
            (locus_x_par, False, hr, hv, hr): 7,
            (locus_x_par, False, nocall, hv, hr): 7,
            (locus_auto, True, hv, hv, hr): 8,
            (locus_auto, False, hv, hv, hr): 8,
            (locus_x_par, False, hv, hv, hr): 8,
            (locus_x_par, None, hv, hv, hr): 8,
            (locus_x_nonpar, False, hr, hv, hr): 9,
            (locus_x_nonpar, False, het, hv, hr): 9,
            (locus_x_nonpar, False, hv, hv, hr): 9,
            (locus_x_nonpar, False, nocall, hv, hr): 9,
            (locus_x_nonpar, False, hr, hr, hv): 10,
            (locus_x_nonpar, False, het, hr, hv): 10,
            (locus_x_nonpar, False, hv, hr, hv): 10,
            (locus_x_nonpar, False, nocall, hr, hv): 10,
            (locus_y_nonpar, False, hv, hr, hr): 11,
            (locus_y_nonpar, False, hv, het, hr): 11,
            (locus_y_nonpar, False, hv, hv, hr): 11,
            (locus_y_nonpar, False, hv, nocall, hr): 11,
            (locus_y_nonpar, False, hr, hr, hv): 12,
            (locus_y_nonpar, False, hr, het, hv): 12,
            (locus_y_nonpar, False, hr, hv, hv): 12,
            (locus_y_nonpar, False, hr, nocall, hv): 12,
            (locus_auto, True, het, het, het): None,
            (locus_auto, True, hv, het, het): None,
            (locus_auto, True, het, hr, het): None,
            (locus_auto, True, hv, hr, het): None,
            (locus_auto, True, hv, hr, het): None,
            (locus_x_nonpar, True, hv, hr, het): None,
            (locus_x_nonpar, False, hv, hr, hr): None,
            (locus_x_nonpar, None, hv, hr, hr): None,
            (locus_x_nonpar, False, het, hr, hr): None,
            (locus_y_nonpar, True, het, hr, het): None,
            (locus_y_nonpar, True, het, hr, hr): None,
            (locus_y_nonpar, True, het, hr, het): None,
            (locus_y_nonpar, True, het, het, het): None,
            (locus_y_nonpar, True, hr, hr, hr): None,
            (locus_y_nonpar, None, hr, hr, hr): None,
            (locus_y_nonpar, False, hr, hv, hr): None,
            (locus_y_nonpar, False, hv, hv, hv): None,
            (locus_y_nonpar, None, hv, hv, hv): None,
        }

        arg_list = hl.literal(list(expected.keys()),
                              hl.tarray(hl.ttuple(hl.tlocus(), hl.tbool, hl.tcall, hl.tcall, hl.tcall)))
        values = arg_list.map(lambda args: hl.mendel_error_code(*args))
        expr = hl.dict(hl.zip(arg_list, values))
        results = hl.eval(expr)
        for args, result in results.items():
            self.assertEqual(result, expected[args], msg=f'expected {expected[args]}, found {result} at {str(args)}')

    def test_min_rep(self):
        def assert_min_reps_to(old, new, pos_change=0):
            self.assertEqual(
                hl.eval(hl.min_rep(hl.locus('1', 10), old)),
                hl.Struct(locus=hl.Locus('1', 10 + pos_change), alleles=new)
            )

        assert_min_reps_to(['TAA', 'TA'], ['TA', 'T'])
        assert_min_reps_to(['ACTG', 'ACT'], ['TG', 'T'], pos_change=2)
        assert_min_reps_to(['AAACAAAC', 'AAAC'], ['AAACA', 'A'])
        assert_min_reps_to(['AATAA', 'AAGAA'], ['T', 'G'], pos_change=2)
        assert_min_reps_to(['AATAA', '*'], ['A', '*'])
        assert_min_reps_to(['TAA', 'TA', 'TTA'], ['TA', 'T', 'TT'])
        assert_min_reps_to(['GCTAA', 'GCAAA', 'G'], ['GCTAA', 'GCAAA', 'G'])
        assert_min_reps_to(['GCTAA', 'GCAAA', 'GCCAA'], ['T', 'A', 'C'], pos_change=2)
        assert_min_reps_to(['GCTAA', 'GCAAA', 'GCCAA', '*'], ['T', 'A', 'C', '*'], pos_change=2)

    def assert_evals_to(self, e, v):
        self.assertEqual(hl.eval(e), v)

    def test_set_functions(self):
        s = hl.set([1, 3, 7])
        t = hl.set([3, 8])

        self.assert_evals_to(s, set([1, 3, 7]))

        self.assert_evals_to(s.add(3), set([1, 3, 7]))
        self.assert_evals_to(s.add(4), set([1, 3, 4, 7]))

        self.assert_evals_to(s.remove(3), set([1, 7]))
        self.assert_evals_to(s.remove(4), set([1, 3, 7]))

        self.assert_evals_to(s.contains(3), True)
        self.assert_evals_to(s.contains(4), False)

        self.assert_evals_to(s.difference(t), set([1, 7]))
        self.assert_evals_to(s.intersection(t), set([3]))

        self.assert_evals_to(s.is_subset(hl.set([1, 3, 4, 7])), True)
        self.assert_evals_to(s.is_subset(hl.set([1, 3])), False)

        self.assert_evals_to(s.union(t), set([1, 3, 7, 8]))

    def test_set_numeric_functions(self):
        s = hl.set([1, 3, 5, hl.null(tint32)])
        self.assert_evals_to(hl.min(s), 1)
        self.assert_evals_to(hl.max(s), 5)
        self.assert_evals_to(hl.mean(s), 3)
        self.assert_evals_to(hl.median(s), 3)

    def test_uniroot(self):
        self.assertAlmostEqual(hl.eval(hl.uniroot(lambda x: x - 1, 0, 3)), 1)

    def test_chi_squared_test(self):
        res = hl.eval(hl.chi_squared_test(0, 0, 0, 0))
        self.assertTrue(math.isnan(res['p_value']))
        self.assertTrue(math.isnan(res['odds_ratio']))

        res = hl.eval(hl.chi_squared_test(51, 43, 22, 92))
        self.assertAlmostEqual(res['p_value'] / 1.462626e-7, 1.0, places=4)
        self.assertAlmostEqual(res['odds_ratio'], 4.95983087)
        
        res = hl.eval(hl.chi_squared_test(61, 17493, 95, 84145))
        self.assertAlmostEqual(res['p_value'] / 4.74710374e-13, 1.0, places=4)
        self.assertAlmostEqual(res['odds_ratio'], 3.08866103)

    def test_fisher_exact_test(self):
        res = hl.eval(hl.fisher_exact_test(0, 0, 0, 0))
        self.assertTrue(math.isnan(res['p_value']))
        self.assertTrue(math.isnan(res['odds_ratio']))
        self.assertTrue(math.isnan(res['ci_95_lower']))
        self.assertTrue(math.isnan(res['ci_95_upper']))

        res = hl.eval(hl.fisher_exact_test(51, 43, 22, 92))
        self.assertAlmostEqual(res['p_value'] / 2.1565e-7, 1.0, places=4)
        self.assertAlmostEqual(res['odds_ratio'], 4.91805817)
        self.assertAlmostEqual(res['ci_95_lower'], 2.56593733)
        self.assertAlmostEqual(res['ci_95_upper'], 9.67792963)

    def test_contingency_table_test(self):
        res = hl.eval(hl.contingency_table_test(51, 43, 22, 92, 22))
        self.assertAlmostEqual(res['p_value'] / 1.462626e-7, 1.0, places=4)
        self.assertAlmostEqual(res['odds_ratio'], 4.95983087)

        res = hl.eval(hl.contingency_table_test(51, 43, 22, 92, 23))
        self.assertAlmostEqual(res['p_value'] / 2.1565e-7, 1.0, places=4)
        self.assertAlmostEqual(res['odds_ratio'], 4.91805817)

    def test_hardy_weinberg_test(self):
        res = hl.eval(hl.hardy_weinberg_test(1, 2, 1))
        self.assertAlmostEqual(res['p_value'], 0.65714285)
        self.assertAlmostEqual(res['het_freq_hwe'], 0.57142857)

    def test_pl_to_gp(self):
        res = hl.eval(hl.pl_to_gp([0, 10, 100]))
        self.assertAlmostEqual(res[0], 0.9090909090082644)
        self.assertAlmostEqual(res[1], 0.09090909090082644)
        self.assertAlmostEqual(res[2], 9.090909090082645e-11)

    def test_pl_dosage(self):
        self.assertAlmostEqual(hl.eval(hl.pl_dosage([0, 20, 100])), 0.009900990296049406)
        self.assertAlmostEqual(hl.eval(hl.pl_dosage([20, 0, 100])), 0.9900990100009803)
        self.assertAlmostEqual(hl.eval(hl.pl_dosage([20, 100, 0])), 1.980198019704931)
        self.assertIsNone(hl.eval(hl.pl_dosage([20, hl.null('int'), 100])))

    def test_collection_method_missingness(self):
        a = [1, hl.null('int')]

        self.assertEqual(hl.eval(hl.min(a)), 1)
        self.assertIsNone(hl.eval(hl.min(a, filter_missing=False)))

        self.assertEqual(hl.eval(hl.max(a)), 1)
        self.assertIsNone(hl.eval(hl.max(a, filter_missing=False)))

        self.assertEqual(hl.eval(hl.mean(a)), 1)
        self.assertIsNone(hl.eval(hl.mean(a, filter_missing=False)))

        self.assertEqual(hl.eval(hl.product(a)), 1)
        self.assertIsNone(hl.eval(hl.product(a, filter_missing=False)))

        self.assertEqual(hl.eval(hl.sum(a)), 1)
        self.assertIsNone(hl.eval(hl.sum(a, filter_missing=False)))

    def test_literal_with_nested_expr(self):
        self.assertEqual(hl.eval(hl.literal(hl.set(['A','B']))), {'A', 'B'})
        self.assertEqual(hl.eval(hl.literal({hl.str('A'), hl.str('B')})), {'A', 'B'})

    def test_format(self):
        self.assertEqual(hl.eval(hl.format("%.4f %s %.3e", 0.25, 'hello', 0.114)), '0.2500 hello 1.140e-01')
        self.assertEqual(hl.eval(hl.format("%.4f %d", hl.null(hl.tint32), hl.null(hl.tint32))), 'null null')
        self.assertEqual(hl.eval(hl.format("%s", hl.struct(foo=5, bar=True, baz=hl.array([4, 5])))), '[5,true,[4,5]]')
        self.assertEqual(hl.eval(hl.format("%s %s", hl.locus("1", 356), hl.tuple([9, True, hl.null(hl.tstr)]))), '1:356 [9,true,null]')
        self.assertEqual(hl.eval(hl.format("%b %B %b %b", hl.null(hl.tint), hl.null(hl.tstr), True, "hello")), "false FALSE true true")

    def test_dict_and_set_type_promotion(self):
        d = hl.literal({5: 5}, dtype='dict<int64, int64>')
        s = hl.literal({5}, dtype='set<int64>')

        self.assertEqual(hl.eval(d[5]), 5)
        self.assertEqual(hl.eval(d.get(5)), 5)
        self.assertEqual(hl.eval(d.get(2, 3)), 3)
        self.assertTrue(hl.eval(d.contains(5)))
        self.assertTrue(hl.eval(~d.contains(2)))

        self.assertTrue(hl.eval(s.contains(5)))
        self.assertTrue(hl.eval(~s.contains(2)))

    def test_nan_roundtrip(self):
        a = [math.nan, math.inf, -math.inf, 0, 1]
        round_trip = hl.eval(hl.literal(a))
        self.assertTrue(math.isnan(round_trip[0]))
        self.assertTrue(math.isinf(round_trip[1]))
        self.assertTrue(math.isinf(round_trip[2]))
        self.assertEqual(round_trip[-2:], [0, 1])

    def test_approx_equal(self):
        self.assertTrue(hl.eval(hl.approx_equal(0.25, 0.25000001)))
        self.assertTrue(hl.eval(hl.approx_equal(hl.null(hl.tint64), 5)) is None)
        self.assertFalse(hl.eval(hl.approx_equal(0.25, 0.251, absolute=True, tolerance=1e-3)))

    def test_issue3729(self):
        t = hl.utils.range_table(10, 3)
        fold_expr = hl.cond(t.idx == 3,
                            [1, 2, 3],
                            [4, 5, 6]).fold(lambda accum, i: accum & (i == t.idx),
                                            True)
        t.annotate(foo=hl.cond(fold_expr, 1, 3))._force_count()

    def assertValueEqual(self, expr, value, t):
        self.assertEqual(expr.dtype, t)
        self.assertEqual(hl.eval(expr), value)

    def test_array_fold_and_scan(self):
        self.assertValueEqual(hl.fold(lambda x, y: x + y, 0, [1, 2, 3]), 6, tint32)
        self.assertValueEqual(hl.array_scan(lambda x, y: x + y, 0, [1, 2, 3]), [0, 1, 3, 6], tarray(tint32))

        self.assertValueEqual(hl.fold(lambda x, y: x + y, 0., [1, 2, 3]), 6., tfloat64)
        self.assertValueEqual(hl.fold(lambda x, y: x + y, 0, [1., 2., 3.]), 6., tfloat64)
        self.assertValueEqual(hl.array_scan(lambda x, y: x + y, 0., [1, 2, 3]), [0., 1., 3., 6.], tarray(tfloat64))
        self.assertValueEqual(hl.array_scan(lambda x, y: x + y, 0, [1., 2., 3.]), [0., 1., 3., 6.], tarray(tfloat64))

    def test_cumulative_sum(self):
        self.assertValueEqual(hl.cumulative_sum([1, 2, 3, 4]), [1, 3, 6, 10], tarray(tint32))
        self.assertValueEqual(hl.cumulative_sum([1.0, 2.0, 3.0, 4.0]), [1.0, 3.0, 6.0, 10.0], tarray(tfloat64))

    def test_nan_inf_checks(self):
        finite = 0
        infinite = float('inf')
        nan = math.nan
        na = hl.null('float64')

        assert hl.eval(hl.is_finite(finite)) == True
        assert hl.eval(hl.is_finite(infinite)) == False
        assert hl.eval(hl.is_finite(nan)) == False
        assert hl.eval(hl.is_finite(na)) == None

        assert hl.eval(hl.is_infinite(finite)) == False
        assert hl.eval(hl.is_infinite(infinite)) == True
        assert hl.eval(hl.is_infinite(nan)) == False
        assert hl.eval(hl.is_infinite(na)) == None

        assert hl.eval(hl.is_nan(finite)) == False
        assert hl.eval(hl.is_nan(infinite)) == False
        assert hl.eval(hl.is_nan(nan)) == True
        assert hl.eval(hl.is_nan(na)) == None

    def test_array_and_if_requiredness(self):
        mt = hl.import_vcf(resource('sample.vcf'), array_elements_required=True)
        hl.tuple((mt.AD, mt.PL)).show()
        hl.array([mt.AD, mt.PL]).show()
        hl.array([mt.AD, [1,2]]).show()
