import math
import pytest
import random
from scipy.stats import pearsonr
import numpy as np

import hail as hl
import hail.expr.aggregators as agg
from hail.expr.types import *
from hail.expr.functions import _error_from_cdf, _cdf_combine, _result_from_raw_cdf
import hail.ir as ir
from ..helpers import *


def _test_many_equal(test_cases):
    expressions = [t[0] for t in test_cases]
    actuals = hl.eval(hl.tuple(expressions))
    expecteds = [t[1] for t in test_cases]
    for actual, expected in zip(actuals, expecteds):
        if actual != expected:
            raise ValueError(f'  actual: {actual}\n  expected: {expected}')


def _test_many_equal_typed(test_cases):
    expressions = [t[0] for t in test_cases]
    actuals, actual_type = hl.eval_typed(hl.tuple(expressions))
    assert isinstance(actual_type, hl.ttuple)
    expecteds = [t[1] for t in test_cases]
    expected_types = [t[2] for t in test_cases]
    for expression, actual, expected, actual_type, expected_type in zip(
        expressions, actuals, expecteds, actual_type.types, expected_types
    ):
        assert expression.dtype == expected_type, (expression.dtype, expected_type)
        assert actual_type == expected_type, (actual_type, expected_type)
        assert actual == expected, (actual, expected)


class Tests(unittest.TestCase):
    def collect_unindexed_expression(self):
        self.assertEqual(hl.array([4, 1, 2, 3]).collect(), [4, 1, 2, 3])

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
            self.assertTrue(ht.aggregate(hl.agg.all((ht.x == ht.y)) & ~hl.agg.all((ht.x == ht.z))))

        test_random_function(lambda: hl.rand_unif(0, 1))
        test_random_function(lambda: hl.rand_int32(10))
        test_random_function(lambda: hl.rand_int32(5, 15))
        test_random_function(lambda: hl.rand_int64(23))
        test_random_function(lambda: hl.rand_int64(5, 15))
        test_random_function(lambda: hl.rand_int64(1 << 33, 1 << 35))
        test_random_function(lambda: hl.rand_bool(0.5))
        test_random_function(lambda: hl.rand_norm(0, 1))
        test_random_function(lambda: hl.rand_pois(1))
        test_random_function(lambda: hl.rand_beta(1, 1))
        test_random_function(lambda: hl.rand_beta(1, 1, 0, 1))
        test_random_function(lambda: hl.rand_gamma(1, 1))
        test_random_function(lambda: hl.rand_cat(hl.array([1, 1, 1, 1])))
        test_random_function(lambda: hl.rand_dirichlet(hl.array([1, 1, 1, 1])))

    def test_range(self):
        def same_as_python(*args):
            self.assertEqual(hl.eval(hl.range(*args)), list(range(*args)))

        same_as_python(10)
        same_as_python(3, 10)
        same_as_python(3, 10, 2)
        same_as_python(3, 10, 3)
        same_as_python(-5)
        same_as_python(10, -5)
        same_as_python(10, -5, -1)
        same_as_python(10, -5, -4)

        with self.assertRaisesRegex(hl.utils.HailUserError, 'Array range cannot have step size 0'):
            hl.eval(hl.range(0, 1, 0))

    def test_zeros(self):
        for size in [0, 3, 10, 1000]:
            evaled = hl.eval(hl.zeros(size))
            assert evaled == [0 for i in range(size)]

    def test_seeded_sampling(self):
        sampled1 = hl.utils.range_table(50, 6).filter(hl.rand_bool(0.5))
        sampled2 = hl.utils.range_table(50, 5).filter(hl.rand_bool(0.5))

        set1 = set(sampled1.idx.collect())
        set2 = set(sampled2.idx.collect())
        expected = set1 & set2

        for i in range(7):
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

        rows = [
            {'a': 4, 'b': 1, 'c': 3, 'd': 5, 'e': "hello", 'f': [1, 2, 3]},
            {'a': 0, 'b': 5, 'c': 13, 'd': -1, 'e': "cat", 'f': []},
            {'a': 4, 'b': 2, 'c': 20, 'd': 3, 'e': "dog", 'f': [5, 6, 7]},
        ]

        kt = hl.Table.parallelize(rows, schema)

        result = convert_struct_to_dict(
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
                x36=True,
                x37=kt.e > "helln",
                x38=kt.e < "hellp",
                x39=kt.e <= "hello",
                x40=kt.e >= "hello",
                x41="helln" > kt.e,
                x42="hellp" < kt.e,
                x43="hello" >= kt.e,
                x44="hello" <= kt.e,
                x45=kt.f > [1, 2],
                x46=kt.f < [1, 3],
                x47=kt.f >= [1, 2, 3],
                x48=kt.f <= [1, 2, 3],
                x49=kt.f < [1.0, 2.0],
                x50=kt.f > [1.0, 3.0],
                x51=[1.0, 2.0, 3.0] <= kt.f,
                x52=[1.0, 2.0, 3.0] >= kt.f,
                x53=hl.tuple([True, 1.0]) < (1.0, 0.0),
                x54=kt.e * kt.a,
            ).take(1)[0]
        )

        expected = {
            'a': 4,
            'b': 1,
            'c': 3,
            'd': 5,
            'e': "hello",
            'f': [1, 2, 3],
            'x1': 9,
            'x2': 9,
            'x3': 5,
            'x4': -1,
            'x5': 1,
            'x6': 3,
            'x7': 20,
            'x8': 20,
            'x9': 4,
            'x10': 4.0 / 5,
            'x11': 5.0 / 4,
            'x12': 4,
            'x13': -4,
            'x14': 4,
            'x15': False,
            'x16': False,
            'x17': False,
            'x18': True,
            'x19': True,
            'x20': True,
            'x21': True,
            'x22': False,
            'x23': True,
            'x24': True,
            'x25': False,
            'x26': True,
            'x27': False,
            'x28': True,
            'x29': False,
            'x30': False,
            'x31': True,
            'x32': False,
            'x33': False,
            'x34': False,
            'x35': False,
            'x36': True,
            'x37': True,
            'x38': True,
            'x39': True,
            'x40': True,
            'x41': False,
            'x42': False,
            'x43': True,
            'x44': True,
            'x45': True,
            'x46': True,
            'x47': True,
            'x48': True,
            'x49': False,
            'x50': False,
            'x51': True,
            'x52': True,
            'x53': False,
            'x54': "hellohellohellohello",
        }

        for k, v in expected.items():
            if isinstance(v, float):
                self.assertAlmostEqual(v, result[k], msg=k)
            else:
                self.assertEqual(v, result[k], msg=k)

    def test_array_slicing(self):
        schema = hl.tstruct(a=hl.tarray(hl.tint32))
        rows = [{'a': [1, 2, 3, 4, 5]}]
        kt = hl.Table.parallelize(rows, schema)
        ha = hl.array(hl.range(100))
        pa = list(range(100))

        result = convert_struct_to_dict(
            kt.annotate(
                x1=kt.a[0],
                x2=kt.a[2],
                x3=kt.a[:],
                x4=kt.a[1:2],
                x5=kt.a[-1:4],
                x6=kt.a[:2],
                x7=kt.a[-20:20:-2],
                x8=kt.a[20:-20:2],
                x9=kt.a[-20:20:2],
                x10=kt.a[20:-20:-2],
            ).take(1)[0]
        )

        expected = {
            'a': [1, 2, 3, 4, 5],
            'x1': 1,
            'x2': 3,
            'x3': [1, 2, 3, 4, 5],
            'x4': [2],
            'x5': [],
            'x6': [1, 2],
            'x7': [],
            'x8': [],
            'x9': [1, 3, 5],
            'x10': [5, 3, 1],
        }

        self.assertDictEqual(result, expected)
        self.assertEqual(pa[60:1:-3], hl.eval(ha[hl.int32(60) : hl.int32(1) : hl.int32(-3)]))
        self.assertEqual(pa[::5], hl.eval(ha[:: hl.int32(5)]))
        self.assertEqual(pa[::-3], hl.eval(ha[::-3]))
        self.assertEqual(pa[:-77:-3], hl.eval(ha[: hl.int32(-77) : -3]))
        self.assertEqual(pa[44::-7], hl.eval(ha[44::-7]))
        self.assertEqual(pa[2:59:7], hl.eval(ha[2:59:7]))
        self.assertEqual(pa[4:40:2], hl.eval(ha[4:40:2]))
        self.assertEqual(pa[-400:-300:2], hl.eval(ha[hl.int32(-400) : -300 : 2]))
        self.assertEqual(pa[-300:-400:-2], hl.eval(ha[-300:-400:-2]))
        self.assertEqual(pa[300:400:2], hl.eval(ha[300:400:2]))
        self.assertEqual(pa[400:300:-2], hl.eval(ha[400:300:-2]))

        with pytest.raises(hl.utils.HailUserError, match='step cannot be 0 for array slice'):
            hl.eval(ha[::0])

    def test_dict_methods(self):
        schema = hl.tstruct(x=hl.tfloat64)
        rows = [{'x': 2.0}]
        kt = hl.Table.parallelize(rows, schema)

        kt = kt.annotate(a={'cat': 3, 'dog': 7})

        result = convert_struct_to_dict(
            kt.annotate(
                x1=kt.a['cat'],
                x2=kt.a['dog'],
                x3=kt.a.keys().contains('rabbit'),
                x4=kt.a.size() == 0,
                x5=kt.a.key_set(),
                x6=kt.a.keys(),
                x7=kt.a.values(),
                x8=kt.a.size(),
                x9=kt.a.map_values(lambda v: v * 2.0),
                x10=kt.a.items(),
            ).take(1)[0]
        )

        expected = {
            'a': {'cat': 3, 'dog': 7},
            'x': 2.0,
            'x1': 3,
            'x2': 7,
            'x3': False,
            'x4': False,
            'x5': {'cat', 'dog'},
            'x6': ['cat', 'dog'],
            'x7': [3, 7],
            'x8': 2,
            'x9': {'cat': 6.0, 'dog': 14.0},
            'x10': [('cat', 3), ('dog', 7)],
        }

        self.assertDictEqual(result, expected)

    def test_dict_missing_error(self):
        d = hl.dict({'a': 2, 'b': 3})
        with pytest.raises(hl.utils.HailUserError, match='Key NA not found in dictionary'):
            hl.eval(d[hl.missing(hl.tstr)])

    def test_numeric_conversion(self):
        schema = hl.tstruct(a=hl.tfloat64, b=hl.tfloat64, c=hl.tint32, d=hl.tint32)
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': 5}]
        kt = hl.Table.parallelize(rows, schema)
        kt = kt.annotate(d=hl.int64(kt.d))

        kt = kt.annotate(x1=[1.0, kt.a, 1], x2=[1, 1.0], x3=[kt.a, kt.c], x4=[kt.c, kt.d], x5=[1, kt.c])

        expected_schema = {
            'a': hl.tfloat64,
            'b': hl.tfloat64,
            'c': hl.tint32,
            'd': hl.tint64,
            'x1': hl.tarray(hl.tfloat64),
            'x2': hl.tarray(hl.tfloat64),
            'x3': hl.tarray(hl.tfloat64),
            'x4': hl.tarray(hl.tint64),
            'x5': hl.tarray(hl.tint32),
        }

        for f, t in kt.row.dtype.items():
            self.assertEqual(expected_schema[f], t)

    def test_genetics_constructors(self):
        rg = hl.ReferenceGenome("foo", ["1"], {"1": 100})

        schema = hl.tstruct(a=hl.tfloat64, b=hl.tfloat64, c=hl.tint32, d=hl.tint32)
        rows = [{'a': 2.0, 'b': 4.0, 'c': 1, 'd': 5}]
        kt = hl.Table.parallelize(rows, schema)
        kt = kt.annotate(d=hl.int64(kt.d))

        kt = kt.annotate(
            l1=hl.parse_locus("1:51"),
            l2=hl.locus("1", 51, reference_genome=rg),
            i1=hl.parse_locus_interval("1:51-56", reference_genome=rg),
            i2=hl.interval(hl.locus("1", 51, reference_genome=rg), hl.locus("1", 56, reference_genome=rg)),
        )

        expected_schema = {
            'a': hl.tfloat64,
            'b': hl.tfloat64,
            'c': hl.tint32,
            'd': hl.tint64,
            'l1': hl.tlocus(),
            'l2': hl.tlocus(rg),
            'i1': hl.tinterval(hl.tlocus(rg)),
            'i2': hl.tinterval(hl.tlocus(rg)),
        }

        self.assertTrue(all([expected_schema[f] == t for f, t in kt.row.dtype.items()]))

    def test_floating_point(self):
        self.assertEqual(hl.eval(1.1e-15), 1.1e-15)

    def test_bind_multiple(self):
        self.assertEqual(hl.eval(hl.bind(lambda x, y: x * y, 2, 3)), 6)
        self.assertEqual(hl.eval(hl.bind(lambda y: y * 2, 3)), 6)

    def test_bind_placement(self):
        self.assertEqual(hl.eval(5 / hl.bind(lambda x: x, 5)), 1.0)

    def test_rbind_multiple(self):
        self.assertEqual(hl.eval(hl.rbind(2, 3, lambda x, y: x * y)), 6)
        self.assertEqual(hl.eval(hl.rbind(3, lambda y: y * 2)), 6)

    def test_rbind_placement(self):
        self.assertEqual(hl.eval(5 / hl.rbind(5, lambda x: x)), 1.0)

    def test_translate(self):
        strs = [None, '', 'TATAN']
        assert hl.eval(hl.literal(strs, 'array<str>').map(lambda x: x.translate({'T': 'A', 'A': 'T'}))) == [
            None,
            '',
            'ATATN',
        ]

        with pytest.raises(hl.utils.FatalError, match='mapping keys must be one character'):
            hl.eval(hl.str('foo').translate({'foo': 'bar'}))

        with pytest.raises(hl.utils.FatalError, match='mapping keys must be one character'):
            hl.eval(hl.str('foo').translate({'': 'bar'}))

    def test_reverse_complement(self):
        strs = ['NNGATTACA', 'NNGATTACA'.lower(), 'foo']
        rna_strs = ['NNGATTACA', 'NNGAUUACA'.lower(), 'foo']
        assert hl.eval(hl.literal(strs).map(lambda s: hl.reverse_complement(s))) == [
            'TGTAATCNN',
            'TGTAATCNN'.lower(),
            'oof',
        ]
        assert hl.eval(hl.literal(rna_strs).map(lambda s: hl.reverse_complement(s, rna=True))) == [
            'UGUAAUCNN',
            'UGUAAUCNN'.lower(),
            'oof',
        ]

    def test_matches(self):
        self.assertEqual(hl.eval('\\d+'), '\\d+')
        string = hl.literal('12345')
        self.assertTrue(hl.eval(string.matches('\\d+')))
        self.assertTrue(hl.eval(string.matches(hl.str('\\d+'))))
        self.assertFalse(hl.eval(string.matches(r'\\d+')))

    def test_string_reverse(self):
        inputs = ['', None, 'ATAT', 'foo']
        assert hl.eval(hl.literal(inputs, 'array<str>').map(lambda s: s.reverse())) == ['', None, 'TATA', 'oof']

    def test_first_match_in(self):
        string = hl.literal('1:25-100')
        self.assertTrue(hl.eval(string.first_match_in("([^:]*)[:\\t](\\d+)[\\-\\t](\\d+)")) == ['1', '25', '100'])
        self.assertIsNone(hl.eval(string.first_match_in(r"hello (\w+)!")))

    def test_string_join(self):
        self.assertEqual(hl.eval(hl.str(":").join(["foo", "bar", "baz"])), "foo:bar:baz")
        self.assertEqual(hl.eval(hl.str(",").join(hl.empty_array(hl.tstr))), "")

        with pytest.raises(TypeError, match="Expected str collection, int32 found"):
            hl.eval(hl.str(",").join([1, 2, 3]))

    def test_string_multiply(self):
        # Want to make sure all implict conversions work.
        ps = "cat"
        pn = 3
        s = hl.str(ps)
        n = hl.int32(pn)
        assert all([x == "catcatcat" for x in hl.eval(hl.array([ps * n, n * ps, s * pn, pn * s]))])

    def test_cond(self):
        self.assertEqual(hl.eval('A' + hl.if_else(True, 'A', 'B')), 'AA')

        self.assertEqual(hl.eval(hl.if_else(True, hl.struct(), hl.missing(hl.tstruct()))), hl.utils.Struct())
        self.assertEqual(hl.eval(hl.if_else(hl.missing(hl.tbool), 1, 2)), None)
        self.assertEqual(hl.eval(hl.if_else(hl.missing(hl.tbool), 1, 2, missing_false=True)), 2)

    def test_if_else(self):
        self.assertEqual(hl.eval('A' + hl.if_else(True, 'A', 'B')), 'AA')

        self.assertEqual(hl.eval(hl.if_else(True, hl.struct(), hl.missing(hl.tstruct()))), hl.utils.Struct())
        self.assertEqual(hl.eval(hl.if_else(hl.missing(hl.tbool), 1, 2)), None)
        self.assertEqual(hl.eval(hl.if_else(hl.missing(hl.tbool), 1, 2, missing_false=True)), 2)

    @qobtest
    def test_aggregators(self):
        table = hl.utils.range_table(10)
        r = table.aggregate(
            hl.struct(
                x=hl.agg.count(),
                y=hl.agg.count_where(table.idx % 2 == 0),
                z=hl.agg.filter(table.idx % 2 == 0, hl.agg.count()),
                arr_sum=hl.agg.array_sum([1, 2, hl.missing(tint32)]),
                bind_agg=hl.agg.count_where(hl.bind(lambda x: x % 2 == 0, table.idx)),
                mean=hl.agg.mean(table.idx),
                mean2=hl.agg.mean(hl.if_else(table.idx == 9, table.idx, hl.missing(tint32))),
                foo=hl.min(3, hl.agg.sum(table.idx)),
            )
        )

        self.assertEqual(r.x, 10)
        self.assertEqual(r.y, 5)
        self.assertEqual(r.z, 5)
        self.assertEqual(r.arr_sum, [10, 20, 0])
        self.assertEqual(r.mean, 4.5)
        self.assertEqual(r.mean2, 9)
        self.assertEqual(r.bind_agg, 5)
        self.assertEqual(r.foo, 3)

        a = hl.literal([1, 2], tarray(tint32))
        self.assertEqual(table.aggregate(hl.agg.filter(True, hl.agg.array_sum(a))), [10, 20])

        r = table.aggregate(
            hl.struct(
                fraction_odd=hl.agg.fraction(table.idx % 2 == 0),
                lessthan6=hl.agg.fraction(table.idx < 6),
                gt6=hl.agg.fraction(table.idx > 6),
                assert1=hl.agg.fraction(table.idx > 6) < 0.50,
                assert2=hl.agg.fraction(table.idx < 6) >= 0.50,
            )
        )
        self.assertEqual(r.fraction_odd, 0.50)
        self.assertEqual(r.lessthan6, 0.60)
        self.assertEqual(r.gt6, 0.30)
        self.assertTrue(r.assert1)
        self.assertTrue(r.assert2)

    def test_agg_nesting(self):
        t = hl.utils.range_table(10)
        aggregated_count = t.aggregate(hl.agg.count(), _localize=False)  # 10

        filter_count = t.aggregate(hl.agg.filter(aggregated_count == 10, hl.agg.count()))
        self.assertEqual(filter_count, 10)

        exploded_count = t.aggregate(hl.agg.explode(lambda x: hl.agg.count(), hl.range(hl.int32(aggregated_count))))
        self.assertEqual(exploded_count, 100)

        grouped_count = t.aggregate(hl.agg.group_by(aggregated_count, hl.agg.count()))
        self.assertEqual(grouped_count, {10: 10})

        array_agg_count = t.aggregate(hl.agg.array_agg(lambda x: hl.agg.count(), hl.range(hl.int32(aggregated_count))))
        self.assertEqual(array_agg_count, [10 for i in range(10)])

    def test_counter_ordering(self):
        ht = hl.utils.range_table(10)
        assert ht.aggregate(hl.agg.counter(10 - ht.idx).get(10, -1)) == 1

    def test_counter(self):
        a = hl.literal(["rabbit", "rabbit", None, "cat", "dog", None], dtype='array<str>')
        b = hl.literal([[], [], [1, 2, 3], [1, 2], [1, 2, 3], None], dtype='array<array<int>>')

        ht = hl.utils.range_table(6)
        ac, bc = ht.aggregate(hl.tuple([hl.agg.counter(a[ht.idx]), hl.array(hl.agg.counter(b[ht.idx]))]))
        assert ac == {'rabbit': 2, 'cat': 1, 'dog': 1, None: 2}
        assert bc == [([], 2), ([1, 2], 1), ([1, 2, 3], 2), (None, 1)]

        c = hl.literal([0, 0, 3, 2, 3, 0], dtype='array<int>')
        actual = ht.aggregate(hl.agg.counter(a[ht.idx], weight=c[ht.idx]))
        expected = {'rabbit': 0, 'cat': 2, 'dog': 3, None: 3}
        assert actual == expected

        c = hl.literal([0.0, 0.0, 3.0, 2.0, 3.0, 0.0], dtype='array<float>')
        actual = ht.aggregate(hl.agg.counter(a[ht.idx], weight=c[ht.idx]))
        expected = {'rabbit': 0.0, 'cat': 2.0, 'dog': 3.0, None: 3.0}
        assert actual == expected

    def test_aggfold_agg(self):
        ht = hl.utils.range_table(100, 5)
        self.assertEqual(ht.aggregate(hl.agg.fold(0, lambda x: x + ht.idx, lambda a, b: a + b)), 4950)

        ht = ht.annotate(s=hl.struct(x=ht.idx, y=ht.idx + 1))
        sum_and_product = ht.aggregate(
            hl.agg.fold(
                hl.struct(x=0, y=1.0),
                lambda accum: hl.struct(x=accum.x + ht.s.x, y=accum.y * ht.s.y),
                lambda a, b: hl.struct(x=a.x + b.x, y=a.y * b.y),
            )
        )
        self.assertEqual(sum_and_product, hl.Struct(x=4950, y=9.332621544394414e157))

        ht = ht.annotate(maybe=hl.if_else(ht.idx % 2 == 0, ht.idx, hl.missing(hl.tint32)))
        sum_evens_missing = ht.aggregate(hl.agg.fold(0, lambda x: x + ht.maybe, lambda a, b: a + b))
        assert sum_evens_missing is None
        sum_evens_only = ht.aggregate(
            hl.agg.fold(0, lambda x: x + hl.coalesce(ht.maybe, 0), lambda a, b: hl.coalesce(a + b, a, b))
        )
        self.assertEqual(sum_evens_only, 2450)

        # Testing types work out
        sum_float64 = ht.aggregate(
            hl.agg.fold(
                hl.int32(0),
                lambda acc: acc + hl.float32(ht.idx),
                lambda acc1, acc2: hl.float64(acc1) + hl.float64(acc2),
            )
        )
        self.assertEqual(sum_float64, 4950.0)

        ht = ht.annotate_globals(foo=7)
        with pytest.raises(hl.utils.java.HailUserError) as exc:
            ht.aggregate(hl.agg.fold(0, lambda x: x + ht.idx, lambda a, b: a + b + ht.foo))
        assert "comb_op function of fold cannot reference any fields" in str(exc.value)

        mt = hl.utils.range_matrix_table(100, 10)
        self.assertEqual(mt.aggregate_rows(hl.agg.fold(0, lambda a: a + mt.row_idx, lambda a, b: a + b)), 4950)
        self.assertEqual(mt.aggregate_cols(hl.agg.fold(0, lambda a: a + mt.col_idx, lambda a, b: a + b)), 45)

    def test_aggfold_scan(self):
        ht = hl.utils.range_table(15, 5)
        ht = ht.annotate(
            s=hl.scan.fold(0, lambda a: a + ht.idx, lambda a, b: a + b),
        )
        self.assertEqual(ht.s.collect(), [0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91])

        mt = hl.utils.range_matrix_table(15, 10, 5)
        mt = mt.annotate_rows(s=hl.scan.fold(0, lambda a: a + mt.row_idx, lambda a, b: a + b))
        mt = mt.annotate_rows(x=hl.scan.fold(0, lambda s: s + 1, lambda a, b: a + b))
        self.assertEqual(mt.s.collect(), [0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91])
        self.assertEqual(
            mt.rows().collect(),
            [
                hl.Struct(row_idx=0, s=0, x=0),
                hl.Struct(row_idx=1, s=0, x=1),
                hl.Struct(row_idx=2, s=1, x=2),
                hl.Struct(row_idx=3, s=3, x=3),
                hl.Struct(row_idx=4, s=6, x=4),
                hl.Struct(row_idx=5, s=10, x=5),
                hl.Struct(row_idx=6, s=15, x=6),
                hl.Struct(row_idx=7, s=21, x=7),
                hl.Struct(row_idx=8, s=28, x=8),
                hl.Struct(row_idx=9, s=36, x=9),
                hl.Struct(row_idx=10, s=45, x=10),
                hl.Struct(row_idx=11, s=55, x=11),
                hl.Struct(row_idx=12, s=66, x=12),
                hl.Struct(row_idx=13, s=78, x=13),
                hl.Struct(row_idx=14, s=91, x=14),
            ],
        )

    def test_agg_filter(self):
        t = hl.utils.range_table(10)
        tests = [
            (hl.agg.filter(t.idx > 7, hl.agg.collect(t.idx + 1).append(0)), [9, 10, 0]),
            (
                hl.agg.filter(
                    t.idx > 7, hl.agg.explode(lambda elt: hl.agg.collect(elt + 1).append(0), [t.idx, t.idx + 1])
                ),
                [9, 10, 10, 11, 0],
            ),
            (
                hl.agg.filter(
                    t.idx > 7, hl.agg.group_by(t.idx % 3, hl.array(hl.agg.collect_as_set(t.idx + 1)).append(0))
                ),
                {0: [10, 0], 2: [9, 0]},
            ),
            (hl.agg.filter(t.idx > 7, hl.agg.count()), 2),
            (hl.agg.filter(t.idx > 7, hl.agg.explode(lambda elt: hl.agg.count(), [t.idx, t.idx + 1])), 4),
            (hl.agg.filter(t.idx > 7, hl.agg.group_by(t.idx % 3, hl.agg.count())), {0: 1, 2: 1}),
        ]
        for aggregation, expected in tests:
            self.assertEqual(t.aggregate(aggregation), expected)

    def test_agg_densify(self):
        mt = hl.utils.range_matrix_table(5, 5, 3)
        mt = mt.filter_entries(mt.row_idx == mt.col_idx)
        mt = mt.annotate_entries(x=(mt.row_idx, mt.col_idx), y=hl.str(mt.row_idx) + ',' + hl.str(mt.col_idx))
        ht = mt.localize_entries('entries', 'cols')
        ht = ht.annotate(dense=hl.scan._densify(hl.len(ht.cols), ht.entries))
        ht = ht.drop('entries', 'cols')
        assert ht.collect() == [
            hl.utils.Struct(row_idx=0, dense=[None, None, None, None, None]),
            hl.utils.Struct(row_idx=1, dense=[hl.utils.Struct(x=(0, 0), y='0,0'), None, None, None, None]),
            hl.utils.Struct(
                row_idx=2,
                dense=[hl.utils.Struct(x=(0, 0), y='0,0'), hl.utils.Struct(x=(1, 1), y='1,1'), None, None, None],
            ),
            hl.utils.Struct(
                row_idx=3,
                dense=[
                    hl.utils.Struct(x=(0, 0), y='0,0'),
                    hl.utils.Struct(x=(1, 1), y='1,1'),
                    hl.utils.Struct(x=(2, 2), y='2,2'),
                    None,
                    None,
                ],
            ),
            hl.utils.Struct(
                row_idx=4,
                dense=[
                    hl.utils.Struct(x=(0, 0), y='0,0'),
                    hl.utils.Struct(x=(1, 1), y='1,1'),
                    hl.utils.Struct(x=(2, 2), y='2,2'),
                    hl.utils.Struct(x=(3, 3), y='3,3'),
                    None,
                ],
            ),
        ]

    @qobtest
    @with_flags(distributed_scan_comb_op='1')
    def test_densify_table(self):
        ht = hl.utils.range_table(100, n_partitions=33)
        ht = ht.annotate(arr=hl.range(100).map(lambda idx: hl.or_missing(idx == ht.idx, idx)))
        ht = ht.annotate(dense=hl.scan._densify(100, ht.arr))
        assert ht.all(ht.dense == hl.range(100).map(lambda idx: hl.or_missing(idx < ht.idx, idx)))

    def test_agg_array_inside_annotate_rows(self):
        n_rows = 10
        n_cols = 5
        mt = hl.utils.range_matrix_table(n_rows, n_cols)
        mt = mt.annotate_rows(x=hl.agg.array_agg(lambda i: hl.agg.sum(i), hl.range(0, mt.row_idx)))
        assert mt.aggregate_rows(hl.agg.all(mt.x == hl.range(0, mt.row_idx).map(lambda i: i * n_cols)))

    def test_agg_array_empty(self):
        ht = hl.utils.range_table(1).annotate(a=[0]).filter(False)
        assert ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.sum(x), ht.a)) == None

    def test_agg_array_non_trivial_post_op(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=[ht.idx, 2 * ht.idx])
        assert ht.aggregate(
            hl.agg.array_agg(lambda x: hl.agg.sum(x) + hl.agg.filter(x % 3 == 0, hl.agg.sum(x)), ht.a)
        ) == [63, 126]

    def test_agg_array_agg_empty_partitions(self):
        ht = hl.utils.range_table(11, 11)
        ht = ht.filter(ht.idx < 10)
        ht = ht.annotate(a=hl.range(ht.idx, ht.idx + 10))
        assert ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.sum(x), ht.a)) == [45 + 10 * x for x in range(10)]

    def test_agg_array_agg_sum_vs_sum(self):
        ht = hl.utils.range_table(25).annotate(x=hl.range(0, 10).map(lambda _: hl.rand_bool(0.5)))
        assert ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.sum(x), ht.x) == hl.agg.array_sum(ht.x))

    def test_agg_array_agg_errors(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=hl.range(0, ht.idx))
        with pytest.raises(hl.utils.FatalError):
            ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.sum(x), ht.a))

    def test_agg_array_explode(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=hl.range(0, ht.idx).map(lambda _: [ht.idx, 2 * ht.idx]))
        r = ht.aggregate(hl.agg.explode(lambda x: hl.agg.array_agg(lambda elt: hl.agg.sum(elt), x), ht.a))
        assert r == [285, 570]

        r = ht.aggregate(hl.agg.explode(lambda x: hl.agg.array_agg(lambda elt: hl.agg.count(), x), ht.a))
        assert r == [45, 45]

        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=[hl.range(0, ht.idx), hl.range(ht.idx, 2 * ht.idx)])
        r = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.explode(lambda elt: hl.agg.sum(elt), x), ht.a))
        assert r == [120, 405]

        r = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.explode(lambda elt: hl.agg.count(), x), ht.a))
        assert r == [45, 45]

    def test_agg_array_filter(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=[ht.idx])
        r = ht.aggregate(hl.agg.filter(ht.idx == 5, hl.agg.array_agg(lambda x: hl.agg.sum(x), ht.a)))
        assert r == [5]

        r2 = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.filter(x == 5, hl.agg.sum(x)), ht.a))
        assert r2 == [5]

        r3 = ht.aggregate(hl.agg.filter(ht.idx == 5, hl.agg.array_agg(lambda x: hl.agg.count(), ht.a)))
        assert r3 == [1]

        r4 = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.filter(x == 5, hl.agg.count()), ht.a))
        assert r4 == [1]

    def test_agg_array_group_by(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=[ht.idx, ht.idx + 1])
        r = ht.aggregate(hl.agg.group_by(ht.idx % 2, hl.agg.array_agg(lambda x: hl.agg.sum(x), ht.a)))
        assert r == {0: [20, 25], 1: [25, 30]}

        r2 = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.group_by(x % 2, hl.agg.sum(x)), ht.a))

        assert r2 == [{0: 20, 1: 25}, {0: 30, 1: 25}]

        r3 = ht.aggregate(hl.agg.group_by(ht.idx % 2, hl.agg.array_agg(lambda x: hl.agg.count(), ht.a)))
        assert r3 == {0: [5, 5], 1: [5, 5]}

        r4 = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.group_by(x % 2, hl.agg.count()), ht.a))

        assert r4 == [{0: 5, 1: 5}, {0: 5, 1: 5}]

    def test_agg_array_nested(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(a=[[[ht.idx]]])
        assert ht.aggregate(
            hl.agg.array_agg(
                lambda x1: hl.agg.array_agg(lambda x2: hl.agg.array_agg(lambda x3: hl.agg.sum(x3), x2), x1), ht.a
            )
        ) == [[[45]]]

    def test_agg_array_take(self):
        ht = hl.utils.range_table(10)
        r = ht.aggregate(hl.agg.array_agg(lambda x: hl.agg.take(x, 2), [ht.idx, ht.idx * 2]))
        assert r == [[0, 1], [0, 2]]

    def test_agg_array_init_op(self):
        ht = hl.utils.range_table(1).annotate_globals(n_alleles=['A', 'T']).annotate(gts=[hl.call(0, 1), hl.call(1, 1)])
        r = ht.aggregate(hl.agg.array_agg(lambda a: hl.agg.call_stats(a, ht.n_alleles), ht.gts))
        assert r == [
            hl.utils.Struct(AC=[1, 1], AF=[0.5, 0.5], AN=2, homozygote_count=[0, 0]),
            hl.utils.Struct(AC=[0, 2], AF=[0.0, 1.0], AN=2, homozygote_count=[0, 1]),
        ]

    def test_agg_collect_all_types_runs(self):
        ht = hl.utils.range_table(2)
        ht = ht.annotate(x=hl.case().when(ht.idx % 1 == 0, True).or_missing())
        ht.aggregate(
            (
                hl.agg.collect(ht.x),
                hl.agg.collect(hl.int32(ht.x)),
                hl.agg.collect(hl.int64(ht.x)),
                hl.agg.collect(hl.float32(ht.x)),
                hl.agg.collect(hl.float64(ht.x)),
                hl.agg.collect(hl.str(ht.x)),
                hl.agg.collect(hl.call(0, 0, phased=ht.x)),
                hl.agg.collect(hl.struct(foo=ht.x)),
                hl.agg.collect(hl.tuple([ht.x])),
                hl.agg.collect([ht.x]),
                hl.agg.collect({ht.x}),
                hl.agg.collect({ht.x: 1}),
                hl.agg.collect(hl.interval(0, 1, includes_start=ht.x)),
            )
        )

    def test_agg_explode(self):
        t = hl.utils.range_table(10)

        tests = [
            (
                hl.agg.explode(
                    lambda elt: hl.agg.collect(elt + 1).append(0),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                [9, 10, 10, 11, 0],
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.explode(lambda elt2: hl.agg.collect(elt2 + 1).append(0), [elt, elt + 1]),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                [9, 10, 10, 11, 10, 11, 11, 12, 0],
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.filter(elt > 8, hl.agg.collect(elt + 1).append(0)),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                [10, 10, 11, 0],
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.group_by(elt % 3, hl.agg.collect(elt + 1).append(0)),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                {0: [10, 10, 0], 1: [11, 0], 2: [9, 0]},
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.count(), hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32))
                ),
                4,
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.explode(lambda elt2: hl.agg.count(), [elt, elt + 1]),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                8,
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.filter(elt > 8, hl.agg.count()),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                3,
            ),
            (
                hl.agg.explode(
                    lambda elt: hl.agg.group_by(elt % 3, hl.agg.count()),
                    hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                ),
                {0: 2, 1: 1, 2: 1},
            ),
        ]
        for aggregation, expected in tests:
            self.assertEqual(t.aggregate(aggregation), expected)

    def test_agg_group_by_1(self):
        t = hl.utils.range_table(10)
        tests = [
            (
                hl.agg.group_by(t.idx % 2, hl.array(hl.agg.collect_as_set(t.idx + 1)).append(0)),
                {0: [1, 3, 5, 7, 9, 0], 1: [2, 4, 6, 8, 10, 0]},
            ),
            (
                hl.agg.group_by(
                    t.idx % 3, hl.agg.filter(t.idx > 7, hl.array(hl.agg.collect_as_set(t.idx + 1)).append(0))
                ),
                {0: [10, 0], 1: [0], 2: [9, 0]},
            ),
            (
                hl.agg.group_by(
                    t.idx % 3,
                    hl.agg.explode(
                        lambda elt: hl.agg.collect(elt + 1).append(0),
                        hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32)),
                    ),
                ),
                {0: [10, 11, 0], 1: [0], 2: [9, 10, 0]},
            ),
            (hl.agg.group_by(t.idx % 2, hl.agg.count()), {0: 5, 1: 5}),
            (hl.agg.group_by(t.idx % 3, hl.agg.filter(t.idx > 7, hl.agg.count())), {0: 1, 1: 0, 2: 1}),
            (
                hl.agg.group_by(
                    t.idx % 3,
                    hl.agg.explode(
                        lambda elt: hl.agg.count(), hl.if_else(t.idx > 7, [t.idx, t.idx + 1], hl.empty_array(hl.tint32))
                    ),
                ),
                {0: 2, 1: 0, 2: 2},
            ),
            (
                hl.agg.group_by(t.idx % 5, hl.agg.group_by(t.idx % 2, hl.agg.count())),
                {i: {0: 1, 1: 1} for i in range(5)},
            ),
        ]
        results = t.aggregate(hl.tuple([x[0] for x in tests]))
        for aggregate, (_, expected) in zip(results, tests):
            assert aggregate == expected

    def test_agg_group_by_2(self):
        t = hl.Table.parallelize(
            [
                {"cohort": None, "pop": "EUR", "GT": hl.Call([0, 0])},
                {"cohort": None, "pop": "ASN", "GT": hl.Call([0, 1])},
                {"cohort": None, "pop": None, "GT": hl.Call([0, 0])},
                {"cohort": "SIGMA", "pop": "AFR", "GT": hl.Call([0, 1])},
                {"cohort": "SIGMA", "pop": "EUR", "GT": hl.Call([1, 1])},
                {"cohort": "IBD", "pop": "EUR", "GT": None},
                {"cohort": "IBD", "pop": "EUR", "GT": hl.Call([0, 0])},
                {"cohort": "IBD", "pop": None, "GT": hl.Call([0, 1])},
            ],
            hl.tstruct(cohort=hl.tstr, pop=hl.tstr, GT=hl.tcall),
            n_partitions=3,
        )

        r = t.aggregate(
            hl.struct(
                count=hl.agg.group_by(t.cohort, hl.agg.group_by(t.pop, hl.agg.count_where(hl.is_defined(t.GT)))),
                inbreeding=hl.agg.group_by(t.cohort, hl.agg.inbreeding(t.GT, 0.1)),
            )
        )

        expected_count = {
            None: {'EUR': 1, 'ASN': 1, None: 1},
            'SIGMA': {'AFR': 1, 'EUR': 1},
            'IBD': {'EUR': 1, None: 1},
        }

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

    def test_agg_group_by_on_call(self):
        t = hl.utils.range_table(10)
        t = t.annotate(call=hl.call(0, 0), x=1)
        res = t.aggregate(hl.agg.group_by(t.call, hl.agg.sum(t.x)))
        self.assertEqual(res, {hl.Call([0, 0]): 10})

    def test_aggregators_with_randomness(self):
        t = hl.utils.range_table(10)
        res = t.aggregate(
            hl.agg.filter(hl.rand_bool(0.5), hl.struct(collection=hl.agg.collect(t.idx), sum=hl.agg.sum(t.idx)))
        )
        self.assertEqual(sum(res.collection), res.sum)

    def test_aggregator_scope(self):
        t = hl.utils.range_table(10)
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.agg.explode(lambda elt: hl.agg.sum(elt) + elt, [t.idx, t.idx + 1]))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.agg.filter(t.idx > 7, hl.agg.sum(t.idx) / t.idx))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.agg.group_by(t.idx % 3, hl.agg.sum(t.idx) / t.idx))
        with self.assertRaises(hl.expr.ExpressionException):
            hl.agg.counter(hl.agg.filter(t.idx > 1, t.idx))
        with self.assertRaises(hl.expr.ExpressionException):
            hl.agg.counter(hl.agg.explode(lambda elt: elt, [t.idx, t.idx + 1]))

        tests = [
            (
                hl.agg.filter(
                    t.idx > 7,
                    hl.agg.explode(lambda x: hl.agg.collect(hl.int64(x + 1)), [t.idx, t.idx + 1]).append(
                        hl.agg.group_by(t.idx % 3, hl.agg.sum(t.idx))[0]
                    ),
                ),
                [9, 10, 10, 11, 9],
            ),
            (
                hl.agg.explode(
                    lambda x: hl.agg.filter(x > 7, hl.agg.collect(x)).extend(
                        hl.agg.group_by(t.idx % 3, hl.array(hl.agg.collect_as_set(x)))[0]
                    ),
                    [t.idx, t.idx + 1],
                ),
                [8, 8, 9, 9, 10, 0, 1, 3, 4, 6, 7, 9, 10],
            ),
            (
                hl.agg.group_by(
                    t.idx % 3,
                    hl.agg.filter(t.idx > 7, hl.agg.collect(t.idx)).extend(
                        hl.agg.explode(lambda x: hl.array(hl.agg.collect_as_set(x)), [t.idx, t.idx + 34])
                    ),
                ),
                {0: [9, 0, 3, 6, 9, 34, 37, 40, 43], 1: [1, 4, 7, 35, 38, 41], 2: [8, 2, 5, 8, 36, 39, 42]},
            ),
        ]
        for aggregation, expected in tests:
            self.assertEqual(t.aggregate(aggregation), expected)

    def test_aggregator_bindings(self):
        t = hl.utils.range_table(5)
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.sum(t.idx + i), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.sum(t.idx + i), 1))
        # filter
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.filter(i == 1, hl.agg.sum(t.idx)), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.filter(t.idx == 1, hl.agg.sum(t.idx) + i), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.filter(i == 1, hl.scan.sum(t.idx)), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.filter(t.idx == 1, hl.scan.sum(t.idx) + i), 1))
        # explode
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.explode(lambda elt: hl.agg.sum(elt), [t.idx, t.idx + i]), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.explode(lambda elt: hl.agg.sum(elt) + i, [t.idx, t.idx + 1]), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.explode(lambda elt: hl.scan.sum(elt), [t.idx, t.idx + i]), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.explode(lambda elt: hl.scan.sum(elt) + i, [t.idx, t.idx + 1]), 1))
        # group_by
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.group_by(t.idx % 3 + i, hl.agg.sum(t.idx)), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.aggregate(hl.bind(lambda i: hl.agg.group_by(t.idx % 3, hl.agg.sum(t.idx) + i), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.group_by(t.idx % 3 + i, hl.scan.sum(t.idx)), 1))
        with self.assertRaises(hl.expr.ExpressionException):
            t.annotate(x=hl.bind(lambda i: hl.scan.group_by(t.idx % 3, hl.scan.sum(t.idx) + i), 1))

        # works with _ctx
        assert t.annotate(x=hl.bind(lambda i: hl.scan.sum(t.idx + i), 1, _ctx='scan')).x.collect() == [0, 1, 3, 6, 10]
        assert t.aggregate(hl.bind(lambda i: hl.agg.collect(i), t.idx * t.idx, _ctx='agg')) == [0, 1, 4, 9, 16]

    @qobtest
    def test_scan(self):
        table = hl.utils.range_table(10)

        t = table.select(
            scan_count=hl.scan.count(),
            scan_count_where=hl.scan.count_where(table.idx % 2 == 0),
            scan_count_where2=hl.scan.filter(table.idx % 2 == 0, hl.scan.count()),
            arr_sum=hl.scan.array_sum([1, 2, hl.missing(tint32)]),
            bind_agg=hl.scan.count_where(hl.bind(lambda x: x % 2 == 0, table.idx)),
            mean=hl.scan.mean(table.idx),
            foo=hl.min(3, hl.scan.sum(table.idx)),
            fraction_odd=hl.scan.fraction(table.idx % 2 == 0),
        )
        rows = t.collect()
        r = hl.Struct(**{n: [i[n] for i in rows] for n in t.row.keys()})

        self.assertEqual(r.scan_count, [i for i in range(10)])
        self.assertEqual(r.scan_count_where, [(i + 1) // 2 for i in range(10)])
        self.assertEqual(r.scan_count_where2, [(i + 1) // 2 for i in range(10)])
        self.assertEqual(r.arr_sum, [None] + [[i * 1, i * 2, 0] for i in range(1, 10)])
        self.assertEqual(r.bind_agg, [(i + 1) // 2 for i in range(10)])
        self.assertEqual(r.foo, [min(sum(range(i)), 3) for i in range(10)])
        for (x, y) in zip(r.fraction_odd, [None] + [((i + 1) // 2) / i for i in range(1, 10)]):
            self.assertAlmostEqual(x, y)

        table = hl.utils.range_table(10)
        r = table.aggregate(hl.struct(x=hl.agg.count()))

        self.assertEqual(r.x, 10)

    def test_scan_filter(self):
        t = hl.utils.range_table(5)
        tests = [
            (
                hl.scan.filter((t.idx % 2) == 0, hl.scan.collect(t.idx).append(t.idx)),
                [[0], [0, 1], [0, 2], [0, 2, 3], [0, 2, 4]],
            ),
            (
                hl.scan.filter(
                    (t.idx % 2) == 0,
                    hl.scan.explode(lambda elt: hl.scan.collect(elt).append(t.idx), [t.idx, t.idx + 1]),
                ),
                [[0], [0, 1, 1], [0, 1, 2], [0, 1, 2, 3, 3], [0, 1, 2, 3, 4]],
            ),
            (
                hl.scan.filter((t.idx % 2) == 0, hl.scan.group_by(t.idx % 3, hl.scan.collect(t.idx).append(t.idx))),
                [{}, {0: [0, 1]}, {0: [0, 2]}, {0: [0, 3], 2: [2, 3]}, {0: [0, 4], 2: [2, 4]}],
            ),
        ]

        for aggregation, expected in tests:
            self.assertEqual(aggregation.collect(), expected)

    def test_scan_explode(self):
        t = hl.utils.range_table(5)
        tests = [
            (
                hl.scan.explode(lambda elt: hl.scan.collect(elt).append(t.idx), [t.idx, t.idx + 1]),
                [[0], [0, 1, 1], [0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 4, 4]],
            ),
            (
                hl.scan.explode(
                    lambda elt: hl.scan.explode(lambda elt2: hl.scan.collect(elt).append(t.idx), [elt]),
                    [t.idx, t.idx + 1],
                ),
                [[0], [0, 1, 1], [0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 4, 4]],
            ),
            (
                hl.scan.explode(
                    lambda elt: hl.scan.filter((elt % 2) == 0, hl.scan.collect(elt).append(t.idx)), [t.idx, t.idx + 1]
                ),
                [[0], [0, 1], [0, 2, 2], [0, 2, 2, 3], [0, 2, 2, 4, 4]],
            ),
            (
                hl.scan.explode(
                    lambda elt: hl.scan.group_by(elt % 3, hl.scan.collect(elt).append(t.idx)), [t.idx, t.idx + 1]
                ),
                [
                    {},
                    {0: [0, 1], 1: [1, 1]},
                    {0: [0, 2], 1: [1, 1, 2], 2: [2, 2]},
                    {0: [0, 3, 3], 1: [1, 1, 3], 2: [2, 2, 3]},
                    {0: [0, 3, 3, 4], 1: [1, 1, 4, 4], 2: [2, 2, 4]},
                ],
            ),
        ]

        for aggregation, expected in tests:
            self.assertEqual(aggregation.collect(), expected)

    def test_scan_group_by(self):
        t = hl.utils.range_table(5)
        tests = [
            (
                hl.scan.group_by(t.idx % 3, hl.scan.collect(t.idx).append(t.idx)),
                [
                    {},
                    {0: [0, 1]},
                    {0: [0, 2], 1: [1, 2]},
                    {0: [0, 3], 1: [1, 3], 2: [2, 3]},
                    {0: [0, 3, 4], 1: [1, 4], 2: [2, 4]},
                ],
            ),
            (
                hl.scan.group_by(t.idx % 3, hl.scan.filter((t.idx % 2) == 0, hl.scan.collect(t.idx).append(t.idx))),
                [{}, {0: [0, 1]}, {0: [0, 2], 1: [2]}, {0: [0, 3], 1: [3], 2: [2, 3]}, {0: [0, 4], 1: [4], 2: [2, 4]}],
            ),
            (
                hl.scan.group_by(
                    t.idx % 3, hl.scan.explode(lambda elt: hl.scan.collect(elt).append(t.idx), [t.idx, t.idx + 1])
                ),
                [
                    {},
                    {0: [0, 1, 1]},
                    {0: [0, 1, 2], 1: [1, 2, 2]},
                    {0: [0, 1, 3], 1: [1, 2, 3], 2: [2, 3, 3]},
                    {0: [0, 1, 3, 4, 4], 1: [1, 2, 4], 2: [2, 3, 4]},
                ],
            ),
        ]

        for aggregation, expected in tests:
            self.assertEqual(aggregation.collect(), expected)

    def test_scan_array_agg(self):
        ht = hl.utils.range_table(5)
        ht = ht.annotate(a=hl.range(0, 5).map(lambda x: ht.idx))
        ht = ht.annotate(a2=hl.scan.array_agg(lambda _: hl.agg.count(), ht.a))
        assert ht.all((ht.idx == 0) | (ht.a == ht.a2))

    def test_aggregators_max_min(self):
        table = hl.utils.range_table(10)
        # FIXME: add boolean when function registry is removed
        for (f, typ) in [
            (lambda x: hl.int32(x), tint32),
            (lambda x: hl.int64(x), tint64),
            (lambda x: hl.float32(x), tfloat32),
            (lambda x: hl.float64(x), tfloat64),
        ]:
            t = table.annotate(x=-1 * f(table.idx) - 5, y=hl.missing(typ))
            r = t.aggregate(
                hl.struct(
                    max=hl.agg.max(t.x), max_empty=hl.agg.max(t.y), min=hl.agg.min(t.x), min_empty=hl.agg.min(t.y)
                )
            )
            self.assertTrue(r.max == -5 and r.max_empty is None and r.min == -14 and r.min_empty is None)

    def test_aggregators_sum_product(self):
        table = hl.utils.range_table(5)
        for (f, typ) in [
            (lambda x: hl.int32(x), tint32),
            (lambda x: hl.int64(x), tint64),
            (lambda x: hl.float32(x), tfloat32),
            (lambda x: hl.float64(x), tfloat64),
        ]:
            t = table.annotate(x=-1 * f(table.idx) - 1, y=f(table.idx), z=hl.missing(typ))
            r = t.aggregate(
                hl.struct(
                    sum_x=hl.agg.sum(t.x),
                    sum_y=hl.agg.sum(t.y),
                    sum_empty=hl.agg.sum(t.z),
                    prod_x=hl.agg.product(t.x),
                    prod_y=hl.agg.product(t.y),
                    prod_empty=hl.agg.product(t.z),
                )
            )
            self.assertTrue(
                r.sum_x == -15
                and r.sum_y == 10
                and r.sum_empty == 0
                and r.prod_x == -120
                and r.prod_y == 0
                and r.prod_empty == 1
            )

    def test_aggregators_hist(self):
        table = hl.utils.range_table(11)
        r = table.aggregate(hl.agg.hist(table.idx - 1, 0, 8, 4))
        self.assertTrue(
            r.bin_edges == [0, 2, 4, 6, 8] and r.bin_freq == [2, 2, 2, 3] and r.n_smaller == 1 and r.n_larger == 1
        )

    def test_aggregators_hist_neg0(self):
        table = hl.utils.range_table(32)
        table = table.annotate(d=hl.if_else(table.idx == 11, -0.0, table.idx / 3))
        r = table.aggregate(hl.agg.hist(table.d, 0, 10, 5))
        self.assertEqual(r.bin_edges, [0, 2, 4, 6, 8, 10])
        self.assertEqual(r.bin_freq, [7, 5, 6, 6, 7])
        self.assertEqual(r.n_smaller, 0)
        self.assertEqual(r.n_larger, 1)

    def test_aggregators_hist_nan(self):
        ht = hl.utils.range_table(3).annotate(x=hl.float('nan'))
        r = ht.aggregate(hl.agg.hist(ht.x, 0, 10, 2))
        assert r.bin_freq == [0, 0]
        assert r.n_smaller == 0
        assert r.n_larger == 0

    def test_aggregator_cse(self):
        ht = hl.utils.range_table(10)
        x = hl.agg.count()
        self.assertEqual(ht.aggregate((x, hl.agg.filter(ht.idx % 2 == 0, x))), (10, 5))

        mt = hl.utils.range_matrix_table(10, 10)
        x = hl.int64(5)
        rows = mt.annotate_rows(agg=hl.agg.sum(x + x), scan=hl.scan.sum(x + x), val=x + x).rows()
        expected = hl.utils.range_table(10)
        expected = expected.key_by(row_idx=expected.idx)
        expected = expected.select(agg=hl.int64(100), scan=hl.int64(expected.row_idx * 10), val=hl.int64(10))
        self.assertTrue(rows._same(expected))

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
        t = hl.Table.parallelize(
            [
                {"y": None, "x": 1.0},
                {"y": 0.0, "x": None},
                {"y": None, "x": None},
                {"y": 0.22848042, "x": 0.2575928},
                {"y": 0.09159706, "x": -0.3445442},
                {"y": -0.43881935, "x": 1.6590146},
                {"y": -0.99106171, "x": -1.1688806},
                {"y": 2.12823289, "x": 0.5587043},
            ],
            hl.tstruct(y=hl.tfloat64, x=hl.tfloat64),
            n_partitions=3,
        )
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

        # swapping the intercept and t.x
        r = t.aggregate(hl.struct(linreg=hl.agg.linreg(t.y, [t.x, 1]))).linreg
        self.assertAlmostEqual(r.beta[1], 0.14069227)
        self.assertAlmostEqual(r.beta[0], 0.32744807)
        self.assertAlmostEqual(r.standard_error[1], 0.59410817)
        self.assertAlmostEqual(r.standard_error[0], 0.61833778)
        self.assertAlmostEqual(r.t_stat[1], 0.23681254)
        self.assertAlmostEqual(r.t_stat[0], 0.52956181)
        self.assertAlmostEqual(r.p_value[1], 0.82805147)
        self.assertAlmostEqual(r.p_value[0], 0.63310173)

        # weighted OLS
        t = t.add_index()
        r = t.aggregate(hl.struct(linreg=hl.agg.linreg(t.y, [1, t.x], weight=t.idx))).linreg
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

    def test_linreg_no_data(self):
        ht = hl.utils.range_table(1).filter(False)
        r = ht.aggregate(hl.agg.linreg(ht.idx, 0))
        for k, v in r.items():
            if k == 'n':
                assert v == 0
            else:
                assert v is None, k

    def test_aggregator_downsample(self):
        xs = [2, 6, 4, 9, 1, 8, 5, 10, 3, 7]
        ys = [2, 6, 4, 9, 1, 8, 5, 10, 3, 7]
        label1 = ["2", "6", "4", "9", "1", "8", "5", "10", "3", "7"]
        label2 = ["two", "six", "four", "nine", "one", "eight", "five", "ten", "three", "seven"]
        table = hl.Table.parallelize(
            [hl.struct(x=x, y=y, label1=label1, label2=label2) for x, y, label1, label2 in zip(xs, ys, label1, label2)]
        )
        r = table.aggregate(
            hl.agg.downsample(table.x, table.y, label=hl.array([table.label1, table.label2]), n_divisions=10)
        )
        xs = [x for (x, y, l) in r]
        ys = [y for (x, y, l) in r]
        label = [tuple(l) for (x, y, l) in r]
        expected = set(
            [
                (1.0, 1.0, ('1', 'one')),
                (2.0, 2.0, ('2', 'two')),
                (3.0, 3.0, ('3', 'three')),
                (4.0, 4.0, ('4', 'four')),
                (5.0, 5.0, ('5', 'five')),
                (6.0, 6.0, ('6', 'six')),
                (7.0, 7.0, ('7', 'seven')),
                (8.0, 8.0, ('8', 'eight')),
                (9.0, 9.0, ('9', 'nine')),
                (10.0, 10.0, ('10', 'ten')),
            ]
        )
        for point in zip(xs, ys, label):
            self.assertTrue(point in expected)

    def test_downsample_aggregator_on_empty_table(self):
        ht = hl.utils.range_table(1)
        ht = ht.annotate(y=ht.idx).filter(False)
        r = ht.aggregate(hl.agg.downsample(ht.idx, ht.y, n_divisions=10))
        self.assertTrue(len(r) == 0)

    def test_downsample_in_array_agg(self):
        mt = hl.utils.range_matrix_table(50, 50)
        mt = mt.annotate_rows(y=hl.rand_unif(0, 1))
        mt = mt.annotate_cols(binned=hl.agg.downsample(mt.row_idx, mt.y, label=hl.str(mt.y), n_divisions=4))
        mt.cols()._force_count()

    def test_aggregator_info_score(self):
        gen_file = resource('infoScoreTest.gen')
        sample_file = resource('infoScoreTest.sample')
        truth_result_file = resource('infoScoreTest.result')

        mt = hl.import_gen(gen_file, sample_file=sample_file)
        mt = mt.annotate_rows(info_score=hl.agg.info_score(mt.GP))

        truth = hl.import_table(truth_result_file, impute=True, delimiter=' ', no_header=True, missing='None')
        truth = truth.drop('f1', 'f2').rename({'f0': 'variant', 'f3': 'score', 'f4': 'n_included'})
        truth = truth.transmute(**hl.parse_variant(truth.variant)).key_by('locus', 'alleles')

        computed = mt.rows()

        joined = truth[computed.key]
        computed = computed.select(
            score=computed.info_score.score,
            score_truth=joined.score,
            n_included=computed.info_score.n_included,
            n_included_truth=joined.n_included,
        )
        violations = computed.filter(
            (computed.n_included != computed.n_included_truth) | (hl.abs(computed.score - computed.score_truth) > 1e-3)
        )
        if not violations.count() == 0:
            violations.show()
            self.fail("disagreement between computed info score and truth")

    def test_aggregator_info_score_works_with_bgen_import(self):
        bgenmt = hl.import_bgen(resource('random.bgen'), ['GT', 'GP'], resource('random.sample'))
        result = bgenmt.annotate_rows(info=hl.agg.info_score(bgenmt.GP)).rows().take(1)
        result = result[0].info
        self.assertAlmostEqual(result.score, -0.235041090, places=3)
        self.assertEqual(result.n_included, 8)

    def test_aggregator_group_by_sorts_result(self):
        t = hl.Table.parallelize(
            [  # the `s` key is stored before the `m` in java.util.HashMap
                {"group": "m", "x": 1},
                {"group": "s", "x": 2},
                {"group": "s", "x": 3},
                {"group": "m", "x": 4},
                {"group": "m", "x": 5},
            ],
            hl.tstruct(group=hl.tstr, x=hl.tint32),
            n_partitions=1,
        )

        grouped_expr = t.aggregate(hl.array(hl.agg.group_by(t.group, hl.agg.sum(t.x))))
        self.assertEqual(grouped_expr, hl.eval(hl.sorted(grouped_expr)))

    def test_agg_corr(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(
            tests=hl.range(0, 10).map(
                lambda i: hl.struct(
                    x=hl.if_else(hl.rand_bool(0.1), hl.missing(hl.tfloat64), hl.rand_unif(-10, 10)),
                    y=hl.if_else(hl.rand_bool(0.1), hl.missing(hl.tfloat64), hl.rand_unif(-10, 10)),
                )
            )
        )

        results = ht.aggregate(
            hl.agg.array_agg(lambda test: (hl.agg.corr(test.x, test.y), hl.agg.collect((test.x, test.y))), ht.tests)
        )

        for corr, xy in results:
            filtered = [(x, y) for x, y in xy if x is not None and y is not None]
            scipy_corr, _ = pearsonr([x for x, _ in filtered], [y for _, y in filtered])
            self.assertAlmostEqual(corr, scipy_corr)

    def test_joins_inside_aggregators(self):
        table = hl.utils.range_table(10)
        table2 = hl.utils.range_table(10)
        self.assertEqual(table.aggregate(hl.agg.count_where(hl.is_defined(table2[table.idx]))), 10)

    def test_switch(self):
        x = hl.literal('1')
        na = hl.missing(tint32)

        expr1 = hl.switch(x).when('123', 5).when('1', 6).when('0', 2).or_missing()
        self.assertEqual(hl.eval(expr1), 6)

        expr2 = hl.switch(x).when('123', 5).when('0', 2).or_missing()
        self.assertEqual(hl.eval(expr2), None)

        expr3 = hl.switch(x).when('123', 5).when('0', 2).default(100)
        self.assertEqual(hl.eval(expr3), 100)

        expr4 = hl.switch(na).when(5, 0).when(6, 1).when(0, 2).when(hl.missing(tint32), 3).default(4)  # NA != NA
        self.assertEqual(hl.eval(expr4), None)

        expr5 = (
            hl.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(hl.missing(tint32), 3)  # NA != NA
            .when_missing(-1)
            .default(4)
        )
        self.assertEqual(hl.eval(expr5), -1)

        with pytest.raises(hl.utils.java.HailUserError) as exc:
            hl.eval(hl.switch(x).when('0', 0).or_error("foo"))
        assert '.or_error("foo")' in str(exc.value)

    def test_case(self):
        def make_case(x):
            x = hl.literal(x)
            return hl.case().when(x == 6, 'A').when(x % 3 == 0, 'B').when(x == 5, 'C').when(x < 2, 'D').or_missing()

        self.assertEqual(hl.eval(make_case(6)), 'A')
        self.assertEqual(hl.eval(make_case(12)), 'B')
        self.assertEqual(hl.eval(make_case(5)), 'C')
        self.assertEqual(hl.eval(make_case(-1)), 'D')
        self.assertEqual(hl.eval(make_case(2)), None)

        self.assertEqual(hl.eval(hl.case().when(hl.missing(hl.tbool), 1).default(2)), None)
        self.assertEqual(hl.eval(hl.case(missing_false=True).when(hl.missing(hl.tbool), 1).default(2)), 2)

        error_case = hl.case().when(False, 1).or_error("foo")
        with pytest.raises(hl.utils.java.HailUserError) as exc:
            hl.eval(error_case)
        assert '.or_error("foo")' in str(exc.value)

    def test_struct_ops(self):
        s = hl.struct(f1=1, f2=2, f3=3)

        def assert_typed(expr, result, dtype):
            self.assertEqual(expr.dtype, dtype)
            r, t = hl.eval_typed(expr)
            self.assertEqual(t, dtype)
            self.assertEqual(result, r)

        assert_typed(s.drop('f3'), hl.Struct(f1=1, f2=2), tstruct(f1=tint32, f2=tint32))

        assert_typed(s.drop('f1'), hl.Struct(f2=2, f3=3), tstruct(f2=tint32, f3=tint32))

        assert_typed(s.drop(), hl.Struct(f1=1, f2=2, f3=3), tstruct(f1=tint32, f2=tint32, f3=tint32))

        assert_typed(s.select('f1', 'f2'), hl.Struct(f1=1, f2=2), tstruct(f1=tint32, f2=tint32))

        assert_typed(
            s.select('f2', 'f1', f4=5, f5=6),
            hl.Struct(f2=2, f1=1, f4=5, f5=6),
            tstruct(f2=tint32, f1=tint32, f4=tint32, f5=tint32),
        )

        assert_typed(s.select(), hl.Struct(), tstruct())

        assert_typed(
            s.annotate(f1=5, f2=10, f4=15),
            hl.Struct(f1=5, f2=10, f3=3, f4=15),
            tstruct(f1=tint32, f2=tint32, f3=tint32, f4=tint32),
        )

        assert_typed(s.annotate(f1=5), hl.Struct(f1=5, f2=2, f3=3), tstruct(f1=tint32, f2=tint32, f3=tint32))

        assert_typed(s.annotate(), hl.Struct(f1=1, f2=2, f3=3), tstruct(f1=tint32, f2=tint32, f3=tint32))

    def test_shadowed_struct_fields(self):
        from typing import Callable

        s = hl.struct(foo=1, values=2, collect=3, _ir=4)
        assert 'foo' not in s._warn_on_shadowed_name
        assert isinstance(s.foo, hl.Expression)
        assert 'values' in s._warn_on_shadowed_name
        assert isinstance(s.values, Callable)
        assert 'values' not in s._warn_on_shadowed_name
        assert 'collect' in s._warn_on_shadowed_name
        assert isinstance(s.collect, Callable)
        assert 'collect' not in s._warn_on_shadowed_name
        assert '_ir' in s._warn_on_shadowed_name
        assert isinstance(s._ir, ir.IR)
        assert '_ir' not in s._warn_on_shadowed_name

        s = hl.StructExpression._from_fields(
            {'foo': hl.int(1), 'values': hl.int(2), 'collect': hl.int(3), '_ir': hl.int(4)}
        )
        assert 'foo' not in s._warn_on_shadowed_name
        assert isinstance(s.foo, hl.Expression)
        assert 'values' in s._warn_on_shadowed_name
        assert isinstance(s.values, Callable)
        assert 'values' not in s._warn_on_shadowed_name
        assert 'collect' in s._warn_on_shadowed_name
        assert isinstance(s.collect, Callable)
        assert 'collect' not in s._warn_on_shadowed_name
        assert '_ir' in s._warn_on_shadowed_name
        assert isinstance(s._ir, ir.IR)
        assert '_ir' not in s._warn_on_shadowed_name

    def test_iter(self):
        a = hl.literal([1, 2, 3])
        self.assertRaises(hl.expr.ExpressionException, lambda: hl.eval(list(a)))

    def test_dict_get(self):
        d = hl.dict({'a': 1, 'b': 2, 'missing_value': hl.missing(hl.tint32), hl.missing(hl.tstr): 5})
        self.assertEqual(hl.eval(d.get('a')), 1)
        self.assertEqual(hl.eval(d['a']), 1)
        self.assertEqual(hl.eval(d.get('b')), 2)
        self.assertEqual(hl.eval(d['b']), 2)
        self.assertEqual(hl.eval(d.get('c')), None)
        self.assertEqual(hl.eval(d.get(hl.missing(hl.tstr))), 5)
        self.assertEqual(hl.eval(d[hl.missing(hl.tstr)]), 5)

        self.assertEqual(hl.eval(d.get('c', 5)), 5)
        self.assertEqual(hl.eval(d.get('a', 5)), 1)

        self.assertEqual(hl.eval(d.get('missing_values')), None)
        self.assertEqual(hl.eval(d.get('missing_values', hl.missing(hl.tint32))), None)
        self.assertEqual(hl.eval(d.get('missing_values', 5)), 5)

    def test_functions_any_and_all(self):
        x1 = hl.literal([], dtype='array<bool>')
        x2 = hl.literal([True], dtype='array<bool>')
        x3 = hl.literal([False], dtype='array<bool>')
        x4 = hl.literal([None], dtype='array<bool>')
        x5 = hl.literal([True, False], dtype='array<bool>')
        x6 = hl.literal([True, None], dtype='array<bool>')
        x7 = hl.literal([False, None], dtype='array<bool>')
        x8 = hl.literal([True, False, None], dtype='array<bool>')

        assert hl.eval(
            (
                (x1.any(lambda x: x), x1.all(lambda x: x)),
                (x2.any(lambda x: x), x2.all(lambda x: x)),
                (x3.any(lambda x: x), x3.all(lambda x: x)),
                (x4.any(lambda x: x), x4.all(lambda x: x)),
                (x5.any(lambda x: x), x5.all(lambda x: x)),
                (x6.any(lambda x: x), x6.all(lambda x: x)),
                (x7.any(lambda x: x), x7.all(lambda x: x)),
                (x8.any(lambda x: x), x8.all(lambda x: x)),
            )
        ) == (
            (False, True),
            (True, True),
            (False, False),
            (None, None),
            (True, False),
            (True, None),
            (None, False),
            (True, False),
        )

    def test_aggregator_any_and_all(self):
        df = hl.utils.range_table(10)
        df = df.annotate(
            all_true=True,
            all_false=False,
            true_or_missing=hl.if_else(df.idx % 2 == 0, True, hl.missing(tbool)),
            false_or_missing=hl.if_else(df.idx % 2 == 0, False, hl.missing(tbool)),
            all_missing=hl.missing(tbool),
            mixed_true_false=hl.if_else(df.idx % 2 == 0, True, False),
            mixed_all=hl.switch(df.idx % 3).when(0, True).when(1, False).or_missing(),
        ).cache()

        self.assertEqual(df.aggregate(hl.agg.any(df.all_true)), True)
        self.assertEqual(df.aggregate(hl.agg.all(df.all_true)), True)
        self.assertEqual(df.aggregate(hl.agg.any(df.all_false)), False)
        self.assertEqual(df.aggregate(hl.agg.any(df.all_false)), False)
        self.assertEqual(df.aggregate(hl.agg.any(df.true_or_missing)), True)
        self.assertEqual(df.aggregate(hl.agg.all(df.true_or_missing)), True)
        self.assertEqual(df.aggregate(hl.agg.any(df.false_or_missing)), False)
        self.assertEqual(df.aggregate(hl.agg.all(df.false_or_missing)), False)
        self.assertEqual(df.aggregate(hl.agg.any(df.all_missing)), False)
        self.assertEqual(df.aggregate(hl.agg.all(df.all_missing)), True)
        self.assertEqual(df.aggregate(hl.agg.any(df.mixed_true_false)), True)
        self.assertEqual(df.aggregate(hl.agg.all(df.mixed_true_false)), False)
        self.assertEqual(df.aggregate(hl.agg.any(df.mixed_all)), True)
        self.assertEqual(df.aggregate(hl.agg.all(df.mixed_all)), False)

        self.assertEqual(df.aggregate(hl.agg.filter(False, hl.agg.any(df.all_true))), False)
        self.assertEqual(df.aggregate(hl.agg.filter(False, hl.agg.all(df.all_true))), True)

    def test_agg_prev_nonnull(self):
        t = hl.utils.range_table(17, n_partitions=8)
        t = t.annotate(prev=hl.scan._prev_nonnull(hl.or_missing((t.idx % 3) != 0, t.row)))
        self.assertTrue(
            t.all(
                hl._values_similar(
                    t.prev.idx,
                    hl.case()
                    .when(t.idx < 2, hl.missing(hl.tint32))
                    .when(((t.idx - 1) % 3) == 0, t.idx - 2)
                    .default(t.idx - 1),
                )
            )
        )

    def test_agg_table_take(self):
        ht = hl.utils.range_table(10).annotate(x='a')
        self.assertEqual(ht.aggregate(agg.take(ht.x, 2)), ['a', 'a'])

    def test_agg_take_by(self):
        ht = hl.utils.range_table(10, 3)
        data1 = hl.literal([str(i) for i in range(10)])
        data2 = hl.literal([i**2 for i in range(10)])
        ht = ht.annotate(d1=data1[ht.idx], d2=data2[ht.idx])

        tb1, tb2, tb3, tb4 = ht.aggregate(
            (
                hl.agg.take(ht.d1, 5, ordering=-ht.idx),
                hl.agg.take(ht.d2, 5, ordering=-ht.idx),
                hl.agg.take(ht.idx, 7, ordering=ht.idx // 5),  # stable sort
                hl.agg.array_agg(
                    lambda elt: hl.agg.take(hl.str(elt) + "_" + hl.str(ht.idx), 4, ordering=ht.idx), hl.range(0, 2)
                ),
            )
        )

        assert tb1 == ['9', '8', '7', '6', '5']
        assert tb2 == [81, 64, 49, 36, 25]
        assert tb3 == [0, 1, 2, 3, 4, 5, 6]
        assert tb4 == [['0_0', '0_1', '0_2', '0_3'], ['1_0', '1_1', '1_2', '1_3']]

    def test_agg_minmax(self):
        nan = float('nan')
        na = hl.missing(hl.tfloat32)
        size = 200
        for aggfunc in (agg.min, agg.max):
            array_with_nan = hl.array([0.0 if i == 1 else nan for i in range(size)])
            array_with_na = hl.array([0.0 if i == 1 else na for i in range(size)])
            t = hl.utils.range_table(size)
            self.assertEqual(t.aggregate(aggfunc(array_with_nan[t.idx])), 0.0)
            self.assertEqual(t.aggregate(aggfunc(array_with_na[t.idx])), 0.0)

    def test_str_ops(self):
        s = hl.literal('abcABC123')
        s_whitespace = hl.literal(' \t 1 2 3 \t\n')
        _test_many_equal(
            [
                (hl.int32(hl.literal('123')), 123),
                (hl.int64(hl.literal("123123123123")), 123123123123),
                (hl.float32(hl.literal('1.5')), 1.5),
                (hl.float64(hl.literal('1.5')), 1.5),
                (s.lower(), 'abcabc123'),
                (s.upper(), 'ABCABC123'),
                (s_whitespace.strip(), '1 2 3'),
                (s.contains('ABC'), True),
                (~s.contains('ABC'), False),
                (s.contains('a'), True),
                (s.contains('C123'), True),
                (s.contains(''), True),
                (s.contains('C1234'), False),
                (s.contains(' '), False),
                (s_whitespace.startswith(' \t'), True),
                (s_whitespace.endswith('\t\n'), True),
                (s_whitespace.startswith('a'), False),
                (s_whitespace.endswith('a'), False),
            ]
        )

    def test_str_parsing(self):
        int_parsers = (hl.int32, hl.int64, hl.parse_int32, hl.parse_int64)
        float_parsers = (hl.float, hl.float32, hl.float64, hl.parse_float32, hl.parse_float64)
        infinity_strings = ('inf', 'Inf', 'iNf', 'InF', 'infinity', 'InfiNitY', 'INFINITY')
        _test_many_equal(
            [
                *[(hl.bool(x), True) for x in ('true', 'True', 'TRUE')],
                *[(hl.bool(x), False) for x in ('false', 'False', 'FALSE')],
                *[
                    (hl.is_nan(f(sgn + x)), True)
                    for x in ('nan', 'Nan', 'naN', 'NaN')
                    for sgn in ('', '+', '-')
                    for f in float_parsers
                ],
                *[
                    (hl.is_infinite(f(sgn + x)), True)
                    for x in infinity_strings
                    for sgn in ('', '+', '-')
                    for f in float_parsers
                ],
                *[(f('-' + x) < 0.0, True) for x in infinity_strings for f in float_parsers],
                *[
                    (hl.tuple([int_parser(hl.literal(x)), float_parser(hl.literal(x))]), (int(x), float(x)))
                    for int_parser in int_parsers
                    for float_parser in float_parsers
                    for x in ('0', '1', '-5', '12382421')
                ],
                *[
                    (hl.tuple([float_parser(hl.literal(x)), flexible_int_parser(hl.literal(x))]), (float(x), None))
                    for float_parser in float_parsers
                    for flexible_int_parser in (hl.parse_int32, hl.parse_int64)
                    for x in ('-1.5', '0.0', '2.5')
                ],
                *[
                    (flexible_numeric_parser(hl.literal(x)), None)
                    for flexible_numeric_parser in (hl.parse_float32, hl.parse_float64, hl.parse_int32, hl.parse_int64)
                    for x in ('abc', '1abc', '')
                ],
            ]
        )

    def test_str_missingness(self):
        self.assertEqual(hl.eval(hl.str(1)), '1')
        self.assertEqual(hl.eval(hl.str(hl.missing('int32'))), None)

    def test_missing_with_field_starting_with_number(self):
        assert hl.eval(hl.missing(hl.tstruct(**{"1kg": hl.tint32}))) is None

    def check_expr(self, expr, expected, expected_type):
        self.assertEqual(expected_type, expr.dtype)
        self.assertEqual((expected, expected_type), hl.eval_typed(expr))

    def test_division(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int64_4 = hl.int64(4)
        int64_4s = int32_4s.map(lambda x: hl.int64(x))
        float32_4 = hl.float32(4)
        float32_4s = int32_4s.map(lambda x: hl.float32(x))
        float64_4 = hl.float64(4)
        float64_4s = int32_4s.map(lambda x: hl.float64(x))

        expected = [0.5, 1.0, 2.0, 4.0, None]
        expected_inv = [2.0, 1.0, 0.5, 0.25, None]

        _test_many_equal_typed(
            [
                (a_int32 / 4, expected, tarray(tfloat64)),
                (a_int64 / 4, expected, tarray(tfloat64)),
                (a_float32 / 4, expected, tarray(tfloat32)),
                (a_float64 / 4, expected, tarray(tfloat64)),
                (int32_4s / a_int32, expected_inv, tarray(tfloat64)),
                (int32_4s / a_int64, expected_inv, tarray(tfloat64)),
                (int32_4s / a_float32, expected_inv, tarray(tfloat32)),
                (int32_4s / a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 / int32_4s, expected, tarray(tfloat64)),
                (a_int64 / int32_4s, expected, tarray(tfloat64)),
                (a_float32 / int32_4s, expected, tarray(tfloat32)),
                (a_float64 / int32_4s, expected, tarray(tfloat64)),
                (a_int32 / int64_4, expected, tarray(tfloat64)),
                (a_int64 / int64_4, expected, tarray(tfloat64)),
                (a_float32 / int64_4, expected, tarray(tfloat32)),
                (a_float64 / int64_4, expected, tarray(tfloat64)),
                (int64_4 / a_int32, expected_inv, tarray(tfloat64)),
                (int64_4 / a_int64, expected_inv, tarray(tfloat64)),
                (int64_4 / a_float32, expected_inv, tarray(tfloat32)),
                (int64_4 / a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 / int64_4s, expected, tarray(tfloat64)),
                (a_int64 / int64_4s, expected, tarray(tfloat64)),
                (a_float32 / int64_4s, expected, tarray(tfloat32)),
                (a_float64 / int64_4s, expected, tarray(tfloat64)),
                (a_int32 / float32_4, expected, tarray(tfloat32)),
                (a_int64 / float32_4, expected, tarray(tfloat32)),
                (a_float32 / float32_4, expected, tarray(tfloat32)),
                (a_float64 / float32_4, expected, tarray(tfloat64)),
                (float32_4 / a_int32, expected_inv, tarray(tfloat32)),
                (float32_4 / a_int64, expected_inv, tarray(tfloat32)),
                (float32_4 / a_float32, expected_inv, tarray(tfloat32)),
                (float32_4 / a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 / float32_4s, expected, tarray(tfloat32)),
                (a_int64 / float32_4s, expected, tarray(tfloat32)),
                (a_float32 / float32_4s, expected, tarray(tfloat32)),
                (a_float64 / float32_4s, expected, tarray(tfloat64)),
                (a_int32 / float64_4, expected, tarray(tfloat64)),
                (a_int64 / float64_4, expected, tarray(tfloat64)),
                (a_float32 / float64_4, expected, tarray(tfloat64)),
                (a_float64 / float64_4, expected, tarray(tfloat64)),
                (float64_4 / a_int32, expected_inv, tarray(tfloat64)),
                (float64_4 / a_int64, expected_inv, tarray(tfloat64)),
                (float64_4 / a_float32, expected_inv, tarray(tfloat64)),
                (float64_4 / a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 / float64_4s, expected, tarray(tfloat64)),
                (a_int64 / float64_4s, expected, tarray(tfloat64)),
                (a_float32 / float64_4s, expected, tarray(tfloat64)),
                (a_float64 / float64_4s, expected, tarray(tfloat64)),
            ]
        )

    def test_floor_division(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.missing(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [0, 1, 2, 5, None]
        expected_inv = [1, 0, 0, 0, None]

        _test_many_equal_typed(
            [
                (a_int32 // 3, expected, tarray(tint32)),
                (a_int64 // 3, expected, tarray(tint64)),
                (a_float32 // 3, expected, tarray(tfloat32)),
                (a_float64 // 3, expected, tarray(tfloat64)),
                (3 // a_int32, expected_inv, tarray(tint32)),
                (3 // a_int64, expected_inv, tarray(tint64)),
                (3 // a_float32, expected_inv, tarray(tfloat32)),
                (3 // a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 // int32_3s, expected, tarray(tint32)),
                (a_int64 // int32_3s, expected, tarray(tint64)),
                (a_float32 // int32_3s, expected, tarray(tfloat32)),
                (a_float64 // int32_3s, expected, tarray(tfloat64)),
                (a_int32 // int64_3, expected, tarray(tint64)),
                (a_int64 // int64_3, expected, tarray(tint64)),
                (a_float32 // int64_3, expected, tarray(tfloat32)),
                (a_float64 // int64_3, expected, tarray(tfloat64)),
                (int64_3 // a_int32, expected_inv, tarray(tint64)),
                (int64_3 // a_int64, expected_inv, tarray(tint64)),
                (int64_3 // a_float32, expected_inv, tarray(tfloat32)),
                (int64_3 // a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 // int64_3s, expected, tarray(tint64)),
                (a_int64 // int64_3s, expected, tarray(tint64)),
                (a_float32 // int64_3s, expected, tarray(tfloat32)),
                (a_float64 // int64_3s, expected, tarray(tfloat64)),
                (a_int32 // float32_3, expected, tarray(tfloat32)),
                (a_int64 // float32_3, expected, tarray(tfloat32)),
                (a_float32 // float32_3, expected, tarray(tfloat32)),
                (a_float64 // float32_3, expected, tarray(tfloat64)),
                (float32_3 // a_int32, expected_inv, tarray(tfloat32)),
                (float32_3 // a_int64, expected_inv, tarray(tfloat32)),
                (float32_3 // a_float32, expected_inv, tarray(tfloat32)),
                (float32_3 // a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 // float32_3s, expected, tarray(tfloat32)),
                (a_int64 // float32_3s, expected, tarray(tfloat32)),
                (a_float32 // float32_3s, expected, tarray(tfloat32)),
                (a_float64 // float32_3s, expected, tarray(tfloat64)),
                (a_int32 // float64_3, expected, tarray(tfloat64)),
                (a_int64 // float64_3, expected, tarray(tfloat64)),
                (a_float32 // float64_3, expected, tarray(tfloat64)),
                (a_float64 // float64_3, expected, tarray(tfloat64)),
                (float64_3 // a_int32, expected_inv, tarray(tfloat64)),
                (float64_3 // a_int64, expected_inv, tarray(tfloat64)),
                (float64_3 // a_float32, expected_inv, tarray(tfloat64)),
                (float64_3 // a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 // float64_3s, expected, tarray(tfloat64)),
                (a_int64 // float64_3s, expected, tarray(tfloat64)),
                (a_float32 // float64_3s, expected, tarray(tfloat64)),
                (a_float64 // float64_3s, expected, tarray(tfloat64)),
            ]
        )

    def test_addition(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.missing(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [5, 7, 11, 19, None]
        expected_inv = expected

        _test_many_equal_typed(
            [
                (a_int32 + 3, expected, tarray(tint32)),
                (a_int64 + 3, expected, tarray(tint64)),
                (a_float32 + 3, expected, tarray(tfloat32)),
                (a_float64 + 3, expected, tarray(tfloat64)),
                (3 + a_int32, expected_inv, tarray(tint32)),
                (3 + a_int64, expected_inv, tarray(tint64)),
                (3 + a_float32, expected_inv, tarray(tfloat32)),
                (3 + a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 + int32_3s, expected, tarray(tint32)),
                (a_int64 + int32_3s, expected, tarray(tint64)),
                (a_float32 + int32_3s, expected, tarray(tfloat32)),
                (a_float64 + int32_3s, expected, tarray(tfloat64)),
                (a_int32 + int64_3, expected, tarray(tint64)),
                (a_int64 + int64_3, expected, tarray(tint64)),
                (a_float32 + int64_3, expected, tarray(tfloat32)),
                (a_float64 + int64_3, expected, tarray(tfloat64)),
                (int64_3 + a_int32, expected_inv, tarray(tint64)),
                (int64_3 + a_int64, expected_inv, tarray(tint64)),
                (int64_3 + a_float32, expected_inv, tarray(tfloat32)),
                (int64_3 + a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 + int64_3s, expected, tarray(tint64)),
                (a_int64 + int64_3s, expected, tarray(tint64)),
                (a_float32 + int64_3s, expected, tarray(tfloat32)),
                (a_float64 + int64_3s, expected, tarray(tfloat64)),
                (a_int32 + float32_3, expected, tarray(tfloat32)),
                (a_int64 + float32_3, expected, tarray(tfloat32)),
                (a_float32 + float32_3, expected, tarray(tfloat32)),
                (a_float64 + float32_3, expected, tarray(tfloat64)),
                (float32_3 + a_int32, expected_inv, tarray(tfloat32)),
                (float32_3 + a_int64, expected_inv, tarray(tfloat32)),
                (float32_3 + a_float32, expected_inv, tarray(tfloat32)),
                (float32_3 + a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 + float32_3s, expected, tarray(tfloat32)),
                (a_int64 + float32_3s, expected, tarray(tfloat32)),
                (a_float32 + float32_3s, expected, tarray(tfloat32)),
                (a_float64 + float32_3s, expected, tarray(tfloat64)),
                (a_int32 + float64_3, expected, tarray(tfloat64)),
                (a_int64 + float64_3, expected, tarray(tfloat64)),
                (a_float32 + float64_3, expected, tarray(tfloat64)),
                (a_float64 + float64_3, expected, tarray(tfloat64)),
                (float64_3 + a_int32, expected_inv, tarray(tfloat64)),
                (float64_3 + a_int64, expected_inv, tarray(tfloat64)),
                (float64_3 + a_float32, expected_inv, tarray(tfloat64)),
                (float64_3 + a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 + float64_3s, expected, tarray(tfloat64)),
                (a_int64 + float64_3s, expected, tarray(tfloat64)),
                (a_float32 + float64_3s, expected, tarray(tfloat64)),
                (a_float64 + float64_3s, expected, tarray(tfloat64)),
            ]
        )

    def test_subtraction(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.missing(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [-1, 1, 5, 13, None]
        expected_inv = [1, -1, -5, -13, None]

        _test_many_equal_typed(
            [
                (a_int32 - 3, expected, tarray(tint32)),
                (a_int64 - 3, expected, tarray(tint64)),
                (a_float32 - 3, expected, tarray(tfloat32)),
                (a_float64 - 3, expected, tarray(tfloat64)),
                (3 - a_int32, expected_inv, tarray(tint32)),
                (3 - a_int64, expected_inv, tarray(tint64)),
                (3 - a_float32, expected_inv, tarray(tfloat32)),
                (3 - a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 - int32_3s, expected, tarray(tint32)),
                (a_int64 - int32_3s, expected, tarray(tint64)),
                (a_float32 - int32_3s, expected, tarray(tfloat32)),
                (a_float64 - int32_3s, expected, tarray(tfloat64)),
                (a_int32 - int64_3, expected, tarray(tint64)),
                (a_int64 - int64_3, expected, tarray(tint64)),
                (a_float32 - int64_3, expected, tarray(tfloat32)),
                (a_float64 - int64_3, expected, tarray(tfloat64)),
                (int64_3 - a_int32, expected_inv, tarray(tint64)),
                (int64_3 - a_int64, expected_inv, tarray(tint64)),
                (int64_3 - a_float32, expected_inv, tarray(tfloat32)),
                (int64_3 - a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 - int64_3s, expected, tarray(tint64)),
                (a_int64 - int64_3s, expected, tarray(tint64)),
                (a_float32 - int64_3s, expected, tarray(tfloat32)),
                (a_float64 - int64_3s, expected, tarray(tfloat64)),
                (a_int32 - float32_3, expected, tarray(tfloat32)),
                (a_int64 - float32_3, expected, tarray(tfloat32)),
                (a_float32 - float32_3, expected, tarray(tfloat32)),
                (a_float64 - float32_3, expected, tarray(tfloat64)),
                (float32_3 - a_int32, expected_inv, tarray(tfloat32)),
                (float32_3 - a_int64, expected_inv, tarray(tfloat32)),
                (float32_3 - a_float32, expected_inv, tarray(tfloat32)),
                (float32_3 - a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 - float32_3s, expected, tarray(tfloat32)),
                (a_int64 - float32_3s, expected, tarray(tfloat32)),
                (a_float32 - float32_3s, expected, tarray(tfloat32)),
                (a_float64 - float32_3s, expected, tarray(tfloat64)),
                (a_int32 - float64_3, expected, tarray(tfloat64)),
                (a_int64 - float64_3, expected, tarray(tfloat64)),
                (a_float32 - float64_3, expected, tarray(tfloat64)),
                (a_float64 - float64_3, expected, tarray(tfloat64)),
                (float64_3 - a_int32, expected_inv, tarray(tfloat64)),
                (float64_3 - a_int64, expected_inv, tarray(tfloat64)),
                (float64_3 - a_float32, expected_inv, tarray(tfloat64)),
                (float64_3 - a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 - float64_3s, expected, tarray(tfloat64)),
                (a_int64 - float64_3s, expected, tarray(tfloat64)),
                (a_float32 - float64_3s, expected, tarray(tfloat64)),
                (a_float64 - float64_3s, expected, tarray(tfloat64)),
            ]
        )

    def test_multiplication(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.missing(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [6, 12, 24, 48, None]
        expected_inv = expected

        _test_many_equal_typed(
            [
                (a_int32 * 3, expected, tarray(tint32)),
                (a_int64 * 3, expected, tarray(tint64)),
                (a_float32 * 3, expected, tarray(tfloat32)),
                (a_float64 * 3, expected, tarray(tfloat64)),
                (3 * a_int32, expected_inv, tarray(tint32)),
                (3 * a_int64, expected_inv, tarray(tint64)),
                (3 * a_float32, expected_inv, tarray(tfloat32)),
                (3 * a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 * int32_3s, expected, tarray(tint32)),
                (a_int64 * int32_3s, expected, tarray(tint64)),
                (a_float32 * int32_3s, expected, tarray(tfloat32)),
                (a_float64 * int32_3s, expected, tarray(tfloat64)),
                (a_int32 * int64_3, expected, tarray(tint64)),
                (a_int64 * int64_3, expected, tarray(tint64)),
                (a_float32 * int64_3, expected, tarray(tfloat32)),
                (a_float64 * int64_3, expected, tarray(tfloat64)),
                (int64_3 * a_int32, expected_inv, tarray(tint64)),
                (int64_3 * a_int64, expected_inv, tarray(tint64)),
                (int64_3 * a_float32, expected_inv, tarray(tfloat32)),
                (int64_3 * a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 * int64_3s, expected, tarray(tint64)),
                (a_int64 * int64_3s, expected, tarray(tint64)),
                (a_float32 * int64_3s, expected, tarray(tfloat32)),
                (a_float64 * int64_3s, expected, tarray(tfloat64)),
                (a_int32 * float32_3, expected, tarray(tfloat32)),
                (a_int64 * float32_3, expected, tarray(tfloat32)),
                (a_float32 * float32_3, expected, tarray(tfloat32)),
                (a_float64 * float32_3, expected, tarray(tfloat64)),
                (float32_3 * a_int32, expected_inv, tarray(tfloat32)),
                (float32_3 * a_int64, expected_inv, tarray(tfloat32)),
                (float32_3 * a_float32, expected_inv, tarray(tfloat32)),
                (float32_3 * a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 * float32_3s, expected, tarray(tfloat32)),
                (a_int64 * float32_3s, expected, tarray(tfloat32)),
                (a_float32 * float32_3s, expected, tarray(tfloat32)),
                (a_float64 * float32_3s, expected, tarray(tfloat64)),
                (a_int32 * float64_3, expected, tarray(tfloat64)),
                (a_int64 * float64_3, expected, tarray(tfloat64)),
                (a_float32 * float64_3, expected, tarray(tfloat64)),
                (a_float64 * float64_3, expected, tarray(tfloat64)),
                (float64_3 * a_int32, expected_inv, tarray(tfloat64)),
                (float64_3 * a_int64, expected_inv, tarray(tfloat64)),
                (float64_3 * a_float32, expected_inv, tarray(tfloat64)),
                (float64_3 * a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 * float64_3s, expected, tarray(tfloat64)),
                (a_int64 * float64_3s, expected, tarray(tfloat64)),
                (a_float32 * float64_3s, expected, tarray(tfloat64)),
                (a_float64 * float64_3s, expected, tarray(tfloat64)),
            ]
        )

    def test_exponentiation(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.missing(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [8, 64, 512, 4096, None]
        expected_inv = [9.0, 81.0, 6561.0, 43046721.0, None]

        _test_many_equal_typed(
            [
                (a_int32**3, expected, tarray(tfloat64)),
                (a_int64**3, expected, tarray(tfloat64)),
                (a_float32**3, expected, tarray(tfloat64)),
                (a_float64**3, expected, tarray(tfloat64)),
                (3**a_int32, expected_inv, tarray(tfloat64)),
                (3**a_int64, expected_inv, tarray(tfloat64)),
                (3**a_float32, expected_inv, tarray(tfloat64)),
                (3**a_float64, expected_inv, tarray(tfloat64)),
                (a_int32**int32_3s, expected, tarray(tfloat64)),
                (a_int64**int32_3s, expected, tarray(tfloat64)),
                (a_float32**int32_3s, expected, tarray(tfloat64)),
                (a_float64**int32_3s, expected, tarray(tfloat64)),
                (a_int32**int64_3, expected, tarray(tfloat64)),
                (a_int64**int64_3, expected, tarray(tfloat64)),
                (a_float32**int64_3, expected, tarray(tfloat64)),
                (a_float64**int64_3, expected, tarray(tfloat64)),
                (int64_3**a_int32, expected_inv, tarray(tfloat64)),
                (int64_3**a_int64, expected_inv, tarray(tfloat64)),
                (int64_3**a_float32, expected_inv, tarray(tfloat64)),
                (int64_3**a_float64, expected_inv, tarray(tfloat64)),
                (a_int32**int64_3s, expected, tarray(tfloat64)),
                (a_int64**int64_3s, expected, tarray(tfloat64)),
                (a_float32**int64_3s, expected, tarray(tfloat64)),
                (a_float64**int64_3s, expected, tarray(tfloat64)),
                (a_int32**float32_3, expected, tarray(tfloat64)),
                (a_int64**float32_3, expected, tarray(tfloat64)),
                (a_float32**float32_3, expected, tarray(tfloat64)),
                (a_float64**float32_3, expected, tarray(tfloat64)),
                (float32_3**a_int32, expected_inv, tarray(tfloat64)),
                (float32_3**a_int64, expected_inv, tarray(tfloat64)),
                (float32_3**a_float32, expected_inv, tarray(tfloat64)),
                (float32_3**a_float64, expected_inv, tarray(tfloat64)),
                (a_int32**float32_3s, expected, tarray(tfloat64)),
                (a_int64**float32_3s, expected, tarray(tfloat64)),
                (a_float32**float32_3s, expected, tarray(tfloat64)),
                (a_float64**float32_3s, expected, tarray(tfloat64)),
                (a_int32**float64_3, expected, tarray(tfloat64)),
                (a_int64**float64_3, expected, tarray(tfloat64)),
                (a_float32**float64_3, expected, tarray(tfloat64)),
                (a_float64**float64_3, expected, tarray(tfloat64)),
                (float64_3**a_int32, expected_inv, tarray(tfloat64)),
                (float64_3**a_int64, expected_inv, tarray(tfloat64)),
                (float64_3**a_float32, expected_inv, tarray(tfloat64)),
                (float64_3**a_float64, expected_inv, tarray(tfloat64)),
                (a_int32**float64_3s, expected, tarray(tfloat64)),
                (a_int64**float64_3s, expected, tarray(tfloat64)),
                (a_float32**float64_3s, expected, tarray(tfloat64)),
                (a_float64**float64_3s, expected, tarray(tfloat64)),
            ]
        )

    def test_modulus(self):
        a_int32 = hl.array([2, 4, 8, 16, hl.missing(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.array([4, 4, 4, 4, hl.missing(tint32)])
        int32_3s = hl.array([3, 3, 3, 3, hl.missing(tint32)])
        int64_3 = hl.int64(3)
        int64_3s = int32_3s.map(lambda x: hl.int64(x))
        float32_3 = hl.float32(3)
        float32_3s = int32_3s.map(lambda x: hl.float32(x))
        float64_3 = hl.float64(3)
        float64_3s = int32_3s.map(lambda x: hl.float64(x))

        expected = [2, 1, 2, 1, None]
        expected_inv = [1, 3, 3, 3, None]

        _test_many_equal_typed(
            [
                (a_int32 % 3, expected, tarray(tint32)),
                (a_int64 % 3, expected, tarray(tint64)),
                (a_float32 % 3, expected, tarray(tfloat32)),
                (a_float64 % 3, expected, tarray(tfloat64)),
                (3 % a_int32, expected_inv, tarray(tint32)),
                (3 % a_int64, expected_inv, tarray(tint64)),
                (3 % a_float32, expected_inv, tarray(tfloat32)),
                (3 % a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 % int32_3s, expected, tarray(tint32)),
                (a_int64 % int32_3s, expected, tarray(tint64)),
                (a_float32 % int32_3s, expected, tarray(tfloat32)),
                (a_float64 % int32_3s, expected, tarray(tfloat64)),
                (a_int32 % int64_3, expected, tarray(tint64)),
                (a_int64 % int64_3, expected, tarray(tint64)),
                (a_float32 % int64_3, expected, tarray(tfloat32)),
                (a_float64 % int64_3, expected, tarray(tfloat64)),
                (int64_3 % a_int32, expected_inv, tarray(tint64)),
                (int64_3 % a_int64, expected_inv, tarray(tint64)),
                (int64_3 % a_float32, expected_inv, tarray(tfloat32)),
                (int64_3 % a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 % int64_3s, expected, tarray(tint64)),
                (a_int64 % int64_3s, expected, tarray(tint64)),
                (a_float32 % int64_3s, expected, tarray(tfloat32)),
                (a_float64 % int64_3s, expected, tarray(tfloat64)),
                (a_int32 % float32_3, expected, tarray(tfloat32)),
                (a_int64 % float32_3, expected, tarray(tfloat32)),
                (a_float32 % float32_3, expected, tarray(tfloat32)),
                (a_float64 % float32_3, expected, tarray(tfloat64)),
                (float32_3 % a_int32, expected_inv, tarray(tfloat32)),
                (float32_3 % a_int64, expected_inv, tarray(tfloat32)),
                (float32_3 % a_float32, expected_inv, tarray(tfloat32)),
                (float32_3 % a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 % float32_3s, expected, tarray(tfloat32)),
                (a_int64 % float32_3s, expected, tarray(tfloat32)),
                (a_float32 % float32_3s, expected, tarray(tfloat32)),
                (a_float64 % float32_3s, expected, tarray(tfloat64)),
                (a_int32 % float64_3, expected, tarray(tfloat64)),
                (a_int64 % float64_3, expected, tarray(tfloat64)),
                (a_float32 % float64_3, expected, tarray(tfloat64)),
                (a_float64 % float64_3, expected, tarray(tfloat64)),
                (float64_3 % a_int32, expected_inv, tarray(tfloat64)),
                (float64_3 % a_int64, expected_inv, tarray(tfloat64)),
                (float64_3 % a_float32, expected_inv, tarray(tfloat64)),
                (float64_3 % a_float64, expected_inv, tarray(tfloat64)),
                (a_int32 % float64_3s, expected, tarray(tfloat64)),
                (a_int64 % float64_3s, expected, tarray(tfloat64)),
                (a_float32 % float64_3s, expected, tarray(tfloat64)),
                (a_float64 % float64_3s, expected, tarray(tfloat64)),
            ]
        )

    def test_comparisons(self):
        f0 = hl.float(0.0)
        fnull = hl.missing(tfloat)
        finf = hl.float(float('inf'))
        fnan = hl.float(float('nan'))

        _test_many_equal_typed(
            [
                (f0 == fnull, None, tbool),
                (f0 < fnull, None, tbool),
                (f0 != fnull, None, tbool),
                (fnan == fnan, False, tbool),
                (f0 == f0, True, tbool),
                (finf == finf, True, tbool),
                (f0 < finf, True, tbool),
                (f0 > finf, False, tbool),
                (fnan <= finf, False, tbool),
                (fnan >= finf, False, tbool),
            ]
        )

    def test_bools_can_math(self):
        b1 = hl.literal(True)
        b2 = hl.literal(False)

        b_array = hl.literal([True, False])
        f1 = hl.float64(5.5)
        f_array = hl.array([1.5, 2.5])

        _test_many_equal(
            [
                (hl.int32(b1), 1),
                (hl.int64(b1), 1),
                (hl.float32(b1), 1.0),
                (hl.float64(b1), 1.0),
                (b1 * b2, 0),
                (b1 + b2, 1),
                (b1 - b2, 1),
                (b1 / b1, 1.0),
                (f1 * b2, 0.0),
                (b_array + f1, [6.5, 5.5]),
                (b_array + f_array, [2.5, 2.5]),
            ]
        )

    def test_int_typecheck(self):
        _test_many_equal([(hl.literal(None, dtype='int32'), None), (hl.literal(None, dtype='int64'), None)])

    def test_is_transition(self):
        _test_many_equal(
            [
                (hl.is_transition("A", "G"), True),
                (hl.is_transition("C", "T"), True),
                (hl.is_transition("AA", "AG"), True),
                (hl.is_transition("AA", "G"), False),
                (hl.is_transition("ACA", "AGA"), False),
                (hl.is_transition("A", "T"), False),
            ]
        )

    def test_is_transversion(self):
        _test_many_equal(
            [
                (hl.is_transversion("A", "T"), True),
                (hl.is_transversion("A", "G"), False),
                (hl.is_transversion("AA", "AT"), True),
                (hl.is_transversion("AA", "T"), False),
                (hl.is_transversion("ACCC", "ACCT"), False),
            ]
        )

    def test_is_snp(self):
        _test_many_equal(
            [
                (hl.is_snp("A", "T"), True),
                (hl.is_snp("A", "G"), True),
                (hl.is_snp("C", "G"), True),
                (hl.is_snp("CC", "CG"), True),
                (hl.is_snp("AT", "AG"), True),
                (hl.is_snp("ATCCC", "AGCCC"), True),
            ]
        )

    def test_is_mnp(self):
        _test_many_equal([(hl.is_mnp("ACTGAC", "ATTGTT"), True), (hl.is_mnp("CA", "TT"), True)])

    def test_is_insertion(self):
        _test_many_equal([(hl.is_insertion("A", "ATGC"), True), (hl.is_insertion("ATT", "ATGCTT"), True)])

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
            hl.eval(
                hl.tuple(
                    (
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
                    )
                )
            ),
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
            ),
        )

    def test_hamming(self):
        _test_many_equal(
            [(hl.hamming('A', 'T'), 1), (hl.hamming('AAAAA', 'AAAAT'), 1), (hl.hamming('abcde', 'edcba'), 4)]
        )

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
        cNull = hl.missing(tcall)
        call_expr_1 = hl.call(1, 2, phased=True)
        a0 = hl.literal(1)
        a1 = 2
        phased = hl.literal(True)
        call_expr_2 = hl.call(a0, a1, phased=phased)
        call_expr_3 = hl.parse_call("1|2")
        call_expr_4 = hl.unphased_diploid_gt_index_call(2)

        _test_many_equal_typed(
            [
                (c2_homref.ploidy, 2, tint32),
                (c2_homref[0], 0, tint32),
                (c2_homref[1], 0, tint32),
                (c2_homref.phased, False, tbool),
                (c2_homref.is_hom_ref(), True, tbool),
                (c2_het.ploidy, 2, tint32),
                (c2_het[0], 1, tint32),
                (c2_het[1], 0, tint32),
                (c2_het.phased, True, tbool),
                (c2_het.is_het(), True, tbool),
                (c2_homvar.ploidy, 2, tint32),
                (c2_homvar[0], 1, tint32),
                (c2_homvar[1], 1, tint32),
                (c2_homvar.phased, False, tbool),
                (c2_homvar.is_hom_var(), True, tbool),
                (c2_homvar.unphased_diploid_gt_index(), 2, tint32),
                (c2_hetvar.ploidy, 2, tint32),
                (c2_hetvar[0], 2, tint32),
                (c2_hetvar[1], 1, tint32),
                (c2_hetvar.phased, True, tbool),
                (c2_hetvar.is_hom_var(), False, tbool),
                (c2_hetvar.is_het_non_ref(), True, tbool),
                (c1.ploidy, 1, tint32),
                (c1[0], 1, tint32),
                (c1.phased, False, tbool),
                (c1.is_hom_var(), True, tbool),
                (c0.ploidy, 0, tint32),
                (c0.phased, False, tbool),
                (c0.is_hom_var(), False, tbool),
                (cNull.ploidy, None, tint32),
                (cNull[0], None, tint32),
                (cNull.phased, None, tbool),
                (cNull.is_hom_var(), None, tbool),
                (call_expr_1[0], 1, tint32),
                (call_expr_1[1], 2, tint32),
                (call_expr_1.ploidy, 2, tint32),
                (call_expr_2[0], 1, tint32),
                (call_expr_2[1], 2, tint32),
                (call_expr_2.ploidy, 2, tint32),
                (call_expr_3[0], 1, tint32),
                (call_expr_3[1], 2, tint32),
                (call_expr_3.ploidy, 2, tint32),
                (call_expr_4[0], 1, tint32),
                (call_expr_4[1], 1, tint32),
                (call_expr_4.ploidy, 2, tint32),
            ]
        )

    def test_call_unphase(self):

        calls = [
            hl.Call([0], phased=True),
            hl.Call([0], phased=False),
            hl.Call([1], phased=True),
            hl.Call([1], phased=False),
            hl.Call([0, 0], phased=True),
            hl.Call([3, 0], phased=True),
            hl.Call([1, 1], phased=False),
            hl.Call([0, 0], phased=False),
        ]

        expected = [
            hl.Call([0], phased=False),
            hl.Call([0], phased=False),
            hl.Call([1], phased=False),
            hl.Call([1], phased=False),
            hl.Call([0, 0], phased=False),
            hl.Call([0, 3], phased=False),
            hl.Call([1, 1], phased=False),
            hl.Call([0, 0], phased=False),
        ]

        assert hl.eval(hl.literal(calls).map(lambda x: x.unphase())) == expected

    def test_call_contains_allele(self):
        c1 = hl.call(1, phased=True)
        c2 = hl.call(1, phased=False)
        c3 = hl.call(3, 1, phased=True)
        c4 = hl.call(1, 3, phased=False)

        for i, b in enumerate(
            hl.eval(
                tuple(
                    [
                        c1.contains_allele(1),
                        ~c1.contains_allele(0),
                        ~c1.contains_allele(2),
                        c2.contains_allele(1),
                        ~c2.contains_allele(0),
                        ~c2.contains_allele(2),
                        c3.contains_allele(1),
                        c3.contains_allele(3),
                        ~c3.contains_allele(0),
                        ~c3.contains_allele(2),
                        c4.contains_allele(1),
                        c4.contains_allele(3),
                        ~c4.contains_allele(0),
                        ~c4.contains_allele(2),
                    ]
                )
            )
        ):
            assert b, i

    def test_call_unphase_diploid_gt_index(self):
        calls_and_indices = [
            (hl.call(0, 0), 0),
            (hl.call(0, 1), 1),
            (hl.call(1, 1), 2),
            (hl.call(0, 2), 3),
            (hl.call(0, 0, phased=True), 0),
            (hl.call(1, 1, phased=True), 2),
            (hl.call(2, 0, phased=True), 3),
        ]

        gt_idx = tuple(c[0].unphased_diploid_gt_index() for c in calls_and_indices)
        assert hl.eval(gt_idx) == tuple(i for c, i in calls_and_indices)

    def test_parse_variant(self):
        self.assertEqual(hl.eval(hl.parse_variant('1:1:A:T')), hl.Struct(locus=hl.Locus('1', 1), alleles=['A', 'T']))

    def test_locus_to_global_position(self):
        self.assertEqual(hl.eval(hl.locus('chr22', 1, 'GRCh38').global_position()), 2824183054)

    def test_locus_from_global_position(self):
        self.assertEqual(
            hl.eval(hl.locus_from_global_position(2824183054, 'GRCh38')), hl.eval(hl.locus('chr22', 1, 'GRCh38'))
        )

    def test_locus_window(self):
        locus = hl.Locus('22', 123456, reference_genome='GRCh37')

        lit = hl.literal(locus)
        results = hl.eval(
            hl.struct(
                zeros=lit.window(0, 0), ones=lit.window(1, 1), big_windows=lit.window(1_000_000_000, 1_000_000_000)
            )
        )

        pt = hl.tinterval(hl.tlocus('GRCh37'))

        assert results.zeros == hl.Interval(
            hl.Locus('22', 123456), hl.Locus('22', 123456), includes_start=True, includes_end=True, point_type=pt
        )
        assert results.ones == hl.Interval(
            hl.Locus('22', 123455), hl.Locus('22', 123457), includes_start=True, includes_end=True, point_type=pt
        )
        assert results.big_windows == hl.Interval(
            hl.Locus('22', 1),
            hl.Locus('22', hl.get_reference('GRCh37').contig_length('22')),
            includes_start=True,
            includes_end=True,
            point_type=pt,
        )

    def test_dict_conversions(self):
        self.assertEqual(sorted(hl.eval(hl.array({1: 1, 2: 2}))), [(1, 1), (2, 2)])
        self.assertEqual(hl.eval(hl.dict(hl.array({1: 1, 2: 2}))), {1: 1, 2: 2})

        self.assertEqual(hl.eval(hl.dict([('1', 2), ('2', 3)])), {'1': 2, '2': 3})
        self.assertEqual(hl.eval(hl.dict({('1', 2), ('2', 3)})), {'1': 2, '2': 3})
        self.assertEqual(hl.eval(hl.dict([('1', 2), (hl.missing(tstr), 3)])), {'1': 2, None: 3})
        self.assertEqual(hl.eval(hl.dict({('1', 2), (hl.missing(tstr), 3)})), {'1': 2, None: 3})

    def test_zip(self):
        a1 = [1, 2, 3]
        a2 = ['a', 'b']
        a3 = [[1]]

        self.assertEqual(hl.eval(hl.zip(a1, a2)), [(1, 'a'), (2, 'b')])
        self.assertEqual(hl.eval(hl.zip(a1, a2, fill_missing=True)), [(1, 'a'), (2, 'b'), (3, None)])

        self.assertEqual(
            hl.eval(hl.zip(a3, a2, a1, fill_missing=True)), [([1], 'a', 1), (None, 'b', 2), (None, None, 3)]
        )
        self.assertEqual(hl.eval(hl.zip(a3, a2, a1)), [([1], 'a', 1)])

    def test_any_form_1(self):
        self.assertEqual(hl.eval(hl.any()), False)

        self.assertEqual(hl.eval(hl.any(True)), True)
        self.assertEqual(hl.eval(hl.any(False)), False)

        self.assertEqual(hl.eval(hl.any(True, True)), True)
        self.assertEqual(hl.eval(hl.any(True, False)), True)
        self.assertEqual(hl.eval(hl.any(False, True)), True)
        self.assertEqual(hl.eval(hl.any(False, False)), False)

    def test_all_form_1(self):
        self.assertEqual(hl.eval(hl.all()), True)

        self.assertEqual(hl.eval(hl.all(True)), True)
        self.assertEqual(hl.eval(hl.all(False)), False)

        self.assertEqual(hl.eval(hl.all(True, True)), True)
        self.assertEqual(hl.eval(hl.all(True, False)), False)
        self.assertEqual(hl.eval(hl.all(False, True)), False)
        self.assertEqual(hl.eval(hl.all(False, False)), False)

    def test_any_form_2(self):
        self.assertEqual(hl.eval(hl.any(hl.empty_array(hl.tbool))), False)

        self.assertEqual(hl.eval(hl.any([True])), True)
        self.assertEqual(hl.eval(hl.any([False])), False)

        self.assertEqual(hl.eval(hl.any([True, True])), True)
        self.assertEqual(hl.eval(hl.any([True, False])), True)
        self.assertEqual(hl.eval(hl.any([False, True])), True)
        self.assertEqual(hl.eval(hl.any([False, False])), False)

    def test_all_form_2(self):
        self.assertEqual(hl.eval(hl.all(hl.empty_array(hl.tbool))), True)

        self.assertEqual(hl.eval(hl.all([True])), True)
        self.assertEqual(hl.eval(hl.all([False])), False)

        self.assertEqual(hl.eval(hl.all([True, True])), True)
        self.assertEqual(hl.eval(hl.all([True, False])), False)
        self.assertEqual(hl.eval(hl.all([False, True])), False)
        self.assertEqual(hl.eval(hl.all([False, False])), False)

    def test_any_form_3(self):
        self.assertEqual(hl.eval(hl.any(lambda x: x % 2 == 0, [1, 3, 5])), False)
        self.assertEqual(hl.eval(hl.any(lambda x: x % 2 == 0, [1, 3, 5, 6])), True)

    def test_all_form_3(self):
        self.assertEqual(hl.eval(hl.all(lambda x: x % 2 == 0, [1, 3, 5, 6])), False)
        self.assertEqual(hl.eval(hl.all(lambda x: x % 2 == 0, [2, 6])), True)

    def test_array_methods(self):
        _test_many_equal(
            [
                (hl.map(lambda x: x % 2 == 0, [0, 1, 4, 6]), [True, False, True, True]),
                (hl.len([0, 1, 4, 6]), 4),
                (math.isnan(hl.eval(hl.mean(hl.empty_array(hl.tint)))), True),
                (hl.mean([0, 1, 4, 6, hl.missing(tint32)]), 2.75),
                (hl.median(hl.empty_array(hl.tint)), None),
                (1 <= hl.eval(hl.median([0, 1, 4, 6])) <= 4, True),
            ]
            + [
                test
                for f in [
                    lambda x: hl.int32(x),
                    lambda x: hl.int64(x),
                    lambda x: hl.float32(x),
                    lambda x: hl.float64(x),
                ]
                for test in [(hl.product([f(x) for x in [1, 4, 6]]), 24), (hl.sum([f(x) for x in [1, 4, 6]]), 11)]
            ]
            + [
                (hl.group_by(lambda x: x % 2 == 0, [0, 1, 4, 6]), {True: [0, 4, 6], False: [1]}),
                (hl.flatmap(lambda x: hl.range(0, x), [1, 2, 3]), [0, 0, 1, 0, 1, 2]),
                (
                    hl.flatmap(lambda x: hl.set(hl.range(0, x.length()).map(lambda i: x[i])), {"ABC", "AAa", "BD"}),
                    {'A', 'a', 'B', 'C', 'D'},
                ),
            ]
        )

    def test_starmap(self):
        self.assertEqual(hl.eval(hl.array([(1, 2), (2, 3)]).starmap(lambda x, y: x + y)), [3, 5])

    def test_array_corr(self):
        x1 = [random.uniform(-10, 10) for x in range(10)]
        x2 = [random.uniform(-10, 10) for x in range(10)]
        self.assertAlmostEqual(hl.eval(hl.corr(x1, x2)), pearsonr(x1, x2)[0])

    def test_array_corr_missingness(self):
        x1 = [None, None, 5.0] + [random.uniform(-10, 10) for x in range(15)]
        x2 = [None, 5.0, None] + [random.uniform(-10, 10) for x in range(15)]
        self.assertAlmostEqual(
            hl.eval(hl.corr(hl.literal(x1, 'array<float>'), hl.literal(x2, 'array<float>'))),
            pearsonr(x1[3:], x2[3:])[0],
        )

    def test_array_grouped(self):
        x = hl.array([0, 1, 2, 3, 4])
        assert hl.eval(x.grouped(1)) == [[0], [1], [2], [3], [4]]
        assert hl.eval(x.grouped(2)) == [[0, 1], [2, 3], [4]]
        assert hl.eval(x.grouped(5)) == [[0, 1, 2, 3, 4]]
        assert hl.eval(x.grouped(100)) == [[0, 1, 2, 3, 4]]

    def test_array_find(self):
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.missing(hl.tarray(hl.tint32)))), None)
        self.assertEqual(hl.eval(hl.find(lambda x: hl.missing(hl.tbool), [1, 0, -4, 6])), None)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, [1, 0, -4, 6])), -4)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, [1, 0, 4, 6])), None)

    def test_set_find(self):
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.missing(hl.tset(hl.tint32)))), None)
        self.assertEqual(hl.eval(hl.find(lambda x: hl.missing(hl.tbool), hl.set([1, 0, -4, 6]))), None)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.set([1, 0, -4, 6]))), -4)
        self.assertEqual(hl.eval(hl.find(lambda x: x < 0, hl.set([1, 0, 4, 6]))), None)

    def test_sorted(self):
        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, 3, 2], lambda x: x % 2)), [0, 4, 2, 1, 3])
        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, 3, 2], lambda x: x % 2, reverse=True)), [1, 3, 0, 4, 2])

        self.assertEqual(hl.eval(hl.sorted([0, 1, 4, hl.missing(tint), 3, 2], lambda x: x)), [0, 1, 2, 3, 4, None])
        self.assertEqual(
            hl.sorted([0, 1, 4, hl.missing(tint), 3, 2], lambda x: x, reverse=True).collect()[0], [4, 3, 2, 1, 0, None]
        )
        self.assertEqual(
            hl.eval(hl.sorted([0, 1, 4, hl.missing(tint), 3, 2], lambda x: x, reverse=True)), [4, 3, 2, 1, 0, None]
        )

        self.assertEqual(hl.eval(hl.sorted({0, 1, 4, 3, 2})), [0, 1, 2, 3, 4])

        self.assertEqual(hl.eval(hl.sorted({"foo": 1, "bar": 2})), [("bar", 2), ("foo", 1)])

    def test_sort_by(self):
        self.assertEqual(
            hl.eval(hl._sort_by(["c", "aaa", "bb", hl.missing(hl.tstr)], lambda l, r: hl.len(l) < hl.len(r))),
            ["c", "bb", "aaa", None],
        )
        self.assertEqual(
            hl.eval(hl._sort_by([hl.Struct(x=i, y="foo", z=5.5) for i in [5, 3, 8, 2, 5]], lambda l, r: l.x < r.x)),
            [hl.Struct(x=i, y="foo", z=5.5) for i in [2, 3, 5, 5, 8]],
        )
        with self.assertRaises(hl.utils.java.FatalError):
            self.assertEqual(
                hl.eval(
                    hl._sort_by(
                        [hl.Struct(x=i, y="foo", z=5.5) for i in [5, 3, 8, 2, 5, hl.missing(hl.tint32)]],
                        lambda l, r: l.x < r.x,
                    )
                ),
                [hl.Struct(x=i, y="foo", z=5.5) for i in [2, 3, 5, 5, 8, None]],
            )

    def test_array_first(self):
        a = hl.array([1, 2, 3])
        assert hl.eval(a.first()) == 1
        assert hl.eval(a.filter(lambda x: x > 5).first()) is None

    def test_array_last(self):
        a = hl.array([1, 2, 3])
        assert hl.eval(a.last()) == 3
        assert hl.eval(a.filter(lambda x: x > 5).last()) is None

    def test_array_index(self):
        a = hl.array([1, 2, 3])
        assert hl.eval(a.index(2) == 1)
        assert hl.eval(a.index(4)) is None
        assert hl.eval(a.index(lambda x: x % 2 == 0) == 1)
        assert hl.eval(a.index(lambda x: x > 5)) is None

    def test_array_empty_struct(self):
        a = hl.array([hl.struct()])
        b = hl.literal([hl.struct()])
        assert hl.eval(a) == hl.eval(b)

    def test_bool_r_ops(self):
        self.assertTrue(hl.eval(hl.literal(True) & True))
        self.assertTrue(hl.eval(True & hl.literal(True)))
        self.assertTrue(hl.eval(hl.literal(False) | True))
        self.assertTrue(hl.eval(True | hl.literal(False)))

    def test_array_neg(self):
        self.assertEqual(hl.eval(-(hl.literal([1, 2, 3]))), [-1, -2, -3])

    def test_max(self):
        exprs_and_results = [
            (hl.max(1, 2), 2),
            (hl.max(1.0, 2), 2.0),
            (hl.max([1, 2]), 2),
            (hl.max([1.0, 2]), 2.0),
            (hl.max(0, 1.0, 2), 2.0),
            (hl.nanmax(0, 1.0, 2), 2.0),
            (hl.max(0, 1, 2), 2),
            (
                hl.max(
                    [
                        0,
                        10,
                        2,
                        3,
                        4,
                        5,
                        6,
                    ]
                ),
                10,
            ),
            (hl.max(0, 10, 2, 3, 4, 5, 6), 10),
            (hl.max([-5, -4, hl.missing(tint32), -3, -2, hl.missing(tint32)]), -2),
            (hl.max([float('nan'), -4, float('nan'), -3, -2, hl.missing(tint32)]), float('nan')),
            (hl.max(0.1, hl.missing('float'), 0.0), 0.1),
            (hl.max(0.1, hl.missing('float'), float('nan')), float('nan')),
            (hl.max(hl.missing('float'), float('nan')), float('nan')),
            (hl.max(0.1, hl.missing('float'), float('nan'), filter_missing=False), None),
            (hl.nanmax(0.1, hl.missing('float'), float('nan')), 0.1),
            (hl.max(hl.missing('float'), float('nan')), float('nan')),
            (hl.nanmax(hl.missing('float'), float('nan')), float('nan')),
            (hl.nanmax(hl.missing('float'), float('nan'), 1.1, filter_missing=False), None),
            (hl.max([0.1, hl.missing('float'), 0.0]), 0.1),
            (hl.max([hl.missing('float'), float('nan')]), float('nan')),
            (hl.max([0.1, hl.missing('float'), float('nan')]), float('nan')),
            (hl.max([0.1, hl.missing('float'), float('nan')], filter_missing=False), None),
            (hl.nanmax([0.1, hl.missing('float'), float('nan')]), 0.1),
            (hl.nanmax([float('nan'), 1.1, 0.1, hl.missing('float'), 0.0]), 1.1),
            (hl.max([float('nan'), 1.1, 0.1, hl.missing('float'), float('nan')]), float('nan')),
            (hl.max([float('nan'), 1.1, 0.1, hl.missing('float'), float('nan')], filter_missing=False), None),
            (hl.nanmax([float('nan'), 1.1, 0.1, hl.missing('float'), float('nan')]), 1.1),
            (hl.nanmax([hl.missing('float'), float('nan'), 1.1], filter_missing=False), None),
            (hl.max({0.1, hl.missing('float'), 0.0}), 0.1),
            (hl.max({hl.missing('float'), float('nan')}), float('nan')),
            (hl.nanmax({float('nan'), 1.1, 0.1, hl.missing('float'), 0.0}), 1.1),
            (hl.nanmax({hl.missing('float'), float('nan'), 1.1}, filter_missing=False), None),
        ]

        r = hl.eval(hl.tuple(x[0] for x in exprs_and_results))
        for i in range(len(r)):
            actual = r[i]
            expected = exprs_and_results[i][1]
            assert actual == expected or (
                actual is not None and expected is not None and (math.isnan(actual) and math.isnan(expected))
            ), f'{i}: {actual}, {expected}'

    def test_min(self):
        exprs_and_results = [
            (hl.min(1, 2), 1),
            (hl.min(1.0, 2), 1.0),
            (hl.min([1, 2]), 1),
            (hl.min([1.0, 2]), 1.0),
            (hl.min(0, 1.0, 2), 0.0),
            (hl.nanmin(0, 1.0, 2), 0.0),
            (hl.min(0, 1, 2), 0),
            (hl.min([10, 10, 2, 3, 4, 5, 6]), 2),
            (hl.min(0, 10, 2, 3, 4, 5, 6), 0),
            (hl.min([-5, -4, hl.missing(tint32), -3, -2, hl.missing(tint32)]), -5),
            (hl.min([float('nan'), -4, float('nan'), -3, -2, hl.missing(tint32)]), float('nan')),
            (hl.min(-0.1, hl.missing('float'), 0.0), -0.1),
            (hl.min(0.1, hl.missing('float'), float('nan')), float('nan')),
            (hl.min(hl.missing('float'), float('nan')), float('nan')),
            (hl.min(0.1, hl.missing('float'), float('nan'), filter_missing=False), None),
            (hl.nanmin(-0.1, hl.missing('float'), float('nan')), -0.1),
            (hl.min(hl.missing('float'), float('nan')), float('nan')),
            (hl.nanmin(hl.missing('float'), float('nan')), float('nan')),
            (hl.nanmin(hl.missing('float'), float('nan'), 1.1, filter_missing=False), None),
            (hl.min([-0.1, hl.missing('float'), 0.0]), -0.1),
            (hl.min([hl.missing('float'), float('nan')]), float('nan')),
            (hl.min([0.1, hl.missing('float'), float('nan')]), float('nan')),
            (hl.min([0.1, hl.missing('float'), float('nan')], filter_missing=False), None),
            (hl.nanmin([-0.1, hl.missing('float'), float('nan')]), -0.1),
            (hl.nanmin([float('nan'), -1.1, 0.1, hl.missing('float'), 0.0]), -1.1),
            (hl.min([float('nan'), 1.1, 0.1, hl.missing('float'), float('nan')]), float('nan')),
            (hl.min([float('nan'), 1.1, 0.1, hl.missing('float'), float('nan')], filter_missing=False), None),
            (hl.nanmin([float('nan'), 1.1, 0.1, hl.missing('float'), float('nan')]), 0.1),
            (hl.nanmin([hl.missing('float'), float('nan'), 1.1], filter_missing=False), None),
            (hl.min({-0.1, hl.missing('float'), 0.0}), -0.1),
            (hl.min({hl.missing('float'), float('nan')}), float('nan')),
            (hl.nanmin({float('nan'), 1.1, -0.1, hl.missing('float'), 0.0}), -0.1),
            (hl.nanmin({hl.missing('float'), float('nan'), 1.1}, filter_missing=False), None),
        ]

        r = hl.eval(hl.tuple(x[0] for x in exprs_and_results))
        for i in range(len(r)):
            actual = r[i]
            expected = exprs_and_results[i][1]
            assert actual == expected or (
                actual is not None and expected is not None and (math.isnan(actual) and math.isnan(expected))
            ), f'{i}: {actual}, {expected}'

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

    def test_show_expression(self):
        ds = hl.utils.range_matrix_table(3, 3)
        result = ds.col_idx.show(handler=str)
        assert (
            result
            == '''+---------+
| col_idx |
+---------+
|   int32 |
+---------+
|       0 |
|       1 |
|       2 |
+---------+
'''
        )

    @test_timeout(4 * 60)
    def test_export_genetic_data(self):
        mt = hl.balding_nichols_model(1, 3, 3)
        mt = mt.key_cols_by(s='s' + hl.str(mt.sample_idx))
        with hl.TemporaryFilename() as f:
            mt.GT.export(f)
            actual = hl.import_matrix_table(
                f, row_fields={'locus': hl.tstr, 'alleles': hl.tstr}, row_key=['locus', 'alleles'], entry_type=hl.tstr
            )
            actual = actual.rename({'col_id': 's'})
            actual = actual.key_rows_by(
                locus=hl.parse_locus(actual.locus),
                alleles=actual.alleles.replace('"', '').replace(r'\[', '').replace(r'\]', '').split(','),
            )
            actual = actual.transmute_entries(GT=hl.parse_call(actual.x))
            expected = mt.select_cols().select_globals().select_rows()
            expected.show()
            actual.show()
            assert expected._same(actual)

    def test_or_else_type_conversion(self):
        self.assertEqual(hl.eval(hl.or_else(0.5, 2)), 0.5)

    def test_coalesce(self):
        self.assertEqual(hl.eval(hl.coalesce(hl.missing('int'), hl.missing('int'), hl.missing('int'))), None)
        self.assertEqual(hl.eval(hl.coalesce(hl.missing('int'), hl.missing('int'), 2)), 2)
        self.assertEqual(hl.eval(hl.coalesce(hl.missing('int'), hl.missing('int'), 2.5)), 2.5)
        self.assertEqual(hl.eval(hl.coalesce(2.5)), 2.5)
        self.assertEqual(hl.eval(hl.coalesce(2.5, hl.missing('int'))), 2.5)
        self.assertEqual(hl.eval(hl.coalesce(hl.missing('int'), 2.5, hl.missing('int'))), 2.5)
        self.assertEqual(hl.eval(hl.coalesce(hl.missing('int'), 2.5, 100)), 2.5)
        self.assertEqual(hl.eval(hl.coalesce(hl.missing('int'), 2.5, hl.int(1) / 0)), 2.5)
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
        self.assertEqual(hl.eval(li), hl.utils.Interval(hl.genetics.Locus("1", 100), hl.genetics.Locus("1", 110)))
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

    def test_locus_interval_constructors(self):
        li_contig_start = hl.locus_interval('1', 0, 2, False, False, invalid_missing=True)
        self.assertTrue(
            hl.eval(li_contig_start)
            == hl.utils.Interval(
                hl.genetics.Locus("1", 1), hl.genetics.Locus("1", 2), includes_start=True, includes_end=False
            )
        )

        li_contig_middle1 = hl.locus_interval('1', 100, 100, True, False, invalid_missing=True)
        self.assertTrue(
            hl.eval(li_contig_middle1)
            == hl.utils.Interval(
                hl.genetics.Locus("1", 99), hl.genetics.Locus("1", 100), includes_start=False, includes_end=False
            )
        )

        li_contig_middle2 = hl.locus_interval('1', 100, 100, False, True, invalid_missing=True)
        self.assertTrue(
            hl.eval(li_contig_middle2)
            == hl.utils.Interval(
                hl.genetics.Locus("1", 100), hl.genetics.Locus("1", 101), includes_start=False, includes_end=False
            )
        )

        li_contig_end = hl.locus_interval('1', 249250621, 249250622, True, False, invalid_missing=True)
        self.assertTrue(
            hl.eval(li_contig_end)
            == hl.utils.Interval(
                hl.genetics.Locus("1", 249250621),
                hl.genetics.Locus("1", 249250621),
                includes_start=True,
                includes_end=True,
            )
        )

        li1 = hl.locus_interval('1', 0, 1, False, False, invalid_missing=True)
        li2 = hl.locus_interval('1', 0, 1, True, False, invalid_missing=True)
        li3 = hl.locus_interval('1', 20, 20, False, False, invalid_missing=True)
        li4 = hl.locus_interval('1', 249250621, 249250622, False, True, invalid_missing=True)
        li5 = hl.locus_interval('1', 20, 19, True, True, invalid_missing=True)

        for expr in [li1, li2, li3, li4, li5]:
            self.assertTrue(hl.eval(expr) is None)

        li_parsed = hl.parse_locus_interval('(1:20-20)', invalid_missing=True)
        self.assertTrue(hl.eval(li_parsed) is None)

    def test_locus_window_type(self):
        locus = hl.parse_locus('chr16:1231231', reference_genome='GRCh38')
        assert locus.dtype.reference_genome.name == 'GRCh38'
        i = locus.window(10, 10)
        assert i.dtype.point_type.reference_genome.name == 'GRCh38'

    def test_reference_genome_fns(self):
        self.assertTrue(hl.eval(hl.is_valid_contig('1', 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_contig('chr1', 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_contig('1', 'GRCh38')))
        self.assertTrue(hl.eval(hl.is_valid_contig('chr1', 'GRCh38')))

        self.assertTrue(hl.eval(hl.is_valid_locus('1', 325423, 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_locus('1', 0, 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_locus('1', 249250622, 'GRCh37')))
        self.assertFalse(hl.eval(hl.is_valid_locus('chr1', 2645, 'GRCh37')))

        assert hl.eval(hl.contig_length('5', 'GRCh37') == 180915260)
        with self.assertRaises(hl.utils.FatalError):
            hl.eval(hl.contig_length('chr5', 'GRCh37'))

    @test_timeout(batch=5 * 60)
    def test_initop_table(self):
        t = hl.utils.range_table(5, 3).annotate(GT=hl.call(0, 1)).annotate_globals(alleles=["A", "T"])

        self.assertTrue(
            t.aggregate(hl.agg.call_stats(t.GT, t.alleles))
            == hl.Struct(AC=[5, 5], AF=[0.5, 0.5], AN=10, homozygote_count=[0, 0])
        )  # Tests table.aggregate initOp

    @test_timeout(batch=5 * 60)
    def test_initop_matrix_table(self):
        mt = (
            hl.utils.range_matrix_table(10, 5, 5)
            .annotate_entries(GT=hl.call(0, 1))
            .annotate_rows(alleles=["A", "T"])
            .annotate_globals(alleles2=["G", "C"])
        )

        row_agg = mt.annotate_rows(call_stats=hl.agg.call_stats(mt.GT, mt.alleles)).rows()  # Tests MatrixMapRows initOp
        col_agg = mt.annotate_cols(
            call_stats=hl.agg.call_stats(mt.GT, mt.alleles2)
        ).cols()  # Tests MatrixMapCols initOp

        # must test that call_stats isn't null, because equality doesn't test for that
        self.assertTrue(
            row_agg.all(
                hl.is_defined(row_agg.call_stats)
                & (row_agg.call_stats == hl.struct(AC=[5, 5], AF=[0.5, 0.5], AN=10, homozygote_count=[0, 0]))
            )
        )
        self.assertTrue(
            col_agg.all(
                hl.is_defined(col_agg.call_stats)
                & (col_agg.call_stats == hl.struct(AC=[10, 10], AF=[0.5, 0.5], AN=20, homozygote_count=[0, 0]))
            )
        )

    @test_timeout(batch=5 * 60)
    def test_initop_table_aggregate_by_key(self):
        t = hl.utils.range_table(5, 3).annotate(GT=hl.call(0, 1)).annotate_globals(alleles=["A", "T"])
        t2 = t.annotate(group=t.idx < 3)
        group_agg = t2.group_by(t2['group']).aggregate(call_stats=hl.agg.call_stats(t2.GT, t2.alleles))

        self.assertTrue(
            group_agg.all(
                hl.if_else(
                    group_agg.group,
                    hl.is_defined(group_agg.call_stats)
                    & (group_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_agg.call_stats)
                    & (group_agg.call_stats == hl.struct(AC=[2, 2], AF=[0.5, 0.5], AN=4, homozygote_count=[0, 0])),
                )
            )
        )

    @test_timeout(batch=5 * 60)
    def test_initop_matrix_aggregate_cols_by_key_entries(self):
        mt = (
            hl.utils.range_matrix_table(10, 5, 5)
            .annotate_entries(GT=hl.call(0, 1))
            .annotate_rows(alleles=["A", "T"])
            .annotate_globals(alleles2=["G", "C"])
        )
        mt2 = mt.annotate_cols(group=mt.col_idx < 3)
        group_cols_agg = (
            mt2.group_cols_by(mt2['group']).aggregate(call_stats=hl.agg.call_stats(mt2.GT, mt2.alleles2)).entries()
        )

        self.assertTrue(
            group_cols_agg.all(
                hl.if_else(
                    group_cols_agg.group,
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[2, 2], AF=[0.5, 0.5], AN=4, homozygote_count=[0, 0])),
                )
            )
        )

    @test_timeout(batch=5 * 60)
    def test_initop_matrix_aggregate_cols_by_key_cols(self):
        mt = (
            hl.utils.range_matrix_table(10, 5, 5)
            .annotate_entries(GT=hl.call(0, 1))
            .annotate_rows(alleles=["A", "T"])
            .annotate_globals(alleles2=["G", "C"])
        )
        mt2 = mt.annotate_cols(group=mt.col_idx < 3, GT_col=hl.call(0, 1))
        group_cols_agg = (
            mt2.group_cols_by(mt2['group'])
            .aggregate_cols(call_stats=hl.agg.call_stats(mt2.GT_col, mt2.alleles2))
            .result()
        ).entries()

        self.assertTrue(
            group_cols_agg.all(
                hl.if_else(
                    group_cols_agg.group,
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_cols_agg.call_stats)
                    & (group_cols_agg.call_stats == hl.struct(AC=[2, 2], AF=[0.5, 0.5], AN=4, homozygote_count=[0, 0])),
                )
            )
        )

    @test_timeout(batch=5 * 60)
    def test_initop_matrix_aggregate_rows_by_key_entries(self):
        mt = (
            hl.utils.range_matrix_table(10, 5, 5)
            .annotate_entries(GT=hl.call(0, 1))
            .annotate_rows(alleles=["A", "T"])
            .annotate_globals(alleles2=["G", "C"])
        )
        mt2 = mt.annotate_rows(group=mt.row_idx < 3)
        group_rows_agg = (
            mt2.group_rows_by(mt2['group']).aggregate(call_stats=hl.agg.call_stats(mt2.GT, mt2.alleles2)).entries()
        )

        self.assertTrue(
            group_rows_agg.all(
                hl.if_else(
                    group_rows_agg.group,
                    hl.is_defined(group_rows_agg.call_stats)
                    & (group_rows_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_rows_agg.call_stats)
                    & (
                        group_rows_agg.call_stats == hl.struct(AC=[7, 7], AF=[0.5, 0.5], AN=14, homozygote_count=[0, 0])
                    ),
                )
            )
        )

    @test_timeout(batch=5 * 60)
    def test_initop_matrix_aggregate_rows_by_key_rows(self):
        mt = (
            hl.utils.range_matrix_table(10, 5, 5)
            .annotate_entries(GT=hl.call(0, 1))
            .annotate_rows(alleles=["A", "T"])
            .annotate_globals(alleles2=["G", "C"])
        )
        mt2 = mt.annotate_rows(group=mt.row_idx < 3, GT_row=hl.call(0, 1))
        group_rows_agg = (
            mt2.group_rows_by(mt2['group'])
            .aggregate_rows(call_stats=hl.agg.call_stats(mt2.GT_row, mt2.alleles2))
            .result()
        ).entries()

        self.assertTrue(
            group_rows_agg.all(
                hl.if_else(
                    group_rows_agg.group,
                    hl.is_defined(group_rows_agg.call_stats)
                    & (group_rows_agg.call_stats == hl.struct(AC=[3, 3], AF=[0.5, 0.5], AN=6, homozygote_count=[0, 0])),
                    hl.is_defined(group_rows_agg.call_stats)
                    & (
                        group_rows_agg.call_stats == hl.struct(AC=[7, 7], AF=[0.5, 0.5], AN=14, homozygote_count=[0, 0])
                    ),
                )
            )
        )

    def test_call_stats_init(self):
        ht = hl.utils.range_table(3)
        ht = ht.annotate(GT=hl.unphased_diploid_gt_index_call(ht.idx))
        assert ht.aggregate(hl.agg.call_stats(ht.GT, 2).AC) == [3, 3]

    def test_mendel_error_code(self):
        locus_auto = hl.Locus('2', 20000000)
        locus_x_par = hl.get_reference('default').par[0].start
        locus_x_nonpar = hl.Locus(locus_x_par.contig, locus_x_par.position - 1)
        locus_y_nonpar = hl.Locus('Y', hl.get_reference('default').lengths['Y'] - 1)

        self.assertTrue(
            hl.eval(
                hl.all(
                    lambda x: x,
                    hl.array(
                        [
                            hl.literal(locus_auto).in_autosome_or_par(),
                            hl.literal(locus_auto).in_autosome_or_par(),
                            ~hl.literal(locus_x_par).in_autosome(),
                            hl.literal(locus_x_par).in_autosome_or_par(),
                            ~hl.literal(locus_x_nonpar).in_autosome_or_par(),
                            hl.literal(locus_x_nonpar).in_x_nonpar(),
                            ~hl.literal(locus_y_nonpar).in_autosome_or_par(),
                            hl.literal(locus_y_nonpar).in_y_nonpar(),
                        ]
                    ),
                )
            )
        )

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

        arg_list = hl.literal(
            list(expected.keys()), hl.tarray(hl.ttuple(hl.tlocus(), hl.tbool, hl.tcall, hl.tcall, hl.tcall))
        )
        values = arg_list.map(lambda args: hl.mendel_error_code(*args))
        expr = hl.dict(hl.zip(arg_list, values))
        results = hl.eval(expr)
        for args, result in results.items():
            self.assertEqual(result, expected[args], msg=f'expected {expected[args]}, found {result} at {str(args)}')

    def test_min_rep(self):
        def assert_min_reps_to(old, new, pos_change=0):
            self.assertEqual(
                hl.eval(hl.min_rep(hl.locus('1', 10), old)),
                hl.Struct(locus=hl.Locus('1', 10 + pos_change), alleles=new),
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

    def test_min_rep_error(self):
        with pytest.raises(hl.utils.FatalError, match='min_rep: found null allele'):
            hl.eval(hl.min_rep(hl.locus('1', 100), ['A', hl.missing('str')]))
        with pytest.raises(hl.utils.FatalError, match='min_rep: expect at least one allele'):
            hl.eval(hl.min_rep(hl.locus('1', 100), hl.empty_array('str')))

    def assert_evals_to(self, e, v):
        assert_evals_to(e, v)

    def test_set_functions(self):
        s = hl.set([1, 3, 7])
        t = hl.set([3, 8])

        self.assert_evals_to(s, set([1, 3, 7]))
        self.assert_evals_to(hl.set(frozenset([1, 2, 3])), set([1, 2, 3]))

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
        s = hl.set([1, 3, 5, hl.missing(tint32)])
        self.assert_evals_to(hl.min(s), 1)
        self.assert_evals_to(hl.max(s), 5)
        self.assert_evals_to(hl.mean(s), 3)
        self.assert_evals_to(hl.median(s), 3)

    def test_set_operators_1(self):
        self.assert_evals_to(hl.set([1, 2, 3]) <= hl.set([1, 2]), False)
        self.assert_evals_to(hl.set([1, 2, 3]) <= hl.set([1, 2, 3]), True)
        self.assert_evals_to(hl.set([1, 2, 3]) <= hl.set([1, 2, 3, 4]), True)

    def test_set_operators_2(self):
        self.assert_evals_to(hl.set([1, 2, 3]) < hl.set([1, 2]), False)
        self.assert_evals_to(hl.set([1, 2, 3]) < hl.set([1, 2, 3]), False)
        self.assert_evals_to(hl.set([1, 2, 3]) < hl.set([1, 2, 3, 4]), True)

    def test_set_operators_3(self):
        self.assert_evals_to(hl.set([1, 2]) >= hl.set([1, 2, 3]), False)
        self.assert_evals_to(hl.set([1, 2, 3]) >= hl.set([1, 2, 3]), True)
        self.assert_evals_to(hl.set([1, 2, 3, 4]) >= hl.set([1, 2, 3]), True)

    def test_set_operators_4(self):
        self.assert_evals_to(hl.set([1, 2]) > hl.set([1, 2, 3]), False)
        self.assert_evals_to(hl.set([1, 2, 3]) > hl.set([1, 2, 3]), False)
        self.assert_evals_to(hl.set([1, 2, 3, 4]) > hl.set([1, 2, 3]), True)

    def test_set_operators_5(self):
        self.assert_evals_to(hl.set([1, 2, 3]) - hl.set([1, 3]), set([2]))
        self.assert_evals_to(hl.set([1, 2, 3]) - set([1, 3]), set([2]))
        self.assert_evals_to(set([1, 2, 3]) - hl.set([1, 3]), set([2]))

    def test_set_operators_6(self):
        self.assert_evals_to(hl.set([1, 2, 3]) | hl.set([3, 4, 5]), set([1, 2, 3, 4, 5]))
        self.assert_evals_to(hl.set([1, 2, 3]) | set([3, 4, 5]), set([1, 2, 3, 4, 5]))
        self.assert_evals_to(set([1, 2, 3]) | hl.set([3, 4, 5]), set([1, 2, 3, 4, 5]))

    def test_set_operators_7(self):
        self.assert_evals_to(hl.set([1, 2, 3]) & hl.set([3, 4, 5]), set([3]))
        self.assert_evals_to(hl.set([1, 2, 3]) & set([3, 4, 5]), set([3]))
        self.assert_evals_to(set([1, 2, 3]) & hl.set([3, 4, 5]), set([3]))

    def test_set_operators_8(self):
        self.assert_evals_to(hl.set([1, 2, 3]) ^ hl.set([3, 4, 5]), set([1, 2, 4, 5]))
        self.assert_evals_to(hl.set([1, 2, 3]) ^ set([3, 4, 5]), set([1, 2, 4, 5]))
        self.assert_evals_to(set([1, 2, 3]) ^ hl.set([3, 4, 5]), set([1, 2, 4, 5]))

    def test_uniroot_1(self):
        tol = 1.220703e-4

        self.assertAlmostEqual(hl.eval(hl.uniroot(lambda x: x - 1, 0, hl.missing('float'), tolerance=tol)), None)
        self.assertAlmostEqual(hl.eval(hl.uniroot(lambda x: x - 1, hl.missing('float'), 3, tolerance=tol)), None)

    def test_uniroot_2(self):
        tol = 1.220703e-4

        self.assertAlmostEqual(hl.eval(hl.uniroot(lambda x: x - 1, 0, 3, tolerance=tol)), 1)
        self.assertAlmostEqual(
            hl.eval(hl.uniroot(lambda x: hl.log(x) - 1, 0, 3, tolerance=tol)), 2.718281828459045, delta=tol
        )

    def test_uniroot_3(self):
        with self.assertRaisesRegex(hl.utils.FatalError, r"value of f\(x\) is missing"):
            hl.eval(hl.uniroot(lambda x: hl.missing('float'), 0, 1))
        with self.assertRaisesRegex(hl.utils.HailUserError, 'opposite signs'):
            hl.eval(hl.uniroot(lambda x: x**2 - 0.5, -1, 1))
        with self.assertRaisesRegex(hl.utils.HailUserError, 'min must be less than max'):
            hl.eval(hl.uniroot(lambda x: x, 1, -1))

    def test_uniroot_multiple_roots(self):
        tol = 1.220703e-4

        def multiple_roots(x):
            return (x - 1.5) * (x - 2) * (x - 3.3) * (x - 4.5) * (x - 5)

        roots = [1.5, 2, 3.3, 4.5, 5]
        result = hl.eval(hl.uniroot(multiple_roots, 0, 5.5, tolerance=tol))
        self.assertTrue(any(abs(result - root) < tol for root in roots))

    def test_dnorm(self):
        self.assert_evals_to(hl.dnorm(0), 0.3989422804014327)
        self.assert_evals_to(hl.dnorm(0, mu=1, sigma=2), 0.17603266338214976)
        self.assert_evals_to(hl.dnorm(0, log_p=True), -0.9189385332046728)

    def test_pnorm(self):
        self.assert_evals_to(hl.pnorm(0), 0.5)
        self.assert_evals_to(hl.pnorm(1, mu=1, sigma=2), 0.5)
        self.assert_evals_to(hl.pnorm(1), 0.8413447460685429)
        self.assert_evals_to(hl.pnorm(1, lower_tail=False), 0.15865525393145705)
        self.assert_evals_to(hl.pnorm(1, log_p=True), -0.17275377902344988)

    def test_qnorm(self):
        self.assert_evals_to(hl.qnorm(hl.pnorm(0)), 0.0)
        self.assert_evals_to(hl.qnorm(hl.pnorm(1, mu=1, sigma=2), mu=1, sigma=2), 1.0)
        self.assert_evals_to(hl.qnorm(hl.pnorm(1)), 1.0)
        self.assert_evals_to(hl.qnorm(hl.pnorm(1, lower_tail=False), lower_tail=False), 1.0)
        self.assert_evals_to(hl.qnorm(hl.pnorm(1, log_p=True), log_p=True), 1.0)

    def test_dchisq(self):
        self.assert_evals_to(hl.dchisq(10, 5), 0.028334555341734464)
        self.assert_evals_to(hl.dchisq(10, 5, ncp=5), 0.07053548900555977)
        self.assert_evals_to(hl.dchisq(10, 5, log_p=True), -3.5636731823817143)

    def test_pchisqtail(self):
        self.assert_evals_to(hl.pchisqtail(10, 5), 0.07523524614651216)
        self.assert_evals_to(hl.pchisqtail(10, 5, ncp=2), 0.20772889456608998)
        self.assert_evals_to(hl.pchisqtail(10, 5, lower_tail=True), 0.9247647538534879)
        self.assert_evals_to(hl.pchisqtail(10, 5, log_p=True), -2.5871354590744855)

    def test_qchisqtail(self):
        self.assertAlmostEqual(hl.eval(hl.qchisqtail(hl.pchisqtail(10, 5), 5)), 10.0)
        self.assertAlmostEqual(hl.eval(hl.qchisqtail(hl.pchisqtail(10, 5, ncp=2), 5, ncp=2)), 10.0)
        self.assertAlmostEqual(hl.eval(hl.qchisqtail(hl.pchisqtail(10, 5, lower_tail=True), 5, lower_tail=True)), 10.0)
        self.assertAlmostEqual(hl.eval(hl.qchisqtail(hl.pchisqtail(10, 5, log_p=True), 5, log_p=True)), 10.0)

    def test_pT(self):
        self.assert_evals_to(hl.pT(0, 10), 0.5)
        self.assert_evals_to(hl.pT(1, 10), 0.82955343384897)
        self.assert_evals_to(hl.pT(1, 10, lower_tail=False), 0.17044656615103004)
        self.assert_evals_to(hl.pT(1, 10, log_p=True), -0.186867754489647)

    def test_pF(self):
        self.assert_evals_to(hl.pF(0, 3, 10), 0.0)
        self.assert_evals_to(hl.pF(1, 3, 10), 0.5676627969783028)
        self.assert_evals_to(hl.pF(1, 3, 10, lower_tail=False), 0.4323372030216972)
        self.assert_evals_to(hl.pF(1, 3, 10, log_p=True), -0.566227703842908)

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
        two_sided_res = hl.eval(hl.hardy_weinberg_test(1, 2, 1, one_sided=False))
        self.assertAlmostEqual(two_sided_res['p_value'], 0.65714285)
        self.assertAlmostEqual(two_sided_res['het_freq_hwe'], 0.57142857)

        one_sided_res = hl.eval(hl.hardy_weinberg_test(1, 2, 1, one_sided=True))
        self.assertAlmostEqual(one_sided_res['p_value'], 0.57142857)
        self.assertAlmostEqual(one_sided_res['het_freq_hwe'], 0.57142857)

    def test_hardy_weinberg_agg_1(self):
        row_idx_col_idx_to_call = {
            (0, 0): hl.call(0, 0),
            (0, 1): hl.call(0),
            (0, 2): hl.call(1, 1),
            (0, 3): hl.call(0, 1),
            (0, 4): hl.call(0, 1),
            (1, 0): hl.call(0, 0),
            (1, 1): hl.call(0, 0),
            (1, 2): hl.call(0, 0),
            (1, 3): hl.call(0, 0),
            (1, 4): hl.call(0, 0),
        }

        mt = hl.utils.range_matrix_table(n_rows=3, n_cols=5)
        mt = mt.annotate_rows(
            hwe_two_sided=hl.agg.hardy_weinberg_test(
                hl.literal(row_idx_col_idx_to_call).get((mt.row_idx, mt.col_idx)), one_sided=False
            ),
            hwe_one_sided=hl.agg.hardy_weinberg_test(
                hl.literal(row_idx_col_idx_to_call).get((mt.row_idx, mt.col_idx)), one_sided=True
            ),
        )
        rows = mt.rows().collect()
        all_hwe_one_sided = [r.hwe_one_sided for r in rows]
        all_hwe_two_sided = [r.hwe_two_sided for r in rows]

        [r1_two_sided, r2_two_sided, r3_two_sided] = all_hwe_two_sided

        self.assertAlmostEqual(r1_two_sided['p_value'], 0.65714285)
        self.assertAlmostEqual(r1_two_sided['het_freq_hwe'], 0.57142857)

        assert r2_two_sided['p_value'] == 0.5
        assert r2_two_sided['het_freq_hwe'] == 0.0

        assert r3_two_sided['p_value'] == 0.5
        assert np.isnan(r3_two_sided['het_freq_hwe'])

        [r1_one_sided, r2_one_sided, r3_one_sided] = all_hwe_one_sided

        self.assertAlmostEqual(r1_one_sided['p_value'], 0.57142857)
        self.assertAlmostEqual(r1_one_sided['het_freq_hwe'], 0.57142857)

        assert r2_one_sided['p_value'] == 0.5
        assert r2_one_sided['het_freq_hwe'] == 0.0

        assert r3_one_sided['p_value'] == 0.5
        assert np.isnan(r3_one_sided['het_freq_hwe'])

    def test_hardy_weinberg_agg_2(self):
        calls = [
            hl.call(0, 0),
            hl.call(0),
            hl.call(1, 1),
            hl.call(0, 1),
            hl.call(0, 1),
        ]

        ht = hl.utils.range_table(6)
        ht = ht.annotate(
            x_two_sided=hl.scan.hardy_weinberg_test(hl.literal(calls)[ht.idx % 5], one_sided=False),
            x_one_sided=hl.scan.hardy_weinberg_test(hl.literal(calls)[ht.idx % 5], one_sided=True),
        )
        rows = ht.collect()
        all_x_one_sided = [r.x_one_sided for r in rows]
        all_x_two_sided = [r.x_two_sided for r in rows]

        [first_two_sided, *mid_two_sided, penultimate_two_sided, last_two_sided] = all_x_two_sided

        assert first_two_sided['p_value'] == 0.5
        assert np.isnan(first_two_sided['het_freq_hwe'])

        self.assertAlmostEqual(penultimate_two_sided['p_value'], 0.7)
        self.assertAlmostEqual(penultimate_two_sided['het_freq_hwe'], 0.6)

        self.assertAlmostEqual(last_two_sided['p_value'], 0.65714285)
        self.assertAlmostEqual(last_two_sided['het_freq_hwe'], 0.57142857)

        [first_one_sided, *mid_one_sided, penultimate_one_sided, last_one_sided] = all_x_one_sided

        assert first_one_sided['p_value'] == 0.5
        assert np.isnan(first_one_sided['het_freq_hwe'])

        self.assertAlmostEqual(penultimate_one_sided['p_value'], 0.7)
        self.assertAlmostEqual(penultimate_one_sided['het_freq_hwe'], 0.6)

        self.assertAlmostEqual(last_one_sided['p_value'], 0.57142857)
        self.assertAlmostEqual(last_one_sided['het_freq_hwe'], 0.57142857)

    def test_inbreeding_aggregator(self):
        data = [
            (0.25, hl.Call([0, 0])),
            (0.5, hl.Call([1, 1])),
            (0.5, hl.Call([1, 1])),
            (0.5, hl.Call([1, 1])),
            (0.0, hl.Call([0, 0])),
            (0.5, hl.Call([0, 1])),
            (0.99, hl.Call([0, 1])),
            (0.99, None),
            (None, hl.Call([0, 1])),
            (None, None),
        ]
        lit = hl.literal(data, dtype='array<tuple(float, call)>')
        ht = hl.utils.range_table(len(data))
        r = ht.aggregate(hl.agg.inbreeding(lit[ht.idx][1], lit[ht.idx][0]))

        expected_homs = sum(1 - (2 * af * (1 - af)) for af, x in data if af is not None and x is not None)
        self.assertAlmostEqual(r['expected_homs'], expected_homs)
        assert r['observed_homs'] == 5
        assert r['n_called'] == 7
        self.assertAlmostEqual(r['f_stat'], (5 - expected_homs) / (7 - expected_homs))

    def test_pl_to_gp(self):
        res = hl.eval(hl.pl_to_gp([0, 10, 100]))
        self.assertAlmostEqual(res[0], 0.9090909090082644)
        self.assertAlmostEqual(res[1], 0.09090909090082644)
        self.assertAlmostEqual(res[2], 9.090909090082645e-11)

    def test_pl_dosage(self):
        self.assertAlmostEqual(hl.eval(hl.pl_dosage([0, 20, 100])), 0.009900990296049406)
        self.assertAlmostEqual(hl.eval(hl.pl_dosage([20, 0, 100])), 0.9900990100009803)
        self.assertAlmostEqual(hl.eval(hl.pl_dosage([20, 100, 0])), 1.980198019704931)
        self.assertIsNone(hl.eval(hl.pl_dosage([20, hl.missing('int'), 100])))

    def test_collection_method_missingness(self):
        a = [1, hl.missing('int')]

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
        self.assertEqual(hl.eval(hl.literal(hl.set(['A', 'B']))), {'A', 'B'})
        self.assertEqual(hl.eval(hl.literal({hl.str('A'), hl.str('B')})), {'A', 'B'})

    def test_format(self):
        self.assertEqual(hl.eval(hl.format("%.4f %s %.3e", 0.25, 'hello', 0.114)), '0.2500 hello 1.140e-01')
        self.assertEqual(hl.eval(hl.format("%.4f %d", hl.missing(hl.tint32), hl.missing(hl.tint32))), 'null null')
        self.assertEqual(
            hl.eval(hl.format("%s", hl.struct(foo=5, bar=True, baz=hl.array([4, 5])))),
            '{foo: 5, bar: true, baz: [4,5]}',
        )
        self.assertEqual(
            hl.eval(hl.format("%s %s", hl.locus("1", 356), hl.tuple([9, True, hl.missing(hl.tstr)]))),
            '1:356 (9, true, null)',
        )
        self.assertEqual(
            hl.eval(hl.format("%b %B %b %b", hl.missing(hl.tint), hl.missing(hl.tstr), True, "hello")),
            "false FALSE true true",
        )

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

    def test_dict_keyed_by_set(self):
        dict_with_set_key = hl.dict({hl.set([1, 2, 3]): 4})
        # Test that it's evalable, since python sets aren't hashable.
        assert hl.eval(dict_with_set_key) == {frozenset([1, 2, 3]): 4}

    def test_dict_keyed_by_dict(self):
        dict_with_dict_key = hl.dict({hl.dict({1: 2, 3: 5}): 4})
        # Test that it's evalable, since python dicts aren't hashable.
        assert hl.eval(dict_with_dict_key) == {hl.utils.frozendict({1: 2, 3: 5}): 4}

    def test_frozendict_as_literal(self):
        fd = hl.utils.frozendict({"a": 4, "b": 8})
        assert hl.eval(hl.literal(fd)) == hl.utils.frozendict({"a": 4, "b": 8})

    def test_literal_with_empty_struct_key(self):
        original = {hl.Struct(): 4}
        assert hl.eval(hl.literal(original)) == original

    def test_nan_roundtrip(self):
        a = [math.nan, math.inf, -math.inf, 0, 1]
        round_trip = hl.eval(hl.literal(a))
        self.assertTrue(math.isnan(round_trip[0]))
        self.assertTrue(math.isinf(round_trip[1]))
        self.assertTrue(math.isinf(round_trip[2]))
        self.assertEqual(round_trip[-2:], [0, 1])

    def test_approx_equal(self):
        self.assertTrue(hl.eval(hl.approx_equal(0.25, 0.25000001)))
        self.assertTrue(hl.eval(hl.approx_equal(hl.missing(hl.tint64), 5)) is None)
        self.assertFalse(hl.eval(hl.approx_equal(0.25, 0.251, absolute=True, tolerance=1e-3)))

    def test_issue3729(self):
        t = hl.utils.range_table(10, 3)
        fold_expr = hl.if_else(t.idx == 3, [1, 2, 3], [4, 5, 6]).fold(lambda accum, i: accum & (i == t.idx), True)
        t.annotate(foo=hl.if_else(fold_expr, 1, 3))._force_count()

    def assertValueEqual(self, expr, value, t):
        self.assertEqual(expr.dtype, t)
        self.assertEqual(hl.eval(expr), value)

    def test_array_fold_and_scan(self):
        self.assertValueEqual(hl.fold(lambda x, y: x + y, 0, [1, 2, 3]), 6, tint32)
        self.assertValueEqual(hl.array_scan(lambda x, y: x + y, 0, [1, 2, 3]), [0, 1, 3, 6], tarray(tint32))

        self.assertValueEqual(hl.fold(lambda x, y: x + y, 0.0, [1, 2, 3]), 6.0, tfloat64)
        self.assertValueEqual(hl.fold(lambda x, y: x + y, 0, [1.0, 2.0, 3.0]), 6.0, tfloat64)
        self.assertValueEqual(hl.array_scan(lambda x, y: x + y, 0.0, [1, 2, 3]), [0.0, 1.0, 3.0, 6.0], tarray(tfloat64))
        self.assertValueEqual(
            hl.array_scan(lambda x, y: x + y, 0, [1.0, 2.0, 3.0]), [0.0, 1.0, 3.0, 6.0], tarray(tfloat64)
        )

    def test_cumulative_sum(self):
        self.assertValueEqual(hl.cumulative_sum([1, 2, 3, 4]), [1, 3, 6, 10], tarray(tint32))
        self.assertValueEqual(hl.cumulative_sum([1.0, 2.0, 3.0, 4.0]), [1.0, 3.0, 6.0, 10.0], tarray(tfloat64))

    def test_nan_inf_checks(self):
        finite = 0
        infinite = float('inf')
        nan = math.nan
        na = hl.missing('float64')

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
        hl.array([mt.AD, [1, 2]]).show()

    def test_string_unicode(self):
        self.assertTrue(hl.eval(hl.str("李") == "李"))

    def test_reversed(self):
        a = hl.array(['a', 'b', 'c'])
        ea = hl.empty_array(hl.tint)
        na = hl.missing(hl.tarray(hl.tbool))

        assert hl.eval(hl.reversed(a)) == ['c', 'b', 'a']
        assert hl.eval(hl.reversed(ea)) == []
        assert hl.eval(hl.reversed(na)) is None

        s = hl.str('abc')
        es = ''
        ns = hl.missing(hl.tstr)

        assert hl.eval(hl.reversed(s)) == 'cba'
        assert hl.eval(hl.reversed(es)) == ''
        assert hl.eval(hl.reversed(ns)) is None

    def test_bit_ops_types(self):
        assert hl.bit_and(1, 1).dtype == hl.tint32
        assert hl.bit_and(hl.int64(1), 1).dtype == hl.tint64

        assert hl.bit_or(1, 1).dtype == hl.tint32
        assert hl.bit_or(hl.int64(1), 1).dtype == hl.tint64

        assert hl.bit_xor(1, 1).dtype == hl.tint32
        assert hl.bit_xor(hl.int64(1), 1).dtype == hl.tint64

        assert hl.bit_lshift(1, 1).dtype == hl.tint32
        assert hl.bit_lshift(hl.int64(1), 1).dtype == hl.tint64

        assert hl.bit_rshift(1, 1).dtype == hl.tint32
        assert hl.bit_rshift(hl.int64(1), 1).dtype == hl.tint64

        assert hl.bit_not(1).dtype == hl.tint32
        assert hl.bit_not(hl.int64(1)).dtype == hl.tint64

        assert hl.bit_count(1).dtype == hl.tint32
        assert hl.bit_count(hl.int64(1)).dtype == hl.tint32

    def test_bit_shifts(self):
        assert hl.eval(hl.bit_lshift(hl.int(8), 2)) == 32
        assert hl.eval(hl.bit_rshift(hl.int(8), 2)) == 2
        assert hl.eval(hl.bit_lshift(hl.int(8), 0)) == 8

        assert hl.eval(hl.bit_lshift(hl.int64(8), 2)) == 32
        assert hl.eval(hl.bit_rshift(hl.int64(8), 2)) == 2
        assert hl.eval(hl.bit_lshift(hl.int64(8), 0)) == 8

    def test_bit_shift_edge_cases(self):
        assert hl.eval(hl.bit_lshift(hl.int(1), 32)) == 0
        assert hl.eval(hl.bit_rshift(hl.int(1), 32)) == 1
        assert hl.eval(hl.bit_rshift(hl.int(1), 32, logical=True)) == 0
        assert hl.eval(hl.bit_rshift(hl.int(-1), 32)) == -1
        assert hl.eval(hl.bit_rshift(hl.int(-1), 32, logical=True)) == 0

        assert hl.eval(hl.bit_lshift(hl.int64(1), 64)) == 0
        assert hl.eval(hl.bit_rshift(hl.int64(1), 64)) == 1
        assert hl.eval(hl.bit_rshift(hl.int64(1), 64, logical=True)) == 0
        assert hl.eval(hl.bit_rshift(hl.int64(-1), 64)) == -1
        assert hl.eval(hl.bit_rshift(hl.int64(-11), 64, logical=True)) == 0

    def test_bit_shift_errors(self):
        with pytest.raises(hl.utils.HailUserError):
            hl.eval(hl.bit_lshift(1, -1))

        with pytest.raises(hl.utils.HailUserError):
            hl.eval(hl.bit_rshift(1, -1))

        with pytest.raises(hl.utils.HailUserError):
            hl.eval(hl.bit_rshift(1, -1, logical=True))

        with pytest.raises(hl.utils.HailUserError):
            hl.eval(hl.bit_lshift(hl.int64(1), -1))

        with pytest.raises(hl.utils.HailUserError):
            hl.eval(hl.bit_rshift(hl.int64(1), -1))

        with pytest.raises(hl.utils.HailUserError):
            hl.eval(hl.bit_rshift(hl.int64(1), -1, logical=True))

    def test_prev_non_null(self):
        ht = hl.utils.range_table(1)

        assert ht.aggregate((hl.agg._prev_nonnull(ht.idx))) == 0

    @test_timeout(batch=5 * 60)
    def test_summarize_runs(self):
        mt = hl.utils.range_matrix_table(3, 3).annotate_entries(
            x1='a',
            x2=1,
            x3=1.5,
            x4=True,
            x5=['1'],
            x6={'1'},
            x7={'1': 5},
            x8=hl.struct(a=5, b='7'),
            x9=(1, 2, 3),
            x10=hl.locus('1', 123123),
            x11=hl.call(0, 1, phased=True),
        )

        mt.summarize()
        mt.entries().summarize()
        mt.x1.summarize()

    def test_variant_str(self):
        assert (
            hl.eval(hl.variant_str(hl.struct(locus=hl.locus('1', 10000), alleles=['A', 'T', 'CCC'])))
            == '1:10000:A:T,CCC'
        )
        assert hl.eval(hl.variant_str(hl.locus('1', 10000), ['A', 'T', 'CCC'])) == '1:10000:A:T,CCC'
        with pytest.raises(ValueError):
            hl.variant_str()

    def test_collection_getitem(self):
        collection_types = [(hl.array, list), (hl.set, frozenset)]
        for (htyp, pytyp) in collection_types:
            x = htyp([hl.struct(a='foo', b=3), hl.struct(a='bar', b=4)])
            assert hl.eval(x.a) == pytyp(['foo', 'bar'])

        a = hl.array([hl.struct(b=[hl.struct(inner=1), hl.struct(inner=2)]), hl.struct(b=[hl.struct(inner=3)])])
        assert hl.eval(a.b) == [[hl.Struct(inner=1), hl.Struct(inner=2)], [hl.Struct(inner=3)]]
        assert hl.eval(hl.flatten(a.b).inner) == [1, 2, 3]
        assert hl.eval(a.b.inner) == [[1, 2], [3]]
        assert hl.eval(a["b"].inner) == [[1, 2], [3]]
        assert hl.eval(a["b"]["inner"]) == [[1, 2], [3]]
        assert hl.eval(a.b["inner"]) == [[1, 2], [3]]

    def test_struct_collection_getattr(self):
        collection_types = [hl.array, hl.set]
        for htyp in collection_types:
            a = htyp([hl.struct(x='foo'), hl.struct(x='bar')])

            assert hasattr(a, 'x') == True
            assert hasattr(a, 'y') == False

            with pytest.raises(AttributeError, match="has no field"):
                getattr(a, 'y')

    def test_binary_search(self):
        a = hl.array([0, 2, 4, 8])
        values = [-1, 0, 1, 2, 3, 4, 10, hl.missing('int32')]
        expected = [0, 0, 1, 1, 2, 2, 4, None]
        assert hl.eval(hl.map(lambda x: hl.binary_search(a, x), values)) == expected

    def verify_6930_still_holds(self):
        rmt33 = hl.utils.range_matrix_table(3, 3, n_partitions=2)

        mt = rmt33.choose_cols([1, 2, 0])
        assert mt.col.collect() == [hl.Struct(col_idx=x) for x in [1, 2, 0]]

        mt = rmt33.key_rows_by(rowkey=-mt.row_idx)
        assert mt.row.collect() == [hl.Struct(row_idx=x) for x in [2, 1, 0]]

        mt = rmt33.annotate_entries(x=(rmt33.row_idx + 1) * (rmt33.col_idx + 1))
        mt = mt.key_rows_by(rowkey=-mt.row_idx)
        mt = mt.choose_cols([2, 1, 0])
        assert mt.x.collect() == [9, 6, 3, 6, 4, 2, 3, 2, 1]

        t = hl.utils.range_table(3)
        t = t.key_by(-t.idx)
        assert t.idx.collect() == [2, 1, 0]

    def test_struct_slice(self):
        assert hl.eval(hl.struct(x=3, y=4, z=5, a=10)[1:]) == hl.Struct(y=4, z=5, a=10)
        assert hl.eval(hl.struct(x=3, y=4, z=5, a=10)[0:4:2]) == hl.Struct(x=3, z=5)
        assert hl.eval(hl.struct(x=3, y=4, z=5, a=10)[-2:]) == hl.Struct(z=5, a=10)

    def test_tuple_slice(self):
        assert hl.eval(hl.tuple((3, 4, 5, 10))[1:]) == (4, 5, 10)
        assert hl.eval(hl.tuple((3, 4, 5, 10))[0:4:2]) == (3, 5)
        assert hl.eval(hl.tuple((3, 4, 5, 10))[-2:]) == (5, 10)

    def test_numpy_conversions(self):
        assert hl.eval(np.int32(3)) == 3
        assert hl.eval(np.int64(1234)) == 1234
        assert hl.eval(np.bool_(True))
        assert not hl.eval(np.bool_(False))
        assert np.allclose(hl.eval(np.float32(3.4)), 3.4)
        assert np.allclose(hl.eval(np.float64(8.89)), 8.89)
        assert hl.eval(np.str_("cat")) == "cat"

    def test_array_struct_error(self):
        a = hl.array([hl.struct(a=5)])
        with pytest.raises(AttributeError, match='ArrayStructExpression instance has no field, method, or property'):
            a.x

    def test_parse_json(self):
        values = [
            hl.missing('int32'),
            hl.missing('str'),
            hl.missing('struct{a:int32,b:str}'),
            hl.locus('1', 10000),
            hl.set({'x', 'y'}),
            hl.array([1, 2, hl.missing('int32')]),
            hl.call(0, 2, phased=True),
            hl.locus_interval('1', 10000, 10005),
            hl.struct(foo='bar'),
            hl.tuple([1, 2, 'str']),
        ]
        assert hl.eval(hl._compare(hl.tuple(values), hl.tuple(hl.parse_json(hl.json(v), v.dtype) for v in values)) == 0)

    def test_expr_persist(self):
        # need to test laziness, so we will overwrite a file
        with hl.TemporaryDirectory(ensure_exists=False) as f:
            hl.utils.range_table(10).write(f, overwrite=True)
            ht = hl.read_table(f)
            count1 = ht.aggregate(hl.agg.count(), _localize=False)._persist()
            assert hl.eval(count1) == 10

            hl.utils.range_table(100).write(f, overwrite=True)
            assert hl.eval(count1) == 10

    def test_struct_expression_expr_rename(self):
        s = hl.struct(f1=1, f2=2, f3=3)

        assert hl.eval(s.rename({'f1': 'foo'})) == hl.Struct(f2=2, f3=3, foo=1)
        assert hl.eval(s.rename({'f3': 'fiddle', 'f1': 'hello'})) == hl.Struct(f2=2, fiddle=3, hello=1)
        assert hl.eval(s.rename({'f3': 'fiddle', 'f1': 'hello', 'f2': 'ohai'})) == hl.Struct(fiddle=3, hello=1, ohai=2)
        assert hl.eval(s.rename({'f3': 'fiddle', 'f1': 'hello', 'f2': 's p a c e'})) == hl.Struct(
            fiddle=3, hello=1, **{'s p a c e': 2}
        )

        try:
            hl.eval(s.rename({'f1': 'f2'}))
        except ValueError as err:
            assert 'already in the struct' in err.args[0]
        else:
            assert False

        try:
            hl.eval(s.rename({'f4': 'f2'}))
        except ValueError as err:
            assert 'f4 is not a field of this struct' in err.args[0]
        else:
            assert False

        try:
            hl.eval(s.rename({'f1': 'f5', 'f2': 'f5'}))
        except ValueError as err:
            assert 'f5 is the new name of both' in err.args[0]
        else:
            assert False

    def test_enumerate(self):
        a1 = hl.literal(['foo', 'bar', 'baz'], 'array<str>')
        a_empty = hl.literal([], 'array<str>')

        exprs = (
            hl.enumerate(a1),
            hl.enumerate(a1, start=-1000),
            hl.enumerate(a1, start=10, index_first=False),
            hl.enumerate(a_empty, start=5),
        )
        assert hl.eval(exprs) == (
            [(0, 'foo'), (1, 'bar'), (2, 'baz')],
            [(-1000, 'foo'), (-999, 'bar'), (-998, 'baz')],
            [('foo', 10), ('bar', 11), ('baz', 12)],
            [],
        )

    def test_split_line(self):
        s1 = '1 2 3 4 5 6 7'
        s2 = '1 2 "3 4" "a b c d"'
        s3 = '"1" "2"'

        assert hl.eval(hl.str(s1)._split_line(' ', ['NA'], quote=None, regex=False)) == s1.split(' ')
        assert hl.eval(hl.str(s1)._split_line(r'\s+', ['NA'], quote=None, regex=True)) == s1.split(' ')
        assert hl.eval(hl.str(s3)._split_line(' ', ['1'], quote='"', regex=False)) == [None, '2']
        assert hl.eval(hl.str(s2)._split_line(' ', ['1', '2'], quote='"', regex=False)) == [
            None,
            None,
            '3 4',
            'a b c d',
        ]
        assert hl.eval(hl.str(s2)._split_line(r'\s+', ['1', '2'], quote='"', regex=True)) == [
            None,
            None,
            '3 4',
            'a b c d',
        ]


def test_approx_cdf():
    table = hl.utils.range_table(100)
    table = table.annotate(i=table.idx)
    table.aggregate(hl.agg.approx_cdf(table.i))
    table.aggregate(hl.agg.approx_cdf(hl.int64(table.i)))
    table.aggregate(hl.agg.approx_cdf(hl.float32(table.i)))
    table.aggregate(hl.agg.approx_cdf(hl.float64(table.i)))


# assumes cdf was computed from a (possibly shuffled) range table
def cdf_max_observed_error(cdf):
    rank_error = max(
        max(abs(cdf['values'][i + 1] - cdf.ranks[i + 1]), abs(cdf['values'][i] + 1 - cdf.ranks[i + 1]))
        for i in range(len(cdf['values']) - 1)
    )
    return rank_error / cdf.ranks[-1]


@pytest.fixture(scope='module')
def cdf_test_data():
    with hl.TemporaryDirectory(ensure_exists=False) as f:
        t = hl.utils.range_table(1_000_000)
        t = t.annotate(x=hl.rand_int64())
        t.key_by(t.x).write(f, overwrite=True)
        t = hl.read_table(f)
        print('generating')
        yield t
        print('deleting')


def test_approx_cdf_accuracy(cdf_test_data):
    t = cdf_test_data
    cdf = t.aggregate(hl.agg.approx_cdf(t.idx, 200))
    error = cdf_max_observed_error(cdf)
    assert error < 0.015


def test_approx_cdf_all_missing():
    table = hl.utils.range_table(10).annotate(foo=hl.missing(tint))
    table.aggregate(hl.agg.approx_quantiles(table.foo, qs=[0.5]))


def test_approx_cdf_col_aggregate():
    mt = hl.utils.range_matrix_table(10, 10)
    mt = mt.annotate_entries(foo=mt.row_idx + mt.col_idx)
    mt = mt.annotate_cols(bar=hl.agg.approx_cdf(mt.foo))
    mt.cols()._force_count()


def test_approx_quantiles():
    table = hl.utils.range_table(100)
    table = table.annotate(i=table.idx)
    table.aggregate(hl.agg.approx_quantiles(table.i, hl.float32(0.5)))
    table.aggregate(hl.agg.approx_median(table.i))
    table.aggregate(hl.agg.approx_quantiles(table.i, [0.0, 0.1, 0.5, 0.9, 1.0]))


def test_error_from_cdf():
    table = hl.utils.range_table(100)
    table = table.annotate(i=table.idx)
    cdf = hl.agg.approx_cdf(table.i)
    table.aggregate(_error_from_cdf(cdf, 0.001))
    table.aggregate(_error_from_cdf(cdf, 0.001, all_quantiles=True))


def test_cdf_combine(cdf_test_data):
    t = cdf_test_data
    t1 = t.filter(t.x < 0)
    cdf1 = t1.aggregate(hl.agg.approx_cdf(t1.idx, 200, _raw=True), _localize=False)
    t2 = t.filter(t.x >= 0)
    cdf2 = t2.aggregate(hl.agg.approx_cdf(t2.idx, 200, _raw=True), _localize=False)
    cdf = _cdf_combine(200, cdf1, cdf2)
    cdf = hl.eval(_result_from_raw_cdf(cdf))
    error = cdf_max_observed_error(cdf)
    assert error < 0.015


def test_approx_cdf_array_agg():
    mt = hl.utils.range_matrix_table(5, 5)
    mt = mt.annotate_entries(x=mt.col_idx)
    mt = mt.group_cols_by(mt.col_idx).aggregate(cdf=hl.agg.approx_cdf(mt.x))
    mt._force_count_rows()


@pytest.mark.parametrize("delimiter", ['\t', ',', '@'])
@pytest.mark.parametrize("missing", ['NA', 'null'])
@pytest.mark.parametrize("header", [True, False])
@test_timeout(local=6 * 60, batch=6 * 60)
def test_export_entry(delimiter, missing, header):
    mt = hl.utils.range_matrix_table(3, 3)
    mt = mt.key_cols_by(col_idx=mt.col_idx + 1)
    mt = mt.annotate_entries(x=mt.row_idx * mt.col_idx)
    mt = mt.annotate_entries(x=hl.or_missing(mt.x != 4, mt.x))
    with hl.TemporaryFilename() as f:
        mt.x.export(f, delimiter=delimiter, header=header, missing=missing)
        if header:
            actual = hl.import_matrix_table(
                f, row_fields={'row_idx': hl.tint32}, row_key=['row_idx'], sep=delimiter, missing=missing
            )
        else:
            actual = hl.import_matrix_table(
                f, row_fields={'f0': hl.tint32}, row_key=['f0'], sep=delimiter, no_header=True, missing=missing
            )
            actual = actual.rename({'f0': 'row_idx'})
        actual = actual.key_cols_by(col_idx=hl.int(actual.col_id))
        actual = actual.drop('col_id')
        if not header:
            actual = actual.key_cols_by(col_idx=actual.col_idx + 1)
        assert mt._same(actual)

        expected_collect = [0, 0, 0, 1, 2, 3, 2, None, 6]
        assert expected_collect == actual.x.collect()


def test_stream_randomness():
    def assert_contains_node(expr, node):
        assert expr._ir.base_search(lambda x: isinstance(x, node))

    def assert_unique_uids(a):
        n1 = hl.eval(a.to_array().length())
        n2 = len(hl.eval(hl.set(a.map(lambda x: hl.rand_int64()).to_array())))
        assert n1 == n2

    # test NA
    a = hl.missing('array<int32>')
    a = a.map(lambda x: x + hl.rand_int32(10))
    assert_contains_node(a, ir.NA)
    assert hl.eval(a) == None

    # test If
    a1 = hl._stream_range(0, 5)
    a2 = hl._stream_range(2, 20)
    a = hl.if_else(False, a1, a2)
    assert_contains_node(a, ir.If)
    assert_unique_uids(a)

    # test StreamIota
    s = hl._stream_range(10).zip_with_index(0)
    assert_contains_node(s, ir.StreamIota)
    assert_unique_uids(s)

    # test ToArray
    a = hl._stream_range(10)
    a = a.map(lambda x: hl.rand_int64()).to_array()
    assert_contains_node(a, ir.ToArray)
    assert len(set(hl.eval(a))) == 10

    # test ToStream
    t = hl.rbind(hl.range(10), lambda a: (a, a.map(lambda x: hl.rand_int64())))
    assert_contains_node(t, ir.ToStream)
    (a, r) = hl.eval(t)
    assert len(set(r)) == len(a)

    # test StreamZip
    a1 = hl._stream_range(10)
    a2 = hl._stream_range(15)
    a = hl._zip_streams(a1, a2, fill_missing=True)
    assert_contains_node(a, ir.StreamZip)
    assert_unique_uids(a)
    a = hl._zip_streams(a1, a2, fill_missing=False)
    assert_contains_node(a, ir.StreamZip)
    assert_unique_uids(a)

    # test StreamFilter
    a = hl._stream_range(15).filter(lambda x: x % 3 != 0)
    assert_contains_node(a, ir.StreamFilter)
    assert_unique_uids(a)

    # test StreamFilter
    a = hl._stream_range(5).flatmap(lambda x: hl._stream_range(x))
    assert_contains_node(a, ir.StreamFlatMap)
    assert_unique_uids(a)

    # test StreamFold
    a = hl._stream_range(10)
    a = a.fold(lambda acc, x: acc.append(hl.rand_int64()), hl.empty_array(hl.tint64))
    assert_contains_node(a, ir.StreamFold)
    assert len(set(hl.eval(a))) == 10

    # test StreamScan
    a = hl._stream_range(5)
    a = a.scan(lambda acc, x: acc.append(hl.rand_int64()), hl.empty_array(hl.tint64))
    assert_contains_node(a, ir.StreamScan)
    assert len(set(hl.eval(a.to_array())[-1])) == 5

    # test StreamAgg
    a = hl._stream_range(10)
    a = a.aggregate(lambda x: hl.agg.collect(hl.rand_int64()))
    assert_contains_node(a, ir.StreamAgg)
    assert len(set(hl.eval(a))) == 10
    a = hl._stream_range(10)
    a = a.map(lambda x: hl._stream_range(10).aggregate(lambda y: hl.agg.count() + hl.rand_int64()))
    assert_contains_node(a, ir.StreamAgg)

    # test AggExplode
    t = hl.utils.range_table(5)
    t = t.annotate(a=hl.range(t.idx))
    a = hl.agg.explode(lambda x: hl.agg.collect_as_set(hl.rand_int64()), t.a)
    assert_contains_node(a, ir.AggExplode)
    assert len(t.aggregate(a)) == 10

    # test TableCount
    t = hl.utils.range_table(10)
    t = t.annotate(x=hl.rand_int64())
    assert t.count() == 10

    # test TableGetGlobals
    t = hl.utils.range_table(10)
    t = t.annotate(x=hl.rand_int64())
    g = t.index_globals()
    assert_contains_node(g, ir.TableGetGlobals)
    assert len(hl.eval(g)) == 0

    # test TableCollect
    t = hl.utils.range_table(10)
    t = t.annotate(x=hl.rand_int64())
    a = t.collect()
    assert len(set(a)) == 10

    # test TableAggregate
    t = hl.utils.range_table(10)
    a = t.aggregate(hl.agg.collect(hl.rand_int64()).map(lambda x: x + hl.rand_int64()))
    assert len(set(a)) == 10

    # test MatrixCount
    mt = hl.utils.range_matrix_table(10, 10)
    mt = mt.annotate_entries(x=hl.rand_int64())
    assert mt.count() == (10, 10)

    # test MatrixAggregate
    mt = hl.utils.range_matrix_table(5, 5)
    a = mt.aggregate_entries(hl.agg.collect(hl.rand_int64()).map(lambda x: x + hl.rand_int64()))
    assert len(set(a)) == 25


def test_keyed_intersection():
    a1 = hl.literal(
        [
            hl.Struct(a=5, b='foo'),
            hl.Struct(a=7, b='bar'),
            hl.Struct(a=9, b='baz'),
        ]
    )
    a2 = hl.literal(
        [
            hl.Struct(a=5, b='foo'),
            hl.Struct(a=6, b='qux'),
            hl.Struct(a=8, b='qux'),
            hl.Struct(a=9, b='baz'),
        ]
    )
    assert hl.eval(hl.keyed_intersection(a1, a2, key=['a'])) == [
        hl.Struct(a=5, b='foo'),
        hl.Struct(a=9, b='baz'),
    ]


def test_keyed_union():
    a1 = hl.literal(
        [
            hl.Struct(a=5, b='foo'),
            hl.Struct(a=7, b='bar'),
            hl.Struct(a=9, b='baz'),
        ]
    )
    a2 = hl.literal(
        [
            hl.Struct(a=5, b='foo'),
            hl.Struct(a=6, b='qux'),
            hl.Struct(a=8, b='qux'),
            hl.Struct(a=9, b='baz'),
        ]
    )
    assert hl.eval(hl.keyed_union(a1, a2, key=['a'])) == [
        hl.Struct(a=5, b='foo'),
        hl.Struct(a=6, b='qux'),
        hl.Struct(a=7, b='bar'),
        hl.Struct(a=8, b='qux'),
        hl.Struct(a=9, b='baz'),
    ]


def test_to_relational_row_and_col_refs():
    mt = hl.utils.range_matrix_table(1, 1)
    mt = mt.annotate_rows(x=1)
    mt = mt.annotate_cols(y=1)
    mt = mt.annotate_entries(z=1)

    assert mt.row._to_relational_preserving_rows_and_cols('x')[1].row.dtype == hl.tstruct(
        row_idx=hl.tint32, x=hl.tint32
    )
    assert mt.row_key._to_relational_preserving_rows_and_cols('x')[1].row.dtype == hl.tstruct(row_idx=hl.tint32)

    assert mt.col._to_relational_preserving_rows_and_cols('x')[1].row.dtype == hl.tstruct(
        col_idx=hl.tint32, y=hl.tint32
    )
    assert mt.col_key._to_relational_preserving_rows_and_cols('x')[1].row.dtype == hl.tstruct(col_idx=hl.tint32)


def test_locus_addition():

    rg = hl.get_reference('GRCh37')
    len_1 = rg.lengths['1']
    loc = hl.locus('1', 5, reference_genome='GRCh37')

    assert hl.eval((loc + 10) == hl.locus('1', 15, reference_genome='GRCh37'))
    assert hl.eval((loc - 10) == hl.locus('1', 1, reference_genome='GRCh37'))
    assert hl.eval((loc + 2_000_000_000) == hl.locus('1', len_1, reference_genome='GRCh37'))


def test_reservoir_sampling_pointer_type():
    ht = hl.utils.range_table(100000, 1)
    assert ht.aggregate(hl.agg._reservoir_sample(hl.str(ht.idx), 1000).all(lambda x: hl.str(hl.int(x)) == x))


def test_reservoir_sampling():
    ht = hl.Table._generate(
        hl.literal([(1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 100000)]),
        5,
        lambda ctx, _: hl.range(ctx[0], ctx[1]).map(lambda i: hl.struct(idx=i)),
    )

    sample_sizes = [99, 811, 900, 1000, 3333]
    (stats, samples) = ht.aggregate(
        (hl.agg.stats(ht.idx), tuple([hl.sorted(hl.agg._reservoir_sample(ht.idx, size)) for size in sample_sizes]))
    )

    sample_variance = stats['stdev'] ** 2
    sample_mean = stats['mean']

    for sample, sample_size in zip(samples, sample_sizes):
        mean = np.mean(sample)
        expected_stdev = math.sqrt(sample_variance / sample_size)
        assert abs(mean - sample_mean) / expected_stdev < 4, (
            iteration,
            sample_size,
            abs(mean - sample_mean) / expected_stdev,
        )


def test_local_agg():
    x = hl.literal([1, 2, 3, 4])
    assert hl.eval(x.aggregate(lambda x: hl.agg.sum(x))) == 10


def test_zip_join_producers():
    contexts = hl.literal([1, 2, 3])
    zj = hl._zip_join_producers(
        contexts,
        lambda i: hl.range(i).map(lambda x: hl.struct(k=x, stream_id=i)),
        ['k'],
        lambda k, vals: k.annotate(vals=vals),
    )
    assert hl.eval(zj) == [
        hl.utils.Struct(
            k=0,
            vals=[
                hl.utils.Struct(k=0, stream_id=1),
                hl.utils.Struct(k=0, stream_id=2),
                hl.utils.Struct(k=0, stream_id=3),
            ],
        ),
        hl.utils.Struct(
            k=1,
            vals=[
                None,
                hl.utils.Struct(k=1, stream_id=2),
                hl.utils.Struct(k=1, stream_id=3),
            ],
        ),
        hl.utils.Struct(
            k=2,
            vals=[
                None,
                None,
                hl.utils.Struct(k=2, stream_id=3),
            ],
        ),
    ]
