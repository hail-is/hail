import unittest

import hail as hl
import hail.expr.aggregators as agg
from hail.expr import dtype, coercer_from_dtype
from hail.expr.types import *
from .utils import startTestHailContext, stopTestHailContext

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class TypeTests(unittest.TestCase):
    def types_to_test(self):
        return [
            tint32,
            tint64,
            tfloat32,
            tfloat64,
            tstr,
            tbool,
            tcall,
            tinterval(tint32),
            tdict(tstr, tint32),
            tarray(tstr),
            tset(tint64),
            tlocus('GRCh37'),
            tlocus('GRCh38'),
            tstruct(),
            tstruct(x=tint32, y=tint64, z=tarray(tset(tstr))),
            tstruct(**{'weird field name 1': tint32,
                       r"""this one ' has "" quotes and `` backticks```""": tint64,
                       '!@#$%^&({[': tarray(tset(tstr))}),
            tinterval(tlocus()),
            tset(tinterval(tlocus())),
            tstruct(a=tint32, b=tint32, c=tarray(tstr)),
            tstruct(a=tfloat64, bb=tint32, c=tbool),
            tstruct(a=tint32, b=tint32),
            tstruct(**{'___': tint32, '_ . _': tint32}),
            ttuple(tstr, tint32),
            ttuple(tarray(tint32), tstr, tstr, tint32, tbool),
            ttuple()]

    def test_parser_roundtrip(self):
        for t in self.types_to_test():
            self.assertEqual(t, dtype(str(t)))

    def test_eval_roundtrip(self):
        for t in self.types_to_test():
            self.assertEqual(t, eval(repr(t)))

    def test_equality(self):
        ts = self.types_to_test()
        ts2 = self.types_to_test()  # reallocates the non-primitive types

        for i in range(len(ts)):
            for j in range(len(ts2)):
                if (i == j):
                    self.assertEqual(ts[i], ts2[j])
                else:
                    self.assertNotEqual(ts[i], ts2[j])

    def test_jvm_roundtrip(self):
        ts = self.types_to_test()
        for t in ts:
            rev_str = t._jtype.toString()
            self.assertEqual(t, dtype(rev_str))

    def test_pretty_roundtrip(self):
        ts = self.types_to_test()
        for t in ts:
            p1 = t.pretty()
            p2 = t.pretty(5, 5)
            self.assertEqual(t, dtype(p1))
            self.assertEqual(t, dtype(p2))

    def test_coercers_can_coerce(self):
        ts = self.types_to_test()
        for t in ts:
            c = coercer_from_dtype(t)
            self.assertTrue(c.can_coerce(t))
            self.assertFalse(c.requires_conversion(t))


class Tests(unittest.TestCase):
    def test_floating_point(self):
        self.assertEqual(hl.eval_expr(1.1e-15), 1.1e-15)

    def test_bind_multiple(self):
        self.assertEqual(hl.bind(lambda x, y: x * y, 2, 3).value, 6)
        self.assertEqual(hl.bind(lambda y: y * 2, 3).value, 6)

    def test_bind_placement(self):
        self.assertEqual((5 / hl.bind(lambda x: x, 5)).value, 1.0)

    def test_matches(self):
        self.assertEqual(hl.eval_expr('\d+'), '\d+')
        string = hl.literal('12345')
        self.assertTrue(hl.eval_expr(string.matches('\d+')))
        self.assertFalse(hl.eval_expr(string.matches(r'\\d+')))

    def test_first_match_in(self):
        string = hl.literal('1:25-100')
        self.assertTrue(string.first_match_in("([^:]*)[:\\t](\\d+)[\\-\\t](\\d+)").value == ['1', '25', '100'])
        self.assertIsNone(string.first_match_in("hello (\w+)!").value)

    def test_cond(self):
        self.assertEqual(hl.eval_expr('A' + hl.cond(True, 'A', 'B')), 'AA')

        self.assertEqual(hl.cond(True, hl.struct(), hl.null(hl.tstruct())).value, hl.utils.Struct())

        self.assertEqual(hl.cond(hl.null(hl.tbool), 1, 2).value, None)
        self.assertEqual(hl.cond(hl.null(hl.tbool), 1, 2, missing_false=True).value, 2)

    def test_aggregators(self):
        table = hl.utils.range_table(10)
        r = table.aggregate(hl.struct(x=agg.count(),
                                      y=agg.count_where(table.idx % 2 == 0),
                                      z=agg.count(agg.filter(lambda x: x % 2 == 0, table.idx)),
                                      arr_sum=agg.array_sum([1, 2, hl.null(tint32)]),
                                      bind_agg=agg.count_where(hl.bind(lambda x: x % 2 == 0, table.idx)),
                                      mean=agg.mean(table.idx)))

        self.assertEqual(r.x, 10)
        self.assertEqual(r.y, 5)
        self.assertEqual(r.z, 5)
        self.assertEqual(r.arr_sum, [10, 20, 0])
        self.assertEqual(r.bind_agg, 5)

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
        self.assertEqual(hl.eval_expr(expr1), 6)

        expr2 = (hl.switch(x)
            .when('123', 5)
            .when('0', 2)
            .or_missing())
        self.assertEqual(hl.eval_expr(expr2), None)

        expr3 = (hl.switch(x)
            .when('123', 5)
            .when('0', 2)
            .default(100))
        self.assertEqual(hl.eval_expr(expr3), 100)

        expr4 = (hl.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(hl.null(tint32), 3)  # NA != NA
            .default(4))
        self.assertEqual(hl.eval_expr(expr4), None)

        expr5 = (hl.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(hl.null(tint32), 3)  # NA != NA
            .when_missing(-1)
            .default(4))
        self.assertEqual(hl.eval_expr(expr5), -1)

    def test_case(self):
        def make_case(x):
            x = hl.literal(x)
            return (hl.case()
                .when(x == 6, 'A')
                .when(x % 3 == 0, 'B')
                .when(x == 5, 'C')
                .when(x < 2, 'D')
                .or_missing())

        self.assertEqual(hl.eval_expr(make_case(6)), 'A')
        self.assertEqual(hl.eval_expr(make_case(12)), 'B')
        self.assertEqual(hl.eval_expr(make_case(5)), 'C')
        self.assertEqual(hl.eval_expr(make_case(-1)), 'D')
        self.assertEqual(hl.eval_expr(make_case(2)), None)

        self.assertEqual(hl.case().when(hl.null(hl.tbool), 1).default(2).value, None)
        self.assertEqual(hl.case(missing_false=True).when(hl.null(hl.tbool), 1).default(2).value, 2)

    def test_struct_ops(self):
        s = hl.struct(f1=1, f2=2, f3=3)

        def assert_typed(expr, result, dtype):
            self.assertEqual(expr.dtype, dtype)
            r, t = hl.eval_expr_typed(expr)
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
        self.assertRaises(TypeError, lambda: hl.eval_expr(list(a)))

    def test_dict_get(self):
        d = hl.dict({'a': 1, 'b': 2, 'missing_value': hl.null(hl.tint32)})
        self.assertEqual(hl.eval_expr(d.get('a')), 1)
        self.assertEqual(hl.eval_expr(d['a']), 1)
        self.assertEqual(hl.eval_expr(d.get('b')), 2)
        self.assertEqual(hl.eval_expr(d['b']), 2)
        self.assertEqual(hl.eval_expr(d.get('c')), None)
        self.assertEqual(hl.eval_expr(d.get('c', 5)), 5)
        self.assertEqual(hl.eval_expr(d.get('a', 5)), 1)

        self.assertEqual(hl.eval_expr(d.get('missing_values')), None)
        self.assertEqual(hl.eval_expr(d.get('missing_values', hl.null(hl.tint32))), None)
        self.assertEqual(hl.eval_expr(d.get('missing_values', 5)), 5)

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

        self.assertEqual(df.aggregate(agg.any(agg.filter(lambda x: False, df.all_true))), False)
        self.assertEqual(df.aggregate(agg.all(agg.filter(lambda x: False, df.all_true))), True)

    def test_str_ops(self):
        s = hl.literal("123")
        self.assertEqual(hl.eval_expr(hl.int32(s)), 123)

        s = hl.literal("123123123123")
        self.assertEqual(hl.eval_expr(hl.int64(s)), 123123123123)

        s = hl.literal("1.5")
        self.assertEqual(hl.eval_expr(hl.float32(s)), 1.5)
        self.assertEqual(hl.eval_expr(hl.float64(s)), 1.5)

        s1 = hl.literal('true')
        s2 = hl.literal('True')
        s3 = hl.literal('TRUE')

        s4 = hl.literal('false')
        s5 = hl.literal('False')
        s6 = hl.literal('FALSE')

        self.assertTrue(hl.eval_expr(hl.bool(s1)))
        self.assertTrue(hl.eval_expr(hl.bool(s2)))
        self.assertTrue(hl.eval_expr(hl.bool(s3)))

        self.assertFalse(hl.eval_expr(hl.bool(s4)))
        self.assertFalse(hl.eval_expr(hl.bool(s5)))
        self.assertFalse(hl.eval_expr(hl.bool(s6)))

        s = hl.literal('abcABC123')
        self.assertEqual(s.lower().value, 'abcabc123')
        self.assertEqual(s.upper().value, 'ABCABC123')

        s_whitespace = hl.literal(' \t 1 2 3 \t\n')
        self.assertEqual(s_whitespace.strip().value, '1 2 3')

        self.assertEqual(s.contains('ABC').value, True)
        self.assertEqual((~s.contains('ABC')).value, False)
        self.assertEqual(s.contains('a').value, True)
        self.assertEqual(s.contains('C123').value, True)
        self.assertEqual(s.contains('').value, True)
        self.assertEqual(s.contains('C1234').value, False)
        self.assertEqual(s.contains(' ').value, False)

        self.assertTrue(s_whitespace.startswith(' \t').value)
        self.assertTrue(s_whitespace.endswith('\t\n').value)
        self.assertFalse(s_whitespace.startswith('a').value)
        self.assertFalse(s_whitespace.endswith('a').value)


    def check_expr(self, expr, expected, expected_type):
        self.assertEqual(expected_type, expr.dtype)
        self.assertEqual((expected, expected_type), hl.eval_expr_typed(expr))

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

        self.assertEqual(hl.int32(b1).value, 1)
        self.assertEqual(hl.int64(b1).value, 1)
        self.assertEqual(hl.float32(b1).value, 1.0)
        self.assertEqual(hl.float64(b1).value, 1.0)
        self.assertEqual((b1 * b2).value, 0)
        self.assertEqual((b1 + b2).value, 1)
        self.assertEqual((b1 - b2).value, 1)
        self.assertEqual((b1 / b1).value, 1.0)
        self.assertEqual((f1 * b2).value, 0.0)
        self.assertEqual((b_array + f1).value, [6.5, 5.5])
        self.assertEqual((b_array + f_array).value, [2.5, 2.5])


    def test_allele_methods(self):
        self.assertTrue(hl.eval_expr(hl.is_transition("A", "G")))
        self.assertFalse(hl.eval_expr(hl.is_transversion("A", "G")))
        self.assertTrue(hl.eval_expr(hl.is_transversion("A", "T")))
        self.assertFalse(hl.eval_expr(hl.is_transition("A", "T")))
        self.assertTrue(hl.eval_expr(hl.is_snp("A", "T")))
        self.assertTrue(hl.eval_expr(hl.is_snp("A", "G")))
        self.assertTrue(hl.eval_expr(hl.is_snp("C", "G")))
        self.assertTrue(hl.eval_expr(hl.is_snp("CC", "CG")))
        self.assertTrue(hl.eval_expr(hl.is_snp("AT", "AG")))
        self.assertTrue(hl.eval_expr(hl.is_snp("ATCCC", "AGCCC")))
        self.assertTrue(hl.eval_expr(hl.is_mnp("ACTGAC", "ATTGTT")))
        self.assertTrue(hl.eval_expr(hl.is_mnp("CA", "TT")))
        self.assertTrue(hl.eval_expr(hl.is_insertion("A", "ATGC")))
        self.assertTrue(hl.eval_expr(hl.is_insertion("ATT", "ATGCTT")))
        self.assertTrue(hl.eval_expr(hl.is_deletion("ATGC", "A")))
        self.assertTrue(hl.eval_expr(hl.is_deletion("GTGTA", "GTA")))
        self.assertTrue(hl.eval_expr(hl.is_indel("A", "ATGC")))
        self.assertTrue(hl.eval_expr(hl.is_indel("ATT", "ATGCTT")))
        self.assertTrue(hl.eval_expr(hl.is_indel("ATGC", "A")))
        self.assertTrue(hl.eval_expr(hl.is_indel("GTGTA", "GTA")))
        self.assertTrue(hl.eval_expr(hl.is_complex("CTA", "ATTT")))
        self.assertTrue(hl.eval_expr(hl.is_complex("A", "TATGC")))
        self.assertTrue(hl.eval_expr(hl.is_star("ATC", "*")))
        self.assertTrue(hl.eval_expr(hl.is_star("A", "*")))
        self.assertTrue(hl.eval_expr(hl.is_strand_ambiguous("A", "T")))
        self.assertFalse(hl.eval_expr(hl.is_strand_ambiguous("G", "T")))

    def test_allele_type(self):
        self.assertEqual(
            hl.tuple((
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
                hl.allele_type('', 'CCT'),
                hl.allele_type('F', 'CCT'),
                hl.allele_type('A', '[ASDASD[A'),
                hl.allele_type('A', ']ASDASD]A'),
                hl.allele_type('A', 'T<ASDASD>]ASDASD]'),
                hl.allele_type('A', 'T<ASDASD>[ASDASD['),
                hl.allele_type('A', '.T'),
                hl.allele_type('A', 'T.'),
            )).value,
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
                'Symbolic',
                'Symbolic',
                'Symbolic',
                'Symbolic',
                'Symbolic',
                'Symbolic',
            )
        )

    def test_hamming(self):
        self.assertEqual(hl.eval_expr(hl.hamming('A', 'T')), 1)
        self.assertEqual(hl.eval_expr(hl.hamming('AAAAA', 'AAAAT')), 1)
        self.assertEqual(hl.eval_expr(hl.hamming('abcde', 'edcba')), 4)

    def test_gp_dosage(self):
        self.assertAlmostEqual(hl.eval_expr(hl.gp_dosage([1.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(hl.eval_expr(hl.gp_dosage([0.0, 1.0, 0.0])), 1.0)
        self.assertAlmostEqual(hl.eval_expr(hl.gp_dosage([0.0, 0.0, 1.0])), 2.0)
        self.assertAlmostEqual(hl.eval_expr(hl.gp_dosage([0.5, 0.5, 0.0])), 0.5)
        self.assertAlmostEqual(hl.eval_expr(hl.gp_dosage([0.0, 0.5, 0.5])), 1.5)

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
        self.check_expr(c2_hetvar.is_het_nonref(), True, tbool)

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
        self.assertEqual(hl.parse_variant('1:1:A:T').value,
                         hl.Struct(locus=hl.Locus('1', 1), alleles=['A', 'T']))

    def test_dict_conversions(self):
        self.assertEqual(sorted(hl.eval_expr(hl.array({1: 1, 2: 2}))), [(1, 1), (2, 2)])
        self.assertEqual(hl.eval_expr(hl.dict(hl.array({1: 1, 2: 2}))), {1: 1, 2: 2})

        self.assertEqual(hl.eval_expr(hl.dict([('1', 2), ('2', 3)])), {'1': 2, '2': 3})
        self.assertEqual(hl.eval_expr(hl.dict({('1', 2), ('2', 3)})), {'1': 2, '2': 3})
        self.assertEqual(hl.eval_expr(hl.dict([('1', 2), (hl.null(tstr), 3)])), {'1': 2, None: 3})
        self.assertEqual(hl.eval_expr(hl.dict({('1', 2), (hl.null(tstr), 3)})), {'1': 2, None: 3})

    def test_zip(self):
        a1 = [1,2,3]
        a2 = ['a', 'b']
        a3 = [[1]]

        self.assertEqual(hl.eval_expr(hl.zip(a1, a2)), [(1, 'a'), (2, 'b')])
        self.assertEqual(hl.eval_expr(hl.zip(a1, a2, fill_missing=True)), [(1, 'a'), (2, 'b'), (3, None)])

        self.assertEqual(hl.eval_expr(hl.zip(a3, a2, a1, fill_missing=True)),
                         [([1], 'a', 1), (None, 'b', 2), (None, None, 3)])
        self.assertEqual(hl.eval_expr(hl.zip(a3, a2, a1)),
                         [([1], 'a', 1)])

    def test_array_methods(self):
        self.assertEqual(hl.eval_expr(hl.any(lambda x: x % 2 == 0, [1, 3, 5])), False)
        self.assertEqual(hl.eval_expr(hl.any(lambda x: x % 2 == 0, [1, 3, 5, 6])), True)

        self.assertEqual(hl.eval_expr(hl.all(lambda x: x % 2 == 0, [1, 3, 5, 6])), False)
        self.assertEqual(hl.eval_expr(hl.all(lambda x: x % 2 == 0, [2, 6])), True)

        self.assertEqual(hl.eval_expr(hl.find(lambda x: x % 2 == 0, [1, 3, 4, 6])), 4)
        self.assertEqual(hl.eval_expr(hl.find(lambda x: x % 2 != 0, [0, 2, 4, 6])), None)

        self.assertEqual(hl.eval_expr(hl.map(lambda x: x % 2 == 0, [0, 1, 4, 6])), [True, False, True, True])

        self.assertEqual(hl.eval_expr(hl.len([0, 1, 4, 6])), 4)

        self.assertEqual(hl.eval_expr(hl.max([0, 1, 4, 6])), 6)

        self.assertEqual(hl.eval_expr(hl.max(hl.empty_array(hl.tint))), None)

        self.assertEqual(hl.eval_expr(hl.min([0, 1, 4, 6])), 0)

        self.assertEqual(hl.eval_expr(hl.mean([0, 1, 4, 6])), 2.75)

        self.assertTrue(1 <= hl.eval_expr(hl.median([0, 1, 4, 6])) <= 4)

        for f in [lambda x: hl.int32(x), lambda x: hl.int64(x), lambda x: hl.float32(x), lambda x: hl.float64(x)]:
            self.assertEqual(hl.product([f(x) for x in [1, 4, 6]]).value, 24)
            self.assertEqual(hl.sum([f(x) for x in [1, 4, 6]]).value, 11)
            self.assertEqual(hl.max([f(x) for x in [-5, -4, -3, -2]]).value, -2)

        self.assertEqual(hl.eval_expr(hl.group_by(lambda x: x % 2 == 0, [0, 1, 4, 6])), {True: [0, 4, 6], False: [1]})

        self.assertEqual(hl.eval_expr(hl.flatmap(lambda x: hl.range(0, x), [1, 2, 3])), [0, 0, 1, 0, 1, 2])

    def test_bool_r_ops(self):
        self.assertTrue(hl.eval_expr(hl.literal(True) & True))
        self.assertTrue(hl.eval_expr(True & hl.literal(True)))
        self.assertTrue(hl.eval_expr(hl.literal(False) | True))
        self.assertTrue(hl.eval_expr(True | hl.literal(False)))

    def test_array_neg(self):
        self.assertEqual(hl.eval_expr(-(hl.literal([1, 2, 3]))), [-1, -2, -3])

    def test_min_max(self):
        self.assertEqual(hl.eval_expr(hl.max(1, 2)), 2)
        self.assertEqual(hl.eval_expr(hl.max(1.0, 2)), 2.0)
        self.assertEqual(hl.eval_expr(hl.max([1, 2])), 2)
        self.assertEqual(hl.eval_expr(hl.max([1.0, 2])), 2.0)
        self.assertEqual(hl.eval_expr(hl.max(0, 1.0, 2)), 2.0)
        self.assertEqual(hl.eval_expr(hl.max(0, 1, 2)), 2)
        self.assertEqual(hl.eval_expr(hl.max([0, 10, 2, 3, 4, 5, 6, ])), 10)
        self.assertEqual(hl.eval_expr(hl.max(0, 10, 2, 3, 4, 5, 6)), 10)

        self.assertEqual(hl.eval_expr(hl.min(1, 2)), 1)
        self.assertEqual(hl.eval_expr(hl.min(1.0, 2)), 1.0)
        self.assertEqual(hl.eval_expr(hl.min([1, 2])), 1)
        self.assertEqual(hl.eval_expr(hl.min([1.0, 2])), 1.0)
        self.assertEqual(hl.eval_expr(hl.min(0, 1.0, 2)), 0.0)
        self.assertEqual(hl.eval_expr(hl.min(0, 1, 2)), 0)
        self.assertEqual(hl.eval_expr(hl.min([0, 10, 2, 3, 4, 5, 6, ])), 0)
        self.assertEqual(hl.eval_expr(hl.min(4, 10, 2, 3, 4, 5, 6)), 2)

    def test_abs(self):
        self.assertEqual(hl.eval_expr(hl.abs(-5)), 5)
        self.assertEqual(hl.eval_expr(hl.abs(-5.5)), 5.5)
        self.assertEqual(hl.eval_expr(hl.abs(5.5)), 5.5)
        self.assertEqual(hl.eval_expr(hl.abs([5.5, -5.5])), [5.5, 5.5])

    def test_signum(self):
        self.assertEqual(hl.eval_expr(hl.signum(-5)), -1)
        self.assertEqual(hl.eval_expr(hl.signum(0.0)), 0)
        self.assertEqual(hl.eval_expr(hl.signum(10.0)), 1)
        self.assertEqual(hl.eval_expr(hl.signum([-5, 0, 10])), [-1, 0, 1])

    def test_argmin_and_argmax(self):
        a = hl.array([2, 1, 1, 4, 4, 3])
        self.assertEqual(hl.eval_expr(hl.argmax(a)), 3)
        self.assertEqual(hl.eval_expr(hl.argmax(a, unique=True)), None)
        self.assertEqual(hl.eval_expr(hl.argmin(a)), 1)
        self.assertEqual(hl.eval_expr(hl.argmin(a, unique=True)), None)
        self.assertEqual(hl.eval_expr(hl.argmin(hl.empty_array(tint32))), None)

    def test_show_row_key_regression(self):
        ds = hl.utils.range_matrix_table(3, 3)
        ds.col_idx.show(3)

    def test_or_else_type_conversion(self):
        self.assertEqual(hl.or_else(0.5, 2).value, 0.5)

    def test_coalesce(self):
        self.assertEqual(hl.coalesce(hl.null('int'), hl.null('int'), hl.null('int')).value, None)
        self.assertEqual(hl.coalesce(hl.null('int'), hl.null('int'), 2).value, 2)
        self.assertEqual(hl.coalesce(hl.null('int'), hl.null('int'), 2.5).value, 2.5)
        self.assertEqual(hl.coalesce(2.5).value, 2.5)
        with self.assertRaises(TypeError):
            hl.coalesce(2.5, 'hello')

    def test_tuple_ops(self):
        t0 = hl.literal(())
        t1 = hl.literal((1,))
        t2 = hl.literal((1, "hello"))
        tn1 = hl.literal((1, (2, (3, 4))))

        t = hl.tuple([1, t1, hl.dict(hl.zip(["a", "b"], [t2, t2])), [1, 5], tn1])

        self.assertTrue(hl.eval_expr(t[0]) == 1)
        self.assertTrue(hl.eval_expr(t[1][0]) == 1)
        self.assertTrue(hl.eval_expr(t[2]["a"]) == (1, "hello"))
        self.assertTrue(hl.eval_expr(t[2]["b"][1]) == "hello")
        self.assertTrue(hl.eval_expr(t[3][1]) == 5)
        self.assertTrue(hl.eval_expr(t[4][1][1][1]) == 4)

        self.assertTrue(hl.eval_expr(hl.len(t0) == 0))
        self.assertTrue(hl.eval_expr(hl.len(t2) == 2))
        self.assertTrue(hl.eval_expr(hl.len(t)) == 5)

    def test_interval_ops(self):
        interval = hl.interval(3, 6)
        self.assertTrue(hl.eval_expr_typed(interval.start) == (3, hl.tint))
        self.assertTrue(hl.eval_expr_typed(interval.end) == (6, hl.tint))
        self.assertTrue(hl.eval_expr_typed(interval.includes_start) == (True, hl.tbool))
        self.assertTrue(hl.eval_expr_typed(interval.includes_end) == (False, hl.tbool))
        self.assertTrue(hl.eval_expr_typed(interval.contains(5)) == (True, hl.tbool))
        self.assertTrue(hl.eval_expr_typed(interval.overlaps(hl.interval(5, 9))) == (True, hl.tbool))

        li = hl.parse_locus_interval('1:100-110')
        self.assertEqual(li.value, hl.utils.Interval(hl.genetics.Locus("1", 100),
                                                     hl.genetics.Locus("1", 110)))
        self.assertTrue(li.dtype.point_type == hl.tlocus())
        self.assertTrue(li.contains(hl.locus("1", 100)).value)
        self.assertTrue(li.contains(hl.locus("1", 109)).value)
        self.assertFalse(li.contains(hl.locus("1", 110)).value)
    
        li2 = hl.parse_locus_interval("1:109-200")
        li3 = hl.parse_locus_interval("1:110-200")
        li4 = hl.parse_locus_interval("1:90-101")
        li5 = hl.parse_locus_interval("1:90-100")
    
        self.assertTrue(li.overlaps(li2).value)
        self.assertTrue(li.overlaps(li4).value)
        self.assertFalse(li.overlaps(li3).value)
        self.assertFalse(li.overlaps(li5).value)

    def test_reference_genome_fns(self):
        self.assertTrue(hl.is_valid_contig('1', 'GRCh37').value)
        self.assertFalse(hl.is_valid_contig('chr1', 'GRCh37').value)
        self.assertFalse(hl.is_valid_contig('1', 'GRCh38').value)
        self.assertTrue(hl.is_valid_contig('chr1', 'GRCh38').value)

        self.assertTrue(hl.is_valid_locus('1', 325423, 'GRCh37').value)
        self.assertFalse(hl.is_valid_locus('1', 0, 'GRCh37').value)
        self.assertFalse(hl.is_valid_locus('1', 249250622, 'GRCh37').value)
        self.assertFalse(hl.is_valid_locus('chr1', 2645, 'GRCh37').value)

    def test_call_stats(self):
        t = (hl.utils.range_table(5, 3)
             .annotate(GT = hl.call(0, 1))
             .annotate_globals(alleles=["A", "T"]))

        self.assertTrue(t.aggregate(agg.call_stats(t.GT, t.alleles)) ==
                        hl.Struct(AC=[5, 5], AF=[0.5, 0.5], AN=10, homozygote_count=[0, 0])) # Tests table.aggregate initOp

        mt = (hl.utils.range_matrix_table(10, 5, 5)
              .annotate_entries(GT=hl.call(0, 1))
              .annotate_rows(alleles=["A", "T"])
              .annotate_globals(alleles2=["G", "C"]))

        row_agg = mt.annotate_rows(call_stats=agg.call_stats(mt.GT, mt.alleles)).rows() # Tests MatrixMapRows initOp
        col_agg = mt.annotate_cols(call_stats=agg.call_stats(mt.GT, mt.alleles2)).cols() # Tests MatrixMapCols initOp

        self.assertTrue(row_agg.all(row_agg.call_stats ==
                                    hl.struct(AC=[5, 5], AF=[0.5, 0.5], AN=10, homozygote_count=[0, 0])))
        self.assertTrue(col_agg.all(col_agg.call_stats ==
                                    hl.struct(AC=[10, 10], AF=[0.5, 0.5], AN=20, homozygote_count=[0, 0])))

    def test_mendel_error_code(self):
        locus_auto = hl.Locus('2', 20000000)
        locus_x_par = hl.get_reference('default').par[0].start
        locus_x_nonpar = hl.Locus(locus_x_par.contig, locus_x_par.position - 1)
        locus_y_nonpar = hl.Locus('Y', hl.get_reference('default').lengths['Y'] - 1)

        self.assertTrue(hl.all(lambda x: x, hl.array([
            hl.literal(locus_auto).in_autosome_or_par(),
            hl.literal(locus_auto).in_autosome_or_par(),
            ~hl.literal(locus_x_par).in_autosome(),
            hl.literal(locus_x_par).in_autosome_or_par(),
            ~hl.literal(locus_x_nonpar).in_autosome_or_par(),
            hl.literal(locus_x_nonpar).in_x_nonpar(),
            ~hl.literal(locus_y_nonpar).in_autosome_or_par(),
            hl.literal(locus_y_nonpar).in_y_nonpar()
        ])).value)

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
        results = expr.value
        for args, result in results.items():
            self.assertEqual(result, expected[args], msg=f'expected {expected[args]}, found {result} at {str(args)}')

    def test_min_rep(self):
        def assert_min_reps_to(old, new, pos_change=0):
            self.assertEqual(
                hl.min_rep(hl.locus('1', 10), old).value,
                (hl.Locus('1', 10 + pos_change), new)
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
        self.assertEqual(e.value, v)

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
