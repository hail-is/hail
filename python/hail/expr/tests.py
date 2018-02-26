import unittest

import hail as hl
from hail import Struct, Table, Locus
import hail.expr.aggregators as agg
from hail.expr.types import *


def setUpModule():
    hl.init(master='local[2]', min_block_size=0)


def tearDownModule():
    hl.stop()


class Tests(unittest.TestCase):
    def test_types(self):
        self.assertEqual(tint32, tint32)
        self.assertEqual(tfloat64, tfloat64)
        self.assertEqual(tarray(tfloat64), tarray(tfloat64))
        self.assertNotEqual(tarray(tfloat64), tarray(tfloat32))
        self.assertNotEqual(tset(tfloat64), tarray(tfloat64))
        self.assertEqual(tset(tfloat64), tset(tfloat64))
        self.assertEqual(tdict(tstr, tarray(tint32)), tdict(tstr, tarray(tint32)))

        some_random_types = [
            tint32,
            tstr,
            tfloat32,
            tfloat64,
            tbool,
            tarray(tstr),
            tset(tarray(tset(tbool))),
            tdict(tstr, tint32),
            tlocus(),
            tcall,
            tinterval(tlocus()),
            tset(tinterval(tlocus())),
            tstruct(['a', 'b', 'c'], [tint32, tint32, tarray(tstr)]),
            tstruct(['a', 'bb', 'c'], [tfloat64, tint32, tbool]),
            tstruct(['a', 'b'], [tint32, tint32])]

        #  copy and reinitialize to check that two initializations produce equality (not reference equality)
        some_random_types_cp = [
            tint32,
            tstr,
            tfloat32,
            tfloat64,
            tbool,
            tarray(tstr),
            tset(tarray(tset(tbool))),
            tdict(tstr, tint32),
            tlocus(),
            tcall,
            tinterval(tlocus()),
            tset(tinterval(tlocus())),
            tstruct(['a', 'b', 'c'], [tint32, tint32, tarray(tstr)]),
            tstruct(['a', 'bb', 'c'], [tfloat64, tint32, tbool]),
            tstruct(['a', 'b'], [tint32, tint32])]

        for i in range(len(some_random_types)):
            for j in range(len(some_random_types)):
                if (i == j):
                    self.assertEqual(some_random_types[i], some_random_types_cp[j])
                else:
                    self.assertNotEqual(some_random_types[i], some_random_types_cp[j])

    def test_floating_point(self):
        self.assertEqual(hl.eval_expr(1.1e-15), 1.1e-15)

    def test_repr(self):
        tl = tlocus()
        ti = tinterval(tlocus())
        tc = tcall

        ti32 = tint32
        ti64 = tint64
        tf32 = tfloat32
        tf64 = tfloat64
        ts = tstr
        tb = tbool

        tdict_ = tdict(tinterval(tlocus()), tfloat32)
        tset_ = tarray(tlocus())
        tarray_ = tarray(tstr)
        tstruct_ = tstruct(['a', 'b'], [tbool, tarray(tstr)])

        for typ in [tl, ti, tc,
                    ti32, ti64, tf32, tf64, ts, tb,
                    tdict_, tarray_, tset_, tstruct_]:
            self.assertEqual(eval(repr(typ)), typ)

    def test_matches(self):
        self.assertEqual(hl.eval_expr('\d+'), '\d+')
        string = hl.capture('12345')
        self.assertTrue(hl.eval_expr(string.matches('\d+')))
        self.assertFalse(hl.eval_expr(string.matches(r'\\d+')))

    def test_cond(self):
        self.assertEqual(hl.eval_expr('A' + hl.cond(True, 'A', 'B')), 'AA')

    def test_aggregators(self):
        table = hl.utils.range_table(10)
        r = table.aggregate(Struct(x=agg.count(),
                                   y=agg.count_where(table.idx % 2 == 0),
                                   z=agg.count(agg.filter(lambda x: x % 2 == 0, table.idx)),
                                   arr_sum=agg.array_sum([1, 2, hl.null(tint32)])))

        self.assertEqual(r.x, 10)
        self.assertEqual(r.y, 5)
        self.assertEqual(r.z, 5)
        self.assertEqual(r.arr_sum, [10, 20, 0])

        r = table.aggregate(Struct(fraction_odd=agg.fraction(table.idx % 2 == 0),
                                   lessthan6=agg.fraction(table.idx < 6),
                                   gt6=agg.fraction(table.idx > 6),
                                   assert1=agg.fraction(table.idx > 6) < 0.50,
                                   assert2=agg.fraction(table.idx < 6) >= 0.50))
        self.assertEqual(r.fraction_odd, 0.50)
        self.assertEqual(r.lessthan6, 0.60)
        self.assertEqual(r.gt6, 0.30)
        self.assertTrue(r.assert1)
        self.assertTrue(r.assert2)

    def test_dtype(self):
        i32 = hl.capture(5)
        self.assertEqual(i32.dtype, tint32)

        str_exp = hl.capture('5')
        self.assertEqual(str_exp.dtype, tstr)

    def test_switch(self):
        x = hl.capture('1')
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
            x = hl.capture(x)
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

    def test_struct_ops(self):
        s = hl.capture(Struct(f1=1, f2=2, f3=3))

        def assert_typed(expr, result, dtype):
            self.assertEqual(expr.dtype, dtype)
            r, t = hl.eval_expr_typed(expr)
            self.assertEqual(t, dtype)
            self.assertEqual(result, r)

        assert_typed(s.drop('f3'),
                     Struct(f1=1, f2=2),
                     tstruct(['f1', 'f2'], [tint32, tint32]))

        assert_typed(s.drop('f1'),
                     Struct(f2=2, f3=3),
                     tstruct(['f2', 'f3'], [tint32, tint32]))

        assert_typed(s.drop(),
                     Struct(f1=1, f2=2, f3=3),
                     tstruct(['f1', 'f2', 'f3'], [tint32, tint32, tint32]))

        assert_typed(s.select('f1', 'f2'),
                     Struct(f1=1, f2=2),
                     tstruct(['f1', 'f2'], [tint32, tint32]))

        assert_typed(s.select('f2', 'f1', f4=5, f5=6),
                     Struct(f2=2, f1=1, f4=5, f5=6),
                     tstruct(['f2', 'f1', 'f4', 'f5'], [tint32, tint32, tint32, tint32]))

        assert_typed(s.select(),
                     Struct(),
                     tstruct([], []))

        assert_typed(s.annotate(f1=5, f2=10, f4=15),
                     Struct(f1=5, f2=10, f3=3, f4=15),
                     tstruct(['f1', 'f2', 'f3', 'f4'], [tint32, tint32, tint32, tint32]))

        assert_typed(s.annotate(f1=5),
                     Struct(f1=5, f2=2, f3=3),
                     tstruct(['f1', 'f2', 'f3'], [tint32, tint32, tint32]))

        assert_typed(s.annotate(),
                     Struct(f1=1, f2=2, f3=3),
                     tstruct(['f1', 'f2', 'f3'], [tint32, tint32, tint32]))

    def test_iter(self):
        a = hl.capture([1, 2, 3])
        self.assertRaises(TypeError, lambda: hl.eval_expr(list(a)))

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
        s = hl.capture("123")
        self.assertEqual(hl.eval_expr(hl.int32(s)), 123)

        s = hl.capture("123123123123")
        self.assertEqual(hl.eval_expr(hl.int64(s)), 123123123123)

        s = hl.capture("1.5")
        self.assertEqual(hl.eval_expr(hl.float32(s)), 1.5)
        self.assertEqual(hl.eval_expr(hl.float64(s)), 1.5)

        s1 = hl.capture('true')
        s2 = hl.capture('True')
        s3 = hl.capture('TRUE')

        s4 = hl.capture('false')
        s5 = hl.capture('False')
        s6 = hl.capture('FALSE')

        self.assertTrue(hl.eval_expr(hl.bool(s1)))
        self.assertTrue(hl.eval_expr(hl.bool(s2)))
        self.assertTrue(hl.eval_expr(hl.bool(s3)))

        self.assertFalse(hl.eval_expr(hl.bool(s4)))
        self.assertFalse(hl.eval_expr(hl.bool(s5)))
        self.assertFalse(hl.eval_expr(hl.bool(s6)))

    def check_expr(self, expr, expected, expected_type):
        self.assertEqual(expected_type, expr.dtype)
        self.assertEqual((expected, expected_type), hl.eval_expr_typed(expr))

    def test_division(self):
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
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
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.capture([3, 3, 3, 3, hl.null(tint32)])
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
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.capture([3, 3, 3, 3, hl.null(tint32)])
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
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.capture([3, 3, 3, 3, hl.null(tint32)])
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
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.capture([3, 3, 3, 3, hl.null(tint32)])
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
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.capture([3, 3, 3, 3, hl.null(tint32)])
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
        a_int32 = hl.capture([2, 4, 8, 16, hl.null(tint32)])
        a_int64 = a_int32.map(lambda x: hl.int64(x))
        a_float32 = a_int32.map(lambda x: hl.float32(x))
        a_float64 = a_int32.map(lambda x: hl.float64(x))
        int32_4s = hl.capture([4, 4, 4, 4, hl.null(tint32)])
        int32_3s = hl.capture([3, 3, 3, 3, hl.null(tint32)])
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
        self.assertTrue(hl.eval_expr(hl.is_star("*", "ATC")))
        self.assertTrue(hl.eval_expr(hl.is_star("*", "A")))

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
        from hail import Call
        c2_homref = hl.capture(Call([0, 0]))
        c2_het = hl.capture(Call([1, 0], phased=True))
        c2_homvar = hl.capture(Call([1, 1]))
        c2_hetvar = hl.capture(Call([2, 1], phased=True))
        c1 = hl.capture(Call([1]))
        c0 = hl.capture(Call([]))
        cNull = hl.capture(hl.null(tcall))

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

        call_expr = hl.call(True, 1, 2)
        self.check_expr(call_expr[0], 1, tint32)
        self.check_expr(call_expr[1], 2, tint32)
        self.check_expr(call_expr.ploidy, 2, tint32)

        a0 = hl.capture(1)
        a1 = 2
        phased = hl.capture(True)
        call_expr = hl.call(phased, a0, a1)
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
        self.assertEqual(hl.eval_expr(hl.parse_variant('1:1:A:T')), Struct(locus=Locus('1', 1),
                                                                           alleles=['A', 'T']))

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

        self.assertEqual(hl.eval_expr(hl.min([0, 1, 4, 6])), 0)

        self.assertEqual(hl.eval_expr(hl.mean([0, 1, 4, 6])), 2.75)

        self.assertTrue(1 <= hl.eval_expr(hl.median([0, 1, 4, 6])) <= 4)

        self.assertEqual(hl.eval_expr(hl.product([1, 4, 6])), 24)

        self.assertEqual(hl.eval_expr(hl.group_by(lambda x: x % 2 == 0, [0, 1, 4, 6])), {True: [0, 4, 6], False: [1]})

        self.assertEqual(hl.eval_expr(hl.flatmap(lambda x: hl.range(0, x), [1, 2, 3])), [0, 0, 1, 0, 1, 2])


    def test_bool_r_ops(self):
        self.assertTrue(hl.eval_expr(hl.capture(True) & True))
        self.assertTrue(hl.eval_expr(True & hl.capture(True)))
        self.assertTrue(hl.eval_expr(hl.capture(False) | True))
        self.assertTrue(hl.eval_expr(True | hl.capture(False)))

    def test_array_neg(self):
        self.assertEqual(hl.eval_expr(-(hl.capture([1, 2, 3]))), [-1, -2, -3])

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

    def test_show_row_key_regression(self):
        ds = hl.utils.range_matrix_table(3, 3)
        ds.col_idx.show(3)
