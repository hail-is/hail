from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail.expr import *
from hail import *

def setUpModule():
    init(master='local[2]', min_block_size=0)

def tearDownModule():
    stop()

class Tests(unittest.TestCase):
    def test_types(self):
        self.assertEqual(TInt32(), TInt32())
        self.assertEqual(TFloat64(), TFloat64())
        self.assertEqual(TArray(TFloat64()), TArray(TFloat64()))
        self.assertNotEqual(TArray(TFloat64()), TArray(TFloat32()))
        self.assertNotEqual(TSet(TFloat64()), TArray(TFloat64()))
        self.assertEqual(TSet(TFloat64()), TSet(TFloat64()))
        self.assertEqual(TDict(TString(), TArray(TInt32())), TDict(TString(), TArray(TInt32())))

        some_random_types = [
            TInt32(),
            TString(),
            TFloat32(),
            TFloat64(),
            TBoolean(),
            TArray(TString()),
            TSet(TArray(TSet(TBoolean()))),
            TDict(TString(), TInt32()),
            TVariant(),
            TLocus(),
            TCall(),
            TAltAllele(),
            TInterval(TLocus()),
            TSet(TInterval(TLocus())),
            TStruct(['a', 'b', 'c'], [TInt32(), TInt32(), TArray(TString())]),
            TStruct(['a', 'bb', 'c'], [TFloat64(), TInt32(), TBoolean()]),
            TStruct(['a', 'b'], [TInt32(), TInt32()])]

        #  copy and reinitialize to check that two initializations produce equality (not reference equality)
        some_random_types_cp = [
            TInt32(),
            TString(),
            TFloat32(),
            TFloat64(),
            TBoolean(),
            TArray(TString()),
            TSet(TArray(TSet(TBoolean()))),
            TDict(TString(), TInt32()),
            TVariant(),
            TLocus(),
            TCall(),
            TAltAllele(),
            TInterval(TLocus()),
            TSet(TInterval(TLocus())),
            TStruct(['a', 'b', 'c'], [TInt32(), TInt32(), TArray(TString())]),
            TStruct(['a', 'bb', 'c'], [TFloat64(), TInt32(), TBoolean()]),
            TStruct(['a', 'b'], [TInt32(), TInt32()])]

        for i in range(len(some_random_types)):
            for j in range(len(some_random_types)):
                if (i == j):
                    self.assertEqual(some_random_types[i], some_random_types_cp[j])
                else:
                    self.assertNotEqual(some_random_types[i], some_random_types_cp[j])

        reqint = TInt32(required=True)
        self.assertEqual(reqint.required, True)
        optint = TInt32()
        self.assertEqual(optint.required, False)
        optint2 = TInt32(required=False)
        self.assertEqual(optint2.required, False)
        self.assertEqual(id(optint), id(optint2))

        reqint2 = TInt32(required=True)
        self.assertEqual(id(reqint), id(reqint2))

    def test_floating_point(self):
        self.assertEqual(eval_expr(1.1e-15), 1.1e-15)

    def test_repr(self):
        tv = TVariant()
        tl = TLocus()
        ti = TInterval(TLocus())
        tc = TCall()
        taa = TAltAllele()

        ti32 = TInt32()
        ti64 = TInt64()
        tf32 = TFloat32()
        tf64 = TFloat64()
        ts = TString()
        tb = TBoolean()

        tdict = TDict(TInterval(TLocus()), TFloat32())
        tarray = TArray(TString())
        tset = TSet(TVariant())
        tstruct = TStruct(['a', 'b'], [TBoolean(), TArray(TString())])

        for typ in [tv, tl, ti, tc, taa,
                    ti32, ti64, tf32, tf64, ts, tb,
                    tdict, tarray, tset, tstruct]:
            self.assertEqual(eval(repr(typ)), typ)

    def test_matches(self):
        self.assertEqual(eval_expr('\d+'), '\d+')
        string = functions.capture('12345')
        self.assertTrue(eval_expr(string.matches('\d+')))
        self.assertFalse(eval_expr(string.matches(r'\\d+')))

    def test_cond(self):
        self.assertEqual(eval_expr('A' + functions.cond(True, 'A', 'B')), 'AA')

    def test_aggregators(self):
        table = Table.range(10)
        r = table.aggregate(x=agg.count(),
                            y=agg.count_where(table.idx % 2 == 0),
                            z=agg.count(agg.filter(lambda x: x % 2 == 0, table.idx)),
                            arr_sum = agg.array_sum([1, 2, functions.null(TInt32())]))

        self.assertEqual(r.x, 10)
        self.assertEqual(r.y, 5)
        self.assertEqual(r.z, 5)
        self.assertEqual(r.arr_sum, [10, 20, 0])

        r = table.aggregate(fraction_odd = agg.fraction(table.idx % 2 == 0),
                            lessthan6 = agg.fraction(table.idx < 6),
                            gt6 = agg.fraction(table.idx > 6),
                            assert1 = agg.fraction(table.idx > 6) < 0.50,
                            assert2 = agg.fraction(table.idx < 6) >= 0.50)
        self.assertEqual(r.fraction_odd, 0.50)
        self.assertEqual(r.lessthan6, 0.60)
        self.assertEqual(r.gt6, 0.30)
        self.assertTrue(r.assert1)
        self.assertTrue(r.assert2)

    def test_dtype(self):
        i32 = functions.capture(5)
        self.assertEqual(i32.dtype, TInt32())

        str_exp = functions.capture('5')
        self.assertEqual(str_exp.dtype, TString())

    def test_switch(self):
        x = functions.capture('1')
        na = functions.null(TInt32())

        expr1 = (functions.switch(x)
            .when('123', 5)
            .when('1', 6)
            .when('0', 2)
            .or_missing())
        self.assertEqual(eval_expr(expr1), 6)

        expr2 = (functions.switch(x)
            .when('123', 5)
            .when('0', 2)
            .or_missing())
        self.assertEqual(eval_expr(expr2), None)

        expr3 = (functions.switch(x)
            .when('123', 5)
            .when('0', 2)
            .default(100))
        self.assertEqual(eval_expr(expr3), 100)

        expr4 = (functions.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(functions.null(TInt32()), 3)  # NA != NA
            .default(4))
        self.assertEqual(eval_expr(expr4), None)

        expr5 = (functions.switch(na)
            .when(5, 0)
            .when(6, 1)
            .when(0, 2)
            .when(functions.null(TInt32()), 3)  # NA != NA
            .when_missing(-1)
            .default(4))
        self.assertEqual(eval_expr(expr5), -1)

    def test_case(self):
        def make_case(x):
            x = functions.capture(x)
            return (functions.case()
            .when(x == 6, 'A')
            .when(x % 3 == 0, 'B')
            .when(x == 5, 'C')
            .when(x < 2, 'D')
            .or_missing())

        self.assertEqual(eval_expr(make_case(6)), 'A')
        self.assertEqual(eval_expr(make_case(12)), 'B')
        self.assertEqual(eval_expr(make_case(5)), 'C')
        self.assertEqual(eval_expr(make_case(-1)), 'D')
        self.assertEqual(eval_expr(make_case(2)), None)

    def test_struct_ops(self):
        s = functions.capture(Struct(f1=1, f2=2, f3=3))

        def assert_typed(expr, result, dtype):
            self.assertEqual(expr.dtype, dtype)
            r, t = eval_expr_typed(expr)
            self.assertEqual(t, dtype)
            self.assertEqual(result, r)

        assert_typed(s.drop('f3'),
                     Struct(f1=1, f2=2),
                     TStruct(['f1', 'f2'], [TInt32(), TInt32()]))

        assert_typed(s.drop('f1'),
                     Struct(f2=2, f3=3),
                     TStruct(['f2', 'f3'], [TInt32(), TInt32()]))

        assert_typed(s.drop(),
                     Struct(f1=1, f2=2, f3=3),
                     TStruct(['f1', 'f2', 'f3'], [TInt32(),TInt32(),TInt32()]))

        assert_typed(s.select('f1', 'f2'),
                     Struct(f1=1, f2=2),
                     TStruct(['f1', 'f2'], [TInt32(), TInt32()]))

        assert_typed(s.select('f2', 'f1', f4=5, f5=6),
                     Struct(f2=2, f1=1, f4=5, f5=6),
                     TStruct(['f2', 'f1', 'f4', 'f5'], [TInt32(), TInt32(), TInt32(), TInt32()]))

        assert_typed(s.select(),
                     Struct(),
                     TStruct([], []))

        assert_typed(s.annotate(f1=5, f2=10, f4=15),
                     Struct(f1=5, f2=10, f3=3, f4=15),
                     TStruct(['f1', 'f2', 'f3', 'f4'], [TInt32(), TInt32(), TInt32(), TInt32()]))

        assert_typed(s.annotate(f1=5),
                     Struct(f1=5, f2=2, f3=3),
                     TStruct(['f1', 'f2', 'f3'], [TInt32(), TInt32(), TInt32()]))

        assert_typed(s.annotate(),
                     Struct(f1=1, f2=2, f3=3),
                     TStruct(['f1', 'f2', 'f3'], [TInt32(),TInt32(),TInt32()]))

    def test_iter(self):
        a = functions.capture([1, 2, 3])
        self.assertRaises(TypeError, lambda: eval_expr(list(a)))

    def test_str_ops(self):
        s = functions.capture("123")
        self.assertEqual(eval_expr(s.to_int32()), 123)

        s = functions.capture("123123123123")
        self.assertEqual(eval_expr(s.to_int64()), 123123123123)

        s = functions.capture("1.5")
        self.assertEqual(eval_expr(s.to_float32()), 1.5)
        self.assertEqual(eval_expr(s.to_float64()), 1.5)

        s1 = functions.capture('true')
        s2 = functions.capture('True')
        s3 = functions.capture('TRUE')

        s4 = functions.capture('false')
        s5 = functions.capture('False')
        s6 = functions.capture('FALSE')

        self.assertTrue(eval_expr(s1.to_boolean()))
        self.assertTrue(eval_expr(s2.to_boolean()))
        self.assertTrue(eval_expr(s3.to_boolean()))

        self.assertFalse(eval_expr(s4.to_boolean()))
        self.assertFalse(eval_expr(s5.to_boolean()))
        self.assertFalse(eval_expr(s6.to_boolean()))

    def check_expr(self, expr, expected, expected_type):
        self.assertEqual(expected_type, expr.dtype)
        self.assertEqual((expected, expected_type), eval_expr_typed(expr))

    def test_division(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int64_4 = functions.capture(4).to_int64()
        int64_4s = int32_4s.map(lambda x: x.to_int64())
        float32_4 = functions.capture(4).to_float32()
        float32_4s = int32_4s.map(lambda x: x.to_float32())
        float64_4 = functions.capture(4).to_float64()
        float64_4s = int32_4s.map(lambda x: x.to_float64())

        expected = [0.5, 1.0, 2.0, 4.0, None]
        expected_inv = [2.0, 1.0, 0.5, 0.25, None]

        self.check_expr(a_int32 / 4, expected, TArray(TFloat32()))
        self.check_expr(a_int64 / 4, expected, TArray(TFloat32()))
        self.check_expr(a_float32 / 4, expected, TArray(TFloat32()))
        self.check_expr(a_float64 / 4, expected, TArray(TFloat64()))

        self.check_expr(int32_4s / a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(int32_4s / a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(int32_4s / a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int32_4s / a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 / int32_4s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 / int32_4s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 / int32_4s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 / int32_4s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 / int64_4, expected, TArray(TFloat32()))
        self.check_expr(a_int64 / int64_4, expected, TArray(TFloat32()))
        self.check_expr(a_float32 / int64_4, expected, TArray(TFloat32()))
        self.check_expr(a_float64 / int64_4, expected, TArray(TFloat64()))

        self.check_expr(int64_4 / a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_4 / a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_4 / a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_4 / a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 / int64_4s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 / int64_4s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 / int64_4s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 / int64_4s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 / float32_4, expected, TArray(TFloat32()))
        self.check_expr(a_int64 / float32_4, expected, TArray(TFloat32()))
        self.check_expr(a_float32 / float32_4, expected, TArray(TFloat32()))
        self.check_expr(a_float64 / float32_4, expected, TArray(TFloat64()))

        self.check_expr(float32_4 / a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_4 / a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_4 / a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_4 / a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 / float32_4s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 / float32_4s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 / float32_4s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 / float32_4s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 / float64_4, expected, TArray(TFloat64()))
        self.check_expr(a_int64 / float64_4, expected, TArray(TFloat64()))
        self.check_expr(a_float32 / float64_4, expected, TArray(TFloat64()))
        self.check_expr(a_float64 / float64_4, expected, TArray(TFloat64()))

        self.check_expr(float64_4 / a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_4 / a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_4 / a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_4 / a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 / float64_4s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 / float64_4s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 / float64_4s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 / float64_4s, expected, TArray(TFloat64()))

    def test_floor_division(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int32_3s = functions.capture([3, 3, 3, 3, functions.null(TInt32())])
        int64_3 = functions.capture(3).to_int64()
        int64_3s = int32_3s.map(lambda x: x.to_int64())
        float32_3 = functions.capture(3).to_float32()
        float32_3s = int32_3s.map(lambda x: x.to_float32())
        float64_3 = functions.capture(3).to_float64()
        float64_3s = int32_3s.map(lambda x: x.to_float64())

        expected = [0, 1, 2, 5, None]
        expected_inv = [1, 0, 0, 0, None]

        self.check_expr(a_int32 // 3, expected, TArray(TInt32()))
        self.check_expr(a_int64 // 3, expected, TArray(TInt64()))
        self.check_expr(a_float32 // 3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 // 3, expected, TArray(TFloat64()))

        self.check_expr(3 // a_int32, expected_inv, TArray(TInt32()))
        self.check_expr(3 // a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(3 // a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(3 // a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 // int32_3s, expected, TArray(TInt32()))
        self.check_expr(a_int64 // int32_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 // int32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 // int32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 // int64_3, expected, TArray(TInt64()))
        self.check_expr(a_int64 // int64_3, expected, TArray(TInt64()))
        self.check_expr(a_float32 // int64_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 // int64_3, expected, TArray(TFloat64()))

        self.check_expr(int64_3 // a_int32, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 // a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 // a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_3 // a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 // int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_int64 // int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 // int64_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 // int64_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 // float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_int64 // float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float32 // float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 // float32_3, expected, TArray(TFloat64()))

        self.check_expr(float32_3 // a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 // a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 // a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 // a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 // float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 // float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 // float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 // float32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 // float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 // float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 // float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 // float64_3, expected, TArray(TFloat64()))

        self.check_expr(float64_3 // a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 // a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 // a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 // a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 // float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 // float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 // float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 // float64_3s, expected, TArray(TFloat64()))

    def test_addition(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int32_3s = functions.capture([3, 3, 3, 3, functions.null(TInt32())])
        int64_3 = functions.capture(3).to_int64()
        int64_3s = int32_3s.map(lambda x: x.to_int64())
        float32_3 = functions.capture(3).to_float32()
        float32_3s = int32_3s.map(lambda x: x.to_float32())
        float64_3 = functions.capture(3).to_float64()
        float64_3s = int32_3s.map(lambda x: x.to_float64())

        expected = [5, 7, 11, 19, None]
        expected_inv = expected

        self.check_expr(a_int32 + 3, expected, TArray(TInt32()))
        self.check_expr(a_int64 + 3, expected, TArray(TInt64()))
        self.check_expr(a_float32 + 3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 + 3, expected, TArray(TFloat64()))

        self.check_expr(3 + a_int32, expected_inv, TArray(TInt32()))
        self.check_expr(3 + a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(3 + a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(3 + a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 + int32_3s, expected, TArray(TInt32()))
        self.check_expr(a_int64 + int32_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 + int32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 + int32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 + int64_3, expected, TArray(TInt64()))
        self.check_expr(a_int64 + int64_3, expected, TArray(TInt64()))
        self.check_expr(a_float32 + int64_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 + int64_3, expected, TArray(TFloat64()))

        self.check_expr(int64_3 + a_int32, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 + a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 + a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_3 + a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 + int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_int64 + int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 + int64_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 + int64_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 + float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_int64 + float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float32 + float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 + float32_3, expected, TArray(TFloat64()))

        self.check_expr(float32_3 + a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 + a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 + a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 + a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 + float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 + float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 + float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 + float32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 + float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 + float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 + float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 + float64_3, expected, TArray(TFloat64()))

        self.check_expr(float64_3 + a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 + a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 + a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 + a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 + float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 + float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 + float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 + float64_3s, expected, TArray(TFloat64()))

    def test_subtraction(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int32_3s = functions.capture([3, 3, 3, 3, functions.null(TInt32())])
        int64_3 = functions.capture(3).to_int64()
        int64_3s = int32_3s.map(lambda x: x.to_int64())
        float32_3 = functions.capture(3).to_float32()
        float32_3s = int32_3s.map(lambda x: x.to_float32())
        float64_3 = functions.capture(3).to_float64()
        float64_3s = int32_3s.map(lambda x: x.to_float64())

        expected = [-1, 1, 5, 13, None]
        expected_inv = [1, -1, -5, -13, None]

        self.check_expr(a_int32 - 3, expected, TArray(TInt32()))
        self.check_expr(a_int64 - 3, expected, TArray(TInt64()))
        self.check_expr(a_float32 - 3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 - 3, expected, TArray(TFloat64()))

        self.check_expr(3 - a_int32, expected_inv, TArray(TInt32()))
        self.check_expr(3 - a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(3 - a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(3 - a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 - int32_3s, expected, TArray(TInt32()))
        self.check_expr(a_int64 - int32_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 - int32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 - int32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 - int64_3, expected, TArray(TInt64()))
        self.check_expr(a_int64 - int64_3, expected, TArray(TInt64()))
        self.check_expr(a_float32 - int64_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 - int64_3, expected, TArray(TFloat64()))

        self.check_expr(int64_3 - a_int32, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 - a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 - a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_3 - a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 - int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_int64 - int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 - int64_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 - int64_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 - float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_int64 - float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float32 - float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 - float32_3, expected, TArray(TFloat64()))

        self.check_expr(float32_3 - a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 - a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 - a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 - a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 - float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 - float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 - float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 - float32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 - float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 - float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 - float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 - float64_3, expected, TArray(TFloat64()))

        self.check_expr(float64_3 - a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 - a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 - a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 - a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 - float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 - float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 - float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 - float64_3s, expected, TArray(TFloat64()))

    def test_multiplication(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int32_3s = functions.capture([3, 3, 3, 3, functions.null(TInt32())])
        int64_3 = functions.capture(3).to_int64()
        int64_3s = int32_3s.map(lambda x: x.to_int64())
        float32_3 = functions.capture(3).to_float32()
        float32_3s = int32_3s.map(lambda x: x.to_float32())
        float64_3 = functions.capture(3).to_float64()
        float64_3s = int32_3s.map(lambda x: x.to_float64())

        expected = [6, 12, 24, 48, None]
        expected_inv = expected

        self.check_expr(a_int32 * 3, expected, TArray(TInt32()))
        self.check_expr(a_int64 * 3, expected, TArray(TInt64()))
        self.check_expr(a_float32 * 3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 * 3, expected, TArray(TFloat64()))

        self.check_expr(3 * a_int32, expected_inv, TArray(TInt32()))
        self.check_expr(3 * a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(3 * a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(3 * a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 * int32_3s, expected, TArray(TInt32()))
        self.check_expr(a_int64 * int32_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 * int32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 * int32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 * int64_3, expected, TArray(TInt64()))
        self.check_expr(a_int64 * int64_3, expected, TArray(TInt64()))
        self.check_expr(a_float32 * int64_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 * int64_3, expected, TArray(TFloat64()))

        self.check_expr(int64_3 * a_int32, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 * a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 * a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_3 * a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 * int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_int64 * int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 * int64_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 * int64_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 * float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_int64 * float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float32 * float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 * float32_3, expected, TArray(TFloat64()))

        self.check_expr(float32_3 * a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 * a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 * a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 * a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 * float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 * float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 * float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 * float32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 * float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 * float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 * float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 * float64_3, expected, TArray(TFloat64()))

        self.check_expr(float64_3 * a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 * a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 * a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 * a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 * float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 * float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 * float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 * float64_3s, expected, TArray(TFloat64()))

    def test_exponentiation(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int32_3s = functions.capture([3, 3, 3, 3, functions.null(TInt32())])
        int64_3 = functions.capture(3).to_int64()
        int64_3s = int32_3s.map(lambda x: x.to_int64())
        float32_3 = functions.capture(3).to_float32()
        float32_3s = int32_3s.map(lambda x: x.to_float32())
        float64_3 = functions.capture(3).to_float64()
        float64_3s = int32_3s.map(lambda x: x.to_float64())

        expected = [8, 64, 512, 4096, None]
        expected_inv = [9.0, 81.0, 6561.0, 43046721.0, None]

        self.check_expr(a_int32 ** 3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** 3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** 3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** 3, expected, TArray(TFloat64()))

        self.check_expr(3 ** a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(3 ** a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(3 ** a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(3 ** a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 ** int32_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** int32_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** int32_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** int32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 ** int64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** int64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** int64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** int64_3, expected, TArray(TFloat64()))

        self.check_expr(int64_3 ** a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(int64_3 ** a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(int64_3 ** a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(int64_3 ** a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 ** int64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** int64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** int64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** int64_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 ** float32_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** float32_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** float32_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** float32_3, expected, TArray(TFloat64()))

        self.check_expr(float32_3 ** a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float32_3 ** a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float32_3 ** a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float32_3 ** a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 ** float32_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** float32_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** float32_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** float32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 ** float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** float64_3, expected, TArray(TFloat64()))

        self.check_expr(float64_3 ** a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 ** a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 ** a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 ** a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 ** float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 ** float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 ** float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 ** float64_3s, expected, TArray(TFloat64()))

    def test_modulus(self):
        a_int32 = functions.capture([2, 4, 8, 16, functions.null(TInt32())])
        a_int64 = a_int32.map(lambda x: x.to_int64())
        a_float32 = a_int32.map(lambda x: x.to_float32())
        a_float64 = a_int32.map(lambda x: x.to_float64())
        int32_4s = functions.capture([4, 4, 4, 4, functions.null(TInt32())])
        int32_3s = functions.capture([3, 3, 3, 3, functions.null(TInt32())])
        int64_3 = functions.capture(3).to_int64()
        int64_3s = int32_3s.map(lambda x: x.to_int64())
        float32_3 = functions.capture(3).to_float32()
        float32_3s = int32_3s.map(lambda x: x.to_float32())
        float64_3 = functions.capture(3).to_float64()
        float64_3s = int32_3s.map(lambda x: x.to_float64())

        expected = [2, 1, 2, 1, None]
        expected_inv = [1, 3, 3, 3, None]

        self.check_expr(a_int32 % 3, expected, TArray(TInt32()))
        self.check_expr(a_int64 % 3, expected, TArray(TInt64()))
        self.check_expr(a_float32 % 3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 % 3, expected, TArray(TFloat64()))

        self.check_expr(3 % a_int32, expected_inv, TArray(TInt32()))
        self.check_expr(3 % a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(3 % a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(3 % a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 % int32_3s, expected, TArray(TInt32()))
        self.check_expr(a_int64 % int32_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 % int32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 % int32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 % int64_3, expected, TArray(TInt64()))
        self.check_expr(a_int64 % int64_3, expected, TArray(TInt64()))
        self.check_expr(a_float32 % int64_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 % int64_3, expected, TArray(TFloat64()))

        self.check_expr(int64_3 % a_int32, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 % a_int64, expected_inv, TArray(TInt64()))
        self.check_expr(int64_3 % a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(int64_3 % a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 % int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_int64 % int64_3s, expected, TArray(TInt64()))
        self.check_expr(a_float32 % int64_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 % int64_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 % float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_int64 % float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float32 % float32_3, expected, TArray(TFloat32()))
        self.check_expr(a_float64 % float32_3, expected, TArray(TFloat64()))

        self.check_expr(float32_3 % a_int32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 % a_int64, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 % a_float32, expected_inv, TArray(TFloat32()))
        self.check_expr(float32_3 % a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 % float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_int64 % float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float32 % float32_3s, expected, TArray(TFloat32()))
        self.check_expr(a_float64 % float32_3s, expected, TArray(TFloat64()))

        self.check_expr(a_int32 % float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_int64 % float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float32 % float64_3, expected, TArray(TFloat64()))
        self.check_expr(a_float64 % float64_3, expected, TArray(TFloat64()))

        self.check_expr(float64_3 % a_int32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 % a_int64, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 % a_float32, expected_inv, TArray(TFloat64()))
        self.check_expr(float64_3 % a_float64, expected_inv, TArray(TFloat64()))

        self.check_expr(a_int32 % float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_int64 % float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float32 % float64_3s, expected, TArray(TFloat64()))
        self.check_expr(a_float64 % float64_3s, expected, TArray(TFloat64()))

    def test_allele_methods(self):
        self.assertTrue(eval_expr(functions.is_transition("A", "G")))
        self.assertFalse(eval_expr(functions.is_transversion("A", "G")))
        self.assertTrue(eval_expr(functions.is_transversion("A", "T")))
        self.assertFalse(eval_expr(functions.is_transition("A", "T")))
        self.assertTrue(eval_expr(functions.is_snp("A", "T")))
        self.assertTrue(eval_expr(functions.is_snp("A", "G")))
        self.assertTrue(eval_expr(functions.is_snp("C", "G")))
        self.assertTrue(eval_expr(functions.is_snp("CC", "CG")))
        self.assertTrue(eval_expr(functions.is_snp("AT", "AG")))
        self.assertTrue(eval_expr(functions.is_snp("ATCCC", "AGCCC")))
        self.assertTrue(eval_expr(functions.is_mnp("ACTGAC", "ATTGTT")))
        self.assertTrue(eval_expr(functions.is_mnp("CA", "TT")))
        self.assertTrue(eval_expr(functions.is_insertion("A", "ATGC")))
        self.assertTrue(eval_expr(functions.is_insertion("ATT", "ATGCTT")))
        self.assertTrue(eval_expr(functions.is_deletion("ATGC", "A")))
        self.assertTrue(eval_expr(functions.is_deletion("GTGTA", "GTA")))
        self.assertTrue(eval_expr(functions.is_indel("A", "ATGC")))
        self.assertTrue(eval_expr(functions.is_indel("ATT", "ATGCTT")))
        self.assertTrue(eval_expr(functions.is_indel("ATGC", "A")))
        self.assertTrue(eval_expr(functions.is_indel("GTGTA", "GTA")))
        self.assertTrue(eval_expr(functions.is_complex("CTA", "ATTT")))
        self.assertTrue(eval_expr(functions.is_complex("A", "TATGC")))
        self.assertTrue(eval_expr(functions.is_star("ATC", "*")))
        self.assertTrue(eval_expr(functions.is_star("A", "*")))
        self.assertTrue(eval_expr(functions.is_star("*", "ATC")))
        self.assertTrue(eval_expr(functions.is_star("*", "A")))

    def test_hamming(self):
        self.assertEqual(eval_expr(functions.hamming('A', 'T')), 1)
        self.assertEqual(eval_expr(functions.hamming('AAAAA', 'AAAAT')), 1)
        self.assertEqual(eval_expr(functions.hamming('abcde', 'edcba')), 4)

    def test_gp_dosage(self):
        self.assertAlmostEqual(eval_expr(functions.gp_dosage([1.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(eval_expr(functions.gp_dosage([0.0, 1.0, 0.0])), 1.0)
        self.assertAlmostEqual(eval_expr(functions.gp_dosage([0.0, 0.0, 1.0])), 2.0)
        self.assertAlmostEqual(eval_expr(functions.gp_dosage([0.5, 0.5, 0.0])), 0.5)
        self.assertAlmostEqual(eval_expr(functions.gp_dosage([0.0, 0.5, 0.5])), 1.5)