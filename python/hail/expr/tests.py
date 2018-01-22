from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext
from hail.expr import *
from hail2 import *

hc = None

def setUpModule():
    global hc
    hc = HailContext()  # master = 'local[2]')

def tearDownModule():
    global hc
    hc.stop()
    hc = None

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
        a = functions.capture([1,2,3])
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



