from __future__ import print_function  # Python 2 and 3 print compatibility

import unittest

from hail import HailContext
from hail.expr import *

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
            TInterval(),
            TSet(TInterval()),
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
            TInterval(),
            TSet(TInterval()),
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

    def test_repr(self):
        tv = TVariant()
        tl = TLocus()
        ti = TInterval()
        tc = TCall()
        taa = TAltAllele()

        ti32 = TInt32()
        ti64 = TInt64()
        tf32 = TFloat32()
        tf64 = TFloat64()
        ts = TString()
        tb = TBoolean()

        tdict = TDict(TInterval(), TFloat32())
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