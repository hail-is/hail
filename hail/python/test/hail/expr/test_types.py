import unittest

from hail.expr import coercer_from_dtype
from hail.expr.types import *
from ..helpers import *
from hail.utils.java import Env

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
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

    def test_type_jvm_roundtrip(self):
        ts = self.types_to_test()
        for t in ts:
            rev_str = t._parsable_string()
            jtyp = Env.hail().expr.ir.IRParser.parseType(rev_str)
            self.assertEqual(t, dtype(jtyp.toString()))

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
