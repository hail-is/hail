import unittest
from typing import Optional

from hail.expr import coercer_from_dtype
from hail.expr.types import *
from hail.utils.java import Env

from ..helpers import *


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
            tndarray(tstr, 1),
            tndarray(tfloat64, 2),
            tset(tint64),
            tlocus('GRCh37'),
            tlocus('GRCh38'),
            tstruct(),
            tstruct(x=tint32, y=tint64, z=tarray(tset(tstr))),
            tstruct(
                **{
                    'weird field name 1': tint32,
                    r"""this one ' has "" quotes and `` backticks```""": tint64,
                    '!@#$%^&({[': tarray(tset(tstr)),
                }
            ),
            tinterval(tlocus()),
            tset(tinterval(tlocus())),
            tstruct(a=tint32, b=tint32, c=tarray(tstr)),
            tstruct(a=tfloat64, bb=tint32, c=tbool),
            tstruct(a=tint32, b=tint32),
            tstruct(**{'___': tint32, '_ . _': tint32}),
            tunion(),
            tunion(a=tint32, b=tstr),
            tunion(**{'!@#$%^&({[': tstr}),
            ttuple(tstr, tint32),
            ttuple(tarray(tint32), tstr, tstr, tint32, tbool),
            ttuple(),
        ]

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
                if i == j:
                    self.assertEqual(ts[i], ts2[j])
                else:
                    self.assertNotEqual(ts[i], ts2[j])

    @skip_unless_spark_backend()
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

    @skip_when_service_backend(reason='to_spark is nonsensical in the service')
    @fails_local_backend()
    def test_nested_type_to_spark(self):
        ht = hl.utils.range_table(10)
        ht = ht.annotate(nested=hl.dict({"tup": hl.tuple([ht.idx])}))
        ht.to_spark()  # should not throw exception

    def test_rename_not_unique(self):
        with self.assertRaisesRegex(ValueError, "attempted to rename 'b' and 'c' both to 'x'"):
            hl.tstruct(a=hl.tbool, b=hl.tint32, c=hl.tint32)._rename({'b': 'x', 'c': 'x'})
        with self.assertRaisesRegex(ValueError, "attempted to rename 'a' and 'b' both to 'a'"):
            hl.tstruct(a=hl.tbool, b=hl.tint32)._rename({'b': 'a'})

    def test_get_context(self):
        tl1 = tlocus('GRCh37')
        tl2 = tlocus('GRCh38')

        types_and_rgs = [
            (
                [
                    tint32,
                    tint64,
                    tfloat32,
                    tfloat64,
                    tstr,
                    tbool,
                    tcall,
                    tinterval(tset(tint32)),
                    tdict(tstr, tarray(tint32)),
                    tndarray(tstr, 1),
                    tstruct(),
                    tstruct(x=tint32, y=tint64, z=tarray(tset(tstr))),
                    tunion(),
                    tunion(a=tint32, b=tstr),
                    ttuple(tstr, tint32),
                    ttuple(),
                ],
                set(),
            ),
            (
                [
                    tl1,
                    tinterval(tl1),
                    tdict(tstr, tl1),
                    tndarray(tl1, 2),
                    tinterval(tl1),
                    tset(tinterval(tl1)),
                    tstruct(a=tint32, b=tint32, c=tarray(tl1)),
                    tunion(a=tint32, b=tl1),
                    ttuple(tarray(tint32), tl1, tstr, tint32, tbool),
                ],
                {"GRCh37"},
            ),
            (
                [
                    tdict(tl1, tl2),
                    ttuple(tarray(tl2), tl1, tstr, tint32, tbool),
                ],
                {"GRCh37", "GRCh38"},
            ),
        ]

        for types, rgs in types_and_rgs:
            for t in types:
                self.assertEqual(t.get_context().references, rgs)

    def test_tlocus_schema_from_rg_matches_scala(self):
        def locus_from_import_vcf(rg: Optional[str]) -> HailType:
            return hl.import_vcf(resource('sample2.vcf'), reference_genome=rg).locus.dtype

        assert tlocus._schema_from_rg(None) == locus_from_import_vcf(None)
        assert tlocus._schema_from_rg('GRCh37') == locus_from_import_vcf('GRCh37')
        assert tlocus._schema_from_rg('GRCh38') == locus_from_import_vcf('GRCh38')
        assert tlocus._schema_from_rg('GRCm38') == locus_from_import_vcf('GRCm38')
        assert tlocus._schema_from_rg('CanFam3') == locus_from_import_vcf('CanFam3')
        assert tlocus._schema_from_rg('default') == locus_from_import_vcf('default')
